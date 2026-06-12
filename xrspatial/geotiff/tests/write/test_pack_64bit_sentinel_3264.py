"""``to_geotiff(pack=True)`` with 64-bit sentinels above 2**53 (#3264).

``_pack`` used to fill NaN holes on the float64 buffer
(``out[isnan] = float(fill)``) and cast to the recorded integer dtype
afterwards. float64 cannot represent integers above 2**53 exactly, so
``float(INT64_MAX)`` rounded to 2**63 and the cast overflowed: holes
came back as INT64_MIN (0 for UINT64_MAX) while GDAL_NODATA still
declared the original sentinel, and a ``masked=True`` re-read returned
them as valid pixels. The fill now runs at the target dtype's native
width (``_pack_restore_int``): park the holes at zero for the round +
cast, then assign the sentinel as a Python int.

The ``nodata=`` kwarg had the mirror-image bug: its range check
round-tripped through ``float()``, so ``nodata=INT64_MAX`` was rejected
as out of range even though it fits int64 exactly. The check now
compares as integers.
"""
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._attrs import _pack_restore_int

from .._helpers.markers import requires_gpu

I64_MAX = np.iinfo(np.int64).max
U64_MAX = np.iinfo(np.uint64).max

BACKENDS = [
    pytest.param(None, False, id="numpy"),
    pytest.param(2, False, id="dask"),
    pytest.param(None, True, marks=requires_gpu, id="gpu"),
    pytest.param(2, True, marks=requires_gpu, id="dask-gpu"),
]


def _write_tiff(path, data, *, nodata):
    h, w = data.shape
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.arange(h, 0, -1) - 0.5, "x": np.arange(w) + 0.5},
        attrs={"crs": 4326, "nodata": nodata},
    )
    to_geotiff(da, path)
    return path


def _reopen(path, chunks, gpu=False):
    kwargs = {"unpack": True}
    if gpu:
        kwargs["gpu"] = True
    if chunks is not None:
        kwargs["chunks"] = chunks
    return open_geotiff(path, **kwargs)


def _pack_write(decoded, out, **kwargs):
    """Run the pack write with float->int cast warnings escalated: the
    broken fill's only signal was numpy's "invalid value encountered in
    cast" RuntimeWarning, so a regression must not pass silently."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        decoded.xrs.to_geotiff(out, pack=True, **kwargs)


# ---------------------------------------------------------------------------
# Round trip: INT64_MAX / UINT64_MAX sentinels survive packing exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks,gpu", BACKENDS)
@pytest.mark.parametrize("dtype,sentinel", [
    (np.int64, I64_MAX),
    (np.uint64, U64_MAX),
], ids=["int64", "uint64"])
def test_pack_64bit_sentinel_round_trip(tmp_path, dtype, sentinel,
                                        chunks, gpu):
    """unpack read masks the sentinel to NaN; pack write must restore the
    exact 64-bit sentinel, so the holes match the GDAL_NODATA tag and a
    masked re-read masks them again."""
    data = np.arange(16, dtype=dtype).reshape(4, 4)
    data[0, 0] = sentinel
    name = f"src_{np.dtype(dtype).name}_3264.tif"
    src = _write_tiff(tmp_path / name, data, nodata=int(sentinel))

    decoded = _reopen(src, chunks, gpu=gpu)
    out = str(tmp_path / f"out_{np.dtype(dtype).name}_3264.tif")
    _pack_write(decoded, out)

    raw = open_geotiff(out)
    assert str(raw.dtype) == np.dtype(dtype).name
    raw_vals = np.asarray(raw.data)
    assert int(raw_vals[0, 0]) == int(sentinel)
    np.testing.assert_array_equal(raw_vals.ravel()[1:], data.ravel()[1:])

    masked = open_geotiff(out, masked=True)
    masked_vals = np.asarray(masked.data)
    assert np.isnan(masked_vals[0, 0])
    assert np.isnan(masked_vals).sum() == 1


# ---------------------------------------------------------------------------
# nodata= kwarg: INT64_MAX is representable and must be accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_nodata_kwarg_accepts_int64_max(tmp_path, chunks):
    data = np.arange(16, dtype=np.int64).reshape(4, 4)
    data[0, 0] = np.iinfo(np.int64).min + 1
    src = _write_tiff(tmp_path / "src_kw_3264.tif", data,
                      nodata=int(data[0, 0]))

    decoded = _reopen(src, chunks)
    out = str(tmp_path / "out_kw_3264.tif")
    _pack_write(decoded, out, nodata=I64_MAX)

    raw = open_geotiff(out)
    assert int(np.asarray(raw.data)[0, 0]) == I64_MAX
    assert raw.attrs.get("nodata") == I64_MAX
    masked = open_geotiff(out, masked=True)
    assert np.isnan(np.asarray(masked.data)[0, 0])


def test_pack_nodata_kwarg_still_rejects_unrepresentable(tmp_path):
    """The integer-exact check keeps rejecting what the float check did:
    fractional, out-of-range, and non-finite overrides."""
    data = np.arange(16, dtype=np.int16).reshape(4, 4)
    data[0, 0] = -9999
    src = _write_tiff(tmp_path / "src_rej_3264.tif", data, nodata=-9999)

    decoded = open_geotiff(src, unpack=True)
    out = str(tmp_path / "out_rej_3264.tif")
    for bad in (3.5, U64_MAX, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cannot be represented"):
            decoded.xrs.to_geotiff(out, pack=True, nodata=bad)


# ---------------------------------------------------------------------------
# Pre-v5 fallback: no mask_and_scale_dtype, target derived from the
# integer-typed attrs sentinel -- routes through the same native-width fill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dask_backed", [False, True], ids=["numpy", "dask"])
def test_pack_pre_v5_fallback_64bit_sentinel(dask_backed):
    """Arrays read by a pre-contract-v5 release carry no
    ``mask_and_scale_dtype``; ``_pack`` derives the target dtype from the
    integer-typed attrs sentinel. With a >2**53 sentinel that path must
    also fill at native width."""
    from xrspatial.geotiff._attrs import _pack

    arr = xr.DataArray(
        np.array([[1.0, np.nan], [3.0, 4.0]]),
        dims=("y", "x"),
        attrs={"nodata": np.int64(I64_MAX), "masked_nodata": True},
    )
    if dask_backed:
        import dask.array as da
        arr = arr.copy(data=da.from_array(arr.values, chunks=(1, 2)))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = _pack(arr)
        vals = np.asarray(out.data)

    assert out.dtype == np.int64
    assert int(vals[0, 1]) == I64_MAX
    np.testing.assert_array_equal(vals[~np.isnan(arr.values)],
                                  [1, 3, 4])


# ---------------------------------------------------------------------------
# ``_pack_restore_int``: unit-pinned on numpy and cupy chunks
# ---------------------------------------------------------------------------


def test_pack_restore_int_native_width_numpy():
    chunk = np.array([[1.0, np.nan], [np.nan, 4.6]])
    out = _pack_restore_int(chunk, I64_MAX, "int64")
    assert out.dtype == np.int64
    np.testing.assert_array_equal(
        out, [[1, I64_MAX], [I64_MAX, 5]])  # 4.6 rounds, holes are exact
    assert np.isnan(chunk[0, 1])  # input untouched


def test_pack_restore_int_uint64_numpy():
    chunk = np.array([[np.nan, 2.0]])
    out = _pack_restore_int(chunk, U64_MAX, "uint64")
    assert out.dtype == np.uint64
    assert int(out[0, 0]) == U64_MAX
    assert out[0, 1] == 2


@requires_gpu
def test_pack_restore_int_handles_cupy_chunks():
    import cupy

    chunk = cupy.asarray(np.array([[1.0, np.nan]]))
    out = _pack_restore_int(chunk, I64_MAX, "int64")
    assert isinstance(out, cupy.ndarray)
    host = out.get()
    assert host.dtype == np.int64
    np.testing.assert_array_equal(host, [[1, I64_MAX]])
