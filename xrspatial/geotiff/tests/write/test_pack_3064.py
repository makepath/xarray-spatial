"""``to_geotiff(pack=True)`` -- the inverse of ``unpack=True`` (#3064).

A ``unpack=True`` read promotes an integer raster to float64,
applies the GDAL SCALE/OFFSET, and masks the nodata sentinel to NaN.
``pack=True`` reverses that on write: it un-scales, fills NaN back to the
sentinel, and restores the integer source dtype recorded on
``attrs['mask_and_scale_dtype']``. The SCALE/OFFSET tags are kept so the
re-packed file unpacks to the same values on the next ``mask_and_scale``
read rather than double-scaling.

``unpack`` reads run on numpy, dask, gpu, and dask+gpu since #3075 (VRT
still rejects it). The round-trip passes on all four backends; the gpu
and dask+gpu legs crashed in ``_pack``'s ``fillna`` until #3112 replaced
it with a buffer-level fill.
"""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._attrs import _pack, _pack_fill_nan

from .._helpers.markers import requires_gpu


def _write_int_tiff(path, data, *, nodata=None, scale=None, offset=None):
    """Write ``data`` (an integer ndarray) as a GeoTIFF, optionally carrying
    a nodata sentinel and GDAL SCALE/OFFSET packing metadata."""
    h, w = data.shape
    attrs = {"crs": 4326}
    if nodata is not None:
        attrs["nodata"] = nodata
    if scale is not None or offset is not None:
        attrs["gdal_metadata"] = {
            "SCALE": str(scale if scale is not None else 1.0),
            "OFFSET": str(offset if offset is not None else 0.0),
        }
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.arange(h, 0, -1) - 0.5, "x": np.arange(w) + 0.5},
        attrs=attrs,
    )
    # A gdal_metadata SCALE/OFFSET dict on a fresh array is an
    # experimental rich-tag write (#3320).
    to_geotiff(da, path, allow_experimental_codecs="gdal_metadata" in attrs)
    return path


def _reopen(path, chunks, gpu=False):
    kwargs = {"unpack": True}
    if gpu:
        kwargs["gpu"] = True
    if chunks is not None:
        kwargs["chunks"] = chunks
    return open_geotiff(path, **kwargs)


def _to_host(data):
    """Materialise a possibly dask- and/or cupy-backed buffer as numpy."""
    if hasattr(data, "compute"):
        data = data.compute()
    if hasattr(data, "get"):
        data = data.get()
    return np.asarray(data)


# ---------------------------------------------------------------------------
# Round trip: no scale/offset (the int8 + nodata case from the issue)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_restores_int8_no_scale(tmp_path, chunks):
    """int8 + nodata, read with mask_and_scale, packs back to int8."""
    data = np.array([[1, 2, -128], [4, 5, 6]], dtype=np.int8)
    src = _write_int_tiff(tmp_path / "src_int8_3064.tif", data, nodata=-128)

    decoded = _reopen(src, chunks)
    assert decoded.dtype == np.float64
    assert decoded.attrs.get("mask_and_scale_dtype") == "int8"

    out = str(tmp_path / "out_int8_3064.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    back = open_geotiff(out)
    assert str(back.dtype) == "int8"
    np.testing.assert_array_equal(back.data, data)


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_restores_uint16_no_scale(tmp_path, chunks):
    data = np.array([[10, 20, 65535], [40, 50, 60]], dtype=np.uint16)
    src = _write_int_tiff(tmp_path / "src_u16_3064.tif", data, nodata=65535)

    decoded = _reopen(src, chunks)
    out = str(tmp_path / "out_u16_3064.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    back = open_geotiff(out)
    assert str(back.dtype) == "uint16"
    np.testing.assert_array_equal(back.data, data)


# ---------------------------------------------------------------------------
# Float source: the masked sentinel must be restored, not left as NaN (#3078)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks,gpu", [
    pytest.param(None, False, id="numpy"),
    pytest.param(2, False, id="dask"),
    # The gpu legs pin the float branch of the #3112 fill fix: a float
    # sentinel restored on a cupy-backed buffer.
    pytest.param(None, True, marks=requires_gpu, id="gpu"),
    pytest.param(2, True, marks=requires_gpu, id="dask-gpu"),
])
def test_pack_restores_float_nodata_sentinel(tmp_path, chunks, gpu):
    """A float raster masks its sentinel to NaN on an ``unpack=True`` read.
    ``pack=True`` must put the declared sentinel back so the pixels on disk
    match the GDAL_NODATA tag -- otherwise the file declares nodata=-9999 but
    stores NaN, which a non-masking reader silently treats as a valid value.
    """
    data = np.array([[1.5, 2.5, -9999.0], [4.5, 5.5, 6.5]], dtype=np.float32)
    src = _write_int_tiff(tmp_path / "src_f32_3078.tif", data, nodata=-9999.0)

    decoded = _reopen(src, chunks, gpu=gpu)
    assert np.isnan(_to_host(decoded.data)).any()  # sentinel was masked

    out = str(tmp_path / "out_f32_3078.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    # Plain (non-masking) read: the sentinel is back and no NaN survives.
    back = np.asarray(open_geotiff(out).data)
    assert not np.isnan(back).any()
    assert back[0, 2] == -9999.0
    assert open_geotiff(out).attrs.get("nodata") == -9999.0

    # An unpack read still round-trips the masked value to NaN.
    repacked = open_geotiff(out, unpack=True)
    np.testing.assert_array_equal(
        np.asarray(repacked.data), _to_host(decoded.data))


# ---------------------------------------------------------------------------
# Round trip: with SCALE/OFFSET -- the file must unpack, not double-scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 2], ids=["numpy", "dask"])
def test_pack_with_scale_offset_round_trip(tmp_path, chunks):
    data = np.array([[1, 2, 3], [4, 5, 255]], dtype=np.uint8)
    src = _write_int_tiff(
        tmp_path / "src_so_3064.tif", data, nodata=255, scale=2.0, offset=10.0)

    decoded = _reopen(src, chunks)
    # unpacked values: data * 2 + 10, sentinel -> NaN
    assert decoded.attrs.get("mask_and_scale_dtype") == "uint8"

    out = str(tmp_path / "out_so_3064.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    # Raw read: the stored integers round-trip exactly.
    raw = open_geotiff(out)
    assert str(raw.dtype) == "uint8"
    np.testing.assert_array_equal(raw.data, data)

    # The SCALE/OFFSET tags are kept, so a mask_and_scale read of the packed
    # file reproduces the original decoded values rather than scaling twice.
    eager_decoded = open_geotiff(src, unpack=True)
    repacked_decoded = open_geotiff(out, unpack=True)
    np.testing.assert_allclose(
        repacked_decoded.data, eager_decoded.data, equal_nan=True)


# ---------------------------------------------------------------------------
# GPU-backed input: ``unpack=True`` works with ``gpu=True`` since #3075, so
# the ``pack=True`` inverse must round-trip there too. Both legs crashed in
# ``_pack``'s ``fillna`` (xarray's where() breaks on cupy) until #3112
# switched the sentinel restore to a buffer-level fill.
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("chunks", [None, 2], ids=["gpu", "dask-gpu"])
def test_pack_round_trip_gpu(tmp_path, chunks):
    """A ``gpu=True`` unpack read packs back to the integer source dtype."""
    data = np.array([[1, 2, 3], [4, 5, 255]], dtype=np.uint8)
    src = _write_int_tiff(
        tmp_path / "src_gpu_3114.tif", data,
        nodata=255, scale=2.0, offset=10.0)

    decoded = (open_geotiff(src, unpack=True, gpu=True) if chunks is None
               else open_geotiff(src, unpack=True, gpu=True, chunks=chunks))
    assert decoded.attrs.get("mask_and_scale_dtype") == "uint8"

    out = str(tmp_path / f"out_gpu_3114_{chunks}.tif")
    decoded.xrs.to_geotiff(out, pack=True)

    # Raw read: the stored integers round-trip exactly.
    raw = open_geotiff(out)
    assert str(raw.dtype) == "uint8"
    np.testing.assert_array_equal(np.asarray(raw.data), data)

    # Parity with the CPU eager unpack of the same source.
    cpu_decoded = open_geotiff(src, unpack=True)
    repacked_decoded = open_geotiff(out, unpack=True)
    np.testing.assert_allclose(
        np.asarray(repacked_decoded.data), np.asarray(cpu_decoded.data),
        equal_nan=True)


# ---------------------------------------------------------------------------
# ``_pack_fill_nan``: the buffer-level NaN -> sentinel fill that replaced
# ``fillna`` (#3112). Unit-pinned on numpy and cupy so a regression back to
# an xarray ``where``-based fill shows up without a full round trip.
# ---------------------------------------------------------------------------


def test_pack_fill_nan_fills_sentinel_numpy():
    chunk = np.array([[1.0, np.nan], [np.nan, 4.0]])
    out = _pack_fill_nan(chunk, np.uint8(255))
    np.testing.assert_array_equal(out, [[1.0, 255.0], [255.0, 4.0]])
    # The input buffer is left untouched (fillna semantics).
    assert np.isnan(chunk[0, 1])


def test_pack_fill_nan_skips_integer_chunk():
    chunk = np.array([[1, 2]], dtype=np.int32)
    assert _pack_fill_nan(chunk, 255) is chunk


@requires_gpu
def test_pack_fill_nan_handles_cupy_chunks():
    import cupy

    chunk = cupy.asarray(np.array([[1.0, np.nan]]))
    out = _pack_fill_nan(chunk, 255)
    assert isinstance(out, cupy.ndarray)
    np.testing.assert_array_equal(out.get(), [[1.0, 255.0]])


def test_pack_sentinel_fill_stays_lazy():
    """The NaN -> sentinel fill must not compute dask input at ``_pack`` time.

    Pins the lazy shape of the #3112 fill the same way
    ``test_pack_lazy_nan_guard_3235.py`` pins the no-sentinel guard: a
    counting identity block threaded through the graph stays at zero
    until something computes.
    """
    import dask.array as dask_array

    counts = {"n": 0}

    def _count(block):
        counts["n"] += 1
        return block

    values = np.array([[1.0, np.nan], [3.0, 4.0]])
    src = dask_array.from_array(values, chunks=2)
    # meta= keeps dask from probing _count with an empty block at graph
    # build time, which would increment the counter without a compute.
    arr = src.map_blocks(_count, dtype=values.dtype, meta=src._meta)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        attrs={"crs": 4326, "nodata": 255, "masked_nodata": True,
               "scale_factor": 1.0, "add_offset": 0.0,
               "mask_and_scale_dtype": "uint8"},
    )

    packed = _pack(da)
    assert hasattr(packed.data, "dask")
    assert counts["n"] == 0  # nothing computed at _pack time

    np.testing.assert_array_equal(
        np.asarray(packed.compute().data), [[1, 255], [3, 4]])


# ---------------------------------------------------------------------------
# Read-side attr: recorded for mask_and_scale, not for a plain masked read
# ---------------------------------------------------------------------------


def test_mask_and_scale_dtype_recorded_eager_dask_match(tmp_path):
    data = np.array([[1, 2, 255], [4, 5, 6]], dtype=np.uint8)
    src = _write_int_tiff(tmp_path / "src_attr_3064.tif", data, nodata=255)

    eager = open_geotiff(src, unpack=True)
    lazy = open_geotiff(src, unpack=True, chunks=2)
    assert eager.attrs.get("mask_and_scale_dtype") == "uint8"
    assert (eager.attrs.get("mask_and_scale_dtype")
            == lazy.attrs.get("mask_and_scale_dtype"))


def test_mask_and_scale_dtype_absent_on_plain_masked_read(tmp_path):
    """A plain ``masked=True`` read (not mask_and_scale) does not stamp the
    attr; ``pack`` is the inverse of ``mask_and_scale`` specifically."""
    data = np.array([[1, 2, 255], [4, 5, 6]], dtype=np.uint8)
    src = _write_int_tiff(tmp_path / "src_masked_3064.tif", data, nodata=255)

    masked = open_geotiff(src, masked=True)
    assert masked.dtype == np.float64  # masking still promotes
    assert "mask_and_scale_dtype" not in masked.attrs


def test_contract_version_is_5(tmp_path):
    src = _write_int_tiff(
        tmp_path / "src_ver_3064.tif",
        np.array([[1, 2], [3, 4]], dtype=np.uint8))
    da = open_geotiff(src)
    assert da.attrs["_xrspatial_geotiff_contract"] == 5


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_pack_rejects_bare_array(tmp_path):
    arr = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="pack=True requires a DataArray"):
        to_geotiff(arr, str(tmp_path / "x_3064.tif"), pack=True)


def test_pack_rejects_array_without_mask_and_scale_state(tmp_path):
    da = xr.DataArray(
        np.ones((2, 2), dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        attrs={"crs": 4326},
    )
    # The message names the current ``unpack`` kwarg, not the deprecated
    # ``mask_and_scale`` alias (#3086); ``mask_and_scale_dtype`` is the only
    # old-name token allowed (it is the contract-v5 attrs key).
    with pytest.raises(ValueError, match="no unpack state") as excinfo:
        da.xrs.to_geotiff(str(tmp_path / "y_3064.tif"), pack=True)
    msg = str(excinfo.value)
    assert "open_geotiff(unpack=True)" in msg
    assert "mask_and_scale=True" not in msg
