"""``to_geotiff(pack=True)`` refuses packed values the dtype cannot hold (#3260).

``_pack`` reverses the unpack transform and casts to the integer dtype
recorded on ``attrs['mask_and_scale_dtype']``. Before #3260 the cast ran
unguarded: a finite value whose packed form fell outside the dtype range
wrapped (40000 -> -25536 in int16) and +/-Inf cast to a platform-defined
integer, both silently. The guard raises ``ValueError`` instead -- at
call time for numpy / cupy buffers, from the write's single compute for
dask backings (same timing as the #3235 NaN guard).
"""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._attrs import _pack_guard_int_range

from .._helpers.markers import requires_gpu


def _write_scaled_int16(path, *, nodata=None):
    """int16 source with SCALE=0.1 so unpacked values are data * 0.1."""
    data = np.array([[100, 200], [300, 32000]], dtype=np.int16)
    attrs = {
        "crs": 4326,
        "gdal_metadata": {"SCALE": "0.1", "OFFSET": "0.0"},
    }
    if nodata is not None:
        attrs["nodata"] = nodata
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        attrs=attrs,
    )
    to_geotiff(da, str(path), nodata=nodata)
    return str(path)


def _unpacked(path, *, gpu=False):
    kwargs = {"unpack": True}
    if gpu:
        kwargs["gpu"] = True
    return open_geotiff(path, **kwargs).copy()


# ---------------------------------------------------------------------------
# Finite overflow wraps in the cast: refused on every backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks,gpu", [
    pytest.param(None, False, id="numpy"),
    pytest.param(1, False, id="dask"),
    pytest.param(None, True, marks=requires_gpu, id="gpu"),
    pytest.param(1, True, marks=requires_gpu, id="dask-gpu"),
])
def test_pack_rejects_finite_overflow(tmp_path, chunks, gpu):
    src = _write_scaled_int16(tmp_path / "src_overflow_3260.tif")
    mod = _unpacked(src, gpu=gpu)
    # Packs to 40000 > int16 max 32767; pre-#3260 this wrapped to -25536.
    mod.data[0, 0] = 4000.0
    if chunks is not None:
        mod = mod.chunk({"y": chunks})
    out = str(tmp_path / "out_overflow_3260.tif")
    with pytest.raises(ValueError, match="cannot represent"):
        to_geotiff(mod, out, pack=True)


@pytest.mark.parametrize("chunks", [None, 1], ids=["numpy", "dask"])
def test_pack_rejects_underflow_unsigned(tmp_path, chunks):
    """A negative packed value must not wrap into the top of a uint range."""
    data = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        attrs={"crs": 4326,
               "gdal_metadata": {"SCALE": "0.5", "OFFSET": "0.0"}},
    )
    src = str(tmp_path / "src_u16_underflow_3260.tif")
    to_geotiff(da, src)

    mod = _unpacked(src)
    mod.data[0, 0] = -1.0  # packs to -2, below uint16 min
    if chunks is not None:
        mod = mod.chunk({"y": chunks})
    out = str(tmp_path / "out_u16_underflow_3260.tif")
    with pytest.raises(ValueError, match="cannot represent"):
        to_geotiff(mod, out, pack=True)


# ---------------------------------------------------------------------------
# +/-Inf passes the NaN fill / NaN guard but must not reach the cast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 1], ids=["numpy", "dask"])
@pytest.mark.parametrize("value", [np.inf, -np.inf], ids=["inf", "neg-inf"])
def test_pack_rejects_inf(tmp_path, chunks, value):
    # A declared sentinel routes Inf past the NaN fill (isnan only), so
    # this leg pins the path where Inf used to reach the cast directly.
    src = _write_scaled_int16(tmp_path / "src_inf_3260.tif", nodata=-32768)
    mod = _unpacked(src)
    mod.data[0, 0] = value
    if chunks is not None:
        mod = mod.chunk({"y": chunks})
    out = str(tmp_path / "out_inf_3260.tif")
    with pytest.raises(ValueError, match="not finite|cannot represent"):
        to_geotiff(mod, out, pack=True)


# ---------------------------------------------------------------------------
# No false positives: boundary and round-back-into-range values pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunks", [None, 1], ids=["numpy", "dask"])
def test_pack_accepts_full_dtype_range(tmp_path, chunks):
    """Exact iinfo.min / iinfo.max packed values are representable."""
    data = np.array([[-32768, 0], [100, 32767]], dtype=np.int16)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        attrs={"crs": 4326,
               "gdal_metadata": {"SCALE": "0.1", "OFFSET": "0.0"}},
    )
    src = str(tmp_path / "src_bounds_3260.tif")
    to_geotiff(da, src)

    mod = _unpacked(src)
    if chunks is not None:
        mod = mod.chunk({"y": chunks})
    out = str(tmp_path / "out_bounds_3260.tif")
    to_geotiff(mod, out, pack=True)

    back = open_geotiff(out)
    assert str(back.dtype) == "int16"
    np.testing.assert_array_equal(np.asarray(back.data), data)


def test_pack_accepts_value_that_rounds_back_into_range(tmp_path):
    """The guard runs after the round: 3276.74 packs to 32767.4 which
    rounds to 32767 and fits."""
    src = _write_scaled_int16(tmp_path / "src_round_3260.tif")
    mod = _unpacked(src)
    mod.data[0, 0] = 3276.74
    out = str(tmp_path / "out_round_3260.tif")
    to_geotiff(mod, out, pack=True)
    back = open_geotiff(out)
    assert np.asarray(back.data)[0, 0] == 32767


def test_pack_float_target_not_range_guarded(tmp_path):
    """Float packed dtypes have no wrap problem; large values pass."""
    data = np.array([[1.0, 2.0], [3.0, -9999.0]], dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        attrs={"crs": 4326, "nodata": -9999.0,
               "gdal_metadata": {"SCALE": "2.0", "OFFSET": "0.0"}},
    )
    src = str(tmp_path / "src_float_3260.tif")
    to_geotiff(da, src, nodata=-9999.0)

    mod = _unpacked(src)
    mod.data[0, 0] = 1e30
    out = str(tmp_path / "out_float_3260.tif")
    to_geotiff(mod, out, pack=True)
    back = open_geotiff(out)
    assert str(back.dtype) == "float32"
    # Packed value is 1e30 / SCALE = 5e29, well inside float32 range.
    assert np.asarray(back.data)[0, 0] == np.float32(5e29)


# ---------------------------------------------------------------------------
# Unit: the 64-bit exclusive upper bound is exact
# ---------------------------------------------------------------------------


def test_guard_rejects_two_pow_63_for_int64():
    """float64 cannot hold INT64_MAX; float(iinfo.max) rounds up to 2**63.
    The guard's exclusive bound must reject exactly-2**63, which an
    inclusive ``> float(iinfo.max)`` test would let through to wrap."""
    info = np.iinfo(np.int64)
    chunk = np.array([float(2 ** 63)])
    with pytest.raises(ValueError, match="cannot represent"):
        _pack_guard_int_range(
            chunk, "int64", float(info.min), float(int(info.max) + 1))
    # One ULP below 2**63 is representable in int64 and passes.
    ok = np.array([np.nextafter(float(2 ** 63), 0.0)])
    _pack_guard_int_range(
        ok, "int64", float(info.min), float(int(info.max) + 1))
