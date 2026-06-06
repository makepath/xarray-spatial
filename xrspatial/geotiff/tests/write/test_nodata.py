"""Write-side nodata semantics and validation.

Covers the write / validation nodata paths:

* refuse non-numeric / bool ``nodata=`` and ``attrs['_FillValue']`` at
  every writer entry point.
* reject ``bool`` / ``np.bool_`` sentinels at ``to_geotiff`` and
  ``build_geo_tags``.
* parse 64-bit integer sentinels at full precision (int-first,
  float-fallback) across the eager / dask / VRT / GPU write+read
  round-trips.
* an out-of-range integer sentinel (e.g. uint16 + ``-9999``) is a
  value-match no-op rather than an ``OverflowError``.
* ``mask_nodata=False`` keeps the source integer dtype so the
  ``dtype=`` cast contract is reachable.

GPU-only nodata cases live with the GPU tests; the GPU parity tests
here are gated through the shared ``requires_gpu`` marker.
"""
from __future__ import annotations

import io
import os
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _build_vrt, _read_geotiff_dask, _read_vrt, open_geotiff, to_geotiff
from xrspatial.geotiff._attrs import _resolve_nodata_attr
from xrspatial.geotiff._geotags import GeoTransform, _parse_nodata_str, build_geo_tags
from xrspatial.geotiff._reader import _int_nodata_in_range, _resolve_masked_fill
from xrspatial.geotiff._validation import _validate_nodata_arg
from xrspatial.geotiff._writer import write
from xrspatial.geotiff.tests._helpers.markers import requires_gpu

# ===========================================================================
# Non-numeric / bool nodata rejection
# ===========================================================================


def _nan_square():
    return xr.DataArray(
        np.full((4, 4), np.nan, dtype=np.float32),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )


@pytest.fixture
def uint8_da():
    """Small uint8 DataArray for nodata round-trip tests."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    return xr.DataArray(arr, dims=['y', 'x'])


@pytest.mark.parametrize("bad", ['missing', object(), [1, 2]])
def test_validate_nodata_arg_rejects_non_numeric(bad):
    with pytest.raises(ValueError, match="nodata must be numeric"):
        _validate_nodata_arg(bad)


@pytest.mark.parametrize(
    "ok", [None, 0, -9999, 1.5, np.float32(-1), np.int64(0)],
)
def test_validate_nodata_arg_accepts_numeric_and_none(ok):
    _validate_nodata_arg(ok)


def test_resolve_nodata_attr_rejects_non_numeric_fillvalue():
    with pytest.raises(ValueError, match="_FillValue"):
        _resolve_nodata_attr({'_FillValue': 'missing'})


def test_resolve_nodata_attr_rejects_non_numeric_nodata_attr():
    with pytest.raises(ValueError, match=r"attrs\['nodata'\]"):
        _resolve_nodata_attr({'nodata': 'missing'})


def test_resolve_nodata_attr_skips_non_numeric_in_nodatavals():
    # nodatavals (rioxarray's per-band tuple) keeps its skip-on-non-numeric
    # behaviour: those values often come from arbitrary upstream pipelines
    # and a single bad entry should not block writing.
    assert _resolve_nodata_attr({'nodatavals': ('NaN ', -9999.0)}) == -9999.0


def test_resolve_nodata_attr_still_accepts_numeric_fillvalue():
    assert _resolve_nodata_attr({'_FillValue': -9999}) == -9999


def test_resolve_nodata_attr_returns_none_for_nan_fillvalue():
    assert _resolve_nodata_attr({'_FillValue': float('nan')}) is None


def test_to_geotiff_rejects_non_numeric_nodata_kwarg():
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="nodata must be numeric"):
        to_geotiff(_nan_square(), buf, nodata='missing')


def test_to_geotiff_rejects_non_numeric_fillvalue_attr():
    da = _nan_square()
    da.attrs['_FillValue'] = 'missing'
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="_FillValue"):
        to_geotiff(da, buf)


def test_to_geotiff_vrt_path_rejects_non_numeric_nodata(tmp_path):
    vrt_path = str(tmp_path / "tmp_1973_vrt.vrt")
    with pytest.raises(ValueError, match="nodata must be numeric"):
        to_geotiff(_nan_square(), vrt_path, nodata='missing')


def test_to_geotiff_accepts_numeric_nodata_kwarg():
    buf = io.BytesIO()
    to_geotiff(_nan_square(), buf, nodata=-9999)
    assert buf.getbuffer().nbytes > 0


# --- Bool rejection across the three writer entry points -------------------
# The eager path already rejected bools but the GPU/VRT validators
# previously routed bool through ``float(True) == 1.0`` and silently coerced.
# The shared validator now refuses bools so all three paths behave the same.


@pytest.mark.parametrize("bad", [True, False])
def test_validate_nodata_arg_rejects_bool(bad):
    with pytest.raises(TypeError, match="nodata must be numeric"):
        _validate_nodata_arg(bad)


def test_validate_nodata_arg_rejects_numpy_bool():
    with pytest.raises(TypeError, match="nodata must be numeric"):
        _validate_nodata_arg(np.bool_(True))


def test_to_geotiff_eager_rejects_bool_nodata():
    buf = io.BytesIO()
    with pytest.raises(TypeError, match="nodata must be numeric"):
        to_geotiff(_nan_square(), buf, nodata=True)


def test_to_geotiff_vrt_rejects_bool_nodata(tmp_path):
    vrt_path = str(tmp_path / "tmp_1973_bool_vrt.vrt")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        to_geotiff(_nan_square(), vrt_path, nodata=True)


@requires_gpu
def test_write_geotiff_gpu_rejects_bool_nodata(tmp_path):
    import cupy

    from xrspatial.geotiff import _write_geotiff_gpu

    da_cpu = _nan_square()
    da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
    out = str(tmp_path / "tmp_1973_bool_gpu.tif")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        _write_geotiff_gpu(da_gpu, out, nodata=True)


# --- All-non-numeric ``attrs['nodatavals']`` warn-and-fall-through ----------
# A tuple where every entry is non-numeric is almost certainly a user error
# rather than a legitimate "no sentinel" signal.


def test_resolve_nodata_attr_warns_when_nodatavals_all_non_numeric():
    with pytest.warns(UserWarning, match="nodatavals"):
        result = _resolve_nodata_attr({'nodatavals': ('foo', 'bar')})
    assert result is None


def test_resolve_nodata_attr_no_warning_when_nodatavals_has_usable_entry():
    # First entry is non-numeric, second is a real sentinel. The loop
    # returns -9999.0 before reaching the warn site, so no warning fires.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve_nodata_attr(
            {'nodatavals': ('foo', -9999.0)}) == -9999.0


def test_resolve_nodata_attr_no_warning_when_nodatavals_all_nan():
    # NaN entries are skipped (they signal "the float NaN is the sentinel",
    # which doesn't need a GDAL_NODATA tag) but they ARE numeric, so the
    # all-non-numeric warning must not fire for an all-NaN tuple.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve_nodata_attr({'nodatavals': (float('nan'),)}) is None


# --- ``build_geo_tags`` lower-level bool rejection -------------------------


@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), np.bool_(False)],
)
def test_to_geotiff_rejects_bool_nodata(uint8_da, tmp_path, bad):
    """``to_geotiff`` raises ``TypeError`` for any bool / np.bool_ nodata."""
    path = str(tmp_path / "tmp_1911_bool.tif")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        to_geotiff(uint8_da, path, nodata=bad)


@pytest.mark.parametrize(
    "good",
    [0, 0.0, -9999, 255, np.int16(-1), np.float32(0.5)],
)
def test_to_geotiff_accepts_numeric_nodata(uint8_da, tmp_path, good):
    """Numeric nodata sentinels still go through without complaint."""
    path = str(tmp_path / "tmp_1911_numeric.tif")
    # uint8 raster cannot hold negative sentinels at the pixel level, but
    # the writer must still accept them at the tag level (the on-disk
    # GDAL_NODATA string round-trips intact).
    to_geotiff(uint8_da, path, nodata=good)


def test_to_geotiff_accepts_none_nodata(uint8_da, tmp_path):
    """``nodata=None`` is the documented default and must keep working."""
    path = str(tmp_path / "tmp_1911_none.tif")
    to_geotiff(uint8_da, path, nodata=None)
    r = open_geotiff(path)
    # No nodata tag was written, so no nodata attribute appears on read.
    assert "nodata" not in r.attrs


def test_to_geotiff_numeric_nodata_roundtrips(tmp_path):
    """A valid numeric sentinel survives the to_geotiff/open_geotiff cycle."""
    arr = np.full((4, 4), 7, dtype=np.uint8)
    arr[0, 0] = 255  # mark one pixel as the sentinel
    da = xr.DataArray(arr, dims=['y', 'x'])
    path = str(tmp_path / "tmp_1911_roundtrip.tif")
    to_geotiff(da, path, nodata=255)

    r = open_geotiff(path)
    # Reader surfaces the numeric sentinel.
    assert "nodata" in r.attrs
    assert int(float(r.attrs["nodata"])) == 255


@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), np.bool_(False)],
)
def test_build_geo_tags_rejects_bool_nodata(bad):
    """The lower-level builder also rejects bool, in case ``to_geotiff``
    is bypassed."""
    transform = GeoTransform()
    with pytest.raises(TypeError, match="nodata must be numeric"):
        build_geo_tags(transform, crs_epsg=4326, nodata=bad)


def test_build_geo_tags_accepts_numeric_nodata():
    """``build_geo_tags`` still writes numeric sentinels into the tag dict."""
    from xrspatial.geotiff._geotags import TAG_GDAL_NODATA
    transform = GeoTransform()
    tags = build_geo_tags(transform, crs_epsg=4326, nodata=-9999)
    assert tags[TAG_GDAL_NODATA] == "-9999"


# ===========================================================================
# 64-bit integer sentinel full-precision parsing
# ===========================================================================


class TestParseNodataStr:
    """Pin the int-first, float-fallback contract."""

    def test_uint64_max_round_trips_as_int(self):
        v = _parse_nodata_str(str(2**64 - 1))
        assert isinstance(v, int)
        assert v == 2**64 - 1

    def test_int64_max_round_trips_as_int(self):
        v = _parse_nodata_str(str(2**63 - 1))
        assert isinstance(v, int)
        assert v == 2**63 - 1

    def test_int64_min_round_trips_as_int(self):
        v = _parse_nodata_str(str(-(2**63)))
        assert isinstance(v, int)
        assert v == -(2**63)

    def test_negative_int_round_trips_as_int(self):
        v = _parse_nodata_str("-9999")
        assert isinstance(v, int)
        assert v == -9999

    def test_uint16_max_round_trips_as_int(self):
        v = _parse_nodata_str("65535")
        assert isinstance(v, int)
        assert v == 65535

    def test_nan_falls_back_to_float(self):
        v = _parse_nodata_str("nan")
        assert isinstance(v, float)
        assert np.isnan(v)

    def test_inf_falls_back_to_float(self):
        v = _parse_nodata_str("inf")
        assert isinstance(v, float)
        assert np.isinf(v)

    def test_negative_inf_falls_back_to_float(self):
        v = _parse_nodata_str("-inf")
        assert isinstance(v, float)
        assert np.isinf(v) and v < 0

    def test_scientific_notation_falls_back_to_float(self):
        v = _parse_nodata_str("1.5e10")
        assert isinstance(v, float)
        assert v == 1.5e10

    def test_fractional_falls_back_to_float(self):
        v = _parse_nodata_str("-9999.25")
        assert isinstance(v, float)
        assert v == -9999.25

    def test_empty_string_returns_none(self):
        assert _parse_nodata_str("") is None
        assert _parse_nodata_str("   ") is None

    def test_whitespace_stripped(self):
        v = _parse_nodata_str("  42  ")
        assert isinstance(v, int)
        assert v == 42

    def test_garbage_returns_none(self):
        assert _parse_nodata_str("hello") is None

    def test_none_input_returns_none(self):
        assert _parse_nodata_str(None) is None


class TestOpenGeotiffEager:
    """``open_geotiff`` must mask the 64-bit sentinel even when its
    float64 representation collides with one ULP above the dtype max."""

    def _write(self, tmp_path, dtype, sentinel):
        arr = np.full((16, 16), 100, dtype=dtype)
        arr[0, 0] = sentinel
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": np.arange(16.0), "x": np.arange(16.0)},
        )
        path = os.path.join(tmp_path, "t.tif")
        to_geotiff(da, path, nodata=sentinel)
        return path

    def test_uint64_max_masked_to_nan(self, tmp_path):
        path = self._write(str(tmp_path), np.uint64, 2**64 - 1)
        da = open_geotiff(path, masked=True)
        assert da.dtype == np.float64
        assert np.isnan(da.values[0, 0])
        assert da.values[1, 1] == 100.0
        # attrs preserves the exact sentinel for a write round-trip.
        assert da.attrs["nodata"] == 2**64 - 1

    def test_int64_max_masked_to_nan(self, tmp_path):
        path = self._write(str(tmp_path), np.int64, 2**63 - 1)
        da = open_geotiff(path, masked=True)
        assert da.dtype == np.float64
        assert np.isnan(da.values[0, 0])
        assert da.values[1, 1] == 100.0
        assert da.attrs["nodata"] == 2**63 - 1

    def test_int64_min_masked_to_nan(self, tmp_path):
        # Regression guard: INT64_MIN is exactly representable in float64
        # and worked before the fix.  Make sure the new int-first path
        # has not broken it.
        path = self._write(str(tmp_path), np.int64, -(2**63))
        da = open_geotiff(path, masked=True)
        assert da.dtype == np.float64
        assert np.isnan(da.values[0, 0])
        assert da.values[1, 1] == 100.0
        assert da.attrs["nodata"] == -(2**63)

    def test_uint16_max_still_masked(self, tmp_path):
        # Regression guard: small integer sentinels still work.
        path = self._write(str(tmp_path), np.uint16, 65535)
        da = open_geotiff(path, masked=True)
        assert da.dtype == np.float64
        assert np.isnan(da.values[0, 0])
        assert da.values[1, 1] == 100.0
        assert da.attrs["nodata"] == 65535

    def test_int32_negative_still_masked(self, tmp_path):
        # Regression guard: signed-int small sentinels still work.
        path = self._write(str(tmp_path), np.int32, -9999)
        da = open_geotiff(path, masked=True)
        assert da.dtype == np.float64
        assert np.isnan(da.values[0, 0])
        assert da.attrs["nodata"] == -9999

    def test_float_nodata_still_parses(self, tmp_path):
        # Regression guard: float dtypes still get float-parsed.
        arr = np.full((8, 8), 1.0, dtype=np.float32)
        arr[0, 0] = -9999.0
        da = xr.DataArray(arr, dims=("y", "x"))
        path = os.path.join(str(tmp_path), "f.tif")
        to_geotiff(da, path, nodata=-9999.0)
        out = open_geotiff(path, masked=True)
        assert np.isnan(out.values[0, 0])


class TestReadGeotiffDask:
    def test_uint64_max_masked_via_dask(self, tmp_path):
        arr = np.full((32, 32), 100, dtype=np.uint64)
        arr[0, 0] = 2**64 - 1
        da_in = xr.DataArray(arr, dims=("y", "x"))
        path = os.path.join(str(tmp_path), "t.tif")
        to_geotiff(da_in, path, nodata=2**64 - 1)
        out = _read_geotiff_dask(path, chunks=16, mask_nodata=True).compute()
        assert out.dtype == np.float64
        assert np.isnan(out.values[0, 0])
        assert out.values[1, 1] == 100.0

    def test_int64_max_masked_via_dask(self, tmp_path):
        arr = np.full((32, 32), 100, dtype=np.int64)
        arr[0, 0] = 2**63 - 1
        da_in = xr.DataArray(arr, dims=("y", "x"))
        path = os.path.join(str(tmp_path), "t.tif")
        to_geotiff(da_in, path, nodata=2**63 - 1)
        out = _read_geotiff_dask(path, chunks=16, mask_nodata=True).compute()
        assert out.dtype == np.float64
        assert np.isnan(out.values[0, 0])


class TestVrtRoundTrip:
    """write_vrt -> _read_vrt round-trip -- the path that surfaced the bug
    in the wild (_build_vrt stringifies geo_info.nodata into XML)."""

    def test_uint64_max_round_trip_via_vrt(self, tmp_path):
        arr = np.full((16, 16), 100, dtype=np.uint64)
        arr[0, 0] = 2**64 - 1
        da_in = xr.DataArray(arr, dims=("y", "x"))
        tif_path = os.path.join(str(tmp_path), "t.tif")
        to_geotiff(da_in, tif_path, nodata=2**64 - 1)

        vrt_path = os.path.join(str(tmp_path), "t.vrt")
        _build_vrt(vrt_path, [tif_path])

        # The VRT XML should carry the integer string literal, not a
        # scientific-notation float that loses one ULP at the dtype max.
        with open(vrt_path) as f:
            xml = f.read()
        assert "<NoDataValue>18446744073709551615</NoDataValue>" in xml

        out = _read_vrt(vrt_path, mask_nodata=True)
        assert out.dtype == np.float64
        assert np.isnan(out.values[0, 0])
        assert out.values[1, 1] == 100.0
        assert out.attrs["nodata"] == 2**64 - 1

    def test_int64_max_round_trip_via_vrt(self, tmp_path):
        arr = np.full((16, 16), 100, dtype=np.int64)
        arr[0, 0] = 2**63 - 1
        da_in = xr.DataArray(arr, dims=("y", "x"))
        tif_path = os.path.join(str(tmp_path), "t.tif")
        to_geotiff(da_in, tif_path, nodata=2**63 - 1)

        vrt_path = os.path.join(str(tmp_path), "t.vrt")
        _build_vrt(vrt_path, [tif_path])

        with open(vrt_path) as f:
            xml = f.read()
        assert "<NoDataValue>9223372036854775807</NoDataValue>" in xml

        out = _read_vrt(vrt_path, mask_nodata=True)
        assert out.dtype == np.float64
        assert np.isnan(out.values[0, 0])
        assert out.values[1, 1] == 100.0


class TestGpuPathParity:
    @requires_gpu
    def test_uint64_max_masked_via_gpu(self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu

        arr = np.full((16, 16), 100, dtype=np.uint64)
        arr[0, 0] = 2**64 - 1
        da_in = xr.DataArray(arr, dims=("y", "x"))
        path = os.path.join(str(tmp_path), "t.tif")
        to_geotiff(da_in, path, nodata=2**64 - 1)

        gpu_da = _read_geotiff_gpu(path, mask_nodata=True)
        host = gpu_da.data.get()
        assert host.dtype == np.float64
        assert np.isnan(host[0, 0])
        assert host[1, 1] == 100.0


# ===========================================================================
# Out-of-range integer sentinel is a value-match no-op
# ===========================================================================


@pytest.fixture
def uint16_neg_nodata_tif(tmp_path):
    """Write a uint16 TIFF with an out-of-range negative nodata sentinel.

    Uses pytest's ``tmp_path`` rather than ``tempfile.NamedTemporaryFile``
    so the mmap cache in ``_reader.py`` does not block teardown on Windows
    (open mmaps cannot be unlinked there).
    """
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    da = xr.DataArray(arr, dims=['y', 'x'])
    path = str(tmp_path / 'uint16_neg_nodata.tif')
    to_geotiff(da, path, crs=4326, nodata=-9999)
    yield path, arr


def test_int_nodata_in_range_helper():
    """The range check rejects out-of-range sentinels per dtype."""
    assert _int_nodata_in_range(-9999, np.dtype('int16')) is True
    assert _int_nodata_in_range(-9999, np.dtype('uint16')) is False
    assert _int_nodata_in_range(255, np.dtype('uint8')) is True
    assert _int_nodata_in_range(256, np.dtype('uint8')) is False
    assert _int_nodata_in_range(0, np.dtype('uint8')) is True
    # float dtype isn't an integer sentinel target
    assert _int_nodata_in_range(0, np.dtype('float32')) is False


def test_resolve_masked_fill_out_of_range_falls_back_to_zero():
    """uint16 + nodata=-9999 cannot be represented; fall back to 0."""
    fill = _resolve_masked_fill('-9999', np.dtype('uint16'))
    assert fill == 0
    assert fill.dtype == np.dtype('uint16')


def test_resolve_masked_fill_in_range_uses_sentinel():
    """uint16 + nodata=65535 stays as 65535 (in range)."""
    fill = _resolve_masked_fill('65535', np.dtype('uint16'))
    assert fill == 65535
    assert fill.dtype == np.dtype('uint16')


def test_open_geotiff_uint16_negative_nodata_does_not_raise(
        uint16_neg_nodata_tif):
    """The eager read path no longer crashes on uint16 + negative nodata."""
    path, expected = uint16_neg_nodata_tif
    result = open_geotiff(path)
    # Dtype is preserved (sentinel can't match -> no float promotion).
    assert result.dtype == np.uint16
    # Pixel values are intact.
    np.testing.assert_array_equal(result.values, expected)
    # The sentinel survives via attrs so a write round-trip keeps it.
    assert result.attrs.get('nodata') == -9999.0


def test_read_geotiff_dask_uint16_negative_nodata_graph(
        uint16_neg_nodata_tif):
    """The dask graph-construction path no longer crashes."""
    path, _ = uint16_neg_nodata_tif
    result = _read_geotiff_dask(path, chunks=2)
    # No promotion to float64 -- sentinel is unrepresentable so masking
    # would be a no-op anyway.
    assert result.dtype == np.uint16
    assert result.shape == (2, 3)
    assert result.attrs.get('nodata') == -9999.0


def test_read_geotiff_dask_uint16_negative_nodata_compute(
        uint16_neg_nodata_tif):
    """Dask compute returns the file's pixels unchanged."""
    path, expected = uint16_neg_nodata_tif
    result = _read_geotiff_dask(path, chunks=2).compute()
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result.values, expected)


def test_open_geotiff_uint16_in_range_nodata_still_masks(tmp_path):
    """The fix doesn't regress the in-range case: uint16 + nodata=65535
    still promotes to float64 and masks to NaN."""
    arr = np.array([[1, 2, 3], [4, 5, 65535]], dtype=np.uint16)
    da = xr.DataArray(arr, dims=['y', 'x'])
    path = str(tmp_path / 'uint16_in_range_nodata.tif')
    to_geotiff(da, path, crs=4326, nodata=65535)
    result = open_geotiff(path, masked=True)
    assert result.dtype == np.float64
    # The 65535 pixel should be NaN; the rest unchanged.
    assert np.isnan(result.values[1, 2])
    assert result.values[0, 0] == 1
    assert result.attrs.get('nodata') == 65535.0


@requires_gpu
def test_apply_nodata_mask_gpu_out_of_range_no_crash():
    """The GPU helper falls back gracefully for unrepresentable sentinels.

    This test exercises ``_apply_nodata_mask_gpu`` directly and requires
    cupy with CUDA available.
    """
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.array([[1, 2, 3], [4, 5, 6]], dtype=cupy.uint16)
    out = _apply_nodata_mask_gpu(arr_gpu, -9999)
    # Out-of-range sentinel: array passes through unchanged.
    assert out.dtype == cupy.uint16
    assert cupy.array_equal(out, arr_gpu)


# ===========================================================================
# ``mask_nodata=False`` keeps the source integer dtype
# ===========================================================================


@pytest.fixture
def uint16_with_matching_sentinel(tmp_path):
    """uint16 TIFF where nodata=0 and the array has zeros in it."""
    arr = np.array([[0, 100, 200, 300],
                    [400, 500, 0, 600],
                    [700, 800, 900, 0],
                    [0, 1100, 1200, 1300]], dtype=np.uint16)
    path = str(tmp_path / 'uint16_match_2052.tif')
    write(arr, path, nodata=0, compression='none', tiled=False)
    return path, arr


@pytest.fixture
def uint16_no_match(tmp_path):
    """uint16 TIFF whose nodata sentinel is not present in any pixel."""
    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8]], dtype=np.uint16)
    path = str(tmp_path / 'uint16_nomatch_2052.tif')
    write(arr, path, nodata=65535, compression='none', tiled=False)
    return path, arr


@pytest.fixture
def float32_tiff(tmp_path):
    """float32 TIFF with NaN nodata."""
    arr = np.array([[1.0, 2.0, np.nan],
                    [4.0, np.nan, 6.0]], dtype=np.float32)
    path = str(tmp_path / 'float32_2052.tif')
    write(arr, path, nodata=float('nan'), compression='none', tiled=False)
    return path, arr


def test_regression_dtype_uint16_was_unreachable(
        uint16_with_matching_sentinel):
    """Without the kwarg, ``dtype="uint16"`` raises on a matching sentinel.

    Baseline that documents the broken contract: this is the original
    failure mode reported in the issue. The ``mask_nodata=False``
    branch below is the fix.
    """
    path, _ = uint16_with_matching_sentinel
    with pytest.raises(ValueError):
        open_geotiff(path, dtype='uint16', masked=True)


def test_mask_nodata_false_preserves_uint16(uint16_with_matching_sentinel):
    """``mask_nodata=False`` keeps the uint16 source dtype."""
    path, arr = uint16_with_matching_sentinel
    da = open_geotiff(path, dtype='uint16', mask_nodata=False)
    assert da.dtype == np.uint16
    # Raw sentinels survive in the data.
    np.testing.assert_array_equal(da.values, arr)
    # The declared sentinel is still surfaced for downstream maskers.
    assert da.attrs['nodata'] == 0


def test_mask_nodata_false_no_dtype_kwarg(uint16_with_matching_sentinel):
    """Without ``dtype=``, the source dtype is preserved as-is."""
    path, arr = uint16_with_matching_sentinel
    da = open_geotiff(path, mask_nodata=False)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.values, arr)
    assert da.attrs['nodata'] == 0


def test_default_mask_nodata_true_still_promotes(
        uint16_with_matching_sentinel):
    """The new default (``masked=False``) does NOT promote or mask.

    ``open_geotiff`` now matches rioxarray's ``open_rasterio``: a bare
    read keeps the source integer dtype and leaves the sentinel pixels
    untouched. Passing ``masked=True`` restores the old promote-to-NaN
    behaviour.
    """
    path, _ = uint16_with_matching_sentinel
    # Default: no masking -> uint16 preserved, sentinels survive.
    da = open_geotiff(path)
    assert da.dtype == np.uint16
    assert not np.isnan(np.asarray(da.values, dtype=np.float64)).any()
    assert da.attrs.get('masked_nodata') in (False, None)

    # Opt in: masking promotes to float64 and replaces the sentinel.
    masked = open_geotiff(path, masked=True)
    assert masked.dtype == np.float64
    assert np.isnan(masked.values).sum() == 4
    assert np.isnan(masked.values[0, 0])
    assert np.isnan(masked.values[1, 2])


def test_no_match_both_modes_agree(uint16_no_match):
    """When the sentinel does not match any pixel, both modes return the
    same uint16 array (no promotion needed in either case).
    """
    path, arr = uint16_no_match
    masked = open_geotiff(path)
    unmasked = open_geotiff(path, mask_nodata=False)
    assert masked.dtype == np.uint16
    assert unmasked.dtype == np.uint16
    np.testing.assert_array_equal(masked.values, arr)
    np.testing.assert_array_equal(unmasked.values, arr)


def test_float_file_mask_nodata_false_keeps_data(float32_tiff):
    """For a float32 file with NaN nodata, ``mask_nodata=False`` is a
    no-op: the sentinel is NaN so the inline mask would do nothing
    anyway, and the float dtype is preserved either way.
    """
    path, arr = float32_tiff
    masked = open_geotiff(path)
    unmasked = open_geotiff(path, mask_nodata=False)
    assert masked.dtype == np.float32
    assert unmasked.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(masked.values),
                                  np.isnan(arr))
    np.testing.assert_array_equal(np.isnan(unmasked.values),
                                  np.isnan(arr))


def test_dtype_cast_preservation_uint8(tmp_path):
    """Casting to a different integer dtype also works with the opt-out.

    The reader keeps the source dtype (uint16) via ``mask_nodata=False``,
    then ``dtype="uint32"`` casts integer-to-integer, which is allowed.
    """
    arr = np.array([[0, 100, 200],
                    [300, 0, 500]], dtype=np.uint16)
    path = str(tmp_path / 'uint16_to_uint32_2052.tif')
    write(arr, path, nodata=0, compression='none', tiled=False)

    da = open_geotiff(path, dtype='uint32', mask_nodata=False)
    assert da.dtype == np.uint32
    np.testing.assert_array_equal(da.values, arr.astype(np.uint32))


def test_dask_path_mask_nodata_false(uint16_with_matching_sentinel):
    """The dask path honours the kwarg too: integer source dtype survives.

    Without this, ``_read_geotiff_dask`` would still promote the dask
    graph dtype to float64 and force the per-chunk cast.
    """
    path, arr = uint16_with_matching_sentinel
    da = open_geotiff(path, chunks=2, mask_nodata=False)
    assert da.dtype == np.uint16
    computed = da.compute()
    assert computed.dtype == np.uint16
    np.testing.assert_array_equal(computed.values, arr)
    assert computed.attrs['nodata'] == 0


def test_dask_path_default_still_promotes(uint16_with_matching_sentinel):
    """The dask default (``masked=False``) does NOT promote to float64.

    Mirrors the eager-path new default: a bare chunked read keeps the
    uint16 source dtype and leaves the sentinel pixels untouched.
    ``masked=True`` restores the promote-to-NaN behaviour.
    """
    path, _ = uint16_with_matching_sentinel
    # Default: no masking -> uint16 preserved on the dask graph.
    da = open_geotiff(path, chunks=2)
    assert da.dtype == np.uint16
    computed = da.compute()
    assert computed.dtype == np.uint16
    assert not np.isnan(
        np.asarray(computed.values, dtype=np.float64)).any()

    # Opt in: masking promotes to float64 and replaces the sentinel.
    masked = open_geotiff(path, chunks=2, masked=True)
    assert masked.dtype == np.float64
    assert np.isnan(masked.compute().values).sum() == 4


def test_dask_dtype_cast_with_opt_out(uint16_with_matching_sentinel):
    """``dtype="uint16"`` + ``mask_nodata=False`` works on the dask path."""
    path, arr = uint16_with_matching_sentinel
    da = open_geotiff(path, chunks=2, dtype='uint16', mask_nodata=False)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.compute().values, arr)
