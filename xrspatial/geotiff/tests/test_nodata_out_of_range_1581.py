"""Regression tests for issue #1581.

A TIFF that pairs an unsigned integer dtype with a negative GDAL_NODATA
sentinel (e.g. uint16 + ``-9999``) used to raise ``OverflowError`` on
every read path because the eager / dask / GPU nodata-mask code all did
``arr.dtype.type(int(nodata))`` without checking dtype range. The fix
treats an out-of-range sentinel as a no-op for value matching (the file
can never contain a uint16 value of -9999), keeps the file dtype, and
still surfaces the sentinel via ``attrs['nodata']`` so write round-trips
preserve it.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, read_geotiff_dask, to_geotiff
from xrspatial.geotiff._reader import _int_nodata_in_range, _resolve_masked_fill


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialized."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


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
    result = read_geotiff_dask(path, chunks=2)
    # No promotion to float64 -- sentinel is unrepresentable so masking
    # would be a no-op anyway.
    assert result.dtype == np.uint16
    assert result.shape == (2, 3)
    assert result.attrs.get('nodata') == -9999.0


def test_read_geotiff_dask_uint16_negative_nodata_compute(
        uint16_neg_nodata_tif):
    """Dask compute returns the file's pixels unchanged."""
    path, expected = uint16_neg_nodata_tif
    result = read_geotiff_dask(path, chunks=2).compute()
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result.values, expected)


def test_open_geotiff_uint16_in_range_nodata_still_masks(tmp_path):
    """The fix doesn't regress the in-range case: uint16 + nodata=65535
    still promotes to float64 and masks to NaN."""
    arr = np.array([[1, 2, 3], [4, 5, 65535]], dtype=np.uint16)
    da = xr.DataArray(arr, dims=['y', 'x'])
    path = str(tmp_path / 'uint16_in_range_nodata.tif')
    to_geotiff(da, path, crs=4326, nodata=65535)
    result = open_geotiff(path)
    assert result.dtype == np.float64
    # The 65535 pixel should be NaN; the rest unchanged.
    assert np.isnan(result.values[1, 2])
    assert result.values[0, 0] == 1
    assert result.attrs.get('nodata') == 65535.0


@_gpu_only
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
