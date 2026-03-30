"""Tests for dtype parameter on open_geotiff."""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


@pytest.fixture
def float64_tif(tmp_path):
    """Write a float64 GeoTIFF for dtype cast tests."""
    arr = np.random.default_rng(99).random((80, 80)).astype(np.float64)
    y = np.linspace(40.0, 41.0, 80)
    x = np.linspace(-105.0, -104.0, 80)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326})
    path = str(tmp_path / 'test_1083_f64.tif')
    to_geotiff(da, path, compression='none')
    return path, arr


@pytest.fixture
def uint16_tif(tmp_path):
    """Write a uint16 GeoTIFF for dtype cast tests."""
    arr = np.random.default_rng(77).integers(0, 10000, (60, 60),
                                             dtype=np.uint16)
    y = np.linspace(40.0, 41.0, 60)
    x = np.linspace(-105.0, -104.0, 60)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326})
    path = str(tmp_path / 'test_1083_u16.tif')
    to_geotiff(da, path, compression='none')
    return path, arr


class TestDtypeEager:
    def test_float64_to_float32(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, dtype='float32')
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(
            result.values, orig.astype(np.float32), decimal=6)

    def test_float64_to_float16(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, dtype=np.float16)
        assert result.dtype == np.float16

    def test_uint16_to_int32(self, uint16_tif):
        path, orig = uint16_tif
        result = open_geotiff(path, dtype='int32')
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result.values, orig.astype(np.int32))

    def test_uint16_to_uint8(self, uint16_tif):
        path, _ = uint16_tif
        result = open_geotiff(path, dtype='uint8')
        assert result.dtype == np.uint8

    def test_float_to_int_raises(self, float64_tif):
        path, _ = float64_tif
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, dtype='int32')

    def test_dtype_none_preserves_native(self, float64_tif):
        path, _ = float64_tif
        result = open_geotiff(path, dtype=None)
        assert result.dtype == np.float64


class TestDtypeDask:
    def test_float64_to_float32_dask(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, dtype='float32', chunks=40)
        assert result.dtype == np.float32
        computed = result.values
        np.testing.assert_array_almost_equal(
            computed, orig.astype(np.float32), decimal=6)

    def test_chunks_are_target_dtype(self, float64_tif):
        path, _ = float64_tif
        result = open_geotiff(path, dtype='float32', chunks=40)
        assert result.data.dtype == np.float32

    def test_float_to_int_raises_dask(self, float64_tif):
        path, _ = float64_tif
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, dtype='int32', chunks=40)
