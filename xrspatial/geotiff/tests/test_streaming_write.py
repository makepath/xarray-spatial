"""Tests for streaming TIFF write from dask-backed DataArrays (#1084)."""
import numpy as np
import os
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


@pytest.fixture
def sample_raster():
    """200x200 float32 raster with coords and CRS."""
    arr = np.random.default_rng(1084).random((200, 200), dtype=np.float32)
    y = np.linspace(41.0, 40.0, 200)
    x = np.linspace(-106.0, -105.0, 200)
    return xr.DataArray(arr, dims=['y', 'x'],
                        coords={'y': y, 'x': x},
                        attrs={'crs': 4326, 'nodata': -9999.0})


@pytest.fixture
def dask_raster(sample_raster):
    return sample_raster.chunk({'y': 100, 'x': 100})


# -- Round-trip correctness --------------------------------------------------

class TestStreamingRoundTrip:
    def test_tiled_zstd(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'tiled_zstd_1084.tif')
        to_geotiff(dask_raster, path, compression='zstd')
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_tiled_deflate(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'tiled_deflate_1084.tif')
        to_geotiff(dask_raster, path, compression='deflate')
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_tiled_lzw(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'tiled_lzw_1084.tif')
        to_geotiff(dask_raster, path, compression='lzw')
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_tiled_uncompressed(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'tiled_none_1084.tif')
        to_geotiff(dask_raster, path, compression='none')
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_stripped(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'stripped_1084.tif')
        to_geotiff(dask_raster, path, tiled=False)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_predictor(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'pred_1084.tif')
        to_geotiff(dask_raster, path, predictor=True)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_compression_level(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'level_1084.tif')
        to_geotiff(dask_raster, path, compression='deflate',
                   compression_level=1)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_matches_eager_write(self, sample_raster, dask_raster, tmp_path):
        """Streaming and eager paths should produce identical pixel data."""
        eager_path = str(tmp_path / 'eager_1084.tif')
        stream_path = str(tmp_path / 'stream_1084.tif')

        to_geotiff(sample_raster, eager_path)   # numpy -> eager
        to_geotiff(dask_raster, stream_path)     # dask -> streaming

        eager = open_geotiff(eager_path)
        stream = open_geotiff(stream_path)
        np.testing.assert_array_equal(eager.values, stream.values)


# -- Geo metadata preservation -----------------------------------------------

class TestStreamingGeoMetadata:
    def test_crs_preserved(self, dask_raster, tmp_path):
        path = str(tmp_path / 'crs_1084.tif')
        to_geotiff(dask_raster, path)
        result = open_geotiff(path)
        assert result.attrs.get('crs') == 4326

    def test_nodata_preserved(self, dask_raster, tmp_path):
        path = str(tmp_path / 'nd_1084.tif')
        to_geotiff(dask_raster, path)
        result = open_geotiff(path)
        assert float(result.attrs.get('nodata')) == pytest.approx(-9999.0)

    def test_coordinates_preserved(self, sample_raster, dask_raster, tmp_path):
        path = str(tmp_path / 'coords_1084.tif')
        to_geotiff(dask_raster, path)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.coords['x'].values, sample_raster.coords['x'].values,
            decimal=6)
        np.testing.assert_array_almost_equal(
            result.coords['y'].values, sample_raster.coords['y'].values,
            decimal=6)


# -- Edge cases ---------------------------------------------------------------

class TestStreamingEdgeCases:
    def test_nan_to_nodata(self, tmp_path):
        """NaN pixels should round-trip through the nodata sentinel."""
        arr = np.ones((100, 100), dtype=np.float32)
        arr[10:20, 10:20] = np.nan
        da = xr.DataArray(arr, dims=['y', 'x'],
                          attrs={'nodata': -9999.0})
        dask_da = da.chunk({'y': 50, 'x': 50})

        path = str(tmp_path / 'nan_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)

        assert np.isnan(result.values[15, 15])
        assert result.values[0, 0] == pytest.approx(1.0)

    def test_single_chunk(self, sample_raster, tmp_path):
        """Single chunk = whole array, but still goes through streaming."""
        dask_da = sample_raster.chunk({'y': 200, 'x': 200})
        path = str(tmp_path / 'single_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_uneven_chunks(self, tmp_path):
        """Chunks that don't divide evenly into tile_size."""
        arr = np.arange(150 * 170, dtype=np.float32).reshape(150, 170)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 64, 'x': 64})

        path = str(tmp_path / 'uneven_1084.tif')
        to_geotiff(dask_da, path, tile_size=128)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_small_raster(self, tmp_path):
        """Raster smaller than one tile."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 2, 'x': 2})

        path = str(tmp_path / 'tiny_1084.tif')
        to_geotiff(dask_da, path, tile_size=256)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_uint16(self, tmp_path):
        arr = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 50, 'x': 50})

        path = str(tmp_path / 'u16_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_int32(self, tmp_path):
        arr = np.arange(10000, dtype=np.int32).reshape(100, 100)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 50, 'x': 50})

        path = str(tmp_path / 'i32_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_float64(self, tmp_path):
        arr = np.random.default_rng(1084).random((80, 80))
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 40, 'x': 40})

        path = str(tmp_path / 'f64_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, arr)


# -- Multiband ----------------------------------------------------------------

class TestStreamingMultiband:
    def test_3d_band_last(self, tmp_path):
        """3D array with (y, x, band) layout."""
        arr = np.random.default_rng(1084).random(
            (100, 100, 3), dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x', 'band'])
        dask_da = da.chunk({'y': 50, 'x': 50})

        path = str(tmp_path / 'band_last_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, arr, decimal=5)

    def test_3d_band_first(self, tmp_path):
        """Band-first (band, y, x) DataArray gets transposed automatically."""
        arr = np.random.default_rng(1084).random(
            (3, 100, 100), dtype=np.float32)
        da = xr.DataArray(arr, dims=['band', 'y', 'x'])
        dask_da = da.chunk({'y': 50, 'x': 50})

        path = str(tmp_path / 'band_first_1084.tif')
        to_geotiff(dask_da, path)
        result = open_geotiff(path)
        # Result is (y, x, band), so compare transposed
        np.testing.assert_array_almost_equal(
            result.values, np.moveaxis(arr, 0, -1), decimal=5)


# -- BigTIFF and error cases --------------------------------------------------

class TestStreamingBigTiffAndErrors:
    def test_forced_bigtiff(self, tmp_path):
        """bigtiff=True on a small array should produce a valid BigTIFF."""
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 4, 'x': 4})

        path = str(tmp_path / 'bigtiff_1084.tif')
        to_geotiff(dask_da, path, bigtiff=True)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_cloud_uri_raises(self, tmp_path):
        """Streaming to cloud storage should raise NotImplementedError."""
        arr = np.ones((10, 10), dtype=np.float32)
        da = xr.DataArray(arr, dims=['y', 'x'])
        dask_da = da.chunk({'y': 5, 'x': 5})

        with pytest.raises(NotImplementedError, match='cloud'):
            to_geotiff(dask_da, 's3://bucket/file.tif')


# -- COG fallback to eager path -----------------------------------------------

class TestCogFallback:
    def test_cog_with_dask_still_works(self, sample_raster, tmp_path):
        """cog=True with dask input should fall through to eager compute."""
        dask_da = sample_raster.chunk({'y': 100, 'x': 100})
        path = str(tmp_path / 'cog_1084.tif')
        to_geotiff(dask_da, path, cog=True)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)
