"""Tests for COG writing and the public API."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._writer import write

from .conftest import gpu_available


class TestCOGWriter:
    def test_cog_layout_ifds_before_data(self, tmp_path):
        """COG spec: all IFDs should come before pixel data."""
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / 'cog.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8,
              cog=True, overview_levels=[2])

        with open(path, 'rb') as f:
            data = f.read()

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)

        assert len(ifds) >= 2  # full res + at least 1 overview

        # All IFD offsets should be < the first tile data offset
        all_tile_offsets = []
        for ifd in ifds:
            tile_off = ifd.tile_offsets
            if tile_off:
                all_tile_offsets.extend(tile_off)

        if all_tile_offsets:
            first_data_offset = min(all_tile_offsets)
            # The last IFD byte should be before the first tile data
            # (This is the COG layout requirement)
            assert header.first_ifd_offset < first_data_offset

    def test_cog_round_trip(self, tmp_path):
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'cog_rt.tif')
        write(arr, path, geo_transform=gt, crs_epsg=4326,
              compression='deflate', tiled=True, tile_size=8,
              cog=True, overview_levels=[2])

        result, geo = read_to_array_local(path)
        np.testing.assert_array_equal(result, arr)
        assert geo.crs_epsg == 4326

    def test_cog_auto_overviews(self, tmp_path):
        """Auto-generate overviews when none specified."""
        arr = np.arange(1024, dtype=np.float32).reshape(32, 32)
        path = str(tmp_path / 'cog_auto.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8,
              cog=True)

        with open(path, 'rb') as f:
            data = f.read()

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        # Should have at least 2 IFDs (full res + overviews)
        assert len(ifds) >= 2


class TestPublicAPI:
    def test_read_write_round_trip(self, tmp_path):
        """Write a DataArray, read it back, verify values and coords."""
        y = np.linspace(45.0, 44.0, 10)
        x = np.linspace(-120.0, -119.0, 12)
        data = np.random.RandomState(42).rand(10, 12).astype(np.float32)

        da = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 4326},
            name='test',
        )

        path = str(tmp_path / 'round_trip.tif')
        to_geotiff(da, path, compression='deflate', tiled=False)

        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, data, decimal=5)
        assert result.attrs.get('crs') == 4326

    def test_open_geotiff_name(self, tmp_path):
        """DataArray name defaults to filename stem."""
        arr = np.zeros((4, 4), dtype=np.float32)
        path = str(tmp_path / 'myfile.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path)
        assert da.name == 'myfile'

    def test_open_geotiff_custom_name(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.float32)
        path = str(tmp_path / 'test.tif')
        write(arr, path, compression='none', tiled=False)

        da = open_geotiff(path, name='custom')
        assert da.name == 'custom'

    def test_write_numpy_array(self, tmp_path):
        """to_geotiff should accept raw numpy arrays too."""
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        path = str(tmp_path / 'numpy.tif')
        to_geotiff(arr, path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_write_3d_rgb(self, tmp_path):
        """3D arrays (height, width, bands) should write multi-band."""
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # red channel
        path = str(tmp_path / 'rgb.tif')
        to_geotiff(arr, path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_write_rejects_4d(self, tmp_path):
        arr = np.zeros((2, 3, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            to_geotiff(arr, str(tmp_path / 'bad.tif'))


class TestCOGOverviewResampling:
    """Test overview resampling methods produce correct results."""

    def test_overview_mean(self, tmp_path):
        arr = np.array([[1, 3, 5, 7],
                        [2, 4, 6, 8],
                        [9, 11, 13, 15],
                        [10, 12, 14, 16]], dtype=np.float32)
        path = str(tmp_path / 'cog_1150_mean.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=4,
              cog=True, overview_levels=[2], overview_resampling='mean')

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) == 2
        # Overview should be 2x2
        ov_ifd = ifds[1]
        assert ov_ifd.width == 2
        assert ov_ifd.height == 2

    def test_overview_nearest(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'cog_1150_nearest.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=4,
              cog=True, overview_levels=[2], overview_resampling='nearest')

        result, _ = read_to_array_local(path)
        np.testing.assert_array_equal(result, arr)

    def test_overview_mode(self, tmp_path):
        # Categorical data: mode should pick the most common value
        arr = np.array([[1, 1, 2, 2],
                        [1, 1, 2, 2],
                        [3, 3, 4, 4],
                        [3, 3, 4, 4]], dtype=np.int32)
        path = str(tmp_path / 'cog_1150_mode.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=4,
              cog=True, overview_levels=[2], overview_resampling='mode')

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) == 2

    @pytest.mark.parametrize('method', ['min', 'max', 'median'])
    def test_overview_other_methods(self, tmp_path, method):
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / f'cog_1150_{method}.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8,
              cog=True, overview_levels=[2], overview_resampling=method)

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) >= 2


class TestCOGMultipleOverviews:
    def test_multiple_overview_levels(self, tmp_path):
        """Multiple explicit overview levels produce correct number of IFDs."""
        arr = np.arange(4096, dtype=np.float32).reshape(64, 64)
        path = str(tmp_path / 'cog_1150_multi.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8,
              cog=True, overview_levels=[2, 4, 8])

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        # Full res + 3 overviews
        assert len(ifds) == 4

    def test_auto_overviews_large_raster(self, tmp_path):
        """Auto-generation on a larger raster produces multiple levels."""
        arr = np.random.RandomState(42).rand(512, 512).astype(np.float32)
        path = str(tmp_path / 'cog_1150_auto_large.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=64,
              cog=True)

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        # 512 -> 256 -> 128 -> 64: should stop, so 3 overview levels + full = 4
        assert len(ifds) >= 3

    def test_cog_overview_round_trip_values(self, tmp_path):
        """Full-res values are preserved through COG write with overviews."""
        arr = np.random.RandomState(99).rand(32, 32).astype(np.float32)
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'cog_1150_rt_values.tif')
        write(arr, path, geo_transform=gt, crs_epsg=4326,
              compression='deflate', tiled=True, tile_size=16,
              cog=True, overview_levels=[2, 4])

        result, geo = read_to_array_local(path)
        np.testing.assert_array_equal(result, arr)
        assert geo.crs_epsg == 4326


class TestCOGPublicAPIOverviews:
    def test_to_geotiff_cog_with_overviews(self, tmp_path):
        """Public to_geotiff() with cog=True writes overviews."""
        y = np.linspace(45.0, 44.0, 32)
        x = np.linspace(-120.0, -119.0, 32)
        data = np.random.RandomState(42).rand(32, 32).astype(np.float32)

        da = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 4326},
        )

        path = str(tmp_path / 'cog_1150_api.tif')
        to_geotiff(da, path, compression='deflate', cog=True,
                   tile_size=16, overview_levels=[2])

        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, data, decimal=5)

        # Verify COG structure
        with open(path, 'rb') as f:
            raw = f.read()
        header = parse_header(raw)
        ifds = parse_all_ifds(raw, header)
        assert len(ifds) >= 2

    def test_to_geotiff_cog_auto_overviews(self, tmp_path):
        """Public API auto-generates overviews when only cog=True."""
        data = np.random.RandomState(7).rand(64, 64).astype(np.float32)
        da = xr.DataArray(data, dims=['y', 'x'])

        path = str(tmp_path / 'cog_1150_api_auto.tif')
        to_geotiff(da, path, compression='deflate', cog=True, tile_size=16)

        with open(path, 'rb') as f:
            raw = f.read()
        header = parse_header(raw)
        ifds = parse_all_ifds(raw, header)
        assert len(ifds) >= 2


_HAS_GPU = gpu_available()


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
class TestGPUCOGOverviews:
    """GPU-specific COG overview tests (require CuPy + CUDA)."""

    def test_gpu_cog_round_trip(self, tmp_path):
        import cupy
        arr = np.random.RandomState(42).rand(32, 32).astype(np.float32)
        gpu_arr = cupy.asarray(arr)

        path = str(tmp_path / 'cog_1150_gpu_rt.tif')
        from xrspatial.geotiff import write_geotiff_gpu
        write_geotiff_gpu(gpu_arr, path, crs=4326, compression='deflate',
                          cog=True, overview_levels=[2])

        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, arr, decimal=5)

        with open(path, 'rb') as f:
            raw = f.read()
        header = parse_header(raw)
        ifds = parse_all_ifds(raw, header)
        assert len(ifds) >= 2

    def test_gpu_cog_auto_overviews(self, tmp_path):
        import cupy
        arr = np.random.RandomState(7).rand(64, 64).astype(np.float32)
        gpu_arr = cupy.asarray(arr)

        path = str(tmp_path / 'cog_1150_gpu_auto.tif')
        from xrspatial.geotiff import write_geotiff_gpu
        write_geotiff_gpu(gpu_arr, path, compression='deflate',
                          cog=True, tile_size=16)

        with open(path, 'rb') as f:
            raw = f.read()
        header = parse_header(raw)
        ifds = parse_all_ifds(raw, header)
        assert len(ifds) >= 2

    def test_gpu_overview_resampling_nearest(self, tmp_path):
        import cupy
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        gpu_arr = cupy.asarray(arr)

        path = str(tmp_path / 'cog_1150_gpu_nearest.tif')
        from xrspatial.geotiff import write_geotiff_gpu
        write_geotiff_gpu(gpu_arr, path, compression='deflate',
                          cog=True, overview_levels=[2],
                          overview_resampling='nearest')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_gpu_make_overview_values(self):
        """GPU overview block-reduce matches CPU for simple case."""
        import cupy

        from xrspatial.geotiff._gpu_decode import make_overview_gpu
        from xrspatial.geotiff._writer import _make_overview

        arr = np.random.RandomState(42).rand(16, 16).astype(np.float32)
        gpu_arr = cupy.asarray(arr)

        for method in ('mean', 'nearest', 'min', 'max'):
            cpu_ov = _make_overview(arr, method=method)
            gpu_ov = make_overview_gpu(gpu_arr, method=method).get()
            np.testing.assert_allclose(gpu_ov, cpu_ov, rtol=1e-5,
                                       err_msg=f"Mismatch for method={method}")

    def test_gpu_to_geotiff_dispatches_with_overviews(self, tmp_path):
        """to_geotiff auto-dispatches CuPy data with overview params."""
        import cupy
        arr = np.random.RandomState(11).rand(32, 32).astype(np.float32)
        da = xr.DataArray(cupy.asarray(arr), dims=['y', 'x'],
                          attrs={'crs': 4326})

        path = str(tmp_path / 'cog_1150_gpu_dispatch.tif')
        to_geotiff(da, path, compression='deflate', cog=True,
                   overview_levels=[2])

        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, arr, decimal=5)

        with open(path, 'rb') as f:
            raw = f.read()
        header = parse_header(raw)
        ifds = parse_all_ifds(raw, header)
        assert len(ifds) >= 2


def read_to_array_local(path):
    """Helper to call read_to_array for local files."""
    from xrspatial.geotiff._reader import read_to_array
    return read_to_array(path)
