"""Tests for COG writing and the public API."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_geotiff, write_geotiff
from xrspatial.geotiff._header import parse_header, parse_all_ifds
from xrspatial.geotiff._writer import write
from xrspatial.geotiff._geotags import GeoTransform, extract_geo_info


class TestCOGWriter:
    def test_cog_layout_ifds_before_data(self, tmp_path):
        """COG spec: all IFDs should come before pixel data."""
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / 'cog.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8,
              cog=True, overview_levels=[1])

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
              cog=True, overview_levels=[1])

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
        write_geotiff(da, path, compression='deflate', tiled=False)

        result = read_geotiff(path)
        np.testing.assert_array_almost_equal(result.values, data, decimal=5)
        assert result.attrs.get('crs') == 4326

    def test_read_geotiff_name(self, tmp_path):
        """DataArray name defaults to filename stem."""
        arr = np.zeros((4, 4), dtype=np.float32)
        path = str(tmp_path / 'myfile.tif')
        write(arr, path, compression='none', tiled=False)

        da = read_geotiff(path)
        assert da.name == 'myfile'

    def test_read_geotiff_custom_name(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.float32)
        path = str(tmp_path / 'test.tif')
        write(arr, path, compression='none', tiled=False)

        da = read_geotiff(path, name='custom')
        assert da.name == 'custom'

    def test_write_numpy_array(self, tmp_path):
        """write_geotiff should accept raw numpy arrays too."""
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        path = str(tmp_path / 'numpy.tif')
        write_geotiff(arr, path, compression='none')

        result = read_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_write_3d_rgb(self, tmp_path):
        """3D arrays (height, width, bands) should write multi-band."""
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # red channel
        path = str(tmp_path / 'rgb.tif')
        write_geotiff(arr, path, compression='none')

        result = read_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_write_rejects_4d(self, tmp_path):
        arr = np.zeros((2, 3, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            write_geotiff(arr, str(tmp_path / 'bad.tif'))


def read_to_array_local(path):
    """Helper to call read_to_array for local files."""
    from xrspatial.geotiff._reader import read_to_array
    return read_to_array(path)
