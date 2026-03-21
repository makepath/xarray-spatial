"""Tests for the GeoTIFF writer."""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._writer import write, _make_overview
from xrspatial.geotiff._reader import read_to_array


class TestMakeOverview:
    def test_2x_decimation(self):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        ov = _make_overview(arr)
        assert ov.shape == (4, 4)
        # Check first value: mean of top-left 2x2 block
        expected = np.mean([0, 1, 8, 9])
        assert ov[0, 0] == pytest.approx(expected)

    def test_integer_rounding(self):
        arr = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8]], dtype=np.uint8)
        ov = _make_overview(arr)
        assert ov.shape == (1, 2)
        assert ov.dtype == np.uint8


class TestWriteRoundTrip:
    def test_uncompressed_stripped(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'uncompressed.tif')
        write(expected, path, compression='none', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_deflate_stripped(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'deflate.tif')
        write(expected, path, compression='deflate', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_uncompressed_tiled(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'tiled.tif')
        write(expected, path, compression='none', tiled=True, tile_size=4)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_deflate_tiled(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'deflate_tiled.tif')
        write(expected, path, compression='deflate', tiled=True, tile_size=4)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_lzw_stripped(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'lzw.tif')
        write(expected, path, compression='lzw', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_uint16(self, tmp_path):
        expected = np.arange(100, dtype=np.uint16).reshape(10, 10)
        path = str(tmp_path / 'uint16.tif')
        write(expected, path, compression='none', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_with_geo_info(self, tmp_path):
        expected = np.ones((4, 4), dtype=np.float32)
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'geo.tif')
        write(expected, path, geo_transform=gt, crs_epsg=4326,
              nodata=-9999.0, compression='none', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)
        assert geo.crs_epsg == 4326
        assert geo.transform.origin_x == pytest.approx(-120.0)
        assert geo.transform.pixel_width == pytest.approx(0.001)

    def test_predictor_deflate(self, tmp_path):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'predictor.tif')
        write(expected, path, compression='deflate', tiled=False, predictor=True)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)


class TestWriteInvalidInput:
    def test_unsupported_compression(self, tmp_path):
        arr = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported compression"):
            write(arr, str(tmp_path / 'bad.tif'), compression='jpeg')
