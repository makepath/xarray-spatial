"""Tests for the TIFF reader."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from xrspatial.geotiff._reader import read_to_array, _read_strips, _read_tiles
from xrspatial.geotiff._header import parse_header, parse_all_ifds
from xrspatial.geotiff._dtypes import tiff_dtype_to_numpy
from xrspatial.geotiff._geotags import extract_geo_info
from .conftest import make_minimal_tiff


class TestReadStrips:
    def test_float32_sequential(self):
        """Read a simple float32 stripped TIFF and verify pixel values."""
        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        data = make_minimal_tiff(4, 4, np.dtype('float32'), pixel_data=expected)

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        arr = _read_strips(data, ifd, header, dtype)
        np.testing.assert_array_equal(arr, expected)

    def test_uint16(self):
        expected = np.arange(20, dtype=np.uint16).reshape(4, 5)
        data = make_minimal_tiff(5, 4, np.dtype('uint16'), pixel_data=expected)

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        arr = _read_strips(data, ifd, header, dtype)
        np.testing.assert_array_equal(arr, expected)

    def test_windowed_read(self):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        data = make_minimal_tiff(8, 8, np.dtype('float32'), pixel_data=expected)

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        window = (2, 3, 6, 7)  # rows 2-5, cols 3-6
        arr = _read_strips(data, ifd, header, dtype, window=window)
        np.testing.assert_array_equal(arr, expected[2:6, 3:7])


class TestReadTiles:
    def test_tiled_float32(self):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        data = make_minimal_tiff(
            8, 8, np.dtype('float32'),
            pixel_data=expected,
            tiled=True,
            tile_size=4,
        )

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        arr = _read_tiles(data, ifd, header, dtype)
        np.testing.assert_array_equal(arr, expected)

    def test_tiled_windowed(self):
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        data = make_minimal_tiff(
            8, 8, np.dtype('float32'),
            pixel_data=expected,
            tiled=True,
            tile_size=4,
        )

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        window = (1, 2, 5, 6)
        arr = _read_tiles(data, ifd, header, dtype, window=window)
        np.testing.assert_array_equal(arr, expected[1:5, 2:6])


class TestReadToArray:
    def test_local_file(self, tmp_path):
        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        tiff_data = make_minimal_tiff(4, 4, np.dtype('float32'), pixel_data=expected)
        path = str(tmp_path / 'test.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        arr, geo_info = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_geo_info(self, tmp_path):
        tiff_data = make_minimal_tiff(
            4, 4, np.dtype('float32'),
            geo_transform=(-120.0, 45.0, 0.001, -0.001),
            epsg=4326,
        )
        path = str(tmp_path / 'geo_test.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        arr, geo_info = read_to_array(path)
        assert geo_info.crs_epsg == 4326
        assert geo_info.transform.origin_x == pytest.approx(-120.0)
