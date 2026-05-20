"""Tests for JPEG 2000 compression codec (#1048)."""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._compression import (
    COMPRESSION_JPEG2000,
    JPEG2000_AVAILABLE,
    jpeg2000_compress,
    jpeg2000_decompress,
    decompress,
)

pytestmark = pytest.mark.skipif(
    not JPEG2000_AVAILABLE,
    reason="glymur not installed",
)


class TestJPEG2000Codec:
    """CPU JPEG 2000 codec roundtrip via glymur."""

    def test_roundtrip_uint8(self):
        arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
        compressed = jpeg2000_compress(
            arr.tobytes(), 8, 8, samples=1, dtype=np.dtype('uint8'),
            lossless=True)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        decompressed = jpeg2000_decompress(compressed, 8, 8, 1)
        result = np.frombuffer(decompressed, dtype=np.uint8).reshape(8, 8)
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_uint16(self):
        arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
        compressed = jpeg2000_compress(
            arr.tobytes(), 8, 8, samples=1, dtype=np.dtype('uint16'),
            lossless=True)
        decompressed = jpeg2000_decompress(compressed, 8, 8, 1)
        result = np.frombuffer(decompressed, dtype=np.uint16).reshape(8, 8)
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_multiband(self):
        arr = np.arange(192, dtype=np.uint8).reshape(8, 8, 3)
        compressed = jpeg2000_compress(
            arr.tobytes(), 8, 8, samples=3, dtype=np.dtype('uint8'),
            lossless=True)
        decompressed = jpeg2000_decompress(compressed, 8, 8, 3)
        result = np.frombuffer(decompressed, dtype=np.uint8).reshape(8, 8, 3)
        np.testing.assert_array_equal(result, arr)

    def test_single_pixel(self):
        arr = np.array([[42]], dtype=np.uint8)
        compressed = jpeg2000_compress(
            arr.tobytes(), 1, 1, samples=1, dtype=np.dtype('uint8'),
            lossless=True)
        decompressed = jpeg2000_decompress(compressed, 1, 1, 1)
        result = np.frombuffer(decompressed, dtype=np.uint8)
        assert result[0] == 42

    def test_lossy_produces_smaller_output(self):
        rng = np.random.RandomState(1048)
        arr = rng.randint(0, 256, size=(64, 64), dtype=np.uint8)
        lossless = jpeg2000_compress(
            arr.tobytes(), 64, 64, samples=1, dtype=np.dtype('uint8'),
            lossless=True)
        lossy = jpeg2000_compress(
            arr.tobytes(), 64, 64, samples=1, dtype=np.dtype('uint8'),
            lossless=False)
        # Lossy should generally be smaller
        assert len(lossy) <= len(lossless)

    def test_dispatch_decompress(self):
        arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
        compressed = jpeg2000_compress(
            arr.tobytes(), 4, 4, samples=1, dtype=np.dtype('uint8'),
            lossless=True)
        result = decompress(compressed, COMPRESSION_JPEG2000,
                            width=4, height=4, samples=1)
        np.testing.assert_array_equal(
            result.reshape(4, 4),
            arr,
        )


class TestJPEG2000WriteRoundTrip:
    """Write-read roundtrip using the TIFF writer with JPEG 2000 compression."""

    def test_tiled_uint8(self, tmp_path):
        from xrspatial.geotiff._writer import write
        from xrspatial.geotiff._reader import read_to_array

        expected = np.arange(64, dtype=np.uint8).reshape(8, 8)
        path = str(tmp_path / 'j2k_1048_tiled_uint8.tif')
        write(expected, path, compression='jpeg2000', tiled=True, tile_size=8)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_tiled_uint16(self, tmp_path):
        from xrspatial.geotiff._writer import write
        from xrspatial.geotiff._reader import read_to_array

        expected = np.arange(64, dtype=np.uint16).reshape(8, 8)
        path = str(tmp_path / 'j2k_1048_tiled_uint16.tif')
        write(expected, path, compression='jpeg2000', tiled=True, tile_size=8)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_stripped_uint8(self, tmp_path):
        from xrspatial.geotiff._writer import write
        from xrspatial.geotiff._reader import read_to_array

        expected = np.arange(64, dtype=np.uint8).reshape(8, 8)
        path = str(tmp_path / 'j2k_1048_stripped.tif')
        write(expected, path, compression='jpeg2000', tiled=False)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_with_geo_info(self, tmp_path):
        from xrspatial.geotiff._writer import write
        from xrspatial.geotiff._reader import read_to_array
        from xrspatial.geotiff._geotags import GeoTransform

        expected = np.ones((8, 8), dtype=np.uint8) * 100
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'j2k_1048_geo.tif')
        write(expected, path, compression='jpeg2000', tiled=True, tile_size=8,
              geo_transform=gt, crs_epsg=4326, nodata=0)

        arr, geo = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)
        assert geo.crs_epsg == 4326

    def test_public_api_roundtrip(self, tmp_path):
        """Test via open_geotiff / to_geotiff public API."""
        import xarray as xr
        from xrspatial.geotiff import open_geotiff, to_geotiff

        data = np.arange(64, dtype=np.uint8).reshape(8, 8)
        da = xr.DataArray(data, dims=['y', 'x'],
                          coords={'y': np.arange(8), 'x': np.arange(8)},
                          attrs={'crs': 4326})
        path = str(tmp_path / 'j2k_1048_api.tif')
        # Tier 3 codec (issue #2137); pass the opt-in so the round-trip
        # test exercises the encode path rather than the rejection gate.
        to_geotiff(da, path, compression='jpeg2000',
                   allow_experimental_codecs=True)

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, data)


class TestJPEG2000Availability:
    """Test the availability flag and error handling.

    These don't need glymur, so they always run.
    """

    # Override the module-level skip for this class
    pytestmark = []

    def test_compression_constant(self):
        assert COMPRESSION_JPEG2000 == 34712

    def test_compression_tag_mapping(self):
        from xrspatial.geotiff._writer import _compression_tag
        assert _compression_tag('jpeg2000') == 34712
        assert _compression_tag('j2k') == 34712

    def test_unavailable_raises_import_error(self):
        """If glymur is missing, codec functions raise ImportError."""
        import unittest.mock
        import importlib
        import xrspatial.geotiff._compression as comp_mod
        # Temporarily pretend glymur is unavailable
        orig = comp_mod.JPEG2000_AVAILABLE
        comp_mod.JPEG2000_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="glymur"):
                comp_mod.jpeg2000_decompress(b'\x00', 1, 1, 1)
            with pytest.raises(ImportError, match="glymur"):
                comp_mod.jpeg2000_compress(b'\x00', 1, 1, dtype=np.dtype('uint8'))
        finally:
            comp_mod.JPEG2000_AVAILABLE = orig
