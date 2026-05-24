"""Tests for LERC compression codec (#1052)."""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._compression import (COMPRESSION_LERC, LERC_AVAILABLE, decompress,
                                            lerc_compress, lerc_decompress)

pytestmark = pytest.mark.skipif(
    not LERC_AVAILABLE,
    reason="lerc not installed",
)


class TestLERCCodec:
    """CPU LERC codec roundtrip."""

    def test_roundtrip_float32_lossless(self):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        compressed = lerc_compress(
            arr.tobytes(), 8, 8, samples=1, dtype=np.dtype('float32'),
            max_z_error=0.0)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        decompressed = lerc_decompress(compressed, 8, 8, 1)
        result = np.frombuffer(decompressed, dtype=np.float32).reshape(8, 8)
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_uint8_lossless(self):
        arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
        compressed = lerc_compress(
            arr.tobytes(), 8, 8, samples=1, dtype=np.dtype('uint8'),
            max_z_error=0.0)
        decompressed = lerc_decompress(compressed, 8, 8, 1)
        result = np.frombuffer(decompressed, dtype=np.uint8).reshape(8, 8)
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_uint16_lossless(self):
        arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
        compressed = lerc_compress(
            arr.tobytes(), 8, 8, samples=1, dtype=np.dtype('uint16'),
            max_z_error=0.0)
        decompressed = lerc_decompress(compressed, 8, 8, 1)
        result = np.frombuffer(decompressed, dtype=np.uint16).reshape(8, 8)
        np.testing.assert_array_equal(result, arr)

    def test_lossy_within_tolerance(self):
        rng = np.random.RandomState(1052)
        arr = rng.rand(32, 32).astype(np.float32) * 100
        max_err = 0.5
        compressed = lerc_compress(
            arr.tobytes(), 32, 32, samples=1, dtype=np.dtype('float32'),
            max_z_error=max_err)
        decompressed = lerc_decompress(compressed, 32, 32, 1)
        result = np.frombuffer(decompressed, dtype=np.float32).reshape(32, 32)
        np.testing.assert_array_less(np.abs(result - arr), max_err + 1e-6)

    def test_lossy_smaller_than_lossless(self):
        rng = np.random.RandomState(1052)
        arr = rng.rand(64, 64).astype(np.float32) * 1000
        lossless = lerc_compress(
            arr.tobytes(), 64, 64, samples=1, dtype=np.dtype('float32'),
            max_z_error=0.0)
        lossy = lerc_compress(
            arr.tobytes(), 64, 64, samples=1, dtype=np.dtype('float32'),
            max_z_error=1.0)
        assert len(lossy) <= len(lossless)

    def test_dispatch_decompress(self):
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        compressed = lerc_compress(
            arr.tobytes(), 4, 4, samples=1, dtype=np.dtype('float32'),
            max_z_error=0.0)
        result = decompress(compressed, COMPRESSION_LERC,
                            width=4, height=4, samples=1)
        # decompress returns uint8, reinterpret as float32
        result_f = np.frombuffer(result.tobytes(), dtype=np.float32).reshape(4, 4)
        np.testing.assert_array_equal(result_f, arr)


class TestLERCWriteRoundTrip:
    """Write-read roundtrip using the TIFF writer with LERC compression."""

    def test_tiled_float32(self, tmp_path):
        from xrspatial.geotiff._reader import read_to_array
        from xrspatial.geotiff._writer import write

        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'lerc_1052_tiled_f32.tif')
        write(expected, path, compression='lerc', tiled=True, tile_size=8)

        arr, geo = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)

    def test_tiled_uint8(self, tmp_path):
        from xrspatial.geotiff._reader import read_to_array
        from xrspatial.geotiff._writer import write

        expected = np.arange(64, dtype=np.uint8).reshape(8, 8)
        path = str(tmp_path / 'lerc_1052_tiled_u8.tif')
        write(expected, path, compression='lerc', tiled=True, tile_size=8)

        arr, geo = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)

    def test_stripped_float32(self, tmp_path):
        from xrspatial.geotiff._reader import read_to_array
        from xrspatial.geotiff._writer import write

        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'lerc_1052_stripped.tif')
        write(expected, path, compression='lerc', tiled=False)

        arr, geo = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)

    def test_public_api_roundtrip(self, tmp_path):
        import xarray as xr

        from xrspatial.geotiff import open_geotiff, to_geotiff

        data = np.arange(64, dtype=np.float32).reshape(8, 8)
        da = xr.DataArray(data, dims=['y', 'x'],
                          coords={'y': np.arange(8), 'x': np.arange(8)},
                          attrs={'crs': 4326})
        path = str(tmp_path / 'lerc_1052_api.tif')
        # Tier 3 codec (issue #2137); pass the opt-in so the round-trip
        # test exercises the encode path rather than the rejection gate.
        to_geotiff(da, path, compression='lerc',
                   allow_experimental_codecs=True)

        # PR 4 of epic #2340: the read side also gates the LERC codec
        # on ``allow_experimental_codecs=True`` so the open here passes
        # the same opt-in the writer required.
        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, data)


class TestLERCAvailability:
    """Test availability flag and error handling (always runs)."""
    pytestmark = []

    def test_compression_constant(self):
        assert COMPRESSION_LERC == 34887

    def test_compression_tag_mapping(self):
        from xrspatial.geotiff._writer import _compression_tag
        assert _compression_tag('lerc') == 34887

    def test_unavailable_raises_import_error(self):
        import xrspatial.geotiff._compression as comp_mod
        orig = comp_mod.LERC_AVAILABLE
        comp_mod.LERC_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="lerc"):
                comp_mod.lerc_decompress(b'\x00')
            with pytest.raises(ImportError, match="lerc"):
                comp_mod.lerc_compress(b'\x00', 1, 1)
        finally:
            comp_mod.LERC_AVAILABLE = orig
