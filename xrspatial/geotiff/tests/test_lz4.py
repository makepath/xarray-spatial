"""Tests for LZ4 compression codec (#1051)."""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._compression import (COMPRESSION_LZ4, LZ4_AVAILABLE, compress, decompress,
                                            lz4_compress, lz4_decompress)

pytestmark = pytest.mark.skipif(
    not LZ4_AVAILABLE,
    reason="lz4 not installed",
)


class TestLZ4Codec:
    """CPU LZ4 codec roundtrip via python-lz4."""

    def test_roundtrip_simple(self):
        data = b'hello world! ' * 100
        compressed = lz4_compress(data)
        assert compressed != data
        assert lz4_decompress(compressed) == data

    def test_roundtrip_binary(self):
        data = bytes(range(256)) * 10
        compressed = lz4_compress(data)
        assert lz4_decompress(compressed) == data

    def test_roundtrip_empty(self):
        compressed = lz4_compress(b'')
        assert lz4_decompress(compressed) == b''

    def test_roundtrip_large(self):
        rng = np.random.RandomState(1051)
        data = bytes(rng.randint(0, 256, size=50000, dtype=np.uint8))
        compressed = lz4_compress(data)
        assert lz4_decompress(compressed) == data

    def test_dispatch_roundtrip(self):
        data = b'test data ' * 50
        compressed = compress(data, COMPRESSION_LZ4)
        result = decompress(compressed, COMPRESSION_LZ4)
        assert result.tobytes() == data


class TestLZ4WriteRoundTrip:
    """Write-read roundtrip using the TIFF writer with LZ4 compression."""

    def test_tiled_uint8(self, tmp_path):
        from xrspatial.geotiff._reader import read_to_array
        from xrspatial.geotiff._writer import write

        expected = np.arange(64, dtype=np.uint8).reshape(8, 8)
        path = str(tmp_path / 'lz4_1051_tiled_uint8.tif')
        write(expected, path, compression='lz4', tiled=True, tile_size=8)

        arr, geo = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)

    def test_tiled_float32(self, tmp_path):
        from xrspatial.geotiff._reader import read_to_array
        from xrspatial.geotiff._writer import write

        expected = np.random.RandomState(1051).rand(16, 16).astype(np.float32)
        path = str(tmp_path / 'lz4_1051_tiled_f32.tif')
        write(expected, path, compression='lz4', tiled=True, tile_size=8)

        arr, geo = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)

    def test_stripped_uint8(self, tmp_path):
        from xrspatial.geotiff._reader import read_to_array
        from xrspatial.geotiff._writer import write

        expected = np.arange(64, dtype=np.uint8).reshape(8, 8)
        path = str(tmp_path / 'lz4_1051_stripped.tif')
        write(expected, path, compression='lz4', tiled=False)

        arr, geo = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)

    def test_with_predictor(self, tmp_path):
        from xrspatial.geotiff._reader import read_to_array
        from xrspatial.geotiff._writer import write

        expected = np.random.RandomState(1051).rand(16, 16).astype(np.float32)
        path = str(tmp_path / 'lz4_1051_predictor.tif')
        write(expected, path, compression='lz4', tiled=True, tile_size=8,
              predictor=True)

        arr, geo = read_to_array(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(arr, expected)

    def test_public_api_roundtrip(self, tmp_path):
        import xarray as xr

        from xrspatial.geotiff import open_geotiff, to_geotiff

        data = np.arange(64, dtype=np.uint8).reshape(8, 8)
        da = xr.DataArray(data, dims=['y', 'x'],
                          coords={'y': np.arange(8), 'x': np.arange(8)},
                          attrs={'crs': 4326})
        path = str(tmp_path / 'lz4_1051_api.tif')
        # Tier 3 codec (issue #2137); pass the opt-in so the round-trip
        # test exercises the encode path rather than the rejection gate.
        to_geotiff(da, path, compression='lz4',
                   allow_experimental_codecs=True)

        result = open_geotiff(path, allow_experimental_codecs=True)
        np.testing.assert_array_equal(result.values, data)


class TestLZ4Availability:
    """Test availability flag and error handling (always runs)."""
    pytestmark = []

    def test_compression_constant(self):
        assert COMPRESSION_LZ4 == 50004

    def test_compression_tag_mapping(self):
        from xrspatial.geotiff._writer import _compression_tag
        assert _compression_tag('lz4') == 50004

    def test_unavailable_raises_import_error(self):
        import xrspatial.geotiff._compression as comp_mod
        orig = comp_mod.LZ4_AVAILABLE
        comp_mod.LZ4_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="lz4"):
                comp_mod.lz4_decompress(b'\x00')
            with pytest.raises(ImportError, match="lz4"):
                comp_mod.lz4_compress(b'\x00')
        finally:
            comp_mod.LZ4_AVAILABLE = orig
