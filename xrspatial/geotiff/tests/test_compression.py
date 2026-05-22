"""Tests for compression codecs."""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._compression import (COMPRESSION_DEFLATE, COMPRESSION_LZW, COMPRESSION_NONE,
                                            compress, decompress, deflate_compress,
                                            deflate_decompress, lzw_compress, lzw_decompress,
                                            predictor_decode, predictor_encode)


class TestDeflate:
    def test_round_trip(self):
        data = b'hello world! ' * 100
        compressed = deflate_compress(data)
        assert compressed != data
        assert deflate_decompress(compressed) == data

    def test_empty(self):
        compressed = deflate_compress(b'')
        assert deflate_decompress(compressed) == b''

    def test_binary_data(self):
        data = bytes(range(256)) * 10
        compressed = deflate_compress(data)
        assert deflate_decompress(compressed) == data


class TestLZW:
    def test_round_trip_simple(self):
        data = b'ABCABCABCABC'
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_round_trip_repetitive(self):
        data = b'\x00' * 1000
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_round_trip_sequential(self):
        data = bytes(range(256))
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_round_trip_random(self):
        rng = np.random.RandomState(42)
        data = bytes(rng.randint(0, 256, size=500, dtype=np.uint8))
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_round_trip_large(self):
        rng = np.random.RandomState(123)
        data = bytes(rng.randint(0, 256, size=10000, dtype=np.uint8))
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_empty(self):
        compressed = lzw_compress(b'')
        decompressed = lzw_decompress(compressed, 0)
        assert decompressed.tobytes() == b''


class TestPredictor:
    def test_round_trip_uint8(self):
        # 4x4 image, 1 byte per sample
        data = np.array([10, 20, 30, 40, 50, 60, 70, 80,
                         90, 100, 110, 120, 130, 140, 150, 160],
                        dtype=np.uint8)
        encoded = predictor_encode(data.copy(), 4, 4, 1)
        decoded = predictor_decode(encoded.copy(), 4, 4, 1)
        np.testing.assert_array_equal(decoded, data)

    def test_round_trip_float32(self):
        # 2x3 image, 4 bytes per sample
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        raw = np.frombuffer(arr.tobytes(), dtype=np.uint8).copy()
        encoded = predictor_encode(raw.copy(), 3, 2, 4)
        decoded = predictor_decode(encoded.copy(), 3, 2, 4)
        np.testing.assert_array_equal(decoded, raw)

    def test_predictor_encode_differences(self):
        # First pixel unchanged, rest are differences
        data = np.array([10, 20, 30, 40], dtype=np.uint8)
        encoded = predictor_encode(data.copy(), 4, 1, 1)
        assert encoded[0] == 10
        assert encoded[1] == 10  # 20 - 10
        assert encoded[2] == 10  # 30 - 20
        assert encoded[3] == 10  # 40 - 30


class TestDispatch:
    def test_none(self):
        data = b'hello'
        assert decompress(data, COMPRESSION_NONE).tobytes() == data
        assert compress(data, COMPRESSION_NONE) == data

    def test_deflate(self):
        data = b'test data ' * 50
        compressed = compress(data, COMPRESSION_DEFLATE)
        assert decompress(compressed, COMPRESSION_DEFLATE).tobytes() == data

    def test_lzw(self):
        data = b'ABCABC' * 20
        compressed = compress(data, COMPRESSION_LZW)
        decompressed = decompress(compressed, COMPRESSION_LZW, len(data))
        assert decompressed.tobytes() == data

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported compression"):
            decompress(b'', 99)
        with pytest.raises(ValueError, match="Unsupported compression"):
            compress(b'', 99)
