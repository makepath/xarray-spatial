"""Tests for JPEG compression support (issue #1050)."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff._compression import (
    COMPRESSION_JPEG,
    jpeg_compress,
    jpeg_decompress,
)
from xrspatial.geotiff._writer import write, _compression_tag
from xrspatial.geotiff._reader import read_to_array


class TestJpegCodec:
    """Low-level JPEG compress/decompress round trips."""

    def test_grayscale_round_trip(self):
        rng = np.random.RandomState(1050)
        arr = rng.randint(0, 256, (32, 32), dtype=np.uint8)
        compressed = jpeg_compress(arr.tobytes(), 32, 32, samples=1)
        decoded = np.frombuffer(
            jpeg_decompress(compressed, 32, 32, samples=1), dtype=np.uint8
        ).reshape(32, 32)
        # JPEG is lossy: check approximate match
        assert decoded.shape == arr.shape
        assert np.abs(decoded.astype(int) - arr.astype(int)).mean() < 10

    def test_rgb_round_trip(self):
        # Use a smooth gradient -- random noise is the worst case for JPEG
        y = np.linspace(50, 200, 32, dtype=np.uint8)
        x = np.linspace(50, 200, 32, dtype=np.uint8)
        r = np.outer(y, np.ones(32, dtype=np.uint8))
        g = np.outer(np.ones(32, dtype=np.uint8), x)
        b = np.full((32, 32), 128, dtype=np.uint8)
        arr = np.stack([r, g, b], axis=2)
        compressed = jpeg_compress(arr.tobytes(), 32, 32, samples=3)
        decoded = np.frombuffer(
            jpeg_decompress(compressed, 32, 32, samples=3), dtype=np.uint8
        ).reshape(32, 32, 3)
        assert decoded.shape == arr.shape
        assert np.abs(decoded.astype(int) - arr.astype(int)).mean() < 10

    def test_quality_affects_size(self):
        rng = np.random.RandomState(1050)
        arr = rng.randint(0, 256, (32, 32), dtype=np.uint8)
        data = arr.tobytes()
        low_q = jpeg_compress(data, 32, 32, samples=1, quality=10)
        high_q = jpeg_compress(data, 32, 32, samples=1, quality=95)
        assert len(low_q) < len(high_q)

    def test_invalid_samples_raises(self):
        with pytest.raises(ValueError, match="1 or 3 bands"):
            jpeg_compress(b'\x00' * 64, 4, 4, samples=2)


class TestCompressionTagJpeg:
    """Verify JPEG is wired into the writer's compression tag map."""

    def test_jpeg_tag_value(self):
        assert _compression_tag('jpeg') == COMPRESSION_JPEG
        assert _compression_tag('JPEG') == COMPRESSION_JPEG

    def test_tag_value_is_7(self):
        assert COMPRESSION_JPEG == 7


class TestJpegWriteRoundTrip:
    """Write JPEG-compressed GeoTIFFs and read them back."""

    def test_grayscale_tiled(self, tmp_path):
        rng = np.random.RandomState(1050)
        expected = rng.randint(50, 200, (32, 32), dtype=np.uint8)
        path = str(tmp_path / 'gray_1050_tiled.tif')
        write(expected, path, compression='jpeg', tiled=True, tile_size=16)

        arr, geo = read_to_array(path)
        assert arr.shape == expected.shape
        assert arr.dtype == np.uint8
        # JPEG is lossy, check approximate
        assert np.abs(arr.astype(int) - expected.astype(int)).mean() < 10

    def test_grayscale_stripped(self, tmp_path):
        rng = np.random.RandomState(1050)
        expected = rng.randint(50, 200, (32, 32), dtype=np.uint8)
        path = str(tmp_path / 'gray_1050_stripped.tif')
        write(expected, path, compression='jpeg', tiled=False)

        arr, geo = read_to_array(path)
        assert arr.shape == expected.shape
        assert np.abs(arr.astype(int) - expected.astype(int)).mean() < 10

    def test_rgb_tiled(self, tmp_path):
        # Smooth gradient for predictable JPEG behavior
        y = np.linspace(50, 200, 32, dtype=np.uint8)
        x = np.linspace(50, 200, 32, dtype=np.uint8)
        r = np.outer(y, np.ones(32, dtype=np.uint8))
        g = np.outer(np.ones(32, dtype=np.uint8), x)
        b = np.full((32, 32), 128, dtype=np.uint8)
        expected = np.stack([r, g, b], axis=2)
        path = str(tmp_path / 'rgb_1050_tiled.tif')
        write(expected, path, compression='jpeg', tiled=True, tile_size=16)

        arr, geo = read_to_array(path)
        assert arr.shape == expected.shape
        assert np.abs(arr.astype(int) - expected.astype(int)).mean() < 10


class TestJpegValidation:
    """Verify that JPEG rejects invalid input."""

    def test_float_data_rejected(self, tmp_path):
        arr = np.zeros((8, 8), dtype=np.float32)
        path = str(tmp_path / 'bad_1050.tif')
        with pytest.raises(ValueError, match="uint8"):
            write(arr, path, compression='jpeg')

    def test_uint16_data_rejected(self, tmp_path):
        arr = np.zeros((8, 8), dtype=np.uint16)
        path = str(tmp_path / 'bad16_1050.tif')
        with pytest.raises(ValueError, match="uint8"):
            write(arr, path, compression='jpeg')

    def test_4band_rejected(self, tmp_path):
        arr = np.zeros((8, 8, 4), dtype=np.uint8)
        path = str(tmp_path / 'bad4b_1050.tif')
        with pytest.raises(ValueError, match="1 or 3 bands"):
            write(arr, path, compression='jpeg')


class TestWriteGeotiffJpeg:
    """Test the public write_geotiff API with compression='jpeg'."""

    def test_write_geotiff_jpeg(self, tmp_path):
        from xrspatial.geotiff import write_geotiff, read_geotiff

        rng = np.random.RandomState(1050)
        data = rng.randint(50, 200, (32, 32), dtype=np.uint8)
        da = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.arange(32, dtype=float),
                    'x': np.arange(32, dtype=float)},
        )
        path = str(tmp_path / 'api_1050.tif')
        write_geotiff(da, path, compression='jpeg', tile_size=16)

        result = read_geotiff(path)
        assert result.shape == (32, 32)
        assert np.abs(result.values.astype(int) - data.astype(int)).mean() < 10
