"""Tests for JPEG compression support (issue #1050)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff._compression import (COMPRESSION_JPEG, _splice_jpeg_tables, jpeg_compress,
                                            jpeg_decompress)
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import _compression_tag, write


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
        write(expected, path, compression='jpeg', tiled=True, tile_size=16,
              allow_internal_only_jpeg=True)

        arr, geo = read_to_array(path)
        assert arr.shape == expected.shape
        assert arr.dtype == np.uint8
        # JPEG is lossy, check approximate
        assert np.abs(arr.astype(int) - expected.astype(int)).mean() < 10

    def test_grayscale_stripped(self, tmp_path):
        rng = np.random.RandomState(1050)
        expected = rng.randint(50, 200, (32, 32), dtype=np.uint8)
        path = str(tmp_path / 'gray_1050_stripped.tif')
        write(expected, path, compression='jpeg', tiled=False,
              allow_internal_only_jpeg=True)

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
        write(expected, path, compression='jpeg', tiled=True, tile_size=16,
              allow_internal_only_jpeg=True)

        arr, geo = read_to_array(path)
        assert arr.shape == expected.shape
        assert np.abs(arr.astype(int) - expected.astype(int)).mean() < 10


class TestJpegValidation:
    """Verify that JPEG rejects invalid input."""

    def test_float_data_rejected(self, tmp_path):
        arr = np.zeros((8, 8), dtype=np.float32)
        path = str(tmp_path / 'bad_1050.tif')
        with pytest.raises(ValueError, match="uint8"):
            write(arr, path, compression='jpeg',
                  allow_internal_only_jpeg=True)

    def test_uint16_data_rejected(self, tmp_path):
        arr = np.zeros((8, 8), dtype=np.uint16)
        path = str(tmp_path / 'bad16_1050.tif')
        with pytest.raises(ValueError, match="uint8"):
            write(arr, path, compression='jpeg',
                  allow_internal_only_jpeg=True)

    def test_4band_rejected(self, tmp_path):
        arr = np.zeros((8, 8, 4), dtype=np.uint8)
        path = str(tmp_path / 'bad4b_1050.tif')
        with pytest.raises(ValueError, match="1 or 3 bands"):
            write(arr, path, compression='jpeg',
                  allow_internal_only_jpeg=True)


class TestWriteGeotiffJpeg:
    """Test the public to_geotiff API with compression='jpeg'."""

    def test_to_geotiff_jpeg_rejected(self, tmp_path):
        """to_geotiff refuses compression='jpeg'.

        The encoder writes self-contained JFIF streams without the
        TIFF-required JPEGTables tag (347), so libtiff / GDAL / rasterio
        cannot decode the file. Round-tripping internally via Pillow
        would mask the interop break, so the public writer rejects the
        codec until a JPEGTables-aware encoder is in place.
        """
        from xrspatial.geotiff import to_geotiff

        rng = np.random.RandomState(1050)
        data = rng.randint(50, 200, (32, 32), dtype=np.uint8)
        da = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.arange(32, dtype=float),
                    'x': np.arange(32, dtype=float)},
        )
        path = str(tmp_path / 'api_1050.tif')
        with pytest.raises(ValueError, match="JPEGTables"):
            to_geotiff(da, path, compression='jpeg', tile_size=16)


class TestJpegTablesSplice:
    """Verify the JPEGTables splice helper used for tiled JPEG TIFFs."""

    def test_splice_reconstructs_complete_jpeg(self):
        # Build a complete JPEG, then split it into a tables stream + a
        # tile fragment. Splicing should recover a decodable stream.
        import io

        from PIL import Image

        rng = np.random.RandomState(1502)
        arr = rng.randint(50, 200, (16, 16, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode='RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        full = buf.getvalue()

        # Find the SOS marker (FF DA): everything before is tables.
        sos = full.index(b'\xff\xda')
        tables = b'\xff\xd8' + full[2:sos] + b'\xff\xd9'
        tile_fragment = b'\xff\xd8' + full[sos:]

        spliced = _splice_jpeg_tables(tile_fragment, tables)
        decoded = Image.open(io.BytesIO(spliced))
        decoded.load()
        assert decoded.size == (16, 16)

    def test_splice_passthrough_on_empty_tables(self):
        payload = b'\xff\xd8\xff\xd9'
        assert _splice_jpeg_tables(payload, b'') == payload
        assert _splice_jpeg_tables(payload, None) == payload

    def test_splice_passthrough_on_invalid_input(self):
        # No SOI -> return unchanged so libjpeg's own error surfaces.
        assert _splice_jpeg_tables(b'no soi', b'\xff\xd8\xff\xd9') == b'no soi'

    def test_jpeg_decompress_accepts_jpeg_tables_kwarg(self):
        import io

        from PIL import Image

        rng = np.random.RandomState(1502)
        arr = rng.randint(50, 200, (16, 16, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode='RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        full = buf.getvalue()
        sos = full.index(b'\xff\xda')
        tables = b'\xff\xd8' + full[2:sos] + b'\xff\xd9'
        fragment = b'\xff\xd8' + full[sos:]

        out = jpeg_decompress(fragment, 16, 16, samples=3, jpeg_tables=tables)
        assert len(out) == 16 * 16 * 3


# rasterio-driven tests for issue #1502: GDAL writes tiled JPEG TIFFs
# whose per-tile fragments share DQT/DHT tables in tag 347. Skip the
# class -- not the whole module -- when rasterio is missing so the
# codec/splice unit tests above still run.


@pytest.mark.skipif(
    importlib.util.find_spec('rasterio') is None,
    reason='rasterio is required to write GDAL-style tiled JPEG TIFFs',
)
class TestGdalTiledJpegRead:
    """Read GDAL-style tiled JPEG TIFFs that use the JPEGTables tag."""

    def _gradient_rgb(self, size=128):
        # Smooth content keeps JPEG error low and detection of bugs easy.
        y = np.linspace(20, 240, size, dtype=np.uint8)
        x = np.linspace(20, 240, size, dtype=np.uint8)
        r = np.broadcast_to(y[:, None], (size, size)).astype(np.uint8)
        g = np.broadcast_to(x[None, :], (size, size)).astype(np.uint8)
        b = np.full((size, size), 128, dtype=np.uint8)
        return np.stack([r, g, b], axis=0)  # rasterio wants (bands, H, W)

    def test_tiled_ycbcr_jpeg(self, tmp_path):
        import rasterio as rio

        from xrspatial.geotiff._header import TAG_JPEG_TABLES, parse_all_ifds, parse_header

        size = 128
        data = self._gradient_rgb(size)
        path = str(tmp_path / 'tiled_jpeg_ycbcr_1502.tif')
        with rio.open(
            path, 'w', driver='GTiff', height=size, width=size, count=3,
            dtype='uint8', tiled=True, blockxsize=64, blockysize=64,
            compress='JPEG', photometric='YCBCR',
        ) as dst:
            dst.write(data)

        # Sanity: the file actually carries JPEGTables (tag 347).
        with open(path, 'rb') as f:
            blob = f.read()
        hdr = parse_header(blob)
        ifds = parse_all_ifds(blob, hdr)
        assert TAG_JPEG_TABLES in ifds[0].entries
        assert ifds[0].jpeg_tables is not None
        assert ifds[0].jpeg_tables[:2] == b'\xff\xd8'

        arr, _ = read_to_array(path)
        assert arr.shape == (size, size, 3)
        assert arr.dtype == np.uint8

        # Compare to rasterio's own decode. JPEG at quality 75 + 4:2:0
        # chroma subsampling shows ~1-3 absolute mean error on smooth
        # gradients; allow a generous 5.
        with rio.open(path) as src:
            ref = src.read()  # (bands, H, W)
        ref = np.transpose(ref, (1, 2, 0))
        assert np.abs(arr.astype(int) - ref.astype(int)).mean() < 5

    def test_tiled_grayscale_jpeg(self, tmp_path):
        import rasterio as rio

        size = 96
        y = np.linspace(20, 240, size, dtype=np.uint8)
        gray = np.broadcast_to(y[:, None], (size, size)).astype(np.uint8)

        path = str(tmp_path / 'tiled_jpeg_gray_1502.tif')
        with rio.open(
            path, 'w', driver='GTiff', height=size, width=size, count=1,
            dtype='uint8', tiled=True, blockxsize=32, blockysize=32,
            compress='JPEG',
        ) as dst:
            dst.write(gray, 1)

        arr, _ = read_to_array(path)
        assert arr.shape == (size, size)

        with rio.open(path) as src:
            ref = src.read(1)
        assert np.abs(arr.astype(int) - ref.astype(int)).mean() < 5
