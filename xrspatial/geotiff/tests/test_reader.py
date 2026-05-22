"""Tests for the TIFF reader."""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._dtypes import tiff_dtype_to_numpy
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._reader import _read_strips, _read_tiles, read_to_array

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


class TestPartialTileValidation_1486:
    """Issue #1486: corrupt tile/strip data should raise a clear error.

    Without validation a truncated deflate stream causes numpy.reshape to
    raise an opaque "cannot reshape array of size N" with no hint of which
    tile is at fault.  These tests pin the new behaviour: a clear ValueError
    naming the size mismatch.
    """

    def _zero_out_last_tile(self, path):
        """Replace the last tile's compressed bytes with zeros so deflate
        decodes a short stream."""
        from xrspatial.geotiff._header import parse_all_ifds, parse_header
        with open(path, 'rb') as f:
            data = bytearray(f.read())
        header = parse_header(bytes(data))
        ifds = parse_all_ifds(bytes(data), header)
        ifd = ifds[0]
        if ifd.tile_offsets is not None:
            offsets = ifd.tile_offsets
            counts = ifd.tile_byte_counts
        else:
            offsets = ifd.strip_offsets
            counts = ifd.strip_byte_counts
        last_off = offsets[-1]
        last_count = counts[-1]
        # Zero deflate stream: header 0x78 0x9C followed by an empty stored
        # block.  zlib will return 0 bytes -- a clear undersized result.
        zero_stream = b'\x78\x9c\x03\x00\x00\x00\x00\x01'
        # Pad with zeros to original length so file layout stays intact.
        padded = zero_stream + b'\x00' * max(0, last_count - len(zero_stream))
        for i, b in enumerate(padded[:last_count]):
            data[last_off + i] = b
        with open(path, 'wb') as f:
            f.write(bytes(data))

    def test_truncated_tile_raises_clear_error(self, tmp_path):
        from xrspatial.geotiff._writer import write

        pixels = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / 'truncated_1486.tif')
        write(pixels, path, compression='deflate', tiled=True, tile_size=8)

        self._zero_out_last_tile(path)

        with pytest.raises(ValueError) as exc:
            read_to_array(path)
        msg = str(exc.value)
        assert 'size mismatch' in msg
        assert 'expected' in msg
        assert 'truncated or corrupt' in msg

    def test_valid_edge_tile_still_works(self, tmp_path):
        """Edge tiles in a valid file decompress to full tile size; the
        validation should not flag this as corrupt."""
        from xrspatial.geotiff._writer import write

        # 9 x 9 with tile_size=4 -> a 3x3 tile grid where the right and
        # bottom tiles are partial.  These exercise the legitimate
        # "decompress full tile, slice top-left actual_h x actual_w" path.
        pixels = np.arange(81, dtype=np.float32).reshape(9, 9)
        path = str(tmp_path / 'edge_tiles_1486.tif')
        write(pixels, path, compression='deflate', tiled=True, tile_size=4)

        arr, _ = read_to_array(path)
        np.testing.assert_array_equal(arr, pixels)
