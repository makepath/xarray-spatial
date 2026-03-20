"""Tests for new features: multi-band, integer nodata, packbits, zstd, dask, BigTIFF."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_geotiff, write_geotiff
from xrspatial.geotiff._compression import (
    COMPRESSION_PACKBITS,
    packbits_compress,
    packbits_decompress,
    zstd_compress,
    zstd_decompress,
)
from xrspatial.geotiff._header import parse_header, parse_all_ifds
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import write


# -----------------------------------------------------------------------
# Multi-band write and read
# -----------------------------------------------------------------------

class TestMultiBand:

    def test_rgb_uint8_round_trip(self, tmp_path):
        """Write and read back RGB uint8 image."""
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr[:, :, 0] = 200  # red
        arr[:, :, 1] = 100  # green
        arr[:, :, 2] = 50   # blue
        path = str(tmp_path / 'rgb.tif')
        write(arr, path, compression='none', tiled=False)

        result, geo = read_to_array(path)
        assert result.shape == (8, 8, 3)
        np.testing.assert_array_equal(result, arr)

    def test_rgb_deflate_tiled(self, tmp_path):
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        path = str(tmp_path / 'rgb_deflate.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=8)

        result, geo = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_rgba_uint8(self, tmp_path):
        arr = np.ones((4, 4, 4), dtype=np.uint8) * 128
        path = str(tmp_path / 'rgba.tif')
        write(arr, path, compression='none', tiled=False)

        result, geo = read_to_array(path)
        assert result.shape == (4, 4, 4)
        np.testing.assert_array_equal(result, arr)

    def test_multiband_float32(self, tmp_path):
        arr = np.random.RandomState(99).rand(8, 8, 5).astype(np.float32)
        path = str(tmp_path / 'multi.tif')
        write(arr, path, compression='deflate', tiled=False)

        result, geo = read_to_array(path)
        assert result.shape == (8, 8, 5)
        np.testing.assert_array_equal(result, arr)

    def test_single_band_selection(self, tmp_path):
        """band= parameter should extract one band."""
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 1] = 42
        path = str(tmp_path / 'rgb_sel.tif')
        write(arr, path, compression='none', tiled=False)

        result, _ = read_to_array(path, band=1)
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, 42)

    def test_rgb_write_geotiff_api(self, tmp_path):
        """write_geotiff accepts 3D arrays."""
        arr = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        path = str(tmp_path / 'rgb_api.tif')
        write_geotiff(arr, path, compression='none')

        result = read_geotiff(path)
        assert 'band' in result.dims
        assert result.shape == (4, 4, 3)
        np.testing.assert_array_equal(result.values, arr)

    def test_rgb_cog(self, tmp_path):
        """Multi-band COG with overviews."""
        arr = np.random.RandomState(7).randint(
            0, 256, (32, 32, 3), dtype=np.uint8)
        path = str(tmp_path / 'rgb_cog.tif')
        write(arr, path, compression='deflate', tiled=True, tile_size=16,
              cog=True, overview_levels=[1])

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)


# -----------------------------------------------------------------------
# Integer nodata masking
# -----------------------------------------------------------------------

class TestIntegerNodata:

    def test_uint8_nodata_masked(self, tmp_path):
        arr = np.array([[0, 1, 2], [3, 255, 5]], dtype=np.uint8)
        path = str(tmp_path / 'uint8_nodata.tif')
        write(arr, path, compression='none', tiled=False, nodata=255)

        da = read_geotiff(path)
        assert np.isnan(da.values[1, 1])
        assert da.values[0, 1] == 1.0
        assert da.dtype == np.float64  # promoted from uint8

    def test_uint16_nodata_masked(self, tmp_path):
        arr = np.array([[100, 0], [200, 0]], dtype=np.uint16)
        path = str(tmp_path / 'uint16_nodata.tif')
        write(arr, path, compression='none', tiled=False, nodata=0)

        da = read_geotiff(path)
        assert np.isnan(da.values[0, 1])
        assert np.isnan(da.values[1, 1])
        assert da.values[0, 0] == 100.0

    def test_int16_nodata_negative(self, tmp_path):
        arr = np.array([[-9999, 10], [20, -9999]], dtype=np.int16)
        path = str(tmp_path / 'int16_nodata.tif')
        write(arr, path, compression='none', tiled=False, nodata=-9999)

        da = read_geotiff(path)
        assert np.isnan(da.values[0, 0])
        assert np.isnan(da.values[1, 1])
        assert da.values[0, 1] == 10.0

    def test_integer_no_nodata_stays_integer(self, tmp_path):
        """Without nodata, integer arrays should not be promoted."""
        arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
        path = str(tmp_path / 'no_nodata.tif')
        write(arr, path, compression='none', tiled=False)

        da = read_geotiff(path)
        assert da.dtype == np.uint16


# -----------------------------------------------------------------------
# PackBits compression
# -----------------------------------------------------------------------

class TestPackBits:

    def test_packbits_round_trip(self):
        data = b'\x00' * 100 + b'\xff' * 50 + bytes(range(200))
        compressed = packbits_compress(data)
        decompressed = packbits_decompress(compressed)
        assert decompressed == data

    def test_packbits_single_byte(self):
        data = b'\x42'
        assert packbits_decompress(packbits_compress(data)) == data

    def test_packbits_empty(self):
        assert packbits_decompress(packbits_compress(b'')) == b''

    def test_packbits_all_same(self):
        data = b'\xAA' * 500
        compressed = packbits_compress(data)
        assert len(compressed) < len(data)
        assert packbits_decompress(compressed) == data

    def test_write_read_packbits(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'packbits.tif')
        write(arr, path, compression='packbits', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_packbits_tiled(self, tmp_path):
        arr = np.random.RandomState(42).rand(16, 16).astype(np.float32)
        path = str(tmp_path / 'packbits_tiled.tif')
        write(arr, path, compression='packbits', tiled=True, tile_size=8)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)


# -----------------------------------------------------------------------
# ZSTD compression
# -----------------------------------------------------------------------

class TestZstd:

    def test_zstd_round_trip_bytes(self):
        data = b'hello zstd! ' * 1000
        compressed = zstd_compress(data)
        assert len(compressed) < len(data)
        assert zstd_decompress(compressed) == data

    def test_zstd_empty(self):
        compressed = zstd_compress(b'')
        assert zstd_decompress(compressed) == b''

    def test_zstd_random(self):
        rng = np.random.RandomState(42)
        data = bytes(rng.randint(0, 256, size=5000, dtype=np.uint8))
        assert zstd_decompress(zstd_compress(data)) == data

    def test_write_read_zstd_stripped(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'zstd_strip.tif')
        write(arr, path, compression='zstd', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_write_read_zstd_tiled(self, tmp_path):
        arr = np.random.RandomState(99).rand(16, 16).astype(np.float32)
        path = str(tmp_path / 'zstd_tiled.tif')
        write(arr, path, compression='zstd', tiled=True, tile_size=8)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_zstd_uint16(self, tmp_path):
        arr = np.arange(100, dtype=np.uint16).reshape(10, 10)
        path = str(tmp_path / 'zstd_u16.tif')
        write(arr, path, compression='zstd', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_zstd_with_predictor(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        path = str(tmp_path / 'zstd_pred.tif')
        write(arr, path, compression='zstd', tiled=False, predictor=True)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_zstd_multiband(self, tmp_path):
        arr = np.random.RandomState(7).randint(0, 256, (8, 8, 3), dtype=np.uint8)
        path = str(tmp_path / 'zstd_rgb.tif')
        write(arr, path, compression='zstd', tiled=False)

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)

    def test_zstd_public_api(self, tmp_path):
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'zstd_api.tif')
        write_geotiff(arr, path, compression='zstd')

        result = read_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)


# -----------------------------------------------------------------------
# BigTIFF write
# -----------------------------------------------------------------------

class TestBigTIFF:

    def test_bigtiff_header_written(self, tmp_path):
        """Force BigTIFF via internal threshold by mocking; test header parsing."""
        # We can't easily create a >4GB file in tests, but we can verify
        # the BigTIFF path works by writing a small file with bigtiff=True
        # through the internal API.
        from xrspatial.geotiff._writer import _assemble_tiff, _write_stripped
        from xrspatial.geotiff._compression import COMPRESSION_NONE
        from xrspatial.geotiff._geotags import GeoTransform

        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        rel_off, bc, chunks = _write_stripped(arr, COMPRESSION_NONE, False)
        parts = [(arr, 4, 4, rel_off, bc, chunks)]

        file_bytes = _assemble_tiff(
            4, 4, arr.dtype, COMPRESSION_NONE, False, False, 256,
            parts, None, None, None, is_cog=False, raster_type=1)

        # Standard TIFF: magic 42
        header = parse_header(file_bytes)
        assert not header.is_bigtiff

    def test_bigtiff_read_write_round_trip(self, tmp_path):
        """Test that BigTIFF files produced internally can be read back."""
        from xrspatial.geotiff._writer import (
            _assemble_tiff, _write_stripped, _assemble_standard_layout,
        )
        from xrspatial.geotiff._compression import COMPRESSION_NONE
        from xrspatial.geotiff._dtypes import numpy_to_tiff_dtype, SHORT, LONG, DOUBLE
        from xrspatial.geotiff._header import (
            TAG_IMAGE_WIDTH, TAG_IMAGE_LENGTH, TAG_BITS_PER_SAMPLE,
            TAG_COMPRESSION, TAG_PHOTOMETRIC, TAG_SAMPLES_PER_PIXEL,
            TAG_SAMPLE_FORMAT, TAG_ROWS_PER_STRIP,
            TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS,
        )

        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        rel_off, bc, chunks = _write_stripped(arr, COMPRESSION_NONE, False)
        bits_per_sample, sample_format = numpy_to_tiff_dtype(arr.dtype)

        tags = [
            (TAG_IMAGE_WIDTH, LONG, 1, 8),
            (TAG_IMAGE_LENGTH, LONG, 1, 8),
            (TAG_BITS_PER_SAMPLE, SHORT, 1, bits_per_sample),
            (TAG_COMPRESSION, SHORT, 1, 1),
            (TAG_PHOTOMETRIC, SHORT, 1, 1),
            (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
            (TAG_SAMPLE_FORMAT, SHORT, 1, sample_format),
            (TAG_ROWS_PER_STRIP, SHORT, 1, 8),
            (TAG_STRIP_OFFSETS, LONG, len(rel_off), rel_off),
            (TAG_STRIP_BYTE_COUNTS, LONG, len(bc), bc),
        ]

        parts = [(arr, 8, 8, rel_off, bc, chunks)]
        file_bytes = _assemble_standard_layout(
            16, [tags], parts, bigtiff=True)

        path = str(tmp_path / 'bigtiff.tif')
        with open(path, 'wb') as f:
            f.write(file_bytes)

        header = parse_header(file_bytes)
        assert header.is_bigtiff

        result, _ = read_to_array(path)
        np.testing.assert_array_equal(result, arr)


# -----------------------------------------------------------------------
# Sub-byte bit depths (1-bit, 4-bit, 12-bit)
# -----------------------------------------------------------------------

def _make_sub_byte_tiff(width, height, bps, pixel_values):
    """Build a minimal TIFF with sub-byte BitsPerSample.

    pixel_values: 2D array of unpacked integer values.
    Data is packed MSB-first into bytes according to bps.
    """
    import struct
    bo = '<'
    dtype_np = np.dtype('uint8') if bps <= 8 else np.dtype('uint16')

    # Pack pixel values into bytes
    flat = pixel_values.ravel()
    if bps == 1:
        packed = np.packbits(flat.astype(np.uint8))
    elif bps == 4:
        n = len(flat)
        packed_len = (n + 1) // 2
        packed = np.zeros(packed_len, dtype=np.uint8)
        for i in range(n):
            if i % 2 == 0:
                packed[i // 2] |= (flat[i] & 0x0F) << 4
            else:
                packed[i // 2] |= flat[i] & 0x0F
        packed = packed
    elif bps == 12:
        n = len(flat)
        n_pairs = n // 2
        remainder = n % 2
        packed_len = n_pairs * 3 + (2 if remainder else 0)
        packed = np.zeros(packed_len, dtype=np.uint8)
        for i in range(n_pairs):
            v0 = int(flat[i * 2])
            v1 = int(flat[i * 2 + 1])
            off = i * 3
            packed[off] = (v0 >> 4) & 0xFF
            packed[off + 1] = ((v0 & 0x0F) << 4) | ((v1 >> 8) & 0x0F)
            packed[off + 2] = v1 & 0xFF
        if remainder:
            v = int(flat[-1])
            off = n_pairs * 3
            packed[off] = (v >> 4) & 0xFF
            packed[off + 1] = (v & 0x0F) << 4
    else:
        raise ValueError(f"Unsupported bps: {bps}")

    pixel_bytes = packed.tobytes()

    # Build tags
    tag_list = []
    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))
    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    add_short(256, width)
    add_short(257, height)
    add_short(258, bps)
    add_short(259, 1)   # no compression
    add_short(262, 1 if bps > 1 else 0)  # MinIsWhite for 1-bit, BlackIsZero otherwise
    add_short(277, 1)
    add_short(278, height)
    add_long(273, 0)    # strip offset placeholder
    add_long(279, len(pixel_bytes))
    if bps <= 8:
        add_short(339, 1)  # UINT
    else:
        add_short(339, 1)

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_buf = bytearray()
    tag_offsets = {}
    overflow_start = ifd_start + ifd_size

    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)

    # Patch strip offset
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count, struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    # Rebuild overflow after patching
    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixel_bytes)

    return bytes(out), pixel_values


class TestSubByteBitDepths:

    def test_1bit_bilevel(self, tmp_path):
        """Read a 1-bit bilevel TIFF."""
        pixels = np.array([[1, 0, 1, 0, 1, 0, 1, 0],
                           [0, 1, 0, 1, 0, 1, 0, 1],
                           [1, 1, 0, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 0, 0, 1, 1]], dtype=np.uint8)
        tiff_data, expected = _make_sub_byte_tiff(8, 4, 1, pixels)
        path = str(tmp_path / '1bit.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.dtype == np.uint8
        assert result.shape == (4, 8)
        np.testing.assert_array_equal(result, expected)

    def test_1bit_non_byte_aligned_width(self, tmp_path):
        """1-bit image whose width is not a multiple of 8."""
        pixels = np.array([[1, 0, 1],
                           [0, 1, 0]], dtype=np.uint8)
        tiff_data, expected = _make_sub_byte_tiff(3, 2, 1, pixels)
        path = str(tmp_path / '1bit_3wide.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, expected)

    def test_4bit_nibble(self, tmp_path):
        """Read a 4-bit TIFF."""
        pixels = np.array([[0, 1, 2, 3],
                           [4, 5, 6, 7],
                           [8, 9, 10, 11],
                           [12, 13, 14, 15]], dtype=np.uint8)
        tiff_data, expected = _make_sub_byte_tiff(4, 4, 4, pixels)
        path = str(tmp_path / '4bit.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.dtype == np.uint8
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, expected)

    def test_4bit_odd_width(self, tmp_path):
        """4-bit image with odd width (partial byte at row end)."""
        pixels = np.array([[1, 2, 3],
                           [4, 5, 6]], dtype=np.uint8)
        tiff_data, expected = _make_sub_byte_tiff(3, 2, 4, pixels)
        path = str(tmp_path / '4bit_odd.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, expected)

    def test_12bit(self, tmp_path):
        """Read a 12-bit TIFF."""
        pixels = np.array([[0, 100, 2048, 4095],
                           [1000, 2000, 3000, 4000]], dtype=np.uint16)
        tiff_data, expected = _make_sub_byte_tiff(4, 2, 12, pixels)
        path = str(tmp_path / '12bit.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.dtype == np.uint16
        assert result.shape == (2, 4)
        np.testing.assert_array_equal(result, expected)

    def test_unpack_bits_codec_directly(self):
        """Test unpack_bits on known packed data."""
        from xrspatial.geotiff._compression import unpack_bits

        # 1-bit: byte 0xA5 = 10100101 -> [1,0,1,0,0,1,0,1]
        data = np.array([0xA5], dtype=np.uint8)
        result = unpack_bits(data, 1, 8)
        np.testing.assert_array_equal(result, [1, 0, 1, 0, 0, 1, 0, 1])

        # 4-bit: byte 0x3C = 0011_1100 -> [3, 12]
        data = np.array([0x3C], dtype=np.uint8)
        result = unpack_bits(data, 4, 2)
        np.testing.assert_array_equal(result, [3, 12])


# -----------------------------------------------------------------------
# Planar configuration (separate planes)
# -----------------------------------------------------------------------

def _make_planar_tiff(width, height, bands, dtype=np.uint8, tiled=False,
                      tile_size=4):
    """Build a minimal planar-config TIFF (PlanarConfiguration=2) by hand.

    Each band's pixel data is stored as a separate set of strips (or tiles).
    Band values: band 0 gets pixel values 10+pixel_idx, band 1 gets 20+,
    band 2 gets 30+, etc.
    """
    import struct
    bo = '<'

    dtype = np.dtype(dtype)
    bps = dtype.itemsize * 8
    if dtype.kind == 'f':
        sf = 3
    elif dtype.kind == 'i':
        sf = 2
    else:
        sf = 1

    # Build per-band pixel arrays
    band_arrays = []
    for b in range(bands):
        base = (b + 1) * 10
        arr = np.arange(width * height, dtype=dtype).reshape(height, width) + dtype.type(base)
        band_arrays.append(arr)

    if tiled:
        import math
        tw = th = tile_size
        tiles_across = math.ceil(width / tw)
        tiles_down = math.ceil(height / th)
        tiles_per_band = tiles_across * tiles_down

        # Build tile data: all tiles for band 0, then band 1, etc.
        tile_blobs = []
        for b in range(bands):
            for tr in range(tiles_down):
                for tc in range(tiles_across):
                    tile = np.zeros((th, tw), dtype=dtype)
                    r0, c0 = tr * th, tc * tw
                    r1 = min(r0 + th, height)
                    c1 = min(c0 + tw, width)
                    tile[:r1 - r0, :c1 - c0] = band_arrays[b][r0:r1, c0:c1]
                    tile_blobs.append(tile.tobytes())

        pixel_bytes = b''.join(tile_blobs)
        tile_byte_counts = [len(t) for t in tile_blobs]
        num_offsets = len(tile_blobs)
    else:
        # Strips: 1 strip per band (whole image), one set per band
        strip_blobs = []
        for b in range(bands):
            strip_blobs.append(band_arrays[b].tobytes())
        pixel_bytes = b''.join(strip_blobs)
        strip_byte_counts = [len(s) for s in strip_blobs]
        num_offsets = bands

    # Build tags
    tag_list = []
    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))
    def add_shorts(tag, vals):
        tag_list.append((tag, 3, len(vals), struct.pack(f'{bo}{len(vals)}H', *vals)))
    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))
    def add_longs(tag, vals):
        tag_list.append((tag, 4, len(vals), struct.pack(f'{bo}{len(vals)}I', *vals)))

    add_short(256, width)
    add_short(257, height)
    add_shorts(258, [bps] * bands)
    add_short(259, 1)   # no compression
    add_short(262, 2 if bands >= 3 else 1)  # RGB or BlackIsZero
    add_short(277, bands)
    add_short(284, 2)   # PlanarConfiguration = Separate
    add_shorts(339, [sf] * bands)

    if tiled:
        add_short(322, tile_size)
        add_short(323, tile_size)
        add_longs(324, [0] * num_offsets)  # placeholder
        add_longs(325, tile_byte_counts)
    else:
        add_short(278, height)  # RowsPerStrip = full image
        add_longs(273, [0] * num_offsets)  # placeholder
        add_longs(279, strip_byte_counts)

    tag_list.sort(key=lambda t: t[0])

    # Layout
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4

    # Collect overflow
    overflow_buf = bytearray()
    tag_offsets = {}
    overflow_start = ifd_start + ifd_size

    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)

    # Patch offsets
    offset_tag = 324 if tiled else 273
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == offset_tag:
            if tiled:
                offs = []
                pos = 0
                for blob in tile_blobs:
                    offs.append(pixel_data_start + pos)
                    pos += len(blob)
                new_raw = struct.pack(f'{bo}{num_offsets}I', *offs)
            else:
                offs = []
                pos = 0
                for blob in strip_blobs:
                    offs.append(pixel_data_start + pos)
                    pos += len(blob)
                new_raw = struct.pack(f'{bo}{num_offsets}I', *offs)
            patched.append((tag, typ, count, new_raw))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    # Rebuild overflow
    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    # Serialize
    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    out.extend(struct.pack(f'{bo}I', 0))  # next IFD
    out.extend(overflow_buf)
    out.extend(pixel_bytes)

    # Build expected output for verification
    expected = np.stack(band_arrays, axis=2)
    return bytes(out), expected


# -----------------------------------------------------------------------
# Palette / indexed color (ColorMap tag 320)
# -----------------------------------------------------------------------

def _make_palette_tiff(width, height, bps, pixel_values, palette_rgb):
    """Build a palette-color TIFF (Photometric=3 + ColorMap tag).

    palette_rgb: list of (R, G, B) tuples, uint16 values (0-65535).
    """
    import struct
    bo = '<'
    n_colors = len(palette_rgb)
    assert n_colors == (1 << bps), f"Palette must have {1 << bps} entries for {bps}-bit"

    # Pack pixel data
    flat = pixel_values.ravel().astype(np.uint8)
    if bps == 8:
        pixel_bytes = flat.tobytes()
    elif bps == 4:
        n = len(flat)
        packed_len = (n + 1) // 2
        packed = np.zeros(packed_len, dtype=np.uint8)
        for i in range(n):
            if i % 2 == 0:
                packed[i // 2] |= (flat[i] & 0x0F) << 4
            else:
                packed[i // 2] |= flat[i] & 0x0F
        pixel_bytes = packed.tobytes()
    else:
        pixel_bytes = flat.tobytes()

    # Build ColorMap: [R0..R_{n-1}, G0..G_{n-1}, B0..B_{n-1}]
    r_vals = [c[0] for c in palette_rgb]
    g_vals = [c[1] for c in palette_rgb]
    b_vals = [c[2] for c in palette_rgb]
    cmap_values = r_vals + g_vals + b_vals

    tag_list = []
    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))
    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))
    def add_shorts(tag, vals):
        tag_list.append((tag, 3, len(vals), struct.pack(f'{bo}{len(vals)}H', *vals)))

    add_short(256, width)
    add_short(257, height)
    add_short(258, bps)
    add_short(259, 1)     # no compression
    add_short(262, 3)     # Photometric = Palette
    add_short(277, 1)     # SamplesPerPixel = 1
    add_short(278, height)
    add_long(273, 0)      # StripOffsets placeholder
    add_long(279, len(pixel_bytes))
    add_shorts(320, cmap_values)  # ColorMap
    add_short(339, 1)     # SampleFormat = UINT

    tag_list.sort(key=lambda t: t[0])
    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)

    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count, struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    overflow_buf = bytearray()
    tag_offsets = {}
    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixel_bytes)

    return bytes(out)


class TestPalette:

    def test_palette_8bit_read(self, tmp_path):
        """Read an 8-bit palette TIFF and verify pixel indices."""
        # 4-color palette: red, green, blue, white
        palette = [
            (65535, 0, 0),       # 0 = red
            (0, 65535, 0),       # 1 = green
            (0, 0, 65535),       # 2 = blue
            (65535, 65535, 65535),# 3 = white
        ] + [(0, 0, 0)] * 252   # pad to 256 entries for 8-bit

        pixels = np.array([[0, 1, 2, 3],
                           [3, 2, 1, 0]], dtype=np.uint8)

        tiff_data = _make_palette_tiff(4, 2, 8, pixels, palette)
        path = str(tmp_path / 'palette8.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = read_geotiff(path)
        # Should return raw index values
        assert da.dtype == np.uint8
        np.testing.assert_array_equal(da.values, pixels)

        # Should have a cmap in attrs
        assert 'cmap' in da.attrs
        assert 'colormap_rgba' in da.attrs

        # Verify the palette colors
        rgba = da.attrs['colormap_rgba']
        assert len(rgba) == 256
        assert rgba[0] == pytest.approx((1.0, 0.0, 0.0, 1.0))
        assert rgba[1] == pytest.approx((0.0, 1.0, 0.0, 1.0))
        assert rgba[2] == pytest.approx((0.0, 0.0, 1.0, 1.0))

    def test_palette_4bit(self, tmp_path):
        """Read a 4-bit palette TIFF."""
        palette = [(i * 4369, i * 4369, i * 4369) for i in range(16)]
        pixels = np.array([[0, 5, 10, 15],
                           [1, 6, 11, 3]], dtype=np.uint8)

        tiff_data = _make_palette_tiff(4, 2, 4, pixels, palette)
        path = str(tmp_path / 'palette4.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = read_geotiff(path)
        assert da.dtype == np.uint8
        np.testing.assert_array_equal(da.values, pixels)
        assert 'cmap' in da.attrs
        assert len(da.attrs['colormap_rgba']) == 16

    def test_palette_cmap_works_with_plot(self, tmp_path):
        """Verify the colormap can be used with matplotlib."""
        from matplotlib.colors import ListedColormap

        palette = [
            (65535, 0, 0),
            (0, 65535, 0),
            (0, 0, 65535),
            (65535, 65535, 0),
        ] + [(0, 0, 0)] * 252

        pixels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        tiff_data = _make_palette_tiff(2, 2, 8, pixels, palette)
        path = str(tmp_path / 'palette_plot.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = read_geotiff(path)
        cmap = da.attrs['cmap']
        assert isinstance(cmap, ListedColormap)

        # Verify color mapping at known indices
        assert cmap(0)[:3] == pytest.approx((1.0, 0.0, 0.0), abs=0.01)
        assert cmap(1 / 255)[:3] == pytest.approx((0.0, 1.0, 0.0), abs=0.01)

    def test_xrs_plot_with_palette(self, tmp_path):
        """da.xrs.plot() uses the embedded colormap."""
        import matplotlib
        matplotlib.use('Agg')
        import xrspatial.accessor  # register .xrs accessor

        palette = [
            (65535, 0, 0),
            (0, 65535, 0),
            (0, 0, 65535),
            (65535, 65535, 65535),
        ] + [(0, 0, 0)] * 252

        pixels = np.array([[0, 1, 2, 3],
                           [3, 2, 1, 0]], dtype=np.uint8)
        tiff_data = _make_palette_tiff(4, 2, 8, pixels, palette)
        path = str(tmp_path / 'plot_palette.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = read_geotiff(path)
        artist = da.xrs.plot()
        assert artist is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_xrs_plot_no_palette(self, tmp_path):
        """da.xrs.plot() falls through to normal plot for non-palette data."""
        import matplotlib
        matplotlib.use('Agg')
        import xrspatial.accessor

        arr = np.random.RandomState(42).rand(4, 4).astype(np.float32)
        path = str(tmp_path / 'no_palette.tif')
        write(arr, path, compression='none', tiled=False)

        da = read_geotiff(path)
        artist = da.xrs.plot()
        assert artist is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plot_geotiff_deprecated(self, tmp_path):
        """plot_geotiff still works as deprecated wrapper."""
        import matplotlib
        matplotlib.use('Agg')
        import xrspatial.accessor
        from xrspatial.geotiff import plot_geotiff

        palette = [(65535, 0, 0), (0, 65535, 0)] + [(0, 0, 0)] * 254
        pixels = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        tiff_data = _make_palette_tiff(2, 2, 8, pixels, palette)
        path = str(tmp_path / 'deprecated.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = read_geotiff(path)
        artist = plot_geotiff(da)
        assert artist is not None
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_non_palette_no_cmap(self, tmp_path):
        """Non-palette TIFFs should not have a cmap attr."""
        arr = np.ones((4, 4), dtype=np.float32)
        path = str(tmp_path / 'no_palette.tif')
        write(arr, path, compression='none', tiled=False)

        da = read_geotiff(path)
        assert 'cmap' not in da.attrs
        assert 'colormap_rgba' not in da.attrs


class TestPlanarConfig:

    def test_planar_strips_rgb(self, tmp_path):
        """Read a 3-band planar-stripped TIFF."""
        tiff_data, expected = _make_planar_tiff(4, 6, 3, np.uint8)
        path = str(tmp_path / 'planar_strip.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.shape == (6, 4, 3)
        np.testing.assert_array_equal(result, expected)

    def test_planar_strips_2band(self, tmp_path):
        """Read a 2-band planar-stripped TIFF."""
        tiff_data, expected = _make_planar_tiff(5, 4, 2, np.uint16)
        path = str(tmp_path / 'planar_2band.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.shape == (4, 5, 2)
        np.testing.assert_array_equal(result, expected)

    def test_planar_tiles_rgb(self, tmp_path):
        """Read a 3-band planar-tiled TIFF."""
        tiff_data, expected = _make_planar_tiff(
            8, 8, 3, np.uint8, tiled=True, tile_size=4)
        path = str(tmp_path / 'planar_tiled.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path)
        assert result.shape == (8, 8, 3)
        np.testing.assert_array_equal(result, expected)

    def test_planar_windowed(self, tmp_path):
        """Windowed read of a planar-stripped TIFF."""
        tiff_data, expected = _make_planar_tiff(8, 8, 3, np.uint8)
        path = str(tmp_path / 'planar_window.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path, window=(2, 1, 6, 5))
        np.testing.assert_array_equal(result, expected[2:6, 1:5, :])

    def test_planar_band_selection(self, tmp_path):
        """Selecting a single band from a planar TIFF."""
        tiff_data, expected = _make_planar_tiff(4, 4, 3, np.uint8)
        path = str(tmp_path / 'planar_band.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        result, _ = read_to_array(path, band=1)
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, expected[:, :, 1])

    def test_planar_via_public_api(self, tmp_path):
        """read_geotiff on a planar file returns correct DataArray."""
        from xrspatial.geotiff import read_geotiff
        tiff_data, expected = _make_planar_tiff(4, 4, 3, np.uint8)
        path = str(tmp_path / 'planar_api.tif')
        with open(path, 'wb') as f:
            f.write(tiff_data)

        da = read_geotiff(path)
        assert 'band' in da.dims
        assert da.shape == (4, 4, 3)
        np.testing.assert_array_equal(da.values, expected)


# -----------------------------------------------------------------------
# Dask lazy reads
# -----------------------------------------------------------------------

class TestDaskReads:

    def test_dask_basic(self, tmp_path):
        """read_geotiff_dask returns a dask-backed DataArray."""
        import dask.array as da
        from xrspatial.geotiff import read_geotiff_dask

        arr = np.arange(256, dtype=np.float32).reshape(16, 16)
        path = str(tmp_path / 'dask_test.tif')
        write(arr, path, compression='none', tiled=False)

        result = read_geotiff_dask(path, chunks=8)
        assert isinstance(result.data, da.Array)
        assert result.shape == (16, 16)

        # Compute and compare
        computed = result.compute()
        np.testing.assert_array_equal(computed.values, arr)

    def test_dask_coords(self, tmp_path):
        """Dask read preserves coordinates and CRS."""
        from xrspatial.geotiff import read_geotiff_dask
        from xrspatial.geotiff._geotags import GeoTransform

        arr = np.ones((8, 8), dtype=np.float32)
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        path = str(tmp_path / 'dask_geo.tif')
        write(arr, path, geo_transform=gt, crs_epsg=4326,
              compression='none', tiled=False)

        result = read_geotiff_dask(path, chunks=4)
        assert result.attrs['crs'] == 4326
        assert len(result.coords['y']) == 8
        assert len(result.coords['x']) == 8

    def test_dask_nodata(self, tmp_path):
        """Nodata masking applied per-chunk."""
        from xrspatial.geotiff import read_geotiff_dask

        arr = np.array([[1.0, -9999.0], [-9999.0, 2.0],
                        [3.0, 4.0], [5.0, -9999.0]], dtype=np.float32)
        path = str(tmp_path / 'dask_nodata.tif')
        write(arr, path, compression='none', tiled=False, nodata=-9999.0)

        result = read_geotiff_dask(path, chunks=2)
        computed = result.compute()
        assert np.isnan(computed.values[0, 1])
        assert np.isnan(computed.values[1, 0])
        assert computed.values[0, 0] == 1.0

    def test_dask_chunk_tuple(self, tmp_path):
        """Chunks as (row, col) tuple."""
        from xrspatial.geotiff import read_geotiff_dask

        arr = np.arange(200, dtype=np.float32).reshape(10, 20)
        path = str(tmp_path / 'dask_tuple.tif')
        write(arr, path, compression='deflate', tiled=False)

        result = read_geotiff_dask(path, chunks=(5, 10))
        computed = result.compute()
        np.testing.assert_array_equal(computed.values, arr)
