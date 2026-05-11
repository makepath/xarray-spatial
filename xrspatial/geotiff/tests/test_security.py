"""Security tests for the geotiff subpackage.

Tests for:
- Unbounded allocation guard (issue #1184)
- VRT path traversal prevention (issue #1185)
- GPU read and VRT read allocation guards (issue #1195)
"""
from __future__ import annotations

import os
import struct
import tempfile
import threading

import numpy as np
import pytest

from xrspatial.geotiff._reader import (
    MAX_PIXELS_DEFAULT,
    _check_dimensions,
    _read_strips,
    _read_tiles,
    read_to_array,
)
from xrspatial.geotiff._header import parse_header, parse_all_ifds
from xrspatial.geotiff._dtypes import tiff_dtype_to_numpy
from .conftest import make_minimal_tiff


# ---------------------------------------------------------------------------
# Cat 1: Unbounded allocation guard
# ---------------------------------------------------------------------------

class TestDimensionGuard:
    def test_check_dimensions_rejects_oversized(self):
        """_check_dimensions raises when total pixels exceed the limit."""
        with pytest.raises(ValueError, match="exceed the safety limit"):
            _check_dimensions(100_000, 100_000, 1, MAX_PIXELS_DEFAULT)

    def test_check_dimensions_accepts_normal(self):
        """_check_dimensions does not raise for normal sizes."""
        _check_dimensions(1000, 1000, 1, MAX_PIXELS_DEFAULT)

    def test_check_dimensions_considers_samples(self):
        """Multi-band images multiply the pixel budget."""
        # 50_000 x 50_000 x 3 = 7.5 billion, should be rejected
        with pytest.raises(ValueError, match="exceed the safety limit"):
            _check_dimensions(50_000, 50_000, 3, MAX_PIXELS_DEFAULT)

    def test_custom_limit(self):
        """A custom max_pixels lets callers tighten or relax the limit."""
        # Tight limit: 100 pixels
        with pytest.raises(ValueError, match="exceed the safety limit"):
            _check_dimensions(20, 20, 1, max_pixels=100)

        # Relaxed: passes with large limit
        _check_dimensions(100_000, 100_000, 1, max_pixels=100_000_000_000)

    def test_read_strips_rejects_huge_header(self):
        """_read_strips refuses to allocate when header claims huge dims."""
        # Build a valid TIFF with small pixel data but huge header dimensions.
        # We fake the header to claim 100000x100000 but only provide 4x4 data.
        data = make_minimal_tiff(4, 4, np.dtype('float32'))
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]

        # Monkey-patch the IFD width/height to simulate a crafted header
        from xrspatial.geotiff._header import IFDEntry
        ifd.entries[256] = IFDEntry(tag=256, type_id=3, count=1, value=100_000)
        ifd.entries[257] = IFDEntry(tag=257, type_id=3, count=1, value=100_000)

        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        with pytest.raises(ValueError, match="exceed the safety limit"):
            _read_strips(data, ifd, header, dtype, max_pixels=1_000_000)

    def test_read_tiles_rejects_huge_header(self):
        """_read_tiles refuses to allocate when header claims huge dims."""
        data = make_minimal_tiff(8, 8, np.dtype('float32'), tiled=True, tile_size=4)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]

        from xrspatial.geotiff._header import IFDEntry
        ifd.entries[256] = IFDEntry(tag=256, type_id=3, count=1, value=100_000)
        ifd.entries[257] = IFDEntry(tag=257, type_id=3, count=1, value=100_000)

        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        with pytest.raises(ValueError, match="exceed the safety limit"):
            _read_tiles(data, ifd, header, dtype, max_pixels=1_000_000)

    def test_read_to_array_max_pixels_kwarg(self, tmp_path):
        """read_to_array passes max_pixels through to the internal readers."""
        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        data = make_minimal_tiff(4, 4, np.dtype('float32'), pixel_data=expected)
        path = str(tmp_path / "small.tif")
        with open(path, 'wb') as f:
            f.write(data)

        # Should succeed with a generous limit
        arr, _ = read_to_array(path, max_pixels=1_000_000)
        np.testing.assert_array_equal(arr, expected)

        # Should fail with a tiny limit
        with pytest.raises(ValueError, match="exceed the safety limit"):
            read_to_array(path, max_pixels=10)

    def test_normal_read_unaffected(self, tmp_path):
        """Normal reads within the default limit are not affected."""
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        data = make_minimal_tiff(8, 8, np.dtype('float32'), pixel_data=expected)
        path = str(tmp_path / "normal.tif")
        with open(path, 'wb') as f:
            f.write(data)

        arr, _ = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)

    def test_open_geotiff_max_pixels(self, tmp_path):
        """open_geotiff passes max_pixels through to the reader."""
        from xrspatial.geotiff import open_geotiff

        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        data = make_minimal_tiff(4, 4, np.dtype('float32'), pixel_data=expected)
        path = str(tmp_path / "small_1195.tif")
        with open(path, 'wb') as f:
            f.write(data)

        # Should succeed with generous limit
        da = open_geotiff(path, max_pixels=1_000_000)
        np.testing.assert_array_equal(da.values, expected)

        # Should fail with tiny limit
        with pytest.raises(ValueError, match="exceed the safety limit"):
            open_geotiff(path, max_pixels=10)


# ---------------------------------------------------------------------------
# Cat 1c: Tile dimension guard (issue #1215)
# ---------------------------------------------------------------------------

class TestTileDimensionGuard:
    """Per-tile dims must also respect max_pixels, not just image dims.

    A crafted TIFF can declare a tiny image while claiming a 2^30 x 2^30
    tile. Without this guard, _decode_strip_or_tile asks the decompressor
    for terabytes.
    """

    def test_read_tiles_rejects_huge_tile_dims(self):
        """_read_tiles refuses to decode when tile dims would OOM."""
        data = make_minimal_tiff(8, 8, np.dtype('float32'),
                                 tiled=True, tile_size=4)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]

        # Forge tile_width / tile_length to simulate an attacker-controlled
        # header. Image dims stay small so the image-level check passes.
        from xrspatial.geotiff._header import IFDEntry
        ifd.entries[322] = IFDEntry(tag=322, type_id=4, count=1,
                                    value=1_000_000)
        ifd.entries[323] = IFDEntry(tag=323, type_id=4, count=1,
                                    value=1_000_000)

        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        with pytest.raises(ValueError, match="exceed the safety limit"):
            _read_tiles(data, ifd, header, dtype, max_pixels=1_000_000)

    def test_read_tiles_rejects_zero_tile_dims(self):
        """_read_tiles rejects tile dims of zero rather than dividing by 0."""
        data = make_minimal_tiff(8, 8, np.dtype('float32'),
                                 tiled=True, tile_size=4)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]

        from xrspatial.geotiff._header import IFDEntry
        ifd.entries[322] = IFDEntry(tag=322, type_id=4, count=1, value=0)
        ifd.entries[323] = IFDEntry(tag=323, type_id=4, count=1, value=0)

        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        with pytest.raises(ValueError, match="Invalid tile dimensions"):
            _read_tiles(data, ifd, header, dtype, max_pixels=1_000_000)

    def test_normal_tile_dims_pass(self, tmp_path):
        """Legitimate tile_size=4 on an 8x8 image still works."""
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        data = make_minimal_tiff(8, 8, np.dtype('float32'),
                                 pixel_data=expected,
                                 tiled=True, tile_size=4)
        path = str(tmp_path / "tile_dims_1215.tif")
        with open(path, 'wb') as f:
            f.write(data)

        # max_pixels=1000 is generous enough for a 4x4 tile (16 pixels)
        arr, _ = read_to_array(path, max_pixels=1000)
        np.testing.assert_array_equal(arr, expected)

    def test_open_geotiff_forged_tile_dims(self, tmp_path):
        """End-to-end: open_geotiff rejects a TIFF with forged tile dims.

        Writes a real TIFF file with a small image but a huge TileWidth
        field, then checks that open_geotiff raises rather than OOMing.
        """
        from xrspatial.geotiff import open_geotiff

        # Build a tiny tiled TIFF, then patch the tile_width field in the
        # raw bytes. make_minimal_tiff stores tile_width as a SHORT at
        # tag 322, so we re-parse, find the entry, and overwrite the
        # inline value with a 32-bit LONG pointing at a huge number.
        base = make_minimal_tiff(8, 8, np.dtype('float32'),
                                 tiled=True, tile_size=4)
        path = str(tmp_path / "forged_tile_1215.tif")
        with open(path, 'wb') as f:
            f.write(base)

        # Parse to locate the tile-width entry, then rewrite it in place.
        # The conftest TIFF uses little-endian SHORT for TileWidth (322).
        import struct
        header = parse_header(base)
        # IFD starts at offset 8, then 2-byte count, then 12-byte entries
        num_entries = struct.unpack_from('<H', base, 8)[0]
        patched = bytearray(base)
        for i in range(num_entries):
            eo = 10 + i * 12
            tag = struct.unpack_from('<H', patched, eo)[0]
            if tag == 322 or tag == 323:
                # Rewrite as LONG (type=4), count=1, value=1_000_000
                struct.pack_into('<HHII', patched, eo,
                                 tag, 4, 1, 1_000_000)

        forged_path = str(tmp_path / "forged_1215_huge.tif")
        with open(forged_path, 'wb') as f:
            f.write(bytes(patched))

        with pytest.raises(ValueError, match="exceed the safety limit"):
            open_geotiff(forged_path, max_pixels=1_000_000)


# ---------------------------------------------------------------------------
# Cat 1b: VRT allocation guard (issue #1195)
# ---------------------------------------------------------------------------

class TestVRTAllocationGuard:
    def test_read_vrt_rejects_huge_dimensions(self, tmp_path):
        """read_vrt refuses to allocate when VRT XML claims huge dims."""
        from xrspatial.geotiff._vrt import read_vrt as _read_vrt_internal

        # Create a VRT with oversized dimensions but no actual source data
        # needed -- _check_dimensions fires before any file reads
        vrt_xml = '''<VRTDataset rasterXSize="100000" rasterYSize="100000">
  <VRTRasterBand dataType="Float32" band="1">
  </VRTRasterBand>
</VRTDataset>'''

        vrt_path = str(tmp_path / "huge_1195.vrt")
        with open(vrt_path, 'w') as f:
            f.write(vrt_xml)

        with pytest.raises(ValueError, match="exceed the safety limit"):
            _read_vrt_internal(vrt_path, max_pixels=1_000_000)

    def test_read_vrt_normal_size_ok(self, tmp_path):
        """Normal-sized VRT passes the allocation guard."""
        from xrspatial.geotiff._vrt import read_vrt as _read_vrt_internal

        vrt_xml = '''<VRTDataset rasterXSize="4" rasterYSize="4">
  <VRTRasterBand dataType="Float32" band="1">
  </VRTRasterBand>
</VRTDataset>'''

        vrt_path = str(tmp_path / "small_1195.vrt")
        with open(vrt_path, 'w') as f:
            f.write(vrt_xml)

        # Should not raise -- 4x4x1 = 16 pixels
        arr, vrt = _read_vrt_internal(vrt_path, max_pixels=1_000_000)
        assert arr.shape == (4, 4)

    def test_open_geotiff_vrt_max_pixels(self, tmp_path):
        """open_geotiff passes max_pixels through to VRT reader."""
        from xrspatial.geotiff import open_geotiff

        vrt_xml = '''<VRTDataset rasterXSize="100000" rasterYSize="100000">
  <VRTRasterBand dataType="Float32" band="1">
  </VRTRasterBand>
</VRTDataset>'''

        vrt_path = str(tmp_path / "huge_vrt_1195.vrt")
        with open(vrt_path, 'w') as f:
            f.write(vrt_xml)

        with pytest.raises(ValueError, match="exceed the safety limit"):
            open_geotiff(vrt_path, max_pixels=1_000_000)

# ---------------------------------------------------------------------------
# Cat 5: VRT path traversal
# ---------------------------------------------------------------------------

class TestVRTPathTraversal:
    def test_relative_path_canonicalized(self, tmp_path):
        """Relative paths in VRT SourceFilename are canonicalized."""
        from xrspatial.geotiff._vrt import parse_vrt

        vrt_xml = '''<VRTDataset rasterXSize="4" rasterYSize="4">
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">../../../etc/shadow</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>'''

        vrt_dir = str(tmp_path / "subdir")
        os.makedirs(vrt_dir)

        vrt = parse_vrt(vrt_xml, vrt_dir)
        source_path = vrt.bands[0].sources[0].filename

        # After canonicalization, the path should NOT contain ".."
        assert ".." not in source_path
        # It should be an absolute path
        assert os.path.isabs(source_path)
        # Verify it was resolved through realpath
        assert source_path == os.path.realpath(source_path)

    def test_normal_relative_path_still_works(self, tmp_path):
        """Normal relative paths without traversal still resolve correctly."""
        from xrspatial.geotiff._vrt import parse_vrt

        vrt_xml = '''<VRTDataset rasterXSize="4" rasterYSize="4">
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">data/tile.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>'''

        vrt_dir = str(tmp_path)
        vrt = parse_vrt(vrt_xml, vrt_dir)
        source_path = vrt.bands[0].sources[0].filename

        expected = os.path.realpath(os.path.join(vrt_dir, "data", "tile.tif"))
        assert source_path == expected

    def test_absolute_path_also_canonicalized(self, tmp_path):
        """Absolute paths in VRT are also canonicalized."""
        from xrspatial.geotiff._vrt import parse_vrt

        vrt_xml = '''<VRTDataset rasterXSize="4" rasterYSize="4">
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">/tmp/../tmp/test.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>'''

        vrt = parse_vrt(vrt_xml, str(tmp_path))
        source_path = vrt.bands[0].sources[0].filename

        assert ".." not in source_path
        assert source_path == os.path.realpath("/tmp/../tmp/test.tif")


# ---------------------------------------------------------------------------
# Tile layout validation (issue #1219)
#
# An adversarial TIFF can declare image dimensions that imply more tiles
# than its TileOffsets tag supplies. The CPU path silently skipped the
# missing tiles (zero-padded output) and the GPU tile-assembly kernel
# read past the end of the decompression-offsets array on device.
# ---------------------------------------------------------------------------

def _make_short_offsets_tiff(
    width: int,
    height: int,
    tile_size: int,
    declared_offset_count: int,
    dtype: np.dtype = np.dtype('float32'),
) -> bytes:
    """Build a tiled TIFF whose TileOffsets tag count is less than what
    the image/tile dimensions imply.

    The produced bytes are a valid TIFF that rioxarray/GDAL might still
    accept with a warning, but our reader should reject them.
    """
    import math

    # Start from a normal tiled TIFF, then rewrite the TileOffsets IFD
    # entry to advertise a smaller count.
    data = bytearray(make_minimal_tiff(
        width, height, dtype, tiled=True, tile_size=tile_size))

    # Parse to locate the TileOffsets entry.
    header = parse_header(bytes(data))
    ifds = parse_all_ifds(bytes(data), header)
    ifd = ifds[0]

    offsets = ifd.tile_offsets
    assert offsets is not None
    tiles_across = math.ceil(width / tile_size)
    tiles_down = math.ceil(height / tile_size)
    true_count = tiles_across * tiles_down
    assert len(offsets) == true_count
    assert declared_offset_count < true_count

    # Find the IFD entry bytes for tag 324 (TileOffsets) and rewrite its
    # count field. TIFF (non-BigTIFF) entries are 12 bytes: HHIi
    # (tag, type, count, value_or_ptr).
    bo = header.byte_order
    ifd_offset = header.first_ifd_offset
    num_entries = struct.unpack_from(f'{bo}H', data, ifd_offset)[0]
    entry_offset = ifd_offset + 2
    for i in range(num_entries):
        eo = entry_offset + i * 12
        tag = struct.unpack_from(f'{bo}H', data, eo)[0]
        if tag == 324:  # TileOffsets
            # Overwrite the count field at eo+4 (4 bytes, unsigned int).
            struct.pack_into(
                f'{bo}I', data, eo + 4, declared_offset_count)
            break
    else:
        raise AssertionError("TileOffsets tag not found in IFD")

    return bytes(data)


class TestTileLayoutValidation:
    """Regression tests for issue #1219."""

    def test_validate_tile_layout_rejects_short_offsets(self):
        """validate_tile_layout raises when offsets count < declared grid."""
        from xrspatial.geotiff._header import validate_tile_layout

        # 16x16 image with 4x4 tiles = 16 tiles, but only 4 offsets declared.
        data = _make_short_offsets_tiff(
            width=16, height=16, tile_size=4, declared_offset_count=4)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]

        with pytest.raises(ValueError, match="Malformed TIFF.*tile offsets"):
            validate_tile_layout(ifd)

    def test_validate_tile_layout_accepts_well_formed(self):
        """validate_tile_layout accepts a normal tiled TIFF."""
        from xrspatial.geotiff._header import validate_tile_layout

        data = make_minimal_tiff(
            8, 8, np.dtype('float32'), tiled=True, tile_size=4)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]

        # Should not raise.
        validate_tile_layout(ifd)

    def test_validate_tile_layout_ignores_stripped(self):
        """validate_tile_layout is a no-op for stripped TIFFs."""
        from xrspatial.geotiff._header import validate_tile_layout

        data = make_minimal_tiff(4, 4, np.dtype('float32'))
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]

        # Should not raise -- stripped file, not tiled.
        validate_tile_layout(ifd)

    def test_read_tiles_rejects_short_offsets(self):
        """_read_tiles surfaces the malformed-TIFF error instead of
        silently zero-padding missing tiles."""
        data = _make_short_offsets_tiff(
            width=16, height=16, tile_size=4, declared_offset_count=4)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd = ifds[0]
        dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)

        with pytest.raises(ValueError, match="Malformed TIFF"):
            _read_tiles(data, ifd, header, dtype)

    def test_read_to_array_rejects_short_offsets(self, tmp_path):
        """End-to-end: reading a short-offsets TIFF raises a clear
        ValueError (not a silent zero output or CUDA crash)."""
        data = _make_short_offsets_tiff(
            width=16, height=16, tile_size=4, declared_offset_count=4)
        path = str(tmp_path / "malformed_1219.tif")
        with open(path, 'wb') as f:
            f.write(data)

        with pytest.raises(ValueError, match="Malformed TIFF"):
            read_to_array(path)

    def test_boundary_exact_count_ok(self, tmp_path):
        """A TIFF with exactly the required number of offsets reads fine."""
        # 8x8 image, 4x4 tiles => 4 tiles exactly.
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        data = make_minimal_tiff(
            8, 8, np.dtype('float32'),
            pixel_data=expected, tiled=True, tile_size=4)
        path = str(tmp_path / "exact_1219.tif")
        with open(path, 'wb') as f:
            f.write(data)

        arr, _ = read_to_array(path)
        np.testing.assert_array_equal(arr, expected)


# ---------------------------------------------------------------------------
# HTTP COG: per-tile compressed-byte cap (issue #1536)
#
# A crafted TIFF served over HTTP can declare arbitrarily large
# TileByteCounts. Without the cap added in #1536, _fetch_decode_cog_http_tiles
# passes those values straight into Range GETs sized by the attacker.
# The local-mmap path is naturally bounded by file size, so these tests
# only exercise the HTTP path through a mock _HTTPSource.
# ---------------------------------------------------------------------------


def _patch_tile_byte_counts(data: bytearray, value: int) -> None:
    """Rewrite every TileByteCounts entry in *data* (in place) to *value*.

    Walks the first IFD looking for tag 325. Handles the LONG (4 byte) and
    SHORT (2 byte) encodings, both inline and via overflow pointer. Used by
    the tests below to forge a COG with attacker-controlled byte counts.
    """
    from xrspatial.geotiff._header import parse_header
    header = parse_header(bytes(data))
    bo = header.byte_order
    ifd_offset = header.first_ifd_offset
    num_entries = struct.unpack_from(f'{bo}H', data, ifd_offset)[0]
    entry_offset = ifd_offset + 2

    for i in range(num_entries):
        eo = entry_offset + i * 12
        tag = struct.unpack_from(f'{bo}H', data, eo)[0]
        if tag != 325:  # TileByteCounts
            continue
        type_id = struct.unpack_from(f'{bo}H', data, eo + 2)[0]
        count = struct.unpack_from(f'{bo}I', data, eo + 4)[0]
        if type_id == 4:  # LONG
            total = count * 4
            if total <= 4:
                for k in range(count):
                    struct.pack_into(f'{bo}I', data, eo + 8 + k * 4, value)
            else:
                ptr = struct.unpack_from(f'{bo}I', data, eo + 8)[0]
                for k in range(count):
                    struct.pack_into(
                        f'{bo}I', data, ptr + k * 4, value)
        elif type_id == 3:  # SHORT
            clipped = min(value, 0xFFFF)
            total = count * 2
            if total <= 4:
                for k in range(count):
                    struct.pack_into(
                        f'{bo}H', data, eo + 8 + k * 2, clipped)
            else:
                ptr = struct.unpack_from(f'{bo}I', data, eo + 8)[0]
                for k in range(count):
                    struct.pack_into(
                        f'{bo}H', data, ptr + k * 2, clipped)
        return
    raise AssertionError("TileByteCounts (tag 325) not found in IFD")


class _MockHTTPSource:
    """Minimal _HTTPSource stand-in that serves bytes from an in-memory buf.

    Tests use this to drive _read_cog_http through the HTTP code path without
    spinning up a real server. read_range / read_ranges_coalesced match the
    real source's signatures so the reader cannot tell the difference.
    """

    def __init__(self, buf: bytes):
        self._url = 'mock://'
        self._size = len(buf)
        self._pool = None
        self._buf = buf
        self._lock = threading.Lock()
        self.calls: list[tuple[int, int]] = []

    def read_range(self, start: int, length: int) -> bytes:
        with self._lock:
            self.calls.append((start, length))
        return self._buf[start:start + length]

    def read_all(self) -> bytes:
        with self._lock:
            self.calls.append((0, len(self._buf)))
        return self._buf

    def read_ranges(self, ranges, max_workers=8):
        return [self.read_range(s, le) for s, le in ranges]

    def read_ranges_coalesced(self, ranges, max_workers=8, gap_threshold=None):
        from xrspatial.geotiff._reader import (
            coalesce_ranges, split_coalesced_bytes,
            COALESCE_GAP_THRESHOLD_DEFAULT,
        )
        if gap_threshold is None:
            gap_threshold = COALESCE_GAP_THRESHOLD_DEFAULT
        merged, mapping = coalesce_ranges(ranges, gap_threshold=gap_threshold)
        merged_bytes = self.read_ranges(merged, max_workers=max_workers)
        return split_coalesced_bytes(merged_bytes, mapping)

    def close(self):
        pass


class TestHTTPTileByteCountCap:
    """Regression tests for the HTTP COG byte_count cap (#1536)."""

    def _build_forged_cog(self, tmp_path, byte_count_value: int) -> bytes:
        """Build a real tiled COG, then patch every TileByteCounts entry."""
        from xrspatial.geotiff import to_geotiff
        import xarray as xr
        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        da = xr.DataArray(arr, dims=['y', 'x'])
        path = str(tmp_path / "forged_1536.tif")
        to_geotiff(da, path, tile_size=32, compression='deflate')
        with open(path, 'rb') as f:
            data = bytearray(f.read())
        _patch_tile_byte_counts(data, byte_count_value)
        return bytes(data)

    def test_huge_byte_count_rejected(self, tmp_path, monkeypatch):
        """A tile claiming more bytes than the cap raises ValueError."""
        from xrspatial.geotiff import _reader as _reader_mod

        # 100 MB > the 1 MB cap we set for this test
        forged = self._build_forged_cog(tmp_path, 100 * 1024 * 1024)
        src = _MockHTTPSource(forged)
        monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)
        monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', str(1024 * 1024))

        with pytest.raises(ValueError, match="TileByteCount"):
            _reader_mod._read_cog_http('http://mock/forged.tif')

    def test_error_message_names_offending_value(self, tmp_path, monkeypatch):
        """The error mentions the byte count and the cap so operators can
        diagnose without re-reading the source."""
        from xrspatial.geotiff import _reader as _reader_mod

        forged = self._build_forged_cog(tmp_path, 50 * 1024 * 1024)
        src = _MockHTTPSource(forged)
        monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)
        monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', str(1024))

        with pytest.raises(ValueError) as excinfo:
            _reader_mod._read_cog_http('http://mock/forged.tif')
        msg = str(excinfo.value)
        assert "52,428,800" in msg or "52428800" in msg  # the byte count
        assert "1,024" in msg or "1024" in msg            # the cap
        assert "denial-of-service" in msg.lower() or "malformed" in msg

    def test_normal_cog_still_reads(self, tmp_path, monkeypatch):
        """Realistic per-tile byte counts pass under the default cap."""
        from xrspatial.geotiff import to_geotiff, _reader as _reader_mod
        import xarray as xr

        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        da = xr.DataArray(arr, dims=['y', 'x'])
        path = str(tmp_path / "normal_1536.tif")
        to_geotiff(da, path, tile_size=32, compression='deflate')
        with open(path, 'rb') as f:
            buf = f.read()

        src = _MockHTTPSource(buf)
        monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)

        result, _ = _reader_mod._read_cog_http('http://mock/normal.tif')
        np.testing.assert_array_equal(result, arr)

    def test_env_override_lifts_cap(self, tmp_path, monkeypatch):
        """A user with legitimate large tiles can raise the cap via env."""
        from xrspatial.geotiff import to_geotiff, _reader as _reader_mod
        import xarray as xr

        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        da = xr.DataArray(arr, dims=['y', 'x'])
        path = str(tmp_path / "normal_env_1536.tif")
        to_geotiff(da, path, tile_size=32, compression='deflate')
        with open(path, 'rb') as f:
            buf = f.read()

        src = _MockHTTPSource(buf)
        monkeypatch.setattr(_reader_mod, '_HTTPSource', lambda url: src)
        # A tiny cap WITHOUT the env override would fire; lift it past
        # the actual byte counts and the read should succeed.
        monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', str(64 * 1024 * 1024))

        result, _ = _reader_mod._read_cog_http('http://mock/normal_env.tif')
        np.testing.assert_array_equal(result, arr)

    def test_local_path_unaffected_by_cap(self, tmp_path, monkeypatch):
        """The local-mmap reader bypasses the HTTP cap.

        A patched on-disk file with huge byte_counts decodes via mmap
        slicing, which silently truncates at EOF. The cap is HTTP-only,
        so a tight env cap should not break legitimate local reads.
        """
        from xrspatial.geotiff import open_geotiff, to_geotiff
        import xarray as xr

        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        da = xr.DataArray(arr, dims=['y', 'x'])
        path = str(tmp_path / "local_normal_1536.tif")
        to_geotiff(da, path, tile_size=32, compression='deflate')

        # Set the HTTP cap to 1 byte; local reads must still succeed
        monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', '1')
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)


# ---------------------------------------------------------------------------
# XML entity expansion (billion-laughs) -- issue #1579
#
# VRT and GDALMetadata payloads go through xml.etree.ElementTree, which by
# default expands internal entities. A crafted file can OOM the host via
# exponential entity expansion. ``_safe_xml.safe_fromstring`` refuses any
# input carrying a DOCTYPE declaration, which is where entity definitions
# live.
# ---------------------------------------------------------------------------


_BILLION_LAUGHS_PROLOGUE = (
    '<?xml version="1.0"?>\n'
    '<!DOCTYPE lolz [\n'
    '  <!ENTITY lol "lol">\n'
    '  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">\n'
    '  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">\n'
    '  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">\n'
    ']>\n'
)


class TestVRTXMLEntityExpansion:
    """Issue #1579: parse_vrt refuses XML entity (billion-laughs) payloads."""

    def test_parse_vrt_rejects_doctype(self, tmp_path):
        """A VRT that declares ``<!DOCTYPE ...>`` is rejected outright."""
        from xrspatial.geotiff._vrt import parse_vrt

        payload = _BILLION_LAUGHS_PROLOGUE + (
            '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
            '  <VRTRasterBand dataType="Float32" band="1">\n'
            '    <Description>&lol4;</Description>\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'
        )

        with pytest.raises(ValueError, match="DOCTYPE"):
            parse_vrt(payload, str(tmp_path))

    def test_open_geotiff_vrt_rejects_doctype(self, tmp_path):
        """Routing through the public open_geotiff entry still rejects DOCTYPEs."""
        from xrspatial.geotiff import open_geotiff

        payload = _BILLION_LAUGHS_PROLOGUE + (
            '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
            '  <VRTRasterBand dataType="Float32" band="1">\n'
            '    <Description>&lol4;</Description>\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
        vrt_path = tmp_path / "bomb.vrt"
        vrt_path.write_text(payload)

        with pytest.raises(ValueError, match="DOCTYPE"):
            open_geotiff(str(vrt_path))

    def test_normal_vrt_still_parses(self, tmp_path):
        """A VRT without a DOCTYPE parses normally."""
        from xrspatial.geotiff._vrt import parse_vrt

        payload = (
            '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
            '  <VRTRasterBand dataType="Float32" band="1">\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
        vrt = parse_vrt(payload, str(tmp_path))
        assert vrt.width == 4
        assert vrt.height == 4


class TestGDALMetadataXMLEntityExpansion:
    """Issue #1579: _parse_gdal_metadata refuses entity-expansion payloads."""

    def test_parse_gdal_metadata_doctype_returns_empty(self):
        """A DOCTYPE in GDALMetadata yields an empty dict, not expansion.

        GDALMetadata is non-essential auxiliary metadata, so the parser
        silently drops malformed payloads rather than failing the whole
        TIFF read. Crucially it must NOT expand the entity.
        """
        from xrspatial.geotiff._geotags import _parse_gdal_metadata

        payload = _BILLION_LAUGHS_PROLOGUE + (
            '<GDALMetadata><Item name="x">&lol4;</Item></GDALMetadata>'
        )
        result = _parse_gdal_metadata(payload)
        # The DOCTYPE branch falls through to an empty dict; the
        # critical assertion is that the value is NOT the expanded
        # entity (10000+ "lol" copies).
        assert result == {} or 'lol' * 100 not in str(result.get('x', ''))

    def test_parse_gdal_metadata_normal_still_works(self):
        """A well-formed GDALMetadata XML parses into a flat dict."""
        from xrspatial.geotiff._geotags import _parse_gdal_metadata

        payload = (
            '<GDALMetadata>'
            '<Item name="STATISTICS_MINIMUM">0</Item>'
            '<Item name="STATISTICS_MAXIMUM" sample="0">255</Item>'
            '</GDALMetadata>'
        )
        result = _parse_gdal_metadata(payload)
        assert result.get('STATISTICS_MINIMUM') == '0'
        assert result.get(('STATISTICS_MAXIMUM', 0)) == '255'


class TestSafeXMLHelper:
    """Direct unit tests for ``_safe_xml.safe_fromstring``."""

    def test_rejects_bytes_doctype(self):
        from xrspatial.geotiff._safe_xml import safe_fromstring
        with pytest.raises(ValueError, match="DOCTYPE"):
            safe_fromstring(b'<!DOCTYPE x><x/>')

    def test_rejects_str_doctype(self):
        from xrspatial.geotiff._safe_xml import safe_fromstring
        with pytest.raises(ValueError, match="DOCTYPE"):
            safe_fromstring('<!DOCTYPE x><x/>')

    def test_rejects_doctype_with_whitespace(self):
        """Whitespace between ``<!`` and ``DOCTYPE`` is still rejected."""
        from xrspatial.geotiff._safe_xml import safe_fromstring
        with pytest.raises(ValueError, match="DOCTYPE"):
            safe_fromstring('<!   DOCTYPE x><x/>')

    def test_parses_normal_xml(self):
        from xrspatial.geotiff._safe_xml import safe_fromstring
        root = safe_fromstring('<root><child>hi</child></root>')
        assert root.tag == 'root'
        assert root.find('child').text == 'hi'
