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
