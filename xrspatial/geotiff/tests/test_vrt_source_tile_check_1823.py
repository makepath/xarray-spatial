"""Regression tests for #1823.

PR #1803 forwarded the caller's ``max_pixels`` to ``read_to_array`` inside
the VRT source loop so that a tiny VRT output could not force a huge
source decode (#1796). A separate per-tile dimension check was
incorrectly using the same ``max_pixels`` value, so a caller setting
``max_pixels`` as an output budget (e.g. 10,000) would also fail the
per-tile sanity check on every normal source whose default tile size is
256x256 (= 65,536 pixels).

After PR #1821 (#1704) the resample path reads only the minimal source
sub-rect that feeds the clipped destination sub-window, so a tiny VRT
output cannot force a huge source decode by construction. The
output-extent check at the top of ``read_vrt`` still rejects requests
whose output itself exceeds ``max_pixels``.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._reader import PixelSafetyLimitError
from xrspatial.geotiff._vrt import read_vrt


def _write_normal_tile_source(td: str) -> str:
    """10x10 uint8 source -- ``to_geotiff`` pads to a 256x256 tile."""
    src = os.path.join(td, 'src.tif')
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src, compression='none')
    return src


def _write_vrt(td: str, *, dst_x_size: int, dst_y_size: int,
               raster_x: int = 100, raster_y: int = 100,
               src_x_size: int = 10, src_y_size: int = 10) -> str:
    vrt = os.path.join(td, 'mosaic.vrt')
    xml = (
        f'<VRTDataset rasterXSize="{raster_x}" rasterYSize="{raster_y}">\n'
        f'  <VRTRasterBand dataType="Byte" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">src.tif</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" '
        f'xSize="{src_x_size}" ySize="{src_y_size}"/>\n'
        f'      <DstRect xOff="0" yOff="0" '
        f'xSize="{dst_x_size}" ySize="{dst_y_size}"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with open(vrt, 'w') as f:
        f.write(xml)
    return vrt


class TestPerTileCheckDoesNotUseCallerBudget:
    """Per-tile dim sanity must not reject normal 256x256 source tiles
    when the caller's ``max_pixels`` is a small output-budget value."""

    def test_normal_tile_source_with_small_max_pixels(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_normal_tile_source(td)
            vrt = _write_vrt(td, dst_x_size=100, dst_y_size=100)
            arr, _ = read_vrt(vrt, max_pixels=10_000)
            assert arr.shape == (100, 100)

    def test_normal_tile_source_with_tiny_max_pixels(self):
        """An output budget below a single tile must still succeed when
        the requested output window itself fits."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_normal_tile_source(td)
            # Output 5x5 = 25 pixels; max_pixels = 100 fits 25 with room.
            vrt = _write_vrt(td, dst_x_size=5, dst_y_size=5,
                             raster_x=5, raster_y=5)
            arr, _ = read_vrt(vrt, max_pixels=100)
            assert arr.shape == (5, 5)


class TestOutputWindowCheckStillEnforced:
    """The output-window check still rejects a read whose VRT extent
    exceeds ``max_pixels``. After #1704 the source read is bounded by
    the clipped destination sub-window, so the per-source guard now
    rarely fires; the top-level ``_check_dimensions`` call against the
    output extent catches over-budget requests up front. The #1796
    protection (tiny VRT cannot force huge source decode) is preserved
    structurally.
    """

    def test_output_extent_exceeds_max_pixels_still_rejected(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            src = os.path.join(td, 'src.tif')
            to_geotiff(np.arange(64, dtype=np.uint8).reshape(8, 8),
                       src, compression='none', tiled=False)
            # VRT output extent is 8x8 = 64 pixels; max_pixels=4 trips
            # the dimensions check before any source read runs.
            vrt = _write_vrt(td, dst_x_size=8, dst_y_size=8,
                             raster_x=8, raster_y=8,
                             src_x_size=4, src_y_size=4)
            with pytest.raises(ValueError, match="exceed|safety limit"):
                read_vrt(vrt, max_pixels=4)


class TestPerTileCheckStillRejectsCraftedHeader:
    """A pathological ``TileWidth``/``TileLength`` must still fail at
    the per-tile sanity check, which uses ``MAX_PIXELS_DEFAULT``."""

    def test_per_tile_check_caps_at_default(self, monkeypatch):
        """Lower ``MAX_PIXELS_DEFAULT`` to verify the per-tile call site
        is wired to it (rather than to the caller's budget)."""
        from xrspatial.geotiff import _reader as reader_mod

        monkeypatch.setattr(reader_mod, "MAX_PIXELS_DEFAULT", 100)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _write_normal_tile_source(td)
            vrt = _write_vrt(td, dst_x_size=100, dst_y_size=100)
            # 256x256 tile > patched MAX_PIXELS_DEFAULT=100 → per-tile
            # check fires regardless of caller's max_pixels (1e9 here).
            with pytest.raises(PixelSafetyLimitError, match="65,536"):
                read_vrt(vrt, max_pixels=1_000_000_000)
