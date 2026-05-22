"""VRT source reads must honor the caller's max_pixels budget (#1796).

Originally the source loop in ``read_vrt`` read the full SrcRect of a
SimpleSource with size mismatch, so a tiny VRT output could force a huge
source decode. #1803 forwarded ``max_pixels`` to ``read_to_array`` to
catch that pattern.

After #1704 / PR #1821 the resample path inverse-maps the clipped
destination sub-window to the minimal SrcRect sub-rect and reads only
that. A tiny VRT output is now bounded structurally: the source read
cannot exceed the dst sub-window size. The per-source ``max_pixels``
guard still applies (defence in depth) and still bites when the
sub-window itself exceeds the budget.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from xrspatial.geotiff import read_vrt, to_geotiff


def test_tiny_vrt_with_huge_srcrect_now_reads_minimally(tmp_path):
    """A 1x1 VRT pointing at a 4x4 SrcRect now reads only the one source
    pixel that maps to the single output pixel, so ``max_pixels=1`` is
    no longer exceeded. Locks in the structural improvement from #1704."""
    src = tmp_path / "tmp_1796_source.tif"
    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    to_geotiff(data, str(src), compression='none')

    vrt = tmp_path / "tmp_1796_source_cap.vrt"
    vrt.write_text(
        '<VRTDataset rasterXSize="1" rasterYSize="1">\n'
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{os.path.basename(src)}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="1" ySize="1"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )

    arr = read_vrt(str(vrt), max_pixels=1)
    assert arr.shape == (1, 1)


def test_source_cap_still_fires_when_sub_window_exceeds_budget(tmp_path):
    """The per-source pixel-budget guard still rejects a sub-window that
    exceeds ``max_pixels``. With the sub-window-bounded read, the cap is
    measured against the clipped destination region rather than the raw
    SrcRect; the protection from #1796 carries over to that new
    measurement.
    """
    src = tmp_path / "tmp_1796_big_source.tif"
    data = np.arange(64, dtype=np.uint8).reshape(8, 8)
    to_geotiff(data, str(src), compression='none', tiled=False)

    vrt = tmp_path / "tmp_1796_big_cap.vrt"
    vrt.write_text(
        '<VRTDataset rasterXSize="8" rasterYSize="8">\n'
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{os.path.basename(src)}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="8" ySize="8"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )

    with pytest.raises(ValueError, match="exceed|safety limit"):
        read_vrt(str(vrt), max_pixels=4)
