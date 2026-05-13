"""VRT source reads must honor the caller's max_pixels budget (#1796)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff, read_vrt


def test_vrt_source_read_forwards_max_pixels(tmp_path):
    """A tiny VRT output cannot force an oversized source-window decode."""
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

    with pytest.raises(ValueError, match="exceed"):
        read_vrt(str(vrt), max_pixels=1)

