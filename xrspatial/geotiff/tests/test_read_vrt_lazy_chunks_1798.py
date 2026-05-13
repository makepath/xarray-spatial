"""read_vrt(chunks=...) should build lazy window tasks (#1798)."""
from __future__ import annotations

import os
import warnings

import numpy as np

from xrspatial.geotiff import to_geotiff, read_vrt


def _write_vrt(vrt_path, source_name):
    vrt_path.write_text(
        '<VRTDataset rasterXSize="6" rasterYSize="4">\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{source_name}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="6" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="6" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_read_vrt_chunks_matches_eager_values(tmp_path):
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    src = tmp_path / "tmp_1798_source.tif"
    to_geotiff(arr, str(src), compression='none')
    vrt = tmp_path / "tmp_1798_source.vrt"
    _write_vrt(vrt, os.path.basename(src))

    eager = read_vrt(str(vrt))
    lazy = read_vrt(str(vrt), chunks=2)

    assert lazy.data.chunks == ((2, 2), (2, 2, 2))
    np.testing.assert_array_equal(lazy.compute().values, eager.values)


def test_read_vrt_chunks_does_not_read_sources_during_construction(tmp_path):
    vrt = tmp_path / "tmp_1798_missing_source.vrt"
    _write_vrt(vrt, "missing.tif")

    with warnings.catch_warnings(record=True) as caught:
        lazy = read_vrt(str(vrt), chunks=2)

    assert caught == []
    assert hasattr(lazy.data, 'compute')

