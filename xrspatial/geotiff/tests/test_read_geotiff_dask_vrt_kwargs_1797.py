"""Direct read_geotiff_dask(.vrt) must forward VRT kwargs (#1797)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff, read_geotiff_dask


def _write_vrt(vrt_path, source_name, *, bands=1):
    band_xml = []
    for i in range(bands):
        band_xml.append(
            f'  <VRTRasterBand dataType="Float32" band="{i + 1}">\n'
            '    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="1">{source_name}'
            '</SourceFilename>\n'
            f'      <SourceBand>{i + 1}</SourceBand>\n'
            '      <SrcRect xOff="0" yOff="0" xSize="6" ySize="4"/>\n'
            '      <DstRect xOff="0" yOff="0" xSize="6" ySize="4"/>\n'
            '    </SimpleSource>\n'
            '  </VRTRasterBand>\n'
        )
    vrt_path.write_text(
        '<VRTDataset rasterXSize="6" rasterYSize="4">\n'
        + ''.join(band_xml)
        + '</VRTDataset>\n'
    )


def test_direct_read_geotiff_dask_vrt_forwards_window_and_band(tmp_path):
    arr = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    src = tmp_path / "tmp_1797_source.tif"
    to_geotiff(arr, str(src), compression='none')
    vrt = tmp_path / "tmp_1797_source.vrt"
    _write_vrt(vrt, os.path.basename(src), bands=2)

    got = read_geotiff_dask(
        str(vrt), chunks=2, window=(1, 2, 4, 6), band=1,
    )

    assert got.shape == (3, 4)
    np.testing.assert_array_equal(got.values, arr[1:4, 2:6, 1])


def test_direct_read_geotiff_dask_vrt_forwards_max_pixels(tmp_path):
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    src = tmp_path / "tmp_1797_source_cap.tif"
    to_geotiff(arr, str(src), compression='none')
    vrt = tmp_path / "tmp_1797_source_cap.vrt"
    _write_vrt(vrt, os.path.basename(src))

    with pytest.raises(ValueError, match="exceed"):
        read_geotiff_dask(str(vrt), chunks=2, max_pixels=10)
