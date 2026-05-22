"""VRT missing-source handling has an explicit policy (#1799)."""
from __future__ import annotations

import pytest

from xrspatial.geotiff import GeoTIFFFallbackWarning, read_vrt


def _write_missing_source_vrt(path):
    path.write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2">\n'
        '  <VRTRasterBand dataType="Byte" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">missing.tif'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )


def test_read_vrt_missing_sources_warns_and_records_hole(tmp_path):
    vrt = tmp_path / "tmp_1799_missing_warn.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.warns(GeoTIFFFallbackWarning, match="could not be read"):
        da = read_vrt(str(vrt), missing_sources='warn')

    assert 'vrt_holes' in da.attrs
    assert da.attrs['vrt_holes'][0]['source'].endswith('missing.tif')


def test_read_vrt_missing_sources_raise_fails_fast(tmp_path):
    vrt = tmp_path / "tmp_1799_missing_raise.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.raises((OSError, ValueError)):
        read_vrt(str(vrt), missing_sources='raise')


def test_read_vrt_missing_sources_validates_policy(tmp_path):
    vrt = tmp_path / "tmp_1799_missing_bad_policy.vrt"
    _write_missing_source_vrt(vrt)

    with pytest.raises(ValueError, match="missing_sources"):
        read_vrt(str(vrt), missing_sources='ignore')
