"""Regression tests for issue #1655.

``read_vrt`` used to evaluate the per-source NODATA fallback as
``src.nodata or nodata``. Python treats ``0.0`` as falsy, so a
SimpleSource that declared ``<NODATA>0</NODATA>`` was silently replaced
with the band-level sentinel (or ``None`` when the band had none of its
own). Pixels equal to 0.0 in the source file survived as valid data and
biased every downstream NaN-aware aggregation.

The fix changes the fallback to an explicit ``is not None`` check so a
legitimate zero sentinel survives.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._vrt import read_vrt
from xrspatial.geotiff._writer import write


def _write_source(tmp_path, arr, name='src_1655.tif'):
    """Write a small float32 GeoTIFF without a GDAL_NODATA tag."""
    p = str(tmp_path / name)
    write(
        arr, p,
        geo_transform=GeoTransform(
            origin_x=0.0, origin_y=0.0,
            pixel_width=1.0, pixel_height=-1.0,
        ),
        crs_epsg=4326,
        compression='none',
        tiled=False,
    )
    return p


def _vrt_with_source_nodata(tmp_path, src_path, nodata_xml,
                            include_band_nodata=False,
                            width=4, height=3,
                            band_nodata='0.0'):
    """Write a single-band Float32 VRT with the supplied ``<NODATA>``
    on its SimpleSource. ``include_band_nodata`` controls whether a
    ``<NoDataValue>`` is emitted on the band as well.
    """
    band_nd_elem = (
        f'<NoDataValue>{band_nodata}</NoDataValue>'
        if include_band_nodata else '')
    vrt_xml = (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'
        f'  <SRS>EPSG:4326</SRS>\n'
        f'  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
        f'  <VRTRasterBand dataType="Float32" band="1">\n'
        f'    {band_nd_elem}\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="0" yOff="0" '
        f'xSize="{width}" ySize="{height}"/>\n'
        f'      <DstRect xOff="0" yOff="0" '
        f'xSize="{width}" ySize="{height}"/>\n'
        f'      <NODATA>{nodata_xml}</NODATA>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    vrt_path = str(tmp_path / 'src_zero_1655.vrt')
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


class TestVRTSourceNodataZero:
    """SimpleSource ``<NODATA>0</NODATA>`` must mask zeros to NaN."""

    def test_source_nodata_zero_no_band_nodata(self, tmp_path):
        """SimpleSource NODATA=0 with no band-level fallback masks zeros."""
        arr = np.array(
            [[1.0, 0.0, 3.0, 0.0],
             [4.0, 0.0, 6.0, 7.0],
             [0.0, 8.0, 9.0, 10.0]],
            dtype=np.float32,
        )
        src = _write_source(tmp_path, arr)
        vrt = _vrt_with_source_nodata(tmp_path, src, '0.0')

        result, _ = read_vrt(vrt)
        assert int(np.isnan(result).sum()) == 4

    def test_source_nodata_zero_integer_xml(self, tmp_path):
        """``<NODATA>0</NODATA>`` (integer literal) also masks zeros."""
        arr = np.array(
            [[1.0, 0.0, 3.0]],
            dtype=np.float32,
        )
        src = _write_source(tmp_path, arr, name='int_xml.tif')
        vrt = _vrt_with_source_nodata(
            tmp_path, src, '0', width=3, height=1)

        result, _ = read_vrt(vrt)
        assert int(np.isnan(result).sum()) == 1
        assert np.isnan(result[0, 1])

    def test_source_nodata_nonzero_unchanged(self, tmp_path):
        """SimpleSource NODATA != 0 keeps masking behaviour."""
        arr = np.array(
            [[1.0, 0.0, 3.0, 0.0]],
            dtype=np.float32,
        )
        src = _write_source(tmp_path, arr, name='nonzero.tif')
        vrt = _vrt_with_source_nodata(
            tmp_path, src, '1.0', width=4, height=1)

        result, _ = read_vrt(vrt)
        # Only the literal 1.0 at [0, 0] should be masked.
        assert int(np.isnan(result).sum()) == 1
        assert np.isnan(result[0, 0])

    def test_band_nodata_zero_still_honoured(self, tmp_path):
        """Band-level ``<NoDataValue>0</NoDataValue>`` keeps working."""
        arr = np.array(
            [[1.0, 0.0, 3.0]],
            dtype=np.float32,
        )
        src = _write_source(tmp_path, arr, name='band_zero.tif')
        # Build a VRT where only the band carries nodata=0 (no NODATA
        # on the SimpleSource).
        vrt_xml = (
            f'<VRTDataset rasterXSize="3" rasterYSize="1">\n'
            f'  <SRS>EPSG:4326</SRS>\n'
            f'  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n'
            f'  <VRTRasterBand dataType="Float32" band="1">\n'
            f'    <NoDataValue>0.0</NoDataValue>\n'
            f'    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="0">{src}</SourceFilename>\n'
            f'      <SourceBand>1</SourceBand>\n'
            f'      <SrcRect xOff="0" yOff="0" xSize="3" ySize="1"/>\n'
            f'      <DstRect xOff="0" yOff="0" xSize="3" ySize="1"/>\n'
            f'    </SimpleSource>\n'
            f'  </VRTRasterBand>\n'
            f'</VRTDataset>\n'
        )
        vrt = str(tmp_path / 'band_zero_1655.vrt')
        with open(vrt, 'w') as f:
            f.write(vrt_xml)

        result, _ = read_vrt(vrt)
        assert int(np.isnan(result).sum()) == 1
        assert np.isnan(result[0, 1])

    def test_source_nodata_zero_overrides_band(self, tmp_path):
        """SimpleSource NODATA=0 takes precedence over band NoDataValue=99."""
        arr = np.array(
            [[1.0, 0.0, 99.0]],
            dtype=np.float32,
        )
        src = _write_source(tmp_path, arr, name='override.tif')
        vrt = _vrt_with_source_nodata(
            tmp_path, src, '0.0',
            include_band_nodata=True, band_nodata='99.0',
            width=3, height=1)

        result, _ = read_vrt(vrt)
        # The SimpleSource sentinel (0.0) wins over the band sentinel
        # (99.0), so only the 0.0 cell becomes NaN. The 99.0 cell stays
        # because the masking is per-source, applied at read time, and
        # the band-level fallback never fires when src.nodata is set.
        assert int(np.isnan(result).sum()) == 1
        assert np.isnan(result[0, 1])
        assert result[0, 2] == pytest.approx(99.0)
