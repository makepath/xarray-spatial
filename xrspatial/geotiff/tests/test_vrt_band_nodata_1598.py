"""Regression tests for issue #1598.

``read_vrt(path, band=N)`` used to always source the nodata sentinel
from ``vrt.bands[0]`` rather than the requested band, so a multi-band
VRT with per-band ``<NoDataValue>`` would mis-mask reads of any band
other than band 0:

* ``attrs['nodata']`` advertised band 0's sentinel (wrong).
* The integer-to-float64 promotion ran against band 0's sentinel, so
  band N's actual nodata pixels stayed as literal integers.
* The returned dtype was integer when it should have been float64.

The fix uses ``vrt.bands[band].nodata`` when a band is selected.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import read_vrt
from xrspatial.geotiff._writer import write


def _write_two_band_per_band_nodata_vrt(tmp_path):
    """Two single-band uint16 sources, each with a distinct nodata
    sentinel, exposed as bands 1 and 2 of a hand-rolled VRT.
    """
    band0 = np.array([[1, 2], [3, 65535]], dtype=np.uint16)
    band1 = np.array([[7, 8], [9, 65000]], dtype=np.uint16)
    p0 = str(tmp_path / 'vrt_band0_1598.tif')
    p1 = str(tmp_path / 'vrt_band1_1598.tif')
    write(band0, p0, nodata=65535, compression='none', tiled=False)
    write(band1, p1, nodata=65000, compression='none', tiled=False)

    vrt_path = str(tmp_path / 'two_band_per_band_nodata_1598.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>65535</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <NoDataValue>65000</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def test_read_vrt_band0_uses_band0_nodata(tmp_path):
    """Sanity check the band-0 selection still works after the fix.

    Confirms the refactor did not flip the index.

    The fixture mosaics two bands with distinct per-band sentinels, so
    after #1987 PR 5 the default read raises ``MixedBandMetadataError``.
    The pre-#1987 flatten-to-first-band semantics this regression tests
    are still reachable via ``band_nodata='first'``; the opt-in surfaces
    at the call site that the test is exercising the legacy behaviour.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band=0, band_nodata='first')
    assert r.dtype == np.float64
    assert r.attrs.get('nodata') == 65535.0
    assert np.isnan(r.values[1, 1])
    assert r.values[0, 0] == 1


def test_read_vrt_band1_uses_band1_nodata(tmp_path):
    """The previously-broken case: band=1 must use band 1's sentinel.

    Before the fix this returned dtype=uint16 with values=[[7,8],
    [9,65000]] and attrs['nodata']=65535.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band=1, band_nodata='first')
    assert r.dtype == np.float64, (
        "band=1 read kept uint16 dtype; per-band nodata regression."
    )
    assert r.attrs.get('nodata') == 65000.0, (
        f"attrs['nodata'] was {r.attrs.get('nodata')}, "
        f"expected 65000 from band 1's <NoDataValue>."
    )
    assert np.isnan(r.values[1, 1]), (
        "band 1's sentinel pixel was not NaN-masked; "
        "promotion ran against the wrong sentinel."
    )
    assert r.values[0, 0] == 7
    assert r.values[1, 0] == 9


def test_read_vrt_no_band_keeps_band0_nodata_attr(tmp_path):
    """Unselected reads still surface band 0's sentinel.

    Multi-band VRTs with mixed sentinels return all bands stacked, and
    the canonical attr cannot encode per-band values; advertising
    band 0's sentinel matches the prior behavior and the documented
    "first band wins" contract for multi-band reads.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band_nodata='first')
    assert r.attrs.get('nodata') == 65535.0


def test_read_vrt_negative_band_raises(tmp_path):
    """Negative band indices used to be silently accepted via Python
    list indexing (``vrt.bands[-1]`` returned the last band) while the
    public reader's nodata lookup rejected them, producing band-N data
    with no nodata sentinel. They are now a clear ValueError up front.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    with pytest.raises(ValueError, match="band"):
        read_vrt(vrt_path, band=-1)


def test_read_vrt_out_of_range_band_raises(tmp_path):
    """Out-of-range band indices used to raise IndexError from deep in
    the read path. They are now a ValueError that names the available
    band count.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        read_vrt(vrt_path, band=5, band_nodata='first')


def test_read_vrt_non_integer_band_raises(tmp_path):
    """A non-int ``band`` would previously have raised TypeError on the
    list index. ValueError here matches the rest of the input
    validation surface.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    with pytest.raises(ValueError, match="band"):
        read_vrt(vrt_path, band="1")
    with pytest.raises(ValueError, match="band"):
        read_vrt(vrt_path, band=True)
