"""Regression tests for issue #1611.

``read_vrt(path)`` on a multi-band integer VRT with per-band
``<NoDataValue>`` tags used to only mask band 0's sentinel. Bands 1
and up retained their integer sentinels as literal finite values in
the returned float64 array, breaking the convention that
``attrs['nodata']`` is present iff sentinel pixels are already NaN.

The float-VRT path masks per-band correctly in ``_vrt._read_data``
(lines 296-297, 347-351). For integer rasters the masking moved to
``__init__.py:read_vrt`` and used ``vrt.bands[0].nodata`` for every
band. The fix walks ``vrt.bands`` when ``band is None`` and masks
each ``arr[..., i]`` slice against its own ``<NoDataValue>``.

This file mirrors test_vrt_band_nodata_1598.py for the multi-band
``band=None`` path. PR #1602 fixed ``band=N`` single-band selection.

After #1987 PR 5 (mixed-band-metadata fail-closed), the fixtures here --
which mosaic bands with deliberately distinct per-band sentinels --
raise ``MixedBandMetadataError`` by default. Each call passes
``band_nodata='first'`` to assert the legacy flatten-to-first-band
behaviour is still reachable via the documented opt-out, which is the
exact semantics the regression covers.
"""
from __future__ import annotations

import numpy as np

from xrspatial.geotiff import read_vrt
from xrspatial.geotiff._writer import write


def _write_two_band_per_band_nodata_vrt(tmp_path, *, dtype_str="UInt16",
                                        np_dtype=np.uint16,
                                        band0_sentinel=65535,
                                        band1_sentinel=65000,
                                        band0_other=(1, 2, 3),
                                        band1_other=(7, 8, 9)):
    """Two single-band integer sources, each with a distinct nodata
    sentinel, exposed as bands 1 and 2 of a hand-rolled VRT.

    Used to be band 0's sentinel was the only one masked. Now every
    band gets its own sentinel.
    """
    b0_arr = np.array([[band0_other[0], band0_other[1]],
                       [band0_other[2], band0_sentinel]], dtype=np_dtype)
    b1_arr = np.array([[band1_other[0], band1_other[1]],
                       [band1_other[2], band1_sentinel]], dtype=np_dtype)
    p0 = str(tmp_path / 'vrt_b0_1611.tif')
    p1 = str(tmp_path / 'vrt_b1_1611.tif')
    write(b0_arr, p0, nodata=band0_sentinel, compression='none',
          tiled=False)
    write(b1_arr, p1, nodata=band1_sentinel, compression='none',
          tiled=False)

    vrt_path = str(tmp_path / 'two_band_per_band_nodata_1611.vrt')
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="{dtype_str}" band="1">
    <NoDataValue>{band0_sentinel}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="{dtype_str}" band="2">
    <NoDataValue>{band1_sentinel}</NoDataValue>
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


def test_multiband_uint16_per_band_sentinel_each_masked(tmp_path):
    """The previously-broken case: every band's sentinel must be NaN.

    Before the fix this returned dtype=float64 with band 0's (1,1) cell
    as NaN but band 1's (1,1) cell as the literal 65000.0.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band_nodata="first")
    assert r.shape == (2, 2, 2)
    assert r.dtype == np.float64, (
        f"expected float64 promotion, got {r.dtype}"
    )
    assert np.isnan(r.values[1, 1, 0]), (
        "band 0's sentinel pixel was not NaN-masked."
    )
    assert np.isnan(r.values[1, 1, 1]), (
        "band 1's sentinel pixel was not NaN-masked; "
        "the regression from issue #1611 has returned."
    )
    # Non-sentinel pixels survive unchanged
    assert r.values[0, 0, 0] == 1
    assert r.values[0, 0, 1] == 7
    assert r.values[1, 0, 0] == 3
    assert r.values[1, 0, 1] == 9


def test_multiband_int32_negative_per_band_sentinel(tmp_path):
    """Negative sentinels in a signed integer VRT also mask per-band.

    The original bug was dtype-independent: any integer dtype with
    per-band <NoDataValue> would have hit it. Cover int32 + negative
    sentinels to make sure the helper handles signed types and the
    range guard accepts negatives.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(
        tmp_path, dtype_str="Int32", np_dtype=np.int32,
        band0_sentinel=-9999, band1_sentinel=-7777,
        band0_other=(10, 20, 30), band1_other=(40, 50, 60))
    r = read_vrt(vrt_path, band_nodata="first")
    assert r.dtype == np.float64
    assert np.isnan(r.values[1, 1, 0])
    assert np.isnan(r.values[1, 1, 1])
    assert r.values[0, 0, 0] == 10
    assert r.values[0, 0, 1] == 40


def test_multiband_only_one_band_has_sentinel_present(tmp_path):
    """If only one band's sentinel actually appears in the data, only
    that band should change. The non-hitting band stays the same float64
    value (no spurious NaN introduced).

    Force band 1's sentinel never to appear by writing 99 instead.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(
        tmp_path,
        band0_sentinel=65535, band1_sentinel=65000,
        band1_other=(7, 8, 9))
    # Overwrite the band 1 source so the sentinel value 65000 is NOT
    # present in band 1's actual data.
    b1_no_sentinel = np.array([[7, 8], [9, 99]], dtype=np.uint16)
    import os
    p1 = os.path.join(os.path.dirname(vrt_path), 'vrt_b1_1611.tif')
    write(b1_no_sentinel, p1, nodata=65000, compression='none',
          tiled=False)

    r = read_vrt(vrt_path, band_nodata="first")
    assert r.dtype == np.float64, (
        "Even when only band 0 has a present sentinel, the array still "
        "needs promotion so band 0's NaN can be expressed."
    )
    assert np.isnan(r.values[1, 1, 0])
    assert r.values[1, 1, 1] == 99.0  # band 1 sentinel absent, no NaN


def test_multiband_no_sentinel_present_anywhere_keeps_int_dtype(tmp_path):
    """When no band actually contains its declared sentinel, skip
    promotion entirely. Avoids a needless float64 cast on integer data.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(
        tmp_path,
        band0_sentinel=65535, band1_sentinel=65000,
        band0_other=(1, 2, 3), band1_other=(7, 8, 9))
    # Replace BOTH source files so neither contains its sentinel
    import os
    b0 = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    b1 = np.array([[7, 8], [9, 10]], dtype=np.uint16)
    p0 = os.path.join(os.path.dirname(vrt_path), 'vrt_b0_1611.tif')
    p1 = os.path.join(os.path.dirname(vrt_path), 'vrt_b1_1611.tif')
    write(b0, p0, nodata=65535, compression='none', tiled=False)
    write(b1, p1, nodata=65000, compression='none', tiled=False)

    r = read_vrt(vrt_path, band_nodata="first")
    # Sentinels not present -> integer dtype preserved (matches the
    # eager open_geotiff fast-path which also skips promotion when the
    # mask is empty).
    assert r.dtype == np.uint16
    assert r.values[1, 1, 0] == 4
    assert r.values[1, 1, 1] == 10


def test_multiband_per_band_out_of_range_sentinel_is_no_op(tmp_path):
    """A sentinel out of the integer dtype's range should be a no-op
    for that band rather than raising. Mirrors PR #1583's behaviour
    (#1581): the helper ``_int_nodata_in_range`` gates the cast.
    """
    # uint16 cannot represent -9999; the helper should skip band 1.
    vrt_path = _write_two_band_per_band_nodata_vrt(
        tmp_path,
        dtype_str="UInt16", np_dtype=np.uint16,
        band0_sentinel=65535, band1_sentinel=10,  # placeholder; rewrite below
        band0_other=(1, 2, 3), band1_other=(7, 8, 9))

    # Hand-rewrite the VRT so band 1's NoDataValue is the out-of-range -9999.
    with open(vrt_path) as f:
        xml = f.read()
    xml = xml.replace("<NoDataValue>10</NoDataValue>",
                      "<NoDataValue>-9999</NoDataValue>")
    with open(vrt_path, 'w') as f:
        f.write(xml)

    # Should not raise and should still mask band 0.
    r = read_vrt(vrt_path, band_nodata="first")
    assert np.isnan(r.values[1, 1, 0])  # band 0 sentinel still masked
    # Band 1's sentinel (-9999) is out of uint16 range; the value 10
    # in band1[1,1] survives unchanged.
    assert r.values[1, 1, 1] == 10.0 or r.values[1, 1, 1] == 10


def test_multiband_band_kwarg_still_per_band_post_pr1602(tmp_path):
    """Non-regression check that PR #1602's band=N path still works.

    The fix here only changes the ``band is None`` branch; ``band=N``
    must still route through the single-band masking with its own
    sentinel.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    r0 = read_vrt(vrt_path, band=0, band_nodata="first")
    r1 = read_vrt(vrt_path, band=1, band_nodata="first")
    assert r0.dtype == np.float64
    assert r1.dtype == np.float64
    assert r0.attrs.get('nodata') == 65535.0
    assert r1.attrs.get('nodata') == 65000.0
    assert np.isnan(r0.values[1, 1])
    assert np.isnan(r1.values[1, 1])


def test_multiband_attrs_nodata_still_band0(tmp_path):
    """``attrs['nodata']`` for band=None reads is documented as band
    0's sentinel (the canonical attr cannot encode per-band values).
    The pixel-level fix must not change that contract.
    """
    vrt_path = _write_two_band_per_band_nodata_vrt(tmp_path)
    r = read_vrt(vrt_path, band_nodata="first")
    assert r.attrs.get('nodata') == 65535.0
