"""Regression tests for issue #1616.

A VRT whose ``<VRTRasterBand dataType="Float32">`` (or Float64) is fed
by an integer source GeoTIFF with an in-range ``GDAL_NODATA`` sentinel
used to leak the sentinel value through as a literal float in the
returned array. ``attrs['nodata']`` advertised the sentinel but the
pixels at the sentinel positions still held the integer value cast to
float (e.g. ``65535.0`` rather than ``np.nan``). NaN-aware downstream
code therefore saw the sentinel as valid data.

The fix masks integer source arrays before they're placed into a float
``result`` buffer in ``_vrt.read_vrt`` so a float-dtype VRT result lands
with NaN at the sentinel pixels, matching what ``open_geotiff`` returns
for a single-file integer raster with the same sentinel.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import read_vrt
from xrspatial.geotiff._writer import write


def _write_uint16_with_sentinel(tmp_path, sentinel=65535, filename='b0.tif'):
    band = np.array([[1, 2], [3, sentinel]], dtype=np.uint16)
    p = str(tmp_path / filename)
    write(band, p, nodata=sentinel, compression='none', tiled=False)
    return p


def _write_int16_with_sentinel(tmp_path, sentinel=-1, filename='b0.tif'):
    band = np.array([[1, 2], [3, sentinel]], dtype=np.int16)
    p = str(tmp_path / filename)
    write(band, p, nodata=sentinel, compression='none', tiled=False)
    return p


def _build_vrt(tmp_path, source_path, vrt_dtype, nodata_value,
               filename='mismatch.vrt'):
    """Hand-roll a VRT with the requested dataType / NoDataValue pair."""
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="{vrt_dtype}" band="1">
    <NoDataValue>{nodata_value}</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{source_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    p = str(tmp_path / filename)
    with open(p, 'w') as f:
        f.write(vrt_xml)
    return p


def test_float32_vrt_uint16_source_masks_in_range_sentinel(tmp_path):
    """Float32 VRT, uint16 source with in-range sentinel: pixel becomes NaN.

    Before the fix this returned dtype=float32 with values[1, 1] == 65535.0
    while ``attrs['nodata']`` advertised the sentinel.
    """
    src = _write_uint16_with_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float32', 65535)
    r = read_vrt(vrt)
    assert r.dtype == np.float32, (
        f"Float32-declared VRT should return float32, got {r.dtype}")
    assert np.isnan(r.values[1, 1]), (
        "Sentinel pixel (uint16 65535 -> float32) should be NaN-masked; "
        f"got values[1, 1]={r.values[1, 1]}")
    assert r.attrs.get('nodata') == 65535.0
    assert r.values[0, 0] == 1.0


def test_float64_vrt_int16_source_masks_negative_sentinel(tmp_path):
    """Float64 VRT, int16 source with negative sentinel: pixel becomes NaN."""
    src = _write_int16_with_sentinel(tmp_path, sentinel=-1)
    vrt = _build_vrt(tmp_path, src, 'Float64', -1)
    r = read_vrt(vrt)
    assert r.dtype == np.float64
    assert np.isnan(r.values[1, 1]), (
        f"Sentinel pixel (-1) should be NaN-masked; "
        f"got values[1, 1]={r.values[1, 1]}")
    assert r.attrs.get('nodata') == -1.0


def test_float32_vrt_out_of_range_sentinel_is_noop(tmp_path):
    """An out-of-range sentinel (e.g. uint16 source + NoDataValue=-9999)
    stays unmasked rather than raising ``OverflowError`` from the
    ``uint16(-9999)`` cast. The pixel data is returned as-is and
    ``attrs['nodata']`` still surfaces the declared sentinel so callers
    can mask in user code or write through.
    """
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    p = str(tmp_path / 'b0_no_nodata.tif')
    write(arr, p, compression='none', tiled=False)  # no GDAL_NODATA on file
    vrt = _build_vrt(tmp_path, p, 'Float32', -9999)
    r = read_vrt(vrt)
    assert r.dtype == np.float32
    # No pixels match the (out-of-range) sentinel, so nothing was masked.
    assert not np.isnan(r.values).any()
    assert r.attrs.get('nodata') == -9999.0


def test_float32_vrt_uint16_source_no_sentinel_pixels(tmp_path):
    """Float32 VRT, uint16 source whose pixels do not match the sentinel:
    the result is a clean float array with no NaNs introduced.

    This exercises the early-out path inside the new mask branch -- a
    declared sentinel that matches no pixels must not perturb the data
    or cause an extra copy that would surface as a different dtype.
    """
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    p = str(tmp_path / 'b0_clean.tif')
    write(arr, p, compression='none', tiled=False)
    vrt = _build_vrt(tmp_path, p, 'Float32', 65535)
    r = read_vrt(vrt)
    assert r.dtype == np.float32
    assert not np.isnan(r.values).any()
    np.testing.assert_array_equal(r.values, arr.astype(np.float32))


def test_float_vrt_int_source_dask_path_masks_sentinel(tmp_path):
    """The dask wrapper path (``chunks=...``) also returns NaN at the
    sentinel pixel. The dask reader chunks the eager result after decode,
    so the bug propagates if the eager path leaks the sentinel.
    """
    src = _write_uint16_with_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float32', 65535)
    r = read_vrt(vrt, chunks=2)
    # Dask path keeps the float32 dtype declared by the VRT.
    assert r.dtype == np.float32
    val = r.values
    assert np.isnan(val[1, 1])


def test_float_vrt_int_source_round_trip_nodata_attr(tmp_path):
    """Even though the masking promotes pixels to NaN, the
    ``attrs['nodata']`` value still carries the original sentinel so a
    downstream write can restore the literal sentinel byte pattern.
    """
    src = _write_uint16_with_sentinel(tmp_path)
    vrt = _build_vrt(tmp_path, src, 'Float32', 65535)
    r = read_vrt(vrt)
    assert r.attrs.get('nodata') == 65535.0


def test_float_vrt_int_source_with_band_select(tmp_path):
    """The band=N selection path also masks integer sentinels for a
    float-declared VRT. The per-band ``NoDataValue`` from the VRT XML
    must reach the source-side masking step, not just ``attrs['nodata']``.
    """
    src_a = _write_uint16_with_sentinel(tmp_path, sentinel=65535,
                                          filename='ba.tif')
    src_b = _write_uint16_with_sentinel(tmp_path, sentinel=65000,
                                          filename='bb.tif')
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <NoDataValue>65535</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_a}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Float32" band="2">
    <NoDataValue>65000</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_b}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = str(tmp_path / 'mb.vrt')
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)

    # band 0 -> 65535 sentinel masked
    r0 = read_vrt(vrt_path, band=0)
    assert r0.dtype == np.float32
    assert np.isnan(r0.values[1, 1])
    assert r0.attrs.get('nodata') == 65535.0

    # band 1 -> 65000 sentinel masked, not 65535
    r1 = read_vrt(vrt_path, band=1)
    assert r1.dtype == np.float32
    # band b had its sentinel at the same [1, 1] cell
    assert np.isnan(r1.values[1, 1])
    assert r1.attrs.get('nodata') == 65000.0
