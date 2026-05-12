"""Regression tests for issue #1694.

``read_vrt`` did not resample source pixel data when a source band's
``<SrcRect>`` size differed from its ``<DstRect>`` size.  Downsampling
raised ``ValueError: could not broadcast input array from shape (S,S)
into shape (D,D)`` and upsampling silently left holes -- only the
top-left ``sr.x_size``/``sr.y_size`` pixels of each destination cell
were written.

The fix:

* when ``sr.size != dr.size``, read the full source rect, apply nodata
  masking, resample to ``(dr.y_size, dr.x_size)`` with nearest-neighbour
  (matching GDAL's SimpleSource semantics), and then clip;
* the same-size case still uses windowed reads as before.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._vrt import _resample_nearest, read_vrt
from xrspatial.geotiff._writer import write


def _write_vrt(tmp_path, xml: str, name: str = 'test.vrt') -> str:
    p = str(tmp_path / name)
    with open(p, 'w') as f:
        f.write(xml)
    return p


def test_downsample_4x4_to_2x2_does_not_raise_and_uses_nearest(tmp_path):
    """SrcRect 4x4 -> DstRect 2x2: result is (2,2), nearest-neighbour.

    Before the fix the source (4,4) array was assigned directly into the
    (2,2) destination slice, raising the broadcast error documented in
    issue #1694.
    """
    src = np.arange(16, dtype=np.uint16).reshape(4, 4)
    src_path = str(tmp_path / 'src.tif')
    write(src, src_path, compression='none', tiled=False)

    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, vrt_xml, 'down.vrt')

    result, _ = read_vrt(vrt_path)

    assert result.shape == (2, 2), (
        f"expected (2,2), got {result.shape}; resample step missing."
    )
    # Nearest-neighbour with the centre-of-output-pixel rule samples
    # source indices floor((i+0.5)*4/2) = floor((i+0.5)*2) -> 1, 3 for
    # i=0, 1.  So we expect src[1, 1], src[1, 3], src[3, 1], src[3, 3].
    expected = np.array([[src[1, 1], src[1, 3]],
                         [src[3, 1], src[3, 3]]], dtype=np.uint16)
    np.testing.assert_array_equal(result, expected)


def test_upsample_2x2_to_4x4_repeats_each_source_pixel(tmp_path):
    """SrcRect 2x2 -> DstRect 4x4: each source pixel repeated 2x2.

    Before the fix only the top-left 2x2 of the destination was written
    and the rest stayed at the fill value (0 for integer, NaN for
    float).
    """
    src = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    src_path = str(tmp_path / 'src.tif')
    write(src, src_path, compression='none', tiled=False)

    vrt_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, vrt_xml, 'up.vrt')

    result, _ = read_vrt(vrt_path)

    assert result.shape == (4, 4)
    expected = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.uint16)
    np.testing.assert_array_equal(result, expected)
    # No holes -- every cell was written.
    assert not (result == 0).any(), (
        "upsample left zero-filled cells; resample not propagated."
    )


def test_non_integer_scale_3x3_to_2x2_no_holes(tmp_path):
    """Non-integer source / destination ratio: covers index-mapping path.

    With src=(3,3) -> dst=(2,2), neither integer-ratio fast path applies.
    Confirms the general nearest-neighbour gather produces the correct
    shape, no holes, no out-of-bounds writes.
    """
    src = np.arange(9, dtype=np.uint16).reshape(3, 3)
    src_path = str(tmp_path / 'src.tif')
    write(src, src_path, compression='none', tiled=False)

    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.5, 0.0, 0.0, 0.0, -1.5</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="3" ySize="3"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, vrt_xml, 'nonint.vrt')

    result, _ = read_vrt(vrt_path)

    assert result.shape == (2, 2)
    # Nearest mapping: floor((i+0.5) * 3/2) = floor((i+0.5)*1.5)
    #   i=0 -> floor(0.75) = 0
    #   i=1 -> floor(2.25) = 2
    # So output samples src[0,0], src[0,2], src[2,0], src[2,2].
    expected = np.array([[src[0, 0], src[0, 2]],
                         [src[2, 0], src[2, 2]]], dtype=np.uint16)
    np.testing.assert_array_equal(result, expected)


def test_per_band_scale_mix(tmp_path):
    """Mixed: band 1 downsampled, band 2 at native resolution.

    Both bands must land in the right places without a broadcast error
    and without bleeding band 1's resampled values into band 2.
    """
    # Band 1 source: 4x4 -- will be downsampled to 2x2 destination.
    band1_src = (np.arange(16, dtype=np.uint16) * 10).reshape(4, 4)
    # Band 2 source: 2x2 -- native resolution.
    band2_src = np.array([[100, 200], [300, 400]], dtype=np.uint16)

    p1 = str(tmp_path / 'b1.tif')
    p2 = str(tmp_path / 'b2.tif')
    write(band1_src, p1, compression='none', tiled=False)
    write(band2_src, p2, compression='none', tiled=False)

    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p2}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, vrt_xml, 'mix.vrt')

    result, _ = read_vrt(vrt_path)

    assert result.shape == (2, 2, 2)
    # Band 1 nearest-neighbour from 4x4 -> 2x2: src[1,1], src[1,3], src[3,1], src[3,3]
    expected_b1 = np.array([[band1_src[1, 1], band1_src[1, 3]],
                            [band1_src[3, 1], band1_src[3, 3]]],
                           dtype=np.uint16)
    np.testing.assert_array_equal(result[..., 0], expected_b1)
    # Band 2 native: untouched.
    np.testing.assert_array_equal(result[..., 1], band2_src)


def test_window_on_downsampled_source_returns_correct_subwindow(tmp_path):
    """``window=(0,0,1,1)`` on a 4x4 -> 2x2 source returns the (0,0) cell.

    The destination cell maps to the source pixel that the resample
    routine would sample for that location.  Confirms the clip-after-
    resample ordering: clipping in source coordinates first (as the old
    code effectively did) would feed the wrong source slice into the
    resampler.
    """
    src = np.arange(16, dtype=np.uint16).reshape(4, 4)
    src_path = str(tmp_path / 'src.tif')
    write(src, src_path, compression='none', tiled=False)

    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, vrt_xml, 'win.vrt')

    result, _ = read_vrt(vrt_path, window=(0, 0, 1, 1))

    assert result.shape == (1, 1)
    # Full-resample value at destination (0,0) is src[1,1] == 5.
    assert result[0, 0] == src[1, 1]


def test_nodata_preserved_across_downsample(tmp_path):
    """Source sentinel pixels survive the resample as NaN in the result.

    Source is uint16 with sentinel=65535.  Pixels at the sampled-from
    positions whose values are 65535 must appear as NaN in the float64
    VRT output.
    """
    sentinel = np.uint16(65535)
    src = np.array([
        [10, 20, 30, 40],
        [50, sentinel, 70, sentinel],
        [90, 100, 110, 120],
        [130, sentinel, 150, sentinel],
    ], dtype=np.uint16)
    src_path = str(tmp_path / 'src_nd.tif')
    write(src, src_path, nodata=int(sentinel), compression='none',
          tiled=False)

    # Float64 VRT so the integer-into-float promotion path runs and
    # leaves NaN at the sentinel pixels.  Use ``<NODATA>`` on the source
    # so the masking branch fires regardless of band-level <NoDataValue>.
    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>
  <VRTRasterBand dataType="Float64" band="1">
    <NoDataValue>-9999</NoDataValue>
    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <NODATA>65535</NODATA>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, vrt_xml, 'nd.vrt')

    result, _ = read_vrt(vrt_path)

    assert result.shape == (2, 2)
    assert result.dtype == np.float64
    # Nearest sampler picks src[1,1], src[1,3], src[3,1], src[3,3].
    # Of those, src[1,1]=65535, src[1,3]=65535, src[3,1]=65535,
    # src[3,3]=65535 -- so every output pixel is the sentinel and must
    # be NaN after masking.
    assert np.isnan(result).all(), (
        f"sentinel did not survive resample as NaN; got {result!r}"
    )


def test_nodata_with_mixed_sentinel_and_valid_pixels(tmp_path):
    """Mixed sentinel / valid source -> mixed NaN / valid destination.

    Confirms the mask resamples *with* the data, not against the
    pre-resampled source.
    """
    sentinel = np.uint16(65535)
    # Build a 4x4 source where sample sites (1,1), (1,3), (3,1), (3,3)
    # are valid, sentinel, valid, sentinel respectively.
    src = np.zeros((4, 4), dtype=np.uint16)
    src[1, 1] = 11
    src[1, 3] = sentinel
    src[3, 1] = 31
    src[3, 3] = sentinel
    src_path = str(tmp_path / 'src_mixed.tif')
    write(src, src_path, nodata=int(sentinel), compression='none',
          tiled=False)

    vrt_xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>
  <VRTRasterBand dataType="Float64" band="1">
    <NoDataValue>-9999</NoDataValue>
    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <NODATA>65535</NODATA>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, vrt_xml, 'nd_mixed.vrt')

    result, _ = read_vrt(vrt_path)

    assert result.shape == (2, 2)
    assert result[0, 0] == 11.0
    assert np.isnan(result[0, 1])
    assert result[1, 0] == 31.0
    assert np.isnan(result[1, 1])


@pytest.mark.parametrize('shape', [(0, 5), (5, 0), (0, 0)])
def test_resample_nearest_rejects_empty_source(shape):
    """``_resample_nearest`` raises ValueError on an empty source array.

    A SimpleSource with ``SrcRect xSize=0`` or ``ySize=0`` -- or a
    windowed read that clamps to an empty slice -- would otherwise feed
    a zero-dim array to the integer-ratio fast paths, which compute
    ``out_h % src_h`` and divide by ``src_h``/``src_w`` and so would
    raise an opaque ``ZeroDivisionError``.  Surface the bad input with
    a clear ``ValueError`` instead.
    """
    src_arr = np.zeros(shape, dtype=np.float64)
    with pytest.raises(ValueError, match='empty source array'):
        _resample_nearest(src_arr, 2, 2)
