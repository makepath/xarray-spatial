"""Regression tests for issue #1696.

``read_vrt`` on a multi-band VRT used to allocate the output buffer from
``selected_bands[0].dtype`` only. Each band's source array was then
placed into the buffer via ``result[..., band_idx] = src_arr[...]``,
which silently casts to the LHS dtype. So a ``Byte`` band 0 followed by
a ``Float32`` band 1 returned uint8 output, with band 1's fractional
values truncated.

The ``ComplexSource`` ``ScaleRatio`` / ``ScaleOffset`` path made this
worse. The decoded source is explicitly promoted to ``float64`` in the
``# Apply ComplexSource scaling`` block of ``read_vrt`` in ``_vrt.py``
(``src_arr.astype(np.float64) * src.scale``), but the destination
buffer stays uint8 if all VRT bands declare ``Byte``, so the
post-scale fractional values are lost on assignment.

The fix computes the common result dtype across all selected bands,
applying the same float64 promotion rule used for the scaled source
arrays, before allocating ``result``. The single-band branch uses the
same logic.
"""
from __future__ import annotations

import numpy as np

from xrspatial.geotiff import read_vrt
from xrspatial.geotiff._writer import write


def _write(arr, path, **kw):
    """Write a 2D array to ``path`` with sensible defaults for tests."""
    write(arr, str(path), compression='none', tiled=False, **kw)


def _build_two_band_vrt(tmp_path, *, b0_dtype_str, b0_path, b1_dtype_str,
                        b1_path, b1_extra="", b0_extra="",
                        filename='mb.vrt', size=2):
    """Hand-roll a two-band VRT with arbitrary dataType strings."""
    vrt_xml = f"""<VRTDataset rasterXSize="{size}" rasterYSize="{size}">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="{b0_dtype_str}" band="1">
{b0_extra}    <SimpleSource>
      <SourceFilename relativeToVRT="0">{b0_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>
      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="{b1_dtype_str}" band="2">
{b1_extra}    <SimpleSource>
      <SourceFilename relativeToVRT="0">{b1_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>
      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    p = tmp_path / filename
    p.write_text(vrt_xml)
    return str(p)


def _build_complex_source_vrt(tmp_path, *, dtype_str, src_path,
                              scale_ratio=None, scale_offset=None,
                              filename='cs.vrt', size=2,
                              band_num=2, other_band_dtype="Byte",
                              other_band_path=None,
                              extra_band=True):
    """Hand-roll a VRT where band 2 (or the only band) uses ComplexSource.

    ``extra_band=False`` writes a single-band VRT.
    """
    cs_lines = []
    if scale_ratio is not None:
        cs_lines.append(f"      <ScaleRatio>{scale_ratio}</ScaleRatio>")
    if scale_offset is not None:
        cs_lines.append(f"      <ScaleOffset>{scale_offset}</ScaleOffset>")
    cs_inner = "\n".join(cs_lines)

    complex_block = f"""    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
{cs_inner}
      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>
      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>
    </ComplexSource>
"""

    if extra_band and other_band_path is not None:
        vrt_xml = f"""<VRTDataset rasterXSize="{size}" rasterYSize="{size}">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="{other_band_dtype}" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{other_band_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>
      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="{dtype_str}" band="{band_num}">
{complex_block}  </VRTRasterBand>
</VRTDataset>"""
    else:
        vrt_xml = f"""<VRTDataset rasterXSize="{size}" rasterYSize="{size}">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="{dtype_str}" band="1">
{complex_block}  </VRTRasterBand>
</VRTDataset>"""
    p = tmp_path / filename
    p.write_text(vrt_xml)
    return str(p)


# ---------------------------------------------------------------------------
# 1. Mixed-dtype bands: Byte + Float32 must not truncate band 1
# ---------------------------------------------------------------------------

def test_mixed_byte_and_float32_bands_preserve_fractional(tmp_path):
    """``Byte`` band 0 + ``Float32`` band 1: band 1's fractional values
    must survive the read. Before the fix the buffer was allocated as
    uint8 and ``1.5, 2.5, 3.5, 4.5`` truncated to ``1, 2, 3, 4``.
    """
    b0 = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    b1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _write(b0, p0)
    _write(b1, p1)

    vrt_path = _build_two_band_vrt(
        tmp_path,
        b0_dtype_str='Byte', b0_path=str(p0),
        b1_dtype_str='Float32', b1_path=str(p1),
    )
    r = read_vrt(vrt_path)
    # numpy 2.x: result_type(uint8, float32) == float32; numpy 1.x: float64.
    # Either is acceptable; both express the float values losslessly.
    assert r.dtype.kind == 'f', (
        f"Mixed Byte+Float32 must widen to float; got {r.dtype}"
    )
    np.testing.assert_allclose(r.values[..., 1], b1.astype(r.dtype))
    np.testing.assert_array_equal(r.values[..., 0], b0.astype(r.dtype))


# ---------------------------------------------------------------------------
# 2. ComplexSource ScaleRatio forces float promotion of the buffer
# ---------------------------------------------------------------------------

def test_complex_source_scale_promotes_buffer_to_float(tmp_path):
    """Both bands declare ``Byte`` but band 1 has ``<ScaleRatio>0.5</ScaleRatio>``.
    The scaled source values include fractional results (11 * 0.5 = 5.5)
    which must survive. Before the fix the buffer stayed uint8 and the
    fractional values rounded down to 5.
    """
    b = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _write(b, p0)
    _write(b, p1)

    vrt_path = _build_complex_source_vrt(
        tmp_path,
        dtype_str='Byte', src_path=str(p1),
        scale_ratio=0.5,
        other_band_dtype='Byte', other_band_path=str(p0),
    )
    r = read_vrt(vrt_path)
    assert r.dtype.kind == 'f', (
        f"ScaleRatio on a Byte band must widen the buffer to float; "
        f"got {r.dtype}"
    )
    expected = b.astype(np.float64) * 0.5
    np.testing.assert_allclose(r.values[..., 1], expected)
    # Band 0 (unscaled Byte) survives losslessly through the wider dtype.
    np.testing.assert_array_equal(
        r.values[..., 0].astype(np.uint8), b
    )


# ---------------------------------------------------------------------------
# 3. All-Byte bands with no scaling stay uint8 (no needless widening)
# ---------------------------------------------------------------------------

def test_all_byte_no_scaling_stays_uint8(tmp_path):
    """Two ``Byte`` bands with no ``ComplexSource`` scaling: the result
    must stay uint8 (memory regression guard). The fix must not widen
    unconditionally to float64.
    """
    b0 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    b1 = np.array([[50, 60], [70, 80]], dtype=np.uint8)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _write(b0, p0)
    _write(b1, p1)

    vrt_path = _build_two_band_vrt(
        tmp_path,
        b0_dtype_str='Byte', b0_path=str(p0),
        b1_dtype_str='Byte', b1_path=str(p1),
    )
    r = read_vrt(vrt_path)
    assert r.dtype == np.uint8, (
        f"All-Byte VRT with no scaling must stay uint8; got {r.dtype}"
    )
    np.testing.assert_array_equal(r.values[..., 0], b0)
    np.testing.assert_array_equal(r.values[..., 1], b1)


# ---------------------------------------------------------------------------
# 4. Per-band ScaleRatio + ScaleOffset combinations preserve precision
# ---------------------------------------------------------------------------

def test_complex_source_scale_and_offset_preserve_precision(tmp_path):
    """``ScaleRatio=0.25`` plus ``ScaleOffset=1.5`` on a uint8 band:
    the scaled-and-offset values (e.g. ``10 * 0.25 + 1.5 = 4.0``,
    ``11 * 0.25 + 1.5 = 4.25``) must survive without truncation.

    Note: the ``ComplexSource`` branch of ``parse_vrt`` in ``_vrt.py``
    maps the XML ``<ScaleRatio>`` to the dataclass ``scale`` attribute
    and ``<ScaleOffset>`` to the ``offset`` attribute, then the
    ``# Apply ComplexSource scaling`` block in ``read_vrt`` applies
    ``src_arr = src_arr * scale + offset``.
    """
    b = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _write(b, p0)
    _write(b, p1)

    vrt_path = _build_complex_source_vrt(
        tmp_path,
        dtype_str='Byte', src_path=str(p1),
        scale_ratio=0.25, scale_offset=1.5,
        other_band_dtype='Byte', other_band_path=str(p0),
    )
    r = read_vrt(vrt_path)
    assert r.dtype.kind == 'f'
    expected = b.astype(np.float64) * 0.25 + 1.5
    np.testing.assert_allclose(r.values[..., 1], expected)


# ---------------------------------------------------------------------------
# 5. NoData round-trip through widened dtype
# ---------------------------------------------------------------------------

def test_nodata_round_trip_through_widened_int_dtype(tmp_path):
    """Band 0 = uint8 with NoData=255; band 1 = int16 with NoData=-9999.
    ``np.result_type(uint8, int16) = int16``. Band 0's value 255 is
    representable as int16 so the nodata fast-path still fires; the
    surviving values must be preserved through the wider buffer.
    """
    b0 = np.array([[1, 2], [3, 255]], dtype=np.uint8)
    b1 = np.array([[100, 200], [300, -9999]], dtype=np.int16)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _write(b0, p0, nodata=255)
    _write(b1, p1, nodata=-9999)

    vrt_path = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Byte" band="1">
    <NoDataValue>255</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="Int16" band="2">
    <NoDataValue>-9999</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    out = tmp_path / 'mixed.vrt'
    out.write_text(vrt_path)
    r = read_vrt(str(out))
    # Per-band nodata masking in __init__.py promotes uint/int VRT
    # buffers to float64 when at least one band's sentinel hits a pixel.
    # Either we land on float64 (NaN-masked) or stay int16 (sentinel
    # literal). Both branches must preserve the non-sentinel values.
    if r.dtype.kind == 'f':
        # Non-sentinel pixels survive
        assert r.values[0, 0, 0] == 1
        assert r.values[0, 1, 0] == 2
        assert r.values[1, 0, 0] == 3
        # Sentinel pixels masked to NaN
        assert np.isnan(r.values[1, 1, 0])
        assert np.isnan(r.values[1, 1, 1])
        # Band 1 non-sentinels survive
        assert r.values[0, 0, 1] == 100
    else:
        # Integer dtype kept; sentinels remain as literal values
        assert r.dtype == np.int16
        assert r.values[0, 0, 0] == 1
        assert r.values[0, 0, 1] == 100


# ---------------------------------------------------------------------------
# 6. Single-band scaled VRT also widens
# ---------------------------------------------------------------------------

def test_single_band_complex_source_scale_widens_buffer(tmp_path):
    """Single-band ``Byte`` VRT with ``<ScaleRatio>0.5</ScaleRatio>``.
    The single-band branch in ``read_vrt`` must mirror the multi-band
    widening logic; previously it used ``selected_bands[0].dtype``
    directly, so the scaled source values truncated back to uint8.
    """
    b = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    p = tmp_path / 'b.tif'
    _write(b, p)

    vrt_path = _build_complex_source_vrt(
        tmp_path,
        dtype_str='Byte', src_path=str(p),
        scale_ratio=0.5,
        extra_band=False,
    )
    r = read_vrt(vrt_path)
    assert r.ndim == 2, (
        f"Single-band VRT must return a 2D array; got shape {r.shape}"
    )
    assert r.dtype.kind == 'f', (
        f"Single-band scaled VRT must widen to float; got {r.dtype}"
    )
    expected = b.astype(np.float64) * 0.5
    np.testing.assert_allclose(r.values, expected)


# ---------------------------------------------------------------------------
# 7. band=N selection from a mixed-dtype VRT picks the right per-band dtype
# ---------------------------------------------------------------------------

def test_band_select_uint8_first_then_float_returns_float_for_band_1(tmp_path):
    """When the caller selects ``band=1`` from a ``Byte`` + ``Float32`` VRT,
    the result dtype must be float (the selected band's declared dtype),
    not uint8 carried over from band 0. The previous code allocated based
    on ``selected_bands[0].dtype`` -- which is correct after band selection
    -- so this is the non-regression check that the new code still does
    the right thing when only one band is selected.
    """
    b0 = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    b1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _write(b0, p0)
    _write(b1, p1)

    vrt_path = _build_two_band_vrt(
        tmp_path,
        b0_dtype_str='Byte', b0_path=str(p0),
        b1_dtype_str='Float32', b1_path=str(p1),
    )
    r = read_vrt(vrt_path, band=1)
    assert r.dtype == np.float32
    np.testing.assert_allclose(r.values, b1)


def test_band_select_uint8_first_then_float_returns_uint8_for_band_0(tmp_path):
    """Selecting ``band=0`` from a ``Byte`` + ``Float32`` VRT must return
    uint8 (band 0's declared dtype) without widening.
    """
    b0 = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    b1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _write(b0, p0)
    _write(b1, p1)

    vrt_path = _build_two_band_vrt(
        tmp_path,
        b0_dtype_str='Byte', b0_path=str(p0),
        b1_dtype_str='Float32', b1_path=str(p1),
    )
    r = read_vrt(vrt_path, band=0)
    assert r.dtype == np.uint8
    np.testing.assert_array_equal(r.values, b0)


# ---------------------------------------------------------------------------
# 8. All-Float32 multi-band stays float32 (do not over-widen)
# ---------------------------------------------------------------------------

def test_all_float32_multiband_stays_float32(tmp_path):
    """Two ``Float32`` bands with no scaling: the buffer must stay
    float32 rather than widening to float64. ``np.result_type`` of two
    identical dtypes returns the same dtype.
    """
    b0 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    b1 = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float32)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _write(b0, p0)
    _write(b1, p1)

    vrt_path = _build_two_band_vrt(
        tmp_path,
        b0_dtype_str='Float32', b0_path=str(p0),
        b1_dtype_str='Float32', b1_path=str(p1),
    )
    r = read_vrt(vrt_path)
    assert r.dtype == np.float32
    np.testing.assert_allclose(r.values[..., 0], b0)
    np.testing.assert_allclose(r.values[..., 1], b1)


# ---------------------------------------------------------------------------
# 9. VRT with zero <VRTRasterBand> elements raises a clear ValueError
# ---------------------------------------------------------------------------

def test_zero_band_vrt_raises_value_error(tmp_path):
    """A malformed VRT with zero ``<VRTRasterBand>`` children must
    surface a clear ``ValueError`` from ``read_vrt`` rather than the
    generic ``"at least one array or dtype is required"`` message
    raised by ``np.result_type`` when called with no arguments.
    """
    import pytest

    vrt_xml = """<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
</VRTDataset>"""
    p = tmp_path / 'empty.vrt'
    p.write_text(vrt_xml)

    with pytest.raises(ValueError, match=r"no <VRTRasterBand>"):
        read_vrt(str(p))
