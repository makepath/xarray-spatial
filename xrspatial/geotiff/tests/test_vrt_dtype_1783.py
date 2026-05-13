"""Regression tests for issue #1783.

Before this fix, :func:`xrspatial.geotiff._vrt.parse_vrt` resolved a
``<VRTRasterBand dataType="...">`` attribute with
``_DTYPE_MAP.get(dtype_name, np.float32)``.  Any GDAL dataType not
present in ``_DTYPE_MAP`` -- the four complex types (``CInt16``,
``CInt32``, ``CFloat32``, ``CFloat64``), the 64-bit integer types that
the map did not yet list (``UInt64``, ``Int64``), or a typo -- silently
collapsed to ``Float32``.  Complex sources lost their imaginary
component, 64-bit integer sources lost precision, and typos produced
wrong values with no diagnostic.

The fix:

* Adds ``UInt64`` / ``Int64`` to ``_DTYPE_MAP``.
* Splits the resolution into "attribute missing" (still defaults to
  Float32 per the GDAL spec) and "attribute present but unsupported"
  (now raises ``ValueError`` naming the band, the offending dataType,
  and the supported types).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import read_vrt
from xrspatial.geotiff._vrt import _parse_band_nodata, parse_vrt
from xrspatial.geotiff._writer import write


def _write(arr, path, **kw):
    """Write a 2D array to ``path`` with sensible defaults for tests."""
    write(arr, str(path), compression='none', tiled=False, **kw)


def _build_single_band_vrt(tmp_path, *, dtype_attr, src_path,
                           filename='b.vrt', size=2, nodata=None):
    """Hand-roll a single-band VRT with an arbitrary ``dataType`` attribute.

    ``dtype_attr`` is rendered verbatim into the ``<VRTRasterBand>``
    element.  Pass an empty string to omit the attribute entirely (the
    "GDAL default" case).

    ``nodata`` (when not ``None``) is rendered verbatim into a
    ``<NoDataValue>`` child so callers can exercise sentinel-parsing
    edge cases (scientific notation, ``nan``, full-range 64-bit
    integers).
    """
    if dtype_attr:
        attr = f' dataType="{dtype_attr}"'
    else:
        attr = ''
    nodata_elem = (f'<NoDataValue>{nodata}</NoDataValue>'
                   if nodata is not None else '')
    vrt_xml = f"""<VRTDataset rasterXSize="{size}" rasterYSize="{size}">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand{attr} band="1">
    {nodata_elem}
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>
      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    p = tmp_path / filename
    p.write_text(vrt_xml)
    return str(p)


# ---------------------------------------------------------------------------
# 1. Complex dataType is rejected (no silent imaginary-component loss)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cdtype', ['CInt16', 'CInt32', 'CFloat32', 'CFloat64'])
def test_complex_dtype_raises_value_error(tmp_path, cdtype):
    """A VRT declaring any complex ``dataType`` must raise ``ValueError``
    rather than silently substituting ``Float32``.  The error message
    must name both the band number and the offending dataType so the
    operator can fix the VRT, and must mention that complex types are
    explicitly unsupported.
    """
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr=cdtype, src_path=str(src),
    )
    with pytest.raises(ValueError) as ei:
        read_vrt(vrt)
    msg = str(ei.value)
    assert cdtype in msg, f"error message must name {cdtype!r}: {msg!r}"
    assert 'band=1' in msg or 'band 1' in msg, (
        f"error message must name the band: {msg!r}"
    )
    assert 'complex' in msg.lower(), (
        f"error message must mention complex types: {msg!r}"
    )


# ---------------------------------------------------------------------------
# 2. Typo / arbitrary garbage dataType is rejected
# ---------------------------------------------------------------------------

def test_garbage_dtype_raises_value_error(tmp_path):
    """An unrecognised non-complex ``dataType`` (e.g. a typo) must also
    raise ``ValueError`` rather than collapsing silently to Float32.
    """
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='Garbage', src_path=str(src),
    )
    with pytest.raises(ValueError, match=r'Garbage'):
        read_vrt(vrt)


def test_typo_for_supported_dtype_is_still_rejected(tmp_path):
    """``Flaot32`` (typo of ``Float32``) is distinct from the empty /
    missing case and must surface as ``ValueError`` instead of silently
    falling back.
    """
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='Flaot32', src_path=str(src),
    )
    with pytest.raises(ValueError, match=r'Flaot32'):
        read_vrt(vrt)


# ---------------------------------------------------------------------------
# 3. UInt64 / Int64 are now supported and round-trip losslessly
# ---------------------------------------------------------------------------

def test_uint64_round_trip(tmp_path):
    """A VRT declaring ``dataType="UInt64"`` whose source GeoTIFF is
    written as uint64 must read back as uint64 with the exact values
    preserved, including values past the float32 / int53 boundary.
    """
    big = np.iinfo(np.uint64).max  # 2**64 - 1
    near_big = big - 7
    b = np.array([[1, 2], [near_big, big]], dtype=np.uint64)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='UInt64', src_path=str(src),
    )
    r = read_vrt(vrt)
    assert r.dtype == np.uint64, (
        f"UInt64 VRT must read as uint64; got {r.dtype}"
    )
    np.testing.assert_array_equal(r.values, b)
    # Largest values must survive bit-for-bit, not collapse to float
    assert int(r.values[1, 1]) == big
    assert int(r.values[1, 0]) == near_big


def test_int64_round_trip(tmp_path):
    """A VRT declaring ``dataType="Int64"`` must read back as int64
    with the full int64 range preserved (positive and negative
    extremes).
    """
    info = np.iinfo(np.int64)
    b = np.array([[info.min, -1], [0, info.max]], dtype=np.int64)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='Int64', src_path=str(src),
    )
    r = read_vrt(vrt)
    assert r.dtype == np.int64, (
        f"Int64 VRT must read as int64; got {r.dtype}"
    )
    np.testing.assert_array_equal(r.values, b)


# ---------------------------------------------------------------------------
# 4. Missing dataType attribute still defaults to Float32 (GDAL default)
# ---------------------------------------------------------------------------

def test_missing_dtype_attribute_defaults_to_float32(tmp_path):
    """``<VRTRasterBand band="1">`` with no ``dataType`` attribute must
    still default to ``Float32``.  This is GDAL's documented default
    and the previous fallback handled it correctly; the new
    "unknown-attribute raises" path must not regress the
    "missing-attribute defaults" path.
    """
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='', src_path=str(src),
    )
    r = read_vrt(vrt)
    assert r.dtype == np.float32, (
        f"missing dataType must default to Float32; got {r.dtype}"
    )
    np.testing.assert_allclose(r.values, b)


# ---------------------------------------------------------------------------
# 5. Pre-existing supported dtypes still read correctly (smoke regression)
# ---------------------------------------------------------------------------

def test_byte_dtype_still_works(tmp_path):
    """``Byte`` reads back as uint8 with values preserved.  Smoke check
    to confirm the rewritten dtype resolution did not break the
    common-case integer path.
    """
    b = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='Byte', src_path=str(src),
    )
    r = read_vrt(vrt)
    assert r.dtype == np.uint8
    np.testing.assert_array_equal(r.values, b)


def test_float64_dtype_still_works(tmp_path):
    """``Float64`` reads back as float64 with values preserved.  Smoke
    check for the wider floating-point path.
    """
    b = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='Float64', src_path=str(src),
    )
    r = read_vrt(vrt)
    assert r.dtype == np.float64
    np.testing.assert_allclose(r.values, b)


# ---------------------------------------------------------------------------
# 6. Integer ``<NoDataValue>`` is parsed as ``int`` for integer bands
#
# Regression for the Copilot review note on the original #1783 PR: now
# that UInt64 / Int64 are supported, parsing the sentinel as ``float``
# silently drops precision near the 64-bit extremes.  ``2**64 - 1``
# rounds up to ``2**64`` in float64, ``INT64_MIN`` survives but only
# barely, and downstream exact-equality masks break.
# ---------------------------------------------------------------------------

def test_parse_band_nodata_uint64_max_exact():
    """``_parse_band_nodata`` must return the exact ``int`` for
    ``2**64 - 1`` (UInt64 max), not a float64 that rounds up to
    ``2**64``.
    """
    big = 2**64 - 1
    nd = _parse_band_nodata(str(big), np.dtype(np.uint64))
    assert isinstance(nd, int), (
        f"UInt64 nodata must parse as int, got {type(nd).__name__}"
    )
    assert nd == big
    # float64 round-trip would equal 2**64, off by one
    assert nd != int(float(big))


def test_parse_band_nodata_int64_min_exact():
    """``INT64_MIN`` (``-2**63``) must survive parsing as an int."""
    info = np.iinfo(np.int64)
    nd = _parse_band_nodata(str(info.min), np.dtype(np.int64))
    assert isinstance(nd, int)
    assert nd == info.min


def test_parse_band_nodata_int32_negative():
    """Common GDAL sentinel ``-9999`` for an Int32 band parses as int."""
    nd = _parse_band_nodata('-9999', np.dtype(np.int32))
    assert isinstance(nd, int)
    assert nd == -9999


def test_parse_band_nodata_int_scientific_notation():
    """GDAL occasionally emits integer nodata in scientific or
    ``-9999.0`` form.  Parsing should still land an int when the
    value is integer-valued and in-range.
    """
    nd = _parse_band_nodata('-9999.0', np.dtype(np.int32))
    assert isinstance(nd, int) and nd == -9999
    nd = _parse_band_nodata('1e3', np.dtype(np.int32))
    assert isinstance(nd, int) and nd == 1000


def test_parse_band_nodata_int_out_of_range_falls_back():
    """An out-of-range sentinel for the band dtype is returned as the
    parsed float so it surfaces via ``attrs['nodata']`` for round-trip
    but can never match an integer pixel (mirroring
    ``_resolve_masked_fill``'s tolerant behaviour).
    """
    # -9999 cannot be represented as uint16; should not raise
    nd = _parse_band_nodata('-9999', np.dtype(np.uint16))
    # ``int('-9999')`` succeeds and out-of-range so we *do* return the
    # int in the cheap path -- _sentinel_for_dtype downstream is then
    # responsible for refusing to use it as a mask sentinel.
    assert nd == -9999


def test_parse_band_nodata_float_nan():
    """Float bands keep NaN sentinels working (no integer-parse
    regression for the floating path).
    """
    nd = _parse_band_nodata('nan', np.dtype(np.float32))
    assert isinstance(nd, float)
    assert np.isnan(nd)


def test_parse_band_nodata_float_scientific():
    """Float bands preserve scientific-notation sentinels."""
    nd = _parse_band_nodata('-1.5e10', np.dtype(np.float64))
    assert isinstance(nd, float)
    assert nd == -1.5e10


def test_parse_band_nodata_empty_or_none():
    """Empty / whitespace / ``None`` input returns ``None`` regardless
    of dtype.
    """
    assert _parse_band_nodata(None, np.dtype(np.int32)) is None
    assert _parse_band_nodata('', np.dtype(np.int32)) is None
    assert _parse_band_nodata('   ', np.dtype(np.float32)) is None


# ---------------------------------------------------------------------------
# 7. End-to-end VRT parse: ``vrt.bands[i].nodata`` is an int for integer
#    bands, a float for float bands.
# ---------------------------------------------------------------------------

def _make_minimal_vrt_xml(dtype_attr, nodata_text):
    """Tiny VRT XML string suitable for direct ``parse_vrt`` calls.

    The SourceFilename here is intentionally minimal -- ``parse_vrt``
    only does the containment check after canonicalising the path, so
    we pass a path inside the temp dir at the call site.
    """
    return (
        '<VRTDataset rasterXSize="1" rasterYSize="1">'
        '<GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>'
        f'<VRTRasterBand dataType="{dtype_attr}" band="1">'
        f'<NoDataValue>{nodata_text}</NoDataValue>'
        '</VRTRasterBand>'
        '</VRTDataset>'
    )


def test_parse_vrt_uint64_nodata_is_int(tmp_path):
    """The dataclass stored on ``_VRTBand.nodata`` is a Python ``int``
    for an integer-dtype band, with the exact 64-bit value.
    """
    big = 2**64 - 1
    xml = _make_minimal_vrt_xml('UInt64', str(big))
    vrt = parse_vrt(xml, vrt_dir=str(tmp_path))
    assert len(vrt.bands) == 1
    nd = vrt.bands[0].nodata
    assert isinstance(nd, int)
    assert nd == big


def test_parse_vrt_int64_min_nodata_is_int(tmp_path):
    info = np.iinfo(np.int64)
    xml = _make_minimal_vrt_xml('Int64', str(info.min))
    vrt = parse_vrt(xml, vrt_dir=str(tmp_path))
    nd = vrt.bands[0].nodata
    assert isinstance(nd, int)
    assert nd == info.min


def test_parse_vrt_float32_nan_nodata_is_float(tmp_path):
    xml = _make_minimal_vrt_xml('Float32', 'nan')
    vrt = parse_vrt(xml, vrt_dir=str(tmp_path))
    nd = vrt.bands[0].nodata
    assert isinstance(nd, float)
    assert np.isnan(nd)


# ---------------------------------------------------------------------------
# 8. Full read_vrt round-trip preserves precision and masks correctly.
# ---------------------------------------------------------------------------

def test_uint64_nodata_round_trip_preserves_max_sentinel(tmp_path):
    """A VRT declaring UInt64 + ``<NoDataValue>2**64 - 1</NoDataValue>``
    must surface ``attrs['nodata']`` as the exact integer value, not a
    float that has rounded past the dtype's range.  Downstream
    consumers rely on exact equality.
    """
    big = 2**64 - 1
    # Fill the source with non-sentinel values so the read keeps the
    # uint64 dtype (a sentinel hit promotes to float64 + NaN, which
    # would defeat the precision check in this test).
    b = np.array([[1, 2], [3, 4]], dtype=np.uint64)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='UInt64', src_path=str(src), nodata=big,
    )
    r = read_vrt(vrt)
    # Either the array stays uint64 (no sentinel hit) or it promotes to
    # float64 (sentinel hit).  In the no-hit case attrs['nodata'] must
    # carry the exact int.
    assert 'nodata' in r.attrs
    assert int(r.attrs['nodata']) == big
    # Critically, the stored attr must not be a float64 that has rounded
    # the sentinel up to 2**64.  ``isinstance`` allows int or np.integer
    # but rejects float / np.floating.
    assert isinstance(r.attrs['nodata'], (int, np.integer))


def test_uint64_nodata_masks_max_sentinel_in_data(tmp_path):
    """When the source pixel actually contains ``2**64 - 1``, the
    masking pipeline must catch it: the result is promoted to float64
    with NaN at the sentinel position.  This is the precision-
    preservation acid test -- if the nodata was rounded to a float
    that doesn't equal the source pixel, the mask never fires and the
    sentinel survives as a 1.8e19 float.
    """
    big = 2**64 - 1
    b = np.array([[1, 2], [3, big]], dtype=np.uint64)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='UInt64', src_path=str(src), nodata=big,
    )
    r = read_vrt(vrt)
    # Sentinel hit -> promote to float64 with NaN
    assert r.dtype == np.float64, (
        f"sentinel hit must promote to float64, got {r.dtype}"
    )
    assert np.isnan(r.values[1, 1]), (
        f"the 2**64-1 cell must be masked to NaN; got {r.values[1, 1]!r}"
    )
    # Non-sentinel cells survive as float64 values
    assert r.values[0, 0] == 1.0
    assert r.values[0, 1] == 2.0
    assert r.values[1, 0] == 3.0


def test_int64_min_nodata_masks_correctly(tmp_path):
    """``INT64_MIN`` as both the nodata sentinel and a real pixel value
    masks correctly without int64 -> float64 rounding aliasing the
    sentinel onto adjacent values.
    """
    info = np.iinfo(np.int64)
    b = np.array([[info.min, -1], [0, info.max]], dtype=np.int64)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='Int64', src_path=str(src), nodata=info.min,
    )
    r = read_vrt(vrt)
    assert r.dtype == np.float64
    assert np.isnan(r.values[0, 0])
    assert r.values[0, 1] == -1.0
    assert r.values[1, 0] == 0.0
    assert r.values[1, 1] == float(info.max)


def test_int32_negative_nodata_still_masks(tmp_path):
    """Smoke regression for the common Int32 + ``-9999`` case.  The
    integer parsing path must not break this when there is no precision
    pressure -- ``-9999`` survives ``float()`` fine but we still want
    the new int-typed parse to mask the same way the old float-typed
    parse did.
    """
    b = np.array([[10, -9999], [-9999, 20]], dtype=np.int32)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='Int32', src_path=str(src), nodata=-9999,
    )
    r = read_vrt(vrt)
    assert r.dtype == np.float64
    assert np.isnan(r.values[0, 1])
    assert np.isnan(r.values[1, 0])
    assert r.values[0, 0] == 10.0
    assert r.values[1, 1] == 20.0


def test_float32_nan_nodata_still_works(tmp_path):
    """``Float32`` + ``<NoDataValue>nan</NoDataValue>`` still parses and
    surfaces NaN via ``attrs['nodata']`` (no regression on the float
    path).
    """
    b = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='Float32', src_path=str(src), nodata='nan',
    )
    r = read_vrt(vrt)
    assert r.dtype == np.float32
    assert np.isnan(r.attrs['nodata'])
    assert np.isnan(r.values[0, 1])


def test_float64_scientific_nodata_still_works(tmp_path):
    """``Float64`` + scientific-notation ``<NoDataValue>`` survives as
    float (no integer-parse regression for the float path).
    """
    b = np.array([[1.0, -1.5e10], [3.0, 4.0]], dtype=np.float64)
    src = tmp_path / 'src.tif'
    _write(b, src)
    vrt = _build_single_band_vrt(
        tmp_path, dtype_attr='Float64', src_path=str(src), nodata='-1.5e10',
    )
    r = read_vrt(vrt)
    assert r.dtype == np.float64
    assert r.attrs['nodata'] == -1.5e10
    # The matching pixel stays as-is for float -- there's no NaN
    # promotion (it's already float64), so the sentinel surfaces as a
    # literal value unless the float-source nodata-masking branch fires.
    # Either behaviour is acceptable; just confirm nodata attr is set.
