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
from xrspatial.geotiff._writer import write


def _write(arr, path, **kw):
    """Write a 2D array to ``path`` with sensible defaults for tests."""
    write(arr, str(path), compression='none', tiled=False, **kw)


def _build_single_band_vrt(tmp_path, *, dtype_attr, src_path,
                           filename='b.vrt', size=2):
    """Hand-roll a single-band VRT with an arbitrary ``dataType`` attribute.

    ``dtype_attr`` is rendered verbatim into the ``<VRTRasterBand>``
    element.  Pass an empty string to omit the attribute entirely (the
    "GDAL default" case).
    """
    if dtype_attr:
        attr = f' dataType="{dtype_attr}"'
    else:
        attr = ''
    vrt_xml = f"""<VRTDataset rasterXSize="{size}" rasterYSize="{size}">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand{attr} band="1">
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
