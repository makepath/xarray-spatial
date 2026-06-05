"""VRT dtype-conversion test suite.

Covers source-vs-output dtype handling, integer-with-nodata promotion,
multiband dtype, and the positive mosaic coverage that exercises dtype
passthrough end to end. Helpers are prefixed (e.g.
``_dtype_validation_*``) to avoid collisions across sections.

Sections:
* VRT dataType attribute validation and band-nodata parsing
* VRT writer dtype name resolution
* Integer source feeding a float VRT
* Multiband dtype promotion and band-select
* Multiband per-band integer nodata
* VRT resample algorithm validation
* Simple VRT mosaic positive coverage
"""
from __future__ import annotations

import os
import uuid

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _read_vrt, to_geotiff
from xrspatial.geotiff._errors import MixedBandMetadataError, VRTUnsupportedError
from xrspatial.geotiff._vrt import (_NP_TO_VRT_DTYPE, _parse_band_nodata, _vrt_dtype_name_for,
                                    parse_vrt)
from xrspatial.geotiff._vrt import read_vrt as _resample_alg_read_vrt_internal
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal
from xrspatial.geotiff._writer import write

# ---------------------------------------------------------------------------
# VRT dataType attribute validation and parsing
# ---------------------------------------------------------------------------


def _dtype_validation_write(arr, path, **kw):
    """Write a 2D array to ``path`` with sensible defaults for tests."""
    write(arr, str(path), compression='none', tiled=False, **kw)


def _dtype_validation_build_single_band_vrt(tmp_path, *, dtype_attr, src_path, filename='b.vrt', size=2, nodata=None):  # noqa: E501
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
    nodata_elem = f'<NoDataValue>{nodata}</NoDataValue>' if nodata is not None else ''
    vrt_xml = f'<VRTDataset rasterXSize="{size}" rasterYSize="{size}">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand{attr} band="1">\n    {nodata_elem}\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    p = tmp_path / filename
    p.write_text(vrt_xml)
    return str(p)


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
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr=cdtype, src_path=str(src))
    with pytest.raises(ValueError) as ei:
        _read_vrt(vrt)
    msg = str(ei.value)
    assert cdtype in msg, f'error message must name {cdtype!r}: {msg!r}'
    assert 'band=1' in msg or 'band 1' in msg, f'error message must name the band: {msg!r}'
    assert 'complex' in msg.lower(), f'error message must mention complex types: {msg!r}'


def test_garbage_dtype_raises_value_error(tmp_path):
    """An unrecognised non-complex ``dataType`` (e.g. a typo) must also
    raise ``ValueError`` rather than collapsing silently to Float32.
    """
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src = tmp_path / 'src.tif'
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='Garbage', src_path=str(src))
    with pytest.raises(ValueError, match='Garbage'):
        _read_vrt(vrt)


def test_typo_for_supported_dtype_is_still_rejected(tmp_path):
    """``Flaot32`` (typo of ``Float32``) is distinct from the empty /
    missing case and must surface as ``ValueError`` instead of silently
    falling back.
    """
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src = tmp_path / 'src.tif'
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='Flaot32', src_path=str(src))
    with pytest.raises(ValueError, match='Flaot32'):
        _read_vrt(vrt)


def test_uint64_round_trip(tmp_path):
    """A VRT declaring ``dataType="UInt64"`` whose source GeoTIFF is
    written as uint64 must read back as uint64 with the exact values
    preserved, including values past the float32 / int53 boundary.
    """
    big = np.iinfo(np.uint64).max
    near_big = big - 7
    b = np.array([[1, 2], [near_big, big]], dtype=np.uint64)
    src = tmp_path / 'src.tif'
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='UInt64', src_path=str(src))
    r = _read_vrt(vrt)
    assert r.dtype == np.uint64, f'UInt64 VRT must read as uint64; got {r.dtype}'
    np.testing.assert_array_equal(r.values, b)
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
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='Int64', src_path=str(src))
    r = _read_vrt(vrt)
    assert r.dtype == np.int64, f'Int64 VRT must read as int64; got {r.dtype}'
    np.testing.assert_array_equal(r.values, b)


def test_missing_dtype_attribute_defaults_to_float32(tmp_path):
    """``<VRTRasterBand band="1">`` with no ``dataType`` attribute must
    still default to ``Float32``.  This is GDAL's documented default
    and the previous fallback handled it correctly; the new
    "unknown-attribute raises" path must not regress the
    "missing-attribute defaults" path.
    """
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src = tmp_path / 'src.tif'
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='', src_path=str(src))
    r = _read_vrt(vrt)
    assert r.dtype == np.float32, f'missing dataType must default to Float32; got {r.dtype}'
    np.testing.assert_allclose(r.values, b)


def test_byte_dtype_still_works(tmp_path):
    """``Byte`` reads back as uint8 with values preserved.  Smoke check
    to confirm the rewritten dtype resolution did not break the
    common-case integer path.
    """
    b = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    src = tmp_path / 'src.tif'
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='Byte', src_path=str(src))
    r = _read_vrt(vrt)
    assert r.dtype == np.uint8
    np.testing.assert_array_equal(r.values, b)


def test_float64_dtype_still_works(tmp_path):
    """``Float64`` reads back as float64 with values preserved.  Smoke
    check for the wider floating-point path.
    """
    b = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    src = tmp_path / 'src.tif'
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='Float64', src_path=str(src))
    r = _read_vrt(vrt)
    assert r.dtype == np.float64
    np.testing.assert_allclose(r.values, b)


def test_parse_band_nodata_uint64_max_exact():
    """``_parse_band_nodata`` must return the exact ``int`` for
    ``2**64 - 1`` (UInt64 max), not a float64 that rounds up to
    ``2**64``.
    """
    big = 2 ** 64 - 1
    nd = _parse_band_nodata(str(big), np.dtype(np.uint64))
    assert isinstance(nd, int), f'UInt64 nodata must parse as int, got {type(nd).__name__}'
    assert nd == big
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
    nd = _parse_band_nodata('-9999', np.dtype(np.uint16))
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
    assert nd == -15000000000.0


def test_parse_band_nodata_empty_or_none():
    """Empty / whitespace / ``None`` input returns ``None`` regardless
    of dtype.
    """
    assert _parse_band_nodata(None, np.dtype(np.int32)) is None
    assert _parse_band_nodata('', np.dtype(np.int32)) is None
    assert _parse_band_nodata('   ', np.dtype(np.float32)) is None


def _dtype_validation_make_minimal_vrt_xml(dtype_attr, nodata_text):
    """Tiny VRT XML string suitable for direct ``parse_vrt`` calls.

    The SourceFilename here is intentionally minimal -- ``parse_vrt``
    only does the containment check after canonicalising the path, so
    we pass a path inside the temp dir at the call site.
    """
    return f'<VRTDataset rasterXSize="1" rasterYSize="1"><GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform><VRTRasterBand dataType="{dtype_attr}" band="1"><NoDataValue>{nodata_text}</NoDataValue></VRTRasterBand></VRTDataset>'  # noqa: E501


def test_parse_vrt_uint64_nodata_is_int(tmp_path):
    """The dataclass stored on ``_VRTBand.nodata`` is a Python ``int``
    for an integer-dtype band, with the exact 64-bit value.
    """
    big = 2 ** 64 - 1
    xml = _dtype_validation_make_minimal_vrt_xml('UInt64', str(big))
    vrt = parse_vrt(xml, vrt_dir=str(tmp_path))
    assert len(vrt.bands) == 1
    nd = vrt.bands[0].nodata
    assert isinstance(nd, int)
    assert nd == big


def test_parse_vrt_int64_min_nodata_is_int(tmp_path):
    info = np.iinfo(np.int64)
    xml = _dtype_validation_make_minimal_vrt_xml('Int64', str(info.min))
    vrt = parse_vrt(xml, vrt_dir=str(tmp_path))
    nd = vrt.bands[0].nodata
    assert isinstance(nd, int)
    assert nd == info.min


def test_parse_vrt_float32_nan_nodata_is_float(tmp_path):
    xml = _dtype_validation_make_minimal_vrt_xml('Float32', 'nan')
    vrt = parse_vrt(xml, vrt_dir=str(tmp_path))
    nd = vrt.bands[0].nodata
    assert isinstance(nd, float)
    assert np.isnan(nd)


def test_uint64_nodata_round_trip_preserves_max_sentinel(tmp_path):
    """A VRT declaring UInt64 + ``<NoDataValue>2**64 - 1</NoDataValue>``
    must surface ``attrs['nodata']`` as the exact integer value, not a
    float that has rounded past the dtype's range.  Downstream
    consumers rely on exact equality.
    """
    big = 2 ** 64 - 1
    b = np.array([[1, 2], [3, 4]], dtype=np.uint64)
    src = tmp_path / 'src.tif'
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='UInt64', src_path=str(src), nodata=big)  # noqa: E501
    r = _read_vrt(vrt)
    assert 'nodata' in r.attrs
    assert int(r.attrs['nodata']) == big
    assert isinstance(r.attrs['nodata'], (int, np.integer))


def test_uint64_nodata_masks_max_sentinel_in_data(tmp_path):
    """When the source pixel actually contains ``2**64 - 1``, the
    masking pipeline must catch it: the result is promoted to float64
    with NaN at the sentinel position.  This is the precision-
    preservation acid test -- if the nodata was rounded to a float
    that doesn't equal the source pixel, the mask never fires and the
    sentinel survives as a 1.8e19 float.
    """
    big = 2 ** 64 - 1
    b = np.array([[1, 2], [3, big]], dtype=np.uint64)
    src = tmp_path / 'src.tif'
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='UInt64', src_path=str(src), nodata=big)  # noqa: E501
    r = _read_vrt(vrt, mask_nodata=True)
    assert r.dtype == np.float64, f'sentinel hit must promote to float64, got {r.dtype}'
    assert np.isnan(r.values[1, 1]), f'the 2**64-1 cell must be masked to NaN; got {r.values[1, 1]!r}'  # noqa: E501
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
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='Int64', src_path=str(src), nodata=info.min)  # noqa: E501
    r = _read_vrt(vrt, mask_nodata=True)
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
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='Int32', src_path=str(src), nodata=-9999)  # noqa: E501
    r = _read_vrt(vrt, mask_nodata=True)
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
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='Float32', src_path=str(src), nodata='nan')  # noqa: E501
    r = _read_vrt(vrt)
    assert r.dtype == np.float32
    assert np.isnan(r.attrs['nodata'])
    assert np.isnan(r.values[0, 1])


def test_float64_scientific_nodata_still_works(tmp_path):
    """``Float64`` + scientific-notation ``<NoDataValue>`` survives as
    float (no integer-parse regression for the float path).
    """
    b = np.array([[1.0, -15000000000.0], [3.0, 4.0]], dtype=np.float64)
    src = tmp_path / 'src.tif'
    _dtype_validation_write(b, src)
    vrt = _dtype_validation_build_single_band_vrt(tmp_path, dtype_attr='Float64', src_path=str(src), nodata='-1.5e10')  # noqa: E501
    r = _read_vrt(vrt)
    assert r.dtype == np.float64
    assert r.attrs['nodata'] == -15000000000.0


# ---------------------------------------------------------------------------
# VRT writer dtype name resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('bps,sf,expected', [(12, 1, 'UInt16'), (12, 4, 'UInt16'), (1, 1, 'Byte'), (2, 1, 'Byte'), (4, 1, 'Byte'), (8, 1, 'Byte'), (16, 1, 'UInt16'), (32, 1, 'UInt32'), (64, 1, 'UInt64'), (8, 2, 'Int8'), (16, 2, 'Int16'), (32, 2, 'Int32'), (64, 2, 'Int64'), (32, 3, 'Float32'), (64, 3, 'Float64')])  # noqa: E501
def test_vrt_dtype_name_for_supported(bps, sf, expected):
    assert _vrt_dtype_name_for(bps, sf) == expected


def test_vrt_dtype_name_for_sample_format_sequence_resolves():
    assert _vrt_dtype_name_for(8, [1, 1]) == 'Byte'
    assert _vrt_dtype_name_for(16, (2, 2)) == 'Int16'


def test_vrt_dtype_name_for_unsupported_raises():
    with pytest.raises(ValueError):
        _vrt_dtype_name_for(24, 2)


def test_np_to_vrt_dtype_table_covers_all_resolver_outputs():
    from xrspatial.geotiff._dtypes import tiff_dtype_to_numpy
    pairs = [(8, 1), (8, 2), (16, 1), (16, 2), (32, 1), (32, 2), (32, 3), (64, 1), (64, 2), (64, 3), (1, 1), (2, 1), (4, 1), (12, 1)]  # noqa: E501
    for bps, sf in pairs:
        np_dtype = tiff_dtype_to_numpy(bps, sf)
        assert np_dtype.type in _NP_TO_VRT_DTYPE, f'resolver yields {np_dtype} for bps={bps}, sf={sf} but _NP_TO_VRT_DTYPE has no entry for it'  # noqa: E501


def _dtype_12bit_unique_dir(tmp_path, label: str) -> str:
    d = tmp_path / f'vrt_1914_{label}_{uuid.uuid4().hex[:8]}'
    d.mkdir()
    return str(d)


def _dtype_12bit_write_uint16_tif(path: str, *, h: int = 4, w: int = 4, origin_x: float = 0.0) -> None:  # noqa: E501
    arr = np.arange(h * w, dtype=np.uint16).reshape(h, w)
    y = 100.0 + (np.arange(h) + 0.5) * -1.0
    x = origin_x + (np.arange(w) + 0.5) * 1.0
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326})
    to_geotiff(da, path, compression='none')


def test_uint16_source_writes_uint16_vrt_datatype(tmp_path):
    d = _dtype_12bit_unique_dir(tmp_path, 'u16')
    a = os.path.join(d, 'a.tif')
    b = os.path.join(d, 'b.tif')
    _dtype_12bit_write_uint16_tif(a)
    _dtype_12bit_write_uint16_tif(b, origin_x=4.0)
    vrt = os.path.join(d, 'out.vrt')
    _write_vrt_internal(vrt, [a, b])
    with open(vrt) as f:
        xml = f.read()
    assert 'dataType="UInt16"' in xml
    assert 'dataType="Byte"' not in xml


def test_int16_source_writes_int16_vrt_datatype(tmp_path):
    d = _dtype_12bit_unique_dir(tmp_path, 'i16')
    a = os.path.join(d, 'a.tif')
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    y = 100.0 + (np.arange(4) + 0.5) * -1.0
    x = (np.arange(4) + 0.5) * 1.0
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs={'crs': 4326})
    to_geotiff(da, a, compression='none')
    vrt = os.path.join(d, 'out.vrt')
    _write_vrt_internal(vrt, [a])
    with open(vrt) as f:
        xml = f.read()
    assert 'dataType="Int16"' in xml


# ---------------------------------------------------------------------------
# integer source feeding float VRT
# ---------------------------------------------------------------------------


def _int_source_float_dtype_write_uint16_with_sentinel(tmp_path, sentinel=65535, filename='b0.tif'):
    band = np.array([[1, 2], [3, sentinel]], dtype=np.uint16)
    p = str(tmp_path / filename)
    write(band, p, nodata=sentinel, compression='none', tiled=False)
    return p


def _int_source_float_dtype_write_int16_with_sentinel(tmp_path, sentinel=-1, filename='b0.tif'):
    band = np.array([[1, 2], [3, sentinel]], dtype=np.int16)
    p = str(tmp_path / filename)
    write(band, p, nodata=sentinel, compression='none', tiled=False)
    return p


def _int_source_float_dtype_build_vrt(tmp_path, source_path, vrt_dtype, nodata_value, filename='mismatch.vrt'):  # noqa: E501
    """Hand-roll a VRT with the requested dataType / NoDataValue pair."""
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="{vrt_dtype}" band="1">\n    <NoDataValue>{nodata_value}</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{source_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    p = str(tmp_path / filename)
    with open(p, 'w') as f:
        f.write(vrt_xml)
    return p


def test_float32_vrt_uint16_source_masks_in_range_sentinel(tmp_path):
    """Float32 VRT, uint16 source with in-range sentinel: pixel becomes NaN.

    Before the fix this returned dtype=float32 with values[1, 1] == 65535.0
    while ``attrs['nodata']`` advertised the sentinel.
    """
    src = _int_source_float_dtype_write_uint16_with_sentinel(tmp_path)
    vrt = _int_source_float_dtype_build_vrt(tmp_path, src, 'Float32', 65535)
    r = _read_vrt(vrt, mask_nodata=True)
    assert r.dtype == np.float32, f'Float32-declared VRT should return float32, got {r.dtype}'
    assert np.isnan(r.values[1, 1]), f'Sentinel pixel (uint16 65535 -> float32) should be NaN-masked; got values[1, 1]={r.values[1, 1]}'  # noqa: E501
    assert r.attrs.get('nodata') == 65535.0
    assert r.values[0, 0] == 1.0


def test_float64_vrt_int16_source_masks_negative_sentinel(tmp_path):
    """Float64 VRT, int16 source with negative sentinel: pixel becomes NaN."""
    src = _int_source_float_dtype_write_int16_with_sentinel(tmp_path, sentinel=-1)
    vrt = _int_source_float_dtype_build_vrt(tmp_path, src, 'Float64', -1)
    r = _read_vrt(vrt, mask_nodata=True)
    assert r.dtype == np.float64
    assert np.isnan(r.values[1, 1]), f'Sentinel pixel (-1) should be NaN-masked; got values[1, 1]={r.values[1, 1]}'  # noqa: E501
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
    write(arr, p, compression='none', tiled=False)
    vrt = _int_source_float_dtype_build_vrt(tmp_path, p, 'Float32', -9999)
    r = _read_vrt(vrt)
    assert r.dtype == np.float32
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
    vrt = _int_source_float_dtype_build_vrt(tmp_path, p, 'Float32', 65535)
    r = _read_vrt(vrt)
    assert r.dtype == np.float32
    assert not np.isnan(r.values).any()
    np.testing.assert_array_equal(r.values, arr.astype(np.float32))


def test_float_vrt_int_source_dask_path_masks_sentinel(tmp_path):
    """The dask wrapper path (``chunks=...``) also returns NaN at the
    sentinel pixel. The dask reader chunks the eager result after decode,
    so the bug propagates if the eager path leaks the sentinel.
    """
    src = _int_source_float_dtype_write_uint16_with_sentinel(tmp_path)
    vrt = _int_source_float_dtype_build_vrt(tmp_path, src, 'Float32', 65535)
    r = _read_vrt(vrt, chunks=2, mask_nodata=True)
    assert r.dtype == np.float32
    val = r.values
    assert np.isnan(val[1, 1])


def test_float_vrt_int_source_round_trip_nodata_attr(tmp_path):
    """Even though the masking promotes pixels to NaN, the
    ``attrs['nodata']`` value still carries the original sentinel so a
    downstream write can restore the literal sentinel byte pattern.
    """
    src = _int_source_float_dtype_write_uint16_with_sentinel(tmp_path)
    vrt = _int_source_float_dtype_build_vrt(tmp_path, src, 'Float32', 65535)
    r = _read_vrt(vrt)
    assert r.attrs.get('nodata') == 65535.0


def test_float_vrt_int_source_with_band_select(tmp_path):
    """The band=N selection path also masks integer sentinels for a
    float-declared VRT. The per-band ``NoDataValue`` from the VRT XML
    must reach the source-side masking step, not just ``attrs['nodata']``.
    """
    src_a = _int_source_float_dtype_write_uint16_with_sentinel(tmp_path, sentinel=65535, filename='ba.tif')  # noqa: E501
    src_b = _int_source_float_dtype_write_uint16_with_sentinel(tmp_path, sentinel=65000, filename='bb.tif')  # noqa: E501
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="Float32" band="1">\n    <NoDataValue>65535</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_a}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n  <VRTRasterBand dataType="Float32" band="2">\n    <NoDataValue>65000</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{src_b}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    vrt_path = str(tmp_path / 'mb.vrt')
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    r0 = _read_vrt(vrt_path, band=0, band_nodata='first', mask_nodata=True)
    assert r0.dtype == np.float32
    assert np.isnan(r0.values[1, 1])
    assert r0.attrs.get('nodata') == 65535.0
    r1 = _read_vrt(vrt_path, band=1, band_nodata='first', mask_nodata=True)
    assert r1.dtype == np.float32
    assert np.isnan(r1.values[1, 1])
    assert r1.attrs.get('nodata') == 65000.0


# ---------------------------------------------------------------------------
# multiband dtype promotion
# ---------------------------------------------------------------------------


def _multiband_dtype_write(arr, path, **kw):
    """Write a 2D array to ``path`` with sensible defaults for tests."""
    write(arr, str(path), compression='none', tiled=False, **kw)


def _multiband_dtype_build_two_band_vrt(tmp_path, *, b0_dtype_str, b0_path, b1_dtype_str, b1_path, b1_extra='', b0_extra='', filename='mb.vrt', size=2):  # noqa: E501
    """Hand-roll a two-band VRT with arbitrary dataType strings."""
    vrt_xml = f'<VRTDataset rasterXSize="{size}" rasterYSize="{size}">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="{b0_dtype_str}" band="1">\n{b0_extra}    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{b0_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n    </SimpleSource>\n  </VRTRasterBand>\n  <VRTRasterBand dataType="{b1_dtype_str}" band="2">\n{b1_extra}    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{b1_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    p = tmp_path / filename
    p.write_text(vrt_xml)
    return str(p)


def _multiband_dtype_build_complex_source_vrt(tmp_path, *, dtype_str, src_path, scale_ratio=None, scale_offset=None, filename='cs.vrt', size=2, band_num=2, other_band_dtype='Byte', other_band_path=None, extra_band=True):  # noqa: E501
    """Hand-roll a VRT where band 2 (or the only band) uses ComplexSource.

    ``extra_band=False`` writes a single-band VRT.
    """
    cs_lines = []
    if scale_ratio is not None:
        cs_lines.append(f'      <ScaleRatio>{scale_ratio}</ScaleRatio>')
    if scale_offset is not None:
        cs_lines.append(f'      <ScaleOffset>{scale_offset}</ScaleOffset>')
    cs_inner = '\n'.join(cs_lines)
    complex_block = f'    <ComplexSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n{cs_inner}\n      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n    </ComplexSource>\n'  # noqa: E501
    if extra_band and other_band_path is not None:
        vrt_xml = f'<VRTDataset rasterXSize="{size}" rasterYSize="{size}">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="{other_band_dtype}" band="1">\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{other_band_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n      <DstRect xOff="0" yOff="0" xSize="{size}" ySize="{size}"/>\n    </SimpleSource>\n  </VRTRasterBand>\n  <VRTRasterBand dataType="{dtype_str}" band="{band_num}">\n{complex_block}  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    else:
        vrt_xml = f'<VRTDataset rasterXSize="{size}" rasterYSize="{size}">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="{dtype_str}" band="1">\n{complex_block}  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    p = tmp_path / filename
    p.write_text(vrt_xml)
    return str(p)


def test_mixed_byte_and_float32_bands_raise(tmp_path):
    """``Byte`` band 0 + ``Float32`` band 1 must raise rather than widen.

    The original #1696 fix widened the output buffer so band 1's
    fractional values survived. The VRT support matrix at
    ``_backends/vrt.py`` later tightened the contract: per-band dtype
    mismatch is a documented error condition, not a silent widening.
    Issue #2485 flipped the behaviour from widening to raising.
    """
    b0 = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    b1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _multiband_dtype_write(b0, p0)
    _multiband_dtype_write(b1, p1)
    vrt_path = _multiband_dtype_build_two_band_vrt(tmp_path, b0_dtype_str='Byte', b0_path=str(p0), b1_dtype_str='Float32', b1_path=str(p1))  # noqa: E501
    with pytest.raises(MixedBandMetadataError) as excinfo:
        _read_vrt(vrt_path)
    msg = str(excinfo.value).lower()
    assert 'band 1' in msg and 'band 2' in msg


def test_complex_source_scale_promotes_buffer_to_float(tmp_path):
    """Both bands declare ``Byte`` but band 1 has ``<ScaleRatio>0.5</ScaleRatio>``.
    The scaled source values include fractional results (11 * 0.5 = 5.5)
    which must survive. Before the fix the buffer stayed uint8 and the
    fractional values rounded down to 5.
    """
    b = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _multiband_dtype_write(b, p0)
    _multiband_dtype_write(b, p1)
    vrt_path = _multiband_dtype_build_complex_source_vrt(tmp_path, dtype_str='Byte', src_path=str(p1), scale_ratio=0.5, other_band_dtype='Byte', other_band_path=str(p0))  # noqa: E501
    r = _read_vrt(vrt_path)
    assert r.dtype.kind == 'f', f'ScaleRatio on a Byte band must widen the buffer to float; got {r.dtype}'  # noqa: E501
    expected = b.astype(np.float64) * 0.5
    np.testing.assert_allclose(r.values[..., 1], expected)
    np.testing.assert_array_equal(r.values[..., 0].astype(np.uint8), b)


def test_all_byte_no_scaling_stays_uint8(tmp_path):
    """Two ``Byte`` bands with no ``ComplexSource`` scaling: the result
    must stay uint8 (memory regression guard). The fix must not widen
    unconditionally to float64.
    """
    b0 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    b1 = np.array([[50, 60], [70, 80]], dtype=np.uint8)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _multiband_dtype_write(b0, p0)
    _multiband_dtype_write(b1, p1)
    vrt_path = _multiband_dtype_build_two_band_vrt(tmp_path, b0_dtype_str='Byte', b0_path=str(p0), b1_dtype_str='Byte', b1_path=str(p1))  # noqa: E501
    r = _read_vrt(vrt_path)
    assert r.dtype == np.uint8, f'All-Byte VRT with no scaling must stay uint8; got {r.dtype}'
    np.testing.assert_array_equal(r.values[..., 0], b0)
    np.testing.assert_array_equal(r.values[..., 1], b1)


def test_complex_source_scale_and_offset_preserve_precision(tmp_path):
    """``ScaleRatio=0.25`` plus ``ScaleOffset=1.5`` on a uint8 band:
    the scaled-and-offset values (e.g. ``10 * 0.25 + 1.5 = 4.0``,
    ``11 * 0.25 + 1.5 = 4.25``) must survive without truncation.

    Note: the ``ComplexSource`` branch of ``parse_vrt`` in ``_vrt.py``
    maps the XML ``<ScaleRatio>`` to the dataclass ``scale`` attribute
    and ``<ScaleOffset>`` to the ``offset`` attribute, then the
    ``# Apply ComplexSource scaling`` block in ``_read_vrt`` applies
    ``src_arr = src_arr * scale + offset``.
    """
    b = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _multiband_dtype_write(b, p0)
    _multiband_dtype_write(b, p1)
    vrt_path = _multiband_dtype_build_complex_source_vrt(tmp_path, dtype_str='Byte', src_path=str(p1), scale_ratio=0.25, scale_offset=1.5, other_band_dtype='Byte', other_band_path=str(p0))  # noqa: E501
    r = _read_vrt(vrt_path)
    assert r.dtype.kind == 'f'
    expected = b.astype(np.float64) * 0.25 + 1.5
    np.testing.assert_allclose(r.values[..., 1], expected)


def test_mixed_byte_and_int16_bands_raise(tmp_path):
    """``Byte`` band 0 + ``Int16`` band 1 must raise rather than widen.

    Previously the reader widened via ``np.result_type(uint8, int16)``
    and round-tripped the nodata sentinels through the wider buffer.
    The #2485 contract rejects this kind of cross-band dtype mismatch
    rather than silently flattening; callers that genuinely want the
    widened behaviour must rebuild their VRT with a single declared
    dataType.
    """
    b0 = np.array([[1, 2], [3, 255]], dtype=np.uint8)
    b1 = np.array([[100, 200], [300, -9999]], dtype=np.int16)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _multiband_dtype_write(b0, p0, nodata=255)
    _multiband_dtype_write(b1, p1, nodata=-9999)
    vrt_path = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="Byte" band="1">\n    <NoDataValue>255</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n  <VRTRasterBand dataType="Int16" band="2">\n    <NoDataValue>-9999</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    out = tmp_path / 'mixed.vrt'
    out.write_text(vrt_path)
    with pytest.raises(MixedBandMetadataError) as excinfo:
        _read_vrt(str(out), band_nodata='first')
    msg = str(excinfo.value).lower()
    assert 'band 1' in msg and 'band 2' in msg


def test_single_band_complex_source_scale_widens_buffer(tmp_path):
    """Single-band ``Byte`` VRT with ``<ScaleRatio>0.5</ScaleRatio>``.
    The single-band branch in ``_read_vrt`` must mirror the multi-band
    widening logic; previously it used ``selected_bands[0].dtype``
    directly, so the scaled source values truncated back to uint8.
    """
    b = np.array([[10, 11], [12, 13]], dtype=np.uint8)
    p = tmp_path / 'b.tif'
    _multiband_dtype_write(b, p)
    vrt_path = _multiband_dtype_build_complex_source_vrt(tmp_path, dtype_str='Byte', src_path=str(p), scale_ratio=0.5, extra_band=False)  # noqa: E501
    r = _read_vrt(vrt_path)
    assert r.ndim == 2, f'Single-band VRT must return a 2D array; got shape {r.shape}'
    assert r.dtype.kind == 'f', f'Single-band scaled VRT must widen to float; got {r.dtype}'
    expected = b.astype(np.float64) * 0.5
    np.testing.assert_allclose(r.values, expected)


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
    _multiband_dtype_write(b0, p0)
    _multiband_dtype_write(b1, p1)
    vrt_path = _multiband_dtype_build_two_band_vrt(tmp_path, b0_dtype_str='Byte', b0_path=str(p0), b1_dtype_str='Float32', b1_path=str(p1))  # noqa: E501
    r = _read_vrt(vrt_path, band=1)
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
    _multiband_dtype_write(b0, p0)
    _multiband_dtype_write(b1, p1)
    vrt_path = _multiband_dtype_build_two_band_vrt(tmp_path, b0_dtype_str='Byte', b0_path=str(p0), b1_dtype_str='Float32', b1_path=str(p1))  # noqa: E501
    r = _read_vrt(vrt_path, band=0)
    assert r.dtype == np.uint8
    np.testing.assert_array_equal(r.values, b0)


def test_all_float32_multiband_stays_float32(tmp_path):
    """Two ``Float32`` bands with no scaling: the buffer must stay
    float32 rather than widening to float64. ``np.result_type`` of two
    identical dtypes returns the same dtype.
    """
    b0 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    b1 = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float32)
    p0 = tmp_path / 'b0.tif'
    p1 = tmp_path / 'b1.tif'
    _multiband_dtype_write(b0, p0)
    _multiband_dtype_write(b1, p1)
    vrt_path = _multiband_dtype_build_two_band_vrt(tmp_path, b0_dtype_str='Float32', b0_path=str(p0), b1_dtype_str='Float32', b1_path=str(p1))  # noqa: E501
    r = _read_vrt(vrt_path)
    assert r.dtype == np.float32
    np.testing.assert_allclose(r.values[..., 0], b0)
    np.testing.assert_allclose(r.values[..., 1], b1)


def test_zero_band_vrt_raises_value_error(tmp_path):
    """A malformed VRT with zero ``<VRTRasterBand>`` children must
    surface a clear ``ValueError`` from ``_read_vrt`` rather than the
    generic ``"at least one array or dtype is required"`` message
    raised by ``np.result_type`` when called with no arguments.
    """
    import pytest
    vrt_xml = '<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n</VRTDataset>'  # noqa: E501
    p = tmp_path / 'empty.vrt'
    p.write_text(vrt_xml)
    with pytest.raises(ValueError, match='no <VRTRasterBand>'):
        _read_vrt(str(p))


# ---------------------------------------------------------------------------
# multiband per-band int nodata
# ---------------------------------------------------------------------------


def _multiband_int_nodata_write_two_band_per_band_nodata_vrt(tmp_path, *, dtype_str='UInt16', np_dtype=np.uint16, band0_sentinel=65535, band1_sentinel=65000, band0_other=(1, 2, 3), band1_other=(7, 8, 9)):  # noqa: E501
    """Two single-band integer sources, each with a distinct nodata
    sentinel, exposed as bands 1 and 2 of a hand-rolled VRT.

    Used to be band 0's sentinel was the only one masked. Now every
    band gets its own sentinel.
    """
    b0_arr = np.array([[band0_other[0], band0_other[1]], [band0_other[2], band0_sentinel]], dtype=np_dtype)  # noqa: E501
    b1_arr = np.array([[band1_other[0], band1_other[1]], [band1_other[2], band1_sentinel]], dtype=np_dtype)  # noqa: E501
    p0 = str(tmp_path / 'vrt_b0_1611.tif')
    p1 = str(tmp_path / 'vrt_b1_1611.tif')
    write(b0_arr, p0, nodata=band0_sentinel, compression='none', tiled=False)
    write(b1_arr, p1, nodata=band1_sentinel, compression='none', tiled=False)
    vrt_path = str(tmp_path / 'two_band_per_band_nodata_1611.vrt')
    vrt_xml = f'<VRTDataset rasterXSize="2" rasterYSize="2">\n  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>\n  <VRTRasterBand dataType="{dtype_str}" band="1">\n    <NoDataValue>{band0_sentinel}</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{p0}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n  <VRTRasterBand dataType="{dtype_str}" band="2">\n    <NoDataValue>{band1_sentinel}</NoDataValue>\n    <SimpleSource>\n      <SourceFilename relativeToVRT="0">{p1}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>\n    </SimpleSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


def test_multiband_uint16_per_band_sentinel_each_masked(tmp_path):
    """The previously-broken case: every band's sentinel must be NaN.

    Before the fix this returned dtype=float64 with band 0's (1,1) cell
    as NaN but band 1's (1,1) cell as the literal 65000.0.
    """
    vrt_path = _multiband_int_nodata_write_two_band_per_band_nodata_vrt(tmp_path)
    r = _read_vrt(vrt_path, band_nodata='first', mask_nodata=True)
    assert r.shape == (2, 2, 2)
    assert r.dtype == np.float64, f'expected float64 promotion, got {r.dtype}'
    assert np.isnan(r.values[1, 1, 0]), "band 0's sentinel pixel was not NaN-masked."
    assert np.isnan(r.values[1, 1, 1]), "band 1's sentinel pixel was not NaN-masked; the regression from issue #1611 has returned."  # noqa: E501
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
    vrt_path = _multiband_int_nodata_write_two_band_per_band_nodata_vrt(tmp_path, dtype_str='Int32', np_dtype=np.int32, band0_sentinel=-9999, band1_sentinel=-7777, band0_other=(10, 20, 30), band1_other=(40, 50, 60))  # noqa: E501
    r = _read_vrt(vrt_path, band_nodata='first', mask_nodata=True)
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
    vrt_path = _multiband_int_nodata_write_two_band_per_band_nodata_vrt(tmp_path, band0_sentinel=65535, band1_sentinel=65000, band1_other=(7, 8, 9))  # noqa: E501
    b1_no_sentinel = np.array([[7, 8], [9, 99]], dtype=np.uint16)
    import os
    p1 = os.path.join(os.path.dirname(vrt_path), 'vrt_b1_1611.tif')
    write(b1_no_sentinel, p1, nodata=65000, compression='none', tiled=False)
    r = _read_vrt(vrt_path, band_nodata='first', mask_nodata=True)
    assert r.dtype == np.float64, "Even when only band 0 has a present sentinel, the array still needs promotion so band 0's NaN can be expressed."  # noqa: E501
    assert np.isnan(r.values[1, 1, 0])
    assert r.values[1, 1, 1] == 99.0


def test_multiband_no_sentinel_present_anywhere_keeps_int_dtype(tmp_path):
    """When no band actually contains its declared sentinel, skip
    promotion entirely. Avoids a needless float64 cast on integer data.
    """
    vrt_path = _multiband_int_nodata_write_two_band_per_band_nodata_vrt(tmp_path, band0_sentinel=65535, band1_sentinel=65000, band0_other=(1, 2, 3), band1_other=(7, 8, 9))  # noqa: E501
    import os
    b0 = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    b1 = np.array([[7, 8], [9, 10]], dtype=np.uint16)
    p0 = os.path.join(os.path.dirname(vrt_path), 'vrt_b0_1611.tif')
    p1 = os.path.join(os.path.dirname(vrt_path), 'vrt_b1_1611.tif')
    write(b0, p0, nodata=65535, compression='none', tiled=False)
    write(b1, p1, nodata=65000, compression='none', tiled=False)
    r = _read_vrt(vrt_path, band_nodata='first')
    assert r.dtype == np.uint16
    assert r.values[1, 1, 0] == 4
    assert r.values[1, 1, 1] == 10


def test_multiband_per_band_out_of_range_sentinel_is_no_op(tmp_path):
    """A sentinel out of the integer dtype's range should be a no-op
    for that band rather than raising: the helper
    ``_int_nodata_in_range`` gates the cast.
    """
    vrt_path = _multiband_int_nodata_write_two_band_per_band_nodata_vrt(tmp_path, dtype_str='UInt16', np_dtype=np.uint16, band0_sentinel=65535, band1_sentinel=10, band0_other=(1, 2, 3), band1_other=(7, 8, 9))  # noqa: E501
    with open(vrt_path) as f:
        xml = f.read()
    xml = xml.replace('<NoDataValue>10</NoDataValue>', '<NoDataValue>-9999</NoDataValue>')
    with open(vrt_path, 'w') as f:
        f.write(xml)
    r = _read_vrt(vrt_path, band_nodata='first', mask_nodata=True)
    assert np.isnan(r.values[1, 1, 0])
    assert r.values[1, 1, 1] == 10.0 or r.values[1, 1, 1] == 10


def test_multiband_band_kwarg_still_per_band_post_pr1602(tmp_path):
    """Non-regression check that the band=N path still works.

    The fix here only changes the ``band is None`` branch; ``band=N``
    must still route through the single-band masking with its own
    sentinel.
    """
    vrt_path = _multiband_int_nodata_write_two_band_per_band_nodata_vrt(tmp_path)
    r0 = _read_vrt(vrt_path, band=0, band_nodata='first', mask_nodata=True)
    r1 = _read_vrt(vrt_path, band=1, band_nodata='first', mask_nodata=True)
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
    vrt_path = _multiband_int_nodata_write_two_band_per_band_nodata_vrt(tmp_path)
    r = _read_vrt(vrt_path, band_nodata='first')
    assert r.attrs.get('nodata') == 65535.0


# ---------------------------------------------------------------------------
# VRT resample algorithm validation
# ---------------------------------------------------------------------------


_UNSUPPORTED_RESAMPLE_EXC = (NotImplementedError, VRTUnsupportedError)


def _resample_alg_write_src(tmp_path) -> str:
    """Write a 4x4 uint16 source TIFF and return its path."""
    src = np.arange(16, dtype=np.uint16).reshape(4, 4)
    src_path = str(tmp_path / 'src.tif')
    write(src, src_path, compression='none', tiled=False)
    return src_path


def _resample_alg_write_vrt(tmp_path, xml: str, name: str = 'test.vrt') -> str:
    p = str(tmp_path / name)
    with open(p, 'w') as f:
        f.write(xml)
    return p


def _resample_alg_vrt_xml(src_path: str, *, alg_elem: str, dst_x: int = 2, dst_y: int = 2) -> str:
    """Render a VRT XML with a 4x4 SrcRect and configurable DstRect+Alg.

    ``alg_elem`` is the raw ``<ResampleAlg>...</ResampleAlg>`` element
    to splice into the ``<ComplexSource>``, or the empty string to
    omit it entirely.
    """
    return f'<VRTDataset rasterXSize="{dst_x}" rasterYSize="{dst_y}">\n  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>\n  <VRTRasterBand dataType="UInt16" band="1">\n    <ComplexSource>\n      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n      <SourceBand>1</SourceBand>\n      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n      <DstRect xOff="0" yOff="0" xSize="{dst_x}" ySize="{dst_y}"/>\n      {alg_elem}\n    </ComplexSource>\n  </VRTRasterBand>\n</VRTDataset>'  # noqa: E501


@pytest.mark.parametrize('alg', ['Bilinear', 'Cubic', 'CubicSpline', 'Lanczos', 'Average', 'Mode'])
def test_unsupported_resample_alg_raises(tmp_path, alg):
    """A ComplexSource declaring any non-nearest algorithm with a size
    change must raise ``NotImplementedError`` rather than return
    silently nearest-sampled pixels."""
    src_path = _resample_alg_write_src(tmp_path)
    xml = _resample_alg_vrt_xml(src_path, alg_elem=f'<ResampleAlg>{alg}</ResampleAlg>')
    vrt_path = _resample_alg_write_vrt(tmp_path, xml, f'{alg.lower()}.vrt')
    with pytest.raises(_UNSUPPORTED_RESAMPLE_EXC) as excinfo:
        _resample_alg_read_vrt_internal(vrt_path)
    msg = str(excinfo.value)
    assert alg in msg
    assert '1751' in msg


def test_unsupported_resample_alg_case_insensitive(tmp_path):
    """Algorithm names are matched case-insensitively: ``bilinear``
    (lowercase) is the same unsupported request as ``Bilinear``."""
    src_path = _resample_alg_write_src(tmp_path)
    xml = _resample_alg_vrt_xml(src_path, alg_elem='<ResampleAlg>bilinear</ResampleAlg>')
    vrt_path = _resample_alg_write_vrt(tmp_path, xml, 'lower.vrt')
    with pytest.raises(_UNSUPPORTED_RESAMPLE_EXC, match='bilinear'):
        _resample_alg_read_vrt_internal(vrt_path)


@pytest.mark.parametrize('alg', ['Nearest', 'NearestNeighbour', 'NearestNeighbor', 'NEAR', 'nearest', 'NEAREST', ''])  # noqa: E501
def test_nearest_variants_accepted(tmp_path, alg):
    """Nearest (and its case / spelling variants, plus empty text) is
    the implemented algorithm and must round-trip without raising."""
    src_path = _resample_alg_write_src(tmp_path)
    xml = _resample_alg_vrt_xml(src_path, alg_elem=f'<ResampleAlg>{alg}</ResampleAlg>')
    vrt_path = _resample_alg_write_vrt(tmp_path, xml, f"near_{alg or 'empty'}.vrt")
    arr, _ = _resample_alg_read_vrt_internal(vrt_path)
    assert arr.shape == (2, 2)


def test_missing_resample_alg_accepted(tmp_path):
    """Absent ``<ResampleAlg>`` (GDAL's nearest default) must still
    round-trip without raising."""
    src_path = _resample_alg_write_src(tmp_path)
    xml = _resample_alg_vrt_xml(src_path, alg_elem='')
    vrt_path = _resample_alg_write_vrt(tmp_path, xml, 'absent.vrt')
    arr, _ = _resample_alg_read_vrt_internal(vrt_path)
    assert arr.shape == (2, 2)


def test_bilinear_at_same_size_does_not_raise(tmp_path):
    """A ``Bilinear`` declaration with matching SrcRect/DstRect sizes
    is nearest-equivalent (no resample step runs) so the read is
    accepted.  This pins down the resample-site placement of the
    check -- a parse-time check would have rejected this case too."""
    src_path = _resample_alg_write_src(tmp_path)
    xml = _resample_alg_vrt_xml(src_path, alg_elem='<ResampleAlg>Bilinear</ResampleAlg>', dst_x=4, dst_y=4)  # noqa: E501
    vrt_path = _resample_alg_write_vrt(tmp_path, xml, 'bilinear_1to1.vrt')
    arr, _ = _resample_alg_read_vrt_internal(vrt_path)
    assert arr.shape == (4, 4)


# ---------------------------------------------------------------------------
# simple VRT mosaic positive coverage
# ---------------------------------------------------------------------------


_PIXEL_W = 0.001


_PIXEL_H = -0.001


_CRS = 4326


_NODATA = -9999.0


def _simple_mosaic_make_tile(tmp_dir, name: str, data: np.ndarray, origin_x: float, origin_y: float, *, nodata: float | None = _NODATA) -> str:  # noqa: E501
    """Write ``data`` as a single-band GeoTIFF anchored at the given origin.

    Returns the on-disk path. ``data`` shape is ``(H, W)``.
    """
    height, width = data.shape
    y = np.array([origin_y + _PIXEL_H * (i + 0.5) for i in range(height)])
    x = np.array([origin_x + _PIXEL_W * (j + 0.5) for j in range(width)])
    attrs = {'crs': _CRS}
    if nodata is not None:
        attrs['nodata'] = nodata
    raster = xr.DataArray(data, dims=['y', 'x'], coords={'y': y, 'x': x}, attrs=attrs)
    path = os.path.join(tmp_dir, name)
    to_geotiff(raster, path, nodata=nodata)
    return path


def _simple_mosaic_make_multiband_tile(tmp_dir, name: str, data: np.ndarray, origin_x: float, origin_y: float) -> str:  # noqa: E501
    """Write a multi-band GeoTIFF anchored at the given origin.

    ``data`` shape is ``(H, W, B)``.
    """
    height, width, nbands = data.shape
    y = np.array([origin_y + _PIXEL_H * (i + 0.5) for i in range(height)])
    x = np.array([origin_x + _PIXEL_W * (j + 0.5) for j in range(width)])
    raster = xr.DataArray(data, dims=['y', 'x', 'band'], coords={'y': y, 'x': x, 'band': np.arange(nbands)}, attrs={'crs': _CRS})  # noqa: E501
    path = os.path.join(tmp_dir, name)
    to_geotiff(raster, path)
    return path


@pytest.fixture
def simple_mosaic_mosaic_2x1(tmp_path):
    """Two 32x32 float32 tiles side-by-side, west and east.

    Yields ``(vrt_path, expected_array, origin_x, origin_y)``. The
    expected array is the horizontal concatenation of the two source
    arrays.
    """
    td = tmp_path / 'tmp_2369_2x1'
    td.mkdir()
    td = str(td)
    height, width = (32, 32)
    left_data = np.arange(height * width, dtype=np.float32).reshape(height, width)
    right_data = (left_data + height * width).astype(np.float32)
    origin_x, origin_y = (-120.0, 45.0)
    left_path = _simple_mosaic_make_tile(td, 'left.tif', left_data, origin_x, origin_y)
    right_path = _simple_mosaic_make_tile(td, 'right.tif', right_data, origin_x + _PIXEL_W * width, origin_y)  # noqa: E501
    vrt_path = os.path.join(td, 'simple_mosaic_mosaic_2x1.vrt')
    _write_vrt_internal(vrt_path, [left_path, right_path])
    expected = np.concatenate([left_data, right_data], axis=1)
    yield (vrt_path, expected, origin_x, origin_y)


@pytest.fixture
def simple_mosaic_mosaic_2x2(tmp_path):
    """Four 32x32 float32 tiles arranged 2 rows by 2 cols.

    Yields ``(vrt_path, expected_array, origin_x, origin_y)`` with the
    expected array stitched in (row, col) order.
    """
    td = tmp_path / 'tmp_2369_2x2'
    td.mkdir()
    td = str(td)
    h, w = (32, 32)
    tile_nw = np.full((h, w), 1.0, dtype=np.float32)
    tile_ne = np.full((h, w), 2.0, dtype=np.float32)
    tile_sw = np.full((h, w), 3.0, dtype=np.float32)
    tile_se = np.full((h, w), 4.0, dtype=np.float32)
    origin_x, origin_y = (-120.0, 45.0)
    nw_path = _simple_mosaic_make_tile(td, 'nw.tif', tile_nw, origin_x, origin_y)
    ne_path = _simple_mosaic_make_tile(td, 'ne.tif', tile_ne, origin_x + _PIXEL_W * w, origin_y)
    sw_path = _simple_mosaic_make_tile(td, 'sw.tif', tile_sw, origin_x, origin_y + _PIXEL_H * h)
    se_path = _simple_mosaic_make_tile(td, 'se.tif', tile_se, origin_x + _PIXEL_W * w, origin_y + _PIXEL_H * h)  # noqa: E501
    vrt_path = os.path.join(td, 'simple_mosaic_mosaic_2x2.vrt')
    _write_vrt_internal(vrt_path, [nw_path, ne_path, sw_path, se_path])
    top = np.concatenate([tile_nw, tile_ne], axis=1)
    bottom = np.concatenate([tile_sw, tile_se], axis=1)
    expected = np.concatenate([top, bottom], axis=0)
    yield (vrt_path, expected, origin_x, origin_y)


@pytest.fixture
def simple_mosaic_mosaic_multiband_2x1(tmp_path):
    """Two 3-band 32x32 float32 tiles side-by-side."""
    td = tmp_path / 'tmp_2369_mb_2x1'
    td.mkdir()
    td = str(td)
    h, w, b = (32, 32, 3)
    rng = np.random.default_rng(2369)
    left_data = rng.random((h, w, b), dtype=np.float32)
    right_data = rng.random((h, w, b), dtype=np.float32)
    origin_x, origin_y = (-120.0, 45.0)
    left_path = _simple_mosaic_make_multiband_tile(td, 'left_mb.tif', left_data, origin_x, origin_y)
    right_path = _simple_mosaic_make_multiband_tile(td, 'right_mb.tif', right_data, origin_x + _PIXEL_W * w, origin_y)  # noqa: E501
    vrt_path = os.path.join(td, 'mosaic_mb.vrt')
    _write_vrt_internal(vrt_path, [left_path, right_path])
    expected = np.stack([np.concatenate([left_data[..., k], right_data[..., k]], axis=1) for k in range(b)], axis=-1)  # noqa: E501
    yield (vrt_path, expected, origin_x, origin_y)


def _simple_mosaic_assert_attrs_ok(result, *, expected_nodata=None, expected_origin_x=None, expected_origin_y=None):  # noqa: E501
    """Common attr assertions for VRT reads in this module.

    Checks that ``crs`` and ``transform`` are present and consistent
    with the fixture constants, and optionally that ``nodata`` matches.
    When ``expected_origin_x`` / ``expected_origin_y`` are passed, the
    transform's origin entries are checked too -- pixel size alone is
    not enough to catch a translation bug.
    """
    assert 'crs' in result.attrs, f'crs missing from attrs; have {sorted(result.attrs)}'
    crs_val = result.attrs['crs']
    if isinstance(crs_val, int):
        assert crs_val == _CRS
    else:
        assert crs_val, 'crs attr is present but empty'
        assert 'WGS' in str(crs_val) or '4326' in str(crs_val), f'crs attr does not look like EPSG:4326: {crs_val!r}'  # noqa: E501
    assert 'transform' in result.attrs, f'transform missing from attrs; have {sorted(result.attrs)}'
    transform = result.attrs['transform']
    assert len(transform) == 6, f'transform should be a 6-tuple, got {transform!r}'
    assert transform[0] == pytest.approx(_PIXEL_W), f'transform pixel width = {transform[0]}, expected {_PIXEL_W}'  # noqa: E501
    assert transform[4] == pytest.approx(_PIXEL_H), f'transform pixel height = {transform[4]}, expected {_PIXEL_H}'  # noqa: E501
    if expected_origin_x is not None:
        assert transform[2] == pytest.approx(expected_origin_x), f'transform origin_x = {transform[2]}, expected {expected_origin_x}'  # noqa: E501
    if expected_origin_y is not None:
        assert transform[5] == pytest.approx(expected_origin_y), f'transform origin_y = {transform[5]}, expected {expected_origin_y}'  # noqa: E501
    if expected_nodata is not None:
        assert 'nodata' in result.attrs, f'nodata missing from attrs; have {sorted(result.attrs)}'
        assert result.attrs['nodata'] == pytest.approx(expected_nodata)


def _simple_mosaic_assert_coords_monotonic(result, *, expected_origin_x, expected_origin_y):
    """Check that x/y coords are monotonic and start at the expected origin
    (within half a pixel: TIFF coords are pixel centers, not corners).
    """
    x = np.asarray(result['x'].values)
    y = np.asarray(result['y'].values)
    assert np.all(np.diff(x) > 0), 'x coord is not strictly increasing'
    assert np.all(np.diff(y) < 0), 'y coord is not strictly decreasing'
    assert x[0] == pytest.approx(expected_origin_x + _PIXEL_W * 0.5)
    assert y[0] == pytest.approx(expected_origin_y + _PIXEL_H * 0.5)


def test_eager_2x1_mosaic_values_coords_attrs(simple_mosaic_mosaic_2x1):
    """Eager read of a 2x1 horizontal mosaic returns the concatenated
    pixel block, with monotonic coords and the fixture's crs / transform
    / nodata on attrs.
    """
    vrt_path, expected, ox, oy = simple_mosaic_mosaic_2x1
    result = _read_vrt(vrt_path)
    assert result.shape == expected.shape, f'eager 2x1 shape {result.shape}, expected {expected.shape}'  # noqa: E501
    np.testing.assert_array_equal(result.values, expected)
    _simple_mosaic_assert_coords_monotonic(result, expected_origin_x=ox, expected_origin_y=oy)
    _simple_mosaic_assert_attrs_ok(result, expected_nodata=_NODATA, expected_origin_x=ox, expected_origin_y=oy)  # noqa: E501


def test_eager_2x2_mosaic_values_coords_attrs(simple_mosaic_mosaic_2x2):
    """Eager read of a 2x2 mosaic stitches tiles in the right order.

    Each tile has a distinct constant value, so a misordered placement
    surfaces immediately in the value assertion rather than appearing
    only as a numeric diff.
    """
    vrt_path, expected, ox, oy = simple_mosaic_mosaic_2x2
    result = _read_vrt(vrt_path)
    assert result.shape == expected.shape, f'eager 2x2 shape {result.shape}, expected {expected.shape}'  # noqa: E501
    np.testing.assert_array_equal(result.values, expected)
    _simple_mosaic_assert_coords_monotonic(result, expected_origin_x=ox, expected_origin_y=oy)
    _simple_mosaic_assert_attrs_ok(result, expected_nodata=_NODATA, expected_origin_x=ox, expected_origin_y=oy)  # noqa: E501


def test_windowed_read_aligned_with_source_boundary(simple_mosaic_mosaic_2x1):
    """A window crossing the seam between the two source tiles returns
    the same pixels as slicing the full mosaic.

    The window picked here covers the right half of the left tile and
    the left half of the right tile: both halves land on whole-pixel
    boundaries inside their respective sources, so this is the
    "request lines up with source pixels" case from the issue.
    """
    vrt_path, expected, ox, oy = simple_mosaic_mosaic_2x1
    h = expected.shape[0]
    r0, c0, r1, c1 = (0, 16, h, 48)
    result = _read_vrt(vrt_path, window=(r0, c0, r1, c1))
    np.testing.assert_array_equal(result.values, expected[r0:r1, c0:c1])
    full = _read_vrt(vrt_path)
    np.testing.assert_array_equal(np.asarray(result['x'].values), np.asarray(full['x'].values)[c0:c1])  # noqa: E501
    np.testing.assert_array_equal(np.asarray(result['y'].values), np.asarray(full['y'].values)[r0:r1])  # noqa: E501
    expected_window_ox = ox + _PIXEL_W * c0
    expected_window_oy = oy + _PIXEL_H * r0
    _simple_mosaic_assert_attrs_ok(result, expected_nodata=_NODATA, expected_origin_x=expected_window_ox, expected_origin_y=expected_window_oy)  # noqa: E501


def test_dask_2x1_mosaic_multi_chunk_matches_eager(simple_mosaic_mosaic_2x1):
    """Dask read with chunks smaller than the mosaic returns the same
    pixels as the eager read, and uses a real multi-block dask graph.
    """
    vrt_path, expected, ox, oy = simple_mosaic_mosaic_2x1
    chunked = _read_vrt(vrt_path, chunks=(16, 16))
    assert isinstance(chunked.data, da.Array), f'expected dask Array, got {type(chunked.data).__name__}'  # noqa: E501
    assert chunked.data.numblocks == (2, 4), f'expected 2x4 blocks, got {chunked.data.numblocks}'
    computed = chunked.compute()
    np.testing.assert_array_equal(computed.values, expected)
    _simple_mosaic_assert_coords_monotonic(computed, expected_origin_x=ox, expected_origin_y=oy)
    _simple_mosaic_assert_attrs_ok(computed, expected_nodata=_NODATA, expected_origin_x=ox, expected_origin_y=oy)  # noqa: E501


def test_dask_2x2_mosaic_multi_chunk_matches_eager(simple_mosaic_mosaic_2x2):
    """Dask read of the 2x2 mosaic with chunk size below tile size.

    Chunks of 16 split each 32x32 tile into 2x2 blocks. The full
    mosaic is 64x64 so the resulting dask array is 4x4 blocks.
    """
    vrt_path, expected, ox, oy = simple_mosaic_mosaic_2x2
    chunked = _read_vrt(vrt_path, chunks=(16, 16))
    assert isinstance(chunked.data, da.Array)
    assert chunked.data.numblocks == (4, 4), f'expected 4x4 blocks, got {chunked.data.numblocks}'
    computed = chunked.compute()
    np.testing.assert_array_equal(computed.values, expected)
    _simple_mosaic_assert_coords_monotonic(computed, expected_origin_x=ox, expected_origin_y=oy)
    _simple_mosaic_assert_attrs_ok(computed, expected_nodata=_NODATA, expected_origin_x=ox, expected_origin_y=oy)  # noqa: E501


def test_eager_multiband_2x1_mosaic(simple_mosaic_mosaic_multiband_2x1):
    """Eager read of a multi-band 2x1 mosaic returns one stitched plane
    per band.

    Multi-band VRT reads return shape ``(H, W, B)`` to match the
    on-disk layout; assert per-band values against the stack built in
    the fixture.
    """
    vrt_path, expected, ox, oy = simple_mosaic_mosaic_multiband_2x1
    result = _read_vrt(vrt_path)
    assert result.shape == expected.shape, f'multiband 2x1 shape {result.shape}, expected {expected.shape}'  # noqa: E501
    np.testing.assert_array_equal(result.values, expected)
    _simple_mosaic_assert_coords_monotonic(result, expected_origin_x=ox, expected_origin_y=oy)
    _simple_mosaic_assert_attrs_ok(result, expected_origin_x=ox, expected_origin_y=oy)


def test_dask_multiband_2x1_mosaic_matches_eager(simple_mosaic_mosaic_multiband_2x1):
    """Dask read of the multi-band 2x1 mosaic with sub-tile chunks must
    match the eager read pixel-for-pixel across every band.

    Chunking exercises per-block band handling: a bug that loses a
    band on one chunk but not another would not appear in the eager
    test above.
    """
    vrt_path, expected, ox, oy = simple_mosaic_mosaic_multiband_2x1
    eager = _read_vrt(vrt_path)
    chunked = _read_vrt(vrt_path, chunks=(16, 16))
    assert isinstance(chunked.data, da.Array), f'expected dask Array, got {type(chunked.data).__name__}'  # noqa: E501
    computed = chunked.compute()
    assert computed.shape == eager.shape
    np.testing.assert_array_equal(computed.values, eager.values)
    np.testing.assert_array_equal(computed.values, expected)
    _simple_mosaic_assert_coords_monotonic(computed, expected_origin_x=ox, expected_origin_y=oy)
    _simple_mosaic_assert_attrs_ok(computed, expected_origin_x=ox, expected_origin_y=oy)
