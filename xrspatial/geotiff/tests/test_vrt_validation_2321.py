"""Regression tests for issue #2329 / parent #2321 sub-task 2.

Centralised VRT capability validation: one negative test per rule the
validator enforces. Every test asserts ``VRTUnsupportedError`` is
raised at validator-time (i.e. before any source decode), and that the
direct ``read_vrt`` entry point and the dispatched
``open_geotiff('foo.vrt')`` entry point produce the same error type
and the same message for the same bad input.

The validator accepts an already-parsed ``VRTDataset``; the tests
exercise it both directly (with a parsed structure) and through the
two public entry points so the wiring is covered too.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._backends.vrt import read_vrt
from xrspatial.geotiff._errors import (
    GeoTIFFAmbiguousMetadataError,
    MixedBandMetadataError,
    RotatedTransformError,
    UnparseableCRSError,
    VRTUnsupportedError,
)
from xrspatial.geotiff._vrt import parse_vrt
from xrspatial.geotiff._vrt_validation import validate_parsed_vrt
from xrspatial.geotiff._writer import write


def _write_src(tmp_path, name: str = 'src.tif',
               shape=(4, 4), dtype=np.uint16) -> str:
    """Write a small source TIFF and return its path."""
    arr = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    p = str(tmp_path / name)
    write(arr, p, compression='none', tiled=False)
    return p


def _write_vrt(tmp_path, xml: str, name: str = 'test.vrt') -> str:
    p = str(tmp_path / name)
    with open(p, 'w') as f:
        f.write(xml)
    return p


def _parse(tmp_path, xml: str, name: str = 'test.vrt'):
    """Helper: write + parse a VRT XML, return (path, parsed)."""
    import os
    path = _write_vrt(tmp_path, xml, name)
    parsed = parse_vrt(xml, os.path.dirname(os.path.abspath(path)))
    return path, parsed


# ---------------------------------------------------------------------------
# Subclassing contract
# ---------------------------------------------------------------------------


def test_vrt_unsupported_error_is_geotiff_metadata_error():
    """``VRTUnsupportedError`` must subclass
    ``GeoTIFFAmbiguousMetadataError`` (and ``ValueError`` via the base)
    so ``except ValueError`` callers keep catching VRT failures."""
    assert issubclass(VRTUnsupportedError, GeoTIFFAmbiguousMetadataError)
    assert issubclass(VRTUnsupportedError, ValueError)


# ---------------------------------------------------------------------------
# Rule 1: band count sanity (zero bands)
# ---------------------------------------------------------------------------


_NO_BANDS_VRT = """<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
</VRTDataset>"""


def test_zero_bands_raises_vrt_unsupported(tmp_path):
    path, parsed = _parse(tmp_path, _NO_BANDS_VRT, 'nobands.vrt')
    with pytest.raises(VRTUnsupportedError, match='band'):
        validate_parsed_vrt(parsed, source=path, mode='read')


def test_zero_bands_parity_across_entry_points(tmp_path):
    path = _write_vrt(tmp_path, _NO_BANDS_VRT, 'nobands_parity.vrt')
    with pytest.raises(VRTUnsupportedError) as a:
        read_vrt(path)
    with pytest.raises(VRTUnsupportedError) as b:
        open_geotiff(path)
    assert type(a.value) is type(b.value)
    assert str(a.value) == str(b.value)


# ---------------------------------------------------------------------------
# Rule 2: dtype compatibility / unsupported dataType
# (``parse_vrt`` already rejects unknown dtype tokens; the validator
# rejects ``Complex`` placeholders that slip past via a typo'd map.
# The negative we exercise here is per-band dtype that is not in the
# supported numpy kind set: complex bands cannot ride through.)
# ---------------------------------------------------------------------------


def test_complex_dtype_band_rejected_by_validator(tmp_path):
    """Even if a complex numpy dtype appears on a parsed band (e.g.
    via a future _DTYPE_MAP entry), the validator must reject the
    band before any read execution begins."""
    src_path = _write_src(tmp_path)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, parsed = _parse(tmp_path, xml, 'complex.vrt')
    # Patch the parsed band dtype to complex128 to simulate a
    # hypothetical map regression. The validator should refuse it.
    parsed.bands[0].dtype = np.dtype('complex128')
    with pytest.raises(VRTUnsupportedError, match='dtype'):
        validate_parsed_vrt(parsed, source=path, mode='read')


# ---------------------------------------------------------------------------
# Rule 3: transform orientation (rotation / shear)
# ---------------------------------------------------------------------------


def test_rotated_transform_rejected_without_opt_in(tmp_path):
    """A VRT with non-zero rotation/shear terms in its GeoTransform must
    be rejected by the validator unless ``allow_rotated=True`` is
    passed."""
    src_path = _write_src(tmp_path)
    # GeoTransform: origin_x, res_x, skew_x, origin_y, skew_y, res_y
    # Non-zero skew_x / skew_y => rotated.
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.1, 0.0, 0.1, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, parsed = _parse(tmp_path, xml, 'rotated.vrt')
    # Rotated transforms raise the existing typed
    # ``RotatedTransformError`` for backward compatibility with
    # callers that ``except RotatedTransformError``; the centralised
    # validator's value here is in adding the source path to the
    # message and lifting the rejection ahead of any decode.
    with pytest.raises(RotatedTransformError, match='rotat'):
        validate_parsed_vrt(parsed, source=path, mode='read',
                            allow_rotated=False)
    # Opt-in path returns silently.
    validate_parsed_vrt(parsed, source=path, mode='read',
                        allow_rotated=True)


# ---------------------------------------------------------------------------
# Rule 4: SrcRect sanity (negative size / negative offset)
# ---------------------------------------------------------------------------


def test_negative_src_rect_size_rejected(tmp_path):
    src_path = _write_src(tmp_path)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="-1" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, parsed = _parse(tmp_path, xml, 'neg_src.vrt')
    with pytest.raises(VRTUnsupportedError, match='SrcRect'):
        validate_parsed_vrt(parsed, source=path, mode='read')


def test_negative_src_rect_offset_rejected(tmp_path):
    src_path = _write_src(tmp_path)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="-1" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, parsed = _parse(tmp_path, xml, 'neg_src_off.vrt')
    with pytest.raises(VRTUnsupportedError, match='SrcRect'):
        validate_parsed_vrt(parsed, source=path, mode='read')


# ---------------------------------------------------------------------------
# Rule 5: DstRect sanity (negative size, plus out-of-VRT-extent rect)
# ---------------------------------------------------------------------------


def test_negative_dst_rect_size_rejected(tmp_path):
    src_path = _write_src(tmp_path)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="-1" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, parsed = _parse(tmp_path, xml, 'neg_dst.vrt')
    with pytest.raises(VRTUnsupportedError, match='DstRect'):
        validate_parsed_vrt(parsed, source=path, mode='read')


def test_dst_rect_outside_vrt_extent_rejected(tmp_path):
    """A DstRect that lands entirely outside the VRT's
    ``rasterXSize x rasterYSize`` extent contributes nothing and is a
    malformed mosaic. Reject up front."""
    src_path = _write_src(tmp_path)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="100" yOff="100" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, parsed = _parse(tmp_path, xml, 'out_dst.vrt')
    with pytest.raises(VRTUnsupportedError, match='DstRect'):
        validate_parsed_vrt(parsed, source=path, mode='read')


# ---------------------------------------------------------------------------
# Rule 6: pixel-size compatibility (zero pixel size in transform)
# ---------------------------------------------------------------------------


def test_zero_pixel_size_rejected(tmp_path):
    """A GeoTransform with zero res_x or res_y produces a degenerate
    coord array and divides by zero in coord generation. Reject up
    front."""
    src_path = _write_src(tmp_path)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 0.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, parsed = _parse(tmp_path, xml, 'zero_px.vrt')
    with pytest.raises(VRTUnsupportedError, match='pixel size'):
        validate_parsed_vrt(parsed, source=path, mode='read')


# ---------------------------------------------------------------------------
# Rule 7: unsupported resampling algorithm
# ---------------------------------------------------------------------------


def test_unsupported_resample_alg_rejected_at_validate(tmp_path):
    """A ComplexSource with a non-nearest resampler and differing
    SrcRect/DstRect sizes must be rejected by the validator before
    any pixels are read."""
    src_path = _write_src(tmp_path)
    xml = f"""<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <ResampleAlg>Bilinear</ResampleAlg>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, parsed = _parse(tmp_path, xml, 'bilinear.vrt')
    with pytest.raises(VRTUnsupportedError, match='[Rr]esampl'):
        validate_parsed_vrt(parsed, source=path, mode='read')


# ---------------------------------------------------------------------------
# Rule 8: nodata policy (mixed per-band sentinels without opt-in)
# ---------------------------------------------------------------------------


def test_mixed_band_nodata_rejected_without_opt_in(tmp_path):
    """A two-band VRT with disagreeing per-band ``<NoDataValue>`` must
    be rejected unless ``band_nodata='first'`` is passed."""
    src_path = _write_src(tmp_path, name='src_mixed.tif')
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>0</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <NoDataValue>9999</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, _ = _parse(tmp_path, xml, 'mixed_nodata.vrt')
    # Mixed per-band sentinels raise the existing typed
    # ``MixedBandMetadataError`` (registered via the
    # ``validate_read_metadata`` hook). The centralised
    # ``validate_parsed_vrt`` delegates the disagreement-detection
    # logic to that hook so the ``None``-counts-as-undeclared
    # semantics stay canonical in one place. The public entry points
    # run both the validator and ``validate_read_metadata``, so the
    # rejection still surfaces at boundary time.
    with pytest.raises(MixedBandMetadataError):
        read_vrt(path)
    with pytest.raises(MixedBandMetadataError):
        open_geotiff(path)
    # Opt-in returns silently.
    r = read_vrt(path, band_nodata='first')
    assert r.shape == (4, 4, 2)


# ---------------------------------------------------------------------------
# Rule 9: CRS compatibility
# (the parsed VRTDataset only carries one ``crs_wkt`` field, so the
# CRS-mismatch case happens when a caller's expected CRS differs from
# the file's. The validator surfaces an unparseable CRS string up
# front unless ``allow_unparseable_crs=True``.)
# ---------------------------------------------------------------------------


def test_unparseable_crs_rejected_without_opt_in(tmp_path):
    src_path = _write_src(tmp_path)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <SRS>GARBAGE_CRS_NOT_A_REAL_WKT</SRS>
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, parsed = _parse(tmp_path, xml, 'badcrs.vrt')
    # CRS strings that pyproj cannot resolve raise the existing typed
    # ``UnparseableCRSError`` so callers that already
    # ``except UnparseableCRSError`` keep working.
    with pytest.raises(UnparseableCRSError):
        validate_parsed_vrt(parsed, source=path, mode='read',
                            allow_unparseable_crs=False)
    # Opt-in path returns silently.
    validate_parsed_vrt(parsed, source=path, mode='read',
                        allow_unparseable_crs=True)


# ---------------------------------------------------------------------------
# Entry-point parity: direct read_vrt and open_geotiff produce the same
# rejection (same error type and same message) for the same bad input.
# ---------------------------------------------------------------------------


_BILINEAR_XML_TEMPLATE = """<VRTDataset rasterXSize="2" rasterYSize="2">
  <GeoTransform>0.0, 2.0, 0.0, 0.0, 0.0, -2.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <ResampleAlg>Bilinear</ResampleAlg>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>"""


def test_resample_parity_across_entry_points(tmp_path):
    src_path = _write_src(tmp_path)
    xml = _BILINEAR_XML_TEMPLATE.format(src=src_path)
    path = _write_vrt(tmp_path, xml, 'bilinear_parity.vrt')

    with pytest.raises(VRTUnsupportedError) as a:
        read_vrt(path)
    with pytest.raises(VRTUnsupportedError) as b:
        open_geotiff(path)
    assert type(a.value) is type(b.value)
    assert str(a.value) == str(b.value)


def test_rotated_parity_across_entry_points(tmp_path):
    src_path = _write_src(tmp_path)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.1, 0.0, 0.1, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path = _write_vrt(tmp_path, xml, 'rotated_parity.vrt')

    # Both entry points raise the same typed error
    # (``RotatedTransformError`` here, since rotated transforms keep
    # their pre-existing subclass) with the same message.
    with pytest.raises(RotatedTransformError) as a:
        read_vrt(path)
    with pytest.raises(RotatedTransformError) as b:
        open_geotiff(path)
    assert type(a.value) is type(b.value)
    assert str(a.value) == str(b.value)


# ---------------------------------------------------------------------------
# Chunked (dask) path: rejection happens at graph build, not in a chunk
# function. Same error type and message as the eager path.
# ---------------------------------------------------------------------------


def test_unsupported_resample_chunked_raises_at_build(tmp_path):
    """The chunked dispatcher must run the validator before building
    the dask graph so the failure surfaces at construction time, not
    deep in a chunk function during ``compute()``."""
    src_path = _write_src(tmp_path)
    xml = _BILINEAR_XML_TEMPLATE.format(src=src_path)
    path = _write_vrt(tmp_path, xml, 'bilinear_chunked.vrt')

    with pytest.raises(VRTUnsupportedError):
        read_vrt(path, chunks=2)


# ---------------------------------------------------------------------------
# Sanity: a well-formed minimal VRT validates with no exception.
# ---------------------------------------------------------------------------


def test_well_formed_vrt_validates_silently(tmp_path):
    src_path = _write_src(tmp_path)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    path, parsed = _parse(tmp_path, xml, 'good.vrt')
    # No exception:
    validate_parsed_vrt(parsed, source=path, mode='read')
    # And the public entry points read it without raising the
    # validator error.
    result = read_vrt(path)
    assert result.shape == (4, 4)
