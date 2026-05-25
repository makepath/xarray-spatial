"""Regression tests for issue #2371 (sub-task of epic #2342).

The centralised VRT capability validator
(``xrspatial.geotiff._vrt_validation.validate_parsed_vrt``, exposed as
``validate_vrt_capability``) now covers four additional rejection
paths and is wired into both the internal ``_vrt.read_vrt`` and the
public ``_backends/vrt.read_vrt`` entry points. The four paths:

1. Nested VRTs: a ``.vrt`` referenced as a ``SourceFilename`` inside
   another VRT.
2. Warped VRTs declaring a ``<GDALWarpOptions>`` block at the dataset
   or band level (the band-level ``subClass="VRTWarpedRasterBand"``
   marker is already rejected by the existing parse-time subclass
   check).
3. Resample algorithm beyond nearest when SrcRect and DstRect sizes
   differ (extended from ``_check_resample_alg_supported`` so the
   chunked path also rejects at graph-build time).
4. Complex mask / alpha source semantics: per-source
   ``<UseMaskBand>true</UseMaskBand>`` flags and per-source
   ``<MaskBand>`` children.

Each test asserts the rejection fires at validator time (before any
source decode) and that the message names the offending source path
or feature so a caller can locate the bad source without re-parsing
the VRT XML themselves.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._backends.vrt import read_vrt as _public_read_vrt
from xrspatial.geotiff._errors import (
    GeoTIFFAmbiguousMetadataError,
    UnsupportedGeoTIFFFeatureError,
    VRTUnsupportedError,
)
from xrspatial.geotiff._vrt import parse_vrt
from xrspatial.geotiff._vrt import read_vrt as _internal_read_vrt
from xrspatial.geotiff._vrt_validation import (
    validate_parsed_vrt,
    validate_vrt_capability,
)
from xrspatial.geotiff._writer import write


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _write_src(tmp_path, name: str = 'src_2371.tif',
               shape=(4, 4), dtype=np.uint16) -> str:
    """Write a small source TIFF and return its path."""
    arr = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    p = str(tmp_path / name)
    write(arr, p, compression='none', tiled=False)
    return p


def _write_vrt(tmp_path, xml: str, name: str = 'mosaic_2371.vrt') -> str:
    """Write a VRT XML to disk and return its path."""
    p = str(tmp_path / name)
    with open(p, 'w') as f:
        f.write(xml)
    return p


def _parse(tmp_path, xml: str, name: str = 'mosaic_2371.vrt'):
    """Write + parse a VRT XML. Returns ``(path, parsed)``."""
    path = _write_vrt(tmp_path, xml, name)
    parsed = parse_vrt(xml, os.path.dirname(os.path.abspath(path)))
    return path, parsed


# ---------------------------------------------------------------------------
# Public-alias contract
# ---------------------------------------------------------------------------


def test_validate_vrt_capability_alias_resolves_to_validate_parsed_vrt():
    """``validate_vrt_capability`` is the public alias matching the
    issue text. It must resolve to the same underlying callable as
    ``validate_parsed_vrt`` so both names share one implementation."""
    assert validate_vrt_capability is validate_parsed_vrt


# ---------------------------------------------------------------------------
# Rule: nested VRT (a .vrt referenced as a SourceFilename)
# ---------------------------------------------------------------------------


def _nested_vrt_xml(inner_vrt_path: str) -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{inner_vrt_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""


def test_nested_vrt_rejected_at_validator(tmp_path):
    """A ``SimpleSource`` referencing another ``.vrt`` file must raise
    ``VRTUnsupportedError`` at validate time with both VRT paths in the
    message."""
    # Build an inner VRT that on its own is well-formed, then build an
    # outer VRT that references it as a source.
    src_path = _write_src(tmp_path)
    inner_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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
    inner_path = _write_vrt(tmp_path, inner_xml, 'inner_2371.vrt')

    outer_path, parsed = _parse(
        tmp_path, _nested_vrt_xml(inner_path), 'outer_2371.vrt'
    )

    with pytest.raises(VRTUnsupportedError) as excinfo:
        validate_parsed_vrt(parsed, source=outer_path, mode='read')
    msg = str(excinfo.value)
    # Outer path appears as the failing VRT.
    assert outer_path in msg
    # Inner path is named so the caller can locate the bad source.
    assert inner_path in msg
    # Message names the failure mode.
    assert 'Nested' in msg or 'nested' in msg


def test_nested_vrt_uppercase_extension_rejected(tmp_path):
    """``.VRT`` (uppercase) trips the same rejection: extension matching
    must be case-insensitive so Windows-style emitters are caught."""
    src_path = _write_src(tmp_path)
    inner_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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
    inner_path = _write_vrt(tmp_path, inner_xml, 'INNER_2371.VRT')

    outer_path, parsed = _parse(
        tmp_path, _nested_vrt_xml(inner_path), 'outer_upper_2371.vrt'
    )
    with pytest.raises(VRTUnsupportedError, match='[Nn]ested'):
        validate_parsed_vrt(parsed, source=outer_path, mode='read')


def test_nested_vrt_rejected_via_public_read_vrt(tmp_path):
    """The public ``_backends/vrt.read_vrt`` entry point must surface
    the same rejection as the direct validator call."""
    src_path = _write_src(tmp_path)
    inner_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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
    inner_path = _write_vrt(tmp_path, inner_xml, 'inner_pub_2371.vrt')
    outer_path = _write_vrt(
        tmp_path,
        _nested_vrt_xml(inner_path),
        'outer_pub_2371.vrt',
    )
    with pytest.raises(VRTUnsupportedError, match='[Nn]ested'):
        _public_read_vrt(outer_path)


def test_nested_vrt_rejected_via_open_geotiff(tmp_path):
    """The dispatched ``open_geotiff('foo.vrt')`` path runs through the
    same backend wrapper and must produce the same rejection."""
    src_path = _write_src(tmp_path)
    inner_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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
    inner_path = _write_vrt(tmp_path, inner_xml, 'inner_og_2371.vrt')
    outer_path = _write_vrt(
        tmp_path,
        _nested_vrt_xml(inner_path),
        'outer_og_2371.vrt',
    )
    with pytest.raises(VRTUnsupportedError, match='[Nn]ested'):
        open_geotiff(outer_path)


def test_nested_vrt_rejected_via_internal_read_vrt(tmp_path):
    """The internal ``_vrt.read_vrt`` is now routed through the
    validator too (issue #2371 wires the same gate at both entry
    points). A direct call must produce the rejection without going
    through the public backend wrapper."""
    src_path = _write_src(tmp_path)
    inner_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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
    inner_path = _write_vrt(tmp_path, inner_xml, 'inner_int_2371.vrt')
    outer_path = _write_vrt(
        tmp_path,
        _nested_vrt_xml(inner_path),
        'outer_int_2371.vrt',
    )
    with pytest.raises(VRTUnsupportedError, match='[Nn]ested'):
        _internal_read_vrt(outer_path)


# ---------------------------------------------------------------------------
# Rule: warped VRT (``<GDALWarpOptions>`` block)
# ---------------------------------------------------------------------------


_WARP_DATASET_XML = """<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <GDALWarpOptions>
    <WarpMemoryLimit>64.0</WarpMemoryLimit>
    <ResampleAlg>NearestNeighbour</ResampleAlg>
  </GDALWarpOptions>
  <VRTRasterBand dataType="UInt16" band="1"/>
</VRTDataset>"""


def test_warp_options_dataset_level_rejected_at_parse(tmp_path):
    """A dataset-level ``<GDALWarpOptions>`` block raises
    ``UnsupportedGeoTIFFFeatureError`` during ``parse_vrt``. The parser
    rejects the element via ``_UNSUPPORTED_DATASET_TAGS`` so callers
    that route through the validator still see a typed failure (the
    parse step runs first, before the validator is reached)."""
    path = _write_vrt(tmp_path, _WARP_DATASET_XML, 'warp_ds_2371.vrt')
    with pytest.raises(
        UnsupportedGeoTIFFFeatureError, match='GDALWarpOptions'
    ):
        parse_vrt(_WARP_DATASET_XML, os.path.dirname(path))


def test_warp_options_dataset_level_rejected_via_public_read_vrt(tmp_path):
    """The public ``_backends/vrt.read_vrt`` entry point surfaces the
    same warp rejection."""
    path = _write_vrt(tmp_path, _WARP_DATASET_XML, 'warp_pub_2371.vrt')
    with pytest.raises(
        UnsupportedGeoTIFFFeatureError, match='GDALWarpOptions'
    ):
        _public_read_vrt(path)


def test_warp_options_dataset_level_rejected_via_internal_read_vrt(tmp_path):
    """The internal ``_vrt.read_vrt`` rejects the same input. Routing
    through the validator preserves the parse-time rejection because
    ``parse_vrt`` runs before ``validate_parsed_vrt``."""
    path = _write_vrt(tmp_path, _WARP_DATASET_XML, 'warp_int_2371.vrt')
    with pytest.raises(
        UnsupportedGeoTIFFFeatureError, match='GDALWarpOptions'
    ):
        _internal_read_vrt(path)


_WARP_BAND_XML = """<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <GDALWarpOptions>
      <ResampleAlg>NearestNeighbour</ResampleAlg>
    </GDALWarpOptions>
  </VRTRasterBand>
</VRTDataset>"""


def test_warp_options_band_level_rejected(tmp_path):
    """A band-level ``<GDALWarpOptions>`` block (rare but possible
    depending on the VRT emitter) is rejected via the band-children
    sweep in ``_UNSUPPORTED_BAND_TAGS``."""
    path = _write_vrt(tmp_path, _WARP_BAND_XML, 'warp_band_2371.vrt')
    with pytest.raises(
        UnsupportedGeoTIFFFeatureError, match='GDALWarpOptions'
    ):
        parse_vrt(_WARP_BAND_XML, os.path.dirname(path))


# ---------------------------------------------------------------------------
# Rule: per-source mask / alpha semantics
# ---------------------------------------------------------------------------


def _use_mask_band_xml(src_path: str, flag: str = 'true') -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <UseMaskBand>{flag}</UseMaskBand>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>"""


def test_use_mask_band_true_rejected_at_validator(tmp_path):
    """A ComplexSource declaring ``<UseMaskBand>true</UseMaskBand>``
    must raise ``VRTUnsupportedError`` at validate time, with the
    offending source path in the message."""
    src_path = _write_src(tmp_path)
    path, parsed = _parse(
        tmp_path, _use_mask_band_xml(src_path), 'use_mask_2371.vrt'
    )
    with pytest.raises(VRTUnsupportedError) as excinfo:
        validate_parsed_vrt(parsed, source=path, mode='read')
    msg = str(excinfo.value)
    assert 'UseMaskBand' in msg
    assert src_path in msg


@pytest.mark.parametrize('flag', ['true', 'True', 'TRUE', '1', 'yes'])
def test_use_mask_band_truthy_spellings_rejected(tmp_path, flag):
    """``<UseMaskBand>`` accepts several truthy spellings (GDAL writes
    lowercase ``true`` but the validator should not depend on case or
    on the exact token)."""
    src_path = _write_src(tmp_path, name=f'src_flag_{flag}_2371.tif')
    path, parsed = _parse(
        tmp_path, _use_mask_band_xml(src_path, flag=flag),
        f'use_mask_{flag}_2371.vrt',
    )
    with pytest.raises(VRTUnsupportedError, match='UseMaskBand'):
        validate_parsed_vrt(parsed, source=path, mode='read')


def test_use_mask_band_false_is_accepted(tmp_path):
    """An explicit ``<UseMaskBand>false</UseMaskBand>`` is a no-op and
    must not trip the rejection. GDAL never writes ``false`` itself,
    but hand-written VRTs occasionally do."""
    src_path = _write_src(tmp_path, name='src_false_2371.tif')
    path, parsed = _parse(
        tmp_path, _use_mask_band_xml(src_path, flag='false'),
        'use_mask_false_2371.vrt',
    )
    # Must not raise.
    validate_parsed_vrt(parsed, source=path, mode='read')


def test_use_mask_band_rejected_via_public_read_vrt(tmp_path):
    """End-to-end: the public backend entry point surfaces the same
    rejection."""
    src_path = _write_src(tmp_path, name='src_pub_mask_2371.tif')
    path = _write_vrt(
        tmp_path, _use_mask_band_xml(src_path), 'use_mask_pub_2371.vrt'
    )
    with pytest.raises(VRTUnsupportedError, match='UseMaskBand'):
        _public_read_vrt(path)


def test_use_mask_band_rejected_via_internal_read_vrt(tmp_path):
    """End-to-end: the internal entry point also surfaces the rejection
    via the validator now that #2371 wires it in."""
    src_path = _write_src(tmp_path, name='src_int_mask_2371.tif')
    path = _write_vrt(
        tmp_path, _use_mask_band_xml(src_path), 'use_mask_int_2371.vrt'
    )
    with pytest.raises(VRTUnsupportedError, match='UseMaskBand'):
        _internal_read_vrt(path)


def _per_source_mask_band_xml(src_path: str) -> str:
    """A ComplexSource with a per-source ``<MaskBand>`` child (distinct
    from a dataset-level ``<MaskBand>`` sibling). GDAL emits this when
    a source TIFF carries an internal mask band that the VRT wires
    through."""
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <MaskBand>
        <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
        <SourceBand>1</SourceBand>
      </MaskBand>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>"""


def test_per_source_mask_band_rejected_at_validator(tmp_path):
    """A per-source ``<MaskBand>`` child raises ``VRTUnsupportedError``
    at validate time naming the source path."""
    src_path = _write_src(tmp_path, name='src_pmask_2371.tif')
    path, parsed = _parse(
        tmp_path, _per_source_mask_band_xml(src_path),
        'per_src_mask_2371.vrt',
    )
    with pytest.raises(VRTUnsupportedError) as excinfo:
        validate_parsed_vrt(parsed, source=path, mode='read')
    msg = str(excinfo.value)
    assert 'MaskBand' in msg
    assert src_path in msg


# ---------------------------------------------------------------------------
# Rule: resample alg gate now fires at the internal entry point
# ---------------------------------------------------------------------------


def test_resample_alg_now_rejected_at_internal_read_vrt(tmp_path):
    """The internal ``_vrt.read_vrt`` was previously not routed through
    the validator and surfaced unsupported-resample as a
    ``NotImplementedError`` at the placement site. After #2371 the
    validator preempts that gate so the failure is now a typed
    ``VRTUnsupportedError`` at graph build / eager setup."""
    src_path = _write_src(tmp_path, name='src_resample_2371.tif')
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
    path = _write_vrt(tmp_path, xml, 'resample_int_2371.vrt')
    with pytest.raises(VRTUnsupportedError, match='Bilinear'):
        _internal_read_vrt(path)


# ---------------------------------------------------------------------------
# Subclassing contract for the new path
# ---------------------------------------------------------------------------


def test_nested_vrt_error_is_value_error(tmp_path):
    """``VRTUnsupportedError`` already subclasses ``ValueError`` via
    ``GeoTIFFAmbiguousMetadataError``. The nested-VRT path uses the
    same class, so ``except ValueError`` keeps catching the new
    rejection too."""
    src_path = _write_src(tmp_path, name='src_subclass_2371.tif')
    inner_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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
    inner_path = _write_vrt(tmp_path, inner_xml, 'inner_sub_2371.vrt')
    outer_path, parsed = _parse(
        tmp_path, _nested_vrt_xml(inner_path), 'outer_sub_2371.vrt'
    )
    with pytest.raises(ValueError):  # via VRTUnsupportedError -> ValueError
        validate_parsed_vrt(parsed, source=outer_path, mode='read')
    with pytest.raises(GeoTIFFAmbiguousMetadataError):
        validate_parsed_vrt(parsed, source=outer_path, mode='read')
