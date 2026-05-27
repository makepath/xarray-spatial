"""VRT validator + reader-error contract.

Covers the validator-side VRT surface:

* centralised ``validate_parsed_vrt`` rule rejections.
* nested/warped VRT, ``UseMaskBand`` / per-source ``MaskBand``,
  internal-entry-point resample-alg gate, and the
  ``validate_vrt_capability`` alias.
* end-to-end negative coverage (warped, nested, mixed CRS, mixed dtype,
  mixed band count, mask, resample alg) through both public entry
  points.
* narrowed-``except`` contract in ``read_vrt`` for source-read failures
  (warn-and-continue vs propagate) under default and
  ``XRSPATIAL_GEOTIFF_STRICT=1`` modes.
* path-traversal rejection in ``parse_vrt`` / ``_read_vrt_internal``
  and the ``XRSPATIAL_VRT_ALLOWED_ROOTS`` opt-in.

Conventions:

* Parametrise IDs are descriptive (``id="reject[bad-resample-alg]"``);
  issue numbers do not appear in test or parametrise names.
* Source-TIFF and VRT filenames carry a uuid suffix so parallel test
  workers across worktrees never collide on the same on-disk name.
"""
from __future__ import annotations

import os
import struct
import uuid
import warnings
import zlib

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import GeoTIFFFallbackWarning, open_geotiff, to_geotiff
from xrspatial.geotiff._backends.vrt import read_vrt as _package_read_vrt
from xrspatial.geotiff._errors import (GeoTIFFAmbiguousMetadataError, MixedBandMetadataError,
                                       RotatedTransformError, UnparseableCRSError,
                                       UnsupportedGeoTIFFFeatureError, VRTUnsupportedError)
from xrspatial.geotiff._vrt import parse_vrt
from xrspatial.geotiff._vrt import read_vrt as _internal_read_vrt
from xrspatial.geotiff._vrt_validation import validate_parsed_vrt, validate_vrt_capability
from xrspatial.geotiff._writer import write

# ``xrspatial.geotiff.read_vrt`` (re-exported from the package init) is the
# same callable as ``_backends.vrt.read_vrt``; the parametrise IDs below
# label the backend-module path as the "package" entry point because that
# is what the public alias resolves to.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniq(prefix: str) -> str:
    """Short unique tag so on-disk artefact names never collide across
    parallel workers or worktrees."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _write_src_uint16(tmp_path, name: str | None = None,
                      shape=(4, 4)) -> str:
    """Write a small uint16 source TIFF via the low-level ``write`` helper.

    Matches the fixture used in 2321 / 2371: ``write(arr, path, ...)``
    bypasses the higher-level ``to_geotiff`` so the validator tests do
    not depend on the writer side adding CRS / nodata metadata that
    would influence parser behaviour.
    """
    name = name or _uniq("src_uint16")
    arr = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    path = str(tmp_path / f"{name}.tif")
    write(arr, path, compression='none', tiled=False)
    return path


def _write_src_float32_geotiff(tmp_path, name: str | None = None,
                               *, dtype=np.float32, shape=(4, 4),
                               crs='EPSG:4326') -> str:
    """Write a small GeoTIFF via the high-level ``to_geotiff`` entry point.

    The 2370 fixtures use this shape so the source carries CRS and
    nodata attrs the cross-CRS / mixed-dtype tests need.
    """
    name = name or _uniq("src_geo")
    arr = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    y = np.linspace(1.0, 0.0, shape[0])
    x = np.linspace(0.0, 1.0, shape[1])
    fill = -9999 if np.issubdtype(arr.dtype, np.integer) else -9999.0
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'nodata': fill, 'crs': crs},
    )
    path = str(tmp_path / f"{name}.tif")
    to_geotiff(da, path)
    return path


def _write_vrt(tmp_path, xml: str, name: str | None = None) -> str:
    """Write VRT XML to disk and return its absolute path."""
    name = name or _uniq("mosaic")
    path = str(tmp_path / f"{name}.vrt")
    with open(path, 'w') as f:
        f.write(xml)
    return path


def _parse(tmp_path, xml: str, name: str | None = None):
    """Write + parse a VRT XML, return ``(path, parsed)``."""
    path = _write_vrt(tmp_path, xml, name)
    parsed = parse_vrt(xml, os.path.dirname(os.path.abspath(path)))
    return path, parsed


def _simple_source_xml(src_path: str, *, band: int = 1) -> str:
    """Render one ``<SimpleSource>`` over a matched 4x4 SrcRect/DstRect."""
    return f"""    <SimpleSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>{band}</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
    </SimpleSource>"""


def _vrt_xml(*, width: int = 4, height: int = 4,
             dtype_name: str = 'Float32',
             body: str = '',
             extra_dataset_inner: str = '',
             srs: str = 'EPSG:4326') -> str:
    """Render a minimal VRT XML wrapper."""
    return f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <SRS>{srs}</SRS>
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  {extra_dataset_inner}
  <VRTRasterBand dataType="{dtype_name}" band="1">
{body}
  </VRTRasterBand>
</VRTDataset>"""


# ---------------------------------------------------------------------------
# Error subclass contracts
# ---------------------------------------------------------------------------


def test_vrt_unsupported_error_subclass_contract():
    """``VRTUnsupportedError`` must subclass
    ``GeoTIFFAmbiguousMetadataError`` (and ``ValueError`` via the base)
    so existing ``except ValueError`` callers keep catching VRT
    failures."""
    assert issubclass(VRTUnsupportedError, GeoTIFFAmbiguousMetadataError)
    assert issubclass(VRTUnsupportedError, ValueError)


def test_validate_vrt_capability_is_validate_parsed_vrt():
    """``validate_vrt_capability`` is the public alias matching the
    issue text. It resolves to the same callable as
    ``validate_parsed_vrt`` so both names share one implementation."""
    assert validate_vrt_capability is validate_parsed_vrt


# ---------------------------------------------------------------------------
# Rule-driven validator rejections.
#
# Each row describes a malformed VRT that ``validate_parsed_vrt`` must
# reject at validate time (before any source decode). The error class
# and a substring of the message are pinned per-rule.
#
# A handful of rules raise typed subclasses (``RotatedTransformError``,
# ``UnparseableCRSError``, ``MixedBandMetadataError``) that callers
# already match against; the parametrise carries the expected class so
# the typed contract stays explicit.
# ---------------------------------------------------------------------------


_BAND_TEMPLATE_NO_GEOTRANSFORM = """<VRTDataset rasterXSize="4" rasterYSize="4">
</VRTDataset>"""

_ZERO_BANDS_VRT = """<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
</VRTDataset>"""


def _rotated_vrt(src_path: str) -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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


def _negative_src_size_vrt(src_path: str) -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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


def _negative_src_offset_vrt(src_path: str) -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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


def _negative_dst_size_vrt(src_path: str) -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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


def _dst_outside_extent_vrt(src_path: str) -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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


def _zero_pixel_size_vrt(src_path: str) -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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


def _bilinear_resample_vrt(src_path: str) -> str:
    """ComplexSource with non-nearest resampler and size-changing rects."""
    return f"""<VRTDataset rasterXSize="2" rasterYSize="2">
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


def _bad_crs_vrt(src_path: str) -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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


class TestValidatorRules:
    """Each rule of ``validate_parsed_vrt`` gets one negative test that
    builds a malformed VRT, parses it, and asserts the validator
    rejects it with the expected typed error and message keyword."""

    def test_zero_bands_rejected(self, tmp_path):
        path, parsed = _parse(tmp_path, _ZERO_BANDS_VRT)
        with pytest.raises(VRTUnsupportedError, match='band'):
            validate_parsed_vrt(parsed, source=path, mode='read')

    def test_complex_dtype_band_rejected(self, tmp_path):
        """A complex numpy dtype on a parsed band (e.g. via a future
        ``_DTYPE_MAP`` regression) must be refused before any decode
        runs."""
        src_path = _write_src_uint16(tmp_path)
        xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
{_simple_source_xml(src_path)}
  </VRTRasterBand>
</VRTDataset>"""
        path, parsed = _parse(tmp_path, xml)
        parsed.bands[0].dtype = np.dtype('complex128')
        with pytest.raises(VRTUnsupportedError, match='dtype'):
            validate_parsed_vrt(parsed, source=path, mode='read')

    def test_rotated_transform_rejected_without_opt_in(self, tmp_path):
        """Rotated transforms raise the existing typed
        ``RotatedTransformError`` for backward compatibility; the
        validator's value here is adding the source path to the message
        and lifting the rejection ahead of any decode. ``allow_rotated``
        is the documented opt-in."""
        src_path = _write_src_uint16(tmp_path)
        path, parsed = _parse(tmp_path, _rotated_vrt(src_path))
        with pytest.raises(RotatedTransformError, match='rotat'):
            validate_parsed_vrt(parsed, source=path, mode='read',
                                allow_rotated=False)
        # Opt-in is silent.
        validate_parsed_vrt(parsed, source=path, mode='read',
                            allow_rotated=True)

    @pytest.mark.parametrize(
        "build_xml, match",
        [
            pytest.param(_negative_src_size_vrt, 'SrcRect',
                         id="reject[negative-src-size]"),
            pytest.param(_negative_src_offset_vrt, 'SrcRect',
                         id="reject[negative-src-offset]"),
            pytest.param(_negative_dst_size_vrt, 'DstRect',
                         id="reject[negative-dst-size]"),
            pytest.param(_dst_outside_extent_vrt, 'DstRect',
                         id="reject[dst-outside-extent]"),
            pytest.param(_zero_pixel_size_vrt, 'pixel size',
                         id="reject[zero-pixel-size]"),
        ],
    )
    def test_geometry_rules_rejected(self, tmp_path, build_xml, match):
        src_path = _write_src_uint16(tmp_path)
        path, parsed = _parse(tmp_path, build_xml(src_path))
        with pytest.raises(VRTUnsupportedError, match=match):
            validate_parsed_vrt(parsed, source=path, mode='read')

    def test_unsupported_resample_alg_rejected(self, tmp_path):
        """A non-nearest resampler paired with size-changing rects must
        be rejected by the validator before any pixels are read."""
        src_path = _write_src_uint16(tmp_path)
        path, parsed = _parse(tmp_path, _bilinear_resample_vrt(src_path))
        with pytest.raises(VRTUnsupportedError, match='[Rr]esampl'):
            validate_parsed_vrt(parsed, source=path, mode='read')

    def test_unparseable_crs_rejected_without_opt_in(self, tmp_path):
        """CRS strings pyproj cannot resolve raise the existing typed
        ``UnparseableCRSError``. ``allow_unparseable_crs=True`` is the
        documented opt-in."""
        src_path = _write_src_uint16(tmp_path)
        path, parsed = _parse(tmp_path, _bad_crs_vrt(src_path))
        with pytest.raises(UnparseableCRSError):
            validate_parsed_vrt(parsed, source=path, mode='read',
                                allow_unparseable_crs=False)
        validate_parsed_vrt(parsed, source=path, mode='read',
                            allow_unparseable_crs=True)

    def test_well_formed_vrt_validates_silently(self, tmp_path):
        src_path = _write_src_uint16(tmp_path)
        xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
{_simple_source_xml(src_path)}
  </VRTRasterBand>
</VRTDataset>"""
        path, parsed = _parse(tmp_path, xml)
        validate_parsed_vrt(parsed, source=path, mode='read')
        result = _package_read_vrt(path)
        assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# Mixed per-band nodata: the rejection lives in ``validate_read_metadata``
# and surfaces through the public entry points, not through
# ``validate_parsed_vrt`` directly. Pin the cross-entry-point behaviour.
# ---------------------------------------------------------------------------


def _mixed_nodata_vrt(src_path: str) -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <NoDataValue>0</NoDataValue>
{_simple_source_xml(src_path)}
  </VRTRasterBand>
  <VRTRasterBand dataType="UInt16" band="2">
    <NoDataValue>9999</NoDataValue>
{_simple_source_xml(src_path)}
  </VRTRasterBand>
</VRTDataset>"""


def test_mixed_band_nodata_rejected_without_opt_in(tmp_path):
    """Two bands with disagreeing per-band ``<NoDataValue>`` raise
    ``MixedBandMetadataError`` unless ``band_nodata='first'`` is
    passed. The validator delegates disagreement detection to the
    ``validate_read_metadata`` hook so the ``None``-counts-as-undeclared
    semantics live in one place."""
    src_path = _write_src_uint16(tmp_path)
    path = _write_vrt(tmp_path, _mixed_nodata_vrt(src_path))
    with pytest.raises(MixedBandMetadataError):
        _package_read_vrt(path)
    with pytest.raises(MixedBandMetadataError):
        open_geotiff(path)
    r = _package_read_vrt(path, band_nodata='first')
    assert r.shape == (4, 4, 2)


# ---------------------------------------------------------------------------
# Nested VRT: a ``.vrt`` referenced as a ``<SourceFilename>``.
# ---------------------------------------------------------------------------


def _nested_outer_xml(inner_vrt_path: str) -> str:
    return f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
{_simple_source_xml(inner_vrt_path)}
  </VRTRasterBand>
</VRTDataset>"""


def _write_inner_vrt(tmp_path, src_path: str, *, name: str = None) -> str:
    name = name or _uniq("inner")
    inner_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
{_simple_source_xml(src_path)}
  </VRTRasterBand>
</VRTDataset>"""
    return _write_vrt(tmp_path, inner_xml, name)


def test_nested_vrt_message_names_outer_and_inner(tmp_path):
    """The validator-level rejection must name both the outer VRT path
    (the failing file) and the inner VRT (the bad source). Inner path
    comparison uses the basename so it survives ``realpath``
    canonicalisation on Windows-style short paths."""
    src_path = _write_src_uint16(tmp_path)
    inner_path = _write_inner_vrt(tmp_path, src_path)
    outer_path, parsed = _parse(tmp_path, _nested_outer_xml(inner_path))

    with pytest.raises(VRTUnsupportedError) as excinfo:
        validate_parsed_vrt(parsed, source=outer_path, mode='read')
    msg = str(excinfo.value)
    assert outer_path in msg
    assert os.path.basename(inner_path) in msg
    assert 'Nested' in msg or 'nested' in msg


def test_nested_vrt_uppercase_extension_rejected(tmp_path):
    """``.VRT`` (uppercase) trips the same rejection. Extension matching
    is case-insensitive so Windows-style emitters are caught."""
    src_path = _write_src_uint16(tmp_path)
    inner_path = _write_inner_vrt(tmp_path, src_path,
                                  name=_uniq("INNER").upper())
    outer_path, parsed = _parse(tmp_path, _nested_outer_xml(inner_path))
    with pytest.raises(VRTUnsupportedError, match='[Nn]ested'):
        validate_parsed_vrt(parsed, source=outer_path, mode='read')


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(_package_read_vrt, id="entry[package-read_vrt]"),
        pytest.param(_internal_read_vrt, id="entry[internal-read_vrt]"),
        pytest.param(open_geotiff, id="entry[open_geotiff]"),
    ],
)
def test_nested_vrt_rejected_via_entry_points(tmp_path, reader):
    """All three public-ish entry points must surface the nested-VRT
    rejection identically: the validator is wired at both the package
    backend wrapper and the internal ``_vrt.read_vrt``."""
    src_path = _write_src_uint16(tmp_path)
    inner_path = _write_inner_vrt(tmp_path, src_path)
    outer_path = _write_vrt(tmp_path, _nested_outer_xml(inner_path))
    with pytest.raises(VRTUnsupportedError, match='[Nn]ested'):
        reader(outer_path)


def test_nested_vrt_error_remains_value_error_subclass(tmp_path):
    """``VRTUnsupportedError`` keeps subclassing ``ValueError`` via
    ``GeoTIFFAmbiguousMetadataError`` so ``except ValueError`` callers
    still catch the new rejection path."""
    src_path = _write_src_uint16(tmp_path)
    inner_path = _write_inner_vrt(tmp_path, src_path)
    outer_path, parsed = _parse(tmp_path, _nested_outer_xml(inner_path))
    with pytest.raises(ValueError):
        validate_parsed_vrt(parsed, source=outer_path, mode='read')
    with pytest.raises(GeoTIFFAmbiguousMetadataError):
        validate_parsed_vrt(parsed, source=outer_path, mode='read')


# ---------------------------------------------------------------------------
# Warped VRT: ``<GDALWarpOptions>`` block (dataset and band level) and
# ``subClass="VRTWarpedRasterBand"`` are all rejected.
# ---------------------------------------------------------------------------


_WARP_DATASET_XML = """<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <GDALWarpOptions>
    <WarpMemoryLimit>64.0</WarpMemoryLimit>
    <ResampleAlg>NearestNeighbour</ResampleAlg>
  </GDALWarpOptions>
  <VRTRasterBand dataType="UInt16" band="1"/>
</VRTDataset>"""

_WARP_BAND_XML = """<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="UInt16" band="1">
    <GDALWarpOptions>
      <ResampleAlg>NearestNeighbour</ResampleAlg>
    </GDALWarpOptions>
  </VRTRasterBand>
</VRTDataset>"""


@pytest.mark.parametrize(
    "xml, scope",
    [
        pytest.param(_WARP_DATASET_XML, 'dataset',
                     id="warp[dataset-level]"),
        pytest.param(_WARP_BAND_XML, 'band',
                     id="warp[band-level]"),
    ],
)
def test_warp_options_rejected_at_parse(tmp_path, xml, scope):
    """Both dataset-level and band-level ``<GDALWarpOptions>`` are
    rejected by ``parse_vrt`` itself (via ``_UNSUPPORTED_*_TAGS``), so
    callers that route through the validator still see a typed
    failure -- parse runs before the validator is reached."""
    path = _write_vrt(tmp_path, xml)
    with pytest.raises(UnsupportedGeoTIFFFeatureError, match='GDALWarpOptions'):
        parse_vrt(xml, os.path.dirname(path))


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(_package_read_vrt, id="entry[package-read_vrt]"),
        pytest.param(_internal_read_vrt, id="entry[internal-read_vrt]"),
    ],
)
def test_warp_options_dataset_rejected_via_entry_points(tmp_path, reader):
    path = _write_vrt(tmp_path, _WARP_DATASET_XML)
    with pytest.raises(UnsupportedGeoTIFFFeatureError, match='GDALWarpOptions'):
        reader(path)


def test_warped_subclass_band_rejected_via_open_geotiff(tmp_path):
    """``subClass="VRTWarpedRasterBand"`` is the band-level warped
    marker; ``open_geotiff`` must reject it too so callers cannot slip
    a warped VRT through the public accessor. The message has to name
    the failure mode (``warp`` or ``vrtwarped``) so a regression that
    raises a generic ``ValueError`` without identifying the cause does
    not slip past the gate."""
    src_path = _write_src_float32_geotiff(tmp_path)
    warped_xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4" subClass="VRTWarpedDataset">
  <SRS>EPSG:4326</SRS>
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1" subClass="VRTWarpedRasterBand">
    <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, warped_xml)
    with pytest.raises((ValueError, NotImplementedError, RuntimeError,
                        UnsupportedGeoTIFFFeatureError)) as excinfo:
        open_geotiff(vrt_path)
    msg = str(excinfo.value).lower()
    assert any(k in msg for k in ('warp', 'vrtwarped')), (
        f"warped-VRT rejection must name 'warp' or 'vrtwarped'; got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Per-source mask / alpha semantics.
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


def test_use_mask_band_message_names_source(tmp_path):
    """``<UseMaskBand>true</UseMaskBand>`` raises at validate time with
    both the feature name and the offending source path in the
    message."""
    src_path = _write_src_uint16(tmp_path)
    path, parsed = _parse(tmp_path, _use_mask_band_xml(src_path))
    with pytest.raises(VRTUnsupportedError) as excinfo:
        validate_parsed_vrt(parsed, source=path, mode='read')
    msg = str(excinfo.value)
    assert 'UseMaskBand' in msg
    assert src_path in msg


@pytest.mark.parametrize(
    "flag",
    ['true', 'True', 'TRUE', '1'],
    ids=lambda f: f"truthy[{f}]",
)
def test_use_mask_band_truthy_spellings_rejected(tmp_path, flag):
    """The canonical truthy set is GDAL's ``true`` (any case) plus the
    digit ``1``. Hand-edited VRTs occasionally normalise booleans to
    ``1`` via XML emitters, so both spellings trip the rejection."""
    src_path = _write_src_uint16(tmp_path)
    path, parsed = _parse(
        tmp_path, _use_mask_band_xml(src_path, flag=flag),
    )
    with pytest.raises(VRTUnsupportedError, match='UseMaskBand'):
        validate_parsed_vrt(parsed, source=path, mode='read')


def test_use_mask_band_false_is_accepted(tmp_path):
    """An explicit ``<UseMaskBand>false</UseMaskBand>`` is a no-op.
    GDAL never writes it, but hand-written VRTs occasionally do."""
    src_path = _write_src_uint16(tmp_path)
    path, parsed = _parse(
        tmp_path, _use_mask_band_xml(src_path, flag='false'),
    )
    validate_parsed_vrt(parsed, source=path, mode='read')


@pytest.mark.parametrize(
    "flag",
    ['yes', 'on', 'Y'],
    ids=lambda f: f"non-canonical[{f}]",
)
def test_use_mask_band_non_canonical_truthy_accepted(tmp_path, flag):
    """Tokens outside the canonical GDAL set are treated as not-mask.
    The parser deliberately narrows the truthy set so a hand-edited VRT
    using a Python-truthy spelling does not silently flip the read into
    the rejection path. If GDAL ever starts emitting one of these, the
    set should be widened then."""
    src_path = _write_src_uint16(tmp_path)
    path, parsed = _parse(
        tmp_path, _use_mask_band_xml(src_path, flag=flag),
    )
    validate_parsed_vrt(parsed, source=path, mode='read')


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(_package_read_vrt, id="entry[package-read_vrt]"),
        pytest.param(_internal_read_vrt, id="entry[internal-read_vrt]"),
    ],
)
def test_use_mask_band_rejected_via_entry_points(tmp_path, reader):
    src_path = _write_src_uint16(tmp_path)
    path = _write_vrt(tmp_path, _use_mask_band_xml(src_path))
    with pytest.raises(VRTUnsupportedError, match='UseMaskBand'):
        reader(path)


def test_per_source_mask_band_message_names_source(tmp_path):
    """A per-source ``<MaskBand>`` child (distinct from a dataset-level
    sibling) raises at validate time naming the source path."""
    src_path = _write_src_uint16(tmp_path)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
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
    path, parsed = _parse(tmp_path, xml)
    with pytest.raises(VRTUnsupportedError) as excinfo:
        validate_parsed_vrt(parsed, source=path, mode='read')
    msg = str(excinfo.value)
    assert 'MaskBand' in msg
    assert src_path in msg


# ---------------------------------------------------------------------------
# Entry-point parity: ``read_vrt`` (package) and ``open_geotiff`` produce
# the same typed exception with the same message for the same bad input.
# ---------------------------------------------------------------------------


def _expect_same_error(path, exc_type):
    """Run ``_package_read_vrt`` and ``open_geotiff``, assert both raise
    the same type with the same message."""
    with pytest.raises(exc_type) as a:
        _package_read_vrt(path)
    with pytest.raises(exc_type) as b:
        open_geotiff(path)
    assert type(a.value) is type(b.value)
    assert str(a.value) == str(b.value)


def test_zero_bands_parity_across_entry_points(tmp_path):
    path = _write_vrt(tmp_path, _ZERO_BANDS_VRT)
    _expect_same_error(path, VRTUnsupportedError)


def test_resample_parity_across_entry_points(tmp_path):
    src_path = _write_src_uint16(tmp_path)
    path = _write_vrt(tmp_path, _bilinear_resample_vrt(src_path))
    _expect_same_error(path, VRTUnsupportedError)


def test_rotated_parity_across_entry_points(tmp_path):
    """Rotated transforms keep their pre-existing
    ``RotatedTransformError`` subclass; both entry points raise it
    with the same message."""
    src_path = _write_src_uint16(tmp_path)
    path = _write_vrt(tmp_path, _rotated_vrt(src_path))
    _expect_same_error(path, RotatedTransformError)


def test_unsupported_resample_chunked_raises_at_build(tmp_path):
    """The chunked dispatcher runs the validator before building the
    dask graph so the failure surfaces at construction time, not deep
    in a chunk function during ``compute()``."""
    src_path = _write_src_uint16(tmp_path)
    path = _write_vrt(tmp_path, _bilinear_resample_vrt(src_path))
    with pytest.raises(VRTUnsupportedError):
        _package_read_vrt(path, chunks=2)


def test_resample_alg_rejected_at_internal_read_vrt(tmp_path):
    """The internal ``_vrt.read_vrt`` is now routed through the
    validator. A direct call produces the typed
    ``VRTUnsupportedError`` at graph build / eager setup rather than the
    old ``NotImplementedError`` at the placement site."""
    src_path = _write_src_uint16(tmp_path)
    path = _write_vrt(tmp_path, _bilinear_resample_vrt(src_path))
    with pytest.raises(VRTUnsupportedError, match='Bilinear'):
        _internal_read_vrt(path)


# ---------------------------------------------------------------------------
# End-to-end negative coverage: the centralised validator fires these
# rejections directly, so the assertions below are not wrapped in xfail.
# ---------------------------------------------------------------------------


def test_mixed_source_dtype_complex_rejected(tmp_path):
    """``dataType="CFloat32"`` (complex) raises ``ValueError``;
    the message must name the rejected dtype AND mention 'complex' so
    the rejection is actionable."""
    src_path = _write_src_float32_geotiff(tmp_path)
    body = _simple_source_xml(src_path)
    xml = _vrt_xml(body=body, dtype_name='CFloat32')
    vrt_path = _write_vrt(tmp_path, xml)
    with pytest.raises(ValueError, match=r'CFloat32') as excinfo:
        _package_read_vrt(vrt_path)
    assert 'complex' in str(excinfo.value).lower()


def test_mixed_source_band_count_rejected(tmp_path):
    """A single-band source referenced via ``SourceBand=2`` cannot read
    band 2 (it does not exist). Must fail with a message that names the
    band rather than silently decoding the wrong one."""
    single_band_src = _write_src_float32_geotiff(tmp_path)
    body = _simple_source_xml(single_band_src, band=2)
    xml = _vrt_xml(body=body)
    vrt_path = _write_vrt(tmp_path, xml)
    with pytest.raises((ValueError, IndexError, RuntimeError,
                        NotImplementedError)) as excinfo:
        _package_read_vrt(vrt_path)
    assert 'band' in str(excinfo.value).lower()


@pytest.mark.parametrize(
    "alg",
    ['Bilinear', 'Cubic', 'Lanczos', 'Average', 'Mode'],
    ids=lambda a: f"resample[{a.lower()}]",
)
@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(_package_read_vrt, id="entry[package-read_vrt]"),
        pytest.param(open_geotiff, id="entry[open_geotiff]"),
    ],
)
def test_unsupported_resample_alg_rejected_end_to_end(tmp_path, alg, reader):
    """A ``<ResampleAlg>`` outside the implemented nearest subset,
    paired with size-changing rects, is rejected by both entry points.
    Either ``NotImplementedError`` (legacy resample-site check) or a
    ``ValueError`` subclass (``VRTUnsupportedError`` from the
    centralised validator) is acceptable so long as the message names
    the offending algorithm."""
    src_path = _write_src_float32_geotiff(tmp_path)
    body = f"""    <ComplexSource>
      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      <DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
      <ResampleAlg>{alg}</ResampleAlg>
    </ComplexSource>"""
    xml = _vrt_xml(width=2, height=2, body=body)
    vrt_path = _write_vrt(tmp_path, xml)

    with pytest.raises((NotImplementedError, ValueError)) as excinfo:
        reader(vrt_path)
    assert alg in str(excinfo.value)


def test_dataset_level_mask_band_rejected(tmp_path):
    """A dataset-level ``<MaskBand>`` declares a per-pixel mask the
    GeoTIFF attrs contract cannot represent. Reading the mosaic and
    silently dropping the mask would produce a result the caller cannot
    distinguish from one with no mask. Must be rejected with 'mask' in
    the message."""
    src_path = _write_src_float32_geotiff(tmp_path)
    mask_src = _write_src_float32_geotiff(tmp_path, dtype=np.uint8)
    body = _simple_source_xml(src_path)
    mask_block = f"""  <MaskBand>
    <VRTRasterBand dataType="Byte">
      <SimpleSource>
        <SourceFilename relativeToVRT="0">{mask_src}</SourceFilename>
        <SourceBand>1</SourceBand>
        <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>
        <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>
      </SimpleSource>
    </VRTRasterBand>
  </MaskBand>"""
    xml = _vrt_xml(body=body, extra_dataset_inner=mask_block)
    vrt_path = _write_vrt(tmp_path, xml)
    with pytest.raises((ValueError, NotImplementedError)) as excinfo:
        _package_read_vrt(vrt_path)
    assert 'mask' in str(excinfo.value).lower()


# The next two cases were ``xfail`` wrappers in the original 2370 file:
# the rejection contract is documented but the centralised validator
# does not yet name the failure mode in a way that ``except (ValueError,
# NotImplementedError)`` catches with the expected keyword. Keep them
# under ``xfail(strict=False)`` so a future validator change starts
# passing them without an edit here.


def test_mixed_source_crs_rejected(tmp_path):
    """Two band sources with disagreeing CRS (EPSG:4326 vs EPSG:3857)
    cannot mosaic correctly without reprojection. The
    validator opens each source TIFF and raises
    ``VRTUnsupportedError`` when the source CRS disagrees with the
    VRT-declared ``<SRS>``. The error message names the offending
    source path and CRS."""
    src_4326 = _write_src_float32_geotiff(tmp_path, crs='EPSG:4326')
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    y = np.linspace(1.0, 0.0, 4)
    x = np.linspace(0.0, 1.0, 4)
    da_3857 = xr.DataArray(
        arr, dims=['y', 'x'], coords={'y': y, 'x': x},
        attrs={'nodata': -9999.0, 'crs': 'EPSG:3857'},
    )
    src_3857 = str(tmp_path / f"{_uniq('src_3857')}.tif")
    to_geotiff(da_3857, src_3857)

    body = _simple_source_xml(src_4326) + "\n" + _simple_source_xml(src_3857)
    xml = _vrt_xml(body=body, srs='EPSG:4326')
    vrt_path = _write_vrt(tmp_path, xml)

    with pytest.raises(VRTUnsupportedError) as excinfo:
        _package_read_vrt(vrt_path)
    msg = str(excinfo.value).lower()
    assert any(k in msg for k in ('crs', 'srs', 'projection', 'epsg'))


def test_mixed_source_crs_message_names_offending_source(tmp_path):
    """The validator must embed the source path and CRS in the error
    message so a caller can locate the bad source without re-parsing
    the VRT. Locked in addition to ``test_mixed_source_crs_rejected``
    so a future refactor that strips the source name from the message
    still trips a guard."""
    src_4326 = _write_src_float32_geotiff(tmp_path, crs='EPSG:4326')
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    y = np.linspace(1.0, 0.0, 4)
    x = np.linspace(0.0, 1.0, 4)
    da_3857 = xr.DataArray(
        arr, dims=['y', 'x'], coords={'y': y, 'x': x},
        attrs={'nodata': -9999.0, 'crs': 'EPSG:3857'},
    )
    src_3857 = str(tmp_path / f"{_uniq('src_3857_named')}.tif")
    to_geotiff(da_3857, src_3857)

    body = _simple_source_xml(src_4326) + "\n" + _simple_source_xml(src_3857)
    xml = _vrt_xml(body=body, srs='EPSG:4326')
    vrt_path = _write_vrt(tmp_path, xml)

    with pytest.raises(VRTUnsupportedError) as excinfo:
        _package_read_vrt(vrt_path)
    msg = str(excinfo.value)
    # The offending source path must appear verbatim in the message,
    # along with both CRS labels for contrast.
    assert src_3857 in msg
    assert '3857' in msg
    assert '4326' in msg


def test_mixed_source_crs_chunked_path_also_rejects(tmp_path):
    """The chunked VRT read path runs the same validator at graph build
    time, so a mixed-CRS VRT must surface ``VRTUnsupportedError``
    before any per-chunk task executes. Without this pin the eager
    path could carry the fix while the dask path silently rode the bad
    mosaic into a ``compute()`` failure."""
    src_4326 = _write_src_float32_geotiff(tmp_path, crs='EPSG:4326')
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    y = np.linspace(1.0, 0.0, 4)
    x = np.linspace(0.0, 1.0, 4)
    da_3857 = xr.DataArray(
        arr, dims=['y', 'x'], coords={'y': y, 'x': x},
        attrs={'nodata': -9999.0, 'crs': 'EPSG:3857'},
    )
    src_3857 = str(tmp_path / f"{_uniq('src_3857_chunked')}.tif")
    to_geotiff(da_3857, src_3857)

    body = _simple_source_xml(src_4326) + "\n" + _simple_source_xml(src_3857)
    xml = _vrt_xml(body=body, srs='EPSG:4326')
    vrt_path = _write_vrt(tmp_path, xml)

    # ``chunks=`` routes through ``_read_vrt_chunked`` rather than the
    # eager dispatcher; the validator must fire here too.
    with pytest.raises(VRTUnsupportedError):
        _package_read_vrt(vrt_path, chunks=(2, 2))


def test_matching_source_crs_passes_validator(tmp_path):
    """Positive pin: a VRT whose source CRS matches the VRT-declared
    SRS must not be rejected by the new per-source CRS check. Catches
    a regression where the validator started raising on a homogeneous
    mosaic (e.g. by reading ``crs_wkt`` for one side and ``crs_epsg``
    for the other and failing to canonicalise)."""
    src_a = _write_src_float32_geotiff(tmp_path, crs='EPSG:4326')
    src_b = _write_src_float32_geotiff(tmp_path, crs='EPSG:4326')
    body = _simple_source_xml(src_a) + "\n" + _simple_source_xml(src_b)
    xml = _vrt_xml(body=body, srs='EPSG:4326')
    vrt_path = _write_vrt(tmp_path, xml)

    # No raise: validator accepts. The reader returns a DataArray
    # regardless of mosaic correctness, which the existing parity
    # tests cover separately.
    da = _package_read_vrt(vrt_path)
    assert da is not None


def test_mixed_source_crs_no_vrt_srs_rejects(tmp_path):
    """When the VRT carries no ``<SRS>``, the reader inherits the CRS
    from the first source. Sources that disagree among themselves are
    still a silent-correctness gap and must be rejected. The validator
    uses the first parseable source as the reference."""
    src_4326 = _write_src_float32_geotiff(tmp_path, crs='EPSG:4326')
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    y = np.linspace(1.0, 0.0, 4)
    x = np.linspace(0.0, 1.0, 4)
    da_3857 = xr.DataArray(
        arr, dims=['y', 'x'], coords={'y': y, 'x': x},
        attrs={'nodata': -9999.0, 'crs': 'EPSG:3857'},
    )
    src_3857 = str(tmp_path / f"{_uniq('src_3857_no_srs')}.tif")
    to_geotiff(da_3857, src_3857)

    body = _simple_source_xml(src_4326) + "\n" + _simple_source_xml(src_3857)
    # No ``<SRS>`` element: build the XML inline rather than through
    # ``_vrt_xml`` (which always emits one).
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
{body}
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, xml)

    with pytest.raises(VRTUnsupportedError) as excinfo:
        _package_read_vrt(vrt_path)
    msg = str(excinfo.value).lower()
    # Message must surface the "no <SRS>" reasoning so the caller
    # understands why their VRT was rejected even though they did not
    # declare a top-level CRS.
    assert 'no <srs>' in msg or 'no srs' in msg
    assert src_3857 in str(excinfo.value)


_VRT_TO_NP_DTYPE = {
    "Byte": np.uint8,
    "UInt16": np.uint16,
    "Int16": np.int16,
    "UInt32": np.uint32,
    "Int32": np.int32,
    "Float32": np.float32,
    "Float64": np.float64,
}


@pytest.mark.parametrize(
    "name_a, name_b",
    [
        ("UInt16", "Float32"),
        ("Int32", "Float64"),
    ],
    ids=lambda n: n,
)
@pytest.mark.parametrize("chunks", [None, 2], ids=["eager", "chunked"])
def test_mixed_source_dtype_ambiguous_widening_rejected(
        tmp_path, name_a, name_b, chunks):
    """Bands declaring incompatible dtypes must raise rather than
    silently widen via ``np.result_type``.

    The VRT support matrix at ``_backends/vrt.py`` requires that
    per-band dtype mismatches surface as ``MixedBandMetadataError``.
    Covers both the original ``UInt16`` + ``Float32`` case and an
    integer-floating combo at higher precision (``Int32`` + ``Float64``)
    so the rule is enforced generally. Parametrising over
    ``chunks=None`` (eager) and ``chunks=2`` (dask) pins the same
    behaviour on both reader paths so a future change that detours
    around ``_effective_dtype_for_bands`` in the chunked path cannot
    regress this silently. Issue #2485.
    """
    src_a = _write_src_float32_geotiff(
        tmp_path, dtype=_VRT_TO_NP_DTYPE[name_a])
    src_b = _write_src_float32_geotiff(
        tmp_path, dtype=_VRT_TO_NP_DTYPE[name_b])
    body_b1 = _simple_source_xml(src_a)
    body_b2 = _simple_source_xml(src_b)
    xml = f"""<VRTDataset rasterXSize="4" rasterYSize="4">
  <SRS>EPSG:4326</SRS>
  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="{name_a}" band="1">
{body_b1}
  </VRTRasterBand>
  <VRTRasterBand dataType="{name_b}" band="2">
{body_b2}
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path = _write_vrt(tmp_path, xml)

    with pytest.raises(MixedBandMetadataError) as excinfo:
        if chunks is None:
            _package_read_vrt(vrt_path)
        else:
            _package_read_vrt(vrt_path, chunks=chunks)
    msg = str(excinfo.value).lower()
    assert any(k in msg for k in ('dtype', 'datatype', 'mixed'))
    # Message should name both conflicting bands so the caller can
    # locate the disagreement without reading the VRT XML by hand.
    assert 'band 1' in msg
    assert 'band 2' in msg
    # And the VRT path so callers can locate the file from the
    # traceback without re-parsing the XML.
    assert vrt_path in str(excinfo.value)


def test_supported_simple_vrt_round_trips_via_open_geotiff(tmp_path):
    """Sanity anchor: a supported single-source VRT opens cleanly via
    ``open_geotiff``, so the negative tests are exercising a live
    extension-dispatch path rather than a broken accessor."""
    src_path = _write_src_float32_geotiff(tmp_path)
    body = _simple_source_xml(src_path)
    xml = _vrt_xml(body=body)
    vrt_path = _write_vrt(tmp_path, xml)
    da = open_geotiff(vrt_path)
    assert da.shape == (4, 4)


# ---------------------------------------------------------------------------
# Reader-error narrowing: ``read_vrt`` historically ``except Exception``-ed
# every source read, swallowing real bugs. The catch is now
# narrowed to I/O / parse / codec-decode errors only.
#
# The matrix is parametrised over exception class x mode:
#   * ``io_or_parse`` (default mode): warn-and-continue
#   * ``io_or_parse`` (strict mode): re-raise
#   * ``bug`` (any mode): propagate
# ---------------------------------------------------------------------------


@pytest.fixture
def clear_strict_env(monkeypatch):
    """``XRSPATIAL_GEOTIFF_STRICT`` unset for default-mode tests."""
    monkeypatch.delenv('XRSPATIAL_GEOTIFF_STRICT', raising=False)


@pytest.fixture
def set_strict_env(monkeypatch):
    """``XRSPATIAL_GEOTIFF_STRICT=1`` for strict-mode tests."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')


def _write_simple_vrt(tmp_path, src_path, *, name: str | None = None):
    """Write a 4x4 single-source float VRT pointing at ``src_path``."""
    name = name or _uniq("simple")
    vrt_path = tmp_path / f"{name}.vrt"
    vrt_path.write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <SRS></SRS>\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <NoDataValue>-9999</NoDataValue>\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{src_path}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    return vrt_path


def _patch_read_to_array(monkeypatch, exc):
    """Make ``_reader.read_to_array`` raise ``exc`` on every call.

    ``read_vrt`` does a local ``from ._reader import read_to_array``
    inside the function body, so patching the source attribute is
    enough -- the import picks up the stub at call time."""
    from xrspatial.geotiff import _reader

    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(_reader, 'read_to_array', _boom)


def _has_zstandard():
    try:
        import zstandard  # noqa: F401
    except ImportError:
        return False
    return True


_NARROW_EXCEPT_IO_OR_PARSE_CASES = [
    pytest.param(
        FileNotFoundError("synthetic missing"),
        'FileNotFoundError',
        id="io[file-not-found]",
    ),
    pytest.param(
        ValueError("bad header"),
        'ValueError',
        id="parse[value-error]",
    ),
    pytest.param(
        struct.error("short buffer"),
        'error',  # ``struct.error`` type name is ``error``
        id="parse[struct-error]",
    ),
    pytest.param(
        PermissionError("denied"),
        'PermissionError',
        id="io[permission-error]",
    ),
    pytest.param(
        zlib.error("synthetic deflate"),
        'error',  # ``zlib.error`` type name is ``error``
        id="codec[zlib-error]",
    ),
]


@pytest.mark.parametrize(
    "exc, expected_type_name",
    _NARROW_EXCEPT_IO_OR_PARSE_CASES,
)
def test_narrow_except_io_or_parse_warns_in_default_mode(
    clear_strict_env, monkeypatch, tmp_path, exc, expected_type_name,
):
    """I/O and parse failures warn-and-continue when
    ``missing_sources='warn'`` is opted in. The warning message names
    the source and the underlying exception type."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / f"src_{_uniq('narrow')}.tif"
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(tmp_path, str(src_path))

    _patch_read_to_array(monkeypatch, exc)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        da = read_vrt(str(vrt_path), missing_sources='warn')

    assert da.shape == (4, 4)
    fallback = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert fallback
    msgs = ' '.join(str(x.message) for x in fallback)
    assert 'VRT source' in msgs
    assert expected_type_name in msgs


@pytest.mark.parametrize(
    "exc, _expected_type_name",
    _NARROW_EXCEPT_IO_OR_PARSE_CASES,
)
def test_narrow_except_io_or_parse_reraises_in_strict_mode(
    set_strict_env, monkeypatch, tmp_path, exc, _expected_type_name,
):
    """Strict mode re-raises all I/O / parse / codec-decode failures."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / f"src_{_uniq('strict')}.tif"
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(tmp_path, str(src_path))

    _patch_read_to_array(monkeypatch, exc)

    with pytest.raises(type(exc)):
        read_vrt(str(vrt_path))


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(RuntimeError("synthetic bug"), id="bug[runtime-error]"),
        pytest.param(MemoryError("OOM"), id="bug[memory-error]"),
    ],
)
def test_narrow_except_bug_classes_propagate_in_default_mode(
    clear_strict_env, monkeypatch, tmp_path, exc,
):
    """Non-I/O bugs (``RuntimeError`` here as a stand-in, plus
    ``MemoryError``) must propagate even in default mode -- they are
    real failures, not "unreadable source" cases."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / f"src_{_uniq('bug')}.tif"
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(tmp_path, str(src_path))

    _patch_read_to_array(monkeypatch, exc)

    with pytest.raises(type(exc)):
        read_vrt(str(vrt_path))


def test_narrow_except_runtime_error_propagates_in_strict_mode(
    set_strict_env, monkeypatch, tmp_path,
):
    """Strict mode propagates non-I/O bugs too (double-checks the
    runtime-error case under the strict flag)."""
    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / f"src_{_uniq('bug_strict')}.tif"
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(tmp_path, str(src_path))

    _patch_read_to_array(monkeypatch, RuntimeError("synthetic strict"))

    with pytest.raises(RuntimeError, match='synthetic strict'):
        read_vrt(str(vrt_path))


@pytest.mark.skipif(not _has_zstandard(),
                    reason="zstandard not installed")
def test_narrow_except_zstd_error_warns_in_default_mode(
    clear_strict_env, monkeypatch, tmp_path,
):
    """``zstandard.ZstdError`` is the zstd-codec equivalent of
    ``zlib.error``: warn-and-continue under the lenient policy so a
    single corrupt ZSTD tile does not abort the whole mosaic."""
    from zstandard import ZstdError

    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / f"src_{_uniq('zstd')}.tif"
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(tmp_path, str(src_path))

    _patch_read_to_array(monkeypatch, ZstdError("synthetic zstd"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        da = read_vrt(str(vrt_path), missing_sources='warn')

    assert da.shape == (4, 4)
    fallback = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert fallback
    msgs = ' '.join(str(x.message) for x in fallback)
    assert 'VRT source' in msgs
    assert 'ZstdError' in msgs


@pytest.mark.skipif(not _has_zstandard(),
                    reason="zstandard not installed")
def test_narrow_except_zstd_error_reraises_in_strict_mode(
    set_strict_env, monkeypatch, tmp_path,
):
    from zstandard import ZstdError

    from xrspatial.geotiff import read_vrt

    src_path = tmp_path / f"src_{_uniq('zstd_strict')}.tif"
    src_path.write_bytes(b'placeholder')
    vrt_path = _write_simple_vrt(tmp_path, str(src_path))

    _patch_read_to_array(monkeypatch, ZstdError("synthetic zstd strict"))

    with pytest.raises(ZstdError, match='synthetic zstd strict'):
        read_vrt(str(vrt_path))


# ---------------------------------------------------------------------------
# Path containment: ``parse_vrt`` rejects sources that resolve outside
# ``vrt_dir`` unless the absolute path lands under one of the
# ``XRSPATIAL_VRT_ALLOWED_ROOTS`` entries.
# ---------------------------------------------------------------------------


@pytest.fixture
def clear_allowlist_env(monkeypatch):
    """Make sure no stray XRSPATIAL_VRT_ALLOWED_ROOTS leaks across tests."""
    monkeypatch.delenv('XRSPATIAL_VRT_ALLOWED_ROOTS', raising=False)


def _unique_dir(tmp_path, label: str) -> str:
    """Sub-directory carrying a uuid so parallel workers cannot collide."""
    d = tmp_path / _uniq(label)
    d.mkdir()
    return str(d)


def _write_minimal_tif(path: str) -> None:
    """4x4 float32 GeoTIFF the path-containment tests reference."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    y = np.linspace(1.0, 0.0, 4)
    x = np.linspace(0.0, 1.0, 4)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326},
    )
    to_geotiff(da, path, compression='none')


def _build_containment_vrt(vrt_path: str, source_filename: str,
                           relative: str) -> None:
    """Write a 4x4 single-band VRT pointing at ``source_filename``."""
    xml = (
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="{relative}">'
        f'{source_filename}</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(xml)


class TestPathContainment:
    """Sources that resolve outside the VRT directory raise
    ``ValueError`` at parse / read time. The ``allowlist`` opt-in
    expands the trusted root set without weakening the relative-source
    rejection."""

    def test_relative_dotdot_traversal_rejected(
        self, clear_allowlist_env, tmp_path,
    ):
        """A relative source resolving outside ``vrt_dir`` raises at
        ``parse_vrt`` so the dangerous path never reaches
        ``read_to_array``."""
        vrt_dir = _unique_dir(tmp_path, "trav_rel")
        xml = (
            '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
            '  <VRTRasterBand dataType="Float32" band="1">\n'
            '    <SimpleSource>\n'
            '      <SourceFilename relativeToVRT="1">'
            '../../../../../etc/passwd</SourceFilename>\n'
            '      <SourceBand>1</SourceBand>\n'
            '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '    </SimpleSource>\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
        with pytest.raises(ValueError, match="outside the VRT directory"):
            parse_vrt(xml, vrt_dir)

    def test_relative_symlink_traversal_rejected(
        self, clear_allowlist_env, tmp_path,
    ):
        """A symlink under ``vrt_dir`` pointing outside is rejected via
        ``realpath``."""
        vrt_dir = _unique_dir(tmp_path, "trav_sym")
        outside_dir = _unique_dir(tmp_path, "trav_sym_outside")
        outside_target = os.path.join(outside_dir, 'secret.tif')
        _write_minimal_tif(outside_target)

        sym = os.path.join(vrt_dir, 'inside.tif')
        try:
            os.symlink(outside_target, sym)
        except (OSError, NotImplementedError) as e:
            pytest.skip(f"symlink not supported in this environment: {e}")

        vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
        _build_containment_vrt(vrt_path, 'inside.tif', relative='1')

        with pytest.raises(ValueError, match="outside the VRT directory"):
            _internal_read_vrt(vrt_path)

    def test_absolute_outside_vrt_dir_rejected(
        self, clear_allowlist_env, tmp_path,
    ):
        vrt_dir = _unique_dir(tmp_path, "abs_outside")
        outside_dir = _unique_dir(tmp_path, "abs_outside_target")
        outside_tif = os.path.join(outside_dir, 'data.tif')
        _write_minimal_tif(outside_tif)

        vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
        _build_containment_vrt(vrt_path, outside_tif, relative='0')

        with pytest.raises(ValueError, match="outside the VRT directory"):
            _internal_read_vrt(vrt_path)

    def test_absolute_inside_vrt_dir_ok(
        self, clear_allowlist_env, tmp_path,
    ):
        """An absolute path that happens to resolve under ``vrt_dir``
        passes -- mirrors the writer's ``relative=False`` round-trip
        case."""
        vrt_dir = _unique_dir(tmp_path, "abs_inside")
        tif_path = os.path.join(vrt_dir, 'data.tif')
        _write_minimal_tif(tif_path)

        vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
        _build_containment_vrt(vrt_path, tif_path, relative='0')

        arr, _ = _internal_read_vrt(vrt_path)
        assert arr.shape == (4, 4)

    def test_normal_relative_source_under_vrt_dir_ok(
        self, clear_allowlist_env, tmp_path,
    ):
        """Happy-path regression: a plain relative source under the VRT
        directory still reads fine."""
        vrt_dir = _unique_dir(tmp_path, "happy")
        tif_path = os.path.join(vrt_dir, 'data.tif')
        _write_minimal_tif(tif_path)

        vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
        _build_containment_vrt(vrt_path, 'data.tif', relative='1')

        arr, _ = _internal_read_vrt(vrt_path)
        assert arr.shape == (4, 4)

    def test_error_message_names_rejected_path(
        self, clear_allowlist_env, tmp_path,
    ):
        """The ``ValueError`` mentions both the offending resolved path
        and the trusted root so operators can diagnose the rejection."""
        vrt_dir = _unique_dir(tmp_path, "msg_check")
        xml = (
            '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
            '  <VRTRasterBand dataType="Float32" band="1">\n'
            '    <SimpleSource>\n'
            '      <SourceFilename relativeToVRT="1">'
            '../../etc/shadow</SourceFilename>\n'
            '      <SourceBand>1</SourceBand>\n'
            '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
            '    </SimpleSource>\n'
            '  </VRTRasterBand>\n'
            '</VRTDataset>\n'
        )
        with pytest.raises(ValueError) as excinfo:
            parse_vrt(xml, vrt_dir)
        msg = str(excinfo.value)
        assert 'shadow' in msg
        assert os.path.realpath(vrt_dir) in msg


class TestPathContainmentAllowlist:
    """The ``XRSPATIAL_VRT_ALLOWED_ROOTS`` env var opts in cross-directory
    reads. The opt-in is for absolute sources only -- relative sources
    that try to escape ``vrt_dir`` stay rejected even when the resolved
    path lands under an allowlisted root."""

    def test_single_root_allows_outside_absolute(self, tmp_path, monkeypatch):
        vrt_dir = _unique_dir(tmp_path, "allow_vrt")
        outside_dir = _unique_dir(tmp_path, "allow_data")
        outside_tif = os.path.join(outside_dir, 'data.tif')
        _write_minimal_tif(outside_tif)

        vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
        _build_containment_vrt(vrt_path, outside_tif, relative='0')

        monkeypatch.setenv('XRSPATIAL_VRT_ALLOWED_ROOTS', outside_dir)
        arr, _ = _internal_read_vrt(vrt_path)
        assert arr.shape == (4, 4)

    def test_multiple_roots_pathsep_separated(self, tmp_path, monkeypatch):
        vrt_dir = _unique_dir(tmp_path, "multi_vrt")
        dir_a = _unique_dir(tmp_path, "multi_a")
        dir_b = _unique_dir(tmp_path, "multi_b")
        tif_b = os.path.join(dir_b, 'data.tif')
        _write_minimal_tif(tif_b)

        vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
        _build_containment_vrt(vrt_path, tif_b, relative='0')

        monkeypatch.setenv(
            'XRSPATIAL_VRT_ALLOWED_ROOTS',
            os.pathsep.join([dir_a, dir_b]),
        )
        arr, _ = _internal_read_vrt(vrt_path)
        assert arr.shape == (4, 4)

    def test_relative_source_escape_still_rejected(self, tmp_path, monkeypatch):
        """A relative source that escapes ``vrt_dir`` stays rejected even
        when the resolved path lands under an allowlisted root.
        Relative paths declare intent to stay inside the VRT directory;
        honouring that intent prevents chaining an allowlist entry into
        a relative-source traversal."""
        vrt_dir = _unique_dir(tmp_path, "rel_with_allow")
        outside_dir = _unique_dir(tmp_path, "rel_with_allow_target")
        outside_tif = os.path.join(outside_dir, 'data.tif')
        _write_minimal_tif(outside_tif)

        rel = os.path.relpath(outside_tif, vrt_dir)
        vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
        _build_containment_vrt(vrt_path, rel, relative='1')

        monkeypatch.setenv('XRSPATIAL_VRT_ALLOWED_ROOTS', outside_dir)
        with pytest.raises(ValueError, match="outside the VRT directory"):
            _internal_read_vrt(vrt_path)

    def test_empty_entries_ignored(self, tmp_path, monkeypatch):
        """Leading / trailing / embedded empty entries from stray
        separators are skipped (and never grant access to ``/``)."""
        vrt_dir = _unique_dir(tmp_path, "empty_entry_vrt")
        outside_dir = _unique_dir(tmp_path, "empty_entry_data")
        outside_tif = os.path.join(outside_dir, 'data.tif')
        _write_minimal_tif(outside_tif)

        vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
        _build_containment_vrt(vrt_path, outside_tif, relative='0')

        sep = os.pathsep
        value = f"{sep}{outside_dir}{sep}{sep}"
        monkeypatch.setenv('XRSPATIAL_VRT_ALLOWED_ROOTS', value)
        arr, _ = _internal_read_vrt(vrt_path)
        assert arr.shape == (4, 4)


# ===========================================================================
# Validation tail cases
# ===========================================================================
#
# * SrcRect negative-size / negative-offset rejection.
# * ``open_geotiff('.vrt')`` rejecting kwargs it silently dropped:
#   ``overview_level`` and ``on_gpu_failure``.
# * ``to_geotiff(..., '.vrt')`` rejecting ``tiled=False`` and validating
#   ``tile_size`` up front instead of crashing in the writer.


# ---------------------------------------------------------------------------
# SrcRect negative-size / negative-offset rejection
# ---------------------------------------------------------------------------
#
# A malformed ``<SrcRect xSize="-100"/>`` (or negative offset) must surface
# as a ``ValueError`` naming the offending field, in both lenient and
# strict modes -- never get swallowed by the missing-source fallback.


def _srcrect_write_source(td: str, name: str = 'src.tif') -> str:
    """Write a 10x10 uint8 source GeoTIFF and return its path."""
    src_path = os.path.join(td, name)
    to_geotiff(np.zeros((10, 10), dtype=np.uint8), src_path,
               compression='none')
    return src_path


def _srcrect_write_vrt(td: str, *,
                       src_x_off: int = 0, src_y_off: int = 0,
                       src_x_size: int = 10, src_y_size: int = 10,
                       src_filename: str = 'src.tif',
                       raster_x: int = 100, raster_y: int = 100) -> str:
    """Write a VRT with a single SimpleSource using the given SrcRect."""
    vrt_path = os.path.join(td, 'mosaic.vrt')
    vrt_xml = (
        f'<VRTDataset rasterXSize="{raster_x}" rasterYSize="{raster_y}">\n'
        f'  <VRTRasterBand dataType="Byte" band="1">\n'
        f'    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="1">{src_filename}'
        f'</SourceFilename>\n'
        f'      <SourceBand>1</SourceBand>\n'
        f'      <SrcRect xOff="{src_x_off}" yOff="{src_y_off}" '
        f'xSize="{src_x_size}" ySize="{src_y_size}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="10" ySize="10"/>\n'
        f'    </SimpleSource>\n'
        f'  </VRTRasterBand>\n'
        f'</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)
    return vrt_path


class TestSrcRectRejection:
    """Malformed ``<SrcRect>`` geometry rejected before the lenient
    missing-source fallback can swallow it."""

    def test_negative_x_size_rejected(self, tmp_path):
        td = str(tmp_path)
        _srcrect_write_source(td)
        vrt_path = _srcrect_write_vrt(td, src_x_size=-50)
        with pytest.raises(ValueError, match=r"SrcRect.*negative size"):
            _internal_read_vrt(vrt_path)

    def test_negative_y_size_rejected(self, tmp_path):
        td = str(tmp_path)
        _srcrect_write_source(td)
        vrt_path = _srcrect_write_vrt(td, src_y_size=-50)
        with pytest.raises(ValueError, match=r"SrcRect.*negative size"):
            _internal_read_vrt(vrt_path)

    def test_negative_x_off_rejected(self, tmp_path):
        td = str(tmp_path)
        _srcrect_write_source(td)
        vrt_path = _srcrect_write_vrt(td, src_x_off=-10)
        with pytest.raises(ValueError, match=r"SrcRect.*negative offset"):
            _internal_read_vrt(vrt_path)

    def test_negative_y_off_rejected(self, tmp_path):
        td = str(tmp_path)
        _srcrect_write_source(td)
        vrt_path = _srcrect_write_vrt(td, src_y_off=-10)
        with pytest.raises(ValueError, match=r"SrcRect.*negative offset"):
            _internal_read_vrt(vrt_path)

    def test_message_names_bad_values(self, tmp_path):
        """The error message names the malformed field and its value so
        the caller can find the offending ``<SrcRect>`` in the VRT."""
        td = str(tmp_path)
        _srcrect_write_source(td)
        vrt_path = _srcrect_write_vrt(td, src_x_size=-7, src_y_size=-3)
        with pytest.raises(ValueError) as excinfo:
            _internal_read_vrt(vrt_path)
        msg = str(excinfo.value)
        assert "SrcRect" in msg
        assert "-7" in msg
        assert "-3" in msg

    def test_missing_source_still_takes_lenient_warning_path(self, tmp_path):
        """A *valid* SrcRect with a missing source file still hits the
        lenient warning path -- the SrcRect check must not swallow the
        missing-file case. ``missing_sources='warn'`` opts into the
        lenient branch since the default is now ``'raise'``."""
        td = str(tmp_path)
        # No source file written; SrcRect itself is well-formed.
        vrt_path = _srcrect_write_vrt(td, src_filename='does_not_exist.tif')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            arr, _ = _internal_read_vrt(vrt_path, missing_sources='warn')
        fallback = [w for w in caught
                    if issubclass(w.category, GeoTIFFFallbackWarning)]
        assert fallback, (
            "expected a GeoTIFFFallbackWarning for the missing source"
        )
        assert arr.shape == (100, 100)

    def test_valid_srcrect_reads_normally(self, tmp_path):
        """A well-formed SrcRect with a real source succeeds -- no false
        positives on valid VRTs."""
        td = str(tmp_path)
        _srcrect_write_source(td)
        vrt_path = _srcrect_write_vrt(td, raster_x=10, raster_y=10)
        arr, _ = _internal_read_vrt(vrt_path)
        assert arr.shape == (10, 10)
        assert np.all(arr == 0)

    def test_negative_srcrect_raises_under_strict_mode(
        self, tmp_path, monkeypatch,
    ):
        """The check runs before the lenient try/except, so strict mode
        and lenient mode both raise."""
        monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')
        td = str(tmp_path)
        _srcrect_write_source(td)
        vrt_path = _srcrect_write_vrt(td, src_x_size=-50)
        with pytest.raises(ValueError, match=r"SrcRect.*negative size"):
            _internal_read_vrt(vrt_path)


# ---------------------------------------------------------------------------
# open_geotiff('.vrt') kwarg-drop rejection
# ---------------------------------------------------------------------------
#
# ``open_geotiff`` documents ``overview_level`` and ``on_gpu_failure`` but
# the VRT dispatch branch routes to ``read_vrt`` whose signature accepts
# neither, so the kwargs were silently dropped. The fix refuses the
# unsupported combinations up front.


@pytest.fixture
def _kwarg_drop_small_vrt(tmp_path):
    """Two-tile uint16 VRT for the kwarg-drop rejection cases."""
    arr_a = np.arange(16, dtype=np.uint16).reshape(4, 4)
    da_a = xr.DataArray(
        arr_a, dims=["y", "x"],
        coords={
            "y": np.array([0.5, 1.5, 2.5, 3.5]),
            "x": np.array([0.5, 1.5, 2.5, 3.5]),
        },
        attrs={"crs": 4326},
    )
    tile_a = tmp_path / "tile_a.tif"
    to_geotiff(da_a, str(tile_a))

    arr_b = np.arange(16, 32, dtype=np.uint16).reshape(4, 4)
    da_b = xr.DataArray(
        arr_b, dims=["y", "x"],
        coords={
            "y": np.array([0.5, 1.5, 2.5, 3.5]),
            "x": np.array([4.5, 5.5, 6.5, 7.5]),
        },
        attrs={"crs": 4326},
    )
    tile_b = tmp_path / "tile_b.tif"
    to_geotiff(da_b, str(tile_b))

    from xrspatial.geotiff import write_vrt
    vrt_path = tmp_path / "mosaic.vrt"
    write_vrt(str(vrt_path), [str(tile_a), str(tile_b)])
    return str(vrt_path)


class TestOpenGeotiffVrtKwargRejection:
    """``open_geotiff('.vrt')`` rejects kwargs it used to silently drop."""

    def test_rejects_overview_level(self, _kwarg_drop_small_vrt):
        """``overview_level`` plus ``.vrt`` raises, not a silent drop."""
        with pytest.raises(
            ValueError, match="overview_level is not supported",
        ):
            open_geotiff(_kwarg_drop_small_vrt, overview_level=1)

    def test_accepts_overview_level_zero(self, _kwarg_drop_small_vrt):
        """``overview_level=0`` is full resolution (the default), so it is
        equivalent to omitting the kwarg and must not raise."""
        da = open_geotiff(_kwarg_drop_small_vrt, overview_level=0)
        assert da.shape == (4, 8)

    def test_rejects_on_gpu_failure_with_gpu_true(self, _kwarg_drop_small_vrt):
        """``on_gpu_failure='strict'`` plus ``.vrt`` (gpu=True) is refused.

        The check fires before any GPU code runs; no CUDA needed."""
        with pytest.raises(
            ValueError, match="on_gpu_failure is not supported",
        ):
            open_geotiff(
                _kwarg_drop_small_vrt, gpu=True, on_gpu_failure="strict",
            )

    def test_without_unsupported_kwargs_still_works(self, _kwarg_drop_small_vrt):
        """The previously-accepted kwargs still flow through to
        ``read_vrt``."""
        da = open_geotiff(_kwarg_drop_small_vrt)
        assert da.shape == (4, 8)

    def test_with_window_still_works(self, _kwarg_drop_small_vrt):
        """``window`` was already forwarded; the fix must not break it."""
        da = open_geotiff(_kwarg_drop_small_vrt, window=(0, 1, 4, 5))
        assert da.shape == (4, 4)

    def test_non_vrt_still_accepts_overview_level(self, tmp_path):
        """The fix is VRT-specific; ``.tif`` sources keep accepting
        ``overview_level``."""
        arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
        da = xr.DataArray(
            arr, dims=["y", "x"],
            coords={
                "y": np.arange(8, dtype=np.float64),
                "x": np.arange(8, dtype=np.float64),
            },
            attrs={"crs": 4326},
        )
        tif_path = tmp_path / "with_ovr.tif"
        to_geotiff(
            da, str(tif_path), cog=True, tile_size=16, overview_levels=[2],
        )
        open_geotiff(str(tif_path), overview_level=0)
        open_geotiff(str(tif_path), overview_level=1)


# ---------------------------------------------------------------------------
# to_geotiff('.vrt') tiled / tile_size validation
# ---------------------------------------------------------------------------
#
# ``to_geotiff(..., '.vrt', tiled=False)`` used to warn then crash with
# ``ZeroDivisionError`` inside the always-tiling VRT writer. The fix
# refuses ``tiled=False`` for ``.vrt`` and validates ``tile_size``
# unconditionally so callers get a clear ``ValueError`` up front.


def _tiled_validation_make_da(shape=(64, 64)):
    arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return xr.DataArray(arr, dims=['y', 'x'])


class TestVrtTiledValidation:
    """VRT writer rejects ``tiled=False`` and bad ``tile_size`` up front."""

    def test_rejects_tiled_false(self, tmp_path):
        """``tiled=False`` is not a valid request for VRT output."""
        da = _tiled_validation_make_da()
        out = os.path.join(str(tmp_path), 'vrt_tiled_false.vrt')
        with pytest.raises(ValueError, match='tiled=False is not compatible'):
            to_geotiff(da, out, tiled=False)

    def test_tiled_false_zero_tile_size_raises_value_error(self, tmp_path):
        """``tiled=False`` plus ``tile_size=0`` raises ``ValueError``, not
        the previous ``ZeroDivisionError`` from inside the writer."""
        da = _tiled_validation_make_da()
        out = os.path.join(str(tmp_path), 'vrt_tiled_false_zero.vrt')
        with pytest.raises(ValueError) as exc:
            to_geotiff(da, out, tiled=False, tile_size=0)
        assert not isinstance(exc.value, ZeroDivisionError)

    def test_zero_tile_size_default_tiled_raises_value_error(self, tmp_path):
        """With the default ``tiled=True``, ``tile_size=0`` surfaces from
        the shared ``_validate_tile_size`` check, not a deep
        ``ZeroDivisionError``."""
        da = _tiled_validation_make_da()
        out = os.path.join(str(tmp_path), 'vrt_default_tiled_zero.vrt')
        with pytest.raises(ValueError, match='tile_size'):
            to_geotiff(da, out, tile_size=0)

    def test_default_args_still_succeeds(self, tmp_path):
        """The default-args VRT write path is unaffected by the fix."""
        da = _tiled_validation_make_da()
        out = os.path.join(str(tmp_path), 'vrt_default.vrt')
        to_geotiff(da, out)
        assert os.path.exists(out)
