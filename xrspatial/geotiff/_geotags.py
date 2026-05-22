"""GeoTIFF tag interpretation: CRS, affine transform, GeoKeys."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


# Stamped by the reader on arrays read from files that carry no
# GeoTIFF transform tags (ModelTransformation, ModelPixelScale, or
# ModelTiepoint). The reader emits int64 ``arange`` placeholder y/x
# coords for those files, and the writer checks this marker to
# distinguish them from user-authored int64 step-1 grids that match
# the same shape. Lives here (alongside the other geotiff tag
# constants) so both ``_coords`` and ``_attrs`` can import it
# without a cycle. See issue #2120.
_NO_GEOREF_KEY = '_xrspatial_no_georef'

from ._header import (
    IFD,
    TAG_NEW_SUBFILE_TYPE,
    TAG_IMAGE_WIDTH, TAG_IMAGE_LENGTH, TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION, TAG_PHOTOMETRIC,
    TAG_STRIP_OFFSETS, TAG_ORIENTATION, TAG_SAMPLES_PER_PIXEL,
    TAG_ROWS_PER_STRIP, TAG_STRIP_BYTE_COUNTS,
    TAG_X_RESOLUTION, TAG_Y_RESOLUTION,
    TAG_PLANAR_CONFIG, TAG_RESOLUTION_UNIT,
    TAG_PREDICTOR, TAG_COLORMAP, TAG_SUB_IFDS,
    TAG_TILE_WIDTH, TAG_TILE_LENGTH,
    TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS,
    TAG_EXTRA_SAMPLES,
    TAG_SAMPLE_FORMAT, TAG_GDAL_METADATA, TAG_GDAL_NODATA,
    TAG_MODEL_PIXEL_SCALE, TAG_MODEL_TIEPOINT,
    TAG_MODEL_TRANSFORMATION,
    TAG_GEO_KEY_DIRECTORY, TAG_GEO_DOUBLE_PARAMS, TAG_GEO_ASCII_PARAMS,
)
from ._dtypes import resolve_bits_per_sample
from ._errors import RotatedTransformError, UnknownCRSModelTypeError

# ImageDescription tag (270). Captured for round-trip but not managed
# by the writer -- it flows through extra_tags pass-through.
TAG_IMAGE_DESCRIPTION = 270

# Tags the writer manages directly. Tags not in this set are collected
# into GeoInfo.extra_tags on read and re-emitted on write via the
# extra_tags pass-through. ColorMap (320), ExtraSamples (338, only emitted
# automatically when samples > 1), and ImageDescription (270) intentionally
# stay OUT of this set so they round-trip without dedicated writer plumbing.
#
# NewSubfileType (254) and SubIFDs (330) are also managed: NewSubfileType
# is a per-IFD status flag (overview / mask marker) that the writer emits
# on its own for level > 0 IFDs, so leaking the source value to extra_tags
# would mis-mark a primary IFD as an overview after a read overview ->
# write round-trip. SubIFDs holds absolute byte offsets into the source
# file, which become garbage in the rewritten output. See issue #1657.
_MANAGED_TAGS = frozenset({
    TAG_NEW_SUBFILE_TYPE,
    TAG_IMAGE_WIDTH, TAG_IMAGE_LENGTH, TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION, TAG_PHOTOMETRIC,
    TAG_STRIP_OFFSETS, TAG_ORIENTATION, TAG_SAMPLES_PER_PIXEL,
    TAG_ROWS_PER_STRIP, TAG_STRIP_BYTE_COUNTS,
    TAG_X_RESOLUTION, TAG_Y_RESOLUTION,
    TAG_PLANAR_CONFIG, TAG_RESOLUTION_UNIT,
    TAG_PREDICTOR, TAG_SUB_IFDS,
    TAG_TILE_WIDTH, TAG_TILE_LENGTH,
    TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS,
    TAG_SAMPLE_FORMAT, TAG_GDAL_METADATA, TAG_GDAL_NODATA,
    TAG_MODEL_PIXEL_SCALE, TAG_MODEL_TIEPOINT,
    TAG_MODEL_TRANSFORMATION,
    TAG_GEO_KEY_DIRECTORY, TAG_GEO_DOUBLE_PARAMS, TAG_GEO_ASCII_PARAMS,
})

# GeoKey IDs
GEOKEY_MODEL_TYPE = 1024
GEOKEY_RASTER_TYPE = 1025
GEOKEY_CITATION = 1026
GEOKEY_GEOGRAPHIC_TYPE = 2048
GEOKEY_GEOG_CITATION = 2049
GEOKEY_GEODETIC_DATUM = 2050
GEOKEY_GEOG_LINEAR_UNITS = 2052
GEOKEY_GEOG_ANGULAR_UNITS = 2054
GEOKEY_GEOG_SEMI_MAJOR_AXIS = 2057
GEOKEY_GEOG_SEMI_MINOR_AXIS = 2058
GEOKEY_GEOG_INV_FLATTENING = 2059
GEOKEY_PROJECTED_CS_TYPE = 3072
GEOKEY_PROJ_CITATION = 3073
GEOKEY_PROJECTION = 3074
GEOKEY_PROJ_LINEAR_UNITS = 3076
GEOKEY_VERTICAL_CS_TYPE = 4096
GEOKEY_VERTICAL_CITATION = 4097
GEOKEY_VERTICAL_DATUM = 4098
GEOKEY_VERTICAL_UNITS = 4099

# Well-known EPSG unit codes
ANGULAR_UNITS = {
    9101: 'radian',
    9102: 'degree',
    9103: 'arc-minute',
    9104: 'arc-second',
    9105: 'grad',
}

LINEAR_UNITS = {
    9001: 'metre',
    9002: 'foot',
    9003: 'us_survey_foot',
    9030: 'nautical_mile',
    9036: 'kilometre',
}

# ModelType values
MODEL_TYPE_PROJECTED = 1
MODEL_TYPE_GEOGRAPHIC = 2
MODEL_TYPE_GEOCENTRIC = 3

# RasterType values
RASTER_PIXEL_IS_AREA = 1
RASTER_PIXEL_IS_POINT = 2


@dataclass
class GeoTransform:
    """Affine transform from pixel to geographic coordinates.

    For pixel (col, row):
        x = origin_x + col * pixel_width
        y = origin_y + row * pixel_height

    pixel_height is typically negative (y decreases downward).

    ``rotated_affine`` carries the original 6-tuple
    ``(pixel_width, b, origin_x, d, pixel_height, origin_y)`` (rasterio
    ``Affine`` ordering) for files opened with ``allow_rotated=True``.
    It is ``None`` for axis-aligned reads. Storing it as a real field
    means ``dataclasses.replace`` and similar helpers preserve it.
    """
    origin_x: float = 0.0
    origin_y: float = 0.0
    pixel_width: float = 1.0
    pixel_height: float = -1.0
    rotated_affine: tuple | None = None


@dataclass
class GeoInfo:
    """Geographic metadata extracted from GeoTIFF tags."""
    transform: GeoTransform = field(default_factory=GeoTransform)
    # True when ModelTransformation, ModelPixelScale, or ModelTiepoint tags
    # were present in the IFD. False for plain TIFFs with no GeoTIFF tags --
    # callers should fall back to integer pixel coordinates rather than
    # using the default transform (which would produce negative y values).
    has_georef: bool = False
    crs_epsg: int | None = None
    model_type: int = 0
    raster_type: int = RASTER_PIXEL_IS_AREA
    # int when GDAL_NODATA is a plain integer literal (so 64-bit sentinels
    # round-trip exactly), float for NaN / Inf / scientific notation /
    # fractional values, None when the tag is absent.  See issue #1847.
    nodata: int | float | None = None
    colormap: list | None = None  # list of (R, G, B, A) float tuples, or None
    x_resolution: float | None = None
    y_resolution: float | None = None
    resolution_unit: int | None = None  # 1=none, 2=inch, 3=cm
    # CRS description fields
    crs_name: str | None = None        # GTCitationGeoKey or ProjCitationGeoKey
    geog_citation: str | None = None   # e.g. "WGS 84", "NAD83"
    datum_code: int | None = None      # GeogGeodeticDatumGeoKey
    angular_units: str | None = None   # e.g. "degree"
    angular_units_code: int | None = None
    linear_units: str | None = None    # e.g. "metre"
    linear_units_code: int | None = None
    semi_major_axis: float | None = None
    inv_flattening: float | None = None
    projection_code: int | None = None
    # Vertical CRS
    vertical_epsg: int | None = None
    vertical_citation: str | None = None
    vertical_datum: int | None = None
    vertical_units: str | None = None
    vertical_units_code: int | None = None
    # WKT CRS string (resolved from EPSG via pyproj, or provided by caller)
    crs_wkt: str | None = None
    # GDAL metadata: dict of {name: value} for dataset-level items,
    # and {(name, band): value} for per-band items.  Raw XML also kept.
    gdal_metadata: dict | None = None
    gdal_metadata_xml: str | None = None
    # Extra TIFF tags not managed by the writer (pass-through on round-trip)
    # List of (tag_id, type_id, count, raw_value) tuples.
    extra_tags: list | None = None
    # ImageDescription tag (270) decoded as a Python str, when present.
    image_description: str | None = None
    # ExtraSamples tag (338) as a tuple of int alpha/extra-sample codes,
    # when present.
    extra_samples: tuple | None = None
    # Raw geokeys dict for anything else
    geokeys: dict[int, int | float | str] = field(default_factory=dict)


def _parse_gdal_metadata(xml_str: str) -> dict:
    """Parse GDALMetadata XML into a flat dict.

    Dataset-level items are stored as ``{name: value}``.
    Per-band items are stored as ``{(name, band_int): value}``.
    """
    import xml.etree.ElementTree as ET
    from ._safe_xml import safe_fromstring
    result = {}
    try:
        # GDALMetadata XML rides inside TIFF tag 42112; a crafted file
        # can carry a billion-laughs payload there, so refuse DOCTYPEs
        # before parsing. See issue #1579.
        root = safe_fromstring(xml_str)
        for item in root.findall('Item'):
            name = item.get('name', '')
            sample = item.get('sample')
            text = item.text or ''
            if sample is not None:
                result[(name, int(sample))] = text
            else:
                result[name] = text
    except (ET.ParseError, ValueError):
        # ValueError surfaces from safe_fromstring when the payload
        # carries a DOCTYPE. GDALMetadata is non-essential metadata, so
        # we silently drop it rather than failing the whole read --
        # matches the existing ParseError fallback.
        pass
    return result


def _build_gdal_metadata_xml(meta: dict) -> str:
    """Serialize a metadata dict back to GDALMetadata XML.

    Accepts the same dict format that _parse_gdal_metadata produces:
    string keys for dataset-level, (name, band) tuples for per-band.

    Every caller-supplied text and attribute slot is routed through
    ``xml.sax.saxutils.escape`` / ``quoteattr`` so a key or value
    containing XML special characters (``& < > " '``) cannot break
    the document or inject extra elements. Sample indices are emitted
    from an ``int(...)`` cast and need no escaping. See issue #1614.
    """
    from xml.sax.saxutils import escape as _xml_escape, quoteattr as _xml_quoteattr

    def _text(v) -> str:
        if v is None:
            return ""
        return _xml_escape(str(v), {'"': "&quot;", "'": "&apos;"})

    def _attr(v) -> str:
        if v is None:
            return '""'
        return _xml_quoteattr(str(v))

    lines = ['<GDALMetadata>']
    for key, value in meta.items():
        if isinstance(key, tuple):
            name, sample = key
            lines.append(
                f'  <Item name={_attr(name)} sample="{int(sample)}">'
                f'{_text(value)}</Item>')
        else:
            lines.append(f'  <Item name={_attr(key)}>{_text(value)}</Item>')
    lines.append('</GDALMetadata>')
    return '\n'.join(lines) + '\n'


def _epsg_to_wkt(epsg: int) -> str | None:
    """Resolve an EPSG code to a WKT string using pyproj.

    Returns None if pyproj is not installed or the code is unknown.

    Under ``XRSPATIAL_GEOTIFF_STRICT=1`` the underlying exception is
    re-raised instead of being swallowed. See issue #1662.
    """
    try:
        from pyproj import CRS
        return CRS.from_epsg(epsg).to_wkt()
    except Exception as e:
        import warnings
        from . import _geotiff_strict_mode, GeoTIFFFallbackWarning
        if _geotiff_strict_mode():
            raise
        warnings.warn(
            f"_epsg_to_wkt({epsg!r}) failed "
            f"({type(e).__name__}: {e}); returning None.",
            GeoTIFFFallbackWarning,
            stacklevel=2,
        )
        return None


# Top-level WKT 1 / WKT 2 keywords. GDAL stores user-defined CRSes
# (GeoKey *CSTypeGeoKey == 32767) as WKT in the citation, so callers
# of ``extract_geo_info`` need a cheap test to decide whether the
# citation string round-trips as a CRS or is just a free-form name.
# Listed in rough order of how often they appear in real-world files.
_WKT_PREFIXES = (
    'PROJCS[',
    'GEOGCS[',
    'PROJCRS[',
    'GEOGCRS[',
    'COMPD_CS[',
    'COMPOUNDCRS[',
    'BOUNDCRS[',
    'VERT_CS[',
    'VERTCRS[',
    'LOCAL_CS[',
    'ENGCRS[',
    'PARAMETRICCRS[',
    'TIMECRS[',
    'DERIVEDPROJCRS[',
)


def _looks_like_wkt(text) -> bool:
    """Return True iff ``text`` opens with a known WKT root keyword.

    The check is intentionally surface-level: only the first
    non-whitespace stretch is inspected, no parser is invoked. The
    callers only need to distinguish "this citation is actually a WKT
    string GDAL stuffed in here" from "this is a human-readable CRS
    name" -- a deeper parse would be both slower and a different bug
    surface. False positives on names that happen to start with a WKT
    keyword followed by '[' are vanishingly rare in practice.
    """
    if not isinstance(text, str):
        return False
    head = text.lstrip().upper()
    return any(head.startswith(p) for p in _WKT_PREFIXES)


def _synthesize_user_defined_wkt(
    model_type: int,
    semi_major: float | None,
    semi_minor: float | None,
    inv_flattening: float | None,
) -> str | None:
    """Build a canonical WKT for a user-defined geographic CRS.

    Triggered from :func:`extract_geo_info` when the file declares a
    user-defined geographic CRS (``GeographicTypeGeoKey == 32767`` and
    no projected EPSG) and the citation does not already carry a WKT
    string. The classic example is a GeoTIFF whose CRS lives only in
    ``GeogCitationGeoKey`` as a free-form name, with the ellipsoid
    exposed via separate GeoKeys (``GeogSemiMajorAxisGeoKey``,
    ``GeogSemiMinorAxisGeoKey``, ``GeogInvFlatteningGeoKey``).
    Rasterio reads such files and reports a GEOGCS CRS; pre-#1930
    xrspatial dropped the projection silently because nothing stamped
    a canonical ``attrs['crs_wkt']``. See issue #1930 (Phase 3
    ``crs_citation_only`` parity gap).

    Returns a WKT string when pyproj is installed and the GeoKeys
    expose enough of the ellipsoid to feed ``pyproj.CRS.from_dict``,
    otherwise ``None``. The synthesized CRS is name-stripped (PROJ
    drops the GEOGCS name on ``to_dict()`` anyway). The underlying
    ``GeoInfo.geog_citation`` field is consumed elsewhere in the
    reader, but contract v2 (issue #2016) stopped surfacing it on
    ``DataArray.attrs``; this helper exists to close the canonical-CRS
    parity gap, not to round-trip the citation field.

    Angular units are not threaded through. PROJ's ``longlat`` always
    emits degrees, and the corpus has no radian-unit user-defined
    fixture; if one shows up, this helper needs an angular_units_code
    parameter and the proj_dict needs a ``to_meter`` / units shim.

    Non-geographic model types fall through to ``None``:

    * ``MODEL_TYPE_PROJECTED`` (1) user-defined CRSes (``ProjectedCSType
      == 32767``) need the GeogPrime / Projection parameters that this
      helper does not yet read.
    * ``MODEL_TYPE_GEOCENTRIC`` (3) and unknown / zero model types are
      not exercised by the corpus and would need their own per-type
      proj_dict construction.

    In all three cases the caller falls through to the existing
    deprecated-attrs path.
    """
    if model_type != MODEL_TYPE_GEOGRAPHIC:
        return None
    if semi_major is None:
        return None
    # No ellipsoid: a degenerate file. Refuse to fabricate one rather
    # than emit a misleading WGS84 fallback.
    if semi_minor is None and inv_flattening is None:
        return None
    try:
        from pyproj import CRS
    except ImportError:
        return None

    proj_dict: dict[str, object] = {'proj': 'longlat', 'no_defs': True}
    proj_dict['a'] = float(semi_major)
    if inv_flattening is not None and float(inv_flattening) != 0.0:
        proj_dict['rf'] = float(inv_flattening)
    elif semi_minor is not None:
        proj_dict['b'] = float(semi_minor)
    else:
        # inv_flattening == 0 is the GeoTIFF convention for a sphere
        # (b == a). Encode it as such for PROJ.
        proj_dict['b'] = float(semi_major)

    try:
        crs = CRS.from_dict(proj_dict)
        return crs.to_wkt()
    except Exception as exc:
        import warnings
        from . import _geotiff_strict_mode, GeoTIFFFallbackWarning
        if _geotiff_strict_mode():
            raise
        warnings.warn(
            f"_synthesize_user_defined_wkt failed "
            f"({type(exc).__name__}: {exc}); returning None.",
            GeoTIFFFallbackWarning,
            stacklevel=2,
        )
        return None


def _parse_geokeys(ifd: IFD, data: bytes | memoryview,
                   byte_order: str) -> dict[int, int | float | str]:
    """Parse the GeoKeyDirectory and resolve values from param tags.

    The GeoKeyDirectoryTag (34735) contains a header:
        [key_directory_version, key_revision, minor_revision, num_keys]
    followed by num_keys entries of:
        [key_id, tiff_tag_location, count, value_offset]

    If tiff_tag_location == 0, value_offset is the value itself.
    If tiff_tag_location == 34736, look up in GeoDoubleParamsTag.
    If tiff_tag_location == 34737, look up in GeoAsciiParamsTag.
    """
    geokeys: dict[int, int | float | str] = {}

    dir_entry = ifd.entries.get(TAG_GEO_KEY_DIRECTORY)
    if dir_entry is None:
        return geokeys

    dir_values = dir_entry.value
    if isinstance(dir_values, int):
        return geokeys
    if not isinstance(dir_values, tuple):
        dir_values = (dir_values,)

    if len(dir_values) < 4:
        return geokeys

    num_keys = dir_values[3]

    # Get param tags
    double_params = ifd.get_value(TAG_GEO_DOUBLE_PARAMS)
    if double_params is not None:
        if not isinstance(double_params, tuple):
            double_params = (double_params,)
    else:
        double_params = ()

    ascii_params = ifd.get_value(TAG_GEO_ASCII_PARAMS)
    if ascii_params is None:
        ascii_params = ''
    if isinstance(ascii_params, bytes):
        ascii_params = ascii_params.decode('ascii', errors='replace')

    for i in range(num_keys):
        base = 4 + i * 4
        if base + 3 >= len(dir_values):
            break

        key_id = dir_values[base]
        tag_loc = dir_values[base + 1]
        count = dir_values[base + 2]
        value_offset = dir_values[base + 3]

        if tag_loc == 0:
            # Value is inline
            geokeys[key_id] = value_offset
        elif tag_loc == TAG_GEO_DOUBLE_PARAMS:
            # Value in double params
            if value_offset < len(double_params):
                if count == 1:
                    geokeys[key_id] = double_params[value_offset]
                else:
                    end = min(value_offset + count, len(double_params))
                    geokeys[key_id] = double_params[value_offset:end]
            else:
                geokeys[key_id] = 0.0
        elif tag_loc == TAG_GEO_ASCII_PARAMS:
            # Value in ASCII params
            end = value_offset + count
            val = ascii_params[value_offset:end].rstrip('|\x00')
            geokeys[key_id] = val
        else:
            geokeys[key_id] = value_offset

    return geokeys


# Default relative tolerance for the multi-tiepoint consistency check.
# Applied as ``rel_tol * max(|sx|, |sy|, 1.0)``, so the absolute threshold
# tracks pixel size: on a 10 m pixel file the threshold is 10 µm, on a
# 1° pixel file it is ~11 cm. Surveying / high-precision geodetic workflows
# that want to catch GCP files with smaller residuals can pass a tighter
# ``rel_tol`` to :func:`_validate_tiepoint_consistency` directly.
_TIEPOINT_CONSISTENCY_REL_TOL = 1e-6


def _validate_tiepoint_consistency(tiepoint: tuple,
                                   origin_x: float,
                                   origin_y: float,
                                   sx: float,
                                   sy: float,
                                   *,
                                   rel_tol: float = _TIEPOINT_CONSISTENCY_REL_TOL,
                                   scale_source: str = "ModelPixelScale") -> None:
    """Verify every ``ModelTiepointTag`` tuple agrees with the inferred affine.

    A ``ModelTiepointTag`` may carry one or many ``(I, J, K, X, Y, Z)``
    tuples. The single-tuple case (paired with ``ModelPixelScale``) is
    by far the most common and is unambiguous. Files with multiple
    tuples either repeat the same affine at every corner -- the tuples
    agree within tolerance and the reader can keep its single-tiepoint
    code path -- or carry a ground-control-point (GCP) set whose
    tuples do not agree, because the mapping from pixel to world is
    non-affine.

    Before this check landed, the GCP case silently fabricated an
    axis-aligned affine from the first tuple and downstream spatial
    ops trusted wrong coordinates. The reader now validates that every
    extra tuple is predicted by the inferred affine within a tolerance
    scaled to the pixel size; mismatches raise ``NotImplementedError``
    with a clear pointer at the GCP case so users know why their file
    is being rejected (issue #2117).

    Parameters
    ----------
    tiepoint : tuple
        Raw ``ModelTiepointTag`` value (length ``6 * N``).
    origin_x, origin_y : float
        World coords of pixel ``(0, 0)`` inferred from the first tuple.
    sx, sy : float
        Pixel size (``ModelPixelScaleTag`` magnitudes, both positive).
    rel_tol : float, optional
        Relative tolerance factor. The absolute threshold is
        ``rel_tol * max(|sx|, |sy|, 1.0)``. Defaults to
        :data:`_TIEPOINT_CONSISTENCY_REL_TOL`.
    scale_source : str, optional
        Where ``sx`` / ``sy`` came from. ``"ModelPixelScale"`` (default)
        names the scale tag in the GCP-case error. ``"unit fallback"``
        is used when ``ModelPixelScale`` was absent and the caller fell
        back to ``sx = sy = 1.0``; in that case a multi-tiepoint file is
        almost certainly malformed rather than a real GCP warp, and the
        error message says so.
    """
    n = len(tiepoint) // 6
    if n <= 1:
        return

    # Tolerance scales with pixel size so files in different units
    # (degrees vs metres) are treated consistently. The factor lives in
    # _TIEPOINT_CONSISTENCY_REL_TOL and is a relative residual on world
    # coordinates -- distinct from the 1e-12 absolute floor that the
    # rotation check in _extract_transform applies to raw
    # ModelTransformation matrix off-diagonals.
    tol = rel_tol * max(abs(sx), abs(sy), 1.0)

    for k in range(1, n):
        base = 6 * k
        tp_i = tiepoint[base + 0]
        tp_j = tiepoint[base + 1]
        tp_x = tiepoint[base + 3]
        tp_y = tiepoint[base + 4]

        # Sign convention: ``_extract_transform`` recovers the origin via
        # ``origin_y = tp_y + tp_j * sy`` because ``sy`` is a positive
        # magnitude and the raster's y decreases as row index increases.
        # Inverting that gives the predicted world y at row J below.
        predicted_x = origin_x + tp_i * sx
        predicted_y = origin_y - tp_j * sy

        dx = tp_x - predicted_x
        dy = tp_y - predicted_y
        if abs(dx) > tol or abs(dy) > tol:
            primary = (
                "ModelTiepointTag carries multiple non-affine tiepoints "
                f"(tuple {k} predicts world coords "
                f"({predicted_x!r}, {predicted_y!r}) but the file "
                f"declares ({tp_x!r}, {tp_y!r}); residual "
                f"({dx!r}, {dy!r}) exceeds tolerance {tol!r})."
            )
            if scale_source == "unit fallback":
                cause = (
                    "The file has multiple tiepoints but no "
                    "ModelPixelScale tag, so the reader cannot recover a "
                    "consistent affine. The file is most likely "
                    "malformed; if it is a real ground-control-point "
                    "warp, add a ModelPixelScale tag or rectify it first."
                )
            else:
                cause = (
                    "The file uses a ground-control-point warp that the "
                    "reader cannot represent as an axis-aligned affine."
                )
            hint = (
                "Rectify the file to a regular grid first (``gdalwarp``, "
                "``rasterio.warp.reproject``, or any GIS tool that "
                "resamples GCP files to an affine raster) and reopen "
                "the rectified output. See issue #2117."
            )
            raise NotImplementedError(f"{primary}\n{cause}\n{hint}")


def _extract_transform(ifd: IFD,
                       allow_rotated: bool = False
                       ) -> tuple[GeoTransform, bool]:
    """Extract affine transform from ModelTransformation, or
    ModelTiepoint + ModelPixelScale tags.

    Parameters
    ----------
    ifd : IFD
        Parsed IFD.
    allow_rotated : bool
        When True, a rotated / sheared / z-coupled ``ModelTransformationTag``
        does not raise. The function returns an *intentionally inert*
        identity ``GeoTransform`` (``origin_x=origin_y=0``,
        ``pixel_width=1``, ``pixel_height=-1``) with ``has_georef=False``
        so the reader treats the file as an ungeoreferenced pixel grid.
        Consumers MUST NOT read ``transform.origin_x`` /
        ``transform.pixel_width`` / etc. as a stand-in for the real
        mapping in this case -- those fields are the default identity,
        not a fallback. The rotated 6-tuple is attached to the returned
        ``GeoTransform`` via ``transform.rotated_affine`` (rasterio
        ``Affine`` ordering: ``(pixel_width, b, origin_x, d,
        pixel_height, origin_y)``); read it directly when the rotated
        mapping is needed. Default ``False`` -- existing behaviour,
        raise ``RotatedTransformError`` (issue #2267; previously
        ``NotImplementedError``).

        This contract is read-only. ``rotated_affine`` is not currently
        emitted by the writer. As of issue #2216 the writer refuses
        such inputs with a ``ValueError`` naming the attr unless the
        caller passes ``drop_rotation=True`` to accept the loss
        explicitly; the silent identity-affine round-trip the previous
        wording warned about is no longer reachable. If round-trip
        preservation matters, the writer needs a separate
        ``ModelTransformationTag`` emit path that consumes
        ``rotated_affine`` (see issue #2115 follow-up).

    Returns
    -------
    (transform, has_georef)
        ``has_georef`` is True when at least one of the geo-transform tags
        was present *and* axis-aligned. When False, ``transform`` is the
        default identity and callers should fall back to pixel coordinates.
        For the ``allow_rotated=True`` opt-in path, ``has_georef`` is
        False and the rotated 6-tuple lives on ``transform.rotated_affine``.
    """

    # Try ModelTransformationTag (4x4 row-major matrix, 16 doubles).
    # Per the GeoTIFF spec this tag wins over ModelPixelScale + ModelTiepoint
    # when present.
    #
    #   x = M[0]*col + M[1]*row + M[2]*z + M[3]
    #   y = M[4]*col + M[5]*row + M[6]*z + M[7]
    #
    # GeoTransform only carries the axis-aligned case.  For rotated, sheared,
    # or z-coupled transforms we raise ``RotatedTransformError`` unless the
    # caller opts out via ``allow_rotated`` (issues #2115, #2267). The opt-out
    # drops the georef so downstream coord generation uses pixel indices and
    # any spatial op that runs on the array sees no geo assumption to violate.
    # ``RotatedTransformError`` is the same typed error the VRT path raises
    # via ``_check_read_rotated_transform`` in ``_validation.py``, so both
    # entry points share one ``except`` contract.
    transform_tag = ifd.get_value(TAG_MODEL_TRANSFORMATION)
    if transform_tag is not None:
        if isinstance(transform_tag, tuple) and len(transform_tag) >= 12:
            m = transform_tag
            # Off-diagonal terms (rotation/skew) and z-coupling.  Use a small
            # tolerance scaled to the diagonal to absorb floating-point noise.
            scale = max(abs(m[0]), abs(m[5]), 1.0)
            tol = 1e-12 * scale
            rotation_terms = (m[1], m[4])
            z_terms = (m[2], m[6]) if len(m) >= 8 else (0.0, 0.0)
            if any(abs(t) > tol for t in rotation_terms + z_terms):
                if not allow_rotated:
                    raise RotatedTransformError(
                        "ModelTransformationTag (34264) contains rotation, "
                        "skew, or z-coupling terms "
                        f"(M[1]={m[1]!r}, M[4]={m[4]!r}, "
                        f"M[2]={m[2] if len(m) > 2 else 0.0!r}, "
                        f"M[6]={m[6] if len(m) > 6 else 0.0!r}). "
                        "Only axis-aligned affine transforms are supported. "
                        "Pass ``allow_rotated=True`` to read the pixel grid "
                        "without the geospatial assumption (issue #2115)."
                    )
                # Opt-in: drop georef, stash the rotated matrix on the
                # GeoTransform so the validator + attrs-roundtrip code
                # can see it. ``rasterio.Affine`` order: (a, b, c, d, e, f)
                # = (pixel_width, b, origin_x, d, pixel_height, origin_y).
                return GeoTransform(
                    rotated_affine=(m[0], m[1], m[3], m[4], m[5], m[7]),
                ), False
            return GeoTransform(
                origin_x=m[3],
                origin_y=m[7],
                pixel_width=m[0],
                pixel_height=m[5],
            ), True

    # Try ModelTiepoint + ModelPixelScale
    tiepoint = ifd.get_value(TAG_MODEL_TIEPOINT)
    scale = ifd.get_value(TAG_MODEL_PIXEL_SCALE)

    if scale is not None:
        if not isinstance(scale, tuple):
            scale = (scale,)

        sx = scale[0] if len(scale) > 0 else 1.0
        sy = scale[1] if len(scale) > 1 else 1.0

        if tiepoint is not None:
            if not isinstance(tiepoint, tuple):
                tiepoint = (tiepoint,)
            # tiepoint: (I, J, K, X, Y, Z); a ModelTiepointTag may carry
            # more than one (I, J, K, X, Y, Z) tuple. Files use that to
            # either repeat the same affine at every corner (the tuples
            # agree, common case) or to encode a GCP warp where the
            # tuples describe a non-affine mapping. Silently picking the
            # first tuple turns the GCP case into wrong coordinates
            # downstream (issue #2117). Validate that every tuple is
            # consistent with the inferred affine; fail closed otherwise.
            tp_i = tiepoint[0] if len(tiepoint) > 0 else 0.0
            tp_j = tiepoint[1] if len(tiepoint) > 1 else 0.0
            tp_x = tiepoint[3] if len(tiepoint) > 3 else 0.0
            tp_y = tiepoint[4] if len(tiepoint) > 4 else 0.0

            origin_x = tp_x - tp_i * sx
            origin_y = tp_y + tp_j * sy  # sy is positive, but y goes down

            _validate_tiepoint_consistency(
                tiepoint, origin_x, origin_y, sx, sy,
            )

            return GeoTransform(
                origin_x=origin_x,
                origin_y=origin_y,
                pixel_width=sx,
                pixel_height=-sy,  # negative because y decreases
            ), True

        return GeoTransform(pixel_width=sx, pixel_height=-sy), True

    # Tiepoint without scale: honour the tiepoint origin and fall back to
    # unit pixel size.  Per the GeoTIFF spec a ModelTiepointTag encodes a
    # real-world (X, Y) for pixel (I, J); dropping it would silently relocate
    # the raster to (0, 0).
    #
    # Sign convention note: ModelPixelScaleTag stores sy as a positive
    # magnitude (the spec does not encode sign because rows are always
    # ordered top-to-bottom in raster space).  The GeoTransform used in
    # this code stores pixel_height as -sy to reflect that y decreases as
    # row index increases.  When ModelPixelScaleTag is absent, the
    # documented unit-scale fallback is sy = 1.0 in spec terms, which
    # becomes pixel_height = -1.0 here.
    if tiepoint is not None:
        if not isinstance(tiepoint, tuple):
            tiepoint = (tiepoint,)
        tp_i = tiepoint[0] if len(tiepoint) > 0 else 0.0
        tp_j = tiepoint[1] if len(tiepoint) > 1 else 0.0
        tp_x = tiepoint[3] if len(tiepoint) > 3 else 0.0
        tp_y = tiepoint[4] if len(tiepoint) > 4 else 0.0

        # Unit scale: pixel_width = 1.0, pixel_height = -1.0
        origin_x = tp_x - tp_i * 1.0
        origin_y = tp_y + tp_j * 1.0

        # Same multi-tiepoint consistency check the scale branch above
        # runs; the absence of ModelPixelScale just means the scale is
        # the unit fallback. ``scale_source`` tells the helper to blame
        # the missing scale tag rather than the GCP-warp case in the
        # error message, since a multi-tiepoint file without
        # ModelPixelScale is almost certainly malformed (issue #2117).
        _validate_tiepoint_consistency(
            tiepoint, origin_x, origin_y, 1.0, 1.0,
            scale_source="unit fallback",
        )

        return GeoTransform(
            origin_x=origin_x,
            origin_y=origin_y,
            pixel_width=1.0,
            pixel_height=-1.0,
        ), True

    return GeoTransform(), False


def _parse_nodata_str(text: str | None) -> int | float | None:
    """Parse a GDAL_NODATA tag string at full integer precision when possible.

    Returns a Python ``int`` for plain integer literals (so 64-bit
    sentinels survive without the float64 round-trip that pushes them one
    ULP past the dtype max), a ``float`` for NaN / Inf / scientific
    notation / fractional values, and ``None`` when the string is not a
    valid number.

    Mirrors :func:`xrspatial.geotiff._vrt._parse_band_nodata` (issue
    #1833) which addressed the same problem on the VRT XML path. See
    issue #1847.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    # Try integer literal first so ``2**64 - 1`` / ``2**63 - 1`` /
    # ``-2**63`` round-trip exactly.  ``int()`` rejects floats like
    # ``"1.5e10"`` or ``"3.5"`` -- those fall through to the float
    # branch below.
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def extract_geo_info(ifd: IFD, data: bytes | memoryview,
                     byte_order: str,
                     *,
                     allow_rotated: bool = False) -> GeoInfo:
    """Extract full geographic metadata from a parsed IFD.

    Parameters
    ----------
    ifd : IFD
        Parsed IFD.
    data : bytes
        Full file data (needed for resolving GeoKey param offsets).
    byte_order : str
        '<' or '>'.
    allow_rotated : bool, optional
        Forwarded to :func:`_extract_transform`. When True, a rotated
        ``ModelTransformationTag`` is read as an ungeoreferenced pixel
        grid instead of raising ``RotatedTransformError`` (issue #2115,
        #2267).

    Returns
    -------
    GeoInfo
    """
    transform, has_georef = _extract_transform(ifd, allow_rotated=allow_rotated)
    geokeys = _parse_geokeys(ifd, data, byte_order)

    # Extract EPSG
    epsg = None
    if GEOKEY_PROJECTED_CS_TYPE in geokeys:
        val = geokeys[GEOKEY_PROJECTED_CS_TYPE]
        if isinstance(val, (int, float)) and val != 32767:
            epsg = int(val)
    if epsg is None and GEOKEY_GEOGRAPHIC_TYPE in geokeys:
        val = geokeys[GEOKEY_GEOGRAPHIC_TYPE]
        if isinstance(val, (int, float)) and val != 32767:
            epsg = int(val)

    model_type = geokeys.get(GEOKEY_MODEL_TYPE, 0)
    raster_type = geokeys.get(GEOKEY_RASTER_TYPE, RASTER_PIXEL_IS_AREA)

    # CRS name: prefer GTCitationGeoKey, fall back to ProjCitationGeoKey
    crs_name = geokeys.get(GEOKEY_CITATION)
    if crs_name is None:
        crs_name = geokeys.get(GEOKEY_PROJ_CITATION)
    if isinstance(crs_name, str):
        crs_name = crs_name.strip().rstrip('|')
    else:
        crs_name = None

    geog_citation = geokeys.get(GEOKEY_GEOG_CITATION)
    if isinstance(geog_citation, str):
        geog_citation = geog_citation.strip().rstrip('|')
    else:
        geog_citation = None

    datum_code = geokeys.get(GEOKEY_GEODETIC_DATUM)
    if isinstance(datum_code, (int, float)):
        datum_code = int(datum_code)
    else:
        datum_code = None

    # Angular units (geographic CRS)
    ang_code = geokeys.get(GEOKEY_GEOG_ANGULAR_UNITS)
    ang_name = None
    if isinstance(ang_code, (int, float)):
        ang_code = int(ang_code)
        ang_name = ANGULAR_UNITS.get(ang_code)
    else:
        ang_code = None

    # Linear units (projected CRS)
    lin_code = geokeys.get(GEOKEY_PROJ_LINEAR_UNITS)
    lin_name = None
    if isinstance(lin_code, (int, float)):
        lin_code = int(lin_code)
        lin_name = LINEAR_UNITS.get(lin_code)
    else:
        lin_code = None

    # Ellipsoid parameters
    semi_major = geokeys.get(GEOKEY_GEOG_SEMI_MAJOR_AXIS)
    if not isinstance(semi_major, (int, float)):
        semi_major = None
    # GeogSemiMinorAxisGeoKey. Kept local for WKT synthesis below; not
    # surfaced as a separate attr because the canonical ellipsoid lives
    # in attrs['crs_wkt'].
    semi_minor = geokeys.get(GEOKEY_GEOG_SEMI_MINOR_AXIS)
    if not isinstance(semi_minor, (int, float)):
        semi_minor = None
    inv_flat = geokeys.get(GEOKEY_GEOG_INV_FLATTENING)
    if not isinstance(inv_flat, (int, float)):
        inv_flat = None

    proj_code = geokeys.get(GEOKEY_PROJECTION)
    if isinstance(proj_code, (int, float)):
        proj_code = int(proj_code)
    else:
        proj_code = None

    # Vertical CRS
    vert_epsg = geokeys.get(GEOKEY_VERTICAL_CS_TYPE)
    if isinstance(vert_epsg, (int, float)) and vert_epsg != 32767:
        vert_epsg = int(vert_epsg)
    else:
        vert_epsg = None

    vert_citation = geokeys.get(GEOKEY_VERTICAL_CITATION)
    if isinstance(vert_citation, str):
        vert_citation = vert_citation.strip().rstrip('|')
    else:
        vert_citation = None

    vert_datum = geokeys.get(GEOKEY_VERTICAL_DATUM)
    if isinstance(vert_datum, (int, float)):
        vert_datum = int(vert_datum)
    else:
        vert_datum = None

    vert_units_code = geokeys.get(GEOKEY_VERTICAL_UNITS)
    vert_units_name = None
    if isinstance(vert_units_code, (int, float)):
        vert_units_code = int(vert_units_code)
        vert_units_name = LINEAR_UNITS.get(vert_units_code)
    else:
        vert_units_code = None

    # Extract nodata from GDAL_NODATA tag.
    #
    # Try ``int()`` first so 64-bit sentinels (``2**64 - 1`` for uint64,
    # ``2**63 - 1`` for int64) round-trip at full precision.  ``float()``
    # rounds those to the nearest representable float64, which sits one
    # ULP above the dtype's max and is then rejected by the downstream
    # ``info.min <= int(nodata) <= info.max`` gate -- the sentinel pixel
    # survives as a literal value rather than being masked to NaN.
    # Float parsing covers everything else: NaN / Inf / scientific
    # notation / fractional values.  Mirrors
    # :func:`xrspatial.geotiff._vrt._parse_band_nodata` (issue #1833)
    # which fixed the same class of bug on the VRT XML path.
    # See issue #1847.
    nodata = None
    nodata_str = ifd.nodata_str
    if nodata_str is not None:
        nodata = _parse_nodata_str(nodata_str)

    # Parse GDALMetadata XML (tag 42112)
    gdal_metadata = None
    gdal_metadata_xml = ifd.gdal_metadata
    if gdal_metadata_xml is not None:
        gdal_metadata = _parse_gdal_metadata(gdal_metadata_xml)

    # Extract palette colormap (Photometric=3, tag 320)
    colormap = None
    if ifd.photometric == 3:
        raw_cmap = ifd.colormap
        if raw_cmap is not None:
            bps_val = resolve_bits_per_sample(ifd.bits_per_sample)
            n_colors = 1 << bps_val  # 2^BitsPerSample
            # TIFF ColorMap: 3 * n_colors uint16 values
            # Layout: [R0..R_{n-1}, G0..G_{n-1}, B0..B_{n-1}]
            # Values are 0-65535, scale to 0.0-1.0 for matplotlib
            if len(raw_cmap) >= 3 * n_colors:
                colormap = []
                for i in range(n_colors):
                    r = raw_cmap[i] / 65535.0
                    g = raw_cmap[n_colors + i] / 65535.0
                    b = raw_cmap[2 * n_colors + i] / 65535.0
                    colormap.append((r, g, b, 1.0))

    # Collect extra (non-managed) tags for pass-through
    extra_tags = []
    image_description = None
    extra_samples = None
    for tag_id, entry in ifd.entries.items():
        if tag_id in _MANAGED_TAGS:
            continue
        extra_tags.append((tag_id, entry.type_id, entry.count, entry.value))
        # Surface a few well-known extras as friendly attrs while still
        # carrying the raw entry in extra_tags so to_geotiff can rewrite
        # it byte-for-byte.
        if tag_id == TAG_IMAGE_DESCRIPTION:
            v = entry.value
            if isinstance(v, bytes):
                v = v.rstrip(b'\x00').decode('ascii', errors='replace')
            elif isinstance(v, str):
                v = v.rstrip('\x00')
            image_description = v
        elif tag_id == TAG_EXTRA_SAMPLES:
            v = entry.value
            if isinstance(v, tuple):
                extra_samples = tuple(int(x) for x in v)
            elif isinstance(v, int):
                extra_samples = (int(v),)
    if not extra_tags:
        extra_tags = None

    # Resolve EPSG -> WKT via pyproj if available
    crs_wkt = None
    if epsg is not None:
        crs_wkt = _epsg_to_wkt(epsg)
    elif _looks_like_wkt(crs_name):
        # User-defined CRS: GeoKey GEOKEY_*_CS_TYPE == 32767 and the WKT
        # lives in the citation. Expose it as crs_wkt so the writer can
        # round-trip it. The citation itself stays in crs_name for callers
        # that expect that key. Without this branch a read -> write cycle
        # silently drops the projection on user-defined CRS files because
        # to_geotiff only consults attrs['crs'] / attrs['crs_wkt']. See
        # issue #1632.
        crs_wkt = crs_name
    else:
        # Citation-only user-defined geographic CRS: no EPSG, no WKT
        # in the citation, but enough ellipsoid / units GeoKeys to
        # synthesize a canonical WKT (issue #1930 Phase 3). Without
        # this branch ``attrs['crs_wkt']`` stays None and the
        # golden-corpus oracle reports a CRS parity gap.
        synth = _synthesize_user_defined_wkt(
            model_type=(
                int(model_type)
                if isinstance(model_type, (int, float)) else 0
            ),
            semi_major=semi_major,
            semi_minor=semi_minor,
            inv_flattening=inv_flat,
        )
        if synth is not None:
            crs_wkt = synth

    return GeoInfo(
        transform=transform,
        has_georef=has_georef,
        crs_epsg=epsg,
        model_type=int(model_type) if isinstance(model_type, (int, float)) else 0,
        raster_type=int(raster_type) if isinstance(raster_type, (int, float)) else RASTER_PIXEL_IS_AREA,
        nodata=nodata,
        colormap=colormap,
        x_resolution=ifd.x_resolution,
        y_resolution=ifd.y_resolution,
        resolution_unit=ifd.resolution_unit,
        crs_name=crs_name,
        geog_citation=geog_citation,
        datum_code=datum_code,
        angular_units=ang_name,
        angular_units_code=ang_code,
        linear_units=lin_name,
        linear_units_code=lin_code,
        semi_major_axis=float(semi_major) if semi_major is not None else None,
        inv_flattening=float(inv_flat) if inv_flat is not None else None,
        projection_code=proj_code,
        vertical_epsg=vert_epsg,
        vertical_citation=vert_citation,
        vertical_datum=vert_datum,
        vertical_units=vert_units_name,
        vertical_units_code=vert_units_code,
        crs_wkt=crs_wkt,
        gdal_metadata=gdal_metadata,
        gdal_metadata_xml=gdal_metadata_xml,
        extra_tags=extra_tags,
        image_description=image_description,
        extra_samples=extra_samples,
        geokeys=geokeys,
    )


def extract_geo_info_with_overview_inheritance(
    ifd: IFD,
    ifds: list,
    data: bytes | memoryview,
    byte_order: str,
    *,
    allow_rotated: bool = False,
) -> GeoInfo:
    """Extract geo metadata, inheriting from level 0 when the IFD lacks it.

    Wraps :func:`extract_geo_info` for overview reads. GDAL-style COG
    writers (including this package's :func:`to_geotiff`) put a handful
    of tags only on the level-0 IFD:

    * GeoKeyDirectory, ModelPixelScale, ModelTiepoint (georef)
    * GDAL_NODATA, GDAL_METADATA (per-IFD pass-through tags)
    * XResolution, YResolution, ResolutionUnit (resolution tags)
    * ColorMap, ImageDescription, ExtraSamples (extra-tag pass-through)

    Calling ``extract_geo_info`` directly on an overview IFD therefore
    returns a default :class:`GeoTransform` with ``has_georef=False``,
    no CRS, and a ``nodata=None`` field, so overview reads silently
    lose their georeferencing and their nodata sentinel.

    When ``ifd`` is a reduced-resolution overview (NewSubfileType bit 0
    set), we re-run ``extract_geo_info`` on the first full-resolution
    IFD (NewSubfileType bit 0 clear, bit 2 clear). Per-IFD pass-through
    tags (nodata, GDAL metadata, resolution, colormap, extra tags,
    image description, extra samples) are inherited when the overview
    lacks its own value, regardless of whether the overview has its own
    georef. The transform and CRS-side fields are additionally
    inherited when the overview lacks its own georef, with the pixel
    size rescaled by ``width_full / width_overview`` so coords cover
    the same extent as level 0.

    If the overview IFD already carries its own value for a given
    field, that value wins -- inheritance is per-field and only fills
    in missing entries. If no full-resolution sibling exists, the
    overview's own (possibly empty) info is returned -- callers get the
    same fallback behaviour they used to.

    Inheriting nodata + the rich-tag set fixes #1739 (silent numerical
    corruption when reading COG overview pixels because attrs['nodata']
    was lost). The georef inheritance is the original fix from #1640.

    Parameters
    ----------
    ifd : IFD
        The IFD selected by ``select_overview_ifd``.
    ifds : list of IFD
        All IFDs parsed from the file. Used to locate the level-0
        sibling for georef inheritance.
    data : bytes or memoryview
        Full file bytes (forwarded to ``extract_geo_info``).
    byte_order : str
        ``'<'`` or ``'>'`` (forwarded to ``extract_geo_info``).

    Returns
    -------
    GeoInfo
    """
    info = extract_geo_info(ifd, data, byte_order,
                            allow_rotated=allow_rotated)

    # Overview IFDs have NewSubfileType bit 0 set; mask IFDs (bit 2) and
    # page IFDs (bit 1) are filtered out by ``select_overview_ifd``
    # before reaching here, so we never inherit a mask's geo info.
    is_overview = bool(ifd.subfile_type & 1)
    if not is_overview:
        return info

    # Find the level-0 IFD: NewSubfileType has bit 0 clear (not an
    # overview) and bit 2 clear (not a transparency mask). This is the
    # same criterion ``select_overview_ifd``'s filter uses to identify
    # the full-resolution pyramid root.
    base_ifd = None
    for cand in ifds:
        if cand is ifd:
            continue
        st = cand.subfile_type
        if (st & 1) == 0 and (st & 4) == 0:
            base_ifd = cand
            break
    if base_ifd is None:
        return info

    base_info = extract_geo_info(base_ifd, data, byte_order,
                                 allow_rotated=allow_rotated)

    # Inherit the per-IFD metadata that the COG writer emits only on the
    # level-0 IFD: GDAL_NODATA, GDAL_METADATA, x/y resolution, colormap,
    # extra tags, image description, extra samples. Without this block
    # an overview read silently drops attrs['nodata'] (so the sentinel
    # pixels the writer baked into the overview survive as ordinary data
    # and poison downstream stats) and attrs['gdal_metadata'] (user
    # metadata loss). See issue #1739.
    #
    # Each field is inherited only when the overview lacks its own
    # value, so an overview IFD that does re-declare any of these keeps
    # its own copy. Mirrors the gate the CRS-side inheritance applies
    # below: prefer the overview's own value when present.
    if info.nodata is None and base_info.nodata is not None:
        info.nodata = base_info.nodata
    if (info.gdal_metadata is None
            and base_info.gdal_metadata is not None):
        info.gdal_metadata = base_info.gdal_metadata
    if (info.gdal_metadata_xml is None
            and base_info.gdal_metadata_xml is not None):
        info.gdal_metadata_xml = base_info.gdal_metadata_xml
    if info.x_resolution is None and base_info.x_resolution is not None:
        info.x_resolution = base_info.x_resolution
    if info.y_resolution is None and base_info.y_resolution is not None:
        info.y_resolution = base_info.y_resolution
    if (info.resolution_unit is None
            and base_info.resolution_unit is not None):
        info.resolution_unit = base_info.resolution_unit
    if info.colormap is None and base_info.colormap is not None:
        info.colormap = base_info.colormap
    if info.extra_tags is None and base_info.extra_tags is not None:
        info.extra_tags = base_info.extra_tags
    if (info.image_description is None
            and base_info.image_description is not None):
        info.image_description = base_info.image_description
    if (info.extra_samples is None
            and base_info.extra_samples is not None):
        info.extra_samples = base_info.extra_samples

    # If the overview already has its own georef, the rest of the
    # inheritance (transform + CRS-side fields) is unnecessary -- return
    # now with just the per-IFD-tag inheritance applied above.
    if info.has_georef:
        return info

    if not base_info.has_georef:
        return info

    # Rescale the pixel size by the integer reduction factor. Width and
    # height ratios should match for power-of-two overview pyramids;
    # average them so a slightly off-by-one edge size still produces a
    # sensible transform.
    base_w = base_ifd.width
    base_h = base_ifd.height
    ov_w = ifd.width
    ov_h = ifd.height
    if base_w <= 0 or base_h <= 0 or ov_w <= 0 or ov_h <= 0:
        return info
    scale_x = base_w / ov_w
    scale_y = base_h / ov_h

    base_t = base_info.transform

    # Decide which raster_type the inherited info should carry, since the
    # origin shift below depends on it. The overview IFD usually does not
    # re-declare GeoKey 1025; when its own value is the default
    # ``PixelIsArea`` (1), prefer the parent's setting (typically
    # ``PixelIsPoint`` is only declared once at level 0). When the overview
    # explicitly declared a different value, keep it.
    if info.raster_type == RASTER_PIXEL_IS_AREA:
        effective_raster_type = base_info.raster_type
    else:
        effective_raster_type = info.raster_type

    # ``origin_x`` / ``origin_y`` semantics depend on raster_type
    # (GeoKey 1025):
    #
    # * ``PixelIsArea`` (the default): origin is the upper-left corner
    #   of pixel (0, 0). An overview pixel covering the first
    #   ``scale_x`` columns of level 0 has its upper-left corner in
    #   exactly the same place as level 0's, so we keep the origin
    #   unchanged.
    # * ``PixelIsPoint`` (common for DEMs, GeoKey 1025 = 2): origin is
    #   the *center* of pixel (0, 0). The overview pixel-0 center sits
    #   at the centroid of the ``scale_x`` x ``scale_y`` level-0 pixels
    #   it covers, which is
    #   ``origin + (scale - 1) * 0.5 * pixel_size_lvl0`` along each
    #   axis (issue #1642).
    if effective_raster_type == RASTER_PIXEL_IS_POINT:
        origin_shift_x = (scale_x - 1.0) * 0.5 * base_t.pixel_width
        origin_shift_y = (scale_y - 1.0) * 0.5 * base_t.pixel_height
    else:
        origin_shift_x = 0.0
        origin_shift_y = 0.0

    info.transform = GeoTransform(
        origin_x=base_t.origin_x + origin_shift_x,
        origin_y=base_t.origin_y + origin_shift_y,
        pixel_width=base_t.pixel_width * scale_x,
        pixel_height=base_t.pixel_height * scale_y,
    )
    info.has_georef = True

    # Inherit CRS-side metadata from the parent. The overview IFD may
    # carry its own raster_type (rare) -- prefer the overview's value
    # when it differs from the default, otherwise inherit.
    info.crs_epsg = base_info.crs_epsg
    info.crs_wkt = base_info.crs_wkt
    info.crs_name = base_info.crs_name
    info.geog_citation = base_info.geog_citation
    info.datum_code = base_info.datum_code
    info.angular_units = base_info.angular_units
    info.angular_units_code = base_info.angular_units_code
    info.linear_units = base_info.linear_units
    info.linear_units_code = base_info.linear_units_code
    info.semi_major_axis = base_info.semi_major_axis
    info.inv_flattening = base_info.inv_flattening
    info.projection_code = base_info.projection_code
    info.vertical_epsg = base_info.vertical_epsg
    info.vertical_citation = base_info.vertical_citation
    info.vertical_datum = base_info.vertical_datum
    info.vertical_units = base_info.vertical_units
    info.vertical_units_code = base_info.vertical_units_code
    info.model_type = base_info.model_type
    # Use the raster_type we already chose above so the value we wrote
    # into the transform stays consistent with the value attached to
    # the GeoInfo (PixelIsPoint is almost never re-declared on overview
    # IFDs, so this normally inherits from the parent).
    info.raster_type = effective_raster_type

    return info


def _model_type_from_wkt(wkt: str) -> int:
    """Guess ModelType from a WKT string prefix."""
    upper = wkt.strip().upper()
    if upper.startswith(('GEOGCS', 'GEOGCRS')):
        return MODEL_TYPE_GEOGRAPHIC
    return MODEL_TYPE_PROJECTED


# Hard-coded fallback set of EPSG codes known to be geographic. Used only
# when pyproj is unavailable. Kept intentionally tight: the historic
# 4000-4999 block plus EPSG 4326 itself, which is the case the legacy
# range heuristic got right. Anything outside this set without pyproj is
# treated as "unknown" and raises -- silent CRS corruption is worse than
# an explicit error.
#
# The range covers the EPSG "geographic CRS" allocation that predates the
# more recent realisations (NAD83(2011) = 6318, GDA2020 = 7844,
# WGS 84 (G2139) = 9057, etc.) which live outside 4000-4999 and need
# pyproj to classify correctly. See issue #2277.
_KNOWN_GEOGRAPHIC_EPSG_FALLBACK = frozenset(range(4000, 5000))


def _model_type_from_epsg(crs_epsg: int) -> int:
    """Return the GeoTIFF ModelType (geographic vs projected) for an EPSG.

    Prefers ``pyproj.CRS.from_epsg(crs_epsg).is_geographic`` so the
    decision matches the actual EPSG registry. Falls back to a tight
    hard-coded allowlist (EPSG 4326 plus the 4000-4999 block) when
    pyproj isn't installed. Anything outside that fallback set raises
    :class:`UnknownCRSModelTypeError` rather than guessing -- the legacy range
    heuristic at this site silently mis-tagged geographic codes like
    6318 (NAD83(2011)) and 7844 (GDA2020) as projected.

    See issue #2277.
    """
    try:
        from pyproj import CRS
    except ImportError:
        CRS = None  # type: ignore[assignment]

    if CRS is not None:
        try:
            crs = CRS.from_epsg(crs_epsg)
        except Exception as e:
            # pyproj is installed but the code is unknown or malformed.
            # Fall back to the hard-coded allowlist before giving up so
            # an offline pyproj-database mismatch doesn't break a code
            # that the legacy heuristic handled correctly.
            if crs_epsg in _KNOWN_GEOGRAPHIC_EPSG_FALLBACK:
                return MODEL_TYPE_GEOGRAPHIC
            raise UnknownCRSModelTypeError(
                f"Cannot determine GeoTIFF model type for EPSG:{crs_epsg}: "
                f"pyproj.CRS.from_epsg failed "
                f"({type(e).__name__}: {e}). Refusing to guess; passing a "
                "known EPSG or installing a current pyproj database will "
                "resolve this."
            ) from e
        if crs.is_geographic:
            return MODEL_TYPE_GEOGRAPHIC
        return MODEL_TYPE_PROJECTED

    # pyproj unavailable. Use the tight fallback set; raise otherwise.
    if crs_epsg in _KNOWN_GEOGRAPHIC_EPSG_FALLBACK:
        return MODEL_TYPE_GEOGRAPHIC
    raise UnknownCRSModelTypeError(
        f"Cannot determine GeoTIFF model type for EPSG:{crs_epsg} without "
        "pyproj. Install pyproj so the writer can consult "
        "CRS.from_epsg(...).is_geographic, or pass an EPSG in the "
        "hard-coded geographic fallback set (4000-4999). Refusing to "
        "guess: the legacy range heuristic silently mis-tagged codes "
        "like 6318 and 7844 as projected. See issue #2277."
    )


def build_geo_tags(transform: GeoTransform, crs_epsg: int | None = None,
                   nodata=None,
                   raster_type: int = RASTER_PIXEL_IS_AREA,
                   crs_wkt: str | None = None) -> dict[int, tuple]:
    """Build GeoTIFF IFD tag entries for writing.

    Parameters
    ----------
    transform : GeoTransform
        Pixel-to-coordinate mapping.
    crs_epsg : int or None
        EPSG code for the CRS.
    nodata : float, int, or None
        NoData value.
    raster_type : int
        RASTER_PIXEL_IS_AREA (1) or RASTER_PIXEL_IS_POINT (2).
    crs_wkt : str or None
        WKT or PROJ string for the CRS.  Used only when *crs_epsg* is
        None so that custom (non-EPSG) coordinate systems survive
        round-trips.  Stored in the GeoAsciiParamsTag and referenced
        from GTCitationGeoKey.

    Notes
    -----
    When ``crs_wkt`` is supplied without ``crs_epsg``, the writer emits
    a *user-defined* CRS: ``ProjectedCSType`` or ``GeographicType`` is
    set to ``32767`` and the raw WKT is stored in ``GeoAsciiParams``,
    referenced from ``GTCitationGeoKey``. No richer projection-parameter
    GeoKeys (``ProjLinearUnitsGeoKey``, ``GeogAngularUnitsGeoKey``,
    ellipsoid params, projection method, etc.) are written. libgeotiff
    and GDAL recover the CRS by parsing the citation, but most other
    GeoTIFF readers treat the citation as a free-form name and lose the
    CRS. Prefer ``crs_epsg`` when the projection is registered with an
    EPSG code -- the EPSG path emits the standard GeoKeys every reader
    understands. A ``UserWarning`` is emitted on the WKT-only path so
    the limitation is visible at call time. See issue #1768.

    Returns
    -------
    dict mapping tag ID to (type_id, count, value_bytes) tuples,
    where value_bytes is already serialized for little-endian output.
    """
    tags = {}

    # Standard north-up rasters have pixel_width > 0 and pixel_height < 0.
    # Anything else (descending x, ascending y) cannot be expressed via
    # ModelPixelScale + ModelTiepoint because the spec requires the scales
    # to be positive.
    north_up = transform.pixel_width > 0 and transform.pixel_height < 0

    if north_up:
        # ModelPixelScaleTag (33550): (ScaleX, ScaleY, ScaleZ)
        sx = transform.pixel_width
        sy = -transform.pixel_height
        tags[TAG_MODEL_PIXEL_SCALE] = (sx, sy, 0.0)

        # ModelTiepointTag (33922): (I, J, K, X, Y, Z)
        tags[TAG_MODEL_TIEPOINT] = (
            0.0, 0.0, 0.0,
            transform.origin_x, transform.origin_y, 0.0,
        )
    else:
        # Why: GeoTIFF spec requires ModelPixelScale entries to be positive,
        # so non-standard orientations (descending x, ascending y) must use
        # ModelTransformationTag (34264) instead.  The 4x4 row-major matrix
        # maps (col, row, 0, 1) -> (X, Y, 0, 1).
        tags[TAG_MODEL_TRANSFORMATION] = (
            transform.pixel_width, 0.0, 0.0, transform.origin_x,
            0.0, transform.pixel_height, 0.0, transform.origin_y,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )

    # GeoKeyDirectoryTag (34735)
    geokeys = []
    # Header: version=1, revision=1, minor=0
    num_keys = 1  # at least RasterType
    key_entries = []

    # Collect ASCII params strings (pipe-delimited in GeoAsciiParamsTag)
    ascii_parts = []
    ascii_offset = 0

    # ModelType
    if crs_epsg is not None:
        # Resolve the GeoTIFF ModelType via pyproj when available; raise
        # on unknown codes when pyproj is missing rather than guessing.
        # See issue #2277 -- the historic EPSG-range heuristic silently
        # mis-tagged geographic codes outside 4000-4999 (e.g. 6318,
        # 7844, 9057) as projected, corrupting the CRS at write time.
        model_type = _model_type_from_epsg(crs_epsg)
        key_entries.append((GEOKEY_MODEL_TYPE, 0, 1, model_type))
        num_keys += 1
    elif crs_wkt is not None:
        model_type = _model_type_from_wkt(crs_wkt)
        key_entries.append((GEOKEY_MODEL_TYPE, 0, 1, model_type))
        num_keys += 1

    # RasterType
    key_entries.append((GEOKEY_RASTER_TYPE, 0, 1, raster_type))

    # CRS
    if crs_epsg is not None:
        if model_type == MODEL_TYPE_GEOGRAPHIC:
            key_entries.append((GEOKEY_GEOGRAPHIC_TYPE, 0, 1, crs_epsg))
        else:
            key_entries.append((GEOKEY_PROJECTED_CS_TYPE, 0, 1, crs_epsg))
        num_keys += 1
    elif crs_wkt is not None:
        # User-defined CRS: store 32767 and write the citation string to
        # GeoAsciiParams. The ``crs_wkt`` parameter holds whatever string
        # the caller supplied (typically WKT, but also accepts PROJ
        # strings that ``_wkt_to_epsg`` could not resolve to an EPSG
        # code); the bytes are written verbatim.
        #
        # libgeotiff and GDAL recover the CRS by parsing the citation
        # back out, but many other GeoTIFF readers treat the citation as
        # a free-form name and silently drop the projection. Warn the
        # caller so the interop limitation is visible at write time.
        # Python's default warning filter dedupes per call site, so the
        # warning fires once per location rather than once per pixel.
        # See issue #1768.
        import warnings as _warnings
        _warnings.warn(
            "Writing a user-defined CRS via WKT only "
            "(ProjectedCSType / GeographicType = 32767 with the supplied "
            "CRS string -- WKT or unresolved PROJ -- stored in "
            "GTCitationGeoKey). libgeotiff and GDAL can round-trip this, "
            "but many other GeoTIFF readers treat the citation as a "
            "free-form name and lose the CRS. Prefer passing an EPSG "
            "code (e.g. attrs['crs'] = 4326) when the projection is "
            "registered with EPSG -- the EPSG path emits the standard "
            "GeoKeys every reader understands.",
            UserWarning,
            stacklevel=2,
        )
        if model_type == MODEL_TYPE_GEOGRAPHIC:
            key_entries.append((GEOKEY_GEOGRAPHIC_TYPE, 0, 1, 32767))
        else:
            key_entries.append((GEOKEY_PROJECTED_CS_TYPE, 0, 1, 32767))
        num_keys += 1
        # GTCitationGeoKey -> GeoAsciiParams
        wkt_with_pipe = crs_wkt + '|'
        key_entries.append((
            GEOKEY_CITATION, TAG_GEO_ASCII_PARAMS,
            len(wkt_with_pipe), ascii_offset,
        ))
        ascii_parts.append(wkt_with_pipe)
        ascii_offset += len(wkt_with_pipe)
        num_keys += 1

    num_keys = len(key_entries)
    header = [1, 1, 0, num_keys]
    flat = header.copy()
    for entry in key_entries:
        flat.extend(entry)

    tags[TAG_GEO_KEY_DIRECTORY] = tuple(flat)

    # GeoAsciiParamsTag (34737)
    if ascii_parts:
        tags[TAG_GEO_ASCII_PARAMS] = ''.join(ascii_parts)

    # GDAL_NODATA
    if nodata is not None:
        # Belt-and-braces guard against bool / np.bool_ sentinels. The
        # writer entry point (``to_geotiff``) already rejects these, but
        # ``build_geo_tags`` is called from a few other code paths and
        # ``str(True) == 'True'`` produces a non-numeric GDAL_NODATA tag
        # that readers silently drop. See issue #1911.
        import numpy as _np
        if isinstance(nodata, (bool, _np.bool_)):
            raise TypeError(
                f"nodata must be numeric (int or float), got {nodata!r}")
        tags[TAG_GDAL_NODATA] = str(nodata)

    return tags
