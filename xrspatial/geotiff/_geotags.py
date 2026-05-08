"""GeoTIFF tag interpretation: CRS, affine transform, GeoKeys."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ._header import (
    IFD,
    TAG_IMAGE_WIDTH, TAG_IMAGE_LENGTH, TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION, TAG_PHOTOMETRIC,
    TAG_STRIP_OFFSETS, TAG_ORIENTATION, TAG_SAMPLES_PER_PIXEL,
    TAG_ROWS_PER_STRIP, TAG_STRIP_BYTE_COUNTS,
    TAG_X_RESOLUTION, TAG_Y_RESOLUTION,
    TAG_PLANAR_CONFIG, TAG_RESOLUTION_UNIT,
    TAG_PREDICTOR, TAG_COLORMAP,
    TAG_TILE_WIDTH, TAG_TILE_LENGTH,
    TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS,
    TAG_EXTRA_SAMPLES,
    TAG_SAMPLE_FORMAT, TAG_GDAL_METADATA, TAG_GDAL_NODATA,
    TAG_MODEL_PIXEL_SCALE, TAG_MODEL_TIEPOINT,
    TAG_MODEL_TRANSFORMATION,
    TAG_GEO_KEY_DIRECTORY, TAG_GEO_DOUBLE_PARAMS, TAG_GEO_ASCII_PARAMS,
)

# ImageDescription tag (270). Captured for round-trip but not managed
# by the writer -- it flows through extra_tags pass-through.
TAG_IMAGE_DESCRIPTION = 270

# Tags the writer manages directly. Tags not in this set are collected
# into GeoInfo.extra_tags on read and re-emitted on write via the
# extra_tags pass-through. ColorMap (320), ExtraSamples (338, only emitted
# automatically when samples > 1), and ImageDescription (270) intentionally
# stay OUT of this set so they round-trip without dedicated writer plumbing.
_MANAGED_TAGS = frozenset({
    TAG_IMAGE_WIDTH, TAG_IMAGE_LENGTH, TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION, TAG_PHOTOMETRIC,
    TAG_STRIP_OFFSETS, TAG_ORIENTATION, TAG_SAMPLES_PER_PIXEL,
    TAG_ROWS_PER_STRIP, TAG_STRIP_BYTE_COUNTS,
    TAG_X_RESOLUTION, TAG_Y_RESOLUTION,
    TAG_PLANAR_CONFIG, TAG_RESOLUTION_UNIT,
    TAG_PREDICTOR,
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
    """
    origin_x: float = 0.0
    origin_y: float = 0.0
    pixel_width: float = 1.0
    pixel_height: float = -1.0


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
    nodata: float | None = None
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
    result = {}
    try:
        root = ET.fromstring(xml_str)
        for item in root.findall('Item'):
            name = item.get('name', '')
            sample = item.get('sample')
            text = item.text or ''
            if sample is not None:
                result[(name, int(sample))] = text
            else:
                result[name] = text
    except ET.ParseError:
        pass
    return result


def _build_gdal_metadata_xml(meta: dict) -> str:
    """Serialize a metadata dict back to GDALMetadata XML.

    Accepts the same dict format that _parse_gdal_metadata produces:
    string keys for dataset-level, (name, band) tuples for per-band.
    """
    lines = ['<GDALMetadata>']
    for key, value in meta.items():
        if isinstance(key, tuple):
            name, sample = key
            lines.append(
                f'  <Item name="{name}" sample="{sample}">{value}</Item>')
        else:
            lines.append(f'  <Item name="{key}">{value}</Item>')
    lines.append('</GDALMetadata>')
    return '\n'.join(lines) + '\n'


def _epsg_to_wkt(epsg: int) -> str | None:
    """Resolve an EPSG code to a WKT string using pyproj.

    Returns None if pyproj is not installed or the code is unknown.
    """
    try:
        from pyproj import CRS
        return CRS.from_epsg(epsg).to_wkt()
    except Exception:
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


def _extract_transform(ifd: IFD) -> tuple[GeoTransform, bool]:
    """Extract affine transform from ModelTransformation, or
    ModelTiepoint + ModelPixelScale tags.

    Returns
    -------
    (transform, has_georef)
        ``has_georef`` is True when at least one of the geo-transform tags
        was present.  When False, ``transform`` is the default identity
        and callers should fall back to pixel coordinates.
    """

    # Try ModelTransformationTag (4x4 row-major matrix, 16 doubles).
    # Per the GeoTIFF spec this tag wins over ModelPixelScale + ModelTiepoint
    # when present.
    #
    #   x = M[0]*col + M[1]*row + M[2]*z + M[3]
    #   y = M[4]*col + M[5]*row + M[6]*z + M[7]
    #
    # GeoTransform only carries the axis-aligned case.  For rotated, sheared,
    # or z-coupled transforms we raise NotImplementedError instead of silently
    # dropping the off-diagonal terms.
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
                raise NotImplementedError(
                    "ModelTransformationTag (34264) contains rotation, "
                    "skew, or z-coupling terms "
                    f"(M[1]={m[1]!r}, M[4]={m[4]!r}, "
                    f"M[2]={m[2] if len(m) > 2 else 0.0!r}, "
                    f"M[6]={m[6] if len(m) > 6 else 0.0!r}). "
                    "Only axis-aligned affine transforms are supported."
                )
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
            # tiepoint: (I, J, K, X, Y, Z)
            tp_i = tiepoint[0] if len(tiepoint) > 0 else 0.0
            tp_j = tiepoint[1] if len(tiepoint) > 1 else 0.0
            tp_x = tiepoint[3] if len(tiepoint) > 3 else 0.0
            tp_y = tiepoint[4] if len(tiepoint) > 4 else 0.0

            origin_x = tp_x - tp_i * sx
            origin_y = tp_y + tp_j * sy  # sy is positive, but y goes down

            return GeoTransform(
                origin_x=origin_x,
                origin_y=origin_y,
                pixel_width=sx,
                pixel_height=-sy,  # negative because y decreases
            ), True

        return GeoTransform(pixel_width=sx, pixel_height=-sy), True

    # Tiepoint without scale: still flag as georeferenced (origin known)
    if tiepoint is not None:
        return GeoTransform(), True

    return GeoTransform(), False


def extract_geo_info(ifd: IFD, data: bytes | memoryview,
                     byte_order: str) -> GeoInfo:
    """Extract full geographic metadata from a parsed IFD.

    Parameters
    ----------
    ifd : IFD
        Parsed IFD.
    data : bytes
        Full file data (needed for resolving GeoKey param offsets).
    byte_order : str
        '<' or '>'.

    Returns
    -------
    GeoInfo
    """
    transform, has_georef = _extract_transform(ifd)
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

    # Extract nodata from GDAL_NODATA tag
    nodata = None
    nodata_str = ifd.nodata_str
    if nodata_str is not None:
        try:
            nodata = float(nodata_str)
        except (ValueError, TypeError):
            pass

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
            bps_val = ifd.bits_per_sample
            if isinstance(bps_val, tuple):
                bps_val = bps_val[0]
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


def _model_type_from_wkt(wkt: str) -> int:
    """Guess ModelType from a WKT string prefix."""
    upper = wkt.strip().upper()
    if upper.startswith(('GEOGCS', 'GEOGCRS')):
        return MODEL_TYPE_GEOGRAPHIC
    return MODEL_TYPE_PROJECTED


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

    Returns
    -------
    dict mapping tag ID to (type_id, count, value_bytes) tuples,
    where value_bytes is already serialized for little-endian output.
    """
    tags = {}

    # ModelPixelScaleTag (33550): (ScaleX, ScaleY, ScaleZ)
    sx = abs(transform.pixel_width)
    sy = abs(transform.pixel_height)
    tags[TAG_MODEL_PIXEL_SCALE] = (sx, sy, 0.0)

    # ModelTiepointTag (33922): (I, J, K, X, Y, Z)
    tags[TAG_MODEL_TIEPOINT] = (
        0.0, 0.0, 0.0,
        transform.origin_x, transform.origin_y, 0.0,
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
        # Guess model type from EPSG (simple heuristic)
        if crs_epsg == 4326 or (crs_epsg >= 4000 and crs_epsg < 5000):
            model_type = MODEL_TYPE_GEOGRAPHIC
        else:
            model_type = MODEL_TYPE_PROJECTED
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
        # User-defined CRS: store 32767 and write WKT to GeoAsciiParams
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
        tags[TAG_GDAL_NODATA] = str(nodata)

    return tags
