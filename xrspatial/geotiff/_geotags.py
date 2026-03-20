"""GeoTIFF tag interpretation: CRS, affine transform, GeoKeys."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ._header import (
    IFD,
    TAG_MODEL_PIXEL_SCALE,
    TAG_MODEL_TIEPOINT,
    TAG_MODEL_TRANSFORMATION,
    TAG_GEO_KEY_DIRECTORY,
    TAG_GEO_DOUBLE_PARAMS,
    TAG_GEO_ASCII_PARAMS,
    TAG_GDAL_NODATA,
)

# GeoKey IDs
GEOKEY_MODEL_TYPE = 1024
GEOKEY_RASTER_TYPE = 1025
GEOKEY_GEOGRAPHIC_TYPE = 2048
GEOKEY_GEOG_CITATION = 2049
GEOKEY_GEODETIC_DATUM = 2050
GEOKEY_GEOG_LINEAR_UNITS = 2052
GEOKEY_GEOG_ANGULAR_UNITS = 2054
GEOKEY_PROJECTED_CS_TYPE = 3072
GEOKEY_PROJ_CITATION = 3073
GEOKEY_PROJECTION = 3074
GEOKEY_PROJ_LINEAR_UNITS = 3076

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
    crs_epsg: int | None = None
    model_type: int = 0
    raster_type: int = RASTER_PIXEL_IS_AREA
    nodata: float | None = None
    colormap: list | None = None  # list of (R, G, B, A) float tuples, or None
    geokeys: dict[int, int | float | str] = field(default_factory=dict)


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


def _extract_transform(ifd: IFD) -> GeoTransform:
    """Extract affine transform from ModelTransformation, or
    ModelTiepoint + ModelPixelScale tags."""

    # Try ModelTransformationTag (4x4 matrix)
    transform_tag = ifd.get_value(TAG_MODEL_TRANSFORMATION)
    if transform_tag is not None:
        if isinstance(transform_tag, tuple) and len(transform_tag) >= 12:
            # 4x4 row-major matrix
            # x = M[0]*col + M[1]*row + M[3]
            # y = M[4]*col + M[5]*row + M[7]
            return GeoTransform(
                origin_x=transform_tag[3],
                origin_y=transform_tag[7],
                pixel_width=transform_tag[0],
                pixel_height=transform_tag[5],
            )

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
            )

        return GeoTransform(pixel_width=sx, pixel_height=-sy)

    return GeoTransform()


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
    transform = _extract_transform(ifd)
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

    # Extract nodata from GDAL_NODATA tag
    nodata = None
    nodata_str = ifd.nodata_str
    if nodata_str is not None:
        try:
            nodata = float(nodata_str)
        except (ValueError, TypeError):
            pass

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

    return GeoInfo(
        transform=transform,
        crs_epsg=epsg,
        model_type=int(model_type) if isinstance(model_type, (int, float)) else 0,
        raster_type=int(raster_type) if isinstance(raster_type, (int, float)) else RASTER_PIXEL_IS_AREA,
        nodata=nodata,
        colormap=colormap,
        geokeys=geokeys,
    )


def build_geo_tags(transform: GeoTransform, crs_epsg: int | None = None,
                   nodata=None,
                   raster_type: int = RASTER_PIXEL_IS_AREA) -> dict[int, tuple]:
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

    # ModelType
    if crs_epsg is not None:
        # Guess model type from EPSG (simple heuristic)
        if crs_epsg == 4326 or (crs_epsg >= 4000 and crs_epsg < 5000):
            model_type = MODEL_TYPE_GEOGRAPHIC
        else:
            model_type = MODEL_TYPE_PROJECTED
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

    num_keys = len(key_entries)
    header = [1, 1, 0, num_keys]
    flat = header.copy()
    for entry in key_entries:
        flat.extend(entry)

    tags[TAG_GEO_KEY_DIRECTORY] = tuple(flat)

    # GDAL_NODATA
    if nodata is not None:
        tags[TAG_GDAL_NODATA] = str(nodata)

    return tags
