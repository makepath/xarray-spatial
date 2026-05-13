"""Tests for GeoTIFF tag interpretation."""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._geotags import (
    GeoInfo,
    GeoTransform,
    build_geo_tags,
    extract_geo_info,
    GEOKEY_GEOGRAPHIC_TYPE,
    GEOKEY_MODEL_TYPE,
    GEOKEY_PROJECTED_CS_TYPE,
    GEOKEY_RASTER_TYPE,
    MODEL_TYPE_GEOGRAPHIC,
    MODEL_TYPE_PROJECTED,
    RASTER_PIXEL_IS_AREA,
    TAG_GEO_KEY_DIRECTORY,
    TAG_GDAL_NODATA,
    TAG_MODEL_PIXEL_SCALE,
    TAG_MODEL_TIEPOINT,
)
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from .conftest import make_minimal_tiff


class TestGeoTransform:
    def test_defaults(self):
        gt = GeoTransform()
        assert gt.origin_x == 0.0
        assert gt.origin_y == 0.0
        assert gt.pixel_width == 1.0
        assert gt.pixel_height == -1.0


class TestExtractGeoInfo:
    def test_with_tiepoint_and_scale(self):
        data = make_minimal_tiff(
            4, 4, np.dtype('float32'),
            geo_transform=(-120.0, 45.0, 0.001, -0.001),
            epsg=4326,
        )
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) == 1

        geo = extract_geo_info(ifds[0], data, header.byte_order)
        assert geo.transform.origin_x == pytest.approx(-120.0)
        assert geo.transform.origin_y == pytest.approx(45.0)
        assert geo.transform.pixel_width == pytest.approx(0.001)
        assert geo.transform.pixel_height == pytest.approx(-0.001)
        assert geo.crs_epsg == 4326
        assert geo.model_type == MODEL_TYPE_GEOGRAPHIC

    def test_projected_crs(self):
        data = make_minimal_tiff(
            4, 4, np.dtype('float32'),
            geo_transform=(500000.0, 4500000.0, 30.0, -30.0),
            epsg=32610,
        )
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        geo = extract_geo_info(ifds[0], data, header.byte_order)
        assert geo.crs_epsg == 32610
        assert geo.model_type == MODEL_TYPE_PROJECTED

    def test_no_geo_tags(self):
        data = make_minimal_tiff(4, 4, np.dtype('float32'))
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        geo = extract_geo_info(ifds[0], data, header.byte_order)
        assert geo.crs_epsg is None
        # Default transform
        assert geo.transform.pixel_width == 1.0


class TestBuildGeoTags:
    def test_basic(self):
        gt = GeoTransform(-120.0, 45.0, 0.001, -0.001)
        tags = build_geo_tags(gt, crs_epsg=4326, nodata=-9999.0)

        assert TAG_MODEL_PIXEL_SCALE in tags
        scale = tags[TAG_MODEL_PIXEL_SCALE]
        assert scale[0] == pytest.approx(0.001)
        assert scale[1] == pytest.approx(0.001)

        assert TAG_MODEL_TIEPOINT in tags
        tp = tags[TAG_MODEL_TIEPOINT]
        assert tp[3] == pytest.approx(-120.0)
        assert tp[4] == pytest.approx(45.0)

        assert TAG_GEO_KEY_DIRECTORY in tags
        assert TAG_GDAL_NODATA in tags
        assert tags[TAG_GDAL_NODATA] == '-9999.0'

    def test_no_crs(self):
        gt = GeoTransform(0.0, 0.0, 1.0, -1.0)
        tags = build_geo_tags(gt, crs_epsg=None, nodata=None)
        assert TAG_MODEL_PIXEL_SCALE in tags
        assert TAG_GEO_KEY_DIRECTORY in tags
        assert TAG_GDAL_NODATA not in tags

    def test_projected_crs_geokey(self):
        gt = GeoTransform(500000.0, 4500000.0, 30.0, -30.0)
        tags = build_geo_tags(gt, crs_epsg=32610)
        geokeys = tags[TAG_GEO_KEY_DIRECTORY]
        # Flatten and check that ProjectedCSType is present
        assert 3072 in geokeys  # GEOKEY_PROJECTED_CS_TYPE


def _build_tiff_with_transformation_tag(matrix_16: tuple) -> bytes:
    """Build a tiny single-strip TIFF carrying a 4x4 ModelTransformationTag.

    No ModelPixelScale or ModelTiepoint -- the reader has to use the
    transformation tag.
    """
    import struct

    bo = '<'
    width, height = 2, 2
    pixels = np.zeros((height, width), dtype=np.uint8)

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_doubles(tag, vals):
        tag_list.append(
            (tag, 12, len(vals), struct.pack(f'{bo}{len(vals)}d', *vals)))

    add_short(256, width)            # ImageWidth
    add_short(257, height)           # ImageLength
    add_short(258, 8)                # BitsPerSample
    add_short(259, 1)                # Compression: none
    add_short(262, 1)                # PhotometricInterpretation: BlackIsZero
    add_short(277, 1)                # SamplesPerPixel
    add_short(278, height)           # RowsPerStrip
    add_long(273, 0)                 # StripOffsets (placeholder)
    add_long(279, len(pixels.tobytes()))  # StripByteCounts
    add_short(339, 1)                # SampleFormat
    # ModelTransformationTag (34264): 16 doubles, row-major 4x4.
    add_doubles(34264, list(matrix_16))

    tag_list.sort(key=lambda t: t[0])

    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4

    # Build overflow buffer for tags whose value exceeds 4 bytes.
    overflow = bytearray()
    overflow_offsets = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            overflow_offsets[tag] = ifd_start + ifd_size + len(overflow)
            overflow.extend(raw)
            if len(overflow) % 2:
                overflow.append(0)

    pixel_start = ifd_start + ifd_size + len(overflow)

    # Patch StripOffsets
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count, struct.pack(f'{bo}I', pixel_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            out.extend(struct.pack(f'{bo}I', overflow_offsets[tag]))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow)
    out.extend(pixels.tobytes())
    return bytes(out)


class TestModelTransformationTag_1486:
    """Issue #1486: handle ModelTransformationTag (34264) explicitly.

    The reader previously read the matrix but silently discarded any
    rotation, skew, or z-coupling.  The fix raises NotImplementedError
    instead of returning a corrupted transform.
    """

    def test_axis_aligned_extracts_correctly(self, tmp_path):
        # M = [[sx, 0, 0, ox], [0, sy, 0, oy], [0, 0, 1, 0], [0, 0, 0, 1]]
        # Note pixel_height is M[5] verbatim -- the writer encodes a negative
        # value here so it stays negative in the read transform.
        ox, oy, sx, sy = 500000.0, 4500000.0, 30.0, -30.0
        matrix = (
            sx, 0.0, 0.0, ox,
            0.0, sy, 0.0, oy,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        data = _build_tiff_with_transformation_tag(matrix)
        path = tmp_path / 'transform_axis_aligned_1486.tif'
        path.write_bytes(data)

        from xrspatial.geotiff._reader import read_to_array
        _, geo_info = read_to_array(str(path))
        assert geo_info.transform.origin_x == pytest.approx(ox)
        assert geo_info.transform.origin_y == pytest.approx(oy)
        assert geo_info.transform.pixel_width == pytest.approx(sx)
        assert geo_info.transform.pixel_height == pytest.approx(sy)

    def test_rotation_raises(self, tmp_path):
        # Non-zero rotation in M[1] / M[4]
        matrix = (
            30.0, 5.0, 0.0, 500000.0,
            5.0, -30.0, 0.0, 4500000.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        data = _build_tiff_with_transformation_tag(matrix)
        path = tmp_path / 'transform_rotated_1486.tif'
        path.write_bytes(data)

        from xrspatial.geotiff._reader import read_to_array
        with pytest.raises(NotImplementedError) as exc:
            read_to_array(str(path))
        assert 'ModelTransformationTag' in str(exc.value)
        assert 'rotation' in str(exc.value).lower() or 'skew' in str(exc.value).lower()

    def test_z_coupling_raises(self, tmp_path):
        # Non-zero z-coupling in M[2] / M[6]
        matrix = (
            30.0, 0.0, 0.5, 500000.0,
            0.0, -30.0, 0.5, 4500000.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        data = _build_tiff_with_transformation_tag(matrix)
        path = tmp_path / 'transform_z_coupled_1486.tif'
        path.write_bytes(data)

        from xrspatial.geotiff._reader import read_to_array
        with pytest.raises(NotImplementedError):
            read_to_array(str(path))


def _build_tiff_with_tiepoint_only(tiepoint_6: tuple) -> bytes:
    """Build a tiny single-strip TIFF carrying ModelTiepointTag but
    no ModelPixelScaleTag or ModelTransformationTag.

    The GeoTIFF spec permits this configuration; the tiepoint encodes a
    real-world (X, Y) origin for pixel (I, J).  When ModelPixelScaleTag
    is absent the spec-level unit pixel scale is (sx, sy) = (1.0, 1.0)
    (sy is a positive magnitude in spec terms).  The reader translates
    that into the GeoTransform sign convention used in this repo, where
    pixel_height is stored as -sy, so the resulting transform has
    pixel_width == 1.0 and pixel_height == -1.0.
    """
    import struct

    bo = '<'
    width, height = 2, 2
    pixels = np.zeros((height, width), dtype=np.uint8)

    tag_list = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_doubles(tag, vals):
        tag_list.append(
            (tag, 12, len(vals), struct.pack(f'{bo}{len(vals)}d', *vals)))

    add_short(256, width)            # ImageWidth
    add_short(257, height)           # ImageLength
    add_short(258, 8)                # BitsPerSample
    add_short(259, 1)                # Compression: none
    add_short(262, 1)                # PhotometricInterpretation
    add_short(277, 1)                # SamplesPerPixel
    add_short(278, height)           # RowsPerStrip
    add_long(273, 0)                 # StripOffsets (placeholder)
    add_long(279, len(pixels.tobytes()))  # StripByteCounts
    add_short(339, 1)                # SampleFormat
    # ModelTiepointTag (33922): 6 doubles (I, J, K, X, Y, Z).
    # Deliberately no ModelPixelScaleTag (33550).
    add_doubles(33922, list(tiepoint_6))

    tag_list.sort(key=lambda t: t[0])

    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4

    overflow = bytearray()
    overflow_offsets = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            overflow_offsets[tag] = ifd_start + ifd_size + len(overflow)
            overflow.extend(raw)
            if len(overflow) % 2:
                overflow.append(0)

    pixel_start = ifd_start + ifd_size + len(overflow)

    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            patched.append((tag, typ, count, struct.pack(f'{bo}I', pixel_start)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            out.extend(struct.pack(f'{bo}I', overflow_offsets[tag]))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow)
    out.extend(pixels.tobytes())
    return bytes(out)


class TestTiepointWithoutScale_1750:
    """Issue #1750: ModelTiepointTag present, ModelPixelScaleTag absent.

    Previously the reader returned a default GeoTransform with origin (0, 0)
    while still flagging the raster as georeferenced, silently relocating it.
    The fix honours the tiepoint's X/Y and falls back to unit pixel scale
    (the GeoTIFF spec convention when ModelPixelScale is absent).
    """

    def test_tiepoint_origin_preserved(self, tmp_path):
        # I=0, J=0 maps to (X=500000, Y=4500000)
        tiepoint = (0.0, 0.0, 0.0, 500000.0, 4500000.0, 0.0)
        data = _build_tiff_with_tiepoint_only(tiepoint)
        path = tmp_path / 'tiepoint_no_scale_1750.tif'
        path.write_bytes(data)

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        from xrspatial.geotiff._geotags import _extract_transform
        transform, has_georef = _extract_transform(ifds[0])

        assert has_georef is True
        assert transform.origin_x == pytest.approx(500000.0)
        assert transform.origin_y == pytest.approx(4500000.0)
        # Unit pixel scale fallback per GeoTIFF spec
        assert transform.pixel_width == pytest.approx(1.0)
        assert transform.pixel_height == pytest.approx(-1.0)

    def test_tiepoint_with_nonzero_ij(self, tmp_path):
        # I=2, J=3 maps to (X=100, Y=200) -- origin shifts to (98, 203)
        tiepoint = (2.0, 3.0, 0.0, 100.0, 200.0, 0.0)
        data = _build_tiff_with_tiepoint_only(tiepoint)
        path = tmp_path / 'tiepoint_no_scale_ij_1750.tif'
        path.write_bytes(data)

        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        from xrspatial.geotiff._geotags import _extract_transform
        transform, has_georef = _extract_transform(ifds[0])

        assert has_georef is True
        # origin_x = tp_x - tp_i * 1.0 = 100 - 2 = 98
        # origin_y = tp_y + tp_j * 1.0 = 200 + 3 = 203
        assert transform.origin_x == pytest.approx(98.0)
        assert transform.origin_y == pytest.approx(203.0)
        assert transform.pixel_width == pytest.approx(1.0)
        assert transform.pixel_height == pytest.approx(-1.0)
