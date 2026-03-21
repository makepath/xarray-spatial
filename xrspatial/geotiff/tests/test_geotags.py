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
