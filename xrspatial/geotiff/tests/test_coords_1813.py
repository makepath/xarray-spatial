"""Unit tests for ``xrspatial.geotiff._coords`` (issue #1813).

Covers the shared coord / transform helpers extracted from
``__init__.py``: ``coords_from_pixel_geometry``,
``transform_tuple_from_pixel_geometry``, and the ``coords_from_geo_info``
wrapper. Each backend's read path now calls these helpers instead of
keeping its own inline copy of the GeoTransform-to-(y, x) maths.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from xrspatial.geotiff._coords import (coords_from_geo_info, coords_from_pixel_geometry,
                                       transform_tuple_from_pixel_geometry)
from xrspatial.geotiff._geotags import RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT, GeoTransform


class TestCoordsFromPixelGeometry:
    def test_basic_area_north_up(self):
        # Standard north-up affine: origin at top-left, negative
        # pixel_height. PixelIsArea => coords shift to pixel centers.
        coords = coords_from_pixel_geometry(
            origin_x=100.0, origin_y=200.0,
            pixel_width=10.0, pixel_height=-10.0,
            height=3, width=4,
        )
        expected_x = np.array([105.0, 115.0, 125.0, 135.0])
        expected_y = np.array([195.0, 185.0, 175.0])
        np.testing.assert_array_equal(coords['x'], expected_x)
        np.testing.assert_array_equal(coords['y'], expected_y)

    def test_windowed_area(self):
        # Window (r0=1, c0=2, r1=3, c1=5) over a virtual source. The
        # returned coords describe absolute pixel-center positions for
        # rows 1..2 and columns 2..4, not 0..height-1 / 0..width-1.
        coords = coords_from_pixel_geometry(
            origin_x=0.0, origin_y=0.0,
            pixel_width=1.0, pixel_height=-1.0,
            height=2, width=3,
            window=(1, 2, 3, 5),
        )
        # PixelIsArea adds half-pixel; column 2 center at 2.5, row 1 at -1.5
        np.testing.assert_array_equal(coords['x'], np.array([2.5, 3.5, 4.5]))
        np.testing.assert_array_equal(coords['y'], np.array([-1.5, -2.5]))

    def test_pixel_is_point_skips_half_pixel_shift(self):
        # PixelIsPoint: the tiepoint already sits at the pixel center,
        # so coords come back as origin + n * step with no offset.
        coords_area = coords_from_pixel_geometry(
            origin_x=0.0, origin_y=0.0,
            pixel_width=10.0, pixel_height=-10.0,
            height=2, width=2,
            is_point=False,
        )
        coords_point = coords_from_pixel_geometry(
            origin_x=0.0, origin_y=0.0,
            pixel_width=10.0, pixel_height=-10.0,
            height=2, width=2,
            is_point=True,
        )
        # Area coords have a +5 / -5 half-pixel shift relative to Point.
        np.testing.assert_array_equal(
            coords_area['x'] - 5.0, coords_point['x'])
        np.testing.assert_array_equal(
            coords_area['y'] + 5.0, coords_point['y'])

    def test_negative_y_resolution_north_up(self):
        # Real GeoTIFFs are normally north-up (origin at top, y decreases
        # with row index). Confirm y[0] > y[-1] and step matches.
        coords = coords_from_pixel_geometry(
            origin_x=500_000.0, origin_y=4_500_000.0,
            pixel_width=30.0, pixel_height=-30.0,
            height=5, width=1,
        )
        assert coords['y'][0] > coords['y'][-1]
        np.testing.assert_allclose(np.diff(coords['y']), -30.0)
        # Half-pixel shift applied for PixelIsArea
        assert coords['y'][0] == pytest.approx(4_500_000.0 - 15.0)

    def test_no_georef_returns_integer_pixel_coords(self):
        coords = coords_from_pixel_geometry(
            origin_x=0.0, origin_y=0.0,
            pixel_width=1.0, pixel_height=1.0,
            height=3, width=4,
            has_georef=False,
        )
        np.testing.assert_array_equal(coords['x'], np.arange(4, dtype=np.int64))
        np.testing.assert_array_equal(coords['y'], np.arange(3, dtype=np.int64))
        # Integer coords, not float
        assert coords['x'].dtype == np.int64
        assert coords['y'].dtype == np.int64

    def test_no_georef_windowed_returns_integer_window_indices(self):
        coords = coords_from_pixel_geometry(
            origin_x=0.0, origin_y=0.0,
            pixel_width=1.0, pixel_height=1.0,
            height=2, width=2,
            window=(5, 7, 7, 9),
            has_georef=False,
        )
        np.testing.assert_array_equal(coords['x'], np.array([7, 8]))
        np.testing.assert_array_equal(coords['y'], np.array([5, 6]))
        assert coords['x'].dtype == np.int64


class TestTransformTupleFromPixelGeometry:
    def test_basic_tuple_ordering(self):
        # Rasterio order: (a, 0, c, 0, e, f)
        tup = transform_tuple_from_pixel_geometry(
            origin_x=100.0, origin_y=200.0,
            pixel_width=10.0, pixel_height=-10.0,
        )
        assert tup == (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)

    def test_windowed_origin_shifts(self):
        # window=(r0, c0, ...) bumps the origin by c0*pixel_width /
        # r0*pixel_height.
        tup = transform_tuple_from_pixel_geometry(
            origin_x=100.0, origin_y=200.0,
            pixel_width=10.0, pixel_height=-10.0,
            window=(3, 4, 0, 0),
        )
        assert tup == (10.0, 0.0, 140.0, 0.0, -10.0, 170.0)


class TestCoordsFromGeoInfo:
    def _geo_info(self, *, transform, raster_type, has_georef=True):
        return SimpleNamespace(
            transform=transform,
            raster_type=raster_type,
            has_georef=has_georef,
        )

    def test_area_full_extent(self):
        gi = self._geo_info(
            transform=GeoTransform(
                origin_x=0.0, origin_y=0.0,
                pixel_width=1.0, pixel_height=-1.0,
            ),
            raster_type=RASTER_PIXEL_IS_AREA,
        )
        coords = coords_from_geo_info(gi, height=2, width=2)
        np.testing.assert_array_equal(coords['x'], np.array([0.5, 1.5]))
        np.testing.assert_array_equal(coords['y'], np.array([-0.5, -1.5]))

    def test_windowed(self):
        gi = self._geo_info(
            transform=GeoTransform(
                origin_x=0.0, origin_y=0.0,
                pixel_width=1.0, pixel_height=-1.0,
            ),
            raster_type=RASTER_PIXEL_IS_AREA,
        )
        coords = coords_from_geo_info(
            gi, height=2, width=2, window=(3, 4, 5, 6),
        )
        np.testing.assert_array_equal(coords['x'], np.array([4.5, 5.5]))
        np.testing.assert_array_equal(coords['y'], np.array([-3.5, -4.5]))

    def test_pixel_is_point(self):
        gi = self._geo_info(
            transform=GeoTransform(
                origin_x=10.0, origin_y=20.0,
                pixel_width=2.0, pixel_height=-2.0,
            ),
            raster_type=RASTER_PIXEL_IS_POINT,
        )
        coords = coords_from_geo_info(gi, height=2, width=2)
        np.testing.assert_array_equal(coords['x'], np.array([10.0, 12.0]))
        np.testing.assert_array_equal(coords['y'], np.array([20.0, 18.0]))

    def test_no_georef_returns_integer_coords(self):
        gi = self._geo_info(
            transform=GeoTransform(),
            raster_type=RASTER_PIXEL_IS_AREA,
            has_georef=False,
        )
        coords = coords_from_geo_info(gi, height=3, width=3)
        np.testing.assert_array_equal(coords['x'], np.arange(3, dtype=np.int64))
        np.testing.assert_array_equal(coords['y'], np.arange(3, dtype=np.int64))
        assert coords['x'].dtype == np.int64

    def test_none_transform_treated_as_no_georef(self):
        gi = self._geo_info(
            transform=None,
            raster_type=RASTER_PIXEL_IS_AREA,
            has_georef=True,
        )
        coords = coords_from_geo_info(gi, height=2, width=2)
        np.testing.assert_array_equal(coords['x'], np.arange(2, dtype=np.int64))
        np.testing.assert_array_equal(coords['y'], np.arange(2, dtype=np.int64))
