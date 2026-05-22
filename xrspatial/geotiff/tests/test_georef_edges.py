"""Edge-case tests for georeferenced and non-georeferenced TIFFs.

Issue #1482:
- T-3: descending-y (south-up) round-trip preserves data orientation.
- T-4: a TIFF with no GeoTIFF tags reads back with integer pixel coords
  and no ``crs`` attr, instead of failing or inventing fractional coords.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import extract_geo_info
from xrspatial.geotiff._header import parse_all_ifds, parse_header

from .conftest import make_minimal_tiff


class TestNonGeoreferencedRead:
    """T-4: opening a TIFF with no GeoTIFF tags should not crash."""

    def test_open_plain_tiff(self, tmp_path):
        # make_minimal_tiff with no geo_transform / epsg = no GeoTIFF tags.
        data = make_minimal_tiff(4, 5, np.dtype('float32'))
        path = str(tmp_path / 'plain_1482.tif')
        with open(path, 'wb') as f:
            f.write(data)

        da = open_geotiff(path)

        assert da.shape == (5, 4)  # height=5, width=4
        # No CRS attribute should be set.
        assert 'crs' not in da.attrs
        assert 'crs_wkt' not in da.attrs

    def test_integer_pixel_coords(self, tmp_path):
        """Non-georef fallback: y=0..H-1, x=0..W-1."""
        data = make_minimal_tiff(6, 3, np.dtype('uint16'))
        path = str(tmp_path / 'plain_coords_1482.tif')
        with open(path, 'wb') as f:
            f.write(data)

        da = open_geotiff(path)
        np.testing.assert_array_equal(da.coords['y'].values, np.arange(3))
        np.testing.assert_array_equal(da.coords['x'].values, np.arange(6))

    def test_has_georef_flag_false(self, tmp_path):
        """The GeoInfo carries the explicit has_georef=False flag."""
        data = make_minimal_tiff(4, 4, np.dtype('float32'))
        header = parse_header(data)
        ifd = parse_all_ifds(data, header)[0]
        info = extract_geo_info(ifd, data, header.byte_order)
        assert info.has_georef is False

    def test_has_georef_flag_true_when_present(self, tmp_path):
        data = make_minimal_tiff(
            4, 4, np.dtype('float32'),
            geo_transform=(-120.0, 45.0, 0.001, -0.001),
            epsg=4326,
        )
        header = parse_header(data)
        ifd = parse_all_ifds(data, header)[0]
        info = extract_geo_info(ifd, data, header.byte_order)
        assert info.has_georef is True

    def test_round_trip_preserves_pixels(self, tmp_path):
        """Read a plain TIFF, write it back, read again -- pixels match."""
        pixels = np.arange(20, dtype=np.float32).reshape(4, 5)
        data = make_minimal_tiff(5, 4, pixel_data=pixels)
        in_path = str(tmp_path / 'plain_in_1482.tif')
        out_path = str(tmp_path / 'plain_out_1482.tif')
        with open(in_path, 'wb') as f:
            f.write(data)

        da = open_geotiff(in_path)
        np.testing.assert_array_equal(da.values, pixels)

        # Round-trip through to_geotiff -- should not crash even without CRS
        to_geotiff(da, out_path)
        da2 = open_geotiff(out_path)
        np.testing.assert_array_equal(da2.values, pixels)


class TestDescendingYWrite:
    """T-3: writing a south-up DataArray (positive pixel_height).

    GeoTIFF's ModelPixelScale tag stores absolute scale values; sign
    information is lost when round-tripping through Tiepoint+Scale.  The
    pixel data orientation, however, is preserved exactly.  This test
    documents that contract.
    """

    def _make_south_up_da(self):
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        # Ascending y values: row 0 is the south edge, y increases northward.
        y = np.array([0.5, 1.5, 2.5, 3.5])
        x = np.array([0.5, 1.5, 2.5, 3.5])
        return xr.DataArray(
            data, dims=('y', 'x'), coords={'y': y, 'x': x},
            attrs={'crs': 4326},
        )

    def test_data_orientation_preserved(self, tmp_path):
        """Pixel ordering is identical on round-trip even for south-up input."""
        da = self._make_south_up_da()
        path = str(tmp_path / 'south_up_1482.tif')
        to_geotiff(da, path)
        da2 = open_geotiff(path)

        # Pixel values land in the same row/col positions.
        np.testing.assert_array_equal(da2.values, da.values)

    def test_x_coords_preserved(self, tmp_path):
        da = self._make_south_up_da()
        path = str(tmp_path / 'south_up_x_1482.tif')
        to_geotiff(da, path)
        da2 = open_geotiff(path)
        np.testing.assert_allclose(da2.coords['x'].values,
                                   da.coords['x'].values)

    def test_y_coords_known_limitation(self, tmp_path):
        """Known limitation: y sign is not preserved through Scale+Tiepoint.

        ModelPixelScale stores |pixel_height|; on read we always restore
        a negative pixel_height per GeoTIFF convention.  Document the
        magnitude is right and the sign is the GeoTIFF default.
        """
        da = self._make_south_up_da()
        path = str(tmp_path / 'south_up_y_1482.tif')
        to_geotiff(da, path)
        da2 = open_geotiff(path)
        # Magnitudes match the original spacing.
        spacing = np.abs(np.diff(da2.coords['y'].values))
        assert np.allclose(spacing, 1.0)

    def test_descending_y_round_trip(self, tmp_path):
        """The standard north-up case (descending y) round-trips cleanly."""
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        # Descending y: row 0 at the north edge.
        y = np.array([3.5, 2.5, 1.5, 0.5])
        x = np.array([0.5, 1.5, 2.5, 3.5])
        da = xr.DataArray(
            data, dims=('y', 'x'), coords={'y': y, 'x': x},
            attrs={'crs': 4326},
        )
        path = str(tmp_path / 'north_up_1482.tif')
        to_geotiff(da, path)
        da2 = open_geotiff(path)
        np.testing.assert_array_equal(da2.values, da.values)
        np.testing.assert_allclose(da2.coords['y'].values,
                                   da.coords['y'].values)
        np.testing.assert_allclose(da2.coords['x'].values,
                                   da.coords['x'].values)
