"""Regression test for the int-coord write hotfix.

PR #1953 added ``require_transform_for_georeferenced`` which raises when
both spatial dims are present in ``da.coords`` and no transform was
resolved. PR #1954 made ``coords_to_transform`` return ``None`` for
integer-dtype x/y coords as a no-georef sentinel (#1949). Their
interaction broke every writer call against an int-coord DataArray:
the resolver returned ``None``, then the guard raised. This test
locks in the int-dtype exemption that mirrors ``coords_to_transform``.
"""
import os
import tempfile

import numpy as np
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff

from .conftest import requires_integration


def _tmp_path(name):
    return os.path.join(tempfile.gettempdir(), name)


@requires_integration
class TestIntCoordRoundTripHotfix1962:
    def test_int_coords_2d_round_trip(self):
        pixels = np.arange(20, dtype=np.float32).reshape(4, 5)
        da = xr.DataArray(
            pixels,
            dims=['y', 'x'],
            coords={
                'y': np.arange(4, dtype=np.int64),
                'x': np.arange(5, dtype=np.int64),
            },
            attrs={'long_name': 'int_coord_2d'},
        )
        path = _tmp_path('hotfix_1962_int_2d.tif')
        try:
            to_geotiff(da, path)
            da2 = open_geotiff(path)
            np.testing.assert_array_equal(da2.values, pixels)
            # Read-side should re-emit the int-coord no-georef
            # placeholders, confirming the no-georef contract held.
            assert da2.coords['x'].dtype.kind in ('i', 'u')
            assert da2.coords['y'].dtype.kind in ('i', 'u')
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_int_coords_3d_band_y_x_round_trip(self):
        pixels = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
        da = xr.DataArray(
            pixels,
            dims=['band', 'y', 'x'],
            coords={
                'band': np.arange(2, dtype=np.int64),
                'y': np.arange(4, dtype=np.int64),
                'x': np.arange(5, dtype=np.int64),
            },
            attrs={'long_name': 'int_coord_3d'},
        )
        path = _tmp_path('hotfix_1962_int_3d.tif')
        try:
            to_geotiff(da, path)
            da2 = open_geotiff(path)
            # open_geotiff may emit (band, y, x) or (y, x, band) layout;
            # compare on a band-by-band basis instead of guessing axis order.
            arr = da2.values
            assert arr.shape in ((2, 4, 5), (4, 5, 2))
            if arr.shape == (2, 4, 5):
                np.testing.assert_array_equal(arr, pixels)
            else:
                np.testing.assert_array_equal(
                    np.moveaxis(arr, -1, 0), pixels
                )
        finally:
            if os.path.exists(path):
                os.remove(path)
