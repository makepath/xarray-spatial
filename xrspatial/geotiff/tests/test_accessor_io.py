"""Tests for .xrs.to_geotiff() and .xrs.open_geotiff() accessor methods."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xrspatial  # noqa: F401 -- registers .xrs accessor
from xrspatial.geotiff import open_geotiff, to_geotiff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_da(height=8, width=10, crs=4326, name='elevation'):
    """Build a georeferenced DataArray for testing."""
    arr = np.arange(height * width, dtype=np.float32).reshape(height, width)
    y = np.linspace(45.0, 44.0, height)
    x = np.linspace(-120.0, -119.0, width)
    return xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name=name,
        attrs={'crs': crs},
    )


def _make_ds(height=8, width=10, crs=4326):
    """Build a georeferenced Dataset for testing."""
    da = _make_da(height, width, crs, name='elevation')
    return xr.Dataset({'elevation': da})


# ---------------------------------------------------------------------------
# DataArray.xrs.to_geotiff
# ---------------------------------------------------------------------------

class TestDataArrayToGeotiff:
    def test_round_trip(self, tmp_path):
        da = _make_da()
        path = str(tmp_path / 'test_1047_da_roundtrip.tif')
        da.xrs.to_geotiff(path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, da.values)

    def test_with_kwargs(self, tmp_path):
        da = _make_da()
        path = str(tmp_path / 'test_1047_da_kwargs.tif')
        da.xrs.to_geotiff(path, compression='deflate', tiled=True,
                          tile_size=256)

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, da.values)

    def test_preserves_crs(self, tmp_path):
        da = _make_da(crs=32610)
        path = str(tmp_path / 'test_1047_da_crs.tif')
        da.xrs.to_geotiff(path, compression='none')

        result = open_geotiff(path)
        assert result.attrs.get('crs') == 32610


# ---------------------------------------------------------------------------
# Dataset.xrs.to_geotiff
# ---------------------------------------------------------------------------

class TestDatasetToGeotiff:
    def test_round_trip(self, tmp_path):
        ds = _make_ds()
        path = str(tmp_path / 'test_1047_ds_roundtrip.tif')
        ds.xrs.to_geotiff(path, compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, ds['elevation'].values)

    def test_explicit_var(self, tmp_path):
        ds = _make_ds()
        ds['slope'] = ds['elevation'] * 2
        path = str(tmp_path / 'test_1047_ds_var.tif')
        ds.xrs.to_geotiff(path, var='slope', compression='none')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, ds['slope'].values)

    def test_no_yx_raises(self, tmp_path):
        ds = xr.Dataset({'vals': xr.DataArray(np.zeros(5), dims=['z'])})
        with pytest.raises(ValueError, match="no variable with 'y' and 'x'"):
            ds.xrs.to_geotiff(str(tmp_path / 'bad.tif'))


# ---------------------------------------------------------------------------
# Dataset.xrs.open_geotiff (spatially-windowed read)
# ---------------------------------------------------------------------------

class TestDatasetOpenGeotiff:
    def test_windowed_read(self, tmp_path):
        """Reading with a Dataset template should return a spatial subset."""
        # Write a 20x20 raster
        big = _make_da(height=20, width=20)
        big_path = str(tmp_path / 'test_1047_big.tif')
        to_geotiff(big, big_path, compression='none')

        # Template dataset covers the center region
        y_sub = big.coords['y'].values[5:15]
        x_sub = big.coords['x'].values[5:15]
        template = xr.Dataset({
            'dummy': xr.DataArray(
                np.zeros((len(y_sub), len(x_sub))),
                dims=['y', 'x'],
                coords={'y': y_sub, 'x': x_sub},
            )
        })

        result = template.xrs.open_geotiff(big_path)
        # Result should be smaller than the full raster
        assert result.shape[0] <= 20
        assert result.shape[1] <= 20
        # And at least as large as the template
        assert result.shape[0] >= len(y_sub)
        assert result.shape[1] >= len(x_sub)

    def test_full_extent_returns_all(self, tmp_path):
        """Template covering full extent should return the whole raster."""
        da = _make_da(height=8, width=10)
        path = str(tmp_path / 'test_1047_full.tif')
        to_geotiff(da, path, compression='none')

        template = xr.Dataset({
            'dummy': xr.DataArray(
                np.zeros_like(da.values),
                dims=['y', 'x'],
                coords={'y': da.coords['y'].values,
                        'x': da.coords['x'].values},
            )
        })
        result = template.xrs.open_geotiff(path)
        np.testing.assert_array_equal(result.values, da.values)

    def test_no_coords_raises(self, tmp_path):
        da = _make_da()
        path = str(tmp_path / 'test_1047_nocoords.tif')
        to_geotiff(da, path, compression='none')

        ds = xr.Dataset({'vals': xr.DataArray(np.zeros(5), dims=['z'])})
        with pytest.raises(ValueError, match="'y' and 'x' coordinates"):
            ds.xrs.open_geotiff(path)

    def test_kwargs_forwarded(self, tmp_path):
        """Extra kwargs like name= should be forwarded to open_geotiff."""
        da = _make_da(height=8, width=10)
        path = str(tmp_path / 'test_1047_kwargs.tif')
        to_geotiff(da, path, compression='none')

        template = xr.Dataset({
            'dummy': xr.DataArray(
                np.zeros_like(da.values),
                dims=['y', 'x'],
                coords={'y': da.coords['y'].values,
                        'x': da.coords['x'].values},
            )
        })
        result = template.xrs.open_geotiff(path, name='myname')
        assert result.name == 'myname'
