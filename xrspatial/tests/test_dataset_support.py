"""Tests for xr.Dataset support (issue #134)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrspatial import slope, aspect
from xrspatial.classify import quantile
from xrspatial.focal import mean as focal_mean
from xrspatial.multispectral import ndvi, evi
from xrspatial.zonal import stats as zonal_stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def elevation_dataset():
    """Dataset with two elevation-like variables."""
    np.random.seed(42)
    y = np.linspace(0, 1, 20)
    x = np.linspace(0, 1, 20)
    dem1 = xr.DataArray(
        np.random.rand(20, 20).astype(np.float64) * 1000,
        dims=['y', 'x'], coords={'y': y, 'x': x},
        attrs={'res': (y[1] - y[0], x[1] - x[0])},
    )
    dem2 = xr.DataArray(
        np.random.rand(20, 20).astype(np.float64) * 500,
        dims=['y', 'x'], coords={'y': y, 'x': x},
        attrs={'res': (y[1] - y[0], x[1] - x[0])},
    )
    return xr.Dataset({'dem1': dem1, 'dem2': dem2}, attrs={'source': 'test'})


@pytest.fixture
def spectral_dataset():
    """Dataset mimicking multi-band satellite imagery."""
    np.random.seed(123)
    data = lambda: np.random.rand(30, 30).astype(np.float64) * 0.5 + 0.1
    dims = ['y', 'x']
    return xr.Dataset({
        'nir': xr.DataArray(data(), dims=dims),
        'red': xr.DataArray(data(), dims=dims),
        'blue': xr.DataArray(data(), dims=dims),
    })


@pytest.fixture
def zones_and_values():
    """Zones raster + values Dataset for zonal stats tests."""
    np.random.seed(7)
    zones_data = np.zeros((10, 10), dtype=np.float64)
    zones_data[:5, :] = 1.0
    zones_data[5:, :] = 2.0
    zones = xr.DataArray(zones_data, dims=['y', 'x'])

    vals_a = np.random.rand(10, 10).astype(np.float64) * 100
    vals_b = np.random.rand(10, 10).astype(np.float64) * 50
    ds = xr.Dataset({
        'elevation': xr.DataArray(vals_a, dims=['y', 'x']),
        'temperature': xr.DataArray(vals_b, dims=['y', 'x']),
    })
    return zones, ds


# ===================================================================
# A. Single-input decorator (supports_dataset)
# ===================================================================

class TestSupportsDataset:

    def test_slope_dataset_returns_dataset(self, elevation_dataset):
        result = slope(elevation_dataset)
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {'dem1', 'dem2'}

    def test_slope_dataset_matches_individual(self, elevation_dataset):
        ds = elevation_dataset
        result = slope(ds)
        for var in ds.data_vars:
            expected = slope(ds[var])
            xr.testing.assert_allclose(result[var], expected)

    def test_slope_dataset_preserves_attrs(self, elevation_dataset):
        result = slope(elevation_dataset)
        assert result.attrs == elevation_dataset.attrs

    def test_slope_dataarray_unchanged(self, elevation_dataset):
        """Existing DataArray path is a passthrough."""
        da = elevation_dataset['dem1']
        result = slope(da)
        assert isinstance(result, xr.DataArray)

    def test_classify_quantile_dataset(self, elevation_dataset):
        result = quantile(elevation_dataset, k=4)
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {'dem1', 'dem2'}
        for var in result.data_vars:
            expected = quantile(elevation_dataset[var], k=4)
            xr.testing.assert_equal(result[var], expected)

    def test_single_var_dataset(self, elevation_dataset):
        ds = elevation_dataset[['dem1']]
        result = slope(ds)
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {'dem1'}

    def test_dataset_error_propagation(self):
        """Bad data in one variable raises immediately."""
        ds = xr.Dataset({
            'ok': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x']),
            'bad': xr.DataArray(np.array(['a', 'b', 'c', 'd', 'e']), dims=['z']),
        })
        with pytest.raises(Exception):
            slope(ds)

    def test_first_param_by_keyword(self, elevation_dataset):
        """The raster binds by its real keyword name, not just positionally.

        Regression test: the wrapper used to require the first argument
        positionally, so slope(agg=...) worked by accident while
        cost_distance(raster=...) and a_star_search(surface=...) raised
        TypeError.
        """
        da = elevation_dataset['dem1']
        xr.testing.assert_allclose(slope(agg=da), slope(da))

        from xrspatial import proximity
        target = da.copy()
        target.data = np.zeros_like(target.data)
        target.data[10, 10] = 1.0
        xr.testing.assert_allclose(
            proximity(raster=target), proximity(target))

        from xrspatial import a_star_search
        surf = da.copy()
        surf['y'] = np.linspace(19, 0, 20)
        surf['x'] = np.linspace(0, 19, 20)
        surf.attrs['res'] = (1.0, 1.0)  # match the replaced coords
        start, goal = (19.0, 0.0), (0.0, 19.0)
        xr.testing.assert_allclose(
            a_star_search(surface=surf, start=start, goal=goal),
            a_star_search(surf, start, goal),
        )

    def test_first_param_missing_raises_typeerror(self):
        with pytest.raises(TypeError):
            slope()

    def test_aspect_dataset(self, elevation_dataset):
        result = aspect(elevation_dataset)
        assert isinstance(result, xr.Dataset)
        for var in elevation_dataset.data_vars:
            expected = aspect(elevation_dataset[var])
            xr.testing.assert_allclose(result[var], expected)

    def test_focal_mean_dataset(self, elevation_dataset):
        result = focal_mean(elevation_dataset)
        assert isinstance(result, xr.Dataset)
        for var in elevation_dataset.data_vars:
            expected = focal_mean(elevation_dataset[var])
            xr.testing.assert_allclose(result[var], expected)


# ===================================================================
# B. Multi-input decorator (supports_dataset_bands)
# ===================================================================

class TestSupportsDatasetBands:

    def test_ndvi_dataset_band_kwargs(self, spectral_dataset):
        result = ndvi(spectral_dataset, nir='nir', red='red')
        assert isinstance(result, xr.DataArray)

    def test_ndvi_dataset_matches_individual(self, spectral_dataset):
        ds = spectral_dataset
        from_ds = ndvi(ds, nir='nir', red='red')
        from_da = ndvi(ds['nir'], ds['red'])
        xr.testing.assert_allclose(from_ds, from_da)

    def test_ndvi_missing_band_kwarg(self, spectral_dataset):
        with pytest.raises(TypeError, match="'red' keyword required"):
            ndvi(spectral_dataset, nir='nir')  # missing red=

    def test_ndvi_invalid_var_name(self, spectral_dataset):
        with pytest.raises(ValueError, match="'nonexistent' not in Dataset"):
            ndvi(spectral_dataset, nir='nonexistent', red='red')

    def test_evi_extra_kwargs_passthrough(self, spectral_dataset):
        """Extra kwargs like soil_factor, gain are passed through."""
        ds = spectral_dataset
        result = evi(ds, nir='nir', red='red', blue='blue',
                     soil_factor=0.5, gain=3.0)
        assert isinstance(result, xr.DataArray)
        expected = evi(ds['nir'], ds['red'], ds['blue'],
                       soil_factor=0.5, gain=3.0)
        xr.testing.assert_allclose(result, expected)

    def test_ndvi_dataarray_unchanged(self, spectral_dataset):
        """Existing positional DataArray call still works."""
        ds = spectral_dataset
        result = ndvi(ds['nir'], ds['red'])
        assert isinstance(result, xr.DataArray)


# ===================================================================
# C. Zonal stats Dataset
# ===================================================================

class TestZonalStatsDataset:

    def test_zonal_stats_dataset_column_naming(self, zones_and_values):
        zones, ds = zones_and_values
        result = zonal_stats(zones, ds)
        assert isinstance(result, pd.DataFrame)
        assert 'zone' in result.columns
        # Check prefixed columns exist
        for var in ds.data_vars:
            assert f'{var}_mean' in result.columns
            assert f'{var}_max' in result.columns

    def test_zonal_stats_dataset_matches_individual(self, zones_and_values):
        zones, ds = zones_and_values
        merged = zonal_stats(zones, ds)
        for var in ds.data_vars:
            individual = zonal_stats(zones, ds[var])
            for stat_col in individual.columns:
                if stat_col == 'zone':
                    continue
                prefixed = f'{var}_{stat_col}'
                assert prefixed in merged.columns
                pd.testing.assert_series_equal(
                    merged[prefixed].reset_index(drop=True),
                    individual[stat_col].reset_index(drop=True),
                    check_names=False,
                )

    def test_zonal_stats_dataset_return_type_error(self, zones_and_values):
        zones, ds = zones_and_values
        with pytest.raises(ValueError, match="return_type must be 'pandas.DataFrame'"):
            zonal_stats(zones, ds, return_type='xarray.DataArray')
