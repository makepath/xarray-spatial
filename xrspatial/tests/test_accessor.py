"""Tests for the .xrs xarray accessors."""

import numpy as np
import pytest
import xarray as xr

import xrspatial  # noqa: F401 — triggers accessor registration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def elevation():
    """Small synthetic elevation raster."""
    np.random.seed(42)
    data = np.random.rand(10, 10).astype(np.float64) * 1000
    return xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'x': np.arange(10, dtype=float), 'y': np.arange(10, dtype=float)},
        name='elevation',
    )


@pytest.fixture
def nir():
    """Synthetic NIR band."""
    np.random.seed(1)
    return xr.DataArray(
        np.random.rand(10, 10).astype(np.float64),
        dims=['y', 'x'],
        name='nir',
    )


@pytest.fixture
def red():
    """Synthetic red band."""
    np.random.seed(2)
    return xr.DataArray(
        np.random.rand(10, 10).astype(np.float64),
        dims=['y', 'x'],
        name='red',
    )


@pytest.fixture
def blue():
    """Synthetic blue band."""
    np.random.seed(3)
    return xr.DataArray(
        np.random.rand(10, 10).astype(np.float64),
        dims=['y', 'x'],
        name='blue',
    )


# ---------------------------------------------------------------------------
# 1. Accessor registration
# ---------------------------------------------------------------------------

def test_dataarray_accessor_registered():
    da = xr.DataArray(np.zeros((3, 3)), dims=['y', 'x'])
    assert hasattr(da, 'xrs')


def test_dataset_accessor_registered():
    ds = xr.Dataset({'a': xr.DataArray(np.zeros((3, 3)), dims=['y', 'x'])})
    assert hasattr(ds, 'xrs')


# ---------------------------------------------------------------------------
# 2. Tab-completion — expected methods present
# ---------------------------------------------------------------------------

def test_dataarray_accessor_has_expected_methods(elevation):
    names = dir(elevation.xrs)
    expected = [
        'slope', 'aspect', 'hillshade', 'curvature', 'viewshed',
        'natural_breaks', 'equal_interval', 'quantile', 'reclassify',
        'binary', 'percentiles',
        'focal_mean',
        'proximity', 'allocation', 'direction', 'cost_distance',
        'a_star_search',
        'zonal_stats', 'zonal_apply', 'zonal_crosstab', 'crop', 'trim',
        'regions',
        'generate_terrain', 'perlin',
        'ndvi', 'evi', 'arvi', 'savi', 'nbr', 'sipi',
    ]
    for name in expected:
        assert name in names, f"Missing method: {name}"


def test_dataset_accessor_has_expected_methods():
    ds = xr.Dataset({'a': xr.DataArray(np.zeros((3, 3)), dims=['y', 'x'])})
    names = dir(ds.xrs)
    expected = [
        'slope', 'aspect', 'hillshade', 'curvature',
        'natural_breaks', 'equal_interval', 'quantile', 'reclassify',
        'binary', 'percentiles',
        'focal_mean',
        'proximity', 'allocation', 'direction', 'cost_distance',
        'ndvi', 'evi', 'arvi', 'savi', 'nbr', 'sipi',
    ]
    for name in expected:
        assert name in names, f"Missing method: {name}"


# ---------------------------------------------------------------------------
# 3. DataArray single-input — accessor matches direct call
# ---------------------------------------------------------------------------

def test_da_slope(elevation):
    from xrspatial.slope import slope
    expected = slope(elevation)
    result = elevation.xrs.slope()
    xr.testing.assert_identical(result, expected)


def test_da_aspect(elevation):
    from xrspatial.aspect import aspect
    expected = aspect(elevation)
    result = elevation.xrs.aspect()
    xr.testing.assert_identical(result, expected)


def test_da_hillshade(elevation):
    from xrspatial.hillshade import hillshade
    expected = hillshade(elevation, azimuth=315, angle_altitude=45)
    result = elevation.xrs.hillshade(azimuth=315, angle_altitude=45)
    xr.testing.assert_identical(result, expected)


def test_da_curvature(elevation):
    from xrspatial.curvature import curvature
    expected = curvature(elevation)
    result = elevation.xrs.curvature()
    xr.testing.assert_identical(result, expected)


def test_da_focal_mean(elevation):
    from xrspatial.focal import mean
    expected = mean(elevation, passes=1)
    result = elevation.xrs.focal_mean(passes=1)
    xr.testing.assert_identical(result, expected)


def test_da_equal_interval(elevation):
    from xrspatial.classify import equal_interval
    expected = equal_interval(elevation, k=5)
    result = elevation.xrs.equal_interval(k=5)
    xr.testing.assert_identical(result, expected)


def test_da_quantile(elevation):
    from xrspatial.classify import quantile
    expected = quantile(elevation, k=4)
    result = elevation.xrs.quantile(k=4)
    xr.testing.assert_identical(result, expected)


# ---------------------------------------------------------------------------
# 4. DataArray multi-input — multispectral
# ---------------------------------------------------------------------------

def test_da_ndvi(nir, red):
    from xrspatial.multispectral import ndvi
    expected = ndvi(nir, red)
    result = nir.xrs.ndvi(red)
    xr.testing.assert_identical(result, expected)


def test_da_evi(nir, red, blue):
    from xrspatial.multispectral import evi
    expected = evi(nir, red, blue)
    result = nir.xrs.evi(red, blue)
    xr.testing.assert_identical(result, expected)


def test_da_savi(nir, red):
    from xrspatial.multispectral import savi
    expected = savi(nir, red, soil_factor=0.5)
    result = nir.xrs.savi(red, soil_factor=0.5)
    xr.testing.assert_identical(result, expected)


def test_da_nbr(nir):
    np.random.seed(7)
    swir2 = xr.DataArray(
        np.random.rand(10, 10).astype(np.float64), dims=['y', 'x'],
    )
    from xrspatial.multispectral import nbr
    expected = nbr(nir, swir2)
    result = nir.xrs.nbr(swir2)
    xr.testing.assert_identical(result, expected)


def test_da_sipi(nir, red, blue):
    from xrspatial.multispectral import sipi
    expected = sipi(nir, red, blue)
    result = nir.xrs.sipi(red, blue)
    xr.testing.assert_identical(result, expected)


def test_da_arvi(nir, red, blue):
    from xrspatial.multispectral import arvi
    expected = arvi(nir, red, blue)
    result = nir.xrs.arvi(red, blue)
    xr.testing.assert_identical(result, expected)


# ---------------------------------------------------------------------------
# 5. Dataset single-input — accessor matches direct call
# ---------------------------------------------------------------------------

def test_ds_slope(elevation):
    from xrspatial.slope import slope
    ds = xr.Dataset({'elev': elevation})
    expected = slope(ds)
    result = ds.xrs.slope()
    xr.testing.assert_identical(result, expected)


def test_ds_hillshade(elevation):
    from xrspatial.hillshade import hillshade
    ds = xr.Dataset({'elev': elevation})
    expected = hillshade(ds, azimuth=315)
    result = ds.xrs.hillshade(azimuth=315)
    xr.testing.assert_identical(result, expected)


# ---------------------------------------------------------------------------
# 6. Dataset multi-input — multispectral band mapping
# ---------------------------------------------------------------------------

def test_ds_ndvi(nir, red):
    from xrspatial.multispectral import ndvi
    ds = xr.Dataset({'b8': nir, 'b4': red})
    expected = ndvi(ds, nir='b8', red='b4')
    result = ds.xrs.ndvi(nir='b8', red='b4')
    xr.testing.assert_identical(result, expected)


def test_ds_evi(nir, red, blue):
    from xrspatial.multispectral import evi
    ds = xr.Dataset({'b8': nir, 'b4': red, 'b2': blue})
    expected = evi(ds, nir='b8', red='b4', blue='b2')
    result = ds.xrs.evi(nir='b8', red='b4', blue='b2')
    xr.testing.assert_identical(result, expected)
