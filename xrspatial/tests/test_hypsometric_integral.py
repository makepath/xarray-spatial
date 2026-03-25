# xrspatial/tests/test_hypsometric_integral.py
try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from .general_checks import create_test_raster


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def hi_zones(backend):
    """Two zones (1, 2) plus nodata (0).

    Zone 1: 5 cells — (0,1), (0,2), (0,3), (1,1), (1,2)
    Zone 2: 6 cells — (1,3), (2,1), (2,2), (2,3), (3,1), (3,2)
    Nodata: 4 cells — column 0 and (3,3)
    """
    data = np.array([
        [0, 1, 1, 1],
        [0, 1, 1, 2],
        [0, 2, 2, 2],
        [0, 2, 2, 0],
    ], dtype=np.float64)
    return create_test_raster(data, backend, dims=['y', 'x'],
                              attrs={'res': (1.0, 1.0)}, chunks=(2, 2))


@pytest.fixture
def hi_values(backend):
    """Elevation values.

    Zone 1 cells: 10, 20, 30, 40, 50
      min=10, max=50, mean=30, HI=(30-10)/(50-10) = 0.5

    Zone 2 cells: 100, 60, 70, 80, 90, 95
      min=60, max=100, mean=82.5, HI=(82.5-60)/(100-60) = 0.5625
    """
    data = np.array([
        [999., 10., 20., 30.],
        [999., 40., 50., 100.],
        [999., 60., 70., 80.],
        [999., 90., 95., 999.],
    ], dtype=np.float64)
    return create_test_raster(data, backend, dims=['y', 'x'],
                              attrs={'res': (1.0, 1.0)}, chunks=(2, 2))


# --- basic -------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_basic(backend, hi_zones, hi_values):
    from xrspatial.zonal import hypsometric_integral

    result = hypsometric_integral(hi_zones, hi_values)

    assert isinstance(result, xr.DataArray)
    assert result.shape == hi_values.shape
    assert result.dims == hi_values.dims
    assert result.name == 'hypsometric_integral'

    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values

    # zone 0 (nodata) cells should be NaN
    nodata_mask = np.array([
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, True],
    ])
    assert np.all(np.isnan(out[nodata_mask]))

    # zone 1: HI = 0.5
    z1_mask = np.array([
        [False, True, True, True],
        [False, True, True, False],
        [False, False, False, False],
        [False, False, False, False],
    ])
    np.testing.assert_allclose(out[z1_mask], 0.5, rtol=1e-10)

    # zone 2: HI = 0.5625
    z2_mask = np.array([
        [False, False, False, False],
        [False, False, False, True],
        [False, True, True, True],
        [False, True, True, False],
    ])
    np.testing.assert_allclose(out[z2_mask], 0.5625, rtol=1e-10)


# --- edge cases --------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_flat_zone(backend):
    """A zone with all identical values has range=0, so HI should be NaN."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 1], [1, 1]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[5.0, 5.0], [5.0, 5.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_nan_in_values(backend):
    """NaN elevation cells should be excluded from per-zone stats."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 1], [1, 1]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[10.0, np.nan], [20.0, 30.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values

    # zone 1 finite values: 10, 20, 30 -> HI = (20-10)/(30-10) = 0.5
    # the NaN cell should remain NaN in output
    assert np.isnan(out[0, 1])
    np.testing.assert_allclose(out[0, 0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(out[1, 0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(out[1, 1], 0.5, rtol=1e-10)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_single_cell_zone(backend):
    """A zone with a single cell has range=0, so HI=NaN."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 2]], dtype=np.float64), backend,
        chunks=(1, 2))
    values = create_test_raster(
        np.array([[10.0, 20.0]]), backend,
        chunks=(1, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values
    # single cell -> range=0 -> NaN
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_all_nan_zone(backend):
    """A zone whose elevation cells are all NaN should produce NaN."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 1], [2, 2]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[np.nan, np.nan], [10.0, 20.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values

    # zone 1: all NaN -> NaN
    assert np.all(np.isnan(out[0, :]))
    # zone 2: HI = (15-10)/(20-10) = 0.5
    np.testing.assert_allclose(out[1, :], 0.5, rtol=1e-10)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_nodata_none(backend):
    """When nodata=None, all zone IDs are included (even 0)."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[0, 0], [1, 1]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[10.0, 20.0], [30.0, 40.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=None)
    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values

    # zone 0: HI = (15-10)/(20-10) = 0.5
    np.testing.assert_allclose(out[0, :], 0.5, rtol=1e-10)
    # zone 1: HI = (35-30)/(40-30) = 0.5
    np.testing.assert_allclose(out[1, :], 0.5, rtol=1e-10)
