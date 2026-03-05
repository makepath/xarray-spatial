import numpy as np
import pytest
import xarray as xr

from xrspatial import sky_view_factor
from xrspatial.tests.general_checks import (
    assert_numpy_equals_cupy,
    assert_numpy_equals_dask_cupy,
    assert_numpy_equals_dask_numpy,
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
    general_output_checks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_surface():
    """Constant elevation -- SVF should be 1.0 everywhere in the interior."""
    return np.full((30, 30), 100.0, dtype=np.float64)


@pytest.fixture
def valley_surface():
    """V-shaped valley along the x axis.

    Cross-section:  z = abs(x - center) * slope_factor
    Cells at the valley floor have obstructed sky on both sides,
    so SVF < 1.
    """
    rows, cols = 40, 40
    x = np.arange(cols, dtype=np.float64)
    z = np.abs(x - cols / 2.0) * 5.0
    return np.broadcast_to(z, (rows, cols)).copy()


@pytest.fixture
def ridge_surface():
    """Cone-shaped peak in the center.

    Cells at the summit have unobstructed sky (SVF close to 1),
    cells at the base have partial obstruction.
    """
    rows, cols = 50, 50
    y, x = np.mgrid[:rows, :cols].astype(np.float64)
    cy, cx = rows / 2.0, cols / 2.0
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    return np.maximum(0.0, 200.0 - dist * 10.0)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_flat_surface_svf(flat_surface):
    """Flat terrain: no horizon obstruction, SVF should be 1.0."""
    agg = create_test_raster(flat_surface)
    result = sky_view_factor(agg, max_radius=5, n_directions=16)
    general_output_checks(agg, result)
    interior = result.data[5:-5, 5:-5]
    np.testing.assert_allclose(interior, 1.0, atol=1e-10)


def test_valley_svf_less_than_one(valley_surface):
    """Valley floor cells should have SVF < 1 due to obstruction."""
    agg = create_test_raster(valley_surface)
    result = sky_view_factor(agg, max_radius=8, n_directions=16)
    # Valley floor at col=20, interior rows
    floor_svf = result.data[20, 20]
    assert np.isfinite(floor_svf)
    assert floor_svf < 1.0, f"Expected SVF < 1 at valley floor, got {floor_svf}"


def test_ridge_summit_high_svf(ridge_surface):
    """Summit of a cone has nearly unobstructed sky."""
    agg = create_test_raster(ridge_surface)
    result = sky_view_factor(agg, max_radius=10, n_directions=16)
    summit_svf = result.data[25, 25]
    assert np.isfinite(summit_svf)
    assert summit_svf > 0.9, f"Expected SVF > 0.9 at summit, got {summit_svf}"


def test_svf_range():
    """All valid SVF values should be in [0, 1]."""
    rng = np.random.default_rng(962)
    data = rng.random((30, 30)).astype(np.float64) * 500
    agg = create_test_raster(data)
    result = sky_view_factor(agg, max_radius=5, n_directions=8)
    valid = result.data[~np.isnan(result.data)]
    assert np.all(valid >= 0.0), "SVF values below 0"
    assert np.all(valid <= 1.0), "SVF values above 1"


def test_more_directions_same_flat(flat_surface):
    """Different n_directions on a flat surface should still give SVF=1."""
    agg = create_test_raster(flat_surface)
    for n_dir in [4, 8, 16, 32]:
        result = sky_view_factor(agg, max_radius=5, n_directions=n_dir)
        interior = result.data[5:-5, 5:-5]
        np.testing.assert_allclose(interior, 1.0, atol=1e-10,
                                   err_msg=f"Failed for n_directions={n_dir}")


# ---------------------------------------------------------------------------
# Edge / NaN handling
# ---------------------------------------------------------------------------

def test_edge_cells_computed():
    """Edge cells should be computed (with truncated rays), not NaN."""
    data = np.random.default_rng(962).random((20, 20)).astype(np.float64) * 100
    agg = create_test_raster(data)
    radius = 3
    result = sky_view_factor(agg, max_radius=radius, n_directions=8)
    # All cells should have valid values (no NaN except where input is NaN)
    assert np.all(np.isfinite(result.data))


def test_nan_in_input():
    """NaN in the DEM: center cell NaN should produce NaN output."""
    data = np.random.default_rng(962).random((20, 20)).astype(np.float64) * 100
    data[10, 10] = np.nan
    agg = create_test_raster(data)
    result = sky_view_factor(agg, max_radius=3, n_directions=8)
    assert np.isnan(result.data[10, 10])


def test_single_cell_raster():
    """A 1x1 raster should return SVF=1 (no neighbors to obstruct)."""
    data = np.array([[42.0]])
    agg = create_test_raster(data)
    result = sky_view_factor(agg, max_radius=1, n_directions=4)
    np.testing.assert_allclose(result.data[0, 0], 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_invalid_max_radius():
    data = np.ones((10, 10), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="max_radius"):
        sky_view_factor(agg, max_radius=0)


def test_invalid_n_directions():
    data = np.ones((10, 10), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="n_directions"):
        sky_view_factor(agg, n_directions=0)


def test_invalid_input_type():
    with pytest.raises(TypeError, match="DataArray"):
        sky_view_factor(np.ones((5, 5)))


# ---------------------------------------------------------------------------
# Cross-backend tests
# ---------------------------------------------------------------------------

@dask_array_available
@pytest.mark.parametrize("size", [(20, 20), (30, 40)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_equals_dask(random_data, size, dtype):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask', chunks=(10, 10))
    func = lambda agg: sky_view_factor(agg, max_radius=3, n_directions=8)
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, func, nan_edges=False)


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(20, 20), (30, 40)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_equals_cupy(random_data, size, dtype):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    cupy_agg = create_test_raster(random_data, backend='cupy')
    func = lambda agg: sky_view_factor(agg, max_radius=3, n_directions=8)
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, func, nan_edges=False)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(20, 20), (30, 40)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_equals_dask_cupy(random_data, size, dtype):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy',
                                       chunks=(10, 10))
    func = lambda agg: sky_view_factor(agg, max_radius=3, n_directions=8)
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, func,
                                  nan_edges=False, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Dataset support
# ---------------------------------------------------------------------------

def test_dataset_support():
    data = np.random.default_rng(962).random((20, 20)).astype(np.float64) * 100
    da1 = xr.DataArray(data, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(9.5, 0, 20)
    da1['x'] = np.linspace(0, 9.5, 20)
    ds = xr.Dataset({'elev1': da1, 'elev2': da1 * 2})
    result = sky_view_factor(ds, max_radius=3, n_directions=8)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'elev1', 'elev2'}
    for var in result.data_vars:
        expected = sky_view_factor(ds[var], max_radius=3, n_directions=8,
                                   name=var)
        np.testing.assert_allclose(
            result[var].data, expected.data, equal_nan=True)


# ---------------------------------------------------------------------------
# Known reference value test
# ---------------------------------------------------------------------------

def test_known_svf_simple_ramp():
    """A tilted plane: SVF should be < 1 for cells looking uphill.

    For a plane tilted at 45 degrees (z = x), the horizon angle
    along the uphill direction is 45 degrees, so SVF should be
    strictly less than 1 at interior cells.
    """
    rows, cols = 30, 30
    x = np.arange(cols, dtype=np.float64)
    data = np.broadcast_to(x * 1.0, (rows, cols)).copy()
    agg = create_test_raster(data)
    result = sky_view_factor(agg, max_radius=5, n_directions=16)
    center = result.data[15, 15]
    assert np.isfinite(center)
    assert center < 1.0, f"Expected SVF < 1 on tilted plane, got {center}"
    assert center > 0.5, f"Expected SVF > 0.5 on gentle tilt, got {center}"
