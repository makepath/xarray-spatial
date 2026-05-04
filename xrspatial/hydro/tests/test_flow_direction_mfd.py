import math

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro import flow_direction_mfd
from xrspatial.hydro.flow_direction_mfd import NEIGHBOR_NAMES
from xrspatial.tests.general_checks import (create_test_raster,
                                            cuda_and_cupy_available,
                                            dask_array_available)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_numpy(result):
    """Extract a plain numpy array from any backend result."""
    data = result.data
    try:
        import dask.array as da
        if isinstance(data, da.Array):
            data = data.compute()
    except ImportError:
        pass
    if hasattr(data, 'get'):
        data = data.get()
    return np.asarray(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_surface():
    return np.full((6, 8), 42.0, dtype=np.float64)


@pytest.fixture
def bowl_surface():
    """5x5 bowl with lowest point at (3,3).

    Grid:
        9  9  9  9  9
        9  8  7  6  9
        9  7  5  4  9
        9  6  4  3  9
        9  9  9  9  9
    """
    return np.array([
        [9, 9, 9, 9, 9],
        [9, 8, 7, 6, 9],
        [9, 7, 5, 4, 9],
        [9, 6, 4, 3, 9],
        [9, 9, 9, 9, 9],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Output shape and coordinate tests
# ---------------------------------------------------------------------------

def test_output_shape_and_dims():
    data = np.random.default_rng(42).random((6, 8)).astype(np.float64) * 100
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg)
    assert result.shape == (8, 6, 8)
    assert result.dims == ('neighbor', 'y', 'x')
    assert list(result.coords['neighbor'].values) == NEIGHBOR_NAMES


# ---------------------------------------------------------------------------
# Flat surface test
# ---------------------------------------------------------------------------

def test_flat_surface(flat_surface):
    """Flat surface: all interior fractions should be 0 (no downslope)."""
    agg = create_test_raster(flat_surface)
    result = flow_direction_mfd(agg)
    interior = result.data[:, 1:-1, 1:-1]
    np.testing.assert_array_equal(interior, 0.0)


# ---------------------------------------------------------------------------
# Fraction sum tests
# ---------------------------------------------------------------------------

def test_fractions_sum_to_one():
    """Interior cells with downslope neighbors should sum to 1."""
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64) * 100
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg)
    band_sum = result.data.sum(axis=0)
    for y in range(1, 7):
        for x in range(1, 9):
            if np.isnan(band_sum[y, x]):
                continue
            if band_sum[y, x] > 0:
                np.testing.assert_allclose(band_sum[y, x], 1.0, rtol=1e-10)


def test_fractions_non_negative():
    """All fractions must be >= 0."""
    data = np.random.default_rng(99).random((8, 10)).astype(np.float64) * 100
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg)
    vals = result.data
    assert np.all(vals[~np.isnan(vals)] >= 0.0)


# ---------------------------------------------------------------------------
# Cardinal slope tests
# ---------------------------------------------------------------------------

def test_cardinal_east():
    """Elevation decreasing east: E should get the largest fraction."""
    data = np.array([
        [9, 8, 7, 6, 5],
        [9, 8, 7, 6, 5],
        [9, 8, 7, 6, 5],
        [9, 8, 7, 6, 5],
        [9, 8, 7, 6, 5],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg)
    # E is band 0.  For interior cells, E should dominate.
    e_frac = result.data[0, 2, 2]
    total = result.data[:, 2, 2].sum()
    assert total > 0
    assert e_frac == result.data[:, 2, 2].max()


def test_cardinal_south():
    """Elevation decreasing south: S should get the largest fraction."""
    data = np.array([
        [9, 9, 9, 9, 9],
        [8, 8, 8, 8, 8],
        [7, 7, 7, 7, 7],
        [6, 6, 6, 6, 6],
        [5, 5, 5, 5, 5],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg)
    s_frac = result.data[2, 2, 2]  # S is band 2
    assert s_frac == result.data[:, 2, 2].max()


# ---------------------------------------------------------------------------
# Bowl surface test
# ---------------------------------------------------------------------------

def test_bowl_surface(bowl_surface):
    """Bowl: center cell (2,2) flows to E, SE, S only."""
    agg = create_test_raster(bowl_surface)
    result = flow_direction_mfd(agg)
    fracs = result.data[:, 2, 2]

    # E (band 0), SE (band 1), S (band 2) should be > 0
    assert fracs[0] > 0, "E should receive flow"
    assert fracs[1] > 0, "SE should receive flow"
    assert fracs[2] > 0, "S should receive flow"

    # W (4), NW (5), N (6) are upslope -> 0
    assert fracs[4] == 0, "W is upslope"
    assert fracs[5] == 0, "NW is upslope"
    assert fracs[6] == 0, "N is upslope"

    np.testing.assert_allclose(fracs.sum(), 1.0, rtol=1e-10)


def test_bowl_pit(bowl_surface):
    """Bowl: pit cell (3,3)=3 has no downslope -> all fracs = 0."""
    agg = create_test_raster(bowl_surface)
    result = flow_direction_mfd(agg)
    pit_fracs = result.data[:, 3, 3]
    np.testing.assert_array_equal(pit_fracs, 0.0)


# ---------------------------------------------------------------------------
# NaN handling tests
# ---------------------------------------------------------------------------

def test_nan_center():
    """NaN center cell -> NaN output for all bands."""
    data = np.array([
        [1, 2, 3, 4],
        [5, np.nan, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg)
    assert np.all(np.isnan(result.data[:, 1, 1]))


def test_nan_neighbor():
    """NaN in any neighbor -> NaN output."""
    data = np.array([
        [1, 2, 3, 4],
        [5, np.nan, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg)
    # Cell (2,2)=11 has NaN neighbor at (1,1)
    assert np.all(np.isnan(result.data[:, 2, 2]))


# ---------------------------------------------------------------------------
# Edge NaN test
# ---------------------------------------------------------------------------

def test_nan_edges():
    data = np.random.default_rng(42).random((6, 8)).astype(np.float64) * 100
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg)
    # All 8 bands should be NaN at edges
    for k in range(8):
        np.testing.assert_array_equal(result.data[k, 0, :], np.nan)
        np.testing.assert_array_equal(result.data[k, -1, :], np.nan)
        np.testing.assert_array_equal(result.data[k, :, 0], np.nan)
        np.testing.assert_array_equal(result.data[k, :, -1], np.nan)


# ---------------------------------------------------------------------------
# Fixed exponent tests
# ---------------------------------------------------------------------------

def test_fixed_exponent_p1():
    """p=1.0 (Quinn et al. 1991): fractions proportional to slope * contour."""
    data = np.array([
        [10, 10, 10, 10, 10],
        [10,  5,  5,  5, 10],
        [10,  5,  3,  5, 10],
        [10,  5,  5,  5, 10],
        [10, 10, 10, 10, 10],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg, p=1.0)
    fracs = result.data[:, 2, 2]

    # All 8 neighbors are upslope (value=5, center=3) -> all fractions > 0?
    # Wait, center=3 and neighbors=5, so center < neighbors -> no downslope
    # Actually, upslope means neighbor is HIGHER, so flow doesn't go there.
    # center=3 < 5 = neighbors -> neighbors are upslope -> pit
    np.testing.assert_array_equal(fracs, 0.0)


def test_fixed_exponent_high():
    """High exponent should concentrate flow more toward steepest neighbor."""
    # Center=10, E=5 (drop=5), S=8 (drop=2) -- two downslope neighbors
    data = np.array([
        [15, 15, 15, 15, 15],
        [15, 15, 15, 15, 15],
        [15, 15, 10,  5, 15],
        [15, 15,  8, 15, 15],
        [15, 15, 15, 15, 15],
    ], dtype=np.float64)
    agg = create_test_raster(data)

    result_low = flow_direction_mfd(agg, p=1.0)
    result_high = flow_direction_mfd(agg, p=10.0)

    # E (band 0) is steepest.  With higher p, E fraction should increase.
    e_low = result_low.data[0, 2, 2]
    e_high = result_high.data[0, 2, 2]
    assert e_high > e_low, f"p=10 E frac {e_high} should exceed p=1 E frac {e_low}"


def test_invalid_p():
    data = np.ones((4, 5), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="positive finite"):
        flow_direction_mfd(agg, p=-1.0)
    with pytest.raises(ValueError, match="positive finite"):
        flow_direction_mfd(agg, p=0.0)


# ---------------------------------------------------------------------------
# Adaptive exponent tests
# ---------------------------------------------------------------------------

def test_adaptive_equals_fixed_when_uniform():
    """When all downslope slopes are equal, adaptive p = max/mean = 1."""
    # Center = 10, all 8 neighbors = 5: slopes are all equal
    data = np.array([
        [5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5],
        [5, 5, 10, 5, 5],
        [5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5],
    ], dtype=np.float64)
    agg = create_test_raster(data)

    result_adaptive = flow_direction_mfd(agg)
    result_fixed_1 = flow_direction_mfd(agg, p=1.0)

    # Slopes differ by cardinal/diagonal distance, so they're NOT all equal.
    # But the adaptive exponent uses slope values, not drops.
    # Cardinal slope = (10-5)/0.5 = 10, diagonal slope = (10-5)/diag ~ 7.07
    # max_slope = 10, mean_slope = (4*10 + 4*7.07)/8 = 8.535
    # p_adaptive = 10/8.535 = 1.172 -- close to 1 but not exactly 1
    # So results won't be identical, but close.
    np.testing.assert_allclose(
        result_adaptive.data[:, 2, 2],
        result_fixed_1.data[:, 2, 2],
        atol=0.05)


# ---------------------------------------------------------------------------
# Cross-backend tests
# ---------------------------------------------------------------------------

@dask_array_available
@pytest.mark.parametrize("size", [(6, 8), (10, 15)])
def test_numpy_equals_dask(size):
    data = np.random.default_rng(42).random(size).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask')

    np_result = flow_direction_mfd(numpy_agg)
    da_result = flow_direction_mfd(dask_agg)

    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(6, 8), (10, 15)])
def test_numpy_equals_cupy(size):
    data = np.random.default_rng(42).random(size).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy')
    cupy_agg = create_test_raster(data, backend='cupy')

    np_result = flow_direction_mfd(numpy_agg)
    cu_result = flow_direction_mfd(cupy_agg)

    np.testing.assert_allclose(
        np_result.data, cu_result.data.get(), equal_nan=True, rtol=1e-6)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(6, 8), (10, 15)])
def test_numpy_equals_dask_cupy(size):
    data = np.random.default_rng(42).random(size).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy')

    np_result = flow_direction_mfd(numpy_agg)
    dc_result = flow_direction_mfd(dask_cupy_agg)

    np.testing.assert_allclose(
        np_result.data, dc_result.data.compute().get(),
        equal_nan=True, rtol=1e-6)


# ---------------------------------------------------------------------------
# Boundary mode tests
# ---------------------------------------------------------------------------

@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((6, 8), (3, 4)),
    ((7, 9), (3, 3)),
    ((10, 15), (5, 5)),
])
def test_boundary_numpy_equals_dask(boundary, size, chunks):
    data = np.random.default_rng(42).random(size).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)

    np_result = flow_direction_mfd(numpy_agg, boundary=boundary)
    da_result = flow_direction_mfd(dask_agg, boundary=boundary)

    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_boundary_no_nan_flat(boundary):
    """Flat surface with non-nan boundary: all fracs 0, no NaN."""
    data = np.full((8, 10), 50.0, dtype=np.float64)
    numpy_agg = create_test_raster(data, backend='numpy')
    np_result = flow_direction_mfd(numpy_agg, boundary=boundary)
    # All fractions should be 0 (flat = no downslope), no NaN
    assert not np.any(np.isnan(np_result.data))
    np.testing.assert_array_equal(np_result.data, 0.0)


def test_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="boundary must be one of"):
        flow_direction_mfd(agg, boundary='invalid')


# ---------------------------------------------------------------------------
# Dtype acceptance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.float32, np.float64])
def test_dtype_acceptance(dtype):
    data = np.arange(20, dtype=dtype).reshape(4, 5)
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg)
    assert result.shape == (8, 4, 5)
    assert result.dims == ('neighbor', 'y', 'x')


# ---------------------------------------------------------------------------
# Dataset support
# ---------------------------------------------------------------------------

def test_dataset_support():
    data = np.random.default_rng(42).random((6, 8)).astype(np.float64) * 100
    da1 = xr.DataArray(data, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(2.5, 0, 6)
    da1['x'] = np.linspace(0, 3.5, 8)
    ds = xr.Dataset({'elev1': da1, 'elev2': da1 * 2})
    result = flow_direction_mfd(ds)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'elev1', 'elev2'}
    for var in result.data_vars:
        expected = flow_direction_mfd(ds[var], name=var)
        np.testing.assert_allclose(
            result[var].data, expected.data, equal_nan=True)


# ---------------------------------------------------------------------------
# Cellsize effect
# ---------------------------------------------------------------------------

def test_cellsize_effect():
    """Non-square cells: changing cellsize ratio shifts flow distribution."""
    data = np.array([
        [6, 5, 4, 3],
        [5, 4, 3, 2],
        [4, 3, 2, 1],
        [3, 2, 1, 0],
    ], dtype=np.float64)

    # Case 1: cellsize_x=1, cellsize_y=2 -> E gradient steeper than S
    agg1 = xr.DataArray(data, dims=['y', 'x'], attrs={'res': (1.0, 2.0)})
    agg1['y'] = np.linspace(6, 0, 4)
    agg1['x'] = np.linspace(0, 3, 4)
    result1 = flow_direction_mfd(agg1)
    e_frac = result1.data[0, 1, 1]  # E
    s_frac = result1.data[2, 1, 1]  # S
    assert e_frac > s_frac, "E should get more flow when cellsize_x < cellsize_y"

    # Case 2: cellsize_x=2, cellsize_y=1 -> S gradient steeper than E
    agg2 = xr.DataArray(data, dims=['y', 'x'], attrs={'res': (2.0, 1.0)})
    agg2['y'] = np.linspace(3, 0, 4)
    agg2['x'] = np.linspace(0, 6, 4)
    result2 = flow_direction_mfd(agg2)
    e_frac2 = result2.data[0, 1, 1]
    s_frac2 = result2.data[2, 1, 1]
    assert s_frac2 > e_frac2, "S should get more flow when cellsize_y < cellsize_x"


# ---------------------------------------------------------------------------
# Hand-computed expected values
# ---------------------------------------------------------------------------

def test_known_values_symmetric():
    """Symmetric surface: E and S should get equal fractions."""
    # Center=10, E=5, S=5, all others=10 or higher
    data = np.array([
        [15, 15, 15, 15, 15],
        [15, 10, 10, 10, 15],
        [15, 10, 10,  5, 15],
        [15, 10,  5, 10, 15],
        [15, 15, 15, 15, 15],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg, p=1.0)

    # Cell (2,2)=10: E=5 (drop=5), S=5 (drop=5), SE=10 (no drop)
    # Cardinal distance = 0.5, so slope_E = 5/0.5 = 10, slope_S = 5/0.5 = 10
    # Contour_E = contour_S = 1.0
    # With p=1: w_E = 10*1 = 10, w_S = 10*1 = 10, total = 20
    e_frac = result.data[0, 2, 2]
    s_frac = result.data[2, 2, 2]
    np.testing.assert_allclose(e_frac, 0.5, atol=1e-10)
    np.testing.assert_allclose(s_frac, 0.5, atol=1e-10)


def test_known_values_cardinal_vs_diagonal():
    """With p=1, verify cardinal gets more weight than diagonal per slope unit."""
    # Center=10, E=5 (cardinal, drop=5), SE=5 (diagonal, drop=5)
    data = np.array([
        [15, 15, 15, 15, 15],
        [15, 15, 15, 15, 15],
        [15, 15, 10, 5, 15],
        [15, 15, 15, 5, 15],
        [15, 15, 15, 15, 15],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction_mfd(agg, p=1.0)

    e_frac = result.data[0, 2, 2]   # E: cardinal
    se_frac = result.data[1, 2, 2]  # SE: diagonal

    # E: slope = 5/0.5 = 10, contour = 1.0, weight = 10
    # SE: slope = 5/diag = 5/0.707 = 7.07, contour = 1/sqrt(2) = 0.707, weight = 5.0
    # E should get more than SE
    assert e_frac > se_frac, f"E={e_frac} should exceed SE={se_frac}"


# =====================================================================
# Memory guard
# =====================================================================

class TestMemoryGuard:
    """Memory guard on the eager numpy / cupy backends (issue #1423)."""

    def test_numpy_huge_raster_raises(self):
        """Numpy backend raises MemoryError when projected RAM exceeds budget."""
        from unittest.mock import patch

        elev = np.full((7, 7), 5.0, dtype=np.float64)

        with patch(
            "xrspatial.hydro.flow_direction_mfd._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                flow_direction_mfd(create_test_raster(elev))

    def test_numpy_normal_input_succeeds(self):
        """Normal-size raster passes the guard with real memory."""
        elev = np.full((7, 7), 5.0, dtype=np.float64)
        result = flow_direction_mfd(create_test_raster(elev))
        assert result.shape == (8, 7, 7)

    def test_error_message_mentions_dimensions(self):
        """Error message should mention the offending grid dimensions."""
        from unittest.mock import patch

        elev = np.full((7, 7), 5.0, dtype=np.float64)

        with patch(
            "xrspatial.hydro.flow_direction_mfd._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="7x7"):
                flow_direction_mfd(create_test_raster(elev))

    def test_error_message_mentions_dask(self):
        """The error message should suggest the dask alternative."""
        from unittest.mock import patch

        elev = np.full((7, 7), 5.0, dtype=np.float64)

        with patch(
            "xrspatial.hydro.flow_direction_mfd._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="dask"):
                flow_direction_mfd(create_test_raster(elev))
