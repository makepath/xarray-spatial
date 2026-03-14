import math

import numpy as np
import pytest
import xarray as xr

from xrspatial import flow_direction_dinf
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy,
                                            create_test_raster,
                                            cuda_and_cupy_available,
                                            dask_array_available,
                                            general_output_checks)


TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Known-angle tests (square cells, cx=cy=1)
# ---------------------------------------------------------------------------

def _make_raster(data, res=(1.0, 1.0)):
    """Create a test DataArray with given resolution."""
    data = np.asarray(data, dtype=np.float64)
    agg = xr.DataArray(data, dims=['y', 'x'], attrs={'res': res})
    agg['y'] = np.linspace((data.shape[0] - 1) * res[0], 0, data.shape[0])
    agg['x'] = np.linspace(0, (data.shape[1] - 1) * res[1], data.shape[1])
    return agg


def test_pure_east_slope():
    """Each column 1 unit lower going east -> angle 0 for all interior."""
    data = np.array([
        [5, 4, 3, 2, 1],
        [5, 4, 3, 2, 1],
        [5, 4, 3, 2, 1],
        [5, 4, 3, 2, 1],
        [5, 4, 3, 2, 1],
    ], dtype=np.float64)
    agg = _make_raster(data)
    result = flow_direction_dinf(agg)
    for y in range(1, 4):
        for x in range(1, 4):
            assert abs(result.data[y, x] - 0.0) < 1e-10, (
                f"Cell ({y},{x}) = {result.data[y,x]}")


def test_pure_north_slope():
    """Each row 1 unit lower going up (decreasing row index) -> angle pi/2."""
    data = np.array([
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5],
    ], dtype=np.float64)
    agg = _make_raster(data)
    result = flow_direction_dinf(agg)
    expected = math.pi / 2.0
    for y in range(1, 4):
        for x in range(1, 4):
            assert abs(result.data[y, x] - expected) < 1e-10, (
                f"Cell ({y},{x}) = {result.data[y,x]}")


def test_pure_ne_slope():
    """Diagonal gradient toward NE -> angle pi/4."""
    # Elevation = -(row + col), so NE (row-1, col+1) is always lowest
    # relative to center along the NE diagonal
    data = np.zeros((5, 5), dtype=np.float64)
    for r in range(5):
        for c in range(5):
            data[r, c] = r - c  # decreases going up and right
    agg = _make_raster(data)
    result = flow_direction_dinf(agg)
    expected = math.pi / 4.0
    for y in range(1, 4):
        for x in range(1, 4):
            assert abs(result.data[y, x] - expected) < 1e-10, (
                f"Cell ({y},{x}) = {result.data[y,x]}")


def test_pure_se_slope():
    """Diagonal gradient toward SE -> angle 7*pi/4."""
    data = np.zeros((5, 5), dtype=np.float64)
    for r in range(5):
        for c in range(5):
            data[r, c] = -r - c  # decreases going down and right
    agg = _make_raster(data)
    result = flow_direction_dinf(agg)
    expected = 7.0 * math.pi / 4.0
    for y in range(1, 4):
        for x in range(1, 4):
            assert abs(result.data[y, x] - expected) < 1e-10, (
                f"Cell ({y},{x}) = {result.data[y,x]}")


def test_interior_facet_angle():
    """Test that a specific 3x3 window gives the expected sub-facet angle.

    E neighbor 2 lower, NE neighbor 3 lower, rest equal to center.
    Facet 0: e1=E, e2=NE, d1=cx=1, d2=cy=1
      s1 = (10-8)/1 = 2, s2 = (8-7)/1 = 1
      r = atan2(1, 2) ~= 0.4636 rad
      s = sqrt(4 + 1) ~= 2.236
    All other facets have zero or negative slope.
    """
    data = np.array([
        [10, 10,  7],
        [10, 10,  8],
        [10, 10, 10],
    ], dtype=np.float64)
    agg = _make_raster(data)
    result = flow_direction_dinf(agg)
    expected = math.atan2(1, 2)
    assert abs(result.data[1, 1] - expected) < 1e-10, (
        f"Center = {result.data[1,1]}, expected {expected}")


# ---------------------------------------------------------------------------
# Tarboton odd-facet tests
# ---------------------------------------------------------------------------

def test_odd_facet_1_tarboton():
    """Facet 1 (N, NE) wins with interior r; angle = pi/2 - r."""
    # Center=10, N=7, NE=6; all others equal to center.
    # Facet 1: s1=(10-7)/1=3, s2=(7-6)/1=1, r=atan2(1,3), s=sqrt(10)
    # Facet 0: s1=(10-10)/1=0, s2=(10-6)/1=4, r clamped pi/4, s=(10-6)/sqrt(2)~2.83
    # Facet 1 wins (sqrt(10)~3.16 > 2.83).
    data = np.array([
        [10, 7,  6],
        [10, 10, 10],
        [10, 10, 10],
    ], dtype=np.float64)
    agg = _make_raster(data)
    result = flow_direction_dinf(agg)
    r = math.atan2(1, 3)
    expected = math.pi / 2.0 - r
    assert abs(result.data[1, 1] - expected) < 1e-10, (
        f"Center = {result.data[1,1]}, expected {expected}")


def test_odd_facet_7_tarboton():
    """Facet 7 (E, SE) wins with interior r; angle = 2*pi - r."""
    # Center=10, E=7, SE=6; all others equal to center.
    # Facet 7: s1=(10-7)/1=3, s2=(7-6)/1=1, r=atan2(1,3), s=sqrt(10)
    data = np.array([
        [10, 10, 10],
        [10, 10,  7],
        [10, 10,  6],
    ], dtype=np.float64)
    agg = _make_raster(data)
    result = flow_direction_dinf(agg)
    r = math.atan2(1, 3)
    expected = 2.0 * math.pi - r
    assert abs(result.data[1, 1] - expected) < 1e-10, (
        f"Center = {result.data[1,1]}, expected {expected}")


def _reference_tarboton(data, cx, cy):
    """Pure-Python reference Tarboton (1997) D-inf implementation."""
    rows, cols = data.shape
    out = np.full(data.shape, np.nan)

    nb_dy = [0, -1, -1, -1, 0, 1, 1, 1]
    nb_dx = [1, 1, 0, -1, -1, -1, 0, 1]
    e1_idx = [0, 2, 2, 4, 4, 6, 6, 0]
    e2_idx = [1, 1, 3, 3, 5, 5, 7, 7]
    d1_arr = [cx, cy, cy, cx, cx, cy, cy, cx]
    d2_arr = [cy, cx, cx, cy, cy, cx, cx, cy]
    ac = [0, 2, 2, 4, 4, 6, 6, 8]
    af = [1, -1, 1, -1, 1, -1, 1, -1]

    diag = math.sqrt(cx * cx + cy * cy)
    pi4 = math.pi / 4.0

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            center = data[y, x]
            if np.isnan(center):
                continue
            has_nan = False
            nbs = []
            for k in range(8):
                v = data[y + nb_dy[k], x + nb_dx[k]]
                if np.isnan(v):
                    has_nan = True
                    break
                nbs.append(v)
            if has_nan:
                continue

            max_slope = -1e308
            best_angle = -1.0
            for k in range(8):
                e1 = nbs[e1_idx[k]]
                e2 = nbs[e2_idx[k]]
                s1 = (center - e1) / d1_arr[k]
                s2 = (e1 - e2) / d2_arr[k]
                r = math.atan2(s2, s1)
                if r < 0.0:
                    r = 0.0
                    s = s1
                elif r > pi4:
                    r = pi4
                    s = (center - e2) / diag
                else:
                    s = math.sqrt(s1 * s1 + s2 * s2)
                if s > max_slope:
                    max_slope = s
                    best_angle = ac[k] * pi4 + af[k] * r

            if max_slope <= 0.0:
                out[y, x] = -1.0
            else:
                if best_angle >= 2.0 * math.pi:
                    best_angle = 0.0
                out[y, x] = best_angle
    return out


def test_reference_tarboton_agreement():
    """Cone DEM: xrspatial matches reference Tarboton on all interior cells."""
    n = 51
    cx, cy = 1.0, 1.0
    yy, xx = np.mgrid[0:n, 0:n]
    center = (n - 1) / 2.0
    data = np.sqrt((yy - center) ** 2 + (xx - center) ** 2).astype(np.float64)

    agg = _make_raster(data, res=(cy, cx))
    result = flow_direction_dinf(agg)
    ref = _reference_tarboton(data, cx, cy)

    # Compare interior cells (skip edges which are NaN)
    interior = ~np.isnan(ref)
    assert interior.sum() > 0
    np.testing.assert_allclose(
        result.data[interior], ref[interior], atol=1e-12,
        err_msg="xrspatial dinf does not match reference Tarboton")


# ---------------------------------------------------------------------------
# Pit / flat / NaN tests
# ---------------------------------------------------------------------------

def test_flat_surface():
    """All equal elevation -> all interior cells -1.0."""
    data = np.full((6, 8), 42.0, dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction_dinf(agg)
    interior = result.data[1:-1, 1:-1]
    np.testing.assert_array_equal(interior, -1.0)


def test_pit_detection():
    """Local minimum surrounded by higher cells -> -1.0."""
    data = np.array([
        [5, 5, 5, 5, 5],
        [5, 3, 3, 3, 5],
        [5, 3, 1, 3, 5],
        [5, 3, 3, 3, 5],
        [5, 5, 5, 5, 5],
    ], dtype=np.float64)
    agg = _make_raster(data)
    result = flow_direction_dinf(agg)
    assert result.data[2, 2] == -1.0


def test_nan_center():
    """NaN center cell -> NaN output."""
    data = np.array([
        [1, 2, 3, 4],
        [5, np.nan, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction_dinf(agg)
    assert np.isnan(result.data[1, 1])


def test_nan_neighbor():
    """NaN in any neighbor of the 3x3 window -> NaN output."""
    data = np.array([
        [1, 2, 3, 4],
        [5, np.nan, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction_dinf(agg)
    # Cell (1,2)=7 has NaN neighbor at (1,1) -> NaN
    assert np.isnan(result.data[1, 2])
    # Cell (2,1)=10 has NaN neighbor at (1,1) -> NaN
    assert np.isnan(result.data[2, 1])
    # Cell (2,2)=11 has NaN neighbor at (1,1) -> NaN
    assert np.isnan(result.data[2, 2])


def test_nan_edges():
    """boundary='nan' -> edge cells NaN."""
    data = np.random.default_rng(42).random((6, 8)).astype(np.float64) * 100
    agg = create_test_raster(data)
    result = flow_direction_dinf(agg)
    np.testing.assert_array_equal(result.data[0, :], np.nan)
    np.testing.assert_array_equal(result.data[-1, :], np.nan)
    np.testing.assert_array_equal(result.data[:, 0], np.nan)
    np.testing.assert_array_equal(result.data[:, -1], np.nan)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_output_range():
    """All values in {NaN, -1.0, [0, 2*pi)}."""
    data = np.random.default_rng(123).random((10, 12)).astype(np.float64) * 100
    agg = create_test_raster(data)
    result = flow_direction_dinf(agg)
    vals = result.data.ravel()
    for v in vals:
        if np.isnan(v):
            continue
        if v == -1.0:
            continue
        assert 0.0 <= v < TWO_PI, f"Value {v} out of range [0, 2*pi)"


def test_output_dtype():
    """Output should be float64."""
    data = np.random.default_rng(42).random((6, 8)).astype(np.float64) * 100
    agg = create_test_raster(data)
    result = flow_direction_dinf(agg)
    assert result.data.dtype == np.float64


def test_dataset_support():
    """@supports_dataset works correctly."""
    data = np.random.default_rng(42).random((6, 8)).astype(np.float64) * 100
    da1 = xr.DataArray(data, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(2.5, 0, 6)
    da1['x'] = np.linspace(0, 3.5, 8)
    ds = xr.Dataset({'elev1': da1, 'elev2': da1 * 2})
    result = flow_direction_dinf(ds)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'elev1', 'elev2'}
    for var in result.data_vars:
        expected = flow_direction_dinf(ds[var], name=var)
        np.testing.assert_allclose(
            result[var].data, expected.data, equal_nan=True)


# ---------------------------------------------------------------------------
# Cross-backend tests
# ---------------------------------------------------------------------------

@dask_array_available
@pytest.mark.parametrize("size", [(4, 6), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_numpy_equals_dask(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, flow_direction_dinf)


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(4, 6), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_numpy_equals_cupy(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    cupy_agg = create_test_raster(random_data, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, flow_direction_dinf)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(4, 6), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_numpy_equals_dask_cupy(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg,
                                  flow_direction_dinf,
                                  atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Boundary mode tests
# ---------------------------------------------------------------------------

@dask_array_available
def test_boundary_modes():
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64) * 100
    numpy_agg = create_test_raster(data)
    dask_agg = create_test_raster(data, backend='dask+numpy')
    assert_boundary_mode_correctness(numpy_agg, dask_agg,
                                     flow_direction_dinf)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((6, 8), (3, 4)),
    ((7, 9), (3, 3)),
    ((10, 15), (5, 5)),
    ((10, 15), (10, 15)),
    ((5, 5), (2, 2)),
])
def test_boundary_numpy_equals_dask(boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    np_result = flow_direction_dinf(numpy_agg, boundary=boundary)
    da_result = flow_direction_dinf(dask_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


def test_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="boundary must be one of"):
        flow_direction_dinf(agg, boundary='invalid')
