import numpy as np
import pytest
import xarray as xr

from xrspatial.hydrology import flow_direction
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy,
                                            create_test_raster,
                                            cuda_and_cupy_available,
                                            dask_array_available,
                                            general_output_checks)

VALID_CODES = {0, 1, 2, 4, 8, 16, 32, 64, 128}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_surface():
    """Constant elevation -- all interior cells should be 0 (pit)."""
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
# Hand-computed expected values
# ---------------------------------------------------------------------------

def _expected_bowl(cellsize_x=0.5, cellsize_y=0.5):
    """Compute expected D8 codes for the bowl_surface fixture.

    With equal cellsize, diag = sqrt(cx^2 + cy^2).
    """
    import math
    data = np.array([
        [9, 9, 9, 9, 9],
        [9, 8, 7, 6, 9],
        [9, 7, 5, 4, 9],
        [9, 6, 4, 3, 9],
        [9, 9, 9, 9, 9],
    ], dtype=np.float64)
    out = np.full((5, 5), np.nan)
    dy_offsets = [0, 1, 1, 1, 0, -1, -1, -1]
    dx_offsets = [1, 1, 0, -1, -1, -1, 0, 1]
    codes = [1, 2, 4, 8, 16, 32, 64, 128]
    diag = math.sqrt(cellsize_x**2 + cellsize_y**2)
    dists = [cellsize_x, diag, cellsize_y, diag,
             cellsize_x, diag, cellsize_y, diag]
    for y in range(1, 4):
        for x in range(1, 4):
            center = data[y, x]
            max_slope = -math.inf
            direction = 0.0
            for k in range(8):
                v = data[y + dy_offsets[k], x + dx_offsets[k]]
                grad = (center - v) / dists[k]
                if grad > max_slope:
                    max_slope = grad
                    direction = codes[k]
            if max_slope <= 0.0:
                out[y, x] = 0.0
            else:
                out[y, x] = direction
    return out


# ---------------------------------------------------------------------------
# Flat surface test
# ---------------------------------------------------------------------------

def test_flat_surface(flat_surface):
    agg = create_test_raster(flat_surface)
    result = flow_direction(agg)
    interior = result.data[1:-1, 1:-1]
    np.testing.assert_array_equal(interior, 0.0)


# ---------------------------------------------------------------------------
# Cardinal slope tests
# ---------------------------------------------------------------------------

def test_cardinal_east():
    """Elevation decreases going east -> flow east (code 1)."""
    data = np.array([
        [9, 8, 7, 6, 5],
        [9, 8, 7, 6, 5],
        [9, 8, 7, 6, 5],
        [9, 8, 7, 6, 5],
        [9, 8, 7, 6, 5],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction(agg)
    # Interior cells should all point east
    for y in range(1, 4):
        for x in range(1, 4):
            assert result.data[y, x] == 1.0, f"Cell ({y},{x}) = {result.data[y,x]}"


def test_cardinal_south():
    """Elevation decreases going south -> flow south (code 4)."""
    data = np.array([
        [9, 9, 9, 9, 9],
        [8, 8, 8, 8, 8],
        [7, 7, 7, 7, 7],
        [6, 6, 6, 6, 6],
        [5, 5, 5, 5, 5],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction(agg)
    for y in range(1, 4):
        for x in range(1, 4):
            assert result.data[y, x] == 4.0, f"Cell ({y},{x}) = {result.data[y,x]}"


# ---------------------------------------------------------------------------
# Diagonal slope test
# ---------------------------------------------------------------------------

def test_diagonal_se():
    """Gradient toward SE should produce code 2 with square cells.

    Surface decreases only along the SE diagonal.
    """
    # Surface where each row and column add 1 unit of elevation
    data = np.array([
        [8, 7, 6, 5, 4],
        [7, 6, 5, 4, 3],
        [6, 5, 4, 3, 2],
        [5, 4, 3, 2, 1],
        [4, 3, 2, 1, 0],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction(agg)
    # With square cells, the diagonal gradient of 2/diag > cardinal 1/cx
    # diag = sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ~ 0.707
    # SE grad = 2/0.707 ~ 2.83; E grad = 1/0.5 = 2.0
    for y in range(1, 4):
        for x in range(1, 4):
            assert result.data[y, x] == 2.0, f"Cell ({y},{x}) = {result.data[y,x]}"


# ---------------------------------------------------------------------------
# Known surface (bowl) test
# ---------------------------------------------------------------------------

def test_bowl_known(bowl_surface):
    agg = create_test_raster(bowl_surface)
    result = flow_direction(agg)
    expected = _expected_bowl(cellsize_x=0.5, cellsize_y=0.5)
    general_output_checks(agg, result, expected)


# ---------------------------------------------------------------------------
# NaN handling tests
# ---------------------------------------------------------------------------

def test_nan_center():
    """NaN center cell -> NaN output."""
    data = np.array([
        [1, 2, 3, 4],
        [5, np.nan, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=np.float64)
    agg = create_test_raster(data)
    result = flow_direction(agg)
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
    result = flow_direction(agg)
    # Cell (1,2)=7 has NaN neighbor at (1,1) -> NaN
    assert np.isnan(result.data[1, 2])
    # Cell (2,1)=10 has NaN neighbor at (1,1) -> NaN
    assert np.isnan(result.data[2, 1])
    # Cell (2,2)=11 has NaN neighbor at (1,1) -> NaN
    assert np.isnan(result.data[2, 2])


# ---------------------------------------------------------------------------
# Edge NaN test
# ---------------------------------------------------------------------------

def test_nan_edges():
    data = np.random.default_rng(42).random((6, 8)).astype(np.float64) * 100
    agg = create_test_raster(data)
    result = flow_direction(agg)
    np.testing.assert_array_equal(result.data[0, :], np.nan)
    np.testing.assert_array_equal(result.data[-1, :], np.nan)
    np.testing.assert_array_equal(result.data[:, 0], np.nan)
    np.testing.assert_array_equal(result.data[:, -1], np.nan)


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
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, flow_direction)


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(4, 6), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_numpy_equals_cupy(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    cupy_agg = create_test_raster(random_data, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, flow_direction)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(4, 6), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_numpy_equals_dask_cupy(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, flow_direction,
                                  atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Boundary mode tests
# ---------------------------------------------------------------------------

@dask_array_available
def test_boundary_modes():
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64) * 100
    numpy_agg = create_test_raster(data)
    dask_agg = create_test_raster(data, backend='dask+numpy')
    assert_boundary_mode_correctness(numpy_agg, dask_agg, flow_direction)


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
    np_result = flow_direction(numpy_agg, boundary=boundary)
    da_result = flow_direction(dask_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_boundary_no_nan_flat(boundary):
    """Flat surface with non-nan boundary should produce all 0, no NaN."""
    data = np.full((8, 10), 50.0, dtype=np.float64)
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=(4, 5))
    np_result = flow_direction(numpy_agg, boundary=boundary)
    da_result = flow_direction(dask_agg, boundary=boundary)
    np.testing.assert_array_equal(np_result.data, 0.0)
    np.testing.assert_allclose(np_result.data, da_result.data.compute(),
                               equal_nan=True, rtol=1e-6)


def test_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float64)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="boundary must be one of"):
        flow_direction(agg, boundary='invalid')


# ---------------------------------------------------------------------------
# Dtype acceptance tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_dtype_acceptance(dtype):
    data = np.arange(20, dtype=dtype).reshape(4, 5)
    agg = create_test_raster(data)
    result = flow_direction(agg)
    assert result.shape == agg.shape
    assert result.dims == agg.dims


# ---------------------------------------------------------------------------
# Dataset support test
# ---------------------------------------------------------------------------

def test_dataset_support():
    data = np.random.default_rng(42).random((6, 8)).astype(np.float64) * 100
    da1 = xr.DataArray(data, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(2.5, 0, 6)
    da1['x'] = np.linspace(0, 3.5, 8)
    ds = xr.Dataset({'elev1': da1, 'elev2': da1 * 2})
    result = flow_direction(ds)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'elev1', 'elev2'}
    for var in result.data_vars:
        expected = flow_direction(ds[var], name=var)
        np.testing.assert_allclose(
            result[var].data, expected.data, equal_nan=True)


# ---------------------------------------------------------------------------
# Cellsize effect test
# ---------------------------------------------------------------------------

def test_cellsize_effect():
    """Non-square cells should change which direction wins."""
    # Surface decreases 1 per step east and 1 per step south
    data = np.array([
        [6, 5, 4, 3],
        [5, 4, 3, 2],
        [4, 3, 2, 1],
        [3, 2, 1, 0],
    ], dtype=np.float64)

    # Case 1: cellsize_x=1, cellsize_y=2 -> E gradient > S gradient
    agg1 = xr.DataArray(data, dims=['y', 'x'], attrs={'res': (1.0, 2.0)})
    agg1['y'] = np.linspace(6, 0, 4)
    agg1['x'] = np.linspace(0, 3, 4)
    result1 = flow_direction(agg1)
    # Interior cell (1,1)=4: E grad = 1/1=1, S grad = 1/2=0.5
    # SE grad = 2/sqrt(1+4)=0.894. E wins -> code 1
    assert result1.data[1, 1] == 1.0

    # Case 2: cellsize_x=2, cellsize_y=1 -> S gradient > E gradient
    agg2 = xr.DataArray(data, dims=['y', 'x'], attrs={'res': (2.0, 1.0)})
    agg2['y'] = np.linspace(3, 0, 4)
    agg2['x'] = np.linspace(0, 6, 4)
    result2 = flow_direction(agg2)
    # Interior cell (1,1)=4: E grad = 1/2=0.5, S grad = 1/1=1
    # SE grad = 2/sqrt(4+1)=0.894. S wins -> code 4
    assert result2.data[1, 1] == 4.0


# ---------------------------------------------------------------------------
# Valid output codes test
# ---------------------------------------------------------------------------

def test_valid_output_codes():
    """All output values should be valid D8 codes or NaN."""
    data = np.random.default_rng(123).random((10, 12)).astype(np.float64) * 100
    agg = create_test_raster(data)
    result = flow_direction(agg)
    vals = result.data.ravel()
    for v in vals:
        if np.isnan(v):
            continue
        assert v in VALID_CODES, f"Invalid code: {v}"
