import numpy as np
import pytest
import xarray as xr

from xrspatial.hydrology import flow_accumulation
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ---------------------------------------------------------------------------
# Known bowl
# ---------------------------------------------------------------------------

def test_known_bowl():
    """5x5 bowl: hand-computed flow_dir and expected accum.

    Bowl surface:
        9  9  9  9  9
        9  8  7  6  9
        9  7  5  4  9
        9  6  4  3  9
        9  9  9  9  9

    Flow directions (edges NaN, interior computed with cx=cy=0.5):
        NaN NaN NaN NaN NaN
        NaN  2   2   4  NaN
        NaN  2   2   4  NaN
        NaN  1   1   0  NaN
        NaN NaN NaN NaN NaN

    Pit at (3,3) collects all 9 interior cells.
    """
    flow_dir = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 1, 1, 0, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    expected = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 1, 1, 1, np.nan],
        [np.nan, 1, 2, 3, np.nan],
        [np.nan, 1, 3, 9, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    np.testing.assert_allclose(result.data, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# Linear chain (east)
# ---------------------------------------------------------------------------

def test_linear_chain_east():
    """All cells flow east, last cell pit. accum = [1, 2, ..., N]."""
    N = 6
    flow_dir = np.full((1, N), 1.0, dtype=np.float64)
    flow_dir[0, -1] = 0.0  # last cell is a pit
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    expected = np.arange(1, N + 1, dtype=np.float64).reshape(1, N)
    np.testing.assert_allclose(result.data, expected)


# ---------------------------------------------------------------------------
# Pit center
# ---------------------------------------------------------------------------

def test_pit_center():
    """3x3 grid, all 8 neighbours flow to center -> center accum = 9."""
    flow_dir = np.array([
        [2.0, 4.0, 8.0],
        [1.0, 0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    assert result.data[1, 1] == 9.0
    # Edge cells have accum = 1 (only themselves)
    for r, c in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2),
                 (2, 0), (2, 1), (2, 2)]:
        assert result.data[r, c] == 1.0, f"Cell ({r},{c}) = {result.data[r,c]}"


# ---------------------------------------------------------------------------
# All pits
# ---------------------------------------------------------------------------

def test_all_pits():
    """Every cell code = 0 -> every cell accum = 1."""
    flow_dir = np.zeros((4, 5), dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    np.testing.assert_array_equal(result.data, 1.0)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_nan_handling():
    """NaN flow_dir -> NaN accum; valid neighbours still correct."""
    flow_dir = np.array([
        [1.0, 1.0, 0.0],
        [np.nan, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    assert np.isnan(result.data[1, 0])
    expected = np.array([
        [1.0, 2.0, 3.0],
        [np.nan, 1.0, 2.0],
        [1.0, 2.0, 3.0],
    ])
    np.testing.assert_allclose(result.data, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# Code 0 pit has accum = 1
# ---------------------------------------------------------------------------

def test_code_0_pit():
    """Pit cells (code 0) count themselves -> accum = 1."""
    flow_dir = np.array([
        [1.0, 0.0],
        [64.0, 0.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    # (1,0) -> N -> (0,0) -> E -> (0,1) pit
    assert result.data[1, 0] == 1.0  # leaf, just itself
    assert result.data[0, 0] == 2.0  # itself + (1,0)
    assert result.data[0, 1] == 3.0  # itself + (0,0) chain
    assert result.data[1, 1] == 1.0  # pit, just itself


# ---------------------------------------------------------------------------
# Valid output values
# ---------------------------------------------------------------------------

def test_valid_output_values():
    """All output values are finite positive or NaN."""
    flow_dir = np.array([
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 2, 4, np.nan],
        [np.nan, 1, 0, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    vals = result.data.ravel()
    for v in vals:
        assert np.isnan(v) or v >= 1.0, f"Invalid value: {v}"


# ---------------------------------------------------------------------------
# Uses float64
# ---------------------------------------------------------------------------

def test_output_dtype():
    """Output dtype is float64."""
    flow_dir = np.zeros((3, 3), dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    assert result.data.dtype == np.float64


# ---------------------------------------------------------------------------
# Dtype acceptance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_dtype_acceptance(dtype):
    """Function accepts int32/int64/float32/float64 flow_dir inputs."""
    # All zeros (pits) -- valid for any dtype
    flow_dir = np.zeros((3, 4), dtype=dtype)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    assert result.shape == agg.shape
    assert result.data.dtype == np.float64


# ---------------------------------------------------------------------------
# Dataset support
# ---------------------------------------------------------------------------

def test_dataset_support():
    """@supports_dataset works."""
    flow_dir = np.zeros((3, 4), dtype=np.float64)
    da1 = xr.DataArray(flow_dir, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(1, 0, 3)
    da1['x'] = np.linspace(0, 1.5, 4)
    ds = xr.Dataset({'fd1': da1, 'fd2': da1.copy()})
    result = flow_accumulation(ds)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'fd1', 'fd2'}
    for var in result.data_vars:
        np.testing.assert_array_equal(result[var].data, 1.0)


# ---------------------------------------------------------------------------
# Cross-backend tests
# ---------------------------------------------------------------------------

def _make_cross_backend_flow_dir():
    """6x6 flow-east grid for cross-backend comparison."""
    flow_dir = np.full((6, 6), 1.0, dtype=np.float64)
    flow_dir[:, -1] = 0.0  # last column = pits
    return flow_dir


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (3, 3), (5, 5), (2, 6), (6, 2), (1, 1), (6, 6),
])
def test_chunk_configs(chunks):
    """Multiple chunk sizes all match numpy result."""
    flow_dir = _make_cross_backend_flow_dir()
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    dask_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    np_result = flow_accumulation(numpy_agg)
    da_result = flow_accumulation(dask_agg)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True)


@dask_array_available
def test_dask_cross_tile_flow_north():
    """Flow northward across tile boundary matches numpy."""
    flow_dir = np.full((6, 4), 64.0, dtype=np.float64)
    flow_dir[0, :] = 0.0  # top row = pits
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    dask_agg = create_test_raster(flow_dir, backend='dask', chunks=(3, 4))
    np_result = flow_accumulation(numpy_agg)
    da_result = flow_accumulation(dask_agg)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True)


@dask_array_available
def test_dask_diagonal_cross_tile():
    """Diagonal SE flow across tile boundary matches numpy."""
    flow_dir = np.full((6, 6), 2.0, dtype=np.float64)  # all SE
    flow_dir[-1, -1] = 0.0  # bottom-right pit
    # Cells flowing outside grid just lose their accum
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    dask_agg = create_test_raster(flow_dir, backend='dask', chunks=(3, 3))
    np_result = flow_accumulation(numpy_agg)
    da_result = flow_accumulation(dask_agg)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True)


@dask_array_available
def test_dask_bowl():
    """Bowl flow_dir with dask matches numpy."""
    flow_dir = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 1, 1, 0, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    for chunks in [(2, 2), (3, 3), (5, 5), (1, 1)]:
        dask_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
        np_result = flow_accumulation(numpy_agg)
        da_result = flow_accumulation(dask_agg)
        np.testing.assert_allclose(
            np_result.data, da_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"


@dask_array_available
def test_dask_random():
    """Random acyclic flow_dir: dask matches numpy across chunk sizes.

    Use flow_direction() on random elevation to guarantee acyclic flow
    graphs (randomly assigned D8 codes can create cycles, which the
    iterative tile sweep cannot handle).
    """
    from xrspatial.hydrology import flow_direction

    rng = np.random.default_rng(42)
    elev = rng.random((10, 12)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    flow_dir = flow_direction(elev_da).data

    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    np_result = flow_accumulation(numpy_agg)

    for chunks in [(3, 3), (5, 6), (2, 4), (10, 12)]:
        dask_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
        da_result = flow_accumulation(dask_agg)
        np.testing.assert_allclose(
            np_result.data, da_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"


@cuda_and_cupy_available
def test_numpy_equals_cupy():
    """CuPy matches NumPy."""
    flow_dir = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 1, 1, 0, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    cupy_agg = create_test_raster(flow_dir, backend='cupy')
    np_result = flow_accumulation(numpy_agg)
    cp_result = flow_accumulation(cupy_agg)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@cuda_and_cupy_available
def test_cupy_pit_center():
    """CuPy: pit center with 8 neighbours draining in."""
    flow_dir = np.array([
        [2.0, 4.0, 8.0],
        [1.0, 0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    cupy_agg = create_test_raster(flow_dir, backend='cupy')
    result = flow_accumulation(cupy_agg)
    data = result.data.get()
    assert data[1, 1] == 9.0


@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy():
    """Dask+CuPy matches NumPy."""
    flow_dir = _make_cross_backend_flow_dir()
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    dask_cupy_agg = create_test_raster(flow_dir, backend='dask+cupy',
                                       chunks=(3, 3))
    np_result = flow_accumulation(numpy_agg)
    dcp_result = flow_accumulation(dask_cupy_agg)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("chunks", [(3, 3), (5, 6), (2, 4)])
def test_dask_cupy_chunk_configs(chunks):
    """Dask+CuPy matches NumPy across chunk sizes."""
    from xrspatial.hydrology import flow_direction

    rng = np.random.default_rng(952)
    elev = rng.random((10, 12)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    flow_dir = flow_direction(elev_da).data

    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    dcp_agg = create_test_raster(flow_dir, backend='dask+cupy', chunks=chunks)
    np_result = flow_accumulation(numpy_agg)
    dcp_result = flow_accumulation(dcp_agg)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True,
    ), f"Mismatch with chunks={chunks}"
