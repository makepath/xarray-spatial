import numpy as np
import pytest
import xarray as xr

from xrspatial.hydrology import watershed, basins
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ====================================================================
# Watershed tests
# ====================================================================

def test_single_pour_point():
    """5x5 grid, one pit at center with pour_point label=42."""
    flow_dir = np.array([
        [2.0,  4.0,  4.0,  4.0,  8.0],
        [1.0,  2.0,  4.0,  8.0, 16.0],
        [1.0,  1.0,  0.0, 16.0, 16.0],
        [1.0, 128.0, 64.0, 32.0, 16.0],
        [128.0, 64.0, 64.0, 64.0, 32.0],
    ], dtype=np.float64)
    pour_points = np.full((5, 5), np.nan, dtype=np.float64)
    pour_points[2, 2] = 42.0

    fd_da = create_test_raster(flow_dir)
    pp_da = create_test_raster(pour_points)
    result = watershed(fd_da, pp_da)
    expected = np.full((5, 5), 42.0, dtype=np.float64)
    np.testing.assert_allclose(result.data, expected)


def test_two_pour_points():
    """4x6 grid with two separate drainage areas."""
    flow_dir = np.array([
        [1.0, 1.0, 4.0, 1.0, 1.0, 4.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 64.0, 1.0, 1.0, 64.0],
        [1.0, 1.0, 64.0, 1.0, 1.0, 64.0],
    ], dtype=np.float64)
    pour_points = np.full((4, 6), np.nan, dtype=np.float64)
    pour_points[1, 2] = 10.0
    pour_points[1, 5] = 20.0

    fd_da = create_test_raster(flow_dir)
    pp_da = create_test_raster(pour_points)
    result = watershed(fd_da, pp_da)

    expected = np.array([
        [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
    ], dtype=np.float64)
    np.testing.assert_allclose(result.data, expected)


def test_unreachable_cells():
    """Cells not draining to any pour point produce NaN."""
    flow_dir = np.array([
        [1.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)
    pour_points = np.full((1, 4), np.nan, dtype=np.float64)
    pour_points[0, 1] = 5.0

    fd_da = create_test_raster(flow_dir)
    pp_da = create_test_raster(pour_points)
    result = watershed(fd_da, pp_da)

    assert result.data[0, 0] == 5.0
    assert result.data[0, 1] == 5.0
    assert np.isnan(result.data[0, 2])
    assert np.isnan(result.data[0, 3])


def test_nan_handling():
    """NaN flow_dir produces NaN label."""
    flow_dir = np.array([
        [1.0, 0.0],
        [np.nan, 1.0],
    ], dtype=np.float64)
    pour_points = np.full((2, 2), np.nan, dtype=np.float64)
    pour_points[0, 1] = 7.0

    fd_da = create_test_raster(flow_dir)
    pp_da = create_test_raster(pour_points)
    result = watershed(fd_da, pp_da)

    assert result.data[0, 0] == 7.0
    assert result.data[0, 1] == 7.0
    assert np.isnan(result.data[1, 0])
    assert np.isnan(result.data[1, 1])


def test_linear_chain():
    """Row of cells flowing east to a pour point at the end."""
    N = 6
    flow_dir = np.full((1, N), 1.0, dtype=np.float64)
    flow_dir[0, -1] = 0.0
    pour_points = np.full((1, N), np.nan, dtype=np.float64)
    pour_points[0, -1] = 99.0

    fd_da = create_test_raster(flow_dir)
    pp_da = create_test_raster(pour_points)
    result = watershed(fd_da, pp_da)

    expected = np.full((1, N), 99.0, dtype=np.float64)
    np.testing.assert_allclose(result.data, expected)


def test_pour_point_at_non_pit():
    """Pour point on a cell that flows downstream (not a pit)."""
    flow_dir = np.array([
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)
    pour_points = np.full((1, 3), np.nan, dtype=np.float64)
    pour_points[0, 1] = 50.0

    fd_da = create_test_raster(flow_dir)
    pp_da = create_test_raster(pour_points)
    result = watershed(fd_da, pp_da)

    assert result.data[0, 0] == 50.0
    assert result.data[0, 1] == 50.0
    assert np.isnan(result.data[0, 2])


def test_multiple_paths_to_same_pour_point():
    """Cells from different directions all reach same pour point."""
    flow_dir = np.array([
        [2.0,  4.0,  8.0],
        [1.0,  0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    pour_points = np.full((3, 3), np.nan, dtype=np.float64)
    pour_points[1, 1] = 1.0

    fd_da = create_test_raster(flow_dir)
    pp_da = create_test_raster(pour_points)
    result = watershed(fd_da, pp_da)

    expected = np.full((3, 3), 1.0, dtype=np.float64)
    np.testing.assert_allclose(result.data, expected)


# ====================================================================
# Backward-compat: basins() still works via wrapper
# ====================================================================

def test_basins_backward_compat():
    """basins() wrapper delegates to basin() correctly."""
    from xrspatial.hydrology import basin

    flow_dir = np.array([
        [2.0,  4.0,  8.0],
        [1.0,  0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    fd_da = create_test_raster(flow_dir)
    old_result = basins(fd_da)
    new_result = basin(fd_da)
    np.testing.assert_allclose(old_result.data, new_result.data, equal_nan=True)


# ====================================================================
# Cross-backend tests
# ====================================================================

@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_watershed_dtype_acceptance(dtype):
    """int32/int64/float32/float64 flow_dir inputs accepted."""
    flow_dir = np.zeros((3, 4), dtype=dtype)
    pour_points = np.full((3, 4), np.nan, dtype=np.float64)
    pour_points[0, 0] = 1.0
    fd_da = create_test_raster(flow_dir)
    pp_da = create_test_raster(pour_points)
    result = watershed(fd_da, pp_da)
    assert result.shape == fd_da.shape
    assert result.data.dtype == np.float64


def test_watershed_dataset_support():
    """@supports_dataset works for watershed."""
    flow_dir = np.zeros((3, 4), dtype=np.float64)
    da1 = xr.DataArray(flow_dir, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(1, 0, 3)
    da1['x'] = np.linspace(0, 1.5, 4)
    ds = xr.Dataset({'fd1': da1, 'fd2': da1.copy()})
    pp = xr.DataArray(
        np.full((3, 4), np.nan, dtype=np.float64),
        dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    pp['y'] = np.linspace(1, 0, 3)
    pp['x'] = np.linspace(0, 1.5, 4)
    result = watershed(ds, pp)
    assert isinstance(result, xr.Dataset)


def test_watershed_output_dtype():
    """Output is float64."""
    flow_dir = np.zeros((3, 3), dtype=np.float64)
    pour_points = np.full((3, 3), np.nan, dtype=np.float64)
    pour_points[1, 1] = 1.0
    fd_da = create_test_raster(flow_dir)
    pp_da = create_test_raster(pour_points)
    result = watershed(fd_da, pp_da)
    assert result.data.dtype == np.float64


# ====================================================================
# Dask cross-backend tests
# ====================================================================

def _make_watershed_test_data():
    """5x5 bowl for cross-backend comparison."""
    flow_dir = np.array([
        [2.0,  4.0,  4.0,  4.0,  8.0],
        [1.0,  2.0,  4.0,  8.0, 16.0],
        [1.0,  1.0,  0.0, 16.0, 16.0],
        [1.0, 128.0, 64.0, 32.0, 16.0],
        [128.0, 64.0, 64.0, 64.0, 32.0],
    ], dtype=np.float64)
    pour_points = np.full((5, 5), np.nan, dtype=np.float64)
    pour_points[2, 2] = 42.0
    return flow_dir, pour_points


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (2, 2), (3, 3), (5, 5), (1, 1), (2, 5),
])
def test_watershed_numpy_equals_dask(chunks):
    """Multiple chunk sizes all match numpy result."""
    flow_dir, pour_points = _make_watershed_test_data()
    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_pp = create_test_raster(pour_points, backend='numpy')
    dk_fd = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    dk_pp = create_test_raster(pour_points, backend='dask', chunks=chunks)
    np_result = watershed(np_fd, np_pp)
    dk_result = watershed(dk_fd, dk_pp)
    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@dask_array_available
def test_watershed_dask_two_pour_points():
    """Dask matches numpy with two pour points."""
    flow_dir = np.array([
        [1.0, 1.0, 4.0, 1.0, 1.0, 4.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 64.0, 1.0, 1.0, 64.0],
        [1.0, 1.0, 64.0, 1.0, 1.0, 64.0],
    ], dtype=np.float64)
    pour_points = np.full((4, 6), np.nan, dtype=np.float64)
    pour_points[1, 2] = 10.0
    pour_points[1, 5] = 20.0

    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_pp = create_test_raster(pour_points, backend='numpy')
    np_result = watershed(np_fd, np_pp)

    for chunks in [(2, 3), (4, 6), (1, 1), (2, 2)]:
        dk_fd = create_test_raster(flow_dir, backend='dask', chunks=chunks)
        dk_pp = create_test_raster(pour_points, backend='dask', chunks=chunks)
        dk_result = watershed(dk_fd, dk_pp)
        np.testing.assert_allclose(
            np_result.data, dk_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"


@dask_array_available
def test_watershed_dask_random():
    """Random acyclic flow_dir: dask watershed matches numpy."""
    from xrspatial.hydrology import flow_direction, flow_accumulation

    rng = np.random.default_rng(456)
    elev = rng.random((8, 10)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    fd_data = flow_direction(elev_da).data

    pp = np.full_like(fd_data, np.nan)
    pit_mask = fd_data == 0
    pp[pit_mask] = np.arange(1, pit_mask.sum() + 1, dtype=np.float64)

    np_fd = create_test_raster(fd_data, backend='numpy')
    np_pp = create_test_raster(pp, backend='numpy')
    np_result = watershed(np_fd, np_pp)

    for chunks in [(3, 3), (4, 5), (2, 2)]:
        dk_fd = create_test_raster(fd_data, backend='dask', chunks=chunks)
        dk_pp = create_test_raster(pp, backend='dask', chunks=chunks)
        dk_result = watershed(dk_fd, dk_pp)
        np.testing.assert_allclose(
            np_result.data, dk_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"


# ====================================================================
# GPU cross-backend tests
# ====================================================================

@cuda_and_cupy_available
def test_watershed_numpy_equals_cupy():
    """GPU matches CPU for watershed."""
    flow_dir, pour_points = _make_watershed_test_data()
    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_pp = create_test_raster(pour_points, backend='numpy')
    cp_fd = create_test_raster(flow_dir, backend='cupy')
    cp_pp = create_test_raster(pour_points, backend='cupy')
    np_result = watershed(np_fd, np_pp)
    cp_result = watershed(cp_fd, cp_pp)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_watershed_numpy_equals_dask_cupy():
    """Dask+CuPy matches NumPy for watershed."""
    flow_dir, pour_points = _make_watershed_test_data()
    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_pp = create_test_raster(pour_points, backend='numpy')
    dcp_fd = create_test_raster(flow_dir, backend='dask+cupy', chunks=(3, 3))
    dcp_pp = create_test_raster(pour_points, backend='dask+cupy', chunks=(3, 3))
    np_result = watershed(np_fd, np_pp)
    dcp_result = watershed(dcp_fd, dcp_pp)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_watershed_dask_cupy_random():
    """Random acyclic flow_dir: dask+cupy watershed matches numpy."""
    from xrspatial.hydrology import flow_direction, flow_accumulation

    rng = np.random.default_rng(952)
    elev = rng.random((8, 10)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    fd_data = flow_direction(elev_da).data

    pp = np.full_like(fd_data, np.nan)
    pit_mask = fd_data == 0
    pp[pit_mask] = np.arange(1, pit_mask.sum() + 1, dtype=np.float64)

    np_fd = create_test_raster(fd_data, backend='numpy')
    np_pp = create_test_raster(pp, backend='numpy')
    np_result = watershed(np_fd, np_pp)

    for chunks in [(3, 3), (4, 5), (2, 2)]:
        dcp_fd = create_test_raster(fd_data, backend='dask+cupy', chunks=chunks)
        dcp_pp = create_test_raster(pp, backend='dask+cupy', chunks=chunks)
        dcp_result = watershed(dcp_fd, dcp_pp)
        np.testing.assert_allclose(
            np_result.data, dcp_result.data.compute().get(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"
