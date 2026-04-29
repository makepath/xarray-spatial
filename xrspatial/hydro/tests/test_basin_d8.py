import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro import basin
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ====================================================================
# Basic functionality tests
# ====================================================================

def test_basin_two_pits():
    """Grid with two pits: each pit and its upstream cells form a basin."""
    flow_dir = np.array([
        [1.0, 0.0, 16.0, 0.0],
        [1.0, 0.0, 16.0, 0.0],
    ], dtype=np.float64)
    fd_da = create_test_raster(flow_dir)
    result = basin(fd_da)

    data = result.data
    # Each pit gets unique ID
    assert data[0, 1] != data[1, 1]  # different pits
    assert data[0, 3] != data[1, 3]  # different pits
    # Cells flow to their respective pits
    assert data[0, 0] == data[0, 1]  # (0,0) -> (0,1)
    assert data[0, 2] == data[0, 1]  # (0,2) -> (0,1)
    assert data[1, 0] == data[1, 1]  # (1,0) -> (1,1)
    assert data[1, 2] == data[1, 1]  # (1,2) -> (1,1)


def test_basin_edge_exits():
    """Cells flowing off-grid form their own basins."""
    flow_dir = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    fd_da = create_test_raster(flow_dir)
    result = basin(fd_da)

    data = result.data
    # Right column cells are edge-exits, each gets unique ID
    assert data[0, 2] != data[1, 2]
    # Cells in row 0 all drain to (0,2)
    assert data[0, 0] == data[0, 2]
    assert data[0, 1] == data[0, 2]
    # Cells in row 1 all drain to (1,2)
    assert data[1, 0] == data[1, 2]
    assert data[1, 1] == data[1, 2]


def test_basin_all_pits():
    """Every cell code=0: each cell is its own basin."""
    flow_dir = np.zeros((3, 4), dtype=np.float64)
    fd_da = create_test_raster(flow_dir)
    result = basin(fd_da)

    data = result.data
    unique_ids = np.unique(data[~np.isnan(data)])
    assert len(unique_ids) == 12  # 3*4


def test_basin_single_basin():
    """All cells drain to one pit: all get same label."""
    flow_dir = np.array([
        [2.0,  4.0,  8.0],
        [1.0,  0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    fd_da = create_test_raster(flow_dir)
    result = basin(fd_da)

    data = result.data
    assert np.all(data == data[1, 1])


def test_basin_nan_handling():
    """NaN flow_dir produces NaN in output."""
    flow_dir = np.array([
        [1.0, 0.0],
        [np.nan, 64.0],
    ], dtype=np.float64)
    fd_da = create_test_raster(flow_dir)
    result = basin(fd_da)

    data = result.data
    assert np.isnan(data[1, 0])
    assert not np.isnan(data[0, 0])
    assert not np.isnan(data[0, 1])
    assert not np.isnan(data[1, 1])


def test_basin_output_name():
    """Default output name is 'basin'."""
    flow_dir = np.zeros((2, 2), dtype=np.float64)
    fd_da = create_test_raster(flow_dir)
    result = basin(fd_da)
    assert result.name == 'basin'


def test_basin_dataset_support():
    """@supports_dataset works for basin."""
    flow_dir = np.zeros((3, 4), dtype=np.float64)
    da1 = xr.DataArray(flow_dir, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(1, 0, 3)
    da1['x'] = np.linspace(0, 1.5, 4)
    ds = xr.Dataset({'fd1': da1, 'fd2': da1.copy()})
    result = basin(ds)
    assert isinstance(result, xr.Dataset)


# ====================================================================
# Dask cross-backend tests
# ====================================================================

@dask_array_available
@pytest.mark.parametrize("chunks", [
    (2, 2), (3, 4), (1, 1), (3, 3),
])
def test_basin_numpy_equals_dask(chunks):
    """Multiple chunk sizes match numpy result."""
    flow_dir = np.array([
        [2.0,  4.0,  8.0],
        [1.0,  0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    np_fd = create_test_raster(flow_dir, backend='numpy')
    dk_fd = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    np_result = basin(np_fd)
    dk_result = basin(dk_fd)
    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@dask_array_available
def test_basin_dask_edge_exits():
    """Dask basins with edge-exit cells matches numpy."""
    flow_dir = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    np_fd = create_test_raster(flow_dir, backend='numpy')
    dk_fd = create_test_raster(flow_dir, backend='dask', chunks=(1, 2))
    np_result = basin(np_fd)
    dk_result = basin(dk_fd)
    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@dask_array_available
def test_basin_dask_random():
    """Random acyclic flow_dir: dask basin matches numpy."""
    from xrspatial.hydro import flow_direction

    rng = np.random.default_rng(123)
    elev = rng.random((8, 10)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    fd_data = flow_direction(elev_da).data

    np_fd = create_test_raster(fd_data, backend='numpy')
    np_result = basin(np_fd)

    for chunks in [(3, 3), (4, 5), (2, 2)]:
        dk_fd = create_test_raster(fd_data, backend='dask', chunks=chunks)
        dk_result = basin(dk_fd)
        np.testing.assert_allclose(
            np_result.data, dk_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"


@dask_array_available
def test_basin_dask_temp_cleanup():
    """BoundaryStore temp files are cleaned up after dask basin."""
    import tempfile
    import os
    import glob

    from xrspatial.hydro import flow_direction

    rng = np.random.default_rng(789)
    elev = rng.random((8, 10)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    fd_data = flow_direction(elev_da).data

    before = set(glob.glob(os.path.join(tempfile.gettempdir(), 'xrs_bdry_*')))

    dk_fd = create_test_raster(fd_data, backend='dask', chunks=(4, 5))
    result = basin(dk_fd)
    _ = result.data.compute()

    after = set(glob.glob(os.path.join(tempfile.gettempdir(), 'xrs_bdry_*')))
    assert before == after, f"Leaked temp dirs: {after - before}"


# ====================================================================
# GPU cross-backend tests
# ====================================================================

@cuda_and_cupy_available
def test_basin_numpy_equals_cupy():
    """GPU matches CPU for basin."""
    flow_dir = np.array([
        [2.0,  4.0,  8.0],
        [1.0,  0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    np_fd = create_test_raster(flow_dir, backend='numpy')
    cp_fd = create_test_raster(flow_dir, backend='cupy')
    np_result = basin(np_fd)
    cp_result = basin(cp_fd)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_basin_numpy_equals_dask_cupy():
    """Dask+CuPy matches NumPy for basin."""
    flow_dir = np.array([
        [2.0,  4.0,  8.0],
        [1.0,  0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    np_fd = create_test_raster(flow_dir, backend='numpy')
    dcp_fd = create_test_raster(flow_dir, backend='dask+cupy', chunks=(2, 2))
    np_result = basin(np_fd)
    dcp_result = basin(dcp_fd)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_basin_dask_cupy_random():
    """Random acyclic flow_dir: dask+cupy basin matches dask+numpy.

    Compared against dask+numpy rather than numpy because the tile-sweep
    has pre-existing convergence limitations for some grids/chunk combos.
    This test verifies the GPU tile kernel produces identical results to
    the CPU tile kernel.
    """
    from xrspatial.hydro import flow_direction

    rng = np.random.default_rng(952)
    elev = rng.random((8, 10)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    fd_data = flow_direction(elev_da).data

    for chunks in [(3, 3), (4, 5), (2, 2)]:
        dk_fd = create_test_raster(fd_data, backend='dask', chunks=chunks)
        dk_result = basin(dk_fd).data.compute()

        dcp_fd = create_test_raster(fd_data, backend='dask+cupy', chunks=chunks)
        dcp_result = basin(dcp_fd).data.compute().get()

        np.testing.assert_allclose(
            dk_result, dcp_result, equal_nan=True,
        ), f"Mismatch with chunks={chunks}"


# ====================================================================
# Memory guard tests
# ====================================================================

class TestMemoryGuard:
    """Memory guard on the eager numpy / cupy backends."""

    def test_numpy_huge_raster_raises(self):
        """Numpy backend raises MemoryError when projected RAM exceeds budget."""
        from unittest.mock import patch

        flow_dir = np.zeros((4, 4), dtype=np.float64)
        fd_da = create_test_raster(flow_dir, backend='numpy')

        with patch(
            "xrspatial.hydro.basin_d8._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                basin(fd_da)

    def test_numpy_normal_input_succeeds(self):
        """Normal-size raster passes the guard with real memory."""
        flow_dir = np.zeros((10, 10), dtype=np.float64)
        fd_da = create_test_raster(flow_dir, backend='numpy')
        result = basin(fd_da)
        assert result.shape == (10, 10)

    @dask_array_available
    def test_dask_path_skips_guard(self):
        """Dask backend bypasses the guard -- per-tile allocations are bounded."""
        from unittest.mock import patch

        flow_dir = np.zeros((6, 6), dtype=np.float64)
        fd_da = create_test_raster(flow_dir, backend='dask', chunks=(3, 3))

        with patch(
            "xrspatial.hydro.basin_d8._available_memory_bytes",
            return_value=1,
        ):
            result = basin(fd_da)
            _ = result.data[:3, :3].compute()

    def test_error_message_mentions_dimensions(self):
        """The error message should mention the grid dimensions and dask."""
        from unittest.mock import patch

        flow_dir = np.zeros((7, 9), dtype=np.float64)
        fd_da = create_test_raster(flow_dir, backend='numpy')

        with patch(
            "xrspatial.hydro.basin_d8._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match=r"7x9.*dask"):
                basin(fd_da)

    @cuda_and_cupy_available
    def test_cupy_huge_raster_raises(self):
        """CuPy backend raises MemoryError when projected GPU RAM exceeds budget."""
        from unittest.mock import patch

        flow_dir = np.zeros((4, 4), dtype=np.float64)
        fd_da = create_test_raster(flow_dir, backend='cupy')

        with patch(
            "xrspatial.hydro.basin_d8._available_gpu_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="GPU working memory"):
                basin(fd_da)
