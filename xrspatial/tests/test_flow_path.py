import numpy as np
import pytest
import xarray as xr

from xrspatial import flow_path
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_fd_and_sp(fd_data, sp_data, backend='numpy', chunks=(3, 3)):
    """Build flow_dir and start_points DataArrays for the given backend."""
    fd = create_test_raster(
        fd_data.astype(np.float64), backend=backend, chunks=chunks)
    sp = create_test_raster(
        sp_data.astype(np.float64), backend=backend, chunks=chunks)
    return fd, sp


# -------------------------------------------------------------------
# Basic functionality tests
# -------------------------------------------------------------------

def test_single_path():
    """One start point traces a multi-cell path to a pit."""
    # 3x3 grid: all flow east (code 1), last column is pit (code 0)
    fd = np.array([
        [1.0,  1.0,  0.0],
        [1.0,  1.0,  0.0],
        [1.0,  1.0,  0.0],
    ])
    sp = np.full((3, 3), np.nan)
    sp[0, 0] = 5.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path(fd_da, sp_da)

    # Path: (0,0) -> (0,1) -> (0,2) and stops at pit
    assert result.data[0, 0] == 5.0
    assert result.data[0, 1] == 5.0
    assert result.data[0, 2] == 5.0
    # Other rows untouched
    assert np.isnan(result.data[1, 0])
    assert np.isnan(result.data[2, 0])


def test_path_stops_at_nan():
    """Path terminates when hitting a NaN cell in the flow direction grid."""
    fd = np.array([
        [1.0,  np.nan, 0.0],
    ])
    sp = np.full((1, 3), np.nan)
    sp[0, 0] = 1.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path(fd_da, sp_da)

    # Path: (0,0) marked, then flow east hits NaN at (0,1) — stop there
    # The start cell (0,0) is labeled; (0,1) is NaN direction so we break
    # before writing (0,1) since the code reads code at current cell first
    # Actually: we mark (0,0)=1, then read code at (0,0)=1 -> move to (0,1).
    # At (0,1), we mark it=1, read code=NaN -> break.
    assert result.data[0, 0] == 1.0
    # (0,1) is reached and marked before its code is read as NaN
    assert result.data[0, 1] == 1.0
    assert np.isnan(result.data[0, 2])


def test_path_stops_at_edge():
    """Path terminates at the grid boundary."""
    fd = np.array([
        [1.0,  1.0,  1.0],
    ])
    sp = np.full((1, 3), np.nan)
    sp[0, 0] = 3.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path(fd_da, sp_da)

    # Path: (0,0)->(0,1)->(0,2)->out of bounds
    assert result.data[0, 0] == 3.0
    assert result.data[0, 1] == 3.0
    assert result.data[0, 2] == 3.0


def test_path_stops_at_pit():
    """Path terminates at code 0 (pit)."""
    fd = np.array([
        [1.0,  0.0],
    ])
    sp = np.full((1, 2), np.nan)
    sp[0, 0] = 2.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path(fd_da, sp_da)

    # Path: (0,0) marked, flow east to (0,1), marked, code=0 -> stop
    assert result.data[0, 0] == 2.0
    assert result.data[0, 1] == 2.0


def test_start_at_pit():
    """Start point on a pit cell — output has just that one cell."""
    fd = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    sp = np.full((2, 2), np.nan)
    sp[0, 0] = 9.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path(fd_da, sp_da)

    assert result.data[0, 0] == 9.0
    # No other cells should be labeled
    assert np.isnan(result.data[0, 1])
    assert np.isnan(result.data[1, 0])
    assert np.isnan(result.data[1, 1])


def test_multiple_paths():
    """Several start points trace independently."""
    # 4x4 grid: top row flows east, bottom row flows east
    fd = np.array([
        [1.0,  1.0,  1.0,  0.0],
        [1.0,  1.0,  1.0,  0.0],
        [1.0,  1.0,  1.0,  0.0],
        [1.0,  1.0,  1.0,  0.0],
    ])
    sp = np.full((4, 4), np.nan)
    sp[0, 0] = 10.0
    sp[3, 0] = 20.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path(fd_da, sp_da)

    # Path 1: row 0
    for c in range(4):
        assert result.data[0, c] == 10.0
    # Path 2: row 3
    for c in range(4):
        assert result.data[3, c] == 20.0
    # Rows 1, 2 untouched
    assert np.isnan(result.data[1, 0])
    assert np.isnan(result.data[2, 0])


def test_paths_overlap_last_wins():
    """Two paths converging on shared cells — raster-scan-order last wins."""
    # Two start points both flow south (code 4) into the same column
    fd = np.array([
        [4.0,  4.0],
        [4.0,  4.0],
        [0.0,  0.0],
    ])
    sp = np.full((3, 2), np.nan)
    sp[0, 0] = 10.0  # flows (0,0)->(1,0)->(2,0)
    sp[0, 1] = 20.0  # flows (0,1)->(1,1)->(2,1)

    # Now make them converge: both flow to same cell
    # Use diagonal: (0,0) flows SE (code 2)
    fd2 = np.array([
        [2.0,  4.0],
        [4.0,  4.0],
        [0.0,  0.0],
    ])
    sp2 = np.full((3, 2), np.nan)
    sp2[0, 0] = 10.0  # flows (0,0)->(1,1)->(2,1)
    sp2[0, 1] = 20.0  # flows (0,1)->(1,1)->(2,1)

    fd_da, sp_da = _make_fd_and_sp(fd2, sp2)
    result = flow_path(fd_da, sp_da)

    # (0,1) is scanned after (0,0) in raster order, so label 20 wins on shared cells
    assert result.data[0, 0] == 10.0
    assert result.data[0, 1] == 20.0
    assert result.data[1, 1] == 20.0  # shared cell — last writer wins
    assert result.data[2, 1] == 20.0  # shared cell — last writer wins


def test_label_preserved():
    """Start point's float label is carried along the entire path."""
    fd = np.array([
        [1.0,  1.0,  0.0],
    ])
    sp = np.full((1, 3), np.nan)
    sp[0, 0] = 42.5

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path(fd_da, sp_da)

    assert result.data[0, 0] == 42.5
    assert result.data[0, 1] == 42.5
    assert result.data[0, 2] == 42.5


def test_output_dtype():
    """Result is float64."""
    fd = np.array([[1.0, 0.0], [1.0, 0.0]])
    sp = np.full((2, 2), np.nan)
    sp[0, 0] = 1.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path(fd_da, sp_da)

    assert result.data.dtype == np.float64


def test_dataset_support():
    """@supports_dataset works."""
    fd_data = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float64)
    sp_data = np.full((2, 2), np.nan)
    sp_data[0, 0] = 1.0

    da1 = xr.DataArray(fd_data, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(0.5, 0, 2)
    da1['x'] = np.linspace(0, 0.5, 2)
    ds = xr.Dataset({'fd1': da1, 'fd2': da1.copy()})

    sp = xr.DataArray(sp_data, dims=['y', 'x'])
    sp['y'] = da1['y']
    sp['x'] = da1['x']

    result = flow_path(ds, sp)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'fd1', 'fd2'}
    for var in result.data_vars:
        assert result[var].data[0, 0] == 1.0
        assert result[var].data[0, 1] == 1.0


# -------------------------------------------------------------------
# Cross-backend tests: dask
# -------------------------------------------------------------------

@dask_array_available
@pytest.mark.parametrize("chunks", [
    (1, 1), (2, 2), (3, 3), (1, 3), (3, 1),
])
def test_numpy_equals_dask(chunks):
    """Dask matches NumPy for flow path."""
    fd = np.array([
        [2.0,  4.0,  8.0],
        [1.0,  0.0, 16.0],
        [128.0, 64.0, 32.0],
    ])
    sp = np.full((3, 3), np.nan)
    sp[0, 0] = 1.0
    sp[2, 2] = 2.0

    fd_np, sp_np = _make_fd_and_sp(fd, sp, backend='numpy')
    fd_dk, sp_dk = _make_fd_and_sp(fd, sp, backend='dask', chunks=chunks)

    np_result = flow_path(fd_np, sp_np)
    dk_result = flow_path(fd_dk, sp_dk)

    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


# -------------------------------------------------------------------
# Cross-backend tests: CuPy
# -------------------------------------------------------------------

@cuda_and_cupy_available
def test_numpy_equals_cupy():
    """CuPy matches NumPy for flow path."""
    fd = np.array([
        [2.0,  4.0,  8.0],
        [1.0,  0.0, 16.0],
        [128.0, 64.0, 32.0],
    ])
    sp = np.full((3, 3), np.nan)
    sp[0, 0] = 1.0
    sp[2, 2] = 2.0

    fd_np, sp_np = _make_fd_and_sp(fd, sp, backend='numpy')
    fd_cp, sp_cp = _make_fd_and_sp(fd, sp, backend='cupy')

    np_result = flow_path(fd_np, sp_np)
    cp_result = flow_path(fd_cp, sp_cp)

    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


# -------------------------------------------------------------------
# Cross-backend tests: Dask+CuPy
# -------------------------------------------------------------------

@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy():
    """Dask+CuPy matches NumPy for flow path."""
    fd = np.array([
        [2.0,  4.0,  8.0],
        [1.0,  0.0, 16.0],
        [128.0, 64.0, 32.0],
    ])
    sp = np.full((3, 3), np.nan)
    sp[0, 0] = 1.0
    sp[2, 2] = 2.0

    fd_np, sp_np = _make_fd_and_sp(fd, sp, backend='numpy')
    fd_dcp, sp_dcp = _make_fd_and_sp(fd, sp, backend='dask+cupy',
                                      chunks=(2, 2))

    np_result = flow_path(fd_np, sp_np)
    dcp_result = flow_path(fd_dcp, sp_dcp)

    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)
