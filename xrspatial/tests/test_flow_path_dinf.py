import math

import numpy as np
import pytest
import xarray as xr

from xrspatial.flow_path_dinf import flow_path_dinf
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def _make_fd_and_sp(fd_data, sp_data, backend='numpy', chunks=(3, 3)):
    fd = create_test_raster(fd_data.astype(np.float64), backend=backend, chunks=chunks)
    sp = create_test_raster(sp_data.astype(np.float64), backend=backend, chunks=chunks)
    return fd, sp


def test_single_path_cardinal_east():
    """All cells angle=0 (pure east), pit at right column."""
    fd = np.full((3, 4), 0.0, dtype=np.float64)
    fd[:, -1] = -1.0  # pits
    sp = np.full((3, 4), np.nan)
    sp[0, 0] = 5.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path_dinf(fd_da, sp_da)

    assert result.data[0, 0] == 5.0
    assert result.data[0, 1] == 5.0
    assert result.data[0, 2] == 5.0
    assert result.data[0, 3] == 5.0
    assert np.isnan(result.data[1, 0])


def test_single_path_diagonal():
    """Angle = pi/4 (NE), verify path traces NE."""
    fd = np.full((4, 4), math.pi / 4, dtype=np.float64)
    fd[0, :] = -1.0  # pits at top row
    sp = np.full((4, 4), np.nan)
    sp[3, 0] = 1.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path_dinf(fd_da, sp_da)

    # pi/4 = pure NE (k=1, alpha=0, w1=1.0 for NE)
    # Path: (3,0) -> (2,1) -> (1,2) -> (0,3) pit
    assert result.data[3, 0] == 1.0
    assert result.data[2, 1] == 1.0
    assert result.data[1, 2] == 1.0
    assert result.data[0, 3] == 1.0


def test_dominant_neighbor():
    """Angle between E and NE, dominant neighbor is E (w1 > w2)."""
    # angle = 0.1 (close to 0, so w1 = E is dominant)
    fd = np.full((2, 4), 0.1, dtype=np.float64)
    fd[:, -1] = -1.0
    sp = np.full((2, 4), np.nan)
    sp[1, 0] = 3.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path_dinf(fd_da, sp_da)

    # Dominant = E, so path goes east: (1,0) -> (1,1) -> (1,2) -> (1,3)
    assert result.data[1, 0] == 3.0
    assert result.data[1, 1] == 3.0
    assert result.data[1, 2] == 3.0
    assert result.data[1, 3] == 3.0


def test_path_stops_at_pit():
    """Pit angle (-1.0) stops the path."""
    fd = np.array([[0.0, -1.0, 0.0]], dtype=np.float64)
    sp = np.full((1, 3), np.nan)
    sp[0, 0] = 2.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path_dinf(fd_da, sp_da)

    assert result.data[0, 0] == 2.0
    assert result.data[0, 1] == 2.0
    assert np.isnan(result.data[0, 2])


def test_path_stops_at_nan():
    fd = np.array([[0.0, np.nan, 0.0]], dtype=np.float64)
    sp = np.full((1, 3), np.nan)
    sp[0, 0] = 1.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path_dinf(fd_da, sp_da)

    assert result.data[0, 0] == 1.0
    assert result.data[0, 1] == 1.0
    assert np.isnan(result.data[0, 2])


def test_path_stops_at_edge():
    fd = np.full((1, 3), 0.0, dtype=np.float64)  # all east
    sp = np.full((1, 3), np.nan)
    sp[0, 0] = 3.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path_dinf(fd_da, sp_da)

    assert result.data[0, 0] == 3.0
    assert result.data[0, 1] == 3.0
    assert result.data[0, 2] == 3.0


def test_multiple_paths():
    fd = np.full((4, 4), 0.0, dtype=np.float64)
    fd[:, -1] = -1.0
    sp = np.full((4, 4), np.nan)
    sp[0, 0] = 10.0
    sp[3, 0] = 20.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path_dinf(fd_da, sp_da)

    for c in range(4):
        assert result.data[0, c] == 10.0
        assert result.data[3, c] == 20.0
    assert np.isnan(result.data[1, 0])


def test_paths_overlap_last_wins():
    # angle ~ 1.178 => k=1, alpha~0.393, w1~0.5 w2~0.5, NE dominant since w1>=w2
    # Use angle for SE: k=6 gives S, k=7 gives SE
    # angle = 7*pi/4 = SE direction (pure), so path goes SE
    se_angle = 7 * math.pi / 4
    fd = np.full((3, 3), se_angle, dtype=np.float64)
    fd[-1, :] = -1.0  # bottom row pits
    fd[:, -1] = -1.0  # right col pits
    sp = np.full((3, 3), np.nan)
    sp[0, 0] = 10.0  # (0,0)->SE->(1,1)->SE->(2,2)
    sp[0, 1] = 20.0  # (0,1)->SE->(1,2) pit

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path_dinf(fd_da, sp_da)

    assert result.data[0, 0] == 10.0
    assert result.data[0, 1] == 20.0
    # (1,1) visited by sp[0,0] then overwritten? No, sp[0,1] path goes to (1,2)
    # sp[0,0] goes SE to (1,1), then (1,1) pit -> stop
    assert result.data[1, 1] == 10.0


def test_output_dtype():
    fd = np.full((2, 2), 0.0, dtype=np.float64)
    fd[:, -1] = -1.0
    sp = np.full((2, 2), np.nan)
    sp[0, 0] = 1.0

    fd_da, sp_da = _make_fd_and_sp(fd, sp)
    result = flow_path_dinf(fd_da, sp_da)
    assert result.data.dtype == np.float64


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (1, 1), (2, 2), (3, 3), (1, 4), (4, 1),
])
def test_numpy_equals_dask(chunks):
    fd = np.full((4, 4), 0.0, dtype=np.float64)
    fd[:, -1] = -1.0
    fd[2, :] = math.pi / 2  # north
    sp = np.full((4, 4), np.nan)
    sp[0, 0] = 1.0
    sp[3, 3] = 2.0

    fd_np, sp_np = _make_fd_and_sp(fd, sp, backend='numpy')
    fd_dk, sp_dk = _make_fd_and_sp(fd, sp, backend='dask', chunks=chunks)

    np_result = flow_path_dinf(fd_np, sp_np)
    dk_result = flow_path_dinf(fd_dk, sp_dk)

    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@cuda_and_cupy_available
def test_numpy_equals_cupy():
    fd = np.full((3, 4), 0.0, dtype=np.float64)
    fd[:, -1] = -1.0
    sp = np.full((3, 4), np.nan)
    sp[0, 0] = 1.0
    sp[2, 0] = 2.0

    fd_np, sp_np = _make_fd_and_sp(fd, sp, backend='numpy')
    fd_cp, sp_cp = _make_fd_and_sp(fd, sp, backend='cupy')

    np_result = flow_path_dinf(fd_np, sp_np)
    cp_result = flow_path_dinf(fd_cp, sp_cp)

    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy():
    fd = np.full((3, 4), 0.0, dtype=np.float64)
    fd[:, -1] = -1.0
    sp = np.full((3, 4), np.nan)
    sp[0, 0] = 1.0
    sp[2, 0] = 2.0

    fd_np, sp_np = _make_fd_and_sp(fd, sp, backend='numpy')
    fd_dcp, sp_dcp = _make_fd_and_sp(fd, sp, backend='dask+cupy', chunks=(2, 2))

    np_result = flow_path_dinf(fd_np, sp_np)
    dcp_result = flow_path_dinf(fd_dcp, sp_dcp)

    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)
