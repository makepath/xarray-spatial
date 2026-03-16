import math

import numpy as np
import pytest

from xrspatial.hydrology.watershed_dinf import watershed_dinf
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def _make_fd_and_pp(fd_data, pp_data, backend='numpy', chunks=(3, 3)):
    fd = create_test_raster(fd_data.astype(np.float64), backend=backend, chunks=chunks)
    pp = create_test_raster(pp_data.astype(np.float64), backend=backend, chunks=chunks)
    return fd, pp


def test_single_pour_point():
    """5x5 bowl: all cells flow toward center, single pour point."""
    # Build angles pointing to center (2,2)
    fd = np.full((5, 5), -1.0, dtype=np.float64)
    # Approximate: use angles that point toward center
    # SE=7*pi/4, S=3*pi/2, SW=5*pi/4, E=0, W=pi, NE=pi/4, N=pi/2, NW=3*pi/4
    for r in range(5):
        for c in range(5):
            if r == 2 and c == 2:
                fd[r, c] = -1.0  # pit
                continue
            dr = 2 - r
            dc = 2 - c
            angle = math.atan2(-dr, dc)  # atan2(-dy, dx) for grid coords
            if angle < 0:
                angle += 2 * math.pi
            fd[r, c] = angle

    pp = np.full((5, 5), np.nan, dtype=np.float64)
    pp[2, 2] = 42.0

    fd_da, pp_da = _make_fd_and_pp(fd, pp)
    result = watershed_dinf(fd_da, pp_da)

    # All cells should reach pour point
    expected = np.full((5, 5), 42.0, dtype=np.float64)
    np.testing.assert_allclose(result.data, expected)


def test_two_pour_points():
    """Two separate drainage areas: single row flows east to two pits."""
    fd = np.full((1, 6), 0.0, dtype=np.float64)  # all east
    fd[0, 2] = -1.0  # pit
    fd[0, 5] = -1.0  # pit

    pp = np.full((1, 6), np.nan, dtype=np.float64)
    pp[0, 2] = 10.0
    pp[0, 5] = 20.0

    fd_da, pp_da = _make_fd_and_pp(fd, pp)
    result = watershed_dinf(fd_da, pp_da)

    for c in range(3):
        assert result.data[0, c] == 10.0, f"Failed at (0,{c})"
    for c in range(3, 6):
        assert result.data[0, c] == 20.0, f"Failed at (0,{c})"


def test_unreachable_cells():
    """Cells not draining to any pour point produce NaN."""
    fd = np.array([
        [0.0, -1.0, 0.0, -1.0],
    ], dtype=np.float64)
    pp = np.full((1, 4), np.nan, dtype=np.float64)
    pp[0, 1] = 5.0

    fd_da, pp_da = _make_fd_and_pp(fd, pp)
    result = watershed_dinf(fd_da, pp_da)

    assert result.data[0, 0] == 5.0
    assert result.data[0, 1] == 5.0
    assert np.isnan(result.data[0, 2])
    assert np.isnan(result.data[0, 3])


def test_nan_handling():
    fd = np.array([
        [0.0, -1.0],
        [np.nan, 0.0],
    ], dtype=np.float64)
    pp = np.full((2, 2), np.nan, dtype=np.float64)
    pp[0, 1] = 7.0

    fd_da, pp_da = _make_fd_and_pp(fd, pp)
    result = watershed_dinf(fd_da, pp_da)

    assert result.data[0, 0] == 7.0
    assert result.data[0, 1] == 7.0
    assert np.isnan(result.data[1, 0])


def test_linear_chain_east():
    N = 6
    fd = np.full((1, N), 0.0, dtype=np.float64)
    fd[0, -1] = -1.0
    pp = np.full((1, N), np.nan, dtype=np.float64)
    pp[0, -1] = 99.0

    fd_da, pp_da = _make_fd_and_pp(fd, pp)
    result = watershed_dinf(fd_da, pp_da)

    expected = np.full((1, N), 99.0, dtype=np.float64)
    np.testing.assert_allclose(result.data, expected)


def test_pour_point_at_non_pit():
    fd = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    pp = np.full((1, 3), np.nan, dtype=np.float64)
    pp[0, 1] = 50.0

    fd_da, pp_da = _make_fd_and_pp(fd, pp)
    result = watershed_dinf(fd_da, pp_da)

    assert result.data[0, 0] == 50.0
    assert result.data[0, 1] == 50.0
    assert np.isnan(result.data[0, 2])


def test_output_dtype():
    fd = np.full((3, 3), -1.0, dtype=np.float64)
    pp = np.full((3, 3), np.nan, dtype=np.float64)
    pp[1, 1] = 1.0

    fd_da, pp_da = _make_fd_and_pp(fd, pp)
    result = watershed_dinf(fd_da, pp_da)
    assert result.data.dtype == np.float64


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (2, 2), (3, 3), (5, 5), (1, 1), (2, 5),
])
def test_numpy_equals_dask(chunks):
    fd = np.full((5, 5), 0.0, dtype=np.float64)
    fd[:, -1] = -1.0
    fd[2, :] = math.pi * 1.5  # south
    pp = np.full((5, 5), np.nan, dtype=np.float64)
    pp[0, 4] = 42.0

    np_fd, np_pp = _make_fd_and_pp(fd, pp, backend='numpy')
    dk_fd, dk_pp = _make_fd_and_pp(fd, pp, backend='dask', chunks=chunks)

    np_result = watershed_dinf(np_fd, np_pp)
    dk_result = watershed_dinf(dk_fd, dk_pp)

    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@cuda_and_cupy_available
def test_numpy_equals_cupy():
    fd = np.full((4, 4), 0.0, dtype=np.float64)
    fd[:, -1] = -1.0
    pp = np.full((4, 4), np.nan, dtype=np.float64)
    pp[0, 3] = 1.0

    np_fd, np_pp = _make_fd_and_pp(fd, pp, backend='numpy')
    cp_fd, cp_pp = _make_fd_and_pp(fd, pp, backend='cupy')

    np_result = watershed_dinf(np_fd, np_pp)
    cp_result = watershed_dinf(cp_fd, cp_pp)

    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy():
    fd = np.full((4, 4), 0.0, dtype=np.float64)
    fd[:, -1] = -1.0
    pp = np.full((4, 4), np.nan, dtype=np.float64)
    pp[0, 3] = 1.0

    np_fd, np_pp = _make_fd_and_pp(fd, pp, backend='numpy')
    dcp_fd, dcp_pp = _make_fd_and_pp(fd, pp, backend='dask+cupy', chunks=(2, 2))

    np_result = watershed_dinf(np_fd, np_pp)
    dcp_result = watershed_dinf(dcp_fd, dcp_pp)

    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)
