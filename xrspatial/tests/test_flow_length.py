"""Tests for xrspatial.flow_length."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.flow_length import flow_length
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def _make_flow_dir_raster(data, backend='numpy', chunks=(3, 3),
                          res=(1.0, 1.0)):
    attrs = {'res': res}
    return create_test_raster(data, backend=backend, name='fdir',
                              attrs=attrs, chunks=chunks)


class TestDownstreamLinear:
    """All cells flow east; pit at far-right column."""

    def test_downstream_linear(self):
        # 3x5 grid: all flow east (code 1), last col = pit (code 0)
        fd = np.full((3, 5), 1.0, dtype=np.float64)
        fd[:, -1] = 0.0  # pits at right edge
        raster = _make_flow_dir_raster(fd)
        result = flow_length(raster, direction='downstream')
        # Cell at col c: downstream length = (4 - c) * cellsize_x
        for r in range(3):
            for c in range(5):
                expected = (4 - c) * 1.0
                np.testing.assert_allclose(result.data[r, c], expected, rtol=1e-10)


class TestUpstreamLinear:
    """All cells flow east; upstream length = col * cellsize_x."""

    def test_upstream_linear(self):
        fd = np.full((3, 5), 1.0, dtype=np.float64)
        fd[:, -1] = 0.0
        raster = _make_flow_dir_raster(fd)
        result = flow_length(raster, direction='upstream')
        for r in range(3):
            for c in range(5):
                expected = c * 1.0
                np.testing.assert_allclose(result.data[r, c], expected, rtol=1e-10)


class TestDownstreamDiagonal:
    """All cells flow SE; pit at bottom-right corner."""

    def test_downstream_diagonal(self):
        # 4x4 grid: all flow SE (code 2), bottom-right = pit
        fd = np.full((4, 4), 2.0, dtype=np.float64)
        fd[3, 3] = 0.0
        raster = _make_flow_dir_raster(fd)
        result = flow_length(raster, direction='downstream')
        diag = np.sqrt(2.0)
        # Cell (r,c): steps to (3,3) = max(3-r, 3-c) but since all SE,
        # it's min(3-r, 3-c) steps (constrained by whichever axis hits edge first)
        # Actually with all SE: from (r,c) to (r+1,c+1) to ... until (3,3) or pit
        # Steps = min(3-r, 3-c) diagonal steps, then the path continues
        # No, all cells flow SE to the cell diagonally below-right.
        # From (0,0): (0,0)->(1,1)->(2,2)->(3,3) = 3 steps
        # From (0,3): (0,3) flows SE to (1,4) which is out of bounds → edge exit = 0
        for r in range(4):
            for c in range(4):
                steps = min(3 - r, 3 - c)
                if steps < 0:
                    steps = 0
                # But cells where the SE step goes out of bounds are edge-exits
                if r < 3 and c < 3:
                    expected = steps * diag
                else:
                    # (3, *) or (*, 3): next step goes out of bounds or is pit
                    if r == 3 and c == 3:
                        expected = 0.0  # pit
                    elif r == 3 or c == 3:
                        expected = 0.0  # flows out of bounds
                    else:
                        expected = steps * diag
                np.testing.assert_allclose(
                    result.data[r, c], expected, rtol=1e-10,
                    err_msg=f"Failed at ({r},{c})")


class TestDownstreamPitCenter:
    """3x3 grid where all cells flow toward center (pit)."""

    def test_pit_center(self):
        # All 8 neighbors flow to center
        fd = np.array([
            [2.0,   4.0,   8.0],    # SE, S, SW
            [1.0,   0.0,  16.0],    # E, pit, W
            [128.0, 64.0, 32.0],    # NE, N, NW
        ], dtype=np.float64)
        raster = _make_flow_dir_raster(fd)
        result = flow_length(raster, direction='downstream')
        diag = np.sqrt(2.0)
        # Center = 0 (pit)
        assert result.data[1, 1] == 0.0
        # Cardinal neighbors: 1 step = cellsize
        for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            np.testing.assert_allclose(result.data[r, c], 1.0, rtol=1e-10)
        # Diagonal neighbors: 1 step = diag
        for r, c in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            np.testing.assert_allclose(result.data[r, c], diag, rtol=1e-10)


class TestUpstreamMaxPath:
    """Verify upstream takes the longest path at junctions."""

    def test_upstream_max_path(self):
        # 1x4 grid: cells 0,1,2 flow east, cell 3 = pit
        # Plus a 2nd row: row 1, col 2 flows north to (0,2)
        # So (0,2) has two upstream paths:
        #   from (0,0)→(0,1)→(0,2): length = 2
        #   from (1,2)→(0,2): length = 1
        # upstream[(0,2)] = max(2, 1) = 2
        fd = np.array([
            [1.0, 1.0,  1.0, 0.0],
            [1.0, 1.0, 64.0, 0.0],
        ], dtype=np.float64)
        raster = _make_flow_dir_raster(fd)
        result = flow_length(raster, direction='upstream')
        # (0,0): divide, upstream=0
        assert result.data[0, 0] == 0.0
        # (0,1): from (0,0) = 1
        np.testing.assert_allclose(result.data[0, 1], 1.0, rtol=1e-10)
        # (0,2): max of (0,1)+1=2 and (1,2)+1=1, BUT
        # (1,2) has upstream from (1,0)→(1,1)→(1,2) = 2 steps
        # So upstream[(1,2)] = 2, and (1,2)→(0,2) adds 1 = 3
        # upstream[(0,2)] = max(2, 3) = 3
        np.testing.assert_allclose(result.data[0, 2], 3.0, rtol=1e-10)


class TestNanHandling:
    """NaN flow_dir should produce NaN output."""

    def test_nan_handling(self):
        fd = np.array([
            [1.0,   np.nan, 0.0],
            [1.0,   1.0,    0.0],
        ], dtype=np.float64)
        raster = _make_flow_dir_raster(fd)
        result_ds = flow_length(raster, direction='downstream')
        result_us = flow_length(raster, direction='upstream')
        assert np.isnan(result_ds.data[0, 1])
        assert np.isnan(result_us.data[0, 1])
        # Non-NaN cells should have valid values
        assert not np.isnan(result_ds.data[1, 0])


class TestEdgeExitZero:
    """Edge-exit cells have downstream flow_length = 0."""

    def test_edge_exit_zero(self):
        # All flow east, no pit - rightmost cells exit the grid
        fd = np.full((2, 4), 1.0, dtype=np.float64)
        raster = _make_flow_dir_raster(fd)
        result = flow_length(raster, direction='downstream')
        # Rightmost column exits grid → 0
        for r in range(2):
            np.testing.assert_allclose(result.data[r, 3], 0.0, rtol=1e-10)
        # Others: distance to edge
        for r in range(2):
            for c in range(3):
                np.testing.assert_allclose(
                    result.data[r, c], (3 - c) * 1.0, rtol=1e-10)


class TestDirectionValidation:
    """Invalid direction string should raise."""

    def test_invalid_direction(self):
        fd = np.full((2, 2), 0.0, dtype=np.float64)
        raster = _make_flow_dir_raster(fd)
        with pytest.raises(ValueError, match="direction"):
            flow_length(raster, direction='sideways')


class TestNonSquareCells:
    """Test with rectangular cells."""

    def test_rectangular_downstream(self):
        # 1x3: all flow east, pit at end. cellsize_x = 2.0
        fd = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
        raster = _make_flow_dir_raster(fd, res=(2.0, 1.0))
        result = flow_length(raster, direction='downstream')
        # col 0: 2 steps × 2.0 = 4.0, col 1: 1 step × 2.0 = 2.0, col 2: 0
        np.testing.assert_allclose(result.data[0, 0], 4.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 1], 2.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 2], 0.0, rtol=1e-10)

    def test_rectangular_south(self):
        # 3x1: all flow south, pit at end. cellsize_y = 3.0
        fd = np.array([[4.0], [4.0], [0.0]], dtype=np.float64)
        raster = _make_flow_dir_raster(fd, res=(1.0, 3.0))
        result = flow_length(raster, direction='downstream')
        np.testing.assert_allclose(result.data[0, 0], 6.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[1, 0], 3.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[2, 0], 0.0, rtol=1e-10)


@dask_array_available
class TestFlowLengthDask:
    """Cross-backend: numpy vs dask for both directions."""

    @pytest.mark.parametrize("chunks", [
        (2, 2), (3, 3), (2, 5), (5, 2), (6, 6),
    ])
    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_numpy_equals_dask(self, chunks, direction):
        from xrspatial.flow_direction import flow_direction

        np.random.seed(123)
        elev = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        fd_r = flow_direction(elev_r)
        fd_data = fd_r.data.astype(np.float64)

        np_raster = _make_flow_dir_raster(fd_data, backend='numpy')
        da_raster = _make_flow_dir_raster(fd_data, backend='dask', chunks=chunks)

        result_np = flow_length(np_raster, direction=direction)
        result_da = flow_length(da_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_da.data.compute(),
            equal_nan=True, rtol=1e-10)

    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_cross_tile_flow(self, direction):
        """Flow crossing tile boundaries should compute correctly."""
        # 6x1: all flow south, pit at bottom
        fd = np.array([[4.0], [4.0], [4.0], [4.0], [4.0], [0.0]],
                      dtype=np.float64)
        np_raster = _make_flow_dir_raster(fd, backend='numpy')
        da_raster = _make_flow_dir_raster(fd, backend='dask', chunks=(2, 1))

        result_np = flow_length(np_raster, direction=direction)
        result_da = flow_length(da_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_da.data.compute(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
class TestFlowLengthCuPy:
    """Cross-backend: numpy vs cupy."""

    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_numpy_equals_cupy(self, direction):
        from xrspatial.flow_direction import flow_direction

        np.random.seed(42)
        elev = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        fd_r = flow_direction(elev_r)
        fd_data = fd_r.data.astype(np.float64)

        np_raster = _make_flow_dir_raster(fd_data, backend='numpy')
        cu_raster = _make_flow_dir_raster(fd_data, backend='cupy')

        result_np = flow_length(np_raster, direction=direction)
        result_cu = flow_length(cu_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_cu.data.get(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
@dask_array_available
class TestFlowLengthDaskCuPy:
    """Cross-backend: numpy vs dask+cupy."""

    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_numpy_equals_dask_cupy(self, direction):
        from xrspatial.flow_direction import flow_direction

        np.random.seed(42)
        elev = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        fd_r = flow_direction(elev_r)
        fd_data = fd_r.data.astype(np.float64)

        np_raster = _make_flow_dir_raster(fd_data, backend='numpy')
        dc_raster = _make_flow_dir_raster(fd_data, backend='dask+cupy',
                                           chunks=(3, 3))

        result_np = flow_length(np_raster, direction=direction)
        result_dc = flow_length(dc_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_dc.data.compute().get(),
            equal_nan=True, rtol=1e-10)
