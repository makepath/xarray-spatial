"""Tests for xrspatial.flow_length_dinf."""

import math

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro.flow_length_dinf import flow_length_dinf
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def _make_dinf_raster(data, backend='numpy', chunks=(3, 3),
                      res=(1.0, 1.0)):
    """Build a 2-D D-inf DataArray from angle data."""
    attrs = {'res': res}
    return create_test_raster(data, backend=backend, name='fdir_dinf',
                              attrs=attrs, chunks=chunks)


class TestDownstreamCardinalEast:
    """All cells flow east (angle=0). Last col = pit."""

    def test_downstream_east(self):
        # angle=0 means pure east, w1=1.0, w2=0.0
        fd = np.full((3, 5), 0.0, dtype=np.float64)
        fd[:, -1] = -1.0  # pits at right edge
        raster = _make_dinf_raster(fd)
        result = flow_length_dinf(raster, direction='downstream')
        for r in range(3):
            for c in range(5):
                expected = (4 - c) * 1.0
                np.testing.assert_allclose(
                    result.data[r, c], expected, rtol=1e-10,
                    err_msg=f"Failed at ({r},{c})")


class TestUpstreamCardinalEast:
    """All cells flow east; upstream length = col * cellsize_x."""

    def test_upstream_east(self):
        fd = np.full((3, 5), 0.0, dtype=np.float64)
        fd[:, -1] = -1.0
        raster = _make_dinf_raster(fd)
        result = flow_length_dinf(raster, direction='upstream')
        for r in range(3):
            for c in range(5):
                expected = c * 1.0
                np.testing.assert_allclose(
                    result.data[r, c], expected, rtol=1e-10,
                    err_msg=f"Failed at ({r},{c})")


class TestDownstreamCardinalSouth:
    """All cells flow south (angle=3*pi/2). Last row = pit."""

    def test_downstream_south(self):
        # S direction: angle = 3*pi/2, k=6, alpha=0, w1=1.0 for S
        angle_s = 3.0 * math.pi / 2.0
        fd = np.full((4, 3), angle_s, dtype=np.float64)
        fd[-1, :] = -1.0
        raster = _make_dinf_raster(fd)
        result = flow_length_dinf(raster, direction='downstream')
        for r in range(4):
            for c in range(3):
                expected = (3 - r) * 1.0
                np.testing.assert_allclose(
                    result.data[r, c], expected, rtol=1e-10,
                    err_msg=f"Failed at ({r},{c})")


class TestDownstreamDiagonalNE:
    """All cells flow NE (angle=pi/4). Pit at top-right corner."""

    def test_downstream_ne(self):
        # NE: angle = pi/4, k=1, alpha=0, w1=1.0 for NE
        angle_ne = math.pi / 4.0
        H, W = 4, 4
        fd = np.full((H, W), angle_ne, dtype=np.float64)
        fd[0, W - 1] = -1.0  # pit at (0, 3)
        # Cells that would flow OOB: treat as pits too
        fd[0, :] = -1.0  # top row flows OOB except corner handled
        fd[:, W - 1] = -1.0  # right col flows OOB
        # Actually let's simplify: only interior cells flow NE
        fd[0, W - 1] = -1.0  # the outlet

        raster = _make_dinf_raster(fd)
        result = flow_length_dinf(raster, direction='downstream')
        diag = math.sqrt(2.0)
        # (3,0): 3 diagonal steps to (0,3)
        np.testing.assert_allclose(result.data[3, 0], 3 * diag, rtol=1e-10)
        # (2,1): 2 diagonal steps
        np.testing.assert_allclose(result.data[2, 1], 2 * diag, rtol=1e-10)
        # (0,3): pit = 0
        np.testing.assert_allclose(result.data[0, 3], 0.0, rtol=1e-10)


class TestDownstreamProportionalSplit:
    """Cell with angle pi/8 splits 50/50 between E and NE."""

    def test_equal_split(self):
        # angle=pi/8: k=0, alpha=pi/8, w1 = 1 - (pi/8)/(pi/4) = 0.5, w2=0.5
        angle = math.pi / 8.0
        # 3x3 grid: center sends 0.5 to E(1,2) and 0.5 to NE(0,2)
        fd = np.full((3, 3), -1.0, dtype=np.float64)  # all pits
        fd[1, 1] = angle

        raster = _make_dinf_raster(fd)
        result = flow_length_dinf(raster, direction='downstream')

        diag = math.sqrt(2.0)
        # center: 0.5 * (1.0 + 0) + 0.5 * (diag + 0)
        expected = 0.5 * 1.0 + 0.5 * diag
        np.testing.assert_allclose(result.data[1, 1], expected, rtol=1e-10)


class TestUpstreamMaxPath:
    """Upstream takes the longest path at junctions."""

    def test_upstream_max_path(self):
        # 2x4 grid:
        # Row 0: all flow east (angle=0)
        # Row 1: cells 0,1 flow east, cell 2 flows north (angle=pi/2)
        fd = np.full((2, 4), 0.0, dtype=np.float64)
        fd[0, 3] = -1.0  # pit
        fd[1, 3] = -1.0  # pit
        fd[1, 2] = math.pi / 2.0  # N: flows to (0,2)

        raster = _make_dinf_raster(fd)
        result = flow_length_dinf(raster, direction='upstream')

        # (0,0): divide -> 0
        assert result.data[0, 0] == 0.0
        # (0,1): from (0,0) -> 1
        np.testing.assert_allclose(result.data[0, 1], 1.0, rtol=1e-10)
        # (1,2): upstream from (1,0)->(1,1)->(1,2) = 2
        np.testing.assert_allclose(result.data[1, 2], 2.0, rtol=1e-10)
        # (0,2): max((0,1)+1=2, (1,2)+1=3) = 3
        np.testing.assert_allclose(result.data[0, 2], 3.0, rtol=1e-10)


class TestNanHandling:
    """NaN flow_dir should produce NaN output."""

    def test_nan_cells(self):
        fd = np.array([
            [0.0,    np.nan, -1.0],
            [0.0,    0.0,    -1.0],
        ], dtype=np.float64)
        raster = _make_dinf_raster(fd)
        result_ds = flow_length_dinf(raster, direction='downstream')
        result_us = flow_length_dinf(raster, direction='upstream')
        assert np.isnan(result_ds.data[0, 1])
        assert np.isnan(result_us.data[0, 1])
        assert not np.isnan(result_ds.data[1, 0])


class TestPitCells:
    """Pit cells (angle=-1) have flow_length = 0."""

    def test_pit_zero(self):
        fd = np.full((2, 2), -1.0, dtype=np.float64)
        raster = _make_dinf_raster(fd)
        result = flow_length_dinf(raster, direction='downstream')
        for r in range(2):
            for c in range(2):
                assert result.data[r, c] == 0.0


class TestDirectionValidation:
    """Invalid direction string should raise."""

    def test_invalid_direction(self):
        fd = np.full((2, 2), -1.0, dtype=np.float64)
        raster = _make_dinf_raster(fd)
        with pytest.raises(ValueError, match="direction"):
            flow_length_dinf(raster, direction='sideways')


class TestNonSquareCells:
    """Test with rectangular cells."""

    def test_rectangular_east(self):
        fd = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
        raster = _make_dinf_raster(fd, res=(2.0, 1.0))
        result = flow_length_dinf(raster, direction='downstream')
        np.testing.assert_allclose(result.data[0, 0], 4.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 1], 2.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 2], 0.0, rtol=1e-10)


class TestFlowDirectionDinfIntegration:
    """Integration test with actual flow_direction_dinf output."""

    def test_bowl_downstream(self):
        from xrspatial.hydro.flow_direction_dinf import flow_direction_dinf

        elev = np.array([
            [9, 8, 9],
            [8, 1, 8],
            [9, 8, 9],
        ], dtype=np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        dinf = flow_direction_dinf(elev_r)
        result = flow_length_dinf(dinf, direction='downstream')

        # Edge cells are NaN (D-inf needs all 8 neighbors)
        assert np.isnan(result.data[0, 0])


@dask_array_available
class TestFlowLengthDinfDask:
    """Cross-backend: numpy vs dask for both directions."""

    @pytest.mark.parametrize("chunks", [
        (2, 2), (3, 3), (2, 5), (5, 2),
    ])
    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_numpy_equals_dask(self, chunks, direction):
        from xrspatial.hydro.flow_direction_dinf import flow_direction_dinf

        np.random.seed(123)
        elev = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        dinf = flow_direction_dinf(elev_r)
        fd_data = dinf.data.astype(np.float64)

        np_raster = _make_dinf_raster(fd_data, backend='numpy')
        da_raster = _make_dinf_raster(fd_data, backend='dask', chunks=chunks)

        result_np = flow_length_dinf(np_raster, direction=direction)
        result_da = flow_length_dinf(da_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_da.data.compute(),
            equal_nan=True, rtol=1e-10)

    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_cross_tile_flow(self, direction):
        """Flow crossing tile boundaries should compute correctly."""
        fd = np.full((2, 6), 0.0, dtype=np.float64)
        fd[:, -1] = -1.0
        np_raster = _make_dinf_raster(fd, backend='numpy')
        da_raster = _make_dinf_raster(fd, backend='dask', chunks=(2, 2))

        result_np = flow_length_dinf(np_raster, direction=direction)
        result_da = flow_length_dinf(da_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_da.data.compute(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
class TestFlowLengthDinfCuPy:
    """Cross-backend: numpy vs cupy."""

    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_numpy_equals_cupy(self, direction):
        from xrspatial.hydro.flow_direction_dinf import flow_direction_dinf

        np.random.seed(42)
        elev = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        dinf = flow_direction_dinf(elev_r)
        fd_data = dinf.data.astype(np.float64)

        np_raster = _make_dinf_raster(fd_data, backend='numpy')
        cu_raster = _make_dinf_raster(fd_data, backend='cupy')

        result_np = flow_length_dinf(np_raster, direction=direction)
        result_cu = flow_length_dinf(cu_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_cu.data.get(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
@dask_array_available
class TestFlowLengthDinfDaskCuPy:
    """Cross-backend: numpy vs dask+cupy."""

    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_numpy_equals_dask_cupy(self, direction):
        from xrspatial.hydro.flow_direction_dinf import flow_direction_dinf

        np.random.seed(42)
        elev = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        dinf = flow_direction_dinf(elev_r)
        fd_data = dinf.data.astype(np.float64)

        np_raster = _make_dinf_raster(fd_data, backend='numpy')
        dc_raster = _make_dinf_raster(fd_data, backend='dask+cupy',
                                       chunks=(3, 3))

        result_np = flow_length_dinf(np_raster, direction=direction)
        result_dc = flow_length_dinf(dc_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_dc.data.compute().get(),
            equal_nan=True, rtol=1e-10)


# ====================================================================
# Memory guard tests
# ====================================================================

class TestMemoryGuard:
    """Memory guard on the eager numpy / cupy backends."""

    def test_numpy_huge_raster_raises(self):
        """Numpy backend raises MemoryError when projected RAM exceeds budget."""
        from unittest.mock import patch

        fd = np.full((4, 4), 0.0, dtype=np.float64)
        raster = _make_dinf_raster(fd)

        with patch(
            "xrspatial.hydro.flow_length_dinf._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                flow_length_dinf(raster, direction='downstream')

    def test_numpy_normal_input_succeeds(self):
        """Normal-size raster passes the guard with real memory."""
        fd = np.full((3, 5), 0.0, dtype=np.float64)
        fd[:, -1] = -1.0
        raster = _make_dinf_raster(fd)
        result = flow_length_dinf(raster, direction='downstream')
        assert result.shape == (3, 5)

    @dask_array_available
    def test_dask_path_skips_guard(self):
        """Dask backend bypasses the guard -- per-tile allocations are bounded."""
        from unittest.mock import patch

        fd = np.full((6, 6), 0.0, dtype=np.float64)
        raster = _make_dinf_raster(fd, backend='dask', chunks=(3, 3))

        with patch(
            "xrspatial.hydro.flow_length_dinf._available_memory_bytes",
            return_value=1,
        ):
            # Dask path must not trigger the guard at dispatch time
            result = flow_length_dinf(raster, direction='downstream')
            assert result is not None

    def test_error_message_mentions_dimensions(self):
        """The error message should mention the grid dimensions and dask."""
        from unittest.mock import patch

        fd = np.full((7, 9), 0.0, dtype=np.float64)
        raster = _make_dinf_raster(fd)

        with patch(
            "xrspatial.hydro.flow_length_dinf._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match=r"7x9.*dask"):
                flow_length_dinf(raster, direction='upstream')

    @cuda_and_cupy_available
    def test_cupy_huge_raster_raises(self):
        """CuPy backend raises MemoryError when projected GPU RAM exceeds budget."""
        from unittest.mock import patch

        fd = np.full((4, 4), 0.0, dtype=np.float64)
        raster = _make_dinf_raster(fd, backend='cupy')

        with patch(
            "xrspatial.hydro.flow_length_dinf._available_gpu_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="GPU working memory"):
                flow_length_dinf(raster, direction='downstream')
