"""Tests for xrspatial.flow_length_mfd."""

import math

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro.flow_length_mfd import flow_length_mfd
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def _make_mfd_raster(fractions, backend='numpy', chunks=(3, 3),
                     res=(1.0, 1.0)):
    """Build a 3-D MFD DataArray from (8, H, W) fractions array.

    Mirrors the output shape of ``flow_direction_mfd``.
    """
    _, H, W = fractions.shape
    neighbor_names = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']
    y_coords = np.arange(H) * abs(res[1])
    x_coords = np.arange(W) * abs(res[0])

    da_obj = xr.DataArray(
        fractions.astype(np.float64),
        dims=('neighbor', 'y', 'x'),
        coords={
            'neighbor': neighbor_names,
            'y': y_coords,
            'x': x_coords,
        },
        attrs={'res': res},
        name='mfd_fdir',
    )

    if backend == 'dask':
        import dask.array as da
        da_obj = da_obj.chunk({'neighbor': 8, 'y': chunks[0], 'x': chunks[1]})
    elif backend == 'cupy':
        import cupy as cp
        da_obj = da_obj.copy(data=cp.asarray(da_obj.data))
    elif backend == 'dask+cupy':
        import cupy as cp
        import dask.array as da
        da_obj = da_obj.chunk({'neighbor': 8, 'y': chunks[0], 'x': chunks[1]})
        da_obj = da_obj.copy(
            data=da_obj.data.map_blocks(cp.asarray, dtype=da_obj.dtype,
                                         meta=cp.array((), dtype=da_obj.dtype)))

    return da_obj


def _all_east_fractions(H, W):
    """All cells flow east (direction 0). Last col = pit (all zeros)."""
    fracs = np.zeros((8, H, W), dtype=np.float64)
    fracs[0, :, :-1] = 1.0  # E direction, all but last col
    # Last col: pit (all zeros)
    # Mark last col as valid (not NaN) by ensuring at least one band is finite
    # Actually pits have all-zero fractions, which is valid (not NaN)
    return fracs


def _all_south_fractions(H, W):
    """All cells flow south (direction 2). Last row = pit."""
    fracs = np.zeros((8, H, W), dtype=np.float64)
    fracs[2, :-1, :] = 1.0  # S direction, all but last row
    return fracs


class TestDownstreamLinear:
    """All cells flow east; pit at rightmost column."""

    def test_downstream_linear(self):
        fracs = _all_east_fractions(3, 5)
        raster = _make_mfd_raster(fracs)
        result = flow_length_mfd(raster, direction='downstream')
        for r in range(3):
            for c in range(5):
                expected = (4 - c) * 1.0
                np.testing.assert_allclose(
                    result.data[r, c], expected, rtol=1e-10,
                    err_msg=f"Failed at ({r},{c})")


class TestUpstreamLinear:
    """All cells flow east; upstream length = col * cellsize_x."""

    def test_upstream_linear(self):
        fracs = _all_east_fractions(3, 5)
        raster = _make_mfd_raster(fracs)
        result = flow_length_mfd(raster, direction='upstream')
        for r in range(3):
            for c in range(5):
                expected = c * 1.0
                np.testing.assert_allclose(
                    result.data[r, c], expected, rtol=1e-10,
                    err_msg=f"Failed at ({r},{c})")


class TestDownstreamSouth:
    """All cells flow south; pit at bottom row."""

    def test_downstream_south(self):
        fracs = _all_south_fractions(4, 3)
        raster = _make_mfd_raster(fracs)
        result = flow_length_mfd(raster, direction='downstream')
        for r in range(4):
            for c in range(3):
                expected = (3 - r) * 1.0
                np.testing.assert_allclose(
                    result.data[r, c], expected, rtol=1e-10,
                    err_msg=f"Failed at ({r},{c})")


class TestDownstreamDiagonal:
    """All cells flow SE; pit at bottom-right corner."""

    def test_downstream_diagonal(self):
        H, W = 4, 4
        fracs = np.zeros((8, H, W), dtype=np.float64)
        fracs[1, :, :] = 0.0
        # Only interior cells flow SE; edge cells that would go OOB get 0
        for r in range(H):
            for c in range(W):
                if r < H - 1 and c < W - 1:
                    fracs[1, r, c] = 1.0  # SE
                # else: pit or flows off grid

        raster = _make_mfd_raster(fracs)
        result = flow_length_mfd(raster, direction='downstream')
        diag = math.sqrt(2.0)
        for r in range(H):
            for c in range(W):
                steps = min(H - 1 - r, W - 1 - c)
                expected = steps * diag
                np.testing.assert_allclose(
                    result.data[r, c], expected, rtol=1e-10,
                    err_msg=f"Failed at ({r},{c})")


class TestDownstreamSplit:
    """Cell splits flow between two neighbors: check weighted average."""

    def test_equal_split(self):
        # 1x3 grid: cell (0,0) sends 0.5 E, 0.5 unused (off grid)
        # Actually, let's make a 3x3 with center sending 0.5 E, 0.5 S
        fracs = np.zeros((8, 3, 3), dtype=np.float64)
        # Center (1,1): 0.5 east, 0.5 south
        fracs[0, 1, 1] = 0.5  # E -> (1,2)
        fracs[2, 1, 1] = 0.5  # S -> (2,1)
        # (1,2) and (2,1) are pits (all zero fracs)
        raster = _make_mfd_raster(fracs)
        result = flow_length_mfd(raster, direction='downstream')

        # center: 0.5 * (1.0 + 0) + 0.5 * (1.0 + 0) = 1.0
        np.testing.assert_allclose(result.data[1, 1], 1.0, rtol=1e-10)
        # pits: 0.0
        assert result.data[1, 2] == 0.0
        assert result.data[2, 1] == 0.0

    def test_unequal_split(self):
        # Center sends 0.7 E, 0.3 SE
        fracs = np.zeros((8, 3, 3), dtype=np.float64)
        fracs[0, 1, 1] = 0.7  # E -> (1,2)
        fracs[1, 1, 1] = 0.3  # SE -> (2,2)
        raster = _make_mfd_raster(fracs)
        result = flow_length_mfd(raster, direction='downstream')

        diag = math.sqrt(2.0)
        expected = 0.7 * 1.0 + 0.3 * diag
        np.testing.assert_allclose(result.data[1, 1], expected, rtol=1e-10)


class TestDownstreamChain:
    """Chain: A -> split(B, C) -> D, verify weighted propagation."""

    def test_chain_weighted(self):
        # 1x4: cell 0 -> E(1.0) -> cell 1 -> 0.6 E, 0.4 goes off grid
        # cell 1 -> E -> cell 2 -> E(1.0) -> cell 3 (pit)
        fracs = np.zeros((8, 1, 4), dtype=np.float64)
        fracs[0, 0, 0] = 1.0  # cell 0 -> E
        fracs[0, 0, 1] = 1.0  # cell 1 -> E
        fracs[0, 0, 2] = 1.0  # cell 2 -> E
        # cell 3 = pit
        raster = _make_mfd_raster(fracs)
        result = flow_length_mfd(raster, direction='downstream')

        # cell 3: 0, cell 2: 1, cell 1: 2, cell 0: 3
        np.testing.assert_allclose(result.data[0, 3], 0.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 2], 1.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 1], 2.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 0], 3.0, rtol=1e-10)


class TestUpstreamMaxPath:
    """Upstream takes the longest path at junctions."""

    def test_upstream_max_path(self):
        # 2x4 grid:
        # Row 0: all flow east
        # Row 1: cells 0,1 flow east, cell 2 flows north to (0,2)
        # (0,2) gets flow from (0,1) path length 2 and (1,2) path length 3
        # upstream[(0,2)] = max(2, 3) = 3
        fracs = np.zeros((8, 2, 4), dtype=np.float64)
        fracs[0, 0, 0] = 1.0  # (0,0) -> E
        fracs[0, 0, 1] = 1.0  # (0,1) -> E
        fracs[0, 0, 2] = 1.0  # (0,2) -> E
        # row 1
        fracs[0, 1, 0] = 1.0  # (1,0) -> E
        fracs[0, 1, 1] = 1.0  # (1,1) -> E
        fracs[6, 1, 2] = 1.0  # (1,2) -> N to (0,2)
        # (0,3) and (1,3) are pits

        raster = _make_mfd_raster(fracs)
        result = flow_length_mfd(raster, direction='upstream')

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
        fracs = np.zeros((8, 2, 3), dtype=np.float64)
        fracs[0, 0, 0] = 1.0  # (0,0) -> E
        fracs[:, 0, 1] = np.nan  # (0,1) is NaN
        # (0,2), (1,*) are pits

        raster = _make_mfd_raster(fracs)
        result_ds = flow_length_mfd(raster, direction='downstream')
        result_us = flow_length_mfd(raster, direction='upstream')

        assert np.isnan(result_ds.data[0, 1])
        assert np.isnan(result_us.data[0, 1])
        # Non-NaN cells should have valid values
        assert not np.isnan(result_ds.data[0, 0])


class TestPitCells:
    """Pit cells (all-zero fractions) have flow_length = 0."""

    def test_pit_zero(self):
        fracs = np.zeros((8, 2, 2), dtype=np.float64)
        # All cells are pits
        raster = _make_mfd_raster(fracs)
        result = flow_length_mfd(raster, direction='downstream')
        for r in range(2):
            for c in range(2):
                assert result.data[r, c] == 0.0


class TestDirectionValidation:
    """Invalid direction string should raise."""

    def test_invalid_direction(self):
        fracs = np.zeros((8, 2, 2), dtype=np.float64)
        raster = _make_mfd_raster(fracs)
        with pytest.raises(ValueError, match="direction"):
            flow_length_mfd(raster, direction='sideways')


class TestNonSquareCells:
    """Test with rectangular cells."""

    def test_rectangular_east(self):
        # All flow east, cellsize_x = 2.0
        fracs = _all_east_fractions(1, 3)
        raster = _make_mfd_raster(fracs, res=(2.0, 1.0))
        result = flow_length_mfd(raster, direction='downstream')
        np.testing.assert_allclose(result.data[0, 0], 4.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 1], 2.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 2], 0.0, rtol=1e-10)

    def test_rectangular_south(self):
        # All flow south, cellsize_y = 3.0
        fracs = _all_south_fractions(3, 1)
        raster = _make_mfd_raster(fracs, res=(1.0, 3.0))
        result = flow_length_mfd(raster, direction='downstream')
        np.testing.assert_allclose(result.data[0, 0], 6.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[1, 0], 3.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[2, 0], 0.0, rtol=1e-10)


class TestShapeValidation:
    """Non-(8, H, W) input should raise."""

    def test_wrong_shape(self):
        data = np.zeros((3, 4, 4), dtype=np.float64)
        raster = xr.DataArray(data, dims=('a', 'b', 'c'),
                              attrs={'res': (1.0, 1.0)})
        with pytest.raises(ValueError, match="shape"):
            flow_length_mfd(raster)


class TestFlowDirectionMfdIntegration:
    """Integration test with actual flow_direction_mfd output."""

    def test_bowl_downstream(self):
        from xrspatial.hydro.flow_direction_mfd import flow_direction_mfd

        # Bowl DEM: center is lowest
        elev = np.array([
            [9, 8, 9],
            [8, 1, 8],
            [9, 8, 9],
        ], dtype=np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        mfd = flow_direction_mfd(elev_r)
        result = flow_length_mfd(mfd, direction='downstream')

        # Center (1,1) is a pit (no downslope) -> 0
        np.testing.assert_allclose(result.data[1, 1], 0.0, atol=1e-10)
        # Edge cells are NaN (MFD needs all 8 neighbors)
        assert np.isnan(result.data[0, 0])


@dask_array_available
class TestFlowLengthMfdDask:
    """Cross-backend: numpy vs dask for both directions."""

    @pytest.mark.parametrize("chunks", [
        (2, 2), (3, 3), (2, 5), (5, 2),
    ])
    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_numpy_equals_dask(self, chunks, direction):
        from xrspatial.hydro.flow_direction_mfd import flow_direction_mfd

        np.random.seed(123)
        elev = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        mfd = flow_direction_mfd(elev_r)
        mfd_data = mfd.data.astype(np.float64)

        np_raster = _make_mfd_raster(mfd_data, backend='numpy')
        da_raster = _make_mfd_raster(mfd_data, backend='dask', chunks=chunks)

        result_np = flow_length_mfd(np_raster, direction=direction)
        result_da = flow_length_mfd(da_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_da.data.compute(),
            equal_nan=True, rtol=1e-10)

    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_cross_tile_flow(self, direction):
        """Flow crossing tile boundaries should compute correctly."""
        fracs = _all_east_fractions(2, 6)
        np_raster = _make_mfd_raster(fracs, backend='numpy')
        da_raster = _make_mfd_raster(fracs, backend='dask', chunks=(2, 2))

        result_np = flow_length_mfd(np_raster, direction=direction)
        result_da = flow_length_mfd(da_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_da.data.compute(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
class TestFlowLengthMfdCuPy:
    """Cross-backend: numpy vs cupy."""

    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_numpy_equals_cupy(self, direction):
        from xrspatial.hydro.flow_direction_mfd import flow_direction_mfd

        np.random.seed(42)
        elev = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        mfd = flow_direction_mfd(elev_r)
        mfd_data = mfd.data.astype(np.float64)

        np_raster = _make_mfd_raster(mfd_data, backend='numpy')
        cu_raster = _make_mfd_raster(mfd_data, backend='cupy')

        result_np = flow_length_mfd(np_raster, direction=direction)
        result_cu = flow_length_mfd(cu_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_cu.data.get(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
@dask_array_available
class TestFlowLengthMfdDaskCuPy:
    """Cross-backend: numpy vs dask+cupy."""

    @pytest.mark.parametrize("direction", ['downstream', 'upstream'])
    def test_numpy_equals_dask_cupy(self, direction):
        from xrspatial.hydro.flow_direction_mfd import flow_direction_mfd

        np.random.seed(42)
        elev = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev, backend='numpy', name='elev')
        mfd = flow_direction_mfd(elev_r)
        mfd_data = mfd.data.astype(np.float64)

        np_raster = _make_mfd_raster(mfd_data, backend='numpy')
        dc_raster = _make_mfd_raster(mfd_data, backend='dask+cupy',
                                      chunks=(3, 3))

        result_np = flow_length_mfd(np_raster, direction=direction)
        result_dc = flow_length_mfd(dc_raster, direction=direction)

        np.testing.assert_allclose(
            result_np.data, result_dc.data.compute().get(),
            equal_nan=True, rtol=1e-10)
