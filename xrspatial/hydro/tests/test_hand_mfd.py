import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro.hand_mfd import hand_mfd
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def _make_mfd_raster(fractions, backend='numpy', chunks=(3, 3)):
    _, H, W = fractions.shape
    neighbor_names = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']
    y_coords = np.arange(H, dtype=np.float64)
    x_coords = np.arange(W, dtype=np.float64)
    da_obj = xr.DataArray(
        fractions.astype(np.float64),
        dims=('neighbor', 'y', 'x'),
        coords={'neighbor': neighbor_names, 'y': y_coords, 'x': x_coords},
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
        da_obj = da_obj.copy(data=da_obj.data.map_blocks(
            cp.asarray, dtype=da_obj.dtype, meta=cp.array((), dtype=da_obj.dtype)))
    return da_obj


def _make_all_east(H, W):
    fracs = np.zeros((8, H, W), dtype=np.float64)
    fracs[0, :, :-1] = 1.0  # E for non-pit cells
    return fracs


def _make_hand_mfd_rasters(fracs, fa_data, elev_data, backend='numpy',
                            chunks=(3, 3)):
    attrs = {'res': (1.0, 1.0)}
    fd = _make_mfd_raster(fracs, backend=backend, chunks=chunks)
    fa = create_test_raster(fa_data, backend=backend, name='fa',
                            attrs=attrs, chunks=chunks)
    el = create_test_raster(elev_data, backend=backend, name='elev',
                            attrs=attrs, chunks=chunks)
    return fd, fa, el


class TestStreamCellsZero:
    def test_stream_cells_zero(self):
        fracs = _make_all_east(3, 3)
        fa = np.array([
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 3.0],
        ], dtype=np.float64)
        elev = np.array([
            [10.0, 5.0, 1.0],
            [10.0, 5.0, 1.0],
            [10.0, 5.0, 1.0],
        ], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(fracs, fa, elev)
        result = hand_mfd(fd_r, fa_r, el_r, threshold=3)
        for r in range(3):
            np.testing.assert_allclose(result.data[r, 2], 0.0, rtol=1e-10)


class TestSlopeToStream:
    def test_slope_to_stream(self):
        fracs = np.zeros((8, 1, 5), dtype=np.float64)
        fracs[0, 0, :-1] = 1.0  # E for all but last
        fa = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
        elev = np.array([[50.0, 40.0, 30.0, 20.0, 10.0]], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(fracs, fa, elev)
        result = hand_mfd(fd_r, fa_r, el_r, threshold=4)
        np.testing.assert_allclose(result.data[0, 3], 0.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 4], 0.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 2], 10.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 1], 20.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 0], 30.0, rtol=1e-10)


class TestAllStream:
    def test_all_stream(self):
        fracs = _make_all_east(2, 3)
        fa = np.array([[1.0, 2.0, 3.0],
                       [1.0, 2.0, 3.0]], dtype=np.float64)
        elev = np.array([[30.0, 20.0, 10.0],
                         [30.0, 20.0, 10.0]], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(fracs, fa, elev)
        result = hand_mfd(fd_r, fa_r, el_r, threshold=0)
        np.testing.assert_allclose(result.data, 0.0, rtol=1e-10)


class TestNoStreamPit:
    def test_no_stream_pit(self):
        fracs = np.zeros((8, 1, 3), dtype=np.float64)
        fracs[0, 0, :-1] = 1.0
        fa = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        elev = np.array([[30.0, 20.0, 10.0]], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(fracs, fa, elev)
        result = hand_mfd(fd_r, fa_r, el_r, threshold=1000)
        np.testing.assert_allclose(result.data[0, 2], 0.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 1], 10.0, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 0], 20.0, rtol=1e-10)


class TestNanHandling:
    def test_nan_handling(self):
        fracs = np.zeros((8, 1, 3), dtype=np.float64)
        fracs[0, 0, 0] = 1.0
        fracs[:, 0, 1] = np.nan
        fa = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        elev = np.array([[30.0, 20.0, 10.0]], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(fracs, fa, elev)
        result = hand_mfd(fd_r, fa_r, el_r, threshold=3)
        assert np.isnan(result.data[0, 1])
        assert not np.isnan(result.data[0, 0])


class TestEdgeExitDrain:
    def test_edge_exit_drain(self):
        fracs = np.zeros((8, 2, 3), dtype=np.float64)
        fracs[0, :, :] = 1.0  # all E, exits at right edge
        fa = np.full((2, 3), 1.0, dtype=np.float64)
        elev = np.array([[30.0, 20.0, 10.0],
                         [30.0, 20.0, 10.0]], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(fracs, fa, elev)
        result = hand_mfd(fd_r, fa_r, el_r, threshold=1000)
        for r in range(2):
            np.testing.assert_allclose(result.data[r, 2], 0.0, rtol=1e-10)
        for r in range(2):
            np.testing.assert_allclose(result.data[r, 1], 10.0, rtol=1e-10)


@dask_array_available
class TestHandMfdDask:
    @pytest.mark.parametrize("chunks", [
        (2, 2), (3, 3), (2, 6), (6, 2), (6, 6),
    ])
    def test_numpy_equals_dask(self, chunks):
        fracs = _make_all_east(6, 6)
        fa = np.ones((6, 6), dtype=np.float64)
        for c in range(6):
            fa[:, c] = c + 1
        elev = np.random.RandomState(42).uniform(0, 100, (6, 6)).astype(np.float64)

        fd_np, fa_np, el_np = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='numpy')
        fd_da, fa_da, el_da = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='dask', chunks=chunks)

        result_np = hand_mfd(fd_np, fa_np, el_np, threshold=5)
        result_da = hand_mfd(fd_da, fa_da, el_da, threshold=5)

        np.testing.assert_allclose(
            result_np.data, result_da.data.compute(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
class TestHandMfdCuPy:
    def test_numpy_equals_cupy(self):
        fracs = _make_all_east(4, 4)
        fa = np.ones((4, 4), dtype=np.float64)
        for c in range(4):
            fa[:, c] = c + 1
        elev = np.array([
            [40, 30, 20, 10],
            [40, 30, 20, 10],
            [40, 30, 20, 10],
            [40, 30, 20, 10],
        ], dtype=np.float64)

        fd_np, fa_np, el_np = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='numpy')
        fd_cu, fa_cu, el_cu = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='cupy')

        result_np = hand_mfd(fd_np, fa_np, el_np, threshold=3)
        result_cu = hand_mfd(fd_cu, fa_cu, el_cu, threshold=3)

        np.testing.assert_allclose(
            result_np.data, result_cu.data.get(),
            equal_nan=True, rtol=1e-10)


class TestMemoryGuard:
    """Memory guard on the eager numpy / cupy backends."""

    def test_numpy_huge_raster_raises(self):
        """Numpy backend raises MemoryError when projected RAM exceeds budget."""
        from unittest.mock import patch

        fracs = _make_all_east(4, 4)
        fa = np.zeros((4, 4), dtype=np.float64)
        elev = np.zeros((4, 4), dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='numpy')

        with patch(
            "xrspatial.hydro.hand_mfd._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                hand_mfd(fd_r, fa_r, el_r, threshold=1)

    def test_numpy_normal_input_succeeds(self):
        """Normal-size raster passes the guard with real memory."""
        fracs = _make_all_east(10, 10)
        fa = np.ones((10, 10), dtype=np.float64)
        elev = np.zeros((10, 10), dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='numpy')
        result = hand_mfd(fd_r, fa_r, el_r, threshold=1)
        assert result.shape == (10, 10)

    @dask_array_available
    def test_dask_path_skips_guard(self):
        """Dask backend bypasses the guard -- per-tile allocations are bounded."""
        from unittest.mock import patch

        fracs = _make_all_east(6, 6)
        fa = np.ones((6, 6), dtype=np.float64)
        elev = np.zeros((6, 6), dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='dask', chunks=(3, 3))

        with patch(
            "xrspatial.hydro.hand_mfd._available_memory_bytes",
            return_value=1,
        ):
            result = hand_mfd(fd_r, fa_r, el_r, threshold=1)
            _ = result.data[:3, :3].compute()

    def test_error_message_mentions_dimensions(self):
        """The error message should mention the grid dimensions and dask."""
        from unittest.mock import patch

        fracs = _make_all_east(7, 9)
        fa = np.ones((7, 9), dtype=np.float64)
        elev = np.zeros((7, 9), dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='numpy')

        with patch(
            "xrspatial.hydro.hand_mfd._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match=r"7x9.*dask"):
                hand_mfd(fd_r, fa_r, el_r, threshold=1)

    @cuda_and_cupy_available
    def test_cupy_huge_raster_raises(self):
        """CuPy backend raises MemoryError when projected GPU RAM exceeds budget."""
        from unittest.mock import patch

        fracs = _make_all_east(4, 4)
        fa = np.zeros((4, 4), dtype=np.float64)
        elev = np.zeros((4, 4), dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='cupy')

        with patch(
            "xrspatial.hydro.hand_mfd._available_gpu_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="GPU working memory"):
                hand_mfd(fd_r, fa_r, el_r, threshold=1)


@cuda_and_cupy_available
@dask_array_available
class TestHandMfdDaskCuPy:
    def test_numpy_equals_dask_cupy(self):
        fracs = _make_all_east(4, 4)
        fa = np.ones((4, 4), dtype=np.float64)
        for c in range(4):
            fa[:, c] = c + 1
        elev = np.array([
            [40, 30, 20, 10],
            [40, 30, 20, 10],
            [40, 30, 20, 10],
            [40, 30, 20, 10],
        ], dtype=np.float64)

        fd_np, fa_np, el_np = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='numpy')
        fd_dc, fa_dc, el_dc = _make_hand_mfd_rasters(
            fracs, fa, elev, backend='dask+cupy', chunks=(2, 2))

        result_np = hand_mfd(fd_np, fa_np, el_np, threshold=3)
        result_dc = hand_mfd(fd_dc, fa_dc, el_dc, threshold=3)

        np.testing.assert_allclose(
            result_np.data, result_dc.data.compute().get(),
            equal_nan=True, rtol=1e-10)
