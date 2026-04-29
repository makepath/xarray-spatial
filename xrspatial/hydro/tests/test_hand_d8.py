"""Tests for xrspatial.hand (Height Above Nearest Drainage)."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro.hand_d8 import hand_d8 as hand
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def _make_hand_rasters(fd_data, fa_data, elev_data, backend='numpy',
                       chunks=(3, 3), res=(1.0, 1.0)):
    """Create flow_dir, flow_accum, and elevation rasters."""
    attrs = {'res': res}
    fd = create_test_raster(fd_data, backend=backend, name='fdir',
                            attrs=attrs, chunks=chunks)
    fa = create_test_raster(fa_data, backend=backend, name='fa',
                            attrs=attrs, chunks=chunks)
    el = create_test_raster(elev_data, backend=backend, name='elev',
                            attrs=attrs, chunks=chunks)
    return fd, fa, el


class TestStreamCellsZero:
    """Stream cells (flow_accum >= threshold) should have HAND = 0."""

    def test_stream_cells_zero(self):
        # 3x3: all flow east, pit at right. Stream at col 2 (accum=3).
        fd = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float64)
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
        fd_r, fa_r, el_r = _make_hand_rasters(fd, fa, elev)
        result = hand(fd_r, fa_r, el_r, threshold=3)
        # Stream cells (col 2): HAND = 0
        for r in range(3):
            np.testing.assert_allclose(result.data[r, 2], 0.0, rtol=1e-10)


class TestSlopeToStream:
    """Linear gradient draining to stream: HAND = elev - stream_elev."""

    def test_slope_to_stream(self):
        # 1x5: all flow east, pit at col 4.
        # Stream starts at col 3 (accum >= 4).
        # elev = [50, 40, 30, 20, 10]
        fd = np.array([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float64)
        fa = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
        elev = np.array([[50.0, 40.0, 30.0, 20.0, 10.0]], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_rasters(fd, fa, elev)
        result = hand(fd_r, fa_r, el_r, threshold=4)
        # col 3: stream, HAND = 0
        np.testing.assert_allclose(result.data[0, 3], 0.0, rtol=1e-10)
        # col 4: stream, HAND = 0
        np.testing.assert_allclose(result.data[0, 4], 0.0, rtol=1e-10)
        # col 2: drains to stream at col 3, HAND = 30 - 20 = 10
        np.testing.assert_allclose(result.data[0, 2], 10.0, rtol=1e-10)
        # col 1: drains to stream at col 3, HAND = 40 - 20 = 20
        np.testing.assert_allclose(result.data[0, 1], 20.0, rtol=1e-10)
        # col 0: drains to stream at col 3, HAND = 50 - 20 = 30
        np.testing.assert_allclose(result.data[0, 0], 30.0, rtol=1e-10)


class TestAllStream:
    """Threshold=0: every cell is a stream, HAND=0 everywhere."""

    def test_all_stream(self):
        fd = np.array([[1.0, 1.0, 0.0],
                       [1.0, 1.0, 0.0]], dtype=np.float64)
        fa = np.array([[1.0, 2.0, 3.0],
                       [1.0, 2.0, 3.0]], dtype=np.float64)
        elev = np.array([[30.0, 20.0, 10.0],
                         [30.0, 20.0, 10.0]], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_rasters(fd, fa, elev)
        result = hand(fd_r, fa_r, el_r, threshold=0)
        np.testing.assert_allclose(result.data, 0.0, rtol=1e-10)


class TestNoStreamPit:
    """No cells meet threshold; cells drain to pit, HAND = elev - pit_elev."""

    def test_no_stream_pit(self):
        # All flow east to pit at col 2. No cell meets threshold=1000.
        fd = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
        fa = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        elev = np.array([[30.0, 20.0, 10.0]], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_rasters(fd, fa, elev)
        result = hand(fd_r, fa_r, el_r, threshold=1000)
        # Pit (col 2): not a stream, but is a pit → drain_elev = self → HAND=0
        np.testing.assert_allclose(result.data[0, 2], 0.0, rtol=1e-10)
        # Col 1: drains to pit, drain_elev = pit elev = 10 → HAND = 20-10 = 10
        np.testing.assert_allclose(result.data[0, 1], 10.0, rtol=1e-10)
        # Col 0: drains to pit → HAND = 30-10 = 20
        np.testing.assert_allclose(result.data[0, 0], 20.0, rtol=1e-10)


class TestNanHandling:
    """NaN flow_dir should produce NaN HAND."""

    def test_nan_handling(self):
        fd = np.array([[1.0, np.nan, 0.0]], dtype=np.float64)
        fa = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        elev = np.array([[30.0, 20.0, 10.0]], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_rasters(fd, fa, elev)
        result = hand(fd_r, fa_r, el_r, threshold=3)
        assert np.isnan(result.data[0, 1])
        # col 0: flows east but downstream is NaN → can't reach stream
        # So it follows code 1 to (0,1) which is NaN → effectively an edge exit
        # drain_elev = own elev, HAND = 0
        # Actually: col 0 flows east to col 1, but col 1 is NaN/invalid
        # In the BFS: col 0 flows to col 1, but col 1 is not valid.
        # So col 0 is like flowing to an invalid cell → treated like flowing off grid
        # drain_elev = self elev → HAND = 0
        assert not np.isnan(result.data[0, 0])


class TestEdgeExitDrain:
    """Cell flows off grid edge: drain_elev = own elevation, HAND=0."""

    def test_edge_exit_drain(self):
        # All flow east, no pit, right column exits grid
        fd = np.full((2, 3), 1.0, dtype=np.float64)
        fa = np.full((2, 3), 1.0, dtype=np.float64)
        elev = np.array([[30.0, 20.0, 10.0],
                         [30.0, 20.0, 10.0]], dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_rasters(fd, fa, elev)
        result = hand(fd_r, fa_r, el_r, threshold=1000)
        # Col 2: exits grid, drain_elev = self → HAND = 0
        for r in range(2):
            np.testing.assert_allclose(result.data[r, 2], 0.0, rtol=1e-10)
        # Col 1: drains through to exit at col 2, drain_elev = col2 elev = 10
        # HAND = 20 - 10 = 10
        for r in range(2):
            np.testing.assert_allclose(result.data[r, 1], 10.0, rtol=1e-10)


@dask_array_available
class TestHandDask:
    """Cross-backend: numpy vs dask."""

    @pytest.mark.parametrize("chunks", [
        (2, 2), (3, 3), (2, 6), (6, 2), (6, 6),
    ])
    def test_numpy_equals_dask(self, chunks):
        from xrspatial.hydro.flow_direction_d8 import flow_direction_d8 as flow_direction
        from xrspatial.hydro.flow_accumulation_d8 import flow_accumulation_d8 as flow_accumulation

        np.random.seed(42)
        elev_data = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev_data, backend='numpy', name='elev')
        fd_r = flow_direction(elev_r)
        fa_r = flow_accumulation(fd_r)
        fd_data = fd_r.data.astype(np.float64)
        fa_data = fa_r.data.astype(np.float64)

        fd_np, fa_np, el_np = _make_hand_rasters(
            fd_data, fa_data, elev_data, backend='numpy')
        fd_da, fa_da, el_da = _make_hand_rasters(
            fd_data, fa_data, elev_data, backend='dask', chunks=chunks)

        result_np = hand(fd_np, fa_np, el_np, threshold=5)
        result_da = hand(fd_da, fa_da, el_da, threshold=5)

        np.testing.assert_allclose(
            result_np.data, result_da.data.compute(),
            equal_nan=True, rtol=1e-10)

    def test_cross_tile_flow(self):
        """Stream in one tile, draining cells in another."""
        # 6x1 column: all flow south, pit at bottom.
        # Stream at row 4,5 (accum >= 5). Elev = 60,50,...,10
        fd = np.array([[4.0], [4.0], [4.0], [4.0], [4.0], [0.0]],
                      dtype=np.float64)
        fa = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
                      dtype=np.float64)
        elev = np.array([[60.0], [50.0], [40.0], [30.0], [20.0], [10.0]],
                        dtype=np.float64)
        fd_np, fa_np, el_np = _make_hand_rasters(
            fd, fa, elev, backend='numpy')
        fd_da, fa_da, el_da = _make_hand_rasters(
            fd, fa, elev, backend='dask', chunks=(2, 1))

        result_np = hand(fd_np, fa_np, el_np, threshold=5)
        result_da = hand(fd_da, fa_da, el_da, threshold=5)

        np.testing.assert_allclose(
            result_np.data, result_da.data.compute(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
class TestHandCuPy:
    """Cross-backend: numpy vs cupy."""

    def test_numpy_equals_cupy(self):
        from xrspatial.hydro.flow_direction_d8 import flow_direction_d8 as flow_direction
        from xrspatial.hydro.flow_accumulation_d8 import flow_accumulation_d8 as flow_accumulation

        np.random.seed(42)
        elev_data = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev_data, backend='numpy', name='elev')
        fd_r = flow_direction(elev_r)
        fa_r = flow_accumulation(fd_r)
        fd_data = fd_r.data.astype(np.float64)
        fa_data = fa_r.data.astype(np.float64)

        fd_np, fa_np, el_np = _make_hand_rasters(
            fd_data, fa_data, elev_data, backend='numpy')
        fd_cu, fa_cu, el_cu = _make_hand_rasters(
            fd_data, fa_data, elev_data, backend='cupy')

        result_np = hand(fd_np, fa_np, el_np, threshold=5)
        result_cu = hand(fd_cu, fa_cu, el_cu, threshold=5)

        np.testing.assert_allclose(
            result_np.data, result_cu.data.get(),
            equal_nan=True, rtol=1e-10)


@cuda_and_cupy_available
@dask_array_available
class TestHandDaskCuPy:
    """Cross-backend: numpy vs dask+cupy."""

    def test_numpy_equals_dask_cupy(self):
        from xrspatial.hydro.flow_direction_d8 import flow_direction_d8 as flow_direction
        from xrspatial.hydro.flow_accumulation_d8 import flow_accumulation_d8 as flow_accumulation

        np.random.seed(42)
        elev_data = np.random.uniform(0, 100, (6, 6)).astype(np.float64)
        elev_r = create_test_raster(elev_data, backend='numpy', name='elev')
        fd_r = flow_direction(elev_r)
        fa_r = flow_accumulation(fd_r)
        fd_data = fd_r.data.astype(np.float64)
        fa_data = fa_r.data.astype(np.float64)

        fd_np, fa_np, el_np = _make_hand_rasters(
            fd_data, fa_data, elev_data, backend='numpy')
        fd_dc, fa_dc, el_dc = _make_hand_rasters(
            fd_data, fa_data, elev_data, backend='dask+cupy', chunks=(3, 3))

        result_np = hand(fd_np, fa_np, el_np, threshold=5)
        result_dc = hand(fd_dc, fa_dc, el_dc, threshold=5)

        np.testing.assert_allclose(
            result_np.data, result_dc.data.compute().get(),
            equal_nan=True, rtol=1e-10)


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

class TestMemoryGuard:
    """Memory guard on the eager numpy / cupy backends."""

    def test_numpy_huge_raster_raises(self):
        """Numpy backend raises MemoryError when projected RAM exceeds budget."""
        from unittest.mock import patch

        fd = np.zeros((4, 4), dtype=np.float64)
        fa = np.zeros((4, 4), dtype=np.float64)
        el = np.zeros((4, 4), dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_rasters(fd, fa, el, backend='numpy')

        with patch(
            "xrspatial.hydro.hand_d8._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                hand(fd_r, fa_r, el_r, threshold=1)

    def test_numpy_normal_input_succeeds(self):
        """Normal-size raster passes the guard with real memory."""
        fd = np.zeros((10, 10), dtype=np.float64)
        fa = np.zeros((10, 10), dtype=np.float64)
        el = np.zeros((10, 10), dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_rasters(fd, fa, el, backend='numpy')
        result = hand(fd_r, fa_r, el_r, threshold=1)
        assert result.shape == (10, 10)

    @dask_array_available
    def test_dask_path_skips_guard(self):
        """Dask backend bypasses the guard -- per-tile allocations are bounded."""
        from unittest.mock import patch

        fd = np.zeros((6, 6), dtype=np.float64)
        fa = np.zeros((6, 6), dtype=np.float64)
        el = np.zeros((6, 6), dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_rasters(
            fd, fa, el, backend='dask', chunks=(3, 3))

        with patch(
            "xrspatial.hydro.hand_d8._available_memory_bytes",
            return_value=1,
        ):
            result = hand(fd_r, fa_r, el_r, threshold=1)
            _ = result.data[:3, :3].compute()

    def test_error_message_mentions_dimensions(self):
        """The error message should mention the grid dimensions and dask."""
        from unittest.mock import patch

        fd = np.zeros((7, 9), dtype=np.float64)
        fa = np.zeros((7, 9), dtype=np.float64)
        el = np.zeros((7, 9), dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_rasters(fd, fa, el, backend='numpy')

        with patch(
            "xrspatial.hydro.hand_d8._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match=r"7x9.*dask"):
                hand(fd_r, fa_r, el_r, threshold=1)

    @cuda_and_cupy_available
    def test_cupy_huge_raster_raises(self):
        """CuPy backend raises MemoryError when projected GPU RAM exceeds budget."""
        from unittest.mock import patch

        fd = np.zeros((4, 4), dtype=np.float64)
        fa = np.zeros((4, 4), dtype=np.float64)
        el = np.zeros((4, 4), dtype=np.float64)
        fd_r, fa_r, el_r = _make_hand_rasters(fd, fa, el, backend='cupy')

        with patch(
            "xrspatial.hydro.hand_d8._available_gpu_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="GPU working memory"):
                hand(fd_r, fa_r, el_r, threshold=1)
