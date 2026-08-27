import numpy as np
import pytest

from xrspatial.hydro.flow_accumulation_dinf import flow_accumulation_dinf
from xrspatial.hydro.flow_accumulation_d8 import _detect_flow_type
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ===========================================================================
# Detection tests
# ===========================================================================

def test_dinf_detection_d8():
    """D8 codes detected as 'd8'."""
    flow_dir = np.array([[1.0, 2.0], [4.0, 0.0]], dtype=np.float64)
    assert _detect_flow_type(flow_dir) == "d8"


def test_dinf_detection_dinf():
    """Dinf angles detected as 'dinf'."""
    flow_dir = np.array([[0.5, 1.2], [-1.0, np.nan]], dtype=np.float64)
    assert _detect_flow_type(flow_dir) == "dinf"


def test_dinf_detection_all_nan():
    """All-NaN grid detected as 'd8' (default)."""
    flow_dir = np.full((3, 3), np.nan, dtype=np.float64)
    assert _detect_flow_type(flow_dir) == "d8"


def test_dinf_detection_pit_minus_one():
    """-1.0 (Dinf pit) triggers 'dinf' detection."""
    flow_dir = np.array([[0.0, -1.0], [0.0, 0.0]], dtype=np.float64)
    assert _detect_flow_type(flow_dir) == "dinf"


# ===========================================================================
# Core functionality tests
# ===========================================================================

def test_dinf_cardinal_east_chain():
    """Pure east angles (0.0) chain: matches D8 code=1 result."""
    N = 6
    # Dinf: angle=0 means east, pit=-1
    flow_dir = np.full((1, N), 0.0, dtype=np.float64)
    flow_dir[0, -1] = -1.0  # last cell is pit
    agg = create_test_raster(flow_dir)
    result = flow_accumulation_dinf(agg)
    expected = np.arange(1, N + 1, dtype=np.float64).reshape(1, N)
    np.testing.assert_allclose(result.data, expected)


def test_dinf_pit_center():
    """8 neighbours with exact cardinal/diagonal angles point to center."""
    pi = np.pi
    flow_dir = np.array([
        [7 * pi / 4,  3 * pi / 2,  5 * pi / 4],
        [0.0,         -1.0,         pi],
        [pi / 4,      pi / 2,       3 * pi / 4],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation_dinf(agg)
    assert result.data[1, 1] == 9.0
    for r, c in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2),
                 (2, 0), (2, 1), (2, 2)]:
        assert result.data[r, c] == 1.0, f"Cell ({r},{c}) = {result.data[r,c]}"


def test_dinf_proportional_split():
    """angle=pi/8 splits flow 50/50 to E and NE."""
    pi = np.pi
    flow_dir = np.array([
        [-1.0,  -1.0,  -1.0],
        [-1.0,  pi / 8, -1.0],
        [-1.0,  -1.0,  -1.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation_dinf(agg)
    # Center splits 50/50 to E (1,2) and NE (0,2)
    np.testing.assert_allclose(result.data[1, 2], 1.5)
    np.testing.assert_allclose(result.data[0, 2], 1.5)
    assert result.data[1, 1] == 1.0


def test_dinf_chain_with_split():
    """Multi-cell chain with fractional flow, verify cascading sums."""
    pi = np.pi
    flow_dir = np.array([
        [-1.0,   -1.0,   -1.0],
        [pi / 8, pi / 8, -1.0],
        [-1.0,   -1.0,   -1.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation_dinf(agg)
    # (1,0) sends 0.5 to E(1,1) and 0.5 to NE(0,1)
    # (1,1) has accum=1.5, sends 0.75 to E(1,2) and 0.75 to NE(0,2)
    np.testing.assert_allclose(result.data[1, 0], 1.0)
    np.testing.assert_allclose(result.data[0, 1], 1.5)
    np.testing.assert_allclose(result.data[1, 1], 1.5)
    np.testing.assert_allclose(result.data[1, 2], 1.75)
    np.testing.assert_allclose(result.data[0, 2], 1.75)


def test_dinf_end_to_end():
    """flow_direction_dinf -> flow_accumulation_dinf produces valid results."""
    from xrspatial.hydro import flow_direction_dinf

    rng = np.random.default_rng(42)
    elev = create_test_raster(rng.random((20, 20)) * 100)
    fdir = flow_direction_dinf(elev)
    acc = flow_accumulation_dinf(fdir)
    data = acc.data
    valid = data[~np.isnan(data)]
    assert len(valid) > 0
    assert np.all(valid >= 1.0)
    assert np.max(valid) <= data.size


def test_dinf_nan_handling():
    """NaN flow dir produces NaN accumulation."""
    pi = np.pi
    flow_dir = np.array([
        [np.nan,   pi / 2],
        [0.0,      -1.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation_dinf(agg)
    assert np.isnan(result.data[0, 0])
    np.testing.assert_allclose(result.data[0, 1], 1.0)
    np.testing.assert_allclose(result.data[1, 0], 1.0)
    np.testing.assert_allclose(result.data[1, 1], 2.0)


# ===========================================================================
# Cross-backend tests
# ===========================================================================

@dask_array_available
@pytest.mark.parametrize("chunks", [
    (3, 3), (5, 5), (2, 6), (6, 2), (1, 1), (6, 6),
])
def test_dinf_dask_matches_numpy(chunks):
    """Dinf dask matches numpy across chunk sizes."""
    from xrspatial.hydro import flow_direction_dinf

    rng = np.random.default_rng(123)
    elev = create_test_raster(rng.random((6, 6)) * 100)
    fdir_np = flow_direction_dinf(elev).data

    numpy_agg = create_test_raster(fdir_np, backend='numpy')
    dask_agg = create_test_raster(fdir_np, backend='dask', chunks=chunks)
    np_result = flow_accumulation_dinf(numpy_agg)
    da_result = flow_accumulation_dinf(dask_agg)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True,
    ), f"Mismatch with chunks={chunks}"


@dask_array_available
def test_dinf_dask_larger_grid():
    """Dinf dask on a larger grid with cross-tile flow."""
    from xrspatial.hydro import flow_direction_dinf

    rng = np.random.default_rng(99)
    elev = create_test_raster(rng.random((12, 12)) * 100)
    fdir_np = flow_direction_dinf(elev).data

    numpy_agg = create_test_raster(fdir_np, backend='numpy')
    np_result = flow_accumulation_dinf(numpy_agg)

    for chunks in [(3, 3), (4, 6), (6, 4), (2, 2)]:
        dask_agg = create_test_raster(fdir_np, backend='dask', chunks=chunks)
        da_result = flow_accumulation_dinf(dask_agg)
        np.testing.assert_allclose(
            np_result.data, da_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"


@cuda_and_cupy_available
def test_dinf_cupy_matches_numpy():
    """Dinf CuPy matches NumPy."""
    from xrspatial.hydro import flow_direction_dinf

    rng = np.random.default_rng(42)
    elev = create_test_raster(rng.random((10, 10)) * 100)
    fdir_np = flow_direction_dinf(elev).data

    numpy_agg = create_test_raster(fdir_np, backend='numpy')
    cupy_agg = create_test_raster(fdir_np, backend='cupy')
    np_result = flow_accumulation_dinf(numpy_agg)
    cp_result = flow_accumulation_dinf(cupy_agg)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_dinf_dask_cupy_matches_numpy():
    """Dinf Dask+CuPy matches NumPy."""
    from xrspatial.hydro import flow_direction_dinf

    rng = np.random.default_rng(42)
    elev = create_test_raster(rng.random((8, 8)) * 100)
    fdir_np = flow_direction_dinf(elev).data

    numpy_agg = create_test_raster(fdir_np, backend='numpy')
    dask_cupy_agg = create_test_raster(fdir_np, backend='dask+cupy',
                                        chunks=(4, 4))
    np_result = flow_accumulation_dinf(numpy_agg)
    dcp_result = flow_accumulation_dinf(dask_cupy_agg)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)


# ===========================================================================
# Memory guard
# ===========================================================================

class TestMemoryGuard:
    """Memory guard on the eager numpy / cupy backends for D-inf."""

    def test_numpy_huge_raster_raises(self):
        """Numpy backend raises MemoryError when projected RAM exceeds budget."""
        from unittest.mock import patch

        flow_dir = np.zeros((4, 4), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='numpy')

        with patch(
            "xrspatial.hydro.flow_accumulation_dinf._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                flow_accumulation_dinf(agg)

    def test_numpy_normal_input_succeeds(self):
        """Normal-size raster passes the guard with real memory."""
        flow_dir = np.zeros((10, 10), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='numpy')
        result = flow_accumulation_dinf(agg)
        assert result.shape == (10, 10)

    @dask_array_available
    def test_dask_path_skips_guard(self):
        """Dask backend bypasses the guard -- per-tile allocations are bounded."""
        from unittest.mock import patch

        flow_dir = np.zeros((20, 20), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='dask+numpy', chunks=(5, 5))

        with patch(
            "xrspatial.hydro.flow_accumulation_dinf._available_memory_bytes",
            return_value=1,
        ):
            result = flow_accumulation_dinf(agg)
            _ = result.data[:4, :4].compute()

    def test_error_message_mentions_dimensions(self):
        """The error message names the grid dimensions so callers know what to shrink."""
        from unittest.mock import patch

        flow_dir = np.zeros((7, 11), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='numpy')

        with patch(
            "xrspatial.hydro.flow_accumulation_dinf._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match=r"7x11"):
                flow_accumulation_dinf(agg)


# ===========================================================================
# Weighted accumulation (#3734)
# ===========================================================================

def _dinf_weighted_case():
    """3x3 with the centre pit fed by cardinal + diagonal neighbours."""
    pi = np.pi
    flow_dir = np.array([
        [7 * pi / 4, 3 * pi / 2, 5 * pi / 4],
        [0.0,        -1.0,       pi],
        [pi / 4,     pi / 2,     3 * pi / 4],
    ], dtype=np.float64)
    weight = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 10.0, np.nan],
        [0.5, 1.5, -2.0],
    ], dtype=np.float64)
    return flow_dir, weight


def test_dinf_weight_known_values():
    """Centre pit collects the weights of all 8 neighbours plus its own."""
    flow_dir, weight = _dinf_weighted_case()
    result = flow_accumulation_dinf(
        create_test_raster(flow_dir), weight=create_test_raster(weight))
    # Each neighbour keeps its own weight (nothing upstream of it);
    # the NaN weight contributes 0 at (1, 2).
    expected = weight.copy()
    expected[1, 2] = 0.0
    expected[1, 1] = 10.0 + 1 + 2 + 3 + 4 + 0 + 0.5 + 1.5 - 2
    np.testing.assert_allclose(result.data, expected)
    assert not np.isnan(result.data).any()


def test_dinf_weight_ones_equals_count():
    flow_dir, _ = _dinf_weighted_case()
    agg = create_test_raster(flow_dir)
    np.testing.assert_allclose(
        flow_accumulation_dinf(agg, weight=create_test_raster(
            np.ones_like(flow_dir))).data,
        flow_accumulation_dinf(agg).data, equal_nan=True)


def test_dinf_weight_split_proportionally():
    """A weight flowing at pi/8 is split 50/50 between E and NE by D-inf."""
    pi = np.pi
    # Bottom-left cell flows at pi/8: half-way between E (0) and NE (pi/4).
    flow_dir = np.array([
        [-1.0, -1.0],
        [pi / 8, -1.0],
    ], dtype=np.float64)
    weight = np.array([[0.0, 0.0], [8.0, 0.0]], dtype=np.float64)
    result = flow_accumulation_dinf(
        create_test_raster(flow_dir), weight=create_test_raster(weight))
    np.testing.assert_allclose(result.data[1, 0], 8.0)
    np.testing.assert_allclose(result.data[1, 1], 4.0)  # E
    np.testing.assert_allclose(result.data[0, 1], 4.0)  # NE
    np.testing.assert_allclose(result.data[0, 0], 0.0)


def test_dinf_weight_shape_mismatch_raises():
    flow_dir, weight = _dinf_weighted_case()
    with pytest.raises(ValueError, match="weight"):
        flow_accumulation_dinf(create_test_raster(flow_dir),
                               weight=create_test_raster(weight[:, :2]))


def _dinf_weighted_reference():
    """Acyclic D-inf grid derived from a noisy tilted DEM.

    Random angles would form cycles, which the BFS does not define
    consistently across tilings, so derive directions from a surface.
    """
    from xrspatial.hydro import flow_direction_dinf
    rng = np.random.default_rng(3734)
    elev = rng.random((12, 15)) + np.add.outer(
        np.arange(12) * 0.5, np.arange(15) * 0.3)
    elev[6, 7] = np.nan
    flow_dir = flow_direction_dinf(create_test_raster(elev)).data
    weight = rng.random(flow_dir.shape) * 5.0
    weight[3, 4] = np.nan
    expected = flow_accumulation_dinf(
        create_test_raster(flow_dir), weight=create_test_raster(weight)).data
    return flow_dir, weight, expected


@dask_array_available
@pytest.mark.parametrize("chunks", [(3, 3), (4, 5), (12, 15), (1, 1)])
def test_dinf_weight_dask_equals_numpy(chunks):
    flow_dir, weight, expected = _dinf_weighted_reference()
    dask_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    for w_chunks in (chunks, (2, 7)):
        dask_w = create_test_raster(weight, backend='dask', chunks=w_chunks)
        np.testing.assert_allclose(
            flow_accumulation_dinf(dask_agg, weight=dask_w).data.compute(),
            expected, equal_nan=True)


@cuda_and_cupy_available
def test_dinf_weight_cupy_equals_numpy():
    flow_dir, weight, expected = _dinf_weighted_reference()
    np.testing.assert_allclose(
        flow_accumulation_dinf(
            create_test_raster(flow_dir, backend='cupy'),
            weight=create_test_raster(weight, backend='cupy')).data.get(),
        expected, equal_nan=True)


@cuda_and_cupy_available
def test_dinf_weight_dask_cupy_equals_numpy():
    flow_dir, weight, expected = _dinf_weighted_reference()
    np.testing.assert_allclose(
        flow_accumulation_dinf(
            create_test_raster(flow_dir, backend='dask+cupy', chunks=(4, 5)),
            weight=create_test_raster(weight, backend='dask+cupy',
                                      chunks=(4, 5))).data.compute().get(),
        expected, equal_nan=True)
