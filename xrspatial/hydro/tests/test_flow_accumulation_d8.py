import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro import flow_accumulation
from xrspatial.hydro.flow_accumulation_d8 import (
    _D8_DX,
    _D8_DY,
    _code_to_offset,
    _code_to_offset_py,
)
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ---------------------------------------------------------------------------
# Known bowl
# ---------------------------------------------------------------------------

def test_known_bowl():
    """5x5 bowl: hand-computed flow_dir and expected accum.

    Bowl surface:
        9  9  9  9  9
        9  8  7  6  9
        9  7  5  4  9
        9  6  4  3  9
        9  9  9  9  9

    Flow directions (edges NaN, interior computed with cx=cy=0.5):
        NaN NaN NaN NaN NaN
        NaN  2   2   4  NaN
        NaN  2   2   4  NaN
        NaN  1   1   0  NaN
        NaN NaN NaN NaN NaN

    Pit at (3,3) collects all 9 interior cells.
    """
    flow_dir = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 1, 1, 0, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    expected = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 1, 1, 1, np.nan],
        [np.nan, 1, 2, 3, np.nan],
        [np.nan, 1, 3, 9, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    np.testing.assert_allclose(result.data, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# Linear chain (east)
# ---------------------------------------------------------------------------

def test_linear_chain_east():
    """All cells flow east, last cell pit. accum = [1, 2, ..., N]."""
    N = 6
    flow_dir = np.full((1, N), 1.0, dtype=np.float64)
    flow_dir[0, -1] = 0.0  # last cell is a pit
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    expected = np.arange(1, N + 1, dtype=np.float64).reshape(1, N)
    np.testing.assert_allclose(result.data, expected)


# ---------------------------------------------------------------------------
# Pit center
# ---------------------------------------------------------------------------

def test_pit_center():
    """3x3 grid, all 8 neighbours flow to center -> center accum = 9."""
    flow_dir = np.array([
        [2.0, 4.0, 8.0],
        [1.0, 0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    assert result.data[1, 1] == 9.0
    # Edge cells have accum = 1 (only themselves)
    for r, c in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2),
                 (2, 0), (2, 1), (2, 2)]:
        assert result.data[r, c] == 1.0, f"Cell ({r},{c}) = {result.data[r,c]}"


# ---------------------------------------------------------------------------
# All pits
# ---------------------------------------------------------------------------

def test_all_pits():
    """Every cell code = 0 -> every cell accum = 1."""
    flow_dir = np.zeros((4, 5), dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    np.testing.assert_array_equal(result.data, 1.0)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_nan_handling():
    """NaN flow_dir -> NaN accum; valid neighbours still correct."""
    flow_dir = np.array([
        [1.0, 1.0, 0.0],
        [np.nan, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    assert np.isnan(result.data[1, 0])
    expected = np.array([
        [1.0, 2.0, 3.0],
        [np.nan, 1.0, 2.0],
        [1.0, 2.0, 3.0],
    ])
    np.testing.assert_allclose(result.data, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# Code 0 pit has accum = 1
# ---------------------------------------------------------------------------

def test_code_0_pit():
    """Pit cells (code 0) count themselves -> accum = 1."""
    flow_dir = np.array([
        [1.0, 0.0],
        [64.0, 0.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    # (1,0) -> N -> (0,0) -> E -> (0,1) pit
    assert result.data[1, 0] == 1.0  # leaf, just itself
    assert result.data[0, 0] == 2.0  # itself + (1,0)
    assert result.data[0, 1] == 3.0  # itself + (0,0) chain
    assert result.data[1, 1] == 1.0  # pit, just itself


# ---------------------------------------------------------------------------
# Valid output values
# ---------------------------------------------------------------------------

def test_valid_output_values():
    """All output values are finite positive or NaN."""
    flow_dir = np.array([
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 2, 4, np.nan],
        [np.nan, 1, 0, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    vals = result.data.ravel()
    for v in vals:
        assert np.isnan(v) or v >= 1.0, f"Invalid value: {v}"


# ---------------------------------------------------------------------------
# Uses float64
# ---------------------------------------------------------------------------

def test_output_dtype():
    """Output dtype is float64."""
    flow_dir = np.zeros((3, 3), dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    assert result.data.dtype == np.float64


# ---------------------------------------------------------------------------
# Dtype acceptance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_dtype_acceptance(dtype):
    """Function accepts int32/int64/float32/float64 flow_dir inputs."""
    # All zeros (pits) -- valid for any dtype
    flow_dir = np.zeros((3, 4), dtype=dtype)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    assert result.shape == agg.shape
    assert result.data.dtype == np.float64


# ---------------------------------------------------------------------------
# Dataset support
# ---------------------------------------------------------------------------

def test_dataset_support():
    """@supports_dataset works."""
    flow_dir = np.zeros((3, 4), dtype=np.float64)
    da1 = xr.DataArray(flow_dir, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(1, 0, 3)
    da1['x'] = np.linspace(0, 1.5, 4)
    ds = xr.Dataset({'fd1': da1, 'fd2': da1.copy()})
    result = flow_accumulation(ds)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'fd1', 'fd2'}
    for var in result.data_vars:
        np.testing.assert_array_equal(result[var].data, 1.0)


# ---------------------------------------------------------------------------
# Cross-backend tests
# ---------------------------------------------------------------------------

def _make_cross_backend_flow_dir():
    """6x6 flow-east grid for cross-backend comparison."""
    flow_dir = np.full((6, 6), 1.0, dtype=np.float64)
    flow_dir[:, -1] = 0.0  # last column = pits
    return flow_dir


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (3, 3), (5, 5), (2, 6), (6, 2), (1, 1), (6, 6),
])
def test_chunk_configs(chunks):
    """Multiple chunk sizes all match numpy result."""
    flow_dir = _make_cross_backend_flow_dir()
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    dask_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    np_result = flow_accumulation(numpy_agg)
    da_result = flow_accumulation(dask_agg)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True)


@dask_array_available
def test_dask_cross_tile_flow_north():
    """Flow northward across tile boundary matches numpy."""
    flow_dir = np.full((6, 4), 64.0, dtype=np.float64)
    flow_dir[0, :] = 0.0  # top row = pits
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    dask_agg = create_test_raster(flow_dir, backend='dask', chunks=(3, 4))
    np_result = flow_accumulation(numpy_agg)
    da_result = flow_accumulation(dask_agg)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True)


@dask_array_available
def test_dask_diagonal_cross_tile():
    """Diagonal SE flow across tile boundary matches numpy."""
    flow_dir = np.full((6, 6), 2.0, dtype=np.float64)  # all SE
    flow_dir[-1, -1] = 0.0  # bottom-right pit
    # Cells flowing outside grid just lose their accum
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    dask_agg = create_test_raster(flow_dir, backend='dask', chunks=(3, 3))
    np_result = flow_accumulation(numpy_agg)
    da_result = flow_accumulation(dask_agg)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True)


@dask_array_available
def test_dask_bowl():
    """Bowl flow_dir with dask matches numpy."""
    flow_dir = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 1, 1, 0, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    for chunks in [(2, 2), (3, 3), (5, 5), (1, 1)]:
        dask_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
        np_result = flow_accumulation(numpy_agg)
        da_result = flow_accumulation(dask_agg)
        np.testing.assert_allclose(
            np_result.data, da_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"


@dask_array_available
def test_dask_random():
    """Random acyclic flow_dir: dask matches numpy across chunk sizes.

    Use flow_direction() on random elevation to guarantee acyclic flow
    graphs (randomly assigned D8 codes can create cycles, which the
    iterative tile sweep cannot handle).
    """
    from xrspatial.hydro import flow_direction

    rng = np.random.default_rng(42)
    elev = rng.random((10, 12)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    flow_dir = flow_direction(elev_da).data

    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    np_result = flow_accumulation(numpy_agg)

    for chunks in [(3, 3), (5, 6), (2, 4), (10, 12)]:
        dask_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
        da_result = flow_accumulation(dask_agg)
        np.testing.assert_allclose(
            np_result.data, da_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"


@cuda_and_cupy_available
def test_numpy_equals_cupy():
    """CuPy matches NumPy."""
    flow_dir = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 2, 2, 4, np.nan],
        [np.nan, 1, 1, 0, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    cupy_agg = create_test_raster(flow_dir, backend='cupy')
    np_result = flow_accumulation(numpy_agg)
    cp_result = flow_accumulation(cupy_agg)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@cuda_and_cupy_available
def test_cupy_pit_center():
    """CuPy: pit center with 8 neighbours draining in."""
    flow_dir = np.array([
        [2.0, 4.0, 8.0],
        [1.0, 0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    cupy_agg = create_test_raster(flow_dir, backend='cupy')
    result = flow_accumulation(cupy_agg)
    data = result.data.get()
    assert data[1, 1] == 9.0


@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy():
    """Dask+CuPy matches NumPy."""
    flow_dir = _make_cross_backend_flow_dir()
    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    dask_cupy_agg = create_test_raster(flow_dir, backend='dask+cupy',
                                       chunks=(3, 3))
    np_result = flow_accumulation(numpy_agg)
    dcp_result = flow_accumulation(dask_cupy_agg)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("chunks", [(3, 3), (5, 6), (2, 4)])
def test_dask_cupy_chunk_configs(chunks):
    """Dask+CuPy matches NumPy across chunk sizes."""
    from xrspatial.hydro import flow_direction

    rng = np.random.default_rng(952)
    elev = rng.random((10, 12)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    flow_dir = flow_direction(elev_da).data

    numpy_agg = create_test_raster(flow_dir, backend='numpy')
    dcp_agg = create_test_raster(flow_dir, backend='dask+cupy', chunks=chunks)
    np_result = flow_accumulation(numpy_agg)
    dcp_result = flow_accumulation(dcp_agg)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True,
    ), f"Mismatch with chunks={chunks}"


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

class TestMemoryGuard:
    """Memory guard on the eager numpy / cupy backends."""

    def test_numpy_huge_raster_raises(self):
        """Numpy backend raises MemoryError when projected RAM exceeds budget."""
        from unittest.mock import patch

        flow_dir = np.zeros((4, 4), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='numpy')

        with patch(
            "xrspatial.hydro.flow_accumulation_d8._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                flow_accumulation(agg)

    def test_numpy_normal_input_succeeds(self):
        """Normal-size raster passes the guard with real memory."""
        flow_dir = np.zeros((10, 10), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='numpy')
        result = flow_accumulation(agg)
        assert result.shape == (10, 10)

    @dask_array_available
    def test_dask_path_skips_guard(self):
        """Dask backend bypasses the guard -- per-tile allocations are bounded."""
        from unittest.mock import patch

        flow_dir = np.zeros((20, 20), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='dask+numpy', chunks=(5, 5))

        with patch(
            "xrspatial.hydro.flow_accumulation_d8._available_memory_bytes",
            return_value=1,
        ):
            result = flow_accumulation(agg)
            _ = result.data[:4, :4].compute()

    def test_error_message_mentions_dask(self):
        """The error message should suggest the dask alternative."""
        from unittest.mock import patch

        flow_dir = np.zeros((4, 4), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='numpy')

        with patch(
            "xrspatial.hydro.flow_accumulation_d8._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="dask"):
                flow_accumulation(agg)


# ---------------------------------------------------------------------------
# Degenerate-shape tests (issue #2713)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1)])
def test_degenerate_shape(shape):
    """Single-pixel and single-row/column input.

    Every cell is a pit (code 0) that drains only itself, so each cell
    accumulates exactly 1 at the input shape.
    """
    flow_dir = np.zeros(shape, dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = flow_accumulation(agg)
    assert result.shape == shape
    assert not np.isnan(result.data).any()
    np.testing.assert_array_equal(result.data, 1.0)


# ---------------------------------------------------------------------------
# Weighted accumulation (#3734)
# ---------------------------------------------------------------------------

_BOWL_FLOW_DIR = np.array([
    [np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, 2, 2, 4, np.nan],
    [np.nan, 2, 2, 4, np.nan],
    [np.nan, 1, 1, 0, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan],
], dtype=np.float64)


def test_weight_known_values():
    """Each cell is the sum of weight over itself and its upstream cells."""
    flow_dir = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
    ], dtype=np.float64)
    weight = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 0.0, -1.0, 2.0],
    ], dtype=np.float64)
    expected = np.array([
        [1.0, 3.0, 6.0, 10.0],
        [0.5, 0.5, -0.5, 1.5],
    ])
    agg = create_test_raster(flow_dir)
    w = create_test_raster(weight)
    result = flow_accumulation(agg, weight=w)
    np.testing.assert_allclose(result.data, expected)


def test_weight_ones_equals_count():
    """weight=1 everywhere reproduces the unweighted cell count."""
    agg = create_test_raster(_BOWL_FLOW_DIR)
    w = create_test_raster(np.ones_like(_BOWL_FLOW_DIR))
    np.testing.assert_allclose(
        flow_accumulation(agg, weight=w).data,
        flow_accumulation(agg).data, equal_nan=True)


def test_weight_nan_contributes_zero():
    """NaN weight at a valid cell adds 0; the NaN mask still follows flow_dir."""
    flow_dir = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [np.nan, 1.0, 1.0, 0.0],
    ], dtype=np.float64)
    weight = np.array([
        [1.0, np.nan, 3.0, 4.0],
        [7.0, 2.0, np.nan, 1.0],
    ], dtype=np.float64)
    expected = np.array([
        [1.0, 1.0, 4.0, 8.0],
        [np.nan, 2.0, 2.0, 3.0],
    ])
    agg = create_test_raster(flow_dir)
    w = create_test_raster(weight)
    result = flow_accumulation(agg, weight=w)
    np.testing.assert_allclose(result.data, expected, equal_nan=True)
    assert np.array_equal(np.isnan(result.data), np.isnan(flow_dir))


def test_weight_shape_mismatch_raises():
    agg = create_test_raster(_BOWL_FLOW_DIR)
    w = create_test_raster(np.ones((5, 4)))
    with pytest.raises(ValueError, match="weight"):
        flow_accumulation(agg, weight=w)


def test_weight_not_dataarray_raises():
    agg = create_test_raster(_BOWL_FLOW_DIR)
    with pytest.raises(TypeError, match="weight"):
        flow_accumulation(agg, weight=np.ones((5, 5)))


def _weighted_reference():
    flow_dir = _make_cross_backend_flow_dir()
    rng = np.random.default_rng(3734)
    weight = rng.random(flow_dir.shape) * 10.0
    weight[2, 3] = np.nan
    weight[5, 1] = -4.0
    np_result = flow_accumulation(
        create_test_raster(flow_dir), weight=create_test_raster(weight))
    return flow_dir, weight, np_result.data


@dask_array_available
@pytest.mark.parametrize("chunks", [(3, 3), (5, 5), (2, 6), (1, 1), (6, 6)])
def test_weight_dask_equals_numpy(chunks):
    """Weighted dask matches numpy for every chunk layout."""
    flow_dir, weight, expected = _weighted_reference()
    dask_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    dask_w = create_test_raster(weight, backend='dask', chunks=chunks)
    np.testing.assert_allclose(
        flow_accumulation(dask_agg, weight=dask_w).data.compute(),
        expected, equal_nan=True)


@dask_array_available
def test_weight_dask_rechunks_weight():
    """A numpy weight or one with different chunks is aligned to flow_dir."""
    flow_dir, weight, expected = _weighted_reference()
    dask_agg = create_test_raster(flow_dir, backend='dask', chunks=(3, 3))
    np.testing.assert_allclose(
        flow_accumulation(dask_agg, weight=create_test_raster(weight))
        .data.compute(), expected, equal_nan=True)
    dask_w = create_test_raster(weight, backend='dask', chunks=(2, 5))
    np.testing.assert_allclose(
        flow_accumulation(dask_agg, weight=dask_w).data.compute(),
        expected, equal_nan=True)


@cuda_and_cupy_available
def test_weight_cupy_equals_numpy():
    flow_dir, weight, expected = _weighted_reference()
    cupy_agg = create_test_raster(flow_dir, backend='cupy')
    cupy_w = create_test_raster(weight, backend='cupy')
    np.testing.assert_allclose(
        flow_accumulation(cupy_agg, weight=cupy_w).data.get(),
        expected, equal_nan=True)


@cuda_and_cupy_available
def test_weight_dask_cupy_equals_numpy():
    flow_dir, weight, expected = _weighted_reference()
    agg = create_test_raster(flow_dir, backend='dask+cupy', chunks=(3, 3))
    w = create_test_raster(weight, backend='dask+cupy', chunks=(3, 3))
    np.testing.assert_allclose(
        flow_accumulation(agg, weight=w).data.compute().get(),
        expected, equal_nan=True)


def test_weight_integer_dtype():
    """Integer weights are coerced to float64."""
    agg = create_test_raster(_BOWL_FLOW_DIR)
    w_int = create_test_raster(np.full((5, 5), 3, dtype=np.int32))
    w_float = create_test_raster(np.full((5, 5), 3.0))
    result = flow_accumulation(agg, weight=w_int)
    assert result.dtype == np.float64
    np.testing.assert_allclose(
        result.data, flow_accumulation(agg, weight=w_float).data, equal_nan=True)


def test_weight_dataset_input():
    """A Dataset flow_dir applies the same weight to every variable."""
    agg = create_test_raster(_BOWL_FLOW_DIR)
    w = create_test_raster(np.full((5, 5), 2.0))
    ds = xr.Dataset({'a': agg, 'b': agg})
    out = flow_accumulation(ds, weight=w)
    expected = flow_accumulation(agg, weight=w).data
    for var in ('a', 'b'):
        np.testing.assert_allclose(out[var].data, expected, equal_nan=True)


def test_weight_forwarded_by_accessor_and_routing():
    """weight= reaches the router via flow_accumulation(routing=) and .xrs."""
    import xrspatial.accessor  # noqa: F401
    agg = create_test_raster(_BOWL_FLOW_DIR)
    w = create_test_raster(np.full((5, 5), 2.0))
    expected = flow_accumulation(agg, weight=w).data
    np.testing.assert_allclose(
        flow_accumulation(agg, weight=w, routing='d8').data, expected,
        equal_nan=True)
    np.testing.assert_allclose(
        agg.xrs.flow_accumulation(weight=w).data, expected, equal_nan=True)


def test_weight_dataset_accessor():
    """ds.xrs.flow_accumulation(weight=) weights every variable."""
    import xrspatial.accessor  # noqa: F401
    agg = create_test_raster(_BOWL_FLOW_DIR)
    w = create_test_raster(np.full((5, 5), 2.0))
    ds = xr.Dataset({'a': agg, 'b': agg})
    out = ds.xrs.flow_accumulation(weight=w)
    expected = flow_accumulation(agg, weight=w).data
    for var in ('a', 'b'):
        np.testing.assert_allclose(out[var].data, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# D8 code -> (dy, dx) lookup (#3738)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code, expected", [
    # the eight valid codes: E, SE, S, SW, W, NW, N, NE
    (1, (0, 1)),
    (2, (1, 1)),
    (4, (1, 0)),
    (8, (1, -1)),
    (16, (0, -1)),
    (32, (-1, -1)),
    (64, (-1, 0)),
    (128, (-1, 1)),
    # float codes as they arrive from a float64 flow-direction raster
    (4.0, (1, 0)),
    (128.0, (-1, 1)),
    # no-flow / pit
    (0, (0, 0)),
    (0.0, (0, 0)),
    # in-range but not a power of two
    (3, (0, 0)),
    (5, (0, 0)),
    # outside the table
    (129, (0, 0)),
    (255, (0, 0)),
    (-1, (0, 0)),
    (1e9, (0, 0)),
    (-1e9, (0, 0)),
    # NaN: int(nan) is INT64_MIN inside numba, the guard must catch it
    # before the table is indexed (run under NUMBA_BOUNDSCHECK=1 to check)
    (np.nan, (0, 0)),
])
def test_code_to_offset_matches_if_chain(code, expected):
    dy, dx = _code_to_offset(code)
    assert (dy, dx) == expected
    assert isinstance(dy, (int, np.integer))
    assert isinstance(dx, (int, np.integer))
    if code == code:  # _code_to_offset_py raises on NaN like int(nan) does
        assert _code_to_offset_py(code) == expected


def test_code_to_offset_tables_only_populate_d8_codes():
    assert _D8_DY.shape == _D8_DX.shape == (129,)
    populated = np.flatnonzero((_D8_DY != 0) | (_D8_DX != 0))
    assert populated.tolist() == [1, 2, 4, 8, 16, 32, 64, 128]
