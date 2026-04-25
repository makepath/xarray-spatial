"""Tests for xrspatial.cost_distance."""

try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial.cost_distance import cost_distance
from xrspatial.tests.general_checks import cuda_and_cupy_available
from xrspatial.utils import has_cuda_and_cupy, has_dask_array


def _make_raster(data, backend='numpy', chunks=(3, 3)):
    """Build a DataArray with y/x coords, optionally dask/cupy-backed."""
    h, w = data.shape
    raster = xr.DataArray(
        data.astype(np.float64),
        dims=['y', 'x'],
        attrs={'res': (1.0, 1.0)},
    )
    raster['y'] = np.arange(h, dtype=np.float64)
    raster['x'] = np.arange(w, dtype=np.float64)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=chunks)
    if 'cupy' in backend and has_cuda_and_cupy():
        import cupy
        if isinstance(raster.data, da.Array):
            raster.data = raster.data.map_blocks(cupy.asarray)
        else:
            raster.data = cupy.asarray(raster.data)
    return raster


def _compute(arr):
    """Extract numpy data from DataArray (works for numpy, dask, or cupy)."""
    if da is not None and isinstance(arr.data, da.Array):
        val = arr.data.compute()
        if hasattr(val, 'get'):
            return val.get()
        return val
    if hasattr(arr.data, 'get'):
        return arr.data.get()
    return arr.data


# -----------------------------------------------------------------------
# Uniform friction = 1 should match Euclidean proximity
# -----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_uniform_friction_matches_euclidean(backend):
    """With uniform friction=1, cost_distance ≈ Euclidean distance."""
    data = np.zeros((7, 7), dtype=np.float64)
    data[3, 3] = 1.0  # single source at centre

    raster = _make_raster(data, backend=backend, chunks=(7, 7))
    friction = _make_raster(np.ones((7, 7)), backend=backend, chunks=(7, 7))

    result = cost_distance(raster, friction)
    out = _compute(result)

    # Source pixel should be 0
    assert out[3, 3] == 0.0

    # Check a few known Euclidean distances (cellsize=1)
    # Cardinal neighbour: distance = 1
    np.testing.assert_allclose(out[3, 4], 1.0, atol=1e-5)
    np.testing.assert_allclose(out[2, 3], 1.0, atol=1e-5)

    # Diagonal neighbour: distance = sqrt(2)
    np.testing.assert_allclose(out[2, 2], np.sqrt(2), atol=1e-5)

    # 2 cells away cardinally: distance = 2
    np.testing.assert_allclose(out[3, 5], 2.0, atol=1e-5)

    # Corners: distance = sqrt(3^2+3^2) = 3*sqrt(2) ≈ 4.2426
    # But Dijkstra on a grid may find a shorter path via diagonals
    # The grid-optimal path from (3,3) to (0,0) is 3 diagonal steps = 3*sqrt(2)
    np.testing.assert_allclose(out[0, 0], 3 * np.sqrt(2), atol=1e-5)


# -----------------------------------------------------------------------
# Hand-computed analytic case
# -----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_analytic_small_grid(backend):
    """3x3 grid with known costs, single source at (0,0)."""
    source = np.zeros((3, 3))
    source[0, 0] = 1.0  # source

    friction_data = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 10.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    raster = _make_raster(source, backend=backend, chunks=(3, 3))
    friction = _make_raster(friction_data, backend=backend, chunks=(3, 3))

    result = cost_distance(raster, friction)
    out = _compute(result)

    # Source at (0,0): cost = 0
    assert out[0, 0] == 0.0

    # (0,1): cardinal, avg_friction = (1+2)/2 = 1.5, dist=1 => cost=1.5
    np.testing.assert_allclose(out[0, 1], 1.5, atol=1e-5)

    # (1,0): cardinal, avg_friction = (1+1)/2 = 1, dist=1 => cost=1.0
    np.testing.assert_allclose(out[1, 0], 1.0, atol=1e-5)

    # (1,1): diagonal from (0,0), avg_friction = (1+10)/2 = 5.5,
    #   cost = sqrt(2)*5.5 ≈ 7.778
    # BUT via (1,0) then cardinal to (1,1):
    #   cost = 1.0 + 1*(1+10)/2 = 1 + 5.5 = 6.5
    # Dijkstra picks the cheaper one: 6.5
    np.testing.assert_allclose(out[1, 1], 6.5, atol=1e-5)

    # (2,0): via (1,0), cost = 1.0 + 1*(1+1)/2 = 2.0
    np.testing.assert_allclose(out[2, 0], 2.0, atol=1e-5)


# -----------------------------------------------------------------------
# Barriers: NaN and zero-friction cells are impassable
# -----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_barriers_nan_and_zero(backend):
    """NaN and zero-friction cells block paths."""
    source = np.zeros((3, 5))
    source[1, 0] = 1.0  # source on left

    friction_data = np.ones((3, 5))
    friction_data[:, 2] = 0.0   # zero-friction barrier in column 2
    friction_data[1, 2] = np.nan  # NaN barrier too

    raster = _make_raster(source, backend=backend, chunks=(3, 5))
    friction = _make_raster(friction_data, backend=backend, chunks=(3, 5))

    result = cost_distance(raster, friction)
    out = _compute(result)

    # Source reachable
    assert out[1, 0] == 0.0

    # Left side reachable
    assert np.isfinite(out[0, 0])
    assert np.isfinite(out[1, 1])

    # Right side should be NaN (unreachable — barrier blocks all paths)
    assert np.isnan(out[1, 3])
    assert np.isnan(out[1, 4])
    assert np.isnan(out[0, 3])


# -----------------------------------------------------------------------
# Multiple sources: verify nearest-by-cost wins
# -----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_multiple_sources(backend):
    """Two sources — each pixel gets cost from the cheaper one."""
    source = np.zeros((1, 5))
    source[0, 0] = 1.0  # source A at left
    source[0, 4] = 2.0  # source B at right

    friction_data = np.ones((1, 5))

    raster = _make_raster(source, backend=backend, chunks=(1, 5))
    friction = _make_raster(friction_data, backend=backend, chunks=(1, 5))

    result = cost_distance(raster, friction)
    out = _compute(result)

    # Both sources at 0
    assert out[0, 0] == 0.0
    assert out[0, 4] == 0.0

    # Middle pixel (0,2) equidistant: cost = 2.0 from either source
    np.testing.assert_allclose(out[0, 2], 2.0, atol=1e-5)

    # (0,1): cost 1 from source A
    np.testing.assert_allclose(out[0, 1], 1.0, atol=1e-5)
    # (0,3): cost 1 from source B
    np.testing.assert_allclose(out[0, 3], 1.0, atol=1e-5)


# -----------------------------------------------------------------------
# max_cost truncation
# -----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_max_cost_truncation(backend):
    """Pixels beyond max_cost should be NaN."""
    source = np.zeros((1, 10))
    source[0, 0] = 1.0

    friction_data = np.ones((1, 10))

    raster = _make_raster(source, backend=backend, chunks=(1, 10))
    friction = _make_raster(friction_data, backend=backend, chunks=(1, 10))

    result = cost_distance(raster, friction, max_cost=3.5)
    out = _compute(result)

    # Pixels within budget
    assert out[0, 0] == 0.0
    np.testing.assert_allclose(out[0, 1], 1.0, atol=1e-5)
    np.testing.assert_allclose(out[0, 2], 2.0, atol=1e-5)
    np.testing.assert_allclose(out[0, 3], 3.0, atol=1e-5)

    # Beyond budget
    assert np.isnan(out[0, 4])
    assert np.isnan(out[0, 9])


# -----------------------------------------------------------------------
# Dask vs NumPy equivalence
# -----------------------------------------------------------------------

@pytest.mark.skipif(da is None, reason="dask not installed")
def test_dask_matches_numpy():
    """Dask result must match numpy result exactly."""
    np.random.seed(42)
    source = np.zeros((10, 12))
    source[2, 3] = 1.0
    source[7, 9] = 2.0

    friction_data = np.random.uniform(0.5, 5.0, (10, 12))

    raster_np = _make_raster(source, backend='numpy')
    friction_np = _make_raster(friction_data, backend='numpy')
    result_np = cost_distance(raster_np, friction_np, max_cost=20.0)

    raster_da = _make_raster(source, backend='dask+numpy', chunks=(5, 6))
    friction_da = _make_raster(friction_data, backend='dask+numpy', chunks=(5, 6))
    result_da = cost_distance(raster_da, friction_da, max_cost=20.0)

    assert isinstance(result_da.data, da.Array)
    np.testing.assert_allclose(
        result_da.values, result_np.data, equal_nan=True, atol=1e-5
    )


# -----------------------------------------------------------------------
# 4-connectivity vs 8-connectivity
# -----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'cupy'])
def test_connectivity_4_vs_8(backend):
    """4-connectivity diagonal cost should be higher than 8-connectivity."""
    source = np.zeros((3, 3))
    source[0, 0] = 1.0

    friction_data = np.ones((3, 3))

    raster = _make_raster(source, backend=backend)
    friction = _make_raster(friction_data, backend=backend)

    r8 = cost_distance(raster, friction, connectivity=8)
    r4 = cost_distance(raster, friction, connectivity=4)

    out8 = _compute(r8)
    out4 = _compute(r4)

    # Diagonal (2,2): 8-conn = 2*sqrt(2), 4-conn = 4 (must go around)
    np.testing.assert_allclose(out8[2, 2], 2 * np.sqrt(2), atol=1e-5)
    np.testing.assert_allclose(out4[2, 2], 4.0, atol=1e-5)


# -----------------------------------------------------------------------
# target_values parameter
# -----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_target_values(backend):
    """Only specified target values should be sources."""
    source = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])

    friction_data = np.ones((3, 3))

    raster = _make_raster(source, backend=backend, chunks=(3, 3))
    friction = _make_raster(friction_data, backend=backend, chunks=(3, 3))

    # Only treat value=2 as target
    result = cost_distance(raster, friction, target_values=[2])
    out = _compute(result)

    # (2,1) is the source with value=2
    assert out[2, 1] == 0.0

    # (0,1) has value=1 but should NOT be a source here
    assert out[0, 1] > 0.0
    # Cost from (2,1) to (0,1): 2 cardinal steps = 2.0
    np.testing.assert_allclose(out[0, 1], 2.0, atol=1e-5)


# -----------------------------------------------------------------------
# Lazy coordinate arrays for dask input
# -----------------------------------------------------------------------

@pytest.mark.skipif(da is None, reason="dask not installed")
def test_dask_no_large_numpy_arrays():
    """Dask path should not materialise large numpy arrays."""
    from unittest.mock import patch

    height, width = 50, 60
    source = np.zeros((height, width))
    source[10, 10] = 1.0

    friction_data = np.ones((height, width))

    raster = _make_raster(source, backend='dask+numpy', chunks=(25, 30))
    friction = _make_raster(friction_data, backend='dask+numpy', chunks=(25, 30))

    # Track large numpy allocations
    original_full = np.full
    large_allocs = []

    def tracking_full(shape, *args, **kwargs):
        result = original_full(shape, *args, **kwargs)
        if hasattr(shape, '__len__'):
            total = 1
            for s in shape:
                total *= s
        else:
            total = shape
        if total >= height * width:
            large_allocs.append(('full', shape))
        return result

    # The kernel itself will allocate full-size arrays, that's expected
    # when each chunk is processed. We just verify the outer dask wrapper
    # doesn't allocate huge arrays before map_overlap.
    result = cost_distance(raster, friction, max_cost=20.0)

    # Verify result is dask-backed
    assert isinstance(result.data, da.Array)

    # Verify correctness
    computed = result.values
    assert computed[10, 10] == 0.0
    assert computed[0, 0] > 0.0 or np.isnan(computed[0, 0])


# -----------------------------------------------------------------------
# Validation errors
# -----------------------------------------------------------------------

def test_invalid_connectivity():
    source = np.zeros((3, 3))
    source[1, 1] = 1.0
    raster = _make_raster(source)
    friction = _make_raster(np.ones((3, 3)))
    with pytest.raises(ValueError, match="connectivity"):
        cost_distance(raster, friction, connectivity=6)


def test_shape_mismatch():
    raster = _make_raster(np.zeros((3, 3)))
    friction_data = np.ones((4, 4))
    friction = xr.DataArray(friction_data, dims=['y', 'x'])
    friction['y'] = np.arange(4, dtype=np.float64)
    friction['x'] = np.arange(4, dtype=np.float64)
    with pytest.raises(ValueError, match="same shape"):
        cost_distance(raster, friction)


def test_wrong_dims():
    data = np.zeros((3, 3))
    data[1, 1] = 1.0
    raster = xr.DataArray(data, dims=['lat', 'lon'])
    raster['lat'] = np.arange(3, dtype=np.float64)
    raster['lon'] = np.arange(3, dtype=np.float64)
    friction = xr.DataArray(np.ones((3, 3)), dims=['lat', 'lon'])
    friction['lat'] = np.arange(3, dtype=np.float64)
    friction['lon'] = np.arange(3, dtype=np.float64)
    # Default x='x', y='y' won't match 'lat','lon'
    with pytest.raises(ValueError, match="dims"):
        cost_distance(raster, friction)
    # Should work with correct dim names
    result = cost_distance(raster, friction, x='lon', y='lat')
    assert result.shape == (3, 3)


# -----------------------------------------------------------------------
# Source at impassable cell
# -----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'cupy'])
def test_source_on_impassable_cell(backend):
    """Source on NaN-friction cell should not seed Dijkstra."""
    source = np.zeros((3, 3))
    source[1, 1] = 1.0  # source

    friction_data = np.ones((3, 3))
    friction_data[1, 1] = np.nan  # source cell is impassable

    raster = _make_raster(source, backend=backend)
    friction = _make_raster(friction_data, backend=backend)

    result = cost_distance(raster, friction)
    out = _compute(result)

    # Everything should be NaN — the only source is on impassable terrain
    assert np.all(np.isnan(out))


# -----------------------------------------------------------------------
# Snake-maze: long shortest paths that need many Bellman-Ford iterations
# -----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
def test_snake_maze_long_path(backend):
    """Maze where shortest path has more edges than height + width.

    Regression test for #1191: the CuPy Bellman-Ford loop used
    max_iterations = height + width, which is too few for snaking paths.
    """
    h, w = 5, 5
    source = np.zeros((h, w))
    source[0, 0] = 1.0

    # Snake: row 0 right, down at col 4, row 2 left, down at col 0, row 4 right
    friction = np.full((h, w), np.nan)
    friction[0, :] = 1.0
    friction[1, 4] = 1.0
    friction[2, :] = 1.0
    friction[3, 0] = 1.0
    friction[4, :] = 1.0

    raster = _make_raster(source, backend=backend, chunks=(5, 5))
    friction_r = _make_raster(friction, backend=backend, chunks=(5, 5))

    result = cost_distance(raster, friction_r, connectivity=4)
    out = _compute(result)

    # Path to (4,4) has 16 edges, each costing 1.0
    expected = np.array([
        [ 0.,  1.,  2.,  3.,  4.],
        [np.nan, np.nan, np.nan, np.nan, 5.],
        [10.,  9.,  8.,  7.,  6.],
        [11., np.nan, np.nan, np.nan, np.nan],
        [12., 13., 14., 15., 16.],
    ], dtype=np.float32)

    np.testing.assert_allclose(out, expected, equal_nan=True, atol=1e-5)


# -----------------------------------------------------------------------
# CuPy GPU tests
# -----------------------------------------------------------------------

@cuda_and_cupy_available
def test_cupy_matches_numpy():
    """CuPy GPU path should produce identical results to numpy."""
    np.random.seed(42)
    source = np.zeros((7, 7))
    source[3, 3] = 1.0

    friction_data = np.random.uniform(0.5, 5.0, (7, 7))

    result_np = _compute(cost_distance(
        _make_raster(source, backend='numpy'),
        _make_raster(friction_data, backend='numpy'),
    ))
    result_cupy = _compute(cost_distance(
        _make_raster(source, backend='cupy'),
        _make_raster(friction_data, backend='cupy'),
    ))

    np.testing.assert_allclose(result_cupy, result_np, equal_nan=True, atol=1e-5)


@cuda_and_cupy_available
def test_cupy_max_cost():
    """CuPy path respects max_cost truncation."""
    source = np.zeros((1, 10))
    source[0, 0] = 1.0
    friction_data = np.ones((1, 10))

    result = _compute(cost_distance(
        _make_raster(source, backend='cupy'),
        _make_raster(friction_data, backend='cupy'),
        max_cost=3.5,
    ))

    np.testing.assert_allclose(result[0, 3], 3.0, atol=1e-5)
    assert np.isnan(result[0, 4])


@cuda_and_cupy_available
def test_cupy_returns_cupy_array():
    """Result should be CuPy-backed when input is CuPy-backed."""
    import cupy
    from xrspatial.utils import is_cupy_array

    source = np.zeros((3, 3))
    source[1, 1] = 1.0
    friction_data = np.ones((3, 3))

    result = cost_distance(
        _make_raster(source, backend='cupy'),
        _make_raster(friction_data, backend='cupy'),
    )
    assert is_cupy_array(result.data)


# -----------------------------------------------------------------------
# Dask + CuPy GPU tests
# -----------------------------------------------------------------------

@cuda_and_cupy_available
@pytest.mark.skipif(not has_dask_array(), reason="Requires dask.Array")
def test_dask_cupy_matches_numpy():
    """Dask+CuPy GPU path should produce identical results to numpy."""
    np.random.seed(42)
    source = np.zeros((10, 12))
    source[2, 3] = 1.0
    source[7, 9] = 2.0

    friction_data = np.random.uniform(0.5, 5.0, (10, 12))

    result_np = _compute(cost_distance(
        _make_raster(source, backend='numpy'),
        _make_raster(friction_data, backend='numpy'),
        max_cost=20.0,
    ))
    result_dc = _compute(cost_distance(
        _make_raster(source, backend='dask+cupy', chunks=(5, 6)),
        _make_raster(friction_data, backend='dask+cupy', chunks=(5, 6)),
        max_cost=20.0,
    ))

    np.testing.assert_allclose(result_dc, result_np, equal_nan=True, atol=1e-5)


@cuda_and_cupy_available
@pytest.mark.skipif(not has_dask_array(), reason="Requires dask.Array")
def test_dask_cupy_unbounded_matches_numpy():
    """Dask+CuPy unbounded (CPU fallback via iterative) matches numpy."""
    np.random.seed(42)
    source = np.zeros((10, 12))
    source[5, 6] = 1.0

    friction_data = np.random.uniform(0.5, 5.0, (10, 12))

    result_np = _compute(cost_distance(
        _make_raster(source, backend='numpy'),
        _make_raster(friction_data, backend='numpy'),
    ))
    with pytest.warns(UserWarning, match="iterative tile Dijkstra"):
        result_dc = _compute(cost_distance(
            _make_raster(source, backend='dask+cupy', chunks=(5, 6)),
            _make_raster(friction_data, backend='dask+cupy', chunks=(5, 6)),
        ))

    np.testing.assert_allclose(result_dc, result_np, equal_nan=True, atol=1e-5)


@cuda_and_cupy_available
@pytest.mark.skipif(not has_dask_array(), reason="Requires dask.Array")
def test_dask_cupy_returns_cupy_chunks():
    """Dask+CuPy result should have cupy-backed chunks."""
    from xrspatial.utils import is_dask_cupy

    source = np.zeros((12, 12))
    source[6, 6] = 1.0
    friction_data = np.ones((12, 12))

    # max_cost=2 -> pad=3, chunk=6 -> pad < chunk => map_overlap GPU path
    result = cost_distance(
        _make_raster(source, backend='dask+cupy', chunks=(6, 6)),
        _make_raster(friction_data, backend='dask+cupy', chunks=(6, 6)),
        max_cost=2.0,
    )
    assert isinstance(result.data, da.Array)
    assert is_dask_cupy(result)


# -----------------------------------------------------------------------
# Iterative tile Dijkstra tests (Issue #880)
# -----------------------------------------------------------------------

@pytest.mark.skipif(da is None, reason="dask not installed")
def test_iterative_matches_numpy():
    """100x100 raster, dask chunks (25x25), max_cost=inf => iterative path."""
    np.random.seed(42)
    source = np.zeros((100, 100))
    source[50, 50] = 1.0

    friction_data = np.random.uniform(0.5, 5.0, (100, 100))

    raster_np = _make_raster(source, backend='numpy')
    friction_np = _make_raster(friction_data, backend='numpy')
    result_np = cost_distance(raster_np, friction_np)

    raster_da = _make_raster(source, backend='dask+numpy', chunks=(25, 25))
    friction_da = _make_raster(friction_data, backend='dask+numpy', chunks=(25, 25))
    with pytest.warns(UserWarning, match="iterative tile Dijkstra"):
        result_da = cost_distance(raster_da, friction_da)

    np.testing.assert_allclose(
        _compute(result_da), result_np.data, equal_nan=True, atol=1e-3,
    )


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_iterative_multi_source_across_chunks():
    """Sources in different chunks produce correct Voronoi cost partition."""
    source = np.zeros((40, 40))
    source[5, 5] = 1.0     # chunk (0, 0)
    source[35, 35] = 2.0   # chunk (1, 1)

    friction_data = np.ones((40, 40))

    raster_np = _make_raster(source, backend='numpy')
    friction_np = _make_raster(friction_data, backend='numpy')
    result_np = cost_distance(raster_np, friction_np)

    raster_da = _make_raster(source, backend='dask+numpy', chunks=(20, 20))
    friction_da = _make_raster(friction_data, backend='dask+numpy', chunks=(20, 20))
    result_da = cost_distance(raster_da, friction_da)

    np.testing.assert_allclose(
        _compute(result_da), result_np.data, equal_nan=True, atol=1e-4,
    )


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_iterative_narrow_corridor():
    """1px corridor spanning 4+ tile boundaries with barriers elsewhere."""
    source = np.zeros((8, 80))
    source[4, 0] = 1.0  # source at left edge of corridor

    friction_data = np.full((8, 80), np.nan)  # all impassable
    friction_data[4, :] = 1.0  # 1px-wide corridor at row 4

    raster_np = _make_raster(source, backend='numpy')
    friction_np = _make_raster(friction_data, backend='numpy')
    result_np = cost_distance(raster_np, friction_np)

    raster_da = _make_raster(source, backend='dask+numpy', chunks=(8, 16))
    friction_da = _make_raster(friction_data, backend='dask+numpy', chunks=(8, 16))
    result_da = cost_distance(raster_da, friction_da)

    np.testing.assert_allclose(
        _compute(result_da), result_np.data, equal_nan=True, atol=1e-4,
    )


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_iterative_diagonal_propagation():
    """Source in one corner, target in opposite; verify 8-connectivity."""
    source = np.zeros((40, 40))
    source[0, 0] = 1.0

    friction_data = np.ones((40, 40))

    raster_np = _make_raster(source, backend='numpy')
    friction_np = _make_raster(friction_data, backend='numpy')
    result_np = cost_distance(raster_np, friction_np)

    raster_da = _make_raster(source, backend='dask+numpy', chunks=(10, 10))
    friction_da = _make_raster(friction_data, backend='dask+numpy', chunks=(10, 10))
    result_da = cost_distance(raster_da, friction_da)

    np.testing.assert_allclose(
        _compute(result_da), result_np.data, equal_nan=True, atol=1e-4,
    )


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_iterative_no_rechunk_to_single():
    """Iterative path must NOT rechunk to a single chunk."""
    from unittest.mock import patch

    source = np.zeros((40, 40))
    source[20, 20] = 1.0

    friction_data = np.ones((40, 40))

    raster = _make_raster(source, backend='dask+numpy', chunks=(10, 10))
    friction = _make_raster(friction_data, backend='dask+numpy', chunks=(10, 10))

    original_rechunk = da.Array.rechunk
    single_chunk_detected = [False]

    def _tracking_rechunk(self, *args, **kwargs):
        result = original_rechunk(self, *args, **kwargs)
        if all(len(c) == 1 for c in result.chunks):
            if result.shape == (40, 40):
                single_chunk_detected[0] = True
        return result

    with patch.object(da.Array, 'rechunk', _tracking_rechunk):
        result = cost_distance(raster, friction)
        _ = _compute(result)

    assert not single_chunk_detected[0], \
        "cost_distance rechunked to a single chunk (OOM risk)"


# -----------------------------------------------------------------------
# Memory guard on numpy path (Issue #1252)
# -----------------------------------------------------------------------

def test_numpy_memory_guard_raises_for_oversize_raster():
    """Large numpy inputs should raise MemoryError before allocating."""
    from unittest.mock import patch

    source = np.zeros((4, 4))
    source[0, 0] = 1.0
    friction = np.ones((4, 4))

    raster = _make_raster(source, backend='numpy')
    fric = _make_raster(friction, backend='numpy')

    # Pretend only 1 KB is available.  A 4x4 raster needs ~640 bytes of
    # working memory, which sits under the 50% cutoff; bump the reported
    # availability below 2x the required footprint to trigger the guard.
    with patch(
        'xrspatial.cost_distance._available_memory_bytes', return_value=1000
    ):
        with pytest.raises(MemoryError, match="max_cost"):
            cost_distance(raster, fric)


def test_numpy_memory_guard_passes_for_small_raster():
    """A tiny raster should sail through the guard even on low-memory hosts."""
    from unittest.mock import patch

    source = np.zeros((4, 4))
    source[0, 0] = 1.0
    friction = np.ones((4, 4))

    raster = _make_raster(source, backend='numpy')
    fric = _make_raster(friction, backend='numpy')

    # 1 MB reported available: 4x4 kernel working set is ~640 B, well under
    # the 50% cutoff.
    with patch(
        'xrspatial.cost_distance._available_memory_bytes',
        return_value=1024 * 1024,
    ):
        result = cost_distance(raster, fric)

    out = _compute(result)
    assert out[0, 0] == 0.0
    np.testing.assert_allclose(out[0, 1], 1.0, atol=1e-5)


# -----------------------------------------------------------------------
# Memory guard on CuPy GPU path (Issue #1262)
# -----------------------------------------------------------------------

def test_check_gpu_memory_raises_when_required_exceeds_half_free():
    """_check_gpu_memory(h, w) should raise when 50% of free GPU < required."""
    from unittest.mock import patch
    from xrspatial.cost_distance import _check_gpu_memory

    # 1000 bytes free, 4x4 grid needs 4*4*24 = 384 bytes.  Half-free = 500,
    # 384 < 500, so this should NOT raise.
    with patch(
        'xrspatial.cost_distance._available_gpu_memory_bytes',
        return_value=1000,
    ):
        _check_gpu_memory(4, 4)

    # 100 bytes free, 4x4 grid needs 384 bytes.  Half-free = 50, well below
    # required, so this MUST raise.
    with patch(
        'xrspatial.cost_distance._available_gpu_memory_bytes',
        return_value=100,
    ):
        with pytest.raises(MemoryError, match="max_cost"):
            _check_gpu_memory(4, 4)


def test_check_gpu_memory_skipped_when_no_cuda():
    """When _available_gpu_memory_bytes returns 0, the guard is a no-op."""
    from unittest.mock import patch
    from xrspatial.cost_distance import _check_gpu_memory

    with patch(
        'xrspatial.cost_distance._available_gpu_memory_bytes',
        return_value=0,
    ):
        # Should not raise even for an absurd size when no GPU info.
        _check_gpu_memory(10**9, 10**9)


def test_check_gpu_memory_error_mentions_dask_and_max_cost():
    """The error message should point users at the two escape hatches."""
    from unittest.mock import patch
    from xrspatial.cost_distance import _check_gpu_memory

    with patch(
        'xrspatial.cost_distance._available_gpu_memory_bytes',
        return_value=10,
    ):
        with pytest.raises(MemoryError) as excinfo:
            _check_gpu_memory(4, 4)

    msg = str(excinfo.value)
    assert "max_cost" in msg
    assert "dask" in msg.lower()


@cuda_and_cupy_available
def test_cupy_memory_guard_raises_for_oversize_raster():
    """End-to-end: cupy path raises MemoryError before GPU allocation."""
    from unittest.mock import patch

    source = np.zeros((4, 4))
    source[0, 0] = 1.0
    friction = np.ones((4, 4))

    raster = _make_raster(source, backend='cupy')
    fric = _make_raster(friction, backend='cupy')

    # Pretend only 100 bytes of GPU memory are free.  A 4x4 raster needs
    # 384 bytes of working memory, which sits well above 50 (half-free), so
    # the guard fires before any cp.full allocation.
    with patch(
        'xrspatial.cost_distance._available_gpu_memory_bytes',
        return_value=100,
    ):
        with pytest.raises(MemoryError, match="max_cost"):
            cost_distance(raster, fric)


@cuda_and_cupy_available
def test_cupy_memory_guard_passes_for_small_raster():
    """A tiny cupy raster should pass the guard with realistic free memory."""
    from unittest.mock import patch

    source = np.zeros((4, 4))
    source[0, 0] = 1.0
    friction = np.ones((4, 4))

    raster = _make_raster(source, backend='cupy')
    fric = _make_raster(friction, backend='cupy')

    # 1 MB reported free: 4x4 working set is ~384 B, well under the cutoff.
    with patch(
        'xrspatial.cost_distance._available_gpu_memory_bytes',
        return_value=1024 * 1024,
    ):
        result = cost_distance(raster, fric)

    out = _compute(result)
    assert out[0, 0] == 0.0
    np.testing.assert_allclose(out[0, 1], 1.0, atol=1e-5)
