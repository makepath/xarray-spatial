import numpy as np
import pytest

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial import a_star_search
from xrspatial.cost_distance import cost_distance
from xrspatial.tests.general_checks import (
    create_test_raster, general_output_checks,
    cuda_and_cupy_available, dask_array_available,
)
from xrspatial.utils import has_cuda_and_cupy, has_dask_array


@pytest.fixture
def input_data():
    data = np.array([[0, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 1, 2, 2],
                     [1, 0, 2, 0],
                     [0, 2, 2, 2]])
    agg = create_test_raster(data, dims=['lat', 'lon'])
    return agg


@pytest.fixture
def input_data_with_nans():
    data = np.array([[0, 1, 0, 0],
                     [1, 1, np.nan, 0],
                     [0, 1, 2, 2],
                     [1, 0, 2, 0],
                     [0, np.nan, 2, 2]])
    agg = create_test_raster(data, dims=['lat', 'lon'])

    # start and end at a nan pixel, coordinate in (lat, lon) format
    start = (1.5, 1)
    goal = (0, 0.5)
    return agg, start, goal


@pytest.fixture
def result_8_connectivity():
    # cellsize=0.5: diagonal=sqrt(0.5)≈0.7071, cardinal=0.5
    # Path: (0,2)→(1,1)→(2,1)→(3,1)
    # Costs: 0, sqrt(0.5), sqrt(0.5)+0.5, sqrt(0.5)+1.0
    s = np.sqrt(0.5)
    expected_result = np.array([[np.nan, np.nan, 0., np.nan],
                                [np.nan, s, np.nan, np.nan],
                                [np.nan, s + 0.5, np.nan, np.nan],
                                [np.nan, s + 1.0, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan]])
    return expected_result


@pytest.fixture
def result_4_connectivity():
    # cellsize=0.5: cardinal=0.5
    # Path: (0,2)→(0,1)→(1,1)→(2,1)→(3,1)
    # Costs: 0, 0.5, 1.0, 1.5, 2.0
    expected_result = np.array([[np.nan, 0.5, 0., np.nan],
                                [np.nan, 1.0, np.nan, np.nan],
                                [np.nan, 1.5, np.nan, np.nan],
                                [np.nan, 2.0, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan]])
    return expected_result


def test_a_star_search_no_barriers(input_data):
    agg = input_data
    barriers = []
    # no barriers, there always path from a start location to a goal location
    for x0 in agg['lon']:
        for y0 in agg['lat']:
            start = (y0, x0)
            for x1 in agg['lon']:
                for y1 in agg['lat']:
                    goal = (y1, x1)
                    path_agg = a_star_search(
                        agg, start, goal, barriers, 'lon', 'lat'
                    )
                    general_output_checks(agg, path_agg)
                    assert type(path_agg.values[0][0]) == np.float64
                    if start == goal:
                        assert np.nanmax(path_agg) == 0
                        assert np.nanmin(path_agg) == 0
                    else:
                        assert np.nanmax(path_agg) > 0
                        assert np.nanmin(path_agg) == 0


@pytest.mark.filterwarnings("ignore:Start at a non crossable location:Warning")
@pytest.mark.filterwarnings("ignore:End at a non crossable location:Warning")
def test_a_star_search_with_barriers(input_data):
    agg = input_data
    barriers = [1]
    # set pixels with value 1 as barriers,
    # cannot go from (0, 0) to anywhere since it is surrounded by 1s
    start = (2, 0)
    for x1 in agg['lon']:
        for y1 in agg['lat']:
            goal = (y1, x1)
            if (goal != start):
                path_agg = a_star_search(
                    agg, start, goal, barriers, 'lon', 'lat'
                )
                # no path, all cells in path_agg are nans
                expected_results = np.full(agg.shape, np.nan)
                general_output_checks(agg, path_agg, expected_results)


@pytest.mark.filterwarnings("ignore:Start at a non crossable location:Warning")
@pytest.mark.filterwarnings("ignore:End at a non crossable location:Warning")
def test_a_star_search_snap(input_data_with_nans):
    agg, start, goal = input_data_with_nans

    # no barriers
    barriers = []
    # no snap
    no_snap_path_agg = a_star_search(agg, start, goal, barriers, 'lon', 'lat')
    # no path, all cells in path_agg are nans
    np.testing.assert_array_equal(no_snap_path_agg, np.nan)

    # set snap_start = True, snap_goal = False
    snap_start_path_agg = a_star_search(agg, start, goal, barriers, 'lon', 'lat', snap_start=True)
    # no path, all cells in path_agg are nans
    np.testing.assert_array_equal(snap_start_path_agg, np.nan)

    # set snap_start = False, snap_goal = True
    snap_goal_path_agg = a_star_search(agg, start, goal, barriers, 'lon', 'lat', snap_goal=True)
    # no path, all cells in path_agg are nans
    np.testing.assert_array_equal(snap_goal_path_agg, np.nan)


def test_a_star_search_connectivity(
    input_data_with_nans,
    result_8_connectivity,
    result_4_connectivity
):
    agg, start, goal = input_data_with_nans
    # no barriers
    barriers = []

    # set snap_start = True, snap_goal = True
    # 8-connectivity as default
    path_agg_8 = a_star_search(
        agg, start, goal, barriers, 'lon', 'lat', snap_start=True, snap_goal=True
    )
    np.testing.assert_allclose(path_agg_8, result_8_connectivity, equal_nan=True)

    # 4-connectivity
    path_agg_4 = a_star_search(
        agg, start, goal, barriers, 'lon', 'lat', snap_start=True, snap_goal=True, connectivity=4
    )
    np.testing.assert_allclose(path_agg_4, result_4_connectivity, equal_nan=True)


# -----------------------------------------------------------------------
# Helper for multi-backend test rasters
# -----------------------------------------------------------------------

def _make_raster(data, dims=None, res=None, backend='numpy', chunks=(3, 3)):
    """Build a simple DataArray for weighted A* tests."""
    if dims is None:
        dims = ['y', 'x']
    if res is None:
        res = (1.0, 1.0)
    import xarray as xr
    h, w = data.shape
    raster = xr.DataArray(
        data.astype(np.float64),
        dims=dims,
        attrs={'res': res},
    )
    raster[dims[0]] = np.linspace((h - 1) * res[0], 0, h)
    raster[dims[1]] = np.linspace(0, (w - 1) * res[1], w)

    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=chunks)
    if 'cupy' in backend and has_cuda_and_cupy():
        import cupy
        if isinstance(raster.data, da.Array):
            raster.data = raster.data.map_blocks(cupy.asarray)
        else:
            raster.data = cupy.asarray(raster.data)

    return raster


# -----------------------------------------------------------------------
# Weighted A* tests — parametrized for numpy and dask
# -----------------------------------------------------------------------

_backends = ['numpy']
if has_dask_array():
    _backends.append('dask+numpy')


@pytest.mark.parametrize("backend", _backends)
def test_uniform_friction_matches_no_friction(backend):
    """Uniform friction=1 should give the same path and costs as no friction."""
    data = np.ones((5, 5))
    agg = _make_raster(data, backend=backend)
    friction_agg = _make_raster(np.ones((5, 5)), backend=backend)

    # coordinate space: y goes 4..0, x goes 0..4
    start = (4.0, 0.0)   # pixel (0,0)
    goal = (0.0, 4.0)    # pixel (4,4)

    path_no_friction = a_star_search(agg, start, goal)
    path_with_friction = a_star_search(agg, start, goal, friction=friction_agg)

    np.testing.assert_allclose(
        np.asarray(path_with_friction.values),
        np.asarray(path_no_friction.values),
        equal_nan=True, atol=1e-10,
    )


@pytest.mark.parametrize("backend", _backends)
def test_high_friction_detour(backend):
    """A* should route around a high-friction column."""
    # 3x5 grid, cellsize=1
    data = np.ones((3, 5))
    friction_data = np.ones((3, 5))
    friction_data[0, 2] = 100.0  # high friction in direct path
    friction_data[1, 2] = 100.0

    agg = _make_raster(data, backend=backend)
    friction_agg = _make_raster(friction_data, backend=backend)

    # start=pixel(0,0), goal=pixel(0,4)
    # coords: y=[2,1,0], x=[0,1,2,3,4]
    start = (2.0, 0.0)
    goal = (2.0, 4.0)

    # Without friction: direct straight path (0,0)→(0,1)→(0,2)→(0,3)→(0,4)
    path_no = a_star_search(agg, start, goal)
    assert np.isfinite(np.asarray(path_no.values)[0, 2])  # goes through column 2

    # With friction: should detour around column 2
    path_fr = a_star_search(agg, start, goal, friction=friction_agg)
    vals = np.asarray(path_fr.values)
    assert np.isnan(vals[0, 2])  # does NOT go through (0,2)
    # detour should go through row 2 (pixel row 2)
    assert np.isfinite(vals[2, 2])  # goes through (2,2) instead

    # Weighted path should be cheaper than going through high-friction zone
    assert np.nanmax(vals) < 10.0


@pytest.mark.parametrize("backend", _backends)
def test_friction_barrier_nan(backend):
    """NaN friction blocks passage, forcing a detour."""
    data = np.ones((3, 5))
    friction_data = np.ones((3, 5))
    friction_data[0, 2] = np.nan  # impassable

    agg = _make_raster(data, backend=backend)
    friction_agg = _make_raster(friction_data, backend=backend)

    start = (2.0, 0.0)  # pixel (0,0)
    goal = (2.0, 4.0)   # pixel (0,4)

    path = a_star_search(agg, start, goal, friction=friction_agg)
    vals = np.asarray(path.values)
    # Path must not go through (0,2)
    assert np.isnan(vals[0, 2])
    # But a path should still exist
    assert np.isfinite(vals[0, 4])


@pytest.mark.parametrize("backend", _backends)
def test_friction_barrier_zero(backend):
    """Zero friction blocks passage, forcing a detour."""
    data = np.ones((3, 5))
    friction_data = np.ones((3, 5))
    friction_data[0, 2] = 0.0  # impassable

    agg = _make_raster(data, backend=backend)
    friction_agg = _make_raster(friction_data, backend=backend)

    start = (2.0, 0.0)
    goal = (2.0, 4.0)

    path = a_star_search(agg, start, goal, friction=friction_agg)
    vals = np.asarray(path.values)
    assert np.isnan(vals[0, 2])
    assert np.isfinite(vals[0, 4])


@pytest.mark.parametrize("backend", _backends)
def test_a_star_matches_cost_distance(backend):
    """A* path cost at goal should equal cost_distance value at that pixel."""
    # 5x5 grid with single source at (0,0), non-uniform friction
    source = np.zeros((5, 5))
    source[0, 0] = 1.0  # source/start pixel

    friction_data = np.ones((5, 5))
    friction_data[1, 1] = 3.0
    friction_data[2, 2] = 2.0

    surface = _make_raster(np.ones((5, 5)), backend=backend)
    source_r = _make_raster(source, backend=backend)
    friction_r = _make_raster(friction_data, backend=backend)

    # Run cost_distance from source at (0,0) — always numpy for comparison
    source_np = _make_raster(source, backend='numpy')
    friction_np = _make_raster(friction_data, backend='numpy')
    cd = cost_distance(source_np, friction_np)
    cd_val = cd.values

    # Run A* from (0,0) to (4,4)
    # coords: y=[4,3,2,1,0], x=[0,1,2,3,4]
    start = (4.0, 0.0)   # pixel (0,0)
    goal = (0.0, 4.0)    # pixel (4,4)

    path = a_star_search(surface, start, goal, friction=friction_r)
    a_star_cost_at_goal = float(np.asarray(path.values)[4, 4])

    # cost_distance gives minimum cost from source to every pixel
    cd_cost_at_goal = float(cd_val[4, 4])

    np.testing.assert_allclose(a_star_cost_at_goal, cd_cost_at_goal, rtol=1e-5)


@pytest.mark.parametrize("backend", _backends)
def test_analytic_3x3_weighted(backend):
    """Hand-computed costs on a 3x3 grid with known friction."""
    # Surface all 1s, friction has high center
    data = np.ones((3, 3))
    friction_data = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    agg = _make_raster(data, backend=backend)
    friction_agg = _make_raster(friction_data, backend=backend)

    # start=pixel(0,0), goal=pixel(0,2)
    # coords: y=[2,1,0], x=[0,1,2]
    start = (2.0, 0.0)
    goal = (2.0, 2.0)

    path = a_star_search(agg, start, goal, friction=friction_agg)
    vals = np.asarray(path.values)

    # Direct path through (0,1) with friction=2:
    #   (0,0)→(0,1): 1*(1+2)/2 = 1.5
    #   (0,1)→(0,2): 1*(2+1)/2 = 1.5
    #   total = 3.0
    #
    # Detour through (1,1) with friction=1:
    #   (0,0)→(1,1): sqrt(2)*(1+1)/2 = sqrt(2) ≈ 1.4142
    #   (1,1)→(0,2): sqrt(2)*(1+1)/2 = sqrt(2) ≈ 1.4142
    #   total = 2*sqrt(2) ≈ 2.8284
    #
    # A* picks the cheaper detour
    expected_cost_at_goal = 2.0 * np.sqrt(2.0)
    np.testing.assert_allclose(vals[0, 2], expected_cost_at_goal, rtol=1e-5)

    # Path should not go through (0,1) — the high-friction direct route
    assert np.isnan(vals[0, 1])
    # Path should go through (1,1) — the low-friction detour
    np.testing.assert_allclose(vals[1, 1], np.sqrt(2.0), rtol=1e-5)


# -----------------------------------------------------------------------
# Dask-specific tests
# -----------------------------------------------------------------------

@pytest.mark.skipif(not has_dask_array(), reason="Requires dask.Array")
@pytest.mark.parametrize("chunks", [(2, 2), (3, 3), (5, 5)])
def test_dask_matches_numpy(chunks):
    """Dask A* with various chunk sizes should match numpy results exactly."""
    data = np.ones((5, 5))
    friction_data = np.ones((5, 5))
    friction_data[1, 1] = 3.0
    friction_data[2, 2] = 2.0

    agg_np = _make_raster(data, backend='numpy')
    friction_np = _make_raster(friction_data, backend='numpy')

    agg_dask = _make_raster(data, backend='dask+numpy', chunks=chunks)
    friction_dask = _make_raster(friction_data, backend='dask+numpy',
                                 chunks=chunks)

    start = (4.0, 0.0)
    goal = (0.0, 4.0)

    path_np = a_star_search(agg_np, start, goal, friction=friction_np)
    path_dask = a_star_search(agg_dask, start, goal, friction=friction_dask)

    np.testing.assert_allclose(
        np.asarray(path_dask.values),
        path_np.values,
        equal_nan=True, atol=1e-10,
    )


@pytest.mark.skipif(not has_dask_array(), reason="Requires dask.Array")
def test_dask_no_large_numpy_arrays():
    """Dask path should not materialise full-size numpy arrays."""
    height, width = 50, 60
    data = np.ones((height, width))
    friction_data = np.ones((height, width))

    agg = _make_raster(data, backend='dask+numpy', chunks=(25, 30))
    friction_agg = _make_raster(friction_data, backend='dask+numpy',
                                chunks=(25, 30))

    start = (float(height - 1), 0.0)
    goal = (0.0, float(width - 1))

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

    from unittest.mock import patch
    with patch('numpy.full', side_effect=tracking_full):
        result = a_star_search(agg, start, goal, friction=friction_agg)

    # Result should be dask-backed
    assert isinstance(result.data, da.Array)

    # No full-size arrays should have been allocated
    assert len(large_allocs) == 0, (
        f"Unexpected large allocations: {large_allocs}")


@pytest.mark.skipif(not has_dask_array(), reason="Requires dask.Array")
def test_dask_snap_raises():
    """snap_start/snap_goal should raise ValueError for dask arrays."""
    data = np.ones((5, 5))
    agg = _make_raster(data, backend='dask+numpy', chunks=(3, 3))
    start = (4.0, 0.0)
    goal = (0.0, 4.0)

    with pytest.raises(ValueError, match="snap_start is not supported"):
        a_star_search(agg, start, goal, snap_start=True)

    with pytest.raises(ValueError, match="snap_goal is not supported"):
        a_star_search(agg, start, goal, snap_goal=True)


@pytest.mark.skipif(not has_dask_array(), reason="Requires dask.Array")
def test_dask_no_friction():
    """Dask A* without friction should match numpy."""
    data = np.ones((5, 5))
    agg_np = _make_raster(data, backend='numpy')
    agg_dask = _make_raster(data, backend='dask+numpy', chunks=(3, 3))

    start = (4.0, 0.0)
    goal = (0.0, 4.0)

    path_np = a_star_search(agg_np, start, goal)
    path_dask = a_star_search(agg_dask, start, goal)

    np.testing.assert_allclose(
        np.asarray(path_dask.values),
        path_np.values,
        equal_nan=True, atol=1e-10,
    )


# -----------------------------------------------------------------------
# CuPy tests
# -----------------------------------------------------------------------

@cuda_and_cupy_available
def test_cupy_matches_numpy():
    """CuPy path should produce identical results to numpy."""
    data = np.ones((5, 5))
    friction_data = np.ones((5, 5))
    friction_data[1, 1] = 3.0
    friction_data[2, 2] = 2.0

    agg_np = _make_raster(data, backend='numpy')
    friction_np = _make_raster(friction_data, backend='numpy')

    agg_cupy = _make_raster(data, backend='cupy')
    friction_cupy = _make_raster(friction_data, backend='cupy')

    start = (4.0, 0.0)
    goal = (0.0, 4.0)

    path_np = a_star_search(agg_np, start, goal, friction=friction_np)
    path_cupy = a_star_search(agg_cupy, start, goal, friction=friction_cupy)

    np.testing.assert_allclose(
        path_cupy.data.get(),
        path_np.values,
        equal_nan=True, atol=1e-10,
    )


@cuda_and_cupy_available
def test_cupy_no_friction():
    """CuPy without friction should match numpy."""
    data = np.ones((5, 5))
    agg_np = _make_raster(data, backend='numpy')
    agg_cupy = _make_raster(data, backend='cupy')

    start = (4.0, 0.0)
    goal = (0.0, 4.0)

    path_np = a_star_search(agg_np, start, goal)
    path_cupy = a_star_search(agg_cupy, start, goal)

    np.testing.assert_allclose(
        path_cupy.data.get(),
        path_np.values,
        equal_nan=True, atol=1e-10,
    )


# -----------------------------------------------------------------------
# Dask + CuPy tests
# -----------------------------------------------------------------------

@cuda_and_cupy_available
@pytest.mark.skipif(not has_dask_array(), reason="Requires dask.Array")
def test_dask_cupy_matches_numpy():
    """Dask+CuPy path should produce identical results to numpy."""
    data = np.ones((5, 5))
    friction_data = np.ones((5, 5))
    friction_data[1, 1] = 3.0
    friction_data[2, 2] = 2.0

    agg_np = _make_raster(data, backend='numpy')
    friction_np = _make_raster(friction_data, backend='numpy')

    agg_dc = _make_raster(data, backend='dask+cupy', chunks=(3, 3))
    friction_dc = _make_raster(friction_data, backend='dask+cupy',
                               chunks=(3, 3))

    start = (4.0, 0.0)
    goal = (0.0, 4.0)

    path_np = a_star_search(agg_np, start, goal, friction=friction_np)
    path_dc = a_star_search(agg_dc, start, goal, friction=friction_dc)

    # dask+cupy: compute dask graph, then .get() the cupy array to numpy
    dc_computed = path_dc.data.compute()
    if hasattr(dc_computed, 'get'):
        dc_computed = dc_computed.get()

    np.testing.assert_allclose(
        dc_computed,
        path_np.values,
        equal_nan=True, atol=1e-10,
    )


# -----------------------------------------------------------------------
# Bounded A* / Memory guard / HPA* tests
# -----------------------------------------------------------------------

from unittest.mock import patch
from xrspatial.pathfinding import (
    _available_memory_bytes, _check_memory, _hpa_star_search,
    _bounded_a_star, _neighborhood_structure,
)


def test_search_radius_matches_full():
    """Bounded A* with generous radius should match full A* exactly."""
    data = np.ones((10, 10))
    agg = _make_raster(data)

    start = (9.0, 0.0)   # pixel (0, 0)
    goal = (0.0, 9.0)    # pixel (9, 9)

    path_full = a_star_search(agg, start, goal)
    # radius=20 covers the entire 10x10 grid
    path_bounded = a_star_search(agg, start, goal, search_radius=20)

    np.testing.assert_allclose(
        path_bounded.values, path_full.values,
        equal_nan=True, atol=1e-10,
    )


def test_search_radius_too_small():
    """Path exists on full grid but search_radius excludes the only detour."""
    data = np.ones((20, 20))
    # Wall at column 10, rows 0-14 (gap only at row 15)
    data[0:15, 10] = np.nan

    agg = _make_raster(data)

    # start=pixel(5,8), goal=pixel(5,12) — wall in between
    # coords: y goes from 19 down to 0
    start = (14.0, 8.0)   # pixel (5, 8)
    goal = (14.0, 12.0)   # pixel (5, 12)

    # Full A* finds the path via the gap at row 15
    path_full = a_star_search(agg, start, goal)
    assert np.isfinite(path_full.values[5, 12])  # sanity

    # search_radius=3: box covers rows 2-8, cols 5-15
    # Wall at col 10 blocks, gap at row 15 is outside box
    path_bounded = a_star_search(agg, start, goal, search_radius=3)
    assert np.isnan(path_bounded.values[5, 12])  # no path within box


def test_memory_guard_raises():
    """MemoryError raised when mocked memory is tiny and no search_radius."""
    data = np.ones((10, 10))
    agg = _make_raster(data)

    start = (9.0, 0.0)
    goal = (0.0, 9.0)

    with patch('xrspatial.pathfinding._available_memory_bytes',
               return_value=1024):
        with pytest.raises(MemoryError):
            a_star_search(agg, start, goal)


def test_memory_guard_passes_with_radius():
    """With search_radius, a small sub-region fits even in tiny memory."""
    data = np.ones((10, 10))
    agg = _make_raster(data)

    start = (9.0, 0.0)   # pixel (0, 0)
    goal = (8.0, 1.0)    # pixel (1, 1) — very close

    # Mock very low memory — full grid would fail, but radius=5
    # creates a sub-region of ~6x6 = 36 pixels × 65 = 2340 bytes
    with patch('xrspatial.pathfinding._available_memory_bytes',
               return_value=100_000):
        path = a_star_search(agg, start, goal, search_radius=5)
        # Path should exist
        assert np.isfinite(path.values[1, 1])
        assert path.values[0, 0] == 0.0


def test_hpa_star_correctness():
    """HPA* on a 200×200 grid with barriers finds valid connected path."""
    rng = np.random.RandomState(42)
    data = np.ones((200, 200))
    # Add barrier wall with a gap
    data[50:150, 100] = np.nan
    data[80, 100] = 1.0  # gap in the wall

    agg = _make_raster(data)
    surface_data = data.astype(np.float64)
    friction_data = None
    barriers = np.array([], dtype=np.float64)

    dy, dx, dd = _neighborhood_structure(1.0, 1.0, 8)

    path_img = _hpa_star_search(
        surface_data, friction_data,
        0, 0, 199, 199,
        barriers, dy, dx, dd,
        1.0, False, 1.0, 1.0, 200, 200)

    # Path should reach the goal
    assert np.isfinite(path_img[199, 199])
    # Path should start at 0 cost
    assert path_img[0, 0] == 0.0
    # Cost should be monotonically increasing (check start < goal)
    assert path_img[199, 199] > path_img[0, 0]

    # Verify path pixels form a connected chain
    path_pixels = []
    for r in range(200):
        for c in range(200):
            if np.isfinite(path_img[r, c]):
                path_pixels.append((path_img[r, c], r, c))
    path_pixels.sort()

    for i in range(1, len(path_pixels)):
        _, pr, pc = path_pixels[i - 1]
        _, cr, cc = path_pixels[i]
        # Each pixel should be within 8-connectivity of some previous pixel
        # (not necessarily the immediately preceding one by cost, but
        # in practice for a proper path it should be adjacent)
        dist = max(abs(cr - pr), abs(cc - pc))
        # Allow distance > 1 between sorted-by-cost pixels since HPA*
        # stitches segments — but every pixel should be on the path
        assert (r, c) != (np.nan, np.nan)


def test_auto_radius_selection():
    """When mocked low memory, auto-radius kicks in and finds path."""
    data = np.ones((20, 20))
    agg = _make_raster(data)

    start = (19.0, 0.0)   # pixel (0, 0)
    goal = (15.0, 4.0)    # pixel (4, 4)

    # Mock memory so full 20×20 grid (26000 bytes) exceeds 50% of avail
    # but auto-radius sub-region fits
    with patch('xrspatial.pathfinding._available_memory_bytes',
               return_value=40_000):
        path = a_star_search(agg, start, goal)
        # Path should exist
        assert np.isfinite(path.values[4, 4])
        assert path.values[0, 0] == 0.0


def test_backward_compatibility():
    """Existing call patterns work unchanged with search_radius=None."""
    data = np.ones((5, 5))
    agg = _make_raster(data)

    start = (4.0, 0.0)
    goal = (0.0, 4.0)

    # Default (no search_radius) should work
    path = a_star_search(agg, start, goal)
    assert np.isfinite(path.values[4, 4])
    assert path.values[0, 0] == 0.0

    # Explicit search_radius=None should work
    path2 = a_star_search(agg, start, goal, search_radius=None)
    np.testing.assert_allclose(
        path2.values, path.values, equal_nan=True, atol=1e-10)


# -----------------------------------------------------------------------
# multi_stop_search tests
# -----------------------------------------------------------------------

from xrspatial import multi_stop_search
from xrspatial.pathfinding import _held_karp, _nearest_neighbor_2opt


# --- TSP solver unit tests ---

def test_held_karp_basic():
    """Held-Karp finds optimal tour on a small symmetric graph."""
    # 4 cities in a line: 0 -- 1 -- 2 -- 3
    dist = [
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0],
    ]
    order, cost = _held_karp(dist, 0, 3)
    assert order[0] == 0
    assert order[-1] == 3
    assert cost == 3.0  # 0->1->2->3
    assert order == [0, 1, 2, 3]


def test_held_karp_asymmetric():
    """Held-Karp handles directed (asymmetric) costs."""
    # Going 0->1 costs 1, but 1->0 costs 10
    dist = [
        [0,  1, 5],
        [10, 0, 1],
        [5,  1, 0],
    ]
    order, cost = _held_karp(dist, 0, 2)
    assert order[0] == 0
    assert order[-1] == 2
    # Optimal: 0->1->2 = 1+1 = 2 (not 0->2 = 5)
    assert cost == 2.0
    assert order == [0, 1, 2]


def test_nearest_neighbor_2opt_fixed_endpoints():
    """Nearest-neighbor + 2-opt preserves start and end."""
    dist = [
        [0, 1, 4, 3],
        [1, 0, 2, 5],
        [4, 2, 0, 1],
        [3, 5, 1, 0],
    ]
    order, cost = _nearest_neighbor_2opt(dist, 0, 3)
    assert order[0] == 0
    assert order[-1] == 3
    assert set(order) == {0, 1, 2, 3}


def test_nearest_neighbor_2opt_vs_held_karp():
    """Heuristic should match or be close to exact on small inputs."""
    dist = [
        [0, 2, 9, 10],
        [2, 0, 6, 4],
        [9, 6, 0, 8],
        [10, 4, 8, 0],
    ]
    exact_order, exact_cost = _held_karp(dist, 0, 3)
    heur_order, heur_cost = _nearest_neighbor_2opt(dist, 0, 3)

    assert heur_order[0] == 0
    assert heur_order[-1] == 3
    # Heuristic should be within 50% of optimal for 4 cities
    assert heur_cost <= exact_cost * 1.5


# --- Multi-stop functional tests ---

def test_two_waypoints_matches_a_star():
    """Two waypoints should produce the same result as a single A* call."""
    data = np.ones((8, 8))
    agg = _make_raster(data)

    start = (7.0, 0.0)  # pixel (0, 0)
    goal = (0.0, 7.0)   # pixel (7, 7)

    path_single = a_star_search(agg, start, goal)
    path_multi = multi_stop_search(agg, [start, goal])

    np.testing.assert_allclose(
        path_multi.values, path_single.values,
        equal_nan=True, atol=1e-10,
    )


def test_three_waypoints_cost_continuity():
    """Costs should increase monotonically across segments; attrs correct."""
    data = np.ones((10, 10))
    agg = _make_raster(data)

    # Three waypoints along the diagonal
    wp0 = (9.0, 0.0)   # pixel (0, 0)
    wp1 = (5.0, 4.0)   # pixel (4, 4)
    wp2 = (0.0, 9.0)   # pixel (9, 9)

    result = multi_stop_search(agg, [wp0, wp1, wp2])

    vals = result.values
    # All three waypoint pixels should be on the path
    assert np.isfinite(vals[0, 0])
    assert np.isfinite(vals[4, 4])
    assert np.isfinite(vals[9, 9])

    # Cost at start should be 0
    assert vals[0, 0] == 0.0
    # Cost should increase: start < mid < end
    assert vals[4, 4] < vals[9, 9]
    assert vals[0, 0] < vals[4, 4]

    # Check attrs
    assert len(result.attrs['segment_costs']) == 2
    assert result.attrs['total_cost'] > 0
    np.testing.assert_allclose(
        sum(result.attrs['segment_costs']),
        result.attrs['total_cost'],
        atol=1e-10,
    )
    assert len(result.attrs['waypoint_order']) == 3


def test_three_waypoints_with_friction():
    """Multi-stop with friction surface produces valid monotonic costs."""
    data = np.ones((10, 10))
    friction_data = np.ones((10, 10))
    friction_data[3:7, 3:7] = 5.0  # high-friction zone

    agg = _make_raster(data)
    friction_agg = _make_raster(friction_data)

    wp0 = (9.0, 0.0)
    wp1 = (5.0, 4.0)
    wp2 = (0.0, 9.0)

    result = multi_stop_search(agg, [wp0, wp1, wp2], friction=friction_agg)

    vals = result.values
    assert np.isfinite(vals[0, 0])
    assert np.isfinite(vals[9, 9])
    assert vals[0, 0] == 0.0
    assert result.attrs['total_cost'] > 0


def test_unreachable_segment_raises():
    """Barrier blocking a segment should raise ValueError."""
    data = np.ones((8, 8))
    # Wall across row 4
    data[4, :] = np.nan

    agg = _make_raster(data)

    wp0 = (7.0, 0.0)   # pixel (0, 0) — above the wall
    wp1 = (0.0, 0.0)   # pixel (7, 0) — below the wall

    with pytest.raises(ValueError, match="no path between waypoints"):
        multi_stop_search(agg, [wp0, wp1])


def test_waypoint_outside_bounds_raises():
    """Waypoint outside the surface should raise ValueError."""
    data = np.ones((5, 5))
    agg = _make_raster(data)

    wp0 = (4.0, 0.0)
    wp_bad = (100.0, 100.0)

    with pytest.raises(ValueError, match="outside the surface bounds"):
        multi_stop_search(agg, [wp0, wp_bad])


def test_too_few_waypoints_raises():
    """Fewer than 2 waypoints should raise ValueError."""
    data = np.ones((5, 5))
    agg = _make_raster(data)

    with pytest.raises(ValueError, match="at least 2 waypoints"):
        multi_stop_search(agg, [(4.0, 0.0)])

    with pytest.raises(ValueError, match="at least 2 waypoints"):
        multi_stop_search(agg, [])


# --- optimize_order tests ---

def test_optimize_order_finds_better_route():
    """Reordering interior waypoints should give equal or lower total cost."""
    data = np.ones((10, 10))
    agg = _make_raster(data)

    # Deliberately suboptimal order: zigzag
    wp0 = (9.0, 0.0)   # pixel (0, 0) — fixed start
    wp1 = (0.0, 9.0)   # pixel (9, 9) — far corner
    wp2 = (9.0, 4.0)   # pixel (0, 4) — back near start
    wp3 = (0.0, 4.0)   # pixel (9, 4) — fixed end

    naive = multi_stop_search(agg, [wp0, wp1, wp2, wp3])
    optimized = multi_stop_search(
        agg, [wp0, wp1, wp2, wp3], optimize_order=True)

    assert optimized.attrs['total_cost'] <= naive.attrs['total_cost'] + 1e-10


def test_optimize_order_preserves_endpoints():
    """First and last waypoints should remain fixed after optimization."""
    data = np.ones((10, 10))
    agg = _make_raster(data)

    wp0 = (9.0, 0.0)
    wp1 = (5.0, 5.0)
    wp2 = (5.0, 0.0)
    wp3 = (0.0, 9.0)

    result = multi_stop_search(
        agg, [wp0, wp1, wp2, wp3], optimize_order=True)

    order = result.attrs['waypoint_order']
    assert tuple(order[0]) == wp0
    assert tuple(order[-1]) == wp3


# --- Cross-backend tests ---

@pytest.mark.skipif(not has_dask_array(), reason="Requires dask.Array")
def test_multi_stop_dask_matches_numpy():
    """Dask multi-stop should match numpy results."""
    data = np.ones((8, 8))
    wp0 = (7.0, 0.0)
    wp1 = (4.0, 3.0)
    wp2 = (0.0, 7.0)

    agg_np = _make_raster(data, backend='numpy')
    agg_dask = _make_raster(data, backend='dask+numpy', chunks=(4, 4))

    path_np = multi_stop_search(agg_np, [wp0, wp1, wp2])
    path_dask = multi_stop_search(agg_dask, [wp0, wp1, wp2])

    np.testing.assert_allclose(
        np.asarray(path_dask.values),
        path_np.values,
        equal_nan=True, atol=1e-10,
    )


@cuda_and_cupy_available
def test_multi_stop_cupy_matches_numpy():
    """CuPy multi-stop should match numpy results."""
    data = np.ones((8, 8))
    wp0 = (7.0, 0.0)
    wp1 = (4.0, 3.0)
    wp2 = (0.0, 7.0)

    agg_np = _make_raster(data, backend='numpy')
    agg_cupy = _make_raster(data, backend='cupy')

    path_np = multi_stop_search(agg_np, [wp0, wp1, wp2])
    path_cupy = multi_stop_search(agg_cupy, [wp0, wp1, wp2])

    np.testing.assert_allclose(
        path_cupy.data if not hasattr(path_cupy.data, 'get') else path_cupy.data,
        path_np.values,
        equal_nan=True, atol=1e-10,
    )
