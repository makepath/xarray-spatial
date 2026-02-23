import numpy as np
import pytest

from xrspatial import a_star_search
from xrspatial.cost_distance import cost_distance
from xrspatial.tests.general_checks import create_test_raster, general_output_checks


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
# Weighted A* tests
# -----------------------------------------------------------------------

def _make_raster(data, dims=None, res=None):
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
    return raster


def test_uniform_friction_matches_no_friction():
    """Uniform friction=1 should give the same path and costs as no friction."""
    data = np.ones((5, 5))
    agg = _make_raster(data)
    friction_agg = _make_raster(np.ones((5, 5)))

    # coordinate space: y goes 4..0, x goes 0..4
    start = (4.0, 0.0)   # pixel (0,0)
    goal = (0.0, 4.0)    # pixel (4,4)

    path_no_friction = a_star_search(agg, start, goal)
    path_with_friction = a_star_search(agg, start, goal, friction=friction_agg)

    np.testing.assert_allclose(
        path_with_friction.values, path_no_friction.values,
        equal_nan=True, atol=1e-10,
    )


def test_high_friction_detour():
    """A* should route around a high-friction column."""
    # 3x5 grid, cellsize=1
    data = np.ones((3, 5))
    friction_data = np.ones((3, 5))
    friction_data[0, 2] = 100.0  # high friction in direct path
    friction_data[1, 2] = 100.0

    agg = _make_raster(data)
    friction_agg = _make_raster(friction_data)

    # start=pixel(0,0), goal=pixel(0,4)
    # coords: y=[2,1,0], x=[0,1,2,3,4]
    start = (2.0, 0.0)
    goal = (2.0, 4.0)

    # Without friction: direct straight path (0,0)→(0,1)→(0,2)→(0,3)→(0,4)
    path_no = a_star_search(agg, start, goal)
    assert np.isfinite(path_no.values[0, 2])  # goes through column 2

    # With friction: should detour around column 2
    path_fr = a_star_search(agg, start, goal, friction=friction_agg)
    assert np.isnan(path_fr.values[0, 2])  # does NOT go through (0,2)
    # detour should go through row 2 (pixel row 2)
    assert np.isfinite(path_fr.values[2, 2])  # goes through (2,2) instead

    # Weighted path should be cheaper than going through high-friction zone
    # Direct cost: 0 + 1 + (1+100)/2 + (100+1)/2 + 1 = 53
    # Detour cost: 2*sqrt(2) + 2 ≈ 4.83
    assert np.nanmax(path_fr) < 10.0


def test_friction_barrier_nan():
    """NaN friction blocks passage, forcing a detour."""
    data = np.ones((3, 5))
    friction_data = np.ones((3, 5))
    friction_data[0, 2] = np.nan  # impassable

    agg = _make_raster(data)
    friction_agg = _make_raster(friction_data)

    start = (2.0, 0.0)  # pixel (0,0)
    goal = (2.0, 4.0)   # pixel (0,4)

    path = a_star_search(agg, start, goal, friction=friction_agg)
    # Path must not go through (0,2)
    assert np.isnan(path.values[0, 2])
    # But a path should still exist
    assert np.isfinite(path.values[0, 4])


def test_friction_barrier_zero():
    """Zero friction blocks passage, forcing a detour."""
    data = np.ones((3, 5))
    friction_data = np.ones((3, 5))
    friction_data[0, 2] = 0.0  # impassable

    agg = _make_raster(data)
    friction_agg = _make_raster(friction_data)

    start = (2.0, 0.0)
    goal = (2.0, 4.0)

    path = a_star_search(agg, start, goal, friction=friction_agg)
    assert np.isnan(path.values[0, 2])
    assert np.isfinite(path.values[0, 4])


def test_a_star_matches_cost_distance():
    """A* path cost at goal should equal cost_distance value at that pixel."""
    # 5x5 grid with single source at (0,0), non-uniform friction
    source = np.zeros((5, 5))
    source[0, 0] = 1.0  # source/start pixel

    friction_data = np.ones((5, 5))
    friction_data[1, 1] = 3.0
    friction_data[2, 2] = 2.0

    surface = _make_raster(np.ones((5, 5)))
    source_r = _make_raster(source)
    friction_r = _make_raster(friction_data)

    # Run cost_distance from source at (0,0)
    cd = cost_distance(source_r, friction_r)
    cd_val = cd.values

    # Run A* from (0,0) to (4,4)
    # coords: y=[4,3,2,1,0], x=[0,1,2,3,4]
    start = (4.0, 0.0)   # pixel (0,0)
    goal = (0.0, 4.0)    # pixel (4,4)

    path = a_star_search(surface, start, goal, friction=friction_r)
    a_star_cost_at_goal = path.values[4, 4]

    # cost_distance gives minimum cost from source to every pixel
    cd_cost_at_goal = float(cd_val[4, 4])

    np.testing.assert_allclose(a_star_cost_at_goal, cd_cost_at_goal, rtol=1e-5)


def test_analytic_3x3_weighted():
    """Hand-computed costs on a 3x3 grid with known friction."""
    # Surface all 1s, friction has high center
    data = np.ones((3, 3))
    friction_data = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    agg = _make_raster(data)
    friction_agg = _make_raster(friction_data)

    # start=pixel(0,0), goal=pixel(0,2)
    # coords: y=[2,1,0], x=[0,1,2]
    start = (2.0, 0.0)
    goal = (2.0, 2.0)

    path = a_star_search(agg, start, goal, friction=friction_agg)

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
    np.testing.assert_allclose(path.values[0, 2], expected_cost_at_goal, rtol=1e-5)

    # Path should not go through (0,1) — the high-friction direct route
    assert np.isnan(path.values[0, 1])
    # Path should go through (1,1) — the low-friction detour
    np.testing.assert_allclose(path.values[1, 1], np.sqrt(2.0), rtol=1e-5)
