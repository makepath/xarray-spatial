import numpy as np
import pytest

from xrspatial import a_star_search
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
    expected_result = np.array([[np.nan, np.nan, 0., np.nan],
                                [np.nan, 1.41421356, np.nan, np.nan],
                                [np.nan, 2.41421356, np.nan, np.nan],
                                [np.nan, 3.41421356, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan]])
    return expected_result


@pytest.fixture
def result_4_connectivity():
    expected_result = np.array([[np.nan, 1, 0., np.nan],
                                [np.nan, 2, np.nan, np.nan],
                                [np.nan, 3, np.nan, np.nan],
                                [np.nan, 4, np.nan, np.nan],
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
