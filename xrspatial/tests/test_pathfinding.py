import numpy as np
import xarray as xr

from xrspatial import a_star_search


def test_a_star_search():
    agg = xr.DataArray(np.array([[0, 1, 0, 0],
                                 [1, 1, 0, 0],
                                 [0, 1, 2, 2],
                                 [1, 0, 2, 0],
                                 [0, 2, 2, 2]]),
                       dims=['lat', 'lon'])

    height, width = agg.shape
    _lon = np.linspace(0, width - 1, width)
    _lat = np.linspace(0, height - 1, height)
    agg['lon'] = _lon
    agg['lat'] = _lat
    barriers = []
    # no barriers, there always path from a start location to a goal location
    for x0 in _lon:
        for y0 in _lat:
            start = (x0, y0)
            for x1 in _lon:
                for y1 in _lat:
                    goal = (x1, y1)
                    path_agg = a_star_search(agg, start, goal, barriers,
                                             'lon', 'lat')
                    assert isinstance(path_agg, xr.DataArray)
                    assert type(path_agg.values[0][0]) == np.float64
                    assert path_agg.shape == agg.shape
                    assert path_agg.dims == agg.dims
                    assert path_agg.attrs == agg.attrs
                    for c in path_agg.coords:
                        assert (path_agg[c] == agg.coords[c]).all()
                    if start == goal:
                        assert np.nanmax(path_agg) == 0 and \
                               np.nanmin(path_agg) == 0
                    else:
                        assert np.nanmax(path_agg) > 0 and \
                               np.nanmin(path_agg) == 0

    barriers = [1]
    # set pixels with value 1 as barriers,
    # cannot go from (0, 0) to anywhere since it is surrounded by 1s
    start = (0, 0)
    for x1 in _lon:
        for y1 in _lat:
            goal = (x1, y1)
            if goal != start:
                path_agg = a_star_search(agg, start, goal, barriers,
                                         'lon', 'lat')
                assert isinstance(path_agg, xr.DataArray)
                assert type(path_agg.values[0][0]) == np.float64
                assert path_agg.shape == agg.shape
                assert path_agg.dims == agg.dims
                assert path_agg.attrs == agg.attrs
                for c in path_agg.coords:
                    assert (path_agg[c] == agg.coords[c]).all()
                # no path, all cells in path_agg are nans
                assert np.isnan(path_agg).all()

    # test with nans
    agg = xr.DataArray(np.array([[0, 1, 0, 0],
                                 [1, 1, np.nan, 0],
                                 [0, 1, 2, 2],
                                 [1, 0, 2, 0],
                                 [0, np.nan, 2, 2]]),
                       dims=['lat', 'lon'])

    height, width = agg.shape
    _lon = np.linspace(0, width - 1, width)
    _lat = np.linspace(0, height - 1, height)
    agg['lon'] = _lon
    agg['lat'] = _lat
    # start and end at a nan pixel, coordinate in (lon, lat) format
    # in this example, each pixel is a unit of lon and lat,
    #                  start = (2, 1) corresponds to pixel at (1, 2),
    #                  goal = (1, 4) corresponds to pixel at (4, 1)
    start = (2, 1)
    goal = (1, 4)
    # no barriers
    barriers = []
    # no snap
    no_snap_path_agg = a_star_search(agg, start, goal, barriers, 'lon', 'lat')
    # no path, all cells in path_agg are nans
    assert np.isnan(no_snap_path_agg).all()

    # set snap_start = True, snap_goal = False
    snap_start_path_agg = a_star_search(agg, start, goal, barriers,
                                        'lon', 'lat', snap_start=True)
    # no path, all cells in path_agg are nans
    assert np.isnan(snap_start_path_agg).all()

    # set snap_start = False, snap_goal = True
    snap_goal_path_agg = a_star_search(agg, start, goal, barriers,
                                       'lon', 'lat', snap_goal=True)
    # no path, all cells in path_agg are nans
    assert np.isnan(snap_goal_path_agg).all()

    # set snap_start = True, snap_goal = True
    # 8-connectivity as default
    path_agg_8 = a_star_search(agg, start, goal, barriers, 'lon', 'lat',
                               snap_start=True, snap_goal=True)
    # path exists
    expected_result_8 = np.array([[np.nan, np.nan, 0., np.nan],
                                 [np.nan, 1.41421356, np.nan, np.nan],
                                 [np.nan, 2.41421356, np.nan, np.nan],
                                 [np.nan, 3.41421356, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan, np.nan]])
    assert ((np.isnan(path_agg_8) & np.isnan(expected_result_8)) | (
                    abs(path_agg_8 - expected_result_8) <= 1e-5)).all()

    # 4-connectivity
    path_agg_4 = a_star_search(agg, start, goal, barriers, 'lon', 'lat',
                               snap_start=True, snap_goal=True, connectivity=4)
    # path exists
    expected_result_4 = np.array([[np.nan, 1, 0., np.nan],
                                  [np.nan, 2, np.nan, np.nan],
                                  [np.nan, 3, np.nan, np.nan],
                                  [np.nan, 4, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan]])
    assert ((np.isnan(path_agg_4) & np.isnan(expected_result_4)) | (
                    abs(path_agg_4 - expected_result_4) <= 1e-5)).all()
