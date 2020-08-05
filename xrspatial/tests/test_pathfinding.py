import numpy as np
import xarray as xr

from xrspatial import a_star_search


def test_a_star_search():
    agg = xr.DataArray(np.array([[0, 1, 0, 0],
                                 [1, 0, 0, 0],
                                 [0, 1, 2, 2],
                                 [1, 0, 2, 0],
                                 [0, 2, 2, 2]]),
                       dims=['lat', 'lon'])

    height, width = agg.shape
    _lon = np.linspace(0, width-1, width)
    _lat = np.linspace(0, height-1, height)
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
                    path, cost = a_star_search(agg, start, goal, barriers,
                                               'lon', 'lat')
                    assert len(path) > 0 and cost >= 0
                    if start == goal:
                        # if start and goal are in same cell,
                        # path is the cell itself
                        # with 0 cost
                        assert len(path) == 1 and cost == 0
