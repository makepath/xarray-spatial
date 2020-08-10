import numpy as np
import xarray as xr

from xrspatial import a_star_search


def test_a_star_search():
    agg = xr.DataArray(np.array([[1, 1, 0, 0],
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
                    path_agg = a_star_search(agg, start, goal, barriers, 'lon',
                                             'lat')
                    assert isinstance(path_agg, xr.DataArray)
                    assert type(path_agg.values[0][0]) == np.float64
                    assert path_agg.shape == agg.shape
                    assert path_agg.dims == agg.dims
                    assert path_agg.attrs == agg.attrs
                    for c in path_agg.coords:
                        assert (path_agg[c] == agg.coords[c]).all()
                    assert np.nanmax(path_agg) >= 0 and np.nanmin(
                        path_agg) == 0

    barriers = [1]
    # set pixels with value 1 as barriers,
    # cannot go from (0, 0) to anywhere since it is surrounded by 1s
    start = (0, 0)
    for x1 in _lon:
        for y1 in _lat:
            goal = (x1, y1)
            if goal != start:
                path_agg = a_star_search(agg, start, goal, barriers, 'lon',
                                         'lat')
                assert isinstance(path_agg, xr.DataArray)
                assert type(path_agg.values[0][0]) == np.float64
                assert path_agg.shape == agg.shape
                assert path_agg.dims == agg.dims
                assert path_agg.attrs == agg.attrs
                for c in path_agg.coords:
                    assert (path_agg[c] == agg.coords[c]).all()
                # no path, all cells in path_agg are nans
                assert np.isnan(path_agg).all()
