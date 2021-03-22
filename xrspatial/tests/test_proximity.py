import pytest

from xrspatial import proximity, allocation, direction
from xrspatial import great_circle_distance, manhattan_distance
from xrspatial import euclidean_distance
from xrspatial.proximity import _calc_direction

import numpy as np
import xarray as xa


def test_great_circle_distance():
    # invalid x_coord
    y1, x1 = 0, 0
    y2, x2 = 0, -181
    y3, x3 = 0, 181
    with pytest.raises(Exception) as e_info:
        great_circle_distance(x1, x2, y1, y2)
        assert e_info
    with pytest.raises(Exception) as e_info:
        great_circle_distance(x1, x3, y1, y3)
        assert e_info

    # invalid y_coord
    y1, x1 = 0, 0
    y2, x2 = -91, 0
    y3, x3 = 91, 0
    with pytest.raises(Exception) as e_info:
        great_circle_distance(x1, x2, y1, y2)
        assert e_info
    with pytest.raises(Exception) as e_info:
        great_circle_distance(x1, x3, y1, y3)
        assert e_info


def create_test_raster():
    height, width = 5, 10
    data = np.asarray([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                       [0., 0., np.inf, 3., 2., 5., 6., 0., 0., 0.],
                       [0., 0., 0., 4., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., np.nan, 0., 0., 0.]])
    _lon = np.linspace(-20, 20, width)
    _lat = np.linspace(-20, 20, height)

    raster = xa.DataArray(data, dims=['lat', 'lon'])
    raster['lon'] = _lon
    raster['lat'] = _lat
    return raster


def test_proximity():
    raster = create_test_raster()

    # DEFAULT SETTINGS
    default_prox = proximity(raster, x='lon', y='lat')
    # output must be an xarray DataArray
    assert isinstance(default_prox, xa.DataArray)
    assert type(default_prox.values[0][0]) == np.float64
    assert default_prox.shape == raster.shape
    # in this test case, where no polygon is completely inside another polygon,
    # number of non-zeros (target pixels) in original image
    # must be equal to the number of zeros (target pixels) in proximity matrix
    assert len(np.where((raster.data != 0) & np.isfinite(raster.data))[0]) == \
        len(np.where(default_prox.data == 0)[0])

    # TARGET VALUES SETTING
    target_values = [2, 3]
    target_prox = proximity(raster, x='lon', y='lat',
                            target_values=target_values)
    # output must be an xarray DataArray
    assert isinstance(target_prox, xa.DataArray)
    assert type(target_prox.values[0][0]) == np.float64
    assert target_prox.shape == raster.shape
    assert (len(np.where(raster.data == 2)[0]) +
            len(np.where(raster.data == 3)[0])) == \
        len(np.where(target_prox.data == 0)[0])

    # distance_metric SETTING: MANHATTAN
    manhattan_prox = proximity(raster, x='lon', y='lat',
                               distance_metric='MANHATTAN')
    # output must be an xarray DataArray
    assert isinstance(manhattan_prox, xa.DataArray)
    assert type(manhattan_prox.values[0][0]) == np.float64
    assert manhattan_prox.shape == raster.shape
    # all output values must be in range [0, max_possible_dist]
    max_possible_dist = manhattan_distance(raster.coords['lon'].values[0],
                                           raster.coords['lon'].values[-1],
                                           raster.coords['lat'].values[0],
                                           raster.coords['lat'].values[-1])
    assert np.nanmax(manhattan_prox.values) <= max_possible_dist
    assert np.nanmin(manhattan_prox.values) == 0

    # distance_metric SETTING: GREAT_CIRCLE
    great_circle_prox = proximity(raster, x='lon', y='lat',
                                  distance_metric='GREAT_CIRCLE')
    # output must be an xarray DataArray
    assert isinstance(great_circle_prox, xa.DataArray)
    assert type(great_circle_prox.values[0][0]) == np.float64
    assert great_circle_prox.shape == raster.shape
    # all output values must be in range [0, max_possible_dist]
    max_possible_dist = great_circle_distance(raster.coords['lon'].values[0],
                                              raster.coords['lon'].values[-1],
                                              raster.coords['lat'].values[0],
                                              raster.coords['lat'].values[-1])
    assert np.nanmax(great_circle_prox.values) <= max_possible_dist
    assert np.nanmin(great_circle_prox.values) == 0


def test_allocation():
    # create test raster, all non-zero cells are unique,
    # this is to test against corresponding proximity
    raster = create_test_raster()

    allocation_agg = allocation(raster, x='lon', y='lat')
    # output must be an xarray DataArray
    assert isinstance(allocation_agg, xa.DataArray)
    assert type(allocation_agg.values[0][0]) == raster.dtype
    assert allocation_agg.shape == raster.shape
    # targets not specified,
    # Thus, targets are set to non-zero values of input @raster
    targets = np.unique(raster.data[np.where((raster.data != 0) &
                                             np.isfinite(raster.data))])
    # non-zero cells (a.k.a targets) remain the same
    for t in targets:
        ry, rx = np.where(raster.data == t)
        for y, x in zip(ry, rx):
            assert allocation_agg.values[y, x] == t
    # values of allocation output
    assert (np.unique(allocation_agg.data) == targets).all()

    # check against corresponding proximity
    proximity_agg = proximity(raster, x='lon', y='lat')
    xcoords = allocation_agg['lon'].data
    ycoords = allocation_agg['lat'].data

    for y in range(raster.shape[0]):
        for x in range(raster.shape[1]):
            a = allocation_agg.data[y, x]
            py, px = np.where(raster.data == a)
            # non-zero cells in raster are unique, thus len(px)=len(py)=1
            d = euclidean_distance(xcoords[x], xcoords[px[0]],
                                   ycoords[y], ycoords[py[0]])
            assert proximity_agg.data[y, x] == d


def test_calc_direction():
    n = 3
    x1, y1 = 1, 1
    output = np.zeros(shape=(n, n))
    for y2 in range(n):
        for x2 in range(n):
            output[y2, x2] = _calc_direction(x2, x1, y2, y1)

    expected_output = np.asarray([[135, 180, 225],
                                  [90,  0,   270],
                                  [45,  360, 315]])
    # set a tolerance of 1e-5
    tolerance = 1e-5
    assert (abs(output-expected_output) <= tolerance).all()


def test_direction():
    raster = create_test_raster()
    direction_agg = direction(raster, x='lon', y='lat')

    # output must be an xarray DataArray
    assert isinstance(direction_agg, xa.DataArray)
    assert type(direction_agg.values[0][0]) == np.float64
    assert direction_agg.shape == raster.shape
    assert direction_agg.dims == raster.dims
    assert direction_agg.attrs == raster.attrs
    for c in direction_agg.coords:
        assert (direction_agg[c] == raster.coords[c]).all()

    # in this test case, where no polygon is completely inside another polygon,
    # number of non-zeros (target pixels) in original image
    # must be equal to the number of zeros (target pixels) in proximity matrix
    assert len(np.where((raster.data != 0) & np.isfinite(raster.data))[0]) == \
        len(np.where(direction_agg.data == 0)[0])

    # values are within [0, 360]
    assert np.min(direction_agg.data) >= 0
    assert np.max(direction_agg.data) <= 360

    # test against allocation
    allocation_agg = allocation(raster, x='lon', y='lat')
    xcoords = allocation_agg['lon'].data
    ycoords = allocation_agg['lat'].data

    for y in range(raster.shape[0]):
        for x in range(raster.shape[1]):
            a = allocation_agg.data[y, x]
            py, px = np.where(raster.data == a)
            # non-zero cells in raster are unique, thus len(px)=len(py)=1
            d = _calc_direction(xcoords[x], xcoords[px[0]],
                                ycoords[y], ycoords[py[0]])
            assert direction_agg.data[y, x] == d
