import pytest

from xrspatial import proximity, allocation
from xrspatial import great_circle_distance, manhattan_distance
from xrspatial import euclidean_distance
import datashader as ds

import numpy as np
import pandas as pd
import xarray as xa

from math import sqrt


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
    df = pd.DataFrame({
        'lat': [-10, -10, -4, -4, 1, 3, 7, 7, 7],
        'lon': [-5, -10, -5, -5, 0, 5, 10, 10, 10]
    })
    cvs = ds.Canvas(plot_width=width, plot_height=height,
                    x_range=(-20, 20), y_range=(-20, 20))

    raster = cvs.points(df, x='lon', y='lat')
    return raster


def test_proximity():
    raster = create_test_raster()

    # DEFAULT SETTINGS
    default_prox = proximity(raster, x='lon', y='lat')
    # output must be an xarray DataArray
    assert isinstance(default_prox, xa.DataArray)
    assert isinstance(default_prox.values, np.ndarray)
    assert type(default_prox.values[0][0]) == np.float64
    assert default_prox.shape == raster.shape
    # number of non-zeros (target pixels) in original image
    # must be equal to the number of zeros (target pixels) in proximity matrix
    assert len(np.where(raster.data != 0)[0]) == \
        len(np.where(default_prox.data == 0)[0])

    # TARGET VALUES SETTING
    target_values = [2, 3]
    target_prox = proximity(raster, x='lon', y='lat',
                            target_values=target_values)
    # output must be an xarray DataArray
    assert isinstance(target_prox, xa.DataArray)
    assert isinstance(target_prox.values, np.ndarray)
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
    assert isinstance(manhattan_prox.values, np.ndarray)
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
    assert isinstance(great_circle_prox.values, np.ndarray)
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
    raster = create_test_raster()
    idy, idx = np.where(raster.data != 0)
    # create test raster, all non-zero cells are unique,
    # this is to test against corresponding proximity
    for i in range(len(idx)):
        raster.data[idy[i], idx[i]] = i + 1

    allocation_agg = allocation(raster, x='lon', y='lat')
    # output must be an xarray DataArray
    assert isinstance(allocation_agg, xa.DataArray)
    assert type(allocation_agg.values[0][0]) == raster.dtype
    assert allocation_agg.shape == raster.shape
    # targets not specified,
    # Thus, targets are set to non-zero values of input @raster
    targets = np.unique(raster.data[np.where(raster.data != 0)])
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
            assert proximity_agg.data[y, x] == sqrt(d)
