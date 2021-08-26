import pytest

from xrspatial import proximity, allocation, direction
from xrspatial import great_circle_distance, manhattan_distance
from xrspatial import euclidean_distance
from xrspatial.proximity import _calc_direction

import numpy as np
import xarray as xr
import dask.array as da


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
    data = np.asarray([[0., 0., 0., 0., 0., 0., 0., 0., 0., 2.],
                       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                       [0., 0., np.inf, 0., 3., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [4., 0., 0., 0., 0., 0., np.nan, 0., 0., 0.]])
    _lon = np.linspace(-20, 20, width)
    _lat = np.linspace(20, -20, height)

    numpy_agg = xr.DataArray(data, dims=['lat', 'lon'])
    dask_numpy_agg = xr.DataArray(
        da.from_array(data, chunks=(3, 3)), dims=['lat', 'lon'])
    numpy_agg['lon'] = dask_numpy_agg['lon'] = _lon
    numpy_agg['lat'] = dask_numpy_agg['lat'] = _lat

    return numpy_agg, dask_numpy_agg


def test_proximity():
    raster_numpy, raster_dask = create_test_raster()

    # DEFAULT SETTINGS
    # numpy case
    default_prox = proximity(raster_numpy, x='lon', y='lat')
    # output must be an xarray DataArray
    assert isinstance(default_prox, xr.DataArray)
    assert type(default_prox.data[0][0]) == np.float64
    assert default_prox.shape == raster_numpy.shape
    # in this test case, where no polygon is completely inside another polygon,
    # number of non-zeros (target pixels) in original image
    # must be equal to the number of zeros (target pixels) in proximity matrix
    assert len(np.where((raster_numpy.data != 0) &
                        np.isfinite(raster_numpy.data))[0]) == \
        len(np.where(default_prox.data == 0)[0])

    # dask case
    default_prox_dask = proximity(raster_dask, x='lon', y='lat')
    assert isinstance(default_prox_dask.data, da.Array)
    assert np.isclose(
        default_prox.data, default_prox_dask.compute().data, equal_nan=True
    ).all()

    # TARGET VALUES SETTING
    target_values = [2, 3]
    # numpy case
    target_prox = proximity(raster_numpy, x='lon', y='lat',
                            target_values=target_values)
    # output must be an xarray DataArray
    assert isinstance(target_prox, xr.DataArray)
    assert type(target_prox.data[0][0]) == np.float64
    assert target_prox.shape == raster_numpy.shape
    assert (len(np.where(raster_numpy.data == 2)[0]) +
            len(np.where(raster_numpy.data == 3)[0])) == \
        len(np.where(target_prox.data == 0)[0])

    # dask case
    target_prox_dask = proximity(raster_dask, x='lon', y='lat',
                                 target_values=target_values)
    assert isinstance(target_prox_dask.data, da.Array)
    assert np.isclose(
        target_prox.data, target_prox_dask.compute().data, equal_nan=True
    ).all()

    # distance_metric SETTING: MANHATTAN
    # numpy case
    manhattan_prox = proximity(raster_numpy, x='lon', y='lat',
                               distance_metric='MANHATTAN')
    # output must be an xarray DataArray
    assert isinstance(manhattan_prox, xr.DataArray)
    assert type(manhattan_prox.data[0][0]) == np.float64
    assert manhattan_prox.shape == raster_numpy.shape
    # all output values must be in range [0, max_possible_dist]
    max_possible_dist = manhattan_distance(
        raster_numpy.coords['lon'].data[0],
        raster_numpy.coords['lon'].data[-1],
        raster_numpy.coords['lat'].data[0],
        raster_numpy.coords['lat'].data[-1]
    )
    assert np.nanmax(manhattan_prox.data) <= max_possible_dist
    assert np.nanmin(manhattan_prox.data) == 0
    # dask case
    manhattan_prox_dask = proximity(raster_dask, x='lon', y='lat',
                                    distance_metric='MANHATTAN')
    assert isinstance(manhattan_prox_dask.data, da.Array)
    assert np.isclose(
        manhattan_prox.data, manhattan_prox_dask.compute().data, equal_nan=True
    ).all()

    # distance_metric SETTING: GREAT_CIRCLE
    great_circle_prox = proximity(raster_numpy, x='lon', y='lat',
                                  distance_metric='GREAT_CIRCLE')
    # output must be an xarray DataArray
    assert isinstance(great_circle_prox, xr.DataArray)
    assert type(great_circle_prox.data[0][0]) == np.float64
    assert great_circle_prox.shape == raster_numpy.shape
    # all output values must be in range [0, max_possible_dist]
    max_possible_dist = great_circle_distance(
        raster_numpy.coords['lon'].data[0],
        raster_numpy.coords['lon'].data[-1],
        raster_numpy.coords['lat'].data[0],
        raster_numpy.coords['lat'].data[-1]
    )
    assert np.nanmax(great_circle_prox.data) <= max_possible_dist
    assert np.nanmin(great_circle_prox.data) == 0
    # dask case
    great_circle_prox_dask = proximity(
        raster_dask, x='lon', y='lat', distance_metric='GREAT_CIRCLE'
    )
    assert isinstance(great_circle_prox_dask.data, da.Array)
    assert np.isclose(
        great_circle_prox.data, great_circle_prox_dask.compute().data,
        equal_nan=True
    ).all()

    # max_distance setting
    for max_distance in range(0, 25):
        # numpy case
        max_distance_prox = proximity(
            raster_numpy, x='lon', y='lat', max_distance=max_distance
        )
        # no proximity distances greater than max_distance
        assert np.nanmax(max_distance_prox.data) <= max_distance

        # dask case
        max_distance_prox_dask = proximity(
            raster_dask, x='lon', y='lat', max_distance=max_distance
        )
        assert isinstance(max_distance_prox_dask.data, da.Array)
        assert np.isclose(
            max_distance_prox.data, max_distance_prox_dask.compute().data,
            equal_nan=True
        ).all()


def test_allocation():
    # create test raster, all non-zero cells are unique,
    # this is to test against corresponding proximity
    raster_numpy, raster_dask = create_test_raster()

    allocation_agg = allocation(raster_numpy, x='lon', y='lat')
    # output must be an xarray DataArray
    assert isinstance(allocation_agg, xr.DataArray)
    assert type(allocation_agg.data[0][0]) == raster_numpy.dtype
    assert allocation_agg.shape == raster_numpy.shape
    # targets not specified,
    # Thus, targets are set to non-zero values of input @raster
    targets = np.unique(raster_numpy.data[np.where(
        (raster_numpy.data != 0) & np.isfinite(raster_numpy.data))])
    # non-zero cells (a.k.a targets) remain the same
    for t in targets:
        ry, rx = np.where(raster_numpy.data == t)
        for y, x in zip(ry, rx):
            assert allocation_agg.data[y, x] == t
    # values of allocation output
    assert (np.unique(allocation_agg.data) == targets).all()

    # check against corresponding proximity
    proximity_agg = proximity(raster_numpy, x='lon', y='lat')
    xcoords = allocation_agg['lon'].data
    ycoords = allocation_agg['lat'].data

    for y in range(raster_numpy.shape[0]):
        for x in range(raster_numpy.shape[1]):
            a = allocation_agg.data[y, x]
            py, px = np.where(raster_numpy.data == a)
            # non-zero cells in raster are unique, thus len(px)=len(py)=1
            d = euclidean_distance(xcoords[x], xcoords[px[0]],
                                   ycoords[y], ycoords[py[0]])
            assert proximity_agg.data[y, x] == d

    # dask case
    allocation_agg_dask = allocation(raster_dask, x='lon', y='lat')
    assert isinstance(allocation_agg_dask.data, da.Array)
    assert np.isclose(
        allocation_agg.data, allocation_agg_dask.compute().data, equal_nan=True
    ).all()

    # max_distance setting
    for max_distance in range(0, 25):
        # numpy case
        max_distance_alloc = allocation(
            raster_numpy, x='lon', y='lat', max_distance=max_distance
        )
        # dask case
        max_distance_alloc_dask = allocation(
            raster_dask, x='lon', y='lat', max_distance=max_distance
        )
        assert isinstance(max_distance_alloc_dask.data, da.Array)
        assert np.isclose(
            max_distance_alloc.data, max_distance_alloc_dask.compute().data,
            equal_nan=True
        ).all()


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
    raster_numpy, raster_dask = create_test_raster()

    # numpy case
    direction_agg = direction(raster_numpy, x='lon', y='lat')

    # output must be an xarray DataArray
    assert isinstance(direction_agg, xr.DataArray)
    assert type(direction_agg.data[0][0]) == np.float64
    assert direction_agg.shape == raster_numpy.shape
    assert direction_agg.dims == raster_numpy.dims
    assert direction_agg.attrs == raster_numpy.attrs
    for c in direction_agg.coords:
        assert (direction_agg[c] == raster_numpy.coords[c]).all()

    # in this test case, where no polygon is completely inside another polygon,
    # number of non-zeros (target pixels) in original image
    # must be equal to the number of zeros (target pixels) in proximity matrix
    assert len(np.where((raster_numpy.data != 0) &
                        np.isfinite(raster_numpy.data))[0]) == \
        len(np.where(direction_agg.data == 0)[0])

    # values are within [0, 360]
    assert np.min(direction_agg.data) >= 0
    assert np.max(direction_agg.data) <= 360

    # test against allocation
    allocation_agg = allocation(raster_numpy, x='lon', y='lat')
    xcoords = allocation_agg['lon'].data
    ycoords = allocation_agg['lat'].data

    for y in range(raster_numpy.shape[0]):
        for x in range(raster_numpy.shape[1]):
            a = allocation_agg.data[y, x]
            py, px = np.where(raster_numpy.data == a)
            # non-zero cells in raster are unique, thus len(px)=len(py)=1
            d = _calc_direction(xcoords[x], xcoords[px[0]],
                                ycoords[y], ycoords[py[0]])
            assert direction_agg.data[y, x] == d

    # dask case
    direction_agg_dask = direction(raster_dask, x='lon', y='lat')
    assert isinstance(direction_agg_dask.data, da.Array)
    assert np.isclose(
        direction_agg.data, direction_agg_dask.compute().data, equal_nan=True
    ).all()

    # max_distance setting
    for max_distance in range(0, 25):
        # numpy case
        max_distance_direction = direction(
            raster_numpy, x='lon', y='lat', max_distance=max_distance
        )
        # dask case
        max_distance_direction_dask = direction(
            raster_dask, x='lon', y='lat', max_distance=max_distance
        )
        assert isinstance(max_distance_direction_dask.data, da.Array)
        assert np.isclose(
            max_distance_direction.data,
            max_distance_direction_dask.compute().data,
            equal_nan=True
        ).all()
