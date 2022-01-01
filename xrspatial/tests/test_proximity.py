import pytest

import numpy as np
import xarray as xr
import dask.array as da

from xrspatial import proximity, allocation, direction
from xrspatial import great_circle_distance
from xrspatial import euclidean_distance
from xrspatial.proximity import _calc_direction

from xrspatial.tests.general_checks import general_output_checks


def test_great_circle_distance():
    # invalid x_coord
    ys = [0, 0, -91, 91]
    xs = [-181, 181, 0, 0]
    for x, y in zip(xs, ys):
        with pytest.raises(Exception) as e_info:
            great_circle_distance(x1=0, x2=x, y1=0, y2=y)
            assert e_info


def create_test_raster():
    height, width = 4, 6
    data = np.asarray([[0., 0., 0., 0., 0., 2.],
                       [0., 0., 1., 0., 0., 0.],
                       [0., np.inf, 3., 0., 0., 0.],
                       [4., 0., 0., 0., np.nan, 0.]])
    _lon = np.linspace(-20, 20, width)
    _lat = np.linspace(20, -20, height)

    numpy_agg = xr.DataArray(data, dims=['lat', 'lon'])
    dask_numpy_agg = xr.DataArray(
        da.from_array(data, chunks=(4, 3)), dims=['lat', 'lon'])
    numpy_agg['lon'] = dask_numpy_agg['lon'] = _lon
    numpy_agg['lat'] = dask_numpy_agg['lat'] = _lat

    return numpy_agg, dask_numpy_agg


def test_proximity():
    raster_numpy, raster_dask = create_test_raster()

    # DEFAULT SETTINGS
    expected_results_default = np.array([
        [20.82733247, 15.54920505, 13.33333333, 15.54920505,  8., 0.],
        [16., 8., 0., 8., 15.54920505, 13.33333333],
        [13.33333333, 8., 0., 8., 16., 24.],
        [0., 8., 13.33333333, 15.54920505, 20.82733247, 27.45501371]
    ])
    # numpy case
    default_prox = proximity(raster_numpy, x='lon', y='lat')
    general_output_checks(raster_numpy, default_prox, expected_results_default)
    # dask case
    default_prox_dask = proximity(raster_dask, x='lon', y='lat')
    general_output_checks(
        raster_dask, default_prox_dask, expected_results_default)

    # TARGET VALUES SETTING
    target_values = [2, 3]
    expected_results_target = np.array([
        [31.09841011, 27.84081736, 24., 16., 8., 0.],
        [20.82733247, 15.54920505, 13.33333333, 15.54920505, 15.54920505, 13.33333333],  # noqa
        [16., 8., 0., 8., 16., 24.],
        [20.82733247, 15.54920505, 13.33333333, 15.54920505, 20.82733247, 27.45501371]  # noqa
    ])
    # numpy case
    target_prox = proximity(
        raster_numpy, x='lon', y='lat', target_values=target_values)
    general_output_checks(raster_numpy, target_prox, expected_results_target)
    # dask case
    target_prox_dask = proximity(
        raster_dask, x='lon', y='lat', target_values=target_values)
    general_output_checks(
        raster_dask, target_prox_dask, expected_results_target)

    # distance_metric SETTING: MANHATTAN
    expected_results_manhattan = np.array([
        [29.33333333, 21.33333333, 13.33333333, 16., 8., 0.],
        [16., 8., 0., 8., 16., 13.33333333],
        [13.33333333, 8., 0., 8., 16., 24.],
        [0., 8., 13.33333333, 21.33333333, 29.33333333, 37.33333333]
    ])
    # numpy case
    manhattan_prox = proximity(
        raster_numpy, x='lon', y='lat', distance_metric='MANHATTAN')
    general_output_checks(
        raster_numpy, manhattan_prox, expected_results_manhattan)
    # dask case
    manhattan_prox_dask = proximity(raster_dask, x='lon', y='lat',
                                    distance_metric='MANHATTAN')
    general_output_checks(
        raster_dask, manhattan_prox_dask, expected_results_manhattan)

    # distance_metric SETTING: GREAT_CIRCLE
    expected_results_great_circle = np.array([
        [2278099.27025501, 1717528.97437217, 1484259.87724365, 1673057.17235307, 836769.1780019, 0.],  # noqa
        [1768990.54084204, 884524.60324856, 0., 884524.60324856, 1717528.97437217, 1484259.87724365],  # noqa
        [1484259.87724365, 884524.60324856, 0., 884524.60324856, 1768990.54084204, 2653336.85436932],  # noqa
        [0., 836769.1780019, 1484259.87724365, 1717528.97437217, 2278099.27025501, 2986647.12982316]  # noqa
    ])
    great_circle_prox = proximity(raster_numpy, x='lon', y='lat',
                                  distance_metric='GREAT_CIRCLE')
    general_output_checks(
        raster_numpy, great_circle_prox, expected_results_great_circle)
    # dask case
    great_circle_prox_dask = proximity(
        raster_dask, x='lon', y='lat', distance_metric='GREAT_CIRCLE'
    )
    general_output_checks(
        raster_dask, great_circle_prox_dask, expected_results_great_circle)

    # max_distance setting
    max_distance = 10
    expected_result_max_distance = np.array([
        [np.nan, np.nan, np.nan, np.nan, 8., 0.],
        [np.nan, 8., 0., 8., np.nan, np.nan],
        [np.nan, 8., 0., 8., np.nan, np.nan],
        [0., 8., np.nan, np.nan, np.nan, np.nan]
    ])
    # numpy case
    max_distance_prox = proximity(
        raster_numpy, x='lon', y='lat', max_distance=max_distance
    )
    general_output_checks(
        raster_numpy, max_distance_prox, expected_result_max_distance)
    # dask case
    max_distance_prox_dask = proximity(
        raster_dask, x='lon', y='lat', max_distance=max_distance
    )
    general_output_checks(
        raster_dask, max_distance_prox_dask, expected_result_max_distance
    )


def test_allocation():
    # create test raster, all non-zero cells are unique,
    # this is to test against corresponding proximity
    raster_numpy, raster_dask = create_test_raster()
    expected_results = np.array([
        [1., 1., 1., 1., 2., 2.],
        [1., 1., 1., 1., 2., 2.],
        [4., 3., 3., 3., 3., 3.],
        [4., 4., 3., 3., 3., 3.]
    ])
    allocation_agg = allocation(raster_numpy, x='lon', y='lat')
    general_output_checks(raster_numpy, allocation_agg, expected_results)

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
            assert proximity_agg.data[y, x] == np.float32(d)

    # dask case
    allocation_agg_dask = allocation(raster_dask, x='lon', y='lat')
    general_output_checks(raster_dask, allocation_agg_dask, expected_results)

    # max_distance setting
    max_distance = 10
    expected_results_max_distance = np.array([
        [np.nan, np.nan, np.nan, np.nan, 2., 2.],
        [np.nan, 1., 1., 1., np.nan, np.nan],
        [np.nan, 3., 3., 3., np.nan, np.nan],
        [4., 4., np.nan, np.nan, np.nan, np.nan]
    ])
    # numpy case
    max_distance_alloc = allocation(
        raster_numpy, x='lon', y='lat', max_distance=max_distance
    )
    general_output_checks(
        raster_numpy, max_distance_alloc, expected_results_max_distance)
    # dask case
    max_distance_alloc_dask = allocation(
        raster_dask, x='lon', y='lat', max_distance=max_distance
    )
    general_output_checks(
        raster_dask, max_distance_alloc_dask, expected_results_max_distance
    )


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

    expected_results = np.array([
        [50.194427, 30.963757, 360., 329.03625, 90., 0.],
        [90., 90., 0., 270., 149.03624, 180.],
        [360., 90., 0., 270., 270., 270.],
        [0., 270., 180., 210.96376, 230.19443, 240.9454]
    ], dtype=np.float32)

    # numpy case
    direction_agg = direction(raster_numpy, x='lon', y='lat')
    # output must be an xarray DataArray
    general_output_checks(raster_numpy, direction_agg, expected_results)

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
    general_output_checks(raster_dask, direction_agg_dask, expected_results)

    # max_distance setting
    max_distance = 10
    expected_results_max_distance = np.array([
        [np.nan, np.nan, np.nan, np.nan, 90., 0.],
        [np.nan, 90., 0., 270., np.nan, np.nan],
        [np.nan, 90., 0., 270., np.nan, np.nan],
        [0., 270., np.nan, np.nan, np.nan, np.nan]
    ], dtype=np.float32)
    # numpy case
    max_distance_direction = direction(
        raster_numpy, x='lon', y='lat', max_distance=max_distance
    )
    general_output_checks(
        raster_numpy, max_distance_direction, expected_results_max_distance
    )
    # dask case
    max_distance_direction_dask = direction(
        raster_dask, x='lon', y='lat', max_distance=max_distance
    )
    general_output_checks(
        raster_dask, max_distance_direction_dask, expected_results_max_distance
    )
