import pytest
import xarray as xr
import numpy as np

import dask.array as da

from xrspatial.utils import doesnt_have_cuda

from xrspatial import mean
from xrspatial.focal import hotspots
from xrspatial.convolution import (
    convolve_2d, cellsize,
    circle_kernel, annulus_kernel,
)
import pytest


def _do_sparse_array(data_array):
    import random
    indx = list(zip(*np.where(data_array)))
    pos = random.sample(range(data_array.size), data_array.size//2)
    indx = np.asarray(indx)[pos]
    r = indx[:, 0]
    c = indx[:, 1]
    data_half = data_array.copy()
    data_half[r, c] = 0
    return data_half


def _do_gaussian_array():
    _x = np.linspace(0, 50, 101)
    _y = _x.copy()
    _mean = 25
    _sdev = 5
    X, Y = np.meshgrid(_x, _y, sparse=True)
    x_fac = -np.power(X-_mean, 2)
    y_fac = -np.power(Y-_mean, 2)
    gaussian = np.exp((x_fac+y_fac)/(2*_sdev**2)) / (2.5*_sdev)
    return gaussian


data_random = np.random.random_sample((100, 100))
data_random_sparse = _do_sparse_array(data_random)
data_gaussian = _do_gaussian_array()


def test_mean_transfer_function():
    da = xr.DataArray(data_random)
    da_mean = mean(da)
    assert da.shape == da_mean.shape

    # Overall mean value should be the same as the original array.
    # Considering the default behaviour to 'mean' is to pad the borders
    # with zeros, the mean value of the filtered array will be slightly
    # smaller (considering 'data_random' is positive).
    assert da_mean.mean() <= data_random.mean()

    # And if we pad the borders with the original values, we should have a
    # 'mean' filtered array with _mean_ value very similar to the original one.
    da_mean[0, :] = data_random[0, :]
    da_mean[-1, :] = data_random[-1, :]
    da_mean[:, 0] = data_random[:, 0]
    da_mean[:, -1] = data_random[:, -1]
    assert abs(da_mean.mean() - data_random.mean()) < 10**-3


convolve_2d_data = np.array([[0., 1., 1., 1., 1., 1.],
                             [1., 0., 1., 1., 1., 1.],
                             [1., 1., 0., 1., 1., 1.],
                             [1., 1., 1., np.nan, 1., 1.],
                             [1., 1., 1., 1., 0., 1.],
                             [1., 1., 1., 1., 1., 0.]])


def test_convolution():
    data = convolve_2d_data

    kernel1 = np.ones((1, 1))
    output_1 = convolve_2d(data, kernel1)
    expected_output_1 = np.array([[0., 1., 1., 1., 1., 1.],
                                  [1., 0., 1., 1., 1., 1.],
                                  [1., 1., 0., 1., 1., 1.],
                                  [1., 1., 1., np.nan, 1., 1.],
                                  [1., 1., 1., 1., 0., 1.],
                                  [1., 1., 1., 1., 1., 0.]])
    assert isinstance(output_1, np.ndarray)
    assert np.isclose(output_1, expected_output_1, equal_nan=True).all()

    kernel2 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    output_2 = convolve_2d(data, kernel2)
    expected_output_2 = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, 4., 3., 5., 5., np.nan],
                                  [np.nan, 3., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, 5., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, 5., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    # kernel2 is of 3x3, thus the border edge is 1 cell long.
    # currently, ignoring border edge (i.e values in edges are all nans)
    assert isinstance(output_2, np.ndarray)
    assert np.isclose(output_2, expected_output_2, equal_nan=True).all()

    kernel3 = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
    output_3 = convolve_2d(data, kernel3)
    expected_output_3 = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, 4., 2., 4., 4., np.nan],
                                  [np.nan, 2., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    # kernel3 is of 3x3, thus the border edge is 1 cell long.
    # currently, ignoring border edge (i.e values in edges are all nans)
    assert isinstance(output_3, np.ndarray)
    assert np.isclose(output_3, expected_output_3, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_2d_convolution_gpu_equals_cpu():

    import cupy

    data = convolve_2d_data
    numpy_agg = xr.DataArray(data)
    cupy_agg = xr.DataArray(cupy.asarray(data))

    kernel1 = np.ones((1, 1))
    output_numpy1 = convolve_2d(numpy_agg.data, kernel1)
    output_cupy1 = convolve_2d(cupy_agg.data, kernel1)
    assert isinstance(output_cupy1, cupy.ndarray)
    assert np.isclose(output_numpy1, output_cupy1.get(), equal_nan=True).all()

    kernel2 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    output_numpy2 = convolve_2d(numpy_agg.data, kernel2)
    output_cupy2 = convolve_2d(cupy_agg.data, kernel2)
    assert isinstance(output_cupy2, cupy.ndarray)
    assert np.isclose(output_numpy2, output_cupy2.get(), equal_nan=True).all()

    kernel3 = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
    output_numpy3 = convolve_2d(numpy_agg.data, kernel3)
    output_cupy3 = convolve_2d(cupy_agg.data, kernel3)
    assert isinstance(output_cupy3, cupy.ndarray)
    assert np.isclose(output_numpy3, output_cupy3.get(), equal_nan=True).all()


# def test_hotspot():
#     n, m = 10, 10
#     raster = xr.DataArray(np.zeros((n, m), dtype=float), dims=['y', 'x'])
#     raster['x'] = np.linspace(0, n, n)
#     raster['y'] = np.linspace(0, m, m)
#     cellsize_x, cellsize_y = cellsize(raster)
#
#     kernel = circle_kernel(cellsize_x, cellsize_y, 2.0)
#
#     all_idx = zip(*np.where(raster.values == 0))
#
#     nan_cells = [(i, i) for i in range(m)]
#     for cell in nan_cells:
#         raster[cell[0], cell[1]] = np.nan
#
#     # add some extreme values
#     hot_region = [(1, 1), (1, 2), (1, 3),
#                   (2, 1), (2, 2), (2, 3),
#                   (3, 1), (3, 2), (3, 3)]
#     cold_region = [(7, 7), (7, 8), (7, 9),
#                    (8, 7), (8, 8), (8, 9),
#                    (9, 7), (9, 8), (9, 9)]
#     for p in hot_region:
#         raster[p[0], p[1]] = 10000
#     for p in cold_region:
#         raster[p[0], p[1]] = -10000
#
#     no_significant_region = [id for id in all_idx if id not in hot_region and
#                              id not in cold_region]
#
#     hotspots_output = hotspots(raster, kernel)
#
#     # check output's properties
#     # output must be an xarray DataArray
#     assert isinstance(hotspots_output, xr.DataArray)
#     assert isinstance(hotspots_output.values, np.ndarray)
#     assert issubclass(hotspots_output.values.dtype.type, np.int8)
#
#     # shape, dims, coords, attr preserved
#     assert raster.shape == hotspots_output.shape
#     assert raster.dims == hotspots_output.dims
#     assert raster.attrs == hotspots_output.attrs
#     for coord in raster.coords:
#         assert np.all(raster[coord] == hotspots_output[coord])
#
#     # no nan in output
#     assert not np.isnan(np.min(hotspots_output))
#
#     # output of extreme regions are non-zeros
#     # hot spots
#     hot_spot = np.asarray([hotspots_output[p] for p in hot_region])
#     assert np.all(hot_spot >= 0)
#     assert np.sum(hot_spot) > 0
#     # cold spots
#     cold_spot = np.asarray([hotspots_output[p] for p in cold_region])
#     assert np.all(cold_spot <= 0)
#     assert np.sum(cold_spot) < 0
#     # output of no significant regions are 0s
#     no_sign = np.asarray([hotspots_output[p] for p in no_significant_region])
#     assert np.all(no_sign == 0)
