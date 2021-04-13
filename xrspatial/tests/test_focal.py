import xarray as xr
import numpy as np

from xrspatial import mean
from xrspatial.convolution import convolve_2d
from xrspatial.focal import (
    calc_cellsize,
    hotspots,
    circle_kernel,
    annulus_kernel,
    _validate_kernel,
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


def test_kernel():
    n, m = 6, 6
    raster = xr.DataArray(np.ones((n, m)), dims=['y', 'x'])
    raster['x'] = np.linspace(0, n, n)
    raster['y'] = np.linspace(0, m, m)

    cellsize_x, cellsize_y = calc_cellsize(raster)
    # Passing invalid radius units for `circle`
    with pytest.raises(Exception) as e_info:
        circle_kernel(cellsize_x, cellsize_y, "10 furlongs")
        assert e_info

    # Passing invalid radius for `annulus`
    with pytest.raises(Exception) as e_info:
        annulus_kernel(cellsize_x, cellsize_y, 4, "2 leagues")
        assert e_info

    # Passing custom kernel with even dimensions
    with pytest.raises(Exception) as e_info:
        _validate_kernel(np.ones((2, 2)))
        assert e_info

    # Passing custom kernel of wrong type
    with pytest.raises(Exception) as e_info:
        _validate_kernel([[1, 1, 1]])
        assert e_info


def test_convolution():
    n, m = 6, 6
    raster = xr.DataArray(np.ones((n, m)), dims=['y', 'x'])
    raster['x'] = np.linspace(0, n, n)
    raster['y'] = np.linspace(0, m, m)
    cellsize_x, cellsize_y = calc_cellsize(raster)

    # add some nan pixels
    nan_cells = [(i, i) for i in range(n)]
    for cell in nan_cells:
        raster[cell[0], cell[1]] = np.nan

    # kernel array = [[1]]
    kernel = np.ones((1, 1))

    # np.nansum(np.array([np.nan])) = 0.0
    expected_out_sum_1 = np.array([[0., 1., 1., 1., 1., 1.],
                                   [1., 0., 1., 1., 1., 1.],
                                   [1., 1., 0., 1., 1., 1.],
                                   [1., 1., 1., 0., 1., 1.],
                                   [1., 1., 1., 1., 0., 1.],
                                   [1., 1., 1., 1., 1., 0.]])
    # Convolution will return np.nan, so convert nan to 0
    assert np.all(np.nan_to_num(expected_out_sum_1) == expected_out_sum_1)

    # np.nanmean(np.array([np.nan])) = nan
    mean_output_1 = convolve_2d(raster.values, kernel / kernel.sum())
    for cell in nan_cells:
        assert np.isnan(mean_output_1[cell[0], cell[1]])
    # remaining cells are 1s
    for i in range(n):
        for j in range(m):
            if i != j:
                assert mean_output_1[i, j] == 1

    # kernel array: [[0, 1, 0],
    #                [1, 1, 1],
    #                [0, 1, 0]]
    kernel = circle_kernel(cellsize_x, cellsize_y, 2)
    sum_output_2 = convolve_2d(np.nan_to_num(raster.values), kernel, pad=False)
    expected_out_sum_2 = np.array([[2., 2., 4., 4., 4., 3.],
                                   [2., 4., 3., 5., 5., 4.],
                                   [4., 3., 4., 3., 5., 4.],
                                   [4., 5., 3., 4., 3., 4.],
                                   [4., 5., 5., 3., 4., 2.],
                                   [3., 4., 4., 4., 2., 2.]])

    assert np.all(sum_output_2 == expected_out_sum_2)

    mean_output_2 = convolve_2d(np.ones((n, m)),
                                kernel / kernel.sum(),
                                pad=True)
    expected_mean_output_2 = np.ones((n, m))
    assert np.all(mean_output_2 == expected_mean_output_2)

    # kernel array: [[0, 1, 0],
    #                [1, 0, 1],
    #                [0, 1, 0]]
    kernel = annulus_kernel(cellsize_x, cellsize_y, 2.0, 0.5)
    sum_output_3 = convolve_2d(np.nan_to_num(raster.values), kernel, pad=False)
    expected_out_sum_3 = np.array([[2., 1., 3., 3., 3., 2.],
                                   [1., 4., 2., 4., 4., 3.],
                                   [3., 2., 4., 2., 4., 3.],
                                   [3., 4., 2., 4., 2., 3.],
                                   [3., 4., 4., 2., 4., 1.],
                                   [2., 3., 3., 3., 1., 2.]])

    assert np.all(sum_output_3 == expected_out_sum_3)

    mean_output_3 = convolve_2d(np.ones((n, m)),
                                kernel / kernel.sum(),
                                pad=True)
    expected_mean_output_3 = np.ones((n, m))
    assert np.all(mean_output_3 == expected_mean_output_3)


def test_hotspot():
    n, m = 10, 10
    raster = xr.DataArray(np.zeros((n, m), dtype=float), dims=['y', 'x'])
    raster['x'] = np.linspace(0, n, n)
    raster['y'] = np.linspace(0, m, m)
    cellsize_x, cellsize_y = calc_cellsize(raster)

    kernel = circle_kernel(cellsize_x, cellsize_y, 2.0)

    all_idx = zip(*np.where(raster.values == 0))

    nan_cells = [(i, i) for i in range(m)]
    for cell in nan_cells:
        raster[cell[0], cell[1]] = np.nan

    # add some extreme values
    hot_region = [(1, 1), (1, 2), (1, 3),
                  (2, 1), (2, 2), (2, 3),
                  (3, 1), (3, 2), (3, 3)]
    cold_region = [(7, 7), (7, 8), (7, 9),
                   (8, 7), (8, 8), (8, 9),
                   (9, 7), (9, 8), (9, 9)]
    for p in hot_region:
        raster[p[0], p[1]] = 10000
    for p in cold_region:
        raster[p[0], p[1]] = -10000

    no_significant_region = [id for id in all_idx if id not in hot_region and
                             id not in cold_region]

    hotspots_output = hotspots(raster, kernel)

    # check output's properties
    # output must be an xarray DataArray
    assert isinstance(hotspots_output, xr.DataArray)
    assert isinstance(hotspots_output.values, np.ndarray)
    assert issubclass(hotspots_output.values.dtype.type, np.int8)

    # shape, dims, coords, attr preserved
    assert raster.shape == hotspots_output.shape
    assert raster.dims == hotspots_output.dims
    assert raster.attrs == hotspots_output.attrs
    for coord in raster.coords:
        assert np.all(raster[coord] == hotspots_output[coord])

    # no nan in output
    assert not np.isnan(np.min(hotspots_output))

    # output of extreme regions are non-zeros
    # hot spots
    hot_spot = np.asarray([hotspots_output[p] for p in hot_region])
    assert np.all(hot_spot >= 0)
    assert np.sum(hot_spot) > 0
    # cold spots
    cold_spot = np.asarray([hotspots_output[p] for p in cold_region])
    assert np.all(cold_spot <= 0)
    assert np.sum(cold_spot) < 0
    # output of no significant regions are 0s
    no_sign = np.asarray([hotspots_output[p] for p in no_significant_region])
    assert np.all(no_sign == 0)
