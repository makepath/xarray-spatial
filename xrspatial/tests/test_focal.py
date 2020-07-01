import xarray as xr
import numpy as np

from xrspatial import mean
from xrspatial.focal import apply, Kernel, calc_mean, calc_sum, hotspots
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
    with pytest.raises(Exception) as e_info:
        invalid_shape_kernel = Kernel(shape='line')

    with pytest.raises(Exception) as e_info:
        invalid_radius_kernel = Kernel(radius='10 inch')


def test_apply_invalid_input():
    kernel = Kernel(radius=1)
    invalid_raster_type = np.array([0, 1, 2, 3])
    with pytest.raises(Exception) as e_info:
        apply(invalid_raster_type, kernel)
        assert e_info

    invalid_raster_dtype = xr.DataArray(np.array([['cat', 'dog']]))
    with pytest.raises(Exception) as e_info:
        apply(invalid_raster_dtype, kernel)
        assert e_info

    invalid_raster_shape = xr.DataArray(np.array([0, 0]))
    with pytest.raises(Exception) as e_info:
        apply(invalid_raster_shape, kernel)
        assert e_info


def test_apply_mean():
    n, m = 10, 15
    raster = xr.DataArray(np.ones((n, m)), dims=['x', 'y'])
    raster['x'] = np.linspace(0, n, n)
    raster['y'] = np.linspace(0, m, m)

    for i in range(m):
        kernel = Kernel(radius=i)
        mean_output = apply(raster, kernel)

        # check output's properties
        # output must be an xarray DataArray
        assert isinstance(mean_output, xr.DataArray)
        assert isinstance(mean_output.values, np.ndarray)
        # shape, dims, coords, attr preserved
        assert raster.shape == mean_output.shape
        assert raster.dims == mean_output.dims
        assert raster.attrs == mean_output.attrs
        for coord in raster.coords:
            assert np.all(raster[coord] == mean_output[coord])
        assert (np.all(mean_output.values == np.ones((n, m))))


def test_apply_sum():
    n, m = 10, 15
    raster = xr.DataArray(np.ones((n, m)), dims=['x', 'y'])
    raster['x'] = np.linspace(0, n, n)
    raster['y'] = np.linspace(0, m, m)

    for i in range(m):
        kernel = Kernel(radius=i)

        kernel_array = kernel.to_array(raster)
        krows, kcols = kernel_array.shape
        hrows, hcols = min(n, int(krows / 2)), min(m, int(kcols / 2))

        sum_output = apply(raster, kernel, calc_sum)

        # check output's properties
        # output must be an xarray DataArray
        assert isinstance(sum_output, xr.DataArray)
        assert isinstance(sum_output.values, np.ndarray)
        # shape, dims, coords, attr preserved
        assert raster.shape == sum_output.shape
        assert raster.dims == sum_output.dims
        assert raster.attrs == sum_output.attrs
        for coord in raster.coords:
            assert np.all(raster[coord] == sum_output[coord])

        kernel_sum = np.sum(kernel_array)
        raster_sum = np.sum(raster.values)
        # in case the kernel smaller than the raster
        # cells that fit the kernel has neighborhood sum equal to sum(kernel)
        for y in range(hrows, n - hrows):
            for x in range(hcols, m - hcols):
                assert sum_output.values[y, x] == min(kernel_sum, raster_sum)

        #  cell in border has sum less than np.sum(kernel)
        for y in range(hrows):
            for x in range(m):
                assert sum_output.values[y, x] < kernel_sum

        for y in range(n-hrows, n):
            for x in range(m):
                assert sum_output.values[y, x] < kernel_sum

        for y in range(n):
            for x in range(hcols):
                assert sum_output.values[y, x] < kernel_sum

        for y in range(n):
            for x in range(m-hcols, m):
                assert sum_output.values[y, x] < kernel_sum


def test_apply_with_nan():
    n, m = 6, 6
    raster = xr.DataArray(np.ones((n, m)), dims=['x', 'y'])
    raster['x'] = np.linspace(0, n, n)
    raster['y'] = np.linspace(0, m, m)

    nan_cells = [(i, i) for i in range(n)]
    for cell in nan_cells:
        raster[cell[0], cell[1]] = np.nan

    kernel_1 = Kernel(radius=1)
    # kernel array = [[1]]
    sum_output_1 = apply(raster, kernel_1, calc_sum)
    # np.nansum(np.array([np.nan])) = 0.0
    expected_out_sum_1 = np.array([[0., 1., 1., 1., 1., 1.],
                                   [1., 0., 1., 1., 1., 1.],
                                   [1., 1., 0., 1., 1., 1.],
                                   [1., 1., 1., 0., 1., 1.],
                                   [1., 1., 1., 1., 0., 1.],
                                   [1., 1., 1., 1., 1., 0.]])
    assert np.all(sum_output_1.values == expected_out_sum_1)

    # np.nanmean(np.array([np.nan])) = nan
    mean_output_1 = apply(raster, kernel_1, calc_mean)
    for cell in nan_cells:
        assert np.isnan(mean_output_1[cell[0], cell[1]])
    # remaining cells are 1s
    for i in range(n):
        for j in range(m):
            if i != j:
                assert mean_output_1[i, j] == 1

    kernel_2 = Kernel(radius=2)
    # kernel array: [[0, 1, 0],
    #                [1, 1, 1],
    #                [0, 1, 0]]
    sum_output_2 = apply(raster, kernel_2, calc_sum)
    expected_out_sum_2 = np.array([[2., 2., 4., 4., 4., 3.],
                                   [2., 4., 3., 5., 5., 4.],
                                   [4., 3., 4., 3., 5., 4.],
                                   [4., 5., 3., 4., 3., 4.],
                                   [4., 5., 5., 3., 4., 2.],
                                   [3., 4., 4., 4., 2., 2.]])

    assert np.all(sum_output_2.values == expected_out_sum_2)

    mean_output_2 = apply(raster, kernel_2, calc_mean)
    expected_mean_output_2 = np.ones((n, m))
    assert np.all(mean_output_2.values == expected_mean_output_2)


def test_hotspot_invalid():
    kernel = Kernel(radius=1)
    invalid_raster_type = np.array([0, 1, 2, 3])
    with pytest.raises(Exception) as e_info:
        hotspots(invalid_raster_type, kernel)
        assert e_info

    invalid_raster_dtype = xr.DataArray(np.array([['cat', 'dog']]))
    with pytest.raises(Exception) as e_info:
        hotspots(invalid_raster_dtype, kernel)
        assert e_info

    invalid_raster_shape = xr.DataArray(np.array([0, 0]))
    with pytest.raises(Exception) as e_info:
        hotspots(invalid_raster_shape, kernel)
        assert e_info

    invalid_raster_std = xr.DataArray(np.ones((10, 10)))
    # std of the raster is 0
    with pytest.raises(Exception) as e_info:
        hotspots(invalid_raster_std, kernel)
        assert e_info


def test_hotspot():
    n, m = 10, 10
    raster = xr.DataArray(np.zeros((n, m), dtype=float), dims=['x', 'y'])
    raster['x'] = np.linspace(0, n, n)
    raster['y'] = np.linspace(0, m, m)

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

    kernel = Kernel(radius=2)
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
