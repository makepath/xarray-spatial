import xarray as xr
import numpy as np

from xrspatial import mean
from xrspatial.focal import focal_analysis
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


def test_focal_invalid_input():
    invalid_raster_type = np.array([0, 1, 2, 3])
    with pytest.raises(Exception) as e_info:
        focal_analysis(invalid_raster_type)
        assert e_info

    invalid_raster_dtype = xr.DataArray(np.array([['cat', 'dog']]))
    with pytest.raises(Exception) as e_info:
        focal_analysis(invalid_raster_dtype)
        assert e_info

    invalid_raster_shape = xr.DataArray(np.array([0, 0]))
    with pytest.raises(Exception) as e_info:
        focal_analysis(invalid_raster_shape)
        assert e_info

    raster = xr.DataArray(np.ones((5, 5)))
    invalid_kernel_shape = 'line'
    with pytest.raises(Exception) as e_info:
        focal_analysis(raster=raster, shape=invalid_kernel_shape)
        assert e_info

    raster = xr.DataArray(np.ones((5, 5)))
    invalid_radius = '10 inch'
    with pytest.raises(Exception) as e_info:
        focal_analysis(raster=raster, radius=invalid_radius)
        assert e_info


def test_focal_default():
    raster = xr.DataArray(np.ones((10, 10)), dims=['x', 'y'])
    raster['x'] = np.linspace(0, 9, 10)
    raster['y'] = np.linspace(0, 9, 10)

    focal_stats = focal_analysis(raster, radius='1m')

    # check output's properties
    # output must be an xarray DataArray
    assert isinstance(focal_stats, xr.DataArray)
    assert isinstance(focal_stats.values, np.ndarray)
    # shape, dims, coords, attr preserved
    assert raster.shape == focal_stats.shape
    assert raster.dims == focal_stats.dims
    assert raster.attrs == focal_stats.attrs
    assert raster.coords == focal_stats.coords

    # TODO: validate output value
