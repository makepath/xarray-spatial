import pytest
import xarray as xr
import numpy as np

from xrspatial import slope


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


def test_invalid_res_attr():
    """
    Assert 'res' attribute of input xarray
    """
    da = xr.DataArray(data_gaussian, attrs={})
    with pytest.raises(ValueError) as e_info:
        da_slope = slope(da)
        assert e_info

    da = xr.DataArray(data_gaussian, attrs={'res': 'any_string'})
    with pytest.raises(ValueError) as e_info:
        da_slope = slope(da)
        assert e_info

    da = xr.DataArray(data_gaussian, attrs={'res': ('str_tuple', 'str_tuple')})
    with pytest.raises(ValueError) as e_info:
        da_slope = slope(da)
        assert e_info

    da = xr.DataArray(data_gaussian, attrs={'res': (1, 2, 3)})
    with pytest.raises(ValueError) as e_info:
        da_slope = slope(da)
        assert e_info


def test_slope_transfer_function():
    """
    Assert slope transfer function
    """
    da = xr.DataArray(data_gaussian, attrs={'res': 1})
    da_slope = slope(da)
    assert da_slope.dims == da.dims
    assert da_slope.coords == da.coords
    assert da_slope.attrs == da.attrs
    assert da.shape == da_slope.shape

    assert da_slope.sum() > 0

    # In the middle of the array, there is the maximum of the gaussian;
    # And there the slope must be zero.
    _imax = np.where(da == da.max())
    assert da_slope[_imax] == 0

    # same result when cellsize_x = cellsize_y = 1
    da = xr.DataArray(data_gaussian, attrs={'res': (1.0, 1.0)})
    da_slope = slope(da)
    assert da_slope.dims == da.dims
    assert da_slope.coords == da.coords
    assert da_slope.attrs == da.attrs
    assert da.shape == da_slope.shape

    assert da_slope.sum() > 0

    # In the middle of the array, there is the maximum of the gaussian;
    # And there the slope must be zero.
    _imax = np.where(da == da.max())
    assert da_slope[_imax] == 0
