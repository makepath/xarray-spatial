import xarray as xr
import numpy as np
import pytest

from xrspatial import aspect


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
#
# -----

data_random = np.random.random_sample((100, 100))
data_random_sparse = _do_sparse_array(data_random)
data_gaussian = _do_gaussian_array()


def test_aspect_transfer_function():
    """
    Assert aspect transfer function
    """
    da = xr.DataArray(data_gaussian, dims=['y', 'x'], attrs={'res': 1})
    da_aspect = aspect(da)
    assert da_aspect.dims == da.dims
    assert da_aspect.coords == da.coords
    assert da_aspect.attrs == da.attrs
    assert da.shape == da_aspect.shape
    assert pytest.approx(da_aspect.data.max(), .1) == 360.
    assert pytest.approx(da_aspect.data.min(), .1) == 0.
