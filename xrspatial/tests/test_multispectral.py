import xarray as xr
import numpy as np
import xarray as xa

from xrspatial.spectral import ndvi
from xrspatial.spectral import savi


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


def create_test_arr(arr):
    n, m = arr.shape
    raster = xa.DataArray(arr, dims=['y', 'x'])
    raster['y'] = np.linspace(0, n, n)
    raster['x'] = np.linspace(0, m, m)
    return raster

def test_ndvi():
    """
    Assert aspect transfer function
    """
    _x = np.mgrid[1:0:21j]
    a, b = np.meshgrid(_x, _x)
    red = a*b
    nir = (a*b)[::-1, ::-1]

    da_red = xr.DataArray(red, dims=['y', 'x'])
    da_nir = xr.DataArray(nir, dims=['y', 'x'])

    da_ndvi = ndvi(da_nir, da_red)

    assert da_ndvi.dims == da_nir.dims
    assert da_ndvi.coords == da_nir.coords
    assert da_ndvi.attrs == da_nir.attrs

    assert da_ndvi[0, 0] == -1
    assert da_ndvi[-1, -1] == 1
    assert da_ndvi[5, 10] == da_ndvi[10, 5] == -0.5
    assert da_ndvi[15, 10] == da_ndvi[10, 15] == 0.5

def test_savi():

    max_val = 2**16 - 1

    arr1 = np.array([[max_val, max_val, max_val, max_val],
                     [max_val, 1000.0, 1000.0, max_val],
                     [max_val, 1000.0, 1000.0, max_val],
                     [max_val, 1000.0, 1000.0, max_val],
                     [max_val, max_val, max_val, max_val]], dtype=np.float64)

    arr2 = np.array([[100.0, 100.0, 100.0, 100.0],
                     [100.0, max_val, max_val, 100.0],
                     [100.0, max_val, max_val, 100.0],
                     [100.0, max_val, max_val, 100.0],
                     [100.0, 100.0, 100.0, 100.0]], dtype=np.float64)

    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)

    soil_factor = 0.0

    result_savi = savi(nir, red, soil_factor=0.0)
    result_ndvi = ndvi(nir, red)

    assert result_savi == result_ndvi

    result_savi = savi(nir, red)
    assert result_savi.dims == nir.dims
    assert result_savi.coords == nir.coords
    assert result_savi.attrs == nir.attrs

















