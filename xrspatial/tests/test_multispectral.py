import pytest
import xarray as xr
import numpy as np
import xarray as xa

from xrspatial.utils import doesnt_have_cuda

from xrspatial.multispectral import arvi
from xrspatial.multispectral import ebbi
from xrspatial.multispectral import evi
from xrspatial.multispectral import nbr
from xrspatial.multispectral import nbr2
from xrspatial.multispectral import ndmi
from xrspatial.multispectral import ndvi
from xrspatial.multispectral import savi
from xrspatial.multispectral import gci
from xrspatial.multispectral import sipi


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

arr3 = np.array([[10.0, 10.0, 10.0, 10.0],
                    [10.0, max_val, max_val, 10.0],
                    [10.0, max_val, max_val, 10.0],
                    [10.0, max_val, max_val, 10.0],
                    [10.0, 10.0, 10.0, 10.0]], dtype=np.float64)


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

    da_nir = xr.DataArray(nir, dims=['y', 'x'])
    da_red = xr.DataArray(red, dims=['y', 'x'])

    da_ndvi = ndvi(da_nir, da_red, use_cuda=False)

    assert da_ndvi.dims == da_nir.dims
    assert da_ndvi.attrs == da_nir.attrs
    for coord in da_nir.coords:
        assert np.all(da_nir[coord] == da_ndvi[coord])

    assert da_ndvi[0, 0] == -1
    assert da_ndvi[-1, -1] == 1
    assert da_ndvi[5, 10] == da_ndvi[10, 5] == -0.5
    assert da_ndvi[15, 10] == da_ndvi[10, 15] == 0.5


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ndvi_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)

    # savi should be same as ndvi at soil_factor=0
    cpu = savi(nir, red, use_cuda=False)
    gpu = savi(nir, red, use_cuda=True)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_savi():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)

    # savi should be same as ndvi at soil_factor=0
    result_savi = savi(nir, red, soil_factor=0.0, use_cuda=False)
    result_ndvi = ndvi(nir, red, use_cuda=False)

    assert np.isclose(result_savi.data, result_ndvi.data, equal_nan=True).all()
    assert result_savi.dims == nir.dims

    result_savi = savi(nir, red, soil_factor=1.0)
    assert isinstance(result_savi, xa.DataArray)
    assert result_savi.dims == nir.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_savi_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)

    # savi should be same as ndvi at soil_factor=0
    cpu = savi(nir, red, soil_factor=0.0, use_cuda=False)
    gpu = savi(nir, red, soil_factor=0.0, use_cuda=True)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_avri():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    result = arvi(nir, red, blue, use_cuda=False)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims

@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_avri_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    cpu = arvi(nir, red, blue, use_cuda=False)
    gpu = arvi(nir, red, blue, use_cuda=True)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_evi():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    result = evi(nir, red, blue, use_cuda=False)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims

    # TODO: Test Gain
    # TODO: Test Soil Factor
    # TODO: Test C1
    # TODO: Test C2


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_evi_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    cpu = evi(nir, red, blue, use_cuda=False)
    gpu = evi(nir, red, blue, use_cuda=True)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_gci():
    nir = create_test_arr(arr1)
    green = create_test_arr(arr2)

    result = gci(nir, green, use_cuda=False)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_gci_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    green = create_test_arr(arr2)

    cpu = gci(nir, green, use_cuda=False)
    gpu = gci(nir, green, use_cuda=True)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_sipi():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    result = sipi(nir, red, blue, use_cuda=False)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_sipi_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    cpu = sipi(nir, red, blue, use_cuda=False)
    gpu = sipi(nir, red, blue, use_cuda=True)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_nbr():
    nir = create_test_arr(arr1)
    swir = create_test_arr(arr2)

    result = nbr(nir, swir, use_cuda=False)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_nbr_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    swir = create_test_arr(arr2)

    cpu = nbr(nir, swir, use_cuda=False)
    gpu = nbr(nir, swir, use_cuda=True)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_nbr2():
    swir1 = create_test_arr(arr1)
    swir2 = create_test_arr(arr2)

    result = nbr2(swir1, swir2, use_cuda=False)

    assert result.dims == swir1.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == swir1.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_nbr2_cpu_equals_gpu():
    swir1 = create_test_arr(arr1)
    swir2 = create_test_arr(arr2)

    cpu = nbr2(swir1, swir2, use_cuda=False)
    gpu = nbr2(swir1, swir2, use_cuda=True)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_ndmi():
    nir = create_test_arr(arr1)
    swir1 = create_test_arr(arr2)

    result = ndmi(nir, swir1, use_cuda=False)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ndmi_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    swir = create_test_arr(arr2)

    cpu = ndmi(nir, swir, use_cuda=False)
    gpu = ndmi(nir, swir, use_cuda=True)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_ebbi():
    red = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    tir = create_test_arr(arr3)

    result = ebbi(red, swir, tir, use_cuda=False)

    assert result.dims == red.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == red.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ebbi_cpu_equals_gpu():
    red = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    tir = create_test_arr(arr3)

    cpu = ebbi(red, swir, tir, use_cuda=False)
    gpu = ebbi(red, swir, tir, use_cuda=True)
    assert np.isclose(cpu, gpu, equal_nan=True).all()