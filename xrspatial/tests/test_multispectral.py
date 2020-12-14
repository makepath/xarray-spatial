import pytest
import xarray as xr
import numpy as np
import xarray as xa

import dask.array as da

from xrspatial.utils import has_cuda
from xrspatial.utils import doesnt_have_cuda
from xrspatial.utils import is_dask_cupy

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


data_gaussian = _do_gaussian_array()


def create_test_arr(arr, backend='numpy'):

    y, x = arr.shape
    raster = xa.DataArray(arr, dims=['y', 'x'])

    if backend == 'numpy':
        raster['y'] = np.linspace(0, y, y)
        raster['x'] = np.linspace(0, x, x)
        return raster

    if has_cuda() and 'cupy' in back:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in back:
        raster.data = da.from_array(raster.data, chunks=(3, 3))

    return raster


# NDVI -------------
def test_ndvi_numpy_contains_valid_values():
    """
    Assert aspect transfer function
    """
    _x = np.mgrid[1:0:21j]
    a, b = np.meshgrid(_x, _x)
    red = a*b
    nir = (a*b)[::-1, ::-1]

    da_nir = xr.DataArray(nir, dims=['y', 'x'])
    da_red = xr.DataArray(red, dims=['y', 'x'])

    da_ndvi = ndvi(da_nir, da_red)

    assert da_ndvi.dims == da_nir.dims
    assert da_ndvi.attrs == da_nir.attrs
    for coord in da_nir.coords:
        assert np.all(da_nir[coord] == da_ndvi[coord])

    assert da_ndvi[0, 0] == -1
    assert da_ndvi[-1, -1] == 1
    assert da_ndvi[5, 10] == da_ndvi[10, 5] == -0.5
    assert da_ndvi[15, 10] == da_ndvi[10, 15] == 0.5


def test_ndvi_dask_equals_numpy():

    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    numpy_result = ndvi(nir, red)

    # dask 
    nir_dask = create_test_arr(arr1, backend='dask')
    red_dask = create_test_arr(arr2, backend='dask')

    test_result = ndvi(nir_dask, red_dask)
    assert isinstance(test_result.data, da.Array)

    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ndvi_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    numpy_result = ndvi(nir, red)

    # cupy
    nir_dask = create_test_arr(arr1, backend='cupy')
    red_dask = create_test_arr(arr2, backend='cupy')
    test_result = ndvi(nir_dask, red_dask)

    assert isinstance(test_result.data, cupy.ndarray)

    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ndvi_dask_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    numpy_result = ndvi(nir, red)

    # dask + cupy
    nir_dask = create_test_arr(arr1, backend='dask+cupy')
    red_dask = create_test_arr(arr2, backend='dask+cupy')
    test_result = ndvi(nir_dask, red_dask)

    assert is_dask_cupy(test_result)

    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()

# SAVI -------------

def test_savi_numpy_contains_valid_values():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)

    # savi should be same as ndvi at soil_factor=0
    result_savi = savi(nir, red, soil_factor=0.0)
    result_ndvi = ndvi(nir, red)

    assert np.isclose(result_savi.data, result_ndvi.data, equal_nan=True).all()
    assert result_savi.dims == nir.dims

    result_savi = savi(nir, red, soil_factor=1.0)
    assert isinstance(result_savi, xa.DataArray)
    assert result_savi.dims == nir.dims


def test_savi_dask_equals_numpy():
    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    numpy_result = ndvi(nir, red)

    # dask 
    nir_dask = create_test_arr(arr1, backend='dask')
    red_dask = create_test_arr(arr2, backend='dask')

    test_result = savi(nir_dask, red_dask)
    assert isinstance(test_result.data, da.Array)

    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()
    pass


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_savi_cupy_equals_numpy():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)

    # savi should be same as ndvi at soil_factor=0
    cpu = savi(nir, red, soil_factor=0.0)
    gpu = savi(nir, red, soil_factor=0.0)
    assert np.isclose(cpu, gpu, equal_nan=True).all()

def test_savi_dask_cupy_equals_numpy():
    pass

# AVRI -------------
# def test_avri_numpy_contains_valid_values():
# def test_avri_dask_equals_numpy():
# def test_avri_cupy_equals_numpy():
# def test_avri_dask_cupy_equals_numpy():

def test_avri():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    result = arvi(nir, red, blue)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_avri_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    cpu = arvi(nir, red, blue)
    gpu = arvi(nir, red, blue)
    assert np.isclose(cpu, gpu, equal_nan=True).all()

# EVI -------------
# def test_evi_numpy_contains_valid_values():
# def test_evi_dask_equals_numpy():
# def test_evi_cupy_equals_numpy():
# def test_evi_dask_cupy_equals_numpy():

def test_evi():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    result = evi(nir, red, blue)

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

    cpu = evi(nir, red, blue)
    gpu = evi(nir, red, blue)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


# GCI -------------
# def test_gci_numpy_contains_valid_values():
# def test_gci_dask_equals_numpy():
# def test_gci_cupy_equals_numpy():
# def test_gci_dask_cupy_equals_numpy():

def test_gci():
    nir = create_test_arr(arr1)
    green = create_test_arr(arr2)

    result = gci(nir, green)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_gci_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    green = create_test_arr(arr2)

    cpu = gci(nir, green)
    gpu = gci(nir, green)
    assert np.isclose(cpu, gpu, equal_nan=True).all()

# SIPI -------------
# def test_sipi_numpy_contains_valid_values():
# def test_sipi_dask_equals_numpy():
# def test_sipi_cupy_equals_numpy():
# def test_sipi_dask_cupy_equals_numpy():

def test_sipi():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    result = sipi(nir, red, blue)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_sipi_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    cpu = sipi(nir, red, blue)
    gpu = sipi(nir, red, blue)
    assert np.isclose(cpu, gpu, equal_nan=True).all()

# NBR -------------
# def test_nbr_numpy_contains_valid_values():
# def test_nbr_dask_equals_numpy():
# def test_nbr_cupy_equals_numpy():
# def test_nbr_dask_cupy_equals_numpy():

def test_nbr():
    nir = create_test_arr(arr1)
    swir = create_test_arr(arr2)

    result = nbr(nir, swir)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_nbr_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    swir = create_test_arr(arr2)

    cpu = nbr(nir, swir)
    gpu = nbr(nir, swir)
    assert np.isclose(cpu, gpu, equal_nan=True).all()

# NBR2 -------------
# def test_nbr2_numpy_contains_valid_values():
# def test_nbr2_dask_equals_numpy():
# def test_nbr2_cupy_equals_numpy():
# def test_nbr2_dask_cupy_equals_numpy():

def test_nbr2():
    swir1 = create_test_arr(arr1)
    swir2 = create_test_arr(arr2)

    result = nbr2(swir1, swir2)

    assert result.dims == swir1.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == swir1.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_nbr2_cpu_equals_gpu():
    swir1 = create_test_arr(arr1)
    swir2 = create_test_arr(arr2)

    cpu = nbr2(swir1, swir2)
    gpu = nbr2(swir1, swir2)
    assert np.isclose(cpu, gpu, equal_nan=True).all()

# NDMI -------------
# def test_ndmi_numpy_contains_valid_values():
# def test_ndmi_dask_equals_numpy():
# def test_ndmi_cupy_equals_numpy():
# def test_ndmi_dask_cupy_equals_numpy():

def test_ndmi():
    nir = create_test_arr(arr1)
    swir1 = create_test_arr(arr2)

    result = ndmi(nir, swir1)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ndmi_cpu_equals_gpu():
    nir = create_test_arr(arr1)
    swir = create_test_arr(arr2)

    cpu = ndmi(nir, swir)
    gpu = ndmi(nir, swir)
    assert np.isclose(cpu, gpu, equal_nan=True).all()

# EBBI -------------
# def test_ebbi_numpy_contains_valid_values():
# def test_ebbi_dask_equals_numpy():
# def test_ebbi_cupy_equals_numpy():
# def test_ebbi_dask_cupy_equals_numpy():

def test_ebbi():
    red = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    tir = create_test_arr(arr3)

    result = ebbi(red, swir, tir)

    assert result.dims == red.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == red.dims


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ebbi_cpu_equals_gpu():
    red = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    tir = create_test_arr(arr3)

    cpu = ebbi(red, swir, tir)
    gpu = ebbi(red, swir, tir)
    assert np.isclose(cpu, gpu, equal_nan=True).all()
