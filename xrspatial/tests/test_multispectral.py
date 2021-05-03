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
from xrspatial.multispectral import true_color


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

    if has_cuda() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend:
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
    nir_cupy = create_test_arr(arr1, backend='cupy')
    red_cupy = create_test_arr(arr2, backend='cupy')
    test_result = ndvi(nir_cupy, red_cupy)

    assert isinstance(test_result.data, cupy.ndarray)
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ndvi_dask_cupy_equals_numpy():

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
def test_savi_numpy():
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
    numpy_result = savi(nir, red)

    # dask
    nir_dask = create_test_arr(arr1, backend='dask')
    red_dask = create_test_arr(arr2, backend='dask')
    test_result = savi(nir_dask, red_dask)

    assert isinstance(test_result.data, da.Array)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_savi_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    numpy_result = savi(nir, red)

    # dask + cupy
    nir_cupy = create_test_arr(arr1, backend='cupy')
    red_cupy = create_test_arr(arr2, backend='cupy')
    test_result = savi(nir_cupy, red_cupy)

    assert isinstance(test_result.data, cupy.ndarray)
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_savi_dask_cupy_equals_numpy():
    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    numpy_result = savi(nir, red)

    # dask + cupy
    nir_dask_cupy = create_test_arr(arr1, backend='dask+cupy')
    red_dask_cupy = create_test_arr(arr2, backend='dask+cupy')
    test_result = savi(nir_dask_cupy, red_dask_cupy)

    assert is_dask_cupy(test_result)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


# arvi -------------
def test_arvi_numpy():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    result = arvi(nir, red, blue)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


def test_arvi_dask_equals_numpy():

    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    numpy_result = arvi(nir, red, blue)

    # dask
    nir_dask = create_test_arr(arr1, backend='dask')
    red_dask = create_test_arr(arr2, backend='dask')
    blue_dask = create_test_arr(arr3, backend='dask')
    test_result = arvi(nir_dask, red_dask, blue_dask)

    assert isinstance(test_result.data, da.Array)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_arvi_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    numpy_result = arvi(nir, red, blue)

    # cupy
    nir_cupy = create_test_arr(arr1, backend='cupy')
    red_cupy = create_test_arr(arr2, backend='cupy')
    blue_cupy = create_test_arr(arr3, backend='cupy')
    test_result = arvi(nir_cupy, red_cupy, blue_cupy)

    assert isinstance(test_result.data, cupy.ndarray)
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_arvi_dask_cupy_equals_numpy():
    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    numpy_result = arvi(nir, red, blue)

    # dask + cupy
    nir_dask_cupy = create_test_arr(arr1, backend='dask+cupy')
    red_dask_cupy = create_test_arr(arr2, backend='dask+cupy')
    blue_dask_cupy = create_test_arr(arr3, backend='dask+cupy')
    test_result = arvi(nir_dask_cupy, red_dask_cupy, blue_dask_cupy)

    assert is_dask_cupy(test_result)

    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


# EVI -------------
def test_evi_numpy():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    gain = 2.5
    c1 = 6.0
    c2 = 7.5
    soil_factor = 1.0

    result = evi(nir, red, blue, c1, c2, soil_factor, gain)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


def test_evi_dask_equals_numpy():

    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    numpy_result = evi(nir, red, blue)

    # dask
    nir_dask = create_test_arr(arr1, backend='dask')
    red_dask = create_test_arr(arr2, backend='dask')
    blue_dask = create_test_arr(arr3, backend='dask')
    test_result = evi(nir_dask, red_dask, blue_dask)

    assert isinstance(test_result.data, da.Array)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_evi_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    numpy_result = evi(nir, red, blue)

    # cupy
    nir_cupy = create_test_arr(arr1, backend='cupy')
    red_cupy = create_test_arr(arr2, backend='cupy')
    blue_cupy = create_test_arr(arr3, backend='cupy')
    test_result = evi(nir_cupy, red_cupy, blue_cupy)

    assert isinstance(test_result.data, cupy.ndarray)
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_evi_dask_cupy_equals_numpy():
    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    numpy_result = evi(nir, red, blue)

    # dask + cupy
    nir_dask_cupy = create_test_arr(arr1, backend='dask+cupy')
    red_dask_cupy = create_test_arr(arr2, backend='dask+cupy')
    blue_dask_cupy = create_test_arr(arr3, backend='dask+cupy')
    test_result = evi(nir_dask_cupy, red_dask_cupy, blue_dask_cupy)

    assert is_dask_cupy(test_result)

    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


# GCI -------------
def test_gci_numpy():
    nir = create_test_arr(arr1)
    green = create_test_arr(arr2)

    result = gci(nir, green)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


def test_gci_dask_equals_numpy():
    # vanilla numpy version
    nir = create_test_arr(arr1)
    green = create_test_arr(arr2)
    numpy_result = gci(nir, green)

    # dask
    nir_dask = create_test_arr(arr1, backend='dask')
    green_dask = create_test_arr(arr2, backend='dask')
    test_result = gci(nir_dask, green_dask)

    assert isinstance(test_result.data, da.Array)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Dgcice not Available")
def test_gci_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    nir = create_test_arr(arr1)
    green = create_test_arr(arr2)
    numpy_result = gci(nir, green)

    # cupy
    nir_cupy = create_test_arr(arr1, backend='cupy')
    green_cupy = create_test_arr(arr2, backend='cupy')
    test_result = gci(nir_cupy, green_cupy)

    assert isinstance(test_result.data, cupy.ndarray)
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Dgcice not Available")
def test_gci_dask_cupy_equals_numpy():
    # vanilla numpy version
    nir = create_test_arr(arr1)
    green = create_test_arr(arr2)
    numpy_result = gci(nir, green)

    # dask + cupy
    nir_dask_cupy = create_test_arr(arr1, backend='dask+cupy')
    green_dask_cupy = create_test_arr(arr2, backend='dask+cupy')
    test_result = gci(nir_dask_cupy, green_dask_cupy)

    assert is_dask_cupy(test_result)

    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


# SIPI -------------
def test_sipi_numpy():
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)

    result = sipi(nir, red, blue)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


def test_sipi_dask_equals_numpy():

    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    numpy_result = sipi(nir, red, blue)

    # dask
    nir_dask = create_test_arr(arr1, backend='dask')
    red_dask = create_test_arr(arr2, backend='dask')
    blue_dask = create_test_arr(arr3, backend='dask')
    test_result = sipi(nir_dask, red_dask, blue_dask)

    assert isinstance(test_result.data, da.Array)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_sipi_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    numpy_result = sipi(nir, red, blue)

    # cupy
    nir_dask = create_test_arr(arr1, backend='cupy')
    red_dask = create_test_arr(arr2, backend='cupy')
    blue_dask = create_test_arr(arr3, backend='cupy')
    test_result = sipi(nir_dask, red_dask, blue_dask)

    assert isinstance(test_result.data, cupy.ndarray)
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_sipi_dask_cupy_equals_numpy():
    # vanilla numpy version
    nir = create_test_arr(arr1)
    red = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    numpy_result = sipi(nir, red, blue)

    # dask + cupy
    nir_dask_cupy = create_test_arr(arr1, backend='dask+cupy')
    red_dask_cupy = create_test_arr(arr2, backend='dask+cupy')
    blue_dask_cupy = create_test_arr(arr3, backend='dask+cupy')
    test_result = sipi(nir_dask_cupy, red_dask_cupy, blue_dask_cupy)

    assert is_dask_cupy(test_result)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


# NBR -------------
def test_nbr_numpy():
    nir = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    result = nbr(nir, swir)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


def test_nbr_dask_equals_numpy():

    # vanilla numpy version
    nir = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    numpy_result = nbr(nir, swir)

    # dask
    nir_dask = create_test_arr(arr1, backend='dask')
    swir_dask = create_test_arr(arr2, backend='dask')
    test_result = nbr(nir_dask, swir_dask)

    assert isinstance(test_result.data, da.Array)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_nbr_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    nir = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    numpy_result = nbr(nir, swir)

    # cupy
    nir_cupy = create_test_arr(arr1, backend='cupy')
    swir_cupy = create_test_arr(arr2, backend='cupy')
    test_result = nbr(nir_cupy, swir_cupy)

    assert isinstance(test_result.data, cupy.ndarray)
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_nbr_dask_cupy_equals_numpy():
    # vanilla numpy version
    nir = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    numpy_result = nbr(nir, swir)

    # dask + cupy
    nir_dask_cupy = create_test_arr(arr1, backend='dask+cupy')
    swir_dask_cupy = create_test_arr(arr2, backend='dask+cupy')
    test_result = nbr(nir_dask_cupy, swir_dask_cupy)

    assert is_dask_cupy(test_result)

    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


# NBR2 -------------
def test_nbr2_numpy():
    swir1 = create_test_arr(arr1)
    swir2 = create_test_arr(arr2)

    result = nbr2(swir1, swir2)

    assert result.dims == swir1.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == swir1.dims


def test_nbr2_dask_equals_numpy():

    # vanilla numpy version
    swir1 = create_test_arr(arr1)
    swir2 = create_test_arr(arr2)
    numpy_result = nbr2(swir1, swir2)

    # dask
    swir1_dask = create_test_arr(arr1, backend='dask')
    swir2_dask = create_test_arr(arr2, backend='dask')
    test_result = nbr2(swir1_dask, swir2_dask)

    assert isinstance(test_result.data, da.Array)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Dnbr2ce not Available")
def test_nbr2_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    swir1 = create_test_arr(arr1)
    swir2 = create_test_arr(arr2)
    numpy_result = nbr2(swir1, swir2)

    # cupy
    swir1_cupy = create_test_arr(arr1, backend='cupy')
    swir2_cupy = create_test_arr(arr2, backend='cupy')
    test_result = nbr2(swir1_cupy, swir2_cupy)

    assert isinstance(test_result.data, cupy.ndarray)
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Dnbr2ce not Available")
def test_nbr2_dask_cupy_equals_numpy():
    # vanilla numpy version
    swir1 = create_test_arr(arr1)
    swir2 = create_test_arr(arr2)
    numpy_result = nbr2(swir1, swir2)

    # dask + cupy
    swir1_dask_cupy = create_test_arr(arr1, backend='dask+cupy')
    swir2_dask_cupy = create_test_arr(arr2, backend='dask+cupy')
    test_result = nbr2(swir1_dask_cupy, swir2_dask_cupy)

    assert is_dask_cupy(test_result)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


# NDMI -------------
def test_ndmi_numpy():
    nir = create_test_arr(arr1)
    swir1 = create_test_arr(arr2)

    result = ndmi(nir, swir1)

    assert result.dims == nir.dims
    assert isinstance(result, xa.DataArray)
    assert result.dims == nir.dims


def test_ndmi_dask_equals_numpy():

    # vanilla numpy version
    nir = create_test_arr(arr1)
    swir1 = create_test_arr(arr2)
    numpy_result = ndmi(nir, swir1)

    # dask
    nir_dask = create_test_arr(arr1, backend='dask')
    swir1_dask = create_test_arr(arr2, backend='dask')
    test_result = ndmi(nir_dask, swir1_dask)

    assert isinstance(test_result.data, da.Array)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ndmi_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    nir = create_test_arr(arr1)
    swir1 = create_test_arr(arr2)
    numpy_result = ndmi(nir, swir1)

    # cupy
    nir_cupy = create_test_arr(arr1, backend='cupy')
    swir1_cupy = create_test_arr(arr2, backend='cupy')
    test_result = ndmi(nir_cupy, swir1_cupy)

    assert isinstance(test_result.data, cupy.ndarray)
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ndmi_dask_cupy_equals_numpy():
    # vanilla numpy version
    nir = create_test_arr(arr1)
    swir1 = create_test_arr(arr2)
    numpy_result = ndmi(nir, swir1)

    # dask + cupy
    nir_dask_cupy = create_test_arr(arr1, backend='dask+cupy')
    swir1_dask_cupy = create_test_arr(arr2, backend='dask+cupy')
    test_result = ndmi(nir_dask_cupy, swir1_dask_cupy)

    assert is_dask_cupy(test_result)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


# EBBI -------------
def test_ebbi_numpy():
    red = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    tir = create_test_arr(arr3)
    numpy_result = ebbi(red, swir, tir)

    assert numpy_result.dims == red.dims
    assert isinstance(numpy_result, xa.DataArray)
    assert numpy_result.dims == red.dims


def test_ebbi_dask_equals_numpy():

    # vanilla numpy version
    red = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    tir = create_test_arr(arr3)
    numpy_result = ebbi(red, swir, tir)

    # dask
    red_dask = create_test_arr(arr1, backend='dask')
    swir_dask = create_test_arr(arr2, backend='dask')
    tir_dask = create_test_arr(arr3, backend='dask')
    test_result = ebbi(red_dask, swir_dask, tir_dask)

    assert isinstance(test_result.data, da.Array)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ebbi_cupy_equals_numpy():

    import cupy

    # vanilla numpy version
    red = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    tir = create_test_arr(arr3)
    numpy_result = ebbi(red, swir, tir)

    # cupy
    red_dask = create_test_arr(arr1, backend='cupy')
    swir_dask = create_test_arr(arr2, backend='cupy')
    tir_dask = create_test_arr(arr3, backend='cupy')
    test_result = ebbi(red_dask, swir_dask, tir_dask)

    assert isinstance(test_result.data, cupy.ndarray)
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_ebbi_dask_cupy_equals_numpy():
    # vanilla numpy version
    red = create_test_arr(arr1)
    swir = create_test_arr(arr2)
    tir = create_test_arr(arr3)
    numpy_result = ebbi(red, swir, tir)

    # dask + cupy
    red_dask_cupy = create_test_arr(arr1, backend='dask+cupy')
    swir_dask_cupy = create_test_arr(arr2, backend='dask+cupy')
    tir_dask_cupy = create_test_arr(arr3, backend='dask+cupy')
    test_result = ebbi(red_dask_cupy, swir_dask_cupy, tir_dask_cupy)

    assert is_dask_cupy(test_result)
    test_result.data = test_result.data.compute()
    assert np.isclose(numpy_result, test_result, equal_nan=True).all()


def test_true_color_cpu():
    # vanilla numpy version
    red = create_test_arr(arr1)
    green = create_test_arr(arr2)
    blue = create_test_arr(arr3)
    numpy_result = true_color(red, green, blue)

    # dask
    red_dask = create_test_arr(arr1, backend='dask')
    green_dask = create_test_arr(arr2, backend='dask')
    blue_dask = create_test_arr(arr3, backend='dask')
    dask_result = true_color(red_dask, green_dask, blue_dask)

    # TODO: test output metadata: dims, coords, attrs
    assert isinstance(numpy_result, xa.DataArray)
    assert isinstance(dask_result.data, da.Array)
    dask_result.data = dask_result.data.compute()
    assert np.isclose(numpy_result, dask_result, equal_nan=True).all()
