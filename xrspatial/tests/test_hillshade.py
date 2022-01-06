import pytest
import xarray as xr
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

import dask.array as da

from xrspatial import hillshade
from xrspatial.utils import doesnt_have_cuda
from ..gpu_rtx import has_rtx

from xrspatial.tests.general_checks import general_output_checks


elevation = np.asarray(
    [[1432.6542, 1432.4764, 1432.4764, 1432.1207, 1431.9429, np.nan],
     [1432.6542, 1432.6542, 1432.4764, 1432.2986, 1432.1207, np.nan],
     [1432.832, 1432.6542, 1432.4764, 1432.2986, 1432.1207, np.nan],
     [1432.832, 1432.6542, 1432.4764, 1432.4764, 1432.1207, np.nan],
     [1432.832, 1432.6542, 1432.6542, 1432.4764, 1432.2986, np.nan],
     [1432.832, 1432.6542, 1432.6542, 1432.4764, 1432.2986, np.nan],
     [1432.832, 1432.832, 1432.6542, 1432.4764, 1432.4764, np.nan]],
    dtype=np.float32)


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


def test_hillshade():
    """
    Assert Simple Hillshade transfer function
    """
    da_gaussian = xr.DataArray(data_gaussian)
    da_gaussian_shade = hillshade(da_gaussian, name='hillshade_agg')
    general_output_checks(da_gaussian, da_gaussian_shade)
    assert da_gaussian_shade.name == 'hillshade_agg'
    assert da_gaussian_shade.mean() > 0
    assert da_gaussian_shade[60, 60] > 0


def test_hillshade_cpu():
    attrs = {'res': (10.0, 10.0)}

    small_numpy_based_data_array = xr.DataArray(elevation, attrs=attrs)
    numpy_result = hillshade(small_numpy_based_data_array, name='numpy')

    dask_data = da.from_array(elevation, chunks=(3, 3))
    small_das_based_data_array = xr.DataArray(dask_data, attrs=attrs)
    dask_result = hillshade(small_das_based_data_array, name='dask')

    general_output_checks(small_das_based_data_array, dask_result)
    np.testing.assert_allclose(
        numpy_result.data, dask_result.data.compute(), equal_nan=True)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_hillshade_gpu_equals_cpu():

    import cupy

    small_da = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    cpu = hillshade(small_da, name='numpy_result')

    small_da_cupy = xr.DataArray(cupy.asarray(elevation),
                                 attrs={'res': (10.0, 10.0)})
    gpu = hillshade(small_da_cupy, name='cupy_result')

    general_output_checks(small_da_cupy, gpu)
    assert isinstance(gpu.data, cupy.ndarray)

    assert np.isclose(cpu.data, gpu.data.get(), equal_nan=True).all()


@pytest.mark.skipif(not has_rtx(), reason="RTX not available")
def test_hillshade_rtx_with_shadows():
    import cupy

    tall_gaussian = 400*data_gaussian
    cpu = hillshade(xr.DataArray(tall_gaussian))

    tall_gaussian = cupy.asarray(tall_gaussian)
    rtx = hillshade(xr.DataArray(tall_gaussian))
    rtx.data = cupy.asnumpy(rtx.data)

    assert cpu.shape == rtx.shape
    nhalf = cpu.shape[0] // 2

    # Quadrant nearest sun direction should be almost identical.
    quad_cpu = cpu.data[nhalf::, ::nhalf]
    quad_rtx = cpu.data[nhalf::, ::nhalf]
    assert_allclose(quad_cpu, quad_rtx, atol=0.03)

    # Opposite diagonal should be in shadow.
    diag_cpu = np.diagonal(cpu.data[::-1])[nhalf:]
    diag_rtx = np.diagonal(rtx.data[::-1])[nhalf:]
    assert_array_less(diag_rtx, diag_cpu + 1e-3)
