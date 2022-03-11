import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_less

from xrspatial import hillshade
from xrspatial.tests.general_checks import (assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_numpy, create_test_raster,
                                            cuda_and_cupy_available, general_output_checks)

from ..gpu_rtx import has_rtx


@pytest.fixture
def data_gaussian():
    _x = np.linspace(0, 50, 101)
    _y = _x.copy()
    _mean = 25
    _sdev = 5
    X, Y = np.meshgrid(_x, _y, sparse=True)
    x_fac = -np.power(X-_mean, 2)
    y_fac = -np.power(Y-_mean, 2)
    gaussian = np.exp((x_fac+y_fac)/(2*_sdev**2)) / (2.5*_sdev)
    return gaussian


def test_hillshade(data_gaussian):
    """
    Assert Simple Hillshade transfer function
    """
    da_gaussian = xr.DataArray(data_gaussian)
    da_gaussian_shade = hillshade(da_gaussian, name='hillshade_agg')
    general_output_checks(da_gaussian, da_gaussian_shade)
    assert da_gaussian_shade.name == 'hillshade_agg'
    assert da_gaussian_shade.mean() > 0
    assert da_gaussian_shade[60, 60] > 0


@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_hillshade_numpy_equals_dask_numpy(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, hillshade)


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.float32, np.float64])
def test_hillshade_gpu_equals_cpu(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    cupy_agg = create_test_raster(random_data, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, hillshade, rtol=1e-6)


@pytest.mark.skipif(not has_rtx(), reason="RTX not available")
def test_hillshade_rtx_with_shadows(data_gaussian):
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
