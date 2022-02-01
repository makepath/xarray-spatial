import pytest
import xarray as xr
import numpy as np

import dask.array as da

from xrspatial.utils import doesnt_have_cuda
from xrspatial.utils import ngjit

from xrspatial import mean
from xrspatial.focal import hotspots, apply, focal_stats
from xrspatial.convolution import (
    convolve_2d, circle_kernel, annulus_kernel
)

from xrspatial.tests.general_checks import create_test_raster
from xrspatial.tests.general_checks import general_output_checks


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


def test_mean_transfer_function_cpu():
    # numpy case
    numpy_agg = xr.DataArray(data_random)
    numpy_mean = mean(numpy_agg)
    general_output_checks(numpy_agg, numpy_mean)

    # dask + numpy case
    dask_numpy_agg = xr.DataArray(da.from_array(data_random, chunks=(3, 3)))
    dask_numpy_mean = mean(dask_numpy_agg)
    general_output_checks(dask_numpy_agg, dask_numpy_mean)

    # both output same results
    np.testing.assert_allclose(
        numpy_mean.data, dask_numpy_mean.data.compute(), equal_nan=True
    )


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_mean_transfer_function_gpu_equals_cpu():

    import cupy

    # cupy case
    cupy_agg = xr.DataArray(cupy.asarray(data_random))
    cupy_mean = mean(cupy_agg)
    general_output_checks(cupy_agg, cupy_mean)

    # numpy case
    numpy_agg = xr.DataArray(data_random)
    numpy_mean = mean(numpy_agg)

    np.testing.assert_allclose(
        numpy_mean.data, cupy_mean.data.get(), equal_nan=True)

    # dask + cupy case not implemented
    dask_cupy_agg = xr.DataArray(
        da.from_array(cupy.asarray(data_random), chunks=(3, 3))
    )
    with pytest.raises(NotImplementedError) as e_info:
        mean(dask_cupy_agg)
        assert e_info


@pytest.fixture
def convolve_2d_data():
    data = np.array([
        [0., 1., 1., 1., 1., 1.],
        [1., 0., 1., 1., 1., 1.],
        [1., 1., 0., 1., 1., 1.],
        [1., 1., 1., np.nan, 1., 1.],
        [1., 1., 1., 1., 0., 1.],
        [1., 1., 1., 1., 1., 0.]
    ])
    return data


@pytest.fixture
def kernel_circle_1_1_1():
    result = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    return result


@pytest.fixture
def kernel_annulus_2_2_2_1():
    result = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    return result


@pytest.fixture
def convolution_kernel_circle_1_1_1():
    expected_result = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., 3., 5., 5., np.nan],
        [np.nan, 3., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 5., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 5., np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    return expected_result


@pytest.fixture
def convolution_kernel_annulus_2_2_1():
    expected_result = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., 2., 4., 4., np.nan],
        [np.nan, 2., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    return expected_result


def test_kernel(kernel_circle_1_1_1, kernel_annulus_2_2_2_1):
    kernel_circle = circle_kernel(1, 1, 1)
    assert isinstance(kernel_circle, np.ndarray)
    np.testing.assert_allclose(kernel_circle, kernel_circle_1_1_1, equal_nan=True)

    kernel_annulus = annulus_kernel(2, 2, 2, 1)
    assert isinstance(kernel_annulus, np.ndarray)
    np.testing.assert_allclose(kernel_annulus, kernel_annulus_2_2_2_1, equal_nan=True)


def test_convolution_numpy(
    convolve_2d_data,
    kernel_circle_1_1_1,
    convolution_kernel_circle_1_1_1,
    kernel_annulus_2_2_2_1,
    convolution_kernel_annulus_2_2_1
):
    kernel_custom = np.ones((1, 1))
    result_kernel_custom = convolve_2d(convolve_2d_data, kernel_custom)
    assert isinstance(result_kernel_custom, np.ndarray)
    # kernel is [[1]], thus the result equals input data
    np.testing.assert_allclose(result_kernel_custom, convolve_2d_data, equal_nan=True)

    result_kernel_circle = convolve_2d(convolve_2d_data, kernel_circle_1_1_1)
    assert isinstance(result_kernel_circle, np.ndarray)
    np.testing.assert_allclose(
        result_kernel_circle, convolution_kernel_circle_1_1_1, equal_nan=True
    )

    result_kernel_annulus = convolve_2d(convolve_2d_data, kernel_annulus_2_2_2_1)
    assert isinstance(result_kernel_annulus, np.ndarray)
    np.testing.assert_allclose(
        result_kernel_annulus, convolution_kernel_annulus_2_2_1, equal_nan=True
    )


def test_convolution_dask_numpy(
    convolve_2d_data,
    kernel_circle_1_1_1,
    convolution_kernel_circle_1_1_1,
    kernel_annulus_2_2_2_1,
    convolution_kernel_annulus_2_2_1
):
    dask_data = da.from_array(convolve_2d_data, chunks=(3, 3))
    kernel_custom = np.ones((1, 1))
    result_kernel_custom = convolve_2d(dask_data, kernel_custom)
    assert isinstance(result_kernel_custom, da.Array)
    # kernel is [[1]], thus the result equals input data
    np.testing.assert_allclose(result_kernel_custom.compute(), convolve_2d_data, equal_nan=True)

    result_kernel_circle = convolve_2d(dask_data, kernel_circle_1_1_1)
    assert isinstance(result_kernel_circle, da.Array)
    np.testing.assert_allclose(
        result_kernel_circle.compute(), convolution_kernel_circle_1_1_1, equal_nan=True
    )

    result_kernel_annulus = convolve_2d(dask_data, kernel_annulus_2_2_2_1)
    assert isinstance(result_kernel_annulus, da.Array)
    np.testing.assert_allclose(
        result_kernel_annulus.compute(), convolution_kernel_annulus_2_2_1, equal_nan=True
    )


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_2d_convolution_gpu(
    convolve_2d_data,
    kernel_circle_1_1_1,
    convolution_kernel_circle_1_1_1,
    kernel_annulus_2_2_2_1,
    convolution_kernel_annulus_2_2_1
):
    import cupy
    cupy_data = cupy.asarray(convolve_2d_data)

    kernel_custom = np.ones((1, 1))
    result_kernel_custom = convolve_2d(cupy_data, kernel_custom)
    assert isinstance(result_kernel_custom, cupy.ndarray)
    # kernel is [[1]], thus the result equals input data
    np.testing.assert_allclose(result_kernel_custom.get(), convolve_2d_data, equal_nan=True)

    result_kernel_circle = convolve_2d(cupy_data, kernel_circle_1_1_1)
    assert isinstance(result_kernel_circle, cupy.ndarray)
    np.testing.assert_allclose(
        result_kernel_circle.get(), convolution_kernel_circle_1_1_1, equal_nan=True
    )

    result_kernel_annulus = convolve_2d(cupy_data, kernel_annulus_2_2_2_1)
    assert isinstance(result_kernel_annulus, cupy.ndarray)
    np.testing.assert_allclose(
        result_kernel_annulus.get(), convolution_kernel_annulus_2_2_1, equal_nan=True
    )

    # dask + cupy case not implemented
    dask_cupy_agg = xr.DataArray(
        da.from_array(cupy.asarray(convolve_2d_data), chunks=(3, 3))
    )
    with pytest.raises(NotImplementedError) as e_info:
        convolve_2d(dask_cupy_agg.data, kernel_custom)
        assert e_info


@pytest.fixture
def data_apply():
    data = np.array([[0, 1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10, 11],
                     [12, 13, 14, 15, 16, 17],
                     [18, 19, 20, 21, 22, 23]])
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    expected_result = np.zeros_like(data)
    return data, kernel, expected_result


def func_zero(x):
    return 0


@ngjit
def func_zero_cpu(x):
    return 0


def test_apply_numpy(data_apply):
    data, kernel, expected_result = data_apply
    numpy_agg = create_test_raster(data)
    numpy_apply = apply(numpy_agg, kernel, func_zero_cpu)
    general_output_checks(numpy_agg, numpy_apply, expected_result)


def test_apply_dask_numpy(data_apply):
    data, kernel, expected_result = data_apply
    dask_numpy_agg = create_test_raster(data, backend='dask')
    dask_numpy_apply = apply(dask_numpy_agg, kernel, func_zero_cpu)
    general_output_checks(dask_numpy_agg, dask_numpy_apply, expected_result)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_apply_gpu(data_apply):
    data, kernel, expected_result = data_apply
    # cupy case
    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_apply = apply(cupy_agg, kernel, func_zero)
    general_output_checks(cupy_agg, cupy_apply, expected_result)

    # dask + cupy case not implemented
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy')
    with pytest.raises(NotImplementedError) as e_info:
        apply(dask_cupy_agg, kernel, func_zero)
        assert e_info


@pytest.fixture
def data_focal_stats():
    data = np.arange(16).reshape(4, 4)
    cellsize = (1, 1)
    kernel = circle_kernel(*cellsize, 1.5)
    expected_result = np.asarray([
        # mean
        [[1.66666667, 2., 3., 4.],
         [4.25, 5., 6., 6.75],
         [8.25, 9., 10., 10.75],
         [11., 12., 13., 13.33333333]],
        # max
        [[4., 5., 6., 7.],
         [8., 9., 10., 11.],
         [12., 13., 14., 15.],
         [13., 14., 15., 15.]],
        # min
        [[0., 0., 1., 2.],
         [0., 1., 2., 3.],
         [4., 5., 6., 7.],
         [8., 9., 10., 11.]],
        # range
        [[4., 5., 5., 5.],
         [8., 8., 8., 8.],
         [8., 8., 8., 8.],
         [5., 5., 5., 4.]],
        # std
        [[1.69967317, 1.87082869, 1.87082869, 2.1602469],
         [2.86138079, 2.60768096, 2.60768096, 2.86138079],
         [2.86138079, 2.60768096, 2.60768096, 2.86138079],
         [2.1602469, 1.87082869, 1.87082869, 1.69967317]],
        # var
        [[2.88888889, 3.5, 3.5, 4.66666667],
         [8.1875, 6.8, 6.8, 8.1875],
         [8.1875, 6.8, 6.8, 8.1875],
         [4.66666667, 3.5, 3.5, 2.88888889]],
        # sum
        [[5., 8., 12., 12.],
         [17., 25., 30., 27.],
         [33., 45., 50., 43.],
         [33., 48., 52., 40.]]
    ])
    return data, kernel, expected_result


def test_focal_stats_numpy(data_focal_stats):
    data, kernel, expected_result = data_focal_stats
    numpy_agg = create_test_raster(data)
    numpy_focalstats = focal_stats(numpy_agg, kernel)
    general_output_checks(
        numpy_agg, numpy_focalstats, verify_attrs=False, expected_results=expected_result
    )
    assert numpy_focalstats.ndim == 3


def test_focal_stats_dask_numpy(data_focal_stats):
    data, kernel, expected_result = data_focal_stats
    dask_numpy_agg = create_test_raster(data, backend='dask')
    dask_numpy_focalstats = focal_stats(dask_numpy_agg, kernel)
    general_output_checks(
        dask_numpy_agg, dask_numpy_focalstats, verify_attrs=False, expected_results=expected_result
    )


@pytest.fixture
def data_hotspots():
    data = np.asarray([
        [np.nan, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 10000., 10000., 10000., 0., 0., 0., 0., 0., 0.],
        [0., 10000., 10000., 10000., 0., 0., 0., 0., 0., 0.],
        [0., 10000., 10000., 10000., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., np.nan, 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., np.nan, 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., np.nan, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., -10000., -10000., -10000.],
        [0., 0., 0., 0., 0., 0., 0., -10000., -10000., -10000.],
        [0., 0., 0., 0., 0., 0., 0., -10000., -10000., -10000.]
    ])
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    expected_result = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 90, 0, 0, 0, 0, 0, 0, 0],
        [0, 90, 95, 90, 0, 0, 0, 0, 0, 0],
        [0, 0, 90, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -90, 0],
        [0, 0, 0, 0, 0, 0, 0, -90, -95, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int8)

    return data, kernel, expected_result


def test_hotspots_numpy(data_hotspots):
    data, kernel, expected_result = data_hotspots
    numpy_agg = create_test_raster(data)
    numpy_hotspots = hotspots(numpy_agg, kernel)
    general_output_checks(numpy_agg, numpy_hotspots, expected_result)


def test_hotspots_dask_numpy(data_hotspots):
    data, kernel, expected_result = data_hotspots
    dask_numpy_agg = create_test_raster(data, backend='dask')
    dask_numpy_hotspots = hotspots(dask_numpy_agg, kernel)
    general_output_checks(dask_numpy_agg, dask_numpy_hotspots, expected_result)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_hotspot_gpu(data_hotspots):
    data, kernel, expected_result = data_hotspots
    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_hotspots = hotspots(cupy_agg, kernel)
    general_output_checks(cupy_agg, cupy_hotspots, expected_result)

    # dask + cupy case not implemented
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy')
    with pytest.raises(NotImplementedError) as e_info:
        hotspots(dask_cupy_agg, kernel)
        assert e_info
