import pytest
import xarray as xr
import numpy as np

import dask.array as da

from xrspatial.utils import doesnt_have_cuda
from xrspatial.utils import ngjit

from xrspatial import mean
from xrspatial.focal import hotspots, apply, focal_stats
from xrspatial.convolution import (
    convolve_2d, calc_cellsize, circle_kernel, annulus_kernel
)

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


convolve_2d_data = np.array([[0., 1., 1., 1., 1., 1.],
                             [1., 0., 1., 1., 1., 1.],
                             [1., 1., 0., 1., 1., 1.],
                             [1., 1., 1., np.nan, 1., 1.],
                             [1., 1., 1., 1., 0., 1.],
                             [1., 1., 1., 1., 1., 0.]])


def test_kernel():
    data = convolve_2d_data
    m, n = data.shape
    agg = xr.DataArray(data, dims=['y', 'x'])
    agg['x'] = np.linspace(0, n, n)
    agg['y'] = np.linspace(0, m, m)

    cellsize_x, cellsize_y = calc_cellsize(agg)

    kernel1 = circle_kernel(cellsize_x, cellsize_y, 2)
    expected_kernel1 = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])
    assert isinstance(kernel1, np.ndarray)
    np.testing.assert_allclose(kernel1, expected_kernel1, equal_nan=True)

    kernel2 = annulus_kernel(cellsize_x, cellsize_y, 2, 0.5)
    expected_kernel2 = np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])
    assert isinstance(kernel2, np.ndarray)
    np.testing.assert_allclose(kernel2, expected_kernel2, equal_nan=True)


def test_convolution():
    data = convolve_2d_data
    dask_data = da.from_array(data, chunks=(3, 3))

    kernel1 = np.ones((1, 1))
    numpy_output_1 = convolve_2d(data, kernel1)
    expected_output_1 = np.array([[0., 1., 1., 1., 1., 1.],
                                  [1., 0., 1., 1., 1., 1.],
                                  [1., 1., 0., 1., 1., 1.],
                                  [1., 1., 1., np.nan, 1., 1.],
                                  [1., 1., 1., 1., 0., 1.],
                                  [1., 1., 1., 1., 1., 0.]])
    assert isinstance(numpy_output_1, np.ndarray)
    np.testing.assert_allclose(
        numpy_output_1, expected_output_1, equal_nan=True)

    dask_output_1 = convolve_2d(dask_data, kernel1)
    assert isinstance(dask_output_1, da.Array)
    np.testing.assert_allclose(
        dask_output_1.compute(), expected_output_1, equal_nan=True
    )

    kernel2 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    numpy_output_2 = convolve_2d(data, kernel2)
    expected_output_2 = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., 3., 5., 5., np.nan],
        [np.nan, 3., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 5., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 5., np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    # kernel2 is of 3x3, thus the border edge is 1 cell long.
    # currently, ignoring border edge (i.e values in edges are all nans)
    assert isinstance(numpy_output_2, np.ndarray)
    np.testing.assert_allclose(
        numpy_output_2, expected_output_2, equal_nan=True
    )

    dask_output_2 = convolve_2d(dask_data, kernel2)
    assert isinstance(dask_output_2, da.Array)
    np.testing.assert_allclose(
        dask_output_2.compute(), expected_output_2, equal_nan=True
    )

    kernel3 = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
    numpy_output_3 = convolve_2d(data, kernel3)
    expected_output_3 = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., 2., 4., 4., np.nan],
        [np.nan, 2., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    # kernel3 is of 3x3, thus the border edge is 1 cell long.
    # currently, values in edges are all nans
    assert isinstance(numpy_output_3, np.ndarray)
    np.testing.assert_allclose(
        numpy_output_3, expected_output_3, equal_nan=True)

    dask_output_3 = convolve_2d(dask_data, kernel3)
    assert isinstance(dask_output_3, da.Array)
    np.testing.assert_allclose(
        dask_output_3.compute(), expected_output_3, equal_nan=True
    )


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_2d_convolution_gpu_equals_cpu():

    import cupy

    data = convolve_2d_data
    numpy_agg = xr.DataArray(data)
    cupy_agg = xr.DataArray(cupy.asarray(data))

    kernel1 = np.ones((1, 1))
    output_numpy1 = convolve_2d(numpy_agg.data, kernel1)
    output_cupy1 = convolve_2d(cupy_agg.data, kernel1)
    assert isinstance(output_cupy1, cupy.ndarray)
    np.testing.assert_allclose(
        output_numpy1, output_cupy1.get(), equal_nan=True)

    kernel2 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    output_numpy2 = convolve_2d(numpy_agg.data, kernel2)
    output_cupy2 = convolve_2d(cupy_agg.data, kernel2)
    assert isinstance(output_cupy2, cupy.ndarray)
    np.testing.assert_allclose(
        output_numpy2, output_cupy2.get(), equal_nan=True)

    kernel3 = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
    output_numpy3 = convolve_2d(numpy_agg.data, kernel3)
    output_cupy3 = convolve_2d(cupy_agg.data, kernel3)
    assert isinstance(output_cupy3, cupy.ndarray)
    np.testing.assert_allclose(
        output_numpy3, output_cupy3.get(), equal_nan=True)

    # dask + cupy case not implemented
    dask_cupy_agg = xr.DataArray(
        da.from_array(cupy.asarray(data), chunks=(3, 3))
    )
    with pytest.raises(NotImplementedError) as e_info:
        convolve_2d(dask_cupy_agg.data, kernel3)
        assert e_info


data_apply = np.array([[0, 1, 2, 3, 4, 5],
                       [6, 7, 8, 9, 10, 11],
                       [12, 13, 14, 15, 16, 17],
                       [18, 19, 20, 21, 22, 23]])
kernel_apply = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])


def test_apply_cpu():
    @ngjit
    def func_zero_cpu(x):
        return 0
    expected_results = np.zeros_like(data_apply)

    # numpy case
    numpy_agg = xr.DataArray(data_apply)
    numpy_apply = apply(numpy_agg, kernel_apply, func_zero_cpu)
    general_output_checks(numpy_agg, numpy_apply, expected_results)

    # dask + numpy case
    dask_numpy_agg = xr.DataArray(da.from_array(data_apply, chunks=(3, 3)))
    dask_numpy_apply = apply(dask_numpy_agg, kernel_apply, func_zero_cpu)
    general_output_checks(dask_numpy_agg, dask_numpy_apply, expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_apply_gpu():
    def func_zero(x):
        return 0

    @ngjit
    def func_zero_cpu(x):
        return 0

    expected_results = np.zeros_like(data_apply)

    # cupy case
    import cupy
    cupy_agg = xr.DataArray(cupy.asarray(data_apply))
    cupy_apply = apply(cupy_agg, kernel_apply, func_zero)
    general_output_checks(cupy_agg, cupy_apply, expected_results)

    # dask + cupy case not implemented
    dask_cupy_agg = xr.DataArray(
        da.from_array(cupy.asarray(data_apply), chunks=(3, 3))
    )
    with pytest.raises(NotImplementedError) as e_info:
        apply(dask_cupy_agg, kernel_apply, func_zero)
        assert e_info


def test_focal_stats_cpu():
    data = np.arange(16).reshape(4, 4)
    numpy_agg = xr.DataArray(data)
    dask_numpy_agg = xr.DataArray(da.from_array(data, chunks=(3, 3)))

    cellsize = (1, 1)
    kernel = circle_kernel(*cellsize, 1.5)

    expected_results = np.asarray([
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

    numpy_focalstats = focal_stats(numpy_agg, kernel)
    general_output_checks(
        numpy_agg, numpy_focalstats,
        verify_attrs=False, expected_results=expected_results
    )
    assert numpy_focalstats.ndim == 3

    dask_numpy_focalstats = focal_stats(dask_numpy_agg, kernel)
    general_output_checks(
        dask_numpy_agg, dask_numpy_focalstats,
        verify_attrs=False, expected_results=expected_results
    )


def test_hotspots():
    n, m = 10, 10
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
    numpy_agg = xr.DataArray(data, dims=['y', 'x'])
    numpy_agg['x'] = np.linspace(0, n, n)
    numpy_agg['y'] = np.linspace(0, m, m)

    cellsize_x, cellsize_y = calc_cellsize(numpy_agg)
    kernel = circle_kernel(cellsize_x, cellsize_y, 2.0)

    expected_results = np.array([
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0, 90,  0,  0,  0,  0,  0,  0,  0],
        [0, 90, 95, 90,  0,  0,  0,  0,  0,  0],
        [0,  0, 90,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0, -90,  0],
        [0,  0,  0,  0,  0,  0,  0, -90, -95,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    ], dtype=np.int8)

    # numpy case
    numpy_hotspots = hotspots(numpy_agg, kernel)
    general_output_checks(numpy_agg, numpy_hotspots, expected_results)

    # dask + numpy
    dask_numpy_agg = xr.DataArray(da.from_array(data, chunks=(3, 3)))
    dask_numpy_hotspots = hotspots(dask_numpy_agg, kernel)
    general_output_checks(
        dask_numpy_agg, dask_numpy_hotspots, expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_hotspot_gpu_equals_cpu():
    n, m = 10, 10
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
    numpy_agg = xr.DataArray(data, dims=['y', 'x'])
    numpy_agg['x'] = np.linspace(0, n, n)
    numpy_agg['y'] = np.linspace(0, m, m)

    cellsize_x, cellsize_y = calc_cellsize(numpy_agg)
    kernel = circle_kernel(cellsize_x, cellsize_y, 2.0)

    # numpy case
    numpy_hotspots = hotspots(numpy_agg, kernel)

    # cupy case
    import cupy
    cupy_agg = xr.DataArray(cupy.asarray(data))
    cupy_hotspots = hotspots(cupy_agg, kernel)

    np.testing.assert_allclose(
        numpy_hotspots.data, cupy_hotspots.data.get(), equal_nan=True)

    # dask + cupy case not implemented
    dask_cupy_agg = xr.DataArray(
        da.from_array(cupy.asarray(data), chunks=(3, 3))
    )
    with pytest.raises(NotImplementedError) as e_info:
        hotspots(dask_cupy_agg, kernel)
        assert e_info
