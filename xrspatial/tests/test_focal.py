import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial import mean
from xrspatial.convolution import (annulus_kernel, calc_cellsize, circle_kernel, convolution_2d,
                                   convolve_2d, custom_kernel)
from xrspatial.focal import apply, focal_stats, hotspots
from xrspatial.tests.general_checks import (create_test_raster, cuda_and_cupy_available,
                                            general_output_checks)
from xrspatial.utils import ngjit


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


@cuda_and_cupy_available
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


@pytest.fixture
def convolution_custom_kernel():
    kernel = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]])
    expected_result = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 2., 3., 3., 4., np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    return kernel, expected_result


def test_kernel_custom_kernel_invalid_type():
    kernel = [1, 0, 0]  # only arrays are accepted, not lists
    with pytest.raises(ValueError):
        custom_kernel(kernel)


def test_kernel_custom_kernel_invalid_shape():
    kernel = np.ones((4, 6))
    with pytest.raises(ValueError):
        custom_kernel(kernel)


def test_kernel(kernel_circle_1_1_1, kernel_annulus_2_2_2_1):
    kernel_circle = circle_kernel(1, 1, 1)
    assert isinstance(kernel_circle, np.ndarray)
    np.testing.assert_allclose(kernel_circle, kernel_circle_1_1_1, equal_nan=True)

    kernel_annulus = annulus_kernel(2, 2, 2, 1)
    assert isinstance(kernel_annulus, np.ndarray)
    np.testing.assert_allclose(kernel_annulus, kernel_annulus_2_2_2_1, equal_nan=True)


def test_convolution_numpy(
    convolve_2d_data,
    convolution_custom_kernel,
    kernel_circle_1_1_1,
    convolution_kernel_circle_1_1_1,
    kernel_annulus_2_2_2_1,
    convolution_kernel_annulus_2_2_1
):
    kernel_custom, expected_result_custom = convolution_custom_kernel
    result_kernel_custom = convolve_2d(convolve_2d_data, kernel_custom)
    assert isinstance(result_kernel_custom, np.ndarray)
    np.testing.assert_allclose(
        result_kernel_custom, expected_result_custom, equal_nan=True
    )

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
    convolution_custom_kernel,
    kernel_circle_1_1_1,
    convolution_kernel_circle_1_1_1,
    kernel_annulus_2_2_2_1,
    convolution_kernel_annulus_2_2_1
):
    dask_agg = create_test_raster(convolve_2d_data, backend='dask+numpy')

    kernel_custom, expected_result_custom = convolution_custom_kernel
    result_kernel_custom = convolution_2d(dask_agg, kernel_custom)
    assert isinstance(result_kernel_custom.data, da.Array)
    np.testing.assert_allclose(
        result_kernel_custom.compute(), expected_result_custom, equal_nan=True
    )

    result_kernel_circle = convolution_2d(dask_agg, kernel_circle_1_1_1)
    assert isinstance(result_kernel_circle.data, da.Array)
    np.testing.assert_allclose(
        result_kernel_circle.compute(), convolution_kernel_circle_1_1_1, equal_nan=True
    )

    result_kernel_annulus = convolution_2d(dask_agg, kernel_annulus_2_2_2_1)
    assert isinstance(result_kernel_annulus.data, da.Array)
    np.testing.assert_allclose(
        result_kernel_annulus.compute(), convolution_kernel_annulus_2_2_1, equal_nan=True
    )


@cuda_and_cupy_available
def test_2d_convolution_gpu(
    convolve_2d_data,
    convolution_custom_kernel,
    kernel_circle_1_1_1,
    convolution_kernel_circle_1_1_1,
    kernel_annulus_2_2_2_1,
    convolution_kernel_annulus_2_2_1
):
    import cupy
    cupy_data = cupy.asarray(convolve_2d_data)

    kernel_custom, expected_result_custom = convolution_custom_kernel
    result_kernel_custom = convolve_2d(cupy_data, kernel_custom)
    assert isinstance(result_kernel_custom, cupy.ndarray)
    np.testing.assert_allclose(
        result_kernel_custom.get(), expected_result_custom, equal_nan=True
    )

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
        convolution_2d(dask_cupy_agg, kernel_custom)
        assert e_info


def test_calc_cellsize_unit_input_attrs(convolve_2d_data):
    agg = create_test_raster(convolve_2d_data, attrs={'res': (1, 1), 'unit': 'km'})
    cellsize = calc_cellsize(agg)
    assert cellsize == (1000, 1000)


def test_calc_cellsize_no_attrs(convolve_2d_data):
    agg = create_test_raster(convolve_2d_data)
    cellsize = calc_cellsize(agg)
    assert cellsize == (0.5, 0.5)


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


@pytest.fixture
def data_focal_stats():
    data = np.arange(16).reshape(4, 4)
    kernel = custom_kernel(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]))
    expected_result = np.asarray([
        # mean
        [[0, 1, 2, 3.],
         [4, 2.5,  3.5,  4.5],
         [8, 6.5,  7.5,  8.5],
         [12, 10.5,  11.5,  12.5]],
        # max
        [[0, 1, 2, 3.],
         [4, 5, 6, 7.],
         [8, 9, 10, 11.],
         [12, 13, 14, 15.]],
        # min
        [[0, 1, 2, 3.],
         [4, 0, 1, 2.],
         [8, 4, 5, 6.],
         [12, 8, 9, 10.]],
        # range
        [[0, 0, 0, 0.],
         [0, 5, 5, 5.],
         [0, 5, 5, 5.],
         [0, 5, 5, 5.]],
        # std
        [[0, 0, 0, 0.],
         [0, 2.5,  2.5,  2.5],
         [0, 2.5,  2.5,  2.5],
         [0, 2.5,  2.5,  2.5]],
        # var
        [[0, 0, 0, 0.],
         [0, 6.25, 6.25, 6.25],
         [0, 6.25, 6.25, 6.25],
         [0, 6.25, 6.25, 6.25]],
        # sum
        [[0, 1, 2, 3.],
         [4, 5, 7, 9.],
         [8, 13, 15, 17.],
         [12, 21, 23, 25.]]
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


@cuda_and_cupy_available
def test_focal_stats_gpu(data_focal_stats):
    data, kernel, expected_result = data_focal_stats
    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_focalstats = focal_stats(cupy_agg, kernel)
    general_output_checks(
        cupy_agg, cupy_focalstats, verify_attrs=False, expected_results=expected_result
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


def test_hotspots_zero_global_std():
    data = np.zeros((10, 20))
    agg = create_test_raster(data)
    kernel = np.zeros((3, 3))
    msg = "Standard deviation of the input raster values is 0."
    with pytest.raises(ZeroDivisionError, match=msg):
        hotspots(agg, kernel)


def test_hotspots_numpy(data_hotspots):
    data, kernel, expected_result = data_hotspots
    numpy_agg = create_test_raster(data)
    numpy_hotspots = hotspots(numpy_agg, kernel)
    general_output_checks(numpy_agg, numpy_hotspots, expected_result, verify_attrs=False)
    # validate attrs
    assert numpy_hotspots.shape == numpy_agg.shape
    assert numpy_hotspots.dims == numpy_agg.dims
    for coord in numpy_agg.coords:
        np.testing.assert_allclose(
            numpy_hotspots[coord].data, numpy_agg[coord].data, equal_nan=True
        )
    assert numpy_hotspots.attrs['unit'] == '%'


def test_hotspots_dask_numpy(data_hotspots):
    data, kernel, expected_result = data_hotspots
    dask_numpy_agg = create_test_raster(data, backend='dask')
    dask_numpy_hotspots = hotspots(dask_numpy_agg, kernel)
    general_output_checks(dask_numpy_agg, dask_numpy_hotspots, expected_result, verify_attrs=False)
    # validate attrs
    assert dask_numpy_hotspots.shape == dask_numpy_agg.shape
    assert dask_numpy_hotspots.dims == dask_numpy_agg.dims
    for coord in dask_numpy_agg.coords:
        np.testing.assert_allclose(
            dask_numpy_hotspots[coord].data, dask_numpy_agg[coord].data, equal_nan=True
        )
    assert dask_numpy_hotspots.attrs['unit'] == '%'


@cuda_and_cupy_available
def test_hotspot_gpu(data_hotspots):
    data, kernel, expected_result = data_hotspots
    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_hotspots = hotspots(cupy_agg, kernel)
    general_output_checks(cupy_agg, cupy_hotspots, expected_result, verify_attrs=False)
    # validate attrs
    assert cupy_hotspots.shape == cupy_agg.shape
    assert cupy_hotspots.dims == cupy_agg.dims
    for coord in cupy_agg.coords:
        np.testing.assert_allclose(
            cupy_hotspots[coord].data, cupy_agg[coord].data, equal_nan=True
        )
    assert cupy_hotspots.attrs['unit'] == '%'

    # dask + cupy case not implemented
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy')
    with pytest.raises(NotImplementedError) as e_info:
        hotspots(dask_cupy_agg, kernel)
        assert e_info
