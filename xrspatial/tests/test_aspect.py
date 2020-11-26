import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial import aspect
from xrspatial.utils import doesnt_have_cuda


def _do_sparse_array(data_array):
    import random
    indx = list(zip(*np.where(data_array)))
    pos = random.sample(range(data_array.size), data_array.size // 2)
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
    x_fac = -np.power(X - _mean, 2)
    y_fac = -np.power(Y - _mean, 2)
    gaussian = np.exp((x_fac + y_fac) / (2 * _sdev ** 2)) / (2.5 * _sdev)
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
    # default name
    assert da_aspect.name == 'aspect'
    assert da_aspect.dims == da.dims
    assert da_aspect.attrs == da.attrs
    assert da.shape == da_aspect.shape
    for coord in da.coords:
        assert np.all(da[coord] == da_aspect[coord])
    assert pytest.approx(np.nanmax(da_aspect.data), .1) == 360.
    assert pytest.approx(np.nanmin(da_aspect.data), .1) == -1.


def test_aspect_against_qgis():
    # input data
    data = np.asarray([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                       [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
                       [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
                       [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
                       [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
                      dtype=np.float32)
    small_da = xr.DataArray(data, attrs={'res': (10.0, 10.0)})

    # aspect by QGIS
    qgis_aspect = np.asarray([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [330.94687, 335.55496, 320.70786, 330.94464, 0., 0.],
        [333.43494, 333.43494, 329.03394, 341.56897, 0., 18.434948],
        [338.9621, 338.20062, 341.56506, 0., 0., 45.],
        [341.56506, 351.8699, 26.56505, 45., -1., 90.],
        [351.86676, 11.306906, 45., 45., 45., 108.431015]],
        dtype=np.float32)

    # aspect by xrspatial
    xrspatial_aspect = aspect(small_da, name='aspect_agg')

    # validate output attributes
    assert xrspatial_aspect.dims == small_da.dims
    assert xrspatial_aspect.attrs == small_da.attrs
    assert xrspatial_aspect.shape == small_da.shape
    assert xrspatial_aspect.name == 'aspect_agg'
    for coord in small_da.coords:
        assert np.all(xrspatial_aspect[coord] == small_da[coord])

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_aspect.values[1:-1, 1:-1]
    qgis_vals = qgis_aspect[1:-1, 1:-1]
    # set a tolerance of 1e-4
    # aspect is nan if nan input
    # aspect is invalid (nan) if slope equals 0
    # otherwise aspect are from 0 - 360
    assert ((np.isnan(xrspatial_vals) & np.isnan(qgis_vals)) | (
            np.isnan(xrspatial_vals) & (qgis_vals == -1)) | (
            abs(xrspatial_vals - qgis_vals) <= 1e-4)).all()

    assert (np.isnan(xrspatial_vals) | (
            (0 <= xrspatial_vals) & (xrspatial_vals <= 360))).all()

def test_aspect_against_qgis():
    # input data
    data = np.asarray([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                       [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
                       [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
                       [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
                       [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
                      dtype=np.float32)
    small_da = xr.DataArray(data, attrs={'res': (10.0, 10.0)})

    # aspect by QGIS
    qgis_aspect = np.asarray([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [330.94687, 335.55496, 320.70786, 330.94464, 0., 0.],
        [333.43494, 333.43494, 329.03394, 341.56897, 0., 18.434948],
        [338.9621, 338.20062, 341.56506, 0., 0., 45.],
        [341.56506, 351.8699, 26.56505, 45., -1., 90.],
        [351.86676, 11.306906, 45., 45., 45., 108.431015]],
        dtype=np.float32)

    # aspect by xrspatial
    xrspatial_aspect = aspect(small_da, name='aspect_agg')

    # validate output attributes
    assert xrspatial_aspect.dims == small_da.dims
    assert xrspatial_aspect.attrs == small_da.attrs
    assert xrspatial_aspect.shape == small_da.shape
    assert xrspatial_aspect.name == 'aspect_agg'
    for coord in small_da.coords:
        assert np.all(xrspatial_aspect[coord] == small_da[coord])

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_aspect.values[1:-1, 1:-1]
    qgis_vals = qgis_aspect[1:-1, 1:-1]
    # set a tolerance of 1e-4
    # aspect is nan if nan input
    # aspect is invalid (nan) if slope equals 0
    # otherwise aspect are from 0 - 360
    assert ((np.isnan(xrspatial_vals) & np.isnan(qgis_vals)) | (
            np.isnan(xrspatial_vals) & (qgis_vals == -1)) | (
            abs(xrspatial_vals - qgis_vals) <= 1e-4)).all()

    assert (np.isnan(xrspatial_vals) | (
            (0 <= xrspatial_vals) & (xrspatial_vals <= 360))).all()


def test_aspect_against_qgis():
    # input data
    data = np.asarray([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                       [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
                       [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
                       [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
                       [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
                      dtype=np.float32)
    small_da = xr.DataArray(data, attrs={'res': (10.0, 10.0)})

    # aspect by QGIS
    qgis_aspect = np.asarray([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [330.94687, 335.55496, 320.70786, 330.94464, 0., 0.],
        [333.43494, 333.43494, 329.03394, 341.56897, 0., 18.434948],
        [338.9621, 338.20062, 341.56506, 0., 0., 45.],
        [341.56506, 351.8699, 26.56505, 45., -1., 90.],
        [351.86676, 11.306906, 45., 45., 45., 108.431015]],
        dtype=np.float32)

    # aspect by xrspatial
    xrspatial_aspect = aspect(small_da, name='aspect_agg')

    # validate output attributes
    assert xrspatial_aspect.dims == small_da.dims
    assert xrspatial_aspect.attrs == small_da.attrs
    assert xrspatial_aspect.shape == small_da.shape
    assert xrspatial_aspect.name == 'aspect_agg'
    for coord in small_da.coords:
        assert np.all(xrspatial_aspect[coord] == small_da[coord])

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_aspect.values[1:-1, 1:-1]
    qgis_vals = qgis_aspect[1:-1, 1:-1]
    # set a tolerance of 1e-4
    # aspect is nan if nan input
    # aspect is invalid (nan) if slope equals 0
    # otherwise aspect are from 0 - 360
    assert ((np.isnan(xrspatial_vals) & np.isnan(qgis_vals)) | (
            np.isnan(xrspatial_vals) & (qgis_vals == -1)) | (
            abs(xrspatial_vals - qgis_vals) <= 1e-4)).all()
    
    assert (np.isnan(xrspatial_vals) | (
            (0 <= xrspatial_vals) & (xrspatial_vals <= 360) | (xrspatial_vals == -1))).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_aspect_against_qgis_gpu():
    import cupy

    # input data
    data = np.asarray([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                       [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
                       [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
                       [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
                       [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
                      dtype=np.float32)

    small_da = xr.DataArray(data, attrs={'res': (10.0, 10.0)})
    small_da_cupy = xr.DataArray(cupy.asarray(data), attrs={'res': (10.0, 10.0)})

    # aspect by QGIS
    qgis_aspect = np.asarray([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [330.94687, 335.55496, 320.70786, 330.94464, 0., 0.],
        [333.43494, 333.43494, 329.03394, 341.56897, 0., 18.434948],
        [338.9621, 338.20062, 341.56506, 0., 0., 45.],
        [341.56506, 351.8699, 26.56505, 45., -1., 90.],
        [351.86676, 11.306906, 45., 45., 45., 108.431015]],
        dtype=np.float32)

    # aspect by xrspatial
    xrspatial_aspect = aspect(small_da_cupy, name='aspect_agg')

    # validate output attributes
    assert xrspatial_aspect.dims == small_da.dims
    assert xrspatial_aspect.attrs == small_da.attrs
    assert xrspatial_aspect.shape == small_da.shape
    assert xrspatial_aspect.name == 'aspect_agg'
    for coord in small_da.coords:
        assert np.all(xrspatial_aspect[coord] == small_da[coord])

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_aspect.values[1:-1, 1:-1]
    qgis_vals = qgis_aspect[1:-1, 1:-1]
    # set a tolerance of 1e-4
    # aspect is nan if nan input
    # aspect is invalid (nan) if slope equals 0
    # otherwise aspect are from 0 - 360


    assert np.isclose(xrspatial_vals, qgis_vals, equal_nan=True).all()

    #assert (np.isnan(xrspatial_vals) | (
    #        (0. <= xrspatial_vals) & (xrspatial_vals <= 360.))).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_aspect_gpu_equals_cpu():

    import cupy

    # input data
    data = np.asarray([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                       [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
                       [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
                       [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
                       [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
                      dtype=np.float32)

    small_da = xr.DataArray(data, attrs={'res': (10.0, 10.0)})
    small_da_cupy = xr.DataArray(cupy.asarray(data), attrs={'res': (10.0, 10.0)})

    # aspect by xrspatial
    xrspatial_aspect_cpu = aspect(small_da, name='aspect_agg')
    xrspatial_aspect_gpu = aspect(small_da_cupy, name='aspect_agg')

    assert np.isclose(xrspatial_aspect_cpu, xrspatial_aspect_gpu, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_aspect_against_qgis_gpu():
    # input data
    data = np.asarray([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                       [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
                       [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
                       [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
                       [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
                      dtype=np.float32)
    small_da = xr.DataArray(data, attrs={'res': (10.0, 10.0)})
    small_da_cupy = xr.DataArray(cupy.asarray(data), attrs={'res': (10.0, 10.0)})

    # aspect by QGIS
    qgis_aspect = np.asarray([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [330.94687, 335.55496, 320.70786, 330.94464, 0., 0.],
        [333.43494, 333.43494, 329.03394, 341.56897, 0., 18.434948],
        [338.9621, 338.20062, 341.56506, 0., 0., 45.],
        [341.56506, 351.8699, 26.56505, 45., -1., 90.],
        [351.86676, 11.306906, 45., 45., 45., 108.431015]],
        dtype=np.float32)

    # aspect by xrspatial
    xrspatial_aspect = aspect(small_da_cupy, name='aspect_agg')

    # validate output attributes
    assert xrspatial_aspect.dims == small_da.dims
    assert xrspatial_aspect.attrs == small_da.attrs
    assert xrspatial_aspect.shape == small_da.shape
    assert xrspatial_aspect.name == 'aspect_agg'
    for coord in small_da.coords:
        assert np.all(xrspatial_aspect[coord] == small_da[coord])

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_aspect.values[1:-1, 1:-1]
    qgis_vals = qgis_aspect[1:-1, 1:-1]
    # set a tolerance of 1e-4
    # aspect is nan if nan input
    # aspect is invalid (nan) if slope equals 0
    # otherwise aspect are from 0 - 360


    assert np.isclose(xrspatial_vals, qgis_vals, equal_nan=True).all()

    #assert (np.isnan(xrspatial_vals) | (
    #        (0. <= xrspatial_vals) & (xrspatial_vals <= 360.))).all()


def test_numpy_equals_dask():

    # input data
    data = np.asarray([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                       [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
                       [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
                       [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
                       [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
                      dtype=np.float32)

    small_numpy_based_data_array = xr.DataArray(data, attrs={'res': (10.0, 10.0)})
    small_das_based_data_array = xr.DataArray(da.from_array(data, chunks=(2, 2)),
                                              attrs={'res': (10.0, 10.0)})

    numpy_result = aspect(small_numpy_based_data_array, name='numpy_slope')
    dask_result = aspect(small_das_based_data_array, name='dask_slope')
    dask_result.data = dask_result.data.compute()

    assert np.isclose(numpy_result, dask_result, equal_nan=True).all()
