import numpy as np
import pytest

from xrspatial import aspect
from xrspatial.utils import doesnt_have_cuda

from xrspatial.tests.general_checks import create_test_raster
from xrspatial.tests.general_checks import assert_numpy_equals_dask_numpy
from xrspatial.tests.general_checks import assert_numpy_equals_cupy
from xrspatial.tests.general_checks import assert_nan_edges_effect
from xrspatial.tests.general_checks import general_output_checks


def input_data(backend='numpy'):
    data = np.asarray([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
        [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
        [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
        [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
        [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
        [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
        dtype=np.float32
    )
    raster = create_test_raster(data, backend, attrs={'res': (10.0, 10.0)})
    return raster


@pytest.fixture
def qgis_output():
    result = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [330.94687, 335.55496, 320.70786, 330.94464, 0., 0.],
        [333.43494, 333.43494, 329.03394, 341.56897, 0., 18.434948],
        [338.9621, 338.20062, 341.56506, 0., 0., 45.],
        [341.56506, 351.8699, 26.56505, 45., -1., 90.],
        [351.86676, 11.306906, 45., 45., 45., 108.431015]], dtype=np.float32
    )
    return result


def test_numpy_equals_qgis(qgis_output):
    numpy_agg = input_data()
    xrspatial_aspect = aspect(numpy_agg, name='numpy_aspect')

    general_output_checks(numpy_agg, xrspatial_aspect, verify_dtype=True)
    assert xrspatial_aspect.name == 'numpy_aspect'

    xrspatial_vals = xrspatial_aspect.data[1:-1, 1:-1]
    qgis_vals = qgis_output[1:-1, 1:-1]
    # aspect is nan if nan input
    # aspect is invalid (-1) if slope equals 0
    # otherwise aspect are from 0 to 360
    np.testing.assert_allclose(xrspatial_vals, qgis_vals, equal_nan=True)
    # nan edge effect
    assert_nan_edges_effect(xrspatial_aspect)


def test_numpy_equals_dask_qgis_data():
    # compare using the data run through QGIS
    numpy_agg = input_data('numpy')
    dask_agg = input_data('dask+numpy')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, aspect)


@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, aspect)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_numpy_equals_cupy_qgis_data():
    # compare using the data run through QGIS
    numpy_agg = input_data()
    cupy_agg = input_data('cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, aspect)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_cupy_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    cupy_agg = create_test_raster(random_data, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, aspect, atol=1e-6, rtol=1e-6)
