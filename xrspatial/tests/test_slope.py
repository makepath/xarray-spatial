import pytest
import numpy as np

from xrspatial import slope
from xrspatial.utils import doesnt_have_cuda

from xrspatial.tests.general_checks import create_test_raster
from xrspatial.tests.general_checks import assert_numpy_equals_dask_numpy
from xrspatial.tests.general_checks import assert_numpy_equals_cupy
from xrspatial.tests.general_checks import assert_nan_edges_effect
from xrspatial.tests.general_checks import general_output_checks


def input_data(backend):
    # Notes:
    # ------
    # The `elevation` data was run through QGIS slope function to
    # get values to compare against.  Xarray-Spatial currently handles
    # edges by padding with nan which is different than QGIS but acknowledged

    elevation = np.asarray([
        [1432.6542, 1432.4764, 1432.4764, 1432.1207, 1431.9429, np.nan],
        [1432.6542, 1432.6542, 1432.4764, 1432.2986, 1432.1207, np.nan],
        [1432.832, 1432.6542, 1432.4764, 1432.2986, 1432.1207, np.nan],
        [1432.832, 1432.6542, 1432.4764, 1432.4764, 1432.1207, np.nan],
        [1432.832, 1432.6542, 1432.6542, 1432.4764, 1432.2986, np.nan],
        [1432.832, 1432.6542, 1432.6542, 1432.4764, 1432.2986, np.nan],
        [1432.832, 1432.832, 1432.6542, 1432.4764, 1432.4764, np.nan]],
        dtype=np.float32
    )
    raster = create_test_raster(elevation, backend, attrs={'res': (10, 10)})
    return raster


@pytest.fixture
def qgis_output():
    qgis_slope = np.asarray(
        [[0.8052942, 0.742317, 1.1390567, 1.3716657, np.nan, np.nan],
         [0.74258685, 0.742317, 1.0500116, 1.2082565, np.nan, np.nan],
         [0.56964326, 0.9002944, 0.9002944, 1.0502871, np.nan, np.nan],
         [0.5095078, 0.9003686, 0.742317, 1.1390567, np.nan, np.nan],
         [0.6494868, 0.64938396, 0.5692523, 1.0500116, np.nan, np.nan],
         [0.80557066, 0.56964326, 0.64914393, 0.9002944, np.nan, np.nan],
         [0.6494868, 0.56964326, 0.8052942, 0.742317, np.nan, np.nan]],
        dtype=np.float32)
    return qgis_slope


def test_numpy_equals_qgis(qgis_output):
    # slope by xrspatial
    numpy_agg = input_data(backend='numpy')
    xrspatial_slope_numpy = slope(numpy_agg, name='slope_numpy')
    general_output_checks(numpy_agg, xrspatial_slope_numpy)
    assert xrspatial_slope_numpy.name == 'slope_numpy'

    xrspatial_vals = xrspatial_slope_numpy.data[1:-1, 1:-1]
    qgis_vals = qgis_output[1:-1, 1:-1]
    np.testing.assert_allclose(xrspatial_vals, qgis_vals, equal_nan=True)

    # nan border edges
    assert_nan_edges_effect(xrspatial_slope_numpy)


def test_numpy_equals_dask_qgis_data():
    # compare using the data run through QGIS
    numpy_agg = input_data('numpy')
    dask_agg = input_data('dask+numpy')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, slope)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_numpy_equals_cupy_qgis_data():
    # compare using the data run through QGIS
    numpy_agg = input_data('numpy')
    cupy_agg = input_data('cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, slope)
