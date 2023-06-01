import numpy as np
import pytest

from xrspatial import slope
from xrspatial.tests.general_checks import (assert_nan_edges_effect, assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_numpy, create_test_raster,
                                            cuda_and_cupy_available, general_output_checks)


def input_data(data, backend):
    # Notes:
    # ------
    # The `elevation` data was run through QGIS slope function to
    # get values to compare against.  Xarray-Spatial currently handles
    # edges by padding with nan which is different than QGIS but acknowledged
    raster = create_test_raster(data, backend, attrs={'res': (1, 1)})
    return raster


@pytest.fixture
def qgis_slope():
    qgis_result = np.array([
        [   np.nan,    np.nan,    np.nan,    np.nan,    np.nan,    np.nan],
        [   np.nan,    np.nan,    np.nan,    np.nan,    np.nan,    np.nan],
        [89.707756, 88.56143 , 89.45366 , 89.50229 , 88.82584 , 89.782394],
        [89.78415 , 89.61588 , 89.47127 , 89.24196 , 88.385376, 89.67071 ],
        [89.7849  , 89.61132 , 89.59183 , 89.56854 , 88.90889 , 89.765114],
        [89.775246, 89.42886 , 89.25054 , 89.60963 , 89.71719 , 89.76396 ],
        [89.85427 , 89.75693 , 89.67336 , 89.502174, 89.24611 , 89.352   ],
        [89.87612 , 89.76542 , 89.269966, 89.78526 , 88.35767 , 89.764206]],
        dtype=np.float32)
    return qgis_result


def test_numpy_equals_qgis(elevation_raster, qgis_slope):
    # slope by xrspatial
    numpy_agg = input_data(elevation_raster, backend='numpy')
    xrspatial_slope_numpy = slope(numpy_agg, name='slope_numpy')
    general_output_checks(numpy_agg, xrspatial_slope_numpy)
    assert xrspatial_slope_numpy.name == 'slope_numpy'
    print('numpy_agg', numpy_agg)
    print('xrspatial_slope_numpy', xrspatial_slope_numpy)
    xrspatial_vals = xrspatial_slope_numpy.data[1:-1, 1:-1]
    qgis_vals = qgis_slope[1:-1, 1:-1]
    print('xrspatial_vals', xrspatial_vals)

    np.testing.assert_allclose(xrspatial_vals, qgis_vals, rtol=1e-05, equal_nan=True)

    # nan border edges
    assert_nan_edges_effect(xrspatial_slope_numpy)


def test_numpy_equals_dask_qgis_data(elevation_raster):
    # compare using the data run through QGIS
    numpy_agg = input_data(elevation_raster, 'numpy')
    dask_agg = input_data(elevation_raster, 'dask+numpy')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, slope)


@cuda_and_cupy_available
def test_numpy_equals_cupy_qgis_data(elevation_raster):
    # compare using the data run through QGIS
    numpy_agg = input_data(elevation_raster, 'numpy')
    cupy_agg = input_data(elevation_raster, 'cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, slope)
