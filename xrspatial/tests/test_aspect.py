import numpy as np
import pytest

from xrspatial import aspect
from xrspatial.tests.general_checks import (assert_nan_edges_effect, assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_numpy, create_test_raster,
                                            cuda_and_cupy_available, general_output_checks)


def input_data(data, backend='numpy'):
    raster = create_test_raster(data, backend)
    return raster


@pytest.fixture
def qgis_aspect():
    result = np.array([
        [    np.nan,     np.nan,     np.nan,     np.nan,     np.nan,    np.nan],
        [    np.nan,     np.nan,     np.nan,     np.nan,     np.nan,    np.nan],
        [233.19478 , 278.358   ,  45.18813 , 306.6476  , 358.34296 , 106.45898 ],
        [267.7002  , 274.42487 ,  11.035832, 357.9641  , 129.98279 , 50.069843],
        [263.18484 , 238.47426 , 196.37103 , 149.25227 , 187.85748 , 263.684   ],
        [266.63937 , 271.05124 , 312.09726 , 348.89136 , 351.618   , 315.59424 ],
        [279.90872 , 314.11356 , 345.76315 , 327.5568  , 339.5455  , 312.9249  ],
        [271.93985 , 268.81046 ,  24.793104, 185.978   , 299.82904 ,159.0188  ]], dtype=np.float32)
    return result


def test_numpy_equals_qgis(elevation_raster, qgis_aspect):
    numpy_agg = input_data(elevation_raster, backend='numpy')
    xrspatial_aspect = aspect(numpy_agg, name='numpy_aspect')

    general_output_checks(numpy_agg, xrspatial_aspect, verify_dtype=True)
    assert xrspatial_aspect.name == 'numpy_aspect'

    xrspatial_vals = xrspatial_aspect.data[1:-1, 1:-1]
    qgis_vals = qgis_aspect[1:-1, 1:-1]
    # aspect is nan if nan input
    # aspect is invalid (-1) if slope equals 0
    # otherwise aspect are from 0 to 360
    np.testing.assert_allclose(xrspatial_vals, qgis_vals, rtol=1e-05, equal_nan=True)
    # nan edge effect
    assert_nan_edges_effect(xrspatial_aspect)


def test_numpy_equals_dask_qgis_data(elevation_raster):
    # compare using the data run through QGIS
    numpy_agg = input_data(elevation_raster, 'numpy')
    dask_agg = input_data(elevation_raster, 'dask+numpy')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, aspect)


@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, aspect)


@cuda_and_cupy_available
def test_numpy_equals_cupy_qgis_data():
    # compare using the data run through QGIS
    numpy_agg = input_data(elevation_raster)
    cupy_agg = input_data(elevation_raster, 'cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, aspect)


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_cupy_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    cupy_agg = create_test_raster(random_data, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, aspect, atol=1e-6, rtol=1e-6)
