import numpy as np
import pytest

from xrspatial import curvature
from xrspatial.tests.general_checks import (assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_numpy, create_test_raster,
                                            cuda_and_cupy_available, general_output_checks)


@pytest.fixture
def flat_surface(size, dtype):
    flat = np.zeros(size, dtype=dtype)
    expected_result = np.zeros(size, dtype=np.float32)
    # nan edges effect
    expected_result[0, :] = np.nan
    expected_result[-1, :] = np.nan
    expected_result[:, 0] = np.nan
    expected_result[:, -1] = np.nan
    return flat, expected_result


@pytest.fixture
def convex_surface():
    convex_data = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])
    expected_result = np.asarray([
         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
         [np.nan, 0,      0.,     100.,     0.,   np.nan],
         [np.nan, 0,      100.,  -400.,   100.,   np.nan],
         [np.nan, 0,      0.,     100.,     0.,   np.nan],
         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ], dtype=np.float32)
    return convex_data, expected_result


@pytest.fixture
def concave_surface():
    concave_data = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])
    expected_result = np.asarray([
         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
         [np.nan, 0,      0.,    -100.,     0.,   np.nan],
         [np.nan, 0,     -100.,   400.,  -100.,   np.nan],
         [np.nan, 0,      0.,    -100.,     0.,   np.nan],
         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ], dtype=np.float32)
    return concave_data, expected_result


@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_curvature_on_flat_surface(flat_surface):
    flat_data, expected_result = flat_surface
    numpy_agg = create_test_raster(flat_data, attrs={'res': (1, 1)})
    numpy_result = curvature(numpy_agg)
    general_output_checks(numpy_agg, numpy_result, expected_result, verify_dtype=True)


def test_curvature_on_convex_surface(convex_surface):
    convex_data, expected_result = convex_surface
    numpy_agg = create_test_raster(convex_data, attrs={'res': (1, 1)})
    numpy_result = curvature(numpy_agg)
    general_output_checks(numpy_agg, numpy_result, expected_result, verify_dtype=True)


def test_curvature_on_concave_surface(concave_surface):
    concave_data, expected_result = concave_surface
    numpy_agg = create_test_raster(concave_data, attrs={'res': (1, 1)})
    numpy_result = curvature(numpy_agg)
    general_output_checks(numpy_agg, numpy_result, expected_result, verify_dtype=True)


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_cupy_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    cupy_agg = create_test_raster(random_data, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, curvature)
    # NOTE: Dask + GPU code paths don't currently work because of
    # dask casting cupy arrays to numpy arrays during
    # https://github.com/dask/dask/issues/4842


@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, curvature)
