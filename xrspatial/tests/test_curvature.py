import numpy as np
import pytest

from xrspatial import curvature
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy,
                                            create_test_raster,
                                            cuda_and_cupy_available,
                                            dask_array_available,
                                            general_output_checks)


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


@dask_array_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, curvature)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_cupy_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, curvature, atol=1e-6, rtol=1e-6)


@dask_array_available
def test_boundary_modes():
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy', attrs={'res': (1, 1)})
    assert_boundary_mode_correctness(numpy_agg, dask_agg, curvature)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((6, 8), (3, 4)),
    ((7, 9), (3, 3)),
    ((10, 15), (5, 5)),
    ((10, 15), (10, 15)),
    ((5, 5), (2, 2)),
])
def test_boundary_numpy_equals_dask(boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=chunks)
    np_result = curvature(numpy_agg, boundary=boundary)
    da_result = curvature(dask_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_boundary_no_nan_flat(boundary):
    """Flat surface with non-nan boundary should produce all zeros, no NaN."""
    data = np.full((8, 10), 50.0, dtype=np.float64)
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=(4, 5))
    np_result = curvature(numpy_agg, boundary=boundary)
    da_result = curvature(dask_agg, boundary=boundary)
    np.testing.assert_allclose(np_result.data, 0.0, atol=1e-10)
    np.testing.assert_allclose(np_result.data, da_result.data.compute(),
                               equal_nan=True, rtol=1e-6)


def test_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float32)
    agg = create_test_raster(data, attrs={'res': (1, 1)})
    with pytest.raises(ValueError, match="boundary must be one of"):
        curvature(agg, boundary='invalid')


@pytest.mark.parametrize("res", [(0, 0), (0, 1), (1, 0),
                                 (np.inf, 1), (1, np.nan)])
def test_curvature_invalid_cellsize(res):
    """curvature() must reject zero or non-finite cell sizes instead of
    silently returning inf for every interior pixel.
    """
    data = np.zeros((5, 5), dtype=np.float32)
    agg = create_test_raster(data, attrs={'res': res})
    with pytest.raises(ValueError, match="non-zero, finite cell size"):
        curvature(agg)
