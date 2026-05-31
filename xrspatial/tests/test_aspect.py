import numpy as np
import pytest

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial import aspect, eastness, northness
from xrspatial.utils import has_dask_array
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_nan_edges_effect,
                                            assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy,
                                            dask_array_available,
                                            create_test_raster,
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


@dask_array_available
def test_numpy_equals_dask_qgis_data(elevation_raster):
    # compare using the data run through QGIS
    numpy_agg = input_data(elevation_raster, 'numpy')
    dask_agg = input_data(elevation_raster, 'dask+numpy')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, aspect)


@dask_array_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, aspect)


@cuda_and_cupy_available
def test_numpy_equals_cupy_qgis_data(elevation_raster):
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


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_cupy_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, aspect, atol=1e-6, rtol=1e-6)


@dask_array_available
def test_boundary_modes(elevation_raster):
    numpy_agg = input_data(elevation_raster, 'numpy')
    dask_agg = input_data(elevation_raster, 'dask+numpy')
    assert_boundary_mode_correctness(numpy_agg, dask_agg, aspect)


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
    data = rng.random(size).astype(np.float64) * 500
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    np_result = aspect(numpy_agg, boundary=boundary)
    da_result = aspect(dask_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_boundary_no_nan_edges(boundary, elevation_raster_no_nans):
    """Non-nan modes produce no NaN output when source has no NaN."""
    numpy_agg = create_test_raster(elevation_raster_no_nans, backend='numpy')
    dask_agg = create_test_raster(elevation_raster_no_nans, backend='dask+numpy',
                                  chunks=(4, 3))
    np_result = aspect(numpy_agg, boundary=boundary)
    da_result = aspect(dask_agg, boundary=boundary)
    # aspect returns -1 for flat areas, but should not return NaN
    assert not np.any(np.isnan(np_result.data))
    assert not np.any(np.isnan(da_result.data.compute()))
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


def test_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float32)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="boundary must be one of"):
        aspect(agg, boundary='invalid')


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
def test_dask_numpy_advertised_dtype_matches_computed(boundary):
    # planar dask map_overlap must advertise float32, matching the realized
    # data and the numpy/cupy backends (issue #2682).
    data = np.random.default_rng(0).random((8, 10)).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy')
    np_result = aspect(numpy_agg, boundary=boundary)
    da_result = aspect(dask_agg, boundary=boundary)
    assert np_result.dtype == np.float32
    assert da_result.dtype == np.float32
    assert da_result.data.compute().dtype == np.float32


@dask_array_available
@cuda_and_cupy_available
def test_dask_cupy_advertised_dtype_matches_computed():
    import cupy
    data = np.random.default_rng(0).random((8, 10)).astype(np.float64) * 100
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy')
    da_result = aspect(dask_cupy_agg)
    assert da_result.dtype == cupy.float32
    assert da_result.data.compute().dtype == cupy.float32


# ---- Degenerate raster shapes (1x1, Nx1, 1xN) ----
#
# A 3x3 kernel has no interior cell when a dimension is smaller than 3,
# so the correct planar result is all-NaN with the input shape preserved.
# These guard against a regression that would crash or reshape on the
# kernel-boundary path. See issue #2742.

_DEGENERATE_SHAPES = [(1, 1), (1, 5), (5, 1), (3, 1), (1, 3)]


def _to_numpy(result_agg):
    data = result_agg.data
    if has_dask_array() and isinstance(data, da.Array):
        data = data.compute()
    if hasattr(data, 'get'):  # cupy
        data = data.get()
    return data


@pytest.mark.parametrize("func", [aspect, northness, eastness])
@pytest.mark.parametrize("shape", _DEGENERATE_SHAPES)
def test_degenerate_shape_numpy(func, shape):
    data = np.ones(shape, dtype=np.float32)
    if shape[0] > 1:
        data[0, :] = 2.0
    agg = create_test_raster(data, backend='numpy')
    result = func(agg)
    general_output_checks(agg, result)
    assert result.shape == shape
    assert np.all(np.isnan(result.data))


@dask_array_available
@pytest.mark.parametrize("func", [aspect, northness, eastness])
@pytest.mark.parametrize("shape", _DEGENERATE_SHAPES)
def test_degenerate_shape_dask_numpy(func, shape):
    data = np.ones(shape, dtype=np.float32)
    if shape[0] > 1:
        data[0, :] = 2.0
    chunks = (min(2, shape[0]), min(2, shape[1]))
    agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    result = func(agg)
    assert result.shape == shape
    assert np.all(np.isnan(_to_numpy(result)))


@cuda_and_cupy_available
@pytest.mark.parametrize("func", [aspect, northness, eastness])
@pytest.mark.parametrize("shape", _DEGENERATE_SHAPES)
def test_degenerate_shape_cupy(func, shape):
    data = np.ones(shape, dtype=np.float32)
    if shape[0] > 1:
        data[0, :] = 2.0
    agg = create_test_raster(data, backend='cupy')
    result = func(agg)
    assert result.shape == shape
    assert np.all(np.isnan(_to_numpy(result)))


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("func", [aspect, northness, eastness])
@pytest.mark.parametrize("shape", _DEGENERATE_SHAPES)
def test_degenerate_shape_dask_cupy(func, shape):
    data = np.ones(shape, dtype=np.float32)
    if shape[0] > 1:
        data[0, :] = 2.0
    chunks = (min(2, shape[0]), min(2, shape[1]))
    agg = create_test_raster(data, backend='dask+cupy', chunks=chunks)
    result = func(agg)
    assert result.shape == shape
    assert np.all(np.isnan(_to_numpy(result)))
