import numpy as np
import pytest

from xrspatial import slope
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_nan_edges_effect, assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy, create_test_raster,
                                            cuda_and_cupy_available,
                                            dask_array_available, general_output_checks)


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


@dask_array_available
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


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_cupy_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, slope, atol=1e-6, rtol=1e-6)


@dask_array_available
def test_boundary_modes(elevation_raster):
    numpy_agg = input_data(elevation_raster, 'numpy')
    dask_agg = input_data(elevation_raster, 'dask+numpy')
    assert_boundary_mode_correctness(numpy_agg, dask_agg, slope)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((6, 8), (3, 4)),      # chunks evenly divide array
    ((7, 9), (3, 3)),      # ragged last chunk
    ((10, 15), (5, 5)),    # larger array, medium chunks
    ((10, 15), (10, 15)),  # single chunk (no overlap needed)
    ((5, 5), (2, 2)),      # many small chunks
])
def test_boundary_numpy_equals_dask(boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float64) * 500
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=chunks)
    np_result = slope(numpy_agg, boundary=boundary)
    da_result = slope(dask_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_boundary_no_nan_edges(boundary, elevation_raster_no_nans):
    """Non-nan modes produce no NaN output when source has no NaN."""
    numpy_agg = create_test_raster(elevation_raster_no_nans, backend='numpy',
                                   attrs={'res': (1, 1)})
    dask_agg = create_test_raster(elevation_raster_no_nans, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=(4, 3))
    np_result = slope(numpy_agg, boundary=boundary)
    da_result = slope(dask_agg, boundary=boundary)
    assert not np.any(np.isnan(np_result.data))
    assert not np.any(np.isnan(da_result.data.compute()))
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
def test_boundary_constant_surface():
    """Constant surface should produce zero slope for all boundary modes."""
    data = np.full((8, 10), 42.0, dtype=np.float64)
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=(4, 5))
    for mode in ('nan', 'nearest', 'reflect', 'wrap'):
        np_result = slope(numpy_agg, boundary=mode)
        da_result = slope(dask_agg, boundary=mode)
        np_data = np_result.data
        da_data = da_result.data.compute()
        # Interior should be zero; edges depend on mode
        if mode != 'nan':
            np.testing.assert_allclose(np_data, 0.0, atol=1e-10)
        np.testing.assert_allclose(np_data, da_data, equal_nan=True, rtol=1e-6)


def test_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float32)
    agg = create_test_raster(data, attrs={'res': (1, 1)})
    with pytest.raises(ValueError, match="boundary must be one of"):
        slope(agg, boundary='invalid')
