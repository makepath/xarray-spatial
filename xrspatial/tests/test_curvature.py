import pytest
import numpy as np
import xarray as xr

import dask.array as da

from xrspatial import curvature
from xrspatial.utils import doesnt_have_cuda

from xrspatial.tests.general_checks import general_output_checks


elevation = np.asarray([
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
    [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
    [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
    [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
    [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
    [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
    dtype=np.float32
)


def test_curvature_on_flat_surface():
    # flat surface
    test_arr1 = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    expected_results = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan,      0,      0,      0, np.nan],
        [np.nan,      0,      0,      0, np.nan],
        [np.nan,      0,      0,      0, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    test_raster1 = xr.DataArray(test_arr1, attrs={'res': (1, 1)})
    curv = curvature(test_raster1)
    general_output_checks(test_raster1, curv, expected_results)


def test_curvature_on_convex_surface():
    # convex
    test_arr2 = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, -1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    expected_results = np.asarray([
         [np.nan, np.nan, np.nan,  np.nan, np.nan],
         [np.nan,     0.,   100.,      0., np.nan],
         [np.nan,   100.,  -400.,    100., np.nan],
         [np.nan,     0.,   100.,      0., np.nan],
         [np.nan,  np.nan, np.nan, np.nan, np.nan]
    ])
    test_raster2 = xr.DataArray(test_arr2, attrs={'res': (1, 1)})
    curv = curvature(test_raster2)
    general_output_checks(test_raster2, curv, expected_results)


def test_curvature_on_concave_surface():
    # concave
    test_arr3 = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    expected_results = np.asarray([
         [np.nan, np.nan, np.nan,  np.nan, np.nan],
         [np.nan,     0.,  -100.,      0., np.nan],
         [np.nan,  -100.,   400.,   -100., np.nan],
         [np.nan,     0.,  -100.,      0., np.nan],
         [np.nan,  np.nan, np.nan, np.nan, np.nan]
    ])
    test_raster3 = xr.DataArray(test_arr3, attrs={'res': (1, 1)})
    curv = curvature(test_raster3)
    general_output_checks(test_raster3, curv, expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_curvature_gpu_equals_cpu():

    import cupy

    agg_numpy = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    cpu = curvature(agg_numpy, name='numpy_result')

    agg_cupy = xr.DataArray(
        cupy.asarray(elevation), attrs={'res': (10.0, 10.0)}
    )
    gpu = curvature(agg_cupy, name='cupy_result')

    general_output_checks(agg_cupy, gpu)
    np.testing.assert_allclose(cpu.data, gpu.data.get(), equal_nan=True)

    # NOTE: Dask + GPU code paths don't currently work because of
    # dask casting cupy arrays to numpy arrays during
    # https://github.com/dask/dask/issues/4842


def test_curvature_numpy_equals_dask():
    agg_numpy = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    numpy_curvature = curvature(agg_numpy, name='numpy_curvature')

    agg_dask = xr.DataArray(
        da.from_array(elevation, chunks=(3, 3)), attrs={'res': (10.0, 10.0)}
    )
    dask_curvature = curvature(agg_dask, name='dask_curvature')
    general_output_checks(agg_dask, dask_curvature)

    # both produce same results
    np.testing.assert_allclose(
        numpy_curvature.data, dask_curvature.data.compute(), equal_nan=True)
