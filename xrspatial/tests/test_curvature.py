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

    small_da = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    cpu = curvature(small_da, name='numpy_result')

    small_da_cupy = xr.DataArray(cupy.asarray(elevation),
                                 attrs={'res': (10.0, 10.0)})
    gpu = curvature(small_da_cupy, name='cupy_result')
    general_output_checks(small_da_cupy, gpu)

    assert np.isclose(cpu, gpu.data.get(), equal_nan=True).all()


def test_curvature_numpy_equals_dask():
    small_numpy_based_data_array = xr.DataArray(
        elevation, attrs={'res': (10.0, 10.0)}
    )
    small_dask_based_data_array = xr.DataArray(
        da.from_array(elevation, chunks=(3, 3)), attrs={'res': (10.0, 10.0)}
    )

    numpy_curvature = curvature(small_numpy_based_data_array,
                                name='numpy_curvature')
    dask_curvature = curvature(small_dask_based_data_array,
                               name='dask_curvature')

    general_output_checks(small_dask_based_data_array, dask_curvature)

    dask_curvature.data = dask_curvature.data.compute()
    assert np.isclose(numpy_curvature, dask_curvature, equal_nan=True).all()
