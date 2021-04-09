import pytest
import numpy as np
import xarray as xr

import dask.array as da

from xrspatial import curvature
from xrspatial.utils import doesnt_have_cuda

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
    test_raster1 = xr.DataArray(test_arr1, attrs={'res': (1, 1)})
    curv = curvature(test_raster1)

    # output must be an xarray DataArray
    assert isinstance(curv, xr.DataArray)
    assert isinstance(curv.values, np.ndarray)
    # shape, dims, coords, attr preserved
    assert test_raster1.shape == curv.shape
    assert test_raster1.dims == curv.dims
    assert test_raster1.attrs == curv.attrs
    for coord in test_raster1.coords:
        assert np.all(test_raster1[coord] == curv[coord])

    # border edges are all nans
    assert np.isnan(curv.values[0, :]).all()
    assert np.isnan(curv.values[-1, :]).all()
    assert np.isnan(curv.values[:, 0]).all()
    assert np.isnan(curv.values[:, -1]).all()

    # curvature of a flat surface is all 0s
    # exclude border edges
    assert np.unique(curv.values[1:-1, 1:-1]) == [0]


def test_curvature_on_convex_surface():
    # convex
    test_arr2 = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, -1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])

    test_raster2 = xr.DataArray(test_arr2, attrs={'res': (1, 1)})
    curv = curvature(test_raster2)

    # output must be an xarray DataArray
    assert isinstance(curv, xr.DataArray)
    assert isinstance(curv.values, np.ndarray)
    # shape, dims, coords, attr preserved
    assert test_raster2.shape == curv.shape
    assert test_raster2.dims == curv.dims
    assert test_raster2.attrs == curv.attrs
    for coord in test_raster2.coords:
        assert np.all(test_raster2[coord] == curv[coord])

    # curvature at a cell (i, j) only considers 4 cells:
    # (i-1, j), (i+1, j), (i, j-1), (i, j+1)

    # id of bottom of the convex shape in the raster
    i, j = (2, 2)

    # The 4 surrounding values should be the same
    assert curv.values[i-1, j] == curv.values[i+1, j]\
        == curv.values[i, j-1] == curv.values[i, j+1]

    # Positive curvature indicates the surface is upwardly convex at that cell
    assert curv.values[i-1, j] > 0

    # Negative curvature indicates the surface is upwardly concave at that cell
    assert curv.values[i, j] < 0

    # A value of 0 indicates the surface is flat.
    # exclude border edges
    for ri in range(1, curv.shape[0] - 1):
        for rj in range(1, curv.shape[1] - 1):
            if ri not in (i-1, i, i+1) and rj not in (j-1, j, j+1):
                assert curv.values[ri, rj] == 0

    # border edges are all nans
    assert np.isnan(curv.values[0, :]).all()
    assert np.isnan(curv.values[-1, :]).all()
    assert np.isnan(curv.values[:, 0]).all()
    assert np.isnan(curv.values[:, -1]).all()


def test_curvature_on_concave_surface():
    # concave
    test_arr3 = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])

    test_raster3 = xr.DataArray(test_arr3, attrs={'res': (1, 1)})
    curv = curvature(test_raster3)

    # output must be an xarray DataArray
    assert isinstance(curv, xr.DataArray)
    assert isinstance(curv.values, np.ndarray)
    # shape, dims, coords, attr preserved
    assert test_raster3.shape == curv.shape
    assert test_raster3.dims == curv.dims
    assert test_raster3.attrs == curv.attrs
    for coord in test_raster3.coords:
        assert np.all(test_raster3[coord] == curv[coord])

    # curvature at a cell (i, j) only considers 4 cells:
    # (i-1, j), (i+1, j), (i, j-1), (i, j+1)

    # id of bottom of the convex shape in the raster
    i, j = (2, 2)

    # The 4 surrounding values should be the same
    assert curv.values[i-1, j] == curv.values[i+1, j]\
        == curv.values[i, j-1] == curv.values[i, j+1]

    # Negative curvature indicates the surface is upwardly concave at that cell
    assert curv.values[i-1, j] < 0

    # Positive curvature indicates the surface is upwardly convex at that cell
    assert curv.values[i, j] > 0

    # A value of 0 indicates the surface is flat.
    # exclude border edges
    for ri in range(1, curv.shape[0] - 1):
        for rj in range(1, curv.shape[1] - 1):
            if ri not in (i-1, i, i+1) and rj not in (j-1, j, j+1):
                assert curv.values[ri, rj] == 0

    # border edges are all nans
    assert np.isnan(curv.values[0, :]).all()
    assert np.isnan(curv.values[-1, :]).all()
    assert np.isnan(curv.values[:, 0]).all()
    assert np.isnan(curv.values[:, -1]).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_curvature_gpu_equals_cpu():

    import cupy

    small_da = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    cpu = curvature(small_da, name='numpy_result')

    small_da_cupy = xr.DataArray(cupy.asarray(elevation),
                                 attrs={'res': (10.0, 10.0)})
    gpu = curvature(small_da_cupy, name='cupy_result')

    assert isinstance(gpu.data, cupy.ndarray)

    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_curvature_numpy_equals_dask():
    small_numpy_based_data_array = xr.DataArray(elevation,
                                                attrs={'res': (10.0, 10.0)})
    small_das_based_data_array = xr.DataArray(da.from_array(elevation,
                                              chunks=(3, 3)),
                                              attrs={'res': (10.0, 10.0)})

    numpy_curvature = curvature(small_numpy_based_data_array,
                                name='numpy_curvature')
    dask_curvature = curvature(small_das_based_data_array,
                               name='dask_curvature')
    assert isinstance(dask_curvature.data, da.Array)

    dask_curvature.data = dask_curvature.data.compute()

    assert np.isclose(numpy_curvature, dask_curvature, equal_nan=True).all()
