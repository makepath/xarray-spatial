import pytest
import numpy as np
import xarray as xr
from xrspatial import curvature


def test_curvature_invalid_input_raster():
    invalid_raster_type = np.array([0, 1, 2, 3])
    with pytest.raises(Exception) as e_info:
        curvature(invalid_raster_type)
        assert e_info

    invalid_raster_dtype = xr.DataArray(np.array([['cat', 'dog']]))
    with pytest.raises(Exception) as e_info:
        curvature(invalid_raster_dtype)
        assert e_info

    invalid_raster_shape = xr.DataArray(np.array([0, 0]))
    with pytest.raises(Exception) as e_info:
        curvature(invalid_raster_shape)
        assert e_info


def test_curvature_on_flat_surface():
    # flat surface
    test_arr1 = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    test_raster1 = xr.DataArray(test_arr1)
    curv = curvature(test_raster1)

    # output must be an xarray DataArray
    assert isinstance(curv, xr.DataArray)
    assert isinstance(curv.values, np.ndarray)
    # shape, dims, coords, attr preserved
    assert test_raster1.shape == curv.shape
    assert test_raster1.dims == curv.dims
    assert test_raster1.attrs == curv.attrs
    assert test_raster1.coords == curv.coords
    # curvature of a flat surface is all 0s
    assert np.unique(curv.values) == [0]


def test_curvature_on_convex_surface():
    # convex
    test_arr2 = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, -1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])

    test_raster2 = xr.DataArray(test_arr2)
    curv = curvature(test_raster2)

    # output must be an xarray DataArray
    assert isinstance(curv, xr.DataArray)
    assert isinstance(curv.values, np.ndarray)
    # shape, dims, coords, attr preserved
    assert test_raster2.shape == curv.shape
    assert test_raster2.dims == curv.dims
    assert test_raster2.attrs == curv.attrs
    assert test_raster2.coords == curv.coords

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
    for ri in range(curv.shape[0]):
        for rj in range(curv.shape[1]):
            if ri not in (i-1, i, i+1) and rj not in (j-1, j, j+1):
                assert curv.values[ri, rj] == 0


def test_curvature_on_concave_surface():
    # concave
    test_arr3 = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])

    test_raster3 = xr.DataArray(test_arr3)
    curv = curvature(test_raster3)

    # output must be an xarray DataArray
    assert isinstance(curv, xr.DataArray)
    assert isinstance(curv.values, np.ndarray)
    # shape, dims, coords, attr preserved
    assert test_raster3.shape == curv.shape
    assert test_raster3.dims == curv.dims
    assert test_raster3.attrs == curv.attrs
    assert test_raster3.coords == curv.coords

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
    for ri in range(curv.shape[0]):
        for rj in range(curv.shape[1]):
            if ri not in (i-1, i, i+1) and rj not in (j-1, j, j+1):
                assert curv.values[ri, rj] == 0
