import pytest
import numpy as np
import xarray as xr
from xrspatial.curvature import curvature


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
