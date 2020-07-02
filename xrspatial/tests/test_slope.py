import pytest
import xarray as xr
import numpy as np

from xrspatial import slope


def _do_sparse_array(data_array):
    import random
    indx = list(zip(*np.where(data_array)))
    pos = random.sample(range(data_array.size), data_array.size//2)
    indx = np.asarray(indx)[pos]
    r = indx[:, 0]
    c = indx[:, 1]
    data_half = data_array.copy()
    data_half[r, c] = 0
    return data_half


def _do_gaussian_array():
    _x = np.linspace(0, 50, 101)
    _y = _x.copy()
    _mean = 25
    _sdev = 5
    X, Y = np.meshgrid(_x, _y, sparse=True)
    x_fac = -np.power(X-_mean, 2)
    y_fac = -np.power(Y-_mean, 2)
    gaussian = np.exp((x_fac+y_fac)/(2*_sdev**2)) / (2.5*_sdev)
    return gaussian


data_random = np.random.random_sample((100, 100))
data_random_sparse = _do_sparse_array(data_random)
data_gaussian = _do_gaussian_array()


def test_invalid_res_attr():
    """
    Assert 'res' attribute of input xarray
    """
    da = xr.DataArray(data_gaussian, attrs={})
    with pytest.raises(ValueError) as e_info:
        da_slope = slope(da)
        assert e_info

    da = xr.DataArray(data_gaussian, attrs={'res': 'any_string'})
    with pytest.raises(ValueError) as e_info:
        da_slope = slope(da)
        assert e_info

    da = xr.DataArray(data_gaussian, attrs={'res': ('str_tuple', 'str_tuple')})
    with pytest.raises(ValueError) as e_info:
        da_slope = slope(da)
        assert e_info

    da = xr.DataArray(data_gaussian, attrs={'res': (1, 2, 3)})
    with pytest.raises(ValueError) as e_info:
        da_slope = slope(da)
        assert e_info


def test_slope_transfer_function():
    """
    Assert slope transfer function
    """
    da = xr.DataArray(data_gaussian, attrs={'res': 1})
    da_slope = slope(da)
    assert da_slope.dims == da.dims
    assert da_slope.coords == da.coords
    assert da_slope.attrs == da.attrs
    assert da.shape == da_slope.shape

    assert da_slope.sum() > 0

    # In the middle of the array, there is the maximum of the gaussian;
    # And there the slope must be zero.
    _imax = np.where(da == da.max())
    assert da_slope[_imax] == 0

    # same result when cellsize_x = cellsize_y = 1
    da = xr.DataArray(data_gaussian, attrs={'res': (1.0, 1.0)})
    da_slope = slope(da)
    assert da_slope.dims == da.dims
    assert da_slope.coords == da.coords
    assert da_slope.attrs == da.attrs
    assert da.shape == da_slope.shape

    assert da_slope.sum() > 0

    # In the middle of the array, there is the maximum of the gaussian;
    # And there the slope must be zero.
    _imax = np.where(da == da.max())
    assert da_slope[_imax] == 0


def test_slope_against_qgis():
    # input data
    data = np.asarray(
        [[1432.6542, 1432.4764, 1432.4764, 1432.1207, 1431.9429, np.nan],
         [1432.6542, 1432.6542, 1432.4764, 1432.2986, 1432.1207, np.nan],
         [1432.832, 1432.6542, 1432.4764, 1432.2986, 1432.1207, np.nan],
         [1432.832, 1432.6542, 1432.4764, 1432.4764, 1432.1207, np.nan],
         [1432.832, 1432.6542, 1432.6542, 1432.4764, 1432.2986, np.nan],
         [1432.832, 1432.6542, 1432.6542, 1432.4764, 1432.2986, np.nan],
         [1432.832, 1432.832, 1432.6542, 1432.4764, 1432.4764, np.nan]],
        dtype=np.float32)
    small_da = xr.DataArray(data, attrs={'res': (10.0, 10.0)})

    # slope by QGIS
    qgis_slope = np.asarray(
        [[0.8052942, 0.742317, 1.1390567, 1.3716657, np.nan, np.nan],
         [0.74258685, 0.742317, 1.0500116, 1.2082565, np.nan, np.nan],
         [0.56964326, 0.9002944, 0.9002944, 1.0502871, np.nan, np.nan],
         [0.5095078, 0.9003686, 0.742317, 1.1390567, np.nan, np.nan],
         [0.6494868, 0.64938396, 0.5692523, 1.0500116, np.nan, np.nan],
         [0.80557066, 0.56964326, 0.64914393, 0.9002944, np.nan, np.nan],
         [0.6494868, 0.56964326, 0.8052942, 0.742317, np.nan, np.nan]],
        dtype=np.float32)

    # slope by xrspatial
    xrspatial_slope = slope(small_da)

    # validate output attributes
    assert xrspatial_slope.dims == small_da.dims
    assert xrspatial_slope.coords == small_da.coords
    assert xrspatial_slope.attrs == small_da.attrs
    assert xrspatial_slope.shape == small_da.shape

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_slope.values[1:-1, 1:-1]
    qgis_vals = qgis_slope[1:-1, 1:-1]
    assert ((xrspatial_vals == qgis_vals) | (
                np.isnan(xrspatial_vals) & np.isnan(qgis_vals))).all()
