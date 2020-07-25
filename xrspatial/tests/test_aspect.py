import xarray as xr
import numpy as np
import pytest

from xrspatial import aspect


def _do_sparse_array(data_array):
    import random
    indx = list(zip(*np.where(data_array)))
    pos = random.sample(range(data_array.size), data_array.size // 2)
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
    x_fac = -np.power(X - _mean, 2)
    y_fac = -np.power(Y - _mean, 2)
    gaussian = np.exp((x_fac + y_fac) / (2 * _sdev ** 2)) / (2.5 * _sdev)
    return gaussian


#
# -----

data_random = np.random.random_sample((100, 100))
data_random_sparse = _do_sparse_array(data_random)
data_gaussian = _do_gaussian_array()


def test_aspect_transfer_function():
    """
    Assert aspect transfer function
    """
    da = xr.DataArray(data_gaussian, dims=['y', 'x'], attrs={'res': 1})
    da_aspect = aspect(da)
    assert da_aspect.dims == da.dims
    assert da_aspect.attrs == da.attrs
    assert da.shape == da_aspect.shape
    for coord in da.coords:
        assert np.all(da[coord] == da_aspect[coord])
    assert pytest.approx(da_aspect.data.max(), .1) == 360.
    assert pytest.approx(da_aspect.data.min(), .1) == 0.


def test_aspect_against_qgis():
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

    # aspect by QGIS
    qgis_aspect = np.asarray(
        [[71.56505, 59.033928, 63.436916, 68.19859, np.nan, np.nan],
         [59.038555, 59.033928, 75.96491, 71.56505, np.nan, np.nan],
         [63.434948, 81.86675, 81.86675, 75.96375, np.nan, np.nan],
         [90., 81.87305, 59.033928, 63.436916, np.nan, np.nan],
         [78.69007, 78.69612, 63.434948, 75.96491, np.nan, np.nan],
         [71.56505, 63.434948, 78.68401, 81.86675, np.nan, np.nan],
         [78.69007, 63.434948, 71.56505, 59.033928, np.nan, np.nan]],
        dtype=np.float32)

    # aspect by xrspatial
    xrspatial_aspect = aspect(small_da)

    # validate output attributes
    assert xrspatial_aspect.dims == small_da.dims
    assert xrspatial_aspect.attrs == small_da.attrs
    assert xrspatial_aspect.shape == small_da.shape
    for coord in small_da.coords:
        assert np.all(xrspatial_aspect[coord] == small_da[coord])

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_aspect.values[1:-1, 1:-1]
    qgis_vals = qgis_aspect[1:-1, 1:-1]
    assert ((xrspatial_vals == qgis_vals) | (
            np.isnan(xrspatial_vals) & np.isnan(qgis_vals))).all()
