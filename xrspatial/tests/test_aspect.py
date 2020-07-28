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
    # default name
    assert da_aspect.name == 'aspect'
    assert da_aspect.dims == da.dims
    assert da_aspect.attrs == da.attrs
    assert da.shape == da_aspect.shape
    for coord in da.coords:
        assert np.all(da[coord] == da_aspect[coord])
    assert pytest.approx(np.nanmax(da_aspect.data), .1) == 360.
    assert pytest.approx(np.nanmin(da_aspect.data), .1) == 0.


def test_aspect_against_qgis():
    # input data
    data = np.asarray([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                       [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
                       [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
                       [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
                       [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
                       [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
                      dtype=np.float32)
    small_da = xr.DataArray(data, attrs={'res': (10.0, 10.0)})

    # aspect by QGIS
    qgis_aspect = np.asarray([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [330.94687, 335.55496, 320.70786, 330.94464, 0., 0.],
        [333.43494, 333.43494, 329.03394, 341.56897, 0., 18.434948],
        [338.9621, 338.20062, 341.56506, 0., 0., 45.],
        [341.56506, 351.8699, 26.56505, 45., -1., 90.],
        [351.86676, 11.306906, 45., 45., 45., 108.431015]],
        dtype=np.float32)

    # aspect by xrspatial
    xrspatial_aspect = aspect(small_da, name='aspect_agg')

    # validate output attributes
    assert xrspatial_aspect.dims == small_da.dims
    assert xrspatial_aspect.attrs == small_da.attrs
    assert xrspatial_aspect.shape == small_da.shape
    assert xrspatial_aspect.name == 'aspect_agg'
    for coord in small_da.coords:
        assert np.all(xrspatial_aspect[coord] == small_da[coord])

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_aspect.values[1:-1, 1:-1]
    qgis_vals = qgis_aspect[1:-1, 1:-1]
    # set a tolerance of 1e-4
    # aspect is nan if nan input
    # aspect is invalid (nan) if slope equals 0
    # otherwise aspect are from 0 - 360
    assert ((np.isnan(xrspatial_vals) & np.isnan(qgis_vals)) | (
            np.isnan(xrspatial_vals) & (qgis_vals == -1)) | (
            abs(xrspatial_vals - qgis_vals) <= 1e-4)).all()

    assert (np.isnan(xrspatial_vals) | (
            (0 <= xrspatial_vals) & (xrspatial_vals <= 360))).all()
