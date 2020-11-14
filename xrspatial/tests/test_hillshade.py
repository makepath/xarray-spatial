import pytest
import xarray as xr
import numpy as np

from xrspatial.utils import doesnt_have_cuda
from xrspatial import hillshade


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
#
# -----

data_random = np.random.random_sample((100, 100))
data_random_sparse = _do_sparse_array(data_random)
data_gaussian = _do_gaussian_array()


def test_hillshade():
    """
    Assert Simple Hillshade transfer function
    """
    da_gaussian = xr.DataArray(data_gaussian)
    da_gaussian_shade = hillshade(da_gaussian, name='hillshade_agg')
    assert da_gaussian_shade.dims == da_gaussian.dims
    assert da_gaussian_shade.attrs == da_gaussian.attrs
    assert da_gaussian_shade.name == 'hillshade_agg'
    for coord in da_gaussian.coords:
        assert np.all(da_gaussian_shade[coord] == da_gaussian[coord])
    assert da_gaussian_shade.mean() > 0
    assert da_gaussian_shade[60, 60] > 0


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_hillshade_gpu_equals_cpu():

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

    cpu = hillshade(small_da, name='aspect_agg', use_cuda=False)
    gpu = hillshade(small_da, name='aspect_agg', use_cuda=True)

    assert np.isclose(cpu, gpu, equal_nan=True).all()
