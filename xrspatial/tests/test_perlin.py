import pytest

import numpy as np
import xarray as xr
import dask.array as da

from xrspatial.utils import has_cuda
from xrspatial.utils import doesnt_have_cuda
from xrspatial import perlin


def create_test_arr(backend='numpy'):
    W = 50
    H = 50
    data = np.zeros((H, W), dtype=np.float32)
    raster = xr.DataArray(data, dims=['y', 'x'])

    if has_cuda() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend:
        raster.data = da.from_array(raster.data, chunks=(10, 10))

    return raster


def test_perlin_cpu():
    # vanilla numpy version
    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)

    # dask
    data_dask = create_test_arr(backend='dask')
    perlin_dask = perlin(data_dask)
    assert isinstance(perlin_dask.data, da.Array)

    perlin_dask = perlin_dask.compute()
    assert np.isclose(
        perlin_numpy.data, perlin_dask.data,
        rtol=1e-05, atol=1e-07, equal_nan=True
    ).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_perlin_gpu():
    # vanilla numpy version
    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)

    # cupy
    data_cupy = create_test_arr(backend='cupy')
    perlin_cupy = perlin(data_cupy)

    assert np.isclose(
        perlin_numpy.data, perlin_cupy.data,
        rtol=1e-05, atol=1e-07, equal_nan=True
    ).all()
