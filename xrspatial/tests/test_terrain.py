import dask.array as da
import numpy as np
import xarray as xr

from xrspatial import generate_terrain
from xrspatial.tests.general_checks import cuda_and_cupy_available
from xrspatial.utils import has_cuda_and_cupy


def create_test_arr(backend='numpy'):
    W = 50
    H = 50
    data = np.zeros((H, W), dtype=np.float32)
    raster = xr.DataArray(data, dims=['y', 'x'])

    if has_cuda_and_cupy() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend:
        raster.data = da.from_array(raster.data, chunks=(10, 10))

    return raster


def test_terrain_cpu():
    # vanilla numpy version
    data_numpy = create_test_arr()
    terrain_numpy = generate_terrain(data_numpy)

    # dask
    data_dask = create_test_arr(backend='dask')
    terrain_dask = generate_terrain(data_dask)
    assert isinstance(terrain_dask.data, da.Array)

    terrain_dask = terrain_dask.compute()
    np.testing.assert_allclose(terrain_numpy.data, terrain_dask.data, rtol=1e-05, atol=1e-07)


@cuda_and_cupy_available
def test_terrain_gpu():
    # vanilla numpy version
    data_numpy = create_test_arr()
    terrain_numpy = generate_terrain(data_numpy)

    # cupy
    data_cupy = create_test_arr(backend='cupy')
    terrain_cupy = generate_terrain(data_cupy)

    np.testing.assert_allclose(terrain_numpy.data, terrain_cupy.data.get(), rtol=1e-05, atol=1e-07)
