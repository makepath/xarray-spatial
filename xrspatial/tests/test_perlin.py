import dask.array as da
import numpy as np
import xarray as xr

from xrspatial import perlin
from xrspatial.tests.general_checks import cuda_and_cupy_available, general_output_checks
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


def test_perlin_cpu():
    # vanilla numpy version
    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)
    general_output_checks(data_numpy, perlin_numpy)

    # dask
    data_dask = create_test_arr(backend='dask')
    perlin_dask = perlin(data_dask)
    general_output_checks(data_dask, perlin_dask)

    np.testing.assert_allclose(
        perlin_numpy.data, perlin_dask.data.compute(),
        rtol=1e-05, atol=1e-07, equal_nan=True
    )


@cuda_and_cupy_available
def test_perlin_gpu():
    # vanilla numpy version
    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)

    # cupy
    data_cupy = create_test_arr(backend='cupy')
    perlin_cupy = perlin(data_cupy)
    general_output_checks(data_cupy, perlin_cupy)
    np.testing.assert_allclose(
        perlin_numpy.data, perlin_cupy.data.get(),
        rtol=1e-05, atol=1e-07, equal_nan=True
    )
