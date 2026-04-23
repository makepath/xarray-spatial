import numpy as np
import pytest
import xarray as xr

from xrspatial import perlin
from xrspatial.tests.general_checks import cuda_and_cupy_available
from xrspatial.tests.general_checks import dask_array_available
from xrspatial.tests.general_checks import general_output_checks

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
        import dask.array as da
        raster.data = da.from_array(raster.data, chunks=(10, 10))

    return raster


def test_perlin_cpu():
    # vanilla numpy version
    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)
    general_output_checks(data_numpy, perlin_numpy)


@dask_array_available
def test_perlin_dask_cpu():
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
        rtol=1e-4, atol=1e-6, equal_nan=True
    )


@cuda_and_cupy_available
@dask_array_available
def test_perlin_dask_gpu():
    # numpy baseline
    data_numpy = create_test_arr()
    perlin_numpy = perlin(data_numpy)

    # cupy baseline
    data_cupy = create_test_arr(backend='cupy')
    perlin_cupy = perlin(data_cupy)

    # dask + cupy
    data_dask_cupy = create_test_arr(backend='dask+cupy')
    perlin_dask_cupy = perlin(data_dask_cupy)
    general_output_checks(data_dask_cupy, perlin_dask_cupy)

    np.testing.assert_allclose(
        perlin_numpy.data, perlin_dask_cupy.data.compute().get(),
        rtol=1e-4, atol=1e-4, equal_nan=True
    )
    np.testing.assert_allclose(
        perlin_cupy.data.get(), perlin_dask_cupy.data.compute().get(),
        rtol=1e-4, atol=1e-4, equal_nan=True
    )


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32],
)
def test_perlin_rejects_integer_dtype(dtype):
    # Regression for issue #1232: integer-dtyped input silently produced
    # INT_MIN everywhere because the float noise was written in place and
    # then normalized by a zero ptp.  We now raise ValueError instead.
    data = np.zeros((20, 20), dtype=dtype)
    raster = xr.DataArray(data, dims=['y', 'x'])
    with pytest.raises(ValueError, match="floating-point dtype"):
        perlin(raster)


def test_perlin_float64_input():
    # float64 should still work (not just float32).
    data = np.zeros((20, 20), dtype=np.float64)
    raster = xr.DataArray(data, dims=['y', 'x'])
    result = perlin(raster)
    assert result.dtype == np.float64
    assert np.isfinite(result.data).all()
    # Normalized to [0, 1]
    assert result.data.min() >= 0.0
    assert result.data.max() <= 1.0
