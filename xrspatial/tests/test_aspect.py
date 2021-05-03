import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial import aspect
from xrspatial.utils import doesnt_have_cuda
from xrspatial.utils import is_cupy_backed


INPUT_DATA = np.asarray([
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
    [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
    [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
    [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
    [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
    [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
    dtype=np.float32
)

QGIS_OUTPUT = np.asarray([
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [330.94687, 335.55496, 320.70786, 330.94464, 0., 0.],
    [333.43494, 333.43494, 329.03394, 341.56897, 0., 18.434948],
    [338.9621, 338.20062, 341.56506, 0., 0., 45.],
    [341.56506, 351.8699, 26.56505, 45., -1., 90.],
    [351.86676, 11.306906, 45., 45., 45., 108.431015]], dtype=np.float32
)


def test_numpy_equals_qgis():

    small_da = xr.DataArray(INPUT_DATA, attrs={'res': (10.0, 10.0)})
    xrspatial_aspect = aspect(small_da, name='numpy_aspect')

    # validate output attributes
    assert xrspatial_aspect.dims == small_da.dims
    assert xrspatial_aspect.attrs == small_da.attrs
    assert xrspatial_aspect.shape == small_da.shape
    assert xrspatial_aspect.name == 'numpy_aspect'
    for coord in small_da.coords:
        assert np.all(xrspatial_aspect[coord] == small_da[coord])

    # TODO: We shouldn't ignore edges!
    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_aspect.values[1:-1, 1:-1]
    qgis_vals = QGIS_OUTPUT[1:-1, 1:-1]

    # TODO: use np.is_close instead
    # set a tolerance of 1e-4
    # aspect is nan if nan input
    # aspect is invalid (nan) if slope equals 0
    # otherwise aspect are from 0 - 360
    assert np.isclose(xrspatial_vals, qgis_vals, equal_nan=True).all()


def test_numpy_equals_dask():
    small_numpy_based_data_array = xr.DataArray(
        INPUT_DATA, attrs={'res': (10.0, 10.0)}
    )
    small_dask_based_data_array = xr.DataArray(
        da.from_array(INPUT_DATA, chunks=(2, 2)), attrs={'res': (10.0, 10.0)}
    )

    numpy_result = aspect(small_numpy_based_data_array, name='numpy_result')
    dask_result = aspect(small_dask_based_data_array,
                         name='dask_result')

    assert isinstance(dask_result.data, da.Array)

    dask_result.data = dask_result.data.compute()
    assert np.isclose(numpy_result, dask_result, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_numpy_equals_cupy():

    import cupy

    small_da = xr.DataArray(INPUT_DATA, attrs={'res': (10.0, 10.0)})
    small_da_cupy = xr.DataArray(cupy.asarray(INPUT_DATA),
                                 attrs={'res': (10.0, 10.0)})

    # aspect by xrspatial
    cpu = aspect(small_da, name='aspect_agg')
    gpu = aspect(small_da_cupy, name='aspect_agg')

    assert isinstance(gpu.data, cupy.ndarray)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_cupy_equals_qgis():

    import cupy

    small_da = xr.DataArray(INPUT_DATA, attrs={'res': (10.0, 10.0)})
    small_da_cupy = xr.DataArray(cupy.asarray(INPUT_DATA),
                                 attrs={'res': (10.0, 10.0)})
    xrspatial_aspect = aspect(small_da_cupy, name='aspect_agg')

    # validate output attributes
    assert xrspatial_aspect.dims == small_da.dims
    assert xrspatial_aspect.attrs == small_da.attrs
    assert xrspatial_aspect.shape == small_da.shape
    assert xrspatial_aspect.name == 'aspect_agg'
    for coord in small_da.coords:
        assert np.all(xrspatial_aspect[coord] == small_da[coord])

    # TODO: We shouldn't ignore edges!
    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_aspect.values[1:-1, 1:-1]
    qgis_vals = QGIS_OUTPUT[1:-1, 1:-1]
    assert np.isclose(xrspatial_vals, qgis_vals, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def _numpy_equals_dask_cupy():

    # NOTE: Dask + GPU code paths don't currently work because of
    # dask casting cupy arrays to numpy arrays during
    # https://github.com/dask/dask/issues/4842

    import cupy

    cupy_data = cupy.asarray(INPUT_DATA)
    dask_cupy_data = da.from_array(cupy_data, chunks=(3, 3))

    small_da = xr.DataArray(INPUT_DATA, attrs={'res': (10.0, 10.0)})
    cpu = aspect(small_da, name='numpy_result')

    small_dask_cupy = xr.DataArray(dask_cupy_data, attrs={'res': (10.0, 10.0)})
    gpu = aspect(small_dask_cupy, name='cupy_result')

    assert is_cupy_backed(gpu)
    assert np.isclose(cpu, gpu, equal_nan=True).all()
