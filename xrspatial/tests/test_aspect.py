import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial import aspect
from xrspatial.utils import doesnt_have_cuda

from xrspatial.tests.general_checks import general_output_checks


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

    general_output_checks(small_da, xrspatial_aspect)
    assert xrspatial_aspect.name == 'numpy_aspect'

    # validate output values
    xrspatial_vals = xrspatial_aspect.data[1:-1, 1:-1]
    qgis_vals = QGIS_OUTPUT[1:-1, 1:-1]
    # aspect is nan if nan input
    # aspect is invalid (-1) if slope equals 0
    # otherwise aspect are from 0 - 360
    np.testing.assert_allclose(xrspatial_vals, qgis_vals, equal_nan=True)

    # nan edge effect
    xrspatial_edges = [
        xrspatial_aspect.data[0, :],
        xrspatial_aspect.data[-1, :],
        xrspatial_aspect.data[:, 0],
        xrspatial_aspect.data[:, -1],
    ]
    for edge in xrspatial_edges:
        np.testing.assert_allclose(
            edge, np.full(edge.shape, np.nan), equal_nan=True
        )


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
    general_output_checks(small_dask_based_data_array, dask_result)
    np.testing.assert_allclose(
        numpy_result.data, dask_result.data.compute(), equal_nan=True)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_cpu_equals_gpu():

    import cupy

    small_da = xr.DataArray(INPUT_DATA, attrs={'res': (10.0, 10.0)})
    small_da_cupy = xr.DataArray(cupy.asarray(INPUT_DATA),
                                 attrs={'res': (10.0, 10.0)})

    # aspect by xrspatial
    cpu = aspect(small_da, name='aspect_agg')
    gpu = aspect(small_da_cupy, name='aspect_agg')
    general_output_checks(small_da_cupy, gpu)
    np.testing.assert_allclose(cpu.data, gpu.data.get(), equal_nan=True)
