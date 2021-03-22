import pytest
import xarray as xr
import numpy as np

import dask.array as da

from xrspatial import slope
from xrspatial.utils import doesnt_have_cuda


# Test Data -----------------------------------------------------------------

'''
Notes:
------
The `elevation` data was run through QGIS slope function to
get values to compare against.  Xarray-Spatial currently handles
edges by padding with nan which is different than QGIS but acknowledged
'''

elevation = np.asarray([
    [1432.6542, 1432.4764, 1432.4764, 1432.1207, 1431.9429, np.nan],
    [1432.6542, 1432.6542, 1432.4764, 1432.2986, 1432.1207, np.nan],
    [1432.832, 1432.6542, 1432.4764, 1432.2986, 1432.1207, np.nan],
    [1432.832, 1432.6542, 1432.4764, 1432.4764, 1432.1207, np.nan],
    [1432.832, 1432.6542, 1432.6542, 1432.4764, 1432.2986, np.nan],
    [1432.832, 1432.6542, 1432.6542, 1432.4764, 1432.2986, np.nan],
    [1432.832, 1432.832, 1432.6542, 1432.4764, 1432.4764, np.nan]],
    dtype=np.float32
)

qgis_slope = np.asarray(
    [[0.8052942, 0.742317, 1.1390567, 1.3716657, np.nan, np.nan],
     [0.74258685, 0.742317, 1.0500116, 1.2082565, np.nan, np.nan],
     [0.56964326, 0.9002944, 0.9002944, 1.0502871, np.nan, np.nan],
     [0.5095078, 0.9003686, 0.742317, 1.1390567, np.nan, np.nan],
     [0.6494868, 0.64938396, 0.5692523, 1.0500116, np.nan, np.nan],
     [0.80557066, 0.56964326, 0.64914393, 0.9002944, np.nan, np.nan],
     [0.6494868, 0.56964326, 0.8052942, 0.742317, np.nan, np.nan]],
    dtype=np.float32)

elevation2 = np.asarray([
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [1584.8767, 1584.8767, 1585.0546, 1585.2324, 1585.2324, 1585.2324],
    [1585.0546, 1585.0546, 1585.2324, 1585.588, 1585.588, 1585.588],
    [1585.2324, 1585.4102, 1585.588, 1585.588, 1585.588, 1585.588],
    [1585.588, 1585.588, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
    [1585.7659, 1585.9437, 1585.7659, 1585.7659, 1585.7659, 1585.7659],
    [1585.9437, 1585.9437, 1585.9437, 1585.7659, 1585.7659, 1585.7659]],
    dtype=np.float32
)


def test_slope_against_qgis():

    small_da = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})

    # slope by xrspatial
    xrspatial_slope = slope(small_da, name='slope_agg')

    # validate output attributes
    assert xrspatial_slope.dims == small_da.dims
    assert xrspatial_slope.attrs == small_da.attrs
    assert xrspatial_slope.shape == small_da.shape
    assert xrspatial_slope.name == 'slope_agg'
    for coord in small_da.coords:
        assert np.all(xrspatial_slope[coord] == small_da[coord])

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_slope.values[1:-1, 1:-1]
    qgis_vals = qgis_slope[1:-1, 1:-1]
    assert (np.isclose(xrspatial_vals, qgis_vals, equal_nan=True).all() | (
                np.isnan(xrspatial_vals) & np.isnan(qgis_vals))).all()


@pytest.mark.skipif(doesnt_have_cuda(),
                    reason="CUDA Device not Available")
def test_slope_against_qgis_gpu():

    import cupy

    small_da = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    small_da_cupy = xr.DataArray(cupy.asarray(elevation),
                                 attrs={'res': (10.0, 10.0)})
    xrspatial_slope = slope(small_da_cupy, name='slope_cupy')

    # validate output attributes
    assert xrspatial_slope.dims == small_da.dims
    assert xrspatial_slope.attrs == small_da.attrs
    assert xrspatial_slope.shape == small_da.shape
    for coord in small_da.coords:
        assert np.all(xrspatial_slope[coord] == small_da[coord])

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_slope.values[1:-1, 1:-1]
    qgis_vals = qgis_slope[1:-1, 1:-1]
    assert (np.isclose(xrspatial_vals, qgis_vals, equal_nan=True).all() | (
                np.isnan(xrspatial_vals) & np.isnan(qgis_vals))).all()


@pytest.mark.skipif(doesnt_have_cuda(),
                    reason="CUDA Device not Available")
def test_slope_gpu_equals_cpu():

    import cupy

    small_da = xr.DataArray(elevation2, attrs={'res': (10.0, 10.0)})
    cpu = slope(small_da, name='numpy_result')

    small_da_cupy = xr.DataArray(cupy.asarray(elevation2),
                                 attrs={'res': (10.0, 10.0)})
    gpu = slope(small_da_cupy, name='cupy_result')
    assert isinstance(gpu.data, cupy.ndarray)

    assert np.isclose(cpu, gpu, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def _dask_cupy_equals_numpy_cpu():

    # NOTE: Dask + GPU code paths don't currently work because of
    # dask casting cupy arrays to numpy arrays during
    # https://github.com/dask/dask/issues/4842

    import cupy

    cupy_data = cupy.asarray(elevation2)
    dask_cupy_data = da.from_array(cupy_data, chunks=(3, 3))

    small_da = xr.DataArray(elevation2, attrs={'res': (10.0, 10.0)})
    cpu = slope(small_da, name='numpy_result')

    small_dask_cupy = xr.DataArray(dask_cupy_data,
                                   attrs={'res': (10.0, 10.0)})
    gpu = slope(small_dask_cupy, name='cupy_result')

    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_slope_numpy_equals_dask():
    small_numpy_based_data_array = xr.DataArray(elevation2,
                                                attrs={'res': (10.0, 10.0)})
    small_das_based_data_array = xr.DataArray(da.from_array(elevation2,
                                              chunks=(3, 3)),
                                              attrs={'res': (10.0, 10.0)})

    numpy_slope = slope(small_numpy_based_data_array, name='numpy_slope')
    dask_slope = slope(small_das_based_data_array, name='dask_slope')
    assert isinstance(dask_slope.data, da.Array)

    dask_slope.data = dask_slope.data.compute()

    assert np.isclose(numpy_slope, dask_slope, equal_nan=True).all()


def test_slope_with_dask_array():

    import dask.array as da

    data = da.from_array(elevation, chunks=(3, 3))
    small_da = xr.DataArray(data, attrs={'res': (10.0, 10.0)})

    # slope by xrspatial
    xrspatial_slope = slope(small_da, name='slope_agg')
    xrspatial_slope.data = xrspatial_slope.data.compute()

    # validate output attributes
    assert xrspatial_slope.dims == small_da.dims
    assert xrspatial_slope.attrs == small_da.attrs
    assert xrspatial_slope.shape == small_da.shape
    assert xrspatial_slope.name == 'slope_agg'
    for coord in small_da.coords:
        assert np.all(xrspatial_slope[coord] == small_da[coord])

    # validate output values
    # ignore border edges
    xrspatial_vals = xrspatial_slope.values[1:-1, 1:-1]
    qgis_vals = qgis_slope[1:-1, 1:-1]
    assert (np.isclose(xrspatial_vals, qgis_vals, equal_nan=True).all() | (
                np.isnan(xrspatial_vals) & np.isnan(qgis_vals))).all()
