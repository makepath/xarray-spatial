import pytest
import xarray as xr
import numpy as np

import dask.array as da

from xrspatial import slope
from xrspatial.utils import doesnt_have_cuda

from xrspatial.tests.general_checks import general_output_checks


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


def test_slope_against_qgis_cpu():

    # slope by xrspatial
    agg_numpy = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    xrspatial_slope_numpy = slope(agg_numpy, name='slope_numpy')
    general_output_checks(agg_numpy, xrspatial_slope_numpy)
    assert xrspatial_slope_numpy.name == 'slope_numpy'

    agg_dask = xr.DataArray(
        da.from_array(elevation, chunks=(3, 3)), attrs={'res': (10.0, 10.0)})
    xrspatial_slope_dask = slope(agg_dask, name='slope_dask')
    general_output_checks(agg_dask, xrspatial_slope_dask)
    assert xrspatial_slope_dask.name == 'slope_dask'

    # numpy and dask case produce same results
    np.testing.assert_allclose(
        xrspatial_slope_numpy.data,
        xrspatial_slope_dask.compute().data,
        equal_nan=True
    )

    # nan border edges
    xrspatial_edges = [
        xrspatial_slope_numpy.data[0, :],
        xrspatial_slope_numpy.data[-1, :],
        xrspatial_slope_numpy.data[:, 0],
        xrspatial_slope_numpy.data[:, -1],
    ]
    for edge in xrspatial_edges:
        np.testing.assert_allclose(
            edge, np.full(edge.shape, np.nan), equal_nan=True
        )

    # test against QGIS
    xrspatial_vals = xrspatial_slope_numpy.values[1:-1, 1:-1]
    qgis_vals = qgis_slope[1:-1, 1:-1]
    np.testing.assert_allclose(xrspatial_vals, qgis_vals, equal_nan=True)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_slope_against_qgis_gpu():

    import cupy

    # slope by xrspatial
    agg_cupy = xr.DataArray(
        cupy.asarray(elevation), attrs={'res': (10.0, 10.0)})
    xrspatial_slope_gpu = slope(agg_cupy, name='slope_cupy')
    general_output_checks(agg_cupy, xrspatial_slope_gpu)
    assert xrspatial_slope_gpu.name == 'slope_cupy'

    agg_numpy = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    xrspatial_slope_cpu = slope(agg_numpy, name='slope_numpy')

    # both cpu and gpu produce same results
    np.testing.assert_allclose(
        xrspatial_slope_cpu.data,
        xrspatial_slope_gpu.data.get(),
        equal_nan=True
    )

    # test against QGIS
    # nan border edges
    xrspatial_vals = xrspatial_slope_gpu.data[1:-1, 1:-1].get()
    qgis_vals = qgis_slope[1:-1, 1:-1]
    np.testing.assert_allclose(
        xrspatial_vals, qgis_vals, equal_nan=True
    )
