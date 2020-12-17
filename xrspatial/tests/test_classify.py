import pytest

import xarray as xr
import numpy as np

import dask.array as da

from xrspatial.utils import doesnt_have_cuda, is_cupy_backed
from xrspatial import equal_interval
from xrspatial import natural_breaks
from xrspatial import quantile
from xrspatial import reclassify


def test_reclassify_cpu():
    bins = [10, 20, 30]
    new_values = [1, 2, 3]

    n, m = 5, 5
    elevation = np.arange(n * m).reshape((n, m))
    # numpy
    numpy_agg = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    numpy_reclassify = reclassify(numpy_agg, bins=bins, new_values=new_values,
                                  name='numpy_reclassify')
    unique_elements, counts_elements = np.unique(numpy_reclassify.data,
                                                 return_counts=True)
    assert len(unique_elements) == 3

    # dask + numpy
    dask_numpy_agg = xr.DataArray(da.from_array(elevation, chunks=(3, 3)),
                                  attrs={'res': (10.0, 10.0)})
    dask_reclassify = reclassify(dask_numpy_agg, bins=bins,
                                 new_values=new_values, name='dask_reclassify')
    assert isinstance(dask_reclassify.data, da.Array)

    dask_reclassify.data = dask_reclassify.data.compute()
    assert np.isclose(numpy_reclassify, dask_reclassify, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_reclassify_cpu_equals_gpu():

    import cupy

    bins = [10, 20, 30]
    new_values = [1, 2, 3]

    n, m = 5, 5
    elevation = np.arange(n * m).reshape((n, m))

    # vanilla numpy version
    numpy_agg = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    cpu = reclassify(numpy_agg, name='numpy_result', bins=bins, new_values=new_values)

    # cupy
    cupy_agg = xr.DataArray(cupy.asarray(elevation), attrs={'res': (10.0, 10.0)})
    gpu = reclassify(cupy_agg, name='cupy_result', bins=bins, new_values=new_values)
    assert isinstance(gpu.data, cupy.ndarray)
    assert np.isclose(cpu, gpu, equal_nan=True).all()

    # dask + cupy
    dask_cupy_agg = xr.DataArray(cupy.asarray(elevation), attrs={'res': (10.0, 10.0)})
    dask_cupy_agg.data = da.from_array(dask_cupy_agg.data, chunks=(3, 3))
    dask_gpu = reclassify(dask_cupy_agg, name='dask_cupy_result',
                          bins=bins, new_values=new_values)
    assert isinstance(dask_gpu.data, da.Array) and is_cupy_backed(dask_gpu)

    dask_gpu.data = dask_gpu.data.compute()
    assert np.isclose(cpu, dask_gpu, equal_nan=True).all()


def test_quantile_cpu():
    k = 5
    n, m = 5, 5
    elevation = np.arange(n * m).reshape((n, m))
    numpy_agg = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})

    quantile_agg = quantile(numpy_agg, k=5)
    unique_elements, counts_elements = np.unique(quantile_agg.data,
                                                 return_counts=True)
    assert len(unique_elements) == k
    assert len(np.unique(counts_elements)) == 1


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_quantile_cpu_equals_gpu():

    import cupy

    k = 5
    n, m = 5, 5
    elevation = np.arange(n * m).reshape((n, m))

    # vanilla numpy version
    numpy_agg = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
    cpu = quantile(numpy_agg, k=k, name='numpy_result')

    # cupy
    cupy_agg = xr.DataArray(cupy.asarray(elevation), attrs={'res': (10.0, 10.0)})
    gpu = quantile(cupy_agg, k=k, name='cupy_result')

    assert isinstance(gpu.data, cupy.ndarray)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_natural_breaks():
    k = 5
    n, m = 4, 3
    agg = xr.DataArray(np.arange(n * m).reshape((n, m)), dims=['y', 'x'])
    agg['y'] = np.linspace(0, n, n)
    agg['x'] = np.linspace(0, m, m)

    natural_breaks_agg = natural_breaks(agg, k=5)

    # shape and other attributes remain the same
    assert agg.shape == natural_breaks_agg.shape
    assert agg.dims == natural_breaks_agg.dims
    assert agg.attrs == natural_breaks_agg.attrs
    for coord in agg.coords:
        assert np.all(agg[coord] == natural_breaks_agg[coord])

    unique_elements, counts_elements = np.unique(natural_breaks_agg.data,
                                                 return_counts=True)
    assert len(unique_elements) == k


def test_equal_interval():
    k = 4
    n, m = 4, 4
    agg = xr.DataArray(np.arange(n * m).reshape((n, m)), dims=['x', 'y'])
    agg['x'] = np.linspace(0, n, n)
    agg['y'] = np.linspace(0, m, m)

    equal_interval_agg = equal_interval(agg, k=4)
    assert equal_interval_agg is not None

    unique_elements, counts_elements = np.unique(equal_interval_agg.data,
                                                 return_counts=True)
    assert len(unique_elements) == k
    assert len(np.unique(counts_elements)) == 1


def test_small_equal_interval():
    k = 4
    n, m = 3, 2
    agg = xr.DataArray(np.arange(n * m).reshape((n, m)), dims=['x', 'y'])
    agg['x'] = np.linspace(0, n, n)
    agg['y'] = np.linspace(0, m, m)

    equal_interval_agg = equal_interval(agg, k=4)
    assert equal_interval_agg is not None

    unique_elements, counts_elements = np.unique(equal_interval_agg.data,
                                                 return_counts=True)
    assert len(unique_elements) == k
    assert len(np.unique(counts_elements)) == n-1
