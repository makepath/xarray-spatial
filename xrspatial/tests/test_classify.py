import pytest

import xarray as xr
import numpy as np

import dask.array as da

from xrspatial.utils import doesnt_have_cuda, is_cupy_backed
from xrspatial import equal_interval
from xrspatial import natural_breaks
from xrspatial import quantile
from xrspatial import reclassify

from xrspatial.tests._crs import _add_EPSG4326_crs_to_da


n, m = 5, 5
elevation = np.arange(n * m).reshape((n, m))
numpy_agg = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})
dask_numpy_agg = xr.DataArray(da.from_array(elevation, chunks=(3, 3)),
                              attrs={'res': (10.0, 10.0)})


def test_reclassify_cpu():
    bins = [10, 20, 30]
    new_values = [1, 2, 3]

    # numpy

    # add crs for tests
    numpy_agg_crs = _add_EPSG4326_crs_to_da(numpy_agg)

    numpy_reclassify = reclassify(numpy_agg_crs, bins=bins, new_values=new_values,
                                  name='numpy_reclassify')
    unique_elements, counts_elements = np.unique(numpy_reclassify.data,
                                                 return_counts=True)
    assert len(unique_elements) == 3

    # crs tests
    assert numpy_reclassify.attrs == numpy_agg_crs.attrs
    for coord in numpy_agg_crs.coords:
        assert np.all(numpy_reclassify[coord] == numpy_agg_crs[coord])

    # dask + numpy
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

    # vanilla numpy version
    cpu = reclassify(numpy_agg,
                     name='numpy_result',
                     bins=bins,
                     new_values=new_values)

    # cupy
    cupy_agg = xr.DataArray(cupy.asarray(elevation),
                            attrs={'res': (10.0, 10.0)})
    gpu = reclassify(cupy_agg,
                     name='cupy_result',
                     bins=bins,
                     new_values=new_values)
    assert isinstance(gpu.data, cupy.ndarray)
    assert np.isclose(cpu, gpu, equal_nan=True).all()

    # dask + cupy
    dask_cupy_agg = xr.DataArray(cupy.asarray(elevation),
                                 attrs={'res': (10.0, 10.0)})
    dask_cupy_agg.data = da.from_array(dask_cupy_agg.data, chunks=(3, 3))
    dask_gpu = reclassify(dask_cupy_agg, name='dask_cupy_result',
                          bins=bins, new_values=new_values)
    assert isinstance(dask_gpu.data, da.Array) and is_cupy_backed(dask_gpu)

    dask_gpu.data = dask_gpu.data.compute()
    assert np.isclose(cpu, dask_gpu, equal_nan=True).all()


def test_quantile_cpu():
    k = 5

    # numpy

    # add crs for tests
    numpy_agg_crs = _add_EPSG4326_crs_to_da(numpy_agg)

    numpy_quantile = quantile(numpy_agg_crs, k=k)

    unique_elements, counts_elements = np.unique(numpy_quantile.data,
                                                 return_counts=True)
    assert isinstance(numpy_quantile.data, np.ndarray)
    assert len(unique_elements) == k
    assert len(np.unique(counts_elements)) == 1
    assert np.unique(counts_elements)[0] == 5

    # crs tests
    assert numpy_quantile.attrs == numpy_agg_crs.attrs
    for coord in numpy_agg_crs.coords:
        assert np.all(numpy_quantile[coord] == numpy_agg_crs[coord])

    # dask + numpy
    dask_quantile = quantile(dask_numpy_agg, k=k)
    assert isinstance(dask_quantile.data, da.Array)

    #     Note that dask's percentile algorithm is
    #     approximate, while numpy's is exact.
    #     This may cause some differences between
    #     results of vanilla numpy and
    #     dask version of the input agg.
    #     https://github.com/dask/dask/issues/3099
    #     This assertion may fail
    # dask_quantile = dask_quantile.compute()
    # assert np.isclose(numpy_quantile, dask_quantile, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_quantile_cpu_equals_gpu():

    import cupy

    k = 5

    # vanilla numpy version
    cpu = quantile(numpy_agg, k=k, name='numpy_result')

    # cupy
    cupy_agg = xr.DataArray(cupy.asarray(elevation),
                            attrs={'res': (10.0, 10.0)})
    gpu = quantile(cupy_agg, k=k, name='cupy_result')

    assert isinstance(gpu.data, cupy.ndarray)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_natural_breaks_cpu():
    k = 5

    # vanilla numpy

    # add crs for tests
    numpy_agg_crs = _add_EPSG4326_crs_to_da(numpy_agg)

    numpy_natural_breaks = natural_breaks(numpy_agg_crs, k=k)

    # shape and other attributes remain the same, as well as coords, including crs
    assert numpy_agg_crs.shape == numpy_natural_breaks.shape
    assert numpy_agg_crs.dims == numpy_natural_breaks.dims
    assert numpy_agg_crs.attrs == numpy_natural_breaks.attrs
    for coord in numpy_agg_crs.coords:
        assert np.all(numpy_agg_crs[coord] == numpy_natural_breaks[coord])

    unique_elements, counts_elements = np.unique(numpy_natural_breaks.data,
                                                 return_counts=True)
    assert len(unique_elements) == k


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_natural_breaks_cpu_equals_gpu():

    import cupy

    k = 5

    # vanilla numpy version
    cpu = natural_breaks(numpy_agg, k=k, name='numpy_result')

    # cupy
    cupy_agg = xr.DataArray(cupy.asarray(elevation),
                            attrs={'res': (10.0, 10.0)})
    gpu = natural_breaks(cupy_agg, k=k, name='cupy_result')

    assert isinstance(gpu.data, cupy.ndarray)
    assert np.isclose(cpu, gpu, equal_nan=True).all()


def test_equal_interval_cpu():
    k = 5
    # numpy

    # add crs for tests
    numpy_agg_crs = _add_EPSG4326_crs_to_da(numpy_agg)

    numpy_ei = equal_interval(numpy_agg_crs, k=5)

    unique_elements, counts_elements = np.unique(numpy_ei.data,
                                                 return_counts=True)
    assert isinstance(numpy_ei.data, np.ndarray)
    assert len(unique_elements) == k

    # crs tests
    assert numpy_ei.attrs == numpy_agg_crs.attrs
    for coord in numpy_agg_crs.coords:
        assert np.all(numpy_ei[coord] == numpy_agg_crs[coord])

    # dask + numpy
    dask_ei = equal_interval(dask_numpy_agg, k=k, name='dask_reclassify')
    assert isinstance(dask_ei.data, da.Array)

    dask_ei.data = dask_ei.data.compute()
    assert np.isclose(numpy_ei, dask_ei, equal_nan=True).all()


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_equal_interval_cpu_equals_gpu():

    import cupy

    k = 5

    # numpy
    cpu = equal_interval(numpy_agg, k=k)

    # cupy
    cupy_agg = xr.DataArray(cupy.asarray(elevation),
                            attrs={'res': (10.0, 10.0)})
    gpu = equal_interval(cupy_agg, k=k)
    assert isinstance(gpu.data, cupy.ndarray)

    assert np.isclose(cpu, gpu, equal_nan=True).all()
