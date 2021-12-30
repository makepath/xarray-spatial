import pytest

import xarray as xr
import numpy as np

import dask.array as da

from xrspatial.utils import doesnt_have_cuda
from xrspatial import equal_interval
from xrspatial import natural_breaks
from xrspatial import quantile
from xrspatial import reclassify

from xrspatial.tests.general_checks import general_output_checks


elevation = np.array([
    [1.,  2.,  3.,  4., np.nan],
    [5.,  6.,  7.,  8.,  9.],
    [10., 11., 12., 13., 14.],
    [15., 16., 17., 18., np.inf],
])

h, w = elevation.shape
ys = np.arange(h)
xs = np.arange(w)

numpy_agg = xr.DataArray(
    elevation, dims=['y', 'x'], attrs={'res': (10.0, 10.0)}
)
numpy_agg['y'] = ys
numpy_agg['x'] = xs

dask_numpy_agg = xr.DataArray(
    da.from_array(elevation, chunks=(3, 3)),
    dims=['y', 'x'],
    attrs={'res': (10.0, 10.0)}
)
dask_numpy_agg['y'] = ys
dask_numpy_agg['x'] = xs


def test_reclassify_cpu():
    bins = [10, 15, np.inf]
    new_values = [1, 2, 3]
    expected_results = np.asarray([
        [1., 1., 1., 1., np.nan],
        [1., 1., 1., 1., 1.],
        [1., 2., 2., 2., 2.],
        [2., 3., 3., 3., 3.]
    ])
    # numpy
    numpy_reclassify = reclassify(
        numpy_agg, bins=bins, new_values=new_values, name='numpy_reclassify'
    )
    general_output_checks(numpy_agg, numpy_reclassify, expected_results)

    # dask + numpy
    dask_reclassify = reclassify(
        dask_numpy_agg, bins=bins,
        new_values=new_values, name='dask_reclassify'
    )
    general_output_checks(dask_numpy_agg, dask_reclassify, expected_results)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_reclassify_cpu_equals_gpu():

    import cupy

    bins = [10, 15, np.inf]
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
    general_output_checks(cupy_agg, gpu)
    np.testing.assert_allclose(cpu.data, gpu.data.get(), equal_nan=True)

    # dask + cupy
    dask_cupy_agg = xr.DataArray(
        cupy.asarray(elevation), attrs={'res': (10.0, 10.0)})
    dask_cupy_agg.data = da.from_array(dask_cupy_agg.data, chunks=(3, 3))
    dask_gpu = reclassify(
        dask_cupy_agg, name='dask_cupy_result',
        bins=bins, new_values=new_values
    )
    general_output_checks(dask_cupy_agg, dask_gpu)
    dask_gpu.data = dask_gpu.data.compute()
    np.testing.assert_allclose(cpu.data, dask_gpu.data.get(), equal_nan=True)


def test_quantile_cpu():
    k = 5
    expected_results = np.asarray([
        [0., 0., 0., 0., np.nan],
        [1., 1., 1., 2., 2.],
        [2., 2., 3., 3., 3.],
        [4., 4., 4., 4., np.nan]
    ])
    # numpy
    numpy_quantile = quantile(numpy_agg, k=k)
    general_output_checks(numpy_agg, numpy_quantile, expected_results)

    # dask + numpy
    dask_quantile = quantile(dask_numpy_agg, k=k)
    general_output_checks(dask_numpy_agg, dask_quantile)
    dask_quantile = dask_quantile.compute()
    unique_elements = np.unique(
        dask_quantile.data[np.isfinite(dask_quantile.data)]
    )
    assert len(unique_elements) == k

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
    cupy_agg = xr.DataArray(
        cupy.asarray(elevation), attrs={'res': (10.0, 10.0)})
    gpu = quantile(cupy_agg, k=k, name='cupy_result')
    general_output_checks(cupy_agg, gpu)
    np.testing.assert_allclose(cpu.data, gpu.data.get(), equal_nan=True)


def test_natural_breaks_cpu():
    k = 5
    expected_results = np.asarray([
        [0., 0., 0., 1., np.nan],
        [1., 1., 2., 2., 2.],
        [2., 3., 3., 3., 3.],
        [4., 4., 4., 4., np.nan]
    ])
    # vanilla numpy
    numpy_natural_breaks = natural_breaks(numpy_agg, k=k)
    general_output_checks(numpy_agg, numpy_natural_breaks, expected_results)


def test_natural_breaks_cpu_deterministic():
    results = []
    elevation = np.arange(100).reshape(10, 10)
    agg = xr.DataArray(elevation, attrs={'res': (10.0, 10.0)})

    k = 5
    numIters = 3
    for i in range(numIters):
        # vanilla numpy
        numpy_natural_breaks = natural_breaks(agg, k=k)
        general_output_checks(agg, numpy_natural_breaks)
        unique_elements = np.unique(
            numpy_natural_breaks.data[np.isfinite(numpy_natural_breaks.data)]
        )
        assert len(unique_elements) == k
        results.append(numpy_natural_breaks)
    # Check that the code is deterministic.
    # Multiple runs on same data should produce same results
    for i in range(numIters-1):
        np.testing.assert_allclose(
            results[i].data, results[i+1].data, equal_nan=True
        )


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
    general_output_checks(cupy_agg, gpu)
    np.testing.assert_allclose(cpu.data, gpu.data.get(), equal_nan=True)


def test_equal_interval_cpu():
    k = 3
    expected_results = np.asarray([
        [0., 0., 0., 0., np.nan],
        [0., 0., 1., 1., 1.],
        [1., 1., 1., 2., 2.],
        [2., 2., 2., 2., np.nan]
    ])

    # numpy
    numpy_ei = equal_interval(numpy_agg, k=k)
    general_output_checks(numpy_agg, numpy_ei, expected_results)

    # dask + numpy
    dask_ei = equal_interval(dask_numpy_agg, k=k, name='dask_reclassify')
    general_output_checks(dask_numpy_agg, dask_ei)
    dask_ei = dask_ei.compute()
    np.testing.assert_allclose(numpy_ei, dask_ei, equal_nan=True)


@pytest.mark.skipif(doesnt_have_cuda(), reason="CUDA Device not Available")
def test_equal_interval_cpu_equals_gpu():

    import cupy

    k = 3
    # numpy
    cpu = equal_interval(numpy_agg, k=k)
    # cupy
    cupy_agg = xr.DataArray(
        cupy.asarray(elevation), attrs={'res': (10.0, 10.0)})
    gpu = equal_interval(cupy_agg, k=k)
    general_output_checks(cupy_agg, gpu)
    np.testing.assert_allclose(cpu.data, gpu.data.get(), equal_nan=True)
