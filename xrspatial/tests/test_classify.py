import numpy as np
import pytest
import xarray as xr

from xrspatial import binary, equal_interval, natural_breaks, quantile, reclassify
from xrspatial.tests.general_checks import (create_test_raster, cuda_and_cupy_available,
                                            general_output_checks)


def input_data(backend='numpy'):
    elevation = np.array([
        [-np.inf,  2.,  3.,  4., np.nan],
        [5.,  6.,  7.,  8.,  9.],
        [10., 11., 12., 13., 14.],
        [15., 16., 17., 18., np.inf],
    ])
    raster = create_test_raster(elevation, backend)
    return raster


@pytest.fixture
def result_binary():
    values = [1, 2, 3]
    expected_result = np.asarray([
        [np.nan, 1, 1, 0, np.nan],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, np.nan]
    ], dtype=np.float32)
    return values, expected_result


def test_binary_numpy(result_binary):
    values, expected_result = result_binary
    numpy_agg = input_data()
    numpy_result = binary(numpy_agg, values)
    general_output_checks(numpy_agg, numpy_result, expected_result)


def test_binary_dask_numpy(result_binary):
    values, expected_result = result_binary
    dask_agg = input_data(backend='dask')
    dask_result = binary(dask_agg, values)
    general_output_checks(dask_agg, dask_result, expected_result)


@cuda_and_cupy_available
def test_binary_cupy(result_binary):
    values, expected_result = result_binary
    cupy_agg = input_data(backend='cupy')
    cupy_result = binary(cupy_agg, values)
    general_output_checks(cupy_agg, cupy_result, expected_result)


@cuda_and_cupy_available
def test_binary_dask_cupy(result_binary):
    values, expected_result = result_binary
    dask_cupy_agg = input_data(backend='dask+cupy')
    dask_cupy_result = binary(dask_cupy_agg, values)
    general_output_checks(dask_cupy_agg, dask_cupy_result, expected_result)


@pytest.fixture
def result_reclassify():
    bins = [10, 15, np.inf]
    new_values = [1, 2, 3]
    expected_result = np.asarray([
        [np.nan, 1., 1., 1., np.nan],
        [1., 1., 1., 1., 1.],
        [1., 2., 2., 2., 2.],
        [2., 3., 3., 3., np.nan]
    ], dtype=np.float32)
    return bins, new_values, expected_result


def test_reclassify_numpy_mismatch_length():
    bins = [10]
    new_values = [1, 2, 3]
    numpy_agg = input_data()
    msg = 'bins and new_values mismatch. Should have same length.'
    with pytest.raises(ValueError, match=msg):
        reclassify(numpy_agg, bins, new_values)


def test_reclassify_numpy(result_reclassify):
    bins, new_values, expected_result = result_reclassify
    numpy_agg = input_data()
    numpy_result = reclassify(numpy_agg, bins=bins, new_values=new_values)
    general_output_checks(numpy_agg, numpy_result, expected_result, verify_dtype=True)


def test_reclassify_dask_numpy(result_reclassify):
    bins, new_values, expected_result = result_reclassify
    dask_agg = input_data(backend='dask')
    dask_result = reclassify(dask_agg, bins=bins, new_values=new_values)
    general_output_checks(dask_agg, dask_result, expected_result, verify_dtype=True)


@cuda_and_cupy_available
def test_reclassify_cupy(result_reclassify):
    bins, new_values, expected_result = result_reclassify
    cupy_agg = input_data(backend='cupy')
    cupy_result = reclassify(cupy_agg, bins=bins, new_values=new_values)
    general_output_checks(cupy_agg, cupy_result, expected_result, verify_dtype=True)


@cuda_and_cupy_available
def test_reclassify_dask_cupy(result_reclassify):
    bins, new_values, expected_result = result_reclassify
    dask_cupy_agg = input_data(backend='dask+cupy')
    dask_cupy_result = reclassify(dask_cupy_agg, bins=bins, new_values=new_values)
    general_output_checks(dask_cupy_agg, dask_cupy_result, expected_result, verify_dtype=True)


@pytest.fixture
def result_quantile():
    k = 5
    expected_result = np.asarray([
        [np.nan, 0., 0., 0., np.nan],
        [0., 1., 1., 1., 2.],
        [2., 2., 3., 3., 3.],
        [4., 4., 4., 4., np.nan]
    ], dtype=np.float32)
    return k, expected_result


def test_quantile_not_enough_unique_values():
    agg = input_data()
    n_uniques = np.isfinite(agg.data).sum()
    k = n_uniques + 1
    result_quantile = quantile(agg, k=k)
    n_uniques_result = np.isfinite(result_quantile.data).sum()
    np.testing.assert_allclose(n_uniques_result, n_uniques)


def test_quantile_numpy(result_quantile):
    k, expected_result = result_quantile
    numpy_agg = input_data()
    numpy_quantile = quantile(numpy_agg, k=k)
    general_output_checks(numpy_agg, numpy_quantile, expected_result, verify_dtype=True)


def test_quantile_dask_numpy(result_quantile):
    #     Note that dask's percentile algorithm is
    #     approximate, while numpy's is exact.
    #     This may cause some differences between
    #     results of vanilla numpy and
    #     dask version of the input agg.
    #     https://github.com/dask/dask/issues/3099

    dask_numpy_agg = input_data('dask+numpy')
    k, expected_result = result_quantile
    dask_quantile = quantile(dask_numpy_agg, k=k)
    general_output_checks(dask_numpy_agg, dask_quantile)
    dask_quantile = dask_quantile.compute()
    unique_elements = np.unique(
        dask_quantile.data[np.isfinite(dask_quantile.data)]
    )
    assert len(unique_elements) == k


@cuda_and_cupy_available
def test_quantile_cupy(result_quantile):
    k, expected_result = result_quantile
    cupy_agg = input_data('cupy')
    cupy_result = quantile(cupy_agg, k=k)
    general_output_checks(cupy_agg, cupy_result, expected_result, verify_dtype=True)


@pytest.fixture
def result_natural_breaks():
    k = 5
    expected_result = np.asarray([
        [np.nan, 0., 0., 0., np.nan],
        [1., 1., 1., 2., 2.],
        [2., 3., 3., 3., 3.],
        [4., 4., 4., 4., np.nan]
    ], dtype=np.float32)
    return k, expected_result


@pytest.fixture
def result_natural_breaks_num_sample():
    k = 5
    num_sample = 8
    expected_result = np.asarray([
        [np.nan, 0., 0., 0., np.nan],
        [0., 1., 1., 1., 2.],
        [2., 3., 3., 3., 3.],
        [4., 4., 4., 4., np.nan]
    ], dtype=np.float32)
    return k, num_sample, expected_result


def test_natural_breaks_not_enough_unique_values():
    agg = input_data()
    n_uniques = np.isfinite(agg.data).sum()
    k = n_uniques + 1
    result_natural_breaks = natural_breaks(agg, k=k)
    n_uniques_result = np.isfinite(result_natural_breaks.data).sum()
    np.testing.assert_allclose(n_uniques_result, n_uniques)


def test_natural_breaks_numpy(result_natural_breaks):
    numpy_agg = input_data()
    k, expected_result = result_natural_breaks
    numpy_natural_breaks = natural_breaks(numpy_agg, k=k)
    general_output_checks(numpy_agg, numpy_natural_breaks, expected_result, verify_dtype=True)


def test_natural_breaks_numpy_num_sample(result_natural_breaks_num_sample):
    numpy_agg = input_data()
    k, num_sample, expected_result = result_natural_breaks_num_sample
    numpy_natural_breaks = natural_breaks(numpy_agg, k=k, num_sample=num_sample)
    general_output_checks(numpy_agg, numpy_natural_breaks, expected_result, verify_dtype=True)


def test_natural_breaks_cpu_deterministic():
    results = []
    elevation = np.arange(100).reshape(10, 10)
    agg = xr.DataArray(elevation)

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


@pytest.fixture
def result_equal_interval():
    k = 3
    expected_result = np.asarray([
        [np.nan, 0., 0., 0., np.nan],
        [0., 0., 0., 1., 1.],
        [1., 1., 1., 2., 2.],
        [2., 2., 2., 2., np.nan]
    ], dtype=np.float32)
    return k, expected_result


def test_equal_interval_numpy(result_equal_interval):
    k, expected_result = result_equal_interval
    numpy_agg = input_data('numpy')
    numpy_result = equal_interval(numpy_agg, k=k)
    general_output_checks(numpy_agg, numpy_result, expected_result, verify_dtype=True)


def test_equal_interval_dask_numpy(result_equal_interval):
    k, expected_result = result_equal_interval
    dask_agg = input_data('dask+numpy')
    dask_numpy_result = equal_interval(dask_agg, k=k)
    general_output_checks(dask_agg, dask_numpy_result, expected_result, verify_dtype=True)


@cuda_and_cupy_available
def test_equal_interval_cupy(result_equal_interval):
    k, expected_result = result_equal_interval
    cupy_agg = input_data(backend='cupy')
    cupy_result = equal_interval(cupy_agg, k=k)
    general_output_checks(cupy_agg, cupy_result, expected_result, verify_dtype=True)
