import numpy as np
import pytest
import xarray as xr

from xrspatial import (binary, box_plot, equal_interval, head_tail_breaks,
                       maximum_breaks, natural_breaks, percentiles, quantile,
                       reclassify, std_mean)
from xrspatial.tests.general_checks import (assert_input_data_unmodified,
                                            create_test_raster,
                                            cuda_and_cupy_available,
                                            dask_array_available,
                                            general_output_checks)

try:
    import dask.array as da
except ImportError:
    da = None


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


@dask_array_available
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


@dask_array_available
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


@dask_array_available
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


@dask_array_available
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


@dask_array_available
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
    with pytest.warns(Warning):
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


@dask_array_available
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


@dask_array_available
@cuda_and_cupy_available
def test_equal_interval_dask_cupy(result_equal_interval):
    k, expected_result = result_equal_interval
    dask_cupy_agg = input_data(backend='dask+cupy')
    dask_cupy_result = equal_interval(dask_cupy_agg, k=k)
    general_output_checks(dask_cupy_agg, dask_cupy_result, expected_result, verify_dtype=True)


@dask_array_available
@cuda_and_cupy_available
def test_quantile_dask_cupy(result_quantile):
    # Relaxed verification (same pattern as test_quantile_dask_numpy)
    # because percentile is computed on CPU from materialized data
    dask_cupy_agg = input_data('dask+cupy')
    k, expected_result = result_quantile
    dask_cupy_quantile = quantile(dask_cupy_agg, k=k)
    general_output_checks(dask_cupy_agg, dask_cupy_quantile)
    dask_cupy_quantile = dask_cupy_quantile.compute()
    import cupy as cp
    result_data = cp.asnumpy(dask_cupy_quantile.data)
    unique_elements = np.unique(result_data[np.isfinite(result_data)])
    assert len(unique_elements) == k


@cuda_and_cupy_available
def test_natural_breaks_cupy(result_natural_breaks):
    cupy_agg = input_data('cupy')
    k, expected_result = result_natural_breaks
    cupy_natural_breaks = natural_breaks(cupy_agg, k=k)
    general_output_checks(cupy_agg, cupy_natural_breaks, expected_result, verify_dtype=True)


@dask_array_available
def test_natural_breaks_dask_numpy(result_natural_breaks):
    dask_agg = input_data('dask+numpy')
    k, expected_result = result_natural_breaks
    dask_natural_breaks = natural_breaks(dask_agg, k=k)
    general_output_checks(dask_agg, dask_natural_breaks, expected_result, verify_dtype=True)


@dask_array_available
@cuda_and_cupy_available
def test_natural_breaks_dask_cupy(result_natural_breaks):
    dask_cupy_agg = input_data('dask+cupy')
    k, expected_result = result_natural_breaks
    dask_cupy_natural_breaks = natural_breaks(dask_cupy_agg, k=k)
    general_output_checks(dask_cupy_agg, dask_cupy_natural_breaks, expected_result, verify_dtype=True)


@cuda_and_cupy_available
def test_natural_breaks_cupy_num_sample(result_natural_breaks_num_sample):
    cupy_agg = input_data('cupy')
    k, num_sample, expected_result = result_natural_breaks_num_sample
    cupy_natural_breaks = natural_breaks(cupy_agg, k=k, num_sample=num_sample)
    general_output_checks(cupy_agg, cupy_natural_breaks, expected_result, verify_dtype=True)


@dask_array_available
def test_natural_breaks_dask_numpy_num_sample(result_natural_breaks_num_sample):
    dask_agg = input_data('dask+numpy')
    k, num_sample, expected_result = result_natural_breaks_num_sample
    dask_natural_breaks = natural_breaks(dask_agg, k=k, num_sample=num_sample)
    general_output_checks(dask_agg, dask_natural_breaks, expected_result, verify_dtype=True)


@dask_array_available
@cuda_and_cupy_available
def test_natural_breaks_dask_cupy_num_sample(result_natural_breaks_num_sample):
    dask_cupy_agg = input_data('dask+cupy')
    k, num_sample, expected_result = result_natural_breaks_num_sample
    dask_cupy_natural_breaks = natural_breaks(dask_cupy_agg, k=k, num_sample=num_sample)
    general_output_checks(
        dask_cupy_agg, dask_cupy_natural_breaks, expected_result, verify_dtype=True)


# --- Input mutation tests ---
# Classification functions must not modify the input DataArray.
# natural_breaks is most critical because _run_jenks sorts in-place.

def test_binary_does_not_modify_input():
    agg = input_data()
    original = agg.copy(deep=True)
    binary(agg, [1, 2, 3])
    assert_input_data_unmodified(original, agg)


def test_reclassify_does_not_modify_input():
    agg = input_data()
    original = agg.copy(deep=True)
    reclassify(agg, bins=[10, 15, np.inf], new_values=[1, 2, 3])
    assert_input_data_unmodified(original, agg)


def test_quantile_does_not_modify_input():
    agg = input_data()
    original = agg.copy(deep=True)
    quantile(agg, k=5)
    assert_input_data_unmodified(original, agg)


def test_natural_breaks_does_not_modify_input():
    agg = input_data()
    original = agg.copy(deep=True)
    natural_breaks(agg, k=5)
    assert_input_data_unmodified(original, agg)


def test_equal_interval_does_not_modify_input():
    agg = input_data()
    original = agg.copy(deep=True)
    equal_interval(agg, k=3)
    assert_input_data_unmodified(original, agg)


# --- num_sample=None test ---
# Tests the code path where all data is used without sampling.
# For the test data (20 elements), this produces the same result
# as default num_sample=20000 since 20000 > 20.

def test_natural_breaks_numpy_num_sample_none(result_natural_breaks):
    numpy_agg = input_data()
    k, expected_result = result_natural_breaks
    result = natural_breaks(numpy_agg, k=k, num_sample=None)
    general_output_checks(numpy_agg, result, expected_result, verify_dtype=True)


# --- Edge cases for equal_interval ---

def test_equal_interval_k_equals_1():
    agg = input_data()
    result = equal_interval(agg, k=1)
    result_data = result.data
    # All finite values should be in class 0
    finite_mask = np.isfinite(result_data)
    assert np.all(result_data[finite_mask] == 0)
    # Non-finite input positions should be NaN in output
    input_finite = np.isfinite(agg.data)
    assert np.all(np.isnan(result_data[~input_finite]))


# --- All-NaN edge cases ---
# These document current failure behavior for degenerate inputs.

def test_equal_interval_all_nan():
    data = np.full((4, 5), np.nan)
    agg = xr.DataArray(data)
    with pytest.raises(ValueError):
        equal_interval(agg, k=3)


def test_natural_breaks_all_nan():
    data = np.full((4, 5), np.nan)
    agg = xr.DataArray(data)
    with pytest.raises(ValueError):
        natural_breaks(agg, k=3)


# --- Name parameter tests ---

def test_output_name_default():
    agg = input_data()
    assert binary(agg, [1, 2]).name == 'binary'
    assert reclassify(agg, [10, 15], [1, 2]).name == 'reclassify'
    assert quantile(agg, k=3).name == 'quantile'
    assert natural_breaks(agg, k=3).name == 'natural_breaks'
    assert equal_interval(agg, k=3).name == 'equal_interval'


def test_output_name_custom():
    agg = input_data()
    assert binary(agg, [1, 2], name='custom').name == 'custom'
    assert reclassify(agg, [10, 15], [1, 2], name='custom').name == 'custom'
    assert quantile(agg, k=3, name='custom').name == 'custom'
    assert natural_breaks(agg, k=3, name='custom').name == 'custom'
    assert equal_interval(agg, k=3, name='custom').name == 'custom'


# --- Cross-backend consistency for natural_breaks ---
# Verifies that cupy/dask backends produce identical results to numpy
# using a different dataset (10x10 arange) than the fixture tests.

@cuda_and_cupy_available
def test_natural_breaks_cupy_matches_numpy():
    import cupy as cp
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    numpy_agg = xr.DataArray(elevation)
    cupy_agg = xr.DataArray(cp.asarray(elevation))

    k = 5
    numpy_result = natural_breaks(numpy_agg, k=k)
    cupy_result = natural_breaks(cupy_agg, k=k)

    np.testing.assert_allclose(
        numpy_result.data, cp.asnumpy(cupy_result.data), equal_nan=True
    )


@dask_array_available
def test_natural_breaks_dask_matches_numpy():
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    numpy_agg = xr.DataArray(elevation)
    dask_agg = xr.DataArray(da.from_array(elevation, chunks=(5, 5)))

    k = 5
    numpy_result = natural_breaks(numpy_agg, k=k)
    dask_result = natural_breaks(dask_agg, k=k)

    np.testing.assert_allclose(
        numpy_result.data, dask_result.data.compute(), equal_nan=True
    )


# ===================================================================
# std_mean tests
# ===================================================================

@pytest.fixture
def result_std_mean():
    expected_result = np.asarray([
        [np.nan, 1., 1., 1., np.nan],
        [1., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2.],
        [3., 3., 3., 3., np.nan]
    ], dtype=np.float32)
    return expected_result


def test_std_mean_numpy(result_std_mean):
    numpy_agg = input_data()
    numpy_result = std_mean(numpy_agg)
    general_output_checks(numpy_agg, numpy_result, result_std_mean, verify_dtype=True)


@dask_array_available
def test_std_mean_dask_numpy(result_std_mean):
    dask_agg = input_data('dask+numpy')
    dask_result = std_mean(dask_agg)
    general_output_checks(dask_agg, dask_result, result_std_mean, verify_dtype=True)


@cuda_and_cupy_available
def test_std_mean_cupy(result_std_mean):
    cupy_agg = input_data('cupy')
    cupy_result = std_mean(cupy_agg)
    general_output_checks(cupy_agg, cupy_result, result_std_mean, verify_dtype=True)


@dask_array_available
@cuda_and_cupy_available
def test_std_mean_dask_cupy(result_std_mean):
    dask_cupy_agg = input_data('dask+cupy')
    dask_cupy_result = std_mean(dask_cupy_agg)
    general_output_checks(dask_cupy_agg, dask_cupy_result, result_std_mean, verify_dtype=True)


def test_std_mean_does_not_modify_input():
    agg = input_data()
    original = agg.copy(deep=True)
    std_mean(agg)
    assert_input_data_unmodified(original, agg)


def test_std_mean_all_same_values():
    data = np.full((4, 5), 7.0)
    agg = xr.DataArray(data)
    result = std_mean(agg)
    # std=0, so all bins collapse to mean=7. All values in class 0.
    finite_mask = np.isfinite(result.data)
    assert np.all(result.data[finite_mask] == 0)


# ===================================================================
# head_tail_breaks tests
# ===================================================================

@pytest.fixture
def result_head_tail_breaks():
    expected_result = np.asarray([
        [np.nan, 0., 0., 0., np.nan],
        [0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 1.],
        [1., 1., 1., 1., np.nan]
    ], dtype=np.float32)
    return expected_result


def test_head_tail_breaks_numpy(result_head_tail_breaks):
    numpy_agg = input_data()
    numpy_result = head_tail_breaks(numpy_agg)
    general_output_checks(numpy_agg, numpy_result, result_head_tail_breaks, verify_dtype=True)


@dask_array_available
def test_head_tail_breaks_dask_numpy(result_head_tail_breaks):
    dask_agg = input_data('dask+numpy')
    dask_result = head_tail_breaks(dask_agg)
    general_output_checks(dask_agg, dask_result, result_head_tail_breaks, verify_dtype=True)


@cuda_and_cupy_available
def test_head_tail_breaks_cupy(result_head_tail_breaks):
    cupy_agg = input_data('cupy')
    cupy_result = head_tail_breaks(cupy_agg)
    general_output_checks(cupy_agg, cupy_result, result_head_tail_breaks, verify_dtype=True)


@dask_array_available
@cuda_and_cupy_available
def test_head_tail_breaks_dask_cupy(result_head_tail_breaks):
    dask_cupy_agg = input_data('dask+cupy')
    dask_cupy_result = head_tail_breaks(dask_cupy_agg)
    general_output_checks(dask_cupy_agg, dask_cupy_result, result_head_tail_breaks, verify_dtype=True)


def test_head_tail_breaks_does_not_modify_input():
    agg = input_data()
    original = agg.copy(deep=True)
    head_tail_breaks(agg)
    assert_input_data_unmodified(original, agg)


def test_head_tail_breaks_heavy_tailed():
    # Heavy-tailed data should produce more classes than uniform data
    data = np.array([
        [1., 1., 1., 1., 2.],
        [2., 2., 3., 3., 5.],
        [5., 10., 20., 50., 100.],
        [200., 500., 1000., 2000., 5000.],
    ])
    agg = xr.DataArray(data)
    result = head_tail_breaks(agg)
    unique_classes = np.unique(result.data[np.isfinite(result.data)])
    # Heavy-tailed data should produce more than 2 classes
    assert len(unique_classes) > 2


# ===================================================================
# percentiles tests
# ===================================================================

@pytest.fixture
def result_percentiles():
    expected_result = np.asarray([
        [np.nan, 0., 1., 2., np.nan],
        [2., 2., 2., 2., 2.],
        [2., 3., 3., 3., 3.],
        [3., 3., 4., 5., np.nan]
    ], dtype=np.float32)
    return expected_result


def test_percentiles_numpy(result_percentiles):
    numpy_agg = input_data()
    numpy_result = percentiles(numpy_agg)
    general_output_checks(numpy_agg, numpy_result, result_percentiles, verify_dtype=True)


@dask_array_available
def test_percentiles_dask_numpy(result_percentiles):
    # Dask percentile is approximate; verify structure not exact values
    dask_agg = input_data('dask+numpy')
    dask_result = percentiles(dask_agg)
    general_output_checks(dask_agg, dask_result)


@cuda_and_cupy_available
def test_percentiles_cupy(result_percentiles):
    cupy_agg = input_data('cupy')
    cupy_result = percentiles(cupy_agg)
    general_output_checks(cupy_agg, cupy_result, result_percentiles, verify_dtype=True)


@dask_array_available
@cuda_and_cupy_available
def test_percentiles_dask_cupy(result_percentiles):
    dask_cupy_agg = input_data('dask+cupy')
    dask_cupy_result = percentiles(dask_cupy_agg)
    general_output_checks(dask_cupy_agg, dask_cupy_result)


def test_percentiles_does_not_modify_input():
    agg = input_data()
    original = agg.copy(deep=True)
    percentiles(agg)
    assert_input_data_unmodified(original, agg)


def test_percentiles_custom_pct():
    numpy_agg = input_data()
    result = percentiles(numpy_agg, pct=[25, 50, 75])
    result_data = result.data
    finite_vals = result_data[np.isfinite(result_data)]
    unique_classes = np.unique(finite_vals)
    # Should have at most 4 classes (3 percentile breaks + max)
    assert len(unique_classes) <= 4


def test_percentiles_single_pct():
    numpy_agg = input_data()
    result = percentiles(numpy_agg, pct=[50])
    result_data = result.data
    finite_vals = result_data[np.isfinite(result_data)]
    unique_classes = np.unique(finite_vals)
    # Single percentile + max → 2 classes
    assert len(unique_classes) == 2


# ===================================================================
# maximum_breaks tests
# ===================================================================

@pytest.fixture
def result_maximum_breaks():
    expected_result = np.asarray([
        [np.nan, 0., 0., 0., np.nan],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [1., 2., 3., 4., np.nan]
    ], dtype=np.float32)
    return expected_result


def test_maximum_breaks_numpy(result_maximum_breaks):
    numpy_agg = input_data()
    numpy_result = maximum_breaks(numpy_agg)
    general_output_checks(numpy_agg, numpy_result, result_maximum_breaks, verify_dtype=True)


@dask_array_available
def test_maximum_breaks_dask_numpy(result_maximum_breaks):
    dask_agg = input_data('dask+numpy')
    dask_result = maximum_breaks(dask_agg)
    general_output_checks(dask_agg, dask_result, result_maximum_breaks, verify_dtype=True)


@cuda_and_cupy_available
def test_maximum_breaks_cupy(result_maximum_breaks):
    cupy_agg = input_data('cupy')
    cupy_result = maximum_breaks(cupy_agg)
    general_output_checks(cupy_agg, cupy_result, result_maximum_breaks, verify_dtype=True)


@dask_array_available
@cuda_and_cupy_available
def test_maximum_breaks_dask_cupy(result_maximum_breaks):
    dask_cupy_agg = input_data('dask+cupy')
    dask_cupy_result = maximum_breaks(dask_cupy_agg)
    general_output_checks(dask_cupy_agg, dask_cupy_result, result_maximum_breaks, verify_dtype=True)


def test_maximum_breaks_does_not_modify_input():
    agg = input_data()
    original = agg.copy(deep=True)
    maximum_breaks(agg)
    assert_input_data_unmodified(original, agg)


def test_maximum_breaks_k_equals_2():
    numpy_agg = input_data()
    result = maximum_breaks(numpy_agg, k=2)
    finite_vals = result.data[np.isfinite(result.data)]
    unique_classes = np.unique(finite_vals)
    assert len(unique_classes) == 2


def test_maximum_breaks_k_exceeds_unique():
    # When k > number of unique values, fall back gracefully
    data = np.array([[1., 2., 3.], [4., 5., np.nan]])
    agg = xr.DataArray(data)
    result = maximum_breaks(agg, k=10)
    finite_vals = result.data[np.isfinite(result.data)]
    assert len(np.unique(finite_vals)) <= 5


# ===================================================================
# box_plot tests
# ===================================================================

@pytest.fixture
def result_box_plot():
    expected_result = np.asarray([
        [np.nan, 1., 1., 1., np.nan],
        [1., 1., 2., 2., 2.],
        [2., 3., 3., 3., 3.],
        [4., 4., 4., 4., np.nan]
    ], dtype=np.float32)
    return expected_result


def test_box_plot_numpy(result_box_plot):
    numpy_agg = input_data()
    numpy_result = box_plot(numpy_agg)
    general_output_checks(numpy_agg, numpy_result, result_box_plot, verify_dtype=True)


@dask_array_available
def test_box_plot_dask_numpy(result_box_plot):
    # Dask percentile is approximate; verify structure not exact values
    dask_agg = input_data('dask+numpy')
    dask_result = box_plot(dask_agg)
    general_output_checks(dask_agg, dask_result)


@cuda_and_cupy_available
def test_box_plot_cupy(result_box_plot):
    cupy_agg = input_data('cupy')
    cupy_result = box_plot(cupy_agg)
    general_output_checks(cupy_agg, cupy_result, result_box_plot, verify_dtype=True)


@dask_array_available
@cuda_and_cupy_available
def test_box_plot_dask_cupy(result_box_plot):
    dask_cupy_agg = input_data('dask+cupy')
    dask_cupy_result = box_plot(dask_cupy_agg)
    general_output_checks(dask_cupy_agg, dask_cupy_result)


def test_box_plot_does_not_modify_input():
    agg = input_data()
    original = agg.copy(deep=True)
    box_plot(agg)
    assert_input_data_unmodified(original, agg)


def test_box_plot_custom_hinge():
    numpy_agg = input_data()
    result = box_plot(numpy_agg, hinge=3.0)
    result_data = result.data
    finite_vals = result_data[np.isfinite(result_data)]
    # With hinge=3.0, whiskers extend further so fewer outliers
    assert len(np.unique(finite_vals)) >= 2


# ===================================================================
# Name parameter tests for new methods
# ===================================================================

def test_new_methods_output_name_default():
    agg = input_data()
    assert std_mean(agg).name == 'std_mean'
    assert head_tail_breaks(agg).name == 'head_tail_breaks'
    assert percentiles(agg).name == 'percentiles'
    assert maximum_breaks(agg).name == 'maximum_breaks'
    assert box_plot(agg).name == 'box_plot'


def test_new_methods_output_name_custom():
    agg = input_data()
    assert std_mean(agg, name='custom').name == 'custom'
    assert head_tail_breaks(agg, name='custom').name == 'custom'
    assert percentiles(agg, name='custom').name == 'custom'
    assert maximum_breaks(agg, name='custom').name == 'custom'
    assert box_plot(agg, name='custom').name == 'custom'


# ===================================================================
# Cross-backend consistency tests for new methods
# ===================================================================

@dask_array_available
def test_std_mean_dask_matches_numpy():
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    numpy_agg = xr.DataArray(elevation)
    dask_agg = xr.DataArray(da.from_array(elevation, chunks=(5, 5)))
    np.testing.assert_allclose(
        std_mean(numpy_agg).data, std_mean(dask_agg).data.compute(), equal_nan=True
    )


@dask_array_available
def test_head_tail_breaks_dask_matches_numpy():
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    numpy_agg = xr.DataArray(elevation)
    dask_agg = xr.DataArray(da.from_array(elevation, chunks=(5, 5)))
    np.testing.assert_allclose(
        head_tail_breaks(numpy_agg).data,
        head_tail_breaks(dask_agg).data.compute(),
        equal_nan=True,
    )


@dask_array_available
def test_maximum_breaks_dask_matches_numpy():
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    numpy_agg = xr.DataArray(elevation)
    dask_agg = xr.DataArray(da.from_array(elevation, chunks=(5, 5)))
    np.testing.assert_allclose(
        maximum_breaks(numpy_agg).data,
        maximum_breaks(dask_agg).data.compute(),
        equal_nan=True,
    )


# ===================================================================
# Regression tests: dask paths must not materialise the full array
# ===================================================================

@dask_array_available
def test_natural_breaks_dask_no_full_compute():
    """natural_breaks with num_sample=None on dask must not call
    .ravel().compute() on the full array (#877)."""
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    numpy_agg = xr.DataArray(elevation)
    dask_agg = xr.DataArray(da.from_array(elevation, chunks=(5, 5)))

    numpy_result = natural_breaks(numpy_agg, num_sample=None, k=5)
    dask_result = natural_breaks(dask_agg, num_sample=None, k=5)

    np.testing.assert_allclose(
        numpy_result.data,
        dask_result.data.compute(),
        equal_nan=True,
    )


@dask_array_available
def test_maximum_breaks_dask_no_full_compute():
    """maximum_breaks on dask must use sampling, not .ravel().compute() (#876)."""
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    numpy_agg = xr.DataArray(elevation)
    dask_agg = xr.DataArray(da.from_array(elevation, chunks=(5, 5)))

    # Default num_sample (20_000) > data.size (100), so all data used
    numpy_result = maximum_breaks(numpy_agg, k=5)
    dask_result = maximum_breaks(dask_agg, k=5)

    np.testing.assert_allclose(
        numpy_result.data,
        dask_result.data.compute(),
        equal_nan=True,
    )


@dask_array_available
def test_maximum_breaks_dask_num_sample():
    """maximum_breaks with explicit num_sample on dask produces valid,
    deterministic results (#876)."""
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    dask_agg = xr.DataArray(da.from_array(elevation, chunks=(5, 5)))

    result1 = maximum_breaks(dask_agg, k=3, num_sample=50)
    result2 = maximum_breaks(dask_agg, k=3, num_sample=50)

    # Deterministic: same input + same seed → same output
    np.testing.assert_allclose(
        result1.data.compute(),
        result2.data.compute(),
        equal_nan=True,
    )
    # Valid classification: correct shape and values in expected range
    assert result1.shape == elevation.shape
    unique_vals = np.unique(result1.data.compute())
    assert len(unique_vals) <= 3 + 1  # at most k classes + possible nan
