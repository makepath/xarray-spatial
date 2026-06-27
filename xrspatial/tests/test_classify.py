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
    general_output_checks(numpy_agg, numpy_result, expected_result, verify_dtype=True)


@dask_array_available
def test_binary_dask_numpy(result_binary):
    values, expected_result = result_binary
    dask_agg = input_data(backend='dask')
    dask_result = binary(dask_agg, values)
    general_output_checks(dask_agg, dask_result, expected_result, verify_dtype=True)


@cuda_and_cupy_available
def test_binary_cupy(result_binary):
    values, expected_result = result_binary
    cupy_agg = input_data(backend='cupy')
    cupy_result = binary(cupy_agg, values)
    general_output_checks(cupy_agg, cupy_result, expected_result, verify_dtype=True)


@dask_array_available
@cuda_and_cupy_available
def test_binary_dask_cupy(result_binary):
    values, expected_result = result_binary
    dask_cupy_agg = input_data(backend='dask+cupy')
    dask_cupy_result = binary(dask_cupy_agg, values)
    # the lazy dask array must advertise the same dtype it computes, otherwise
    # a downstream consumer reads float64 metadata for a float32 result
    assert dask_cupy_result.data.dtype == np.float32
    general_output_checks(dask_cupy_agg, dask_cupy_result, expected_result, verify_dtype=True)


def test_binary_output_dtype_float32():
    # binary() must emit float32 regardless of input dtype so its result
    # dtype matches the cupy/dask+cupy backends and the other classifiers
    # (regression for the numpy/dask paths returning the input dtype).
    for in_dtype in (np.float64, np.float32, np.int32):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=in_dtype)
        result = binary(xr.DataArray(data), [2, 5])
        assert result.data.dtype == np.float32


@dask_array_available
def test_binary_dask_output_dtype_float32():
    data = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64)
    dask_agg = xr.DataArray(da.from_array(data, chunks=(1, 3)))
    result = binary(dask_agg, [2, 5])
    assert result.data.compute().dtype == np.float32


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


# --- Memory guard for natural_breaks ---

def test_natural_breaks_rejects_oversize_jenks(monkeypatch):
    # A 400x400 raster produces 160_000 finite points -> Jenks matrices
    # ~15 MB at k=5. Shrink the memory budget so the guard trips.
    import xrspatial.classify as classify_mod
    monkeypatch.setattr(
        classify_mod, '_available_memory_bytes', lambda: 1_000_000
    )
    agg = xr.DataArray(np.random.RandomState(0).rand(400, 400))
    with pytest.raises(MemoryError, match='natural_breaks would allocate'):
        natural_breaks(agg, num_sample=None, k=5)


def test_natural_breaks_allows_within_budget(monkeypatch):
    # Same raster, generous budget -> no error.
    import xrspatial.classify as classify_mod
    monkeypatch.setattr(
        classify_mod, '_available_memory_bytes', lambda: 8 * 1024 ** 3
    )
    agg = xr.DataArray(np.random.RandomState(0).rand(50, 50))
    result = natural_breaks(agg, num_sample=None, k=3)
    assert result.shape == (50, 50)


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


# --- Degenerate input edge cases ---

def test_equal_interval_all_nan():
    data = np.full((4, 5), np.nan)
    agg = xr.DataArray(data)
    result = equal_interval(agg, k=3)
    assert np.all(np.isnan(result.data))


def test_equal_interval_all_same_values():
    data = np.full((4, 5), 7.0)
    agg = xr.DataArray(data)
    result = equal_interval(agg, k=3)
    finite_mask = np.isfinite(result.data)
    assert np.all(result.data[finite_mask] == 0)


def test_equal_interval_all_inf():
    data = np.full((4, 5), np.inf)
    agg = xr.DataArray(data)
    result = equal_interval(agg, k=3)
    # inf values are non-finite and should map to NaN
    assert np.all(np.isnan(result.data))


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


def test_natural_breaks_large_offset_1100():
    """Jenks should separate tight clusters even when data has a large offset.

    Regression test for #1100: float32 internals caused the variance
    formula to lose all significant digits for offset data.
    """
    rng = np.random.default_rng(0)
    centers = np.array([100_000, 100_010, 100_020, 100_030, 100_040])
    data = np.concatenate([c + rng.uniform(-1, 1, 200) for c in centers])
    agg = xr.DataArray(data.reshape(10, 100))

    result = natural_breaks(agg, k=5, num_sample=None)
    result_data = result.data
    if hasattr(result_data, 'compute'):
        result_data = result_data.compute()

    # All 5 classes should be present
    unique_classes = np.unique(result_data[~np.isnan(result_data)])
    assert len(unique_classes) == 5, (
        f"Expected 5 classes, got {len(unique_classes)}: {unique_classes}"
    )

    # Each center's 200 points should be in a single class
    flat = result_data.ravel()
    for i, center in enumerate(centers):
        start = i * 200
        end = start + 200
        chunk_classes = np.unique(flat[start:end])
        assert len(chunk_classes) == 1, (
            f"Center {center}: expected 1 class, got {len(chunk_classes)}"
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


# ===================================================================
# Regression tests: dask paths must not use boolean fancy indexing
# ===================================================================

@dask_array_available
def test_quantile_dask_no_unknown_chunks():
    """quantile on dask must not create unknown chunk sizes (#884)."""
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    numpy_agg = xr.DataArray(elevation)
    dask_agg = xr.DataArray(da.from_array(elevation, chunks=(5, 5)))

    numpy_result = quantile(numpy_agg, k=5)
    dask_result = quantile(dask_agg, k=5)

    # Dask percentile is approximate, so just check same shape and k classes
    assert dask_result.shape == numpy_result.shape
    dask_vals = dask_result.data.compute()
    unique_vals = np.unique(dask_vals[np.isfinite(dask_vals)])
    assert len(unique_vals) == 5


@dask_array_available
def test_quantile_dask_with_nan_inf():
    """quantile on dask handles NaN and inf without unknown chunks (#884)."""
    elevation = np.array([
        [-np.inf, 2., 3., 4., np.nan],
        [5., 6., 7., 8., 9.],
        [10., 11., 12., 13., 14.],
        [15., 16., 17., 18., np.inf],
    ])
    dask_agg = xr.DataArray(da.from_array(elevation, chunks=(2, 5)))
    result = quantile(dask_agg, k=5)
    result_data = result.data.compute()
    # NaN and inf inputs should produce NaN in the output
    assert np.isnan(result_data[0, 0])   # was -inf
    assert np.isnan(result_data[0, 4])   # was nan
    assert np.isnan(result_data[3, 4])   # was inf
    # Finite values should be classified
    finite_result = result_data[np.isfinite(result_data)]
    assert len(np.unique(finite_result)) == 5


@dask_array_available
def test_percentiles_dask_no_unknown_chunks():
    """percentiles on dask must not create unknown chunk sizes (#884)."""
    from xrspatial import percentiles as percentiles_fn
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    numpy_agg = xr.DataArray(elevation)
    dask_agg = xr.DataArray(da.from_array(elevation, chunks=(5, 5)))

    numpy_result = percentiles_fn(numpy_agg)
    dask_result = percentiles_fn(dask_agg)

    np.testing.assert_allclose(
        numpy_result.data,
        dask_result.data.compute(),
        equal_nan=True,
    )


def test_natural_breaks_positional_k_matches_siblings():
    """natural_breaks second positional arg is k, like quantile/maximum_breaks."""
    agg = input_data()
    positional = natural_breaks(agg, 3)
    keyword = natural_breaks(agg, k=3)
    np.testing.assert_array_equal(positional.data, keyword.data)


def test_natural_breaks_legacy_positional_num_sample_warns():
    """Legacy (agg, num_sample, k=...) order warns and still maps correctly."""
    agg = input_data()
    with pytest.warns(DeprecationWarning):
        legacy = natural_breaks(agg, 20000, k=3)
    new = natural_breaks(agg, k=3, num_sample=20000)
    np.testing.assert_array_equal(legacy.data, new.data)


# ===================================================================
# Geometric edge cases: 1x1 single pixel and Nx1 strip
# ===================================================================
# These classifiers are kernel-free, but the per-pixel bin search and the
# bin-edge computation both have degenerate behaviour on a one-cell or
# one-column raster (zero-width interval, a single unique value triggering
# the "not enough unique values" fallback). These shapes were previously
# untested for every classifier.

def _single_finite_class_zero(fn, agg, **kwargs):
    """Run fn and assert the lone finite pixel maps to class 0."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = fn(agg, **kwargs)
    assert result.shape == agg.shape
    finite = result.data[np.isfinite(result.data)]
    assert np.all(finite == 0)


def test_classify_single_pixel():
    agg = xr.DataArray(np.array([[5.0]]))
    _single_finite_class_zero(equal_interval, agg, k=3)
    _single_finite_class_zero(std_mean, agg)
    _single_finite_class_zero(head_tail_breaks, agg)
    _single_finite_class_zero(percentiles, agg)
    _single_finite_class_zero(box_plot, agg)
    # k>=2 classifiers fall back to a single bin on one unique value
    _single_finite_class_zero(quantile, agg, k=2)
    _single_finite_class_zero(natural_breaks, agg, k=2)
    _single_finite_class_zero(maximum_breaks, agg, k=2)


def test_binary_single_pixel():
    # A pixel that matches a target value maps to class 1; a non-match to 0.
    match = binary(xr.DataArray(np.array([[5.0]])), [5])
    assert match.shape == (1, 1)
    assert match.data[0, 0] == 1
    no_match = binary(xr.DataArray(np.array([[5.0]])), [9])
    assert no_match.data[0, 0] == 0


def test_reclassify_single_pixel():
    # The lone pixel falls in the first bin and takes that bin's new value.
    result = reclassify(xr.DataArray(np.array([[5.0]])), bins=[10], new_values=[7])
    assert result.shape == (1, 1)
    assert result.data[0, 0] == 7


def test_classify_nx1_strip():
    # 4x1 column strip with increasing values.
    strip = xr.DataArray(np.array([[1.], [2.], [3.], [4.]]))
    for fn, kwargs in [
        (equal_interval, dict(k=3)),
        (std_mean, {}),
        (head_tail_breaks, {}),
        (percentiles, {}),
        (box_plot, {}),
        (quantile, dict(k=3)),
        (natural_breaks, dict(k=3)),
        (maximum_breaks, dict(k=3)),
    ]:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fn(strip, **kwargs)
        assert result.shape == strip.shape
        finite = result.data[np.isfinite(result.data)]
        # Monotonic input -> classes are non-decreasing down the column.
        assert np.all(np.diff(finite) >= 0)


def test_classify_1xn_strip():
    # 1x4 row strip; mirror of the column case to catch axis-specific bugs.
    strip = xr.DataArray(np.array([[1., 2., 3., 4.]]))
    result = equal_interval(strip, k=3)
    assert result.shape == strip.shape
    finite = result.data[np.isfinite(result.data)]
    assert np.all(np.diff(finite) >= 0)


# ===================================================================
# Error paths: k below the validated minimum
# ===================================================================
# _validate_scalar enforces a minimum k for the parametric classifiers.
# Only reclassify's length-mismatch guard and the "not enough unique
# values" path were previously exercised; these min_val guards were not.

def test_quantile_k_below_min_raises():
    agg = input_data()
    with pytest.raises(ValueError, match='must be >= 2'):
        quantile(agg, k=1)


def test_natural_breaks_k_below_min_raises():
    agg = input_data()
    with pytest.raises(ValueError, match='must be >= 2'):
        natural_breaks(agg, k=1)


def test_maximum_breaks_k_below_min_raises():
    agg = input_data()
    with pytest.raises(ValueError, match='must be >= 2'):
        maximum_breaks(agg, k=1)


def test_equal_interval_k_below_min_raises():
    agg = input_data()
    with pytest.raises(ValueError, match='must be >= 1'):
        equal_interval(agg, k=0)


# ===================================================================
# Regression test: large-array sampler must be O(num_sample) in memory
# ===================================================================

def test_generate_sample_indices_large_is_sublinear_memory():
    """The >10M-element branch of _generate_sample_indices must not
    allocate an O(num_data) index permutation on the driver host.

    The legacy RandomState.choice(replace=False) builds a full
    arange(num_data) internally, so peak host memory scaled with the
    whole array and OOM'd on very large dask arrays. The Generator.choice
    (Floyd) path keeps peak memory proportional to num_sample.
    """
    import tracemalloc
    from xrspatial.classify import _generate_sample_indices

    num_data = 20_000_000  # exceeds the 10M small-branch threshold
    num_sample = 20_000

    tracemalloc.start()
    idx = _generate_sample_indices(num_data, num_sample)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert idx.size == num_sample
    # sorted ascending for efficient dask chunk access
    assert np.all(np.diff(idx) >= 0)
    assert idx.max() < num_data
    # A full int64 permutation of num_data would be ~160 MB. The Floyd
    # sampler stays far below that; allow generous head-room.
    assert peak < 20 * 1024 ** 2, (
        f"peak host memory {peak / 1024 ** 2:.1f} MB scales with num_data, "
        "not num_sample"
    )


def test_generate_sample_indices_large_is_deterministic():
    """The large-array sampler is seeded and reproducible across calls."""
    from xrspatial.classify import _generate_sample_indices

    a = _generate_sample_indices(20_000_000, 20_000)
    b = _generate_sample_indices(20_000_000, 20_000)
    np.testing.assert_array_equal(a, b)


# ===================================================================
# Degenerate all-NaN input for the remaining classifiers
# ===================================================================
# equal_interval and natural_breaks already cover all-NaN. The other
# classifiers were not exercised on an all-non-finite raster, where the
# finite mask removes every element. All of them now degrade to an all-NaN
# result. head_tail_breaks, percentiles, and box_plot used to raise an opaque
# reduction error on the eager (numpy/cupy) backends until #3510 added an
# empty-finite guard. See the dask section below for the per-backend split.

def test_std_mean_all_nan():
    import warnings
    agg = xr.DataArray(np.full((4, 5), np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = std_mean(agg)
    assert np.all(np.isnan(result.data))


def test_maximum_breaks_all_nan():
    agg = xr.DataArray(np.full((4, 5), np.nan))
    result = maximum_breaks(agg, k=5)
    assert np.all(np.isnan(result.data))


def test_head_tail_breaks_all_nan():
    agg = xr.DataArray(np.full((4, 5), np.nan))
    result = head_tail_breaks(agg)
    assert np.all(np.isnan(result.data))


def test_percentiles_all_nan():
    import warnings
    agg = xr.DataArray(np.full((4, 5), np.nan))
    # percentiles still takes nanmax over the (all-NaN) cleaned array to set
    # the top bin edge, which warns like std_mean does on this input.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = percentiles(agg)
    assert np.all(np.isnan(result.data))


def test_box_plot_all_nan():
    agg = xr.DataArray(np.full((4, 5), np.nan))
    result = box_plot(agg)
    assert np.all(np.isnan(result.data))


# All-inf input is mapped to NaN before binning, so it hits the same
# empty-finite path as all-NaN and must also return an all-NaN result (#3510).

def test_head_tail_breaks_all_inf():
    agg = xr.DataArray(np.full((4, 5), np.inf))
    result = head_tail_breaks(agg)
    assert np.all(np.isnan(result.data))


def test_percentiles_all_inf():
    import warnings
    agg = xr.DataArray(np.full((4, 5), np.inf))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = percentiles(agg)
    assert np.all(np.isnan(result.data))


def test_box_plot_all_inf():
    agg = xr.DataArray(np.full((4, 5), -np.inf))
    result = box_plot(agg)
    assert np.all(np.isnan(result.data))


# All-NaN on the dask backend. The dask paths are separate implementations.
# std_mean, maximum_breaks, and head_tail_breaks always returned all-NaN on
# dask (head_tail_breaks's dask path has a total_count == 0 guard the eager
# path lacked); percentiles and box_plot used to crash on dask too until
# #3510 added empty-finite guards to their dask sample paths. Pin all of it
# so the per-backend behaviour stays explicit.

@dask_array_available
def test_std_mean_all_nan_dask():
    import warnings
    agg = xr.DataArray(da.full((4, 5), np.nan, chunks=(2, 5)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = std_mean(agg)
    assert np.all(np.isnan(result.data.compute()))


@dask_array_available
def test_maximum_breaks_all_nan_dask():
    agg = xr.DataArray(da.full((4, 5), np.nan, chunks=(2, 5)))
    result = maximum_breaks(agg, k=5)
    assert np.all(np.isnan(result.data.compute()))


@dask_array_available
def test_head_tail_breaks_all_nan_dask():
    # Diverges from the eager path: dask returns all-NaN cleanly (#3510 is
    # eager-only). Plain passing test, not xfail.
    agg = xr.DataArray(da.full((4, 5), np.nan, chunks=(2, 5)))
    result = head_tail_breaks(agg)
    assert np.all(np.isnan(result.data.compute()))


@dask_array_available
def test_percentiles_all_nan_dask():
    agg = xr.DataArray(da.full((4, 5), np.nan, chunks=(2, 5)))
    result = percentiles(agg)
    assert np.all(np.isnan(result.data.compute()))


@dask_array_available
def test_box_plot_all_nan_dask():
    agg = xr.DataArray(da.full((4, 5), np.nan, chunks=(2, 5)))
    result = box_plot(agg)
    assert np.all(np.isnan(result.data.compute()))
