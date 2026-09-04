from unittest.mock import patch

try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial import mean
from xrspatial.convolution import (annulus_kernel, calc_cellsize, circle_kernel, convolution_2d,
                                   convolve_2d, custom_kernel)
from xrspatial.focal import apply, focal_stats, hotspots
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            create_test_raster,
                                            cuda_and_cupy_available,
                                            dask_array_available,
                                            general_output_checks)
from xrspatial.utils import ngjit


data_random = np.random.default_rng(0).random((40, 40))


def test_mean_transfer_function_cpu():
    # numpy case
    numpy_agg = xr.DataArray(data_random)
    numpy_mean = mean(numpy_agg)
    general_output_checks(numpy_agg, numpy_mean)


def test_mean_default_excludes_does_not_leak():
    # excludes defaults to None and is resolved to [np.nan] inside the body,
    # so the default must not be a shared mutable object across calls.
    numpy_agg = xr.DataArray(data_random)

    first = mean(numpy_agg)
    second = mean(numpy_agg)

    # default None resolves to the same behaviour as an explicit [np.nan]
    explicit = mean(numpy_agg, excludes=[np.nan])
    np.testing.assert_array_equal(first.data, explicit.data)
    np.testing.assert_array_equal(first.data, second.data)


@dask_array_available
def test_mean_transfer_function_dask_cpu():
    # numpy case
    numpy_agg = xr.DataArray(data_random)
    numpy_mean = mean(numpy_agg)
    general_output_checks(numpy_agg, numpy_mean)

    # dask + numpy case
    dask_numpy_agg = xr.DataArray(da.from_array(data_random, chunks=(20, 20)))
    dask_numpy_mean = mean(dask_numpy_agg)
    general_output_checks(dask_numpy_agg, dask_numpy_mean)

    # both output same results
    np.testing.assert_allclose(
        numpy_mean.data, dask_numpy_mean.data.compute(), equal_nan=True
    )


@cuda_and_cupy_available
def test_mean_transfer_function_gpu_equals_cpu():

    import cupy

    # cupy case
    cupy_agg = xr.DataArray(cupy.asarray(data_random))
    cupy_mean = mean(cupy_agg)
    general_output_checks(cupy_agg, cupy_mean)

    # numpy case
    numpy_agg = xr.DataArray(data_random)
    numpy_mean = mean(numpy_agg)

    np.testing.assert_allclose(
        numpy_mean.data, cupy_mean.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_mean_transfer_function_dask_gpu():

    import cupy

    # numpy reference
    numpy_agg = xr.DataArray(data_random)
    numpy_mean = mean(numpy_agg)

    # dask + cupy case
    dask_cupy_agg = xr.DataArray(
        da.from_array(cupy.asarray(data_random), chunks=(20, 20))
    )
    dask_cupy_mean = mean(dask_cupy_agg)
    general_output_checks(dask_cupy_agg, dask_cupy_mean)

    np.testing.assert_allclose(
        numpy_mean.data, dask_cupy_mean.data.compute().get(),
        equal_nan=True, rtol=1e-4)


@pytest.fixture
def convolve_2d_data():
    data = np.array([
        [0., 1., 1., 1., 1., 1.],
        [1., 0., 1., 1., 1., 1.],
        [1., 1., 0., 1., 1., 1.],
        [1., 1., 1., np.nan, 1., 1.],
        [1., 1., 1., 1., 0., 1.],
        [1., 1., 1., 1., 1., 0.]
    ])
    return data


@pytest.fixture
def kernel_circle_1_1_1():
    result = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    return result


@pytest.fixture
def kernel_annulus_2_2_2_1():
    result = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    return result


@pytest.fixture
def convolution_kernel_circle_1_1_1():
    expected_result = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., 3., 5., 5., np.nan],
        [np.nan, 3., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 5., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 5., np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    return expected_result


@pytest.fixture
def convolution_kernel_annulus_2_2_1():
    expected_result = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., 2., 4., 4., np.nan],
        [np.nan, 2., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    return expected_result


@pytest.fixture
def convolution_custom_kernel():
    kernel = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]])
    expected_result = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 2., 3., 3., 4., np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4., np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    return kernel, expected_result


def test_kernel_custom_kernel_invalid_type():
    kernel = [1, 0, 0]  # only arrays are accepted, not lists
    with pytest.raises(ValueError):
        custom_kernel(kernel)


def test_kernel_custom_kernel_invalid_shape():
    kernel = np.ones((4, 6))
    with pytest.raises(ValueError):
        custom_kernel(kernel)


def test_kernel(kernel_circle_1_1_1, kernel_annulus_2_2_2_1):
    kernel_circle = circle_kernel(1, 1, 1)
    assert isinstance(kernel_circle, np.ndarray)
    np.testing.assert_allclose(kernel_circle, kernel_circle_1_1_1, equal_nan=True)

    kernel_annulus = annulus_kernel(2, 2, 2, 1)
    assert isinstance(kernel_annulus, np.ndarray)
    np.testing.assert_allclose(kernel_annulus, kernel_annulus_2_2_2_1, equal_nan=True)


def test_circle_kernel_rejects_oversize_radius_1241():
    # Regression test for #1241: circle_kernel() with no radius cap.
    # cellsize=1, radius=1_000_000 implies a ~2M x 2M float64 kernel
    # (~32 TB), which should be rejected before allocation.
    with pytest.raises(MemoryError, match="radius=1000000"):
        circle_kernel(1, 1, 1_000_000)


def test_annulus_kernel_rejects_oversize_radius_1241():
    # Regression test for #1241: annulus_kernel() calls circle_kernel
    # twice, so the same oversize-radius guard must fire for either
    # outer or inner radius.
    with pytest.raises(MemoryError, match="radius=1000000"):
        annulus_kernel(1, 1, 1_000_000, 1)


def test_circle_kernel_small_radius_not_rejected_1241():
    # Regression test for #1241: the guard must not fire for realistic
    # kernel sizes.  A radius=100 on cellsize=1 gives a 201x201 kernel
    # (~320 KB float64) which should allocate fine.
    kernel = circle_kernel(1, 1, 100)
    assert kernel.shape == (201, 201)


def test_apply_rejects_oversize_kernel_1284():
    # Regression for #1284: focal.apply must reject a user-supplied
    # kernel that would OOM on the padded raster + kernel allocation.
    # Patch the memory probe so a tiny kernel still trips the guard.
    raster = xr.DataArray(np.zeros((10, 10), dtype=np.float32))
    kernel = np.ones((101, 101), dtype=np.float32)
    with patch('xrspatial.focal._available_memory_bytes', return_value=1):
        with pytest.raises(MemoryError, match=r"apply\(\): kernel of shape"):
            apply(raster, kernel)


def test_focal_stats_rejects_oversize_kernel_1284():
    # Regression for #1284: focal_stats must apply the same kernel
    # vs raster guard before dispatching to any backend.
    raster = xr.DataArray(np.zeros((10, 10), dtype=np.float32))
    kernel = np.ones((101, 101), dtype=np.float32)
    with patch('xrspatial.focal._available_memory_bytes', return_value=1):
        with pytest.raises(MemoryError, match=r"focal_stats\(\): kernel of shape"):
            focal_stats(raster, kernel, stats_funcs=['mean'])


def test_hotspots_rejects_oversize_kernel_1284():
    # Regression for #1284: hotspots calls convolve_2d under the hood,
    # which inherits the same padded-allocation footprint.
    raster = xr.DataArray(np.zeros((10, 10), dtype=np.float32))
    kernel = np.ones((101, 101), dtype=np.float32)
    with patch('xrspatial.focal._available_memory_bytes', return_value=1):
        with pytest.raises(MemoryError, match=r"hotspots\(\): kernel of shape"):
            hotspots(raster, kernel)


def test_apply_oversize_kernel_accounts_for_float64_3223():
    # Regression for #3223: the guard budgeted 4 bytes/cell (float32) but
    # apply() preserves float64 input since #2805, so float64 combos could
    # pass the guard and then allocate twice the estimate. With 1 MB
    # "available", a (201, 201) kernel on a 10x10 raster needs ~338 KB in
    # float32 (within the 50% threshold) but ~676 KB in float64, which
    # must be rejected.
    kernel = np.ones((201, 201), dtype=np.float32)
    raster64 = xr.DataArray(np.zeros((10, 10), dtype=np.float64))
    with patch('xrspatial.focal._available_memory_bytes',
               return_value=1_000_000):
        with pytest.raises(MemoryError, match=r"apply\(\): kernel of shape"):
            apply(raster64, kernel)

    # The same combination with float32 input stays within budget.
    raster32 = xr.DataArray(np.zeros((10, 10), dtype=np.float32))
    with patch('xrspatial.focal._available_memory_bytes',
               return_value=1_000_000):
        out = apply(raster32, kernel)
    assert out.shape == (10, 10)


def test_focal_stats_oversize_kernel_accounts_for_float64_3223():
    # Regression for #3223: same float64 budget check for focal_stats.
    kernel = np.ones((201, 201), dtype=np.float32)
    raster64 = xr.DataArray(np.zeros((10, 10), dtype=np.float64))
    with patch('xrspatial.focal._available_memory_bytes',
               return_value=1_000_000):
        with pytest.raises(MemoryError,
                           match=r"focal_stats\(\): kernel of shape"):
            focal_stats(raster64, kernel, stats_funcs=['mean'])


def test_hotspots_float64_keeps_float32_budget_3223():
    # hotspots() computes in float32 on every backend, so float64 input
    # must stay on the 4 bytes/cell budget and not be over-rejected with
    # the 8-byte budget apply()/focal_stats() use for float64.
    kernel = np.ones((201, 201), dtype=np.float32)
    data = np.random.default_rng(3223).random((10, 10)).astype(np.float64)
    with patch('xrspatial.focal._available_memory_bytes',
               return_value=1_000_000):
        out = hotspots(xr.DataArray(data), kernel)
    assert out.shape == (10, 10)


@dask_array_available
@pytest.mark.parametrize("entry_point", [
    lambda agg, kernel: apply(agg, kernel),
    lambda agg, kernel: focal_stats(agg, kernel, stats_funcs=['mean']),
    lambda agg, kernel: hotspots(agg, kernel),
])
def test_memory_guard_accepts_large_dask_raster_3218(entry_point):
    # Regression for #3218: the guard budgeted the padded FULL raster on
    # every backend, so a dask raster bigger than ~half host RAM was
    # rejected at graph-build time even though map_overlap only ever
    # materializes one padded chunk per task. With 1 MB "available", the
    # full padded raster (~4 MB) would trip the old guard; the padded
    # 100x100 chunk (~42 KB) must pass.
    data = da.zeros((1000, 1000), chunks=(100, 100), dtype=np.float32)
    agg = xr.DataArray(data, dims=['y', 'x'])
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32)
    with patch('xrspatial.focal._available_memory_bytes',
               return_value=1_000_000):
        result = entry_point(agg, kernel)
    assert isinstance(result.data, da.Array)


def test_memory_guard_numpy_raster_still_rejected_3218():
    # The numpy path really does allocate full-size arrays, so the
    # full-raster budget must keep firing for in-memory input.
    agg = xr.DataArray(np.zeros((1000, 1000), dtype=np.float32),
                       dims=['y', 'x'])
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32)
    with patch('xrspatial.focal._available_memory_bytes',
               return_value=1_000_000):
        with pytest.raises(MemoryError, match="padded raster"):
            apply(agg, kernel)


@dask_array_available
def test_memory_guard_dask_oversize_kernel_still_rejected_3218():
    # An oversized kernel must still be rejected on the dask path: the
    # kernel itself plus one padded chunk blows the per-task budget. The
    # message reports the chunk, not the raster.
    data = da.zeros((1000, 1000), chunks=(100, 100), dtype=np.float32)
    agg = xr.DataArray(data, dims=['y', 'x'])
    kernel = np.ones((301, 301), dtype=np.float32)
    with patch('xrspatial.focal._available_memory_bytes',
               return_value=1_000_000):
        with pytest.raises(MemoryError, match="padded chunk"):
            apply(agg, kernel)


def test_apply_small_kernel_not_rejected_1284():
    # The guard must not fire for realistic kernel + raster combos.
    raster = xr.DataArray(np.ones((50, 50), dtype=np.float32))
    kernel = circle_kernel(1, 1, 3)
    out = apply(raster, kernel)
    assert out.shape == (50, 50)


@pytest.mark.parametrize("entry_point", [
    lambda agg, kernel: apply(agg, kernel),
    lambda agg, kernel: focal_stats(agg, kernel, stats_funcs=['mean']),
    lambda agg, kernel: hotspots(agg, kernel),
])
@pytest.mark.parametrize("bad_kernel", [
    np.ones(3, dtype=np.float32),            # 1D
    np.ones((3, 3, 3), dtype=np.float32),    # 3D
])
def test_entry_points_reject_non_2d_kernel_2842(entry_point, bad_kernel):
    # Regression for #2842: a non-2D kernel must raise a clear, descriptive
    # error rather than the raw "not enough values to unpack" ValueError
    # leaking out of custom_kernel's `rows, cols = kernel.shape`.
    raster = xr.DataArray(np.ones((10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="not a 2D array"):
        entry_point(raster, bad_kernel)


def test_convolution_numpy(
    convolve_2d_data,
    convolution_custom_kernel,
    kernel_circle_1_1_1,
    convolution_kernel_circle_1_1_1,
    kernel_annulus_2_2_2_1,
    convolution_kernel_annulus_2_2_1
):
    kernel_custom, expected_result_custom = convolution_custom_kernel
    result_kernel_custom = convolve_2d(convolve_2d_data, kernel_custom)
    assert isinstance(result_kernel_custom, np.ndarray)
    np.testing.assert_allclose(
        result_kernel_custom, expected_result_custom, equal_nan=True
    )

    result_kernel_circle = convolve_2d(convolve_2d_data, kernel_circle_1_1_1)
    assert isinstance(result_kernel_circle, np.ndarray)
    np.testing.assert_allclose(
        result_kernel_circle, convolution_kernel_circle_1_1_1, equal_nan=True
    )

    result_kernel_annulus = convolve_2d(convolve_2d_data, kernel_annulus_2_2_2_1)
    assert isinstance(result_kernel_annulus, np.ndarray)
    np.testing.assert_allclose(
        result_kernel_annulus, convolution_kernel_annulus_2_2_1, equal_nan=True
    )


@dask_array_available
def test_convolution_dask_numpy(
    convolve_2d_data,
    convolution_custom_kernel,
    kernel_circle_1_1_1,
    convolution_kernel_circle_1_1_1,
    kernel_annulus_2_2_2_1,
    convolution_kernel_annulus_2_2_1
):
    dask_agg = create_test_raster(convolve_2d_data, backend='dask+numpy')

    kernel_custom, expected_result_custom = convolution_custom_kernel
    result_kernel_custom = convolution_2d(dask_agg, kernel_custom)
    assert isinstance(result_kernel_custom.data, da.Array)
    np.testing.assert_allclose(
        result_kernel_custom.compute(), expected_result_custom, equal_nan=True
    )

    result_kernel_circle = convolution_2d(dask_agg, kernel_circle_1_1_1)
    assert isinstance(result_kernel_circle.data, da.Array)
    np.testing.assert_allclose(
        result_kernel_circle.compute(), convolution_kernel_circle_1_1_1, equal_nan=True
    )

    result_kernel_annulus = convolution_2d(dask_agg, kernel_annulus_2_2_2_1)
    assert isinstance(result_kernel_annulus.data, da.Array)
    np.testing.assert_allclose(
        result_kernel_annulus.compute(), convolution_kernel_annulus_2_2_1, equal_nan=True
    )


@cuda_and_cupy_available
def test_2d_convolution_gpu(
    convolve_2d_data,
    convolution_custom_kernel,
    kernel_circle_1_1_1,
    convolution_kernel_circle_1_1_1,
    kernel_annulus_2_2_2_1,
    convolution_kernel_annulus_2_2_1
):
    import cupy
    cupy_data = cupy.asarray(convolve_2d_data)

    kernel_custom, expected_result_custom = convolution_custom_kernel
    result_kernel_custom = convolve_2d(cupy_data, kernel_custom)
    assert isinstance(result_kernel_custom, cupy.ndarray)
    np.testing.assert_allclose(
        result_kernel_custom.get(), expected_result_custom, equal_nan=True
    )

    result_kernel_circle = convolve_2d(cupy_data, kernel_circle_1_1_1)
    assert isinstance(result_kernel_circle, cupy.ndarray)
    np.testing.assert_allclose(
        result_kernel_circle.get(), convolution_kernel_circle_1_1_1, equal_nan=True
    )

    result_kernel_annulus = convolve_2d(cupy_data, kernel_annulus_2_2_2_1)
    assert isinstance(result_kernel_annulus, cupy.ndarray)
    np.testing.assert_allclose(
        result_kernel_annulus.get(), convolution_kernel_annulus_2_2_1, equal_nan=True
    )

    # dask + cupy case not implemented
    # TODO: break this into its own test.
    if da is not None:
        dask_cupy_agg = xr.DataArray(
            da.from_array(cupy.asarray(convolve_2d_data), chunks=(3, 3))
        )
        result_kernel_annulus = convolve_2d(dask_cupy_agg.data, kernel_annulus_2_2_2_1)
        assert isinstance(result_kernel_annulus, da.Array)
        np.testing.assert_allclose(
            result_kernel_annulus.compute().get(), convolution_kernel_annulus_2_2_1, equal_nan=True
        )


def test_calc_cellsize_unit_input_attrs(convolve_2d_data):
    agg = create_test_raster(convolve_2d_data, attrs={'res': (1, 1), 'unit': 'km'})
    cellsize = calc_cellsize(agg)
    assert cellsize == (1000, 1000)


def test_calc_cellsize_no_attrs(convolve_2d_data):
    agg = create_test_raster(convolve_2d_data)
    cellsize = calc_cellsize(agg)
    assert cellsize == (0.5, 0.5)


@pytest.fixture
def data_apply():
    data = np.array([[0, 1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10, 11],
                     [12, 13, 14, 15, 16, 17],
                     [18, 19, 20, 21, 22, 23]])
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    expected_result = np.zeros_like(data)
    return data, kernel, expected_result


def func_zero(x):
    return 0


@ngjit
def func_zero_cpu(x):
    return 0


def test_apply_numpy(data_apply):
    data, kernel, expected_result = data_apply
    numpy_agg = create_test_raster(data)
    numpy_apply = apply(numpy_agg, kernel, func_zero_cpu)
    general_output_checks(numpy_agg, numpy_apply, expected_result)


def test_apply_dask_numpy(data_apply):
    data, kernel, expected_result = data_apply
    dask_numpy_agg = create_test_raster(data, backend='dask')
    dask_numpy_apply = apply(dask_numpy_agg, kernel, func_zero_cpu)
    general_output_checks(dask_numpy_agg, dask_numpy_apply, expected_result)


@cuda_and_cupy_available
def test_apply_cupy(data_apply):
    from xrspatial.focal import _focal_mean_cuda

    data, kernel, expected_result_zero = data_apply
    # numpy reference using _calc_mean
    numpy_agg = create_test_raster(data)
    numpy_apply = apply(numpy_agg, kernel)

    # cupy case with equivalent CUDA kernel
    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_apply = apply(cupy_agg, kernel, _focal_mean_cuda)
    general_output_checks(cupy_agg, cupy_apply)

    np.testing.assert_allclose(
        numpy_apply.data, cupy_apply.data.get(),
        equal_nan=True, rtol=1e-4)


def test_apply_default_func_numpy_3215(data_apply):
    # default func=None resolves to _calc_mean on the numpy backend
    from xrspatial.focal import _calc_mean

    data, kernel, _ = data_apply
    numpy_agg = create_test_raster(data)
    default_apply = apply(numpy_agg, kernel)
    explicit_apply = apply(numpy_agg, kernel, _calc_mean)
    general_output_checks(numpy_agg, default_apply)
    np.testing.assert_allclose(
        default_apply.data, explicit_apply.data, equal_nan=True)


@dask_array_available
def test_apply_default_func_dask_numpy_3215(data_apply):
    from xrspatial.focal import _calc_mean

    data, kernel, _ = data_apply
    dask_numpy_agg = create_test_raster(data, backend='dask')
    default_apply = apply(dask_numpy_agg, kernel)
    explicit_apply = apply(dask_numpy_agg, kernel, _calc_mean)
    general_output_checks(dask_numpy_agg, default_apply)
    np.testing.assert_allclose(
        default_apply.data.compute(), explicit_apply.data.compute(),
        equal_nan=True)


@cuda_and_cupy_available
def test_apply_default_func_cupy_3215(data_apply):
    # apply(cupy_agg, kernel) used to raise TypeError because the default
    # func was the @ngjit _calc_mean, which cannot be launched as a CUDA
    # kernel. The default now resolves to _focal_mean_cuda on GPU backends.
    from xrspatial.focal import _focal_mean_cuda

    data, kernel, _ = data_apply
    numpy_default = apply(create_test_raster(data), kernel)

    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_default = apply(cupy_agg, kernel)
    explicit_apply = apply(cupy_agg, kernel, _focal_mean_cuda)
    general_output_checks(cupy_agg, cupy_default)

    np.testing.assert_allclose(
        cupy_default.data.get(), explicit_apply.data.get(), equal_nan=True)
    # default funcs on both backends compute the same focal mean
    np.testing.assert_allclose(
        numpy_default.data, cupy_default.data.get(),
        equal_nan=True, rtol=1e-4)


@dask_array_available
@cuda_and_cupy_available
def test_apply_default_func_dask_cupy_3215():
    rng = np.random.default_rng(7)
    data = rng.random((20, 24)).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    cupy_default = apply(create_test_raster(data, backend='cupy'), kernel)

    dask_cupy_agg = create_test_raster(data, backend='dask+cupy',
                                       chunks=(10, 12))
    dask_cupy_default = apply(dask_cupy_agg, kernel)
    general_output_checks(dask_cupy_agg, dask_cupy_default,
                          verify_attrs=False)

    # Compare interior (boundary='nan' causes edge differences between
    # cupy single-GPU bounds-clamping and dask map_overlap NaN-padding)
    pad = kernel.shape[0] // 2
    np.testing.assert_allclose(
        cupy_default.data[pad:-pad, pad:-pad].get(),
        dask_cupy_default.data[pad:-pad, pad:-pad].compute().get(),
        equal_nan=True, rtol=1e-4)


@dask_array_available
@cuda_and_cupy_available
def test_apply_dask_cupy():
    from xrspatial.focal import _focal_mean_cuda

    # Use a larger array so chunk interiors are meaningful
    rng = np.random.default_rng(42)
    data = rng.random((20, 24)).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    # cupy reference (same CUDA kernel)
    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_apply = apply(cupy_agg, kernel, _focal_mean_cuda)

    # dask + cupy case
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy', chunks=(10, 12))
    dask_cupy_apply = apply(dask_cupy_agg, kernel, _focal_mean_cuda)
    general_output_checks(dask_cupy_agg, dask_cupy_apply, verify_attrs=False)

    # Compare interior (boundary='nan' causes edge differences between
    # cupy single-GPU bounds-clamping and dask map_overlap NaN-padding)
    pad = kernel.shape[0] // 2
    np.testing.assert_allclose(
        cupy_apply.data[pad:-pad, pad:-pad].get(),
        dask_cupy_apply.data[pad:-pad, pad:-pad].compute().get(),
        equal_nan=True, rtol=1e-4)


@pytest.fixture
def data_focal_stats():
    data = np.arange(16).reshape(4, 4)
    kernel = custom_kernel(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]))
    expected_result = np.asarray([
        # mean
        [[0, 1, 2, 3.],
         [4, 2.5,  3.5,  4.5],
         [8, 6.5,  7.5,  8.5],
         [12, 10.5,  11.5,  12.5]],
        # max
        [[0, 1, 2, 3.],
         [4, 5, 6, 7.],
         [8, 9, 10, 11.],
         [12, 13, 14, 15.]],
        # min
        [[0, 1, 2, 3.],
         [4, 0, 1, 2.],
         [8, 4, 5, 6.],
         [12, 8, 9, 10.]],
        # range
        [[0, 0, 0, 0.],
         [0, 5, 5, 5.],
         [0, 5, 5, 5.],
         [0, 5, 5, 5.]],
        # std
        [[0, 0, 0, 0.],
         [0, 2.5,  2.5,  2.5],
         [0, 2.5,  2.5,  2.5],
         [0, 2.5,  2.5,  2.5]],
        # var
        [[0, 0, 0, 0.],
         [0, 6.25, 6.25, 6.25],
         [0, 6.25, 6.25, 6.25],
         [0, 6.25, 6.25, 6.25]],
        # sum
        [[0, 1, 2, 3.],
         [4, 5, 7, 9.],
         [8, 13, 15, 17.],
         [12, 21, 23, 25.]],
        # variety -- arange(16) so every value is unique;
        # kernel hits center + upper-left diagonal only
        [[1, 1, 1, 1.],
         [1, 2, 2, 2.],
         [1, 2, 2, 2.],
         [1, 2, 2, 2.]]
    ])
    return data, kernel, expected_result


def test_focal_stats_numpy(data_focal_stats):
    data, kernel, expected_result = data_focal_stats
    numpy_agg = create_test_raster(data)
    numpy_focalstats = focal_stats(numpy_agg, kernel)
    general_output_checks(
        numpy_agg, numpy_focalstats, verify_attrs=False, expected_results=expected_result
    )
    assert numpy_focalstats.ndim == 3


def test_focal_stats_dask_numpy(data_focal_stats):
    data, kernel, expected_result = data_focal_stats
    dask_numpy_agg = create_test_raster(data, backend='dask')
    dask_numpy_focalstats = focal_stats(dask_numpy_agg, kernel)
    general_output_checks(
        dask_numpy_agg, dask_numpy_focalstats, verify_attrs=False, expected_results=expected_result
    )


@cuda_and_cupy_available
def test_focal_stats_gpu(data_focal_stats):
    data, kernel, expected_result = data_focal_stats
    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_focalstats = focal_stats(cupy_agg, kernel)
    general_output_checks(
        cupy_agg, cupy_focalstats, verify_attrs=False, expected_results=expected_result
    )


@dask_array_available
@cuda_and_cupy_available
def test_focal_stats_dask_cupy():
    # Use larger data so chunk interiors are meaningful
    rng = np.random.default_rng(42)
    data = rng.random((20, 24)).astype(np.float64)
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    # cupy reference
    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_focalstats = focal_stats(cupy_agg, kernel)

    # dask + cupy case
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy', chunks=(10, 12))
    dask_cupy_focalstats = focal_stats(dask_cupy_agg, kernel)
    assert dask_cupy_focalstats.ndim == 3

    # Compare interior (boundary='nan' causes edge differences between
    # cupy single-GPU bounds-clamping and dask map_overlap NaN-padding)
    pad = kernel.shape[0] // 2
    np.testing.assert_allclose(
        cupy_focalstats.data[:, pad:-pad, pad:-pad].get(),
        dask_cupy_focalstats.data[:, pad:-pad, pad:-pad].compute().get(),
        equal_nan=True, rtol=1e-4)


@cuda_and_cupy_available
def test_focal_stats_cupy_casts_input_once_3231():
    # Regression for #3231: the cupy stats loop re-ran agg.data.astype()
    # (a full-raster device copy, even for an unchanged dtype) once per
    # stat. The input cast is now hoisted above the loop, so
    # _promote_float runs once for the input plus once per stat for the
    # output allocation inside _focal_stats_func_cupy, plus one
    # dtype-only call in the memory guard at the entry point (#3223).
    import xrspatial.focal as focal_module
    data = np.arange(48, dtype=np.float64).reshape(6, 8)
    agg = create_test_raster(data, backend='cupy')
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    stats = ['mean', 'sum', 'min', 'max']
    with patch.object(focal_module, '_promote_float',
                      wraps=focal_module._promote_float) as spy:
        result = focal_stats(agg, kernel, stats_funcs=stats)
    assert result.shape == (len(stats), 6, 8)
    assert spy.call_count == len(stats) + 2


# --- float64 preservation (issue-2769) ------------------------------------
# apply() and focal_stats() used to cast every input to float32 internally,
# silently downcasting float64 rasters. convolve_2d() preserves the input
# floating dtype (see test_convolution); these mirror that contract for the
# focal APIs across all four backends.


def _compute_dtype(agg):
    """Return the materialised dtype regardless of backend."""
    data = agg.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'get'):
        data = data.get()
    return data.dtype


@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
def test_apply_preserves_float64(backend):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    if 'cupy' in backend:
        from xrspatial.focal import _focal_mean_cuda
        func = _focal_mean_cuda
    else:
        from xrspatial.focal import _calc_mean
        func = _calc_mean

    agg = create_test_raster(data, backend=backend, chunks=(2, 3))
    result = apply(agg, kernel, func)

    assert _compute_dtype(result) == np.float64


@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
def test_focal_stats_preserves_float64(backend):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    agg = create_test_raster(data, backend=backend, chunks=(2, 3))
    result = focal_stats(agg, kernel,
                         stats_funcs=['mean', 'sum', 'min', 'max',
                                      'std', 'var', 'range'])

    assert _compute_dtype(result) == np.float64


# --- non-binary kernel rejection (issue-2848) -----------------------------
# apply() and focal_stats() document the kernel as a binary membership mask
# ("values of 1 indicate the kernel"). The CPU path only kept cells equal to
# 1 (dropping a weight of 2 entirely) while the GPU sum/mean kernels weighted
# every nonzero cell by its value, so the same non-binary kernel produced
# backend-dependent output. Both APIs now reject non-binary kernels on every
# backend, so the inconsistency cannot arise.

NON_BINARY_KERNELS = [
    np.array([[0, 2, 0], [2, 2, 2], [0, 2, 0]]),   # all weights are 2
    np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]),   # mixed 1 and 2
    np.array([[0, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 0]]),  # fractional weights
]


@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
@pytest.mark.parametrize("kernel", NON_BINARY_KERNELS)
def test_apply_rejects_non_binary_kernel(backend, kernel):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    agg = create_test_raster(data, backend=backend, chunks=(2, 3))

    with pytest.raises(ValueError, match="kernel must be binary"):
        apply(agg, kernel)


@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
@pytest.mark.parametrize("kernel", NON_BINARY_KERNELS)
def test_focal_stats_rejects_non_binary_kernel(backend, kernel):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    agg = create_test_raster(data, backend=backend, chunks=(2, 3))

    with pytest.raises(ValueError, match="kernel must be binary"):
        focal_stats(agg, kernel, stats_funcs=['mean', 'sum'])


@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
def test_apply_and_focal_stats_accept_binary_kernel(backend):
    # The binary 0/1 contract still works on every backend after the guard.
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    agg = create_test_raster(data, backend=backend, chunks=(2, 3))

    if 'cupy' in backend:
        from xrspatial.focal import _focal_mean_cuda
        func = _focal_mean_cuda
    else:
        from xrspatial.focal import _calc_mean
        func = _calc_mean

    apply_result = apply(agg, kernel, func)
    general_output_checks(agg, apply_result)

    stats_result = focal_stats(agg, kernel, stats_funcs=['mean', 'sum'])
    assert stats_result.ndim == 3


def test_apply_rejects_nan_kernel():
    # A NaN kernel cell is neither 0 nor 1, so it is rejected like any other
    # non-binary value (the error message calls out NaN explicitly).
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    kernel = np.array([[0, np.nan, 0], [1, 1, 1], [0, 1, 0]])
    agg = xr.DataArray(data, dims=['y', 'x'])
    with pytest.raises(ValueError, match="kernel must be binary"):
        apply(agg, kernel)


def test_apply_focal_stats_agree_on_binary_kernel_numpy():
    # Reference invariant the guard protects: on a binary kernel,
    # focal_stats(mean) and apply(mean) agree (focal_stats delegates to
    # apply on the CPU). This is the consistency the issue is about; with a
    # non-binary kernel the two would have diverged, which is now blocked.
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    agg = xr.DataArray(data, dims=['y', 'x'])

    from xrspatial.focal import _calc_mean
    apply_mean = apply(agg, kernel, _calc_mean)
    stats_mean = focal_stats(agg, kernel, stats_funcs=['mean']).sel(stats='mean')

    np.testing.assert_allclose(
        apply_mean.data, stats_mean.data, equal_nan=True)


@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
def test_apply_keeps_float32(backend):
    # The other side of the contract: a float32 input must not be promoted
    # to float64. (On dask the lazy dtype is float64, but the computed
    # result is float32 -- matching convolve_2d.)
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    if 'cupy' in backend:
        from xrspatial.focal import _focal_mean_cuda
        func = _focal_mean_cuda
    else:
        from xrspatial.focal import _calc_mean
        func = _calc_mean

    agg = create_test_raster(data, backend=backend, chunks=(2, 3))
    result = apply(agg, kernel, func)

    assert _compute_dtype(result) == np.float32


# --- mean dtype preservation (issue-3214) ----------------------------------
# mean() used to cast to float64 on numpy/dask+numpy (agg.data.astype(float))
# and to float32 on cupy/dask+cupy, so the output dtype depended on the
# backend and float64 rasters lost precision on GPU. It now follows the same
# _promote_float contract as apply()/focal_stats() (#2769) and convolve_2d()
# (#1096): float dtypes are preserved, ints promote to float32.


@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
def test_mean_preserves_float64_3214(backend):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    agg = create_test_raster(data, backend=backend, chunks=(2, 3))
    result = mean(agg)
    assert _compute_dtype(result) == np.float64


@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
def test_mean_keeps_float32_3214(backend):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    agg = create_test_raster(data, backend=backend, chunks=(2, 3))
    result = mean(agg)
    assert _compute_dtype(result) == np.float32


@pytest.mark.parametrize("backend", ['numpy', 'cupy'])
def test_mean_int_promotes_to_float32_3214(backend):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    data = np.arange(20, dtype=np.int32).reshape(4, 5)
    agg = create_test_raster(data, backend=backend)
    result = mean(agg)
    assert _compute_dtype(result) == np.float32


@cuda_and_cupy_available
def test_mean_large_offset_gpu_matches_numpy_3214():
    """GPU mean() must not collapse on large-offset float64 rasters (#3214).

    The old cupy path cast float64 input to float32. At an offset of 1e7
    the float32 resolution (~1) exceeds the spread of the true focal means
    (~0.4), so the GPU result carried no signal. Computing in float64 keeps
    the two backends in agreement.
    """
    import cupy
    rng = np.random.default_rng(1)
    data = (1e7 + rng.random((8, 8))).astype(np.float64)

    numpy_mean = mean(xr.DataArray(data))
    cupy_mean = mean(xr.DataArray(cupy.asarray(data)))

    np.testing.assert_allclose(
        numpy_mean.data, cupy_mean.data.get(), rtol=1e-12, equal_nan=True)
    # the focal-mean variation must survive the round trip
    spread = numpy_mean.data.max() - numpy_mean.data.min()
    gpu_spread = cupy_mean.data.get().max() - cupy_mean.data.get().min()
    assert spread > 0.1
    assert abs(gpu_spread - spread) < 1e-6


@pytest.mark.parametrize("backend", ['numpy', 'cupy'])
def test_mean_excludes_match_in_working_dtype_3214(backend):
    # excludes are cast to the working dtype before matching, so a float32
    # raster cell equal to float32(0.1) is excluded even though
    # float64(0.1) != float32(0.1). The cupy path casts excludes separately
    # in _mean_cupy, so both backends are checked.
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    data = np.full((4, 4), 5.0, dtype=np.float32)
    data[1, 1] = np.float32(0.1)
    agg = create_test_raster(data, backend=backend)
    result = mean(agg, excludes=[0.1])
    out = result.data.get() if hasattr(result.data, 'get') else result.data
    # the excluded cell is passed through unchanged
    assert out[1, 1] == np.float32(0.1)


# --- focal_stats NaN handling (issue-1092) --------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'dask+numpy', 'dask+cupy'])
def test_focal_stats_nan_handling_1092(backend):
    """All backends should skip NaN neighbors, not propagate them.

    Regression test for #1092: CUDA kernels propagated NaN through
    arithmetic instead of skipping.
    """
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = np.array([
        [1.0, np.nan, 3.0],
        [4.0,   5.0, 6.0],
        [7.0,   8.0, 9.0],
    ])
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    agg = create_test_raster(data, backend=backend, chunks=(3, 3))
    result = focal_stats(agg, kernel,
                         stats_funcs=['mean', 'sum', 'min', 'max', 'std', 'var', 'range'])

    if hasattr(result.data, 'compute'):
        result = result.compute()

    def _val(stat, r, c):
        d = result.sel(stats=stat).data
        if hasattr(d, 'get'):
            d = d.get()
        return float(np.asarray(d)[r, c])

    # Center pixel (1,1): kernel hits [NaN, 4, 5, 6, 8] -> skip NaN -> [4,5,6,8]
    center_vals = np.array([4.0, 5.0, 6.0, 8.0])
    atol = 1e-3  # float32 tolerance

    mean_val = _val('mean', 1, 1)
    sum_val = _val('sum', 1, 1)
    min_val = _val('min', 1, 1)
    max_val = _val('max', 1, 1)
    std_val = _val('std', 1, 1)
    var_val = _val('var', 1, 1)
    range_val = _val('range', 1, 1)

    assert abs(mean_val - np.nanmean(center_vals)) < atol, f"mean={mean_val}"
    assert abs(sum_val - np.nansum(center_vals)) < atol, f"sum={sum_val}"
    assert abs(min_val - np.nanmin(center_vals)) < atol, f"min={min_val}"
    assert abs(max_val - np.nanmax(center_vals)) < atol, f"max={max_val}"
    assert abs(std_val - np.nanstd(center_vals)) < atol, f"std={std_val}"
    assert abs(var_val - np.nanvar(center_vals)) < atol, f"var={var_val}"
    assert abs(range_val - (np.nanmax(center_vals) - np.nanmin(center_vals))) < atol, (
        f"range={range_val}"
    )

    # Top-left corner (0,0): kernel hits [NaN, 4, 1] (cross pattern)
    # NaN is from data[0,1] (up direction is OOB, left is OOB)
    # Wait: the cross kernel at (0,0) covers:
    #   up=(-1,0)=OOB, down=(1,0)=4, left=(0,-1)=OOB, right=(0,1)=NaN, center=(0,0)=1
    # So valid values = [1, 4], NaN is skipped
    corner_vals = np.array([1.0, 4.0])
    mean_corner = _val('mean', 0, 0)
    assert abs(mean_corner - np.nanmean(corner_vals)) < atol, (
        f"corner mean={mean_corner}"
    )


@pytest.mark.parametrize("backend", ['numpy', 'cupy'])
def test_focal_stats_all_nan_window_1092(backend):
    """A pixel whose entire kernel window is NaN should produce NaN."""
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    data = np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, 1.0],
    ])
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    agg = create_test_raster(data, backend=backend)
    result = focal_stats(agg, kernel, stats_funcs=['mean', 'sum', 'min', 'max'])

    if hasattr(result.data, 'compute'):
        result = result.compute()

    def _val(stat, r, c):
        d = result.sel(stats=stat).data
        if hasattr(d, 'get'):
            d = d.get()
        return float(np.asarray(d)[r, c])

    # Center pixel (1,1): kernel hits [NaN, NaN, NaN, NaN, NaN] -> all NaN
    assert np.isnan(_val('mean', 1, 1))
    assert _val('sum', 1, 1) == 0.0  # nansum of all-NaN = 0 (numpy behavior)
    assert np.isnan(_val('min', 1, 1))
    assert np.isnan(_val('max', 1, 1))


# --- GPU std/var precision on large-offset rasters (issue-2831) -----------


@cuda_and_cupy_available
@pytest.mark.parametrize("offset", [1e6, 1e7])
def test_focal_stats_std_var_large_offset_gpu_matches_numpy_2831(offset):
    """GPU std/var must not collapse on large-offset rasters (#2831).

    The old GPU kernels used a one-pass E[x^2] - E[x]^2 variance in
    float32. On values with a large offset (~1e6-1e7) the two terms are
    nearly equal, so the subtraction lost all precision and the result
    collapsed toward zero -- diverging from the float64 two-pass numpy
    path. The two-pass kernel subtracts the window mean before squaring,
    which holds precision at any offset.
    """
    rng = np.random.default_rng(0)
    data = (offset + rng.random((8, 8))).astype(np.float64)
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    numpy_agg = create_test_raster(data, backend='numpy')
    cupy_agg = create_test_raster(data, backend='cupy')

    np_res = focal_stats(numpy_agg, kernel, stats_funcs=['std', 'var'])
    cp_res = focal_stats(cupy_agg, kernel, stats_funcs=['std', 'var'])

    for stat in ['std', 'var']:
        # Interior only: boundary='nan' blanks the outer ring identically
        # on both backends, but the interior is where the variance lives.
        np_interior = np_res.sel(stats=stat).data[1:-1, 1:-1]
        cp_interior = cp_res.sel(stats=stat).data.get()[1:-1, 1:-1]
        # The variance is ~0.08 here; the old one-pass kernel returned ~0.
        assert np.nanmax(np_interior) > 0.01, f"{stat} reference is flat"
        np.testing.assert_allclose(
            cp_interior, np_interior, rtol=1e-3, atol=1e-4,
            err_msg=f"{stat} diverges at offset {offset:g}")


# --- focal variety (issue-1040) ------------------------------------------

def _variety_reference_data():
    """Categorical 6x6 grid with known variety counts for a 3x3 box kernel."""
    data = np.array([
        [1, 1, 2, 2, 3, 3],
        [1, 1, 2, 2, 3, 3],
        [4, 4, 5, 5, 6, 6],
        [4, 4, 5, 5, 6, 6],
        [7, 7, 8, 8, 9, 9],
        [7, 7, 8, 8, 9, 9],
    ], dtype=np.float64)
    kernel = np.ones((3, 3))
    return data, kernel


def test_variety_numpy():
    data, kernel = _variety_reference_data()
    agg = create_test_raster(data)
    result = focal_stats(agg, kernel, stats_funcs=['variety'])
    vals = result.sel(stats='variety').values
    # Interior pixel (2,2) sees values {1,2,4,5} -> 4
    assert vals[2, 2] == 4.0
    # Corner pixel (0,0) window is 2x2 -> {1,1,1,1} -> 1
    assert vals[0, 0] == 1.0
    # Edge pixel (0,2) window is 2x3 -> {1,1,2,2,2,2} -> 2
    assert vals[0, 2] == 2.0
    # All interior values should be positive integers
    assert np.all(vals[1:-1, 1:-1] >= 1)


@dask_array_available
def test_variety_dask_numpy():
    data, kernel = _variety_reference_data()
    np_agg = create_test_raster(data)
    dk_agg = create_test_raster(data, backend='dask')
    np_result = focal_stats(np_agg, kernel, stats_funcs=['variety'])
    dk_result = focal_stats(dk_agg, kernel, stats_funcs=['variety'])
    np.testing.assert_allclose(
        np_result.values, dk_result.values, equal_nan=True)


def test_variety_nan_handling():
    """NaN cells should not count as a distinct value."""
    data = np.array([
        [1, np.nan, 2],
        [1,     1, 2],
        [3,     3, 3],
    ], dtype=np.float64)
    kernel = np.ones((3, 3))
    agg = create_test_raster(data)
    result = focal_stats(agg, kernel, stats_funcs=['variety'])
    vals = result.sel(stats='variety').values
    # Center pixel (1,1) sees {1,2,1,1,2,3,3,3} (NaN skipped) -> {1,2,3} -> 3
    assert vals[1, 1] == 3.0


def test_variety_all_nan():
    """A window of all NaN should produce NaN variety."""
    data = np.full((3, 3), np.nan)
    kernel = np.ones((3, 3))
    agg = create_test_raster(data)
    result = focal_stats(agg, kernel, stats_funcs=['variety'])
    vals = result.sel(stats='variety').values
    assert np.all(np.isnan(vals))


def test_variety_single_cell():
    """1x1 raster, 1x1 kernel -> variety 1."""
    data = np.array([[42.0]])
    kernel = np.ones((1, 1))
    agg = create_test_raster(data)
    result = focal_stats(agg, kernel, stats_funcs=['variety'])
    assert result.sel(stats='variety').values.item() == 1.0


def _all_unique_window(n):
    """(n x n) raster of all-distinct values with an (n x n) all-ones kernel.

    The center pixel's window is the whole raster, so its variety equals n*n.
    """
    data = np.arange(n * n, dtype=np.float64).reshape(n, n)
    kernel = np.ones((n, n))
    return data, kernel


@pytest.mark.parametrize("n", [7, 9])
def test_variety_large_kernel_numpy(n):
    """Variety must not cap below the true distinct count for large kernels."""
    data, kernel = _all_unique_window(n)
    agg = create_test_raster(data)
    result = focal_stats(agg, kernel, stats_funcs=['variety'])
    vals = result.sel(stats='variety').values
    center = n // 2
    assert vals[center, center] == float(n * n)


@cuda_and_cupy_available
@pytest.mark.parametrize("n", [7, 9])
def test_variety_gpu_large_kernel_parity(n):
    """GPU variety must match numpy for kernels larger than 5x5 (#2775).

    The old CUDA kernel capped unique counts at 25, so a 7x7 all-unique
    window returned 25 on GPU vs 49 on CPU. The center pixel's window covers
    the whole raster, so its variety equals n*n (49 for 7x7, 81 for 9x9).
    """
    data, kernel = _all_unique_window(n)
    np_agg = create_test_raster(data)
    cupy_agg = create_test_raster(data, backend='cupy')
    np_result = focal_stats(np_agg, kernel, stats_funcs=['variety'])
    cupy_result = focal_stats(cupy_agg, kernel, stats_funcs=['variety'])
    center = n // 2
    assert cupy_result.sel(stats='variety').data.get()[center, center] == float(n * n)
    np.testing.assert_allclose(
        np_result.values, cupy_result.data.get(), equal_nan=True)


@pytest.fixture
def data_hotspots():
    # Clusters sit fully in the interior (away from the outer ring) so the
    # single-array NaN-edge behavior and the dask map_overlap result agree
    # on every cell. NaN cells are excluded from the Gi* sums and stats.
    data = np.asarray([
        [np.nan, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 10000., 10000., 10000., 0., 0., 0., 0., 0., 0.],
        [0., 10000., 10000., 10000., 0., 0., np.nan, 0., 0., 0.],
        [0., 10000., 10000., 10000., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., -10000., -10000., -10000., 0.],
        [0., 0., 0., 0., 0., 0., -10000., -10000., -10000., 0.],
        [0., 0., 0., 0., 0., 0., -10000., -10000., -10000., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    expected_result = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 99, 99, 99, 0, 0, 0, 0, 0, 0],
        [0, 99, 99, 99, 0, 0, 0, 0, 0, 0],
        [0, 99, 99, 99, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -99, -99, -99, 0],
        [0, 0, 0, 0, 0, 0, -99, -99, -99, 0],
        [0, 0, 0, 0, 0, 0, -99, -99, -99, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int8)

    return data, kernel, expected_result


def test_hotspots_zero_global_std():
    data = np.zeros((10, 20))
    agg = create_test_raster(data)
    kernel = np.ones((3, 3))
    msg = "Standard deviation of the input raster values is 0."
    with pytest.raises(ZeroDivisionError, match=msg):
        hotspots(agg, kernel)


# Degenerate inputs: constant raster (std == 0), all-NaN raster (n == 0),
# a single valid cell (n == 1), and a raster containing Inf, which makes
# the global std NaN (issue #3219). The numpy/cupy paths raise eagerly via
# _gistar_global_stats; the dask paths must raise the same error at compute
# time instead of silently classifying to all zeros (issue #2843).
def _hotspots_degenerate_cases():
    constant = np.zeros((10, 10), dtype=np.float32)

    all_nan = np.full((10, 10), np.nan, dtype=np.float32)

    single_valid = np.full((10, 10), np.nan, dtype=np.float32)
    single_valid[0, 0] = 5.0

    # A single Inf of either sign poisons nanstd the same way (std == NaN);
    # both signs are pinned so neither regresses independently.
    pos_inf = np.arange(100, dtype=np.float32).reshape(10, 10)
    pos_inf[4, 4] = np.inf

    neg_inf = np.arange(100, dtype=np.float32).reshape(10, 10)
    neg_inf[4, 4] = -np.inf

    std_msg = "Standard deviation of the input raster values is 0."
    n_msg = "needs at least 2 valid"
    finite_msg = "Standard deviation of the input raster values is not finite"
    return [
        ('constant', constant, ZeroDivisionError, std_msg),
        ('all_nan', all_nan, ValueError, n_msg),
        ('single_valid', single_valid, ValueError, n_msg),
        ('pos_inf', pos_inf, ValueError, finite_msg),
        ('neg_inf', neg_inf, ValueError, finite_msg),
    ]


_HOTSPOTS_DEGENERATE = _hotspots_degenerate_cases()


@pytest.mark.parametrize('case,data,exc,msg', _HOTSPOTS_DEGENERATE,
                         ids=[c[0] for c in _HOTSPOTS_DEGENERATE])
def test_hotspots_degenerate_numpy_2843(case, data, exc, msg):
    agg = create_test_raster(data)
    kernel = np.ones((3, 3))
    with pytest.raises(exc, match=msg):
        hotspots(agg, kernel)


@dask_array_available
@pytest.mark.parametrize('case,data,exc,msg', _HOTSPOTS_DEGENERATE,
                         ids=[c[0] for c in _HOTSPOTS_DEGENERATE])
def test_hotspots_degenerate_dask_numpy_2843(case, data, exc, msg):
    # The dask backend must reject degenerate inputs the same way numpy does,
    # but lazily: the error fires at compute(), not at graph-build time.
    agg = create_test_raster(data, backend='dask')
    kernel = np.ones((3, 3))
    result = hotspots(agg, kernel)
    with pytest.raises(exc, match=msg):
        result.data.compute()


@cuda_and_cupy_available
@pytest.mark.parametrize('case,data,exc,msg', _HOTSPOTS_DEGENERATE,
                         ids=[c[0] for c in _HOTSPOTS_DEGENERATE])
def test_hotspots_degenerate_cupy_2843(case, data, exc, msg):
    agg = create_test_raster(data, backend='cupy')
    kernel = np.ones((3, 3))
    with pytest.raises(exc, match=msg):
        hotspots(agg, kernel)


@cuda_and_cupy_available
@dask_array_available
@pytest.mark.parametrize('case,data,exc,msg', _HOTSPOTS_DEGENERATE,
                         ids=[c[0] for c in _HOTSPOTS_DEGENERATE])
def test_hotspots_degenerate_dask_cupy_2843(case, data, exc, msg):
    agg = create_test_raster(data, backend='dask+cupy')
    kernel = np.ones((3, 3))
    result = hotspots(agg, kernel)
    with pytest.raises(exc, match=msg):
        result.data.compute()


def test_hotspots_kernel_none_2771():
    # Regression for #2771: hotspots skipped custom_kernel validation, so a
    # None kernel raised AttributeError on kernel.shape instead of ValueError.
    agg = create_test_raster(np.ones((10, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        hotspots(agg, None)


def test_hotspots_kernel_list_of_list_2771():
    # Regression for #2771: a list-of-list kernel reached kernel.shape and
    # raised AttributeError; it should be rejected as a non-ndarray.
    agg = create_test_raster(np.ones((10, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        hotspots(agg, [[1, 1, 1]])


def test_hotspots_kernel_even_dim_2771():
    # Regression for #2771: an even-dimensioned kernel silently succeeded
    # before; custom_kernel now rejects it for improper shape.
    agg = create_test_raster(np.ones((10, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        hotspots(agg, np.ones((2, 2)))


def test_hotspots_kernel_zero_sum_2771():
    # Regression for #2771: hotspots normalizes by kernel.sum(); a zero-sum
    # kernel divided by zero instead of raising a clear error.
    agg = create_test_raster(np.ones((10, 10), dtype=np.float32))
    kernel = np.zeros((3, 3))
    with pytest.raises(ValueError, match=r"hotspots\(\): kernel sums to zero"):
        hotspots(agg, kernel)


def test_hotspots_valid_kernel_happy_path_2771(data_hotspots):
    # Regression for #2771: the added validation must not reject valid kernels.
    data, kernel, expected_result = data_hotspots
    numpy_agg = create_test_raster(data)
    numpy_hotspots = hotspots(numpy_agg, kernel)
    general_output_checks(numpy_agg, numpy_hotspots, expected_result, verify_attrs=False)


def test_hotspots_numpy(data_hotspots):
    data, kernel, expected_result = data_hotspots
    numpy_agg = create_test_raster(data)
    numpy_hotspots = hotspots(numpy_agg, kernel)
    general_output_checks(numpy_agg, numpy_hotspots, expected_result, verify_attrs=False)
    # validate attrs
    assert numpy_hotspots.shape == numpy_agg.shape
    assert numpy_hotspots.dims == numpy_agg.dims
    for coord in numpy_agg.coords:
        np.testing.assert_allclose(
            numpy_hotspots[coord].data, numpy_agg[coord].data, equal_nan=True
        )
    assert numpy_hotspots.attrs['unit'] == '%'


@dask_array_available
def test_hotspots_dask_numpy(data_hotspots):
    data, kernel, expected_result = data_hotspots
    dask_numpy_agg = create_test_raster(data, backend='dask')
    dask_numpy_hotspots = hotspots(dask_numpy_agg, kernel)
    general_output_checks(dask_numpy_agg, dask_numpy_hotspots, expected_result, verify_attrs=False)
    # validate attrs
    assert dask_numpy_hotspots.shape == dask_numpy_agg.shape
    assert dask_numpy_hotspots.dims == dask_numpy_agg.dims
    for coord in dask_numpy_agg.coords:
        np.testing.assert_allclose(
            dask_numpy_hotspots[coord].data, dask_numpy_agg[coord].data, equal_nan=True
        )
    assert dask_numpy_hotspots.attrs['unit'] == '%'


@cuda_and_cupy_available
def test_hotspot_gpu(data_hotspots):
    data, kernel, expected_result = data_hotspots
    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_hotspots = hotspots(cupy_agg, kernel)
    general_output_checks(cupy_agg, cupy_hotspots, expected_result, verify_attrs=False)
    # validate attrs
    assert cupy_hotspots.shape == cupy_agg.shape
    assert cupy_hotspots.dims == cupy_agg.dims
    for coord in cupy_agg.coords:
        np.testing.assert_allclose(
            cupy_hotspots[coord].data, cupy_agg[coord].data, equal_nan=True
        )
    assert cupy_hotspots.attrs['unit'] == '%'


@dask_array_available
@cuda_and_cupy_available
def test_hotspots_dask_cupy():
    import cupy

    # Use a larger array so chunk interiors are meaningful
    rng = np.random.default_rng(42)
    data = rng.random((20, 24)).astype(np.float64) * 1000
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])

    # cupy reference (classification runs on the GPU)
    cupy_agg = create_test_raster(data, backend='cupy')
    cupy_hotspots = hotspots(cupy_agg, kernel)

    # dask + cupy case
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy', chunks=(10, 12))
    dask_cupy_hotspots = hotspots(dask_cupy_agg, kernel)
    general_output_checks(dask_cupy_agg, dask_cupy_hotspots, verify_attrs=False)

    # the result must stay a cupy-backed dask array end to end
    assert isinstance(dask_cupy_hotspots.data, da.Array)
    assert isinstance(dask_cupy_hotspots.data._meta, cupy.ndarray)

    # Compare interior (boundary='nan' causes edge differences between
    # cupy single-GPU bounds-clamping and dask map_overlap NaN-padding)
    pad = kernel.shape[0] // 2
    np.testing.assert_array_equal(
        cupy_hotspots.data[pad:-pad, pad:-pad].get(),
        dask_cupy_hotspots.data[pad:-pad, pad:-pad].compute().get())


def test_hotspots_classifier_thresholds_3737():
    # Regression for #3737: the z-score classifier was rewritten from a
    # nine-branch threshold ladder into boolean arithmetic. Pin the
    # classification at every threshold and its float64 neighbours, at
    # both signs, so the open/closed side of each interval cannot drift.
    from xrspatial.focal import _calc_hotspots_numpy

    thresholds = np.array([1.29, 1.65, 1.96, 2.33, 2.58])
    below = np.nextafter(thresholds, -np.inf)
    above = np.nextafter(thresholds, np.inf)
    z = np.stack([below, thresholds, above])
    z = np.concatenate([z, -z])

    # Confidence is 99 for |z| > 2.58, 95 for 1.96 < |z| <= 2.58,
    # 90 for 1.65 < |z| <= 1.96, else 0. 1.29 and 2.33 are p-value
    # thresholds in the original ladder and never change the output.
    expected = np.array([
        [0, 0, 90, 95, 95],     # just below each threshold
        [0, 0, 90, 95, 95],     # exactly on each threshold
        [0, 90, 95, 95, 99],    # just above each threshold
    ], dtype=np.int8)
    expected = np.concatenate([expected, -expected])

    out = _calc_hotspots_numpy(z)
    assert out.dtype == np.int8
    np.testing.assert_array_equal(out, expected)


def test_hotspots_classifier_float32_3737():
    # The production callers feed the classifier float32 z-scores
    # (_hotspots_numpy casts before classifying). The thresholds are
    # float64 literals, so a float32 input is promoted before comparison:
    # np.float32(1.65) sits just below 1.65 and np.float32(1.96) just
    # above it. Pin that the float32 result equals the classification of
    # the same values widened to float64, so a dtype-specific comparison
    # cannot creep in.
    from xrspatial.focal import _calc_hotspots_numpy

    thresholds = np.array([1.29, 1.65, 1.96, 2.33, 2.58])
    z = np.stack([np.nextafter(thresholds, -np.inf), thresholds,
                  np.nextafter(thresholds, np.inf)])
    z = np.concatenate([z, -z]).astype(np.float32)
    z = np.concatenate([np.nextafter(z, np.float32(-np.inf)), z,
                        np.nextafter(z, np.float32(np.inf))])

    out32 = _calc_hotspots_numpy(z)
    out64 = _calc_hotspots_numpy(z.astype(np.float64))
    assert out32.dtype == np.int8
    np.testing.assert_array_equal(out32, out64)
    # The float32 grid must still hit every band on both sides.
    assert set(np.unique(out32).tolist()) == {-99, -95, -90, 0, 90, 95, 99}


def test_hotspots_classifier_nonfinite_3737():
    # NaN compares False against every threshold and classifies to 0;
    # +/-inf land in the top band with the matching sign; signed zeros
    # are neither hot nor cold.
    from xrspatial.focal import _calc_hotspots_numpy

    z = np.array([[np.nan, np.inf, -np.inf, 0.0, -0.0]])
    out = _calc_hotspots_numpy(z)
    assert out.dtype == np.int8
    np.testing.assert_array_equal(out, [[0, 99, -99, 0, 0]])


@dask_array_available
@cuda_and_cupy_available
def test_hotspots_dask_cupy_matches_numpy():
    # The dask+cupy backend (_hotspots_dask_cupy) is registered in the
    # dispatch table but was previously exercised by no test. Verify it
    # runs and matches the numpy reference on the chunk interior.
    rng = np.random.default_rng(42)
    data = rng.standard_normal((12, 14)).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)

    # numpy reference
    numpy_agg = create_test_raster(data)
    numpy_hotspots = hotspots(numpy_agg, kernel)

    # dask + cupy case
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy', chunks=(6, 7))
    dask_cupy_hotspots = hotspots(dask_cupy_agg, kernel)
    general_output_checks(dask_cupy_agg, dask_cupy_hotspots, verify_attrs=False)
    assert dask_cupy_hotspots.attrs['unit'] == '%'

    # Compare interior (boundary='nan' causes edge differences between the
    # single-GPU bounds-clamping convolution and dask map_overlap NaN-padding)
    pad = kernel.shape[0] // 2
    np.testing.assert_array_equal(
        dask_cupy_hotspots.data[pad:-pad, pad:-pad].compute().get(),
        numpy_hotspots.data[pad:-pad, pad:-pad])


def _gistar_reference(data, kernel):
    """Brute-force Getis-Ord Gi* z-scores for the interior cells.

    Mirrors the closed-form definition cell-by-cell so the production
    code is checked against an independent computation, not against
    itself. NaN cells (and out-of-raster neighbors) are dropped from the
    neighborhood sums and the global statistics, matching hotspots().
    """
    data = data.astype(np.float64)
    valid = ~np.isnan(data)
    x = data[valid]
    n = x.size
    xbar = x.mean()
    s = x.std()  # population std (ddof=0)
    rows, cols = data.shape
    kr, kc = kernel.shape
    pr, pc = kr // 2, kc // 2
    z = np.zeros(data.shape, dtype=np.float64)
    # Only the interior is well defined for boundary='nan'; the outer ring
    # is blanked by the convolution and classified as 0.
    for i in range(pr, rows - pr):
        for j in range(pc, cols - pc):
            wx = w = w2 = 0.0
            for a in range(kr):
                for b in range(kc):
                    ii, jj = i + a - pr, j + b - pc
                    if 0 <= ii < rows and 0 <= jj < cols and valid[ii, jj]:
                        weight = kernel[a, b]
                        wx += weight * data[ii, jj]
                        w += weight
                        w2 += weight * weight
            var_term = (n * w2 - w * w) / (n - 1)
            if var_term <= 0:
                z[i, j] = 0.0
            else:
                z[i, j] = (wx - xbar * w) / (s * np.sqrt(var_term))
    return z


def test_hotspots_gistar_zscore_matches_reference():
    # Validate the Gi* z-scores themselves (not just the classification)
    # against a hand-rolled reference on a small known window.
    from xrspatial.focal import _gistar_convolutions_numpy, _gistar_zscore
    data = np.array([
        [1., 1., 1., 1., 1.],
        [1., 9., 9., 1., 1.],
        [1., 9., 9., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
    ], dtype=np.float64)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])

    d32 = data.astype(np.float32)
    n = int((~np.isnan(d32)).sum())
    gm = np.float32(np.nanmean(d32))
    gs = np.float32(np.nanstd(d32))
    ws, wsum, sq = _gistar_convolutions_numpy(d32, kernel, 'nan')
    z = _gistar_zscore(ws, wsum, sq, gm, gs, n)

    expected = _gistar_reference(data, kernel)
    pad = kernel.shape[0] // 2
    np.testing.assert_allclose(
        z[pad:-pad, pad:-pad], expected[pad:-pad, pad:-pad], rtol=1e-4)
    # The 9-valued cluster center is a significant hot spot.
    assert z[1, 1] == pytest.approx(2.9399, abs=1e-3)


def test_hotspots_gistar_weighted_kernel_matches_reference():
    # Non-binary kernel: weight sum and squared-weight sum diverge, which
    # exercises the W2 term that the old normalized-mean code ignored.
    rng = np.random.default_rng(7)
    data = (rng.standard_normal((8, 9)) * 5).astype(np.float64)
    kernel = np.array([[0., 2., 0.], [2., 3., 2.], [0., 2., 0.]])

    result = hotspots(create_test_raster(data), kernel)

    from xrspatial.focal import _gistar_convolutions_numpy, _gistar_zscore
    d32 = data.astype(np.float32)
    n = int((~np.isnan(d32)).sum())
    gm = np.float32(np.nanmean(d32))
    gs = np.float32(np.nanstd(d32))
    ws, wsum, sq = _gistar_convolutions_numpy(d32, kernel, 'nan')
    z = _gistar_zscore(ws, wsum, sq, gm, gs, n)

    expected_z = _gistar_reference(data, kernel)
    pad = kernel.shape[0] // 2
    np.testing.assert_allclose(
        z[pad:-pad, pad:-pad], expected_z[pad:-pad, pad:-pad], rtol=1e-3)
    # Output is the classified raster; confirm it is the int8 banding.
    assert result.data.dtype == np.int8
    assert set(np.unique(result.data)).issubset(
        {-99, -95, -90, 0, 90, 95, 99})


def test_hotspots_gistar_nan_excluded_from_stats():
    # NaN cells must not enter the neighborhood sums or the global mean/std.
    data = np.zeros((7, 7), dtype=np.float64)
    data[2:5, 2:5] = 50.0
    data[0, 0] = np.nan
    data[6, 6] = np.nan
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])

    result = hotspots(create_test_raster(data), kernel)
    assert not np.any(np.isnan(result.data))
    assert result.data[3, 3] > 0  # cluster center is a hot spot


def test_hotspots_single_valid_cell_raises():
    # n < 2 leaves the Gi* variance term undefined (divides by n - 1).
    data = np.full((4, 4), np.nan, dtype=np.float64)
    data[1, 1] = 5.0
    kernel = np.ones((3, 3))
    with pytest.raises(ValueError, match="at least 2 valid"):
        hotspots(create_test_raster(data), kernel)


@dask_array_available
def test_hotspots_gistar_dask_matches_numpy_full():
    # With clusters kept off the outer ring, numpy and dask agree on every
    # cell, so this is a full-array parity check of the Gi* dask path.
    data = np.zeros((12, 12), dtype=np.float64)
    data[3:6, 3:6] = 100.0
    data[7:10, 7:10] = -100.0
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])

    numpy_res = hotspots(create_test_raster(data), kernel)
    dask_res = hotspots(
        create_test_raster(data, backend='dask+numpy', chunks=(6, 6)), kernel)
    np.testing.assert_array_equal(numpy_res.data, dask_res.data.compute())


@dask_array_available
def test_convolution_2d_boundary_modes():
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)
    numpy_agg = create_test_raster(data)
    dask_agg = create_test_raster(data, backend='dask+numpy')
    from functools import partial
    func = partial(convolution_2d, kernel=kernel)
    assert_boundary_mode_correctness(numpy_agg, dask_agg, func)


def test_convolution_2d_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float32)
    agg = create_test_raster(data)
    kernel = np.ones((3, 3))
    with pytest.raises(ValueError, match="boundary must be one of"):
        convolution_2d(agg, kernel, boundary='invalid')


@dask_array_available
def test_mean_boundary_modes():
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64)
    numpy_agg = xr.DataArray(data, dims=['y', 'x'])
    dask_numpy_agg = xr.DataArray(da.from_array(data, chunks=(4, 5)), dims=['y', 'x'])
    assert_boundary_mode_correctness(numpy_agg, dask_numpy_agg, mean, nan_edges=False)


def test_mean_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float32)
    agg = xr.DataArray(data, dims=['y', 'x'])
    with pytest.raises(ValueError, match="boundary must be one of"):
        mean(agg, boundary='invalid')


@dask_array_available
def test_apply_boundary_modes():
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    numpy_agg = create_test_raster(data)
    dask_agg = create_test_raster(data, backend='dask+numpy')
    from functools import partial
    func = partial(apply, kernel=kernel, func=func_zero_cpu)
    assert_boundary_mode_correctness(numpy_agg, dask_agg, func, nan_edges=False)


def test_apply_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float32)
    agg = create_test_raster(data)
    kernel = np.ones((3, 3))
    with pytest.raises(ValueError, match="boundary must be one of"):
        apply(agg, kernel, func_zero_cpu, boundary='invalid')


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
def test_hotspots_boundary_modes(boundary):
    data = np.random.default_rng(42).standard_normal((10, 12)).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)
    numpy_agg = create_test_raster(data)
    dask_agg = create_test_raster(data, backend='dask+numpy')
    numpy_res = hotspots(numpy_agg, kernel, boundary=boundary).data
    dask_res = hotspots(dask_agg, kernel, boundary=boundary).data.compute()
    assert numpy_res.shape == data.shape
    if boundary == 'nan':
        # The outer ring diverges: the single-array convolution blanks it
        # while the dask map_overlap path computes a partial neighborhood.
        # The Gi* statistic matches on every interior cell.
        np.testing.assert_array_equal(numpy_res[1:-1, 1:-1],
                                      dask_res[1:-1, 1:-1])
    else:
        np.testing.assert_array_equal(numpy_res, dask_res)


def test_hotspots_boundary_invalid():
    data = np.random.default_rng(42).standard_normal((10, 12)).astype(np.float64)
    agg = create_test_raster(data)
    kernel = np.ones((3, 3))
    with pytest.raises(ValueError, match="boundary must be one of"):
        hotspots(agg, kernel, boundary='invalid')


# --- Parametrized numpy-vs-dask cross-backend boundary tests ---


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((5, 5), (2, 2)),
    ((7, 9), (3, 3)),
    ((10, 15), (10, 15)),
])
def test_convolution_2d_boundary_numpy_equals_dask(boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    np_result = convolution_2d(numpy_agg, kernel, boundary=boundary)
    da_result = convolution_2d(dask_agg, kernel, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((5, 5), (2, 2)),
    ((8, 10), (4, 5)),
    ((12, 12), (6, 4)),
])
def test_mean_boundary_numpy_equals_dask(boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float64)
    numpy_agg = xr.DataArray(data, dims=['y', 'x'])
    dask_agg = xr.DataArray(da.from_array(data, chunks=chunks), dims=['y', 'x'])
    np_result = mean(numpy_agg, boundary=boundary)
    da_result = mean(dask_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((5, 5), (2, 2)),
    ((7, 9), (3, 3)),
    ((10, 15), (5, 5)),
])
def test_apply_boundary_numpy_equals_dask(boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    np_result = apply(numpy_agg, kernel, func_zero_cpu, boundary=boundary)
    da_result = apply(dask_agg, kernel, func_zero_cpu, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((5, 6), (2, 3)),
    ((7, 9), (3, 3)),
    ((10, 12), (5, 6)),
])
def test_hotspots_boundary_numpy_equals_dask(boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.standard_normal(size).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    np_result = hotspots(numpy_agg, kernel, boundary=boundary).data
    da_result = hotspots(dask_agg, kernel, boundary=boundary).data.compute()
    if boundary == 'nan':
        # boundary='nan' diverges on the outer ring (single-array blanking
        # vs dask map_overlap partial neighborhoods); compare the interior.
        np.testing.assert_allclose(
            np_result[1:-1, 1:-1], da_result[1:-1, 1:-1],
            equal_nan=True, rtol=1e-5)
    else:
        np.testing.assert_allclose(
            np_result, da_result, equal_nan=True, rtol=1e-5)


# --- cupy honours boundary (issue-2730) ---


@cuda_and_cupy_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
def test_mean_boundary_numpy_equals_cupy_2730(boundary):
    """The cupy mean() must honour boundary, matching the numpy result.

    Regression for #2730: the cupy backend ignored boundary and always
    behaved as 'nan' (edge clamping).
    """
    import cupy
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64)
    numpy_agg = xr.DataArray(data, dims=['y', 'x'])
    cupy_agg = xr.DataArray(cupy.asarray(data), dims=['y', 'x'])
    np_result = mean(numpy_agg, boundary=boundary)
    cp_result = mean(cupy_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True, rtol=1e-4)


@cuda_and_cupy_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
def test_apply_boundary_numpy_equals_cupy_2730(boundary):
    """The cupy apply() must honour boundary, matching the numpy result."""
    import cupy
    from xrspatial.focal import _focal_mean_cuda
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)
    numpy_agg = create_test_raster(data, backend='numpy')
    cupy_agg = create_test_raster(data, backend='cupy')
    np_result = apply(numpy_agg, kernel, boundary=boundary)
    cp_result = apply(cupy_agg, kernel, _focal_mean_cuda, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True, rtol=1e-4)


@cuda_and_cupy_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
def test_focal_stats_boundary_numpy_equals_cupy_2730(boundary):
    """The cupy focal_stats() must honour boundary, matching numpy."""
    import cupy
    data = np.random.default_rng(42).random((8, 10)).astype(np.float64)
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    stats = ['mean', 'sum', 'min', 'max', 'range', 'std', 'var']
    numpy_agg = create_test_raster(data, backend='numpy')
    cupy_agg = create_test_raster(data, backend='cupy')
    np_result = focal_stats(numpy_agg, kernel, stats_funcs=stats, boundary=boundary)
    cp_result = focal_stats(cupy_agg, kernel, stats_funcs=stats, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True, rtol=1e-4)


@cuda_and_cupy_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_focal_stats_cupy_boundary_preserves_coords_2730(boundary):
    """Non-nan boundary on the cupy focal_stats path keeps coords/attrs."""
    import cupy
    data = np.random.default_rng(7).random((5, 6)).astype(np.float64)
    coords = {'y': np.arange(5) * 2.0, 'x': np.arange(6) * 3.0}
    cupy_agg = xr.DataArray(
        cupy.asarray(data), dims=['y', 'x'], coords=coords, attrs={'unit': 'm'})
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    result = focal_stats(cupy_agg, kernel, stats_funcs=['mean', 'max'],
                         boundary=boundary)
    assert result.shape == (2, 5, 6)
    np.testing.assert_array_equal(result['y'].data, coords['y'])
    np.testing.assert_array_equal(result['x'].data, coords['x'])
    assert list(result['stats'].data) == ['mean', 'max']
    assert result.attrs['unit'] == 'm'


@dask_array_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_convolution_2d_boundary_no_nan(boundary):
    """Non-nan modes produce no NaN output when source has no NaN."""
    rng = np.random.default_rng(99)
    data = rng.random((10, 12)).astype(np.float64)
    kernel = np.ones((3, 3), dtype=np.float64)
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=(5, 4))
    np_result = convolution_2d(numpy_agg, kernel, boundary=boundary)
    da_result = convolution_2d(dask_agg, kernel, boundary=boundary)
    assert not np.any(np.isnan(np_result.data))
    assert not np.any(np.isnan(da_result.data.compute()))
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


# --- convolve_2d float64 preservation (issue-1096) ---


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_convolve_2d_preserves_float64_1096(backend):
    """Float64 input should produce float64 output, not float32.

    Regression test for #1096: convolve_2d hardcoded .astype(float32)
    across all backends.
    """
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    # Values near 1e7 where float32 loses the 0.0x differences
    data = np.array([[1e7 + 0.01, 1e7 + 0.02, 1e7 + 0.03],
                     [1e7 + 0.04, 1e7 + 0.05, 1e7 + 0.06],
                     [1e7 + 0.07, 1e7 + 0.08, 1e7 + 0.09],
                     [1e7 + 0.10, 1e7 + 0.11, 1e7 + 0.12]], dtype=np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)

    agg = create_test_raster(data, backend=backend, chunks=(4, 3))
    result = convolve_2d(agg.data, kernel)

    if hasattr(result, 'compute'):
        result = result.compute()
    if hasattr(result, 'get'):
        result = result.get()
    result = np.asarray(result)

    # Output must be float64
    assert result.dtype == np.float64, f"got {result.dtype}"

    # Interior pixel (1,1): kernel cross hits [0.02, 0.04, 0.05, 0.06, 0.08]
    # Expected: sum = 5e7 + 0.25
    expected_center = 5e7 + 0.25
    assert abs(float(result[1, 1]) - expected_center) < 1e-8, (
        f"center={result[1, 1]}, expected={expected_center}"
    )


def test_convolve_2d_int_promotes_to_float32_1096():
    """Integer input should be promoted to float32 (not stay int)."""
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=np.int32)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)

    result = convolve_2d(data, kernel)
    assert np.issubdtype(result.dtype, np.floating), f"got {result.dtype}"
    # Interior pixel (1,1): cross sum = 2+4+5+6+8 = 25
    assert float(result[1, 1]) == 25.0


# --- 3D (multi-band) focal tests ---


@pytest.fixture
def rgb_data():
    rng = np.random.default_rng(123)
    return rng.random((3, 12, 14)).astype(np.float64)


def test_mean_3d_numpy(rgb_data):
    agg = xr.DataArray(rgb_data, dims=['band', 'y', 'x'])
    result = mean(agg)
    assert result.shape == (3, 12, 14)
    assert result.dims == ('band', 'y', 'x')
    for i in range(3):
        band_result = mean(agg.isel(band=i))
        np.testing.assert_allclose(result.isel(band=i).data, band_result.data)


@dask_array_available
def test_mean_3d_dask(rgb_data):
    dask_data = da.from_array(rgb_data, chunks=(1, 6, 7))
    agg = xr.DataArray(dask_data, dims=['band', 'y', 'x'])
    result = mean(agg)
    assert result.shape == (3, 12, 14)
    # compare against numpy per-band
    numpy_agg = xr.DataArray(rgb_data, dims=['band', 'y', 'x'])
    numpy_result = mean(numpy_agg)
    np.testing.assert_allclose(
        result.data.compute(), numpy_result.data, equal_nan=True, rtol=1e-5)


def test_apply_3d_numpy(rgb_data):
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    agg = xr.DataArray(rgb_data, dims=['band', 'y', 'x'])
    result = apply(agg, kernel)
    assert result.shape == (3, 12, 14)
    assert result.dims == ('band', 'y', 'x')
    for i in range(3):
        band_result = apply(agg.isel(band=i), kernel)
        np.testing.assert_allclose(result.isel(band=i).data, band_result.data)


@dask_array_available
def test_apply_3d_dask(rgb_data):
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    dask_data = da.from_array(rgb_data, chunks=(1, 6, 7))
    agg = xr.DataArray(dask_data, dims=['band', 'y', 'x'])
    result = apply(agg, kernel)
    assert result.shape == (3, 12, 14)
    numpy_agg = xr.DataArray(rgb_data, dims=['band', 'y', 'x'])
    numpy_result = apply(numpy_agg, kernel)
    np.testing.assert_allclose(
        result.data.compute(), numpy_result.data, equal_nan=True, rtol=1e-5)


def test_focal_stats_3d_numpy(rgb_data):
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    stats = ['mean', 'max']
    agg = xr.DataArray(rgb_data, dims=['band', 'y', 'x'])
    result = focal_stats(agg, kernel, stats_funcs=stats)
    # 3D input -> 4D output: (band, stats, y, x)
    assert result.shape == (3, 2, 12, 14)
    for i in range(3):
        band_result = focal_stats(agg.isel(band=i), kernel, stats_funcs=stats)
        np.testing.assert_allclose(
            result.isel(band=i).data, band_result.data, equal_nan=True)


@dask_array_available
def test_focal_stats_3d_dask(rgb_data):
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    stats = ['mean', 'max']
    dask_data = da.from_array(rgb_data, chunks=(1, 6, 7))
    agg = xr.DataArray(dask_data, dims=['band', 'y', 'x'])
    result = focal_stats(agg, kernel, stats_funcs=stats)
    assert result.shape == (3, 2, 12, 14)
    numpy_agg = xr.DataArray(rgb_data, dims=['band', 'y', 'x'])
    numpy_result = focal_stats(numpy_agg, kernel, stats_funcs=stats)
    np.testing.assert_allclose(
        result.data.compute(), numpy_result.data, equal_nan=True, rtol=1e-5)


def test_hotspots_3d_numpy():
    rng = np.random.default_rng(42)
    data_2d = rng.standard_normal((10, 12)).astype(np.float64)
    # stack 3 copies with different scales to avoid zero-std bands
    data_3d = np.stack([data_2d, data_2d * 2, data_2d * 0.5])
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)
    agg = xr.DataArray(data_3d, dims=['band', 'y', 'x'])
    result = hotspots(agg, kernel)
    assert result.shape == (3, 10, 12)
    assert result.dims == ('band', 'y', 'x')
    for i in range(3):
        band_result = hotspots(agg.isel(band=i), kernel)
        np.testing.assert_array_equal(result.isel(band=i).data, band_result.data)


@dask_array_available
def test_hotspots_3d_dask():
    rng = np.random.default_rng(42)
    data_2d = rng.standard_normal((10, 12)).astype(np.float64)
    data_3d = np.stack([data_2d, data_2d * 2, data_2d * 0.5])
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)
    # numpy reference
    numpy_agg = xr.DataArray(data_3d, dims=['band', 'y', 'x'])
    numpy_result = hotspots(numpy_agg, kernel)
    # dask
    dask_data = da.from_array(data_3d, chunks=(1, 5, 6))
    dask_agg = xr.DataArray(dask_data, dims=['band', 'y', 'x'])
    dask_result = hotspots(dask_agg, kernel)
    assert dask_result.shape == (3, 10, 12)
    # Compare the interior only: boundary='nan' diverges on the outer ring
    # between the single-array convolution (blanks the ring) and the dask
    # map_overlap path (NaN-pads the halo). The Gi* statistic itself is
    # identical on every interior cell.
    pad = kernel.shape[0] // 2
    np.testing.assert_array_equal(
        dask_result.data.compute()[:, pad:-pad, pad:-pad],
        numpy_result.data[:, pad:-pad, pad:-pad])


# --- result .name consistency across backends (metadata sweep) ----------
#
# Regression: the dask paths of focal_stats and hotspots constructed the
# output DataArray without an explicit name=, so xarray adopted the
# internal dask graph token (e.g. '_trim-<hash>') as the public .name.
# This made .name differ across the four backends (numpy/cupy gave one
# value, dask paths leaked a non-deterministic token).


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_focal_stats_name_consistent_across_backends(backend):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = (np.arange(16).reshape(4, 4) + 0.5).astype(np.float64)
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    agg = create_test_raster(data, backend=backend, chunks=(2, 2))
    result = focal_stats(agg, kernel, stats_funcs=['mean', 'max'])
    assert result.name == 'focal_stats'


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hotspots_name_consistent_across_backends(backend):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = (np.arange(16).reshape(4, 4) + 0.5).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)
    agg = create_test_raster(data, backend=backend, chunks=(2, 2))
    result = hotspots(agg, kernel)
    assert result.name == 'hotspots'


# --- output dtype consistency across backends (issue #3217) -------------
#
# Regression: mean() hardcoded float32 on the cupy and dask+cupy paths
# while the numpy and dask+numpy paths returned float64, so a float64
# raster silently lost precision on the GPU (fixed on main via the
# duplicate issue #3214: mean() now follows the _promote_float contract,
# preserving float dtypes and promoting ints to float32). The dask paths
# of mean, apply, and focal_stats also passed an untyped meta to
# map_overlap, so the lazy DataArray advertised float64 while .compute()
# returned the promoted float32 for float32/integer input.


def _computed_dtype(result):
    data = result.data
    if da is not None and isinstance(data, da.Array):
        data = data.compute()
    return data.dtype


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("in_dtype", [np.float64, np.float32, np.int32])
def test_mean_dtype_consistent_across_backends_3217(backend, in_dtype):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")

    data = (np.arange(16).reshape(4, 4) + 0.5).astype(in_dtype)
    agg = create_test_raster(data, backend=backend, chunks=(2, 2))
    result = mean(agg)
    # mean() promotes via _promote_float before dispatch (#3214); every
    # backend must keep that dtype, lazily and computed.
    expected = np.float64 if in_dtype == np.float64 else np.float32
    assert result.dtype == expected
    assert _computed_dtype(result) == expected


@cuda_and_cupy_available
def test_mean_gpu_matches_cpu_float64_3217():
    # The old float32 cast on the GPU paths produced ~1e-4 relative error
    # against the CPU result. In float64 the two backends agree exactly.
    import cupy
    data = np.random.default_rng(7).random((16, 16))
    cpu = mean(xr.DataArray(data))
    gpu = mean(xr.DataArray(cupy.asarray(data)))
    np.testing.assert_array_equal(cpu.data, gpu.data.get())


@pytest.mark.parametrize("backend", ['dask+numpy', 'dask+cupy'])
@pytest.mark.parametrize("in_dtype", [np.float64, np.float32, np.int32])
def test_apply_dask_advertised_dtype_matches_computed_3217(backend, in_dtype):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if da is None:
        pytest.skip("Requires Dask")

    data = (np.arange(16).reshape(4, 4) + 0.5).astype(in_dtype)
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    agg = create_test_raster(data, backend=backend, chunks=(2, 2))
    if 'cupy' in backend:
        from xrspatial.focal import _focal_mean_cuda
        result = apply(agg, kernel, func=_focal_mean_cuda)
    else:
        result = apply(agg, kernel)
    expected = np.float64 if in_dtype == np.float64 else np.float32
    assert result.dtype == expected
    assert _computed_dtype(result) == expected


@pytest.mark.parametrize("backend", ['dask+numpy', 'dask+cupy'])
@pytest.mark.parametrize("in_dtype", [np.float64, np.float32, np.int32])
def test_focal_stats_dask_advertised_dtype_matches_computed_3217(backend, in_dtype):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if da is None:
        pytest.skip("Requires Dask")

    data = (np.arange(16).reshape(4, 4) + 0.5).astype(in_dtype)
    kernel = custom_kernel(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    agg = create_test_raster(data, backend=backend, chunks=(2, 2))
    result = focal_stats(agg, kernel, stats_funcs=['mean', 'std'])
    expected = np.float64 if in_dtype == np.float64 else np.float32
    assert result.dtype == expected
    assert _computed_dtype(result) == expected


# ---------------------------------------------------------------------------
# API-consistency regressions (issue #2689)
# ---------------------------------------------------------------------------

_api_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)


def test_apply_raster_kwarg_deprecated():
    agg = xr.DataArray(data_random)
    with pytest.warns(DeprecationWarning, match="raster"):
        deprecated = apply(raster=agg, kernel=_api_kernel)
    current = apply(agg, _api_kernel)
    np.testing.assert_array_equal(deprecated.data, current.data)


def test_apply_raster_and_agg_conflict():
    agg = xr.DataArray(data_random)
    with pytest.raises(TypeError):
        apply(agg, _api_kernel, raster=agg)


def test_hotspots_raster_kwarg_deprecated():
    agg = xr.DataArray(data_random * 100)
    with pytest.warns(DeprecationWarning, match="raster"):
        deprecated = hotspots(raster=agg, kernel=_api_kernel)
    current = hotspots(agg, _api_kernel)
    np.testing.assert_array_equal(deprecated.data, current.data)


def test_hotspots_raster_and_agg_conflict():
    agg = xr.DataArray(data_random * 100)
    with pytest.raises(TypeError):
        hotspots(agg, _api_kernel, raster=agg)


def test_focal_stats_name():
    agg = xr.DataArray(data_random)
    result = focal_stats(agg, _api_kernel, stats_funcs=['mean', 'sum'])
    assert result.name == 'focal_stats'
    assert focal_stats(
        agg, _api_kernel, stats_funcs=['mean'], name='custom').name == 'custom'


def test_hotspots_name():
    agg = xr.DataArray(data_random * 100)
    assert hotspots(agg, _api_kernel).name == 'hotspots'
    assert hotspots(agg, _api_kernel, name='custom').name == 'custom'


def test_mean_excludes_default_not_shared():
    # Regression: mutable default replaced by a None sentinel. Calling with
    # an explicit excludes must not leak into the next default-args call.
    agg = xr.DataArray(data_random)
    mean(agg, excludes=[0.0])
    again = mean(agg)
    reference = mean(agg, excludes=[np.nan])
    np.testing.assert_array_equal(again.data, reference.data)


def test_focal_stats_default_stats_funcs():
    agg = xr.DataArray(data_random)
    result = focal_stats(agg, _api_kernel)
    assert result.sizes['stats'] == 8


def test_focal_stats_rejects_unknown_stats_func():
    # Regression for #2770: an unknown name used to fall through as a raw
    # KeyError. It must now raise a clear ValueError listing valid options.
    agg = xr.DataArray(data_random)
    with pytest.raises(ValueError, match=r"Invalid stats_funcs.*bogus"):
        focal_stats(agg, _api_kernel, stats_funcs=['bogus'])


def test_focal_stats_accepts_bare_string():
    # Regression for #2770: a bare string used to be iterated character by
    # character (e.g. 'mean' -> 'm','e','a','n') and fail. It must be treated
    # as a single stat name.
    agg = xr.DataArray(data_random)
    result = focal_stats(agg, _api_kernel, stats_funcs='mean')
    assert result.sizes['stats'] == 1
    assert list(result.coords['stats'].values) == ['mean']


def test_focal_stats_rejects_empty_stats_funcs():
    # Regression for #2770: an empty list used to reach xr.concat and fail with
    # an obscure error. It must raise a clear ValueError instead.
    agg = xr.DataArray(data_random)
    with pytest.raises(ValueError, match=r"stats_funcs must not be empty"):
        focal_stats(agg, _api_kernel, stats_funcs=[])


def test_focal_stats_valid_list_happy_path():
    agg = xr.DataArray(data_random)
    result = focal_stats(agg, _api_kernel, stats_funcs=['mean', 'sum'])
    assert result.sizes['stats'] == 2
    assert list(result.coords['stats'].values) == ['mean', 'sum']


@cuda_and_cupy_available
def test_focal_stats_name_gpu():
    import cupy
    agg = xr.DataArray(cupy.asarray(data_random))
    result = focal_stats(agg, _api_kernel, stats_funcs=['mean', 'sum'])
    assert result.name == 'focal_stats'


@cuda_and_cupy_available
def test_hotspots_name_gpu():
    import cupy
    agg = xr.DataArray(cupy.asarray(data_random * 100))
    assert hotspots(agg, _api_kernel).name == 'hotspots'
    with pytest.warns(DeprecationWarning, match="raster"):
        hotspots(raster=agg, kernel=_api_kernel)


# ---------------------------------------------------------------------------
# Coverage-gap regressions from the test-coverage sweep (issue #3220):
# Inf inputs, mean() NaN / excludes / passes behavior, 1x1 and strip rasters,
# empty rasters, and dask+cupy boundary modes.
# ---------------------------------------------------------------------------

ALL_BACKENDS = ['numpy', 'cupy', 'dask+numpy', 'dask+cupy']

_sweep_cross_kernel = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)


def _skip_unavailable_backend(backend):
    from xrspatial.tests.general_checks import has_cuda_and_cupy
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and da is None:
        pytest.skip("Requires Dask")


def _materialize(data):
    """Bring any backend's array back to a host numpy array."""
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'get'):
        data = data.get()
    return np.asarray(data)


def _apply_func_for(backend):
    if 'cupy' in backend:
        from xrspatial.focal import _focal_mean_cuda
        return _focal_mean_cuda
    from xrspatial.focal import _calc_mean
    return _calc_mean


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_mean_nan_input_3220(backend):
    # mean()'s default excludes=[np.nan] gives NaN cells specific semantics:
    # the NaN cell itself is left unchanged, while neighbors compute a
    # nanmean that skips it. No prior test fed mean() a NaN on any backend.
    _skip_unavailable_backend(backend)
    data = np.array([
        [1., 2., 3.],
        [4., np.nan, 6.],
        [7., 8., 9.],
    ])
    expected = np.array([
        [7 / 3, 16 / 5, 11 / 3],
        [22 / 5, np.nan, 28 / 5],
        [19 / 3, 34 / 5, 23 / 3],
    ])
    agg = create_test_raster(data, backend=backend, chunks=(2, 2))
    result = mean(agg)
    np.testing.assert_allclose(
        _materialize(result.data), expected, equal_nan=True, rtol=1e-4)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_mean_excludes_sentinel_3220(backend):
    # excludes had no behavioral test: a cell matching an exclude value must
    # be left unchanged, and (per the documented semantics) the sentinel
    # still participates in its neighbors' window means.
    _skip_unavailable_backend(backend)
    data = np.array([
        [1., 2., 3.],
        [4., -9999., 6.],
        [7., 8., 9.],
    ])
    agg = create_test_raster(data, backend=backend, chunks=(2, 2))
    result = _materialize(mean(agg, excludes=[-9999.]).data)
    assert result[1, 1] == -9999.0
    # (0, 0) window is [1, 2, 4, -9999] -> mean -2498
    np.testing.assert_allclose(result[0, 0], -2498.0, rtol=1e-6)

    numpy_agg = create_test_raster(data, backend='numpy')
    expected = mean(numpy_agg, excludes=[-9999.]).data
    np.testing.assert_allclose(result, expected, equal_nan=True, rtol=1e-4)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_mean_passes_3220(backend):
    # passes was only ever tested at its default of 1. passes=2 must equal
    # mean(mean(x)) and agree across backends.
    _skip_unavailable_backend(backend)
    data = np.zeros((5, 5), dtype=np.float64)
    data[2, 2] = 9.0

    numpy_agg = create_test_raster(data, backend='numpy')
    expected = mean(mean(numpy_agg)).data

    agg = create_test_raster(data, backend=backend, chunks=(3, 3))
    result = mean(agg, passes=2)
    np.testing.assert_allclose(
        _materialize(result.data), expected, equal_nan=True, rtol=1e-4)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_mean_inf_input_3220(backend):
    # No focal test exercised Inf at all. For mean(), Inf is not excluded
    # (only NaN is by default), so every window touching it goes to +Inf.
    _skip_unavailable_backend(backend)
    data = np.array([
        [1., 2., 3.],
        [4., np.inf, 6.],
        [7., 8., 9.],
    ])
    agg = create_test_raster(data, backend=backend, chunks=(2, 2))
    result = _materialize(mean(agg).data)
    # every 3x3 window on this raster contains the Inf center
    assert np.all(np.isinf(result)) and np.all(result > 0)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_focal_stats_inf_input_3220(backend):
    # Inf propagates through mean/sum/max/range, is ignored by min when a
    # smaller finite value exists, and poisons std/var to NaN (the window
    # mean is Inf, so deviations are NaN). All four backends must agree.
    # hotspots() with Inf is NOT pinned here: it silently classifies
    # everything to 0 today, which is tracked as a bug in issue #3219.
    _skip_unavailable_backend(backend)
    data = np.array([
        [1., 2., 3.],
        [4., np.inf, 6.],
        [7., 8., 9.],
    ])
    stats = ['mean', 'sum', 'min', 'max', 'range', 'std', 'var']
    agg = create_test_raster(data, backend=backend, chunks=(2, 2))
    result = focal_stats(agg, custom_kernel(_sweep_cross_kernel),
                         stats_funcs=stats)
    out = _materialize(result.data)

    by_stat = dict(zip(stats, out))
    # center cell: cross window is [2, 4, inf, 6, 8]
    assert np.isinf(by_stat['mean'][1, 1])
    assert np.isinf(by_stat['sum'][1, 1])
    assert by_stat['min'][1, 1] == 2.0
    assert np.isinf(by_stat['max'][1, 1])
    assert np.isinf(by_stat['range'][1, 1])
    assert np.isnan(by_stat['std'][1, 1])
    assert np.isnan(by_stat['var'][1, 1])
    # corner cell (0, 0): cross window is [1, 2, 4], fully finite
    np.testing.assert_allclose(by_stat['mean'][0, 0], 7 / 3, rtol=1e-4)
    assert np.isfinite(by_stat['std'][0, 0])

    numpy_agg = create_test_raster(data, backend='numpy')
    expected = focal_stats(numpy_agg, custom_kernel(_sweep_cross_kernel),
                           stats_funcs=stats).data
    np.testing.assert_allclose(out, expected, equal_nan=True, rtol=1e-4)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_focal_1x1_raster_3220(backend):
    # Degenerate single-pixel raster with a 3x3 kernel: the window clamps
    # to the one real cell on every backend.
    _skip_unavailable_backend(backend)
    data = np.array([[5.0]])
    agg = create_test_raster(data, backend=backend, chunks=(3, 3))

    mean_result = _materialize(mean(agg).data)
    np.testing.assert_allclose(mean_result, [[5.0]])

    apply_result = _materialize(
        apply(agg, _sweep_cross_kernel, _apply_func_for(backend)).data)
    np.testing.assert_allclose(apply_result, [[5.0]])

    fs = focal_stats(agg, custom_kernel(_sweep_cross_kernel),
                     stats_funcs=['mean', 'sum', 'std'])
    np.testing.assert_allclose(
        _materialize(fs.data).ravel(), [5.0, 5.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("shape,chunks", [((1, 6), (1, 3)), ((6, 1), (3, 1))])
def test_focal_strip_raster_3220(backend, shape, chunks):
    # Nx1 / 1xN strips exercise the kernel-window clamping on a raster
    # thinner than the kernel. The cross kernel reduces to a 1D window
    # along the strip, so both orientations share one expected result.
    _skip_unavailable_backend(backend)
    data = np.arange(6, dtype=np.float64).reshape(shape)
    expected = np.array([0.5, 1., 2., 3., 4., 4.5]).reshape(shape)
    agg = create_test_raster(data, backend=backend, chunks=chunks)

    mean_result = _materialize(mean(agg).data)
    np.testing.assert_allclose(mean_result, expected, rtol=1e-6)

    apply_result = _materialize(
        apply(agg, _sweep_cross_kernel, _apply_func_for(backend)).data)
    np.testing.assert_allclose(apply_result, expected, rtol=1e-6)

    fs = focal_stats(agg, custom_kernel(_sweep_cross_kernel),
                     stats_funcs=['mean'])
    np.testing.assert_allclose(
        _materialize(fs.data).reshape(shape), expected, rtol=1e-6)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_focal_empty_raster_3220(backend, shape):
    # A raster with a 0-length spatial axis has no cells to filter. numpy
    # returns an empty result with the input shape preserved; the cupy and
    # dask backends used to crash on it (a zero-sized kernel grid or an
    # overlap depth larger than the axis). After issue #3225 every backend
    # matches numpy: an empty result of the input shape and backend type.
    _skip_unavailable_backend(backend)
    data = np.empty(shape, dtype=np.float64)
    agg = create_test_raster(data, backend=backend, chunks=(3, 3))

    mean_result = mean(agg)
    assert mean_result.shape == shape
    assert isinstance(mean_result.data, type(agg.data))
    assert _materialize(mean_result.data).shape == shape

    apply_result = apply(agg, _sweep_cross_kernel, _apply_func_for(backend))
    assert apply_result.shape == shape
    assert isinstance(apply_result.data, type(agg.data))

    fs = focal_stats(agg, custom_kernel(_sweep_cross_kernel),
                     stats_funcs=['mean'])
    assert fs.shape == (1, *shape)
    assert isinstance(fs.data, type(agg.data))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_hotspots_empty_raster_3225(backend, shape):
    # An empty raster has no valid cells, so Gi* is undefined. numpy already
    # raises a clear "needs at least 2 valid cells" error; the cupy and dask
    # backends used to crash on a zero-sized reduction or overlap. Every
    # backend now raises the same error up front (issue #3225).
    _skip_unavailable_backend(backend)
    data = np.empty(shape, dtype=np.float64)
    agg = create_test_raster(data, backend=backend, chunks=(3, 3))
    with pytest.raises(ValueError, match="needs at least 2 valid"):
        hotspots(agg, np.ones((3, 3)))


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_focal_boundary_dask_cupy_3220(boundary):
    # The dask+cupy backend was never exercised with a non-default boundary
    # mode. Unlike boundary='nan' (where the single-array and map_overlap
    # paths legitimately differ on the outer ring), the padded modes must
    # match numpy on every cell.
    from xrspatial.focal import _focal_mean_cuda
    rng = np.random.default_rng(42)
    data = rng.random((8, 10)).astype(np.float64)
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy',
                                       chunks=(4, 5))

    np_mean = mean(numpy_agg, boundary=boundary).data
    dc_mean = _materialize(mean(dask_cupy_agg, boundary=boundary).data)
    np.testing.assert_allclose(dc_mean, np_mean, equal_nan=True, rtol=1e-4)

    np_apply = apply(numpy_agg, _sweep_cross_kernel, boundary=boundary).data
    dc_apply = _materialize(
        apply(dask_cupy_agg, _sweep_cross_kernel, _focal_mean_cuda,
              boundary=boundary).data)
    np.testing.assert_allclose(dc_apply, np_apply, equal_nan=True, rtol=1e-4)

    stats = ['mean', 'std']
    np_fs = focal_stats(numpy_agg, custom_kernel(_sweep_cross_kernel),
                        stats_funcs=stats, boundary=boundary).data
    dc_fs = _materialize(
        focal_stats(dask_cupy_agg, custom_kernel(_sweep_cross_kernel),
                    stats_funcs=stats, boundary=boundary).data)
    np.testing.assert_allclose(dc_fs, np_fs, equal_nan=True, rtol=1e-4)
