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


def test_apply_small_kernel_not_rejected_1284():
    # The guard must not fire for realistic kernel + raster combos.
    raster = xr.DataArray(np.ones((50, 50), dtype=np.float32))
    kernel = circle_kernel(1, 1, 3)
    out = apply(raster, kernel)
    assert out.shape == (50, 50)


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


@pytest.fixture
def data_hotspots():
    data = np.asarray([
        [np.nan, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 10000., 10000., 10000., 0., 0., 0., 0., 0., 0.],
        [0., 10000., 10000., 10000., 0., 0., 0., 0., 0., 0.],
        [0., 10000., 10000., 10000., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., np.nan, 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., np.nan, 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., np.nan, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., -10000., -10000., -10000.],
        [0., 0., 0., 0., 0., 0., 0., -10000., -10000., -10000.],
        [0., 0., 0., 0., 0., 0., 0., -10000., -10000., -10000.]
    ])
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    expected_result = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 90, 0, 0, 0, 0, 0, 0, 0],
        [0, 90, 95, 90, 0, 0, 0, 0, 0, 0],
        [0, 0, 90, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -90, 0],
        [0, 0, 0, 0, 0, 0, 0, -90, -95, 0],
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
def test_hotspots_boundary_modes():
    data = np.random.default_rng(42).standard_normal((10, 12)).astype(np.float64)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)
    numpy_agg = create_test_raster(data)
    dask_agg = create_test_raster(data, backend='dask+numpy')
    from functools import partial
    func = partial(hotspots, kernel=kernel)
    assert_boundary_mode_correctness(numpy_agg, dask_agg, func, nan_edges=False)


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
    np_result = hotspots(numpy_agg, kernel, boundary=boundary)
    da_result = hotspots(dask_agg, kernel, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


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
    np.testing.assert_array_equal(
        dask_result.data.compute(), numpy_result.data)
