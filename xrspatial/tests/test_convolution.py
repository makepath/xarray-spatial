from functools import partial

import numpy as np
import pytest
import xarray as xr

from xrspatial.convolution import (annulus_kernel, calc_cellsize, circle_kernel, convolution_2d,
                                   convolve_2d, custom_kernel)
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_numpy_equals_cupy, assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy, create_test_raster,
                                            cuda_and_cupy_available, dask_array_available)
from xrspatial.utils import VALID_BOUNDARY_MODES

KERNEL = circle_kernel(1, 1, 1)


def test_convolve_2d_rejects_boolean_dtype():
    # Boolean DataArrays used to crash deep inside numba with a
    # cryptic TypingError; _validate_raster should reject up front.
    agg = xr.DataArray(np.ones((10, 10), dtype=bool))
    with pytest.raises(ValueError, match="numeric dtype"):
        convolve_2d(agg.data, KERNEL)


def test_convolve_2d_rejects_string_dtype():
    agg = xr.DataArray(np.array([['a', 'b'], ['c', 'd']]))
    with pytest.raises(ValueError, match="numeric dtype"):
        convolve_2d(agg.data, KERNEL)


def test_convolve_2d_rejects_wrong_ndim():
    # 1D input should be rejected before reaching the kernel
    with pytest.raises(ValueError, match="must be 2D"):
        convolve_2d(np.zeros(10, dtype=np.float64), KERNEL)


@pytest.mark.parametrize("bad_kernel", [
    np.ones(3, dtype=np.float32),            # 1D
    np.ones((3, 3, 3), dtype=np.float32),    # 3D
])
def test_custom_kernel_rejects_non_2d(bad_kernel):
    # Regression for #2842: custom_kernel must raise a clear error for a
    # non-2D kernel instead of leaking "not enough values to unpack" from
    # the `rows, cols = kernel.shape` line.
    with pytest.raises(ValueError, match="not a 2D array"):
        custom_kernel(bad_kernel)


def test_convolve_2d_accepts_float64():
    # Positive path: a float64 raster still works end-to-end.
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    out = convolve_2d(data, KERNEL)
    assert out.shape == data.shape
    # Centre cell is finite; edges are NaN by default boundary mode.
    assert np.isfinite(out[2, 2])
    assert np.isnan(out[0, 0])


# ---------------------------------------------------------------------------
# Backend parity (Cat 1) -- convolve_2d dispatches to numpy / cupy /
# dask+numpy / dask+cupy via ArrayTypeFunctionMapping, but only the numpy
# path was exercised. Assert the other three registered backends match numpy.
# ---------------------------------------------------------------------------

def _sample_raster_data():
    return np.arange(7 * 8, dtype=np.float64).reshape(7, 8)


@dask_array_available
def test_convolution_2d_numpy_equals_dask_numpy():
    data = _sample_raster_data()
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy')
    assert_numpy_equals_dask_numpy(
        numpy_agg, dask_agg, partial(convolution_2d, kernel=KERNEL))


@cuda_and_cupy_available
def test_convolution_2d_numpy_equals_cupy():
    data = _sample_raster_data()
    numpy_agg = create_test_raster(data, backend='numpy')
    cupy_agg = create_test_raster(data, backend='cupy')
    assert_numpy_equals_cupy(
        numpy_agg, cupy_agg, partial(convolution_2d, kernel=KERNEL))


@cuda_and_cupy_available
def test_convolution_2d_numpy_equals_dask_cupy():
    data = _sample_raster_data()
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(
        numpy_agg, dask_cupy_agg, partial(convolution_2d, kernel=KERNEL))


# ---------------------------------------------------------------------------
# Boundary-mode parameter coverage (Cat 4) -- only the default 'nan' mode was
# exercised. 'nearest', 'reflect', and 'wrap' are documented public modes.
# ---------------------------------------------------------------------------

@dask_array_available
def test_convolution_2d_boundary_modes_numpy_and_dask():
    data = _sample_raster_data()
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy')
    assert_boundary_mode_correctness(
        numpy_agg, dask_agg, partial(convolution_2d, kernel=KERNEL), depth=1)


@cuda_and_cupy_available
def test_convolution_2d_boundary_modes_gpu_match_numpy():
    data = _sample_raster_data()
    numpy_agg = create_test_raster(data, backend='numpy')
    cupy_agg = create_test_raster(data, backend='cupy')
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy')
    for mode in VALID_BOUNDARY_MODES:
        expected = convolution_2d(numpy_agg, KERNEL, boundary=mode).data
        got_cupy = convolution_2d(cupy_agg, KERNEL, boundary=mode).data.get()
        got_dask_cupy = convolution_2d(
            dask_cupy_agg, KERNEL, boundary=mode).data.compute().get()
        np.testing.assert_allclose(expected, got_cupy, equal_nan=True, rtol=1e-6)
        np.testing.assert_allclose(
            expected, got_dask_cupy, equal_nan=True, rtol=1e-6)


def test_convolve_2d_rejects_invalid_boundary():
    data = _sample_raster_data()
    with pytest.raises(ValueError, match="boundary must be one of"):
        convolve_2d(data, KERNEL, boundary='bogus')


# ---------------------------------------------------------------------------
# Metadata preservation (Cat 5) -- convolution_2d exists to wrap the raw-array
# convolve_2d and re-attach coords/dims/attrs/name. Nothing asserted that.
# ---------------------------------------------------------------------------

def test_convolution_2d_preserves_metadata():
    data = _sample_raster_data()
    agg = create_test_raster(data, backend='numpy')
    out = convolution_2d(agg, KERNEL)
    assert out.name == 'convolution_2d'
    assert out.dims == agg.dims
    assert out.attrs == agg.attrs
    for coord in agg.coords:
        np.testing.assert_allclose(out[coord].data, agg[coord].data)


def test_convolution_2d_custom_name():
    data = _sample_raster_data()
    agg = create_test_raster(data, backend='numpy')
    out = convolution_2d(agg, KERNEL, name='smoothed')
    assert out.name == 'smoothed'


# ---------------------------------------------------------------------------
# NaN edge case (Cat 2) -- a NaN inside the raster must propagate through the
# kernel window, not be silently ignored.
# ---------------------------------------------------------------------------

def test_convolve_2d_nan_interior_propagates():
    data = np.arange(7 * 7, dtype=np.float64).reshape(7, 7)
    data[3, 3] = np.nan
    out = convolve_2d(data, KERNEL)
    # Every inner cell whose 3x3 circle window overlaps (3, 3) is NaN.
    assert np.isnan(out[3, 3])
    assert np.isnan(out[2, 3])
    assert np.isnan(out[3, 2])
    # A distant inner cell is unaffected and finite.
    assert np.isfinite(out[1, 1])


# ---------------------------------------------------------------------------
# Geometric degeneracies (Cat 3) -- a 3x3 kernel on a raster with a dimension
# smaller than the kernel leaves no valid inner cells; output is all-NaN and
# does not crash.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (8, 1)])
def test_convolve_2d_degenerate_shapes_all_nan(shape):
    data = np.ones(shape, dtype=np.float64)
    out = convolve_2d(data, KERNEL)
    assert out.shape == shape
    assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------
# Kernel generators and calc_cellsize -- previously untested public functions.
# ---------------------------------------------------------------------------

def test_calc_cellsize_from_res_attr():
    data = np.ones((10, 10))
    raster = xr.DataArray(data, attrs={'res': (0.5, 0.5)})
    assert calc_cellsize(raster) == (0.5, 0.5)


def test_calc_cellsize_unit_conversion_km():
    # 'unit' attr in km converts cellsize to meters; y is returned abs().
    data = np.ones((3, 4))
    raster = xr.DataArray(data, dims=['y', 'x'], attrs={'unit': 'km'})
    raster['y'] = np.linspace(3, 1, 3)
    raster['x'] = np.linspace(1, 4, 4)
    cx, cy = calc_cellsize(raster)
    assert cx == pytest.approx(1000.0)
    assert cy == pytest.approx(1000.0)


def test_annulus_kernel_is_ring():
    kernel = annulus_kernel(1, 1, 3, 1)
    # Ring shape: centre cell (inner radius) is hollowed out.
    assert kernel.shape == (7, 7)
    assert kernel[3, 3] == 0
    assert set(np.unique(kernel)).issubset({0.0, 1.0})


def test_custom_kernel_accepts_valid_odd_2d():
    kernel = np.ones((3, 3), dtype=np.float64)
    assert custom_kernel(kernel) is kernel


def test_custom_kernel_rejects_non_ndarray():
    with pytest.raises(ValueError, match="not a Numpy array"):
        custom_kernel([[1, 0, 1], [0, 1, 0], [1, 0, 1]])


def test_custom_kernel_rejects_even_dims():
    with pytest.raises(ValueError, match="improper dimensions"):
        custom_kernel(np.ones((2, 2), dtype=np.float64))


@pytest.mark.parametrize("bad_radius, match", [
    (-3, "positive"),
    ('3 lightyears', "Distance unit should be one of"),
])
def test_circle_kernel_rejects_bad_radius(bad_radius, match):
    with pytest.raises(ValueError, match=match):
        circle_kernel(1, 1, bad_radius)
