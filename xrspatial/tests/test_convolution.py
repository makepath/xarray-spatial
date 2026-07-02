import numpy as np
import pytest
import xarray as xr

from xrspatial.convolution import (
    circle_kernel, convolution_2d, convolve_2d, custom_kernel,
)


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


DATA = np.arange(25, dtype=np.float64).reshape(5, 5)


@pytest.mark.parametrize("bad_kernel", [
    None,
    np.ones(3, dtype=np.float64),            # 1D
    np.ones((3, 3, 3), dtype=np.float64),    # 3D
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],       # python list, not an array
])
def test_convolve_2d_rejects_bad_kernel(bad_kernel):
    # Bad kernels used to crash deep in numba with a cryptic TypingError.
    # convolve_2d must reject them up front with a clear message that names
    # `kernel`.
    with pytest.raises(ValueError, match="kernel"):
        convolve_2d(DATA, bad_kernel)


@pytest.mark.parametrize("even_kernel", [
    np.ones((2, 2), dtype=np.float64),
    np.ones((4, 4), dtype=np.float64),
    np.ones((2, 3), dtype=np.float64),
])
def test_convolve_2d_rejects_even_kernel(even_kernel):
    # An even side length has no well-defined center; convolve_2d used to
    # silently produce an off-center result. custom_kernel already rejects
    # even kernels, so convolve_2d must too.
    with pytest.raises(ValueError, match="odd"):
        convolve_2d(DATA, even_kernel)


def test_convolution_2d_rejects_non_dataarray():
    # Passing a plain numpy array used to fail with an inscrutable
    # "'memoryview' object has no attribute 'astype'"; validate up front.
    with pytest.raises(TypeError, match="DataArray"):
        convolution_2d(DATA, KERNEL)


def test_convolution_2d_accepts_dataarray():
    # Positive path unchanged.
    agg = xr.DataArray(DATA, dims=['y', 'x'])
    out = convolution_2d(agg, KERNEL)
    assert isinstance(out, xr.DataArray)
    assert out.shape == agg.shape
