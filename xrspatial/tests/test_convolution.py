import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.convolution import circle_kernel, convolve_2d, custom_kernel
from xrspatial.tests.general_checks import cuda_and_cupy_available

KERNEL = circle_kernel(1, 1, 1)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_convolve_2d_dask_dtype_matches_numpy(dtype):
    # The dask backend used to advertise float64 via ``meta=np.array(())``
    # even when the eager numpy backend promoted an int32/float32 raster to
    # float32, so ``result.dtype`` disagreed across backends and the lazy
    # dtype did not match the computed chunks. The declared dask dtype must
    # equal both the eager dtype and the actually-computed chunk dtype.
    data = np.arange(64, dtype=dtype).reshape(8, 8)
    eager = convolve_2d(data, KERNEL)
    lazy = convolve_2d(da.from_array(data, chunks=(4, 4)), KERNEL)
    assert lazy.dtype == eager.dtype
    assert lazy.compute().dtype == eager.dtype


@cuda_and_cupy_available
@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_convolve_2d_cupy_dtype_matches_numpy(dtype):
    # The dask+cupy backend carried the same untyped ``meta=cupy.array(())``
    # float64 mismatch the numpy path had, so the GPU dtypes need the same
    # check: cupy and dask+cupy must declare (and compute) the promoted dtype
    # the eager numpy backend returns.
    import cupy

    data = np.arange(64, dtype=dtype).reshape(8, 8)
    eager = convolve_2d(data, KERNEL)

    cupy_out = convolve_2d(cupy.asarray(data), KERNEL)
    assert cupy_out.dtype == eager.dtype

    dask_cupy_out = convolve_2d(
        da.from_array(cupy.asarray(data), chunks=(4, 4)), KERNEL)
    assert dask_cupy_out.dtype == eager.dtype
    assert dask_cupy_out.compute().dtype == eager.dtype


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


def test_convolve_2d_uses_correlation_convention():
    # Golden values verified against scipy.ndimage 1.16.1: convolve_2d
    # applies the kernel by cross-correlation (kernel NOT flipped), so it
    # matches scipy.ndimage.correlate, not scipy.ndimage.convolve. An
    # asymmetric kernel distinguishes the two. This pins the documented
    # convention so a rename/refactor cannot silently flip the kernel.
    data = np.arange(24, dtype=np.float64).reshape(4, 6)
    kernel = custom_kernel(np.array([[1., 0., 0.],
                                     [1., 1., 0.],
                                     [1., 0., 0.]]))
    out = convolve_2d(data, kernel)
    # scipy.ndimage.correlate(data, kernel)[1:-1, 1:-1]
    expected_correlate = np.array([[25., 29., 33., 37.],
                                   [49., 53., 57., 61.]])
    np.testing.assert_array_equal(out[1:-1, 1:-1], expected_correlate)
    # A true convolution (scipy.ndimage.convolve, kernel flipped) would
    # instead give these values; convolve_2d must NOT match them.
    expected_convolve = np.array([[31., 35., 39., 43.],
                                  [55., 59., 63., 67.]])
    assert not np.array_equal(out[1:-1, 1:-1], expected_convolve)
