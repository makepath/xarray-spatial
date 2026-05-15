"""Regression tests for issue #1934.

``_apply_nodata_mask_gpu`` used to replace the sentinel pixels via
``cupy.where(arr_gpu == sentinel, nan, arr_gpu)``, which allocates a fresh
output buffer the same shape as the input. Every call site passes a
freshly decoded GPU buffer that no caller-visible state aliases, so the
fix writes NaN into the existing buffer with ``cupy.putmask`` and drops
one chunk-sized device allocation per call.

Two guards here:

1. Correctness -- float and integer paths match the pre-fix behaviour on
   a representative input (sentinel masked to NaN, non-sentinel pixels
   preserved, integer dtype promoted to float64).
2. In-place mutation -- on the float path the output array shares the
   same device pointer as the input, confirming no fresh allocation. The
   integer path still allocates via ``astype(float64)``; the test checks
   the post-astype buffer is then mutated in place rather than copied
   again by ``cupy.where``.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


@_gpu_only
def test_apply_nodata_mask_gpu_float_masks_sentinel_to_nan_1934():
    """Float path masks the sentinel to NaN and leaves other pixels alone."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(
        np.array([[1.0, 2.0], [-9999.0, 4.0]], dtype=np.float32)
    )
    out = _apply_nodata_mask_gpu(arr_gpu, -9999.0)
    assert out.dtype == cupy.float32
    host = out.get()
    assert np.isnan(host[1, 0])
    assert host[0, 0] == 1.0
    assert host[0, 1] == 2.0
    assert host[1, 1] == 4.0


@_gpu_only
def test_apply_nodata_mask_gpu_float_in_place_no_copy_1934():
    """Float path mutates the input buffer in place.

    Before the fix, ``cupy.where`` returned a fresh array, so ``out`` had
    a different device pointer than ``arr_gpu``. After the fix the input
    buffer is reused and the pointers match.
    """
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(
        np.array([[1.0, 2.0], [-9999.0, 4.0]], dtype=np.float32)
    )
    input_ptr = arr_gpu.data.ptr
    out = _apply_nodata_mask_gpu(arr_gpu, -9999.0)
    assert out.data.ptr == input_ptr


@_gpu_only
def test_apply_nodata_mask_gpu_float_alloc_count_unchanged_1934():
    """Float path does not pull a fresh chunk-sized buffer from the pool.

    The default memory pool exposes ``n_free_blocks`` / ``used_bytes`` and
    ``total_bytes``. The pre-fix ``cupy.where`` path allocated one extra
    chunk-sized buffer; the in-place fix should not grow ``used_bytes``
    once the input is already on device.
    """
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    # Large enough that an extra allocation would be visible.
    arr_gpu = cupy.full((512, 512), -9999.0, dtype=cupy.float32)
    # Plant a non-sentinel pixel so the mask is not empty.
    arr_gpu[0, 0] = 1.0

    pool = cupy.get_default_memory_pool()
    cupy.cuda.Stream.null.synchronize()
    used_before = pool.used_bytes()

    out = _apply_nodata_mask_gpu(arr_gpu, -9999.0)
    cupy.cuda.Stream.null.synchronize()
    used_after = pool.used_bytes()

    # The mask is a transient bool array (1/4 the byte count of float32),
    # so used_bytes can rise by the mask size but must not rise by the
    # array's full byte count. Pre-fix would add at least one float32
    # buffer of the same shape (512*512*4 = 1 MiB).
    array_bytes = arr_gpu.nbytes
    growth = used_after - used_before
    assert growth < array_bytes, (
        f"unexpected allocation growth {growth} bytes >= "
        f"array_bytes {array_bytes}; in-place mutation regressed"
    )
    # And the returned buffer is the same one we passed in.
    assert out.data.ptr == arr_gpu.data.ptr


@_gpu_only
def test_apply_nodata_mask_gpu_int_promotes_and_masks_1934():
    """Integer path still promotes to float64 and masks the sentinel."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(
        np.array([[1, 2], [3, 4]], dtype=np.uint16)
    )
    out = _apply_nodata_mask_gpu(arr_gpu, 3)
    assert out.dtype == cupy.float64
    host = out.get()
    assert np.isnan(host[1, 0])
    assert host[0, 0] == 1.0
    assert host[0, 1] == 2.0
    assert host[1, 1] == 4.0


@_gpu_only
def test_apply_nodata_mask_gpu_int_no_extra_buffer_after_astype_1934():
    """Integer path: only the ``astype(float64)`` buffer is allocated.

    Before the fix the trailing ``cupy.where`` allocated a second
    chunk-sized float64 buffer. After the fix the ``astype`` buffer is
    mutated in place.
    """
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.full((512, 512), 3, dtype=cupy.uint16)
    arr_gpu[0, 0] = 1  # ensure non-sentinel pixel exists

    pool = cupy.get_default_memory_pool()
    cupy.cuda.Stream.null.synchronize()
    used_before = pool.used_bytes()

    out = _apply_nodata_mask_gpu(arr_gpu, 3)
    cupy.cuda.Stream.null.synchronize()
    used_after = pool.used_bytes()

    # Required: one float64 buffer (512*512*8 = 2 MiB) from astype.
    # Pre-fix would have allocated a second float64 buffer for cupy.where
    # (another 2 MiB) on top of that.
    float64_bytes = out.nbytes
    growth = used_after - used_before
    # Allow some slack for the bool mask + .any() scalar (well under
    # one float64 buffer of slack).
    assert growth < 2 * float64_bytes, (
        f"unexpected allocation growth {growth} bytes >= "
        f"2 * float64_bytes {2 * float64_bytes}; pre-fix double-alloc"
    )


@_gpu_only
def test_apply_nodata_mask_gpu_float_nan_sentinel_noop_1934():
    """NaN nodata on a float array stays a no-op."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    input_ptr = arr_gpu.data.ptr
    out = _apply_nodata_mask_gpu(arr_gpu, float('nan'))
    # Same buffer back, untouched.
    assert out.data.ptr == input_ptr
    np.testing.assert_array_equal(out.get(), [[1.0, 2.0], [3.0, 4.0]])


@_gpu_only
def test_apply_nodata_mask_gpu_none_nodata_passthrough_1934():
    """``nodata is None`` returns the input array untouched."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.int32))
    input_ptr = arr_gpu.data.ptr
    out = _apply_nodata_mask_gpu(arr_gpu, None)
    assert out.data.ptr == input_ptr
    assert out.dtype == cupy.int32
