"""Nodata propagation on read.

Consolidates the GPU nodata-mask reader coverage:

* ``test_apply_nodata_mask_gpu_inplace_1934.py`` -- in-place mask
  semantics for ``_apply_nodata_mask_gpu`` (float and integer paths).
* ``test_apply_nodata_mask_gpu_with_presence_removed_2208.py`` -- the
  removed sibling helper stays gone after #2207 wired every GPU eager
  site through ``_finalize_eager_read``.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._backends import _gpu_helpers
from xrspatial.geotiff.tests.conftest import requires_gpu as _gpu_only


@_gpu_only
def test_apply_nodata_mask_gpu_float_masks_sentinel_to_nan():
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
def test_apply_nodata_mask_gpu_float_in_place_no_copy():
    """Float path mutates the input buffer in place."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(
        np.array([[1.0, 2.0], [-9999.0, 4.0]], dtype=np.float32)
    )
    input_ptr = arr_gpu.data.ptr
    out = _apply_nodata_mask_gpu(arr_gpu, -9999.0)
    assert out.data.ptr == input_ptr


@_gpu_only
def test_apply_nodata_mask_gpu_float_alloc_count_unchanged():
    """Float path does not pull a fresh chunk-sized buffer from the pool."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    isolated_pool = cupy.cuda.MemoryPool()
    prev_allocator = cupy.cuda.get_allocator()
    cupy.cuda.set_allocator(isolated_pool.malloc)
    try:
        arr_gpu = cupy.full((512, 512), -9999.0, dtype=cupy.float32)
        arr_gpu[0, 0] = 1.0  # plant a non-sentinel pixel

        cupy.cuda.Stream.null.synchronize()
        isolated_pool.free_all_blocks()
        total_before = isolated_pool.total_bytes()

        out = _apply_nodata_mask_gpu(arr_gpu, -9999.0)
        cupy.cuda.Stream.null.synchronize()
        isolated_pool.free_all_blocks()
        total_after = isolated_pool.total_bytes()

        array_bytes = arr_gpu.nbytes
        growth = total_after - total_before
        assert growth < array_bytes, (
            f"unexpected allocation growth {growth} bytes >= "
            f"array_bytes {array_bytes}; in-place mutation regressed"
        )
        assert out.data.ptr == arr_gpu.data.ptr
    finally:
        cupy.cuda.set_allocator(prev_allocator)
        isolated_pool.free_all_blocks()


@_gpu_only
def test_apply_nodata_mask_gpu_int_promotes_and_masks():
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
def test_apply_nodata_mask_gpu_int_no_extra_buffer_after_astype():
    """Integer path: only the ``astype(float64)`` buffer is allocated."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    isolated_pool = cupy.cuda.MemoryPool()
    prev_allocator = cupy.cuda.get_allocator()
    cupy.cuda.set_allocator(isolated_pool.malloc)
    try:
        arr_gpu = cupy.full((512, 512), 3, dtype=cupy.uint16)
        arr_gpu[0, 0] = 1  # ensure non-sentinel pixel exists

        cupy.cuda.Stream.null.synchronize()
        isolated_pool.free_all_blocks()
        total_before = isolated_pool.total_bytes()

        out = _apply_nodata_mask_gpu(arr_gpu, 3)
        cupy.cuda.Stream.null.synchronize()
        isolated_pool.free_all_blocks()
        total_after = isolated_pool.total_bytes()

        float64_bytes = out.nbytes
        growth = total_after - total_before
        assert growth < 2 * float64_bytes, (
            f"unexpected allocation growth {growth} bytes >= "
            f"2 * float64_bytes {2 * float64_bytes}; pre-fix double-alloc"
        )
    finally:
        cupy.cuda.set_allocator(prev_allocator)
        isolated_pool.free_all_blocks()


@_gpu_only
def test_apply_nodata_mask_gpu_float_nan_sentinel_noop():
    """NaN nodata on a float array stays a no-op."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    input_ptr = arr_gpu.data.ptr
    out = _apply_nodata_mask_gpu(arr_gpu, float('nan'))
    assert out.data.ptr == input_ptr
    np.testing.assert_array_equal(out.get(), [[1.0, 2.0], [3.0, 4.0]])


@_gpu_only
def test_apply_nodata_mask_gpu_none_nodata_passthrough():
    """``nodata is None`` returns the input array untouched."""
    import cupy

    from xrspatial.geotiff import _apply_nodata_mask_gpu

    arr_gpu = cupy.asarray(np.array([[1, 2], [3, 4]], dtype=np.int32))
    input_ptr = arr_gpu.data.ptr
    out = _apply_nodata_mask_gpu(arr_gpu, None)
    assert out.data.ptr == input_ptr
    assert out.dtype == cupy.int32


# ---------------------------------------------------------------------------
# Helper removal pin (#2208)
# ---------------------------------------------------------------------------


def test_apply_nodata_mask_gpu_with_presence_not_importable():
    """The dead sibling helper stays removed after #2207."""
    # Covers both module-attribute absence and the import-time surface.
    with pytest.raises(ImportError):
        from xrspatial.geotiff._backends._gpu_helpers import \
            _apply_nodata_mask_gpu_with_presence  # noqa: F401


def test_apply_nodata_mask_gpu_still_present():
    """``_apply_nodata_mask_gpu`` is still on the chunked GPU dask path."""
    assert hasattr(_gpu_helpers, '_apply_nodata_mask_gpu')
    assert callable(_gpu_helpers._apply_nodata_mask_gpu)
