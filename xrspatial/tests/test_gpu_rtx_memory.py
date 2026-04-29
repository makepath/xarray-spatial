"""Tests for the gpu_rtx memory guard (Issue #1308).

The guard sits at the entry of ``hillshade_rtx`` and ``viewshed_gpu`` and
raises ``MemoryError`` before any cupy device allocation when the input
raster would exceed 50% of free VRAM at the 120 B/pixel worst-case budget.

Tests for the helper itself don't need RTX (they patch the free-memory
sentinel).  End-to-end tests that hit the public ``hillshade()`` and
``viewshed()`` entry points are gated on ``has_rtx()``.
"""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from xrspatial.gpu_rtx._memory import (
    _GPU_BYTES_PER_PIXEL,
    _available_gpu_memory_bytes,
    _check_gpu_memory,
)

from ..gpu_rtx import has_rtx


# ----------------------------------------------------------------------
# Unit tests for the helper itself (no RTX needed)
# ----------------------------------------------------------------------

def test_bytes_per_pixel_is_worst_case():
    """The shared budget should cover the worst entry point (viewshed_gpu).

    viewshed_gpu allocations:
        mesh:      verts (12) + triangles (24)            = 36 B/px
        kernels:   d_rays (32) + d_hits (16)
                 + d_visgrid (4) + d_vsrays (32)          = 84 B/px
        total                                             = 120 B/px
    """
    assert _GPU_BYTES_PER_PIXEL >= 120


def test_check_gpu_memory_raises_when_required_exceeds_half_free():
    """Below half-free passes; above half-free raises."""
    # 1_000_000 bytes free, 4x4 raster needs 4*4*120 = 1920 B.
    # half-free = 500_000 -> 1920 < 500_000 -> no raise.
    with patch(
        "xrspatial.gpu_rtx._memory._available_gpu_memory_bytes",
        return_value=1_000_000,
    ):
        _check_gpu_memory("hillshade_rtx", 4, 4)

    # 100 bytes free, 4x4 raster needs 1920 B.  half-free = 50 < 1920 -> raise.
    with patch(
        "xrspatial.gpu_rtx._memory._available_gpu_memory_bytes",
        return_value=100,
    ):
        with pytest.raises(MemoryError, match="hillshade_rtx"):
            _check_gpu_memory("hillshade_rtx", 4, 4)


def test_check_gpu_memory_skipped_when_no_cuda():
    """When the helper can't query free memory, the guard is a no-op."""
    with patch(
        "xrspatial.gpu_rtx._memory._available_gpu_memory_bytes",
        return_value=0,
    ):
        # Absurd size still passes when no GPU info is available.
        _check_gpu_memory("hillshade_rtx", 10**6, 10**6)


def test_check_gpu_memory_error_names_func_and_shape():
    """The error message should include the func name and the raster shape."""
    with patch(
        "xrspatial.gpu_rtx._memory._available_gpu_memory_bytes",
        return_value=10,
    ):
        with pytest.raises(MemoryError) as excinfo:
            _check_gpu_memory("viewshed_gpu", 4, 4)

    msg = str(excinfo.value)
    assert "viewshed_gpu" in msg
    assert "4x4" in msg
    assert "GB" in msg


def test_available_gpu_memory_returns_int():
    """The helper should always return a non-negative int."""
    val = _available_gpu_memory_bytes()
    assert isinstance(val, int)
    assert val >= 0


# ----------------------------------------------------------------------
# End-to-end tests that hit the RTX entry points
# ----------------------------------------------------------------------

def _make_small_raster():
    """A tiny gaussian bump raster suitable for hillshade/viewshed."""
    n = 16
    _x = np.linspace(0, 50, n)
    X, Y = np.meshgrid(_x, _x, sparse=True)
    gauss = np.exp(-((X - 25) ** 2 + (Y - 25) ** 2) / 50.0).astype(np.float32)
    raster = xr.DataArray(
        gauss,
        dims=["y", "x"],
        coords={"y": np.arange(n)[::-1].astype(float), "x": np.arange(n).astype(float)},
    )
    return raster


@pytest.mark.skipif(not has_rtx(), reason="RTX not available")
def test_hillshade_rtx_memory_guard_raises_for_oversize():
    """hillshade_rtx should raise MemoryError before any cupy.empty call."""
    import cupy

    from xrspatial import hillshade

    raster = _make_small_raster()
    raster.data = cupy.asarray(raster.data)

    with patch(
        "xrspatial.gpu_rtx._memory._available_gpu_memory_bytes",
        return_value=100,
    ):
        with pytest.raises(MemoryError, match="hillshade_rtx"):
            hillshade(raster, shadows=True)


@pytest.mark.skipif(not has_rtx(), reason="RTX not available")
def test_hillshade_rtx_memory_guard_passes_for_small_raster():
    """A tiny cupy raster should pass the guard at realistic free memory."""
    import cupy

    from xrspatial import hillshade

    raster = _make_small_raster()
    raster.data = cupy.asarray(raster.data)

    # No patch -- use real free GPU memory.  16x16 raster needs 30 KB at
    # 120 B/px, well below half of any reasonable free-memory level.
    out = hillshade(raster, shadows=True)
    assert out.shape == raster.shape


@pytest.mark.skipif(not has_rtx(), reason="RTX not available")
def test_viewshed_gpu_memory_guard_raises_for_oversize():
    """viewshed_gpu should raise MemoryError before any cupy.empty call."""
    import cupy

    from xrspatial import viewshed

    raster = _make_small_raster()
    raster.data = cupy.asarray(raster.data)

    with patch(
        "xrspatial.gpu_rtx._memory._available_gpu_memory_bytes",
        return_value=100,
    ):
        with pytest.raises(MemoryError, match="viewshed_gpu"):
            viewshed(raster, x=8.0, y=8.0, observer_elev=10.0, target_elev=0.0)


@pytest.mark.skipif(not has_rtx(), reason="RTX not available")
def test_viewshed_gpu_memory_guard_passes_for_small_raster():
    """A tiny cupy raster should pass the guard at realistic free memory."""
    import cupy

    from xrspatial import viewshed

    raster = _make_small_raster()
    raster.data = cupy.asarray(raster.data)

    out = viewshed(raster, x=8.0, y=8.0, observer_elev=10.0, target_elev=0.0)
    assert out.shape == raster.shape
