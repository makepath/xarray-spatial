"""Tests for the gpu_rtx mesh-building guard (Issue #1378).

``create_triangulation()`` in ``xrspatial/gpu_rtx/mesh_utils.py`` divides
the raster's max width/height by ``cupy.amax(raster.data)`` to produce a
ratio-preserving z-scale.  An all-zero raster (or one whose max is NaN /
non-positive) made the divide produce ``inf`` / ``NaN`` and propagated
garbage geometry into OptiX.

These tests verify that the guard raises ``ValueError`` before any hash
or device-buffer work happens.  They only need cupy (for the cupy.amax
call inside the function); they do not need an actual RTX device because
the guard runs before ``optix.getHash()``.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import has_cuda_and_cupy


pytestmark = pytest.mark.skipif(
    not has_cuda_and_cupy(),
    reason="cupy / CUDA not available",
)


def _cupy_raster(data):
    """Wrap a numpy array as a cupy-backed DataArray."""
    import cupy

    return xr.DataArray(cupy.asarray(data), dims=["y", "x"])


def test_create_triangulation_rejects_all_zero_raster():
    """An all-zero raster has maxH=0.0 -> divide-by-zero -> ValueError."""
    from xrspatial.gpu_rtx.mesh_utils import create_triangulation

    raster = _cupy_raster(np.zeros((8, 8), dtype=np.float32))
    optix = MagicMock()  # never reached: guard runs first

    with pytest.raises(ValueError, match="no positive elevation variance"):
        create_triangulation(raster, optix)

    # The guard must short-circuit before touching optix.
    optix.getHash.assert_not_called()
    optix.build.assert_not_called()


def test_create_triangulation_rejects_all_nan_raster():
    """An all-NaN raster has maxH=nan -> non-finite -> ValueError."""
    from xrspatial.gpu_rtx.mesh_utils import create_triangulation

    raster = _cupy_raster(np.full((8, 8), np.nan, dtype=np.float32))
    optix = MagicMock()

    with pytest.raises(ValueError, match="no positive elevation variance"):
        create_triangulation(raster, optix)

    optix.getHash.assert_not_called()


def test_create_triangulation_rejects_non_positive_maxima():
    """A raster whose only non-NaN values are <= 0 must also be rejected.

    The mesh-scale formula assumes maxH > 0; otherwise the resulting
    scale either inverts the mesh (negative max) or is infinite (zero
    max).  Both are garbage-geometry conditions.
    """
    from xrspatial.gpu_rtx.mesh_utils import create_triangulation

    raster = _cupy_raster(np.full((8, 8), -3.0, dtype=np.float32))
    optix = MagicMock()

    with pytest.raises(ValueError, match="no positive elevation variance"):
        create_triangulation(raster, optix)


def test_create_triangulation_accepts_single_nonzero_pixel():
    """A raster with only one positive pixel must NOT raise.

    maxH is finite and > 0, so the guard passes and the function
    proceeds to mesh building.  We use a fake optix that returns a
    hash matching the data hash, so the build path is skipped and the
    function returns the computed scale without trying to allocate
    device buffers we don't care about for this test.
    """
    from xrspatial.gpu_rtx.mesh_utils import create_triangulation

    data = np.zeros((4, 4), dtype=np.float32)
    data[2, 2] = 5.0  # single non-zero -> maxH = 5.0
    raster = _cupy_raster(data)

    # Make optix.getHash() return the same hash the function will compute
    # for the data, so the `if optixhash != datahash:` block is skipped.
    expected_hash = np.uint64(hash(str(raster.data.get())) % (1 << 64))
    optix = MagicMock()
    optix.getHash.return_value = int(expected_hash)

    scale = create_triangulation(raster, optix)

    # max(H, W) / maxH = 4 / 5.0
    assert scale == pytest.approx(4.0 / 5.0)
    optix.build.assert_not_called()  # hashes matched, no rebuild


def test_create_triangulation_error_message_includes_max_value():
    """The ValueError message should report the offending max value."""
    from xrspatial.gpu_rtx.mesh_utils import create_triangulation

    raster = _cupy_raster(np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(ValueError) as excinfo:
        create_triangulation(raster, MagicMock())

    msg = str(excinfo.value)
    assert "max=" in msg
    assert "0.0" in msg
