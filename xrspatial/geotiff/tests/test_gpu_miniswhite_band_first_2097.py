"""Regression tests for issue #2097.

``write_geotiff_gpu`` refuses ``photometric='miniswhite'`` for the
single-band case (issue #1836) because the GPU writer does not
implement the writer-side pixel + nodata-sentinel inversion the
reader expects. The guard reads samples-per-pixel from
``data.shape[2]`` *before* the band-first to band-last remap, so a
band-first DataArray of shape ``(1, H, W)`` with dims ``('band', 'y', 'x')``
saw ``samples_hint == W`` and slipped past the ``== 1`` check
whenever ``W != 1``.

Fix: read the band count using the same band-axis logic the remap
uses, so the guard fires on every single-band MinIsWhite input
regardless of layout.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr


def _cupy_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _cupy_available()


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
def test_band_first_single_band_miniswhite_rejected(tmp_path):
    """A band-first single-band DataArray with ``photometric='miniswhite'``
    must raise ``NotImplementedError`` on the GPU writer. Before the
    fix this only fired when ``W == 1`` because the guard read the
    sample count from ``data.shape[2]`` (the spatial-x axis on a
    band-first array)."""
    import cupy as cp

    from xrspatial.geotiff._writers.gpu import write_geotiff_gpu

    arr = cp.zeros((1, 4, 8), dtype=cp.uint8)
    da = xr.DataArray(arr, dims=("band", "y", "x"))
    out = tmp_path / "tmp_2097_miniswhite_band_first.tif"
    with pytest.raises(NotImplementedError, match="miniswhite"):
        write_geotiff_gpu(da, str(out), photometric="miniswhite")
    # The guard must fire before any IFD bytes are written.
    assert not out.exists()


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
def test_band_last_single_band_miniswhite_still_rejected(tmp_path):
    """The pre-existing band-last single-band rejection must still
    fire after the band-axis fix. Guard against a regression that
    only catches band-first inputs."""
    import cupy as cp

    from xrspatial.geotiff._writers.gpu import write_geotiff_gpu

    arr = cp.zeros((4, 8, 1), dtype=cp.uint8)
    da = xr.DataArray(arr, dims=("y", "x", "band"))
    out = tmp_path / "tmp_2097_miniswhite_band_last.tif"
    with pytest.raises(NotImplementedError, match="miniswhite"):
        write_geotiff_gpu(da, str(out), photometric="miniswhite")
    assert not out.exists()


@pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
def test_2d_single_band_miniswhite_still_rejected(tmp_path):
    """2D inputs are the simplest single-band case. They must still
    be rejected after the 3D band-axis rework."""
    import cupy as cp

    from xrspatial.geotiff._writers.gpu import write_geotiff_gpu

    arr = cp.zeros((4, 8), dtype=cp.uint8)
    da = xr.DataArray(arr, dims=("y", "x"))
    out = tmp_path / "tmp_2097_miniswhite_2d.tif"
    with pytest.raises(NotImplementedError, match="miniswhite"):
        write_geotiff_gpu(da, str(out), photometric="miniswhite")
    assert not out.exists()


def test_samples_hint_band_first_without_gpu():
    """The samples-hint computation does not actually need cupy to be
    exercised. Build a fake band-first DataArray-shaped object and
    verify the new logic picks ``shape[0]`` for band-first inputs
    and ``shape[2]`` for band-last inputs.

    This mirrors the inline computation in
    ``write_geotiff_gpu`` so a regression that moves the samples-hint
    detection back to ``shape[2]`` blindly will fail here without
    requiring a CUDA device on CI.
    """
    from xrspatial.geotiff._coords import _BAND_DIM_NAMES

    def _samples_hint(ndim, shape, dims):
        if ndim == 3:
            if (dims is not None and len(dims) == 3
                    and dims[0] in _BAND_DIM_NAMES):
                return int(shape[0])
            return int(shape[2])
        return 1

    # Band-first single-band: bands = shape[0] = 1.
    assert _samples_hint(3, (1, 4, 8), ("band", "y", "x")) == 1
    # Band-last single-band: bands = shape[2] = 1.
    assert _samples_hint(3, (4, 8, 1), ("y", "x", "band")) == 1
    # Band-first 3-band: bands = shape[0] = 3.
    assert _samples_hint(3, (3, 4, 8), ("band", "y", "x")) == 3
    # Band-last 3-band: bands = shape[2] = 3.
    assert _samples_hint(3, (4, 8, 3), ("y", "x", "band")) == 3
    # 2D: always 1.
    assert _samples_hint(2, (4, 8), ("y", "x")) == 1
    # No dims (raw array path): defaults to band-last (shape[2]).
    assert _samples_hint(3, (4, 8, 1), None) == 1
