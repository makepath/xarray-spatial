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


class _FakeArray:
    """Stand-in for a DataArray-like object that exposes ``ndim``,
    ``shape``, and ``dims``. Used to drive ``_compute_gpu_samples_hint``
    on a CI runner without cupy or a CUDA device."""

    def __init__(self, shape, dims):
        self.shape = shape
        self.dims = dims
        self.ndim = len(shape)


def test_samples_hint_band_first_without_gpu():
    """Call the production ``_compute_gpu_samples_hint`` helper
    directly so a future change that moves the samples-hint detection
    back to a blind ``shape[2]`` lookup fails here even on a no-GPU
    CI runner. (Pre-fix the test redefined the logic inline, which
    would not catch a regression in the production code.)
    """
    from xrspatial.geotiff._writers.gpu import _compute_gpu_samples_hint

    cases = [
        # (shape, dims, expected_samples)
        ((1, 4, 8), ("band", "y", "x"), 1),    # band-first single-band
        ((4, 8, 1), ("y", "x", "band"), 1),    # band-last single-band
        ((3, 4, 8), ("band", "y", "x"), 3),    # band-first 3-band
        ((4, 8, 3), ("y", "x", "band"), 3),    # band-last 3-band
        ((4, 8), ("y", "x"), 1),                # 2D always 1
        ((4, 8, 1), None, 1),                   # raw 3D defaults band-last
    ]
    for shape, dims, expected in cases:
        got = _compute_gpu_samples_hint(_FakeArray(shape, dims))
        assert got == expected, (shape, dims, got, expected)
