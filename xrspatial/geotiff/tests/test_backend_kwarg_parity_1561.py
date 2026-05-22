"""Regression tests for issue #1561.

``open_geotiff`` and ``to_geotiff`` route to backend-specific entry
points (``read_geotiff_dask``, ``write_geotiff_gpu``) whose kwarg sets
were narrower than the dispatcher's. The dispatcher silently dropped
the missing kwargs when it routed to the smaller-API backend.

These tests pin the kwargs through to each backend so dispatcher calls
no longer lose them.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialized."""
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


@pytest.fixture
def small_tiff_path(tmp_path):
    """4x6 single-band tiled tiff with a small CRS+transform."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'parity_1561_small.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


@pytest.fixture
def small_multiband_tiff_path(tmp_path):
    """4x6 three-band tiled tiff."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(72, dtype=np.float32).reshape(4, 6, 3)
    da = xr.DataArray(
        arr,
        dims=['y', 'x', 'band'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
            'band': [0, 1, 2],
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'parity_1561_mb.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


# --------------------------------------------------------------------
# read_geotiff_dask: window / band / max_pixels now threaded through
# --------------------------------------------------------------------


def test_read_geotiff_dask_window_clips_region(small_tiff_path):
    """``window=`` restricts the lazy region; chunks span only the window."""
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = small_tiff_path
    da = read_geotiff_dask(path, chunks=2, window=(1, 2, 4, 6))
    assert da.shape == (3, 4)
    np.testing.assert_array_equal(da.values, arr[1:4, 2:6])


def test_read_geotiff_dask_window_via_dispatcher(small_tiff_path):
    """``open_geotiff(window=..., chunks=...)`` now keeps the window."""
    from xrspatial.geotiff import open_geotiff

    path, arr = small_tiff_path
    da = open_geotiff(path, window=(0, 1, 3, 4), chunks=2)
    assert da.shape == (3, 3)
    np.testing.assert_array_equal(da.values, arr[0:3, 1:4])


def test_read_geotiff_dask_band_selects_single_band(small_multiband_tiff_path):
    """``band=`` produces a 2D DataArray with the selected band."""
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = small_multiband_tiff_path
    da = read_geotiff_dask(path, chunks=4, band=1)
    assert da.ndim == 2
    np.testing.assert_array_equal(da.values, arr[:, :, 1])


def test_read_geotiff_dask_band_via_dispatcher(small_multiband_tiff_path):
    """``open_geotiff(band=..., chunks=...)`` now keeps the band."""
    from xrspatial.geotiff import open_geotiff

    path, arr = small_multiband_tiff_path
    da = open_geotiff(path, band=2, chunks=4)
    assert da.ndim == 2
    np.testing.assert_array_equal(da.values, arr[:, :, 2])


def test_read_geotiff_dask_max_pixels_rejects_oversized(small_tiff_path):
    """``max_pixels=`` rejects the windowed region up front."""
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="exceeds max_pixels"):
        read_geotiff_dask(path, chunks=2, max_pixels=10)


def test_read_geotiff_dask_window_band_combined(small_multiband_tiff_path):
    """``window`` and ``band`` cooperate."""
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = small_multiband_tiff_path
    da = read_geotiff_dask(path, chunks=2, window=(1, 1, 4, 5), band=0)
    assert da.shape == (3, 4)
    np.testing.assert_array_equal(da.values, arr[1:4, 1:5, 0])


def test_read_geotiff_dask_invalid_window_raises(small_tiff_path):
    """Out-of-bounds windows fail loudly instead of silently clipping."""
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="window=.* is outside"):
        read_geotiff_dask(path, chunks=2, window=(0, 0, 100, 100))


def test_read_geotiff_dask_invalid_band_raises(small_multiband_tiff_path):
    """Out-of-range band indexes fail with IndexError."""
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = small_multiband_tiff_path
    with pytest.raises(IndexError, match="band=5 out of range"):
        read_geotiff_dask(path, chunks=4, band=5)


# --------------------------------------------------------------------
# write_geotiff_gpu: bigtiff / tiled / max_z_error / streaming_buffer_bytes
# now accepted (with appropriate rejections where the GPU path can't
# implement them).
# --------------------------------------------------------------------


def test_write_geotiff_gpu_rejects_tiled_false(tmp_path):
    """The GPU writer is tiled-only; ``tiled=False`` must fail loudly."""
    from xrspatial.geotiff import write_geotiff_gpu

    dummy = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="tiled=True"):
        # path is irrelevant -- validation fires before any file I/O.
        write_geotiff_gpu(dummy, str(tmp_path / 'never.tif'), tiled=False)


def test_write_geotiff_gpu_rejects_nonzero_max_z_error(tmp_path):
    """LERC budget is not implementable on the GPU path."""
    from xrspatial.geotiff import write_geotiff_gpu

    dummy = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="max_z_error is not supported"):
        write_geotiff_gpu(dummy, str(tmp_path / 'never.tif'), max_z_error=1.0)


@_gpu_only
def test_write_geotiff_gpu_accepts_streaming_buffer_bytes_as_noop(tmp_path):
    """``streaming_buffer_bytes`` is accepted for API parity (no-op)."""
    import cupy

    from xrspatial.geotiff import open_geotiff, write_geotiff_gpu

    arr = cupy.arange(16, dtype=cupy.float32).reshape(4, 4)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': np.arange(4, dtype=np.float64),
                              'x': np.arange(4, dtype=np.float64)})
    p = tmp_path / 'parity_1561_streaming.tif'
    # Argument is accepted; result must round-trip identically to a
    # call without it.
    write_geotiff_gpu(da, str(p), streaming_buffer_bytes=4096, tile_size=16)
    rd = open_geotiff(str(p))
    np.testing.assert_array_equal(rd.values, arr.get())


@_gpu_only
def test_to_geotiff_threads_tiled_false_into_gpu_dispatcher(tmp_path):
    """``to_geotiff(..., gpu=True, tiled=False)`` rejects, not silently flips."""
    import cupy

    from xrspatial.geotiff import to_geotiff

    arr = cupy.zeros((2, 2), dtype=cupy.float32)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': [0.0, 1.0], 'x': [0.0, 1.0]})
    with pytest.raises(ValueError, match="tiled=False"):
        to_geotiff(da, str(tmp_path / 'never.tif'),
                   gpu=True, tiled=False)
