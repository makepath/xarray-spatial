"""Regression tests for issue #1605.

``open_geotiff(source, gpu=True, window=..., band=...)`` used to silently
drop ``window`` and ``band``: the dispatcher did not forward them, and
``read_geotiff_gpu`` did not declare them. The fix adds both kwargs to
``read_geotiff_gpu`` and threads them through the dispatcher so the GPU
path returns the same subrectangle / band selection as the CPU eager
path and the dask path.

These tests pin:

1. Direct ``read_geotiff_gpu(..., window=..., band=...)`` accepts the
   kwargs and produces the expected shape/values.
2. ``open_geotiff(..., gpu=True, window=..., band=...)`` no longer
   silently returns the full file.
3. The GPU windowed coords / transform attr match the CPU eager path
   bit-for-bit (so the only difference is the array backend).
4. Bounds validation for ``window`` and ``band`` raises the same errors
   the CPU paths raise.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr


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
    not _HAS_GPU, reason="cupy + CUDA required",
)


@pytest.fixture
def single_band_tiff(tmp_path):
    """16x20 single-band tiled tiff with a non-trivial transform."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={
            'y': np.arange(16, dtype=np.float64) * 0.25 + 100.0,
            'x': np.arange(20, dtype=np.float64) * 0.25 + 200.0,
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'window_band_1605_single.tif'
    to_geotiff(da, str(p), tile_size=8)
    return str(p), arr


@pytest.fixture
def multi_band_tiff(tmp_path):
    """16x20x3 tiled tiff -- exercises the band-selection branch."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(16 * 20 * 3, dtype=np.float32).reshape(16, 20, 3)
    da = xr.DataArray(
        arr,
        dims=['y', 'x', 'band'],
        coords={
            'y': np.arange(16, dtype=np.float64) * 0.25 + 100.0,
            'x': np.arange(20, dtype=np.float64) * 0.25 + 200.0,
            'band': np.arange(3),
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'window_band_1605_multi.tif'
    to_geotiff(da, str(p), tile_size=8)
    return str(p), arr


@_gpu_only
def test_read_geotiff_gpu_window_matches_eager(single_band_tiff):
    """Direct call: GPU window slice matches CPU eager window slice."""
    path, source_arr = single_band_tiff
    from xrspatial.geotiff import open_geotiff, read_geotiff_gpu

    window = (2, 4, 12, 14)

    cpu = open_geotiff(path, window=window)
    gpu = read_geotiff_gpu(path, window=window)

    assert gpu.shape == cpu.shape == (10, 10)
    np.testing.assert_array_equal(gpu.data.get(), cpu.data)
    np.testing.assert_array_equal(gpu['y'].values, cpu['y'].values)
    np.testing.assert_array_equal(gpu['x'].values, cpu['x'].values)
    assert gpu.attrs.get('transform') == cpu.attrs.get('transform')


@_gpu_only
def test_open_geotiff_gpu_window_no_longer_silently_dropped(single_band_tiff):
    """Issue #1605: open_geotiff(gpu=True, window=...) honors the window."""
    path, source_arr = single_band_tiff
    from xrspatial.geotiff import open_geotiff

    window = (3, 5, 9, 13)
    gpu = open_geotiff(path, gpu=True, window=window)

    # Pre-fix shape was the full (16, 20). Post-fix it must shrink.
    assert gpu.shape == (6, 8)
    np.testing.assert_array_equal(
        gpu.data.get(),
        source_arr[3:9, 5:13],
    )


@_gpu_only
def test_read_geotiff_gpu_band_selection(multi_band_tiff):
    """Direct call: band=k returns the kth band as a 2D DataArray."""
    path, source_arr = multi_band_tiff
    from xrspatial.geotiff import open_geotiff, read_geotiff_gpu

    cpu = open_geotiff(path, band=1)
    gpu = read_geotiff_gpu(path, band=1)

    assert gpu.shape == cpu.shape == (16, 20)
    assert gpu.ndim == 2
    np.testing.assert_array_equal(gpu.data.get(), cpu.data)


@_gpu_only
def test_open_geotiff_gpu_band_no_longer_silently_dropped(multi_band_tiff):
    """Issue #1605: open_geotiff(gpu=True, band=...) honors band."""
    path, source_arr = multi_band_tiff
    from xrspatial.geotiff import open_geotiff

    gpu = open_geotiff(path, gpu=True, band=2)

    # Pre-fix would have returned a (16, 20, 3) DataArray (full file).
    assert gpu.shape == (16, 20)
    assert gpu.ndim == 2
    np.testing.assert_array_equal(
        gpu.data.get(),
        source_arr[:, :, 2],
    )


@_gpu_only
def test_read_geotiff_gpu_window_and_band(multi_band_tiff):
    """window + band combine cleanly."""
    path, source_arr = multi_band_tiff
    from xrspatial.geotiff import open_geotiff, read_geotiff_gpu

    window = (1, 2, 11, 17)
    cpu = open_geotiff(path, window=window, band=0)
    gpu = read_geotiff_gpu(path, window=window, band=0)

    assert gpu.shape == cpu.shape == (10, 15)
    assert gpu.ndim == 2
    np.testing.assert_array_equal(gpu.data.get(), cpu.data)


@_gpu_only
def test_read_geotiff_gpu_window_bounds_validation(single_band_tiff):
    """Out-of-bounds window raises ValueError, mirroring the dask path."""
    path, _ = single_band_tiff
    from xrspatial.geotiff import read_geotiff_gpu

    with pytest.raises(ValueError, match="outside the source extent"):
        read_geotiff_gpu(path, window=(0, 0, 100, 100))

    with pytest.raises(ValueError, match="non-positive size"):
        read_geotiff_gpu(path, window=(5, 0, 5, 10))

    with pytest.raises(ValueError, match="must be a 4-tuple"):
        read_geotiff_gpu(path, window=(0, 0, 5))


@_gpu_only
def test_read_geotiff_gpu_band_bounds_validation(multi_band_tiff,
                                                  single_band_tiff):
    """Out-of-range band raises IndexError."""
    multi_path, _ = multi_band_tiff
    single_path, _ = single_band_tiff
    from xrspatial.geotiff import read_geotiff_gpu

    with pytest.raises(IndexError, match="out of range"):
        read_geotiff_gpu(multi_path, band=10)

    with pytest.raises(IndexError, match="single-band file"):
        read_geotiff_gpu(single_path, band=1)
