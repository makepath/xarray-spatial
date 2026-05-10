"""GPU read backend nodata propagation tests for issue #1542.

Before the fix, ``read_geotiff_gpu`` silently differed from the CPU
eager path on rasters with a declared nodata sentinel:

* The returned DataArray's ``attrs`` had no ``nodata`` key, even when
  the file declared one via the GDAL_NODATA tag.
* The pixel data still carried the raw sentinel: integer rasters were
  not promoted to float64 with NaN, and float rasters kept the
  sentinel rather than NaN.

These tests pin the contract that the GPU read agrees with the CPU
read on dtype, NaN positions, and ``attrs['nodata']`` for the
combinations the audit found broken.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
import pytest


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialised."""
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
def test_gpu_uint16_nodata_promoted_and_masked_tiled(tmp_path):
    """uint16 + nodata sentinel -> GPU returns float64 with NaN, attrs[nodata] set."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'gpu_u16_nodata_1542_tiled.tif')
    write(arr, path, nodata=65535, compression='deflate',
          tiled=True, tile_size=16)

    cpu = open_geotiff(path)
    gpu = open_geotiff(path, gpu=True)

    assert cpu.dtype == gpu.dtype == np.float64
    assert cpu.attrs.get('nodata') == 65535.0
    assert gpu.attrs.get('nodata') == 65535.0
    cpu_arr = cpu.values
    gpu_arr = gpu.data.get()
    np.testing.assert_array_equal(np.isnan(cpu_arr), np.isnan(gpu_arr))
    finite = ~np.isnan(cpu_arr)
    np.testing.assert_array_equal(cpu_arr[finite], gpu_arr[finite])


@_gpu_only
def test_gpu_uint16_nodata_promoted_and_masked_stripped(tmp_path):
    """Stripped fallback path also promotes + masks integer nodata."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'gpu_u16_nodata_1542_stripped.tif')
    write(arr, path, nodata=65535, compression='deflate', tiled=False)

    cpu = open_geotiff(path)
    gpu = open_geotiff(path, gpu=True)

    assert cpu.dtype == gpu.dtype == np.float64
    assert gpu.attrs.get('nodata') == 65535.0
    np.testing.assert_array_equal(np.isnan(cpu.values),
                                  np.isnan(gpu.data.get()))


@_gpu_only
def test_gpu_float32_sentinel_replaced_with_nan(tmp_path):
    """float32 + finite sentinel -> GPU returns float32 with NaN at sentinel positions."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    arr[0, 0] = -9999.0
    arr[2, 3] = -9999.0
    path = str(tmp_path / 'gpu_f32_sentinel_1542.tif')
    write(arr, path, nodata=-9999.0, compression='deflate',
          tiled=True, tile_size=16)

    cpu = open_geotiff(path)
    gpu = open_geotiff(path, gpu=True)

    assert cpu.dtype == gpu.dtype == np.float32
    assert gpu.attrs.get('nodata') == -9999.0
    cpu_arr = cpu.values
    gpu_arr = gpu.data.get()
    np.testing.assert_array_equal(np.isnan(cpu_arr), np.isnan(gpu_arr))
    finite = ~np.isnan(cpu_arr)
    np.testing.assert_array_equal(cpu_arr[finite], gpu_arr[finite])


@_gpu_only
def test_gpu_no_nodata_keeps_dtype(tmp_path):
    """No nodata declared -> GPU keeps source dtype, no nodata attr added."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'gpu_u16_no_nodata_1542.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=16)

    gpu = open_geotiff(path, gpu=True)
    assert gpu.dtype == np.uint16
    assert gpu.attrs.get('nodata') is None
    np.testing.assert_array_equal(gpu.data.get(), arr)


@_gpu_only
def test_gpu_nan_nodata_passes_through(tmp_path):
    """nodata=NaN on float data -> GPU returns NaN positions intact, attrs[nodata]=nan."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
    path = str(tmp_path / 'gpu_f32_nan_1542.tif')
    write(arr, path, nodata=float('nan'), compression='deflate',
          tiled=True, tile_size=16)

    gpu = open_geotiff(path, gpu=True)
    assert np.isnan(gpu.attrs.get('nodata'))
    gpu_arr = gpu.data.get()
    assert np.isnan(gpu_arr[1, 0])
    finite = ~np.isnan(gpu_arr)
    np.testing.assert_array_equal(gpu_arr[finite], arr[~np.isnan(arr)])


@_gpu_only
def test_gpu_all_four_backends_agree_on_nodata(tmp_path):
    """numpy / dask+numpy / cupy / dask+cupy all agree on dtype + nodata + NaN."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[1, 2, 3], [65535, 5, 6]], dtype=np.uint16)
    path = str(tmp_path / 'gpu_4backends_1542.tif')
    write(arr, path, nodata=65535, compression='deflate',
          tiled=True, tile_size=16)

    da_np = open_geotiff(path)
    da_dask = open_geotiff(path, chunks=512)
    da_gpu = open_geotiff(path, gpu=True)
    da_gpu_dask = open_geotiff(path, gpu=True, chunks=512)

    # dtype
    for label, da in [('np', da_np), ('dask+np', da_dask),
                      ('gpu', da_gpu), ('gpu+dask', da_gpu_dask)]:
        assert da.dtype == np.float64, f"{label}: dtype={da.dtype}"
        assert da.attrs.get('nodata') == 65535.0, (
            f"{label}: missing nodata attr (got {da.attrs.get('nodata')!r})"
        )

    # NaN positions
    np_arr = da_np.values
    dask_arr = da_dask.compute().values
    gpu_arr = da_gpu.data.get()
    gpu_dask_arr = da_gpu_dask.compute().data.get()
    np.testing.assert_array_equal(np.isnan(np_arr), np.isnan(dask_arr))
    np.testing.assert_array_equal(np.isnan(np_arr), np.isnan(gpu_arr))
    np.testing.assert_array_equal(np.isnan(np_arr), np.isnan(gpu_dask_arr))


@_gpu_only
def test_gpu_int16_negative_nodata(tmp_path):
    """Signed integer with negative nodata: also promoted to float64 + NaN."""
    from xrspatial.geotiff import open_geotiff
    from xrspatial.geotiff._writer import write

    arr = np.array([[-9999, 10], [20, -9999]], dtype=np.int16)
    path = str(tmp_path / 'gpu_i16_nodata_1542.tif')
    write(arr, path, nodata=-9999, compression='deflate',
          tiled=True, tile_size=16)

    cpu = open_geotiff(path)
    gpu = open_geotiff(path, gpu=True)
    assert cpu.dtype == gpu.dtype == np.float64
    assert gpu.attrs.get('nodata') == -9999.0
    np.testing.assert_array_equal(np.isnan(cpu.values),
                                  np.isnan(gpu.data.get()))
