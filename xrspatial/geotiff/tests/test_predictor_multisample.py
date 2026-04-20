"""Regression tests for GPU predictor decode on multi-sample tiled TIFFs.

Covers issue #1220: the GPU predictor=2 decode path passed
`width=tile_width * samples` and `bytes_per_sample=itemsize * samples`
to `_predictor_decode_kernel`, causing `row_bytes` to be
`tile_width * samples**2 * itemsize` instead of
`tile_width * samples * itemsize`. That walks the cumulative-sum loop
past the end of each tile row and, on the last tile, past the end of
the `d_decomp` buffer (silent OOB GPU write).

These tests build multi-sample tiled TIFFs with predictor=2 on the
CPU path, decode them via the GPU path, and assert byte-for-byte
equality with the CPU decode.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


try:
    import cupy  # noqa: F401
    from numba import cuda
    _HAS_GPU = cuda.is_available()
except Exception:
    _HAS_GPU = False


pytestmark = pytest.mark.skipif(
    not _HAS_GPU, reason="Requires CuPy + CUDA for GPU decode path")


def _gpu_to_numpy(da: xr.DataArray) -> np.ndarray:
    """Pull a CuPy-backed DataArray to host memory."""
    arr = da.data
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


@pytest.mark.parametrize("samples,dtype_str", [
    (3, "uint8"),   # RGB uint8
    (4, "uint8"),   # RGBA uint8
    (3, "uint16"),  # RGB uint16
])
def test_gpu_predictor2_multisample_matches_cpu(tmp_path, samples, dtype_str):
    """GPU decode of a tiled multi-sample TIFF with predictor=2 matches CPU.

    Before the fix, the GPU predictor=2 kernel over-indexed the
    decompressed buffer by a factor of `samples` on the inner loop,
    corrupting the decoded pixels and writing past the end of
    ``d_decomp`` on the last tile.
    """
    dtype = np.dtype(dtype_str)
    h, w = 16, 16
    rng = np.random.RandomState(42)
    if dtype.kind == 'u' and dtype.itemsize == 1:
        data = rng.randint(0, 256, size=(h, w, samples), dtype=dtype)
    else:
        high = np.iinfo(dtype).max if dtype.kind in ('u', 'i') else 1000
        data = rng.randint(0, high, size=(h, w, samples), dtype=dtype)

    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    path = str(tmp_path / f"rgb_pred_{samples}_{dtype_str}.tif")
    # tile_size smaller than image so we exercise more than one tile.
    to_geotiff(da, path, compression='deflate', tile_size=8, predictor=True)

    cpu_arr = open_geotiff(path).values
    assert cpu_arr.shape == (h, w, samples)
    assert cpu_arr.dtype == dtype
    # Sanity: CPU round-trip reproduces the source exactly.
    np.testing.assert_array_equal(cpu_arr, data)

    gpu_da = open_geotiff(path, gpu=True)
    gpu_arr = _gpu_to_numpy(gpu_da)

    assert gpu_arr.shape == cpu_arr.shape
    assert gpu_arr.dtype == cpu_arr.dtype
    # Byte-for-byte equality is the whole point: predictor=2 is exact.
    np.testing.assert_array_equal(gpu_arr, cpu_arr)


def test_gpu_predictor2_multisample_uneven_tiles(tmp_path):
    """Image size not a multiple of tile size, multi-sample, predictor=2.

    Exercises partial edge tiles which share the same predictor decode
    path and are most likely to trip any lingering stride mismatch.
    """
    h, w, samples = 20, 20, 3  # 20 is not a multiple of 8
    rng = np.random.RandomState(7)
    data = rng.randint(0, 256, size=(h, w, samples), dtype=np.uint8)
    da = xr.DataArray(data, dims=['y', 'x', 'band'])

    path = str(tmp_path / "rgb_pred_uneven.tif")
    to_geotiff(da, path, compression='deflate', tile_size=8, predictor=True)

    cpu_arr = open_geotiff(path).values
    gpu_arr = _gpu_to_numpy(open_geotiff(path, gpu=True))

    np.testing.assert_array_equal(gpu_arr, cpu_arr)
    np.testing.assert_array_equal(gpu_arr, data)
