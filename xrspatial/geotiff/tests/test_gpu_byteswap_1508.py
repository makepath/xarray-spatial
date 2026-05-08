"""Regression test for issue #1508.

Big-endian multi-byte TIFFs read via ``read_geotiff_gpu`` used to crash
inside the GPU decode pipeline with::

    AttributeError: 'ndarray' object has no attribute 'byteswap'

because ``cupy.ndarray`` (as of cupy 13.x) does not expose ``byteswap()``.
The dispatcher in ``read_geotiff_gpu`` caught the error and silently fell
back to CPU, so results stayed correct but the GPU fast path was lost.

These tests confirm the GPU path now decodes BE multi-byte data directly
(result is a CuPy array, not a NumPy fallback) and matches the CPU read.
"""
from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
cupy = pytest.importorskip("cupy")
if not cupy.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from xrspatial.geotiff import read_geotiff_gpu  # noqa: E402
from xrspatial.geotiff._reader import read_to_array  # noqa: E402


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.uint32, np.int32])
def test_read_geotiff_gpu_big_endian_multibyte(tmp_path, dtype):
    """GPU path decodes BE multi-byte tiles and stays on GPU."""
    rng = np.random.RandomState(20260507)
    info = np.iinfo(dtype)
    arr = rng.randint(
        info.min, info.max, size=(32, 48), dtype=np.int64
    ).astype(dtype)

    path = tmp_path / f"be_{np.dtype(dtype).name}.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", compression="deflate",
        tile=(16, 16),
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    gpu_da = read_geotiff_gpu(str(path))

    # The GPU path was actually exercised (no silent CPU fallback masking
    # a crash inside gpu_decode_tiles_from_file).
    assert isinstance(gpu_da.data, cupy.ndarray), (
        "expected cupy-backed DataArray, got "
        f"{type(gpu_da.data).__name__} -- the GPU path likely fell back "
        "to CPU again"
    )

    np.testing.assert_array_equal(gpu_da.data.get(), arr)


def test_read_geotiff_gpu_big_endian_uncompressed(tmp_path):
    """Uncompressed BE multi-byte tiles also stay on the GPU."""
    rng = np.random.RandomState(20260507)
    arr = rng.randint(0, 60000, size=(32, 48), dtype=np.uint16)

    path = tmp_path / "be_uint16_raw.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", compression=None, tile=(16, 16),
    )

    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray), (
        "expected cupy-backed DataArray; GPU path may have fallen back"
    )
    np.testing.assert_array_equal(gpu_da.data.get(), arr)
