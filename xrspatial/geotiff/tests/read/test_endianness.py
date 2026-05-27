"""Big-endian / little-endian GeoTIFF reader paths.

Big-endian multi-byte TIFFs read via ``read_geotiff_gpu`` once crashed
inside the GPU decode pipeline because ``cupy.ndarray`` does not expose
``byteswap()``. The dispatcher caught the error and silently fell back to
CPU, so results stayed correct but the GPU fast path was lost. These
tests pin the GPU byteswap path.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from .._helpers.markers import gpu_available

_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None
_gpu_only = pytest.mark.skipif(
    not (gpu_available() and _HAS_TIFFFILE),
    reason="cupy + CUDA + tifffile required",
)


@_gpu_only
@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.uint32, np.int32])
def test_read_geotiff_gpu_big_endian_multibyte(tmp_path, dtype):
    """GPU path decodes BE multi-byte tiles and stays on GPU."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

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
    assert cpu.dtype == np.dtype(dtype), (
        f"CPU baseline drifted from native dtype: got {cpu.dtype}"
    )

    gpu_da = read_geotiff_gpu(str(path))

    assert isinstance(gpu_da.data, cupy.ndarray), (
        "expected cupy-backed DataArray, got "
        f"{type(gpu_da.data).__name__} -- the GPU path likely fell back "
        "to CPU again"
    )

    assert gpu_da.data.dtype == np.dtype(dtype), (
        f"GPU result dtype {gpu_da.data.dtype} drifted from native "
        f"{np.dtype(dtype)}"
    )
    assert gpu_da.data.dtype.isnative, (
        f"GPU result dtype is non-native byteorder: {gpu_da.data.dtype!r}"
    )

    np.testing.assert_array_equal(gpu_da.data.get(), arr)


@_gpu_only
def test_read_geotiff_gpu_big_endian_uncompressed(tmp_path):
    """Uncompressed BE multi-byte tiles also stay on the GPU."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu

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
    assert gpu_da.data.dtype == np.dtype(np.uint16)
    assert gpu_da.data.dtype.isnative
    np.testing.assert_array_equal(gpu_da.data.get(), arr)


def test_xp_byteswap_preserves_dtype():
    """``_xp_byteswap`` must keep the input dtype (just like numpy.byteswap)."""
    from xrspatial.geotiff._gpu_decode import _xp_byteswap

    for dtype in (np.uint16, np.int16, np.uint32, np.int32, np.float32,
                  np.float64):
        a = np.array([1, 2, 3, 4], dtype=dtype)
        swapped = _xp_byteswap(a)
        assert swapped.dtype == a.dtype, (
            f"{dtype.__name__}: dtype changed from {a.dtype} to {swapped.dtype}"
        )
        assert swapped.dtype.isnative
        np.testing.assert_array_equal(swapped, a.byteswap())


def test_xp_byteswap_uint8_passthrough():
    """1-byte dtypes have nothing to swap; helper returns input unchanged."""
    from xrspatial.geotiff._gpu_decode import _xp_byteswap

    a = np.array([1, 2, 3], dtype=np.uint8)
    out = _xp_byteswap(a)
    assert out is a or np.array_equal(out, a)
    assert out.dtype == np.uint8
