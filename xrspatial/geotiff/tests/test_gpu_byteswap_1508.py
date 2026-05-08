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

import importlib.util

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
_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None
_gpu_only = pytest.mark.skipif(
    not (_HAS_GPU and _HAS_TIFFFILE),
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

    # The GPU path was actually exercised (no silent CPU fallback masking
    # a crash inside gpu_decode_tiles_from_file).
    assert isinstance(gpu_da.data, cupy.ndarray), (
        "expected cupy-backed DataArray, got "
        f"{type(gpu_da.data).__name__} -- the GPU path likely fell back "
        "to CPU again"
    )

    # The fix must preserve the native dtype contract. An earlier version
    # used ``arr.view(arr.dtype.newbyteorder()).copy()`` which produced an
    # array tagged with non-native byteorder (``>u2`` instead of ``<u2``).
    # That is values-correct but breaks downstream consumers that expect
    # native dtypes (numba ``@ngjit`` rejects non-native arrays -- this is
    # the same class of bug PR #1507 fixed for predictor=2 BE).
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
