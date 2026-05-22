"""Predictor=2 with signed 8-bit (``int8``) samples.

The predictor=2 (horizontal differencing) path historically tested
``uint8`` (byte-wise kernel) and 16/32-bit signed and unsigned integers
(multi-byte kernel). ``int8`` slipped through both buckets:

  * It has a single-byte stride so the reader takes the byte-wise
    path, but the sample format flag is ``2`` (signed) -- a different
    branch in the sample-format decoder than the well-trodden
    ``uint8`` path (sample format ``1``).
  * Signed 8-bit values exercise the integer-promotion semantics of
    the cumulative-sum reconstruction. If the kernel accidentally
    treated the byte stream as unsigned, the decode would produce
    wraparound errors at every transition through zero.

This module pins the read path:

  * Little-endian and big-endian predictor=2 ``int8`` files decode
    back to the original signed values byte-for-byte.
  * The GPU read path matches the CPU baseline.
  * Stripped and tiled layouts both succeed (separate code paths).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


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
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


def _signed_int8_grid(h: int = 16, w: int = 16) -> np.ndarray:
    """Deterministic ``int8`` raster spanning the full -128..127 range.

    The pattern is chosen to walk through the signed wraparound (going
    from positive into negative and back) on every row, so a bug that
    decodes the byte stream as unsigned would produce a different
    cumulative sum than the signed reference.
    """
    info = np.iinfo(np.int8)
    rng = np.random.RandomState(0x1781)
    return rng.randint(info.min, info.max + 1,
                       size=(h, w)).astype(np.int8)


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_cpu_predictor2_int8_round_trip(tmp_path, byteorder):
    """``int8`` predictor=2 round-trips through the CPU reader.

    Byte order is irrelevant for an 8-bit sample but the writer still
    accepts the parameter; both forms must decode identically.
    """
    from xrspatial.geotiff._reader import read_to_array

    arr = _signed_int8_grid()
    path = tmp_path / f"int8_pred2_{'be' if byteorder == '>' else 'le'}.tif"
    tifffile.imwrite(str(path), arr, byteorder=byteorder, predictor=2,
                     compression="deflate")

    out, _ = read_to_array(str(path))
    assert out.dtype == np.int8, f"dtype changed: {out.dtype}"
    np.testing.assert_array_equal(out, arr)


def test_cpu_predictor2_int8_tiled(tmp_path):
    """Tiled layout (separate ``_decode_tile`` path) handles int8 + pred=2."""
    from xrspatial.geotiff._reader import read_to_array

    arr = _signed_int8_grid(32, 48)
    path = tmp_path / "int8_pred2_tiled.tif"
    tifffile.imwrite(str(path), arr, predictor=2, compression="deflate",
                     tile=(16, 16))

    out, _ = read_to_array(str(path))
    assert out.dtype == np.int8
    np.testing.assert_array_equal(out, arr)


@_gpu_only
def test_gpu_predictor2_int8_tiled_matches_cpu(tmp_path):
    """The GPU tiled decoder matches the CPU baseline for int8 + pred=2."""
    import cupy

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    arr = _signed_int8_grid(32, 48)
    path = tmp_path / "int8_pred2_tiled_gpu.tif"
    tifffile.imwrite(str(path), arr, predictor=2, compression="deflate",
                     tile=(16, 16))

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.int8
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


@_gpu_only
def test_gpu_predictor2_int8_stripped_matches_cpu(tmp_path):
    """Stripped layout: GPU falls back to CPU; the result must still match."""
    import cupy

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    arr = _signed_int8_grid()
    path = tmp_path / "int8_pred2_stripped_gpu.tif"
    tifffile.imwrite(str(path), arr, predictor=2, compression="deflate")

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.int8
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)
