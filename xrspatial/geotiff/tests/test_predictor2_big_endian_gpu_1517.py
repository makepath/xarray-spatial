"""Regression tests for issue #1517.

PR #1515 fixed the ``AttributeError: 'ndarray' object has no attribute
'byteswap'`` crash on big-endian multi-byte TIFFs read via
``read_geotiff_gpu``. After that fix the GPU path no longer raised, but
predictor=2 BE files came back with wrong values: the per-dtype
predictor kernels view the byte buffer as native unsigned integers, so
on a BE file the prefix-sum runs on the wrong integer interpretation
and the differencing produces garbage.

These tests confirm the GPU output now matches the CPU
``read_to_array`` baseline for predictor=2 BE files across several
dtypes and tile layouts, and that the LE predictor=2 path still
round-trips.
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


def _block_cpu_fallback(monkeypatch):
    """Make any call to ``read_to_array`` from ``read_geotiff_gpu`` fail loudly.

    ``read_geotiff_gpu`` returns a cupy-backed array even when its silent CPU
    fallback fires (the fallback wraps the CPU result with ``cupy.asarray``),
    so ``isinstance(gpu_da.data, cupy.ndarray)`` cannot distinguish the two
    paths. ``read_geotiff_gpu`` lives in ``xrspatial.geotiff._backends.gpu``
    and calls the locally bound ``_read_to_array`` symbol there; patching
    that binding to raise turns any silent fallback into a hard test failure,
    which is what we want when the point of a test is to exercise the actual
    GPU decode kernels.

    Tests that legitimately rely on the CPU fallback (e.g. stripped
    layouts) must not call this helper.
    """
    from xrspatial.geotiff._backends import gpu as gpu_backend

    def _no_fallback(*args, **kwargs):
        raise AssertionError(
            "read_geotiff_gpu fell back to read_to_array; "
            "the GPU decode path was not exercised."
        )

    monkeypatch.setattr(
        gpu_backend, '_read_to_array', _no_fallback, raising=True,
    )


@_gpu_only
def test_gpu_predictor2_big_endian_int32_tiled_reproducer(tmp_path, monkeypatch):
    """Exact reproducer from issue #1517: BE int32 tiled deflate + pred=2."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260507)
    arr = rng.randint(
        -1_000_000, 1_000_000, size=(32, 48), dtype=np.int64
    ).astype(np.int32)

    path = tmp_path / "be_pred2_int32.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=2,
        compression="deflate", tile=(16, 16),
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    _block_cpu_fallback(monkeypatch)
    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.dtype(np.int32)
    assert gpu_da.data.dtype.isnative
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


@_gpu_only
@pytest.mark.parametrize(
    "dtype",
    [np.uint16, np.int16, np.uint32, np.int32],
)
def test_gpu_predictor2_big_endian_dtypes_tiled(tmp_path, monkeypatch, dtype):
    """BE predictor=2 tiled files match CPU baseline across dtypes."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260508)
    info = np.iinfo(dtype)
    arr = rng.randint(
        max(info.min, -1_000_000),
        min(info.max, 1_000_000),
        size=(32, 48),
        dtype=np.int64,
    ).astype(dtype)

    path = tmp_path / f"be_pred2_{np.dtype(dtype).name}.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=2,
        compression="deflate", tile=(16, 16),
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    _block_cpu_fallback(monkeypatch)
    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.dtype(dtype)
    assert gpu_da.data.dtype.isnative
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


@_gpu_only
def test_gpu_predictor2_big_endian_stripped_uint16(tmp_path):
    """Stripped BE predictor=2 files take the CPU fallback but stay correct.

    ``read_geotiff_gpu`` falls back to the CPU reader for stripped
    layouts, then transfers the result to GPU. The fix must not regress
    that path.
    """
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260509)
    arr = rng.randint(0, 60000, size=(32, 48), dtype=np.uint16)

    path = tmp_path / "be_pred2_uint16_strip.tif"
    # Omit ``tile`` to get the strip layout.
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=2, compression="deflate",
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.dtype(np.uint16)
    assert gpu_da.data.dtype.isnative
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


@_gpu_only
def test_gpu_predictor2_little_endian_still_works(tmp_path, monkeypatch):
    """LE predictor=2 must still round-trip after the BE fix."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260510)
    arr = rng.randint(
        -1_000_000, 1_000_000, size=(32, 48), dtype=np.int64
    ).astype(np.int32)

    path = tmp_path / "le_pred2_int32.tif"
    tifffile.imwrite(
        str(path), arr, byteorder="<", predictor=2,
        compression="deflate", tile=(16, 16),
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    _block_cpu_fallback(monkeypatch)
    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


@_gpu_only
def test_gpu_predictor3_big_endian_still_works(tmp_path, monkeypatch):
    """Floating-point predictor BE must still match CPU after the fix."""
    import cupy
    import tifffile

    from xrspatial.geotiff import read_geotiff_gpu
    from xrspatial.geotiff._reader import read_to_array

    rng = np.random.RandomState(20260511)
    arr = rng.standard_normal((32, 48)).astype(np.float32)

    path = tmp_path / "be_pred3_float32.tif"
    tifffile.imwrite(
        str(path), arr, byteorder=">", predictor=3,
        compression="deflate", tile=(16, 16),
    )

    cpu, _ = read_to_array(str(path))
    np.testing.assert_array_equal(cpu, arr)

    _block_cpu_fallback(monkeypatch)
    gpu_da = read_geotiff_gpu(str(path))
    assert isinstance(gpu_da.data, cupy.ndarray)
    assert gpu_da.data.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(gpu_da.data.get(), cpu)


def test_swap_byte_lanes_numpy_bps2():
    """The byte-swap helper reverses bytes per sample on a numpy buffer."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    # uint16 values 0x0102, 0x0304 in BE bytes: 01 02 03 04
    buf = np.array([0x01, 0x02, 0x03, 0x04], dtype=np.uint8)
    _swap_byte_lanes(buf, 2)
    np.testing.assert_array_equal(buf, np.array([0x02, 0x01, 0x04, 0x03],
                                                dtype=np.uint8))


def test_swap_byte_lanes_numpy_bps4():
    """bps=4: full byte reversal within each 4-byte sample."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.array([0x01, 0x02, 0x03, 0x04,
                    0x05, 0x06, 0x07, 0x08], dtype=np.uint8)
    _swap_byte_lanes(buf, 4)
    np.testing.assert_array_equal(
        buf, np.array([0x04, 0x03, 0x02, 0x01,
                       0x08, 0x07, 0x06, 0x05], dtype=np.uint8))


def test_swap_byte_lanes_numpy_bps8():
    """bps=8: full byte reversal within each 8-byte sample."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    sample = np.arange(1, 9, dtype=np.uint8)
    buf = np.tile(sample, 2).copy()
    _swap_byte_lanes(buf, 8)
    np.testing.assert_array_equal(
        buf, np.tile(sample[::-1], 2))


def test_swap_byte_lanes_uint8_noop():
    """bps=1 must be a no-op."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.array([1, 2, 3], dtype=np.uint8)
    _swap_byte_lanes(buf, 1)
    np.testing.assert_array_equal(buf, np.array([1, 2, 3], dtype=np.uint8))


def test_swap_byte_lanes_rejects_unsupported_bps():
    """Unsupported bps values raise ValueError rather than corrupt data."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.zeros(6, dtype=np.uint8)
    with pytest.raises(ValueError, match="unsupported bps"):
        _swap_byte_lanes(buf, 3)


def test_swap_byte_lanes_rejects_misaligned_size():
    """Buffer size must be a multiple of bps."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.zeros(5, dtype=np.uint8)
    with pytest.raises(ValueError, match="not a multiple"):
        _swap_byte_lanes(buf, 2)


def test_swap_byte_lanes_numpy_is_zero_temp():
    """The numpy path must mutate the original buffer without realloc."""
    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    buf = np.array([0x01, 0x02, 0x03, 0x04], dtype=np.uint8)
    addr_before = buf.ctypes.data
    _swap_byte_lanes(buf, 2)
    assert buf.ctypes.data == addr_before
    np.testing.assert_array_equal(buf, np.array([0x02, 0x01, 0x04, 0x03],
                                                dtype=np.uint8))


@_gpu_only
@pytest.mark.parametrize("bps,dtype", [
    (2, np.uint16),
    (4, np.uint32),
    (8, np.uint64),
])
def test_swap_byte_lanes_cupy_kernel(bps, dtype):
    """The cupy path runs the CUDA kernel and matches numpy.byteswap."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    rng = np.random.RandomState(20260512 + bps)
    n_samples = 1024
    src = rng.randint(0, np.iinfo(dtype).max, size=n_samples,
                      dtype=np.uint64).astype(dtype)
    expected = src.byteswap()  # numpy reference, returns swapped copy

    d_buf = cupy.asarray(src.view(np.uint8))
    addr_before = int(d_buf.data.ptr)
    _swap_byte_lanes(d_buf, bps)
    addr_after = int(d_buf.data.ptr)

    assert addr_after == addr_before, "kernel must operate in place"
    np.testing.assert_array_equal(
        d_buf.get().view(dtype), expected,
    )


@_gpu_only
def test_swap_byte_lanes_cupy_uint8_noop():
    """bps=1 leaves cupy buffers untouched (no kernel launch)."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _swap_byte_lanes

    src = np.arange(16, dtype=np.uint8)
    d_buf = cupy.asarray(src)
    _swap_byte_lanes(d_buf, 1)
    np.testing.assert_array_equal(d_buf.get(), src)
