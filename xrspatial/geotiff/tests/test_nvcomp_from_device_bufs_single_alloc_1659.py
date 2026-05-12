"""Regression tests for the single-buffer pattern in _try_nvcomp_from_device_bufs.

Issue #1659: ``_try_nvcomp_from_device_bufs`` used to allocate N separate
``cupy.empty(tile_bytes)`` output buffers and run ``cupy.concatenate`` after
the nvCOMP decompress kernel returned. That kept two copies of the
decompressed data alive at once and ran a serial concat the other nvCOMP
paths in this module already avoid. The fix matches the single-contiguous-
buffer + pointer-offset pattern used by the deflate / LZW / host-buffer
paths nearby.

These tests skip when CuPy + CUDA are not available. They also skip the
end-to-end nvCOMP integration check when ``kvikio`` or the nvCOMP shared
library are not installed, which is the common case on developer hosts;
the unit-level checks (contract + memory guard) run regardless.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff._gpu_decode import _try_nvcomp_from_device_bufs


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


def _nvcomp_available() -> bool:
    from xrspatial.geotiff._gpu_decode import _get_nvcomp
    return _get_nvcomp() is not None


@pytest.mark.skipif(not _gpu_available(), reason="cupy + CUDA required")
def test_unsupported_codec_short_circuits_before_allocation():
    """Non-ZSTD codecs must return None without allocating output buffers.

    Pins the early-return contract that lets the caller pick a different
    decoder when nvCOMP cannot handle this codec.
    """
    import cupy

    # Use Deflate (8), which is unsupported by this function (ZSTD only).
    d_tiles = [cupy.zeros(1024, dtype=cupy.uint8) for _ in range(4)]
    assert _try_nvcomp_from_device_bufs(d_tiles, 1024, 8) is None


@pytest.mark.skipif(not _gpu_available(), reason="cupy + CUDA required")
def test_no_nvcomp_lib_returns_none(monkeypatch):
    """When the nvCOMP library is missing, the function must return None.

    The caller relies on this signal to fall back to the bytes-based decode
    path. Without it, callers would hit a ctypes ``getattr`` AttributeError
    deeper in the function.
    """
    import cupy
    from xrspatial.geotiff import _gpu_decode

    monkeypatch.setattr(_gpu_decode, "_get_nvcomp", lambda: None)

    d_tiles = [cupy.zeros(1024, dtype=cupy.uint8)]
    assert _try_nvcomp_from_device_bufs(d_tiles, 1024, 50000) is None


@pytest.mark.skipif(not _gpu_available(), reason="cupy + CUDA required")
def test_memory_guard_runs_with_full_decomp_size(monkeypatch):
    """The single-buffer allocation must be size-checked before cupy.empty.

    The new pattern allocates one contiguous ``n * tile_bytes`` buffer
    instead of N small buffers. The OOM guard is what tells the caller
    early that the decode will not fit on the device; a regression that
    removed the guard would surface as an opaque CUDA OOM instead.
    """
    import cupy
    from xrspatial.geotiff import _gpu_decode

    seen = {"total_bytes": None, "what": None, "called": False}

    def fake_check(required_bytes, what="tile buffer"):
        seen["total_bytes"] = int(required_bytes)
        seen["what"] = what
        seen["called"] = True
        raise MemoryError("simulated OOM")

    # Pin _get_nvcomp to something truthy so the function does not bail
    # before reaching the allocation step. The fake check raises before
    # any nvCOMP call would happen, so the lib value never gets used.
    monkeypatch.setattr(_gpu_decode, "_get_nvcomp", lambda: object())
    monkeypatch.setattr(_gpu_decode, "_check_gpu_memory", fake_check)

    n_tiles = 8
    tile_bytes = 65536
    d_tiles = [cupy.zeros(128, dtype=cupy.uint8) for _ in range(n_tiles)]

    with pytest.raises(MemoryError):
        _try_nvcomp_from_device_bufs(d_tiles, tile_bytes, 50000)

    assert seen["called"], "_check_gpu_memory was not called"
    expected_bytes = n_tiles * tile_bytes
    assert seen["total_bytes"] == expected_bytes, (
        f"expected total {expected_bytes}, got {seen['total_bytes']}"
    )
    assert "decompressed" in seen["what"] or "nvCOMP" in seen["what"], (
        f"unhelpful 'what' label: {seen['what']!r}"
    )


@pytest.mark.skipif(
    not _gpu_available() or not _nvcomp_available(),
    reason="cupy + CUDA + nvCOMP shared lib required",
)
def test_zstd_decompress_roundtrip_returns_single_contiguous_buffer():
    """End-to-end: feed real ZSTD-compressed device buffers in, check the
    output is a single flat ``cupy.uint8`` array of length n*tile_bytes.

    This test confirms the return contract that ``_apply_predictor_and_assemble``
    depends on: ``out`` is the contiguous concatenation of the N decompressed
    tiles, not a list. The previous implementation returned the same shape but
    via ``cupy.concatenate``; the new one allocates the contig buffer up front
    and writes through per-tile pointers, so a regression that dropped the
    return value would surface here.
    """
    import cupy
    import zstandard as zstd

    rng = np.random.default_rng(seed=1659)
    tile_bytes = 4096
    n_tiles = 8

    cctx = zstd.ZstdCompressor()
    host_tiles = [rng.integers(0, 256, size=tile_bytes, dtype=np.uint8)
                  for _ in range(n_tiles)]
    compressed = [cctx.compress(t.tobytes()) for t in host_tiles]
    d_tiles = [cupy.asarray(np.frombuffer(c, dtype=np.uint8))
               for c in compressed]

    result = _try_nvcomp_from_device_bufs(d_tiles, tile_bytes, 50000)

    # nvCOMP may be present but mis-configured on the host (e.g. driver
    # version mismatch); skip rather than fail in that case so the test is
    # informative when run on a real GDS rig.
    if result is None:
        pytest.skip("nvCOMP returned None; library may be unusable on this host")

    assert isinstance(result, cupy.ndarray)
    assert result.dtype == cupy.uint8
    assert result.shape == (n_tiles * tile_bytes,)
    assert result.flags.c_contiguous

    # Decoded payload must match the original host tiles. The buffer is a
    # single flat array; tile i lives at offset i*tile_bytes.
    host_out = result.get()
    for i, expected in enumerate(host_tiles):
        decoded = host_out[i * tile_bytes:(i + 1) * tile_bytes]
        assert np.array_equal(decoded, expected), (
            f"tile {i} decoded payload differs from input"
        )


@pytest.mark.skipif(not _gpu_available(), reason="cupy + CUDA required")
def test_no_orphan_decomp_buffers_after_call(monkeypatch):
    """Earlier code held a Python list of N device buffers in scope
    alongside the concatenated result. The replacement allocates once
    and returns that one buffer.

    The check here is structural rather than numerical: after a successful
    call the only cupy ndarray the caller receives is ``result`` itself,
    and inspecting it confirms ``result.size == n_tiles * tile_bytes``.
    """
    import cupy
    from xrspatial.geotiff import _gpu_decode

    # Stub the nvCOMP entry points so the decompress "succeeds" without an
    # actual library. Force the function down the success branch, capture
    # the returned buffer, then verify shape + ownership.
    monkeypatch.setattr(_gpu_decode, "_get_nvcomp",
                        lambda: _FakeNvcompLib())

    n_tiles = 4
    tile_bytes = 2048
    d_tiles = [cupy.zeros(64, dtype=cupy.uint8) for _ in range(n_tiles)]
    result = _try_nvcomp_from_device_bufs(d_tiles, tile_bytes, 50000)

    # The fake lib reports success and zero-fills the output buffer; the
    # function returns the contiguous buffer as-is.
    assert result is not None
    assert isinstance(result, cupy.ndarray)
    assert result.size == n_tiles * tile_bytes
    assert result.flags.c_contiguous
    # The contract requires uint8 -- not a uint8 view of something else.
    assert result.dtype == cupy.uint8


class _FakeNvcompLib:
    """Stand-in for the nvCOMP CDLL handle used in tests.

    The real function calls ``getattr(lib, fn_name)`` for two entry points
    and invokes each as a ctypes function. We expose those entry-point
    names as Python callables that pretend the work succeeded.
    """

    def __getattr__(self, name):
        if name == 'nvcompBatchedZstdDecompressGetTempSizeAsync':
            return _fake_temp_size_fn
        if name == 'nvcompBatchedZstdDecompressAsync':
            return _fake_decompress_fn
        raise AttributeError(name)


def _fake_temp_size_fn(n, tile_bytes, opts, p_temp_size, total):
    """Stub for nvcompBatchedZstdDecompressGetTempSizeAsync."""
    # Write a tiny temp-size value into the caller's c_size_t.
    p_temp_size._obj.value = 1
    return 0


def _fake_decompress_fn(*args):
    """Stub for nvcompBatchedZstdDecompressAsync.

    The function's return-value test is ``s != 0``. We return 0 (success).
    The caller's d_statuses array is already zero from ``cupy.zeros``, so
    the post-decode any-nonzero check passes.
    """
    return 0
