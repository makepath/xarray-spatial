"""Regression tests for batched D2H transfer on the GDS fallback path.

Issue #1552: ``gpu_decode_tiles_from_file`` previously did one ``.get()``
per tile when GDS read succeeded but nvCOMP could not decompress on
device (LZW or non-ZSTD/non-deflate codecs). Each ``.get()`` was an
independent D2H copy on the default stream, so the transfers serialised
and per-DMA setup overhead dominated wall time.

The fix concatenates all device buffers into one cupy array, runs a
single ``.get()``, and slices the host bytes by per-tile offsets,
mirroring the symmetric H2D batched-upload pattern in
``_try_nvcomp_decompress``.

These tests skip when CuPy + a CUDA device are not available.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff._gpu_decode import _batched_d2h_to_bytes


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


def test_batched_d2h_empty_list():
    """Empty input must return an empty list without touching cupy."""
    assert _batched_d2h_to_bytes([]) == []


@pytest.mark.skipif(not _gpu_available(), reason="cupy + CUDA required")
def test_batched_d2h_matches_per_tile_get():
    """Batched D2H output must equal the per-tile ``.get().tobytes()`` baseline."""
    import cupy

    rng = np.random.default_rng(seed=1552)

    # Mix of sizes to exercise variable-length concatenation.
    host_tiles = [
        rng.integers(0, 256, size=n, dtype=np.uint8)
        for n in [1, 7, 64, 4096, 1, 65537, 256]
    ]
    d_tiles = [cupy.asarray(t) for t in host_tiles]

    # Baseline: the old per-tile loop the fix replaces.
    expected = [t.get().tobytes() for t in d_tiles]

    actual = _batched_d2h_to_bytes(d_tiles)

    assert len(actual) == len(expected)
    for i, (got, want) in enumerate(zip(actual, expected)):
        assert isinstance(got, bytes), f"tile {i}: not bytes ({type(got)})"
        assert got == want, f"tile {i} mismatch"


@pytest.mark.skipif(not _gpu_available(), reason="cupy + CUDA required")
def test_batched_d2h_single_tile():
    """Single-tile input is a degenerate case worth covering explicitly."""
    import cupy

    payload = bytes(range(64))
    d_tiles = [cupy.asarray(np.frombuffer(payload, dtype=np.uint8))]
    out = _batched_d2h_to_bytes(d_tiles)

    assert len(out) == 1
    assert out[0] == payload


@pytest.mark.skipif(not _gpu_available(), reason="cupy + CUDA required")
def test_batched_d2h_zero_size_tile_in_list():
    """A zero-sized tile mixed with real tiles must round-trip cleanly."""
    import cupy

    real = np.array([10, 20, 30], dtype=np.uint8)
    empty = np.zeros(0, dtype=np.uint8)
    d_tiles = [cupy.asarray(real), cupy.asarray(empty), cupy.asarray(real[::-1])]

    out = _batched_d2h_to_bytes(d_tiles)

    assert out[0] == real.tobytes()
    assert out[1] == b''
    assert out[2] == real[::-1].tobytes()


@pytest.mark.skipif(not _gpu_available(), reason="cupy + CUDA required")
def test_batched_d2h_checks_gpu_memory_before_concat(monkeypatch):
    """The concat allocates sum(sizes) bytes; the guard must fire before it.

    Pins the OOM-handling contract: ``_check_gpu_memory`` is called with
    the total batch size before ``cupy.concatenate``. A monkeypatch that
    raises from the guard must stop execution before any allocation
    happens.
    """
    import cupy

    from xrspatial.geotiff import _gpu_decode

    seen = {"total_bytes": None, "what": None, "called": False}

    def fake_check(required_bytes, what="tile buffer"):
        seen["total_bytes"] = int(required_bytes)
        seen["what"] = what
        seen["called"] = True
        raise MemoryError("simulated OOM")

    monkeypatch.setattr(_gpu_decode, "_check_gpu_memory", fake_check)

    sizes = [4096, 8192, 1024]
    d_tiles = [cupy.zeros(n, dtype=cupy.uint8) for n in sizes]

    with pytest.raises(MemoryError, match="simulated OOM"):
        _gpu_decode._batched_d2h_to_bytes(d_tiles)

    assert seen["called"], "_check_gpu_memory was not called"
    assert seen["total_bytes"] == sum(sizes), (
        f"expected total {sum(sizes)}, got {seen['total_bytes']}"
    )
    assert "D2H" in seen["what"] or "staging" in seen["what"], (
        f"unhelpful 'what' label: {seen['what']!r}"
    )


@pytest.mark.skipif(not _gpu_available(), reason="cupy + CUDA required")
def test_batched_d2h_many_small_tiles():
    """Many tiles is the regime the batching speedup actually targets."""
    import cupy

    rng = np.random.default_rng(seed=42)
    host_tiles = [rng.integers(0, 256, size=128, dtype=np.uint8) for _ in range(256)]
    d_tiles = [cupy.asarray(t) for t in host_tiles]

    out = _batched_d2h_to_bytes(d_tiles)

    assert len(out) == 256
    for i, (got, src) in enumerate(zip(out, host_tiles)):
        assert got == src.tobytes(), f"tile {i} mismatch"
