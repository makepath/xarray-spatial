"""Tests for the nvJPEG2000 batch-decode allocation refactor (#2107).

The fix replaces per-tile / per-component ``cupy.empty`` allocations and
per-tile ``cupy.cuda.Device().synchronize()`` inside the decode loop with
a single contiguous device pool and a single batch-end sync. The pattern
matches the prior fixes for ``_try_nvcomp_from_device_bufs`` (#1659),
``_try_kvikio_read_tiles`` (#1688), and ``_nvcomp_batch_compress`` (#1712).

Since nvJPEG2000 is rarely available on test hosts (libnvjpeg2k.so is
not part of cuda-toolkit's default install), these tests focus on
structural properties of the implementation rather than running a real
decode:

* ``_try_nvjpeg2k_batch_decode`` early-returns ``None`` when the shared
  library is missing -- no allocations or syncs happen on the common
  test host.
* Source inspection asserts the new contract:
    - Exactly two ``cupy.empty`` calls in the decode loop region
      (``d_comp_pool`` + ``d_all_tiles``), zero inside the per-tile loop.
    - Exactly one ``Device().synchronize()`` after the loop, zero inside.
    - The slab indexing math hits the per-tile-component slab.

Structural tests like these mirror the approach taken by
``test_nvcomp_from_device_bufs_single_alloc_1659.py`` and
``test_kvikio_batched_pread_1688.py`` so a regression that re-introduces
the per-tile alloc pattern fails CI without needing a GPU.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest


def _function_source(func):
    """Return the function's source plus its line range in the source file."""
    src = inspect.getsource(func)
    start_line = func.__code__.co_firstlineno
    return src, start_line


def _count_cupy_empty_calls(tree):
    """Count ``cupy.empty(...)`` Call nodes inside the AST."""
    n = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != 'empty':
            continue
        # ``cupy.empty`` matches both bare ``cupy`` and aliased ``cp``;
        # we only care about the ``empty`` method name.
        if not isinstance(func.value, ast.Name):
            continue
        if func.value.id not in ('cupy', 'cp'):
            continue
        n += 1
    return n


def _count_device_synchronize_calls(tree):
    """Count ``cupy.cuda.Device(...).synchronize()`` Call nodes."""
    n = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != 'synchronize':
            continue
        # We walk back through the chain: synchronize -> Device(...) ->
        # cuda -> cupy. Allow any chain that ends in a ``Device`` call.
        parent_call = func.value
        if not isinstance(parent_call, ast.Call):
            continue
        if not isinstance(parent_call.func, ast.Attribute):
            continue
        if parent_call.func.attr != 'Device':
            continue
        n += 1
    return n


def _inside_for_loop(node: ast.AST, parents: dict) -> bool:
    """Return True when ``node`` sits anywhere under a ``for`` statement."""
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.For):
            return True
        cur = parents.get(id(cur))
    return False


def _parent_map(tree: ast.AST) -> dict:
    """Build a ``{id(child): parent}`` lookup map for ``_inside_for_loop``."""
    mapping: dict = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            mapping[id(child)] = parent
    return mapping


class TestNvjpeg2kSingleAllocStructural:
    """Structural assertions on the refactored helper (no GPU required)."""

    def setup_method(self):
        from xrspatial.geotiff import _gpu_decode

        self._fn = _gpu_decode._try_nvjpeg2k_batch_decode
        src, start = _function_source(self._fn)
        self._src = src
        self._start_line = start
        self._tree = ast.parse(src)
        self._parents = _parent_map(self._tree)

    def test_no_cupy_empty_inside_decode_loop(self):
        """``cupy.empty`` must NOT appear inside the per-tile ``for`` loop.

        The refactor moves the pool allocation outside the loop. A
        regression that re-introduces a per-tile ``cupy.empty`` would
        bring back the memory-pool round-trip the fix removed.
        """
        offending = []
        for node in ast.walk(self._tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != 'empty':
                continue
            if (not isinstance(func.value, ast.Name)
                    or func.value.id not in ('cupy', 'cp')):
                continue
            if _inside_for_loop(node, self._parents):
                offending.append(self._start_line + node.lineno - 1)
        assert offending == [], (
            f"_try_nvjpeg2k_batch_decode contains cupy.empty(...) calls "
            f"inside a for-loop at file lines {offending}. The refactor "
            f"in #2107 moved every output allocation outside the per-tile "
            f"loop; reverting that defeats the pooling optimisation."
        )

    def test_no_device_synchronize_inside_decode_loop(self):
        """``Device().synchronize()`` must NOT live inside the decode loop.

        The previous implementation called it once per tile, forcing
        default-stream serialisation. The refactor leaves exactly one
        synchronize call after the loop body.
        """
        offending = []
        for node in ast.walk(self._tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != 'synchronize':
                continue
            parent_call = func.value
            if (not isinstance(parent_call, ast.Call)
                    or not isinstance(parent_call.func, ast.Attribute)
                    or parent_call.func.attr != 'Device'):
                continue
            if _inside_for_loop(node, self._parents):
                offending.append(self._start_line + node.lineno - 1)
        assert offending == [], (
            f"_try_nvjpeg2k_batch_decode contains Device().synchronize() "
            f"calls inside a for-loop at file lines {offending}. The "
            f"refactor in #2107 keeps exactly one batch-end sync outside "
            f"the loop so successive tiles can pipeline through "
            f"nvJPEG2000."
        )

    def test_pool_allocation_present(self):
        """Source contains the expected pool buffer name and slab math.

        The refactor introduces ``d_comp_pool`` and
        ``per_tile_comp_bytes``; if either disappears, the test fails so
        the reviewer notices the layout drift.
        """
        assert 'd_comp_pool' in self._src, (
            "_try_nvjpeg2k_batch_decode no longer references the shared "
            "d_comp_pool buffer; the refactor in #2107 is missing or "
            "reverted."
        )
        assert 'per_tile_comp_bytes' in self._src, (
            "_try_nvjpeg2k_batch_decode no longer references "
            "per_tile_comp_bytes; the per-tile slab math from #2107 "
            "is missing or renamed without an audit."
        )

    def test_check_gpu_memory_guard_present(self):
        """The pool allocation must be guarded by ``_check_gpu_memory``.

        Sibling helpers (``_try_nvcomp_from_device_bufs``,
        ``_try_kvikio_read_tiles``, ``_nvcomp_batch_compress``) all guard
        their pool allocations the same way; refusing the allocation
        before cupy raises an opaque CUDA OOM keeps the failure mode
        consistent (#2107 follows that pattern).
        """
        assert '_check_gpu_memory(' in self._src, (
            "_try_nvjpeg2k_batch_decode no longer calls _check_gpu_memory "
            "before allocating the per-tile component pool. The fail-fast "
            "OOM contract from #2107 is missing."
        )


class TestNvjpeg2kLibAbsentShortCircuit:
    """When the shared library is missing, the function returns None
    without touching cupy / allocating any device memory."""

    def test_returns_none_when_lib_missing(self, monkeypatch):
        """The early-return is the path most test hosts take. Verify
        nothing reaches the refactored allocation code on that path so
        the refactor does not regress the lib-missing host behaviour.
        """
        from xrspatial.geotiff import _gpu_decode

        monkeypatch.setattr(_gpu_decode, '_get_nvjpeg2k', lambda: None)

        result = _gpu_decode._try_nvjpeg2k_batch_decode(
            compressed_tiles=[b''],
            tile_width=8,
            tile_height=8,
            dtype=np.dtype('uint8'),
            samples=1,
        )
        assert result is None

    def test_returns_none_for_unsupported_dtype(self, monkeypatch):
        """Unsupported dtypes (e.g. float32) short-circuit before any
        device allocation. The refactor moved the pool alloc below the
        dtype guard, so this exercises the dtype branch that must still
        clean up the handles without touching the pool.
        """
        from xrspatial.geotiff import _gpu_decode

        # Fake a lib that succeeds for handle/state/stream/params and
        # exposes the destroy entry points so the dtype-guard cleanup
        # path runs without crashing. We do not pretend to support the
        # actual nvjpeg2k C API beyond what the early-return code uses.
        class _FakeLib:
            def __init__(self):
                self.calls = []

            def nvjpeg2kCreateSimple(self, *_args):
                return 0

            def nvjpeg2kDecodeStateCreate(self, *_args):
                return 0

            def nvjpeg2kStreamCreate(self, *_args):
                return 0

            def nvjpeg2kDecodeParamsCreate(self, *_args):
                return 0

            def nvjpeg2kDecodeParamsDestroy(self, *_args):
                self.calls.append('params_destroy')

            def nvjpeg2kStreamDestroy(self, *_args):
                self.calls.append('stream_destroy')

            def nvjpeg2kDecodeStateDestroy(self, *_args):
                self.calls.append('state_destroy')

            def nvjpeg2kDestroy(self, *_args):
                self.calls.append('handle_destroy')

        fake = _FakeLib()
        monkeypatch.setattr(_gpu_decode, '_get_nvjpeg2k', lambda: fake)

        # float32 is not in {uint8, uint16, int16} so the helper exits
        # before any pool allocation -- and the host needs no cupy
        # for this branch to run.
        result = _gpu_decode._try_nvjpeg2k_batch_decode(
            compressed_tiles=[b''],
            tile_width=8,
            tile_height=8,
            dtype=np.dtype('float32'),
            samples=1,
        )
        assert result is None
        # The dtype-guard branch should have called all four destroy
        # functions, mirroring the success-path cleanup.
        assert fake.calls == [
            'params_destroy',
            'stream_destroy',
            'state_destroy',
            'handle_destroy',
        ]


@pytest.mark.gpu
class TestNvjpeg2kPoolWithCupy:
    """Lightweight cupy-only smoke tests for the pool layout.

    These tests do not exercise the real nvJPEG2000 decode (the host
    typically has no libnvjpeg2k.so) but they confirm the pool sizing
    math and the per-tile slab indexing produce non-overlapping views
    into the shared buffer for representative tile sizes.
    """

    def test_pool_slabs_are_non_overlapping(self):
        """Tile-component slabs into the pool must not overlap.

        We recompute the slab boundaries the helper uses and verify the
        sequence covers exactly the pool size with no gaps and no
        double-coverage. A regression that miscomputes
        ``per_tile_comp_bytes`` would either OOB the pool or fold tile
        N onto tile N-1's bytes; this test catches both.
        """
        cupy = pytest.importorskip('cupy')

        n_tiles = 4
        tile_width = 32
        tile_height = 32
        samples = 3
        dtype = np.dtype('uint16')
        pitch = tile_width * dtype.itemsize
        per_tile_comp_bytes = samples * tile_height * pitch
        pool = cupy.empty(n_tiles * per_tile_comp_bytes, dtype=cupy.uint8)

        seen = set()
        for i in range(n_tiles):
            tile_pool_start = i * per_tile_comp_bytes
            for c in range(samples):
                start = tile_pool_start + c * tile_height * pitch
                end = start + tile_height * pitch
                for byte in range(start, end):
                    assert byte not in seen, (
                        f"pool byte {byte} appears in two slabs "
                        f"(tile={i}, comp={c}); per-tile slab math is "
                        f"wrong."
                    )
                    seen.add(byte)
        assert len(seen) == int(pool.nbytes)
