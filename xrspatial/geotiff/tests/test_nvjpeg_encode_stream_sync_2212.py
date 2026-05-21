"""Tests for the nvJPEG / nvJPEG2000 encoder default-stream-sync fix (#2212).

The fix replaces ``cupy.cuda.Device().synchronize()`` inside the per-tile
encode loops in ``_nvjpeg_batch_encode`` and ``_nvjpeg2k_batch_encode``
with ``cupy.cuda.Stream.null.synchronize()``. ``Device().synchronize()``
is a whole-device fence that blocks every CUDA stream; the encode /
retrieve sequence only depends on the default stream the calls were
issued on, so scoping the sync to the null stream lets concurrent work
on other streams continue.

The matching decoder ``_try_nvjpeg_batch_decode`` already uses
``cupy.cuda.Stream.null.synchronize()`` (the pattern was inconsistent
before this fix); the nvJPEG2000 decode-side regression was already
caught by #2107.

These tests skip the end-to-end encode path because nvJPEG / nvJPEG2000
shared libraries are rarely installed on developer hosts; the
structural AST checks run regardless and catch a future regression that
re-introduces the device-wide sync.
"""
from __future__ import annotations

import ast
import inspect


def _function_source(func):
    src = inspect.getsource(func)
    start_line = func.__code__.co_firstlineno
    return src, start_line


def _parent_map(tree: ast.AST) -> dict:
    mapping: dict = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            mapping[id(child)] = parent
    return mapping


def _inside_for_loop(node: ast.AST, parents: dict) -> bool:
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.For):
            return True
        cur = parents.get(id(cur))
    return False


def _device_synchronize_lines(tree: ast.AST, start_line: int,
                              parents: dict, *, only_in_loop: bool):
    """Return file line numbers of ``cupy.cuda.Device().synchronize()`` calls."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != 'synchronize':
            continue
        parent_call = func.value
        if not isinstance(parent_call, ast.Call):
            continue
        if not isinstance(parent_call.func, ast.Attribute):
            continue
        if parent_call.func.attr != 'Device':
            continue
        if only_in_loop and not _inside_for_loop(node, parents):
            continue
        if not only_in_loop and _inside_for_loop(node, parents):
            continue
        out.append(start_line + node.lineno - 1)
    return out


def _stream_null_synchronize_lines(tree: ast.AST, start_line: int,
                                   parents: dict, *, only_in_loop: bool):
    """Return file lines of ``cupy.cuda.Stream.null.synchronize()`` calls."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != 'synchronize':
            continue
        # Stream.null is an Attribute chain, not a Call -- the value is
        # ``cupy.cuda.Stream.null``.
        chain = func.value
        if isinstance(chain, ast.Call):
            continue
        if not isinstance(chain, ast.Attribute):
            continue
        # Walk back to find ``Stream`` in the chain.
        found_stream_null = False
        cur = chain
        if cur.attr == 'null':
            inner = cur.value
            if isinstance(inner, ast.Attribute) and inner.attr == 'Stream':
                found_stream_null = True
        if not found_stream_null:
            continue
        if only_in_loop and not _inside_for_loop(node, parents):
            continue
        if not only_in_loop and _inside_for_loop(node, parents):
            continue
        out.append(start_line + node.lineno - 1)
    return out


class TestNvjpegEncodeStreamSync:
    """Structural assertions on the encoder sync fix (no GPU required)."""

    def setup_method(self):
        from xrspatial.geotiff import _gpu_decode
        self._fn = _gpu_decode._nvjpeg_batch_encode
        src, start = _function_source(self._fn)
        self._src = src
        self._start_line = start
        self._tree = ast.parse(src)
        self._parents = _parent_map(self._tree)

    def test_no_device_synchronize_inside_encode_loop(self):
        offending = _device_synchronize_lines(
            self._tree, self._start_line, self._parents, only_in_loop=True,
        )
        assert offending == [], (
            "_nvjpeg_batch_encode contains cupy.cuda.Device().synchronize() "
            f"calls inside a for-loop at file lines {offending}. The fix "
            "in #2212 scopes the per-tile sync to the default stream via "
            "cupy.cuda.Stream.null.synchronize() so concurrent CUDA work "
            "on other streams is not serialised behind every tile encode."
        )

    def test_stream_null_synchronize_present(self):
        """At least one ``Stream.null.synchronize()`` call must be present."""
        found = _stream_null_synchronize_lines(
            self._tree, self._start_line, self._parents, only_in_loop=True,
        )
        assert len(found) >= 1, (
            "_nvjpeg_batch_encode no longer calls "
            "cupy.cuda.Stream.null.synchronize() inside the encode loop. "
            "The fix in #2212 requires the per-tile sync to be scoped to "
            "the default stream so encode/retrieve ordering is preserved "
            "without blocking other CUDA streams."
        )


class TestNvjpeg2kEncodeStreamSync:
    """Structural assertions on the nvJPEG2000 encoder sync fix."""

    def setup_method(self):
        from xrspatial.geotiff import _gpu_decode
        self._fn = _gpu_decode._nvjpeg2k_batch_encode
        src, start = _function_source(self._fn)
        self._src = src
        self._start_line = start
        self._tree = ast.parse(src)
        self._parents = _parent_map(self._tree)

    def test_no_device_synchronize_inside_encode_loop(self):
        offending = _device_synchronize_lines(
            self._tree, self._start_line, self._parents, only_in_loop=True,
        )
        assert offending == [], (
            "_nvjpeg2k_batch_encode contains "
            "cupy.cuda.Device().synchronize() calls inside a for-loop at "
            f"file lines {offending}. The fix in #2212 scopes the per-tile "
            "sync to the default stream via "
            "cupy.cuda.Stream.null.synchronize()."
        )

    def test_stream_null_synchronize_present(self):
        found = _stream_null_synchronize_lines(
            self._tree, self._start_line, self._parents, only_in_loop=True,
        )
        assert len(found) >= 1, (
            "_nvjpeg2k_batch_encode no longer calls "
            "cupy.cuda.Stream.null.synchronize() inside the encode loop. "
            "The fix in #2212 requires the per-tile sync to be scoped to "
            "the default stream so encode/retrieve ordering is preserved "
            "without blocking other CUDA streams."
        )


class TestDecodeReferencePattern:
    """The decoder pattern is the contract we mirror. Pin it as the reference.

    If ``_try_nvjpeg_batch_decode`` ever swaps back to
    ``Device().synchronize()`` inside its loop, the encoder fix would
    drift from the codebase's own established pattern; pin it.
    """

    def setup_method(self):
        from xrspatial.geotiff import _gpu_decode
        self._fn = _gpu_decode._try_nvjpeg_batch_decode
        src, start = _function_source(self._fn)
        self._src = src
        self._start_line = start
        self._tree = ast.parse(src)
        self._parents = _parent_map(self._tree)

    def test_decoder_uses_stream_null_sync_in_loop(self):
        found = _stream_null_synchronize_lines(
            self._tree, self._start_line, self._parents, only_in_loop=True,
        )
        assert len(found) >= 1, (
            "_try_nvjpeg_batch_decode no longer uses "
            "cupy.cuda.Stream.null.synchronize() inside the decode loop. "
            "This is the pattern #2212 mirrors for the encoder side; "
            "drifting away from it means both sides will need to be "
            "re-aligned."
        )
