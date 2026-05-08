"""Regression tests for issue #1516.

``read_geotiff_gpu`` previously wrapped the GPU decode in a too-broad
``try/except Exception: pass`` that silently swallowed any failure and
fell through to the CPU path. Real GPU regressions (#1508 was an
``AttributeError``) lived undetected because the user-visible result
was still numerically correct.

The fix:

1. Default ``gpu='auto'`` still falls back to CPU, but emits a
   ``RuntimeWarning`` reporting the original exception type and
   message so failures are visible.
2. New ``gpu='strict'`` mode re-raises instead of falling back, so
   tests and CI for the GPU fast path see real errors.

These tests monkeypatch ``gpu_decode_tiles_from_file`` to raise a
synthetic exception. They do not require a real GPU because we stub
``cupy`` at the ``sys.modules`` level when it is not already
available; ``cupy.asarray`` is only called in the CPU-fallback branch
and is satisfied by a thin numpy-backed shim.
"""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

from .conftest import make_minimal_tiff


_CUPY_ORIG_SENTINEL = object()
_cupy_saved = _CUPY_ORIG_SENTINEL
_cupy_cuda_saved = _CUPY_ORIG_SENTINEL


def _cuda_actually_available() -> bool:
    """Return True only if cupy + CUDA are usable on this host.

    cupy may be importable on a machine without a working CUDA runtime
    (no driver, no device, ROCm-only, etc.). The CPU-fallback branch in
    ``read_geotiff_gpu`` calls ``cupy.asarray`` which would then fail at
    allocation time. Treat that case the same as cupy-not-installed.
    """
    try:
        import cupy
    except ImportError:
        return False
    try:
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


def _ensure_cupy_stub() -> bool:
    """Install a numpy-backed ``cupy`` shim if real cupy isn't usable.

    Replaces ``sys.modules['cupy']`` whenever cupy is missing OR cupy is
    installed but CUDA isn't available. The original module (if any) is
    saved so :func:`_restore_cupy` can put it back.
    """
    global _cupy_saved, _cupy_cuda_saved

    if _cuda_actually_available():
        return False

    _cupy_saved = sys.modules.get('cupy', _CUPY_ORIG_SENTINEL)
    _cupy_cuda_saved = sys.modules.get('cupy.cuda', _CUPY_ORIG_SENTINEL)

    stub = types.ModuleType('cupy')
    stub.ndarray = np.ndarray
    stub.asarray = np.asarray

    cuda_mod = types.ModuleType('cupy.cuda')
    cuda_mod.is_available = lambda: False
    stub.cuda = cuda_mod

    sys.modules['cupy'] = stub
    sys.modules['cupy.cuda'] = cuda_mod
    return True


def _restore_cupy() -> None:
    """Undo :func:`_ensure_cupy_stub`."""
    global _cupy_saved, _cupy_cuda_saved
    for name, saved in (
        ('cupy', _cupy_saved),
        ('cupy.cuda', _cupy_cuda_saved),
    ):
        if saved is _CUPY_ORIG_SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved
    _cupy_saved = _CUPY_ORIG_SENTINEL
    _cupy_cuda_saved = _CUPY_ORIG_SENTINEL
    importlib.invalidate_caches()


@pytest.fixture
def tiled_tiff_path(tmp_path):
    """A small tiled TIFF on disk that exercises the GPU tile path."""
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    raw = make_minimal_tiff(
        8, 8, np.dtype('float32'),
        pixel_data=data,
        tiled=True,
        tile_size=4,
    )
    path = tmp_path / "strict_fallback_1516.tif"
    path.write_bytes(raw)
    return str(path), data


def _patch_gpu_decode_to_raise(monkeypatch, exc):
    """Replace ``gpu_decode_tiles_from_file`` with one that raises ``exc``."""
    from xrspatial.geotiff import _gpu_decode

    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles_from_file', _boom, raising=True,
    )


def _patch_both_gpu_stages_to_raise(monkeypatch, exc):
    """Make both GPU decode stages raise ``exc`` to exercise the second handler."""
    from xrspatial.geotiff import _gpu_decode

    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles_from_file', _boom, raising=True,
    )
    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles', _boom, raising=True,
    )


def test_default_mode_warns_on_gpu_failure(tiled_tiff_path, monkeypatch):
    """Default ``gpu='auto'`` warns and falls back to the CPU result."""
    inserted_stub = _ensure_cupy_stub()
    try:
        from xrspatial.geotiff import read_geotiff_gpu

        path, expected = tiled_tiff_path

        synthetic = RuntimeError("simulated GPU failure")
        _patch_gpu_decode_to_raise(monkeypatch, synthetic)

        with pytest.warns(RuntimeWarning, match="GPU decode failed"):
            result = read_geotiff_gpu(path)

        # Fallback returned the CPU-decoded data. Real cupy arrays expose
        # ``.get()`` to copy back to host; the numpy stub returns a
        # plain ndarray.
        out = result.data
        if hasattr(out, 'get'):
            out = out.get()
        np.testing.assert_array_equal(np.asarray(out), expected)
    finally:
        if inserted_stub:
            _restore_cupy()


def test_strict_mode_reraises(tiled_tiff_path, monkeypatch):
    """``gpu='strict'`` re-raises the original GPU exception."""
    inserted_stub = _ensure_cupy_stub()
    try:
        from xrspatial.geotiff import read_geotiff_gpu

        path, _ = tiled_tiff_path

        synthetic = RuntimeError("simulated GPU failure")
        _patch_gpu_decode_to_raise(monkeypatch, synthetic)

        with pytest.raises(RuntimeError, match="simulated GPU failure"):
            read_geotiff_gpu(path, gpu='strict')
    finally:
        if inserted_stub:
            _restore_cupy()


def test_strict_mode_reraises_second_stage(tiled_tiff_path, monkeypatch):
    """``gpu='strict'`` re-raises if the second-stage GPU decode fails too.

    Regression for the case where ``gpu_decode_tiles_from_file`` and the
    follow-up ``gpu_decode_tiles`` both fail. Previously the second
    failure was caught by an unconditional ``except (ValueError, Exception)``
    that fell back to CPU regardless of mode.
    """
    inserted_stub = _ensure_cupy_stub()
    try:
        from xrspatial.geotiff import read_geotiff_gpu

        path, _ = tiled_tiff_path

        synthetic = RuntimeError("simulated second-stage GPU failure")
        _patch_both_gpu_stages_to_raise(monkeypatch, synthetic)

        with pytest.raises(RuntimeError,
                           match="simulated second-stage GPU failure"):
            read_geotiff_gpu(path, gpu='strict')
    finally:
        if inserted_stub:
            _restore_cupy()


def test_default_mode_warns_on_second_stage_failure(tiled_tiff_path, monkeypatch):
    """``gpu='auto'`` warns once per stage failure and falls back to CPU."""
    inserted_stub = _ensure_cupy_stub()
    try:
        from xrspatial.geotiff import read_geotiff_gpu

        path, expected = tiled_tiff_path

        synthetic = RuntimeError("simulated second-stage GPU failure")
        _patch_both_gpu_stages_to_raise(monkeypatch, synthetic)

        with pytest.warns(RuntimeWarning, match="GPU decode failed"):
            result = read_geotiff_gpu(path)

        out = result.data
        if hasattr(out, 'get'):
            out = out.get()
        np.testing.assert_array_equal(np.asarray(out), expected)
    finally:
        if inserted_stub:
            _restore_cupy()


def test_invalid_gpu_kwarg_rejected(tiled_tiff_path):
    """An unknown ``gpu=`` value raises ``ValueError`` with a clear message."""
    inserted_stub = _ensure_cupy_stub()
    try:
        from xrspatial.geotiff import read_geotiff_gpu

        path, _ = tiled_tiff_path

        with pytest.raises(ValueError, match="gpu must be 'auto' or 'strict'"):
            read_geotiff_gpu(path, gpu='loose')
    finally:
        if inserted_stub:
            _restore_cupy()
