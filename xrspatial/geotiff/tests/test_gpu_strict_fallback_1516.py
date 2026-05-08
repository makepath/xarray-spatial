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


def _ensure_cupy_stub():
    """Install a numpy-backed ``cupy`` shim if cupy is unavailable.

    The CPU fallback path inside ``read_geotiff_gpu`` calls
    ``cupy.asarray(arr_cpu)`` to upload the CPU result onto the GPU.
    On CPU-only CI we replace cupy with a numpy passthrough so the
    function still returns a DataArray we can assert on.
    """
    if 'cupy' in sys.modules:
        return False  # real cupy already imported, leave it alone
    try:
        import cupy  # noqa: F401
        return False
    except ImportError:
        pass

    stub = types.ModuleType('cupy')
    stub.ndarray = np.ndarray
    stub.asarray = np.asarray

    cuda_mod = types.ModuleType('cupy.cuda')

    def _is_available():
        return False

    cuda_mod.is_available = _is_available
    stub.cuda = cuda_mod

    sys.modules['cupy'] = stub
    sys.modules['cupy.cuda'] = cuda_mod
    return True


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
            sys.modules.pop('cupy', None)
            sys.modules.pop('cupy.cuda', None)
            importlib.invalidate_caches()


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
            sys.modules.pop('cupy', None)
            sys.modules.pop('cupy.cuda', None)
            importlib.invalidate_caches()


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
            sys.modules.pop('cupy', None)
            sys.modules.pop('cupy.cuda', None)
            importlib.invalidate_caches()
