"""CUDA preflight in ``read_geotiff_gpu``.

Regression for issue #1903. When CuPy imports but the CUDA driver is
unusable (older driver than the build expects, suspended VM, etc.),
the failure used to surface as ``cudaErrorInsufficientDriver`` from a
``cupy.asarray(...)`` call deep in the CPU-fallback path. The fix
preflights the runtime via ``cupy.cuda.runtime.getDeviceCount()``
right after the cupy import and raises a clean ``RuntimeError``.

These tests stub ``cupy.cuda.runtime.getDeviceCount`` so they exercise
the preflight branch without requiring a real GPU. The function under
test is called directly to skip the file-source setup.
"""
from __future__ import annotations

import importlib.util
import sys
import types

import pytest


_CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None


def _install_cupy_stub(monkeypatch, *, get_device_count):
    """Install a minimal stub ``cupy`` module so the preflight runs.

    Used on machines without cupy installed; lets us exercise the
    preflight failure path on CPU-only CI.
    """
    cupy_mod = types.ModuleType("cupy")
    cuda_mod = types.ModuleType("cupy.cuda")
    runtime_mod = types.ModuleType("cupy.cuda.runtime")
    runtime_mod.getDeviceCount = get_device_count
    cuda_mod.runtime = runtime_mod
    cupy_mod.cuda = cuda_mod
    monkeypatch.setitem(sys.modules, "cupy", cupy_mod)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda_mod)
    monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", runtime_mod)


def test_preflight_raises_on_runtime_error(monkeypatch):
    """A simulated cudaErrorInsufficientDriver becomes a clean RuntimeError."""
    from xrspatial.geotiff._backends import gpu as gpu_mod

    class FakeCudaError(RuntimeError):
        pass

    def _raise(*_a, **_kw):
        raise FakeCudaError("cudaErrorInsufficientDriver")

    _install_cupy_stub(monkeypatch, get_device_count=_raise)
    import cupy
    with pytest.raises(RuntimeError, match="CUDA runtime is not usable"):
        gpu_mod._preflight_cuda_runtime(cupy)


def test_preflight_raises_on_zero_devices(monkeypatch):
    """``getDeviceCount()`` returning 0 also raises."""
    from xrspatial.geotiff._backends import gpu as gpu_mod

    _install_cupy_stub(monkeypatch, get_device_count=lambda: 0)
    import cupy
    with pytest.raises(RuntimeError, match="reports 0 CUDA devices"):
        gpu_mod._preflight_cuda_runtime(cupy)


def test_preflight_returns_silently_when_device_present(monkeypatch):
    """A normal positive device count must not raise."""
    from xrspatial.geotiff._backends import gpu as gpu_mod

    _install_cupy_stub(monkeypatch, get_device_count=lambda: 1)
    import cupy
    # Should not raise.
    gpu_mod._preflight_cuda_runtime(cupy)


def test_read_geotiff_gpu_preflight_surface(monkeypatch, tmp_path):
    """End-to-end: read_geotiff_gpu raises before touching any IFDs.

    Build a real TIFF so the function gets past the file-source setup,
    then verify the CUDA preflight RuntimeError surfaces from the
    public entry point rather than from a deep cupy.asarray() call.
    """
    import numpy as np
    import xarray as xr
    from xrspatial.geotiff import to_geotiff
    from xrspatial.geotiff._backends.gpu import read_geotiff_gpu

    da = xr.DataArray(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        dims=["y", "x"],
        coords={
            "y": np.array([0.5, 1.5, 2.5, 3.5]),
            "x": np.array([0.5, 1.5, 2.5, 3.5]),
        },
        attrs={"crs": 4326},
    )
    path = str(tmp_path / "preflight_1903.tif")
    to_geotiff(da, path, tile_size=16)

    class FakeCudaError(RuntimeError):
        pass

    def _raise(*_a, **_kw):
        raise FakeCudaError("cudaErrorInsufficientDriver")

    _install_cupy_stub(monkeypatch, get_device_count=_raise)

    with pytest.raises(RuntimeError, match="CUDA runtime is not usable"):
        read_geotiff_gpu(path)


@pytest.mark.skipif(
    not _CUPY_AVAILABLE,
    reason="cupy required to verify monkeypatch composes with a real import",
)
def test_preflight_when_real_cupy_present(monkeypatch):
    """When cupy is really installed, monkeypatching the runtime symbol
    works the same way -- the import in read_geotiff_gpu finds the
    patched attribute."""
    import cupy
    from xrspatial.geotiff._backends import gpu as gpu_mod

    class FakeCudaError(RuntimeError):
        pass

    def _raise(*_a, **_kw):
        raise FakeCudaError("cudaErrorInsufficientDriver")

    monkeypatch.setattr(cupy.cuda.runtime, "getDeviceCount", _raise)
    with pytest.raises(RuntimeError, match="CUDA runtime is not usable"):
        gpu_mod._preflight_cuda_runtime(cupy)
