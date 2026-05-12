"""Regression tests for issue #1674.

``to_geotiff(..., gpu=True)`` used to wrap the GPU writer in a too-broad
``except (ImportError, Exception)`` (equivalent to ``except Exception``)
that silently swallowed every failure and fell through to the CPU
pipeline. Real GPU regressions, CRS errors, and CuPy mismatches all
disappeared without warning even though the GPU and CPU writers do not
guarantee bit-identical output.

The fix:

1. ``ImportError`` (cupy missing, nvCOMP module load failure) falls back
   to the CPU writer and emits a ``GeoTIFFFallbackWarning``.
2. ``RuntimeError`` whose message names a GPU-availability signal (one
   of ``nvCOMP``, ``CUDA``, ``no device``, ``no GPU``, ``cuInit``) also
   falls back with a warning. Any other ``RuntimeError`` propagates.
3. Every other exception propagates unchanged.
4. ``XRSPATIAL_GEOTIFF_STRICT=1`` re-raises every fallback case.

These tests monkeypatch ``write_geotiff_gpu`` to raise synthetic
exceptions, so no real GPU is required. The output file is only checked
for existence (the CPU writer is tested elsewhere); the focus here is
on which exceptions propagate vs. trigger a fallback warning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import GeoTIFFFallbackWarning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def clear_strict_env(monkeypatch):
    """Default mode: ``XRSPATIAL_GEOTIFF_STRICT`` is unset."""
    monkeypatch.delenv('XRSPATIAL_GEOTIFF_STRICT', raising=False)


@pytest.fixture
def set_strict_env(monkeypatch):
    """Strict mode: ``XRSPATIAL_GEOTIFF_STRICT=1`` is set."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')


@pytest.fixture
def cpu_data():
    """A small 2D numpy-backed DataArray suitable for the CPU writer."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={
            'y': np.arange(8, dtype=np.float64),
            'x': np.arange(8, dtype=np.float64),
        },
        attrs={'crs': 4326},
    )


def _patch_gpu_writer_to_raise(monkeypatch, exc):
    """Replace ``write_geotiff_gpu`` (as resolved by ``to_geotiff``) with a
    stub that raises ``exc``.

    ``to_geotiff`` calls ``write_geotiff_gpu`` directly inside its own
    module, so the patch targets the module-level name there.
    """
    from xrspatial import geotiff as g

    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(g, 'write_geotiff_gpu', _boom, raising=True)


# ---------------------------------------------------------------------------
# Non-GPU exceptions must propagate, even with gpu=True.
# ---------------------------------------------------------------------------

def test_runtime_error_without_gpu_signal_propagates(
        tmp_path, cpu_data, clear_strict_env, monkeypatch):
    """A bare ``RuntimeError`` from the GPU writer must NOT be swallowed.

    This is the regression target. Before the fix, the bare except caught
    every exception type and silently dropped the user onto the CPU
    pipeline, producing a different file from the one they asked for.
    """
    from xrspatial.geotiff import to_geotiff

    _patch_gpu_writer_to_raise(
        monkeypatch, RuntimeError("synthetic non-GPU error"))

    path = tmp_path / "should_not_exist.tif"
    with pytest.raises(RuntimeError, match="synthetic non-GPU error"):
        to_geotiff(cpu_data, str(path), gpu=True)


def test_value_error_propagates(
        tmp_path, cpu_data, clear_strict_env, monkeypatch):
    """A ``ValueError`` from inside the GPU writer must not be swallowed."""
    from xrspatial.geotiff import to_geotiff

    _patch_gpu_writer_to_raise(
        monkeypatch, ValueError("synthetic value error"))

    path = tmp_path / "should_not_exist.tif"
    with pytest.raises(ValueError, match="synthetic value error"):
        to_geotiff(cpu_data, str(path), gpu=True)


# ---------------------------------------------------------------------------
# ImportError: cupy / nvCOMP not installed. Falls back with warning.
# ---------------------------------------------------------------------------

def test_import_error_falls_back_with_warning(
        tmp_path, cpu_data, clear_strict_env, monkeypatch):
    """``ImportError`` from the GPU writer triggers a CPU fallback.

    The user asked for ``gpu=True`` on a system without cupy. A
    ``GeoTIFFFallbackWarning`` makes the substitution visible and the
    text names the explicit request so users know which knob to tune.
    """
    from xrspatial.geotiff import to_geotiff

    _patch_gpu_writer_to_raise(monkeypatch, ImportError("no cupy"))

    path = tmp_path / "fallback.tif"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        to_geotiff(cpu_data, str(path), gpu=True)

    assert path.exists()
    fallback_warnings = [
        w for w in records
        if issubclass(w.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    msg = str(fallback_warnings[0].message)
    # Explicit gpu=True wording: blame the request, not the data.
    assert 'to_geotiff(gpu=True)' in msg
    assert 'Data is on the GPU' not in msg
    assert 'ImportError' in msg
    assert 'no cupy' in msg


def test_import_error_strict_mode_reraises(
        tmp_path, cpu_data, set_strict_env, monkeypatch):
    """``XRSPATIAL_GEOTIFF_STRICT=1`` promotes the ``ImportError`` fallback
    to a re-raise so CI catches the case where the GPU path silently
    degrades to CPU compression."""
    from xrspatial.geotiff import to_geotiff

    _patch_gpu_writer_to_raise(monkeypatch, ImportError("no nvCOMP"))

    path = tmp_path / "should_not_exist.tif"
    with pytest.raises(ImportError, match="no nvCOMP"):
        to_geotiff(cpu_data, str(path), gpu=True)


# ---------------------------------------------------------------------------
# RuntimeError with a GPU-availability signal: falls back with warning.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('msg', [
    "CUDA not available",
    "no device found",
    "nvCOMP library not loadable",
    "cuInit failed: no driver",
    "no GPU on this host",
])
def test_runtime_error_with_gpu_signal_falls_back(
        tmp_path, cpu_data, clear_strict_env, monkeypatch, msg):
    """A ``RuntimeError`` whose text names a GPU-availability problem is
    treated like ``ImportError``: fall back to CPU with a warning.

    Pattern matches keep the catch narrow without requiring a custom
    ``nvCompUnavailableError`` class. Anything that does not name CUDA /
    nvCOMP / no device / no GPU / cuInit is treated as a real bug and
    propagated by ``test_runtime_error_without_gpu_signal_propagates``.
    """
    from xrspatial.geotiff import to_geotiff

    _patch_gpu_writer_to_raise(monkeypatch, RuntimeError(msg))

    path = tmp_path / "fallback.tif"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        to_geotiff(cpu_data, str(path), gpu=True)

    assert path.exists()
    fallback_warnings = [
        w for w in records
        if issubclass(w.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    text = str(fallback_warnings[0].message)
    assert 'RuntimeError' in text
    # Explicit gpu=True branch shares the same template as ImportError;
    # the auto-detected wording must never appear here.
    assert 'to_geotiff(gpu=True)' in text
    assert 'Data is on the GPU' not in text


def test_runtime_error_with_gpu_signal_strict_reraises(
        tmp_path, cpu_data, set_strict_env, monkeypatch):
    """Strict mode re-raises GPU-availability ``RuntimeError`` too."""
    from xrspatial.geotiff import to_geotiff

    _patch_gpu_writer_to_raise(
        monkeypatch, RuntimeError("CUDA not available"))

    path = tmp_path / "should_not_exist.tif"
    with pytest.raises(RuntimeError, match="CUDA not available"):
        to_geotiff(cpu_data, str(path), gpu=True)


# ---------------------------------------------------------------------------
# Auto-detected gpu (from CuPy data): same fallback semantics.
# ---------------------------------------------------------------------------

def _make_synthetic_gpu_data():
    """Return a numpy-backed DataArray that ``_is_gpu_data`` will be
    patched to treat as GPU-resident."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={
            'y': np.arange(8, dtype=np.float64),
            'x': np.arange(8, dtype=np.float64),
        },
        attrs={'crs': 4326},
    )


def test_auto_detected_gpu_fallback_warns(
        tmp_path, clear_strict_env, monkeypatch):
    """When ``gpu`` is auto-detected from CuPy-backed data, the same
    fallback rules apply: ``ImportError`` triggers the warning.

    Users whose data was CuPy-backed deserve a warning every time the
    GPU writer failed so they know their array was copied to host
    before the CPU writer wrote it. The warning text must blame the
    auto-detect path, not an ``gpu=True`` argument the caller never
    passed.
    """
    from xrspatial.geotiff import to_geotiff

    # Synthesise a "CuPy-looking" DataArray via _is_gpu_data's hook.
    # Easiest: patch _is_gpu_data to True. The CPU fallback then
    # operates on the numpy buffer underneath.
    from xrspatial import geotiff as g
    monkeypatch.setattr(g, '_is_gpu_data', lambda data: True, raising=True)

    _patch_gpu_writer_to_raise(monkeypatch, ImportError("no cupy"))

    da = _make_synthetic_gpu_data()

    path = tmp_path / "auto.tif"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        # gpu defaults to None -> auto-detect path
        to_geotiff(da, str(path))

    assert path.exists()
    fallback_warnings = [
        w for w in records
        if issubclass(w.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    text = str(fallback_warnings[0].message)
    # Auto-detected branch wording: blame the data, not gpu=True.
    assert 'Data is on the GPU' in text
    assert 'to_geotiff(gpu=True)' not in text
    assert 'ImportError' in text
    assert 'no cupy' in text


def test_auto_detected_gpu_runtime_error_falls_back_with_warning(
        tmp_path, clear_strict_env, monkeypatch):
    """Same shape for the ``RuntimeError`` branch under auto-detect.

    Both fallback branches (ImportError, RuntimeError-with-GPU-signal)
    must use the same template so call sites do not diverge over time.
    """
    from xrspatial.geotiff import to_geotiff
    from xrspatial import geotiff as g

    monkeypatch.setattr(g, '_is_gpu_data', lambda data: True, raising=True)
    _patch_gpu_writer_to_raise(
        monkeypatch, RuntimeError("CUDA not available"))

    da = _make_synthetic_gpu_data()

    path = tmp_path / "auto_rt.tif"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        to_geotiff(da, str(path))

    assert path.exists()
    fallback_warnings = [
        w for w in records
        if issubclass(w.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    text = str(fallback_warnings[0].message)
    assert 'Data is on the GPU' in text
    assert 'to_geotiff(gpu=True)' not in text
    assert 'RuntimeError' in text
    assert 'CUDA not available' in text


def test_explicit_gpu_false_then_true_uses_explicit_template(
        tmp_path, cpu_data, clear_strict_env, monkeypatch):
    """``gpu=True`` plus non-CuPy data must use the explicit template
    even when ``_is_gpu_data`` would return False on its own.

    This pins down that the template is selected from ``gpu is None``,
    not from the resolved ``use_gpu`` value -- so passing ``gpu=True``
    on numpy data still attributes the fallback to the explicit flag.
    """
    from xrspatial.geotiff import to_geotiff
    from xrspatial import geotiff as g

    # Even if auto-detect would say "not GPU", the explicit request
    # should drive the wording.
    monkeypatch.setattr(g, '_is_gpu_data', lambda data: False, raising=True)
    _patch_gpu_writer_to_raise(monkeypatch, ImportError("no cupy"))

    path = tmp_path / "explicit.tif"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        to_geotiff(cpu_data, str(path), gpu=True)

    fallback_warnings = [
        w for w in records
        if issubclass(w.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    text = str(fallback_warnings[0].message)
    assert 'to_geotiff(gpu=True)' in text
    assert 'Data is on the GPU' not in text
