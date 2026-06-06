"""GPU kernels and kwarg-surface tests for the GeoTIFF module.

This module consolidates the GPU kernel + kwarg-surface tests
previously scattered across twelve top-level files.

Each section folds one source file. Module helpers and fixtures that
collided across sources are suffixed with the source issue number so
sibling sections stay independent. GPU gating goes through the shared
``requires_gpu`` marker from ``_helpers/markers.py`` (aliased
``_gpu_only`` for brevity); CPU-only validation tests (signature,
kwarg parsing, default dispatch) stay ungated so they run on every
host.

Sections (in source-file order):

* ``test_gpu_cuda_preflight_1903.py`` -- CUDA preflight in
  ``_read_geotiff_gpu``.
* ``test_gpu_kwarg_rename_1560.py`` -- ``gpu=`` -> ``on_gpu_failure=``
  rename with deprecation shim.
* ``test_gpu_strict_fallback_1516.py`` -- ``gpu='auto'`` warns +
  fallbacks; ``gpu='strict'`` re-raises.
* ``test_gpu_fallback_forwards_kwargs_2238.py`` -- CPU-fallback sites
  in ``_read_geotiff_gpu`` forward caller kwargs.
* ``test_kvikio_batched_pread_1688.py`` -- single-allocation batched
  pread in ``_try_kvikio_read_tiles``.
* ``test_gds_chunked_gpu_parity_1896.py`` -- bool-band rejection and
  LERC ``masked_fill`` forwarding on the GDS chunked path.
* ``test_gds_fallback_batched_d2h_1552.py`` -- batched D2H transfer
  on the GDS fallback.
* ``test_orientation_gpu.py`` -- TIFF Orientation tag on GPU reads.
* ``test_size_param_validation_gpu_vrt_1776.py`` -- ``tile_size``/
  ``chunks`` validation on GPU + VRT entry points.
* ``test_mask_nodata_gpu_vrt_2052.py`` -- ``mask_nodata=False`` on
  GPU + VRT entry points.
* ``test_open_geotiff_on_gpu_failure_1615.py`` -- ``on_gpu_failure``
  threading through ``open_geotiff``.
* ``test_crs_fail_closed_gpu_1929.py`` -- ``allow_unparseable_crs``
  fail-closed on GPU + dispatcher entry points.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import struct
import sys
import types
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff.tests.conftest import requires_gpu as _gpu_only

_CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None


# =====================================================================
# Section 1903 -- CUDA preflight in ``_read_geotiff_gpu``.
# =====================================================================
#
# When CuPy imports but the CUDA driver is
# unusable (older driver than the build expects, suspended VM, etc.),
# the failure used to surface as ``cudaErrorInsufficientDriver`` from a
# ``cupy.asarray(...)`` call deep in the CPU-fallback path. The fix
# preflights the runtime via ``cupy.cuda.runtime.getDeviceCount()``
# right after the cupy import and raises a clean ``RuntimeError``.
#
# These tests stub ``cupy.cuda.runtime.getDeviceCount`` so they
# exercise the preflight branch without requiring a real GPU.


def _install_cupy_stub_1903(monkeypatch, *, get_device_count):
    """Install a minimal stub ``cupy`` module so the preflight runs.

    Lets us drive ``_preflight_cuda_runtime`` deterministically
    regardless of the host's real CUDA state. Pass a ``get_device_count``
    that raises to exercise the preflight failure path (CPU-only CI or
    broken-CUDA hosts), or one that returns a positive int to exercise
    the success path (e.g. so the alias-validation test in #2515 can
    reach the file-read stage without depending on real CUDA).
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


def test_preflight_raises_on_runtime_error_1903(monkeypatch):
    """A simulated cudaErrorInsufficientDriver becomes a clean RuntimeError."""
    from xrspatial.geotiff._backends import gpu as gpu_mod

    class FakeCudaError(RuntimeError):
        pass

    def _raise(*_a, **_kw):
        raise FakeCudaError("cudaErrorInsufficientDriver")

    _install_cupy_stub_1903(monkeypatch, get_device_count=_raise)
    import cupy
    with pytest.raises(RuntimeError, match="CUDA runtime is not usable"):
        gpu_mod._preflight_cuda_runtime(cupy)


def test_preflight_raises_on_zero_devices_1903(monkeypatch):
    """``getDeviceCount()`` returning 0 also raises."""
    from xrspatial.geotiff._backends import gpu as gpu_mod

    _install_cupy_stub_1903(monkeypatch, get_device_count=lambda: 0)
    import cupy
    with pytest.raises(RuntimeError, match="reports 0 CUDA devices"):
        gpu_mod._preflight_cuda_runtime(cupy)


def test_preflight_returns_silently_when_device_present_1903(monkeypatch):
    """A normal positive device count must not raise."""
    from xrspatial.geotiff._backends import gpu as gpu_mod

    _install_cupy_stub_1903(monkeypatch, get_device_count=lambda: 1)
    import cupy

    # Should not raise.
    gpu_mod._preflight_cuda_runtime(cupy)


def test_read_geotiff_gpu_preflight_surface_1903(monkeypatch, tmp_path):
    """End-to-end: _read_geotiff_gpu raises before touching any IFDs.

    Build a real TIFF so the function gets past the file-source setup,
    then verify the CUDA preflight RuntimeError surfaces from the
    public entry point rather than from a deep cupy.asarray() call.
    """
    from xrspatial.geotiff import to_geotiff
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu

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

    _install_cupy_stub_1903(monkeypatch, get_device_count=_raise)

    with pytest.raises(RuntimeError, match="CUDA runtime is not usable"):
        _read_geotiff_gpu(path)


@pytest.mark.skipif(
    not _CUPY_AVAILABLE,
    reason="cupy required to verify monkeypatch composes with a real import",
)
def test_preflight_when_real_cupy_present_1903(monkeypatch):
    """When cupy is really installed, monkeypatching the runtime symbol
    works the same way -- the import in _read_geotiff_gpu finds the
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


# =====================================================================
# Section 1560 -- ``gpu=`` -> ``on_gpu_failure=`` rename
# =====================================================================
#
# ``_read_geotiff_gpu`` previously took a ``gpu={'auto','strict'}`` kwarg
# that controlled GPU-failure policy, sharing a name with the boolean
# ``gpu=`` kwarg on ``open_geotiff`` / ``to_geotiff`` / ``_read_vrt``.
# Calling ``_read_geotiff_gpu(path, gpu=True)`` -- the mental model after
# using ``open_geotiff(path, gpu=True)`` -- raised the unhelpful
# ``ValueError: gpu must be 'auto' or 'strict', got True``.
#
# The fix renames the kwarg to ``on_gpu_failure`` and keeps ``gpu=`` as
# a deprecation shim. These tests exercise the validation path only,
# which fires before the ``cupy`` import inside ``_read_geotiff_gpu``;
# no GPU runtime needed.


def test_on_gpu_failure_invalid_value_raises_value_error_1560():
    """Bad ``on_gpu_failure`` value still raises ``ValueError``."""
    from xrspatial.geotiff import _read_geotiff_gpu

    with pytest.raises(ValueError, match="on_gpu_failure must be"):
        _read_geotiff_gpu("/nonexistent.tif", on_gpu_failure='loose')


def test_gpu_alias_emits_deprecation_warning_1560():
    """Old ``gpu=`` kwarg still routes through, with a DeprecationWarning."""
    from xrspatial.geotiff import _read_geotiff_gpu

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        # Pass an invalid sentinel so we don't have to mock the full GPU
        # pipeline; ValueError fires after the deprecation handler runs.
        with pytest.raises(ValueError, match="on_gpu_failure must be"):
            _read_geotiff_gpu("/nonexistent.tif", gpu='loose')

    deprecations = [
        r for r in records if issubclass(r.category, DeprecationWarning)
    ]
    assert deprecations, "expected DeprecationWarning when gpu= is used"
    assert "on_gpu_failure" in str(deprecations[0].message)


def test_gpu_alias_accepts_old_values_without_validation_error_1560(
        monkeypatch):
    """``gpu='strict'`` was the legacy spelling; should still validate.

    Stub ``cupy`` with a working preflight so the test exercises the
    alias-validation path on every host, including ones where CuPy
    imports but the CUDA runtime is unusable (issue #2515). Without the
    stub, the real ``_preflight_cuda_runtime`` raises ``RuntimeError``
    on broken-CUDA hosts and the test fails for an environmental
    reason that has nothing to do with the alias logic under test.
    """
    from xrspatial.geotiff import _read_geotiff_gpu

    # Install a stub cupy whose preflight passes; the call should then
    # reach the file-read stage and raise ``FileNotFoundError``. This
    # decouples the alias-validation check from real CUDA state.
    _install_cupy_stub_1903(monkeypatch, get_device_count=lambda: 1)

    with warnings.catch_warnings():
        # Suppress the deprecation noise; we only care that the value
        # passes validation and the call proceeds past the value check.
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(
                (FileNotFoundError, OSError, ValueError)
        ) as exc_info:
            _read_geotiff_gpu("/nonexistent.tif", gpu='strict')

    # The validation ValueError carries our exact message; a generic
    # file-read failure is fine because it means validation passed.
    if isinstance(exc_info.value, ValueError):
        assert "on_gpu_failure must be" not in str(exc_info.value)


def test_passing_both_raises_type_error_1560():
    """Mixing the new and deprecated names is ambiguous; refuse."""
    from xrspatial.geotiff import _read_geotiff_gpu

    with pytest.raises(TypeError, match="pass either 'on_gpu_failure' or"):
        _read_geotiff_gpu(
            "/nonexistent.tif",
            on_gpu_failure='strict',
            gpu='auto',
        )


@pytest.mark.parametrize("on_gpu_failure_val,gpu_val", [
    ('auto', 'strict'),
    ('auto', 'auto'),
    ('strict', 'strict'),
])
def test_passing_both_raises_regardless_of_values_1560(
        on_gpu_failure_val, gpu_val):
    """Both-supplied is rejected even when ``on_gpu_failure='auto'``.

    A sentinel-based detection (rather than ``!= 'auto'``) catches the
    case where the caller passes the default value explicitly alongside
    the deprecated alias.
    """
    from xrspatial.geotiff import _read_geotiff_gpu

    with pytest.raises(TypeError, match="pass either 'on_gpu_failure' or"):
        _read_geotiff_gpu(
            "/nonexistent.tif",
            on_gpu_failure=on_gpu_failure_val,
            gpu=gpu_val,
        )


def test_gpu_alias_bool_no_longer_misleading_value_error_1560():
    """Calling with ``gpu=True`` -- the documented bool from the
    dispatchers -- used to raise ``ValueError: gpu must be 'auto' or
    'strict', got True``. The new error explicitly names
    ``on_gpu_failure`` so the rename is discoverable from the traceback.
    """
    from xrspatial.geotiff import _read_geotiff_gpu

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="on_gpu_failure must be"):
            _read_geotiff_gpu("/nonexistent.tif", gpu=True)


# =====================================================================
# Section 1516 -- ``gpu='auto'`` warns + falls back; ``'strict'`` raises
# =====================================================================
#
# ``_read_geotiff_gpu`` previously wrapped the GPU decode in a too-broad
# ``try/except Exception: pass`` that silently swallowed any failure
# and fell through to the CPU path. Real GPU regressions (an
# ``AttributeError``, for one) lived undetected because the user-visible result
# was still numerically correct.
#
# The fix:
# 1. Default ``gpu='auto'`` still falls back to CPU, but emits a
#    ``RuntimeWarning`` reporting the original exception type and
#    message so failures are visible.
# 2. New ``gpu='strict'`` mode re-raises instead of falling back, so
#    tests and CI for the GPU fast path see real errors.
#
# These tests monkeypatch ``gpu_decode_tiles_from_file`` to raise a
# synthetic exception. They do not require a real GPU because we stub
# ``cupy`` at the ``sys.modules`` level when it is not already
# available; ``cupy.asarray`` is only called in the CPU-fallback branch
# and is satisfied by a thin numpy-backed shim.


_CUPY_ORIG_SENTINEL_1516 = object()
_cupy_saved_1516 = _CUPY_ORIG_SENTINEL_1516
_cupy_cuda_saved_1516 = _CUPY_ORIG_SENTINEL_1516
_cupy_cuda_runtime_saved_1516 = _CUPY_ORIG_SENTINEL_1516


def _cuda_actually_available_1516() -> bool:
    """Return True only if cupy + CUDA are usable on this host.

    cupy may be importable on a machine without a working CUDA runtime
    (no driver, no device, ROCm-only, etc.). The CPU-fallback branch in
    ``_read_geotiff_gpu`` calls ``cupy.asarray`` which would then fail at
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


def _ensure_cupy_stub_1516() -> bool:
    """Install a numpy-backed ``cupy`` shim if real cupy isn't usable.

    Replaces ``sys.modules['cupy']`` whenever cupy is missing OR cupy
    is installed but CUDA isn't available. The original module (if any)
    is saved so :func:`_restore_cupy_1516` can put it back.
    """
    global _cupy_saved_1516, _cupy_cuda_saved_1516, _cupy_cuda_runtime_saved_1516

    if _cuda_actually_available_1516():
        return False

    _cupy_saved_1516 = sys.modules.get('cupy', _CUPY_ORIG_SENTINEL_1516)
    _cupy_cuda_saved_1516 = sys.modules.get(
        'cupy.cuda', _CUPY_ORIG_SENTINEL_1516)
    _cupy_cuda_runtime_saved_1516 = sys.modules.get(
        'cupy.cuda.runtime', _CUPY_ORIG_SENTINEL_1516)

    stub = types.ModuleType('cupy')
    stub.ndarray = np.ndarray
    stub.asarray = np.asarray

    cuda_mod = types.ModuleType('cupy.cuda')
    cuda_mod.is_available = lambda: False

    # Pre-flight check in ``_read_geotiff_gpu`` calls
    # ``cupy.cuda.runtime.getDeviceCount()`` to surface a clean
    # ``RuntimeError`` for broken-driver setups. Tests in this section
    # want to exercise the downstream simulated-failure paths, so the
    # stubbed runtime reports one device and the preflight lets
    # execution through. The real preflight tests live in section 1903
    # above.
    runtime_mod = types.ModuleType('cupy.cuda.runtime')
    runtime_mod.getDeviceCount = lambda: 1
    cuda_mod.runtime = runtime_mod
    stub.cuda = cuda_mod

    sys.modules['cupy'] = stub
    sys.modules['cupy.cuda'] = cuda_mod
    sys.modules['cupy.cuda.runtime'] = runtime_mod
    return True


def _restore_cupy_1516() -> None:
    """Undo :func:`_ensure_cupy_stub_1516`."""
    global _cupy_saved_1516, _cupy_cuda_saved_1516, _cupy_cuda_runtime_saved_1516
    for name, saved in (
        ('cupy', _cupy_saved_1516),
        ('cupy.cuda', _cupy_cuda_saved_1516),
        ('cupy.cuda.runtime', _cupy_cuda_runtime_saved_1516),
    ):
        if saved is _CUPY_ORIG_SENTINEL_1516:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved
    _cupy_saved_1516 = _CUPY_ORIG_SENTINEL_1516
    _cupy_cuda_saved_1516 = _CUPY_ORIG_SENTINEL_1516
    _cupy_cuda_runtime_saved_1516 = _CUPY_ORIG_SENTINEL_1516
    importlib.invalidate_caches()


@pytest.fixture
def tiled_tiff_path_1516(tmp_path):
    """A small tiled TIFF on disk that exercises the GPU tile path."""
    from xrspatial.geotiff.tests.conftest import make_minimal_tiff

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


def _patch_gpu_decode_to_raise_1516(monkeypatch, exc):
    """Replace ``gpu_decode_tiles_from_file`` with one that raises ``exc``."""
    from xrspatial.geotiff import _gpu_decode

    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles_from_file', _boom, raising=True,
    )


def _patch_both_gpu_stages_to_raise_1516(monkeypatch, exc):
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


def test_default_mode_warns_on_gpu_failure_1516(tiled_tiff_path_1516, monkeypatch):
    """Default ``gpu='auto'`` warns and falls back to the CPU result."""
    inserted_stub = _ensure_cupy_stub_1516()
    try:
        from xrspatial.geotiff import _read_geotiff_gpu

        path, expected = tiled_tiff_path_1516

        synthetic = RuntimeError("simulated GPU failure")
        _patch_gpu_decode_to_raise_1516(monkeypatch, synthetic)

        with pytest.warns(RuntimeWarning, match="GPU decode failed"):
            result = _read_geotiff_gpu(path)

        # Fallback returned the CPU-decoded data. Real cupy arrays
        # expose ``.get()`` to copy back to host; the numpy stub returns
        # a plain ndarray.
        out = result.data
        if hasattr(out, 'get'):
            out = out.get()
        np.testing.assert_array_equal(np.asarray(out), expected)
    finally:
        if inserted_stub:
            _restore_cupy_1516()


def test_strict_mode_reraises_1516(tiled_tiff_path_1516, monkeypatch):
    """``gpu='strict'`` re-raises the original GPU exception."""
    inserted_stub = _ensure_cupy_stub_1516()
    try:
        from xrspatial.geotiff import _read_geotiff_gpu

        path, _ = tiled_tiff_path_1516

        synthetic = RuntimeError("simulated GPU failure")
        _patch_gpu_decode_to_raise_1516(monkeypatch, synthetic)

        with pytest.raises(RuntimeError, match="simulated GPU failure"):
            _read_geotiff_gpu(path, gpu='strict')
    finally:
        if inserted_stub:
            _restore_cupy_1516()


def test_strict_mode_reraises_second_stage_1516(
        tiled_tiff_path_1516, monkeypatch):
    """``gpu='strict'`` re-raises if the second-stage GPU decode fails too.

    Regression for the case where ``gpu_decode_tiles_from_file`` and the
    follow-up ``gpu_decode_tiles`` both fail. Previously the second
    failure was caught by an unconditional ``except (ValueError, Exception)``
    that fell back to CPU regardless of mode.
    """
    inserted_stub = _ensure_cupy_stub_1516()
    try:
        from xrspatial.geotiff import _read_geotiff_gpu

        path, _ = tiled_tiff_path_1516

        synthetic = RuntimeError("simulated second-stage GPU failure")
        _patch_both_gpu_stages_to_raise_1516(monkeypatch, synthetic)

        with pytest.raises(RuntimeError,
                           match="simulated second-stage GPU failure"):
            _read_geotiff_gpu(path, gpu='strict')
    finally:
        if inserted_stub:
            _restore_cupy_1516()


def test_default_mode_warns_on_second_stage_failure_1516(
        tiled_tiff_path_1516, monkeypatch):
    """``gpu='auto'`` warns once per stage failure and falls back to CPU.

    Both GPU decode stages are forced to raise, so the user sees two
    distinct ``RuntimeWarning`` records (one per stage) before the CPU
    fallback fires. Asserting the exact count guards against a
    regression where one of the two handlers stops warning.
    """
    inserted_stub = _ensure_cupy_stub_1516()
    try:
        from xrspatial.geotiff import _read_geotiff_gpu

        path, expected = tiled_tiff_path_1516

        synthetic = RuntimeError("simulated second-stage GPU failure")
        _patch_both_gpu_stages_to_raise_1516(monkeypatch, synthetic)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            result = _read_geotiff_gpu(path)

        gpu_warnings = [
            w for w in records
            if issubclass(w.category, RuntimeWarning)
            and "GPU decode failed" in str(w.message)
        ]
        assert len(gpu_warnings) == 2, (
            f"expected one warning per GPU stage; got {len(gpu_warnings)}: "
            f"{[str(w.message) for w in gpu_warnings]}"
        )

        out = result.data
        if hasattr(out, 'get'):
            out = out.get()
        np.testing.assert_array_equal(np.asarray(out), expected)
    finally:
        if inserted_stub:
            _restore_cupy_1516()


def test_invalid_gpu_kwarg_rejected_1516(tiled_tiff_path_1516):
    """An unknown ``gpu=`` value raises ``ValueError`` with a clear message."""
    inserted_stub = _ensure_cupy_stub_1516()
    try:
        from xrspatial.geotiff import _read_geotiff_gpu

        path, _ = tiled_tiff_path_1516

        # The kwarg was renamed to ``on_gpu_failure``; the
        # validation error now names the new kwarg even when callers
        # supply the deprecated ``gpu=`` alias.
        with pytest.raises(ValueError,
                           match="on_gpu_failure must be 'auto' or 'strict'"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                _read_geotiff_gpu(path, gpu='loose')
    finally:
        if inserted_stub:
            _restore_cupy_1516()


# =====================================================================
# Section 2238 -- CPU-fallback sites forward caller kwargs
# =====================================================================
#
# ``_read_geotiff_gpu`` has four CPU-fallback call sites to
# ``_read_to_array``:
# - the stripped-layout branch
# - the planar=2 per-band stage-2 fallback
# - the sparse-tile fallback
# - the GPU-decode-failure fallback
#
# The last three used to drop ``allow_rotated``, ``window``, ``band``,
# and ``max_pixels`` and the stripped branch dropped ``allow_rotated``.
# These tests pin the contract: each fallback site must hand the
# caller's kwargs through to ``_read_to_array``.


# Rotated 4x4 ModelTransformation: pixel_width 1.0, b=0.1 (column-axis
# rotation), pixel_height -1.0, origin (100, 200).
_ROTATED_M_2238 = (
    1.0, 0.1, 0.0, 100.0,
    0.0, -1.0, 0.0, 200.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _write_rotated_tiled_tiff_2238(path, arr: np.ndarray, *,
                                   tile_w: int = 16, tile_h: int = 16,
                                   sparse: bool = False) -> None:
    """Write a single-IFD tiled TIFF with a rotated ModelTransformation.

    Hand-rolled to keep the fixture independent of rasterio/GDAL. The
    output has TileWidth/TileLength tags so the GPU reader takes the
    tiled branch, plus the rotated transform so ``allow_rotated`` is
    required to read it.

    When ``sparse=True``, the last tile is marked with offset=0 and
    byte_count=0 (or the single tile is, for 1x1 grids). This drives
    the reader's ``has_sparse_tile=True`` path so the GPU sparse-tile
    fallback is exercised.
    """
    from xrspatial.geotiff._geotags import TAG_MODEL_TRANSFORMATION

    h, w = arr.shape
    arr = np.ascontiguousarray(arr.astype('<u2'))
    tiles_across = (w + tile_w - 1) // tile_w
    tiles_down = (h + tile_h - 1) // tile_h
    n_tiles = tiles_across * tiles_down
    tile_bytes = tile_w * tile_h * 2

    # Pad each tile up to tile_w x tile_h and pack tiles row-major.
    tile_payloads = []
    for ty in range(tiles_down):
        for tx in range(tiles_across):
            tile = np.zeros((tile_h, tile_w), dtype='<u2')
            r0, c0 = ty * tile_h, tx * tile_w
            r1, c1 = min(r0 + tile_h, h), min(c0 + tile_w, w)
            tile[: r1 - r0, : c1 - c0] = arr[r0:r1, c0:c1]
            tile_payloads.append(tile.tobytes())

    if sparse:
        real_tiles = max(n_tiles - 1, 1) if n_tiles > 1 else 0
    else:
        real_tiles = n_tiles

    header_size = 8
    tile_data_off = header_size
    tile_data_size = real_tiles * tile_bytes
    offsets_arr_off = tile_data_off + tile_data_size
    offsets_arr_size = n_tiles * 4
    bytecounts_arr_off = offsets_arr_off + offsets_arr_size
    bytecounts_arr_size = n_tiles * 4
    transform_off = bytecounts_arr_off + bytecounts_arr_size
    transform_size = 16 * 8
    ifd_off = transform_off + transform_size

    tile_offsets = [tile_data_off + i * tile_bytes for i in range(real_tiles)]
    tile_byte_counts = [tile_bytes] * real_tiles
    if sparse:
        n_sparse = n_tiles - real_tiles
        tile_offsets.extend([0] * n_sparse)
        tile_byte_counts.extend([0] * n_sparse)

    entries = [
        (256, 3, 1, w),                # ImageWidth
        (257, 3, 1, h),                # ImageLength
        (258, 3, 1, 16),               # BitsPerSample = 16
        (259, 3, 1, 1),                # Compression = none
        (262, 3, 1, 1),                # Photometric = BlackIsZero
        (277, 3, 1, 1),                # SamplesPerPixel
        (322, 3, 1, tile_w),           # TileWidth
        (323, 3, 1, tile_h),           # TileLength
        (324, 4, n_tiles, offsets_arr_off),       # TileOffsets
        (325, 4, n_tiles, bytecounts_arr_off),    # TileByteCounts
        (339, 3, 1, 1),                # SampleFormat = unsigned int
        (TAG_MODEL_TRANSFORMATION, 12, 16, transform_off),
    ]
    entries.sort(key=lambda e: e[0])

    ifd_bytes = struct.pack('<H', len(entries))
    for tag, type_id, count, val in entries:
        if type_id == 3:  # SHORT
            ifd_bytes += struct.pack('<HHIHH', tag, type_id, count, val, 0)
        else:
            ifd_bytes += struct.pack('<HHII', tag, type_id, count, val)
    ifd_bytes += struct.pack('<I', 0)  # next IFD

    with open(path, 'wb') as f:
        f.write(struct.pack('<HHI', 0x4949, 42, ifd_off))
        for payload in tile_payloads[:real_tiles]:
            f.write(payload)
        f.write(struct.pack(f'<{n_tiles}I', *tile_offsets))
        f.write(struct.pack(f'<{n_tiles}I', *tile_byte_counts))
        f.write(struct.pack('<16d', *_ROTATED_M_2238))
        f.write(ifd_bytes)


def _make_kwarg_recorder_2238():
    """Build a wrapper that records every kwargs dict it sees.

    The wrapper still calls through to the real ``_read_to_array`` so
    the GPU pipeline keeps working; tests assert against the recorded
    kwargs.
    """
    from xrspatial.geotiff._reader import read_to_array as _real

    seen: list[dict] = []

    def _wrapper(*args, **kwargs):
        seen.append(dict(kwargs))
        return _real(*args, **kwargs)

    return _wrapper, seen


@_gpu_only
def test_stripped_fallback_forwards_allow_rotated_2238(tmp_path, monkeypatch):
    """Stripped-layout CPU fallback receives ``allow_rotated``.

    Earlier the stripped branch already forwarded window/band/max_pixels
    but dropped ``allow_rotated``. The xfail-flipped test in
    ``test_allow_rotated_no_crs_2122.py`` proves the end-to-end
    behaviour; this one pins the kwarg-forwarding contract via a
    recorder so a later refactor cannot silently drop the kwarg again.
    """
    from xrspatial.geotiff import _read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gpu_backend
    from xrspatial.geotiff._geotags import TAG_MODEL_TRANSFORMATION

    # Build a stripped (non-tiled) rotated single-strip TIFF directly.
    src = tmp_path / "2238_stripped_rotated.tif"
    h, w = 4, 5
    arr = np.arange(h * w, dtype='<u2').reshape(h, w)
    header_size = 8
    strip_size = h * w * 2
    transform_off = header_size + strip_size
    ifd_off = transform_off + 16 * 8

    entries = [
        (256, 3, 1, w), (257, 3, 1, h), (258, 3, 1, 16),
        (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, header_size),
        (277, 3, 1, 1), (278, 3, 1, h), (279, 4, 1, strip_size),
        (339, 3, 1, 1),
        (TAG_MODEL_TRANSFORMATION, 12, 16, transform_off),
    ]
    entries.sort(key=lambda e: e[0])
    ifd_bytes = struct.pack('<H', len(entries))
    for tag, type_id, count, val in entries:
        if type_id == 3:
            ifd_bytes += struct.pack('<HHIHH', tag, type_id, count, val, 0)
        else:
            ifd_bytes += struct.pack('<HHII', tag, type_id, count, val)
    ifd_bytes += struct.pack('<I', 0)
    with open(src, 'wb') as f:
        f.write(struct.pack('<HHI', 0x4949, 42, ifd_off))
        f.write(arr.tobytes())
        f.write(struct.pack('<16d', *_ROTATED_M_2238))
        f.write(ifd_bytes)

    wrapper, seen = _make_kwarg_recorder_2238()
    monkeypatch.setattr(gpu_backend, '_read_to_array', wrapper, raising=True)

    da = _read_geotiff_gpu(str(src), allow_rotated=True)

    assert len(seen) == 1, f"expected one fallback call, got {len(seen)}"
    assert seen[0].get('allow_rotated') is True, (
        f"stripped fallback dropped allow_rotated; kwargs={seen[0]}"
    )
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_sparse_tile_fallback_forwards_all_kwargs_2238(tmp_path, monkeypatch):
    """Sparse-tile fallback hands every caller kwarg to ``_read_to_array``."""
    from xrspatial.geotiff import _read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gpu_backend

    src = tmp_path / "2238_sparse_rotated.tif"
    h, w = 32, 32
    arr = np.arange(h * w, dtype='<u2').reshape(h, w)
    _write_rotated_tiled_tiff_2238(
        str(src), arr, tile_w=16, tile_h=16, sparse=True)

    wrapper, seen = _make_kwarg_recorder_2238()
    monkeypatch.setattr(gpu_backend, '_read_to_array', wrapper, raising=True)

    requested_window = (0, 0, 16, 16)
    requested_max_pixels = 10_000

    da = _read_geotiff_gpu(
        str(src),
        allow_rotated=True,
        window=requested_window,
        max_pixels=requested_max_pixels,
    )

    assert len(seen) == 1, (
        f"expected one sparse-tile fallback call to _read_to_array, "
        f"got {len(seen)} (kwargs sequence: {seen})"
    )
    call = seen[0]
    assert call.get('allow_rotated') is True, (
        f"sparse-tile fallback dropped allow_rotated; kwargs={call}"
    )
    assert call.get('window') == requested_window, (
        f"sparse-tile fallback dropped window; kwargs={call}"
    )
    assert call.get('max_pixels') == requested_max_pixels, (
        f"sparse-tile fallback dropped max_pixels; kwargs={call}"
    )
    assert 'band' in call, (
        f"sparse-tile fallback did not pass band kwarg; kwargs={call}"
    )

    host = da.data.get()
    assert host.shape == (16, 16), host.shape
    assert np.array_equal(host, arr[:16, :16].astype(host.dtype))


@_gpu_only
def test_gpu_decode_failure_fallback_forwards_all_kwargs_2238(
        tmp_path, monkeypatch):
    """``gpu_decode_tiles`` failure routes through fallback with kwargs."""
    from xrspatial.geotiff import _gpu_decode, _read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gpu_backend

    src = tmp_path / "2238_decode_fail.tif"
    h, w = 32, 32
    arr = np.arange(h * w, dtype='<u2').reshape(h, w)
    _write_rotated_tiled_tiff_2238(str(src), arr, tile_w=16, tile_h=16)

    def _raise_from_file(*args, **kwargs):
        raise RuntimeError("synthetic GDS decode failure (test #2238)")

    def _raise_decode(*args, **kwargs):
        raise RuntimeError("synthetic GPU tile decode failure (test #2238)")

    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles', _raise_decode, raising=True)
    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles_from_file', _raise_from_file,
        raising=True)

    wrapper, seen = _make_kwarg_recorder_2238()
    monkeypatch.setattr(gpu_backend, '_read_to_array', wrapper, raising=True)

    requested_window = (0, 0, 16, 16)
    requested_max_pixels = 5_000

    da = _read_geotiff_gpu(
        str(src),
        allow_rotated=True,
        window=requested_window,
        max_pixels=requested_max_pixels,
    )

    assert seen, "GPU-decode-failure fallback did not call _read_to_array"
    call = seen[-1]
    assert call.get('allow_rotated') is True, (
        f"decode-failure fallback dropped allow_rotated; kwargs={call}"
    )
    assert call.get('window') == requested_window, (
        f"decode-failure fallback dropped window; kwargs={call}"
    )
    assert call.get('max_pixels') == requested_max_pixels, (
        f"decode-failure fallback dropped max_pixels; kwargs={call}"
    )
    assert 'band' in call, (
        f"decode-failure fallback did not pass band kwarg; kwargs={call}"
    )

    host = da.data.get()
    assert host.shape == (16, 16), host.shape
    np.testing.assert_array_equal(host, arr[:16, :16].astype(host.dtype))


@_gpu_only
def test_planar2_fallback_forwards_all_kwargs_2238(tmp_path, monkeypatch):
    """planar=2 per-band stage-2 fallback forwards every read kwarg.

    Uses a non-rotated planar=2 multi-band file (rotated + multi-band
    is a harder fixture to write by hand) and monkeypatches the
    per-band decoder to return ``None``, which drives
    ``cpu_fallback_needed`` true and exercises the planar=2 fallback. The kwarg
    assertions are the invariant; the rotated case is covered by the
    sparse-tile and decode-failure tests above.
    """
    tifffile = pytest.importorskip("tifffile")

    from xrspatial.geotiff import _read_geotiff_gpu
    from xrspatial.geotiff._backends import gpu as gpu_backend

    src = tmp_path / "2238_planar2.tif"
    bands, h, w = 2, 64, 64
    rng = np.random.RandomState(2238)
    data = rng.randint(0, 255, size=(bands, h, w)).astype(np.uint8)
    tifffile.imwrite(
        str(src), data,
        photometric='minisblack',
        planarconfig='separate',
        tile=(32, 32),
    )

    def _none_band(*args, **kwargs):
        return None

    monkeypatch.setattr(
        gpu_backend, '_gpu_decode_single_band_tiles', _none_band,
        raising=True)

    wrapper, seen = _make_kwarg_recorder_2238()
    monkeypatch.setattr(gpu_backend, '_read_to_array', wrapper, raising=True)

    requested_max_pixels = 50_000

    da = _read_geotiff_gpu(
        str(src),
        max_pixels=requested_max_pixels,
    )

    assert seen, "planar=2 fallback did not call _read_to_array"
    call = seen[-1]
    assert call.get('max_pixels') == requested_max_pixels, (
        f"planar=2 fallback dropped max_pixels; kwargs={call}"
    )
    assert 'allow_rotated' in call, (
        f"planar=2 fallback did not pass allow_rotated; kwargs={call}"
    )
    assert call.get('allow_rotated') is False, (
        f"planar=2 fallback default allow_rotated should be False; "
        f"kwargs={call}"
    )
    assert 'window' in call and 'band' in call, (
        f"planar=2 fallback did not pass window/band; kwargs={call}"
    )

    host = da.data.get()
    assert host.shape == (h, w, bands), host.shape
    np.testing.assert_array_equal(host, np.transpose(data, (1, 2, 0)))


@_gpu_only
def test_decode_failure_fallback_applies_window_band_2238(
        tmp_path, monkeypatch):
    """Window/band selection on the decode-failure fallback path matches.

    The fix forwards ``window``/``band`` to ``_read_to_array`` and then
    skips the post-decode ``_gpu_apply_window_band`` slicer (the buffer
    is already windowed). Verifies the end-to-end shape and values do
    not depend on the slicer for the CPU-fallback path.
    """
    tifffile = pytest.importorskip("tifffile")

    from xrspatial.geotiff import _gpu_decode, _read_geotiff_gpu

    src = tmp_path / "2238_windowed_fallback.tif"
    bands, h, w = 3, 64, 64
    rng = np.random.RandomState(0x2238)
    data = rng.randint(0, 200, size=(bands, h, w)).astype(np.uint8)
    tifffile.imwrite(
        str(src), np.transpose(data, (1, 2, 0)),
        photometric='rgb',
        planarconfig='contig',
        tile=(32, 32),
    )

    def _raise_decode(*args, **kwargs):
        raise RuntimeError("force CPU fallback (test #2238)")

    def _raise_from_file(*args, **kwargs):
        raise RuntimeError("force CPU fallback (test #2238)")

    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles', _raise_decode, raising=True)
    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles_from_file', _raise_from_file,
        raising=True)

    requested_window = (8, 4, 40, 36)
    requested_band = 1

    da = _read_geotiff_gpu(
        str(src),
        window=requested_window,
        band=requested_band,
    )

    expected_h = requested_window[2] - requested_window[0]
    expected_w = requested_window[3] - requested_window[1]
    host = da.data.get()
    assert host.shape == (expected_h, expected_w), host.shape

    r0, c0, r1, c1 = requested_window
    expected = data[requested_band, r0:r1, c0:c1]
    np.testing.assert_array_equal(host, expected)


# =====================================================================
# Section 1688 -- ``_try_kvikio_read_tiles`` single-buffer batched pread
# =====================================================================
#
# ``_try_kvikio_read_tiles`` used to allocate one
# ``cupy.empty(bc)`` per tile and block on ``IOFuture.get()`` between
# successive ``pread`` calls. That forced the GDS reads to serialise
# in kvikio's worker pool and paid the per-tile cupy allocation cost
# N times. The fix:
# - pre-allocates one contiguous ``cupy.empty(sum(tile_byte_counts))``
#   buffer guarded by ``_check_gpu_memory``;
# - submits every ``pread`` call before waiting on any of them;
# - returns ``list[cupy.ndarray]`` per-tile views into the shared
#   buffer so the downstream nvCOMP / batched-D2H consumers are
#   unchanged.
#
# The kvikio integration path uses a fake CuFile so the structural
# checks (single allocation, submit-then-wait ordering, memory guard)
# run on hosts without kvikio.


class _FakeIOFuture_1688:
    """Stand-in for ``kvikio._lib.cufile.IOFuture``.

    ``.get()`` returns the requested byte count; tests that exercise
    the partial-read fallback construct one with ``bc - 1`` instead so
    the function returns None.
    """

    def __init__(self, value):
        self._value = value
        self.get_called = False

    def get(self):
        self.get_called = True
        return self._value


class _RecordingCuFile_1688:
    """In-memory CuFile that records pread arguments and write order.

    Writes deterministic bytes into the provided buffer so the test
    can verify the result is a list of per-tile views over one
    contiguous buffer (not a list of independent allocations).
    """

    def __init__(self, file_bytes):
        self.file_bytes = file_bytes
        self.preads = []
        self.pread_order = []
        self.gets_after_preads = 0
        self._preads_seen = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def pread(self, buf, file_offset=0, size=None, task_size=None):
        if size is None:
            size = int(buf.size)
        import cupy

        host_chunk = np.frombuffer(
            self.file_bytes[file_offset:file_offset + size], dtype=np.uint8)
        buf[:] = cupy.asarray(host_chunk)

        self.pread_order.append(
            (int(file_offset), int(size), int(buf.data.ptr)))
        self._preads_seen += 1
        return _FakeIOFuture_1688(size)


class _PartialRecordingCuFile_1688(_RecordingCuFile_1688):
    """CuFile whose nth pread short-reads (returns ``bc - 1`` bytes)."""

    def __init__(self, file_bytes, fail_index):
        super().__init__(file_bytes)
        self.fail_index = fail_index

    def pread(self, buf, file_offset=0, size=None, task_size=None):
        if size is None:
            size = int(buf.size)
        future = super().pread(buf, file_offset=file_offset, size=size)
        if len(self.pread_order) == self.fail_index + 1:
            return _FakeIOFuture_1688(size - 1)
        return future


def _install_fake_kvikio_1688(monkeypatch, cufile_cls):
    """Make ``import kvikio`` inside ``_try_kvikio_read_tiles`` resolve
    to a module whose ``CuFile`` is our recording stand-in.

    The function imports kvikio lazily, so monkeypatching the module
    via ``sys.modules`` is enough for both the install-present and
    install-absent code paths to see the fake.
    """
    fake_mod = types.ModuleType("kvikio")
    fake_mod.CuFile = cufile_cls
    monkeypatch.setitem(sys.modules, "kvikio", fake_mod)
    return fake_mod


def test_empty_tile_list_returns_empty_list_1688():
    """Zero tiles must return ``[]`` without touching cupy or kvikio."""
    from xrspatial.geotiff._gpu_decode import _try_kvikio_read_tiles

    assert _try_kvikio_read_tiles("/nonexistent", [], [], 0) == []


@_gpu_only
def test_kvikio_missing_returns_none_1688(monkeypatch):
    """When kvikio is not installed, the function must return None.

    ``gpu_decode_tiles_from_file`` relies on this signal to switch to
    the CPU mmap fallback. Without it, the caller would see an
    ImportError it cannot recover from.
    """
    from xrspatial.geotiff._gpu_decode import _try_kvikio_read_tiles

    # Force the kvikio import inside _try_kvikio_read_tiles to ImportError.
    monkeypatch.setitem(sys.modules, "kvikio", None)

    result = _try_kvikio_read_tiles(
        "/path/does/not/matter", [0, 1024], [1024, 1024], 1024)
    assert result is None


@_gpu_only
def test_single_buffer_allocation_1688(monkeypatch):
    """The fix allocates one contiguous device buffer, not N small ones.

    Verified structurally: every pread's destination data-pointer must
    fall within the single base allocation. The buffer pointer for
    tile i must equal ``base_ptr + offset_i`` where ``offset_i =
    sum(sizes[:i])``.
    """
    import cupy

    from xrspatial.geotiff._gpu_decode import _try_kvikio_read_tiles

    tile_offsets = [0, 4096, 8192, 12288]
    tile_byte_counts = [1024, 2048, 512, 768]
    file_size = max(o + bc for o, bc in zip(tile_offsets, tile_byte_counts))
    file_bytes = np.arange(file_size, dtype=np.uint64).tobytes()[:file_size]

    fake_cufile = _RecordingCuFile_1688(file_bytes)
    _install_fake_kvikio_1688(monkeypatch, lambda path, mode='r': fake_cufile)

    result = _try_kvikio_read_tiles(
        "/fake/path.tif", tile_offsets, tile_byte_counts,
        max(tile_byte_counts))

    assert result is not None
    assert len(result) == len(tile_byte_counts)
    for view, expected_bc in zip(result, tile_byte_counts):
        assert isinstance(view, cupy.ndarray)
        assert int(view.size) == expected_bc
        assert view.dtype == cupy.uint8

    ptrs = [int(v.data.ptr) for v in result]
    base = ptrs[0]
    for i, view in enumerate(result):
        expected = base + sum(tile_byte_counts[:i])
        assert int(view.data.ptr) == expected, (
            f"tile {i} pointer {int(view.data.ptr)} != expected {expected}; "
            "per-tile allocations leaked back in?"
        )


@_gpu_only
def test_all_preads_submitted_before_any_get_1688(monkeypatch):
    """Every ``pread`` must be submitted before the first ``IOFuture.get()``.

    The old loop alternated submit -> wait -> submit -> wait, which
    serialised the IO in kvikio's worker pool. The new loop submits N
    preads then waits on them in order; observing all N submissions
    before any ``.get()`` is the structural signature of the fix.
    """
    from xrspatial.geotiff._gpu_decode import _try_kvikio_read_tiles

    events = []

    class _LoggingFuture(_FakeIOFuture_1688):
        def __init__(self, value, tag):
            super().__init__(value)
            self._tag = tag

        def get(self):
            events.append(('get', self._tag))
            return super().get()

    class _LoggingCuFile(_RecordingCuFile_1688):
        def pread(self, buf, file_offset=0, size=None, task_size=None):
            if size is None:
                size = int(buf.size)
            tag = sum(1 for e in events if e[0] == 'submit')
            events.append(('submit', tag))
            super().pread(buf, file_offset=file_offset, size=size)
            return _LoggingFuture(size, tag)

    file_bytes = bytes(4096)
    _install_fake_kvikio_1688(
        monkeypatch, lambda path, mode='r': _LoggingCuFile(file_bytes))

    tile_offsets = [0, 256, 512, 768]
    tile_byte_counts = [256, 256, 256, 256]
    _try_kvikio_read_tiles(
        "/fake/path.tif", tile_offsets, tile_byte_counts, 256)

    n_tiles = len(tile_byte_counts)
    submit_indices = [i for i, e in enumerate(events) if e[0] == 'submit']
    get_indices = [i for i, e in enumerate(events) if e[0] == 'get']

    assert len(submit_indices) == n_tiles
    assert len(get_indices) == n_tiles
    assert [e[1] for e in events if e[0] == 'submit'] == list(range(n_tiles))
    assert [e[1] for e in events if e[0] == 'get'] == list(range(n_tiles))

    assert max(submit_indices) < min(get_indices), (
        "preads and gets are interleaved (legacy serial pattern); "
        f"events timeline: {events}"
    )


@_gpu_only
def test_memory_guard_runs_with_total_byte_count_1688(monkeypatch):
    """The single-buffer allocation must be size-checked before ``cupy.empty``.

    The OOM guard tells the caller early that the read will not fit on
    the device. A regression that removed it would surface as an
    opaque CUDA OOM only after the first ``cupy.empty`` failed.
    """
    from xrspatial.geotiff import _gpu_decode
    from xrspatial.geotiff._gpu_decode import _try_kvikio_read_tiles

    seen = {"total_bytes": None, "what": None, "called": False}

    def fake_check(required_bytes, what="tile buffer"):
        seen["total_bytes"] = int(required_bytes)
        seen["what"] = what
        seen["called"] = True
        raise MemoryError("simulated OOM")

    file_bytes = bytes(4096)
    _install_fake_kvikio_1688(
        monkeypatch,
        lambda path, mode='r': _RecordingCuFile_1688(file_bytes))
    monkeypatch.setattr(_gpu_decode, "_check_gpu_memory", fake_check)

    tile_offsets = [0, 1024, 2048]
    tile_byte_counts = [1024, 1024, 1024]

    with pytest.raises(MemoryError, match="simulated OOM"):
        _try_kvikio_read_tiles(
            "/fake/path.tif", tile_offsets, tile_byte_counts, 1024)

    assert seen["called"], "_check_gpu_memory was not called"
    assert seen["total_bytes"] == sum(tile_byte_counts), (
        f"expected total {sum(tile_byte_counts)}, got {seen['total_bytes']}"
    )
    assert "kvikio" in seen["what"] or "read buffer" in seen["what"], (
        f"unhelpful 'what' label: {seen['what']!r}"
    )


@_gpu_only
def test_partial_read_returns_none_1688(monkeypatch):
    """When any pread reports fewer bytes than requested, the function
    must return None so the caller falls back.
    """
    from xrspatial.geotiff._gpu_decode import _try_kvikio_read_tiles

    file_bytes = bytes(4096)
    fail_at = 1

    def _factory(path, mode='r'):
        return _PartialRecordingCuFile_1688(file_bytes, fail_at)

    _install_fake_kvikio_1688(monkeypatch, _factory)

    tile_offsets = [0, 256, 512]
    tile_byte_counts = [256, 256, 256]
    result = _try_kvikio_read_tiles(
        "/fake/path.tif", tile_offsets, tile_byte_counts, 256)
    assert result is None


@_gpu_only
def test_round_trip_data_preserved_1688(monkeypatch):
    """End-to-end: the bytes read into the per-tile views must match
    the bytes at the corresponding file offsets.
    """
    from xrspatial.geotiff._gpu_decode import _try_kvikio_read_tiles

    rng = np.random.default_rng(seed=1688)

    tile_offsets = [0, 1024, 2048, 3072]
    tile_byte_counts = [1024, 1024, 1024, 1024]
    file_size = 4096
    file_data = rng.integers(0, 256, size=file_size, dtype=np.uint8)
    file_bytes = file_data.tobytes()

    _install_fake_kvikio_1688(
        monkeypatch,
        lambda path, mode='r': _RecordingCuFile_1688(file_bytes))

    result = _try_kvikio_read_tiles(
        "/fake/path.tif", tile_offsets, tile_byte_counts, 1024)

    assert result is not None
    assert len(result) == 4
    for i, view in enumerate(result):
        got = view.get()
        want = file_data[
            tile_offsets[i]:tile_offsets[i] + tile_byte_counts[i]]
        assert np.array_equal(got, want), f"tile {i} payload mismatch"


@_gpu_only
def test_zero_size_tile_returns_zero_length_view_1688(monkeypatch):
    """A tile with ``byte_count == 0`` (sparse tile) must round-trip
    as a zero-length view in the result list so the caller's
    iteration order matches the original tile order.
    """
    import cupy

    from xrspatial.geotiff._gpu_decode import _try_kvikio_read_tiles

    tile_offsets = [0, 1024, 1024, 2048]
    tile_byte_counts = [1024, 0, 1024, 1024]
    file_bytes = bytes(3072)

    _install_fake_kvikio_1688(
        monkeypatch,
        lambda path, mode='r': _RecordingCuFile_1688(file_bytes))

    result = _try_kvikio_read_tiles(
        "/fake/path.tif", tile_offsets, tile_byte_counts, 1024)

    assert result is not None
    assert len(result) == 4
    assert int(result[1].size) == 0
    assert isinstance(result[1], cupy.ndarray)
    assert result[1].dtype == cupy.uint8


@_gpu_only
def test_all_zero_size_tiles_returns_zero_length_views_1688(monkeypatch):
    """Edge: every tile is sparse (sum bytes == 0). Must return a list
    of zero-length views without allocating a zero-sized buffer.
    """
    import cupy

    from xrspatial.geotiff._gpu_decode import _try_kvikio_read_tiles

    _install_fake_kvikio_1688(
        monkeypatch,
        lambda path, mode='r': _RecordingCuFile_1688(b""))

    result = _try_kvikio_read_tiles(
        "/fake/path.tif", [0, 0, 0], [0, 0, 0], 0)

    assert result is not None
    assert len(result) == 3
    for view in result:
        assert isinstance(view, cupy.ndarray)
        assert int(view.size) == 0


# =====================================================================
# Section 1896 -- GDS chunked path parity (bool-band reject, LERC mask)
# =====================================================================
#
# Pin two parity gaps in the GDS chunked GPU read path.
#
# 1. Bool-band rejection. ``_read_geotiff_gpu_chunked_gds`` used to
#    validate ``band`` with a plain numeric range check. Because
#    ``isinstance(True, int)`` is True in Python, ``band=True`` slipped
#    past ``True < n_bands_out`` and silently selected band 1. The
#    eager GPU path and the dask path already rejected bools
#    up front; the GDS chunked path now does too.
#
# 2. LERC ``masked_fill`` forwarding. ``_decode_window_gpu_direct``
#    used to call ``gpu_decode_tiles_from_file`` and ``gpu_decode_tiles``
#    without forwarding ``masked_fill``. On a LERC file with a per-pixel
#    valid mask, the chunked path left invalid pixels at LERC's zero
#    fill instead of restoring the sentinel before
#    ``_apply_nodata_mask_gpu`` ran. The eager GPU path resolved
#    ``masked_fill`` once and threaded it through both kernels;
#    the chunked path now mirrors that.


def _parse_for_gds_1896(path: str):
    """Return ``(ifd, geo_info, header)`` for the GDS entry point."""
    from xrspatial.geotiff._geotags import extract_geo_info_with_overview_inheritance
    from xrspatial.geotiff._header import parse_all_ifds, parse_header, select_overview_ifd
    from xrspatial.geotiff._reader import _FileSource

    with _FileSource(path) as fs:
        raw = fs.read_all()
    header = parse_header(raw)
    ifds = parse_all_ifds(raw, header)
    ifd = select_overview_ifd(ifds, None)
    geo_info = extract_geo_info_with_overview_inheritance(
        ifd, ifds, raw, header.byte_order,
    )
    return ifd, geo_info, header


@pytest.fixture
def multiband_tiff_1896(tmp_path):
    """4x6 three-band tiled tiff for the bool-rejection test."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(72, dtype=np.float32).reshape(4, 6, 3)
    da = xr.DataArray(
        arr,
        dims=['y', 'x', 'band'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
            'band': [0, 1, 2],
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'mb_gds_chunked_1896.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p)


@_gpu_only
def test_gds_chunked_band_true_rejected_1896(multiband_tiff_1896):
    """``_read_geotiff_gpu_chunked_gds(band=True)`` raises ValueError."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds_1896(multiband_tiff_1896)
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        _read_geotiff_gpu_chunked_gds(
            multiband_tiff_1896, ifd, geo_info, header,
            dtype=None, chunks=4, window=None, band=True,
            name=None, max_pixels=None,
        )


@_gpu_only
def test_gds_chunked_band_false_rejected_1896(multiband_tiff_1896):
    """``band=False`` is rejected the same way."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds_1896(multiband_tiff_1896)
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        _read_geotiff_gpu_chunked_gds(
            multiband_tiff_1896, ifd, geo_info, header,
            dtype=None, chunks=4, window=None, band=False,
            name=None, max_pixels=None,
        )


@_gpu_only
def test_gds_chunked_band_np_bool_rejected_1896(multiband_tiff_1896):
    """``np.bool_`` is rejected too (not a subclass of ``bool``)."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds_1896(multiband_tiff_1896)
    with pytest.raises(ValueError, match="band must be a non-negative int"):
        _read_geotiff_gpu_chunked_gds(
            multiband_tiff_1896, ifd, geo_info, header,
            dtype=None, chunks=4, window=None, band=np.bool_(True),
            name=None, max_pixels=None,
        )


@_gpu_only
def test_gds_chunked_band_int_still_works_1896(multiband_tiff_1896):
    """Plain int ``band=1`` continues to select the right band."""
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds

    ifd, geo_info, header = _parse_for_gds_1896(multiband_tiff_1896)
    result = _read_geotiff_gpu_chunked_gds(
        multiband_tiff_1896, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=1,
        name=None, max_pixels=None,
    )
    expected = np.arange(72, dtype=np.float32).reshape(4, 6, 3)[:, :, 1]
    out = result.data.compute().get()
    np.testing.assert_array_equal(out, expected)


# LERC sub-section: gated on cupy + CUDA + lerc. The module-level
# ``pytest.importorskip("lerc")`` in the original file would skip every
# test in this consolidated module if lerc were missing; gate per-test
# instead so unrelated sections still collect.
_LERC_INSTALLED = importlib.util.find_spec("lerc") is not None
try:
    from xrspatial.geotiff._compression import LERC_AVAILABLE
except ImportError:
    LERC_AVAILABLE = False

_lerc_required_1896 = pytest.mark.skipif(
    not (_LERC_INSTALLED and LERC_AVAILABLE),
    reason="lerc required",
)


@pytest.fixture
def lerc_writer_with_mask_1896(monkeypatch):
    """Inject a per-tile mask into the LERC writer (see #1529 / #1817).

    The xrspatial writer hard-codes ``hasMask=False`` in its
    ``lerc.encode`` call. Monkeypatch ``lerc_compress`` so the mask
    survives the encode and reappears at decode time -- the only way
    to exercise the GPU LERC mask path without an external mask-bearing
    fixture file.
    """
    import lerc as _lerc_mod

    holder = {"invalid": None}

    def _patched(data, width, height, samples=1,
                 dtype=np.dtype('float32'), max_z_error=0.0):
        if samples == 1:
            arr = np.frombuffer(data, dtype=dtype).reshape(height, width)
        else:
            arr = np.frombuffer(data, dtype=dtype).reshape(
                height, width, samples)
        invalid_pred = holder["invalid"]
        if invalid_pred is None:
            mask = None
            has_mask = False
        else:
            invalid = invalid_pred(arr)
            mask = np.where(invalid, np.uint8(0), np.uint8(1))
            has_mask = True
        result = _lerc_mod.encode(
            arr, samples, has_mask, mask, max_z_error, 1)
        if result[0] != 0:
            raise RuntimeError(
                f"LERC encode failed with error code {result[0]}")
        return bytes(result[2])

    monkeypatch.setattr(
        "xrspatial.geotiff._compression.lerc_compress", _patched,
    )
    return holder


@_gpu_only
@_lerc_required_1896
def test_gds_chunked_lerc_mask_matches_eager_1896(
        tmp_path, lerc_writer_with_mask_1896):
    """Chunked GDS path restores LERC-masked pixels to the nodata sentinel.

    Before #1896, ``_decode_window_gpu_direct`` dropped ``masked_fill``,
    so masked pixels read back at LERC's zero fill rather than NaN.
    The eager GPU path resolves and forwards ``masked_fill``; this
    test pins the chunked path to the same behaviour.
    """
    from xrspatial.geotiff import _read_geotiff_gpu
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds
    from xrspatial.geotiff._writer import write

    arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    invalid_positions = {(0, 1), (3, 3), (7, 7)}

    def invalid_pred(a):
        m = np.zeros(a.shape[:2], dtype=bool)
        for r, c in invalid_positions:
            m[r, c] = True
        return m
    lerc_writer_with_mask_1896["invalid"] = invalid_pred

    path = str(tmp_path / "lerc_gds_chunked_1896.tif")
    write(arr, path, compression="lerc", tiled=True, tile_size=8,
          nodata=float("nan"))

    eager = _read_geotiff_gpu(
        path, on_gpu_failure='strict',
        allow_experimental_codecs=True,
    ).data.get()

    ifd, geo_info, header = _parse_for_gds_1896(path)
    chunked_da = _read_geotiff_gpu_chunked_gds(
        path, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None,
    )
    chunked = chunked_da.data.compute().get()

    for (r, c) in invalid_positions:
        assert np.isnan(eager[r, c]), (
            "eager path should NaN-mask invalid pixels"
        )
        assert np.isnan(chunked[r, c]), (
            f"chunked GDS path left ({r},{c}) at LERC zero fill "
            f"{chunked[r, c]!r}; expected NaN")

    eager_valid = np.where(np.isnan(eager), 0.0, eager)
    chunked_valid = np.where(np.isnan(chunked), 0.0, chunked)
    np.testing.assert_array_equal(eager_valid, chunked_valid)


@_gpu_only
@_lerc_required_1896
def test_gds_chunked_lerc_mask_sentinel_nodata_1896(
        tmp_path, lerc_writer_with_mask_1896):
    """Sentinel nodata (-9999) on float LERC: chunked path matches eager."""
    from xrspatial.geotiff import _read_geotiff_gpu
    from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu_chunked_gds
    from xrspatial.geotiff._writer import write

    arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    invalid_positions = {(0, 1), (5, 4)}

    def invalid_pred(a):
        m = np.zeros(a.shape[:2], dtype=bool)
        for r, c in invalid_positions:
            m[r, c] = True
        return m
    lerc_writer_with_mask_1896["invalid"] = invalid_pred

    path = str(tmp_path / "lerc_gds_chunked_sentinel_1896.tif")
    write(arr, path, compression="lerc", tiled=True, tile_size=8,
          nodata=-9999.0)

    # The backend default is unmasked (#2976); request masking on both
    # the eager and chunked GPU reads so the sentinel promotes to NaN.
    eager = _read_geotiff_gpu(
        path, on_gpu_failure='strict',
        allow_experimental_codecs=True, mask_nodata=True,
    ).data.get()

    ifd, geo_info, header = _parse_for_gds_1896(path)
    chunked_da = _read_geotiff_gpu_chunked_gds(
        path, ifd, geo_info, header,
        dtype=None, chunks=4, window=None, band=None,
        name=None, max_pixels=None, mask_nodata=True,
    )
    chunked = chunked_da.data.compute().get()

    for (r, c) in invalid_positions:
        assert np.isnan(eager[r, c])
        assert np.isnan(chunked[r, c])
    np.testing.assert_array_equal(
        np.where(np.isnan(eager), 0.0, eager),
        np.where(np.isnan(chunked), 0.0, chunked),
    )


# =====================================================================
# Section 1552 -- batched D2H transfer on the GDS fallback
# =====================================================================
#
# ``gpu_decode_tiles_from_file`` previously did one
# ``.get()`` per tile when GDS read succeeded but nvCOMP could not
# decompress on device (LZW or non-ZSTD/non-deflate codecs). Each
# ``.get()`` was an independent D2H copy on the default stream, so
# the transfers serialised and per-DMA setup overhead dominated wall
# time. The fix concatenates all device buffers into one cupy array,
# runs a single ``.get()``, and slices the host bytes by per-tile
# offsets.


def test_batched_d2h_empty_list_1552():
    """Empty input must return an empty list without touching cupy."""
    from xrspatial.geotiff._gpu_decode import _batched_d2h_to_bytes

    assert _batched_d2h_to_bytes([]) == []


@_gpu_only
def test_batched_d2h_matches_per_tile_get_1552():
    """Batched D2H output must equal the per-tile ``.get().tobytes()`` baseline."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _batched_d2h_to_bytes

    rng = np.random.default_rng(seed=1552)

    host_tiles = [
        rng.integers(0, 256, size=n, dtype=np.uint8)
        for n in [1, 7, 64, 4096, 1, 65537, 256]
    ]
    d_tiles = [cupy.asarray(t) for t in host_tiles]

    expected = [t.get().tobytes() for t in d_tiles]
    actual = _batched_d2h_to_bytes(d_tiles)

    assert len(actual) == len(expected)
    for i, (got, want) in enumerate(zip(actual, expected)):
        assert isinstance(got, bytes), f"tile {i}: not bytes ({type(got)})"
        assert got == want, f"tile {i} mismatch"


@_gpu_only
def test_batched_d2h_single_tile_1552():
    """Single-tile input is a degenerate case worth covering explicitly."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _batched_d2h_to_bytes

    payload = bytes(range(64))
    d_tiles = [cupy.asarray(np.frombuffer(payload, dtype=np.uint8))]
    out = _batched_d2h_to_bytes(d_tiles)

    assert len(out) == 1
    assert out[0] == payload


@_gpu_only
def test_batched_d2h_zero_size_tile_in_list_1552():
    """A zero-sized tile mixed with real tiles must round-trip cleanly."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _batched_d2h_to_bytes

    real = np.array([10, 20, 30], dtype=np.uint8)
    empty = np.zeros(0, dtype=np.uint8)
    d_tiles = [
        cupy.asarray(real), cupy.asarray(empty), cupy.asarray(real[::-1])
    ]

    out = _batched_d2h_to_bytes(d_tiles)

    assert out[0] == real.tobytes()
    assert out[1] == b''
    assert out[2] == real[::-1].tobytes()


@_gpu_only
def test_batched_d2h_checks_gpu_memory_before_concat_1552(monkeypatch):
    """The concat allocates sum(sizes) bytes; the guard must fire before it."""
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


@_gpu_only
def test_batched_d2h_many_small_tiles_1552():
    """Many tiles is the regime the batching speedup actually targets."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _batched_d2h_to_bytes

    rng = np.random.default_rng(seed=42)
    host_tiles = [
        rng.integers(0, 256, size=128, dtype=np.uint8) for _ in range(256)
    ]
    d_tiles = [cupy.asarray(t) for t in host_tiles]

    out = _batched_d2h_to_bytes(d_tiles)

    assert len(out) == 256
    for i, (got, src) in enumerate(zip(out, host_tiles)):
        assert got == src.tobytes(), f"tile {i} mismatch"


# =====================================================================
# Section orientation -- TIFF Orientation tag on GPU reads
# =====================================================================
#
# The CPU reader applies the Orientation tag (274) post-decode so
# pixel (0, 0) is always the visual top-left. The GPU read path used
# to skip this remap, so reads of any file with orientation != 1
# returned a different pixel buffer than the CPU reader.


# tifffile is required for the orientation section. Gate per-test
# rather than at module scope: a module-level importorskip would skip
# every section in this consolidated file when tifffile is missing.
_TIFFFILE_INSTALLED = importlib.util.find_spec("tifffile") is not None
_tifffile_required = pytest.mark.skipif(
    not _TIFFFILE_INSTALLED, reason="tifffile required",
)

_ORIENTATIONS_GPU = [1, 2, 3, 4, 5, 6, 7, 8]


def _expected_for_orientation_gpu(stored, orientation):
    """Spec-defined remap so tests don't depend on the production helper."""
    if orientation == 1:
        return stored
    if orientation == 2:
        return stored[:, ::-1]
    if orientation == 3:
        return stored[::-1, ::-1]
    if orientation == 4:
        return stored[::-1, :]
    if orientation == 5:
        return stored.T if stored.ndim == 2 else stored.transpose(1, 0, 2)
    if orientation == 6:
        return (stored.T[:, ::-1] if stored.ndim == 2
                else stored.transpose(1, 0, 2)[:, ::-1])
    if orientation == 7:
        return (stored.T[::-1, ::-1] if stored.ndim == 2
                else stored.transpose(1, 0, 2)[::-1, ::-1])
    if orientation == 8:
        return (stored.T[::-1, :] if stored.ndim == 2
                else stored.transpose(1, 0, 2)[::-1, :])
    raise AssertionError(orientation)


def _write_tiled_gpu(path, arr, orientation, tile=(16, 16), extra=None):
    """Write *arr* tiled with the requested Orientation tag."""
    import tifffile

    extras = [(274, 'H', 1, orientation, True)]
    if extra:
        extras.extend(extra)
    tifffile.imwrite(str(path), arr, tile=tile, extratags=extras)


def _write_stripped_gpu(path, arr, orientation, extra=None):
    """Write *arr* stripped (no tile=) with the requested Orientation tag."""
    import tifffile

    extras = [(274, 'H', 1, orientation, True)]
    if extra:
        extras.extend(extra)
    tifffile.imwrite(str(path), arr, extratags=extras)


@_gpu_only
@_tifffile_required
@pytest.mark.parametrize("orientation", _ORIENTATIONS_GPU)
def test_gpu_tiled_matches_cpu_orientation(tmp_path, orientation):
    """Tiled GPU read of every orientation matches the spec-remapped buffer."""
    from xrspatial.geotiff import _read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / f"gpu_orient_{orientation}.tif"
    _write_tiled_gpu(path, arr, orientation)

    da = _read_geotiff_gpu(str(path))
    expected = _expected_for_orientation_gpu(arr, orientation)
    np.testing.assert_array_equal(da.data.get(), expected)


@_gpu_only
@_tifffile_required
@pytest.mark.parametrize("orientation", _ORIENTATIONS_GPU)
def test_gpu_stripped_matches_cpu_orientation(tmp_path, orientation):
    """Stripped GPU read also applies orientation (via CPU fallback path)."""
    from xrspatial.geotiff import _read_geotiff_gpu

    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    path = tmp_path / f"gpu_strip_orient_{orientation}.tif"
    _write_stripped_gpu(path, arr, orientation)

    da = _read_geotiff_gpu(str(path))
    expected = _expected_for_orientation_gpu(arr, orientation)
    np.testing.assert_array_equal(da.data.get(), expected)


@_gpu_only
@_tifffile_required
@pytest.mark.parametrize("orientation", _ORIENTATIONS_GPU)
def test_gpu_3band_tiled_matches_cpu_orientation(tmp_path, orientation):
    """3-band tiled (planar=2) read also applies orientation per band."""
    import tifffile

    from xrspatial.geotiff import _read_geotiff_gpu

    rgb = np.arange(3 * 16 * 16, dtype=np.uint8).reshape(3, 16, 16)
    path = tmp_path / f"gpu_rgb_orient_{orientation}.tif"
    tifffile.imwrite(
        str(path), rgb, photometric='rgb', tile=(16, 16),
        extratags=[(274, 'H', 1, orientation, True)],
    )

    da = _read_geotiff_gpu(str(path))
    stored = np.transpose(rgb, (1, 2, 0))
    expected = _expected_for_orientation_gpu(stored, orientation)
    np.testing.assert_array_equal(da.data.get(), expected)


@_gpu_only
@_tifffile_required
@pytest.mark.parametrize("orientation", [2, 3, 4])
def test_gpu_orient_2_3_4_coords_track_pixel_flip(tmp_path, orientation):
    """For mirror-flip orientations, GPU coord array also flips."""
    from xrspatial.geotiff import _read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / f"gpu_orient_geo_{orientation}.tif"
    _write_tiled_gpu(
        path, arr, orientation,
        extra=[
            (33550, 'd', 3, (1.0, 1.0, 0.0), True),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 100.0, 50.0, 0.0), True),
            (34735, 'H', 12, (
                1, 1, 0, 2,
                1024, 0, 1, 2,
                2048, 0, 1, 4326,
            ), True),
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        da = _read_geotiff_gpu(str(path))

    targets = [
        (100.5, 49.5, 0),
        (115.5, 49.5, 15),
        (100.5, 34.5, 240),
        (115.5, 34.5, 255),
    ]
    for x, y, expected in targets:
        got = int(da.sel(x=x, y=y).item())
        assert got == expected, (
            f"orient={orientation}: GPU sel(x={x}, y={y})={got}, "
            f"expected {expected}"
        )


@_gpu_only
@_tifffile_required
def test_gpu_default_orientation_unchanged(tmp_path):
    """Files without Orientation tag still decode unchanged on GPU."""
    import tifffile

    from xrspatial.geotiff import _read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / "gpu_no_orient.tif"
    tifffile.imwrite(str(path), arr, tile=(16, 16))

    da = _read_geotiff_gpu(str(path))
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
@_tifffile_required
@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_gpu_orientation_5_to_8_raise_on_georef(tmp_path, orientation):
    """GPU reader refuses georef'd files with axis-swap orientations.

    Mirrors the CPU behaviour added for issue #1765.
    ``_read_geotiff_gpu`` used to warn and return silently wrong x/y
    coords for these cases.
    """
    from xrspatial.geotiff import _read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / f"gpu_orient_georef_raise_1765_{orientation}.tif"
    _write_tiled_gpu(
        path, arr, orientation,
        extra=[
            (33550, 'd', 3, (1.0, 1.0, 0.0), True),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 100.0, 50.0, 0.0), True),
            (34735, 'H', 12, (
                1, 1, 0, 2,
                1024, 0, 1, 2,
                2048, 0, 1, 4326,
            ), True),
        ],
    )

    with pytest.raises(NotImplementedError, match=str(orientation)):
        _read_geotiff_gpu(str(path))


@_gpu_only
@_tifffile_required
@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_gpu_orientation_5_to_8_transform_only_raises(tmp_path, orientation):
    """``has_georef`` without CRS still triggers the raise on GPU."""
    from xrspatial.geotiff import _read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / f"gpu_orient_transform_only_1765_{orientation}.tif"
    _write_tiled_gpu(
        path, arr, orientation,
        extra=[
            (33550, 'd', 3, (1.0, 1.0, 0.0), True),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 100.0, 50.0, 0.0), True),
        ],
    )

    with pytest.raises(NotImplementedError, match=str(orientation)):
        _read_geotiff_gpu(str(path))


@_gpu_only
@_tifffile_required
@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_gpu_orientation_5_to_8_no_georef_still_swaps(tmp_path, orientation):
    """Without any geo tags, GPU orientation 5-8 still swaps axes."""
    from xrspatial.geotiff import _read_geotiff_gpu

    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / f"gpu_orient_no_geo_1765_{orientation}.tif"
    _write_tiled_gpu(path, arr, orientation)

    da = _read_geotiff_gpu(str(path))
    expected = _expected_for_orientation_gpu(arr, orientation)
    assert da.data.shape == expected.shape
    np.testing.assert_array_equal(da.data.get(), expected)


# =====================================================================
# Section 1776 -- size-param validation on GPU + VRT entry points
# =====================================================================
#
# ``tile_size`` validation on ``to_geotiff`` and ``chunks`` validation
# on ``_read_geotiff_dask`` landed first. The matching kwargs
# on three sibling entry points were left unchecked:
# - ``_write_geotiff_gpu(tile_size=)``
# - ``_read_geotiff_gpu(chunks=)`` / ``_read_vrt(chunks=)``
#
# The fix factors the shared validators ``_validate_tile_size_arg`` and
# ``_validate_chunks_arg`` and calls them up front from each entry
# point, so all four read paths and both write paths emit the same
# error format.


def _make_tif_1776(tmp_path) -> str:
    """Write a 10x10 float32 GeoTIFF and return its path."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(10), 'x': np.arange(10)},
        attrs={'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)},
    )
    path = os.path.join(str(tmp_path), 'src_1776.tif')
    to_geotiff(da, path)
    return path


def _make_vrt_1776(tmp_path) -> str:
    """Write a 10x10 GeoTIFF plus a single-source VRT and return the .vrt path."""
    from xrspatial.geotiff import _build_vrt

    tif = _make_tif_1776(tmp_path)
    vrt = os.path.join(str(tmp_path), 'src_1776.vrt')
    _build_vrt(vrt, [tif])
    return vrt


@_gpu_only
class TestWriteGeotiffGpuTileSize_1776:
    """Mirror ``test_size_param_validation_1752`` for ``_write_geotiff_gpu``."""

    @pytest.fixture
    def gpu_da(self):
        import cupy
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        return xr.DataArray(cupy.asarray(arr), dims=['y', 'x'])

    def test_tile_size_zero_raises(self, gpu_da, tmp_path):
        from xrspatial.geotiff import _write_geotiff_gpu
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        with pytest.raises(ValueError, match='tile_size'):
            _write_geotiff_gpu(gpu_da, out, tile_size=0)

    def test_tile_size_negative_raises(self, gpu_da, tmp_path):
        from xrspatial.geotiff import _write_geotiff_gpu
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        with pytest.raises(ValueError, match='tile_size'):
            _write_geotiff_gpu(gpu_da, out, tile_size=-1)

    def test_tile_size_float_raises(self, gpu_da, tmp_path):
        from xrspatial.geotiff import _write_geotiff_gpu
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        with pytest.raises(ValueError, match='tile_size'):
            _write_geotiff_gpu(gpu_da, out, tile_size=256.0)

    def test_tile_size_bool_true_raises(self, gpu_da, tmp_path):
        """``tile_size=True`` is an int subclass that used to silently
        write a 1x1-tile file. Reject it with a clear ValueError."""
        from xrspatial.geotiff import _write_geotiff_gpu
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        with pytest.raises(ValueError, match='tile_size'):
            _write_geotiff_gpu(gpu_da, out, tile_size=True)

    def test_tile_size_bool_false_raises(self, gpu_da, tmp_path):
        """``tile_size=False`` was the worst case: ``False == 0`` slipped
        through the integer check and hit ZeroDivisionError downstream."""
        from xrspatial.geotiff import _write_geotiff_gpu
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        with pytest.raises(ValueError, match='tile_size'):
            _write_geotiff_gpu(gpu_da, out, tile_size=False)

    def test_tile_size_positive_works(self, gpu_da, tmp_path):
        """tile_size=16 still round-trips."""
        from xrspatial.geotiff import _write_geotiff_gpu
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        _write_geotiff_gpu(gpu_da, out, tile_size=16)
        assert os.path.exists(out)

    def test_tile_size_numpy_int_scalar_works(self, gpu_da, tmp_path):
        """``np.int64(N)`` is accepted."""
        from xrspatial.geotiff import _write_geotiff_gpu
        out = os.path.join(str(tmp_path), 'out_1776.tif')
        _write_geotiff_gpu(gpu_da, out, tile_size=np.int64(256))
        assert os.path.exists(out)


@_gpu_only
class TestReadGeotiffGpuChunks_1776:
    """Mirror ``test_size_param_validation_1752`` for ``_read_geotiff_gpu``."""

    def test_chunks_zero_raises(self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_geotiff_gpu(path, chunks=0)

    def test_chunks_negative_raises(self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_geotiff_gpu(path, chunks=-1)

    def test_chunks_tuple_zero_row_raises(self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_geotiff_gpu(path, chunks=(0, 256))

    def test_chunks_tuple_negative_col_raises(self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_geotiff_gpu(path, chunks=(256, -1))

    def test_chunks_tuple_wrong_length_raises(self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_geotiff_gpu(path, chunks=(64, 64, 64))

    def test_chunks_bool_raises(self, tmp_path):
        """``chunks=True``/``False`` are int subclasses that used to slip
        through. Reject them with the same error as a non-int scalar."""
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_geotiff_gpu(path, chunks=True)

    def test_chunks_non_int_raises(self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_geotiff_gpu(path, chunks='256')

    def test_chunks_tuple_float_raises(self, tmp_path):
        """Tuple entries that aren't int should reject too."""
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_geotiff_gpu(path, chunks=(64, 64.5))

    def test_positive_int_chunks_works(self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        r = _read_geotiff_gpu(path, chunks=64)
        assert r.shape == (10, 10)

    def test_positive_tuple_chunks_works(self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        r = _read_geotiff_gpu(path, chunks=(4, 8))
        assert r.shape == (10, 10)

    def test_numpy_int_scalar_chunks_works(self, tmp_path):
        """``np.int64(N)`` scalar chunk size is accepted."""
        from xrspatial.geotiff import _read_geotiff_gpu
        path = _make_tif_1776(tmp_path)
        r = _read_geotiff_gpu(path, chunks=np.int64(64))
        assert r.shape == (10, 10)


class TestReadVrtChunks_1776:
    """Same matrix as ``TestReadGeotiffGpuChunks_1776`` but for the VRT
    entry point. Runs without CUDA because ``_read_vrt(chunks=)``
    returns a Dask-backed numpy DataArray; no GPU is required.
    """

    def test_chunks_zero_raises(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_vrt(vrt, chunks=0)

    def test_chunks_negative_raises(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_vrt(vrt, chunks=-1)

    def test_chunks_tuple_zero_row_raises(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_vrt(vrt, chunks=(0, 256))

    def test_chunks_tuple_negative_col_raises(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_vrt(vrt, chunks=(256, -1))

    def test_chunks_tuple_wrong_length_raises(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_vrt(vrt, chunks=(64, 64, 64))

    def test_chunks_bool_raises(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_vrt(vrt, chunks=True)

    def test_chunks_non_int_raises(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_vrt(vrt, chunks='256')

    def test_chunks_tuple_float_raises(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_vrt(vrt, chunks=(64, 64.5))

    def test_positive_int_chunks_works(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        r = _read_vrt(vrt, chunks=64)
        assert r.shape == (10, 10)

    def test_positive_tuple_chunks_works(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        r = _read_vrt(vrt, chunks=(4, 8))
        assert r.shape == (10, 10)

    def test_numpy_int_scalar_chunks_works(self, tmp_path):
        from xrspatial.geotiff import _read_vrt
        vrt = _make_vrt_1776(tmp_path)
        r = _read_vrt(vrt, chunks=np.int64(64))
        assert r.shape == (10, 10)


@_gpu_only
class TestOpenGeotiffGpuChunksDispatch_1776:
    """``open_geotiff(gpu=True, chunks=X)`` routes through
    ``_read_geotiff_gpu``; pin that the validation propagates through
    the dispatcher.
    """

    def test_open_geotiff_gpu_chunks_zero_raises(self, tmp_path):
        from xrspatial.geotiff import open_geotiff
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            open_geotiff(path, gpu=True, chunks=0)

    def test_open_geotiff_gpu_chunks_negative_raises(self, tmp_path):
        from xrspatial.geotiff import open_geotiff
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            open_geotiff(path, gpu=True, chunks=-1)

    def test_open_geotiff_gpu_chunks_tuple_zero_raises(self, tmp_path):
        from xrspatial.geotiff import open_geotiff
        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            open_geotiff(path, gpu=True, chunks=(0, 256))


class TestToGeotiffGpuTileSizeAlreadyChecked_1776:
    """``to_geotiff(gpu=True, tile_size=0)`` was already validated by
    #1752's CPU-side check before dispatching to ``_write_geotiff_gpu``.
    """

    @_gpu_only
    def test_to_geotiff_gpu_tile_size_zero_raises_same_message(
            self, tmp_path):
        import cupy

        from xrspatial.geotiff import to_geotiff

        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        da = xr.DataArray(cupy.asarray(arr), dims=['y', 'x'])
        out = os.path.join(str(tmp_path), 'out.tif')
        with pytest.raises(ValueError, match='tile_size'):
            to_geotiff(da, out, gpu=True, tile_size=0)


@_gpu_only
class TestNoDoubleValidationSideEffects_1776:
    """The factored ``_validate_chunks_arg`` returns the coerced int
    when given an ``np.integer`` scalar.
    """

    def test_read_geotiff_gpu_chunks_numpy_int_no_side_effect(
            self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu

        path = _make_tif_1776(tmp_path)
        r = _read_geotiff_gpu(path, chunks=np.int64(64))
        assert r.shape == (10, 10)
        out = r.data
        if hasattr(out, 'compute'):
            out.compute()


class TestChunksNoneAcrossEntryPoints_1776:
    """``_read_geotiff_gpu`` and ``_read_vrt`` default to ``chunks=None``
    (eager read), so the factored ``_validate_chunks_arg`` must let
    ``None`` through for them. ``_read_geotiff_dask`` does not accept
    ``None`` -- its chunk-unpacking math would otherwise fail with a
    confusing ``TypeError`` instead of a parameter-named ``ValueError``.
    """

    def test_read_geotiff_dask_chunks_none_raises_value_error(
            self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_dask

        path = _make_tif_1776(tmp_path)
        with pytest.raises(ValueError, match='chunks'):
            _read_geotiff_dask(path, chunks=None)

    def test_read_vrt_chunks_none_returns_eager(self, tmp_path):
        from xrspatial.geotiff import _read_vrt

        vrt = _make_vrt_1776(tmp_path)
        r = _read_vrt(vrt, chunks=None)
        assert r.shape == (10, 10)

    @_gpu_only
    def test_read_geotiff_gpu_chunks_none_returns_eager(self, tmp_path):
        from xrspatial.geotiff import _read_geotiff_gpu

        path = _make_tif_1776(tmp_path)
        r = _read_geotiff_gpu(path, chunks=None)
        assert r.shape == (10, 10)


# =====================================================================
# Section 2052 -- ``mask_nodata=False`` on GPU + VRT entry points
# =====================================================================
#
# The ``mask_nodata`` kwarg is wired through every
# public reader entry point in ``xrspatial.geotiff``
# (``open_geotiff``, ``_read_geotiff_gpu``, ``_read_geotiff_dask``,
# ``_read_vrt``). The original test module
# ``test_mask_nodata_kwarg_2052.py`` only exercises eager-numpy and
# dask+numpy; this section closes the GPU, dask+GPU, and VRT (eager +
# dask) gap.


_SENTINEL_POS_2052 = [(0, 0), (1, 2), (2, 3), (3, 0)]


@pytest.fixture
def uint16_with_matching_sentinel_2052(tmp_path):
    """uint16 TIFF where nodata=0 and the array has zeros in it.

    Identical layout to the eager fixture in the original
    ``test_mask_nodata_kwarg_2052.py`` so the cross-backend comparison
    is straightforward.
    """
    from xrspatial.geotiff._writer import write

    arr = np.array(
        [[0, 100, 200, 300],
         [400, 500, 0, 600],
         [700, 800, 900, 0],
         [0, 1100, 1200, 1300]],
        dtype=np.uint16,
    )
    path = str(tmp_path / "uint16_match_2052_gpu_vrt.tif")
    write(arr, path, nodata=0, compression="deflate", tiled=True,
          tile_size=256)
    return path, arr


@pytest.fixture
def uint16_vrt_with_matching_sentinel_2052(tmp_path):
    """A single-source VRT wrapping the uint16 fixture above."""
    from xrspatial.geotiff import _build_vrt
    from xrspatial.geotiff._writer import write

    arr = np.array(
        [[0, 100, 200, 300],
         [400, 500, 0, 600],
         [700, 800, 900, 0],
         [0, 1100, 1200, 1300]],
        dtype=np.uint16,
    )
    tif_path = str(tmp_path / "uint16_match_2052_vrt_src.tif")
    write(arr, tif_path, nodata=0, compression="none", tiled=False)

    vrt_path = str(tmp_path / "uint16_match_2052.vrt")
    _build_vrt(vrt_path, [tif_path])
    return vrt_path, arr


@_gpu_only
def test_read_geotiff_gpu_mask_nodata_false_preserves_uint16_2052(
        uint16_with_matching_sentinel_2052):
    """``_read_geotiff_gpu(mask_nodata=False)`` keeps the uint16 dtype."""
    from xrspatial.geotiff import _read_geotiff_gpu

    path, arr = uint16_with_matching_sentinel_2052
    da = _read_geotiff_gpu(path, mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.data.get(), arr)
    assert da.attrs["nodata"] == 0


@_gpu_only
def test_read_geotiff_gpu_default_unmasked_keeps_uint16_2976(
        uint16_with_matching_sentinel_2052):
    """The GPU default matches ``open_geotiff`` (unmasked): a bare
    ``_read_geotiff_gpu(path)`` keeps the source dtype and sentinel."""
    import cupy

    from xrspatial.geotiff import _read_geotiff_gpu

    path, arr = uint16_with_matching_sentinel_2052
    da = _read_geotiff_gpu(path)

    assert da.dtype == np.uint16
    assert int(cupy.isnan(da.data.astype(cupy.float64)).sum().get()) == 0
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_open_geotiff_gpu_mask_nodata_false_threads_through_2052(
        uint16_with_matching_sentinel_2052):
    """``open_geotiff(gpu=True, mask_nodata=False)`` reaches the GPU
    reader's opt-out branch with the kwarg intact.
    """
    from xrspatial.geotiff import open_geotiff

    path, arr = uint16_with_matching_sentinel_2052
    da = open_geotiff(path, gpu=True, masked=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_read_geotiff_gpu_dtype_uint16_with_mask_nodata_false_2052(
        uint16_with_matching_sentinel_2052):
    """``dtype="uint16"`` works on the GPU branch when the opt-out is set."""
    from xrspatial.geotiff import _read_geotiff_gpu

    path, arr = uint16_with_matching_sentinel_2052
    da = _read_geotiff_gpu(path, dtype="uint16", mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_read_geotiff_gpu_dask_mask_nodata_false_preserves_uint16_2052(
        uint16_with_matching_sentinel_2052):
    """``_read_geotiff_gpu(chunks=..., mask_nodata=False)`` keeps uint16."""
    from xrspatial.geotiff import _read_geotiff_gpu

    path, arr = uint16_with_matching_sentinel_2052
    da = _read_geotiff_gpu(path, chunks=2, mask_nodata=False)

    assert da.dtype == np.uint16

    computed = da.compute()
    assert computed.dtype == np.uint16
    np.testing.assert_array_equal(computed.data.get(), arr)


@_gpu_only
def test_read_geotiff_gpu_dask_default_unmasked_keeps_uint16_2976(
        uint16_with_matching_sentinel_2052):
    """The dask+GPU default matches ``open_geotiff`` (unmasked): the
    graph dtype stays uint16 and the sentinel survives."""
    import cupy

    from xrspatial.geotiff import _read_geotiff_gpu

    path, arr = uint16_with_matching_sentinel_2052
    da = _read_geotiff_gpu(path, chunks=2)

    assert da.dtype == np.uint16
    computed = da.compute()
    assert computed.dtype == np.uint16
    assert int(cupy.isnan(
        computed.data.astype(cupy.float64)).sum().get()) == 0
    np.testing.assert_array_equal(computed.data.get(), arr)


@_gpu_only
def test_open_geotiff_dask_gpu_mask_nodata_false_threads_through_2052(
        uint16_with_matching_sentinel_2052):
    """``open_geotiff(gpu=True, chunks=N, mask_nodata=False)`` flows
    the kwarg all the way through the dispatcher + dask-graph builder
    to the cupy chunk reader."""
    from xrspatial.geotiff import open_geotiff

    path, arr = uint16_with_matching_sentinel_2052
    da = open_geotiff(path, gpu=True, chunks=2, masked=False)

    assert da.dtype == np.uint16
    computed = da.compute()
    np.testing.assert_array_equal(computed.data.get(), arr)


def test_read_vrt_mask_nodata_false_preserves_uint16_2052(
        uint16_vrt_with_matching_sentinel_2052):
    """``_read_vrt(mask_nodata=False)`` keeps the uint16 dtype on the
    eager VRT path."""
    from xrspatial.geotiff import _read_vrt

    vrt_path, arr = uint16_vrt_with_matching_sentinel_2052
    da = _read_vrt(vrt_path, mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(da.values), arr)
    assert da.attrs["nodata"] == 0


def test_read_vrt_default_unmasked_keeps_uint16_2976(
        uint16_vrt_with_matching_sentinel_2052):
    """The VRT default matches ``open_geotiff`` (unmasked): a bare
    ``_read_vrt(path)`` keeps the source dtype and sentinel."""
    from xrspatial.geotiff import _read_vrt

    vrt_path, arr = uint16_vrt_with_matching_sentinel_2052
    da = _read_vrt(vrt_path)

    assert da.dtype == np.uint16
    assert int(np.isnan(np.asarray(da.values, dtype=float)).sum()) == 0
    np.testing.assert_array_equal(np.asarray(da.values), arr)


def test_open_geotiff_vrt_mask_nodata_false_threads_through_2052(
        uint16_vrt_with_matching_sentinel_2052):
    """``open_geotiff(<vrt>, mask_nodata=False)`` reaches the VRT
    reader's opt-out branch via the dispatcher."""
    from xrspatial.geotiff import open_geotiff

    vrt_path, arr = uint16_vrt_with_matching_sentinel_2052
    da = open_geotiff(vrt_path, masked=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(da.values), arr)


def test_read_vrt_dtype_uint16_with_mask_nodata_false_2052(
        uint16_vrt_with_matching_sentinel_2052):
    """``dtype="uint16"`` works on the VRT path with the opt-out."""
    from xrspatial.geotiff import _read_vrt

    vrt_path, arr = uint16_vrt_with_matching_sentinel_2052
    da = _read_vrt(vrt_path, dtype="uint16", mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(da.values), arr)


def test_read_vrt_chunked_mask_nodata_false_preserves_uint16_2052(
        uint16_vrt_with_matching_sentinel_2052):
    """``_read_vrt(chunks=..., mask_nodata=False)`` keeps uint16 on
    the dask+VRT path."""
    from xrspatial.geotiff import _read_vrt

    vrt_path, arr = uint16_vrt_with_matching_sentinel_2052
    da = _read_vrt(vrt_path, chunks=2, mask_nodata=False)

    assert da.dtype == np.uint16
    computed = da.compute()
    assert computed.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(computed.values), arr)


def test_read_vrt_chunked_default_unmasked_keeps_uint16_2976(
        uint16_vrt_with_matching_sentinel_2052):
    """The chunked-VRT default matches ``open_geotiff`` (unmasked): the
    graph dtype stays uint16 and the sentinel survives."""
    from xrspatial.geotiff import _read_vrt

    vrt_path, arr = uint16_vrt_with_matching_sentinel_2052
    da = _read_vrt(vrt_path, chunks=2)

    assert da.dtype == np.uint16
    computed = da.compute()
    assert computed.dtype == np.uint16
    assert int(np.isnan(np.asarray(computed.values, dtype=float)).sum()) == 0
    np.testing.assert_array_equal(np.asarray(computed.values), arr)


def test_open_geotiff_vrt_chunked_mask_nodata_false_threads_through_2052(
        uint16_vrt_with_matching_sentinel_2052):
    """``open_geotiff(<vrt>, chunks=N, mask_nodata=False)`` reaches
    the chunked-VRT opt-out branch via the dispatcher."""
    from xrspatial.geotiff import open_geotiff

    vrt_path, arr = uint16_vrt_with_matching_sentinel_2052
    da = open_geotiff(vrt_path, chunks=2, masked=False)

    assert da.dtype == np.uint16
    computed = da.compute()
    np.testing.assert_array_equal(np.asarray(computed.values), arr)


def test_cross_backend_parity_eager_dask_numpy_2052(
        uint16_with_matching_sentinel_2052):
    """Eager-numpy and dask+numpy agree bit-exact with ``mask_nodata=False``."""
    from xrspatial.geotiff import open_geotiff

    path, arr = uint16_with_matching_sentinel_2052

    eager = open_geotiff(path, masked=False)
    dask_ = open_geotiff(path, chunks=2, masked=False).compute()

    assert eager.dtype == np.uint16
    assert dask_.dtype == np.uint16
    np.testing.assert_array_equal(eager.values, dask_.values)
    np.testing.assert_array_equal(eager.values, arr)


@_gpu_only
def test_cross_backend_parity_eager_gpu_2052(
        uint16_with_matching_sentinel_2052):
    """Eager-numpy and eager-GPU agree bit-exact with ``mask_nodata=False``."""
    from xrspatial.geotiff import open_geotiff

    path, arr = uint16_with_matching_sentinel_2052

    eager = open_geotiff(path, masked=False)
    gpu = open_geotiff(path, gpu=True, masked=False)

    assert eager.dtype == np.uint16
    assert gpu.dtype == np.uint16
    np.testing.assert_array_equal(eager.values, gpu.data.get())
    np.testing.assert_array_equal(eager.values, arr)


@_gpu_only
def test_cross_backend_parity_eager_dask_gpu_2052(
        uint16_with_matching_sentinel_2052):
    """Eager-numpy and dask+GPU agree bit-exact with ``mask_nodata=False``."""
    from xrspatial.geotiff import open_geotiff

    path, arr = uint16_with_matching_sentinel_2052

    eager = open_geotiff(path, masked=False)
    dgpu = open_geotiff(
        path, gpu=True, chunks=2, masked=False).compute()

    assert eager.dtype == np.uint16
    assert dgpu.dtype == np.uint16
    np.testing.assert_array_equal(eager.values, dgpu.data.get())
    np.testing.assert_array_equal(eager.values, arr)


def test_cross_backend_parity_eager_vrt_2052(
        uint16_with_matching_sentinel_2052,
        uint16_vrt_with_matching_sentinel_2052):
    """A single-source VRT and the underlying TIFF return identical
    arrays under ``mask_nodata=False``."""
    from xrspatial.geotiff import open_geotiff

    tif_path, arr = uint16_with_matching_sentinel_2052
    vrt_path, _ = uint16_vrt_with_matching_sentinel_2052

    eager = open_geotiff(tif_path, masked=False)
    vrt = open_geotiff(vrt_path, masked=False)

    assert eager.dtype == vrt.dtype == np.uint16
    np.testing.assert_array_equal(eager.values, np.asarray(vrt.values))
    np.testing.assert_array_equal(eager.values, arr)


def test_read_geotiff_dask_direct_mask_nodata_false_2052(
        uint16_with_matching_sentinel_2052):
    """The direct ``_read_geotiff_dask`` entry point also accepts the kwarg."""
    from xrspatial.geotiff import _read_geotiff_dask

    path, arr = uint16_with_matching_sentinel_2052
    da = _read_geotiff_dask(path, chunks=2, mask_nodata=False)

    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.compute().values, arr)


# =====================================================================
# Section 1615 -- ``on_gpu_failure`` threads through ``open_geotiff``
# =====================================================================
#
# ``open_geotiff(gpu=True)`` used to silently drop the
# ``on_gpu_failure`` kwarg: the dispatcher did not declare it and the
# GPU branch did not forward it to ``_read_geotiff_gpu``. Callers that
# wanted strict GPU failure semantics had to bypass ``open_geotiff``
# entirely and call ``_read_geotiff_gpu`` directly, defeating the
# dispatcher.


@pytest.fixture
def small_tiff_path_1615(tmp_path):
    """4x6 single-band tiled tiff usable by both CPU and GPU readers."""
    from xrspatial.geotiff import to_geotiff

    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'on_gpu_failure_1615.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


def test_open_geotiff_signature_includes_on_gpu_failure_1615():
    """Direct ``inspect.signature`` check on the public dispatcher."""
    from xrspatial.geotiff import open_geotiff

    sig = inspect.signature(open_geotiff)
    assert 'on_gpu_failure' in sig.parameters, (
        "open_geotiff must declare on_gpu_failure so the GPU policy is "
        "addressable through the dispatcher entry point"
    )


def test_on_gpu_failure_with_gpu_false_raises_value_error_1615(
        small_tiff_path_1615):
    """Refuse rather than silently dropping the kwarg on the CPU path."""
    from xrspatial.geotiff import open_geotiff

    path, _ = small_tiff_path_1615
    with pytest.raises(ValueError, match="on_gpu_failure only applies"):
        open_geotiff(path, on_gpu_failure='strict')


def test_on_gpu_failure_with_explicit_gpu_false_raises_1615(
        small_tiff_path_1615):
    """``gpu=False`` explicitly is rejected just like the default."""
    from xrspatial.geotiff import open_geotiff

    path, _ = small_tiff_path_1615
    with pytest.raises(ValueError, match="on_gpu_failure only applies"):
        open_geotiff(path, gpu=False, on_gpu_failure='auto')


def test_on_gpu_failure_with_chunks_only_raises_1615(small_tiff_path_1615):
    """Dask CPU path is not GPU and should refuse the kwarg."""
    from xrspatial.geotiff import open_geotiff

    path, _ = small_tiff_path_1615
    with pytest.raises(ValueError, match="on_gpu_failure only applies"):
        open_geotiff(path, chunks=2, on_gpu_failure='auto')


def test_default_dispatch_unchanged_cpu_1615(small_tiff_path_1615):
    """Not passing ``on_gpu_failure`` keeps CPU dispatch byte-stable."""
    from xrspatial.geotiff import open_geotiff

    path, arr = small_tiff_path_1615
    da = open_geotiff(path)
    np.testing.assert_array_equal(da.values, arr)


def test_default_dispatch_unchanged_dask_1615(small_tiff_path_1615):
    """Not passing ``on_gpu_failure`` keeps Dask CPU dispatch byte-stable."""
    from xrspatial.geotiff import open_geotiff

    path, arr = small_tiff_path_1615
    da = open_geotiff(path, chunks=2)
    np.testing.assert_array_equal(da.values, arr)


@_gpu_only
def test_open_geotiff_gpu_forwards_on_gpu_failure_auto_1615(
        small_tiff_path_1615):
    """``open_geotiff(gpu=True, on_gpu_failure='auto')`` works end-to-end."""
    from xrspatial.geotiff import open_geotiff

    path, arr = small_tiff_path_1615
    da = open_geotiff(path, gpu=True, on_gpu_failure='auto')
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_open_geotiff_gpu_forwards_on_gpu_failure_strict_1615(
        small_tiff_path_1615):
    """``on_gpu_failure='strict'`` also reaches the GPU pipeline."""
    from xrspatial.geotiff import open_geotiff

    path, arr = small_tiff_path_1615
    da = open_geotiff(path, gpu=True, on_gpu_failure='strict')
    np.testing.assert_array_equal(da.data.get(), arr)


@_gpu_only
def test_open_geotiff_gpu_rejects_invalid_on_gpu_failure_1615(
        small_tiff_path_1615):
    """An invalid value still surfaces from the underlying validator."""
    from xrspatial.geotiff import open_geotiff

    path, _ = small_tiff_path_1615
    with pytest.raises(ValueError, match="on_gpu_failure must be"):
        open_geotiff(path, gpu=True, on_gpu_failure='loose')


def test_invalid_on_gpu_failure_reaches_gpu_validator_on_cpu_1615(
        small_tiff_path_1615):
    """Even on a CPU-only host, the kwarg should reach
    ``_read_geotiff_gpu``.

    Without GPU hardware, ``_read_geotiff_gpu`` raises ``ImportError``
    (no cupy) or runs through to the actual decode. The kwarg
    validator runs before the cupy import, so an invalid value
    surfaces deterministically in both environments. This pins the
    forwarding wire even on CPU-only CI.
    """
    from xrspatial.geotiff import open_geotiff

    path, _ = small_tiff_path_1615
    with pytest.raises(ValueError, match="on_gpu_failure must be"):
        open_geotiff(path, gpu=True, on_gpu_failure='loose')


# =====================================================================
# Section 1929 -- ``allow_unparseable_crs`` fail-closed on GPU writers
# =====================================================================
#
# ``_validate_crs_fallback`` wires ``allow_unparseable_crs`` into
# ``to_geotiff``, ``_write_geotiff_gpu``, and the ``to_geotiff(gpu=True)``
# dispatcher. ``test_crs_fail_closed_1929`` only exercises the eager
# CPU writer; this section closes the GPU and dispatcher gap.


def _make_gpu_da_1929() -> xr.DataArray:
    """Build a tiny CuPy-backed DataArray for the GPU writer."""
    import cupy

    arr = cupy.asarray(np.arange(16, dtype=np.float32).reshape(4, 4))
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.arange(4.0, 0, -1), "x": np.arange(4.0)},
    )


def _make_cpu_da_1929() -> xr.DataArray:
    """Numpy-backed twin used to exercise the to_geotiff(gpu=True) path."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": np.arange(4.0, 0, -1), "x": np.arange(4.0)},
    )


@_gpu_only
class TestWriteGeotiffGpuFailClosed_1929:
    """``_write_geotiff_gpu`` refuses to land an unvalidatable CRS by default."""

    def test_garbage_string_kwarg_raises(self, tmp_path):
        """A free-form non-WKT, non-PROJ string raises by default."""
        from xrspatial.geotiff import _write_geotiff_gpu

        out = str(tmp_path / "gpu_garbage_kwarg_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                _write_geotiff_gpu(
                    _make_gpu_da_1929(), out, crs="absolute-garbage")

    def test_garbage_string_attr_raises(self, tmp_path):
        """Same guard fires when garbage arrives via ``attrs['crs']``."""
        from xrspatial.geotiff import _write_geotiff_gpu

        out = str(tmp_path / "gpu_garbage_attr_1929.tif")
        da = _make_gpu_da_1929()
        da.attrs["crs"] = "still-garbage"
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                _write_geotiff_gpu(da, out)

    def test_opt_in_allows_garbage(self, tmp_path):
        """``allow_unparseable_crs=True`` restores the citation-only write."""
        from xrspatial.geotiff import _write_geotiff_gpu

        out = str(tmp_path / "gpu_optin_1929.tif")
        with pytest.warns(Warning):
            _write_geotiff_gpu(
                _make_gpu_da_1929(),
                out,
                crs="absolute-garbage",
                allow_unparseable_crs=True,
            )
        assert os.path.exists(out)

    def test_message_recommends_alternatives(self, tmp_path):
        """The error message points to the four recovery options."""
        from xrspatial.geotiff import _write_geotiff_gpu

        out = str(tmp_path / "gpu_msg_check_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError) as exc:
                _write_geotiff_gpu(_make_gpu_da_1929(), out, crs="bogus")
        msg = str(exc.value)
        assert "EPSG" in msg
        assert "WKT" in msg
        assert "allow_unparseable_crs" in msg

    def test_epsg_int_unchanged(self, tmp_path):
        """An int EPSG kwarg never reaches the fallback path."""
        from xrspatial.geotiff import _write_geotiff_gpu

        out = str(tmp_path / "gpu_epsg_int_1929.tif")
        _write_geotiff_gpu(_make_gpu_da_1929(), out, crs=4326)
        assert os.path.exists(out)

    def test_valid_wkt_unchanged(self, tmp_path):
        """A WKT-shaped string is accepted by the structural check."""
        from xrspatial.geotiff import _write_geotiff_gpu

        out = str(tmp_path / "gpu_wkt_shaped_1929.tif")
        wkt = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",'
            "6378137,298.257223563]],PRIMEM[\"Greenwich\",0],"
            "UNIT[\"degree\",0.0174532925199433]]"
        )
        _write_geotiff_gpu(_make_gpu_da_1929(), out, crs=wkt)
        assert os.path.exists(out)

    def test_no_crs_at_all_unchanged(self, tmp_path):
        """No CRS supplied means no GTCitationGeoKey."""
        from xrspatial.geotiff import _write_geotiff_gpu

        out = str(tmp_path / "gpu_no_crs_1929.tif")
        _write_geotiff_gpu(_make_gpu_da_1929(), out)
        assert os.path.exists(out)


@_gpu_only
class TestToGeotiffGpuDispatcherFailClosed_1929:
    """``to_geotiff(gpu=True)`` threads the validator through to the GPU writer."""

    @staticmethod
    def _install_gpu_spy(monkeypatch):
        """Wrap ``_write_geotiff_gpu`` so the dispatcher entry is recorded."""
        from xrspatial.geotiff._writers import eager as _eager

        real = _eager._write_geotiff_gpu
        calls = []

        def _spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(_eager, "_write_geotiff_gpu", _spy)
        return calls

    def test_dispatcher_garbage_raises_with_cupy_input(
            self, tmp_path, monkeypatch):
        """CuPy-backed input auto-routes to the GPU writer."""
        from xrspatial.geotiff import to_geotiff

        calls = self._install_gpu_spy(monkeypatch)
        out = str(tmp_path / "dispatcher_garbage_gpu_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                to_geotiff(_make_gpu_da_1929(), out, crs="absolute-garbage")
        assert calls, (
            "dispatcher silently fell back to CPU; GPU path never entered"
        )

    def test_dispatcher_gpu_kwarg_garbage_raises(
            self, tmp_path, monkeypatch):
        """Explicit ``gpu=True`` on numpy data also threads through."""
        from xrspatial.geotiff import to_geotiff

        calls = self._install_gpu_spy(monkeypatch)
        out = str(tmp_path / "dispatcher_gpu_kwarg_1929.tif")
        with pytest.warns(Warning):
            with pytest.raises(ValueError, match="GTCitationGeoKey"):
                to_geotiff(
                    _make_cpu_da_1929(), out, gpu=True,
                    crs="absolute-garbage")
        assert calls, (
            "dispatcher silently fell back to CPU; GPU path never entered"
        )

    def test_dispatcher_opt_in_forwarded(self, tmp_path, monkeypatch):
        """``allow_unparseable_crs=True`` is forwarded into the GPU writer."""
        from xrspatial.geotiff import to_geotiff

        calls = self._install_gpu_spy(monkeypatch)
        out = str(tmp_path / "dispatcher_optin_gpu_1929.tif")
        with pytest.warns(Warning):
            to_geotiff(
                _make_gpu_da_1929(),
                out,
                crs="absolute-garbage",
                allow_unparseable_crs=True,
            )
        assert os.path.exists(out)
        assert calls, (
            "dispatcher silently fell back to CPU; GPU path never entered"
        )
        assert calls[0][1].get("allow_unparseable_crs") is True

    def test_dispatcher_opt_in_explicit_gpu(self, tmp_path, monkeypatch):
        """``to_geotiff(gpu=True, allow_unparseable_crs=True)`` also works."""
        from xrspatial.geotiff import to_geotiff

        calls = self._install_gpu_spy(monkeypatch)
        out = str(tmp_path / "dispatcher_optin_gpu_explicit_1929.tif")
        with pytest.warns(Warning):
            to_geotiff(
                _make_cpu_da_1929(),
                out,
                gpu=True,
                crs="absolute-garbage",
                allow_unparseable_crs=True,
            )
        assert os.path.exists(out)
        assert calls, (
            "dispatcher silently fell back to CPU; GPU path never entered"
        )
        assert calls[0][1].get("allow_unparseable_crs") is True


@_gpu_only
class TestErrorMessageParity_1929:
    """The GPU and eager error messages share the recovery hints.

    A user catching ``ValueError`` from ``to_geotiff`` should see the
    same recovery guidance whether the backend is CPU or GPU.
    """

    def test_gpu_vs_cpu_message_matches(self, tmp_path):
        from xrspatial.geotiff import _write_geotiff_gpu, to_geotiff

        out_cpu = str(tmp_path / "cpu_msg_1929.tif")
        out_gpu = str(tmp_path / "gpu_msg_1929.tif")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc_cpu:
                to_geotiff(
                    _make_cpu_da_1929(), out_cpu, crs="absolute-garbage")
            with pytest.raises(ValueError) as exc_gpu:
                _write_geotiff_gpu(
                    _make_gpu_da_1929(), out_gpu, crs="absolute-garbage")

        msg_cpu = str(exc_cpu.value)
        msg_gpu = str(exc_gpu.value)
        recovery_keywords = (
            "GTCitationGeoKey",
            "EPSG",
            "WKT",
            "pyproj",
            "allow_unparseable_crs",
            "absolute-garbage",
        )
        for token in recovery_keywords:
            assert token in msg_cpu, (
                f"CPU error message missing recovery keyword {token!r}: "
                f"{msg_cpu!r}"
            )
            assert token in msg_gpu, (
                f"GPU error message missing recovery keyword {token!r}: "
                f"{msg_gpu!r}"
            )


class TestKwargDefaultParity_1929:
    """``allow_unparseable_crs`` defaults to False on every writer.

    Runs on every host; the inspection only needs the function objects
    to be importable, not a real GPU.
    """

    def test_default_is_false_on_all_writers(self):
        from xrspatial.geotiff import _write_geotiff_gpu, to_geotiff
        from xrspatial.geotiff._writers.eager import _write_vrt_tiled

        for fn in (to_geotiff, _write_geotiff_gpu, _write_vrt_tiled):
            sig = inspect.signature(fn)
            param = sig.parameters.get("allow_unparseable_crs")
            assert param is not None, (
                f"{fn.__name__} must accept allow_unparseable_crs"
            )
            assert param.default is False, (
                f"{fn.__name__}.allow_unparseable_crs default "
                f"drifted to {param.default!r}; #1929 requires fail-closed."
            )
