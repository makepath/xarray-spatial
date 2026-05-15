"""Identity contract: sentinels survive the move from __init__.py to _runtime.py.

PR for issue #1880 (step 2 of #1813) extracted four module-level
sentinels, a UserWarning subclass, the strict-mode helper, and the
GPU-fallback message helper from ``xrspatial/geotiff/__init__.py``
into ``xrspatial/geotiff/_runtime.py``. ``__init__.py`` keeps every
name importable via re-export.

If a future refactor accidentally rebinds one of these names to a
fresh ``object()`` (or shadows ``GeoTIFFFallbackWarning`` with a local
class), every ``is`` comparison against the sentinel inside the entry
points would silently start failing -- ``open_geotiff`` would no
longer recognise the canonical sentinel and would treat "caller passed
no kwarg" as "caller passed the default value". This file pins the
contract: the names imported through ``xrspatial.geotiff`` and through
``xrspatial.geotiff._runtime`` are the same Python object.
"""
from __future__ import annotations

import xrspatial.geotiff as geotiff_pkg
from xrspatial.geotiff import _runtime


def test_gpu_deprecated_sentinel_is_singleton():
    assert geotiff_pkg._GPU_DEPRECATED_SENTINEL is _runtime._GPU_DEPRECATED_SENTINEL


def test_on_gpu_failure_sentinel_is_singleton():
    assert geotiff_pkg._ON_GPU_FAILURE_SENTINEL is _runtime._ON_GPU_FAILURE_SENTINEL


def test_crs_wkt_deprecated_sentinel_is_singleton():
    assert geotiff_pkg._CRS_WKT_DEPRECATED_SENTINEL is \
        _runtime._CRS_WKT_DEPRECATED_SENTINEL


def test_missing_sources_sentinel_is_singleton():
    assert geotiff_pkg._MISSING_SOURCES_SENTINEL is \
        _runtime._MISSING_SOURCES_SENTINEL


def test_fallback_warning_class_is_singleton():
    """``GeoTIFFFallbackWarning`` is the same class through both import paths.

    This is the only re-exported name from ``_runtime`` that is in
    ``__all__``. A duplicate class would still print the right name in
    a ``warns(GeoTIFFFallbackWarning)`` context but ``issubclass``
    chains in user code would break.
    """
    assert geotiff_pkg.GeoTIFFFallbackWarning is _runtime.GeoTIFFFallbackWarning


def test_dim_name_tuples_are_singleton():
    assert geotiff_pkg._Y_DIM_NAMES is _runtime._Y_DIM_NAMES
    assert geotiff_pkg._X_DIM_NAMES is _runtime._X_DIM_NAMES


def test_strict_mode_helper_is_singleton():
    assert geotiff_pkg._geotiff_strict_mode is _runtime._geotiff_strict_mode


def test_gpu_fallback_warning_message_is_singleton():
    assert geotiff_pkg._gpu_fallback_warning_message is \
        _runtime._gpu_fallback_warning_message


def test_strict_mode_env_var_round_trips(monkeypatch):
    """The strict-mode helper still reads the env var after the move.

    Guards against an accidental hard-coded return value or wrong env
    var name introduced during the relocation.
    """
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_STRICT", "1")
    assert geotiff_pkg._geotiff_strict_mode() is True
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_STRICT", "true")
    assert geotiff_pkg._geotiff_strict_mode() is True
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_STRICT", "0")
    assert geotiff_pkg._geotiff_strict_mode() is False
    monkeypatch.delenv("XRSPATIAL_GEOTIFF_STRICT", raising=False)
    assert geotiff_pkg._geotiff_strict_mode() is False


def test_fallback_message_includes_exception_type_and_message():
    """The two GPU-fallback wording branches both surface the exception."""
    exc = RuntimeError("nvcomp not installed")
    explicit = geotiff_pkg._gpu_fallback_warning_message(
        auto_detected=False, exc=exc)
    auto = geotiff_pkg._gpu_fallback_warning_message(
        auto_detected=True, exc=exc)
    for msg in (explicit, auto):
        assert "RuntimeError" in msg
        assert "nvcomp not installed" in msg
    assert "to_geotiff(gpu=True) was requested" in explicit
    assert "Data is on the GPU" in auto
