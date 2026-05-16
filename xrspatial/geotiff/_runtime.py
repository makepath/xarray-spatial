"""Geotiff module-level runtime state: sentinels, fallback warning, strict mode.

These live in their own module so that backend extractions can import
a single canonical binding for each sentinel and helper without
threading them through ``__init__.py``. Sentinel object identity is
preserved by Python's module cache: every import of this module
returns the same module instance, so ``_GPU_DEPRECATED_SENTINEL is
other._GPU_DEPRECATED_SENTINEL`` resolves correctly regardless of
which caller imported it.

See issue #1813 step 2 for the rationale; PR for issue #1880.
"""
from __future__ import annotations

import os


# Sentinels distinguishing "user passed this kwarg explicitly" from "user
# passed nothing". A plain default of None does not work because None is
# itself a value a caller could supply. ``read_geotiff_gpu`` needs both
# sentinels so it can tell whether the deprecated ``gpu=`` and the new
# ``on_gpu_failure=`` were *each* supplied, and refuse the ambiguous
# both-supplied case regardless of which values were chosen.
# ``open_geotiff`` also uses ``_ON_GPU_FAILURE_SENTINEL`` to distinguish
# "caller never set on_gpu_failure" (default sentinel: skip forwarding so
# the read_geotiff_gpu signature default applies) from "caller set
# on_gpu_failure=<value>" (forward verbatim).
_GPU_DEPRECATED_SENTINEL = object()
_ON_GPU_FAILURE_SENTINEL = object()
# ``write_vrt`` needs to distinguish "user passed crs_wkt= explicitly"
# (deprecation path) from "user passed nothing" (no warning, pick CRS
# from the first source). A plain default of None does not work because
# None is itself a value a caller could supply alongside crs=. See
# issue #1715.
_CRS_WKT_DEPRECATED_SENTINEL = object()
# ``open_geotiff`` needs to tell "caller never set missing_sources" (default
# sentinel: skip forwarding so the read_vrt default applies, and reject the
# kwarg up front for non-VRT sources) from "caller set missing_sources=<value>"
# (forward verbatim to read_vrt). Mirrors the on_gpu_failure pattern. See
# issue #1810.
_MISSING_SOURCES_SENTINEL = object()
# ``write_vrt`` historically named its first positional kwarg ``vrt_path``
# while ``to_geotiff`` / ``write_geotiff_gpu`` use ``path``. The deprecation
# shim adds ``path`` as the new name and accepts ``vrt_path`` with a
# DeprecationWarning. The sentinel pattern distinguishes "user passed
# vrt_path= explicitly" from "user passed nothing", which is the same
# rationale ``_CRS_WKT_DEPRECATED_SENTINEL`` documents above. See
# issue #1946.
_VRT_PATH_DEPRECATED_SENTINEL = object()
# ``write_vrt`` also needs to distinguish "user passed path= explicitly"
# (including an explicit ``path=None``, which is an error) from "user
# passed nothing" (fall through to the ``vrt_path`` shim). Without this
# sentinel, ``write_vrt(None, sources)`` silently fell through to the
# ``path is None`` branch and raised a "missing required argument"
# TypeError for the wrong reason. See PR #1962 review.
_VRT_PATH_MISSING_SENTINEL = object()


# Spatial dim names recognised on 3D writer inputs. ``y``/``x`` are the
# canonical TIFF axes; aliases are accepted so a user who happens to use
# ``lat``/``lon`` or ``row``/``col`` is not bounced by the validator.
_Y_DIM_NAMES = ('y', 'lat', 'latitude', 'row')
_X_DIM_NAMES = ('x', 'lon', 'longitude', 'col')

# Temporal dim names. Used by the 3D writer validator (#1972) to refuse
# ``(y, x, <temporal>)`` inputs that would otherwise be silently treated
# as multiband rasters. CF / xarray conventions cover ``time`` and ``t``;
# the rest match common upstream-pipeline aliases.
_TIME_DIM_NAMES = ('time', 't', 'date', 'datetime', 'times', 'dates')


class GeoTIFFFallbackWarning(UserWarning):
    """Warning emitted when a geotiff helper falls back to a slower path.

    Raised in the same call sites that would silently return ``None`` under
    the historic ``except Exception: return None`` pattern. See issue #1662
    for the audit and the ``XRSPATIAL_GEOTIFF_STRICT=1`` env var that
    promotes these warnings to exceptions.
    """


def _geotiff_strict_mode() -> bool:
    """Return True when ``XRSPATIAL_GEOTIFF_STRICT`` is set to a truthy value.

    Strict mode promotes the silent fallbacks audited in issue #1662 into
    raised exceptions. Useful in CI to catch GPU-path or VRT regressions
    that would otherwise hide behind a CPU fallback or a missing tile.
    """
    return os.environ.get(
        'XRSPATIAL_GEOTIFF_STRICT', '').lower() in ('1', 'true', 'yes')


def _gpu_fallback_warning_message(auto_detected: bool, exc: BaseException) -> str:
    """Build the ``to_geotiff`` GPU-to-CPU fallback warning text.

    ``to_geotiff`` reaches the GPU writer two ways: an explicit
    ``gpu=True`` argument, or the auto-detect branch when ``gpu is
    None`` and the data lives on a CuPy device. The wording differs
    because blaming the fallback on a flag the caller never set sends
    them to fix the wrong thing. Both routes share the exception
    payload format so callers can grep ``type(e).__name__: e`` either
    way.
    """
    suffix = f"({type(exc).__name__}: {exc})."
    if auto_detected:
        return (
            "Data is on the GPU and was routed to the GPU writer, but "
            "the writer is unavailable; falling back to CPU and copying "
            "the array to host. " + suffix
        )
    return (
        "to_geotiff(gpu=True) was requested but the GPU writer is "
        "unavailable; falling back to CPU. " + suffix
    )
