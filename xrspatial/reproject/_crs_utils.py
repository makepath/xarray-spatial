"""CRS detection utilities and optional pyproj import guard.

Uses a two-tier strategy: try the lightweight built-in CRS first,
then fall back to pyproj for codes/formats not in the built-in table.
"""
from __future__ import annotations

import numpy as np

from xrspatial.reproject._lite_crs import CRS as LiteCRS


def _try_import_pyproj():
    """Try to import pyproj, returning the module or None."""
    try:
        import pyproj
        return pyproj
    except ImportError:
        return None


def _require_pyproj():
    """Import and return the pyproj module, raising a clear error if missing."""
    pyproj = _try_import_pyproj()
    if pyproj is None:
        raise ImportError(
            "pyproj is required for CRS reprojection. "
            "Install it with:  pip install pyproj  "
            "or:  pip install xarray-spatial[reproject]"
        )
    return pyproj


def _resolve_crs(crs_input):
    """Convert *crs_input* to a CRS object.

    Resolution order:

    1. ``None`` passes through as ``None``.
    2. An existing ``LiteCRS`` instance passes through unchanged.
    3. An existing ``pyproj.CRS`` instance passes through unchanged
       (only checked when pyproj is importable).
    4. Try ``LiteCRS(crs_input)`` -- covers EPSG ints and ``"EPSG:XXXX"``
       strings for codes in the built-in table.
    5. Fall back to ``pyproj.CRS(crs_input)`` -- raises ``ImportError``
       if pyproj is not installed.
    """
    if crs_input is None:
        return None

    # Pass through existing LiteCRS
    if isinstance(crs_input, LiteCRS):
        return crs_input

    # Pass through existing pyproj.CRS (if pyproj available)
    pyproj = _try_import_pyproj()
    if pyproj is not None and isinstance(crs_input, pyproj.CRS):
        return crs_input

    # Try lite CRS first
    try:
        return LiteCRS(crs_input)
    except (ValueError, TypeError):
        pass

    # Fall back to pyproj
    pyproj = _require_pyproj()
    return pyproj.CRS(crs_input)


def _crs_from_wkt(wkt):
    """Build a CRS from an OGC WKT string.

    Tries ``LiteCRS.from_wkt`` first (extracts the AUTHORITY tag),
    then falls back to ``pyproj.CRS.from_wkt``.
    """
    try:
        return LiteCRS.from_wkt(wkt)
    except (ValueError, TypeError):
        pass

    pyproj = _require_pyproj()
    return pyproj.CRS.from_wkt(wkt)


def _detect_source_crs(raster):
    """Auto-detect the CRS of a DataArray.

    Fallback chain:
    1. ``raster.attrs['crs']`` (EPSG int from xrspatial.geotiff)
    2. ``raster.attrs['crs_wkt']`` (WKT string from xrspatial.geotiff)
    3. ``raster.rio.crs`` (rioxarray, if installed)
    4. None
    """
    # attrs (xrspatial.geotiff convention)
    crs_attr = raster.attrs.get('crs')
    if crs_attr is not None:
        return _resolve_crs(crs_attr)

    crs_wkt = raster.attrs.get('crs_wkt')
    if crs_wkt is not None:
        return _crs_from_wkt(crs_wkt)

    # rioxarray fallback
    try:
        rio_crs = raster.rio.crs
        if rio_crs is not None:
            return _resolve_crs(rio_crs)
    except Exception:
        pass

    return None


def _default_integer_nodata(dtype):
    """Return the default nodata sentinel for an integer dtype.

    Follows the rasterio/GDAL convention: signed integers use the dtype
    minimum (e.g. int16 -> -32768) and unsigned integers use the dtype
    maximum (e.g. uint16 -> 65535, uint8 -> 255). Returning a value that
    actually fits the dtype avoids the silent NaN -> 0 corruption that
    happens when NaN is cast back to an integer array.
    """
    info = np.iinfo(dtype)
    if np.issubdtype(dtype, np.signedinteger):
        return float(info.min)
    return float(info.max)


def _detect_nodata(raster, nodata=None, dtype=None):
    """Determine nodata value from explicit arg, rioxarray, or attrs.

    NaN is the canonical sentinel for floating-point missing data and
    is allowed. +/-Inf would break downstream ``np.isnan`` masks, so it
    is rejected.

    Integer dtypes can't carry NaN. When *dtype* is integer and no
    upstream nodata is found, fall back to a dtype-appropriate sentinel
    (see :func:`_default_integer_nodata`) instead of returning NaN.
    Without this, the worker's cast-back step silently turned every
    out-of-bounds pixel into ``0`` while ``attrs['nodata']`` still
    advertised NaN.
    """
    if nodata is not None:
        nd = float(nodata)
        if np.isinf(nd):
            raise ValueError(
                f"nodata must be finite or NaN, got {nodata!r}"
            )
        return nd

    # rioxarray
    try:
        rio_nd = raster.rio.nodata
        if rio_nd is not None:
            return float(rio_nd)
    except Exception:
        pass

    # attrs
    for key in ('_FillValue', 'nodata', 'missing_value'):
        val = raster.attrs.get(key)
        if val is not None:
            return float(val)

    # `nodatavals` is the rasterio convention (tuple of per-band sentinels).
    # Accept either a tuple/list or a bare scalar. This matches what
    # `xrspatial.resample` already does and keeps reproject behaving
    # consistently for rasterio-style inputs when rioxarray is not
    # installed.
    nv = raster.attrs.get('nodatavals')
    if nv is not None:
        try:
            first = nv[0]
        except (TypeError, IndexError):
            first = nv
        if first is not None:
            return float(first)

    # No upstream nodata. For integer dtypes, NaN would round to 0 on
    # cast-back so use a sentinel that fits the dtype instead.
    if dtype is not None:
        dt = np.dtype(dtype)
        if np.issubdtype(dt, np.integer):
            return _default_integer_nodata(dt)

    return float('nan')
