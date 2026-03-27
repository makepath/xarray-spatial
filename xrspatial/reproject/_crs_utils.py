"""CRS detection utilities and optional pyproj import guard.

Uses a two-tier strategy: try the lightweight built-in CRS first,
then fall back to pyproj for codes/formats not in the built-in table.
"""
from __future__ import annotations

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


def _detect_nodata(raster, nodata=None):
    """Determine nodata value from explicit arg, rioxarray, or attrs."""
    if nodata is not None:
        return float(nodata)

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

    return float('nan')
