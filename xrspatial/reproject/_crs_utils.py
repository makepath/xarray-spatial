"""CRS detection utilities and optional pyproj import guard."""
from __future__ import annotations


def _require_pyproj():
    """Import and return the pyproj module, raising a clear error if missing."""
    try:
        import pyproj
        return pyproj
    except ImportError:
        raise ImportError(
            "pyproj is required for CRS reprojection. "
            "Install it with:  pip install pyproj  "
            "or:  pip install xarray-spatial[reproject]"
        )


def _resolve_crs(crs_input):
    """Convert *crs_input* to a ``pyproj.CRS`` object.

    Accepts anything ``pyproj.CRS()`` accepts: EPSG int, authority string,
    WKT, proj4 dict, or an existing ``pyproj.CRS`` instance.

    Returns None if *crs_input* is None.
    """
    if crs_input is None:
        return None
    pyproj = _require_pyproj()
    if isinstance(crs_input, pyproj.CRS):
        return crs_input
    return pyproj.CRS(crs_input)


def _detect_source_crs(raster):
    """Auto-detect the CRS of a DataArray.

    Fallback chain:
    1. ``raster.rio.crs`` (rioxarray)
    2. ``raster.attrs['crs']``
    3. None
    """
    # rioxarray
    try:
        rio_crs = raster.rio.crs
        if rio_crs is not None:
            return _resolve_crs(rio_crs)
    except Exception:
        pass

    # attrs
    crs_attr = raster.attrs.get('crs')
    if crs_attr is not None:
        return _resolve_crs(crs_attr)

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
