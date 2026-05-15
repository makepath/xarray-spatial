"""CRS resolution helpers for geotiff readers and writers.

``_wkt_to_epsg`` and ``_resolve_crs_to_wkt`` are pure leaves over
``pyproj`` (lazy-imported inside) and the strict-mode / fallback-warning
machinery from ``_runtime``. They are called from ``to_geotiff``,
``write_geotiff_gpu``, and ``write_vrt`` to normalise the EPSG / WKT /
PROJ kwarg they each accept.

Extracted here in step 3 of issue #1813 so the still-inline writer
entry points and the future ``_backends/`` / ``_writers/`` modules can
import one canonical version.
"""
from __future__ import annotations

import warnings

from ._runtime import GeoTIFFFallbackWarning, _geotiff_strict_mode


def _wkt_to_epsg(wkt_or_proj: str) -> int | None:
    """Try to extract an EPSG code from a WKT or PROJ string.

    Returns None if pyproj is not installed or the string can't be parsed.

    Under ``XRSPATIAL_GEOTIFF_STRICT=1`` the underlying exception is
    re-raised instead of being swallowed. In the default mode a
    ``GeoTIFFFallbackWarning`` is emitted so callers can tell
    pyproj-missing from pyproj-broken-input.
    """
    try:
        from pyproj import CRS
        crs = CRS.from_user_input(wkt_or_proj)
        epsg = crs.to_epsg()
        return epsg
    except Exception as e:
        if _geotiff_strict_mode():
            raise
        warnings.warn(
            f"_wkt_to_epsg failed ({type(e).__name__}: {e}); returning None.",
            GeoTIFFFallbackWarning,
            stacklevel=2,
        )
        return None


def _resolve_crs_to_wkt(crs) -> str | None:
    """Normalise a CRS argument to a WKT string for downstream writers.

    Mirrors ``to_geotiff`` / ``write_geotiff_gpu``'s ``crs`` kwarg semantics
    so callers can pass an int EPSG code, a WKT string, or a PROJ string
    interchangeably. Returns the canonical WKT string (or ``None`` if
    ``crs`` is ``None``) for forwarding to ``_vrt.write_vrt``, which only
    speaks WKT.

    Used by ``write_vrt`` (see issue #1715) to close the parameter-naming
    drift versus the eager and GPU writer entry points.

    Parameters
    ----------
    crs : int, str, or None
        EPSG code (int), WKT string, or PROJ string. ``None`` returns
        ``None`` (the downstream writer falls back to the first source
        file's CRS).

    Returns
    -------
    str or None
        Canonical WKT string, or ``None`` if ``crs`` is ``None``.

    Raises
    ------
    TypeError
        If ``crs`` is not an int, str, or ``None``.
    ValueError
        If ``crs`` is an int that pyproj cannot resolve to a known CRS,
        or a string that pyproj cannot parse.
    ImportError
        If pyproj is not installed and ``crs`` is supplied as something
        other than a string. (A string is passed through verbatim so the
        WKT-only path keeps working without pyproj.)
    """
    if crs is None:
        return None
    if not isinstance(crs, (int, str)):
        raise TypeError(
            f"crs must be int (EPSG code), str (WKT or PROJ), or None; "
            f"got {type(crs).__name__}")
    if isinstance(crs, str):
        # Empty string is a common "no CRS" sentinel from upstream
        # GeoTIFFs; preserve the existing _vrt.write_vrt semantics (it
        # falls back to the first source's CRS for empty strings too).
        if not crs:
            return None
        # If the caller already handed us a WKT, return it untouched.
        # PROJCS/GEOGCS/PROJCRS/GEOGCRS are the standard WKT root
        # keywords; anything else (EPSG:NNNN, +proj=...) gets normalised
        # through pyproj so the downstream XML sees a canonical WKT.
        if crs.lstrip().startswith(('PROJCS', 'GEOGCS', 'PROJCRS', 'GEOGCRS',
                                     'COMPD_CS', 'COMPOUNDCRS')):
            return crs
        try:
            from pyproj import CRS
        except ImportError as exc:
            raise ImportError(
                "pyproj is required to convert non-WKT CRS strings (got "
                f"{crs!r}). Pass a WKT string directly, or install pyproj."
            ) from exc
        try:
            return CRS.from_user_input(crs).to_wkt()
        except Exception as exc:
            raise ValueError(
                f"Could not parse crs={crs!r} as an EPSG/PROJ/WKT string: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    # int branch: convert EPSG -> WKT via pyproj.
    try:
        from pyproj import CRS
    except ImportError as exc:
        raise ImportError(
            f"pyproj is required to convert crs={crs} (EPSG int) to WKT. "
            "Install pyproj, or pass crs as a WKT string."
        ) from exc
    try:
        return CRS.from_epsg(crs).to_wkt()
    except Exception as exc:
        raise ValueError(
            f"Could not resolve EPSG:{crs}: {type(exc).__name__}: {exc}"
        ) from exc
