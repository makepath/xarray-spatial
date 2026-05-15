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


#: WKT root keywords. A string that starts with one of these (after
#: stripping leading whitespace) is structurally a WKT and is allowed
#: to land in ``GTCitationGeoKey`` even when pyproj is not available
#: to validate it. Anything else (``"EPSG:4326"`` minus pyproj,
#: ``"+proj=..."`` minus pyproj, free-form garbage) is rejected by
#: :func:`_validate_crs_fallback` unless the caller opts in. See
#: issue #1929.
_WKT_ROOT_KEYWORDS = (
    'PROJCS', 'GEOGCS', 'PROJCRS', 'GEOGCRS',
    'COMPD_CS', 'COMPOUNDCRS', 'BOUNDCRS', 'LOCAL_CS', 'ENGCRS',
    'VERT_CS', 'VERTCRS', 'PARAMETRICCRS', 'TIMECRS', 'DERIVEDPROJCRS',
)


def _looks_like_wkt(s: str) -> bool:
    """Cheap structural check: does ``s`` start with a WKT root keyword?

    Used by :func:`_validate_crs_fallback` to decide whether a string
    that pyproj could not (or was not asked to) validate is at least
    *shaped* like WKT. WKT-shaped strings land in ``GTCitationGeoKey``
    verbatim; everything else is refused by default.
    """
    if not isinstance(s, str):
        return False
    return s.lstrip().upper().startswith(_WKT_ROOT_KEYWORDS)


def _validate_crs_arg(crs) -> None:
    """Reject malformed ``crs=`` arguments before they reach the writer.

    Closes two gaps in the writer entry points (issue #1971):

    * ``bool`` is an ``int`` subclass, so ``crs=True`` and ``crs=False``
      would otherwise slip through ``isinstance(crs, int)`` and write
      ``EPSG=1`` / ``EPSG=0`` to the file. No CRS database resolves
      those, so the result is silent metadata corruption.
    * An ``int`` EPSG code that pyproj cannot resolve gets written
      verbatim into ``ProjectedCSType`` / ``GeographicType``. The
      file then round-trips with ``attrs['crs']`` set to the bad
      value and only a ``GeoTIFFFallbackWarning`` to tell the caller
      something is wrong.

    Validates ``crs`` is one of ``None`` (no-op), ``int`` (a valid
    EPSG code), or ``str`` (WKT/PROJ -- left for ``_wkt_to_epsg``
    downstream). Pyproj is optional; the EPSG-resolves check is
    skipped when pyproj is not installed, matching the rest of the
    module's pyproj-optional posture. Under
    ``XRSPATIAL_GEOTIFF_STRICT=1`` the pyproj error is re-raised
    instead of being wrapped.
    """
    if crs is None:
        return
    if isinstance(crs, bool):
        raise ValueError(
            f"crs must be an int (EPSG code), str (WKT/PROJ), or None; "
            f"got bool ({crs!r}). bool is an int subclass in Python, so "
            f"passing True/False would otherwise be written as EPSG=1 / "
            f"EPSG=0 -- neither resolves with any CRS database."
        )
    if isinstance(crs, int):
        try:
            from pyproj import CRS
        except ImportError:
            return
        try:
            CRS.from_epsg(crs)
        except Exception as e:
            if _geotiff_strict_mode():
                raise
            raise ValueError(
                f"crs={crs!r} is not a valid EPSG code "
                f"(pyproj: {type(e).__name__}: {e}). Pass a valid "
                f"EPSG integer, a WKT string, or None."
            ) from e
        return
    if isinstance(crs, str):
        return
    raise TypeError(
        f"crs must be int (EPSG code), str (WKT/PROJ), or None; "
        f"got {type(crs).__name__}."
    )


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


def _validate_crs_fallback(
    wkt_fallback: str | None,
    allow_unparseable_crs: bool,
) -> None:
    """Refuse to land an unvalidatable string in ``GTCitationGeoKey``.

    Issue #1929: when ``_wkt_to_epsg`` cannot resolve the caller's CRS
    to an EPSG code, the writer stores the original string as
    ``wkt_fallback`` and emits it into ``GTCitationGeoKey``. If the
    string is a malformed PROJ / EPSG token (e.g. ``"EPSG:4326"`` on a
    host without pyproj, or a typo'd PROJ string), the file ends up
    with garbage in the citation field. For a foundational I/O module
    the default has to be fail-closed.

    Raises ``ValueError`` when ``wkt_fallback`` is non-None, the string
    does not structurally look like WKT (:func:`_looks_like_wkt`), and
    the caller has not opted in via ``allow_unparseable_crs=True``.

    Pyproj-validatable strings never reach the fallback because
    ``_wkt_to_epsg`` returns an EPSG (or, under
    ``XRSPATIAL_GEOTIFF_STRICT=1``, raises). The remaining failure
    modes -- pyproj missing, or pyproj installed and parse-fails --
    both leave a non-None ``wkt_fallback``, and this helper is what
    closes the gap.
    """
    if wkt_fallback is None:
        return
    if _looks_like_wkt(wkt_fallback):
        return
    if allow_unparseable_crs:
        return
    raise ValueError(
        "crs is not an EPSG code, is not a WKT string "
        "(no PROJCS / GEOGCS / PROJCRS / GEOGCRS root), and could not "
        f"be parsed: got {wkt_fallback!r}. Writing it verbatim to "
        "GTCitationGeoKey would produce a file most GeoTIFF readers "
        "cannot interpret. Pass an EPSG int (recommended), a real "
        "WKT string, install pyproj so EPSG / PROJ tokens can be "
        "resolved, or pass allow_unparseable_crs=True to keep the "
        "pre-#1929 citation-only behaviour."
    )


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
    _validate_crs_arg(crs)
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
