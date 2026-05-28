"""CRS resolution helpers for geotiff readers and writers.

``_wkt_to_epsg`` and ``_resolve_crs_to_wkt`` are pure leaves over
``pyproj`` (lazy-imported inside) and the strict-mode / fallback-warning
machinery from ``_runtime``. They are called from ``to_geotiff``,
``write_geotiff_gpu``, and ``write_vrt`` to normalise the EPSG / WKT /
PROJ kwarg they each accept.
"""
from __future__ import annotations

import numbers
import warnings

from ._errors import NonRepresentableEPSGCRSError, UnparseableCRSError
from ._runtime import GeoTIFFFallbackWarning, _geotiff_strict_mode

#: WKT root keywords. A string that starts with one of these (after
#: stripping leading whitespace) is structurally a WKT and is allowed
#: to land in ``GTCitationGeoKey`` even when pyproj is not available
#: to validate it. Anything else (``"EPSG:4326"`` minus pyproj,
#: ``"+proj=..."`` minus pyproj, free-form garbage) is rejected by
#: :func:`_validate_crs_fallback` unless the caller opts in.
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


def _reject_non_representable_epsg(crs_int: int, crs_obj) -> None:
    """Refuse compound EPSG codes that the stable EPSG writer cannot
    round-trip through rasterio / GDAL.

    The GeoTIFF writer's stable-path emits only two EPSG-carrying
    GeoKeys: ``GeographicTypeGeoKey`` (2048) and
    ``ProjectedCSTypeGeoKey`` (3072). Both are spec'd to hold a 2D
    horizontal CRS code. When the caller passes a compound EPSG code
    (horizontal + vertical, e.g. EPSG:6349 = "NAD83(2011) + NAVD88
    height"), pyproj reports the compound CRS as ``is_geographic`` or
    ``is_projected`` based on its horizontal sub-CRS, and the writer
    stores the compound code in the horizontal-CRS slot. The resulting
    GeoTIFF either reads back through GDAL as the wrong CRS (the
    horizontal sub-CRS, with the vertical component silently dropped)
    or as no CRS at all, because the GeoKey value claims to be a 2D
    horizontal CRS but the EPSG database registers it as compound.

    Reject the code at validation time so the failure surfaces at the
    write call instead of after the file is on disk. Callers who need
    to preserve the compound semantics can pass the full WKT (which
    takes the user-defined / citation path) or pass the horizontal
    sub-CRS code directly.

    pyproj exposes ``is_compound`` on every CRS object since 2.x;
    callers without pyproj never reach this function (the caller
    site short-circuits on ``ImportError``).
    """
    if getattr(crs_obj, "is_compound", False):
        raise NonRepresentableEPSGCRSError(
            f"EPSG:{crs_int} is a compound CRS ({crs_obj.name!r}) and "
            "cannot be written through the integer-EPSG path: the "
            "GeoTIFF writer only emits GeographicTypeGeoKey / "
            "ProjectedCSTypeGeoKey, which are specified to carry a 2D "
            "horizontal CRS code. Storing a compound EPSG there "
            "produces a file that rasterio / GDAL reads back as the "
            "horizontal sub-CRS (vertical component lost) or as no "
            "CRS at all. Pass the full compound WKT as crs= to take "
            "the user-defined CRS path (see issue #1768), or pass the "
            "horizontal sub-CRS EPSG directly if the vertical "
            "component is not needed."
        )


def _validate_crs_arg(crs) -> None:
    """Reject malformed ``crs=`` arguments before they reach the writer.

    Closes two gaps in the writer entry points:

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
    # ``numbers.Integral`` covers plain ``int`` and numpy integer scalars
    # (``np.int32``, ``np.int64``, ...). Without this branch the type
    # check below rejects numpy-typed CRS values that callers previously
    # got away with (pre-PR they silently fell through to "no EPSG
    # written"; the post-PR ``isinstance(crs, int)`` check would raise
    # ``TypeError`` on the same input).
    if isinstance(crs, numbers.Integral):
        crs_int = int(crs)
        try:
            from pyproj import CRS
        except ImportError:
            return
        try:
            crs_obj = CRS.from_epsg(crs_int)
        except Exception as e:
            if _geotiff_strict_mode():
                raise
            raise ValueError(
                f"crs={crs!r} is not a valid EPSG code "
                f"(pyproj: {type(e).__name__}: {e}). Pass a valid "
                f"EPSG integer, a WKT string, or None."
            ) from e
        _reject_non_representable_epsg(crs_int, crs_obj)
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
        # ``epsg`` is set when pyproj recognises the WKT,
        # but if the resolved CRS is compound (horizontal + vertical),
        # the integer-EPSG writer path cannot represent it -- the GeoKey
        # slots only carry 2D horizontal codes. Return None so the
        # caller routes through the WKT / citation fallback instead of
        # writing a compound EPSG into a horizontal slot. The original
        # WKT is preserved on the fallback path.
        if epsg is not None and getattr(crs, "is_compound", False):
            return None
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

    When ``_wkt_to_epsg`` cannot resolve the caller's CRS
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
    raise UnparseableCRSError(
        "crs is not an EPSG code, is not a WKT string "
        "(no PROJCS / GEOGCS / PROJCRS / GEOGCRS root), and could not "
        f"be parsed: got {wkt_fallback!r}. Writing it verbatim to "
        "GTCitationGeoKey would produce a file most GeoTIFF readers "
        "cannot interpret. Pass an EPSG int (recommended), a real "
        "WKT string, install pyproj so EPSG / PROJ tokens can be "
        "resolved, or pass allow_unparseable_crs=True to keep the "
        "pre-#1929 citation-only behaviour. See issue #1987."
    )


def _resolve_crs_to_wkt(crs) -> str | None:
    """Normalise a CRS argument to a WKT string for downstream writers.

    Mirrors ``to_geotiff`` / ``write_geotiff_gpu``'s ``crs`` kwarg semantics
    so callers can pass an int EPSG code, a WKT string, or a PROJ string
    interchangeably. Returns the canonical WKT string (or ``None`` if
    ``crs`` is ``None``) for forwarding to ``_vrt.write_vrt``, which only
    speaks WKT.

    Used by ``write_vrt`` to close the parameter-naming
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
    # ``_validate_crs_arg`` already accepts ``numbers.Integral`` (incl.
    # numpy integer scalars); coerce here so the pyproj path and the
    # str-only branch below see a plain ``int``.
    if isinstance(crs, numbers.Integral) and not isinstance(crs, bool):
        crs = int(crs)
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
