"""VRT writer entry point.

Wraps ``_vrt.write_vrt`` with the public ``write_vrt`` surface:
deprecation handling for ``crs_wkt`` (#1715) and ``vrt_path`` (#1946),
normalisation of the ``crs`` kwarg to WKT via ``_resolve_crs_to_wkt``,
and the parity docstring vs ``to_geotiff`` / ``write_geotiff_gpu``.
"""
from __future__ import annotations

import warnings

from .._crs import _resolve_crs_to_wkt
from .._runtime import (
    _CRS_WKT_DEPRECATED_SENTINEL,
    _VRT_PATH_DEPRECATED_SENTINEL,
    _VRT_PATH_MISSING_SENTINEL,
)
from .._validation import _validate_nodata_arg


def write_vrt(path: str = _VRT_PATH_MISSING_SENTINEL,
              source_files: list[str] | None = None, *,
              vrt_path: str | None = _VRT_PATH_DEPRECATED_SENTINEL,
              relative: bool = True,
              crs: int | str | None = None,
              crs_wkt: str | None = _CRS_WKT_DEPRECATED_SENTINEL,
              nodata: float | int | None = None) -> str:
    """Generate a VRT file that mosaics multiple GeoTIFF tiles.

    Tier: Advanced (issue #2137). VRT mosaic output is supported but
    the caller should know the failure modes on the read side: a
    consumer reading the resulting ``.vrt`` may hit cross-source
    nodata mismatch, missing backing files, or per-band metadata
    disagreement. See :data:`xrspatial.geotiff.SUPPORTED_FEATURES` for
    the full tier map.

    Parameters
    ----------
    path : str
        Output .vrt file path. Mirrors the ``path`` kwarg on
        ``to_geotiff`` and ``write_geotiff_gpu`` so the writer trio
        shares a single destination-arg name (issue #1946).
    source_files : list of str
        Paths to the source GeoTIFF files.
    vrt_path : str, optional
        Deprecated alias for ``path``. Emits ``DeprecationWarning`` when
        supplied; passing both ``path`` and ``vrt_path`` raises
        ``TypeError``. Kept so existing callers (``write_vrt(vrt_path,
        sources)`` positional or ``write_vrt(vrt_path=...)`` keyword)
        keep working through the deprecation window. New code should
        use ``path``. See issue #1946.
    relative : bool, optional
        Store source paths relative to the VRT file (default True).
    crs : int, str, or None, optional
        EPSG code (int), WKT string, or PROJ string. If None, the CRS
        is taken from the first source GeoTIFF. Mirrors the ``crs``
        kwarg on ``to_geotiff`` and ``write_geotiff_gpu`` so the same
        value can be forwarded to whichever writer the caller picked
        without per-writer special-casing (issue #1715).
    crs_wkt : str or None, optional
        Deprecated alias for ``crs``. Emits ``DeprecationWarning`` when
        supplied (including ``crs_wkt=None``); passing both ``crs`` and
        ``crs_wkt`` raises ``TypeError``. The value is forwarded through
        the same ``_resolve_crs_to_wkt`` path as ``crs``, so any string
        the resolver accepts (WKT root keyword, PROJ string,
        ``"EPSG:NNNN"``) and ``None`` work here. The historic
        ``str | None`` surface is preserved; new code should use ``crs``
        instead, which additionally accepts ``int`` EPSG codes.
    nodata : float, int, or None, optional
        NoData value. If None, taken from the first source GeoTIFF.
        Integer sentinels (e.g. ``65535`` for uint16, ``-9999`` for
        int32) are accepted so the surface lines up with the
        ``nodata`` kwarg on ``to_geotiff`` and ``write_geotiff_gpu``.

    Returns
    -------
    str
        Path to the written VRT file.
    """
    # Explicit signature (previously ``**kwargs``) so ``inspect.signature``,
    # IDE autocomplete, and ``mypy --strict`` can see the accepted kwargs
    # without parsing the docstring. Mirrors ``_vrt.write_vrt`` for the
    # historic ``crs_wkt`` path; the new ``crs`` path normalises through
    # ``_resolve_crs_to_wkt`` before forwarding because the internal
    # writer still only speaks WKT.
    #
    # The ``path`` / ``vrt_path`` shim resolves the destination kwarg
    # before any other processing so the rest of the function works
    # uniformly against a single ``vrt_path`` local. ``path`` is the
    # new name (parity with to_geotiff / write_geotiff_gpu); ``vrt_path``
    # is kept as a deprecated alias to preserve back-compat for callers
    # using either positional ``write_vrt(vrt_path, sources)`` or
    # keyword ``write_vrt(vrt_path=...)``.
    path_passed = path is not _VRT_PATH_MISSING_SENTINEL
    vrt_path_passed = vrt_path is not _VRT_PATH_DEPRECATED_SENTINEL
    if path_passed and vrt_path_passed:
        # Both supplied is ambiguous regardless of whether the two values
        # happen to be the same string. Refuse rather than silently
        # picking one. Mirrors the same rule the ``crs`` / ``crs_wkt``
        # shim below applies.
        raise TypeError(
            "write_vrt: pass either 'path' or the deprecated 'vrt_path' "
            "alias, not both.")
    if vrt_path_passed:
        warnings.warn(
            "write_vrt(..., vrt_path=...) is deprecated; use path=... "
            "instead. The kwarg was renamed for parity with to_geotiff "
            "and write_geotiff_gpu, which already accept 'path' as the "
            "destination kwarg.",
            DeprecationWarning,
            stacklevel=2,
        )
        path = vrt_path
    elif not path_passed:
        # Neither name supplied. Match the previous ``TypeError: missing
        # required positional argument`` semantics by raising rather than
        # forwarding the sentinel into ``_write_vrt_internal``.
        raise TypeError(
            "write_vrt: missing required argument 'path'")
    if path is None:
        # Explicit ``path=None`` (including positional ``write_vrt(None,
        # sources)``) is rejected up front so the error message names the
        # offending kwarg instead of crashing deep in
        # ``os.path.dirname(os.path.abspath(None))``. The sentinel default
        # on ``path`` is what lets us distinguish this case from "caller
        # passed nothing" above.
        raise TypeError(
            "write_vrt: 'path' must be a str, got None")
    if source_files is None:
        raise TypeError(
            "write_vrt: missing required argument 'source_files'")

    crs_wkt_passed = crs_wkt is not _CRS_WKT_DEPRECATED_SENTINEL
    if crs is not None and crs_wkt_passed:
        # Both supplied is ambiguous regardless of whether the WKT happens
        # to encode the same CRS as the int. Refuse rather than silently
        # picking one.
        raise TypeError(
            "write_vrt: pass either 'crs' or the deprecated 'crs_wkt' "
            "alias, not both.")
    if crs_wkt_passed:
        warnings.warn(
            "write_vrt(..., crs_wkt=...) is deprecated; use crs=... "
            "instead. The kwarg was renamed for parity with to_geotiff "
            "and write_geotiff_gpu, which already accept 'crs' as either "
            "an int EPSG code or a WKT string.",
            DeprecationWarning,
            stacklevel=2,
        )
        crs = crs_wkt

    # Reject bool / non-numeric nodata at the entry point so write_vrt
    # matches the to_geotiff / write_geotiff_gpu surface. ``bool`` is a
    # subclass of ``int`` in Python, so a typo like ``nodata=True`` would
    # slip past every downstream ``isinstance(nodata, (int, float))``
    # guard and the VRT XML emitter would write ``<NoDataValue>True
    # </NoDataValue>``. No reader parses ``"True"`` as numeric, so the
    # round-trip would silently drop the sentinel. See issue #1921.
    _validate_nodata_arg(nodata)

    resolved_wkt = _resolve_crs_to_wkt(crs)

    from .._vrt import write_vrt as _write_vrt_internal
    return _write_vrt_internal(
        path, source_files,
        relative=relative,
        crs_wkt=resolved_wkt,
        nodata=nodata,
    )
