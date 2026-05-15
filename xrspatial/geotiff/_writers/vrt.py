"""VRT writer entry point.

Wraps ``_vrt.write_vrt`` with the public ``write_vrt`` surface:
deprecation handling for ``crs_wkt`` (#1715), normalisation of the
``crs`` kwarg to WKT via ``_resolve_crs_to_wkt``, and the parity
docstring vs ``to_geotiff`` / ``write_geotiff_gpu``.
"""
from __future__ import annotations

import warnings

from .._crs import _resolve_crs_to_wkt
from .._runtime import _CRS_WKT_DEPRECATED_SENTINEL


def write_vrt(vrt_path: str, source_files: list[str], *,
              relative: bool = True,
              crs: int | str | None = None,
              crs_wkt: str | None = _CRS_WKT_DEPRECATED_SENTINEL,
              nodata: float | int | None = None) -> str:
    """Generate a VRT file that mosaics multiple GeoTIFF tiles.

    Parameters
    ----------
    vrt_path : str
        Output .vrt file path.
    source_files : list of str
        Paths to the source GeoTIFF files.
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

    resolved_wkt = _resolve_crs_to_wkt(crs)

    from .._vrt import write_vrt as _write_vrt_internal
    return _write_vrt_internal(
        vrt_path, source_files,
        relative=relative,
        crs_wkt=resolved_wkt,
        nodata=nodata,
    )
