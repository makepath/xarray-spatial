"""Centralised VRT capability validation (issue #2329 / parent #2321).

A single :func:`validate_parsed_vrt` entry point that audits an
already-parsed :class:`xrspatial.geotiff._vrt.VRTDataset` against every
capability the read pipeline does not honour. Both the direct
``read_vrt`` entry point in ``_backends/vrt.py`` and the dispatched
``open_geotiff('foo.vrt')`` branch in ``__init__.py`` call this
validator before any source bytes are decoded, so the two entry points
produce equivalent failures for the same bad input.

Why centralise: prior to this module the capability checks were spread
across ``_vrt.read_vrt`` (per-source ``SrcRect`` / ``DstRect`` rejects
mid-decode), ``_backends/vrt.read_vrt`` (CRS / rotated-transform /
mixed-nodata checks via ``validate_read_metadata`` at the eager
boundary), and ``_check_resample_alg_supported`` (per-source resample
reject at the placement site). Under chunked dispatch the per-source
checks fired inside dask tasks, so a malformed VRT could build a graph
successfully and then blow up deep in a ``compute()``. The validator
moves all of those checks to graph-build / eager-read setup time.

The function never re-parses the VRT XML. Callers pass an already-
parsed ``VRTDataset`` plus the source path (used only for error
messages). Raises :class:`VRTUnsupportedError` on the first failure
with the offending source path and field embedded in the message.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._errors import (
    RotatedTransformError,
    UnparseableCRSError,
    VRTUnsupportedError,
)

if TYPE_CHECKING:
    from ._vrt import VRTDataset


# Numpy dtype kinds the VRT read path can serve correctly. Bool / complex
# / object / void / string fall outside the supported kind set: the
# decode + sentinel-mask pipeline assumes scalar numeric pixels.
_SUPPORTED_DTYPE_KINDS = frozenset({'u', 'i', 'f'})


def _is_nearest_resample(alg: str | None) -> bool:
    """True iff ``alg`` is a recognised nearest-neighbour spelling.

    Delegates to the canonical ``_NEAREST_RESAMPLE_ALGS`` set in
    ``_vrt.py``. Lazy-imported here so the validator stays a leaf
    module with no eager back-reference to ``_vrt``, while still
    matching whatever ``_vrt`` recognises as nearest. Avoids the
    drift risk of a hand-copied set.
    """
    if alg is None:
        return True
    from ._vrt import _NEAREST_RESAMPLE_ALGS
    return alg.strip().lower() in _NEAREST_RESAMPLE_ALGS


def validate_parsed_vrt(
    parsed: 'VRTDataset',
    *,
    source: str,
    mode: str = 'read',
    allow_rotated: bool = False,
    allow_unparseable_crs: bool = False,
    band_nodata: str | None = None,
) -> None:
    """Audit a parsed VRT for capability mismatches before read execution.

    Parameters
    ----------
    parsed : VRTDataset
        The already-parsed VRT structure. The validator never re-parses
        XML; callers feed in the output of ``parse_vrt`` or the value
        threaded through ``read_vrt(..., parsed=...)``.
    source : str
        Path to the ``.vrt`` file. Used only for error messages so a
        caller can locate the offending file without re-parsing.
    mode : {'read'}
        Reserved for a future write-side validator. Currently only
        ``'read'`` is supported; other modes are rejected.
    allow_rotated : bool, default False
        Caller opt-in for non-axis-aligned ``GeoTransform``. ``False``
        rejects any non-zero rotation / shear term.
    allow_unparseable_crs : bool, default False
        Caller opt-in for CRS strings that pyproj cannot resolve and
        that do not structurally look like WKT. ``False`` rejects the
        string with :class:`VRTUnsupportedError`.
    band_nodata : {None, 'first'}, optional
        Mixed-band-nodata policy. ``None`` (the default) rejects a VRT
        whose bands declare disagreeing ``<NoDataValue>`` sentinels;
        ``'first'`` keeps the legacy flatten-to-band-0 behaviour and
        the validator returns silently for the mixed case.

    Raises
    ------
    VRTUnsupportedError
        On the first capability mismatch encountered. The message names
        the offending source path and field so the caller can locate
        the bad source without re-parsing the VRT XML themselves.

    Notes
    -----
    The rules audited here are not exhaustive of every VRT failure
    mode -- the parse step (``parse_vrt``) already rejects unknown
    ``dataType`` tokens, source paths that escape the VRT directory,
    and so on. The validator covers the residual capability checks
    that previously fired mid-decode (or per chunk task under chunked
    dispatch).

    Overlap with registered ``validate_read_metadata`` hooks
    --------------------------------------------------------
    The rotated-transform and unparseable-CRS branches in this
    validator overlap with the registered hooks
    ``_check_read_rotated_transform`` and
    ``_check_read_unparseable_crs`` in :mod:`._validation`. The VRT
    call sites in ``_backends/vrt.py`` call this validator first and
    ``validate_read_metadata`` immediately after, so the validator's
    branches preempt the registered hooks for the VRT path: the
    registered hooks remain reachable only when the validator passes
    the case through (e.g. ``allow_rotated=True``, or a non-VRT call
    site that does not run this validator at all). The duplication
    is intentional -- the validator embeds the source path in the
    message and lifts the check ahead of any decode -- but maintainers
    editing the registered hooks should keep both copies in sync, or
    fold one into the other.
    """
    if mode != 'read':
        raise ValueError(
            f"validate_parsed_vrt only supports mode='read', got "
            f"{mode!r}. The write-side validator is reserved for a "
            f"future change."
        )

    # Rule 1: band count sanity.
    #
    # A VRT with zero ``<VRTRasterBand>`` elements would land in
    # ``_effective_dtype_for_bands`` (which raises a ValueError with a
    # generic message) or, worse, build an empty graph. Reject up front
    # so the message names the file. The "no <VRTRasterBand>" substring
    # keeps backward compatibility with the existing message check in
    # test_vrt_multiband_dtype_1696.py.
    if not parsed.bands:
        raise VRTUnsupportedError(
            f"VRT '{source}' declares no <VRTRasterBand> elements. "
            f"The read pipeline cannot serve a band-less mosaic; the "
            f"VRT must declare at least one raster band."
        )

    # Rule 2: dtype compatibility.
    #
    # ``parse_vrt`` already rejects unknown ``dataType`` tokens (#1783),
    # so this branch only fires when a future ``_DTYPE_MAP`` change
    # adds a complex / bool / object entry, or when a caller hand-builds
    # a ``VRTDataset``. Surface the bad band number and dtype so the
    # message is actionable.
    for band in parsed.bands:
        if band.dtype.kind not in _SUPPORTED_DTYPE_KINDS:
            raise VRTUnsupportedError(
                f"VRT '{source}' band {band.band_num} declares dtype "
                f"{band.dtype!r}, which the read pipeline does not "
                f"support. Supported dtype kinds: unsigned integer "
                f"('u'), signed integer ('i'), and float ('f'). "
                f"Complex / bool / object bands fall outside the "
                f"decode + sentinel-mask pipeline."
            )

    # Rule 3: transform orientation.
    #
    # A non-zero skew_x (gt[2]) or skew_y (gt[4]) means the VRT is
    # rotated / sheared. The read pipeline treats those as no-georef
    # (#2122) only when the caller opts in via ``allow_rotated=True``;
    # otherwise the projected coords would mislabel a pixel grid as a
    # georeferenced raster. Reject up front to mirror the read-side
    # ``RotatedTransformError`` contract.
    gt = parsed.geo_transform
    if gt is not None and (gt[2] != 0.0 or gt[4] != 0.0) and not allow_rotated:
        # Use the existing typed error so callers that already
        # ``except RotatedTransformError`` keep catching this case. The
        # message names the source path so the centralised validator
        # still adds value over the registered check in
        # ``validate_read_metadata``.
        raise RotatedTransformError(
            f"VRT '{source}' GeoTransform has non-zero rotation/shear "
            f"terms (skew_x={gt[2]}, skew_y={gt[4]}). The read pipeline "
            f"treats the array as no-georef under allow_rotated=True; "
            f"pass allow_rotated=True to opt in, or re-export the VRT "
            f"with an axis-aligned GeoTransform."
        )

    # Rule 4: pixel-size compatibility.
    #
    # A zero ``res_x`` (gt[1]) or ``res_y`` (gt[5]) would produce a
    # degenerate coord array (every coordinate equals the origin) and
    # divide-by-zero in any pixel-to-world conversion downstream. The
    # eager read path would currently surface this as an opaque coord
    # generation error; raise here instead with the file name.
    if gt is not None:
        # Skip the rotated case: it already raised above unless the
        # caller opted in via allow_rotated, in which case the pipeline
        # treats the array as no-georef and the pixel size is moot.
        is_rotated = (gt[2] != 0.0 or gt[4] != 0.0)
        if not is_rotated and (gt[1] == 0.0 or gt[5] == 0.0):
            raise VRTUnsupportedError(
                f"VRT '{source}' GeoTransform declares a zero pixel "
                f"size (res_x={gt[1]}, res_y={gt[5]}). The read pipeline "
                f"divides by these to compute pixel-to-world coordinates; "
                f"a zero term produces a degenerate raster. Re-export "
                f"the VRT with non-zero res_x / res_y."
            )

    # Rule 5: nodata policy (mixed per-band sentinels).
    #
    # Delegate the disagreement check to the existing
    # ``_check_read_mixed_band_metadata`` registered hook so the
    # ``MixedBandMetadataError`` message and the ``None``-counts-as-
    # undeclared semantics stay canonical in one place. The validator's
    # job here is to reject typo'd ``band_nodata`` values up front
    # (matching the boundary-level value check in ``_backends/vrt.py``
    # and ``__init__.py``) so a typo never silently degrades to strict
    # mode at the registered hook below.
    if band_nodata not in (None, 'first'):
        raise ValueError(
            f"band_nodata must be None or 'first', got {band_nodata!r}."
        )

    # Rule 6: CRS compatibility (unparseable / not-WKT-shaped).
    #
    # An empty ``<SRS>`` (``crs_wkt == ''``) is the no-CRS case and is
    # legal; the read path lands ``crs_wkt=None`` and emits the no-georef
    # marker. A non-empty CRS string that does not structurally look
    # like WKT and that pyproj cannot resolve is rejected unless the
    # caller opted in. Defer the pyproj probe to ``_wkt_to_epsg`` so a
    # missing-pyproj host still rejects the obviously-malformed case
    # (the ``_looks_like_wkt`` cheap check).
    crs_wkt = parsed.crs_wkt
    if crs_wkt and not allow_unparseable_crs:
        # Lazy-import the WKT structural check from ``_crs`` so the two
        # call sites cannot drift on which root keywords count as WKT.
        from ._crs import _looks_like_wkt
        if not _looks_like_wkt(crs_wkt):
            # Probe pyproj directly so a valid PROJ / EPSG token passes
            # when pyproj is installed, without going through
            # ``_wkt_to_epsg``. The wrapper emits a
            # ``GeoTIFFFallbackWarning`` on parse failure; the
            # registered ``_check_read_unparseable_crs`` hook in
            # ``_validation.py`` does not. Routing through the wrapper
            # would emit a warning here that the existing check would
            # not have emitted, which is a small but visible regression
            # in the unparseable-CRS path. Match the existing hook's
            # silent-probe behaviour.
            try:
                from pyproj import CRS as _PyProjCRS
                from pyproj.exceptions import CRSError as _PyProjCRSError
            except ImportError:
                # No pyproj available: cannot prove the CRS is bad, so
                # mirror the registered hook's no-op behaviour and let
                # downstream code see the unparseable string.
                pass
            else:
                try:
                    _PyProjCRS.from_user_input(crs_wkt)
                except _PyProjCRSError:
                    # Use the existing typed error so callers that
                    # already ``except UnparseableCRSError`` keep
                    # catching this case. Adds the source path to the
                    # message.
                    raise UnparseableCRSError(
                        f"VRT '{source}' declares a CRS string that "
                        f"does not look like WKT and that pyproj could "
                        f"not resolve: {crs_wkt!r}. Pass "
                        f"allow_unparseable_crs=True to read the VRT "
                        f"anyway, or re-export the VRT with a "
                        f"WKT-formatted <SRS>."
                    )

    # Rule 7 / 8: SrcRect and DstRect sanity, plus the supported
    # resampling set. Walk every source on every band so the message
    # names the offending source path and field.
    for band in parsed.bands:
        for src in band.sources:
            sr = src.src_rect
            dr = src.dst_rect

            # Rule 7a: negative SrcRect size.
            if sr.x_size < 0 or sr.y_size < 0:
                raise VRTUnsupportedError(
                    f"VRT '{source}' SimpleSource '{src.filename}' "
                    f"(band {band.band_num}) has negative SrcRect size "
                    f"(xSize={sr.x_size}, ySize={sr.y_size}); SrcRect "
                    f"sizes must be non-negative."
                )
            # Rule 7b: negative SrcRect offset.
            if sr.x_off < 0 or sr.y_off < 0:
                raise VRTUnsupportedError(
                    f"VRT '{source}' SimpleSource '{src.filename}' "
                    f"(band {band.band_num}) has negative SrcRect offset "
                    f"(xOff={sr.x_off}, yOff={sr.y_off}); SrcRect "
                    f"offsets must be non-negative."
                )

            # Rule 8a: negative DstRect size.
            if dr.x_size < 0 or dr.y_size < 0:
                raise VRTUnsupportedError(
                    f"VRT '{source}' SimpleSource '{src.filename}' "
                    f"(band {band.band_num}) has negative DstRect size "
                    f"(xSize={dr.x_size}, ySize={dr.y_size}); DstRect "
                    f"sizes must be non-negative."
                )

            # Rule 8b: DstRect outside the VRT extent. A source whose
            # destination rect lies entirely outside the
            # ``rasterXSize x rasterYSize`` window contributes no pixels
            # to the mosaic; the read path silently skips it via the
            # clip-to-window check. Refuse the malformed mosaic up
            # front so the message names the bad source.
            dst_r0 = dr.y_off
            dst_c0 = dr.x_off
            dst_r1 = dr.y_off + dr.y_size
            dst_c1 = dr.x_off + dr.x_size
            if (dst_r0 >= parsed.height or dst_c0 >= parsed.width
                    or dst_r1 <= 0 or dst_c1 <= 0):
                raise VRTUnsupportedError(
                    f"VRT '{source}' SimpleSource '{src.filename}' "
                    f"(band {band.band_num}) has a DstRect "
                    f"(xOff={dr.x_off}, yOff={dr.y_off}, "
                    f"xSize={dr.x_size}, ySize={dr.y_size}) entirely "
                    f"outside the VRT extent "
                    f"({parsed.height}x{parsed.width}); the source "
                    f"would contribute no pixels."
                )

            # Rule 9: unsupported resampling. Only matters when the
            # SrcRect / DstRect sizes differ (the read path falls back
            # to a no-op placement when they match, regardless of
            # ``<ResampleAlg>``). Matches the ``_check_resample_alg_supported``
            # gate in ``_vrt.py`` but lifts it before any pixel decode
            # so the chunked path raises at graph build too.
            needs_resample = (
                sr.x_size != dr.x_size or sr.y_size != dr.y_size
            )
            if needs_resample and not _is_nearest_resample(src.resample_alg):
                raise VRTUnsupportedError(
                    f"VRT '{source}' ComplexSource '{src.filename}' "
                    f"(band {band.band_num}) requests "
                    f"<ResampleAlg>{src.resample_alg}</ResampleAlg> "
                    f"with differing SrcRect "
                    f"({sr.x_size}x{sr.y_size}) and DstRect "
                    f"({dr.x_size}x{dr.y_size}) sizes. The read "
                    f"pipeline only implements nearest-neighbour "
                    f"resampling; substituting nearest would silently "
                    f"mislabel the output. Re-export with "
                    f"<ResampleAlg>Nearest</ResampleAlg> or matching "
                    f"SrcRect/DstRect sizes."
                )


__all__ = ['validate_parsed_vrt']
