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

import struct
from typing import TYPE_CHECKING

from ._errors import RotatedTransformError, UnparseableCRSError, VRTUnsupportedError

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
    #
    # Rule 10 (source CRS mismatch) needs the file metadata of every
    # source, but only for sources that survive the cheaper rect/mask/
    # resample rejects below. We collect the unique source paths during
    # this walk and probe them in a single deduplicated pass afterwards
    # so a multi-band VRT that references the same TIFF on each band
    # opens the file once rather than per-band.
    _unique_source_paths: list[str] = []
    _seen_source_paths: set[str] = set()
    for band in parsed.bands:
        for src in band.sources:
            sr = src.src_rect
            dr = src.dst_rect

            # Rule 6b: nested VRT. A SourceFilename ending in ``.vrt``
            # would recurse the read pipeline through a second VRT
            # parse, which the mosaic reader does not implement. Catch
            # the case here (not at parse time) so the validator owns
            # every capability rejection and the message names both
            # outer and inner VRT paths. Case-insensitive on the
            # extension so ``.VRT`` (Windows-style) also trips the
            # rejection. See issue #2371.
            if src.filename.lower().endswith('.vrt'):
                # Direct interpolation (not !r) for ``src.filename`` so
                # Windows paths render with single backslashes rather
                # than the doubled escapes ``repr`` emits, matching the
                # ``parse_vrt`` pattern at the path-containment check.
                # Without this, a callers ``in`` check against the raw
                # Windows path would fail because ``repr`` doubles
                # every backslash.
                raise VRTUnsupportedError(
                    f"VRT '{source}' references another VRT as a source "
                    f"('{src.filename}', band {band.band_num}). Nested "
                    f"VRTs are not a supported feature in this release; "
                    f"the mosaic reader assembles pixel data from "
                    f"GeoTIFF sources only. See "
                    f"`xrspatial.geotiff.SUPPORTED_FEATURES` for the "
                    f"release tier map. Materialise the inner VRT to a "
                    f"GeoTIFF with ``gdal_translate`` (or "
                    f"``xrspatial.geotiff.to_geotiff`` after reading "
                    f"the inner VRT separately) and reference the "
                    f"resulting GeoTIFF in the outer VRT instead."
                )

            # Rule 6c: complex mask / alpha source semantics.
            # ``<UseMaskBand>true</UseMaskBand>`` and per-source
            # ``<MaskBand>`` children declare that the source's
            # per-band mask drives placement. The read pipeline ignores
            # the mask, so the per-pixel mask would silently drop and
            # the dispatched array would mis-label every masked pixel
            # as valid. Reject with the source path and the offending
            # flag. See issue #2371.
            if src.use_mask_band:
                raise VRTUnsupportedError(
                    f"VRT '{source}' source '{src.filename}' (band "
                    f"{band.band_num}) declares "
                    f"<UseMaskBand>true</UseMaskBand>. The read "
                    f"pipeline does not honour per-source mask bands "
                    f"and would silently drop the per-pixel mask, "
                    f"mis-labelling masked pixels as valid. Re-export "
                    f"the source with the mask burned into the band's "
                    f"nodata sentinel and drop the <UseMaskBand> flag."
                )
            if src.has_mask_source:
                raise VRTUnsupportedError(
                    f"VRT '{source}' source '{src.filename}' (band "
                    f"{band.band_num}) declares a per-source <MaskBand> "
                    f"child. The read pipeline does not honour mask "
                    f"bands and would silently drop the per-pixel "
                    f"mask. Re-export the source with the mask burned "
                    f"into the band's nodata sentinel and drop the "
                    f"<MaskBand> child."
                )

            # Rule 7a: negative SrcRect size. Keep the "SrcRect ...
            # negative size" phrasing so the legacy regex pattern in
            # ``test_geotiff_vrt_srcrect_validation_1784.py`` still
            # matches now that the validator preempts the per-source
            # check that originally raised this message.
            if sr.x_size < 0 or sr.y_size < 0:
                raise VRTUnsupportedError(
                    f"VRT '{source}' SimpleSource '{src.filename}' "
                    f"(band {band.band_num}) SrcRect has negative size "
                    f"(xSize={sr.x_size}, ySize={sr.y_size}); SrcRect "
                    f"sizes must be non-negative."
                )
            # Rule 7b: negative SrcRect offset. Same phrasing rationale
            # as Rule 7a (legacy regex match).
            if sr.x_off < 0 or sr.y_off < 0:
                raise VRTUnsupportedError(
                    f"VRT '{source}' SimpleSource '{src.filename}' "
                    f"(band {band.band_num}) SrcRect has negative offset "
                    f"(xOff={sr.x_off}, yOff={sr.y_off}); SrcRect "
                    f"offsets must be non-negative."
                )

            # Rule 8a: negative DstRect size. Same phrasing rationale
            # as Rule 7a (legacy regex match).
            if dr.x_size < 0 or dr.y_size < 0:
                raise VRTUnsupportedError(
                    f"VRT '{source}' SimpleSource '{src.filename}' "
                    f"(band {band.band_num}) DstRect has negative size "
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
                    f"SrcRect/DstRect sizes. See issue #1751."
                )

            # Collect the source for the Rule 10 CRS sweep below. We
            # only reach this line after every cheaper rule has accepted
            # the source, so a bad rect / mask / resample still raises
            # before we open any file for metadata. Path is the parsed
            # ``_Source.filename``, which ``parse_vrt`` has already
            # resolved to an absolute / VRT-relative path that survived
            # the directory-containment check.
            if src.filename not in _seen_source_paths:
                _seen_source_paths.add(src.filename)
                _unique_source_paths.append(src.filename)

    # Rule 10: per-source CRS agreement (#2321 mixed-CRS gap, #2444).
    #
    # The VRT mosaic reader has no reprojection step: every source's
    # pixels are placed by integer offset into a single output buffer
    # whose CRS is taken from the VRT-declared ``<SRS>`` (or, when the
    # VRT carries no ``<SRS>``, the first source's CRS). If a source
    # carries a CRS that disagrees with the VRT-declared SRS, the read
    # path silently composites those pixels into a mosaic whose
    # ``attrs['crs']`` advertises one CRS while some pixels actually
    # come from another. The output is no longer geospatially
    # meaningful, but nothing in the eager or chunked path objects.
    #
    # Move the rejection here so the typed error fires before any
    # decode, with the offending source path and CRS named. The check
    # runs at graph build (chunked path) and eager-read setup (eager
    # path) via the existing call sites in ``_backends/vrt.py``, so a
    # mixed-CRS VRT is refused at the same boundary as the other
    # capability checks above.
    _check_mixed_source_crs(
        _unique_source_paths,
        vrt_crs_wkt=parsed.crs_wkt,
        source=source,
    )


def _check_mixed_source_crs(
    source_paths: list[str],
    *,
    vrt_crs_wkt: str | None,
    source: str,
) -> None:
    """Rule 10: reject a VRT whose sources disagree on CRS.

    Walks ``source_paths`` (already deduplicated by the caller), opens
    each one for metadata-only IFD parsing, and pulls
    ``crs_wkt`` / ``crs_epsg`` via :func:`extract_geo_info`. Compares
    each source CRS to the VRT-declared ``vrt_crs_wkt`` after pyproj
    canonicalisation. When ``vrt_crs_wkt`` is empty / None, the
    sources are required to agree with each other (the first parseable
    source becomes the reference).

    Skip conditions (no raise):

    * pyproj is not installed. Mirrors the disposition of
      ``_check_write_conflicting_crs`` in ``_validation.py``: a string
      compare would false-positive on equivalent representations (e.g.
      ``"EPSG:4326"`` vs. the canonical WGS84 WKT), and a host without
      pyproj cannot prove inequality, so we cannot fail closed.
    * A source is unreadable, has no parseable header, or carries no
      CRS at all. The first two land on the existing ``missing_sources``
      / ``parse_header`` rejection paths during the real read; the
      third is the "VRT inherits its CRS from the source" case that the
      reader has always handled.
    * A source CRS string is non-empty but pyproj cannot canonicalise
      it. Conservative: we can prove the string is broken but cannot
      prove inequality against the VRT-declared SRS, and the read path
      will surface its own typed error on the unparseable CRS.

    Raises
    ------
    VRTUnsupportedError
        On the first source CRS that pyproj resolves to a different
        :class:`pyproj.CRS` than the reference. Message names the
        source path, the source CRS (EPSG when available, else a
        truncated WKT excerpt), and the VRT-declared SRS for contrast.
    """
    if not source_paths:
        return

    # pyproj is the only way to canonicalise. Mirror the
    # ``_check_write_conflicting_crs`` skip-on-missing disposition so
    # the validator does not regress hosts without pyproj.
    try:
        from pyproj import CRS as _PyProjCRS
        from pyproj.exceptions import CRSError as _PyProjCRSError
    except ImportError:
        return

    # Parse the VRT-declared SRS into a pyproj CRS if it is non-empty.
    # ``parsed.crs_wkt`` carries the raw ``<SRS>`` text, which may be a
    # WKT block or a short token like ``"EPSG:4326"`` / ``"+proj=..."``.
    # An empty / None value means "no VRT-level SRS"; the reference CRS
    # then becomes the first parseable source.
    vrt_crs = None
    vrt_crs_display: str | None = None
    raw_vrt_srs = vrt_crs_wkt
    if raw_vrt_srs:
        try:
            vrt_crs = _PyProjCRS.from_user_input(raw_vrt_srs)
            vrt_crs_display = _format_crs_for_message(raw_vrt_srs, vrt_crs)
        except _PyProjCRSError:
            # The unparseable-SRS case is already handled by Rule 6
            # above (with the ``allow_unparseable_crs`` opt-in). If we
            # got here, either the caller opted in or the string passed
            # the ``_looks_like_wkt`` cheap check but pyproj still
            # rejects it. Either way, we cannot canonicalise the VRT
            # side, so we cannot prove a source disagrees with it.
            return

    # Lazy imports of the on-disk metadata extractors. Keep them inside
    # this helper so the module stays a leaf with no eager dependency
    # on the reader / header parsers; the centralised validator pays
    # the import cost only when it actually needs to open sources.
    from ._geotags import extract_geo_info
    from ._header import parse_all_ifds, parse_header
    from ._sources import _FileSource

    reference_crs = vrt_crs
    reference_display = vrt_crs_display
    reference_source: str | None = None  # None == VRT-declared

    for src_path in source_paths:
        try:
            with _FileSource(src_path) as src_handle:
                data = src_handle.read_all()
                header = parse_header(data)
                ifds = parse_all_ifds(data, header)
                if not ifds:
                    continue
                geo = extract_geo_info(ifds[0], data, header.byte_order)
        except (OSError, ValueError, struct.error):
            # Unreadable / malformed source. The real read path
            # surfaces this via the ``missing_sources`` contract
            # (default ``'raise'``); the validator deliberately
            # stays quiet so the canonical error fires from there
            # with the existing message and code path.
            continue

        src_crs_raw = geo.crs_wkt
        if not src_crs_raw and geo.crs_epsg is not None:
            src_crs_raw = f"EPSG:{geo.crs_epsg}"
        if not src_crs_raw:
            # Source has no CRS at all. The VRT can legitimately
            # provide one in this case (reader inherits the
            # VRT-declared SRS), so skip.
            continue

        try:
            src_crs = _PyProjCRS.from_user_input(src_crs_raw)
        except _PyProjCRSError:
            # Cannot canonicalise the source CRS; cannot prove
            # disagreement. The decode path will surface its own
            # error if the broken CRS matters.
            continue

        if reference_crs is None:
            # No VRT-level SRS and this is the first source we
            # could read; use it as the reference for the rest.
            reference_crs = src_crs
            reference_display = _format_crs_for_message(
                src_crs_raw, src_crs,
            )
            reference_source = src_path
            continue

        if src_crs.equals(reference_crs):
            continue

        src_display = _format_crs_for_message(src_crs_raw, src_crs)
        if reference_source is None:
            # Disagreement against the VRT-declared SRS.
            raise VRTUnsupportedError(
                f"VRT '{source}' source '{src_path}' carries CRS "
                f"{src_display}, which does not match the "
                f"VRT-declared <SRS> ({reference_display}). The "
                f"mosaic reader has no reprojection step, so the "
                f"two sets of pixels cannot be composited into a "
                f"single CRS without misplacing one of them. "
                f"Reproject the source to the VRT-declared CRS "
                f"(e.g. ``gdalwarp -t_srs``) before referencing it, "
                f"or split the mosaic into per-CRS VRTs. See issue "
                f"#2321."
            )
        # Disagreement among sources (no VRT-level SRS).
        raise VRTUnsupportedError(
            f"VRT '{source}' has no <SRS> and its sources disagree "
            f"on CRS: '{reference_source}' carries "
            f"{reference_display} but '{src_path}' carries "
            f"{src_display}. The mosaic reader has no reprojection "
            f"step, so the sources cannot be composited into a "
            f"single CRS. Reproject every source to a common CRS "
            f"(e.g. ``gdalwarp -t_srs``) before assembling the VRT, "
            f"or declare an authoritative <SRS> at the VRT level "
            f"and use sources whose CRS already matches it. See "
            f"issue #2321."
        )


def _format_crs_for_message(raw: str, crs) -> str:
    """Render a CRS for inclusion in a ``VRTUnsupportedError`` message.

    Prefers ``EPSG:<code>`` when pyproj resolves the input to a known
    EPSG code (the common case for a well-formed source TIFF). Falls
    back to a truncated excerpt of the raw text otherwise so the
    message stays readable for hand-built or non-EPSG WKT strings.

    ``CRS.to_epsg()`` is documented to return ``None`` on failure
    rather than raise, so the call is unguarded; we still defend
    against a ``crs`` that does not expose ``to_epsg`` at all (the
    helper accepts an untyped ``crs`` because tests and future callers
    may pass a mock or a parsed-but-not-pyproj CRS shim).
    """
    epsg = None
    to_epsg = getattr(crs, 'to_epsg', None)
    if callable(to_epsg):
        epsg = to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    raw = raw.strip()
    if len(raw) <= 60:
        return repr(raw)
    return repr(raw[:57] + '...')


# Public alias matching the issue #2371 / epic #2342 naming. The
# implementation continues to live under ``validate_parsed_vrt`` for
# backward compatibility with the ``_backends/vrt.py`` call sites and
# the existing test files; new call sites should prefer the
# capability-validator spelling.
validate_vrt_capability = validate_parsed_vrt


__all__ = ['validate_parsed_vrt', 'validate_vrt_capability']
