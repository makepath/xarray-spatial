"""VRT read backend: ``read_vrt`` and its dask helpers.

Step 9 of issue #1813. The XML parsing, source-path containment,
per-source decode, and integer-sentinel masking already live in
``xrspatial/geotiff/_vrt.py``; this module holds only the orchestration
that picks among the eager, dask, and GPU paths exposed through the
public ``read_vrt`` entry point.
"""
from __future__ import annotations

import math

import numpy as np
import xarray as xr

from .._attrs import (
    _finalize_lazy_read_attrs,
)
from .._coords import (
    coords_from_pixel_geometry as _coords_from_pixel_geometry,
)
from .._crs import _wkt_to_epsg
from .._geotags import (
    RASTER_PIXEL_IS_AREA,
    RASTER_PIXEL_IS_POINT,
    GeoInfo,
    GeoTransform,
)
from .._reader import _MAX_CLOUD_BYTES_SENTINEL
from .._runtime import _ON_GPU_FAILURE_SENTINEL
from .._validation import (
    _gdal_geotransform_to_affine_tuple,
    _validate_chunks_arg,
    _validate_dispatch_kwargs,
    _validate_dtype_cast,
)


# Hard cap on the per-VRT chunk task count. Matches the
# ``_MAX_DASK_CHUNKS`` value used by ``read_geotiff_dask`` so the two
# entry points refuse the same scheduler-busting chunk grids. See
# issue #1814.
_MAX_VRT_DASK_CHUNKS = 50_000


def _vrt_to_synthetic_geo_info(vrt) -> GeoInfo:
    """Project a parsed ``VRTDataset`` onto a ``GeoInfo`` for the helpers.

    Wave 3 of #2162. Lets the VRT eager and chunked paths feed the shared
    ``_finalize_lazy_read_attrs`` helper instead of building
    ``GeoTIFFMetadata`` from VRT internals by hand. The synthesis follows
    the existing VRT contract:

    * The rotated-VRT case (non-zero skew terms on the GDAL GeoTransform,
      opened with ``allow_rotated=True``) sets ``has_georef=False`` and
      stamps ``rotated_affine`` on the embedded ``GeoTransform``. The
      ``geo_info_to_metadata`` rotated-opt-in branch then drops the
      ``crs`` / ``crs_wkt`` / ``transform`` attrs and emits
      ``rotated_affine`` plus the no-georef marker, matching the rotated
      branch on the non-VRT path (#2122 / #2129).
    * The plain no-``<GeoTransform>`` case sets ``has_georef=False`` with
      a default-unit ``GeoTransform``. ``geo_info_to_metadata`` lands on
      ``transform=None`` and emits only the no-georef marker plus CRS
      attrs when the VRT carries a ``<SRS>``.
    * The axis-aligned georeferenced case sets ``has_georef=True`` with
      the parsed origin / pixel size.

    ``vrt.crs_wkt`` carries an empty string when the VRT XML has a
    ``<SRS>`` element with no recognised CRS body; treat empty as absent
    so the helper does not emit ``attrs['crs_wkt']=''``. CRS resolution
    to an EPSG code mirrors the inline code from the pre-migration
    branch (issue #1734 / #2122 / #2136 / #2139).

    The synthesised ``GeoInfo`` carries only the fields the VRT path
    actually owns: transform, has_georef, crs_epsg / crs_wkt,
    raster_type. ``extra_tags`` / ``gdal_metadata`` / resolution tags
    stay at their dataclass defaults so the helper's
    ``geo_info_to_metadata`` does not synthesise attrs the VRT branch
    never emitted. ``vrt_holes`` is documented divergence: ``GeoInfo``
    has no slot for it, so the call site stamps the attr post-helper.

    Issue #2225: the resolved ``georef_status`` for this synthesised
    ``GeoInfo`` flows through the shared
    :func:`xrspatial.geotiff._coords.resolve_georef` resolver via
    :func:`_compute_georef_status`, so the VRT branch lands on the
    same bucket as the eager / dask / GPU read paths for matching
    input shapes (full / transform_only / crs_only / none /
    rotated_dropped).
    """
    gt = vrt.geo_transform
    is_rotated = gt is not None and (gt[2] != 0.0 or gt[4] != 0.0)
    keep_crs = bool(vrt.crs_wkt) and not is_rotated
    crs_epsg = _wkt_to_epsg(vrt.crs_wkt) if keep_crs else None
    crs_wkt = vrt.crs_wkt if keep_crs else None
    raster_type = (
        RASTER_PIXEL_IS_POINT if vrt.raster_type == 'point'
        else RASTER_PIXEL_IS_AREA
    )
    if is_rotated:
        rotated_affine = _gdal_geotransform_to_affine_tuple(gt)
        return GeoInfo(
            transform=GeoTransform(rotated_affine=tuple(rotated_affine)),
            has_georef=False,
            crs_epsg=crs_epsg,
            crs_wkt=crs_wkt,
            raster_type=raster_type,
        )
    if gt is None:
        return GeoInfo(
            transform=GeoTransform(),
            has_georef=False,
            crs_epsg=crs_epsg,
            crs_wkt=crs_wkt,
            raster_type=raster_type,
        )
    origin_x, res_x, _, origin_y, _, res_y = gt
    return GeoInfo(
        transform=GeoTransform(
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            pixel_width=float(res_x),
            pixel_height=float(res_y),
        ),
        has_georef=True,
        crs_epsg=crs_epsg,
        crs_wkt=crs_wkt,
        raster_type=raster_type,
    )


def read_vrt(source: str, *,
             dtype: str | np.dtype | None = None,
             window: tuple | None = None,
             overview_level: int | None = None,
             band: int | None = None,
             name: str | None = None,
             chunks: int | tuple | None = None,
             gpu: bool = False,
             max_pixels: int | None = None,
             max_cloud_bytes: int | None = _MAX_CLOUD_BYTES_SENTINEL,  # type: ignore[assignment]
             on_gpu_failure: str = _ON_GPU_FAILURE_SENTINEL,
             missing_sources: str = 'raise',
             allow_rotated: bool = False,
             allow_unparseable_crs: bool = False,
             band_nodata: str | None = None,
             mask_nodata: bool = True) -> xr.DataArray:
    """Read a GDAL Virtual Raster Table (.vrt) into an xarray.DataArray.

    Tier: Advanced (issue #2137). VRT mosaics work and are tested, but
    the caller should know the failure modes: cross-source nodata can
    disagree (gated by ``band_nodata``), backing files can be missing
    (gated by ``missing_sources``), and per-band metadata mismatch
    raises a typed error rather than silently flattening. See
    :data:`xrspatial.geotiff.SUPPORTED_FEATURES` for the full tier map.

    The VRT's source GeoTIFFs are read via windowed reads and assembled
    into a single array.

    Parameters
    ----------
    source : str
        Path to the .vrt file.
    dtype : str, numpy.dtype, or None
        Cast the result to this dtype after reading. None keeps the
        file's native dtype. Float-to-int casts raise ValueError.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop) for windowed reading.
    band : int or None
        Band index (0-based). None returns all bands.
    name : str or None
        Name for the DataArray.
    chunks : int, tuple, or None
        If set, return a Dask-chunked DataArray. int for square chunks,
        (row, col) tuple for rectangular.
    gpu : bool
        If True, return a CuPy-backed DataArray on GPU.
    max_pixels : int or None
        Maximum allowed pixel count (width * height * samples) for the
        assembled VRT region. None uses the reader default (~1 billion).
        Matches ``open_geotiff`` / ``read_geotiff_dask`` /
        ``read_geotiff_gpu``.
    missing_sources : {'raise', 'warn'}, default 'raise'
        Policy for unreadable source files referenced by the VRT.
        ``'raise'`` (the default since #1860) fails immediately on an
        unreadable backing source so a partial mosaic never surfaces
        silently. This matches the internal ``_vrt.read_vrt`` default
        and the rest of the geotiff module's up-front rejection of
        malformed input. Prior to #1860 the public default was
        ``'warn'``; callers that relied on the lenient behaviour pass
        ``missing_sources='warn'`` explicitly.
        ``'warn'`` is the opt-in escape hatch for partial mosaics: it
        emits ``GeoTIFFFallbackWarning``, records ``attrs['vrt_holes']``,
        and returns the mosaic with holes left as the band's nodata
        sentinel (or zero on integer bands without a sentinel).
        ``XRSPATIAL_GEOTIFF_STRICT=1`` forces a raise across the whole
        module regardless of this kwarg.
    band_nodata : {'first', None}, optional
        Opt-out for the fail-closed mixed-band-metadata check (issue
        #1987 PR 5). ``None`` (the default) rejects a VRT whose bands
        declare disagreeing per-band ``<NoDataValue>`` sentinels with
        ``MixedBandMetadataError``; flattening to one value would
        otherwise let one band's valid pixels collide with another
        band's sentinel after the flatten. Pass ``band_nodata='first'``
        to keep the legacy flatten-to-band-0 semantics explicitly. Any
        other value raises ``ValueError`` at the boundary so typos
        surface up front instead of degrading silently into strict mode.
    mask_nodata : bool, default True
        If True, run the integer-sentinel-to-NaN promotion on the
        assembled mosaic. If False, skip it and keep the source dtype
        with the raw sentinel still in the data. ``attrs['nodata']``
        carries the sentinel either way. Pass ``mask_nodata=False``
        together with ``dtype=<integer>`` when you need to preserve an
        integer source dtype on a VRT whose declared sentinel matches
        actual pixels. See issue #2052. Float source bands are NaN-aware
        by virtue of how the internal reader handles them, so this kwarg
        is most useful for integer-dtype mosaics.

    Returns
    -------
    xr.DataArray
        NumPy, Dask, CuPy, or Dask+CuPy backed depending on options.

    Notes
    -----
    Like ``open_geotiff``, the CRS lands as an int EPSG in
    ``attrs['crs']`` when the VRT's WKT resolves to a known EPSG code.
    Otherwise ``attrs['crs']`` stays unset and ``attrs['crs_wkt']`` carries
    the original WKT. The source GeoTransform is preserved as a
    rasterio-style 6-tuple in ``attrs['transform']``.

    Source-path containment (issue #1671): every ``<SourceFilename>`` in
    the VRT must resolve (after canonicalising ``..`` segments and
    symlinks) to a path under the VRT's own directory.  Absolute paths
    pointing elsewhere are rejected with ``ValueError`` by default.
    Operators that legitimately need to mosaic files from outside the
    VRT directory can opt in by setting the
    ``XRSPATIAL_VRT_ALLOWED_ROOTS`` environment variable to a
    ``os.pathsep``-separated list of trusted directory roots; sources
    resolving under any listed root are then accepted.  A
    ``relativeToVRT='1'`` source that escapes the VRT directory (e.g.
    ``../../etc/passwd`` or a symlink to a file outside the directory)
    is rejected regardless of the allowlist.

    Lazy chunked reads (issue #1814): when ``chunks=`` is set, the
    returned DataArray wraps a dask graph that decodes one chunk
    window per task.  Construction does not materialise any pixels;
    only the VRT XML is parsed.  The eager read populates
    ``attrs['vrt_holes']`` from skipped sources at decode time. The
    chunked path approximates the same contract via a parse-time
    ``os.path.exists`` sweep over every source; that catches the
    dominant missing-file case but does not detect decode-time codec
    failures, which surface as per-task ``GeoTIFFFallbackWarning``
    instead. Each worker still emits ``GeoTIFFFallbackWarning`` for
    missing sources at execution time as well.
    """
    from .._reader import _coerce_path
    from .._vrt import (
        read_vrt as _read_vrt_internal,
        _apply_integer_sentinel_mask_with_presence as _vrt_mask_with_presence,
        _scan_for_sentinel as _vrt_scan_for_sentinel,
    )

    source = _coerce_path(source)

    # Shared dispatcher-kwarg validator so direct callers see the same
    # rejections as ``open_geotiff`` (issue #2175 / parent #2162). For
    # ``read_vrt`` the helper rejects ``on_gpu_failure`` (VRT reads do
    # not go through a GPU decoder pipeline), ``max_cloud_bytes`` (the
    # VRT reader does not consume the cloud-byte budget, issue #1974),
    # and validates ``overview_level``'s type. ``missing_sources`` and
    # ``band_nodata`` are legitimate VRT kwargs so the helper's
    # VRT-only guard is a no-op here. ``gpu=False`` is passed so that
    # an explicit ``on_gpu_failure`` is rejected regardless of the
    # ``read_vrt(gpu=)`` output-device kwarg.
    _validate_dispatch_kwargs(
        source=source,
        gpu=False,
        chunks=chunks,
        overview_level=overview_level,
        on_gpu_failure=on_gpu_failure,
        missing_sources=missing_sources,
        band_nodata=band_nodata,
        max_cloud_bytes=max_cloud_bytes,
    )

    # ``overview_level`` is not consumed by ``read_vrt`` (the VRT XML
    # references its own source files; overview selection would need to
    # apply to each one). ``overview_level=0`` matches the documented
    # "full resolution" default, so treat it as a no-op. Mirrors the
    # existing rejection inside ``open_geotiff``'s VRT branch at
    # ``__init__.py:551-555`` (issue #1685). Keep both sites in sync; a
    # future refactor that moves this into ``_validate_dispatch_kwargs``
    # should drop both at once.
    if overview_level not in (None, 0):
        raise ValueError(
            "overview_level is not supported for VRT sources. "
            "VRT references its own source files; pass overview_level "
            "to open_geotiff on a .tif source, or drop the kwarg.")

    # Reject non-positive chunk sizes up front so the VRT dask path
    # surfaces the same error as ``read_geotiff_dask`` (#1776). Without
    # this check ``chunks=0`` raised ``ZeroDivisionError`` deep in dask
    # and ``chunks=-1`` was silently accepted. ``chunks=None`` is the
    # default (eager read), so allow it through here.
    chunks = _validate_chunks_arg(chunks, allow_none=True)

    if missing_sources not in ('warn', 'raise'):
        raise ValueError(
            f"missing_sources must be 'warn' or 'raise', got "
            f"{missing_sources!r}")

    # ``band_nodata`` accepts only ``None`` (strict, the default) and
    # ``'first'`` (legacy flatten-to-band-0 opt-out for the fail-closed
    # mixed-band-metadata check, issue #1987 PR 5). Any other value would
    # silently degrade to strict mode because the registered check treats
    # anything other than ``'first'`` as "no opt-out", which means a typo
    # like ``band_nodata='firs'`` looks like opt-out at the call site but
    # raises ``MixedBandMetadataError`` at run time. Mirror the
    # ``missing_sources`` value-validation pattern above so the typo
    # surfaces at the boundary instead.
    if band_nodata not in (None, 'first'):
        raise ValueError(
            f"band_nodata must be None or 'first', got {band_nodata!r}. "
            f"Pass ``band_nodata='first'`` to opt back into the legacy "
            f"flatten-to-first-band semantics for VRT sources with "
            f"disagreeing per-band nodata sentinels, or drop the kwarg "
            f"to keep the fail-closed default. See issue #1987.")

    # Lazy chunked path (issue #1814). The eager call below materialises
    # the full mosaic on host RAM and then wraps the array via
    # ``.chunk()``, so chunks= gave no memory protection and gpu=True +
    # chunks= still assembled the full mosaic on the CPU before moving to
    # the device. When chunks= is set, dispatch to a delayed-per-window
    # builder so each task decodes only the sources intersecting its
    # destination window.
    if chunks is not None:
        return _read_vrt_chunked(
            source,
            window=window,
            band=band,
            name=name,
            chunks=chunks,
            gpu=gpu,
            dtype=dtype,
            max_pixels=max_pixels,
            missing_sources=missing_sources,
            allow_rotated=allow_rotated,
            allow_unparseable_crs=allow_unparseable_crs,
            band_nodata=band_nodata,
            mask_nodata=mask_nodata,
        )

    # Issue #1987 ambiguous-metadata checks for the eager VRT path. Parse
    # the VRT XML up front and validate before ``_read_vrt_internal``
    # touches any pixel data, so a rejected file does not first
    # materialise the full mosaic into host memory. The parsed
    # ``VRTDataset`` is threaded into the internal reader via ``parsed=``
    # so we don't double-parse the XML.
    #
    # ``_finalize_lazy_read_attrs`` re-runs the same check set after the
    # read with the same ``band_nodata`` / ``band_nodata_values`` context
    # threaded through (issue #2210). Keeping the inline pre-read call
    # preserves the pre-materialise rejection guard for big mosaics; the
    # helper-routed post-read call acts as a defensive consistency check
    # rather than the no-op it was before #2210.
    import os as _os
    from .._validation import validate_read_metadata
    from .._vrt import parse_vrt as _parse_vrt, _read_vrt_xml
    _xml_str = _read_vrt_xml(source)
    _vrt_dir = _os.path.dirname(_os.path.abspath(source))
    _parsed_vrt = _parse_vrt(_xml_str, _vrt_dir)
    _band_nodata_values = (
        [b.nodata for b in _parsed_vrt.bands]
        if _parsed_vrt.bands else None
    )
    validate_read_metadata({
        'allow_rotated': allow_rotated,
        'allow_unparseable_crs': allow_unparseable_crs,
        'transform': _gdal_geotransform_to_affine_tuple(
            _parsed_vrt.geo_transform
        ),
        'crs_wkt': _parsed_vrt.crs_wkt,
        'band_nodata': band_nodata,
        'band_nodata_values': _band_nodata_values,
    })

    # Thread ``mask_nodata`` into the internal reader so the float
    # source NaN masking inside ``_read_data`` honors the opt-out
    # alongside the integer ``_vrt_mask_with_presence`` helper called
    # further down. Without this the eager path silently rewrote
    # literal float sentinels (e.g. ``-9999.0``) to NaN regardless of
    # the kwarg, which violates the documented opt-out and can drop
    # legitimate sentinel pixels from downstream consumers. The
    # ``nodata_pixels_present`` proxy below still reports correctly
    # because it falls back to a float-buffer scan when the helper
    # short-circuits. See issue #2158.
    arr, vrt = _read_vrt_internal(
        source, window=window, band=band, max_pixels=max_pixels,
        missing_sources=missing_sources, parsed=_parsed_vrt,
        mask_nodata=mask_nodata,
    )

    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    # Build coordinates from GeoTransform.
    #
    # GDAL's convention: when AREA_OR_POINT=Area (default) the
    # GeoTransform origin is the top-left corner of pixel (0, 0) and
    # pixel centers need a half-pixel shift.  When AREA_OR_POINT=Point
    # the origin already *is* the center of pixel (0, 0) and no shift
    # is applied.  This mirrors ``_geo_to_coords`` for non-VRT reads.
    gt = vrt.geo_transform
    # A rotated VRT under ``allow_rotated=True`` is treated as no-georef
    # by the GeoTIFF contract (#2115): the in-memory array is a pixel
    # grid, not a projected raster, so ``attrs['crs']`` would mislead
    # downstream code that branches on ``'crs' in attrs`` (#2122). The
    # rotated case is the non-zero ``b`` / ``d`` term on the GDAL
    # GeoTransform (positions 2 and 4). A VRT with no ``<GeoTransform>``
    # at all is the general no-georef case and the CRS is still
    # surfaced (the writer-side ``crs=`` kwarg path emits CRS without a
    # transform; see ``test_crs_kwarg_no_coords``).
    _vrt_is_rotated = (
        gt is not None and (gt[2] != 0.0 or gt[4] != 0.0)
    )
    if gt is not None:
        origin_x, res_x, _, origin_y, _, res_y = gt
        height, width = arr.shape[:2]
        if window is not None:
            r0 = max(0, window[0])
            c0 = max(0, window[1])
            coord_window = (r0, c0, r0 + height, c0 + width)
        else:
            coord_window = None
        # Rotated VRTs emit int64 pixel coords to match the eager
        # non-VRT rotated path (#2122 follow-up). Without this gate
        # the VRT branch handed back float projected coords while
        # the rest of the read pretended the array had no georef,
        # so a downstream consumer saw float64 x/y dtypes on a
        # no-georef array, the inverse of the contract documented in
        # ``docs/source/user_guide/attrs_contract.rst``.
        coords = _coords_from_pixel_geometry(
            origin_x, origin_y, res_x, res_y, height, width,
            is_point=vrt.raster_type == 'point',
            window=coord_window,
            has_georef=not _vrt_is_rotated,
        )
    else:
        coords = {}

    # Select the per-band nodata sentinel. When a specific band is
    # selected, source its nodata from that band's ``<NoDataValue>``
    # instead of band 0's. Otherwise multi-band VRTs with per-band
    # sentinels would mis-mask the read: attrs would advertise band 0's
    # sentinel, the integer-promotion block below would mask against
    # band 0's sentinel, and band N's actual nodata pixels would
    # survive as literal integers. See issue #1598. ``band`` has
    # already been validated by ``_vrt.read_vrt`` as
    # 0 <= band < len(vrt.bands), so a simple lookup is safe here.
    #
    # Documented divergence: per-band sentinel selection cannot ride
    # through ``_finalize_lazy_read_attrs`` because the helper expects
    # a single ``nodata`` value from ``geo_info.nodata``. The shared
    # GeoInfo dataclass does not model per-band sentinels.
    nodata = None
    if vrt.bands:
        band_idx_for_nodata = band if band is not None else 0
        nodata = vrt.bands[band_idx_for_nodata].nodata

    # Apply the VRT-specific masking step. The shared helper's
    # ``_apply_eager_nodata_mask`` is single-sentinel, but the VRT
    # multi-band case masks each band against its own ``<NoDataValue>``,
    # so we run the VRT-aware helper here and feed the post-mask buffer
    # to ``_finalize_lazy_read_attrs``. This is documented divergence
    # tied to the per-band selection above.
    nodata_pixels_present: bool | None = None
    if mask_nodata:
        arr, nodata_pixels_present = _vrt_mask_with_presence(arr, vrt, band)
    elif nodata is not None:
        # ``mask_nodata=False``: skip the masking helper but still scan
        # for the literal sentinel so callers can branch on the attr.
        nodata_pixels_present = _vrt_scan_for_sentinel(arr, vrt, band)

    # Capture pre-cast dtype: ``_vrt._read_data`` NaN-masks float source
    # arrays (and int sources feeding a float VRT dataType) inline, and
    # ``_vrt_mask_with_presence`` above promotes int-with-sentinel to
    # float64 with NaNs written in place. Any of those paths yield a
    # float pre-cast dtype; an int pre-cast dtype means literal sentinels
    # are still in the buffer. The optional user dtype cast below may
    # promote int -> float without masking, so reading dtype after the
    # cast would falsely claim ``masked_nodata=True`` (issue #2092
    # follow-up).
    pre_cast_dtype = np.dtype(str(arr.dtype))

    # Float-NaN proxy for ``nodata_pixels_present``. The signal for a
    # float buffer can come from any of three places, only one of which
    # actually fires per call:
    #
    # * ``mask_nodata=True`` + float source + declared sentinel: the
    #   inline NaN masking in ``_vrt._read_data`` rewrote sentinels to
    #   NaN, so ``np.isnan(arr).any()`` is the presence answer.
    # * ``mask_nodata=False`` + declared sentinel: ``_vrt_scan_for_sentinel``
    #   above already set ``nodata_pixels_present`` from a literal-value
    #   scan, so this block short-circuits via ``not present``.
    # * ``mask_nodata=True`` + float source with no inline masking (e.g.
    #   source's natural NaN values, sentinel absent from window): the
    #   NaN proxy catches the in-buffer NaNs.
    #
    # Invariant: at most one branch sets ``nodata_pixels_present`` to True
    # before this block runs. ``_vrt_mask_with_presence`` only sets True
    # on integer buffers (it short-circuits on float dtype), and
    # ``_vrt_scan_for_sentinel`` is gated on the ``mask_nodata=False``
    # branch above. Either way, the float-NaN proxy below is the only
    # remaining presence signal for float buffers, so the ``not present``
    # guard is sufficient -- we will not double-scan the same buffer.
    # The ``mask_nodata=False`` arm fix from #2158 means the inline NaN
    # masking no longer runs under the opt-out; the ``_vrt_scan_for_sentinel``
    # short-circuit above is what keeps the presence attr honest there.
    if (nodata is not None
            and pre_cast_dtype.kind == 'f'
            and not nodata_pixels_present):
        # ``arr`` is still on the CPU at this point (the optional
        # ``cupy.asarray`` lift happens further down).
        try:
            nodata_pixels_present = bool(np.isnan(arr).any())
        except (TypeError, ValueError):
            # Defensive: an unexpected dtype that fails ``np.isnan`` falls
            # back to the helper's answer rather than raising mid-read.
            pass

    # Transfer to GPU if requested. The dtype cast lands after the lift
    # so ``cupy.ndarray.astype`` runs on the device.
    if gpu:
        import cupy
        arr = cupy.asarray(arr)

    if dtype is not None:
        target = np.dtype(dtype)
        _validate_dtype_cast(np.dtype(str(arr.dtype)), target)
        arr = arr.astype(target)

    # Wave 3 of #2162: route attrs assembly through
    # ``_finalize_lazy_read_attrs`` so the VRT eager path shares the
    # validate-then-populate-then-stamp block with the eager numpy,
    # eager GPU, and dask backends. ``geo_info`` is a synthesised
    # ``GeoInfo`` carrying only the fields the VRT path owns; the
    # helper handles ``georef_status`` / ``rotated_affine`` / no-georef
    # marker emission via the same ``geo_info_to_metadata`` ->
    # ``metadata_to_attrs`` pipeline.
    #
    # ``graph_dtype`` is the **pre-cast** dtype so ``masked_nodata``
    # reflects whether VRT-side masking actually ran (an int -> float
    # caller cast must not flip the attr; see #2092 follow-up).
    # ``caller_dtype`` is the user's ``dtype=`` kwarg so
    # ``nodata_dtype_cast`` records caller intent rather than the
    # masking-induced graph dtype.
    #
    # Documented divergence from the eager helper:
    #
    # * ``vrt_holes`` is VRT-specific and ``GeoInfo`` has no slot for it.
    #   The seed dict carries the value through ``attrs_in`` so the
    #   helper's ``attrs.update`` leaves it intact; the helper's own
    #   metadata copy does not emit ``vrt_holes`` for a synthetic
    #   GeoInfo.
    # * Per-band nodata selection happens above this call, not inside
    #   the helper. ``_apply_eager_nodata_mask`` is single-sentinel.
    # * ``nodata_pixels_present`` is computed above (VRT-aware scan)
    #   and stamped post-helper because ``_finalize_lazy_read_attrs``
    #   passes ``pixels_present=None`` unconditionally.
    synth_geo_info = _vrt_to_synthetic_geo_info(vrt)
    attrs_seed: dict = {}
    if vrt.holes:
        attrs_seed['vrt_holes'] = list(vrt.holes)
    attrs = _finalize_lazy_read_attrs(
        geo_info=synth_geo_info,
        nodata=nodata,
        mask_nodata=mask_nodata,
        graph_dtype=pre_cast_dtype,
        caller_dtype=dtype,
        window=window,
        allow_rotated=allow_rotated,
        allow_unparseable_crs=allow_unparseable_crs,
        band_nodata=band_nodata,
        band_nodata_values=_band_nodata_values,
        attrs_in=attrs_seed,
    )
    if nodata is not None and nodata_pixels_present is not None:
        attrs['nodata_pixels_present'] = bool(nodata_pixels_present)

    if arr.ndim == 3:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(arr.shape[2])
    else:
        dims = ['y', 'x']

    result = xr.DataArray(arr, dims=dims, coords=coords, name=name, attrs=attrs)

    # ``chunks is not None`` is handled by ``_read_vrt_chunked`` higher up
    # in this function (issue #1814); reaching this point implies the
    # eager path, so no post-decode chunking is needed.
    return result


def _vrt_chunk_read(source, r0, c0, r1, c1, *,
                    band, max_pixels, missing_sources,
                    declared_dtype, gpu, parsed_vrt,
                    mask_nodata: bool = True):
    """Decode a single chunk window from a VRT.

    Called by ``dask.delayed`` from :func:`_read_vrt_chunked`. The
    function reads only the destination window via the existing VRT
    internal reader, applies the same integer-sentinel masking the
    eager :func:`read_vrt` does post-decode, casts to the dtype the
    dask graph declared up front, and optionally moves the block to
    the GPU.

    ``parsed_vrt`` is the parent dispatcher's already-parsed
    :class:`VRTDataset`; the internal reader skips the XML parse and
    source-path containment check when this is supplied, removing the
    N+1 parse cost an earlier implementation had (issue #1825).

    Returning a ``numpy.ndarray`` (or ``cupy.ndarray`` when ``gpu`` is
    set) whose shape and dtype match the ``shape=`` / ``dtype=`` kwargs
    of the surrounding ``dask.array.from_delayed`` is the contract; a
    mismatch would silently produce a wrong-shape dask array.
    """
    from .._vrt import (
        read_vrt as _read_vrt_internal,
        _apply_integer_sentinel_mask,
    )

    # Forward ``mask_nodata`` to the internal reader so the float
    # source NaN masking inside ``_read_data`` honors the opt-out too,
    # not just the integer post-decode helper below. Without this the
    # chunked path would silently rewrite literal float sentinels to
    # NaN even when the caller asked to keep them. See issue #2158.
    arr, vrt = _read_vrt_internal(
        source, window=(r0, c0, r1, c1), band=band,
        max_pixels=max_pixels, missing_sources=missing_sources,
        parsed=parsed_vrt, mask_nodata=mask_nodata,
    )

    # Mirror the eager post-decode integer-sentinel masking via the
    # shared helper. The internal reader NaN-masks float source arrays
    # inline (gated on the same ``mask_nodata`` kwarg via #2158) but
    # leaves integer sentinels untouched, so the eager path promotes
    # to float64 when sentinels hit. The surrounding dask graph
    # already declared float64 when any band has a representable integer
    # sentinel, so any chunk that actually fires the mask returns a
    # buffer whose dtype matches the declared one. Skip the helper when
    # ``mask_nodata=False`` so the source integer dtype survives (#2052).
    if mask_nodata:
        arr = _apply_integer_sentinel_mask(arr, vrt, band)

    if declared_dtype is not None and arr.dtype != declared_dtype:
        arr = arr.astype(declared_dtype)

    if gpu:
        import cupy
        arr = cupy.asarray(arr)

    return arr


def _read_vrt_chunked(source, *, window, band, name, chunks, gpu, dtype,
                      max_pixels, missing_sources,
                      allow_rotated: bool = False,
                      allow_unparseable_crs: bool = False,
                      band_nodata: str | None = None,
                      mask_nodata: bool = True):
    """Lazy ``read_vrt`` dispatch when ``chunks=`` is set (issue #1814).

    Parses the VRT XML once to recover the extent, CRS, GeoTransform,
    and per-band metadata, then builds a dask graph with one task per
    chunk window. Each task calls into the existing VRT internal reader
    with its own ``window=`` so only the sources intersecting the
    chunk's destination rectangle are decoded.

    ``attrs['vrt_holes']`` is populated from a parse-time
    ``os.path.exists`` sweep over every source referenced by the parsed
    VRT; this preserves the eager-path contract documented in #1734 so
    callers switching from eager to chunked can still detect partial
    mosaics by attribute lookup (rather than monitoring the
    ``GeoTIFFFallbackWarning`` stream). The check is a static
    approximation of the eager path's per-source decode-time exception
    handling: it catches the dominant "missing file" case but does not
    detect decode-time codec failures, which surface as per-task
    ``GeoTIFFFallbackWarning`` from each worker.
    """
    import os as _os
    import dask
    import dask.array as da

    from .._reader import MAX_PIXELS_DEFAULT
    from .._vrt import (
        parse_vrt,
        _read_vrt_xml,
        _effective_dtype_for_bands,
        _sentinel_for_dtype,
    )

    # Parse the VRT XML up-front (cheap; no pixel decode). Route through
    # ``_read_vrt_xml`` so the 64 MiB ``XRSPATIAL_VRT_MAX_XML_BYTES`` cap
    # added in #1818 applies to the chunked dispatcher too; a raw
    # ``open().read()`` here would let a multi-GB attacker-supplied VRT
    # exhaust memory before any parser-side guard fires (issue #1831).
    # The parsed VRTDataset is plumbed into every per-chunk task so each
    # task can skip the redundant XML parse and source-path allowlist
    # validation the internal reader otherwise performs (issue #1825).
    xml_str = _read_vrt_xml(source)
    vrt_dir = _os.path.dirname(_os.path.abspath(source))
    vrt = parse_vrt(xml_str, vrt_dir)

    # Issue #1987 ambiguous-metadata checks on the chunked VRT path. Run
    # before the band-count validator below so a rejected file does not
    # produce side effects. ``_finalize_lazy_read_attrs`` re-runs the
    # same check set at graph-build time with the same
    # ``band_nodata`` / ``band_nodata_values`` context threaded through
    # (issue #2210); this inline call keeps the rejection in front of
    # the ``band`` / window / ``max_pixels`` validators below so the
    # error ordering matches the eager path.
    from .._validation import validate_read_metadata
    band_nodata_values = (
        [b.nodata for b in vrt.bands] if vrt.bands else None
    )
    validate_read_metadata({
        'allow_rotated': allow_rotated,
        'allow_unparseable_crs': allow_unparseable_crs,
        'transform': _gdal_geotransform_to_affine_tuple(vrt.geo_transform),
        'crs_wkt': vrt.crs_wkt,
        'band_nodata': band_nodata,
        'band_nodata_values': band_nodata_values,
    })

    # Validate ``band`` against the parsed band count, matching the
    # internal reader's contract so the failure mode is the same whether
    # the user reads eagerly or chunked.
    if band is not None:
        if not isinstance(band, (int, np.integer)) or isinstance(band, bool):
            raise ValueError(
                f"band must be a non-negative int, got {band!r}")
        if band < 0 or band >= len(vrt.bands):
            raise ValueError(
                f"band index {band} out of range for VRT with "
                f"{len(vrt.bands)} band(s)")

    # Resolve the windowed extent against the VRT.
    if window is not None:
        r0, c0, r1, c1 = window
        if (r0 < 0 or c0 < 0
                or r1 > vrt.height or c1 > vrt.width
                or r0 >= r1 or c0 >= c1):
            raise ValueError(
                f"window={window} is outside the VRT extent "
                f"({vrt.height}x{vrt.width}) or has non-positive size.")
        win_r0, win_c0 = r0, c0
        full_h, full_w = r1 - r0, c1 - c0
    else:
        win_r0, win_c0 = 0, 0
        full_h, full_w = vrt.height, vrt.width

    max_pixels_effective = (
        max_pixels if max_pixels is not None else MAX_PIXELS_DEFAULT
    )

    # Up-front pixel-count guard against the windowed extent. Mirrors
    # the eager ``_vrt.read_vrt`` (which calls ``_check_dimensions`` on
    # the full output shape) and ``read_geotiff_dask`` (which guards
    # ``full_h * full_w * eff_bands`` before scheduling any task). Each
    # chunk task additionally re-checks via ``max_pixels`` through the
    # internal reader, but catching an oversized request up front saves
    # the caller from a misleading per-chunk error.
    eff_bands = 1 if band is not None else max(1, len(vrt.bands))
    if full_h * full_w * eff_bands > max_pixels_effective:
        raise ValueError(
            f"Requested region {full_h}x{full_w}x{eff_bands} exceeds "
            f"max_pixels={max_pixels_effective:,}.")

    if isinstance(chunks, int):
        ch_h = ch_w = chunks
    else:
        ch_h, ch_w = chunks

    # Refuse chunk grids that would build more tasks than the scheduler
    # can hold without OOMing the driver. ``read_geotiff_dask`` uses the
    # same cap with the same suggestion logic (see issue #1814 and the
    # ``_MAX_DASK_CHUNKS`` guard upstream).
    n_chunks = ((full_h + ch_h - 1) // ch_h) * ((full_w + ch_w - 1) // ch_w)
    if n_chunks > _MAX_VRT_DASK_CHUNKS:
        scale = math.sqrt(n_chunks / _MAX_VRT_DASK_CHUNKS)
        suggested_h = int(math.ceil(ch_h * scale))
        suggested_w = int(math.ceil(ch_w * scale))
        raise ValueError(
            f"read_vrt: chunks=({ch_h}, {ch_w}) on a "
            f"{full_h}x{full_w} VRT region would produce {n_chunks:,} "
            f"dask tasks, exceeding the {_MAX_VRT_DASK_CHUNKS:,}-task "
            f"cap. Pass a larger chunks=... value explicitly (e.g. "
            f"chunks=({suggested_h}, {suggested_w}) keeps the task "
            f"count under the cap)."
        )

    # Select bands for shape/dtype declaration.
    if band is not None:
        selected_bands = [vrt.bands[band]]
    else:
        selected_bands = vrt.bands

    if not selected_bands:
        raise ValueError(
            "VRT has no <VRTRasterBand> elements; cannot determine "
            "output dtype")

    # Compute the declared dtype. Share the per-band effective-dtype
    # rule (ComplexSource scale/offset promotes to float64) with the
    # eager path via ``_effective_dtype_for_bands`` so both paths agree
    # on the result_type (issue #1825). Then widen to float64 if any
    # selected band declares an integer nodata sentinel that round-trips
    # through the band's dtype.
    #
    # The eager path defers the promotion to runtime: it scans every
    # band's pixels and promotes only if at least one sentinel hits.
    # The chunked path cannot afford that scan up front (it would
    # require decoding the mosaic the dask graph was constructed to
    # defer), so this is a parse-time approximation. The trade-off:
    #   * if a band declares nodata and no chunk contains the
    #     sentinel, the chunked output is float64 where the eager
    #     output would have stayed integer (acceptable: the user
    #     asked the source for nodata, so they expect NaN masking);
    #   * if a band does not declare nodata, both paths keep the
    #     source integer dtype (handled by the ``promotes is False``
    #     fall-through below).
    # See also Copilot review on PR #1822.
    declared_dtype = _effective_dtype_for_bands(selected_bands)

    if mask_nodata and declared_dtype.kind in ('u', 'i'):
        promotes = False
        for vrt_band in selected_bands:
            if _sentinel_for_dtype(vrt_band.nodata,
                                   declared_dtype) is not None:
                promotes = True
                break
        if promotes:
            declared_dtype = np.dtype(np.float64)

    out_has_band_axis = band is None and len(vrt.bands) > 1
    n_out_bands = len(selected_bands)

    # Build the dask graph: one ``from_delayed`` per chunk window. The
    # destination coordinate space is the VRT's full extent (or the
    # windowed extent), so chunk windows are computed relative to that
    # space and translated to absolute VRT coords before being passed
    # into the per-chunk reader.
    rows = list(range(0, full_h, ch_h))
    cols = list(range(0, full_w, ch_w))

    delayed_read = dask.delayed(_vrt_chunk_read)

    # Wrap the parsed VRTDataset in a single dask Delayed so every
    # chunk task takes it as one graph input rather than a per-task
    # closure copy. Without this, dask embeds the full source list
    # (filenames, src/dst rects, per-source nodata) into every task's
    # pickled graph entry; a 1000-source VRT split into 1000 chunks
    # would build a ~57 MB driver graph and re-serialise the same
    # metadata 1000 times under distributed/process schedulers. The
    # sibling COG-HTTP and GDS chunked paths use the same single-
    # delayed-input pattern (see ``http_meta_key`` in ``dask.py`` and
    # ``meta_key`` in ``gpu.py``). See issue #1923.
    parsed_vrt_key = dask.delayed(vrt, pure=True)

    if gpu:
        import cupy
        meta = cupy.empty((0,) * (3 if out_has_band_axis else 2),
                          dtype=declared_dtype)
    else:
        meta = np.empty((0,) * (3 if out_has_band_axis else 2),
                        dtype=declared_dtype)

    dask_rows = []
    for r0c in rows:
        r1c = min(r0c + ch_h, full_h)
        dask_cols = []
        for c0c in cols:
            c1c = min(c0c + ch_w, full_w)
            if out_has_band_axis:
                block_shape = (r1c - r0c, c1c - c0c, n_out_bands)
            else:
                block_shape = (r1c - r0c, c1c - c0c)
            d = delayed_read(
                source,
                r0c + win_r0, c0c + win_c0,
                r1c + win_r0, c1c + win_c0,
                band=band,
                max_pixels=max_pixels_effective,
                missing_sources=missing_sources,
                declared_dtype=declared_dtype,
                gpu=gpu,
                parsed_vrt=parsed_vrt_key,
                mask_nodata=mask_nodata,
            )
            block = da.from_delayed(d, shape=block_shape,
                                    dtype=declared_dtype, meta=meta)
            dask_cols.append(block)
        dask_rows.append(da.concatenate(dask_cols, axis=1))

    dask_arr = da.concatenate(dask_rows, axis=0)

    # Optional user-requested dtype cast happens lazily on the dask
    # array so the per-chunk decode dtype stays predictable.
    if dtype is not None:
        target = np.dtype(dtype)
        _validate_dtype_cast(declared_dtype, target)
        dask_arr = dask_arr.astype(target)
        final_dtype = target
    else:
        final_dtype = declared_dtype

    # Coordinates: derive from the VRT GeoTransform and the windowed
    # extent. Mirrors the eager branch in ``read_vrt`` so chunked and
    # eager reads share the same x/y arrays.
    gt = vrt.geo_transform
    coords = {}
    _vrt_is_rotated = (
        gt is not None and (gt[2] != 0.0 or gt[4] != 0.0)
    )
    if gt is not None:
        origin_x, res_x, _, origin_y, _, res_y = gt
        coord_window = (win_r0, win_c0, win_r0 + full_h, win_c0 + full_w)
        # Rotated VRTs emit int64 pixel coords to match the eager
        # non-VRT rotated path (#2122 follow-up). Without this gate the
        # chunked VRT branch handed back float projected coords on a
        # no-georef array, the inverse of the contract documented in
        # ``docs/source/user_guide/attrs_contract.rst``.
        coords = _coords_from_pixel_geometry(
            origin_x, origin_y, res_x, res_y, full_h, full_w,
            is_point=vrt.raster_type == 'point',
            window=coord_window,
            has_georef=not _vrt_is_rotated,
        )

    # Surface the nodata sentinel for the selected band. The chunked
    # path declares ``float64`` up front whenever any selected band has
    # a representable integer sentinel (see the ``declared_dtype`` block
    # earlier in this function), so the dask graph dtype drives
    # ``masked_nodata`` (issue #1988). ``final_dtype`` is the post-cast
    # dtype the dask array was reshaped to above, which is what the
    # caller will see on the returned DataArray.
    #
    # Documented divergence (see ``_finalize_lazy_read_attrs`` call): the
    # synthesised ``GeoInfo`` does not model per-band nodata, so the
    # caller picks the right band's sentinel here and forwards it via
    # the helper's ``nodata`` kwarg.
    nodata_meta = None
    if vrt.bands:
        band_idx_for_nodata = band if band is not None else 0
        nodata_meta = vrt.bands[band_idx_for_nodata].nodata

    # Static hole detection: mirror the eager-path ``attrs['vrt_holes']``
    # contract (#1734) by scanning every source referenced in the parsed
    # VRT and recording the ones whose backing file does not exist on
    # disk. The eager path discovers holes at decode time (per-source
    # OSError / codec error) and aggregates them onto ``vrt.holes``;
    # under chunked dispatch each per-task decode catches its own
    # missing source and warns, but those records cannot be reduced
    # back onto the parent DataArray without an extra synchronisation
    # pass. The parse-time existence sweep catches the dominant
    # missing-file case before scheduling and lets callers branch on
    # ``"vrt_holes" in da.attrs`` exactly as with the eager reader.
    # Empty list is omitted so the attr only appears when a hole is
    # actually present. Each entry mirrors the eager schema:
    # ``{'source', 'band', 'dst_rect', 'error'}``.
    chunked_holes: list[dict] = []
    for vrt_band in vrt.bands:
        for src in vrt_band.sources:
            if not _os.path.exists(src.filename):
                chunked_holes.append({
                    'source': src.filename,
                    'band': vrt_band.band_num,
                    'dst_rect': (src.dst_rect.x_off, src.dst_rect.y_off,
                                 src.dst_rect.x_size, src.dst_rect.y_size),
                    'error': 'FileNotFoundError: source file not found',
                })

    # Wave 3 of #2162: route attrs assembly through
    # ``_finalize_lazy_read_attrs`` so the VRT chunked path shares the
    # validate-then-populate-then-stamp block with the eager VRT path
    # and the dask backends. ``geo_info`` is the same synthesised
    # ``GeoInfo`` the eager path uses (rotated transform / no-georef /
    # axis-aligned arms). ``window`` carries the parent windowed extent
    # so the helper's ``geo_info_to_metadata`` emits the windowed
    # transform.
    #
    # ``graph_dtype`` is ``declared_dtype`` (the pre-cast graph dtype)
    # so ``masked_nodata`` reflects whether the per-chunk reader
    # actually masked rather than the post-cast ``final_dtype``.
    # ``caller_dtype`` is the user's ``dtype=`` so ``nodata_dtype_cast``
    # records the user's intent rather than the masking-induced
    # auto-promotion (#2092 follow-up). ``vrt_holes`` rides through
    # ``attrs_in`` because ``GeoInfo`` has no slot for it (documented
    # divergence from the helper contract).
    synth_geo_info = _vrt_to_synthetic_geo_info(vrt)
    helper_window = (win_r0, win_c0, win_r0 + full_h, win_c0 + full_w)
    attrs_seed: dict = {}
    if chunked_holes:
        attrs_seed['vrt_holes'] = chunked_holes
    attrs = _finalize_lazy_read_attrs(
        geo_info=synth_geo_info,
        nodata=nodata_meta,
        mask_nodata=mask_nodata,
        graph_dtype=declared_dtype,
        caller_dtype=dtype,
        window=helper_window,
        allow_rotated=allow_rotated,
        allow_unparseable_crs=allow_unparseable_crs,
        band_nodata=band_nodata,
        band_nodata_values=band_nodata_values,
        attrs_in=attrs_seed,
    )

    if out_has_band_axis:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(n_out_bands)
    else:
        dims = ['y', 'x']

    if name is None:
        name = _os.path.splitext(_os.path.basename(source))[0]

    result = xr.DataArray(
        dask_arr, dims=dims, coords=coords, name=name, attrs=attrs,
    )
    # Sanity: the declared dtype on the dask array is what we return.
    assert result.dtype == final_dtype, (
        f"internal: result dtype {result.dtype} != declared {final_dtype}"
    )
    return result
