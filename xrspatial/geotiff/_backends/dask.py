"""Dask read backend: ``read_geotiff_dask`` and ``_delayed_read_window``.

Step 8 of issue #1813. With validators (#1882), attrs helpers (#1883),
and runtime sentinels (#1880) already extracted, the dask entry-point
body and its delayed-read helper move cleanly into this module.

``read_vrt`` is statically imported from the sibling ``.vrt`` module
since #1898 promoted it from a lazy import once the target moved out
of ``__init__.py``. ``_read_geo_info`` still lives in ``__init__.py``
and is lazy-imported inside ``read_geotiff_dask``'s body to avoid a
circular import (``__init__.py`` re-exports ``read_geotiff_dask`` from
here); a later step of #1813 will move it out and the lazy import can
become static.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .._attrs import _finalize_lazy_read_attrs
from .._coords import coords_from_geo_info as _coords_from_geo_info
from .._coords import geo_to_coords as _geo_to_coords
from .._nodata import NodataLifecycle
from .._reader import _MAX_CLOUD_BYTES_SENTINEL
from .._reader import read_to_array as _read_to_array
from .._runtime import _MISSING_SOURCES_SENTINEL, _ON_GPU_FAILURE_SENTINEL
from .._validation import _validate_chunks_arg, _validate_dispatch_kwargs, _validate_dtype_cast
from .vrt import read_vrt


def read_geotiff_dask(source: str, *,
                      dtype: str | np.dtype | None = None,
                      window: tuple | None = None,
                      overview_level: int | None = None,
                      band: int | None = None,
                      name: str | None = None,
                      chunks: int | tuple = 512,
                      max_pixels: int | None = None,
                      max_cloud_bytes: int | None = (
                          _MAX_CLOUD_BYTES_SENTINEL),  # type: ignore[assignment]
                      on_gpu_failure: str = _ON_GPU_FAILURE_SENTINEL,
                      missing_sources: str = _MISSING_SOURCES_SENTINEL,
                      allow_rotated: bool = False,
                      allow_unparseable_crs: bool = False,
                      band_nodata: str | None = None,
                      mask_nodata: bool = True) -> xr.DataArray:
    """Read a GeoTIFF as a dask-backed DataArray for out-of-core processing.

    Tier: Stable for local-file reads on axis-aligned grids with the
    Tier 1 codec set. ``allow_rotated`` / ``allow_unparseable_crs``
    are Advanced (read-only opt-ins; round-trip semantics are listed
    on the parameter docs). See
    :data:`xrspatial.geotiff.SUPPORTED_FEATURES` for the full tier map
    (issue #2137).

    Each chunk is loaded lazily via windowed reads.

    Parameters
    ----------
    source : str
        File path.
    dtype : str, numpy.dtype, or None
        Cast each chunk to this dtype after reading. None keeps the
        file's native dtype. Float-to-int casts raise ValueError.
    chunks : int or (row_chunk, col_chunk) tuple
        Chunk size in pixels. Default 512.
    overview_level : int or None
        Overview level (0 = full resolution).
    window : tuple or None
        ``(row_start, col_start, row_stop, col_stop)`` to restrict
        chunking to a sub-region of the file. Chunks are laid out
        relative to the window origin. None reads the full raster.
    band : int or None
        Zero-based band index. None returns all bands (3D for
        multi-band files, 2D for single-band). Selecting a single band
        produces a 2D DataArray.
    max_pixels : int or None
        Maximum allowed pixel count (width * height * samples) for the
        windowed region. None uses the reader default (~1 billion).
        The cap is checked once up-front against the lazy region; each
        chunk task also re-checks against ``max_pixels`` so windowed
        reads stay bounded even when ``read_to_array`` is invoked
        directly.
    name : str or None
        Name for the DataArray.
    band_nodata : {'first', None}, optional
        VRT-only opt-out for the fail-closed mixed-band-metadata check
        (issue #1987 PR 5). Forwarded verbatim to ``read_vrt`` when the
        source is a ``.vrt`` file. Passing it with a non-VRT GeoTIFF
        source raises ``ValueError``.
    mask_nodata : bool, default True
        If True, replace the nodata sentinel with NaN per chunk (integer
        rasters get promoted to ``float64``). If False, skip the
        sentinel-to-NaN step so the source dtype survives. The raw
        sentinel is still carried on ``attrs['nodata']`` either way.
        Pass ``mask_nodata=False`` together with ``dtype=<integer>`` to
        keep an integer source dtype; the default promotes to
        ``float64`` and the cast then raises. See issue #2052.
    allow_rotated : bool, default False
        Read-side opt-in for rotated / sheared ``ModelTransformationTag``
        files. Forwarded to every per-chunk read so a rotated source
        yields an ungeoreferenced pixel grid instead of raising
        ``NotImplementedError``. See ``open_geotiff`` for the full
        contract; the dask path honours the same attrs (``crs`` /
        ``crs_wkt`` dropped, ``rotated_affine`` set).
    allow_unparseable_crs : bool, default False
        Read-side opt-in for CRS strings that pyproj cannot resolve and
        do not parse as WKT. When ``False`` (the default since #1929)
        the chunk task raises ``UnparseableCRSError`` instead of
        carrying the unrecognised payload through ``attrs['crs_wkt']``.
        See ``open_geotiff`` for the full description.
    on_gpu_failure : str, optional
        Accepted for cross-backend signature symmetry only. The dask
        path runs CPU decoders, so passing this kwarg raises
        ``ValueError`` at dispatch. See ``read_geotiff_gpu`` for the
        kwarg's meaning on the GPU reader.
    missing_sources : {'raise', 'warn'}, optional
        VRT-only. Forwarded to ``read_vrt`` when the source ends in
        ``.vrt``; otherwise raises ``ValueError`` at dispatch. See
        ``read_vrt`` for the full description.
    max_cloud_bytes : int or None, optional
        Accepted for cross-backend signature symmetry only. The dask
        reader uses bounded range GETs and does not consume the
        cloud-byte budget, so passing this kwarg raises ``ValueError``
        at dispatch. See ``open_geotiff`` for the eager-path
        description (issue #1974).

    Returns
    -------
    xr.DataArray
        Dask-backed DataArray with y/x coordinates.
    """
    import dask.array as da

    from .._reader import _coerce_path

    source = _coerce_path(source)

    # Shared dispatcher-kwarg validator so direct callers see the same
    # rejections as ``open_geotiff`` (issue #2175 / parent #2162).
    # Runs ``_validate_overview_level_arg`` first to match ``open_geotiff``'s
    # ordering -- a bad ``overview_level`` is reported before unrelated
    # source / ``chunks=`` errors mask it (issue #2160). The helper also
    # rejects ``on_gpu_failure`` (CPU dask has no GPU policy),
    # ``missing_sources`` on non-VRT, ``band_nodata`` on non-VRT (issue
    # #1987), ``max_cloud_bytes`` (dask reader does not apply the
    # cloud-byte budget, issue #1974), and the file-like-source guard.
    # ``gpu=False`` because this entry point is always CPU dask.
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

    # Reject non-positive chunk sizes up front. ``chunks=0`` and negative
    # values otherwise propagate into dask chunk math (``range(0, N, 0)``
    # ValueError, or empty chunk grids) with no indication that ``chunks``
    # was the problem. Shared with ``read_geotiff_gpu`` / ``read_vrt`` via
    # ``_validate_chunks_arg`` so all three entry points emit the same
    # error format (#1752 / #1776). ``allow_none=False`` (the default)
    # rejects ``chunks=None`` with the same ValueError; this entry point
    # requires a concrete chunk size since the chunk-unpacking math below
    # would otherwise fail with a confusing TypeError.
    chunks = _validate_chunks_arg(chunks)

    # ``open_geotiff`` already routes ``.vrt`` to ``read_vrt`` before
    # reaching here, so this branch is only hit when ``read_geotiff_dask``
    # is called directly with a VRT path. Keep it as a defensive fallback
    # rather than letting the windowed-read path try to parse VRT XML as
    # TIFF bytes. ``read_vrt`` is the single source of truth for VRT.
    if isinstance(source, str) and source.lower().endswith('.vrt'):
        vrt_kwargs = {}
        if missing_sources is not _MISSING_SOURCES_SENTINEL:
            vrt_kwargs['missing_sources'] = missing_sources
        return read_vrt(
            source, dtype=dtype, window=window, band=band, name=name,
            chunks=chunks, max_pixels=max_pixels,
            allow_rotated=allow_rotated,
            allow_unparseable_crs=allow_unparseable_crs,
            band_nodata=band_nodata,
            mask_nodata=mask_nodata,
            **vrt_kwargs,
        )

    # P5: HTTP COG sources used to fire one IFD/header GET per chunk
    # task. Parse metadata once here so every delayed task can reuse it.
    # The same prefetch path also covers fsspec URIs (s3://, gs://, ...);
    # ``_parse_cog_http_meta`` only needs a ``read_range``-having source,
    # and ``_CloudSource`` satisfies that contract. Going through it
    # bounds metadata reads to ``MAX_HTTP_HEADER_BYTES`` instead of
    # fetching the whole remote object up front. See PR #1755 review.
    from .._sources import _is_http_source
    from .._reader import _is_fsspec_uri
    is_http = _is_http_source(source)
    is_fsspec = isinstance(source, str) and _is_fsspec_uri(source)
    http_meta = None
    http_meta_key = None
    # ``effective_source`` is the path each per-chunk task should
    # construct its ``_HTTPSource`` / ``_CloudSource`` against. It
    # matches ``source`` unless the requested ``overview_level``
    # resolves into an external ``.tif.ovr`` sidecar (issue #2239), in
    # which case it is swapped to the sidecar URL so the per-chunk
    # range GETs land on the right object.
    effective_source = source
    if is_http or is_fsspec:
        import dask

        from .._cog_http import _parse_cog_http_meta
        if is_http:
            # ``_HTTPSource`` is resolved through ``_reader`` so existing
            # tests that ``monkeypatch.setattr(_reader, '_HTTPSource', ...)``
            # keep intercepting the construction after the COG-HTTP
            # helpers moved to ``_cog_http`` (PR-J / #2258).
            from .._reader import _HTTPSource
            _src = _HTTPSource(source)
        else:
            from .._reader import _CloudSource
            _src = _CloudSource(source)
        sidecar = None
        try:
            # Forward ``allow_rotated`` so the HTTP dask path honours the
            # rotated opt-in the same way as the eager HTTP path and the
            # local chunked path (#2130). Without this, rotated remote
            # GeoTIFFs raised ``NotImplementedError`` from
            # ``_parse_cog_http_meta`` (now ``RotatedTransformError``
            # after #2267) even when the caller had passed
            # ``allow_rotated=True``.
            #
            # ``source_path=source`` and ``return_sidecar=True`` opt the
            # parser into external ``.tif.ovr`` discovery so chunked
            # remote reads honour the same overview pyramid the eager
            # reader resolves (issue #2239). When the selected overview
            # lives in the sidecar, ``route_path`` points at the
            # sidecar URL and we swap ``effective_source`` so every
            # per-chunk task reads from the sidecar instead of the base
            # URL.
            (http_header, http_ifd, http_geo, _, sidecar_meta
             ) = _parse_cog_http_meta(
                _src, overview_level=overview_level,
                allow_rotated=allow_rotated,
                source_path=source,
                return_sidecar=True,
            )
            sidecar, route_path, used_sidecar = sidecar_meta
            if used_sidecar:
                effective_source = route_path
        finally:
            _src.close()
            # Sidecar bytes (HTTP / fsspec) live in memory; the
            # per-chunk task does not reuse them, so the metadata-only
            # graph build releases them here. The local-file mmap path
            # is owned by this function's frame; ``close_sidecar`` is
            # a no-op for ``bytes`` and idempotent for ``mmap``.
            if sidecar is not None:
                from .._sidecar import close_sidecar
                close_sidecar(sidecar)
        # ``http_header`` carries the sidecar's ``TIFFHeader`` when
        # ``used_sidecar`` was True (``_parse_cog_http_meta`` swaps it
        # so ``byte_order`` matches the file the per-chunk range GETs
        # land on). Pass that through to the per-chunk decode step so a
        # mixed-endian base / ``.ovr`` pair decodes against the right
        # endianness. Issue #2314.
        http_meta = (http_header, http_ifd)
        if http_ifd.orientation != 1:
            raise ValueError(
                f"Orientation tag (274) is {http_ifd.orientation}; "
                f"dask-chunked reads (chunks=...) are not supported for "
                f"non-default orientation on remote GeoTIFF sources. Read "
                f"the full array first, then slice/chunk it."
            )
        # Wrap the parsed metadata in a single dask Delayed so every
        # window task takes it as a graph input, not a Python closure.
        # Without this, the (TIFFHeader, IFD) pair -- which can carry
        # multi-million-entry TileOffsets/TileByteCounts tuples on
        # large COGs -- would be embedded in each chunk task and
        # serialised N times under distributed/process schedulers.
        http_meta_key = dask.delayed(http_meta, pure=True)
        geo_info = http_geo
        full_h = http_ifd.height
        full_w = http_ifd.width
        from .._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
        from .._validation import _validate_predictor_sample_format
        bps = resolve_bits_per_sample(http_ifd.bits_per_sample)
        file_dtype = tiff_dtype_to_numpy(bps, http_ifd.sample_format)
        _validate_predictor_sample_format(
            http_ifd.predictor, http_ifd.sample_format)
        n_bands = (
            http_ifd.samples_per_pixel
            if http_ifd.samples_per_pixel > 1 else 0
        )
        # Stash IFD photometric for the MinIsWhite nodata-inversion check below.
        geo_info._ifd_photometric = http_ifd.photometric
        geo_info._ifd_samples_per_pixel = http_ifd.samples_per_pixel
    else:
        # Metadata-only read: O(1) memory via mmap, no pixel decompression.
        # Lazy import for the same circular-import reason as ``read_vrt``
        # above: ``_read_geo_info`` still lives in ``xrspatial.geotiff``.
        from .. import _read_geo_info
        geo_info, full_h, full_w, file_dtype, n_bands = _read_geo_info(
            source, overview_level=overview_level,
            allow_rotated=allow_rotated)
    # PR-C #2226: centralize the nodata lifecycle in one value object.
    # ``raw_sentinel`` carries the pre-inversion sentinel that
    # ``attrs['nodata']`` must preserve; ``effective_sentinel`` is what
    # the per-chunk reader's mask compares against (post-MinIsWhite).
    # ``sentinel_fits_buffer`` collapses the previous finite/integer/
    # in-range gate so the integer auto-promotion below skips out-of-
    # range / fractional / non-finite sentinels (#1774, #1564, #1616).
    #
    # The ``_ifd_photometric`` and ``_ifd_samples_per_pixel`` attrs are
    # stashed in lockstep by ``_read_geo_info`` (``__init__.py``) and the
    # HTTP / sidecar branches above. When both are missing (synthesised
    # geo_info from a non-TIFF source), ``photometric=None`` already
    # short-circuits ``_applies_miniswhite`` to False inside the helper,
    # so the ``or 1`` fallback on ``samples_per_pixel`` is harmless --
    # it never reaches the inversion branch on a real
    # ``samples_per_pixel=N`` file.
    lifecycle = NodataLifecycle(
        declared=geo_info.nodata,
        photometric=getattr(geo_info, '_ifd_photometric', None),
        dtype_in=file_dtype,
        dtype_request=dtype,
        samples_per_pixel=getattr(geo_info, '_ifd_samples_per_pixel', 1) or 1,
    )
    nodata_attr = lifecycle.raw_sentinel  # original sentinel for attrs['nodata']
    nodata = lifecycle.effective_sentinel  # post-MinIsWhite for masking

    # Nodata masking promotes integer arrays to float64 (for NaN).
    # The lifecycle's ``sentinel_fits_buffer`` already encapsulates the
    # finite / integer / in-range gates (#1774, #1564, #1616), so the
    # promotion fires iff the helper says the sentinel is comparable.
    effective_dtype = file_dtype
    if (mask_nodata
            and file_dtype.kind in ('u', 'i')
            and lifecycle.sentinel_fits_buffer):
        effective_dtype = np.dtype('float64')

    if dtype is not None:
        target_dtype = np.dtype(dtype)
        _validate_dtype_cast(effective_dtype, target_dtype)
    else:
        target_dtype = effective_dtype

    # Window clipping: restrict the lazy region to the requested
    # sub-rectangle. ``read_to_array`` already accepts ``window=`` per
    # chunk; we only need to remap the chunk grid so its origin moves to
    # ``(win_r0, win_c0)`` and its extent shrinks to the window.
    win_r0 = win_c0 = 0
    if window is not None:
        win_r0, win_c0, win_r1, win_c1 = window
        if (win_r0 < 0 or win_c0 < 0
                or win_r1 > full_h or win_c1 > full_w
                or win_r0 >= win_r1 or win_c0 >= win_c1):
            raise ValueError(
                f"window={window} is outside the source extent "
                f"({full_h}x{full_w}) or has non-positive size.")
        # Mirror the eager-path windowed coord computation in open_geotiff,
        # including the ``has_georef=False`` shortcut to integer pixel
        # coords for non-georef files (#1710).
        coords = _coords_from_geo_info(
            geo_info, win_r1 - win_r0, win_c1 - win_c0, window=window,
        )
        full_h = win_r1 - win_r0
        full_w = win_c1 - win_c0
    else:
        coords = _geo_to_coords(geo_info, full_h, full_w)

    if band is not None:
        # Reject ``bool`` and ``np.bool_`` up front; ``isinstance(True, int)``
        # is True in Python so ``True < n_bands`` evaluates without raising
        # and silently reads band 1. ``np.bool_`` is not a subclass of
        # ``bool`` so it needs its own check to match the VRT path's
        # rejection. See #1786.
        if isinstance(band, (bool, np.bool_)):
            raise ValueError(
                f"band must be a non-negative int, got {band!r}")
        # Reject non-integer numeric types and anything else that would
        # slip past the bool guard. ``band=0.0`` passes the range check
        # below and either silently reads band 0 (single-band) or fails
        # with a raw numpy ``IndexError`` (multi-band). The VRT paths
        # already enforce this; mirror them here. See #1910.
        if not isinstance(band, (int, np.integer)):
            raise TypeError(
                f"band must be a non-negative int, got {band!r}")
        if n_bands == 0:
            if band != 0:
                raise IndexError(
                    f"band={band} requested on a single-band file.")
        elif not 0 <= band < n_bands:
            raise IndexError(
                f"band={band} out of range for {n_bands}-band file.")

    # Up-front pixel-count guard against the windowed extent. Chunk
    # tasks re-check via read_to_array's own ``max_pixels`` (which we
    # forward through ``_delayed_read_window``), but catching an
    # oversized request before any task is scheduled saves the caller
    # from a misleading "tile size exceeds max_pixels" error in a
    # chunk that happens to align with the file's tile grid.
    # ``max_pixels=None`` substitutes the module default to match the
    # eager (``read_to_array``) and VRT chunked paths. Without the
    # substitution the guard would skip entirely on ``None`` and a
    # caller could build a lazy graph over a region far larger than the
    # documented safety cap. See issue #1838.
    from .._reader import MAX_PIXELS_DEFAULT as _MAX_PIXELS_DEFAULT
    effective_max_pixels = (max_pixels if max_pixels is not None
                            else _MAX_PIXELS_DEFAULT)
    eff_bands = (1 if band is not None
                 else (n_bands if n_bands > 0 else 1))
    if full_h * full_w * eff_bands > effective_max_pixels:
        raise ValueError(
            f"Requested region {full_h}x{full_w}x{eff_bands} "
            f"exceeds max_pixels={effective_max_pixels:,}.")

    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    # Wave 2 of #2162: share the validate-then-populate-then-stamp
    # block with the dask+GPU backend via ``_finalize_lazy_read_attrs``.
    #
    # ``graph_dtype`` is the resolved dask graph dtype so
    # ``masked_nodata`` reflects whether the per-chunk mask actually
    # ran (masking on an int source auto-promotes to float64; an
    # un-promoted int graph means masking didn't run -- #2092).
    # ``caller_dtype`` is the caller's ``dtype=`` kwarg verbatim so
    # ``nodata_dtype_cast`` records caller intent rather than the
    # masking-induced auto-promotion.
    #
    # ``nodata_attr`` (not the MinIsWhite-inverted ``nodata``) is what
    # ``attrs['nodata']`` must carry; the helper threads it through
    # ``_set_nodata_attrs``. ``nodata_pixels_present`` stays unset on
    # this path so the lazy contract from #2135 holds (a strict
    # per-chunk reduction would force eager compute).
    attrs = _finalize_lazy_read_attrs(
        geo_info=geo_info,
        nodata=nodata_attr,
        mask_nodata=mask_nodata,
        graph_dtype=target_dtype,
        caller_dtype=dtype,
        window=window,
        allow_rotated=allow_rotated,
        allow_unparseable_crs=allow_unparseable_crs,
    )

    if isinstance(chunks, int):
        ch_h = ch_w = chunks
    else:
        ch_h, ch_w = chunks

    # Graph-size guard. Each chunk becomes a delayed task whose Python graph
    # entry retains ~1KB. At very large chunk counts the graph itself OOMs
    # the driver before any read executes (30TB at chunks=256 => ~500M tasks
    # => ~500GB graph on host). Refuse anything past the cap and ask the
    # caller to pick a chunk size, rather than silently rescaling -- the
    # rescaled chunks may not align with the user's downstream pipeline.
    _MAX_DASK_CHUNKS = 50_000
    n_chunks = ((full_h + ch_h - 1) // ch_h) * ((full_w + ch_w - 1) // ch_w)
    if n_chunks > _MAX_DASK_CHUNKS:
        import math
        scale = math.sqrt(n_chunks / _MAX_DASK_CHUNKS)
        suggested_h = int(math.ceil(ch_h * scale))
        suggested_w = int(math.ceil(ch_w * scale))
        raise ValueError(
            f"read_geotiff_dask: chunks=({ch_h}, {ch_w}) on a "
            f"{full_h}x{full_w} image would produce {n_chunks:,} dask "
            f"tasks, exceeding the {_MAX_DASK_CHUNKS:,}-task cap. Pass a "
            f"larger chunks=... value explicitly (e.g. chunks="
            f"({suggested_h}, {suggested_w}) keeps the task count under "
            "the cap)."
        )

    # Build dask array from delayed windowed reads
    rows = list(range(0, full_h, ch_h))
    cols = list(range(0, full_w, ch_w))

    # For multi-band, each window read returns (h, w, bands); for single-band (h, w)
    # read_to_array with band=0 extracts a single band, band=None returns all
    band_arg = band  # None => all bands (or 2D for single-band file)

    # When ``band`` is set, each chunk reads a 2D slice -- collapse the
    # output dims so the returned DataArray is 2D regardless of file band
    # count.
    out_has_band_axis = band is None and n_bands > 0

    dask_rows = []
    for r0 in rows:
        r1 = min(r0 + ch_h, full_h)
        dask_cols = []
        for c0 in cols:
            c1 = min(c0 + ch_w, full_w)
            if out_has_band_axis:
                block_shape = (r1 - r0, c1 - c0, n_bands)
            else:
                block_shape = (r1 - r0, c1 - c0)
            # Translate window-relative chunk coords back to file-relative
            # coords for ``read_to_array``. ``win_r0`` / ``win_c0`` are 0
            # when no window was requested.
            # Always thread ``target_dtype`` so each delayed chunk lands
            # in the same dtype that the dask array declared. Without
            # this, an integer raster with an in-range nodata sentinel
            # would have ``effective_dtype=float64`` declared on the
            # dask graph but per-chunk arrays only promoted when a
            # chunk happened to contain a sentinel pixel. Dask
            # concatenation then preallocates from the first chunk's
            # actual dtype (uint16), silently casting later float64
            # chunks back to int and converting their NaNs to 0. See
            # issue #1597.
            # Per-chunk nodata mask is skipped when ``mask_nodata=False``;
            # passing ``nodata=None`` short-circuits both the float-NaN and
            # int-promotion branches in ``_delayed_read_window``. The
            # original sentinel is still carried in ``attrs['nodata']`` via
            # ``nodata_attr`` so write round-trips preserve the tag. See
            # issue #2052.
            chunk_nodata = nodata if mask_nodata else None
            # ``effective_source`` swaps in the sidecar URL when the
            # requested overview lives in an external ``.tif.ovr``
            # (issue #2239). For local files and non-sidecar remote
            # reads it is identical to ``source``. The per-chunk task
            # uses ``effective_source`` for byte fetches; combined
            # with ``http_meta_key`` (which already carries the
            # sidecar's ``(header, ifd)`` pair), this routes every
            # range GET at the sidecar object instead of the base URL.
            block = da.from_delayed(
                _delayed_read_window(effective_source,
                                     r0 + win_r0, c0 + win_c0,
                                     r1 + win_r0, c1 + win_c0,
                                     overview_level, chunk_nodata,
                                     band_arg,
                                     target_dtype=target_dtype,
                                     http_meta_key=http_meta_key,
                                     max_pixels=max_pixels,
                                     allow_rotated=allow_rotated),
                shape=block_shape,
                dtype=target_dtype,
            )
            dask_cols.append(block)
        dask_rows.append(da.concatenate(dask_cols, axis=1))

    dask_arr = da.concatenate(dask_rows, axis=0)

    if out_has_band_axis:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(n_bands)
    else:
        dims = ['y', 'x']

    return xr.DataArray(
        dask_arr, dims=dims, coords=coords, name=name, attrs=attrs,
    )


def _delayed_read_window(source, r0, c0, r1, c1, overview_level, nodata,
                         band, *, target_dtype=None, http_meta_key=None,
                         max_pixels=None, allow_rotated=False):
    """Dask-delayed function to read a single window.

    *http_meta_key* is an optional ``Delayed[(TIFFHeader, IFD)]`` parsed
    once by :func:`read_geotiff_dask` and wrapped via ``dask.delayed``.
    Passing it as a function argument (rather than a closure capture)
    makes the metadata a single graph input that all window tasks
    depend on, so distributed/process schedulers serialise it once
    instead of once per chunk. For local sources it is ``None``.
    """
    import dask

    @dask.delayed
    def _read(http_meta):
        # The prefetched-metadata fast path covers both HTTP COGs and
        # fsspec-addressable remotes (s3://, gs://, az://, memory://, ...).
        # Both source classes expose ``read_range``, which is all
        # ``_fetch_decode_cog_http_tiles`` needs.
        _is_http_src = isinstance(source, str) and source.startswith(
            ('http://', 'https://'))
        _is_fsspec_src = False
        if http_meta is not None and isinstance(source, str) and \
                not _is_http_src:
            from .._reader import _is_fsspec_uri as _ifs
            _is_fsspec_src = _ifs(source)
        if http_meta is not None and (_is_http_src or _is_fsspec_src):
            from .._cog_http import _fetch_decode_cog_http_tiles
            from .._decode import _apply_photometric_miniswhite
            from .._layout import MAX_PIXELS_DEFAULT
            header, ifd = http_meta
            if _is_http_src:
                # See PR-J / #2258: keep the per-chunk ``_HTTPSource`` /
                # ``_CloudSource`` construction routed through ``_reader``
                # so monkeypatches against ``_reader._HTTPSource`` (used
                # by the dask-HTTP coalesce tests) still take effect.
                from .._reader import _HTTPSource
                src = _HTTPSource(source)
            else:
                from .._reader import _CloudSource
                src = _CloudSource(source)
            try:
                arr = _fetch_decode_cog_http_tiles(
                    src, header, ifd,
                    max_pixels=(max_pixels if max_pixels is not None
                                else MAX_PIXELS_DEFAULT),
                    window=(r0, c0, r1, c1))
            finally:
                src.close()
            if (arr.ndim == 3 and ifd.samples_per_pixel > 1
                    and band is not None):
                arr = arr[:, :, band]
            arr = _apply_photometric_miniswhite(arr, ifd)
        else:
            _r2a_kwargs = {}
            if max_pixels is not None:
                _r2a_kwargs['max_pixels'] = max_pixels
            arr, _ = _read_to_array(source, window=(r0, c0, r1, c1),
                                    overview_level=overview_level,
                                    band=band,
                                    allow_rotated=allow_rotated,
                                    **_r2a_kwargs)
        if nodata is not None:
            # ``arr`` was just decoded by ``_fetch_decode_cog_http_tiles``
            # or ``read_to_array``; both return freshly-allocated buffers
            # that nothing else references, so the in-place sentinel
            # rewrite is safe. Skip the defensive ``arr.copy()`` to
            # avoid a peak-memory doubler on every dask chunk.
            if arr.dtype.kind == 'f' and not np.isnan(nodata):
                arr[arr == arr.dtype.type(nodata)] = np.nan
            elif (arr.dtype.kind in ('u', 'i')
                  and np.isfinite(nodata)
                  and float(nodata).is_integer()):
                # Out-of-range sentinels (e.g. uint16 + nodata=-9999)
                # cannot match any pixel; skip the cast that would
                # otherwise raise OverflowError and leave arr unchanged.
                # Non-finite sentinels ("NaN" / "Inf" GDAL_NODATA strings)
                # also cannot match an integer pixel and would raise
                # ValueError on ``int(nodata)``; the ``np.isfinite`` gate
                # mirrors ``_resolve_masked_fill`` in ``_reader.py``
                # (#1774). Fractional sentinels (e.g. ``"3.5"`` on a
                # ``uint16`` file) also cannot match an integer pixel and
                # ``int(3.5)`` would truncate to 3 and silently mask
                # pixel value 3; the ``float(nodata).is_integer()`` gate
                # short-circuits them too.
                nodata_int = int(nodata)
                info = np.iinfo(arr.dtype)
                if info.min <= nodata_int <= info.max:
                    mask = arr == arr.dtype.type(nodata_int)
                    if mask.any():
                        arr = arr.astype(np.float64)
                        arr[mask] = np.nan
        if target_dtype is not None and arr.dtype != target_dtype:
            # Skip the cast when dtype already matches. ``numpy.astype``
            # defaults to ``copy=True`` and would otherwise allocate a
            # full chunk-sized buffer and memcpy on every read just to
            # land in the same dtype the array already has. The int->
            # float64 promotion above (sentinel-hit branch) keeps the
            # contract that every chunk lands in the dask-declared
            # dtype; this guard only elides no-op casts. See #1624.
            arr = arr.astype(target_dtype)
        return arr
    return _read(http_meta_key)
