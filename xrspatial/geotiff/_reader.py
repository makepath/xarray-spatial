"""TIFF/COG reader: tile/strip assembly, windowed reads, HTTP range requests.

This module is private to :mod:`xrspatial.geotiff`. The supported public
read entry points are :func:`xrspatial.geotiff.open_geotiff`,
:func:`xrspatial.geotiff.read_geotiff_gpu`,
:func:`xrspatial.geotiff.read_geotiff_dask`, and
:func:`xrspatial.geotiff.read_vrt`. Direct callers of the helpers
defined here bypass the DataArray-level work that the public wrappers
perform (ambiguous-metadata fail-closed, nodata-to-NaN promotion,
``masked_nodata`` attr, ``transform`` / ``crs`` attrs population) and
have to replicate those steps by hand. See issue #2138.

For source modules inside :mod:`xrspatial.geotiff`, the canonical
internal name for the array-level reader is :func:`_read_to_array`.
The non-underscored :func:`read_to_array` is kept as an alias for
internal call sites that pre-date the rename.
"""
from __future__ import annotations

import numpy as np
# ``urllib3`` is kept as a top-level import here even though the HTTP
# source moved to ``_sources`` in #2228. ``test_http_no_stdlib_fallback_2050``
# asserts the reader module carries a module-level urllib3 reference so a
# build that silently drops the dependency cannot ship. The HTTP code path
# itself uses the ``_sources`` import; this binding is purely the
# "urllib3 is a hard install dep" guard.
import urllib3  # noqa: F401

# COG-over-HTTP transport (bounded header prefetch, range-based tile/strip
# fetch + decode) lives in ``_cog_http``. It is imported back here so that:
#   * existing call sites inside this module (``_read_cog_http``,
#     ``_parse_cog_http_meta``) keep their bare names, and
#   * the historical public import surface
#     (``from xrspatial.geotiff._reader import _read_cog_http`` and
#     friends, used by the dask backend, the test suite, and external
#     code that patches ``_reader._HTTPSource`` / ``_reader._parse_cog_http_meta``)
#     stays intact without churn.
# ``_cog_http._read_cog_http`` resolves ``_HTTPSource`` and
# ``_parse_cog_http_meta`` through this module (via ``from . import _reader``
# at call time) so monkeypatches against ``_reader._HTTPSource`` /
# ``_reader._parse_cog_http_meta`` continue to take effect after the move.
# Source: PR-J of the GeoTIFF refactor epic, issue #2258.
from ._cog_http import (INITIAL_HTTP_HEADER_BYTES, MAX_HTTP_HEADER_BYTES,  # noqa: F401
                        _fetch_decode_cog_http_strips, _fetch_decode_cog_http_tiles,
                        _parse_cog_http_meta, _read_cog_http)
# Strip/tile decode orchestration lives in ``_decode``. It is imported
# back here so that:
#   * existing call sites inside this module (``_read_strips``,
#     ``_read_tiles``, ``_decode_strip_or_tile``, the photometric and
#     orientation helpers) keep their bare names, and
#   * the historical public import surface
#     (``from xrspatial.geotiff._reader import _read_strips`` and
#     friends, used by VRT / GPU / dask backends, the writer, and the
#     test suite) is preserved without churn.
# Source: PR-G of the GeoTIFF refactor epic, issue #2246.
from ._decode import (_NATIVE_ORDER, _PARALLEL_DECODE_PIXEL_THRESHOLD,  # noqa: F401
                      _apply_orientation, _apply_orientation_with_geo,
                      _apply_photometric_miniswhite, _apply_predictor, _decode_strip_or_tile,
                      _int_nodata_in_range, _miniswhite_inverted_nodata, _packed_byte_count,
                      _read_strips, _read_tiles, _resolve_masked_fill)
from ._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
from ._geotags import GeoInfo, extract_geo_info_with_overview_inheritance
from ._header import parse_all_ifds, parse_header, select_overview_ifd
# Layout / validation helpers live in ``_layout``. They are imported back
# here so that:
#   * existing call sites inside this module keep their bare names, and
#   * the historical public import surface
#     (``from xrspatial.geotiff._reader import PixelSafetyLimitError``,
#     ``MAX_PIXELS_DEFAULT``, ``_check_dimensions`` and friends -- used
#     by sidecar / VRT / GPU / dask backends and by the test suite) is
#     preserved without churn.
# Source: PR-H of the GeoTIFF refactor epic, issue #2247.
from ._layout import (_FULL_IMAGE_BUDGET_HEADER_SLACK, MAX_PIXELS_DEFAULT,  # noqa: F401
                      PixelSafetyLimitError, _check_dimensions, _check_source_dimensions,
                      _compute_full_image_byte_budget, _has_sparse, _ifd_required_extent,
                      _sparse_fill_value)
# The data-source layer (local mmap, HTTP with SSRF defences and DNS-rebind
# pinning, fsspec cloud, BytesIO) lives in ``_sources``. It is imported back
# here so that:
#   * existing call sites inside this module (``_open_source``, ``_HTTPSource``,
#     ``_FileSource`` etc.) keep their bare names, and
#   * the historical public import surface
#     (``from xrspatial.geotiff._reader import _HTTPSource`` and friends,
#     used by sidecar / VRT / GPU / dask backends and by the test suite) is
#     preserved without churn.
# Source: PR-E of the GeoTIFF refactor epic, issue #2228.
from ._sources import (_CLOUD_SCHEMES, _DEFAULT_MMAP_CACHE_SIZE,  # noqa: F401
                       _HTTP_ALLOWED_SCHEMES, _HTTP_CONNECT_TIMEOUT_DEFAULT, _HTTP_MAX_REDIRECTS,
                       _HTTP_READ_TIMEOUT_DEFAULT, _MAX_CLOUD_BYTES_SENTINEL,
                       COALESCE_GAP_THRESHOLD_DEFAULT, MAX_CLOUD_BYTES_DEFAULT,
                       MAX_COALESCED_RANGE_BYTES_DEFAULT, MAX_TILE_BYTES_DEFAULT,
                       CloudSizeLimitError, UnsafeURLError, _build_pinned_connection_classes,
                       _BytesIOSource, _CloudSource, _coerce_path, _FileSource, _get_http_pool,
                       _get_pinned_conn_classes, _http_allow_private_hosts, _http_connect_timeout,
                       _http_read_timeout, _http_timeout_from_env, _HTTPSource, _ip_is_private,
                       _is_file_like, _is_fsspec_uri, _make_pinned_pool,
                       _max_coalesced_range_bytes_from_env, _max_tile_bytes_from_env, _mmap_cache,
                       _mmap_cache_size_from_env, _MmapCache, _open_source,
                       _resolve_max_cloud_bytes, _validate_http_url, coalesce_ranges,
                       split_coalesced_bytes)

# ---------------------------------------------------------------------------
# Main read function
# ---------------------------------------------------------------------------


def _read_to_array(source, *, window=None, overview_level: int | None = None,
                   band: int | None = None,
                   max_pixels: int = MAX_PIXELS_DEFAULT,
                   max_cloud_bytes=_MAX_CLOUD_BYTES_SENTINEL,
                   allow_rotated: bool = False,
                   ) -> tuple[np.ndarray, GeoInfo]:
    """Read a GeoTIFF/COG to a numpy array (module-private).

    Parameters
    ----------
    source : str or binary file-like
        File path, URL, or a file-like object with ``read``/``seek``.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop).
    overview_level : int or None
        Overview level (0 = full res).
    band : int
        Band index for multi-band files.
    max_pixels : int
        Maximum allowed total pixel count (width * height * samples).
        Prevents memory exhaustion from crafted TIFF headers.
        Default is 1 billion (~4 GB for float32 single-band).
    max_cloud_bytes : int or None, optional
        Byte ceiling for eager reads from fsspec sources (``s3://``,
        ``gs://``, ``az://``, ``abfs://``, ``memory://``, ...). The
        compressed object size is checked against this budget before any
        bytes are downloaded. Default is :data:`MAX_CLOUD_BYTES_DEFAULT`
        (256 MiB), overridable via the
        ``XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES`` env var. Pass ``None`` to
        skip the check entirely (pre-#1928 behaviour). The HTTP path
        already reads only what it needs via range requests and is not
        subject to this limit. See issue #1928.

    Returns
    -------
    (np.ndarray, GeoInfo) tuple
    """
    source = _coerce_path(source)
    if isinstance(source, str) and source.startswith(('http://', 'https://')):
        return _read_cog_http(source, overview_level=overview_level, band=band,
                              max_pixels=max_pixels, window=window,
                              allow_rotated=allow_rotated)

    # Local file, cloud storage, or file-like buffer: read all bytes then parse
    # Resolve the cloud byte budget once so both the base-file ``_CloudSource``
    # size guard and the sidecar download below see the same effective cap.
    # ``_resolve_max_cloud_bytes`` honours the kwarg, the env var, and the
    # default in that order; the result is ``None`` only when the caller
    # explicitly passed ``max_cloud_bytes=None``.
    cloud_budget = _resolve_max_cloud_bytes(max_cloud_bytes)
    if _is_file_like(source):
        src = _BytesIOSource(source)
    elif _is_fsspec_uri(source):
        src = _CloudSource(source)
        # Check the compressed object size before any bytes are
        # downloaded. ``_CloudSource.__init__`` already fetched the size
        # via ``fsspec.size()``, so this is free. See issue #1928.
        if cloud_budget is not None:
            size = src.size
            if size is None:
                src.close()
                raise CloudSizeLimitError(
                    f"Cloud source {source!r} reports unknown size; "
                    f"refusing to download to avoid an unbounded read. "
                    f"Pass max_cloud_bytes=None to disable the size "
                    f"check for this source. Raising the byte limit "
                    f"does not help when the source size is unknown.")
            if size > cloud_budget:
                src.close()
                raise CloudSizeLimitError(
                    f"Cloud source {source!r} is {size:,} bytes, which "
                    f"exceeds max_cloud_bytes={cloud_budget:,}. Eager "
                    f"reads pull the full object before any TIFF header "
                    f"parse; raise max_cloud_bytes (or set "
                    f"XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES) if the file is "
                    f"legitimate, pass max_cloud_bytes=None to disable "
                    f"the check, or use chunks=... for a windowed dask "
                    f"read.")
    else:
        src = _FileSource(source)
    data = src.read_all()

    sidecar = None
    try:
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)

        if len(ifds) == 0:
            raise ValueError("No IFDs found in TIFF file")

        # External `.tif.ovr` sidecar (issue #2112). GDAL/rasterio write
        # overview pyramids to a sibling file when the source is not a
        # COG; the sidecar's IFDs are the continuation of the base
        # file's pyramid. Discovery fires for local files, HTTP, and
        # fsspec sources; file-like buffers skip the lookup.
        # ``max_cloud_bytes`` propagates to ``load_sidecar`` so the
        # sidecar fetch inherits the same byte budget the base file
        # enforces (#2121). The sidecar must be loaded before IFD
        # selection so ``overview_level`` indexes into a unified
        # pyramid list.
        from ._sidecar import attach_sidecar_origin, find_sidecar, load_sidecar
        sidecar_origin: dict[int, tuple] = {}
        sidecar_path = find_sidecar(source)
        if sidecar_path is not None:
            sidecar = load_sidecar(sidecar_path,
                                   max_cloud_bytes=cloud_budget)
            sidecar_origin = attach_sidecar_origin(
                sidecar.ifds, sidecar.data, sidecar.header)
            ifds = ifds + sidecar.ifds

        # Select IFD, skipping any mask IFDs
        ifd = select_overview_ifd(ifds, overview_level)

        # If the selected IFD came from the sidecar, swap the data /
        # header used for strip / tile reads below so byte offsets
        # resolve against the right buffer.
        ifd_data, ifd_header = sidecar_origin.get(id(ifd), (data, header))

        bps = resolve_bits_per_sample(ifd.bits_per_sample)
        dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
        # Inherit georef from level 0 when an overview IFD lacks its own
        # geokeys (issue #1640). For overview_level=0 (or None) this is a
        # no-op: the helper short-circuits when the IFD is not a
        # NewSubfileType=overview entry. Sidecar IFDs typically lack
        # geokeys (the GDAL convention), so the inheritance pulls from
        # the base file's level-0 IFD (kept first in the merged list).
        # A sidecar that does declare its own georef payload is a corner
        # case: ``georef_origin`` maps the sidecar IFDs to
        # ``(data, byte_order)`` from the sidecar so the helper resolves
        # those tags against the right buffer. Sidecar IFDs without
        # geokeys still inherit from the base file via the existing
        # overview-inheritance path. See issues #1640 and #2315.
        georef_origin = (
            {iid: (od, oh.byte_order)
             for iid, (od, oh) in sidecar_origin.items()}
            if sidecar_origin else None
        )
        geo_info = extract_geo_info_with_overview_inheritance(
            ifd, ifds, data, header.byte_order,
            allow_rotated=allow_rotated,
            sidecar_origin=georef_origin)

        # Orientation tag (274): values 2-8 mean the stored pixel order
        # differs from display order. We need to remap the array post
        # decode. A windowed read against a non-default orientation has
        # ambiguous semantics (does the window refer to file pixels or
        # display pixels?) so we reject that combo rather than guess.
        # ``read_geotiff_dask`` chunks the file by issuing windowed reads,
        # so this check also rejects ``chunks=`` for non-default
        # orientation; the error mentions both so the failure is easy to
        # diagnose if it surfaces under dask.
        orientation = ifd.orientation
        if orientation != 1 and window is not None:
            raise ValueError(
                f"Orientation tag (274) is {orientation}; windowed reads "
                f"(window=...) and dask-chunked reads (chunks=...) are not "
                f"supported for non-default orientation. Read the full "
                f"array first, then slice."
            )

        # Validate ``window`` against the selected IFD's extent. Without
        # this, ``_read_tiles`` / ``_read_strips`` silently clamp an
        # out-of-bounds window and return a smaller array, which then
        # mismatches caller-built coord arrays in ``open_geotiff`` and
        # surfaces as an opaque ``CoordinateValidationError``. Raising
        # here matches the dask path's pre-flight validator (see
        # ``read_geotiff_dask`` in ``__init__.py``) so all backends
        # agree on the contract. Reuses the IFD already parsed above,
        # so callers pay no extra metadata-parse cost (file-like
        # sources are read once instead of twice). See issue #1634.
        if window is not None:
            w_r0, w_c0, w_r1, w_c1 = window
            if (w_r0 < 0 or w_c0 < 0
                    or w_r1 > ifd.height or w_c1 > ifd.width
                    or w_r0 >= w_r1 or w_c0 >= w_c1):
                raise ValueError(
                    f"window={window} is outside the source extent "
                    f"({ifd.height}x{ifd.width}) or has non-positive size.")

        # Validate ``band`` against the selected IFD's sample count.
        # Without this, ``band=-1`` silently selects the last channel
        # via numpy negative indexing and ``band>=samples_per_pixel``
        # leaks a raw numpy ``IndexError`` with the internal slice
        # shape. Mirrors the dask path's pre-flight validator (see
        # ``read_geotiff_dask`` in ``__init__.py``), the GPU path, and
        # the HTTP path (``_read_cog_http`` above, as of issue #1695)
        # so all backends agree on the contract: 0-based non-negative
        # index only. See issue #1673.
        ifd_samples = ifd.samples_per_pixel
        if band is not None:
            # Reject ``bool`` and ``np.bool_`` before the range check.
            # ``isinstance(True, int)`` is True in Python and
            # ``True < ifd_samples`` evaluates as ``1``, so without this
            # guard ``band=True`` silently reads band 1 and ``band=False``
            # reads band 0. ``np.bool_`` is not a subclass of ``bool`` so it
            # needs its own check to match the VRT path's existing
            # rejection. See #1786.
            if isinstance(band, (bool, np.bool_)):
                raise ValueError(
                    f"band must be a non-negative int, got {band!r}")
            # Reject non-integer numeric types and anything else that
            # would slip past the bool guard. ``band=0.0`` passes
            # ``0 <= 0.0 < n_bands`` and silently selects band 0 on a
            # single-band file or raises a raw numpy ``IndexError`` from
            # deep in the read path on multi-band files. The VRT paths
            # already enforce this; mirror them here. See #1910.
            if not isinstance(band, (int, np.integer)):
                raise TypeError(
                    f"band must be a non-negative int, got {band!r}")
            if ifd_samples <= 1:
                if band != 0:
                    raise IndexError(
                        f"band={band} requested on a single-band file.")
            elif not 0 <= band < ifd_samples:
                raise IndexError(
                    f"band={band} out of range for {ifd_samples}-band file.")

        if ifd.is_tiled:
            arr = _read_tiles(ifd_data, ifd, ifd_header, dtype, window,
                              max_pixels=max_pixels)
        else:
            arr = _read_strips(ifd_data, ifd, ifd_header, dtype, window,
                               max_pixels=max_pixels)

        # Extract the requested band before reorienting so we work on a
        # smaller 2D array rather than reorienting a full multi-band cube
        # only to slice it afterwards.
        if arr.ndim == 3 and ifd.samples_per_pixel > 1 and band is not None:
            arr = arr[:, :, band]

        if orientation != 1:
            arr, geo_info = _apply_orientation_with_geo(
                arr, geo_info, orientation)

        if ifd.photometric == 0 and ifd.samples_per_pixel == 1:
            # The MinIsWhite inversion rewrites the original sentinel
            # value, so any downstream nodata-to-NaN mask must compare
            # against the inverted sentinel instead.  Stash the inverted
            # sentinel on geo_info as a private attribute so callers can
            # apply the mask post-inversion while keeping the original
            # sentinel on ``geo_info.nodata`` for the attrs round-trip
            # (issue #1809).
            inverted_nodata = _miniswhite_inverted_nodata(
                geo_info.nodata, ifd, arr.dtype)
            arr = _apply_photometric_miniswhite(arr, ifd)
            geo_info._mask_nodata = inverted_nodata
    finally:
        src.close()
        from ._sidecar import close_sidecar
        close_sidecar(sidecar)

    return arr, geo_info


# Backward-compatible alias for internal call sites that pre-date the
# rename to :func:`_read_to_array`. New code inside
# ``xrspatial.geotiff`` should import :func:`_read_to_array` directly.
# See issue #2138.
read_to_array = _read_to_array
