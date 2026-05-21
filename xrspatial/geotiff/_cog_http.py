"""COG-over-HTTP transport: bounded header prefetch, tiled/stripped range reads.

Extracted from :mod:`xrspatial.geotiff._reader` in PR-J of the GeoTIFF
refactor epic (issue #2258). Behaviour-neutral move: every helper kept
its signature, return contract, environment variables, and side
effects. The helpers stay private to :mod:`xrspatial.geotiff`; public
callers go through :func:`xrspatial.geotiff.open_geotiff` /
:func:`xrspatial.geotiff.read_geotiff_dask`.

Monkeypatch contract
--------------------
``test_http_cog_coalesce.py``, ``test_security.py``,
``test_http_stripped_window_max_pixels_issue_A_1842.py`` and others
patch ``_reader._HTTPSource`` and ``_reader._parse_cog_http_meta`` and
then call ``_reader._read_cog_http``. To honour that contract after
the move, :func:`_read_cog_http` resolves both names through the
``_reader`` module namespace at call time (``_reader._HTTPSource(url)``
and ``_reader._parse_cog_http_meta(...)``) rather than binding the
imported attributes at module-load time. ``_reader`` re-exports both
names from ``_sources`` / this module, so the indirection has the same
runtime cost as a single attribute lookup but keeps the existing
patches working unchanged.

Every ``from . import _reader`` in this module is lazy (inside a
function body) for the same reason: ``_reader.py`` already imports
the helpers defined here at module load, so a top-of-file
``from . import _reader`` would race the circular load and resolve
``_reader`` to a half-built module. By the time any of these
functions are *called*, both modules have fully loaded and the
attribute lookup succeeds.

The full list of names looked up via ``_reader`` (so that the
corresponding ``monkeypatch.setattr(_reader, '<name>', ...)`` in the
test suite keeps intercepting the call):

* ``_HTTPSource``
* ``_parse_cog_http_meta``
* ``_fetch_decode_cog_http_tiles``
* ``_decode_strip_or_tile``
* ``_apply_photometric_miniswhite``
* ``INITIAL_HTTP_HEADER_BYTES``
* ``MAX_HTTP_HEADER_BYTES``
"""
from __future__ import annotations

import math
import os as _os_module
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from ._compression import COMPRESSION_LERC
# ``_apply_photometric_miniswhite`` and ``_decode_strip_or_tile`` are
# resolved through ``_reader`` at call time (see ``_read_cog_http`` and
# the per-function notes below), so the names imported here are only
# used by helpers that the test fleet does not patch
# (``_apply_orientation_with_geo``, ``_miniswhite_inverted_nodata``,
# ``_read_strips``, ``_resolve_masked_fill``). The patched two are
# pulled back through ``_reader`` so monkeypatches against
# ``_reader._apply_photometric_miniswhite`` / ``_reader._decode_strip_or_tile``
# keep working after the helper move (PR-J / #2258).
from ._decode import (
    _PARALLEL_DECODE_PIXEL_THRESHOLD,
    _apply_orientation_with_geo,
    _miniswhite_inverted_nodata,
    _read_strips,
    _resolve_masked_fill,
)
from ._dtypes import SUB_BYTE_BPS, resolve_bits_per_sample, tiff_dtype_to_numpy
from ._geotags import (
    GeoInfo,
    extract_geo_info_with_overview_inheritance,
)
from ._header import (
    IFD,
    TIFFHeader,
    parse_all_ifds,
    parse_header,
    select_overview_ifd,
    validate_tile_layout,
)
from ._layout import (
    MAX_PIXELS_DEFAULT,
    _check_dimensions,
    _check_source_dimensions,
    _compute_full_image_byte_budget,
    _has_sparse,
    _ifd_required_extent,
    _sparse_fill_value,
)
from ._sources import (
    COALESCE_GAP_THRESHOLD_DEFAULT,
    _max_tile_bytes_from_env,
)
from ._validation import _validate_predictor_sample_format

if TYPE_CHECKING:
    # ``_HTTPSource`` is only referenced as a type annotation in this
    # module. Every runtime use goes through ``_reader._HTTPSource`` so
    # monkeypatches against the ``_reader`` namespace keep intercepting
    # the construction (PR-J / #2258). Keeping the import under
    # TYPE_CHECKING removes the misleading appearance of a live binding.
    from ._sources import _HTTPSource

#: Initial prefetch size for ``_parse_cog_http_meta``. Sized for the common
#: case (a single-IFD COG with modest GeoTIFF tags) so the fast path is a
#: single range GET.
INITIAL_HTTP_HEADER_BYTES = 16 * 1024

#: Upper bound on how far ``_parse_cog_http_meta`` will grow its prefetch
#: buffer before giving up. 4 MiB comfortably covers deep pyramids whose
#: IFD chains plus tag arrays (TileOffsets, GeoAsciiParams, GDAL_METADATA)
#: extend far past the initial fetch window. See issue #1718.
MAX_HTTP_HEADER_BYTES = 4 * 1024 * 1024


def _parse_cog_http_meta(
    source: _HTTPSource,
    overview_level: int | None = None,
    *,
    allow_rotated: bool = False,
    source_path: str | None = None,
    max_cloud_bytes: int | None = None,
    return_sidecar: bool = False,
):
    """Fetch + parse the leading IFDs of an HTTP COG once.

    The fast path is a single 16 KiB range GET. When the IFD chain or its
    out-of-line tag arrays extend past that window the buffer is doubled
    and reparsed until either the chain is fully resolved or the cap at
    :data:`MAX_HTTP_HEADER_BYTES` is reached. Real COGs whose pyramid
    metadata legitimately exceeds the cap need a different strategy
    (lazy per-IFD reads); the cap exists to bound a malformed-file blast
    radius rather than to constrain valid pyramids.

    Pulled out of :func:`_read_cog_http` so :func:`read_geotiff_dask`
    can parse metadata once per graph rather than once per chunk task
    (P5: each delayed task used to fire its own 16 KB header GET).

    ``source_path`` is the original URL or fsspec URI used to construct
    ``source``. When provided, the parser probes for a sibling ``.ovr``
    sidecar and merges its IFDs onto the pyramid list so an
    ``overview_level`` that lives in the sidecar resolves the same way
    the eager local/fsspec reader resolves it (issue #2239). Without
    it, the function preserves the prior base-only behaviour.

    When ``return_sidecar=True`` and the source has been probed, the
    return tuple grows an extra slot carrying ``(sidecar, route_path,
    used_sidecar)`` so callers (the dask backend, the eager HTTP path)
    can route per-chunk reads at the sidecar URL when the selected IFD
    came from the sidecar. ``route_path`` is the sidecar URL when
    ``used_sidecar`` is True, else the original ``source_path``.
    ``sidecar`` is the :class:`SidecarOverviews` instance the caller
    must close (or ``None`` when no sidecar was found).
    ``return_sidecar=True`` requires ``source_path``; the function
    asserts the precondition so a future caller cannot end up with
    ``route_path=None`` and then construct ``_HTTPSource(None)`` for
    per-chunk reads.

    Returns
    -------
    tuple
        ``(header, ifd, geo_info, header_bytes)`` for the default
        ``return_sidecar=False`` path. When ``return_sidecar=True``,
        the return is the same 4-tuple with a fifth element appended:
        ``(sidecar, route_path, used_sidecar)``. The caller owns
        ``sidecar`` and must close it; ``route_path`` is the URL/URI
        that per-chunk fetches should target; ``used_sidecar`` is
        ``True`` iff the selected IFD came from the sidecar.
    """
    if return_sidecar and source_path is None:
        # The 5-tuple contract guarantees ``route_path`` is a usable
        # path. Callers thread it straight into ``_HTTPSource`` /
        # ``_CloudSource``; ``None`` would crash later with a less
        # diagnosable error than this precondition does up front.
        raise TypeError(
            "_parse_cog_http_meta(return_sidecar=True) requires "
            "source_path; got source_path=None."
        )
    # Resolve the prefetch-window constants through ``_reader`` so a test
    # that ``monkeypatch.setattr(_reader, 'MAX_HTTP_HEADER_BYTES', ...)``
    # (the pattern used by ``test_http_meta_buffer_1718``) keeps shrinking
    # the cap after the helper moved into ``_cog_http`` (PR-J / #2258).
    # The capture is one-shot at function entry rather than per-iteration:
    # the original ``_reader._parse_cog_http_meta`` referenced the module
    # globals each loop pass, but the constants are not mutated mid-call
    # anywhere in the codebase or test suite, so a single read at the top
    # is behaviourally equivalent and avoids a per-iteration attribute
    # lookup on the ``_reader`` module.
    from . import _reader
    initial_bytes = _reader.INITIAL_HTTP_HEADER_BYTES
    max_bytes = _reader.MAX_HTTP_HEADER_BYTES
    fetch_size = initial_bytes
    header_bytes = source.read_range(0, fetch_size)
    header = parse_header(header_bytes)

    last_len = len(header_bytes)
    ifds: list[IFD] = []
    while True:
        try:
            ifds = parse_all_ifds(header_bytes, header)
            required = _ifd_required_extent(ifds)
            # Chain is fully resolved when every IFD parsed cleanly and
            # the tail next_ifd_offset is reachable within the buffer
            # (required == 0 means end-of-chain).
            if ifds and required <= len(header_bytes):
                break
        except ValueError:
            # parse_ifd raises when an out-of-line tag points past the
            # buffer. Treat it the same as a truncated chain: grow and
            # retry. If we are already at the cap and still failing, let
            # the next iteration's cap check raise a clear error.
            ifds = []

        if fetch_size >= max_bytes:
            raise ValueError(
                f"COG IFD chain or tag arrays extend past "
                f"MAX_HTTP_HEADER_BYTES={max_bytes} bytes; "
                f"the file may be malformed or its pyramid metadata is "
                f"unusually large for HTTP prefetch")
        fetch_size = min(fetch_size * 2, max_bytes)
        header_bytes = source.read_range(0, fetch_size)
        # Server returned the same number of bytes as last time: we have
        # hit EOF on the underlying file. No point growing further; if
        # the IFD chain still doesn't resolve, the file is truncated.
        if len(header_bytes) == last_len:
            try:
                ifds = parse_all_ifds(header_bytes, header)
            except ValueError:
                ifds = []
            break
        last_len = len(header_bytes)

    if len(ifds) == 0:
        raise ValueError("No IFDs found in COG")

    # Discover an external `.tif.ovr` sidecar and merge its IFDs onto the
    # pyramid list before ``select_overview_ifd`` runs. The eager
    # local/fsspec reader already does this at ``_reader.py``'s
    # ``read_to_array`` site; without the equivalent step here the dask
    # metadata path and the eager HTTP path picked a stale (too-shallow)
    # pyramid for remote GDAL external-overview files and diverged from
    # the eager local read. Issue #2239.
    sidecar = None
    sidecar_ifd_ids: set[int] = set()
    if source_path is not None:
        from ._sidecar import discover_remote_sidecar
        ifds, sidecar, sidecar_ifd_ids = discover_remote_sidecar(
            source_path, ifds, max_cloud_bytes=max_cloud_bytes,
        )

    ifd = select_overview_ifd(ifds, overview_level)
    used_sidecar = id(ifd) in sidecar_ifd_ids
    # When the requested IFD is an overview that lacks its own geokeys
    # (the common case for COG writers, including this package's
    # ``to_geotiff``), inherit and rescale the georef from the level-0
    # IFD so overview reads do not silently lose CRS / transform.
    # See issue #1640. We pass ``header_bytes`` even when the selected
    # IFD lives in the sidecar; that mirrors the eager local reader,
    # whose sidecar IFDs typically carry no out-of-line geokeys and
    # inherit from level-0 (which sits in the base buffer). #2239.
    geo_info = extract_geo_info_with_overview_inheritance(
        ifd, ifds, header_bytes, header.byte_order,
        allow_rotated=allow_rotated)
    if return_sidecar:
        route_path = sidecar.path if used_sidecar else source_path
        return (header, ifd, geo_info, header_bytes,
                (sidecar, route_path, used_sidecar))
    # Caller did not opt into sidecar metadata. Close the sidecar (if
    # any was loaded) before returning so the buffer does not leak --
    # the legacy return tuple has no slot to hand it back through.
    if sidecar is not None:
        from ._sidecar import close_sidecar
        close_sidecar(sidecar)
    return header, ifd, geo_info, header_bytes


def _read_cog_http(url: str, overview_level: int | None = None,
                   band: int | None = None,
                   max_pixels: int = MAX_PIXELS_DEFAULT,
                   window: tuple[int, int, int, int] | None = None,
                   *,
                   allow_rotated: bool = False,
                   ) -> tuple[np.ndarray, GeoInfo]:
    """Read a COG via HTTP range requests.

    Tile fetches run concurrently through a small thread pool so that the
    total wall time is bounded by the slowest tile request rather than
    ``num_tiles * RTT``. The pool size can be overridden with the
    ``XRSPATIAL_COG_HTTP_WORKERS`` environment variable (default 8).

    Parameters
    ----------
    url : str
        HTTP(S) URL to the COG file.
    overview_level : int or None
        Which overview to read (0 = full res, 1 = first overview, etc.).
    band : int
        Band index (0-based, for multi-band files).
    max_pixels : int
        Maximum allowed pixel count (width * height * samples).
    window : tuple or None
        ``(row_start, col_start, row_stop, col_stop)``. Forwarded to
        ``_fetch_decode_cog_http_tiles`` so HTTP reads honour the same
        windowed contract as the local-file path. See issue #1669.

    Returns
    -------
    (array, geo_info) tuple
    """
    # Resolve every patchable collaborator through the ``_reader`` module
    # namespace so existing tests that
    # ``monkeypatch.setattr(_reader, '_HTTPSource', ...)``,
    # ``monkeypatch.setattr(_reader, '_parse_cog_http_meta', ...)``,
    # ``monkeypatch.setattr(_reader, '_fetch_decode_cog_http_tiles', ...)``
    # or ``monkeypatch.setattr(_reader, '_apply_photometric_miniswhite', ...)``
    # keep working after the helper move (PR-J / #2258). ``_reader`` is
    # imported lazily inside the function to avoid the circular import
    # at module load (``_reader`` re-exports our helpers).
    from . import _reader
    source = _reader._HTTPSource(url)
    # Issue #1816: wrap everything after the ``_HTTPSource`` construction
    # in try/finally so ``source.close()`` runs even when header parsing,
    # validation, fetch/decode, or orientation/photometric post-processing
    # raises. ``_HTTPSource.close()`` is a no-op today, but a future
    # resource-holding source would leak on the error path without this.
    # ``close()`` is idempotent, so the explicit pre-raise ``source.close()``
    # calls in the validation blocks below stay as-is.
    sidecar = None
    try:
        # Issue #2239: discover the sibling ``.tif.ovr`` sidecar so HTTP
        # COG reads honour external overview pyramids the same way the
        # eager local/fsspec path does. ``source_path=url`` opts into
        # the discovery; ``return_sidecar=True`` hands back the loaded
        # sidecar so we can switch the byte-fetch source when the
        # requested overview lives there.
        (header, ifd, geo_info, header_bytes, sidecar_meta
         ) = _reader._parse_cog_http_meta(
            source, overview_level=overview_level,
            allow_rotated=allow_rotated,
            source_path=url,
            return_sidecar=True,
        )
        sidecar, route_path, used_sidecar = sidecar_meta
        # When the chosen IFD lives in the sidecar, tile/strip byte
        # offsets resolve against the sidecar URL, not the base URL.
        # Swap the ``_HTTPSource`` to the sidecar before any pixel
        # fetch goes out, so the range GETs land on the right object.
        if used_sidecar:
            source.close()
            source = _reader._HTTPSource(route_path)

        # Mirror the local-path orientation guard in ``read_to_array``: a
        # windowed read against a non-default Orientation tag (274) has
        # ambiguous semantics (does the window refer to file pixels or to
        # display pixels?) and the HTTP path does not yet implement
        # ``_apply_orientation``. Reject the combination here so HTTP and
        # local reads agree on the contract for oriented TIFFs instead of
        # silently returning a different region or pixel order. See PR
        # #1680 review feedback on issue #1669.
        if ifd.orientation != 1 and window is not None:
            source.close()
            raise ValueError(
                f"Orientation tag (274) is {ifd.orientation}; windowed reads "
                f"(window=...) and dask-chunked reads (chunks=...) are not "
                f"supported for non-default orientation. Read the full "
                f"array first, then slice."
            )

        # Validate ``window`` against the selected IFD's extent before the
        # tile fetch is built. Without this, the helper silently clamps an
        # out-of-bounds window and returns a smaller array, mismatching
        # ``open_geotiff``'s caller-built coord arrays. Mirrors the
        # local-path validator in ``read_to_array`` (#1634).
        if window is not None:
            w_r0, w_c0, w_r1, w_c1 = window
            if (w_r0 < 0 or w_c0 < 0
                    or w_r1 > ifd.height or w_c1 > ifd.width
                    or w_r0 >= w_r1 or w_c0 >= w_c1):
                source.close()
                raise ValueError(
                    f"window={window} is outside the source extent "
                    f"({ifd.height}x{ifd.width}) or has non-positive size.")

        # Validate ``band`` against the selected IFD's sample count before
        # the tile fetch. Without this, ``band=-1`` silently picks the last
        # channel via numpy negative indexing and ``band>=samples_per_pixel``
        # leaks a raw numpy ``IndexError``; on a single-band file ``band=N``
        # (N != 0) is dropped on the floor because the post-decode slice
        # below is gated on ``arr.ndim == 3 and samples_per_pixel > 1``.
        # Mirrors the local-path validator in ``read_to_array`` so all
        # backends agree on the contract: 0-based non-negative index only.
        # ``source.close()`` is called for symmetry with the success-path
        # teardown below; it is a no-op on ``_HTTPSource`` today (the
        # urllib3 ``PoolManager`` is shared module-level, not per-source)
        # but a future resource-holding source will need it. See issue #1695.
        if band is not None:
            # Reject ``bool`` (and ``np.bool_``) up front; ``isinstance(True, int)``
            # is True in Python so ``True < samples_per_pixel`` evaluates without
            # raising and silently reads band 1. ``np.bool_`` is not a subclass of
            # ``bool`` so it needs its own check to match the VRT path's
            # rejection. See #1786.
            if isinstance(band, (bool, np.bool_)):
                source.close()
                raise ValueError(
                    f"band must be a non-negative int, got {band!r}")
            # Reject non-integer numeric types and anything else that
            # would slip past the bool guard. ``band=0.0`` passes
            # ``0 <= 0.0 < n_bands`` and silently selects band 0 on a
            # single-band file or raises a raw numpy ``IndexError`` from
            # deep in the read path on multi-band files; ``band="0"``
            # fails the comparison with an opaque ``TypeError``. The VRT
            # paths already enforce this; mirror them here. See #1910.
            if not isinstance(band, (int, np.integer)):
                source.close()
                raise TypeError(
                    f"band must be a non-negative int, got {band!r}")
            if ifd.samples_per_pixel <= 1:
                if band != 0:
                    source.close()
                    raise IndexError(
                        f"band={band} requested on a single-band file.")
            elif not 0 <= band < ifd.samples_per_pixel:
                source.close()
                raise IndexError(
                    f"band={band} out of range for "
                    f"{ifd.samples_per_pixel}-band file.")

        arr = _reader._fetch_decode_cog_http_tiles(
            source, header, ifd, max_pixels=max_pixels, window=window)

        # Mirror the local-path band selection in ``read_to_array``: extract
        # the requested band only after the array is materialised so the
        # multi-band tile decode can populate every plane first. ``band``
        # outside the valid range raises ``IndexError`` the same as numpy.
        if arr.ndim == 3 and ifd.samples_per_pixel > 1 and band is not None:
            arr = arr[:, :, band]

        # Apply Orientation tag (274) so HTTP reads return the same pixel
        # order and transform as the local-file path. Only the full-read
        # branch reaches here; the windowed-read branch is rejected above
        # for non-default orientation. See issue #1717.
        if ifd.orientation != 1:
            arr, geo_info = _apply_orientation_with_geo(
                arr, geo_info, ifd.orientation)

        if ifd.photometric == 0 and ifd.samples_per_pixel == 1:
            # Stash the inverted sentinel on geo_info so the caller's
            # sentinel-to-NaN mask runs against the post-MinIsWhite value
            # while ``attrs['nodata']`` keeps the original sentinel for
            # round-trip on write (issue #1809).
            inverted_nodata = _miniswhite_inverted_nodata(
                geo_info.nodata, ifd, arr.dtype)
            geo_info._mask_nodata = inverted_nodata
        arr = _reader._apply_photometric_miniswhite(arr, ifd)
    finally:
        source.close()
        # Issue #2239: free the sidecar buffer when one was loaded.
        # ``close_sidecar`` is a no-op for ``None`` and for the HTTP
        # ``bytes`` buffer, but keeps the contract consistent with the
        # local/fsspec eager path that also routes through it.
        if sidecar is not None:
            from ._sidecar import close_sidecar
            close_sidecar(sidecar)

    return arr, geo_info


def _fetch_decode_cog_http_strips(
    source: _HTTPSource,
    header: TIFFHeader,
    ifd: IFD,
    dtype: np.dtype,
    bps: int,
    *,
    max_pixels: int = MAX_PIXELS_DEFAULT,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Fetch and decode the strips of a stripped TIFF over HTTP.

    Stripped HTTP companion to :func:`_fetch_decode_cog_http_tiles`. When
    *window* is given, only the strip byte-ranges that intersect the
    window are fetched + decoded; the result is sized to the (clamped)
    window rather than the full image, so a small window read of a
    multi-billion-pixel stripped file does not download the whole
    raster. Adjacent strip ranges are coalesced via
    :meth:`_HTTPSource.read_ranges_coalesced` the same way the tiled
    path does. ``max_pixels`` is applied to the *materialised* pixel
    count (window for windowed reads, full image otherwise) so a small
    caller cap on a tiny window passes a large source the same way the
    tiled branch does (#1823). When *window* is None, the function
    falls back to ``source.read_all()`` and dispatches to
    :func:`_read_strips`; the caller's ``max_pixels`` is threaded
    through so the full-image dim check honours the user's cap.
    See issues #1664 and #1823 for the safety contract this restores.
    """
    width = ifd.width
    height = ifd.height
    samples = ifd.samples_per_pixel
    # Source-IFD dim check (issue #2053). Mirror of the local-path
    # check in ``_read_strips`` so HTTP COG reads of a malformed
    # stripped file fail at the source rather than collapsing to an
    # empty post-clamp window. Tiled paths already get the equivalent
    # check from ``validate_tile_layout``.
    _check_source_dimensions(width, height, samples)
    compression = ifd.compression
    rps = ifd.rows_per_strip
    offsets = ifd.strip_offsets
    byte_counts = ifd.strip_byte_counts
    pred = ifd.predictor
    _validate_predictor_sample_format(pred, ifd.sample_format)
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS
    jpeg_tables = ifd.jpeg_tables
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, dtype)
                   if compression == COMPRESSION_LERC else None)
    planar = ifd.planar_config

    if offsets is None or byte_counts is None:
        raise ValueError("Missing strip offsets or byte counts")
    if rps is None or rps <= 0:
        raise ValueError(f"Invalid RowsPerStrip: {rps!r}")

    # Per-strip compressed-byte cap (#1664). A crafted ``StripByteCounts``
    # entry can request an unbounded HTTP Range GET or decompress a few
    # KiB into gigabytes. The cap applies to strips we actually fetch:
    # - Full-image path: validated inside ``_read_strips`` over every
    #   strip (full file is materialised regardless).
    # - Windowed path: validated inside the fetch-range loop below so a
    #   small window only fails on strips it intersects -- mirrors the
    #   tiled HTTP path's per-tile check (#1851).
    max_tile_bytes = _max_tile_bytes_from_env()

    # Full-image read: keep the legacy ``read_all`` + ``_read_strips``
    # path so anything _read_strips already validates (sparse strips,
    # strip-table truncation, LERC masked_fill, per-strip byte cap, etc.)
    # stays in one place. Just thread the caller's ``max_pixels`` through
    # so the dim check uses their cap instead of the default 1B.
    if window is None:
        _check_dimensions(width, height, samples, max_pixels)
        # Bound the HTTP body to the byte size implied by the TIFF strip
        # table. Without this cap, a tiny declared raster (which sails
        # past ``_check_dimensions``) can still pull a multi-gigabyte
        # body off the wire and into memory before ``_read_strips``
        # gets a chance to reject anything. The strip table tells us
        # the maximum legitimate byte offset; anything beyond that is
        # either a malformed file or a hostile server. Issue #2051.
        max_bytes = _compute_full_image_byte_budget(offsets, byte_counts)
        all_data = source.read_all(max_bytes=max_bytes)
        return _read_strips(all_data, ifd, header, dtype,
                            max_pixels=max_pixels)

    # Windowed read: fetch only the strips that intersect the window.
    r0, c0, r1, c1 = window
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(height, r1)
    c1 = min(width, c1)
    out_h = r1 - r0
    out_w = c1 - c0
    _check_dimensions(out_w, out_h, samples, max_pixels)

    strips_per_band = (height + rps - 1) // rps
    if planar == 2 and samples > 1:
        n_strips_expected = strips_per_band * samples
        if (len(offsets) < n_strips_expected
                or len(byte_counts) < n_strips_expected):
            raise ValueError(
                f"Strip table truncated for planar layout "
                f"(PlanarConfiguration=2): expected "
                f"{n_strips_expected} entries "
                f"({strips_per_band} strips x {samples} samples), got "
                f"offsets={len(offsets)}, byte_counts={len(byte_counts)}")
    else:
        n_strips_expected = strips_per_band
        if (len(offsets) < n_strips_expected
                or len(byte_counts) < n_strips_expected):
            raise ValueError(
                f"Strip table truncated: expected "
                f"{n_strips_expected} entries, got "
                f"offsets={len(offsets)}, byte_counts={len(byte_counts)}")

    first_strip = r0 // rps
    last_strip = min((r1 - 1) // rps, strips_per_band - 1)

    # Sparse strips (StripByteCounts == 0) must materialise as nodata or 0,
    # mirroring the local strip path. Detect sparsity over the *whole*
    # strip table so an empty strip outside the window does not change
    # the windowed allocation contract.
    sparse = _has_sparse(byte_counts)
    if sparse:
        fill = _sparse_fill_value(ifd, dtype)
        if samples > 1:
            result = np.full((out_h, out_w, samples), fill, dtype=dtype)
        else:
            result = np.full((out_h, out_w), fill, dtype=dtype)
    elif samples > 1:
        result = np.empty((out_h, out_w, samples), dtype=dtype)
    else:
        result = np.empty((out_h, out_w), dtype=dtype)

    # Pass 1: build the list of byte ranges + placements. Skip sparse
    # strips and any strips whose intersected row range is empty.
    band_count = samples if (planar == 2 and samples > 1) else 1
    strip_samples = 1 if band_count > 1 else samples
    fetch_ranges: list[tuple[int, int]] = []
    placements: list[tuple[int, int]] = []
    for band_idx in range(band_count):
        band_offset = band_idx * strips_per_band if band_count > 1 else 0
        for strip_idx in range(first_strip, last_strip + 1):
            global_idx = band_offset + strip_idx
            if global_idx >= len(offsets):
                continue
            bc = byte_counts[global_idx]
            if bc == 0:
                # Sparse strip: result is already pre-filled above.
                continue
            # Per-strip byte cap, scoped to strips the window actually
            # fetches (#1851). Mirrors the per-tile check in
            # ``_fetch_decode_cog_http_tiles`` so a window over a benign
            # strip is not rejected because some unrelated strip in the
            # file exceeds the cap.
            if bc > max_tile_bytes:
                raise ValueError(
                    f"TIFF strip {global_idx} declares "
                    f"StripByteCount={bc:,} bytes, which exceeds the "
                    f"per-strip safety cap of {max_tile_bytes:,} bytes. "
                    f"The file is malformed or attempting denial-of-service. "
                    f"Override via XRSPATIAL_COG_MAX_TILE_BYTES if this file "
                    f"is legitimate."
                )
            fetch_ranges.append((offsets[global_idx], bc))
            placements.append((band_idx, strip_idx))

    # Pass 2: fetch the strip bytes, coalescing adjacent ranges (mirrors
    # the tiled HTTP path; see #1823 / coalescing rationale on line ~2145).
    try:
        workers = max(1, int(
            _os_module.environ.get('XRSPATIAL_COG_HTTP_WORKERS', '8')))
    except ValueError:
        workers = 8
    try:
        gap = int(_os_module.environ.get(
            'XRSPATIAL_COG_COALESCE_GAP',
            str(COALESCE_GAP_THRESHOLD_DEFAULT)))
    except ValueError:
        gap = COALESCE_GAP_THRESHOLD_DEFAULT
    if fetch_ranges:
        strip_bytes_list = source.read_ranges_coalesced(
            fetch_ranges, max_workers=workers, gap_threshold=gap)
    else:
        strip_bytes_list = []

    # Pass 3: decode each strip and place its intersection with the window.
    #
    # Codec decode (deflate, zstd, LZW, ...) releases the GIL inside the C
    # extension, so threading the per-strip decode overlaps codec work
    # across cores. The local tile / strip paths and the HTTP tile path
    # use the same ``_PARALLEL_DECODE_PIXEL_THRESHOLD`` gate; mirror it
    # here so HTTP COG strip reads of wide windows benefit from the same
    # parallelism rather than serialising the decode after a parallel
    # fetch. The placement loop that copies pixels into ``result`` stays
    # serial to avoid contending writes to the output buffer. Issue #2100.
    n_decode_strips = len(strip_bytes_list)
    strip_pixel_count = width * rps
    decode_in_parallel = (
        n_decode_strips > 1
        and strip_pixel_count >= _PARALLEL_DECODE_PIXEL_THRESHOLD)

    # Resolve ``_decode_strip_or_tile`` through the ``_reader`` module so
    # ``monkeypatch.setattr(_reader, '_decode_strip_or_tile', ...)``
    # (used by the strip-decode parallelism tests in
    # ``test_parallel_strip_decode_2100``) still observes the patched
    # callable after the helper moved to ``_cog_http`` (PR-J / #2258).
    from . import _reader

    def _decode_http_strip(args):
        strip_idx, strip_data = args
        strip_row = strip_idx * rps
        strip_rows = min(rps, height - strip_row)
        if strip_rows <= 0:
            return None
        # Per-strip decoded-dimension cap (#1851). See note below.
        _check_dimensions(width, strip_rows, strip_samples,
                          MAX_PIXELS_DEFAULT)
        return _reader._decode_strip_or_tile(
            strip_data, compression, width, strip_rows, strip_samples,
            bps, bytes_per_sample, is_sub_byte, dtype, pred,
            byte_order=header.byte_order,
            jpeg_tables=jpeg_tables,
            masked_fill=masked_fill)

    decode_inputs = [(strip_idx, strip_data)
                     for (_band, strip_idx), strip_data
                     in zip(placements, strip_bytes_list)]

    if decode_in_parallel:
        n_decode_workers = min(n_decode_strips, _os_module.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_decode_workers) as pool:
            decoded_strips = list(pool.map(_decode_http_strip, decode_inputs))
    else:
        decoded_strips = [_decode_http_strip(item) for item in decode_inputs]

    for (band_idx, strip_idx), strip_pixels in zip(placements, decoded_strips):
        if strip_pixels is None:
            continue
        strip_row = strip_idx * rps
        strip_rows = min(rps, height - strip_row)
        if strip_rows <= 0:
            continue

        # Per-strip decoded-dimension cap (#1851). Mirrors the per-tile
        # ``_check_dimensions(tw, th, samples, MAX_PIXELS_DEFAULT)`` in
        # the tiled HTTP path: a tiny window intersecting an oversized
        # strip would otherwise force ``_decode_strip_or_tile`` to
        # allocate ``width * strip_rows * strip_samples`` bytes before
        # the window clip. Use ``MAX_PIXELS_DEFAULT`` rather than the
        # caller's ``max_pixels`` so a small output-window budget does
        # not reject normal strip sizes.
        # The decoded-dimension cap fired inside ``_decode_http_strip``.

        src_r0 = max(r0 - strip_row, 0)
        src_r1 = min(r1 - strip_row, strip_rows)
        dst_r0 = max(strip_row - r0, 0)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        if dst_r1 <= dst_r0:
            continue

        if band_count > 1:
            # Planar=2 strip holds one band; place into the per-band slot.
            result[dst_r0:dst_r1, :, band_idx] = (
                strip_pixels[src_r0:src_r1, c0:c1])
        else:
            result[dst_r0:dst_r1] = strip_pixels[src_r0:src_r1, c0:c1]

    return result


def _fetch_decode_cog_http_tiles(
    source: _HTTPSource,
    header: TIFFHeader,
    ifd: IFD,
    *,
    max_pixels: int = MAX_PIXELS_DEFAULT,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Fetch and decode the tiles of a tiled COG over HTTP.

    Pulled out of :func:`_read_cog_http` so that callers with
    pre-parsed metadata (notably :func:`read_geotiff_dask`) can reuse a
    single IFD parse across many tile-fetch calls. When *window* is
    given, only tiles intersecting the window are fetched + decoded;
    the result is sized to the (clamped) window rather than the full
    image. Coalescing of adjacent ranges still applies.
    """
    bps = resolve_bits_per_sample(ifd.bits_per_sample)
    dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
    if not ifd.is_tiled:
        return _fetch_decode_cog_http_strips(
            source, header, ifd, dtype, bps,
            max_pixels=max_pixels, window=window,
        )

    width = ifd.width
    height = ifd.height
    tw = ifd.tile_width
    th = ifd.tile_height
    samples = ifd.samples_per_pixel
    planar = ifd.planar_config
    compression = ifd.compression
    pred = ifd.predictor
    _validate_predictor_sample_format(pred, ifd.sample_format)
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS
    jpeg_tables = ifd.jpeg_tables
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, dtype)
                   if compression == COMPRESSION_LERC else None)

    offsets = ifd.tile_offsets
    byte_counts = ifd.tile_byte_counts

    if tw <= 0 or th <= 0:
        raise ValueError(
            f"Invalid tile dimensions: TileWidth={tw}, TileLength={th}")

    tiles_across = math.ceil(width / tw)
    tiles_down = math.ceil(height / th)

    # Cap the *materialised* pixel count, not the declared image size.
    # A windowed HTTP read of a multi-billion-pixel COG only allocates
    # the window, so capping the full image would reject legitimate
    # tiled reads. The full-image cap still applies for whole-file
    # reads (window is None). The per-tile dim check below guards the
    # TIFF header against absurd ``TileWidth`` / ``TileLength`` values
    # (e.g. 2**31) and uses ``MAX_PIXELS_DEFAULT`` so a caller's small
    # ``max_pixels`` -- intended as an output-window budget -- does not
    # reject normal 256x256 tiles. See #1823.
    if window is None:
        _check_dimensions(width, height, samples, max_pixels)
    _check_dimensions(tw, th, samples, MAX_PIXELS_DEFAULT)

    # Reject malformed TIFFs whose declared tile grid exceeds the supplied
    # TileOffsets length. See issue #1219.
    validate_tile_layout(ifd)

    if window is None:
        r0_out, c0_out, r1_out, c1_out = 0, 0, height, width
    else:
        r0_out, c0_out, r1_out, c1_out = window
        r0_out = max(0, r0_out)
        c0_out = max(0, c0_out)
        r1_out = min(height, r1_out)
        c1_out = min(width, c1_out)

    out_h = r1_out - r0_out
    out_w = c1_out - c0_out
    _check_dimensions(out_w, out_h, samples, max_pixels)

    # ``PlanarConfiguration=2`` stores one tile sequence per band,
    # concatenated in TileOffsets. ``tiles_per_band`` selects the right
    # slab when computing ``tile_idx``; ``band_count == 1`` for chunky
    # files keeps the original single-loop fetch behaviour. Mirrors the
    # local ``_read_tiles`` path (#1669).
    band_count = samples if (planar == 2 and samples > 1) else 1
    tiles_per_band = tiles_across * tiles_down
    # Per-tile sample count: planar=2 tiles hold one band each, planar=1
    # tiles interleave ``samples`` components per pixel.
    tile_samples = 1 if band_count > 1 else samples

    # Sparse tiles (TileByteCounts == 0) need to land on the file's nodata
    # value (or 0 if unset) rather than uninitialised memory.  Detect them
    # up front so the result buffer is pre-filled before tile placement.
    sparse = _has_sparse(byte_counts)
    if sparse:
        fill = _sparse_fill_value(ifd, dtype)
        if samples > 1:
            result = np.full((out_h, out_w, samples), fill, dtype=dtype)
        else:
            result = np.full((out_h, out_w), fill, dtype=dtype)
    elif samples > 1:
        result = np.empty((out_h, out_w, samples), dtype=dtype)
    else:
        result = np.empty((out_h, out_w), dtype=dtype)

    tile_row_start = r0_out // th
    tile_row_end = min(math.ceil(r1_out / th), tiles_down)
    tile_col_start = c0_out // tw
    tile_col_end = min(math.ceil(c1_out / tw), tiles_across)

    # Pass 1: collect every tile's range and where it lands in the output.
    # Empty tiles (byte_count == 0) and any tile_idx beyond the offsets
    # array are skipped here so the fetch list stays exactly aligned with
    # the placements list.
    #
    # Each tile's compressed size is checked against the cap returned by
    # _max_tile_bytes_from_env() (default MAX_TILE_BYTES_DEFAULT, 256 MiB)
    # before the fetch list is built. A crafted COG can claim arbitrarily
    # large TileByteCounts; without this guard the HTTP layer would issue
    # a Range request sized by the attacker's value (issue #1536). The cap
    # is overridable via XRSPATIAL_COG_MAX_TILE_BYTES. The local-mmap path
    # applies the same cap in _read_tiles / _read_strips (issue #1664).
    max_tile_bytes = _max_tile_bytes_from_env()
    fetch_ranges: list[tuple[int, int]] = []
    # Placement record: (band_idx, tr, tc). band_idx is 0 for chunky
    # files; for planar=2 it indicates which sample axis slot the
    # decoded tile fills.
    placements: list[tuple[int, int, int]] = []
    for band_idx in range(band_count):
        band_tile_offset = (band_idx * tiles_per_band
                            if band_count > 1 else 0)
        for tr in range(tile_row_start, tile_row_end):
            for tc in range(tile_col_start, tile_col_end):
                tile_idx = band_tile_offset + tr * tiles_across + tc
                if tile_idx >= len(offsets):
                    continue
                off = offsets[tile_idx]
                bc = byte_counts[tile_idx]
                if bc == 0:
                    continue
                if bc > max_tile_bytes:
                    raise ValueError(
                        f"TIFF tile {tile_idx} declares "
                        f"TileByteCount={bc:,} bytes, which exceeds the HTTP "
                        f"COG safety cap of {max_tile_bytes:,} bytes. The "
                        f"file is malformed or attempting denial-of-service. "
                        f"Override via XRSPATIAL_COG_MAX_TILE_BYTES if this "
                        f"file is legitimate."
                    )
                fetch_ranges.append((off, bc))
                placements.append((band_idx, tr, tc))

    # Pass 2: fetch all tile bytes in parallel. Worker pool size is tunable
    # via XRSPATIAL_COG_HTTP_WORKERS so users on very slow links can dial
    # it up without code changes.
    #
    # COG tile offsets are sorted and usually back-to-back, so we coalesce
    # adjacent ranges into fewer larger GETs (P2). The 1 MB gap threshold
    # tolerates small interleaved metadata between tiles without dragging
    # in unrelated overview data. Set XRSPATIAL_COG_COALESCE_GAP=-1 to
    # disable merging (one GET per tile, the legacy behaviour).
    try:
        workers = max(1, int(_os_module.environ.get('XRSPATIAL_COG_HTTP_WORKERS', '8')))
    except ValueError:
        workers = 8
    try:
        gap = int(_os_module.environ.get(
            'XRSPATIAL_COG_COALESCE_GAP',
            str(COALESCE_GAP_THRESHOLD_DEFAULT)))
    except ValueError:
        gap = COALESCE_GAP_THRESHOLD_DEFAULT
    tile_bytes_list = source.read_ranges_coalesced(
        fetch_ranges, max_workers=workers, gap_threshold=gap)

    # Pass 3: decode each tile and place it (clipped to the window).
    #
    # Codec decode (deflate, zstd, LZW, ...) releases the GIL inside the
    # C extension, so a thread pool over the per-tile decode actually
    # overlaps codec work across cores. The local-file path in
    # ``_read_tiles`` uses the same pattern with a 64K-pixel threshold to
    # skip the pool-startup cost on small tiles; mirror that gate here so
    # HTTP COG reads of wide windows benefit from the same parallelism
    # rather than serialising the decode after a parallel fetch. The
    # placement loop that copies pixels into ``result`` stays serial to
    # avoid contending writes to the output buffer.
    n_decode_tiles = len(placements)
    decode_in_parallel = (
        n_decode_tiles > 1 and tw * th >= _PARALLEL_DECODE_PIXEL_THRESHOLD)

    # Resolve ``_decode_strip_or_tile`` through the ``_reader`` module so
    # ``monkeypatch.setattr(_reader, '_decode_strip_or_tile', ...)``
    # (used by ``test_cog_http_parallel_decode_2026_05_15``) still
    # observes the patched callable after the helper moved to
    # ``_cog_http`` (PR-J / #2258).
    from . import _reader

    def _decode_one(tile_data):
        return _reader._decode_strip_or_tile(
            tile_data, compression, tw, th, tile_samples,
            bps, bytes_per_sample, is_sub_byte, dtype, pred,
            byte_order=header.byte_order,
            jpeg_tables=jpeg_tables,
            masked_fill=masked_fill)

    if decode_in_parallel:
        # Re-import ``ThreadPoolExecutor`` inside the branch (mirrors the
        # original ``_reader._fetch_decode_cog_http_tiles``) so tests
        # that ``monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', ...)``
        # observe the patched class on every call.
        from concurrent.futures import ThreadPoolExecutor
        n_decode_workers = min(n_decode_tiles, _os_module.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_decode_workers) as pool:
            decoded_tiles = list(pool.map(_decode_one, tile_bytes_list))
    else:
        decoded_tiles = [_decode_one(tile_data) for tile_data in tile_bytes_list]

    for (band_idx, tr, tc), tile_pixels in zip(placements, decoded_tiles):
        # Tile position in image coordinates.
        ty0 = tr * th
        tx0 = tc * tw
        ty1 = ty0 + th
        tx1 = tx0 + tw

        # Intersect with the requested window.
        iy0 = max(ty0, r0_out)
        ix0 = max(tx0, c0_out)
        iy1 = min(ty1, r1_out)
        ix1 = min(tx1, c1_out)
        if iy1 <= iy0 or ix1 <= ix0:
            continue

        # Source slice within the decoded tile pixels.
        sy0 = iy0 - ty0
        sx0 = ix0 - tx0
        sy1 = sy0 + (iy1 - iy0)
        sx1 = sx0 + (ix1 - ix0)

        # Destination slice within the output buffer.
        dy0 = iy0 - r0_out
        dx0 = ix0 - c0_out
        dy1 = iy1 - r0_out
        dx1 = ix1 - c0_out

        if band_count > 1:
            # Planar=2 tile holds one band; place into the per-band slot
            # of the (out_h, out_w, samples) result. ``tile_pixels`` from
            # ``_decode_strip_or_tile`` with ``samples=1`` is 2D.
            result[dy0:dy1, dx0:dx1, band_idx] = tile_pixels[sy0:sy1, sx0:sx1]
        else:
            result[dy0:dy1, dx0:dx1] = tile_pixels[sy0:sy1, sx0:sx1]

    return result
