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

import math
import os as _os_module
from concurrent.futures import ThreadPoolExecutor

import numpy as np
# ``urllib3`` is kept as a top-level import here even though the HTTP
# source moved to ``_sources`` in #2228. ``test_http_no_stdlib_fallback_2050``
# asserts the reader module carries a module-level urllib3 reference so a
# build that silently drops the dependency cannot ship. The HTTP code path
# itself uses the ``_sources`` import; this binding is purely the
# "urllib3 is a hard install dep" guard.
import urllib3  # noqa: F401

from ._compression import (
    COMPRESSION_LERC,
    COMPRESSION_NONE,
    decompress,
    fp_predictor_decode,
    lerc_decompress_with_mask,
    predictor_decode,
    unpack_bits,
)
from ._dtypes import SUB_BYTE_BPS, resolve_bits_per_sample, tiff_dtype_to_numpy
from ._geotags import (
    GeoInfo,
    GeoTransform,
    RASTER_PIXEL_IS_POINT,
    extract_geo_info,
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
from ._validation import _validate_predictor_sample_format

# ---------------------------------------------------------------------------
# Allocation guard: reject TIFF dimensions that would exhaust memory
# ---------------------------------------------------------------------------

#: Default maximum total pixel count (width * height * samples).
#: ~1 billion pixels, which is ~4 GB for float32 single-band.
#: Override per-call via the ``max_pixels`` keyword argument.
MAX_PIXELS_DEFAULT = 1_000_000_000


class PixelSafetyLimitError(ValueError):
    """Raised when a requested TIFF allocation exceeds max_pixels."""


def _check_dimensions(width, height, samples, max_pixels):
    """Raise PixelSafetyLimitError if the request exceeds *max_pixels*."""
    total = width * height * samples
    if total > max_pixels:
        raise PixelSafetyLimitError(
            f"TIFF image dimensions ({width} x {height} x {samples} = "
            f"{total:,} pixels) exceed the safety limit of "
            f"{max_pixels:,} pixels.  Pass a larger max_pixels value to "
            f"read_to_array() if this file is legitimate."
        )


def _check_source_dimensions(width, height, samples):
    """Validate the source IFD dimensions of a TIFF before any windowing.

    Companion to :func:`_check_dimensions`, which only enforces the
    upper bound. The stripped read paths read ``width``,  ``height``,
    and ``samples_per_pixel`` straight off the IFD and then clamp the
    output window to those values, so a malformed file with
    ``ImageWidth = 0`` (or a negative value, which would parse as a
    huge unsigned int but can also surface via signed-cast errors)
    would produce an empty array silently. The tiled paths are already
    protected by :func:`validate_tile_layout` in ``_header.py``; this
    helper closes the same gap for the stripped path. Issue #2053.
    """
    if width <= 0 or height <= 0 or samples <= 0:
        raise ValueError(
            f"Invalid TIFF dimensions: ImageWidth={width}, "
            f"ImageLength={height}, SamplesPerPixel={samples} "
            f"(all must be > 0)"
        )


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
from ._sources import (  # noqa: F401
    COALESCE_GAP_THRESHOLD_DEFAULT,
    MAX_CLOUD_BYTES_DEFAULT,
    MAX_TILE_BYTES_DEFAULT,
    CloudSizeLimitError,
    UnsafeURLError,
    _BytesIOSource,
    _CloudSource,
    _CLOUD_SCHEMES,
    _FileSource,
    _HTTPSource,
    _HTTP_ALLOWED_SCHEMES,
    _HTTP_CONNECT_TIMEOUT_DEFAULT,
    _HTTP_MAX_REDIRECTS,
    _HTTP_READ_TIMEOUT_DEFAULT,
    _MAX_CLOUD_BYTES_SENTINEL,
    _MmapCache,
    _DEFAULT_MMAP_CACHE_SIZE,
    _build_pinned_connection_classes,
    _coerce_path,
    _get_http_pool,
    _get_pinned_conn_classes,
    _http_allow_private_hosts,
    _http_connect_timeout,
    _http_read_timeout,
    _http_timeout_from_env,
    _ip_is_private,
    _is_file_like,
    _is_fsspec_uri,
    _make_pinned_pool,
    _max_tile_bytes_from_env,
    _mmap_cache,
    _mmap_cache_size_from_env,
    _open_source,
    _resolve_max_cloud_bytes,
    _validate_http_url,
    coalesce_ranges,
    split_coalesced_bytes,
)

#: Per-tile pixel count at and above which the local and HTTP tile-read paths
#: spread codec decode across a ``ThreadPoolExecutor``. Below this, pool
#: startup costs outweigh the parallelism win (issue #1551). Bound is inclusive
#: so the default ``tile_size=256`` (256*256 == 64*1024) lands on the parallel
#: path. Used by both ``_read_tiles`` and ``_fetch_decode_cog_http_tiles``.
_PARALLEL_DECODE_PIXEL_THRESHOLD = 64 * 1024


def _apply_predictor(chunk: np.ndarray, pred: int, width: int,
                     height: int, bytes_per_sample: int,
                     samples: int = 1,
                     byte_order: str = '<') -> np.ndarray:
    """Apply the appropriate predictor decode to decompressed data.

    ``width``, ``height``, ``bytes_per_sample``, and ``samples`` describe
    the raw pixel layout before predictor inversion: ``width * samples``
    samples per row, each ``bytes_per_sample`` bytes wide.

    Predictor=2 (horizontal differencing) operates at the *sample* level
    per TIFF Technical Note (libtiff/GDAL convention): the difference is
    taken between adjacent same-component samples in the sample's
    natural bit width, with stride equal to ``samples`` samples.  A
    byte-wise implementation drops the inter-byte carry for multi-byte
    samples and produces wrong values.

    Predictor=3 (floating-point) byte-swizzles each row into
    ``bytes_per_sample`` interleaved lanes of length ``width * samples``,
    per TIFF Technical Note 3.  The un-transpose stage has to put the
    MSB lane at the file's high-order byte position, which differs for
    big- vs little-endian files; ``byte_order`` carries that.
    """
    if pred == 2:
        return predictor_decode(chunk, width, height,
                                bytes_per_sample, samples=samples,
                                byte_order=byte_order)
    elif pred == 3:
        return fp_predictor_decode(chunk, width * samples, height,
                                   bytes_per_sample,
                                   big_endian=(byte_order == '>'))
    return chunk


def _packed_byte_count(pixel_count: int, bps: int) -> int:
    """Compute the number of packed bytes for sub-byte bit depths."""
    return (pixel_count * bps + 7) // 8


def _int_nodata_in_range(nodata_int: int, dtype: np.dtype) -> bool:
    """Return True iff *nodata_int* is representable as *dtype*.

    Used to gate ``dtype.type(int(...))`` casts that would otherwise raise
    ``OverflowError`` on real-world files that pair an unsigned dtype with
    a negative GDAL_NODATA sentinel (e.g. uint16 + ``-9999``). When the
    sentinel cannot be represented, the file's pixels can never match it,
    so the caller should treat the sentinel as a no-op for value matching
    (still surfacing it via ``attrs['nodata']`` so write round-trips
    preserve the original tag).
    """
    if dtype.kind not in ('u', 'i'):
        return False
    info = np.iinfo(dtype)
    return info.min <= nodata_int <= info.max


def _resolve_masked_fill(nodata_str: str | None, dtype: np.dtype):
    """Resolve the value to use when restoring LERC-masked pixels.

    Mirrors :func:`_sparse_fill_value` but defaults to NaN for floating
    dtypes when the file does not declare a nodata sentinel.  Float
    rasters with no GDAL_NODATA tag still benefit from NaN propagation
    because LERC's zero fill would silently masquerade as a real
    measurement at z == 0.

    Note: integer dtypes with no GDAL_NODATA tag fall back to ``0``,
    which is the same value LERC zero-fills masked pixels with -- in
    that case the mask application is intentionally a no-op.  We avoid
    inventing an integer sentinel (e.g. iinfo.max) because doing so
    would silently change pixel values for files that never declared
    one, breaking downstream consumers that key off the original data.

    Out-of-range integer sentinels (e.g. ``uint16`` paired with
    ``GDAL_NODATA="-9999"``, common on legacy GDAL files) cannot be
    represented in the file dtype and so cannot match any decoded
    pixel; we fall back to ``0`` rather than raising ``OverflowError``
    on the dtype cast.
    """
    if nodata_str is not None:
        # Try ``int`` first so 64-bit sentinels survive without the
        # float64 round-trip; fall back to ``float`` for NaN / Inf /
        # scientific notation / fractional values.  See issue #1847.
        from ._geotags import _parse_nodata_str as _parse_nd
        parsed = _parse_nd(nodata_str)
        if parsed is not None:
            if dtype.kind == 'f':
                return dtype.type(parsed)
            if isinstance(parsed, int):
                if _int_nodata_in_range(parsed, dtype):
                    return dtype.type(parsed)
            elif not math.isnan(parsed) and not math.isinf(parsed):
                if float(parsed).is_integer():
                    nodata_int = int(parsed)
                    if _int_nodata_in_range(nodata_int, dtype):
                        return dtype.type(nodata_int)
    if dtype.kind == 'f':
        return dtype.type(np.nan)
    return dtype.type(0)


def _decode_strip_or_tile(data_slice, compression, width, height, samples,
                          bps, bytes_per_sample, is_sub_byte, dtype, pred,
                          byte_order='<', jpeg_tables=None,
                          masked_fill=None):
    """Decompress, apply predictor, unpack sub-byte, and reshape a strip/tile.

    Parameters
    ----------
    byte_order : str
        '<' for little-endian, '>' for big-endian.  When the file byte
        order differs from the system's native order, pixel data is
        byte-swapped after decompression.
    jpeg_tables : bytes or None
        Raw bytes of the file's JPEGTables tag (347), or None if the file
        doesn't have one. GDAL-style tiled JPEG TIFFs store DQT/DHT tables
        once in this tag and each tile is a JPEG fragment that depends on
        them; the JPEG decoder splices the tables in before handing the
        tile to libjpeg. Ignored for non-JPEG compressions.
    masked_fill : scalar or None
        Fill value written into pixels that the LERC valid-mask flags as
        invalid.  Only consulted for ``compression == COMPRESSION_LERC``
        when the decoder returns a non-trivial mask; ignored for every
        other codec.  Callers should compute it once per IFD via
        :func:`_resolve_masked_fill` (typically NaN for float dtypes or
        the parsed ``GDAL_NODATA`` sentinel).  When ``None``, masked
        pixels are left at LERC's zero fill.

    Returns an array shaped (height, width) or (height, width, samples).
    """
    pixel_count = width * height * samples
    if is_sub_byte:
        expected = _packed_byte_count(pixel_count, bps)
    else:
        expected = pixel_count * bytes_per_sample

    lerc_mask = None
    if compression == COMPRESSION_LERC:
        # LERC needs special handling: lerc.decode also returns a
        # valid-mask which the generic decompress() dispatcher discards.
        # We capture it here so masked pixels can be restored to nodata
        # below, instead of leaking LERC's zero fill into the output.
        # Forward ``expected`` so the wrapper rejects bombs at the
        # blob-header level rather than after the full buffer is
        # materialised (issue #1625).
        decoded_bytes, lerc_mask = lerc_decompress_with_mask(
            data_slice, expected_size=expected)
        chunk = np.frombuffer(decoded_bytes, dtype=np.uint8)
    else:
        chunk = decompress(data_slice, compression, expected,
                           width=width, height=height, samples=samples,
                           jpeg_tables=jpeg_tables)

    # Validate the decompressed byte count.  A truncated deflate stream or a
    # buggy compressor can produce fewer or more bytes than expected.  Without
    # this check the downstream reshape raises an opaque "cannot reshape array
    # of size N into shape (h, w)" that hides which tile/strip broke.  Edge
    # tiles in a valid TIFF still decompress to the full tile_height x
    # tile_width (the caller slices the top-left region), so this only fires
    # on genuine corruption.
    if chunk.size != expected:
        raise ValueError(
            f"Decompressed tile/strip size mismatch: expected {expected} "
            f"bytes for a {width} x {height} x {samples} block "
            f"(bps={bps}, compression={compression}), got {chunk.size}. "
            f"The TIFF data is likely truncated or corrupt."
        )

    if pred in (2, 3) and not is_sub_byte:
        if not chunk.flags.writeable:
            chunk = chunk.copy()
        chunk = _apply_predictor(chunk, pred, width, height,
                                 bytes_per_sample, samples=samples,
                                 byte_order=byte_order)

    if is_sub_byte:
        pixels = unpack_bits(chunk, bps, pixel_count)
    else:
        # Use the file's byte order for the view, then convert to native.
        # The view dtype must match the on-disk sample width: float16
        # files (bps=16 + SampleFormat=3) are auto-promoted to float32
        # for the user-visible array, but the raw bytes have to be
        # viewed as float16 first then cast (#1941). Detect the
        # promotion via the bps-vs-dtype.itemsize mismatch so the
        # surrounding pipeline stays unchanged for byte-equal cases.
        if dtype.itemsize * 8 != bps and bps == 16 and dtype.kind == 'f':
            storage_dtype = np.dtype('float16').newbyteorder(byte_order)
            pixels = chunk.view(storage_dtype).astype(dtype)
        else:
            file_dtype = dtype.newbyteorder(byte_order)
            pixels = chunk.view(file_dtype)
            if file_dtype.byteorder not in ('=', '|', _NATIVE_ORDER):
                pixels = pixels.astype(dtype)

    if samples > 1:
        out = pixels.reshape(height, width, samples)
    else:
        out = pixels.reshape(height, width)

    # Restore nodata in positions LERC flagged as invalid.  LERC
    # zero-fills masked pixels in the data array, which would otherwise
    # be indistinguishable from real zero readings downstream.
    if lerc_mask is not None and masked_fill is not None:
        mask_arr = np.asarray(lerc_mask)
        if mask_arr.ndim == 2 and out.ndim == 3:
            mask_arr = mask_arr[..., None]
        invalid = np.broadcast_to(mask_arr == 0, out.shape)
        if invalid.any():
            if not out.flags.writeable:
                out = out.copy()
            np.putmask(out, invalid, masked_fill)
    return out


import sys as _sys
_NATIVE_ORDER = '<' if _sys.byteorder == 'little' else '>'


def _sparse_fill_value(ifd: IFD, dtype: np.dtype):
    """Resolve the fill value for sparse tiles/strips.

    A sparse TIFF entry has TileByteCounts/StripByteCounts == 0 (and
    typically the matching Offset == 0). GDAL emits these for SPARSE_OK
    files where blocks containing only the nodata value are omitted.
    The reader is expected to materialise such blocks as nodata, or
    zero when nodata is unset (the default per the GDAL convention).
    """
    nodata_str = ifd.nodata_str
    if nodata_str is not None:
        # Try ``int`` first so 64-bit sentinels survive without the
        # float64 round-trip; fall back to ``float`` for NaN / Inf /
        # scientific notation / fractional values.  See issue #1847.
        from ._geotags import _parse_nodata_str as _parse_nd
        parsed = _parse_nd(nodata_str)
        if parsed is not None:
            if dtype.kind == 'f':
                return dtype.type(parsed)
            if isinstance(parsed, int):
                if _int_nodata_in_range(parsed, dtype):
                    return dtype.type(parsed)
            elif not math.isnan(parsed) and not math.isinf(parsed):
                if float(parsed).is_integer():
                    nodata_int = int(parsed)
                    if _int_nodata_in_range(nodata_int, dtype):
                        return dtype.type(nodata_int)
    return dtype.type(0)


def _has_sparse(byte_counts) -> bool:
    """Return True if any tile/strip is empty (byte_count == 0)."""
    if byte_counts is None:
        return False
    for bc in byte_counts:
        if bc == 0:
            return True
    return False


#: Slack added to the strip-table byte budget for the TIFF header,
#: trailing IFD chain, ExifIFD, GeoKey directory, GDAL_METADATA, and any
#: ICC profile or XMP packet. 4 MiB is comfortable for real-world COGs
#: (the prefetch path already tolerates up to ``MAX_HTTP_HEADER_BYTES``
#: of header bytes) while still bounding the body away from gigabyte
#: scale. Issue #2051.
_FULL_IMAGE_BUDGET_HEADER_SLACK = 4 * 1024 * 1024


def _compute_full_image_byte_budget(offsets, byte_counts) -> int:
    """Compute an upper bound on the legitimate HTTP body size for a stripped TIFF.

    A stripped TIFF body is laid out as: [TIFF header + IFDs + tag value
    arrays] followed by strip payloads at the offsets listed in
    ``StripOffsets``. The largest byte index any strip references is
    ``max(offset + byte_count)`` across the strip table; the body cannot
    legitimately extend past that point plus a small tail for trailing
    metadata. We add :data:`_FULL_IMAGE_BUDGET_HEADER_SLACK` to cover the
    header prologue (which lives at offset 0) and any tags that follow
    the last strip. The cap is loose by design -- it exists to reject
    bodies that are orders of magnitude larger than the file claims to
    be, not to second-guess legitimate layouts.

    If the strip table is missing or empty (sparse-only, malformed),
    fall back to the per-strip safety cap so the read is still bounded.
    Issue #2051.
    """
    fallback = _max_tile_bytes_from_env() + _FULL_IMAGE_BUDGET_HEADER_SLACK
    if not offsets or not byte_counts:
        return fallback
    max_end = 0
    for off, bc in zip(offsets, byte_counts):
        try:
            end = int(off) + int(bc)
        except (TypeError, ValueError):
            continue
        if end > max_end:
            max_end = end
    if max_end <= 0:
        return fallback
    return max_end + _FULL_IMAGE_BUDGET_HEADER_SLACK


# ---------------------------------------------------------------------------
# Strip reader
# ---------------------------------------------------------------------------

def _read_strips(data: bytes, ifd: IFD, header: TIFFHeader,
                 dtype: np.dtype, window=None,
                 max_pixels: int = MAX_PIXELS_DEFAULT) -> np.ndarray:
    """Read a strip-organized TIFF image.

    Parameters
    ----------
    data : bytes
        Full file data.
    ifd : IFD
        Parsed IFD for this image.
    header : TIFFHeader
        File header.
    dtype : np.dtype
        Output pixel dtype.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop) or None for full image.
    max_pixels : int
        Maximum allowed pixel count (width * height * samples).

    Returns
    -------
    np.ndarray with shape (height, width) or windowed subset.
    """
    width = ifd.width
    height = ifd.height
    samples = ifd.samples_per_pixel
    # Source-IFD dim check (issue #2053). The tiled path is already
    # covered by ``validate_tile_layout``; this is its stripped-path
    # parity. Run before any window clamping so a malformed
    # ``ImageWidth=0`` IFD fails at the source rather than collapsing
    # to an empty post-clamp window.
    _check_source_dimensions(width, height, samples)
    compression = ifd.compression
    rps = ifd.rows_per_strip
    offsets = ifd.strip_offsets
    byte_counts = ifd.strip_byte_counts
    pred = ifd.predictor
    _validate_predictor_sample_format(pred, ifd.sample_format)
    bps = resolve_bits_per_sample(ifd.bits_per_sample)
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS
    jpeg_tables = ifd.jpeg_tables
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, dtype)
                   if compression == COMPRESSION_LERC else None)

    if offsets is None or byte_counts is None:
        raise ValueError("Missing strip offsets or byte counts")

    # Per-strip compressed-byte cap (issue #1664). Mirrors the HTTP path:
    # a crafted ``StripByteCounts`` can declare a huge value and even
    # though mmap slicing on the local path is bounded by the file size,
    # the slice is still passed into the decompressor which can expand
    # a few KiB of crafted deflate/zstd into gigabytes of decoded output.
    # Override via ``XRSPATIAL_COG_MAX_TILE_BYTES`` (the env var is shared
    # with the tile path because the budget is the same).
    max_tile_bytes = _max_tile_bytes_from_env()
    for _strip_idx, _bc in enumerate(byte_counts):
        if _bc > max_tile_bytes:
            raise ValueError(
                f"TIFF strip {_strip_idx} declares "
                f"StripByteCount={_bc:,} bytes, which exceeds the "
                f"per-strip safety cap of {max_tile_bytes:,} bytes. "
                f"The file is malformed or attempting denial-of-service. "
                f"Override via XRSPATIAL_COG_MAX_TILE_BYTES if this file "
                f"is legitimate."
            )

    # A corrupt header can report RowsPerStrip=0, which would divide by zero
    # below.  Reject it as a typed parse error rather than letting the
    # ZeroDivisionError leak out to the caller.
    if rps is None or rps <= 0:
        raise ValueError(f"Invalid RowsPerStrip: {rps!r}")

    planar = ifd.planar_config  # 1=chunky (interleaved), 2=planar (separate)

    # Determine output region
    if window is not None:
        r0, c0, r1, c1 = window
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(height, r1)
        c1 = min(width, c1)
    else:
        r0, c0, r1, c1 = 0, 0, height, width

    out_h = r1 - r0
    out_w = c1 - c0

    _check_dimensions(out_w, out_h, samples, max_pixels)

    # StripByteCounts must have at least one entry per strip; a corrupt count
    # field can shrink it.  Detect the mismatch after the dimension safety
    # check so an oversized header raises the safety-limit error first, then
    # raise a typed ValueError here instead of IndexError when the loop
    # indexes past the end.
    #
    # For PlanarConfiguration=2 (separate / planar) each sample plane has its
    # own run of strips, so the table must hold strips_per_band * samples
    # entries.  PlanarConfiguration=1 (chunky) interleaves samples within a
    # single run of strips_per_band entries.
    strips_per_band = (height + rps - 1) // rps
    if planar == 2 and samples > 1:
        n_strips_expected = strips_per_band * samples
        if len(offsets) < n_strips_expected or len(byte_counts) < n_strips_expected:
            raise ValueError(
                f"Strip table truncated for planar layout "
                f"(PlanarConfiguration=2): expected "
                f"{n_strips_expected} entries "
                f"({strips_per_band} strips x {samples} samples), got "
                f"offsets={len(offsets)}, byte_counts={len(byte_counts)}")
    else:
        n_strips_expected = strips_per_band
        if len(offsets) < n_strips_expected or len(byte_counts) < n_strips_expected:
            raise ValueError(
                f"Strip table truncated: expected {n_strips_expected} entries, "
                f"got offsets={len(offsets)}, byte_counts={len(byte_counts)}")

    # Sparse strips (StripByteCounts == 0) must materialise as nodata or 0
    # rather than be decoded.  Pre-fill the result so any skipped strips
    # land on a known fill value.
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

    # Collect strip jobs; decode in parallel when the pool overhead pays off.
    # Mirrors the tile path's gate at ``_PARALLEL_DECODE_PIXEL_THRESHOLD``:
    # codec decode (deflate, zstd, LZW) releases the GIL inside the C
    # extension so threads actually overlap codec work across cores. The
    # placement loop that copies pixels into ``result`` stays serial to
    # avoid contending writes to the output buffer. See issue #2100.
    strip_jobs: list[tuple[int, int, int]] = []  # (band_idx, strip_idx, global_idx)
    if planar == 2 and samples > 1:
        first_strip = r0 // rps
        last_strip = min((r1 - 1) // rps, strips_per_band - 1)
        for band_idx in range(samples):
            band_offset = band_idx * strips_per_band
            for strip_idx in range(first_strip, last_strip + 1):
                global_idx = band_offset + strip_idx
                if byte_counts[global_idx] == 0:
                    continue
                strip_row = strip_idx * rps
                strip_rows = min(rps, height - strip_row)
                if strip_rows <= 0:
                    continue
                strip_jobs.append((band_idx, strip_idx, global_idx))
        strip_samples = 1
    else:
        first_strip = r0 // rps
        last_strip = min((r1 - 1) // rps, len(offsets) - 1)
        for strip_idx in range(first_strip, last_strip + 1):
            strip_row = strip_idx * rps
            strip_rows = min(rps, height - strip_row)
            if strip_rows <= 0:
                continue
            if byte_counts[strip_idx] == 0:
                continue
            strip_jobs.append((0, strip_idx, strip_idx))
        strip_samples = samples

    def _decode_strip_job(job):
        _band_idx, strip_idx, global_idx = job
        strip_row = strip_idx * rps
        strip_rows = min(rps, height - strip_row)
        strip_data = data[
            offsets[global_idx]:offsets[global_idx] + byte_counts[global_idx]]
        return _decode_strip_or_tile(
            strip_data, compression, width, strip_rows, strip_samples,
            bps, bytes_per_sample, is_sub_byte, dtype, pred,
            byte_order=header.byte_order,
            jpeg_tables=jpeg_tables,
            masked_fill=masked_fill)

    n_strips = len(strip_jobs)
    strip_pixel_count = width * rps
    use_parallel = (n_strips > 1
                    and strip_pixel_count >= _PARALLEL_DECODE_PIXEL_THRESHOLD)
    if use_parallel:
        n_workers = min(n_strips, _os_module.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            decoded_strips = list(pool.map(_decode_strip_job, strip_jobs))
    else:
        decoded_strips = [_decode_strip_job(job) for job in strip_jobs]

    for (band_idx, strip_idx, _global_idx), strip_pixels in zip(
            strip_jobs, decoded_strips):
        strip_row = strip_idx * rps
        strip_rows = min(rps, height - strip_row)
        src_r0 = max(r0 - strip_row, 0)
        src_r1 = min(r1 - strip_row, strip_rows)
        dst_r0 = max(strip_row - r0, 0)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        if dst_r1 > dst_r0:
            if planar == 2 and samples > 1:
                result[dst_r0:dst_r1, :, band_idx] = (
                    strip_pixels[src_r0:src_r1, c0:c1])
            else:
                result[dst_r0:dst_r1] = strip_pixels[src_r0:src_r1, c0:c1]

    return result


# ---------------------------------------------------------------------------
# Tile reader
# ---------------------------------------------------------------------------

def _read_tiles(data: bytes, ifd: IFD, header: TIFFHeader,
                dtype: np.dtype, window=None,
                max_pixels: int = MAX_PIXELS_DEFAULT) -> np.ndarray:
    """Read a tile-organized TIFF image.

    Parameters
    ----------
    data : bytes
        Full file data.
    ifd : IFD
        Parsed IFD for this image.
    header : TIFFHeader
        File header.
    dtype : np.dtype
        Output pixel dtype.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop) or None for full image.
    max_pixels : int
        Maximum allowed pixel count (width * height * samples).

    Returns
    -------
    np.ndarray with shape (height, width) or windowed subset.
    """
    width = ifd.width
    height = ifd.height
    tw = ifd.tile_width
    th = ifd.tile_height
    samples = ifd.samples_per_pixel
    compression = ifd.compression
    pred = ifd.predictor
    _validate_predictor_sample_format(pred, ifd.sample_format)
    bps = resolve_bits_per_sample(ifd.bits_per_sample)
    bytes_per_sample = bps // 8
    is_sub_byte = bps in SUB_BYTE_BPS
    jpeg_tables = ifd.jpeg_tables
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, dtype)
                   if compression == COMPRESSION_LERC else None)

    offsets = ifd.tile_offsets
    byte_counts = ifd.tile_byte_counts
    if offsets is None or byte_counts is None:
        raise ValueError("Missing tile offsets or byte counts")

    if tw <= 0 or th <= 0:
        raise ValueError(
            f"Invalid tile dimensions: TileWidth={tw}, TileLength={th}")

    # Reject crafted tile dims (e.g. TileWidth = 2**31). This guards the
    # TIFF header against malformed values; it is not the caller's output
    # budget. The output-window check below uses ``max_pixels`` and is
    # what enforces the user's per-call memory cap. The source-read path
    # under ``read_vrt`` (#1796) relies on that output check to honour a
    # small caller ``max_pixels`` against a normal-tile source; see
    # #1823.
    _check_dimensions(tw, th, samples, MAX_PIXELS_DEFAULT)

    # Per-tile compressed-byte cap (issue #1664). Same env var as the
    # HTTP path. mmap slicing is bounded by the file size, but the slice
    # gets handed to the decompressor, and a small slice can balloon
    # into gigabytes through deflate / zstd / lzw / lerc.
    max_tile_bytes = _max_tile_bytes_from_env()
    for _tile_idx, _bc in enumerate(byte_counts):
        if _bc > max_tile_bytes:
            raise ValueError(
                f"TIFF tile {_tile_idx} declares "
                f"TileByteCount={_bc:,} bytes, which exceeds the "
                f"per-tile safety cap of {max_tile_bytes:,} bytes. "
                f"The file is malformed or attempting denial-of-service. "
                f"Override via XRSPATIAL_COG_MAX_TILE_BYTES if this file "
                f"is legitimate."
            )

    planar = ifd.planar_config
    tiles_across = math.ceil(width / tw)
    tiles_down = math.ceil(height / th)

    if window is not None:
        r0, c0, r1, c1 = window
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(height, r1)
        c1 = min(width, c1)
    else:
        r0, c0, r1, c1 = 0, 0, height, width

    out_h = r1 - r0
    out_w = c1 - c0

    _check_dimensions(out_w, out_h, samples, max_pixels)

    # Reject malformed TIFFs whose declared tile grid exceeds the number of
    # supplied TileOffsets entries. Silent skipping in the CPU loop below
    # would mask the problem, and the GPU path reads OOB. See issue #1219.
    validate_tile_layout(ifd)

    # Sparse tiles (TileByteCounts == 0) must materialise as nodata or 0
    # rather than be decoded.  Pre-fill the result so any skipped tiles
    # land on a known fill value; otherwise sparse regions would leak
    # uninitialised memory (full-image read) or stay zeroed regardless
    # of the file's nodata setting (windowed read).
    sparse = _has_sparse(byte_counts)
    if sparse:
        fill = _sparse_fill_value(ifd, dtype)
        if samples > 1:
            result = np.full((out_h, out_w, samples), fill, dtype=dtype)
        else:
            result = np.full((out_h, out_w), fill, dtype=dtype)
    else:
        _alloc = np.zeros if window is not None else np.empty
        if samples > 1:
            result = _alloc((out_h, out_w, samples), dtype=dtype)
        else:
            result = _alloc((out_h, out_w), dtype=dtype)

    tile_row_start = r0 // th
    tile_row_end = min(math.ceil(r1 / th), tiles_down)
    tile_col_start = c0 // tw
    tile_col_end = min(math.ceil(c1 / tw), tiles_across)

    band_count = samples if (planar == 2 and samples > 1) else 1
    tiles_per_band = tiles_across * tiles_down

    # Build list of tiles to decode.  Sparse tiles (byte_count==0) are
    # skipped here -- the result is pre-filled with the sparse fill value.
    tile_jobs = []
    for band_idx in range(band_count):
        band_tile_offset = band_idx * tiles_per_band if band_count > 1 else 0
        tile_samples = 1 if band_count > 1 else samples

        for tr in range(tile_row_start, tile_row_end):
            for tc in range(tile_col_start, tile_col_end):
                tile_idx = band_tile_offset + tr * tiles_across + tc
                if tile_idx >= len(offsets):
                    continue
                if byte_counts[tile_idx] == 0:
                    continue
                tile_jobs.append((band_idx, tr, tc, tile_idx, tile_samples))

    # Decode tiles in parallel when the work per tile is large enough to
    # outweigh the thread-pool overhead. Uncompressed multi-tile reads also
    # benefit because numpy frombuffer + slice copies aren't free at large
    # tile sizes. Threshold is shared with the HTTP COG path below
    # (issue #1551).
    n_tiles = len(tile_jobs)
    tile_pixels = tw * th
    use_parallel = (n_tiles > 1 and tile_pixels >= _PARALLEL_DECODE_PIXEL_THRESHOLD)

    def _decode_one(job):
        band_idx, tr, tc, tile_idx, tile_samples = job
        tile_data = data[offsets[tile_idx]:offsets[tile_idx] + byte_counts[tile_idx]]
        return _decode_strip_or_tile(
            tile_data, compression, tw, th, tile_samples,
            bps, bytes_per_sample, is_sub_byte, dtype, pred,
            byte_order=header.byte_order,
            jpeg_tables=jpeg_tables,
            masked_fill=masked_fill)

    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor
        import os as _os
        n_workers = min(n_tiles, _os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            decoded = list(pool.map(_decode_one, tile_jobs))
    else:
        decoded = [_decode_one(job) for job in tile_jobs]

    # Place decoded tiles into the output array
    for (band_idx, tr, tc, tile_idx, tile_samples), tile_pixels in zip(tile_jobs, decoded):
        tile_r0 = tr * th
        tile_c0 = tc * tw

        src_r0 = max(r0 - tile_r0, 0)
        src_c0 = max(c0 - tile_c0, 0)
        src_r1 = min(r1 - tile_r0, th)
        src_c1 = min(c1 - tile_c0, tw)

        dst_r0 = max(tile_r0 - r0, 0)
        dst_c0 = max(tile_c0 - c0, 0)

        actual_tile_h = min(th, height - tile_r0)
        actual_tile_w = min(tw, width - tile_c0)
        src_r1 = min(src_r1, actual_tile_h)
        src_c1 = min(src_c1, actual_tile_w)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c1 = dst_c0 + (src_c1 - src_c0)

        if dst_r1 > dst_r0 and dst_c1 > dst_c0:
            src_slice = tile_pixels[src_r0:src_r1, src_c0:src_c1]
            if band_count > 1:
                result[dst_r0:dst_r1, dst_c0:dst_c1, band_idx] = src_slice
            else:
                result[dst_r0:dst_r1, dst_c0:dst_c1] = src_slice

    return result


# ---------------------------------------------------------------------------
# COG HTTP reader
# ---------------------------------------------------------------------------

#: Initial prefetch size for ``_parse_cog_http_meta``. Sized for the common
#: case (a single-IFD COG with modest GeoTIFF tags) so the fast path is a
#: single range GET.
INITIAL_HTTP_HEADER_BYTES = 16 * 1024

#: Upper bound on how far ``_parse_cog_http_meta`` will grow its prefetch
#: buffer before giving up. 4 MiB comfortably covers deep pyramids whose
#: IFD chains plus tag arrays (TileOffsets, GeoAsciiParams, GDAL_METADATA)
#: extend far past the initial fetch window. See issue #1718.
MAX_HTTP_HEADER_BYTES = 4 * 1024 * 1024


def _ifd_required_extent(
    ifds: list[IFD], header: TIFFHeader, data_len: int,
) -> int:
    """Return the highest byte offset the parsed IFDs reference.

    Used to decide whether the prefetch buffer is large enough to hold the
    entire IFD chain plus every out-of-line tag value. We compare this
    against ``len(data)`` in :func:`_parse_cog_http_meta`; if it exceeds the
    buffer, the chain is truncated and the caller must grow and retry.

    The walk re-derives each tag's value-area placement directly from the
    IFD layout (entry table base + entry slot) rather than re-parsing the
    raw bytes. For out-of-line tags ``parse_ifd`` already resolved the
    pointer and validated ``ptr + size <= data_len``; the *interesting*
    extent for the grow loop is the next-IFD pointer of the chain tail,
    plus an "is there a next IFD we have not yet seen" probe.
    """
    if not ifds:
        return 0

    required = 0
    # Last IFD's next_ifd_offset: 0 means end-of-chain; anything else
    # points at an IFD we haven't parsed yet because it sat past the
    # buffer (parse_all_ifds stops on offset >= len(data)).
    tail_next = ifds[-1].next_ifd_offset
    if tail_next != 0:
        # Need at least enough bytes to reach the next IFD header. Pad
        # by a small amount so parse_ifd can read the num_entries field
        # without truncation -- the actual entry table is bounded by the
        # parser's own checks on the next grow iteration.
        required = max(required, tail_next + 64)

    # Out-of-line tag values are already parsed (parse_ifd bounds-checked
    # ptr + total_size <= len(data) before reading). For grow logic we
    # only need to ensure those checks did not *fail*; a thrown
    # ValueError surfaces in parse_all_ifds and is handled by the loop.
    return required


def _parse_cog_http_meta(
    source: _HTTPSource,
    overview_level: int | None = None,
    *,
    allow_rotated: bool = False,
) -> tuple[TIFFHeader, IFD, GeoInfo, bytes]:
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
    """
    fetch_size = INITIAL_HTTP_HEADER_BYTES
    header_bytes = source.read_range(0, fetch_size)
    header = parse_header(header_bytes)

    last_len = len(header_bytes)
    ifds: list[IFD] = []
    while True:
        try:
            ifds = parse_all_ifds(header_bytes, header)
            required = _ifd_required_extent(ifds, header, len(header_bytes))
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

        if fetch_size >= MAX_HTTP_HEADER_BYTES:
            raise ValueError(
                f"COG IFD chain or tag arrays extend past "
                f"MAX_HTTP_HEADER_BYTES={MAX_HTTP_HEADER_BYTES} bytes; "
                f"the file may be malformed or its pyramid metadata is "
                f"unusually large for HTTP prefetch")
        fetch_size = min(fetch_size * 2, MAX_HTTP_HEADER_BYTES)
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

    ifd = select_overview_ifd(ifds, overview_level)
    # When the requested IFD is an overview that lacks its own geokeys
    # (the common case for COG writers, including this package's
    # ``to_geotiff``), inherit and rescale the georef from the level-0
    # IFD so overview reads do not silently lose CRS / transform.
    # See issue #1640.
    geo_info = extract_geo_info_with_overview_inheritance(
        ifd, ifds, header_bytes, header.byte_order,
        allow_rotated=allow_rotated)
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
    source = _HTTPSource(url)
    # Issue #1816: wrap everything after the ``_HTTPSource`` construction
    # in try/finally so ``source.close()`` runs even when header parsing,
    # validation, fetch/decode, or orientation/photometric post-processing
    # raises. ``_HTTPSource.close()`` is a no-op today, but a future
    # resource-holding source would leak on the error path without this.
    # ``close()`` is idempotent, so the explicit pre-raise ``source.close()``
    # calls in the validation blocks below stay as-is.
    try:
        header, ifd, geo_info, header_bytes = _parse_cog_http_meta(
            source, overview_level=overview_level,
            allow_rotated=allow_rotated)

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

        arr = _fetch_decode_cog_http_tiles(
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
        arr = _apply_photometric_miniswhite(arr, ifd)
    finally:
        source.close()

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

    def _decode_http_strip(args):
        strip_idx, strip_data = args
        strip_row = strip_idx * rps
        strip_rows = min(rps, height - strip_row)
        if strip_rows <= 0:
            return None
        # Per-strip decoded-dimension cap (#1851). See note below.
        _check_dimensions(width, strip_rows, strip_samples,
                          MAX_PIXELS_DEFAULT)
        return _decode_strip_or_tile(
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

    def _decode_one(tile_data):
        return _decode_strip_or_tile(
            tile_data, compression, tw, th, tile_samples,
            bps, bytes_per_sample, is_sub_byte, dtype, pred,
            byte_order=header.byte_order,
            jpeg_tables=jpeg_tables,
            masked_fill=masked_fill)

    if decode_in_parallel:
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


# ---------------------------------------------------------------------------
# Main read function
# ---------------------------------------------------------------------------

def _apply_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Reorient a decoded TIFF array according to the Orientation tag (274).

    The TIFF 6.0 spec defines eight orientations describing where the
    *first row* and *first column* of the stored data sit relative to the
    visual top-left of the image:

    ===  =================  ========================================
     1   top-left           identity (default, no transform)
     2   top-right          mirror horizontally (flip columns)
     3   bottom-right       rotate 180 degrees
     4   bottom-left        mirror vertically (flip rows)
     5   left-top           transpose (rows<->columns)
     6   right-top          rotate 90 clockwise
     7   right-bottom       transverse (anti-transpose)
     8   left-bottom        rotate 90 counter-clockwise
    ===  =================  ========================================

    Values 5-8 swap rows and columns: the file's stored width becomes the
    output's height and vice versa.

    The input ``arr`` is shaped ``(height, width)`` or
    ``(height, width, samples)``. Multi-band 3D arrays only have their
    first two axes transformed; the sample axis is preserved.
    """
    if orientation == 1:
        return arr
    if orientation == 2:
        return np.ascontiguousarray(arr[:, ::-1])
    if orientation == 3:
        return np.ascontiguousarray(arr[::-1, ::-1])
    if orientation == 4:
        return np.ascontiguousarray(arr[::-1, :])
    # Orientations 5-8 swap rows and columns.
    if arr.ndim == 3:
        # Transpose only the spatial axes; keep the sample axis trailing.
        if orientation == 5:
            return np.ascontiguousarray(arr.transpose(1, 0, 2))
        if orientation == 6:
            return np.ascontiguousarray(arr.transpose(1, 0, 2)[:, ::-1])
        if orientation == 7:
            return np.ascontiguousarray(arr.transpose(1, 0, 2)[::-1, ::-1])
        if orientation == 8:
            return np.ascontiguousarray(arr.transpose(1, 0, 2)[::-1, :])
    else:
        if orientation == 5:
            return np.ascontiguousarray(arr.T)
        if orientation == 6:
            return np.ascontiguousarray(arr.T[:, ::-1])
        if orientation == 7:
            return np.ascontiguousarray(arr.T[::-1, ::-1])
        if orientation == 8:
            return np.ascontiguousarray(arr.T[::-1, :])
    raise ValueError(
        f"Invalid TIFF Orientation tag value: {orientation} "
        f"(must be 1-8 per TIFF 6.0)"
    )


def _apply_orientation_with_geo(
    arr: np.ndarray, geo_info: GeoInfo, orientation: int,
) -> tuple[np.ndarray, GeoInfo]:
    """Apply Orientation tag to ``arr`` and update ``geo_info`` to match.

    Shared helper used by the local-file and HTTP COG paths so both
    return the same pixel order and transform for a given file. See
    issue #1717 for the HTTP-path parity break this consolidates.
    """
    if orientation == 1:
        return arr, geo_info
    # Use the *file* dimensions (before orientation) for the transform
    # math below. After ``_apply_orientation`` the array shape may swap
    # (orientations 5-8), so capture them now.
    file_h = arr.shape[0]
    file_w = arr.shape[1]
    arr = _apply_orientation(arr, orientation)
    t = geo_info.transform
    if not geo_info.has_georef:
        pass
    elif orientation in (2, 3, 4):
        if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
            x_shift = file_w - 1
            y_shift = file_h - 1
        else:
            x_shift = file_w
            y_shift = file_h
        new_origin_x = t.origin_x
        new_origin_y = t.origin_y
        new_px_w = t.pixel_width
        new_px_h = t.pixel_height
        if orientation in (2, 3):  # x flipped
            new_origin_x = t.origin_x + x_shift * t.pixel_width
            new_px_w = -t.pixel_width
        if orientation in (3, 4):  # y flipped
            new_origin_y = t.origin_y + y_shift * t.pixel_height
            new_px_h = -t.pixel_height
        geo_info.transform = GeoTransform(
            origin_x=new_origin_x,
            origin_y=new_origin_y,
            pixel_width=new_px_w,
            pixel_height=new_px_h,
        )
    elif orientation in (5, 6, 7, 8):
        # ``has_georef`` is True whenever ModelTransformation,
        # ModelPixelScale, or ModelTiepoint is present, even without a
        # CRS. The pixel-size swap below cannot express the
        # per-orientation origin shift plus rotation these orientations
        # require, so the x/y coords would be wrong whether or not a
        # CRS tag accompanies the transform. Refuse the file in that
        # case rather than warn and return silently wrong coords.
        raise NotImplementedError(
            f"TIFF Orientation {orientation} on a georeferenced file "
            f"requires a per-orientation origin shift plus a rotation "
            f"that the axis-aligned GeoTransform used here cannot "
            f"represent, so the returned x/y coords would be wrong. "
            f"Reproject the file with another tool (e.g. GDAL) or "
            f"strip the Orientation tag before reading. See issue "
            f"#1765."
        )
    return arr, geo_info


def _apply_photometric_miniswhite(arr: np.ndarray, ifd: IFD) -> np.ndarray:
    """Apply TIFF MinIsWhite inversion for single-band grayscale images."""
    if ifd.photometric != 0 or ifd.samples_per_pixel != 1:
        return arr
    if arr.dtype.kind == 'u':
        return np.iinfo(arr.dtype).max - arr
    if arr.dtype.kind == 'f':
        return -arr
    return arr


def _miniswhite_inverted_nodata(nodata, ifd: IFD, dtype: np.dtype):
    """Return the nodata sentinel value after MinIsWhite inversion.

    When the reader applied MinIsWhite (``photometric == 0``,
    ``samples_per_pixel == 1``), the original integer sentinel ``s`` is
    rewritten to ``iinfo(dtype).max - s`` and the float sentinel ``s`` to
    ``-s``.  Downstream nodata-to-NaN masks must compare against the
    inverted sentinel rather than the original, otherwise they flag the
    wrong pixels: inverted real data colliding with the original
    sentinel value is incorrectly masked while the real nodata cells
    keep their inverted-sentinel value (issue #1809).

    Returns the inverted nodata sentinel, or the original ``nodata``
    when MinIsWhite was not applied / not applicable.  Non-finite or
    out-of-range nodata is returned unchanged so callers' downstream
    skip-the-mask logic stays unchanged.
    """
    if nodata is None:
        return nodata
    if ifd.photometric != 0 or ifd.samples_per_pixel != 1:
        return nodata
    if dtype.kind == 'u':
        if not np.isfinite(nodata):
            return nodata
        if not float(nodata).is_integer():
            return nodata
        vi = int(nodata)
        info = np.iinfo(dtype)
        if not (info.min <= vi <= info.max):
            return nodata
        return info.max - vi
    if dtype.kind == 'f':
        if np.isnan(nodata):
            return nodata
        return -float(nodata)
    return nodata


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
        from ._sidecar import (
            attach_sidecar_origin, find_sidecar, load_sidecar,
        )
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
        # NewSubfileType=overview entry. Sidecar IFDs always lack
        # geokeys, so the inheritance pulls from the base file's
        # level-0 IFD (kept first in the merged list) which is the
        # GDAL convention.
        geo_info = extract_geo_info_with_overview_inheritance(
            ifd, ifds, data, header.byte_order,
            allow_rotated=allow_rotated)

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
