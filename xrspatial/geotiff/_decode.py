"""Strip/tile decode orchestration for TIFF/COG reads.

This module holds the transport-independent decode helpers extracted
from :mod:`xrspatial.geotiff._reader` in PR-G of the GeoTIFF refactor
epic (issue #2211, sub-issue #2246). The functions here take an already
materialised byte slice plus an IFD and a header, and produce a numpy
array. Everything that knows how to *fetch* those bytes (local mmap,
HTTP, fsspec) stays in :mod:`xrspatial.geotiff._sources`; everything
that orchestrates the top-level read flow (windowing, overview
selection, DataArray packaging) stays in :mod:`xrspatial.geotiff._reader`.

Public import surface from :mod:`xrspatial.geotiff._reader` is preserved
by re-importing the names defined here back at the call sites that the
test suite and downstream backends depend on. See the import block at
the top of ``_reader.py``.

This module deliberately has *no* module-level import of
:mod:`xrspatial.geotiff._reader` so the two files can sit on either side
of a circular relationship: ``_reader.py`` imports the decode functions
back at module load, and the few things ``_decode`` still needs from
``_reader`` (``MAX_PIXELS_DEFAULT``, ``_check_dimensions``,
``_check_source_dimensions``, ``_sparse_fill_value``, ``_has_sparse``)
are imported lazily inside ``_read_strips`` / ``_read_tiles`` at call
time. Those names move with ``_layout.py`` in PR-H (issue #2247), at
which point the lazy imports can collapse back into top-level ones.
"""
from __future__ import annotations

import math
import os as _os_module
import sys as _sys

import numpy as np

from ._compression import (
    COMPRESSION_LERC,
    decompress,
    fp_predictor_decode,
    lerc_decompress_with_mask,
    predictor_decode,
    unpack_bits,
)
from ._dtypes import SUB_BYTE_BPS, resolve_bits_per_sample
from ._geotags import (
    GeoInfo,
    GeoTransform,
    RASTER_PIXEL_IS_POINT,
)
from ._header import (
    IFD,
    TIFFHeader,
    validate_tile_layout,
)
from ._sources import _max_tile_bytes_from_env
from ._validation import _validate_predictor_sample_format

_NATIVE_ORDER = '<' if _sys.byteorder == 'little' else '>'

#: Sentinel used as the default value of ``max_pixels`` so the actual
#: ``MAX_PIXELS_DEFAULT`` can stay in :mod:`._reader` without making this
#: module import-time dependent on it. ``_resolve_max_pixels`` does the
#: lazy lookup at call time.
_MAX_PIXELS_UNSET = object()


def _resolve_max_pixels(value):
    """Return ``MAX_PIXELS_DEFAULT`` when *value* is the unset sentinel."""
    if value is _MAX_PIXELS_UNSET:
        from ._reader import MAX_PIXELS_DEFAULT
        return MAX_PIXELS_DEFAULT
    return value

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


# ---------------------------------------------------------------------------
# Strip reader
# ---------------------------------------------------------------------------

def _read_strips(data: bytes, ifd: IFD, header: TIFFHeader,
                 dtype: np.dtype, window=None,
                 max_pixels: int = _MAX_PIXELS_UNSET) -> np.ndarray:  # type: ignore[assignment]
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
    # Imported lazily so this module does not have an import-time
    # dependency on ``_reader``. The pixel-safety guards
    # (``_check_dimensions``/``_check_source_dimensions``) and the
    # sparse-layout helpers (``_sparse_fill_value``/``_has_sparse``)
    # both stay in ``_reader.py`` until PR-H (issue #2247); rebinding
    # them here keeps PR-G mechanical.
    from ._reader import (
        _check_dimensions,
        _check_source_dimensions,
        _has_sparse,
        _sparse_fill_value,
    )
    max_pixels = _resolve_max_pixels(max_pixels)
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
        # Function-local import (rather than the module-level binding)
        # so tests that monkey-patch ``concurrent.futures.ThreadPoolExecutor``
        # see the spy class here. The tile path uses the same pattern;
        # both gates share ``_PARALLEL_DECODE_PIXEL_THRESHOLD`` (#1551).
        from concurrent.futures import ThreadPoolExecutor
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
                max_pixels: int = _MAX_PIXELS_UNSET) -> np.ndarray:  # type: ignore[assignment]
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
    from ._reader import (
        MAX_PIXELS_DEFAULT,
        _check_dimensions,
        _has_sparse,
        _sparse_fill_value,
    )
    max_pixels = _resolve_max_pixels(max_pixels)
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
        # Function-local import (rather than the module-level binding)
        # so tests that monkey-patch ``concurrent.futures.ThreadPoolExecutor``
        # see the spy class here. Issue #1551.
        from concurrent.futures import ThreadPoolExecutor
        n_workers = min(n_tiles, _os_module.cpu_count() or 4)
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
# Orientation and photometric post-processing
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
