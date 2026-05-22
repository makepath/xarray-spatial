"""Strip/tile encode orchestration for TIFF/COG writes.

This module holds the transport-independent encode helpers extracted
from :mod:`xrspatial.geotiff._writer` in PR-L of the GeoTIFF refactor
epic (issue #2211, sub-issue #2260). The functions here take an already
materialised numpy array plus codec / predictor / photometric kwargs
and produce on-disk byte chunks (compressed strips, compressed tiles)
or normalised tag values. Everything that orchestrates the top-level
write flow (DataArray packaging, IFD assembly, fsspec destinations,
streaming dask compute) stays in :mod:`xrspatial.geotiff._writer`;
everything that assembles the IFD / file layout itself lives in
:mod:`xrspatial.geotiff._write_layout`.

Public import surface from :mod:`xrspatial.geotiff._writer` is preserved
by re-importing the names defined here back at the call sites that the
test suite and the ``_writers`` subpackage depend on. See the import
block at the top of ``_writer.py``.

This module is the write-side analog of :mod:`xrspatial.geotiff._decode`
(PR-G). Both modules sit between the codec primitives in ``_compression``
and the higher-level orchestration in ``_reader`` / ``_writer``.

This module deliberately has *no* module-level import of
:mod:`xrspatial.geotiff._writer` or :mod:`xrspatial.geotiff._write_layout`
beyond the small set of constants the encode helpers need (``BO`` for
on-disk byte order). ``_writer.py`` imports the encode functions back
at module load. The two orchestrators that have to honour an
existing monkeypatch contract -- ``_write_stripped`` and
``_write_tiled`` -- look up ``_prepare_strip`` / ``_prepare_tile``
through ``_writer`` lazily at call time, so the import graph stays
acyclic and the test-suite monkeypatches on ``_writer.<name>`` flow
through to the orchestrators unchanged.
"""
from __future__ import annotations

import math

import numpy as np

from ._compression import (COMPRESSION_DEFLATE, COMPRESSION_JPEG, COMPRESSION_JPEG2000,
                           COMPRESSION_LERC, COMPRESSION_LZ4, COMPRESSION_LZW, COMPRESSION_NONE,
                           COMPRESSION_PACKBITS, COMPRESSION_ZSTD, compress, fp_predictor_encode,
                           jpeg_compress, predictor_encode)
from ._header import TAG_PHOTOMETRIC
from ._write_layout import BO

# TIFF Photometric Interpretation values (TIFF 6 spec, tag 262).
PHOTOMETRIC_MINISBLACK = 1
PHOTOMETRIC_RGB = 2

# Friendly names accepted by the ``photometric`` writer kwarg. ``'auto'``
# defaults to MinIsBlack so scientific multispectral rasters (e.g.
# R,G,B,NIR) round-trip without being silently tagged as RGB+alpha.
# ``'rgba'`` is a convenience for "RGB plus an unassociated-alpha extra
# sample".  See issue #1769.
_PHOTOMETRIC_NAME_MAP = {
    'auto': 'auto',
    'minisblack': PHOTOMETRIC_MINISBLACK,
    'miniswhite': 0,
    'rgb': PHOTOMETRIC_RGB,
    'rgba': 'rgba',
}

#: Total uncompressed payload (bytes) below which the strip and tile
#: writers stay sequential. The thread-pool startup cost dominates on
#: small rasters; above this size the per-block compression cost more
#: than pays for it. 4 MiB was chosen empirically on a 20-core box:
#: parallel becomes a net win around ~2 MiB, and the 4 MiB margin keeps
#: a few-tile / two-strip layout from incurring a slowdown.
_PARALLEL_MIN_BYTES = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Photometric / nodata helpers
# ---------------------------------------------------------------------------

def _invert_nodata_for_miniswhite(nodata, dtype: np.dtype):
    """Invert a nodata sentinel for MinIsWhite writes.

    The reader's mask path (see ``_reader._miniswhite_inverted_nodata``)
    treats the stored sentinel as living in the on-disk display domain,
    so the writer pre-inverts the user-supplied sentinel alongside the
    pixels. After the writer pre-inversion, the on-disk sentinel byte
    matches the on-disk pixel byte that represents "missing", and the
    reader inverts both back to the user domain to drive the NaN mask.
    Returns ``nodata`` unchanged for signed integer pixels, NaN
    sentinels, and unsigned sentinels that are out-of-range or
    fractional or non-finite -- matching the reader exactly. ``+/-inf``
    on a float sentinel is negated (the reader does the same). See
    issue #1836.
    """
    if nodata is None:
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


def _apply_photometric_miniswhite_invert(
    arr: np.ndarray, resolved_photometric: int, samples_per_pixel: int,
) -> np.ndarray:
    """Mirror the reader's MinIsWhite inversion on the writer side.

    The reader unconditionally inverts single-band ``photometric == 0``
    data via ``_reader._apply_photometric_miniswhite``. Without a
    matching writer-side inversion, ``to_geotiff(da, photometric=
    'miniswhite')`` silently corrupts pixel values on the round trip.
    See issue #1836.

    Returns the pre-inverted array (a new array) so that the reader's
    inversion restores the original values. Multi-band data passes
    through unchanged. Signed-integer single-band MinIsWhite raises
    ``NotImplementedError`` -- the previous passthrough produced files
    whose pixel values disagreed with the on-disk Photometric tag
    against every standards-compliant TIFF consumer (GDAL, libtiff).
    See issue #2278.
    """
    if resolved_photometric != 0 or samples_per_pixel != 1:
        return arr
    if arr.dtype.kind == 'u':
        return np.iinfo(arr.dtype).max - arr
    if arr.dtype.kind == 'f':
        return -arr
    if arr.dtype.kind == 'i':
        # Defer the import to dodge any module-load cycle through
        # ``_dtypes``.
        from ._dtypes import numpy_to_tiff_dtype
        bps, sf = numpy_to_tiff_dtype(arr.dtype)
        raise NotImplementedError(
            f"Writing signed-integer MinIsWhite TIFFs is not supported "
            f"(issue #2278): Photometric=0 (MinIsWhite), "
            f"SampleFormat={sf} (signed int), BitsPerSample={bps}, "
            f"dtype={arr.dtype}. xrspatial has no semantically correct "
            f"inversion for signed pixels here, and writing them "
            f"un-inverted would produce a file whose pixels disagree "
            f"with the Photometric tag against every standards-"
            f"compliant TIFF reader (GDAL, libtiff, etc.). Cast to an "
            f"unsigned dtype, or write with photometric='minisblack' "
            f"/ 'auto'."
        )
    return arr


def _resolve_photometric(photometric, samples_per_pixel: int):
    """Resolve the ``photometric`` writer kwarg to a TIFF photometric int
    and the matching ExtraSamples list.

    Returns ``(photometric_int, extra_samples_list)`` where
    ``extra_samples_list`` has one entry per band beyond what the
    photometric model accounts for (empty when no ExtraSamples tag is
    needed).
    """
    if isinstance(photometric, str):
        key = photometric.lower()
        if key not in _PHOTOMETRIC_NAME_MAP:
            valid = sorted(_PHOTOMETRIC_NAME_MAP)
            raise ValueError(
                f"photometric={photometric!r} is not a valid name; "
                f"expected one of {valid} or an int.")
        resolved = _PHOTOMETRIC_NAME_MAP[key]
    elif isinstance(photometric, (int, np.integer)) and not isinstance(
            photometric, bool):
        resolved = int(photometric)
    else:
        raise TypeError(
            f"photometric must be a str or int, got "
            f"{type(photometric).__name__}")

    if resolved == 'auto':
        # MinIsBlack default for every band count. Scientific
        # multispectral rasters fall through here; previous behaviour
        # silently called any 4th band an alpha channel.
        n_extra = max(0, samples_per_pixel - 1)
        return PHOTOMETRIC_MINISBLACK, [0] * n_extra

    if resolved == 'rgba':
        if samples_per_pixel < 4:
            raise ValueError(
                f"photometric='rgba' requires at least 4 bands, got "
                f"samples_per_pixel={samples_per_pixel}.")
        n_extra = samples_per_pixel - 3
        return PHOTOMETRIC_RGB, [2] + [0] * (n_extra - 1)

    photo_int = int(resolved)
    if photo_int == PHOTOMETRIC_RGB and samples_per_pixel < 3:
        raise ValueError(
            f"photometric=RGB requires at least 3 bands, got "
            f"samples_per_pixel={samples_per_pixel}.")
    consumed = 3 if photo_int == PHOTOMETRIC_RGB else 1
    if samples_per_pixel > consumed:
        return photo_int, [0] * (samples_per_pixel - consumed)
    return photo_int, []


def _reject_disagreeing_photometric_override(
    extra_tags, resolved_photo: int, samples: int, photometric
) -> None:
    """Reject an ``extra_tags`` entry that overrides ``TAG_PHOTOMETRIC``
    across the MinIsWhite boundary for a single-band raster.

    The single-band MinIsWhite path requires the writer to pre-invert
    pixels (and the nodata sentinel) so the round-trip matches what the
    reader unconditionally inverts. An ``extra_tags`` entry that flips
    ``TAG_PHOTOMETRIC`` between MinIsWhite (0) and anything else makes
    the on-disk tag advertise one model while the bytes were
    pre-processed for the other -- the round-trip silently corrupts.

    The eager and streaming writers both call this guard before any
    pre-inversion runs. Only the MinIsWhite-crossing single-band case
    is rejected; multi-band rasters and non-crossing overrides (e.g.
    photometric='minisblack' with extra_tags=[(262, SHORT, 1, 1)])
    pass through unchanged. Issues #2073 / #1769 / #1836.
    """
    if extra_tags is None:
        return
    override = None
    for _et in extra_tags:
        if _et[0] == TAG_PHOTOMETRIC:
            override = int(_et[3])
            break
    if override is None:
        return
    if override == resolved_photo:
        return
    if not (override == 0 or resolved_photo == 0):
        return
    if samples != 1:
        return
    raise ValueError(
        f"extra_tags TAG_PHOTOMETRIC override ({override}) "
        f"disagrees with photometric={photometric!r} for a "
        f"single-band raster where MinIsWhite (photometric=0) "
        f"requires writer-side pixel inversion. The override would "
        f"either pre-invert pixels for a non-MinIsWhite tag or skip "
        f"inversion for a MinIsWhite tag. Pass photometric= directly "
        f"instead, or drop the override."
    )


# ---------------------------------------------------------------------------
# Predictor helpers
# ---------------------------------------------------------------------------

def normalize_predictor(predictor, dtype, compression: int) -> int:
    """Normalize a user-supplied predictor value to a TIFF predictor int.

    Accepts ``False``/``True`` (legacy) and integers ``1``/``2``/``3``.
    Returns ``1`` (no predictor), ``2`` (horizontal differencing), or ``3``
    (floating-point predictor).
    """
    if predictor is False or predictor == 0:
        return 1
    if predictor is True or predictor == 2:
        return 2
    if predictor == 1:
        return 1
    if predictor == 3:
        if np.dtype(dtype).kind != 'f':
            raise ValueError(
                "predictor=3 (floating-point) requires float data, "
                f"got dtype={np.dtype(dtype)}")
        return 3
    raise ValueError(
        f"predictor must be False/True or 1/2/3, got {predictor!r}")


def _apply_predictor_encode(buf: np.ndarray, predictor: int,
                            width: int, height: int,
                            bytes_per_sample: int, samples: int) -> np.ndarray:
    """Apply the chosen predictor to a flat uint8 buffer.

    Files always go to disk in little-endian order (see ``BO``), so
    ``predictor_encode`` is invoked with ``byte_order='<'``.
    """
    if predictor == 2:
        return predictor_encode(buf, width, height,
                                bytes_per_sample, samples=samples,
                                byte_order=BO)
    if predictor == 3:
        return fp_predictor_encode(buf, width * samples, height,
                                   bytes_per_sample)
    return buf


# ---------------------------------------------------------------------------
# Compression name -> TIFF tag value
# ---------------------------------------------------------------------------

def _compression_tag(compression_name: str) -> int:
    """Convert compression name to TIFF tag value."""
    _map = {
        'none': COMPRESSION_NONE,
        'deflate': COMPRESSION_DEFLATE,
        'lzw': COMPRESSION_LZW,
        'jpeg': COMPRESSION_JPEG,
        'packbits': COMPRESSION_PACKBITS,
        'zstd': COMPRESSION_ZSTD,
        'lz4': COMPRESSION_LZ4,
        'jpeg2000': COMPRESSION_JPEG2000,
        'j2k': COMPRESSION_JPEG2000,
        'lerc': COMPRESSION_LERC,
    }
    name = compression_name.lower()
    if name not in _map:
        raise ValueError(f"Unsupported compression: {compression_name!r}. "
                         f"Use one of: {list(_map.keys())}")
    return _map[name]


# ---------------------------------------------------------------------------
# Strip writer
# ---------------------------------------------------------------------------

def _prepare_strip(data, i, rows_per_strip, height, width, samples, dtype,
                   bytes_per_sample, predictor: int, compression,
                   compression_level=None, max_z_error: float = 0.0,
                   gil_friendly: bool = False):
    """Extract and compress a single strip. Thread-safe."""
    r0 = i * rows_per_strip
    r1 = min(r0 + rows_per_strip, height)
    strip_rows = r1 - r0

    if compression == COMPRESSION_JPEG:
        strip_data = np.ascontiguousarray(data[r0:r1]).tobytes()
        return jpeg_compress(strip_data, width, strip_rows, samples)
    if predictor != 1 and compression != COMPRESSION_NONE:
        strip_arr = np.ascontiguousarray(data[r0:r1])
        buf = strip_arr.view(np.uint8).ravel().copy()
        buf = _apply_predictor_encode(
            buf, predictor, width, strip_rows, bytes_per_sample, samples)
        strip_data = buf.tobytes()
    else:
        strip_data = np.ascontiguousarray(data[r0:r1]).tobytes()

    if compression == COMPRESSION_JPEG2000:
        from ._compression import jpeg2000_compress
        return jpeg2000_compress(
            strip_data, width, strip_rows, samples=samples, dtype=dtype)
    if compression == COMPRESSION_LERC:
        from ._compression import lerc_compress
        return lerc_compress(
            strip_data, width, strip_rows, samples=samples, dtype=dtype,
            max_z_error=max_z_error)
    if compression_level is None:
        return compress(strip_data, compression, gil_friendly=gil_friendly)
    return compress(strip_data, compression, level=compression_level,
                    gil_friendly=gil_friendly)


def _write_stripped(data: np.ndarray, compression: int, predictor: int,
                    rows_per_strip: int = 256,
                    compression_level: int | None = None,
                    max_z_error: float = 0.0) -> tuple[list, list, list]:
    """Compress data as strips.

    For compressed formats (deflate, lzw, zstd, lz4, ...) strips are
    compressed in parallel using a thread pool: zlib, zstandard, lz4,
    and the Numba LZW kernel all release the GIL during compression.

    Returns
    -------
    (offsets_placeholder, byte_counts, compressed_chunks)
        offsets are relative to the start of the compressed data block.
        compressed_chunks is a list of bytes objects (one per strip).
    """
    # Look up ``_prepare_strip`` via ``_writer`` so tests that
    # monkeypatch ``xrspatial.geotiff._writer._prepare_strip`` still
    # observe the override. ``_writer`` re-exports the name from this
    # module, so the indirection is a no-op outside of patched tests.
    # See ``test_gil_friendly_kwarg_1830`` for the matching contract
    # on the tile path (``_prepare_tile``); the same shape applies here.
    from . import _writer as _writer_mod
    _prepare_strip_fn = _writer_mod._prepare_strip

    height, width = data.shape[:2]
    samples = data.shape[2] if data.ndim == 3 else 1
    dtype = data.dtype
    bytes_per_sample = dtype.itemsize

    num_strips = math.ceil(height / rows_per_strip)

    total_bytes = int(data.nbytes)

    # Sequential path: uncompressed, few strips, or small payload.  The
    # threshold mirrors the tile writer so we don't pay thread-pool
    # overhead on tiny rasters.
    use_parallel = (
        compression != COMPRESSION_NONE
        and num_strips > 2
        and total_bytes > _PARALLEL_MIN_BYTES
    )

    if not use_parallel:
        strips = []
        rel_offsets = []
        byte_counts = []
        current_offset = 0
        for i in range(num_strips):
            compressed = _prepare_strip_fn(
                data, i, rows_per_strip, height, width, samples, dtype,
                bytes_per_sample, predictor, compression,
                compression_level, max_z_error,
            )
            rel_offsets.append(current_offset)
            byte_counts.append(len(compressed))
            strips.append(compressed)
            current_offset += len(compressed)
        return rel_offsets, byte_counts, strips

    # Parallel strip compression -- zlib/zstd/lz4/LZW all release the GIL.
    # ``gil_friendly=True`` keeps deflate on stdlib zlib here: the
    # ``deflate`` (libdeflate) binding holds the GIL during compress, so
    # 8 threads run effectively serially through it. Sequential callers
    # still get libdeflate's per-call speedup (~3x).
    import os
    from concurrent.futures import ThreadPoolExecutor

    n_workers = min(num_strips, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        compressed_strips = list(pool.map(
            lambda i: _prepare_strip_fn(
                data, i, rows_per_strip, height, width, samples, dtype,
                bytes_per_sample, predictor, compression,
                compression_level, max_z_error,
                gil_friendly=True,
            ),
            range(num_strips),
        ))

    rel_offsets = []
    byte_counts = []
    current_offset = 0
    for cs in compressed_strips:
        rel_offsets.append(current_offset)
        byte_counts.append(len(cs))
        current_offset += len(cs)

    return rel_offsets, byte_counts, compressed_strips


# ---------------------------------------------------------------------------
# Tile writer
# ---------------------------------------------------------------------------

def _prepare_tile(data, tr, tc, th, tw, height, width, samples, dtype,
                  bytes_per_sample, predictor: int, compression,
                  compression_level=None, max_z_error: float = 0.0,
                  gil_friendly: bool = False):
    """Extract, pad, and compress a single tile.  Thread-safe."""
    r0 = tr * th
    c0 = tc * tw
    r1 = min(r0 + th, height)
    c1 = min(c0 + tw, width)
    actual_h = r1 - r0
    actual_w = c1 - c0

    tile_slice = data[r0:r1, c0:c1]

    if actual_h < th or actual_w < tw:
        if data.ndim == 3:
            padded = np.empty((th, tw, samples), dtype=dtype)
        else:
            padded = np.empty((th, tw), dtype=dtype)
        padded[:actual_h, :actual_w] = tile_slice
        if actual_h < th:
            padded[actual_h:, :] = 0
        if actual_w < tw:
            padded[:actual_h, actual_w:] = 0
        tile_arr = padded
    else:
        tile_arr = np.ascontiguousarray(tile_slice)

    if compression == COMPRESSION_JPEG:
        tile_data = tile_arr.tobytes()
        return jpeg_compress(tile_data, tw, th, samples)
    elif predictor != 1 and compression != COMPRESSION_NONE:
        buf = tile_arr.view(np.uint8).ravel().copy()
        buf = _apply_predictor_encode(
            buf, predictor, tw, th, bytes_per_sample, samples)
        tile_data = buf.tobytes()
    else:
        tile_data = tile_arr.tobytes()

    if compression == COMPRESSION_JPEG2000:
        from ._compression import jpeg2000_compress
        return jpeg2000_compress(
            tile_data, tw, th, samples=samples, dtype=dtype)
    if compression == COMPRESSION_LERC:
        from ._compression import lerc_compress
        return lerc_compress(
            tile_data, tw, th, samples=samples, dtype=dtype,
            max_z_error=max_z_error)
    if compression_level is None:
        return compress(tile_data, compression, gil_friendly=gil_friendly)
    return compress(tile_data, compression, level=compression_level,
                    gil_friendly=gil_friendly)


def _write_tiled(data: np.ndarray, compression: int, predictor: int,
                 tile_size: int = 256,
                 compression_level: int | None = None,
                 max_z_error: float = 0.0) -> tuple[list, list, list]:
    """Compress data as tiles, using parallel compression.

    For compressed formats (deflate, lzw, zstd), tiles are compressed
    in parallel using a thread pool.  zlib, zstandard, and our Numba
    LZW all release the GIL.

    Returns
    -------
    (relative_offsets, byte_counts, compressed_chunks)
        compressed_chunks is a list of bytes objects (one per tile).
    """
    # Look up ``_prepare_tile`` via ``_writer`` so tests that
    # monkeypatch ``xrspatial.geotiff._writer._prepare_tile`` still
    # observe the override. The pinned contract lives in
    # ``test_gil_friendly_kwarg_1830::
    # test_write_tiled_parallel_passes_gil_friendly_positionally``.
    # ``_writer`` re-exports the name from this module, so the
    # indirection is a no-op outside of patched tests.
    from . import _writer as _writer_mod
    _prepare_tile_fn = _writer_mod._prepare_tile

    height, width = data.shape[:2]
    samples = data.shape[2] if data.ndim == 3 else 1
    dtype = data.dtype
    bytes_per_sample = dtype.itemsize

    tw = tile_size
    th = tile_size
    tiles_across = math.ceil(width / tw)
    tiles_down = math.ceil(height / th)
    n_tiles = tiles_across * tiles_down

    if compression == COMPRESSION_NONE:
        # Uncompressed: build tiles one at a time. An earlier version
        # pre-allocated a contiguous ``bytearray(n_tiles * tile_bytes)``
        # buffer here on the theory that we'd copy each tile into it
        # directly, but the loop below ended up calling ``tobytes()``
        # per tile anyway and never read the buffer. That left a dead
        # allocation roughly the size of the full uncompressed raster
        # alongside the actual tile list, doubling peak memory and
        # turning OOM-marginal writes into OOM-failing ones (#1736).
        tiles = []
        rel_offsets = []
        byte_counts = []
        current_offset = 0

        for tr in range(tiles_down):
            for tc in range(tiles_across):
                r0 = tr * th
                c0 = tc * tw
                r1 = min(r0 + th, height)
                c1 = min(c0 + tw, width)
                actual_h = r1 - r0
                actual_w = c1 - c0

                tile_slice = data[r0:r1, c0:c1]
                if actual_h < th or actual_w < tw:
                    if data.ndim == 3:
                        padded = np.zeros((th, tw, samples), dtype=dtype)
                    else:
                        padded = np.zeros((th, tw), dtype=dtype)
                    padded[:actual_h, :actual_w] = tile_slice
                    tile_arr = padded
                else:
                    tile_arr = np.ascontiguousarray(tile_slice)

                chunk = tile_arr.tobytes()
                rel_offsets.append(current_offset)
                byte_counts.append(len(chunk))
                tiles.append(chunk)
                current_offset += len(chunk)

        return rel_offsets, byte_counts, tiles

    # Sequential path: very few tiles, or small total payload. A previous
    # ``n_tiles <= 4`` cutoff sent ``tile_size=1024`` writes on a 2048x2048
    # image down the serial path (n_tiles=4) and made them ~8x slower than
    # the parallel path. Switching to a bytes-based threshold lets
    # large-tile writes parallelize while still skipping the pool on
    # small rasters where its setup cost dominates.
    if n_tiles <= 2 or int(data.nbytes) <= _PARALLEL_MIN_BYTES:
        tiles = []
        rel_offsets = []
        byte_counts = []
        current_offset = 0
        for tr in range(tiles_down):
            for tc in range(tiles_across):
                compressed = _prepare_tile_fn(
                    data, tr, tc, th, tw, height, width,
                    samples, dtype, bytes_per_sample, predictor, compression,
                    compression_level, max_z_error,
                )
                rel_offsets.append(current_offset)
                byte_counts.append(len(compressed))
                tiles.append(compressed)
                current_offset += len(compressed)
        return rel_offsets, byte_counts, tiles

    # Parallel tile compression -- zlib/zstd/LZW all release the GIL
    import os
    from concurrent.futures import ThreadPoolExecutor

    n_workers = min(n_tiles, os.cpu_count() or 4)
    tile_indices = [(tr, tc) for tr in range(tiles_down)
                    for tc in range(tiles_across)]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _prepare_tile_fn, data, tr, tc, th, tw, height, width,
                samples, dtype, bytes_per_sample, predictor, compression,
                compression_level, max_z_error, True,
            )
            for tr, tc in tile_indices
        ]
        compressed_tiles = [f.result() for f in futures]

    rel_offsets = []
    byte_counts = []
    current_offset = 0
    for ct in compressed_tiles:
        rel_offsets.append(current_offset)
        byte_counts.append(len(ct))
        current_offset += len(ct)

    return rel_offsets, byte_counts, compressed_tiles


# ---------------------------------------------------------------------------
# Streaming block compression
# ---------------------------------------------------------------------------

def _compress_block(arr, block_w, block_h, samples, dtype, bytes_per_sample,
                    predictor: int, compression, compression_level=None,
                    max_z_error: float = 0.0, gil_friendly: bool = False):
    """Compress a tile or strip.  *arr* must be contiguous and correctly sized."""
    if compression == COMPRESSION_JPEG:
        return jpeg_compress(arr.tobytes(), block_w, block_h, samples)

    if predictor != 1 and compression != COMPRESSION_NONE:
        buf = arr.view(np.uint8).ravel().copy()
        buf = _apply_predictor_encode(
            buf, predictor, block_w, block_h, bytes_per_sample, samples)
        raw_data = buf.tobytes()
    else:
        raw_data = arr.tobytes()

    if compression == COMPRESSION_JPEG2000:
        from ._compression import jpeg2000_compress
        return jpeg2000_compress(raw_data, block_w, block_h,
                                 samples=samples, dtype=dtype)
    if compression == COMPRESSION_LERC:
        from ._compression import lerc_compress
        return lerc_compress(raw_data, block_w, block_h,
                             samples=samples, dtype=dtype,
                             max_z_error=max_z_error)
    if compression_level is None:
        return compress(raw_data, compression, gil_friendly=gil_friendly)
    return compress(raw_data, compression, level=compression_level,
                    gil_friendly=gil_friendly)
