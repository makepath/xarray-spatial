"""GeoTIFF/COG writer."""
from __future__ import annotations

import math
import struct

import numpy as np

from ._compression import (
    COMPRESSION_DEFLATE,
    COMPRESSION_JPEG,
    COMPRESSION_JPEG2000,
    COMPRESSION_LERC,
    COMPRESSION_LZ4,
    COMPRESSION_LZW,
    COMPRESSION_NONE,
    COMPRESSION_PACKBITS,
    COMPRESSION_ZSTD,
    compress,
    jpeg_compress,
    predictor_encode,
)
from ._dtypes import (
    DOUBLE,
    RATIONAL,
    SHORT,
    LONG,
    ASCII,
    numpy_to_tiff_dtype,
    TIFF_TYPE_SIZES,
)
from ._geotags import (
    GeoTransform,
    build_geo_tags,
    TAG_GEO_KEY_DIRECTORY,
    TAG_GDAL_NODATA,
    TAG_MODEL_PIXEL_SCALE,
    TAG_MODEL_TIEPOINT,
)
from ._header import (
    TAG_IMAGE_WIDTH,
    TAG_IMAGE_LENGTH,
    TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION,
    TAG_PHOTOMETRIC,
    TAG_SAMPLES_PER_PIXEL,
    TAG_SAMPLE_FORMAT,
    TAG_STRIP_OFFSETS,
    TAG_ROWS_PER_STRIP,
    TAG_STRIP_BYTE_COUNTS,
    TAG_X_RESOLUTION,
    TAG_Y_RESOLUTION,
    TAG_RESOLUTION_UNIT,
    TAG_TILE_WIDTH,
    TAG_TILE_LENGTH,
    TAG_TILE_OFFSETS,
    TAG_TILE_BYTE_COUNTS,
    TAG_EXTRA_SAMPLES,
    TAG_PREDICTOR,
    TAG_GDAL_METADATA,
)

# Byte order: always write little-endian
BO = '<'


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


OVERVIEW_METHODS = ('mean', 'nearest', 'min', 'max', 'median', 'mode', 'cubic')


def _block_reduce_2d(arr2d, method):
    """2x block-reduce a single 2D plane using *method*."""
    h, w = arr2d.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    cropped = arr2d[:h2, :w2]
    oh, ow = h2 // 2, w2 // 2

    if method == 'nearest':
        # Top-left pixel of each 2x2 block
        return cropped[::2, ::2].copy()

    if method == 'cubic':
        try:
            from scipy.ndimage import zoom
        except ImportError:
            raise ImportError(
                "scipy is required for cubic overview resampling. "
                "Install it with: pip install scipy")
        return zoom(arr2d, 0.5, order=3).astype(arr2d.dtype)

    if method == 'mode':
        # Most-common value per 2x2 block (useful for classified rasters)
        blocks = cropped.reshape(oh, 2, ow, 2).transpose(0, 2, 1, 3).reshape(oh, ow, 4)
        out = np.empty((oh, ow), dtype=arr2d.dtype)
        for r in range(oh):
            for c in range(ow):
                vals, counts = np.unique(blocks[r, c], return_counts=True)
                out[r, c] = vals[counts.argmax()]
        return out

    # Block reshape for mean/min/max/median
    if arr2d.dtype.kind == 'f':
        blocks = cropped.reshape(oh, 2, ow, 2)
    else:
        blocks = cropped.astype(np.float64).reshape(oh, 2, ow, 2)

    if method == 'mean':
        result = np.nanmean(blocks, axis=(1, 3))
    elif method == 'min':
        result = np.nanmin(blocks, axis=(1, 3))
    elif method == 'max':
        result = np.nanmax(blocks, axis=(1, 3))
    elif method == 'median':
        flat = blocks.transpose(0, 2, 1, 3).reshape(oh, ow, 4)
        result = np.nanmedian(flat, axis=2)
    else:
        raise ValueError(
            f"Unknown overview resampling method: {method!r}. "
            f"Use one of: {OVERVIEW_METHODS}")

    if arr2d.dtype.kind != 'f':
        return np.round(result).astype(arr2d.dtype)
    return result.astype(arr2d.dtype)


def _make_overview(arr: np.ndarray, method: str = 'mean') -> np.ndarray:
    """Generate a 2x decimated overview.

    Parameters
    ----------
    arr : np.ndarray
        2D or 3D (height, width, bands) array.
    method : str
        Resampling method: 'mean' (default), 'nearest', 'min', 'max',
        'median', 'mode', or 'cubic'.

    Returns
    -------
    np.ndarray
        Half-resolution array.
    """
    if arr.ndim == 3:
        bands = [_block_reduce_2d(arr[:, :, b], method) for b in range(arr.shape[2])]
        return np.stack(bands, axis=2)
    return _block_reduce_2d(arr, method)


# ---------------------------------------------------------------------------
# Tag serialization
# ---------------------------------------------------------------------------

def _float_to_rational(val):
    """Convert a float to a TIFF RATIONAL (numerator, denominator) pair."""
    if val == int(val):
        return (int(val), 1)
    # Use a denominator of 10000 for reasonable precision
    den = 10000
    num = int(round(val * den))
    return (num, den)


def _serialize_tag_value(type_id, count, values):
    """Serialize tag values to bytes."""
    if type_id == ASCII:
        if isinstance(values, str):
            return values.encode('ascii') + b'\x00'
        return values + b'\x00'
    elif type_id == SHORT:
        if isinstance(values, (list, tuple)):
            return struct.pack(f'{BO}{count}H', *values)
        return struct.pack(f'{BO}H', values)
    elif type_id == LONG:
        if isinstance(values, (list, tuple)):
            return struct.pack(f'{BO}{count}I', *values)
        return struct.pack(f'{BO}I', values)
    elif type_id == RATIONAL:
        # RATIONAL = two LONGs (numerator, denominator) per value
        if isinstance(values, (list, tuple)) and isinstance(values[0], (list, tuple)):
            parts = []
            for num, den in values:
                parts.extend([int(num), int(den)])
            return struct.pack(f'{BO}{count * 2}I', *parts)
        else:
            num, den = _float_to_rational(float(values))
            return struct.pack(f'{BO}II', num, den)
    elif type_id == DOUBLE:
        if isinstance(values, (list, tuple)):
            return struct.pack(f'{BO}{count}d', *values)
        return struct.pack(f'{BO}d', values)
    else:
        if isinstance(values, bytes):
            return values
        return struct.pack(f'{BO}I', values)


def _pack_tag_value(tag_id: int, type_id: int, count: int,
                    values, overflow_buf: bytearray,
                    overflow_base: int, bigtiff: bool = False) -> bytes:
    """Pack a single IFD entry.

    Standard TIFF: 12 bytes (tag:2, type:2, count:4, value:4).
    BigTIFF: 20 bytes (tag:2, type:2, count:8, value:8).
    """
    val_bytes = _serialize_tag_value(type_id, count, values)

    # For ASCII, count is the actual byte length
    if type_id == ASCII:
        count = len(val_bytes)

    inline_max = 8 if bigtiff else 4

    if bigtiff:
        entry = struct.pack(f'{BO}HHQ', tag_id, type_id, count)
    else:
        entry = struct.pack(f'{BO}HHI', tag_id, type_id, count)

    if len(val_bytes) <= inline_max:
        value_field = val_bytes.ljust(inline_max, b'\x00')
    else:
        offset = overflow_base + len(overflow_buf)
        if bigtiff:
            value_field = struct.pack(f'{BO}Q', offset)
        else:
            value_field = struct.pack(f'{BO}I', offset)
        overflow_buf.extend(val_bytes)
        if len(overflow_buf) % 2:
            overflow_buf.append(0)

    return entry + value_field


def _build_ifd(tags: list[tuple], overflow_base: int,
               bigtiff: bool = False) -> tuple[bytes, bytes]:
    """Build a complete IFD block.

    Parameters
    ----------
    tags : list of (tag_id, type_id, count, values)
        Tags sorted by tag_id.
    overflow_base : int
        Where overflow data starts in the file.

    Returns
    -------
    (ifd_bytes, overflow_bytes)
    """
    # Sort by tag ID (TIFF spec requires this)
    tags = sorted(tags, key=lambda t: t[0])

    num_entries = len(tags)
    overflow_buf = bytearray()

    if bigtiff:
        ifd_parts = [struct.pack(f'{BO}Q', num_entries)]
    else:
        ifd_parts = [struct.pack(f'{BO}H', num_entries)]

    for tag_id, type_id, count, values in tags:
        entry = _pack_tag_value(tag_id, type_id, count, values,
                                overflow_buf, overflow_base, bigtiff=bigtiff)
        ifd_parts.append(entry)

    # Next IFD offset (0 = no more IFDs, will be patched for COG)
    if bigtiff:
        ifd_parts.append(struct.pack(f'{BO}Q', 0))
    else:
        ifd_parts.append(struct.pack(f'{BO}I', 0))

    return b''.join(ifd_parts), bytes(overflow_buf)


# ---------------------------------------------------------------------------
# Strip writer
# ---------------------------------------------------------------------------

def _write_stripped(data: np.ndarray, compression: int, predictor: bool,
                    rows_per_strip: int = 256) -> tuple[list, list, list]:
    """Compress data as strips.

    Returns
    -------
    (offsets_placeholder, byte_counts, compressed_chunks)
        offsets are relative to the start of the compressed data block.
        compressed_chunks is a list of bytes objects (one per strip).
    """
    height, width = data.shape[:2]
    samples = data.shape[2] if data.ndim == 3 else 1
    dtype = data.dtype
    bytes_per_sample = dtype.itemsize

    strips = []
    rel_offsets = []
    byte_counts = []
    current_offset = 0

    num_strips = math.ceil(height / rows_per_strip)
    for i in range(num_strips):
        r0 = i * rows_per_strip
        r1 = min(r0 + rows_per_strip, height)
        strip_rows = r1 - r0

        if compression == COMPRESSION_JPEG:
            strip_data = np.ascontiguousarray(data[r0:r1]).tobytes()
            compressed = jpeg_compress(strip_data, width, strip_rows, samples)
        elif predictor and compression != COMPRESSION_NONE:
            strip_arr = np.ascontiguousarray(data[r0:r1])
            buf = strip_arr.view(np.uint8).ravel().copy()
            buf = predictor_encode(buf, width, strip_rows, bytes_per_sample * samples)
            strip_data = buf.tobytes()
            compressed = compress(strip_data, compression)
        else:
            strip_data = np.ascontiguousarray(data[r0:r1]).tobytes()

            if compression == COMPRESSION_JPEG2000:
                from ._compression import jpeg2000_compress
                compressed = jpeg2000_compress(
                    strip_data, width, strip_rows, samples=samples, dtype=dtype)
            elif compression == COMPRESSION_LERC:
                from ._compression import lerc_compress
                compressed = lerc_compress(
                    strip_data, width, strip_rows, samples=samples, dtype=dtype)
            else:
                compressed = compress(strip_data, compression)

        rel_offsets.append(current_offset)
        byte_counts.append(len(compressed))
        strips.append(compressed)
        current_offset += len(compressed)

    return rel_offsets, byte_counts, strips


# ---------------------------------------------------------------------------
# Tile writer
# ---------------------------------------------------------------------------

def _prepare_tile(data, tr, tc, th, tw, height, width, samples, dtype,
                  bytes_per_sample, predictor, compression):
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
    elif predictor and compression != COMPRESSION_NONE:
        buf = tile_arr.view(np.uint8).ravel().copy()
        buf = predictor_encode(buf, tw, th, bytes_per_sample * samples)
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
            tile_data, tw, th, samples=samples, dtype=dtype)
    return compress(tile_data, compression)


def _write_tiled(data: np.ndarray, compression: int, predictor: bool,
                 tile_size: int = 256) -> tuple[list, list, list]:
    """Compress data as tiles, using parallel compression.

    For compressed formats (deflate, lzw, zstd), tiles are compressed
    in parallel using a thread pool.  zlib, zstandard, and our Numba
    LZW all release the GIL.

    Returns
    -------
    (relative_offsets, byte_counts, compressed_chunks)
        compressed_chunks is a list of bytes objects (one per tile).
    """
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
        # Uncompressed: pre-allocate a contiguous buffer for all tiles
        # and copy tile data directly, avoiding per-tile Python overhead.
        tile_bytes = tw * th * bytes_per_sample * samples
        total_buf = bytearray(n_tiles * tile_bytes)
        mv = memoryview(total_buf)
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

    if n_tiles <= 4:
        # Very few tiles: sequential (thread pool overhead not worth it)
        tiles = []
        rel_offsets = []
        byte_counts = []
        current_offset = 0
        for tr in range(tiles_down):
            for tc in range(tiles_across):
                compressed = _prepare_tile(
                    data, tr, tc, th, tw, height, width,
                    samples, dtype, bytes_per_sample, predictor, compression,
                )
                rel_offsets.append(current_offset)
                byte_counts.append(len(compressed))
                tiles.append(compressed)
                current_offset += len(compressed)
        return rel_offsets, byte_counts, tiles

    # Parallel tile compression -- zlib/zstd/LZW all release the GIL
    from concurrent.futures import ThreadPoolExecutor
    import os

    n_workers = min(n_tiles, os.cpu_count() or 4)
    tile_indices = [(tr, tc) for tr in range(tiles_down)
                    for tc in range(tiles_across)]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _prepare_tile, data, tr, tc, th, tw, height, width,
                samples, dtype, bytes_per_sample, predictor, compression,
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
# File assembly
# ---------------------------------------------------------------------------

def _assemble_tiff(width: int, height: int, dtype: np.dtype,
                   compression: int, predictor: bool,
                   tiled: bool, tile_size: int,
                   pixel_data_parts: list[tuple],
                   geo_transform: GeoTransform | None,
                   crs_epsg: int | None,
                   nodata,
                   is_cog: bool = False,
                   raster_type: int = 1,
                   gdal_metadata_xml: str | None = None,
                   extra_tags: list | None = None,
                   x_resolution: float | None = None,
                   y_resolution: float | None = None,
                   resolution_unit: int | None = None,
                   force_bigtiff: bool | None = None) -> bytes:
    """Assemble a complete TIFF file.

    Parameters
    ----------
    pixel_data_parts : list of (array, width, height, relative_offsets, byte_counts, compressed_data)
        One entry per resolution level (full res first, then overviews).
    is_cog : bool
        If True, layout IFDs contiguously at file start (COG layout).
    raster_type : int
        1 = PixelIsArea, 2 = PixelIsPoint.

    Returns
    -------
    bytes
        Complete TIFF file.
    """
    bits_per_sample, sample_format = numpy_to_tiff_dtype(dtype)

    # Determine samples per pixel from the pixel data
    first_arr = pixel_data_parts[0][0]
    samples_per_pixel = first_arr.shape[2] if first_arr.ndim == 3 else 1

    # Build geo tags
    geo_tags_dict = {}
    if geo_transform is not None:
        geo_tags_dict = build_geo_tags(
            geo_transform, crs_epsg, nodata, raster_type=raster_type)
    else:
        # No spatial reference -- still write CRS and nodata if provided
        if crs_epsg is not None or nodata is not None:
            geo_tags_dict = build_geo_tags(
                GeoTransform(), crs_epsg, nodata, raster_type=raster_type,
            )
            # Remove the default pixel scale / tiepoint tags since we
            # have no real transform -- keep only GeoKeys and NODATA.
            geo_tags_dict.pop(TAG_MODEL_PIXEL_SCALE, None)
            geo_tags_dict.pop(TAG_MODEL_TIEPOINT, None)

    # Compression tag for predictor
    pred_val = 2 if (predictor and compression != COMPRESSION_NONE) else 1

    # Build IFDs for each resolution level
    ifd_specs = []
    for level_idx, (arr, lw, lh, rel_offsets, byte_counts, comp_data) in enumerate(pixel_data_parts):
        tags = []

        tags.append((TAG_IMAGE_WIDTH, LONG, 1, lw))
        tags.append((TAG_IMAGE_LENGTH, LONG, 1, lh))
        if samples_per_pixel > 1:
            tags.append((TAG_BITS_PER_SAMPLE, SHORT, samples_per_pixel,
                         [bits_per_sample] * samples_per_pixel))
        else:
            tags.append((TAG_BITS_PER_SAMPLE, SHORT, 1, bits_per_sample))
        tags.append((TAG_COMPRESSION, SHORT, 1, compression))
        # Photometric: RGB for 3+ bands, BlackIsZero for single-band
        photometric = 2 if samples_per_pixel >= 3 else 1
        tags.append((TAG_PHOTOMETRIC, SHORT, 1, photometric))
        tags.append((TAG_SAMPLES_PER_PIXEL, SHORT, 1, samples_per_pixel))
        if samples_per_pixel > 1:
            tags.append((TAG_SAMPLE_FORMAT, SHORT, samples_per_pixel,
                         [sample_format] * samples_per_pixel))
        else:
            tags.append((TAG_SAMPLE_FORMAT, SHORT, 1, sample_format))

        # ExtraSamples: for bands beyond what Photometric accounts for
        # Photometric=2 (RGB) accounts for 3 bands; any extra are alpha/other
        if photometric == 2 and samples_per_pixel > 3:
            n_extra = samples_per_pixel - 3
            # 2 = unassociated alpha for the first extra, 0 = unspecified for rest
            extra_vals = [2] + [0] * (n_extra - 1)
            tags.append((TAG_EXTRA_SAMPLES, SHORT, n_extra, extra_vals))
        elif photometric == 1 and samples_per_pixel > 1:
            n_extra = samples_per_pixel - 1
            extra_vals = [0] * n_extra  # unspecified
            tags.append((TAG_EXTRA_SAMPLES, SHORT, n_extra, extra_vals))

        if pred_val != 1:
            tags.append((TAG_PREDICTOR, SHORT, 1, pred_val))

        # Resolution / DPI tags
        if x_resolution is not None:
            tags.append((TAG_X_RESOLUTION, RATIONAL, 1, x_resolution))
        if y_resolution is not None:
            tags.append((TAG_Y_RESOLUTION, RATIONAL, 1, y_resolution))
        if resolution_unit is not None:
            tags.append((TAG_RESOLUTION_UNIT, SHORT, 1, resolution_unit))

        if tiled:
            tags.append((TAG_TILE_WIDTH, SHORT, 1, tile_size))
            tags.append((TAG_TILE_LENGTH, SHORT, 1, tile_size))
            # Placeholder offsets/counts -- will be patched
            tags.append((TAG_TILE_OFFSETS, LONG, len(rel_offsets), rel_offsets))
            tags.append((TAG_TILE_BYTE_COUNTS, LONG, len(byte_counts), byte_counts))
        else:
            rows_per_strip = 256
            if lh <= rows_per_strip:
                rows_per_strip = lh
            tags.append((TAG_ROWS_PER_STRIP, SHORT, 1, rows_per_strip))
            tags.append((TAG_STRIP_OFFSETS, LONG, len(rel_offsets), rel_offsets))
            tags.append((TAG_STRIP_BYTE_COUNTS, LONG, len(byte_counts), byte_counts))

        # Geo tags only on first IFD
        if level_idx == 0:
            for gtag, gval in geo_tags_dict.items():
                if gtag == TAG_MODEL_PIXEL_SCALE:
                    tags.append((gtag, DOUBLE, 3, list(gval)))
                elif gtag == TAG_MODEL_TIEPOINT:
                    tags.append((gtag, DOUBLE, 6, list(gval)))
                elif gtag == TAG_GEO_KEY_DIRECTORY:
                    tags.append((gtag, SHORT, len(gval), list(gval)))
                elif gtag == TAG_GDAL_NODATA:
                    tags.append((gtag, ASCII, len(str(gval)) + 1, str(gval)))

            # GDALMetadata XML (tag 42112)
            if gdal_metadata_xml is not None:
                tags.append((TAG_GDAL_METADATA, ASCII,
                             len(gdal_metadata_xml) + 1, gdal_metadata_xml))

            # Extra tags (pass-through from source file)
            if extra_tags is not None:
                for etag_id, etype_id, ecount, evalue in extra_tags:
                    # Skip any tag we already wrote to avoid duplicates
                    existing_ids = {t[0] for t in tags}
                    if etag_id not in existing_ids:
                        tags.append((etag_id, etype_id, ecount, evalue))

        ifd_specs.append(tags)

    # --- Determine if BigTIFF is needed ---
    # Classic TIFF uses 32-bit offsets (max ~4.29 GB). Estimate total file
    # size including headers, IFDs, overflow data, and all pixel data.
    # Switch to BigTIFF if any offset could exceed 2^32.
    total_pixel_data = sum(sum(len(c) for c in chunks)
                           for _, _, _, _, _, chunks in pixel_data_parts)
    # Conservative overhead estimate: header + IFDs + overflow + geo tags
    num_levels = len(ifd_specs)
    max_tags_per_ifd = max(len(tags) for tags in ifd_specs) if ifd_specs else 20
    ifd_overhead = num_levels * (2 + 12 * max_tags_per_ifd + 4 + 1024)  # ~1KB overflow per IFD
    estimated_file_size = 8 + ifd_overhead + total_pixel_data

    UINT32_MAX = 0xFFFFFFFF  # 4,294,967,295
    if force_bigtiff is not None:
        bigtiff = force_bigtiff
    else:
        bigtiff = estimated_file_size > UINT32_MAX

    header_size = 16 if bigtiff else 8

    if is_cog and len(ifd_specs) > 1:
        return _assemble_cog_layout(header_size, ifd_specs, pixel_data_parts,
                                    bigtiff=bigtiff)
    else:
        return _assemble_standard_layout(header_size, ifd_specs, pixel_data_parts,
                                         bigtiff=bigtiff)


def _assemble_standard_layout(header_size: int,
                              ifd_specs: list,
                              pixel_data_parts: list,
                              bigtiff: bool = False) -> bytes:
    """Assemble standard TIFF layout (one IFD at a time)."""
    output = bytearray()
    entry_size = 20 if bigtiff else 12

    # TIFF header
    output.extend(b'II')  # little-endian
    if bigtiff:
        output.extend(struct.pack(f'{BO}H', 43))   # BigTIFF magic
        output.extend(struct.pack(f'{BO}H', 8))    # offset size
        output.extend(struct.pack(f'{BO}H', 0))    # padding
        output.extend(struct.pack(f'{BO}Q', 0))    # first IFD offset placeholder
    else:
        output.extend(struct.pack(f'{BO}H', 42))   # magic
        output.extend(struct.pack(f'{BO}I', 0))    # first IFD offset placeholder

    for level_idx, (tags, (_arr, _lw, _lh, rel_offsets, byte_counts, comp_chunks)) in enumerate(
            zip(ifd_specs, pixel_data_parts)):

        ifd_offset = len(output)

        if level_idx == 0:
            if bigtiff:
                struct.pack_into(f'{BO}Q', output, 8, ifd_offset)
            else:
                struct.pack_into(f'{BO}I', output, 4, ifd_offset)

        num_entries = len(tags)
        count_size = 8 if bigtiff else 2
        next_size = 8 if bigtiff else 4
        ifd_block_size = count_size + entry_size * num_entries + next_size
        overflow_base = ifd_offset + ifd_block_size

        ifd_bytes, overflow_bytes = _build_ifd(tags, overflow_base, bigtiff=bigtiff)

        pixel_data_offset = overflow_base + len(overflow_bytes)

        patched_tags = []
        for tag_id, type_id, count, values in tags:
            if tag_id in (TAG_STRIP_OFFSETS, TAG_TILE_OFFSETS):
                actual_offsets = [pixel_data_offset + ro for ro in rel_offsets]
                patched_tags.append((tag_id, type_id, count, actual_offsets))
            else:
                patched_tags.append((tag_id, type_id, count, values))

        ifd_bytes, overflow_bytes = _build_ifd(patched_tags, overflow_base,
                                                bigtiff=bigtiff)

        output.extend(ifd_bytes)
        output.extend(overflow_bytes)
        # Extend directly from chunk list (no intermediate join copy)
        for chunk in comp_chunks:
            output.extend(chunk)

        # Patch next IFD pointer if there are more levels
        if level_idx < len(ifd_specs) - 1:
            next_ifd_offset = len(output)
            next_ptr_pos = ifd_offset + count_size + entry_size * num_entries
            if bigtiff:
                struct.pack_into(f'{BO}Q', output, next_ptr_pos, next_ifd_offset)
            else:
                struct.pack_into(f'{BO}I', output, next_ptr_pos, next_ifd_offset)

    return bytes(output)


def _assemble_cog_layout(header_size: int,
                         ifd_specs: list,
                         pixel_data_parts: list,
                         bigtiff: bool = False) -> bytes:
    """Assemble COG layout: all IFDs first, then all pixel data."""
    entry_size = 20 if bigtiff else 12
    count_size = 8 if bigtiff else 2
    next_size = 8 if bigtiff else 4

    # First pass: compute IFD sizes
    ifd_blocks = []
    for tags in ifd_specs:
        num_entries = len(tags)
        ifd_block_size = count_size + entry_size * num_entries + next_size
        _, overflow = _build_ifd(tags, 0, bigtiff=bigtiff)
        ifd_blocks.append((ifd_block_size, len(overflow)))

    total_ifd_size = sum(bs + ov for bs, ov in ifd_blocks)
    pixel_data_start = header_size + total_ifd_size

    # Second pass: pixel data offsets per level
    current_pixel_offset = pixel_data_start
    level_pixel_offsets = []
    for _arr, _lw, _lh, rel_offsets, byte_counts, comp_chunks in pixel_data_parts:
        level_pixel_offsets.append(current_pixel_offset)
        current_pixel_offset += sum(len(c) for c in comp_chunks)

    # Third pass: build IFDs with correct offsets
    output = bytearray()
    output.extend(b'II')
    if bigtiff:
        output.extend(struct.pack(f'{BO}H', 43))
        output.extend(struct.pack(f'{BO}H', 8))
        output.extend(struct.pack(f'{BO}H', 0))
        output.extend(struct.pack(f'{BO}Q', header_size))
    else:
        output.extend(struct.pack(f'{BO}H', 42))
        output.extend(struct.pack(f'{BO}I', header_size))

    current_ifd_pos = header_size
    for level_idx, (tags, (_arr, _lw, _lh, rel_offsets, byte_counts, comp_chunks)) in enumerate(
            zip(ifd_specs, pixel_data_parts)):

        pixel_base = level_pixel_offsets[level_idx]

        patched_tags = []
        for tag_id, type_id, count, values in tags:
            if tag_id in (TAG_STRIP_OFFSETS, TAG_TILE_OFFSETS):
                actual_offsets = [pixel_base + ro for ro in rel_offsets]
                patched_tags.append((tag_id, type_id, count, actual_offsets))
            else:
                patched_tags.append((tag_id, type_id, count, values))

        num_entries = len(patched_tags)
        ifd_block_size = count_size + entry_size * num_entries + next_size
        overflow_base = current_ifd_pos + ifd_block_size

        ifd_bytes, overflow_bytes = _build_ifd(patched_tags, overflow_base,
                                                bigtiff=bigtiff)

        # Patch next IFD offset
        if level_idx < len(ifd_specs) - 1:
            next_ifd_pos = current_ifd_pos + ifd_block_size + len(overflow_bytes)
            ifd_ba = bytearray(ifd_bytes)
            next_ptr_pos = count_size + entry_size * num_entries
            if bigtiff:
                struct.pack_into(f'{BO}Q', ifd_ba, next_ptr_pos, next_ifd_pos)
            else:
                struct.pack_into(f'{BO}I', ifd_ba, next_ptr_pos, next_ifd_pos)
            ifd_bytes = bytes(ifd_ba)

        output.extend(ifd_bytes)
        output.extend(overflow_bytes)
        current_ifd_pos = len(output)

    # Append all pixel data
    for _arr, _lw, _lh, _rel_offsets, _byte_counts, comp_chunks in pixel_data_parts:
        for chunk in comp_chunks:
            output.extend(chunk)

    return bytes(output)


# ---------------------------------------------------------------------------
# Public write function
# ---------------------------------------------------------------------------

def write(data: np.ndarray, path: str, *,
          geo_transform: GeoTransform | None = None,
          crs_epsg: int | None = None,
          nodata=None,
          compression: str = 'zstd',
          tiled: bool = True,
          tile_size: int = 256,
          predictor: bool = False,
          cog: bool = False,
          overview_levels: list[int] | None = None,
          overview_resampling: str = 'mean',
          raster_type: int = 1,
          x_resolution: float | None = None,
          y_resolution: float | None = None,
          resolution_unit: int | None = None,
          gdal_metadata_xml: str | None = None,
          extra_tags: list | None = None,
          bigtiff: bool | None = None) -> None:
    """Write a numpy array as a GeoTIFF or COG.

    Parameters
    ----------
    data : np.ndarray
        2D array (height x width).
    path : str
        Output file path.
    geo_transform : GeoTransform or None
        Pixel-to-coordinate mapping.
    crs_epsg : int or None
        EPSG code.
    nodata : float, int, or None
        NoData value.
    compression : str
        'none', 'deflate', or 'lzw'.
    tiled : bool
        Use tiled layout (vs strips).
    tile_size : int
        Tile width and height.
    predictor : bool
        Use horizontal differencing predictor.
    cog : bool
        Write as Cloud Optimized GeoTIFF.
    overview_levels : list of int or None
        Overview decimation factors (e.g. [2, 4, 8]).
        Only used if cog=True. If None and cog=True, auto-generate.
    """
    comp_tag = _compression_tag(compression)

    # JPEG validation: only uint8, 1 or 3 bands
    if comp_tag == COMPRESSION_JPEG:
        samples = data.shape[2] if data.ndim == 3 else 1
        if data.dtype != np.uint8:
            raise ValueError(
                f"JPEG compression requires uint8 data, got {data.dtype}. "
                f"JPEG is lossy and only supports 8-bit unsigned data.")
        if samples not in (1, 3):
            raise ValueError(
                f"JPEG compression requires 1 or 3 bands, got {samples}")

    # Build pixel data parts
    parts = []

    # Full resolution
    if tiled:
        rel_off, bc, comp_data = _write_tiled(data, comp_tag, predictor, tile_size)
    else:
        rel_off, bc, comp_data = _write_stripped(data, comp_tag, predictor)

    h, w = data.shape[:2]
    parts.append((data, w, h, rel_off, bc, comp_data))

    # Overviews
    if cog:
        if overview_levels is None:
            # Auto-generate: keep halving until < tile_size
            overview_levels = []
            oh, ow = h, w
            while oh > tile_size and ow > tile_size:
                oh //= 2
                ow //= 2
                if oh > 0 and ow > 0:
                    overview_levels.append(len(overview_levels) + 1)

        current = data
        for _ in overview_levels:
            current = _make_overview(current, method=overview_resampling)
            oh, ow = current.shape[:2]
            if tiled:
                o_off, o_bc, o_data = _write_tiled(current, comp_tag, predictor, tile_size)
            else:
                o_off, o_bc, o_data = _write_stripped(current, comp_tag, predictor)
            parts.append((current, ow, oh, o_off, o_bc, o_data))

    file_bytes = _assemble_tiff(
        w, h, data.dtype, comp_tag, predictor, tiled, tile_size,
        parts, geo_transform, crs_epsg, nodata, is_cog=cog,
        raster_type=raster_type,
        gdal_metadata_xml=gdal_metadata_xml,
        extra_tags=extra_tags,
        x_resolution=x_resolution, y_resolution=y_resolution,
        resolution_unit=resolution_unit,
        force_bigtiff=bigtiff,
    )

    _write_bytes(file_bytes, path)

    # Post-write validation: verify the header is parseable
    from ._header import parse_header as _ph
    try:
        _ph(file_bytes[:16])
    except Exception as e:
        import warnings
        warnings.warn(f"Written file may be corrupt: {e}", stacklevel=2)


def _is_fsspec_uri(path: str) -> bool:
    """Check if a path is a fsspec-compatible URI."""
    if path.startswith(('http://', 'https://')):
        return False
    return '://' in path


def _write_bytes(file_bytes: bytes, path: str) -> None:
    """Write bytes to a local file (atomic) or cloud storage (via fsspec)."""
    import os

    if _is_fsspec_uri(path):
        try:
            import fsspec
        except ImportError:
            raise ImportError(
                "fsspec is required to write to cloud storage. "
                "Install it with: pip install fsspec")
        fs, fspath = fsspec.core.url_to_fs(path)
        with fs.open(fspath, 'wb') as f:
            f.write(file_bytes)
        return

    # Local file: write to temp file then atomically rename
    import tempfile
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tif.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(file_bytes)
        os.replace(tmp_path, path)  # atomic on POSIX
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
