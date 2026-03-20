"""GeoTIFF/COG writer."""
from __future__ import annotations

import math
import struct

import numpy as np

from ._compression import (
    COMPRESSION_DEFLATE,
    COMPRESSION_LZW,
    COMPRESSION_NONE,
    COMPRESSION_PACKBITS,
    COMPRESSION_ZSTD,
    compress,
    predictor_encode,
)
from ._dtypes import (
    DOUBLE,
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
    TAG_TILE_WIDTH,
    TAG_TILE_LENGTH,
    TAG_TILE_OFFSETS,
    TAG_TILE_BYTE_COUNTS,
    TAG_PREDICTOR,
)

# Byte order: always write little-endian
BO = '<'


def _compression_tag(compression_name: str) -> int:
    """Convert compression name to TIFF tag value."""
    _map = {
        'none': COMPRESSION_NONE,
        'deflate': COMPRESSION_DEFLATE,
        'lzw': COMPRESSION_LZW,
        'packbits': COMPRESSION_PACKBITS,
        'zstd': COMPRESSION_ZSTD,
    }
    name = compression_name.lower()
    if name not in _map:
        raise ValueError(f"Unsupported compression: {compression_name!r}. "
                         f"Use one of: {list(_map.keys())}")
    return _map[name]


def _make_overview(arr: np.ndarray) -> np.ndarray:
    """Generate a 2x decimated overview using 2x2 block averaging.

    Parameters
    ----------
    arr : np.ndarray
        2D or 3D (height, width, bands) array.

    Returns
    -------
    np.ndarray
        Half-resolution array.
    """
    h, w = arr.shape[:2]
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    cropped = arr[:h2, :w2]

    if arr.ndim == 3:
        # Multi-band: average each band independently
        bands = arr.shape[2]
        if arr.dtype.kind == 'f':
            blocks = cropped.reshape(h2 // 2, 2, w2 // 2, 2, bands)
            return np.nanmean(blocks, axis=(1, 3)).astype(arr.dtype)
        else:
            blocks = cropped.astype(np.float64).reshape(h2 // 2, 2, w2 // 2, 2, bands)
            return np.round(blocks.mean(axis=(1, 3))).astype(arr.dtype)
    else:
        if arr.dtype.kind == 'f':
            blocks = cropped.reshape(h2 // 2, 2, w2 // 2, 2)
            return np.nanmean(blocks, axis=(1, 3)).astype(arr.dtype)
        else:
            blocks = cropped.astype(np.float64).reshape(h2 // 2, 2, w2 // 2, 2)
            return np.round(blocks.mean(axis=(1, 3))).astype(arr.dtype)


# ---------------------------------------------------------------------------
# Tag serialization
# ---------------------------------------------------------------------------

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

        if predictor and compression != COMPRESSION_NONE:
            strip_arr = np.ascontiguousarray(data[r0:r1])
            buf = strip_arr.view(np.uint8).ravel().copy()
            buf = predictor_encode(buf, width, strip_rows, bytes_per_sample * samples)
            strip_data = buf.tobytes()
        else:
            strip_data = np.ascontiguousarray(data[r0:r1]).tobytes()

        compressed = compress(strip_data, compression)

        rel_offsets.append(current_offset)
        byte_counts.append(len(compressed))
        strips.append(compressed)
        current_offset += len(compressed)

    return rel_offsets, byte_counts, strips


# ---------------------------------------------------------------------------
# Tile writer
# ---------------------------------------------------------------------------

def _write_tiled(data: np.ndarray, compression: int, predictor: bool,
                 tile_size: int = 256) -> tuple[list, list, list]:
    """Compress data as tiles.

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

            # Extract tile, pad to full tile size if needed
            tile_slice = data[r0:r1, c0:c1]

            if actual_h < th or actual_w < tw:
                if data.ndim == 3:
                    padded = np.empty((th, tw, samples), dtype=dtype)
                else:
                    padded = np.empty((th, tw), dtype=dtype)
                padded[:actual_h, :actual_w] = tile_slice
                # Zero only the padding regions
                if actual_h < th:
                    padded[actual_h:, :] = 0
                if actual_w < tw:
                    padded[:actual_h, actual_w:] = 0
                tile_arr = padded
            else:
                tile_arr = np.ascontiguousarray(tile_slice)

            if predictor and compression != COMPRESSION_NONE:
                buf = tile_arr.view(np.uint8).ravel().copy()
                buf = predictor_encode(buf, tw, th, bytes_per_sample * samples)
                tile_data = buf.tobytes()
            else:
                tile_data = tile_arr.tobytes()

            compressed = compress(tile_data, compression)

            rel_offsets.append(current_offset)
            byte_counts.append(len(compressed))
            tiles.append(compressed)
            current_offset += len(compressed)

    return rel_offsets, byte_counts, tiles


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
                   raster_type: int = 1) -> bytes:
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

        if pred_val != 1:
            tags.append((TAG_PREDICTOR, SHORT, 1, pred_val))

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

        ifd_specs.append(tags)

    # --- Determine if BigTIFF is needed ---
    total_data = sum(sum(len(c) for c in chunks)
                     for _, _, _, _, _, chunks in pixel_data_parts)
    bigtiff = total_data > 3_900_000_000  # ~4GB threshold with margin

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
          compression: str = 'deflate',
          tiled: bool = True,
          tile_size: int = 256,
          predictor: bool = False,
          cog: bool = False,
          overview_levels: list[int] | None = None,
          raster_type: int = 1) -> None:
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
            current = _make_overview(current)
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
    )

    # Write to a temp file then atomically rename, so concurrent writes to
    # the same path don't interleave and readers never see partial output.
    import os
    import tempfile
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tif.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(file_bytes)
        os.replace(tmp_path, path)  # atomic on POSIX
    except BaseException:
        # Clean up the temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
