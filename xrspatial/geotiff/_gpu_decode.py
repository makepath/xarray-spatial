"""GPU-accelerated TIFF tile decompression via Numba CUDA.

Provides CUDA kernels for LZW decode, horizontal predictor decode,
and floating-point predictor decode. Each tile is processed by one
thread (LZW is sequential per-stream), but all tiles run in parallel.
"""
from __future__ import annotations

import math

import numpy as np
from numba import cuda

# LZW constants (same as _compression.py)
LZW_CLEAR_CODE = 256
LZW_EOI_CODE = 257
LZW_FIRST_CODE = 258
LZW_MAX_CODE = 4095
LZW_MAX_BITS = 12


# ---------------------------------------------------------------------------
# LZW decode kernel -- one thread per tile
# ---------------------------------------------------------------------------

@cuda.jit
def _lzw_decode_tiles_kernel(
    compressed_buf,       # uint8: all compressed tile data concatenated
    tile_offsets,         # int64: start offset of each tile in compressed_buf
    tile_sizes,           # int64: compressed size of each tile
    decompressed_buf,     # uint8: output buffer (all tiles concatenated)
    tile_out_offsets,     # int64: start offset of each tile in decompressed_buf
    tile_out_sizes,       # int64: expected decompressed size per tile
    tile_actual_sizes,    # int64: actual bytes written per tile (output)
):
    """Decode one LZW tile per thread block.

    One thread block = one tile. Thread 0 in each block does the sequential
    LZW decode. The table lives in shared memory (fast, ~20KB per block)
    instead of local memory (slow DRAM spill).
    """
    tile_idx = cuda.blockIdx.x
    if tile_idx >= tile_offsets.shape[0]:
        return

    # Only thread 0 in each block does the work
    if cuda.threadIdx.x != 0:
        return

    src_start = tile_offsets[tile_idx]
    src_len = tile_sizes[tile_idx]
    dst_start = tile_out_offsets[tile_idx]
    dst_len = tile_out_sizes[tile_idx]

    if src_len == 0:
        tile_actual_sizes[tile_idx] = 0
        return

    # LZW table in shared memory (fast on-chip SRAM)
    table_prefix = cuda.shared.array(4096, dtype=numba_int32)
    table_suffix = cuda.shared.array(4096, dtype=numba_uint8)
    stack = cuda.shared.array(4096, dtype=numba_uint8)

    # Initialize single-byte entries
    for i in range(256):
        table_prefix[i] = -1
        table_suffix[i] = numba_uint8(i)
    for i in range(256, 4096):
        table_prefix[i] = -1
        table_suffix[i] = numba_uint8(0)

    bit_pos = 0
    code_size = 9
    next_code = LZW_FIRST_CODE
    out_pos = 0
    old_code = -1

    while True:
        # Read next code (MSB-first)
        byte_offset = bit_pos >> 3
        if byte_offset >= src_len:
            break

        b0 = numba_int32(compressed_buf[src_start + byte_offset]) << 16
        if byte_offset + 1 < src_len:
            b0 |= numba_int32(compressed_buf[src_start + byte_offset + 1]) << 8
        if byte_offset + 2 < src_len:
            b0 |= numba_int32(compressed_buf[src_start + byte_offset + 2])

        bit_off = bit_pos & 7
        code = (b0 >> (24 - bit_off - code_size)) & ((1 << code_size) - 1)
        bit_pos += code_size

        if code == LZW_EOI_CODE:
            break

        if code == LZW_CLEAR_CODE:
            code_size = 9
            next_code = LZW_FIRST_CODE
            old_code = -1
            continue

        if old_code == -1:
            if code < 256 and out_pos < dst_len:
                decompressed_buf[dst_start + out_pos] = numba_uint8(code)
                out_pos += 1
            old_code = code
            continue

        if code < next_code:
            # Walk chain, push to stack
            c = code
            sp = 0
            while c >= 0 and c < 4096 and sp < 4096:
                stack[sp] = table_suffix[c]
                sp += 1
                c = table_prefix[c]

            # Emit reversed
            for i in range(sp - 1, -1, -1):
                if out_pos < dst_len:
                    decompressed_buf[dst_start + out_pos] = stack[i]
                    out_pos += 1

            if next_code <= LZW_MAX_CODE and sp > 0:
                table_prefix[next_code] = old_code
                table_suffix[next_code] = stack[sp - 1]
                next_code += 1
        else:
            # Special case: code == next_code
            c = old_code
            sp = 0
            while c >= 0 and c < 4096 and sp < 4096:
                stack[sp] = table_suffix[c]
                sp += 1
                c = table_prefix[c]

            if sp == 0:
                old_code = code
                continue

            first_char = stack[sp - 1]
            for i in range(sp - 1, -1, -1):
                if out_pos < dst_len:
                    decompressed_buf[dst_start + out_pos] = stack[i]
                    out_pos += 1
            if out_pos < dst_len:
                decompressed_buf[dst_start + out_pos] = first_char
                out_pos += 1

            if next_code <= LZW_MAX_CODE:
                table_prefix[next_code] = old_code
                table_suffix[next_code] = first_char
                next_code += 1

        # Early change
        if next_code > (1 << code_size) - 2 and code_size < LZW_MAX_BITS:
            code_size += 1

        old_code = code

    tile_actual_sizes[tile_idx] = out_pos


# Type aliases for Numba CUDA local arrays
from numba import int32 as numba_int32, uint8 as numba_uint8


# ---------------------------------------------------------------------------
# Predictor decode kernels -- one thread per row
# ---------------------------------------------------------------------------

@cuda.jit
def _predictor_decode_kernel(data, width, height, bytes_per_sample):
    """Undo horizontal differencing (predictor=2), one thread per row."""
    row = cuda.grid(1)
    if row >= height:
        return

    row_bytes = width * bytes_per_sample
    row_start = row * row_bytes

    for col in range(bytes_per_sample, row_bytes):
        idx = row_start + col
        data[idx] = numba_uint8(
            (numba_int32(data[idx]) + numba_int32(data[idx - bytes_per_sample])) & 0xFF)


@cuda.jit
def _fp_predictor_decode_kernel(data, tmp, width, height, bps):
    """Undo floating-point predictor (predictor=3), one thread per row.

    data: flat uint8 device array
    tmp: scratch buffer, same size as data
    """
    row = cuda.grid(1)
    if row >= height:
        return

    row_len = width * bps
    start = row * row_len

    # Step 1: undo horizontal differencing
    for i in range(1, row_len):
        idx = start + i
        data[idx] = numba_uint8(
            (numba_int32(data[idx]) + numba_int32(data[idx - 1])) & 0xFF)

    # Step 2: un-transpose byte lanes (MSB-first) back to native order
    for sample in range(width):
        for b in range(bps):
            tmp[start + sample * bps + b] = data[start + (bps - 1 - b) * width + sample]

    # Copy back
    for i in range(row_len):
        data[start + i] = tmp[start + i]


# ---------------------------------------------------------------------------
# Tile assembly kernel -- one thread per output pixel
# ---------------------------------------------------------------------------

@cuda.jit
def _assemble_tiles_kernel(
    decompressed_buf,     # uint8: all decompressed tiles concatenated
    tile_out_offsets,     # int64: byte offset of each tile in decompressed_buf
    tile_width,           # int: tile width in pixels
    tile_height,          # int: tile height in pixels
    bytes_per_pixel,      # int: dtype.itemsize * samples_per_pixel
    image_width,          # int: output image width
    image_height,         # int: output image height
    tiles_across,         # int: number of tile columns
    output,               # uint8: output image buffer (flat, row-major)
):
    """Copy decompressed tile pixels into the output image, one thread per pixel."""
    pixel_idx = cuda.grid(1)
    total_pixels = image_width * image_height
    if pixel_idx >= total_pixels:
        return

    # Output row and column
    out_row = pixel_idx // image_width
    out_col = pixel_idx % image_width

    # Which tile does this pixel belong to?
    tile_row = out_row // tile_height
    tile_col = out_col // tile_width
    tile_idx = tile_row * tiles_across + tile_col

    # Position within the tile
    local_row = out_row - tile_row * tile_height
    local_col = out_col - tile_col * tile_width

    # Source and destination byte offsets
    tile_offset = tile_out_offsets[tile_idx]
    src_byte = tile_offset + (local_row * tile_width + local_col) * bytes_per_pixel
    dst_byte = (out_row * image_width + out_col) * bytes_per_pixel

    for b in range(bytes_per_pixel):
        output[dst_byte + b] = decompressed_buf[src_byte + b]


# ---------------------------------------------------------------------------
# High-level GPU decode pipeline
# ---------------------------------------------------------------------------

def gpu_decode_tiles(
    compressed_tiles: list[bytes],
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    compression: int,
    predictor: int,
    dtype: np.dtype,
    samples: int = 1,
):
    """Decode and assemble TIFF tiles entirely on GPU.

    Parameters
    ----------
    compressed_tiles : list of bytes
        One entry per tile, in row-major tile order.
    tile_width, tile_height : int
        Tile dimensions.
    image_width, image_height : int
        Output image dimensions.
    compression : int
        TIFF compression tag (5=LZW, 1=none).
    predictor : int
        Predictor tag (1=none, 2=horizontal, 3=float).
    dtype : np.dtype
        Output pixel dtype.
    samples : int
        Samples per pixel.

    Returns
    -------
    cupy.ndarray
        Decoded image on GPU device.
    """
    import cupy

    n_tiles = len(compressed_tiles)
    bytes_per_pixel = dtype.itemsize * samples
    tile_bytes = tile_width * tile_height * bytes_per_pixel

    if compression == 5:  # LZW
        # Concatenate all compressed tiles into one device buffer
        comp_sizes = [len(t) for t in compressed_tiles]
        comp_offsets = np.zeros(n_tiles, dtype=np.int64)
        for i in range(1, n_tiles):
            comp_offsets[i] = comp_offsets[i - 1] + comp_sizes[i - 1]
        total_comp = sum(comp_sizes)

        comp_buf_host = np.empty(total_comp, dtype=np.uint8)
        for i, tile in enumerate(compressed_tiles):
            comp_buf_host[comp_offsets[i]:comp_offsets[i] + comp_sizes[i]] = \
                np.frombuffer(tile, dtype=np.uint8)

        # Transfer to device
        d_comp = cupy.asarray(comp_buf_host)
        d_comp_offsets = cupy.asarray(comp_offsets)
        d_comp_sizes = cupy.asarray(np.array(comp_sizes, dtype=np.int64))

        # Allocate decompressed buffer on device
        decomp_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
        d_decomp = cupy.zeros(n_tiles * tile_bytes, dtype=cupy.uint8)
        d_decomp_offsets = cupy.asarray(decomp_offsets)
        d_tile_sizes = cupy.full(n_tiles, tile_bytes, dtype=cupy.int64)
        d_actual_sizes = cupy.zeros(n_tiles, dtype=cupy.int64)

        # Launch LZW decode: one thread block per tile (thread 0 decodes,
        # table in shared memory). Block size 32 for warp scheduling.
        _lzw_decode_tiles_kernel[n_tiles, 32](
            d_comp, d_comp_offsets, d_comp_sizes,
            d_decomp, d_decomp_offsets, d_tile_sizes, d_actual_sizes,
        )
        cuda.synchronize()

    elif compression == 1:  # Uncompressed
        # Just copy raw tile bytes to device
        raw_host = np.empty(n_tiles * tile_bytes, dtype=np.uint8)
        for i, tile in enumerate(compressed_tiles):
            start = i * tile_bytes
            t = np.frombuffer(tile, dtype=np.uint8)
            raw_host[start:start + len(t)] = t[:tile_bytes]
        d_decomp = cupy.asarray(raw_host)
        decomp_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
        d_decomp_offsets = cupy.asarray(decomp_offsets)

    else:
        raise ValueError(
            f"GPU decode only supports LZW (5) and uncompressed (1), "
            f"got compression={compression}")

    # Apply predictor on GPU
    if predictor == 2:
        # Horizontal differencing: one thread per row across all tiles
        total_rows = n_tiles * tile_height
        tpb = min(256, total_rows)
        bpg = math.ceil(total_rows / tpb)
        # Reshape so each tile's rows are contiguous (they already are)
        _predictor_decode_kernel[bpg, tpb](
            d_decomp, tile_width * samples, total_rows, dtype.itemsize * samples)
        cuda.synchronize()

    elif predictor == 3:
        # Float predictor: one thread per row
        total_rows = n_tiles * tile_height
        tpb = min(256, total_rows)
        bpg = math.ceil(total_rows / tpb)
        d_tmp = cupy.empty_like(d_decomp)
        _fp_predictor_decode_kernel[bpg, tpb](
            d_decomp, d_tmp, tile_width * samples, total_rows, dtype.itemsize)
        cuda.synchronize()

    # Assemble tiles into output image on GPU
    tiles_across = math.ceil(image_width / tile_width)
    total_pixels = image_width * image_height
    d_output = cupy.empty(total_pixels * bytes_per_pixel, dtype=cupy.uint8)

    tpb = 256
    bpg = math.ceil(total_pixels / tpb)
    _assemble_tiles_kernel[bpg, tpb](
        d_decomp, d_decomp_offsets,
        tile_width, tile_height, bytes_per_pixel,
        image_width, image_height, tiles_across,
        d_output,
    )
    cuda.synchronize()

    # Reshape to image
    if samples > 1:
        return d_output.view(dtype=cupy.dtype(dtype)).reshape(
            image_height, image_width, samples)
    return d_output.view(dtype=cupy.dtype(dtype)).reshape(
        image_height, image_width)
