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
from numba import int32 as numba_int32, uint8 as numba_uint8, int64 as numba_int64


# ---------------------------------------------------------------------------
# Deflate/inflate decode kernel -- one thread block per tile
# ---------------------------------------------------------------------------

# Static tables for deflate
# Length base values and extra bits for codes 257-285
_LEN_BASE = np.array([
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
], dtype=np.int32)
_LEN_EXTRA = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
], dtype=np.int32)
# Distance base values and extra bits for codes 0-29
_DIST_BASE = np.array([
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193,
    12289, 16385, 24577,
], dtype=np.int32)
_DIST_EXTRA = np.array([
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
], dtype=np.int32)
# Code length code order (for dynamic Huffman)
_CL_ORDER = np.array([
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
], dtype=np.int32)


@cuda.jit(device=True)
def _inflate_read_bits(src, src_start, src_len, bit_pos, n):
    """Read n bits (LSB-first) from the source stream."""
    val = numba_int32(0)
    for i in range(n):
        byte_idx = (bit_pos[0] >> 3)
        bit_idx = bit_pos[0] & 7
        if byte_idx < src_len:
            val |= numba_int32((src[src_start + byte_idx] >> bit_idx) & 1) << i
        bit_pos[0] += 1
    return val


@cuda.jit(device=True)
def _inflate_build_table(lengths, n_codes, table, max_bits,
                          overflow_codes, overflow_lens, n_overflow):
    """Build a Huffman decode table from code lengths.

    Codes <= max_bits go into the fast table: table[reversed_code] = (sym << 5) | length.
    Codes > max_bits go into overflow arrays for slow-path decode.
    """
    bl_count = cuda.local.array(16, dtype=numba_int32)
    for i in range(16):
        bl_count[i] = 0
    for i in range(n_codes):
        bl_count[lengths[i]] += 1
    bl_count[0] = 0

    next_code = cuda.local.array(16, dtype=numba_int32)
    code = 0
    for bits in range(1, 16):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    for i in range(1 << max_bits):
        table[i] = 0

    n_overflow[0] = 0

    for sym in range(n_codes):
        ln = lengths[sym]
        if ln == 0:
            continue
        code = next_code[ln]
        next_code[ln] += 1

        # Reverse the code bits for LSB-first lookup
        rev = numba_int32(0)
        c = code
        for b in range(ln):
            rev = (rev << 1) | (c & 1)
            c >>= 1

        if ln <= max_bits:
            # Fast table: fill all entries that share this prefix
            # (entries where the extra high bits vary)
            step = 1 << ln
            idx = rev
            while idx < (1 << max_bits):
                table[idx] = numba_int32((sym << 5) | ln)
                idx += step
        else:
            # Overflow: store reversed code + length for slow-path scan
            oi = n_overflow[0]
            if oi < overflow_codes.shape[0]:
                overflow_codes[oi] = rev
                overflow_lens[oi] = (sym << 5) | ln
                n_overflow[0] = oi + 1


@cuda.jit(device=True)
def _inflate_decode_symbol(src, src_start, src_len, bit_pos, table, max_bits,
                            overflow_codes, overflow_lens, n_overflow):
    """Decode one Huffman symbol. Fast table for short codes, overflow scan for long."""
    # Peek 15 bits (max deflate code length)
    peek = numba_int64(0)
    for i in range(15):
        byte_idx = (bit_pos[0] + i) >> 3
        bit_idx = (bit_pos[0] + i) & 7
        if byte_idx < src_len:
            peek |= numba_int64((src[src_start + byte_idx] >> bit_idx) & 1) << i

    # Try fast table first
    entry = table[numba_int32(peek) & ((1 << max_bits) - 1)]
    length = entry & 0x1F
    symbol = entry >> 5

    if length > 0:
        bit_pos[0] += length
        return symbol

    # Slow path: scan overflow entries
    for i in range(n_overflow[0]):
        ov_rev = overflow_codes[i]
        ov_entry = overflow_lens[i]
        ov_len = ov_entry & 0x1F
        ov_sym = ov_entry >> 5
        mask = (1 << ov_len) - 1
        if (numba_int32(peek) & mask) == ov_rev:
            bit_pos[0] += ov_len
            return ov_sym

    # Should not happen with valid data -- advance 1 bit to avoid freeze
    bit_pos[0] += 1
    return 0


@cuda.jit
def _inflate_tiles_kernel(
    compressed_buf,
    tile_offsets,
    tile_sizes,
    decompressed_buf,
    tile_out_offsets,
    tile_out_sizes,
    tile_actual_sizes,
    d_len_base, d_len_extra, d_dist_base, d_dist_extra, d_cl_order,
):
    """Inflate (decompress) one zlib-wrapped deflate tile per thread block.

    Thread 0 in each block does the sequential inflate.
    Huffman table in shared memory.
    """
    tile_idx = cuda.blockIdx.x
    if tile_idx >= tile_offsets.shape[0]:
        return
    if cuda.threadIdx.x != 0:
        return

    src_start = tile_offsets[tile_idx]
    src_len = tile_sizes[tile_idx]
    dst_start = tile_out_offsets[tile_idx]
    dst_len = tile_out_sizes[tile_idx]

    if src_len <= 2:
        tile_actual_sizes[tile_idx] = 0
        return

    # Skip 2-byte zlib header (0x78 0x9C or similar)
    bit_pos = cuda.local.array(1, dtype=numba_int64)
    bit_pos[0] = numba_int64(16)  # skip 2 bytes = 16 bits

    out_pos = 0

    # Two-level Huffman tables:
    # Level 1 (shared memory, fast): 10-bit lookup (1024 entries)
    # Level 2 (local memory, slow): overflow for codes > 10 bits
    MAX_LIT_BITS = 10
    MAX_DIST_BITS = 10
    lit_table = cuda.shared.array(1024, dtype=numba_int32)
    dist_table = cuda.shared.array(1024, dtype=numba_int32)

    # Overflow arrays for long codes (rarely > 50 entries)
    lit_ov_codes = cuda.local.array(64, dtype=numba_int32)
    lit_ov_lens = cuda.local.array(64, dtype=numba_int32)
    n_lit_ov = cuda.local.array(1, dtype=numba_int32)
    dist_ov_codes = cuda.local.array(32, dtype=numba_int32)
    dist_ov_lens = cuda.local.array(32, dtype=numba_int32)
    n_dist_ov = cuda.local.array(1, dtype=numba_int32)
    n_lit_ov[0] = 0
    n_dist_ov[0] = 0

    code_lengths = cuda.local.array(320, dtype=numba_int32)

    while True:
        # Read block header
        bfinal = _inflate_read_bits(compressed_buf, src_start, src_len, bit_pos, 1)
        btype = _inflate_read_bits(compressed_buf, src_start, src_len, bit_pos, 2)

        if btype == 0:
            # Stored block: align to byte boundary, read len
            bit_pos[0] = ((bit_pos[0] + 7) >> 3) << 3
            ln = _inflate_read_bits(compressed_buf, src_start, src_len, bit_pos, 16)
            _inflate_read_bits(compressed_buf, src_start, src_len, bit_pos, 16)  # nlen (complement)
            for i in range(ln):
                byte_idx = bit_pos[0] >> 3
                if byte_idx < src_len and out_pos < dst_len:
                    decompressed_buf[dst_start + out_pos] = compressed_buf[src_start + byte_idx]
                    out_pos += 1
                bit_pos[0] += 8

        elif btype == 1:
            # Fixed Huffman: build fixed tables
            for i in range(144):
                code_lengths[i] = 8
            for i in range(144, 256):
                code_lengths[i] = 9
            for i in range(256, 280):
                code_lengths[i] = 7
            for i in range(280, 288):
                code_lengths[i] = 8
            _inflate_build_table(code_lengths, 288, lit_table, MAX_LIT_BITS,
                                 lit_ov_codes, lit_ov_lens, n_lit_ov)

            for i in range(30):
                code_lengths[i] = 5
            _inflate_build_table(code_lengths, 30, dist_table, MAX_DIST_BITS,
                                 dist_ov_codes, dist_ov_lens, n_dist_ov)

            # Decode symbols
            while True:
                sym = _inflate_decode_symbol(
                    compressed_buf, src_start, src_len, bit_pos,
                    lit_table, MAX_LIT_BITS,
                    lit_ov_codes, lit_ov_lens, n_lit_ov)

                if sym < 256:
                    if out_pos < dst_len:
                        decompressed_buf[dst_start + out_pos] = numba_uint8(sym)
                        out_pos += 1
                elif sym == 256:
                    break
                else:
                    # Length-distance pair
                    li = sym - 257
                    if li < 29:
                        length = d_len_base[li]
                        if d_len_extra[li] > 0:
                            length += _inflate_read_bits(
                                compressed_buf, src_start, src_len,
                                bit_pos, d_len_extra[li])
                    else:
                        length = 3

                    dsym = _inflate_decode_symbol(
                        compressed_buf, src_start, src_len, bit_pos,
                        dist_table, MAX_DIST_BITS,
                        dist_ov_codes, dist_ov_lens, n_dist_ov)
                    if dsym < 30:
                        dist = d_dist_base[dsym]
                        if d_dist_extra[dsym] > 0:
                            dist += _inflate_read_bits(
                                compressed_buf, src_start, src_len,
                                bit_pos, d_dist_extra[dsym])
                    else:
                        dist = 1

                    # Copy from output window
                    for i in range(length):
                        if out_pos < dst_len and dist <= out_pos:
                            decompressed_buf[dst_start + out_pos] = \
                                decompressed_buf[dst_start + out_pos - dist]
                            out_pos += 1

        elif btype == 2:
            # Dynamic Huffman: read code length codes, then build tables
            hlit = _inflate_read_bits(compressed_buf, src_start, src_len, bit_pos, 5) + 257
            hdist = _inflate_read_bits(compressed_buf, src_start, src_len, bit_pos, 5) + 1
            hclen = _inflate_read_bits(compressed_buf, src_start, src_len, bit_pos, 4) + 4

            # Read code length code lengths
            cl_lengths = cuda.local.array(19, dtype=numba_int32)
            for i in range(19):
                cl_lengths[i] = 0
            for i in range(hclen):
                cl_lengths[d_cl_order[i]] = _inflate_read_bits(
                    compressed_buf, src_start, src_len, bit_pos, 3)

            # Build code length Huffman table (small: 7 bits max, no overflow)
            cl_table = cuda.local.array(128, dtype=numba_int32)
            cl_ov_c = cuda.local.array(4, dtype=numba_int32)
            cl_ov_l = cuda.local.array(4, dtype=numba_int32)
            n_cl_ov = cuda.local.array(1, dtype=numba_int32)
            n_cl_ov[0] = 0
            _inflate_build_table(cl_lengths, 19, cl_table, 7,
                                 cl_ov_c, cl_ov_l, n_cl_ov)

            # Decode literal/length + distance code lengths
            total_codes = hlit + hdist
            idx = 0
            for i in range(320):
                code_lengths[i] = 0

            while idx < total_codes:
                sym = numba_int32(0)
                # Decode from cl_table (7-bit)
                peek = numba_int32(0)
                for b in range(7):
                    byte_idx = (bit_pos[0] + b) >> 3
                    bit_idx = (bit_pos[0] + b) & 7
                    if byte_idx < src_len:
                        peek |= numba_int32(
                            (compressed_buf[src_start + byte_idx] >> bit_idx) & 1) << b
                entry = cl_table[peek & 127]
                ln = entry & 0x1F
                sym = entry >> 5
                if ln > 0:
                    bit_pos[0] += ln
                else:
                    bit_pos[0] += 1

                if sym < 16:
                    code_lengths[idx] = sym
                    idx += 1
                elif sym == 16:
                    rep = _inflate_read_bits(
                        compressed_buf, src_start, src_len, bit_pos, 2) + 3
                    val = code_lengths[idx - 1] if idx > 0 else 0
                    for _ in range(rep):
                        if idx < 320:
                            code_lengths[idx] = val
                            idx += 1
                elif sym == 17:
                    rep = _inflate_read_bits(
                        compressed_buf, src_start, src_len, bit_pos, 3) + 3
                    for _ in range(rep):
                        if idx < 320:
                            code_lengths[idx] = 0
                            idx += 1
                elif sym == 18:
                    rep = _inflate_read_bits(
                        compressed_buf, src_start, src_len, bit_pos, 7) + 11
                    for _ in range(rep):
                        if idx < 320:
                            code_lengths[idx] = 0
                            idx += 1

            # Build lit/len and dist tables
            n_lit_ov[0] = 0
            _inflate_build_table(code_lengths, hlit, lit_table, MAX_LIT_BITS,
                                 lit_ov_codes, lit_ov_lens, n_lit_ov)
            # Distance codes start at code_lengths[hlit]
            dist_lengths = cuda.local.array(32, dtype=numba_int32)
            for i in range(32):
                dist_lengths[i] = 0
            for i in range(hdist):
                dist_lengths[i] = code_lengths[hlit + i]
            n_dist_ov[0] = 0
            _inflate_build_table(dist_lengths, hdist, dist_table, MAX_DIST_BITS,
                                 dist_ov_codes, dist_ov_lens, n_dist_ov)

            # Decode symbols (same loop as fixed Huffman)
            while True:
                sym = _inflate_decode_symbol(
                    compressed_buf, src_start, src_len, bit_pos,
                    lit_table, MAX_LIT_BITS,
                    lit_ov_codes, lit_ov_lens, n_lit_ov)

                if sym < 256:
                    if out_pos < dst_len:
                        decompressed_buf[dst_start + out_pos] = numba_uint8(sym)
                        out_pos += 1
                elif sym == 256:
                    break
                else:
                    li = sym - 257
                    if li < 29:
                        length = d_len_base[li]
                        if d_len_extra[li] > 0:
                            length += _inflate_read_bits(
                                compressed_buf, src_start, src_len,
                                bit_pos, d_len_extra[li])
                    else:
                        length = 3

                    dsym = _inflate_decode_symbol(
                        compressed_buf, src_start, src_len, bit_pos,
                        dist_table, MAX_DIST_BITS,
                        dist_ov_codes, dist_ov_lens, n_dist_ov)
                    if dsym < 30:
                        dist = d_dist_base[dsym]
                        if d_dist_extra[dsym] > 0:
                            dist += _inflate_read_bits(
                                compressed_buf, src_start, src_len,
                                bit_pos, d_dist_extra[dsym])
                    else:
                        dist = 1

                    for i in range(length):
                        if out_pos < dst_len and dist <= out_pos:
                            decompressed_buf[dst_start + out_pos] = \
                                decompressed_buf[dst_start + out_pos - dist]
                            out_pos += 1
        else:
            break  # invalid block type

        if bfinal:
            break

    tile_actual_sizes[tile_idx] = out_pos


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
# nvCOMP batch decompression (optional, fast path)
# ---------------------------------------------------------------------------

def _find_nvcomp_lib():
    """Find and load libnvcomp.so. Returns ctypes.CDLL or None."""
    import ctypes
    import os

    # Try common locations
    search_paths = [
        'libnvcomp.so',  # system LD_LIBRARY_PATH
    ]

    # Check conda envs
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        search_paths.append(os.path.join(conda_prefix, 'lib', 'libnvcomp.so'))

    # Also check sibling conda envs that might have rapids
    conda_base = os.path.dirname(conda_prefix) if conda_prefix else ''
    if conda_base:
        for env in ['rapids', 'test-again', 'rtxpy-fire']:
            p = os.path.join(conda_base, env, 'lib', 'libnvcomp.so')
            if os.path.exists(p):
                search_paths.append(p)

    for path in search_paths:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


_nvcomp_lib = None
_nvcomp_checked = False


def _get_nvcomp():
    """Get the nvCOMP library handle (cached). Returns CDLL or None."""
    global _nvcomp_lib, _nvcomp_checked
    if not _nvcomp_checked:
        _nvcomp_checked = True
        _nvcomp_lib = _find_nvcomp_lib()
    return _nvcomp_lib


def _try_nvcomp_batch_decompress(compressed_tiles, tile_bytes, compression):
    """Try batch decompression via nvCOMP C API. Returns CuPy array or None.

    Uses nvcompBatchedDeflateDecompressAsync to decompress all tiles in
    one GPU API call. Falls back to None if nvCOMP is not available.
    """
    if compression not in (8, 32946, 50000):  # Deflate and ZSTD
        return None

    lib = _get_nvcomp()
    if lib is None:
        # Try kvikio.nvcomp as alternative
        try:
            import kvikio.nvcomp as nvcomp
        except ImportError:
            return None

        import cupy
        try:
            raw_tiles = []
            for tile in compressed_tiles:
                raw_tiles.append(tile[2:-4] if len(tile) > 6 else tile)
            manager = nvcomp.DeflateManager(chunk_size=tile_bytes)
            d_compressed = [cupy.asarray(np.frombuffer(t, dtype=np.uint8))
                            for t in raw_tiles]
            d_decompressed = manager.decompress(d_compressed)
            return cupy.concatenate([d.ravel() for d in d_decompressed])
        except Exception:
            return None

    # Direct ctypes nvCOMP C API
    import ctypes
    import cupy

    class _NvcompDecompOpts(ctypes.Structure):
        """nvCOMP batched decompression options (passed by value)."""
        _fields_ = [
            ('backend', ctypes.c_int),
            ('reserved', ctypes.c_char * 60),
        ]

    # Deflate has a different struct with sort_before_hw_decompress field
    class _NvcompDeflateDecompOpts(ctypes.Structure):
        _fields_ = [
            ('backend', ctypes.c_int),
            ('sort_before_hw_decompress', ctypes.c_int),
            ('reserved', ctypes.c_char * 56),
        ]

    try:
        n_tiles = len(compressed_tiles)

        # Prepare compressed tiles for nvCOMP
        if compression in (8, 32946):  # Deflate
            # Strip 2-byte zlib header + 4-byte adler32 checksum
            raw_tiles = [t[2:-4] if len(t) > 6 else t for t in compressed_tiles]
            get_temp_fn = 'nvcompBatchedDeflateDecompressGetTempSizeAsync'
            decomp_fn = 'nvcompBatchedDeflateDecompressAsync'
            opts = _NvcompDeflateDecompOpts(backend=0, sort_before_hw_decompress=0,
                                            reserved=b'\x00' * 56)
        elif compression == 50000:  # ZSTD
            raw_tiles = list(compressed_tiles)  # no header stripping
            get_temp_fn = 'nvcompBatchedZstdDecompressGetTempSizeAsync'
            decomp_fn = 'nvcompBatchedZstdDecompressAsync'
            opts = _NvcompDecompOpts(backend=0, reserved=b'\x00' * 60)
        else:
            return None

        # Upload compressed tiles to device
        d_comp_bufs = [cupy.asarray(np.frombuffer(t, dtype=np.uint8)) for t in raw_tiles]
        d_decomp_bufs = [cupy.empty(tile_bytes, dtype=cupy.uint8) for _ in range(n_tiles)]

        d_comp_ptrs = cupy.array([b.data.ptr for b in d_comp_bufs], dtype=cupy.uint64)
        d_decomp_ptrs = cupy.array([b.data.ptr for b in d_decomp_bufs], dtype=cupy.uint64)
        d_comp_sizes = cupy.array([len(t) for t in raw_tiles], dtype=cupy.uint64)
        d_buf_sizes = cupy.full(n_tiles, tile_bytes, dtype=cupy.uint64)
        d_actual = cupy.empty(n_tiles, dtype=cupy.uint64)

        # Set argtypes for proper struct passing
        temp_fn = getattr(lib, get_temp_fn)
        temp_fn.restype = ctypes.c_int

        temp_size = ctypes.c_size_t(0)
        status = temp_fn(
            ctypes.c_size_t(n_tiles),
            ctypes.c_size_t(tile_bytes),
            opts,
            ctypes.byref(temp_size),
            ctypes.c_size_t(n_tiles * tile_bytes),
        )
        if status != 0:
            return None

        ts = max(temp_size.value, 1)
        d_temp = cupy.empty(ts, dtype=cupy.uint8)
        d_statuses = cupy.zeros(n_tiles, dtype=cupy.int32)

        dec_fn = getattr(lib, decomp_fn)
        dec_fn.restype = ctypes.c_int

        status = dec_fn(
            ctypes.c_void_p(d_comp_ptrs.data.ptr),
            ctypes.c_void_p(d_comp_sizes.data.ptr),
            ctypes.c_void_p(d_buf_sizes.data.ptr),
            ctypes.c_void_p(d_actual.data.ptr),
            ctypes.c_size_t(n_tiles),
            ctypes.c_void_p(d_temp.data.ptr),
            ctypes.c_size_t(ts),
            ctypes.c_void_p(d_decomp_ptrs.data.ptr),
            opts,
            ctypes.c_void_p(d_statuses.data.ptr),
            ctypes.c_void_p(0),  # default stream
        )
        if status != 0:
            return None

        cupy.cuda.Device().synchronize()

        if int(cupy.any(d_statuses != 0)):
            return None

        return cupy.concatenate(d_decomp_bufs)

    except Exception:
        return None


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

    # Try nvCOMP batch decompression first (much faster if available)
    nvcomp_result = _try_nvcomp_batch_decompress(
        compressed_tiles, tile_bytes, compression)
    if nvcomp_result is not None:
        d_decomp = nvcomp_result
        decomp_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
        d_decomp_offsets = cupy.asarray(decomp_offsets)
    elif compression == 5:  # LZW
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

    elif compression in (8, 32946):  # Deflate / Adobe Deflate
        comp_sizes = [len(t) for t in compressed_tiles]
        comp_offsets = np.zeros(n_tiles, dtype=np.int64)
        for i in range(1, n_tiles):
            comp_offsets[i] = comp_offsets[i - 1] + comp_sizes[i - 1]
        total_comp = sum(comp_sizes)

        comp_buf_host = np.empty(total_comp, dtype=np.uint8)
        for i, tile in enumerate(compressed_tiles):
            comp_buf_host[comp_offsets[i]:comp_offsets[i] + comp_sizes[i]] = \
                np.frombuffer(tile, dtype=np.uint8)

        d_comp = cupy.asarray(comp_buf_host)
        d_comp_offsets = cupy.asarray(comp_offsets)
        d_comp_sizes = cupy.asarray(np.array(comp_sizes, dtype=np.int64))

        decomp_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
        d_decomp = cupy.zeros(n_tiles * tile_bytes, dtype=cupy.uint8)
        d_decomp_offsets = cupy.asarray(decomp_offsets)
        d_tile_sizes = cupy.full(n_tiles, tile_bytes, dtype=cupy.int64)
        d_actual_sizes = cupy.zeros(n_tiles, dtype=cupy.int64)

        # Static deflate tables on device
        d_len_base = cupy.asarray(_LEN_BASE)
        d_len_extra = cupy.asarray(_LEN_EXTRA)
        d_dist_base = cupy.asarray(_DIST_BASE)
        d_dist_extra = cupy.asarray(_DIST_EXTRA)
        d_cl_order = cupy.asarray(_CL_ORDER)

        # One thread block per tile, thread 0 does the inflate
        _inflate_tiles_kernel[n_tiles, 32](
            d_comp, d_comp_offsets, d_comp_sizes,
            d_decomp, d_decomp_offsets, d_tile_sizes, d_actual_sizes,
            d_len_base, d_len_extra, d_dist_base, d_dist_extra, d_cl_order,
        )
        cuda.synchronize()

    elif compression == 1:  # Uncompressed
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
            f"GPU decode supports LZW (5), deflate (8), and uncompressed (1), "
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
