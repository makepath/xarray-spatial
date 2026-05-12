"""GPU-accelerated TIFF tile decompression via Numba CUDA.

Provides CUDA kernels for LZW decode, horizontal predictor decode,
and floating-point predictor decode. Each tile is processed by one
thread (LZW is sequential per-stream), but all tiles run in parallel.
"""
from __future__ import annotations

import math

import numpy as np
from numba import cuda

#: Fraction of free GPU memory we're willing to allocate in a single call.
#: Above this, raise MemoryError up-front so the caller gets an actionable
#: error rather than a CUDA OOM deep inside the kernel launch.
_GPU_FREE_MEMORY_FRACTION = 0.9


def _check_gpu_memory(required_bytes: int, what: str = "tile buffer") -> None:
    """Raise MemoryError if *required_bytes* would exhaust the GPU.

    Calls ``cupy.cuda.runtime.memGetInfo()`` and refuses any allocation
    that would consume more than ``_GPU_FREE_MEMORY_FRACTION`` of the
    currently free memory. This is a soft guard -- another process can
    grab memory between the check and the allocation -- but it catches
    the common 'this single tensor is way too big' case before CUDA
    raises a less informative error.

    Parameters
    ----------
    required_bytes : int
        Bytes the caller is about to allocate (sum across all buffers in
        the same logical step).
    what : str
        Short label included in the error message, e.g. ``"tile buffer"``.
    """
    if required_bytes <= 0:
        return
    try:
        import cupy
        free, total = cupy.cuda.runtime.memGetInfo()
    except Exception:
        # If we can't query, fall through and let the real allocation
        # surface the error. Don't add a second failure mode here.
        return

    budget = int(free * _GPU_FREE_MEMORY_FRACTION)
    if required_bytes > budget:
        raise MemoryError(
            f"GPU out of memory: {what} needs {required_bytes:,} bytes "
            f"but only {free:,} bytes free on device (cap is "
            f"{_GPU_FREE_MEMORY_FRACTION:.0%} of free = {budget:,} "
            "bytes). Consider reading the file in chunks via "
            "read_geotiff_dask(..., chunks=...) or freeing GPU memory "
            "with cupy.get_default_memory_pool().free_all_blocks()."
        )

def _xp_byteswap(arr):
    """Return *arr* with each element's bytes physically reversed.

    Equivalent to ``numpy.ndarray.byteswap()``: the dtype is preserved
    (still native-endian on output), and the bytes that make up each
    element are flipped end-for-end. Works on both numpy and cupy.

    The earlier ``arr.view(arr.dtype.newbyteorder()).copy()`` shortcut
    looked equivalent but produced an array whose dtype was tagged with
    the opposite byte order (e.g. ``>u2`` instead of ``<u2``). Downstream
    consumers -- numba ``@ngjit`` kernels in particular -- reject
    non-native dtypes (#1507 was exactly this), and the CPU reader's
    contract is that decoded arrays come back native, so we mirror that
    here by working in a uint8 view, reversing along the byte axis, and
    re-viewing as the original dtype.
    """
    if arr.itemsize == 1:
        return arr
    u8 = arr.view('u1').reshape(*arr.shape, arr.itemsize)
    return u8[..., ::-1].copy().view(arr.dtype).reshape(arr.shape)


@cuda.jit
def _byte_swap_lanes_kernel(buf, bps):
    """Reverse bytes within each *bps*-sized sample, one thread per sample.

    Each thread loads ``bps`` bytes from its sample, swaps them in place
    via register-resident temporaries, and writes them back. No global
    memory beyond *buf* is touched, so peak GPU memory is unchanged.
    """
    i = cuda.grid(1)
    n_samples = buf.size // bps
    if i >= n_samples:
        return
    base = i * bps
    half = bps // 2
    for j in range(half):
        a = buf[base + j]
        b = buf[base + bps - 1 - j]
        buf[base + j] = b
        buf[base + bps - 1 - j] = a


_BPS_TO_UINT = {2: np.uint16, 4: np.uint32, 8: np.uint64}


def _swap_byte_lanes(buf, bps: int) -> None:
    """Reverse bytes within each *bps*-sized sample of a flat uint8 buffer.

    Used by the GPU predictor=2 path to convert the raw decompressed byte
    stream from file byte order to native byte order before differencing
    (#1517). The per-dtype predictor kernels view ``buf`` as native
    unsigned integers, so on big-endian files the prefix-sum would run on
    a byte-swapped integer interpretation and produce wrong values.

    The swap is true in-place: no same-sized temporary is allocated.
    On numpy arrays it dispatches to ``ndarray.byteswap(inplace=True)``
    via a uint16/32/64 view; on cupy device arrays it launches
    :func:`_byte_swap_lanes_kernel`, which swaps bytes per sample using
    only register-resident temporaries.
    """
    if bps <= 1:
        return
    n = buf.size
    if n % bps != 0:
        raise ValueError(
            f"buffer size {n} is not a multiple of bps={bps}")
    if bps not in _BPS_TO_UINT:
        raise ValueError(f"unsupported bps={bps}; expected 2, 4, or 8")

    if isinstance(buf, np.ndarray):
        buf.view(_BPS_TO_UINT[bps]).byteswap(inplace=True)
        return

    n_samples = n // bps
    threads = 256
    blocks = (n_samples + threads - 1) // threads
    _byte_swap_lanes_kernel[blocks, threads](buf, bps)


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
from numba import (
    int32 as numba_int32,
    uint8 as numba_uint8,
    uint16 as numba_uint16,
    uint32 as numba_uint32,
    uint64 as numba_uint64,
    int64 as numba_int64,
)


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
def _predictor_decode_kernel_u8(data, width, height, samples_per_pixel):
    """Undo predictor=2 for 8-bit samples, one thread per row.

    Stride is ``samples_per_pixel`` bytes.  Byte-wise modular sum is
    correct here because each sample fits in a single byte.
    """
    row = cuda.grid(1)
    if row >= height:
        return

    row_bytes = width * samples_per_pixel
    row_start = row * row_bytes

    for col in range(samples_per_pixel, row_bytes):
        idx = row_start + col
        data[idx] = numba_uint8(
            (numba_int32(data[idx]) + numba_int32(data[idx - samples_per_pixel])) & 0xFF)


@cuda.jit
def _predictor_decode_kernel_u16(view, width, height, samples_per_pixel):
    """Undo predictor=2 on a uint16 view, one thread per row."""
    row = cuda.grid(1)
    if row >= height:
        return

    row_samples = width * samples_per_pixel
    row_start = row * row_samples

    for col in range(samples_per_pixel, row_samples):
        idx = row_start + col
        view[idx] = (view[idx] + view[idx - samples_per_pixel]) & numba_int32(0xFFFF)


@cuda.jit
def _predictor_decode_kernel_u32(view, width, height, samples_per_pixel):
    """Undo predictor=2 on a uint32 view, one thread per row."""
    row = cuda.grid(1)
    if row >= height:
        return

    row_samples = width * samples_per_pixel
    row_start = row * row_samples

    for col in range(samples_per_pixel, row_samples):
        idx = row_start + col
        view[idx] = (view[idx] + view[idx - samples_per_pixel]) & numba_uint32(0xFFFFFFFF)


@cuda.jit
def _predictor_decode_kernel_u64(view, width, height, samples_per_pixel):
    """Undo predictor=2 on a uint64 view, one thread per row."""
    row = cuda.grid(1)
    if row >= height:
        return

    row_samples = width * samples_per_pixel
    row_start = row * row_samples

    for col in range(samples_per_pixel, row_samples):
        idx = row_start + col
        view[idx] = view[idx] + view[idx - samples_per_pixel]


@cuda.jit
def _fp_predictor_decode_kernel(data, tmp, width, height, bps, big_endian):
    """Undo floating-point predictor (predictor=3), one thread per row.

    data: flat uint8 device array
    tmp: scratch buffer, same size as data
    big_endian: when True, place the MSB lane at byte index 0 of each
        output sample (file is big-endian); when False, place it at
        byte index ``bps-1`` (file is little-endian).
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

    # Step 2: un-transpose byte lanes back to the file's native sample
    # order.  Lane 0 always contains the MSB byte (TIFF Tech Note 3); the
    # MSB lands at byte index 0 (BE) or bps-1 (LE) of each output sample.
    if big_endian:
        for sample in range(width):
            for b in range(bps):
                tmp[start + sample * bps + b] = data[start + b * width + sample]
    else:
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
# KvikIO GDS (GPUDirect Storage) -- read file directly to GPU
# ---------------------------------------------------------------------------

def _batched_d2h_to_bytes(d_tiles):
    """Copy a list of cupy.uint8 1-D buffers to host as a list of ``bytes``.

    Issues one concat + one D2H transfer instead of per-tile ``.get()``
    calls, which serialise on the default stream and where the per-DMA
    setup overhead dominates wall time when there are many tiles.

    Mirrors the H2D batched-upload pattern in ``_try_nvcomp_decompress``
    (see "Batch host->device upload" near the deflate/zstd batch
    decompress branch). Same shape, opposite direction.

    Parameters
    ----------
    d_tiles : list of cupy.ndarray
        1-D ``cupy.uint8`` arrays. Sizes may differ between tiles.

    Returns
    -------
    list of bytes
        One ``bytes`` object per input tile, in the same order.
    """
    if len(d_tiles) == 0:
        return []

    import cupy

    sizes = [int(t.size) for t in d_tiles]
    offsets = np.empty(len(d_tiles) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])

    # The concat allocates a fresh device buffer of sum(sizes) bytes --
    # a peak-VRAM bump that the prior per-tile .get() loop avoided.
    # Fail early with a clear message if there isn't headroom for it.
    total_bytes = int(offsets[-1])
    _check_gpu_memory(total_bytes, what="batched D2H staging buffer")

    combined = cupy.concatenate(d_tiles)
    host_buf = combined.get()  # one D2H DMA for the whole batch

    return [
        bytes(host_buf[offsets[i]:offsets[i + 1]])
        for i in range(len(d_tiles))
    ]


def _try_kvikio_read_tiles(file_path, tile_offsets, tile_byte_counts, tile_bytes):
    """Read compressed tile bytes directly from SSD to GPU via GDS.

    When kvikio is available and GDS is supported, file data is DMA'd
    directly from the NVMe drive to GPU VRAM, bypassing CPU entirely.
    Falls back to None if kvikio is not installed or GDS is not available.

    Returns list of cupy arrays (one per tile) on GPU, or None.
    """
    try:
        import kvikio
        import cupy
    except ImportError:
        return None

    try:
        d_tiles = []
        with kvikio.CuFile(file_path, 'r') as f:
            for off, bc in zip(tile_offsets, tile_byte_counts):
                buf = cupy.empty(bc, dtype=cupy.uint8)
                nbytes = f.pread(buf, file_offset=off)
                # Verify the read completed correctly
                actual = nbytes.get() if hasattr(nbytes, 'get') else int(nbytes)
                if actual != bc:
                    return None  # partial read, fall back
                d_tiles.append(buf)
        cupy.cuda.Device().synchronize()
        return d_tiles
    except Exception:
        # GDS not available, version mismatch, or CUDA error
        # Reset CUDA error state if possible
        try:
            import cupy
            cupy.cuda.Device().synchronize()
        except Exception:
            pass
        return None


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
        # Fall back to kvikio.nvcomp. We only use DeflateManager here, so
        # ZSTD (compression=50000) is not supported through this path --
        # let the caller pick another decoder rather than feed ZSTD bytes
        # into a Deflate manager (which would also strip what looks like a
        # zlib header from a ZSTD frame).
        if compression == 50000:
            return None
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
            # Batch host->device upload: concatenate all tiles into one host
            # buffer, then a single cupy.asarray transfer. Mirrors the
            # LZW/Deflate concat-then-upload pattern below (~L1714-1722).
            comp_sizes = [len(t) for t in raw_tiles]
            comp_offsets = np.zeros(len(raw_tiles), dtype=np.int64)
            for i in range(1, len(raw_tiles)):
                comp_offsets[i] = comp_offsets[i - 1] + comp_sizes[i - 1]
            total_comp = sum(comp_sizes)
            comp_buf_host = np.empty(total_comp, dtype=np.uint8)
            for i, tile in enumerate(raw_tiles):
                comp_buf_host[comp_offsets[i]:comp_offsets[i] + comp_sizes[i]] = \
                    np.frombuffer(tile, dtype=np.uint8)
            d_comp = cupy.asarray(comp_buf_host)
            # Build per-tile device views as slices of the single buffer so
            # nvcomp's list-of-arrays API gets device pointers without extra
            # H2D transfers.
            d_compressed = [
                d_comp[comp_offsets[i]:comp_offsets[i] + comp_sizes[i]]
                for i in range(len(raw_tiles))
            ]
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
            # backend=2 (CUDA) works on all GPUs; backend=1 (HW) needs Ada/Hopper
            opts = _NvcompDeflateDecompOpts(backend=2, sort_before_hw_decompress=0,
                                            reserved=b'\x00' * 56)
        elif compression == 50000:  # ZSTD
            raw_tiles = list(compressed_tiles)  # no header stripping
            get_temp_fn = 'nvcompBatchedZstdDecompressGetTempSizeAsync'
            decomp_fn = 'nvcompBatchedZstdDecompressAsync'
            opts = _NvcompDecompOpts(backend=0, reserved=b'\x00' * 60)
        else:
            return None

        # Batch host->device upload: concatenate all compressed tiles into a
        # single host buffer, do one cupy.asarray transfer, then derive
        # per-tile device pointers as base_ptr + offsets. Mirrors the
        # LZW/Deflate concat-then-upload pattern below (~L1714-1722).
        # Per-tile cupy.asarray was measured at 256x64KB -> 6.07 ms vs 3.65 ms
        # for the batched form (~1.66x speedup, scales worse with more tiles).
        comp_sizes_list = [len(t) for t in raw_tiles]
        comp_offsets_h = np.zeros(n_tiles, dtype=np.int64)
        for i in range(1, n_tiles):
            comp_offsets_h[i] = comp_offsets_h[i - 1] + comp_sizes_list[i - 1]
        total_comp = sum(comp_sizes_list)

        comp_buf_host = np.empty(total_comp, dtype=np.uint8)
        for i, tile in enumerate(raw_tiles):
            comp_buf_host[comp_offsets_h[i]:comp_offsets_h[i] + comp_sizes_list[i]] = \
                np.frombuffer(tile, dtype=np.uint8)

        d_comp = cupy.asarray(comp_buf_host)
        d_decomp = cupy.empty(n_tiles * tile_bytes, dtype=cupy.uint8)

        base_comp_ptr = int(d_comp.data.ptr)
        base_decomp_ptr = int(d_decomp.data.ptr)
        d_comp_ptrs = cupy.asarray(
            base_comp_ptr + comp_offsets_h.astype(np.uint64))
        decomp_offsets_h = (np.arange(n_tiles, dtype=np.uint64)
                            * np.uint64(tile_bytes))
        d_decomp_ptrs = cupy.asarray(base_decomp_ptr + decomp_offsets_h)
        d_comp_sizes = cupy.asarray(
            np.array(comp_sizes_list, dtype=np.uint64))
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

        return d_decomp

    except Exception:
        return None


# ---------------------------------------------------------------------------
# nvJPEG batch decode/encode (optional, GPU-accelerated JPEG)
# ---------------------------------------------------------------------------

def _find_nvjpeg_lib():
    """Find and load libnvjpeg.so from the CUDA toolkit. Returns CDLL or None."""
    import ctypes
    import os

    search_paths = [
        'libnvjpeg.so',  # system LD_LIBRARY_PATH
    ]

    # CUDA toolkit path
    cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH', ''))
    if cuda_home:
        for subdir in ('lib64', 'lib'):
            search_paths.append(os.path.join(cuda_home, subdir, 'libnvjpeg.so'))

    # Conda env
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        search_paths.append(os.path.join(conda_prefix, 'lib', 'libnvjpeg.so'))

    # Common CUDA toolkit install locations
    for ver_dir in ('/usr/local/cuda/lib64', '/usr/local/cuda/lib'):
        search_paths.append(os.path.join(ver_dir, 'libnvjpeg.so'))

    for path in search_paths:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


_nvjpeg_lib = None
_nvjpeg_checked = False


def _get_nvjpeg():
    """Get the nvJPEG library handle (cached). Returns CDLL or None."""
    global _nvjpeg_lib, _nvjpeg_checked
    if not _nvjpeg_checked:
        _nvjpeg_checked = True
        _nvjpeg_lib = _find_nvjpeg_lib()
    return _nvjpeg_lib


# nvJPEG status codes
_NVJPEG_STATUS_SUCCESS = 0

# nvJPEG output formats. Values must match ``nvjpegOutputFormat_t`` in
# ``nvjpeg.h`` (CUDA Toolkit). They were previously off-by-two, which made
# ``_NVJPEG_OUTPUT_RGBI`` resolve to the SDK's ``NVJPEG_OUTPUT_RGB`` (planar)
# constant. nvJPEG then dereferenced ``channel[1]``/``channel[2]`` of the
# output struct, both of which the wrappers below set to NULL for
# interleaved layouts, producing an out-of-bounds GPU write inside
# ``ycbcr_to_format_kernel_roi`` and a sticky ``cudaErrorIllegalAddress``
# (issue #1549).
_NVJPEG_OUTPUT_UNCHANGED = 0  # source colorspace (channel[0] only for Y)
_NVJPEG_OUTPUT_Y = 2         # luma plane only
_NVJPEG_OUTPUT_RGB = 3       # planar RGB
_NVJPEG_OUTPUT_RGBI = 5      # interleaved RGB (R0G0B0 R1G1B1 ...)

# nvJPEG backend
_NVJPEG_BACKEND_DEFAULT = 0
_NVJPEG_BACKEND_GPU_HYBRID = 2


def _try_nvjpeg_batch_decode(compressed_tiles, tile_width, tile_height,
                              samples):
    """Try batch JPEG decode via nvJPEG. Returns CuPy buffer or None.

    Decodes all JPEG tiles on GPU in one batched call. Falls back to None
    if nvJPEG is unavailable or any decode fails.
    """
    lib = _get_nvjpeg()
    if lib is None:
        return None

    import ctypes
    import cupy

    try:
        n_tiles = len(compressed_tiles)
        tile_pixels = tile_width * tile_height
        tile_bytes = tile_pixels * samples  # JPEG is always uint8

        # nvJPEG handle type (opaque pointer)
        nvjpeg_handle = ctypes.c_void_p()

        # nvjpegCreateSimple(&handle)
        create_fn = getattr(lib, 'nvjpegCreateSimple', None)
        if create_fn is None:
            return None
        create_fn.restype = ctypes.c_int
        status = create_fn(ctypes.byref(nvjpeg_handle))
        if status != _NVJPEG_STATUS_SUCCESS:
            return None

        try:
            # Create JPEG state: nvjpegJpegStateCreate(handle, &state)
            jpeg_state = ctypes.c_void_p()
            state_create = getattr(lib, 'nvjpegJpegStateCreate')
            state_create.restype = ctypes.c_int
            status = state_create(nvjpeg_handle, ctypes.byref(jpeg_state))
            if status != _NVJPEG_STATUS_SUCCESS:
                return None

            try:
                # Decode tiles one at a time using the simple API.
                # nvJPEG batch API requires more setup; the simple decode
                # is still GPU-accelerated and avoids complex state management.
                output_format = _NVJPEG_OUTPUT_RGBI if samples == 3 else _NVJPEG_OUTPUT_UNCHANGED

                # nvjpegImage_t: array of 4 channel pointers + 4 pitches
                class _NvjpegImage(ctypes.Structure):
                    _fields_ = [
                        ('channel', ctypes.c_void_p * 4),
                        ('pitch', ctypes.c_size_t * 4),
                    ]

                _check_gpu_memory(n_tiles * tile_bytes,
                                  what="nvJPEG output buffer")
                d_all = cupy.empty(n_tiles * tile_bytes, dtype=cupy.uint8)

                decode_fn = getattr(lib, 'nvjpegDecode')
                decode_fn.restype = ctypes.c_int

                for i, tile_data in enumerate(compressed_tiles):
                    d_out = d_all[i * tile_bytes:(i + 1) * tile_bytes]

                    nv_img = _NvjpegImage()
                    nv_img.channel[0] = ctypes.c_void_p(d_out.data.ptr)
                    for ch in range(1, 4):
                        nv_img.channel[ch] = ctypes.c_void_p(0)
                    nv_img.pitch[0] = ctypes.c_size_t(tile_width * samples)
                    for ch in range(1, 4):
                        nv_img.pitch[ch] = ctypes.c_size_t(0)

                    src = tile_data if isinstance(tile_data, bytes) else bytes(tile_data)

                    status = decode_fn(
                        nvjpeg_handle,
                        jpeg_state,
                        ctypes.c_char_p(src),
                        ctypes.c_size_t(len(src)),
                        ctypes.c_int(output_format),
                        ctypes.byref(nv_img),
                        ctypes.c_void_p(0),  # default CUDA stream
                    )
                    if status != _NVJPEG_STATUS_SUCCESS:
                        return None
                    # nvjpegDecode is asynchronous on the default stream
                    # (we pass stream=0 above); the shared jpeg_state must
                    # not be reused for the next tile until this decode is
                    # complete.  Sync only the default stream so concurrent
                    # work on other streams isn't blocked -- the data
                    # dependency is on jpeg_state, not on the whole device.
                    cupy.cuda.Stream.null.synchronize()

                return d_all

            finally:
                destroy_state = getattr(lib, 'nvjpegJpegStateDestroy', None)
                if destroy_state is not None:
                    destroy_state(jpeg_state)
        finally:
            destroy_fn = getattr(lib, 'nvjpegDestroy', None)
            if destroy_fn is not None:
                destroy_fn(nvjpeg_handle)

    except Exception:
        return None


def _nvjpeg_batch_encode(d_tile_bufs, tile_width, tile_height, samples,
                          quality=75):
    """Encode tiles as JPEG on GPU via nvJPEG. Returns list of bytes or None.

    Each tile must be a CuPy uint8 array of interleaved pixel data.
    """
    lib = _get_nvjpeg()
    if lib is None:
        return None

    import ctypes
    import cupy

    try:
        n_tiles = len(d_tile_bufs)

        nvjpeg_handle = ctypes.c_void_p()
        create_fn = getattr(lib, 'nvjpegCreateSimple', None)
        if create_fn is None:
            return None
        create_fn.restype = ctypes.c_int
        status = create_fn(ctypes.byref(nvjpeg_handle))
        if status != _NVJPEG_STATUS_SUCCESS:
            return None

        try:
            # Create encoder state and params
            enc_state = ctypes.c_void_p()
            enc_state_create = getattr(lib, 'nvjpegEncoderStateCreate', None)
            if enc_state_create is None:
                return None
            enc_state_create.restype = ctypes.c_int
            status = enc_state_create(
                nvjpeg_handle, ctypes.byref(enc_state),
                ctypes.c_void_p(0))  # default stream
            if status != _NVJPEG_STATUS_SUCCESS:
                return None

            try:
                enc_params = ctypes.c_void_p()
                params_create = getattr(lib, 'nvjpegEncoderParamsCreate')
                params_create.restype = ctypes.c_int
                status = params_create(
                    nvjpeg_handle, ctypes.byref(enc_params),
                    ctypes.c_void_p(0))
                if status != _NVJPEG_STATUS_SUCCESS:
                    return None

                try:
                    # Set quality
                    set_quality = getattr(lib, 'nvjpegEncoderParamsSetQuality')
                    set_quality.restype = ctypes.c_int
                    set_quality(enc_params, ctypes.c_int(quality),
                                ctypes.c_void_p(0))

                    # Set interleaved sampling
                    set_sampling = getattr(lib, 'nvjpegEncoderParamsSetSamplingFactors', None)
                    # 0 = NVJPEG_CSS_444
                    if set_sampling is not None:
                        set_sampling.restype = ctypes.c_int
                        set_sampling(enc_params, ctypes.c_int(0),
                                     ctypes.c_void_p(0))

                    class _NvjpegImage(ctypes.Structure):
                        _fields_ = [
                            ('channel', ctypes.c_void_p * 4),
                            ('pitch', ctypes.c_size_t * 4),
                        ]

                    # Choose input format
                    input_format = _NVJPEG_OUTPUT_RGBI if samples == 3 else _NVJPEG_OUTPUT_UNCHANGED

                    encode_fn = getattr(lib, 'nvjpegEncodeImage')
                    encode_fn.restype = ctypes.c_int

                    retrieve_fn = getattr(lib, 'nvjpegEncodeRetrieveBitstream')
                    retrieve_fn.restype = ctypes.c_int

                    result = []
                    for d_tile in d_tile_bufs:
                        nv_img = _NvjpegImage()
                        nv_img.channel[0] = ctypes.c_void_p(d_tile.data.ptr)
                        for ch in range(1, 4):
                            nv_img.channel[ch] = ctypes.c_void_p(0)
                        nv_img.pitch[0] = ctypes.c_size_t(tile_width * samples)
                        for ch in range(1, 4):
                            nv_img.pitch[ch] = ctypes.c_size_t(0)

                        status = encode_fn(
                            nvjpeg_handle, enc_state, enc_params,
                            ctypes.byref(nv_img),
                            ctypes.c_int(input_format),
                            ctypes.c_int(tile_width),
                            ctypes.c_int(tile_height),
                            ctypes.c_void_p(0),  # default stream
                        )
                        if status != _NVJPEG_STATUS_SUCCESS:
                            return None

                        cupy.cuda.Device().synchronize()

                        # Get compressed size
                        length = ctypes.c_size_t(0)
                        status = retrieve_fn(
                            nvjpeg_handle, enc_state,
                            ctypes.c_void_p(0),  # NULL = query size
                            ctypes.byref(length),
                            ctypes.c_void_p(0),
                        )
                        if status != _NVJPEG_STATUS_SUCCESS:
                            return None

                        # Retrieve compressed data
                        buf = ctypes.create_string_buffer(length.value)
                        status = retrieve_fn(
                            nvjpeg_handle, enc_state,
                            buf,
                            ctypes.byref(length),
                            ctypes.c_void_p(0),
                        )
                        if status != _NVJPEG_STATUS_SUCCESS:
                            return None

                        result.append(buf.raw[:length.value])

                    return result

                finally:
                    params_destroy = getattr(lib, 'nvjpegEncoderParamsDestroy', None)
                    if params_destroy is not None:
                        params_destroy(enc_params)
            finally:
                state_destroy = getattr(lib, 'nvjpegEncoderStateDestroy', None)
                if state_destroy is not None:
                    state_destroy(enc_state)
        finally:
            destroy_fn = getattr(lib, 'nvjpegDestroy', None)
            if destroy_fn is not None:
                destroy_fn(nvjpeg_handle)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# High-level GPU decode pipeline
# ---------------------------------------------------------------------------

def gpu_decode_tiles_from_file(
    file_path: str,
    tile_offsets: list | tuple,
    tile_byte_counts: list | tuple,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    compression: int,
    predictor: int,
    dtype: np.dtype,
    samples: int = 1,
    byte_order: str = '<',
    masked_fill=None,
):
    """Decode tiles from a file, using GDS if available.

    Tries KvikIO GDS (SSD → GPU direct) first, then falls back to
    CPU mmap + gpu_decode_tiles.
    """
    import cupy

    # Try GDS: read compressed tiles directly from SSD to GPU
    d_tiles = _try_kvikio_read_tiles(
        file_path, tile_offsets, tile_byte_counts,
        tile_width * tile_height * dtype.itemsize * samples)

    if d_tiles is not None:
        # Tiles are already on GPU as cupy arrays.
        # Try nvCOMP batch decompress on them directly.
        tile_bytes = tile_width * tile_height * dtype.itemsize * samples

        if compression in (50000,) and _get_nvcomp() is not None:
            # ZSTD: nvCOMP can decompress directly from GPU buffers
            result = _try_nvcomp_from_device_bufs(
                d_tiles, tile_bytes, compression)
            if result is not None:
                decomp_offsets = np.arange(len(d_tiles), dtype=np.int64) * tile_bytes
                d_decomp = result
                d_decomp_offsets = cupy.asarray(decomp_offsets)
                # Apply predictor + assemble (shared code below)
                return _apply_predictor_and_assemble(
                    d_decomp, d_decomp_offsets, len(d_tiles),
                    tile_width, tile_height, image_width, image_height,
                    predictor, dtype, samples, tile_bytes,
                    byte_order=byte_order)

        # GDS read succeeded but nvCOMP can't decompress on GPU,
        # or it's LZW/deflate. Copy tiles to host and use normal path.
        compressed_tiles = _batched_d2h_to_bytes(d_tiles)
    else:
        # No GDS -- read tiles via CPU mmap (caller provides bytes)
        # This path is used when called from gpu_decode_tiles()
        return None  # signal caller to use the bytes-based path

    return gpu_decode_tiles(
        compressed_tiles, tile_width, tile_height,
        image_width, image_height, compression, predictor, dtype, samples,
        byte_order=byte_order, masked_fill=masked_fill)


def _try_nvcomp_from_device_bufs(d_tiles, tile_bytes, compression):
    """Run nvCOMP batch decompress on tiles already in GPU memory.

    The decompressed output uses a single contiguous device buffer with
    per-tile pointers derived as ``base_ptr + i * tile_bytes``. The previous
    implementation allocated N separate ``cupy.empty(tile_bytes)`` buffers
    and ran ``cupy.concatenate`` after the decompress kernel; that pattern
    kept two copies of the decompressed data alive at once (the per-tile
    buffers and the concatenated result) and ran a serial concat that the
    rest of the GPU paths avoid. The other nvCOMP code paths in this module
    (LZW at ~L1847, deflate at ~L1878, host-buffer at ~L1114) already use
    the single-buffer pattern; this brings the GDS path in line with them.
    See issue #1659.
    """
    import ctypes
    import cupy

    lib = _get_nvcomp()
    if lib is None:
        return None

    class _NvcompDecompOpts(ctypes.Structure):
        _fields_ = [('backend', ctypes.c_int), ('reserved', ctypes.c_char * 60)]

    try:
        n = len(d_tiles)
        # Single contiguous output buffer. nvCOMP's batched decompress takes
        # an array of per-tile device pointers; derive those from the base
        # pointer + per-tile byte offsets so we never allocate N small
        # buffers (one cupy.empty per tile is ~tens of microseconds each)
        # and never run a trailing cupy.concatenate.
        _check_gpu_memory(n * tile_bytes,
                          what="GDS+nvCOMP decompressed buffer")
        d_decomp = cupy.empty(n * tile_bytes, dtype=cupy.uint8)
        base_decomp_ptr = int(d_decomp.data.ptr)
        decomp_offsets = (np.arange(n, dtype=np.uint64)
                          * np.uint64(tile_bytes))
        d_decomp_ptrs = cupy.asarray(base_decomp_ptr + decomp_offsets)

        d_comp_ptrs = cupy.array([t.data.ptr for t in d_tiles], dtype=cupy.uint64)
        d_comp_sizes = cupy.array([t.size for t in d_tiles], dtype=cupy.uint64)
        d_buf_sizes = cupy.full(n, tile_bytes, dtype=cupy.uint64)
        d_actual = cupy.empty(n, dtype=cupy.uint64)

        opts = _NvcompDecompOpts(backend=0, reserved=b'\x00' * 60)

        fn_name = {50000: 'nvcompBatchedZstdDecompressGetTempSizeAsync'}.get(compression)
        dec_name = {50000: 'nvcompBatchedZstdDecompressAsync'}.get(compression)
        if fn_name is None:
            return None

        temp_fn = getattr(lib, fn_name)
        temp_fn.restype = ctypes.c_int
        temp_size = ctypes.c_size_t(0)
        s = temp_fn(n, tile_bytes, opts, ctypes.byref(temp_size), n * tile_bytes)
        if s != 0:
            return None

        ts = max(temp_size.value, 1)
        d_temp = cupy.empty(ts, dtype=cupy.uint8)
        d_statuses = cupy.zeros(n, dtype=cupy.int32)

        dec_fn = getattr(lib, dec_name)
        dec_fn.restype = ctypes.c_int
        s = dec_fn(
            ctypes.c_void_p(d_comp_ptrs.data.ptr),
            ctypes.c_void_p(d_comp_sizes.data.ptr),
            ctypes.c_void_p(d_buf_sizes.data.ptr),
            ctypes.c_void_p(d_actual.data.ptr),
            ctypes.c_size_t(n),
            ctypes.c_void_p(d_temp.data.ptr), ctypes.c_size_t(ts),
            ctypes.c_void_p(d_decomp_ptrs.data.ptr),
            opts,
            ctypes.c_void_p(d_statuses.data.ptr),
            ctypes.c_void_p(0),
        )
        if s != 0:
            return None

        cupy.cuda.Device().synchronize()
        if int(cupy.any(d_statuses != 0)):
            return None

        return d_decomp
    except Exception:
        return None


def _gpu_predictor2_decode(d_decomp, tile_width, total_rows, dtype, samples):
    """Run the right predictor=2 decode kernel for *dtype*.

    TIFF predictor=2 differences adjacent same-component samples in the
    sample's natural width (uint8/16/32/64).  We view the byte buffer as
    the matching unsigned dtype and dispatch to a per-width kernel so
    the modular wrap matches what GDAL/libtiff write.
    """
    import cupy

    tpb = min(256, total_rows) if total_rows > 0 else 1
    bpg = math.ceil(total_rows / tpb) if tpb > 0 else 1
    bps = dtype.itemsize

    if bps == 1:
        _predictor_decode_kernel_u8[bpg, tpb](
            d_decomp, tile_width, total_rows, samples)
    elif bps == 2:
        view = d_decomp.view(cupy.uint16)
        _predictor_decode_kernel_u16[bpg, tpb](
            view, tile_width, total_rows, samples)
    elif bps == 4:
        view = d_decomp.view(cupy.uint32)
        _predictor_decode_kernel_u32[bpg, tpb](
            view, tile_width, total_rows, samples)
    elif bps == 8:
        view = d_decomp.view(cupy.uint64)
        _predictor_decode_kernel_u64[bpg, tpb](
            view, tile_width, total_rows, samples)
    else:
        raise ValueError(
            f"GPU predictor=2 unsupported for bytes_per_sample={bps}")
    cuda.synchronize()


def _gpu_predictor2_encode(d_decomp, tile_width, total_rows, dtype, samples):
    """Run the right predictor=2 encode kernel for *dtype*."""
    import cupy

    tpb = min(256, total_rows) if total_rows > 0 else 1
    bpg = math.ceil(total_rows / tpb) if tpb > 0 else 1
    bps = dtype.itemsize

    if bps == 1:
        _predictor_encode_kernel_u8[bpg, tpb](
            d_decomp, tile_width, total_rows, samples)
    elif bps == 2:
        view = d_decomp.view(cupy.uint16)
        _predictor_encode_kernel_u16[bpg, tpb](
            view, tile_width, total_rows, samples)
    elif bps == 4:
        view = d_decomp.view(cupy.uint32)
        _predictor_encode_kernel_u32[bpg, tpb](
            view, tile_width, total_rows, samples)
    elif bps == 8:
        view = d_decomp.view(cupy.uint64)
        _predictor_encode_kernel_u64[bpg, tpb](
            view, tile_width, total_rows, samples)
    else:
        raise ValueError(
            f"GPU predictor=2 unsupported for bytes_per_sample={bps}")
    cuda.synchronize()


def _apply_predictor_and_assemble(d_decomp, d_decomp_offsets, n_tiles,
                                    tile_width, tile_height,
                                    image_width, image_height,
                                    predictor, dtype, samples, tile_bytes,
                                    byte_order: str = '<'):
    """Apply predictor decode and tile assembly on GPU."""
    import cupy

    bytes_per_pixel = dtype.itemsize * samples
    big_endian = (byte_order == '>')

    if predictor == 2:
        total_rows = n_tiles * tile_height
        # Predictor=2 differences adjacent samples at the sample's natural
        # width (uint8/16/32/64). The per-dtype kernels view the byte
        # buffer as native unsigned dtype, so on big-endian files we must
        # swap the bytes to native order BEFORE running the kernel,
        # otherwise the prefix-sum runs on the wrong integer
        # interpretation (#1517). The pre-swap then makes the post-
        # assembly byteswap unnecessary; see the BE branch below.
        if big_endian and dtype.itemsize > 1:
            _swap_byte_lanes(d_decomp, dtype.itemsize)
        # Sample-level differencing: stride is samples_per_pixel samples,
        # row width is tile_width pixels.  Per-dtype kernels handle the
        # modular wrap at the sample's natural bit width.
        _gpu_predictor2_decode(
            d_decomp, tile_width, total_rows, dtype, samples)
    elif predictor == 3:
        total_rows = n_tiles * tile_height
        tpb = min(256, total_rows)
        bpg = math.ceil(total_rows / tpb)
        d_tmp = cupy.empty_like(d_decomp)
        _fp_predictor_decode_kernel[bpg, tpb](
            d_decomp, d_tmp, tile_width * samples, total_rows,
            dtype.itemsize, big_endian)
        cuda.synchronize()

    tiles_across = math.ceil(image_width / tile_width)
    total_pixels = image_width * image_height
    _check_gpu_memory(total_pixels * bytes_per_pixel,
                      what="full-image output buffer")
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

    if samples > 1:
        out = d_output.view(dtype=cupy.dtype(dtype)).reshape(
            image_height, image_width, samples)
    else:
        out = d_output.view(dtype=cupy.dtype(dtype)).reshape(
            image_height, image_width)
    # Predictor=2 BE swapped d_decomp to native order pre-decode (#1517),
    # so the assembled output is already native; skip the final swap.
    needs_post_swap = (
        big_endian and dtype.itemsize > 1 and predictor != 2
    )
    if needs_post_swap:
        # See gpu_decode_tiles for why BE samples need a final byteswap.
        # cupy.ndarray has no .byteswap(), so use the dtype-view helper.
        out = _xp_byteswap(out)
    return out


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
    byte_order: str = '<',
    masked_fill=None,
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
    masked_fill : scalar or None
        Value to write into pixels that LERC reports as invalid.  Only
        consulted for LERC-compressed inputs (tag 34887).  ``None``
        leaves any masked pixels at LERC's zero fill.  Use
        ``_resolve_masked_fill(ifd.nodata_str, dtype)`` from
        ``_reader.py`` to mirror the CPU reader's nodata semantics.

    Returns
    -------
    cupy.ndarray
        Decoded image on GPU device.
    """
    import cupy

    n_tiles = len(compressed_tiles)
    bytes_per_pixel = dtype.itemsize * samples
    tile_bytes = tile_width * tile_height * bytes_per_pixel

    # Per-tile LERC valid masks; populated only by the LERC branch
    # below.  Kept as a flat local so the post-assembly fill block at
    # the end of this function can find it regardless of which decode
    # branch ran.
    _lerc_masks: list[np.ndarray | None] | None = None

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
        _check_gpu_memory(n_tiles * tile_bytes, what="tile decode buffer")
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
        _check_gpu_memory(n_tiles * tile_bytes, what="tile decode buffer")
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

    elif compression == 7:  # JPEG
        # Try nvJPEG GPU decode first, fall back to CPU Pillow
        nvjpeg_result = _try_nvjpeg_batch_decode(
            compressed_tiles, tile_width, tile_height, samples)
        if nvjpeg_result is not None:
            d_decomp = nvjpeg_result
            decomp_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
            d_decomp_offsets = cupy.asarray(decomp_offsets)
        else:
            from ._compression import jpeg_decompress
            raw_host = np.empty(n_tiles * tile_bytes, dtype=np.uint8)
            for i, tile in enumerate(compressed_tiles):
                start = i * tile_bytes
                decoded = np.frombuffer(
                    jpeg_decompress(tile, tile_width, tile_height, samples),
                    dtype=np.uint8)
                n = min(len(decoded), tile_bytes)
                raw_host[start:start + n] = decoded[:n]
                if n < tile_bytes:
                    raw_host[start + n:start + tile_bytes] = 0
            d_decomp = cupy.asarray(raw_host)
            decomp_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
            d_decomp_offsets = cupy.asarray(decomp_offsets)

    elif compression == 34712:  # JPEG 2000
        nvj2k_result = _try_nvjpeg2k_batch_decode(
            compressed_tiles, tile_width, tile_height, dtype, samples)
        if nvj2k_result is not None:
            d_decomp = nvj2k_result
            decomp_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
            d_decomp_offsets = cupy.asarray(decomp_offsets)
        else:
            # CPU fallback for JPEG 2000
            from ._compression import jpeg2000_decompress
            raw_host = np.empty(n_tiles * tile_bytes, dtype=np.uint8)
            for i, tile in enumerate(compressed_tiles):
                start = i * tile_bytes
                chunk = np.frombuffer(
                    jpeg2000_decompress(
                        tile, tile_width, tile_height, samples,
                        expected_size=tile_bytes),
                    dtype=np.uint8)
                raw_host[start:start + min(len(chunk), tile_bytes)] = \
                    chunk[:tile_bytes] if len(chunk) >= tile_bytes else \
                    np.pad(chunk, (0, tile_bytes - len(chunk)))
            d_decomp = cupy.asarray(raw_host)
            decomp_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
            d_decomp_offsets = cupy.asarray(decomp_offsets)

    elif compression == 34887:  # LERC
        from ._compression import lerc_decompress_with_mask
        raw_host = np.empty(n_tiles * tile_bytes, dtype=np.uint8)
        # Per-tile valid masks captured from LERC.  None entries mean
        # "all valid" (LERC returned no mask, or an all-True mask).
        # The host-side mask buffer is materialised lazily: it stays
        # None until at least one tile reports a real partial mask.
        per_tile_masks: list[np.ndarray | None] = [None] * n_tiles
        any_lerc_mask = False
        for i, tile in enumerate(compressed_tiles):
            start = i * tile_bytes
            decoded_bytes, valid_mask = lerc_decompress_with_mask(
                tile, expected_size=tile_bytes)
            chunk = np.frombuffer(decoded_bytes, dtype=np.uint8)
            raw_host[start:start + min(len(chunk), tile_bytes)] = \
                chunk[:tile_bytes] if len(chunk) >= tile_bytes else \
                np.pad(chunk, (0, tile_bytes - len(chunk)))
            if valid_mask is not None:
                per_tile_masks[i] = np.asarray(valid_mask)
                any_lerc_mask = True
        d_decomp = cupy.asarray(raw_host)
        decomp_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
        d_decomp_offsets = cupy.asarray(decomp_offsets)
        if any_lerc_mask:
            # Stash per-tile masks for the post-assembly fill pass below.
            # Stored in a ``_lerc_masks`` local that the LERC fill block
            # picks up after predictor decode and tile assembly.
            _lerc_masks = per_tile_masks
        else:
            _lerc_masks = None

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
        # Unsupported GPU codec: decompress on CPU, transfer to GPU
        from ._compression import decompress as cpu_decompress
        raw_host = np.empty(n_tiles * tile_bytes, dtype=np.uint8)
        for i, tile in enumerate(compressed_tiles):
            start = i * tile_bytes
            chunk = cpu_decompress(tile, compression, tile_bytes)
            raw_host[start:start + min(len(chunk), tile_bytes)] = \
                chunk[:tile_bytes] if len(chunk) >= tile_bytes else \
                np.pad(chunk, (0, tile_bytes - len(chunk)))
        d_decomp = cupy.asarray(raw_host)
        decomp_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
        d_decomp_offsets = cupy.asarray(decomp_offsets)

    # Apply predictor on GPU
    if predictor == 2:
        # Sample-level horizontal differencing: stride is samples_per_pixel
        # samples; per-dtype kernels handle the natural-width modular wrap.
        # On big-endian multi-byte files the kernels would otherwise view
        # the buffer with the wrong integer interpretation, so swap to
        # native order first and skip the post-assembly swap below
        # (#1517).
        if byte_order == '>' and dtype.itemsize > 1:
            _swap_byte_lanes(d_decomp, dtype.itemsize)
        total_rows = n_tiles * tile_height
        _gpu_predictor2_decode(
            d_decomp, tile_width, total_rows, dtype, samples)

    elif predictor == 3:
        # Float predictor: one thread per row
        total_rows = n_tiles * tile_height
        tpb = min(256, total_rows)
        bpg = math.ceil(total_rows / tpb)
        d_tmp = cupy.empty_like(d_decomp)
        _fp_predictor_decode_kernel[bpg, tpb](
            d_decomp, d_tmp, tile_width * samples, total_rows,
            dtype.itemsize, byte_order == '>')
        cuda.synchronize()

    # Assemble tiles into output image on GPU
    tiles_across = math.ceil(image_width / tile_width)
    total_pixels = image_width * image_height
    _check_gpu_memory(total_pixels * bytes_per_pixel,
                      what="full-image output buffer")
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
        out = d_output.view(dtype=cupy.dtype(dtype)).reshape(
            image_height, image_width, samples)
    else:
        out = d_output.view(dtype=cupy.dtype(dtype)).reshape(
            image_height, image_width)
    # The decoded byte stream is in the file's byte order; cupy view
    # interprets it as native (always little-endian on supported GPUs),
    # so big-endian samples that are wider than a byte must be swapped
    # back to native before the values mean anything. Predictor=2 BE
    # already swapped the buffer pre-decode (#1517), so skip the swap
    # in that case.
    if byte_order == '>' and dtype.itemsize > 1 and predictor != 2:
        # cupy.ndarray has no .byteswap(), so use the dtype-view helper.
        out = _xp_byteswap(out)

    # LERC valid-mask fill: GDAL writes LERC TIFFs with masked pixels
    # zero-filled in the data array, so without restoring nodata here a
    # masked pixel reads back as a real zero measurement.  Mirrors the
    # CPU path in ``_decode_strip_or_tile`` (PR #1529).
    if _lerc_masks is not None and masked_fill is not None:
        out = _apply_lerc_mask_fill(
            out, _lerc_masks, tile_width, tile_height,
            image_width, image_height, samples, masked_fill)
    return out


def _apply_lerc_mask_fill(out, per_tile_masks, tile_width, tile_height,
                          image_width, image_height, samples, masked_fill):
    """Write *masked_fill* into LERC-invalid positions of an assembled image.

    Each tile contributes either an ``(h, w)``/``(h, w, samples)`` valid
    mask (1=valid, 0=invalid) or ``None`` for "all valid".  The host
    builds a single assembled boolean invalid-mask sized to the output
    image, transfers it to the GPU once, and uses it to overwrite
    masked positions on device.
    """
    import cupy

    n_tiles = len(per_tile_masks)
    tiles_across = math.ceil(image_width / tile_width)
    tiles_down = math.ceil(image_height / tile_height)
    if n_tiles != tiles_across * tiles_down:
        # Defensive: tile grid mismatch means we cannot place masks
        # safely.  Skip the fill rather than corrupt the output.
        return out

    invalid = np.zeros((image_height, image_width), dtype=bool)
    for idx, mask in enumerate(per_tile_masks):
        if mask is None:
            continue
        tr = idx // tiles_across
        tc = idx % tiles_across
        y0 = tr * tile_height
        x0 = tc * tile_width
        # Trim the tile mask to the visible portion of the image so
        # right- and bottom-edge tile padding does not leak into the
        # assembled mask.
        h = min(tile_height, image_height - y0)
        w = min(tile_width, image_width - x0)
        if h <= 0 or w <= 0:
            continue
        m = np.asarray(mask)
        # LERC may report the mask as (h, w) or (h, w, samples); for the
        # invalid-fill we collapse multi-sample masks via "any sample
        # invalid".
        if m.ndim == 3:
            m2 = (m == 0).any(axis=2)
        else:
            m2 = (m == 0)
        invalid[y0:y0 + h, x0:x0 + w] = m2[:h, :w]

    if not invalid.any():
        return out

    # Account for the boolean mask buffer up front so a borderline-OK
    # decode doesn't tip into CUDA OOM on the mask transfer. Boolean
    # indexing on cupy materialises a temporary index array (typically
    # int64 of length sum(invalid)); cap that at the worst case of
    # one int64 per pixel so the budget covers both buffers.
    _check_gpu_memory(invalid.nbytes, what="LERC valid-mask buffer")
    _check_gpu_memory(invalid.size * 8, what="LERC mask index buffer")

    d_invalid = cupy.asarray(invalid)
    if out.ndim == 3:
        # Broadcast (H, W) mask across the sample axis.
        out[d_invalid, ...] = out.dtype.type(masked_fill)
    else:
        out[d_invalid] = out.dtype.type(masked_fill)
    return out


# ---------------------------------------------------------------------------
# GPU tile extraction kernel -- image → individual tiles
# ---------------------------------------------------------------------------

@cuda.jit
def _extract_tiles_kernel(
    image,            # uint8: flat row-major image
    tile_bufs,        # uint8: output buffer (all tiles concatenated)
    tile_offsets,     # int64: byte offset of each tile in tile_bufs
    tile_width,
    tile_height,
    bytes_per_pixel,
    image_width,
    image_height,
    tiles_across,
):
    """Extract tile pixels from image into per-tile buffers, one thread per pixel."""
    pixel_idx = cuda.grid(1)
    total_pixels = image_width * image_height
    if pixel_idx >= total_pixels:
        return

    row = pixel_idx // image_width
    col = pixel_idx % image_width

    tile_row = row // tile_height
    tile_col = col // tile_width
    tile_idx = tile_row * tiles_across + tile_col

    local_row = row - tile_row * tile_height
    local_col = col - tile_col * tile_width

    src_byte = (row * image_width + col) * bytes_per_pixel
    tile_off = tile_offsets[tile_idx]
    dst_byte = tile_off + (local_row * tile_width + local_col) * bytes_per_pixel

    for b in range(bytes_per_pixel):
        tile_bufs[dst_byte + b] = image[src_byte + b]


# ---------------------------------------------------------------------------
# GPU predictor encode kernels
# ---------------------------------------------------------------------------

@cuda.jit
def _predictor_encode_kernel_u8(data, width, height, samples_per_pixel):
    """Apply predictor=2 for 8-bit samples (right-to-left), one thread per row."""
    row = cuda.grid(1)
    if row >= height:
        return

    row_bytes = width * samples_per_pixel
    row_start = row * row_bytes

    for col in range(row_bytes - 1, samples_per_pixel - 1, -1):
        idx = row_start + col
        data[idx] = numba_uint8(
            (numba_int32(data[idx]) - numba_int32(data[idx - samples_per_pixel])) & 0xFF)


@cuda.jit
def _predictor_encode_kernel_u16(view, width, height, samples_per_pixel):
    """Apply predictor=2 on a uint16 view (right-to-left), one thread per row."""
    row = cuda.grid(1)
    if row >= height:
        return

    row_samples = width * samples_per_pixel
    row_start = row * row_samples

    for col in range(row_samples - 1, samples_per_pixel - 1, -1):
        idx = row_start + col
        view[idx] = (view[idx] - view[idx - samples_per_pixel]) & numba_int32(0xFFFF)


@cuda.jit
def _predictor_encode_kernel_u32(view, width, height, samples_per_pixel):
    """Apply predictor=2 on a uint32 view (right-to-left), one thread per row."""
    row = cuda.grid(1)
    if row >= height:
        return

    row_samples = width * samples_per_pixel
    row_start = row * row_samples

    for col in range(row_samples - 1, samples_per_pixel - 1, -1):
        idx = row_start + col
        view[idx] = (view[idx] - view[idx - samples_per_pixel]) & numba_uint32(0xFFFFFFFF)


@cuda.jit
def _predictor_encode_kernel_u64(view, width, height, samples_per_pixel):
    """Apply predictor=2 on a uint64 view (right-to-left), one thread per row."""
    row = cuda.grid(1)
    if row >= height:
        return

    row_samples = width * samples_per_pixel
    row_start = row * row_samples

    for col in range(row_samples - 1, samples_per_pixel - 1, -1):
        idx = row_start + col
        view[idx] = view[idx] - view[idx - samples_per_pixel]


@cuda.jit
def _fp_predictor_encode_kernel(data, tmp, width, height, bps):
    """Apply floating-point predictor (predictor=3), one thread per row."""
    row = cuda.grid(1)
    if row >= height:
        return

    row_len = width * bps
    start = row * row_len

    # Step 1: transpose to byte-swizzled layout (MSB lane first)
    for sample in range(width):
        for b in range(bps):
            tmp[start + (bps - 1 - b) * width + sample] = data[start + sample * bps + b]

    # Copy back
    for i in range(row_len):
        data[start + i] = tmp[start + i]

    # Step 2: horizontal differencing (right to left)
    for i in range(row_len - 1, 0, -1):
        idx = start + i
        data[idx] = numba_uint8(
            (numba_int32(data[idx]) - numba_int32(data[idx - 1])) & 0xFF)


# ---------------------------------------------------------------------------
# nvCOMP batch compress
# ---------------------------------------------------------------------------

def _nvcomp_batch_compress(d_tile_bufs, tile_byte_counts, tile_bytes,
                           compression, n_tiles):
    """Compress tiles on GPU via nvCOMP. Returns list of bytes on CPU.

    Parameters
    ----------
    d_tile_bufs : list of cupy arrays
        Uncompressed tile data on GPU.
    tile_byte_counts : not used (all tiles same size)
    tile_bytes : int
        Size of each uncompressed tile in bytes.
    compression : int
        TIFF compression tag (8=deflate, 50000=ZSTD).
    n_tiles : int
        Number of tiles.

    Returns
    -------
    list of bytes
        Compressed tile data on CPU, ready for file assembly.
    """
    import ctypes
    import cupy

    lib = _get_nvcomp()
    if lib is None:
        return None

    class _CompOpts(ctypes.Structure):
        _fields_ = [('algorithm', ctypes.c_int), ('reserved', ctypes.c_char * 60)]

    class _DeflateCompOpts(ctypes.Structure):
        _fields_ = [('algorithm', ctypes.c_int), ('reserved', ctypes.c_char * 60)]

    try:
        # Select codec
        if compression == 50000:  # ZSTD
            get_max_fn = 'nvcompBatchedZstdCompressGetMaxOutputChunkSize'
            get_temp_fn = 'nvcompBatchedZstdCompressGetTempSizeAsync'
            compress_fn = 'nvcompBatchedZstdCompressAsync'
            opts = _CompOpts(algorithm=0, reserved=b'\x00' * 60)
        elif compression in (8, 32946):  # Deflate
            get_max_fn = 'nvcompBatchedDeflateCompressGetMaxOutputChunkSize'
            get_temp_fn = 'nvcompBatchedDeflateCompressGetTempSizeAsync'
            compress_fn = 'nvcompBatchedDeflateCompressAsync'
            opts = _DeflateCompOpts(algorithm=1, reserved=b'\x00' * 60)
        else:
            return None

        # Get max compressed chunk size
        max_comp_size = ctypes.c_size_t(0)
        fn = getattr(lib, get_max_fn)
        fn.restype = ctypes.c_int
        s = fn(ctypes.c_size_t(tile_bytes), opts, ctypes.byref(max_comp_size))
        if s != 0:
            return None
        max_cs = max_comp_size.value

        # Allocate compressed output buffers on device
        d_comp_bufs = [cupy.empty(max_cs, dtype=cupy.uint8) for _ in range(n_tiles)]

        # Build pointer and size arrays
        d_uncomp_ptrs = cupy.array([b.data.ptr for b in d_tile_bufs], dtype=cupy.uint64)
        d_comp_ptrs = cupy.array([b.data.ptr for b in d_comp_bufs], dtype=cupy.uint64)
        d_uncomp_sizes = cupy.full(n_tiles, tile_bytes, dtype=cupy.uint64)
        d_comp_sizes = cupy.empty(n_tiles, dtype=cupy.uint64)

        # Get temp size
        temp_size = ctypes.c_size_t(0)
        fn2 = getattr(lib, get_temp_fn)
        fn2.restype = ctypes.c_int
        s = fn2(ctypes.c_size_t(n_tiles), ctypes.c_size_t(tile_bytes),
                opts, ctypes.byref(temp_size), ctypes.c_size_t(n_tiles * tile_bytes))
        if s != 0:
            return None

        d_temp = cupy.empty(max(temp_size.value, 1), dtype=cupy.uint8)
        d_statuses = cupy.zeros(n_tiles, dtype=cupy.int32)

        # Compress
        fn3 = getattr(lib, compress_fn)
        fn3.restype = ctypes.c_int
        s = fn3(
            ctypes.c_void_p(d_uncomp_ptrs.data.ptr),
            ctypes.c_void_p(d_uncomp_sizes.data.ptr),
            ctypes.c_size_t(tile_bytes),
            ctypes.c_size_t(n_tiles),
            ctypes.c_void_p(d_temp.data.ptr),
            ctypes.c_size_t(max(temp_size.value, 1)),
            ctypes.c_void_p(d_comp_ptrs.data.ptr),
            ctypes.c_void_p(d_comp_sizes.data.ptr),
            opts,
            ctypes.c_void_p(d_statuses.data.ptr),
            ctypes.c_void_p(0),  # default stream
        )
        if s != 0:
            return None

        cupy.cuda.Device().synchronize()

        if int(cupy.any(d_statuses != 0)):
            return None

        # For deflate, compute adler32 checksums from uncompressed tiles
        # before reading compressed data (need the originals).
        # Batch the GPU->CPU transfer so all tiles move in a single DMA
        # instead of one .get() per tile (which serializes on the default
        # stream and is the dominant cost on the deflate path).
        adler_checksums = None
        if compression in (8, 32946):
            import zlib
            import struct
            adler_checksums = [None] * n_tiles
            if n_tiles > 0:
                d_contig = cupy.empty(n_tiles * tile_bytes, dtype=cupy.uint8)
                for i in range(n_tiles):
                    d_contig[i * tile_bytes:(i + 1) * tile_bytes] = \
                        d_tile_bufs[i][:tile_bytes]
                host_view = memoryview(d_contig.get())
                for i in range(n_tiles):
                    adler_checksums[i] = zlib.adler32(
                        host_view[i * tile_bytes:(i + 1) * tile_bytes])

        # Read compressed sizes and data back to CPU
        comp_sizes = d_comp_sizes.get().astype(int)
        result = []
        for i in range(n_tiles):
            cs = int(comp_sizes[i])
            raw = d_comp_bufs[i][:cs].get().tobytes()

            if adler_checksums is not None:
                # Wrap raw deflate in zlib format: header + data + adler32
                checksum = struct.pack('>I', adler_checksums[i] & 0xFFFFFFFF)
                raw = b'\x78\x9c' + raw + checksum

            result.append(raw)

        return result

    except Exception:
        return None


# ---------------------------------------------------------------------------
# nvJPEG2000 batch decode/encode (optional, GPU-accelerated JPEG 2000)
# ---------------------------------------------------------------------------

_nvjpeg2k_lib = None
_nvjpeg2k_checked = False


def _find_nvjpeg2k_lib():
    """Find and load libnvjpeg2k.so. Returns ctypes.CDLL or None."""
    import ctypes
    import os

    search_paths = [
        'libnvjpeg2k.so',  # system LD_LIBRARY_PATH
    ]

    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        search_paths.append(os.path.join(conda_prefix, 'lib', 'libnvjpeg2k.so'))

    conda_base = os.path.dirname(conda_prefix) if conda_prefix else ''
    if conda_base:
        for env in ['rapids', 'test-again', 'rtxpy-fire']:
            p = os.path.join(conda_base, env, 'lib', 'libnvjpeg2k.so')
            if os.path.exists(p):
                search_paths.append(p)

    for path in search_paths:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


def _get_nvjpeg2k():
    """Get the nvJPEG2000 library handle (cached). Returns CDLL or None."""
    global _nvjpeg2k_lib, _nvjpeg2k_checked
    if not _nvjpeg2k_checked:
        _nvjpeg2k_checked = True
        _nvjpeg2k_lib = _find_nvjpeg2k_lib()
    return _nvjpeg2k_lib


def _try_nvjpeg2k_batch_decode(compressed_tiles, tile_width, tile_height,
                                dtype, samples):
    """Try decoding JPEG 2000 tiles via nvJPEG2000. Returns list of CuPy arrays or None.

    Each tile is decoded independently. The decoded pixels are returned as a
    flat CuPy uint8 buffer (all tiles concatenated), matching the layout
    expected by _apply_predictor_and_assemble / the assembly kernel.
    """
    lib = _get_nvjpeg2k()
    if lib is None:
        return None

    import ctypes
    import cupy

    n_tiles = len(compressed_tiles)
    bytes_per_pixel = dtype.itemsize * samples
    tile_bytes = tile_width * tile_height * bytes_per_pixel

    try:
        # Create nvjpeg2k handle
        handle = ctypes.c_void_p()
        s = lib.nvjpeg2kCreateSimple(ctypes.byref(handle))
        if s != 0:
            return None

        # Create decode state and params
        state = ctypes.c_void_p()
        s = lib.nvjpeg2kDecodeStateCreate(handle, ctypes.byref(state))
        if s != 0:
            lib.nvjpeg2kDestroy(handle)
            return None

        stream = ctypes.c_void_p()
        s = lib.nvjpeg2kStreamCreate(ctypes.byref(stream))
        if s != 0:
            lib.nvjpeg2kDecodeStateDestroy(state)
            lib.nvjpeg2kDestroy(handle)
            return None

        params = ctypes.c_void_p()
        s = lib.nvjpeg2kDecodeParamsCreate(ctypes.byref(params))
        if s != 0:
            lib.nvjpeg2kStreamDestroy(stream)
            lib.nvjpeg2kDecodeStateDestroy(state)
            lib.nvjpeg2kDestroy(handle)
            return None

        # nvjpeg2kImage_t: array of pointers (pixel_data) + array of pitches
        MAX_COMPONENTS = 4

        class _NvJpeg2kImage(ctypes.Structure):
            _fields_ = [
                ('pixel_data', ctypes.c_void_p * MAX_COMPONENTS),
                ('pitch_in_bytes', ctypes.c_size_t * MAX_COMPONENTS),
                ('num_components', ctypes.c_uint32),
                ('pixel_type', ctypes.c_int),  # NVJPEG2K_UINT8=0, UINT16=1, INT16=2
            ]

        # Map numpy dtype to nvjpeg2k pixel type
        if dtype == np.uint8:
            pixel_type = 0  # NVJPEG2K_UINT8
        elif dtype == np.uint16:
            pixel_type = 1  # NVJPEG2K_UINT16
        elif dtype == np.int16:
            pixel_type = 2  # NVJPEG2K_INT16
        else:
            # Unsupported dtype for nvJPEG2000 -- fall back
            lib.nvjpeg2kDecodeParamsDestroy(params)
            lib.nvjpeg2kStreamDestroy(stream)
            lib.nvjpeg2kDecodeStateDestroy(state)
            lib.nvjpeg2kDestroy(handle)
            return None

        # Decode each tile
        d_all_tiles = cupy.empty(n_tiles * tile_bytes, dtype=cupy.uint8)

        for i, tile_data in enumerate(compressed_tiles):
            # Parse the J2K codestream
            src = np.frombuffer(tile_data, dtype=np.uint8)
            s = lib.nvjpeg2kStreamParse(
                handle,
                ctypes.c_void_p(src.ctypes.data),
                ctypes.c_size_t(len(src)),
                ctypes.c_int(0),  # save_metadata
                ctypes.c_int(0),  # save_stream
                stream,
            )
            if s != 0:
                continue

            # Allocate per-component output buffers on GPU
            comp_bufs = []
            pitch = tile_width * dtype.itemsize
            for c in range(samples):
                buf = cupy.empty(tile_height * pitch, dtype=cupy.uint8)
                comp_bufs.append(buf)

            # Build nvjpeg2kImage_t
            img = _NvJpeg2kImage()
            img.num_components = samples
            img.pixel_type = pixel_type
            for c in range(samples):
                img.pixel_data[c] = comp_bufs[c].data.ptr
                img.pitch_in_bytes[c] = pitch

            # Decode
            s = lib.nvjpeg2kDecode(
                handle, state, stream, params,
                ctypes.byref(img),
                ctypes.c_void_p(0),  # default CUDA stream
            )
            cupy.cuda.Device().synchronize()

            if s != 0:
                continue

            # Interleave components into pixel order (comp0,comp1,...) per pixel
            tile_offset = i * tile_bytes
            if samples == 1:
                d_all_tiles[tile_offset:tile_offset + tile_bytes] = comp_bufs[0][:tile_bytes]
            else:
                # Interleave: separate planes -> pixel-interleaved
                comp_arrays = [
                    comp_bufs[c][:tile_height * pitch].view(
                        dtype=cupy.dtype(dtype)).reshape(tile_height, tile_width)
                    for c in range(samples)
                ]
                interleaved = cupy.stack(comp_arrays, axis=-1)
                d_all_tiles[tile_offset:tile_offset + tile_bytes] = \
                    interleaved.view(cupy.uint8).ravel()

        # Cleanup
        lib.nvjpeg2kDecodeParamsDestroy(params)
        lib.nvjpeg2kStreamDestroy(stream)
        lib.nvjpeg2kDecodeStateDestroy(state)
        lib.nvjpeg2kDestroy(handle)

        return d_all_tiles

    except Exception:
        return None


def _nvjpeg2k_batch_encode(d_tile_bufs, tile_width, tile_height,
                            dtype, samples, n_tiles, lossless=True):
    """Encode tiles as JPEG 2000 via nvJPEG2000. Returns list of bytes or None."""
    lib = _get_nvjpeg2k()
    if lib is None:
        return None

    import ctypes
    import cupy

    try:
        bytes_per_pixel = dtype.itemsize * samples
        tile_bytes = tile_width * tile_height * bytes_per_pixel

        # Create encoder
        encoder = ctypes.c_void_p()
        s = lib.nvjpeg2kEncoderCreateSimple(ctypes.byref(encoder))
        if s != 0:
            return None

        enc_state = ctypes.c_void_p()
        s = lib.nvjpeg2kEncodeStateCreate(encoder, ctypes.byref(enc_state))
        if s != 0:
            lib.nvjpeg2kEncoderDestroy(encoder)
            return None

        enc_params = ctypes.c_void_p()
        s = lib.nvjpeg2kEncodeParamsCreate(ctypes.byref(enc_params))
        if s != 0:
            lib.nvjpeg2kEncodeStateDestroy(enc_state)
            lib.nvjpeg2kEncoderDestroy(encoder)
            return None

        # Set encoding parameters
        if lossless:
            lib.nvjpeg2kEncodeParamsSetQuality(enc_params, ctypes.c_int(1))

        MAX_COMPONENTS = 4

        class _NvJpeg2kImage(ctypes.Structure):
            _fields_ = [
                ('pixel_data', ctypes.c_void_p * MAX_COMPONENTS),
                ('pitch_in_bytes', ctypes.c_size_t * MAX_COMPONENTS),
                ('num_components', ctypes.c_uint32),
                ('pixel_type', ctypes.c_int),
            ]

        if dtype == np.uint8:
            pixel_type = 0
        elif dtype == np.uint16:
            pixel_type = 1
        elif dtype == np.int16:
            pixel_type = 2
        else:
            lib.nvjpeg2kEncodeParamsDestroy(enc_params)
            lib.nvjpeg2kEncodeStateDestroy(enc_state)
            lib.nvjpeg2kEncoderDestroy(encoder)
            return None

        pitch = tile_width * dtype.itemsize
        result = []

        for i in range(n_tiles):
            tile_data = d_tile_bufs[i * tile_bytes:(i + 1) * tile_bytes]

            # De-interleave into per-component planes for the encoder
            if samples == 1:
                comp_bufs = [tile_data]
            else:
                tile_arr = tile_data.view(dtype=cupy.dtype(dtype)).reshape(
                    tile_height, tile_width, samples)
                comp_bufs = [
                    cupy.ascontiguousarray(tile_arr[:, :, c]).view(cupy.uint8).ravel()
                    for c in range(samples)
                ]

            img = _NvJpeg2kImage()
            img.num_components = samples
            img.pixel_type = pixel_type
            for c in range(samples):
                img.pixel_data[c] = comp_bufs[c].data.ptr
                img.pitch_in_bytes[c] = pitch

            # Set image info on params
            class _CompInfo(ctypes.Structure):
                _fields_ = [
                    ('component_width', ctypes.c_uint32),
                    ('component_height', ctypes.c_uint32),
                    ('precision', ctypes.c_uint8),
                    ('sgn', ctypes.c_uint8),
                ]

            precision = dtype.itemsize * 8
            sgn = 1 if dtype.kind == 'i' else 0

            comp_info = (_CompInfo * samples)()
            for c in range(samples):
                comp_info[c].component_width = tile_width
                comp_info[c].component_height = tile_height
                comp_info[c].precision = precision
                comp_info[c].sgn = sgn

            # Encode
            s = lib.nvjpeg2kEncode(
                encoder, enc_state, enc_params,
                ctypes.byref(img),
                ctypes.c_void_p(0),  # default CUDA stream
            )
            cupy.cuda.Device().synchronize()
            if s != 0:
                lib.nvjpeg2kEncodeParamsDestroy(enc_params)
                lib.nvjpeg2kEncodeStateDestroy(enc_state)
                lib.nvjpeg2kEncoderDestroy(encoder)
                return None

            # Retrieve bitstream size
            bs_size = ctypes.c_size_t(0)
            lib.nvjpeg2kEncoderRetrieveBitstream(
                encoder, enc_state,
                ctypes.c_void_p(0),
                ctypes.byref(bs_size),
                ctypes.c_void_p(0),
            )

            # Retrieve bitstream data
            bs_buf = np.empty(bs_size.value, dtype=np.uint8)
            lib.nvjpeg2kEncoderRetrieveBitstream(
                encoder, enc_state,
                ctypes.c_void_p(bs_buf.ctypes.data),
                ctypes.byref(bs_size),
                ctypes.c_void_p(0),
            )

            result.append(bs_buf[:bs_size.value].tobytes())

        lib.nvjpeg2kEncodeParamsDestroy(enc_params)
        lib.nvjpeg2kEncodeStateDestroy(enc_state)
        lib.nvjpeg2kEncoderDestroy(encoder)

        return result

    except Exception:
        return None


# ---------------------------------------------------------------------------
# High-level GPU write pipeline
# ---------------------------------------------------------------------------

def gpu_compress_tiles(d_image, tile_width, tile_height,
                       image_width, image_height,
                       compression, predictor, dtype,
                       samples=1):
    """Extract and compress tiles from a CuPy image on GPU.

    Parameters
    ----------
    d_image : cupy.ndarray
        2D or 3D image on GPU device.
    tile_width, tile_height : int
        Tile dimensions.
    image_width, image_height : int
        Image dimensions.
    compression : int
        TIFF compression tag.
    predictor : int
        Predictor tag (1=none, 2=horizontal, 3=float).
    dtype : np.dtype
        Pixel dtype.
    samples : int
        Samples per pixel.

    Returns
    -------
    list of bytes
        Compressed tile data on CPU, ready for _assemble_tiff.
    """
    import cupy

    bytes_per_pixel = dtype.itemsize * samples
    tile_bytes = tile_width * tile_height * bytes_per_pixel
    tiles_across = math.ceil(image_width / tile_width)
    tiles_down = math.ceil(image_height / tile_height)
    n_tiles = tiles_across * tiles_down

    # Flatten image to uint8
    d_flat = d_image.view(cupy.uint8).ravel()

    # Allocate tile buffer
    d_tile_buf = cupy.zeros(n_tiles * tile_bytes, dtype=cupy.uint8)
    tile_offsets = np.arange(n_tiles, dtype=np.int64) * tile_bytes
    d_tile_offsets = cupy.asarray(tile_offsets)

    # Extract tiles on GPU
    total_pixels = image_width * image_height
    tpb = 256
    bpg = math.ceil(total_pixels / tpb)
    _extract_tiles_kernel[bpg, tpb](
        d_flat, d_tile_buf, d_tile_offsets,
        tile_width, tile_height, bytes_per_pixel,
        image_width, image_height, tiles_across)
    cuda.synchronize()

    # Apply predictor encode on GPU
    total_rows = n_tiles * tile_height
    if predictor == 2:
        # Sample-level differencing: stride is samples_per_pixel samples,
        # row width is tile_width pixels.
        _gpu_predictor2_encode(
            d_tile_buf, tile_width, total_rows, dtype, samples)
    elif predictor == 3:
        tpb_r = min(256, total_rows)
        bpg_r = math.ceil(total_rows / tpb_r)
        d_tmp = cupy.empty_like(d_tile_buf)
        _fp_predictor_encode_kernel[bpg_r, tpb_r](
            d_tile_buf, d_tmp, tile_width * samples, total_rows, dtype.itemsize)
        cuda.synchronize()

    # Split into per-tile buffers
    d_tiles = [d_tile_buf[i * tile_bytes:(i + 1) * tile_bytes] for i in range(n_tiles)]

    # JPEG: try nvJPEG encode, fall back to Pillow
    if compression == 7:
        result = _nvjpeg_batch_encode(d_tiles, tile_width, tile_height, samples)
        if result is not None:
            return result
        # Fallback: CPU Pillow encode
        from ._compression import jpeg_compress
        cpu_buf = d_tile_buf.get()
        result = []
        for i in range(n_tiles):
            start = i * tile_bytes
            tile_data = bytes(cpu_buf[start:start + tile_bytes])
            result.append(jpeg_compress(tile_data, tile_width, tile_height,
                                        samples))
        return result

    # JPEG 2000: use nvJPEG2000 (image codec, not byte-stream codec)
    if compression == 34712:
        result = _nvjpeg2k_batch_encode(
            d_tile_buf, tile_width, tile_height, dtype, samples, n_tiles)
        if result is not None:
            return result
        # CPU fallback for JPEG 2000
        from ._compression import jpeg2000_compress
        cpu_buf = d_tile_buf.get()
        result = []
        for i in range(n_tiles):
            start = i * tile_bytes
            tile_data = bytes(cpu_buf[start:start + tile_bytes])
            result.append(jpeg2000_compress(
                tile_data, tile_width, tile_height,
                samples=samples, dtype=dtype))
        return result

    # LERC: CPU only, no GPU library
    if compression == 34887:
        from ._compression import lerc_compress
        cpu_buf = d_tile_buf.get()
        result = []
        for i in range(n_tiles):
            start = i * tile_bytes
            tile_data = bytes(cpu_buf[start:start + tile_bytes])
            result.append(lerc_compress(
                tile_data, tile_width, tile_height,
                samples=samples, dtype=dtype))
        return result

    # Try nvCOMP batch compress
    result = _nvcomp_batch_compress(d_tiles, None, tile_bytes, compression, n_tiles)

    if result is not None:
        return result

    # Fallback: copy to CPU, compress with CPU codecs
    from ._compression import compress as cpu_compress
    cpu_buf = d_tile_buf.get()
    result = []
    for i in range(n_tiles):
        start = i * tile_bytes
        tile_data = bytes(cpu_buf[start:start + tile_bytes])
        result.append(cpu_compress(tile_data, compression))

    return result


# ---------------------------------------------------------------------------
# GPU overview (pyramid) generation
# ---------------------------------------------------------------------------

GPU_OVERVIEW_METHODS = ('mean', 'nearest', 'min', 'max', 'median', 'mode',
                        'cubic')


def _block_reduce_2d_gpu(arr2d, method, nodata=None):
    """2x block-reduce a single 2D CuPy plane using *method*.

    When ``nodata`` is supplied and ``arr2d`` is a float dtype, cells that
    equal the sentinel are masked back to NaN before the reduction so the
    ``cupy.nan*`` aggregation routines correctly skip them. Mirrors the
    CPU helper :func:`xrspatial.geotiff._writer._block_reduce_2d` so the
    two backends produce identical overviews when ``nodata`` is set.
    """
    import cupy
    import numpy as np

    h, w = arr2d.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    cropped = arr2d[:h2, :w2]
    oh, ow = h2 // 2, w2 // 2

    if method == 'nearest':
        return cropped[::2, ::2].copy()

    if method == 'mode':
        # Mode is expensive on GPU; fall back to CPU
        cpu_arr = arr2d.get()
        from ._writer import _block_reduce_2d
        cpu_result = _block_reduce_2d(cpu_arr, 'mode', nodata=nodata)
        return cupy.asarray(cpu_result)

    if method == 'cubic':
        # No native cupy cubic resampler that handles arbitrary zoom
        # factors with the same prefilter=False NaN-safety the CPU
        # helper uses for issue #1623. Fall back to CPU so cubic on
        # the GPU writer path produces the same overview bytes as the
        # CPU writer and so the sentinel handling matches.
        cpu_arr = arr2d.get()
        from ._writer import _block_reduce_2d
        cpu_result = _block_reduce_2d(cpu_arr, 'cubic', nodata=nodata)
        return cupy.asarray(cpu_result)

    # Block reshape for mean/min/max/median
    if arr2d.dtype.kind == 'f':
        blocks = cropped.reshape(oh, 2, ow, 2)
        # Mask the sentinel back to NaN so cupy.nanmean and friends
        # honour it as missing-data (issue #1613). Match the upstream
        # NaN->sentinel rewrite gate so ``nodata=+/-inf`` is masked here.
        if nodata is not None and not np.isnan(nodata):
            try:
                sentinel = np.dtype(str(arr2d.dtype)).type(nodata)
            except (OverflowError, ValueError):
                sentinel = None
            if sentinel is not None:
                mask = blocks == sentinel
                if bool(mask.any().item()):
                    blocks = cupy.where(
                        mask, cupy.float64('nan'), blocks)
    else:
        blocks = cropped.astype(cupy.float64).reshape(oh, 2, ow, 2)

    if method == 'mean':
        result = cupy.nanmean(blocks, axis=(1, 3))
    elif method == 'min':
        result = cupy.nanmin(blocks, axis=(1, 3))
    elif method == 'max':
        result = cupy.nanmax(blocks, axis=(1, 3))
    elif method == 'median':
        flat = blocks.transpose(0, 2, 1, 3).reshape(oh, ow, 4)
        result = cupy.nanmedian(flat, axis=2)
    else:
        raise ValueError(
            f"Unknown GPU overview resampling method: {method!r}. "
            f"Use one of: {GPU_OVERVIEW_METHODS}")

    if arr2d.dtype.kind != 'f':
        return cupy.around(result).astype(arr2d.dtype)
    return result.astype(arr2d.dtype)


def make_overview_gpu(arr, method='mean', nodata=None):
    """Generate a 2x decimated overview on GPU.

    Parameters
    ----------
    arr : cupy.ndarray
        2D or 3D (height, width, bands) array on GPU.
    method : str
        Resampling method: 'mean', 'nearest', 'min', 'max', 'median',
        'mode', or 'cubic'. ``mode`` and ``cubic`` fall back to the CPU
        implementation in :mod:`xrspatial.geotiff._writer` so the GPU
        writer path produces the same overview bytes as the CPU writer.
    nodata : scalar or None
        When supplied and ``arr`` is a float dtype, cells equal to the
        sentinel are masked back to NaN before the reduction so the
        sentinel does not bias the result. Required for COG output that
        sets ``nodata=...`` (issue #1613, extended to ``cubic`` in
        issue #1623). Ignored for integer arrays and for ``nearest``.

    Returns
    -------
    cupy.ndarray
        Half-resolution array on GPU.
    """
    import cupy

    if arr.ndim == 3:
        bands = [_block_reduce_2d_gpu(arr[:, :, b], method, nodata=nodata)
                 for b in range(arr.shape[2])]
        return cupy.stack(bands, axis=2)
    return _block_reduce_2d_gpu(arr, method, nodata=nodata)
