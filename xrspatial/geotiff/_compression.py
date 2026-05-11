"""Compression codecs: deflate (zlib) and LZW (Numba), plus horizontal predictor."""
from __future__ import annotations

import zlib

import numpy as np

from xrspatial.utils import ngjit

# -- Decompression-bomb defenses ---------------------------------------------
#
# A malicious TIFF can declare a small strip/tile compressed payload that
# expands to multiple gigabytes when decoded.  Without a cap the reader is
# OOM-killed before the post-decode size check (in ``_decode_strip_or_tile``)
# ever runs, since by then the bomb has already been allocated.  Each codec
# below takes an ``expected_size`` (the byte length the caller computed from
# the IFD's declared dimensions) and refuses to produce more than
# ``expected_size * _DECOMPRESS_MARGIN`` bytes.  The margin allows for the
# small amount of legitimate codec metadata that some encoders emit, while
# still rejecting the 1000:1 ratios characteristic of bomb attacks.
_DECOMPRESS_MARGIN = 1.05


def _max_output_with_margin(expected_size: int) -> int:
    """Return the cap (in bytes) for a codec given the caller's expected size.

    A positive ``expected_size`` returns ``int(expected_size * 1.05) + 1``,
    leaving a one-byte sentinel above the legitimate margin so a single
    byte of overflow is detectable.

    A non-positive ``expected_size`` (the default ``0``) returns ``0``,
    which the codec wrappers interpret as "no cap": they fall through to
    the unbounded library decode for backward compatibility with callers
    that don't supply a size. The reader always supplies a size; this
    branch is only hit by direct callers and round-trip tests.
    """
    if expected_size <= 0:
        return 0
    return int(expected_size * _DECOMPRESS_MARGIN) + 1


# -- Deflate (zlib wrapper) --------------------------------------------------


def deflate_decompress(data: bytes, expected_size: int = 0) -> bytes:
    """Decompress deflate/zlib data with an optional output-size cap.

    Parameters
    ----------
    data : bytes
        Deflate/zlib compressed payload.
    expected_size : int, optional
        Caller's expected uncompressed byte count.  When > 0, the decoder
        refuses to produce more than ``expected_size * 1.05 + 1`` bytes
        and raises ``ValueError`` on overflow (decompression-bomb guard).

    Returns
    -------
    bytes
        Uncompressed data.
    """
    if expected_size <= 0:
        # Backward-compat path: caller hasn't supplied an expected size.
        return zlib.decompress(data)
    cap = _max_output_with_margin(expected_size)
    decompressor = zlib.decompressobj()
    # Accumulate into a bytearray rather than reassigning bytes via ``+=``;
    # the bytes path was O(n^2) for large strips because every chunk
    # reallocated and copied the whole accumulated buffer.
    out = bytearray()
    # Read one byte beyond the cap so that an overflowing stream is detected
    # rather than silently truncated.
    out += decompressor.decompress(data, max_length=cap + 1)
    # Drain any output the decompressor has buffered (including unconsumed
    # input).  We stop as soon as output exceeds the cap, or when the
    # decompressor declares EOF, or when no further bytes are produced.
    while not decompressor.eof and len(out) <= cap:
        feed = decompressor.unconsumed_tail
        more = decompressor.decompress(feed, max_length=cap + 1 - len(out))
        if not more:
            break
        out += more
    if len(out) > cap:
        raise ValueError(
            f"deflate decode exceeded expected size: {len(out)} bytes "
            f"produced, cap is {cap} (expected {expected_size}).  Likely "
            f"a decompression bomb."
        )
    # Flush any remaining state to surface tail bytes / errors.
    out += decompressor.flush()
    if len(out) > cap:
        raise ValueError(
            f"deflate decode exceeded expected size: {len(out)} bytes "
            f"produced, cap is {cap} (expected {expected_size}).  Likely "
            f"a decompression bomb."
        )
    return bytes(out)


def deflate_compress(data: bytes, level: int = 6) -> bytes:
    """Compress data with deflate/zlib."""
    return zlib.compress(data, level)


# -- LZW constants -----------------------------------------------------------

LZW_CLEAR_CODE = 256
LZW_EOI_CODE = 257
LZW_FIRST_CODE = 258
LZW_MAX_CODE = 4095
LZW_MAX_BITS = 12


# -- LZW decode (Numba) ------------------------------------------------------

@ngjit
def _lzw_decode_kernel(src, src_len, dst, dst_len):
    """Decode TIFF-variant LZW (MSB-first) into dst buffer.

    Parameters
    ----------
    src : uint8 array
        Compressed bytes.
    src_len : int
        Number of valid bytes in src.
    dst : uint8 array
        Output buffer (must be pre-allocated large enough).
    dst_len : int
        Maximum bytes to write.

    Returns
    -------
    int
        Number of bytes written to dst.
    """
    # Table: prefix-chain representation
    table_prefix = np.full(4096, -1, dtype=np.int32)
    table_suffix = np.zeros(4096, dtype=np.uint8)
    table_length = np.zeros(4096, dtype=np.int32)

    # Small stack for chain reversal
    stack = np.empty(4096, dtype=np.uint8)

    # Bit reader state
    bit_pos = 0
    code_size = 9
    next_code = LZW_FIRST_CODE

    # Initialize table with single-byte entries
    for i in range(256):
        table_prefix[i] = -1
        table_suffix[i] = np.uint8(i)
        table_length[i] = 1

    out_pos = 0
    old_code = -1

    while True:
        # Read next code (MSB-first bit packing)
        byte_offset = bit_pos >> 3
        if byte_offset >= src_len:
            break

        # Gather up to 24 bits from available bytes
        bits = np.int32(src[byte_offset]) << 16
        if byte_offset + 1 < src_len:
            bits |= np.int32(src[byte_offset + 1]) << 8
        if byte_offset + 2 < src_len:
            bits |= np.int32(src[byte_offset + 2])

        bit_offset_in_byte = bit_pos & 7
        # Shift to align the code_size bits at the LSB side
        bits = (bits >> (24 - bit_offset_in_byte - code_size)) & ((1 << code_size) - 1)
        bit_pos += code_size
        code = bits

        if code == LZW_EOI_CODE:
            break

        if code == LZW_CLEAR_CODE:
            code_size = 9
            next_code = LZW_FIRST_CODE
            old_code = -1
            continue

        if old_code == -1:
            # First code after clear
            if code < 256:
                if out_pos < dst_len:
                    dst[out_pos] = np.uint8(code)
                    out_pos += 1
            old_code = code
            continue

        # Determine the string for this code
        if code < next_code:
            # Code is in table -- walk the chain, push to stack, emit reversed
            c = code
            stack_pos = 0
            while c >= 0 and c < 4096 and stack_pos < 4096:
                stack[stack_pos] = table_suffix[c]
                stack_pos += 1
                c = table_prefix[c]

            # Emit in correct order
            for i in range(stack_pos - 1, -1, -1):
                if out_pos < dst_len:
                    dst[out_pos] = stack[i]
                    out_pos += 1

            # Add new entry: old_code string + first char of code string
            if next_code <= LZW_MAX_CODE and stack_pos > 0:
                table_prefix[next_code] = old_code
                table_suffix[next_code] = stack[stack_pos - 1]  # first char
                table_length[next_code] = table_length[old_code] + 1
                next_code += 1
        else:
            # Special case: code == next_code
            # String = old_code string + first char of old_code string
            c = old_code
            stack_pos = 0
            while c >= 0 and c < 4096 and stack_pos < 4096:
                stack[stack_pos] = table_suffix[c]
                stack_pos += 1
                c = table_prefix[c]

            if stack_pos == 0:
                old_code = code
                continue

            first_char = stack[stack_pos - 1]

            # Emit old_code string
            for i in range(stack_pos - 1, -1, -1):
                if out_pos < dst_len:
                    dst[out_pos] = stack[i]
                    out_pos += 1
            # Emit first char again
            if out_pos < dst_len:
                dst[out_pos] = first_char
                out_pos += 1

            # Add new entry
            if next_code <= LZW_MAX_CODE:
                table_prefix[next_code] = old_code
                table_suffix[next_code] = first_char
                table_length[next_code] = table_length[old_code] + 1
                next_code += 1

        # Bump code size (TIFF LZW uses "early change": bump one code before
        # the table fills the current code_size capacity)
        if next_code > (1 << code_size) - 2 and code_size < LZW_MAX_BITS:
            code_size += 1

        old_code = code

    return out_pos


def lzw_decompress(data: bytes, expected_size: int = 0) -> np.ndarray:
    """Decompress TIFF-variant LZW data.

    Parameters
    ----------
    data : bytes
        LZW compressed data.
    expected_size : int
        Expected decompressed size. If 0, uses 10x compressed size as buffer.

    Returns
    -------
    np.ndarray
        Mutable uint8 array of decompressed data.
    """
    src = np.frombuffer(data, dtype=np.uint8)
    if expected_size <= 0:
        expected_size = len(data) * 10
    dst = np.empty(expected_size, dtype=np.uint8)
    n = _lzw_decode_kernel(src, len(src), dst, expected_size)
    return dst[:n].copy()  # owned, mutable slice


# -- LZW encode (Numba) ------------------------------------------------------

@ngjit
def _lzw_encode_kernel(src, src_len, dst, dst_len):
    """Encode data as TIFF-variant LZW (MSB-first).

    Returns number of bytes written to dst.
    """
    # Hash table for string matching
    # Key: (prefix_code << 8) | suffix_byte -> code
    # Uses generation counter to avoid clearing: an entry is valid only when
    # ht_gen[slot] == current_gen.
    HT_SIZE = 8209  # prime > 4096*2
    ht_keys = np.empty(HT_SIZE, dtype=np.int64)
    ht_values = np.empty(HT_SIZE, dtype=np.int32)
    ht_gen = np.zeros(HT_SIZE, dtype=np.int32)
    current_gen = np.int32(1)

    # Bit accumulator: collect bits and flush whole bytes
    bit_buf = np.int32(0)   # up to 24 bits pending
    bits_in_buf = np.int32(0)
    out_pos = 0

    code_size = 9
    next_code = LZW_FIRST_CODE

    def flush_code(code, code_size, bit_buf, bits_in_buf, dst, dst_len, out_pos):
        """Pack a code into the bit accumulator and flush complete bytes."""
        # Merge code bits (MSB-first) into accumulator
        bit_buf = (bit_buf << code_size) | code
        bits_in_buf += code_size
        # Flush whole bytes from the top of the accumulator
        while bits_in_buf >= 8:
            bits_in_buf -= 8
            if out_pos < dst_len:
                dst[out_pos] = np.uint8((bit_buf >> bits_in_buf) & 0xFF)
                out_pos += 1
        return bit_buf, bits_in_buf, out_pos

    # Write initial clear code
    bit_buf, bits_in_buf, out_pos = flush_code(
        LZW_CLEAR_CODE, code_size, bit_buf, bits_in_buf, dst, dst_len, out_pos)

    if src_len == 0:
        bit_buf, bits_in_buf, out_pos = flush_code(
            LZW_EOI_CODE, code_size, bit_buf, bits_in_buf, dst, dst_len, out_pos)
        # Flush remaining bits
        if bits_in_buf > 0 and out_pos < dst_len:
            dst[out_pos] = np.uint8((bit_buf << (8 - bits_in_buf)) & 0xFF)
            out_pos += 1
        return out_pos

    prefix = np.int32(src[0])
    pos = 1

    while pos < src_len:
        suffix = np.int32(src[pos])
        # Look up (prefix, suffix) in hash table
        key = np.int64(prefix) * 256 + np.int64(suffix)
        h = int(key % HT_SIZE)
        if h < 0:
            h += HT_SIZE

        found = False
        for _ in range(HT_SIZE):
            if ht_gen[h] == current_gen and ht_keys[h] == key:
                prefix = ht_values[h]
                found = True
                break
            elif ht_gen[h] != current_gen:
                break
            h = (h + 1) % HT_SIZE

        if not found:
            # Output the prefix code
            bit_buf, bits_in_buf, out_pos = flush_code(
                prefix, code_size, bit_buf, bits_in_buf, dst, dst_len, out_pos)

            # Add new entry to table
            if next_code <= LZW_MAX_CODE:
                ht_gen[h] = current_gen
                ht_keys[h] = key
                ht_values[h] = next_code
                next_code += 1

                # Encoder bumps one entry later than decoder (decoder trails by 1)
                if next_code > (1 << code_size) - 1 and code_size < LZW_MAX_BITS:
                    code_size += 1

            else:
                # Table full, emit clear code and reset
                bit_buf, bits_in_buf, out_pos = flush_code(
                    LZW_CLEAR_CODE, code_size, bit_buf, bits_in_buf, dst, dst_len, out_pos)
                code_size = 9
                next_code = LZW_FIRST_CODE
                current_gen += 1

            prefix = suffix
        pos += 1

    # Output last prefix
    bit_buf, bits_in_buf, out_pos = flush_code(
        prefix, code_size, bit_buf, bits_in_buf, dst, dst_len, out_pos)
    bit_buf, bits_in_buf, out_pos = flush_code(
        LZW_EOI_CODE, code_size, bit_buf, bits_in_buf, dst, dst_len, out_pos)

    # Flush remaining bits
    if bits_in_buf > 0 and out_pos < dst_len:
        dst[out_pos] = np.uint8((bit_buf << (8 - bits_in_buf)) & 0xFF)
        out_pos += 1

    return out_pos


def lzw_compress(data: bytes) -> bytes:
    """Compress data using TIFF-variant LZW.

    Parameters
    ----------
    data : bytes
        Raw data to compress.

    Returns
    -------
    bytes
    """
    src = np.frombuffer(data, dtype=np.uint8)
    # Worst case: output slightly larger than input
    max_out = len(data) + len(data) // 2 + 256
    dst = np.empty(max_out, dtype=np.uint8)
    n = _lzw_encode_kernel(src, len(src), dst, max_out)
    return dst[:n].tobytes()


# -- Horizontal predictor (Numba) --------------------------------------------
#
# TIFF predictor=2 (horizontal differencing) operates at the *sample* level,
# not the byte level.  Per TIFF Technical Note: "the difference is taken
# between adjacent samples (not between adjacent bytes) of the same
# component."  This means for multi-byte samples the encoder computes
#
#     diff[i] = sample[i] - sample[i - samples_per_pixel]   (mod 2^bits)
#
# in the sample's natural bit width (uint16 wraps at 65536, etc.), not
# byte-by-byte.  A byte-wise implementation drops the inter-byte carry
# and produces wrong values for any sample wider than one byte.  This
# is what libtiff's horAcc16/horAcc32 do, and what GDAL/rasterio/tifffile
# write to disk.  The byte-wise path remains correct (and faster) for the
# 8-bit case.


@ngjit
def _predictor_decode_u8(data, width, height, samples_per_pixel):
    """Undo predictor=2 for 8-bit samples (one byte per sample).

    Stride is samples_per_pixel bytes.  The byte-wise modular sum is
    correct because each sample fits in a single byte.
    """
    row_bytes = width * samples_per_pixel
    for row in range(height):
        row_start = row * row_bytes
        for col in range(samples_per_pixel, row_bytes):
            idx = row_start + col
            data[idx] = np.uint8(
                (np.int32(data[idx]) + np.int32(data[idx - samples_per_pixel])) & 0xFF)


@ngjit
def _predictor_decode_u16(view, width, height, samples_per_pixel):
    """Undo predictor=2 on a uint16 view (in-place, sample-resolution).

    *view* indexes by sample.  Stride is *samples_per_pixel* samples.
    """
    row_samples = width * samples_per_pixel
    for row in range(height):
        row_start = row * row_samples
        for col in range(samples_per_pixel, row_samples):
            idx = row_start + col
            view[idx] = np.uint16(
                (np.int32(view[idx]) + np.int32(view[idx - samples_per_pixel])) & 0xFFFF)


@ngjit
def _predictor_decode_u32(view, width, height, samples_per_pixel):
    """Undo predictor=2 on a uint32 view (in-place, sample-resolution)."""
    row_samples = width * samples_per_pixel
    for row in range(height):
        row_start = row * row_samples
        for col in range(samples_per_pixel, row_samples):
            idx = row_start + col
            view[idx] = np.uint32(
                (np.uint64(view[idx]) + np.uint64(view[idx - samples_per_pixel]))
                & np.uint64(0xFFFFFFFF))


@ngjit
def _predictor_decode_u64(view, width, height, samples_per_pixel):
    """Undo predictor=2 on a uint64 view (in-place, sample-resolution)."""
    row_samples = width * samples_per_pixel
    for row in range(height):
        row_start = row * row_samples
        for col in range(samples_per_pixel, row_samples):
            idx = row_start + col
            view[idx] = np.uint64(view[idx]) + np.uint64(view[idx - samples_per_pixel])


@ngjit
def _predictor_encode_u8(data, width, height, samples_per_pixel):
    """Apply predictor=2 for 8-bit samples (right-to-left, in-place)."""
    row_bytes = width * samples_per_pixel
    for row in range(height):
        row_start = row * row_bytes
        for col in range(row_bytes - 1, samples_per_pixel - 1, -1):
            idx = row_start + col
            data[idx] = np.uint8(
                (np.int32(data[idx]) - np.int32(data[idx - samples_per_pixel])) & 0xFF)


@ngjit
def _predictor_encode_u16(view, width, height, samples_per_pixel):
    """Apply predictor=2 on a uint16 view (right-to-left, in-place)."""
    row_samples = width * samples_per_pixel
    for row in range(height):
        row_start = row * row_samples
        for col in range(row_samples - 1, samples_per_pixel - 1, -1):
            idx = row_start + col
            view[idx] = np.uint16(
                (np.int32(view[idx]) - np.int32(view[idx - samples_per_pixel])) & 0xFFFF)


@ngjit
def _predictor_encode_u32(view, width, height, samples_per_pixel):
    """Apply predictor=2 on a uint32 view (right-to-left, in-place)."""
    row_samples = width * samples_per_pixel
    for row in range(height):
        row_start = row * row_samples
        for col in range(row_samples - 1, samples_per_pixel - 1, -1):
            idx = row_start + col
            view[idx] = np.uint32(
                (np.uint64(view[idx]) - np.uint64(view[idx - samples_per_pixel]))
                & np.uint64(0xFFFFFFFF))


@ngjit
def _predictor_encode_u64(view, width, height, samples_per_pixel):
    """Apply predictor=2 on a uint64 view (right-to-left, in-place)."""
    row_samples = width * samples_per_pixel
    for row in range(height):
        row_start = row * row_samples
        for col in range(row_samples - 1, samples_per_pixel - 1, -1):
            idx = row_start + col
            view[idx] = np.uint64(view[idx]) - np.uint64(view[idx - samples_per_pixel])


import sys as _sys
_NATIVE_BO = '<' if _sys.byteorder == 'little' else '>'


def _sample_view(buf: np.ndarray, bytes_per_sample: int) -> np.ndarray:
    """Return *buf* viewed as a 1-D array of native-byte-order samples.

    The view shares memory with *buf*. Numba's nopython mode rejects arrays
    with a non-native byte order ("Unsupported array dtype: >u2" etc.), so
    callers that need to operate on data from a big-endian file must
    byteswap the underlying bytes before and after invoking the kernels.
    """
    if bytes_per_sample == 1:
        return buf
    sample_dtype = np.dtype(f'u{bytes_per_sample}')
    return buf.view(sample_dtype)


def predictor_decode(data: np.ndarray, width: int, height: int,
                     bytes_per_sample: int, samples: int = 1,
                     byte_order: str = '<') -> np.ndarray:
    """Undo horizontal differencing predictor (predictor=2).

    Operates at the sample level, per TIFF Technical Note: differences
    are taken between adjacent same-component samples in the sample's
    natural bit width.  For multi-band chunky data the stride is the
    number of samples per pixel.

    Parameters
    ----------
    data : np.ndarray
        Flat uint8 array of decompressed pixel data (modified in-place).
    width, height : int
        Image dimensions in pixels.
    bytes_per_sample : int
        Bytes per sample (1, 2, 4, or 8).
    samples : int
        Samples per pixel (1 for single-band, 3 for RGB, ...).
    byte_order : str
        '<' for little-endian, '>' for big-endian.  Used to interpret
        multi-byte sample bytes during the in-place differencing.

    Returns
    -------
    np.ndarray
        The same uint8 array with predictor=2 inverted.
    """
    buf = np.ascontiguousarray(data)
    if bytes_per_sample == 1:
        _predictor_decode_u8(buf, width, height, samples)
        return buf

    # Numba can't compile non-native-byte-order arrays. For big-endian
    # files, swap the bytes in place so the native-order kernel sees
    # correct sample values, then swap back so the caller's downstream
    # ``chunk.view(file_dtype)`` reads the bytes in the file's order.
    swap = (byte_order != _NATIVE_BO)
    view = _sample_view(buf, bytes_per_sample)
    if swap:
        view.byteswap(inplace=True)
    if bytes_per_sample == 2:
        _predictor_decode_u16(view, width, height, samples)
    elif bytes_per_sample == 4:
        _predictor_decode_u32(view, width, height, samples)
    elif bytes_per_sample == 8:
        _predictor_decode_u64(view, width, height, samples)
    else:
        if swap:
            view.byteswap(inplace=True)
        raise ValueError(
            f"predictor=2 with bytes_per_sample={bytes_per_sample} is not "
            "supported (TIFF specifies sample-level differencing for 1/2/4/8 "
            "byte samples)."
        )
    if swap:
        view.byteswap(inplace=True)
    return buf


def predictor_encode(data: np.ndarray, width: int, height: int,
                     bytes_per_sample: int, samples: int = 1,
                     byte_order: str = '<') -> np.ndarray:
    """Apply horizontal differencing predictor (predictor=2).

    See :func:`predictor_decode` for the sample-level semantics.

    Parameters
    ----------
    data : np.ndarray
        Flat uint8 array of pixel data (modified in-place).
    width, height : int
        Image dimensions in pixels.
    bytes_per_sample : int
        Bytes per sample.
    samples : int
        Samples per pixel.
    byte_order : str
        '<' or '>'.  Defaults to little-endian, the writer's output order.

    Returns
    -------
    np.ndarray
        The same uint8 array with predictor=2 applied.
    """
    buf = np.ascontiguousarray(data)
    if bytes_per_sample == 1:
        _predictor_encode_u8(buf, width, height, samples)
        return buf

    # Mirror the decode path: byteswap to native around the kernel call so
    # numba can compile the in-place differencing. The writer always emits
    # ``BO='<'`` today so this is normally a no-op, but the function stays
    # robust to callers that pass the BE branch.
    swap = (byte_order != _NATIVE_BO)
    view = _sample_view(buf, bytes_per_sample)
    if swap:
        view.byteswap(inplace=True)
    if bytes_per_sample == 2:
        _predictor_encode_u16(view, width, height, samples)
    elif bytes_per_sample == 4:
        _predictor_encode_u32(view, width, height, samples)
    elif bytes_per_sample == 8:
        _predictor_encode_u64(view, width, height, samples)
    else:
        if swap:
            view.byteswap(inplace=True)
        raise ValueError(
            f"predictor=2 with bytes_per_sample={bytes_per_sample} is not "
            "supported (TIFF specifies sample-level differencing for 1/2/4/8 "
            "byte samples)."
        )
    if swap:
        view.byteswap(inplace=True)
    return buf


# -- Floating-point predictor (predictor=3) -----------------------------------
#
# TIFF predictor=3 (floating-point horizontal differencing):
#   During encoding, bytes of each sample are rearranged into byte-lane order
#   (MSB lane first, LSB lane last), then horizontal differencing is applied
#   across the entire transposed row.
#
#   For little-endian float32 with N samples:
#     Swizzled layout: [MSB_s0..MSB_sN-1, byte2_s0..byte2_sN-1,
#                       byte1_s0..byte1_sN-1, LSB_s0..LSB_sN-1]
#     i.e. lane 0 = native byte (bps-1), lane 1 = native byte (bps-2), etc.
#
# Decode: undo differencing, then un-transpose (lane b -> native byte bps-1-b).

@ngjit
def _fp_predictor_decode_row(row_data, width, bps, big_endian):
    """Undo floating-point predictor for one row (in-place).

    row_data: uint8 array of length width * bps.

    Per TIFF Tech Note 3, lane 0 contains the most significant byte of
    each sample regardless of the file's byte order.  After
    un-transposing, the MSB has to land at the byte position that
    corresponds to "most significant" *in the file's byte order*: index 0
    for big-endian files and index ``bps-1`` for little-endian files.
    The previous implementation always wrote MSB at index ``bps-1``,
    which scrambled big-endian predictor=3 reads.
    """
    n = width * bps

    # Step 1: undo horizontal differencing on the byte-swizzled row
    for i in range(1, n):
        row_data[i] = np.uint8((np.int32(row_data[i]) + np.int32(row_data[i - 1])) & 0xFF)

    # Step 2: un-transpose bytes back to the file's native sample order
    tmp = np.empty(n, dtype=np.uint8)
    if big_endian:
        # MSB lane (lane 0) -> byte index 0 in each output sample.
        for sample in range(width):
            for b in range(bps):
                tmp[sample * bps + b] = row_data[b * width + sample]
    else:
        # MSB lane -> byte index ``bps-1`` (the high-order byte in LE).
        for sample in range(width):
            for b in range(bps):
                tmp[sample * bps + b] = row_data[(bps - 1 - b) * width + sample]
    for i in range(n):
        row_data[i] = tmp[i]


@ngjit
def _fp_predictor_decode_rows(data, width, height, bps, big_endian):
    """Dispatch per-row decode from Numba, avoiding Python loop overhead."""
    row_len = width * bps
    for row in range(height):
        start = row * row_len
        _fp_predictor_decode_row(
            data[start:start + row_len], width, bps, big_endian)


def fp_predictor_decode(data: np.ndarray, width: int, height: int,
                        bytes_per_sample: int,
                        big_endian: bool = False) -> np.ndarray:
    """Undo floating-point predictor (predictor=3).

    Parameters
    ----------
    data : np.ndarray
        Flat uint8 array of decompressed tile/strip data.
    width, height : int
        Tile/strip dimensions.
    bytes_per_sample : int
        Bytes per sample (e.g. 4 for float32, 8 for float64).
    big_endian : bool
        Whether the source file is big-endian (BOM ``MM``).  The
        un-transpose puts the MSB lane at byte index 0 of each output
        sample for BE files, and at byte index ``bps-1`` for LE files.

    Returns
    -------
    np.ndarray
        Corrected array.
    """
    buf = np.ascontiguousarray(data)
    _fp_predictor_decode_rows(buf, width, height, bytes_per_sample,
                              big_endian)
    return buf


@ngjit
def _fp_predictor_encode_row(row_data, width, bps):
    """Apply floating-point predictor for one row (in-place)."""
    n = width * bps

    # Step 1: transpose to byte-swizzled layout (MSB lane first)
    # Native byte b of each sample goes to lane (bps-1-b).
    tmp = np.empty(n, dtype=np.uint8)
    for sample in range(width):
        for b in range(bps):
            tmp[(bps - 1 - b) * width + sample] = row_data[sample * bps + b]
    for i in range(n):
        row_data[i] = tmp[i]

    # Step 2: horizontal differencing on the swizzled row (right to left)
    for i in range(n - 1, 0, -1):
        row_data[i] = np.uint8((np.int32(row_data[i]) - np.int32(row_data[i - 1])) & 0xFF)


@ngjit
def _fp_predictor_encode_rows(data, width, height, bps):
    """Dispatch per-row encode from Numba, avoiding Python loop overhead."""
    row_len = width * bps
    for row in range(height):
        start = row * row_len
        _fp_predictor_encode_row(data[start:start + row_len], width, bps)


def fp_predictor_encode(data: np.ndarray, width: int, height: int,
                        bytes_per_sample: int) -> np.ndarray:
    """Apply floating-point predictor (predictor=3).

    Parameters
    ----------
    data : np.ndarray
        Flat uint8 array of pixel data.
    width, height : int
        Dimensions.
    bytes_per_sample : int
        Bytes per sample.

    Returns
    -------
    np.ndarray
        Encoded array.
    """
    buf = np.ascontiguousarray(data)
    _fp_predictor_encode_rows(buf, width, height, bytes_per_sample)
    return buf


# -- Sub-byte bit unpacking ---------------------------------------------------

def unpack_bits(data: np.ndarray, bps: int, pixel_count: int) -> np.ndarray:
    """Unpack sub-byte pixel data into one value per array element.

    Parameters
    ----------
    data : np.ndarray
        Flat uint8 array of packed bytes.
    bps : int
        Bits per sample (1, 2, 4, or 12).
    pixel_count : int
        Number of pixels to unpack.

    Returns
    -------
    np.ndarray
        uint8 for bps <= 8, uint16 for bps=12.
    """
    if bps == 1:
        # MSB-first: each byte holds 8 pixels
        out = np.unpackbits(data)[:pixel_count]
        return out.astype(np.uint8)
    elif bps == 2:
        # 4 pixels per byte, MSB-first
        out = np.empty(pixel_count, dtype=np.uint8)
        for i in range(min(len(data), (pixel_count + 3) // 4)):
            b = data[i]
            base = i * 4
            if base < pixel_count:
                out[base] = (b >> 6) & 0x03
            if base + 1 < pixel_count:
                out[base + 1] = (b >> 4) & 0x03
            if base + 2 < pixel_count:
                out[base + 2] = (b >> 2) & 0x03
            if base + 3 < pixel_count:
                out[base + 3] = b & 0x03
        return out
    elif bps == 4:
        # 2 pixels per byte, high nibble first
        out = np.empty(pixel_count, dtype=np.uint8)
        for i in range(min(len(data), (pixel_count + 1) // 2)):
            b = data[i]
            base = i * 2
            if base < pixel_count:
                out[base] = (b >> 4) & 0x0F
            if base + 1 < pixel_count:
                out[base + 1] = b & 0x0F
        return out
    elif bps == 12:
        # 2 pixels per 3 bytes, MSB-first
        out = np.empty(pixel_count, dtype=np.uint16)
        n_pairs = pixel_count // 2
        remainder = pixel_count % 2
        for i in range(n_pairs):
            off = i * 3
            if off + 2 < len(data):
                b0 = int(data[off])
                b1 = int(data[off + 1])
                b2 = int(data[off + 2])
                out[i * 2] = (b0 << 4) | (b1 >> 4)
                out[i * 2 + 1] = ((b1 & 0x0F) << 8) | b2
        if remainder and n_pairs * 3 + 1 < len(data):
            off = n_pairs * 3
            out[pixel_count - 1] = (int(data[off]) << 4) | (int(data[off + 1]) >> 4)
        return out
    else:
        raise ValueError(f"Unsupported sub-byte bit depth: {bps}")


# -- PackBits (simple RLE) ----------------------------------------------------

def packbits_decompress(data: bytes, expected_size: int = 0) -> bytes:
    """Decompress PackBits (TIFF compression tag 32773).

    Simple RLE: read a header byte n.
    - 0 <= n <= 127: copy the next n+1 bytes literally.
    - -127 <= n <= -1: repeat the next byte 1-n times.
    - n == -128: no-op.

    When ``expected_size`` > 0, the decoder refuses to emit more than
    ``expected_size * 1.05 + 1`` bytes and raises ``ValueError`` on overflow
    (decompression-bomb guard).
    """
    src = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    cap = _max_output_with_margin(expected_size)
    out = bytearray()
    i = 0
    length = len(src)
    while i < length:
        n = src[i]
        if n > 127:
            n = n - 256  # interpret as signed
        i += 1
        if 0 <= n <= 127:
            count = n + 1
            out.extend(src[i:i + count])
            i += count
        elif -127 <= n <= -1:
            if i < length:
                out.extend(bytes([src[i]]) * (1 - n))
                i += 1
        # n == -128: skip
        if cap and len(out) > cap:
            raise ValueError(
                f"packbits decode exceeded expected size: {len(out)} bytes "
                f"produced, cap is {cap} (expected {expected_size}).  "
                f"Likely a decompression bomb."
            )
    return bytes(out)


def packbits_compress(data: bytes) -> bytes:
    """Compress data using PackBits."""
    src = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    out = bytearray()
    i = 0
    length = len(src)
    while i < length:
        # Check for a run of identical bytes
        j = i + 1
        while j < length and j - i < 128 and src[j] == src[i]:
            j += 1
        run_len = j - i

        if run_len >= 3:
            # Encode as run
            out.append((256 - (run_len - 1)) & 0xFF)
            out.append(src[i])
            i = j
        else:
            # Literal run: accumulate non-repeating bytes
            lit_start = i
            i = j
            while i < length and i - lit_start < 128:
                # Check if a run starts here
                if i + 2 < length and src[i] == src[i + 1] == src[i + 2]:
                    break
                i += 1
            lit_len = i - lit_start
            out.append(lit_len - 1)
            out.extend(src[lit_start:lit_start + lit_len])
    return bytes(out)


# -- JPEG codec (via Pillow) --------------------------------------------------

JPEG_AVAILABLE = False
try:
    from PIL import Image
    JPEG_AVAILABLE = True
except ImportError:
    pass


def _splice_jpeg_tables(tile_data: bytes,
                        jpeg_tables: bytes | None) -> bytes:
    """Splice a JPEGTables stream into a tile's JPEG fragment.

    GDAL-style tiled JPEG TIFFs store DQT/DHT tables once in tag 347
    (an abbreviated JPEG: SOI + tables + EOI) and each tile is a JPEG
    fragment whose own DQT/DHT segments were stripped. To make a tile
    self-contained, drop the tables stream's leading SOI and trailing
    EOI and insert what remains after the tile's SOI marker.

    Both buffers must start with SOI (FF D8). If either does not, the
    tile data is returned unchanged so libjpeg sees its original input
    and raises a meaningful error.
    """
    if not jpeg_tables:
        return tile_data
    if len(tile_data) < 2 or tile_data[0] != 0xFF or tile_data[1] != 0xD8:
        return tile_data
    if len(jpeg_tables) < 4:
        return tile_data
    if jpeg_tables[0] != 0xFF or jpeg_tables[1] != 0xD8:
        return tile_data
    # Strip SOI from the tables stream, and EOI if present at the end.
    tables_body = jpeg_tables[2:]
    if len(tables_body) >= 2 and tables_body[-2] == 0xFF and tables_body[-1] == 0xD9:
        tables_body = tables_body[:-2]
    return tile_data[:2] + tables_body + tile_data[2:]


def jpeg_decompress(data: bytes, width: int = 0, height: int = 0,
                    samples: int = 1, jpeg_tables: bytes | None = None) -> bytes:
    """Decompress JPEG tile/strip data. Requires Pillow.

    Parameters
    ----------
    data : bytes
        Raw JPEG bytes from one TIFF strip or tile. May be a fragment
        when ``jpeg_tables`` is supplied (GDAL tiled JPEG).
    jpeg_tables : bytes, optional
        Contents of TIFF tag 347 (JPEGTables). If supplied, the shared
        DQT/DHT segments are spliced into ``data`` before decoding so
        the resulting stream is a complete JPEG.
    """
    if not JPEG_AVAILABLE:
        raise ImportError(
            "Pillow is required to read JPEG-compressed TIFFs. "
            "Install it with: pip install Pillow")
    import io
    if jpeg_tables:
        data = _splice_jpeg_tables(data, jpeg_tables)
    img = Image.open(io.BytesIO(data))
    # libjpeg already converts YCbCr->RGB during decode, so rely on the
    # mode Pillow returns. Calling .convert() unnecessarily would copy.
    return np.asarray(img).tobytes()


def jpeg_compress(data: bytes, width: int, height: int,
                  samples: int = 1, quality: int = 75) -> bytes:
    """Compress raw pixel data as JPEG. Requires Pillow."""
    if not JPEG_AVAILABLE:
        raise ImportError(
            "Pillow is required to write JPEG-compressed TIFFs. "
            "Install it with: pip install Pillow")
    import io
    if samples == 1:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        img = Image.fromarray(arr, mode='L')
    elif samples == 3:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        img = Image.fromarray(arr, mode='RGB')
    else:
        raise ValueError(f"JPEG compression requires 1 or 3 bands, got {samples}")
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


# -- ZSTD codec (via zstandard) -----------------------------------------------

ZSTD_AVAILABLE = False
try:
    import zstandard as _zstd
    ZSTD_AVAILABLE = True
except ImportError:
    _zstd = None


def zstd_decompress(data: bytes, expected_size: int = 0) -> bytes:
    """Decompress Zstandard data with an optional output-size cap.

    Requires the ``zstandard`` package.  When ``expected_size`` > 0 the
    decoder refuses to emit more than ``expected_size * 1.05 + 1`` bytes
    and raises ``ValueError`` on overflow (decompression-bomb guard).
    """
    if not ZSTD_AVAILABLE:
        raise ImportError(
            "zstandard is required to read ZSTD-compressed TIFFs. "
            "Install it with: pip install zstandard")
    if expected_size <= 0:
        return _zstd.ZstdDecompressor().decompress(data)
    cap = _max_output_with_margin(expected_size)
    # ``decompress(data, max_output_size=...)`` is not enforced when the
    # frame embeds the content size, so use ``stream_reader`` and read at
    # most ``cap + 1`` bytes.  If the decoder still has data after that, the
    # frame was bigger than the cap.
    reader = _zstd.ZstdDecompressor().stream_reader(data)
    try:
        out = reader.read(cap + 1)
        if len(out) > cap:
            raise ValueError(
                f"zstd decode exceeded expected size: produced more than "
                f"{cap} bytes (expected {expected_size}).  Likely a "
                f"decompression bomb."
            )
        # Probe for additional bytes — a zero-length read confirms EOF.
        extra = reader.read(1)
        if extra:
            raise ValueError(
                f"zstd decode exceeded expected size: produced more than "
                f"{cap} bytes (expected {expected_size}).  Likely a "
                f"decompression bomb."
            )
    finally:
        reader.close()
    return out


def zstd_compress(data: bytes, level: int = 3) -> bytes:
    """Compress data with Zstandard. Requires the ``zstandard`` package."""
    if not ZSTD_AVAILABLE:
        raise ImportError(
            "zstandard is required to write ZSTD-compressed TIFFs. "
            "Install it with: pip install zstandard")
    return _zstd.ZstdCompressor(level=level).compress(data)


# -- JPEG 2000 codec (via glymur) --------------------------------------------

JPEG2000_AVAILABLE = False
try:
    import glymur as _glymur
    JPEG2000_AVAILABLE = True
except ImportError:
    _glymur = None


def jpeg2000_decompress(data: bytes, width: int = 0, height: int = 0,
                        samples: int = 1, expected_size: int = 0) -> bytes:
    """Decompress a JPEG 2000 codestream. Requires ``glymur``.

    When ``expected_size`` > 0 the wrapper inspects the codestream's
    declared ``shape`` and ``dtype`` via :class:`glymur.Jp2k` (which
    parses only the SIZ marker and does not trigger pixel decoding)
    and raises ``ValueError`` when ``np.prod(shape) * dtype_bytes``
    exceeds ``expected_size * 1.05 + 1`` bytes.  This blocks
    decompression-bomb attacks where a tiny on-disk JPEG 2000 tile
    declares multi-gigabyte output dimensions.
    """
    if not JPEG2000_AVAILABLE:
        raise ImportError(
            "glymur is required to read JPEG 2000-compressed TIFFs. "
            "Install it with: pip install glymur")
    import tempfile
    import os
    # glymur reads from files, so write the codestream to a temp file
    fd, tmp = tempfile.mkstemp(suffix='.j2k')
    try:
        os.write(fd, data)
        os.close(fd)
        jp2 = _glymur.Jp2k(tmp)
        if expected_size > 0:
            try:
                shape = jp2.shape
                dtype = np.dtype(getattr(jp2, 'dtype', np.uint8))
            except Exception:
                shape = None
                dtype = None
            if shape is not None and dtype is not None:
                declared = int(np.prod(shape)) * dtype.itemsize
                cap = _max_output_with_margin(expected_size)
                if declared > cap:
                    raise ValueError(
                        f"jpeg2000 decode would exceed expected size: "
                        f"declared output is {declared} bytes (shape "
                        f"{shape}, {dtype.itemsize} B/sample), cap is "
                        f"{cap} (expected {expected_size}).  Likely a "
                        f"decompression bomb."
                    )
        arr = jp2[:]
        return arr.tobytes()
    finally:
        os.unlink(tmp)


def jpeg2000_compress(data: bytes, width: int, height: int,
                      samples: int = 1, dtype: np.dtype = np.dtype('uint8'),
                      lossless: bool = True) -> bytes:
    """Compress raw pixel data as JPEG 2000 codestream. Requires ``glymur``."""
    if not JPEG2000_AVAILABLE:
        raise ImportError(
            "glymur is required to write JPEG 2000-compressed TIFFs. "
            "Install it with: pip install glymur")
    import math
    import tempfile
    import os
    if samples == 1:
        arr = np.frombuffer(data, dtype=dtype).reshape(height, width)
    else:
        arr = np.frombuffer(data, dtype=dtype).reshape(height, width, samples)
    fd, tmp = tempfile.mkstemp(suffix='.j2k')
    os.close(fd)
    os.unlink(tmp)  # glymur needs the file to not exist
    try:
        cratios = [1] if lossless else [20]
        # numres must be <= log2(min_dim) + 1 to avoid OpenJPEG errors
        min_dim = max(min(width, height), 1)
        numres = min(6, int(math.log2(min_dim)) + 1)
        numres = max(numres, 1)
        _glymur.Jp2k(tmp, data=arr, cratios=cratios, numres=numres)
        with open(tmp, 'rb') as f:
            return f.read()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# -- LERC codec (via lerc) ----------------------------------------------------

LERC_AVAILABLE = False
try:
    import lerc as _lerc
    LERC_AVAILABLE = True
except ImportError:
    _lerc = None


# LERC dataType code -> bytes/sample.  Mirrors the enum in the LERC C++
# header: 0 int8, 1 uint8, 2 int16, 3 uint16, 4 int32, 5 uint32,
# 6 float32, 7 float64.  Used by the decompression-bomb pre-check on
# ``lerc.getLercBlobInfo`` so the blob's declared decoded byte count can
# be validated before ``lerc.decode`` allocates the full buffer.
_LERC_DTYPE_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 8}


def _check_lerc_bomb(data: bytes, expected_size: int) -> None:
    """Reject LERC blobs whose declared output exceeds the bomb cap.

    ``lerc.getLercBlobInfo`` parses the blob header without decoding,
    returning ``(errCode, version, dataType, nDim, nCols, nRows,
    nBands, ...)``. We compute ``nCols * nRows * nBands * dtype_bytes``
    and raise ``ValueError`` when the projected output exceeds the
    same margin cap (``expected_size * 1.05 + 1``) used by every other
    codec wrapper. Skipping when ``expected_size <= 0`` matches the
    existing convention: a zero (or unset) expected size disables the
    cap so direct callers and round-trip tests still work.
    """
    if expected_size <= 0:
        return
    try:
        info = _lerc.getLercBlobInfo(data)
    except Exception:
        # If the header itself is malformed, hand the blob to lerc.decode
        # so it produces the canonical error rather than masking it here.
        return
    if len(info) < 7:
        return
    data_type = int(info[2])
    n_cols = int(info[4])
    n_rows = int(info[5])
    n_bands = int(info[6])
    bytes_per_sample = _LERC_DTYPE_BYTES.get(data_type)
    if bytes_per_sample is None:
        return
    declared = n_cols * n_rows * n_bands * bytes_per_sample
    cap = _max_output_with_margin(expected_size)
    if declared > cap:
        raise ValueError(
            f"lerc decode would exceed expected size: declared output is "
            f"{declared} bytes ({n_cols}x{n_rows}x{n_bands}, "
            f"{bytes_per_sample} B/sample), cap is {cap} "
            f"(expected {expected_size}).  Likely a decompression bomb."
        )


def lerc_decompress(data: bytes, width: int = 0, height: int = 0,
                    samples: int = 1, expected_size: int = 0) -> bytes:
    """Decompress LERC data. Requires the ``lerc`` package.

    Returns the raw decoded pixel bytes.  Any LERC valid-mask is dropped
    here; masked pixels are returned as LERC's zero fill (the wire
    format's default).  Callers that need to honour the file's nodata
    value should use :func:`lerc_decompress_with_mask` instead and apply
    nodata at the array level once dtype is known.

    When ``expected_size`` > 0 the wrapper queries the blob's declared
    output size via :func:`lerc.getLercBlobInfo` and raises
    ``ValueError`` when it exceeds ``expected_size * 1.05 + 1`` bytes,
    matching the bomb cap applied by every other codec wrapper.
    """
    decoded_bytes, _mask = lerc_decompress_with_mask(
        data, expected_size=expected_size)
    return decoded_bytes


def lerc_decompress_with_mask(data: bytes, expected_size: int = 0):
    """Decompress LERC data and return ``(bytes, valid_mask_or_None)``.

    ``valid_mask`` is ``None`` when LERC reports the block is fully
    valid (no encoded mask, or a mask that is True everywhere);
    otherwise it is a numpy array whose shape matches the decoded data
    (height, width) or (height, width, samples), with ``0`` marking
    pixels the encoder flagged as invalid.  LERC zero fills masked
    positions in the data array, so the returned mask is the only
    signal that lets a reader restore the file's nodata sentinel.

    When ``expected_size`` > 0 the wrapper queries the blob's declared
    output size via :func:`lerc.getLercBlobInfo` and raises
    ``ValueError`` when it exceeds ``expected_size * 1.05 + 1`` bytes
    (decompression-bomb guard).  A 94-byte LERC blob can otherwise
    request 64 MiB of host memory because LERC compresses constant
    blocks at >700,000:1; without this pre-check the post-decode size
    check in :func:`_decode_strip_or_tile` fires only after the bomb
    has already been materialised.
    """
    if not LERC_AVAILABLE:
        raise ImportError(
            "lerc is required to read LERC-compressed TIFFs. "
            "Install it with: pip install lerc")
    _check_lerc_bomb(data, expected_size)
    result = _lerc.decode(data)
    # lerc.decode returns (result_code, data_array, valid_mask, ...)
    if result[0] != 0:
        raise RuntimeError(f"LERC decode failed with error code {result[0]}")
    arr = result[1]
    valid_mask = result[2] if len(result) > 2 else None
    if valid_mask is None:
        return arr.tobytes(), None
    valid_mask = np.asarray(valid_mask)
    if valid_mask.size == 0 or bool(valid_mask.all()):
        return arr.tobytes(), None
    return arr.tobytes(), valid_mask


def lerc_compress(data: bytes, width: int, height: int,
                  samples: int = 1, dtype: np.dtype = np.dtype('float32'),
                  max_z_error: float = 0.0) -> bytes:
    """Compress raw pixel data with LERC. Requires the ``lerc`` package.

    Parameters
    ----------
    max_z_error : float
        Maximum encoding error per pixel. 0 = lossless.
    """
    if not LERC_AVAILABLE:
        raise ImportError(
            "lerc is required to write LERC-compressed TIFFs. "
            "Install it with: pip install lerc")
    if samples == 1:
        arr = np.frombuffer(data, dtype=dtype).reshape(height, width)
    else:
        arr = np.frombuffer(data, dtype=dtype).reshape(height, width, samples)
    n_values_per_pixel = samples
    # lerc.encode(npArr, nValuesPerPixel, bHasMask, npValidMask,
    #             maxZErr, nBytesHint)
    # nBytesHint=1 triggers actual encoding (0 = compute size only)
    result = _lerc.encode(arr, n_values_per_pixel, False, None,
                          max_z_error, 1)
    if result[0] != 0:
        raise RuntimeError(f"LERC encode failed with error code {result[0]}")
    # result is (error_code, nBytesWritten, ctypes_buffer)
    return bytes(result[2])


# -- LZ4 codec (via python-lz4) -----------------------------------------------

LZ4_AVAILABLE = False
try:
    import lz4.frame as _lz4
    LZ4_AVAILABLE = True
except ImportError:
    _lz4 = None


def lz4_decompress(data: bytes, expected_size: int = 0) -> bytes:
    """Decompress LZ4 frame data with an optional output-size cap.

    Requires the ``lz4`` package.  When ``expected_size`` > 0 the decoder
    refuses to emit more than ``expected_size * 1.05 + 1`` bytes and raises
    ``ValueError`` on overflow (decompression-bomb guard).
    """
    if not LZ4_AVAILABLE:
        raise ImportError(
            "lz4 is required to read LZ4-compressed TIFFs. "
            "Install it with: pip install lz4")
    if expected_size <= 0:
        return _lz4.decompress(data)
    cap = _max_output_with_margin(expected_size)
    decompressor = _lz4.LZ4FrameDecompressor()
    out = decompressor.decompress(data, max_length=cap + 1)
    if len(out) > cap:
        raise ValueError(
            f"lz4 decode exceeded expected size: {len(out)} bytes produced, "
            f"cap is {cap} (expected {expected_size}).  Likely a "
            f"decompression bomb."
        )
    # ``needs_input == False`` means the decoder has buffered output it
    # couldn't deliver because ``max_length`` was reached.  That implies the
    # frame is larger than the cap, so the input is a bomb.
    if not decompressor.needs_input:
        raise ValueError(
            f"lz4 decode exceeded expected size: produced more than {cap} "
            f"bytes (expected {expected_size}).  Likely a decompression "
            f"bomb."
        )
    return out


def lz4_compress(data: bytes, level: int = 0) -> bytes:
    """Compress data with LZ4 frame format. Requires the ``lz4`` package."""
    if not LZ4_AVAILABLE:
        raise ImportError(
            "lz4 is required to write LZ4-compressed TIFFs. "
            "Install it with: pip install lz4")
    return _lz4.compress(data, compression_level=level)


# -- Dispatch helpers ---------------------------------------------------------

# TIFF compression tag values
COMPRESSION_NONE = 1
COMPRESSION_LZW = 5
COMPRESSION_JPEG = 7
COMPRESSION_DEFLATE = 8
COMPRESSION_JPEG2000 = 34712
COMPRESSION_ZSTD = 50000
COMPRESSION_LZ4 = 50004
COMPRESSION_PACKBITS = 32773
COMPRESSION_LERC = 34887
COMPRESSION_ADOBE_DEFLATE = 32946


def decompress(data, compression: int, expected_size: int = 0,
               width: int = 0, height: int = 0, samples: int = 1,
               jpeg_tables: bytes | None = None) -> np.ndarray:
    """Decompress tile/strip data based on TIFF compression tag.

    Parameters
    ----------
    data : bytes
        Compressed data.
    compression : int
        TIFF compression tag value.
    expected_size : int
        Expected decompressed size (used for LZW buffer allocation).

    Returns
    -------
    np.ndarray
        uint8 array. Mutable for LZW/deflate; may be read-only view for
        uncompressed data (caller must .copy() if mutation is needed).
    """
    if compression == COMPRESSION_NONE:
        return np.frombuffer(data, dtype=np.uint8)
    elif compression in (COMPRESSION_DEFLATE, COMPRESSION_ADOBE_DEFLATE):
        return np.frombuffer(
            deflate_decompress(data, expected_size), dtype=np.uint8)
    elif compression == COMPRESSION_LZW:
        return lzw_decompress(data, expected_size)
    elif compression == COMPRESSION_PACKBITS:
        return np.frombuffer(
            packbits_decompress(data, expected_size), dtype=np.uint8)
    elif compression == COMPRESSION_JPEG:
        return np.frombuffer(
            jpeg_decompress(data, width, height, samples,
                            jpeg_tables=jpeg_tables),
            dtype=np.uint8)
    elif compression == COMPRESSION_ZSTD:
        return np.frombuffer(
            zstd_decompress(data, expected_size), dtype=np.uint8)
    elif compression == COMPRESSION_JPEG2000:
        return np.frombuffer(
            jpeg2000_decompress(data, width, height, samples,
                                expected_size=expected_size),
            dtype=np.uint8)
    elif compression == COMPRESSION_LZ4:
        return np.frombuffer(
            lz4_decompress(data, expected_size), dtype=np.uint8)
    elif compression == COMPRESSION_LERC:
        return np.frombuffer(
            lerc_decompress(data, width, height, samples,
                            expected_size=expected_size),
            dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported compression type: {compression}")


def compress(data: bytes, compression: int, level: int = 6) -> bytes:
    """Compress data based on TIFF compression tag.

    Parameters
    ----------
    data : bytes
        Raw data.
    compression : int
        TIFF compression tag value.
    level : int
        Compression level (deflate: 1-9, zstd: 1-22, lz4: 0-16).
        Ignored for codecs without level support.

    Returns
    -------
    bytes
    """
    if compression == COMPRESSION_NONE:
        return data
    elif compression in (COMPRESSION_DEFLATE, COMPRESSION_ADOBE_DEFLATE):
        return deflate_compress(data, level)
    elif compression == COMPRESSION_LZW:
        return lzw_compress(data)
    elif compression == COMPRESSION_PACKBITS:
        return packbits_compress(data)
    elif compression == COMPRESSION_ZSTD:
        return zstd_compress(data, level)
    elif compression == COMPRESSION_LZ4:
        return lz4_compress(data, level)
    elif compression == COMPRESSION_JPEG:
        raise ValueError("Use jpeg_compress() directly with width/height/samples")
    elif compression == COMPRESSION_JPEG2000:
        raise ValueError("Use jpeg2000_compress() directly with width/height/samples/dtype")
    elif compression == COMPRESSION_LERC:
        raise ValueError("Use lerc_compress() directly with width/height/samples/dtype")
    else:
        raise ValueError(f"Unsupported compression type: {compression}")
