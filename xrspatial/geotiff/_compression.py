"""Compression codecs: deflate (zlib) and LZW (Numba), plus horizontal predictor."""
from __future__ import annotations

import warnings
import zlib

import numpy as np

from xrspatial.utils import ngjit

# -- Optional libdeflate backend --------------------------------------------
#
# When the ``deflate`` PyPI package (Christian Heimes's libdeflate binding)
# is installed, ``deflate_compress`` routes through it. On the 1 MB random
# float32 buffer used in our write bench, libdeflate level 6 is ~3x faster
# than stdlib ``zlib`` level 6 (7.6 ms vs 24 ms), matching what GDAL/libtiff
# achieves when GDAL is built against libdeflate (the default in recent
# rasterio/GDAL wheels). Output is wire-compatible (zlib format), so encoded
# streams round-trip through stdlib ``zlib.decompress`` unchanged.
try:  # pragma: no cover - exercised only when the deflate package is installed
    import deflate as _deflate
    _HAVE_LIBDEFLATE = True
except ImportError:
    _deflate = None
    _HAVE_LIBDEFLATE = False


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


_zlib_fallback_warned = False


def deflate_compress(data: bytes, level: int = 6,
                     gil_friendly: bool = False) -> bytes:
    """Compress data with deflate/zlib.

    Uses the ``deflate`` package (libdeflate binding) when installed
    (~3x faster than stdlib ``zlib`` on typical raster payloads) and
    falls back to ``zlib.compress`` otherwise. Output is wire-compatible
    either way: the stdlib ``zlib.decompress`` accepts both.

    ``gil_friendly=True`` forces stdlib ``zlib`` regardless of libdeflate
    availability. The ``deflate`` PyPI binding does not release the GIL
    during compression (measured: 1.2x speedup across 8 threads vs zlib's
    5x speedup), so the writer's parallel paths request the GIL-releasing
    codec to keep thread-pool scaling. The sequential path leaves the
    default, picking up libdeflate's per-call speedup.
    """
    if _HAVE_LIBDEFLATE and not gil_friendly:
        # ``deflate.zlib_compress`` returns ``bytearray``; cast for the
        # ``-> bytes`` contract callers (and tests) rely on.
        return bytes(_deflate.zlib_compress(data, level))
    if not _HAVE_LIBDEFLATE:
        global _zlib_fallback_warned
        if not _zlib_fallback_warned:
            _zlib_fallback_warned = True
            warnings.warn(
                "xrspatial.geotiff: the `deflate` package is not installed; "
                "falling back to stdlib zlib for deflate-compressed writes "
                "(~3x slower on typical rasters). Install with `pip install "
                "deflate` or `pip install xarray-spatial[geotiff]` to recover "
                "full throughput.",
                stacklevel=2,
            )
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
        # 4 pixels per byte, MSB-first. Vectorised across the packed
        # byte array so a large tile decodes in one pass instead of N
        # Python iterations (issue #1713). The strided assignments
        # below mirror the original per-byte loop's write pattern,
        # including the contract that positions beyond
        # ``len(data) * 4`` are left as the ``np.empty`` initial
        # garbage value -- matching the prior behaviour for short
        # input buffers.
        out = np.empty(pixel_count, dtype=np.uint8)
        n_full_bytes = min(len(data), (pixel_count + 3) // 4)
        if n_full_bytes > 0:
            raw = data[:n_full_bytes]
            written = n_full_bytes * 4
            limit = min(written, pixel_count)
            # Compute each of the four bit-slots for every covered byte
            # then write the prefix that fits into ``out``.
            slot0 = (raw >> 6) & np.uint8(0x03)
            slot1 = (raw >> 4) & np.uint8(0x03)
            slot2 = (raw >> 2) & np.uint8(0x03)
            slot3 = raw & np.uint8(0x03)
            full_idx = limit - (limit % 4)
            if full_idx > 0:
                quart = full_idx // 4
                out[0:full_idx:4] = slot0[:quart]
                out[1:full_idx:4] = slot1[:quart]
                out[2:full_idx:4] = slot2[:quart]
                out[3:full_idx:4] = slot3[:quart]
            # Handle the tail (1 to 3 elements past the last full quad)
            tail = limit - full_idx
            tail_byte = full_idx // 4
            if tail >= 1:
                out[full_idx] = slot0[tail_byte]
            if tail >= 2:
                out[full_idx + 1] = slot1[tail_byte]
            if tail >= 3:
                out[full_idx + 2] = slot2[tail_byte]
        return out
    elif bps == 4:
        # 2 pixels per byte, high nibble first. Vectorised across the
        # packed byte array (issue #1713). Positions beyond
        # ``len(data) * 2`` keep the ``np.empty`` initial value,
        # matching the prior per-byte loop's contract for short
        # input buffers.
        out = np.empty(pixel_count, dtype=np.uint8)
        n_full_bytes = min(len(data), (pixel_count + 1) // 2)
        if n_full_bytes > 0:
            raw = data[:n_full_bytes]
            written = n_full_bytes * 2
            limit = min(written, pixel_count)
            hi = (raw >> 4) & np.uint8(0x0F)
            lo = raw & np.uint8(0x0F)
            full_idx = limit - (limit % 2)
            if full_idx > 0:
                pairs = full_idx // 2
                out[0:full_idx:2] = hi[:pairs]
                out[1:full_idx:2] = lo[:pairs]
            if limit - full_idx >= 1:
                out[full_idx] = hi[full_idx // 2]
        return out
    elif bps == 12:
        # 2 pixels per 3 bytes, MSB-first. Vectorised across the
        # packed byte array (issue #1713). Pairs whose third byte is
        # past ``len(data)`` are skipped, preserving the prior
        # ``np.empty`` semantics for short input buffers.
        out = np.empty(pixel_count, dtype=np.uint16)
        n_pairs = pixel_count // 2
        remainder = pixel_count % 2
        if n_pairs > 0:
            # Only those triples that fit entirely within ``data`` get
            # written. The prior code used ``off + 2 < len(data)``
            # (strict less than), which is satisfied when ``3i + 2 <
            # len(data)``. Solving for the largest valid ``i`` gives
            # ``i_max = (len(data) - 3) // 3``, so the pair count cap
            # is ``i_max + 1 = len(data) // 3``.
            max_pairs = len(data) // 3
            usable_pairs = min(n_pairs, max_pairs)
            if usable_pairs > 0:
                raw = data[:usable_pairs * 3].astype(np.uint16)
                b0 = raw[0::3]
                b1 = raw[1::3]
                b2 = raw[2::3]
                out[0:usable_pairs * 2:2] = (b0 << 4) | (b1 >> 4)
                out[1:usable_pairs * 2:2] = ((b1 & 0x0F) << 8) | b2
        if remainder and n_pairs * 3 + 1 < len(data):
            off = n_pairs * 3
            out[pixel_count - 1] = (int(data[off]) << 4) | (int(data[off + 1]) >> 4)
        return out
    else:
        raise ValueError(f"Unsupported sub-byte bit depth: {bps}")


# -- PackBits (simple RLE) ----------------------------------------------------

@ngjit
def _packbits_decode_kernel(src, src_len, dst, dst_cap):
    """Decode PackBits (TIFF compression tag 32773) into a uint8 buffer.

    Parameters
    ----------
    src : uint8 array
        Compressed bytes.
    src_len : int
        Number of valid bytes in ``src``.
    dst : uint8 array
        Pre-allocated output buffer.
    dst_cap : int
        Maximum bytes to write before bailing out. When ``0``, the kernel
        writes up to ``len(dst)`` (the wrapper sizes ``dst`` for the
        no-cap path so this is safe).

    Returns
    -------
    int
        Number of bytes written. ``-1`` signals the cap was exceeded
        (the wrapper turns this into ``ValueError``).
    """
    out_pos = 0
    i = 0
    # When dst_cap is zero we use the full buffer length as the write limit.
    if dst_cap > 0:
        write_limit = dst_cap
        enforce_cap = True
    else:
        write_limit = len(dst)
        enforce_cap = False

    while i < src_len:
        header = src[i]
        i += 1
        if header < 128:
            # Literal run of header + 1 bytes.
            count = header + 1
            # Don't read past end of src on a truncated stream.
            available = src_len - i
            if count > available:
                count = available
            for k in range(count):
                if out_pos >= write_limit:
                    if enforce_cap:
                        return -1
                    # write_limit equals len(dst); refuse to write past it.
                    return out_pos
                dst[out_pos] = src[i + k]
                out_pos += 1
            i += count
        elif header > 128:
            # Replicate run: header interpreted as signed yields n in
            # [-127, -1]; repeat count is 1 - n == 257 - header.
            count = 257 - header
            if i >= src_len:
                break
            byte_val = src[i]
            i += 1
            for _ in range(count):
                if out_pos >= write_limit:
                    if enforce_cap:
                        return -1
                    return out_pos
                dst[out_pos] = byte_val
                out_pos += 1
        # header == 128: no-op marker.

    return out_pos


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
    if isinstance(data, (bytes, bytearray, memoryview)):
        src = np.frombuffer(data, dtype=np.uint8)
    else:
        src = np.frombuffer(bytes(data), dtype=np.uint8)

    src_len = len(src)
    if src_len == 0:
        return b""

    cap = _max_output_with_margin(expected_size)
    if cap > 0:
        # dst sized exactly to the cap; the kernel returns -1 the moment
        # out_pos would reach write_limit (= cap), so we never need a
        # sentinel byte past the cap.
        dst = np.empty(cap, dtype=np.uint8)
        n_written = _packbits_decode_kernel(src, src_len, dst, cap)
    else:
        # No cap supplied. PackBits expands by at most 128:1 (a single
        # replicate header byte yields 128 output bytes), so this bound
        # is tight enough to keep the decoder bounded even on adversarial
        # input while still always fitting the legitimate output. The
        # reader path always supplies expected_size, so this branch is
        # only hit by direct callers and round-trip tests where the
        # peak allocation is acceptable.
        worst_case = src_len * 128
        dst = np.empty(worst_case, dtype=np.uint8)
        n_written = _packbits_decode_kernel(src, src_len, dst, 0)

    if n_written < 0:
        raise ValueError(
            f"packbits decode exceeded expected size: produced more than "
            f"{cap} bytes (cap = expected_size * 1.05 + 1, expected "
            f"{expected_size}).  Likely a decompression bomb."
        )
    return bytes(dst[:n_written])


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


# JPEG Start-Of-Frame marker codes: 0xFFC0..0xFFCF *except* DHT (0xC4),
# JPG (0xC8), and DAC (0xCC). The payload format is the same across every
# SOF variant: 2-byte segment length, 1-byte sample precision, 2-byte
# image height (big-endian), 2-byte image width (big-endian), 1-byte
# component count.
_JPEG_SOF_CODES = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _read_jpeg_sof(data: bytes) -> tuple[int, int, int] | None:
    """Return ``(height, width, components)`` from the first JPEG SOF marker.

    Scans *data* for the first Start-Of-Frame marker without decoding any
    pixel data. Returns ``None`` when the buffer is too short, when the
    SOI marker is missing, when a length-prefixed segment runs off the
    end, or when no SOF is found before EOI/end-of-buffer. The caller
    treats a ``None`` return as "could not determine declared size" and
    falls back to Pillow's own guard.

    Used by :func:`jpeg_decompress` to enforce a pre-decode bomb cap
    analogous to the JP2K SIZ pre-check and the LERC ``getLercBlobInfo``
    pre-check.
    """
    n = len(data)
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    while i + 3 < n:
        if data[i] != 0xFF:
            return None
        # Skip JPEG fill bytes (0xFF padding).
        marker = data[i + 1]
        while marker == 0xFF and i + 2 < n:
            i += 1
            marker = data[i + 1]
        # Standalone markers without payload: SOI, EOI, RSTm, TEM.
        # Encountering EOI before SOF means the stream is malformed for
        # this purpose; caller falls back to Pillow.
        if marker == 0xD9:  # EOI
            return None
        if marker in (0xD8,) or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        if marker in _JPEG_SOF_CODES:
            # SOF is a length-prefixed segment; validate the declared
            # segment length before reading the fixed fields so a
            # truncated/malformed segment header is rejected as "unknown
            # size" rather than silently reading past the segment into
            # later bytes. The fields we need are at offsets 5..9 in the
            # segment, which sits inside the declared seg_len bytes.
            if i + 3 >= n:
                return None
            seg_len = (data[i + 2] << 8) | data[i + 3]
            # SOF requires at least: 2 (length) + 1 (precision) + 2 (h)
            # + 2 (w) + 1 (components) = 8 bytes; the segment must also
            # fit inside the buffer.
            if seg_len < 8 or i + 2 + seg_len > n:
                return None
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            components = data[i + 9]
            return height, width, components
        # Length-prefixed segment: payload length includes the two length
        # bytes themselves but not the marker.
        if i + 3 >= n:
            return None
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


def _check_jpeg_bomb(data: bytes, expected_width: int, expected_height: int,
                     expected_samples: int) -> None:
    """Reject JPEG blobs whose SOF dimensions exceed the expected tile size.

    The caller passes the TIFF-declared tile width/height/samples; we
    parse the JPEG SOF marker to discover the JPEG's own declared
    dimensions and refuse to decode when the projected output exceeds
    the same ``expected * 1.05 + 1`` margin used by every other codec
    wrapper. Skipping when any expected value is non-positive matches
    the convention from the other codecs: callers that don't supply a
    size fall back to Pillow's built-in ``DecompressionBombError``
    guard.

    A return of ``None`` from :func:`_read_jpeg_sof` is treated as
    "could not determine declared size"; in that case we defer to
    Pillow rather than raising, so legitimate streams with unusual
    framing still decode.
    """
    if expected_width <= 0 or expected_height <= 0 or expected_samples <= 0:
        return
    info = _read_jpeg_sof(data)
    if info is None:
        return
    declared_h, declared_w, declared_components = info
    # JPEG components ship as 8-bit samples in every TIFF JPEG we
    # support, so ``width * height * components`` equals the post-decode
    # byte count exactly. JPEG-12 (12-bit precision) would round up to
    # 2 bytes per sample, but tifffile/Pillow doesn't surface those on
    # the read path so we treat samples as bytes here. The expected
    # side uses what the *caller* declared (TIFF tile size and
    # samples-per-pixel), so a JPEG claiming extra components is
    # rejected as a bomb.
    expected_bytes = expected_width * expected_height * expected_samples
    declared_bytes = declared_w * declared_h * max(declared_components, 1)
    cap = _max_output_with_margin(expected_bytes)
    if declared_bytes > cap:
        raise ValueError(
            f"jpeg decode would exceed expected size: declared output is "
            f"{declared_bytes} bytes ({declared_w}x{declared_h}x"
            f"{declared_components}), cap is {cap} (expected "
            f"{expected_bytes}). Likely a decompression bomb."
        )


def jpeg_decompress(data: bytes, width: int = 0, height: int = 0,
                    samples: int = 1, jpeg_tables: bytes | None = None) -> bytes:
    """Decompress JPEG tile/strip data. Requires Pillow.

    Parameters
    ----------
    data : bytes
        Raw JPEG bytes from one TIFF strip or tile. May be a fragment
        when ``jpeg_tables`` is supplied (GDAL tiled JPEG).
    width, height, samples : int, optional
        TIFF-declared tile dimensions. When all three are positive the
        wrapper inspects the JPEG SOF marker before decoding and raises
        ``ValueError`` when the JPEG's declared output exceeds
        ``width * height * samples * 1.05 + 1`` bytes
        (decompression-bomb guard). Defaults of 0/0/1 disable the cap
        for direct callers and round-trip tests.
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
    # Pre-decode bomb cap (issue #1792). Pillow's built-in
    # DecompressionBombError fires at ~178M pixels (~500 MB RGB) which
    # is well above a typical tile's expected size; without this
    # per-tile pre-check a single malicious tile can allocate hundreds
    # of MB before the downstream chunk.size != expected check fires.
    _check_jpeg_bomb(data, width, height, samples)
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
    and raises ``ValueError`` when ``prod(shape) * dtype_bytes`` exceeds
    ``expected_size * 1.05 + 1`` bytes.  The cap fails closed: if the
    SIZ marker is unreadable, the function raises rather than letting
    ``glymur.Jp2k[:]`` materialise an attacker-controlled buffer.
    """
    if not JPEG2000_AVAILABLE:
        raise ImportError(
            "glymur is required to read JPEG 2000-compressed TIFFs. "
            "Install it with: pip install glymur")
    import math
    import tempfile
    import os
    # glymur reads from files, so write the codestream to a temp file
    fd, tmp = tempfile.mkstemp(suffix='.j2k')
    try:
        os.write(fd, data)
        os.close(fd)
        jp2 = _glymur.Jp2k(tmp)
        if expected_size > 0:
            # Fail-closed pre-decode cap. ``jp2.shape`` reads only the
            # SIZ marker (sub-millisecond, no pixel decoding); if we
            # cannot determine the declared shape and dtype, refuse to
            # call ``jp2[:]`` rather than silently disabling the cap --
            # an attacker who can produce a malformed-but-decodable
            # codestream would otherwise bypass the guard.
            try:
                shape = jp2.shape
            except Exception as exc:
                raise ValueError(
                    "jpeg2000 decode rejected: unable to read declared "
                    "shape from SIZ marker"
                ) from exc
            try:
                dtype = np.dtype(getattr(jp2, 'dtype', None))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "jpeg2000 decode rejected: unable to read declared "
                    "dtype from codestream"
                ) from exc
            # ``math.prod`` uses arbitrary-precision Python ints, so it
            # cannot overflow on attacker-declared dimensions (unlike
            # ``np.prod`` which multiplies in fixed-width intp and can
            # wrap to a small positive number, masking a bomb).
            declared = math.prod(int(d) for d in shape) * dtype.itemsize
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
    # ``info[0]`` is the LERC errCode; when non-zero the remaining
    # fields may be uninitialised or garbage. Defer to ``lerc.decode``
    # for the canonical error instead of computing a declared size
    # from invalid header values.
    if int(info[0]) != 0:
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


def compress(data: bytes, compression: int, level: int = 6,
             gil_friendly: bool = False) -> bytes:
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
    gil_friendly : bool
        When True, prefer codec variants that release the GIL (used by
        the parallel writer paths so a thread pool actually scales). Only
        affects deflate: stdlib ``zlib`` releases the GIL, the ``deflate``
        package's binding does not. All other codecs already release the
        GIL and ignore this flag.

    Returns
    -------
    bytes
    """
    if compression == COMPRESSION_NONE:
        return data
    elif compression in (COMPRESSION_DEFLATE, COMPRESSION_ADOBE_DEFLATE):
        return deflate_compress(data, level, gil_friendly=gil_friendly)
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
