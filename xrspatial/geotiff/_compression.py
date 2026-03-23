"""Compression codecs: deflate (zlib) and LZW (Numba), plus horizontal predictor."""
from __future__ import annotations

import zlib

import numpy as np

from xrspatial.utils import ngjit

# -- Deflate (zlib wrapper) --------------------------------------------------


def deflate_decompress(data: bytes) -> bytes:
    """Decompress deflate/zlib data."""
    return zlib.decompress(data)


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

@ngjit
def _predictor_decode(data, width, height, bytes_per_sample):
    """Undo horizontal differencing predictor (TIFF predictor=2).

    Operates in-place on the flat byte array, performing cumulative sum
    per row at the sample level.
    """
    row_bytes = width * bytes_per_sample
    for row in range(height):
        row_start = row * row_bytes
        for col in range(bytes_per_sample, row_bytes):
            idx = row_start + col
            data[idx] = np.uint8((np.int32(data[idx]) + np.int32(data[idx - bytes_per_sample])) & 0xFF)


@ngjit
def _predictor_encode(data, width, height, bytes_per_sample):
    """Apply horizontal differencing predictor (TIFF predictor=2).

    Operates in-place, converting absolute values to differences.
    Process right-to-left to avoid overwriting values we still need.
    """
    row_bytes = width * bytes_per_sample
    for row in range(height):
        row_start = row * row_bytes
        for col in range(row_bytes - 1, bytes_per_sample - 1, -1):
            idx = row_start + col
            data[idx] = np.uint8((np.int32(data[idx]) - np.int32(data[idx - bytes_per_sample])) & 0xFF)


def predictor_decode(data: np.ndarray, width: int, height: int,
                     bytes_per_sample: int) -> np.ndarray:
    """Undo horizontal differencing predictor (predictor=2).

    Parameters
    ----------
    data : np.ndarray
        Flat uint8 array of decompressed pixel data (modified in-place).
    width, height : int
        Image dimensions.
    bytes_per_sample : int
        Bytes per sample (e.g. 1 for uint8, 4 for float32).

    Returns
    -------
    np.ndarray
        Same array, modified in-place.
    """
    buf = np.ascontiguousarray(data)
    _predictor_decode(buf, width, height, bytes_per_sample)
    return buf


def predictor_encode(data: np.ndarray, width: int, height: int,
                     bytes_per_sample: int) -> np.ndarray:
    """Apply horizontal differencing predictor (predictor=2).

    Parameters
    ----------
    data : np.ndarray
        Flat uint8 array of pixel data (modified in-place).
    width, height : int
        Image dimensions.
    bytes_per_sample : int
        Bytes per sample.

    Returns
    -------
    np.ndarray
        Same array, modified in-place.
    """
    buf = np.ascontiguousarray(data)
    _predictor_encode(buf, width, height, bytes_per_sample)
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
def _fp_predictor_decode_row(row_data, width, bps):
    """Undo floating-point predictor for one row (in-place).

    row_data: uint8 array of length width * bps
    """
    n = width * bps

    # Step 1: undo horizontal differencing on the byte-swizzled row
    for i in range(1, n):
        row_data[i] = np.uint8((np.int32(row_data[i]) + np.int32(row_data[i - 1])) & 0xFF)

    # Step 2: un-transpose bytes back to native sample order
    tmp = np.empty(n, dtype=np.uint8)
    for sample in range(width):
        for b in range(bps):
            tmp[sample * bps + b] = row_data[(bps - 1 - b) * width + sample]
    for i in range(n):
        row_data[i] = tmp[i]


@ngjit
def _fp_predictor_decode_rows(data, width, height, bps):
    """Dispatch per-row decode from Numba, avoiding Python loop overhead."""
    row_len = width * bps
    for row in range(height):
        start = row * row_len
        _fp_predictor_decode_row(data[start:start + row_len], width, bps)


def fp_predictor_decode(data: np.ndarray, width: int, height: int,
                        bytes_per_sample: int) -> np.ndarray:
    """Undo floating-point predictor (predictor=3).

    Parameters
    ----------
    data : np.ndarray
        Flat uint8 array of decompressed tile/strip data.
    width, height : int
        Tile/strip dimensions.
    bytes_per_sample : int
        Bytes per sample (e.g. 4 for float32, 8 for float64).

    Returns
    -------
    np.ndarray
        Corrected array.
    """
    buf = np.ascontiguousarray(data)
    _fp_predictor_decode_rows(buf, width, height, bytes_per_sample)
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
    row_len = width * bytes_per_sample
    for row in range(height):
        start = row * row_len
        _fp_predictor_encode_row(buf[start:start + row_len], width, bytes_per_sample)
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

def packbits_decompress(data: bytes) -> bytes:
    """Decompress PackBits (TIFF compression tag 32773).

    Simple RLE: read a header byte n.
    - 0 <= n <= 127: copy the next n+1 bytes literally.
    - -127 <= n <= -1: repeat the next byte 1-n times.
    - n == -128: no-op.
    """
    src = data if isinstance(data, (bytes, bytearray)) else bytes(data)
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


def jpeg_decompress(data: bytes, width: int = 0, height: int = 0,
                    samples: int = 1) -> bytes:
    """Decompress JPEG tile/strip data. Requires Pillow."""
    if not JPEG_AVAILABLE:
        raise ImportError(
            "Pillow is required to read JPEG-compressed TIFFs. "
            "Install it with: pip install Pillow")
    import io
    img = Image.open(io.BytesIO(data))
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


def zstd_decompress(data: bytes) -> bytes:
    """Decompress Zstandard data. Requires the ``zstandard`` package."""
    if not ZSTD_AVAILABLE:
        raise ImportError(
            "zstandard is required to read ZSTD-compressed TIFFs. "
            "Install it with: pip install zstandard")
    return _zstd.ZstdDecompressor().decompress(data)


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
                        samples: int = 1) -> bytes:
    """Decompress a JPEG 2000 codestream. Requires ``glymur``."""
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


def lerc_decompress(data: bytes, width: int = 0, height: int = 0,
                    samples: int = 1) -> bytes:
    """Decompress LERC data. Requires the ``lerc`` package."""
    if not LERC_AVAILABLE:
        raise ImportError(
            "lerc is required to read LERC-compressed TIFFs. "
            "Install it with: pip install lerc")
    result = _lerc.decode(data)
    # lerc.decode returns (result_code, data_array, valid_mask, ...)
    if result[0] != 0:
        raise RuntimeError(f"LERC decode failed with error code {result[0]}")
    arr = result[1]
    return arr.tobytes()


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


def lz4_decompress(data: bytes) -> bytes:
    """Decompress LZ4 frame data. Requires the ``lz4`` package."""
    if not LZ4_AVAILABLE:
        raise ImportError(
            "lz4 is required to read LZ4-compressed TIFFs. "
            "Install it with: pip install lz4")
    return _lz4.decompress(data)


def lz4_compress(data: bytes, level: int = 0) -> bytes:
    """Compress data with LZ4 frame format. Requires the ``lz4`` package."""
    if not LZ4_AVAILABLE:
        raise ImportError(
            "lz4 is required to write LZ4-compressed TIFFs. "
            "Install it with: pip install lz4")
    return _lz4.compress(data, compression_level=level)


# -- LERC codec (via lerc) ----------------------------------------------------

LERC_AVAILABLE = False
try:
    import lerc as _lerc
    LERC_AVAILABLE = True
except ImportError:
    _lerc = None


def lerc_decompress(data: bytes, width: int = 0, height: int = 0,
                    samples: int = 1) -> bytes:
    """Decompress LERC data. Requires the ``lerc`` package."""
    if not LERC_AVAILABLE:
        raise ImportError(
            "lerc is required to read LERC-compressed TIFFs. "
            "Install it with: pip install lerc")
    result = _lerc.decode(data)
    # lerc.decode returns (result_code, data_array, valid_mask, ...)
    if result[0] != 0:
        raise RuntimeError(f"LERC decode failed with error code {result[0]}")
    arr = result[1]
    return arr.tobytes()


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
               width: int = 0, height: int = 0, samples: int = 1) -> np.ndarray:
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
        return np.frombuffer(deflate_decompress(data), dtype=np.uint8)
    elif compression == COMPRESSION_LZW:
        return lzw_decompress(data, expected_size)
    elif compression == COMPRESSION_PACKBITS:
        return np.frombuffer(packbits_decompress(data), dtype=np.uint8)
    elif compression == COMPRESSION_JPEG:
        return np.frombuffer(jpeg_decompress(data, width, height, samples),
                             dtype=np.uint8)
    elif compression == COMPRESSION_ZSTD:
        return np.frombuffer(zstd_decompress(data), dtype=np.uint8)
    elif compression == COMPRESSION_JPEG2000:
        return np.frombuffer(
            jpeg2000_decompress(data, width, height, samples), dtype=np.uint8)
    elif compression == COMPRESSION_LZ4:
        return np.frombuffer(lz4_decompress(data), dtype=np.uint8)
    elif compression == COMPRESSION_LERC:
        return np.frombuffer(
            lerc_decompress(data, width, height, samples), dtype=np.uint8)
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
        Compression level (for deflate).

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
