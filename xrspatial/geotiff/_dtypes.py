"""TIFF type ID <-> numpy dtype mapping."""
from __future__ import annotations

import numpy as np

# TIFF type IDs (baseline + BigTIFF extensions)
BYTE = 1
ASCII = 2
SHORT = 3
LONG = 4
RATIONAL = 5
SBYTE = 6
UNDEFINED = 7
SSHORT = 8
SLONG = 9
SRATIONAL = 10
FLOAT = 11
DOUBLE = 12
# BigTIFF additions
LONG8 = 16
SLONG8 = 17
IFD8 = 18

# Bytes per element for each TIFF type
TIFF_TYPE_SIZES: dict[int, int] = {
    BYTE: 1,
    ASCII: 1,
    SHORT: 2,
    LONG: 4,
    RATIONAL: 8,    # two LONGs
    SBYTE: 1,
    UNDEFINED: 1,
    SSHORT: 2,
    SLONG: 4,
    SRATIONAL: 8,   # two SLONGs
    FLOAT: 4,
    DOUBLE: 8,
    LONG8: 8,
    SLONG8: 8,
    IFD8: 8,
}

# struct format characters for single values (excludes RATIONAL/SRATIONAL)
TIFF_TYPE_STRUCT_CODES: dict[int, str] = {
    BYTE: 'B',
    ASCII: 's',
    SHORT: 'H',
    LONG: 'I',
    SBYTE: 'b',
    UNDEFINED: 'B',
    SSHORT: 'h',
    SLONG: 'i',
    FLOAT: 'f',
    DOUBLE: 'd',
    LONG8: 'Q',
    SLONG8: 'q',
    IFD8: 'Q',
}

# SampleFormat tag values
SAMPLE_FORMAT_UINT = 1
SAMPLE_FORMAT_INT = 2
SAMPLE_FORMAT_FLOAT = 3
SAMPLE_FORMAT_UNDEFINED = 4


def tiff_dtype_to_numpy(bits_per_sample: int, sample_format: int = 1) -> np.dtype:
    """Convert TIFF BitsPerSample + SampleFormat to a numpy dtype.

    Parameters
    ----------
    bits_per_sample : int
        Bits per sample (8, 16, 32, 64).
    sample_format : int
        TIFF SampleFormat tag value (1=uint, 2=int, 3=float).

    Returns
    -------
    np.dtype
    """
    _map = {
        (8, SAMPLE_FORMAT_UINT): np.dtype('uint8'),
        (8, SAMPLE_FORMAT_INT): np.dtype('int8'),
        (16, SAMPLE_FORMAT_UINT): np.dtype('uint16'),
        (16, SAMPLE_FORMAT_INT): np.dtype('int16'),
        (32, SAMPLE_FORMAT_UINT): np.dtype('uint32'),
        (32, SAMPLE_FORMAT_INT): np.dtype('int32'),
        (32, SAMPLE_FORMAT_FLOAT): np.dtype('float32'),
        (64, SAMPLE_FORMAT_UINT): np.dtype('uint64'),
        (64, SAMPLE_FORMAT_INT): np.dtype('int64'),
        (64, SAMPLE_FORMAT_FLOAT): np.dtype('float64'),
        # treat UNDEFINED same as UINT
        (8, SAMPLE_FORMAT_UNDEFINED): np.dtype('uint8'),
        (16, SAMPLE_FORMAT_UNDEFINED): np.dtype('uint16'),
        (32, SAMPLE_FORMAT_UNDEFINED): np.dtype('uint32'),
        (64, SAMPLE_FORMAT_UNDEFINED): np.dtype('uint64'),
        # Sub-byte and non-standard bit depths: promoted to smallest
        # numpy type that can hold the values.
        (1, SAMPLE_FORMAT_UINT): np.dtype('uint8'),
        (1, SAMPLE_FORMAT_UNDEFINED): np.dtype('uint8'),
        (2, SAMPLE_FORMAT_UINT): np.dtype('uint8'),
        (2, SAMPLE_FORMAT_UNDEFINED): np.dtype('uint8'),
        (4, SAMPLE_FORMAT_UINT): np.dtype('uint8'),
        (4, SAMPLE_FORMAT_UNDEFINED): np.dtype('uint8'),
        (12, SAMPLE_FORMAT_UINT): np.dtype('uint16'),
        (12, SAMPLE_FORMAT_UNDEFINED): np.dtype('uint16'),
    }
    key = (bits_per_sample, sample_format)
    if key not in _map:
        raise ValueError(
            f"Unsupported BitsPerSample={bits_per_sample}, "
            f"SampleFormat={sample_format}"
        )
    return _map[key]


# Set of BitsPerSample values that require bit-level unpacking
SUB_BYTE_BPS = {1, 2, 4, 12}


def numpy_to_tiff_dtype(dt: np.dtype) -> tuple[int, int]:
    """Convert a numpy dtype to (bits_per_sample, sample_format).

    Returns
    -------
    (bits_per_sample, sample_format) tuple
    """
    dt = np.dtype(dt)
    if dt.kind == 'u':
        return (dt.itemsize * 8, SAMPLE_FORMAT_UINT)
    elif dt.kind == 'i':
        return (dt.itemsize * 8, SAMPLE_FORMAT_INT)
    elif dt.kind == 'f':
        return (dt.itemsize * 8, SAMPLE_FORMAT_FLOAT)
    else:
        raise ValueError(f"Unsupported numpy dtype: {dt}")
