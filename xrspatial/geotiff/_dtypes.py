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


_GDAL_OT_FOR_BPS = {
    8: 'Byte',
    16: 'UInt16',
    32: 'UInt32',
    64: 'Float64',
}


def _suggest_gdal_ot(bps_values, sample_format=None) -> str:
    """Pick a sensible ``gdal_translate -ot`` value for a mixed-bps file.

    Returns a real GDAL type name (``Byte``, ``UInt16`` etc.) when the
    widest band has a recognised mapping, or ``<desired_type>`` as a
    placeholder otherwise. ``sample_format`` (TIFF SampleFormat: 1=uint,
    2=int, 3=float) refines the integer choice when known.
    """
    if not bps_values:
        return '<desired_type>'
    widest = max(bps_values)
    if sample_format == 3 and widest in (32, 64):
        return 'Float32' if widest == 32 else 'Float64'
    if sample_format == 2 and widest in (8, 16, 32):
        return {8: 'Int8', 16: 'Int16', 32: 'Int32'}[widest]
    return _GDAL_OT_FOR_BPS.get(widest, '<desired_type>')


def resolve_bits_per_sample(bps, sample_format=None) -> int:
    """Resolve a TIFF ``BitsPerSample`` tag value to a single integer.

    The TIFF spec allows ``BitsPerSample`` to be either a scalar or a
    sequence with one entry per sample. xarray-spatial decodes a whole
    IFD with one numpy dtype, so the per-sample widths must agree.

    Parameters
    ----------
    bps : int or sequence of int
        Raw value from ``IFD.bits_per_sample``. Accepts ``int``, ``tuple``,
        or ``list``.
    sample_format : int, optional
        TIFF SampleFormat (1=uint, 2=int, 3=float). Used only to make the
        ``gdal_translate`` hint in the error message more accurate when
        the entries don't agree; not consulted when they do.

    Returns
    -------
    int
        The shared bits-per-sample value.

    Raises
    ------
    ValueError
        If ``bps`` is a sequence whose entries are not all equal. Files
        with per-band bit depths (e.g. RGB+8-bit-alpha with
        ``(16, 16, 16, 8)``) are not supported; convert with GDAL or
        rasterio first.
    """
    if isinstance(bps, (tuple, list)):
        if len(bps) == 0:
            raise ValueError("BitsPerSample tuple is empty")
        first = bps[0]
        for v in bps[1:]:
            if v != first:
                ot = _suggest_gdal_ot(bps, sample_format)
                raise ValueError(
                    f"Mixed BitsPerSample per band is not supported: "
                    f"{tuple(bps)}. xarray-spatial decodes all bands with "
                    f"a single dtype. Convert the file to a uniform bit "
                    f"depth first, e.g. "
                    f"`gdal_translate -ot {ot} in.tif out.tif`."
                )
        return int(first)
    return int(bps)


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
