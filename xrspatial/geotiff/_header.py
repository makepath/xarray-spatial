"""TIFF/BigTIFF header and IFD parsing."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from ._dtypes import (
    TIFF_TYPE_SIZES,
    TIFF_TYPE_STRUCT_CODES,
    RATIONAL,
    SRATIONAL,
    ASCII,
    UNDEFINED,
)

# Well-known TIFF tag IDs
TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC = 262
TAG_STRIP_OFFSETS = 273
TAG_SAMPLES_PER_PIXEL = 277
TAG_ROWS_PER_STRIP = 278
TAG_STRIP_BYTE_COUNTS = 279
TAG_X_RESOLUTION = 282
TAG_Y_RESOLUTION = 283
TAG_PLANAR_CONFIG = 284
TAG_RESOLUTION_UNIT = 296
TAG_PREDICTOR = 317
TAG_TILE_WIDTH = 322
TAG_TILE_LENGTH = 323
TAG_TILE_OFFSETS = 324
TAG_TILE_BYTE_COUNTS = 325
TAG_COLORMAP = 320
TAG_SAMPLE_FORMAT = 339
TAG_GDAL_METADATA = 42112
TAG_GDAL_NODATA = 42113

# GeoTIFF tags
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922
TAG_MODEL_TRANSFORMATION = 34264
TAG_GEO_KEY_DIRECTORY = 34735
TAG_GEO_DOUBLE_PARAMS = 34736
TAG_GEO_ASCII_PARAMS = 34737


@dataclass
class TIFFHeader:
    """Parsed TIFF file header."""
    byte_order: str  # '<' or '>'
    is_bigtiff: bool
    first_ifd_offset: int


@dataclass
class IFDEntry:
    """A single IFD entry with its resolved value."""
    tag: int
    type_id: int
    count: int
    value: Any  # resolved: int, float, tuple, bytes, or str


@dataclass
class IFD:
    """Parsed Image File Directory."""
    entries: dict[int, IFDEntry] = field(default_factory=dict)
    next_ifd_offset: int = 0

    def get_value(self, tag: int, default: Any = None) -> Any:
        """Get the resolved value for a tag, or default if absent."""
        entry = self.entries.get(tag)
        if entry is None:
            return default
        return entry.value

    def get_values(self, tag: int) -> tuple | None:
        """Get a tag's value as a tuple (even if scalar)."""
        entry = self.entries.get(tag)
        if entry is None:
            return None
        v = entry.value
        if isinstance(v, tuple):
            return v
        return (v,)

    # Convenience properties
    @property
    def width(self) -> int:
        return self.get_value(TAG_IMAGE_WIDTH, 0)

    @property
    def height(self) -> int:
        return self.get_value(TAG_IMAGE_LENGTH, 0)

    @property
    def bits_per_sample(self) -> int | tuple:
        v = self.get_value(TAG_BITS_PER_SAMPLE, 8)
        if isinstance(v, tuple):
            return v[0] if len(v) == 1 else v
        return v

    @property
    def samples_per_pixel(self) -> int:
        return self.get_value(TAG_SAMPLES_PER_PIXEL, 1)

    @property
    def sample_format(self) -> int:
        v = self.get_value(TAG_SAMPLE_FORMAT, 1)
        if isinstance(v, tuple):
            return v[0]
        return v

    @property
    def compression(self) -> int:
        return self.get_value(TAG_COMPRESSION, 1)

    @property
    def predictor(self) -> int:
        return self.get_value(TAG_PREDICTOR, 1)

    @property
    def is_tiled(self) -> bool:
        return TAG_TILE_WIDTH in self.entries

    @property
    def tile_width(self) -> int:
        return self.get_value(TAG_TILE_WIDTH, 0)

    @property
    def tile_height(self) -> int:
        return self.get_value(TAG_TILE_LENGTH, 0)

    @property
    def rows_per_strip(self) -> int:
        # Default: entire image in one strip
        return self.get_value(TAG_ROWS_PER_STRIP, self.height)

    @property
    def strip_offsets(self) -> tuple | None:
        return self.get_values(TAG_STRIP_OFFSETS)

    @property
    def strip_byte_counts(self) -> tuple | None:
        return self.get_values(TAG_STRIP_BYTE_COUNTS)

    @property
    def tile_offsets(self) -> tuple | None:
        return self.get_values(TAG_TILE_OFFSETS)

    @property
    def tile_byte_counts(self) -> tuple | None:
        return self.get_values(TAG_TILE_BYTE_COUNTS)

    @property
    def photometric(self) -> int:
        return self.get_value(TAG_PHOTOMETRIC, 1)

    @property
    def planar_config(self) -> int:
        return self.get_value(TAG_PLANAR_CONFIG, 1)

    @property
    def x_resolution(self) -> float | None:
        """XResolution tag (282), or None if absent."""
        v = self.get_value(TAG_X_RESOLUTION)
        return float(v) if v is not None else None

    @property
    def y_resolution(self) -> float | None:
        """YResolution tag (283), or None if absent."""
        v = self.get_value(TAG_Y_RESOLUTION)
        return float(v) if v is not None else None

    @property
    def resolution_unit(self) -> int | None:
        """ResolutionUnit tag (296): 1=none, 2=inch, 3=cm. None if absent."""
        return self.get_value(TAG_RESOLUTION_UNIT)

    @property
    def colormap(self) -> tuple | None:
        """ColorMap tag (320) values, or None if absent."""
        return self.get_values(TAG_COLORMAP)

    @property
    def gdal_metadata(self) -> str | None:
        """GDALMetadata XML string (tag 42112), or None if absent."""
        v = self.get_value(TAG_GDAL_METADATA)
        if v is None:
            return None
        if isinstance(v, bytes):
            return v.rstrip(b'\x00').decode('ascii', errors='replace')
        return str(v).rstrip('\x00')

    @property
    def nodata_str(self) -> str | None:
        """GDAL_NODATA tag value as string, or None."""
        v = self.get_value(TAG_GDAL_NODATA)
        if v is None:
            return None
        if isinstance(v, bytes):
            return v.rstrip(b'\x00').decode('ascii', errors='replace')
        return str(v).rstrip('\x00')


def parse_header(data: bytes | memoryview) -> TIFFHeader:
    """Parse a TIFF/BigTIFF file header.

    Parameters
    ----------
    data : bytes
        At least the first 16 bytes of the file.

    Returns
    -------
    TIFFHeader
    """
    if len(data) < 8:
        raise ValueError("Not enough data for TIFF header")

    bom = data[0:2]
    if bom == b'II':
        bo = '<'
    elif bom == b'MM':
        bo = '>'
    else:
        raise ValueError(f"Invalid TIFF byte order marker: {bom!r}")

    magic = struct.unpack_from(f'{bo}H', data, 2)[0]

    if magic == 42:
        # Standard TIFF
        offset = struct.unpack_from(f'{bo}I', data, 4)[0]
        return TIFFHeader(byte_order=bo, is_bigtiff=False, first_ifd_offset=offset)
    elif magic == 43:
        # BigTIFF
        if len(data) < 16:
            raise ValueError("Not enough data for BigTIFF header")
        offset_size = struct.unpack_from(f'{bo}H', data, 4)[0]
        if offset_size != 8:
            raise ValueError(f"Unexpected BigTIFF offset size: {offset_size}")
        # skip 2 bytes padding
        offset = struct.unpack_from(f'{bo}Q', data, 8)[0]
        return TIFFHeader(byte_order=bo, is_bigtiff=True, first_ifd_offset=offset)
    else:
        raise ValueError(f"Invalid TIFF magic number: {magic}")


def _read_value(data: bytes | memoryview, offset: int, type_id: int,
                count: int, bo: str) -> Any:
    """Read a typed value array from data at the given offset."""
    type_size = TIFF_TYPE_SIZES.get(type_id, 1)

    if type_id == ASCII:
        raw = bytes(data[offset:offset + count])
        # Strip trailing null
        return raw.rstrip(b'\x00').decode('ascii', errors='replace')

    if type_id == UNDEFINED:
        return bytes(data[offset:offset + count])

    if type_id == RATIONAL:
        values = []
        for i in range(count):
            off = offset + i * 8
            num = struct.unpack_from(f'{bo}I', data, off)[0]
            den = struct.unpack_from(f'{bo}I', data, off + 4)[0]
            values.append(num / den if den != 0 else 0.0)
        return tuple(values) if count > 1 else values[0]

    if type_id == SRATIONAL:
        values = []
        for i in range(count):
            off = offset + i * 8
            num = struct.unpack_from(f'{bo}i', data, off)[0]
            den = struct.unpack_from(f'{bo}i', data, off + 4)[0]
            values.append(num / den if den != 0 else 0.0)
        return tuple(values) if count > 1 else values[0]

    fmt_char = TIFF_TYPE_STRUCT_CODES.get(type_id)
    if fmt_char is None:
        return bytes(data[offset:offset + count * type_size])

    if count == 1:
        return struct.unpack_from(f'{bo}{fmt_char}', data, offset)[0]

    # Batch unpack: single call for all elements
    return struct.unpack_from(f'{bo}{count}{fmt_char}', data, offset)


def parse_ifd(data: bytes | memoryview, offset: int,
              header: TIFFHeader) -> IFD:
    """Parse a single IFD at the given offset.

    Parameters
    ----------
    data : bytes
        Full file data (or at least enough of it).
    offset : int
        Byte offset of this IFD.
    header : TIFFHeader
        Parsed file header.

    Returns
    -------
    IFD
    """
    bo = header.byte_order
    is_big = header.is_bigtiff

    if is_big:
        num_entries = struct.unpack_from(f'{bo}Q', data, offset)[0]
        entry_offset = offset + 8
        entry_size = 20
    else:
        num_entries = struct.unpack_from(f'{bo}H', data, offset)[0]
        entry_offset = offset + 2
        entry_size = 12

    inline_max = 8 if is_big else 4
    entries = {}

    for i in range(num_entries):
        eo = entry_offset + i * entry_size

        if is_big:
            tag = struct.unpack_from(f'{bo}H', data, eo)[0]
            type_id = struct.unpack_from(f'{bo}H', data, eo + 2)[0]
            count = struct.unpack_from(f'{bo}Q', data, eo + 4)[0]
            value_area_offset = eo + 12
        else:
            tag = struct.unpack_from(f'{bo}H', data, eo)[0]
            type_id = struct.unpack_from(f'{bo}H', data, eo + 2)[0]
            count = struct.unpack_from(f'{bo}I', data, eo + 4)[0]
            value_area_offset = eo + 8

        type_size = TIFF_TYPE_SIZES.get(type_id, 1)
        total_size = count * type_size

        if total_size <= inline_max:
            value = _read_value(data, value_area_offset, type_id, count, bo)
        else:
            if is_big:
                ptr = struct.unpack_from(f'{bo}Q', data, value_area_offset)[0]
            else:
                ptr = struct.unpack_from(f'{bo}I', data, value_area_offset)[0]
            value = _read_value(data, ptr, type_id, count, bo)

        entries[tag] = IFDEntry(tag=tag, type_id=type_id, count=count, value=value)

    # Next IFD offset
    next_offset_pos = entry_offset + num_entries * entry_size
    if is_big:
        next_ifd = struct.unpack_from(f'{bo}Q', data, next_offset_pos)[0]
    else:
        next_ifd = struct.unpack_from(f'{bo}I', data, next_offset_pos)[0]

    return IFD(entries=entries, next_ifd_offset=next_ifd)


def parse_all_ifds(data: bytes | memoryview,
                   header: TIFFHeader) -> list[IFD]:
    """Parse all IFDs in a TIFF file.

    Parameters
    ----------
    data : bytes
        Full file data.
    header : TIFFHeader
        Parsed file header.

    Returns
    -------
    list[IFD]
    """
    ifds = []
    offset = header.first_ifd_offset
    seen = set()

    while offset != 0 and offset not in seen:
        seen.add(offset)
        if offset >= len(data):
            break
        ifd = parse_ifd(data, offset, header)
        ifds.append(ifd)
        offset = ifd.next_ifd_offset

    return ifds
