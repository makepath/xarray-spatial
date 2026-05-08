"""TIFF/BigTIFF header and IFD parsing."""
from __future__ import annotations

import math
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
TAG_NEW_SUBFILE_TYPE = 254
TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC = 262
TAG_STRIP_OFFSETS = 273
TAG_ORIENTATION = 274
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
TAG_EXTRA_SAMPLES = 338
TAG_SAMPLE_FORMAT = 339
TAG_JPEG_TABLES = 347
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
    def subfile_type(self) -> int:
        """NewSubfileType (tag 254) bit flags. 0 if absent.

        Bit flags (TIFF 6.0 spec):
            bit 0 (& 1) - reduced-resolution overview
            bit 1 (& 2) - page of multi-page document
            bit 2 (& 4) - transparency mask
        """
        v = self.get_value(TAG_NEW_SUBFILE_TYPE, 0)
        if isinstance(v, tuple):
            v = v[0] if v else 0
        return int(v)

    @property
    def is_mask(self) -> bool:
        """True if this IFD's NewSubfileType marks it as a transparency mask."""
        return bool(self.subfile_type & 4)

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
    def orientation(self) -> int:
        """Orientation tag (274). Default 1 = top-left (no transform).

        Per TIFF 6.0 the eight valid values are:
        1=top-left, 2=top-right, 3=bottom-right, 4=bottom-left,
        5=left-top, 6=right-top, 7=right-bottom, 8=left-bottom.
        Values 5-8 swap rows and columns relative to the stored layout.
        """
        v = self.get_value(TAG_ORIENTATION, 1)
        if isinstance(v, tuple):
            v = v[0]
        return int(v)

    @property
    def planar_config(self) -> int:
        return self.get_value(TAG_PLANAR_CONFIG, 1)

    @property
    def jpeg_tables(self) -> bytes | None:
        """JPEGTables tag (347): shared DQT/DHT segments for tiled JPEG.

        GDAL-tiled ``compress=JPEG`` TIFFs store the quantization and
        Huffman tables once in this tag; each tile's payload is a JPEG
        fragment that needs the tables spliced in before libjpeg can
        decode it. Returns the raw bytes of the abbreviated JPEG stream
        (SOI ... DQT/DHT ... EOI), or None if absent.
        """
        v = self.get_value(TAG_JPEG_TABLES)
        if v is None:
            return None
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        # BYTE arrays may surface as a tuple/list of ints
        if isinstance(v, (tuple, list)):
            return bytes(v)
        # A single-byte tag value comes back as an int; wrap it in a
        # one-element bytes object. Plain ``bytes(v)`` would (incorrectly)
        # allocate v zero bytes -- a malformed file with a huge int here
        # could otherwise blow up memory.
        if isinstance(v, int):
            return bytes([v & 0xFF])
        raise TypeError(
            f"unexpected JPEGTables tag value type: {type(v).__name__}"
        )

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


def validate_tile_layout(ifd: IFD) -> None:
    """Validate that a tiled IFD's TileOffsets covers the declared tile grid.

    A well-formed tiled TIFF must supply at least `tiles_across * tiles_down`
    TileOffsets entries (times samples_per_pixel for planar config 2). An
    adversarial or malformed file can declare larger image dimensions than
    its offsets array covers, which causes out-of-bounds reads in
    downstream decoders (notably the GPU tile-assembly kernel).

    Parameters
    ----------
    ifd : IFD
        Parsed IFD. Must be tiled.

    Raises
    ------
    ValueError
        If TileOffsets or TileByteCounts is missing, if tile width/height
        is zero, or if the declared grid exceeds the offsets array length.
    """
    if not ifd.is_tiled:
        return

    offsets = ifd.tile_offsets
    byte_counts = ifd.tile_byte_counts
    if offsets is None or byte_counts is None:
        raise ValueError("Tiled TIFF is missing TileOffsets or TileByteCounts")

    tw = ifd.tile_width
    th = ifd.tile_height
    if tw <= 0 or th <= 0:
        raise ValueError(
            f"Invalid tile dimensions: tile_width={tw}, tile_height={th}")

    width = ifd.width
    height = ifd.height
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: width={width}, height={height}")

    tiles_across = math.ceil(width / tw)
    tiles_down = math.ceil(height / th)
    planar = ifd.planar_config
    samples = ifd.samples_per_pixel
    bands = samples if (planar == 2 and samples > 1) else 1
    expected = tiles_across * tiles_down * bands

    if len(offsets) < expected:
        raise ValueError(
            f"Malformed TIFF: declared tile grid requires {expected} tile "
            f"offsets ({tiles_across} x {tiles_down}"
            f"{f' x {bands} bands' if bands > 1 else ''}), "
            f"but TileOffsets has only {len(offsets)} entries"
        )
    if len(byte_counts) < expected:
        raise ValueError(
            f"Malformed TIFF: declared tile grid requires {expected} tile "
            f"byte counts, but TileByteCounts has only {len(byte_counts)} "
            f"entries"
        )


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


def _is_overview_or_full_res(ifd: IFD) -> bool:
    """Return True if *ifd* is the full-resolution image or an overview.

    NewSubfileType (tag 254) is a bit field per TIFF 6.0:

    * bit 0 (value 1) -- reduced-resolution version of another image (overview)
    * bit 1 (value 2) -- single page of a multi-page document
    * bit 2 (value 4) -- transparency mask

    The full-resolution IFD has ``NewSubfileType=0``. We accept it plus
    any IFD that is an overview *and* not a mask. Pages and any future
    flag combinations get filtered out so ``overview_level`` indexes the
    pyramid only.
    """
    st = ifd.subfile_type
    if st & 4:
        return False  # transparency mask (or overview-of-mask, st=5)
    return st == 0 or (st & 1) != 0


def select_overview_ifd(ifds: list[IFD], overview_level: int | None) -> IFD:
    """Pick the IFD for a requested overview level, skipping non-pyramid IFDs.

    Some COG variants (notably GDAL with internal masks) interleave
    transparency-mask IFDs (NewSubfileType bit 2 set) with overview IFDs.
    Multi-page TIFFs additionally carry page IFDs (bit 1 set). Indexing the
    raw IFD list by ``overview_level`` returns the wrong layer in either
    case. This helper builds a filtered list of full-resolution and
    overview IFDs only, and indexes into that.

    ``overview_level=0`` (or ``None``) returns the full-resolution IFD;
    ``overview_level=1`` returns the first overview, and so on.

    Parameters
    ----------
    ifds : list[IFD]
        All IFDs as parsed from the file.
    overview_level : int or None
        Which overview to return. ``None`` is treated as ``0``.

    Returns
    -------
    IFD

    Raises
    ------
    ValueError
        If ``ifds`` is empty, or if ``overview_level`` exceeds the number
        of pyramid IFDs in the file.
    """
    if not ifds:
        raise ValueError("No IFDs found in TIFF file")

    filtered = [ifd for ifd in ifds if _is_overview_or_full_res(ifd)]
    if not filtered:
        raise ValueError(
            "TIFF file contains no full-resolution or overview IFDs "
            "(every IFD is a mask, page, or other non-pyramid layer)")

    level = 0 if overview_level is None else overview_level
    if level < 0:
        raise ValueError(f"overview_level must be >= 0, got {level}")
    if level >= len(filtered):
        n_overviews = len(filtered) - 1
        n_skipped = len(ifds) - len(filtered)
        raise ValueError(
            f"overview_level={level} is out of range: TIFF has "
            f"{len(filtered)} pyramid IFDs (1 full-resolution + "
            f"{n_overviews} overview{'s' if n_overviews != 1 else ''}"
            f"{f', plus {n_skipped} non-pyramid IFD' if n_skipped else ''}"
            f"{'s' if n_skipped > 1 else ''}). Valid overview_level values "
            f"are 0..{len(filtered) - 1}.")

    return filtered[level]


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
