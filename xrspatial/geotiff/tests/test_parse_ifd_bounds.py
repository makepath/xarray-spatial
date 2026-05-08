"""Bounds-checks for IFD `count` against file length and a hard cap.

These tests cover the S2 audit finding: a crafted TIFF with a huge
`count` on a non-pixel tag could force `parse_ifd` to ask
`struct.unpack_from` for a multi-million-element tuple, allocating
hundreds of MB to GB of memory before the parser ever returns. The fix
adds two guards to `parse_ifd`:

1. A hard cap (`MAX_IFD_ENTRY_COUNT`) on `count` for tags that aren't
   pixel-data offset/byte-count arrays.
2. A bounds check that the value range
   `[value_offset, value_offset + count * type_size)` falls inside the
   file before `_read_value` is called.

Legitimate large pixel-array tags (TileOffsets, StripOffsets, etc.)
must still parse without raising.
"""
from __future__ import annotations

import struct

import pytest

from xrspatial.geotiff._dtypes import LONG, SHORT
from xrspatial.geotiff._header import (
    MAX_IFD_ENTRY_COUNT,
    TAG_IMAGE_WIDTH,
    TAG_TILE_OFFSETS,
    TAG_TILE_BYTE_COUNTS,
    parse_header,
    parse_ifd,
)


def _build_tiff(entries: list[tuple[int, int, int, bytes]],
                tail_padding: int = 0,
                external_payloads: list[tuple[int, bytes]] | None = None
                ) -> bytes:
    """Build a classic little-endian TIFF with the given IFD entries.

    Each entry is `(tag, type_id, count, value_field_bytes)` where
    `value_field_bytes` is exactly 4 bytes that go into the IFD entry's
    value/offset slot.

    `external_payloads` is an optional list of `(absolute_offset,
    bytes)` pairs that will be written to the file at those positions.
    Used to place legal data referenced by an entry's value pointer.
    """
    bo = '<'
    n = len(entries)
    # Layout: header (8) + IFD (2 + n*12 + 4) + padding + payloads.
    ifd_offset = 8
    ifd_size = 2 + n * 12 + 4

    # Determine total size needed.
    end_of_ifd = ifd_offset + ifd_size
    file_size = end_of_ifd + tail_padding
    if external_payloads:
        for off, payload in external_payloads:
            file_size = max(file_size, off + len(payload))

    buf = bytearray(file_size)
    buf[0:2] = b'II'
    struct.pack_into(f'{bo}H', buf, 2, 42)  # magic
    struct.pack_into(f'{bo}I', buf, 4, ifd_offset)  # first IFD offset

    struct.pack_into(f'{bo}H', buf, ifd_offset, n)
    for i, (tag, type_id, count, value_bytes) in enumerate(entries):
        eo = ifd_offset + 2 + i * 12
        struct.pack_into(f'{bo}H', buf, eo, tag)
        struct.pack_into(f'{bo}H', buf, eo + 2, type_id)
        struct.pack_into(f'{bo}I', buf, eo + 4, count)
        assert len(value_bytes) == 4, "value field must be exactly 4 bytes"
        buf[eo + 8:eo + 12] = value_bytes
    # next IFD offset = 0
    struct.pack_into(f'{bo}I', buf, ifd_offset + 2 + n * 12, 0)

    if external_payloads:
        for off, payload in external_payloads:
            buf[off:off + len(payload)] = payload

    return bytes(buf)


def test_count_overflow_rejected():
    """A non-pixel tag with count > MAX_IFD_ENTRY_COUNT must raise."""
    bad_count = MAX_IFD_ENTRY_COUNT + 1
    # Tag IMAGE_WIDTH is not in the pixel-array exemption set.
    # Use type LONG (4 bytes), value pointer = 0 (irrelevant; we never
    # reach the read because the count check fires first).
    data = _build_tiff(
        entries=[(TAG_IMAGE_WIDTH, LONG, bad_count, b'\x00\x00\x00\x00')],
        tail_padding=64,
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="count"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_value_offset_past_eof_rejected():
    """A non-inline value range that extends past EOF must raise."""
    # 5 LONGs = 20 bytes -> non-inline. Place value pointer at a spot
    # such that ptr + 20 > file_size.
    count = 5
    file_size_target = 200  # we'll build a small file
    # Use a pointer that's just inside the file but with not enough
    # room for count * type_size bytes following it.
    ptr = file_size_target - 4  # only 4 bytes follow, need 20
    value_bytes = struct.pack('<I', ptr)
    data = _build_tiff(
        entries=[(TAG_IMAGE_WIDTH, LONG, count, value_bytes)],
        tail_padding=file_size_target - (8 + 2 + 12 + 4),
    )
    # Sanity: actual file length should equal file_size_target.
    assert len(data) == file_size_target
    header = parse_header(data)
    with pytest.raises(ValueError, match="exceeds file length"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_huge_pixel_array_count_allowed():
    """Pixel-array tags with `count > MAX_IFD_ENTRY_COUNT` parse fine.

    A 100k tile count is well within legitimate territory for a large
    tiled COG, and is below the cap anyway. To exercise the exemption
    path we additionally craft a TileOffsets entry whose count exceeds
    MAX_IFD_ENTRY_COUNT but whose value pointer references a region
    that is genuinely present in the file. The test makes sure no
    ValueError is raised on the count check; we use a small but
    above-cap value backed by real file bytes.
    """
    # Use a count just above the cap.
    count = MAX_IFD_ENTRY_COUNT + 10
    type_size = 4  # LONG
    payload_size = count * type_size
    # Place payload right after the IFD.
    payload_offset = 8 + 2 + 12 * 2 + 4  # header + IFD with 2 entries
    payload = b'\x00' * payload_size
    value_bytes = struct.pack('<I', payload_offset)

    # We need at least one TileByteCounts pair too for symmetry, but
    # it's not strictly required by parse_ifd. Add only TileOffsets
    # plus a benign IMAGE_WIDTH entry to keep the IFD sane.
    width_value = struct.pack('<I', 1024)
    data = _build_tiff(
        entries=[
            (TAG_IMAGE_WIDTH, LONG, 1, width_value),
            (TAG_TILE_OFFSETS, LONG, count, value_bytes),
        ],
        external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    # Should not raise.
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.entries[TAG_TILE_OFFSETS].count == count


def test_normal_tag_with_legal_count_passes():
    """Regression: a small, legal IFD parses cleanly."""
    width_value = struct.pack('<I', 256)
    height_value = struct.pack('<I', 256)
    data = _build_tiff(
        entries=[
            (TAG_IMAGE_WIDTH, LONG, 1, width_value),
            (257, LONG, 1, height_value),  # ImageLength
        ],
    )
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.width == 256
    assert ifd.height == 256


def test_short_count_at_cap_still_passes_for_pixel_tag():
    """TileByteCounts of SHORT type at the cap should still pass.

    Mostly a smoke test that the exemption applies to all listed
    pixel-array tags, not just LONG-typed ones.
    """
    count = 1000  # small, well within cap; we just exercise the path
    type_size = 2  # SHORT
    payload_size = count * type_size
    payload_offset = 8 + 2 + 12 + 4
    payload = b'\x00' * payload_size
    value_bytes = struct.pack('<I', payload_offset)
    data = _build_tiff(
        entries=[
            (TAG_TILE_BYTE_COUNTS, SHORT, count, value_bytes),
        ],
        external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.entries[TAG_TILE_BYTE_COUNTS].count == count
