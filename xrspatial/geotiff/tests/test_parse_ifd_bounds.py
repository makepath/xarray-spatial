"""Bounds-checks for IFD `count` and value range against file length.

These tests cover the S2 audit finding: a crafted TIFF with a huge
`count` on a non-pixel tag could force `parse_ifd` to ask
`struct.unpack_from` for a multi-million-element tuple, allocating
hundreds of MB to GB of memory before the parser ever returns. The fix
adds three guards to `parse_ifd`:

1. A hard cap (`MAX_IFD_ENTRY_COUNT`) on `count` for non-pixel tags.
2. A hard cap (`MAX_IFD_ENTRY_BYTES`) on `count * type_size` for
   non-pixel tags. Catches large-itemsize abuse (e.g. DOUBLE arrays)
   where the count alone may pass.
3. A bounds check that `value_offset + count * type_size <= len(data)`
   before `_read_value` is called, for any tag with a non-inline value.

Legitimate large pixel-array tags (TileOffsets, StripOffsets, etc.)
must still parse without raising.

Tests monkeypatch the cap constants down to small values where helpful,
so they exercise the limits with tiny in-memory payloads instead of
allocating hundreds of MB in CI.
"""
from __future__ import annotations

import struct

import pytest

from xrspatial.geotiff import _header
from xrspatial.geotiff._dtypes import DOUBLE, LONG, SHORT
from xrspatial.geotiff._header import (
    MAX_IFD_ENTRY_BYTES,
    MAX_IFD_ENTRY_COUNT,
    TAG_IMAGE_WIDTH,
    TAG_TILE_BYTE_COUNTS,
    TAG_TILE_OFFSETS,
    parse_header,
    parse_ifd,
)


def _build_tiff(
    entries: list[tuple[int, int, int, bytes]],
    tail_padding: int = 0,
    external_payloads: list[tuple[int, bytes]] | None = None,
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
    # value pointer = 0 is irrelevant; the count check fires first.
    data = _build_tiff(
        entries=[(TAG_IMAGE_WIDTH, LONG, bad_count, b'\x00\x00\x00\x00')],
        tail_padding=64,
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="count"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_byte_size_overflow_rejected_with_legal_count(monkeypatch):
    """A count under the count cap but bytes over the byte cap raises.

    Monkeypatches both caps to small values so the test exercises the
    byte-cap path independently of the count-cap path. With count=33
    and DOUBLE (8 bytes), bytes=264 > 256 (byte cap) while count < 100
    (count cap), so only the byte cap fires.
    """
    monkeypatch.setattr(_header, 'MAX_IFD_ENTRY_COUNT', 100)
    monkeypatch.setattr(_header, 'MAX_IFD_ENTRY_BYTES', 256)

    count = 33
    data = _build_tiff(
        entries=[(TAG_IMAGE_WIDTH, DOUBLE, count, b'\x00\x00\x00\x00')],
        tail_padding=64,
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="bytes exceeds MAX_IFD_ENTRY_BYTES"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_byte_cap_fires_at_default_values():
    """Default caps catch a realistic large-itemsize abuse.

    With default caps (count=100K, bytes=256KiB), a non-pixel DOUBLE
    tag with count=32_770 has bytes=262_160 > 262_144 (256KiB) but
    count is well under 100K. This proves the byte cap is independently
    useful at production values, not just under monkeypatch.
    """
    count = (MAX_IFD_ENTRY_BYTES // 8) + 2
    assert count < MAX_IFD_ENTRY_COUNT, (
        "test must exercise byte cap, not count cap"
    )
    data = _build_tiff(
        entries=[(TAG_IMAGE_WIDTH, DOUBLE, count, b'\x00\x00\x00\x00')],
        tail_padding=64,
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="bytes exceeds MAX_IFD_ENTRY_BYTES"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_value_offset_past_eof_rejected():
    """A non-inline value range that extends past EOF must raise."""
    # 5 LONGs = 20 bytes -> non-inline. Place value pointer at a spot
    # such that ptr + 20 > file_size.
    count = 5
    file_size_target = 200
    ptr = file_size_target - 4  # only 4 bytes follow, need 20
    value_bytes = struct.pack('<I', ptr)
    data = _build_tiff(
        entries=[(TAG_IMAGE_WIDTH, LONG, count, value_bytes)],
        tail_padding=file_size_target - (8 + 2 + 12 + 4),
    )
    assert len(data) == file_size_target
    header = parse_header(data)
    with pytest.raises(ValueError, match="exceeds file length"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_pixel_array_tag_exemption(monkeypatch):
    """Pixel-array tags bypass both caps; a tiny lowered cap proves it.

    Monkeypatches both caps to 10 so the test exercises the exemption
    without allocating large buffers. With non-pixel tags, count=11
    would raise; with TileOffsets it must pass.
    """
    monkeypatch.setattr(_header, 'MAX_IFD_ENTRY_COUNT', 10)
    monkeypatch.setattr(_header, 'MAX_IFD_ENTRY_BYTES', 16)

    count = 11
    type_size = 4  # LONG
    payload_size = count * type_size
    # Place payload right after the IFD with 2 entries.
    payload_offset = 8 + 2 + 12 * 2 + 4
    payload = b'\x00' * payload_size
    value_bytes = struct.pack('<I', payload_offset)

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


def test_count_cap_fires_for_non_pixel_tag_under_lowered_caps(monkeypatch):
    """Count cap fires before any allocation when lowered for the test."""
    monkeypatch.setattr(_header, 'MAX_IFD_ENTRY_COUNT', 10)

    bad_count = 11
    data = _build_tiff(
        entries=[(TAG_IMAGE_WIDTH, LONG, bad_count, b'\x00\x00\x00\x00')],
        tail_padding=64,
    )
    header = parse_header(data)
    with pytest.raises(ValueError,
                       match=r"count=11 exceeds MAX_IFD_ENTRY_COUNT=10"):
        parse_ifd(data, header.first_ifd_offset, header)


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
    """TileByteCounts of SHORT type still passes for pixel-array tags.

    Smoke test that the exemption applies regardless of element type.
    """
    count = 1000
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
