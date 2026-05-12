"""Bounds checks for the IFD entry table itself (issue #1672).

The S2 work in #1661 added typed-error guards inside `parse_ifd` for the
*value area* of each entry (count cap, byte cap, value-range vs EOF). It
did not cover three earlier `struct.unpack_from` calls that read from the
entry table itself:

1. The `num_entries` field at the start of an IFD.
2. Each entry's tag/type/count fields inside the per-entry loop.
3. The next-IFD pointer immediately following the entry table.

A truncated or crafted buffer that lands on any of these three reads used
to escape with `struct.error`, which is not in the documented
`ValueError`/`TypeError` contract the Hypothesis fuzz tests rely on.

These tests build buffers that are exactly one byte short of the
respective read and assert the new `ValueError`s.
"""
from __future__ import annotations

import struct

import pytest

from xrspatial.geotiff._dtypes import LONG
from xrspatial.geotiff._header import (
    TIFFHeader,
    TAG_IMAGE_WIDTH,
    parse_header,
    parse_ifd,
)


# --- Classic TIFF ---


def _build_minimal_classic_ifd(num_entries: int = 1) -> bytes:
    """Build a complete little-endian classic TIFF with `num_entries` entries.

    The single entry is `TAG_IMAGE_WIDTH` (LONG, count=1, value=1) so the
    overall file is otherwise well-formed. The returned bytes can be
    sliced shorter to exercise the bounds checks.
    """
    bo = '<'
    ifd_offset = 8
    buf = bytearray()
    buf.extend(b'II')
    buf.extend(struct.pack(f'{bo}H', 42))
    buf.extend(struct.pack(f'{bo}I', ifd_offset))
    # IFD: num_entries (2) + n*12 + next_ifd (4)
    buf.extend(struct.pack(f'{bo}H', num_entries))
    for _ in range(num_entries):
        buf.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, LONG, 1))
        buf.extend(struct.pack(f'{bo}I', 1))  # inline value = 1
    buf.extend(struct.pack(f'{bo}I', 0))  # next IFD offset
    return bytes(buf)


def test_classic_num_entries_truncated_raises_valueerror():
    """Buffer ending mid-`num_entries` must raise ValueError, not struct.error."""
    full = _build_minimal_classic_ifd(num_entries=1)
    header = parse_header(full)
    # Slice off everything after offset 8 (the IFD start). `num_entries`
    # is a 2-byte field starting at offset 8, so a 9-byte buffer is one
    # byte short.
    truncated = full[:9]
    with pytest.raises(ValueError, match="num_entries"):
        parse_ifd(truncated, header.first_ifd_offset, header)


def test_classic_num_entries_zero_buffer_raises_valueerror():
    """Buffer ending exactly at the IFD offset must raise ValueError."""
    full = _build_minimal_classic_ifd(num_entries=1)
    header = parse_header(full)
    # 8-byte buffer: nothing at offset 8.
    truncated = full[:8]
    with pytest.raises(ValueError, match="num_entries"):
        parse_ifd(truncated, header.first_ifd_offset, header)


def test_classic_entry_table_truncated_raises_valueerror():
    """Buffer that cuts off mid-entry-table must raise ValueError."""
    full = _build_minimal_classic_ifd(num_entries=3)
    header = parse_header(full)
    # IFD layout: 2-byte count + 3 * 12-byte entries + 4-byte next-IFD
    # Entry table occupies offsets 10..45. Truncate one byte before the
    # entry table ends.
    entry_table_end = 8 + 2 + 3 * 12
    truncated = full[:entry_table_end - 1]
    with pytest.raises(ValueError, match="entry table"):
        parse_ifd(truncated, header.first_ifd_offset, header)


def test_classic_next_ifd_truncated_raises_valueerror():
    """Buffer that ends right at the entry table (no next-IFD) raises."""
    full = _build_minimal_classic_ifd(num_entries=1)
    header = parse_header(full)
    # IFD: 8 + 2 + 12 = 22 is the start of the next-IFD field; 22+4=26
    # is the end. Slice to 22 (no next-IFD bytes) and 25 (one short).
    next_ifd_pos = 8 + 2 + 12
    with pytest.raises(ValueError, match="next-IFD"):
        parse_ifd(full[:next_ifd_pos], header.first_ifd_offset, header)
    with pytest.raises(ValueError, match="next-IFD"):
        parse_ifd(full[:next_ifd_pos + 3], header.first_ifd_offset, header)


# --- BigTIFF ---


def _build_minimal_bigtiff_ifd(num_entries: int = 1) -> bytes:
    """Build a complete little-endian BigTIFF with `num_entries` entries."""
    bo = '<'
    ifd_offset = 16
    buf = bytearray()
    buf.extend(b'II')
    buf.extend(struct.pack(f'{bo}H', 43))
    buf.extend(struct.pack(f'{bo}H', 8))   # offset size
    buf.extend(b'\x00\x00')                # reserved
    buf.extend(struct.pack(f'{bo}Q', ifd_offset))
    # IFD: num_entries (8) + n*20 + next_ifd (8)
    buf.extend(struct.pack(f'{bo}Q', num_entries))
    for _ in range(num_entries):
        buf.extend(struct.pack(f'{bo}HH', TAG_IMAGE_WIDTH, LONG))
        buf.extend(struct.pack(f'{bo}Q', 1))      # count
        buf.extend(struct.pack(f'{bo}Q', 1))      # inline value = 1
    buf.extend(struct.pack(f'{bo}Q', 0))    # next IFD offset
    return bytes(buf)


def test_bigtiff_num_entries_truncated_raises_valueerror():
    """BigTIFF buffer ending mid-`num_entries` (8 bytes) raises ValueError."""
    full = _build_minimal_bigtiff_ifd(num_entries=1)
    header = parse_header(full)
    # `num_entries` is 8 bytes starting at offset 16. A 23-byte buffer is
    # one byte short.
    truncated = full[:23]
    with pytest.raises(ValueError, match="num_entries"):
        parse_ifd(truncated, header.first_ifd_offset, header)


def test_bigtiff_entry_table_truncated_raises_valueerror():
    """BigTIFF buffer that cuts off mid-entry-table raises ValueError."""
    full = _build_minimal_bigtiff_ifd(num_entries=2)
    header = parse_header(full)
    # IFD layout: 8-byte count + 2 * 20-byte entries + 8-byte next-IFD
    entry_table_end = 16 + 8 + 2 * 20
    truncated = full[:entry_table_end - 1]
    with pytest.raises(ValueError, match="entry table"):
        parse_ifd(truncated, header.first_ifd_offset, header)


def test_bigtiff_next_ifd_truncated_raises_valueerror():
    """BigTIFF buffer ending right at the entry table raises ValueError."""
    full = _build_minimal_bigtiff_ifd(num_entries=1)
    header = parse_header(full)
    next_ifd_pos = 16 + 8 + 20
    with pytest.raises(ValueError, match="next-IFD"):
        parse_ifd(full[:next_ifd_pos], header.first_ifd_offset, header)
    with pytest.raises(ValueError, match="next-IFD"):
        parse_ifd(full[:next_ifd_pos + 7], header.first_ifd_offset, header)


# --- Sanity: a complete buffer still parses ---


def test_classic_complete_buffer_parses_ok():
    """The build helper produces a well-formed file that parses."""
    data = _build_minimal_classic_ifd(num_entries=1)
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.width == 1
    assert ifd.next_ifd_offset == 0


def test_bigtiff_complete_buffer_parses_ok():
    """The BigTIFF helper produces a well-formed file that parses."""
    data = _build_minimal_bigtiff_ifd(num_entries=1)
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.width == 1
    assert ifd.next_ifd_offset == 0


# --- Regression: parse_ifd with a wildly out-of-bounds offset ---


def test_offset_past_eof_raises_valueerror():
    """Calling parse_ifd with offset > len(data) must not crash with struct.error."""
    full = _build_minimal_classic_ifd(num_entries=1)
    header = parse_header(full)
    # Walk completely off the end.
    with pytest.raises(ValueError, match="num_entries"):
        parse_ifd(full, len(full) + 100, header)
