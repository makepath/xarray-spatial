"""Tests for parse_all_ifds rejecting malformed IFD chain offsets (#1863).

A TIFF whose IFD-chain pointer (either the file header's
``first_ifd_offset`` or any IFD's ``next_ifd_offset``) lands past EOF is
malformed. ``parse_all_ifds`` used to silently ``break`` out of the walk
in that case, returning a truncated list and masking corrupt overview
metadata. It now raises ``ValueError`` so callers see the corruption,
matching the convention used by the ``MAX_IFDS`` guard a few lines down.
"""
from __future__ import annotations

import struct

import pytest

from xrspatial.geotiff._header import (
    TAG_IMAGE_WIDTH,
    TIFFHeader,
    parse_all_ifds,
    parse_header,
)


def _build_single_ifd_with_next_offset_1863(next_offset: int,
                                            big_endian: bool = False) -> bytes:
    """Build a classic TIFF with one valid IFD that points to ``next_offset``.

    The single IFD carries an ImageWidth tag so it parses cleanly. Its
    next-IFD pointer is set to ``next_offset`` regardless of whether that
    is a valid in-file location, so callers can construct a chain that
    points past EOF.
    """
    bo = '>' if big_endian else '<'
    bom = b'MM' if big_endian else b'II'

    out = bytearray()
    out.extend(bom)
    out.extend(struct.pack(f'{bo}H', 42))
    first_ifd_offset = 8
    out.extend(struct.pack(f'{bo}I', first_ifd_offset))

    # One IFD: num_entries=1, ImageWidth tag inline, next_ifd_offset.
    out.extend(struct.pack(f'{bo}H', 1))
    out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 4, 1))
    out.extend(struct.pack(f'{bo}I', 42))  # width value, arbitrary
    out.extend(struct.pack(f'{bo}I', next_offset))

    return bytes(out)


class TestParseAllIFDsMalformedChain1863:

    def test_next_ifd_offset_past_eof_raises(self):
        """A chain whose ``next_ifd_offset`` points beyond EOF raises."""
        buf_1863 = _build_single_ifd_with_next_offset_1863(
            next_offset=0xDEADBEEF
        )
        header_1863 = parse_header(buf_1863)
        with pytest.raises(ValueError) as excinfo:
            parse_all_ifds(buf_1863, header_1863)
        msg = str(excinfo.value)
        # Error should mention the bad offset and the file length so an
        # operator can diagnose truncation vs. corruption.
        assert str(0xDEADBEEF) in msg
        assert "malformed" in msg

    def test_next_ifd_offset_equals_file_length_raises(self):
        """Boundary: pointer one byte past EOF (offset == len(data))."""
        buf_1863 = _build_single_ifd_with_next_offset_1863(next_offset=0)
        # Replace the trailing next-IFD pointer with exactly len(buf)
        # (i.e. pointing one byte past the buffer end).
        tail_offset = len(buf_1863) - 4
        eof_offset = len(buf_1863)
        patched = bytearray(buf_1863)
        struct.pack_into('<I', patched, tail_offset, eof_offset)
        data_1863 = bytes(patched)

        header_1863 = parse_header(data_1863)
        with pytest.raises(ValueError, match="malformed"):
            parse_all_ifds(data_1863, header_1863)

    def test_first_ifd_offset_past_eof_raises(self):
        """A header whose ``first_ifd_offset`` is past EOF raises.

        We build a header-only buffer pointing at offset 0xCAFEBABE,
        which is well beyond the 8 bytes of header we actually wrote.
        """
        bo = '<'
        bom = b'II'
        first_ifd_offset = 0xCAFEBABE
        buf_1863 = bytearray()
        buf_1863.extend(bom)
        buf_1863.extend(struct.pack(f'{bo}H', 42))
        buf_1863.extend(struct.pack(f'{bo}I', first_ifd_offset))
        data_1863 = bytes(buf_1863)

        header_1863 = parse_header(data_1863)
        assert header_1863.first_ifd_offset == first_ifd_offset
        with pytest.raises(ValueError) as excinfo:
            parse_all_ifds(data_1863, header_1863)
        msg = str(excinfo.value)
        assert str(first_ifd_offset) in msg

    def test_first_ifd_offset_past_eof_raises_synthetic_header(self):
        """Same as above but with a synthesised ``TIFFHeader`` directly.

        Catches the case where someone bypasses ``parse_header`` and
        passes a fabricated header to ``parse_all_ifds``.
        """
        data_1863 = b'II' + struct.pack('<H', 42) + struct.pack('<I', 8)
        # Header claims first IFD at byte 9999, file is 8 bytes.
        header_1863 = TIFFHeader(
            byte_order='<', is_bigtiff=False, first_ifd_offset=9999
        )
        with pytest.raises(ValueError, match="9999"):
            parse_all_ifds(data_1863, header_1863)

    def test_big_endian_next_offset_past_eof_raises(self):
        """The same guard fires for big-endian files."""
        buf_1863 = _build_single_ifd_with_next_offset_1863(
            next_offset=0xDEADBEEF, big_endian=True
        )
        header_1863 = parse_header(buf_1863)
        assert header_1863.byte_order == '>'
        with pytest.raises(ValueError, match="malformed"):
            parse_all_ifds(buf_1863, header_1863)

    def test_valid_chain_terminator_still_parses(self):
        """A normal ``next_ifd_offset=0`` still terminates cleanly."""
        buf_1863 = _build_single_ifd_with_next_offset_1863(next_offset=0)
        header_1863 = parse_header(buf_1863)
        ifds = parse_all_ifds(buf_1863, header_1863)
        assert len(ifds) == 1
        assert ifds[0].width == 42
