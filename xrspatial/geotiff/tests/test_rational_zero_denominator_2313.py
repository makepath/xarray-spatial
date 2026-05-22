"""Regression tests for issue #2313.

A RATIONAL or SRATIONAL tag with a zero denominator is malformed by
the TIFF spec. The reader used to coerce it to 0.0 silently, which
let corrupted `XResolution` / `YResolution` metadata round-trip as if
the file were valid. After the fix the reader raises `ValueError`
with the tag name and the denominator in the message.
"""
from __future__ import annotations

import io
import struct

import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._dtypes import RATIONAL, SRATIONAL
from xrspatial.geotiff._header import (TAG_X_RESOLUTION, TAG_Y_RESOLUTION,
                                       parse_all_ifds, parse_header)


def _build_tiff_with_rational_xres(numerator: int, denominator: int,
                                   *, srational: bool = False) -> bytes:
    """Build a minimal little-endian TIFF whose XResolution is a single
    RATIONAL (or SRATIONAL) pointing at (numerator, denominator).

    The file lays out:
      - 8-byte TIFF header
      - IFD with the minimum tags a parser will accept plus
        XResolution; XResolution is 8 bytes so it lives in an overflow
        block after the IFD entry table
      - 1 byte of strip data so StripOffsets / StripByteCounts are
        consistent
    """
    bo = '<'
    out = bytearray()
    # Header: little-endian, classic TIFF, first IFD at offset 8.
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', 8))

    tags = [
        # (tag, type_id, count, raw_bytes)
        (256, 3, 1, struct.pack(f'{bo}H', 4)),    # ImageWidth
        (257, 3, 1, struct.pack(f'{bo}H', 4)),    # ImageLength
        (258, 3, 1, struct.pack(f'{bo}H', 8)),    # BitsPerSample
        (259, 3, 1, struct.pack(f'{bo}H', 1)),    # Compression
        (262, 3, 1, struct.pack(f'{bo}H', 1)),    # PhotometricInterpretation
        (273, 4, 1, b'\x00\x00\x00\x00'),         # StripOffsets (patched)
        (277, 3, 1, struct.pack(f'{bo}H', 1)),    # SamplesPerPixel
        (278, 3, 1, struct.pack(f'{bo}H', 4)),    # RowsPerStrip
        (279, 4, 1, struct.pack(f'{bo}I', 16)),   # StripByteCounts
        # XResolution / YResolution as RATIONAL or SRATIONAL.
        (282, SRATIONAL if srational else RATIONAL, 1,
         struct.pack(f'{bo}ii' if srational else f'{bo}II',
                     numerator, denominator)),
        (283, SRATIONAL if srational else RATIONAL, 1,
         struct.pack(f'{bo}ii' if srational else f'{bo}II', 72, 1)),
    ]
    tags.sort(key=lambda t: t[0])

    num_entries = len(tags)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    # Lay out overflow values for any tag whose raw bytes exceed 4.
    overflow_buf = bytearray()
    tag_offsets: dict[int, int | None] = {}
    for tag, typ, count, raw in tags:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    pixel_data_start = overflow_start + len(overflow_buf)
    # Patch StripOffsets to point at the actual pixel block.
    patched = []
    for tag, typ, count, raw in tags:
        if tag == 273:
            patched.append((tag, typ, count,
                            struct.pack(f'{bo}I', pixel_data_start)))
        else:
            patched.append((tag, typ, count, raw))
    tags = patched

    # IFD entries.
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tags:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))
    # Next IFD pointer (none).
    out.extend(struct.pack(f'{bo}I', 0))
    # Overflow block.
    out.extend(overflow_buf)
    # Pad to pixel_data_start.
    while len(out) < pixel_data_start:
        out.append(0)
    # 16 bytes of pixel payload (matches StripByteCounts above).
    out.extend(b'\x00' * 16)

    return bytes(out)


class TestRationalZeroDenominator:
    """Issue #2313: zero-denominator rationals must fail loudly."""

    def test_rational_zero_denominator_surfaces_from_parse_all_ifds(self):
        data = _build_tiff_with_rational_xres(72, 0)
        header = parse_header(data)
        with pytest.raises(ValueError, match="XResolution"):
            parse_all_ifds(data, header)

    def test_rational_zero_denominator_message_includes_denominator(self):
        data = _build_tiff_with_rational_xres(150, 0)
        header = parse_header(data)
        with pytest.raises(ValueError) as exc:
            parse_all_ifds(data, header)
        message = str(exc.value)
        assert "Malformed RATIONAL" in message
        assert "XResolution" in message
        assert "denominator=0" in message
        assert "numerator=150" in message

    def test_srational_zero_denominator_surfaces_from_parse_all_ifds(self):
        data = _build_tiff_with_rational_xres(-5, 0, srational=True)
        header = parse_header(data)
        with pytest.raises(ValueError, match="Malformed SRATIONAL"):
            parse_all_ifds(data, header)

    def test_rational_zero_denominator_fails_open_geotiff(self):
        # The public read entry point should fail loudly too, not just
        # the low-level header parser.
        data = _build_tiff_with_rational_xres(72, 0)
        buf = io.BytesIO(data)
        with pytest.raises(ValueError, match="XResolution"):
            open_geotiff(buf)

    def test_yresolution_zero_denominator_named_in_error(self):
        # Same path, different tag. Build a TIFF where only YResolution
        # is malformed by swapping the values via direct construction.
        bo = '<'
        out = bytearray()
        out.extend(b'II')
        out.extend(struct.pack(f'{bo}H', 42))
        out.extend(struct.pack(f'{bo}I', 8))

        tags = [
            (256, 3, 1, struct.pack(f'{bo}H', 4)),
            (257, 3, 1, struct.pack(f'{bo}H', 4)),
            (258, 3, 1, struct.pack(f'{bo}H', 8)),
            (259, 3, 1, struct.pack(f'{bo}H', 1)),
            (262, 3, 1, struct.pack(f'{bo}H', 1)),
            (273, 4, 1, b'\x00\x00\x00\x00'),
            (277, 3, 1, struct.pack(f'{bo}H', 1)),
            (278, 3, 1, struct.pack(f'{bo}H', 4)),
            (279, 4, 1, struct.pack(f'{bo}I', 16)),
            (282, RATIONAL, 1, struct.pack(f'{bo}II', 72, 1)),
            # YResolution with zero denominator.
            (283, RATIONAL, 1, struct.pack(f'{bo}II', 72, 0)),
        ]
        tags.sort(key=lambda t: t[0])
        num_entries = len(tags)
        ifd_start = 8
        ifd_size = 2 + 12 * num_entries + 4
        overflow_start = ifd_start + ifd_size
        overflow_buf = bytearray()
        tag_offsets: dict[int, int | None] = {}
        for tag, typ, count, raw in tags:
            if len(raw) > 4:
                tag_offsets[tag] = len(overflow_buf)
                overflow_buf.extend(raw)
                if len(overflow_buf) % 2:
                    overflow_buf.append(0)
            else:
                tag_offsets[tag] = None
        pixel_data_start = overflow_start + len(overflow_buf)
        patched = []
        for tag, typ, count, raw in tags:
            if tag == 273:
                patched.append((tag, typ, count,
                                struct.pack(f'{bo}I', pixel_data_start)))
            else:
                patched.append((tag, typ, count, raw))
        tags = patched
        out.extend(struct.pack(f'{bo}H', num_entries))
        for tag, typ, count, raw in tags:
            out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
            if len(raw) <= 4:
                out.extend(raw.ljust(4, b'\x00'))
            else:
                out.extend(struct.pack(f'{bo}I',
                                       overflow_start + tag_offsets[tag]))
        out.extend(struct.pack(f'{bo}I', 0))
        out.extend(overflow_buf)
        while len(out) < pixel_data_start:
            out.append(0)
        out.extend(b'\x00' * 16)

        data = bytes(out)
        header = parse_header(data)
        with pytest.raises(ValueError, match="YResolution"):
            parse_all_ifds(data, header)

    # Sanity check: the helpers we use to assert tag ids actually exist.
    def test_tag_constants_present(self):
        assert TAG_X_RESOLUTION == 282
        assert TAG_Y_RESOLUTION == 283
