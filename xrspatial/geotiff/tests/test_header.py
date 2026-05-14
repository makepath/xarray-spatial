"""Tests for TIFF header and IFD parsing."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from xrspatial.geotiff._dtypes import RATIONAL, SRATIONAL
from xrspatial.geotiff._header import (
    IFD,
    TIFFHeader,
    _read_value,
    parse_all_ifds,
    parse_header,
    parse_ifd,
    TAG_IMAGE_WIDTH,
    TAG_IMAGE_LENGTH,
    TAG_BITS_PER_SAMPLE,
    TAG_COMPRESSION,
)
from .conftest import make_minimal_tiff


class TestParseHeader:
    def test_little_endian(self):
        data = make_minimal_tiff(4, 4)
        header = parse_header(data)
        assert header.byte_order == '<'
        assert not header.is_bigtiff
        assert header.first_ifd_offset == 8

    def test_big_endian(self):
        data = make_minimal_tiff(4, 4, big_endian=True)
        header = parse_header(data)
        assert header.byte_order == '>'
        assert not header.is_bigtiff

    def test_invalid_bom(self):
        with pytest.raises(ValueError, match="Invalid TIFF byte order"):
            parse_header(b'XX\x00\x2a\x00\x00\x00\x08')

    def test_invalid_magic(self):
        with pytest.raises(ValueError, match="Invalid TIFF magic"):
            parse_header(b'II\x00\x99\x00\x00\x00\x08')

    def test_too_short(self):
        with pytest.raises(ValueError, match="Not enough data"):
            parse_header(b'II\x00')


class TestParseIFD:
    def test_basic_tags(self):
        data = make_minimal_tiff(10, 20, np.dtype('uint16'))
        header = parse_header(data)
        ifd = parse_ifd(data, header.first_ifd_offset, header)

        assert ifd.width == 10
        assert ifd.height == 20
        assert ifd.bits_per_sample == 16
        assert ifd.compression == 1  # uncompressed
        assert ifd.samples_per_pixel == 1

    def test_float32_tags(self):
        data = make_minimal_tiff(8, 8, np.dtype('float32'))
        header = parse_header(data)
        ifd = parse_ifd(data, header.first_ifd_offset, header)

        assert ifd.bits_per_sample == 32
        assert ifd.sample_format == 3  # float

    def test_strip_layout(self):
        data = make_minimal_tiff(4, 4)
        header = parse_header(data)
        ifd = parse_ifd(data, header.first_ifd_offset, header)

        assert not ifd.is_tiled
        assert ifd.strip_offsets is not None
        assert ifd.strip_byte_counts is not None

    def test_next_ifd_zero(self):
        data = make_minimal_tiff(4, 4)
        header = parse_header(data)
        ifd = parse_ifd(data, header.first_ifd_offset, header)
        assert ifd.next_ifd_offset == 0


class TestParseAllIFDs:
    def test_single_ifd(self):
        data = make_minimal_tiff(4, 4)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) == 1
        assert ifds[0].width == 4

    def test_tiled_ifd(self):
        data = make_minimal_tiff(
            8, 8, np.dtype('float32'),
            pixel_data=np.arange(64, dtype=np.float32).reshape(8, 8),
            tiled=True, tile_size=4,
        )
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) == 1
        assert ifds[0].is_tiled
        assert ifds[0].tile_width == 4
        assert ifds[0].tile_height == 4


class TestIFDProperties:
    def test_nodata_str(self):
        ifd = IFD()
        assert ifd.nodata_str is None

    def test_defaults(self):
        ifd = IFD()
        assert ifd.width == 0
        assert ifd.height == 0
        assert ifd.bits_per_sample == 8
        assert ifd.compression == 1
        assert ifd.predictor == 1
        assert ifd.samples_per_pixel == 1
        assert ifd.photometric == 1
        assert ifd.planar_config == 1
        assert not ifd.is_tiled


class TestPlanarConfigValidation:
    """PlanarConfiguration must be 1 (Chunky) or 2 (Planar) per TIFF 6.0.

    Issue #1870: ``planar_config`` previously returned the raw tag value
    unchecked, so a malformed file with e.g. ``PlanarConfiguration=0``
    decoded under an assumed chunky layout instead of being rejected.
    """

    def _ifd_with_planar(self, value) -> IFD:
        from xrspatial.geotiff._header import IFDEntry, TAG_PLANAR_CONFIG
        ifd = IFD()
        ifd.entries[TAG_PLANAR_CONFIG] = IFDEntry(
            tag=TAG_PLANAR_CONFIG, type_id=3, count=1, value=value,
        )
        return ifd

    def test_chunky_accepted(self):
        assert self._ifd_with_planar(1).planar_config == 1

    def test_planar_accepted(self):
        assert self._ifd_with_planar(2).planar_config == 2

    def test_zero_rejected(self):
        with pytest.raises(ValueError, match=r"Invalid PlanarConfiguration"):
            _ = self._ifd_with_planar(0).planar_config

    def test_three_rejected(self):
        with pytest.raises(ValueError, match=r"Invalid PlanarConfiguration"):
            _ = self._ifd_with_planar(3).planar_config

    def test_large_value_rejected(self):
        with pytest.raises(ValueError, match=r"Invalid PlanarConfiguration"):
            _ = self._ifd_with_planar(255).planar_config

    def test_missing_tag_defaults_to_one(self):
        # No TAG_PLANAR_CONFIG entry => default 1 (chunky) per spec.
        assert IFD().planar_config == 1

    def test_tuple_value_uniform_one_accepted(self):
        # Some encoders write count>1 even though the tag is scalar.
        assert self._ifd_with_planar((1,)).planar_config == 1

    def test_tuple_value_invalid_rejected(self):
        with pytest.raises(ValueError, match=r"Invalid PlanarConfiguration"):
            _ = self._ifd_with_planar((7,)).planar_config


class TestIFDChainLoop:
    """Verify parse_all_ifds bails out on a malicious IFD chain cycle.

    Issue #1482 (T-2): the parser keeps a ``seen`` set of offsets and
    breaks the loop when an IFD's next-pointer points back to an offset
    already visited.  Without that guard, a crafted file with IFD2
    pointing back at IFD1 would loop forever.
    """

    def _build_two_ifd_loop(self) -> bytes:
        """Build a TIFF where IFD #2's next-pointer points back at IFD #1."""
        bo = '<'
        # Header: II, magic 42, first-IFD offset
        out = bytearray()
        out.extend(b'II')
        out.extend(struct.pack(f'{bo}H', 42))

        ifd1_offset = 8
        # IFD layout: 1 entry + next-IFD pointer = 2 + 12 + 4 = 18 bytes
        ifd2_offset = ifd1_offset + 18
        out.extend(struct.pack(f'{bo}I', ifd1_offset))

        # IFD #1: one entry (ImageWidth=1), next = ifd2_offset
        out.extend(struct.pack(f'{bo}H', 1))  # num_entries
        out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 3, 1))
        out.extend(struct.pack(f'{bo}I', 1))  # value (inline padded)
        out.extend(struct.pack(f'{bo}I', ifd2_offset))  # next-IFD

        # IFD #2: one entry (ImageWidth=2), next = ifd1_offset (cycle!)
        out.extend(struct.pack(f'{bo}H', 1))
        out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 3, 1))
        out.extend(struct.pack(f'{bo}I', 2))
        out.extend(struct.pack(f'{bo}I', ifd1_offset))  # cycle back

        return bytes(out)

    def test_cycle_does_not_infinite_loop(self):
        data = self._build_two_ifd_loop()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)

        # Each unique offset is parsed exactly once.  Cycle detection
        # must stop us at IFD #2 even though its next-pointer is non-zero.
        assert len(ifds) == 2
        # Values should match what we wrote, in order.
        assert ifds[0].width == 1
        assert ifds[1].width == 2

    def test_cycle_completes_quickly(self):
        """Parsing must terminate without exhausting the buffer."""
        data = self._build_two_ifd_loop()
        header = parse_header(data)
        # If the seen-set guard regresses this would hang; pytest-timeout
        # would catch it but the bounded len() check below is enough.
        ifds = parse_all_ifds(data, header)
        assert len(ifds) <= 4  # generous upper bound -- we expect 2

    def test_self_loop(self):
        """An IFD whose next-pointer is its own offset terminates."""
        bo = '<'
        out = bytearray()
        out.extend(b'II')
        out.extend(struct.pack(f'{bo}H', 42))
        ifd_offset = 8
        out.extend(struct.pack(f'{bo}I', ifd_offset))
        out.extend(struct.pack(f'{bo}H', 1))
        out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 3, 1))
        out.extend(struct.pack(f'{bo}I', 7))
        out.extend(struct.pack(f'{bo}I', ifd_offset))  # points to self
        ifds = parse_all_ifds(bytes(out), parse_header(bytes(out)))
        assert len(ifds) == 1
        assert ifds[0].width == 7


class TestReadValueRationals:
    """T-8 coverage for RATIONAL / SRATIONAL edge cases in _read_value."""

    def test_rational_denominator_zero_returns_zero(self):
        # numerator=5, denominator=0 -- by convention return 0.0
        buf = struct.pack('<II', 5, 0)
        result = _read_value(buf, 0, RATIONAL, 1, '<')
        assert result == 0.0

    def test_srational_denominator_zero_returns_zero(self):
        buf = struct.pack('<ii', -3, 0)
        result = _read_value(buf, 0, SRATIONAL, 1, '<')
        assert result == 0.0

    def test_rational_normal_value(self):
        buf = struct.pack('<II', 22, 7)
        result = _read_value(buf, 0, RATIONAL, 1, '<')
        assert result == pytest.approx(22 / 7)

    def test_srational_negative_value(self):
        buf = struct.pack('<ii', -10, 4)
        result = _read_value(buf, 0, SRATIONAL, 1, '<')
        assert result == pytest.approx(-2.5)

    def test_rational_count_overflow_raises(self):
        # Buffer holds one RATIONAL (8 bytes) but count=4 demands 32.
        # struct.error is fine -- the point is that we don't silently
        # read past the buffer.
        buf = struct.pack('<II', 1, 1)
        with pytest.raises(struct.error):
            _read_value(buf, 0, RATIONAL, 4, '<')



class TestTruncatedIFD:
    """T-8: a truncated IFD entry buffer should fail cleanly, not crash silently."""

    def test_ifd_count_exceeds_buffer(self):
        bo = '<'
        out = bytearray()
        out.extend(b'II')
        out.extend(struct.pack(f'{bo}H', 42))
        out.extend(struct.pack(f'{bo}I', 8))  # first IFD at 8
        # Claim 5 entries but only provide 3 full ones (36 bytes < 60).
        out.extend(struct.pack(f'{bo}H', 5))
        for _ in range(3):
            out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 3, 1))
            out.extend(struct.pack(f'{bo}I', 1))
        # Stop here -- last 2 entries plus next-IFD pointer are missing.
        data = bytes(out)
        header = parse_header(data)
        # struct.error is acceptable; what we want is "raises something",
        # not infinite read or silent truncation to a half-parsed IFD.
        with pytest.raises((struct.error, ValueError)):
            parse_ifd(data, header.first_ifd_offset, header)


class TestBigTIFFEdges:
    """T-8: BigTIFF malformations."""

    def test_bigtiff_offset_size_not_eight(self):
        """BigTIFF with magic 43 but offset_size != 8 must raise."""
        # II, magic 43, offset_size=4 (wrong -- must be 8), pad, ifd offset
        bad = b'II' + struct.pack('<H', 43) + struct.pack('<H', 4) \
            + struct.pack('<H', 0) + struct.pack('<Q', 16)
        with pytest.raises(ValueError, match="BigTIFF offset size"):
            parse_header(bad)

    def test_bigtiff_truncated_header(self):
        """BigTIFF magic but file shorter than the 16-byte header."""
        # Need at least 8 bytes to get past the initial length check, then
        # parse_header should refuse because BigTIFF needs 16.
        short = b'II' + struct.pack('<H', 43) + struct.pack('<H', 8) \
            + struct.pack('<H', 0)  # 8 bytes total
        with pytest.raises(ValueError, match="BigTIFF"):
            parse_header(short)


class TestClassicTIFFLargeOffset:
    """T-8: classic TIFF stores 32-bit offsets; offsets > 4 GB don't fit.

    There's no way to actually express an offset > 4 GB in a classic
    TIFF (the field is uint32). What we want to verify is that pointing
    a classic-TIFF first-IFD at an offset beyond the buffer is rejected
    by ``parse_all_ifds`` with a clear error rather than silently
    yielding an empty list (which used to mask truncated files).
    See issue #1863.
    """

    def test_first_ifd_offset_past_buffer(self):
        bo = '<'
        # Header points to offset 0xFFFFFF00, well past any plausible
        # buffer. parse_all_ifds raises ValueError on chain offsets
        # beyond EOF (matching the MAX_IFDS convention).
        out = bytearray()
        out.extend(b'II')
        out.extend(struct.pack(f'{bo}H', 42))
        out.extend(struct.pack(f'{bo}I', 0xFFFFFF00))
        header = parse_header(bytes(out))
        with pytest.raises(ValueError, match="malformed"):
            parse_all_ifds(bytes(out), header)
