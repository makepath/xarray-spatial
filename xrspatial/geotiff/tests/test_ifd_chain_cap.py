"""Tests for the IFD-chain length cap in parse_all_ifds (security S3).

A crafted TIFF can chain millions of distinct IFD offsets via
``next_ifd_offset``; the cycle-detection ``seen`` set in
``parse_all_ifds`` won't catch those because every offset is unique.
``MAX_IFDS`` bounds the chain length to keep memory predictable on
untrusted input.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._header import (
    MAX_IFDS,
    TAG_IMAGE_LENGTH,
    TAG_IMAGE_WIDTH,
    parse_all_ifds,
    parse_header,
)


def _build_chained_ifd_bytes(n_ifds: int, big_endian: bool = False) -> bytes:
    """Build a classic TIFF whose IFD chain has exactly ``n_ifds`` IFDs.

    Each IFD carries a single tag (ImageWidth) so the parser accepts it,
    and points at the next IFD via the trailing next-IFD offset. The
    final IFD has next-pointer 0 (chain terminates cleanly), which means
    ``parse_all_ifds`` would happily walk all ``n_ifds`` of them in the
    absence of the cap.
    """
    bo = '>' if big_endian else '<'
    bom = b'MM' if big_endian else b'II'

    # Each classic-TIFF IFD here is:
    #   2 bytes num_entries
    #   12 bytes per entry (1 entry)
    #   4 bytes next-IFD pointer
    # = 18 bytes
    ifd_size = 18

    # Header is 8 bytes, then IFDs back-to-back.
    out = bytearray()
    out.extend(bom)
    out.extend(struct.pack(f'{bo}H', 42))
    first_ifd_offset = 8
    out.extend(struct.pack(f'{bo}I', first_ifd_offset))

    for i in range(n_ifds):
        next_offset = first_ifd_offset + (i + 1) * ifd_size
        if i == n_ifds - 1:
            next_offset = 0  # terminate chain cleanly
        out.extend(struct.pack(f'{bo}H', 1))  # num_entries
        # ImageWidth, type=LONG (4), count=1, value=i+1 inline
        out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 4, 1))
        out.extend(struct.pack(f'{bo}I', i + 1))
        out.extend(struct.pack(f'{bo}I', next_offset))

    return bytes(out)


class TestIFDChainCap:

    def test_ifd_chain_at_limit_rejected(self):
        """A chain well past MAX_IFDS must raise, not silently grow."""
        data = _build_chained_ifd_bytes(MAX_IFDS + 50)
        header = parse_header(data)
        with pytest.raises(ValueError, match=str(MAX_IFDS)):
            parse_all_ifds(data, header)

    def test_chain_at_boundary_passes(self):
        """Exactly MAX_IFDS IFDs is allowed; MAX_IFDS + 1 is rejected.

        Convention: we raise once ``len(ifds) >= MAX_IFDS`` after appending,
        so a chain of length exactly MAX_IFDS triggers the error and
        MAX_IFDS - 1 is the largest accepted chain.
        """
        # MAX_IFDS - 1 IFDs: passes, returns all of them.
        data_under = _build_chained_ifd_bytes(MAX_IFDS - 1)
        header_under = parse_header(data_under)
        ifds_under = parse_all_ifds(data_under, header_under)
        assert len(ifds_under) == MAX_IFDS - 1

        # MAX_IFDS IFDs: rejected (cap hit on the MAX_IFDS-th append).
        data_at = _build_chained_ifd_bytes(MAX_IFDS)
        header_at = parse_header(data_at)
        with pytest.raises(ValueError, match=str(MAX_IFDS)):
            parse_all_ifds(data_at, header_at)

    def test_error_message_mentions_dos_and_limit(self):
        data = _build_chained_ifd_bytes(MAX_IFDS + 5)
        header = parse_header(data)
        with pytest.raises(ValueError) as excinfo:
            parse_all_ifds(data, header)
        msg = str(excinfo.value)
        assert "MAX_IFDS" in msg
        assert str(MAX_IFDS) in msg
        # Threat-model language so operators see why it tripped.
        assert "denial-of-service" in msg or "malformed" in msg

    def test_short_chain_passes(self):
        """A small handful of IFDs (typical pyramid depth) parses fine."""
        data = _build_chained_ifd_bytes(8)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) == 8
        # Tags survived: each IFD's ImageWidth equals its index + 1.
        for i, ifd in enumerate(ifds):
            assert ifd.width == i + 1

    def test_legitimate_cog_with_overviews_passes(self, tmp_path):
        """A real COG with several overview levels parses fine.

        Real-world COGs have <30 IFDs even with many overview levels and
        per-band masks; the cap should never get in their way.
        """
        # 256 x 256 array with explicit overview levels triggers a small
        # pyramid in the writer. With levels [2, 4, 8] we get full + 3
        # overviews = 4 IFDs.
        arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        path = str(tmp_path / 'real_cog.tif')
        to_geotiff(arr, path, compression='deflate', tiled=True,
                   tile_size=64, cog=True, overview_levels=[2, 4, 8])

        with open(path, 'rb') as f:
            data = f.read()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert 1 < len(ifds) < MAX_IFDS
        assert len(ifds) <= 16  # well under the cap


class TestIFDChainCapBigEndian:
    """Same cap, but on a big-endian file."""

    def test_big_endian_chain_rejected(self):
        data = _build_chained_ifd_bytes(MAX_IFDS + 10, big_endian=True)
        header = parse_header(data)
        assert header.byte_order == '>'
        with pytest.raises(ValueError, match=str(MAX_IFDS)):
            parse_all_ifds(data, header)
