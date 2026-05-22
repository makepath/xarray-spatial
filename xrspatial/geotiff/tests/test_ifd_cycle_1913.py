"""Tests for IFD-chain cycle rejection in parse_all_ifds (#1913).

``parse_all_ifds`` already raises ``ValueError`` for two malformed-chain
conditions: an offset past EOF and a chain longer than ``MAX_IFDS``.
A cyclic chain (offset A -> offset B -> offset A) is just as malformed,
but the original loop condition (``offset not in seen``) exited
silently and returned a truncated list. The fix raises ``ValueError``
with matching ``file is malformed`` wording so the contract is
consistent across all three malformed-chain branches.

The regression-guard tests in this file overlap with
``test_ifd_chain_cap.py``; they live here so a future refactor of the
loop is forced to keep both behaviours intact.
"""
from __future__ import annotations

import struct

import pytest

from xrspatial.geotiff._header import MAX_IFDS, TAG_IMAGE_WIDTH, parse_all_ifds, parse_header


def _build_cyclic_ifd_bytes(big_endian: bool = False) -> bytes:
    """Build a classic TIFF whose IFD chain forms a cycle: A -> B -> A.

    Both IFDs carry a single ImageWidth tag so ``parse_ifd`` accepts
    them. The second IFD's next-pointer points back to the first IFD's
    offset, closing the loop.
    """
    bo = '>' if big_endian else '<'
    bom = b'MM' if big_endian else b'II'

    out = bytearray()
    out.extend(bom)
    out.extend(struct.pack(f'{bo}H', 42))
    first = 8
    out.extend(struct.pack(f'{bo}I', first))

    # IFD A at offset 8 (length 18 bytes) -> IFD B at offset 26.
    ifd_a_off = 8
    ifd_b_off = 26
    out.extend(struct.pack(f'{bo}H', 1))
    out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 4, 1))
    out.extend(struct.pack(f'{bo}I', 1))
    out.extend(struct.pack(f'{bo}I', ifd_b_off))

    # IFD B -> back to IFD A (cycle).
    out.extend(struct.pack(f'{bo}H', 1))
    out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 4, 1))
    out.extend(struct.pack(f'{bo}I', 2))
    out.extend(struct.pack(f'{bo}I', ifd_a_off))

    return bytes(out)


def _build_self_cycle_ifd_bytes() -> bytes:
    """Build a TIFF whose first IFD points at itself (degenerate cycle)."""
    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack('<H', 42))
    first = 8
    out.extend(struct.pack('<I', first))
    # IFD at offset 8 with next-pointer back to itself.
    out.extend(struct.pack('<H', 1))
    out.extend(struct.pack('<HHI', TAG_IMAGE_WIDTH, 4, 1))
    out.extend(struct.pack('<I', 1))
    out.extend(struct.pack('<I', first))
    return bytes(out)


def _build_chained_ifd_bytes(n_ifds: int) -> bytes:
    """Build a non-cyclic chain of ``n_ifds`` IFDs (regression-guard helper)."""
    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack('<H', 42))
    first = 8
    out.extend(struct.pack('<I', first))
    ifd_size = 18
    for i in range(n_ifds):
        next_off = first + (i + 1) * ifd_size
        if i == n_ifds - 1:
            next_off = 0
        out.extend(struct.pack('<H', 1))
        out.extend(struct.pack('<HHI', TAG_IMAGE_WIDTH, 4, 1))
        out.extend(struct.pack('<I', i + 1))
        out.extend(struct.pack('<I', next_off))
    return bytes(out)


class TestIFDChainCycle:

    def test_two_ifd_cycle_rejected(self):
        """A -> B -> A must raise, not return [A, B] silently."""
        data = _build_cyclic_ifd_bytes()
        header = parse_header(data)
        with pytest.raises(ValueError, match="cycle"):
            parse_all_ifds(data, header)

    def test_self_cycle_rejected(self):
        """A single IFD pointing at itself is a cycle of length one."""
        data = _build_self_cycle_ifd_bytes()
        header = parse_header(data)
        with pytest.raises(ValueError, match="cycle"):
            parse_all_ifds(data, header)

    def test_cycle_error_message_mentions_offset_and_malformed(self):
        """Error message should name the repeat offset and call the file malformed."""
        data = _build_cyclic_ifd_bytes()
        header = parse_header(data)
        with pytest.raises(ValueError) as excinfo:
            parse_all_ifds(data, header)
        msg = str(excinfo.value)
        # The repeat offset (8) appears in the message.
        assert "8" in msg
        assert "malformed" in msg
        assert "cycle" in msg

    def test_big_endian_cycle_rejected(self):
        """Cycle detection works on big-endian files too."""
        data = _build_cyclic_ifd_bytes(big_endian=True)
        header = parse_header(data)
        assert header.byte_order == '>'
        with pytest.raises(ValueError, match="cycle"):
            parse_all_ifds(data, header)


class TestMalformedChainSiblingsStillRaise:
    """Regression guards for the two other malformed-chain branches.

    Reorganising the loop to handle cycles must not break the past-EOF
    and MAX_IFDS branches.
    """

    def test_offset_past_eof_still_raises(self):
        """A first-IFD offset past EOF must still raise ValueError."""
        out = bytearray()
        out.extend(b'II')
        out.extend(struct.pack('<H', 42))
        # Point at offset 9999, well past the 8-byte header.
        out.extend(struct.pack('<I', 9999))
        data = bytes(out)
        header = parse_header(data)
        with pytest.raises(ValueError, match="past end of file"):
            parse_all_ifds(data, header)

    def test_max_ifds_still_raises(self):
        """A chain longer than MAX_IFDS must still raise ValueError."""
        data = _build_chained_ifd_bytes(MAX_IFDS + 1)
        header = parse_header(data)
        with pytest.raises(ValueError, match=str(MAX_IFDS)):
            parse_all_ifds(data, header)


class TestNormalChainStillParses:

    def test_short_acyclic_chain_parses(self):
        """A normal multi-IFD chain still works post-fix."""
        data = _build_chained_ifd_bytes(5)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) == 5
        for i, ifd in enumerate(ifds):
            assert ifd.width == i + 1

    def test_single_ifd_chain_parses(self):
        """A one-IFD file (no next pointer) still parses."""
        data = _build_chained_ifd_bytes(1)
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        assert len(ifds) == 1
