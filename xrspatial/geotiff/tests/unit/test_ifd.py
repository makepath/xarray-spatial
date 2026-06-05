"""Unit tests for the GeoTIFF IFD (Image File Directory) parser.

Every
test here exercises the IFD-chain / entry-table parser in isolation,
without standing up a full read pipeline. The sparse-block and
sparse-strip sections at the end stand up the public reader because
the sparse-IFD encoding (``TileByteCounts == 0`` / ``StripByteCounts
== 0``) is observable only through the read pipeline.

Sections, in failure-mode order:

1. ``parse_ifd`` entry-table bounds -- truncated ``num_entries``,
   truncated entry table, truncated next-IFD field, negative offsets,
   classic + BigTIFF.
2. ``parse_ifd`` entry-value bounds -- ``MAX_IFD_ENTRY_COUNT``,
   ``MAX_IFD_ENTRY_BYTES``, value-range past EOF, pixel-array
   exemptions.
2b. ``parse_ifd`` duplicate-tag rejection (issue #2483) -- TIFF 6.0
   forbids duplicate tag ids within one IFD; the parser must raise
   ``DuplicateIFDTagError`` instead of silently letting the last
   duplicate win.
3. ``parse_all_ifds`` chain length cap (``MAX_IFDS``) -- classic and
   big-endian, boundary, legitimate-COG sanity.
4. ``parse_all_ifds`` chain cycle detection -- A->B->A, self-cycle,
   error message shape.
5. ``parse_all_ifds`` malformed chain offsets -- ``first_ifd_offset``
   and ``next_ifd_offset`` past EOF, big-endian variant, valid
   terminator still parses.
6. Sparse blocks via the read pipeline -- tile + strip, with and
   without nodata, GPU path.
7. Sparse strips through the parallel-decode pipeline -- local file
   and HTTP COG, planar=1 and planar=2.
"""
from __future__ import annotations

import concurrent.futures
import http.server
import importlib.util
import socket
import struct
import threading
from unittest.mock import patch

import numpy as np
import pytest

from xrspatial.geotiff import DuplicateIFDTagError
from xrspatial.geotiff import _decode as _decode_mod
from xrspatial.geotiff import _header
from xrspatial.geotiff import _reader as _reader_mod
from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._dtypes import DOUBLE, LONG, SHORT
from xrspatial.geotiff._header import (MAX_IFD_ENTRY_BYTES, MAX_IFD_ENTRY_COUNT, MAX_IFDS,
                                       TAG_BITS_PER_SAMPLE, TAG_GEO_KEY_DIRECTORY, TAG_IMAGE_LENGTH,
                                       TAG_IMAGE_WIDTH, TAG_TILE_BYTE_COUNTS, TAG_TILE_OFFSETS,
                                       TIFFHeader, parse_all_ifds, parse_header, parse_ifd)
from xrspatial.geotiff._reader import read_to_array

from .._helpers.markers import requires_gpu, requires_loopback

_HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None
# Note: ``_HAS_CUPY`` (bare import probe) is intentionally NOT used to
# gate GPU tests below. A bare ``import cupy`` succeeds on hosts where
# the CUDA runtime is missing or unusable, which made
# ``TestSparseTilesGPU`` fail at device-call time instead of skipping.
# Use ``requires_gpu`` from ``_helpers.markers`` (which also probes
# ``cupy.cuda.is_available()``) for any test that needs a working GPU
# device. See issue #2487.

requires_rasterio = pytest.mark.skipif(
    not _HAS_RASTERIO, reason="rasterio required to write sparse fixtures"
)

if _HAS_RASTERIO:
    import rasterio
else:  # pragma: no cover - exercised only when rasterio is unavailable
    # ``rasterio`` stays defined at module scope so the sparse-fixture
    # helpers below (``_write_sparse_*``) parse at import time without
    # ``NameError``. They are only called from tests gated by
    # ``@requires_rasterio``, which skip before any helper runs.
    rasterio = None

# ---------------------------------------------------------------------------
# Shared TIFF byte builders
# ---------------------------------------------------------------------------


def _build_minimal_classic_ifd(num_entries: int = 1) -> bytes:
    """Complete little-endian classic TIFF with ``num_entries`` entries.

    Each entry is ``(TAG_IMAGE_WIDTH, LONG, count=1, inline value=1)``.
    Used as a base buffer for the entry-table truncation tests; callers
    slice it shorter to land mid-read.
    """
    bo = '<'
    ifd_offset = 8
    buf = bytearray()
    buf.extend(b'II')
    buf.extend(struct.pack(f'{bo}H', 42))
    buf.extend(struct.pack(f'{bo}I', ifd_offset))
    buf.extend(struct.pack(f'{bo}H', num_entries))
    for _ in range(num_entries):
        buf.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, LONG, 1))
        buf.extend(struct.pack(f'{bo}I', 1))
    buf.extend(struct.pack(f'{bo}I', 0))
    return bytes(buf)


def _build_minimal_bigtiff_ifd(num_entries: int = 1) -> bytes:
    """Complete little-endian BigTIFF with ``num_entries`` entries."""
    bo = '<'
    ifd_offset = 16
    buf = bytearray()
    buf.extend(b'II')
    buf.extend(struct.pack(f'{bo}H', 43))
    buf.extend(struct.pack(f'{bo}H', 8))
    buf.extend(b'\x00\x00')
    buf.extend(struct.pack(f'{bo}Q', ifd_offset))
    buf.extend(struct.pack(f'{bo}Q', num_entries))
    for _ in range(num_entries):
        buf.extend(struct.pack(f'{bo}HH', TAG_IMAGE_WIDTH, LONG))
        buf.extend(struct.pack(f'{bo}Q', 1))
        buf.extend(struct.pack(f'{bo}Q', 1))
    buf.extend(struct.pack(f'{bo}Q', 0))
    return bytes(buf)


def _build_tiff_with_entries(
    entries: list[tuple[int, int, int, bytes]],
    tail_padding: int = 0,
    external_payloads: list[tuple[int, bytes]] | None = None,
) -> bytes:
    """Classic little-endian TIFF with arbitrary IFD entries.

    Each entry is ``(tag, type_id, count, value_field_bytes)`` where the
    value field is exactly 4 bytes (inline value or overflow pointer).
    ``external_payloads`` lets the caller drop bytes at absolute file
    offsets, used to satisfy non-inline value pointers.
    """
    bo = '<'
    n = len(entries)
    ifd_offset = 8
    ifd_size = 2 + n * 12 + 4
    end_of_ifd = ifd_offset + ifd_size
    file_size = end_of_ifd + tail_padding
    if external_payloads:
        for off, payload in external_payloads:
            file_size = max(file_size, off + len(payload))

    buf = bytearray(file_size)
    buf[0:2] = b'II'
    struct.pack_into(f'{bo}H', buf, 2, 42)
    struct.pack_into(f'{bo}I', buf, 4, ifd_offset)
    struct.pack_into(f'{bo}H', buf, ifd_offset, n)
    for i, (tag, type_id, count, value_bytes) in enumerate(entries):
        eo = ifd_offset + 2 + i * 12
        struct.pack_into(f'{bo}H', buf, eo, tag)
        struct.pack_into(f'{bo}H', buf, eo + 2, type_id)
        struct.pack_into(f'{bo}I', buf, eo + 4, count)
        assert len(value_bytes) == 4, "value field must be exactly 4 bytes"
        buf[eo + 8:eo + 12] = value_bytes
    struct.pack_into(f'{bo}I', buf, ifd_offset + 2 + n * 12, 0)
    if external_payloads:
        for off, payload in external_payloads:
            buf[off:off + len(payload)] = payload
    return bytes(buf)


def _build_chained_classic_tiff(n_ifds: int, big_endian: bool = False) -> bytes:
    """Classic TIFF whose IFD chain has exactly ``n_ifds`` IFDs.

    Each IFD carries a single ``ImageWidth`` LONG entry and points to
    the next IFD via the trailing 4-byte next-pointer. The final IFD
    terminates with a 0 next-pointer.
    """
    bo = '>' if big_endian else '<'
    bom = b'MM' if big_endian else b'II'
    ifd_size = 18  # 2 (count) + 12 (one entry) + 4 (next-pointer)

    out = bytearray()
    out.extend(bom)
    out.extend(struct.pack(f'{bo}H', 42))
    first = 8
    out.extend(struct.pack(f'{bo}I', first))
    for i in range(n_ifds):
        next_off = first + (i + 1) * ifd_size
        if i == n_ifds - 1:
            next_off = 0
        out.extend(struct.pack(f'{bo}H', 1))
        out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 4, 1))
        out.extend(struct.pack(f'{bo}I', i + 1))
        out.extend(struct.pack(f'{bo}I', next_off))
    return bytes(out)


def _build_cyclic_two_ifd_tiff(big_endian: bool = False) -> bytes:
    """TIFF whose IFD chain forms a two-node cycle A -> B -> A.

    Both IFDs carry a single ``ImageWidth`` so each parses cleanly; the
    second IFD's next-pointer references the first IFD's offset to
    close the loop.
    """
    bo = '>' if big_endian else '<'
    bom = b'MM' if big_endian else b'II'

    out = bytearray()
    out.extend(bom)
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', 8))  # first_ifd_offset

    ifd_a_off = 8
    ifd_b_off = 26
    out.extend(struct.pack(f'{bo}H', 1))
    out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 4, 1))
    out.extend(struct.pack(f'{bo}I', 1))
    out.extend(struct.pack(f'{bo}I', ifd_b_off))
    out.extend(struct.pack(f'{bo}H', 1))
    out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 4, 1))
    out.extend(struct.pack(f'{bo}I', 2))
    out.extend(struct.pack(f'{bo}I', ifd_a_off))
    return bytes(out)


def _build_self_cycle_tiff() -> bytes:
    """TIFF whose first IFD points at itself (degenerate cycle)."""
    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack('<H', 42))
    first = 8
    out.extend(struct.pack('<I', first))
    out.extend(struct.pack('<H', 1))
    out.extend(struct.pack('<HHI', TAG_IMAGE_WIDTH, 4, 1))
    out.extend(struct.pack('<I', 1))
    out.extend(struct.pack('<I', first))
    return bytes(out)


def _build_single_ifd_with_next_offset(
    next_offset: int, big_endian: bool = False
) -> bytes:
    """Classic TIFF with one valid IFD that points to ``next_offset``.

    The single IFD carries ``ImageWidth=42`` so it parses cleanly. The
    trailing next-IFD pointer is set to whatever the caller asks for,
    so tests can drop it past EOF and assert the chain walker raises.
    """
    bo = '>' if big_endian else '<'
    bom = b'MM' if big_endian else b'II'

    out = bytearray()
    out.extend(bom)
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', 8))
    out.extend(struct.pack(f'{bo}H', 1))
    out.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, 4, 1))
    out.extend(struct.pack(f'{bo}I', 42))
    out.extend(struct.pack(f'{bo}I', next_offset))
    return bytes(out)


# ===========================================================================
# Section 1: parse_ifd entry-table bounds
# ===========================================================================


_ENTRY_TABLE_BOUND_CASES = [
    pytest.param(
        "classic", 1, lambda full: full[:9], "num_entries",
        id="entry_table[classic-num_entries-truncated]",
    ),
    pytest.param(
        "classic", 1, lambda full: full[:8], "num_entries",
        id="entry_table[classic-num_entries-zero-buffer]",
    ),
    pytest.param(
        "classic", 3, lambda full: full[: 8 + 2 + 3 * 12 - 1],
        "entry table",
        id="entry_table[classic-entry-table-truncated]",
    ),
    pytest.param(
        "classic", 1, lambda full: full[: 8 + 2 + 12], "next-IFD",
        id="entry_table[classic-next_ifd-truncated]",
    ),
    pytest.param(
        "classic", 1, lambda full: full[: 8 + 2 + 12 + 3], "next-IFD",
        id="entry_table[classic-next_ifd-one-short]",
    ),
    pytest.param(
        "bigtiff", 1, lambda full: full[:23], "num_entries",
        id="entry_table[bigtiff-num_entries-truncated]",
    ),
    pytest.param(
        "bigtiff", 2, lambda full: full[: 16 + 8 + 2 * 20 - 1],
        "entry table",
        id="entry_table[bigtiff-entry-table-truncated]",
    ),
    pytest.param(
        "bigtiff", 1, lambda full: full[: 16 + 8 + 20], "next-IFD",
        id="entry_table[bigtiff-next_ifd-truncated]",
    ),
    pytest.param(
        "bigtiff", 1, lambda full: full[: 16 + 8 + 20 + 7], "next-IFD",
        id="entry_table[bigtiff-next_ifd-one-short]",
    ),
]


@pytest.mark.parametrize(
    "flavour, num_entries, slicer, match", _ENTRY_TABLE_BOUND_CASES,
)
def test_entry_table_bounds_rejected(flavour, num_entries, slicer, match):
    """Buffers that end mid-read across the three entry-table reads
    must raise ``ValueError`` (not ``struct.error``)."""
    if flavour == "classic":
        full = _build_minimal_classic_ifd(num_entries=num_entries)
    else:
        full = _build_minimal_bigtiff_ifd(num_entries=num_entries)
    header = parse_header(full)
    truncated = slicer(full)
    with pytest.raises(ValueError, match=match):
        parse_ifd(truncated, header.first_ifd_offset, header)


@pytest.mark.parametrize(
    "flavour",
    [
        pytest.param("classic", id="entry_table[classic-complete]"),
        pytest.param("bigtiff", id="entry_table[bigtiff-complete]"),
    ],
)
def test_entry_table_complete_buffer_parses(flavour):
    """The build helpers produce well-formed files that parse cleanly."""
    if flavour == "classic":
        data = _build_minimal_classic_ifd(num_entries=1)
    else:
        data = _build_minimal_bigtiff_ifd(num_entries=1)
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.width == 1
    assert ifd.next_ifd_offset == 0


def test_entry_table_offset_past_eof_rejected():
    """``parse_ifd`` with ``offset > len(data)`` must raise ValueError,
    not ``struct.error`` from a deep ``unpack_from`` call."""
    full = _build_minimal_classic_ifd(num_entries=1)
    header = parse_header(full)
    with pytest.raises(ValueError, match="num_entries"):
        parse_ifd(full, len(full) + 100, header)


@pytest.mark.parametrize(
    "flavour, offset",
    [
        pytest.param("classic", -1, id="entry_table[classic-neg1]"),
        pytest.param("bigtiff", -1, id="entry_table[bigtiff-neg1]"),
        pytest.param("classic", -10_000, id="entry_table[classic-neg-10000]"),
    ],
)
def test_entry_table_negative_offset_rejected(flavour, offset):
    """``struct.unpack_from`` accepts negative offsets via Python
    indexing semantics; the explicit ``< 0`` guard must short-circuit
    that and raise ``ValueError`` instead of reading garbage."""
    if flavour == "classic":
        full = _build_minimal_classic_ifd(num_entries=1)
    else:
        full = _build_minimal_bigtiff_ifd(num_entries=1)
    header = parse_header(full)
    with pytest.raises(ValueError, match="num_entries"):
        parse_ifd(full, offset, header)


def test_entry_table_overrun_via_huge_num_entries_rejected():
    """A buffer that claims many more entries than it can hold trips
    the entry-table bounds guard."""
    full = _build_minimal_classic_ifd(num_entries=1)
    header = parse_header(full)
    bo = '<'
    huge_count = 1000
    buf = bytearray(full[:8])
    buf.extend(struct.pack(f'{bo}H', huge_count))
    buf.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, LONG, 1))
    buf.extend(struct.pack(f'{bo}I', 1))
    with pytest.raises(ValueError, match="entry table"):
        parse_ifd(bytes(buf), header.first_ifd_offset, header)


# ===========================================================================
# Section 2: parse_ifd entry-value bounds (S2)
# ===========================================================================


def test_entry_value_count_cap_rejected_at_default():
    """A non-pixel tag with ``count > MAX_IFD_ENTRY_COUNT`` must raise."""
    bad_count = MAX_IFD_ENTRY_COUNT + 1
    data = _build_tiff_with_entries(
        entries=[(TAG_IMAGE_WIDTH, LONG, bad_count, b'\x00\x00\x00\x00')],
        tail_padding=64,
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="count"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_entry_value_count_cap_rejected_under_lowered_caps(monkeypatch):
    """Count cap fires before any allocation when caps are lowered."""
    monkeypatch.setattr(_header, 'MAX_IFD_ENTRY_COUNT', 10)
    data = _build_tiff_with_entries(
        entries=[(TAG_IMAGE_WIDTH, LONG, 11, b'\x00\x00\x00\x00')],
        tail_padding=64,
    )
    header = parse_header(data)
    with pytest.raises(
        ValueError, match=r"count=11 exceeds MAX_IFD_ENTRY_COUNT=10"
    ):
        parse_ifd(data, header.first_ifd_offset, header)


def test_entry_value_byte_cap_rejected_under_lowered_caps(monkeypatch):
    """A count under the count cap but bytes over the byte cap raises.

    With ``count=33`` and ``DOUBLE`` (8 bytes per element), bytes=264 >
    256 (byte cap) while count < 100 (count cap), so the byte cap is
    the only guard that fires.
    """
    monkeypatch.setattr(_header, 'MAX_IFD_ENTRY_COUNT', 100)
    monkeypatch.setattr(_header, 'MAX_IFD_ENTRY_BYTES', 256)

    data = _build_tiff_with_entries(
        entries=[(TAG_IMAGE_WIDTH, DOUBLE, 33, b'\x00\x00\x00\x00')],
        tail_padding=64,
    )
    header = parse_header(data)
    with pytest.raises(
        ValueError, match="bytes exceeds MAX_IFD_ENTRY_BYTES"
    ):
        parse_ifd(data, header.first_ifd_offset, header)


def test_entry_value_byte_cap_rejected_at_default():
    """Default caps catch realistic large-itemsize abuse.

    With default caps (count=100K, bytes=256KiB), a non-pixel DOUBLE tag
    with count=32_770 has bytes=262_160 > 262_144 (256KiB) but count is
    well under 100K. Confirms the byte cap is independently useful at
    production values, not just under monkeypatch.
    """
    count = (MAX_IFD_ENTRY_BYTES // 8) + 2
    assert count < MAX_IFD_ENTRY_COUNT, (
        "test must exercise byte cap, not count cap"
    )
    data = _build_tiff_with_entries(
        entries=[(TAG_IMAGE_WIDTH, DOUBLE, count, b'\x00\x00\x00\x00')],
        tail_padding=64,
    )
    header = parse_header(data)
    with pytest.raises(
        ValueError, match="bytes exceeds MAX_IFD_ENTRY_BYTES"
    ):
        parse_ifd(data, header.first_ifd_offset, header)


def test_entry_value_range_past_eof_rejected():
    """A non-inline value range that extends past EOF must raise."""
    count = 5
    file_size_target = 200
    ptr = file_size_target - 4  # only 4 bytes follow, need 20
    value_bytes = struct.pack('<I', ptr)
    data = _build_tiff_with_entries(
        entries=[(TAG_IMAGE_WIDTH, LONG, count, value_bytes)],
        tail_padding=file_size_target - (8 + 2 + 12 + 4),
    )
    assert len(data) == file_size_target
    header = parse_header(data)
    with pytest.raises(ValueError, match="exceeds file length"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_entry_value_pixel_array_tag_exempt_from_caps(monkeypatch):
    """Pixel-array tags bypass both caps; a tiny lowered cap proves it.

    With non-pixel tags, count=11 would raise; with TileOffsets it
    must pass through.
    """
    monkeypatch.setattr(_header, 'MAX_IFD_ENTRY_COUNT', 10)
    monkeypatch.setattr(_header, 'MAX_IFD_ENTRY_BYTES', 16)

    count = 11
    payload_offset = 8 + 2 + 12 * 2 + 4
    payload = b'\x00' * (count * 4)  # LONG = 4 bytes
    value_bytes = struct.pack('<I', payload_offset)

    width_value = struct.pack('<I', 1024)
    data = _build_tiff_with_entries(
        entries=[
            (TAG_IMAGE_WIDTH, LONG, 1, width_value),
            (TAG_TILE_OFFSETS, LONG, count, value_bytes),
        ],
        external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.entries[TAG_TILE_OFFSETS].count == count


def test_entry_value_normal_tag_with_legal_count_passes():
    """Regression: a small, legal IFD parses cleanly."""
    width_value = struct.pack('<I', 256)
    height_value = struct.pack('<I', 256)
    data = _build_tiff_with_entries(
        entries=[
            (TAG_IMAGE_WIDTH, LONG, 1, width_value),
            (257, LONG, 1, height_value),  # ImageLength
        ],
    )
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.width == 256
    assert ifd.height == 256


def test_entry_value_short_count_pixel_tag_passes():
    """TileByteCounts of SHORT type passes for pixel-array tags."""
    count = 1000
    payload_offset = 8 + 2 + 12 + 4
    payload = b'\x00' * (count * 2)  # SHORT = 2 bytes
    value_bytes = struct.pack('<I', payload_offset)
    data = _build_tiff_with_entries(
        entries=[(TAG_TILE_BYTE_COUNTS, SHORT, count, value_bytes)],
        external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.entries[TAG_TILE_BYTE_COUNTS].count == count


# ===========================================================================
# Section 2b: parse_ifd duplicate-tag rejection (#2483)
# ===========================================================================
#
# TIFF 6.0 section 2 requires IFD entries to be sorted in ascending
# order by tag id with no duplicates. The legacy parser stored entries
# in a dict keyed by tag and let the last duplicate win, so a file
# with two ImageWidth (or any other tag) entries silently parsed to
# whichever value happened to come second. These tests pin the
# fail-closed behavior: every duplicate raises ``DuplicateIFDTagError``
# at parse time, and the error message names the duplicated tag id and
# the byte offsets of the two conflicting entries.


def test_duplicate_image_width_rejected_reproduction_2483():
    """Regression: the exact reproduction case from issue #2483.

    Two ``ImageWidth`` entries with values 4 and 999 must not silently
    resolve to 999; the parser must raise instead.
    """
    bo = '<'
    buf = bytearray()
    buf.extend(b'II')
    buf.extend(struct.pack(f'{bo}H', 42))
    buf.extend(struct.pack(f'{bo}I', 8))  # IFD at offset 8
    buf.extend(struct.pack(f'{bo}H', 2))  # num_entries
    buf.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, LONG, 1))
    buf.extend(struct.pack(f'{bo}I', 4))
    buf.extend(struct.pack(f'{bo}HHI', TAG_IMAGE_WIDTH, LONG, 1))
    buf.extend(struct.pack(f'{bo}I', 999))
    buf.extend(struct.pack(f'{bo}I', 0))
    data = bytes(buf)
    header = parse_header(data)
    with pytest.raises(DuplicateIFDTagError, match="tag 256 twice"):
        parse_ifd(data, header.first_ifd_offset, header)


@pytest.mark.parametrize(
    "tag, label",
    [
        pytest.param(TAG_IMAGE_WIDTH, "ImageWidth", id="duplicate_tag[image-width]"),
        pytest.param(TAG_IMAGE_LENGTH, "ImageLength", id="duplicate_tag[image-length]"),
        pytest.param(TAG_BITS_PER_SAMPLE, "BitsPerSample", id="duplicate_tag[bits-per-sample]"),
        pytest.param(TAG_GEO_KEY_DIRECTORY, "GeoKeyDirectory",
                     id="duplicate_tag[geo-key-directory]"),
    ],
)
def test_duplicate_tag_rejected_across_critical_tags(tag, label):
    """Duplicate detection fires on every tag, not just ImageWidth.

    ImageLength controls pixel-array sizing, BitsPerSample controls the
    decoded dtype, and the GeoKey directory carries the CRS contract.
    All three were vulnerable to the same last-value-wins bug before
    #2483.
    """
    value_bytes = struct.pack('<I', 1)
    data = _build_tiff_with_entries(
        entries=[
            (tag, LONG, 1, value_bytes),
            (tag, LONG, 1, struct.pack('<I', 999)),
        ],
    )
    header = parse_header(data)
    with pytest.raises(DuplicateIFDTagError, match=f"tag {tag} twice"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_duplicate_tag_error_names_offsets():
    """The error message must locate both conflicting entries by byte offset.

    Without these offsets a caller cannot find the bad bytes without
    re-parsing the IFD themselves.
    """
    value_bytes = struct.pack('<I', 1)
    data = _build_tiff_with_entries(
        entries=[
            (TAG_IMAGE_WIDTH, LONG, 1, value_bytes),
            (TAG_IMAGE_WIDTH, LONG, 1, struct.pack('<I', 999)),
        ],
    )
    header = parse_header(data)
    # IFD starts at byte 8; entry table starts at byte 10 (after 2-byte
    # num_entries). First entry at 10, second at 22 (10 + 12).
    with pytest.raises(DuplicateIFDTagError) as exc_info:
        parse_ifd(data, header.first_ifd_offset, header)
    msg = str(exc_info.value)
    assert "byte offset 10" in msg
    assert "byte offset 22" in msg


def test_duplicate_tag_rejected_in_bigtiff():
    """BigTIFF (entry size 20) shares the same fail-closed contract."""
    bo = '<'
    ifd_offset = 16
    buf = bytearray()
    buf.extend(b'II')
    buf.extend(struct.pack(f'{bo}H', 43))
    buf.extend(struct.pack(f'{bo}H', 8))
    buf.extend(b'\x00\x00')
    buf.extend(struct.pack(f'{bo}Q', ifd_offset))
    buf.extend(struct.pack(f'{bo}Q', 2))  # num_entries
    for value in (4, 999):
        buf.extend(struct.pack(f'{bo}HH', TAG_IMAGE_WIDTH, LONG))
        buf.extend(struct.pack(f'{bo}Q', 1))  # count
        buf.extend(struct.pack(f'{bo}Q', value))  # inline value
    buf.extend(struct.pack(f'{bo}Q', 0))  # next IFD
    data = bytes(buf)
    header = parse_header(data)
    assert header.is_bigtiff
    with pytest.raises(DuplicateIFDTagError, match="tag 256 twice"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_duplicate_tag_rejected_before_dimension_prescan_emits():
    """Dimension pre-scan path can't bypass the duplicate check either.

    The pre-scan reads inline values for ``ImageWidth`` /
    ``ImageLength`` to bound pixel-array counts in the second loop. The
    duplicate check runs first, so a malformed file with two
    ``ImageWidth`` entries fails at the pre-scan step rather than
    reaching the early-exit code path with the wrong dimension value.
    """
    data = _build_tiff_with_entries(
        entries=[
            (TAG_IMAGE_WIDTH, LONG, 1, struct.pack('<I', 4)),
            (TAG_IMAGE_LENGTH, LONG, 1, struct.pack('<I', 4)),
            (TAG_IMAGE_WIDTH, LONG, 1, struct.pack('<I', 999)),
        ],
    )
    header = parse_header(data)
    with pytest.raises(DuplicateIFDTagError, match="tag 256 twice"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_duplicate_tag_error_subclasses_value_error():
    """``except ValueError`` callers keep catching the case.

    ``DuplicateIFDTagError`` subclasses ``ValueError`` so existing
    consumers (the public ``open_geotiff`` boundary, the legacy
    ``ValueError`` family) continue to catch malformed-IFD failures
    without code changes; new code can ``except`` the more specific
    type to distinguish duplicate-tag rejection from other parse
    failures.
    """
    assert issubclass(DuplicateIFDTagError, ValueError)
    data = _build_tiff_with_entries(
        entries=[
            (TAG_IMAGE_WIDTH, LONG, 1, struct.pack('<I', 4)),
            (TAG_IMAGE_WIDTH, LONG, 1, struct.pack('<I', 999)),
        ],
    )
    header = parse_header(data)
    with pytest.raises(ValueError):
        parse_ifd(data, header.first_ifd_offset, header)


def test_duplicate_tag_surfaces_at_public_read_boundary(tmp_path):
    """``open_geotiff`` must surface ``DuplicateIFDTagError`` to callers.

    The duplicate check lives in the IFD parser, but the contract is
    that the failure reaches the public read boundary so a caller using
    ``open_geotiff`` (not the private ``parse_ifd``) sees it.
    """
    data = _build_tiff_with_entries(
        entries=[
            (TAG_IMAGE_WIDTH, LONG, 1, struct.pack('<I', 4)),
            (TAG_IMAGE_WIDTH, LONG, 1, struct.pack('<I', 999)),
        ],
    )
    path = tmp_path / "issue_2483_dup_image_width.tif"
    path.write_bytes(data)
    with pytest.raises(DuplicateIFDTagError, match="tag 256 twice"):
        open_geotiff(str(path))


def test_legal_distinct_tags_still_parse_after_duplicate_check():
    """Regression: distinct tag ids in a well-formed IFD still parse.

    Confirms the duplicate check does not over-fire on a normal IFD
    that legitimately carries many different tag ids.
    """
    width_value = struct.pack('<I', 256)
    height_value = struct.pack('<I', 256)
    data = _build_tiff_with_entries(
        entries=[
            (TAG_IMAGE_WIDTH, LONG, 1, width_value),
            (TAG_IMAGE_LENGTH, LONG, 1, height_value),
            (TAG_BITS_PER_SAMPLE, SHORT, 1, b'\x08\x00\x00\x00'),
        ],
    )
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.width == 256
    assert ifd.height == 256


# ===========================================================================
# Section 3: parse_all_ifds chain length cap (MAX_IFDS, security S3)
# ===========================================================================


@pytest.mark.parametrize(
    "big_endian, overshoot",
    [
        pytest.param(False, 50, id="chain_cap[over-limit-le]"),
        pytest.param(True, 10, id="chain_cap[over-limit-be]"),
    ],
)
def test_chain_cap_rejects_oversized(big_endian, overshoot):
    """A chain past ``MAX_IFDS`` must raise, not silently grow.

    Big-endian variant proves the cap is byte-order independent.
    """
    data = _build_chained_classic_tiff(
        MAX_IFDS + overshoot, big_endian=big_endian
    )
    header = parse_header(data)
    if big_endian:
        assert header.byte_order == '>'
    with pytest.raises(ValueError, match=str(MAX_IFDS)):
        parse_all_ifds(data, header)


def test_chain_cap_boundary_passes_at_max_and_fails_at_max_plus_one():
    """Convention: ``MAX_IFDS`` IFDs is the largest accepted chain.

    The parser raises once ``len(ifds) > MAX_IFDS`` after appending,
    matching the ``> MAX_IFD_ENTRY_COUNT`` pattern elsewhere in the
    module so "MAX = N" reads as "up to and including N is allowed".
    """
    data_at = _build_chained_classic_tiff(MAX_IFDS)
    header_at = parse_header(data_at)
    ifds_at = parse_all_ifds(data_at, header_at)
    assert len(ifds_at) == MAX_IFDS

    data_over = _build_chained_classic_tiff(MAX_IFDS + 1)
    header_over = parse_header(data_over)
    with pytest.raises(ValueError, match=str(MAX_IFDS)):
        parse_all_ifds(data_over, header_over)


def test_chain_cap_error_message_mentions_dos_and_limit():
    """The error message names ``MAX_IFDS``, the numeric value, and the
    threat-model language so operators see why it tripped."""
    data = _build_chained_classic_tiff(MAX_IFDS + 5)
    header = parse_header(data)
    with pytest.raises(ValueError) as excinfo:
        parse_all_ifds(data, header)
    msg = str(excinfo.value)
    assert "MAX_IFDS" in msg
    assert str(MAX_IFDS) in msg
    assert "denial-of-service" in msg or "malformed" in msg


def test_chain_cap_short_chain_passes():
    """A small handful of IFDs (typical pyramid depth) parses fine."""
    data = _build_chained_classic_tiff(8)
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    assert len(ifds) == 8
    for i, ifd in enumerate(ifds):
        assert ifd.width == i + 1


def test_chain_single_ifd_parses():
    """A one-IFD file (no next pointer) still parses."""
    data = _build_chained_classic_tiff(1)
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    assert len(ifds) == 1


def test_chain_cap_legitimate_cog_with_overviews_passes(tmp_path):
    """A real COG with several overview levels parses fine.

    Real-world COGs have well under ``MAX_IFDS`` IFDs even with many
    overview levels and per-band masks; the cap should never get in
    their way. Unlike the other Section 3 tests, this one stands up
    the public ``to_geotiff`` writer to produce the COG fixture; the
    assertion is still on the parsed IFD chain.
    """
    arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    path = str(tmp_path / 'real_cog_2426.tif')
    to_geotiff(
        arr, path, compression='deflate', tiled=True, tile_size=64,
        cog=True, overview_levels=[2, 4, 8],
    )
    with open(path, 'rb') as f:
        data = f.read()
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    assert 1 < len(ifds) < MAX_IFDS
    assert len(ifds) <= 16  # well under the cap


# ===========================================================================
# Section 4: parse_all_ifds chain cycle detection
# ===========================================================================


@pytest.mark.parametrize(
    "builder, big_endian",
    [
        pytest.param(
            lambda be: _build_cyclic_two_ifd_tiff(big_endian=be), False,
            id="chain_cycle[a-b-a-le]",
        ),
        pytest.param(
            lambda be: _build_cyclic_two_ifd_tiff(big_endian=be), True,
            id="chain_cycle[a-b-a-be]",
        ),
        pytest.param(
            lambda be: _build_self_cycle_tiff(), False,
            id="chain_cycle[self-le]",
        ),
    ],
)
def test_chain_cycle_rejected(builder, big_endian):
    """A cyclic IFD chain must raise rather than return a truncated
    list. Covers the two-node, self-cycle, and big-endian variants."""
    data = builder(big_endian)
    header = parse_header(data)
    if big_endian:
        assert header.byte_order == '>'
    with pytest.raises(ValueError, match="cycle"):
        parse_all_ifds(data, header)


def test_chain_cycle_error_message_mentions_offset_and_malformed():
    """The error message should name the repeat offset and the file
    state so an operator can diagnose the loop."""
    data = _build_cyclic_two_ifd_tiff()
    header = parse_header(data)
    with pytest.raises(ValueError) as excinfo:
        parse_all_ifds(data, header)
    msg = str(excinfo.value)
    assert "8" in msg
    assert "malformed" in msg
    assert "cycle" in msg


# ===========================================================================
# Section 5: parse_all_ifds malformed chain offsets
# ===========================================================================


@pytest.mark.parametrize(
    "big_endian",
    [
        pytest.param(False, id="chain_offset[next-past-eof-le]"),
        pytest.param(True, id="chain_offset[next-past-eof-be]"),
    ],
)
def test_chain_next_ifd_offset_past_eof_rejected(big_endian):
    """A chain whose ``next_ifd_offset`` points past EOF raises with a
    message that names the bad offset and calls the file malformed."""
    data = _build_single_ifd_with_next_offset(
        next_offset=0xDEADBEEF, big_endian=big_endian
    )
    header = parse_header(data)
    if big_endian:
        assert header.byte_order == '>'
    with pytest.raises(ValueError) as excinfo:
        parse_all_ifds(data, header)
    msg = str(excinfo.value)
    assert "malformed" in msg
    # The offset value is decoded to the same integer regardless of
    # byte order, so the message contract holds on both branches.
    assert str(0xDEADBEEF) in msg


def test_chain_next_ifd_offset_at_file_length_rejected():
    """Boundary: pointer at offset == len(data) (one byte past EOF)."""
    buf = _build_single_ifd_with_next_offset(next_offset=0)
    tail_offset = len(buf) - 4
    eof_offset = len(buf)
    patched = bytearray(buf)
    struct.pack_into('<I', patched, tail_offset, eof_offset)
    data = bytes(patched)
    header = parse_header(data)
    with pytest.raises(ValueError, match="malformed"):
        parse_all_ifds(data, header)


def test_first_ifd_offset_past_eof_rejected():
    """A header whose ``first_ifd_offset`` is past EOF must raise."""
    bo = '<'
    first_ifd_offset = 0xCAFEBABE
    buf = bytearray()
    buf.extend(b'II')
    buf.extend(struct.pack(f'{bo}H', 42))
    buf.extend(struct.pack(f'{bo}I', first_ifd_offset))
    data = bytes(buf)
    header = parse_header(data)
    assert header.first_ifd_offset == first_ifd_offset
    with pytest.raises(ValueError) as excinfo:
        parse_all_ifds(data, header)
    msg = str(excinfo.value)
    assert str(first_ifd_offset) in msg
    # The phrase comes from _header.py and lets operators diagnose
    # truncation vs corruption at a glance.
    assert "past end of file" in msg


def test_first_ifd_offset_past_eof_rejected_synthetic_header():
    """A fabricated ``TIFFHeader`` with first_ifd_offset > len(data)
    must also raise, catching callers that bypass ``parse_header``."""
    data = b'II' + struct.pack('<H', 42) + struct.pack('<I', 8)
    header = TIFFHeader(
        byte_order='<', is_bigtiff=False, first_ifd_offset=9999
    )
    with pytest.raises(ValueError, match="9999"):
        parse_all_ifds(data, header)


def test_chain_valid_terminator_still_parses():
    """A normal ``next_ifd_offset=0`` still terminates cleanly."""
    data = _build_single_ifd_with_next_offset(next_offset=0)
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    assert len(ifds) == 1
    assert ifds[0].width == 42


# ===========================================================================
# Section 6: sparse blocks via the public read pipeline
# ===========================================================================
#
# These exercise the reader's handling of the IFD-level sparse encoding
# (TileByteCounts == 0 / StripByteCounts == 0). They depend on the
# rasterio GDAL bridge to write the sparse fixture; the reader-side
# code under test is pure xrspatial. Sections 1-5 above do not need
# rasterio, so the dependency is scoped per-class via
# ``@requires_rasterio`` rather than a module-level ``importorskip``.


def _write_sparse_tiled(
    path, *, dtype='uint16', nodata=0, compress='DEFLATE',
    filled_value=100,
):
    """128x128 raster where only the top-left 64x64 tile is filled."""
    profile = {
        'driver': 'GTiff', 'dtype': dtype,
        'height': 128, 'width': 128, 'count': 1,
        'tiled': True, 'blockxsize': 64, 'blockysize': 64,
        'compress': compress, 'SPARSE_OK': 'TRUE',
    }
    if nodata is not None:
        profile['nodata'] = nodata
    fill = np.full((64, 64), filled_value, dtype=np.dtype(dtype))
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(fill, 1, window=rasterio.windows.Window(0, 0, 64, 64))


def _write_sparse_stripped_small(path, *, dtype='uint16', nodata=0):
    """128x128 raster with the top 32 rows filled, rest sparse."""
    profile = {
        'driver': 'GTiff', 'dtype': dtype,
        'height': 128, 'width': 128, 'count': 1,
        'tiled': False, 'blockysize': 16,
        'compress': 'DEFLATE', 'SPARSE_OK': 'TRUE',
    }
    if nodata is not None:
        profile['nodata'] = nodata
    fill = np.full((32, 128), 200, dtype=np.dtype(dtype))
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(fill, 1, window=rasterio.windows.Window(0, 0, 128, 32))


@requires_rasterio
class TestSparseTiles:

    def test_sparse_tile_with_nodata_round_trips(self, tmp_path):
        path = str(tmp_path / 'sparse_nodata_2426.tif')
        _write_sparse_tiled(path, nodata=0)

        arr = open_geotiff(path, masked=True)
        arr_np = np.asarray(arr)
        assert arr_np[:64, :64].sum() == 64 * 64 * 100
        assert np.all(np.isnan(arr_np[:64, 64:]))
        assert np.all(np.isnan(arr_np[64:, :]))

    def test_sparse_tile_without_nodata_fills_zero(self, tmp_path):
        path = str(tmp_path / 'sparse_no_nodata_2426.tif')
        _write_sparse_tiled(path, nodata=None)

        arr = open_geotiff(path)
        arr_np = np.asarray(arr)
        assert arr_np.dtype == np.uint16
        assert arr_np[:64, :64].sum() == 64 * 64 * 100
        assert arr_np[:64, 64:].sum() == 0
        assert arr_np[64:, :].sum() == 0

    def test_sparse_tile_raw_read_uses_nodata_sentinel(self, tmp_path):
        """``read_to_array`` returns raw values; sparse tiles == nodata."""
        path = str(tmp_path / 'sparse_raw_2426.tif')
        _write_sparse_tiled(path, nodata=255)

        arr, geo = read_to_array(path)
        assert arr.dtype == np.uint16
        assert arr[:64, :64].sum() == 64 * 64 * 100
        assert np.all(arr[:64, 64:] == 255)
        assert np.all(arr[64:, :] == 255)


@requires_rasterio
class TestSparseStrips:

    def test_sparse_strip_with_nodata(self, tmp_path):
        path = str(tmp_path / 'sparse_strips_2426.tif')
        _write_sparse_stripped_small(path, nodata=0)

        arr = open_geotiff(path, masked=True)
        arr_np = np.asarray(arr)
        assert arr_np[:32, :].sum() == 32 * 128 * 200
        assert np.all(np.isnan(arr_np[32:, :]))


@requires_rasterio
@requires_gpu
class TestSparseTilesGPU:

    def test_sparse_tile_gpu_round_trip(self, tmp_path):
        path = str(tmp_path / 'sparse_gpu_2426.tif')
        _write_sparse_tiled(path, nodata=0)

        arr = open_geotiff(path, gpu=True, masked=True)
        # GPU read applies the high-level nodata mask: the
        # source uint16 raster is promoted to float64 and sentinel
        # values become NaN, matching the CPU eager path.
        host = arr.data.get()
        assert host.dtype == np.float64
        assert arr.attrs.get('nodata') == 0.0
        np.testing.assert_array_equal(
            host[:64, :64], np.full((64, 64), 100.0)
        )
        assert np.all(np.isnan(host[:64, 64:]))
        assert np.all(np.isnan(host[64:, :]))


def test_sparse_tiles_gpu_uses_capability_marker_2487():
    """Pin the gate for ``TestSparseTilesGPU`` to the capability marker.

    Regression test for #2487. The class previously gated on a bare
    ``_HAS_CUPY = importlib.util.find_spec("cupy") is not None`` probe.
    On hosts where cupy imports but the CUDA runtime is unusable, that
    gate let the test run and fail at device-call time instead of
    skipping cleanly. The gate must be ``requires_gpu`` from
    ``_helpers.markers``, which also probes ``cupy.cuda.is_available()``.
    """
    from .._helpers import markers as _markers_mod

    marks = list(getattr(TestSparseTilesGPU, 'pytestmark', []))
    reasons = {getattr(m, 'kwargs', {}).get('reason', '') for m in marks}
    # Both gates apply: rasterio (fixture writer) and gpu (device).
    assert any('cupy + CUDA required' in r for r in reasons), (
        f"TestSparseTilesGPU must gate on requires_gpu "
        f"(reason 'cupy + CUDA required'); saw {reasons}"
    )
    # The marker the class actually carries must be the shared
    # ``requires_gpu`` from ``_helpers.markers`` (which probes
    # ``cupy.cuda.is_available()``), not a locally-built skipif on a
    # bare ``import cupy`` probe. Compare the underlying ``Mark`` so
    # this works regardless of whether pytest stored the
    # ``MarkDecorator`` wrapper or unwrapped it.
    expected_mark = _markers_mod.requires_gpu.mark
    assert expected_mark in marks, (
        f"TestSparseTilesGPU is not gated on the shared "
        f"_helpers.markers.requires_gpu marker; saw marks={marks}"
    )


# ===========================================================================
# Section 7: sparse strips through the parallel-decode pipeline
# ===========================================================================
#
# The strip-decode parallelisation added a collect-decode-
# place pipeline in both ``_read_strips`` and
# ``_fetch_decode_cog_http_strips``. The job-collection loop filters
# sparse strips (``byte_counts[idx] == 0``) so the pool never decodes
# an empty byte slice, and the pre-allocated result already carries the
# sparse fill value. A regression that lost the sparse filter (for
# example by appending a job before the ``if byte_counts[...] == 0:
# continue`` guard) would slip past the small-fixture test in section
# 6 because that fixture is 128x128, well below the 64K-pixel
# parallel-decode gate.
#
# The fixtures here build a large (>= 64K strip pixels, multi-strip)
# sparse-stripped TIFF so the parallel branch engages, then assert
# parallel == serial output and that the pool is engaged (or not, for
# the all-sparse case).


def _write_sparse_stripped_large(
    path: str,
    *,
    width: int = 2048,
    height: int = 2048,
    rps: int = 64,
    filled_rows: int = 256,
    fill_value: int = 200,
    dtype: str = "uint16",
    nodata: int = 0,
    bands: int = 1,
    planar: str = "pixel",
):
    """Large stripped TIFF with sparse strips below ``filled_rows``.

    Default geometry (2048x2048, rps=64) yields ``width * rps =
    131_072`` pixels per strip -- clear of the 64K parallel-decode gate
    -- and 32 strips per band. Leaving rows below ``filled_rows``
    un-written produces ``32 - filled_rows / rps`` sparse strips that
    the job-collection loop must filter.

    ``planar``: ``"pixel"`` (contig, planar=1) or ``"band"``
    (planar=2 / separate). Rasterio accepts only those literals.
    """
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "height": height,
        "width": width,
        "count": bands,
        "tiled": False,
        "blockysize": rps,
        "compress": "DEFLATE",
        "SPARSE_OK": "TRUE",
        "nodata": nodata,
        "interleave": planar,
    }
    fill = np.full((filled_rows, width), fill_value, dtype=np.dtype(dtype))
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(1, bands + 1):
            dst.write(
                fill, b,
                window=rasterio.windows.Window(0, 0, width, filled_rows),
            )


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    blob: bytes = b""

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.blob)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            r0, r1 = rng[len("bytes="):].split("-")
            r0 = int(r0)
            r1 = int(r1) if r1 else len(self.blob) - 1
            r1 = min(r1, len(self.blob) - 1)
            body = self.blob[r0:r1 + 1]
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {r0}-{r1}/{len(self.blob)}",
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.blob)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(self.blob)

    def log_message(self, format, *args):
        return


def _start_server(blob: bytes):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    handler = type("BlobHandler", (_RangeHandler,), {"blob": blob})
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server, port


@requires_rasterio
def test_sparse_strips_full_image_parallel_matches_serial(tmp_path):
    """Sparse + non-sparse strips: parallel and serial paths return
    bit-identical output, and sparse rows carry the nodata sentinel."""
    path = str(tmp_path / "sparse_par_full_2426.tif")
    _write_sparse_stripped_large(path)

    par, _ = read_to_array(path)
    # Patch the threshold in ``_decode`` (the live binding home),
    # not in ``_reader``: the back-imported name in ``_reader`` is a
    # separate reference and patching it would leave the live binding
    # in ``_decode`` unchanged.
    with patch.object(
        _decode_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10 ** 12
    ):
        ser, _ = read_to_array(path)

    np.testing.assert_array_equal(par, ser)
    assert np.all(par[:256, :] == 200)
    assert np.all(par[256:, :] == 0)


@requires_rasterio
def test_sparse_strips_parallel_pool_engages_on_multi_strip(tmp_path):
    """A multi-strip sparse fixture must engage the parallel pool.

    Validates that the sparse-strip filter does not regress the gate by
    pruning the job list below ``n_strips > 1``.
    """
    path = str(tmp_path / "sparse_par_gate_2426.tif")
    # 4 strips filled, 28 sparse -> 4 non-sparse strips; pool engages
    # because n_strips = 4 > 1 and strip_pixel_count = 2048 * 64 =
    # 131_072 >= 65_536.
    _write_sparse_stripped_large(path, filled_rows=256)
    # Patch concurrent.futures.ThreadPoolExecutor rather than the
    # reader module binding: strip decode lives in ``_decode`` and
    # re-imports the executor function-locally.
    with patch.object(
        concurrent.futures, "ThreadPoolExecutor",
        wraps=concurrent.futures.ThreadPoolExecutor,
    ) as mock_pool:
        out, _ = read_to_array(path)
        assert mock_pool.called, (
            "parallel-decode pool was not engaged for a multi-strip "
            "sparse-stripped TIFF whose non-sparse strips clear the "
            "parallel gate"
        )
    assert np.all(out[:256, :] == 200)
    assert np.all(out[256:, :] == 0)


@requires_rasterio
def test_sparse_strips_windowed_across_boundary(tmp_path):
    """A window that straddles the filled/sparse boundary returns the
    filled rows on top and sparse rows below, parallel == serial."""
    path = str(tmp_path / "sparse_par_win_2426.tif")
    _write_sparse_stripped_large(path)

    win = (128, 0, 384, 1024)
    par, _ = read_to_array(path, window=win)
    with patch.object(
        _decode_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10 ** 12
    ):
        ser, _ = read_to_array(path, window=win)

    np.testing.assert_array_equal(par, ser)
    assert par.shape == (256, 1024)
    assert np.all(par[:128, :] == 200)
    assert np.all(par[128:, :] == 0)


@requires_rasterio
def test_sparse_strips_all_sparse_image_returns_fill(tmp_path):
    """An image with zero filled rows: every strip sparse, the job
    list is empty, and the parallel branch's ``n_strips > 1`` gate
    short-circuits without instantiating the pool."""
    path = str(tmp_path / "all_sparse_2426.tif")
    _write_sparse_stripped_large(path, filled_rows=0)
    with patch.object(
        concurrent.futures, "ThreadPoolExecutor",
        wraps=concurrent.futures.ThreadPoolExecutor,
    ) as mock_pool:
        out, _ = read_to_array(path)
        assert not mock_pool.called, (
            "parallel-decode pool was instantiated for an all-sparse "
            "image; the strip-job filter should have left the job list "
            "empty and the gate should have short-circuited"
        )
    assert out.shape == (2048, 2048)
    assert np.all(out == 0)


@requires_rasterio
def test_sparse_strips_planar2_parallel_matches_serial(tmp_path):
    """``_read_strips`` planar=2 branch with sparse strips.

    The strip-job collection loop has a dedicated
    ``planar == 2 and samples > 1`` branch with its own
    ``if byte_counts[global_idx] == 0: continue`` guard. The
    existing parallel-strip planar=2 tests fill every strip, so a
    regression in this branch's sparse filter would survive without
    this case.
    """
    path = str(tmp_path / "planar2_sparse_2426.tif")
    _write_sparse_stripped_large(
        path,
        width=1024,
        height=1024,
        rps=64,
        filled_rows=128,
        bands=3,
        planar="band",
    )

    par, _ = read_to_array(path)
    with patch.object(
        _decode_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10 ** 12
    ):
        ser, _ = read_to_array(path)

    np.testing.assert_array_equal(par, ser)
    assert par.shape == (1024, 1024, 3)
    for b in range(3):
        assert np.all(par[:128, :, b] == 200)
        assert np.all(par[128:, :, b] == 0)


@requires_rasterio
@requires_loopback
class TestHttpStripsSparseParallel:
    """``_fetch_decode_cog_http_strips`` with sparse strips.

    The HTTP strip path also filters ``byte_counts[idx] == 0`` from the
    fetch-range list; a window that targets only non-sparse strips
    still parallel-decodes, and the final placement loop must match the
    local path.
    """

    def test_sparse_strips_http_windowed_strict_subset_parallel(
        self, tmp_path, monkeypatch
    ):
        """HTTP windowed read on a sparse-stripped TIFF.

        Targeted window covers only filled rows so the fetch list
        excludes the sparse strips, the parallel-decode gate engages,
        and the result matches the local file read.
        """
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
        path = str(tmp_path / "sparse_http_2426.tif")
        _write_sparse_stripped_large(path, filled_rows=256)
        with open(path, "rb") as f:
            blob = f.read()

        server, port = _start_server(blob)
        try:
            url = f"http://127.0.0.1:{port}/sparse.tif"
            par, _ = read_to_array(url, window=(0, 0, 256, 2048))
            with patch.object(
                _reader_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10 ** 12
            ):
                ser, _ = read_to_array(url, window=(0, 0, 256, 2048))
        finally:
            server.shutdown()

        np.testing.assert_array_equal(par, ser)
        assert np.all(par == 200)

    def test_sparse_strips_http_windowed_across_boundary(
        self, tmp_path, monkeypatch
    ):
        """HTTP windowed read straddling the sparse boundary.

        The fetch path emits a fetch range per non-sparse strip the
        window touches, the decoder runs in parallel on those, and the
        sparse strips inside the window carry the pre-filled fill
        value.
        """
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
        path = str(tmp_path / "sparse_http_boundary_2426.tif")
        _write_sparse_stripped_large(path, filled_rows=256)
        with open(path, "rb") as f:
            blob = f.read()

        server, port = _start_server(blob)
        try:
            url = f"http://127.0.0.1:{port}/sparse2.tif"
            par, _ = read_to_array(url, window=(128, 0, 384, 2048))
            with patch.object(
                _reader_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10 ** 12
            ):
                ser, _ = read_to_array(url, window=(128, 0, 384, 2048))
        finally:
            server.shutdown()

        np.testing.assert_array_equal(par, ser)
        assert par.shape == (256, 2048)
        assert np.all(par[:128, :] == 200)
        assert np.all(par[128:, :] == 0)
