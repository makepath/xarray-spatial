"""Pixel-array IFD tag count must be bounded against IFD dimensions.

Regression for issue #1901. `StripOffsets`, `StripByteCounts`,
`TileOffsets`, `TileByteCounts`, and `ColorMap` were exempt from the
generic `MAX_IFD_ENTRY_COUNT` cap because their `count` legitimately
scales with image size. The exemption made it possible to craft a
TIFF whose value pointer falls inside the file but whose `count` is
astronomically large; `parse_ifd` would then allocate a Python tuple
of `count` PyLong objects before any layout validation ran.

The fix:

* `MAX_PIXEL_ARRAY_COUNT` (100M) caps any pixel-array tag absolutely.
* `_expected_pixel_array_count` derives a tighter cap from the IFD's
  ImageWidth / ImageLength / TileWidth / TileLength / RowsPerStrip /
  SamplesPerPixel / PlanarConfiguration / BitsPerSample tags. The
  parser pre-scans those (inline only) before the main entry loop.
"""
from __future__ import annotations

import struct

import pytest

from xrspatial.geotiff import _header
from xrspatial.geotiff._dtypes import LONG, SHORT
from xrspatial.geotiff._header import (
    MAX_PIXEL_ARRAY_COUNT,
    TAG_BITS_PER_SAMPLE,
    TAG_COLORMAP,
    TAG_IMAGE_LENGTH,
    TAG_IMAGE_WIDTH,
    TAG_PLANAR_CONFIG,
    TAG_ROWS_PER_STRIP,
    TAG_SAMPLES_PER_PIXEL,
    TAG_STRIP_BYTE_COUNTS,
    TAG_STRIP_OFFSETS,
    TAG_TILE_BYTE_COUNTS,
    TAG_TILE_LENGTH,
    TAG_TILE_OFFSETS,
    TAG_TILE_WIDTH,
    parse_header,
    parse_ifd,
)


def _short_bytes(v: int) -> bytes:
    return struct.pack('<H', v) + b'\x00\x00'


def _long_bytes(v: int) -> bytes:
    return struct.pack('<I', v)


def _build_classic_tiff(
    entries: list[tuple[int, int, int, bytes]],
    tail_padding: int = 0,
    external_payloads: list[tuple[int, bytes]] | None = None,
) -> bytes:
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
        assert len(value_bytes) == 4
        buf[eo + 8:eo + 12] = value_bytes
    struct.pack_into(f'{bo}I', buf, ifd_offset + 2 + n * 12, 0)
    if external_payloads:
        for off, payload in external_payloads:
            buf[off:off + len(payload)] = payload
    return bytes(buf)


def test_tile_offsets_count_exceeds_geometry_rejected():
    """TileOffsets `count` larger than tiles_across * tiles_down raises.

    1024x1024 image, 256x256 tiles -> 16 tiles. count=100 must raise.
    """
    payload_offset = 8 + 2 + 12 * 5 + 4
    bad_count = 100
    payload = b'\x00' * (bad_count * 4)
    entries = [
        (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(1024)),
        (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(1024)),
        (TAG_TILE_WIDTH, LONG, 1, _long_bytes(256)),
        (TAG_TILE_LENGTH, LONG, 1, _long_bytes(256)),
        (TAG_TILE_OFFSETS, LONG, bad_count, _long_bytes(payload_offset)),
    ]
    data = _build_classic_tiff(
        entries, external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="exceeds expected value 16"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_tile_offsets_count_matching_geometry_passes():
    """16 tiles in a 1024x1024 image with 256x256 tiles must parse."""
    payload_offset = 8 + 2 + 12 * 5 + 4
    good_count = 16
    payload = b'\x00' * (good_count * 4)
    entries = [
        (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(1024)),
        (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(1024)),
        (TAG_TILE_WIDTH, LONG, 1, _long_bytes(256)),
        (TAG_TILE_LENGTH, LONG, 1, _long_bytes(256)),
        (TAG_TILE_OFFSETS, LONG, good_count, _long_bytes(payload_offset)),
    ]
    data = _build_classic_tiff(
        entries, external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.entries[TAG_TILE_OFFSETS].count == good_count


def test_strip_offsets_count_exceeds_geometry_rejected():
    """StripOffsets count larger than ceil(height / rows_per_strip) raises.

    256x256 with RowsPerStrip=64 -> 4 strips. count=200 must raise.
    """
    payload_offset = 8 + 2 + 12 * 4 + 4
    bad_count = 200
    payload = b'\x00' * (bad_count * 4)
    entries = [
        (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(256)),
        (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(256)),
        (TAG_ROWS_PER_STRIP, LONG, 1, _long_bytes(64)),
        (TAG_STRIP_OFFSETS, LONG, bad_count, _long_bytes(payload_offset)),
    ]
    data = _build_classic_tiff(
        entries, external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="exceeds expected value 4"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_strip_byte_counts_planar_multiplies_by_samples():
    """PlanarConfig=2 multiplies expected strip count by samples_per_pixel.

    256x256 with RowsPerStrip=64 and 3 samples planar -> 12 entries.
    count=12 passes; count=13 raises.
    """
    payload_offset = 8 + 2 + 12 * 6 + 4
    payload = b'\x00' * (12 * 4)
    base_entries = [
        (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(256)),
        (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(256)),
        (TAG_ROWS_PER_STRIP, LONG, 1, _long_bytes(64)),
        (TAG_SAMPLES_PER_PIXEL, SHORT, 1, _short_bytes(3)),
        (TAG_PLANAR_CONFIG, SHORT, 1, _short_bytes(2)),
    ]
    good = base_entries + [
        (TAG_STRIP_BYTE_COUNTS, LONG, 12, _long_bytes(payload_offset)),
    ]
    data = _build_classic_tiff(
        good, external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.entries[TAG_STRIP_BYTE_COUNTS].count == 12

    bad = base_entries + [
        (TAG_STRIP_BYTE_COUNTS, LONG, 13, _long_bytes(payload_offset)),
    ]
    data = _build_classic_tiff(
        bad, external_payloads=[(payload_offset, b'\x00' * (13 * 4))],
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="exceeds expected value 12"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_colormap_count_exceeds_bits_per_sample_rejected():
    """ColorMap count > 3 * 2^bits_per_sample raises.

    BitsPerSample=8 -> expected 3 * 256 = 768. count=2000 must raise.
    """
    payload_offset = 8 + 2 + 12 * 2 + 4
    bad_count = 2000
    payload = b'\x00' * (bad_count * 2)
    entries = [
        (TAG_BITS_PER_SAMPLE, SHORT, 1, _short_bytes(8)),
        (TAG_COLORMAP, SHORT, bad_count, _long_bytes(payload_offset)),
    ]
    data = _build_classic_tiff(
        entries, external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="exceeds expected value 768"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_colormap_count_at_expected_passes():
    """ColorMap with the exact expected count for BPS=8 must parse."""
    payload_offset = 8 + 2 + 12 * 2 + 4
    good_count = 3 * 256
    payload = b'\x00' * (good_count * 2)
    entries = [
        (TAG_BITS_PER_SAMPLE, SHORT, 1, _short_bytes(8)),
        (TAG_COLORMAP, SHORT, good_count, _long_bytes(payload_offset)),
    ]
    data = _build_classic_tiff(
        entries, external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.entries[TAG_COLORMAP].count == good_count


def test_absolute_cap_fires_when_dimensions_missing():
    """With no geometry tags in the IFD, MAX_PIXEL_ARRAY_COUNT alone caps.

    Monkeypatched down to keep the test cheap.
    """
    cap = 100
    monkey_value = cap
    orig = _header.MAX_PIXEL_ARRAY_COUNT
    _header.MAX_PIXEL_ARRAY_COUNT = monkey_value
    try:
        bad_count = cap + 1
        entries = [
            (TAG_TILE_OFFSETS, LONG, bad_count, _long_bytes(0)),
        ]
        data = _build_classic_tiff(entries, tail_padding=512)
        header = parse_header(data)
        with pytest.raises(
            ValueError, match=r"exceeds MAX_PIXEL_ARRAY_COUNT=100"
        ):
            parse_ifd(data, header.first_ifd_offset, header)
    finally:
        _header.MAX_PIXEL_ARRAY_COUNT = orig


def test_absolute_cap_constant_is_reasonable():
    """Sanity check: 100M elements is enough for any realistic image but
    far below the count required to drive a multi-GiB allocation."""
    # 1M x 1M image at 256-pixel tiles is ~16M tiles.
    assert MAX_PIXEL_ARRAY_COUNT >= 16_000_000
    # 100M PyLongs is roughly 3 GiB; refuse to allocate more than that.
    assert MAX_PIXEL_ARRAY_COUNT <= 1_000_000_000


def test_dimensions_listed_after_pixel_array_tag_still_validate():
    """Pre-scan must collect dimensions even when the pixel-array tag
    appears earlier in tag-numeric order than they do.

    A malicious file could reorder entries; the parser pre-scan walks
    the whole entry table before validating counts.
    """
    payload_offset = 8 + 2 + 12 * 5 + 4
    bad_count = 100
    payload = b'\x00' * (bad_count * 4)
    # Same 1024x1024, 256x256 case (16 tiles), but TileOffsets first.
    entries = [
        (TAG_TILE_OFFSETS, LONG, bad_count, _long_bytes(payload_offset)),
        (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(1024)),
        (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(1024)),
        (TAG_TILE_WIDTH, LONG, 1, _long_bytes(256)),
        (TAG_TILE_LENGTH, LONG, 1, _long_bytes(256)),
    ]
    # Note: TIFF spec says entries should be tag-sorted, but the parser
    # doesn't enforce that. We test that out-of-order entries still get
    # validated against the geometry.
    data = _build_classic_tiff(
        entries, external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="exceeds expected value 16"):
        parse_ifd(data, header.first_ifd_offset, header)


def test_strip_byte_counts_chunky_uses_image_length_only():
    """PlanarConfig=1 (chunky) does NOT multiply expected strip count.

    256x256 with RowsPerStrip=64 and 3 samples chunky -> 4 entries.
    """
    payload_offset = 8 + 2 + 12 * 6 + 4
    good_count = 4
    payload = b'\x00' * (good_count * 4)
    entries = [
        (TAG_IMAGE_WIDTH, LONG, 1, _long_bytes(256)),
        (TAG_IMAGE_LENGTH, LONG, 1, _long_bytes(256)),
        (TAG_ROWS_PER_STRIP, LONG, 1, _long_bytes(64)),
        (TAG_SAMPLES_PER_PIXEL, SHORT, 1, _short_bytes(3)),
        (TAG_PLANAR_CONFIG, SHORT, 1, _short_bytes(1)),
        (TAG_STRIP_OFFSETS, LONG, good_count, _long_bytes(payload_offset)),
    ]
    data = _build_classic_tiff(
        entries, external_payloads=[(payload_offset, payload)],
    )
    header = parse_header(data)
    ifd = parse_ifd(data, header.first_ifd_offset, header)
    assert ifd.entries[TAG_STRIP_OFFSETS].count == good_count

    # And chunky with count=5 raises.
    bad = entries[:-1] + [
        (TAG_STRIP_OFFSETS, LONG, 5, _long_bytes(payload_offset)),
    ]
    data = _build_classic_tiff(
        bad, external_payloads=[(payload_offset, b'\x00' * (5 * 4))],
    )
    header = parse_header(data)
    with pytest.raises(ValueError, match="exceeds expected value 4"):
        parse_ifd(data, header.first_ifd_offset, header)
