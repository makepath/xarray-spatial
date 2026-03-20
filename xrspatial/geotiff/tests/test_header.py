"""Tests for TIFF header and IFD parsing."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from xrspatial.geotiff._header import (
    IFD,
    TIFFHeader,
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
