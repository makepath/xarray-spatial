"""Tests for non-uniform SampleFormat handling (issue #1868).

A TIFF whose ``SampleFormat`` tag carries different values per band
(e.g. ``(3, 3, 1)`` for two float bands plus one uint band at the same
bit depth) cannot be decoded into a single numpy dtype. The reader
previously collapsed the tuple to ``v[0]``, silently reinterpreting
mismatched bands. xarray-spatial now rejects such files with a typed
error, mirroring the mixed ``BitsPerSample`` behaviour.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._dtypes import resolve_sample_format


def _build_multi_band_tiff_sf(
    width: int,
    height: int,
    samples: int,
    sample_format,
    bits_per_sample: int = 32,
    pixel_dtype: np.dtype = np.dtype('float32'),
) -> bytes:
    """Build a minimal stripped uncompressed multi-band TIFF.

    ``sample_format`` is written as-is into tag 339 -- pass a list/tuple
    to exercise the per-band code path.
    """
    bo = '<'
    pixel_data = np.zeros((height, width, samples), dtype=pixel_dtype)
    pixel_bytes = pixel_data.tobytes()

    tags: list[tuple[int, int, int, bytes]] = []

    def add_short(tag, val):
        tags.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tags.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_shorts(tag, vals):
        tags.append((tag, 3, len(vals),
                     struct.pack(f'{bo}{len(vals)}H', *vals)))

    add_short(256, width)                          # ImageWidth
    add_short(257, height)                         # ImageLength
    add_shorts(258, [bits_per_sample] * samples)   # BitsPerSample
    add_short(259, 1)                              # Compression = none
    add_short(262, 2 if samples >= 3 else 1)       # PhotometricInterpretation
    add_long(273, 0)                               # StripOffsets (patched)
    add_short(277, samples)                        # SamplesPerPixel
    add_short(278, height)                         # RowsPerStrip
    add_long(279, len(pixel_bytes))                # StripByteCounts
    add_short(284, 1)                              # PlanarConfiguration = chunky
    if isinstance(sample_format, (list, tuple)):
        add_shorts(339, list(sample_format))       # SampleFormat (per band)
    else:
        add_short(339, sample_format)

    tags.sort(key=lambda t: t[0])

    num_entries = len(tags)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_overflow_offsets: dict[int, int | None] = {}
    for tag, typ, count, raw in tags:
        if len(raw) > 4:
            tag_overflow_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_overflow_offsets[tag] = None

    pixel_start = overflow_start + len(overflow_buf)

    # Patch StripOffsets to point at the pixel block
    patched = []
    for tag, typ, count, raw in tags:
        if tag == 273:
            patched.append((tag, 4, 1, struct.pack(f'{bo}I', pixel_start)))
        else:
            patched.append((tag, typ, count, raw))
    tags = patched

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))
    for tag, typ, count, raw in tags:
        out.extend(struct.pack(f'{bo}H', tag))
        out.extend(struct.pack(f'{bo}H', typ))
        out.extend(struct.pack(f'{bo}I', count))
        if len(raw) <= 4:
            payload = raw + b'\x00' * (4 - len(raw))
            out.extend(payload)
        else:
            out.extend(struct.pack(f'{bo}I',
                                   overflow_start + tag_overflow_offsets[tag]))
    out.extend(struct.pack(f'{bo}I', 0))
    out.extend(overflow_buf)
    out.extend(pixel_bytes)
    return bytes(out)


class TestResolveSampleFormat:
    """Unit tests for the helper."""

    def test_scalar(self):
        assert resolve_sample_format(3) == 3

    def test_one_element_tuple(self):
        assert resolve_sample_format((1,)) == 1

    def test_uniform_tuple(self):
        assert resolve_sample_format((3, 3, 3)) == 3

    def test_uniform_list(self):
        assert resolve_sample_format([2, 2, 2, 2]) == 2

    def test_mixed_tuple_raises(self):
        with pytest.raises(ValueError, match=r"Mixed SampleFormat"):
            resolve_sample_format((3, 3, 1))

    def test_error_message_contains_values(self):
        with pytest.raises(ValueError) as exc:
            resolve_sample_format((3, 3, 1))
        msg = str(exc.value)
        assert "(3, 3, 1)" in msg
        assert "gdal_translate" in msg or "rasterio" in msg

    def test_mixed_signed_unsigned_raises(self):
        with pytest.raises(ValueError, match=r"Mixed SampleFormat"):
            resolve_sample_format((1, 2, 1))

    def test_empty_tuple_falls_back_to_default(self):
        # Issue #1661 regression: empty SampleFormat from malformed TIFFs
        # must not raise IndexError. Falling back to 1 (uint) is intentional.
        assert resolve_sample_format(()) == 1


class TestMixedSampleFormatTiff:
    """End-to-end tests against open_geotiff."""

    def test_uniform_sample_format_reads_fine(self, tmp_path):
        path = tmp_path / "uniform_sf_1868.tif"
        path.write_bytes(
            _build_multi_band_tiff_sf(
                width=4, height=3, samples=3,
                sample_format=(3, 3, 3),
                bits_per_sample=32,
                pixel_dtype=np.dtype('float32'),
            )
        )
        da = open_geotiff(str(path))
        assert da.dtype == np.float32
        assert da.sizes['y'] == 3
        assert da.sizes['x'] == 4
        assert da.sizes['band'] == 3

    def test_mixed_float_uint_rejected(self, tmp_path):
        """The canonical silent-corruption case: two float bands plus one
        uint band at the same bit depth. Previously decoded as float32
        across all bands."""
        path = tmp_path / "mixed_sf_1868.tif"
        path.write_bytes(
            _build_multi_band_tiff_sf(
                width=4, height=3, samples=3,
                sample_format=(3, 3, 1),
                bits_per_sample=32,
                pixel_dtype=np.dtype('float32'),
            )
        )
        with pytest.raises(ValueError) as exc:
            open_geotiff(str(path))
        msg = str(exc.value)
        assert "Mixed SampleFormat" in msg
        assert "(3, 3, 1)" in msg

    def test_mixed_signed_unsigned_rejected(self, tmp_path):
        """Mixed signed/unsigned integer bands at the same bit depth would
        also corrupt silently."""
        path = tmp_path / "mixed_sf_int_1868.tif"
        path.write_bytes(
            _build_multi_band_tiff_sf(
                width=4, height=3, samples=3,
                sample_format=(1, 2, 1),
                bits_per_sample=16,
                pixel_dtype=np.dtype('uint16'),
            )
        )
        with pytest.raises(ValueError, match=r"Mixed SampleFormat"):
            open_geotiff(str(path))
