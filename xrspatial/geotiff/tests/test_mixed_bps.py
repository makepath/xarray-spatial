"""Tests for non-uniform BitsPerSample handling (issue #1505).

A TIFF whose BitsPerSample tag carries different values per band
(e.g. ``(16, 16, 16, 8)`` for RGB plus an 8-bit alpha) cannot be
decoded into a single numpy dtype. xarray-spatial rejects such files
with a clear error so the user can convert them with GDAL/rasterio
instead of silently getting garbage in the mismatched bands.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._dtypes import resolve_bits_per_sample


def _build_multi_band_tiff(
    width: int,
    height: int,
    samples: int,
    bits_per_sample,
    pixel_dtype: np.dtype = np.dtype('uint16'),
) -> bytes:
    """Build a minimal stripped uncompressed multi-band TIFF.

    ``bits_per_sample`` is written as-is into tag 258 — pass a list/tuple
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
    if isinstance(bits_per_sample, (list, tuple)):
        add_shorts(258, list(bits_per_sample))     # BitsPerSample (per band)
    else:
        add_short(258, bits_per_sample)
    add_short(259, 1)                              # Compression = none
    add_short(262, 2 if samples >= 3 else 1)       # PhotometricInterpretation
    add_long(273, 0)                               # StripOffsets (patched)
    add_short(277, samples)                        # SamplesPerPixel
    add_short(278, height)                         # RowsPerStrip
    add_long(279, len(pixel_bytes))                # StripByteCounts
    add_short(284, 1)                              # PlanarConfiguration = chunky
    add_shorts(339, [1] * samples)                 # SampleFormat = uint

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

    # Re-serialize the IFD with final layout
    out = bytearray()
    out.extend(b'II')                              # little-endian
    out.extend(struct.pack(f'{bo}H', 42))          # magic
    out.extend(struct.pack(f'{bo}I', ifd_start))   # offset of first IFD
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
    out.extend(struct.pack(f'{bo}I', 0))           # next IFD = 0
    out.extend(overflow_buf)
    out.extend(pixel_bytes)
    return bytes(out)


class TestResolveBitsPerSample:
    """Unit tests for the helper itself."""

    def test_scalar(self):
        assert resolve_bits_per_sample(16) == 16

    def test_one_element_tuple(self):
        assert resolve_bits_per_sample((8,)) == 8

    def test_uniform_tuple(self):
        assert resolve_bits_per_sample((16, 16, 16)) == 16

    def test_uniform_list(self):
        assert resolve_bits_per_sample([32, 32, 32, 32]) == 32

    def test_mixed_tuple_raises(self):
        with pytest.raises(ValueError, match=r"Mixed BitsPerSample"):
            resolve_bits_per_sample((16, 16, 16, 8))

    def test_error_message_contains_values(self):
        with pytest.raises(ValueError) as exc:
            resolve_bits_per_sample((16, 16, 16, 8))
        msg = str(exc.value)
        assert "(16, 16, 16, 8)" in msg
        assert "gdal_translate" in msg

    def test_error_message_ot_matches_widest_bps(self):
        """gdal_translate hint should suggest a type wide enough for input."""
        # 32-bit + 8-bit alpha -> widest is 32, default sample format = uint
        with pytest.raises(ValueError) as exc:
            resolve_bits_per_sample((32, 32, 32, 8))
        assert "-ot UInt32" in str(exc.value)

        # Widest is 16-bit -> UInt16 (the original hard-coded suggestion).
        with pytest.raises(ValueError) as exc:
            resolve_bits_per_sample((16, 16, 16, 8))
        assert "-ot UInt16" in str(exc.value)

    def test_error_message_ot_uses_sample_format_hint(self):
        """sample_format=3 (float) at 32-bit -> Float32 instead of UInt32."""
        with pytest.raises(ValueError) as exc:
            resolve_bits_per_sample((32, 32, 32, 8), sample_format=3)
        assert "-ot Float32" in str(exc.value)

        # sample_format=2 (int) at 16-bit -> Int16 instead of UInt16.
        with pytest.raises(ValueError) as exc:
            resolve_bits_per_sample((16, 16, 8), sample_format=2)
        assert "-ot Int16" in str(exc.value)

    def test_empty_tuple_raises(self):
        with pytest.raises(ValueError):
            resolve_bits_per_sample(())


class TestMixedBitsPerSampleTiff:
    """End-to-end tests against open_geotiff."""

    def test_uniform_bps_reads_fine(self, tmp_path):
        path = tmp_path / "uniform_rgba.tif"
        path.write_bytes(
            _build_multi_band_tiff(
                width=4, height=3, samples=4,
                bits_per_sample=(16, 16, 16, 16),
                pixel_dtype=np.dtype('uint16'),
            )
        )
        da = open_geotiff(str(path))
        assert da.dtype == np.uint16
        # Multi-band TIFFs come back as (y, x, band)
        assert da.sizes['y'] == 3
        assert da.sizes['x'] == 4
        assert da.sizes['band'] == 4

    def test_mixed_bps_rgb_plus_8bit_alpha_rejected(self, tmp_path):
        """RGB+8-bit-alpha is the canonical case from issue #1505."""
        path = tmp_path / "mixed_rgba.tif"
        # NB: the pixel block here is uint16 throughout; the test only
        # exercises the dispatch, not the (impossible) decode path.
        path.write_bytes(
            _build_multi_band_tiff(
                width=4, height=3, samples=4,
                bits_per_sample=(16, 16, 16, 8),
                pixel_dtype=np.dtype('uint16'),
            )
        )
        with pytest.raises(ValueError) as exc:
            open_geotiff(str(path))
        msg = str(exc.value)
        assert "(16, 16, 16, 8)" in msg
        assert "Mixed BitsPerSample" in msg
