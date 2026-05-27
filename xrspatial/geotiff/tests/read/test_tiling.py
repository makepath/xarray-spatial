"""Tile / strip decoder paths: byte caps, dtype dispatch, planar layout,
sub-byte unpack, parallel-decode gate.

Sections cover:

* Tiled / stripped byte-cap defenses (sections 1-2).
* ``resolve_bits_per_sample`` unit cases and end-to-end mixed-BPS
  rejection.
* ``resolve_sample_format`` unit cases and end-to-end mixed-SampleFormat
  rejection.
* Vectorised sub-byte unpack vs a loop-based reference.
* planar=2 strip table truncation.
* CPU + GPU planar / layout / band / dtype matrix.
* Parallel tile decode gate at the default tile_size.
* Parallel strip decode on local + HTTP COG paths, planar=1 and planar=2.
"""
from __future__ import annotations

import concurrent.futures
import http.server
import importlib.util
import io
import math
import os
import socket
import struct
import tempfile
import threading
from unittest.mock import patch  # used by section 9 parallel-strip tests

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _decode as _decode_mod
from xrspatial.geotiff import _reader as _reader_mod
from xrspatial.geotiff import open_geotiff, read_geotiff_gpu, to_geotiff
from xrspatial.geotiff._compression import COMPRESSION_NONE, unpack_bits
from xrspatial.geotiff._dtypes import (resolve_bits_per_sample, resolve_sample_format,
                                       tiff_dtype_to_numpy)
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writer import (_assemble_cog_layout, _assemble_standard_layout,
                                       _assemble_tiff, _write_stripped, _write_tiled)

from .._helpers.markers import requires_gpu as _gpu_only
from .._helpers.markers import requires_loopback
from .._helpers.tiff_builders import make_minimal_tiff
from .._helpers.tiff_surgery import patch_byte_counts as _patch_byte_counts

# ---------------------------------------------------------------------------
# Section 1 helpers: forged tiled / stripped tiffs for byte-cap tests
# ---------------------------------------------------------------------------


def _build_forged_tiled_cog(tmp_path, byte_count_value: int,
                            *, basename: str = "forged_tiles") -> str:
    """Write a real tiled COG, patch every TileByteCounts entry, return path."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    da = xr.DataArray(arr, dims=['y', 'x'])
    path = str(tmp_path / f"{basename}.tif")
    to_geotiff(da, path, tile_size=32, compression='deflate')
    with open(path, 'rb') as f:
        data = bytearray(f.read())
    _patch_byte_counts(data, 325, byte_count_value)  # 325 = TileByteCounts
    with open(path, 'wb') as f:
        f.write(data)
    return path


def _build_forged_stripped_tif(tmp_path, byte_count_value: int) -> str:
    """Write a strip-organized TIFF, patch every StripByteCounts entry."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    da = xr.DataArray(arr, dims=['y', 'x'])
    path = str(tmp_path / "forged_strips.tif")
    to_geotiff(da, path, tiled=False, compression='deflate')
    with open(path, 'rb') as f:
        data = bytearray(f.read())
    _patch_byte_counts(data, 279, byte_count_value)  # 279 = StripByteCounts
    with open(path, 'wb') as f:
        f.write(data)
    return path


# ---------------------------------------------------------------------------
# Section 1: tiled local byte cap
# ---------------------------------------------------------------------------


class TestLocalTileByteCap:
    def test_huge_tile_byte_count_rejected(self, tmp_path, monkeypatch):
        """A local tile with a huge TileByteCount raises before decode."""
        path = _build_forged_tiled_cog(tmp_path, 100 * 1024 * 1024)
        monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', str(1024 * 1024))

        with pytest.raises(ValueError, match="TileByteCount"):
            open_geotiff(path)

    def test_error_message_names_value_and_cap(self, tmp_path, monkeypatch):
        path = _build_forged_tiled_cog(tmp_path, 50 * 1024 * 1024)
        monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', str(1024))

        with pytest.raises(ValueError) as excinfo:
            open_geotiff(path)
        msg = str(excinfo.value)
        assert "52,428,800" in msg or "52428800" in msg
        assert "1,024" in msg or "1024" in msg
        assert "denial-of-service" in msg.lower() or "malformed" in msg

    def test_normal_local_cog_under_default_cap(self, tmp_path):
        """Legitimate local reads with the default cap still succeed."""
        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        da = xr.DataArray(arr, dims=['y', 'x'])
        path = str(tmp_path / "normal_local.tif")
        to_geotiff(da, path, tile_size=32, compression='deflate')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_env_override_lifts_cap(self, tmp_path, monkeypatch):
        """A user with legitimate large tiles can lift the cap via env."""
        path = _build_forged_tiled_cog(tmp_path, 50 * 1024 * 1024)
        monkeypatch.setenv(
            'XRSPATIAL_COG_MAX_TILE_BYTES', str(64 * 1024 * 1024))

        try:
            open_geotiff(path)
        except ValueError as e:
            assert "exceeds the per-tile safety cap" not in str(e)


class TestLocalStripByteCap:
    def test_huge_strip_byte_count_rejected(self, tmp_path, monkeypatch):
        path = _build_forged_stripped_tif(tmp_path, 100 * 1024 * 1024)
        monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', str(1024 * 1024))

        with pytest.raises(ValueError, match="StripByteCount"):
            open_geotiff(path)

    def test_strip_error_message_mentions_strip(self, tmp_path, monkeypatch):
        path = _build_forged_stripped_tif(tmp_path, 50 * 1024 * 1024)
        monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', str(2048))

        with pytest.raises(ValueError) as excinfo:
            open_geotiff(path)
        msg = str(excinfo.value)
        assert "strip" in msg.lower()
        assert "safety cap" in msg.lower()


# ---------------------------------------------------------------------------
# Cap helper directly
# ---------------------------------------------------------------------------


def test_max_tile_bytes_env_negative_falls_back(monkeypatch):
    """Negative env value falls back to the default, not a 1-byte cap."""
    monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', '-5')
    assert (
        _reader_mod._max_tile_bytes_from_env()
        == _reader_mod.MAX_TILE_BYTES_DEFAULT
    )


def test_max_tile_bytes_env_zero_falls_back(monkeypatch):
    """Zero env value falls back to the default for the same reason."""
    monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', '0')
    assert (
        _reader_mod._max_tile_bytes_from_env()
        == _reader_mod.MAX_TILE_BYTES_DEFAULT
    )


def test_max_tile_bytes_env_garbage_falls_back(monkeypatch):
    monkeypatch.setenv('XRSPATIAL_COG_MAX_TILE_BYTES', 'not-a-number')
    assert (
        _reader_mod._max_tile_bytes_from_env()
        == _reader_mod.MAX_TILE_BYTES_DEFAULT
    )


# ---------------------------------------------------------------------------
# Section 2: GPU eager / chunked byte cap
# ---------------------------------------------------------------------------


class TestGpuTileByteCap:
    @_gpu_only
    def test_huge_tile_byte_count_rejected(self, tmp_path, monkeypatch):
        """A local tile with a huge TileByteCount raises before GPU decode."""
        path = _build_forged_tiled_cog(
            tmp_path, 100 * 1024 * 1024, basename="forged_gpu_tiles")
        monkeypatch.setenv("XRSPATIAL_COG_MAX_TILE_BYTES", str(1024 * 1024))

        with pytest.raises(ValueError, match="TileByteCount"):
            read_geotiff_gpu(path)

    @_gpu_only
    def test_error_message_names_value_and_cap(self, tmp_path, monkeypatch):
        path = _build_forged_tiled_cog(
            tmp_path, 50 * 1024 * 1024, basename="forged_gpu_tiles_msg")
        monkeypatch.setenv("XRSPATIAL_COG_MAX_TILE_BYTES", str(1024))

        with pytest.raises(ValueError) as excinfo:
            read_geotiff_gpu(path)
        msg = str(excinfo.value)
        assert "52,428,800" in msg or "52428800" in msg
        assert "1,024" in msg or "1024" in msg
        assert "denial-of-service" in msg.lower() or "malformed" in msg

    @_gpu_only
    def test_normal_gpu_read_under_default_cap(self, tmp_path):
        """Legitimate GPU reads with the default cap still succeed."""
        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        da = xr.DataArray(arr, dims=["y", "x"])
        path = str(tmp_path / "normal_gpu.tif")
        to_geotiff(da, path, tile_size=32, compression="deflate")

        result = read_geotiff_gpu(path)
        np.testing.assert_array_equal(result.data.get(), arr)

    @_gpu_only
    def test_env_override_lifts_cap(self, tmp_path, monkeypatch):
        """A user with legitimate large tiles can lift the cap via env."""
        path = _build_forged_tiled_cog(
            tmp_path, 50 * 1024 * 1024, basename="forged_gpu_tiles_override")
        monkeypatch.setenv(
            "XRSPATIAL_COG_MAX_TILE_BYTES", str(64 * 1024 * 1024))

        try:
            read_geotiff_gpu(path)
        except Exception as exc:
            assert "exceeds the per-tile safety cap" not in str(exc), (
                "cap loop fired despite the env override lifting the cap"
            )


class TestGpuChunkedTileByteCap:
    @_gpu_only
    def test_chunked_huge_tile_byte_count_rejected(
            self, tmp_path, monkeypatch):
        """Sibling check on the dask + GPU chunked path."""
        path = _build_forged_tiled_cog(
            tmp_path, 100 * 1024 * 1024, basename="forged_gpu_chunked")
        monkeypatch.setenv(
            "XRSPATIAL_COG_MAX_TILE_BYTES", str(1024 * 1024))

        with pytest.raises(ValueError, match="TileByteCount"):
            read_geotiff_gpu(path, chunks=32)


# ---------------------------------------------------------------------------
# Section 3 helpers: hand-rolled multi-band stripped TIFF for BPS / SF tests
# ---------------------------------------------------------------------------


def _finalize_minimal_ifd(
        tags: list[tuple[int, int, int, bytes]],
        pixel_bytes: bytes,
        bo: str = '<') -> bytes:
    """Serialise a tag list + pixel block into a one-IFD TIFF.

    Patches the StripOffsets tag (273, single-entry) to point at the
    pixel block. Shared by the BPS and SampleFormat builders so the
    layout logic only lives once.
    """
    tags.sort(key=lambda t: t[0])

    num_entries = len(tags)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4
    overflow_start = ifd_start + ifd_size

    overflow_buf = bytearray()
    tag_overflow_offsets: dict[int, int | None] = {}
    for tag, _typ, _count, raw in tags:
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


def _build_multi_band_tiff_bps(
    width: int,
    height: int,
    samples: int,
    bits_per_sample,
    pixel_dtype: np.dtype = np.dtype('uint16'),
) -> bytes:
    """Build a minimal stripped uncompressed multi-band TIFF.

    ``bits_per_sample`` is written as-is into tag 258 -- pass a list/tuple
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

    return _finalize_minimal_ifd(tags, pixel_bytes, bo)


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

    return _finalize_minimal_ifd(tags, pixel_bytes, bo)


# ---------------------------------------------------------------------------
# Section 3: mixed BitsPerSample dispatch
# ---------------------------------------------------------------------------


class TestResolveBitsPerSample:
    """Unit tests for the helper itself."""

    def test_resolve_bps_scalar(self):
        assert resolve_bits_per_sample(16) == 16

    def test_resolve_bps_one_element_tuple(self):
        assert resolve_bits_per_sample((8,)) == 8

    @pytest.mark.parametrize(
        "vals,expected",
        [((16, 16, 16), 16), ([32, 32, 32, 32], 32)],
        ids=["tuple", "list"],
    )
    def test_resolve_bps_uniform(self, vals, expected):
        assert resolve_bits_per_sample(vals) == expected

    def test_resolve_bps_mixed_tuple_raises(self):
        with pytest.raises(ValueError, match=r"Mixed BitsPerSample"):
            resolve_bits_per_sample((16, 16, 16, 8))

    def test_resolve_bps_error_message_contains_values(self):
        with pytest.raises(ValueError) as exc:
            resolve_bits_per_sample((16, 16, 16, 8))
        msg = str(exc.value)
        assert "(16, 16, 16, 8)" in msg
        assert "gdal_translate" in msg

    @pytest.mark.parametrize(
        "vals,expected_token",
        [
            ((32, 32, 32, 8), "-ot UInt32"),
            ((16, 16, 16, 8), "-ot UInt16"),
        ],
        ids=["uint32", "uint16"],
    )
    def test_resolve_bps_error_message_ot_matches_widest(
            self, vals, expected_token):
        """gdal_translate hint should suggest a type wide enough for input."""
        with pytest.raises(ValueError) as exc:
            resolve_bits_per_sample(vals)
        assert expected_token in str(exc.value)

    @pytest.mark.parametrize(
        "vals,sample_format,expected_token",
        [
            ((32, 32, 32, 8), 3, "-ot Float32"),
            ((16, 16, 8), 2, "-ot Int16"),
        ],
        ids=["float32", "int16"],
    )
    def test_resolve_bps_error_message_ot_uses_sample_format_hint(
            self, vals, sample_format, expected_token):
        """sample_format=3 (float) at 32-bit -> Float32 instead of UInt32."""
        with pytest.raises(ValueError) as exc:
            resolve_bits_per_sample(vals, sample_format=sample_format)
        assert expected_token in str(exc.value)

    def test_resolve_bps_empty_tuple_raises(self):
        with pytest.raises(ValueError):
            resolve_bits_per_sample(())


class TestMixedBitsPerSampleTiff:
    """End-to-end tests against open_geotiff."""

    def test_mixed_bps_uniform_reads_fine(self, tmp_path):
        path = tmp_path / "uniform_rgba_2429.tif"
        path.write_bytes(
            _build_multi_band_tiff_bps(
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
        """RGB+8-bit-alpha is the canonical mixed-BPS case."""
        path = tmp_path / "mixed_rgba_2429.tif"
        # NB: the pixel block here is uint16 throughout; the test only
        # exercises the dispatch, not the (impossible) decode path.
        path.write_bytes(
            _build_multi_band_tiff_bps(
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


# ---------------------------------------------------------------------------
# Section 4: mixed SampleFormat dispatch
# ---------------------------------------------------------------------------


class TestResolveSampleFormat:
    """Unit tests for the helper."""

    def test_resolve_sf_scalar(self):
        assert resolve_sample_format(3) == 3

    def test_resolve_sf_one_element_tuple(self):
        assert resolve_sample_format((1,)) == 1

    @pytest.mark.parametrize(
        "vals,expected",
        [((3, 3, 3), 3), ([2, 2, 2, 2], 2)],
        ids=["tuple", "list"],
    )
    def test_resolve_sf_uniform(self, vals, expected):
        assert resolve_sample_format(vals) == expected

    def test_resolve_sf_mixed_tuple_raises(self):
        with pytest.raises(ValueError, match=r"Mixed SampleFormat"):
            resolve_sample_format((3, 3, 1))

    def test_resolve_sf_error_message_contains_values(self):
        with pytest.raises(ValueError) as exc:
            resolve_sample_format((3, 3, 1))
        msg = str(exc.value)
        assert "(3, 3, 1)" in msg
        assert "gdal_translate" in msg or "rasterio" in msg

    def test_resolve_sf_mixed_signed_unsigned_raises(self):
        with pytest.raises(ValueError, match=r"Mixed SampleFormat"):
            resolve_sample_format((1, 2, 1))

    def test_resolve_sf_empty_tuple_falls_back_to_default(self):
        # Empty SampleFormat from malformed TIFFs must not raise
        # IndexError. Falling back to 1 (uint) is intentional.
        assert resolve_sample_format(()) == 1


class TestMixedSampleFormatTiff:
    """End-to-end tests against open_geotiff."""

    def test_mixed_sample_format_uniform_reads_fine(self, tmp_path):
        path = tmp_path / "uniform_sf_2429.tif"
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

    def test_mixed_sample_format_float_uint_rejected(self, tmp_path):
        """The canonical silent-corruption case: two float bands plus one
        uint band at the same bit depth. Previously decoded as float32
        across all bands."""
        path = tmp_path / "mixed_sf_2429.tif"
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

    def test_mixed_sample_format_signed_unsigned_rejected(self, tmp_path):
        """Mixed signed/unsigned integer bands at the same bit depth would
        also corrupt silently."""
        path = tmp_path / "mixed_sf_int_2429.tif"
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


# ---------------------------------------------------------------------------
# Section 5: sub-byte BPS unpack
# ---------------------------------------------------------------------------


def _reference_unpack_bits(data: np.ndarray, bps: int,
                           pixel_count: int) -> np.ndarray:
    """Bit-for-bit copy of the original loop-based implementation.

    Kept here (rather than imported) so the test survives any future
    deletion of the loop-based code path.
    """
    if bps == 1:
        out = np.unpackbits(data)[:pixel_count]
        return out.astype(np.uint8)
    if bps == 2:
        out = np.empty(pixel_count, dtype=np.uint8)
        for i in range(min(len(data), (pixel_count + 3) // 4)):
            b = data[i]
            base = i * 4
            if base < pixel_count:
                out[base] = (b >> 6) & 0x03
            if base + 1 < pixel_count:
                out[base + 1] = (b >> 4) & 0x03
            if base + 2 < pixel_count:
                out[base + 2] = (b >> 2) & 0x03
            if base + 3 < pixel_count:
                out[base + 3] = b & 0x03
        return out
    if bps == 4:
        out = np.empty(pixel_count, dtype=np.uint8)
        for i in range(min(len(data), (pixel_count + 1) // 2)):
            b = data[i]
            base = i * 2
            if base < pixel_count:
                out[base] = (b >> 4) & 0x0F
            if base + 1 < pixel_count:
                out[base + 1] = b & 0x0F
        return out
    if bps == 12:
        out = np.empty(pixel_count, dtype=np.uint16)
        n_pairs = pixel_count // 2
        remainder = pixel_count % 2
        for i in range(n_pairs):
            off = i * 3
            if off + 2 < len(data):
                b0 = int(data[off])
                b1 = int(data[off + 1])
                b2 = int(data[off + 2])
                out[i * 2] = (b0 << 4) | (b1 >> 4)
                out[i * 2 + 1] = ((b1 & 0x0F) << 8) | b2
        if remainder and n_pairs * 3 + 1 < len(data):
            off = n_pairs * 3
            out[pixel_count - 1] = (
                (int(data[off]) << 4) | (int(data[off + 1]) >> 4)
            )
        return out
    raise ValueError(f"Unsupported bps: {bps}")


def _written_positions(bps: int, pixel_count: int, data_len: int) -> set:
    """Return the indices the original loop *wrote to*.

    The pre-vectorisation code used ``np.empty`` and then conditionally
    skipped writes when the input buffer was too short. The new code
    must agree on positions that the old code wrote; positions that
    were never written are pure ``np.empty`` garbage and must not be
    compared. This helper enumerates the written set so the test can
    stick to it.
    """
    positions: set[int] = set()
    if bps == 2:
        n_bytes = min(data_len, (pixel_count + 3) // 4)
        for i in range(n_bytes):
            base = i * 4
            for n in range(4):
                if base + n < pixel_count:
                    positions.add(base + n)
    elif bps == 4:
        n_bytes = min(data_len, (pixel_count + 1) // 2)
        for i in range(n_bytes):
            base = i * 2
            for n in range(2):
                if base + n < pixel_count:
                    positions.add(base + n)
    elif bps == 12:
        n_pairs = pixel_count // 2
        for i in range(n_pairs):
            off = i * 3
            if off + 2 < data_len:
                positions.add(i * 2)
                positions.add(i * 2 + 1)
        rem = pixel_count % 2
        if rem and n_pairs * 3 + 1 < data_len:
            positions.add(pixel_count - 1)
    elif bps == 1:
        # bps=1 covers every position via np.unpackbits.
        positions.update(range(pixel_count))
    return positions


@pytest.mark.parametrize("bps", [2, 4, 12])
@pytest.mark.parametrize(
    "pixel_count", [0, 1, 2, 3, 4, 7, 8, 100, 10_000])
@pytest.mark.parametrize("data_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_unpack_bits_matches_reference(bps, pixel_count, data_factor):
    """Vectorised output equals the original for every covered position."""
    if bps == 2:
        bytes_per_pixel = 0.25
    elif bps == 4:
        bytes_per_pixel = 0.5
    else:  # bps == 12
        bytes_per_pixel = 1.5

    required = int(np.ceil(pixel_count * bytes_per_pixel))
    n_bytes = max(0, int(required * data_factor))
    rng = np.random.default_rng(seed=bps * 10_000 + pixel_count)
    data = rng.integers(0, 256, size=n_bytes, dtype=np.uint8)

    ref = _reference_unpack_bits(data, bps, pixel_count)
    new = unpack_bits(data, bps, pixel_count)

    assert ref.shape == new.shape
    assert ref.dtype == new.dtype

    for p in _written_positions(bps, pixel_count, len(data)):
        assert ref[p] == new[p], (
            f"bps={bps} pc={pixel_count} data_factor={data_factor}: "
            f"position {p} differs ref={ref[p]} new={new[p]}"
        )


def test_unpack_bits_bps1_unchanged():
    """bps=1 still routes through ``np.unpackbits`` and returns uint8."""
    data = np.array([0b10101100, 0b00001111], dtype=np.uint8)
    out = unpack_bits(data, 1, 16)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(
        out,
        np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                 dtype=np.uint8),
    )


def test_unpack_bits_bps12_three_byte_buffer_decodes_one_pair():
    """bps=12 with exactly 3 input bytes writes the single pair.

    The original loop's guard was ``off + 2 < len(data)``, which is
    satisfied for ``off=0, len(data)=3``. The vectorised implementation
    must keep the same decision and not skip the pair (this was the
    boundary case that surfaced during the rewrite).
    """
    # Two 12-bit values: 0x123 and 0x456 packed MSB-first into 3 bytes.
    data = np.array([0x12, 0x34, 0x56], dtype=np.uint8)
    out = unpack_bits(data, 12, 2)
    assert tuple(int(v) for v in out) == (0x123, 0x456)


def test_unpack_bits_bps12_two_byte_buffer_no_pair_decoded():
    """A 2-byte buffer cannot satisfy ``off+2 < 2`` so no pair is written.

    Mirrors the original loop semantics. ``np.empty`` initial garbage
    at those positions is fine -- the test only asserts that the
    function does not crash and returns an array of the right shape.
    """
    data = np.array([0x12, 0x34], dtype=np.uint8)
    out = unpack_bits(data, 12, 2)
    assert out.shape == (2,)
    assert out.dtype == np.uint16


def test_unpack_bits_unsupported_bps_raises():
    """Unknown sub-byte bps still raises a clear ValueError."""
    with pytest.raises(ValueError, match="Unsupported"):
        unpack_bits(np.zeros(10, dtype=np.uint8), 3, 10)


# ---------------------------------------------------------------------------
# Section 6 helpers: hand-rolled planar=2 stripped TIFF
# ---------------------------------------------------------------------------


def _make_planar2_stripped_tiff(
    width: int,
    height: int,
    bands: int,
    data: np.ndarray,
    *,
    rows_per_strip: int,
    truncate_strip_table_to: int | None = None,
) -> bytes:
    """Build an uncompressed ``PlanarConfiguration=2`` stripped TIFF.

    ``data`` is shaped ``(bands, height, width)``. ``rows_per_strip``
    controls how many image rows each strip covers (per band).

    If ``truncate_strip_table_to`` is set, both ``StripOffsets`` (273)
    and ``StripByteCounts`` (279) are written with only that many entries
    instead of the full ``strips_per_band * bands`` count. This is the
    strip-table truncation corruption pattern.
    """
    bo = '<'
    assert data.shape == (bands, height, width)
    dtype = data.dtype
    bps = dtype.itemsize * 8
    sf = 1  # unsigned int

    strips_per_band = math.ceil(height / rows_per_strip)

    # planar=2: every strip for band 0, then every strip for band 1, ...
    strip_blobs: list[bytes] = []
    for b in range(bands):
        for s in range(strips_per_band):
            r0 = s * rows_per_strip
            r1 = min(r0 + rows_per_strip, height)
            strip = data[b, r0:r1, :]
            strip_blobs.append(strip.tobytes())

    pixel_bytes = b''.join(strip_blobs)
    full_byte_counts = [len(b) for b in strip_blobs]
    full_num = len(strip_blobs)

    # The strip table that gets *written* to the header. When
    # ``truncate_strip_table_to`` is set, the header advertises fewer
    # entries than the file actually contains.
    if truncate_strip_table_to is None:
        table_len = full_num
    else:
        table_len = truncate_strip_table_to
    table_byte_counts = full_byte_counts[:table_len]

    tag_list: list[tuple[int, int, int, bytes]] = []

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_shorts(tag, vals):
        tag_list.append(
            (tag, 3, len(vals), struct.pack(f'{bo}{len(vals)}H', *vals))
        )

    def add_longs(tag, vals):
        tag_list.append(
            (tag, 4, len(vals), struct.pack(f'{bo}{len(vals)}I', *vals))
        )

    add_short(256, width)
    add_short(257, height)
    add_shorts(258, [bps] * bands)            # BitsPerSample
    add_short(259, 1)                         # Compression = none
    add_short(262, 2 if bands >= 3 else 1)    # Photometric (RGB or BlackIsZero)
    add_longs(273, [0] * table_len)           # StripOffsets placeholder
    add_short(277, bands)                     # SamplesPerPixel
    add_short(278, rows_per_strip)            # RowsPerStrip
    add_longs(279, table_byte_counts)         # StripByteCounts (matches table_len)
    add_short(284, 2)                         # PlanarConfiguration = Separate
    add_shorts(339, [sf] * bands)             # SampleFormat per band

    tag_list.sort(key=lambda t: t[0])

    num_entries = len(tag_list)
    ifd_start = 8
    ifd_size = 2 + 12 * num_entries + 4

    # First pass: compute overflow size and the pixel-data start.
    overflow_buf = bytearray()
    for _tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
    overflow_start = ifd_start + ifd_size
    pixel_data_start = overflow_start + len(overflow_buf)

    # Patch StripOffsets with real byte positions for the *advertised*
    # entries only. (When truncated, only the first ``table_len`` strips
    # are addressable; the rest of the pixel bytes still sit in the file
    # but are unreachable through the strip table.)
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:
            offs = []
            pos = 0
            for i in range(table_len):
                offs.append(pixel_data_start + pos)
                pos += full_byte_counts[i]
            new_raw = struct.pack(f'{bo}{table_len}I', *offs)
            patched.append((tag, typ, count, new_raw))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    overflow_buf = bytearray()
    tag_offsets: dict[int, int | None] = {}
    for tag, _typ, _count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    out = bytearray()
    out.extend(b'II')
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            out.extend(raw.ljust(4, b'\x00'))
        else:
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    out.extend(struct.pack(f'{bo}I', 0))  # next IFD = none
    out.extend(overflow_buf)
    out.extend(pixel_bytes)
    return bytes(out)


def _rgb_4x4() -> np.ndarray:
    """Deterministic 3-band 4x4 uint8 raster shaped (3, 4, 4)."""
    rng = np.random.RandomState(1782)
    return rng.randint(0, 256, size=(3, 4, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Section 6: planar=2 strip table truncation
# ---------------------------------------------------------------------------


def test_planar_strip_table_truncated_raises_typed_error():
    """A planar=2 strip table holding only one band's worth of strips
    must raise ``ValueError`` naming the planar layout and the expected
    entry count, instead of silently returning a partially initialised
    buffer.
    """
    data = _rgb_4x4()
    rps = 2
    strips_per_band = math.ceil(data.shape[1] / rps)  # = 2
    # Advertise only one band's worth of strips (2 entries) instead of
    # the required strips_per_band * bands = 6.
    tiff = _make_planar2_stripped_tiff(
        width=4,
        height=4,
        bands=3,
        data=data,
        rows_per_strip=rps,
        truncate_strip_table_to=strips_per_band,
    )

    with pytest.raises(ValueError) as exc:
        open_geotiff(io.BytesIO(tiff))

    msg = str(exc.value).lower()
    assert "planar" in msg, f"error should mention planar layout: {exc.value!r}"
    # The error must surface the *real* expected count, not the chunky one.
    assert "6" in str(exc.value), (
        f"error should name the planar expected count (6 = 2 strips x 3 "
        f"samples): {exc.value!r}"
    )


def test_planar_strip_table_complete_reads_correct_pixels():
    """A correctly-formed planar=2 RGB stripped TIFF must read back the
    per-band pixel values it was written with. Guards against the planar
    branch losing pixels once the pre-flight check is tightened.
    """
    data = _rgb_4x4()
    tiff = _make_planar2_stripped_tiff(
        width=4,
        height=4,
        bands=3,
        data=data,
        rows_per_strip=2,
        truncate_strip_table_to=None,  # full strip table
    )

    da = open_geotiff(io.BytesIO(tiff))
    arr = np.asarray(da)

    # The CPU reader returns (y, x, band) for multi-band stripped files.
    # Compare against the writer-side (band, y, x) layout transposed the
    # same way.
    expected = np.transpose(data, (1, 2, 0))
    if arr.ndim == 3 and arr.shape == expected.shape:
        np.testing.assert_array_equal(arr, expected)
    elif arr.ndim == 3 and arr.shape == data.shape:
        np.testing.assert_array_equal(arr, data)
    else:
        raise AssertionError(
            f"unexpected output shape {arr.shape}; expected one of "
            f"{expected.shape} (y,x,band) or {data.shape} (band,y,x)"
        )


def test_chunky_single_band_truncated_strip_table_still_raises():
    """Regression guard for the non-planar path. A chunky (planar=1)
    single-band TIFF whose strip table is shorter than ``ceil(height/rps)``
    must keep raising the existing typed ``ValueError`` after the
    planar-aware tightening.
    """
    # Reuse the project's minimal-TIFF builder, then truncate its
    # ``StripByteCounts`` count field. ``test_fuzz_hypothesis`` uses
    # the same byte offset (110) for this corruption.
    base = make_minimal_tiff(4, 4, np.dtype('float32'))
    mut = bytearray(base)
    mut[110] = 0  # zero StripByteCounts count -> truncates the strip table

    with pytest.raises(ValueError):
        open_geotiff(io.BytesIO(bytes(mut)))


# ---------------------------------------------------------------------------
# Section 7 helpers: planar config x layout x bands x dtype matrix
# ---------------------------------------------------------------------------


# ``tifffile`` is the only viable way to control planarconfig + tile/strip
# independently for these matrix tests, so the section is gated on it.
_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None
_requires_tifffile = pytest.mark.skipif(
    not _HAS_TIFFFILE, reason="tifffile required for planar / layout matrix")


def _make_planar_matrix_data(
        bands: int, height: int, width: int, dtype) -> np.ndarray:
    """Deterministic random multi-band raster shaped (bands, height, width)."""
    rng = np.random.RandomState(
        0xA2A3 + bands * 1000 + height + np.dtype(dtype).itemsize)
    info = np.iinfo(dtype)
    high = min(int(info.max), 60_000) + 1
    return rng.randint(0, high, size=(bands, height, width)).astype(dtype)


def _write_planar_matrix_tiff(
        path: str, data: np.ndarray, *, planar: str, tiled: bool):
    """Write *data* (bands, height, width) with the requested planar + layout.

    tifffile defaults: when bands are first axis, planar='separate'.
    Pass planar='contig' to interleave samples (data is transposed to
    (height, width, bands) before write).
    """
    import tifffile  # local import; gated by _requires_tifffile

    kwargs = {"photometric": "rgb" if data.shape[0] == 3 else "minisblack"}
    if data.shape[0] not in (1, 3):
        # tifffile rejects 'rgb' for non-3-band; use a generic photometric
        # tag and tell it to skip extra-sample warnings.
        kwargs = {"photometric": "minisblack"}
    if tiled:
        kwargs["tile"] = (32, 32)
    if planar == "separate":
        kwargs["planarconfig"] = "separate"
        tifffile.imwrite(path, data, **kwargs)
    elif planar == "contig":
        kwargs["planarconfig"] = "contig"
        tifffile.imwrite(path, np.transpose(data, (1, 2, 0)), **kwargs)
    else:
        raise ValueError(f"unknown planar={planar!r}")


# ---------------------------------------------------------------------------
# Section 7: planar config x layout x bands x dtype matrix
# ---------------------------------------------------------------------------


@_requires_tifffile
@pytest.mark.parametrize("planar", ["separate", "contig"])
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("bands", [2, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_planar_multiband_cpu(planar, tiled, bands, dtype, tmp_path):
    """CPU ``open_geotiff`` returns (y, x, band) for every planar+layout combo."""
    height, width = 64, 96
    data = _make_planar_matrix_data(bands, height, width, dtype)
    expected = np.transpose(data, (1, 2, 0))

    path = os.path.join(str(tmp_path), "planar_cpu_2429.tif")
    _write_planar_matrix_tiff(path, data, planar=planar, tiled=tiled)
    out = open_geotiff(path)

    assert out.dims == ("y", "x", "band"), (
        f"got dims {out.dims} for planar={planar} tiled={tiled} "
        f"bands={bands} dtype={np.dtype(dtype).name}"
    )
    assert out.shape == (height, width, bands)
    arr = np.asarray(out.data)
    assert arr.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(arr, expected)


@_requires_tifffile
@_gpu_only
@pytest.mark.parametrize("planar", ["separate", "contig"])
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("bands", [2, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_planar_multiband_gpu_matches_cpu(
        planar, tiled, bands, dtype, tmp_path):
    """GPU read agrees with CPU read AND with the source array."""
    height, width = 64, 96
    data = _make_planar_matrix_data(bands, height, width, dtype)
    expected = np.transpose(data, (1, 2, 0))

    path = os.path.join(str(tmp_path), "planar_gpu_2429.tif")
    _write_planar_matrix_tiff(path, data, planar=planar, tiled=tiled)
    cpu = np.asarray(open_geotiff(path).data)
    gpu_da = read_geotiff_gpu(path)

    assert gpu_da.dims == ("y", "x", "band")
    assert gpu_da.shape == (height, width, bands)
    gpu = gpu_da.data.get()

    np.testing.assert_array_equal(cpu, expected)
    np.testing.assert_array_equal(gpu, expected)
    np.testing.assert_array_equal(gpu, cpu)


@_requires_tifffile
@pytest.mark.parametrize("tiled", [True, False])
def test_planar_singleband_cpu(tiled, tmp_path):
    """Single-band reads stay 2-D regardless of layout."""
    import tifffile

    rng = np.random.RandomState(7)
    data = rng.randint(0, 200, size=(48, 80)).astype(np.uint8)
    path = os.path.join(str(tmp_path), "single_cpu_2429.tif")
    kwargs = {"photometric": "minisblack"}
    if tiled:
        kwargs["tile"] = (32, 32)
    tifffile.imwrite(path, data, **kwargs)
    out = open_geotiff(path)
    assert out.dims == ("y", "x")
    assert out.shape == (48, 80)
    np.testing.assert_array_equal(np.asarray(out.data), data)


@_requires_tifffile
@_gpu_only
@pytest.mark.parametrize("tiled", [True, False])
def test_planar_singleband_gpu(tiled, tmp_path):
    """Single-band GPU reads stay 2-D regardless of layout."""
    import tifffile

    rng = np.random.RandomState(7)
    data = rng.randint(0, 200, size=(48, 80)).astype(np.uint8)
    path = os.path.join(str(tmp_path), "single_gpu_2429.tif")
    kwargs = {"photometric": "minisblack"}
    if tiled:
        kwargs["tile"] = (32, 32)
    tifffile.imwrite(path, data, **kwargs)
    out = read_geotiff_gpu(path)
    assert out.dims == ("y", "x")
    assert out.shape == (48, 80)
    np.testing.assert_array_equal(out.data.get(), data)


@_requires_tifffile
def test_planar_stripped_separate_axis_order(tmp_path):
    """Spec-level guard: stripped planar=2 must yield (y, x, band)."""
    import tifffile

    data = _make_planar_matrix_data(3, 64, 96, np.uint8)
    path = os.path.join(str(tmp_path), "a3_repro_2429.tif")
    tifffile.imwrite(path, data, photometric="rgb")  # default planar=2
    out = open_geotiff(path)
    assert out.dims == ("y", "x", "band")
    assert out.shape == (64, 96, 3), (
        f"got shape {out.shape} for dims {out.dims} -- A3 regressed"
    )


# ---------------------------------------------------------------------------
# Section 8 helpers: ThreadPoolExecutor spy for parallel tile decode
# ---------------------------------------------------------------------------


class _PoolSpy:
    """Drop-in replacement for ThreadPoolExecutor that records calls.

    Wraps the real executor so tests still get correct decoded output;
    only the construction is observed. Each instance writes its
    construction (and the test's thread id) into the supplied list so
    pytest-xdist runs do not cross-contaminate.
    """

    def __init__(self, record, real_cls):
        self._record = record
        self._real_cls = real_cls
        self._real = None

    def __call__(self, *args, **kwargs):
        self._record.append({
            'args': args,
            'kwargs': dict(kwargs),
            'thread': threading.get_ident(),
        })
        self._real = self._real_cls(*args, **kwargs)
        return self._real


def _build_tiled_tiff_for_parallel(
        tile_size: int, tiles_across: int, tiles_down: int) -> bytes:
    """Build a tiled TIFF with deterministic pixel content."""
    width = tile_size * tiles_across
    height = tile_size * tiles_down
    pixel_data = np.arange(
        width * height, dtype=np.float32).reshape(height, width)
    return make_minimal_tiff(
        width, height, np.dtype('float32'),
        pixel_data=pixel_data,
        tiled=True,
        tile_size=tile_size,
    )


def _decode_tiled(data: bytes) -> np.ndarray:
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    ifd = ifds[0]
    dtype = tiff_dtype_to_numpy(ifd.bits_per_sample, ifd.sample_format)
    return _reader_mod._read_tiles(data, ifd, header, dtype)


# ---------------------------------------------------------------------------
# Section 8: parallel tile decode gate at the default tile_size
# ---------------------------------------------------------------------------


def test_parallel_tile_decode_engages_at_default_tile_size_256(monkeypatch):
    """At tile_size=256 with multiple tiles, parallel decode must run.

    256 * 256 == 64 * 1024 exactly, so this is the boundary case the
    old strict ``>`` excluded.
    """
    record: list[dict] = []
    real_cls = concurrent.futures.ThreadPoolExecutor
    monkeypatch.setattr(
        concurrent.futures,
        'ThreadPoolExecutor',
        _PoolSpy(record, real_cls),
    )

    # 2x2 tile grid at the default tile_size=256 -> 4 tiles, each 64K pixels.
    data = _build_tiled_tiff_for_parallel(
        tile_size=256, tiles_across=2, tiles_down=2)
    arr = _decode_tiled(data)

    # Output is correct.
    assert arr.shape == (512, 512)
    assert arr.dtype == np.float32
    assert arr[0, 0] == 0.0
    assert arr[-1, -1] == float(512 * 512 - 1)

    # Most importantly, the parallel path engaged.
    assert len(record) == 1, (
        f"Expected ThreadPoolExecutor to be constructed exactly once, "
        f"got {len(record)} (parallel path did not run for default tile_size)"
    )


def test_parallel_tile_decode_sequential_for_small_tiles(monkeypatch):
    """At tile_size=128 (16K pixels), parallel must NOT run.

    The threshold is meant to avoid pool overhead for small tiles, and
    128*128 = 16384 < 64*1024. This guards against an over-eager fix.
    """
    record: list[dict] = []
    real_cls = concurrent.futures.ThreadPoolExecutor
    monkeypatch.setattr(
        concurrent.futures,
        'ThreadPoolExecutor',
        _PoolSpy(record, real_cls),
    )

    data = _build_tiled_tiff_for_parallel(
        tile_size=128, tiles_across=4, tiles_down=4)
    arr = _decode_tiled(data)

    assert arr.shape == (512, 512)
    assert len(record) == 0, (
        f"Expected sequential decode below threshold, but "
        f"ThreadPoolExecutor was constructed {len(record)} time(s)"
    )


def test_parallel_tile_decode_sequential_when_only_one_tile(monkeypatch):
    """A single tile must stay on the sequential path even at large size."""
    record: list[dict] = []
    real_cls = concurrent.futures.ThreadPoolExecutor
    monkeypatch.setattr(
        concurrent.futures,
        'ThreadPoolExecutor',
        _PoolSpy(record, real_cls),
    )

    data = _build_tiled_tiff_for_parallel(
        tile_size=256, tiles_across=1, tiles_down=1)
    arr = _decode_tiled(data)

    assert arr.shape == (256, 256)
    assert len(record) == 0, (
        f"Single-tile reads must stay sequential, but the pool was "
        f"constructed {len(record)} time(s)"
    )


# ---------------------------------------------------------------------------
# Section 9 helpers: stripped TIFF builders + HTTP server
# ---------------------------------------------------------------------------


def _make_stripped_uint16(
        height: int, width: int, *,
        compression: str = "deflate") -> tuple[np.ndarray, bytes]:
    """Build a stripped TIFF in memory; return (numpy_array, file_bytes)."""
    rng = np.random.default_rng(seed=12345)
    arr = rng.integers(0, 256, size=(height, width), dtype=np.uint16)
    da = xr.DataArray(arr, dims=["y", "x"])
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        path = f.name
    try:
        to_geotiff(da, path, compression=compression, tiled=False)
        with open(path, "rb") as f:
            return arr, f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serve a fixed byte blob with HTTP Range support."""
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
            self.send_header("Content-Range",
                             f"bytes {r0}-{r1}/{len(self.blob)}")
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

    def log_message(self, format, *args):  # quiet
        return


def _start_server(blob: bytes):
    """Start an HTTP server serving the given blob on a free port."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    handler = type(
        "BlobHandler", (_RangeHandler,), {"blob": blob})
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server, port


# ---------------------------------------------------------------------------
# Section 9a: local strip decode parallel gate
# ---------------------------------------------------------------------------


class TestReadStripsParallelGate:
    """Local strip path: the parallel-decode gate engages on multi-strip."""

    def test_parallel_strip_decode_matches_serial_local(self, tmp_path):
        """A wide stripped TIFF: parallel and serial decode return
        identical bytes.

        Uses pytest's ``tmp_path`` rather than
        ``tempfile.TemporaryDirectory()`` because ``read_to_array``
        leaves the mmap entry cached on close (see ``_MmapCache``);
        on Windows the cached mmap holds the file handle, so an
        eager ``TemporaryDirectory.__exit__`` rmtree races the mmap
        and raises ``WinError 32``. ``tmp_path`` defers cleanup to
        pytest's session teardown, which tolerates that race.
        """
        arr, blob = _make_stripped_uint16(2048, 4096)
        p = str(tmp_path / "s_2429.tif")
        with open(p, "wb") as f:
            f.write(blob)
        par, _ = read_to_array(p)
        # Patch the threshold in ``_decode`` (where ``_read_strips``
        # lives), not in ``_reader``: the back-imported binding in
        # ``_reader`` is a separate reference.
        with patch.object(_decode_mod,
                          "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10**12):
            ser, _ = read_to_array(p)
        np.testing.assert_array_equal(par, ser)
        np.testing.assert_array_equal(par, arr)

    def test_parallel_strip_decode_pool_engages_on_multi_strip(self, tmp_path):
        """Confirm the parallel branch is taken: patch ThreadPoolExecutor
        and assert it is constructed when n_strips > 1 and strip pixels
        clear the gate."""
        arr, blob = _make_stripped_uint16(1024, 2048)
        p = str(tmp_path / "multi_strip_2429.tif")
        with open(p, "wb") as f:
            f.write(blob)
        # Patch ``concurrent.futures.ThreadPoolExecutor`` rather than the
        # reader module binding because the strip decode lives in
        # ``_decode`` and re-imports the executor function-locally.
        with patch.object(concurrent.futures, "ThreadPoolExecutor",
                          wraps=concurrent.futures.ThreadPoolExecutor
                          ) as mock_pool:
            out, _ = read_to_array(p)
            # n_strips for a 1024-row file with default rps -> at
            # least 4 strips (TIFFs default rps=8KB / row).
            assert mock_pool.called
        np.testing.assert_array_equal(out, arr)

    def test_parallel_strip_decode_serial_for_single_strip(self, tmp_path):
        """A single-row image will produce a single strip; the parallel
        gate must short-circuit to the serial branch."""
        arr = np.arange(2 * 32, dtype=np.uint16).reshape(2, 32)
        da = xr.DataArray(arr, dims=["y", "x"])
        p = str(tmp_path / "tiny_2429.tif")
        to_geotiff(da, p, compression="deflate", tiled=False)
        with patch.object(concurrent.futures, "ThreadPoolExecutor",
                          wraps=concurrent.futures.ThreadPoolExecutor
                          ) as mock_pool:
            out, _ = read_to_array(p)
            # Single-strip file => no pool.
            assert not mock_pool.called
        np.testing.assert_array_equal(out, arr)

    def test_parallel_strip_decode_windowed_matches_serial(self, tmp_path):
        """A windowed read across multiple strips still matches the full
        decode."""
        arr, blob = _make_stripped_uint16(2048, 2048)
        p = str(tmp_path / "win_2429.tif")
        with open(p, "wb") as f:
            f.write(blob)
        par, _ = read_to_array(p, window=(100, 100, 1500, 1500))
        with patch.object(_decode_mod,
                          "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10**12):
            ser, _ = read_to_array(p, window=(100, 100, 1500, 1500))
        np.testing.assert_array_equal(par, ser)
        np.testing.assert_array_equal(par, arr[100:1500, 100:1500])


# ---------------------------------------------------------------------------
# Section 9b: HTTP COG strip path
# ---------------------------------------------------------------------------


@requires_loopback
class TestHttpStripParallelDecode:
    def test_parallel_strip_decode_http_matches_serial(self, monkeypatch):
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
        arr, blob = _make_stripped_uint16(1024, 2048, compression="deflate")
        server, port = _start_server(blob)
        try:
            url = f"http://127.0.0.1:{port}/s_2429.tif"
            par, _ = read_to_array(url, window=(0, 0, 1024, 2048))
            with patch.object(_reader_mod,
                              "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10**12):
                ser, _ = read_to_array(url, window=(0, 0, 1024, 2048))
        finally:
            server.shutdown()
        np.testing.assert_array_equal(par, ser)
        np.testing.assert_array_equal(par, arr)

    def test_parallel_strip_decode_http_serial_on_single_strip(
            self, monkeypatch):
        """HTTP windowed-strip path with a single-strip source: the
        gate at ``_fetch_decode_cog_http_strips`` (n_decode_strips > 1)
        must short-circuit to the serial branch.

        ``ThreadPoolExecutor`` is also used by ``read_ranges`` to fan
        out fetches, but for a single-range fetch that path takes the
        ``len(ranges) == 1`` short-circuit and never instantiates a
        pool. So spying on ``_decode_strip_or_tile`` and asserting the
        thread identity catches a regression that fires the pool when
        it shouldn't, without false positives from the fetch layer.
        """
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
        # 2x32 uint16 -> 1 strip with the writer's default rps.
        arr = np.arange(2 * 32, dtype=np.uint16).reshape(2, 32)
        da = xr.DataArray(arr, dims=["y", "x"])
        with tempfile.NamedTemporaryFile(suffix=".tif",
                                         delete=False) as f:
            tif_path = f.name
        try:
            to_geotiff(da, tif_path, compression="deflate",
                       tiled=False)
            with open(tif_path, "rb") as f:
                blob = f.read()
        finally:
            try:
                os.remove(tif_path)
            except OSError:
                pass

        threads_seen: list[int] = []
        real_decode = _reader_mod._decode_strip_or_tile

        def _spy(*args, **kwargs):
            threads_seen.append(threading.get_ident())
            return real_decode(*args, **kwargs)

        monkeypatch.setattr(
            _reader_mod, "_decode_strip_or_tile", _spy)

        server, port = _start_server(blob)
        try:
            url = f"http://127.0.0.1:{port}/tiny_2429.tif"
            out, _ = read_to_array(url, window=(0, 0, 2, 32))
        finally:
            server.shutdown()

        main_tid = threading.get_ident()
        assert threads_seen, "_decode_strip_or_tile was never called"
        assert all(tid == main_tid for tid in threads_seen), (
            f"decode ran in worker thread(s) "
            f"{set(threads_seen) - {main_tid}}; main is {main_tid}. "
            f"The serial gate failed to short-circuit on a "
            f"single-strip HTTP read."
        )
        np.testing.assert_array_equal(out, arr)


# ---------------------------------------------------------------------------
# Section 9c: planar=2 multi-band stripped TIFF parallel decode
# ---------------------------------------------------------------------------


@_requires_tifffile
class TestPlanar2MultibandStripParallel:
    """Planar=2 multi-band stripped TIFF.

    ``_read_strips`` has a separate ``planar == 2 and samples > 1``
    branch in the strip-job collection loop (xrspatial/geotiff/
    _reader.py:1949-1963). The parallel pool itself is planar-agnostic,
    so a regression in band-ordering or per-band strip indexing would
    survive the chunky tests above; this class covers that branch.
    """

    def test_parallel_strip_decode_planar2_matches_serial(self, tmp_path):
        """Parallel and serial decode produce bit-identical output on
        a planar=2 multi-band stripped TIFF."""
        import tifffile

        rng = np.random.default_rng(seed=20260518)
        bands, height, width = 3, 1024, 1024
        arr = rng.integers(
            0, 1000, size=(bands, height, width), dtype=np.uint16)
        p = str(tmp_path / "planar2_2429.tif")
        tifffile.imwrite(
            p, arr,
            photometric="rgb",
            planarconfig="separate",
            compression="deflate",
            rowsperstrip=128,
        )

        par, _ = read_to_array(p)
        with patch.object(
                _decode_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD",
                10 ** 12):
            ser, _ = read_to_array(p)

        np.testing.assert_array_equal(par, ser)
        # The reader returns (height, width, samples); the fixture is
        # (bands, height, width). Reorder for comparison.
        np.testing.assert_array_equal(par, np.moveaxis(arr, 0, -1))

    def test_parallel_strip_decode_planar2_windowed_matches_serial(
            self, tmp_path):
        """A window across multiple strips on a planar=2 multi-band
        stripped TIFF still matches the slice of the full decode."""
        import tifffile

        rng = np.random.default_rng(seed=20260519)
        bands, height, width = 3, 1024, 1024
        arr = rng.integers(
            0, 1000, size=(bands, height, width), dtype=np.uint16)
        p = str(tmp_path / "planar2_win_2429.tif")
        tifffile.imwrite(
            p, arr,
            photometric="rgb",
            planarconfig="separate",
            compression="deflate",
            rowsperstrip=128,
        )

        par, _ = read_to_array(p, window=(100, 100, 900, 900))
        with patch.object(
                _decode_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD",
                10 ** 12):
            ser, _ = read_to_array(p, window=(100, 100, 900, 900))

        np.testing.assert_array_equal(par, ser)
        expected = np.moveaxis(arr, 0, -1)[100:900, 100:900]
        np.testing.assert_array_equal(par, expected)

    @requires_loopback
    def test_parallel_strip_decode_http_planar2_windowed_matches_serial(
            self, monkeypatch):
        """HTTP windowed strip path on planar=2 multi-band: pins the
        per-band strip-job loop inside
        ``_fetch_decode_cog_http_strips`` (the local path's planar=2
        coverage above does not reach the HTTP fetch+decode branch
        because full-image HTTP reads dispatch back to
        ``_read_strips``)."""
        import tifffile

        monkeypatch.setenv(
            "XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
        rng = np.random.default_rng(seed=20260520)
        bands, height, width = 3, 1024, 1024
        arr = rng.integers(
            0, 1000, size=(bands, height, width), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(
                suffix=".tif", delete=False) as f:
            tif_path = f.name
        try:
            tifffile.imwrite(
                tif_path, arr,
                photometric="rgb",
                planarconfig="separate",
                compression="deflate",
                rowsperstrip=128,
            )
            with open(tif_path, "rb") as f:
                blob = f.read()
        finally:
            try:
                os.remove(tif_path)
            except OSError:
                pass

        server, port = _start_server(blob)
        try:
            url = f"http://127.0.0.1:{port}/planar2_2429.tif"
            par, _ = read_to_array(
                url, window=(100, 100, 900, 900))
            with patch.object(
                    _reader_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD",
                    10 ** 12):
                ser, _ = read_to_array(
                    url, window=(100, 100, 900, 900))
        finally:
            server.shutdown()

        np.testing.assert_array_equal(par, ser)
        expected = np.moveaxis(arr, 0, -1)[100:900, 100:900]
        np.testing.assert_array_equal(par, expected)

# ===========================================================================
# Layout assembly without bytes-copy (#1756)
# Source: test_assemble_layout_no_bytes_copy_1756.py
# ===========================================================================


def _build_parts(arr: np.ndarray):
    """Helper: build the (rel_offsets, byte_counts, chunks) parts for *arr*."""
    rel_off, bc, chunks = _write_stripped(arr, COMPRESSION_NONE, False)
    return [(arr, arr.shape[1], arr.shape[0], rel_off, bc, chunks)]


def test_assemble_standard_layout_returns_bytearray():
    """``_assemble_standard_layout`` returns a bytearray, not bytes."""
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    parts = _build_parts(arr)

    # Minimal tag set sufficient for the assembler to lay out the file.
    from xrspatial.geotiff._dtypes import LONG, SHORT, numpy_to_tiff_dtype
    from xrspatial.geotiff._header import (TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_IMAGE_LENGTH,
                                           TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC, TAG_ROWS_PER_STRIP,
                                           TAG_SAMPLE_FORMAT, TAG_SAMPLES_PER_PIXEL,
                                           TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS)
    bps, sf = numpy_to_tiff_dtype(arr.dtype)
    rel_off, bc, _ = parts[0][3], parts[0][4], parts[0][5]
    tags = [
        (TAG_IMAGE_WIDTH, LONG, 1, arr.shape[1]),
        (TAG_IMAGE_LENGTH, LONG, 1, arr.shape[0]),
        (TAG_BITS_PER_SAMPLE, SHORT, 1, bps),
        (TAG_COMPRESSION, SHORT, 1, 1),
        (TAG_PHOTOMETRIC, SHORT, 1, 1),
        (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
        (TAG_SAMPLE_FORMAT, SHORT, 1, sf),
        (TAG_ROWS_PER_STRIP, SHORT, 1, arr.shape[0]),
        (TAG_STRIP_OFFSETS, LONG, len(rel_off), rel_off),
        (TAG_STRIP_BYTE_COUNTS, LONG, len(bc), bc),
    ]
    result = _assemble_standard_layout(8, [tags], parts, bigtiff=False)
    assert isinstance(result, bytearray), (
        f"expected bytearray (no copy), got {type(result).__name__}"
    )


def test_assemble_cog_layout_returns_bytearray():
    """``_assemble_cog_layout`` returns a bytearray for the COG path."""
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rel_off, bc, chunks = _write_tiled(arr, COMPRESSION_NONE, False, tile_size=16)
    parts = [
        (arr, 16, 16, rel_off, bc, chunks),
        (arr[:8, :8], 8, 8, rel_off, bc, chunks),  # mock overview
    ]
    from xrspatial.geotiff._dtypes import LONG, SHORT, numpy_to_tiff_dtype
    from xrspatial.geotiff._header import (TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_IMAGE_LENGTH,
                                           TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC, TAG_SAMPLE_FORMAT,
                                           TAG_SAMPLES_PER_PIXEL, TAG_TILE_BYTE_COUNTS,
                                           TAG_TILE_LENGTH, TAG_TILE_OFFSETS, TAG_TILE_WIDTH)
    bps, sf = numpy_to_tiff_dtype(arr.dtype)

    def _build_tags(w, h):
        return [
            (TAG_IMAGE_WIDTH, LONG, 1, w),
            (TAG_IMAGE_LENGTH, LONG, 1, h),
            (TAG_BITS_PER_SAMPLE, SHORT, 1, bps),
            (TAG_COMPRESSION, SHORT, 1, 1),
            (TAG_PHOTOMETRIC, SHORT, 1, 1),
            (TAG_SAMPLES_PER_PIXEL, SHORT, 1, 1),
            (TAG_SAMPLE_FORMAT, SHORT, 1, sf),
            (TAG_TILE_WIDTH, SHORT, 1, 16),
            (TAG_TILE_LENGTH, SHORT, 1, 16),
            (TAG_TILE_OFFSETS, LONG, len(rel_off), rel_off),
            (TAG_TILE_BYTE_COUNTS, LONG, len(bc), bc),
        ]
    ifd_specs = [_build_tags(16, 16), _build_tags(8, 8)]
    result = _assemble_cog_layout(8, ifd_specs, parts, bigtiff=False)
    assert isinstance(result, bytearray)


def test_assemble_tiff_returns_bytearray():
    """``_assemble_tiff`` propagates the bytearray return through both layouts."""
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    parts = _build_parts(arr)
    out = _assemble_tiff(
        8, 8, arr.dtype, COMPRESSION_NONE, 1, False, 256,
        parts, None, None, None, is_cog=False, raster_type=1)
    assert isinstance(out, bytearray), (
        f"expected bytearray, got {type(out).__name__}"
    )


def test_assemble_tiff_round_trips_through_bytes_io():
    """End-to-end: write a bytearray output to BytesIO and read it back."""
    import xarray as xr

    arr = xr.DataArray(
        np.arange(64, dtype=np.uint8).reshape(8, 8),
        dims=['y', 'x'],
    )
    buf = io.BytesIO()
    # ``to_geotiff`` -> ``write`` -> ``_assemble_tiff`` (bytearray) ->
    # ``_write_bytes`` -> ``buf.write(bytearray)``. The whole chain
    # needs to accept the buffer protocol; a regression that re-introduces
    # ``bytes(output)`` would still pass this test, but the
    # ``isinstance`` checks above would fail first.
    to_geotiff(arr, buf, compression='none', tiled=False)
    buf.seek(0)
    da = open_geotiff(buf)
    np.testing.assert_array_equal(da.values, arr.values)


def test_assemble_tiff_round_trips_through_disk(tmp_path):
    """End-to-end: bytearray output to a local file is parseable."""
    import xarray as xr

    arr = xr.DataArray(
        np.random.randint(0, 255, (128, 128), dtype=np.uint8),
        dims=['y', 'x'],
    )
    path = str(tmp_path / 'no_bytes_copy_1756.tif')
    to_geotiff(arr, path, compression='deflate', tiled=True, tile_size=64)
    da = open_geotiff(path)
    np.testing.assert_array_equal(da.values, arr.values)


def test_assemble_tiff_output_is_mutable_buffer_with_valid_header():
    """Verify the assembler returns a mutable ``bytearray`` whose buffer
    slices behave correctly for downstream consumers.

    A regression that re-introduced ``return bytes(output)`` in the
    assembler would surface here in two ways:

    1. ``isinstance(out, bytearray)`` would fail (the type would be
       ``bytes``, immutable).
    2. ``out[:16]`` would be ``bytes`` rather than ``bytearray``, and
       the validation slice that feeds ``parse_header`` in ``write``
       would carry the wrong type.

    We do not try to monkey-patch the builtin ``bytes``: the writer
    module looks ``bytes`` up via ``builtins``, so patching the
    module namespace would not intercept calls without invasive
    rebinding. The type and slice-type assertions below are sufficient
    to catch a re-introduction of the full-buffer copy in the assembler
    return statement, which was the specific regression fixed in #1756.
    """
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    parts = _build_parts(arr)
    out = _assemble_tiff(
        8, 8, arr.dtype, COMPRESSION_NONE, 1, False, 256,
        parts, None, None, None, is_cog=False, raster_type=1)
    # Confirm the buffer is still a writeable bytearray: a regression that
    # converts back to ``bytes`` would produce an immutable object.
    assert isinstance(out, bytearray)
    # Buffer-protocol slicing returns a bytearray for bytearray inputs,
    # which the validation path in ``write`` slices for ``parse_header``.
    sliced = out[:16]
    assert isinstance(sliced, bytearray)
    assert sliced[:2] == b'II'
