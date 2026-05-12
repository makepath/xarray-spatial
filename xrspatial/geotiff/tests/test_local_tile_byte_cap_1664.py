"""Local-file tile/strip byte-count cap (issue #1664).

Before #1664, ``XRSPATIAL_COG_MAX_TILE_BYTES`` only fired in the HTTP
fetch path. A crafted local TIFF with a huge ``TileByteCounts`` /
``StripByteCounts`` could still feed an enormous slice into the
decompressor, which can balloon into gigabytes of decoded output even
when the underlying mmap slice is bounded by the file size.

These tests fabricate small COGs / strip-TIFFs, rewrite their byte
counts to oversized values, and check that the cap raises before the
decoder runs.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff import _reader as _reader_mod


# ---------------------------------------------------------------------------
# Helpers -- patch in-place IFD entries for tile / strip byte counts
# ---------------------------------------------------------------------------


def _patch_byte_counts(data: bytearray, tag: int, value: int) -> None:
    """Rewrite every entry for *tag* (325=TileByteCounts, 279=StripByteCounts)."""
    from xrspatial.geotiff._header import parse_header
    header = parse_header(bytes(data))
    bo = header.byte_order
    ifd_offset = header.first_ifd_offset
    num_entries = struct.unpack_from(f'{bo}H', data, ifd_offset)[0]
    entry_offset = ifd_offset + 2

    for i in range(num_entries):
        eo = entry_offset + i * 12
        cur_tag = struct.unpack_from(f'{bo}H', data, eo)[0]
        if cur_tag != tag:
            continue
        type_id = struct.unpack_from(f'{bo}H', data, eo + 2)[0]
        count = struct.unpack_from(f'{bo}I', data, eo + 4)[0]
        if type_id == 4:  # LONG
            total = count * 4
            if total <= 4:
                for k in range(count):
                    struct.pack_into(f'{bo}I', data, eo + 8 + k * 4, value)
            else:
                ptr = struct.unpack_from(f'{bo}I', data, eo + 8)[0]
                for k in range(count):
                    struct.pack_into(f'{bo}I', data, ptr + k * 4, value)
        elif type_id == 3:  # SHORT
            clipped = min(value, 0xFFFF)
            total = count * 2
            if total <= 4:
                for k in range(count):
                    struct.pack_into(
                        f'{bo}H', data, eo + 8 + k * 2, clipped)
            else:
                ptr = struct.unpack_from(f'{bo}I', data, eo + 8)[0]
                for k in range(count):
                    struct.pack_into(
                        f'{bo}H', data, ptr + k * 2, clipped)
        return
    raise AssertionError(f"tag {tag} not found in IFD")


def _build_forged_tiled_cog(tmp_path, byte_count_value: int) -> str:
    """Write a real tiled COG, patch every TileByteCounts entry, return path."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    da = xr.DataArray(arr, dims=['y', 'x'])
    path = str(tmp_path / "forged_local_tiles_1664.tif")
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
    path = str(tmp_path / "forged_local_strips_1664.tif")
    # tiled=False forces strip layout; deflate gets the decompressor on
    # the hot path so a huge declared size matters.
    to_geotiff(da, path, tiled=False, compression='deflate')
    with open(path, 'rb') as f:
        data = bytearray(f.read())
    _patch_byte_counts(data, 279, byte_count_value)  # 279 = StripByteCounts
    with open(path, 'wb') as f:
        f.write(data)
    return path


# ---------------------------------------------------------------------------
# Tiled local reads
# ---------------------------------------------------------------------------


class TestLocalTileByteCap:
    def test_huge_tile_byte_count_rejected(self, tmp_path, monkeypatch):
        """A local tile with a huge TileByteCount raises before decode."""
        # 100 MB > the 1 MB cap we set below.
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
        # The forged value (52,428,800) and the cap (1,024) both appear.
        assert "52,428,800" in msg or "52428800" in msg
        assert "1,024" in msg or "1024" in msg
        assert "denial-of-service" in msg.lower() or "malformed" in msg

    def test_normal_local_cog_under_default_cap(self, tmp_path):
        """Legitimate local reads with the default cap still succeed."""
        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        da = xr.DataArray(arr, dims=['y', 'x'])
        path = str(tmp_path / "normal_local_1664.tif")
        to_geotiff(da, path, tile_size=32, compression='deflate')

        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, arr)

    def test_env_override_lifts_cap(self, tmp_path, monkeypatch):
        """A user with legitimate large tiles can lift the cap via env."""
        # 50 MB declared. With cap=64 MB the read succeeds even though
        # the underlying compressed slice is smaller (mmap truncates at
        # EOF).
        path = _build_forged_tiled_cog(tmp_path, 50 * 1024 * 1024)
        monkeypatch.setenv(
            'XRSPATIAL_COG_MAX_TILE_BYTES', str(64 * 1024 * 1024))

        # Read may raise inside the decompressor (the truncated mmap
        # slice is garbage to deflate) but it must NOT raise the cap
        # error. The thing we care about is that the cap check passes.
        try:
            open_geotiff(path)
        except ValueError as e:
            assert "exceeds the per-tile safety cap" not in str(e)


# ---------------------------------------------------------------------------
# Strip-organized local reads
# ---------------------------------------------------------------------------


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
    """Negative env value falls back to the default, not a 1-byte cap.

    Earlier drafts clamped to ``max(1, val)`` which made a typo
    (``XRSPATIAL_COG_MAX_TILE_BYTES=-1``) silently reject every tile.
    The current policy matches ``_http_timeout_from_env``: any non-
    positive integer is ignored.
    """
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
