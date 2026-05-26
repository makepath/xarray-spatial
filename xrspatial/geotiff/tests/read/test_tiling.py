"""Tiled-read paths, tile boundaries, byte caps.

Consolidates:

* ``test_local_tile_byte_cap_1664.py`` -- local-file ``TileByteCounts`` /
  ``StripByteCounts`` cap and the env-driven override (CPU path).
* ``test_gpu_tile_byte_cap_2026_05_18.py`` -- the matching GPU eager and
  dask + GPU chunked paths.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _reader as _reader_mod
from xrspatial.geotiff import open_geotiff, read_geotiff_gpu, to_geotiff

from .._helpers.markers import requires_gpu as _gpu_only
from .._helpers.tiff_surgery import patch_byte_counts as _patch_byte_counts


# ---------------------------------------------------------------------------
# Helpers
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
# Tiled local reads
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
# GPU eager path: per-tile byte cap
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
