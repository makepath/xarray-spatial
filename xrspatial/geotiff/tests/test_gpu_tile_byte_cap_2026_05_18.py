"""GPU read path per-tile byte cap (security sweep follow-up).

The CPU readers ``_read_tiles`` (xrspatial/geotiff/_reader.py:2084) and
``_fetch_decode_cog_http_tiles`` (xrspatial/geotiff/_reader.py:2563)
reject a tile whose declared ``TileByteCount`` exceeds the env-driven
``_max_tile_bytes_from_env()`` cap (default 256 MiB). The eager GPU
read path in ``xrspatial.geotiff._backends.gpu.read_geotiff_gpu`` did
not run the same check; ``validate_tile_layout`` bounds the offsets
array length but not the byte-count entries. A crafted local TIFF with
a multi-hundred-MB ``TileByteCount`` could then pass through to GPU
decode, where ``_check_gpu_memory`` only catches the aggregate at
~90% of free VRAM and not the per-tile asymmetry between the CPU and
GPU paths.

The GPU eager path now applies the same per-tile cap so the CPU and
GPU contracts agree. These tests cover the rejection, the wording of
the rejection message, the env-override escape hatch, and the legit-
read pass-through under the default cap.

Mirrors the structure of ``test_local_tile_byte_cap_1664.py`` for the
CPU paths so a side-by-side comparison is easy.
"""
from __future__ import annotations

import importlib.util
import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import read_geotiff_gpu, to_geotiff


def _cupy_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _cupy_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required for the GPU read path",
)


def _patch_byte_counts(data: bytearray, tag: int, value: int) -> None:
    """Rewrite every entry for *tag* in the first IFD.

    Mirrors the helper in ``test_local_tile_byte_cap_1664.py``: parses
    the TIFF header, walks the IFD, and writes ``value`` over every
    occurrence of the named tag's value array. ``tag=325`` is
    ``TileByteCounts``; ``tag=279`` is ``StripByteCounts``.
    """
    from xrspatial.geotiff._header import parse_header

    header = parse_header(bytes(data))
    bo = header.byte_order
    ifd_offset = header.first_ifd_offset
    num_entries = struct.unpack_from(f"{bo}H", data, ifd_offset)[0]
    entry_offset = ifd_offset + 2

    for i in range(num_entries):
        eo = entry_offset + i * 12
        cur_tag = struct.unpack_from(f"{bo}H", data, eo)[0]
        if cur_tag != tag:
            continue
        type_id = struct.unpack_from(f"{bo}H", data, eo + 2)[0]
        count = struct.unpack_from(f"{bo}I", data, eo + 4)[0]
        if type_id == 4:  # LONG
            total = count * 4
            if total <= 4:
                for k in range(count):
                    struct.pack_into(f"{bo}I", data, eo + 8 + k * 4, value)
            else:
                ptr = struct.unpack_from(f"{bo}I", data, eo + 8)[0]
                for k in range(count):
                    struct.pack_into(f"{bo}I", data, ptr + k * 4, value)
        elif type_id == 3:  # SHORT
            clipped = min(value, 0xFFFF)
            total = count * 2
            if total <= 4:
                for k in range(count):
                    struct.pack_into(
                        f"{bo}H", data, eo + 8 + k * 2, clipped)
            else:
                ptr = struct.unpack_from(f"{bo}I", data, eo + 8)[0]
                for k in range(count):
                    struct.pack_into(
                        f"{bo}H", data, ptr + k * 2, clipped)
        return
    raise AssertionError(f"tag {tag} not found in IFD")


def _build_forged_tiled_cog(tmp_path, byte_count_value: int) -> str:
    """Write a real tiled COG, patch every TileByteCounts entry, return path."""
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    da = xr.DataArray(arr, dims=["y", "x"])
    path = str(tmp_path / "forged_gpu_tiles_2026_05_18.tif")
    to_geotiff(da, path, tile_size=32, compression="deflate")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    _patch_byte_counts(data, 325, byte_count_value)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ---------------------------------------------------------------------------
# GPU eager path: per-tile byte cap
# ---------------------------------------------------------------------------


class TestGpuTileByteCap:
    @_gpu_only
    def test_huge_tile_byte_count_rejected(self, tmp_path, monkeypatch):
        """A local tile with a huge TileByteCount raises before GPU decode."""
        path = _build_forged_tiled_cog(tmp_path, 100 * 1024 * 1024)
        monkeypatch.setenv("XRSPATIAL_COG_MAX_TILE_BYTES", str(1024 * 1024))

        with pytest.raises(ValueError, match="TileByteCount"):
            read_geotiff_gpu(path)

    @_gpu_only
    def test_error_message_names_value_and_cap(self, tmp_path, monkeypatch):
        path = _build_forged_tiled_cog(tmp_path, 50 * 1024 * 1024)
        monkeypatch.setenv("XRSPATIAL_COG_MAX_TILE_BYTES", str(1024))

        with pytest.raises(ValueError) as excinfo:
            read_geotiff_gpu(path)
        msg = str(excinfo.value)
        # The forged value (52,428,800) and the cap (1,024) both appear.
        assert "52,428,800" in msg or "52428800" in msg
        assert "1,024" in msg or "1024" in msg
        assert "denial-of-service" in msg.lower() or "malformed" in msg

    @_gpu_only
    def test_normal_gpu_read_under_default_cap(self, tmp_path):
        """Legitimate GPU reads with the default cap still succeed."""
        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        da = xr.DataArray(arr, dims=["y", "x"])
        path = str(tmp_path / "normal_gpu_2026_05_18.tif")
        to_geotiff(da, path, tile_size=32, compression="deflate")

        result = read_geotiff_gpu(path)
        # CuPy -> numpy for comparison.
        np.testing.assert_array_equal(result.data.get(), arr)

    @_gpu_only
    def test_env_override_lifts_cap(self, tmp_path, monkeypatch):
        """A user with legitimate large tiles can lift the cap via env."""
        path = _build_forged_tiled_cog(tmp_path, 50 * 1024 * 1024)
        monkeypatch.setenv(
            "XRSPATIAL_COG_MAX_TILE_BYTES", str(64 * 1024 * 1024))

        # The decompressor may raise on the truncated mmap slice, but
        # the per-tile cap error must not be the source. Match the
        # behaviour pinned by ``test_env_override_lifts_cap`` in the
        # CPU companion module.
        try:
            read_geotiff_gpu(path)
        except ValueError as exc:
            assert "exceeds the per-tile safety cap" not in str(exc)
        except Exception:
            # Codec failures on the truncated payload are acceptable;
            # we only care that the cap check did not fire.
            pass
