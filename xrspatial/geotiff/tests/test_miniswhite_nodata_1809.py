"""MinIsWhite + nodata interaction must mask sentinel pixels, not real data (#1809).

Before the fix, ``open_geotiff`` applied the MinIsWhite inversion
(``np.iinfo(dtype).max - arr`` for uint, ``-arr`` for float) **before** the
sentinel-to-NaN nodata mask.  The mask then compared against the *original*
sentinel value, so:

* pixels whose stored value was the sentinel survived as ``iinfo.max -
  sentinel`` instead of becoming ``NaN``
* pixels whose stored value happened to equal ``iinfo.max - sentinel`` were
  incorrectly converted to ``NaN``

All four backends (eager numpy, dask, eager GPU, HTTP COG dask) shared the
same ordering bug.  The fix stashes the post-MinIsWhite sentinel on
``geo_info._mask_nodata`` and routes every backend's mask through it,
keeping ``attrs['nodata']`` at the original sentinel for round-trip on
write.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff

tifffile = pytest.importorskip("tifffile")


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.is_available())
    except Exception:  # pragma: no cover - import errors only
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


def _write_miniswhite_tiff(path: str, stored: np.ndarray, nodata_str: str,
                           tiled: bool = False) -> None:
    extratags = [("GDAL_NODATA", "s", 0, f"{nodata_str}\0", True)]
    kwargs = {"photometric": "miniswhite", "extratags": extratags}
    if tiled:
        kwargs["tile"] = (16, 16)
    tifffile.imwrite(path, stored, **kwargs)


# ---------------------------------------------------------------------------
# uint8 + nodata=0 — the sentinel collides with stored 255 after inversion
# ---------------------------------------------------------------------------


def _uint8_case():
    stored = np.array([[0, 100, 200], [50, 0, 255]], dtype=np.uint8)
    expected = np.array(
        [[np.nan, 155.0, 55.0], [205.0, np.nan, 0.0]], dtype=np.float64
    )
    return stored, expected


def test_eager_numpy_uint8_nodata_zero(tmp_path):
    stored, expected = _uint8_case()
    path = str(tmp_path / "mw_uint8.tif")
    _write_miniswhite_tiff(path, stored, "0")

    arr = open_geotiff(path)

    assert arr.attrs["nodata"] == 0
    np.testing.assert_array_equal(arr.values, expected)


def test_dask_uint8_nodata_zero(tmp_path):
    stored, expected = _uint8_case()
    path = str(tmp_path / "mw_uint8_dask.tif")
    _write_miniswhite_tiff(path, stored, "0")

    arr = open_geotiff(path, chunks=2).compute()

    assert arr.attrs["nodata"] == 0
    np.testing.assert_array_equal(arr.values, expected)


@_gpu_only
def test_gpu_eager_uint8_nodata_zero(tmp_path):
    stored, expected = _uint8_case()
    path = str(tmp_path / "mw_uint8_gpu.tif")
    _write_miniswhite_tiff(path, stored, "0", tiled=True)

    arr = open_geotiff(path, gpu=True)

    assert arr.attrs["nodata"] == 0
    np.testing.assert_array_equal(arr.data.get(), expected)


# ---------------------------------------------------------------------------
# uint16 + nodata=65535 — sentinel collides with stored 0 after inversion
# ---------------------------------------------------------------------------


def test_eager_numpy_uint16_nodata_max(tmp_path):
    stored = np.array([[0, 1000, 32000], [65535, 50000, 0]], dtype=np.uint16)
    path = str(tmp_path / "mw_uint16.tif")
    _write_miniswhite_tiff(path, stored, "65535")
    expected = np.array(
        [[65535.0, 64535.0, 33535.0], [np.nan, 15535.0, 65535.0]],
        dtype=np.float64,
    )

    arr = open_geotiff(path)

    assert arr.attrs["nodata"] == 65535
    np.testing.assert_array_equal(arr.values, expected)


# ---------------------------------------------------------------------------
# float32 + nodata=-9999
# ---------------------------------------------------------------------------


def test_eager_numpy_float32_nodata(tmp_path):
    stored = np.array(
        [[0.0, 100.0, -9999.0], [50.0, 9999.0, 200.0]], dtype=np.float32
    )
    path = str(tmp_path / "mw_float.tif")
    _write_miniswhite_tiff(path, stored, "-9999")
    expected = np.array(
        [[-0.0, -100.0, np.nan], [-50.0, -9999.0, -200.0]], dtype=np.float32
    )

    arr = open_geotiff(path)

    assert arr.attrs["nodata"] == -9999
    np.testing.assert_array_equal(arr.values, expected)


def test_dask_float32_nodata(tmp_path):
    stored = np.array(
        [[0.0, 100.0, -9999.0], [50.0, 9999.0, 200.0]], dtype=np.float32
    )
    path = str(tmp_path / "mw_float_dask.tif")
    _write_miniswhite_tiff(path, stored, "-9999")
    expected = np.array(
        [[-0.0, -100.0, np.nan], [-50.0, -9999.0, -200.0]], dtype=np.float32
    )

    arr = open_geotiff(path, chunks=2).compute()

    assert arr.attrs["nodata"] == -9999
    np.testing.assert_array_equal(arr.values, expected)


# ---------------------------------------------------------------------------
# Non-colliding nodata still works — the existing path was correct here, but
# we must not regress it.
# ---------------------------------------------------------------------------


def test_eager_numpy_uint8_nodata_no_collision(tmp_path):
    # Sentinel 7 inverts to 248; no stored pixel equals 248 so no collision.
    stored = np.array([[7, 100, 200], [50, 7, 230]], dtype=np.uint8)
    path = str(tmp_path / "mw_no_collision.tif")
    _write_miniswhite_tiff(path, stored, "7")
    expected = np.array(
        [[np.nan, 155.0, 55.0], [205.0, np.nan, 25.0]], dtype=np.float64
    )

    arr = open_geotiff(path)

    np.testing.assert_array_equal(arr.values, expected)


# ---------------------------------------------------------------------------
# No nodata at all — inversion stays in integer dtype (existing contract).
# ---------------------------------------------------------------------------


def test_eager_numpy_no_nodata_stays_integer(tmp_path):
    stored = np.array([[0, 50, 100, 200]], dtype=np.uint8).repeat(4, axis=0)
    path = str(tmp_path / "mw_no_nodata.tif")
    tifffile.imwrite(path, stored, photometric="miniswhite")

    arr = open_geotiff(path)

    assert arr.dtype == np.uint8
    np.testing.assert_array_equal(arr.values, 255 - stored)


# ---------------------------------------------------------------------------
# Backend parity — every available backend agrees on the same input.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GPU CPU-fallback path: sparse-tile MinIsWhite TIFF (#1817 Copilot review).
#
# The GPU tiled read routes sparse-tile, planar=2 auto-fallback, and final
# CPU-fallback branches through ``read_to_array``. Before the fix those
# branches discarded ``_mask_nodata`` and masked against the original
# sentinel, so a MinIsWhite raster decoded via CPU fallback still produced
# wrong NaN placement.
# ---------------------------------------------------------------------------


def _patch_first_tile_sparse(path: str) -> None:
    """Zero TileOffsets[0] and TileByteCounts[0] in-place to mark tile 0 sparse.

    Mirrors the rewrite pattern used in test_security.py but targets a
    single entry (tile index 0) rather than every entry. Handles SHORT or
    LONG type encodings for both tags.
    """
    import struct

    from xrspatial.geotiff._header import parse_header

    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    header = parse_header(bytes(data))
    bo = header.byte_order
    ifd_offset = header.first_ifd_offset
    num_entries = struct.unpack_from(f"{bo}H", data, ifd_offset)[0]
    entry_offset = ifd_offset + 2

    targets = {324, 325}  # TileOffsets, TileByteCounts
    for i in range(num_entries):
        eo = entry_offset + i * 12
        tag = struct.unpack_from(f"{bo}H", data, eo)[0]
        if tag not in targets:
            continue
        type_id = struct.unpack_from(f"{bo}H", data, eo + 2)[0]
        count = struct.unpack_from(f"{bo}I", data, eo + 4)[0]
        if type_id == 4:  # LONG
            total = count * 4
            if total <= 4:
                struct.pack_into(f"{bo}I", data, eo + 8, 0)
            else:
                ptr = struct.unpack_from(f"{bo}I", data, eo + 8)[0]
                struct.pack_into(f"{bo}I", data, ptr, 0)
        elif type_id == 3:  # SHORT
            total = count * 2
            if total <= 4:
                struct.pack_into(f"{bo}H", data, eo + 8, 0)
            else:
                ptr = struct.unpack_from(f"{bo}I", data, eo + 8)[0]
                struct.pack_into(f"{bo}H", data, ptr, 0)

    with open(path, "wb") as fh:
        fh.write(data)


@_gpu_only
def test_gpu_sparse_tile_miniswhite_nodata_zero(tmp_path):
    """GPU CPU-fallback path must use the post-MinIsWhite sentinel for masking.

    Build a tiled MinIsWhite uint8 raster with ``GDAL_NODATA=0``, then patch
    the first tile's TileOffsets/TileByteCounts to 0 so the read routes
    through the sparse-tile CPU-fallback branch of ``read_geotiff_gpu``.
    """
    # 32x32 raster, 16x16 tiles -> 4 tiles. Tile 0 becomes sparse.
    stored = np.zeros((32, 32), dtype=np.uint8)
    # Distinct values per tile so we can verify each region independently.
    stored[:16, :16] = 0  # tile 0 (will be made sparse) -- value irrelevant
    stored[:16, 16:] = 100  # tile 1
    stored[16:, :16] = 200  # tile 2
    stored[16:, 16:] = 255  # tile 3 collides with inverted nodata sentinel

    path = str(tmp_path / "mw_sparse_gpu.tif")
    _write_miniswhite_tiff(path, stored, "0", tiled=True)
    _patch_first_tile_sparse(path)

    arr = open_geotiff(path, gpu=True)
    out = arr.data.get()

    # Sparse tile materialises as the file's nodata sentinel (0), which is
    # then inverted by MinIsWhite to 255 -- and 255 is the post-inversion
    # mask sentinel, so those pixels must become NaN.
    assert np.all(np.isnan(out[:16, :16]))
    # Tile 1: 100 -> 155 after inversion, not the mask sentinel.
    assert np.all(out[:16, 16:] == 155.0)
    # Tile 2: 200 -> 55 after inversion.
    assert np.all(out[16:, :16] == 55.0)
    # Tile 3 stored 255 -> 0 after inversion. Before the fix, this entire
    # tile was incorrectly masked to NaN because the mask compared against
    # the original sentinel (0). After the fix, the mask sees the inverted
    # sentinel (255) and these pixels survive as 0.
    assert np.all(out[16:, 16:] == 0.0)
    assert arr.attrs["nodata"] == 0


def test_backend_parity_uint8_nodata_zero(tmp_path):
    stored, expected = _uint8_case()
    path = str(tmp_path / "mw_parity.tif")
    _write_miniswhite_tiff(path, stored, "0", tiled=True)

    eager = open_geotiff(path).values
    dask_result = open_geotiff(path, chunks=2).compute().values
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(dask_result, expected)
    np.testing.assert_array_equal(eager, dask_result)
    if _HAS_GPU:
        gpu = open_geotiff(path, gpu=True).data.get()
        np.testing.assert_array_equal(gpu, expected)
        np.testing.assert_array_equal(eager, gpu)
