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
import os
import tempfile

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
