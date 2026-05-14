"""Regression tests for full-precision parsing of 64-bit integer nodata sentinels.

Before issue #1847 the reader parsed ``GDAL_NODATA`` via ``float()``
unconditionally in three call sites (``_geotags.extract_geo_info``,
``_reader._resolve_masked_fill``, ``_reader._sparse_fill_value``).
``2**64 - 1`` (uint64 max) and ``2**63 - 1`` (int64 max) are not exactly
representable in float64; the nearest float sits one ULP above the
dtype's max so the downstream ``info.min <= int(nodata) <= info.max``
gate rejected the cast and the sentinel pixel survived as a literal
valid integer rather than being masked to NaN.

The fix mirrors :func:`xrspatial.geotiff._vrt._parse_band_nodata` (PR
#1833) which addressed the same class of bug on the VRT XML path: try
``int()`` first to preserve full precision, fall back to ``float()`` for
NaN / Inf / scientific notation / fractional values.

See issue #1847.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    read_geotiff_dask,
    read_vrt,
    to_geotiff,
    write_vrt,
)
from xrspatial.geotiff._geotags import _parse_nodata_str


# ---------------------------------------------------------------------------
# Unit-level helper
# ---------------------------------------------------------------------------


class TestParseNodataStr:
    """Pin the int-first, float-fallback contract."""

    def test_uint64_max_round_trips_as_int(self):
        v = _parse_nodata_str(str(2**64 - 1))
        assert isinstance(v, int)
        assert v == 2**64 - 1

    def test_int64_max_round_trips_as_int(self):
        v = _parse_nodata_str(str(2**63 - 1))
        assert isinstance(v, int)
        assert v == 2**63 - 1

    def test_int64_min_round_trips_as_int(self):
        v = _parse_nodata_str(str(-(2**63)))
        assert isinstance(v, int)
        assert v == -(2**63)

    def test_negative_int_round_trips_as_int(self):
        v = _parse_nodata_str("-9999")
        assert isinstance(v, int)
        assert v == -9999

    def test_uint16_max_round_trips_as_int(self):
        v = _parse_nodata_str("65535")
        assert isinstance(v, int)
        assert v == 65535

    def test_nan_falls_back_to_float(self):
        v = _parse_nodata_str("nan")
        assert isinstance(v, float)
        assert np.isnan(v)

    def test_inf_falls_back_to_float(self):
        v = _parse_nodata_str("inf")
        assert isinstance(v, float)
        assert np.isinf(v)

    def test_negative_inf_falls_back_to_float(self):
        v = _parse_nodata_str("-inf")
        assert isinstance(v, float)
        assert np.isinf(v) and v < 0

    def test_scientific_notation_falls_back_to_float(self):
        v = _parse_nodata_str("1.5e10")
        assert isinstance(v, float)
        assert v == 1.5e10

    def test_fractional_falls_back_to_float(self):
        v = _parse_nodata_str("-9999.25")
        assert isinstance(v, float)
        assert v == -9999.25

    def test_empty_string_returns_none(self):
        assert _parse_nodata_str("") is None
        assert _parse_nodata_str("   ") is None

    def test_whitespace_stripped(self):
        v = _parse_nodata_str("  42  ")
        assert isinstance(v, int)
        assert v == 42

    def test_garbage_returns_none(self):
        assert _parse_nodata_str("hello") is None

    def test_none_input_returns_none(self):
        assert _parse_nodata_str(None) is None


# ---------------------------------------------------------------------------
# Eager open_geotiff -- the primary repro path
# ---------------------------------------------------------------------------


class TestOpenGeotiffEager:
    """``open_geotiff`` must mask the 64-bit sentinel even when its
    float64 representation collides with one ULP above the dtype max."""

    def _write(self, tmp_path, dtype, sentinel):
        arr = np.full((16, 16), 100, dtype=dtype)
        arr[0, 0] = sentinel
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": np.arange(16.0), "x": np.arange(16.0)},
        )
        path = os.path.join(tmp_path, "t.tif")
        to_geotiff(da, path, nodata=sentinel)
        return path

    def test_uint64_max_masked_to_nan(self, tmp_path):
        path = self._write(str(tmp_path), np.uint64, 2**64 - 1)
        da = open_geotiff(path)
        assert da.dtype == np.float64
        assert np.isnan(da.values[0, 0])
        assert da.values[1, 1] == 100.0
        # attrs preserves the exact sentinel for a write round-trip.
        assert da.attrs["nodata"] == 2**64 - 1

    def test_int64_max_masked_to_nan(self, tmp_path):
        path = self._write(str(tmp_path), np.int64, 2**63 - 1)
        da = open_geotiff(path)
        assert da.dtype == np.float64
        assert np.isnan(da.values[0, 0])
        assert da.values[1, 1] == 100.0
        assert da.attrs["nodata"] == 2**63 - 1

    def test_int64_min_masked_to_nan(self, tmp_path):
        # Regression guard: INT64_MIN is exactly representable in float64
        # and worked before the fix.  Make sure the new int-first path
        # has not broken it.
        path = self._write(str(tmp_path), np.int64, -(2**63))
        da = open_geotiff(path)
        assert da.dtype == np.float64
        assert np.isnan(da.values[0, 0])
        assert da.values[1, 1] == 100.0
        assert da.attrs["nodata"] == -(2**63)

    def test_uint16_max_still_masked(self, tmp_path):
        # Regression guard: small integer sentinels still work.
        path = self._write(str(tmp_path), np.uint16, 65535)
        da = open_geotiff(path)
        assert da.dtype == np.float64
        assert np.isnan(da.values[0, 0])
        assert da.values[1, 1] == 100.0
        assert da.attrs["nodata"] == 65535

    def test_int32_negative_still_masked(self, tmp_path):
        # Regression guard: signed-int small sentinels still work.
        path = self._write(str(tmp_path), np.int32, -9999)
        da = open_geotiff(path)
        assert da.dtype == np.float64
        assert np.isnan(da.values[0, 0])
        assert da.attrs["nodata"] == -9999

    def test_float_nodata_still_parses(self, tmp_path):
        # Regression guard: float dtypes still get float-parsed.
        arr = np.full((8, 8), 1.0, dtype=np.float32)
        arr[0, 0] = -9999.0
        da = xr.DataArray(arr, dims=("y", "x"))
        path = os.path.join(str(tmp_path), "f.tif")
        to_geotiff(da, path, nodata=-9999.0)
        out = open_geotiff(path)
        assert np.isnan(out.values[0, 0])


# ---------------------------------------------------------------------------
# Dask path -- the windowed reader uses the same geo_info.nodata
# ---------------------------------------------------------------------------


class TestReadGeotiffDask:
    def test_uint64_max_masked_via_dask(self, tmp_path):
        arr = np.full((32, 32), 100, dtype=np.uint64)
        arr[0, 0] = 2**64 - 1
        da_in = xr.DataArray(arr, dims=("y", "x"))
        path = os.path.join(str(tmp_path), "t.tif")
        to_geotiff(da_in, path, nodata=2**64 - 1)
        out = read_geotiff_dask(path, chunks=16).compute()
        assert out.dtype == np.float64
        assert np.isnan(out.values[0, 0])
        assert out.values[1, 1] == 100.0

    def test_int64_max_masked_via_dask(self, tmp_path):
        arr = np.full((32, 32), 100, dtype=np.int64)
        arr[0, 0] = 2**63 - 1
        da_in = xr.DataArray(arr, dims=("y", "x"))
        path = os.path.join(str(tmp_path), "t.tif")
        to_geotiff(da_in, path, nodata=2**63 - 1)
        out = read_geotiff_dask(path, chunks=16).compute()
        assert out.dtype == np.float64
        assert np.isnan(out.values[0, 0])


# ---------------------------------------------------------------------------
# write_vrt -> read_vrt round-trip -- the path that surfaced the bug
# in the wild (write_vrt stringifies geo_info.nodata into XML).
# ---------------------------------------------------------------------------


class TestVrtRoundTrip:
    def test_uint64_max_round_trip_via_vrt(self, tmp_path):
        arr = np.full((16, 16), 100, dtype=np.uint64)
        arr[0, 0] = 2**64 - 1
        da_in = xr.DataArray(arr, dims=("y", "x"))
        tif_path = os.path.join(str(tmp_path), "t.tif")
        to_geotiff(da_in, tif_path, nodata=2**64 - 1)

        vrt_path = os.path.join(str(tmp_path), "t.vrt")
        write_vrt(vrt_path, [tif_path])

        # The VRT XML should carry the integer string literal, not a
        # scientific-notation float that loses one ULP at the dtype max.
        with open(vrt_path) as f:
            xml = f.read()
        assert "<NoDataValue>18446744073709551615</NoDataValue>" in xml

        out = read_vrt(vrt_path)
        assert out.dtype == np.float64
        assert np.isnan(out.values[0, 0])
        assert out.values[1, 1] == 100.0
        assert out.attrs["nodata"] == 2**64 - 1

    def test_int64_max_round_trip_via_vrt(self, tmp_path):
        arr = np.full((16, 16), 100, dtype=np.int64)
        arr[0, 0] = 2**63 - 1
        da_in = xr.DataArray(arr, dims=("y", "x"))
        tif_path = os.path.join(str(tmp_path), "t.tif")
        to_geotiff(da_in, tif_path, nodata=2**63 - 1)

        vrt_path = os.path.join(str(tmp_path), "t.vrt")
        write_vrt(vrt_path, [tif_path])

        with open(vrt_path) as f:
            xml = f.read()
        assert "<NoDataValue>9223372036854775807</NoDataValue>" in xml

        out = read_vrt(vrt_path)
        assert out.dtype == np.float64
        assert np.isnan(out.values[0, 0])
        assert out.values[1, 1] == 100.0


# ---------------------------------------------------------------------------
# GPU path parity (gated on cupy availability)
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialised."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


class TestGpuPathParity:
    @_gpu_only
    def test_uint64_max_masked_via_gpu(self, tmp_path):
        from xrspatial.geotiff import read_geotiff_gpu

        arr = np.full((16, 16), 100, dtype=np.uint64)
        arr[0, 0] = 2**64 - 1
        da_in = xr.DataArray(arr, dims=("y", "x"))
        path = os.path.join(str(tmp_path), "t.tif")
        to_geotiff(da_in, path, nodata=2**64 - 1)

        gpu_da = read_geotiff_gpu(path)
        host = gpu_da.data.get()
        assert host.dtype == np.float64
        assert np.isnan(host[0, 0])
        assert host[1, 1] == 100.0
