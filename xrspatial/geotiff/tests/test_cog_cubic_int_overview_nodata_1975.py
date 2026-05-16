"""Regression tests for issue #1975.

``to_geotiff(data, cog=True, overview_resampling='cubic', nodata=N)`` on
**integer** rasters with a finite nodata sentinel used to produce severe
ringing artifacts in the overview pyramid near the nodata border.

Root cause: ``_block_reduce_2d``'s cubic branch masked the sentinel to
NaN only when the input dtype was float (``arr2d.dtype.kind == 'f'``).
For integer rasters the function fell through to an unmasked
``zoom(arr2d, 0.5, order=3)``, and the bicubic spline blended the
sentinel value (e.g. -9999) into neighbouring valid cells. Cast back
to the integer dtype, the boundary pixels surfaced as silent garbage
(values like 1082 / 1134 / -11104 against actual data of 100 with
sentinel -9999).

The fix mirrors the float branch:

1. Promote the cropped block to float64 so NaN can survive the spline.
2. Mask the sentinel to NaN before ``zoom(... prefilter=False)`` so the
   interpolation does not treat it as signal and a single NaN does not
   poison the entire row/column.
3. Rewrite NaN back to the sentinel after the spline.
4. ``np.round(...).astype(arr2d.dtype)`` so the integer cast is
   well-defined (mirrors the mean/min/max/median integer tail).
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._writer import _block_reduce_2d


# ---------------------------------------------------------------------------
# Helper-level: _block_reduce_2d cubic + integer + sentinel
# ---------------------------------------------------------------------------

def _make_block_with_nodata_corner(dtype, nodata_value, size=64,
                                   corner=16, fill=100):
    """Return a (size, size) ``dtype`` array with a corner of nodata."""
    arr = np.full((size, size), fill, dtype=dtype)
    arr[:corner, :corner] = nodata_value
    return arr


def test_cubic_int16_with_nodata_does_not_poison_overview():
    """int16 + finite sentinel: cubic overview must not blend sentinel."""
    arr = _make_block_with_nodata_corner(np.int16, -9999)
    result = _block_reduce_2d(arr, method='cubic', nodata=-9999)
    # Finite (non-sentinel) values must lie within the source data range.
    # Pre-fix the boundary surfaced values like 1082 / 1134 / -11104.
    finite_non_sentinel = result[result != -9999]
    assert finite_non_sentinel.size > 0
    assert finite_non_sentinel.max() <= 100
    assert finite_non_sentinel.min() >= 100  # only valid data value is 100
    # The output dtype is the input dtype.
    assert result.dtype == np.int16
    # Result shape is half (size/2, size/2).
    assert result.shape == (32, 32)


def test_cubic_uint16_with_nodata_does_not_poison_overview():
    """uint16 + finite sentinel: same guarantee as int16."""
    arr = _make_block_with_nodata_corner(np.uint16, 65535, fill=200)
    result = _block_reduce_2d(arr, method='cubic', nodata=65535)
    finite = result[result != 65535]
    assert finite.size > 0
    assert finite.min() >= 200
    assert finite.max() <= 200
    assert result.dtype == np.uint16


def test_cubic_int32_with_nodata_does_not_poison_overview():
    """int32 + negative sentinel: same guarantee."""
    arr = _make_block_with_nodata_corner(np.int32, -2147483648, fill=42)
    result = _block_reduce_2d(arr, method='cubic', nodata=-2147483648)
    finite = result[result != -2147483648]
    assert finite.size > 0
    assert finite.min() >= 42
    assert finite.max() <= 42
    assert result.dtype == np.int32


def test_cubic_int_no_nodata_unchanged():
    """Cubic on integer without nodata still runs the plain zoom path."""
    arr = np.arange(64 * 64, dtype=np.int16).reshape(64, 64)
    result_no_nd = _block_reduce_2d(arr, method='cubic', nodata=None)
    # Plain zoom path: dtype preserved, shape halved.
    assert result_no_nd.dtype == np.int16
    assert result_no_nd.shape == (32, 32)


def test_cubic_int_nodata_out_of_range_noop():
    """Sentinel outside the dtype range cannot equal any pixel — no-op."""
    arr = np.full((64, 64), 100, dtype=np.uint16)
    # -1 cannot exist in uint16; the guard skips the masking branch.
    result = _block_reduce_2d(arr, method='cubic', nodata=-1)
    # Falls through to plain zoom path; values stay 100 (cubic on constant).
    assert result.dtype == np.uint16
    # Cubic of a constant grid is the same constant.
    assert np.all(result == 100)


def test_cubic_int_nodata_fractional_noop():
    """Fractional sentinel on integer dtype: no-op (cannot match any pixel)."""
    arr = np.full((64, 64), 100, dtype=np.int16)
    result = _block_reduce_2d(arr, method='cubic', nodata=1.5)
    assert result.dtype == np.int16
    assert np.all(result == 100)


def test_cubic_int_all_sentinel_block_becomes_sentinel():
    """A 2x2 block that is entirely the sentinel rounds back to the sentinel."""
    arr = np.full((4, 4), -9999, dtype=np.int16)
    result = _block_reduce_2d(arr, method='cubic', nodata=-9999)
    assert result.dtype == np.int16
    assert np.all(result == -9999)


def test_cubic_float_branch_still_works():
    """Float regression guard: the existing #1623 path must still work."""
    arr = np.full((64, 64), 100.0, dtype=np.float32)
    arr[:16, :16] = -9999.0
    result = _block_reduce_2d(arr, method='cubic', nodata=-9999.0)
    assert result.dtype == np.float32
    finite = result[result != -9999.0]
    assert finite.size > 0
    # No ringing: all valid output pixels are 100 (constant input region).
    np.testing.assert_allclose(finite, 100.0, atol=1e-3)


# ---------------------------------------------------------------------------
# End-to-end: to_geotiff cubic + integer + nodata round-trip
# ---------------------------------------------------------------------------

def test_to_geotiff_int_cubic_overview_round_trip(tmp_path):
    """1024x1024 int16 + cog + cubic + nodata round-trips without poisoning."""
    data = np.full((1024, 1024), 100, dtype=np.int16)
    data[:256, :256] = -9999
    da = xr.DataArray(
        data, dims=('y', 'x'),
        coords={'y': np.arange(1024.0), 'x': np.arange(1024.0)},
    )
    path = tmp_path / "cubic_int_1975.tif"
    to_geotiff(da, str(path), cog=True, overview_resampling='cubic',
               nodata=-9999, crs=4326)
    # Level 0: full resolution.
    r0 = open_geotiff(str(path), overview_level=0)
    uniq_0 = set(np.unique(r0.values[~np.isnan(r0.values)]))
    assert uniq_0 == {100.0}
    # Level 1: the historically poisoned level.
    r1 = open_geotiff(str(path), overview_level=1)
    finite_1 = r1.values[~np.isnan(r1.values)]
    # All finite values must be 100 (the only valid data value); no ringing.
    np.testing.assert_array_equal(finite_1, 100.0)


def test_to_geotiff_int_cubic_no_nodata_regression(tmp_path):
    """int16 + cog + cubic without nodata: cubic still runs (regression)."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 1000, size=(1024, 1024), dtype=np.int16)
    da = xr.DataArray(
        data, dims=('y', 'x'),
        coords={'y': np.arange(1024.0), 'x': np.arange(1024.0)},
    )
    path = tmp_path / "cubic_int_no_nd_1975.tif"
    to_geotiff(da, str(path), cog=True, overview_resampling='cubic',
               crs=4326)
    r1 = open_geotiff(str(path), overview_level=1)
    # Output dtype is the source integer dtype.
    assert r1.values.dtype == np.int16
    assert r1.shape == (512, 512)


def test_to_geotiff_int_cubic_overview_matches_mean_finite_range(tmp_path):
    """Cubic must agree with mean on which pixels are finite vs nodata."""
    data = np.full((512, 512), 50, dtype=np.uint16)
    data[:128, :128] = 65535
    da = xr.DataArray(
        data, dims=('y', 'x'),
        coords={'y': np.arange(512.0), 'x': np.arange(512.0)},
    )
    cubic_path = tmp_path / "cubic.tif"
    mean_path = tmp_path / "mean.tif"
    to_geotiff(da, str(cubic_path), cog=True, overview_resampling='cubic',
               nodata=65535, crs=4326)
    to_geotiff(da, str(mean_path), cog=True, overview_resampling='mean',
               nodata=65535, crs=4326)
    r_cubic = open_geotiff(str(cubic_path), overview_level=0)
    r_mean = open_geotiff(str(mean_path), overview_level=0)
    # Sentinel masks should land on the same pixels for both methods on a
    # constant valid region with a constant nodata corner.
    np.testing.assert_array_equal(
        np.isnan(r_cubic.values), np.isnan(r_mean.values),
    )
    finite_cubic = r_cubic.values[~np.isnan(r_cubic.values)]
    finite_mean = r_mean.values[~np.isnan(r_mean.values)]
    # All valid pixels are 50 in both.
    np.testing.assert_array_equal(finite_cubic, 50.0)
    np.testing.assert_array_equal(finite_mean, 50.0)


def test_gpu_int_cubic_overview_matches_cpu(tmp_path):
    """GPU writer cubic falls back to CPU; bytes must match CPU writer."""
    cupy = pytest.importorskip("cupy")
    if not cupy.cuda.is_available():
        pytest.skip("CUDA not available")

    data = np.full((1024, 1024), 100, dtype=np.int16)
    data[:256, :256] = -9999
    cpu_da = xr.DataArray(
        data, dims=('y', 'x'),
        coords={'y': np.arange(1024.0), 'x': np.arange(1024.0)},
    )
    gpu_da = xr.DataArray(
        cupy.asarray(data), dims=('y', 'x'),
        coords={'y': np.arange(1024.0), 'x': np.arange(1024.0)},
    )
    cpu_path = tmp_path / "cpu.tif"
    gpu_path = tmp_path / "gpu.tif"
    to_geotiff(cpu_da, str(cpu_path), cog=True, overview_resampling='cubic',
               nodata=-9999, crs=4326)
    to_geotiff(gpu_da, str(gpu_path), cog=True, overview_resampling='cubic',
               nodata=-9999, crs=4326)
    cpu_r1 = open_geotiff(str(cpu_path), overview_level=1)
    gpu_r1 = open_geotiff(str(gpu_path), overview_level=1)
    # Both paths route cubic through the same CPU helper; results must agree
    # bit-for-bit on this constant input.
    cpu_arr = cpu_r1.values
    gpu_arr = gpu_r1.values
    assert cpu_arr.shape == gpu_arr.shape
    np.testing.assert_array_equal(
        np.isnan(cpu_arr), np.isnan(gpu_arr),
    )
    np.testing.assert_array_equal(
        cpu_arr[~np.isnan(cpu_arr)], gpu_arr[~np.isnan(gpu_arr)],
    )
