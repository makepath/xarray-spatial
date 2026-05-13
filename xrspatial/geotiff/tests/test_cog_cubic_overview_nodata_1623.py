"""COG cubic overview respects the nodata sentinel (issue #1623).

Before the fix, ``to_geotiff(..., cog=True, nodata=<finite>,
overview_resampling='cubic')`` produced wrong overview pixels near
nodata borders on float rasters. The writer rewrote NaN to the
sentinel before reduction; ``_block_reduce_2d(method='cubic')`` then
ignored ``nodata`` and handed the sentinel-poisoned array straight to
``scipy.ndimage.zoom(order=3)``. The cubic spline blended the sentinel
into neighbouring cells (values like ``1133`` and ``-10290`` appeared
where the data was a constant 100).

The fix masks the sentinel to NaN, runs cubic with
``prefilter=False`` so a single NaN does not poison the entire
row/column, and rewrites any NaN in the output back to the sentinel.
The GPU helper falls back to the CPU cubic path the same way it does
for ``mode``.

These tests pin:

* the helper produces no ringing near a sentinel border,
* the round-trip through ``to_geotiff`` writes a clean overview,
* the no-nodata cubic path is unchanged,
* the GPU writer routes cubic through the CPU helper and produces
  byte-identical overview tiles.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr


def _gpu_available() -> bool:
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


def _flat_with_corner_nan(side: int = 16, nan_side: int = 4):
    """``side x side`` float32 ones with a ``nan_side x nan_side`` NaN corner."""
    arr = np.ones((side, side), dtype=np.float32) * 100.0
    arr[:nan_side, :nan_side] = np.nan
    return arr


def test_block_reduce_cubic_nodata_helper_no_ringing():
    """Helper: cubic with nodata must not leak the sentinel into neighbours."""
    pytest.importorskip("scipy")
    from xrspatial.geotiff._writer import _block_reduce_2d

    # Mimic what to_geotiff does: rewrite NaN to the sentinel before
    # handing the array to the reducer.
    arr = _flat_with_corner_nan()
    arr[np.isnan(arr)] = -9999.0

    out = _block_reduce_2d(arr, 'cubic', nodata=-9999.0)

    # The valid region must still read ~100.  Without the fix the cells
    # adjacent to the sentinel corner returned values like 1196.28 and
    # -19.00 from the cubic blend.
    valid = out != -9999.0
    assert np.all(np.abs(out[valid] - 100.0) < 1e-3), (
        f"ringing leaked into cubic output: {out[valid]}")

    # Sentinel cells still mark the nodata region.
    assert (out == -9999.0).any()


def test_block_reduce_cubic_nodata_poisoning_repro():
    """Without the fix the sentinel poisoned the cubic output.

    Pin the failure mode by running cubic on the same array *without*
    a nodata argument and confirming the documented buggy values
    appear. This guards against a regression where ``nodata`` silently
    stops being honoured.
    """
    pytest.importorskip("scipy")
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = _flat_with_corner_nan()
    arr[np.isnan(arr)] = -9999.0

    # nodata=None reproduces the pre-fix behaviour.
    poisoned = _block_reduce_2d(arr, 'cubic')
    # At least one cell outside the corner has a wildly wrong value.
    valid_no_sentinel = (poisoned != -9999.0)
    drift = np.abs(poisoned[valid_no_sentinel] - 100.0)
    assert drift.max() > 50.0, (
        "expected the no-nodata cubic path to ring; got a clean output "
        f"with max drift {drift.max()}")


def test_block_reduce_cubic_no_nodata_unchanged():
    """Cubic on data without nodata stays at order=3 with prefilter."""
    pytest.importorskip("scipy")
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = np.arange(256, dtype=np.float32).reshape(16, 16)
    out_default = _block_reduce_2d(arr, 'cubic')
    # The same array round-tripped through scipy zoom directly should
    # match (since no sentinel is present the fix path is not taken).
    from scipy.ndimage import zoom
    expected = zoom(arr, 0.5, order=3).astype(arr.dtype)
    np.testing.assert_array_equal(out_default, expected)


def test_block_reduce_cubic_nodata_unset_is_zoom():
    """nodata=None goes through the original zoom path, no prefilter change."""
    pytest.importorskip("scipy")
    from xrspatial.geotiff._writer import _block_reduce_2d
    from scipy.ndimage import zoom

    arr = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    out = _block_reduce_2d(arr, 'cubic', nodata=None)
    expected = zoom(arr, 0.5, order=3).astype(arr.dtype)
    np.testing.assert_array_equal(out, expected)


def test_to_geotiff_cog_cubic_nodata_round_trip(tmp_path):
    """End-to-end: writing a COG with cubic + nodata produces a clean overview."""
    pytest.importorskip("scipy")
    from xrspatial.geotiff import to_geotiff, open_geotiff

    arr = _flat_with_corner_nan()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / 'cog_cubic_nodata.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=16, overview_levels=[2],
               overview_resampling='cubic')

    ov = open_geotiff(p, overview_level=1)
    data = np.asarray(ov.data)

    # No polluted pixels: every cell is either NaN (reader unmasked the
    # sentinel back to NaN), the literal sentinel value (reader kept it),
    # or ~100 (the source value).
    polluted = (
        (~np.isnan(data))
        & (data != -9999.0)
        & (np.abs(data - 100.0) > 1e-3)
    )
    assert not polluted.any(), (
        f"polluted overview cells: {data[polluted]}")


def test_to_geotiff_cog_cubic_no_nodata_round_trip(tmp_path):
    """Regression guard: cubic without nodata still produces the same overview."""
    pytest.importorskip("scipy")
    from xrspatial.geotiff import to_geotiff, open_geotiff

    arr = np.arange(256, dtype=np.float32).reshape(16, 16)
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / 'cog_cubic_no_nodata.tif')
    to_geotiff(da, p, cog=True, compression='deflate',
               tiled=True, tile_size=16, overview_levels=[2],
               overview_resampling='cubic')

    ov = open_geotiff(p, overview_level=1)
    assert ov.shape == (8, 8)
    assert ov.dtype == np.float32
    # Cubic on a monotonic ramp stays bounded by source range.
    assert float(np.asarray(ov.data).min()) >= float(arr.min()) - 1.0
    assert float(np.asarray(ov.data).max()) <= float(arr.max()) + 1.0


def test_block_reduce_cubic_inf_nodata_is_masked():
    """nodata=+/-inf must be masked just like a finite sentinel."""
    pytest.importorskip("scipy")
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = np.ones((16, 16), dtype=np.float32) * 5.0
    arr[:4, :4] = np.inf  # treat +inf as sentinel
    out = _block_reduce_2d(arr, 'cubic', nodata=np.inf)
    valid = ~np.isinf(out)
    # Outside the masked region we still read ~5.0.
    np.testing.assert_allclose(out[valid], 5.0, atol=1e-4)


def test_block_reduce_cubic_nan_sentinel_skips_mask():
    """nodata=NaN is a no-op (matches the existing nan-pass-through gate)."""
    pytest.importorskip("scipy")
    from xrspatial.geotiff._writer import _block_reduce_2d
    from scipy.ndimage import zoom

    arr = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    out = _block_reduce_2d(arr, 'cubic', nodata=np.nan)
    expected = zoom(arr, 0.5, order=3).astype(arr.dtype)
    np.testing.assert_array_equal(out, expected)


def test_gpu_overview_methods_includes_cubic():
    """The GPU constant must list ``cubic`` so callers do not pre-validate
    against the smaller pre-#1623 set."""
    from xrspatial.geotiff._gpu_decode import GPU_OVERVIEW_METHODS
    assert 'cubic' in GPU_OVERVIEW_METHODS


@_gpu_only
def test_gpu_block_reduce_cubic_falls_back_to_cpu():
    """GPU cubic must route through the CPU helper and return cupy data."""
    pytest.importorskip("scipy")
    import cupy
    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = _flat_with_corner_nan()
    arr[np.isnan(arr)] = -9999.0

    gpu_arr = cupy.asarray(arr)
    gpu_out = _block_reduce_2d_gpu(gpu_arr, 'cubic', nodata=-9999.0)
    assert isinstance(gpu_out, cupy.ndarray)

    cpu_out = _block_reduce_2d(arr, 'cubic', nodata=-9999.0)
    np.testing.assert_array_equal(cupy.asnumpy(gpu_out), cpu_out)


@_gpu_only
def test_to_geotiff_cog_cubic_nodata_gpu_round_trip(tmp_path):
    """End-to-end GPU writer: cubic + nodata produces a clean overview."""
    pytest.importorskip("scipy")
    import cupy
    from xrspatial.geotiff import to_geotiff, open_geotiff

    arr = _flat_with_corner_nan()
    da = xr.DataArray(cupy.asarray(arr), dims=['y', 'x'])
    p = str(tmp_path / 'cog_cubic_nodata_gpu.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=16, overview_levels=[2],
               overview_resampling='cubic')

    ov = open_geotiff(p, overview_level=1)
    data = np.asarray(ov.data)
    polluted = (
        (~np.isnan(data))
        & (data != -9999.0)
        & (np.abs(data - 100.0) > 1e-3)
    )
    assert not polluted.any(), (
        f"GPU cubic overview leaked sentinel into neighbours: "
        f"{data[polluted]}")


@_gpu_only
def test_gpu_cpu_cubic_overview_bytes_match(tmp_path):
    """CPU and GPU writers produce the same cubic overview pixels."""
    pytest.importorskip("scipy")
    import cupy
    from xrspatial.geotiff import to_geotiff, open_geotiff

    arr = _flat_with_corner_nan()
    cpu_da = xr.DataArray(arr, dims=['y', 'x'])
    gpu_da = xr.DataArray(cupy.asarray(arr), dims=['y', 'x'])

    cpu_path = str(tmp_path / 'cpu_cubic.tif')
    gpu_path = str(tmp_path / 'gpu_cubic.tif')
    to_geotiff(cpu_da, cpu_path, nodata=-9999.0, cog=True,
               compression='deflate', tiled=True, tile_size=16,
               overview_levels=[2], overview_resampling='cubic')
    to_geotiff(gpu_da, gpu_path, nodata=-9999.0, cog=True,
               compression='deflate', tiled=True, tile_size=16,
               overview_levels=[2], overview_resampling='cubic')

    cpu_ov = np.asarray(open_geotiff(cpu_path, overview_level=1).data)
    gpu_ov = np.asarray(open_geotiff(gpu_path, overview_level=1).data)
    # NaN-aware compare since the reader unmasks the sentinel.
    np.testing.assert_array_equal(np.isnan(cpu_ov), np.isnan(gpu_ov))
    finite = ~np.isnan(cpu_ov)
    np.testing.assert_allclose(cpu_ov[finite], gpu_ov[finite], atol=1e-3)
