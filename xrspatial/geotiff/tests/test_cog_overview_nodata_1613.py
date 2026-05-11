"""COG overview generation respects the nodata sentinel (issue #1613).

Before the fix, ``to_geotiff(..., cog=True, nodata=<finite>)`` rewrote NaN
to the sentinel *before* the overview-generation loop. ``_make_overview``
then ran ``np.nanmean`` / ``np.nanmin`` / ``np.nanmax`` / ``np.nanmedian``
over the rewritten array, treating the sentinel as a real number and
biasing every overview pixel toward the sentinel.

These tests pin the contract that the CPU and GPU writers ignore the
sentinel during overview reduction, so the resulting pyramid matches
``np.nanmean``-style aggregation on the original NaN-keyed data.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr


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


def _arr_with_partial_nan():
    """4x4 float raster: row 1 is all-NaN, rest is finite."""
    return np.array([
        [1.0, 2.0, 3.0, 4.0],
        [np.nan, np.nan, np.nan, np.nan],
        [10.0, 20.0, 30.0, 40.0],
        [10.0, 20.0, 30.0, 40.0],
    ], dtype=np.float32)


def _arr_with_full_nan_block():
    """4x4 float raster: top-left 2x2 entirely NaN."""
    return np.array([
        [np.nan, np.nan, 3.0, 4.0],
        [np.nan, np.nan, 7.0, 8.0],
        [10.0, 20.0, 30.0, 40.0],
        [10.0, 20.0, 30.0, 40.0],
    ], dtype=np.float32)


def test_cpu_cog_overview_mean_ignores_sentinel(tmp_path):
    """CPU writer: overview 'mean' must skip sentinel pixels (issue #1613)."""
    from xrspatial.geotiff import to_geotiff, open_geotiff

    arr = _arr_with_partial_nan()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / 'cog_mean_nodata.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=2, overview_levels=[1],
               overview_resampling='mean')

    ov = open_geotiff(p, overview_level=1)
    expected = np.array([[1.5, 3.5], [15.0, 35.0]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(ov.data), expected)


def test_cpu_cog_overview_mean_partial_block(tmp_path):
    """CPU writer: partial-NaN 2x2 block averages over the finite cells only."""
    from xrspatial.geotiff import to_geotiff, open_geotiff

    arr = _arr_with_full_nan_block()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / 'cog_mean_nodata_full_block.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=2, overview_levels=[1],
               overview_resampling='mean')

    ov = open_geotiff(p, overview_level=1)
    # Top-left 2x2 was all-NaN -> reduces to NaN -> rewritten to -9999
    # Top-right 2x2 [3,4,7,8] -> mean 5.5
    # Bottom-left [10,20,10,20] -> 15
    # Bottom-right [30,40,30,40] -> 35
    data = np.asarray(ov.data)
    assert data[0, 0] == -9999.0
    np.testing.assert_allclose(data[0, 1], 5.5)
    np.testing.assert_allclose(data[1, 0], 15.0)
    np.testing.assert_allclose(data[1, 1], 35.0)


@pytest.mark.parametrize('method,expected', [
    ('min', np.array([[1.0, 3.0], [10.0, 30.0]], dtype=np.float32)),
    ('max', np.array([[2.0, 4.0], [20.0, 40.0]], dtype=np.float32)),
    ('median', np.array([[1.5, 3.5], [15.0, 35.0]], dtype=np.float32)),
])
def test_cpu_cog_overview_aggregations_ignore_sentinel(
        tmp_path, method, expected):
    """min/max/median overview reductions must also skip the sentinel."""
    from xrspatial.geotiff import to_geotiff, open_geotiff

    arr = _arr_with_partial_nan()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / f'cog_{method}_nodata.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=2, overview_levels=[1],
               overview_resampling=method)

    ov = open_geotiff(p, overview_level=1)
    np.testing.assert_allclose(np.asarray(ov.data), expected)


def test_cpu_cog_overview_mean_no_nodata_passes(tmp_path):
    """When nodata is unset the reducer behaves as before."""
    from xrspatial.geotiff import to_geotiff, open_geotiff

    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / 'cog_mean_no_nodata.tif')
    to_geotiff(da, p, cog=True, compression='deflate',
               tiled=True, tile_size=2, overview_levels=[1],
               overview_resampling='mean')

    ov = open_geotiff(p, overview_level=1)
    # mean of 2x2 blocks of arange(16).reshape(4,4)
    expected = np.array([
        [(0 + 1 + 4 + 5) / 4, (2 + 3 + 6 + 7) / 4],
        [(8 + 9 + 12 + 13) / 4, (10 + 11 + 14 + 15) / 4],
    ], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(ov.data), expected)


def test_block_reduce_2d_nodata_kwarg_directly():
    """Exercise the helper directly so a regression here is caught fast."""
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = _arr_with_partial_nan()
    # Without nodata, the sentinel poisons the reduction.
    arr_sentinel = arr.copy()
    arr_sentinel[np.isnan(arr_sentinel)] = -9999.0
    poisoned = _block_reduce_2d(arr_sentinel, 'mean')
    assert poisoned[0, 0] < -1000.0  # confirms the bug shape

    # With nodata, the sentinel is treated as missing.
    fixed = _block_reduce_2d(arr_sentinel, 'mean', nodata=-9999.0)
    np.testing.assert_allclose(fixed[0, 0], 1.5)
    np.testing.assert_allclose(fixed[0, 1], 3.5)


def test_block_reduce_2d_nodata_all_sentinel_block_yields_nan():
    """All-sentinel block reduces to NaN under nan-aware aggregation."""
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = np.full((2, 2), -9999.0, dtype=np.float32)
    out = _block_reduce_2d(arr, 'mean', nodata=-9999.0)
    assert out.shape == (1, 1)
    assert np.isnan(out[0, 0])


@_gpu_only
def test_gpu_cog_overview_mean_ignores_sentinel(tmp_path):
    """GPU writer: overview 'mean' must skip sentinel pixels (issue #1613)."""
    import cupy
    from xrspatial.geotiff import to_geotiff, open_geotiff

    arr_cpu = _arr_with_partial_nan()
    arr_gpu = cupy.asarray(arr_cpu)
    da = xr.DataArray(arr_gpu, dims=['y', 'x'])

    p = str(tmp_path / 'gpu_cog_mean_nodata.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=2, overview_levels=[1],
               overview_resampling='mean', gpu=True)

    ov = open_geotiff(p, overview_level=1)
    expected = np.array([[1.5, 3.5], [15.0, 35.0]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(ov.data), expected)


@_gpu_only
def test_gpu_block_reduce_nodata_kwarg_directly():
    """Exercise the GPU helper directly so a regression is caught fast."""
    import cupy
    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr_cpu = _arr_with_partial_nan()
    arr_cpu[np.isnan(arr_cpu)] = -9999.0
    arr_gpu = cupy.asarray(arr_cpu)

    poisoned = _block_reduce_2d_gpu(arr_gpu, 'mean')
    assert float(poisoned[0, 0].get()) < -1000.0

    fixed = _block_reduce_2d_gpu(arr_gpu, 'mean', nodata=-9999.0)
    np.testing.assert_allclose(float(fixed[0, 0].get()), 1.5)
    np.testing.assert_allclose(float(fixed[0, 1].get()), 3.5)


@_gpu_only
def test_gpu_cog_overview_matches_cpu(tmp_path):
    """CPU and GPU overview pyramids must agree on nodata-masked data."""
    import cupy
    from xrspatial.geotiff import to_geotiff, open_geotiff

    arr = _arr_with_partial_nan()

    # CPU
    da_cpu = xr.DataArray(arr, dims=['y', 'x'])
    p_cpu = str(tmp_path / 'cpu_pyramid.tif')
    to_geotiff(da_cpu, p_cpu, nodata=-9999.0, cog=True,
               compression='deflate', tiled=True, tile_size=2,
               overview_levels=[1], overview_resampling='mean')
    cpu_ov = np.asarray(open_geotiff(p_cpu, overview_level=1).data)

    # GPU
    da_gpu = xr.DataArray(cupy.asarray(arr), dims=['y', 'x'])
    p_gpu = str(tmp_path / 'gpu_pyramid.tif')
    to_geotiff(da_gpu, p_gpu, nodata=-9999.0, cog=True,
               compression='deflate', tiled=True, tile_size=2,
               overview_levels=[1], overview_resampling='mean', gpu=True)
    gpu_ov = np.asarray(open_geotiff(p_gpu, overview_level=1).data)

    np.testing.assert_allclose(cpu_ov, gpu_ov)
