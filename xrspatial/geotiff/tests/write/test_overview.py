"""Overview-level and nodata-aware overview tests.

Consolidates ``test_cog_overview_ceil_2105.py``,
``test_cog_overview_nodata_1613.py``,
``test_cog_cubic_overview_nodata_1623.py``,
``test_cog_cubic_int_overview_nodata_1975.py``, and
``test_cog_int_overview_nodata_2026_05_12.py`` into one overview-policy
module. Tests-only restructure for epic #2390.
"""

from __future__ import annotations

import importlib.util
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff._writer import _block_reduce_2d
from xrspatial.geotiff import open_geotiff, to_geotiff


# -------------------------------------------------------------------------
# Folded from: test_cog_overview_ceil_2105.py
# -------------------------------------------------------------------------

def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


# ---------------------------------------------------------------------------
# Output-shape contract: ceil((h+1)/2, (w+1)/2) for every method and dtype.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,expected",
    [
        ((5, 5), (3, 3)),
        ((5, 4), (3, 2)),
        ((4, 5), (2, 3)),
        ((7, 3), (4, 2)),
        ((1, 1), (1, 1)),
        ((1, 5), (1, 3)),
        ((5, 1), (3, 1)),
        ((4, 4), (2, 2)),
        ((6, 6), (3, 3)),
    ],
)
@pytest.mark.parametrize(
    "method", ["nearest", "mean", "min", "max", "median", "mode"]
)
def test_ceil_output_shape_float(shape, expected, method):
    arr = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    out = _block_reduce_2d(arr, method)
    assert out.shape == expected


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((5, 5), (3, 3)),
        ((3, 7), (2, 4)),
        ((1, 1), (1, 1)),
    ],
)
@pytest.mark.parametrize("method", ["nearest", "mean", "min", "max", "median", "mode"])
def test_ceil_output_shape_int(shape, expected, method):
    arr = np.arange(shape[0] * shape[1], dtype=np.int16).reshape(shape)
    out = _block_reduce_2d(arr, method)
    assert out.shape == expected
    assert out.dtype == arr.dtype


def test_ceil_output_shape_cubic_float():
    pytest.importorskip("scipy")
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    out = _block_reduce_2d(arr, "cubic")
    assert out.shape == (3, 3)


def test_ceil_output_shape_cubic_int():
    pytest.importorskip("scipy")
    arr = np.arange(25, dtype=np.int16).reshape(5, 5)
    out = _block_reduce_2d(arr, "cubic")
    assert out.shape == (3, 3)
    assert out.dtype == arr.dtype


# ---------------------------------------------------------------------------
# Trailing-edge pixel values: the last row/col of the source must reach the
# overview rather than being dropped.
# ---------------------------------------------------------------------------
def test_nearest_5x5_preserves_trailing_pixels():
    arr = np.arange(25, dtype=np.float64).reshape(5, 5)
    out = _block_reduce_2d(arr, "nearest")
    # Nearest = top-left of every 2x2 block. With ceil, that's arr[::2, ::2].
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out, arr[::2, ::2])
    # The trailing row/col of the source IS represented in the overview.
    assert out[-1, -1] == arr[4, 4]


def test_mean_5x5_trailing_residual_block_uses_valid_cell():
    # Residual at row 4, col 4 is a 1x1 block containing arr[4,4] alone.
    arr = np.zeros((5, 5), dtype=np.float32)
    arr[4, 4] = 100.0
    out = _block_reduce_2d(arr, "mean")
    assert out.shape == (3, 3)
    # The corner residual is a 1x1 block, so its mean is the single pixel.
    assert out[2, 2] == pytest.approx(100.0)


def test_max_5x5_residual_block_uses_valid_cell():
    arr = np.zeros((5, 5), dtype=np.float32)
    arr[4, :] = 9.0  # trailing row should reach overview[2, :]
    arr[:, 4] = 7.0
    out = _block_reduce_2d(arr, "max")
    assert out.shape == (3, 3)
    # Bottom overview row picks max of (arr[4, 2*j:2*j+2]) -> 9.0 everywhere.
    np.testing.assert_array_equal(out[2, :2], [9.0, 9.0])
    # Right column gets max from (arr[2*i:2*i+2, 4]) -> 7.0 except corner.
    assert out[0, 2] == 7.0
    assert out[1, 2] == 7.0
    # arr[4, 4] = 7.0 (set by the trailing-column sweep, after row sweep).
    assert out[2, 2] == 7.0


def test_min_5x5_residual_block_uses_valid_cell():
    arr = np.full((5, 5), 10.0, dtype=np.float32)
    arr[4, 4] = -1.0
    out = _block_reduce_2d(arr, "min")
    assert out[2, 2] == -1.0


def test_median_5x5_residual_block_uses_valid_cell():
    arr = np.full((5, 5), 5.0, dtype=np.float32)
    arr[4, 4] = 99.0
    out = _block_reduce_2d(arr, "median")
    # 1x1 residual: median is the single value.
    assert out[2, 2] == pytest.approx(99.0)


def test_mode_5x5_residual_block_picks_valid_cell():
    arr = np.array(
        [[1, 1, 2, 2, 3],
         [1, 1, 2, 2, 3],
         [4, 4, 5, 5, 6],
         [4, 4, 5, 5, 6],
         [7, 7, 8, 8, 9]],
        dtype=np.int16,
    )
    out = _block_reduce_2d(arr, "mode")
    assert out.shape == (3, 3)
    # Trailing 1x1 block at (4,4) is just the value 9.
    assert out[2, 2] == 9
    # Trailing column (rows 0..1 / 2..3, col 4) is 1x1 blocks containing 3 / 6.
    assert out[0, 2] == 3
    assert out[1, 2] == 6
    # Trailing row (col 0..1 / 2..3, row 4) is 1x2 blocks: [7,7] -> 7, [8,8] -> 8.
    assert out[2, 0] == 7
    assert out[2, 1] == 8


def test_cubic_5x5_covers_source_extent():
    pytest.importorskip("scipy")
    # Smoothly varying ramp so cubic interpolation is well-behaved.
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    out = _block_reduce_2d(arr, "cubic")
    assert out.shape == (3, 3)
    # Output should not be entirely zero/NaN, and trailing corner should
    # roughly reflect the high source values around (4, 4).
    assert np.isfinite(out).all()
    assert out[2, 2] > out[0, 0]


# ---------------------------------------------------------------------------
# Sentinel masking still works on odd-sized inputs.
# ---------------------------------------------------------------------------
def test_mean_5x5_with_nodata_excludes_sentinel_in_residual():
    sentinel = -9999.0
    arr = np.full((5, 5), 1.0, dtype=np.float32)
    arr[4, 4] = sentinel
    out = _block_reduce_2d(arr, "mean", nodata=sentinel)
    # The 1x1 trailing residual is all-sentinel -> all-NaN block, which
    # the post-overview rewrite (in the caller) handles. Here we just
    # confirm the sentinel did not bias the reduction: out[2, 2] is NaN,
    # not (1.0 + sentinel)/2 or similar.
    assert np.isnan(out[2, 2])
    # Other overview cells with at least one valid neighbour stay valid.
    assert out[0, 0] == pytest.approx(1.0)


def test_min_int_5x5_with_nodata_does_not_select_sentinel_in_residual():
    sentinel = -9999
    arr = np.full((5, 5), 10, dtype=np.int16)
    # Trailing column has a sentinel + valid cell in 2x1 residual blocks.
    arr[0, 4] = sentinel
    arr[2, 4] = sentinel
    arr[4, 4] = sentinel
    out = _block_reduce_2d(arr, "min", nodata=sentinel)
    # The 2x1 residual at (0..1, 4) is [-9999, 10] -> min ignoring sentinel = 10.
    assert out[0, 2] == 10
    assert out[1, 2] == 10
    # The 1x1 residual at (4, 4) is sentinel -> rewritten to sentinel.
    assert out[2, 2] == sentinel


def test_int64_sentinel_near_max_masks_in_padded_branch():
    # INT64_MAX is not exactly representable in float64: float(INT64_MAX)
    # rounds up to 2**63, which would miss the sentinel if the mask were
    # computed against the float-padded view. The reader must compute the
    # mask at native integer width before padding.
    sentinel = np.iinfo(np.int64).max
    arr = np.full((5, 5), 10, dtype=np.int64)
    arr[0, 0] = sentinel
    # Pad branch fires because shape (5, 5) is odd.
    out = _block_reduce_2d(arr, "min", nodata=sentinel)
    # Top-left 2x2 block has 1 sentinel + 3 valid 10s. nanmin -> 10
    # (sentinel masked out). If the mask missed the sentinel, the int64
    # value would be cast to float and the float min would pick up the
    # sentinel's value or produce noise; either way out[0,0] would not
    # be 10.
    assert out[0, 0] == 10


def test_uint64_sentinel_near_max_masks_in_padded_branch():
    # UINT64_MAX = 2**64 - 1 is also not exactly representable in float64
    # (float(UINT64_MAX) rounds up to 2**64). The native-width mask path
    # must catch the sentinel for unsigned 64-bit dtypes too.
    sentinel = np.iinfo(np.uint64).max
    arr = np.full((5, 5), 10, dtype=np.uint64)
    arr[0, 0] = sentinel
    out = _block_reduce_2d(arr, "min", nodata=sentinel)
    assert out[0, 0] == 10


def test_float32_padded_branch_keeps_source_dtype():
    # The padded mean/min/max/median branch used to allocate a float64
    # NaN buffer regardless of the source dtype, doubling intermediate
    # memory for an odd-shape float32 read. Verify the helper now keeps
    # the source dtype across the pad so a float32 input round-trips as
    # float32. The contract is checked end-to-end via the output dtype.
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    out = _block_reduce_2d(arr, "mean")
    assert out.dtype == np.float32
    # And the values still match what a manual ceil-mean would produce
    # for the top-left 2x2 block.
    top_left_mean = float(arr[:2, :2].mean())
    assert out[0, 0] == pytest.approx(top_left_mean)


def test_max_int_5x5_with_nodata_does_not_select_sentinel_in_residual():
    sentinel = -9999
    arr = np.full((5, 5), 10, dtype=np.int16)
    arr[4, 4] = sentinel
    out = _block_reduce_2d(arr, "max", nodata=sentinel)
    # The 1x1 corner block is all-sentinel -> sentinel.
    assert out[2, 2] == sentinel
    # Adjacent 2x1 residual (rows 4, cols 0..1) has valid values only.
    assert out[2, 0] == 10
    assert out[2, 1] == 10


# ---------------------------------------------------------------------------
# Even-sized inputs keep the existing fast-path semantics.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method", ["nearest", "mean", "min", "max", "median", "mode"]
)
def test_even_input_matches_legacy_2x2_behaviour(method):
    rng = np.random.default_rng(2105)
    arr = rng.integers(0, 100, size=(6, 8)).astype(np.int16)
    out = _block_reduce_2d(arr, method)
    assert out.shape == (3, 4)
    # Spot-check a single block matches a direct reduction.
    block = arr[0:2, 0:2]
    if method == "nearest":
        assert out[0, 0] == block[0, 0]
    elif method == "mean":
        # Integer outputs are rounded after the float reduction.
        assert out[0, 0] == int(round(block.astype(np.float64).mean()))
    elif method == "min":
        assert out[0, 0] == block.min()
    elif method == "max":
        assert out[0, 0] == block.max()
    elif method == "median":
        assert out[0, 0] == int(round(float(np.median(block))))
    elif method == "mode":
        # Lowest-value tie-break for unique cells.
        vals, counts = np.unique(block, return_counts=True)
        expected = vals[np.argmax(counts)]
        assert out[0, 0] == expected


# ---------------------------------------------------------------------------
# GPU mirror: identical shape and identical values on odd-sized inputs.
# ---------------------------------------------------------------------------
@_gpu_only
@pytest.mark.parametrize(
    "method", ["nearest", "mean", "min", "max", "median"]
)
@pytest.mark.parametrize("shape", [(5, 5), (5, 4), (4, 5), (7, 3)])
def test_gpu_block_reduce_matches_cpu_on_odd_shapes(method, shape):
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    rng = np.random.default_rng(2105)
    arr = rng.random(shape, dtype=np.float32)
    cpu_out = _block_reduce_2d(arr, method)
    gpu_out = _block_reduce_2d_gpu(cupy.asarray(arr), method).get()
    assert gpu_out.shape == cpu_out.shape
    np.testing.assert_allclose(gpu_out, cpu_out, equal_nan=True, rtol=1e-6)


@_gpu_only
def test_gpu_block_reduce_int_5x5_with_nodata():
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    sentinel = -9999
    arr = np.full((5, 5), 10, dtype=np.int16)
    arr[4, 4] = sentinel
    cpu_out = _block_reduce_2d(arr, "max", nodata=sentinel)
    gpu_out = _block_reduce_2d_gpu(cupy.asarray(arr), "max", nodata=sentinel).get()
    np.testing.assert_array_equal(gpu_out, cpu_out)


# -------------------------------------------------------------------------
# Folded from: test_cog_overview_nodata_1613.py
# -------------------------------------------------------------------------


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
    from xrspatial.geotiff import open_geotiff, to_geotiff

    arr = _arr_with_partial_nan()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / 'cog_mean_nodata.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=16, overview_levels=[2],
               overview_resampling='mean')

    ov = open_geotiff(p, overview_level=1)
    expected = np.array([[1.5, 3.5], [15.0, 35.0]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(ov.data), expected)


def test_cpu_cog_overview_mean_partial_block(tmp_path):
    """CPU writer: partial-NaN 2x2 block averages over the finite cells only."""
    from xrspatial.geotiff import open_geotiff, to_geotiff

    arr = _arr_with_full_nan_block()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / 'cog_mean_nodata_full_block.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=16, overview_levels=[2],
               overview_resampling='mean')

    ov = open_geotiff(p, overview_level=1)
    # Top-left 2x2 was all-NaN -> reduces to NaN -> rewritten to -9999
    #   on disk, then read back as NaN once the overview-nodata
    #   inheritance fix (#1739) restores attrs['nodata'] and re-masks
    #   the sentinel.
    # Top-right 2x2 [3,4,7,8] -> mean 5.5
    # Bottom-left [10,20,10,20] -> 15
    # Bottom-right [30,40,30,40] -> 35
    data = np.asarray(ov.data)
    assert ov.attrs.get('nodata') == -9999.0
    assert np.isnan(data[0, 0])
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
    from xrspatial.geotiff import open_geotiff, to_geotiff

    arr = _arr_with_partial_nan()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / f'cog_{method}_nodata.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=16, overview_levels=[2],
               overview_resampling=method)

    ov = open_geotiff(p, overview_level=1)
    np.testing.assert_allclose(np.asarray(ov.data), expected)


def test_cpu_cog_overview_mean_no_nodata_passes(tmp_path):
    """When nodata is unset the reducer behaves as before."""
    from xrspatial.geotiff import open_geotiff, to_geotiff

    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / 'cog_mean_no_nodata.tif')
    to_geotiff(da, p, cog=True, compression='deflate',
               tiled=True, tile_size=16, overview_levels=[2],
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


def test_block_reduce_2d_inf_nodata_is_masked():
    """nodata=+/-inf must be masked back to NaN like a finite sentinel.

    The upstream NaN->sentinel rewrite only gates on ``not np.isnan``,
    so ``nodata=inf`` is a real (if uncommon) caller choice. The reducer
    has to match that gate or it re-poisons the overview with inf.
    """
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [np.inf, np.inf, np.inf, np.inf],
        [10.0, 20.0, 30.0, 40.0],
        [10.0, 20.0, 30.0, 40.0],
    ], dtype=np.float32)
    out = _block_reduce_2d(arr, 'mean', nodata=float('inf'))
    np.testing.assert_allclose(out[0, 0], 1.5)
    np.testing.assert_allclose(out[0, 1], 3.5)


def test_block_reduce_2d_all_nan_block_does_not_warn():
    """All-NaN blocks must not surface RuntimeWarning to user logs."""
    import warnings as _warnings

    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = np.array([
        [-9999.0, -9999.0, 3.0, 4.0],
        [-9999.0, -9999.0, 7.0, 8.0],
    ], dtype=np.float32)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        out = _block_reduce_2d(arr, 'mean', nodata=-9999.0)

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert np.isnan(out[0, 0])
    np.testing.assert_allclose(out[0, 1], 5.5)


@_gpu_only
def test_gpu_cog_overview_mean_ignores_sentinel(tmp_path):
    """GPU writer: overview 'mean' must skip sentinel pixels (issue #1613)."""
    import cupy

    from xrspatial.geotiff import open_geotiff, to_geotiff

    arr_cpu = _arr_with_partial_nan()
    arr_gpu = cupy.asarray(arr_cpu)
    da = xr.DataArray(arr_gpu, dims=['y', 'x'])

    p = str(tmp_path / 'gpu_cog_mean_nodata.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=16, overview_levels=[2],
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
def test_gpu_block_reduce_inf_nodata_is_masked():
    """GPU helper mirrors the CPU isnan-only gate for nodata=inf."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr_cpu = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [np.inf, np.inf, np.inf, np.inf],
        [10.0, 20.0, 30.0, 40.0],
        [10.0, 20.0, 30.0, 40.0],
    ], dtype=np.float32)
    arr_gpu = cupy.asarray(arr_cpu)

    out = _block_reduce_2d_gpu(arr_gpu, 'mean', nodata=float('inf'))
    np.testing.assert_allclose(float(out[0, 0].get()), 1.5)
    np.testing.assert_allclose(float(out[0, 1].get()), 3.5)


@_gpu_only
def test_gpu_cog_overview_matches_cpu(tmp_path):
    """CPU and GPU overview pyramids must agree on nodata-masked data."""
    import cupy

    from xrspatial.geotiff import open_geotiff, to_geotiff

    arr = _arr_with_partial_nan()

    # CPU
    da_cpu = xr.DataArray(arr, dims=['y', 'x'])
    p_cpu = str(tmp_path / 'cpu_pyramid.tif')
    to_geotiff(da_cpu, p_cpu, nodata=-9999.0, cog=True,
               compression='deflate', tiled=True, tile_size=16,
               overview_levels=[2], overview_resampling='mean')
    cpu_ov = np.asarray(open_geotiff(p_cpu, overview_level=1).data)

    # GPU
    da_gpu = xr.DataArray(cupy.asarray(arr), dims=['y', 'x'])
    p_gpu = str(tmp_path / 'gpu_pyramid.tif')
    to_geotiff(da_gpu, p_gpu, nodata=-9999.0, cog=True,
               compression='deflate', tiled=True, tile_size=16,
               overview_levels=[2], overview_resampling='mean', gpu=True)
    gpu_ov = np.asarray(open_geotiff(p_gpu, overview_level=1).data)

    np.testing.assert_allclose(cpu_ov, gpu_ov)


# -------------------------------------------------------------------------
# Folded from: test_cog_cubic_overview_nodata_1623.py
# -------------------------------------------------------------------------


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
    from scipy.ndimage import zoom

    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    out = _block_reduce_2d(arr, 'cubic', nodata=None)
    expected = zoom(arr, 0.5, order=3).astype(arr.dtype)
    np.testing.assert_array_equal(out, expected)


def test_to_geotiff_cog_cubic_nodata_round_trip(tmp_path):
    """End-to-end: writing a COG with cubic + nodata produces a clean overview."""
    pytest.importorskip("scipy")
    from xrspatial.geotiff import open_geotiff, to_geotiff

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
    from xrspatial.geotiff import open_geotiff, to_geotiff

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
    from scipy.ndimage import zoom

    from xrspatial.geotiff._writer import _block_reduce_2d

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

    from xrspatial.geotiff import open_geotiff, to_geotiff

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

    from xrspatial.geotiff import open_geotiff, to_geotiff

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


# -------------------------------------------------------------------------
# Folded from: test_cog_cubic_int_overview_nodata_1975.py
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# Folded from: test_cog_int_overview_nodata_2026_05_12.py
# -------------------------------------------------------------------------


_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


# ---------------------------------------------------------------------------
# Unit-level: _block_reduce_2d on integer dtypes
# ---------------------------------------------------------------------------

def _int_block_partial_sentinel(sentinel, dtype):
    """4x4 integer raster where the right two columns of each row pair
    mix valid and sentinel cells. Block (0, 1) has (100, 100, sentinel,
    sentinel); block (1, 1) has (200, 200, sentinel, sentinel)."""
    arr = np.array([
        [100, 100, 100, 100],
        [100, 100, sentinel, sentinel],
        [200, 200, 200, 200],
        [200, 200, sentinel, sentinel],
    ], dtype=dtype)
    return arr


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
@pytest.mark.parametrize('dtype,sentinel', [
    (np.uint8, 255),
    (np.uint16, 65535),
    (np.int16, -9999),
    (np.int32, -2_000_000_000),
])
def test_block_reduce_int_sentinel_masked(method, dtype, sentinel):
    """Integer overview reductions must skip sentinel cells.

    Before the fix, mean produced averages like ``(100+sentinel)/2`` cast
    back to the integer dtype -- a non-sentinel value that the reader
    leaves untouched. The fix masks the sentinel to NaN before the
    reduction so nan-aware aggregation skips it.
    """
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = _int_block_partial_sentinel(sentinel, dtype)
    out = _block_reduce_2d(arr, method, nodata=sentinel)

    # Every block now has at least one valid 100/200; result should equal
    # the valid value (since for mean/min/max/median over {100, 100} is
    # 100, and over {200, 200} is 200). Neither block has any cell that
    # isn't 100, 200, or sentinel, so the output must be a subset of
    # {100, 200}.
    assert out.dtype == arr.dtype
    out_vals = set(out.flatten().tolist())
    assert out_vals.issubset({100, 200}), (
        f"method={method} dtype={dtype} sentinel={sentinel} "
        f"produced poisoned values: {out_vals - {100, 200}}"
    )


@pytest.mark.parametrize('dtype,sentinel', [
    (np.uint16, 65535),
    (np.int16, -9999),
])
def test_block_reduce_int_all_sentinel_block(dtype, sentinel):
    """A 2x2 block that's entirely sentinel reduces to the sentinel.

    Without the post-reduction NaN-to-sentinel rewrite in the integer
    branch, the all-NaN block from nanmean would cast to undefined
    integer behaviour (zero or INT_MIN depending on platform).
    """
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = np.array([
        [100,      100,      sentinel, sentinel],
        [100,      100,      sentinel, sentinel],
        [200,      200,      200,      200],
        [200,      200,      200,      200],
    ], dtype=dtype)

    out = _block_reduce_2d(arr, 'mean', nodata=sentinel)
    assert out.dtype == arr.dtype
    # Top-right block is all-sentinel; output must be the sentinel
    assert out[0, 1] == sentinel
    # Other blocks contain only valid values
    assert out[0, 0] == 100
    assert out[1, 0] == 200
    assert out[1, 1] == 200


def test_block_reduce_int_no_nodata_unchanged():
    """Without ``nodata``, the integer reduction code path stays unchanged.

    Regression check: the fix must not alter the no-sentinel case.
    """
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=np.int16)

    out = _block_reduce_2d(arr, 'mean')
    # Block (0,0) = mean(1,2,5,6) = 3.5 -> round -> 4
    # Block (0,1) = mean(3,4,7,8) = 5.5 -> round -> 6
    # Block (1,0) = mean(9,10,13,14) = 11.5 -> round -> 12
    # Block (1,1) = mean(11,12,15,16) = 13.5 -> round -> 14
    expected = np.array([[4, 6], [12, 14]], dtype=np.int16)
    np.testing.assert_array_equal(out, expected)


def test_block_reduce_int_out_of_range_sentinel_noop():
    """A sentinel outside the dtype's range is a no-op (no mask applied).

    Mirrors the ``_int_nodata_in_range`` gating in ``_reader.py``: a
    uint16 file with GDAL_NODATA="-9999" cannot match any decoded pixel,
    so the reduction proceeds without the mask. This keeps the fix from
    raising OverflowError on the dtype cast.
    """
    from xrspatial.geotiff._writer import _block_reduce_2d

    # uint16 with nodata=-9999: out of range, no-op
    arr = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ], dtype=np.uint16)
    out = _block_reduce_2d(arr, 'mean', nodata=-9999)
    # Should produce the same result as without the kwarg
    expected = _block_reduce_2d(arr, 'mean')
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# End-to-end: to_geotiff + open_geotiff round trip
# ---------------------------------------------------------------------------

@pytest.fixture
def _int_cog_inputs(tmp_path):
    """uint16 raster, full of 100 with a 65x65 sentinel patch."""
    H, W = 256, 256
    data = np.full((H, W), 100, dtype=np.uint16)
    data[64:129, 64:129] = 65535
    da = xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={'y': np.arange(H, dtype=np.float64),
                'x': np.arange(W, dtype=np.float64)},
        attrs={'crs': 4326},
    )
    return da, tmp_path


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
def test_cpu_int_cog_overview_not_poisoned(_int_cog_inputs, method):
    """End-to-end: integer COG overview pyramid contains only valid values.

    Before the fix, the level-1 read contained values like 16459 and
    32818 -- nan-aware-mean of (sentinel, 100, 100, 100) and (sentinel,
    sentinel, 100, 100) cast back to uint16. The reader can't mask them
    because they don't equal 65535.
    """
    from xrspatial.geotiff import open_geotiff, to_geotiff

    da, tmp_path = _int_cog_inputs
    path = str(tmp_path / f'int_overview_{method}_2026_05_12.tif')
    to_geotiff(da, path, nodata=65535, cog=True,
               overview_levels=[2], overview_resampling=method)

    ov = open_geotiff(path, overview_level=1)
    arr = np.asarray(ov.data)
    unique = set(int(v) for v in np.unique(arr) if not np.isnan(v))
    poisoned = unique - {100, 65535}
    assert not poisoned, (
        f"method={method} produced poisoned overview values: {poisoned}"
    )


def test_cpu_int_cog_overview_3band_not_poisoned(tmp_path):
    """3-band integer COG: same fix applies via the 3D _make_overview branch."""
    from xrspatial.geotiff import open_geotiff, to_geotiff

    H, W = 256, 256
    data = np.full((H, W, 3), 100, dtype=np.uint16)
    data[64:129, 64:129, :] = 65535
    da = xr.DataArray(
        data,
        dims=('y', 'x', 'band'),
        coords={'y': np.arange(H, dtype=np.float64),
                'x': np.arange(W, dtype=np.float64),
                'band': [0, 1, 2]},
        attrs={'crs': 4326},
    )

    path = str(tmp_path / 'int_overview_3band_2026_05_12.tif')
    to_geotiff(da, path, nodata=65535, cog=True,
               overview_levels=[2], overview_resampling='mean')

    ov = open_geotiff(path, overview_level=1)
    arr = np.asarray(ov.data)
    unique = set(int(v) for v in np.unique(arr) if not np.isnan(v))
    poisoned = unique - {100, 65535}
    assert not poisoned, (
        f"3-band integer overview produced poisoned values: {poisoned}"
    )


def test_cpu_int_cog_no_nodata_unchanged(tmp_path):
    """No nodata kwarg: integer overview path stays as it was."""
    from xrspatial.geotiff import open_geotiff, to_geotiff

    H, W = 256, 256
    data = np.full((H, W), 100, dtype=np.uint16)
    data[100:200, 100:200] = 50
    da = xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={'y': np.arange(H, dtype=np.float64),
                'x': np.arange(W, dtype=np.float64)},
        attrs={'crs': 4326},
    )

    path = str(tmp_path / 'int_overview_no_nodata_2026_05_12.tif')
    to_geotiff(da, path, cog=True,
               overview_levels=[2], overview_resampling='mean')

    ov = open_geotiff(path, overview_level=1)
    arr = np.asarray(ov.data)
    # No sentinel, so every overview pixel is a real average of 50 / 100.
    # Block-boundary pixels are weighted means: (50,50,50,100)/4 = 62.5 -> 63
    unique = set(int(v) for v in np.unique(arr))
    # Must contain at least 50 and 100; boundary-mixing averages allowed.
    assert 50 in unique
    assert 100 in unique


# ---------------------------------------------------------------------------
# GPU mirror
# ---------------------------------------------------------------------------

@_gpu_only
@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
@pytest.mark.parametrize('dtype,sentinel', [
    (np.uint16, 65535),
    (np.int16, -9999),
])
def test_gpu_block_reduce_int_sentinel_masked(method, dtype, sentinel):
    """GPU mirror of the CPU integer sentinel-mask fix."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr = _int_block_partial_sentinel(sentinel, dtype)
    cpu_arr = cupy.asarray(arr)
    out_gpu = _block_reduce_2d_gpu(cpu_arr, method, nodata=sentinel)
    out = out_gpu.get()

    assert out.dtype == arr.dtype
    out_vals = set(out.flatten().tolist())
    assert out_vals.issubset({100, 200}), (
        f"GPU method={method} dtype={dtype} produced poisoned values: "
        f"{out_vals - {100, 200}}"
    )


@_gpu_only
@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
def test_gpu_cpu_int_overview_byte_match(method):
    """CPU and GPU integer overview reductions agree byte-for-byte.

    Same parity contract as #1623 (cubic). Without the GPU fix, the GPU
    pyramid would carry poisoned values while the CPU pyramid carried
    sentinels -- two backends disagreeing on identical input.
    """
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu
    from xrspatial.geotiff._writer import _block_reduce_2d

    arr = _int_block_partial_sentinel(-9999, np.int16)
    cpu_out = _block_reduce_2d(arr, method, nodata=-9999)
    gpu_out = _block_reduce_2d_gpu(
        cupy.asarray(arr), method, nodata=-9999).get()

    np.testing.assert_array_equal(cpu_out, gpu_out)
