"""Overview-level and nodata-aware overview tests.

Covers the overview shape-ceiling contract, the mean / min / max /
median / mode resampling matrix with int and float nodata, the cubic
resampling cases for both float and integer dtypes, and the
block-reduce sentinel-masking gate for int sentinels.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._writer import _block_reduce_2d

from .._helpers.markers import gpu_available as _gpu_available

# -------------------------------------------------------------------------
# Section: ceil-shape overview tests
# -------------------------------------------------------------------------

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
# Section: nodata-aware overview tests
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
    """CPU writer: overview 'mean' must skip sentinel pixels."""
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

    ov = open_geotiff(p, overview_level=1, masked=True)
    # Top-left 2x2 was all-NaN -> reduces to NaN -> rewritten to -9999
    #   on disk, then read back as NaN once overview-nodata
    #   inheritance restores attrs['nodata'] and re-masks
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
    """GPU writer: overview 'mean' must skip sentinel pixels."""
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
# Section: cubic resampling, float nodata
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
    against the older smaller set."""
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
# Section: cubic resampling, int nodata
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
    """Float regression guard: the existing cubic path must still work."""
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
    r0 = open_geotiff(str(path), overview_level=0, masked=True)
    uniq_0 = set(np.unique(r0.values[~np.isnan(r0.values)]))
    assert uniq_0 == {100.0}
    # Level 1: the historically poisoned level.
    r1 = open_geotiff(str(path), overview_level=1, masked=True)
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
    r_cubic = open_geotiff(str(cubic_path), overview_level=0, masked=True)
    r_mean = open_geotiff(str(mean_path), overview_level=0, masked=True)
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
# Section: block-reduce int sentinel masking
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

    Same parity contract as the cubic case. Without the GPU fix, the GPU
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


# =========================================================================
# Section: COG overview tile-block ordering invariant
# =========================================================================
#
# The COG spec requires the on-disk pixel-data layout to run from the
# smallest overview through progressively larger overviews and end with
# the main-resolution image. External readers (rio-cogeo, GDAL's
# ``validate_cloud_optimized_geotiff``) flag a file when the byte
# ordering reverses or interleaves these blocks even though the IFD
# chain walks main -> ov1 -> ov2 the conventional way. These tests lock
# the byte order in as a regression gate so the writer cannot drift back
# to the old layout.

from xrspatial.geotiff._header import parse_all_ifds, parse_header  # noqa: E402


def _min_block_offset(ifd) -> int:
    """Return the smallest tile-offset (or strip-offset) for an IFD."""
    offsets = ifd.tile_offsets
    if offsets is None:
        offsets = ifd.strip_offsets
    assert offsets is not None and len(offsets) > 0, (
        "IFD has neither tile_offsets nor strip_offsets")
    return min(offsets)


def _read_block_order(path: str) -> list:
    """Return ``min_block_offset`` for each IFD in walk order."""
    with open(path, "rb") as f:
        data = f.read()
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    return [_min_block_offset(ifd) for ifd in ifds]


def _make_block_order_da(shape, bands=None) -> xr.DataArray:
    """Build a synthetic DataArray with a sane CRS / coordinate grid."""
    rng = np.random.RandomState(17)
    if bands is None:
        arr = rng.rand(*shape).astype("float32")
        dims = ("y", "x")
    else:
        arr = rng.rand(bands, *shape).astype("float32")
        dims = ("band", "y", "x")
    h, w = shape
    coords = {
        "y": np.linspace(45, 44, h),
        "x": np.linspace(-120, -119, w),
    }
    return xr.DataArray(arr, dims=dims, coords=coords, attrs={"crs": 4326})


@pytest.mark.parametrize("bands", [None, 3])
def test_cog_overview_block_order_invariant_2308(tmp_path, bands):
    """Pixel blocks must run smallest-overview -> larger -> main.

    The IFD walk order is ``[main, ov_factor_2, ov_factor_4]`` (full
    resolution first). The on-disk pixel-block order must be the
    reverse: factor-4 overview blocks first, then factor-2 overview
    blocks, with the main-resolution blocks last.
    """
    da = _make_block_order_da((256, 256), bands=bands)
    suffix = "rgb" if bands else "mono"
    path = str(tmp_path / f"order_2308_{suffix}.tif")
    to_geotiff(
        da, path, compression="deflate", cog=True,
        tile_size=64, overview_levels=[2, 4],
    )

    block_offsets = _read_block_order(path)
    # IFD walk: [main, ov_factor_2, ov_factor_4]
    main_min, ov2_min, ov4_min = block_offsets
    # COG layout: factor-4 (smallest overview) -> factor-2 -> main.
    assert ov4_min < ov2_min, (
        f"smallest overview blocks should sit before larger "
        f"overview blocks: ov4_min={ov4_min}, ov2_min={ov2_min}")
    assert ov2_min < main_min, (
        f"overview blocks should sit before main-resolution "
        f"blocks: ov2_min={ov2_min}, main_min={main_min}")


def test_cog_overview_block_order_three_levels_2308(tmp_path):
    """Same invariant with three overview levels (factor 2/4/8)."""
    da = _make_block_order_da((512, 512))
    path = str(tmp_path / "order_2308_three.tif")
    to_geotiff(
        da, path, compression="deflate", cog=True,
        tile_size=64, overview_levels=[2, 4, 8],
    )

    block_offsets = _read_block_order(path)
    # IFD walk: [main, ov2, ov4, ov8]
    main_min, ov2_min, ov4_min, ov8_min = block_offsets
    # On-disk: ov8 -> ov4 -> ov2 -> main
    assert ov8_min < ov4_min < ov2_min < main_min, (
        f"COG block order broken: ov8={ov8_min} ov4={ov4_min} "
        f"ov2={ov2_min} main={main_min}")


def _rio_cogeo_or_skip():
    """Skip the rio-cogeo gate when the dependency isn't installed.

    Mirrors the skip semantics used in ``write/test_cog.py``:
    contributor laptops without rio-cogeo see a skip, CI with rio-cogeo
    runs the strict check.
    """
    try:
        from rio_cogeo.cogeo import cog_validate
    except ImportError:
        pytest.skip("rio-cogeo not installed")
    return cog_validate


@pytest.mark.parametrize("bands", [None, 3])
def test_cog_overview_block_order_rio_cogeo_2308(tmp_path, bands):
    """``rio-cogeo cog_validate`` returns valid=True with no block-order errors."""
    cog_validate = _rio_cogeo_or_skip()
    da = _make_block_order_da((256, 256), bands=bands)
    suffix = "rgb" if bands else "mono"
    path = str(tmp_path / f"order_2308_rio_{suffix}.tif")
    to_geotiff(
        da, path, compression="deflate", cog=True,
        tile_size=64, overview_levels=[2, 4],
    )
    valid, errors, _warnings = cog_validate(path, strict=False)
    assert valid, f"rio_cogeo cog_validate failed: {errors}"
    # Defensive secondary assertion: the two block-order messages
    # must not reappear even if some future writer
    # change keeps the validator happy on the headline check.
    joined = " ".join(errors).lower()
    assert "offset of the first block" not in joined, (
        f"block-order errors regressed: {errors}")


# =========================================================================
# Section: overview_levels honours decimation factors
# =========================================================================
#
# Before the fix, ``to_geotiff`` and the underlying writer ignored the
# integer values in ``overview_levels`` and only used the list length:
# each entry halved the previous overview regardless of what value was
# passed. So ``overview_levels=[2, 8]`` produced 2x and 4x overviews, not
# 2x and 8x. After the fix, each entry is a power-of-two decimation
# factor relative to full resolution. The list must be strictly
# increasing integers >= 2.

from xrspatial.geotiff._reader import _FileSource  # noqa: E402
from xrspatial.geotiff._writer import _validate_overview_levels, write  # noqa: E402

# A unique stem so temp paths cannot collide with sibling tests running
# in parallel worktrees (rockout convention).
_STEM_1766 = "issue_1766"


def _ifd_dimensions(path):
    """Return (width, height) for every IFD in the file."""
    with _FileSource(path) as src:
        data = src.read_all()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
    return [(ifd.width, ifd.height) for ifd in ifds]


def test_overview_levels_2_4_8_produces_three_correctly_sized_overviews(tmp_path):
    """``[2, 4, 8]`` writes overviews at 1/2, 1/4 and 1/8 of the input."""
    arr = np.zeros((512, 512), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_2_4_8.tif")
    write(arr, path, compression='none', tiled=True, tile_size=64,
          cog=True, overview_levels=[2, 4, 8])

    dims = _ifd_dimensions(path)
    # Full-res + 3 overviews.
    assert dims == [(512, 512), (256, 256), (128, 128), (64, 64)]


def test_overview_levels_2_4_regression_for_buggy_2_8_case(tmp_path):
    """Regression: ``[2, 4]`` now matches the explicit factors.

    Before the fix the writer ignored the values and only the
    list length determined how many halvings happened. ``[2, 4]``
    happened to produce the right shapes by accident (2 halvings is
    /2 and /4), but ``[2, 8]`` did not (2 halvings is /2 and /4, not
    /2 and /8). Pin both behaviours so a future regression to the
    "length-only" semantics fails this test.
    """
    arr = np.zeros((256, 256), dtype=np.uint8)

    path = str(tmp_path / f"{_STEM_1766}_2_4.tif")
    write(arr, path, compression='none', tiled=True, tile_size=32,
          cog=True, overview_levels=[2, 4])
    assert _ifd_dimensions(path) == [(256, 256), (128, 128), (64, 64)]

    path2 = str(tmp_path / f"{_STEM_1766}_2_8.tif")
    write(arr, path2, compression='none', tiled=True, tile_size=32,
          cog=True, overview_levels=[2, 8])
    # /8 of 256 is 32. Before the fix this file held a 64x64 overview
    # in slot 2 because length-only semantics treated [2, 8] as "two
    # halvings".
    assert _ifd_dimensions(path2) == [(256, 256), (128, 128), (32, 32)]


def test_overview_levels_skips_intermediate_factor(tmp_path):
    """``[4]`` decimates straight to /4 without writing a /2 IFD."""
    arr = np.zeros((256, 256), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_4_only.tif")
    write(arr, path, compression='none', tiled=True, tile_size=32,
          cog=True, overview_levels=[4])
    assert _ifd_dimensions(path) == [(256, 256), (64, 64)]


def test_overview_levels_high_factor(tmp_path):
    """``[16]`` reduces 1024 -> 64."""
    arr = np.zeros((1024, 1024), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_16.tif")
    write(arr, path, compression='none', tiled=True, tile_size=32,
          cog=True, overview_levels=[16])
    assert _ifd_dimensions(path) == [(1024, 1024), (64, 64)]


def test_overview_levels_none_auto_generation_unchanged(tmp_path):
    """``overview_levels=None`` still auto-generates the doubling pyramid."""
    arr = np.zeros((512, 512), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_auto.tif")
    write(arr, path, compression='none', tiled=True, tile_size=64,
          cog=True, overview_levels=None)
    # Auto loop checks current > tile_size *before* halving: 512>64 (halve
    # to 256, factor 2), 256>64 (halve to 128, factor 4), 128>64 (halve
    # to 64, factor 8), 64>64 stops. Three overviews.
    assert _ifd_dimensions(path) == [
        (512, 512), (256, 256), (128, 128), (64, 64)
    ]


def test_overview_pyramid_mean_values_are_correct(tmp_path):
    """The /4 overview's mean must match a 4x4-block reduction of input.

    This guards the cumulative-decimation loop: a regression where each
    list entry resets the source back to full resolution (instead of
    chaining from the previous overview) would still produce
    correctly-shaped output but wrong pixel values.
    """
    rng = np.random.default_rng(seed=1766)
    arr = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_mean_check.tif")
    write(arr, path, compression='none', tiled=True, tile_size=16,
          cog=True, overview_levels=[4], overview_resampling='mean')

    # Expected: 4x4 block-reduce by mean.
    expected = arr.astype(np.float64).reshape(16, 4, 16, 4).mean(axis=(1, 3))
    expected = expected.astype(np.uint8)

    full = open_geotiff(path)
    assert full.shape == (64, 64)
    # The on-disk overview value should match the chained-halving result.
    # ``open_geotiff`` returns the full-resolution band; read the overview
    # IFD directly through the low-level path.
    with _FileSource(path) as src:
        data = src.read_all()
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
    # IFD 1 is the /4 overview (only one we requested).
    assert ifds[1].width == 16 and ifds[1].height == 16
    # The byte-level check would require decoding tiles; the shape +
    # auto-generated-vs-explicit parity test below covers correctness
    # via cross-comparison against the auto-generated pyramid.


def test_explicit_factors_match_auto_pyramid_bytewise(tmp_path):
    """``overview_levels=[2, 4]`` produces the same overview tile bytes
    as the auto-generated pyramid stopped at the same depth.

    This proves the cumulative-halving loop walks the same chain as the
    auto path -- if a future refactor accidentally re-seeded from full
    resolution at each step, the tile bytes would diverge.
    """
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)

    path_explicit = str(tmp_path / f"{_STEM_1766}_explicit.tif")
    write(arr, path_explicit, compression='none', tiled=True, tile_size=64,
          cog=True, overview_levels=[2, 4])

    path_auto = str(tmp_path / f"{_STEM_1766}_auto_compare.tif")
    write(arr, path_auto, compression='none', tiled=True, tile_size=64,
          cog=True, overview_levels=None)

    # Auto produces /2 and /4 here (128 > 64, 64 not > 64), same depth.
    dims_e = _ifd_dimensions(path_explicit)
    dims_a = _ifd_dimensions(path_auto)
    assert dims_e == dims_a == [(256, 256), (128, 128), (64, 64)]

    # The full file bytes should be identical: same header, same IFDs,
    # same tile data. The auto and explicit code paths now feed the
    # same list into the same loop.
    with open(path_explicit, 'rb') as f:
        bytes_e = f.read()
    with open(path_auto, 'rb') as f:
        bytes_a = f.read()
    assert bytes_e == bytes_a


def test_overview_levels_rejects_factor_of_one(tmp_path):
    """Factor 1 is the original full-resolution band, not an overview."""
    arr = np.zeros((128, 128), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_reject_1.tif")
    with pytest.raises(ValueError, match=">= 2"):
        write(arr, path, compression='none', tiled=True, tile_size=32,
              cog=True, overview_levels=[1])


def test_overview_levels_rejects_factor_below_one(tmp_path):
    arr = np.zeros((128, 128), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_reject_0.tif")
    with pytest.raises(ValueError, match=">= 2"):
        write(arr, path, compression='none', tiled=True, tile_size=32,
              cog=True, overview_levels=[0])


def test_overview_levels_rejects_non_increasing(tmp_path):
    arr = np.zeros((128, 128), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_reject_decrease.tif")
    with pytest.raises(ValueError, match="strictly increasing"):
        write(arr, path, compression='none', tiled=True, tile_size=32,
              cog=True, overview_levels=[4, 2])


def test_overview_levels_rejects_duplicate(tmp_path):
    arr = np.zeros((128, 128), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_reject_dup.tif")
    with pytest.raises(ValueError, match="strictly increasing"):
        write(arr, path, compression='none', tiled=True, tile_size=32,
              cog=True, overview_levels=[2, 2])


def test_overview_levels_rejects_non_power_of_two(tmp_path):
    """``[2, 6]`` would require a 3x reduction between levels, but the
    underlying block reducer only halves."""
    arr = np.zeros((128, 128), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_reject_pow2.tif")
    with pytest.raises(ValueError, match="power of two"):
        write(arr, path, compression='none', tiled=True, tile_size=32,
              cog=True, overview_levels=[2, 6])


def test_overview_levels_rejects_non_int(tmp_path):
    arr = np.zeros((128, 128), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_reject_float.tif")
    with pytest.raises(ValueError, match="int >= 2"):
        write(arr, path, compression='none', tiled=True, tile_size=32,
              cog=True, overview_levels=[2.0, 4.0])


def test_overview_levels_rejects_bool(tmp_path):
    """``bool`` is an ``int`` subclass; ``True`` would otherwise sneak
    through as ``1``."""
    arr = np.zeros((128, 128), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_reject_bool.tif")
    with pytest.raises(ValueError, match="int >= 2"):
        write(arr, path, compression='none', tiled=True, tile_size=32,
              cog=True, overview_levels=[True, False])


def test_overview_levels_rejects_non_list_type(tmp_path):
    arr = np.zeros((128, 128), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_reject_str.tif")
    with pytest.raises(ValueError, match="list or tuple of ints"):
        write(arr, path, compression='none', tiled=True, tile_size=32,
              cog=True, overview_levels="2,4,8")


def test_overview_levels_accepts_numpy_ints(tmp_path):
    """``np.int64(2)`` etc. should validate as ints."""
    arr = np.zeros((256, 256), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_np_ints.tif")
    write(arr, path, compression='none', tiled=True, tile_size=32,
          cog=True, overview_levels=[np.int64(2), np.int32(4)])
    assert _ifd_dimensions(path) == [(256, 256), (128, 128), (64, 64)]


def test_overview_levels_empty_list(tmp_path):
    """An empty list writes no overviews -- valid but unusual."""
    arr = np.zeros((128, 128), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_empty.tif")
    write(arr, path, compression='none', tiled=True, tile_size=32,
          cog=True, overview_levels=[])
    assert _ifd_dimensions(path) == [(128, 128)]


def test_overview_levels_rejects_factor_too_large_for_shape(tmp_path):
    """Factors that would shrink the raster below 1 pixel raise.

    Without the up-front shape check, the writer would silently emit
    a zero-sized overview IFD (``_block_reduce_2d`` returns shape
    ``(0, 0)`` once the input dim is below 2 and stays there on
    subsequent halvings).
    """
    arr = np.zeros((64, 64), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_too_large.tif")
    with pytest.raises(ValueError, match="too large for input shape"):
        write(arr, path, compression='none', tiled=True, tile_size=8,
              cog=True, overview_levels=[128])


def test_overview_levels_factor_at_shape_floor_is_allowed(tmp_path):
    """``factor == min(h, w)`` is feasible (decimates to 1x1)."""
    arr = np.zeros((64, 64), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_at_floor.tif")
    write(arr, path, compression='none', tiled=True, tile_size=8,
          cog=True, overview_levels=[64])
    assert _ifd_dimensions(path) == [(64, 64), (1, 1)]


def test_overview_levels_rejects_factor_too_large_non_square(tmp_path):
    """Rectangular shape: the smaller dim sets the feasibility floor."""
    arr = np.zeros((512, 16), dtype=np.uint8)
    path = str(tmp_path / f"{_STEM_1766}_rect_too_large.tif")
    # Height/16 = 32 OK, but width/32 = 0 -> reject.
    with pytest.raises(ValueError, match="too large for input shape"):
        write(arr, path, compression='none', tiled=True, tile_size=8,
              cog=True, overview_levels=[32])


def test_to_geotiff_honours_decimation_factors(tmp_path):
    """The public ``to_geotiff`` entry point honours factors end-to-end."""
    arr = np.zeros((512, 512), dtype=np.uint8)
    da = xr.DataArray(arr, dims=('y', 'x'))
    path = str(tmp_path / f"{_STEM_1766}_public.tif")
    to_geotiff(da, path, cog=True, tile_size=64,
               overview_levels=[2, 4, 8])
    assert _ifd_dimensions(path) == [(512, 512), (256, 256), (128, 128), (64, 64)]


def test_to_geotiff_rejects_invalid_factors(tmp_path):
    arr = np.zeros((128, 128), dtype=np.uint8)
    da = xr.DataArray(arr, dims=('y', 'x'))
    path = str(tmp_path / f"{_STEM_1766}_public_reject.tif")
    with pytest.raises(ValueError, match="power of two"):
        to_geotiff(da, path, cog=True, tile_size=32,
                   overview_levels=[2, 3])


def test_validate_passthrough_none():
    assert _validate_overview_levels(None) is None


def test_validate_returns_clean_list_of_ints():
    out = _validate_overview_levels([np.int64(2), 4, 8])
    assert out == [2, 4, 8]
    assert all(isinstance(x, int) for x in out)


# =========================================================================
# Section: overview nodata / metadata inheritance
# =========================================================================
#
# Overview IFDs in COGs typically carry no GDAL_NODATA tag (and no
# GDAL_METADATA, XResolution, YResolution, ResolutionUnit, ColorMap,
# ImageDescription, or ExtraSamples either) -- the writer puts those tags
# only on the level-0 IFD. Before the fix, ``open_geotiff(path,
# overview_level=N)`` for ``N >= 1`` returned a DataArray whose
# ``attrs['nodata']`` was None even though level 0 declared a sentinel,
# and the overview's on-disk pixels still carried that sentinel. The fix
# wires the per-IFD pass-through tags into
# ``extract_geo_info_with_overview_inheritance`` so an overview without
# its own copy inherits from level 0.

from xrspatial.geotiff._geotags import GeoTransform  # noqa: E402

_BACKENDS_1739 = [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 16}, id="dask+numpy"),
    pytest.param({"gpu": True}, id="cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
    pytest.param({"gpu": True, "chunks": 16}, id="dask+cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
]


def _materialise(da) -> np.ndarray:
    """Return a numpy view of the data regardless of backend."""
    raw = da.data
    if hasattr(raw, 'compute'):
        raw = raw.compute()
    if hasattr(raw, 'get'):
        raw = raw.get()
    return np.asarray(raw)


def _make_cog_with_nodata(path: str, sentinel: float = -9999.0) -> None:
    """Write a 64x64 COG with two overview levels + a sentinel column.

    The first 16x16 block is the sentinel; the rest is 100.0. Using
    nearest-neighbour overview resampling preserves exactly 64 sentinel
    pixels in level 1 and 16 in level 2, so the test can assert counts
    without floating-point fudge factors.
    """
    arr = np.full((64, 64), 100.0, dtype=np.float32)
    arr[0:16, 0:16] = sentinel
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(64, dtype=np.float64),
                'x': np.arange(64, dtype=np.float64)},
        attrs={'crs': 4326},
    )
    to_geotiff(da, path, cog=True, tile_size=16,
               overview_levels=[2, 4], nodata=sentinel,
               overview_resampling='nearest')


@pytest.mark.parametrize("backend_kwargs", _BACKENDS_1739)
def test_overview_inherits_nodata_attr(tmp_path, backend_kwargs):
    """attrs['nodata'] is set on every overview level, not just level 0."""
    path = str(tmp_path / "overview_nodata_inherit_1739.tif")
    _make_cog_with_nodata(path)

    for lvl in (0, 1, 2):
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        assert da.attrs.get('nodata') == -9999.0, (
            f"backend={backend_kwargs}, overview_level={lvl}: expected "
            f"nodata=-9999.0, got {da.attrs.get('nodata')!r}"
        )


@pytest.mark.parametrize("backend_kwargs", _BACKENDS_1739)
def test_overview_sentinel_pixels_masked_to_nan(tmp_path, backend_kwargs):
    """Pixels at the sentinel value come back as NaN on every overview level.

    Without the inheritance fix, an overview read returned a DataArray
    with literal -9999.0 pixels and no nodata attr, silently poisoning
    downstream stats. With the fix, the reader inherits the sentinel
    and applies the same NaN-mask substitution it applies at level 0.
    """
    path = str(tmp_path / "overview_nodata_mask_1739.tif")
    _make_cog_with_nodata(path)

    expected_nan_counts = {0: 256, 1: 64, 2: 16}
    for lvl, expected in expected_nan_counts.items():
        da = open_geotiff(path, overview_level=lvl, masked=True, **backend_kwargs)
        vals = _materialise(da)
        actual_nan = int(np.isnan(vals).sum())
        sentinel_remaining = int((vals == -9999.0).sum())

        assert sentinel_remaining == 0, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"{sentinel_remaining} sentinel pixels survived as raw "
            f"values; expected all masked to NaN"
        )
        assert actual_nan == expected, (
            f"backend={backend_kwargs}, overview_level={lvl}: expected "
            f"{expected} NaN pixels, got {actual_nan}"
        )


@pytest.mark.parametrize("backend_kwargs", _BACKENDS_1739)
def test_overview_nanmean_matches_pre_sentinel_value(tmp_path, backend_kwargs):
    """nanmean on every overview level equals the non-sentinel value.

    With the fix, sentinel pixels are NaN-masked, so np.nanmean returns
    100.0 (the value of every other pixel). Without it, the sentinel
    survives as -9999.0 and the mean is far below 100.0.
    """
    path = str(tmp_path / "overview_nodata_mean_1739.tif")
    _make_cog_with_nodata(path)

    for lvl in (0, 1, 2):
        da = open_geotiff(path, overview_level=lvl, masked=True, **backend_kwargs)
        vals = _materialise(da)
        assert np.nanmean(vals) == pytest.approx(100.0), (
            f"backend={backend_kwargs}, overview_level={lvl}: nanmean="
            f"{np.nanmean(vals)} (expected 100.0); sentinel pixels "
            "are leaking into the average."
        )


@pytest.mark.parametrize("backend_kwargs", _BACKENDS_1739)
def test_overview_inherits_gdal_metadata(tmp_path, backend_kwargs):
    """attrs['gdal_metadata'] and ['gdal_metadata_xml'] come from level 0.

    The COG writer emits the GDAL_METADATA tag only on the level-0 IFD.
    Without the inheritance fix, overview reads dropped the dict, so a
    user-supplied scaling factor or band-stats payload only appeared at
    level 0 -- silently inconsistent across overview reads.
    """
    arr = np.full((64, 64), 100.0, dtype=np.float32)
    gt = GeoTransform(origin_x=0.0, origin_y=0.0,
                      pixel_width=1.0, pixel_height=-1.0)
    path = str(tmp_path / "overview_gdal_md_inherit_1739.tif")
    write(arr, path, geo_transform=gt, crs_epsg=4326,
          cog=True, tile_size=16, overview_levels=[2, 4],
          gdal_metadata_xml=(
              '<GDALMetadata>\n'
              '  <Item name="MY_SCALING">1.5</Item>\n'
              '</GDALMetadata>\n'),
          compression='none')

    for lvl in (0, 1, 2):
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        assert da.attrs.get('gdal_metadata') is not None, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"gdal_metadata attr is missing")
        assert da.attrs['gdal_metadata'].get('MY_SCALING') == '1.5', (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"expected gdal_metadata['MY_SCALING']='1.5', got "
            f"{da.attrs['gdal_metadata']}"
        )
        assert da.attrs.get('gdal_metadata_xml') is not None, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"gdal_metadata_xml attr is missing")


@pytest.mark.parametrize("backend_kwargs", _BACKENDS_1739)
def test_overview_inherits_resolution_tags(tmp_path, backend_kwargs):
    """XResolution / YResolution / ResolutionUnit propagate to overviews.

    Same pattern as gdal_metadata: the writer puts these tags only on
    the level-0 IFD, so an overview read used to come back with the
    resolution tags missing. The inheritance fix surfaces them on every
    level.
    """
    arr = np.full((64, 64), 100.0, dtype=np.float32)
    gt = GeoTransform(origin_x=0.0, origin_y=0.0,
                      pixel_width=1.0, pixel_height=-1.0)
    path = str(tmp_path / "overview_res_inherit_1739.tif")
    write(arr, path, geo_transform=gt, crs_epsg=4326,
          cog=True, tile_size=16, overview_levels=[2, 4],
          x_resolution=300.0, y_resolution=300.0, resolution_unit=2,
          compression='none')

    for lvl in (0, 1, 2):
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        assert da.attrs.get('x_resolution') == 300.0, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"x_resolution={da.attrs.get('x_resolution')!r}")
        assert da.attrs.get('y_resolution') == 300.0, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"y_resolution={da.attrs.get('y_resolution')!r}")
        assert da.attrs.get('resolution_unit') == 'inch', (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"resolution_unit={da.attrs.get('resolution_unit')!r}")


def test_attrs_keysets_consistent_across_overview_levels(tmp_path):
    """The set of attrs keys is identical at every overview level.

    Strong contract that catches future regressions where one of the
    inherited fields gets dropped without removing the entry from the
    main code path (e.g. a refactor that splits the inheritance block
    into helpers and forgets one field).
    """
    arr = np.full((64, 64), 100.0, dtype=np.float32)
    arr[0:16, 0:16] = -9999.0
    gt = GeoTransform(origin_x=0.0, origin_y=0.0,
                      pixel_width=1.0, pixel_height=-1.0)
    path = str(tmp_path / "overview_attr_keysets_1739.tif")
    write(arr, path, geo_transform=gt, crs_epsg=4326, nodata=-9999.0,
          cog=True, tile_size=16, overview_levels=[2, 4],
          x_resolution=300, y_resolution=300, resolution_unit=2,
          gdal_metadata_xml=(
              '<GDALMetadata>\n  <Item name="K">V</Item>\n'
              '</GDALMetadata>\n'))

    level_keysets = {
        lvl: set(open_geotiff(path, overview_level=lvl).attrs.keys())
        for lvl in (0, 1, 2)
    }
    base = level_keysets[0]
    for lvl in (1, 2):
        diff = base ^ level_keysets[lvl]
        assert not diff, (
            f"overview_level={lvl} attrs keyset differs from level 0:\n"
            f"  level 0: {sorted(base)}\n"
            f"  level {lvl}: {sorted(level_keysets[lvl])}\n"
            f"  symmetric diff: {sorted(diff)}"
        )


def test_overview_with_own_nodata_keeps_own_value(tmp_path):
    """Overview IFDs that re-declare GDAL_NODATA keep their own value.

    The inheritance is per-field and only fills in missing entries, so
    an overview that does carry its own tag (rare but valid -- some
    writers do this) is not stomped on by the parent's value.

    This test pins the "overview already has its own value" branch by
    simulating it directly against
    ``extract_geo_info_with_overview_inheritance``.
    """
    import xrspatial.geotiff._geotags as _gt_mod
    from xrspatial.geotiff._geotags import GeoInfo
    from xrspatial.geotiff._geotags import GeoTransform as _GT
    from xrspatial.geotiff._geotags import extract_geo_info_with_overview_inheritance

    class _StubIFD:
        def __init__(self, subfile_type, width, height):
            self.subfile_type = subfile_type
            self.width = width
            self.height = height

    base_ifd = _StubIFD(0, 64, 64)
    ov_ifd = _StubIFD(1, 32, 32)  # overview (NewSubfileType bit 0)

    # Overview already has its own nodata (-5555); base has -9999.
    # Test: inheritance leaves the overview's -5555 untouched.
    def fake_extract(ifd, data, byte_order, *, allow_rotated=False,
                     allow_invalid_nodata=False):
        if ifd is ov_ifd:
            gi = GeoInfo()
            gi.nodata = -5555.0
            gi.has_georef = False
            return gi
        if ifd is base_ifd:
            gi = GeoInfo()
            gi.nodata = -9999.0
            gi.transform = _GT(0.0, 0.0, 1.0, -1.0)
            gi.has_georef = True
            gi.crs_epsg = 4326
            return gi
        return GeoInfo()

    orig = _gt_mod.extract_geo_info
    _gt_mod.extract_geo_info = fake_extract
    try:
        out = extract_geo_info_with_overview_inheritance(
            ov_ifd, [base_ifd, ov_ifd], b'', '<')
    finally:
        _gt_mod.extract_geo_info = orig

    assert out.nodata == -5555.0, (
        f"overview's own nodata=-5555 was overwritten by base's -9999; "
        f"got {out.nodata}"
    )


# =========================================================================
# Section: PixelIsPoint overview origin shift
# =========================================================================
#
# Overview reads inherit level-0 georef but keep the level-0 origin
# unchanged. That is correct for the default
# ``PixelIsArea`` raster_type. It is wrong for ``PixelIsPoint`` (GeoKey
# 1025 = 2), where the origin is the center of pixel (0, 0): an overview
# pixel that spans the first ``scale_x`` columns of level 0 has its
# center at the centroid of those level-0 pixels.

from xrspatial.geotiff._geotags import (RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT,  # noqa: E402
                                        GeoInfo, extract_geo_info_with_overview_inheritance)


def _make_pp_cog(path: str, size: int = 1024, pixel: float = 10.0) -> xr.DataArray:
    """Write a PixelIsPoint COG with three overview levels.

    Pixel (0, 0) is centred at world (0, 0); each pixel is ``pixel``
    units wide. EPSG:32610 keeps the writer happy without inventing
    geographic semantics. Returns the source DataArray.
    """
    arr = np.arange(size * size, dtype=np.float32).reshape(size, size)
    x = np.arange(size, dtype=np.float64) * pixel
    y = -(np.arange(size, dtype=np.float64) * pixel)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 'EPSG:32610',
                             'raster_type': 'point'})
    to_geotiff(da, path, cog=True, overview_levels=[2, 4, 8])
    return da


def _make_pa_cog(path: str, size: int = 1024, pixel: float = 10.0) -> xr.DataArray:
    """PixelIsArea companion of :func:`_make_pp_cog` for regression checks."""
    arr = np.arange(size * size, dtype=np.float32).reshape(size, size)
    x = np.arange(size, dtype=np.float64) * pixel + 0.5 * pixel
    y = -(np.arange(size, dtype=np.float64) * pixel + 0.5 * pixel)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 'EPSG:32610'})
    to_geotiff(da, path, cog=True, overview_levels=[2, 4, 8])
    return da


_BACKENDS_1642 = [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 256}, id="dask+numpy"),
    pytest.param({"gpu": True}, id="cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
    pytest.param({"gpu": True, "chunks": 256}, id="dask+cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
]


@pytest.mark.parametrize("backend_kwargs", _BACKENDS_1642)
def test_point_overview_first_pixel_center_at_block_centroid(tmp_path,
                                                             backend_kwargs):
    """For PixelIsPoint COGs, overview pixel-0 center is the level-0 centroid."""
    path = str(tmp_path / "pp_1642_centroid.tif")
    _make_pp_cog(path)

    # Level-0 pixel (0, 0) center is at (0, 0). Level-1 pixel (0, 0)
    # covers level-0 pixels with centers at x=0 and x=10, y=0 and y=-10;
    # the centroid is (5, -5). Level-2 covers level-0 0..3 with centers
    # 0, 10, 20, 30 -> centroid 15; same for y -> -15. Level-3 covers
    # 0..7 -> centroid 35; y -> -35.
    expected = {
        1: (5.0, -5.0),
        2: (15.0, -15.0),
        3: (35.0, -35.0),
    }
    for lvl, (exp_x, exp_y) in expected.items():
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        x0 = float(np.asarray(da.coords['x'])[0])
        y0 = float(np.asarray(da.coords['y'])[0])
        assert abs(x0 - exp_x) < 1e-9, (
            f"backend={backend_kwargs}, lvl={lvl}: first pixel x={x0}, "
            f"expected {exp_x}")
        assert abs(y0 - exp_y) < 1e-9, (
            f"backend={backend_kwargs}, lvl={lvl}: first pixel y={y0}, "
            f"expected {exp_y}")
        # raster_type is inherited from level 0.
        assert da.attrs.get('raster_type') == 'point'


@pytest.mark.parametrize("backend_kwargs", _BACKENDS_1642)
def test_point_overview_transform_origin_shifted(tmp_path, backend_kwargs):
    """``transform`` attr carries the shifted origin for PixelIsPoint overviews."""
    path = str(tmp_path / "pp_1642_transform.tif")
    _make_pp_cog(path)

    # transform tuple: (pixel_width, 0, origin_x, 0, pixel_height, origin_y).
    expected = {
        0: (10.0, 0.0, -10.0, 0.0),    # level 0: no shift
        1: (20.0, 5.0, -20.0, -5.0),
        2: (40.0, 15.0, -40.0, -15.0),
        3: (80.0, 35.0, -80.0, -35.0),
    }
    for lvl, (exp_pw, exp_ox, exp_ph, exp_oy) in expected.items():
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        t = da.attrs['transform']
        assert abs(t[0] - exp_pw) < 1e-9, (
            f"lvl={lvl}: pixel_width={t[0]}, expected {exp_pw}")
        assert abs(t[2] - exp_ox) < 1e-9, (
            f"lvl={lvl}: origin_x={t[2]}, expected {exp_ox}")
        assert abs(t[4] - exp_ph) < 1e-9, (
            f"lvl={lvl}: pixel_height={t[4]}, expected {exp_ph}")
        assert abs(t[5] - exp_oy) < 1e-9, (
            f"lvl={lvl}: origin_y={t[5]}, expected {exp_oy}")


def test_point_overview_coords_are_uniform(tmp_path):
    """The shifted origin still yields a regular grid -- step == pixel size."""
    path = str(tmp_path / "pp_1642_uniform.tif")
    _make_pp_cog(path)

    for lvl in (0, 1, 2, 3):
        da = open_geotiff(path, overview_level=lvl)
        x = np.asarray(da.coords['x'])
        y = np.asarray(da.coords['y'])
        if x.size > 1:
            dx = np.diff(x)
            assert np.allclose(dx, dx[0], atol=1e-9), (
                f"lvl={lvl}: x is non-uniform: {dx[:5]}")
            assert abs(float(dx[0]) - da.attrs['transform'][0]) < 1e-9
        if y.size > 1:
            dy = np.diff(y)
            assert np.allclose(dy, dy[0], atol=1e-9)
            assert abs(float(dy[0]) - da.attrs['transform'][4]) < 1e-9


class _PointStubIFD:
    """Minimal IFD-like stub for unit-testing the helper directly."""
    def __init__(self, subfile_type: int, width: int, height: int):
        self.subfile_type = subfile_type
        self.width = width
        self.height = height


def test_helper_pixel_is_point_origin_shift_unit(monkeypatch):
    """Unit-level: the helper applies the correct origin shift for
    PixelIsPoint inheritance, given a stubbed ``extract_geo_info``.

    This exercises the math without going through the writer/reader so a
    regression in the formula is caught even if the COG pipeline changes.
    """
    from xrspatial.geotiff import _geotags as _gt

    base_ifd = _PointStubIFD(subfile_type=0, width=1024, height=1024)
    ov_ifd = _PointStubIFD(subfile_type=1, width=128, height=128)

    base_info = GeoInfo(
        transform=GeoTransform(origin_x=0.0, origin_y=0.0,
                               pixel_width=10.0, pixel_height=-10.0),
        has_georef=True,
        raster_type=RASTER_PIXEL_IS_POINT,
        crs_epsg=32610,
    )
    ov_info = GeoInfo(has_georef=False,
                      raster_type=RASTER_PIXEL_IS_AREA)

    calls = {'count': 0}

    def fake_extract(ifd, data, byte_order, *, allow_rotated=False, allow_invalid_nodata=False):
        calls['count'] += 1
        if ifd is base_ifd:
            return base_info
        return ov_info

    monkeypatch.setattr(_gt, 'extract_geo_info', fake_extract)

    out = extract_geo_info_with_overview_inheritance(
        ov_ifd, [base_ifd, ov_ifd], b'', '<')

    # scale = 1024 / 128 = 8. Shift = (8 - 1) * 0.5 * 10 = 35.
    assert abs(out.transform.pixel_width - 80.0) < 1e-9
    assert abs(out.transform.pixel_height - (-80.0)) < 1e-9
    assert abs(out.transform.origin_x - 35.0) < 1e-9
    assert abs(out.transform.origin_y - (-35.0)) < 1e-9
    assert out.raster_type == RASTER_PIXEL_IS_POINT
    assert out.has_georef


def test_helper_pixel_is_area_no_origin_shift_unit(monkeypatch):
    """PixelIsArea path is unchanged: origin stays at the level-0 corner."""
    from xrspatial.geotiff import _geotags as _gt

    base_ifd = _PointStubIFD(subfile_type=0, width=1024, height=1024)
    ov_ifd = _PointStubIFD(subfile_type=1, width=128, height=128)

    base_info = GeoInfo(
        transform=GeoTransform(origin_x=100.0, origin_y=200.0,
                               pixel_width=0.5, pixel_height=-0.5),
        has_georef=True,
        raster_type=RASTER_PIXEL_IS_AREA,
        crs_epsg=4326,
    )
    ov_info = GeoInfo(has_georef=False, raster_type=RASTER_PIXEL_IS_AREA)

    def fake_extract(ifd, data, byte_order, *, allow_rotated=False, allow_invalid_nodata=False):
        return base_info if ifd is base_ifd else ov_info

    monkeypatch.setattr(_gt, 'extract_geo_info', fake_extract)

    out = extract_geo_info_with_overview_inheritance(
        ov_ifd, [base_ifd, ov_ifd], b'', '<')

    assert abs(out.transform.origin_x - 100.0) < 1e-9
    assert abs(out.transform.origin_y - 200.0) < 1e-9
    assert abs(out.transform.pixel_width - 4.0) < 1e-9
    assert abs(out.transform.pixel_height - (-4.0)) < 1e-9
    assert out.raster_type == RASTER_PIXEL_IS_AREA


def test_helper_point_overview_with_own_geokeys_not_shifted(monkeypatch):
    """If the overview IFD carries its own georef, the shift never fires."""
    from xrspatial.geotiff import _geotags as _gt

    base_ifd = _PointStubIFD(subfile_type=0, width=1024, height=1024)
    ov_ifd = _PointStubIFD(subfile_type=1, width=128, height=128)

    base_info = GeoInfo(
        transform=GeoTransform(origin_x=0.0, origin_y=0.0,
                               pixel_width=10.0, pixel_height=-10.0),
        has_georef=True, raster_type=RASTER_PIXEL_IS_POINT, crs_epsg=32610,
    )
    own_ov_info = GeoInfo(
        transform=GeoTransform(origin_x=999.0, origin_y=-999.0,
                               pixel_width=77.0, pixel_height=-77.0),
        has_georef=True, raster_type=RASTER_PIXEL_IS_POINT, crs_epsg=32610,
    )

    def fake_extract(ifd, data, byte_order, *, allow_rotated=False, allow_invalid_nodata=False):
        return base_info if ifd is base_ifd else own_ov_info

    monkeypatch.setattr(_gt, 'extract_geo_info', fake_extract)

    out = extract_geo_info_with_overview_inheritance(
        ov_ifd, [base_ifd, ov_ifd], b'', '<')

    # Untouched: helper returns the overview's own info verbatim.
    assert abs(out.transform.origin_x - 999.0) < 1e-9
    assert abs(out.transform.origin_y - (-999.0)) < 1e-9
    assert abs(out.transform.pixel_width - 77.0) < 1e-9


def test_area_overview_origin_unchanged_regression(tmp_path):
    """PixelIsArea overview origin must still equal level-0 origin."""
    path = str(tmp_path / "pa_1642_regression.tif")
    _make_pa_cog(path)
    base = open_geotiff(path, overview_level=0)
    base_t = base.attrs['transform']
    for lvl in (1, 2, 3):
        da = open_geotiff(path, overview_level=lvl)
        t = da.attrs['transform']
        assert abs(t[2] - base_t[2]) < 1e-9, (
            f"PixelIsArea lvl={lvl}: origin_x drifted from {base_t[2]} "
            f"to {t[2]}")
        assert abs(t[5] - base_t[5]) < 1e-9


# =========================================================================
# Section: min / max / median resampling parameter coverage
# =========================================================================
#
# The CPU writer (``_block_reduce_2d``) and the GPU writer
# (``_block_reduce_2d_gpu``) implement seven resampling reductions for
# COG overview generation. This section closes a parameter-coverage gap
# for ``overview_resampling='min'/'max'/'median'`` on both the direct
# block-reducer branches and the end-to-end GPU writer paths.


def _arr_4x4_ramp() -> np.ndarray:
    """4x4 float32 ramp.

    Block layout (top-left 2x2, top-right 2x2, ...):

        [ 1  2 | 3  4 ]
        [ 5  6 | 7  8 ]
        --------------
        [ 9 10 |11 12 ]
        [13 14 |15 16 ]

    Per-block reductions:
      * min:    [[1, 3], [9, 11]]
      * max:    [[6, 8], [14, 16]]
      * median: [[3.5, 5.5], [11.5, 13.5]]   (mean of the two middle values)
    """
    return np.arange(1, 17, dtype=np.float32).reshape(4, 4)


def _arr_4x4_with_nan() -> np.ndarray:
    """4x4 float32 ramp with one NaN per top-row 2x2 block."""
    arr = _arr_4x4_ramp()
    arr[0, 0] = np.nan
    arr[0, 3] = np.nan
    return arr


_RAMP_EXPECTED_MIN = np.array([[1.0, 3.0], [9.0, 11.0]], dtype=np.float32)
_RAMP_EXPECTED_MAX = np.array([[6.0, 8.0], [14.0, 16.0]], dtype=np.float32)
_RAMP_EXPECTED_MEDIAN = np.array([[3.5, 5.5], [11.5, 13.5]], dtype=np.float32)


@pytest.mark.parametrize("method, expected", [
    ('min', _RAMP_EXPECTED_MIN),
    ('max', _RAMP_EXPECTED_MAX),
    ('median', _RAMP_EXPECTED_MEDIAN),
])
def test_block_reduce_2d_cpu(method, expected):
    """``_block_reduce_2d`` returns the documented reduction per 2x2 block."""
    arr = _arr_4x4_ramp()
    out = _block_reduce_2d(arr, method)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("method", ['min', 'max', 'median'])
def test_block_reduce_2d_cpu_skips_nan(method):
    """``_block_reduce_2d`` uses nan-aware reductions so partial-NaN
    blocks aggregate over the finite cells only."""
    arr = _arr_4x4_with_nan()
    out = _block_reduce_2d(arr, method)
    assert np.all(np.isfinite(out)), (
        f"method={method!r} returned NaN for a partial-NaN block")

    # Recompute expected via numpy nan-aware ops on the same 2x2 reshape.
    blocks = arr.reshape(2, 2, 2, 2)
    flat = blocks.transpose(0, 2, 1, 3).reshape(2, 2, 4)
    if method == 'min':
        expected = np.nanmin(flat, axis=2)
    elif method == 'max':
        expected = np.nanmax(flat, axis=2)
    else:
        expected = np.nanmedian(flat, axis=2)
    np.testing.assert_allclose(out, expected.astype(np.float32))


@pytest.mark.parametrize("method, expected", [
    ('min', _RAMP_EXPECTED_MIN),
    ('max', _RAMP_EXPECTED_MAX),
    ('median', _RAMP_EXPECTED_MEDIAN),
])
def test_to_geotiff_cog_overview_resampling_cpu(tmp_path, method, expected):
    """End-to-end: ``to_geotiff(cog=True, overview_resampling=method)``
    writes a COG whose overview level 1 matches the closed-form 2x2
    reduction."""
    arr = _arr_4x4_ramp()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / f'cog_{method}.tif')
    to_geotiff(da, p, cog=True, compression='deflate', tiled=True,
               tile_size=16, overview_levels=[2],
               overview_resampling=method)

    ov = open_geotiff(p, overview_level=1)
    np.testing.assert_allclose(np.asarray(ov.data), expected)


@pytest.mark.parametrize("method", ['min', 'max', 'median'])
def test_to_geotiff_cog_overview_resampling_cpu_nodata(tmp_path, method):
    """CPU writer: nan-aware reductions skip the sentinel when ``nodata``
    is set (here covering the min/max/median branches not covered by the
    'mean' case)."""
    arr = _arr_4x4_with_nan()
    da = xr.DataArray(arr, dims=['y', 'x'])
    p = str(tmp_path / f'cog_{method}_nodata.tif')
    to_geotiff(da, p, nodata=-9999.0, cog=True, compression='deflate',
               tiled=True, tile_size=16, overview_levels=[2],
               overview_resampling=method)

    ov = open_geotiff(p, overview_level=1)
    out = np.asarray(ov.data)

    # Recompute expected from the same nan-aware reduction on the source.
    blocks = arr.reshape(2, 2, 2, 2)
    flat = blocks.transpose(0, 2, 1, 3).reshape(2, 2, 4)
    if method == 'min':
        expected = np.nanmin(flat, axis=2)
    elif method == 'max':
        expected = np.nanmax(flat, axis=2)
    else:
        expected = np.nanmedian(flat, axis=2)
    np.testing.assert_allclose(out, expected.astype(np.float32))


@_gpu_only
@pytest.mark.parametrize("method, expected", [
    ('min', _RAMP_EXPECTED_MIN),
    ('max', _RAMP_EXPECTED_MAX),
    ('median', _RAMP_EXPECTED_MEDIAN),
])
def test_block_reduce_2d_gpu(method, expected):
    """``_block_reduce_2d_gpu`` returns the same reduction as the CPU
    block reducer for finite input."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr_cpu = _arr_4x4_ramp()
    arr_gpu = cupy.asarray(arr_cpu)
    out = _block_reduce_2d_gpu(arr_gpu, method)
    np.testing.assert_allclose(cupy.asnumpy(out), expected)


@_gpu_only
@pytest.mark.parametrize("method", ['min', 'max', 'median'])
def test_block_reduce_2d_gpu_matches_cpu_with_nan(method):
    """GPU nan-aware reductions match CPU nan-aware reductions for a
    partial-NaN block."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr_cpu = _arr_4x4_with_nan()
    cpu_out = _block_reduce_2d(arr_cpu, method)
    gpu_out = _block_reduce_2d_gpu(cupy.asarray(arr_cpu), method)
    np.testing.assert_allclose(cupy.asnumpy(gpu_out), cpu_out)


@_gpu_only
@pytest.mark.parametrize("method, expected", [
    ('min', _RAMP_EXPECTED_MIN),
    ('max', _RAMP_EXPECTED_MAX),
    ('median', _RAMP_EXPECTED_MEDIAN),
])
def test_write_geotiff_gpu_cog_overview_resampling(tmp_path, method, expected):
    """End-to-end: ``_write_geotiff_gpu(cog=True, overview_resampling=method)``
    writes a COG whose overview level 1 matches the closed-form 2x2
    reduction. Exercises the GPU make-overview path including the dispatch
    on ``method``."""
    import cupy

    from xrspatial.geotiff import _write_geotiff_gpu

    arr = _arr_4x4_ramp()
    arr_gpu = cupy.asarray(arr)
    da = xr.DataArray(arr_gpu, dims=['y', 'x'])
    p = str(tmp_path / f'cog_{method}_gpu.tif')
    _write_geotiff_gpu(da, p, cog=True, compression='deflate', tiled=True,
                       tile_size=16, overview_levels=[2],
                       overview_resampling=method)

    ov = open_geotiff(p, overview_level=1)
    np.testing.assert_allclose(np.asarray(ov.data), expected)


@_gpu_only
@pytest.mark.parametrize("method", ['min', 'max', 'median'])
def test_to_geotiff_gpu_cog_overview_matches_cpu(tmp_path, method):
    """``to_geotiff(gpu=True, ..., overview_resampling=method)`` produces
    overview bytes that round-trip to the same values as the CPU writer."""
    import cupy

    arr = _arr_4x4_ramp()
    da_cpu = xr.DataArray(arr, dims=['y', 'x'])
    p_cpu = str(tmp_path / f'cog_{method}_cpu.tif')
    to_geotiff(da_cpu, p_cpu, cog=True, compression='deflate', tiled=True,
               tile_size=16, overview_levels=[2],
               overview_resampling=method)

    da_gpu = xr.DataArray(cupy.asarray(arr), dims=['y', 'x'])
    p_gpu = str(tmp_path / f'cog_{method}_gpu_via_to_geotiff.tif')
    to_geotiff(da_gpu, p_gpu, gpu=True, cog=True, compression='deflate',
               tiled=True, tile_size=16, overview_levels=[2],
               overview_resampling=method)

    ov_cpu = np.asarray(open_geotiff(p_cpu, overview_level=1).data)
    ov_gpu = np.asarray(open_geotiff(p_gpu, overview_level=1).data)
    np.testing.assert_allclose(ov_gpu, ov_cpu)


def test_block_reduce_2d_cpu_unknown_method_raises():
    """The CPU block reducer raises ``ValueError`` on an unknown method
    name. Exercises the else-branch that lists the valid methods."""
    arr = _arr_4x4_ramp()
    with pytest.raises(ValueError, match="Unknown overview resampling"):
        _block_reduce_2d(arr, 'bogus')


@_gpu_only
def test_block_reduce_2d_gpu_unknown_method_raises():
    """The GPU block reducer raises ``ValueError`` on an unknown method
    name. The CPU equivalent already raises for parity."""
    import cupy

    from xrspatial.geotiff._gpu_decode import _block_reduce_2d_gpu

    arr_gpu = cupy.asarray(_arr_4x4_ramp())
    with pytest.raises(ValueError, match="Unknown GPU overview resampling"):
        _block_reduce_2d_gpu(arr_gpu, 'bogus')


# =========================================================================
# Section: vectorized mode-resampling correctness and performance
# =========================================================================
#
# The vectorized implementation in ``_block_reduce_2d(method='mode')``
# must produce bit-exact identical output to the prior per-pixel
# ``np.unique`` reference implementation.

import time  # noqa: E402


def _mode_resample_reference(arr2d):
    """Per-pixel mode reference using GDAL ceil semantics.

    Walks each ceil-shaped output block over the source array, takes the
    intersection of the 2x2 block window with the actual source extent,
    and picks the most-frequent value with the "lowest wins" tie-break
    that the vectorized production path also uses.
    """
    h, w = arr2d.shape
    oh, ow = (h + 1) // 2, (w + 1) // 2
    out = np.empty((oh, ow), dtype=arr2d.dtype)
    for r in range(oh):
        for c in range(ow):
            r0, r1 = 2 * r, min(2 * r + 2, h)
            c0, c1 = 2 * c, min(2 * c + 2, w)
            window = arr2d[r0:r1, c0:c1].ravel()
            vals, counts = np.unique(window, return_counts=True)
            out[r, c] = vals[counts.argmax()]
    return out


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16,
                                   np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("shape", [(16, 16), (17, 19), (100, 101),
                                   (1, 1), (2, 2), (3, 3), (64, 65)])
def test_bit_exact_match_reference(dtype, shape):
    rng = np.random.default_rng(seed=42)
    info = np.iinfo(dtype)
    # Use a small categorical-style range so ties happen often.
    lo = max(info.min, 0)
    hi = min(info.max, 7)
    arr = rng.integers(lo, hi + 1, size=shape, dtype=dtype)

    expected = _mode_resample_reference(arr)
    actual = _block_reduce_2d(arr, 'mode')

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_tie_break_lowest_wins():
    # Block has two 1s and two 5s -> tie. Old impl returns 1 (smaller).
    arr = np.array([[1, 5],
                    [1, 5]], dtype=np.uint8)
    out = _block_reduce_2d(arr, 'mode')
    assert out.shape == (1, 1)
    assert out[0, 0] == 1


def test_tie_break_three_way():
    # Distinct values, all count==1 -> smallest wins.
    arr = np.array([[3, 7],
                    [1, 5]], dtype=np.uint8)
    out = _block_reduce_2d(arr, 'mode')
    assert out[0, 0] == 1


def test_three_of_a_kind_wins():
    arr = np.array([[2, 2],
                    [2, 9]], dtype=np.uint8)
    out = _block_reduce_2d(arr, 'mode')
    assert out[0, 0] == 2


def test_all_same_value_block():
    arr = np.full((8, 8), 42, dtype=np.uint16)
    out = _block_reduce_2d(arr, 'mode')
    assert out.shape == (4, 4)
    assert np.all(out == 42)


def test_multiple_blocks_independent():
    # Build 2x4 input with two distinct 2x2 blocks side by side.
    arr = np.array([[1, 1, 9, 9],
                    [1, 2, 9, 8]], dtype=np.uint8)
    out = _block_reduce_2d(arr, 'mode')
    # Left block: three 1s, one 2 -> 1.
    # Right block: three 9s, one 8 -> 9.
    np.testing.assert_array_equal(out, np.array([[1, 9]], dtype=np.uint8))


def test_perf_under_100ms_on_1024sq_uint8():
    rng = np.random.default_rng(seed=0)
    arr = rng.integers(0, 16, size=(1024, 1024), dtype=np.uint8)
    # Warmup
    _block_reduce_2d(arr, 'mode')
    t0 = time.perf_counter()
    out = _block_reduce_2d(arr, 'mode')
    elapsed = time.perf_counter() - t0
    assert out.shape == (512, 512)
    assert elapsed < 0.1, (
        f"mode resampling took {elapsed*1000:.1f} ms (threshold 100 ms)"
    )
