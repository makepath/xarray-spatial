"""COG overviews use ceil semantics for odd-sized rasters (issue #2105).

Before the fix, ``_block_reduce_2d`` floored both dimensions to an even
multiple and cropped the trailing row/col before reducing. A 5x5 input
became a 4x4 crop and then a 2x2 overview, silently dropping the bottom
row and right column. GDAL's overview generator uses ceil semantics
(5x5 -> 3x3) so the residual edge cells still contribute.

These tests pin the contract that every base pixel reaches the overview,
across all resampling methods, on both even and odd input shapes, and
with the nodata sentinel honoured along the trailing edge.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff._writer import _block_reduce_2d


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
