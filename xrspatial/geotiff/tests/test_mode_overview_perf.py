"""Correctness and performance tests for vectorized mode-resampling.

The vectorized implementation in ``_block_reduce_2d(method='mode')``
must produce bit-exact identical output to the prior per-pixel
``np.unique`` reference implementation.
"""

import time

import numpy as np
import pytest

from xrspatial.geotiff._writer import _block_reduce_2d


def _mode_resample_reference(arr2d):
    """Per-pixel mode reference using GDAL ceil semantics (issue #2105).

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
