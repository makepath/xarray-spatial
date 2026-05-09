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
    """Original per-pixel reference implementation (pre-vectorization).

    Copied verbatim from the loop body that used ``np.unique`` per 2x2
    block, with the same crop-to-even-dims behavior as the production
    function.
    """
    h, w = arr2d.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    cropped = arr2d[:h2, :w2]
    oh, ow = h2 // 2, w2 // 2
    blocks = (
        cropped.reshape(oh, 2, ow, 2).transpose(0, 2, 1, 3).reshape(oh, ow, 4)
    )
    out = np.empty((oh, ow), dtype=arr2d.dtype)
    for r in range(oh):
        for c in range(ow):
            vals, counts = np.unique(blocks[r, c], return_counts=True)
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

    # Skip pathological 1x1 / 1xN / Nx1 cases that crop to empty.
    h2 = (shape[0] // 2) * 2
    w2 = (shape[1] // 2) * 2
    if h2 == 0 or w2 == 0:
        return

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
