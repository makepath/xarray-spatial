"""Parity tests for the ngjit 2x2 overview kernels.

The float ``mean`` / ``min`` / ``max`` / ``median`` paths in
:func:`xrspatial.geotiff._overview._block_reduce_2d` route through the
type-specialized kernels in :mod:`xrspatial.geotiff._overview_kernels`.
These tests pin the kernels' output against a numpy reference
implementation that mirrors the original nan-aware aggregation, across
float32 / float64, even / odd shapes, with and without a sentinel, and
across NaN edge cases (no valid cells, all-sentinel block, partial
block).
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.geotiff._overview import _block_reduce_2d
from xrspatial.geotiff._overview_kernels import KERNELS


def _reference_reduce(arr, method, nodata=None):
    """Numpy reference: pad to even, mask sentinel to NaN, nan-aggregate.

    Mirrors the original implementation so the new kernels can be
    compared cell-by-cell on the same input.
    """
    h, w = arr.shape
    oh, ow = (h + 1) // 2, (w + 1) // 2
    h2, w2 = 2 * oh, 2 * ow
    padded = np.full((h2, w2), np.nan, dtype=np.float64)
    padded[:h, :w] = arr.astype(np.float64)
    if nodata is not None and not np.isnan(nodata):
        padded = np.where(padded == float(nodata), np.nan, padded)
    blocks = padded.reshape(oh, 2, ow, 2)
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore', RuntimeWarning)
        if method == 'mean':
            ref = np.nanmean(blocks, axis=(1, 3))
        elif method == 'min':
            ref = np.nanmin(blocks, axis=(1, 3))
        elif method == 'max':
            ref = np.nanmax(blocks, axis=(1, 3))
        elif method == 'median':
            flat = blocks.transpose(0, 2, 1, 3).reshape(oh, ow, 4)
            ref = np.nanmedian(flat, axis=2)
        else:
            raise ValueError(method)
    return ref.astype(arr.dtype)


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(2, 2), (4, 4), (8, 8), (5, 5),
                                   (5, 4), (4, 5), (7, 3), (1, 1),
                                   (1, 5), (5, 1)])
def test_kernel_matches_numpy_reference_no_nodata(method, dtype, shape):
    rng = np.random.default_rng(2413)
    arr = rng.random(shape, dtype=np.float32).astype(dtype)
    out = _block_reduce_2d(arr, method)
    ref = _reference_reduce(arr, method)
    assert out.dtype == dtype
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(4, 4), (5, 5), (5, 4), (7, 3)])
def test_kernel_matches_numpy_reference_with_sentinel(method, dtype, shape):
    sentinel = -9999.0
    rng = np.random.default_rng(2413)
    arr = rng.random(shape, dtype=np.float32).astype(dtype)
    # Sprinkle ~25 % sentinels so most blocks are mixed valid+sentinel
    # while a few are all-sentinel; covers both branches in one pass.
    mask = rng.random(shape) < 0.25
    arr[mask] = sentinel
    out = _block_reduce_2d(arr, method, nodata=sentinel)
    ref = _reference_reduce(arr, method, nodata=sentinel)
    np.testing.assert_allclose(out, ref, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
def test_kernel_all_sentinel_block_returns_nan(method):
    # Every cell is the sentinel -> kernel must report NaN so the
    # caller's downstream sentinel-rewrite still kicks in.
    arr = np.full((2, 2), -9999.0, dtype=np.float32)
    out = _block_reduce_2d(arr, method, nodata=-9999.0)
    assert out.shape == (1, 1)
    assert np.isnan(out[0, 0])


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
def test_kernel_all_nan_block_returns_nan(method):
    arr = np.full((2, 2), np.nan, dtype=np.float32)
    out = _block_reduce_2d(arr, method)
    assert np.isnan(out[0, 0])


@pytest.mark.parametrize('method', ['mean', 'min', 'max', 'median'])
def test_kernel_residual_block_uses_only_in_bounds_cells(method):
    # 3x3 input -> 2x2 output. The trailing column / row produces 2x1
    # / 1x2 / 1x1 residual blocks; only the in-bounds source cells
    # should contribute.
    arr = np.array(
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    out = _block_reduce_2d(arr, method)
    if method == 'mean':
        # block (0,1): [3, 6]
        assert out[0, 1] == pytest.approx((3.0 + 6.0) / 2)
        # block (1,0): [7, 8]
        assert out[1, 0] == pytest.approx((7.0 + 8.0) / 2)
        # block (1,1): [9]
        assert out[1, 1] == pytest.approx(9.0)
    elif method == 'min':
        assert out[0, 1] == 3.0
        assert out[1, 0] == 7.0
        assert out[1, 1] == 9.0
    elif method == 'max':
        assert out[0, 1] == 6.0
        assert out[1, 0] == 8.0
        assert out[1, 1] == 9.0
    else:  # median
        assert out[0, 1] == pytest.approx(4.5)
        assert out[1, 0] == pytest.approx(7.5)
        assert out[1, 1] == pytest.approx(9.0)


def test_kernel_median_4_values_matches_numpy():
    # Pin median-of-4 against np.median over many random 2x2 blocks.
    rng = np.random.default_rng(2413)
    arr = rng.random((128, 128), dtype=np.float32)
    out = _block_reduce_2d(arr, 'median')
    ref = np.median(arr.reshape(64, 2, 64, 2).transpose(0, 2, 1, 3).reshape(64, 64, 4),
                    axis=2).astype(np.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-6)


def test_kernel_inf_sentinel_is_masked():
    # nodata=inf must be honoured exactly like a finite sentinel; the
    # kernel's ``v != sentinel`` check works because inf == inf is True.
    arr = np.array([[1.0, 2.0, 3.0, 4.0],
                    [np.inf, np.inf, np.inf, np.inf],
                    [10.0, 20.0, 30.0, 40.0],
                    [10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    out = _block_reduce_2d(arr, 'mean', nodata=float('inf'))
    np.testing.assert_allclose(out[0, 0], 1.5)
    np.testing.assert_allclose(out[0, 1], 3.5)


def test_kernel_nan_nodata_is_noop():
    # nodata=NaN must not mask anything beyond the existing NaN
    # detection in the kernel.
    arr = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = _block_reduce_2d(arr, 'mean', nodata=np.nan)
    ref = _reference_reduce(arr, 'mean')
    np.testing.assert_allclose(out, ref, rtol=1e-6)


def test_kernel_dispatch_table_has_expected_methods():
    # Guard against silent removal of a method from the dispatch table
    # (would silently fall back to the slower numpy path).
    assert set(KERNELS.keys()) == {'mean', 'min', 'max', 'median'}


def test_integer_dtype_still_routes_through_numpy_path():
    # The integer path is intentionally left on the numpy nan-aware
    # branch because the sentinel mask must be computed at native
    # integer width (the 64-bit sentinel cases hit this); this test
    # pins that contract so a future change does not accidentally
    # divert integers through the float kernels.
    sentinel = np.iinfo(np.int64).max
    arr = np.full((5, 5), 10, dtype=np.int64)
    arr[0, 0] = sentinel
    out = _block_reduce_2d(arr, 'min', nodata=sentinel)
    # The top-left 2x2 has 1 sentinel + 3 valid 10s; correct min is 10.
    # A float-cast comparison would lose the sentinel near INT64_MAX.
    assert out[0, 0] == 10
    assert out.dtype == np.int64
