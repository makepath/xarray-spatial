"""Dask float polygonize chunk-invariance for rtol (#2666).

The numpy connected-component predicate in ``_calculate_regions`` is
direction-aware: for an adjacent pixel pair it tests
``abs(value - reference) <= atol + rtol*abs(reference)`` where
``reference`` is the higher-``ij`` pixel in raster-scan order.  Because
``rtol`` scales by ``abs(reference)``, the test is asymmetric.

The old dask cross-chunk merge compared unordered ``(min, max)`` value
ranges, so the merge decision flipped with chunk order instead of
matching numpy.  Concretely, ``[[10.0, 11.05]]`` with ``atol=0``,
``rtol=0.1`` merged into one polygon under numpy but split into two
under single-pixel-chunk dask; reversing the row flipped the failure.

The fix carries each chunk-boundary polygon's per-cell boundary values
and applies numpy's higher-``ij``-is-reference orientation at the exact
pixel pair straddling a chunk boundary.  These tests pin numpy/dask
parity for the repro, its reverse, transitive chains, and a sweep of
chunkings so a chunked float raster yields the same polygon partition
as the unchunked input.
"""
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

try:
    import cupy
except ImportError:
    cupy = None

try:
    import dask.array as da
except ImportError:
    da = None

from ..polygonize import polygonize
from .general_checks import cuda_and_cupy_available, dask_array_available


def _to_dask(arr, chunks):
    return xr.DataArray(da.from_array(arr, chunks=chunks))


def _to_dask_cupy(arr, chunks):
    return xr.DataArray(da.from_array(cupy.asarray(arr), chunks=chunks))


def _all_chunkings(shape):
    """Enumerate a useful spread of 2D chunkings for an array shape."""
    ny, nx = shape
    candidates = {
        (1, 1),
        (1, nx),
        (ny, 1),
        (ny, nx),
        (max(1, ny // 2) or 1, max(1, nx // 2) or 1),
    }
    return sorted(candidates)


# The exact repro from the issue and its row-reversed twin.  Forward:
# numpy merges (1 polygon), old dask split (2).  Reverse: numpy splits
# (2), old dask merged (1).
_REPRO_FORWARD = np.array([[10.0, 11.05]], dtype=np.float64)
_REPRO_REVERSE = np.array([[11.05, 10.0]], dtype=np.float64)


@dask_array_available
class TestReproDirectionAware:
    """The minimal repro and its reverse must match numpy in both
    directions with ``atol=0``, ``rtol=0.1``.
    """

    def test_forward_merges_like_numpy(self):
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_FORWARD), atol=0.0, rtol=0.1)
        v_dk, _ = polygonize(
            _to_dask(_REPRO_FORWARD, chunks=(1, 1)), atol=0.0, rtol=0.1)
        assert len(v_np) == 1
        assert len(v_dk) == len(v_np)
        assert_allclose(sorted(v_dk), sorted(v_np))

    def test_reverse_splits_like_numpy(self):
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_REVERSE), atol=0.0, rtol=0.1)
        v_dk, _ = polygonize(
            _to_dask(_REPRO_REVERSE, chunks=(1, 1)), atol=0.0, rtol=0.1)
        assert len(v_np) == 2
        assert len(v_dk) == len(v_np)
        assert_allclose(sorted(v_dk), sorted(v_np))


@dask_array_available
class TestChunkInvariance:
    """For each raster + tolerance, every chunking must produce the same
    polygon count and DN multiset as the single-chunk numpy reference.
    """

    @pytest.mark.parametrize(
        "arr",
        [
            _REPRO_FORWARD,
            _REPRO_REVERSE,
            # Direction-sensitive chain forward and reversed.  Each
            # adjacent pair is within rtol=0.1 only in one orientation.
            np.array([[10.0, 11.0, 12.1]], dtype=np.float64),
            np.array([[12.1, 11.0, 10.0]], dtype=np.float64),
            # Same idea down a column (tests North/South orientation).
            np.array([[10.0], [11.0], [12.1]], dtype=np.float64),
            np.array([[12.1], [11.0], [10.0]], dtype=np.float64),
            # 2D mix of close and distinct values.
            np.array([[10.0, 11.05], [11.1, 11.16]], dtype=np.float64),
            # Disconnected close-valued pixels separated by a distinct
            # value: must stay distinct, not over-merge.
            np.array([[10.0, 50.0, 10.5]], dtype=np.float64),
        ],
    )
    @pytest.mark.parametrize("rtol", [0.0, 0.05, 0.1])
    def test_parity_4conn(self, arr, rtol):
        v_np, _ = polygonize(
            xr.DataArray(arr), atol=0.0, rtol=rtol, connectivity=4)
        for chunks in _all_chunkings(arr.shape):
            v_dk, _ = polygonize(
                _to_dask(arr, chunks=chunks),
                atol=0.0, rtol=rtol, connectivity=4)
            assert len(v_dk) == len(v_np), (
                f"count mismatch arr={arr.tolist()} rtol={rtol} "
                f"chunks={chunks}: numpy={len(v_np)} dask={len(v_dk)}"
            )
            assert_allclose(sorted(v_dk), sorted(v_np))


@dask_array_available
class TestDefaultToleranceUnaffected:
    """The default tolerance path (no explicit atol/rtol) must keep its
    existing chunk-invariant behaviour after the orientation fix.
    """

    def test_symptom_a_chain_still_merges(self):
        arr = np.array([[1.0, 1.000009, 1.000018]], dtype=np.float64)
        v_np, _ = polygonize(xr.DataArray(arr))
        for chunks in [(1, 1), (1, 2), (1, 3)]:
            v_dk, _ = polygonize(_to_dask(arr, chunks=chunks))
            assert len(v_dk) == len(v_np) == 1
            assert_allclose(sorted(v_dk), sorted(v_np))

    def test_disconnected_close_keeps_dn(self):
        arr = np.array([[1.0, 9.0, 1.000009]], dtype=np.float64)
        v_np, _ = polygonize(xr.DataArray(arr))
        v_dk, _ = polygonize(_to_dask(arr, chunks=(1, 1)))
        assert len(v_dk) == len(v_np) == 3
        assert_allclose(sorted(v_dk), sorted(v_np))
        assert any(abs(v - 1.000009) < 1e-12 for v in v_dk)


@cuda_and_cupy_available
@dask_array_available
class TestReproDaskCuPy:
    """The repro and its reverse must also match numpy on dask+cupy,
    which shares the cross-chunk merge path with dask+numpy.
    """

    def test_forward_dask_cupy(self):
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_FORWARD), atol=0.0, rtol=0.1)
        v_dc, _ = polygonize(
            _to_dask_cupy(_REPRO_FORWARD, chunks=(1, 1)), atol=0.0, rtol=0.1)
        assert len(v_dc) == len(v_np) == 1
        assert_allclose(sorted(v_dc), sorted(v_np))

    def test_reverse_dask_cupy(self):
        v_np, _ = polygonize(
            xr.DataArray(_REPRO_REVERSE), atol=0.0, rtol=0.1)
        v_dc, _ = polygonize(
            _to_dask_cupy(_REPRO_REVERSE, chunks=(1, 1)), atol=0.0, rtol=0.1)
        assert len(v_dc) == len(v_np) == 2
        assert_allclose(sorted(v_dc), sorted(v_np))


@dask_array_available
class TestIntegerStrictEqualityUnaffected:
    """Integer rasters use strict equality; chunking must not change the
    result and atol/rtol must have no effect.
    """

    def test_integer_chunk_invariance(self):
        arr = np.array([[1, 1, 2], [1, 3, 2]], dtype=np.int32)
        v_np, _ = polygonize(xr.DataArray(arr))
        for chunks in _all_chunkings(arr.shape):
            v_dk, _ = polygonize(_to_dask(arr, chunks=chunks))
            assert len(v_dk) == len(v_np)
            assert sorted(v_dk) == sorted(v_np)
