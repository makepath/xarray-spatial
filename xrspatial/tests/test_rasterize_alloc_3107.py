"""Avoidable full-raster allocations in rasterize backends (issue #3107).

Locks in two allocation fixes:

1. Every backend returns through ``astype(dtype, copy=False)``, so the
   default float64 case no longer copies the full work buffer.
2. The CPU paths allocate ``order`` as int8 (1 byte/pixel) for
   order-insensitive merges (max/min/sum/count, user callables), keeping
   int64 only for the ``first`` / ``last`` predicates that actually read
   the stored owner indices.

The int8 buffer still receives int64 owner-index stores (numba wraps
them), so the tests here use more than 127 geometries to push the
wrapped values through their full range and prove the output never
depends on them.
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

try:
    from shapely.geometry import LineString, Point, box
    has_shapely = True
except ImportError:
    has_shapely = False

try:
    import dask.array as da  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False

if has_shapely:
    from xrspatial.rasterize import (
        _alloc_order,
        _should_write_any,
        _should_write_first,
        _should_write_last,
        rasterize,
    )
    from xrspatial.utils import ngjit

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)

skip_no_dask = pytest.mark.skipif(
    not has_dask, reason="dask not installed")


# ---------------------------------------------------------------------------
# _alloc_order dtype selection
# ---------------------------------------------------------------------------

class TestAllocOrder:
    def test_order_insensitive_predicate_gets_int8(self):
        order = _alloc_order(4, 5, _should_write_any)
        assert order.shape == (4, 5)
        assert order.dtype == np.int8

    @pytest.mark.parametrize(
        "predicate", [_should_write_first, _should_write_last],
        ids=["first", "last"])
    def test_ordered_predicates_keep_int64(self, predicate):
        order = _alloc_order(4, 5, predicate)
        assert order.shape == (4, 5)
        assert order.dtype == np.int64
        assert (np.asarray(order) == -1).all()


# ---------------------------------------------------------------------------
# Output parity with >127 geometries (int8 owner indices wrap)
# ---------------------------------------------------------------------------

def _many_overlapping_geoms(n=300):
    """n overlapping boxes plus a line and a point, all over one spot."""
    geoms = [(box(0, 0, 10, 10), float(i + 1)) for i in range(n)]
    geoms.append((LineString([(0, 0), (20, 20)]), 500.0))
    geoms.append((Point(5.0, 5.0), 600.0))
    return geoms


class TestWrappedOrderIndices:
    """300+ geometries: every int8 owner-index store wraps at least twice."""

    @pytest.mark.parametrize("merge", ["sum", "count", "min", "max"])
    def test_order_insensitive_merge_values(self, merge):
        geoms = _many_overlapping_geoms()
        result = rasterize(geoms, width=64, height=64,
                           bounds=(0, 0, 20, 20), merge=merge)
        data = result.data
        # Pixel (60, 4) sees only the 300 boxes: the diagonal line burns
        # (60, 3) / (59, 4), and the point lands on (48, 16).
        boxes_only = {'sum': 45150.0, 'count': 300.0,
                      'min': 1.0, 'max': 300.0}
        assert data[60, 4] == boxes_only[merge]
        # Pixel (48, 16) is the point burn (600) on top of the boxes.
        with_point = {'sum': 45750.0, 'count': 301.0,
                      'min': 1.0, 'max': 600.0}
        assert data[48, 16] == with_point[merge]

    @pytest.mark.parametrize("merge", ["first", "last"])
    def test_ordered_merges_unaffected(self, merge):
        geoms = _many_overlapping_geoms()
        result = rasterize(geoms, width=64, height=64,
                           bounds=(0, 0, 20, 20), merge=merge)
        # Away from the line and point, input order decides.
        expected = 1.0 if merge == 'first' else 300.0
        assert result.data[60, 4] == expected

    def test_user_callable_gets_int8_order_and_correct_output(self):
        @ngjit
        def merge_accumulate(pixel, props, is_first):
            if is_first:
                return props[0]
            return pixel + props[0]

        geoms = _many_overlapping_geoms()
        result = rasterize(geoms, width=64, height=64,
                           bounds=(0, 0, 20, 20),
                           merge=merge_accumulate)
        assert result.data[60, 4] == sum(range(1, 301))

    @skip_no_dask
    @pytest.mark.parametrize(
        "merge", ["sum", "count", "min", "max", "first", "last"])
    def test_numpy_dask_parity(self, merge):
        geoms = _many_overlapping_geoms()
        r_np = rasterize(geoms, width=64, height=64,
                         bounds=(0, 0, 20, 20), merge=merge)
        r_da = rasterize(geoms, width=64, height=64,
                         bounds=(0, 0, 20, 20), merge=merge,
                         chunks=(16, 16))
        np.testing.assert_allclose(
            r_np.data, r_da.data.compute(), equal_nan=True)


# ---------------------------------------------------------------------------
# Peak-memory regression
# ---------------------------------------------------------------------------

class TestPeakMemory:
    """tracemalloc bounds on the numpy backend.

    Before #3107 the numpy path peaked at ~25 B/px for any merge:
    out f64 (8) + order i64 (8) + final astype copy (8) + written i8 (1).
    After: ~10 B/px for order-insensitive merges, ~17 B/px for
    first/last.  The thresholds leave several bytes/pixel of headroom so
    interpreter noise cannot flake the test, while staying far below the
    old values.
    """

    H = W = 2000  # 4M px: buffers dwarf interpreter overhead

    def _measure(self, merge):
        geoms = [(box(0, 0, 1000, 1000), float(i + 1)) for i in range(8)]
        # Warm up JIT so compilation allocations don't pollute the peak.
        rasterize(geoms, width=32, height=32, bounds=(0, 0, 2000, 2000),
                  merge=merge)
        tracemalloc.start()
        rasterize(geoms, width=self.W, height=self.H,
                  bounds=(0, 0, 2000, 2000), merge=merge)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / (self.H * self.W)

    def test_order_insensitive_merge_peak(self):
        # out f64 + written i8 + order i8 = 10 B/px; was 25 B/px.
        assert self._measure('sum') < 16.0

    def test_ordered_merge_peak(self):
        # out f64 + written i8 + order i64 = 17 B/px; was 25 B/px.
        assert self._measure('last') < 21.0


# ---------------------------------------------------------------------------
# dtype handling still intact
# ---------------------------------------------------------------------------

class TestDtypeStillRespected:
    def test_default_dtype_is_float64(self):
        result = rasterize([(box(0, 0, 5, 5), 3.0)], width=8, height=8,
                           bounds=(0, 0, 8, 8))
        assert result.dtype == np.float64
        assert result.data[7, 0] == 3.0

    def test_non_default_dtype_still_casts(self):
        result = rasterize([(box(0, 0, 5, 5), 3.0)], width=8, height=8,
                           bounds=(0, 0, 8, 8), dtype=np.int32, fill=0)
        assert result.dtype == np.int32
        assert result.data[7, 0] == 3

    @skip_no_dask
    def test_dask_tile_dtype(self):
        result = rasterize([(box(0, 0, 5, 5), 3.0)], width=8, height=8,
                           bounds=(0, 0, 8, 8), dtype=np.float32, fill=0,
                           chunks=(4, 4))
        out = result.data.compute()
        assert out.dtype == np.float32
        assert out[7, 0] == 3.0
