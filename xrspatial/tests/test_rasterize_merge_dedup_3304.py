"""Per-geometry write dedup for merge='sum' / 'count' (issue #3304).

A single geometry used to burn the same pixel more than once under the
two non-idempotent merges:

* ``all_touched=True`` polygons: the scanline fill burned boundary
  cells whose centers are inside, then the supercover boundary pass
  burned them again, and consecutive ring segments re-burned their
  shared vertex cell -- ``merge='count'`` reported up to 3 for one
  polygon.
* lines: each Bresenham segment burned both endpoints, so the
  connecting vertex of consecutive segments counted once per segment.

The fix enumerates and deduplicates line / boundary cells per geometry
before burning, matching rasterio's ``MergeAlg.add`` (1 per geometry
per pixel, even for self-crossing lines).  Points stay un-deduplicated
on purpose: GDAL burns once per point, so a MultiPoint with repeated
coordinates adding twice already matches.
"""
import numpy as np
import pytest

try:
    from shapely.geometry import LineString, MultiPoint, Polygon
    has_shapely = True
except ImportError:
    has_shapely = False

try:
    import rasterio.features  # noqa: F401
    from rasterio.enums import MergeAlg
    from rasterio.transform import from_bounds
    has_rasterio = True
except ImportError:
    has_rasterio = False

try:
    import dask.array  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False

try:
    import cupy  # noqa: F401
    from numba import cuda
    has_cuda = cuda.is_available()
except Exception:
    has_cuda = False

if has_shapely:
    from xrspatial.rasterize import rasterize

pytestmark = pytest.mark.skipif(not has_shapely,
                                reason="shapely not installed")

_GRID = dict(width=10, height=10, bounds=(0.0, 0.0, 10.0, 10.0), fill=0)

# Subpixel vertices so no edge lies exactly on a cell boundary or
# pixel center -- tie-breaking is out of scope here.
_POLY = Polygon([(2.2, 2.2), (7.4, 3.1), (5.0, 7.9)])
_LINE = LineString([(0.5, 0.5), (5.5, 3.5), (9.5, 0.5)])
_SELF_CROSSING = LineString([(1.5, 1.5), (8.5, 8.5), (1.5, 8.5), (8.5, 1.5)])

_BACKENDS = [
    pytest.param({}, id='numpy'),
    pytest.param(dict(chunks=(4, 7)), id='dask+numpy',
                 marks=pytest.mark.skipif(not has_dask,
                                          reason="dask not installed")),
    pytest.param(dict(gpu=True), id='cupy',
                 marks=pytest.mark.skipif(not has_cuda,
                                          reason="CUDA not available")),
    pytest.param(dict(chunks=(4, 7), gpu=True), id='dask+cupy',
                 marks=pytest.mark.skipif(not (has_cuda and has_dask),
                                          reason="dask+CUDA not available")),
]


def _vals(agg):
    data = agg.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'get'):
        data = data.get()
    return np.asarray(data)


class TestSingleGeometryCountsOnce:
    @pytest.mark.parametrize("kw", _BACKENDS)
    def test_polygon_all_touched_count_is_one(self, kw):
        r = _vals(rasterize([(_POLY, 1.0)], merge='count',
                            all_touched=True, **_GRID, **kw))
        burned = r[r != 0]
        assert burned.size > 0
        assert np.all(burned == 1.0), (
            f"single polygon counted >1: max={r.max()}")

    @pytest.mark.parametrize("kw", _BACKENDS)
    def test_polygon_all_touched_sum_is_value(self, kw):
        r = _vals(rasterize([(_POLY, 2.5)], merge='sum',
                            all_touched=True, **_GRID, **kw))
        burned = r[r != 0]
        assert burned.size > 0
        assert np.all(burned == 2.5)

    @pytest.mark.parametrize("kw", _BACKENDS)
    def test_line_vertex_count_is_one(self, kw):
        r = _vals(rasterize([(_LINE, 1.0)], merge='count', **_GRID, **kw))
        burned = r[r != 0]
        assert burned.size > 0
        assert np.all(burned == 1.0), (
            f"connecting vertex double-counted: max={r.max()}")

    @pytest.mark.parametrize("kw", _BACKENDS)
    def test_self_crossing_line_count_is_one(self, kw):
        # GDAL also yields 1 at the crossing (rasterio MergeAlg.add).
        r = _vals(rasterize([(_SELF_CROSSING, 1.0)], merge='count',
                            **_GRID, **kw))
        burned = r[r != 0]
        assert burned.size > 0
        assert np.all(burned == 1.0)


class TestCoverageUnchanged:
    """Dedup must change values, never which pixels are burned."""

    @pytest.mark.parametrize("kw", _BACKENDS)
    def test_all_touched_count_coverage_matches_last(self, kw):
        cnt = _vals(rasterize([(_POLY, 1.0)], merge='count',
                              all_touched=True, **_GRID, **kw))
        last = _vals(rasterize([(_POLY, 1.0)], merge='last',
                               all_touched=True, **_GRID, **kw))
        np.testing.assert_array_equal(cnt != 0, last != 0)

    @pytest.mark.parametrize("kw", _BACKENDS)
    def test_line_sum_coverage_matches_last(self, kw):
        s = _vals(rasterize([(_LINE, 1.0)], merge='sum', **_GRID, **kw))
        last = _vals(rasterize([(_LINE, 1.0)], merge='last', **_GRID, **kw))
        np.testing.assert_array_equal(s != 0, last != 0)


class TestOverlapStillCounted:
    """Dedup is per geometry; distinct geometries still accumulate."""

    @pytest.mark.parametrize("kw", _BACKENDS)
    def test_two_identical_polygons_count_two(self, kw):
        r = _vals(rasterize([(_POLY, 1.0), (_POLY, 1.0)], merge='count',
                            all_touched=True, **_GRID, **kw))
        burned = r[r != 0]
        assert burned.size > 0
        assert np.all(burned == 2.0)

    @pytest.mark.parametrize("kw", _BACKENDS)
    def test_two_crossing_lines_sum_at_crossing(self, kw):
        a = LineString([(0.5, 5.5), (9.5, 5.5)])
        b = LineString([(5.5, 0.5), (5.5, 9.5)])
        r = _vals(rasterize([(a, 1.0), (b, 1.0)], merge='sum',
                            **_GRID, **kw))
        assert r.max() == 2.0
        assert np.sum(r == 2.0) == 1  # exactly the crossing cell


class TestBackendParity:
    @pytest.mark.parametrize("kw", _BACKENDS[1:])
    def test_matches_numpy(self, kw):
        pairs = [(_POLY, 2.0), (_LINE, 3.0), (_SELF_CROSSING, 1.0)]
        for merge in ('sum', 'count'):
            ref = _vals(rasterize(pairs, merge=merge, all_touched=True,
                                  **_GRID))
            got = _vals(rasterize(pairs, merge=merge, all_touched=True,
                                  **_GRID, **kw))
            np.testing.assert_array_equal(ref, got)


@pytest.mark.skipif(not has_rasterio, reason="rasterio not installed")
class TestRasterioAddParity:
    def _rio(self, pairs, all_touched=False):
        transform = from_bounds(0.0, 0.0, 10.0, 10.0, 10, 10)
        return rasterio.features.rasterize(
            pairs, out_shape=(10, 10), transform=transform, fill=0,
            all_touched=all_touched, merge_alg=MergeAlg.add,
            dtype='float64')

    def test_polygon_all_touched_add(self):
        rio = self._rio([(_POLY, 1.0)], all_touched=True)
        xs = _vals(rasterize([(_POLY, 1.0)], merge='sum',
                             all_touched=True, **_GRID))
        np.testing.assert_array_equal(rio, xs)

    def test_line_add_values_on_common_coverage(self):
        # Bresenham and GDAL's walker disagree on a few covered cells
        # (pre-existing, pinned in test_rasterize_coverage_2026_06_09);
        # where both burn, the dedup'd values must agree.
        rio = self._rio([(_LINE, 1.0)])
        xs = _vals(rasterize([(_LINE, 1.0)], merge='sum', **_GRID))
        common = (rio > 0) & (xs > 0)
        assert common.any()
        np.testing.assert_array_equal(rio[common], xs[common])


class TestPointsNotDeduplicated:
    @pytest.mark.parametrize("kw", _BACKENDS)
    def test_multipoint_repeated_coordinate_counts_twice(self, kw):
        # Matches rasterio: MergeAlg.add burns once per point, so a
        # repeated coordinate inside one MultiPoint adds twice.
        mp = MultiPoint([(2.5, 2.5), (2.5, 2.5)])
        r = _vals(rasterize([(mp, 1.0)], merge='count', **_GRID, **kw))
        assert r.max() == 2.0


class TestOtherMergesUntouched:
    @pytest.mark.parametrize("merge", ['last', 'first', 'min', 'max'])
    def test_single_polygon_all_touched_value(self, merge):
        r = _vals(rasterize([(_POLY, 7.0)], merge=merge,
                            all_touched=True, **_GRID))
        burned = r[r != 0]
        assert burned.size > 0
        assert np.all(burned == 7.0)


class TestTileBoundaries:
    """Geometry spanning several tiles: per-tile dedup equals global."""

    @pytest.mark.skipif(not has_dask, reason="dask not installed")
    @pytest.mark.parametrize("chunks", [(3, 3), (5, 5), (4, 7)])
    def test_chunked_equals_eager(self, chunks):
        pairs = [(_POLY, 1.0), (_LINE, 1.0)]
        for merge in ('sum', 'count'):
            eager = _vals(rasterize(pairs, merge=merge, all_touched=True,
                                    **_GRID))
            chunked = _vals(rasterize(pairs, merge=merge, all_touched=True,
                                      chunks=chunks, **_GRID))
            np.testing.assert_array_equal(eager, chunked)
