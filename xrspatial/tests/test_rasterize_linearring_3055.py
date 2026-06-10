"""Issue #3055 -- non-empty LinearRing and other unsupported geometries
were silently dropped by the rasterizer.

``_classify_geometries`` sorts inputs into polygon / line / point buckets.
LinearRing (shapely type id 2) matched no bucket, so a non-empty ring came
back in all three empty buckets and ``rasterize`` returned an all-fill raster
with no error or warning. This covers:

- LinearRing is now treated as a line on both the fast path and the
  GeometryCollection slow path.
- A LinearRing burns the same boundary across the numpy, dask, cupy, and
  dask+cupy backends (the classifier is shared by all of them).
- Any remaining non-empty geometry that matches no bucket warns instead of
  vanishing, while None / empty inputs stay silent.
"""

import warnings

import numpy as np
import pytest

try:
    from shapely.geometry import (
        LinearRing, LineString, Point, GeometryCollection,
    )
    has_shapely = True
except ImportError:
    has_shapely = False

if has_shapely:
    from xrspatial.rasterize import rasterize, _classify_geometries

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)

try:
    import dask.array as da  # noqa: F401
    has_dask = True
except Exception:
    has_dask = False

try:
    import cupy  # noqa: F401
    from numba import cuda
    has_cuda = cuda.is_available()
except Exception:
    has_cuda = False

skip_no_dask = pytest.mark.skipif(not has_dask, reason="dask not installed")
skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason="CUDA / CuPy not available")


# A square ring whose four edges land on the border pixels of a 5x5 raster.
_RING = LinearRing([(0.5, 0.5), (0.5, 4.5), (4.5, 4.5), (4.5, 0.5), (0.5, 0.5)])

# Expected burn: the border is 1, the 3x3 interior stays at the fill value.
_EXPECTED = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
], dtype=float)


def _props(n, p=1):
    return np.ones((n, p), dtype=np.float64)


# ---------------------------------------------------------------------------
# Classifier: LinearRing routes to the line bucket
# ---------------------------------------------------------------------------

class TestClassifierLinearRing:
    def test_fast_path_linearring_is_line(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            (poly, line, point) = _classify_geometries([_RING], _props(1))
        assert len(line[0]) == 1          # routed to lines
        assert len(poly[0]) == 0
        assert len(point[0]) == 0
        assert w == []                    # no spurious warning

    def test_slow_path_linearring_in_collection(self):
        # A GeometryCollection forces the recursive slow path.
        gc = GeometryCollection([
            LinearRing([(0, 0), (0, 5), (5, 5), (0, 0)]),
            Point(1, 1),
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            (poly, line, point) = _classify_geometries([gc], _props(1))
        assert len(line[0]) == 1
        assert len(point[0]) == 1
        assert w == []


# ---------------------------------------------------------------------------
# Classifier: unsupported non-empty geometries warn, empties stay silent
# ---------------------------------------------------------------------------

class TestClassifierWarnings:
    def test_none_geometry_skipped_silently(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            (poly, line, point) = _classify_geometries(
                [None, Point(2.5, 2.5)], _props(2))
        # None reports as non-empty with type id -1; it must be skipped like
        # the slow path's `geom is None` guard, not warned about.
        assert len(point[0]) == 1
        assert w == []

    def test_empty_geometry_skipped_silently(self):
        empty = Point().buffer(0)  # empty polygon
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _classify_geometries([empty, Point(1, 1)], _props(2))
        assert w == []

    def test_warn_helper_resolves_type_id(self):
        # Fast path passes shapely type ids (ints); the helper resolves names.
        from xrspatial.rasterize import _warn_dropped_geometries
        with pytest.warns(UserWarning, match="GeometryCollection"):
            _warn_dropped_geometries([7])

    def test_warn_helper_accepts_geom_type_name(self):
        # Slow path passes geom_type names (strings) straight through.
        from xrspatial.rasterize import _warn_dropped_geometries
        with pytest.warns(UserWarning, match="Curve"):
            _warn_dropped_geometries(["Curve"])

    def test_warn_helper_unknown_id(self):
        # A type id with no name still warns rather than dropping silently.
        from xrspatial.rasterize import _warn_dropped_geometries
        with pytest.warns(UserWarning, match="type id 99"):
            _warn_dropped_geometries([99])

    def test_fast_path_warns_on_unsupported_type_id(self, monkeypatch):
        # Simulate a future/unknown non-empty shapely type id (e.g. 8) that
        # matches no bucket. The fast path must warn, not drop it silently.
        import sys
        rasterize_mod = sys.modules["xrspatial.rasterize"]
        shp = rasterize_mod._require_shapely()
        monkeypatch.setattr(shp, "get_type_id", lambda arr: np.array([8]))
        monkeypatch.setattr(shp, "is_empty", lambda arr: np.array([False]))
        with pytest.warns(UserWarning, match="type id 8"):
            _classify_geometries([Point(1, 1)], _props(1))


# ---------------------------------------------------------------------------
# End-to-end rasterize across backends
# ---------------------------------------------------------------------------

class TestRasterizeLinearRing:
    def test_numpy_burns_ring(self):
        result = rasterize([(_RING, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        np.testing.assert_array_equal(result.values, _EXPECTED)
        # Regression guard: not an all-fill raster.
        assert np.any(result.values == 1.0)

    def test_numpy_ring_matches_equivalent_line(self):
        # A LinearRing rasterizes the same as the closed LineString of its
        # coordinates.
        line = LineString(list(_RING.coords))
        ring_r = rasterize([(_RING, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        line_r = rasterize([(line, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0)
        np.testing.assert_array_equal(ring_r.values, line_r.values)

    @skip_no_dask
    def test_dask_burns_ring(self):
        result = rasterize([(_RING, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0, chunks=3)
        assert isinstance(result.data, da.Array)
        np.testing.assert_array_equal(
            np.asarray(result.data), _EXPECTED)

    @skip_no_cuda
    def test_cupy_burns_ring(self):
        result = rasterize([(_RING, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0, gpu=True)
        np.testing.assert_array_equal(result.data.get(), _EXPECTED)

    @skip_no_cuda
    @skip_no_dask
    def test_dask_cupy_burns_ring(self):
        result = rasterize([(_RING, 1.0)], width=5, height=5,
                           bounds=(0, 0, 5, 5), fill=0,
                           gpu=True, chunks=3)
        np.testing.assert_array_equal(
            np.asarray(result.data.map_blocks(lambda b: b.get())),
            _EXPECTED)
