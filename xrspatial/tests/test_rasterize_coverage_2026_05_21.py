"""Coverage-gap tests for xrspatial.rasterize (deep-sweep test-coverage, pass 2).

Closes the documented public-API gaps left after the pass-1 audit on
2026-05-17:

- Cat 2 HIGH  -- Inf burn-in values were never tested as a property /
  geometry value on any backend.  rasterize() builds a numeric output by
  burning the per-geometry value into the raster, so passing
  +Inf / -Inf / Inf+(-Inf) flows through every merge function: sum should
  propagate the infinity, min/max should observe inf-vs-finite ordering,
  Inf+(-Inf) under sum should produce NaN by IEEE arithmetic.  Pin those
  behaviours on all four backends so a regression that special-cases
  non-finite at the kernel boundary (e.g. early-exit on isfinite) ships
  visibly.
- Cat 2 HIGH  -- NaN burn-in values were never tested.  A geometry with a
  NaN property is documented to write NaN into the covered pixels (and
  any subsequent merge picks up NaN per IEEE rules).  Pin the
  cross-backend agreement so a future kernel optimisation that "skips
  NaN" silently drops coverage instead of writing the documented value.
- Cat 1 MEDIUM -- Nested GeometryCollection: rasterize.py:1995 documents
  "GeometryCollection -- recursively unpacked", and
  ``_classify_geometries`` implements the recursion through the slow
  path's ``_classify_one(geom, ...)`` callback.  The pass-1 coverage file
  only tests a single-level GC; a GC nested inside another GC has no
  direct test.  All four backends route through this code path because
  the GC fan-out happens before backend dispatch.
- Cat 1 MEDIUM -- ``columns=`` (multi-column properties) on the cupy and
  dask+cupy backends.  test_rasterize.TestMultiColumn covers numpy and
  dask+numpy parity but the GPU paths -- which thread the (N, P) props
  array through the GPU init / scanline kernels and the per-tile dask
  graph -- have no direct coverage.
- Cat 3 LOW (documented, not fixed) -- non-square cell size with
  resolution=(rx, ry) and rx != ry already has indirect coverage via
  TestResolutionParameter.test_tuple_resolution_branch but no test pins
  the rectangular-pixel parity across backends.  Pin a single eager
  parity check so a regression that swaps rx/ry between code paths is
  visible.

The "fix" in this sweep is *adding tests*.  No source changes.  CUDA is
available on this host so cupy / dask+cupy tests execute live.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from shapely.geometry import (
        box, GeometryCollection, LineString, MultiPoint, Point,
    )
    has_shapely = True
except ImportError:
    has_shapely = False

try:
    import geopandas as gpd
    has_geopandas = True
except ImportError:
    has_geopandas = False

if has_shapely:
    from xrspatial.rasterize import rasterize

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)

try:
    import cupy
    has_cupy = True
except ImportError:
    has_cupy = False

try:
    import dask  # noqa: F401  (availability probe only)
    has_dask = True
except ImportError:
    has_dask = False

try:
    from numba import cuda
    has_cuda = has_cupy and cuda.is_available()
except Exception:
    has_cuda = False

skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason="CUDA / CuPy not available")
skip_no_dask = pytest.mark.skipif(
    not has_dask, reason="dask not installed")
skip_no_geopandas = pytest.mark.skipif(
    not has_geopandas, reason="geopandas not installed")


def _materialise(result):
    """Compute (if dask) and copy to host (if cupy) so callers see ndarray."""
    data = result.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if has_cupy and isinstance(data, cupy.ndarray):
        return cupy.asnumpy(data)
    return np.asarray(data)


_BACKEND_KWARGS = {
    'numpy': {},
    'cupy': {'use_cuda': True},
    'dask_numpy': {'chunks': (5, 5)},
    'dask_cupy': {'use_cuda': True, 'chunks': (5, 5)},
}


def _backend_param(name):
    """Wrap a backend kwargs entry in the right pytest skip marker."""
    if name == 'cupy':
        return pytest.param('cupy', _BACKEND_KWARGS['cupy'],
                            marks=skip_no_cuda, id='cupy')
    if name == 'dask_numpy':
        return pytest.param('dask_numpy', _BACKEND_KWARGS['dask_numpy'],
                            marks=skip_no_dask, id='dask_numpy')
    if name == 'dask_cupy':
        return pytest.param('dask_cupy', _BACKEND_KWARGS['dask_cupy'],
                            marks=[skip_no_cuda, skip_no_dask],
                            id='dask_cupy')
    return pytest.param('numpy', _BACKEND_KWARGS['numpy'], id='numpy')


ALL_BACKENDS = [_backend_param(name) for name in _BACKEND_KWARGS]


# ---------------------------------------------------------------------------
# Cat 2 HIGH -- Inf burn-in values
# ---------------------------------------------------------------------------


class TestInfBurnValues:
    """rasterize() burns the geometry value into covered pixels.

    Passing +Inf / -Inf as the burn value flows through every merge
    function.  Sum should propagate Inf; min and max should obey IEEE
    ordering against finite values; Inf+(-Inf) under sum should yield
    NaN.  No test exists today for any of these on any backend; a
    regression that gates writes on ``isfinite`` at the kernel boundary
    would silently drop the pixels.
    """

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_positive_inf_burn_default(self, backend_name, kw):
        """+Inf burn fills the covered region with +Inf under the default
        ``last`` merge."""
        r = rasterize([(box(0, 0, 10, 5), np.inf)],
                      width=10, height=5, bounds=(0, 0, 10, 5),
                      fill=0, **kw)
        data = _materialise(r)
        assert np.all(np.isposinf(data))

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_negative_inf_burn_default(self, backend_name, kw):
        """-Inf burn fills covered region with -Inf under the default merge."""
        r = rasterize([(box(0, 0, 10, 5), -np.inf)],
                      width=10, height=5, bounds=(0, 0, 10, 5),
                      fill=0, **kw)
        data = _materialise(r)
        assert np.all(np.isneginf(data))

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_inf_plus_finite_sum_remains_inf(self, backend_name, kw):
        """Inf + finite under ``sum`` should remain Inf, not collapse to
        the finite value.  IEEE arithmetic guarantees Inf + 1.0 == Inf."""
        r = rasterize(
            [(box(0, 0, 10, 5), np.inf), (box(0, 0, 10, 5), 1.0)],
            width=10, height=5, bounds=(0, 0, 10, 5),
            fill=0, merge='sum', **kw)
        data = _materialise(r)
        assert np.all(np.isposinf(data))

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_inf_plus_neg_inf_sum_is_nan(self, backend_name, kw):
        """Inf + (-Inf) under ``sum`` produces NaN by IEEE arithmetic.
        Pinning the NaN result makes a future kernel that special-cases
        infinity (e.g. saturating arithmetic) surface in CI."""
        r = rasterize(
            [(box(0, 0, 10, 5), np.inf), (box(0, 0, 10, 5), -np.inf)],
            width=10, height=5, bounds=(0, 0, 10, 5),
            fill=0, merge='sum', **kw)
        data = _materialise(r)
        assert np.all(np.isnan(data))

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_min_inf_vs_finite_picks_finite(self, backend_name, kw):
        """min(Inf, 1.0) == 1.0; the finite value wins."""
        r = rasterize(
            [(box(0, 0, 10, 5), np.inf), (box(0, 0, 10, 5), 1.0)],
            width=10, height=5, bounds=(0, 0, 10, 5),
            fill=0, merge='min', **kw)
        data = _materialise(r)
        np.testing.assert_array_equal(data, np.ones_like(data))

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_max_neg_inf_vs_finite_picks_finite(self, backend_name, kw):
        """max(-Inf, 1.0) == 1.0; the finite value wins."""
        r = rasterize(
            [(box(0, 0, 10, 5), -np.inf), (box(0, 0, 10, 5), 1.0)],
            width=10, height=5, bounds=(0, 0, 10, 5),
            fill=0, merge='max', **kw)
        data = _materialise(r)
        np.testing.assert_array_equal(data, np.ones_like(data))

    def test_inf_bounds_rejected(self):
        """+Inf bounds are rejected with the same ValueError as NaN bounds.
        The eager NaN-bound rejection was tested by
        test_explicit_nan_bounds_rejected; this pin extends that to ``inf``
        so a future refactor that switches the check from ``isfinite`` to
        ``not isnan`` (which would accept inf) surfaces in CI.

        Match is anchored on the bounds error prefix ``Invalid bounds:``
        so a future refactor that adds a different "must be finite" check
        (e.g. on resolution) earlier in the call cannot accidentally
        satisfy this assertion via the wrong code path."""
        with pytest.raises(ValueError,
                           match=r"Invalid bounds:.*must be finite"):
            rasterize([(box(0, 0, 1, 1), 1.0)], width=2, height=2,
                      bounds=(0, 0, float('inf'), 1))
        with pytest.raises(ValueError,
                           match=r"Invalid bounds:.*must be finite"):
            rasterize([(box(0, 0, 1, 1), 1.0)], width=2, height=2,
                      bounds=(0, 0, 1, -float('inf')))


# ---------------------------------------------------------------------------
# Cat 2 HIGH -- NaN burn-in values
# ---------------------------------------------------------------------------


class TestNaNBurnValues:
    """A geometry value of NaN writes NaN into the covered pixels.

    No test pins this behaviour on any backend.  A kernel optimisation
    that drops NaN writes (e.g. ``if isnan(val) continue``) would silently
    leave the fill value in covered cells, which is a different
    observable than emitting NaN there.

    Issue #2255 aligned the cross-backend contract on strict NaN
    propagation for ``max`` / ``min``: any NaN burn poisons the output
    pixel regardless of order.  See ``TestNaNPropagationAcrossBackends``
    below for the cross-backend pins.
    """

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_nan_burn_polygon(self, backend_name, kw):
        """A NaN-valued polygon burn produces NaN in every covered pixel,
        not the fill sentinel."""
        r = rasterize([(box(0, 0, 5, 5), np.nan),
                       (box(5, 0, 10, 5), 1.0)],
                      width=10, height=5, bounds=(0, 0, 10, 5),
                      fill=0, **kw)
        data = _materialise(r)
        # Left half (NaN polygon) is NaN, not 0 (the fill).
        assert np.all(np.isnan(data[:, :5])), (
            f"{backend_name}: left half should be NaN, got {data[:, :5]}")
        # Right half (finite polygon) is 1.0, not NaN.
        np.testing.assert_array_equal(data[:, 5:], np.ones((5, 5)))

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_nan_burn_point(self, backend_name, kw):
        """NaN burn through the point kernel produces NaN at the point's
        cell.  Pixel grid spans 10x5 at unit cells: Point(2.5, 2.5) lands
        at row=2 col=2 (y descends from ymax=5)."""
        r = rasterize([(Point(2.5, 2.5), np.nan)],
                      width=10, height=5, bounds=(0, 0, 10, 5),
                      fill=0.0, **kw)
        data = _materialise(r)
        assert np.isnan(data[2, 2])
        # All other pixels keep the fill.
        mask = np.ones_like(data, dtype=bool)
        mask[2, 2] = False
        assert np.all(data[mask] == 0)

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_nan_burn_line(self, backend_name, kw):
        """NaN burn through the line Bresenham kernel produces NaN along
        the rasterized line."""
        line = LineString([(0.5, 2.5), (9.5, 2.5)])
        r = rasterize([(line, np.nan)],
                      width=10, height=5, bounds=(0, 0, 10, 5),
                      fill=0.0, **kw)
        data = _materialise(r)
        # The line falls on the central row.  Find which row by counting
        # NaN per row -- this avoids backend-specific Bresenham rounding.
        nan_rows = [int(np.isnan(data[r_idx]).sum()) for r_idx in range(5)]
        assert max(nan_rows) >= 9, (
            f"{backend_name}: expected at least 9 NaN cells in some row, "
            f"got per-row NaN counts {nan_rows}"
        )


# ---------------------------------------------------------------------------
# Cat 1 MEDIUM -- Nested GeometryCollection
# ---------------------------------------------------------------------------


class TestNestedGeometryCollection:
    """A GeometryCollection containing another GeometryCollection.

    rasterize.py:1995 documents: "GeometryCollection -- recursively
    unpacked".  ``_classify_geometries`` implements that recursion through
    a closure that walks ``sub.geoms`` for any ``GeometryCollection``
    child.  The pass-1 coverage file only tests a flat GC (Polygon +
    Point + Line inside one GC); a regression that limited the recursion
    depth to 1 would silently drop deeper geometries.
    """

    @staticmethod
    def _nested():
        """Outer GC = [ inner GC([box, Point]), Point ]."""
        inner = GeometryCollection([box(0, 0, 4, 4), Point(7.5, 7.5)])
        return GeometryCollection([inner, Point(2.5, 2.5)])

    @staticmethod
    def _flat_equivalent():
        """Flattened list with identical pixel coverage."""
        return [box(0, 0, 4, 4), Point(7.5, 7.5), Point(2.5, 2.5)]

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_nested_gc_matches_flat(self, backend_name, kw):
        nested = self._nested()
        flat = self._flat_equivalent()
        nested_r = rasterize([(nested, 1.0)], width=10, height=10,
                             bounds=(0, 0, 10, 10), fill=0, **kw)
        flat_r = rasterize([(g, 1.0) for g in flat], width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0, **kw)
        np.testing.assert_array_equal(
            _materialise(nested_r), _materialise(flat_r),
            err_msg=f"backend {backend_name}: nested GC did not match flat")

    @pytest.mark.parametrize('backend_name,kw', [
        ALL_BACKENDS[0],  # numpy
        ALL_BACKENDS[2],  # dask_numpy
    ])
    def test_deeply_nested_gc(self, backend_name, kw):
        """Three levels deep: GC(GC(GC([poly]))).

        Exercised on numpy and dask+numpy.  The dask run additionally
        checks that the recursive GC pre-classification survives the
        per-tile graph builder, not just the eager path.
        """
        l3 = GeometryCollection([box(2, 2, 8, 8)])
        l2 = GeometryCollection([l3])
        l1 = GeometryCollection([l2])
        r = rasterize([(l1, 5.0)], width=10, height=10,
                      bounds=(0, 0, 10, 10), fill=0, **kw)
        # The inner polygon (box 2..8 in a 10x10 raster) writes 36 pixels
        # of 5.0 -- pin the count rather than a per-pixel mask so the
        # test is robust to scanline tie-breaks.
        burned = (_materialise(r) == 5.0)
        assert burned.sum() == 36, (
            f"{backend_name} deeply nested GC: expected 36 burned pixels, "
            f"got {burned.sum()}"
        )

    @skip_no_geopandas
    def test_nested_gc_in_geodataframe(self):
        """GeometryCollections inside a GeoDataFrame also unpack."""
        nested = self._nested()
        gdf = gpd.GeoDataFrame({'value': [3.0]}, geometry=[nested])
        r = rasterize(gdf, width=10, height=10,
                      bounds=(0, 0, 10, 10), fill=0, column='value')
        data = _materialise(r)
        # Polygon covers the 4x4 SW block (row 6..9, col 0..3 in standard
        # y-descending image orientation).  The two points appear at
        # (2.5, 2.5) -> row 7 col 2 (covered by the polygon already) and
        # (7.5, 7.5) -> row 2 col 7.
        assert data[7, 2] == 3.0  # inside polygon
        assert data[2, 7] == 3.0  # standalone point burns


# ---------------------------------------------------------------------------
# Cat 1 MEDIUM -- columns= on cupy and dask+cupy
# ---------------------------------------------------------------------------


@skip_no_geopandas
class TestMultiColumnGPU:
    """``columns=`` parity on the cupy and dask+cupy backends.

    TestMultiColumn in test_rasterize.py covers numpy + dask+numpy parity
    but the GPU paths thread the (N, P) props array through dedicated GPU
    kernels and per-tile dask graphs.  A regression on the GPU
    multi-column wiring would not surface from any existing test.
    """

    @staticmethod
    def _fixture():
        return gpd.GeoDataFrame({
            'num': [6.0, 12.0],
            'den': [2.0, 3.0],
            'geometry': [box(0, 0, 5, 5), box(5, 0, 10, 5)],
        })

    @skip_no_cuda
    def test_multi_column_sum_cupy_matches_numpy(self):
        gdf = self._fixture()
        np_r = rasterize(gdf, columns=['num', 'den'], merge='sum',
                         width=10, height=5, bounds=(0, 0, 10, 5),
                         fill=0)
        cp_r = rasterize(gdf, columns=['num', 'den'], merge='sum',
                         width=10, height=5, bounds=(0, 0, 10, 5),
                         fill=0, use_cuda=True)
        np.testing.assert_array_equal(np_r.values, _materialise(cp_r))

    @skip_no_cuda
    @skip_no_dask
    def test_multi_column_sum_dask_cupy_matches_numpy(self):
        gdf = self._fixture()
        np_r = rasterize(gdf, columns=['num', 'den'], merge='sum',
                         width=10, height=5, bounds=(0, 0, 10, 5),
                         fill=0)
        dc_r = rasterize(gdf, columns=['num', 'den'], merge='sum',
                         width=10, height=5, bounds=(0, 0, 10, 5),
                         fill=0, use_cuda=True, chunks=(3, 3))
        np.testing.assert_array_equal(np_r.values, _materialise(dc_r))

    @skip_no_cuda
    def test_multi_column_props_count_cupy(self):
        """Multi-column with merge='count' uses props[0]: pin the (N, P)
        array shape survives the GPU init.  count=2 because two polygons
        share no pixels but each writes once -- so per-pixel count is 1
        everywhere geometries are present."""
        gdf = self._fixture()
        cp_r = rasterize(gdf, columns=['num', 'den'], merge='count',
                         width=10, height=5, bounds=(0, 0, 10, 5),
                         fill=0, use_cuda=True)
        data = _materialise(cp_r)
        # Every covered pixel has count==1; uncovered pixels are 0.
        assert (data == 1).sum() == 50  # full 10x5 covered by union

    @skip_no_cuda
    def test_multi_column_three_columns_cupy(self):
        """A third column exercises the props array shape >2 on GPU.
        Built-in merges read props[0] only; the loop that copies all P
        columns into the per-pixel state still has to run."""
        gdf = gpd.GeoDataFrame({
            'a': [1.0, 4.0],
            'b': [2.0, 5.0],
            'c': [3.0, 6.0],
            'geometry': [box(0, 0, 5, 5), box(5, 0, 10, 5)],
        })
        np_r = rasterize(gdf, columns=['a', 'b', 'c'], merge='sum',
                         width=10, height=5, bounds=(0, 0, 10, 5),
                         fill=0)
        cp_r = rasterize(gdf, columns=['a', 'b', 'c'], merge='sum',
                         width=10, height=5, bounds=(0, 0, 10, 5),
                         fill=0, use_cuda=True)
        np.testing.assert_array_equal(np_r.values, _materialise(cp_r))


# ---------------------------------------------------------------------------
# Cat 3 LOW -- rectangular pixel parity (non-square cells)
# ---------------------------------------------------------------------------


class TestRectangularPixels:
    """resolution=(rx, ry) with rx != ry produces non-square cells.

    The pass-1 coverage file exercises the tuple branch through
    test_tuple_resolution_branch but does not assert per-backend parity.
    Pin a single rectangular-pixel parity check so a regression that
    swaps rx/ry in the height/width derivation surfaces in CI.
    """

    @pytest.mark.parametrize('backend_name,kw', ALL_BACKENDS)
    def test_rectangular_pixels_polygon_parity(self, backend_name, kw):
        """resolution=(2, 1) -> width=5, height=10 over (0,0)-(10,10)."""
        geom = box(2, 2, 8, 8)
        np_r = rasterize([(geom, 3.0)],
                         resolution=(2.0, 1.0),
                         bounds=(0, 0, 10, 10), fill=0)
        backend_r = rasterize([(geom, 3.0)],
                              resolution=(2.0, 1.0),
                              bounds=(0, 0, 10, 10), fill=0, **kw)
        assert backend_r.shape == (10, 5)
        np.testing.assert_array_equal(
            np_r.values, _materialise(backend_r),
            err_msg=f"{backend_name} disagreed on (rx=2, ry=1) cell size")

    def test_rectangular_pixels_attrs_res(self):
        """The resolved cell size lands on the output if a like attrs
        dict is supplied with res; without like, no res attr is set --
        this test pins that contract so callers know which path emits
        the metadata."""
        # No like -> no res attr.
        r = rasterize([(box(2, 2, 8, 8), 3.0)],
                      resolution=(2.0, 1.0),
                      bounds=(0, 0, 10, 10), fill=0)
        assert 'res' not in r.attrs
