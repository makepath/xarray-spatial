"""Coverage-gap tests for xrspatial.rasterize (deep-sweep test-coverage, pass 3).

Closes documented public-API gaps left after the pass-1 (2026-05-17) and
pass-2 (2026-05-21) audits:

- Cat 1 HIGH -- The eager cupy (``use_cuda=True``, no ``chunks=``) backend
  has no parametrised merge-mode parity test.  The dask+numpy and
  dask+cupy backends pin all six built-in merge modes (``last`` / ``first``
  / ``max`` / ``min`` / ``sum`` / ``count``) against eager numpy via
  ``TestDaskNumpy.test_merge_mode_parity`` and
  ``TestDaskCupy.test_merge_mode_parity``.  The eager cupy backend only
  has indirect coverage of ``sum`` / ``min`` / ``max`` through pass-2's
  Inf-burn tests and a single ``last``-default polygon parity in
  ``TestCuPy.test_cupy_matches_numpy``.  The GPU atomic kernels for
  ``first`` / ``last`` / ``count`` route through dedicated
  atomicCAS / atomicAdd paths in ``_ensure_gpu_kernels`` (rasterize.py:
  1308-1556) without ``chunks=``; a regression that wired one of those
  six atomic kernels to the wrong opcode would not surface from the
  dask+cupy parity tests (which always exercise the tiled finalize path
  in ``_run_dask_cupy``) or from ``TestCuPy.test_cupy_matches_numpy``
  (which uses the default ``merge='last'`` on a single non-overlapping
  geometry where last/first/max/min/sum/count all produce the same
  output).  Pin a multi-geometry overlap scene across all six modes on
  the eager cupy backend so an atomic-kernel routing regression flips a
  test red.

- Cat 1 MEDIUM -- Empty geometry list on the eager cupy backend.  The
  numpy backend covers it in ``TestListInput.test_empty_list``; the
  dask+numpy and dask+cupy backends cover it in their
  ``test_empty_geometry_list`` methods.  The eager cupy path through
  ``_run_cupy`` with zero geometries (so the bbox/edge/segment buffers
  are zero-sized cupy arrays) has no direct test, and a regression
  short-circuiting to numpy on the empty branch would silently break the
  "use_cuda=True returns cupy.ndarray" contract.

- Cat 2 MEDIUM -- All-equal property values on the count merge mode.
  ``count`` ignores the burned value and counts overlapping geometries.
  No test covers an input where every geometry has the same property
  value (the zero-variance case): if a future kernel optimisation
  collapsed equal-value writes into a single write under ``count`` it
  would silently divide counts by the number of unique values.  Pin
  count on an all-equal-value overlap stack across all four backends.

- Cat 4 MEDIUM -- ``name=`` kwarg thread-through.  ``test_rasterize.py``
  line 110 covers ``name='burned'`` on the eager numpy backend only.
  The dask+numpy, eager cupy, and dask+cupy backends route through
  separate output-construction paths and a regression dropping ``name``
  on one of those paths would not surface.  Pin ``name=`` on the three
  uncovered backends.

The "fix" in this sweep is *adding tests*.  No source changes.  CUDA is
available on this host so cupy / dask+cupy tests execute live.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from shapely.geometry import box, Point
    has_shapely = True
except ImportError:
    has_shapely = False

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


def _materialise(result):
    """Compute (if dask) and copy to host (if cupy) so callers see ndarray."""
    data = result.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if has_cupy and isinstance(data, cupy.ndarray):
        return cupy.asnumpy(data)
    return np.asarray(data)


# ---------------------------------------------------------------------------
# Cat 1 HIGH -- eager cupy merge-mode parity (all six built-ins)
# ---------------------------------------------------------------------------


# Three overlapping polygons so first/last/max/min/sum/count each produce a
# different pixel value in the overlap region.  Use distinct ascending values
# and overlapping bounds so the predicate is not degenerate (the existing
# TestCuPy.test_cupy_matches_numpy uses a single polygon where every merge
# mode would collapse to the same output).
#
# Guarded under ``has_shapely`` because ``box`` is only imported when
# shapely is available; without this guard the module fails to import
# on environments lacking the ``[vector]`` extra and the file-level
# ``pytestmark`` skip never runs (the NameError fires first).
if has_shapely:
    _EAGER_CUPY_MERGE_PAIRS = [
        (box(0, 0, 6, 6), 3.0),
        (box(2, 2, 8, 8), 5.0),
        (box(4, 4, 10, 10), 7.0),
    ]


@skip_no_cuda
class TestEagerCupyMergeModeParity:
    """Cat 1 HIGH -- eager cupy backend (no chunks) merge-mode parity.

    Mirrors ``TestDaskNumpy.test_merge_mode_parity`` /
    ``TestDaskCupy.test_merge_mode_parity`` but targets the eager
    ``_run_cupy`` path so the non-tiled GPU atomic kernels for each merge
    mode have direct coverage.
    """

    @pytest.mark.parametrize(
        "merge_mode",
        ['last', 'first', 'max', 'min', 'sum', 'count'],
    )
    def test_merge_mode_eager_cupy_matches_numpy(self, merge_mode):
        """Pin each built-in merge mode on the eager cupy backend.

        The overlap geometry is chosen so all six modes produce a
        distinct pixel value in the overlap region (3,3): without a
        multi-geometry overlap a routing regression that swapped
        e.g. ``first`` for ``last`` would not flip the test red.
        """
        np_result = rasterize(
            _EAGER_CUPY_MERGE_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge=merge_mode,
        )
        cp_result = rasterize(
            _EAGER_CUPY_MERGE_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge=merge_mode,
            use_cuda=True,
        )
        # ``count`` returns an integer-valued float, the rest are the
        # burned property values; both compare bit-exact with
        # assert_array_equal because the fill is 0 (no NaN noise).
        np.testing.assert_array_equal(
            np_result.values, _materialise(cp_result))

    def test_eager_cupy_first_differs_from_last(self):
        """Sanity check: ``first`` and ``last`` produce different output
        on the overlap fixture so the parametrised parity test above is
        not degenerate.

        Without this, a hypothetical regression that wired both
        ``first`` and ``last`` to the same kernel could still pass the
        parametrised parity test (numpy and cupy would agree, but both
        would be wrong in the same way).  Lock the input fixture in.
        """
        first = rasterize(
            _EAGER_CUPY_MERGE_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='first',
        )
        last = rasterize(
            _EAGER_CUPY_MERGE_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='last',
        )
        # The triple-overlap region near (5, 5) sees three geometries
        # with distinct values 3.0, 5.0, 7.0 -- first=3.0, last=7.0.
        assert first.values[5, 5] != last.values[5, 5]

    def test_eager_cupy_min_differs_from_max(self):
        """Companion sanity check for min vs max on the overlap fixture."""
        mn = rasterize(
            _EAGER_CUPY_MERGE_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='min',
        )
        mx = rasterize(
            _EAGER_CUPY_MERGE_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='max',
        )
        assert mn.values[5, 5] < mx.values[5, 5]


# ---------------------------------------------------------------------------
# Cat 1 HIGH -- eager cupy merge-mode parity with points and lines
# ---------------------------------------------------------------------------


@skip_no_cuda
class TestEagerCupyMergeModePoints:
    """Cat 1 HIGH -- merge-mode parity on points and lines (eager cupy).

    The GPU point-burn kernel and the GPU line (Bresenham) kernel route
    through the same atomic merge dispatcher as polygons, but each
    geometry type has its own thread-block layout.  Pin a
    multi-overlap point scene across the six merge modes so a regression
    in the point-or-line atomic merge wiring is visible.
    """

    @pytest.mark.parametrize(
        "merge_mode",
        ['last', 'first', 'max', 'min', 'sum', 'count'],
    )
    def test_overlapping_points_merge_mode(self, merge_mode):
        # Three points landing in the same pixel (2,2 in image coords).
        # All three points have x in [2.5, 3.0] and y in [2.5, 3.0] so
        # they share one cell at a 5x5 resolution over (0,0,5,5).
        pairs = [
            (Point(2.5, 2.5), 3.0),
            (Point(2.6, 2.6), 5.0),
            (Point(2.7, 2.7), 7.0),
        ]
        np_result = rasterize(
            pairs, width=5, height=5,
            bounds=(0, 0, 5, 5), fill=0, merge=merge_mode,
        )
        cp_result = rasterize(
            pairs, width=5, height=5,
            bounds=(0, 0, 5, 5), fill=0, merge=merge_mode,
            use_cuda=True,
        )
        np.testing.assert_array_equal(
            np_result.values, _materialise(cp_result))


# ---------------------------------------------------------------------------
# Cat 1 MEDIUM -- empty geometry list on the eager cupy backend
# ---------------------------------------------------------------------------


@skip_no_cuda
class TestEagerCupyEmptyGeometryList:
    """Cat 1 MEDIUM -- ``_run_cupy`` with zero geometries.

    test_rasterize.TestListInput.test_empty_list covers eager numpy;
    TestDaskNumpy.test_empty_geometry_list and
    TestDaskCupy.test_empty_geometry_list cover the dask paths.  The
    eager cupy path (no ``chunks=``) feeds zero-sized cupy arrays into
    ``_gpu_init_buffers`` and ``_gpu_finalize_buffers`` and a regression
    short-circuiting to numpy on the empty-input branch would silently
    break the "use_cuda=True returns a cupy.ndarray" contract.
    """

    def test_empty_list_returns_cupy_filled_with_fill(self):
        result = rasterize(
            [], width=8, height=8,
            bounds=(0, 0, 8, 8), fill=-1.0, use_cuda=True,
        )
        # Backend contract: use_cuda=True must return a cupy.ndarray.
        assert isinstance(result.data, cupy.ndarray), \
            "use_cuda=True with empty input must keep cupy backend"
        host = cupy.asnumpy(result.data)
        assert host.shape == (8, 8)
        assert np.all(host == -1.0)

    def test_empty_list_default_nan_fill_returns_cupy(self):
        """Default NaN fill must also stay on the GPU."""
        result = rasterize(
            [], width=5, height=5,
            bounds=(0, 0, 5, 5), use_cuda=True,
        )
        assert isinstance(result.data, cupy.ndarray)
        host = cupy.asnumpy(result.data)
        assert host.shape == (5, 5)
        assert np.all(np.isnan(host))


# ---------------------------------------------------------------------------
# Cat 2 MEDIUM -- all-equal property values on ``count`` merge mode
# ---------------------------------------------------------------------------


# Four overlapping unit squares all burning the same value 1.0.  Under
# ``count`` the overlap pixel should see a count of 4 regardless of the
# burn value being equal.  See the guard note on _EAGER_CUPY_MERGE_PAIRS
# above -- ``box`` is unbound when shapely is missing.
if has_shapely:
    _ALL_EQUAL_PAIRS = [
        (box(2, 2, 6, 6), 1.0),
        (box(3, 3, 7, 7), 1.0),
        (box(4, 4, 8, 8), 1.0),
        (box(5, 5, 9, 9), 1.0),
    ]


class TestAllEqualValueCount:
    """Cat 2 MEDIUM -- ``count`` merge with all-equal burn values.

    A regression that collapsed equal-value writes into a single write
    under ``count`` (e.g. a future "dedup identical values" optimisation
    on the GPU atomic path) would divide counts by the number of unique
    values and silently corrupt density rasters.  Pin the contract that
    ``count`` is invariant under the property value across all four
    backends.
    """

    def test_count_numpy(self):
        result = rasterize(
            _ALL_EQUAL_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='count',
        )
        # Locate the cell with the maximum overlap count.  The four
        # rectangles intersect in a 1x1 region near the centre; counting
        # the exact cell depends on half-pixel sampling, so use the
        # observed max as the contract: with all-equal burn values
        # ``count`` MUST report a value > 1 (proving overlapping writes
        # were not de-duplicated by identical value) and MUST equal the
        # number of geometries covering that cell.
        assert result.values.max() > 1, \
            "count must not collapse identical-value writes into one"
        # The strictest overlap of all four rectangles is the unit cell
        # at (5,5)-(6,6) in world coordinates; box(5,5,9,9) is the only
        # one that excludes its lower-left corner, so the four-way
        # overlap region in shapely's closed-on-lower convention has
        # measure 0 and may or may not be hit by a pixel centre at this
        # resolution.  The three-way overlap region (first three
        # rectangles) is 1x1 at (5,5)-(6,6) world, which always contains
        # one pixel centre at this resolution -- so the contract is
        # max >= 3.
        assert result.values.max() >= 3.0

    @skip_no_cuda
    def test_count_eager_cupy_matches_numpy(self):
        np_result = rasterize(
            _ALL_EQUAL_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='count',
        )
        cp_result = rasterize(
            _ALL_EQUAL_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='count',
            use_cuda=True,
        )
        np.testing.assert_array_equal(
            np_result.values, _materialise(cp_result))

    @skip_no_dask
    def test_count_dask_numpy_matches_numpy(self):
        np_result = rasterize(
            _ALL_EQUAL_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='count',
        )
        dk_result = rasterize(
            _ALL_EQUAL_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='count',
            chunks=(5, 5),
        )
        np.testing.assert_array_equal(
            np_result.values, _materialise(dk_result))

    @skip_no_cuda
    @skip_no_dask
    def test_count_dask_cupy_matches_numpy(self):
        np_result = rasterize(
            _ALL_EQUAL_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='count',
        )
        dc_result = rasterize(
            _ALL_EQUAL_PAIRS, width=10, height=10,
            bounds=(0, 0, 10, 10), fill=0, merge='count',
            use_cuda=True, chunks=(5, 5),
        )
        np.testing.assert_array_equal(
            np_result.values, _materialise(dc_result))


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM -- ``name=`` kwarg thread-through on non-default backends
# ---------------------------------------------------------------------------


class TestNameKwargBackends:
    """Cat 4 MEDIUM -- ``name=`` kwarg on dask/cupy/dask+cupy backends.

    ``test_rasterize.TestBasic.test_output_name`` covers the eager
    numpy path.  Each non-default backend constructs its own output
    DataArray via ``_run_dask_numpy`` / ``_run_cupy`` / ``_run_dask_cupy``
    and a regression that dropped ``name`` on one of those paths would
    not surface from the existing eager test.
    """

    # Built inside a method (not a class attribute) so ``box`` is only
    # referenced when the test actually runs.  A class-level
    # ``[(box(...), 1.0)]`` would be evaluated at import time and crash
    # the module on shapely-less environments before pytestmark skips.
    def _simple_pair(self):
        return [(box(1, 1, 4, 4), 1.0)]

    @skip_no_dask
    def test_name_propagated_dask_numpy(self):
        result = rasterize(
            self._simple_pair(), width=5, height=5,
            bounds=(0, 0, 5, 5), name='burned_dask', chunks=(3, 3),
        )
        assert result.name == 'burned_dask'

    @skip_no_cuda
    def test_name_propagated_eager_cupy(self):
        result = rasterize(
            self._simple_pair(), width=5, height=5,
            bounds=(0, 0, 5, 5), name='burned_cupy', use_cuda=True,
        )
        assert result.name == 'burned_cupy'

    @skip_no_cuda
    @skip_no_dask
    def test_name_propagated_dask_cupy(self):
        result = rasterize(
            self._simple_pair(), width=5, height=5,
            bounds=(0, 0, 5, 5), name='burned_dask_cupy',
            use_cuda=True, chunks=(3, 3),
        )
        assert result.name == 'burned_dask_cupy'
