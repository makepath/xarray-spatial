"""Regression tests for issue #2167.

GPU point and line rasterize kernels previously did non-atomic
read-modify-write on the per-pixel state.  When multiple threads
landed on the same pixel (overlapping points, crossing line
segments, duplicate segments, shared polygon boundaries with
``all_touched=True``) the writes raced and the cupy backend could
return a value that disagreed with the numpy backend, and varied
from run to run.

These tests pin down deterministic, numpy-matching behaviour for
the six built-in aggregators (``last``, ``first``, ``sum``,
``count``, ``min``, ``max``) on overlapping geometries.

When cupy / CUDA are not available the whole module skips.
"""
import numpy as np
import pytest

try:
    from shapely.geometry import LineString, Point, box
    has_shapely = True
except ImportError:
    has_shapely = False

if has_shapely:
    from xrspatial.rasterize import rasterize

try:
    import cupy  # noqa: F401
    has_cupy = True
except ImportError:
    has_cupy = False

try:
    from numba import cuda
    has_cuda = has_cupy and cuda.is_available()
except Exception:
    has_cuda = False


pytestmark = [
    pytest.mark.skipif(not has_shapely, reason="shapely not installed"),
    pytest.mark.skipif(not has_cuda, reason="CUDA / CuPy not available"),
]


BOUNDS = (0.0, 0.0, 10.0, 10.0)
HEIGHT = 10
WIDTH = 10
MERGES = ('last', 'first', 'sum', 'count', 'min', 'max')


def _as_numpy(arr):
    """Return a plain numpy ndarray regardless of backend."""
    data = arr.data
    if hasattr(data, 'get'):
        return data.get()
    return np.asarray(data)


def _run(geom_list, merge, use_cuda, all_touched=False):
    return rasterize(
        geom_list, width=WIDTH, height=HEIGHT, bounds=BOUNDS,
        merge=merge, use_cuda=use_cuda, fill=np.nan,
        all_touched=all_touched,
    )


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def _coincident_points():
    """Three points that land on the same pixel, plus one offset point."""
    # Pixel grid spans 0-10 in 10 steps -> 1 unit per pixel.  All three
    # of these land in the pixel centred at (5.5, 5.5).  The fourth one
    # is well away so we can confirm non-overlap is unaffected.
    return [
        (Point(5.5, 5.5), 1.0),
        (Point(5.7, 5.3), 2.0),
        (Point(5.2, 5.8), 4.0),
        (Point(1.5, 1.5), 9.0),
    ]


def _crossing_lines():
    """Two line segments that cross at one or more pixels."""
    return [
        (LineString([(0.5, 0.5), (9.5, 9.5)]), 1.0),
        (LineString([(0.5, 9.5), (9.5, 0.5)]), 2.0),
        (LineString([(2.5, 5.5), (7.5, 5.5)]), 4.0),
    ]


def _duplicate_segments():
    """Two identical line segments -- every pixel they cover overlaps."""
    coords = [(1.5, 1.5), (8.5, 8.5)]
    return [
        (LineString(coords), 1.0),
        (LineString(coords), 8.0),
    ]


SCENARIOS = {
    'coincident_points': _coincident_points,
    'crossing_lines': _crossing_lines,
    'duplicate_segments': _duplicate_segments,
}


# ---------------------------------------------------------------------------
# Cross-backend parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scenario', list(SCENARIOS))
@pytest.mark.parametrize('merge', MERGES)
def test_cupy_matches_numpy_on_overlap(scenario, merge):
    geoms = SCENARIOS[scenario]()
    expected = _as_numpy(_run(geoms, merge, use_cuda=False))
    actual = _as_numpy(_run(geoms, merge, use_cuda=True))

    # Pixels touched by no geometry should be ``fill`` (NaN here) on
    # both backends.  np.testing.assert_array_equal treats NaN==NaN
    # when equal_nan=True is requested via assert_array_equal in
    # recent NumPy; use the explicit form for portability.
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=0, equal_nan=True,
        err_msg=(
            f"cupy backend disagrees with numpy for scenario "
            f"{scenario!r} and merge {merge!r}.\n"
            f"cupy:\n{actual}\nnumpy:\n{expected}"
        ),
    )


# ---------------------------------------------------------------------------
# Determinism: same input must give identical output across runs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scenario', list(SCENARIOS))
@pytest.mark.parametrize('merge', MERGES)
def test_cupy_is_deterministic_across_runs(scenario, merge):
    geoms = SCENARIOS[scenario]()
    first = _as_numpy(_run(geoms, merge, use_cuda=True))
    # Six repeats is enough to surface a thread-interleaving race
    # without making the test slow.
    for _ in range(5):
        again = _as_numpy(_run(geoms, merge, use_cuda=True))
        np.testing.assert_allclose(
            again, first, rtol=0, atol=0, equal_nan=True,
            err_msg=(
                f"cupy backend produced different results across runs "
                f"for scenario {scenario!r} and merge {merge!r}.\n"
                f"first:\n{first}\nagain:\n{again}"
            ),
        )


# ---------------------------------------------------------------------------
# Spot-check that sum on coincident points actually accumulates.  Without
# the atomic-add fix, this test would either return one of the input
# values or NaN, never the true sum.
# ---------------------------------------------------------------------------

def test_sum_of_coincident_points_equals_total():
    geoms = _coincident_points()
    # The first three points all land in the same pixel; sum should be 7.
    result = _as_numpy(_run(geoms, 'sum', use_cuda=True))
    # Find the pixel with the largest accumulated value.
    finite = result[np.isfinite(result)]
    assert finite.size > 0
    # Three coincident points (1 + 2 + 4) plus an isolated point of 9.
    assert 7.0 in finite, (
        f"expected the three-point pixel to sum to 7.0; got {finite}"
    )


def test_count_of_coincident_points_equals_three():
    geoms = _coincident_points()
    result = _as_numpy(_run(geoms, 'count', use_cuda=True))
    finite = result[np.isfinite(result)]
    assert 3.0 in finite, (
        f"expected the three-point pixel to count 3; got {finite}"
    )


# ---------------------------------------------------------------------------
# all_touched=True polygon boundaries are the polygon analogue of the
# line-overlap case: two polygons sharing a boundary write the same
# pixels twice via the Bresenham boundary pass, on top of the scanline
# fill.  Confirm cupy still matches numpy across all six aggregators.
# ---------------------------------------------------------------------------

def _shared_boundary_polygons():
    """Two rectangles that share an edge."""
    return [
        (box(1.0, 1.0, 5.0, 5.0), 1.0),
        (box(5.0, 1.0, 9.0, 5.0), 2.0),
    ]


@pytest.mark.parametrize('merge', MERGES)
def test_cupy_matches_numpy_all_touched_shared_boundary(merge):
    geoms = _shared_boundary_polygons()
    expected = _as_numpy(_run(geoms, merge, use_cuda=False, all_touched=True))
    actual = _as_numpy(_run(geoms, merge, use_cuda=True, all_touched=True))
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=0, equal_nan=True,
        err_msg=(
            f"cupy backend disagrees with numpy for shared-boundary "
            f"polygons with all_touched=True and merge {merge!r}.\n"
            f"cupy:\n{actual}\nnumpy:\n{expected}"
        ),
    )


@pytest.mark.parametrize('merge', MERGES)
def test_cupy_deterministic_all_touched_shared_boundary(merge):
    geoms = _shared_boundary_polygons()
    first = _as_numpy(_run(geoms, merge, use_cuda=True, all_touched=True))
    for _ in range(5):
        again = _as_numpy(_run(geoms, merge, use_cuda=True, all_touched=True))
        np.testing.assert_allclose(
            again, first, rtol=0, atol=0, equal_nan=True,
            err_msg=(
                f"cupy backend produced different results across runs "
                f"for shared-boundary polygons with all_touched=True and "
                f"merge {merge!r}."
            ),
        )
