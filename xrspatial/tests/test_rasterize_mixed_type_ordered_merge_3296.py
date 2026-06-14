"""Regression tests for issue #3296.

``merge='first'`` / ``merge='last'`` resolve overlapping pixels by global
input index across geometry types (issue #2064).  The CPU backends gate
writes with the ``_should_write_first`` / ``_should_write_last``
predicates inside the burn kernels; the GPU backends use a two-pass
scheme instead: pass 1 resolves the winning input index with
``cuda.atomic.min`` / ``cuda.atomic.max`` on the ``order`` array across
the scanline, line, and point kernels, and pass 2 stamps the winner's
value.

Before these tests, the cross-type input-order semantics were pinned on
numpy and dask+numpy only (``TestMergeOrderAcrossTypes`` in
test_rasterize.py).  The GPU first/last tests all used a single geometry
type per scene, so a regression in the GPU cross-type order resolution
(per-kernel order buffers, or a pass-2 kernel firing before all pass-1
kernels) would have passed the whole suite.

When cupy / CUDA are not available the GPU tests skip.
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

try:
    import dask  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False


pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed")

skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason="CUDA / CuPy not available")
skip_no_dask = pytest.mark.skipif(
    not has_dask, reason="dask not installed")


def _as_numpy(result):
    """Materialise a rasterize result to a plain numpy array."""
    data = result.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'get'):
        data = data.get()
    return np.asarray(data)


def _mixed_three_types():
    """Point, line, and polygon competing for pixels, in that input order.

    Same scene as ``TestMergeOrderAcrossTypes`` in test_rasterize.py:
    on a 3x3 grid over (0, 0, 3, 3) the point covers pixel (1, 1), the
    horizontal line covers row 1, and the polygon covers every pixel.
    Distinct burn values so first and last produce different output.
    """
    return [
        (Point(1.5, 1.5), 9.0),
        (LineString([(0.0, 1.5), (3.0, 1.5)]), 5.0),
        (box(0, 0, 3, 3), 1.0),
    ]


def _run(merge, all_touched=False, **kwargs):
    return rasterize(
        _mixed_three_types(), width=3, height=3, bounds=(0, 0, 3, 3),
        fill=0, merge=merge, all_touched=all_touched, **kwargs)


#: Backend-selection kwargs for the four supported backends.
_BACKENDS = [
    pytest.param({}, id='numpy'),
    pytest.param({'chunks': 2}, id='dask_numpy',
                 marks=skip_no_dask),
    pytest.param({'gpu': True}, id='cupy',
                 marks=skip_no_cuda),
    pytest.param({'gpu': True, 'chunks': 2}, id='dask_cupy',
                 marks=[skip_no_cuda, skip_no_dask]),
]


# ---------------------------------------------------------------------------
# Known-winner pins (independent of any backend comparison)
# ---------------------------------------------------------------------------

class TestKnownWinners:
    """Pin the expected winners so the parity tests below cannot pass
    with both backends wrong in the same way."""

    @pytest.mark.parametrize('backend_kwargs', _BACKENDS)
    def test_last_polygon_wins_everywhere(self, backend_kwargs):
        # Polygon is the LAST input and covers the whole grid, so it
        # beats the point and the line at every pixel.
        vals = _as_numpy(_run('last', **backend_kwargs))
        assert (vals == 1.0).all()

    @pytest.mark.parametrize('backend_kwargs', _BACKENDS)
    def test_first_earlier_inputs_win(self, backend_kwargs):
        # Point (input 0) wins its pixel, line (input 1) wins the rest
        # of row 1, polygon (input 2) fills everything else.
        vals = _as_numpy(_run('first', **backend_kwargs))
        assert vals[1, 1] == 9.0
        assert vals[1, 0] == 5.0
        assert vals[1, 2] == 5.0
        assert vals[0, 0] == 1.0
        assert vals[2, 2] == 1.0


# ---------------------------------------------------------------------------
# dask+numpy all_touched parity (CPU-only, runs without CUDA)
# ---------------------------------------------------------------------------

@skip_no_dask
class TestDaskNumpyAllTouchedOrderedMerge:

    @pytest.mark.parametrize('merge', ['first', 'last'])
    def test_dask_numpy_all_touched_matches_numpy(self, merge):
        # The dask+numpy tile path runs the supercover boundary burn
        # against a tile-local ``order`` array whose global indices are
        # sliced by ``_slice_per_tile``, so ordered merges exercise
        # tile-local ordering machinery the eager path never touches.
        expected = _as_numpy(_run(merge, all_touched=True))
        got = _as_numpy(_run(merge, all_touched=True, chunks=2))
        np.testing.assert_array_equal(expected, got)


# ---------------------------------------------------------------------------
# GPU parity against the numpy reference
# ---------------------------------------------------------------------------

@skip_no_cuda
class TestGpuMixedTypeOrderedMergeParity:

    @pytest.mark.parametrize('merge', ['first', 'last'])
    def test_eager_cupy_matches_numpy(self, merge):
        expected = _as_numpy(_run(merge))
        got = _as_numpy(_run(merge, gpu=True))
        np.testing.assert_array_equal(expected, got)

    @skip_no_dask
    @pytest.mark.parametrize('merge', ['first', 'last'])
    def test_dask_cupy_matches_numpy(self, merge):
        # chunks=2 splits the 3x3 grid into four tiles, so the point,
        # line, and polygon land in different per-tile launches while
        # still competing inside the shared tiles.
        expected = _as_numpy(_run(merge))
        got = _as_numpy(_run(merge, gpu=True, chunks=2))
        np.testing.assert_array_equal(expected, got)

    @pytest.mark.parametrize('merge', ['first', 'last'])
    def test_eager_cupy_all_touched_matches_numpy(self, merge):
        # all_touched=True adds the supercover polygon-boundary burn,
        # which has its own pass-1/pass-2 kernel pair.  Cross-type
        # competition through that path was previously untested.
        #
        # dask+cupy is deliberately NOT asserted here: all_touched
        # parity on that backend is blocked by a pre-existing
        # boundary-segment-on-tile-border bug (see the smoke-test note
        # in test_rasterize_props_hoist_2506.py), unrelated to merge
        # ordering.
        expected = _as_numpy(_run(merge, all_touched=True))
        got = _as_numpy(_run(merge, all_touched=True, gpu=True))
        np.testing.assert_array_equal(expected, got)

    def test_fixture_not_degenerate_on_gpu(self):
        # Guard: first and last must disagree on this scene, otherwise
        # the parity tests above could not catch a regression that
        # wires both modes to the same kernel.
        first = _as_numpy(_run('first', gpu=True))
        last = _as_numpy(_run('last', gpu=True))
        assert (first != last).any()
