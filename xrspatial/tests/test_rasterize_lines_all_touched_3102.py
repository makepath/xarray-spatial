"""rasterize ``all_touched`` with line input, plus the non-iterable input
error path (issues #3102 and #3105).

#3102: ``all_touched=True`` used to be a no-op for LineString /
MultiLineString -- lines always went through the Bresenham burner, so the
output matched ``all_touched=False``.  The fix routes lines through the
same supercover (Amanatides & Woo) grid traversal that polygon boundaries
use, so every cell a line crosses is burned.  These tests pin that across
numpy, cupy, dask+numpy, and dask+cupy.

#3105: the two coverage gaps the test-coverage sweep flagged -- lines with
``all_touched`` on every backend, and the ``_parse_input`` TypeError for
non-iterable input.

Single-segment lines are nudged off integer cell boundaries so the
supercover walk and rasterio agree pixel-for-pixel.  Multi-segment lines
can disagree with rasterio by a cell or two at shared vertices / exact
corner crossings (the same tie-breaking wiggle the polygon parity tests
in ``test_rasterize_all_touched_supercover_2169.py`` allow), so those are
checked for cross-backend equality and for covering at least as many
cells as the Bresenham path rather than for exact rasterio parity.
"""
import numpy as np
import pytest

try:
    from shapely.geometry import LineString, MultiLineString
    has_shapely = True
except ImportError:
    has_shapely = False

try:
    import rasterio  # noqa: F401
    import rasterio.features  # noqa: F401
    from rasterio.transform import from_bounds
    has_rasterio = True
except ImportError:
    has_rasterio = False

try:
    import cupy  # noqa: F401
    has_cupy = True
except ImportError:
    has_cupy = False

try:
    import dask.array as da  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False

try:
    from numba import cuda
    has_cuda = has_cupy and cuda.is_available()
except Exception:
    has_cuda = False

if has_shapely:
    from xrspatial.rasterize import rasterize

pytestmark = pytest.mark.skipif(not has_shapely, reason="shapely not installed")


_H, _W = 20, 20
_BOUNDS = (0.0, 0.0, 20.0, 20.0)


def _to_host(arr):
    values = arr.data
    if hasattr(values, 'compute'):
        values = values.compute()
    if hasattr(values, 'get'):
        values = values.get()
    return np.asarray(values)


def _xs_mask(line, *, all_touched, chunks=None, gpu=False):
    kwargs = dict(width=_W, height=_H, bounds=_BOUNDS,
                  all_touched=all_touched, fill=0)
    if chunks is not None:
        kwargs['chunks'] = chunks
    if gpu:
        kwargs['gpu'] = True
    arr = rasterize([(line, 1.0)], **kwargs)
    return (_to_host(arr) > 0).astype('uint8')


def _rio_mask(line):
    transform = from_bounds(*_BOUNDS, _W, _H)
    return rasterio.features.rasterize(
        [(line, 1)], out_shape=(_H, _W), transform=transform,
        fill=0, all_touched=True, dtype='uint8')


# Single-segment lines with sub-pixel endpoints -- supercover and
# rasterio agree exactly here (no shared vertex, no exact corner cross).
def _exact_cases():
    return {
        'shallow': LineString([(0.7, 1.3), (19.3, 7.6)]),
        'steep': LineString([(2.4, 0.5), (6.1, 19.4)]),
        'descending': LineString([(0.6, 18.7), (19.4, 2.2)]),
        'multi_disjoint': MultiLineString([
            [(0.7, 1.3), (19.3, 7.6)],
            [(2.4, 0.5), (6.1, 19.4)],
        ]),
    }


# Multi-segment line: cross-backend equality + supercover-covers-Bresenham,
# but not pinned to rasterio (corner tie-break wiggle at shared vertices).
def _coverage_case():
    return LineString([(1.2, 1.4), (9.6, 7.3), (3.1, 14.8), (18.4, 18.2)])


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------

class TestNumpyLinesAllTouched:
    @pytest.mark.skipif(not has_rasterio, reason="rasterio not installed")
    @pytest.mark.parametrize("name,line", list(_exact_cases().items()))
    def test_matches_rasterio(self, name, line):
        rio = _rio_mask(line)
        xs = _xs_mask(line, all_touched=True)
        assert np.array_equal(rio, xs), (
            f"{name}: rio={int(rio.sum())} xs={int(xs.sum())} "
            f"diff={int(np.abs(rio.astype(int) - xs.astype(int)).sum())}"
        )

    def test_all_touched_burns_more_than_bresenham(self):
        """The bug: all_touched=True returned the same pixels as
        all_touched=False for lines.  After the fix the supercover walk
        must add the off-axis cells a diagonal segment crosses."""
        line = _exact_cases()['shallow']
        at = _xs_mask(line, all_touched=True)
        no = _xs_mask(line, all_touched=False)
        assert int(at.sum()) > int(no.sum())

    @pytest.mark.skipif(not has_rasterio, reason="rasterio not installed")
    def test_coverage_at_least_bresenham(self):
        line = _coverage_case()
        at = _xs_mask(line, all_touched=True)
        no = _xs_mask(line, all_touched=False)
        rio = _rio_mask(line)
        assert int(at.sum()) >= int(no.sum())
        # Within a couple of cells of rasterio (corner tie-break wiggle).
        assert abs(int(at.sum()) - int(rio.sum())) <= 2


# ---------------------------------------------------------------------------
# dask + numpy backend
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_dask, reason="dask not installed")
class TestDaskNumpyLinesAllTouched:
    @pytest.mark.skipif(not has_rasterio, reason="rasterio not installed")
    @pytest.mark.parametrize("name,line", list(_exact_cases().items()))
    def test_matches_rasterio(self, name, line):
        rio = _rio_mask(line)
        xs = _xs_mask(line, all_touched=True, chunks=(_H // 2, _W // 2))
        assert np.array_equal(rio, xs), (
            f"{name}: rio={int(rio.sum())} xs={int(xs.sum())}"
        )

    @pytest.mark.parametrize("name,line",
                             list(_exact_cases().items()) +
                             [('coverage', _coverage_case())])
    def test_dask_matches_numpy(self, name, line):
        eager = _xs_mask(line, all_touched=True)
        chunked = _xs_mask(line, all_touched=True, chunks=(_H // 2, _W // 2))
        assert np.array_equal(eager, chunked), (
            f"{name}: eager={int(eager.sum())} chunked={int(chunked.sum())}"
        )


# ---------------------------------------------------------------------------
# cupy backend (skipped without a GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_cuda, reason="CUDA / CuPy not available")
class TestCupyLinesAllTouched:
    @pytest.mark.skipif(not has_rasterio, reason="rasterio not installed")
    @pytest.mark.parametrize("name,line", list(_exact_cases().items()))
    def test_matches_rasterio(self, name, line):
        rio = _rio_mask(line)
        xs = _xs_mask(line, all_touched=True, gpu=True)
        assert np.array_equal(rio, xs), (
            f"{name}: rio={int(rio.sum())} xs={int(xs.sum())}"
        )

    def test_cupy_matches_numpy(self):
        line = _coverage_case()
        cpu = _xs_mask(line, all_touched=True)
        gpu = _xs_mask(line, all_touched=True, gpu=True)
        assert np.array_equal(cpu, gpu)


# ---------------------------------------------------------------------------
# dask + cupy backend (skipped without a GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (has_cuda and has_dask),
                    reason="dask + CUDA / CuPy not available")
class TestDaskCupyLinesAllTouched:
    @pytest.mark.skipif(not has_rasterio, reason="rasterio not installed")
    @pytest.mark.parametrize("name,line", list(_exact_cases().items()))
    def test_matches_rasterio(self, name, line):
        rio = _rio_mask(line)
        xs = _xs_mask(line, all_touched=True, chunks=(_H // 2, _W // 2),
                      gpu=True)
        assert np.array_equal(rio, xs), (
            f"{name}: rio={int(rio.sum())} xs={int(xs.sum())}"
        )

    def test_dask_cupy_matches_numpy(self):
        line = _coverage_case()
        cpu = _xs_mask(line, all_touched=True)
        gpu = _xs_mask(line, all_touched=True, chunks=(_H // 2, _W // 2),
                       gpu=True)
        assert np.array_equal(cpu, gpu)


# ---------------------------------------------------------------------------
# merge='sum' / 'count' dedup with all_touched lines (issue #3304 path)
# ---------------------------------------------------------------------------

class TestLinesAllTouchedDedup:
    def test_self_crossing_sum_counts_once(self):
        """A self-crossing line under merge='sum' must contribute its
        value at most once per pixel (per-geometry dedup), even though
        the supercover walk visits the crossing cell from two segments."""
        line = LineString([(0.5, 0.5), (8.5, 8.5), (0.5, 8.5), (8.5, 0.5)])
        arr = rasterize([(line, 2.0)], width=10, height=10,
                        bounds=(0.0, 0.0, 10.0, 10.0), fill=0,
                        all_touched=True, merge='sum')
        host = _to_host(arr)
        assert np.nanmax(host) == 2.0

    @pytest.mark.skipif(not has_dask, reason="dask not installed")
    def test_sum_dask_matches_numpy(self):
        line = LineString([(0.5, 0.5), (8.5, 8.5), (0.5, 8.5), (8.5, 0.5)])
        common = dict(width=10, height=10, bounds=(0.0, 0.0, 10.0, 10.0),
                      fill=0, all_touched=True, merge='sum')
        eager = _to_host(rasterize([(line, 2.0)], **common))
        chunked = _to_host(rasterize([(line, 2.0)], chunks=(5, 5), **common))
        assert np.array_equal(np.nan_to_num(eager), np.nan_to_num(chunked))


# The non-iterable ``geometries`` TypeError path (issue #3105, item 2) is
# pinned by ``TestNonIterableGeometries`` in
# ``test_rasterize_coverage_2026_06_09.py``; not duplicated here.
