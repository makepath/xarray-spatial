"""rasterio parity for ``rasterize(all_touched=True)`` (issue #2169).

The previous implementation drew polygon boundaries with a Bresenham
line, which steps one pixel per dominant-axis step and misses the
off-axis cells a shallow or diagonal edge crosses.  This module pins
the supercover (Amanatides & Woo grid traversal) behaviour against
``rasterio.features.rasterize(..., all_touched=True)`` for four
polygons across every backend the project supports.

Polygons are chosen so their edges do not lie exactly on cell
boundaries: rasterio and the supercover walker can legitimately
disagree on tie-breaking when an edge runs along a grid line, and the
issue scope explicitly allows that wiggle room.  Polygons whose edges
sit on pixel *centres* (x.5, y.5) are still well defined and are
included as a regression case for the older Bresenham path.
"""
import numpy as np
import pytest

try:
    from shapely.geometry import Polygon
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
    import cupy
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

pytestmark = [
    pytest.mark.skipif(not has_shapely, reason="shapely not installed"),
    pytest.mark.skipif(not has_rasterio, reason="rasterio not installed"),
]


# Raster used for every parity case in this module.
_H, _W = 20, 20
_BOUNDS = (0.0, 0.0, 20.0, 20.0)


def _rio_mask(poly):
    """Reference rasterio mask for ``poly`` over the shared raster."""
    transform = from_bounds(*_BOUNDS, _W, _H)
    return rasterio.features.rasterize(
        [(poly, 1)],
        out_shape=(_H, _W),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='uint8',
    )


def _xs_mask(poly, *, chunks=None, use_cuda=False):
    """xrspatial mask for ``poly`` on the requested backend."""
    kwargs = dict(width=_W, height=_H, bounds=_BOUNDS,
                  all_touched=True, fill=0)
    if chunks is not None:
        kwargs['chunks'] = chunks
    if use_cuda:
        kwargs['use_cuda'] = True
    arr = rasterize([(poly, 1.0)], **kwargs)
    values = arr.data
    # dask -> concrete; cupy -> host numpy.
    if hasattr(values, 'compute'):
        values = values.compute()
    if hasattr(values, 'get'):
        values = values.get()
    return (np.asarray(values) > 0).astype('uint8')


# Polygons exercised by every parity test in this module.  Vertices are
# nudged off integer boundaries so the only thing being verified is
# supercover coverage (not edge tie-breaking on shared grid lines).
def _cases():
    return {
        'diagonal_strip': Polygon([
            (0.5, 0.5), (19.5, 18.5), (19.5, 19.5), (0.5, 1.5),
        ]),
        'thin_triangle': Polygon([
            (1.2, 1.0), (18.7, 2.3), (10.0, 3.1),
        ]),
        'edges_on_pixel_centers': Polygon([
            (2.5, 2.5), (8.5, 2.5), (8.5, 8.5), (2.5, 8.5),
        ]),
        'concave_subpixel': Polygon([
            (2.2, 2.2), (15.4, 5.1), (10.0, 10.0),
            (15.4, 15.1), (2.2, 18.2),
        ]),
    }


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------

class TestNumpySupercoverParity:
    @pytest.mark.parametrize("name,poly", list(_cases().items()))
    def test_matches_rasterio(self, name, poly):
        rio = _rio_mask(poly)
        xs = _xs_mask(poly)
        # Pixel-for-pixel match -- supercover should now cover every
        # cell rasterio covers for these subpixel-vertex polygons.
        assert np.array_equal(rio, xs), (
            f"{name}: rio={int(rio.sum())} xs={int(xs.sum())} "
            f"diff_pixels={int(np.abs(rio.astype(int) - xs.astype(int)).sum())}"
        )

    def test_diagonal_strip_was_undercounted_before(self):
        """The original Bresenham boundary missed off-axis pixels on a
        shallow diagonal.  The supercover burn must now meet or beat
        rasterio on this case (it should match exactly)."""
        poly = _cases()['diagonal_strip']
        rio = _rio_mask(poly)
        xs = _xs_mask(poly)
        assert int(xs.sum()) == int(rio.sum()) > 10


# ---------------------------------------------------------------------------
# dask + numpy backend
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_dask, reason="dask not installed")
class TestDaskNumpySupercoverParity:
    @pytest.mark.parametrize("name,poly", list(_cases().items()))
    def test_matches_rasterio(self, name, poly):
        rio = _rio_mask(poly)
        # Two chunks per axis so we exercise the per-tile boundary burn
        # rather than the eager single-tile path.
        xs = _xs_mask(poly, chunks=(_H // 2, _W // 2))
        assert np.array_equal(rio, xs), (
            f"{name}: rio={int(rio.sum())} xs={int(xs.sum())}"
        )

    def test_dask_matches_numpy(self):
        """Dask + numpy and eager numpy must agree pixel-for-pixel under
        all_touched=True now that both use the supercover burn."""
        poly = _cases()['concave_subpixel']
        eager = _xs_mask(poly)
        chunked = _xs_mask(poly, chunks=(_H // 2, _W // 2))
        assert np.array_equal(eager, chunked)


# ---------------------------------------------------------------------------
# cupy backend (skipped without a GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_cuda, reason="CUDA / CuPy not available")
class TestCupySupercoverParity:
    @pytest.mark.parametrize("name,poly", list(_cases().items()))
    def test_matches_rasterio(self, name, poly):
        rio = _rio_mask(poly)
        xs = _xs_mask(poly, use_cuda=True)
        assert np.array_equal(rio, xs), (
            f"{name}: rio={int(rio.sum())} xs={int(xs.sum())}"
        )


# ---------------------------------------------------------------------------
# dask + cupy backend (skipped without a GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (has_cuda and has_dask),
                    reason="dask + CUDA / CuPy not available")
class TestDaskCupySupercoverParity:
    @pytest.mark.parametrize("name,poly", list(_cases().items()))
    def test_matches_rasterio(self, name, poly):
        rio = _rio_mask(poly)
        xs = _xs_mask(poly, chunks=(_H // 2, _W // 2), use_cuda=True)
        assert np.array_equal(rio, xs), (
            f"{name}: rio={int(rio.sum())} xs={int(xs.sum())}"
        )
