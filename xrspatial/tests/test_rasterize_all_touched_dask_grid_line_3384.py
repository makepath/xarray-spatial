"""Dask/eager parity for ``rasterize(all_touched=True)`` when a polygon
boundary lies exactly on a pixel-grid line (issue #3384).

The eager numpy/cupy backends convert polygon boundary vertices to pixel
coordinates against the full raster origin and walk them with the
Amanatides & Woo supercover traversal.  The dask tile workers used to
re-extract those boundary segments per tile against each tile's own
world origin, which reintroduced a different floating-point rounding: a
segment landing on an exact integer pixel row in the full grid (e.g.
``(10.0 - 4.0) / 0.4 == 15.0``) lands at ``1.9999999999999996`` in
tile-local space, so ``floor()`` picked the wrong cell and the dask
backends burned a different row/column than the eager backend.

The existing supercover parity module deliberately nudges polygon
vertices off integer boundaries, so this on-grid case was untested.
These tests pin dask == eager for a polygon whose edges sit exactly on
pixel-grid lines, across every backend and merge mode, including the
chunk layouts that split an on-grid edge at a non-aligned tile boundary.
"""
import numpy as np
import pytest

try:
    from shapely.geometry import Polygon
    has_shapely = True
except ImportError:
    has_shapely = False

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
    pytest.mark.skipif(not has_dask, reason="dask not installed"),
]

skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason="CUDA / CuPy not available")

# 25 x 25 raster over a 10 x 10 world extent => px == py == 0.4.  A
# polygon edge at world y == 4 maps to float pixel row (10 - 4) / 0.4,
# which is exactly 15.0 in the full grid but rounds to 14.999... when a
# tile recomputes it against a tile origin of ymax == 4.8.
_W = _H = 25
_BOUNDS = (0.0, 0.0, 10.0, 10.0)

# Axis-aligned box whose four edges all land on pixel-grid lines.
_ON_GRID_POLY = Polygon([(4, 4), (9, 4), (9, 9), (4, 9)])

# Chunk layouts: (13, 13) and (7, 7) split the on-grid edges at tile
# boundaries that do not themselves sit on those grid lines, which is
# what triggered the divergence.  (10, 10) and (5, 5) align with the
# edges and matched even before the fix -- kept as controls.
_CHUNKS = [(13, 13), (7, 7), (10, 10), (5, 5)]

_MERGES = ['last', 'first', 'max', 'min', 'sum', 'count']


def _host(arr):
    """Materialize a rasterize result to a host numpy array."""
    data = arr.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if has_cupy and isinstance(data, cupy.ndarray):
        data = cupy.asnumpy(data)
    return np.asarray(data)


def _eager(merge, *, gpu=False):
    fill = 0.0 if merge in ('sum', 'count') else np.nan
    return _host(rasterize(
        [(_ON_GRID_POLY, 1.0)], width=_W, height=_H, bounds=_BOUNDS,
        all_touched=True, fill=fill, merge=merge, gpu=gpu))


def _chunked(merge, chunks, *, gpu=False):
    fill = 0.0 if merge in ('sum', 'count') else np.nan
    return _host(rasterize(
        [(_ON_GRID_POLY, 1.0)], width=_W, height=_H, bounds=_BOUNDS,
        all_touched=True, fill=fill, merge=merge, chunks=chunks, gpu=gpu))


def _assert_equal_with_nan(got, ref, msg):
    same = (got == ref) | (np.isnan(got) & np.isnan(ref))
    assert np.all(same), (
        f"{msg}: {int(np.size(got) - same.sum())} pixels differ")


@pytest.mark.parametrize('merge', _MERGES)
@pytest.mark.parametrize('chunks', _CHUNKS)
def test_dask_numpy_matches_eager_on_grid_boundary(merge, chunks):
    ref = _eager(merge)
    got = _chunked(merge, chunks)
    _assert_equal_with_nan(
        got, ref, f"dask+numpy merge={merge} chunks={chunks}")


@skip_no_cuda
@pytest.mark.parametrize('merge', _MERGES)
@pytest.mark.parametrize('chunks', _CHUNKS)
def test_dask_cupy_matches_eager_on_grid_boundary(merge, chunks):
    ref = _eager(merge, gpu=True)
    got = _chunked(merge, chunks, gpu=True)
    _assert_equal_with_nan(
        got, ref, f"dask+cupy merge={merge} chunks={chunks}")


def test_regression_specific_pixel_count():
    """The original report: chunks=(13, 13) under-filled by 13 pixels."""
    eager = _eager('last')
    dask = _chunked('last', (13, 13))
    assert int((eager > 0).sum()) == int((dask > 0).sum()) == 182
    _assert_equal_with_nan(dask, eager, "regression chunks=(13, 13)")
