"""Coverage-gap tests for xrspatial.rasterize (deep-sweep test-coverage, pass 5).

Closes two parameter-coverage gaps left open by the pass-1 (2026-05-17),
pass-2 (2026-05-21), pass-3 (2026-05-27), and pass-4 (2026-05-29) audits.
Issue #3105.

- ``all_touched=True`` with LineString input had no coverage on any
  backend.  The flag currently changes polygon handling only: lines burn
  through Bresenham whether the flag is set or not, so the output matches
  rasterio's *default* mode rather than its all_touched mode.  Issue
  #3102 tracks that divergence.  The tests here pin the current contract
  (all_touched is a no-op for lines) on numpy / cupy / dask+numpy /
  dask+cupy, plus a strict xfail for rasterio all_touched parity that
  flips visibly when #3102 lands.

- the non-iterable ``geometries`` TypeError in ``_parse_input``
  (``"geometries must be a GeoDataFrame or iterable of (geometry, value)
  pairs"``) had no test.

The "fix" in this sweep is *adding tests*.  No source changes.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from shapely.geometry import LineString, MultiLineString
    has_shapely = True
except ImportError:
    has_shapely = False

try:
    import cupy
    has_cupy = True
except ImportError:
    cupy = None
    has_cupy = False

try:
    import dask.array  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False

try:
    import rasterio
    import rasterio.features
    from rasterio.transform import from_bounds
    has_rasterio = True
except ImportError:
    has_rasterio = False

try:
    from numba import cuda
    has_cuda = has_cupy and cuda.is_available()
except Exception:
    has_cuda = False

if has_shapely:
    from xrspatial.rasterize import rasterize

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)

skip_no_cuda = pytest.mark.skipif(
    not has_cuda, reason="CUDA not available")
skip_no_dask = pytest.mark.skipif(
    not has_dask, reason="dask not installed")


def _materialise(result):
    """Bring any backend's DataArray data down to a numpy array."""
    data = result.data
    if has_dask and isinstance(data, dask.array.Array):
        data = data.compute()
    if has_cupy and isinstance(data, cupy.ndarray):
        return cupy.asnumpy(data)
    return np.asarray(data)


_BACKENDS = [
    pytest.param('numpy', {}, id='numpy'),
    pytest.param('cupy', {'use_cuda': True}, marks=skip_no_cuda, id='cupy'),
    pytest.param('dask_numpy', {'chunks': (3, 3)}, marks=skip_no_dask,
                 id='dask_numpy'),
    pytest.param('dask_cupy', {'use_cuda': True, 'chunks': (3, 3)},
                 marks=[skip_no_cuda, skip_no_dask], id='dask_cupy'),
]

# Diagonal line crossing pixel corners: Bresenham burns the 5-pixel
# anti-diagonal, while a supercover (rasterio all_touched) traversal
# burns 9 pixels.  The gap between the two is what makes this geometry
# a sharp probe for the all_touched x line interaction.
_DIAG = [(0.2, 0.2), (4.8, 4.8)]
# Shallow line crossing cell interiors (never a pixel corner), so the
# no-op pin is not hostage to tie-breaking on shared corners.
_INTERIOR = [(0.5, 0.7), (4.5, 3.3)]
_GRID = dict(width=5, height=5, bounds=(0.0, 0.0, 5.0, 5.0), fill=0.0)


def _line():
    return LineString(_DIAG)


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM -- all_touched with line input (issue #3102 pins)
# ---------------------------------------------------------------------------

class TestAllTouchedLineNoOp:
    """Pin the current contract: ``all_touched`` does not affect lines.

    ``_run_numpy`` applies the supercover burn to polygon boundary
    segments only; lines always go through ``_burn_lines_cpu`` (and the
    GPU / dask tile paths mirror that layout).  Issue #3102 tracks
    whether lines should join the supercover path.  Until that is
    decided, pin the no-op on every backend so any behavior change is
    visible in CI rather than shipping silently.
    """

    @pytest.mark.parametrize('coords', [_DIAG, _INTERIOR],
                             ids=['corner_crossing', 'interior_crossing'])
    @pytest.mark.parametrize('backend_name,kw', _BACKENDS)
    def test_line_all_touched_equals_default(self, backend_name, kw, coords):
        line = LineString(coords)
        flagged = rasterize([(line, 1.0)], **_GRID,
                            all_touched=True, **kw)
        default = rasterize([(line, 1.0)], **_GRID,
                            all_touched=False, **kw)
        np.testing.assert_array_equal(
            _materialise(flagged), _materialise(default))

    def test_line_all_touched_burns_bresenham_diagonal(self):
        """The numpy result is the 5-pixel Bresenham anti-diagonal."""
        r = rasterize([(_line(), 1.0)], **_GRID, all_touched=True)
        expected = np.eye(5)[:, ::-1]
        np.testing.assert_array_equal(r.data, expected)

    def test_multilinestring_all_touched_equals_default(self):
        """The MultiLineString explode path is also unaffected."""
        mls = MultiLineString([_DIAG, [(0.5, 4.5), (4.5, 4.5)]])
        flagged = rasterize([(mls, 1.0)], **_GRID, all_touched=True)
        default = rasterize([(mls, 1.0)], **_GRID, all_touched=False)
        np.testing.assert_array_equal(flagged.data, default.data)


@pytest.mark.skipif(not has_rasterio, reason="rasterio not installed")
class TestAllTouchedLineRasterioParity:
    """Where lines + all_touched stand relative to rasterio."""

    def _rio(self, all_touched):
        transform = from_bounds(*_GRID['bounds'],
                                _GRID['width'], _GRID['height'])
        return rasterio.features.rasterize(
            [(_line(), 1)],
            out_shape=(_GRID['height'], _GRID['width']),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype='uint8',
        )

    def test_line_all_touched_matches_rasterio_default_mode(self):
        """Today's output equals rasterio's *default* (Bresenham-style)
        burn, not its all_touched burn."""
        r = rasterize([(_line(), 1.0)], **_GRID, all_touched=True)
        np.testing.assert_array_equal(
            r.data.astype(np.uint8), self._rio(all_touched=False))

    @pytest.mark.xfail(
        strict=True,
        reason="all_touched=True is a no-op for lines; supercover burn "
               "is applied to polygon boundaries only (issue #3102)")
    def test_line_all_touched_matches_rasterio_all_touched(self):
        """Flips when #3102 plumbs line segments through the supercover
        burn.  Strict, so the fix must update the no-op pins above."""
        r = rasterize([(_line(), 1.0)], **_GRID, all_touched=True)
        np.testing.assert_array_equal(
            r.data.astype(np.uint8), self._rio(all_touched=True))


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM -- non-iterable ``geometries`` error path
# ---------------------------------------------------------------------------

class TestNonIterableGeometries:
    """``_parse_input`` rejects input that is neither a GeoDataFrame nor
    an iterable of (geometry, value) pairs.  The guard had no test."""

    @pytest.mark.parametrize('bad', [42, 1.5, None, object()],
                             ids=['int', 'float', 'none', 'object'])
    def test_non_iterable_raises_typeerror(self, bad):
        with pytest.raises(
            TypeError,
            match=r"geometries must be a GeoDataFrame or iterable",
        ):
            rasterize(bad, width=4, height=4, bounds=(0, 0, 4, 4))
