"""Coverage-gap tests for xrspatial.rasterize (deep-sweep test-coverage, pass 5).

Closes two parameter-coverage gaps left open by the pass-1 (2026-05-17),
pass-2 (2026-05-21), pass-3 (2026-05-27), and pass-4 (2026-05-29) audits.
Issue #3105.

- ``all_touched=True`` with LineString input had no coverage on any
  backend.  Issue #3102 fixed the flag to route lines through the same
  supercover (Amanatides & Woo) traversal that polygon boundaries use,
  so every cell a line crosses is burned.  The tests here pin that
  behavior on numpy / cupy / dask+numpy / dask+cupy and check rasterio
  all_touched parity for an off-corner line.

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
# Cat 4 MEDIUM -- all_touched with line input (issue #3102)
# ---------------------------------------------------------------------------

# Off-corner line: every endpoint and crossing lands inside a cell, so
# the supercover walk and rasterio agree pixel-for-pixel (no exact
# corner-crossing tie-break to disagree on).
_OFF_CORNER = [(0.5, 0.7), (4.5, 3.3)]


class TestAllTouchedLineSupercover:
    """``all_touched`` routes lines through the supercover walk (#3102).

    Before the fix lines always went through ``_burn_lines_cpu`` (and the
    GPU / dask tile paths mirrored that layout), so ``all_touched=True``
    returned the same pixels as ``all_touched=False``.  These pin the
    fixed behavior on every backend.
    """

    @pytest.mark.parametrize('backend_name,kw', _BACKENDS)
    def test_line_all_touched_burns_more_than_default(self, backend_name, kw):
        line = LineString(_OFF_CORNER)
        flagged = _materialise(
            rasterize([(line, 1.0)], **_GRID, all_touched=True, **kw))
        default = _materialise(
            rasterize([(line, 1.0)], **_GRID, all_touched=False, **kw))
        assert int((flagged > 0).sum()) > int((default > 0).sum())

    @pytest.mark.parametrize('backend_name,kw', _BACKENDS[1:])
    def test_line_all_touched_matches_numpy(self, backend_name, kw):
        """Every backend agrees with the eager numpy supercover burn."""
        line = LineString(_OFF_CORNER)
        cpu = _materialise(
            rasterize([(line, 1.0)], **_GRID, all_touched=True))
        other = _materialise(
            rasterize([(line, 1.0)], **_GRID, all_touched=True, **kw))
        np.testing.assert_array_equal(other, cpu)

    def test_multilinestring_all_touched_burns_more(self):
        """The MultiLineString explode path also joins the supercover."""
        mls = MultiLineString([_OFF_CORNER, [(0.5, 4.5), (4.5, 4.5)]])
        flagged = rasterize([(mls, 1.0)], **_GRID, all_touched=True)
        default = rasterize([(mls, 1.0)], **_GRID, all_touched=False)
        assert int((flagged.data > 0).sum()) > int((default.data > 0).sum())


@pytest.mark.skipif(not has_rasterio, reason="rasterio not installed")
class TestAllTouchedLineRasterioParity:
    """Lines + all_touched now match rasterio's all_touched burn."""

    def _rio(self, coords, all_touched):
        transform = from_bounds(*_GRID['bounds'],
                                _GRID['width'], _GRID['height'])
        return rasterio.features.rasterize(
            [(LineString(coords), 1)],
            out_shape=(_GRID['height'], _GRID['width']),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype='uint8',
        )

    def test_default_line_matches_rasterio_default_mode(self):
        """all_touched=False still equals rasterio's default burn."""
        r = rasterize([(_line(), 1.0)], **_GRID, all_touched=False)
        np.testing.assert_array_equal(
            r.data.astype(np.uint8), self._rio(_DIAG, all_touched=False))

    def test_line_all_touched_matches_rasterio_all_touched(self):
        """Off-corner line: supercover matches rasterio's all_touched
        burn pixel-for-pixel."""
        r = rasterize([(LineString(_OFF_CORNER), 1.0)], **_GRID,
                      all_touched=True)
        np.testing.assert_array_equal(
            r.data.astype(np.uint8), self._rio(_OFF_CORNER, all_touched=True))


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
