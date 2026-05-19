"""Coverage-gap tests for xrspatial.rasterize (deep-sweep test-coverage, pass 1).

Closes documented but untested public-API surface flagged by the
test-coverage sweep on 2026-05-17:

- Cat 3 HIGH  -- 1x1 single-pixel raster across numpy / cupy / dask+numpy /
  dask+cupy (test_rasterize.py covers 1xN strips and Nx1 strips but never
  the single-pixel degenerate case).
- Cat 4 HIGH  -- ``like=`` template-raster parameter forwards through
  ``_extract_grid_from_like`` into width/height/bounds/dtype resolution,
  but no test in test_rasterize.py exercises it.  The dtype-inheritance
  branch and the bounds-from-like branch ship without coverage on any
  backend.
- Cat 4 HIGH  -- ``resolution=`` parameter happy-path: only the
  oversize-rejection error path is tested; the scalar / tuple branches
  and the ceil-and-clamp-to-1 logic have no positive coverage on any
  backend.
- Cat 4 HIGH  -- Non-empty ``GeometryCollection`` unpacking is
  implemented by the GeometryCollection slow path in
  ``_classify_geometries`` but only the empty-GC case is tested.
  All four backends route through this path.
- Cat 1 MEDIUM -- eager cupy ``all_touched=True`` is covered only on the
  dask+cupy path; the eager cupy branch invokes a different kernel and
  had no direct test.
- Cat 2 MEDIUM -- integer dtype with the default ``fill=nan`` is
  unpinned behaviour: ``np.full(..., np.nan).astype(int)`` silently
  casts to the platform-specific int-min sentinel.  Pin the observed
  cast (numpy backend) so a future refactor that switches to an explicit
  raise surfaces in CI.

The "fix" in this sweep is *adding tests*.  No source changes.  CUDA is
available on this host so cupy / dask+cupy tests execute live.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

try:
    from shapely.geometry import (
        box, GeometryCollection, LineString, Point,
    )
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


def _as_numpy(result):
    """Materialise any backend's DataArray data to a numpy array."""
    data = result.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if has_cupy and isinstance(data, cupy.ndarray):
        return cupy.asnumpy(data)
    return np.asarray(data)


# ---------------------------------------------------------------------------
# Cat 3 HIGH -- 1x1 single-pixel raster
# ---------------------------------------------------------------------------

class TestSinglePixelRaster:
    """rasterize() on a 1x1 grid: the smallest legal degenerate shape.

    test_single_row_raster / test_single_column_raster cover the 1xN and
    Nx1 cases but the 1x1 case is its own degeneracy: the bounds-to-pixel
    transform collapses to a single (xmin, ymin)..(xmax, ymax) cell and
    every kernel (polygon scanline, line Bresenham, point burn) has to
    handle a 1-element output array.  A regression that mis-handled the
    height==1 or width==1 branch would already be caught, but the
    height==1 AND width==1 case has no test today.
    """

    def test_polygon_eager_numpy(self):
        """A polygon covering the bounds burns the single pixel."""
        r = rasterize([(box(0, 0, 5, 5), 7.0)],
                      width=1, height=1, bounds=(0, 0, 5, 5))
        assert r.shape == (1, 1)
        assert r.values[0, 0] == 7.0
        # Coords are the cell centre.
        assert r.coords['x'].values[0] == pytest.approx(2.5)
        assert r.coords['y'].values[0] == pytest.approx(2.5)

    def test_polygon_eager_numpy_fill_when_outside(self):
        """Polygon outside the single pixel leaves the fill value."""
        r = rasterize([(box(10, 10, 20, 20), 7.0)],
                      width=1, height=1, bounds=(0, 0, 5, 5), fill=-1.0)
        assert r.shape == (1, 1)
        assert r.values[0, 0] == -1.0

    def test_point_eager_numpy(self):
        """A point inside the single pixel burns it."""
        r = rasterize([(Point(2.5, 2.5), 9.0)],
                      width=1, height=1, bounds=(0, 0, 5, 5), fill=0)
        assert r.values[0, 0] == 9.0

    def test_line_eager_numpy(self):
        """A line crossing the single pixel burns it."""
        r = rasterize([(LineString([(0.0, 2.5), (5.0, 2.5)]), 3.0)],
                      width=1, height=1, bounds=(0, 0, 5, 5), fill=0)
        assert r.values[0, 0] == 3.0

    @skip_no_cuda
    def test_polygon_eager_cupy_matches_numpy(self):
        """1x1 raster on cupy matches numpy."""
        np_r = rasterize([(box(0, 0, 5, 5), 7.0)],
                         width=1, height=1, bounds=(0, 0, 5, 5))
        cp_r = rasterize([(box(0, 0, 5, 5), 7.0)],
                         width=1, height=1, bounds=(0, 0, 5, 5),
                         use_cuda=True)
        assert cp_r.shape == (1, 1)
        # Pin the absolute value too: a co-regression in eager numpy and
        # eager cupy (both writing fill instead of the burn value) would
        # otherwise slip past a pure parity check.
        assert _as_numpy(cp_r)[0, 0] == 7.0
        np.testing.assert_array_equal(np_r.values, _as_numpy(cp_r))

    @skip_no_dask
    def test_polygon_dask_numpy_matches_numpy(self):
        """1x1 raster on dask+numpy matches numpy."""
        np_r = rasterize([(box(0, 0, 5, 5), 7.0)],
                         width=1, height=1, bounds=(0, 0, 5, 5))
        dk_r = rasterize([(box(0, 0, 5, 5), 7.0)],
                         width=1, height=1, bounds=(0, 0, 5, 5),
                         chunks=(1, 1))
        assert dk_r.shape == (1, 1)
        assert _as_numpy(dk_r)[0, 0] == 7.0
        # Dask single-chunk pipeline must produce the same value.
        np.testing.assert_array_equal(np_r.values, _as_numpy(dk_r))

    @skip_no_cuda
    @skip_no_dask
    def test_polygon_dask_cupy_matches_numpy(self):
        """1x1 raster on dask+cupy matches numpy."""
        np_r = rasterize([(box(0, 0, 5, 5), 7.0)],
                         width=1, height=1, bounds=(0, 0, 5, 5))
        dkcp_r = rasterize([(box(0, 0, 5, 5), 7.0)],
                           width=1, height=1, bounds=(0, 0, 5, 5),
                           chunks=(1, 1), use_cuda=True)
        assert dkcp_r.shape == (1, 1)
        assert _as_numpy(dkcp_r)[0, 0] == 7.0
        np.testing.assert_array_equal(np_r.values, _as_numpy(dkcp_r))


# ---------------------------------------------------------------------------
# Cat 4 HIGH -- ``like=`` template-raster parameter
# ---------------------------------------------------------------------------

class TestLikeParameter:
    """``like=`` inherits width/height/bounds/dtype from a template.

    The public docstring at rasterize.py:2038 promises a "Template raster.
    Width, height, bounds, and dtype are copied from this array (any can
    still be overridden explicitly)".  No test in test_rasterize.py
    invokes the function with ``like=``, so each of the four inheritance
    branches and the three validation branches in
    ``_extract_grid_from_like`` ship without direct coverage.
    """

    @staticmethod
    def _template(height=4, width=6, dtype=np.float32):
        """A small north-up template with float32 dtype and explicit coords."""
        # y descends top-to-bottom (north-up convention used elsewhere).
        return xr.DataArray(
            np.zeros((height, width), dtype=dtype),
            dims=['y', 'x'],
            coords={
                'y': np.linspace(height - 0.5, 0.5, height),
                'x': np.linspace(0.5, width - 0.5, width),
            },
        )

    def test_like_inherits_width_height_dtype(self):
        """Output shape and dtype match the template."""
        template = self._template(height=4, width=6, dtype=np.float32)
        r = rasterize([(box(0, 0, 6, 4), 9.0)], like=template, fill=0)
        assert r.shape == (4, 6)
        assert r.dtype == np.float32

    def test_like_inherits_bounds_from_coords(self):
        """Bounds are reconstructed from the template's coordinate centres."""
        template = self._template(height=4, width=6, dtype=np.float32)
        r = rasterize([(box(0, 0, 6, 4), 9.0)], like=template, fill=0)
        # Cell centres should match the template exactly (so the half-pixel
        # offsets that _extract_grid_from_like applies are consistent).
        np.testing.assert_allclose(
            r.coords['y'].values, template.coords['y'].values)
        np.testing.assert_allclose(
            r.coords['x'].values, template.coords['x'].values)

    def test_like_dtype_override(self):
        """Explicit ``dtype=`` wins over the template dtype."""
        template = self._template(dtype=np.float32)
        r = rasterize([(box(0, 0, 6, 4), 9.0)],
                      like=template, dtype=np.float64, fill=0)
        assert r.dtype == np.float64

    def test_like_bounds_override(self):
        """Explicit ``bounds=`` wins over the template bounds (width/height
        from template are still honoured)."""
        template = self._template(height=4, width=6, dtype=np.float32)
        r = rasterize([(box(0, 0, 2, 2), 1.0)],
                      like=template, bounds=(0, 0, 2, 2), fill=0)
        # Shape stays from template, but the coords are recomputed off the
        # overridden bounds so the pixel size shrinks.
        assert r.shape == (4, 6)
        # width=6 over x in [0, 2] -> px=1/3, centres at [1, 3, 5, 7, 9, 11]/6.
        expected_x = np.array([1, 3, 5, 7, 9, 11]) / 6.
        np.testing.assert_allclose(r.coords['x'].values, expected_x)
        # height=4 over y in [0, 2] -> py=0.5, centres descend 1.75 -> 0.25.
        expected_y = np.array([1.75, 1.25, 0.75, 0.25])
        np.testing.assert_allclose(r.coords['y'].values, expected_y)

    def test_like_width_height_override(self):
        """Explicit ``width``/``height`` win over the template shape."""
        template = self._template(height=4, width=6, dtype=np.float32)
        r = rasterize([(box(0, 0, 6, 4), 1.0)],
                      like=template, width=3, height=2, fill=0)
        assert r.shape == (2, 3)
        # Dtype still inherited.
        assert r.dtype == np.float32

    @skip_no_cuda
    def test_like_with_use_cuda(self):
        """``like=`` works on the cupy backend (dtype + shape inherited)."""
        template = self._template(dtype=np.float32)
        r = rasterize([(box(0, 0, 6, 4), 9.0)],
                      like=template, fill=0, use_cuda=True)
        assert r.shape == template.shape
        assert r.dtype == np.float32
        assert isinstance(r.data, cupy.ndarray)

    @skip_no_dask
    def test_like_with_chunks(self):
        """``like=`` works on the dask+numpy backend."""
        template = self._template(dtype=np.float32)
        r = rasterize([(box(0, 0, 6, 4), 9.0)],
                      like=template, fill=0, chunks=(2, 3))
        assert r.shape == template.shape
        assert r.dtype == np.float32
        # Dask-backed.
        assert hasattr(r.data, 'dask')

    @skip_no_cuda
    @skip_no_dask
    def test_like_with_dask_cupy(self):
        """``like=`` works on the dask+cupy backend."""
        template = self._template(dtype=np.float32)
        r = rasterize([(box(0, 0, 6, 4), 9.0)],
                      like=template, fill=0, chunks=(2, 3),
                      use_cuda=True)
        assert r.shape == template.shape
        assert r.dtype == np.float32

    def test_like_rejects_non_dataarray(self):
        """Passing a numpy array as ``like`` raises ``TypeError``.

        Targets the ``isinstance(like, xr.DataArray)`` guard in
        ``_extract_grid_from_like``.
        """
        with pytest.raises(TypeError, match="must be an xr.DataArray"):
            rasterize([(box(0, 0, 5, 5), 1.0)],
                      like=np.zeros((3, 3)))

    def test_like_rejects_3d(self):
        """A 3D DataArray is rejected by the 2D shape guard.

        Note: this and ``test_like_rejects_wrong_dim_names`` both target
        the same compound ``ndim != 2 or 'y' not in dims or 'x' not in
        dims`` branch.  The two tests are kept distinct to document both
        sub-conditions; either would suffice for line coverage.
        """
        bad = xr.DataArray(np.zeros((2, 3, 3)), dims=['b', 'y', 'x'])
        with pytest.raises(ValueError, match="must be 2D"):
            rasterize([(box(0, 0, 5, 5), 1.0)], like=bad)

    def test_like_rejects_wrong_dim_names(self):
        """A 2D DataArray without 'y' and 'x' dims is rejected.

        Companion to ``test_like_rejects_3d``; targets the dim-name
        sub-condition of the same compound guard.
        """
        bad = xr.DataArray(np.zeros((3, 3)), dims=['lat', 'lon'])
        with pytest.raises(ValueError, match="'y' and 'x'"):
            rasterize([(box(0, 0, 5, 5), 1.0)], like=bad)


# ---------------------------------------------------------------------------
# Cat 4 HIGH -- ``resolution=`` parameter happy path
# ---------------------------------------------------------------------------

class TestResolutionParameter:
    """``resolution=`` resolves to width/height via ceil(extent / res).

    Only the oversize-rejection error path (test_oversize_resolution_rejected)
    is tested in test_rasterize.py.  The scalar and tuple branches in
    rasterize.py:2158-2164 and the ``max(..., 1)`` clamp have no positive
    coverage, on any backend.
    """

    def test_scalar_resolution_eager(self):
        """A single float resolution applies to both axes."""
        r = rasterize([(box(0, 0, 4, 4), 1.0)],
                      resolution=1.0, bounds=(0, 0, 4, 4), fill=0)
        assert r.shape == (4, 4)
        # Pixel covers (0..1)..(3..4); polygon fills all 16.
        assert int((r.values == 1.0).sum()) == 16

    def test_tuple_resolution_asymmetric(self):
        """A tuple resolution can give different x and y pixel counts."""
        r = rasterize([(box(0, 0, 10, 8), 1.0)],
                      resolution=(2.0, 4.0), bounds=(0, 0, 10, 8), fill=0)
        # width  = ceil(10 / 2) = 5
        # height = ceil( 8 / 4) = 2
        assert r.shape == (2, 5)

    def test_resolution_ceils_partial_extent(self):
        """Non-integer division ceils up to a full pixel."""
        r = rasterize([(box(0, 0, 3, 3), 1.0)],
                      resolution=1.5, bounds=(0, 0, 3.5, 3.5), fill=0)
        # ceil(3.5 / 1.5) = ceil(2.333) = 3
        assert r.shape == (3, 3)

    def test_resolution_clamps_to_at_least_one_pixel(self):
        """A resolution larger than the extent clamps to a 1x1 output
        rather than 0x0."""
        # extent 0.5 / resolution 1.0 = 0.5 -> ceil = 1 -> max(1, 1) = 1.
        r = rasterize([(box(0, 0, 1, 1), 5.0)],
                      resolution=10.0, bounds=(0, 0, 0.5, 0.5), fill=0)
        assert r.shape == (1, 1)

    @skip_no_cuda
    def test_scalar_resolution_cupy_matches_numpy(self):
        """resolution= on the cupy backend gives the same shape and values."""
        np_r = rasterize([(box(0, 0, 5, 5), 1.0)],
                         resolution=1.0, bounds=(0, 0, 5, 5), fill=0)
        cp_r = rasterize([(box(0, 0, 5, 5), 1.0)],
                         resolution=1.0, bounds=(0, 0, 5, 5), fill=0,
                         use_cuda=True)
        assert cp_r.shape == (5, 5)
        # Positive pin: polygon covers the full 5x5 grid.
        assert int((_as_numpy(cp_r) == 1.0).sum()) == 25
        np.testing.assert_array_equal(np_r.values, _as_numpy(cp_r))

    @skip_no_dask
    def test_scalar_resolution_dask_matches_numpy(self):
        """resolution= on the dask+numpy backend gives matching output."""
        np_r = rasterize([(box(0, 0, 5, 5), 1.0)],
                         resolution=1.0, bounds=(0, 0, 5, 5), fill=0)
        dk_r = rasterize([(box(0, 0, 5, 5), 1.0)],
                         resolution=1.0, bounds=(0, 0, 5, 5), fill=0,
                         chunks=(2, 2))
        assert dk_r.shape == (5, 5)
        assert int((_as_numpy(dk_r) == 1.0).sum()) == 25
        np.testing.assert_array_equal(np_r.values, _as_numpy(dk_r))

    @skip_no_cuda
    @skip_no_dask
    def test_scalar_resolution_dask_cupy_matches_numpy(self):
        """resolution= on the dask+cupy backend gives matching output."""
        np_r = rasterize([(box(0, 0, 5, 5), 1.0)],
                         resolution=1.0, bounds=(0, 0, 5, 5), fill=0)
        dkcp_r = rasterize([(box(0, 0, 5, 5), 1.0)],
                           resolution=1.0, bounds=(0, 0, 5, 5), fill=0,
                           chunks=(2, 2), use_cuda=True)
        assert dkcp_r.shape == (5, 5)
        assert int((_as_numpy(dkcp_r) == 1.0).sum()) == 25
        np.testing.assert_array_equal(np_r.values, _as_numpy(dkcp_r))


# ---------------------------------------------------------------------------
# Cat 4 HIGH -- Non-empty GeometryCollection unpacking
# ---------------------------------------------------------------------------

class TestGeometryCollection:
    """Non-empty GeometryCollections should be recursively unpacked.

    rasterize.py:1995 documents: "GeometryCollection -- recursively
    unpacked".  The fast-path classifier in ``_classify_geometries``
    falls through to the per-geometry GC slow path whenever any element
    is a GeometryCollection, so this path has its own polygon / line /
    point sub-bucketing logic that test_rasterize.py only exercises
    with empty collections (test_unsupported_geom_type_skipped at
    line 269).  A regression in the slow-path classifier (dropping a
    geometry type, mis-counting indices) would ship undetected.
    """

    @staticmethod
    def _mixed_collection():
        """Polygon + Point inside a single GeometryCollection."""
        return GeometryCollection([box(0, 0, 5, 5), Point(7.5, 7.5)])

    def test_polygon_and_point_in_collection_eager(self):
        """Both the polygon and the point inside the GC are burned."""
        gc = self._mixed_collection()
        r = rasterize([(gc, 1.0)], width=10, height=10,
                      bounds=(0, 0, 10, 10), fill=0)
        vals = r.values
        # Polygon covers rows 5..9, cols 0..4 -> 25 pixels.
        # Point at (7.5, 7.5) -> one additional pixel.
        assert int((vals == 1.0).sum()) == 26
        # The point pixel is in the upper-right quadrant (y descends).
        # Row 2 (y in [7, 8]), col 7 (x in [7, 8]).
        assert vals[2, 7] == 1.0
        # The polygon pixel sample at (2.5, 2.5) -> row 7 col 2.
        assert vals[7, 2] == 1.0

    def test_polygon_line_point_in_collection(self):
        """All three primitive types inside a single GC are rasterized.

        Uses a 45-degree diagonal line so the Bresenham branch actually
        steps in both axes (a horizontal/vertical line is the trivial
        degenerate case).
        """
        gc = GeometryCollection([
            box(0, 0, 4, 4),
            # Diagonal from (col=5, row=4) to (col=9, row=0) inclusive:
            # Bresenham steps (5,4) (6,3) (7,2) (8,1) (9,0).
            LineString([(5.5, 5.5), (9.5, 9.5)]),
            Point(7.5, 8.5),
        ])
        r = rasterize([(gc, 1.0)], width=10, height=10,
                      bounds=(0, 0, 10, 10), fill=0)
        vals = r.values
        # Polygon: 16 cells (4x4).  Line: 5 cells along the diagonal.
        # Point: 1.  No overlaps.
        assert int((vals == 1.0).sum()) == 16 + 5 + 1
        # Specific spot checks.
        assert vals[8, 2] == 1.0  # polygon interior
        # Mid-line cell: row=3, col=6 -- only the diagonal Bresenham
        # branch can light this exact cell.
        assert vals[3, 6] == 1.0
        assert vals[1, 7] == 1.0  # point cell

    @skip_no_cuda
    def test_collection_eager_cupy_matches_numpy(self):
        """GeometryCollection unpacking is identical on cupy."""
        gc = self._mixed_collection()
        np_r = rasterize([(gc, 1.0)], width=10, height=10,
                         bounds=(0, 0, 10, 10), fill=0)
        cp_r = rasterize([(gc, 1.0)], width=10, height=10,
                         bounds=(0, 0, 10, 10), fill=0, use_cuda=True)
        # 25 polygon cells + 1 point cell = 26 (eager case pins this too).
        assert int((_as_numpy(cp_r) == 1.0).sum()) == 26
        np.testing.assert_array_equal(np_r.values, _as_numpy(cp_r))

    @skip_no_dask
    def test_collection_dask_numpy_matches_numpy(self):
        """GeometryCollection unpacking is identical on dask+numpy."""
        gc = self._mixed_collection()
        np_r = rasterize([(gc, 1.0)], width=10, height=10,
                         bounds=(0, 0, 10, 10), fill=0)
        dk_r = rasterize([(gc, 1.0)], width=10, height=10,
                         bounds=(0, 0, 10, 10), fill=0, chunks=(5, 5))
        assert int((_as_numpy(dk_r) == 1.0).sum()) == 26
        np.testing.assert_array_equal(np_r.values, _as_numpy(dk_r))

    @skip_no_cuda
    @skip_no_dask
    def test_collection_dask_cupy_matches_numpy(self):
        """GeometryCollection unpacking is identical on dask+cupy."""
        gc = self._mixed_collection()
        np_r = rasterize([(gc, 1.0)], width=10, height=10,
                         bounds=(0, 0, 10, 10), fill=0)
        dkcp_r = rasterize([(gc, 1.0)], width=10, height=10,
                           bounds=(0, 0, 10, 10), fill=0,
                           chunks=(5, 5), use_cuda=True)
        assert int((_as_numpy(dkcp_r) == 1.0).sum()) == 26
        np.testing.assert_array_equal(np_r.values, _as_numpy(dkcp_r))


# ---------------------------------------------------------------------------
# Cat 1 MEDIUM -- eager cupy ``all_touched=True``
# ---------------------------------------------------------------------------

class TestEagerCupyAllTouched:
    """``all_touched=True`` switches polygons to a different inclusion rule.

    test_rasterize.py covers all_touched on the eager numpy backend
    (test_all_touched_fills_more_pixels at line 351) and the dask+cupy
    backend (test_all_touched_parity at line 1369), but skips the eager
    cupy path which invokes the GPU all_touched kernel directly.  This
    test pins eager-cupy/eager-numpy parity for that mode.
    """

    @skip_no_cuda
    def test_eager_cupy_all_touched_matches_numpy(self):
        # A tiny 0.2x0.2 polygon straddling pixel-centre boundaries on a
        # 5x5 grid: with all_touched=False the centre-test misses every
        # cell, with all_touched=True the kernel picks up the four cells
        # whose corners the polygon overlaps.  Eager cupy must match
        # eager numpy on both kernels.
        geom = box(1.9, 1.9, 2.1, 2.1)
        np_r = rasterize([(geom, 1.0)], width=5, height=5,
                         bounds=(0, 0, 5, 5), fill=0,
                         all_touched=True)
        cp_r = rasterize([(geom, 1.0)], width=5, height=5,
                         bounds=(0, 0, 5, 5), fill=0,
                         all_touched=True, use_cuda=True)
        np.testing.assert_array_equal(np_r.values, _as_numpy(cp_r))
        # Sanity: the touched mode lights the four corner cells.
        assert int((np_r.values == 1.0).sum()) == 4

    @skip_no_cuda
    def test_eager_cupy_all_touched_superset_of_default(self):
        """all_touched=True burns >= the cells that all_touched=False burns."""
        # Small fractional polygon -- default centre-test fills zero
        # cells, all_touched fills the four cells whose corners the
        # polygon overlaps.
        geom = box(1.9, 1.9, 2.1, 2.1)
        cp_default = rasterize([(geom, 1.0)], width=5, height=5,
                               bounds=(0, 0, 5, 5), fill=0,
                               use_cuda=True)
        cp_touched = rasterize([(geom, 1.0)], width=5, height=5,
                               bounds=(0, 0, 5, 5), fill=0,
                               all_touched=True, use_cuda=True)
        default_mask = (_as_numpy(cp_default) == 1.0)
        touched_mask = (_as_numpy(cp_touched) == 1.0)
        # all_touched must fill everywhere the default mode filled.
        assert np.all(touched_mask[default_mask])
        # And strictly more, given the centre-miss polygon.
        assert touched_mask.sum() > default_mask.sum()


# ---------------------------------------------------------------------------
# Cat 2 MEDIUM -- integer dtype with the default NaN fill
# ---------------------------------------------------------------------------

class TestIntegerDtypeNanFill:
    """Pin the observed behaviour when ``dtype`` is integer but ``fill``
    defaults to ``np.nan``.

    Scope: numpy backend only.  ``np.full((H, W), np.nan).astype(np.int32)``
    silently casts NaN to a platform-dependent sentinel: x86 yields
    ``INT32_MIN`` while Apple Silicon yields ``0``.  Both values are
    unspecified by C and by numpy, so the test pins "rasterize emits the
    same cast numpy emits" rather than a specific number.  The cupy and
    dask+cupy backends allocate their own backing arrays and the
    CUDA-side NaN-to-int cast may differ from numpy's by CUDA version;
    a cross-backend parametrization is deferred to a follow-up sweep
    that can investigate per-backend cast semantics.  This is
    undocumented but must remain stable on the numpy backend: a future
    refactor that switched to raising
    ``ValueError("integer dtype requires explicit fill")`` would break
    every caller that currently passes ``dtype=np.int32`` without
    overriding ``fill``.  Pin the cast so the choice is visible as a
    code-review diff.
    """

    def test_int32_dtype_with_default_nan_fill_pins_sentinel(self):
        """NaN fill on int32 dtype takes numpy's platform NaN-cast."""
        r = rasterize([(box(0, 0, 3, 3), 7.0)],
                      width=5, height=5, bounds=(0, 0, 5, 5),
                      dtype=np.int32)
        assert r.dtype == np.int32
        # Derive the sentinel from numpy itself: whatever the platform
        # produces when casting NaN to int32 is what rasterize must
        # produce too.  x86 -> INT32_MIN, Apple Silicon -> 0.
        with np.errstate(invalid="ignore"):
            sentinel = np.array([np.nan], dtype=np.float64).astype(np.int32)[0]
        # Lower-left quadrant covered by polygon.
        assert r.values[4, 0] == 7
        # Outside the polygon (top-right corner) takes the platform NaN-cast.
        assert r.values[0, 4] == sentinel

    def test_int32_dtype_with_explicit_int_fill(self):
        """Explicit int fill is honoured exactly (no NaN cast surprise)."""
        r = rasterize([(box(0, 0, 3, 3), 7.0)],
                      width=5, height=5, bounds=(0, 0, 5, 5),
                      fill=-1, dtype=np.int32)
        assert r.dtype == np.int32
        assert r.values[4, 0] == 7
        assert r.values[0, 4] == -1
