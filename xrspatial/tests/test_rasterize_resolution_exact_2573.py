"""Regression tests for issue #2573.

Before the fix, ``rasterize(..., resolution=R, bounds=B)`` computed
``width = ceil((xmax-xmin)/R)`` but then rebuilt output coords over the
*original* bounds, so the actual cell size was ``(xmax-xmin)/width`` --
smaller than ``R`` whenever the bounds didn't divide evenly.

The contract is now: the requested resolution is honored exactly.  The
output grid is anchored at ``(xmin, ymax)`` and extends right / down by
exactly ``width * x_res`` and ``height * y_res``, matching
``rasterio.transform.from_origin(west, north, x_res, y_res)``.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from shapely.geometry import box
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
    import dask  # noqa: F401
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


def _cell_size(coords):
    """Compute the cell pitch from a 1D coord vector (uniform spacing)."""
    arr = np.asarray(coords)
    return float(abs(arr[1] - arr[0]))


# ---------------------------------------------------------------------------
# Direct regression: the example in the finding (#2573 body).
# ---------------------------------------------------------------------------

class TestResolutionExactRegression2573:
    """Pin the exact example from the issue: bounds=(0,0,1,1), resolution=0.3
    must produce a 4x4 raster whose actual cell size is 0.3, not 0.25."""

    def test_resolution_0p3_over_unit_bounds(self):
        r = rasterize([(box(0, 0, 1, 1), 1.0)],
                      resolution=0.3, bounds=(0, 0, 1, 1), fill=0)
        # ceil(1 / 0.3) = 4
        assert r.shape == (4, 4)
        # The bug returned 0.25 here.  Tolerance well under any
        # plausible regression: any rebuild over original bounds gives
        # 0.25, and the correct value is 0.3.
        assert _cell_size(r.x) == pytest.approx(0.3, abs=1e-9)
        assert _cell_size(r.y) == pytest.approx(0.3, abs=1e-9)

    def test_resolution_0p3_grid_anchored_at_upper_left(self):
        """xmin / ymax stay put; the grid extends past the original
        xmax / ymin to honor the cell size.  rasterio's
        ``from_origin`` convention."""
        r = rasterize([(box(0, 0, 1, 1), 1.0)],
                      resolution=0.3, bounds=(0, 0, 1, 1), fill=0)
        # First pixel centre is xmin + px/2 = 0 + 0.15 = 0.15.
        assert float(r.x.values[0]) == pytest.approx(0.15, abs=1e-9)
        # Last pixel centre is xmin + (width - 0.5) * px = 3.5 * 0.3 = 1.05.
        assert float(r.x.values[-1]) == pytest.approx(1.05, abs=1e-9)
        # Top row centre is ymax - py/2 = 1 - 0.15 = 0.85.
        assert float(r.y.values[0]) == pytest.approx(0.85, abs=1e-9)
        # Bottom row centre is ymax - (height - 0.5) * py = 1 - 1.05 = -0.05.
        assert float(r.y.values[-1]) == pytest.approx(-0.05, abs=1e-9)


# ---------------------------------------------------------------------------
# Contract: resolution is honored exactly for divides-evenly and
# doesn't-divide-evenly cases alike.
# ---------------------------------------------------------------------------

class TestResolutionHonoredExactly:
    """The new ``resolution=`` contract: cell size matches the request."""

    def test_divides_evenly_unchanged(self):
        """bounds=(0,0,5,5), resolution=1.0 still gives a 5x5 raster with
        cell size 1.0 and original bounds (no overshoot).  This pins the
        no-regression case for the existing happy path."""
        r = rasterize([(box(0, 0, 5, 5), 1.0)],
                      resolution=1.0, bounds=(0, 0, 5, 5), fill=0)
        assert r.shape == (5, 5)
        assert _cell_size(r.x) == pytest.approx(1.0, abs=1e-12)
        assert _cell_size(r.y) == pytest.approx(1.0, abs=1e-12)
        # First / last centre lines: original bounds preserved.
        assert float(r.x.values[0]) == pytest.approx(0.5, abs=1e-12)
        assert float(r.x.values[-1]) == pytest.approx(4.5, abs=1e-12)

    def test_does_not_divide_evenly_cell_size_exact(self):
        """The headline case: a resolution that doesn't divide the bounds
        evenly still gets the requested cell size."""
        # bounds extent 3.5, resolution 1.5 -> ceil(3.5/1.5) = 3 cells,
        # grid extends to xmin + 3*1.5 = 4.5 (was 3.5).
        r = rasterize([(box(0, 0, 3, 3), 1.0)],
                      resolution=1.5, bounds=(0, 0, 3.5, 3.5), fill=0)
        assert r.shape == (3, 3)
        assert _cell_size(r.x) == pytest.approx(1.5, abs=1e-12)
        assert _cell_size(r.y) == pytest.approx(1.5, abs=1e-12)

    def test_asymmetric_resolution_tuple(self):
        """A (x_res, y_res) tuple with non-divisible bounds still honors
        both axes exactly."""
        # x: extent 10, x_res 3 -> ceil(10/3) = 4 cols, x grid extends to 12.
        # y: extent  8, y_res 3 -> ceil( 8/3) = 3 rows, y grid extends to 9.
        r = rasterize([(box(0, 0, 10, 8), 1.0)],
                      resolution=(3.0, 3.0),
                      bounds=(0, 0, 10, 8), fill=0)
        assert r.shape == (3, 4)
        assert _cell_size(r.x) == pytest.approx(3.0, abs=1e-12)
        assert _cell_size(r.y) == pytest.approx(3.0, abs=1e-12)

    def test_burn_value_within_original_bounds_preserved(self):
        """The expanded grid must still rasterize geometries inside the
        original bounds correctly.  A unit square at (0,0)-(1,1) under
        bounds=(0,0,1,1), resolution=0.3 should burn the four cells whose
        centres fall inside it (the centres at x=0.15, 0.45, 0.75 and
        y=0.85, 0.55, 0.25 are inside; the cells at x=1.05 and y=-0.05
        are outside the polygon and stay at fill).
        """
        r = rasterize([(box(0, 0, 1, 1), 1.0)],
                      resolution=0.3, bounds=(0, 0, 1, 1), fill=0)
        vals = r.values
        # 3x3 block of burned cells in the top-left, surrounded by fill.
        assert (vals[:3, :3] == 1.0).all()
        # Outside-original-bounds cells stay at fill.
        assert (vals[3, :] == 0.0).all()  # row below original ymin
        assert (vals[:, 3] == 0.0).all()  # col past original xmax

    def test_get_dataarray_resolution_matches_request(self):
        """Downstream tools that call ``get_dataarray_resolution`` on the
        result now read back the cell size the caller asked for.  Before
        the fix this was the shrunk value computed from the original
        bounds (0.25 for the 4x4 / 0.3 case), which silently poisoned
        slope / aspect / proximity callers that prefer the coord-derived
        cellsize."""
        from xrspatial.utils import get_dataarray_resolution
        r = rasterize([(box(0, 0, 1, 1), 1.0)],
                      resolution=0.3, bounds=(0, 0, 1, 1), fill=0)
        cx, cy = get_dataarray_resolution(r)
        assert cx == pytest.approx(0.3, abs=1e-9)
        assert cy == pytest.approx(0.3, abs=1e-9)


# ---------------------------------------------------------------------------
# ``like=`` + ``resolution=`` with a template whose bounds don't divide
# evenly by the resolution.  The resolution branch runs after the
# like-bounds resolution, so the fix should still expand the grid.
# ---------------------------------------------------------------------------

class TestLikePlusResolutionOvershoot:
    """``like=`` + ``resolution=`` with non-divisible like-bounds still
    honors the requested cell size.  The existing
    ``test_resolution_override_strips_stale_grid_attrs`` in test_rasterize.py
    uses ``resolution=0.5`` against a 10x10 template (divides evenly), so
    the overshoot branch wasn't pinned for the like path."""

    @staticmethod
    def _template():
        # 10x10 raster on bounds (0, 0, 10, 10).  Cellsize 1.0.
        x = np.linspace(0.5, 9.5, 10)
        y = np.linspace(9.5, 0.5, 10)
        import xarray as xr
        return xr.DataArray(
            np.zeros((10, 10), dtype=np.float64),
            dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={
                'crs': 'EPSG:32610',
                'res': (1.0, 1.0),
                'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
            },
        )

    def test_like_plus_non_divisible_resolution_cell_size_exact(self):
        like = self._template()
        r = rasterize(
            [(box(2, 2, 8, 8), 1.0)],
            like=like, resolution=0.3, fill=0,
        )
        # ceil(10 / 0.3) = 34 cells.
        assert r.shape == (34, 34)
        assert _cell_size(r.x) == pytest.approx(0.3, abs=1e-9)
        assert _cell_size(r.y) == pytest.approx(0.3, abs=1e-9)


# ---------------------------------------------------------------------------
# Inferred-bounds + resolution: when ``bounds=`` is omitted and resolved
# from the geometry envelope, ``total_bounds`` is typically non-divisible.
# The fix should still honor the requested cell size.
# ---------------------------------------------------------------------------

class TestInferredBoundsPlusResolution:
    """No explicit ``bounds=`` plus ``resolution=``: geometry envelope
    drives final_bounds, then the fix extends it to honor the cell size."""

    def test_inferred_bounds_non_divisible_resolution(self):
        # Box extent 1.0 in each axis, resolution 0.3 -> ceil(1/0.3) = 4 cells.
        # No explicit bounds -- inferred from the geom's bbox (0..1).
        r = rasterize([(box(0, 0, 1, 1), 1.0)],
                      resolution=0.3, fill=0)
        assert r.shape == (4, 4)
        assert _cell_size(r.x) == pytest.approx(0.3, abs=1e-9)
        assert _cell_size(r.y) == pytest.approx(0.3, abs=1e-9)


# ---------------------------------------------------------------------------
# Cross-backend parity: the new contract holds on cupy and dask backends
# too, since the fix lives upstream of backend dispatch.
# ---------------------------------------------------------------------------

class TestResolutionExactAcrossBackends:
    """The exact-resolution contract holds for every backend."""

    @skip_no_cuda
    def test_cupy_matches_numpy(self):
        np_r = rasterize([(box(0, 0, 1, 1), 1.0)],
                         resolution=0.3, bounds=(0, 0, 1, 1), fill=0)
        cp_r = rasterize([(box(0, 0, 1, 1), 1.0)],
                         resolution=0.3, bounds=(0, 0, 1, 1), fill=0,
                         gpu=True)
        assert cp_r.shape == np_r.shape == (4, 4)
        assert _cell_size(cp_r.x) == pytest.approx(0.3, abs=1e-9)
        cp_vals = cupy.asnumpy(cp_r.data)
        np.testing.assert_array_equal(np_r.values, cp_vals)

    @skip_no_dask
    def test_dask_numpy_matches_numpy(self):
        np_r = rasterize([(box(0, 0, 1, 1), 1.0)],
                         resolution=0.3, bounds=(0, 0, 1, 1), fill=0)
        dk_r = rasterize([(box(0, 0, 1, 1), 1.0)],
                         resolution=0.3, bounds=(0, 0, 1, 1), fill=0,
                         chunks=(2, 2))
        assert dk_r.shape == np_r.shape == (4, 4)
        assert _cell_size(dk_r.x) == pytest.approx(0.3, abs=1e-9)
        np.testing.assert_array_equal(np_r.values, dk_r.data.compute())

    @skip_no_cuda
    @skip_no_dask
    def test_dask_cupy_matches_numpy(self):
        np_r = rasterize([(box(0, 0, 1, 1), 1.0)],
                         resolution=0.3, bounds=(0, 0, 1, 1), fill=0)
        dkcp_r = rasterize([(box(0, 0, 1, 1), 1.0)],
                           resolution=0.3, bounds=(0, 0, 1, 1), fill=0,
                           chunks=(2, 2), gpu=True)
        assert dkcp_r.shape == np_r.shape == (4, 4)
        assert _cell_size(dkcp_r.x) == pytest.approx(0.3, abs=1e-9)
        np.testing.assert_array_equal(
            np_r.values, cupy.asnumpy(dkcp_r.data.compute()))
