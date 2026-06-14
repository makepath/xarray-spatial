"""Tests for xrspatial.reproject module."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import has_cuda_and_cupy

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    import dask.array as da
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

try:
    import cupy as cp
except ImportError:
    cp = None

# Gate GPU tests on a real CUDA runtime probe, not just that ``cupy`` imports.
# An import-only check leaves tests erroring with ``cudaErrorInsufficientDriver``
# on hosts where ``cupy`` is installed but the driver is missing or too old.
HAS_CUPY = has_cuda_and_cupy()

pytestmark = pytest.mark.skipif(
    not HAS_PYPROJ, reason="pyproj required for reproject tests"
)

# WGS84 constants for projection round-trip tests
_WGS84_E2 = 2.0 * (1.0 / 298.257223563) - (1.0 / 298.257223563) ** 2
_WGS84_A = 6378137.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raster(data, crs='EPSG:4326', x_range=(-1, 1), y_range=(-1, 1),
                 nodata=np.nan, name='test'):
    """Create a test DataArray with geographic coordinates and CRS metadata."""
    h, w = data.shape
    y = np.linspace(y_range[1], y_range[0], h)   # north-up (descending)
    x = np.linspace(x_range[0], x_range[1], w)
    da_obj = xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name=name,
        attrs={'crs': crs, 'nodata': nodata},
    )
    return da_obj


def _gradient_raster(h=64, w=64, crs='EPSG:4326',
                     x_range=(-10, 10), y_range=(-10, 10)):
    """Raster with values equal to x + y (easy to verify after transform)."""
    y = np.linspace(y_range[1], y_range[0], h)
    x = np.linspace(x_range[0], x_range[1], w)
    xx, yy = np.meshgrid(x, y)
    data = (xx + yy).astype(np.float64)
    return _make_raster(data, crs=crs, x_range=x_range, y_range=y_range)


def _pyproj_geoid_probe_is_usable(probe, zero_tol=1.0):
    """Return True if a pyproj vertical-transform probe value indicates the
    geoid grid is actually installed and usable for a cross-check.

    pyproj has two failure modes when the EGM96 grid is missing and PROJ
    network access is disabled:

    - it silently returns the input ellipsoidal height unchanged (so the
      probe comes back as ~0.0 at a well-known sample point where the
      true geoid undulation is tens of metres), or
    - it returns a non-finite sentinel (``-inf``, ``+inf``, ``nan``).

    Treat both as "grid unavailable" so the caller skips the cross-check
    instead of asserting against the local lookup.
    """
    if not np.isfinite(probe):
        return False
    if abs(probe) < zero_tol:
        return False
    return True


# ---------------------------------------------------------------------------
# CRS utils
# ---------------------------------------------------------------------------

class TestCrsUtils:
    def test_require_pyproj(self):
        from xrspatial.reproject._crs_utils import _require_pyproj
        mod = _require_pyproj()
        assert hasattr(mod, 'CRS')

    def test_resolve_crs_none(self):
        from xrspatial.reproject._crs_utils import _resolve_crs
        assert _resolve_crs(None) is None

    def test_resolve_crs_epsg_string(self):
        from xrspatial.reproject._crs_utils import _resolve_crs
        crs = _resolve_crs('EPSG:4326')
        assert crs is not None
        assert crs.to_epsg() == 4326

    def test_resolve_crs_epsg_int(self):
        from xrspatial.reproject._crs_utils import _resolve_crs
        crs = _resolve_crs(4326)
        assert crs.to_epsg() == 4326

    def test_detect_source_crs_from_attrs(self):
        from xrspatial.reproject._crs_utils import _detect_source_crs
        raster = _make_raster(np.zeros((4, 4)), crs='EPSG:4326')
        crs = _detect_source_crs(raster)
        assert crs is not None
        assert crs.to_epsg() == 4326

    def test_detect_source_crs_none(self):
        from xrspatial.reproject._crs_utils import _detect_source_crs
        raster = xr.DataArray(np.zeros((4, 4)), dims=['y', 'x'])
        crs = _detect_source_crs(raster)
        assert crs is None

    def test_detect_nodata_explicit(self):
        from xrspatial.reproject._crs_utils import _detect_nodata
        raster = _make_raster(np.zeros((4, 4)))
        assert _detect_nodata(raster, nodata=-9999) == -9999.0

    def test_detect_nodata_from_attrs(self):
        from xrspatial.reproject._crs_utils import _detect_nodata
        raster = _make_raster(np.zeros((4, 4)), nodata=-1)
        val = _detect_nodata(raster)
        assert val == -1.0


class TestDetectNodataDtypeRange:
    """Regression coverage for #2572.

    Explicit out-of-range nodata used to silently wrap during the
    worker's cast-back step (e.g. ``-9999`` in a ``uint8`` array landed
    at ``0``) while ``attrs['nodata']`` kept advertising the original
    value. ``_detect_nodata`` must reject the explicit case and warn
    on the attrs-derived case.
    """

    def test_explicit_negative_nodata_for_uint8_raises(self):
        from xrspatial.reproject._crs_utils import _detect_nodata
        raster = _make_raster(np.zeros((4, 4), dtype=np.uint8))
        with pytest.raises(ValueError, match="uint8"):
            _detect_nodata(raster, nodata=-9999, dtype=np.uint8)

    def test_explicit_too_large_nodata_for_uint16_raises(self):
        from xrspatial.reproject._crs_utils import _detect_nodata
        raster = _make_raster(np.zeros((4, 4), dtype=np.uint16))
        with pytest.raises(ValueError, match="uint16"):
            _detect_nodata(raster, nodata=70000, dtype=np.uint16)

    def test_explicit_in_range_nodata_passes(self):
        """Boundary case: dtype.max stays untouched."""
        from xrspatial.reproject._crs_utils import _detect_nodata
        raster = _make_raster(np.zeros((4, 4), dtype=np.uint8))
        assert _detect_nodata(raster, nodata=255, dtype=np.uint8) == 255.0

    def test_explicit_in_range_signed_min(self):
        from xrspatial.reproject._crs_utils import _detect_nodata
        raster = _make_raster(np.zeros((4, 4), dtype=np.int16))
        assert _detect_nodata(raster, nodata=-32768, dtype=np.int16) == -32768.0

    def test_attrs_out_of_range_warns_and_falls_back(self):
        """Legacy files (uint16 + nodata=-9999) should warn, not crash."""
        from xrspatial.reproject._crs_utils import _detect_nodata
        # nodata in attrs is out of range for uint16
        raster = _make_raster(np.zeros((4, 4), dtype=np.uint16), nodata=-9999)
        with pytest.warns(UserWarning, match="uint16"):
            val = _detect_nodata(raster, dtype=np.uint16)
        # Falls back to dtype.max for unsigned
        assert val == float(np.iinfo(np.uint16).max)


class TestReprojectIntegerNodataDtypeRange:
    """End-to-end regression coverage for #2572 through ``reproject()``."""

    def test_reproject_uint8_negative_nodata_raises(self):
        from xrspatial.reproject import reproject
        arr = (np.ones((4, 4), dtype=np.uint8) * 10)
        da_obj = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.linspace(40, 30, 4),
                    'x': np.linspace(-5, 5, 4)},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match="uint8"):
            reproject(da_obj, 'EPSG:4326', nodata=-9999)

    def test_reproject_uint16_too_large_nodata_raises(self):
        from xrspatial.reproject import reproject
        arr = (np.ones((4, 4), dtype=np.uint16) * 10)
        da_obj = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.linspace(40, 30, 4),
                    'x': np.linspace(-5, 5, 4)},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match="uint16"):
            reproject(da_obj, 'EPSG:4326', nodata=70000)

    def test_reproject_uint8_in_range_nodata_writes_correct_pixels(self):
        """Happy path: representable nodata produces non-corrupted output
        where unfilled pixels carry the declared sentinel.
        """
        from xrspatial.reproject import reproject
        arr = (np.ones((4, 4), dtype=np.uint8) * 10)
        da_obj = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.linspace(40, 30, 4),
                    'x': np.linspace(-5, 5, 4)},
            attrs={'crs': 'EPSG:4326'},
        )
        # Expand bounds so the output has nodata around the edges.
        out = reproject(
            da_obj, 'EPSG:4326', nodata=255,
            bounds=(-20, 20, 20, 50),
        )
        assert out.dtype == np.uint8
        assert out.attrs['nodata'] == 255.0
        # The sentinel actually appears in the array (no silent wrap).
        n_nodata = int((out.values == 255).sum())
        assert n_nodata > 0
        # The entire top row sits above the source extent (y=40 is the
        # source max), so it must be all nodata, not a mix of 0 and 255.
        assert (out.values[0, :] == 255).all(), (
            f"top row should be all nodata=255, got {out.values[0, :]}"
        )

    def test_reproject_int16_negative_nodata_works(self):
        from xrspatial.reproject import reproject
        arr = (np.ones((4, 4), dtype=np.int16) * 10)
        da_obj = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.linspace(40, 30, 4),
                    'x': np.linspace(-5, 5, 4)},
            attrs={'crs': 'EPSG:4326'},
        )
        out = reproject(
            da_obj, 'EPSG:4326', nodata=-32768,
            bounds=(-20, 20, 20, 50),
        )
        assert out.dtype == np.int16
        assert out.attrs['nodata'] == -32768.0
        assert (out.values == -32768).any()


# ---------------------------------------------------------------------------
# ApproximateTransform
# ---------------------------------------------------------------------------

class TestApproximateTransform:
    def test_identity_transform(self):
        """Control grid for same-CRS should have near-zero error."""
        from xrspatial.reproject._transform import ApproximateTransform

        transformer = pyproj.Transformer.from_crs(
            'EPSG:4326', 'EPSG:4326', always_xy=True
        )
        approx = ApproximateTransform(
            transformer,
            out_bounds=(-10, -10, 10, 10),
            out_shape=(100, 100),
            precision=16,
        )
        err = approx.max_error_estimate()
        assert err < 1e-6

    def test_4326_to_3857(self):
        """Approx error should be < 0.1 source pixels for a typical reproject."""
        from xrspatial.reproject._transform import ApproximateTransform

        transformer = pyproj.Transformer.from_crs(
            'EPSG:3857', 'EPSG:4326', always_xy=True
        )
        # A Web Mercator chunk around 0,0
        bounds = (-100000, -100000, 100000, 100000)
        shape = (512, 512)
        approx = ApproximateTransform(
            transformer, out_bounds=bounds, out_shape=shape, precision=16,
        )
        err = approx.max_error_estimate()
        # Error should be very small for this smooth transform
        assert err < 0.5, f"Approx error too large: {err}"

    def test_interpolation_shape(self):
        from xrspatial.reproject._transform import ApproximateTransform

        transformer = pyproj.Transformer.from_crs(
            'EPSG:4326', 'EPSG:4326', always_xy=True
        )
        approx = ApproximateTransform(
            transformer,
            out_bounds=(0, 0, 1, 1),
            out_shape=(50, 60),
            precision=8,
        )
        rows = np.arange(50, dtype=np.float64)
        cols = np.arange(60, dtype=np.float64)
        cc, rr = np.meshgrid(cols, rows)
        src_y, src_x = approx(rr, cc)
        assert src_y.shape == (50, 60)
        assert src_x.shape == (50, 60)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_resample_nearest(self):
        from xrspatial.reproject._interpolate import _resample_numpy
        src = np.array([[1, 2], [3, 4]], dtype=np.float64)
        rows = np.array([[0.1, 0.1], [0.9, 0.9]])
        cols = np.array([[0.1, 0.9], [0.1, 0.9]])
        result = _resample_numpy(src, rows, cols, resampling='nearest')
        expected = np.array([[1, 2], [3, 4]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_resample_bilinear(self):
        from xrspatial.reproject._interpolate import _resample_numpy
        src = np.array([[0, 10], [0, 10]], dtype=np.float64)
        rows = np.array([[0.5]])
        cols = np.array([[0.5]])
        result = _resample_numpy(src, rows, cols, resampling='bilinear')
        assert abs(result[0, 0] - 5.0) < 0.5

    def test_resample_oob_fills_nodata(self):
        from xrspatial.reproject._interpolate import _resample_numpy
        src = np.ones((4, 4), dtype=np.float64)
        rows = np.array([[-5.0]])
        cols = np.array([[0.0]])
        result = _resample_numpy(src, rows, cols, nodata=-999)
        assert result[0, 0] == -999

    def test_nearest_negative_rounding(self):
        """int(r + 0.5) must round toward -inf, not toward zero (#1086)."""
        from xrspatial.reproject._interpolate import _resample_numpy
        src = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
        # r = -0.6 is beyond the half-pixel boundary of pixel 0 -> nodata
        rows = np.array([[-0.6]])
        cols = np.array([[1.0]])
        result = _resample_numpy(src, rows, cols, resampling='nearest', nodata=-999)
        assert result[0, 0] == -999, (
            f"r=-0.6 should be nodata, got {result[0, 0]}"
        )
        # r = -0.4 is within pixel 0's domain -> pixel 0
        rows2 = np.array([[-0.4]])
        result2 = _resample_numpy(src, rows2, cols, resampling='nearest', nodata=-999)
        assert result2[0, 0] == src[0, 1], (
            f"r=-0.4 should map to pixel 0, got {result2[0, 0]}"
        )
        # r = -0.5 is exactly on the half-pixel boundary: floor(-0.5+0.5)=0 -> pixel 0
        rows3 = np.array([[-0.5]])
        result3 = _resample_numpy(src, rows3, cols, resampling='nearest', nodata=-999)
        assert result3[0, 0] == src[0, 1], (
            f"r=-0.5 should map to pixel 0, got {result3[0, 0]}"
        )

    def test_cubic_oob_fallback(self):
        """Cubic must fall back to bilinear when stencil extends outside source (#1086)."""
        from xrspatial.reproject._interpolate import _resample_numpy
        # 6x6 source with a gradient
        src = np.arange(36, dtype=np.float64).reshape(6, 6)
        # Query at r=0.5, c=0.5: cubic stencil needs row -1, which is OOB.
        # Should fall back to bilinear using pixels (0,0),(0,1),(1,0),(1,1).
        rows = np.array([[0.5]])
        cols = np.array([[0.5]])
        cubic_result = _resample_numpy(src, rows, cols, resampling='cubic', nodata=-999)
        bilinear_result = _resample_numpy(src, rows, cols, resampling='bilinear', nodata=-999)
        # At the boundary, cubic should produce the same result as bilinear
        np.testing.assert_allclose(
            cubic_result, bilinear_result, atol=1e-10,
            err_msg="Cubic near boundary should fall back to bilinear"
        )
        # Interior query at r=2.5, c=2.5: full stencil fits, cubic should differ from bilinear
        rows_int = np.array([[2.5]])
        cols_int = np.array([[2.5]])
        cubic_int = _resample_numpy(src, rows_int, cols_int, resampling='cubic', nodata=-999)
        bilinear_int = _resample_numpy(src, rows_int, cols_int, resampling='bilinear', nodata=-999)
        # For a linear gradient, cubic and bilinear should agree closely
        # but the point is the code path exercises the non-fallback branch
        assert cubic_int[0, 0] != -999

    def test_cubic_oob_fallback_far_edge(self):
        """Cubic at bottom-right boundary: stencil needs row sh, same fallback (#1086)."""
        from xrspatial.reproject._interpolate import _resample_numpy
        src = np.arange(36, dtype=np.float64).reshape(6, 6)
        # r=4.5: cubic stencil needs row 6 (= sh), which is OOB
        rows = np.array([[4.5]])
        cols = np.array([[4.5]])
        cubic = _resample_numpy(src, rows, cols, resampling='cubic', nodata=-999)
        bilinear = _resample_numpy(src, rows, cols, resampling='bilinear', nodata=-999)
        np.testing.assert_allclose(cubic, bilinear, atol=1e-10)

    def test_cubic_oob_bilinear_fallback_renormalizes(self):
        """Cubic at (-0.8,-0.8): stencil OOB triggers bilinear, which
        finds pixel (0,0) as the only valid neighbor and returns it (#1086)."""
        from xrspatial.reproject._interpolate import _resample_numpy
        src = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
        rows = np.array([[-0.8]])
        cols = np.array([[-0.8]])
        result = _resample_numpy(src, rows, cols, resampling='cubic', nodata=-999)
        # bilinear fallback: r0=-1 (OOB), r1=0, c0=-1 (OOB), c1=0
        # only (r1,c1)=(0,0) is valid -> returns src[0,0]=1.0
        assert result[0, 0] == 1.0

    def test_invalid_resampling(self):
        from xrspatial.reproject._interpolate import _validate_resampling
        with pytest.raises(ValueError, match="resampling"):
            _validate_resampling('lanczos')


# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------

class TestGrid:
    def test_compute_output_grid_identity(self):
        from xrspatial.reproject._grid import _compute_output_grid
        crs = pyproj.CRS('EPSG:4326')
        grid = _compute_output_grid(
            source_bounds=(-10, -10, 10, 10),
            source_shape=(100, 100),
            source_crs=crs,
            target_crs=crs,
        )
        assert grid['shape'][0] > 0
        assert grid['shape'][1] > 0
        left, bottom, right, top = grid['bounds']
        assert left < right
        assert bottom < top

    def test_explicit_resolution(self):
        from xrspatial.reproject._grid import _compute_output_grid
        crs = pyproj.CRS('EPSG:4326')
        grid = _compute_output_grid(
            source_bounds=(-10, -10, 10, 10),
            source_shape=(100, 100),
            source_crs=crs,
            target_crs=crs,
            resolution=1.0,
        )
        assert abs(grid['res_x'] - 1.0) < 1e-6
        assert abs(grid['res_y'] - 1.0) < 1e-6

    def test_explicit_width_height(self):
        from xrspatial.reproject._grid import _compute_output_grid
        crs = pyproj.CRS('EPSG:4326')
        grid = _compute_output_grid(
            source_bounds=(-10, -10, 10, 10),
            source_shape=(100, 100),
            source_crs=crs,
            target_crs=crs,
            width=50,
            height=50,
        )
        assert grid['shape'] == (50, 50)

    def test_make_output_coords(self):
        from xrspatial.reproject._grid import _make_output_coords
        y, x = _make_output_coords((-10, -10, 10, 10), (20, 20))
        assert len(y) == 20
        assert len(x) == 20
        assert y[0] > y[-1]  # north-up
        assert x[0] < x[-1]

    def test_chunk_layout(self):
        from xrspatial.reproject._grid import _compute_chunk_layout
        rc, cc = _compute_chunk_layout((1000, 1200), 512)
        assert sum(rc) == 1000
        assert sum(cc) == 1200

    def test_chunk_bounds(self):
        from xrspatial.reproject._grid import _chunk_bounds
        cb = _chunk_bounds(
            grid_bounds=(0, 0, 100, 100),
            grid_shape=(100, 100),
            row_start=0, row_end=50,
            col_start=0, col_end=50,
        )
        assert cb == (0, 50, 50, 100)


# ---------------------------------------------------------------------------
# Datum-shift bounds estimation (GH #2649)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj required")
class TestDatumShiftBounds:
    """Output bounds estimation must account for datum shifts.

    The Numba fast path ``transform_points`` runs its projection kernels
    in WGS84 and does not apply a datum shift. The per-pixel data path
    (``try_numba_transform``) does apply a Helmert shift for the same CRS
    pairs, so bounds estimated without the shift disagree with the
    reprojected data. For NAD27 (EPSG:4267) the shift is tens to over a
    hundred metres in CONUS -- many pixels on a high-resolution raster.
    The fix bails datum-shift pairs to pyproj. See GH #2649.
    """

    # CONUS sample points where the NAD27 shift is largest.
    XS = np.array([-105.0, -95.0, -120.0, -80.0])
    YS = np.array([35.0, 45.0, 40.0, 30.0])

    def test_transform_points_bails_for_nad27(self):
        from xrspatial.reproject._projections import transform_points
        src = pyproj.CRS.from_epsg(4267)  # NAD27
        tgt = pyproj.CRS.from_epsg(3857)  # Web Mercator
        assert transform_points(src, tgt, self.XS, self.YS) is None
        # And the reverse direction.
        assert transform_points(tgt, src, self.XS, self.YS) is None

    def test_transform_points_keeps_fast_path_no_shift(self):
        # WGS84 and NAD83 need no shift, so the fast path must stay.
        from xrspatial.reproject._projections import transform_points
        for epsg in (4326, 4269):
            src = pyproj.CRS.from_epsg(epsg)
            tgt = pyproj.CRS.from_epsg(3857)
            fast = transform_points(src, tgt, self.XS, self.YS)
            assert fast is not None
            ref = pyproj.Transformer.from_crs(
                src, tgt, always_xy=True).transform(self.XS, self.YS)
            np.testing.assert_allclose(fast[0], ref[0], atol=1e-3)
            np.testing.assert_allclose(fast[1], ref[1], atol=1e-3)

    def test_boundary_matches_pyproj_for_nad27(self):
        # _transform_boundary must agree with pyproj once the fast path
        # bails. Before the fix it was off by ~45 m in x.
        from xrspatial.reproject._grid import _transform_boundary
        src = pyproj.CRS.from_epsg(4267)
        tgt = pyproj.CRS.from_epsg(3857)
        tx, ty = _transform_boundary(src, tgt, self.XS, self.YS)
        ref_x, ref_y = pyproj.Transformer.from_crs(
            src, tgt, always_xy=True).transform(self.XS, self.YS)
        np.testing.assert_allclose(np.asarray(tx), ref_x, atol=1e-6)
        np.testing.assert_allclose(np.asarray(ty), ref_y, atol=1e-6)

    def test_output_grid_bounds_correct_high_res_nad27(self):
        # End-to-end: a high-resolution NAD27 raster reprojected to Web
        # Mercator. The computed grid bounds must match the datum-shifted
        # corner transform, not the unshifted one. At 1 m resolution the
        # ~45 m shift error would be ~45 pixels.
        from xrspatial.reproject._grid import _compute_output_grid
        src = pyproj.CRS.from_epsg(4267)
        tgt = pyproj.CRS.from_epsg(3857)
        # ~1 arcsec cells near 40N -> sub-30 m pixels (high resolution).
        source_bounds = (-105.0, 39.9, -104.9, 40.0)
        grid = _compute_output_grid(
            source_bounds=source_bounds,
            source_shape=(360, 360),
            source_crs=src,
            target_crs=tgt,
        )
        left, bottom, right, top = grid['bounds']

        # Reference bounds from pyproj corner transform (datum-shifted).
        cx = np.array([source_bounds[0], source_bounds[2],
                       source_bounds[0], source_bounds[2]])
        cy = np.array([source_bounds[1], source_bounds[1],
                       source_bounds[3], source_bounds[3]])
        rx, ry = pyproj.Transformer.from_crs(
            src, tgt, always_xy=True).transform(cx, cy)
        # Bounds enclose the projected corners up to grid snapping
        # (bounds are rounded to an integer number of cells, so each
        # edge can move by up to one resolution). One cell here is
        # ~35 m, well under the ~45 m datum error this guards against.
        res = max(grid['res_x'], grid['res_y'])
        assert left <= rx.min() + res
        assert right >= rx.max() - res
        assert bottom <= ry.min() + res
        assert top >= ry.max() - res
        # Guard against regression to the unshifted bounds: each edge
        # must be closer to the datum-shifted corner than to the
        # unshifted one. The shift (~45 m) exceeds the snapping (~35 m).
        unshifted_x, _ = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(4326), tgt, always_xy=True
        ).transform(cx, cy)
        assert abs(left - rx.min()) < abs(left - unshifted_x.min())
        assert abs(right - rx.max()) < abs(right - unshifted_x.max())


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj required")
class TestDatumProbeNoProjWarning:
    """_get_datum_params probes the datum via crs.to_dict(), which routes
    through pyproj's to_proj4() and warns that a PROJ string drops detail.
    The probe never uses that lossy string, so the warning must not leak to
    callers of reproject(). See GH #3076.
    """

    def test_get_datum_params_silences_proj_warning(self):
        from xrspatial.reproject._projections import _get_datum_params
        # EPSG:3857 round-trips through to_proj4() inside to_dict() and is
        # the CRS that surfaced the warning in the original report.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            _get_datum_params(pyproj.CRS.from_epsg(3857))
        leaked = [w for w in rec
                  if 'lose important projection information' in str(w.message)]
        assert leaked == []

    def test_get_datum_params_still_resolves_known_datum(self):
        # Silencing the warning must not break the datum lookup: NAD27
        # (EPSG:4267) is in the shift table and must still return params.
        from xrspatial.reproject._projections import _get_datum_params
        assert _get_datum_params(pyproj.CRS.from_epsg(4267)) is not None


# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------

class TestMergeStrategies:
    def test_first(self):
        from xrspatial.reproject._merge import _merge_arrays_numpy
        a = np.array([[1, np.nan], [3, 4]])
        b = np.array([[10, 20], [np.nan, 40]])
        result = _merge_arrays_numpy([a, b], np.nan, 'first')
        expected = np.array([[1, 20], [3, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_last(self):
        from xrspatial.reproject._merge import _merge_arrays_numpy
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[10, np.nan], [np.nan, 40]])
        result = _merge_arrays_numpy([a, b], np.nan, 'last')
        expected = np.array([[10, 2], [3, 40]])
        np.testing.assert_array_equal(result, expected)

    def test_mean(self):
        from xrspatial.reproject._merge import _merge_arrays_numpy
        a = np.array([[2.0, np.nan], [6.0, 8.0]])
        b = np.array([[4.0, 10.0], [np.nan, 12.0]])
        result = _merge_arrays_numpy([a, b], np.nan, 'mean')
        assert result[0, 0] == 3.0
        assert result[0, 1] == 10.0
        assert result[1, 0] == 6.0
        assert result[1, 1] == 10.0

    def test_max(self):
        from xrspatial.reproject._merge import _merge_arrays_numpy
        a = np.array([[1.0, 5.0]])
        b = np.array([[3.0, 2.0]])
        result = _merge_arrays_numpy([a, b], np.nan, 'max')
        np.testing.assert_array_equal(result, [[3.0, 5.0]])

    def test_min(self):
        from xrspatial.reproject._merge import _merge_arrays_numpy
        a = np.array([[1.0, 5.0]])
        b = np.array([[3.0, 2.0]])
        result = _merge_arrays_numpy([a, b], np.nan, 'min')
        np.testing.assert_array_equal(result, [[1.0, 2.0]])

    def test_invalid_strategy(self):
        from xrspatial.reproject._merge import _validate_strategy
        with pytest.raises(ValueError, match="strategy"):
            _validate_strategy('median')


# ---------------------------------------------------------------------------
# reproject() end-to-end
# ---------------------------------------------------------------------------

class TestReproject:
    def test_identity_reproject(self):
        """Reproject EPSG:4326 -> EPSG:4326 should preserve values."""
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=32, w=32, x_range=(-5, 5), y_range=(-5, 5))
        result = reproject(raster, 'EPSG:4326', resolution=raster.attrs.get('res'))
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        # Center pixel should be close to 0 (x=0 + y=0)
        cy, cx = result.shape[0] // 2, result.shape[1] // 2
        center_val = float(result.values[cy, cx])
        assert abs(center_val) < 2.0, f"Center value {center_val} too far from 0"

    def test_4326_to_3857(self):
        """Reproject from geographic to Web Mercator."""
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=32, w=32, x_range=(-10, 10), y_range=(-10, 10))
        result = reproject(raster, 'EPSG:3857')
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        # Output should have CRS in attrs
        assert 'crs' in result.attrs

    def test_3857_to_4326(self):
        """Reproject from Web Mercator to geographic."""
        from xrspatial.reproject import reproject

        # Create raster in EPSG:3857
        h, w = 32, 32
        data = np.random.RandomState(42).rand(h, w).astype(np.float64)
        y = np.linspace(1000000, -1000000, h)
        x = np.linspace(-1000000, 1000000, w)
        raster = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:3857'},
        )
        result = reproject(raster, 'EPSG:4326')
        assert result.shape[0] > 0

    def test_explicit_resolution(self):
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=32, w=32)
        result = reproject(raster, 'EPSG:4326', resolution=0.5)
        # With 0.5 degree resolution over -10..10 range -> ~40 pixels
        assert result.shape[0] > 30
        assert result.shape[1] > 30

    def test_explicit_bounds(self):
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=32, w=32)
        result = reproject(
            raster, 'EPSG:4326',
            bounds=(-5, -5, 5, 5), resolution=0.5,
        )
        x = result.coords['x'].values
        y = result.coords['y'].values
        assert float(x[0]) > -5.5
        assert float(x[-1]) < 5.5

    def test_explicit_width_height(self):
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=32, w=32)
        result = reproject(raster, 'EPSG:4326', width=20, height=20)
        assert result.shape == (20, 20)

    def test_nodata_propagation(self):
        from xrspatial.reproject import reproject
        data = np.ones((32, 32), dtype=np.float64)
        data[:, :16] = np.nan
        raster = _make_raster(data, x_range=(-10, 10), y_range=(-10, 10))
        result = reproject(raster, 'EPSG:4326')
        # Some nodata should remain in the output
        assert np.isnan(result.values).any()

    def test_nearest_resampling(self):
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=32, w=32)
        result = reproject(raster, 'EPSG:4326', resampling='nearest')
        assert result.shape[0] > 0

    def test_cubic_resampling(self):
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=32, w=32)
        result = reproject(raster, 'EPSG:4326', resampling='cubic')
        assert result.shape[0] > 0

    def test_invalid_resampling(self):
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=8, w=8)
        with pytest.raises(ValueError, match="resampling"):
            reproject(raster, 'EPSG:4326', resampling='lanczos')

    def test_missing_crs_raises(self):
        from xrspatial.reproject import reproject
        raster = xr.DataArray(
            np.zeros((4, 4)), dims=['y', 'x'],
            coords={'y': [3, 2, 1, 0], 'x': [0, 1, 2, 3]},
        )
        with pytest.raises(ValueError, match="source CRS"):
            reproject(raster, 'EPSG:3857')

    def test_non_dataarray_raises(self):
        from xrspatial.reproject import reproject
        with pytest.raises(TypeError, match="xarray.DataArray"):
            reproject(np.zeros((4, 4)), 'EPSG:4326')

    def test_output_has_crs_attr(self):
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=16, w=16)
        result = reproject(raster, 'EPSG:3857')
        assert 'crs' in result.attrs
        crs_out = pyproj.CRS.from_wkt(result.attrs['crs'])
        assert crs_out.to_epsg() == 3857

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_dask_numpy_backend(self):
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=32, w=32)
        raster.data = da.from_array(raster.values, chunks=(16, 16))
        result = reproject(raster, 'EPSG:4326', chunk_size=16)
        assert isinstance(result.data, da.Array)
        computed = result.compute()
        assert computed.shape[0] > 0

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_dask_lazy_evaluation(self):
        """Verify dask output is lazy (no premature .compute())."""
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=32, w=32)
        raster.data = da.from_array(raster.values, chunks=(16, 16))
        result = reproject(raster, 'EPSG:3857', chunk_size=16)
        assert isinstance(result.data, da.Array)
        # Key count is a proxy for laziness -- graph should exist
        assert len(result.data.__dask_graph__()) > 0

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_dask_matches_numpy(self):
        """Dask+numpy result should match pure numpy result."""
        from xrspatial.reproject import reproject
        raster_np = _gradient_raster(h=32, w=32)
        result_np = reproject(
            raster_np, 'EPSG:4326', resolution=1.0,
        )

        raster_dask = raster_np.copy()
        raster_dask.data = da.from_array(raster_np.values, chunks=(16, 16))
        result_dask = reproject(
            raster_dask, 'EPSG:4326', resolution=1.0,
        ).compute()

        np.testing.assert_allclose(
            result_np.values, result_dask.values,
            rtol=1e-5, atol=1e-5, equal_nan=True,
        )


# ---------------------------------------------------------------------------
# merge() end-to-end
# ---------------------------------------------------------------------------

class TestMerge:
    def test_non_overlapping_merge(self):
        """Two adjacent rasters should merge into a seamless mosaic."""
        from xrspatial.reproject import merge
        left_data = np.ones((16, 16), dtype=np.float64) * 10
        right_data = np.ones((16, 16), dtype=np.float64) * 20
        left_raster = _make_raster(
            left_data, x_range=(-10, 0), y_range=(-5, 5)
        )
        right_raster = _make_raster(
            right_data, x_range=(0, 10), y_range=(-5, 5)
        )
        result = merge([left_raster, right_raster], resolution=1.0)
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        # Left side should have 10, right side should have 20
        vals = result.values
        x = result.coords['x'].values
        left_mask = x < -2
        right_mask = x > 2
        if left_mask.any():
            left_vals = vals[:, left_mask]
            valid = ~np.isnan(left_vals)
            if valid.any():
                assert np.nanmean(left_vals[valid]) > 5

    def test_overlapping_merge_first(self):
        from xrspatial.reproject import merge
        a = _make_raster(
            np.full((16, 16), 10.0), x_range=(-5, 5), y_range=(-5, 5)
        )
        b = _make_raster(
            np.full((16, 16), 20.0), x_range=(-5, 5), y_range=(-5, 5)
        )
        result = merge([a, b], strategy='first', resolution=1.0)
        # First raster wins in the interior (edge pixels may be nodata/0)
        vals = result.values
        interior = vals[2:-2, 2:-2]
        valid = ~np.isnan(interior) & (interior != 0)
        if valid.any():
            np.testing.assert_allclose(interior[valid], 10.0, atol=1.0)

    def test_overlapping_merge_mean(self):
        from xrspatial.reproject import merge
        a = _make_raster(
            np.full((16, 16), 10.0), x_range=(-5, 5), y_range=(-5, 5)
        )
        b = _make_raster(
            np.full((16, 16), 20.0), x_range=(-5, 5), y_range=(-5, 5)
        )
        result = merge([a, b], strategy='mean', resolution=1.0)
        # Interior pixels should be mean of 10 and 20
        vals = result.values
        interior = vals[2:-2, 2:-2]
        valid = ~np.isnan(interior) & (interior != 0)
        if valid.any():
            np.testing.assert_allclose(interior[valid], 15.0, atol=1.0)

    def test_merge_different_crs(self):
        """Merge rasters with different CRS into a common grid."""
        from xrspatial.reproject import merge

        # Raster A in EPSG:4326
        a = _gradient_raster(h=16, w=16, x_range=(-5, 0), y_range=(-5, 5))

        # Raster B in EPSG:3857 (covering roughly 0..5 degrees lon)
        data_b = np.random.RandomState(42).rand(16, 16).astype(np.float64) * 10
        y = np.linspace(500000, -500000, 16)
        x = np.linspace(0, 500000, 16)
        b = xr.DataArray(
            data_b, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:3857'},
        )
        result = merge([a, b], target_crs='EPSG:4326', resolution=1.0)
        assert result.shape[0] > 0
        assert 'crs' in result.attrs

    def test_merge_empty_raises(self):
        from xrspatial.reproject import merge
        with pytest.raises(ValueError, match="empty"):
            merge([])

    def test_merge_invalid_strategy(self):
        from xrspatial.reproject import merge
        raster = _gradient_raster(h=8, w=8)
        with pytest.raises(ValueError, match="strategy"):
            merge([raster], strategy='median')

    def test_merge_strategy_last(self):
        """merge() with strategy='last' uses the last valid value."""
        from xrspatial.reproject import merge
        a = _make_raster(
            np.full((16, 16), 10.0), x_range=(-5, 5), y_range=(-5, 5)
        )
        b = _make_raster(
            np.full((16, 16), 20.0), x_range=(-5, 5), y_range=(-5, 5)
        )
        result = merge([a, b], strategy='last', resolution=1.0)
        vals = result.values
        interior = vals[2:-2, 2:-2]
        valid = ~np.isnan(interior) & (interior != 0)
        if valid.any():
            np.testing.assert_allclose(interior[valid], 20.0, atol=1.0)

    def test_merge_strategy_max(self):
        """merge() with strategy='max' takes the maximum."""
        from xrspatial.reproject import merge
        a = _make_raster(
            np.full((16, 16), 10.0), x_range=(-5, 5), y_range=(-5, 5)
        )
        b = _make_raster(
            np.full((16, 16), 20.0), x_range=(-5, 5), y_range=(-5, 5)
        )
        result = merge([a, b], strategy='max', resolution=1.0)
        vals = result.values
        interior = vals[2:-2, 2:-2]
        valid = ~np.isnan(interior) & (interior != 0)
        if valid.any():
            np.testing.assert_allclose(interior[valid], 20.0, atol=1.0)

    def test_merge_strategy_min(self):
        """merge() with strategy='min' takes the minimum."""
        from xrspatial.reproject import merge
        a = _make_raster(
            np.full((16, 16), 10.0), x_range=(-5, 5), y_range=(-5, 5)
        )
        b = _make_raster(
            np.full((16, 16), 20.0), x_range=(-5, 5), y_range=(-5, 5)
        )
        result = merge([a, b], strategy='min', resolution=1.0)
        vals = result.values
        interior = vals[2:-2, 2:-2]
        valid = ~np.isnan(interior) & (interior != 0)
        if valid.any():
            np.testing.assert_allclose(interior[valid], 10.0, atol=1.0)

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_merge_dask(self):
        from xrspatial.reproject import merge
        a = _make_raster(
            np.full((16, 16), 10.0), x_range=(-10, 0), y_range=(-5, 5)
        )
        b = _make_raster(
            np.full((16, 16), 20.0), x_range=(0, 10), y_range=(-5, 5)
        )
        a.data = da.from_array(a.values, chunks=(8, 8))
        b.data = da.from_array(b.values, chunks=(8, 8))
        result = merge([a, b], resolution=1.0, chunk_size=8)
        assert isinstance(result.data, da.Array)
        computed = result.compute()
        assert computed.shape[0] > 0


class TestMergeSameCrsYOrientation:
    """``merge()`` same-CRS fast path must honor input y orientation (#2186).

    The output of ``merge()`` is always north-up. When the source CRS
    equals the target CRS, ``_place_same_crs`` does a direct pixel copy
    of the source window into the output. A y-ascending source must be
    flipped along y during placement so the result matches what
    ``reproject(r, target_crs=r.crs)`` would emit.
    """

    @staticmethod
    def _y_ascending_raster(values=None, shape=(16, 16),
                            x_range=(-5, 5), y_range=(-5, 5),
                            crs='EPSG:4326'):
        h, w = shape
        if values is None:
            values = np.arange(h * w, dtype=np.float64).reshape(h, w)
        y = np.linspace(y_range[0], y_range[1], h)  # ascending
        x = np.linspace(x_range[0], x_range[1], w)
        return xr.DataArray(
            values, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': crs, 'nodata': np.nan},
        )

    def test_y_ascending_single_raster_matches_reproject(self):
        from xrspatial.reproject import merge, reproject
        r = self._y_ascending_raster()
        merged = merge([r], target_crs='EPSG:4326')
        reprojected = reproject(r, target_crs='EPSG:4326',
                                width=merged.shape[1],
                                height=merged.shape[0])
        np.testing.assert_allclose(
            merged.values, reprojected.values,
            atol=1e-10, equal_nan=True,
        )

    def test_y_ascending_preserves_north_south_gradient(self):
        """Encode latitude in the data and verify the row order is north-up."""
        from xrspatial.reproject import merge
        h, w = 16, 16
        y_asc = np.linspace(-5.0, 5.0, h)
        # data[i, j] = y[i] -- so y-ascending input has small values in
        # row 0 (south) and large values in row -1 (north).
        data = np.broadcast_to(y_asc[:, None], (h, w)).astype(np.float64)
        r = self._y_ascending_raster(values=data)
        merged = merge([r])
        # Output is always north-up: row 0 (top) should hold the
        # largest y values, row -1 (bottom) the smallest.
        assert merged.values[0, 0] > merged.values[-1, 0]
        np.testing.assert_allclose(merged.values[0], y_asc[-1])
        np.testing.assert_allclose(merged.values[-1], y_asc[0])

    def test_y_descending_single_raster_unchanged(self):
        """Regression guard: north-up inputs must keep working."""
        from xrspatial.reproject import merge, reproject
        data = np.arange(16 * 16, dtype=np.float64).reshape(16, 16)
        r = _make_raster(data, x_range=(-5, 5), y_range=(-5, 5))
        merged = merge([r], target_crs='EPSG:4326')
        reprojected = reproject(r, target_crs='EPSG:4326',
                                width=merged.shape[1],
                                height=merged.shape[0])
        np.testing.assert_allclose(
            merged.values, reprojected.values,
            atol=1e-10, equal_nan=True,
        )

    def test_mixed_orientation_multi_raster_merge(self):
        """Two tiles with different y orientations should merge cleanly."""
        from xrspatial.reproject import merge
        h, w = 16, 16
        left_vals = np.full((h, w), 1.0)
        right_vals = np.full((h, w), 2.0)
        # Left tile is y-descending (north-up); right tile is y-ascending.
        left = _make_raster(left_vals, x_range=(-10, 0), y_range=(-5, 5))
        right = xr.DataArray(
            right_vals, dims=['y', 'x'],
            coords={'y': np.linspace(-5, 5, h),  # ascending
                    'x': np.linspace(0, 10, w)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        merged = merge([left, right], resolution=1.0)
        vals = merged.values
        x = merged.coords['x'].values
        left_col = vals[:, x < -2]
        right_col = vals[:, x > 2]
        valid_l = ~np.isnan(left_col)
        valid_r = ~np.isnan(right_col)
        # Up-front asserts so the test can't quietly degenerate into a
        # no-op if the output grid shape ever shifts.
        assert valid_l.any(), "left tile produced no valid output pixels"
        assert valid_r.any(), "right tile produced no valid output pixels"
        np.testing.assert_allclose(left_col[valid_l], 1.0, atol=1e-9)
        np.testing.assert_allclose(right_col[valid_r], 2.0, atol=1e-9)

    def test_mixed_orientation_gradient_alignment(self):
        """Per-cell parity for a gradient that pins the orientation."""
        from xrspatial.reproject import merge, reproject
        h, w = 16, 16
        y_asc = np.linspace(-5, 5, h)
        x = np.linspace(-5, 5, w)
        # values depend on y so any vertical flip is visible.
        vals = np.broadcast_to(y_asc[:, None], (h, w)).astype(np.float64)
        r = xr.DataArray(
            vals, dims=['y', 'x'],
            coords={'y': y_asc, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        merged = merge([r], target_crs='EPSG:4326')
        # Compare row-by-row vs reproject() with the same output grid.
        reprojected = reproject(r, target_crs='EPSG:4326',
                                width=merged.shape[1],
                                height=merged.shape[0])
        np.testing.assert_allclose(
            merged.values, reprojected.values,
            atol=1e-10, equal_nan=True,
        )

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_dask_y_ascending_matches_numpy(self):
        from xrspatial.reproject import merge
        h, w = 32, 32
        y_asc = np.linspace(-5, 5, h)
        vals = np.broadcast_to(y_asc[:, None], (h, w)).astype(np.float64)
        np_raster = xr.DataArray(
            vals, dims=['y', 'x'],
            coords={'y': y_asc, 'x': np.linspace(-5, 5, w)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        dask_raster = np_raster.copy()
        dask_raster.data = da.from_array(vals, chunks=(16, 16))

        numpy_result = merge([np_raster], chunk_size=16)
        dask_result = merge([dask_raster], chunk_size=16)
        assert isinstance(dask_result.data, da.Array)
        np.testing.assert_allclose(
            numpy_result.values, dask_result.compute().values,
            atol=1e-10, equal_nan=True,
        )


class TestMergeMixedNodata:
    """merge() must honor each raster's own nodata sentinel."""

    def test_merge_mixed_nodata_sentinels(self):
        """Raster A NaN sentinel, raster B -9999 sentinel.

        B's -9999 pixels must be recognized as nodata, not leaked as
        real data into the merged output.
        """
        from xrspatial.reproject import merge

        # Raster A: all valid, value 10
        a_data = np.full((16, 16), 10.0, dtype=np.float64)
        a = _make_raster(
            a_data, x_range=(-10, 0), y_range=(-5, 5), nodata=np.nan
        )

        # Raster B: half valid (=20), half -9999 sentinel
        b_data = np.full((16, 16), 20.0, dtype=np.float64)
        b_data[:, :8] = -9999.0  # left half is nodata
        b = _make_raster(
            b_data, x_range=(0, 10), y_range=(-5, 5), nodata=-9999.0
        )

        result = merge([a, b], strategy='mean', resolution=1.0)
        vals = result.values

        # The output should never contain -9999 as a data value.
        # B's -9999 pixels were correctly recognized as nodata.
        assert not np.any(vals == -9999.0), (
            "B's -9999 nodata pixels leaked into the merged output"
        )

        # B's right half (x > ~5) should still surface as 20.
        x = result.coords['x'].values
        right_mask = x > 6
        if right_mask.any():
            right = vals[:, right_mask]
            valid = ~np.isnan(right)
            if valid.any():
                np.testing.assert_allclose(
                    right[valid], 20.0, atol=1.0
                )

    def test_merge_nan_then_int_sentinel(self):
        """Mean strategy must not fold sentinel zeros into the average."""
        from xrspatial.reproject import merge

        a_data = np.full((8, 8), 10.0, dtype=np.float64)
        a = _make_raster(
            a_data, x_range=(-5, 5), y_range=(-5, 5), nodata=np.nan
        )

        # Raster B uses 0.0 as nodata sentinel
        b_data = np.full((8, 8), 0.0, dtype=np.float64)
        b = _make_raster(
            b_data, x_range=(-5, 5), y_range=(-5, 5), nodata=0.0
        )

        result = merge([a, b], strategy='mean', resolution=1.0)
        vals = result.values
        interior = vals[1:-1, 1:-1]
        valid = ~np.isnan(interior)
        if valid.any():
            # If B's zeros were treated as data, mean would be ~5.
            # Treated as nodata, mean is just 10.
            np.testing.assert_allclose(
                interior[valid], 10.0, atol=1.0
            )

    def test_merge_explicit_user_nodata_with_mixed_inputs(self):
        """User-specified output nodata is independent of input sentinels."""
        from xrspatial.reproject import merge

        # Raster A: NaN nodata, all valid
        a_data = np.full((16, 16), 10.0, dtype=np.float64)
        a = _make_raster(
            a_data, x_range=(-10, 0), y_range=(-5, 5), nodata=np.nan
        )

        # Raster B: -9999 nodata, half valid (=20)
        b_data = np.full((16, 16), 20.0, dtype=np.float64)
        b_data[:, :8] = -9999.0
        b = _make_raster(
            b_data, x_range=(0, 10), y_range=(-5, 5), nodata=-9999.0
        )

        result = merge(
            [a, b], strategy='mean', resolution=1.0, nodata=-9999.0
        )
        vals = result.values

        # Output uses -9999 as the nodata sentinel, but data pixels must
        # never be 0 from B's zero-sentinel test (different test). Here
        # the only -9999 in the output should be true nodata regions
        # (no overlap with any input). We verify B's right half surfaces
        # as 20 (not -9999) and A's region surfaces as 10.
        x = result.coords['x'].values

        right_mask = x > 6
        if right_mask.any():
            right = vals[:, right_mask]
            data_mask = right != -9999.0
            if data_mask.any():
                np.testing.assert_allclose(
                    right[data_mask], 20.0, atol=1.0
                )

        left_mask = x < -6
        if left_mask.any():
            left = vals[:, left_mask]
            data_mask = left != -9999.0
            if data_mask.any():
                np.testing.assert_allclose(
                    left[data_mask], 10.0, atol=1.0
                )

        # No NaN in the output -- user requested -9999 as the sentinel.
        assert not np.any(np.isnan(vals)), (
            "user requested -9999 nodata but output contains NaN"
        )


# ---------------------------------------------------------------------------
# Accessor integration
# ---------------------------------------------------------------------------

class TestAccessor:
    def test_xrs_reproject(self):
        import xrspatial  # noqa: F401 - registers accessor
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=16, w=16)
        result = raster.xrs.reproject('EPSG:3857')
        assert result.shape[0] > 0


# ---------------------------------------------------------------------------
# Integer rasters
# ---------------------------------------------------------------------------

class TestIntegerRaster:
    def test_integer_nearest(self):
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.int32).reshape(8, 8)
        raster = _make_raster(data, x_range=(-4, 4), y_range=(-4, 4))
        result = reproject(raster, 'EPSG:4326', resampling='nearest')
        assert result.shape[0] > 0

    def test_integer_bilinear(self):
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.int32).reshape(8, 8)
        raster = _make_raster(data, x_range=(-4, 4), y_range=(-4, 4))
        result = reproject(raster, 'EPSG:4326', resampling='bilinear')
        assert result.shape[0] > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_1x1_raster(self):
        """Single-pixel raster should not crash."""
        from xrspatial.reproject import reproject
        raster = _make_raster(np.array([[42.0]]), x_range=(0, 0), y_range=(0, 0))
        result = reproject(raster, 'EPSG:3857')
        assert result.shape[0] >= 1
        assert result.shape[1] >= 1

    def test_2x2_raster(self):
        from xrspatial.reproject import reproject
        data = np.array([[1, 2], [3, 4]], dtype=np.float64)
        raster = _make_raster(data, x_range=(-1, 1), y_range=(-1, 1))
        result = reproject(raster, 'EPSG:3857')
        assert result.shape[0] > 0
        valid = result.values[np.isfinite(result.values)]
        assert len(valid) > 0

    def test_antimeridian_east(self):
        """Raster near 180E should reproject without grid blow-up."""
        from xrspatial.reproject import reproject
        data = np.ones((16, 16), dtype=np.float64) * 42
        raster = _make_raster(data, x_range=(176, 180), y_range=(-20, -16))
        result = reproject(raster, 'EPSG:3857')
        # Should not produce an absurdly wide output
        assert result.shape[1] < 200

    def test_antimeridian_west(self):
        """Raster near 180W should reproject without grid blow-up."""
        from xrspatial.reproject import reproject
        data = np.ones((16, 16), dtype=np.float64) * 42
        raster = _make_raster(data, x_range=(-180, -177), y_range=(-20, -16))
        result = reproject(raster, 'EPSG:3857')
        assert result.shape[1] < 200

    def test_arctic_to_mercator(self):
        """High-latitude reproject to Web Mercator."""
        from xrspatial.reproject import reproject
        data = np.random.RandomState(42).rand(16, 16).astype(np.float64)
        raster = _make_raster(data, x_range=(-30, 30), y_range=(60, 80))
        result = reproject(raster, 'EPSG:3857')
        assert result.shape[0] > 0
        assert np.isfinite(result.values).any()

    def test_arctic_beyond_mercator_limit(self):
        """Latitudes beyond 85N should not crash for Mercator."""
        from xrspatial.reproject import reproject
        data = np.random.RandomState(42).rand(16, 16).astype(np.float64)
        raster = _make_raster(data, x_range=(-30, 30), y_range=(80, 90))
        result = reproject(raster, 'EPSG:3857')
        assert result.shape[0] > 0

    def test_polar_stereographic(self):
        """Reproject to polar stereographic CRS."""
        from xrspatial.reproject import reproject
        data = np.random.RandomState(42).rand(16, 16).astype(np.float64)
        raster = _make_raster(data, x_range=(-30, 30), y_range=(60, 80))
        result = reproject(raster, 'EPSG:3413')
        assert result.shape[0] > 0

    def test_south_up_matches_north_up(self):
        """Y-ascending (south-up) should produce same result as Y-descending."""
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.float64).reshape(8, 8)
        y_asc = np.linspace(-10, 10, 8)
        x = np.linspace(-10, 10, 8)

        south_up = xr.DataArray(data, dims=['y', 'x'],
                                coords={'y': y_asc, 'x': x},
                                attrs={'crs': 'EPSG:4326'})
        north_up = xr.DataArray(data[::-1], dims=['y', 'x'],
                                coords={'y': y_asc[::-1], 'x': x},
                                attrs={'crs': 'EPSG:4326'})
        r_south = reproject(south_up, 'EPSG:3857', width=16, height=16)
        r_north = reproject(north_up, 'EPSG:3857', width=16, height=16)
        np.testing.assert_allclose(
            r_south.values, r_north.values, atol=1e-10, equal_nan=True)


class TestXDescendingReproject:
    """Regression tests for #2183: x-descending input handling."""

    def test_x_descending_same_crs_nearest(self):
        """X-descending raster reprojected to same CRS+grid must mirror cols.

        Regression test for #2183: before the fix, an x-descending input
        was silently treated as x-ascending and the output columns were
        not mirrored.
        """
        from xrspatial.reproject import reproject
        data = np.arange(9, dtype=np.float64).reshape(3, 3)
        # x = [2.5, 1.5, 0.5] -> column 0 is at max x
        x_desc = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': [2.5, 1.5, 0.5], 'x': [2.5, 1.5, 0.5]},
            attrs={'crs': 'EPSG:4326'},
        )
        out = reproject(x_desc, 'EPSG:4326', resampling='nearest',
                        width=3, height=3, bounds=(0, 0, 3, 3))
        # Output x is always ascending, so each row should be reversed
        expected = data[:, ::-1]
        np.testing.assert_array_equal(out.values, expected)
        # And the output x coord is ascending
        np.testing.assert_array_less(0, np.diff(out.coords['x'].values))

    def test_x_descending_matches_x_ascending(self):
        """X-descending input should produce the same output as the
        equivalent x-ascending input (data mirrored, coords reversed)."""
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.float64).reshape(8, 8)
        y = np.linspace(10, -10, 8)  # descending y (north-up)
        x_asc = np.linspace(-10, 10, 8)
        x_desc = x_asc[::-1]

        asc = xr.DataArray(data, dims=['y', 'x'],
                           coords={'y': y, 'x': x_asc},
                           attrs={'crs': 'EPSG:4326'})
        desc = xr.DataArray(data[:, ::-1], dims=['y', 'x'],
                            coords={'y': y, 'x': x_desc},
                            attrs={'crs': 'EPSG:4326'})
        r_asc = reproject(asc, 'EPSG:3857', width=16, height=16)
        r_desc = reproject(desc, 'EPSG:3857', width=16, height=16)
        np.testing.assert_allclose(
            r_asc.values, r_desc.values, atol=1e-10, equal_nan=True)

    def test_x_descending_y_descending(self):
        """X-descending + Y-descending should match the canonical layout."""
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.float64).reshape(8, 8)
        y_desc = np.linspace(10, -10, 8)
        x_asc = np.linspace(-10, 10, 8)
        x_desc = x_asc[::-1]

        canonical = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y_desc, 'x': x_asc},
            attrs={'crs': 'EPSG:4326'})
        both_desc = xr.DataArray(
            data[:, ::-1], dims=['y', 'x'],
            coords={'y': y_desc, 'x': x_desc},
            attrs={'crs': 'EPSG:4326'})
        r_canon = reproject(canonical, 'EPSG:3857', width=16, height=16)
        r_both = reproject(both_desc, 'EPSG:3857', width=16, height=16)
        np.testing.assert_allclose(
            r_canon.values, r_both.values, atol=1e-10, equal_nan=True)

    def test_x_descending_y_ascending(self):
        """X-descending + Y-ascending should also match the canonical layout."""
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.float64).reshape(8, 8)
        y_desc = np.linspace(10, -10, 8)
        y_asc = y_desc[::-1]
        x_asc = np.linspace(-10, 10, 8)
        x_desc = x_asc[::-1]

        canonical = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y_desc, 'x': x_asc},
            attrs={'crs': 'EPSG:4326'})
        # Flip both axes vs canonical -- data needs the same flipping.
        mixed = xr.DataArray(
            data[::-1, ::-1], dims=['y', 'x'],
            coords={'y': y_asc, 'x': x_desc},
            attrs={'crs': 'EPSG:4326'})
        r_canon = reproject(canonical, 'EPSG:3857', width=16, height=16)
        r_mixed = reproject(mixed, 'EPSG:3857', width=16, height=16)
        np.testing.assert_allclose(
            r_canon.values, r_mixed.values, atol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_x_descending_dask_backend(self):
        """Dask+numpy backend should honor x_desc the same as the numpy path."""
        import dask.array as da
        from xrspatial.reproject import reproject

        data = np.arange(64, dtype=np.float64).reshape(8, 8)
        y = np.linspace(10, -10, 8)
        x_asc = np.linspace(-10, 10, 8)
        x_desc = x_asc[::-1]

        asc = xr.DataArray(
            da.from_array(data, chunks=4),
            dims=['y', 'x'],
            coords={'y': y, 'x': x_asc},
            attrs={'crs': 'EPSG:4326'})
        desc = xr.DataArray(
            da.from_array(data[:, ::-1], chunks=4),
            dims=['y', 'x'],
            coords={'y': y, 'x': x_desc},
            attrs={'crs': 'EPSG:4326'})
        r_asc = reproject(asc, 'EPSG:3857', width=16, height=16).compute()
        r_desc = reproject(desc, 'EPSG:3857', width=16, height=16).compute()
        np.testing.assert_allclose(
            r_asc.values, r_desc.values, atol=1e-10, equal_nan=True)

    @pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
    def test_x_descending_cupy_backend(self):
        """CuPy backend should honor x_desc the same as the numpy path."""
        import cupy as cp
        from xrspatial.reproject import reproject

        data = np.arange(64, dtype=np.float64).reshape(8, 8)
        y = np.linspace(10, -10, 8)
        x_asc = np.linspace(-10, 10, 8)
        x_desc = x_asc[::-1]

        asc = xr.DataArray(
            cp.asarray(data),
            dims=['y', 'x'],
            coords={'y': y, 'x': x_asc},
            attrs={'crs': 'EPSG:4326'})
        desc = xr.DataArray(
            cp.asarray(data[:, ::-1]),
            dims=['y', 'x'],
            coords={'y': y, 'x': x_desc},
            attrs={'crs': 'EPSG:4326'})
        r_asc = reproject(asc, 'EPSG:3857', width=16, height=16)
        r_desc = reproject(desc, 'EPSG:3857', width=16, height=16)
        np.testing.assert_allclose(
            cp.asnumpy(r_asc.data), cp.asnumpy(r_desc.data),
            atol=1e-10, equal_nan=True)

    def test_merge_x_descending_same_crs(self):
        """Same-CRS merge of x-descending tiles should place values correctly."""
        from xrspatial.reproject import merge
        # x-descending tile: column 0 is at the max x value
        data_a = np.full((8, 8), 1.0)
        data_b = np.full((8, 8), 2.0)
        y = np.linspace(5, -5, 8)
        # tile A covers x in [-5, 0], tile B covers x in [0, 5] -- both
        # expressed in descending x order to exercise the x_desc path.
        x_a = np.linspace(0, -5, 8)
        x_b = np.linspace(5, 0, 8)
        tile_a = xr.DataArray(
            data_a, dims=['y', 'x'],
            coords={'y': y, 'x': x_a},
            attrs={'crs': 'EPSG:4326'})
        tile_b = xr.DataArray(
            data_b, dims=['y', 'x'],
            coords={'y': y, 'x': x_b},
            attrs={'crs': 'EPSG:4326'})
        result = merge([tile_a, tile_b], resolution=0.5)
        # Output x is always ascending. The leftmost x should have value 1
        # (from tile A), the rightmost x should have value 2 (from tile B).
        vals = result.values
        x_out = result.coords['x'].values
        assert x_out[0] < x_out[-1]
        # Sample a few interior rows away from edges
        left_col = vals[2:6, 1]
        right_col = vals[2:6, -2]
        assert np.all(left_col == 1.0), f"left edge: {left_col}"
        assert np.all(right_col == 2.0), f"right edge: {right_col}"

    def test_utm_roundtrip(self):
        """4326 -> UTM -> 4326 should recover original values."""
        from xrspatial.reproject import reproject
        data = np.random.RandomState(42).rand(16, 16).astype(np.float64) * 100
        raster = _make_raster(data, x_range=(13, 17), y_range=(50, 54))
        to_utm = reproject(raster, 'EPSG:32633')
        back = reproject(to_utm, 'EPSG:4326', source_crs='EPSG:32633',
                         width=16, height=16)
        # Interior should match within interpolation tolerance
        valid = np.isfinite(back.values) & (back.values > 0)
        assert valid.sum() > 50

    def test_all_nan_raster(self):
        """All-NaN raster should produce all-NaN output."""
        from xrspatial.reproject import reproject
        raster = _make_raster(np.full((16, 16), np.nan),
                              x_range=(-5, 5), y_range=(-5, 5))
        result = reproject(raster, 'EPSG:3857')
        assert np.isnan(result.values).all()

    def test_nodata_sentinel_propagation(self):
        """Sentinel nodata value should be preserved in output."""
        from xrspatial.reproject import reproject
        data = np.full((16, 16), 42.0)
        data[:4, :] = -9999
        raster = _make_raster(data, x_range=(-5, 5), y_range=(-5, 5))
        raster.attrs['nodata'] = -9999
        result = reproject(raster, 'EPSG:4326', nodata=-9999,
                           width=16, height=16)
        vals = result.values
        # Interior valid pixels should be close to 42
        valid_42 = (vals > 40) & (vals < 44)
        assert valid_42.sum() > 50
        # Nodata regions should be -9999
        assert (vals == -9999).sum() > 0

    def test_merge_with_gap(self):
        """Merge tiles with a gap should have nodata in the gap."""
        from xrspatial.reproject import merge
        left = _make_raster(np.full((16, 16), 10.0),
                            x_range=(-10, -2), y_range=(-5, 5))
        right = _make_raster(np.full((16, 16), 20.0),
                             x_range=(2, 10), y_range=(-5, 5))
        result = merge([left, right], resolution=0.5)
        x = result.coords['x'].values
        gap = result.sel(x=slice(-1, 1)).values
        assert np.isnan(gap).mean() > 0.8

    def test_conus_to_albers(self):
        """CONUS extent to Albers Equal Area (large coordinate shift)."""
        from xrspatial.reproject import reproject
        data = np.random.RandomState(42).rand(32, 64).astype(np.float64) * 1000
        raster = _make_raster(data, x_range=(-120, -70), y_range=(25, 50))
        result = reproject(raster, 'EPSG:5070')
        assert result.shape[0] > 0
        assert np.isfinite(result.values).sum() > result.values.size * 0.5

    def test_wide_raster(self):
        """Extreme aspect ratio (4x256) should not crash."""
        from xrspatial.reproject import reproject
        raster = _make_raster(np.ones((4, 256), dtype=np.float64) * 42,
                              x_range=(-170, 170), y_range=(-2, 2))
        result = reproject(raster, 'EPSG:3857')
        assert result.shape[0] > 0


def test_reproject_1x1_raster():
    """Reprojecting a single-pixel raster should not crash."""
    from xrspatial.reproject import reproject
    da = xr.DataArray(
        np.array([[42.0]]), dims=['y', 'x'],
        coords={'y': [50.0], 'x': [10.0]},
        attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
    )
    result = reproject(da, 'EPSG:32633')
    assert result.shape[0] >= 1 and result.shape[1] >= 1


def test_reproject_all_nan():
    """Reprojecting an all-NaN raster should produce all-NaN output."""
    from xrspatial.reproject import reproject
    da = xr.DataArray(
        np.full((64, 64), np.nan), dims=['y', 'x'],
        coords={'y': np.linspace(55, 45, 64), 'x': np.linspace(-5, 5, 64)},
        attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
    )
    result = reproject(da, 'EPSG:32633')
    assert np.all(np.isnan(result.values))


def test_reproject_uint8_cubic_no_overflow():
    """Cubic resampling on uint8 should clamp, not wrap."""
    from xrspatial.reproject import reproject
    # Create a raster with sharp edge (0 to 255)
    data = np.zeros((64, 64), dtype=np.uint8)
    data[:, 32:] = 255
    da = xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': np.linspace(55, 45, 64), 'x': np.linspace(-5, 5, 64)},
        attrs={'crs': 'EPSG:4326', 'nodata': 0},
    )
    result = reproject(da, 'EPSG:32633', resampling='cubic')
    vals = result.values
    # Should be within uint8 range (clamped, not wrapped)
    valid = vals[vals != 0]  # exclude nodata
    if len(valid) > 0:
        assert np.all(valid >= 0) and np.all(valid <= 255)


# ---------------------------------------------------------------------------
# Per-band nodata (#2647)
# ---------------------------------------------------------------------------

def _make_per_band_nodata_raster():
    """3-band raster with a distinct source sentinel baked into each band.

    Band b is filled with the valid value ``10*(b+1)`` and a 2x2 corner
    block of band b's own nodata sentinel. The ``nodatavals`` attr declares
    the per-band sentinels in band order: ``(-9999, 255, 0)``.
    """
    ny, nx = 16, 16
    sentinels = (-9999.0, 255.0, 0.0)
    valids = (10.0, 20.0, 30.0)
    bands = []
    for sentinel, valid in zip(sentinels, valids):
        plane = np.full((ny, nx), valid, dtype=np.float64)
        plane[:2, :2] = sentinel  # corner block of this band's nodata
        bands.append(plane)
    data = np.stack(bands, axis=0)  # (band, y, x)
    raster = xr.DataArray(
        data, dims=['band', 'y', 'x'],
        coords={'band': [1, 2, 3],
                'y': np.linspace(55, 45, ny),
                'x': np.linspace(-5, 5, nx)},
        attrs={'crs': 'EPSG:4326', 'nodatavals': sentinels},
    )
    return raster, sentinels, valids


def _assert_each_band_masked(result, sentinels, valids):
    """Every band's own sentinel must be gone; its valid value must survive."""
    arr = result.transpose('band', 'y', 'x').values
    for b, (sentinel, valid) in enumerate(zip(sentinels, valids)):
        band = arr[b]
        finite = band[np.isfinite(band)]
        # The raw source sentinel for this band must not leak through as a
        # resampled "valid" sample. (-9999 is the resolved output sentinel
        # used for masked pixels, so it is expected and excluded here.)
        if sentinel != -9999.0:
            assert not np.any(finite == sentinel), (
                f"band {b}: source sentinel {sentinel} leaked into output"
            )
        # The band's valid fill value must still be present somewhere.
        assert np.any(np.isclose(finite, valid)), (
            f"band {b}: valid value {valid} did not survive reprojection"
        )


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
class TestPerBandNodata:
    """Multi-band rasters with distinct per-band nodata sentinels (#2647).

    Before the fix, ``_detect_nodata_raw`` read only ``nodatavals[0]`` and
    the worker masked every band with that single value, so bands 1+ leaked
    their invalid pixels into the output as valid data.
    """

    def _reproject(self, *args, **kwargs):
        from xrspatial.reproject import reproject
        return reproject(*args, **kwargs)

    def test_detect_band_nodata_helper(self):
        from xrspatial.reproject._crs_utils import _detect_band_nodata
        raster, sentinels, _ = _make_per_band_nodata_raster()
        # Canonical layout is (y, x, band); the public path transposes
        # before calling, but the helper only reads attrs + band count.
        assert _detect_band_nodata(raster, None, 3) == sentinels
        # Explicit nodata arg overrides per-band detection.
        assert _detect_band_nodata(raster, 0.0, 3) is None
        # Single-band rasters never get a per-band tuple.
        assert _detect_band_nodata(raster, None, 1) is None

    def test_detect_band_nodata_uniform_returns_none(self):
        from xrspatial.reproject._crs_utils import _detect_band_nodata
        raster = xr.DataArray(
            np.zeros((3, 4, 4)), dims=['band', 'y', 'x'],
            attrs={'nodatavals': (0.0, 0.0, 0.0)},
        )
        assert _detect_band_nodata(raster, None, 3) is None

    def test_numpy_per_band_masking(self):
        raster, sentinels, valids = _make_per_band_nodata_raster()
        r = self._reproject(raster, 'EPSG:32633', resampling='nearest')
        assert r.ndim == 3
        _assert_each_band_masked(r, sentinels, valids)

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_dask_numpy_per_band_masking(self):
        raster, sentinels, valids = _make_per_band_nodata_raster()
        raster.data = da.from_array(raster.values, chunks=(3, 8, 8))
        r = self._reproject(raster, 'EPSG:32633', resampling='nearest')
        r = r.compute() if hasattr(r.data, 'compute') else r
        _assert_each_band_masked(r, sentinels, valids)

    @pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
    def test_cupy_per_band_masking(self):
        raster, sentinels, valids = _make_per_band_nodata_raster()
        raster.data = cp.asarray(raster.values)
        r = self._reproject(raster, 'EPSG:32633', resampling='nearest')
        host = r.transpose('band', 'y', 'x')
        host = xr.DataArray(host.data.get(), dims=host.dims, coords=host.coords)
        _assert_each_band_masked(host, sentinels, valids)

    @pytest.mark.skipif(not HAS_CUPY or not HAS_DASK,
                        reason="cupy and dask required")
    def test_dask_cupy_per_band_masking(self):
        raster, sentinels, valids = _make_per_band_nodata_raster()
        raster.data = da.from_array(cp.asarray(raster.values), chunks=(3, 8, 8))
        r = self._reproject(raster, 'EPSG:32633', resampling='nearest')
        computed = r.compute()
        host = computed.transpose('band', 'y', 'x')
        host = xr.DataArray(host.data.get(), dims=host.dims, coords=host.coords)
        _assert_each_band_masked(host, sentinels, valids)

    def test_output_nodatavals_band_count_preserved(self):
        raster, sentinels, _ = _make_per_band_nodata_raster()
        r = self._reproject(raster, 'EPSG:32633', resampling='nearest')
        assert 'nodatavals' in r.attrs
        # Output uses one resolved sentinel; the tuple keeps the band count.
        assert len(r.attrs['nodatavals']) == len(sentinels)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
class TestEdgeCases:
    """Edge cases that previously caused crashes or wrong results."""

    def _do_reproject(self, *args, **kwargs):
        from xrspatial.reproject import reproject
        return reproject(*args, **kwargs)

    def test_multiband_rgb(self):
        da = xr.DataArray(
            np.random.rand(32, 32, 3).astype(np.float32),
            dims=['y', 'x', 'band'],
            coords={'y': np.linspace(55, 45, 32), 'x': np.linspace(-5, 5, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        r = self._do_reproject(da, 'EPSG:32633')
        assert r.ndim == 3 and r.shape[2] == 3 and 'band' in r.dims

    def test_multiband_uint8(self):
        da = xr.DataArray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
            dims=['y', 'x', 'band'],
            coords={'y': np.linspace(55, 45, 32), 'x': np.linspace(-5, 5, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': 0},
        )
        r = self._do_reproject(da, 'EPSG:32633')
        assert r.dtype == np.uint8

    def test_antimeridian_crossing(self):
        da = xr.DataArray(
            np.ones((32, 32)), dims=['y', 'x'],
            coords={'y': np.linspace(50, 40, 32), 'x': np.linspace(170, -170, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        r = self._do_reproject(da, 'EPSG:32660')
        assert r.shape[0] > 0

    def test_y_ascending(self):
        da = xr.DataArray(
            np.ones((64, 64)), dims=['y', 'x'],
            coords={'y': np.linspace(45, 55, 64), 'x': np.linspace(-5, 5, 64)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        r = self._do_reproject(da, 'EPSG:32633')
        assert np.any(np.isfinite(r.values))

    def test_checkerboard_nan(self):
        data = np.ones((64, 64))
        data[::2, ::2] = np.nan
        data[1::2, 1::2] = np.nan
        da = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(55, 45, 64), 'x': np.linspace(-5, 5, 64)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        r = self._do_reproject(da, 'EPSG:32633')
        assert np.any(np.isfinite(r.values))

    def test_utm_to_geographic(self):
        da = xr.DataArray(
            np.ones((64, 64)), dims=['y', 'x'],
            coords={'y': np.linspace(5600000, 5500000, 64),
                    'x': np.linspace(300000, 400000, 64)},
            attrs={'crs': 'EPSG:32633', 'nodata': np.nan},
        )
        r = self._do_reproject(da, 'EPSG:4326')
        assert np.any(np.isfinite(r.values))

    def test_proj_to_proj(self):
        da = xr.DataArray(
            np.ones((64, 64)), dims=['y', 'x'],
            coords={'y': np.linspace(6500000, 6000000, 64),
                    'x': np.linspace(200000, 800000, 64)},
            attrs={'crs': 'EPSG:2154', 'nodata': np.nan},
        )
        r = self._do_reproject(da, 'EPSG:32632')
        assert np.any(np.isfinite(r.values))

    def test_sentinel_nodata(self):
        data = np.where(np.random.rand(64, 64) > 0.8, -9999, 500).astype(np.float64)
        da = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(55, 45, 64), 'x': np.linspace(-5, 5, 64)},
            attrs={'crs': 'EPSG:4326', 'nodata': -9999},
        )
        r = self._do_reproject(da, 'EPSG:32633')
        assert r is not None

    def test_target_crs_as_integer(self):
        da = xr.DataArray(
            np.ones((32, 32)), dims=['y', 'x'],
            coords={'y': np.linspace(55, 45, 32), 'x': np.linspace(-5, 5, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        r = self._do_reproject(da, 32633)
        assert r.shape[0] > 0

    def test_explicit_resolution(self):
        da = xr.DataArray(
            np.ones((64, 64)), dims=['y', 'x'],
            coords={'y': np.linspace(55, 45, 64), 'x': np.linspace(-5, 5, 64)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        r = self._do_reproject(da, 'EPSG:32633', resolution=1000)
        assert r.shape[0] > 0

    def test_explicit_width_height(self):
        da = xr.DataArray(
            np.ones((64, 64)), dims=['y', 'x'],
            coords={'y': np.linspace(55, 45, 64), 'x': np.linspace(-5, 5, 64)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        r = self._do_reproject(da, 'EPSG:32633', width=100, height=100)
        assert r.shape == (100, 100)

    def test_merge_non_overlapping(self):
        from xrspatial.reproject import merge
        t1 = xr.DataArray(
            np.full((32, 32), 1.0), dims=['y', 'x'],
            coords={'y': np.linspace(55, 50, 32), 'x': np.linspace(-5, 0, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        t2 = xr.DataArray(
            np.full((32, 32), 2.0), dims=['y', 'x'],
            coords={'y': np.linspace(45, 40, 32), 'x': np.linspace(5, 10, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        r = merge([t1, t2])
        assert r.shape[0] > 32 and r.shape[1] > 32

    def test_merge_single_tile(self):
        from xrspatial.reproject import merge
        t = xr.DataArray(
            np.ones((32, 32)), dims=['y', 'x'],
            coords={'y': np.linspace(55, 45, 32), 'x': np.linspace(-5, 5, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        r = merge([t])
        assert np.any(np.isfinite(r.values))


# ---------------------------------------------------------------------------
# CuPy resampler unit tests (integer clipping + cubic NaN fallback)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
class TestCuPyResamplerClipping:
    """Verify uint8 overflow protection in CuPy resampling paths."""

    def _sharp_edge_inputs(self):
        """Build a uint8 source with a sharp 0->255 edge and coordinate grids
        that place sample points right at the transition (where cubic ringing
        produces out-of-range values)."""
        src = np.zeros((16, 16), dtype=np.float64)
        src[:, 8:] = 255.0

        # Sample at half-pixel offsets across the edge
        rows, cols = np.meshgrid(
            np.linspace(2, 13, 24), np.linspace(6.5, 9.5, 24), indexing='ij'
        )
        return src, rows.astype(np.float64), cols.astype(np.float64)

    def test_cupy_native_nearest_uint8_clamp(self):
        from xrspatial.reproject._interpolate import _resample_cupy_native
        src, rows, cols = self._sharp_edge_inputs()
        src_gpu = cp.asarray(np.zeros((16, 16), dtype=np.uint8))
        src_gpu[:, 8:] = 255
        result = _resample_cupy_native(src_gpu, rows, cols,
                                       resampling='nearest', nodata=np.nan)
        assert result.dtype == np.uint8
        vals = cp.asnumpy(result)
        assert np.all((vals == 0) | (vals == 255) | np.isnan(vals.astype(float)))

    def test_cupy_native_bilinear_uint8_clamp(self):
        from xrspatial.reproject._interpolate import _resample_cupy_native
        src_gpu = cp.zeros((16, 16), dtype=np.uint8)
        src_gpu[:, 8:] = 255
        _, rows, cols = self._sharp_edge_inputs()
        result = _resample_cupy_native(src_gpu, rows, cols,
                                       resampling='bilinear', nodata=np.nan)
        assert result.dtype == np.uint8
        vals = cp.asnumpy(result)
        assert np.all(vals <= 255)
        assert np.all(vals >= 0)

    def test_cupy_native_cubic_uint8_clamp(self):
        from xrspatial.reproject._interpolate import _resample_cupy_native
        src_gpu = cp.zeros((16, 16), dtype=np.uint8)
        src_gpu[:, 8:] = 255
        _, rows, cols = self._sharp_edge_inputs()
        result = _resample_cupy_native(src_gpu, rows, cols,
                                       resampling='cubic', nodata=np.nan)
        assert result.dtype == np.uint8
        vals = cp.asnumpy(result)
        assert np.all(vals <= 255)
        assert np.all(vals >= 0)

    def test_cupy_map_coords_bilinear_uint8_clamp(self):
        from xrspatial.reproject._interpolate import _resample_cupy
        src_gpu = cp.zeros((16, 16), dtype=np.uint8)
        src_gpu[:, 8:] = 255
        _, rows, cols = self._sharp_edge_inputs()
        result = _resample_cupy(src_gpu, rows, cols,
                                resampling='bilinear', nodata=np.nan)
        assert result.dtype == np.uint8
        vals = cp.asnumpy(result)
        assert np.all(vals <= 255)
        assert np.all(vals >= 0)

    def test_cupy_map_coords_cubic_uint8_clamp(self):
        from xrspatial.reproject._interpolate import _resample_cupy
        src_gpu = cp.zeros((16, 16), dtype=np.uint8)
        src_gpu[:, 8:] = 255
        _, rows, cols = self._sharp_edge_inputs()
        result = _resample_cupy(src_gpu, rows, cols,
                                resampling='cubic', nodata=np.nan)
        assert result.dtype == np.uint8
        vals = cp.asnumpy(result)
        assert np.all(vals <= 255)
        assert np.all(vals >= 0)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
class TestCudaCubicNanFallback:
    """Verify _resample_cubic_cuda falls back to bilinear near NaN instead
    of writing nodata."""

    def test_cubic_nan_fallback_produces_valid_values(self):
        """Cubic with a few NaN neighbors should interpolate from valid
        neighbors (bilinear fallback), not produce nodata everywhere."""
        from xrspatial.reproject._interpolate import _resample_cupy_native

        # 16x16 source with value 100.0, a few NaN pixels scattered
        src = np.full((16, 16), 100.0, dtype=np.float64)
        src[5, 5] = np.nan
        src[10, 10] = np.nan

        src_gpu = cp.asarray(src)

        # Sample at points near (but not on) NaN pixels
        rows = np.array([[5.3, 6.0, 10.3, 8.0]], dtype=np.float64)
        cols = np.array([[5.3, 6.0, 10.3, 8.0]], dtype=np.float64)

        result = _resample_cupy_native(src_gpu, rows, cols,
                                       resampling='cubic', nodata=np.nan)
        vals = cp.asnumpy(result).ravel()

        # Points near NaN should get valid interpolated values (bilinear
        # fallback), not NaN.  Point (6.0, 6.0) and (8.0, 8.0) are far
        # enough from any NaN that cubic should succeed directly.
        assert np.isfinite(vals[1]), "point far from NaN should be finite"
        assert np.isfinite(vals[3]), "point far from NaN should be finite"
        # Points adjacent to NaN should also be finite via bilinear fallback
        assert np.isfinite(vals[0]), "bilinear fallback should produce finite value near NaN"
        assert np.isfinite(vals[2]), "bilinear fallback should produce finite value near NaN"

    def test_cubic_nan_fallback_matches_cpu(self):
        """CUDA cubic NaN fallback should produce values close to the CPU
        Numba JIT version."""
        from xrspatial.reproject._interpolate import (
            _resample_cupy_native,
            _resample_numpy,
        )

        src = np.full((16, 16), 50.0, dtype=np.float64)
        src[4, 4] = np.nan
        src[7, 12] = np.nan

        # Sample grid covering the whole raster
        rows, cols = np.meshgrid(
            np.linspace(1, 14, 12), np.linspace(1, 14, 12), indexing='ij'
        )
        rows = rows.astype(np.float64)
        cols = cols.astype(np.float64)

        cpu_result = _resample_numpy(src, rows, cols,
                                     resampling='cubic', nodata=np.nan)
        gpu_result = _resample_cupy_native(
            cp.asarray(src), rows, cols,
            resampling='cubic', nodata=np.nan
        )
        gpu_np = cp.asnumpy(gpu_result)

        # Both should have the same NaN pattern
        np.testing.assert_array_equal(np.isnan(cpu_result), np.isnan(gpu_np))
        # Finite values should match closely
        finite = np.isfinite(cpu_result)
        np.testing.assert_allclose(cpu_result[finite], gpu_np[finite],
                                   rtol=1e-10)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
class TestCupyPyprojFallbackParity:
    """The cupy backend must match numpy even when no CUDA coordinate
    kernel exists for the CRS pair (#2620).

    For a projected->projected reprojection such as EPSG:32633 ->
    EPSG:3857 (neither side is geographic WGS84/NAD83), both backends
    fall back to pyproj for coordinates. The cupy resampler previously
    used cupyx.scipy.ndimage.map_coordinates, which bled the cval=0.0
    constant into the half-pixel boundary band and used a B-spline for
    cubic instead of the Catmull-Rom kernel the numpy path uses, so the
    two backends produced different numbers.
    """

    def _make_utm_source(self):
        ny, nx = 64, 64
        y = np.linspace(5610000.0, 5600000.0, ny)  # descending (north-up)
        x = np.linspace(500000.0, 510000.0, nx)
        data = (
            np.add.outer(np.sin(np.linspace(0, 4, ny)),
                         np.cos(np.linspace(0, 4, nx))) * 100 + 500
        ).astype(np.float64)
        return y, x, data

    def test_resample_cupy_native_renormalizes_boundary_band(self):
        """The native kernel must renormalize in the half-pixel border
        band rather than bleed a zero, matching numpy bilinear."""
        from xrspatial.reproject._interpolate import (
            _resample_cupy_native,
            _resample_numpy,
        )
        src = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
        # r=4.6 sits in the (sh-1, sh) border band; r=-0.4 in (-1, 0).
        rows = np.array([[4.6, -0.4]], dtype=np.float64)
        cols = np.array([[2.0, 2.0]], dtype=np.float64)
        cpu = _resample_numpy(src, rows, cols,
                              resampling='bilinear', nodata=np.nan)
        gpu = cp.asnumpy(_resample_cupy_native(
            cp.asarray(src), rows, cols,
            resampling='bilinear', nodata=np.nan))
        # numpy returns the renormalized edge value (23.0, 3.0), not 0.0.
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)
        assert not np.any(gpu == 0.0)

    @pytest.mark.parametrize('resampling', ['nearest', 'bilinear', 'cubic'])
    def test_projected_to_projected_numpy_cupy_match(self, resampling):
        from xrspatial.reproject import reproject
        y, x, data = self._make_utm_source()
        attrs = {'crs': 'EPSG:32633'}
        da_np = xr.DataArray(data, dims=('y', 'x'),
                             coords={'y': y, 'x': x}, attrs=attrs)
        da_cp = xr.DataArray(cp.asarray(data), dims=('y', 'x'),
                             coords={'y': y, 'x': x}, attrs=attrs)
        out_np = np.asarray(
            reproject(da_np, 'EPSG:3857', resampling=resampling).data)
        out_cp = cp.asnumpy(
            reproject(da_cp, 'EPSG:3857', resampling=resampling).data)
        assert out_np.shape == out_cp.shape
        # Same finite/nodata pattern (edge band no longer diverges).
        np.testing.assert_array_equal(
            np.isfinite(out_np), np.isfinite(out_cp))
        finite = np.isfinite(out_np) & np.isfinite(out_cp)
        np.testing.assert_allclose(out_np[finite], out_cp[finite],
                                   rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    @pytest.mark.parametrize('resampling', ['nearest', 'bilinear', 'cubic'])
    def test_projected_to_projected_dask_cupy_match(self, resampling):
        # The dask+cupy chunk-assembly path must thread every resampling
        # mode through to _resample_cupy_native per chunk, not just the
        # 'cubic' mode that used to be the only one covered here (#3050).
        from xrspatial.reproject import reproject
        y, x, data = self._make_utm_source()
        attrs = {'crs': 'EPSG:32633'}
        ref = np.asarray(
            reproject(
                xr.DataArray(data, dims=('y', 'x'),
                             coords={'y': y, 'x': x}, attrs=attrs),
                'EPSG:3857', resampling=resampling).data)
        dc = xr.DataArray(
            da.from_array(cp.asarray(data), chunks=(32, 32)),
            dims=('y', 'x'), coords={'y': y, 'x': x}, attrs=attrs)
        out = reproject(dc, 'EPSG:3857', resampling=resampling).data
        if hasattr(out, 'compute'):
            out = out.compute()
        out = cp.asnumpy(out) if isinstance(out, cp.ndarray) else np.asarray(out)
        assert ref.shape == out.shape
        np.testing.assert_array_equal(np.isfinite(ref), np.isfinite(out))
        finite = np.isfinite(ref) & np.isfinite(out)
        np.testing.assert_allclose(ref[finite], out[finite],
                                   rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
class TestCupyMultibandCoordUpload:
    """Multi-band cupy reproject on the CPU-fallback transform path must
    move the shared coordinate arrays to the device once per chunk, not
    once per band (#3268).

    EPSG:4326 -> EPSG:28992 (sterea) has no CUDA transform kernel, so the
    chunk worker computes coordinates on the CPU and hands numpy arrays to
    the per-band resample loop.
    """

    def _make_multiband(self, n_bands):
        ny, nx = 48, 48
        y = np.linspace(53.5, 50.5, ny)
        x = np.linspace(3.5, 7.0, nx)
        rng = np.random.default_rng(3268)
        data = rng.random((ny, nx, n_bands))
        coords = {'y': y, 'x': x, 'band': list(range(1, n_bands + 1))}
        return data, coords

    def test_coord_uploads_do_not_scale_with_bands(self, monkeypatch):
        from xrspatial.reproject import reproject

        data, coords = self._make_multiband(6)
        raster = xr.DataArray(
            cp.asarray(data), dims=('y', 'x', 'band'),
            coords=coords, attrs={'crs': 'EPSG:4326'},
        )

        uploads = []
        orig_asarray = cp.asarray

        def counting_asarray(a, *args, **kwargs):
            if isinstance(a, np.ndarray) and a.ndim == 2:
                uploads.append(a.shape)
            return orig_asarray(a, *args, **kwargs)

        monkeypatch.setattr(cp, 'asarray', counting_asarray)
        result = reproject(raster, 'EPSG:28992')

        out_shape = result.shape[:2]
        coord_uploads = [s for s in uploads if s == out_shape]
        # One upload each for the shared row and column coordinate arrays,
        # regardless of the band count. Before #3268 this was 2 * n_bands.
        # All recorded 2-D upload shapes are included in the failure
        # message so a future failure shows what extra upload appeared.
        assert len(coord_uploads) == 2, (
            f"expected 2 coordinate uploads of shape {out_shape}, got "
            f"{len(coord_uploads)}; all 2-D ndarray uploads: {uploads}"
        )

    def test_multiband_fallback_matches_numpy(self):
        from xrspatial.reproject import reproject

        data, coords = self._make_multiband(3)
        raster_np = xr.DataArray(
            data, dims=('y', 'x', 'band'),
            coords=coords, attrs={'crs': 'EPSG:4326'},
        )
        raster_cp = xr.DataArray(
            cp.asarray(data), dims=('y', 'x', 'band'),
            coords=coords, attrs={'crs': 'EPSG:4326'},
        )

        ref = np.asarray(reproject(raster_np, 'EPSG:28992').data)
        out = cp.asnumpy(reproject(raster_cp, 'EPSG:28992').data)

        assert ref.shape == out.shape
        np.testing.assert_array_equal(np.isfinite(ref), np.isfinite(out))
        finite = np.isfinite(ref) & np.isfinite(out)
        np.testing.assert_allclose(ref[finite], out[finite],
                                   rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Dask graph optimization tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_DASK, reason="dask not installed")
class TestDaskGraphOptimization:
    """Verify map_blocks conversion and empty-chunk skipping."""

    def test_dask_reproject_uses_map_blocks(self):
        """The dask path should produce a blockwise layer, not N delayed nodes."""
        from xrspatial.reproject import reproject
        data = np.ones((64, 64), dtype=np.float64)
        da_data = da.from_array(data, chunks=(32, 32))
        raster = xr.DataArray(
            da_data, dims=['y', 'x'],
            coords={'y': np.linspace(55, 45, 64), 'x': np.linspace(-5, 5, 64)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        result = reproject(raster, 'EPSG:32633', chunk_size=32)
        # Result should be a dask array
        assert hasattr(result.data, 'dask')
        # Should have few graph layers (map_blocks creates 1-2, not N)
        graph = result.data.__dask_graph__()
        assert len(graph.layers) <= 3

    def test_source_not_whole_array_dependency(self):
        """Source dask array should not be a dependency of every output block.

        When source_data is passed as a map_blocks kwarg, dask adds the
        full source as a dependency of every output block -- this causes
        MemoryError on distributed schedulers when the source exceeds
        worker memory.  Using functools.partial avoids this.
        """
        from xrspatial.reproject import reproject
        data = np.ones((64, 64), dtype=np.float64)
        da_data = da.from_array(data, chunks=(32, 32))
        src_name = da_data.name  # e.g. 'array-abc123'
        raster = xr.DataArray(
            da_data, dims=['y', 'x'],
            coords={'y': np.linspace(55, 45, 64), 'x': np.linspace(-5, 5, 64)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        result = reproject(raster, 'EPSG:32633', chunk_size=32)
        graph = result.data.__dask_graph__()
        # The source array's layer should NOT be in the output graph's
        # dependencies (it's captured in the function closure instead).
        assert src_name not in graph.layers, (
            f"source array '{src_name}' should not be a graph layer "
            f"dependency -- use functools.partial to bind it"
        )

    def test_dask_reproject_matches_numpy(self):
        """Dask map_blocks path should produce same values as numpy."""
        from xrspatial.reproject import reproject
        data = np.random.RandomState(42).rand(64, 64).astype(np.float64)
        coords = {
            'y': np.linspace(55, 45, 64),
            'x': np.linspace(-5, 5, 64),
        }
        attrs = {'crs': 'EPSG:4326', 'nodata': np.nan}

        np_raster = xr.DataArray(data, dims=['y', 'x'],
                                 coords=coords, attrs=attrs)
        da_raster = xr.DataArray(
            da.from_array(data, chunks=(32, 32)),
            dims=['y', 'x'], coords=coords, attrs=attrs,
        )
        np_result = reproject(np_raster, 'EPSG:32633')
        da_result = reproject(da_raster, 'EPSG:32633')

        np_vals = np_result.values
        da_vals = da_result.values
        # Same shape
        assert np_vals.shape == da_vals.shape
        # Same NaN pattern
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(da_vals))
        # Same finite values
        finite = np.isfinite(np_vals)
        if finite.any():
            np.testing.assert_allclose(np_vals[finite], da_vals[finite],
                                       rtol=1e-10)

    def test_empty_chunk_skipping(self):
        """Chunks outside the source footprint should be nodata-filled
        without touching pyproj."""
        import dask

        from xrspatial.reproject import reproject
        # Small raster in a corner of the output grid
        data = np.ones((16, 16), dtype=np.float64) * 42.0
        raster = xr.DataArray(
            da.from_array(data, chunks=(16, 16)),
            dims=['y', 'x'],
            coords={'y': np.linspace(50.1, 50.0, 16),
                    'x': np.linspace(10.0, 10.1, 16)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        # Force a large output grid with small chunks so many are empty.
        # Use synchronous scheduler to avoid PROJ C library thread-safety
        # crashes on macOS when many chunks call pyproj.CRS concurrently.
        with dask.config.set(scheduler='synchronous'):
            result = reproject(raster, 'EPSG:32633', chunk_size=64,
                               width=256, height=256)
            vals = result.values
        # Should have some valid pixels and some NaN (empty chunks)
        assert np.any(np.isfinite(vals))
        assert np.any(np.isnan(vals))

    def test_merge_dask_uses_map_blocks(self):
        """The merge dask path should also use map_blocks."""
        from xrspatial.reproject import merge
        t1 = xr.DataArray(
            da.from_array(np.full((32, 32), 1.0), chunks=(16, 16)),
            dims=['y', 'x'],
            coords={'y': np.linspace(55, 50, 32),
                    'x': np.linspace(-5, 0, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        t2 = xr.DataArray(
            da.from_array(np.full((32, 32), 2.0), chunks=(16, 16)),
            dims=['y', 'x'],
            coords={'y': np.linspace(50, 45, 32),
                    'x': np.linspace(0, 5, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        result = merge([t1, t2])
        vals = result.values
        assert np.any(np.isfinite(vals))

    def test_source_footprint_helper(self):
        """_source_footprint_in_target should return a valid bbox."""
        from xrspatial.reproject import _source_footprint_in_target
        src_bounds = (-5.0, 45.0, 5.0, 55.0)
        fp = _source_footprint_in_target(
            src_bounds, 'EPSG:4326', 'EPSG:32633'
        )
        # Should return a tuple of 4 finite values
        assert fp is not None
        assert len(fp) == 4
        assert all(np.isfinite(v) for v in fp)
        # left < right, bottom < top
        assert fp[0] < fp[2]
        assert fp[1] < fp[3]

    def test_finite_pair_bbox_joint_mask(self):
        """_finite_pair_bbox keeps x/y from the same point only (#2643).

        Independent finite-filtering of tx and ty would build a bbox from
        coordinates that never belonged to the same transformed point. Here
        the only point finite in both coordinates is (5.0, 6.0), so the bbox
        must be that single point -- not the (1, 2, 5, 6) box that
        independent filtering produces.
        """
        from xrspatial.reproject import _finite_pair_bbox
        tx = [1.0, np.nan, 5.0]
        ty = [np.nan, 2.0, 6.0]
        bbox = _finite_pair_bbox(tx, ty)
        assert bbox == (5.0, 6.0, 5.0, 6.0)
        # Independent filtering would have leaked x=1.0 (from a NaN-y point)
        # and y=2.0 (from a NaN-x point) into the box.
        assert bbox[0] != 1.0
        assert bbox[1] != 2.0

    def test_finite_pair_bbox_all_nan(self):
        """_finite_pair_bbox returns None when no pair is finite (#2643)."""
        from xrspatial.reproject import _finite_pair_bbox
        assert _finite_pair_bbox([np.nan, np.nan], [np.nan, 1.0]) is None
        assert _finite_pair_bbox([np.inf], [1.0]) is None

    def test_footprint_chunk_skip_with_unpaired_nan(self):
        """Chunk-skipping must use the joint-filtered footprint (#2643).

        When independent filtering would widen the footprint with mismatched
        coordinates, a chunk that only overlaps the spurious region must
        still be skipped under joint filtering.
        """
        from xrspatial.reproject import _bounds_overlap, _finite_pair_bbox
        # Real footprint is the point (5, 6). Independent filtering would
        # report (1, 2, 5, 6), which overlaps a chunk sitting at (1..3, 2..4).
        tx = [1.0, np.nan, 5.0]
        ty = [np.nan, 2.0, 6.0]
        fp = _finite_pair_bbox(tx, ty)
        chunk = (1.0, 2.0, 3.0, 4.0)
        # Joint footprint does not overlap the chunk -> chunk is skipped.
        assert not _bounds_overlap(chunk, fp)
        # The spurious independent-filter footprint would have overlapped.
        spurious = (1.0, 2.0, 5.0, 6.0)
        assert _bounds_overlap(chunk, spurious)

    def test_bounds_overlap(self):
        """_bounds_overlap should correctly detect overlap."""
        from xrspatial.reproject import _bounds_overlap
        a = (0, 0, 10, 10)
        assert _bounds_overlap(a, (5, 5, 15, 15))   # partial overlap
        assert _bounds_overlap(a, (0, 0, 10, 10))   # identical
        assert not _bounds_overlap(a, (11, 0, 20, 10))  # no overlap x
        assert not _bounds_overlap(a, (0, 11, 10, 20))  # no overlap y


class TestLongitudeNormalization:
    """CPU projection round-trips should keep longitude in [-180, 180] (#1088)."""

    def test_sinusoidal_round_trip_stays_in_range(self):
        """Sinusoidal inverse must normalize longitude near antimeridian."""
        from xrspatial.reproject._projections import (
            _sinu_fwd_point, _sinu_inv_point, _MLFN_EN,
        )
        # Forward: WGS84 point near antimeridian
        lon_in, lat_in = 179.5, 30.0
        lon0 = 0.0  # central meridian at 0
        x, y = _sinu_fwd_point(lon_in, lat_in, lon0, _WGS84_E2, _WGS84_A, _MLFN_EN)
        # Inverse: should return longitude in [-180, 180]
        lon_out, lat_out = _sinu_inv_point(x, y, lon0, _WGS84_E2, _WGS84_A, _MLFN_EN)
        assert -180 <= lon_out <= 180, f"lon {lon_out} outside [-180, 180]"
        assert abs(lon_out - lon_in) < 1e-6
        assert abs(lat_out - lat_in) < 1e-6

    def test_lcc_round_trip_stays_in_range(self):
        """LCC inverse must normalize longitude."""
        from xrspatial.reproject._projections import (
            _lcc_fwd_point, _lcc_inv_point, _WGS84_E, _WGS84_A,
        )
        import math
        # EPSG:2154 (France): lon0=3, lat1=44, lat2=49
        lon0 = math.radians(3.0)
        lat1, lat2, lat0 = math.radians(44.0), math.radians(49.0), math.radians(46.5)
        e = _WGS84_E
        a = _WGS84_A
        k0 = 1.0
        # Compute n, c, rho0 for LCC
        from xrspatial.reproject._projections import _pj_tsfn
        s1, s2 = math.sin(lat1), math.sin(lat2)
        ts1 = _pj_tsfn(lat1, s1, e)
        ts2 = _pj_tsfn(lat2, s2, e)
        m1 = math.cos(lat1) / math.sqrt(1.0 - e * e * s1 * s1)
        m2 = math.cos(lat2) / math.sqrt(1.0 - e * e * s2 * s2)
        n = (math.log(m1) - math.log(m2)) / (math.log(ts1) - math.log(ts2))
        c = m1 / (n * math.pow(ts1, n))
        ts0 = _pj_tsfn(lat0, math.sin(lat0), e)
        rho0 = a * k0 * c * math.pow(ts0, n)
        # Forward + inverse round trip
        lon_in, lat_in = 2.5, 47.0
        x, y = _lcc_fwd_point(lon_in, lat_in, lon0, n, c, rho0, k0, e, a)
        lon_out, lat_out = _lcc_inv_point(x, y, lon0, n, c, rho0, k0, e, a)
        assert -180 <= lon_out <= 180
        assert abs(lon_out - lon_in) < 1e-6
        assert abs(lat_out - lat_in) < 1e-6


class TestReprojWithLiteCRS:
    def test_reproject_wgs84_to_utm_with_lite_crs(self):
        import xarray as xr
        from xrspatial.reproject import reproject
        import numpy as np
        h, w = 32, 32
        y = np.linspace(49, 47, h)
        x = np.linspace(8, 10, w)
        data = np.random.default_rng(42).random((h, w))
        raster = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 4326},
        )
        result = reproject(raster, target_crs=32632)
        assert result.attrs['crs'] is not None
        assert result.shape[0] > 0 and result.shape[1] > 0


# ---------------------------------------------------------------------------
# Security guards (Cat 1: unbounded allocation)
# ---------------------------------------------------------------------------

class TestSecurityGuards:
    """Verify that memory guards prevent unbounded allocations."""

    def test_output_grid_too_large_raises(self):
        """_compute_output_grid should reject grids > 1 billion pixels."""
        from xrspatial.reproject._grid import _compute_output_grid
        from xrspatial.reproject._crs_utils import _resolve_crs

        src_crs = _resolve_crs(4326)
        tgt_crs = _resolve_crs(4326)

        # Tiny resolution on a wide extent would produce > 1e9 pixels.
        with pytest.raises(ValueError, match="too large"):
            _compute_output_grid(
                source_bounds=(-180, -90, 180, 90),
                source_shape=(1000, 1000),
                source_crs=src_crs,
                target_crs=tgt_crs,
                resolution=1e-6,  # ~360M cols x 180M rows >> 1e9
            )

    def test_output_grid_normal_resolution_ok(self):
        """Normal resolution should not be rejected."""
        from xrspatial.reproject._grid import _compute_output_grid
        from xrspatial.reproject._crs_utils import _resolve_crs

        src_crs = _resolve_crs(4326)
        tgt_crs = _resolve_crs(4326)

        result = _compute_output_grid(
            source_bounds=(-10, -10, 10, 10),
            source_shape=(100, 100),
            source_crs=src_crs,
            target_crs=tgt_crs,
            resolution=0.1,
        )
        assert result['shape'] == (200, 200)

    def test_output_grid_too_large_lazy_output_ok(self):
        """lazy_output=True bypasses the >1e9 pixel guard (issue #3046)."""
        from xrspatial.reproject._grid import _compute_output_grid
        from xrspatial.reproject._crs_utils import _resolve_crs

        src_crs = _resolve_crs(4326)
        tgt_crs = _resolve_crs(4326)

        # Same grid that raises without lazy_output must now succeed,
        # because a dask output never materializes the full array.
        result = _compute_output_grid(
            source_bounds=(-180, -90, 180, 90),
            source_shape=(1000, 1000),
            source_crs=src_crs,
            target_crs=tgt_crs,
            resolution=1e-6,  # ~360M cols x 180M rows >> 1e9
            lazy_output=True,
        )
        h, w = result['shape']
        assert w * h > 1_000_000_000

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_reproject_dask_output_over_limit_stays_lazy(self):
        """A dask input whose output exceeds 1e9 pixels reprojects lazily
        instead of raising (issue #3046)."""
        from xrspatial.reproject import reproject

        raster = _make_raster(
            np.arange(64 * 64, dtype='float64').reshape(64, 64),
            crs='EPSG:4326',
            x_range=(-105, -104),
            y_range=(39, 40),
        )
        raster.data = da.from_array(raster.values, chunks=(32, 32))

        # Tiny resolution forces >1e9 output pixels. A modest chunk_size
        # keeps each computed block small so the test stays cheap.
        out = reproject(raster, target_crs='EPSG:3857',
                        resolution=2.0, chunk_size=1024)

        assert isinstance(out.data, da.Array)
        assert out.shape[0] * out.shape[1] > 1_000_000_000
        # A single block computes without materializing the whole grid.
        assert out.data.blocks[0, 0].compute().shape == (1024, 1024)

    def test_numpy_chunk_source_window_guard(self):
        """_reproject_chunk_numpy should return nodata for huge source windows."""
        from xrspatial.reproject import reproject

        # A raster that covers a small area but projected to a CRS where
        # the inverse transform maps to a large source region.
        # We just verify the function doesn't crash for normal inputs.
        raster = _make_raster(
            np.ones((32, 32)),
            crs='EPSG:4326',
            x_range=(-1, 1),
            y_range=(-1, 1),
        )
        result = reproject(raster, target_crs='EPSG:3857')
        assert result.shape[0] > 0 and result.shape[1] > 0

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_merge_dask_output_over_limit_stays_lazy(self):
        """A dask-backed merge whose output exceeds 1e9 pixels runs through
        the lazy path instead of tripping the in-memory guard (issue #3048)."""
        from xrspatial.reproject import merge

        a = _make_raster(
            np.full((32, 32), 1.0), x_range=(-105, -104), y_range=(39, 40)
        )
        b = _make_raster(
            np.full((32, 32), 2.0), x_range=(-104, -103), y_range=(39, 40)
        )
        a.data = da.from_array(a.values, chunks=(16, 16))
        b.data = da.from_array(b.values, chunks=(16, 16))

        # Tiny resolution forces > 1e9 output pixels. The merge is dask
        # backed, so the result stays lazy and the guard must not fire.
        out = merge([a, b], resolution=4e-5, chunk_size=1024)

        assert isinstance(out.data, da.Array)
        assert out.shape[0] * out.shape[1] > 1_000_000_000
        # A single block computes without materializing the whole grid.
        assert out.data.blocks[0, 0].compute().shape == (1024, 1024)

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_merge_inmemory_auto_promote_over_limit_stays_lazy(self):
        """An in-memory merge whose output exceeds 1e9 pixels auto-promotes
        to the dask path and must not raise the in-memory guard (issue #3048)."""
        from xrspatial.reproject import merge

        a = _make_raster(
            np.full((32, 32), 1.0), x_range=(-105, -104), y_range=(39, 40)
        )
        b = _make_raster(
            np.full((32, 32), 2.0), x_range=(-104, -103), y_range=(39, 40)
        )

        # Numpy inputs over the in-memory size threshold auto-promote to
        # dask, so the > 1e9 pixel output must not trip the guard.
        out = merge([a, b], resolution=4e-5, chunk_size=1024)

        assert isinstance(out.data, da.Array)
        assert out.shape[0] * out.shape[1] > 1_000_000_000

    def test_merge_inmemory_over_limit_still_raises(self, monkeypatch):
        """A genuinely in-memory merge over the pixel limit still raises the
        'too large' guard (issue #3048 must not weaken the in-memory path).

        Disabling the auto-promote-to-dask threshold keeps a > 1e9 pixel
        output on the in-memory path, where the guard must still reject it.
        """
        import importlib

        from xrspatial.reproject import merge
        _reproject = importlib.import_module('xrspatial.reproject')

        # Raise the auto-promote byte budget above the test output so the
        # merge stays on the in-memory branch instead of promoting to dask.
        monkeypatch.setattr(_reproject, '_MERGE_OOM_THRESHOLD', 1 << 60)

        a = _make_raster(
            np.full((32, 32), 1.0), x_range=(-105, -104), y_range=(39, 40)
        )
        b = _make_raster(
            np.full((32, 32), 2.0), x_range=(-104, -103), y_range=(39, 40)
        )

        with pytest.raises(ValueError, match="too large"):
            merge([a, b], resolution=4e-5)


# ---------------------------------------------------------------------------
# Issue #3267: output-size based promotion to the dask path
# ---------------------------------------------------------------------------

class TestReprojectOutputSizePromotion:
    """reproject() must consider the *output* size when deciding between the
    in-memory and dask paths (#3267). The eager numpy path holds several
    output-sized float64 temporaries, so a small input upsampled to a large
    output OOMs long before the pixel-count guard trips.
    """

    def _patch_threshold(self, monkeypatch, value):
        import importlib
        _reproject = importlib.import_module('xrspatial.reproject')
        monkeypatch.setattr(_reproject, '_REPROJECT_OOM_THRESHOLD', value)

    def test_small_output_stays_numpy(self):
        from xrspatial.reproject import reproject
        raster = _gradient_raster(h=32, w=32)
        result = reproject(raster, 'EPSG:3857')
        assert isinstance(result.data, np.ndarray)

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_large_output_auto_promotes_to_dask(self, monkeypatch):
        """A numpy input whose output exceeds the byte budget comes back
        dask-backed instead of allocating the whole working set."""
        from xrspatial.reproject import reproject

        raster = _gradient_raster(h=32, w=32)

        # Reference result on the eager path (budget effectively disabled).
        self._patch_threshold(monkeypatch, 1 << 60)
        eager = reproject(raster, 'EPSG:3857', width=128, height=128)
        assert isinstance(eager.data, np.ndarray)

        # Tiny budget: the same call must promote to the dask path.
        self._patch_threshold(monkeypatch, 1024)
        lazy = reproject(raster, 'EPSG:3857', width=128, height=128)
        assert isinstance(lazy.data, da.Array)

        computed = lazy.compute()
        np.testing.assert_allclose(
            eager.values, computed.values,
            rtol=1e-5, atol=1e-5, equal_nan=True,
        )
        np.testing.assert_allclose(eager.y.values, computed.y.values)
        np.testing.assert_allclose(eager.x.values, computed.x.values)
        assert eager.attrs['crs'] == computed.attrs['crs']

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_large_output_auto_promotes_3d(self, monkeypatch):
        """The promotion also covers multi-band inputs."""
        from xrspatial.reproject import reproject

        h, w, b = 16, 16, 3
        data = np.random.default_rng(3267).random((h, w, b))
        raster = xr.DataArray(
            data, dims=['y', 'x', 'band'],
            coords={'y': np.linspace(10, 0, h), 'x': np.linspace(0, 10, w),
                    'band': [1, 2, 3]},
            attrs={'crs': 'EPSG:4326'},
        )

        self._patch_threshold(monkeypatch, 1 << 60)
        eager = reproject(raster, 'EPSG:3857', width=64, height=64)

        self._patch_threshold(monkeypatch, 1024)
        lazy = reproject(raster, 'EPSG:3857', width=64, height=64)
        assert isinstance(lazy.data, da.Array)

        np.testing.assert_allclose(
            eager.values, lazy.compute().values,
            rtol=1e-5, atol=1e-5, equal_nan=True,
        )

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_numpy_input_over_pixel_limit_promotes_instead_of_raising(self):
        """A numpy input whose output exceeds 1e9 pixels promotes to the
        lazy path rather than tripping the guard, matching merge() (#3048)."""
        from xrspatial.reproject import reproject

        raster = _make_raster(
            np.arange(64 * 64, dtype='float64').reshape(64, 64),
            crs='EPSG:4326',
            x_range=(-105, -104),
            y_range=(39, 40),
        )

        out = reproject(raster, target_crs='EPSG:3857',
                        resolution=2.0, chunk_size=1024)

        assert isinstance(out.data, da.Array)
        assert out.shape[0] * out.shape[1] > 1_000_000_000
        # A single block computes without materializing the whole grid.
        assert out.data.blocks[0, 0].compute().shape == (1024, 1024)

    def test_inmemory_over_limit_still_raises(self, monkeypatch):
        """With promotion disabled, the pixel-count guard still rejects a
        genuinely in-memory output over the limit."""
        from xrspatial.reproject import reproject

        self._patch_threshold(monkeypatch, 1 << 60)

        raster = _make_raster(
            np.arange(64 * 64, dtype='float64').reshape(64, 64),
            crs='EPSG:4326',
            x_range=(-105, -104),
            y_range=(39, 40),
        )

        with pytest.raises(ValueError, match="too large"):
            reproject(raster, target_crs='EPSG:3857', resolution=2.0)


# =====================================================================
# Issue #1431: _validate_raster on public API inputs
# =====================================================================

class TestValidateRasterInputs:
    """reproject(), merge(), geoid_height_raster() validate inputs (#1431)."""

    def test_reproject_rejects_1d_dataarray(self):
        from xrspatial.reproject import reproject
        bad = xr.DataArray(np.zeros(5, dtype=np.float64), dims=('y',))
        with pytest.raises(ValueError, match=r"must be 2D ?or 3D"):
            reproject(bad, 'EPSG:4326')

    def test_reproject_rejects_complex_dtype(self):
        from xrspatial.reproject import reproject
        bad = xr.DataArray(
            np.zeros((4, 4), dtype=np.complex128),
            dims=('y', 'x'),
            coords={'y': np.arange(4), 'x': np.arange(4)},
        )
        with pytest.raises(ValueError, match="real numeric"):
            reproject(bad, 'EPSG:4326')

    def test_merge_rejects_non_dataarray_element(self):
        from xrspatial.reproject import merge
        good = xr.DataArray(
            np.zeros((4, 4), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': np.arange(4), 'x': np.arange(4)},
        )
        with pytest.raises(TypeError, match="xarray.DataArray"):
            merge([good, np.zeros((4, 4))])

    def test_geoid_height_raster_rejects_non_dataarray(self):
        from xrspatial.reproject import geoid_height_raster
        with pytest.raises(TypeError, match="xarray.DataArray"):
            geoid_height_raster(np.zeros((4, 4)))

    def test_geoid_height_raster_rejects_1d_dataarray(self):
        from xrspatial.reproject import geoid_height_raster
        bad = xr.DataArray(np.zeros(5, dtype=np.float64), dims=('y',))
        with pytest.raises(ValueError, match=r"must be 2D ?or 3D"):
            geoid_height_raster(bad)


# =====================================================================
# Issue #1433: grid/bounds/precision parameter validation
# =====================================================================

class TestValidateGridParams:
    """reproject(): grid params reject zero / negative / non-finite."""

    @staticmethod
    def _good_raster():
        return xr.DataArray(
            np.zeros((4, 4), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': np.arange(4), 'x': np.arange(4)},
            attrs={'crs': 'EPSG:4326'},
        )

    @pytest.mark.parametrize("res", [0, 0.0, -1, -2.5,
                                     float('inf'), float('-inf'),
                                     float('nan')])
    def test_resolution_rejected(self, res):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="resolution"):
            reproject(r, 'EPSG:4326', resolution=res)

    def test_resolution_tuple_with_zero_rejected(self):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="resolution"):
            reproject(r, 'EPSG:4326', resolution=(1.0, 0.0))

    def test_resolution_tuple_wrong_length_rejected(self):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="length 2"):
            reproject(r, 'EPSG:4326', resolution=(1.0, 2.0, 3.0))

    @pytest.mark.parametrize("w", [0, -1, 1.5])
    def test_width_rejected(self, w):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="width"):
            reproject(r, 'EPSG:4326', width=w, height=10)

    @pytest.mark.parametrize("h", [0, -1, 1.5])
    def test_height_rejected(self, h):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="height"):
            reproject(r, 'EPSG:4326', width=10, height=h)

    def test_bounds_collapsed_x_rejected(self):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="right"):
            reproject(r, 'EPSG:4326', bounds=(10, 0, 10, 10))

    def test_bounds_collapsed_y_rejected(self):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="top"):
            reproject(r, 'EPSG:4326', bounds=(0, 10, 10, 10))

    def test_bounds_inverted_x_rejected(self):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="right"):
            reproject(r, 'EPSG:4326', bounds=(10, 0, 0, 10))

    def test_bounds_nan_rejected(self):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="finite"):
            reproject(r, 'EPSG:4326', bounds=(0, 0, float('nan'), 10))

    def test_bounds_wrong_length_rejected(self):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="4-tuple"):
            reproject(r, 'EPSG:4326', bounds=(0, 0, 10))

    def test_transform_precision_negative_rejected(self):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="transform_precision"):
            reproject(r, 'EPSG:4326', transform_precision=-1)

    def test_transform_precision_float_rejected(self):
        from xrspatial.reproject import reproject
        r = self._good_raster()
        with pytest.raises(ValueError, match="transform_precision"):
            reproject(r, 'EPSG:4326', transform_precision=1.5)


class TestValidateMergeGridParams:
    @staticmethod
    def _raster():
        return xr.DataArray(
            np.zeros((4, 4), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': np.arange(4), 'x': np.arange(4)},
            attrs={'crs': 'EPSG:4326'},
        )

    def test_merge_resolution_rejected(self):
        from xrspatial.reproject import merge
        with pytest.raises(ValueError, match="resolution"):
            merge([self._raster()], resolution=-1.0)

    def test_merge_bounds_rejected(self):
        from xrspatial.reproject import merge
        with pytest.raises(ValueError, match="right"):
            merge([self._raster()], bounds=(10, 0, 0, 10))

    def test_merge_accepts_transform_precision_zero(self):
        """``transform_precision=0`` requests exact per-pixel transforms."""
        from xrspatial.reproject import merge
        raster = _gradient_raster(h=8, w=8)
        result = merge([raster], transform_precision=0, resolution=1.0)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_merge_accepts_transform_precision_default(self):
        """Default ``transform_precision`` (16) leaves merge() callable."""
        from xrspatial.reproject import merge
        raster = _gradient_raster(h=8, w=8)
        result = merge([raster], resolution=1.0)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_merge_rejects_negative_transform_precision(self):
        from xrspatial.reproject import merge
        with pytest.raises(ValueError, match="transform_precision"):
            merge([self._raster()], transform_precision=-1)

    def test_merge_rejects_float_transform_precision(self):
        from xrspatial.reproject import merge
        with pytest.raises(ValueError, match="transform_precision"):
            merge([self._raster()], transform_precision=1.5)

    def test_merge_transform_precision_threaded_to_chunks(self):
        """precision=0 (exact) and precision=16 should agree on smooth inputs.

        For inputs where the control-grid approximation is already very
        close to the per-pixel transform, the two paths should give the
        same merged output to floating-point tolerance.
        """
        from xrspatial.reproject import merge
        # Two adjacent same-CRS gradients in EPSG:4326 reprojected to
        # the same CRS: the control grid is dense enough that precision=16
        # and precision=0 produce identical numbers.
        a = _gradient_raster(h=16, w=16, x_range=(-5, 0), y_range=(-5, 5))
        b = _gradient_raster(h=16, w=16, x_range=(0, 5), y_range=(-5, 5))
        out16 = merge([a, b], target_crs='EPSG:4326',
                      resolution=1.0, transform_precision=16)
        out0 = merge([a, b], target_crs='EPSG:4326',
                     resolution=1.0, transform_precision=0)
        assert out16.shape == out0.shape
        v16 = out16.values
        v0 = out0.values
        valid = ~np.isnan(v16) & ~np.isnan(v0)
        assert valid.any()
        np.testing.assert_allclose(v0[valid], v16[valid], rtol=1e-10)


# =====================================================================
# Issue #2184: irregular / non-monotonic source coords are rejected
# =====================================================================


def _regular_raster(h=8, w=8):
    """Strictly regular raster used as the baseline in coord-validation tests."""
    return _gradient_raster(h=h, w=w)


class TestValidateSourceCoords:
    """reproject() and merge() reject irregular / non-monotonic source coords."""

    # ------------------------------------------------------------------
    # Positive cases: well-formed inputs pass through.
    # ------------------------------------------------------------------

    def test_reproject_accepts_regular_descending_y(self):
        from xrspatial.reproject import reproject
        # _gradient_raster builds y descending (north-up), x ascending.
        out = reproject(_regular_raster(), 'EPSG:4326', resolution=1.0)
        assert out.shape[0] > 0 and out.shape[1] > 0

    def test_reproject_accepts_regular_ascending_y(self):
        from xrspatial.reproject import reproject
        h, w = 8, 8
        y = np.linspace(-5, 5, h)   # ascending
        x = np.linspace(-5, 5, w)
        raster = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        out = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert out.shape[0] > 0 and out.shape[1] > 0

    def test_reproject_accepts_tiny_floating_drift(self):
        """Coords from real-world GeoTIFFs drift a few ULPs; that must pass."""
        from xrspatial.reproject import reproject
        h, w = 8, 8
        y = np.linspace(5, -5, h)
        x = np.linspace(-5, 5, w)
        # Inject sub-ULP-scale drift well below the 1e-6 relative tolerance.
        rng = np.random.default_rng(0)
        x = x + rng.uniform(-1e-10, 1e-10, size=w)
        raster = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        out = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert out.shape[0] > 0 and out.shape[1] > 0

    def test_reproject_accepts_single_pixel_raster(self):
        """Single-pixel rasters have no spacing to validate."""
        from xrspatial.reproject import reproject
        raster = xr.DataArray(
            np.zeros((1, 1), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': [0.0], 'x': [0.0]},
            attrs={'crs': 'EPSG:4326'},
        )
        # Should not raise; output grid math falls back to res=1.0.
        out = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert out.size >= 1

    # ------------------------------------------------------------------
    # Irregular spacing.
    # ------------------------------------------------------------------

    def test_reproject_rejects_irregular_x(self):
        from xrspatial.reproject import reproject
        h, w = 8, 8
        y = np.linspace(5, -5, h)
        x = np.linspace(-5, 5, w)
        x[4] += 0.1  # perturb one sample
        raster = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"coordinate 'x' is not regularly"):
            reproject(raster, 'EPSG:3857')

    def test_reproject_rejects_irregular_y(self):
        from xrspatial.reproject import reproject
        h, w = 8, 8
        y = np.linspace(5, -5, h)
        y[3] += 0.05
        x = np.linspace(-5, 5, w)
        raster = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"coordinate 'y' is not regularly"):
            reproject(raster, 'EPSG:3857')

    def test_reproject_irregular_error_names_index(self):
        """The error message points at the offending sample index."""
        from xrspatial.reproject import reproject
        h, w = 8, 8
        y = np.linspace(5, -5, h)
        x = np.linspace(-5, 5, w)
        x[5] += 0.2
        raster = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError) as exc:
            reproject(raster, 'EPSG:3857')
        msg = str(exc.value)
        # Step index 4 (x[4]->x[5]) or 5 (x[5]->x[6]) is the worst,
        # both touch the perturbed sample.
        assert "at index 4" in msg or "at index 5" in msg
        assert "Median step" in msg

    # ------------------------------------------------------------------
    # Non-monotonic coords.
    # ------------------------------------------------------------------

    def test_reproject_rejects_non_monotonic_x(self):
        from xrspatial.reproject import reproject
        h, w = 4, 4
        y = np.linspace(5, -5, h)
        x = np.array([0.0, 1.0, 0.5, 2.0])
        raster = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"coordinate 'x' must be strictly"):
            reproject(raster, 'EPSG:3857')

    def test_reproject_rejects_non_monotonic_y(self):
        from xrspatial.reproject import reproject
        h, w = 4, 4
        y = np.array([0.0, 1.0, 0.5, 2.0])
        x = np.linspace(-5, 5, w)
        raster = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"coordinate 'y' must be strictly"):
            reproject(raster, 'EPSG:3857')

    def test_reproject_rejects_repeated_coord(self):
        """Repeated values break strict monotonicity (zero step)."""
        from xrspatial.reproject import reproject
        h, w = 4, 4
        y = np.linspace(5, -5, h)
        x = np.array([0.0, 1.0, 1.0, 2.0])
        raster = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError,
                           match=r"coordinate 'x' must be strictly monotonic"):
            reproject(raster, 'EPSG:3857')

    def test_reproject_rejects_nan_in_coord(self):
        from xrspatial.reproject import reproject
        h, w = 4, 4
        y = np.linspace(5, -5, h)
        x = np.array([0.0, 1.0, np.nan, 3.0])
        raster = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"non-finite"):
            reproject(raster, 'EPSG:3857')

    # ------------------------------------------------------------------
    # Validation runs before expensive work.
    # ------------------------------------------------------------------

    def test_reproject_rejects_irregular_before_crs_resolution(self):
        """Bad coords must be caught even when source_crs is unresolvable.

        If validation ran after CRS resolution, an irregular raster with no
        CRS attribute would raise the "Could not detect source CRS" error
        first, hiding the real defect.
        """
        from xrspatial.reproject import reproject
        h, w = 4, 4
        y = np.linspace(5, -5, h)
        x = np.array([0.0, 1.0, 1.5, 2.0])  # irregular
        raster = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            # NB: no crs attr -- detection would normally raise here.
        )
        with pytest.raises(ValueError, match=r"not regularly"):
            reproject(raster, 'EPSG:3857')

    # ------------------------------------------------------------------
    # merge() applies the same checks.
    # ------------------------------------------------------------------

    def test_merge_rejects_irregular_x(self):
        from xrspatial.reproject import merge
        good = _regular_raster()
        h, w = 8, 8
        y = np.linspace(5, -5, h)
        x = np.linspace(-5, 5, w)
        x[4] += 0.1
        bad = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"rasters\[1\].*coordinate 'x'"):
            merge([good, bad], resolution=1.0)

    def test_merge_rejects_non_monotonic_y(self):
        from xrspatial.reproject import merge
        h, w = 4, 4
        y = np.array([0.0, 1.0, 0.5, 2.0])
        x = np.linspace(-5, 5, w)
        bad = xr.DataArray(
            np.zeros((h, w), dtype=np.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"coordinate 'y' must be strictly"):
            merge([bad], resolution=1.0)

    # ------------------------------------------------------------------
    # Backends: validation fires identically regardless of array type.
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not HAS_DASK, reason="dask not installed")
    def test_reproject_rejects_irregular_dask(self):
        from xrspatial.reproject import reproject
        h, w = 8, 8
        y = np.linspace(5, -5, h)
        x = np.linspace(-5, 5, w)
        x[4] += 0.1
        raster = xr.DataArray(
            da.zeros((h, w), dtype=np.float64, chunks=(4, 4)),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"not regularly"):
            reproject(raster, 'EPSG:3857')

    @pytest.mark.skipif(not HAS_CUPY, reason="cupy not installed")
    def test_reproject_rejects_irregular_cupy(self):
        from xrspatial.reproject import reproject
        h, w = 8, 8
        y = np.linspace(5, -5, h)
        x = np.linspace(-5, 5, w)
        x[4] += 0.1
        raster = xr.DataArray(
            cp.zeros((h, w), dtype=cp.float64),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"not regularly"):
            reproject(raster, 'EPSG:3857')

    @pytest.mark.skipif(not (HAS_DASK and HAS_CUPY),
                        reason="dask and cupy required")
    def test_reproject_rejects_irregular_dask_cupy(self):
        from xrspatial.reproject import reproject
        h, w = 8, 8
        y = np.linspace(5, -5, h)
        x = np.linspace(-5, 5, w)
        x[4] += 0.1
        raster = xr.DataArray(
            da.from_array(cp.zeros((h, w), dtype=cp.float64), chunks=(4, 4)),
            dims=('y', 'x'),
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"not regularly"):
            reproject(raster, 'EPSG:3857')


# =====================================================================
# Issue #1435: NaN/Inf rejection in scalar inputs
# =====================================================================

class TestItrfFiniteness:
    @pytest.mark.parametrize("epoch", [float('nan'), float('inf'), float('-inf')])
    def test_itrf_rejects_non_finite_epoch(self, epoch):
        from xrspatial.reproject import itrf_transform
        with pytest.raises(ValueError, match="epoch"):
            itrf_transform(0.0, 0.0, 0.0,
                           src='ITRF2014', tgt='ITRF2020', epoch=epoch)

    def test_itrf_rejects_empty_src(self):
        from xrspatial.reproject import itrf_transform
        with pytest.raises(ValueError, match="src"):
            itrf_transform(0.0, 0.0, 0.0,
                           src='', tgt='ITRF2020', epoch=2024.0)

    def test_itrf_rejects_empty_tgt(self):
        from xrspatial.reproject import itrf_transform
        with pytest.raises(ValueError, match="tgt"):
            itrf_transform(0.0, 0.0, 0.0,
                           src='ITRF2014', tgt='', epoch=2024.0)


class TestGeoidFiniteness:
    @pytest.mark.parametrize("lon", [float('nan'), float('inf')])
    def test_geoid_rejects_non_finite_lon(self, lon):
        from xrspatial.reproject import geoid_height
        with pytest.raises(ValueError, match="lon"):
            geoid_height(lon, 0.0)

    @pytest.mark.parametrize("lat", [float('nan'), float('inf')])
    def test_geoid_rejects_non_finite_lat(self, lat):
        from xrspatial.reproject import geoid_height
        with pytest.raises(ValueError, match="lat"):
            geoid_height(0.0, lat)

    @pytest.mark.parametrize("lat", [-91.0, 91.0])
    def test_geoid_rejects_out_of_range_lat(self, lat):
        from xrspatial.reproject import geoid_height
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            geoid_height(0.0, lat)

    def test_geoid_rejects_array_with_nan(self):
        from xrspatial.reproject import geoid_height
        lon = np.array([0.0, float('nan'), 10.0])
        lat = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="lon"):
            geoid_height(lon, lat)


# ---------------------------------------------------------------------------
# Shape-mismatch validation in geoid_height and itrf_transform (#2026)
# ---------------------------------------------------------------------------

class TestGeoidShapeMismatch:
    """geoid_height must reject lon/lat with mismatched shapes (#2026).

    Without the check the numba @njit(parallel=True) kernel reads past the
    end of the shorter array and silently returns wrong values.
    """

    def test_geoid_rejects_1d_mismatch(self):
        from xrspatial.reproject import geoid_height
        lon = np.array([0.0, 90.0, 45.0])
        lat = np.array([0.0, 45.0])
        with pytest.raises(ValueError, match="same shape"):
            geoid_height(lon, lat)

    def test_geoid_rejects_2d_mismatch(self):
        from xrspatial.reproject import geoid_height
        lon = np.zeros((3, 4))
        lat = np.zeros((4, 3))
        with pytest.raises(ValueError, match="same shape"):
            geoid_height(lon, lat)

    def test_geoid_rejects_scalar_lat_array_lon(self):
        # 0-D and 1-D have different shapes; reject before raveling.
        from xrspatial.reproject import geoid_height
        with pytest.raises(ValueError, match="same shape"):
            geoid_height(np.array([0.0, 10.0]), 0.0)

    def test_geoid_accepts_matching_1d(self):
        from xrspatial.reproject import geoid_height
        lon = np.array([0.0, 90.0, 45.0])
        lat = np.array([0.0, 45.0, 30.0])
        result = geoid_height(lon, lat)
        assert result.shape == (3,)
        assert np.isfinite(result).all()

    def test_geoid_accepts_matching_2d(self):
        from xrspatial.reproject import geoid_height
        lon = np.array([[0.0, 10.0], [20.0, 30.0]])
        lat = np.array([[0.0, 5.0], [10.0, 15.0]])
        result = geoid_height(lon, lat)
        assert result.shape == (2, 2)

    def test_geoid_accepts_scalar_pair(self):
        from xrspatial.reproject import geoid_height
        # Both scalar -- should still work and return a Python float.
        result = geoid_height(0.0, 0.0)
        assert isinstance(result, float)


class TestItrfShapeMismatch:
    """itrf_transform must reject lon/lat with mismatched shapes (#2026)."""

    def test_itrf_rejects_1d_mismatch(self):
        from xrspatial.reproject import itrf_transform
        lon = np.array([-74.0, 0.0, 45.0])
        lat = np.array([40.7, 0.0])
        with pytest.raises(ValueError, match="same shape"):
            itrf_transform(lon, lat,
                           src='ITRF2014', tgt='ITRF2020', epoch=2024.0)

    def test_itrf_rejects_2d_mismatch(self):
        from xrspatial.reproject import itrf_transform
        lon = np.zeros((3, 4))
        lat = np.zeros((4, 3))
        with pytest.raises(ValueError, match="same shape"):
            itrf_transform(lon, lat,
                           src='ITRF2014', tgt='ITRF2020', epoch=2024.0)

    def test_itrf_accepts_matching_1d(self):
        from xrspatial.reproject import itrf_transform
        lon = np.array([-74.0, 0.0, 45.0])
        lat = np.array([40.7, 0.0, 10.0])
        # Default h=0 is scalar and broadcasts.
        out_lon, out_lat, out_h = itrf_transform(
            lon, lat, src='ITRF2014', tgt='ITRF2020', epoch=2024.0,
        )
        assert out_lon.shape == (3,)
        assert out_lat.shape == (3,)
        assert out_h.shape == (3,)

    def test_itrf_accepts_scalar_h_with_array_lonlat(self):
        # 0-D h must still broadcast to lon's 1-D shape.
        from xrspatial.reproject import itrf_transform
        lon = np.array([-74.0, 0.0])
        lat = np.array([40.7, 0.0])
        out_lon, out_lat, out_h = itrf_transform(
            lon, lat, h=10.0,
            src='ITRF2014', tgt='ITRF2020', epoch=2024.0,
        )
        assert out_lon.shape == (2,)

    def test_itrf_accepts_matching_h(self):
        from xrspatial.reproject import itrf_transform
        lon = np.array([-74.0, 0.0])
        lat = np.array([40.7, 0.0])
        h = np.array([10.0, 20.0])
        out_lon, out_lat, out_h = itrf_transform(
            lon, lat, h=h,
            src='ITRF2014', tgt='ITRF2020', epoch=2024.0,
        )
        assert out_lon.shape == (2,)
        assert out_h.shape == (2,)

    def test_itrf_rejects_non_broadcastable_h(self):
        from xrspatial.reproject import itrf_transform
        lon = np.array([-74.0, 0.0, 45.0])
        lat = np.array([40.7, 0.0, 10.0])
        h = np.array([1.0, 2.0])  # length 2 vs lon length 3
        with pytest.raises(ValueError, match="broadcast"):
            itrf_transform(lon, lat, h=h,
                           src='ITRF2014', tgt='ITRF2020', epoch=2024.0)

    def test_itrf_rejects_multidim_h_vs_1d_lonlat(self):
        # h=(1,3) vs lon=(3,) used to slip past the broadcast_shapes
        # pre-check (they broadcast to (1,3)) and then fail downstream
        # with numpy's raw broadcast_to error against the raveled 1-D
        # lon_arr. Confirm the public API now raises with shape info.
        from xrspatial.reproject import itrf_transform
        lon = np.array([-74.0, 0.0, 45.0])
        lat = np.array([40.7, 0.0, 10.0])
        h = np.array([[5.0, 6.0, 7.0]])
        with pytest.raises(ValueError, match=r"h shape .* lon shape"):
            itrf_transform(lon, lat, h=h,
                           src='ITRF2014', tgt='ITRF2020', epoch=2024.0)


class TestNodataFiniteness:
    def test_detect_nodata_rejects_inf(self):
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(np.zeros((4, 4)), dims=('y', 'x'))
        with pytest.raises(ValueError, match="nodata"):
            _detect_nodata(r, nodata=float('inf'))

    def test_detect_nodata_rejects_neg_inf(self):
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(np.zeros((4, 4)), dims=('y', 'x'))
        with pytest.raises(ValueError, match="nodata"):
            _detect_nodata(r, nodata=float('-inf'))

    def test_detect_nodata_accepts_nan(self):
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(np.zeros((4, 4)), dims=('y', 'x'))
        nd = _detect_nodata(r, nodata=float('nan'))
        assert np.isnan(nd)

    def test_detect_nodata_accepts_finite(self):
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(np.zeros((4, 4)), dims=('y', 'x'))
        assert _detect_nodata(r, nodata=-9999) == -9999.0


def _egm2008_available():
    """Return True if the EGM2008 grid can be loaded."""
    try:
        from xrspatial.reproject._vertical import _load_geoid
        _load_geoid('EGM2008')
        return True
    except (FileNotFoundError, OSError, Exception):
        return False


class TestVerticalShift:
    """End-to-end coverage for source_vertical_crs / target_vertical_crs."""

    def _ny_raster(self, h=8, w=8, value=100.0, nodata=np.nan):
        # Small raster centred on New York. EGM96 undulation there is ~-33 m.
        y = np.linspace(41.1, 40.3, h)
        x = np.linspace(-74.4, -73.6, w)
        data = np.full((h, w), value, dtype=np.float64)
        return xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': nodata},
        )

    def test_reproject_egm96_to_ellipsoidal(self):
        """Orthometric to ellipsoidal: output = input + N (negative near NY)."""
        from xrspatial.reproject import reproject, geoid_height
        raster = self._ny_raster(value=100.0)
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96', target_vertical_crs='ellipsoidal',
        )
        # Reference undulation at the centre.
        cy = float(result.coords['y'].values[result.shape[0] // 2])
        cx = float(result.coords['x'].values[result.shape[1] // 2])
        N = geoid_height(cx, cy, model='EGM96')
        assert N < 0  # geoid below ellipsoid in NY
        cval = float(result.values[result.shape[0] // 2, result.shape[1] // 2])
        # 100 m orthometric + N -> ~67 m ellipsoidal. Allow generous tolerance.
        assert abs(cval - (100.0 + N)) < 1.0
        # vertical_crs now records the EPSG code (4979 = WGS84 3D
        # ellipsoidal), matching the xrspatial.geotiff convention; the
        # friendly token is preserved under vertical_datum.
        assert result.attrs.get('vertical_crs') == 4979
        assert result.attrs.get('vertical_datum') == 'ellipsoidal'

    def test_reproject_ellipsoidal_to_egm96(self):
        """Ellipsoidal to orthometric: shift has the opposite sign."""
        from xrspatial.reproject import reproject, geoid_height
        raster = self._ny_raster(value=100.0)
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='ellipsoidal', target_vertical_crs='EGM96',
        )
        cy = float(result.coords['y'].values[result.shape[0] // 2])
        cx = float(result.coords['x'].values[result.shape[1] // 2])
        N = geoid_height(cx, cy, model='EGM96')
        cval = float(result.values[result.shape[0] // 2, result.shape[1] // 2])
        # 100 m ellipsoidal - N -> ~133 m orthometric.
        assert abs(cval - (100.0 - N)) < 1.0

    @pytest.mark.skipif(
        not _egm2008_available(),
        reason="EGM2008 grid not available",
    )
    def test_reproject_egm96_to_egm2008(self):
        """Two geoid-based vertical CRSes: shift is small everywhere."""
        from xrspatial.reproject import reproject
        raster = self._ny_raster(value=100.0)
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96', target_vertical_crs='EGM2008',
        )
        diffs = result.values - 100.0
        # EGM96 vs EGM2008 differ by under 2 m globally.
        assert np.all(np.abs(diffs) < 2.0)

    def test_reproject_no_vertical_shift_when_same(self):
        """Identical src and tgt vertical CRS leaves values untouched."""
        from xrspatial.reproject import reproject
        raster = self._ny_raster(value=100.0)
        baseline = reproject(raster, 'EPSG:4326')
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96', target_vertical_crs='EGM96',
        )
        np.testing.assert_array_equal(result.values, baseline.values)

    def test_reproject_no_vertical_shift_when_one_none(self):
        """Only one side set -> no shift applied."""
        from xrspatial.reproject import reproject
        raster = self._ny_raster(value=100.0)
        baseline = reproject(raster, 'EPSG:4326')
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96',
            target_vertical_crs=None,
        )
        np.testing.assert_array_equal(result.values, baseline.values)

    def test_reproject_vertical_shift_with_projected_crs(self):
        """Projected target exercises the inverse-projection branch."""
        from xrspatial.reproject import reproject
        # Build a raster in UTM 33N (around 12 E, 48 N). EGM96 N is ~46 m there.
        h, w = 8, 8
        data = np.full((h, w), 100.0, dtype=np.float64)
        # ~10 km box near Vienna in EPSG:32633.
        y = np.linspace(5_330_000, 5_320_000, h)
        x = np.linspace(595_000, 605_000, w)
        raster = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:32633', 'nodata': np.nan},
        )
        result = reproject(
            raster, 'EPSG:32633',
            source_vertical_crs='EGM96', target_vertical_crs='ellipsoidal',
        )
        vals = result.values
        finite = vals[np.isfinite(vals)]
        assert finite.size > 0
        # Shift over central Europe is roughly 40-50 m.
        shifts = finite - 100.0
        assert np.all(shifts > 30.0)
        assert np.all(shifts < 60.0)

    def test_reproject_vertical_shift_handles_polar_singularity(self):
        """Regression test: polar-stereographic inverse can emit non-finite
        coords near the pole; the call must not hang on the inf longitude
        wrap loop in _interp_geoid_point."""
        from xrspatial.reproject import reproject
        # Source raster spans 89 to 90 N in lon/lat.
        h, w = 8, 16
        y = np.linspace(90.0, 89.0, h)
        x = np.linspace(-180.0, 180.0, w)
        data = np.full((h, w), 100.0, dtype=np.float64)
        raster = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        # EPSG:3413 is North Polar Stereographic. The inverse transform at
        # x=y=0 maps to the pole, which often returns inf longitude.
        result = reproject(
            raster, 'EPSG:3413',
            source_vertical_crs='EGM96', target_vertical_crs='ellipsoidal',
        )
        # Must produce some finite output where the source had finite values.
        assert np.isfinite(result.values).any()
        # NaN at the singularity is acceptable; inf is not.
        assert not np.isinf(result.values).any()

    def test_vertical_crs_attr_is_epsg_int(self):
        """attrs['vertical_crs'] must be an EPSG int to match xrspatial.geotiff.

        Both ``xrspatial.geotiff.open_geotiff()`` and ``reproject()`` write
        the ``vertical_crs`` attribute. The geotiff path writes the EPSG
        integer code, so reproject must do the same. The friendly string
        token is preserved under ``vertical_datum``. See GH #1570.
        """
        from xrspatial.reproject import reproject
        cases = [
            ('EGM96', 5773),
            ('EGM2008', 3855),
            ('ellipsoidal', 4979),
        ]
        for tgt, expected_epsg in cases:
            raster = self._ny_raster(value=10.0)
            result = reproject(
                raster, 'EPSG:4326',
                source_vertical_crs='EGM96', target_vertical_crs=tgt,
            )
            assert result.attrs.get('vertical_crs') == expected_epsg, (
                f"vertical_crs for tgt={tgt!r} should be EPSG {expected_epsg}, "
                f"got {result.attrs.get('vertical_crs')!r}"
            )
            assert isinstance(result.attrs.get('vertical_crs'), int)
            assert result.attrs.get('vertical_datum') == tgt

    def test_unknown_vertical_crs_raises(self):
        """Typos / unsupported tokens must raise rather than silently
        write ``attrs['vertical_crs'] = None``."""
        from xrspatial.reproject import reproject
        raster = self._ny_raster(value=10.0)
        with pytest.raises(ValueError, match="target_vertical_crs"):
            reproject(raster, 'EPSG:4326',
                      source_vertical_crs='EGM96', target_vertical_crs='NAVD88')
        with pytest.raises(ValueError, match="source_vertical_crs"):
            reproject(raster, 'EPSG:4326',
                      source_vertical_crs='egm96',  # case-sensitive
                      target_vertical_crs='ellipsoidal')

    def test_deprecated_vertical_kwargs_still_work(self):
        """Old src_/tgt_vertical_crs names map to the new ones with a warning.

        Renamed to source_/target_vertical_crs for consistency with the
        source_crs/target_crs spelling (#2613). The old names stay working
        through a deprecation shim.
        """
        from xrspatial.reproject import reproject
        raster = self._ny_raster(value=100.0)
        new = reproject(raster, 'EPSG:4326',
                        source_vertical_crs='EGM96',
                        target_vertical_crs='ellipsoidal')
        with pytest.warns(DeprecationWarning, match="src_vertical_crs"):
            old_src = reproject(raster, 'EPSG:4326',
                                src_vertical_crs='EGM96',
                                target_vertical_crs='ellipsoidal')
        with pytest.warns(DeprecationWarning, match="tgt_vertical_crs"):
            old_tgt = reproject(raster, 'EPSG:4326',
                                source_vertical_crs='EGM96',
                                tgt_vertical_crs='ellipsoidal')
        np.testing.assert_array_equal(old_src.values, new.values)
        np.testing.assert_array_equal(old_tgt.values, new.values)

    def test_deprecated_and_new_vertical_kwarg_conflict(self):
        """Passing both the old and new spelling for one side is an error."""
        from xrspatial.reproject import reproject
        raster = self._ny_raster(value=100.0)
        with pytest.warns(DeprecationWarning):
            with pytest.raises(TypeError, match="not both"):
                reproject(raster, 'EPSG:4326',
                          source_vertical_crs='EGM96',
                          src_vertical_crs='EGM96',
                          target_vertical_crs='ellipsoidal')

    def test_dask_backend_matches_numpy(self):
        """Dask-backed input must apply the vertical shift correctly (#2025).

        Boolean fancy indexing on a dask array used to crash; the dask
        path now runs through ``map_blocks`` and matches the numpy result
        bit-for-bit.
        """
        import dask.array as da
        from xrspatial.reproject import reproject

        np.random.seed(0)
        host = (np.random.rand(48, 48) * 100).astype(np.float64)
        ds = xr.DataArray(
            host, dims=['y', 'x'],
            coords={'y': np.linspace(41.1, 40.3, 48),
                    'x': np.linspace(-74.4, -73.6, 48)},
            attrs={'crs': 'EPSG:4326'},
        )
        ds_d = xr.DataArray(
            da.from_array(host, chunks=(16, 16)), dims=['y', 'x'],
            coords=ds.coords, attrs=ds.attrs,
        )
        out_np = reproject(ds, 'EPSG:4326',
                           source_vertical_crs='EGM96',
                           target_vertical_crs='ellipsoidal')
        out_da = reproject(ds_d, 'EPSG:4326',
                           source_vertical_crs='EGM96',
                           target_vertical_crs='ellipsoidal')
        # Output is still dask-backed so the graph stays lazy.
        assert isinstance(out_da.data, da.Array)
        np.testing.assert_allclose(
            np.asarray(out_da.data), out_np.values, rtol=0, atol=1e-12,
        )

    def test_multiband_3d_applies_shift_per_band(self):
        """3-D (y, x, band) result must apply the same N per pixel to
        every band (#2025).

        The earlier per-strip boolean update raised a broadcasting
        ValueError for any 3-D source. The shift now loops over bands.
        """
        from xrspatial.reproject import reproject

        np.random.seed(1)
        data = (np.random.rand(48, 48, 3) * 100).astype(np.float64)
        raster = xr.DataArray(
            data, dims=['y', 'x', 'band'],
            coords={'y': np.linspace(41.1, 40.3, 48),
                    'x': np.linspace(-74.4, -73.6, 48),
                    'band': [1, 2, 3]},
            attrs={'crs': 'EPSG:4326'},
        )
        result = reproject(raster, 'EPSG:4326',
                           source_vertical_crs='EGM96',
                           target_vertical_crs='ellipsoidal')
        assert result.shape == (48, 48, 3)

        # Same N applied to every band -> inter-band differences are
        # preserved up to interpolation noise.
        for b in range(1, 3):
            diff_in = data[:, :, 0] - data[:, :, b]
            diff_out = result.values[:, :, 0] - result.values[:, :, b]
            np.testing.assert_allclose(diff_out, diff_in, rtol=0, atol=1e-6)

        # Reference: band 0 should equal the 2-D shift result.
        raster_2d = xr.DataArray(
            data[:, :, 0], dims=['y', 'x'],
            coords={'y': raster.coords['y'], 'x': raster.coords['x']},
            attrs={'crs': 'EPSG:4326'},
        )
        out_2d = reproject(raster_2d, 'EPSG:4326',
                           source_vertical_crs='EGM96',
                           target_vertical_crs='ellipsoidal')
        np.testing.assert_allclose(
            result.values[:, :, 0], out_2d.values, rtol=0, atol=1e-9,
        )

    def test_cupy_backend_matches_numpy(self):
        """CuPy-backed input must apply the vertical shift correctly (#2025).

        The CPU JIT geoid lookup cannot accept cupy arrays directly; the
        shift now round-trips through host and returns cupy output. Only
        the vertical-shift increment is compared so this test does not
        require the cupy reproject path to match numpy bit-for-bit (which
        is tracked separately).
        """
        cp = pytest.importorskip('cupy')
        from xrspatial.reproject import reproject

        host = np.full((32, 32), 100.0, dtype=np.float64)
        ds_np = xr.DataArray(
            host, dims=['y', 'x'],
            coords={'y': np.linspace(41.1, 40.3, 32),
                    'x': np.linspace(-74.4, -73.6, 32)},
            attrs={'crs': 'EPSG:4326'},
        )
        ds_cu = xr.DataArray(
            cp.asarray(host), dims=['y', 'x'],
            coords=ds_np.coords, attrs=ds_np.attrs,
        )

        base_np = reproject(ds_np, 'EPSG:4326')
        shifted_np = reproject(ds_np, 'EPSG:4326',
                               source_vertical_crs='EGM96',
                               target_vertical_crs='ellipsoidal')
        delta_np = shifted_np.values - base_np.values

        base_cu = reproject(ds_cu, 'EPSG:4326')
        shifted_cu = reproject(ds_cu, 'EPSG:4326',
                               source_vertical_crs='EGM96',
                               target_vertical_crs='ellipsoidal')
        assert isinstance(shifted_cu.data, cp.ndarray)
        delta_cu = (cp.asnumpy(shifted_cu.data)
                    - cp.asnumpy(base_cu.data))

        # Guard against a silent no-op regression: if the cupy shift
        # ever fails to fire, delta_cu collapses to zero and the
        # cross-backend allclose below would still pass wherever
        # delta_np is also zero.
        assert np.any(np.abs(delta_cu) > 0), (
            "vertical shift did not fire on cupy backend"
        )

        # The increment from the geoid shift must agree across backends.
        finite = np.isfinite(delta_np) & np.isfinite(delta_cu)
        np.testing.assert_allclose(
            delta_cu[finite], delta_np[finite], rtol=0, atol=1e-9,
        )


class TestVerticalShiftIntegerDtype:
    """Vertical shift must work on integer DEMs by promoting to float (#2565).

    Real-world DEM products (SRTM, ASTER GDEM, Copernicus DEM) ship as
    int16, so the vertical-shift path used to crash with
    ``UFuncTypeError`` when callers asked for a geoid -> ellipsoidal
    transform. The fix promotes the array to a float dtype before the
    shift and rewrites the integer nodata sentinel to NaN.
    """

    def _ny_raster_int(self, dtype, value, nodata, h=8, w=8):
        # Same NY footprint as TestVerticalShift._ny_raster so the
        # reference geoid undulation is known to be ~-33 m there.
        y = np.linspace(41.1, 40.3, h)
        x = np.linspace(-74.4, -73.6, w)
        data = np.full((h, w), value, dtype=dtype)
        return xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': nodata},
        )

    def test_int16_promotes_to_float32(self):
        """int16 DEM with EGM96 -> ellipsoidal shift returns float32."""
        from xrspatial.reproject import reproject, geoid_height
        raster = self._ny_raster_int(np.int16, 100, -32768)
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96', target_vertical_crs='ellipsoidal',
        )
        assert result.dtype == np.float32, (
            f"expected float32 output for int16 input, got {result.dtype}"
        )
        # Nodata sentinel should follow the dtype promotion.
        assert np.isnan(result.attrs['nodata']), (
            f"expected NaN nodata after promotion, got {result.attrs['nodata']!r}"
        )
        # Numerical check against the known reference undulation.
        cy = float(result.coords['y'].values[result.shape[0] // 2])
        cx = float(result.coords['x'].values[result.shape[1] // 2])
        N = geoid_height(cx, cy, model='EGM96')
        cval = float(result.values[result.shape[0] // 2, result.shape[1] // 2])
        # Allow loose tolerance for float32 plus interpolation noise.
        assert abs(cval - (100.0 + N)) < 1.0, (
            f"shifted value {cval} not within 1 m of expected {100.0 + N}"
        )

    def test_uint8_promotes_to_float32(self):
        """uint8 raster also promotes to float32 (covers small unsigned ints)."""
        from xrspatial.reproject import reproject
        raster = self._ny_raster_int(np.uint8, 100, 0)
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96', target_vertical_crs='ellipsoidal',
        )
        assert result.dtype == np.float32
        assert np.isnan(result.attrs['nodata'])
        # Geoid undulation in NY is ~-33 m, so 100 + N is ~67. Sanity
        # check that the shift actually happened and stays in a sane band.
        finite = result.values[np.isfinite(result.values)]
        assert finite.size > 0
        assert np.all(finite < 100.0)  # ellipsoidal height < orthometric here
        assert np.all(finite > 50.0)

    def test_float32_no_vertical_promotion(self):
        """float32 input is not further promoted by the vertical shift.

        ``reproject()`` itself upcasts float32 to float64 in its resample
        path; the vertical-shift code must not stack a second promotion
        on top of that. Compare the dtype of the shifted output to the
        dtype of a plain reproject of the same raster.
        """
        from xrspatial.reproject import reproject
        y = np.linspace(41.1, 40.3, 8)
        x = np.linspace(-74.4, -73.6, 8)
        data = np.full((8, 8), 100.0, dtype=np.float32)
        raster = xr.DataArray(
            data, dims=['y', 'x'], coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': np.float32('nan')},
        )
        baseline = reproject(raster, 'EPSG:4326')
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96', target_vertical_crs='ellipsoidal',
        )
        assert result.dtype == baseline.dtype, (
            f"vertical shift changed dtype from {baseline.dtype} to {result.dtype}"
        )

    def test_float64_stays_float64(self):
        """float64 input keeps its precision through the shift."""
        from xrspatial.reproject import reproject
        y = np.linspace(41.1, 40.3, 8)
        x = np.linspace(-74.4, -73.6, 8)
        data = np.full((8, 8), 100.0, dtype=np.float64)
        raster = xr.DataArray(
            data, dims=['y', 'x'], coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96', target_vertical_crs='ellipsoidal',
        )
        assert result.dtype == np.float64

    def test_int16_nodata_becomes_nan(self):
        """Integer nodata pixels must map to NaN in the promoted output."""
        from xrspatial.reproject import reproject
        # Put a nodata pixel right in the middle of an otherwise valid raster.
        y = np.linspace(41.1, 40.3, 8)
        x = np.linspace(-74.4, -73.6, 8)
        data = np.full((8, 8), 100, dtype=np.int16)
        data[4, 4] = -32768  # mark one cell as nodata
        raster = xr.DataArray(
            data, dims=['y', 'x'], coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': -32768},
        )
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96', target_vertical_crs='ellipsoidal',
        )
        assert result.dtype == np.float32
        # The nodata cell must have propagated as NaN, not as -32768 + N.
        assert np.isnan(result.values[4, 4]), (
            f"expected NaN at nodata cell, got {result.values[4, 4]}"
        )
        # Surrounding cells should still carry a finite shifted value.
        finite = np.isfinite(result.values)
        assert finite.sum() >= 50  # most cells finite

    def test_int16_fillvalue_attr_promoted(self):
        """attrs['_FillValue'] and attrs['nodatavals'] follow the dtype.

        ``reproject()`` carries both keys forward when the source had
        them. After dtype promotion the values used to keep the original
        integer sentinel, which contradicts the now-float array contents.
        """
        from xrspatial.reproject import reproject
        y = np.linspace(41.1, 40.3, 8)
        x = np.linspace(-74.4, -73.6, 8)
        data = np.full((8, 8), 100, dtype=np.int16)
        raster = xr.DataArray(
            data, dims=['y', 'x'], coords={'y': y, 'x': x},
            attrs={
                'crs': 'EPSG:4326',
                'nodata': -32768,
                '_FillValue': -32768,
                'nodatavals': (-32768,),
            },
        )
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96', target_vertical_crs='ellipsoidal',
        )
        assert np.isnan(result.attrs['_FillValue'])
        assert np.isnan(result.attrs['nodatavals'][0])

    def test_int16_dask_promotes_to_float32(self):
        """Dask-backed int16 input must also promote correctly."""
        import dask.array as da
        from xrspatial.reproject import reproject
        host = np.full((48, 48), 100, dtype=np.int16)
        y = np.linspace(41.1, 40.3, 48)
        x = np.linspace(-74.4, -73.6, 48)
        raster = xr.DataArray(
            da.from_array(host, chunks=(16, 16)), dims=['y', 'x'],
            coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': -32768},
        )
        result = reproject(
            raster, 'EPSG:4326',
            source_vertical_crs='EGM96', target_vertical_crs='ellipsoidal',
        )
        # The dask graph must advertise float32 so downstream consumers
        # don't get a dtype lie.
        assert result.dtype == np.float32
        assert np.isnan(result.attrs['nodata'])
        vals = result.values
        finite = vals[np.isfinite(vals)]
        assert finite.size > 0
        # NY undulation is ~-33 m, so 100 + N should sit around 67 m.
        assert np.all(finite < 100.0)
        assert np.all(finite > 50.0)


class TestMetadataPreservation:
    """reproject() and merge() must carry input attrs forward."""

    @staticmethod
    def _raster_with_attrs(extra_attrs=None, h=8, w=8,
                           crs='EPSG:4326',
                           x_range=(-1, 1), y_range=(-1, 1),
                           name='dem'):
        data = np.ones((h, w), dtype=np.float64)
        attrs = {'crs': crs, 'nodata': np.nan}
        if extra_attrs:
            attrs.update(extra_attrs)
        y = np.linspace(y_range[1], y_range[0], h)
        x = np.linspace(x_range[0], x_range[1], w)
        return xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y, 'x': x},
            name=name, attrs=attrs,
        )

    # reproject() ----------------------------------------------------------

    def test_reproject_preserves_units_attr(self):
        from xrspatial.reproject import reproject
        raster = self._raster_with_attrs({'units': 'meters'})
        result = reproject(raster, 'EPSG:4326', resolution=0.25)
        assert result.attrs.get('units') == 'meters'

    def test_reproject_preserves_scale_offset(self):
        from xrspatial.reproject import reproject
        raster = self._raster_with_attrs(
            {'scale_factor': 0.1, 'add_offset': 10.0}
        )
        result = reproject(raster, 'EPSG:4326', resolution=0.25)
        assert result.attrs.get('scale_factor') == 0.1
        assert result.attrs.get('add_offset') == 10.0

    def test_reproject_preserves_long_name(self):
        from xrspatial.reproject import reproject
        raster = self._raster_with_attrs({'long_name': 'elevation'})
        result = reproject(raster, 'EPSG:4326', resolution=0.25)
        assert result.attrs.get('long_name') == 'elevation'

    def test_reproject_replaces_stale_transform(self):
        from xrspatial.reproject import reproject
        stale = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        raster = self._raster_with_attrs({'transform': stale})
        result = reproject(raster, 'EPSG:3857')
        assert 'transform' in result.attrs
        assert tuple(result.attrs['transform']) != stale

    def test_reproject_replaces_stale_res(self):
        from xrspatial.reproject import reproject
        raster = self._raster_with_attrs({'res': (1.0, 1.0)})
        result = reproject(raster, 'EPSG:3857')
        assert 'res' in result.attrs
        assert tuple(result.attrs['res']) != (1.0, 1.0)

    def test_reproject_overrides_crs(self):
        from xrspatial.reproject import reproject
        raster = self._raster_with_attrs(crs='EPSG:4326')
        result = reproject(raster, 'EPSG:3857')
        # Output crs is the new target CRS WKT, not the input EPSG:4326
        assert 'crs' in result.attrs
        out_crs = result.attrs['crs']
        assert out_crs != 'EPSG:4326'
        # WKT for 3857 mentions Mercator / pseudo-mercator
        assert 'Mercator' in out_crs or '3857' in out_crs

    def test_reproject_drops_stale_crs_wkt(self):
        from xrspatial.reproject import reproject
        raster = self._raster_with_attrs({'crs_wkt': 'OLD_DUPLICATE_WKT'})
        result = reproject(raster, 'EPSG:3857')
        assert 'crs_wkt' not in result.attrs

    # merge() --------------------------------------------------------------

    def test_merge_preserves_first_raster_attrs(self):
        from xrspatial.reproject import merge
        a = self._raster_with_attrs(
            {'units': 'm', 'long_name': 'elev'},
            x_range=(-5, 0), y_range=(-5, 5), name='dem_a',
        )
        b = self._raster_with_attrs(
            {'units': 'feet'},
            x_range=(0, 5), y_range=(-5, 5), name='dem_b',
        )
        result = merge([a, b], resolution=1.0)
        assert result.attrs.get('units') == 'm'
        assert result.attrs.get('long_name') == 'elev'

    def test_merge_replaces_stale_transform(self):
        from xrspatial.reproject import merge
        stale = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        a = self._raster_with_attrs(
            {'transform': stale},
            x_range=(-5, 0), y_range=(-5, 5),
        )
        b = self._raster_with_attrs(
            x_range=(0, 5), y_range=(-5, 5),
        )
        result = merge([a, b], resolution=1.0)
        assert 'transform' in result.attrs
        assert tuple(result.attrs['transform']) != stale

    # Fresh transform/res emission ----------------------------------------

    def test_reproject_emits_fresh_transform(self):
        from xrspatial.reproject import reproject
        stale = (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        raster = self._raster_with_attrs({'transform': stale})
        result = reproject(raster, 'EPSG:3857', resolution=50000.0)
        t = result.attrs['transform']
        assert len(t) == 6
        # Output is in EPSG:3857 so transform values cannot match the stale
        # geographic-degree input.
        assert tuple(t) != stale
        # transform[0] is res_x, transform[4] is -res_y, transform[2] is
        # left edge, transform[5] is top edge.
        res_x, res_y = result.attrs['res']
        assert t[0] == res_x
        assert t[4] == -res_y
        # Top edge: y coord of first row plus half a pixel.
        y0 = float(result.coords['y'].values[0])
        assert t[5] == pytest.approx(y0 + res_y / 2)
        # Left edge: x coord of first col minus half a pixel.
        x0 = float(result.coords['x'].values[0])
        assert t[2] == pytest.approx(x0 - res_x / 2)

    def test_reproject_emits_fresh_res(self):
        from xrspatial.reproject import reproject
        raster = self._raster_with_attrs({'res': (1.0, 1.0)})
        # Use an explicit resolution very different from input.
        result = reproject(raster, 'EPSG:4326', resolution=0.25)
        assert 'res' in result.attrs
        res_x, res_y = result.attrs['res']
        # Pixel size derived from output coords must match.
        x = result.coords['x'].values
        y = result.coords['y'].values
        actual_res_x = float(abs(x[1] - x[0]))
        actual_res_y = float(abs(y[1] - y[0]))
        assert res_x == pytest.approx(actual_res_x)
        assert res_y == pytest.approx(actual_res_y)

    def test_reproject_no_input_transform_still_emits_one(self):
        from xrspatial.reproject import reproject
        raster = self._raster_with_attrs()
        assert 'transform' not in raster.attrs
        result = reproject(raster, 'EPSG:4326', resolution=0.25)
        assert 'transform' in result.attrs
        assert 'res' in result.attrs
        assert len(result.attrs['transform']) == 6

    def test_merge_emits_fresh_transform_and_res(self):
        from xrspatial.reproject import merge
        a = self._raster_with_attrs(
            x_range=(-5, 0), y_range=(-5, 5),
        )
        b = self._raster_with_attrs(
            x_range=(0, 5), y_range=(-5, 5),
        )
        result = merge([a, b], resolution=1.0)
        assert 'transform' in result.attrs
        assert 'res' in result.attrs
        t = result.attrs['transform']
        assert len(t) == 6
        res_x, res_y = result.attrs['res']
        assert t[0] == res_x
        assert t[4] == -res_y
        y0 = float(result.coords['y'].values[0])
        x0 = float(result.coords['x'].values[0])
        assert t[5] == pytest.approx(y0 + res_y / 2)
        assert t[2] == pytest.approx(x0 - res_x / 2)

    def test_merge_finds_spatial_dims_with_lat_lon(self):
        from xrspatial.reproject import merge
        a_data = np.ones((8, 8), dtype=np.float64)
        b_data = np.ones((8, 8), dtype=np.float64) * 2
        attrs = {'crs': 'EPSG:4326', 'nodata': np.nan}
        a = xr.DataArray(
            a_data, dims=['lat', 'lon'],
            coords={
                'lat': np.linspace(5, -5, 8),
                'lon': np.linspace(-5, 0, 8),
            },
            name='a', attrs=attrs,
        )
        b = xr.DataArray(
            b_data, dims=['lat', 'lon'],
            coords={
                'lat': np.linspace(5, -5, 8),
                'lon': np.linspace(0, 5, 8),
            },
            name='b', attrs=attrs,
        )
        result = merge([a, b], resolution=1.0)
        assert result.dims == ('lat', 'lon')
        assert 'lat' in result.coords
        assert 'lon' in result.coords

    # _FillValue propagation -----------------------------------------------

    def test_reproject_propagates_fill_value(self):
        from xrspatial.reproject import reproject
        # Build a raster with _FillValue set and no nodata key.
        data = np.ones((8, 8), dtype=np.float64)
        attrs = {'crs': 'EPSG:4326', '_FillValue': -9999}
        raster = xr.DataArray(
            data, dims=['y', 'x'],
            coords={
                'y': np.linspace(1, -1, 8),
                'x': np.linspace(-1, 1, 8),
            },
            attrs=attrs,
        )
        result = reproject(raster, 'EPSG:4326', resolution=0.25)
        assert '_FillValue' in result.attrs
        assert 'nodata' in result.attrs
        assert result.attrs['_FillValue'] == result.attrs['nodata']
        assert result.attrs['_FillValue'] == -9999

    def test_reproject_omits_fill_value_when_input_omits(self):
        from xrspatial.reproject import reproject
        raster = self._raster_with_attrs()
        assert '_FillValue' not in raster.attrs
        result = reproject(raster, 'EPSG:4326', resolution=0.25)
        assert '_FillValue' not in result.attrs
        assert 'nodata' in result.attrs

    def test_merge_propagates_fill_value(self):
        from xrspatial.reproject import merge
        a_data = np.ones((8, 8), dtype=np.float64)
        b_data = np.ones((8, 8), dtype=np.float64) * 2
        attrs_a = {'crs': 'EPSG:4326', '_FillValue': -9999}
        attrs_b = {'crs': 'EPSG:4326', '_FillValue': -9999}
        a = xr.DataArray(
            a_data, dims=['y', 'x'],
            coords={
                'y': np.linspace(5, -5, 8),
                'x': np.linspace(-5, 0, 8),
            },
            name='a', attrs=attrs_a,
        )
        b = xr.DataArray(
            b_data, dims=['y', 'x'],
            coords={
                'y': np.linspace(5, -5, 8),
                'x': np.linspace(0, 5, 8),
            },
            name='b', attrs=attrs_b,
        )
        result = merge([a, b], resolution=1.0)
        assert '_FillValue' in result.attrs
        assert 'nodata' in result.attrs
        assert result.attrs['_FillValue'] == result.attrs['nodata']
        assert result.attrs['_FillValue'] == -9999

    def test_merge_name_falls_back_to_first_raster(self):
        from xrspatial.reproject import merge
        a = self._raster_with_attrs(
            x_range=(-5, 0), y_range=(-5, 5), name='dem_a',
        )
        b = self._raster_with_attrs(
            x_range=(0, 5), y_range=(-5, 5), name='dem_b',
        )
        result = merge([a, b], resolution=1.0)
        assert result.name == 'dem_a'

    # nodatavals (rasterio convention) -- #1573 ----------------------------

    def test_reproject_detects_nodata_from_nodatavals(self):
        from xrspatial.reproject import reproject
        # Input has nodatavals but no nodata / _FillValue. Without rioxarray
        # in the lookup chain, reproject must still pick up the sentinel.
        raster = xr.DataArray(
            np.full((8, 8), -9999.0, dtype=np.float64),
            dims=['y', 'x'],
            coords={'y': np.linspace(5, -5, 8), 'x': np.linspace(-5, 5, 8)},
            attrs={'crs': 'EPSG:4326', 'nodatavals': (-9999,)},
        )
        # Remove `nodata` key so the lookup must walk to nodatavals.
        assert 'nodata' not in raster.attrs
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert result.attrs.get('nodata') == -9999.0

    def test_reproject_refreshes_nodatavals_to_resolved_nodata(self):
        from xrspatial.reproject import reproject
        raster = xr.DataArray(
            np.full((8, 8), -9999.0, dtype=np.float64),
            dims=['y', 'x'],
            coords={'y': np.linspace(5, -5, 8), 'x': np.linspace(-5, 5, 8)},
            attrs={'crs': 'EPSG:4326', 'nodatavals': (-9999,)},
        )
        result = reproject(raster, 'EPSG:4326', resolution=1.0,
                           nodata=np.nan)
        # nodata key reflects user-provided sentinel
        assert np.isnan(result.attrs['nodata'])
        # nodatavals tuple is refreshed to match (no stale -9999)
        nv = result.attrs['nodatavals']
        assert isinstance(nv, tuple) and len(nv) == 1
        assert np.isnan(nv[0])

    def test_reproject_omits_nodatavals_when_input_omits(self):
        from xrspatial.reproject import reproject
        raster = self._raster_with_attrs()
        assert 'nodatavals' not in raster.attrs
        result = reproject(raster, 'EPSG:4326', resolution=0.25)
        assert 'nodatavals' not in result.attrs

    def test_merge_propagates_nodatavals(self):
        from xrspatial.reproject import merge
        a = xr.DataArray(
            np.full((8, 8), -9999.0, dtype=np.float64),
            dims=['y', 'x'],
            coords={'y': np.linspace(5, -5, 8), 'x': np.linspace(-5, 0, 8)},
            name='a',
            attrs={'crs': 'EPSG:4326', 'nodatavals': (-9999,)},
        )
        b = xr.DataArray(
            np.full((8, 8), -9999.0, dtype=np.float64),
            dims=['y', 'x'],
            coords={'y': np.linspace(5, -5, 8), 'x': np.linspace(0, 5, 8)},
            name='b',
            attrs={'crs': 'EPSG:4326', 'nodatavals': (-9999,)},
        )
        result = merge([a, b], resolution=1.0, nodata=-9999)
        assert result.attrs['nodata'] == -9999.0
        assert result.attrs['nodatavals'] == (-9999.0,)


# ---------------------------------------------------------------------------
# geoid_height_raster -- metadata propagation (#1572)
# ---------------------------------------------------------------------------

class TestGeoidHeightRasterMetadata:
    """geoid_height_raster must preserve georef attrs and handle 3D inputs."""

    def test_geoid_height_raster_carries_input_attrs(self):
        from xrspatial.reproject import geoid_height_raster
        raster = xr.DataArray(
            np.zeros((4, 4)),
            dims=['y', 'x'],
            coords={'y': [3.0, 2.0, 1.0, 0.0], 'x': [0.0, 1.0, 2.0, 3.0]},
            attrs={
                'crs': 'EPSG:4326',
                'res': (1.0, 1.0),
                'transform': (1.0, 0.0, -0.5, 0.0, -1.0, 3.5),
                '_FillValue': -9999.0,
                'long_name': 'orthometric_height',
                'scale_factor': 0.001,
            },
        )
        result = geoid_height_raster(raster)
        # Input georef attrs must survive.
        assert result.attrs['crs'] == 'EPSG:4326'
        assert result.attrs['res'] == (1.0, 1.0)
        assert result.attrs['transform'] == (
            1.0, 0.0, -0.5, 0.0, -1.0, 3.5,
        )
        assert result.attrs['_FillValue'] == -9999.0
        assert result.attrs['long_name'] == 'orthometric_height'
        assert result.attrs['scale_factor'] == 0.001
        # The function's own attrs are layered on top.
        assert result.attrs['units'] == 'metres'
        assert result.attrs['model'] == 'EGM96'

    def test_geoid_height_raster_3d_reduces_to_2d(self):
        from xrspatial.reproject import geoid_height_raster
        # 3D input with band as the trailing axis.
        raster = xr.DataArray(
            np.zeros((4, 4, 3)),
            dims=['y', 'x', 'band'],
            coords={
                'y': [3.0, 2.0, 1.0, 0.0],
                'x': [0.0, 1.0, 2.0, 3.0],
                'band': [1, 2, 3],
            },
            attrs={'crs': 'EPSG:4326'},
        )
        result = geoid_height_raster(raster)
        # Output is 2D on the y/x grid -- band is dropped because the
        # geoid is purely a function of position.
        assert result.dims == ('y', 'x')
        assert result.shape == (4, 4)
        # Coordinate values come from the spatial dims of the input,
        # not raster.dims[-2:] which would be ('x', 'band').
        np.testing.assert_array_equal(
            result.coords['y'].values, [3.0, 2.0, 1.0, 0.0],
        )
        np.testing.assert_array_equal(
            result.coords['x'].values, [0.0, 1.0, 2.0, 3.0],
        )
        assert result.attrs['crs'] == 'EPSG:4326'

    def test_geoid_height_raster_2d_unchanged_shape(self):
        from xrspatial.reproject import geoid_height_raster
        raster = xr.DataArray(
            np.zeros((4, 4)),
            dims=['y', 'x'],
            coords={'y': [3.0, 2.0, 1.0, 0.0], 'x': [0.0, 1.0, 2.0, 3.0]},
            attrs={'crs': 'EPSG:4326'},
        )
        result = geoid_height_raster(raster)
        assert result.dims == ('y', 'x')
        assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# Backend parity: dask dtype + same-CRS dask merge + cupy
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_DASK, reason="dask required")
class TestDaskDtypeParity:
    """Dask reproject should preserve source integer dtype (matches numpy)."""

    def test_dask_reproject_int8_preserves_dtype(self):
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.int8).reshape(8, 8)
        coords = {'y': np.linspace(5, -5, 8), 'x': np.linspace(-5, 5, 8)}
        attrs = {'crs': 'EPSG:4326', 'nodata': -1}
        raster = xr.DataArray(
            da.from_array(data, chunks=(4, 4)),
            dims=['y', 'x'], coords=coords, attrs=attrs,
        )
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        # Lazy meta dtype should match
        assert result.data.dtype == np.int8
        # Computed dtype should also match
        assert result.compute().dtype == np.int8

    def test_dask_reproject_uint16_preserves_dtype(self):
        from xrspatial.reproject import reproject
        data = (np.arange(64, dtype=np.uint16) * 100).reshape(8, 8)
        coords = {'y': np.linspace(5, -5, 8), 'x': np.linspace(-5, 5, 8)}
        attrs = {'crs': 'EPSG:4326', 'nodata': 0}
        raster = xr.DataArray(
            da.from_array(data, chunks=(4, 4)),
            dims=['y', 'x'], coords=coords, attrs=attrs,
        )
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert result.data.dtype == np.uint16
        assert result.compute().dtype == np.uint16

    def test_dask_reproject_float32_stays_float64(self):
        """Float input still upcasts to float64 (existing behaviour guard)."""
        from xrspatial.reproject import reproject
        data = np.random.RandomState(0).rand(8, 8).astype(np.float32)
        coords = {'y': np.linspace(5, -5, 8), 'x': np.linspace(-5, 5, 8)}
        attrs = {'crs': 'EPSG:4326', 'nodata': np.nan}
        raster = xr.DataArray(
            da.from_array(data, chunks=(4, 4)),
            dims=['y', 'x'], coords=coords, attrs=attrs,
        )
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert result.data.dtype == np.float64
        assert result.compute().dtype == np.float64


class TestStreamingDtypeParity:
    """The streaming fallback must match the other backends' dtype rule (#3093).

    ``_reproject_streaming`` is only reachable through ``reproject()`` when
    dask is not installed and the in-memory source exceeds 512 MB, so these
    tests call the helper directly with grid parameters built the same way
    ``reproject()`` builds them. Before the fix it allocated the assembled
    output as float64 regardless of the source dtype (the other four
    backends round-trip integer dtypes, see #2505) and allocated it 2-D,
    which crashed on 3-D ``(y, x, band)`` sources.
    """

    def _streaming_args(self, raster):
        from xrspatial.reproject import (
            _is_y_descending,
            _source_bounds,
        )
        from xrspatial.reproject._crs_utils import _detect_nodata, _resolve_crs
        from xrspatial.reproject._grid import _compute_output_grid

        src_crs = _resolve_crs('EPSG:4326')
        tgt_crs = _resolve_crs('EPSG:3857')
        src_bounds = _source_bounds(raster)
        src_shape = raster.shape[:2]
        grid = _compute_output_grid(src_bounds, src_shape, src_crs, tgt_crs)
        nd = _detect_nodata(raster, None, dtype=raster.dtype)
        return (
            raster, src_bounds, src_shape, _is_y_descending(raster),
            src_crs.to_wkt(), tgt_crs.to_wkt(),
            grid['bounds'], grid['shape'],
            'nearest', nd, 16,
            8,          # tile_size: force multiple tiles
            1024 ** 3,  # max_memory_bytes
        )

    def _make_raster(self, data):
        coords = {
            'y': np.linspace(50, 45, data.shape[0]),
            'x': np.linspace(-5, 0, data.shape[1]),
        }
        dims = ['y', 'x'] if data.ndim == 2 else ['y', 'x', 'band']
        if data.ndim == 3:
            coords['band'] = np.arange(data.shape[2])
        return xr.DataArray(data, dims=dims, coords=coords,
                            attrs={'crs': 'EPSG:4326'})

    def test_streaming_int16_preserves_dtype(self):
        from xrspatial.reproject import _reproject_streaming
        data = (np.arange(32 * 32).reshape(32, 32) % 100).astype(np.int16)
        out = _reproject_streaming(*self._streaming_args(self._make_raster(data)))
        assert out.dtype == np.int16

    def test_streaming_uint8_preserves_dtype(self):
        from xrspatial.reproject import _reproject_streaming
        data = (np.arange(32 * 32).reshape(32, 32) % 200).astype(np.uint8)
        out = _reproject_streaming(*self._streaming_args(self._make_raster(data)))
        assert out.dtype == np.uint8

    def test_streaming_float64_stays_float64(self):
        from xrspatial.reproject import _reproject_streaming
        data = np.random.RandomState(0).rand(32, 32)
        out = _reproject_streaming(*self._streaming_args(self._make_raster(data)))
        assert out.dtype == np.float64

    def test_streaming_matches_inmemory_values(self):
        """Streaming and in-memory numpy paths agree on values and dtype."""
        from xrspatial.reproject import _reproject_streaming, reproject
        data = (np.arange(32 * 32).reshape(32, 32) % 100).astype(np.int16)
        raster = self._make_raster(data)
        out = _reproject_streaming(*self._streaming_args(raster))
        expected = reproject(raster, 'EPSG:3857', resampling='nearest')
        assert out.dtype == expected.dtype
        np.testing.assert_array_equal(out, expected.values)

    def test_streaming_3d_band_axis(self):
        """3-D (y, x, band) sources assemble instead of crashing (#3093)."""
        from xrspatial.reproject import _reproject_streaming
        base = (np.arange(32 * 32).reshape(32, 32) % 100).astype(np.uint8)
        data = np.dstack([base, base + 1, base + 2])
        out = _reproject_streaming(*self._streaming_args(self._make_raster(data)))
        assert out.ndim == 3
        assert out.shape[2] == 3
        assert out.dtype == np.uint8

    def test_streaming_distributed_branch_preserves_dtype(self):
        """The dask.bag distributed branch uses the same dtype rule (#3093).

        ``_reproject_streaming`` switches to the distributed branch when a
        ``dask.distributed`` client is active and there are more tiles than
        workers, so run it under an in-process LocalCluster and check both
        dtype and value parity with the local-branch result.
        """
        distributed = pytest.importorskip('distributed')
        from xrspatial.reproject import _reproject_streaming
        data = (np.arange(32 * 32).reshape(32, 32) % 100).astype(np.int16)
        args = self._streaming_args(self._make_raster(data))
        local_out = _reproject_streaming(*args)
        with distributed.LocalCluster(
            n_workers=1, processes=False, threads_per_worker=1,
            dashboard_address=None,
        ) as cluster, distributed.Client(cluster):
            dist_out = _reproject_streaming(*args)
        assert dist_out.dtype == np.int16
        np.testing.assert_array_equal(dist_out, local_out)


class TestParallelKernelThreadSafety:
    """Concurrent launches of the numba projection kernels must not abort.

    The kernels in _projections.py are ``parallel=True``, and numba's
    default 'workqueue' threading layer terminates the process (SIGABRT
    on macOS, see the #3093 CI failure) when two host threads enter a
    parallel region concurrently. The streaming tile pool and dask's
    threaded scheduler both launch these kernels from worker threads, so
    try_numba_transform / transform_points serialize launches behind a
    module lock. Run the hammer in a subprocess with the workqueue layer
    forced so a regression aborts the child, not the test session.
    """

    _SCRIPT = """
import threading
import numpy as np
from xrspatial.reproject._projections import transform_points, try_numba_transform
from xrspatial.reproject._crs_utils import _resolve_crs

src = _resolve_crs('EPSG:4326')
tgt = _resolve_crs('EPSG:3857')
bounds = (-561014.0, 5621521.0, -556014.0, 6453998.0)
xs = np.linspace(-5.0, 5.0, 1000)
ys = np.linspace(40.0, 50.0, 1000)
errs = []

def work():
    try:
        for _ in range(25):
            try_numba_transform(src, tgt, bounds, (128, 128))
            transform_points(src, tgt, xs, ys)
    except BaseException as e:  # noqa: BLE001 - report everything
        errs.append(e)

threads = [threading.Thread(target=work) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert not errs, errs
print('OK')
"""

    def test_concurrent_kernel_launches_survive_workqueue(self):
        import os
        import subprocess
        import sys
        env = dict(os.environ, NUMBA_THREADING_LAYER='workqueue')
        proc = subprocess.run(
            [sys.executable, '-c', self._SCRIPT],
            capture_output=True, text=True, timeout=600, env=env,
        )
        assert proc.returncode == 0, (
            f"subprocess exited {proc.returncode}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        assert 'OK' in proc.stdout
        assert 'not threadsafe' not in proc.stderr


@pytest.mark.skipif(not (HAS_DASK and HAS_CUPY),
                    reason="dask + cupy required")
class TestDaskCupyDtypeParity:
    """Dask+CuPy reproject should preserve source integer dtype (#2505).

    Mirrors :class:`TestDaskDtypeParity`. The previous behaviour of the
    eager fast path in ``_reproject_dask_cupy`` silently promoted
    integer inputs to float64 while the other three backends (numpy,
    cupy, dask+numpy) and the chunked dask+cupy fallback preserved the
    source dtype.
    """

    def _make_dask_cupy_raster(self, data, nodata):
        coords = {
            'y': np.linspace(5, -5, data.shape[0]),
            'x': np.linspace(-5, 5, data.shape[1]),
        }
        attrs = {'crs': 'EPSG:4326', 'nodata': nodata}
        chunks = (max(1, data.shape[0] // 2), max(1, data.shape[1] // 2))
        return xr.DataArray(
            da.from_array(cp.asarray(data), chunks=chunks),
            dims=['y', 'x'], coords=coords, attrs=attrs,
        )

    def test_dask_cupy_reproject_int8_preserves_dtype(self):
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.int8).reshape(8, 8)
        raster = self._make_dask_cupy_raster(data, nodata=-1)
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        # The fast path returns an eager cupy array, not a dask array,
        # so result.dtype and result.data.dtype are the same object.
        # Assert both for full symmetry with TestDaskDtypeParity.
        assert result.dtype == np.int8
        assert result.data.dtype == np.int8

    def test_dask_cupy_reproject_int16_preserves_dtype(self):
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.int16).reshape(8, 8)
        raster = self._make_dask_cupy_raster(data, nodata=-32768)
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert result.dtype == np.int16
        assert result.data.dtype == np.int16

    def test_dask_cupy_reproject_uint16_preserves_dtype(self):
        from xrspatial.reproject import reproject
        data = (np.arange(64, dtype=np.uint16) * 100).reshape(8, 8)
        raster = self._make_dask_cupy_raster(data, nodata=0)
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert result.dtype == np.uint16
        assert result.data.dtype == np.uint16

    def test_dask_cupy_reproject_uint8_preserves_dtype(self):
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.uint8).reshape(8, 8)
        raster = self._make_dask_cupy_raster(data, nodata=255)
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert result.dtype == np.uint8
        assert result.data.dtype == np.uint8

    def test_dask_cupy_reproject_float32_stays_float64(self):
        """Float input still upcasts to float64 -- matches the numpy /
        dask+numpy paths so the four-backend grid is consistent."""
        from xrspatial.reproject import reproject
        data = np.random.RandomState(0).rand(8, 8).astype(np.float32)
        raster = self._make_dask_cupy_raster(data, nodata=np.nan)
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert result.dtype == np.float64
        assert result.data.dtype == np.float64

    def test_dask_cupy_reproject_int16_matches_dask_numpy_dtype(self):
        """Cross-backend parity: dask+cupy and dask+numpy must agree on
        output dtype for the same integer input. This is the exact case
        that regressed before #2505 was fixed."""
        from xrspatial.reproject import reproject
        data = np.arange(64, dtype=np.int16).reshape(8, 8)
        coords = {'y': np.linspace(5, -5, 8), 'x': np.linspace(-5, 5, 8)}
        attrs = {'crs': 'EPSG:4326', 'nodata': -32768}
        dask_np = xr.DataArray(
            da.from_array(data, chunks=(4, 4)),
            dims=['y', 'x'], coords=coords, attrs=attrs,
        )
        dask_cp = xr.DataArray(
            da.from_array(cp.asarray(data), chunks=(4, 4)),
            dims=['y', 'x'], coords=coords, attrs=attrs,
        )
        r_np = reproject(dask_np, 'EPSG:4326', resolution=1.0)
        r_cp = reproject(dask_cp, 'EPSG:4326', resolution=1.0)
        assert r_np.dtype == r_cp.dtype == np.int16


class TestEmptyChunkDtype:
    """Empty / no-overlap chunks must keep the integer source dtype (#3096).

    The empty-chunk fills in the chunk workers and the footprint-skip
    path in ``_reproject_block_adapter`` were hardcoded to float64. With
    an integer source, one no-overlap chunk was enough to promote the
    whole computed dask array to float64 while the lazy array (and the
    eager backend) advertised the integer dtype.
    """

    # Output bounds in EPSG:3857 that are far larger than the projected
    # footprint of the source raster below, so corner chunks have no
    # source overlap and take the empty-chunk path.
    _WIDE_BOUNDS = (-2_000_000.0, 5_000_000.0, 2_000_000.0, 9_000_000.0)

    def _make_int16_data(self, n=200):
        rng = np.random.default_rng(3096)
        return (rng.random((n, n)) * 1000).astype(np.int16)

    def _coords(self, n=200):
        return {'y': np.linspace(52.0, 51.0, n), 'x': np.linspace(-2.0, -1.0, n)}

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_dask_int16_empty_chunks_keep_dtype(self):
        from xrspatial.reproject import reproject
        data = self._make_int16_data()
        raster = xr.DataArray(
            da.from_array(data, chunks=(64, 64)),
            dims=['y', 'x'], coords=self._coords(),
            attrs={'crs': 'EPSG:4326', 'nodata': -32768},
        )
        result = reproject(raster, 'EPSG:3857', bounds=self._WIDE_BOUNDS,
                           resolution=20000, chunk_size=64)
        assert result.dtype == np.int16
        computed = result.compute()
        assert computed.dtype == np.int16
        # The no-overlap corners must hold the integer sentinel.
        assert computed.values[0, 0] == -32768
        # Some chunk must contain real data, or this test exercises
        # nothing but empty fills.
        assert (computed.values != -32768).any()

    def test_numpy_int16_no_overlap_output_keeps_dtype(self):
        # Eager backend, output grid entirely outside the source
        # footprint: the single chunk takes the no-overlap early return.
        from xrspatial.reproject import reproject
        data = self._make_int16_data(32)
        raster = xr.DataArray(
            data, dims=['y', 'x'], coords=self._coords(32),
            attrs={'crs': 'EPSG:4326', 'nodata': -32768},
        )
        result = reproject(raster, 'EPSG:3857',
                           bounds=(5_000_000.0, 5_000_000.0,
                                   6_000_000.0, 6_000_000.0),
                           resolution=20000)
        assert result.dtype == np.int16
        assert (result.values == -32768).all()

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_dask_float_empty_chunks_stay_float64(self):
        # Float sources keep returning float64 empty chunks (NaN fill).
        from xrspatial.reproject import reproject
        data = np.random.RandomState(0).rand(200, 200)
        raster = xr.DataArray(
            da.from_array(data, chunks=(64, 64)),
            dims=['y', 'x'], coords=self._coords(),
            attrs={'crs': 'EPSG:4326'},
        )
        result = reproject(raster, 'EPSG:3857', bounds=self._WIDE_BOUNDS,
                           resolution=20000, chunk_size=64)
        assert result.dtype == np.float64
        computed = result.compute()
        assert computed.dtype == np.float64
        assert np.isnan(computed.values[0, 0])

    @pytest.mark.skipif(not (HAS_DASK and HAS_CUPY),
                        reason="dask + cupy required")
    def test_dask_cupy_int16_empty_chunks_keep_dtype(self):
        from xrspatial.reproject import reproject
        data = self._make_int16_data()
        raster = xr.DataArray(
            da.from_array(cp.asarray(data), chunks=(64, 64)),
            dims=['y', 'x'], coords=self._coords(),
            attrs={'crs': 'EPSG:4326', 'nodata': -32768},
        )
        result = reproject(raster, 'EPSG:3857', bounds=self._WIDE_BOUNDS,
                           resolution=20000, chunk_size=64)
        computed = result.compute()
        assert computed.dtype == np.int16


@pytest.mark.skipif(not HAS_DASK, reason="dask required")
class TestMergeDaskParity:
    """Dask merge should match the eager numpy merge."""

    def test_merge_dask_same_crs_matches_eager(self):
        """Same-CRS merge should be bit-equal between eager and dask paths.

        Source and output resolutions match (within 1%) so
        ``_place_same_crs`` activates in both paths -- direct pixel copy
        means the dask result must equal the eager result bit-for-bit.
        """
        from xrspatial.reproject import merge
        # 16 pixels with center-to-center spacing of exactly 1.0 -> bounds
        # extend half a pixel past coords, source resolution matches output.
        a_data = np.arange(256, dtype=np.float64).reshape(16, 16)
        b_data = (np.arange(256, dtype=np.float64) * 2).reshape(16, 16)
        a = _make_raster(a_data, x_range=(-7.5, 7.5), y_range=(-7.5, 7.5))
        b = _make_raster(b_data, x_range=(8.5, 23.5), y_range=(-7.5, 7.5))

        eager = merge([a, b], resolution=1.0).compute().values

        a_dask = a.copy()
        b_dask = b.copy()
        a_dask.data = da.from_array(a_data, chunks=(8, 8))
        b_dask.data = da.from_array(b_data, chunks=(8, 8))
        dasked = merge(
            [a_dask, b_dask], resolution=1.0, chunk_size=8,
        ).compute().values

        assert eager.shape == dasked.shape
        eager_nan = np.isnan(eager)
        dask_nan = np.isnan(dasked)
        np.testing.assert_array_equal(eager_nan, dask_nan)
        # Finite values must be bit-equal: same-CRS path is direct copy
        np.testing.assert_array_equal(eager[~eager_nan], dasked[~dask_nan])

    def test_merge_dask_different_crs_matches_eager(self):
        """Different-CRS merge should match within float tolerance.

        Uses the synchronous dask scheduler. Multi-CRS reprojection
        creates a fresh ``pyproj.Transformer`` per chunk, and PROJ's
        first-time CRS-database load is not safe under concurrent
        threaded workers on macOS (the test would SIGABRT mid-compute).
        Synchronous compute exercises the same dask graph without the
        threading dimension; CRS thread-safety is its own concern,
        outside the scope of this parity test.
        """
        import pyproj
        from xrspatial.reproject import merge
        # Pre-warm the PROJ database in this thread so that any first-init
        # work happens here, not concurrently inside dask compute.
        pyproj.CRS.from_epsg(4326)
        pyproj.CRS.from_epsg(3857)

        a_data = np.arange(256, dtype=np.float64).reshape(16, 16)
        b_data = (np.arange(256, dtype=np.float64) + 100.0).reshape(16, 16)
        # One in WGS84, one in Web Mercator (forces reprojection)
        a = _make_raster(a_data, crs='EPSG:4326',
                         x_range=(-10, 0), y_range=(-5, 5))
        # Build a Web-Mercator tile that overlaps the target
        b = _make_raster(b_data, crs='EPSG:3857',
                         x_range=(0, 1_000_000), y_range=(-500_000, 500_000))

        eager = merge(
            [a, b], target_crs='EPSG:4326', resolution=1.0,
        ).compute(scheduler='synchronous').values

        a_dask = a.copy()
        b_dask = b.copy()
        a_dask.data = da.from_array(a_data, chunks=(8, 8))
        b_dask.data = da.from_array(b_data, chunks=(8, 8))
        dasked = merge(
            [a_dask, b_dask], target_crs='EPSG:4326',
            resolution=1.0, chunk_size=8,
        ).compute(scheduler='synchronous').values

        assert eager.shape == dasked.shape
        np.testing.assert_array_equal(np.isnan(eager), np.isnan(dasked))
        finite = np.isfinite(eager)
        if finite.any():
            np.testing.assert_allclose(
                eager[finite], dasked[finite], rtol=1e-10, atol=1e-10,
            )

    def test_merge_dask_same_crs_bounded_materialization(self, monkeypatch):
        """Same-CRS dask merge must not materialize full source per chunk.

        Regression test for issue #1571: ``_merge_block_adapter`` used to
        call ``.compute()`` on the full dask source array for every
        output chunk, amplifying driver-side data flow by O(N_chunks).
        The fix slices the source window first and computes only that
        slice. Total pixels materialized should be bounded by the total
        source size (within a small constant for the placement overlap).
        """
        from xrspatial.reproject import merge
        orig_compute = da.Array.compute
        records = []

        def trace(self, *a, **kw):
            records.append(int(np.prod(self.shape)))
            return orig_compute(self, *a, **kw)

        # Two 256x256 sources, 32x32 output chunks -> 8x8x2 = 128 chunks
        t1 = xr.DataArray(
            da.from_array(
                np.arange(256 * 256, dtype=np.float64).reshape(256, 256),
                chunks=(64, 64),
            ),
            dims=['y', 'x'],
            coords={'y': np.linspace(40, 35, 256),
                    'x': np.linspace(-10, -5, 256)},
            attrs={'crs': 'EPSG:4326'},
        )
        t2 = xr.DataArray(
            da.from_array(
                np.ones((256, 256), dtype=np.float64) * 2.0,
                chunks=(64, 64),
            ),
            dims=['y', 'x'],
            coords={'y': np.linspace(40, 35, 256),
                    'x': np.linspace(-5, 0, 256)},
            attrs={'crs': 'EPSG:4326'},
        )

        monkeypatch.setattr(da.Array, 'compute', trace)
        merge([t1, t2], strategy='first', chunk_size=32).compute()

        total_src_pixels = 2 * 256 * 256
        # Pre-fix: ~68x amplification. Post-fix: ~1x.
        # Allow a 3x ceiling to leave room for unrelated dask compute
        # calls in the pipeline (output assembly etc.).
        materialized = sum(records)
        assert materialized < 3 * total_src_pixels, (
            f"same-CRS dask merge materialized {materialized} pixels "
            f"for {total_src_pixels} total source pixels "
            f"(ratio {materialized / total_src_pixels:.1f}x); "
            f"this indicates full-source materialization per chunk."
        )


@pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
class TestCupyReprojectParity:
    """End-to-end cupy backend parity checks."""

    def test_cupy_reproject_matches_numpy(self):
        from xrspatial.reproject import reproject
        data = np.random.RandomState(7).rand(32, 32).astype(np.float64)
        coords = {'y': np.linspace(55, 45, 32), 'x': np.linspace(-5, 5, 32)}
        attrs = {'crs': 'EPSG:4326', 'nodata': np.nan}

        np_raster = xr.DataArray(data, dims=['y', 'x'],
                                 coords=coords, attrs=attrs)
        cp_raster = xr.DataArray(cp.asarray(data), dims=['y', 'x'],
                                 coords=coords, attrs=attrs)
        np_result = reproject(np_raster, 'EPSG:3857').values
        cp_result_arr = reproject(cp_raster, 'EPSG:3857').data
        # cupy DataArray: pull through .get() to avoid implicit numpy convert
        if hasattr(cp_result_arr, 'get'):
            cp_vals = cp_result_arr.get()
        else:
            cp_vals = np.asarray(cp_result_arr)

        assert np_result.shape == cp_vals.shape
        np.testing.assert_array_equal(
            np.isnan(np_result), np.isnan(cp_vals),
        )
        finite = np.isfinite(np_result)
        if finite.any():
            np.testing.assert_allclose(
                np_result[finite], cp_vals[finite], rtol=1e-5, atol=1e-5,
            )

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_dask_cupy_reproject_matches_numpy(self):
        from xrspatial.reproject import reproject
        data = np.random.RandomState(11).rand(32, 32).astype(np.float64)
        coords = {'y': np.linspace(55, 45, 32), 'x': np.linspace(-5, 5, 32)}
        attrs = {'crs': 'EPSG:4326', 'nodata': np.nan}

        np_raster = xr.DataArray(data, dims=['y', 'x'],
                                 coords=coords, attrs=attrs)
        dc_raster = xr.DataArray(
            da.from_array(cp.asarray(data), chunks=(16, 16)),
            dims=['y', 'x'], coords=coords, attrs=attrs,
        )
        np_result = reproject(np_raster, 'EPSG:3857').values
        dc_arr = reproject(dc_raster, 'EPSG:3857').data
        if hasattr(dc_arr, 'compute'):
            dc_arr = dc_arr.compute()
        if hasattr(dc_arr, 'get'):
            dc_vals = dc_arr.get()
        else:
            dc_vals = np.asarray(dc_arr)

        assert np_result.shape == dc_vals.shape
        np.testing.assert_array_equal(
            np.isnan(np_result), np.isnan(dc_vals),
        )
        finite = np.isfinite(np_result)
        if finite.any():
            np.testing.assert_allclose(
                np_result[finite], dc_vals[finite], rtol=1e-5, atol=1e-5,
            )

    def test_cupy_reproject_with_nan_chunks(self):
        """Regression: target chunks projecting outside the source must
        return all-nodata, exercising the batched min/max early-return."""
        from xrspatial.reproject import reproject
        data = np.random.RandomState(3).rand(16, 16).astype(np.float64)
        # Source covers a small region near the prime meridian / equator.
        coords = {'y': np.linspace(2, -2, 16), 'x': np.linspace(-2, 2, 16)}
        attrs = {'crs': 'EPSG:4326', 'nodata': np.nan}
        cp_raster = xr.DataArray(cp.asarray(data), dims=['y', 'x'],
                                 coords=coords, attrs=attrs)

        # Reproject to a target far outside the source. Coordinates that fall
        # outside the source produce NaN row/col pixels, so the batched
        # nanmin/nanmax should be NaN and trigger the all-nodata early return.
        target_bounds = (5_000_000, 5_000_000, 5_100_000, 5_100_000)
        out = reproject(cp_raster, 'EPSG:3857', bounds=target_bounds,
                        width=8, height=8)
        out_vals = out.data.get() if hasattr(out.data, 'get') else np.asarray(out.data)
        assert out_vals.shape == (8, 8)
        # Out-of-bounds output: all entries must be nodata (NaN here).
        assert np.all(np.isnan(out_vals))

        # Same target as the source exercises the in-bounds branch and must
        # return finite values from the same batched-reduction code path.
        in_bounds = reproject(cp_raster, 'EPSG:4326',
                              bounds=(-1.5, -1.5, 1.5, 1.5),
                              width=8, height=8)
        in_vals = (in_bounds.data.get() if hasattr(in_bounds.data, 'get')
                   else np.asarray(in_bounds.data))
        assert np.isfinite(in_vals).any()


class TestDegenerateShapeReproject:
    """Single-row, single-column, and constant-value rasters (#2618).

    A strip raster has one spatial axis of size 1, which hits the
    ``size < 2`` early-return in ``_validate_regular_axis`` and runs the
    resampling kernel on a degenerate axis. A constant-value raster has
    zero gradient, exercising the all-equal interpolation path. All four
    backends return correct output today; these tests lock that in.
    """

    @staticmethod
    def _strip(values, n, axis, use_dask=False, use_cupy=False, chunks=None):
        """Build a 1xN (axis='row') or Nx1 (axis='col') strip raster.

        The degenerate axis spans a single coordinate; the long axis is a
        regular ramp so the projection has a real extent to work with.
        """
        arr = np.asarray(values, dtype=np.float64)
        if axis == 'row':
            data = arr.reshape(1, n)
            y = np.array([0.0])
            x = np.linspace(-5, 5, n)
        else:
            data = arr.reshape(n, 1)
            y = np.linspace(5, -5, n)
            x = np.array([0.0])
        if use_cupy:
            data = cp.asarray(data)
        if use_dask:
            block = chunks if chunks is not None else data.shape
            data = da.from_array(data, chunks=block)
        return xr.DataArray(
            data, dims=['y', 'x'], coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )

    @staticmethod
    def _to_host(result):
        arr = result.data
        if hasattr(arr, 'compute'):
            arr = arr.compute()
        if hasattr(arr, 'get'):
            arr = arr.get()
        return np.asarray(arr)

    # -- single-row (1xN) strip --------------------------------------------

    def test_single_row_strip_numpy(self):
        from xrspatial.reproject import reproject
        raster = self._strip(np.arange(8), 8, 'row')
        out = reproject(raster, 'EPSG:3857')
        vals = self._to_host(out)
        assert out.ndim == 2
        assert np.isfinite(vals).any()

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_single_row_strip_dask_matches_numpy(self):
        from xrspatial.reproject import reproject
        np_raster = self._strip(np.arange(8), 8, 'row')
        da_raster = self._strip(np.arange(8), 8, 'row',
                                use_dask=True, chunks=(1, 4))
        np_vals = self._to_host(reproject(np_raster, 'EPSG:3857'))
        da_vals = self._to_host(reproject(da_raster, 'EPSG:3857'))
        assert np_vals.shape == da_vals.shape
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(da_vals))
        finite = np.isfinite(np_vals)
        if finite.any():
            np.testing.assert_allclose(
                np_vals[finite], da_vals[finite], rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
    def test_single_row_strip_cupy_matches_numpy(self):
        from xrspatial.reproject import reproject
        np_raster = self._strip(np.arange(8), 8, 'row')
        cp_raster = self._strip(np.arange(8), 8, 'row', use_cupy=True)
        np_vals = self._to_host(reproject(np_raster, 'EPSG:3857'))
        cp_vals = self._to_host(reproject(cp_raster, 'EPSG:3857'))
        assert np_vals.shape == cp_vals.shape
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(cp_vals))
        finite = np.isfinite(np_vals)
        if finite.any():
            np.testing.assert_allclose(
                np_vals[finite], cp_vals[finite], rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not (HAS_DASK and HAS_CUPY),
                        reason="dask and cupy required")
    def test_single_row_strip_dask_cupy_matches_numpy(self):
        from xrspatial.reproject import reproject
        np_raster = self._strip(np.arange(8), 8, 'row')
        dc_raster = self._strip(np.arange(8), 8, 'row',
                                use_dask=True, use_cupy=True, chunks=(1, 4))
        np_vals = self._to_host(reproject(np_raster, 'EPSG:3857'))
        dc_vals = self._to_host(reproject(dc_raster, 'EPSG:3857'))
        assert np_vals.shape == dc_vals.shape
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(dc_vals))
        finite = np.isfinite(np_vals)
        if finite.any():
            np.testing.assert_allclose(
                np_vals[finite], dc_vals[finite], rtol=1e-5, atol=1e-5)

    # -- single-column (Nx1) strip -----------------------------------------

    def test_single_col_strip_numpy(self):
        from xrspatial.reproject import reproject
        raster = self._strip(np.arange(8), 8, 'col')
        out = reproject(raster, 'EPSG:3857')
        vals = self._to_host(out)
        assert out.ndim == 2
        assert np.isfinite(vals).any()

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_single_col_strip_dask_matches_numpy(self):
        from xrspatial.reproject import reproject
        np_raster = self._strip(np.arange(8), 8, 'col')
        da_raster = self._strip(np.arange(8), 8, 'col',
                                use_dask=True, chunks=(4, 1))
        np_vals = self._to_host(reproject(np_raster, 'EPSG:3857'))
        da_vals = self._to_host(reproject(da_raster, 'EPSG:3857'))
        assert np_vals.shape == da_vals.shape
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(da_vals))
        finite = np.isfinite(np_vals)
        if finite.any():
            np.testing.assert_allclose(
                np_vals[finite], da_vals[finite], rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
    def test_single_col_strip_cupy_matches_numpy(self):
        from xrspatial.reproject import reproject
        np_raster = self._strip(np.arange(8), 8, 'col')
        cp_raster = self._strip(np.arange(8), 8, 'col', use_cupy=True)
        np_vals = self._to_host(reproject(np_raster, 'EPSG:3857'))
        cp_vals = self._to_host(reproject(cp_raster, 'EPSG:3857'))
        assert np_vals.shape == cp_vals.shape
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(cp_vals))
        finite = np.isfinite(np_vals)
        if finite.any():
            np.testing.assert_allclose(
                np_vals[finite], cp_vals[finite], rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not (HAS_DASK and HAS_CUPY),
                        reason="dask and cupy required")
    def test_single_col_strip_dask_cupy_matches_numpy(self):
        from xrspatial.reproject import reproject
        np_raster = self._strip(np.arange(8), 8, 'col')
        dc_raster = self._strip(np.arange(8), 8, 'col',
                                use_dask=True, use_cupy=True, chunks=(4, 1))
        np_vals = self._to_host(reproject(np_raster, 'EPSG:3857'))
        dc_vals = self._to_host(reproject(dc_raster, 'EPSG:3857'))
        assert np_vals.shape == dc_vals.shape
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(dc_vals))
        finite = np.isfinite(np_vals)
        if finite.any():
            np.testing.assert_allclose(
                np_vals[finite], dc_vals[finite], rtol=1e-5, atol=1e-5)

    # -- constant-value (zero-gradient) raster -----------------------------

    def _constant(self, fill=7.0, use_dask=False, use_cupy=False,
                  chunks=(8, 8)):
        data = np.full((16, 16), fill, dtype=np.float64)
        y = np.linspace(5, -5, 16)
        x = np.linspace(-5, 5, 16)
        if use_cupy:
            data = cp.asarray(data)
        if use_dask:
            data = da.from_array(data, chunks=chunks)
        return xr.DataArray(
            data, dims=['y', 'x'], coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )

    def test_constant_raster_numpy_preserves_value(self):
        from xrspatial.reproject import reproject
        out = reproject(self._constant(fill=7.0), 'EPSG:3857',
                        resampling='bilinear')
        vals = self._to_host(out)
        finite = vals[np.isfinite(vals)]
        assert finite.size > 0
        # Zero gradient: every interpolated pixel must equal the fill value.
        np.testing.assert_allclose(finite, 7.0, rtol=0, atol=1e-9)

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_constant_raster_dask_matches_numpy(self):
        from xrspatial.reproject import reproject
        np_vals = self._to_host(reproject(self._constant(), 'EPSG:3857'))
        da_vals = self._to_host(
            reproject(self._constant(use_dask=True), 'EPSG:3857'))
        assert np_vals.shape == da_vals.shape
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(da_vals))
        finite = np.isfinite(np_vals)
        np.testing.assert_allclose(
            np_vals[finite], da_vals[finite], rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
    def test_constant_raster_cupy_matches_numpy(self):
        from xrspatial.reproject import reproject
        np_vals = self._to_host(reproject(self._constant(), 'EPSG:3857'))
        cp_vals = self._to_host(
            reproject(self._constant(use_cupy=True), 'EPSG:3857'))
        assert np_vals.shape == cp_vals.shape
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(cp_vals))
        finite = np.isfinite(np_vals)
        np.testing.assert_allclose(
            np_vals[finite], cp_vals[finite], rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not (HAS_DASK and HAS_CUPY),
                        reason="dask and cupy required")
    def test_constant_raster_dask_cupy_matches_numpy(self):
        from xrspatial.reproject import reproject
        np_vals = self._to_host(reproject(self._constant(), 'EPSG:3857'))
        dc_vals = self._to_host(reproject(
            self._constant(use_dask=True, use_cupy=True), 'EPSG:3857'))
        assert np_vals.shape == dc_vals.shape
        np.testing.assert_array_equal(np.isnan(np_vals), np.isnan(dc_vals))
        finite = np.isfinite(np_vals)
        np.testing.assert_allclose(
            np_vals[finite], dc_vals[finite], rtol=1e-5, atol=1e-5)


class TestCoordsPreservation:
    """Non-spatial coords pass through reproject() and merge()."""

    def _small_raster(self, name='test'):
        from xrspatial.tests.test_reproject import _make_raster
        data = np.random.RandomState(0).rand(8, 8).astype(np.float64)
        return _make_raster(data, name=name)

    def test_reproject_preserves_scalar_time_coord(self):
        from xrspatial.reproject import reproject
        raster = self._small_raster()
        ts = np.datetime64('2024-01-15')
        raster = raster.assign_coords(time=ts)

        out = reproject(raster, 'EPSG:3857')
        assert 'time' in out.coords
        assert out.coords['time'].values == ts

    def test_reproject_preserves_non_spatial_string_coord(self):
        from xrspatial.reproject import reproject
        raster = self._small_raster()
        raster = raster.assign_coords(source='tile_a')

        out = reproject(raster, 'EPSG:3857')
        assert 'source' in out.coords
        assert str(out.coords['source'].values) == 'tile_a'

    def test_reproject_drops_stale_y_coord_alias(self):
        from xrspatial.reproject import reproject
        raster = self._small_raster()
        # 'latitude' is a non-dim coord aligned to the y dim.
        latitude = ('y', raster.coords['y'].values.copy())
        raster = raster.assign_coords(latitude=latitude)
        assert 'latitude' in raster.coords

        out = reproject(raster, 'EPSG:3857')
        # The new grid's y values do not match the stale 'latitude'
        # values, so it must be dropped.
        assert 'latitude' not in out.coords

    def test_reproject_preserves_band_coord(self):
        from xrspatial.reproject import reproject
        data = np.random.RandomState(1).rand(8, 8, 3).astype(np.float64)
        y = np.linspace(1, -1, 8)
        x = np.linspace(-1, 1, 8)
        raster = xr.DataArray(
            data, dims=['y', 'x', 'band'],
            coords={'y': y, 'x': x, 'band': ['R', 'G', 'B']},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )

        out = reproject(raster, 'EPSG:3857')
        assert 'band' in out.coords
        assert list(out.coords['band'].values) == ['R', 'G', 'B']

    def test_merge_preserves_first_raster_scalar_coord(self):
        from xrspatial.reproject import merge
        r1 = self._small_raster(name='r1')
        r2 = self._small_raster(name='r2')
        ts = np.datetime64('2024-06-01')
        r1 = r1.assign_coords(time=ts)

        out = merge([r1, r2], target_crs='EPSG:4326')
        assert 'time' in out.coords
        assert out.coords['time'].values == ts

    def test_reproject_y_descending_regardless_of_input(self):
        from xrspatial.reproject import reproject
        # Build a y-ascending input (override default y direction)
        data = np.random.RandomState(2).rand(8, 8).astype(np.float64)
        y_asc = np.linspace(-1, 1, 8)  # ascending
        x = np.linspace(-1, 1, 8)
        raster = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': y_asc, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )

        out = reproject(raster, 'EPSG:3857')
        y_out = out.coords['y'].values
        # Strictly descending (top-down, north-up).
        assert np.all(np.diff(y_out) < 0), (
            f"Output y must be descending, got {y_out}"
        )

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_reproject_y_descending_dask(self):
        from xrspatial.reproject import reproject
        data = np.random.RandomState(3).rand(8, 8).astype(np.float64)
        y_asc = np.linspace(-1, 1, 8)
        x = np.linspace(-1, 1, 8)
        raster = xr.DataArray(
            da.from_array(data, chunks=4), dims=['y', 'x'],
            coords={'y': y_asc, 'x': x},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )

        out = reproject(raster, 'EPSG:3857')
        y_out = out.coords['y'].values
        assert np.all(np.diff(y_out) < 0)


# ---------------------------------------------------------------------------
# Inf input and chunk_size / max_memory parameter coverage
# ---------------------------------------------------------------------------

def test_reproject_handles_inf_input():
    """Reprojecting a raster with +/-Inf pixels must not crash.

    The output behavior is implementation-defined: Inf may propagate
    or be coerced to NaN. We only assert that the call returns and
    the spatial geometry is intact.
    """
    from xrspatial.reproject import reproject
    data = np.ones((32, 32), dtype=np.float64)
    data[0, 0] = np.inf
    data[1, 1] = -np.inf
    raster = xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': np.linspace(55, 45, 32),
                'x': np.linspace(-5, 5, 32)},
        attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
    )
    result = reproject(raster, 'EPSG:32633')
    assert result.ndim == 2
    assert result.shape[0] >= 1 and result.shape[1] >= 1


@pytest.mark.skipif(not HAS_DASK, reason="dask required")
def test_reproject_chunk_size_tuple():
    """Tuple chunk_size should propagate to the output dask chunks."""
    from xrspatial.reproject import reproject
    data = np.random.RandomState(0).rand(128, 128).astype(np.float64)
    raster = xr.DataArray(
        da.from_array(data, chunks=64), dims=['y', 'x'],
        coords={'y': np.linspace(55, 45, 128),
                'x': np.linspace(-5, 5, 128)},
        attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
    )
    out = reproject(raster, 'EPSG:32633', chunk_size=(64, 32))
    assert hasattr(out.data, 'chunks'), "expected dask-backed output"
    row_chunks, col_chunks = out.data.chunks
    # Allow the last chunk to be a remainder, but the leading chunk
    # should match the requested size.
    assert row_chunks[0] == 64
    assert col_chunks[0] == 32


def test_reproject_max_memory_string_arg():
    """Reproject must accept human-readable max_memory strings."""
    from xrspatial.reproject import reproject
    data = np.random.RandomState(0).rand(32, 32).astype(np.float64)
    raster = xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': np.linspace(55, 45, 32),
                'x': np.linspace(-5, 5, 32)},
        attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
    )
    for mem in ('256MB', '1GB'):
        out = reproject(raster, 'EPSG:32633', max_memory=mem)
        assert out.ndim == 2


def test_reproject_max_memory_int_arg():
    """Reproject must accept integer byte counts for max_memory."""
    from xrspatial.reproject import reproject
    data = np.random.RandomState(0).rand(32, 32).astype(np.float64)
    raster = xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': np.linspace(55, 45, 32),
                'x': np.linspace(-5, 5, 32)},
        attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
    )
    out = reproject(raster, 'EPSG:32633', max_memory=512 * 1024 * 1024)
    assert out.ndim == 2


# ---------------------------------------------------------------------------
# 2026-05-10 test-coverage sweep additions
# ---------------------------------------------------------------------------

class TestLiteCRS:
    """Direct coverage for the no-pyproj fallback CRS class.

    ``_lite_crs.CRS`` ships as the fast path inside ``_resolve_crs`` and as
    the only CRS implementation when pyproj is unavailable. Without these
    tests a regression in the built-in EPSG table or the WKT generator
    would only surface in an environment that drops pyproj.
    """

    def test_construct_from_int(self):
        from xrspatial.reproject._lite_crs import CRS
        c = CRS(4326)
        assert c.to_epsg() == 4326
        assert c.is_geographic is True

    def test_construct_from_string(self):
        from xrspatial.reproject._lite_crs import CRS
        c = CRS('EPSG:3857')
        assert c.to_epsg() == 3857
        assert c.is_geographic is False

    def test_construct_from_lowercase_epsg_string(self):
        from xrspatial.reproject._lite_crs import CRS
        c = CRS('epsg:4326')
        assert c.to_epsg() == 4326

    def test_unknown_epsg_rejected(self):
        from xrspatial.reproject._lite_crs import CRS
        with pytest.raises(ValueError, match="not in the built-in table"):
            CRS(9_999_999)

    def test_bad_string_rejected(self):
        from xrspatial.reproject._lite_crs import CRS
        with pytest.raises(ValueError, match="Cannot parse"):
            CRS('not-a-crs')

    def test_bad_type_rejected(self):
        from xrspatial.reproject._lite_crs import CRS
        with pytest.raises(TypeError):
            CRS(4326.0)

    def test_to_authority(self):
        from xrspatial.reproject._lite_crs import CRS
        assert CRS(4326).to_authority() == ('EPSG', '4326')

    def test_to_dict_strips_internal_keys(self):
        from xrspatial.reproject._lite_crs import CRS
        d = CRS(4326).to_dict()
        # Internal keys like _is_geographic must not leak into the dict
        assert all(not k.startswith('_') for k in d)
        assert d.get('proj') == 'longlat'

    def test_equality_and_hash(self):
        from xrspatial.reproject._lite_crs import CRS
        assert CRS(4326) == CRS(4326)
        assert CRS(4326) != CRS(3857)
        # Hashable for use as dict key
        s = {CRS(4326), CRS(4326), CRS(3857)}
        assert len(s) == 2

    def test_wkt_geographic(self):
        from xrspatial.reproject._lite_crs import CRS
        wkt = CRS(4326).to_wkt()
        assert 'GEOGCS' in wkt
        assert 'AUTHORITY["EPSG","4326"]' in wkt

    def test_wkt_projected(self):
        from xrspatial.reproject._lite_crs import CRS
        wkt = CRS(3857).to_wkt()
        assert 'PROJCS' in wkt
        assert 'AUTHORITY["EPSG","3857"]' in wkt

    def test_wkt_utm_zone_expanded(self):
        from xrspatial.reproject._lite_crs import CRS
        # UTM 33N: central_meridian = 33*6 - 183 = 15
        wkt = CRS(32633).to_wkt()
        assert 'central_meridian' in wkt
        assert '15' in wkt  # the central meridian for UTM 33N

    def test_wkt_roundtrip(self):
        from xrspatial.reproject._lite_crs import CRS
        for code in (4326, 3857, 32633, 5070):
            recovered = CRS.from_wkt(CRS(code).to_wkt())
            assert recovered.to_epsg() == code

    def test_from_wkt_rejects_string_without_authority(self):
        from xrspatial.reproject._lite_crs import CRS
        with pytest.raises(ValueError, match="No AUTHORITY"):
            CRS.from_wkt('PROJCS["no-authority-here"]')

    def test_lite_crs_used_when_pyproj_missing(self, monkeypatch):
        """_resolve_crs must succeed for table EPSG codes even without pyproj."""
        from xrspatial.reproject import _crs_utils as cu
        from xrspatial.reproject._lite_crs import CRS as LiteCRS

        monkeypatch.setattr(cu, '_try_import_pyproj', lambda: None)
        # Built-in code: should round-trip through LiteCRS only
        resolved = cu._resolve_crs(4326)
        assert isinstance(resolved, LiteCRS)
        assert resolved.to_epsg() == 4326

    def test_crs_from_wkt_uses_lite_first(self, monkeypatch):
        """_crs_from_wkt extracts AUTHORITY tag without invoking pyproj."""
        from xrspatial.reproject import _crs_utils as cu
        from xrspatial.reproject._lite_crs import CRS as LiteCRS

        def _no_pyproj():
            raise ImportError("pyproj disabled for this test")

        # If lite path works, _require_pyproj must not be reached.
        monkeypatch.setattr(cu, '_require_pyproj', _no_pyproj)
        wkt = LiteCRS(4326).to_wkt()
        recovered = cu._crs_from_wkt(wkt)
        assert recovered.to_epsg() == 4326


class TestItrfBehaviour:
    """Numerical behaviour of itrf_transform / itrf_frames.

    Existing tests only cover error paths. These add a frame-listing
    smoke check and a round-trip behavioural check so that a change to
    the 14-parameter Helmert math would surface.
    """

    def test_itrf_frames_lists_known_frames(self):
        from xrspatial.reproject import itrf_frames
        frames = itrf_frames()
        assert isinstance(frames, list)
        # The four standard ITRF realizations should be present.
        for f in ('ITRF2000', 'ITRF2008', 'ITRF2014', 'ITRF2020'):
            assert f in frames, f"missing frame: {f}"

    def test_itrf_transform_scalar_small_shift(self):
        """ITRF2014 -> ITRF2020 shift is at the sub-mm/m level for short
        epochs, so the output coordinates must be very close to the input."""
        from xrspatial.reproject import itrf_transform
        lon, lat, h = -74.0, 40.7, 10.0
        out_lon, out_lat, out_h = itrf_transform(
            lon, lat, h, src='ITRF2014', tgt='ITRF2020', epoch=2024.0,
        )
        # Sanity: a few-cm-level shift in geographic coords (~1e-7 deg)
        # and a few-mm to cm shift in height.
        assert abs(out_lon - lon) < 1e-5
        assert abs(out_lat - lat) < 1e-5
        assert abs(out_h - h) < 0.05

    def test_itrf_transform_roundtrip(self):
        """Forward then reverse should recover the input."""
        from xrspatial.reproject import itrf_transform
        lon, lat, h = -74.0, 40.7, 10.0
        fwd = itrf_transform(lon, lat, h, src='ITRF2014', tgt='ITRF2020',
                             epoch=2024.0)
        back = itrf_transform(fwd[0], fwd[1], fwd[2],
                              src='ITRF2020', tgt='ITRF2014',
                              epoch=2024.0)
        assert abs(back[0] - lon) < 1e-9
        assert abs(back[1] - lat) < 1e-9
        assert abs(back[2] - h) < 1e-6

    def test_itrf_transform_array_input(self):
        """Array inputs produce array outputs of matching shape."""
        from xrspatial.reproject import itrf_transform
        lons = np.array([-74.0, 0.0, 10.0])
        lats = np.array([40.7, 0.0, 50.0])
        hs = np.array([10.0, 0.0, 100.0])
        out_lon, out_lat, out_h = itrf_transform(
            lons, lats, hs, src='ITRF2014', tgt='ITRF2020', epoch=2024.0,
        )
        assert out_lon.shape == lons.shape
        assert out_lat.shape == lats.shape
        assert out_h.shape == hs.shape
        # Each coordinate must shift by less than a few cm at this epoch.
        assert np.all(np.abs(out_lon - lons) < 1e-5)
        assert np.all(np.abs(out_lat - lats) < 1e-5)

    def test_itrf_transform_unknown_frame_raises(self):
        from xrspatial.reproject import itrf_transform
        with pytest.raises(ValueError, match="No transform"):
            itrf_transform(0.0, 0.0, 0.0,
                           src='ITRF1900', tgt='ITRF2020', epoch=2024.0)


class TestGeoidHeightBehaviour:
    """Numerical correctness for the public geoid helpers.

    Existing tests cover error paths and use these only as references.
    Their direct numerical behaviour is not asserted anywhere, so a
    silent regression in the EGM96 grid loader or the bilinear
    interpolation would not be caught.
    """

    # Reference EGM96 undulation at known locations (metres). These were
    # produced by the same code path under test so they pin the current
    # behaviour rather than an external authority. A drift of more than
    # a few metres in either direction would indicate a real change.
    _REFERENCE_N = {
        # (lon, lat): expected N in metres
        (-74.0, 40.7): -33.0,    # New York
        (0.0, 0.0): 17.2,        # null island
        (139.7, 35.7): 38.7,     # Tokyo
        (-150.0, 60.0): 13.3,    # central Alaska
    }

    def test_geoid_height_scalar(self):
        from xrspatial.reproject import geoid_height
        for (lon, lat), expected in self._REFERENCE_N.items():
            N = geoid_height(lon, lat)
            assert isinstance(N, float)
            assert abs(N - expected) < 3.0, (
                f"N({lon},{lat}) = {N}, expected ~{expected}"
            )

    def test_geoid_height_array_matches_scalar(self):
        from xrspatial.reproject import geoid_height
        coords = list(self._REFERENCE_N.keys())
        lons = np.array([c[0] for c in coords])
        lats = np.array([c[1] for c in coords])
        batch = geoid_height(lons, lats)
        assert batch.shape == lons.shape
        for i, c in enumerate(coords):
            scalar = geoid_height(c[0], c[1])
            assert abs(batch[i] - scalar) < 1e-9

    def test_geoid_height_longitude_wrap(self):
        """Lon and lon+360 must give the same value (grid wraps globally)."""
        from xrspatial.reproject import geoid_height
        for lon in (-179.5, 0.0, 179.5):
            for lat in (-45.0, 0.0, 45.0):
                a = geoid_height(lon, lat)
                b = geoid_height(lon + 360.0, lat)
                assert abs(a - b) < 1e-9, (
                    f"lon={lon} vs lon+360: {a} != {b}"
                )

    def test_geoid_height_near_poles_finite(self):
        from xrspatial.reproject import geoid_height
        N_north = geoid_height(0.0, 89.5)
        N_south = geoid_height(0.0, -89.5)
        assert np.isfinite(N_north)
        assert np.isfinite(N_south)

    def test_geoid_height_2d_array_input(self):
        """A 2D coord grid produces a 2D output of the same shape."""
        from xrspatial.reproject import geoid_height
        lons2d, lats2d = np.meshgrid(
            np.linspace(-10, 10, 5), np.linspace(40, 50, 4),
        )
        out = geoid_height(lons2d, lats2d)
        assert out.shape == lons2d.shape
        assert np.isfinite(out).all()

    def test_geoid_height_raster_happy_path(self):
        """``geoid_height_raster`` returns an N raster whose values agree
        with point-wise ``geoid_height`` at each pixel."""
        from xrspatial.reproject import geoid_height, geoid_height_raster

        y = np.linspace(45.0, 35.0, 6)
        x = np.linspace(-80.0, -70.0, 7)
        raster = xr.DataArray(
            np.zeros((y.size, x.size), dtype=np.float64),
            dims=['y', 'x'],
            coords={'y': y, 'x': x},
        )
        out = geoid_height_raster(raster)

        assert out.shape == raster.shape
        assert out.dims == ('y', 'x')
        np.testing.assert_array_equal(out.coords['y'].values, y)
        np.testing.assert_array_equal(out.coords['x'].values, x)
        assert out.attrs.get('units') == 'metres'
        assert out.attrs.get('model') == 'EGM96'

        # Every pixel must match the scalar function.
        for i, yi in enumerate(y):
            for j, xj in enumerate(x):
                expected = geoid_height(float(xj), float(yi))
                assert abs(float(out.values[i, j]) - expected) < 1e-9

    def test_geoid_height_raster_with_lat_lon_dims(self):
        """``geoid_height_raster`` works on rasters with lat/lon dim names."""
        from xrspatial.reproject import geoid_height_raster

        lat = np.linspace(45.0, 35.0, 5)
        lon = np.linspace(-80.0, -70.0, 5)
        raster = xr.DataArray(
            np.zeros((lat.size, lon.size), dtype=np.float64),
            dims=['lat', 'lon'],
            coords={'lat': lat, 'lon': lon},
        )
        out = geoid_height_raster(raster)
        assert out.dims == ('lat', 'lon')
        assert np.isfinite(out.values).all()


class TestPyprojGeoidProbeUsable:
    """Coverage for ``_pyproj_geoid_probe_is_usable`` (#2567).

    The helper guards pyproj-based geoid cross-checks against runners
    where the EGM96 grid is not installed. Both the no-op fallback (~0)
    and the non-finite fallback (-inf / +inf / nan) must be classified
    as "grid unavailable" so the test skips instead of asserting.
    """

    def test_typical_finite_probe_is_usable(self):
        # ~-32.8 m at New York when the grid is actually installed.
        assert _pyproj_geoid_probe_is_usable(-32.8)

    def test_near_zero_probe_is_not_usable(self):
        # No-op fallback at a point with real undulation.
        assert not _pyproj_geoid_probe_is_usable(0.0)
        assert not _pyproj_geoid_probe_is_usable(0.5)
        assert not _pyproj_geoid_probe_is_usable(-0.5)

    def test_negative_inf_probe_is_not_usable(self):
        # Regression for the original bug: -inf used to slip past the
        # near-zero guard and fire the assert in the pyproj cross-check.
        assert not _pyproj_geoid_probe_is_usable(float('-inf'))

    def test_positive_inf_probe_is_not_usable(self):
        assert not _pyproj_geoid_probe_is_usable(float('inf'))

    def test_nan_probe_is_not_usable(self):
        assert not _pyproj_geoid_probe_is_usable(float('nan'))

    def test_zero_tol_is_configurable(self):
        # A real lookup that happens to be 0.5 m should still count as
        # usable when the caller picks a tighter tolerance.
        assert _pyproj_geoid_probe_is_usable(0.5, zero_tol=0.1)


class TestGeoidPixelCenterIndexing:
    """Regression coverage for the half-pixel offset bug (#2508).

    The EGM96 GeoTIFF is pixel-center anchored: ``data[r, c]`` is the
    value at ``(left + (c + 0.5) * res_x, top - (r + 0.5) * res_y)``.
    Before #2508 the bilinear lookup indexed in pixel-edge space, which
    produced up to ~2 m error at pixel centers and an 8-9 cm error at
    representative locations like New York vs pyproj's geoid lookup.
    """

    def test_geoid_at_pixel_center_returns_stored_value(self):
        """A query at the exact pixel center must return the stored cell
        value (modulo float round-off), not a blend with the neighbour.
        """
        from xrspatial.reproject._vertical import (
            _interp_geoid_point, _load_geoid,
        )

        data, left, top, res_x, res_y, h, w = _load_geoid('EGM96')

        for (i, j) in [(0, 0), (10, 100), (h // 2, w // 2),
                       (h - 1, w - 1)]:
            lon_c = left + (j + 0.5) * res_x
            lat_c = top - (i + 0.5) * res_y
            N = _interp_geoid_point(
                lon_c, lat_c, data, left, top, res_x, res_y, h, w,
            )
            assert abs(N - data[i, j]) < 1e-9, (
                f"pixel ({i},{j}) center query expected "
                f"data[{i},{j}]={data[i, j]!r}, got {N!r}; "
                f"half-pixel offset bug from #2508?"
            )

    def test_geoid_height_matches_pyproj_within_cm(self):
        """``geoid_height`` must agree with pyproj's EGM96 lookup to the
        centimetre at well-sampled locations. The old half-pixel bias was
        ~9 cm at New York; this test would fail by ~9 cm if reintroduced.
        """
        pyproj = pytest.importorskip('pyproj')
        from xrspatial.reproject import geoid_height

        src_crs = pyproj.CRS('EPSG:4979')
        tgt_crs = pyproj.CRS('EPSG:5773')
        transformer = pyproj.Transformer.from_crs(
            src_crs, tgt_crs, always_xy=True,
        )

        # pyproj falls back when the EGM96 grid is not installed locally
        # and PROJ network access is disabled (typical CI). The fallback
        # is either a no-op transform (~0 at New York, where the real
        # geoid undulation is tens of metres) or a non-finite sentinel
        # (-inf / +inf / nan). Probe at New York and skip in either case
        # -- there's nothing to cross-check against.
        _, _, h_probe = transformer.transform(-74.0, 40.7, 0.0)
        if not _pyproj_geoid_probe_is_usable(h_probe):
            pytest.skip(
                "pyproj EGM96 grid unavailable on this runner "
                f"(probe at New York returned {h_probe!r}); "
                "cannot cross-check"
            )

        sample_points = [
            (-74.0, 40.7),
            (0.0, 0.0),
            (139.7, 35.7),
            (-150.0, 60.0),
            (-180.0, 90.0),  # data[0,0]: the offset bug was largest here
        ]
        for lon, lat in sample_points:
            _, _, h_ortho = transformer.transform(lon, lat, 0.0)
            N_expected = -h_ortho  # h_ellip(=0) - h_ortho = -h_ortho
            N_actual = geoid_height(lon, lat)
            assert abs(N_actual - N_expected) < 1e-2, (
                f"N({lon},{lat}) = {N_actual}, pyproj says "
                f"{N_expected}; diff {N_actual - N_expected:.4f} m"
            )


class TestVerticalHelperConversions:
    """Direct coverage for the four public vertical-conversion helpers.

    ``ellipsoidal_to_orthometric``, ``orthometric_to_ellipsoidal``,
    ``depth_to_ellipsoidal`` and ``ellipsoidal_to_depth`` are exported
    from ``xrspatial.reproject`` but only the reproject() integration
    path is exercised in existing tests.
    """

    @staticmethod
    def _ny():
        return (-74.0, 40.7)

    def test_ellipsoidal_to_orthometric_scalar(self):
        from xrspatial.reproject import (
            ellipsoidal_to_orthometric, geoid_height,
        )
        lon, lat = self._ny()
        N = geoid_height(lon, lat)
        H = ellipsoidal_to_orthometric(100.0, lon, lat)
        # H = h - N
        assert abs(float(H) - (100.0 - N)) < 1e-9

    def test_orthometric_to_ellipsoidal_scalar(self):
        from xrspatial.reproject import (
            geoid_height, orthometric_to_ellipsoidal,
        )
        lon, lat = self._ny()
        N = geoid_height(lon, lat)
        h = orthometric_to_ellipsoidal(100.0, lon, lat)
        # h = H + N
        assert abs(float(h) - (100.0 + N)) < 1e-9

    def test_ellipsoidal_orthometric_roundtrip(self):
        from xrspatial.reproject import (
            ellipsoidal_to_orthometric, orthometric_to_ellipsoidal,
        )
        lon, lat = self._ny()
        h0 = 1234.5
        H = ellipsoidal_to_orthometric(h0, lon, lat)
        h1 = orthometric_to_ellipsoidal(H, lon, lat)
        assert abs(float(h1) - h0) < 1e-9

    def test_depth_to_ellipsoidal_scalar(self):
        from xrspatial.reproject import (
            depth_to_ellipsoidal, geoid_height,
        )
        lon, lat = self._ny()
        N = geoid_height(lon, lat)
        h = depth_to_ellipsoidal(50.0, lon, lat)
        # h = -depth + N
        assert abs(float(h) - (-50.0 + N)) < 1e-9

    def test_ellipsoidal_to_depth_scalar(self):
        from xrspatial.reproject import (
            ellipsoidal_to_depth, geoid_height,
        )
        lon, lat = self._ny()
        N = geoid_height(lon, lat)
        depth = ellipsoidal_to_depth(-50.0, lon, lat)
        # depth = N - h
        assert abs(float(depth) - (N - (-50.0))) < 1e-9

    def test_depth_ellipsoidal_roundtrip(self):
        from xrspatial.reproject import (
            depth_to_ellipsoidal, ellipsoidal_to_depth,
        )
        lon, lat = self._ny()
        depth0 = 20.0
        h = depth_to_ellipsoidal(depth0, lon, lat)
        depth1 = ellipsoidal_to_depth(h, lon, lat)
        assert abs(float(depth1) - depth0) < 1e-9

    def test_vertical_helpers_array_input(self):
        """Array inputs broadcast to the same shape as the input height."""
        from xrspatial.reproject import (
            ellipsoidal_to_orthometric, orthometric_to_ellipsoidal,
        )
        heights = np.array([0.0, 100.0, -50.0, 1234.5])
        lons = np.full_like(heights, -74.0)
        lats = np.full_like(heights, 40.7)
        H = ellipsoidal_to_orthometric(heights, lons, lats)
        assert H.shape == heights.shape
        # Roundtrip every element.
        back = orthometric_to_ellipsoidal(H, lons, lats)
        np.testing.assert_allclose(back, heights, atol=1e-9)


class TestReprojectLatLonDimPropagation:
    """Cat 5 (metadata preservation): reproject() must keep ``lat``/``lon``
    dim names when the input uses them instead of the canonical ``y``/``x``.

    A regression that renames the spatial dims to ``y``/``x`` would
    silently break any downstream code keyed on the input naming.
    """

    @staticmethod
    def _lat_lon_raster(crs='EPSG:4326'):
        data = np.ones((8, 8), dtype=np.float64)
        lat = np.linspace(5.0, -5.0, 8)
        lon = np.linspace(-5.0, 5.0, 8)
        return xr.DataArray(
            data, dims=['lat', 'lon'],
            coords={'lat': lat, 'lon': lon},
            attrs={'crs': crs, 'nodata': np.nan},
        )

    def test_reproject_preserves_lat_lon_dim_names_same_crs(self):
        from xrspatial.reproject import reproject
        raster = self._lat_lon_raster()
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert result.dims == ('lat', 'lon')
        assert 'lat' in result.coords
        assert 'lon' in result.coords

    def test_reproject_preserves_lat_lon_dim_names_cross_crs(self):
        from xrspatial.reproject import reproject
        raster = self._lat_lon_raster()
        # Cross-CRS reprojection: lat/lon are no longer geographic in the
        # target, but the dim names must still flow through.
        result = reproject(raster, 'EPSG:3857')
        assert result.dims == ('lat', 'lon')

    def test_reproject_preserves_latitude_longitude_dim_names(self):
        """Long-form ``latitude``/``longitude`` are also recognised."""
        from xrspatial.reproject import reproject
        data = np.ones((8, 8), dtype=np.float64)
        lat = np.linspace(5.0, -5.0, 8)
        lon = np.linspace(-5.0, 5.0, 8)
        raster = xr.DataArray(
            data, dims=['latitude', 'longitude'],
            coords={'latitude': lat, 'longitude': lon},
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        result = reproject(raster, 'EPSG:4326', resolution=1.0)
        assert result.dims == ('latitude', 'longitude')

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_reproject_preserves_lat_lon_dim_names_dask(self):
        from xrspatial.reproject import reproject
        raster = self._lat_lon_raster()
        raster.data = da.from_array(raster.values, chunks=(4, 4))
        result = reproject(raster, 'EPSG:3857', chunk_size=4)
        assert result.dims == ('lat', 'lon')


# =====================================================================
# Issue #2027: 3-D (y, x, band) inputs across all backends
# =====================================================================

class TestReproject3DBackends:
    """reproject() must honour the band axis on every backend.

    The 2-D path worked for years; the dask, cupy, and dask+cupy paths
    either silently dropped the band dim from the lazy DataArray or
    crashed with a CUDA signature mismatch on 3-D inputs (#2027).
    """

    @staticmethod
    def _make_3d_raster(rng_seed=0, h=32, w=32, n_bands=3, dtype=np.float32):
        rng = np.random.default_rng(rng_seed)
        data = rng.random((h, w, n_bands), dtype=np.float32).astype(dtype)
        return xr.DataArray(
            data,
            dims=['y', 'x', 'band'],
            coords={
                'y': np.linspace(55, 45, h),
                'x': np.linspace(-5, 5, w),
                'band': list(range(n_bands)),
            },
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )

    def test_reproject_3d_numpy(self):
        """Baseline: 3-D numpy reproject keeps band dim."""
        from xrspatial.reproject import reproject
        raster = self._make_3d_raster()
        result = reproject(raster, 'EPSG:32633')
        assert result.ndim == 3
        assert result.dims == ('y', 'x', 'band')
        assert result.shape[2] == 3
        # Computed values should be finite for at least part of the output
        assert np.any(np.isfinite(result.values))

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_reproject_3d_dask_lazy_shape(self):
        """Lazy dask DataArray must advertise 3-D shape (not 2-D)."""
        from xrspatial.reproject import reproject
        raster = self._make_3d_raster()
        raster = raster.copy(
            data=da.from_array(raster.values, chunks=(16, 16, 3))
        )
        result = reproject(raster, 'EPSG:32633')
        assert result.ndim == 3
        assert result.dims == ('y', 'x', 'band')
        assert result.shape[2] == 3

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_reproject_3d_dask_compute(self):
        """Computed dask result keeps band axis without ValueError."""
        from xrspatial.reproject import reproject
        raster = self._make_3d_raster()
        raster = raster.copy(
            data=da.from_array(raster.values, chunks=(16, 16, 3))
        )
        result = reproject(raster, 'EPSG:32633').compute()
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert np.any(np.isfinite(result.values))

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_reproject_3d_dask_matches_numpy(self):
        """Dask 3-D output should match eager numpy output pixel-for-pixel."""
        from xrspatial.reproject import reproject
        raster = self._make_3d_raster()
        eager = reproject(raster, 'EPSG:32633')
        lazy_src = raster.copy(
            data=da.from_array(raster.values, chunks=(16, 16, 3))
        )
        lazy = reproject(lazy_src, 'EPSG:32633').compute()
        np.testing.assert_allclose(
            np.asarray(eager.values), np.asarray(lazy.values),
            rtol=1e-6, atol=1e-6, equal_nan=True,
        )

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_reproject_3d_dask_uint8_dtype_roundtrip(self):
        """Integer 3-D dask inputs round-trip to source dtype."""
        from xrspatial.reproject import reproject
        rng = np.random.default_rng(1)
        data = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        raster = xr.DataArray(
            da.from_array(data, chunks=(16, 16, 3)),
            dims=['y', 'x', 'band'],
            coords={'y': np.linspace(55, 45, 32), 'x': np.linspace(-5, 5, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': 0},
        )
        result = reproject(raster, 'EPSG:32633').compute()
        assert result.dtype == np.uint8
        assert result.shape[2] == 3

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
    def test_reproject_3d_cupy(self):
        """CuPy 3-D reproject keeps band dim without CUDA signature crash."""
        from xrspatial.reproject import reproject
        host = self._make_3d_raster()
        gpu_data = cp.asarray(host.values)
        raster = host.copy(data=gpu_data)
        result = reproject(raster, 'EPSG:32633')
        assert result.ndim == 3
        assert result.shape[2] == 3
        # Pull back to host to verify finite values
        out = cp.asnumpy(result.data) if isinstance(result.data, cp.ndarray) \
            else np.asarray(result.values)
        assert np.any(np.isfinite(out))

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
    def test_reproject_3d_cupy_matches_numpy(self):
        from xrspatial.reproject import reproject
        host = self._make_3d_raster()
        eager = reproject(host, 'EPSG:32633').values
        gpu = host.copy(data=cp.asarray(host.values))
        gpu_out = reproject(gpu, 'EPSG:32633')
        gpu_arr = cp.asnumpy(gpu_out.data) if isinstance(gpu_out.data, cp.ndarray) \
            else np.asarray(gpu_out.values)
        np.testing.assert_allclose(
            eager, gpu_arr, rtol=1e-4, atol=1e-4, equal_nan=True,
        )

    @pytest.mark.skipif(
        not (HAS_CUPY and HAS_DASK), reason="CuPy + dask required",
    )
    def test_reproject_3d_dask_cupy(self):
        """dask+cupy 3-D reproject keeps band dim."""
        from xrspatial.reproject import reproject
        host = self._make_3d_raster()
        gpu_data = da.from_array(cp.asarray(host.values), chunks=(16, 16, 3))
        raster = host.copy(data=gpu_data)
        result = reproject(raster, 'EPSG:32633')
        assert result.ndim == 3
        assert result.dims == ('y', 'x', 'band')
        computed = result.compute()
        assert computed.shape[2] == 3

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
    def test_reproject_3d_cupy_uint8_sentinel_nodata(self):
        """3-D cupy with integer sentinel nodata round-trips to source dtype.

        Exercises the non-NaN nodata path that the float tests skip.
        """
        from xrspatial.reproject import reproject
        rng = np.random.default_rng(2)
        host = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        raster = xr.DataArray(
            cp.asarray(host),
            dims=['y', 'x', 'band'],
            coords={'y': np.linspace(55, 45, 32), 'x': np.linspace(-5, 5, 32)},
            attrs={'crs': 'EPSG:4326', 'nodata': 0},
        )
        result = reproject(raster, 'EPSG:32633')
        assert result.dtype == np.uint8
        assert result.shape[2] == 3


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
class TestMerge3DRejection:
    """merge() must reject 3-D inputs with a clear error (#2027).

    Before the fix, merge() advertised 3-D support via its validator but
    crashed at output DataArray construction because the merge strategies,
    same-CRS placement, and final `dims=[ydim, xdim]` all assume 2-D. We
    tighten the validator so callers see a clean message instead.
    """

    def test_merge_rejects_3d_dataarray(self):
        from xrspatial.reproject import merge
        a = xr.DataArray(
            np.random.rand(8, 8, 3),
            dims=['y', 'x', 'band'],
            coords={
                'y': np.linspace(5, -5, 8),
                'x': np.linspace(-5, 0, 8),
                'band': [1, 2, 3],
            },
            attrs={'crs': 'EPSG:4326'},
        )
        b = xr.DataArray(
            np.random.rand(8, 8, 3),
            dims=['y', 'x', 'band'],
            coords={
                'y': np.linspace(5, -5, 8),
                'x': np.linspace(0, 5, 8),
                'band': [1, 2, 3],
            },
            attrs={'crs': 'EPSG:4326'},
        )
        with pytest.raises(ValueError, match=r"must be 2D"):
            merge([a, b], resolution=1.0)


# =====================================================================
# Issue #2182: 3-D (band, y, x) inputs across all backends
# =====================================================================

@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
class TestReproject3DBandFirst:
    """reproject() must accept (band, y, x) inputs (rasterio convention).

    Before the fix, the worker sliced the source as ``source_data[r:, c:]``
    and read ``window.shape[2]`` for the band count, both of which assume
    a trailing band axis. A ``(band, y, x)`` source therefore sliced the
    band/y axes instead of y/x and either crashed with a coord-length
    mismatch or returned wrong-shape data (#2182).
    """

    @staticmethod
    def _make_band_first_raster(rng_seed=2182, h=32, w=32, n_bands=3,
                                dtype=np.float32):
        rng = np.random.default_rng(rng_seed)
        data = rng.random((h, w, n_bands), dtype=np.float32).astype(dtype)
        # Build (y, x, band) first so we can transpose to (band, y, x) and
        # keep coords aligned to the same underlying values.
        yxb = xr.DataArray(
            data,
            dims=['y', 'x', 'band'],
            coords={
                'y': np.linspace(55, 45, h),
                'x': np.linspace(-5, 5, w),
                'band': list(range(n_bands)),
            },
            attrs={'crs': 'EPSG:4326', 'nodata': np.nan},
        )
        return yxb.transpose('band', 'y', 'x')

    def test_band_first_numpy_dims_preserved(self):
        """``(band, y, x)`` input must produce ``(band, y, x)`` output."""
        from xrspatial.reproject import reproject
        raster = self._make_band_first_raster()
        result = reproject(raster, 'EPSG:32633')
        assert result.dims == ('band', 'y', 'x')
        assert result.shape[0] == 3
        assert np.any(np.isfinite(result.values))

    def test_band_first_numpy_band_coord_preserved(self):
        """Band coord values must round-trip through reproject."""
        from xrspatial.reproject import reproject
        raster = self._make_band_first_raster(n_bands=3)
        result = reproject(raster, 'EPSG:32633')
        assert 'band' in result.coords
        assert list(result.coords['band'].values) == [0, 1, 2]

    def test_band_first_matches_band_last(self):
        """The two layouts must produce identical pixel values."""
        from xrspatial.reproject import reproject
        bxy = self._make_band_first_raster()
        yxb = bxy.transpose('y', 'x', 'band')
        out_bxy = reproject(bxy, 'EPSG:32633').transpose('y', 'x', 'band')
        out_yxb = reproject(yxb, 'EPSG:32633')
        np.testing.assert_array_equal(
            np.asarray(out_bxy.values), np.asarray(out_yxb.values),
        )

    def test_band_first_uint8_dtype_roundtrip(self):
        """Integer (band, y, x) inputs round-trip to source dtype."""
        from xrspatial.reproject import reproject
        rng = np.random.default_rng(11)
        data = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
        raster = xr.DataArray(
            data,
            dims=['band', 'y', 'x'],
            coords={
                'band': [1, 2, 3],
                'y': np.linspace(55, 45, 32),
                'x': np.linspace(-5, 5, 32),
            },
            attrs={'crs': 'EPSG:4326', 'nodata': 0},
        )
        result = reproject(raster, 'EPSG:32633')
        assert result.dtype == np.uint8
        assert result.dims == ('band', 'y', 'x')
        assert result.shape[0] == 3

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_band_first_dask_lazy_shape(self):
        """Lazy dask (band, y, x) DataArray must advertise 3-D shape."""
        from xrspatial.reproject import reproject
        raster = self._make_band_first_raster()
        raster = raster.copy(
            data=da.from_array(raster.values, chunks=(3, 16, 16))
        )
        result = reproject(raster, 'EPSG:32633')
        assert result.ndim == 3
        assert result.dims == ('band', 'y', 'x')
        assert result.shape[0] == 3

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_band_first_dask_compute(self):
        """Computed dask result keeps band axis without ValueError."""
        from xrspatial.reproject import reproject
        raster = self._make_band_first_raster()
        raster = raster.copy(
            data=da.from_array(raster.values, chunks=(3, 16, 16))
        )
        result = reproject(raster, 'EPSG:32633').compute()
        assert result.dims == ('band', 'y', 'x')
        assert result.shape[0] == 3
        assert np.any(np.isfinite(result.values))

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_band_first_dask_matches_numpy(self):
        """Dask (band, y, x) output must match eager numpy output."""
        from xrspatial.reproject import reproject
        host = self._make_band_first_raster()
        eager = reproject(host, 'EPSG:32633')
        lazy_src = host.copy(
            data=da.from_array(host.values, chunks=(3, 16, 16))
        )
        lazy = reproject(lazy_src, 'EPSG:32633').compute()
        np.testing.assert_allclose(
            np.asarray(eager.values), np.asarray(lazy.values),
            rtol=1e-6, atol=1e-6, equal_nan=True,
        )

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
    def test_band_first_cupy(self):
        """CuPy (band, y, x) reproject keeps band dim and dim order."""
        from xrspatial.reproject import reproject
        host = self._make_band_first_raster()
        gpu_data = cp.asarray(host.values)
        raster = host.copy(data=gpu_data)
        result = reproject(raster, 'EPSG:32633')
        assert result.dims == ('band', 'y', 'x')
        assert result.shape[0] == 3
        out = (cp.asnumpy(result.data) if isinstance(result.data, cp.ndarray)
               else np.asarray(result.values))
        assert np.any(np.isfinite(out))

    @pytest.mark.skipif(
        not (HAS_CUPY and HAS_DASK), reason="CuPy + dask required",
    )
    def test_band_first_dask_cupy(self):
        """dask+cupy (band, y, x) reproject keeps band dim and dim order."""
        from xrspatial.reproject import reproject
        host = self._make_band_first_raster()
        gpu_data = da.from_array(cp.asarray(host.values), chunks=(3, 16, 16))
        raster = host.copy(data=gpu_data)
        result = reproject(raster, 'EPSG:32633')
        assert result.dims == ('band', 'y', 'x')
        computed = result.compute()
        assert computed.shape[0] == 3


# ---------------------------------------------------------------------------
# Issue #2187: bounds_policy parameter
# ---------------------------------------------------------------------------

class TestBoundsPolicy:
    """reproject(): bounds_policy controls the bounds-derivation heuristics.

    Without the policy knob, _compute_output_grid silently clamps
    geographic bounds and falls back to 2/98 percentile bounds when the
    projected extent blows up. These tests pin the four policy options:
    auto (default, current behaviour with warnings), raw (no heuristic),
    clamp (geographic clamp only), and percentile (force 2/98 fallback).
    """

    @staticmethod
    def _global_geographic():
        """Global lat/lon raster that triggers the polar / antimeridian
        blow-up when projected to Web Mercator."""
        data = np.random.RandomState(0).rand(50, 100).astype(np.float32)
        return xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(90, -90, 50),
                    'x': np.linspace(-180, 180, 100)},
            attrs={'crs': 'EPSG:4326'},
        )

    @staticmethod
    def _benign_geographic():
        """Mid-latitude raster well away from any singularity."""
        data = np.random.RandomState(0).rand(32, 32).astype(np.float32)
        return xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(55, 45, 32),
                    'x': np.linspace(-5, 5, 32)},
            attrs={'crs': 'EPSG:4326'},
        )

    def test_raw_skips_clamp_and_percentile(self):
        """bounds_policy='raw' returns un-cropped bounds for a blow-up case.

        A global geographic raster projected to Web Mercator hits the
        polar singularity. Under 'auto' the percentile fallback fires
        and crops the output extent. Under 'raw' the caller gets the
        true projected extent of the corners/edges.
        """
        from xrspatial.reproject import reproject

        r = self._global_geographic()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            auto_result = reproject(r, 'EPSG:3857', bounds_policy='auto')
            raw_result = reproject(r, 'EPSG:3857', bounds_policy='raw')

        auto_x = auto_result.coords['x'].values
        raw_x = raw_result.coords['x'].values
        auto_y = auto_result.coords['y'].values
        raw_y = raw_result.coords['y'].values

        auto_x_span = auto_x.max() - auto_x.min()
        raw_x_span = raw_x.max() - raw_x.min()
        auto_y_span = auto_y.max() - auto_y.min()
        raw_y_span = raw_y.max() - raw_y.min()

        # Raw should be at least as wide as auto on x, and meaningfully
        # taller on y (the polar blow-up is the y axis under EPSG:3857).
        assert raw_x_span >= auto_x_span - 1.0
        assert raw_y_span > auto_y_span * 1.1, (
            f"raw y span {raw_y_span} should exceed auto {auto_y_span}"
        )

    def test_percentile_reproduces_98_2_behaviour(self):
        """bounds_policy='percentile' matches the previous 2/98 fallback
        even on inputs that wouldn't trigger the blow-up heuristic."""
        from xrspatial.reproject import reproject
        from xrspatial.reproject._grid import _compute_output_grid
        from xrspatial.reproject._crs_utils import _resolve_crs

        r = self._benign_geographic()
        src_crs = _resolve_crs('EPSG:4326')
        tgt_crs = _resolve_crs('EPSG:3857')

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            grid_percentile = _compute_output_grid(
                (-5.0, 45.0, 5.0, 55.0), (32, 32),
                src_crs, tgt_crs, bounds_policy='percentile',
            )
            grid_raw = _compute_output_grid(
                (-5.0, 45.0, 5.0, 55.0), (32, 32),
                src_crs, tgt_crs, bounds_policy='raw',
            )

        # Percentile bounds should be strictly inside raw bounds (or
        # equal to floating-point precision) for a benign input.
        pl, pb, pr, pt = grid_percentile['bounds']
        rl, rb, rr, rt = grid_raw['bounds']
        assert pl >= rl - 1.0
        assert pr <= rr + 1.0
        assert pb >= rb - 1.0
        assert pt <= rt + 1.0

    def test_warns_when_percentile_fires_under_auto(self):
        """auto policy emits UserWarning when the 2/98 fallback triggers."""
        from xrspatial.reproject import reproject

        r = self._global_geographic()
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            reproject(r, 'EPSG:3857', bounds_policy='auto')

        matched = [
            wi for wi in w
            if issubclass(wi.category, UserWarning)
            and 'bounds_policy' in str(wi.message)
        ]
        assert matched, "expected a bounds_policy warning under auto"
        assert any('blow-up' in str(wi.message) or 'percentile' in str(wi.message)
                   for wi in matched)

    def test_warns_when_clamp_actually_trims(self):
        """clamp policy emits UserWarning when source bounds get trimmed."""
        from xrspatial.reproject._grid import _compute_output_grid
        from xrspatial.reproject._crs_utils import _resolve_crs

        src_crs = _resolve_crs('EPSG:4326')
        tgt_crs = _resolve_crs('EPSG:3857')

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _compute_output_grid(
                (-180.0, -90.0, 180.0, 90.0), (50, 100),
                src_crs, tgt_crs, bounds_policy='clamp',
            )

        matched = [
            wi for wi in w
            if issubclass(wi.category, UserWarning)
            and 'clamp' in str(wi.message)
        ]
        assert matched, "expected a clamp warning on full-globe input"

    def test_no_warning_on_benign_input(self):
        """auto policy stays silent on inputs that don't trigger heuristics.

        Same-units projections (UTM->UTM) have comparable spans so the
        blow-up ratio stays below the 50x threshold and the geographic
        clamp doesn't apply. No warning should fire.
        """
        from xrspatial.reproject import reproject

        data = np.random.RandomState(0).rand(32, 32).astype(np.float32)
        r = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(5000000, 4000000, 32),
                    'x': np.linspace(400000, 600000, 32)},
            attrs={'crs': 'EPSG:32633'},
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            reproject(r, 'EPSG:32632', bounds_policy='auto')

        matched = [
            wi for wi in w
            if issubclass(wi.category, UserWarning)
            and 'bounds_policy' in str(wi.message)
        ]
        assert not matched, (
            f"unexpected bounds_policy warning(s): {[str(m.message) for m in matched]}"
        )

    def test_invalid_policy_rejected(self):
        """Unknown bounds_policy tokens raise ValueError at the API boundary."""
        from xrspatial.reproject import reproject

        r = self._benign_geographic()
        with pytest.raises(ValueError, match=r"bounds_policy"):
            reproject(r, 'EPSG:3857', bounds_policy='bogus')

    def test_invalid_policy_rejected_in_merge(self):
        from xrspatial.reproject import merge

        r = self._benign_geographic()
        with pytest.raises(ValueError, match=r"bounds_policy"):
            merge([r], target_crs='EPSG:3857', bounds_policy='bogus')

    def test_explicit_bounds_skips_policy_logic(self):
        """When the caller passes bounds, the policy heuristics don't run."""
        from xrspatial.reproject import reproject

        r = self._global_geographic()
        # Mercator-y bounds chosen well inside the projection envelope.
        explicit = (-2.0e7, -2.0e7, 2.0e7, 2.0e7)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            out = reproject(
                r, 'EPSG:3857',
                bounds=explicit,
                resolution=2e5,
                bounds_policy='auto',
            )

        # No bounds_policy warning fires when bounds are explicit.
        matched = [
            wi for wi in w
            if issubclass(wi.category, UserWarning)
            and 'bounds_policy' in str(wi.message)
        ]
        assert not matched
        # Output extent reflects the explicit bounds (within one pixel).
        out_x = out.coords['x'].values
        out_y = out.coords['y'].values
        assert abs(out_x.min() - explicit[0]) < 2.5e5
        assert abs(out_x.max() - explicit[2]) < 2.5e5
        assert abs(out_y.min() - explicit[1]) < 2.5e5
        assert abs(out_y.max() - explicit[3]) < 2.5e5

    def test_merge_passes_policy_through(self):
        """merge() plumbs bounds_policy to _compute_output_grid."""
        from xrspatial.reproject import merge

        r = self._global_geographic()
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            merge([r], target_crs='EPSG:3857', bounds_policy='auto')

        matched = [
            wi for wi in w
            if issubclass(wi.category, UserWarning)
            and 'bounds_policy' in str(wi.message)
        ]
        # merge() on a single global geographic raster should also
        # trigger the percentile fallback warning under auto.
        assert matched

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_raw_policy_dask_backend(self):
        """bounds_policy='raw' works with a dask-backed input."""
        from xrspatial.reproject import reproject

        r = self._benign_geographic()
        r = r.chunk({'y': 16, 'x': 16})
        out = reproject(r, 'EPSG:3857', bounds_policy='raw')
        # Result is also dask-backed (lazy).
        assert hasattr(out.data, 'dask')
        # Compute and confirm we got finite output.
        arr = out.compute()
        assert np.isfinite(arr.data).any()

    def test_clamp_policy_noop_on_benign_geographic(self):
        """bounds_policy='clamp' is silent on a mid-latitude geographic
        input whose extent does not touch +/-180 or +/-90.

        The clamp branch runs but trims nothing, so no warning should
        fire. This pins the behaviour so a future change that always
        emits a clamp warning shows up here.
        """
        from xrspatial.reproject import reproject

        r = self._benign_geographic()
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            reproject(r, 'EPSG:3857', bounds_policy='clamp')

        matched = [
            wi for wi in w
            if issubclass(wi.category, UserWarning)
            and 'bounds_policy' in str(wi.message)
        ]
        assert not matched, (
            f"clamp on benign geographic input should not warn; got "
            f"{[str(m.message) for m in matched]}"
        )

    def test_clamp_policy_noop_on_projected_source(self):
        """bounds_policy='clamp' is a no-op when source CRS is projected.

        The clamp condition is gated on `source_crs.is_geographic`, so
        a UTM input under 'clamp' should run without trimming or
        warning regardless of how close to a singularity the extent is.
        """
        from xrspatial.reproject import reproject

        data = np.random.RandomState(0).rand(32, 32).astype(np.float32)
        # UTM-style coords, mid-latitudes.
        r = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(5000000, 4000000, 32),
                    'x': np.linspace(400000, 600000, 32)},
            attrs={'crs': 'EPSG:32633'},
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            reproject(r, 'EPSG:4326', bounds_policy='clamp')

        matched = [
            wi for wi in w
            if issubclass(wi.category, UserWarning)
            and 'bounds_policy' in str(wi.message)
        ]
        assert not matched

    def test_merge_dedupes_per_input_warnings(self):
        """merge() collapses per-input bounds_policy warnings into one.

        When several inputs all trigger the percentile fallback, the
        caller should see a single summary warning rather than N
        near-identical messages.
        """
        from xrspatial.reproject import merge

        r = self._global_geographic()
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            merge([r, r, r], target_crs='EPSG:3857', bounds_policy='auto')

        matched = [
            wi for wi in w
            if issubclass(wi.category, UserWarning)
            and 'bounds_policy' in str(wi.message)
        ]
        # Three identical inputs should yield exactly one summary
        # warning from merge(), not three.
        summary = [m for m in matched if 'merge:' in str(m.message)]
        assert len(summary) == 1, (
            f"expected one merge summary warning, got "
            f"{[str(m.message) for m in matched]}"
        )

    # -- Issue #2582: unit-aware blow-up heuristic --------------------

    def test_auto_does_not_crop_benign_geographic_to_mercator(self):
        """Regression for #2582.

        Reprojecting a small EPSG:4326 bbox to EPSG:3857 under
        bounds_policy='auto' must match bounds_policy='raw' to a
        small tolerance. The old span-ratio heuristic compared
        degrees to metres and always tripped on geographic-to-
        projected pairs, silently trimming tens of km per side.
        """
        from xrspatial.reproject import reproject

        data = np.random.RandomState(0).rand(64, 64).astype(np.float32)
        r = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(10, -10, 64),
                    'x': np.linspace(-10, 10, 64)},
            attrs={'crs': 'EPSG:4326'},
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            auto_r = reproject(r, 'EPSG:3857', bounds_policy='auto')
            raw_r = reproject(r, 'EPSG:3857', bounds_policy='raw')

        # Spans should match within one output pixel of res.
        ax = auto_r.coords['x'].values
        rx = raw_r.coords['x'].values
        ay = auto_r.coords['y'].values
        ry = raw_r.coords['y'].values

        res_x = (rx.max() - rx.min()) / max(1, len(rx) - 1)
        res_y = (ry.max() - ry.min()) / max(1, len(ry) - 1)

        # No more than one pixel of crop on each side (was ~106 km / ~70 km).
        assert abs((ax.max() - ax.min()) - (rx.max() - rx.min())) < 2 * res_x
        assert abs((ay.max() - ay.min()) - (ry.max() - ry.min())) < 2 * res_y

    def test_auto_silent_on_benign_geographic_to_mercator(self):
        """No bounds_policy warning fires for a benign 4326->3857 case (#2582)."""
        from xrspatial.reproject import reproject

        data = np.random.RandomState(0).rand(32, 32).astype(np.float32)
        r = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(10, -10, 32),
                    'x': np.linspace(-10, 10, 32)},
            attrs={'crs': 'EPSG:4326'},
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            reproject(r, 'EPSG:3857', bounds_policy='auto')
        matched = [
            wi for wi in w
            if issubclass(wi.category, UserWarning)
            and 'bounds_policy' in str(wi.message)
        ]
        assert not matched, (
            f"unexpected bounds_policy warning(s) on benign 4326->3857 "
            f"input: {[str(m.message) for m in matched]}"
        )

    def test_auto_still_trips_polar_stereographic_blowup(self):
        """Pathological 4326->polar-stereo case must still trip auto (#2582).

        A global EPSG:4326 raster projected to EPSG:3413 (NSIDC north
        polar stereographic) produces finite-but-astronomical
        coordinates near the south pole. The new unit-agnostic
        heuristic must still catch this and apply the percentile
        fallback.
        """
        from xrspatial.reproject._grid import _compute_output_grid
        from xrspatial.reproject._crs_utils import _resolve_crs

        src_crs = _resolve_crs('EPSG:4326')
        tgt_crs = _resolve_crs('EPSG:3413')

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            grid = _compute_output_grid(
                (-180.0, -90.0, 180.0, 90.0), (100, 200),
                src_crs, tgt_crs, bounds_policy='auto',
            )
        # The auto bounds should be reasonable Earth-scale (well under
        # 1e10 m), not the 1e23-scale raw projection.
        left, bottom, right, top = grid['bounds']
        assert abs(left) < 1e10 and abs(right) < 1e10
        assert abs(bottom) < 1e10 and abs(top) < 1e10
        # And the warning must fire.
        matched = [
            wi for wi in w
            if issubclass(wi.category, UserWarning)
            and 'bounds_policy' in str(wi.message)
            and ('blow-up' in str(wi.message) or 'percentile' in str(wi.message))
        ]
        assert matched, (
            "expected a blow-up/percentile warning under auto for "
            "global 4326 -> polar stereographic"
        )

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_auto_does_not_crop_benign_geographic_dask(self):
        """Dask-backed input also gets the unit-aware fix (#2582)."""
        from xrspatial.reproject import reproject

        data = np.random.RandomState(0).rand(64, 64).astype(np.float32)
        r = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(10, -10, 64),
                    'x': np.linspace(-10, 10, 64)},
            attrs={'crs': 'EPSG:4326'},
        ).chunk({'y': 32, 'x': 32})
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            auto_r = reproject(r, 'EPSG:3857', bounds_policy='auto')
            raw_r = reproject(r, 'EPSG:3857', bounds_policy='raw')
        ax = auto_r.coords['x'].values
        rx = raw_r.coords['x'].values
        res_x = (rx.max() - rx.min()) / max(1, len(rx) - 1)
        assert abs((ax.max() - ax.min()) - (rx.max() - rx.min())) < 2 * res_x


# ---------------------------------------------------------------------------
# Integer dtype nodata handling (#2185)
# ---------------------------------------------------------------------------

class TestIntegerNodataDefaults:
    """Default nodata picks for integer dtypes follow rasterio/GDAL.

    Signed integers get dtype.min (e.g. int16 -> -32768). Unsigned integers
    get dtype.max (e.g. uint16 -> 65535). Without this, the worker casts
    NaN back to the integer dtype and silently produces 0 for every
    out-of-bounds pixel while attrs['nodata'] still advertises NaN.
    """

    def test_default_integer_nodata_signed(self):
        from xrspatial.reproject._crs_utils import _default_integer_nodata
        assert _default_integer_nodata(np.int8) == -128.0
        assert _default_integer_nodata(np.int16) == -32768.0
        assert _default_integer_nodata(np.int32) == float(np.iinfo(np.int32).min)
        assert _default_integer_nodata(np.int64) == float(np.iinfo(np.int64).min)

    def test_default_integer_nodata_unsigned(self):
        from xrspatial.reproject._crs_utils import _default_integer_nodata
        assert _default_integer_nodata(np.uint8) == 255.0
        assert _default_integer_nodata(np.uint16) == 65535.0
        assert _default_integer_nodata(np.uint32) == float(np.iinfo(np.uint32).max)

    def test_detect_nodata_int_dtype_hint_returns_sentinel(self):
        """When dtype is integer and no nodata is set anywhere, use a sentinel."""
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(np.zeros((4, 4), dtype=np.int16), dims=('y', 'x'))
        assert _detect_nodata(r, dtype=np.int16) == -32768.0

    def test_detect_nodata_float_dtype_hint_still_nan(self):
        """Float dtype keeps the historical NaN default."""
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(np.zeros((4, 4), dtype=np.float32), dims=('y', 'x'))
        nd = _detect_nodata(r, dtype=np.float32)
        assert np.isnan(nd)

    def test_detect_nodata_dtype_hint_does_not_override_explicit(self):
        """Explicit nodata wins over the dtype-based default."""
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(np.zeros((4, 4), dtype=np.int16), dims=('y', 'x'))
        assert _detect_nodata(r, nodata=-1, dtype=np.int16) == -1.0

    def test_detect_nodata_dtype_hint_does_not_override_attrs(self):
        """attrs['_FillValue'] etc. still win over the dtype-based default."""
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(
            np.zeros((4, 4), dtype=np.int16),
            dims=('y', 'x'),
            attrs={'_FillValue': -1},
        )
        assert _detect_nodata(r, dtype=np.int16) == -1.0

    def test_detect_nodata_swaps_nan_attrs_for_int_sentinel(self):
        """Explicit NaN in attrs gets swapped for an int sentinel.

        Some workflows write ``attrs['nodata'] = nan`` even on integer
        rasters (e.g. when generated by code that targets float
        outputs). Returning NaN from _detect_nodata would put us right
        back in the #2185 corruption path, so the dtype-aware swap has
        to apply post-resolution, not just at the absent-upstream tail.
        """
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(
            np.zeros((4, 4), dtype=np.int16),
            dims=('y', 'x'),
            attrs={'nodata': float('nan')},
        )
        assert _detect_nodata(r, dtype=np.int16) == -32768.0

    def test_detect_nodata_swaps_explicit_nan_arg_for_int_sentinel(self):
        """Explicit NaN passed as nodata= also gets swapped for int dtypes."""
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(np.zeros((4, 4), dtype=np.int16), dims=('y', 'x'))
        assert _detect_nodata(r, nodata=float('nan'), dtype=np.int16) == -32768.0

    def test_detect_nodata_float_dtype_keeps_nan_attrs(self):
        """Float rasters keep NaN attrs as-is."""
        from xrspatial.reproject._crs_utils import _detect_nodata
        r = xr.DataArray(
            np.zeros((4, 4), dtype=np.float32),
            dims=('y', 'x'),
            attrs={'nodata': float('nan')},
        )
        nd = _detect_nodata(r, dtype=np.float32)
        assert np.isnan(nd)


def _int_raster_with_oob(dtype, fill_value=100):
    """Build an int raster that produces out-of-bounds pixels in EPSG:32633.

    Source is high-latitude WGS84 (y in [50, 60], x in [-10, 10]) so
    reprojecting to UTM zone 33N leaves rotated corners outside the
    source footprint. Those corners are the pixels the bug surfaces on.

    The OOB count depends on the specific source bounds and target CRS
    chosen here. Callers that swap in a different target CRS need to
    re-check that OOB pixels actually exist in the output -- the
    assertions in the integer-nodata tests rely on this combination.
    """
    h, w = 64, 64
    data = np.full((h, w), fill_value, dtype=dtype)
    y = np.linspace(60.0, 50.0, h)
    x = np.linspace(-10.0, 10.0, w)
    return xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 'EPSG:4326'},
    )


@pytest.mark.parametrize("dtype, expected_sentinel", [
    (np.int8, -128),
    (np.int16, -32768),
    (np.int32, np.iinfo(np.int32).min),
    (np.uint8, 255),
    (np.uint16, 65535),
])
class TestReprojectIntegerNodataNumpyParametrized:
    """End-to-end: numpy backend, int dtypes, no user-supplied nodata.

    The OOB pixels in the output must equal attrs['nodata'] exactly, and
    they must not silently become 0 (the pre-#2185 behaviour).

    Class-level ``@pytest.mark.parametrize`` applies to every method
    added here. Tests that do not need to fan out across all dtypes
    belong in a different class -- otherwise they will run N times for
    no reason.
    """

    def test_oob_pixels_match_attrs_nodata(self, dtype, expected_sentinel):
        from xrspatial.reproject import reproject

        # Pick a fill value that isn't equal to the sentinel.
        fill = 100 if expected_sentinel != 100 else 50
        raster = _int_raster_with_oob(dtype, fill_value=fill)
        result = reproject(raster, 'EPSG:32633')

        assert result.dtype == np.dtype(dtype)
        assert result.attrs['nodata'] == float(expected_sentinel)

        # There must actually be some OOB pixels in this test setup --
        # otherwise the assertion below would pass trivially.
        oob_mask = result.values == expected_sentinel
        assert oob_mask.any(), (
            f"test setup produced no OOB pixels for dtype={dtype}; "
            f"adjust the raster bounds"
        )

        # The pre-fix behaviour collapsed OOB pixels to 0 for signed
        # ints. Make sure that didn't happen here.
        if expected_sentinel != 0:
            assert not ((result.values == 0) & oob_mask).any()

        # Everything that isn't OOB should be the fill value.
        valid = ~oob_mask
        assert (result.values[valid] == fill).all()


class TestReprojectIntegerNodataExplicit:
    """User-supplied nodata still wins for integer rasters."""

    def test_explicit_nodata_used(self):
        from xrspatial.reproject import reproject

        raster = _int_raster_with_oob(np.int16, fill_value=100)
        result = reproject(raster, 'EPSG:32633', nodata=-1)
        assert result.attrs['nodata'] == -1.0
        assert (result.values == -1).any()
        # Default sentinel should not appear in the output now.
        assert not (result.values == -32768).any()


class TestReprojectIntegerNodataDask:
    """Same guarantee on the dask+numpy backend."""

    def test_dask_oob_pixels_match_attrs_nodata(self):
        if not HAS_DASK:
            pytest.skip("dask required")
        from xrspatial.reproject import reproject

        raster = _int_raster_with_oob(np.int16, fill_value=100)
        # Wrap in dask.
        dask_data = da.from_array(raster.values, chunks=(32, 32))
        dask_raster = xr.DataArray(
            dask_data, dims=raster.dims, coords=raster.coords,
            attrs=raster.attrs,
        )

        result = reproject(dask_raster, 'EPSG:32633')
        computed = result.compute() if hasattr(result, 'compute') else result
        assert computed.dtype == np.int16
        assert computed.attrs['nodata'] == -32768.0
        assert (computed.values == -32768).any()
        assert not (computed.values == 0).any()


class TestReprojectIntegerNodataCupy:
    """Same guarantee on the cupy backend."""

    def test_cupy_oob_pixels_match_attrs_nodata(self):
        if not HAS_CUPY:
            pytest.skip("cupy required")
        from xrspatial.reproject import reproject

        raster = _int_raster_with_oob(np.int16, fill_value=100)
        cupy_data = cp.asarray(raster.values)
        cupy_raster = xr.DataArray(
            cupy_data, dims=raster.dims, coords=raster.coords,
            attrs=raster.attrs,
        )

        result = reproject(cupy_raster, 'EPSG:32633')
        host = cp.asnumpy(result.data) if isinstance(result.data, cp.ndarray) else result.values
        assert result.dtype == np.int16
        assert result.attrs['nodata'] == -32768.0
        assert (host == -32768).any()
        assert not (host == 0).any()


class TestReprojectIntegerNodataDaskCupy:
    """Same guarantee on the dask+cupy backend."""

    def test_dask_cupy_oob_pixels_match_attrs_nodata(self):
        if not (HAS_DASK and HAS_CUPY):
            pytest.skip("dask and cupy required")
        from xrspatial.reproject import reproject

        raster = _int_raster_with_oob(np.int16, fill_value=100)
        cupy_data = cp.asarray(raster.values)
        dask_cupy_data = da.from_array(cupy_data, chunks=(32, 32))
        dask_cupy_raster = xr.DataArray(
            dask_cupy_data, dims=raster.dims, coords=raster.coords,
            attrs=raster.attrs,
        )

        result = reproject(dask_cupy_raster, 'EPSG:32633')
        computed = result.compute() if hasattr(result, 'compute') else result
        host = (
            cp.asnumpy(computed.data)
            if isinstance(computed.data, cp.ndarray)
            else np.asarray(computed.values)
        )
        assert computed.attrs['nodata'] == -32768.0
        assert (host == -32768).any()
        assert not (host == 0).any()


class TestReprojectIntegerNodataRegression:
    """The exact #2185 reproduction case must not regress."""

    def test_no_runtime_warning_and_no_zero_corruption(self):
        import warnings
        from xrspatial.reproject import reproject

        data = np.full((64, 64), 100, dtype=np.int16)
        y = np.linspace(60.0, 50.0, 64)
        x = np.linspace(-10.0, 10.0, 64)
        raster = xr.DataArray(
            data, dims=['y', 'x'], coords={'y': y, 'x': x},
            attrs={'crs': 'EPSG:4326'},
        )

        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            result = reproject(raster, 'EPSG:32633')

        assert result.attrs['nodata'] == -32768.0
        # Pre-fix output had 435 stealth 0-pixels marked as real data.
        # After the fix, every non-fill pixel must be the sentinel.
        unique = set(np.unique(result.values).tolist())
        assert unique == {100, -32768}


# ---------------------------------------------------------------------------
# transform_precision=0 forces the exact pyproj path (#2646)
# ---------------------------------------------------------------------------

class TestExactPrecisionEscapeHatch:
    """transform_precision=0 must bypass the Numba/CUDA fast path entirely
    and transform every pixel through pyproj, as the docstring promises.

    The bug: the fast path was tried before checking transform_precision == 0,
    so for CRS pairs the Numba path supports (WGS84/NAD83 <-> UTM, WGS84 <->
    Web Mercator) the escape hatch did nothing.
    """

    # A 4326 <-> 3857 chunk; the Numba fast path supports this pair, so the
    # pre-fix code never reached the pyproj branch for transform_precision=0.
    _BOUNDS = (-1_000_000.0, -1_000_000.0, 1_000_000.0, 1_000_000.0)
    _SHAPE = (37, 53)  # non-square, odd dims to catch axis/reshape mistakes

    def _pyproj_exact_reference(self):
        """Per-pixel source coords computed straight from pyproj.

        Output bounds are in the target CRS (3857); the transform maps each
        output pixel center back to the source CRS (4326), matching how
        ``_transform_coords`` is invoked (target -> source).
        """
        transformer = pyproj.Transformer.from_crs(
            'EPSG:3857', 'EPSG:4326', always_xy=True
        )
        height, width = self._SHAPE
        left, bottom, right, top = self._BOUNDS
        res_x = (right - left) / width
        res_y = (top - bottom) / height
        out_x = left + (np.arange(width, dtype=np.float64) + 0.5) * res_x
        out_y = top - (np.arange(height, dtype=np.float64) + 0.5) * res_y
        xx, yy = np.meshgrid(out_x, out_y)
        sx, sy = transformer.transform(xx.ravel(), yy.ravel())
        return (
            np.asarray(sy, dtype=np.float64).reshape(self._SHAPE),
            np.asarray(sx, dtype=np.float64).reshape(self._SHAPE),
        )

    def test_numba_fast_path_active_for_this_pair(self):
        """Sanity check: the Numba fast path really does fire for 4326<->3857,
        otherwise the regression below would pass vacuously."""
        from xrspatial.reproject._projections import try_numba_transform
        src = pyproj.CRS('EPSG:4326')
        tgt = pyproj.CRS('EPSG:3857')
        result = try_numba_transform(src, tgt, self._BOUNDS, self._SHAPE)
        assert result is not None

    def test_transform_coords_precision_zero_matches_pyproj(self):
        from xrspatial.reproject import _transform_coords
        src = pyproj.CRS('EPSG:4326')
        tgt = pyproj.CRS('EPSG:3857')
        transformer = pyproj.Transformer.from_crs(
            tgt, src, always_xy=True
        )
        src_y, src_x = _transform_coords(
            transformer, self._BOUNDS, self._SHAPE, 0,
            src_crs=src, tgt_crs=tgt,
        )
        ref_y, ref_x = self._pyproj_exact_reference()
        # Exact path: should match pyproj to floating-point precision.
        np.testing.assert_allclose(src_x, ref_x, rtol=0, atol=1e-6)
        np.testing.assert_allclose(src_y, ref_y, rtol=0, atol=1e-6)

    def test_transform_coords_precision_zero_skips_numba(self, monkeypatch):
        """transform_precision=0 must not call try_numba_transform."""
        from xrspatial.reproject import _transform_coords
        from xrspatial.reproject import _projections

        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            raise AssertionError(
                "try_numba_transform called with transform_precision=0"
            )

        monkeypatch.setattr(_projections, 'try_numba_transform', _spy)

        src = pyproj.CRS('EPSG:4326')
        tgt = pyproj.CRS('EPSG:3857')
        transformer = pyproj.Transformer.from_crs(tgt, src, always_xy=True)
        # Must not raise: the Numba path is never entered.
        _transform_coords(
            transformer, self._BOUNDS, self._SHAPE, 0,
            src_crs=src, tgt_crs=tgt,
        )
        assert calls == []

    def test_reproject_chunk_numpy_precision_zero_skips_numba(self, monkeypatch):
        """The numpy chunk worker honors the escape hatch too."""
        from xrspatial.reproject import _reproject_chunk_numpy
        from xrspatial.reproject import _projections

        def _spy(*args, **kwargs):
            raise AssertionError(
                "try_numba_transform called with transform_precision=0"
            )

        monkeypatch.setattr(_projections, 'try_numba_transform', _spy)

        src_wkt = pyproj.CRS('EPSG:4326').to_wkt()
        tgt_wkt = pyproj.CRS('EPSG:3857').to_wkt()
        source_data = np.arange(32 * 32, dtype=np.float64).reshape(32, 32)
        # Must not raise: with transform_precision=0 the worker takes the
        # pyproj branch instead of the Numba fast path.
        out = _reproject_chunk_numpy(
            source_data,
            (-2_000_000.0, -2_000_000.0, 2_000_000.0, 2_000_000.0),
            (32, 32), True,
            src_wkt, tgt_wkt,
            self._BOUNDS, self._SHAPE,
            'nearest', np.nan, 0,
        )
        assert out.shape == self._SHAPE

    def test_reproject_end_to_end_precision_zero_skips_numba(self, monkeypatch):
        """reproject() with transform_precision=0 routes through pyproj and
        still produces a sensible result."""
        from xrspatial.reproject import reproject
        from xrspatial.reproject import _projections

        def _spy(*args, **kwargs):
            raise AssertionError(
                "try_numba_transform called with transform_precision=0"
            )

        monkeypatch.setattr(_projections, 'try_numba_transform', _spy)

        raster = _gradient_raster(
            h=32, w=32, x_range=(-10, 10), y_range=(-10, 10)
        )
        result = reproject(raster, 'EPSG:3857', transform_precision=0)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_reproject_precision_zero_matches_default_for_smooth_pair(self):
        """For a smooth, well-behaved pair the exact path and the default
        approximate path should agree closely on overlapping pixels."""
        from xrspatial.reproject import reproject
        raster = _gradient_raster(
            h=48, w=48, x_range=(-8, 8), y_range=(-8, 8)
        )
        exact = reproject(
            raster, 'EPSG:3857', resolution=200000.0, transform_precision=0
        )
        approx = reproject(
            raster, 'EPSG:3857', resolution=200000.0
        )
        assert exact.shape == approx.shape
        a = exact.values
        b = approx.values
        both = np.isfinite(a) & np.isfinite(b)
        assert both.any()
        np.testing.assert_allclose(a[both], b[both], rtol=0, atol=1e-3)


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
class TestNoDuplicateNumbaFastPathProbe:
    """The numpy chunk worker must probe the numba fast path exactly once
    per chunk (#3106).

    The bug: for CRS pairs with no fast path, _reproject_chunk_numpy
    called try_numba_transform (None), then fell into _transform_coords
    which called try_numba_transform again before the pyproj control
    grid. Each wasted probe re-parses CRS params and allocates four
    chunk-sized coordinate arrays.
    """

    _BOUNDS = (-2_000_000.0, 4_000_000.0, -1_000_000.0, 5_000_000.0)
    _SHAPE = (16, 16)

    @staticmethod
    def _wkts():
        # WGS84 -> Mollweide has no numba fast path, so the worker takes
        # the pyproj fallback where the duplicate probe used to happen.
        return (pyproj.CRS('EPSG:4326').to_wkt(),
                pyproj.CRS('ESRI:54009').to_wkt())

    def test_chunk_numpy_probes_fast_path_exactly_once(self, monkeypatch):
        from xrspatial.reproject import _reproject_chunk_numpy
        from xrspatial.reproject import _projections

        calls = []
        real = _projections.try_numba_transform

        def _spy(*args, **kwargs):
            calls.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(_projections, 'try_numba_transform', _spy)

        src_wkt, tgt_wkt = self._wkts()
        source_data = np.arange(32 * 32, dtype=np.float64).reshape(32, 32)
        out = _reproject_chunk_numpy(
            source_data,
            (-20.0, 35.0, -10.0, 45.0), (32, 32), True,
            src_wkt, tgt_wkt,
            self._BOUNDS, self._SHAPE,
            'bilinear', np.nan, 16,
        )
        assert out.shape == self._SHAPE
        assert len(calls) == 1, (
            f"expected one try_numba_transform probe per chunk, "
            f"got {len(calls)} (#3106)"
        )

    def test_transform_coords_still_probes_when_given_crs(self, monkeypatch):
        """_transform_coords keeps its own probe for callers that have not
        tried the numba path yet (the cupy CPU fallbacks rely on it)."""
        from xrspatial.reproject import _transform_coords
        from xrspatial.reproject import _projections

        calls = []
        real = _projections.try_numba_transform

        def _spy(*args, **kwargs):
            calls.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(_projections, 'try_numba_transform', _spy)

        src = pyproj.CRS('EPSG:4326')
        tgt = pyproj.CRS('EPSG:3857')
        transformer = pyproj.Transformer.from_crs(tgt, src, always_xy=True)
        _transform_coords(
            transformer, self._BOUNDS, self._SHAPE, 16,
            src_crs=src, tgt_crs=tgt,
        )
        assert len(calls) == 1

    def test_fallback_pair_values_match_pyproj_reference(self):
        """The skipped retry must not change the worker's coordinates:
        the pyproj fallback output stays identical for a no-fast-path
        pair (exact path, so it is directly comparable to pyproj)."""
        from xrspatial.reproject import _transform_coords

        src = pyproj.CRS('EPSG:4326')
        tgt = pyproj.CRS('ESRI:54009')
        transformer = pyproj.Transformer.from_crs(tgt, src, always_xy=True)
        src_y, src_x = _transform_coords(
            transformer, self._BOUNDS, self._SHAPE, 0,
        )

        h, w = self._SHAPE
        left, bottom, right, top = self._BOUNDS
        res_x = (right - left) / w
        res_y = (top - bottom) / h
        out_x = left + (np.arange(w) + 0.5) * res_x
        out_y = top - (np.arange(h) + 0.5) * res_y
        gx, gy = np.meshgrid(out_x, out_y)
        ref_x, ref_y = transformer.transform(gx.ravel(), gy.ravel())
        np.testing.assert_allclose(
            src_x, np.asarray(ref_x).reshape(h, w), atol=1e-9)
        np.testing.assert_allclose(
            src_y, np.asarray(ref_y).reshape(h, w), atol=1e-9)


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
class TestNonWgsDatumNumbaFastPath:
    """The numba fast path must not corrupt coordinates for non-WGS84
    datums (GH #2651).

    The projection kernels run in WGS84. The old datum-shift wrapper
    applied a degree-based Helmert shift to the kernel output, which is
    wrong whenever the source is a projected CRS (the output is
    easting/northing in metres) and ignored a non-WGS84 target datum
    entirely. The fix disables the numba fast path for any non-WGS84
    datum so pyproj handles those transforms.
    """

    def test_fast_path_disabled_for_projected_non_wgs_source(self):
        # OSGB36 / British National Grid (Airy datum), projected.
        from xrspatial.reproject._projections import try_numba_transform
        src = pyproj.CRS('EPSG:27700')
        tgt = pyproj.CRS('EPSG:4326')
        # Output chunk in WGS84 lon/lat over Great Britain.
        result = try_numba_transform(src, tgt, (-2.0, 51.0, -1.0, 52.0), (4, 4))
        assert result is None

    def test_fast_path_disabled_for_geographic_non_wgs_source(self):
        # NAD27 geographic (Clarke 1866 datum).
        from xrspatial.reproject._projections import try_numba_transform
        src = pyproj.CRS('EPSG:4267')
        tgt = pyproj.CRS('EPSG:3857')
        result = try_numba_transform(
            src, tgt, (-8000000.0, 4000000.0, -7900000.0, 4100000.0), (4, 4),
        )
        assert result is None

    def test_fast_path_disabled_for_non_wgs_target(self):
        from xrspatial.reproject._projections import try_numba_transform
        src = pyproj.CRS('EPSG:4326')
        tgt = pyproj.CRS('EPSG:27700')
        result = try_numba_transform(src, tgt, (400000.0, 200000.0, 410000.0, 210000.0), (4, 4))
        assert result is None

    def test_wgs_fast_path_still_active(self):
        # WGS84 UTM <-> WGS84 geographic must keep using the fast path.
        from xrspatial.reproject._projections import try_numba_transform
        src = pyproj.CRS('EPSG:32617')
        tgt = pyproj.CRS('EPSG:4326')
        result = try_numba_transform(src, tgt, (-84.0, 40.0, -83.0, 41.0), (4, 4))
        assert result is not None

    def test_source_coords_match_pyproj_for_osgb36(self):
        # End-to-end through _transform_coords: with the fast path
        # disabled, the per-pixel source coordinates must match pyproj.
        # Before the fix, the numba path returned easting ~5 where
        # pyproj returns ~408701 metres -- a corrupt grid.
        from xrspatial.reproject import _transform_coords
        src = pyproj.CRS('EPSG:27700')
        tgt = pyproj.CRS('EPSG:4326')
        chunk_bounds = (-2.0, 51.0, -1.0, 52.0)
        chunk_shape = (8, 8)

        transformer = pyproj.Transformer.from_crs(tgt, src, always_xy=True)
        src_y, src_x = _transform_coords(
            transformer, chunk_bounds, chunk_shape, 0,
            src_crs=src, tgt_crs=tgt,
        )

        # Reference: transform every output-pixel centre with pyproj.
        h, w = chunk_shape
        left, bottom, right, top = chunk_bounds
        res_x = (right - left) / w
        res_y = (top - bottom) / h
        out_x = left + (np.arange(w) + 0.5) * res_x
        out_y = top - (np.arange(h) + 0.5) * res_y
        gx, gy = np.meshgrid(out_x, out_y)
        ref_x, ref_y = transformer.transform(gx.ravel(), gy.ravel())
        ref_x = np.asarray(ref_x).reshape(h, w)
        ref_y = np.asarray(ref_y).reshape(h, w)

        # Eastings/northings are ~4e5 metres; require mm-level agreement.
        np.testing.assert_allclose(src_x, ref_x, atol=1e-3)
        np.testing.assert_allclose(src_y, ref_y, atol=1e-3)
        # Guard against the old corruption: coords must be metres, not degrees.
        assert np.all(np.abs(src_x) > 1000.0)
        assert np.all(np.abs(src_y) > 1000.0)


class TestVerticalReturnTypes:
    """Pin the return types the _vertical.py docstrings describe (#3097).

    The Returns sections used to claim "same type as input", which was
    wrong for DataArray input (plain ndarray comes back) and for scalar
    input to the conversion wrappers (numpy scalar, not Python float).
    These tests pin the actual behaviour the docs now state.
    """

    def test_geoid_height_scalar_returns_python_float(self):
        from xrspatial.reproject import geoid_height
        out = geoid_height(-74.0, 40.7)
        assert type(out) is float

    def test_geoid_height_array_returns_ndarray(self):
        from xrspatial.reproject import geoid_height
        out = geoid_height(np.array([-74.0, 0.0]), np.array([40.7, 0.0]))
        assert type(out) is np.ndarray
        assert out.shape == (2,)

    def test_geoid_height_dataarray_returns_ndarray(self):
        from xrspatial.reproject import geoid_height
        lon = xr.DataArray(np.array([-74.0, 0.0]))
        lat = xr.DataArray(np.array([40.7, 0.0]))
        out = geoid_height(lon, lat)
        # Documented: DataArray input comes back as a plain ndarray.
        assert type(out) is np.ndarray

    def test_conversion_wrappers_return_numpy_types(self):
        from xrspatial.reproject import (
            depth_to_ellipsoidal,
            ellipsoidal_to_depth,
            ellipsoidal_to_orthometric,
            orthometric_to_ellipsoidal,
        )
        for func in (ellipsoidal_to_orthometric, orthometric_to_ellipsoidal,
                     depth_to_ellipsoidal, ellipsoidal_to_depth):
            scalar_out = func(100.0, -74.0, 40.7)
            assert isinstance(scalar_out, np.floating), func.__name__
            arr_out = func(np.array([100.0, 50.0]),
                           np.array([-74.0, 0.0]), np.array([40.7, 0.0]))
            assert type(arr_out) is np.ndarray, func.__name__
            assert arr_out.shape == (2,), func.__name__


@pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
class TestMergeCupyBackends:
    """merge() accepts GPU-backed inputs and returns a GPU mosaic (#3095).

    Before the fix, _merge_inmemory called ``.values`` on cupy-backed
    DataArrays and xarray raised ``TypeError: Implicit conversion to a
    NumPy array is not allowed``. The merge runs on the host; these
    tests pin the round-trip and exact value parity with the numpy path.
    """

    def _pair(self):
        rng = np.random.default_rng(3095)
        data_a = rng.random((16, 16)).astype(np.float32)
        data_b = rng.random((16, 16)).astype(np.float32)
        a = _make_raster(data_a, x_range=(-10, 0), y_range=(-5, 5))
        b = _make_raster(data_b, x_range=(0, 10), y_range=(-5, 5))
        return a, b

    def test_cupy_inputs_return_cupy(self):
        import cupy as cp

        from xrspatial.reproject import merge
        a, b = self._pair()
        expected = merge([a, b], resolution=1.0)
        a_gpu = a.copy(data=cp.asarray(a.data))
        b_gpu = b.copy(data=cp.asarray(b.data))
        result = merge([a_gpu, b_gpu], resolution=1.0)
        assert isinstance(result.data, cp.ndarray)
        np.testing.assert_array_equal(
            cp.asnumpy(result.data), expected.values
        )
        assert result.dims == expected.dims
        np.testing.assert_array_equal(
            result.coords['x'].values, expected.coords['x'].values
        )

    def test_mixed_numpy_cupy_inputs_return_cupy(self):
        import cupy as cp

        from xrspatial.reproject import merge
        a, b = self._pair()
        expected = merge([a, b], resolution=1.0)
        b_gpu = b.copy(data=cp.asarray(b.data))
        result = merge([a, b_gpu], resolution=1.0)
        assert isinstance(result.data, cp.ndarray)
        np.testing.assert_array_equal(
            cp.asnumpy(result.data), expected.values
        )

    def test_cupy_inputs_do_not_mutate_sources(self):
        import cupy as cp

        from xrspatial.reproject import merge
        a, b = self._pair()
        a_gpu = a.copy(data=cp.asarray(a.data))
        b_gpu = b.copy(data=cp.asarray(b.data))
        merge([a_gpu, b_gpu], resolution=1.0)
        # Inputs stay on the GPU; the host conversion works on copies.
        assert isinstance(a_gpu.data, cp.ndarray)
        assert isinstance(b_gpu.data, cp.ndarray)
        # And the values are untouched (no in-place mutation).
        np.testing.assert_array_equal(cp.asnumpy(a_gpu.data), a.data)
        np.testing.assert_array_equal(cp.asnumpy(b_gpu.data), b.data)

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_dask_cupy_inputs_stay_lazy_and_match_numpy(self):
        import cupy as cp
        import dask.array as dask_array

        from xrspatial.reproject import merge
        a, b = self._pair()
        expected = merge([a, b], resolution=1.0)
        a_gpu = a.copy(data=dask_array.from_array(
            cp.asarray(a.data), chunks=8))
        b_gpu = b.copy(data=dask_array.from_array(
            cp.asarray(b.data), chunks=8))
        result = merge([a_gpu, b_gpu], resolution=1.0)
        # Graph construction must not materialize anything.
        assert isinstance(result.data, dask_array.Array)
        assert isinstance(result.data._meta, cp.ndarray)
        computed = result.data.compute()
        assert isinstance(computed, cp.ndarray)
        np.testing.assert_array_equal(
            cp.asnumpy(computed), expected.values
        )

    def test_cupy_merge_strategies_match_numpy(self):
        import cupy as cp

        from xrspatial.reproject import merge
        rng = np.random.default_rng(30952)
        a = _make_raster(rng.random((16, 16)).astype(np.float32),
                         x_range=(-5, 5), y_range=(-5, 5))
        b = _make_raster(rng.random((16, 16)).astype(np.float32),
                         x_range=(-5, 5), y_range=(-5, 5))
        a_gpu = a.copy(data=cp.asarray(a.data))
        b_gpu = b.copy(data=cp.asarray(b.data))
        for strategy in ('first', 'last', 'mean', 'max', 'min'):
            expected = merge([a, b], strategy=strategy, resolution=1.0)
            result = merge([a_gpu, b_gpu], strategy=strategy, resolution=1.0)
            np.testing.assert_array_equal(
                cp.asnumpy(result.data), expected.values,
                err_msg=f"strategy={strategy!r}",
            )


@pytest.mark.skipif(not HAS_CUPY, reason="cupy required")
class TestNonWgsDatumCudaFastPath:
    """The CUDA fast path must bail for non-WGS84 datums (GH #3094).

    GH #2651 gated the CPU fast paths so non-WGS84 datums fall back to
    pyproj, but try_cuda_transform kept dispatching them. The projected
    CRS matchers accept any datum in the Helmert table, so a pair like
    EPSG:4326 <-> EPSG:27700 (OSGB36 / Airy) ran the WGS84 Krueger
    series with no datum shift and returned coordinates ~100 m off,
    making the cupy and dask+cupy backends diverge from numpy.
    """

    def test_cuda_fast_path_disabled_for_non_wgs_target(self):
        from xrspatial.reproject._projections_cuda import try_cuda_transform
        src = pyproj.CRS('EPSG:4326')
        tgt = pyproj.CRS('EPSG:27700')
        result = try_cuda_transform(
            src, tgt, (400000.0, 200000.0, 410000.0, 210000.0), (4, 4),
        )
        assert result is None

    def test_cuda_fast_path_disabled_for_projected_non_wgs_source(self):
        from xrspatial.reproject._projections_cuda import try_cuda_transform
        src = pyproj.CRS('EPSG:27700')
        tgt = pyproj.CRS('EPSG:4326')
        result = try_cuda_transform(src, tgt, (-2.0, 51.0, -1.0, 52.0), (4, 4))
        assert result is None

    def test_cuda_fast_path_disabled_for_geographic_non_wgs_source(self):
        # NAD27 geographic (Clarke 1866 datum) -> Web Mercator.
        from xrspatial.reproject._projections_cuda import try_cuda_transform
        src = pyproj.CRS('EPSG:4267')
        tgt = pyproj.CRS('EPSG:3857')
        result = try_cuda_transform(
            src, tgt, (-8000000.0, 4000000.0, -7900000.0, 4100000.0), (4, 4),
        )
        assert result is None

    def test_cuda_wgs_fast_path_still_active(self):
        # WGS84 UTM <-> WGS84 geographic must keep using the CUDA path.
        from xrspatial.reproject._projections_cuda import try_cuda_transform
        src = pyproj.CRS('EPSG:32617')
        tgt = pyproj.CRS('EPSG:4326')
        result = try_cuda_transform(src, tgt, (-84.0, 40.0, -83.0, 41.0), (4, 4))
        assert result is not None

    def test_cupy_reproject_matches_numpy_for_osgb36_target(self):
        # End to end: cupy must agree with numpy for a non-WGS84 datum
        # target. Before the fix the cupy backend sampled ~10 pixels away
        # from the right source location (~100 m datum/ellipsoid error).
        from xrspatial.reproject import reproject
        rng = np.random.default_rng(3094)
        data = rng.random((64, 64))
        coords = {'y': np.linspace(52.0, 51.0, 64),
                  'x': np.linspace(-2.0, -1.0, 64)}
        host = xr.DataArray(
            data, dims=['y', 'x'], coords=coords,
            attrs={'crs': 'EPSG:4326'},
        )
        eager = reproject(host, 'EPSG:27700').values
        gpu = host.copy(data=cp.asarray(data))
        gpu_out = reproject(gpu, 'EPSG:27700')
        gpu_arr = cp.asnumpy(gpu_out.data)
        assert eager.shape == gpu_arr.shape
        # NaN masks must agree cell for cell.
        np.testing.assert_array_equal(
            np.isfinite(eager), np.isfinite(gpu_arr),
        )
        np.testing.assert_allclose(
            eager, gpu_arr, rtol=1e-4, atol=1e-4, equal_nan=True,
        )


# ---------------------------------------------------------------------------
# merge() integer dtype round-trip (#3262)
# ---------------------------------------------------------------------------

class TestMergeIntegerDtype:
    """merge() casts back to the shared integer input dtype (#3262).

    Matches the reproject() convention (#2505/#3093): integer sources
    round-trip back to their dtype after the float64 merge; mixed or
    float inputs keep returning float64.
    """

    @staticmethod
    def _tile_3262(x0, dtype=np.int16, fill=None, attrs=None):
        if fill is None:
            data = np.arange(64, dtype=dtype).reshape(8, 8)
        else:
            data = np.full((8, 8), fill, dtype=dtype)
        base_attrs = {'crs': 'EPSG:4326'}
        if attrs:
            base_attrs.update(attrs)
        return xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(5, -5, 8),
                    'x': np.linspace(x0, x0 + 10, 8)},
            attrs=base_attrs,
        )

    def test_merge_int16_preserves_dtype(self):
        from xrspatial.reproject import merge
        result = merge([self._tile_3262(-5), self._tile_3262(5)])
        assert result.dtype == np.int16

    def test_merge_uint8_preserves_dtype(self):
        from xrspatial.reproject import merge
        result = merge([self._tile_3262(-5, np.uint8),
                        self._tile_3262(5, np.uint8)])
        assert result.dtype == np.uint8

    def test_merge_int16_same_crs_values_exact(self):
        from xrspatial.reproject import merge
        tile = self._tile_3262(-5)
        result = merge([tile, self._tile_3262(5)])
        # Same-CRS direct placement must not perturb integer values.
        np.testing.assert_array_equal(
            result.values[:, :8], tile.values,
        )

    def test_merge_int16_default_nodata_sentinel(self):
        from xrspatial.reproject import merge
        # Tiles with a gap between them: the gap must hold the int16
        # default sentinel, not 0 (NaN collapsed by the cast) -- the
        # same hazard reproject fixed in #2185.
        result = merge([self._tile_3262(-20), self._tile_3262(5)])
        assert result.dtype == np.int16
        assert result.attrs['nodata'] == np.iinfo(np.int16).min
        assert (result.values == np.iinfo(np.int16).min).any()

    def test_merge_uint16_default_nodata_sentinel_is_max(self):
        from xrspatial.reproject import merge
        result = merge([self._tile_3262(-20, np.uint16),
                        self._tile_3262(5, np.uint16)])
        assert result.dtype == np.uint16
        assert result.attrs['nodata'] == np.iinfo(np.uint16).max

    def test_merge_declared_sentinel_respected(self):
        from xrspatial.reproject import merge
        result = merge([
            self._tile_3262(-20, np.int16, attrs={'nodata': -7}),
            self._tile_3262(5, np.int16, attrs={'nodata': -7}),
        ])
        assert result.dtype == np.int16
        assert result.attrs['nodata'] == -7.0
        assert (result.values == -7).any()

    def test_merge_explicit_out_of_range_nodata_raises(self):
        from xrspatial.reproject import merge
        with pytest.raises(ValueError, match='nodata'):
            merge([self._tile_3262(-5, np.uint8),
                   self._tile_3262(5, np.uint8)], nodata=-9999)

    def test_merge_mixed_dtypes_return_float64(self):
        from xrspatial.reproject import merge
        result = merge([self._tile_3262(-5, np.int16),
                        self._tile_3262(5, np.float32)])
        assert result.dtype == np.float64

    def test_merge_float_inputs_stay_float64(self):
        from xrspatial.reproject import merge
        result = merge([self._tile_3262(-5, np.float64),
                        self._tile_3262(5, np.float64)])
        assert result.dtype == np.float64
        assert np.isnan(result.attrs['nodata'])

    def test_merge_mean_strategy_rounds_to_int(self):
        from xrspatial.reproject import merge
        result = merge([
            self._tile_3262(-5, np.int16, fill=2),
            self._tile_3262(-5, np.int16, fill=5),
        ], strategy='mean')
        assert result.dtype == np.int16
        # mean(2, 5) = 3.5 rounds (half-to-even) to 4
        assert result.values[4, 4] == 4

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_merge_dask_int16_preserves_dtype(self):
        from xrspatial.reproject import merge
        lazy = self._tile_3262(-5).copy(
            data=da.from_array(
                np.arange(64, dtype=np.int16).reshape(8, 8), chunks=(4, 4),
            ),
        )
        result = merge([lazy, self._tile_3262(5)])
        # The lazy graph must advertise the same dtype the chunks return.
        assert result.data.dtype == np.int16
        assert result.compute().dtype == np.int16

    @pytest.mark.skipif(not HAS_DASK, reason="dask required")
    def test_merge_dask_empty_chunks_keep_dtype(self):
        from xrspatial.reproject import merge
        # A gap forces no-overlap output chunks; their fills must not
        # promote the assembled mosaic (#3096 trap).
        lazy_a = self._tile_3262(-30).copy(
            data=da.from_array(
                np.arange(64, dtype=np.int16).reshape(8, 8), chunks=(4, 4),
            ),
        )
        result = merge([lazy_a, self._tile_3262(5)], chunk_size=4)
        computed = result.compute()
        assert computed.dtype == np.int16
        assert (computed.values == np.iinfo(np.int16).min).any()

    @pytest.mark.skipif(not HAS_CUPY, reason="cupy/CUDA required")
    def test_merge_cupy_int16_preserves_dtype(self):
        from xrspatial.reproject import merge
        gpu_a = self._tile_3262(-5).copy(
            data=cp.asarray(np.arange(64, dtype=np.int16).reshape(8, 8)),
        )
        gpu_b = self._tile_3262(5).copy(
            data=cp.asarray(np.arange(64, dtype=np.int16).reshape(8, 8)),
        )
        result = merge([gpu_a, gpu_b])
        assert isinstance(result.data, cp.ndarray)
        assert result.dtype == np.int16

    @pytest.mark.skipif(not (HAS_DASK and HAS_CUPY),
                        reason="dask and cupy/CUDA required")
    def test_merge_dask_cupy_int16_preserves_dtype(self):
        from xrspatial.reproject import merge

        def gpu_lazy_tile(x0):
            d = cp.asarray(np.arange(64, dtype=np.int16).reshape(8, 8))
            return self._tile_3262(x0).copy(
                data=da.from_array(d, chunks=(4, 4)),
            )

        result = merge([gpu_lazy_tile(-5), gpu_lazy_tile(5)])
        assert result.data.dtype == np.int16
        computed = result.data.compute()
        assert isinstance(computed, cp.ndarray)
        assert computed.dtype == np.int16

    def test_merge_rejects_inf_nodata(self):
        # Explicit nodata now goes through _detect_nodata, matching
        # reproject(): inf is rejected because it breaks np.isnan masks.
        from xrspatial.reproject import merge
        with pytest.raises(ValueError, match='finite'):
            merge([self._tile_3262(-5, np.float64),
                   self._tile_3262(5, np.float64)], nodata=np.inf)

    def test_merge_explicit_int_sentinel_lands_as_float_attr(self):
        # Same convention as reproject(): the resolved sentinel is a
        # float even when passed as an int.
        from xrspatial.reproject import merge
        result = merge([self._tile_3262(-5), self._tile_3262(5)],
                       nodata=-7)
        assert result.attrs['nodata'] == -7.0
        assert isinstance(result.attrs['nodata'], float)
