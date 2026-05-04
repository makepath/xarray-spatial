"""Tests for xrspatial.reproject module."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

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
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

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
        """Different-CRS merge should match within float tolerance."""
        from xrspatial.reproject import merge
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
        ).compute().values

        a_dask = a.copy()
        b_dask = b.copy()
        a_dask.data = da.from_array(a_data, chunks=(8, 8))
        b_dask.data = da.from_array(b_data, chunks=(8, 8))
        dasked = merge(
            [a_dask, b_dask], target_crs='EPSG:4326',
            resolution=1.0, chunk_size=8,
        ).compute().values

        assert eager.shape == dasked.shape
        np.testing.assert_array_equal(np.isnan(eager), np.isnan(dasked))
        finite = np.isfinite(eager)
        if finite.any():
            np.testing.assert_allclose(
                eager[finite], dasked[finite], rtol=1e-10, atol=1e-10,
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
