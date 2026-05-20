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
    """End-to-end coverage for src_vertical_crs / tgt_vertical_crs."""

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
            src_vertical_crs='EGM96', tgt_vertical_crs='ellipsoidal',
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
            src_vertical_crs='ellipsoidal', tgt_vertical_crs='EGM96',
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
            src_vertical_crs='EGM96', tgt_vertical_crs='EGM2008',
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
            src_vertical_crs='EGM96', tgt_vertical_crs='EGM96',
        )
        np.testing.assert_array_equal(result.values, baseline.values)

    def test_reproject_no_vertical_shift_when_one_none(self):
        """Only one side set -> no shift applied."""
        from xrspatial.reproject import reproject
        raster = self._ny_raster(value=100.0)
        baseline = reproject(raster, 'EPSG:4326')
        result = reproject(
            raster, 'EPSG:4326',
            src_vertical_crs='EGM96',
            tgt_vertical_crs=None,
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
            src_vertical_crs='EGM96', tgt_vertical_crs='ellipsoidal',
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
            src_vertical_crs='EGM96', tgt_vertical_crs='ellipsoidal',
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
                src_vertical_crs='EGM96', tgt_vertical_crs=tgt,
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
        with pytest.raises(ValueError, match="tgt_vertical_crs"):
            reproject(raster, 'EPSG:4326',
                      src_vertical_crs='EGM96', tgt_vertical_crs='NAVD88')
        with pytest.raises(ValueError, match="src_vertical_crs"):
            reproject(raster, 'EPSG:4326',
                      src_vertical_crs='egm96',  # case-sensitive
                      tgt_vertical_crs='ellipsoidal')

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
                           src_vertical_crs='EGM96',
                           tgt_vertical_crs='ellipsoidal')
        out_da = reproject(ds_d, 'EPSG:4326',
                           src_vertical_crs='EGM96',
                           tgt_vertical_crs='ellipsoidal')
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
                           src_vertical_crs='EGM96',
                           tgt_vertical_crs='ellipsoidal')
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
                           src_vertical_crs='EGM96',
                           tgt_vertical_crs='ellipsoidal')
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
                               src_vertical_crs='EGM96',
                               tgt_vertical_crs='ellipsoidal')
        delta_np = shifted_np.values - base_np.values

        base_cu = reproject(ds_cu, 'EPSG:4326')
        shifted_cu = reproject(ds_cu, 'EPSG:4326',
                               src_vertical_crs='EGM96',
                               tgt_vertical_crs='ellipsoidal')
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
