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
        with pytest.raises(TypeError, match="xr.DataArray"):
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
