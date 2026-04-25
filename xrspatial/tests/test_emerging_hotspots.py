"""Tests for xrspatial.emerging_hotspots."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.emerging_hotspots import (
    _MK_ALPHA,
    _classify_single_pixel,
    _mann_kendall_statistic_numpy,
    _mk_pvalue,
    _norm_cdf_approx,
    emerging_hotspots,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kernel_3x3():
    return np.ones((3, 3), dtype=np.float32)


def _make_raster(data, dims=('time', 'y', 'x')):
    return xr.DataArray(data, dims=dims)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_not_a_dataarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            emerging_hotspots(np.zeros((5, 10, 10)), _kernel_3x3())

    def test_not_3d(self):
        raster = _make_raster(np.zeros((10, 10)), dims=('y', 'x'))
        with pytest.raises(ValueError, match="3-D"):
            emerging_hotspots(raster, _kernel_3x3())

    def test_single_time_step(self):
        raster = _make_raster(np.zeros((1, 10, 10)))
        with pytest.raises(ValueError, match="at least 2"):
            emerging_hotspots(raster, _kernel_3x3())

    def test_zero_variance(self):
        raster = _make_raster(np.ones((3, 10, 10), dtype=np.float32))
        with pytest.raises(ZeroDivisionError):
            emerging_hotspots(raster, _kernel_3x3())

    def test_invalid_boundary(self):
        raster = _make_raster(
            np.random.default_rng(0).standard_normal((3, 10, 10)).astype('f4')
        )
        with pytest.raises(ValueError, match="boundary"):
            emerging_hotspots(raster, _kernel_3x3(), boundary='invalid')


# ---------------------------------------------------------------------------
# Normal CDF approximation
# ---------------------------------------------------------------------------

class TestNormCDF:
    def test_symmetry(self):
        assert _norm_cdf_approx(0.0) == pytest.approx(0.5, abs=1e-7)

    def test_large_positive(self):
        assert _norm_cdf_approx(4.0) == pytest.approx(1.0, abs=1e-4)

    def test_large_negative(self):
        assert _norm_cdf_approx(-4.0) == pytest.approx(0.0, abs=1e-4)

    def test_accuracy_vs_scipy(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        for x in [-3.0, -1.96, -1.0, 0.0, 0.5, 1.0, 1.65, 1.96, 2.58, 3.0]:
            expected = scipy_stats.norm.cdf(x)
            got = _norm_cdf_approx(x)
            assert got == pytest.approx(expected, abs=2e-7), f"x={x}"


# ---------------------------------------------------------------------------
# Mann-Kendall unit tests
# ---------------------------------------------------------------------------

class TestMannKendall:
    def test_perfect_increasing(self):
        series = np.arange(10, dtype=np.float64)
        z, s, var_s = _mann_kendall_statistic_numpy(series, len(series))
        assert s > 0
        assert z > 0
        assert _mk_pvalue(z) < 0.05

    def test_perfect_decreasing(self):
        series = np.arange(10, 0, -1, dtype=np.float64)
        z, s, var_s = _mann_kendall_statistic_numpy(series, len(series))
        assert s < 0
        assert z < 0
        assert _mk_pvalue(z) < 0.05

    def test_constant(self):
        series = np.full(10, 5.0)
        z, s, var_s = _mann_kendall_statistic_numpy(series, len(series))
        assert s == 0.0
        assert z == 0.0

    def test_with_ties(self):
        # Ties should reduce the variance compared to no-tie formula
        series = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5, 5], dtype=np.float64)
        z, s, var_s = _mann_kendall_statistic_numpy(series, len(series))
        n = len(series)
        no_tie_var = n * (n - 1) * (2 * n + 5) / 18.0
        assert var_s < no_tie_var

    def test_nan_handling(self):
        series = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
                          dtype=np.float64)
        z, s, var_s = _mann_kendall_statistic_numpy(series, len(series))
        assert s > 0
        assert z > 0

    def test_all_nan(self):
        series = np.full(5, np.nan)
        z, s, var_s = _mann_kendall_statistic_numpy(series, len(series))
        assert z == 0.0
        assert s == 0.0

    def test_too_few_valid(self):
        series = np.array([1.0, np.nan, np.nan, np.nan, np.nan])
        z, s, var_s = _mann_kendall_statistic_numpy(series, len(series))
        assert z == 0.0

    def test_pvalue_at_zero(self):
        assert _mk_pvalue(0.0) == 1.0

    def test_pvalue_range(self):
        for z_val in [-3.0, -1.0, 0.5, 1.96, 3.0]:
            p = _mk_pvalue(z_val)
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Classification unit tests
# ---------------------------------------------------------------------------

class TestClassification:
    """Test each category with synthetic bin arrays."""

    def _bins(self, values):
        return np.array(values, dtype=np.int8)

    def test_new_hot(self):
        # Hot only at final step
        bins = self._bins([0, 0, 0, 0, 0, 0, 0, 0, 0, 90])
        cat = _classify_single_pixel(bins, 10, 0.5, 0.6, _MK_ALPHA)
        assert cat == 1

    def test_consecutive_hot(self):
        # Hot for final 3 steps, never before
        bins = self._bins([0, 0, 0, 0, 0, 0, 0, 95, 95, 95])
        cat = _classify_single_pixel(bins, 10, 1.0, 0.3, _MK_ALPHA)
        assert cat == 2

    def test_intensifying_hot(self):
        # Hot >=90% (all 10), including final, MK increasing (z>0, p<0.05)
        bins = self._bins([90, 90, 90, 90, 90, 90, 90, 90, 90, 99])
        cat = _classify_single_pixel(bins, 10, 3.0, 0.001, _MK_ALPHA)
        assert cat == 3

    def test_persistent_hot(self):
        # Hot >=90% including final, no significant MK trend
        bins = self._bins([95, 95, 95, 95, 95, 95, 95, 95, 95, 95])
        cat = _classify_single_pixel(bins, 10, 0.1, 0.9, _MK_ALPHA)
        assert cat == 4

    def test_diminishing_hot(self):
        # Hot >=90% including final, MK decreasing (z<0, p<0.05)
        bins = self._bins([99, 99, 99, 99, 99, 95, 95, 95, 90, 90])
        cat = _classify_single_pixel(bins, 10, -3.0, 0.001, _MK_ALPHA)
        assert cat == 5

    def test_sporadic_hot(self):
        # Hot at final, <90% of steps, never cold
        bins = self._bins([0, 0, 90, 0, 0, 0, 0, 0, 0, 95])
        cat = _classify_single_pixel(bins, 10, 0.5, 0.6, _MK_ALPHA)
        assert cat == 6

    def test_oscillating_hot(self):
        # Hot at final, was cold at least once
        bins = self._bins([0, -90, 0, 90, 0, -95, 0, 0, 0, 99])
        cat = _classify_single_pixel(bins, 10, 0.5, 0.6, _MK_ALPHA)
        assert cat == 7

    def test_historical_hot(self):
        # Hot >=90% of steps, NOT at final step
        bins = self._bins([95, 95, 95, 95, 95, 95, 95, 95, 95, 0])
        cat = _classify_single_pixel(bins, 10, 1.0, 0.3, _MK_ALPHA)
        assert cat == 8

    def test_cold_mirrors(self):
        """Cold categories are negative mirrors of hot."""
        # New cold: cold only at final step
        bins = self._bins([0, 0, 0, 0, 0, 0, 0, 0, 0, -90])
        cat = _classify_single_pixel(bins, 10, -0.5, 0.6, _MK_ALPHA)
        assert cat == -1

        # Persistent cold
        bins = self._bins([-95] * 10)
        cat = _classify_single_pixel(bins, 10, -0.1, 0.9, _MK_ALPHA)
        assert cat == -4

        # Historical cold
        bins = self._bins([-95, -95, -95, -95, -95, -95, -95, -95, -95, 0])
        cat = _classify_single_pixel(bins, 10, -1.0, 0.3, _MK_ALPHA)
        assert cat == -8

    def test_no_pattern(self):
        bins = self._bins([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cat = _classify_single_pixel(bins, 10, 0.0, 1.0, _MK_ALPHA)
        assert cat == 0


# ---------------------------------------------------------------------------
# Integration tests – NumPy backend
# ---------------------------------------------------------------------------

class TestEmergeNumpy:
    def test_smoke(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 20, 20)).astype('f4')
        raster = _make_raster(data)
        ds = emerging_hotspots(raster, _kernel_3x3())

        assert set(ds.data_vars) == {
            'category', 'gi_zscore', 'gi_bin', 'trend_zscore', 'trend_pvalue'
        }
        assert ds['category'].shape == (20, 20)
        assert ds['gi_zscore'].shape == (8, 20, 20)
        assert ds['gi_bin'].shape == (8, 20, 20)
        assert ds['trend_zscore'].shape == (20, 20)
        assert ds['trend_pvalue'].shape == (20, 20)

        assert ds['category'].dtype == np.int8
        assert ds['gi_zscore'].dtype == np.float32
        assert ds['gi_bin'].dtype == np.int8
        assert ds['trend_zscore'].dtype == np.float32
        assert ds['trend_pvalue'].dtype == np.float32

        cats = ds['category'].values
        assert np.all((cats >= -8) & (cats <= 8))

        pvals = ds['trend_pvalue'].values
        valid = pvals[~np.isnan(pvals)]
        assert np.all((valid >= 0) & (valid <= 1))

    def test_persistent_hot_spot(self):
        """Plant a strong signal at center across all time steps."""
        rng = np.random.default_rng(123)
        data = rng.standard_normal((15, 30, 30)).astype('f4')
        # Plant strong hot spot in center
        data[:, 13:17, 13:17] += 50.0
        raster = _make_raster(data)
        ds = emerging_hotspots(raster, _kernel_3x3())

        center_cat = ds['category'].values[15, 15]
        # Should be persistent hot (4) or intensifying (3)
        assert center_cat in (3, 4), f"Expected 3 or 4, got {center_cat}"

    def test_persistent_cold_spot(self):
        """Plant a strong negative signal at center."""
        rng = np.random.default_rng(456)
        data = rng.standard_normal((15, 30, 30)).astype('f4')
        data[:, 13:17, 13:17] -= 50.0
        raster = _make_raster(data)
        ds = emerging_hotspots(raster, _kernel_3x3())

        center_cat = ds['category'].values[15, 15]
        assert center_cat in (-3, -4), f"Expected -3 or -4, got {center_cat}"

    def test_all_nan_pixel(self):
        """All-NaN pixels should get category 0 and NaN trend stats."""
        data = np.full((5, 10, 10), np.nan, dtype=np.float32)
        # Need at least *some* non-nan pixels for global_std != 0
        data[:, 0:3, 0:3] = np.random.default_rng(0).standard_normal(
            (5, 3, 3)
        ).astype('f4')
        raster = _make_raster(data)
        ds = emerging_hotspots(raster, _kernel_3x3())

        # Pixel (5, 5) should be all NaN across time
        assert ds['category'].values[5, 5] == 0
        assert np.isnan(ds['trend_zscore'].values[5, 5])
        assert np.isnan(ds['trend_pvalue'].values[5, 5])

    def test_boundary_modes(self):
        rng = np.random.default_rng(99)
        data = rng.standard_normal((5, 15, 15)).astype('f4')
        raster = _make_raster(data)
        for mode in ('nan', 'nearest', 'reflect', 'wrap'):
            ds = emerging_hotspots(raster, _kernel_3x3(), boundary=mode)
            assert ds['category'].shape == (15, 15)


# ---------------------------------------------------------------------------
# Integration tests – CuPy backend
# ---------------------------------------------------------------------------

@pytest.fixture
def cupy_available():
    cupy = pytest.importorskip("cupy")
    return cupy


class TestEmergeCuPy:
    def test_cupy_matches_numpy(self, cupy_available):
        cp = cupy_available
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 20, 20)).astype('f4')

        ds_np = emerging_hotspots(_make_raster(data), _kernel_3x3())
        ds_cp = emerging_hotspots(
            _make_raster(cp.asarray(data)), _kernel_3x3()
        )

        np.testing.assert_array_equal(
            ds_cp['category'].data.get(), ds_np['category'].values
        )
        np.testing.assert_allclose(
            ds_cp['gi_zscore'].data.get(), ds_np['gi_zscore'].values,
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            ds_cp['gi_bin'].data.get(), ds_np['gi_bin'].values
        )

    def test_cupy_backed_output(self, cupy_available):
        cp = cupy_available
        rng = np.random.default_rng(42)
        data = cp.asarray(
            rng.standard_normal((5, 15, 15)).astype('f4')
        )
        ds = emerging_hotspots(_make_raster(data), _kernel_3x3())
        assert isinstance(ds['gi_zscore'].data, cp.ndarray)
        assert isinstance(ds['category'].data, cp.ndarray)

    def test_cupy_shapes_dtypes(self, cupy_available):
        cp = cupy_available
        rng = np.random.default_rng(42)
        data = cp.asarray(
            rng.standard_normal((5, 15, 15)).astype('f4')
        )
        ds = emerging_hotspots(_make_raster(data), _kernel_3x3())
        assert ds['category'].shape == (15, 15)
        assert ds['gi_zscore'].shape == (5, 15, 15)


# ---------------------------------------------------------------------------
# Integration tests – Dask + NumPy backend
# ---------------------------------------------------------------------------

@pytest.fixture
def dask_available():
    da = pytest.importorskip("dask.array")
    return da


class TestEmergeDask:
    def test_dask_matches_numpy(self, dask_available):
        da = dask_available
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 20, 20)).astype('f4')

        ds_np = emerging_hotspots(_make_raster(data), _kernel_3x3())
        dask_data = da.from_array(data, chunks=(8, 10, 10))
        ds_dk = emerging_hotspots(_make_raster(dask_data), _kernel_3x3())

        # Force computation
        ds_dk = ds_dk.compute()

        np.testing.assert_array_equal(
            ds_dk['category'].values, ds_np['category'].values
        )
        np.testing.assert_allclose(
            ds_dk['gi_zscore'].values, ds_np['gi_zscore'].values, atol=1e-5
        )
        np.testing.assert_array_equal(
            ds_dk['gi_bin'].values, ds_np['gi_bin'].values
        )

    def test_dask_lazy(self, dask_available):
        da = dask_available
        rng = np.random.default_rng(42)
        data = da.from_array(
            rng.standard_normal((5, 15, 15)).astype('f4'),
            chunks=(5, 8, 8),
        )
        ds = emerging_hotspots(_make_raster(data), _kernel_3x3())
        assert isinstance(ds['gi_zscore'].data, da.Array)

    def test_dask_boundary_modes(self, dask_available):
        da = dask_available
        rng = np.random.default_rng(99)
        np_data = rng.standard_normal((5, 15, 15)).astype('f4')

        for mode in ('nan', 'nearest', 'reflect', 'wrap'):
            ds_np = emerging_hotspots(
                _make_raster(np_data), _kernel_3x3(), boundary=mode
            )
            dask_data = da.from_array(np_data, chunks=(5, 8, 8))
            ds_dk = emerging_hotspots(
                _make_raster(dask_data), _kernel_3x3(), boundary=mode
            ).compute()
            np.testing.assert_allclose(
                ds_dk['gi_zscore'].values,
                ds_np['gi_zscore'].values,
                atol=1e-5,
                err_msg=f"boundary={mode!r}",
            )


# ---------------------------------------------------------------------------
# Dask + CuPy backend
# ---------------------------------------------------------------------------

class TestEmergeDaskCuPy:
    def test_dask_cupy_matches_numpy(self):
        cupy = pytest.importorskip("cupy")
        da = pytest.importorskip("dask.array")

        rng = np.random.default_rng(42)
        np_data = rng.standard_normal((8, 20, 20)).astype('f4')

        ds_np = emerging_hotspots(_make_raster(np_data), _kernel_3x3())

        dask_cupy_data = da.from_array(
            cupy.asarray(np_data), chunks=(8, 10, 10)
        )
        ds_dc = emerging_hotspots(
            _make_raster(dask_cupy_data), _kernel_3x3()
        ).compute()

        for var in ('category', 'gi_bin'):
            dc_vals = ds_dc[var].data
            if isinstance(dc_vals, cupy.ndarray):
                dc_vals = dc_vals.get()
            np.testing.assert_array_equal(
                dc_vals, ds_np[var].values,
                err_msg=f"mismatch in {var}",
            )

        for var in ('gi_zscore', 'trend_zscore', 'trend_pvalue'):
            dc_vals = ds_dc[var].data
            if isinstance(dc_vals, cupy.ndarray):
                dc_vals = dc_vals.get()
            np.testing.assert_allclose(
                dc_vals, ds_np[var].values, atol=1e-5,
                err_msg=f"mismatch in {var}",
            )


# ---------------------------------------------------------------------------
# Memory guard (issue #1274)
# ---------------------------------------------------------------------------

class TestMemoryGuard:
    """Memory guard on the public emerging_hotspots() API (#1274).

    The numpy and cupy backends materialise three full ``(T, H, W)``
    cubes (a float32 input copy, gi_zscore float32, gi_bin int8) plus
    small H*W temporaries.  The public API estimates this footprint and
    raises ``MemoryError`` before the first allocation when it would
    exceed 50% of available RAM.  Dask-backed inputs are not guarded
    because their map_blocks/map_overlap path processes per-chunk.
    """

    def _stub_available(self, monkeypatch, n_bytes):
        """Pin _available_memory_bytes to a fixed return value."""
        import sys
        mod = sys.modules['xrspatial.emerging_hotspots']
        monkeypatch.setattr(mod, '_available_memory_bytes', lambda: n_bytes)

    def test_numpy_raises_when_budget_exceeded(self, monkeypatch):
        """numpy backend should refuse a cube too large for memory."""
        # A (5, 100, 100) cube needs ~600 KB; with 1 KB "available" the
        # guard must trip.
        self._stub_available(monkeypatch, 1024)
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 100, 100)).astype('f4')
        raster = _make_raster(data)
        with pytest.raises(MemoryError, match=r"emerging_hotspots"):
            emerging_hotspots(raster, _kernel_3x3())

    def test_numpy_passes_when_budget_ample(self, monkeypatch):
        """Small inputs should pass the guard with normal RAM headroom."""
        self._stub_available(monkeypatch, 64 * 1024 ** 3)
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 20, 20)).astype('f4')
        raster = _make_raster(data)
        ds = emerging_hotspots(raster, _kernel_3x3())
        assert ds['category'].shape == (20, 20)

    def test_error_message_mentions_shape_and_gb(self, monkeypatch):
        """Error message should surface the cube shape and projected size."""
        self._stub_available(monkeypatch, 1024)
        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 50, 50)).astype('f4')
        raster = _make_raster(data)
        with pytest.raises(MemoryError, match=r"\(4, 50, 50\)"):
            emerging_hotspots(raster, _kernel_3x3())

    def test_cupy_raises_when_budget_exceeded(self, monkeypatch):
        """CuPy backend honours the same in-RAM guard."""
        cp = pytest.importorskip("cupy")
        self._stub_available(monkeypatch, 1024)
        rng = np.random.default_rng(0)
        data = cp.asarray(
            rng.standard_normal((5, 100, 100)).astype('f4')
        )
        raster = _make_raster(data)
        with pytest.raises(MemoryError, match=r"emerging_hotspots"):
            emerging_hotspots(raster, _kernel_3x3())

    def test_dask_skips_in_ram_guard(self, monkeypatch):
        """Dask-backed inputs should not be blocked by the in-RAM guard.

        The numpy/cupy guard targets backends that materialise the full
        cube; dask paths process per-chunk via map_blocks/map_overlap.
        """
        da_mod = pytest.importorskip("dask.array")
        # Even with 1 byte "available", the dask path should pass the
        # public-API check because the guard is skipped for dask inputs.
        self._stub_available(monkeypatch, 1)
        rng = np.random.default_rng(0)
        data = da_mod.from_array(
            rng.standard_normal((5, 20, 20)).astype('f4'),
            chunks=(5, 10, 10),
        )
        raster = _make_raster(data)
        ds = emerging_hotspots(raster, _kernel_3x3())
        # Materialise the lazy result to confirm the full pipeline runs.
        ds = ds.compute()
        assert ds['category'].shape == (20, 20)
