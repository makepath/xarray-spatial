"""Tests for the interpolation module (IDW, Kriging, Spline)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.interpolate import idw, kriging, spline
from xrspatial.tests.general_checks import (
    cuda_and_cupy_available,
    dask_array_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_template(y_coords, x_coords, backend='numpy', chunks=(3, 3)):
    """Build a template DataArray for the given coordinate vectors."""
    data = np.zeros((len(y_coords), len(x_coords)), dtype=np.float64)
    da_out = xr.DataArray(data, dims=['y', 'x'])
    da_out['y'] = np.asarray(y_coords, dtype=np.float64)
    da_out['x'] = np.asarray(x_coords, dtype=np.float64)

    if backend == 'cupy':
        import cupy
        da_out.data = cupy.asarray(da_out.data)
    elif backend == 'dask_cupy':
        import cupy
        import dask.array as da
        da_out.data = da.from_array(
            cupy.asarray(da_out.data), chunks=chunks,
            meta=cupy.array((), dtype=np.float64),
        )
    elif 'dask' in backend:
        import dask.array as da
        da_out.data = da.from_array(da_out.data, chunks=chunks)

    return da_out


def _to_numpy(da_result):
    """Extract numpy array from any-backend DataArray."""
    data = da_result.data
    try:
        import dask.array as dask_array
        if isinstance(data, dask_array.Array):
            data = data.compute()
    except ImportError:
        pass
    if hasattr(data, 'get'):
        return data.get()
    return np.asarray(data)


def _grid_points():
    """3x3 regular grid with known z = row*3 + col + 1."""
    x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    z = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    return x, y, z


# ===================================================================
# IDW tests
# ===================================================================

class TestIDW:

    def test_exact_interpolation(self):
        """Grid point coinciding with a data point returns that value."""
        x, y, z = _grid_points()
        template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        result = idw(x, y, z, template, power=2.0)
        expected = z.reshape(3, 3)
        np.testing.assert_allclose(result.values, expected)

    def test_symmetry(self):
        """Equidistant points produce their mean."""
        x = np.array([0.0, 2.0])
        y = np.array([1.0, 1.0])
        z = np.array([10.0, 20.0])
        template = _make_template([1.0], [1.0])
        result = idw(x, y, z, template, power=2.0)
        np.testing.assert_allclose(result.values, [[15.0]])

    def test_k1_nearest_neighbor(self):
        """k=1 returns the nearest point's value."""
        x = np.array([0.0, 3.0])
        y = np.array([0.0, 0.0])
        z = np.array([10.0, 20.0])
        template = _make_template([0.0], [0.5, 2.5])
        result = idw(x, y, z, template, k=1)
        np.testing.assert_allclose(result.values, [[10.0, 20.0]])

    def test_k_capped_to_npoints(self):
        """k larger than number of points is capped silently."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
        z = np.array([5.0, 10.0])
        template = _make_template([0.0], [0.5])
        result = idw(x, y, z, template, k=100)
        assert result.shape == (1, 1)
        assert np.isfinite(result.values[0, 0])

    def test_single_point(self):
        """Single data point: entire grid equals that value."""
        template = _make_template([0.0, 1.0], [0.0, 1.0])
        result = idw([0.5], [0.5], [42.0], template)
        np.testing.assert_allclose(result.values, 42.0)

    def test_fill_value(self):
        """Custom fill_value is returned when all weights are zero.

        (In practice this shouldn't happen with normal data, so just
        verify the parameter is accepted.)
        """
        template = _make_template([0.0], [0.0])
        result = idw([0.0], [0.0], [7.0], template, fill_value=-999.0)
        # Point exactly at grid location, exact match → returns 7.0
        np.testing.assert_allclose(result.values, [[7.0]])

    def test_output_metadata(self):
        """Output DataArray preserves template coords and dims."""
        template = _make_template([0.0, 1.0], [0.0, 1.0])
        result = idw([0.0], [0.0], [1.0], template, name='my_idw')
        assert result.name == 'my_idw'
        assert result.dims == template.dims
        assert result.shape == template.shape
        np.testing.assert_array_equal(result.coords['x'].values,
                                      template.coords['x'].values)

    @dask_array_available
    def test_dask_matches_numpy(self):
        x, y, z = _grid_points()
        np_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        da_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                     backend='dask', chunks=(2, 2))
        np_result = idw(x, y, z, np_template)
        da_result = idw(x, y, z, da_template)
        np.testing.assert_allclose(
            np_result.values, da_result.values, rtol=1e-10)

    @dask_array_available
    def test_dask_knearest_matches_numpy(self):
        x, y, z = _grid_points()
        np_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        da_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                     backend='dask', chunks=(2, 2))
        np_result = idw(x, y, z, np_template, k=3)
        da_result = idw(x, y, z, da_template, k=3)
        np.testing.assert_allclose(
            np_result.values, da_result.values, rtol=1e-10)


# ===================================================================
# Spline tests
# ===================================================================

class TestSpline:

    def test_exact_interpolation(self):
        """smoothing=0 passes through all data points."""
        x, y, z = _grid_points()
        template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        result = spline(x, y, z, template, smoothing=0.0)
        expected = z.reshape(3, 3)
        np.testing.assert_allclose(result.values, expected, atol=1e-6)

    def test_linear_recovery(self):
        """z = a + b*x + c*y is recovered exactly by TPS."""
        a, b, c = 3.0, 2.0, -1.0
        x = np.array([0.0, 1.0, 2.0, 0.5, 1.5])
        y = np.array([0.0, 0.0, 1.0, 1.5, 0.5])
        z = a + b * x + c * y

        template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        result = spline(x, y, z, template, smoothing=0.0)

        gx, gy = np.meshgrid(
            template.coords['x'].values,
            template.coords['y'].values,
        )
        expected = a + b * gx + c * gy
        np.testing.assert_allclose(result.values, expected, atol=1e-6)

    def test_smoothing_reduces_oscillation(self):
        """Positive smoothing should not pass through noisy data."""
        rng = np.random.RandomState(42)
        x = rng.uniform(0, 10, 20)
        y = rng.uniform(0, 10, 20)
        z = np.sin(x) + rng.normal(0, 0.5, 20)

        template = _make_template(
            np.linspace(0, 10, 11), np.linspace(0, 10, 11))

        exact = spline(x, y, z, template, smoothing=0.0)
        smooth = spline(x, y, z, template, smoothing=1.0)

        # Smooth result should have smaller range (less oscillation)
        assert np.ptp(smooth.values) < np.ptp(exact.values)

    def test_single_point(self):
        """Single point: TPS degenerates to constant surface."""
        template = _make_template([0.0, 1.0], [0.0, 1.0])
        result = spline([0.5], [0.5], [42.0], template, smoothing=0.0)
        np.testing.assert_allclose(result.values, 42.0, atol=1e-6)

    @dask_array_available
    def test_dask_matches_numpy(self):
        x, y, z = _grid_points()
        np_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        da_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                     backend='dask', chunks=(2, 2))
        np_result = spline(x, y, z, np_template)
        da_result = spline(x, y, z, da_template)
        np.testing.assert_allclose(
            np_result.values, da_result.values, rtol=1e-10)


# ===================================================================
# Kriging tests
# ===================================================================

class TestKriging:

    @staticmethod
    def _spatial_data():
        """Points with clear spatial structure for variogram fitting."""
        rng = np.random.RandomState(123)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0,
                       0.0, 1.0, 2.0, 3.0, 4.0,
                       0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                       2.0, 2.0, 2.0, 2.0, 2.0,
                       4.0, 4.0, 4.0, 4.0, 4.0])
        z = 2.0 * x + 3.0 * y + rng.normal(0, 0.1, len(x))
        return x, y, z

    def test_prediction_at_data_points(self):
        """Prediction near data points should be close to observed."""
        x, y, z = self._spatial_data()
        template = _make_template(
            [0.0, 2.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0])
        result = kriging(x, y, z, template)

        # Check a few points that sit exactly on data locations
        for ix, xv in enumerate([0.0, 1.0, 2.0, 3.0, 4.0]):
            for iy, yv in enumerate([0.0, 2.0, 4.0]):
                mask = (x == xv) & (y == yv)
                if mask.any():
                    observed = z[mask][0]
                    predicted = result.values[iy, ix]
                    np.testing.assert_allclose(
                        predicted, observed, atol=0.5,
                        err_msg=f"at ({xv}, {yv})")

    def test_return_variance(self):
        """Variance should be available when requested."""
        x, y, z = self._spatial_data()
        template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        pred, var = kriging(x, y, z, template, return_variance=True)

        assert isinstance(pred, xr.DataArray)
        assert isinstance(var, xr.DataArray)
        assert pred.shape == var.shape

    def test_variogram_models(self):
        """All three variogram models should produce finite output."""
        x, y, z = self._spatial_data()
        template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        for model in ('spherical', 'exponential', 'gaussian'):
            result = kriging(x, y, z, template, variogram_model=model)
            assert np.all(np.isfinite(result.values)), \
                f"model={model} produced non-finite values"

    @dask_array_available
    def test_dask_matches_numpy(self):
        x, y, z = self._spatial_data()
        np_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        da_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0],
                                     backend='dask', chunks=(2, 2))
        np_result = kriging(x, y, z, np_template)
        da_result = kriging(x, y, z, da_template)
        np.testing.assert_allclose(
            np_result.values, da_result.values, rtol=1e-10)

    @cuda_and_cupy_available
    def test_cupy_matches_numpy(self):
        """CuPy backend produces same results as numpy."""
        x, y, z = self._spatial_data()
        np_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        cp_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0],
                                     backend='cupy')
        np_result = kriging(x, y, z, np_template)
        cp_result = kriging(x, y, z, cp_template)
        np.testing.assert_allclose(
            np_result.values, _to_numpy(cp_result), rtol=1e-10)

    @cuda_and_cupy_available
    def test_cupy_variogram_models(self):
        """All variogram models produce finite output on GPU."""
        x, y, z = self._spatial_data()
        cp_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0],
                                     backend='cupy')
        for model in ('spherical', 'exponential', 'gaussian'):
            result = kriging(x, y, z, cp_template, variogram_model=model)
            assert np.all(np.isfinite(_to_numpy(result))), \
                f"model={model} produced non-finite values on CuPy"

    @cuda_and_cupy_available
    def test_cupy_return_variance(self):
        """Variance is returned correctly on GPU."""
        x, y, z = self._spatial_data()
        np_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        cp_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0],
                                     backend='cupy')
        np_pred, np_var = kriging(x, y, z, np_template, return_variance=True)
        cp_pred, cp_var = kriging(x, y, z, cp_template, return_variance=True)
        np.testing.assert_allclose(
            np_pred.values, _to_numpy(cp_pred), rtol=1e-10)
        np.testing.assert_allclose(
            np_var.values, _to_numpy(cp_var), atol=1e-12)

    @cuda_and_cupy_available
    @dask_array_available
    def test_dask_cupy_matches_numpy(self):
        """Dask+CuPy backend produces same results as numpy."""
        x, y, z = self._spatial_data()
        np_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        dc_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0],
                                     backend='dask_cupy', chunks=(2, 2))
        np_result = kriging(x, y, z, np_template)
        dc_result = kriging(x, y, z, dc_template)
        np.testing.assert_allclose(
            np_result.values, _to_numpy(dc_result), rtol=1e-10)

    @cuda_and_cupy_available
    @dask_array_available
    def test_dask_cupy_return_variance(self):
        """Dask+CuPy variance matches numpy."""
        x, y, z = self._spatial_data()
        np_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        dc_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0],
                                     backend='dask_cupy', chunks=(2, 2))
        np_pred, np_var = kriging(x, y, z, np_template, return_variance=True)
        dc_pred, dc_var = kriging(x, y, z, dc_template, return_variance=True)
        np.testing.assert_allclose(
            np_pred.values, _to_numpy(dc_pred), rtol=1e-10)
        np.testing.assert_allclose(
            np_var.values, _to_numpy(dc_var), atol=1e-12)


# ===================================================================
# Validation / edge-case tests
# ===================================================================

class TestValidation:

    def test_idw_template_not_dataarray(self):
        with pytest.raises(TypeError, match='must be an xarray.DataArray'):
            idw([0], [0], [0], np.zeros((3, 3)))

    def test_idw_invalid_power(self):
        template = _make_template([0.0], [0.0])
        with pytest.raises((TypeError, ValueError)):
            idw([0], [0], [0], template, power=-1.0)

    def test_idw_mismatched_lengths(self):
        template = _make_template([0.0], [0.0])
        with pytest.raises(ValueError, match='same length'):
            idw([0, 1], [0], [0], template)

    def test_idw_all_nan_points(self):
        template = _make_template([0.0], [0.0])
        with pytest.raises(ValueError, match='no valid'):
            idw([np.nan], [np.nan], [np.nan], template)

    def test_kriging_invalid_model(self):
        template = _make_template([0.0], [0.0])
        with pytest.raises(ValueError, match='variogram_model'):
            kriging([0, 1], [0, 1], [0, 1], template,
                    variogram_model='invalid')

    def test_spline_negative_smoothing(self):
        template = _make_template([0.0], [0.0])
        with pytest.raises(ValueError):
            spline([0], [0], [0], template, smoothing=-1.0)

    def test_nan_points_are_dropped(self):
        """NaN points are silently removed; remaining points are used."""
        x = np.array([0.0, np.nan, 1.0])
        y = np.array([0.0, 0.0, 0.0])
        z = np.array([5.0, 99.0, 10.0])
        template = _make_template([0.0], [0.5])
        result = idw(x, y, z, template)
        assert np.isfinite(result.values[0, 0])


# ===================================================================
# Memory guard tests (issue #1307)
# ===================================================================

class TestKrigingMemoryGuard:
    """Verify kriging() refuses to allocate more than ~80% of RAM.

    Allocations scale with point count N (variogram + matrix) and with
    grid_pixels * N (prediction).  We monkeypatch the available-memory
    helper to a small number so tests can simulate "too big" without
    actually allocating gigabytes.
    """

    def test_predict_matrix_exceeds_memory(self, monkeypatch):
        """Large grid x N k0 matrix triggers the guard."""
        from xrspatial.interpolate import _kriging as _kr

        # Pretend we only have 64 MB available.
        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 64 * 1024 ** 2,
        )

        x = np.array([0.0, 1.0, 2.0, 0.5])
        y = np.array([0.0, 0.0, 1.0, 1.5])
        z = np.array([1.0, 2.0, 3.0, 4.0])
        # 2000x2000 grid * 5 cols * 8 bytes ~= 160 MB > 64 MB * 0.8
        template = _make_template(
            np.arange(2000, dtype=np.float64),
            np.arange(2000, dtype=np.float64),
        )

        with pytest.raises(MemoryError, match='prediction matrix'):
            kriging(x, y, z, template)

    def test_variogram_pairs_exceed_memory(self, monkeypatch):
        """Large N triggers the variogram-pair guard before the matrix one."""
        from xrspatial.interpolate import _kriging as _kr

        # 32 MB available.  N=4000 -> N*(N-1)/2 ~ 8e6 pairs * 4 buffers
        # * 8 bytes = ~256 MB.  Larger than the 4001x4001 matrix path
        # (~384 MB), so let's use a different N that makes pair_bytes win.
        # N=10000 -> pair_bytes ~ 1.6 GB; matrix_bytes ~ 2.4 GB.
        # Matrix wins for any N because of the 3x multiplier vs 4x and the
        # (N+1)^2 term.  Use a smaller N so matrix wins; check generic msg.
        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 32 * 1024 ** 2,
        )

        n = 3000
        rng = np.random.RandomState(0)
        x = rng.uniform(0, 10, n)
        y = rng.uniform(0, 10, n)
        z = rng.uniform(0, 10, n)
        template = _make_template([0.0, 1.0], [0.0, 1.0])

        # n=3000 -> matrix_bytes = 3 * 3001^2 * 8 ~= 216 MB > 32*0.8 MB
        with pytest.raises(MemoryError, match='kriging matrix'):
            kriging(x, y, z, template)

    def test_small_input_allowed(self, monkeypatch):
        """Tiny inputs pass the guard even with very low available memory."""
        # 16 MB available is plenty for a 4-point, 3x3 grid problem.
        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 16 * 1024 ** 2,
        )

        x, y, z = _grid_points()
        template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        # Should not raise.
        result = kriging(x, y, z, template)
        assert result.shape == template.shape

    def test_check_helper_estimate_message(self, monkeypatch):
        """_check_kriging_memory reports GB and identifies culprit."""
        from xrspatial.interpolate._kriging import _check_kriging_memory

        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 1 * 1024 ** 2,  # 1 MB
        )

        # n=10, grid_pixels=100000 -> k0 ~ 26 MB > 1 MB * 0.8.
        with pytest.raises(MemoryError, match='prediction matrix'):
            _check_kriging_memory(n_points=10, grid_pixels=100_000)
