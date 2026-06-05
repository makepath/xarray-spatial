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

    @cuda_and_cupy_available
    def test_cupy_matches_numpy(self):
        """CuPy backend (all-points) produces same results as numpy."""
        x, y, z = _grid_points()
        np_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        cp_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                     backend='cupy')
        np_result = idw(x, y, z, np_template)
        cp_result = idw(x, y, z, cp_template)
        np.testing.assert_allclose(
            np_result.values, _to_numpy(cp_result), rtol=1e-10)

    @cuda_and_cupy_available
    def test_cupy_exact_interpolation(self):
        """CuPy exact-match path returns the coincident point's value."""
        x, y, z = _grid_points()
        cp_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                     backend='cupy')
        result = idw(x, y, z, cp_template, power=2.0)
        expected = z.reshape(3, 3)
        np.testing.assert_allclose(_to_numpy(result), expected)

    @cuda_and_cupy_available
    def test_cupy_knearest_rejected(self):
        """k-nearest mode is rejected on the CuPy backend."""
        x, y, z = _grid_points()
        cp_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                     backend='cupy')
        with pytest.raises(NotImplementedError, match='k-nearest'):
            idw(x, y, z, cp_template, k=2)

    @cuda_and_cupy_available
    @dask_array_available
    def test_dask_cupy_matches_numpy(self):
        """Dask+CuPy backend (all-points) produces same results as numpy."""
        x, y, z = _grid_points()
        np_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        dc_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                     backend='dask_cupy', chunks=(2, 2))
        np_result = idw(x, y, z, np_template)
        dc_result = idw(x, y, z, dc_template)
        np.testing.assert_allclose(
            np_result.values, _to_numpy(dc_result), rtol=1e-10)

    @cuda_and_cupy_available
    @dask_array_available
    def test_dask_cupy_knearest_rejected(self):
        """k-nearest mode is rejected on the Dask+CuPy backend."""
        x, y, z = _grid_points()
        dc_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                     backend='dask_cupy', chunks=(2, 2))
        with pytest.raises(NotImplementedError):
            idw(x, y, z, dc_template, k=2)


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

    def test_two_point_affine_fit(self):
        """n == 2 falls back to a least-squares affine fit.

        With two points the full TPS system is underdetermined, so
        _tps_build_and_solve fits z = a0 + a1*x + a2*y instead. Two
        points on the x-axis with z = 10 and 20 define the gradient
        along x; the midpoint should read 15.
        """
        x = np.array([0.0, 2.0])
        y = np.array([0.0, 0.0])
        z = np.array([10.0, 20.0])
        template = _make_template([0.0], [0.0, 1.0, 2.0])
        result = spline(x, y, z, template, smoothing=0.0)
        np.testing.assert_allclose(
            result.values, [[10.0, 15.0, 20.0]], atol=1e-6)

    def test_output_metadata(self):
        """Output DataArray preserves template coords, dims, and name."""
        x, y, z = _grid_points()
        template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        template.attrs['res'] = (1.0, 1.0)
        result = spline(x, y, z, template, name='my_spline')
        assert result.name == 'my_spline'
        assert result.dims == template.dims
        assert result.shape == template.shape
        assert result.attrs == template.attrs
        np.testing.assert_array_equal(result.coords['x'].values,
                                      template.coords['x'].values)
        np.testing.assert_array_equal(result.coords['y'].values,
                                      template.coords['y'].values)

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

    @cuda_and_cupy_available
    def test_cupy_matches_numpy(self):
        """CuPy backend produces same results as numpy."""
        x, y, z = _grid_points()
        np_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        cp_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                     backend='cupy')
        np_result = spline(x, y, z, np_template)
        cp_result = spline(x, y, z, cp_template)
        np.testing.assert_allclose(
            np_result.values, _to_numpy(cp_result), rtol=1e-10)

    @cuda_and_cupy_available
    @dask_array_available
    def test_dask_cupy_matches_numpy(self):
        """Dask+CuPy backend produces same results as numpy."""
        x, y, z = _grid_points()
        np_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        dc_template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
                                     backend='dask_cupy', chunks=(2, 2))
        np_result = spline(x, y, z, np_template)
        dc_result = spline(x, y, z, dc_template)
        np.testing.assert_allclose(
            np_result.values, _to_numpy(dc_result), rtol=1e-10)

    @cuda_and_cupy_available
    @dask_array_available
    def test_dask_cupy_uploads_points_once(self, monkeypatch):
        """The dask+cupy path uploads the point/weight arrays once.

        These arrays are the same for every chunk, so the number of
        host-to-device transfers of them must not scale with chunk
        count.  Regression guard against re-uploading inside the
        per-chunk closure.
        """
        import cupy

        from xrspatial.interpolate import _spline as spline_mod

        n_points = 8
        rng = np.random.RandomState(0)
        x = rng.uniform(0, 2, n_points)
        y = rng.uniform(0, 2, n_points)
        z = rng.uniform(0, 2, n_points)

        coords = np.linspace(0.0, 2.0, 8)

        orig_asarray = cupy.asarray
        invariant_uploads = {'n': 0}

        def counting_asarray(a, *args, **kwargs):
            # Point coordinate vectors (len n_points) and the weight
            # vector (len n_points + 3) are the chunk-invariant uploads.
            if isinstance(a, np.ndarray) and a.size in (
                    n_points, n_points + 3):
                invariant_uploads['n'] += 1
            return orig_asarray(a, *args, **kwargs)

        monkeypatch.setattr(spline_mod.cupy, 'asarray', counting_asarray)

        template = _make_template(coords, coords,
                                  backend='dask_cupy', chunks=(2, 2))
        result = spline(x, y, z, template)
        result.data.compute()

        # x_pts, y_pts and weights -> exactly three uploads, regardless
        # of how many chunks the grid was split into.
        assert invariant_uploads['n'] == 3


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

    def test_matrix_diagonal_is_zero_with_nugget(self):
        """gamma(0)=0: a non-zero nugget must not land on the diagonal.

        vario_func(0) returns the nugget c0, but the semivariogram is
        zero at lag 0 by definition. _build_kriging_matrix must zero the
        diagonal of the variogram block so a non-zero nugget does not
        force exact interpolation or shrink the kriging variance.
        """
        from xrspatial.interpolate._kriging import (
            _build_kriging_matrix,
            _spherical,
        )

        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])

        c0 = 0.5  # non-zero nugget
        assert _spherical(np.array([0.0]), c0, 1.0, 2.0)[0] == c0

        def vario(h):
            return _spherical(h, c0, 1.0, 2.0)

        k_inv = _build_kriging_matrix(x, y, vario)
        assert k_inv is not None
        # Reconstruct K from its inverse and check the variogram-block
        # diagonal is 0, not the nugget.
        k = np.linalg.inv(k_inv)
        n = len(x)
        np.testing.assert_allclose(np.diag(k[:n, :n]), 0.0, atol=1e-9)

    def test_nugget_smooths_and_inflates_variance(self):
        """A non-zero nugget should smooth predictions and lift variance.

        With the nugget on the diagonal (the old bug) prediction at a
        data point reproduces the observed value exactly and the
        variance there collapses to ~the nugget. With gamma(0)=0 the
        predictor smooths through the noise, so the prediction at a data
        location differs from the observed value and the variance is
        strictly larger than that buggy floor.
        """
        from xrspatial.interpolate._kriging import (
            _build_kriging_matrix,
            _kriging_predict,
            _spherical,
        )

        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.zeros_like(x)
        z = np.array([1.0, 3.0, 2.0, 5.0, 4.0])

        c0 = 0.5
        def vario(h):
            return _spherical(h, c0, 1.0, 3.0)

        k_inv = _build_kriging_matrix(x, y, vario)
        # Predict exactly at the data point x=2 (observed z=2.0).
        pred, var = _kriging_predict(
            x, y, z, np.array([2.0]), np.array([0.0]),
            vario, k_inv, True)

        # Smoothing: prediction does not reproduce the observed value.
        assert abs(pred.ravel()[0] - 2.0) > 1e-3
        # Variance reflects the nugget rather than collapsing to it.
        assert var.ravel()[0] > c0

    def test_single_point(self):
        """A single input point falls back to a constant prediction.

        With one point there are no distance pairs, so the experimental
        variogram is empty and kriging cannot fit spatial structure. The
        function should reach the default-variogram fallback (issue #2917)
        rather than crashing on an empty-array reduction.
        """
        template = _make_template([0.0, 1.0], [0.0, 1.0])
        with pytest.warns(UserWarning, match='fewer than 3'):
            result = kriging([5.0], [5.0], [42.0], template)
        assert np.all(np.isfinite(result.values))
        np.testing.assert_allclose(result.values, 42.0)

    def test_single_valid_point_after_nan_drop(self):
        """NaN filtering down to one point reaches the same fallback."""
        template = _make_template([0.0, 1.0], [0.0, 1.0])
        x = np.array([1.0, np.nan, np.nan])
        y = np.array([2.0, np.nan, np.nan])
        z = np.array([7.0, np.nan, np.nan])
        with pytest.warns(UserWarning, match='fewer than 3'):
            result = kriging(x, y, z, template)
        assert np.all(np.isfinite(result.values))
        np.testing.assert_allclose(result.values, 7.0)

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

    @dask_array_available
    def test_dask_return_variance_matches_numpy(self):
        """Dask+numpy variance matches numpy (exercises _chunk_var)."""
        x, y, z = self._spatial_data()
        np_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        da_template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0],
                                     backend='dask', chunks=(2, 2))
        np_pred, np_var = kriging(x, y, z, np_template, return_variance=True)
        da_pred, da_var = kriging(x, y, z, da_template, return_variance=True)
        np.testing.assert_allclose(
            np_pred.values, da_pred.values, rtol=1e-10)
        np.testing.assert_allclose(
            np_var.values, da_var.values, atol=1e-12)

    def test_nlags_nondefault(self):
        """A non-default nlags value still produces finite output."""
        x, y, z = self._spatial_data()
        template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        result = kriging(x, y, z, template, nlags=5)
        assert np.all(np.isfinite(result.values))

    def test_two_point_warns_few_lag_bins(self):
        """Two points yield fewer than 3 lag bins and warn; output finite."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
        z = np.array([5.0, 10.0])
        template = _make_template([0.0, 1.0], [0.0, 1.0])
        with pytest.warns(UserWarning, match='fewer than 3'):
            result = kriging(x, y, z, template)
        assert np.all(np.isfinite(result.values))

    def test_output_metadata(self):
        """Output preserves template coords, dims, attrs, and name."""
        x, y, z = self._spatial_data()
        template = _make_template([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        template.attrs['res'] = (2.0, 2.0)
        result = kriging(x, y, z, template, name='my_kriging')
        assert result.name == 'my_kriging'
        assert result.dims == template.dims
        assert result.shape == template.shape
        assert result.attrs.get('res') == (2.0, 2.0)
        np.testing.assert_array_equal(result.coords['x'].values,
                                      template.coords['x'].values)
        np.testing.assert_array_equal(result.coords['y'].values,
                                      template.coords['y'].values)


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

    @pytest.mark.parametrize('nlags', [0, -1])
    def test_kriging_nlags_below_min_raises(self, nlags):
        template = _make_template([0.0], [0.0])
        with pytest.raises(ValueError, match='nlags'):
            kriging([0, 1], [0, 1], [0, 1], template, nlags=nlags)

    def test_kriging_nlags_non_int_raises(self):
        template = _make_template([0.0], [0.0])
        with pytest.raises(TypeError, match='nlags'):
            kriging([0, 1], [0, 1], [0, 1], template, nlags=2.5)

    def test_kriging_all_nan_points(self):
        template = _make_template([0.0], [0.0])
        with pytest.raises(ValueError, match='no valid'):
            kriging([np.nan], [np.nan], [np.nan], template)

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

    def test_check_helper_dask_skips_k0_term(self):
        """is_dask=True drops the grid-sized k0 term from the estimate."""
        from xrspatial.interpolate._kriging import _check_kriging_memory

        # Same n and grid_pixels as the message test above, which trips
        # the guard at 1 MB available.  With is_dask=True the k0 term is
        # dropped, so the tiny point-based terms stay well under 1 MB and
        # nothing is raised.
        _check_kriging_memory(n_points=10, grid_pixels=100_000,
                              is_dask=True)

    def test_check_helper_dask_still_guards_matrix(self, monkeypatch):
        """is_dask=True keeps the point-based matrix guard."""
        from xrspatial.interpolate._kriging import _check_kriging_memory

        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 32 * 1024 ** 2,
        )

        # n=3000 -> matrix_bytes ~ 216 MB regardless of backend.
        with pytest.raises(MemoryError, match='kriging matrix'):
            _check_kriging_memory(n_points=3000, grid_pixels=4,
                                  is_dask=True)

    @dask_array_available
    def test_dask_template_skips_grid_memory_guard(self, monkeypatch):
        """A large chunked dask template is not rejected by the guard.

        The dask backend builds the prediction matrix per chunk, so the
        full-grid k0 estimate must not gate it (issue #2923).
        """
        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 64 * 1024 ** 2,
        )

        rng = np.random.RandomState(0)
        x = rng.uniform(0, 100, 20)
        y = rng.uniform(0, 100, 20)
        z = 2.0 * x + 3.0 * y + rng.normal(0, 0.1, 20)

        # Full-grid k0 would be ~8 GB, far over 64 MB, but each
        # 200x200 chunk only needs ~20 MB.
        template = _make_template(
            np.linspace(0, 100, 4000),
            np.linspace(0, 100, 4000),
            backend='dask', chunks=(200, 200),
        )

        result = kriging(x, y, z, template)
        assert result.shape == template.shape


class TestIDWMemoryGuard:
    """Verify idw(k=...) refuses to allocate more than ~80% of RAM.

    The k-nearest path materialises a ``(grid_pixels, k)`` float64
    distance array and an int64 index array via ``cKDTree.query``.  Peak
    use is ``grid_pixels * k * 16`` bytes.  The guard is monkeypatched
    against a small memory budget so tests can simulate "too big"
    without allocating GB.
    """

    def test_oversize_grid_raises(self, monkeypatch):
        """Large grid x k allocation triggers the guard."""
        # 32 MB available.  2000x2000 grid * k=4 * 16 B = 256 MB > 0.8*32 MB.
        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 32 * 1024 ** 2,
        )

        x, y, z = _grid_points()
        template = _make_template(
            np.arange(2000, dtype=np.float64),
            np.arange(2000, dtype=np.float64),
        )

        with pytest.raises(MemoryError, match='idw'):
            idw(x, y, z, template, k=4)

    def test_normal_succeeds(self, monkeypatch):
        """Small inputs pass the guard with a tight memory budget."""
        # 16 MB available is plenty for a 9-point, 3x3 grid problem.
        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 16 * 1024 ** 2,
        )

        x, y, z = _grid_points()
        template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

        # Should not raise.
        result = idw(x, y, z, template, k=3)
        assert result.shape == template.shape

    def test_message_names_grid_and_k(self, monkeypatch):
        """Error message identifies grid size and k for the user."""
        from xrspatial.interpolate._idw import _check_idw_memory

        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 1 * 1024 ** 2,  # 1 MB
        )

        # grid_pixels=1_000_000, k=8 -> 128 MB > 0.8 MB.
        with pytest.raises(MemoryError) as excinfo:
            _check_idw_memory(grid_pixels=1_000_000, k=8)

        msg = str(excinfo.value)
        assert '1000000' in msg
        assert '8' in msg
        assert 'GB' in msg


# ===================================================================
# Spline memory guard tests (issue #1372)
# ===================================================================

class TestSplineMemoryGuard:
    """Verify spline() refuses to allocate more than ~80% of RAM.

    The TPS solver builds N x N kernel matrices (K, dx, dy, r2) and an
    (N+3) x (N+3) augmented system.  Tests monkeypatch the
    available-memory helper so we can simulate "too big" without
    actually allocating gigabytes.
    """

    def test_kernel_block_exceeds_memory(self, monkeypatch):
        """Large N triggers the kernel-block guard."""
        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 8 * 1024 ** 2,  # 8 MB
        )

        # N=600 -> kernel_bytes = 4 * 600**2 * 8 = ~11.5 MB > 8*0.8 MB.
        n = 600
        rng = np.random.RandomState(0)
        x = rng.uniform(0, 10, n)
        y = rng.uniform(0, 10, n)
        z = rng.uniform(0, 10, n)
        template = _make_template([0.0, 1.0], [0.0, 1.0])

        with pytest.raises(MemoryError, match='kernel block K'):
            spline(x, y, z, template)

    def test_grid_size_irrelevant_to_guard(self, monkeypatch):
        """Guard fires on N alone; grid pixels do not enter the estimate.

        TPS evaluates the spline cell-by-cell inside a numba kernel,
        so no grid x N prediction matrix is materialized.  A huge grid
        with a small N must still pass the guard.
        """
        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 8 * 1024 ** 2,  # 8 MB
        )

        x = np.array([0.0, 1.0, 2.0, 0.5])
        y = np.array([0.0, 0.0, 1.0, 1.5])
        z = np.array([1.0, 2.0, 3.0, 4.0])
        # Big grid, but N is tiny -> guard must let it through.
        template = _make_template(
            np.arange(1000, dtype=np.float64),
            np.arange(1000, dtype=np.float64),
        )
        result = spline(x, y, z, template)
        assert result.shape == template.shape

    def test_small_input_allowed(self, monkeypatch):
        """Tiny inputs pass the guard even with very low available memory."""
        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 16 * 1024 ** 2,  # 16 MB
        )

        x, y, z = _grid_points()
        template = _make_template([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        # Should not raise.
        result = spline(x, y, z, template)
        assert result.shape == template.shape

    def test_check_helper_estimate_message(self, monkeypatch):
        """_check_spline_memory reports GB and identifies the culprit."""
        from xrspatial.interpolate._spline import _check_spline_memory

        monkeypatch.setattr(
            'xrspatial.zonal._available_memory_bytes',
            lambda: 1 * 1024 ** 2,  # 1 MB
        )

        # N=500 -> kernel_bytes = 4 * 500**2 * 8 = 8 MB > 1*0.8 MB.
        with pytest.raises(MemoryError, match='kernel block K'):
            _check_spline_memory(n_points=500, grid_pixels=100)
