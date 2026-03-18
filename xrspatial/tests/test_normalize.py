"""Tests for xrspatial.normalize (rescale and standardize)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.normalize import rescale, standardize
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
    general_output_checks,
)


# ---- helpers ---------------------------------------------------------------

def _make_data():
    """6x4 float64 array with NaN in first row and an inf."""
    return np.array([
        [np.nan, np.nan, np.nan, np.nan],
        [10.,    20.,    30.,    40.],
        [50.,    60.,    70.,    80.],
        [0.,      5.,    15.,    25.],
        [35.,    45.,    55.,    65.],
        [90.,   100.,   np.inf,  75.],
    ], dtype=np.float64)


# ---- rescale ---------------------------------------------------------------

class TestRescaleNumpy:

    def test_default_range(self):
        data = _make_data()
        agg = create_test_raster(data, backend='numpy')
        result = rescale(agg)
        general_output_checks(agg, result, verify_attrs=True)

        vals = result.values
        finite = vals[np.isfinite(vals)]
        assert np.nanmin(finite) == pytest.approx(0.0)
        assert np.nanmax(finite) == pytest.approx(1.0)

    def test_custom_range(self):
        data = _make_data()
        agg = create_test_raster(data, backend='numpy')
        result = rescale(agg, new_min=0.0, new_max=255.0)
        vals = result.values
        finite = vals[np.isfinite(vals)]
        assert np.nanmin(finite) == pytest.approx(0.0)
        assert np.nanmax(finite) == pytest.approx(255.0)

    def test_nan_passthrough(self):
        data = _make_data()
        agg = create_test_raster(data, backend='numpy')
        result = rescale(agg)
        # First row was all NaN.
        assert np.all(np.isnan(result.values[0, :]))

    def test_inf_passthrough(self):
        data = _make_data()
        agg = create_test_raster(data, backend='numpy')
        result = rescale(agg)
        # inf stays as nan in output (not finite -> nan).
        assert np.isnan(result.values[5, 2])

    def test_constant_raster(self):
        data = np.full((4, 4), 42.0)
        agg = create_test_raster(data, backend='numpy')
        result = rescale(agg, new_min=0.0, new_max=1.0)
        # All values equal -> output should be new_min.
        np.testing.assert_allclose(result.values, 0.0)

    def test_all_nan(self):
        data = np.full((3, 3), np.nan)
        agg = create_test_raster(data, backend='numpy')
        result = rescale(agg)
        assert np.all(np.isnan(result.values))

    def test_single_cell(self):
        data = np.array([[7.0]])
        agg = create_test_raster(data, backend='numpy')
        result = rescale(agg)
        # Single finite value, range=0 -> output = new_min = 0.0
        assert result.values[0, 0] == pytest.approx(0.0)

    def test_known_values(self):
        data = np.array([
            [0., 50., 100.],
            [25., 75., np.nan],
        ], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = rescale(agg)
        expected = np.array([
            [0.0, 0.5, 1.0],
            [0.25, 0.75, np.nan],
        ])
        np.testing.assert_allclose(result.values, expected, rtol=1e-10)

    def test_new_min_gt_new_max_raises(self):
        data = np.array([[1., 2.], [3., 4.]])
        agg = create_test_raster(data, backend='numpy')
        with pytest.raises(ValueError, match="new_min.*new_max"):
            rescale(agg, new_min=10.0, new_max=0.0)

    def test_preserves_coords_attrs(self):
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        attrs = {'res': (1.0, 1.0), 'crs': 'EPSG:4326'}
        agg = create_test_raster(data, backend='numpy', attrs=attrs)
        result = rescale(agg)
        assert result.attrs == agg.attrs
        assert result.dims == agg.dims


@dask_array_available
class TestRescaleDask:

    def test_default_range(self):
        data = _make_data()
        agg = create_test_raster(data, backend='dask+numpy', chunks=(3, 4))
        result = rescale(agg)
        general_output_checks(agg, result, verify_attrs=True)

        vals = result.values
        finite = vals[np.isfinite(vals)]
        assert np.nanmin(finite) == pytest.approx(0.0)
        assert np.nanmax(finite) == pytest.approx(1.0)

    def test_custom_range(self):
        data = _make_data()
        agg = create_test_raster(data, backend='dask+numpy', chunks=(3, 4))
        result = rescale(agg, new_min=-1.0, new_max=1.0)
        vals = result.values
        finite = vals[np.isfinite(vals)]
        assert np.nanmin(finite) == pytest.approx(-1.0)
        assert np.nanmax(finite) == pytest.approx(1.0)

    def test_matches_numpy(self):
        data = _make_data()
        np_agg = create_test_raster(data, backend='numpy')
        dk_agg = create_test_raster(data, backend='dask+numpy', chunks=(3, 4))
        np_result = rescale(np_agg).values
        dk_result = rescale(dk_agg).values
        np.testing.assert_allclose(dk_result, np_result, equal_nan=True)


@cuda_and_cupy_available
class TestRescaleCuPy:

    def test_default_range(self):
        data = _make_data()
        agg = create_test_raster(data, backend='cupy')
        result = rescale(agg)
        general_output_checks(agg, result, verify_attrs=True)

        vals = result.data.get()
        finite = vals[np.isfinite(vals)]
        assert np.nanmin(finite) == pytest.approx(0.0)
        assert np.nanmax(finite) == pytest.approx(1.0)

    def test_matches_numpy(self):
        data = _make_data()
        np_agg = create_test_raster(data, backend='numpy')
        cu_agg = create_test_raster(data, backend='cupy')
        np_result = rescale(np_agg).values
        cu_result = rescale(cu_agg).data.get()
        np.testing.assert_allclose(cu_result, np_result, equal_nan=True)


# ---- standardize -----------------------------------------------------------

class TestStandardizeNumpy:

    def test_default(self):
        data = _make_data()
        agg = create_test_raster(data, backend='numpy')
        result = standardize(agg)
        general_output_checks(agg, result, verify_attrs=True)

        vals = result.values
        finite = vals[np.isfinite(vals)]
        assert np.mean(finite) == pytest.approx(0.0, abs=1e-10)
        assert np.std(finite) == pytest.approx(1.0, abs=1e-10)

    def test_ddof_1(self):
        data = np.array([
            [10., 20., 30.],
            [40., 50., 60.],
        ], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = standardize(agg, ddof=1)
        vals = result.values.ravel()
        # With ddof=1, std of output with population std != 1
        # but the formula (x-mean)/std(ddof=1) is applied consistently.
        expected_mean = np.mean(data.ravel())
        expected_std = np.std(data.ravel(), ddof=1)
        expected = (data - expected_mean) / expected_std
        np.testing.assert_allclose(result.values, expected, rtol=1e-10)

    def test_nan_passthrough(self):
        data = _make_data()
        agg = create_test_raster(data, backend='numpy')
        result = standardize(agg)
        assert np.all(np.isnan(result.values[0, :]))

    def test_inf_passthrough(self):
        data = _make_data()
        agg = create_test_raster(data, backend='numpy')
        result = standardize(agg)
        assert np.isnan(result.values[5, 2])

    def test_constant_raster(self):
        data = np.full((4, 4), 42.0)
        agg = create_test_raster(data, backend='numpy')
        result = standardize(agg)
        # std == 0 -> output should be 0.0
        np.testing.assert_allclose(result.values, 0.0)

    def test_all_nan(self):
        data = np.full((3, 3), np.nan)
        agg = create_test_raster(data, backend='numpy')
        result = standardize(agg)
        assert np.all(np.isnan(result.values))

    def test_single_cell(self):
        data = np.array([[7.0]])
        agg = create_test_raster(data, backend='numpy')
        result = standardize(agg)
        # std == 0 -> 0.0
        assert result.values[0, 0] == pytest.approx(0.0)

    def test_known_values(self):
        data = np.array([
            [1., 2., 3.],
            [4., 5., np.nan],
        ], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = standardize(agg)
        finite_data = np.array([1., 2., 3., 4., 5.])
        mean = finite_data.mean()
        std = finite_data.std(ddof=0)
        expected = np.array([
            [(1 - mean) / std, (2 - mean) / std, (3 - mean) / std],
            [(4 - mean) / std, (5 - mean) / std, np.nan],
        ])
        np.testing.assert_allclose(result.values, expected, rtol=1e-10)

    def test_invalid_ddof_raises(self):
        data = np.array([[1., 2.], [3., 4.]])
        agg = create_test_raster(data, backend='numpy')
        with pytest.raises((TypeError, ValueError)):
            standardize(agg, ddof=-1)
        with pytest.raises((TypeError, ValueError)):
            standardize(agg, ddof=1.5)

    def test_preserves_coords_attrs(self):
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        attrs = {'res': (1.0, 1.0), 'crs': 'EPSG:4326'}
        agg = create_test_raster(data, backend='numpy', attrs=attrs)
        result = standardize(agg)
        assert result.attrs == agg.attrs
        assert result.dims == agg.dims


@dask_array_available
class TestStandardizeDask:

    def test_default(self):
        data = _make_data()
        agg = create_test_raster(data, backend='dask+numpy', chunks=(3, 4))
        result = standardize(agg)
        general_output_checks(agg, result, verify_attrs=True)

        vals = result.values
        finite = vals[np.isfinite(vals)]
        assert np.mean(finite) == pytest.approx(0.0, abs=1e-10)
        assert np.std(finite) == pytest.approx(1.0, abs=1e-10)

    def test_matches_numpy(self):
        data = _make_data()
        np_agg = create_test_raster(data, backend='numpy')
        dk_agg = create_test_raster(data, backend='dask+numpy', chunks=(3, 4))
        np_result = standardize(np_agg).values
        dk_result = standardize(dk_agg).values
        np.testing.assert_allclose(dk_result, np_result, equal_nan=True)


@cuda_and_cupy_available
class TestStandardizeCuPy:

    def test_default(self):
        data = _make_data()
        agg = create_test_raster(data, backend='cupy')
        result = standardize(agg)
        general_output_checks(agg, result, verify_attrs=True)

        vals = result.data.get()
        finite = vals[np.isfinite(vals)]
        assert np.mean(finite) == pytest.approx(0.0, abs=1e-10)
        assert np.std(finite) == pytest.approx(1.0, abs=1e-10)

    def test_matches_numpy(self):
        data = _make_data()
        np_agg = create_test_raster(data, backend='numpy')
        cu_agg = create_test_raster(data, backend='cupy')
        np_result = standardize(np_agg).values
        cu_result = standardize(cu_agg).data.get()
        np.testing.assert_allclose(cu_result, np_result, equal_nan=True)
