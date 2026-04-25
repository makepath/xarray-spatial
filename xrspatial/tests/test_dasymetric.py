"""Tests for xrspatial.dasymetric."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.dasymetric import (
    disaggregate,
    pycnophylactic,
    validate_disaggregation,
)
from xrspatial.tests.general_checks import (
    create_test_raster,
    has_cuda_and_cupy,
)
from xrspatial.utils import has_dask_array


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_zones_data():
    """3x4 raster with 3 zones (1, 2, 3)."""
    return np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 3, 3],
    ], dtype=np.float64)


@pytest.fixture
def simple_weight_data():
    """3x4 raster of varying weights."""
    return np.array([
        [1.0, 3.0, 2.0, 2.0],
        [2.0, 4.0, 1.0, 1.0],
        [0.5, 0.5, 1.0, 2.0],
    ], dtype=np.float64)


@pytest.fixture
def simple_values():
    return {1: 100.0, 2: 200.0, 3: 40.0}


@pytest.fixture
def simple_zones(simple_zones_data):
    return create_test_raster(simple_zones_data, backend='numpy',
                              chunks=(2, 2))


@pytest.fixture
def simple_weight(simple_weight_data):
    return create_test_raster(simple_weight_data, backend='numpy',
                              chunks=(2, 2))


# ---------------------------------------------------------------------------
# TestWeighted
# ---------------------------------------------------------------------------

class TestWeighted:
    def test_known_values(self, simple_zones, simple_weight, simple_values):
        result = disaggregate(simple_zones, simple_values, simple_weight,
                              method='weighted')
        data = result.values

        # zone 1 weight sum = 1+3+2+4 = 10
        assert data[0, 0] == pytest.approx(100.0 * 1.0 / 10.0)  # 10
        assert data[0, 1] == pytest.approx(100.0 * 3.0 / 10.0)  # 30
        assert data[1, 0] == pytest.approx(100.0 * 2.0 / 10.0)  # 20
        assert data[1, 1] == pytest.approx(100.0 * 4.0 / 10.0)  # 40

        # zone 2 weight sum = 2+2+1+1 = 6
        assert data[0, 2] == pytest.approx(200.0 * 2.0 / 6.0)
        assert data[0, 3] == pytest.approx(200.0 * 2.0 / 6.0)

        # zone 3 weight sum = 0.5+0.5+1+2 = 4
        assert data[2, 0] == pytest.approx(40.0 * 0.5 / 4.0)  # 5
        assert data[2, 3] == pytest.approx(40.0 * 2.0 / 4.0)  # 20

    def test_conservation(self, simple_zones, simple_weight, simple_values):
        """Sum of result pixels per zone should equal input zone value."""
        result = disaggregate(simple_zones, simple_values, simple_weight)
        data = result.values
        zones = simple_zones.values

        for zid, zval in simple_values.items():
            total = np.nansum(data[zones == zid])
            assert total == pytest.approx(zval, rel=1e-10)

    def test_uniform_weights(self, simple_zones, simple_values):
        """Uniform weights should distribute equally."""
        uniform = create_test_raster(
            np.ones((3, 4), dtype=np.float64), backend='numpy')
        result = disaggregate(simple_zones, simple_values, uniform)
        data = result.values
        zones = simple_zones.values

        # zone 1 has 4 pixels -> each gets 100/4 = 25
        np.testing.assert_allclose(data[zones == 1], 25.0)
        # zone 2 has 4 pixels -> each gets 200/4 = 50
        np.testing.assert_allclose(data[zones == 2], 50.0)
        # zone 3 has 4 pixels -> each gets 40/4 = 10
        np.testing.assert_allclose(data[zones == 3], 10.0)

    def test_output_metadata(self, simple_zones, simple_weight, simple_values):
        """Output should preserve dims, coords, and attrs from zones."""
        result = disaggregate(simple_zones, simple_values, simple_weight,
                              name='pop')
        assert result.dims == simple_zones.dims
        assert result.name == 'pop'
        assert result.shape == simple_zones.shape


# ---------------------------------------------------------------------------
# TestBinary
# ---------------------------------------------------------------------------

class TestBinary:
    def test_known_values(self, simple_zones_data, simple_values):
        """Binary should split equally among nonzero-weight pixels."""
        # weight with a zero
        weight_data = np.array([
            [1.0, 0.0, 5.0, 3.0],
            [2.0, 7.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ], dtype=np.float64)
        zones = create_test_raster(simple_zones_data, backend='numpy')
        weight = create_test_raster(weight_data, backend='numpy')

        result = disaggregate(zones, simple_values, weight, method='binary')
        data = result.values

        # zone 1: 3 nonzero pixels (skip [0,1] which is 0) -> 100/3
        z1_mask = (simple_zones_data == 1) & (weight_data != 0)
        np.testing.assert_allclose(data[z1_mask], 100.0 / 3.0)
        assert data[0, 1] == pytest.approx(0.0)  # zero-weight pixel

        # zone 2: 3 nonzero pixels -> 200/3
        z2_mask = (simple_zones_data == 2) & (weight_data != 0)
        np.testing.assert_allclose(data[z2_mask], 200.0 / 3.0)

        # zone 3: 3 nonzero pixels -> 40/3
        z3_mask = (simple_zones_data == 3) & (weight_data != 0)
        np.testing.assert_allclose(data[z3_mask], 40.0 / 3.0)

    def test_all_nonzero(self, simple_zones, simple_weight, simple_values):
        """All weights nonzero: binary = equal split."""
        result = disaggregate(simple_zones, simple_values, simple_weight,
                              method='binary')
        data = result.values
        zones = simple_zones.values

        # zone 1: 4 pixels -> 100/4 = 25
        np.testing.assert_allclose(data[zones == 1], 25.0)

    def test_all_zero_zone(self, simple_zones_data, simple_values):
        """Zone where all weights are zero should produce NaN."""
        weight_data = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],  # zone 3 all zero
        ], dtype=np.float64)
        zones = create_test_raster(simple_zones_data, backend='numpy')
        weight = create_test_raster(weight_data, backend='numpy')

        result = disaggregate(zones, simple_values, weight, method='binary')
        data = result.values
        # zone 3 should be all NaN
        assert np.all(np.isnan(data[2, :]))


# ---------------------------------------------------------------------------
# TestLimitingVariable
# ---------------------------------------------------------------------------

class TestLimitingVariable:
    def test_basic_numpy(self, simple_zones, simple_weight, simple_values):
        """Limiting variable should run on numpy without error."""
        result = disaggregate(simple_zones, simple_values, simple_weight,
                              method='limiting_variable')
        data = result.values
        zones = simple_zones.values

        # conservation should hold
        for zid, zval in simple_values.items():
            total = np.nansum(data[zones == zid])
            assert total == pytest.approx(zval, rel=1e-10)

    def test_zero_weight_pixels_get_zero(self):
        """Zero-weight pixels must receive zero population, not absorb it."""
        zones_data = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 3, 3],
        ], dtype=np.float64)
        weight_data = np.array([
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ], dtype=np.float64)
        zones = create_test_raster(zones_data, backend='numpy')
        weight = create_test_raster(weight_data, backend='numpy')
        values = {1: 100.0, 2: 200.0, 3: 30.0}

        result = disaggregate(zones, values, weight,
                              method='limiting_variable')
        data = result.values

        # zero-weight pixels must be zero
        assert data[0, 0] == pytest.approx(0.0)  # zone 1 zero-weight
        assert data[1, 2] == pytest.approx(0.0)  # zone 2 zero-weight
        assert data[2, 3] == pytest.approx(0.0)  # zone 3 zero-weight

        # nonzero-weight pixels must receive population
        assert data[0, 1] > 0  # zone 1 habitable
        assert data[0, 2] > 0  # zone 2 habitable
        assert data[2, 0] > 0  # zone 3 habitable

        # conservation
        for zid, zval in values.items():
            total = np.nansum(data[zones_data == zid])
            assert total == pytest.approx(zval, rel=1e-10)

    def test_custom_density_caps(self):
        """Custom density caps should limit per-pixel allocation."""
        zones_data = np.array([[1, 1, 1, 1]], dtype=np.float64)
        # class 0: weight <= 0  (1 pixel)
        # class 1: weight > 0   (3 pixels)
        weight_data = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
        zones = create_test_raster(zones_data, backend='numpy')
        weight = create_test_raster(weight_data, backend='numpy')

        # cap class 1 at 10 per pixel -> 3 pixels * 10 = 30 max
        # zone value = 100 -> 30 allocated, 70 leftover -> 70/3 added
        result = disaggregate(zones, {1: 100.0}, weight,
                              method='limiting_variable',
                              class_breaks=(0.0,),
                              density_caps=[0.0, 10.0])
        data = result.values

        # zero-weight pixel gets 0
        assert data[0, 0] == pytest.approx(0.0)
        # class 1 pixels: 10 + 70/3 each
        expected = 10.0 + 70.0 / 3.0
        assert data[0, 1] == pytest.approx(expected)
        assert data[0, 2] == pytest.approx(expected)
        assert data[0, 3] == pytest.approx(expected)
        # conservation
        assert np.nansum(data) == pytest.approx(100.0)

    @pytest.mark.skipif(not has_dask_array(), reason="dask not available")
    def test_raises_for_dask(self, simple_zones_data, simple_weight_data,
                             simple_values):
        zones = create_test_raster(simple_zones_data, backend='dask+numpy',
                                   chunks=(2, 2))
        weight = create_test_raster(simple_weight_data, backend='dask+numpy',
                                    chunks=(2, 2))
        with pytest.raises(NotImplementedError):
            disaggregate(zones, simple_values, weight,
                         method='limiting_variable')

    @pytest.mark.skipif(not has_cuda_and_cupy(),
                        reason="CUDA/CuPy not available")
    def test_raises_for_cupy(self, simple_zones_data, simple_weight_data,
                             simple_values):
        zones = create_test_raster(simple_zones_data, backend='cupy')
        weight = create_test_raster(simple_weight_data, backend='cupy')
        with pytest.raises(NotImplementedError):
            disaggregate(zones, simple_values, weight,
                         method='limiting_variable')


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_missing_zone_id(self, simple_zones, simple_weight):
        """Zone present in raster but absent from values -> NaN."""
        values = {1: 100.0}  # zones 2 and 3 not in values
        result = disaggregate(simple_zones, values, simple_weight)
        data = result.values
        zones = simple_zones.values
        # zone 1 should have values
        assert not np.any(np.isnan(data[zones == 1]))
        # zones 2, 3 should be NaN
        assert np.all(np.isnan(data[zones == 2]))
        assert np.all(np.isnan(data[zones == 3]))

    def test_nan_zones(self, simple_weight_data, simple_values):
        """NaN in zones raster -> NaN output."""
        zones_data = np.array([
            [1, 1, 2, 2],
            [1, np.nan, 2, 2],
            [3, 3, 3, 3],
        ], dtype=np.float64)
        zones = create_test_raster(zones_data, backend='numpy')
        weight = create_test_raster(simple_weight_data, backend='numpy')
        result = disaggregate(zones, simple_values, weight)
        assert np.isnan(result.values[1, 1])

    def test_nan_weights(self, simple_zones_data, simple_values):
        """NaN in weight raster -> NaN for that pixel."""
        weight_data = np.array([
            [1.0, np.nan, 2.0, 2.0],
            [2.0, 4.0, 1.0, 1.0],
            [0.5, 0.5, 1.0, 2.0],
        ], dtype=np.float64)
        zones = create_test_raster(simple_zones_data, backend='numpy')
        weight = create_test_raster(weight_data, backend='numpy')
        result = disaggregate(zones, simple_values, weight)
        assert np.isnan(result.values[0, 1])

    def test_zero_sum_zone(self, simple_zones_data, simple_values):
        """Zone with all-zero weights -> NaN."""
        weight_data = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float64)
        zones = create_test_raster(simple_zones_data, backend='numpy')
        weight = create_test_raster(weight_data, backend='numpy')
        result = disaggregate(zones, simple_values, weight)
        data = result.values
        # zone 1 has all-zero weights -> NaN
        assert np.all(np.isnan(data[simple_zones_data == 1]))

    def test_negative_weights(self, simple_zones, simple_values):
        """Negative weights should be clamped to 0."""
        weight_data = np.array([
            [-1.0, 3.0, 2.0, 2.0],
            [2.0, 4.0, 1.0, 1.0],
            [0.5, 0.5, 1.0, 2.0],
        ], dtype=np.float64)
        weight = create_test_raster(weight_data, backend='numpy')
        result = disaggregate(simple_zones, simple_values, weight)
        data = result.values
        # pixel [0,0] had negative weight -> clamped to 0 -> gets 0 value
        assert data[0, 0] == pytest.approx(0.0)

    def test_nodata_zone(self, simple_zones_data, simple_weight_data,
                         simple_values):
        """Pixels matching nodata_zone should be NaN."""
        zones = create_test_raster(simple_zones_data, backend='numpy')
        weight = create_test_raster(simple_weight_data, backend='numpy')
        result = disaggregate(zones, simple_values, weight, nodata_zone=3)
        data = result.values
        # zone 3 should be NaN
        assert np.all(np.isnan(data[simple_zones_data == 3]))
        # zones 1 and 2 should still have values
        assert not np.any(np.isnan(data[simple_zones_data == 1]))

    def test_single_pixel_zone(self):
        """Single-pixel zone should get the full zone value."""
        zones_data = np.array([[1, 2]], dtype=np.float64)
        weight_data = np.array([[5.0, 3.0]], dtype=np.float64)
        zones = create_test_raster(zones_data, backend='numpy')
        weight = create_test_raster(weight_data, backend='numpy')
        result = disaggregate(zones, {1: 100.0, 2: 200.0}, weight)
        assert result.values[0, 0] == pytest.approx(100.0)
        assert result.values[0, 1] == pytest.approx(200.0)

    def test_series_input(self, simple_zones, simple_weight):
        """pd.Series input should work like dict."""
        values_dict = {1: 100.0, 2: 200.0, 3: 40.0}
        values_series = pd.Series(values_dict)
        result_dict = disaggregate(simple_zones, values_dict, simple_weight)
        result_series = disaggregate(simple_zones, values_series, simple_weight)
        np.testing.assert_allclose(result_dict.values, result_series.values,
                                   equal_nan=True)

    def test_extra_key_silently_ignored(self, simple_zones, simple_weight):
        """Value keys absent from zones raster should be silently ignored."""
        values = {1: 100.0, 2: 200.0, 3: 40.0, 999: 5000.0}
        result = disaggregate(simple_zones, values, simple_weight)
        # should not raise; zone 999 doesn't exist, just ignore
        assert result.shape == simple_zones.shape


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_zones_not_dataarray(self, simple_weight, simple_values):
        with pytest.raises(TypeError, match="zones"):
            disaggregate(np.ones((3, 4)), simple_values, simple_weight)

    def test_weight_not_dataarray(self, simple_zones, simple_values):
        with pytest.raises(TypeError, match="weight"):
            disaggregate(simple_zones, simple_values, np.ones((3, 4)))

    def test_zones_wrong_ndim(self, simple_weight, simple_values):
        zones_3d = xr.DataArray(np.ones((2, 3, 4)), dims=['z', 'y', 'x'])
        with pytest.raises(ValueError, match="2-D"):
            disaggregate(zones_3d, simple_values, simple_weight)

    def test_weight_wrong_ndim(self, simple_zones, simple_values):
        weight_1d = xr.DataArray(np.ones(12), dims=['x'])
        with pytest.raises(ValueError, match="2-D"):
            disaggregate(simple_zones, simple_values, weight_1d)

    def test_shape_mismatch(self, simple_zones, simple_values):
        weight_wrong = xr.DataArray(np.ones((5, 5)), dims=['y', 'x'])
        with pytest.raises(ValueError, match="same shape"):
            disaggregate(simple_zones, simple_values, weight_wrong)

    def test_invalid_method(self, simple_zones, simple_weight, simple_values):
        with pytest.raises(ValueError, match="Invalid method"):
            disaggregate(simple_zones, simple_values, simple_weight,
                         method='invalid')

    def test_values_wrong_type(self, simple_zones, simple_weight):
        with pytest.raises(TypeError, match="dict or pd.Series"):
            disaggregate(simple_zones, [100, 200], simple_weight)


# ---------------------------------------------------------------------------
# TestCrossBackend
# ---------------------------------------------------------------------------

class TestCrossBackend:
    @pytest.mark.skipif(not has_dask_array(), reason="dask not available")
    @pytest.mark.parametrize("method", ['weighted', 'binary'])
    def test_numpy_equals_dask_numpy(self, simple_zones_data,
                                     simple_weight_data, simple_values,
                                     method):
        zones_np = create_test_raster(simple_zones_data, backend='numpy')
        weight_np = create_test_raster(simple_weight_data, backend='numpy')
        zones_dk = create_test_raster(simple_zones_data, backend='dask+numpy',
                                      chunks=(2, 2))
        weight_dk = create_test_raster(simple_weight_data,
                                       backend='dask+numpy', chunks=(2, 2))

        result_np = disaggregate(zones_np, simple_values, weight_np,
                                 method=method)
        result_dk = disaggregate(zones_dk, simple_values, weight_dk,
                                 method=method)

        assert isinstance(result_dk.data, da.Array)
        np.testing.assert_allclose(result_np.values, result_dk.values,
                                   equal_nan=True)

    @pytest.mark.skipif(not has_dask_array(), reason="dask not available")
    def test_dask_conservation(self, simple_zones_data, simple_weight_data,
                               simple_values):
        zones_dk = create_test_raster(simple_zones_data, backend='dask+numpy',
                                      chunks=(2, 2))
        weight_dk = create_test_raster(simple_weight_data,
                                       backend='dask+numpy', chunks=(2, 2))
        result = disaggregate(zones_dk, simple_values, weight_dk)
        data = result.values
        zones = simple_zones_data

        for zid, zval in simple_values.items():
            total = np.nansum(data[zones == zid])
            assert total == pytest.approx(zval, rel=1e-10)

    @pytest.mark.skipif(not has_cuda_and_cupy(),
                        reason="CUDA/CuPy not available")
    @pytest.mark.parametrize("method", ['weighted', 'binary'])
    def test_numpy_equals_cupy(self, simple_zones_data, simple_weight_data,
                               simple_values, method):
        zones_np = create_test_raster(simple_zones_data, backend='numpy')
        weight_np = create_test_raster(simple_weight_data, backend='numpy')
        zones_cu = create_test_raster(simple_zones_data, backend='cupy')
        weight_cu = create_test_raster(simple_weight_data, backend='cupy')

        result_np = disaggregate(zones_np, simple_values, weight_np,
                                 method=method)
        result_cu = disaggregate(zones_cu, simple_values, weight_cu,
                                 method=method)

        np.testing.assert_allclose(result_np.values,
                                   result_cu.data.get(),
                                   equal_nan=True)

    @pytest.mark.skipif(
        not (has_cuda_and_cupy() and has_dask_array()),
        reason="CUDA/CuPy and dask not available")
    @pytest.mark.parametrize("method", ['weighted', 'binary'])
    def test_numpy_equals_dask_cupy(self, simple_zones_data,
                                    simple_weight_data, simple_values,
                                    method):
        zones_np = create_test_raster(simple_zones_data, backend='numpy')
        weight_np = create_test_raster(simple_weight_data, backend='numpy')
        zones_dc = create_test_raster(simple_zones_data,
                                      backend='dask+cupy', chunks=(2, 2))
        weight_dc = create_test_raster(simple_weight_data,
                                       backend='dask+cupy', chunks=(2, 2))

        result_np = disaggregate(zones_np, simple_values, weight_np,
                                 method=method)
        result_dc = disaggregate(zones_dc, simple_values, weight_dc,
                                 method=method)

        np.testing.assert_allclose(result_np.values,
                                   result_dc.values,
                                   equal_nan=True)


# ---------------------------------------------------------------------------
# TestPycnophylactic
# ---------------------------------------------------------------------------

class TestPycnophylactic:
    def test_conservation(self, simple_zones, simple_values):
        """Zone totals must be preserved."""
        result = pycnophylactic(simple_zones, simple_values)
        data = result.values
        zones = simple_zones.values

        for zid, zval in simple_values.items():
            total = np.nansum(data[zones == zid])
            assert total == pytest.approx(zval, rel=1e-10)

    def test_output_shape_and_metadata(self, simple_zones, simple_values):
        result = pycnophylactic(simple_zones, simple_values, name='smooth')
        assert result.shape == simple_zones.shape
        assert result.dims == simple_zones.dims
        assert result.name == 'smooth'

    def test_single_zone_uniform(self):
        """A single zone should converge to a uniform surface."""
        zones_data = np.ones((5, 5), dtype=np.float64)
        zones = create_test_raster(zones_data, backend='numpy')
        result = pycnophylactic(zones, {1: 100.0}, convergence=1e-10)
        # all pixels should be 100/25 = 4.0
        np.testing.assert_allclose(result.values, 4.0, atol=1e-8)

    def test_nodata_zone(self, simple_zones, simple_values):
        """Nodata zone pixels should be NaN."""
        result = pycnophylactic(simple_zones, simple_values, nodata_zone=3)
        data = result.values
        zones = simple_zones.values
        assert np.all(np.isnan(data[zones == 3]))
        # other zones still conserve
        for zid in [1, 2]:
            total = np.nansum(data[zones == zid])
            assert total == pytest.approx(simple_values[zid], rel=1e-10)

    def test_smoothness(self):
        """Result should be smoother than uniform initial -- check that
        adjacent pixels within a zone don't have wild jumps."""
        zones_data = np.array([
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 1, 2, 2, 2, 2],
        ], dtype=np.float64)
        zones = create_test_raster(zones_data, backend='numpy')
        result = pycnophylactic(zones, {1: 160.0, 2: 320.0},
                                max_iterations=200, convergence=1e-8)
        data = result.values
        # pixel-to-pixel differences should be small
        dy = np.abs(np.diff(data, axis=0))
        dx = np.abs(np.diff(data, axis=1))
        # max gradient should be modest relative to the mean value
        mean_val = np.nanmean(data)
        assert np.nanmax(dy) < mean_val
        assert np.nanmax(dx) < mean_val

    def test_missing_zone_produces_nan(self, simple_zones):
        """Zones absent from values dict should be NaN in output."""
        result = pycnophylactic(simple_zones, {1: 100.0})
        data = result.values
        zones = simple_zones.values
        assert not np.any(np.isnan(data[zones == 1]))
        assert np.all(np.isnan(data[zones == 2]))
        assert np.all(np.isnan(data[zones == 3]))

    def test_series_input(self, simple_zones, simple_values):
        """pd.Series should work like dict."""
        result_dict = pycnophylactic(simple_zones, simple_values)
        result_series = pycnophylactic(simple_zones, pd.Series(simple_values))
        np.testing.assert_allclose(result_dict.values, result_series.values,
                                   equal_nan=True)

    @pytest.mark.skipif(not has_dask_array(), reason="dask not available")
    def test_raises_for_dask(self, simple_zones_data, simple_values):
        zones = create_test_raster(simple_zones_data, backend='dask+numpy',
                                   chunks=(2, 2))
        with pytest.raises(NotImplementedError, match="dask"):
            pycnophylactic(zones, simple_values)

    @pytest.mark.skipif(not has_cuda_and_cupy(),
                        reason="CUDA/CuPy not available")
    def test_cupy_matches_numpy(self, simple_zones_data, simple_values):
        zones_np = create_test_raster(simple_zones_data, backend='numpy')
        zones_cu = create_test_raster(simple_zones_data, backend='cupy')
        result_np = pycnophylactic(zones_np, simple_values)
        result_cu = pycnophylactic(zones_cu, simple_values)
        np.testing.assert_allclose(result_np.values, result_cu.data.get(),
                                   equal_nan=True)

    def test_validation(self):
        """Wrong types and dims should raise."""
        with pytest.raises(TypeError, match="zones"):
            pycnophylactic(np.ones((3, 4)), {1: 100.0})
        zones_3d = xr.DataArray(np.ones((2, 3, 4)), dims=['z', 'y', 'x'])
        with pytest.raises(ValueError, match="2-D"):
            pycnophylactic(zones_3d, {1: 100.0})


# ---------------------------------------------------------------------------
# TestValidateDisaggregation
# ---------------------------------------------------------------------------

class TestValidateDisaggregation:
    def test_valid_result(self, simple_zones, simple_weight, simple_values):
        result = disaggregate(simple_zones, simple_values, simple_weight)
        assert validate_disaggregation(result, simple_zones, simple_values)

    def test_valid_pycnophylactic(self, simple_zones, simple_values):
        result = pycnophylactic(simple_zones, simple_values)
        assert validate_disaggregation(result, simple_zones, simple_values)

    def test_invalid_result_raises(self, simple_zones, simple_values):
        """Manually corrupted result should fail validation."""
        zones_data = simple_zones.values
        bad_data = np.ones_like(zones_data, dtype=np.float64)
        bad_result = xr.DataArray(bad_data, dims=simple_zones.dims,
                                  coords=simple_zones.coords)
        with pytest.raises(ValueError, match="Zone totals not preserved"):
            validate_disaggregation(bad_result, simple_zones, simple_values)

    def test_skips_all_nan_zones(self, simple_zones_data, simple_values):
        """Zones that are all NaN in result should be skipped."""
        # make zone 1 all-zero weights -> all NaN in result
        weight_data = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float64)
        zones = create_test_raster(simple_zones_data, backend='numpy')
        weight = create_test_raster(weight_data, backend='numpy')
        result = disaggregate(zones, simple_values, weight)
        # zone 1 is all NaN, but validation should still pass
        # (only zones 2 and 3 are checked)
        assert validate_disaggregation(result, zones, simple_values)

    def test_tight_tolerance(self, simple_zones, simple_weight, simple_values):
        """With tolerance=0, even floating-point noise could fail."""
        result = disaggregate(simple_zones, simple_values, simple_weight)
        # tolerance=1e-15 should still pass for exact arithmetic
        assert validate_disaggregation(result, simple_zones, simple_values,
                                       tolerance=1e-10)

    def test_series_values(self, simple_zones, simple_weight):
        """pd.Series values should work."""
        values = pd.Series({1: 100.0, 2: 200.0, 3: 40.0})
        result = disaggregate(simple_zones, values, simple_weight)
        assert validate_disaggregation(result, simple_zones, values)

    @pytest.mark.skipif(not has_dask_array(), reason="dask not available")
    def test_dask_result(self, simple_zones_data, simple_weight_data,
                         simple_values):
        """Should handle dask-backed result arrays."""
        zones = create_test_raster(simple_zones_data, backend='dask+numpy',
                                   chunks=(2, 2))
        weight = create_test_raster(simple_weight_data, backend='dask+numpy',
                                    chunks=(2, 2))
        result = disaggregate(zones, simple_values, weight)
        zones_np = create_test_raster(simple_zones_data, backend='numpy')
        assert validate_disaggregation(result, zones_np, simple_values)

    @pytest.mark.skipif(not has_cuda_and_cupy(),
                        reason="CUDA/CuPy not available")
    def test_cupy_result(self, simple_zones_data, simple_weight_data,
                         simple_values):
        """Should handle cupy-backed result arrays."""
        zones = create_test_raster(simple_zones_data, backend='cupy')
        weight = create_test_raster(simple_weight_data, backend='cupy')
        result = disaggregate(zones, simple_values, weight)
        assert validate_disaggregation(result, zones, simple_values)


# ---------------------------------------------------------------------------
# TestMemoryGuard (issue #1261)
# ---------------------------------------------------------------------------

class TestMemoryGuard:
    """Memory guards on disaggregate() and pycnophylactic() (#1261).

    Both public functions now estimate peak working memory and raise
    ``MemoryError`` before the first allocation when the projected
    footprint exceeds 50% of available RAM.  pycnophylactic in particular
    keeps one bool mask per zone alive, so the budget grows with the
    number of zones in ``values``.
    """

    def _stub_available(self, monkeypatch, n_bytes):
        """Pin _available_memory_bytes to a fixed return value."""
        import sys
        mod = sys.modules['xrspatial.dasymetric']
        monkeypatch.setattr(mod, '_available_memory_bytes', lambda: n_bytes)

    def test_disaggregate_weighted_raises_when_budget_exceeded(
        self, monkeypatch
    ):
        """weighted method should refuse a raster too large for memory."""
        # ~3 GB worth of working memory required, but only 64 MB available
        self._stub_available(monkeypatch, 64 * 1024 * 1024)
        zones = xr.DataArray(
            np.ones((20000, 20000), dtype=np.float64), dims=['y', 'x'],
        )
        weight = xr.DataArray(
            np.ones((20000, 20000), dtype=np.float64), dims=['y', 'x'],
        )
        with pytest.raises(MemoryError, match=r"disaggregate.*weighted"):
            disaggregate(zones, {1: 100.0}, weight)

    def test_disaggregate_limiting_variable_raises_when_budget_exceeded(
        self, monkeypatch
    ):
        """limiting_variable method has higher per-pixel cost; same guard."""
        self._stub_available(monkeypatch, 64 * 1024 * 1024)
        zones = xr.DataArray(
            np.ones((20000, 20000), dtype=np.float64), dims=['y', 'x'],
        )
        weight = xr.DataArray(
            np.ones((20000, 20000), dtype=np.float64), dims=['y', 'x'],
        )
        with pytest.raises(MemoryError,
                           match=r"disaggregate.*limiting_variable"):
            disaggregate(zones, {1: 100.0}, weight,
                         method='limiting_variable')

    def test_disaggregate_passes_when_budget_ample(
        self, simple_zones, simple_weight, simple_values, monkeypatch
    ):
        """Small rasters should pass the guard with normal RAM headroom."""
        self._stub_available(monkeypatch, 64 * 1024 ** 3)
        result = disaggregate(simple_zones, simple_values, simple_weight)
        assert result.shape == simple_zones.shape

    def test_pycnophylactic_raises_when_zone_count_explodes(
        self, monkeypatch
    ):
        """Per-zone mask budget should reject many-zone inputs early.

        With 1000 zones and a 10000x10000 raster the masks alone come to
        ~100 GB; the guard should fire before any allocation when only
        2 GB are available.
        """
        self._stub_available(monkeypatch, 2 * 1024 ** 3)
        zones_data = np.tile(np.arange(1, 1001, dtype=np.float64),
                             (10000, 10))
        zones = xr.DataArray(zones_data, dims=['y', 'x'])
        values = {i: 100.0 for i in range(1, 1001)}
        with pytest.raises(MemoryError,
                           match=r"pycnophylactic.*n_zones=1000"):
            pycnophylactic(zones, values)

    def test_pycnophylactic_raises_when_raster_too_large(
        self, monkeypatch
    ):
        """Even with few zones, a giant raster should be rejected."""
        self._stub_available(monkeypatch, 64 * 1024 * 1024)
        zones = xr.DataArray(
            np.ones((20000, 20000), dtype=np.float64), dims=['y', 'x'],
        )
        with pytest.raises(MemoryError, match=r"pycnophylactic"):
            pycnophylactic(zones, {1: 100.0})

    def test_pycnophylactic_passes_when_budget_ample(
        self, simple_zones, simple_values, monkeypatch
    ):
        """Small inputs should pass the guard with normal RAM headroom."""
        self._stub_available(monkeypatch, 64 * 1024 ** 3)
        result = pycnophylactic(simple_zones, simple_values)
        assert result.shape == simple_zones.shape

    def test_disaggregate_dask_skips_in_ram_guard(
        self, simple_zones_data, simple_weight_data, simple_values,
        monkeypatch
    ):
        """Dask-backed inputs should not be blocked by the in-RAM guard.

        The numpy-only guard targets backends that materialize the
        full array; dask paths process per-chunk and budget themselves.
        """
        if not has_dask_array():
            pytest.skip("dask not available")
        # Even with 0 bytes "available", dask path should succeed because
        # the in-RAM guard is skipped for dask.Array inputs.
        self._stub_available(monkeypatch, 1)
        zones = create_test_raster(simple_zones_data,
                                   backend='dask+numpy', chunks=(2, 2))
        weight = create_test_raster(simple_weight_data,
                                    backend='dask+numpy', chunks=(2, 2))
        result = disaggregate(zones, simple_values, weight)
        assert result.shape == zones.shape

    def test_pycnophylactic_error_mentions_n_zones(self, monkeypatch):
        """Error message should surface the zone count for diagnostics."""
        self._stub_available(monkeypatch, 32 * 1024 * 1024)
        zones = xr.DataArray(
            np.ones((5000, 5000), dtype=np.float64), dims=['y', 'x'],
        )
        values = {i: 1.0 for i in range(1, 51)}
        with pytest.raises(MemoryError, match="n_zones=50"):
            pycnophylactic(zones, values)
