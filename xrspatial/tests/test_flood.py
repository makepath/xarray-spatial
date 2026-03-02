"""Tests for xrspatial.flood (flood prediction tools)."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.flood import (
    curve_number_runoff,
    flood_depth,
    inundation,
    travel_time,
)
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
    general_output_checks,
)


# ===================================================================
# flood_depth
# ===================================================================

class TestFloodDepthKnownValues:

    def test_basic(self):
        # HAND=[0, 2, 5, 10], water_level=5
        # depth=[5, 3, 0, NaN]
        hand_data = np.array([[0.0, 2.0], [5.0, 10.0]], dtype=np.float64)
        hand = create_test_raster(hand_data, name='hand')
        result = flood_depth(hand, water_level=5.0)
        expected = np.array([[5.0, 3.0], [0.0, np.nan]])
        np.testing.assert_allclose(result.data, expected, equal_nan=True)

    def test_water_level_zero(self):
        # Only cells with HAND=0 are inundated (depth=0)
        hand_data = np.array([[0.0, 1.0], [0.0, 5.0]], dtype=np.float64)
        hand = create_test_raster(hand_data, name='hand')
        result = flood_depth(hand, water_level=0.0)
        expected = np.array([[0.0, np.nan], [0.0, np.nan]])
        np.testing.assert_allclose(result.data, expected, equal_nan=True)

    def test_all_inundated(self):
        hand_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        hand = create_test_raster(hand_data, name='hand')
        result = flood_depth(hand, water_level=10.0)
        expected = np.array([[9.0, 8.0], [7.0, 6.0]])
        np.testing.assert_allclose(result.data, expected)

    def test_none_inundated(self):
        hand_data = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float64)
        hand = create_test_raster(hand_data, name='hand')
        result = flood_depth(hand, water_level=2.0)
        assert np.all(np.isnan(result.data))


class TestFloodDepthNaN:

    def test_nan_propagation(self):
        hand_data = np.array([[np.nan, 2.0], [1.0, np.nan]], dtype=np.float64)
        hand = create_test_raster(hand_data, name='hand')
        result = flood_depth(hand, water_level=5.0)
        assert np.isnan(result.data[0, 0])
        assert np.isnan(result.data[1, 1])
        np.testing.assert_allclose(result.data[0, 1], 3.0)
        np.testing.assert_allclose(result.data[1, 0], 4.0)


class TestFloodDepthValidation:

    def test_negative_water_level(self):
        hand = create_test_raster(np.ones((3, 3), dtype=np.float64))
        with pytest.raises(ValueError, match="water_level must be >= 0"):
            flood_depth(hand, water_level=-1.0)

    def test_non_numeric_water_level(self):
        hand = create_test_raster(np.ones((3, 3), dtype=np.float64))
        with pytest.raises(TypeError, match="water_level must be numeric"):
            flood_depth(hand, water_level="high")


class TestFloodDepthOutput:

    def test_coords_preserved(self):
        hand_data = np.ones((4, 4), dtype=np.float64)
        hand = create_test_raster(hand_data, name='hand')
        result = flood_depth(hand, water_level=5.0)
        general_output_checks(hand, result, verify_attrs=True)


@dask_array_available
class TestFloodDepthDask:

    @pytest.mark.parametrize("chunks", [(2, 2), (3, 3)])
    def test_numpy_equals_dask(self, chunks):
        hand_data = np.array([[0.0, 2.0, 5.0],
                              [1.0, np.nan, 8.0],
                              [3.0, 7.0, 10.0]], dtype=np.float64)
        hand_np = create_test_raster(hand_data, backend='numpy', name='hand')
        hand_da = create_test_raster(hand_data, backend='dask', name='hand',
                                     chunks=chunks)
        result_np = flood_depth(hand_np, water_level=5.0)
        result_da = flood_depth(hand_da, water_level=5.0)
        general_output_checks(hand_da, result_da,
                              expected_results=result_np.data)


@cuda_and_cupy_available
class TestFloodDepthCuPy:

    def test_numpy_equals_cupy(self):
        hand_data = np.array([[0.0, 2.0], [5.0, np.nan]], dtype=np.float64)
        hand_np = create_test_raster(hand_data, backend='numpy', name='hand')
        hand_cu = create_test_raster(hand_data, backend='cupy', name='hand')
        result_np = flood_depth(hand_np, water_level=5.0)
        result_cu = flood_depth(hand_cu, water_level=5.0)
        general_output_checks(hand_cu, result_cu,
                              expected_results=result_np.data)


@cuda_and_cupy_available
@dask_array_available
class TestFloodDepthDaskCuPy:

    def test_numpy_equals_dask_cupy(self):
        hand_data = np.array([[0.0, 2.0], [5.0, np.nan]], dtype=np.float64)
        hand_np = create_test_raster(hand_data, backend='numpy', name='hand')
        hand_dc = create_test_raster(hand_data, backend='dask+cupy',
                                     name='hand', chunks=(2, 2))
        result_np = flood_depth(hand_np, water_level=5.0)
        result_dc = flood_depth(hand_dc, water_level=5.0)
        general_output_checks(hand_dc, result_dc,
                              expected_results=result_np.data)


# ===================================================================
# inundation
# ===================================================================

class TestInundationKnownValues:

    def test_basic(self):
        hand_data = np.array([[0.0, 2.0], [5.0, 10.0]], dtype=np.float64)
        hand = create_test_raster(hand_data, name='hand')
        result = inundation(hand, water_level=5.0)
        expected = np.array([[1.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(result.data, expected)

    def test_water_level_zero(self):
        hand_data = np.array([[0.0, 1.0], [0.0, 5.0]], dtype=np.float64)
        hand = create_test_raster(hand_data, name='hand')
        result = inundation(hand, water_level=0.0)
        expected = np.array([[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(result.data, expected)

    def test_all_dry(self):
        hand_data = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float64)
        hand = create_test_raster(hand_data, name='hand')
        result = inundation(hand, water_level=1.0)
        expected = np.zeros((2, 2))
        np.testing.assert_allclose(result.data, expected)


class TestInundationNaN:

    def test_nan_preserved(self):
        hand_data = np.array([[np.nan, 2.0], [1.0, np.nan]], dtype=np.float64)
        hand = create_test_raster(hand_data, name='hand')
        result = inundation(hand, water_level=5.0)
        assert np.isnan(result.data[0, 0])
        assert np.isnan(result.data[1, 1])
        np.testing.assert_allclose(result.data[0, 1], 1.0)
        np.testing.assert_allclose(result.data[1, 0], 1.0)


class TestInundationValidation:

    def test_negative_water_level(self):
        hand = create_test_raster(np.ones((3, 3), dtype=np.float64))
        with pytest.raises(ValueError, match="water_level must be >= 0"):
            inundation(hand, water_level=-1.0)


@dask_array_available
class TestInundationDask:

    def test_numpy_equals_dask(self):
        hand_data = np.array([[0.0, 2.0, 5.0],
                              [1.0, np.nan, 8.0],
                              [3.0, 7.0, 10.0]], dtype=np.float64)
        hand_np = create_test_raster(hand_data, backend='numpy', name='hand')
        hand_da = create_test_raster(hand_data, backend='dask', name='hand')
        result_np = inundation(hand_np, water_level=5.0)
        result_da = inundation(hand_da, water_level=5.0)
        general_output_checks(hand_da, result_da,
                              expected_results=result_np.data)


@cuda_and_cupy_available
class TestInundationCuPy:

    def test_numpy_equals_cupy(self):
        hand_data = np.array([[0.0, 2.0], [5.0, np.nan]], dtype=np.float64)
        hand_np = create_test_raster(hand_data, backend='numpy', name='hand')
        hand_cu = create_test_raster(hand_data, backend='cupy', name='hand')
        result_np = inundation(hand_np, water_level=5.0)
        result_cu = inundation(hand_cu, water_level=5.0)
        general_output_checks(hand_cu, result_cu,
                              expected_results=result_np.data)


@cuda_and_cupy_available
@dask_array_available
class TestInundationDaskCuPy:

    def test_numpy_equals_dask_cupy(self):
        hand_data = np.array([[0.0, 2.0], [5.0, np.nan]], dtype=np.float64)
        hand_np = create_test_raster(hand_data, backend='numpy', name='hand')
        hand_dc = create_test_raster(hand_data, backend='dask+cupy',
                                     name='hand', chunks=(2, 2))
        result_np = inundation(hand_np, water_level=5.0)
        result_dc = inundation(hand_dc, water_level=5.0)
        general_output_checks(hand_dc, result_dc,
                              expected_results=result_np.data)


# ===================================================================
# curve_number_runoff
# ===================================================================

class TestCNRunoffKnownValues:

    def test_basic(self):
        # CN=80, P=100 mm
        # S = 25400/80 - 254 = 317.5 - 254 = 63.5
        # Ia = 0.2 * 63.5 = 12.7
        # Q = (100 - 12.7)^2 / (100 + 0.8*63.5) = 87.3^2 / 150.8
        #   = 7621.29 / 150.8 = 50.539...
        p_data = np.array([[100.0]], dtype=np.float64)
        rainfall = create_test_raster(p_data, name='rainfall')
        result = curve_number_runoff(rainfall, curve_number=80.0)
        s = 63.5
        ia = 12.7
        expected = (100.0 - ia) ** 2 / (100.0 + 0.8 * s)
        np.testing.assert_allclose(result.data[0, 0], expected, rtol=1e-10)

    def test_below_initial_abstraction(self):
        # CN=80 => Ia=12.7, rainfall=10 < 12.7 => Q=0
        p_data = np.array([[10.0]], dtype=np.float64)
        rainfall = create_test_raster(p_data, name='rainfall')
        result = curve_number_runoff(rainfall, curve_number=80.0)
        np.testing.assert_allclose(result.data[0, 0], 0.0)

    def test_zero_rainfall(self):
        p_data = np.zeros((2, 2), dtype=np.float64)
        rainfall = create_test_raster(p_data, name='rainfall')
        result = curve_number_runoff(rainfall, curve_number=90.0)
        np.testing.assert_allclose(result.data, 0.0)

    def test_cn_100(self):
        # CN=100 => S=0, Ia=0, Q=P (all rainfall becomes runoff)
        p_data = np.array([[50.0, 100.0]], dtype=np.float64)
        rainfall = create_test_raster(p_data, name='rainfall')
        result = curve_number_runoff(rainfall, curve_number=100.0)
        np.testing.assert_allclose(result.data, p_data, rtol=1e-10)

    def test_spatially_varying_cn(self):
        p_data = np.array([[100.0, 100.0]], dtype=np.float64)
        cn_data = np.array([[80.0, 100.0]], dtype=np.float64)
        rainfall = create_test_raster(p_data, name='rainfall')
        cn_raster = create_test_raster(cn_data, name='cn')
        result = curve_number_runoff(rainfall, curve_number=cn_raster)
        # CN=80 cell
        s80 = 63.5
        ia80 = 12.7
        q80 = (100.0 - ia80) ** 2 / (100.0 + 0.8 * s80)
        # CN=100 cell: Q=P
        np.testing.assert_allclose(result.data[0, 0], q80, rtol=1e-10)
        np.testing.assert_allclose(result.data[0, 1], 100.0, rtol=1e-10)


class TestCNRunoffNaN:

    def test_nan_in_rainfall(self):
        p_data = np.array([[np.nan, 50.0]], dtype=np.float64)
        rainfall = create_test_raster(p_data, name='rainfall')
        result = curve_number_runoff(rainfall, curve_number=80.0)
        assert np.isnan(result.data[0, 0])
        assert not np.isnan(result.data[0, 1])


class TestCNRunoffValidation:

    def test_cn_zero(self):
        rainfall = create_test_raster(np.ones((2, 2), dtype=np.float64))
        with pytest.raises(ValueError, match="curve_number must be in"):
            curve_number_runoff(rainfall, curve_number=0.0)

    def test_cn_negative(self):
        rainfall = create_test_raster(np.ones((2, 2), dtype=np.float64))
        with pytest.raises(ValueError, match="curve_number must be in"):
            curve_number_runoff(rainfall, curve_number=-10.0)

    def test_cn_over_100(self):
        rainfall = create_test_raster(np.ones((2, 2), dtype=np.float64))
        with pytest.raises(ValueError, match="curve_number must be in"):
            curve_number_runoff(rainfall, curve_number=101.0)

    def test_cn_wrong_type(self):
        rainfall = create_test_raster(np.ones((2, 2), dtype=np.float64))
        with pytest.raises(TypeError, match="curve_number must be numeric"):
            curve_number_runoff(rainfall, curve_number="high")


@dask_array_available
class TestCNRunoffDask:

    def test_numpy_equals_dask(self):
        np.random.seed(42)
        p_data = np.random.uniform(0, 200, (5, 5)).astype(np.float64)
        p_np = create_test_raster(p_data, backend='numpy', name='rainfall')
        p_da = create_test_raster(p_data, backend='dask', name='rainfall')
        result_np = curve_number_runoff(p_np, curve_number=75.0)
        result_da = curve_number_runoff(p_da, curve_number=75.0)
        general_output_checks(p_da, result_da,
                              expected_results=result_np.data)


@cuda_and_cupy_available
class TestCNRunoffCuPy:

    def test_numpy_equals_cupy(self):
        np.random.seed(42)
        p_data = np.random.uniform(0, 200, (5, 5)).astype(np.float64)
        p_np = create_test_raster(p_data, backend='numpy', name='rainfall')
        p_cu = create_test_raster(p_data, backend='cupy', name='rainfall')
        result_np = curve_number_runoff(p_np, curve_number=75.0)
        result_cu = curve_number_runoff(p_cu, curve_number=75.0)
        general_output_checks(p_cu, result_cu,
                              expected_results=result_np.data)


@cuda_and_cupy_available
@dask_array_available
class TestCNRunoffDaskCuPy:

    def test_numpy_equals_dask_cupy(self):
        np.random.seed(42)
        p_data = np.random.uniform(0, 200, (5, 5)).astype(np.float64)
        p_np = create_test_raster(p_data, backend='numpy', name='rainfall')
        p_dc = create_test_raster(p_data, backend='dask+cupy',
                                  name='rainfall', chunks=(3, 3))
        result_np = curve_number_runoff(p_np, curve_number=75.0)
        result_dc = curve_number_runoff(p_dc, curve_number=75.0)
        general_output_checks(p_dc, result_dc,
                              expected_results=result_np.data)


# ===================================================================
# travel_time
# ===================================================================

class TestTravelTimeKnownValues:

    def test_basic(self):
        # flow_length=100, slope=45 deg, n=0.03
        # tan(45)=1, v = (1/0.03)*sqrt(1) = 33.333...
        # tt = 100 / 33.333... = 3.0
        fl_data = np.array([[100.0]], dtype=np.float64)
        sl_data = np.array([[45.0]], dtype=np.float64)
        fl = create_test_raster(fl_data, name='flow_length')
        sl = create_test_raster(sl_data, name='slope')
        result = travel_time(fl, sl, mannings_n=0.03)
        v = (1.0 / 0.03) * np.sqrt(np.tan(np.radians(45.0)))
        expected = 100.0 / v
        np.testing.assert_allclose(result.data[0, 0], expected, rtol=1e-10)

    def test_zero_slope_clamped(self):
        # slope=0 should be clamped, no inf/NaN
        fl_data = np.array([[100.0]], dtype=np.float64)
        sl_data = np.array([[0.0]], dtype=np.float64)
        fl = create_test_raster(fl_data, name='flow_length')
        sl = create_test_raster(sl_data, name='slope')
        result = travel_time(fl, sl, mannings_n=0.03)
        assert not np.any(np.isinf(result.data))
        assert not np.any(np.isnan(result.data))

    def test_varying_mannings(self):
        # Spatially varying n via DataArray
        fl_data = np.array([[100.0, 100.0]], dtype=np.float64)
        sl_data = np.array([[45.0, 45.0]], dtype=np.float64)
        n_data = np.array([[0.03, 0.06]], dtype=np.float64)
        fl = create_test_raster(fl_data, name='flow_length')
        sl = create_test_raster(sl_data, name='slope')
        n_raster = create_test_raster(n_data, name='mannings_n')
        result = travel_time(fl, sl, mannings_n=n_raster)
        # n=0.06 should give twice the travel time of n=0.03
        np.testing.assert_allclose(result.data[0, 1],
                                   result.data[0, 0] * 2.0, rtol=1e-10)


class TestTravelTimeNaN:

    def test_nan_propagation(self):
        fl_data = np.array([[np.nan, 100.0]], dtype=np.float64)
        sl_data = np.array([[45.0, np.nan]], dtype=np.float64)
        fl = create_test_raster(fl_data, name='flow_length')
        sl = create_test_raster(sl_data, name='slope')
        result = travel_time(fl, sl, mannings_n=0.03)
        assert np.isnan(result.data[0, 0])
        assert np.isnan(result.data[0, 1])


class TestTravelTimeValidation:

    def test_negative_mannings_n(self):
        fl = create_test_raster(np.ones((2, 2), dtype=np.float64))
        sl = create_test_raster(np.ones((2, 2), dtype=np.float64))
        with pytest.raises(ValueError, match="mannings_n must be > 0"):
            travel_time(fl, sl, mannings_n=-0.01)

    def test_zero_mannings_n(self):
        fl = create_test_raster(np.ones((2, 2), dtype=np.float64))
        sl = create_test_raster(np.ones((2, 2), dtype=np.float64))
        with pytest.raises(ValueError, match="mannings_n must be > 0"):
            travel_time(fl, sl, mannings_n=0.0)

    def test_wrong_type_mannings_n(self):
        fl = create_test_raster(np.ones((2, 2), dtype=np.float64))
        sl = create_test_raster(np.ones((2, 2), dtype=np.float64))
        with pytest.raises(TypeError, match="mannings_n must be numeric"):
            travel_time(fl, sl, mannings_n="rough")


class TestTravelTimeOutput:

    def test_coords_preserved(self):
        fl_data = np.ones((4, 4), dtype=np.float64) * 100.0
        sl_data = np.ones((4, 4), dtype=np.float64) * 10.0
        fl = create_test_raster(fl_data, name='flow_length')
        sl = create_test_raster(sl_data, name='slope')
        result = travel_time(fl, sl, mannings_n=0.03)
        general_output_checks(fl, result, verify_attrs=True)


@dask_array_available
class TestTravelTimeDask:

    @pytest.mark.parametrize("chunks", [(2, 2), (3, 3)])
    def test_numpy_equals_dask(self, chunks):
        np.random.seed(42)
        fl_data = np.random.uniform(10, 500, (5, 5)).astype(np.float64)
        sl_data = np.random.uniform(0.1, 60, (5, 5)).astype(np.float64)

        fl_np = create_test_raster(fl_data, backend='numpy', name='fl')
        sl_np = create_test_raster(sl_data, backend='numpy', name='sl')
        fl_da = create_test_raster(fl_data, backend='dask', name='fl',
                                   chunks=chunks)
        sl_da = create_test_raster(sl_data, backend='dask', name='sl',
                                   chunks=chunks)

        result_np = travel_time(fl_np, sl_np, mannings_n=0.03)
        result_da = travel_time(fl_da, sl_da, mannings_n=0.03)

        general_output_checks(fl_da, result_da,
                              expected_results=result_np.data)


@cuda_and_cupy_available
class TestTravelTimeCuPy:

    def test_numpy_equals_cupy(self):
        np.random.seed(42)
        fl_data = np.random.uniform(10, 500, (5, 5)).astype(np.float64)
        sl_data = np.random.uniform(0.1, 60, (5, 5)).astype(np.float64)

        fl_np = create_test_raster(fl_data, backend='numpy', name='fl')
        sl_np = create_test_raster(sl_data, backend='numpy', name='sl')
        fl_cu = create_test_raster(fl_data, backend='cupy', name='fl')
        sl_cu = create_test_raster(sl_data, backend='cupy', name='sl')

        result_np = travel_time(fl_np, sl_np, mannings_n=0.03)
        result_cu = travel_time(fl_cu, sl_cu, mannings_n=0.03)

        general_output_checks(fl_cu, result_cu,
                              expected_results=result_np.data)


@cuda_and_cupy_available
@dask_array_available
class TestTravelTimeDaskCuPy:

    def test_numpy_equals_dask_cupy(self):
        np.random.seed(42)
        fl_data = np.random.uniform(10, 500, (5, 5)).astype(np.float64)
        sl_data = np.random.uniform(0.1, 60, (5, 5)).astype(np.float64)

        fl_np = create_test_raster(fl_data, backend='numpy', name='fl')
        sl_np = create_test_raster(sl_data, backend='numpy', name='sl')
        fl_dc = create_test_raster(fl_data, backend='dask+cupy', name='fl',
                                   chunks=(3, 3))
        sl_dc = create_test_raster(sl_data, backend='dask+cupy', name='sl',
                                   chunks=(3, 3))

        result_np = travel_time(fl_np, sl_np, mannings_n=0.03)
        result_dc = travel_time(fl_dc, sl_dc, mannings_n=0.03)

        general_output_checks(fl_dc, result_dc,
                              expected_results=result_np.data)
