"""Tests for xrspatial.fire module."""
from functools import partial
from math import exp, sqrt, tan

import numpy as np
import pytest
import xarray as xr

from xrspatial.fire import (
    ANDERSON_13,
    burn_severity_class,
    dnbr,
    fireline_intensity,
    flame_length,
    kbdi,
    rate_of_spread,
    rdnbr,
)
from xrspatial.tests.general_checks import (
    assert_numpy_equals_cupy,
    assert_numpy_equals_dask_cupy,
    assert_numpy_equals_dask_numpy,
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
    general_output_checks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nbr_data():
    """Pre- and post-fire NBR arrays (4x5)."""
    pre = np.array([
        [0.5,  0.6,  0.7,  0.3, np.nan],
        [0.4,  0.2,  0.8,  0.1,  0.9],
        [0.0,  0.55, 0.45, 0.65, 0.35],
        [0.72, 0.33, 0.88, 0.11, 0.44],
    ], dtype=np.float32)
    post = np.array([
        [0.1,  0.3,  0.2,  0.25, 0.0],
        [0.35, 0.1,  0.3,  0.05, np.nan],
        [0.0,  0.50, 0.10, 0.60, 0.30],
        [0.20, 0.30, 0.40, 0.05, 0.40],
    ], dtype=np.float32)
    return pre, post


@pytest.fixture
def intensity_data():
    """Fireline intensity array (3x4)."""
    return np.array([
        [100.0, 500.0,  0.0, np.nan],
        [1000., 50.0, -10.0,  250.0],
        [5000., 1.0,   10.0,  np.nan],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# dNBR
# ---------------------------------------------------------------------------
class TestDnbr:
    def test_known_values(self, nbr_data):
        pre, post = nbr_data
        pre_agg = create_test_raster(pre)
        post_agg = create_test_raster(post)
        result = dnbr(pre_agg, post_agg)
        expected = np.array([
            [0.4, 0.3, 0.5, 0.05, np.nan],
            [0.05, 0.1, 0.5, 0.05, np.nan],
            [0.0, 0.05, 0.35, 0.05, 0.05],
            [0.52, 0.03, 0.48, 0.06, 0.04],
        ], dtype=np.float32)
        np.testing.assert_allclose(result.data, expected, rtol=1e-5,
                                   equal_nan=True)
        assert result.name == 'dnbr'

    def test_nan_propagation(self):
        data = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
        a = create_test_raster(data)
        b = create_test_raster(np.ones((2, 2), dtype=np.float32))
        result = dnbr(a, b)
        assert np.isnan(result.data[0, 1])
        assert np.isnan(result.data[1, 0])

    def test_output_preserves_metadata(self, nbr_data):
        pre, post = nbr_data
        pre_agg = create_test_raster(pre)
        post_agg = create_test_raster(post)
        result = dnbr(pre_agg, post_agg)
        general_output_checks(pre_agg, result, verify_dtype=False)

    def test_custom_name(self, nbr_data):
        pre, post = nbr_data
        result = dnbr(create_test_raster(pre), create_test_raster(post),
                      name='my_dnbr')
        assert result.name == 'my_dnbr'

    def test_non_dataarray_raises(self):
        with pytest.raises(TypeError):
            dnbr(np.ones((3, 3)), create_test_raster(np.ones((3, 3))))

    def test_shape_mismatch_raises(self):
        a = create_test_raster(np.ones((3, 3), dtype='f4'))
        b = create_test_raster(np.ones((4, 4), dtype='f4'))
        with pytest.raises(ValueError):
            dnbr(a, b)

    @dask_array_available
    def test_numpy_equals_dask(self, nbr_data):
        pre, post = nbr_data
        np_pre = create_test_raster(pre, backend='numpy')
        np_post = create_test_raster(post, backend='numpy')
        dk_pre = create_test_raster(pre, backend='dask+numpy')
        dk_post = create_test_raster(post, backend='dask+numpy')
        np_result = dnbr(np_pre, np_post)
        dk_result = dnbr(dk_pre, dk_post)
        np.testing.assert_allclose(dk_result.data.compute(), np_result.data,
                                   rtol=1e-5, equal_nan=True)

    @cuda_and_cupy_available
    def test_numpy_equals_cupy(self, nbr_data):
        pre, post = nbr_data
        np_pre = create_test_raster(pre, backend='numpy')
        np_post = create_test_raster(post, backend='numpy')
        cp_pre = create_test_raster(pre, backend='cupy')
        cp_post = create_test_raster(post, backend='cupy')
        np_result = dnbr(np_pre, np_post)
        cp_result = dnbr(cp_pre, cp_post)
        np.testing.assert_allclose(cp_result.data.get(), np_result.data,
                                   rtol=1e-5, equal_nan=True)

    @dask_array_available
    @cuda_and_cupy_available
    def test_numpy_equals_dask_cupy(self, nbr_data):
        pre, post = nbr_data
        np_pre = create_test_raster(pre, backend='numpy')
        np_post = create_test_raster(post, backend='numpy')
        dc_pre = create_test_raster(pre, backend='dask+cupy')
        dc_post = create_test_raster(post, backend='dask+cupy')
        np_result = dnbr(np_pre, np_post)
        dc_result = dnbr(dc_pre, dc_post)
        np.testing.assert_allclose(dc_result.data.compute().get(),
                                   np_result.data,
                                   rtol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# RdNBR
# ---------------------------------------------------------------------------
class TestRdnbr:
    def test_known_values(self):
        dnbr_arr = np.array([[0.4, 0.3], [0.5, np.nan]], dtype=np.float32)
        pre_arr = np.array([[500.0, 200.0], [800.0, 100.0]], dtype=np.float32)
        d_agg = create_test_raster(dnbr_arr)
        p_agg = create_test_raster(pre_arr)
        result = rdnbr(d_agg, p_agg)
        # expected: dnbr / sqrt(abs(pre/1000))
        expected = np.array([
            [0.4 / sqrt(0.5), 0.3 / sqrt(0.2)],
            [0.5 / sqrt(0.8), np.nan],
        ], dtype=np.float32)
        np.testing.assert_allclose(result.data, expected, rtol=1e-5,
                                   equal_nan=True)
        assert result.name == 'rdnbr'

    def test_division_guard_near_zero_pre(self):
        dnbr_arr = np.array([[0.5, 0.3]], dtype=np.float32)
        pre_arr = np.array([[0.0, 1e-12]], dtype=np.float32)
        result = rdnbr(create_test_raster(dnbr_arr),
                       create_test_raster(pre_arr))
        # both should be NaN due to near-zero guard
        assert np.isnan(result.data[0, 0])
        assert np.isnan(result.data[0, 1])

    def test_nan_propagation(self):
        d = np.array([[np.nan, 0.5]], dtype=np.float32)
        p = np.array([[500.0, np.nan]], dtype=np.float32)
        result = rdnbr(create_test_raster(d), create_test_raster(p))
        assert np.isnan(result.data[0, 0])
        assert np.isnan(result.data[0, 1])

    @dask_array_available
    def test_numpy_equals_dask(self):
        rng = np.random.default_rng(42)
        d = rng.uniform(-1, 1, (6, 8)).astype('f4')
        p = rng.uniform(0.01, 1.0, (6, 8)).astype('f4') * 1000
        func = lambda a, b: rdnbr(a, b)
        np_d = create_test_raster(d, backend='numpy')
        np_p = create_test_raster(p, backend='numpy')
        dk_d = create_test_raster(d, backend='dask+numpy')
        dk_p = create_test_raster(p, backend='dask+numpy')
        np_r = rdnbr(np_d, np_p)
        dk_r = rdnbr(dk_d, dk_p)
        np.testing.assert_allclose(dk_r.data.compute(), np_r.data,
                                   rtol=1e-5, equal_nan=True)

    @cuda_and_cupy_available
    def test_numpy_equals_cupy(self):
        rng = np.random.default_rng(42)
        d = rng.uniform(-1, 1, (6, 8)).astype('f4')
        p = rng.uniform(0.01, 1.0, (6, 8)).astype('f4') * 1000
        np_r = rdnbr(create_test_raster(d, 'numpy'),
                      create_test_raster(p, 'numpy'))
        cp_r = rdnbr(create_test_raster(d, 'cupy'),
                      create_test_raster(p, 'cupy'))
        np.testing.assert_allclose(cp_r.data.get(), np_r.data,
                                   rtol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# Burn severity class
# ---------------------------------------------------------------------------
class TestBurnSeverityClass:
    def test_known_thresholds(self):
        # one value per class boundary
        vals = np.array([
            [-0.5, -0.2, -0.001, 0.1],
            [0.3,  0.45,  0.7,   np.nan],
        ], dtype=np.float32)
        result = burn_severity_class(create_test_raster(vals))
        expected = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 0],
        ], dtype=np.int8)
        np.testing.assert_array_equal(result.data, expected)
        assert result.dtype == np.int8
        assert result.name == 'burn_severity_class'

    def test_boundary_values(self):
        # values just above each threshold (float64 to avoid rounding)
        vals = np.array([[-0.250, -0.100, 0.100, 0.270, 0.440, 0.660]],
                        dtype=np.float32)
        result = burn_severity_class(create_test_raster(vals))
        expected = np.array([[2, 3, 4, 5, 6, 7]], dtype=np.int8)
        np.testing.assert_array_equal(result.data, expected)

    def test_nan_produces_zero(self):
        vals = np.array([[np.nan, 0.7]], dtype=np.float32)
        result = burn_severity_class(create_test_raster(vals))
        assert result.data[0, 0] == 0
        assert result.data[0, 1] == 7

    def test_dataset_support(self):
        vals = np.array([[0.5, -0.3]], dtype=np.float32)
        ds = xr.Dataset({
            'a': xr.DataArray(vals, dims=['y', 'x']),
            'b': xr.DataArray(vals * -1, dims=['y', 'x']),
        })
        result = burn_severity_class(ds)
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {'a', 'b'}

    @dask_array_available
    def test_numpy_equals_dask(self):
        rng = np.random.default_rng(99)
        data = rng.uniform(-1, 1, (8, 10)).astype('f4')
        np_r = burn_severity_class(create_test_raster(data, 'numpy'))
        dk_r = burn_severity_class(create_test_raster(data, 'dask+numpy'))
        np.testing.assert_array_equal(dk_r.data.compute(), np_r.data)

    @cuda_and_cupy_available
    def test_numpy_equals_cupy(self):
        rng = np.random.default_rng(99)
        data = rng.uniform(-1, 1, (8, 10)).astype('f4')
        np_r = burn_severity_class(create_test_raster(data, 'numpy'))
        cp_r = burn_severity_class(create_test_raster(data, 'cupy'))
        np.testing.assert_array_equal(cp_r.data.get(), np_r.data)


# ---------------------------------------------------------------------------
# Fireline intensity
# ---------------------------------------------------------------------------
class TestFirelineIntensity:
    def test_known_values(self):
        fuel = np.array([[2.0, 0.5], [1.0, np.nan]], dtype=np.float32)
        spread = np.array([[0.1, 0.2], [np.nan, 0.3]], dtype=np.float32)
        result = fireline_intensity(create_test_raster(fuel),
                                    create_test_raster(spread),
                                    heat_content=18000)
        expected = np.array([
            [18000 * 2.0 * 0.1, 18000 * 0.5 * 0.2],
            [np.nan, np.nan],
        ], dtype=np.float32)
        np.testing.assert_allclose(result.data, expected, rtol=1e-5,
                                   equal_nan=True)

    def test_custom_heat_content(self):
        fuel = np.array([[1.0]], dtype=np.float32)
        spread = np.array([[1.0]], dtype=np.float32)
        result = fireline_intensity(create_test_raster(fuel),
                                    create_test_raster(spread),
                                    heat_content=20000)
        np.testing.assert_allclose(result.data[0, 0], 20000.0, rtol=1e-5)

    def test_nan_propagation(self):
        fuel = np.array([[np.nan, 1.0]], dtype=np.float32)
        spread = np.array([[1.0, np.nan]], dtype=np.float32)
        result = fireline_intensity(create_test_raster(fuel),
                                    create_test_raster(spread))
        assert np.all(np.isnan(result.data))

    @dask_array_available
    def test_numpy_equals_dask(self):
        rng = np.random.default_rng(7)
        f = rng.uniform(0, 5, (6, 8)).astype('f4')
        s = rng.uniform(0, 1, (6, 8)).astype('f4')
        np_r = fireline_intensity(create_test_raster(f, 'numpy'),
                                  create_test_raster(s, 'numpy'))
        dk_r = fireline_intensity(create_test_raster(f, 'dask+numpy'),
                                  create_test_raster(s, 'dask+numpy'))
        np.testing.assert_allclose(dk_r.data.compute(), np_r.data,
                                   rtol=1e-5, equal_nan=True)

    @cuda_and_cupy_available
    def test_numpy_equals_cupy(self):
        rng = np.random.default_rng(7)
        f = rng.uniform(0, 5, (6, 8)).astype('f4')
        s = rng.uniform(0, 1, (6, 8)).astype('f4')
        np_r = fireline_intensity(create_test_raster(f, 'numpy'),
                                  create_test_raster(s, 'numpy'))
        cp_r = fireline_intensity(create_test_raster(f, 'cupy'),
                                  create_test_raster(s, 'cupy'))
        np.testing.assert_allclose(cp_r.data.get(), np_r.data,
                                   rtol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# Flame length
# ---------------------------------------------------------------------------
class TestFlameLength:
    def test_known_values(self, intensity_data):
        agg = create_test_raster(intensity_data)
        result = flame_length(agg)
        # manually compute expected
        expected = np.full_like(intensity_data, np.nan)
        for y in range(intensity_data.shape[0]):
            for x in range(intensity_data.shape[1]):
                v = intensity_data[y, x]
                if np.isnan(v):
                    continue
                if v <= 0:
                    expected[y, x] = 0.0
                else:
                    expected[y, x] = 0.0775 * v ** 0.46
        np.testing.assert_allclose(result.data, expected, rtol=1e-5,
                                   equal_nan=True)

    def test_zero_and_negative_intensity(self):
        vals = np.array([[0.0, -5.0, -100.0]], dtype=np.float32)
        result = flame_length(create_test_raster(vals))
        np.testing.assert_array_equal(result.data, [[0.0, 0.0, 0.0]])

    def test_nan_propagation(self):
        vals = np.array([[np.nan]], dtype=np.float32)
        result = flame_length(create_test_raster(vals))
        assert np.isnan(result.data[0, 0])

    def test_dataset_support(self):
        vals = np.array([[100.0, 500.0]], dtype=np.float32)
        ds = xr.Dataset({
            'i1': xr.DataArray(vals, dims=['y', 'x']),
            'i2': xr.DataArray(vals * 2, dims=['y', 'x']),
        })
        result = flame_length(ds)
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {'i1', 'i2'}

    @dask_array_available
    def test_numpy_equals_dask(self, intensity_data):
        np_r = flame_length(create_test_raster(intensity_data, 'numpy'))
        dk_r = flame_length(create_test_raster(intensity_data, 'dask+numpy'))
        np.testing.assert_allclose(dk_r.data.compute(), np_r.data,
                                   rtol=1e-5, equal_nan=True)

    @cuda_and_cupy_available
    def test_numpy_equals_cupy(self, intensity_data):
        np_r = flame_length(create_test_raster(intensity_data, 'numpy'))
        cp_r = flame_length(create_test_raster(intensity_data, 'cupy'))
        np.testing.assert_allclose(cp_r.data.get(), np_r.data,
                                   rtol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# Rate of spread (Rothermel)
# ---------------------------------------------------------------------------
class TestRateOfSpread:
    @pytest.fixture
    def ros_inputs(self):
        """slope=10 deg, wind=10 km/h, moisture=0.06 (6%)."""
        shape = (4, 5)
        slope = np.full(shape, 10.0, dtype=np.float32)
        wind = np.full(shape, 10.0, dtype=np.float32)
        moisture = np.full(shape, 0.06, dtype=np.float32)
        return slope, wind, moisture

    def test_positive_output(self, ros_inputs):
        slope, wind, moisture = ros_inputs
        result = rate_of_spread(
            create_test_raster(slope),
            create_test_raster(wind),
            create_test_raster(moisture),
            fuel_model=1,
        )
        assert np.all(result.data > 0)
        assert result.name == 'rate_of_spread'

    def test_all_fuel_models_produce_finite_output(self, ros_inputs):
        slope, wind, moisture = ros_inputs
        for fm in range(1, 14):
            result = rate_of_spread(
                create_test_raster(slope),
                create_test_raster(wind),
                create_test_raster(moisture),
                fuel_model=fm,
            )
            assert np.all(np.isfinite(result.data)), f"fuel_model={fm} non-finite"
            assert np.all(result.data >= 0), f"fuel_model={fm} negative"

    def test_slope_increases_spread(self):
        shape = (3, 3)
        wind = np.full(shape, 10.0, dtype=np.float32)
        moisture = np.full(shape, 0.06, dtype=np.float32)
        flat = np.zeros(shape, dtype=np.float32)
        steep = np.full(shape, 30.0, dtype=np.float32)
        r_flat = rate_of_spread(create_test_raster(flat),
                                create_test_raster(wind),
                                create_test_raster(moisture))
        r_steep = rate_of_spread(create_test_raster(steep),
                                 create_test_raster(wind),
                                 create_test_raster(moisture))
        assert np.all(r_steep.data > r_flat.data)

    def test_wind_increases_spread(self):
        shape = (3, 3)
        slope = np.full(shape, 5.0, dtype=np.float32)
        moisture = np.full(shape, 0.06, dtype=np.float32)
        low_wind = np.full(shape, 5.0, dtype=np.float32)
        high_wind = np.full(shape, 30.0, dtype=np.float32)
        r_low = rate_of_spread(create_test_raster(slope),
                               create_test_raster(low_wind),
                               create_test_raster(moisture))
        r_high = rate_of_spread(create_test_raster(slope),
                                create_test_raster(high_wind),
                                create_test_raster(moisture))
        assert np.all(r_high.data > r_low.data)

    def test_high_moisture_suppresses_spread(self):
        shape = (3, 3)
        slope = np.full(shape, 5.0, dtype=np.float32)
        wind = np.full(shape, 10.0, dtype=np.float32)
        dry = np.full(shape, 0.03, dtype=np.float32)
        wet = np.full(shape, 0.10, dtype=np.float32)
        r_dry = rate_of_spread(create_test_raster(slope),
                               create_test_raster(wind),
                               create_test_raster(dry))
        r_wet = rate_of_spread(create_test_raster(slope),
                               create_test_raster(wind),
                               create_test_raster(wet))
        assert np.all(r_dry.data > r_wet.data)

    def test_nan_propagation(self):
        shape = (2, 2)
        slope = np.array([[10.0, np.nan], [10.0, 10.0]], dtype=np.float32)
        wind = np.array([[10.0, 10.0], [np.nan, 10.0]], dtype=np.float32)
        moisture = np.array([[0.06, 0.06], [0.06, np.nan]], dtype=np.float32)
        result = rate_of_spread(create_test_raster(slope),
                                create_test_raster(wind),
                                create_test_raster(moisture))
        assert np.isfinite(result.data[0, 0])
        assert np.isnan(result.data[0, 1])
        assert np.isnan(result.data[1, 0])
        assert np.isnan(result.data[1, 1])

    def test_invalid_fuel_model_raises(self):
        shape = (2, 2)
        args = [create_test_raster(np.ones(shape, dtype='f4'))] * 3
        with pytest.raises(ValueError):
            rate_of_spread(*args, fuel_model=0)
        with pytest.raises(ValueError):
            rate_of_spread(*args, fuel_model=14)

    @dask_array_available
    def test_numpy_equals_dask(self, ros_inputs):
        slope, wind, moisture = ros_inputs
        np_r = rate_of_spread(
            create_test_raster(slope, 'numpy'),
            create_test_raster(wind, 'numpy'),
            create_test_raster(moisture, 'numpy'),
        )
        dk_r = rate_of_spread(
            create_test_raster(slope, 'dask+numpy'),
            create_test_raster(wind, 'dask+numpy'),
            create_test_raster(moisture, 'dask+numpy'),
        )
        np.testing.assert_allclose(dk_r.data.compute(), np_r.data,
                                   rtol=1e-5, equal_nan=True)

    @cuda_and_cupy_available
    def test_numpy_equals_cupy(self, ros_inputs):
        slope, wind, moisture = ros_inputs
        np_r = rate_of_spread(
            create_test_raster(slope, 'numpy'),
            create_test_raster(wind, 'numpy'),
            create_test_raster(moisture, 'numpy'),
        )
        cp_r = rate_of_spread(
            create_test_raster(slope, 'cupy'),
            create_test_raster(wind, 'cupy'),
            create_test_raster(moisture, 'cupy'),
        )
        np.testing.assert_allclose(cp_r.data.get(), np_r.data,
                                   rtol=1e-4, equal_nan=True)


# ---------------------------------------------------------------------------
# KBDI
# ---------------------------------------------------------------------------
class TestKbdi:
    def test_known_value_warm_dry(self):
        """Warm day, no precip -> KBDI increases."""
        prev = np.array([[100.0]], dtype=np.float32)
        temp = np.array([[35.0]], dtype=np.float32)  # 35 C
        precip = np.array([[0.0]], dtype=np.float32)
        result = kbdi(create_test_raster(prev),
                      create_test_raster(temp),
                      create_test_raster(precip),
                      annual_precip=1500.0)
        assert result.data[0, 0] > 100.0  # should increase
        assert result.data[0, 0] <= 800.0

    def test_known_value_heavy_rain(self):
        """Heavy rain reduces KBDI."""
        prev = np.array([[400.0]], dtype=np.float32)
        temp = np.array([[25.0]], dtype=np.float32)
        precip = np.array([[50.0]], dtype=np.float32)  # 50 mm
        result = kbdi(create_test_raster(prev),
                      create_test_raster(temp),
                      create_test_raster(precip),
                      annual_precip=1500.0)
        # net precip = 50 - 5.08 = 44.92 mm, so Q drops from 400
        assert result.data[0, 0] < 400.0

    def test_cold_day_no_accumulation(self):
        """Below 10 C threshold, drought factor does not accumulate."""
        prev = np.array([[200.0]], dtype=np.float32)
        temp = np.array([[5.0]], dtype=np.float32)  # 5 C, below threshold
        precip = np.array([[0.0]], dtype=np.float32)
        result = kbdi(create_test_raster(prev),
                      create_test_raster(temp),
                      create_test_raster(precip),
                      annual_precip=1500.0)
        np.testing.assert_allclose(result.data[0, 0], 200.0, rtol=1e-5)

    def test_clamp_zero(self):
        """Heavy rain on low KBDI should clamp to 0."""
        prev = np.array([[10.0]], dtype=np.float32)
        temp = np.array([[5.0]], dtype=np.float32)
        precip = np.array([[100.0]], dtype=np.float32)
        result = kbdi(create_test_raster(prev),
                      create_test_raster(temp),
                      create_test_raster(precip),
                      annual_precip=1500.0)
        assert result.data[0, 0] == 0.0

    def test_clamp_800(self):
        """Extreme drought should not exceed 800."""
        prev = np.array([[799.0]], dtype=np.float32)
        temp = np.array([[45.0]], dtype=np.float32)
        precip = np.array([[0.0]], dtype=np.float32)
        result = kbdi(create_test_raster(prev),
                      create_test_raster(temp),
                      create_test_raster(precip),
                      annual_precip=500.0)
        assert result.data[0, 0] <= 800.0

    def test_nan_propagation(self):
        prev = np.array([[100.0, np.nan]], dtype=np.float32)
        temp = np.array([[30.0, 30.0]], dtype=np.float32)
        precip = np.array([[0.0, 0.0]], dtype=np.float32)
        result = kbdi(create_test_raster(prev),
                      create_test_raster(temp),
                      create_test_raster(precip),
                      annual_precip=1500.0)
        assert np.isfinite(result.data[0, 0])
        assert np.isnan(result.data[0, 1])

    def test_invalid_annual_precip_raises(self):
        shape = (2, 2)
        args = [create_test_raster(np.ones(shape, dtype='f4'))] * 3
        with pytest.raises(ValueError):
            kbdi(*args, annual_precip=0)
        with pytest.raises(ValueError):
            kbdi(*args, annual_precip=-100)

    @dask_array_available
    def test_numpy_equals_dask(self):
        rng = np.random.default_rng(123)
        prev = rng.uniform(0, 400, (6, 8)).astype('f4')
        temp = rng.uniform(15, 40, (6, 8)).astype('f4')
        precip = rng.uniform(0, 20, (6, 8)).astype('f4')
        np_r = kbdi(create_test_raster(prev, 'numpy'),
                    create_test_raster(temp, 'numpy'),
                    create_test_raster(precip, 'numpy'),
                    annual_precip=1200.0)
        dk_r = kbdi(create_test_raster(prev, 'dask+numpy'),
                    create_test_raster(temp, 'dask+numpy'),
                    create_test_raster(precip, 'dask+numpy'),
                    annual_precip=1200.0)
        np.testing.assert_allclose(dk_r.data.compute(), np_r.data,
                                   rtol=1e-5, equal_nan=True)

    @cuda_and_cupy_available
    def test_numpy_equals_cupy(self):
        rng = np.random.default_rng(123)
        prev = rng.uniform(0, 400, (6, 8)).astype('f4')
        temp = rng.uniform(15, 40, (6, 8)).astype('f4')
        precip = rng.uniform(0, 20, (6, 8)).astype('f4')
        np_r = kbdi(create_test_raster(prev, 'numpy'),
                    create_test_raster(temp, 'numpy'),
                    create_test_raster(precip, 'numpy'),
                    annual_precip=1200.0)
        cp_r = kbdi(create_test_raster(prev, 'cupy'),
                    create_test_raster(temp, 'cupy'),
                    create_test_raster(precip, 'cupy'),
                    annual_precip=1200.0)
        np.testing.assert_allclose(cp_r.data.get(), np_r.data,
                                   rtol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# Accessor smoke tests
# ---------------------------------------------------------------------------
class TestAccessors:
    def test_dnbr_accessor(self):
        pre = create_test_raster(np.array([[0.5, 0.3]], dtype='f4'))
        post = create_test_raster(np.array([[0.1, 0.2]], dtype='f4'))
        result = pre.xrs.dnbr(post)
        np.testing.assert_allclose(result.data, [[0.4, 0.1]], rtol=1e-5)

    def test_burn_severity_class_accessor(self):
        vals = create_test_raster(np.array([[0.5, -0.3]], dtype='f4'))
        result = vals.xrs.burn_severity_class()
        assert result.dtype == np.int8

    def test_flame_length_accessor(self):
        vals = create_test_raster(np.array([[100.0]], dtype='f4'))
        result = vals.xrs.flame_length()
        assert result.data[0, 0] > 0

    def test_dataset_burn_severity_class_accessor(self):
        ds = xr.Dataset({
            'a': xr.DataArray(np.array([[0.5]], dtype='f4'), dims=['y', 'x']),
        })
        result = ds.xrs.burn_severity_class()
        assert isinstance(result, xr.Dataset)

    def test_dataset_flame_length_accessor(self):
        ds = xr.Dataset({
            'a': xr.DataArray(np.array([[100.0]], dtype='f4'),
                              dims=['y', 'x']),
        })
        result = ds.xrs.flame_length()
        assert isinstance(result, xr.Dataset)
