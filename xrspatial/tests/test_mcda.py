"""Tests for the xrspatial.mcda module (#1030)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.mcda import (
    ahp_weights,
    boolean_overlay,
    constrain,
    fuzzy_overlay,
    owa,
    rank_weights,
    sensitivity,
    standardize,
    wlc,
    wpm,
)

try:
    import dask.array as da
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def criterion_raster():
    """Simple 4x4 raster with known values and one NaN."""
    data = np.array([
        [0.0, 10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0, 70.0],
        [80.0, 90.0, np.nan, 100.0],
        [5.0, 15.0, 25.0, 45.0],
    ], dtype=np.float64)
    return xr.DataArray(data, dims=["y", "x"])


@pytest.fixture
def criteria_dataset():
    """Dataset with three standardized criteria for combination tests."""
    np.random.seed(1030)
    slope = xr.DataArray(
        np.array([[0.2, 0.8, 0.5],
                  [0.9, 0.1, 0.6],
                  [0.4, 0.7, 0.3]], dtype=np.float64),
        dims=["y", "x"], name="slope",
    )
    distance = xr.DataArray(
        np.array([[0.5, 0.3, 0.7],
                  [0.2, 0.9, 0.4],
                  [0.8, 0.1, 0.6]], dtype=np.float64),
        dims=["y", "x"], name="distance",
    )
    ndvi = xr.DataArray(
        np.array([[0.6, 0.4, 0.8],
                  [0.3, 0.7, 0.5],
                  [0.9, 0.2, 0.1]], dtype=np.float64),
        dims=["y", "x"], name="ndvi",
    )
    return xr.Dataset({"slope": slope, "distance": distance, "ndvi": ndvi})


@pytest.fixture
def weights_3():
    return {"slope": 0.4, "distance": 0.35, "ndvi": 0.25}


# ===========================================================================
# standardize
# ===========================================================================

class TestStandardizeLinear:
    def test_higher_is_better(self, criterion_raster):
        result = standardize(
            criterion_raster, method="linear",
            direction="higher_is_better", bounds=(0, 100),
        )
        # 0 -> 0.0, 100 -> 1.0
        assert float(result.values[0, 0]) == pytest.approx(0.0)
        assert float(result.values[2, 3]) == pytest.approx(1.0)
        assert float(result.values[1, 2]) == pytest.approx(0.6)

    def test_lower_is_better(self, criterion_raster):
        result = standardize(
            criterion_raster, method="linear",
            direction="lower_is_better", bounds=(0, 100),
        )
        # 0 -> 1.0, 100 -> 0.0
        assert float(result.values[0, 0]) == pytest.approx(1.0)
        assert float(result.values[2, 3]) == pytest.approx(0.0)

    def test_nan_preserved(self, criterion_raster):
        result = standardize(
            criterion_raster, method="linear", bounds=(0, 100),
        )
        assert np.isnan(result.values[2, 2])

    def test_no_bounds_uses_data_range(self, criterion_raster):
        result = standardize(criterion_raster, method="linear")
        finite = criterion_raster.values[np.isfinite(criterion_raster.values)]
        # min value -> 0, max value -> 1
        min_idx = np.unravel_index(
            np.nanargmin(criterion_raster.values), criterion_raster.shape
        )
        max_idx = np.unravel_index(
            np.nanargmax(criterion_raster.values), criterion_raster.shape
        )
        assert float(result.values[min_idx]) == pytest.approx(0.0)
        assert float(result.values[max_idx]) == pytest.approx(1.0)

    def test_constant_surface(self):
        data = np.full((3, 3), 5.0)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(agg, method="linear", bounds=(5, 5))
        # Should return 0.5 for constant
        assert np.all(result.values == pytest.approx(0.5))

    def test_clipping(self):
        data = np.array([[0.0, 50.0, 200.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(agg, method="linear", bounds=(0, 100))
        assert float(result.values[0, 2]) == pytest.approx(1.0)

    def test_invalid_direction(self, criterion_raster):
        with pytest.raises(ValueError, match="direction"):
            standardize(
                criterion_raster, method="linear", direction="bad"
            )


class TestStandardizeSigmoidal:
    def test_basic(self):
        data = np.array([[0.0, 5000.0, 10000.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(
            agg, method="sigmoidal", midpoint=5000, spread=-0.001,
        )
        # At midpoint, sigmoid = 0.5
        assert float(result.values[0, 1]) == pytest.approx(0.5)
        # Negative spread: higher values -> lower score
        assert float(result.values[0, 0]) > float(result.values[0, 2])

    def test_nan_preserved(self):
        data = np.array([[np.nan, 5000.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(agg, method="sigmoidal", midpoint=5000, spread=0.001)
        assert np.isnan(result.values[0, 0])


class TestStandardizeGaussian:
    def test_peak_at_mean(self):
        data = np.array([[0.0, 0.65, 1.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(
            agg, method="gaussian", mean=0.65, std=0.15,
        )
        # At mean, gaussian = 1.0
        assert float(result.values[0, 1]) == pytest.approx(1.0)
        # Away from mean, values decrease
        assert float(result.values[0, 0]) < 1.0
        assert float(result.values[0, 2]) < 1.0

    def test_invalid_std(self):
        agg = xr.DataArray(np.array([[1.0]]), dims=["y", "x"])
        with pytest.raises(ValueError, match="std must be > 0"):
            standardize(agg, method="gaussian", mean=0, std=0)


class TestStandardizeTriangular:
    def test_basic(self):
        data = np.array([[10.0, 22.0, 30.0, 35.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(
            agg, method="triangular", low=15, peak=22, high=30,
        )
        # At peak -> 1.0
        assert float(result.values[0, 1]) == pytest.approx(1.0)
        # At high -> 0.0
        assert float(result.values[0, 2]) == pytest.approx(0.0)
        # Below low -> 0.0
        assert float(result.values[0, 0]) == pytest.approx(0.0)
        # Above high -> 0.0
        assert float(result.values[0, 3]) == pytest.approx(0.0)

    def test_invalid_vertices(self):
        agg = xr.DataArray(np.array([[1.0]]), dims=["y", "x"])
        with pytest.raises(ValueError, match="low <= peak <= high"):
            standardize(agg, method="triangular", low=30, peak=20, high=10)


class TestStandardizePiecewise:
    def test_basic(self):
        data = np.array([[0.0, 100.0, 500.0, 1000.0, 2000.0]],
                        dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(
            agg, method="piecewise",
            breakpoints=[0, 100, 500, 1000, 2000],
            values=[0.0, 0.3, 1.0, 0.6, 0.0],
        )
        assert float(result.values[0, 0]) == pytest.approx(0.0)
        assert float(result.values[0, 1]) == pytest.approx(0.3)
        assert float(result.values[0, 2]) == pytest.approx(1.0)
        assert float(result.values[0, 3]) == pytest.approx(0.6)
        assert float(result.values[0, 4]) == pytest.approx(0.0)

    def test_missing_params(self):
        agg = xr.DataArray(np.array([[1.0]]), dims=["y", "x"])
        with pytest.raises(ValueError, match="breakpoints and values"):
            standardize(agg, method="piecewise")

    def test_length_mismatch(self):
        agg = xr.DataArray(np.array([[1.0]]), dims=["y", "x"])
        with pytest.raises(ValueError, match="same length"):
            standardize(
                agg, method="piecewise",
                breakpoints=[0, 1, 2], values=[0.0, 1.0],
            )


class TestStandardizeCategorical:
    def test_basic(self):
        data = np.array([[1, 2, 3, 4]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(
            agg, method="categorical",
            mapping={1: 0.9, 2: 0.7, 3: 0.4, 4: 0.1},
        )
        assert float(result.values[0, 0]) == pytest.approx(0.9)
        assert float(result.values[0, 1]) == pytest.approx(0.7)
        assert float(result.values[0, 2]) == pytest.approx(0.4)
        assert float(result.values[0, 3]) == pytest.approx(0.1)

    def test_unmapped_values_nan(self):
        data = np.array([[1, 99]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(
            agg, method="categorical", mapping={1: 0.5},
        )
        assert float(result.values[0, 0]) == pytest.approx(0.5)
        assert np.isnan(result.values[0, 1])

    def test_missing_mapping(self):
        agg = xr.DataArray(np.array([[1.0]]), dims=["y", "x"])
        with pytest.raises(ValueError, match="mapping"):
            standardize(agg, method="categorical")


class TestStandardizeTriangularDegenerate:
    """Degenerate triangles where peak coincides with low or high."""

    def test_peak_equals_low(self):
        """Right triangle: 1.0 at peak=low, falling to 0 at high."""
        data = np.array([[10.0, 15.0, 20.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(agg, method="triangular", low=10, peak=10, high=20)
        assert float(result.values[0, 0]) == pytest.approx(1.0)
        assert float(result.values[0, 1]) == pytest.approx(0.5)
        assert float(result.values[0, 2]) == pytest.approx(0.0)

    def test_peak_equals_high(self):
        """Left triangle: 0 at low, rising to 1.0 at peak=high."""
        data = np.array([[10.0, 15.0, 20.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(agg, method="triangular", low=10, peak=20, high=20)
        assert float(result.values[0, 0]) == pytest.approx(0.0)
        assert float(result.values[0, 1]) == pytest.approx(0.5)
        assert float(result.values[0, 2]) == pytest.approx(1.0)

    def test_all_equal(self):
        """Fully degenerate: low=peak=high -> 1.0 at that point, 0 elsewhere."""
        data = np.array([[5.0, 10.0, 15.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        result = standardize(agg, method="triangular", low=10, peak=10, high=10)
        assert float(result.values[0, 0]) == pytest.approx(0.0)
        assert float(result.values[0, 1]) == pytest.approx(1.0)
        assert float(result.values[0, 2]) == pytest.approx(0.0)


class TestStandardizeSigmoidalOverflow:
    """Sigmoid must not produce warnings or inf for extreme values."""

    def test_extreme_values_no_warning(self):
        import warnings
        data = np.array([[0.0, 1e6, -1e6]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = standardize(
                agg, method="sigmoidal", midpoint=0, spread=1.0,
            )
        assert float(result.values[0, 0]) == pytest.approx(0.5)
        assert float(result.values[0, 1]) == pytest.approx(1.0)
        assert float(result.values[0, 2]) == pytest.approx(0.0, abs=1e-100)

    def test_large_spread_no_warning(self):
        import warnings
        data = np.array([[0.0, 100.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = standardize(
                agg, method="sigmoidal", midpoint=50, spread=100.0,
            )
        assert np.all(np.isfinite(result.values))


class TestStandardizeAllNaN:
    """All-NaN input should return all NaN without raising."""

    def test_linear_all_nan_with_bounds(self):
        agg = xr.DataArray(np.full((3, 3), np.nan), dims=["y", "x"])
        result = standardize(agg, method="linear", bounds=(0, 100))
        assert np.all(np.isnan(result.values))

    def test_linear_all_nan_no_bounds(self):
        """nanmin/nanmax on all-NaN should not raise."""
        import warnings
        agg = xr.DataArray(np.full((2, 2), np.nan), dims=["y", "x"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Should NOT raise RuntimeWarning
            result = standardize(agg, method="linear")
        assert np.all(np.isnan(result.values))

    def test_gaussian_all_nan(self):
        agg = xr.DataArray(np.full((2, 2), np.nan), dims=["y", "x"])
        result = standardize(agg, method="gaussian", mean=0, std=1)
        assert np.all(np.isnan(result.values))


class TestStandardizeUnknownMethod:
    def test_unknown_method(self):
        agg = xr.DataArray(np.array([[1.0]]), dims=["y", "x"])
        with pytest.raises(ValueError, match="Unknown method"):
            standardize(agg, method="magic")


@pytest.mark.skipif(not HAS_DASK, reason="Requires dask")
class TestStandardizeDask:
    def test_linear_dask(self, criterion_raster):
        dask_raster = criterion_raster.chunk({"y": 2, "x": 2})
        result = standardize(
            dask_raster, method="linear", bounds=(0, 100),
        )
        assert hasattr(result.data, 'compute')
        computed = result.compute()
        numpy_result = standardize(
            criterion_raster, method="linear", bounds=(0, 100),
        )
        np.testing.assert_allclose(
            computed.values, numpy_result.values, equal_nan=True,
        )

    def test_sigmoidal_dask(self):
        data = np.array([[0.0, 5000.0, 10000.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"]).chunk({"x": 2})
        result = standardize(
            agg, method="sigmoidal", midpoint=5000, spread=-0.001,
        )
        assert hasattr(result.data, 'compute')
        computed = result.compute()
        assert float(computed.values[0, 1]) == pytest.approx(0.5)

    def test_categorical_dask(self):
        data = np.array([[1, 2, 3]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"]).chunk({"x": 2})
        result = standardize(
            agg, method="categorical", mapping={1: 0.9, 2: 0.5, 3: 0.1},
        )
        computed = result.compute()
        assert float(computed.values[0, 0]) == pytest.approx(0.9)

    def test_piecewise_dask(self):
        data = np.array([[0.0, 50.0, 100.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"]).chunk({"x": 2})
        result = standardize(
            agg, method="piecewise",
            breakpoints=[0, 50, 100], values=[0.0, 1.0, 0.5],
        )
        computed = result.compute()
        assert float(computed.values[0, 1]) == pytest.approx(1.0)


# ===========================================================================
# weights
# ===========================================================================

class TestAHPWeights:
    def test_equal_comparisons(self):
        """All criteria equally important -> equal weights."""
        w, c = ahp_weights(
            criteria=["a", "b", "c"],
            comparisons={("a", "b"): 1, ("a", "c"): 1, ("b", "c"): 1},
        )
        assert w["a"] == pytest.approx(1 / 3, abs=0.01)
        assert w["b"] == pytest.approx(1 / 3, abs=0.01)
        assert w["c"] == pytest.approx(1 / 3, abs=0.01)
        assert c.is_consistent

    def test_dominant_criterion(self):
        """One criterion strongly preferred -> gets highest weight."""
        w, c = ahp_weights(
            criteria=["a", "b", "c"],
            comparisons={("a", "b"): 9, ("a", "c"): 9, ("b", "c"): 1},
        )
        assert w["a"] > w["b"]
        assert w["a"] > w["c"]
        assert sum(w.values()) == pytest.approx(1.0)

    def test_consistency_ratio(self):
        """Consistent judgments should have CR < 0.10."""
        w, c = ahp_weights(
            criteria=["slope", "distance", "ndvi"],
            comparisons={
                ("slope", "distance"): 3,
                ("slope", "ndvi"): 2,
                ("distance", "ndvi"): 1 / 2,
            },
        )
        assert c.ratio < 0.10
        assert c.is_consistent

    def test_weights_sum_to_one(self):
        w, _ = ahp_weights(
            criteria=["a", "b", "c", "d"],
            comparisons={
                ("a", "b"): 3, ("a", "c"): 5, ("a", "d"): 2,
                ("b", "c"): 2, ("b", "d"): 1,
                ("c", "d"): 1 / 3,
            },
        )
        assert sum(w.values()) == pytest.approx(1.0)

    def test_unknown_criterion_raises(self):
        with pytest.raises(ValueError, match="Unknown criterion"):
            ahp_weights(
                criteria=["a", "b"],
                comparisons={("a", "z"): 2},
            )

    def test_too_few_criteria_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ahp_weights(criteria=["a"], comparisons={})


class TestRankWeights:
    def test_roc(self):
        w = rank_weights(["a", "b", "c"], method="roc")
        # First-ranked should have highest weight
        assert w["a"] > w["b"] > w["c"]
        assert sum(w.values()) == pytest.approx(1.0)

    def test_rs(self):
        w = rank_weights(["a", "b", "c"], method="rs")
        assert w["a"] > w["b"] > w["c"]
        assert sum(w.values()) == pytest.approx(1.0)

    def test_rr(self):
        w = rank_weights(["a", "b", "c"], method="rr")
        assert w["a"] > w["b"] > w["c"]
        assert sum(w.values()) == pytest.approx(1.0)

    def test_single_criterion(self):
        w = rank_weights(["only"], method="roc")
        assert w["only"] == pytest.approx(1.0)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            rank_weights(["a", "b"], method="bad")


# ===========================================================================
# combine
# ===========================================================================

class TestWLC:
    def test_basic(self, criteria_dataset, weights_3):
        result = wlc(criteria_dataset, weights_3)
        # Manual check for [0,0]: 0.2*0.4 + 0.5*0.35 + 0.6*0.25 = 0.405
        assert float(result.values[0, 0]) == pytest.approx(0.405)

    def test_output_shape(self, criteria_dataset, weights_3):
        result = wlc(criteria_dataset, weights_3)
        assert result.shape == (3, 3)

    def test_missing_weight_raises(self, criteria_dataset):
        with pytest.raises(ValueError, match="Missing weights"):
            wlc(criteria_dataset, {"slope": 0.5, "distance": 0.5})

    def test_weights_not_summing_raises(self, criteria_dataset):
        with pytest.raises(ValueError, match="sum to"):
            wlc(criteria_dataset, {"slope": 0.5, "distance": 0.5, "ndvi": 0.5})

    def test_empty_dataset_raises(self):
        ds = xr.Dataset()
        with pytest.raises(ValueError, match="no variables"):
            wlc(ds, {})

    def test_not_dataset_raises(self):
        da = xr.DataArray(np.array([[1.0]]), dims=["y", "x"])
        with pytest.raises(TypeError, match="xr.Dataset"):
            wlc(da, {"a": 1.0})


class TestWPM:
    def test_basic(self, criteria_dataset, weights_3):
        result = wpm(criteria_dataset, weights_3)
        # Manual check for [0,0]: 0.2^0.4 * 0.5^0.35 * 0.6^0.25
        expected = 0.2 ** 0.4 * 0.5 ** 0.35 * 0.6 ** 0.25
        assert float(result.values[0, 0]) == pytest.approx(expected)

    def test_zero_value_kills_product(self):
        """A zero criterion value should produce zero output."""
        ds = xr.Dataset({
            "a": xr.DataArray(np.array([[0.0]]), dims=["y", "x"]),
            "b": xr.DataArray(np.array([[0.8]]), dims=["y", "x"]),
        })
        result = wpm(ds, {"a": 0.5, "b": 0.5})
        assert float(result.values[0, 0]) == pytest.approx(0.0)


class TestWLCNaN:
    def test_nan_propagates(self):
        """NaN in any criterion should propagate to the output."""
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.array([[0.5, np.nan]], dtype=np.float64), dims=["y", "x"],
            ),
            "b": xr.DataArray(
                np.array([[0.7, 0.4]], dtype=np.float64), dims=["y", "x"],
            ),
        })
        result = wlc(ds, {"a": 0.6, "b": 0.4})
        assert float(result.values[0, 0]) == pytest.approx(0.58)
        assert np.isnan(result.values[0, 1])


class TestOWA:
    def test_uniform_order_weights_equals_wlc_exact(
        self, criteria_dataset, weights_3
    ):
        """Uniform order weights should EXACTLY equal WLC."""
        n = 3
        ow = [1.0 / n] * n
        owa_result = owa(criteria_dataset, weights_3, ow)
        wlc_result = wlc(criteria_dataset, weights_3)
        np.testing.assert_allclose(
            owa_result.values, wlc_result.values, atol=1e-14,
        )

    def test_wrong_order_weights_length_raises(
        self, criteria_dataset, weights_3
    ):
        with pytest.raises(ValueError, match="order_weights length"):
            owa(criteria_dataset, weights_3, [0.5, 0.5])

    def test_order_weights_not_summing_raises(
        self, criteria_dataset, weights_3
    ):
        with pytest.raises(ValueError, match="order_weights must sum"):
            owa(criteria_dataset, weights_3, [0.5, 0.5, 0.5])


class TestFuzzyOverlay:
    def test_and_is_min(self, criteria_dataset):
        result = fuzzy_overlay(criteria_dataset, operator="and")
        # [0,0]: min(0.2, 0.5, 0.6) = 0.2
        assert float(result.values[0, 0]) == pytest.approx(0.2)

    def test_or_is_max(self, criteria_dataset):
        result = fuzzy_overlay(criteria_dataset, operator="or")
        # [0,0]: max(0.2, 0.5, 0.6) = 0.6
        assert float(result.values[0, 0]) == pytest.approx(0.6)

    def test_product(self, criteria_dataset):
        result = fuzzy_overlay(criteria_dataset, operator="product")
        # [0,0]: 0.2 * 0.5 * 0.6 = 0.06
        assert float(result.values[0, 0]) == pytest.approx(0.06)

    def test_sum(self, criteria_dataset):
        result = fuzzy_overlay(criteria_dataset, operator="sum")
        # [0,0]: 1 - (1-0.2)*(1-0.5)*(1-0.6) = 1 - 0.8*0.5*0.4 = 1 - 0.16 = 0.84
        assert float(result.values[0, 0]) == pytest.approx(0.84)

    def test_gamma(self, criteria_dataset):
        result = fuzzy_overlay(
            criteria_dataset, operator="gamma", gamma=0.5,
        )
        # gamma=0.5: (fuzzy_sum^0.5) * (product^0.5)
        prod = 0.2 * 0.5 * 0.6
        fsum = 1 - (0.8 * 0.5 * 0.4)
        expected = (fsum ** 0.5) * (prod ** 0.5)
        assert float(result.values[0, 0]) == pytest.approx(expected)

    def test_gamma_0_equals_product(self, criteria_dataset):
        gamma0 = fuzzy_overlay(
            criteria_dataset, operator="gamma", gamma=0.0,
        )
        product = fuzzy_overlay(criteria_dataset, operator="product")
        np.testing.assert_allclose(gamma0.values, product.values, atol=1e-10)

    def test_gamma_1_equals_sum(self, criteria_dataset):
        gamma1 = fuzzy_overlay(
            criteria_dataset, operator="gamma", gamma=1.0,
        )
        fsum = fuzzy_overlay(criteria_dataset, operator="sum")
        np.testing.assert_allclose(gamma1.values, fsum.values, atol=1e-10)

    def test_invalid_operator_raises(self, criteria_dataset):
        with pytest.raises(ValueError, match="Unknown operator"):
            fuzzy_overlay(criteria_dataset, operator="xor")

    def test_invalid_gamma_raises(self, criteria_dataset):
        with pytest.raises(ValueError, match="gamma must be in"):
            fuzzy_overlay(criteria_dataset, operator="gamma", gamma=1.5)


class TestBooleanOverlay:
    def test_and(self):
        a = xr.DataArray(
            np.array([[True, True, False]]), dims=["y", "x"],
        )
        b = xr.DataArray(
            np.array([[True, False, False]]), dims=["y", "x"],
        )
        result = boolean_overlay({"a": a, "b": b}, operator="and")
        np.testing.assert_array_equal(
            result.values, [[True, False, False]],
        )

    def test_or(self):
        a = xr.DataArray(
            np.array([[True, True, False]]), dims=["y", "x"],
        )
        b = xr.DataArray(
            np.array([[True, False, False]]), dims=["y", "x"],
        )
        result = boolean_overlay({"a": a, "b": b}, operator="or")
        np.testing.assert_array_equal(
            result.values, [[True, True, False]],
        )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            boolean_overlay({})

    def test_invalid_operator_raises(self):
        a = xr.DataArray(np.array([[True]]), dims=["y", "x"])
        with pytest.raises(ValueError, match="Unknown operator"):
            boolean_overlay({"a": a}, operator="xor")


# ===========================================================================
# constrain
# ===========================================================================

class TestConstrain:
    def test_basic(self):
        suit = xr.DataArray(
            np.array([[0.8, 0.6, 0.9],
                      [0.5, 0.7, 0.3]], dtype=np.float64),
            dims=["y", "x"],
        )
        water = xr.DataArray(
            np.array([[False, False, True],
                      [False, False, False]]),
            dims=["y", "x"],
        )
        result = constrain(suit, exclude=[water])
        assert np.isnan(result.values[0, 2])
        assert float(result.values[0, 0]) == pytest.approx(0.8)

    def test_multiple_masks(self):
        suit = xr.DataArray(
            np.array([[0.8, 0.6, 0.9]], dtype=np.float64),
            dims=["y", "x"],
        )
        mask1 = xr.DataArray(
            np.array([[True, False, False]]), dims=["y", "x"],
        )
        mask2 = xr.DataArray(
            np.array([[False, True, False]]), dims=["y", "x"],
        )
        result = constrain(suit, exclude=[mask1, mask2])
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[0, 1])
        assert float(result.values[0, 2]) == pytest.approx(0.9)

    def test_custom_fill(self):
        suit = xr.DataArray(
            np.array([[0.8, 0.6]], dtype=np.float64),
            dims=["y", "x"],
        )
        mask = xr.DataArray(
            np.array([[True, False]]), dims=["y", "x"],
        )
        result = constrain(suit, exclude=[mask], fill=-1.0)
        assert float(result.values[0, 0]) == pytest.approx(-1.0)


# ===========================================================================
# sensitivity
# ===========================================================================

class TestSensitivityOAT:
    def test_returns_dataset(self, criteria_dataset, weights_3):
        result = sensitivity(
            criteria_dataset, weights_3,
            method="one_at_a_time", delta=0.05,
        )
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"slope", "distance", "ndvi"}

    def test_zero_delta_gives_zero_sensitivity(
        self, criteria_dataset, weights_3
    ):
        result = sensitivity(
            criteria_dataset, weights_3,
            method="one_at_a_time", delta=0.0,
        )
        for var in result.data_vars:
            np.testing.assert_allclose(
                result[var].values, 0.0, atol=1e-10,
            )


class TestSensitivityMonteCarlo:
    def test_returns_dataarray(self, criteria_dataset, weights_3):
        result = sensitivity(
            criteria_dataset, weights_3,
            method="monte_carlo", n_samples=50,
        )
        assert isinstance(result, xr.DataArray)
        assert result.shape == (3, 3)

    def test_cv_non_negative(self, criteria_dataset, weights_3):
        result = sensitivity(
            criteria_dataset, weights_3,
            method="monte_carlo", n_samples=50,
        )
        assert np.all(result.values >= 0)

    def test_unknown_method_raises(self, criteria_dataset, weights_3):
        with pytest.raises(ValueError, match="Unknown method"):
            sensitivity(criteria_dataset, weights_3, method="bad")

    def test_unknown_combine_raises(self, criteria_dataset, weights_3):
        with pytest.raises(ValueError, match="Unknown combine_method"):
            sensitivity(
                criteria_dataset, weights_3, combine_method="bad",
            )


@pytest.mark.skipif(not HAS_DASK, reason="Requires dask")
class TestSensitivityDask:
    def test_monte_carlo_dask_returns_numpy(self):
        """MC with dask input should compute eagerly, not build huge graph."""
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.random.rand(20, 20), dims=["y", "x"],
            ).chunk({"y": 10, "x": 10}),
            "b": xr.DataArray(
                np.random.rand(20, 20), dims=["y", "x"],
            ).chunk({"y": 10, "x": 10}),
        })
        result = sensitivity(
            ds, {"a": 0.6, "b": 0.4},
            method="monte_carlo", n_samples=30,
        )
        # Result should be eagerly computed numpy, not dask
        assert isinstance(result.data, np.ndarray)
        assert result.shape == (20, 20)
        assert np.all(result.values >= 0)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestSinglePixelRaster:
    """All functions should work on a 1x1 raster."""

    def test_standardize_methods(self):
        single = xr.DataArray(np.array([[50.0]]), dims=["y", "x"])
        assert float(standardize(
            single, method="linear", bounds=(0, 100),
        ).values[0, 0]) == pytest.approx(0.5)
        assert float(standardize(
            single, method="sigmoidal", midpoint=50, spread=0.1,
        ).values[0, 0]) == pytest.approx(0.5)
        assert float(standardize(
            single, method="gaussian", mean=50, std=10,
        ).values[0, 0]) == pytest.approx(1.0)
        assert float(standardize(
            single, method="triangular", low=0, peak=50, high=100,
        ).values[0, 0]) == pytest.approx(1.0)
        assert float(standardize(
            single, method="piecewise",
            breakpoints=[0, 50, 100], values=[0.0, 1.0, 0.5],
        ).values[0, 0]) == pytest.approx(1.0)
        assert float(standardize(
            single, method="categorical", mapping={50: 0.8},
        ).values[0, 0]) == pytest.approx(0.8)

    def test_combine_single_pixel(self):
        ds = xr.Dataset({
            "a": xr.DataArray(np.array([[0.5]]), dims=["y", "x"]),
            "b": xr.DataArray(np.array([[0.7]]), dims=["y", "x"]),
        })
        w = {"a": 0.6, "b": 0.4}
        assert float(wlc(ds, w).values[0, 0]) == pytest.approx(0.58)
        expected_wpm = 0.5 ** 0.6 * 0.7 ** 0.4
        assert float(wpm(ds, w).values[0, 0]) == pytest.approx(expected_wpm)

    def test_full_pipeline_single_pixel(self):
        raw = xr.DataArray(np.array([[50.0]]), dims=["y", "x"])
        std = standardize(raw, method="linear", bounds=(0, 100))
        ds = xr.Dataset({"a": std, "b": std})
        result = wlc(ds, {"a": 0.6, "b": 0.4})
        result = constrain(
            result,
            exclude=[xr.DataArray(np.array([[False]]), dims=["y", "x"])],
        )
        assert result.shape == (1, 1)
        assert float(result.values[0, 0]) == pytest.approx(0.5)


class TestNaNPropagation:
    """NaN in any criterion should propagate through all combine methods."""

    @pytest.fixture
    def ds_with_nan(self):
        return xr.Dataset({
            "a": xr.DataArray(
                np.array([[0.5, np.nan, 0.3]], dtype=np.float64),
                dims=["y", "x"],
            ),
            "b": xr.DataArray(
                np.array([[0.7, 0.4, np.nan]], dtype=np.float64),
                dims=["y", "x"],
            ),
        })

    def test_wlc_nan(self, ds_with_nan):
        r = wlc(ds_with_nan, {"a": 0.6, "b": 0.4})
        assert np.isfinite(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert np.isnan(r.values[0, 2])

    def test_wpm_nan(self, ds_with_nan):
        r = wpm(ds_with_nan, {"a": 0.6, "b": 0.4})
        assert np.isfinite(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert np.isnan(r.values[0, 2])

    def test_fuzzy_and_nan(self, ds_with_nan):
        r = fuzzy_overlay(ds_with_nan, operator="and")
        assert np.isfinite(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert np.isnan(r.values[0, 2])

    def test_fuzzy_or_nan(self, ds_with_nan):
        r = fuzzy_overlay(ds_with_nan, operator="or")
        assert np.isfinite(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert np.isnan(r.values[0, 2])

    def test_fuzzy_product_nan(self, ds_with_nan):
        r = fuzzy_overlay(ds_with_nan, operator="product")
        assert np.isfinite(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert np.isnan(r.values[0, 2])

    def test_fuzzy_sum_nan(self, ds_with_nan):
        r = fuzzy_overlay(ds_with_nan, operator="sum")
        assert np.isfinite(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert np.isnan(r.values[0, 2])

    def test_fuzzy_gamma_nan(self, ds_with_nan):
        r = fuzzy_overlay(ds_with_nan, operator="gamma", gamma=0.9)
        assert np.isfinite(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert np.isnan(r.values[0, 2])

    def test_owa_nan(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.array([[0.5, np.nan]], dtype=np.float64),
                dims=["y", "x"],
            ),
            "b": xr.DataArray(
                np.array([[0.7, 0.4]], dtype=np.float64),
                dims=["y", "x"],
            ),
            "c": xr.DataArray(
                np.array([[0.3, 0.6]], dtype=np.float64),
                dims=["y", "x"],
            ),
        })
        w = {"a": 0.4, "b": 0.35, "c": 0.25}
        r = owa(ds, w, [0.5, 0.3, 0.2])
        assert np.isfinite(r.values[0, 0])
        assert np.isnan(r.values[0, 1])


class TestPiecewiseExtrapolation:
    """np.interp clamps values outside the breakpoint range."""

    def test_below_range_clamps_to_first_value(self):
        data = np.array([[-100.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(
            agg, method="piecewise",
            breakpoints=[0, 100], values=[0.2, 0.9],
        )
        assert float(r.values[0, 0]) == pytest.approx(0.2)

    def test_above_range_clamps_to_last_value(self):
        data = np.array([[500.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(
            agg, method="piecewise",
            breakpoints=[0, 100], values=[0.2, 0.9],
        )
        assert float(r.values[0, 0]) == pytest.approx(0.9)


class TestCategoricalPrecision:
    """Categorical uses exact equality; near-miss float values are NaN."""

    def test_exact_float_match(self):
        data = np.array([[1.0, 2.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(
            agg, method="categorical", mapping={1: 0.9, 2: 0.5},
        )
        assert float(r.values[0, 0]) == pytest.approx(0.9)
        assert float(r.values[0, 1]) == pytest.approx(0.5)

    def test_float_near_miss_is_nan(self):
        """Floating point near-miss should not match."""
        data = np.array([[1.0000000001]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(
            agg, method="categorical", mapping={1: 0.9},
        )
        assert np.isnan(r.values[0, 0])

    def test_integer_data(self):
        """Integer data should match float keys via implicit cast."""
        data = np.array([[1, 2, 3]], dtype=np.int32)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(
            agg, method="categorical",
            mapping={1: 0.9, 2: 0.7, 3: 0.4},
        )
        assert float(r.values[0, 0]) == pytest.approx(0.9)
        assert float(r.values[0, 1]) == pytest.approx(0.7)
        assert float(r.values[0, 2]) == pytest.approx(0.4)


class TestAHPIncomplete:
    """AHP with fewer than n(n-1)/2 comparisons should warn."""

    def test_warns_on_incomplete(self):
        with pytest.warns(UserWarning, match="Only 1 of 3"):
            ahp_weights(
                criteria=["a", "b", "c"],
                comparisons={("a", "b"): 3},
            )

    def test_no_warning_when_complete(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ahp_weights(
                criteria=["a", "b", "c"],
                comparisons={
                    ("a", "b"): 3,
                    ("a", "c"): 5,
                    ("b", "c"): 2,
                },
            )

    def test_missing_defaults_to_equal(self):
        """Missing comparisons default to 1 (equal importance)."""
        w_full, _ = ahp_weights(
            criteria=["a", "b"],
            comparisons={("a", "b"): 1},
        )
        with pytest.warns(UserWarning):
            w_missing, _ = ahp_weights(
                criteria=["a", "b", "c"],
                comparisons={("a", "b"): 1},
            )
        # b and c should be approximately equal since their pairwise
        # comparison defaults to 1
        assert abs(w_missing["b"] - w_missing["c"]) < 0.05


class TestMismatchedCoords:
    """Criteria with partially overlapping coords produce NaN outside overlap."""

    def test_partial_overlap(self):
        a = xr.DataArray(
            np.array([[0.5, 0.3], [0.8, 0.2]], dtype=np.float64),
            dims=["y", "x"],
            coords={"y": [0, 1], "x": [0, 1]},
        )
        b = xr.DataArray(
            np.array([[0.7, 0.4], [0.1, 0.9]], dtype=np.float64),
            dims=["y", "x"],
            coords={"y": [1, 2], "x": [0, 1]},
        )
        ds = xr.Dataset({"a": a, "b": b})
        result = wlc(ds, {"a": 0.6, "b": 0.4})
        # y=0: only a has data, b is NaN -> result NaN
        assert np.isnan(result.sel(y=0, x=0).values)
        # y=1: both have data -> valid
        expected = 0.8 * 0.6 + 0.7 * 0.4
        assert float(result.sel(y=1, x=0).values) == pytest.approx(expected)
        # y=2: only b has data, a is NaN -> result NaN
        assert np.isnan(result.sel(y=2, x=0).values)


@pytest.mark.skipif(not HAS_DASK, reason="Requires dask")
class TestDaskChunkAlignment:
    """Dask operations should preserve chunks and produce correct results."""

    def test_standardize_preserves_chunks(self):
        data = np.random.rand(20, 20)
        agg = xr.DataArray(data, dims=["y", "x"]).chunk({"y": 10, "x": 10})
        for method in ["linear", "sigmoidal", "gaussian"]:
            kw = {"bounds": (0, 1)} if method == "linear" else {}
            if method == "sigmoidal":
                kw = {"midpoint": 0.5, "spread": 1.0}
            elif method == "gaussian":
                kw = {"mean": 0.5, "std": 0.2}
            result = standardize(agg, method=method, **kw)
            assert hasattr(result.data, "compute"), f"{method} lost dask"
            np_result = standardize(
                xr.DataArray(data, dims=["y", "x"]),
                method=method, **kw,
            )
            np.testing.assert_allclose(
                result.compute().values, np_result.values,
                equal_nan=True, atol=1e-14,
            )

    def test_wlc_dask(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.random.rand(20, 20), dims=["y", "x"],
            ).chunk({"y": 10}),
            "b": xr.DataArray(
                np.random.rand(20, 20), dims=["y", "x"],
            ).chunk({"y": 10}),
        })
        w = {"a": 0.6, "b": 0.4}
        result = wlc(ds, w)
        assert hasattr(result.data, "compute")
        ds_np = ds.compute()
        np_result = wlc(ds_np, w)
        np.testing.assert_allclose(
            result.compute().values, np_result.values, atol=1e-14,
        )

    def test_fuzzy_overlay_dask(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.random.rand(20, 20), dims=["y", "x"],
            ).chunk({"y": 10}),
            "b": xr.DataArray(
                np.random.rand(20, 20), dims=["y", "x"],
            ).chunk({"y": 10}),
        })
        for op in ["and", "or", "product", "sum"]:
            result = fuzzy_overlay(ds, operator=op)
            ds_np = ds.compute()
            np_result = fuzzy_overlay(ds_np, operator=op)
            np.testing.assert_allclose(
                result.compute().values, np_result.values, atol=1e-14,
                err_msg=f"fuzzy {op} mismatch",
            )

    def test_oat_sensitivity_dask(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.random.rand(20, 20), dims=["y", "x"],
            ).chunk({"y": 10}),
            "b": xr.DataArray(
                np.random.rand(20, 20), dims=["y", "x"],
            ).chunk({"y": 10}),
        })
        w = {"a": 0.6, "b": 0.4}
        result = sensitivity(ds, w, method="one_at_a_time", delta=0.05)
        assert isinstance(result, xr.Dataset)
        for var in result.data_vars:
            assert np.all(np.isfinite(result[var].compute().values))


class TestWPMEdgeCases:
    def test_all_ones(self):
        """All criteria at 1.0 should produce 1.0 regardless of weights."""
        ds = xr.Dataset({
            "a": xr.DataArray(np.array([[1.0]]), dims=["y", "x"]),
            "b": xr.DataArray(np.array([[1.0]]), dims=["y", "x"]),
        })
        r = wpm(ds, {"a": 0.3, "b": 0.7})
        assert float(r.values[0, 0]) == pytest.approx(1.0)

    def test_all_zeros(self):
        """All criteria at 0.0 should produce 0.0."""
        ds = xr.Dataset({
            "a": xr.DataArray(np.array([[0.0]]), dims=["y", "x"]),
            "b": xr.DataArray(np.array([[0.0]]), dims=["y", "x"]),
        })
        r = wpm(ds, {"a": 0.5, "b": 0.5})
        assert float(r.values[0, 0]) == pytest.approx(0.0)


class TestConstrainEdgeCases:
    def test_empty_exclude_list(self):
        """Empty exclude list should return input unchanged."""
        suit = xr.DataArray(
            np.array([[0.8, 0.6]], dtype=np.float64),
            dims=["y", "x"],
        )
        result = constrain(suit, exclude=[])
        np.testing.assert_array_equal(result.values, suit.values)

    def test_all_excluded(self):
        """All pixels excluded should be fill value."""
        suit = xr.DataArray(
            np.array([[0.8, 0.6]], dtype=np.float64),
            dims=["y", "x"],
        )
        mask = xr.DataArray(
            np.array([[True, True]]), dims=["y", "x"],
        )
        result = constrain(suit, exclude=[mask])
        assert np.all(np.isnan(result.values))


# ===========================================================================
# Input validation edge cases
# ===========================================================================

class TestInvertedBounds:
    def test_raises(self):
        agg = xr.DataArray(np.array([[50.0]]), dims=["y", "x"])
        with pytest.raises(ValueError, match="bounds\\[0\\] must be <= bounds\\[1\\]"):
            standardize(agg, method="linear", bounds=(100, 0))


class TestPiecewiseDuplicateBreakpoints:
    def test_raises(self):
        agg = xr.DataArray(np.array([[5.0]]), dims=["y", "x"])
        with pytest.raises(ValueError, match="unique"):
            standardize(
                agg, method="piecewise",
                breakpoints=[0, 5, 5, 10], values=[0, 0.3, 0.7, 1],
            )


class TestAHPValidation:
    def test_self_comparison_raises(self):
        with pytest.raises(ValueError, match="Self-comparison"):
            ahp_weights(
                criteria=["a", "b"],
                comparisons={("a", "a"): 5, ("a", "b"): 3},
            )

    def test_zero_comparison_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            ahp_weights(
                criteria=["a", "b"],
                comparisons={("a", "b"): 0},
            )

    def test_negative_comparison_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            ahp_weights(
                criteria=["a", "b"],
                comparisons={("a", "b"): -3},
            )


class TestExtraWeights:
    def test_extra_keys_raises(self):
        ds = xr.Dataset({
            "a": xr.DataArray(np.array([[0.5]]), dims=["y", "x"]),
        })
        with pytest.raises(ValueError, match="not in criteria"):
            wlc(ds, {"a": 0.5, "b": 0.3, "c": 0.2})

    def test_extra_keys_wpm_raises(self):
        ds = xr.Dataset({
            "a": xr.DataArray(np.array([[0.5]]), dims=["y", "x"]),
        })
        with pytest.raises(ValueError, match="not in criteria"):
            wpm(ds, {"a": 0.5, "b": 0.5})


class TestBooleanOverlayFloat:
    """Float data should be cast to bool (nonzero = True)."""

    def test_float_and(self):
        a = xr.DataArray(
            np.array([[0.5, 0.0, 1.0]]), dims=["y", "x"],
        )
        b = xr.DataArray(
            np.array([[0.3, 0.0, 1.0]]), dims=["y", "x"],
        )
        result = boolean_overlay({"a": a, "b": b}, operator="and")
        np.testing.assert_array_equal(
            result.values, [[True, False, True]],
        )

    def test_float_or(self):
        a = xr.DataArray(
            np.array([[0.5, 0.0, 0.0]]), dims=["y", "x"],
        )
        b = xr.DataArray(
            np.array([[0.0, 0.0, 1.0]]), dims=["y", "x"],
        )
        result = boolean_overlay({"a": a, "b": b}, operator="or")
        np.testing.assert_array_equal(
            result.values, [[True, False, True]],
        )


# ===========================================================================
# Inf handling
# ===========================================================================

class TestInfValues:
    """Inf values should be treated as non-finite (mapped to NaN)."""

    def test_linear_inf(self):
        data = np.array([[np.inf, -np.inf, 50.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(agg, method="linear", bounds=(0, 100))
        assert np.isnan(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert float(r.values[0, 2]) == pytest.approx(0.5)

    def test_sigmoidal_inf(self):
        data = np.array([[np.inf, -np.inf, 50.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(agg, method="sigmoidal", midpoint=50, spread=0.1)
        assert np.isnan(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert np.isfinite(r.values[0, 2])

    def test_gaussian_inf(self):
        data = np.array([[np.inf, -np.inf, 50.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(agg, method="gaussian", mean=50, std=10)
        assert np.isnan(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert float(r.values[0, 2]) == pytest.approx(1.0)

    def test_triangular_inf(self):
        data = np.array([[np.inf, -np.inf, 50.0]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(agg, method="triangular", low=0, peak=50, high=100)
        assert np.isnan(r.values[0, 0])
        assert np.isnan(r.values[0, 1])
        assert float(r.values[0, 2]) == pytest.approx(1.0)


# ===========================================================================
# Sensitivity edge cases
# ===========================================================================

class TestSensitivityWeightClamping:
    """Perturbed weights should be clamped to [0, 1]."""

    def test_clamp_to_zero(self):
        """Weight near 0 perturbed negative should clamp to 0."""
        from xrspatial.mcda.sensitivity import _perturb_weights

        w = {"a": 0.02, "b": 0.58, "c": 0.40}
        _, w_minus = _perturb_weights(w, "a", 0.05)
        assert w_minus["a"] == pytest.approx(0.0)
        assert sum(w_minus.values()) == pytest.approx(1.0)

    def test_clamp_to_one(self):
        """Weight near 1 perturbed up should clamp to 1."""
        from xrspatial.mcda.sensitivity import _perturb_weights

        w = {"a": 0.98, "b": 0.02}
        w_plus, _ = _perturb_weights(w, "a", 0.05)
        assert w_plus["a"] == pytest.approx(1.0)
        assert w_plus["b"] == pytest.approx(0.0)
        assert sum(w_plus.values()) == pytest.approx(1.0)

    def test_perturbed_weights_sum_to_one(self):
        """All perturbed weight sets must sum to 1.0."""
        from xrspatial.mcda.sensitivity import _perturb_weights

        w = {"a": 0.4, "b": 0.35, "c": 0.25}
        for target in w:
            for delta in [0.01, 0.05, 0.10, 0.20]:
                wp, wm = _perturb_weights(w, target, delta)
                assert sum(wp.values()) == pytest.approx(1.0, abs=1e-10)
                assert sum(wm.values()) == pytest.approx(1.0, abs=1e-10)


class TestMonteCarloReproducibility:
    """Same seed must produce identical results."""

    def test_same_seed_same_result(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.random.rand(10, 10), dims=["y", "x"],
            ),
            "b": xr.DataArray(
                np.random.rand(10, 10), dims=["y", "x"],
            ),
        })
        w = {"a": 0.6, "b": 0.4}
        r1 = sensitivity(ds, w, method="monte_carlo", n_samples=30, seed=42)
        r2 = sensitivity(ds, w, method="monte_carlo", n_samples=30, seed=42)
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_different_seed_different_result(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.random.rand(10, 10), dims=["y", "x"],
            ),
            "b": xr.DataArray(
                np.random.rand(10, 10), dims=["y", "x"],
            ),
        })
        w = {"a": 0.6, "b": 0.4}
        r1 = sensitivity(ds, w, method="monte_carlo", n_samples=30, seed=42)
        r2 = sensitivity(ds, w, method="monte_carlo", n_samples=30, seed=99)
        assert not np.array_equal(r1.values, r2.values)


# ===========================================================================
# Behavioral edge cases
# ===========================================================================

class TestGaussianSmallStd:
    """Very small std approaches a delta function."""

    def test_tiny_std(self):
        data = np.array([[49.9, 50.0, 50.1]], dtype=np.float64)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(agg, method="gaussian", mean=50, std=1e-10)
        assert float(r.values[0, 0]) == pytest.approx(0.0, abs=1e-100)
        assert float(r.values[0, 1]) == pytest.approx(1.0)
        assert float(r.values[0, 2]) == pytest.approx(0.0, abs=1e-100)


class TestOWARiskAttitude:
    """Conservative OWA <= optimistic OWA for mixed-quality pixels."""

    def test_conservative_le_optimistic(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.array([[0.9]]), dims=["y", "x"],
            ),
            "b": xr.DataArray(
                np.array([[0.1]]), dims=["y", "x"],
            ),
            "c": xr.DataArray(
                np.array([[0.5]]), dims=["y", "x"],
            ),
        })
        w = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        conservative = owa(ds, w, [0.1, 0.2, 0.7])
        optimistic = owa(ds, w, [0.7, 0.2, 0.1])
        assert float(conservative.values[0, 0]) <= float(
            optimistic.values[0, 0]
        )


class TestRankWeightsManyCriteria:
    """Rank weights with many criteria should still sum to 1.0
    and decrease monotonically."""

    def test_20_criteria_roc(self):
        names = [f"c{i}" for i in range(20)]
        w = rank_weights(names, method="roc")
        vals = list(w.values())
        assert sum(vals) == pytest.approx(1.0)
        assert all(vals[i] >= vals[i + 1] for i in range(19))

    def test_20_criteria_rs(self):
        names = [f"c{i}" for i in range(20)]
        w = rank_weights(names, method="rs")
        vals = list(w.values())
        assert sum(vals) == pytest.approx(1.0)
        assert all(vals[i] >= vals[i + 1] for i in range(19))

    def test_20_criteria_rr(self):
        names = [f"c{i}" for i in range(20)]
        w = rank_weights(names, method="rr")
        vals = list(w.values())
        assert sum(vals) == pytest.approx(1.0)
        assert all(vals[i] >= vals[i + 1] for i in range(19))


class TestConstantSurface:
    """All-identical finite values with no explicit bounds."""

    def test_linear_constant_no_bounds(self):
        data = np.full((3, 3), 42.0)
        agg = xr.DataArray(data, dims=["y", "x"])
        r = standardize(agg, method="linear")
        np.testing.assert_allclose(r.values, 0.5)


class TestConstrainOverlappingMasks:
    """Multiple masks excluding the same pixel should still work."""

    def test_overlapping(self):
        suit = xr.DataArray(
            np.array([[0.8, 0.6, 0.9]], dtype=np.float64),
            dims=["y", "x"],
        )
        mask1 = xr.DataArray(
            np.array([[True, True, False]]), dims=["y", "x"],
        )
        mask2 = xr.DataArray(
            np.array([[True, False, False]]), dims=["y", "x"],
        )
        result = constrain(suit, exclude=[mask1, mask2])
        assert np.isnan(result.values[0, 0])  # excluded by both
        assert np.isnan(result.values[0, 1])  # excluded by mask1
        assert float(result.values[0, 2]) == pytest.approx(0.9)


# ===========================================================================
# Degenerate input shapes
# ===========================================================================

class TestSingleCriterionDataset:
    """Dataset with a single variable should work for all combine methods."""

    @pytest.fixture
    def single_ds(self):
        return xr.Dataset({
            "a": xr.DataArray(
                np.array([[0.3, 0.7, 0.5]], dtype=np.float64),
                dims=["y", "x"],
            ),
        })

    def test_wlc_single(self, single_ds):
        r = wlc(single_ds, {"a": 1.0})
        np.testing.assert_allclose(r.values, [[0.3, 0.7, 0.5]])

    def test_wpm_single(self, single_ds):
        r = wpm(single_ds, {"a": 1.0})
        np.testing.assert_allclose(r.values, [[0.3, 0.7, 0.5]])

    def test_owa_single(self, single_ds):
        r = owa(single_ds, {"a": 1.0}, [1.0])
        np.testing.assert_allclose(r.values, [[0.3, 0.7, 0.5]])

    def test_fuzzy_all_operators_single(self, single_ds):
        for op in ["and", "or", "product", "sum", "gamma"]:
            kw = {"gamma": 0.9} if op == "gamma" else {}
            r = fuzzy_overlay(single_ds, operator=op, **kw)
            np.testing.assert_allclose(
                r.values, [[0.3, 0.7, 0.5]], err_msg=f"fuzzy {op}",
            )


class TestBooleanOverlayNaN:
    """NaN cast to bool is True -- document this gotcha."""

    def test_nan_treated_as_true(self):
        a = xr.DataArray(
            np.array([[True, False]]), dims=["y", "x"],
        )
        b = xr.DataArray(
            np.array([[np.nan, np.nan]]), dims=["y", "x"],
        )
        # bool(NaN) = True, so NaN pixels are treated as True
        r_and = boolean_overlay({"a": a, "b": b}, operator="and")
        np.testing.assert_array_equal(r_and.values, [[True, False]])
        r_or = boolean_overlay({"a": a, "b": b}, operator="or")
        np.testing.assert_array_equal(r_or.values, [[True, True]])


class TestSensitivitySingleCriterion:
    """Single criterion: perturbation is a no-op, sensitivity is zero."""

    def test_oat_single_criterion(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.array([[0.5, 0.8]], dtype=np.float64),
                dims=["y", "x"],
            ),
        })
        r = sensitivity(ds, {"a": 1.0}, method="one_at_a_time", delta=0.05)
        assert isinstance(r, xr.Dataset)
        np.testing.assert_allclose(r["a"].values, 0.0, atol=1e-15)

    def test_mc_single_criterion(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.array([[0.5, 0.8]], dtype=np.float64),
                dims=["y", "x"],
            ),
        })
        r = sensitivity(
            ds, {"a": 1.0}, method="monte_carlo", n_samples=20,
        )
        # Single criterion -> all weight vectors are [1.0] -> zero variance
        np.testing.assert_allclose(r.values, 0.0, atol=1e-10)


class TestSensitivityWithWPM:
    """OAT and MC should work with WPM combine method."""

    def test_oat_wpm(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.array([[0.5, 0.8]], dtype=np.float64),
                dims=["y", "x"],
            ),
            "b": xr.DataArray(
                np.array([[0.3, 0.9]], dtype=np.float64),
                dims=["y", "x"],
            ),
        })
        r = sensitivity(
            ds, {"a": 0.6, "b": 0.4},
            method="one_at_a_time", combine_method="wpm", delta=0.05,
        )
        assert isinstance(r, xr.Dataset)
        for var in r.data_vars:
            assert np.all(np.isfinite(r[var].values))
            assert np.all(r[var].values >= 0)

    def test_mc_wpm(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.array([[0.5, 0.8]], dtype=np.float64),
                dims=["y", "x"],
            ),
            "b": xr.DataArray(
                np.array([[0.3, 0.9]], dtype=np.float64),
                dims=["y", "x"],
            ),
        })
        r = sensitivity(
            ds, {"a": 0.6, "b": 0.4},
            method="monte_carlo", combine_method="wpm", n_samples=20,
        )
        assert np.all(np.isfinite(r.values))
        assert np.all(r.values >= 0)


class TestAHPManyCriteria:
    """AHP with >15 criteria uses last _RI value as fallback."""

    def test_16_criteria_all_equal(self):
        names = [f"c{i}" for i in range(16)]
        comps = {}
        for i in range(16):
            for j in range(i + 1, 16):
                comps[(names[i], names[j])] = 1
        w, c = ahp_weights(names, comps)
        assert sum(w.values()) == pytest.approx(1.0)
        # All equal comparisons -> all equal weights
        for v in w.values():
            assert v == pytest.approx(1 / 16, abs=0.001)
        assert c.is_consistent


class TestDuplicateCriteriaNames:
    """Duplicate names in criteria lists should raise."""

    def test_ahp_duplicates_raise(self):
        with pytest.raises(ValueError, match="Duplicate"):
            ahp_weights(
                criteria=["a", "a", "b"],
                comparisons={("a", "b"): 3},
            )

    def test_rank_weights_duplicates_raise(self):
        with pytest.raises(ValueError, match="Duplicate"):
            rank_weights(["a", "a", "b"])


class TestConstrainIntMask:
    """Integer masks should work (nonzero = excluded)."""

    def test_int_mask(self):
        suit = xr.DataArray(
            np.array([[0.8, 0.6, 0.9]], dtype=np.float64),
            dims=["y", "x"],
        )
        mask = xr.DataArray(
            np.array([[1, 0, 0]], dtype=np.int32), dims=["y", "x"],
        )
        r = constrain(suit, exclude=[mask])
        assert np.isnan(r.values[0, 0])
        assert float(r.values[0, 1]) == pytest.approx(0.6)
        assert float(r.values[0, 2]) == pytest.approx(0.9)


class TestMonteCarloSingleSample:
    """n_samples=1 produces zero variance (CV=0 everywhere)."""

    def test_single_sample_zero_cv(self):
        ds = xr.Dataset({
            "a": xr.DataArray(
                np.random.rand(5, 5), dims=["y", "x"],
            ),
            "b": xr.DataArray(
                np.random.rand(5, 5), dims=["y", "x"],
            ),
        })
        r = sensitivity(
            ds, {"a": 0.6, "b": 0.4},
            method="monte_carlo", n_samples=1,
        )
        np.testing.assert_allclose(r.values, 0.0)


class TestThreeDimensionalInput:
    """3D DataArrays should work with explicit bounds."""

    def test_3d_linear(self):
        data = np.random.rand(3, 10, 10)
        agg = xr.DataArray(data, dims=["band", "y", "x"])
        r = standardize(agg, method="linear", bounds=(0, 1))
        assert r.shape == (3, 10, 10)
        # All values should be in [0, 1] since data is in [0, 1)
        assert np.nanmin(r.values) >= 0.0
        assert np.nanmax(r.values) <= 1.0

    def test_3d_gaussian(self):
        data = np.random.rand(2, 5, 5)
        agg = xr.DataArray(data, dims=["band", "y", "x"])
        r = standardize(agg, method="gaussian", mean=0.5, std=0.2)
        assert r.shape == (2, 5, 5)
        assert np.all(np.isfinite(r.values))


# ===========================================================================
# Integration: end-to-end workflow
# ===========================================================================

class TestEndToEnd:
    def test_site_suitability_workflow(self):
        """Full MCDA workflow: standardize -> weight -> combine -> constrain."""
        np.random.seed(1030)
        shape = (10, 10)
        slope = xr.DataArray(
            np.random.uniform(0, 45, shape), dims=["y", "x"],
        )
        distance = xr.DataArray(
            np.random.uniform(0, 10000, shape), dims=["y", "x"],
        )
        ndvi = xr.DataArray(
            np.random.uniform(0, 1, shape), dims=["y", "x"],
        )
        water = xr.DataArray(
            np.random.random(shape) > 0.9, dims=["y", "x"],
        )

        # Standardize
        slope_std = standardize(
            slope, method="linear",
            direction="lower_is_better", bounds=(0, 45),
        )
        dist_std = standardize(
            distance, method="sigmoidal", midpoint=5000, spread=-0.001,
        )
        ndvi_std = standardize(
            ndvi, method="gaussian", mean=0.65, std=0.15,
        )

        # Combine
        criteria = xr.Dataset({
            "slope": slope_std,
            "distance": dist_std,
            "ndvi": ndvi_std,
        })
        weights = {"slope": 0.4, "distance": 0.35, "ndvi": 0.25}
        suitability = wlc(criteria, weights)

        # Constrain
        suitability = constrain(suitability, exclude=[water])

        # Check output
        assert suitability.shape == shape
        # Water pixels should be NaN
        assert np.isnan(suitability.values[water.values]).all()
        # Non-water pixels should be in [0, 1]
        valid = suitability.values[~water.values & np.isfinite(suitability.values)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0


# ===========================================================================
# NaN/Inf weight rejection (#1311)
# ===========================================================================

class TestNonFiniteWeights1311:
    """Reject NaN and Inf weights before they propagate to all-NaN output.

    Without these guards, ``abs(NaN - 1.0) > 0.01`` is False, the
    sum-to-1 check passes, and the bad value reaches every pixel.
    """

    def test_wlc_rejects_nan_weight(self, criteria_dataset):
        bad = {"slope": 0.4, "distance": 0.35, "ndvi": float("nan")}
        with pytest.raises(ValueError, match="finite"):
            wlc(criteria_dataset, bad)

    def test_wlc_rejects_inf_weight(self, criteria_dataset):
        bad = {"slope": 0.4, "distance": 0.35, "ndvi": float("inf")}
        with pytest.raises(ValueError, match="finite"):
            wlc(criteria_dataset, bad)

    def test_wlc_rejects_negative_inf_weight(self, criteria_dataset):
        bad = {"slope": 0.4, "distance": 0.35, "ndvi": float("-inf")}
        with pytest.raises(ValueError, match="finite"):
            wlc(criteria_dataset, bad)

    def test_wlc_error_names_offending_criterion(self, criteria_dataset):
        bad = {"slope": 0.4, "distance": 0.35, "ndvi": float("nan")}
        with pytest.raises(ValueError, match="ndvi"):
            wlc(criteria_dataset, bad)

    def test_wpm_rejects_nan_weight(self, criteria_dataset):
        bad = {"slope": 0.4, "distance": 0.35, "ndvi": float("nan")}
        with pytest.raises(ValueError, match="finite"):
            wpm(criteria_dataset, bad)

    def test_wpm_rejects_inf_weight(self, criteria_dataset):
        bad = {"slope": 0.4, "distance": 0.35, "ndvi": float("inf")}
        with pytest.raises(ValueError, match="finite"):
            wpm(criteria_dataset, bad)

    def test_owa_rejects_nan_criterion_weight(self, criteria_dataset):
        bad = {"slope": 0.4, "distance": 0.35, "ndvi": float("nan")}
        with pytest.raises(ValueError, match="finite"):
            owa(criteria_dataset, bad, [0.5, 0.3, 0.2])

    def test_owa_rejects_nan_order_weight(self, criteria_dataset, weights_3):
        bad_order = [0.5, float("nan"), 0.5]
        with pytest.raises(ValueError, match="finite"):
            owa(criteria_dataset, weights_3, bad_order)

    def test_owa_rejects_inf_order_weight(self, criteria_dataset, weights_3):
        bad_order = [0.5, float("inf"), 0.5]
        with pytest.raises(ValueError, match="finite"):
            owa(criteria_dataset, weights_3, bad_order)

    def test_owa_order_weight_error_names_position(
        self, criteria_dataset, weights_3,
    ):
        bad_order = [0.5, float("nan"), 0.5]
        with pytest.raises(ValueError, match=r"\[1\]"):
            owa(criteria_dataset, weights_3, bad_order)

    def test_ahp_rejects_nan_comparison(self):
        with pytest.raises(ValueError, match="finite"):
            ahp_weights(
                ["a", "b", "c"],
                {("a", "b"): float("nan"), ("a", "c"): 2.0, ("b", "c"): 3.0},
            )

    def test_ahp_rejects_inf_comparison(self):
        with pytest.raises(ValueError, match="finite"):
            ahp_weights(
                ["a", "b", "c"],
                {("a", "b"): float("inf"), ("a", "c"): 2.0, ("b", "c"): 3.0},
            )

    def test_ahp_rejects_negative_inf_comparison(self):
        # -inf is non-finite; the finite check fires before <= 0.
        with pytest.raises(ValueError, match="finite"):
            ahp_weights(
                ["a", "b", "c"],
                {("a", "b"): float("-inf"), ("a", "c"): 2.0,
                 ("b", "c"): 3.0},
            )

    def test_ahp_still_rejects_zero(self):
        # Pre-existing positivity guard must still fire.
        with pytest.raises(ValueError, match="positive"):
            ahp_weights(
                ["a", "b", "c"],
                {("a", "b"): 0.0, ("a", "c"): 2.0, ("b", "c"): 3.0},
            )

    def test_ahp_still_rejects_negative(self):
        with pytest.raises(ValueError, match="positive"):
            ahp_weights(
                ["a", "b", "c"],
                {("a", "b"): -1.0, ("a", "c"): 2.0, ("b", "c"): 3.0},
            )

    def test_wlc_valid_weights_still_work(self, criteria_dataset, weights_3):
        # Sanity: the new check does not break the happy path.
        result = wlc(criteria_dataset, weights_3)
        assert np.all(np.isfinite(result.values))

    def test_owa_valid_weights_still_work(self, criteria_dataset, weights_3):
        result = owa(criteria_dataset, weights_3, [0.5, 0.3, 0.2])
        assert np.all(np.isfinite(result.values))

    def test_ahp_valid_comparisons_still_work(self):
        weights, _ = ahp_weights(
            ["a", "b", "c"],
            {("a", "b"): 2.0, ("a", "c"): 4.0, ("b", "c"): 2.0},
        )
        assert all(np.isfinite(v) for v in weights.values())


# ===========================================================================
# Memory guard for owa() (#1370)
# ===========================================================================

class TestOWAMemoryGuard:
    """Memory guard on the eager owa() path."""

    def test_oversize_raises(self, criteria_dataset, weights_3):
        """owa raises MemoryError when the criteria stack exceeds the budget."""
        from unittest.mock import patch

        with patch(
            "xrspatial.mcda.combine._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                owa(criteria_dataset, weights_3, [0.5, 0.3, 0.2])

    def test_normal_succeeds(self, criteria_dataset, weights_3):
        """A normal-size dataset passes the guard with real available memory."""
        result = owa(criteria_dataset, weights_3, [0.5, 0.3, 0.2])
        assert result.shape == criteria_dataset["slope"].shape
        assert np.all(np.isfinite(result.values))

    def test_error_message_names_dimensions_and_count(
        self, criteria_dataset, weights_3,
    ):
        """The error message identifies the criterion count and grid shape."""
        from unittest.mock import patch

        n = len(criteria_dataset.data_vars)
        h, w = criteria_dataset["slope"].shape
        shape_token = f"{h}x{w}"

        with patch(
            "xrspatial.mcda.combine._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError) as exc_info:
                owa(criteria_dataset, weights_3, [0.5, 0.3, 0.2])

        msg = str(exc_info.value)
        assert f"{n} criteria" in msg
        assert shape_token in msg
        assert "dask" in msg
