"""Input-validation tests for xrspatial.resample (issue #2574).

`target_resolution` and `scale_factor` must fail fast with a clear
``ValueError`` for:
  * overlong tuples (length > 2)
  * short tuples (length < 2)
  * zero / negative components
  * non-finite values (NaN, inf)

These used to surface deep inside the function as IndexError /
ZeroDivisionError / opaque numpy conversion errors.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrspatial.resample import resample
from xrspatial.tests.general_checks import create_test_raster


@pytest.fixture
def grid_4x4():
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    return create_test_raster(
        data, backend='numpy', attrs={'res': (1.0, 1.0)},
    )


# ---------------------------------------------------------------------------
# target_resolution
# ---------------------------------------------------------------------------

class TestTargetResolutionValidation:
    def test_overlong_tuple(self, grid_4x4):
        with pytest.raises(
            ValueError,
            match=r"target_resolution must have length 2, got length 3",
        ):
            resample(grid_4x4, target_resolution=(1.0, 1.0, 1.0))

    def test_short_tuple(self, grid_4x4):
        with pytest.raises(
            ValueError,
            match=r"target_resolution must have length 2, got length 1",
        ):
            resample(grid_4x4, target_resolution=(1.0,))

    def test_zero_scalar(self, grid_4x4):
        with pytest.raises(
            ValueError, match=r"target_resolution must be > 0, got 0",
        ):
            resample(grid_4x4, target_resolution=0)

    def test_zero_component(self, grid_4x4):
        # Pair errors point at the failing index.
        with pytest.raises(
            ValueError,
            match=r"target_resolution must be > 0, got 0.0 at index 0 of",
        ):
            resample(grid_4x4, target_resolution=(0.0, 1.0))

    def test_negative_scalar(self, grid_4x4):
        with pytest.raises(
            ValueError, match=r"target_resolution must be > 0, got -2.0",
        ):
            resample(grid_4x4, target_resolution=-2.0)

    def test_nan_scalar(self, grid_4x4):
        with pytest.raises(
            ValueError,
            match=r"target_resolution must be finite and > 0, got nan",
        ):
            resample(grid_4x4, target_resolution=float('nan'))

    def test_inf_scalar(self, grid_4x4):
        with pytest.raises(
            ValueError,
            match=r"target_resolution must be finite and > 0, got inf",
        ):
            resample(grid_4x4, target_resolution=float('inf'))


# ---------------------------------------------------------------------------
# scale_factor
# ---------------------------------------------------------------------------

class TestScaleFactorValidation:
    def test_overlong_tuple(self, grid_4x4):
        with pytest.raises(
            ValueError,
            match=r"scale_factor must have length 2, got length 3",
        ):
            resample(grid_4x4, scale_factor=(2.0, 2.0, 2.0))

    def test_short_tuple(self, grid_4x4):
        with pytest.raises(
            ValueError,
            match=r"scale_factor must have length 2, got length 1",
        ):
            resample(grid_4x4, scale_factor=(2.0,))

    def test_zero_scalar(self, grid_4x4):
        with pytest.raises(
            ValueError, match=r"scale_factor must be > 0, got 0",
        ):
            resample(grid_4x4, scale_factor=0)

    def test_zero_component(self, grid_4x4):
        # Pair errors point at the failing index.
        with pytest.raises(
            ValueError,
            match=r"scale_factor must be > 0, got 0.0 at index 0 of",
        ):
            resample(grid_4x4, scale_factor=(0.0, 2.0))

    def test_nan_scalar(self, grid_4x4):
        with pytest.raises(
            ValueError,
            match=r"scale_factor must be finite and > 0, got nan",
        ):
            resample(grid_4x4, scale_factor=float('nan'))

    def test_inf_scalar(self, grid_4x4):
        with pytest.raises(
            ValueError,
            match=r"scale_factor must be finite and > 0, got inf",
        ):
            resample(grid_4x4, scale_factor=float('inf'))


# ---------------------------------------------------------------------------
# Valid inputs still work (regression guard)
# ---------------------------------------------------------------------------

class TestValidInputsStillWork:
    def test_scalar_scale_factor(self, grid_4x4):
        out = resample(grid_4x4, scale_factor=2.0)
        assert out.shape == (8, 8)

    def test_tuple_scale_factor(self, grid_4x4):
        out = resample(grid_4x4, scale_factor=(0.5, 0.5))
        assert out.shape == (2, 2)

    def test_scalar_target_resolution(self, grid_4x4):
        out = resample(grid_4x4, target_resolution=0.5)
        assert out.shape == (8, 8)

    def test_tuple_target_resolution(self, grid_4x4):
        out = resample(grid_4x4, target_resolution=(0.5, 0.5))
        assert out.shape == (8, 8)
