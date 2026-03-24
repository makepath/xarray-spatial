"""Tests for fused_overlap and helpers."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import _normalize_depth, _pad_nan


class TestNormalizeDepth:
    def test_int_input(self):
        assert _normalize_depth(2, ndim=2) == {0: 2, 1: 2}

    def test_tuple_input(self):
        assert _normalize_depth((3, 1), ndim=2) == {0: 3, 1: 1}

    def test_dict_input(self):
        assert _normalize_depth({0: 2, 1: 4}, ndim=2) == {0: 2, 1: 4}

    def test_dict_missing_axis_raises(self):
        with pytest.raises(ValueError, match="missing axes"):
            _normalize_depth({0: 1}, ndim=2)

    def test_dict_extra_axis_raises(self):
        with pytest.raises(ValueError, match="extra axes"):
            _normalize_depth({0: 1, 1: 1, 2: 1}, ndim=2)

    def test_negative_depth_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _normalize_depth(-1, ndim=2)

    def test_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            _normalize_depth((1, 2, 3), ndim=2)


class TestPadNan:
    def test_2d_pads_with_nan(self):
        data = np.ones((4, 4), dtype=np.float32)
        result = _pad_nan(data, depth=(1, 1))
        assert result.shape == (6, 6)
        assert np.isnan(result[0, 0])
        np.testing.assert_array_equal(result[1:-1, 1:-1], data)

    def test_asymmetric_depth(self):
        data = np.ones((4, 4), dtype=np.float32)
        result = _pad_nan(data, depth=(2, 1))
        assert result.shape == (8, 6)

    def test_integer_dtype_promotes_to_float(self):
        data = np.ones((4, 4), dtype=np.int32)
        result = _pad_nan(data, depth=(1, 1))
        assert np.issubdtype(result.dtype, np.floating)
