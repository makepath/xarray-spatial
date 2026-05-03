"""Tests for issue #1427: hydro scalar parameter validation.

Several public functions accept scalar parameters that previously did not
reject NaN/Inf or out-of-range values, leading to silent all-NaN output or
no-op behavior.
"""

import numpy as np
import pytest

from xrspatial.hydro import (
    fill_d8,
    flow_direction_d8,
    flow_direction_dinf,
    flow_direction_mfd,
    hand_d8,
    hand_dinf,
    hand_mfd,
    snap_pour_point_d8,
)
from xrspatial.tests.general_checks import create_test_raster


def _elev():
    return create_test_raster(np.array([
        [9, 9, 9, 9, 9],
        [9, 8, 7, 6, 9],
        [9, 7, 5, 4, 9],
        [9, 6, 4, 3, 9],
        [9, 9, 9, 9, 9],
    ], dtype=np.float64))


def _stream_inputs(method):
    elev = _elev()
    if method == 'd8':
        fd = flow_direction_d8(elev)
    elif method == 'dinf':
        fd = flow_direction_dinf(elev)
    else:
        fd = flow_direction_mfd(elev)
    fa = create_test_raster(np.ones((5, 5), dtype=np.float64))
    el = elev
    return fd, fa, el


# ---------------------------------------------------------------------------
# flow_direction_mfd p
# ---------------------------------------------------------------------------

class TestFlowDirectionMfdP:
    @pytest.mark.parametrize("p", [float('nan'), float('inf'), float('-inf')])
    def test_rejects_non_finite_p(self, p):
        with pytest.raises(ValueError, match="positive finite"):
            flow_direction_mfd(_elev(), p=p)

    @pytest.mark.parametrize("p", [0, -1, -0.5])
    def test_rejects_non_positive_p(self, p):
        with pytest.raises(ValueError, match="positive finite"):
            flow_direction_mfd(_elev(), p=p)

    def test_accepts_positive_finite_p(self):
        result = flow_direction_mfd(_elev(), p=1.5)
        assert result.shape == (8, 5, 5)


# ---------------------------------------------------------------------------
# snap_pour_point_d8 search_radius
# ---------------------------------------------------------------------------

class TestSnapPourPointSearchRadius:
    @pytest.mark.parametrize("r", [0, -1, -5])
    def test_rejects_non_positive(self, r):
        fa = create_test_raster(np.ones((5, 5), dtype=np.float64))
        pp = create_test_raster(np.full((5, 5), np.nan, dtype=np.float64))
        with pytest.raises(ValueError, match="positive integer"):
            snap_pour_point_d8(fa, pp, search_radius=r)

    @pytest.mark.parametrize("r", [5.5, float('nan'), float('inf')])
    def test_rejects_non_int(self, r):
        fa = create_test_raster(np.ones((5, 5), dtype=np.float64))
        pp = create_test_raster(np.full((5, 5), np.nan, dtype=np.float64))
        with pytest.raises(ValueError, match="positive integer"):
            snap_pour_point_d8(fa, pp, search_radius=r)


# ---------------------------------------------------------------------------
# hand_*  threshold
# ---------------------------------------------------------------------------

class TestHandThreshold:
    @pytest.mark.parametrize("method,fn", [
        ('d8', hand_d8),
        ('dinf', hand_dinf),
        ('mfd', hand_mfd),
    ])
    @pytest.mark.parametrize("t", [float('nan'), float('inf'), float('-inf')])
    def test_rejects_non_finite_threshold(self, method, fn, t):
        fd, fa, el = _stream_inputs(method)
        with pytest.raises(ValueError, match="threshold must be a finite"):
            fn(fd, fa, el, threshold=t)


# ---------------------------------------------------------------------------
# fill_d8 z_limit
# ---------------------------------------------------------------------------

class TestFillZLimit:
    @pytest.mark.parametrize("z", [float('nan'), float('inf'), -0.5, -100])
    def test_rejects_bad_z_limit(self, z):
        with pytest.raises(ValueError, match="z_limit"):
            fill_d8(_elev(), z_limit=z)

    def test_accepts_none(self):
        result = fill_d8(_elev(), z_limit=None)
        assert result.shape == (5, 5)

    def test_accepts_zero(self):
        result = fill_d8(_elev(), z_limit=0)
        assert result.shape == (5, 5)

    def test_accepts_positive(self):
        result = fill_d8(_elev(), z_limit=1.0)
        assert result.shape == (5, 5)
