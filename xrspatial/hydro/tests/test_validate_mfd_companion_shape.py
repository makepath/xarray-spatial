"""Tests for issue #2863: MFD hydro APIs validate companion raster shape.

Each MFD function below validated that its companion raster (start points,
pour points, flow accumulation, elevation) is a 2-D DataArray, but never
checked that the companion's (H, W) matched the primary MFD grid.  A
mismatched companion let the CPU kernel read out of bounds and return
uninitialised garbage instead of raising.  The fix adds an early
``ValueError`` when the shapes disagree.
"""

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro import (
    flow_path_mfd,
    hand_mfd,
    stream_link_mfd,
    stream_order_mfd,
    watershed_mfd,
)
from xrspatial.tests.general_checks import create_test_raster


def _mfd_fractions(H, W):
    """All-east MFD fraction grid of shape (8, H, W)."""
    fracs = np.zeros((8, H, W), dtype=np.float64)
    fracs[0, :, :-1] = 1.0  # E for non-pit cells
    neighbor = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']
    return xr.DataArray(
        fracs,
        dims=('neighbor', 'y', 'x'),
        coords={
            'neighbor': neighbor,
            'y': np.arange(H, dtype=np.float64),
            'x': np.arange(W, dtype=np.float64),
        },
        name='mfd_fdir',
        attrs={'res': (1.0, 1.0)},
    )


def _companion(H, W, fill=1.0):
    return create_test_raster(
        np.full((H, W), fill, dtype=np.float64), name='companion')


class TestFlowPathMfdShape:
    def test_mismatched_start_points_raises(self):
        fr = _mfd_fractions(3, 3)
        sp = _companion(1, 1, fill=5.0)
        with pytest.raises(ValueError, match='does not match'):
            flow_path_mfd(fr, sp)

    def test_matching_start_points_ok(self):
        fr = _mfd_fractions(3, 3)
        sp = create_test_raster(
            np.full((3, 3), np.nan, dtype=np.float64), name='sp')
        sp.data[0, 0] = 7.0
        out = flow_path_mfd(fr, sp)
        assert out.shape == (3, 3)
        # All-east flow: the path from (0, 0) labels the whole top row 7.
        np.testing.assert_array_equal(out.data[0], [7.0, 7.0, 7.0])
        assert np.isnan(out.data[1:]).all()


class TestWatershedMfdShape:
    def test_mismatched_pour_points_raises(self):
        fr = _mfd_fractions(3, 3)
        pp = _companion(2, 4, fill=1.0)
        with pytest.raises(ValueError, match='does not match'):
            watershed_mfd(fr, pp)

    def test_matching_pour_points_ok(self):
        fr = _mfd_fractions(3, 3)
        pp = create_test_raster(
            np.full((3, 3), np.nan, dtype=np.float64), name='pp')
        pp.data[0, 2] = 1.0
        out = watershed_mfd(fr, pp)
        assert out.shape == (3, 3)


class TestHandMfdShape:
    def test_mismatched_flow_accum_raises(self):
        fr = _mfd_fractions(3, 3)
        fa = _companion(2, 2)
        el = _companion(3, 3)
        with pytest.raises(ValueError, match='`flow_accum`.*does not match'):
            hand_mfd(fr, fa, el)

    def test_mismatched_elevation_raises(self):
        fr = _mfd_fractions(3, 3)
        fa = _companion(3, 3)
        el = _companion(4, 4)
        with pytest.raises(ValueError, match='`elevation`.*does not match'):
            hand_mfd(fr, fa, el)

    def test_matching_companions_ok(self):
        fr = _mfd_fractions(3, 3)
        fa = _companion(3, 3, fill=5.0)
        el = _companion(3, 3, fill=10.0)
        out = hand_mfd(fr, fa, el, threshold=3)
        assert out.shape == (3, 3)


class TestStreamOrderMfdShape:
    def test_mismatched_flow_accum_raises(self):
        fr = _mfd_fractions(3, 3)
        fa = _companion(1, 3)
        with pytest.raises(ValueError, match='does not match'):
            stream_order_mfd(fr, fa)

    def test_matching_flow_accum_ok(self):
        fr = _mfd_fractions(3, 3)
        fa = _companion(3, 3, fill=5.0)
        out = stream_order_mfd(fr, fa, threshold=1)
        assert out.shape == (3, 3)


class TestStreamLinkMfdShape:
    def test_mismatched_flow_accum_raises(self):
        fr = _mfd_fractions(3, 3)
        fa = _companion(3, 1)
        with pytest.raises(ValueError, match='does not match'):
            stream_link_mfd(fr, fa)

    def test_matching_flow_accum_ok(self):
        fr = _mfd_fractions(3, 3)
        fa = _companion(3, 3, fill=5.0)
        out = stream_link_mfd(fr, fa, threshold=1)
        assert out.shape == (3, 3)
