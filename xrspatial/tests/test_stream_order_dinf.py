import math

import numpy as np
import pytest
import xarray as xr

from xrspatial.stream_order_dinf import stream_order_dinf
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)

# D-inf angle convention (counterclockwise from East):
#   E=0, NE=pi/4, N=pi/2, NW=3pi/4, W=pi, SW=5pi/4, S=3pi/2, SE=7pi/4
PI4 = math.pi / 4.0
ANGLE_E  = 0.0
ANGLE_NE = PI4
ANGLE_N  = 2 * PI4
ANGLE_NW = 3 * PI4
ANGLE_W  = 4 * PI4
ANGLE_SW = 5 * PI4
ANGLE_S  = 6 * PI4
ANGLE_SE = 7 * PI4
PIT = -1.0


# ====================================================================
# Helpers
# ====================================================================

def _call(angles, accum, threshold=0, **kwargs):
    """Wrap raw arrays in DataArrays and call stream_order_dinf."""
    a_da = create_test_raster(angles)
    fa_da = create_test_raster(accum)
    return stream_order_dinf(a_da, fa_da, threshold=threshold, **kwargs)


# ====================================================================
# Strahler tests
# ====================================================================

def test_y_confluence_strahler():
    """Two order-1 streams merge -> order 2 downstream.

    Same topology as D8 test but with D-inf angles.
    (0,0)->SE to (1,1), (0,2)->SW to (1,1), (1,1)->S to (2,1), (2,1) pit
    """
    angles = np.array([
        [ANGLE_SE, PIT, ANGLE_SW],
        [PIT,      ANGLE_S, PIT],
        [PIT,      PIT,     PIT],
    ], dtype=np.float64)
    accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 4.0, 1.0],
    ], dtype=np.float64)
    result = _call(angles, accum, threshold=1, method='strahler')
    assert result.data[0, 0] == 1.0
    assert result.data[0, 2] == 1.0
    assert result.data[1, 1] == 2.0
    assert result.data[2, 1] == 2.0


def test_unequal_confluence_strahler():
    """Order 1 meets order 2 -> stays order 2."""
    angles = np.array([
        [ANGLE_SE, PIT, ANGLE_SW],
        [PIT,      ANGLE_S, PIT],
        [ANGLE_E,  PIT,     PIT],
    ], dtype=np.float64)
    accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 5.0, 1.0],
    ], dtype=np.float64)
    result = _call(angles, accum, threshold=1, method='strahler')
    assert result.data[0, 0] == 1.0
    assert result.data[0, 2] == 1.0
    assert result.data[1, 1] == 2.0
    assert result.data[2, 1] == 2.0


# ====================================================================
# Shreve tests
# ====================================================================

def test_y_confluence_shreve():
    """Two headwaters merge -> Shreve magnitude = 2."""
    angles = np.array([
        [ANGLE_SE, PIT, ANGLE_SW],
        [PIT,      ANGLE_S, PIT],
        [PIT,      PIT,     PIT],
    ], dtype=np.float64)
    accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 4.0, 1.0],
    ], dtype=np.float64)
    result = _call(angles, accum, threshold=1, method='shreve')
    assert result.data[0, 0] == 1.0
    assert result.data[0, 2] == 1.0
    assert result.data[1, 1] == 2.0
    assert result.data[2, 1] == 2.0


# ====================================================================
# D-inf specific: split angle
# ====================================================================

def test_split_angle_strahler():
    """Cell with angle between two cardinals flows to both neighbors.

    H(0,0) has angle pi/8 (between E and NE), so it flows to both
    (0,1) via E and (-1,1) via NE.  Only (0,1) is in bounds.
    So effectively one downstream neighbor -> (0,1) gets order 1.
    """
    # angle = pi/8 means between E(0) and NE(pi/4)
    # k=0 (E), k2=1 (NE), frac1=0.5, frac2=0.5
    # Neighbor E: (0, 0+1) = (0, 1) -- in bounds
    # Neighbor NE: (0-1, 0+1) = (-1, 1) -- out of bounds
    angles = np.array([
        [math.pi / 8.0, PIT],
    ], dtype=np.float64)
    accum = np.array([[5.0, 5.0]], dtype=np.float64)
    result = _call(angles, accum, threshold=1, method='strahler')
    assert result.data[0, 0] == 1.0  # headwater
    assert result.data[0, 1] == 1.0  # single inflow


# ====================================================================
# Edge cases
# ====================================================================

def test_nan_handling():
    """NaN angles -> non-stream (NaN output)."""
    angles = np.full((2, 2), np.nan, dtype=np.float64)
    angles[0, 0] = PIT  # valid pit
    accum = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    result = _call(angles, accum, threshold=1, method='strahler')
    assert result.data[0, 0] == 1.0
    assert np.isnan(result.data[0, 1])
    assert np.isnan(result.data[1, 0])


def test_pit_handling():
    """Pit cells (angle=-1) with high accum are headwaters."""
    angles = np.array([[PIT]], dtype=np.float64)
    accum = np.array([[10.0]], dtype=np.float64)
    result = _call(angles, accum, threshold=1, method='strahler')
    assert result.data[0, 0] == 1.0


def test_threshold_excludes():
    """Cells below threshold are NaN."""
    angles = np.array([[ANGLE_E, PIT]], dtype=np.float64)
    accum = np.array([[1.0, 100.0]], dtype=np.float64)
    result = _call(angles, accum, threshold=50, method='strahler')
    assert np.isnan(result.data[0, 0])
    assert result.data[0, 1] == 1.0


def test_invalid_method():
    """Invalid method raises ValueError."""
    angles = np.array([[PIT]], dtype=np.float64)
    accum = np.array([[1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="method must be"):
        _call(angles, accum, threshold=0, method='bogus')


@dask_array_available
def test_dask_matches_numpy():
    """Dask result matches numpy."""
    import dask.array as da

    angles = np.array([
        [ANGLE_SE, PIT, ANGLE_SW],
        [PIT,      ANGLE_S, PIT],
        [ANGLE_E,  PIT,     PIT],
    ], dtype=np.float64)
    accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 5.0, 1.0],
    ], dtype=np.float64)

    a_np = create_test_raster(angles)
    fa_np = create_test_raster(accum)
    np_result = stream_order_dinf(a_np, fa_np, threshold=1, method='strahler')

    a_dask = xr.DataArray(
        da.from_array(angles, chunks=(2, 2)),
        dims=['y', 'x'])
    fa_dask = xr.DataArray(
        da.from_array(accum, chunks=(2, 2)),
        dims=['y', 'x'])
    dask_result = stream_order_dinf(a_dask, fa_dask, threshold=1,
                                     method='strahler')

    np.testing.assert_array_equal(
        np.nan_to_num(np_result.values, nan=-999),
        np.nan_to_num(dask_result.values, nan=-999))
