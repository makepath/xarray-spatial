import math

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro.stream_link_dinf import stream_link_dinf
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)

# D-inf angle convention (counterclockwise from East):
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
    """Wrap raw arrays and call stream_link_dinf."""
    a_da = create_test_raster(angles)
    fa_da = create_test_raster(accum)
    return stream_link_dinf(a_da, fa_da, threshold=threshold, **kwargs)


# ====================================================================
# Tests
# ====================================================================

def test_linear_chain():
    """Single stream, no junctions -> all one link_id."""
    angles = np.array([[ANGLE_E, ANGLE_E, ANGLE_E, ANGLE_E, PIT]],
                      dtype=np.float64)
    accum = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    result = _call(angles, accum, threshold=1)
    vals = result.data
    assert not np.any(np.isnan(vals))
    unique = np.unique(vals)
    assert len(unique) == 1
    assert unique[0] == 1.0  # headwater at (0,0)


def test_y_confluence():
    """Two headwaters merge -> 3 link IDs."""
    angles = np.array([
        [ANGLE_SE, np.nan, ANGLE_SW],
        [np.nan,   ANGLE_S, np.nan],
        [np.nan,   PIT,     np.nan],
    ], dtype=np.float64)
    accum = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float64)
    result = _call(angles, accum, threshold=1)
    vals = result.data

    assert np.isnan(vals[0, 1])
    assert np.isnan(vals[1, 0])
    # (0,0) headwater: ID = 0*3 + 0 + 1 = 1
    assert vals[0, 0] == 1.0
    # (0,2) headwater: ID = 0*3 + 2 + 1 = 3
    assert vals[0, 2] == 3.0
    # (1,1) junction: ID = 1*3 + 1 + 1 = 5
    assert vals[1, 1] == 5.0
    # (2,1) inherits from junction: ID = 5
    assert vals[2, 1] == 5.0
    stream_vals = vals[~np.isnan(vals)]
    assert len(np.unique(stream_vals)) == 3


def test_cascade_junctions():
    """Sequential junctions."""
    # A(0,0)->E, C(1,0)->NE => B(0,1) junction
    # B(0,1)->E, E(1,2)->N  => D(0,2) junction
    # D(0,2)->E              => F(0,3) pit
    angles = np.array([
        [ANGLE_E, ANGLE_E, ANGLE_E, PIT],
        [ANGLE_NE, np.nan, ANGLE_N, np.nan],
    ], dtype=np.float64)
    accum = np.array([
        [1.0, 3.0, 5.0, 6.0],
        [1.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)
    result = _call(angles, accum, threshold=1)
    vals = result.data

    assert vals[0, 0] == 1.0       # headwater
    assert vals[1, 0] == 5.0       # headwater: 1*4 + 0 + 1
    assert vals[0, 1] == 2.0       # junction: 0*4 + 1 + 1
    assert vals[1, 2] == 7.0       # headwater: 1*4 + 2 + 1
    assert vals[0, 2] == 3.0       # junction: 0*4 + 2 + 1
    assert vals[0, 3] == 3.0       # inherits from D


# ====================================================================
# Edge cases
# ====================================================================

def test_nan_handling():
    """NaN angles -> NaN output."""
    angles = np.full((2, 2), np.nan, dtype=np.float64)
    angles[0, 0] = PIT
    accum = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    result = _call(angles, accum, threshold=1)
    assert not np.isnan(result.data[0, 0])
    assert np.isnan(result.data[0, 1])


def test_pit_as_stream():
    """Pit with high accum is headwater link."""
    angles = np.array([[PIT]], dtype=np.float64)
    accum = np.array([[10.0]], dtype=np.float64)
    result = _call(angles, accum, threshold=1)
    assert result.data[0, 0] == 1.0


def test_single_cell():
    """Single cell -> headwater ID."""
    angles = np.array([[PIT]], dtype=np.float64)
    accum = np.array([[5.0]], dtype=np.float64)
    result = _call(angles, accum, threshold=1)
    assert result.data[0, 0] == 1.0


@dask_array_available
def test_dask_matches_numpy():
    """Dask result matches numpy."""
    import dask.array as da

    angles = np.array([
        [ANGLE_SE, np.nan, ANGLE_SW],
        [np.nan,   ANGLE_S, np.nan],
        [np.nan,   PIT,     np.nan],
    ], dtype=np.float64)
    accum = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float64)

    a_np = create_test_raster(angles)
    fa_np = create_test_raster(accum)
    np_result = stream_link_dinf(a_np, fa_np, threshold=1)

    a_dask = xr.DataArray(
        da.from_array(angles, chunks=(2, 2)),
        dims=['y', 'x'])
    fa_dask = xr.DataArray(
        da.from_array(accum, chunks=(2, 2)),
        dims=['y', 'x'])
    dask_result = stream_link_dinf(a_dask, fa_dask, threshold=1)

    np.testing.assert_array_equal(
        np.nan_to_num(np_result.values, nan=-999),
        np.nan_to_num(dask_result.values, nan=-999))
