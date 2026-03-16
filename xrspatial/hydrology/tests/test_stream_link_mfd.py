import numpy as np
import pytest
import xarray as xr

from xrspatial.hydrology.stream_link_mfd import stream_link_mfd
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ====================================================================
# Helpers
# ====================================================================

def _make_fractions(dirs, shape):
    """Build (8, H, W) fractions from a dict of {(r,c): [(k, frac), ...]}.

    Cells not in *dirs* get NaN (nodata).  Cells with empty lists are
    pits (all-zero fractions).
    """
    H, W = shape
    fracs = np.full((8, H, W), np.nan, dtype=np.float64)
    for (r, c), entries in dirs.items():
        fracs[:, r, c] = 0.0
        for k, f in entries:
            fracs[k, r, c] = f
    return fracs


def _call(fracs, accum, threshold=0, **kwargs):
    """Wrap raw arrays and call stream_link_mfd."""
    frac_da = xr.DataArray(fracs, dims=['neighbor', 'y', 'x'])
    fa_da = create_test_raster(accum)
    return stream_link_mfd(frac_da, fa_da, threshold=threshold, **kwargs)


# ====================================================================
# Tests
# ====================================================================

def test_linear_chain():
    """Single stream, no junctions -> all one link_id."""
    fracs = _make_fractions({
        (0, 0): [(0, 1.0)],  # E
        (0, 1): [(0, 1.0)],  # E
        (0, 2): [(0, 1.0)],  # E
        (0, 3): [(0, 1.0)],  # E
        (0, 4): [],           # pit
    }, (1, 5))
    accum = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    result = _call(fracs, accum, threshold=1)
    vals = result.data
    assert not np.any(np.isnan(vals))
    unique = np.unique(vals[~np.isnan(vals)])
    assert len(unique) == 1
    # Headwater at (0,0), width=5 -> ID = 0*5 + 0 + 1 = 1
    assert unique[0] == 1.0


def test_y_confluence():
    """Two headwaters merge at junction -> 3 distinct link_ids."""
    fracs = _make_fractions({
        (0, 0): [(1, 1.0)],   # SE
        (0, 2): [(3, 1.0)],   # SW
        (1, 1): [(2, 1.0)],   # S
        (2, 1): [],            # pit
    }, (3, 3))
    accum = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float64)
    result = _call(fracs, accum, threshold=1)
    vals = result.data

    # Non-stream cells are NaN
    assert np.isnan(vals[0, 1])
    assert np.isnan(vals[1, 0])
    # (0,0) headwater: ID = 0*3 + 0 + 1 = 1
    assert vals[0, 0] == 1.0
    # (0,2) headwater: ID = 0*3 + 2 + 1 = 3
    assert vals[0, 2] == 3.0
    # (1,1) junction (in_degree=2): ID = 1*3 + 1 + 1 = 5
    assert vals[1, 1] == 5.0
    # (2,1) inherits from (1,1): ID = 5
    assert vals[2, 1] == 5.0
    # 3 distinct IDs
    stream_vals = vals[~np.isnan(vals)]
    assert len(np.unique(stream_vals)) == 3


def test_cascade_junctions():
    """Sequential junctions -> each segment has distinct ID.

    A(0,0)->E, C(1,0)->NE(7) => B(0,1) is junction
    B(0,1)->E, E(1,2)->N(6)  => D(0,2) is junction
    D(0,2)->E                => F(0,3) pit
    """
    fracs = _make_fractions({
        (0, 0): [(0, 1.0)],   # E to (0,1)
        (0, 1): [(0, 1.0)],   # E to (0,2)
        (0, 2): [(0, 1.0)],   # E to (0,3)
        (0, 3): [],            # pit
        (1, 0): [(7, 1.0)],   # NE to (0,1)
        (1, 2): [(6, 1.0)],   # N to (0,2)
    }, (2, 4))
    accum = np.array([
        [1.0, 3.0, 5.0, 6.0],
        [1.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)
    result = _call(fracs, accum, threshold=1)
    vals = result.data

    # A(0,0) headwater: link_id = 0*4 + 0 + 1 = 1
    assert vals[0, 0] == 1.0
    # C(1,0) headwater: link_id = 1*4 + 0 + 1 = 5
    assert vals[1, 0] == 5.0
    # B(0,1) junction: link_id = 0*4 + 1 + 1 = 2
    assert vals[0, 1] == 2.0
    # E(1,2) headwater: link_id = 1*4 + 2 + 1 = 7
    assert vals[1, 2] == 7.0
    # D(0,2) junction: link_id = 0*4 + 2 + 1 = 3
    assert vals[0, 2] == 3.0
    # F(0,3) inherits from D: link_id = 3
    assert vals[0, 3] == 3.0


# ====================================================================
# Edge cases
# ====================================================================

def test_nan_handling():
    """NaN fractions -> NaN output."""
    fracs = np.full((8, 2, 2), np.nan, dtype=np.float64)
    fracs[:, 0, 0] = 0.0  # valid pit
    accum = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    result = _call(fracs, accum, threshold=1)
    assert not np.isnan(result.data[0, 0])
    assert np.isnan(result.data[0, 1])


def test_single_cell():
    """Single cell -> headwater with position-based ID."""
    fracs = np.zeros((8, 1, 1), dtype=np.float64)
    accum = np.array([[5.0]], dtype=np.float64)
    result = _call(fracs, accum, threshold=1)
    assert result.data[0, 0] == 1.0  # 0*1 + 0 + 1


@dask_array_available
def test_dask_matches_numpy():
    """Dask result matches numpy."""
    import dask.array as da

    fracs = _make_fractions({
        (0, 0): [(1, 1.0)],
        (0, 2): [(3, 1.0)],
        (1, 1): [(2, 1.0)],
        (2, 1): [],
    }, (3, 3))
    accum = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float64)

    frac_np = xr.DataArray(fracs, dims=['neighbor', 'y', 'x'])
    fa_np = create_test_raster(accum)
    np_result = stream_link_mfd(frac_np, fa_np, threshold=1)

    frac_dask = xr.DataArray(
        da.from_array(fracs, chunks=(8, 2, 2)),
        dims=['neighbor', 'y', 'x'])
    fa_dask = xr.DataArray(
        da.from_array(accum, chunks=(2, 2)),
        dims=['y', 'x'])
    dask_result = stream_link_mfd(frac_dask, fa_dask, threshold=1)

    np.testing.assert_array_equal(
        np.nan_to_num(np_result.values, nan=-999),
        np.nan_to_num(dask_result.values, nan=-999))
