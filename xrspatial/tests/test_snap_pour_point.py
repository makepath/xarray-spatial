import numpy as np
import pytest
import xarray as xr

from xrspatial import snap_pour_point
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_accum_and_pp(accum_data, pp_data, backend='numpy', chunks=(3, 3)):
    """Build flow_accum and pour_points DataArrays for the given backend."""
    accum = create_test_raster(
        accum_data.astype(np.float64), backend=backend, chunks=chunks)
    pp = create_test_raster(
        pp_data.astype(np.float64), backend=backend, chunks=chunks)
    return accum, pp


# -------------------------------------------------------------------
# Basic functionality tests
# -------------------------------------------------------------------

def test_single_pour_point():
    """One pour point snaps to max-accum cell in radius."""
    accum = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 100.0],
    ])
    pp = np.full((3, 3), np.nan)
    pp[0, 0] = 1.0  # pour point at (0,0)

    fa, pour = _make_accum_and_pp(accum, pp)
    result = snap_pour_point(fa, pour, search_radius=3)

    # Should snap to (2,2) which has accum=100
    assert result.data[2, 2] == 1.0
    # Original location should be NaN
    assert np.isnan(result.data[0, 0])


def test_no_move_when_already_at_max():
    """Pour point already at peak stays put."""
    accum = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 100.0],
    ])
    pp = np.full((3, 3), np.nan)
    pp[2, 2] = 7.0  # already at max

    fa, pour = _make_accum_and_pp(accum, pp)
    result = snap_pour_point(fa, pour, search_radius=3)

    assert result.data[2, 2] == 7.0
    # All other cells NaN
    for r in range(3):
        for c in range(3):
            if (r, c) != (2, 2):
                assert np.isnan(result.data[r, c])


def test_multiple_pour_points():
    """Several pour points each snap independently."""
    accum = np.array([
        [50.0, 1.0, 1.0, 1.0, 80.0],
        [1.0,  1.0, 1.0, 1.0, 1.0],
        [1.0,  1.0, 1.0, 1.0, 1.0],
    ])
    pp = np.full((3, 5), np.nan)
    pp[0, 1] = 10.0  # near (0,0) which has accum=50
    pp[0, 3] = 20.0  # near (0,4) which has accum=80

    fa, pour = _make_accum_and_pp(accum, pp)
    result = snap_pour_point(fa, pour, search_radius=2)

    assert result.data[0, 0] == 10.0
    assert result.data[0, 4] == 20.0


def test_radius_limits_search():
    """Pour point does NOT snap to a higher cell outside the radius."""
    accum = np.array([
        [1.0, 1.0, 1.0, 1.0, 1000.0],
    ])
    pp = np.full((1, 5), np.nan)
    pp[0, 0] = 1.0

    fa, pour = _make_accum_and_pp(accum, pp)
    result = snap_pour_point(fa, pour, search_radius=2)

    # Can't reach (0,4) with radius=2 from (0,0)
    assert np.isnan(result.data[0, 4])
    # Should stay at (0,0) since all reachable cells have accum=1
    assert result.data[0, 0] == 1.0


def test_circular_radius():
    """Cells outside the circular (Euclidean) radius are excluded."""
    # 5x5 grid. Pour point at center (2,2), radius=2.
    # Corner cells at (0,0), (0,4), (4,0), (4,4) are at distance sqrt(8) > 2.
    accum = np.ones((5, 5), dtype=np.float64)
    accum[0, 0] = 999.0  # outside radius (distance = 2*sqrt(2) ~ 2.83)
    accum[2, 0] = 50.0   # inside radius (distance = 2)

    pp = np.full((5, 5), np.nan)
    pp[2, 2] = 1.0

    fa, pour = _make_accum_and_pp(accum, pp)
    result = snap_pour_point(fa, pour, search_radius=2)

    # Should snap to (2,0) not (0,0)
    assert result.data[2, 0] == 1.0
    assert np.isnan(result.data[0, 0])


def test_nan_accum_skipped():
    """NaN accumulation cells are never snap targets."""
    accum = np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, 5.0,    np.nan],
        [np.nan, np.nan, 10.0],
    ])
    pp = np.full((3, 3), np.nan)
    pp[1, 1] = 1.0

    fa, pour = _make_accum_and_pp(accum, pp)
    result = snap_pour_point(fa, pour, search_radius=2)

    # Should snap to (2,2) - it's in range and has the highest non-NaN accum
    assert result.data[2, 2] == 1.0


def test_all_nan_accum_in_radius():
    """Pour point stays in place if all neighbors have NaN accumulation."""
    accum = np.full((3, 3), np.nan)
    accum[1, 1] = 5.0  # only the pour point cell itself is valid

    pp = np.full((3, 3), np.nan)
    pp[1, 1] = 1.0

    fa, pour = _make_accum_and_pp(accum, pp)
    result = snap_pour_point(fa, pour, search_radius=1)

    # Should stay at (1,1) since it's the only non-NaN cell
    assert result.data[1, 1] == 1.0


def test_conflict_last_wins():
    """Two pour points snapping to the same cell: raster-scan-order last wins."""
    accum = np.array([
        [1.0, 100.0, 1.0],
    ])
    pp = np.full((1, 3), np.nan)
    pp[0, 0] = 10.0  # will snap to (0,1)
    pp[0, 2] = 20.0  # will also snap to (0,1)

    fa, pour = _make_accum_and_pp(accum, pp)
    result = snap_pour_point(fa, pour, search_radius=2)

    # (0,2) is scanned after (0,0), so label 20 wins
    assert result.data[0, 1] == 20.0


def test_label_preserved():
    """The pour point's original label value carries through."""
    accum = np.array([
        [1.0, 50.0],
        [1.0, 1.0],
    ])
    pp = np.full((2, 2), np.nan)
    pp[0, 0] = 42.5

    fa, pour = _make_accum_and_pp(accum, pp)
    result = snap_pour_point(fa, pour, search_radius=2)

    assert result.data[0, 1] == 42.5


def test_output_dtype():
    """Result is float64."""
    accum = np.array([[1.0, 2.0], [3.0, 4.0]])
    pp = np.full((2, 2), np.nan)
    pp[0, 0] = 1.0

    fa, pour = _make_accum_and_pp(accum, pp)
    result = snap_pour_point(fa, pour, search_radius=1)

    assert result.data.dtype == np.float64


def test_dataset_support():
    """@supports_dataset works."""
    accum = np.array([
        [1.0, 50.0],
        [1.0, 1.0],
    ], dtype=np.float64)
    pp_data = np.full((2, 2), np.nan)
    pp_data[0, 0] = 1.0

    da1 = xr.DataArray(accum, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(0.5, 0, 2)
    da1['x'] = np.linspace(0, 0.5, 2)
    ds = xr.Dataset({'fa1': da1, 'fa2': da1.copy()})

    pp = xr.DataArray(pp_data, dims=['y', 'x'])
    pp['y'] = da1['y']
    pp['x'] = da1['x']

    result = snap_pour_point(ds, pp, search_radius=2)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'fa1', 'fa2'}
    for var in result.data_vars:
        assert result[var].data[0, 1] == 1.0


# -------------------------------------------------------------------
# Cross-backend tests: dask
# -------------------------------------------------------------------

@dask_array_available
@pytest.mark.parametrize("chunks", [
    (1, 1), (2, 2), (3, 3), (1, 3), (3, 1),
])
def test_numpy_equals_dask(chunks):
    """Dask matches NumPy for snap pour point."""
    accum = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 100.0],
    ])
    pp = np.full((3, 3), np.nan)
    pp[0, 0] = 1.0
    pp[1, 0] = 2.0

    fa_np, pp_np = _make_accum_and_pp(accum, pp, backend='numpy')
    fa_dk, pp_dk = _make_accum_and_pp(accum, pp, backend='dask', chunks=chunks)

    np_result = snap_pour_point(fa_np, pp_np, search_radius=3)
    dk_result = snap_pour_point(fa_dk, pp_dk, search_radius=3)

    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


# -------------------------------------------------------------------
# Cross-backend tests: CuPy
# -------------------------------------------------------------------

@cuda_and_cupy_available
def test_numpy_equals_cupy():
    """CuPy matches NumPy for snap pour point."""
    accum = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 100.0],
    ])
    pp = np.full((3, 3), np.nan)
    pp[0, 0] = 1.0
    pp[1, 0] = 2.0

    fa_np, pp_np = _make_accum_and_pp(accum, pp, backend='numpy')
    fa_cp, pp_cp = _make_accum_and_pp(accum, pp, backend='cupy')

    np_result = snap_pour_point(fa_np, pp_np, search_radius=3)
    cp_result = snap_pour_point(fa_cp, pp_cp, search_radius=3)

    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


# -------------------------------------------------------------------
# Cross-backend tests: Dask+CuPy
# -------------------------------------------------------------------

@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy():
    """Dask+CuPy matches NumPy for snap pour point."""
    accum = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 100.0],
    ])
    pp = np.full((3, 3), np.nan)
    pp[0, 0] = 1.0
    pp[1, 0] = 2.0

    fa_np, pp_np = _make_accum_and_pp(accum, pp, backend='numpy')
    fa_dcp, pp_dcp = _make_accum_and_pp(accum, pp, backend='dask+cupy',
                                         chunks=(2, 2))

    np_result = snap_pour_point(fa_np, pp_np, search_radius=3)
    dcp_result = snap_pour_point(fa_dcp, pp_dcp, search_radius=3)

    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)
