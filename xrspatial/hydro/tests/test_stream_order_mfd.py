import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro.stream_order_mfd import stream_order_mfd
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
    """Wrap raw arrays in DataArrays and call stream_order_mfd."""
    frac_da = xr.DataArray(fracs, dims=['neighbor', 'y', 'x'])
    fa_da = create_test_raster(accum)
    return stream_order_mfd(frac_da, fa_da, threshold=threshold, **kwargs)


# ====================================================================
# Strahler tests
# ====================================================================

def test_y_confluence_strahler():
    """Two order-1 streams merge -> order 2 downstream."""
    # (0,0)->SE(1) to (1,1), (0,2)->SW(3) to (1,1)
    # (1,1)->S(2) to (2,1), (2,1) pit
    fracs = _make_fractions({
        (0, 0): [(1, 1.0)],   # SE
        (0, 1): [],
        (0, 2): [(3, 1.0)],   # SW
        (1, 0): [],
        (1, 1): [(2, 1.0)],   # S
        (1, 2): [],
        (2, 0): [],
        (2, 1): [],            # pit
        (2, 2): [],
    }, (3, 3))
    accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 4.0, 1.0],
    ], dtype=np.float64)
    result = _call(fracs, accum, threshold=1, method='strahler')
    assert result.data[0, 0] == 1.0
    assert result.data[0, 2] == 1.0
    assert result.data[1, 1] == 2.0
    assert result.data[2, 1] == 2.0


def test_unequal_confluence_strahler():
    """Order 1 meets order 2 -> stays order 2."""
    fracs = _make_fractions({
        (0, 0): [(1, 1.0)],   # SE
        (0, 1): [],
        (0, 2): [(3, 1.0)],   # SW
        (1, 0): [],
        (1, 1): [(2, 1.0)],   # S
        (1, 2): [],
        (2, 0): [(0, 1.0)],   # E
        (2, 1): [],            # pit
        (2, 2): [],
    }, (3, 3))
    accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 5.0, 1.0],
    ], dtype=np.float64)
    result = _call(fracs, accum, threshold=1, method='strahler')
    assert result.data[0, 0] == 1.0
    assert result.data[0, 2] == 1.0
    assert result.data[1, 1] == 2.0
    # (2,1): max=2 from (1,1), cnt_max=1, also gets 1 from (2,0)
    # max stays 2, cnt stays 1 -> order 2
    assert result.data[2, 1] == 2.0


# ====================================================================
# Shreve tests
# ====================================================================

def test_y_confluence_shreve():
    """Two headwaters merge -> Shreve magnitude = 2."""
    fracs = _make_fractions({
        (0, 0): [(1, 1.0)],   # SE
        (0, 1): [],
        (0, 2): [(3, 1.0)],   # SW
        (1, 0): [],
        (1, 1): [(2, 1.0)],   # S
        (1, 2): [],
        (2, 0): [],
        (2, 1): [],            # pit
        (2, 2): [],
    }, (3, 3))
    accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 4.0, 1.0],
    ], dtype=np.float64)
    result = _call(fracs, accum, threshold=1, method='shreve')
    assert result.data[0, 0] == 1.0
    assert result.data[0, 2] == 1.0
    assert result.data[1, 1] == 2.0
    assert result.data[2, 1] == 2.0


# ====================================================================
# MFD-specific: split flow
# ====================================================================

def test_split_flow_strahler():
    """Cell sends flow to two neighbors via MFD fractions.

    H(0,1) sends 50% SE and 50% S.  Both targets are stream cells
    that are pits.  H has in-degree 0 (headwater, order 1).
    Both targets have in-degree 1 (from H), so they get order 1.
    """
    fracs = _make_fractions({
        (0, 0): [],
        (0, 1): [(1, 0.5), (2, 0.5)],   # SE + S
        (0, 2): [],
        (1, 0): [],
        (1, 1): [],            # pit, receives from H via S
        (1, 2): [],            # pit, receives from H via SE
    }, (2, 3))
    accum = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 2.0, 2.0],
    ], dtype=np.float64)
    result = _call(fracs, accum, threshold=1, method='strahler')
    assert result.data[0, 1] == 1.0  # headwater
    assert result.data[1, 1] == 1.0  # single inflow
    assert result.data[1, 2] == 1.0  # single inflow


# ====================================================================
# Edge cases
# ====================================================================

def test_nan_handling():
    """NaN in fractions -> non-stream cell (NaN output)."""
    fracs = np.full((8, 2, 2), np.nan, dtype=np.float64)
    # Only (0,0) is valid, pit
    fracs[:, 0, 0] = 0.0
    accum = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    result = _call(fracs, accum, threshold=1, method='strahler')
    assert result.data[0, 0] == 1.0  # isolated headwater
    assert np.isnan(result.data[0, 1])
    assert np.isnan(result.data[1, 0])
    assert np.isnan(result.data[1, 1])


def test_threshold_excludes_cells():
    """Cells below threshold are NaN even if valid fractions exist."""
    fracs = _make_fractions({
        (0, 0): [(0, 1.0)],  # E
        (0, 1): [],           # pit
    }, (1, 2))
    accum = np.array([[1.0, 100.0]], dtype=np.float64)
    result = _call(fracs, accum, threshold=50, method='strahler')
    assert np.isnan(result.data[0, 0])   # below threshold
    assert result.data[0, 1] == 1.0       # above threshold, headwater


def test_single_cell():
    """Single-cell raster -> headwater order 1."""
    fracs = np.zeros((8, 1, 1), dtype=np.float64)
    accum = np.array([[5.0]], dtype=np.float64)
    result = _call(fracs, accum, threshold=1, method='strahler')
    assert result.data[0, 0] == 1.0


def test_invalid_method():
    """Invalid method raises ValueError."""
    fracs = np.zeros((8, 1, 1), dtype=np.float64)
    accum = np.array([[1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="method must be"):
        _call(fracs, accum, threshold=0, method='bogus')


@dask_array_available
def test_dask_matches_numpy():
    """Dask result matches numpy for a small grid."""
    import dask.array as da

    fracs = _make_fractions({
        (0, 0): [(1, 1.0)],
        (0, 1): [],
        (0, 2): [(3, 1.0)],
        (1, 0): [],
        (1, 1): [(2, 1.0)],
        (1, 2): [],
        (2, 0): [(0, 1.0)],
        (2, 1): [],
        (2, 2): [],
    }, (3, 3))
    accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 5.0, 1.0],
    ], dtype=np.float64)

    frac_da_np = xr.DataArray(fracs, dims=['neighbor', 'y', 'x'])
    fa_da_np = create_test_raster(accum)
    np_result = stream_order_mfd(frac_da_np, fa_da_np, threshold=1,
                                  method='strahler')

    frac_dask = xr.DataArray(
        da.from_array(fracs, chunks=(8, 2, 2)),
        dims=['neighbor', 'y', 'x'])
    fa_dask = xr.DataArray(
        da.from_array(accum, chunks=(2, 2)),
        dims=['y', 'x'])
    dask_result = stream_order_mfd(frac_dask, fa_dask, threshold=1,
                                    method='strahler')

    np.testing.assert_array_equal(
        np.nan_to_num(np_result.values, nan=-999),
        np.nan_to_num(dask_result.values, nan=-999))


# ====================================================================
# Memory guard tests
# ====================================================================

class TestMemoryGuard:
    """Memory guard on the eager numpy / cupy backends."""

    def test_numpy_huge_raster_raises(self):
        """Numpy backend raises MemoryError when projected RAM exceeds budget."""
        from unittest.mock import patch

        fracs = _make_fractions({(0, 0): []}, (4, 4))
        accum = np.ones((4, 4), dtype=np.float64)

        with patch(
            "xrspatial.hydro.stream_order_mfd._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                _call(fracs, accum, threshold=1)

    def test_numpy_normal_input_succeeds(self):
        """Normal-size raster passes the guard with real memory."""
        fracs = _make_fractions({
            (0, 0): [(0, 1.0)],
            (0, 1): [(0, 1.0)],
            (0, 2): [],
        }, (1, 3))
        accum = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        result = _call(fracs, accum, threshold=1)
        assert result.shape == (1, 3)

    @dask_array_available
    def test_dask_path_skips_guard(self):
        """Dask backend bypasses the guard -- per-tile allocations are bounded."""
        from unittest.mock import patch
        import dask.array as da

        fracs = _make_fractions({(0, 0): []}, (6, 6))
        # Replace NaN with zeros so dask path doesn't choke
        fracs = np.nan_to_num(fracs, nan=0.0)
        accum = np.ones((6, 6), dtype=np.float64)

        frac_dask = xr.DataArray(
            da.from_array(fracs, chunks=(8, 3, 3)),
            dims=['neighbor', 'y', 'x'])
        fa_dask = xr.DataArray(
            da.from_array(accum, chunks=(3, 3)),
            dims=['y', 'x'])

        with patch(
            "xrspatial.hydro.stream_order_mfd._available_memory_bytes",
            return_value=1,
        ):
            result = stream_order_mfd(frac_dask, fa_dask, threshold=1)
            assert result is not None

    def test_error_message_mentions_dimensions(self):
        """The error message should mention the grid dimensions and dask."""
        from unittest.mock import patch

        fracs = _make_fractions({(0, 0): []}, (7, 9))
        accum = np.ones((7, 9), dtype=np.float64)

        with patch(
            "xrspatial.hydro.stream_order_mfd._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match=r"7x9.*dask"):
                _call(fracs, accum, threshold=1)

    @cuda_and_cupy_available
    def test_cupy_huge_raster_raises(self):
        """CuPy backend raises MemoryError when projected GPU RAM exceeds budget."""
        from unittest.mock import patch
        import cupy as cp

        fracs = _make_fractions({(0, 0): []}, (4, 4))
        accum = np.ones((4, 4), dtype=np.float64)

        frac_da_cp = xr.DataArray(
            cp.asarray(fracs), dims=['neighbor', 'y', 'x'])
        fa_da_cp = xr.DataArray(cp.asarray(accum), dims=['y', 'x'])

        with patch(
            "xrspatial.hydro.stream_order_mfd._available_gpu_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="GPU working memory"):
                stream_order_mfd(frac_da_cp, fa_da_cp, threshold=1)
