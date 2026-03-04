"""Tests for xrspatial.corridor.least_cost_corridor."""

try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial.corridor import least_cost_corridor
from xrspatial.cost_distance import cost_distance
from xrspatial.utils import has_cuda_and_cupy


def _make_raster(data, backend="numpy", chunks=(3, 3)):
    """Build a DataArray with y/x coords, optionally dask/cupy-backed."""
    h, w = data.shape
    raster = xr.DataArray(
        data.astype(np.float64),
        dims=["y", "x"],
        attrs={"res": (1.0, 1.0)},
    )
    raster["y"] = np.arange(h, dtype=np.float64)
    raster["x"] = np.arange(w, dtype=np.float64)
    if "dask" in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=chunks)
    if "cupy" in backend and has_cuda_and_cupy():
        import cupy

        if isinstance(raster.data, da.Array):
            raster.data = raster.data.map_blocks(cupy.asarray)
        else:
            raster.data = cupy.asarray(raster.data)
    return raster


def _compute(arr):
    """Extract numpy data from DataArray (works for numpy, dask, or cupy)."""
    if da is not None and isinstance(arr.data, da.Array):
        val = arr.data.compute()
        if hasattr(val, "get"):
            return val.get()
        return val
    if hasattr(arr.data, "get"):
        return arr.data.get()
    return arr.data


# -----------------------------------------------------------------------
# Basic corridor correctness
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_basic_corridor_symmetry(backend):
    """Corridor between two sources on uniform friction is symmetric."""
    n = 7
    friction_data = np.ones((n, n))

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0  # left edge

    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0  # right edge

    friction = _make_raster(friction_data, backend=backend, chunks=(7, 7))
    sa = _make_raster(src_a, backend=backend, chunks=(7, 7))
    sb = _make_raster(src_b, backend=backend, chunks=(7, 7))

    result = least_cost_corridor(friction, sa, sb)
    out = _compute(result)

    # Minimum corridor cost should be 0 (after normalization)
    assert np.nanmin(out) == pytest.approx(0.0, abs=1e-5)

    # Corridor should be symmetric about the vertical midline
    np.testing.assert_allclose(out[:, :3], out[:, -1:-4:-1], atol=1e-5)


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_corridor_minimum_on_optimal_path(backend):
    """Cells on the optimal path between sources have corridor value 0."""
    n = 5
    friction_data = np.ones((n, n))

    src_a = np.zeros((n, n))
    src_a[2, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[2, 4] = 1.0

    friction = _make_raster(friction_data, backend=backend, chunks=(5, 5))
    sa = _make_raster(src_a, backend=backend, chunks=(5, 5))
    sb = _make_raster(src_b, backend=backend, chunks=(5, 5))

    result = least_cost_corridor(friction, sa, sb)
    out = _compute(result)

    # The middle row (row 2) should be the optimal path on uniform friction.
    # All cells on row 2 should have the minimum corridor value (0).
    for col in range(n):
        assert out[2, col] == pytest.approx(0.0, abs=1e-5)


# -----------------------------------------------------------------------
# Threshold tests
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_absolute_threshold(backend):
    """Absolute threshold masks cells with normalized cost > threshold."""
    n = 7
    friction_data = np.ones((n, n))

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0

    friction = _make_raster(friction_data, backend=backend, chunks=(7, 7))
    sa = _make_raster(src_a, backend=backend, chunks=(7, 7))
    sb = _make_raster(src_b, backend=backend, chunks=(7, 7))

    result = least_cost_corridor(friction, sa, sb, threshold=0.5)
    out = _compute(result)

    # Cells with normalized cost > 0.5 should be NaN
    assert np.all(np.isnan(out) | (out <= 0.5 + 1e-5))

    # The optimal path (row 3) should not be masked
    for col in range(n):
        assert np.isfinite(out[3, col])


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_relative_threshold(backend):
    """Relative threshold uses fraction of minimum corridor cost."""
    n = 7
    friction_data = np.ones((n, n))

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0

    friction = _make_raster(friction_data, backend=backend, chunks=(7, 7))
    sa = _make_raster(src_a, backend=backend, chunks=(7, 7))
    sb = _make_raster(src_b, backend=backend, chunks=(7, 7))

    # No threshold -- get full corridor
    full = least_cost_corridor(friction, sa, sb)
    full_out = _compute(full)

    # Relative threshold of 50%
    result = least_cost_corridor(
        friction, sa, sb, threshold=0.5, relative=True
    )
    out = _compute(result)

    # Count finite cells -- threshold version should have fewer
    assert np.sum(np.isfinite(out)) < np.sum(np.isfinite(full_out))

    # Optimal path cells should survive
    for col in range(n):
        assert np.isfinite(out[3, col])


# -----------------------------------------------------------------------
# Precomputed cost-distance surfaces
# -----------------------------------------------------------------------


def test_precomputed_matches_regular():
    """Precomputed=True with manual cost_distance matches default path."""
    n = 7
    friction_data = np.ones((n, n))

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0

    friction = _make_raster(friction_data)
    sa = _make_raster(src_a)
    sb = _make_raster(src_b)

    # Regular path
    result_regular = least_cost_corridor(friction, sa, sb)

    # Precomputed path
    cd_a = cost_distance(sa, friction)
    cd_b = cost_distance(sb, friction)
    result_precomputed = least_cost_corridor(
        friction, cd_a, cd_b, precomputed=True
    )

    np.testing.assert_allclose(
        _compute(result_regular),
        _compute(result_precomputed),
        atol=1e-5,
    )


# -----------------------------------------------------------------------
# Multi-source pairwise
# -----------------------------------------------------------------------


def test_pairwise_corridor():
    """Pairwise mode with 3 sources returns Dataset with 3 corridors."""
    n = 7
    friction_data = np.ones((n, n))

    sources = []
    for r, c in [(0, 0), (0, 6), (6, 3)]:
        s = np.zeros((n, n))
        s[r, c] = 1.0
        sources.append(_make_raster(s))

    friction = _make_raster(friction_data)

    result = least_cost_corridor(
        friction, sources=sources, pairwise=True
    )

    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {
        "corridor_0_1",
        "corridor_0_2",
        "corridor_1_2",
    }

    # Each corridor should have minimum 0
    for name in result.data_vars:
        out = _compute(result[name])
        assert np.nanmin(out) == pytest.approx(0.0, abs=1e-5)


def test_pairwise_two_sources_returns_dataset():
    """Pairwise=True with exactly 2 sources still returns a Dataset."""
    n = 5
    friction_data = np.ones((n, n))

    s0 = np.zeros((n, n))
    s0[0, 0] = 1.0
    s1 = np.zeros((n, n))
    s1[4, 4] = 1.0

    friction = _make_raster(friction_data)
    result = least_cost_corridor(
        friction,
        sources=[_make_raster(s0), _make_raster(s1)],
        pairwise=True,
    )

    assert isinstance(result, xr.Dataset)
    assert "corridor_0_1" in result.data_vars


# -----------------------------------------------------------------------
# NaN / barrier handling
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_barrier_blocks_corridor(backend):
    """NaN barrier between sources makes certain cells unreachable."""
    n = 7
    friction_data = np.ones((n, n))
    # Wall of NaN except a gap at row 3
    friction_data[:3, 3] = np.nan
    friction_data[4:, 3] = np.nan

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0

    friction = _make_raster(friction_data, backend=backend, chunks=(7, 7))
    sa = _make_raster(src_a, backend=backend, chunks=(7, 7))
    sb = _make_raster(src_b, backend=backend, chunks=(7, 7))

    result = least_cost_corridor(friction, sa, sb)
    out = _compute(result)

    # The gap row should still be reachable
    assert np.isfinite(out[3, 3])


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_unreachable_sources(backend):
    """Full barrier between sources produces all-NaN corridor."""
    n = 5
    friction_data = np.ones((n, n))
    friction_data[:, 2] = np.nan  # impenetrable wall

    src_a = np.zeros((n, n))
    src_a[2, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[2, 4] = 1.0

    friction = _make_raster(friction_data, backend=backend, chunks=(5, 5))
    sa = _make_raster(src_a, backend=backend, chunks=(5, 5))
    sb = _make_raster(src_b, backend=backend, chunks=(5, 5))

    result = least_cost_corridor(friction, sa, sb)
    out = _compute(result)

    assert np.all(np.isnan(out))


# -----------------------------------------------------------------------
# Edge cases and validation
# -----------------------------------------------------------------------


def test_single_cell_raster():
    """1x1 raster where both sources are the same cell."""
    friction = _make_raster(np.ones((1, 1)))
    src = _make_raster(np.ones((1, 1)))

    result = least_cost_corridor(friction, src, src)
    out = _compute(result)

    assert out[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_missing_sources_raises():
    """Omitting both source_a/source_b and sources raises ValueError."""
    friction = _make_raster(np.ones((3, 3)))
    with pytest.raises(ValueError, match="source_a and source_b are required"):
        least_cost_corridor(friction)


def test_both_source_modes_raises():
    """Providing source_a/source_b AND sources raises ValueError."""
    friction = _make_raster(np.ones((3, 3)))
    src = _make_raster(np.ones((3, 3)))
    with pytest.raises(ValueError, match="not both"):
        least_cost_corridor(friction, src, src, sources=[src, src])


def test_negative_threshold_raises():
    """Negative threshold raises ValueError."""
    friction = _make_raster(np.ones((3, 3)))
    src = _make_raster(np.ones((3, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        least_cost_corridor(friction, src, src, threshold=-1.0)


def test_single_source_in_list_raises():
    """sources with fewer than 2 entries raises ValueError."""
    friction = _make_raster(np.ones((3, 3)))
    src = _make_raster(np.ones((3, 3)))
    with pytest.raises(ValueError, match="at least 2"):
        least_cost_corridor(friction, sources=[src])
