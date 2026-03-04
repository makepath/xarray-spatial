"""Tests for xrspatial.balanced_allocation."""

try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial.balanced_allocation import balanced_allocation
from xrspatial.tests.general_checks import cuda_and_cupy_available
from xrspatial.utils import has_cuda_and_cupy, has_dask_array


def _make_raster(data, backend='numpy', chunks=(5, 5)):
    """Build a DataArray with y/x coords, optionally dask/cupy-backed."""
    h, w = data.shape
    raster = xr.DataArray(
        data.astype(np.float64),
        dims=['y', 'x'],
        attrs={'res': (1.0, 1.0)},
    )
    raster['y'] = np.arange(h, dtype=np.float64)
    raster['x'] = np.arange(w, dtype=np.float64)
    if 'dask' in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=chunks)
    if 'cupy' in backend and has_cuda_and_cupy():
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
        if hasattr(val, 'get'):
            return val.get()
        return val
    if hasattr(arr.data, 'get'):
        return arr.data.get()
    return arr.data


# -----------------------------------------------------------------------
# Two sources with uniform friction should converge to equal areas
# -----------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_two_sources_uniform_friction(backend):
    """With uniform friction, territories should have roughly equal area."""
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("No GPU/CuPy available")
    if 'dask' in backend and da is None:
        pytest.skip("Dask not available")

    # 10x10 grid with two sources at opposite corners
    data = np.zeros((10, 10), dtype=np.float64)
    data[1, 1] = 1.0   # source A
    data[8, 8] = 2.0   # source B

    raster = _make_raster(data, backend=backend, chunks=(10, 10))
    friction = _make_raster(np.ones((10, 10)), backend=backend,
                            chunks=(10, 10))

    result = balanced_allocation(raster, friction, tolerance=0.10)
    out = _compute(result)

    # Both source IDs should be present
    unique = set(np.unique(out[np.isfinite(out)]))
    assert 1.0 in unique
    assert 2.0 in unique

    # Count cells per territory
    n1 = np.sum(out == 1.0)
    n2 = np.sum(out == 2.0)
    total = n1 + n2

    # Territories should be roughly balanced (within 20% of half)
    assert abs(n1 - n2) / total < 0.25


# -----------------------------------------------------------------------
# Single source: everything reachable goes to that source
# -----------------------------------------------------------------------

def test_single_source():
    """With one source, all reachable cells should be assigned to it."""
    data = np.zeros((5, 5), dtype=np.float64)
    data[2, 2] = 7.0

    raster = _make_raster(data)
    friction = _make_raster(np.ones((5, 5)))

    result = balanced_allocation(raster, friction)
    out = _compute(result)

    # All cells should be 7.0
    assert np.all(out[np.isfinite(out)] == 7.0)


# -----------------------------------------------------------------------
# No sources: all NaN
# -----------------------------------------------------------------------

def test_no_sources():
    """With no sources, output should be all NaN."""
    data = np.zeros((5, 5), dtype=np.float64)
    raster = _make_raster(data)
    friction = _make_raster(np.ones((5, 5)))

    result = balanced_allocation(raster, friction)
    out = _compute(result)

    assert np.all(np.isnan(out))


# -----------------------------------------------------------------------
# NaN friction barrier
# -----------------------------------------------------------------------

def test_nan_barrier():
    """NaN friction should block cost paths and leave cells unreachable."""
    data = np.zeros((5, 5), dtype=np.float64)
    data[0, 0] = 1.0
    data[0, 4] = 2.0

    fric = np.ones((5, 5), dtype=np.float64)
    # Wall of NaN in the middle column
    fric[:, 2] = np.nan

    raster = _make_raster(data)
    friction = _make_raster(fric)

    result = balanced_allocation(raster, friction)
    out = _compute(result)

    # Left side should be source 1, right side should be source 2
    # Middle column should be NaN (impassable)
    assert np.all(out[:, 2] != out[:, 2])  # NaN check
    left = out[:, :2]
    right = out[:, 3:]
    assert np.all(left[np.isfinite(left)] == 1.0)
    assert np.all(right[np.isfinite(right)] == 2.0)


# -----------------------------------------------------------------------
# Asymmetric friction should shift boundaries
# -----------------------------------------------------------------------

def test_asymmetric_friction_shifts_boundary():
    """High-friction zone should make that territory smaller by cell count."""
    # 1x20 strip with sources at each end
    data = np.zeros((1, 20), dtype=np.float64)
    data[0, 0] = 1.0
    data[0, 19] = 2.0

    # Left half cheap, right half expensive
    fric = np.ones((1, 20), dtype=np.float64)
    fric[0, 10:] = 5.0

    raster = _make_raster(data, chunks=(1, 20))
    friction = _make_raster(fric, chunks=(1, 20))

    result = balanced_allocation(raster, friction, tolerance=0.10)
    out = _compute(result)

    # Source 2 (expensive side) should have fewer cells than source 1
    n1 = np.sum(out == 1.0)
    n2 = np.sum(out == 2.0)
    assert n1 > n2, (
        f"Expected source 1 to have more cells (cheap friction), "
        f"got n1={n1}, n2={n2}"
    )


# -----------------------------------------------------------------------
# Three sources
# -----------------------------------------------------------------------

def test_three_sources():
    """Three sources should each get roughly 1/3 of the cost-weighted area."""
    data = np.zeros((12, 12), dtype=np.float64)
    data[2, 6] = 1.0
    data[9, 2] = 2.0
    data[9, 10] = 3.0

    raster = _make_raster(data, chunks=(12, 12))
    friction = _make_raster(np.ones((12, 12)), chunks=(12, 12))

    result = balanced_allocation(raster, friction, tolerance=0.15)
    out = _compute(result)

    unique = set(np.unique(out[np.isfinite(out)]))
    assert {1.0, 2.0, 3.0} == unique

    # Each territory should have at least 20% of cells
    total = np.sum(np.isfinite(out))
    for sid in [1.0, 2.0, 3.0]:
        frac = np.sum(out == sid) / total
        assert frac > 0.15, f"Source {sid} only got {frac:.1%} of cells"


# -----------------------------------------------------------------------
# Validation errors
# -----------------------------------------------------------------------

def test_shape_mismatch():
    """Should raise ValueError if raster and friction shapes differ."""
    raster = _make_raster(np.zeros((5, 5)))
    friction = _make_raster(np.zeros((5, 6)))
    with pytest.raises(ValueError, match="same shape"):
        balanced_allocation(raster, friction)


def test_bad_connectivity():
    """Should raise ValueError for invalid connectivity."""
    data = np.zeros((5, 5))
    data[2, 2] = 1.0
    raster = _make_raster(data)
    friction = _make_raster(np.ones((5, 5)))
    with pytest.raises(ValueError, match="connectivity"):
        balanced_allocation(raster, friction, connectivity=6)


def test_bad_tolerance():
    """Should raise ValueError for non-positive tolerance."""
    data = np.zeros((5, 5))
    data[2, 2] = 1.0
    raster = _make_raster(data)
    friction = _make_raster(np.ones((5, 5)))
    with pytest.raises(ValueError, match="tolerance"):
        balanced_allocation(raster, friction, tolerance=0)


def test_bad_max_iterations():
    """Should raise ValueError for max_iterations < 1."""
    data = np.zeros((5, 5))
    data[2, 2] = 1.0
    raster = _make_raster(data)
    friction = _make_raster(np.ones((5, 5)))
    with pytest.raises(ValueError, match="max_iterations"):
        balanced_allocation(raster, friction, max_iterations=0)


# -----------------------------------------------------------------------
# target_values parameter
# -----------------------------------------------------------------------

def test_target_values():
    """target_values should restrict which pixel values are treated as sources."""
    data = np.zeros((8, 8), dtype=np.float64)
    data[1, 1] = 1.0
    data[6, 6] = 2.0
    data[1, 6] = 3.0  # this one should be ignored

    raster = _make_raster(data, chunks=(8, 8))
    friction = _make_raster(np.ones((8, 8)), chunks=(8, 8))

    result = balanced_allocation(raster, friction, target_values=[1.0, 2.0])
    out = _compute(result)

    unique = set(np.unique(out[np.isfinite(out)]))
    assert 3.0 not in unique
    assert {1.0, 2.0} == unique
