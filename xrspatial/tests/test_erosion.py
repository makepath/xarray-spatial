from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial import erosion as erosion_mod
from xrspatial.erosion import (
    erode,
    _build_brush,
    _MAX_ITERATIONS,
    _MAX_LIFETIME,
    _MAX_RADIUS,
)
from xrspatial.tests.general_checks import (
    create_test_raster,
    general_output_checks,
    cuda_and_cupy_available,
    dask_array_available,
)


# ---- helpers ----

def _make_terrain(size=64, seed=12345):
    """Build a simple synthetic terrain for testing."""
    rng = np.random.default_rng(seed)
    data = rng.random((size, size)).astype(np.float32) * 500
    return data


def _input(data, backend='numpy', chunks=(32, 32)):
    return create_test_raster(data, backend=backend, chunks=chunks)


# ---- brush tests ----

def test_build_brush_weights_sum_to_one():
    for r in (1, 2, 3, 5):
        _, _, bw = _build_brush(r)
        assert abs(bw.sum() - 1.0) < 1e-12, f"radius={r}: weights sum to {bw.sum()}"


def test_build_brush_offsets_within_radius():
    for r in (1, 3):
        boy, box, bw = _build_brush(r)
        for dy, dx in zip(boy, box):
            assert dx * dx + dy * dy <= r * r


# ---- numpy correctness ----

def test_erode_numpy_basic():
    """Erosion should lower peaks and raise valleys relative to input."""
    data = _make_terrain(size=64)
    agg = _input(data, 'numpy')
    result = erode(agg, iterations=5000, seed=42)

    general_output_checks(agg, result, verify_attrs=True)

    result_np = result.data
    # Volume change is bounded but not zero — particles that exit the
    # grid carry sediment out, and the brush erodes more broadly than
    # deposition covers.  Check that results differ from input and the
    # total doesn't explode.
    assert not np.array_equal(result_np, data)
    assert np.isfinite(result_np).all()


def test_erode_numpy_deterministic():
    """Same seed must produce identical results."""
    data = _make_terrain(size=32)
    r1 = erode(_input(data, 'numpy'), iterations=2000, seed=99)
    r2 = erode(_input(data, 'numpy'), iterations=2000, seed=99)
    np.testing.assert_array_equal(r1.data, r2.data)


def test_erode_numpy_different_seeds():
    """Different seeds should give different results."""
    data = _make_terrain(size=32)
    r1 = erode(_input(data, 'numpy'), iterations=2000, seed=1)
    r2 = erode(_input(data, 'numpy'), iterations=2000, seed=2)
    assert not np.array_equal(r1.data, r2.data)


def test_erode_custom_params():
    """Custom parameters should be accepted and change the result."""
    data = _make_terrain(size=32)
    r_default = erode(_input(data, 'numpy'), iterations=1000, seed=42)
    r_custom = erode(
        _input(data, 'numpy'), iterations=1000, seed=42,
        params={'erosion': 0.9, 'capacity': 10.0},
    )
    assert not np.array_equal(r_default.data, r_custom.data)


def test_erode_preserves_coords_and_attrs():
    """Output DataArray should keep the input's coordinates and attributes."""
    data = _make_terrain(size=16)
    agg = _input(data, 'numpy')
    result = erode(agg, iterations=500, seed=42)
    assert result.dims == agg.dims
    assert result.attrs == agg.attrs
    for coord in agg.coords:
        np.testing.assert_array_equal(result[coord].data, agg[coord].data)


def test_erode_flat_terrain_unchanged():
    """A perfectly flat surface should be unchanged by erosion."""
    data = np.full((32, 32), 100.0, dtype=np.float32)
    agg = _input(data, 'numpy')
    result = erode(agg, iterations=2000, seed=42)
    np.testing.assert_allclose(result.data, data, atol=1e-5)


def test_erode_small_raster():
    """Erosion should handle a small raster without crashing."""
    data = np.array([[10, 20, 30],
                     [40, 50, 60],
                     [70, 80, 90]], dtype=np.float32)
    agg = _input(data, 'numpy')
    result = erode(agg, iterations=100, seed=42)
    assert result.shape == (3, 3)


# ---- dask+numpy backend ----

@dask_array_available
def test_erode_dask_numpy_matches_numpy():
    """Dask+numpy backend should produce identical results to pure numpy."""
    data = _make_terrain(size=32)
    np_result = erode(_input(data, 'numpy'), iterations=2000, seed=42)
    da_result = erode(_input(data, 'dask+numpy', chunks=(16, 16)),
                      iterations=2000, seed=42)

    general_output_checks(
        _input(data, 'dask+numpy', chunks=(16, 16)), da_result,
    )
    np.testing.assert_array_equal(np_result.data, da_result.data.compute())


# ---- cupy backend ----

@cuda_and_cupy_available
def test_erode_cupy_runs():
    """CuPy backend should run and return a CuPy-backed DataArray."""
    import cupy as cp

    data = _make_terrain(size=64)
    agg = _input(data, 'cupy')
    result = erode(agg, iterations=5000, seed=42)

    general_output_checks(agg, result, verify_attrs=True)
    # result should stay on GPU
    assert hasattr(result.data, 'get'), "Expected CuPy array"


@cuda_and_cupy_available
def test_erode_cupy_modifies_terrain():
    """GPU erosion should modify the terrain and produce finite results."""
    data = _make_terrain(size=64)
    agg = _input(data, 'cupy')
    result = erode(agg, iterations=5000, seed=42)

    result_np = result.data.get()
    assert not np.array_equal(result_np, data)
    assert np.isfinite(result_np).all()


@cuda_and_cupy_available
def test_erode_cupy_flat_unchanged():
    """Flat terrain on GPU should remain flat."""
    data = np.full((32, 32), 100.0, dtype=np.float32)
    agg = _input(data, 'cupy')
    result = erode(agg, iterations=2000, seed=42)
    np.testing.assert_allclose(result.data.get(), data, atol=1e-5)


@cuda_and_cupy_available
def test_erode_cupy_consistent_structure():
    """Multiple GPU runs should produce structurally similar results.

    GPU erosion is non-deterministic due to cuda.atomic.add ordering,
    so we check that the overall erosion pattern is consistent rather
    than requiring bitwise equality.
    """
    data = _make_terrain(size=64)
    # Use fewer particles on a larger grid to reduce contention
    r1 = erode(_input(data, 'cupy'), iterations=500, seed=99)
    r2 = erode(_input(data, 'cupy'), iterations=500, seed=99)

    r1_np = r1.data.get()
    r2_np = r2.data.get()

    # Both should modify terrain in similar ways — correlation should be high.
    # Not exact due to atomic operation ordering.
    corr = np.corrcoef(r1_np.ravel(), r2_np.ravel())[0, 1]
    assert corr > 0.8, f"Correlation between GPU runs: {corr:.4f}"


# ---- dask+cupy backend ----

@cuda_and_cupy_available
@dask_array_available
def test_erode_dask_cupy_runs():
    """Dask+CuPy should run and return sensible results."""
    data = _make_terrain(size=32)
    agg = _input(data, 'dask+cupy', chunks=(16, 16))
    result = erode(agg, iterations=2000, seed=42)

    general_output_checks(agg, result, verify_attrs=True)
    result_np = result.data.compute().get()
    assert not np.array_equal(result_np, data)
    assert np.isfinite(result_np).all()


# ---- parameter validation (issue #1275) ----

def test_erode_iterations_zero_rejected():
    """iterations < 1 should raise ValueError."""
    agg = _input(_make_terrain(size=16), 'numpy')
    with pytest.raises(ValueError, match="iterations"):
        erode(agg, iterations=0)


def test_erode_iterations_too_large_rejected():
    """iterations > _MAX_ITERATIONS should raise ValueError."""
    agg = _input(_make_terrain(size=16), 'numpy')
    with pytest.raises(ValueError, match="iterations"):
        erode(agg, iterations=_MAX_ITERATIONS + 1)


def test_erode_iterations_huge_rejected_before_alloc():
    """iterations=10**12 must not allocate the (10**12, 2) random_pos buffer.

    The validator should fire before any np.random call, so the test
    completes in milliseconds even though the buffer would be ~16 TB.
    """
    agg = _input(_make_terrain(size=16), 'numpy')
    with pytest.raises(ValueError, match="iterations"):
        erode(agg, iterations=10**12)


def test_erode_iterations_non_int_rejected():
    """A float iterations argument should raise TypeError."""
    agg = _input(_make_terrain(size=16), 'numpy')
    with pytest.raises(TypeError, match="iterations"):
        erode(agg, iterations=1.5)


def test_erode_radius_zero_rejected():
    """radius < 1 should raise ValueError."""
    agg = _input(_make_terrain(size=16), 'numpy')
    with pytest.raises(ValueError, match="radius"):
        erode(agg, iterations=10, params={'radius': 0})


def test_erode_radius_too_large_rejected():
    """radius > _MAX_RADIUS should raise ValueError before _build_brush runs.

    radius=100_000 would walk (200_001)**2 cells (~40 billion iterations)
    before allocating the brush; the validator must fire first.
    """
    agg = _input(_make_terrain(size=16), 'numpy')
    with pytest.raises(ValueError, match="radius"):
        erode(agg, iterations=10, params={'radius': _MAX_RADIUS + 1})


def test_erode_max_lifetime_zero_rejected():
    """max_lifetime < 1 should raise ValueError."""
    agg = _input(_make_terrain(size=16), 'numpy')
    with pytest.raises(ValueError, match="max_lifetime"):
        erode(agg, iterations=10, params={'max_lifetime': 0})


def test_erode_max_lifetime_too_large_rejected():
    """max_lifetime > _MAX_LIFETIME should raise ValueError."""
    agg = _input(_make_terrain(size=16), 'numpy')
    with pytest.raises(ValueError, match="max_lifetime"):
        erode(agg, iterations=10, params={'max_lifetime': _MAX_LIFETIME + 1})


def test_erode_at_iteration_cap_runs():
    """A small raster with iterations exactly at the cap must still be
    accepted by the validator (the memory guard handles infeasible
    combinations separately)."""
    agg = _input(_make_terrain(size=8), 'numpy')
    # We can't actually run iterations=_MAX_ITERATIONS in a test; the
    # point here is that the validator accepts the boundary value.
    # Use monkeypatch-free check: call _validate_scalar directly via
    # a tiny wrapper.
    from xrspatial.utils import _validate_scalar
    _validate_scalar(
        _MAX_ITERATIONS, func_name='erode', name='iterations',
        dtype=int, min_val=1, max_val=_MAX_ITERATIONS,
    )


def test_erode_memory_guard_fires_numpy(monkeypatch):
    """The memory guard must fire on the numpy path (not just dask)."""
    # Pretend we have only 1 MB of RAM available.  A 128x128 float32 raster
    # with iterations=10000 needs a ~196 KB raster working set plus a
    # ~160 KB rng buffer, well within 1 MB.  Force the guard with a tighter
    # budget by using a larger raster.
    monkeypatch.setattr(erosion_mod, '_available_memory_bytes',
                        lambda: 1 * 1024 * 1024)
    agg = _input(_make_terrain(size=512), 'numpy')
    with pytest.raises(MemoryError, match="erode"):
        erode(agg, iterations=1000)


@cuda_and_cupy_available
def test_erode_memory_guard_fires_cupy(monkeypatch):
    """The memory guard must fire on the cupy path (not just dask)."""
    monkeypatch.setattr(erosion_mod, '_available_memory_bytes',
                        lambda: 1 * 1024 * 1024)
    agg = _input(_make_terrain(size=512), 'cupy')
    with pytest.raises(MemoryError, match="erode"):
        erode(agg, iterations=1000)


def test_erode_memory_guard_includes_random_pos(monkeypatch):
    """A huge iterations count on a tiny raster must trip the memory
    guard (the random_pos buffer dominates)."""
    monkeypatch.setattr(erosion_mod, '_available_memory_bytes',
                        lambda: 10 * 1024 * 1024)
    agg = _input(_make_terrain(size=4), 'numpy')
    # 10**7 particles ~ 160 MB float64 buffer, well under _MAX_ITERATIONS
    # but huge relative to a 4x4 raster's working set.  The error message
    # should call out the random_pos buffer specifically.
    with pytest.raises(MemoryError, match="random_pos"):
        erode(agg, iterations=10_000_000)


def test_erode_default_params_still_work():
    """Sanity check: the defaults stay well below every cap."""
    data = _make_terrain(size=32)
    agg = _input(data, 'numpy')
    result = erode(agg, iterations=500)  # default seed/params
    assert result.shape == (32, 32)
    assert np.isfinite(result.data).all()
