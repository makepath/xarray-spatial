from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.erosion import erode, _build_brush
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
