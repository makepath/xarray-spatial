"""Tests for xrspatial.morphology (erode, dilate, opening, closing)."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import xarray as xr

import xrspatial.morphology as morphology_mod
from xrspatial.morphology import (
    _circle_kernel,
    morph_closing,
    morph_dilate,
    morph_erode,
    morph_opening,
)
from xrspatial.tests.general_checks import (
    assert_boundary_mode_correctness,
    assert_numpy_equals_cupy,
    assert_numpy_equals_dask_cupy,
    assert_numpy_equals_dask_numpy,
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
    general_output_checks,
)


# ---------------------------------------------------------------------------
# Fixtures / test data
# ---------------------------------------------------------------------------

_DATA = np.array([
    [1., 5., 3., 2., 7.],
    [4., 8., 6., 1., 3.],
    [2., 9., 7., 5., 4.],
    [3., 1., 4., 8., 6.],
    [6., 2., 3., 7., 9.],
], dtype=np.float64)

_DATA_NAN = _DATA.copy()
_DATA_NAN[0, 2] = np.nan
_DATA_NAN[3, 1] = np.nan

_KERNEL_3x3 = np.ones((3, 3), dtype=np.uint8)

_KERNEL_CROSS = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def test_circle_kernel_radius1():
    k = _circle_kernel(1)
    expected = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(k, expected)


def test_circle_kernel_radius2():
    k = _circle_kernel(2)
    assert k.shape == (5, 5)
    # corners should be zero
    assert k[0, 0] == 0
    assert k[0, 4] == 0
    # centre and cardinal neighbours should be one
    assert k[2, 2] == 1
    assert k[0, 2] == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_erode_rejects_non_dataarray():
    with pytest.raises(TypeError, match="must be an xarray.DataArray"):
        morph_erode(np.ones((5, 5)))


def test_erode_rejects_3d_input():
    agg = xr.DataArray(np.ones((3, 5, 5)), dims=['z', 'y', 'x'])
    with pytest.raises(ValueError, match="must be 2D"):
        morph_erode(agg)


def test_erode_rejects_even_kernel():
    agg = create_test_raster(_DATA)
    with pytest.raises(ValueError, match="must be odd"):
        morph_erode(agg, kernel=np.ones((4, 4), dtype=np.uint8))


def test_erode_rejects_bad_boundary():
    agg = create_test_raster(_DATA)
    with pytest.raises(ValueError, match="boundary must be one of"):
        morph_erode(agg, boundary='invalid')


# ---------------------------------------------------------------------------
# Correctness -- numpy backend
# ---------------------------------------------------------------------------

def test_erode_numpy_correctness():
    """Verify erosion matches hand-computed local minima."""
    agg = create_test_raster(_DATA)
    result = morph_erode(agg, kernel=_KERNEL_3x3)
    general_output_checks(agg, result, verify_attrs=True)

    # Interior cell [2, 2]: min of 3x3 neighbourhood centred at (2,2)
    # Neighbourhood: [[8,6,1],[9,7,5],[1,4,8]]
    assert result.data[2, 2] == 1.0

    # Interior cell [2, 1]: min of [[1,5,3],[4,8,6],[2,9,7]]
    assert result.data[2, 1] == 1.0


def test_dilate_numpy_correctness():
    """Verify dilation matches hand-computed local maxima."""
    agg = create_test_raster(_DATA)
    result = morph_dilate(agg, kernel=_KERNEL_3x3)
    general_output_checks(agg, result, verify_attrs=True)

    # Interior cell [2, 2]: max of 3x3 neighbourhood centred at (2,2)
    # Neighbourhood: [[8,6,1],[9,7,5],[1,4,8]]
    assert result.data[2, 2] == 9.0


def test_erode_cross_kernel():
    """Verify cross kernel only considers cardinal neighbours + centre."""
    agg = create_test_raster(_DATA)
    result = morph_erode(agg, kernel=_KERNEL_CROSS)

    # Interior cell [2, 2]: min of {data[1,2], data[2,1], data[2,2], data[2,3], data[3,2]}
    # = min(6, 9, 7, 5, 4) = 4
    assert result.data[2, 2] == 4.0


def test_opening_removes_bright_spike():
    """Opening should suppress isolated bright peaks."""
    data = np.zeros((7, 7), dtype=np.float64)
    data[3, 3] = 100.0  # single bright spike
    agg = create_test_raster(data)
    result = morph_opening(agg, kernel=_KERNEL_3x3)
    # After erosion the spike disappears; after dilation it stays gone
    assert result.data[3, 3] == 0.0


def test_closing_fills_dark_hole():
    """Closing should fill isolated dark pits."""
    data = np.full((7, 7), 100.0, dtype=np.float64)
    data[3, 3] = 0.0  # single dark pit
    agg = create_test_raster(data)
    result = morph_closing(agg, kernel=_KERNEL_3x3)
    assert result.data[3, 3] == 100.0


def test_opening_idempotent():
    """Applying opening twice should give the same result as once."""
    agg = create_test_raster(_DATA)
    once = morph_opening(agg, kernel=_KERNEL_3x3, boundary='nearest')
    twice = morph_opening(once, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(once.data, twice.data, equal_nan=True)


def test_closing_idempotent():
    """Applying closing twice should give the same result as once."""
    agg = create_test_raster(_DATA)
    once = morph_closing(agg, kernel=_KERNEL_3x3, boundary='nearest')
    twice = morph_closing(once, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(once.data, twice.data, equal_nan=True)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_erode_nan_propagation():
    """NaN cells in the source should remain NaN; NaN neighbours propagate."""
    agg = create_test_raster(_DATA_NAN)
    result = morph_erode(agg, kernel=_KERNEL_3x3)
    # Cell [0, 2] is NaN in source -> stays NaN
    assert np.isnan(result.data[0, 2])
    # Cell [0, 1] touches NaN at [0, 2] -> becomes NaN
    assert np.isnan(result.data[0, 1])


def test_dilate_nan_propagation():
    agg = create_test_raster(_DATA_NAN)
    result = morph_dilate(agg, kernel=_KERNEL_3x3)
    assert np.isnan(result.data[0, 2])
    assert np.isnan(result.data[0, 1])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_cell_raster():
    data = np.array([[42.0]], dtype=np.float64)
    agg = create_test_raster(data)
    # With boundary='nan', the NaN padding propagates into the result.
    assert np.isnan(morph_erode(agg).data[0, 0])
    # With boundary='nearest', padding replicates the single value.
    assert morph_erode(agg, boundary='nearest').data[0, 0] == 42.0
    assert morph_dilate(agg, boundary='nearest').data[0, 0] == 42.0


def test_uniform_raster():
    """All values identical -> all operations should be identity."""
    data = np.full((5, 5), 7.0, dtype=np.float64)
    agg = create_test_raster(data)
    for func in [morph_erode, morph_dilate, morph_opening, morph_closing]:
        result = func(agg, kernel=_KERNEL_3x3, boundary='nearest')
        np.testing.assert_allclose(result.data, 7.0)


def test_default_kernel():
    """Functions work with the default (None) kernel argument."""
    agg = create_test_raster(_DATA)
    result = morph_erode(agg)
    general_output_checks(agg, result, verify_attrs=True)


# ---------------------------------------------------------------------------
# Boundary modes -- numpy vs dask
# ---------------------------------------------------------------------------

@dask_array_available
def test_erode_boundary_modes():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_agg = create_test_raster(_DATA, backend='dask')
    assert_boundary_mode_correctness(
        numpy_agg, dask_agg, morph_erode, depth=1, nan_edges=True,
    )


@dask_array_available
def test_dilate_boundary_modes():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_agg = create_test_raster(_DATA, backend='dask')
    assert_boundary_mode_correctness(
        numpy_agg, dask_agg, morph_dilate, depth=1, nan_edges=True,
    )


# ---------------------------------------------------------------------------
# Dask backend
# ---------------------------------------------------------------------------

@dask_array_available
def test_erode_dask_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_agg = create_test_raster(_DATA, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, morph_erode, nan_edges=True)


@dask_array_available
def test_dilate_dask_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_agg = create_test_raster(_DATA, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, morph_dilate, nan_edges=True)


@dask_array_available
def test_opening_dask_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_agg = create_test_raster(_DATA, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, morph_opening, nan_edges=True)


@dask_array_available
def test_closing_dask_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_agg = create_test_raster(_DATA, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, morph_closing, nan_edges=True)


# ---------------------------------------------------------------------------
# CuPy backend
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
def test_erode_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    cupy_agg = create_test_raster(_DATA, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, morph_erode, nan_edges=True)


@cuda_and_cupy_available
def test_dilate_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    cupy_agg = create_test_raster(_DATA, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, morph_dilate, nan_edges=True)


@cuda_and_cupy_available
def test_opening_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    cupy_agg = create_test_raster(_DATA, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, morph_opening, nan_edges=True)


@cuda_and_cupy_available
def test_closing_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    cupy_agg = create_test_raster(_DATA, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, morph_closing, nan_edges=True)


# ---------------------------------------------------------------------------
# Dask + CuPy backend
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
@dask_array_available
def test_erode_dask_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_cupy_agg = create_test_raster(_DATA, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, morph_erode, nan_edges=True)


@cuda_and_cupy_available
@dask_array_available
def test_dilate_dask_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_cupy_agg = create_test_raster(_DATA, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, morph_dilate, nan_edges=True)


# ---------------------------------------------------------------------------
# 5x5 kernel
# ---------------------------------------------------------------------------

def test_erode_5x5_kernel():
    """Larger kernel should produce a wider erosion effect."""
    data = np.full((9, 9), 10.0, dtype=np.float64)
    data[4, 4] = 0.0  # single dark pit
    agg = create_test_raster(data)
    k3 = np.ones((3, 3), dtype=np.uint8)
    k5 = np.ones((5, 5), dtype=np.uint8)

    r3 = morph_erode(agg, kernel=k3, boundary='nearest')
    r5 = morph_erode(agg, kernel=k5, boundary='nearest')

    # 3x3 kernel: cell [3,3] should be 0 (touches the pit)
    assert r3.data[3, 3] == 0.0
    # but cell [2,2] should remain 10
    assert r3.data[2, 2] == 10.0

    # 5x5 kernel: cell [2,2] should now see the pit too
    assert r5.data[2, 2] == 0.0


# ---------------------------------------------------------------------------
# Dataset support
# ---------------------------------------------------------------------------

def test_erode_dataset():
    """morph_erode should work on xr.Dataset via @supports_dataset."""
    data = np.random.default_rng(42).random((5, 5)).astype(np.float64)
    ds = xr.Dataset({
        'a': xr.DataArray(data, dims=['y', 'x']),
        'b': xr.DataArray(data * 2, dims=['y', 'x']),
    })
    result = morph_erode(ds, boundary='nearest')
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'a', 'b'}


# ---------------------------------------------------------------------------
# Memory guard (issue #1256)
# ---------------------------------------------------------------------------

def test_oversized_kernel_raises_memory_error(monkeypatch):
    """Oversized kernel should raise MemoryError with an informative message.

    Patches ``_available_memory_bytes`` to a tiny value so a modest
    kernel trips the 50% budget check without actually allocating
    anything large.
    """
    monkeypatch.setattr(
        morphology_mod, "_available_memory_bytes", lambda: 1 * 1024 * 1024,
    )
    agg = create_test_raster(np.ones((100, 100), dtype=np.float64))
    # 101x101 kernel on a 100x100 raster -> padded (200, 200) float64
    # ~ 640 KB which exceeds 50% of the 1 MB budget.
    big_kernel = np.ones((101, 101), dtype=np.uint8)
    with pytest.raises(MemoryError, match="kernel shape"):
        morph_erode(agg, kernel=big_kernel)


def test_kernel_memory_guard_allows_normal_use(monkeypatch):
    """The guard must not block ordinary kernels under realistic memory."""
    monkeypatch.setattr(
        morphology_mod, "_available_memory_bytes", lambda: 2 * 1024 ** 3,
    )
    agg = create_test_raster(_DATA)
    # Default 3x3 kernel on a 5x5 raster fits comfortably.
    result = morph_erode(agg, kernel=_KERNEL_3x3)
    general_output_checks(agg, result, verify_attrs=True)


def test_kernel_memory_guard_runs_for_all_public_apis(monkeypatch):
    """Every public API entry point should trigger the guard via _dispatch."""
    monkeypatch.setattr(
        morphology_mod, "_available_memory_bytes", lambda: 1 * 1024 * 1024,
    )
    agg = create_test_raster(np.ones((100, 100), dtype=np.float64))
    big_kernel = np.ones((101, 101), dtype=np.uint8)
    for func in [
        morph_erode, morph_dilate, morph_opening, morph_closing,
        morphology_mod.morph_gradient,
        morphology_mod.morph_white_tophat,
        morphology_mod.morph_black_tophat,
    ]:
        with pytest.raises(MemoryError, match="kernel shape"):
            func(agg, kernel=big_kernel)
