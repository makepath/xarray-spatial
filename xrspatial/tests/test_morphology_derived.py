"""Tests for derived morphological ops: gradient, white top-hat, black top-hat."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.morphology import (
    morph_black_tophat,
    morph_closing,
    morph_dilate,
    morph_erode,
    morph_gradient,
    morph_opening,
    morph_white_tophat,
)
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
    general_output_checks,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

_DATA = np.array([
    [1., 5., 3., 2., 7.],
    [4., 8., 6., 1., 3.],
    [2., 9., 7., 5., 4.],
    [3., 1., 4., 8., 6.],
    [6., 2., 3., 7., 9.],
], dtype=np.float64)

_KERNEL_3x3 = np.ones((3, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# morph_gradient correctness
# ---------------------------------------------------------------------------

def test_gradient_equals_dilate_minus_erode():
    """Gradient must equal dilate - erode."""
    agg = create_test_raster(_DATA)
    grad = morph_gradient(agg, kernel=_KERNEL_3x3, boundary='nearest')
    dilated = morph_dilate(agg, kernel=_KERNEL_3x3, boundary='nearest')
    eroded = morph_erode(agg, kernel=_KERNEL_3x3, boundary='nearest')
    expected = dilated.data - eroded.data
    np.testing.assert_allclose(grad.data, expected, equal_nan=True)


def test_gradient_nonnegative():
    """Gradient is always >= 0 for non-NaN cells."""
    agg = create_test_raster(_DATA)
    grad = morph_gradient(agg, kernel=_KERNEL_3x3, boundary='nearest')
    assert np.all(grad.data[~np.isnan(grad.data)] >= 0)


def test_gradient_uniform_is_zero():
    """Uniform raster produces zero gradient."""
    data = np.full((7, 7), 5.0, dtype=np.float64)
    agg = create_test_raster(data)
    grad = morph_gradient(agg, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(grad.data, 0.0)


def test_gradient_output_metadata():
    agg = create_test_raster(_DATA)
    grad = morph_gradient(agg, kernel=_KERNEL_3x3)
    general_output_checks(agg, grad, verify_attrs=True)
    assert grad.name == 'gradient'


# ---------------------------------------------------------------------------
# morph_white_tophat correctness
# ---------------------------------------------------------------------------

def test_white_tophat_equals_original_minus_opening():
    agg = create_test_raster(_DATA)
    wth = morph_white_tophat(agg, kernel=_KERNEL_3x3, boundary='nearest')
    opened = morph_opening(agg, kernel=_KERNEL_3x3, boundary='nearest')
    expected = agg.data - opened.data
    np.testing.assert_allclose(wth.data, expected, equal_nan=True)


def test_white_tophat_isolates_bright_spike():
    """A single bright spike should appear in the white top-hat."""
    data = np.zeros((7, 7), dtype=np.float64)
    data[3, 3] = 100.0
    agg = create_test_raster(data)
    wth = morph_white_tophat(agg, kernel=_KERNEL_3x3, boundary='nearest')
    # The spike is removed by opening, so original - opening = spike
    assert wth.data[3, 3] == 100.0
    # Background stays zero
    assert wth.data[0, 0] == 0.0


def test_white_tophat_nonnegative():
    """White top-hat is >= 0 (opening <= original)."""
    agg = create_test_raster(_DATA)
    wth = morph_white_tophat(agg, kernel=_KERNEL_3x3, boundary='nearest')
    assert np.all(wth.data[~np.isnan(wth.data)] >= -1e-10)


def test_white_tophat_output_metadata():
    agg = create_test_raster(_DATA)
    wth = morph_white_tophat(agg, kernel=_KERNEL_3x3)
    general_output_checks(agg, wth, verify_attrs=True)
    assert wth.name == 'white_tophat'


# ---------------------------------------------------------------------------
# morph_black_tophat correctness
# ---------------------------------------------------------------------------

def test_black_tophat_equals_closing_minus_original():
    agg = create_test_raster(_DATA)
    bth = morph_black_tophat(agg, kernel=_KERNEL_3x3, boundary='nearest')
    closed = morph_closing(agg, kernel=_KERNEL_3x3, boundary='nearest')
    expected = closed.data - agg.data
    np.testing.assert_allclose(bth.data, expected, equal_nan=True)


def test_black_tophat_isolates_dark_pit():
    """A single dark pit should appear in the black top-hat."""
    data = np.full((7, 7), 100.0, dtype=np.float64)
    data[3, 3] = 0.0
    agg = create_test_raster(data)
    bth = morph_black_tophat(agg, kernel=_KERNEL_3x3, boundary='nearest')
    # The pit is filled by closing, so closing - original = pit depth
    assert bth.data[3, 3] == 100.0
    # Background stays zero
    assert bth.data[0, 0] == 0.0


def test_black_tophat_nonnegative():
    """Black top-hat is >= 0 (closing >= original)."""
    agg = create_test_raster(_DATA)
    bth = morph_black_tophat(agg, kernel=_KERNEL_3x3, boundary='nearest')
    assert np.all(bth.data[~np.isnan(bth.data)] >= -1e-10)


def test_black_tophat_output_metadata():
    agg = create_test_raster(_DATA)
    bth = morph_black_tophat(agg, kernel=_KERNEL_3x3)
    general_output_checks(agg, bth, verify_attrs=True)
    assert bth.name == 'black_tophat'


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_gradient_nan_propagation():
    data = _DATA.copy()
    data[2, 2] = np.nan
    agg = create_test_raster(data)
    grad = morph_gradient(agg, kernel=_KERNEL_3x3)
    assert np.isnan(grad.data[2, 2])


def test_white_tophat_nan_propagation():
    data = _DATA.copy()
    data[2, 2] = np.nan
    agg = create_test_raster(data)
    wth = morph_white_tophat(agg, kernel=_KERNEL_3x3)
    assert np.isnan(wth.data[2, 2])


def test_black_tophat_nan_propagation():
    data = _DATA.copy()
    data[2, 2] = np.nan
    agg = create_test_raster(data)
    bth = morph_black_tophat(agg, kernel=_KERNEL_3x3)
    assert np.isnan(bth.data[2, 2])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_default_kernel():
    """Functions work with the default kernel argument."""
    agg = create_test_raster(_DATA)
    for func in [morph_gradient, morph_white_tophat, morph_black_tophat]:
        result = func(agg)
        general_output_checks(agg, result, verify_attrs=True)


def test_single_cell_raster():
    data = np.array([[42.0]], dtype=np.float64)
    agg = create_test_raster(data)
    # With boundary='nearest', single cell -> all ops see same value -> 0
    for func in [morph_gradient, morph_white_tophat, morph_black_tophat]:
        result = func(agg, boundary='nearest')
        assert result.data[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Dask backend
# ---------------------------------------------------------------------------

@dask_array_available
def test_gradient_dask_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_agg = create_test_raster(_DATA, backend='dask')
    np_result = morph_gradient(numpy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    dk_result = morph_gradient(dask_agg, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(
        dk_result.data.compute(), np_result.data, equal_nan=True,
    )


@dask_array_available
def test_white_tophat_dask_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_agg = create_test_raster(_DATA, backend='dask')
    np_result = morph_white_tophat(numpy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    dk_result = morph_white_tophat(dask_agg, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(
        dk_result.data.compute(), np_result.data, equal_nan=True,
    )


@dask_array_available
def test_black_tophat_dask_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_agg = create_test_raster(_DATA, backend='dask')
    np_result = morph_black_tophat(numpy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    dk_result = morph_black_tophat(dask_agg, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(
        dk_result.data.compute(), np_result.data, equal_nan=True,
    )


# ---------------------------------------------------------------------------
# CuPy backend
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
def test_gradient_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    cupy_agg = create_test_raster(_DATA, backend='cupy')
    np_result = morph_gradient(numpy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    cp_result = morph_gradient(cupy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(
        cp_result.data.get(), np_result.data, equal_nan=True,
    )


@cuda_and_cupy_available
def test_white_tophat_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    cupy_agg = create_test_raster(_DATA, backend='cupy')
    np_result = morph_white_tophat(numpy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    cp_result = morph_white_tophat(cupy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(
        cp_result.data.get(), np_result.data, equal_nan=True,
    )


@cuda_and_cupy_available
def test_black_tophat_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    cupy_agg = create_test_raster(_DATA, backend='cupy')
    np_result = morph_black_tophat(numpy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    cp_result = morph_black_tophat(cupy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(
        cp_result.data.get(), np_result.data, equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Dask + CuPy backend
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
@dask_array_available
def test_gradient_dask_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_cupy_agg = create_test_raster(_DATA, backend='dask+cupy')
    np_result = morph_gradient(numpy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    dc_result = morph_gradient(dask_cupy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(
        dc_result.data.compute().get(), np_result.data, equal_nan=True,
    )


@cuda_and_cupy_available
@dask_array_available
def test_white_tophat_dask_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_cupy_agg = create_test_raster(_DATA, backend='dask+cupy')
    np_result = morph_white_tophat(numpy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    dc_result = morph_white_tophat(dask_cupy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(
        dc_result.data.compute().get(), np_result.data, equal_nan=True,
    )


@cuda_and_cupy_available
@dask_array_available
def test_black_tophat_dask_cupy_equals_numpy():
    numpy_agg = create_test_raster(_DATA, backend='numpy')
    dask_cupy_agg = create_test_raster(_DATA, backend='dask+cupy')
    np_result = morph_black_tophat(numpy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    dc_result = morph_black_tophat(dask_cupy_agg, kernel=_KERNEL_3x3, boundary='nearest')
    np.testing.assert_allclose(
        dc_result.data.compute().get(), np_result.data, equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Dataset support
# ---------------------------------------------------------------------------

def test_gradient_dataset():
    data = np.random.default_rng(1025).random((5, 5)).astype(np.float64)
    ds = xr.Dataset({
        'a': xr.DataArray(data, dims=['y', 'x']),
        'b': xr.DataArray(data * 2, dims=['y', 'x']),
    })
    for func in [morph_gradient, morph_white_tophat, morph_black_tophat]:
        result = func(ds, boundary='nearest')
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {'a', 'b'}
