"""Tests for xrspatial.bilateral."""
try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial.bilateral import bilateral, _bilateral_numpy, _kernel_radius
from xrspatial.tests.general_checks import (
    assert_boundary_mode_correctness,
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
    general_output_checks,
)


# ---- fixtures ----

@pytest.fixture
def step_edge():
    """5x10 raster with a sharp vertical edge: left=0, right=100."""
    data = np.zeros((5, 10), dtype=np.float64)
    data[:, 5:] = 100.0
    return data


@pytest.fixture
def random_data_969():
    rng = np.random.default_rng(969)
    return rng.standard_normal((30, 30)).astype(np.float64)


# ---- unit tests ----

def test_kernel_radius():
    assert _kernel_radius(1.0) == 2
    assert _kernel_radius(0.5) == 1
    assert _kernel_radius(2.0) == 4
    assert _kernel_radius(1.5) == 3


def test_bilateral_preserves_flat_surface():
    """Flat raster should be unchanged after bilateral filtering."""
    data = np.full((8, 8), 42.0)
    agg = xr.DataArray(data)
    result = bilateral(agg, sigma_spatial=1.0, sigma_range=10.0)
    np.testing.assert_allclose(result.data, 42.0)


def test_bilateral_preserves_edge(step_edge):
    """With a small sigma_range, the sharp edge should remain sharp."""
    agg = xr.DataArray(step_edge)
    result = bilateral(agg, sigma_spatial=1.0, sigma_range=1.0)

    # Interior left columns (far from edge) should stay near 0
    np.testing.assert_allclose(result.data[:, 0:3], 0.0, atol=1e-10)

    # Interior right columns (far from edge) should stay near 100
    np.testing.assert_allclose(result.data[:, 7:10], 100.0, atol=1e-10)


def test_bilateral_smooths_with_large_sigma_range(step_edge):
    """With a large sigma_range the filter should blur across the edge."""
    agg = xr.DataArray(step_edge)
    result = bilateral(agg, sigma_spatial=1.0, sigma_range=1000.0)

    # Column 4 (just left of edge) should now be pulled toward 100
    col4_mean = result.data[2, 4]
    assert col4_mean > 10.0, f"Expected blurring but got {col4_mean}"

    # Column 5 (just right of edge) should be pulled toward 0
    col5_mean = result.data[2, 5]
    assert col5_mean < 90.0, f"Expected blurring but got {col5_mean}"


def test_bilateral_nan_handling():
    """NaN center pixels should stay NaN; NaN neighbors should be skipped."""
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
    ])
    agg = xr.DataArray(data)
    result = bilateral(agg, sigma_spatial=1.0, sigma_range=100.0)

    # Center pixel stays NaN
    assert np.isnan(result.data[1, 1])

    # Non-NaN pixels should produce finite values
    assert np.all(np.isfinite(result.data[~np.isnan(data)]))


def test_bilateral_all_nan():
    """All-NaN raster should remain all-NaN."""
    data = np.full((4, 4), np.nan)
    agg = xr.DataArray(data)
    result = bilateral(agg, sigma_spatial=1.0, sigma_range=10.0)
    assert np.all(np.isnan(result.data))


def test_bilateral_single_cell():
    """Single-cell raster should return same value."""
    data = np.array([[7.0]])
    agg = xr.DataArray(data)
    result = bilateral(agg, sigma_spatial=1.0, sigma_range=10.0)
    np.testing.assert_allclose(result.data, 7.0)


def test_bilateral_output_checks(random_data_969):
    """Output should preserve shape, dims, coords, attrs."""
    agg = create_test_raster(random_data_969)
    result = bilateral(agg)
    general_output_checks(agg, result)


def test_bilateral_validation_errors():
    """Check that invalid inputs raise appropriate errors."""
    data = np.ones((5, 5))
    agg = xr.DataArray(data)

    with pytest.raises(TypeError):
        bilateral("not_a_dataarray")

    with pytest.raises(ValueError):
        bilateral(agg, sigma_spatial=-1.0)

    with pytest.raises(ValueError):
        bilateral(agg, sigma_range=0.0)

    with pytest.raises(ValueError):
        bilateral(agg, boundary='invalid')


def test_bilateral_known_values():
    """Verify output against hand-computed bilateral filter on a 3x3 raster."""
    data = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 10.0, 50.0],
        [10.0, 10.0, 10.0],
    ])
    agg = xr.DataArray(data)
    result = bilateral(agg, sigma_spatial=1.0, sigma_range=5.0)

    # Center pixel (1,1): neighbors are mostly 10, one is 50.
    # With sigma_range=5, the 50-value neighbor has very low weight
    # (diff=40, exp(-40^2/(2*25)) ~ 0), so result should be near 10.
    assert abs(result.data[1, 1] - 10.0) < 1.0

    # The 50-value pixel (1,2): all its non-NaN neighbors are 10 except itself.
    # The 10-value neighbors have diff=40 so low weight.
    # Self (distance=0) has full weight. Result should stay near 50.
    assert abs(result.data[1, 2] - 50.0) < 1.0


def test_bilateral_symmetry():
    """Symmetric input should produce symmetric output."""
    data = np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 100., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ])
    agg = xr.DataArray(data)
    result = bilateral(agg, sigma_spatial=1.0, sigma_range=50.0)
    r = result.data

    # Should be symmetric around center
    np.testing.assert_allclose(r, r[::-1, :], atol=1e-12)
    np.testing.assert_allclose(r, r[:, ::-1], atol=1e-12)


# ---- cross-backend tests ----

def test_bilateral_cpu(random_data_969):
    numpy_agg = create_test_raster(random_data_969)
    result = bilateral(numpy_agg)
    general_output_checks(numpy_agg, result)


@dask_array_available
def test_bilateral_dask_cpu(random_data_969):
    numpy_agg = create_test_raster(random_data_969)
    numpy_result = bilateral(numpy_agg)

    dask_agg = create_test_raster(random_data_969, backend='dask+numpy',
                                  chunks=(15, 15))
    dask_result = bilateral(dask_agg)
    general_output_checks(dask_agg, dask_result)

    np.testing.assert_allclose(
        numpy_result.data, dask_result.data.compute(),
        equal_nan=True, rtol=1e-6,
    )


@cuda_and_cupy_available
def test_bilateral_gpu_equals_cpu(random_data_969):
    import cupy

    numpy_agg = create_test_raster(random_data_969)
    numpy_result = bilateral(numpy_agg)

    cupy_agg = create_test_raster(random_data_969, backend='cupy')
    cupy_result = bilateral(cupy_agg)
    general_output_checks(cupy_agg, cupy_result)

    np.testing.assert_allclose(
        numpy_result.data, cupy_result.data.get(),
        equal_nan=True, rtol=1e-6,
    )


@dask_array_available
@cuda_and_cupy_available
def test_bilateral_dask_gpu(random_data_969):
    import cupy

    numpy_agg = create_test_raster(random_data_969)
    numpy_result = bilateral(numpy_agg)

    dask_cupy_agg = create_test_raster(random_data_969,
                                       backend='dask+cupy',
                                       chunks=(15, 15))
    dask_cupy_result = bilateral(dask_cupy_agg)
    general_output_checks(dask_cupy_agg, dask_cupy_result)

    np.testing.assert_allclose(
        numpy_result.data, dask_cupy_result.data.compute().get(),
        equal_nan=True, rtol=1e-4,
    )


# ---- boundary mode tests ----

@dask_array_available
def test_bilateral_boundary_modes(random_data_969):
    numpy_agg = create_test_raster(random_data_969)
    dask_agg = create_test_raster(random_data_969, backend='dask+numpy',
                                  chunks=(15, 15))
    assert_boundary_mode_correctness(
        numpy_agg, dask_agg, bilateral,
        depth=2, rtol=1e-5, nan_edges=False,
    )


# ---- 3D multi-band test ----

def test_bilateral_3d():
    band0 = np.random.default_rng(100).standard_normal((10, 10))
    band1 = np.random.default_rng(200).standard_normal((10, 10))
    data = np.stack([band0, band1])
    agg = xr.DataArray(data, dims=['band', 'y', 'x'])

    result = bilateral(agg, sigma_spatial=1.0, sigma_range=10.0)
    assert result.shape == agg.shape

    # Each band should match independent 2D processing
    for i in range(2):
        band_agg = xr.DataArray(data[i])
        band_result = bilateral(band_agg, sigma_spatial=1.0, sigma_range=10.0)
        np.testing.assert_allclose(
            result.data[i], band_result.data, equal_nan=True,
        )


# ---- accessor test ----

def test_bilateral_accessor():
    import xrspatial  # noqa - registers accessor
    data = np.random.default_rng(42).standard_normal((10, 10))
    agg = xr.DataArray(data, dims=['y', 'x'])
    result = agg.xrs.bilateral(sigma_spatial=1.0, sigma_range=10.0)
    assert result.shape == agg.shape
    assert isinstance(result, xr.DataArray)
