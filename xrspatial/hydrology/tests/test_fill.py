import numpy as np
import pytest
import xarray as xr

from xrspatial.hydrology import fill
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def test_single_cell_depression():
    """3x3 DEM with low center filled to rim level."""
    dem = np.array([
        [10.0, 10.0, 10.0],
        [10.0,  5.0, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float64)
    agg = create_test_raster(dem)
    result = fill(agg)
    expected = np.full((3, 3), 10.0, dtype=np.float64)
    np.testing.assert_allclose(result.data, expected)


def test_no_depression():
    """Monotone slope: output == input."""
    dem = np.array([
        [10.0, 9.0, 8.0],
        [7.0, 6.0, 5.0],
        [4.0, 3.0, 2.0],
    ], dtype=np.float64)
    agg = create_test_raster(dem)
    result = fill(agg)
    np.testing.assert_allclose(result.data, dem)


def test_multi_cell_depression():
    """5x5 DEM with basin filled to pour point elevation."""
    dem = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0,  5.0,  3.0,  5.0, 10.0],
        [10.0,  3.0,  1.0,  3.0, 10.0],
        [10.0,  5.0,  3.0,  5.0, 10.0],
        [10.0, 10.0,  8.0, 10.0, 10.0],
    ], dtype=np.float64)
    agg = create_test_raster(dem)
    result = fill(agg)
    expected = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0,  8.0,  8.0,  8.0, 10.0],
        [10.0,  8.0,  8.0,  8.0, 10.0],
        [10.0,  8.0,  8.0,  8.0, 10.0],
        [10.0, 10.0,  8.0, 10.0, 10.0],
    ], dtype=np.float64)
    np.testing.assert_allclose(result.data, expected)


def test_z_limit_blocks_deep():
    """Depression deeper than z_limit reverts to DEM."""
    dem = np.array([
        [10.0, 10.0, 10.0],
        [10.0,  2.0, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float64)
    agg = create_test_raster(dem)
    result = fill(agg, z_limit=3.0)
    # fill depth = 10 - 2 = 8 > 3 -> revert to DEM
    assert result.data[1, 1] == 2.0
    assert result.data[0, 0] == 10.0


def test_z_limit_allows_shallow():
    """Shallow depression within z_limit stays filled."""
    dem = np.array([
        [10.0, 10.0, 10.0],
        [10.0,  8.0, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float64)
    agg = create_test_raster(dem)
    result = fill(agg, z_limit=5.0)
    # fill depth = 10 - 8 = 2 <= 5 -> keep
    assert result.data[1, 1] == 10.0


def test_nan_handling():
    """NaN cells stay NaN, don't propagate fill."""
    dem = np.array([
        [10.0, np.nan, 10.0],
        [10.0,  5.0,  10.0],
        [10.0, 10.0,  10.0],
    ], dtype=np.float64)
    agg = create_test_raster(dem)
    result = fill(agg)
    assert np.isnan(result.data[0, 1])
    # Center fills to 10 (NaN neighbor skipped, all other neighbors = 10)
    assert result.data[1, 1] == 10.0


def test_flat_surface():
    """No depressions: output == input."""
    dem = np.full((4, 5), 7.0, dtype=np.float64)
    agg = create_test_raster(dem)
    result = fill(agg)
    np.testing.assert_allclose(result.data, dem)


def test_dataset_support():
    """@supports_dataset works."""
    dem = np.array([
        [10.0, 10.0, 10.0],
        [10.0,  5.0, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float64)
    da1 = xr.DataArray(dem, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(1, 0, 3)
    da1['x'] = np.linspace(0, 1, 3)
    ds = xr.Dataset({'dem1': da1, 'dem2': da1.copy()})
    result = fill(ds)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'dem1', 'dem2'}


def test_output_dtype():
    """Output dtype is float64."""
    dem = np.full((3, 3), 5.0, dtype=np.float64)
    agg = create_test_raster(dem)
    result = fill(agg)
    assert result.data.dtype == np.float64


# -------------------------------------------------------------------
# Dask cross-backend tests
# -------------------------------------------------------------------

@dask_array_available
@pytest.mark.parametrize("chunks", [
    (2, 2), (3, 3), (5, 5), (1, 1), (2, 5),
])
def test_dask_equivalence(chunks):
    """Multiple chunk sizes all match numpy result."""
    dem = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0,  5.0,  3.0,  5.0, 10.0],
        [10.0,  3.0,  1.0,  3.0, 10.0],
        [10.0,  5.0,  3.0,  5.0, 10.0],
        [10.0, 10.0,  8.0, 10.0, 10.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(dem, backend='numpy')
    dk_agg = create_test_raster(dem, backend='dask', chunks=chunks)
    np_result = fill(np_agg)
    dk_result = fill(dk_agg)
    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@dask_array_available
def test_dask_cross_tile_depression():
    """Depression spanning multiple tiles matches numpy."""
    dem = np.array([
        [10.0, 10.0, 10.0, 10.0],
        [10.0,  3.0,  3.0, 10.0],
        [10.0,  3.0,  3.0, 10.0],
        [10.0, 10.0,  7.0, 10.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(dem, backend='numpy')
    dk_agg = create_test_raster(dem, backend='dask', chunks=(2, 2))
    np_result = fill(np_agg)
    dk_result = fill(dk_agg)
    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@dask_array_available
def test_dask_no_depression():
    """Monotone slope with dask: output == input."""
    dem = np.array([
        [10.0, 9.0, 8.0, 7.0],
        [6.0, 5.0, 4.0, 3.0],
        [4.0, 3.0, 2.0, 1.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(dem, backend='numpy')
    dk_agg = create_test_raster(dem, backend='dask', chunks=(2, 2))
    np_result = fill(np_agg)
    dk_result = fill(dk_agg)
    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@dask_array_available
def test_dask_z_limit():
    """z_limit with dask matches numpy."""
    dem = np.array([
        [10.0, 10.0, 10.0],
        [10.0,  2.0, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(dem, backend='numpy')
    dk_agg = create_test_raster(dem, backend='dask', chunks=(2, 2))
    np_result = fill(np_agg, z_limit=3.0)
    dk_result = fill(dk_agg, z_limit=3.0)
    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


# -------------------------------------------------------------------
# GPU cross-backend tests
# -------------------------------------------------------------------

@cuda_and_cupy_available
def test_numpy_equals_cupy():
    """CuPy matches NumPy."""
    dem = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0,  5.0,  3.0,  5.0, 10.0],
        [10.0,  3.0,  1.0,  3.0, 10.0],
        [10.0,  5.0,  3.0,  5.0, 10.0],
        [10.0, 10.0,  8.0, 10.0, 10.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(dem, backend='numpy')
    cp_agg = create_test_raster(dem, backend='cupy')
    np_result = fill(np_agg)
    cp_result = fill(cp_agg)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy():
    """Dask+CuPy matches NumPy."""
    dem = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0,  5.0,  3.0,  5.0, 10.0],
        [10.0,  3.0,  1.0,  3.0, 10.0],
        [10.0,  5.0,  3.0,  5.0, 10.0],
        [10.0, 10.0,  8.0, 10.0, 10.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(dem, backend='numpy')
    dcp_agg = create_test_raster(dem, backend='dask+cupy', chunks=(3, 3))
    np_result = fill(np_agg)
    dcp_result = fill(dcp_agg)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)
