import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro import sink
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def test_single_pit():
    """3x3 grid, center code 0, all neighbors flow to center."""
    flow_dir = np.array([
        [2.0, 4.0, 8.0],
        [1.0, 0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = sink(agg)
    # Only center is a sink (code 0)
    assert not np.isnan(result.data[1, 1])
    for r in range(3):
        for c in range(3):
            if (r, c) != (1, 1):
                assert np.isnan(result.data[r, c]), f"({r},{c}) should be NaN"


def test_multiple_isolated_pits():
    """Two separated pits get different labels."""
    flow_dir = np.array([
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = sink(agg)
    assert not np.isnan(result.data[0, 0])
    assert not np.isnan(result.data[0, 2])
    assert np.isnan(result.data[0, 1])
    assert result.data[0, 0] != result.data[0, 2]


def test_connected_pits():
    """Two adjacent code-0 cells share one label."""
    flow_dir = np.array([
        [1.0, 0.0, 0.0, 16.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = sink(agg)
    assert not np.isnan(result.data[0, 1])
    assert not np.isnan(result.data[0, 2])
    assert result.data[0, 1] == result.data[0, 2]
    assert np.isnan(result.data[0, 0])
    assert np.isnan(result.data[0, 3])


def test_no_sinks():
    """All valid flow codes -> all NaN output."""
    flow_dir = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = sink(agg)
    assert np.all(np.isnan(result.data))


def test_nan_handling():
    """NaN cells are not sinks."""
    flow_dir = np.array([
        [np.nan, 0.0],
        [0.0, np.nan],
    ], dtype=np.float64)
    agg = create_test_raster(flow_dir)
    result = sink(agg)
    assert np.isnan(result.data[0, 0])
    assert np.isnan(result.data[1, 1])
    assert not np.isnan(result.data[0, 1])
    assert not np.isnan(result.data[1, 0])
    # (0,1) and (1,0) are 8-connected diagonally -> same label
    assert result.data[0, 1] == result.data[1, 0]


def test_dataset_support():
    """@supports_dataset works."""
    flow_dir = np.array([
        [2.0, 4.0, 8.0],
        [1.0, 0.0, 16.0],
        [128.0, 64.0, 32.0],
    ], dtype=np.float64)
    da1 = xr.DataArray(flow_dir, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    da1['y'] = np.linspace(1, 0, 3)
    da1['x'] = np.linspace(0, 1, 3)
    ds = xr.Dataset({'fd1': da1, 'fd2': da1.copy()})
    result = sink(ds)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'fd1', 'fd2'}


# -------------------------------------------------------------------
# Dask cross-backend tests
# -------------------------------------------------------------------

@dask_array_available
@pytest.mark.parametrize("chunks", [
    (1, 1), (2, 2), (3, 3), (1, 3), (3, 1),
])
def test_dask_isolated_sinks(chunks):
    """Isolated single-cell sinks: dask exactly matches numpy."""
    flow_dir = np.array([
        [0.0, 1.0, 0.0],
        [64.0, 1.0, 1.0],
        [64.0, 64.0, 0.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(flow_dir, backend='numpy')
    dk_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    np_result = sink(np_agg)
    dk_result = sink(dk_agg)
    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (1, 1), (2, 2), (3, 3),
])
def test_dask_nan_positions(chunks):
    """Dask identifies the same cells as sinks (NaN pattern matches)."""
    flow_dir = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 16.0],
        [128.0, 0.0, 0.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(flow_dir, backend='numpy')
    dk_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    np_result = sink(np_agg)
    dk_result = sink(dk_agg)
    np.testing.assert_array_equal(
        np.isnan(np_result.data), np.isnan(dk_result.data.compute()))


# -------------------------------------------------------------------
# GPU cross-backend tests
# -------------------------------------------------------------------

@cuda_and_cupy_available
def test_numpy_equals_cupy():
    """CuPy matches NumPy for connected sinks."""
    flow_dir = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 16.0],
        [128.0, 0.0, 0.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(flow_dir, backend='numpy')
    cp_agg = create_test_raster(flow_dir, backend='cupy')
    np_result = sink(np_agg)
    cp_result = sink(cp_agg)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy():
    """Dask+CuPy matches NumPy for isolated sinks."""
    flow_dir = np.array([
        [0.0, 1.0, 0.0],
        [64.0, 1.0, 1.0],
        [64.0, 64.0, 0.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(flow_dir, backend='numpy')
    dcp_agg = create_test_raster(flow_dir, backend='dask+cupy',
                                  chunks=(2, 2))
    np_result = sink(np_agg)
    dcp_result = sink(dcp_agg)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)
