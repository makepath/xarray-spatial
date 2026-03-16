import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro.watershed_mfd import watershed_mfd
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


def _make_mfd_raster(fractions, backend='numpy', chunks=(3, 3)):
    _, H, W = fractions.shape
    neighbor_names = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']
    y_coords = np.arange(H, dtype=np.float64)
    x_coords = np.arange(W, dtype=np.float64)
    da_obj = xr.DataArray(
        fractions.astype(np.float64),
        dims=('neighbor', 'y', 'x'),
        coords={'neighbor': neighbor_names, 'y': y_coords, 'x': x_coords},
        name='mfd_fdir',
    )
    if backend == 'dask':
        import dask.array as da
        da_obj = da_obj.chunk({'neighbor': 8, 'y': chunks[0], 'x': chunks[1]})
    elif backend == 'cupy':
        import cupy as cp
        da_obj = da_obj.copy(data=cp.asarray(da_obj.data))
    elif backend == 'dask+cupy':
        import cupy as cp
        import dask.array as da
        da_obj = da_obj.chunk({'neighbor': 8, 'y': chunks[0], 'x': chunks[1]})
        da_obj = da_obj.copy(data=da_obj.data.map_blocks(
            cp.asarray, dtype=da_obj.dtype, meta=cp.array((), dtype=da_obj.dtype)))
    return da_obj


def _make_all_east(H, W):
    fracs = np.zeros((8, H, W), dtype=np.float64)
    fracs[0, :, :-1] = 1.0  # E
    return fracs


def test_single_pour_point():
    """All cells flow east, pour point at right column."""
    fracs = _make_all_east(3, 5)
    pp = np.full((3, 5), np.nan, dtype=np.float64)
    pp[1, 4] = 42.0

    fd_da = _make_mfd_raster(fracs)
    pp_da = create_test_raster(pp)
    result = watershed_mfd(fd_da, pp_da)

    # Row 1: all reach pour point
    for c in range(5):
        assert result.data[1, c] == 42.0
    # Other rows: flow east but never reach the pour point (different row)
    # Actually: all cells in row 0 and 2 flow east to col 4 which is a pit (frac=0)
    # They never reach pp at (1,4) since they go to (r, 4) not (1, 4)
    assert np.isnan(result.data[0, 0])


def test_two_pour_points():
    fracs = np.zeros((8, 3, 6), dtype=np.float64)
    fracs[0, :, :2] = 1.0  # E in first 2 cols
    fracs[0, :, 3:5] = 1.0  # E in cols 3-4
    # cols 2 and 5 are pits

    pp = np.full((3, 6), np.nan, dtype=np.float64)
    pp[1, 2] = 10.0
    pp[1, 5] = 20.0

    fd_da = _make_mfd_raster(fracs)
    pp_da = create_test_raster(pp)
    result = watershed_mfd(fd_da, pp_da)

    for c in range(3):
        assert result.data[1, c] == 10.0
    for c in range(3, 6):
        assert result.data[1, c] == 20.0


def test_unreachable_cells():
    fracs = np.zeros((8, 1, 4), dtype=np.float64)
    fracs[0, 0, 0] = 1.0  # E at (0,0)
    # (0,1) pit, (0,2) flow E, (0,3) pit
    fracs[0, 0, 2] = 1.0

    pp = np.full((1, 4), np.nan, dtype=np.float64)
    pp[0, 1] = 5.0

    fd_da = _make_mfd_raster(fracs)
    pp_da = create_test_raster(pp)
    result = watershed_mfd(fd_da, pp_da)

    assert result.data[0, 0] == 5.0
    assert result.data[0, 1] == 5.0
    assert np.isnan(result.data[0, 2])
    assert np.isnan(result.data[0, 3])


def test_nan_handling():
    fracs = np.zeros((8, 2, 2), dtype=np.float64)
    fracs[0, 0, 0] = 1.0  # (0,0) flows E
    fracs[:, 1, 0] = np.nan  # (1,0) nodata
    # (0,1) pit, (1,1) pit

    pp = np.full((2, 2), np.nan, dtype=np.float64)
    pp[0, 1] = 7.0

    fd_da = _make_mfd_raster(fracs)
    pp_da = create_test_raster(pp)
    result = watershed_mfd(fd_da, pp_da)

    assert result.data[0, 0] == 7.0
    assert result.data[0, 1] == 7.0
    assert np.isnan(result.data[1, 0])


def test_linear_chain_east():
    N = 6
    fracs = np.zeros((8, 1, N), dtype=np.float64)
    fracs[0, 0, :-1] = 1.0  # E for all but last

    pp = np.full((1, N), np.nan, dtype=np.float64)
    pp[0, -1] = 99.0

    fd_da = _make_mfd_raster(fracs)
    pp_da = create_test_raster(pp)
    result = watershed_mfd(fd_da, pp_da)

    expected = np.full((1, N), 99.0, dtype=np.float64)
    np.testing.assert_allclose(result.data, expected)


def test_output_dtype():
    fracs = _make_all_east(2, 3)
    pp = np.full((2, 3), np.nan, dtype=np.float64)
    pp[0, 0] = 1.0

    fd_da = _make_mfd_raster(fracs)
    pp_da = create_test_raster(pp)
    result = watershed_mfd(fd_da, pp_da)
    assert result.data.dtype == np.float64


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (2, 2), (3, 3), (1, 1), (2, 6),
])
def test_numpy_equals_dask(chunks):
    fracs = _make_all_east(4, 6)
    pp = np.full((4, 6), np.nan, dtype=np.float64)
    pp[0, 5] = 1.0
    pp[3, 5] = 2.0

    fd_np = _make_mfd_raster(fracs, backend='numpy')
    pp_np = create_test_raster(pp, backend='numpy')
    fd_dk = _make_mfd_raster(fracs, backend='dask', chunks=chunks)
    pp_dk = create_test_raster(pp, backend='dask', chunks=chunks)

    np_result = watershed_mfd(fd_np, pp_np)
    dk_result = watershed_mfd(fd_dk, pp_dk)

    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@cuda_and_cupy_available
def test_numpy_equals_cupy():
    fracs = _make_all_east(3, 4)
    pp = np.full((3, 4), np.nan, dtype=np.float64)
    pp[1, 3] = 1.0

    fd_np = _make_mfd_raster(fracs, backend='numpy')
    pp_np = create_test_raster(pp, backend='numpy')
    fd_cp = _make_mfd_raster(fracs, backend='cupy')
    pp_cp = create_test_raster(pp, backend='cupy')

    np_result = watershed_mfd(fd_np, pp_np)
    cp_result = watershed_mfd(fd_cp, pp_cp)

    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy():
    fracs = _make_all_east(3, 4)
    pp = np.full((3, 4), np.nan, dtype=np.float64)
    pp[1, 3] = 1.0

    fd_np = _make_mfd_raster(fracs, backend='numpy')
    pp_np = create_test_raster(pp, backend='numpy')
    fd_dcp = _make_mfd_raster(fracs, backend='dask+cupy', chunks=(2, 2))
    pp_dcp = create_test_raster(pp, backend='dask+cupy', chunks=(2, 2))

    np_result = watershed_mfd(fd_np, pp_np)
    dcp_result = watershed_mfd(fd_dcp, pp_dcp)

    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)
