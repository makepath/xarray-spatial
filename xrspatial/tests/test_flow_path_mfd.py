import numpy as np
import pytest
import xarray as xr

from xrspatial.flow_path_mfd import flow_path_mfd
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
    """MFD fractions: all flow east, pits at right column."""
    fracs = np.zeros((8, H, W), dtype=np.float64)
    fracs[0, :, :-1] = 1.0  # E direction for non-pit cells
    # Right column: all zero = pit
    return fracs


def _make_all_south(H, W):
    """MFD fractions: all flow south, pits at bottom row."""
    fracs = np.zeros((8, H, W), dtype=np.float64)
    fracs[2, :-1, :] = 1.0  # S direction for non-pit cells
    return fracs


def test_single_path_east():
    fracs = _make_all_east(3, 4)
    sp = np.full((3, 4), np.nan)
    sp[0, 0] = 5.0

    fd_da = _make_mfd_raster(fracs)
    sp_da = create_test_raster(sp.astype(np.float64))
    result = flow_path_mfd(fd_da, sp_da)

    for c in range(4):
        assert result.data[0, c] == 5.0
    assert np.isnan(result.data[1, 0])


def test_single_path_south():
    fracs = _make_all_south(4, 3)
    sp = np.full((4, 3), np.nan)
    sp[0, 1] = 7.0

    fd_da = _make_mfd_raster(fracs)
    sp_da = create_test_raster(sp.astype(np.float64))
    result = flow_path_mfd(fd_da, sp_da)

    for r in range(4):
        assert result.data[r, 1] == 7.0
    assert np.isnan(result.data[0, 0])


def test_dominant_neighbor_selection():
    """Fractions split between E (0.7) and SE (0.3); dominant = E."""
    fracs = np.zeros((8, 2, 4), dtype=np.float64)
    fracs[0, :, :-1] = 0.7  # E
    fracs[1, :, :-1] = 0.3  # SE
    sp = np.full((2, 4), np.nan)
    sp[0, 0] = 1.0

    fd_da = _make_mfd_raster(fracs)
    sp_da = create_test_raster(sp.astype(np.float64))
    result = flow_path_mfd(fd_da, sp_da)

    # Should go east: (0,0)->(0,1)->(0,2)->(0,3)
    for c in range(4):
        assert result.data[0, c] == 1.0


def test_path_stops_at_pit():
    fracs = np.zeros((8, 1, 3), dtype=np.float64)
    fracs[0, 0, 0] = 1.0  # E at (0,0)
    # (0,1) all zero = pit
    sp = np.full((1, 3), np.nan)
    sp[0, 0] = 2.0

    fd_da = _make_mfd_raster(fracs)
    sp_da = create_test_raster(sp.astype(np.float64))
    result = flow_path_mfd(fd_da, sp_da)

    assert result.data[0, 0] == 2.0
    assert result.data[0, 1] == 2.0
    assert np.isnan(result.data[0, 2])


def test_path_stops_at_nan():
    fracs = np.zeros((8, 1, 3), dtype=np.float64)
    fracs[0, 0, 0] = 1.0  # E at (0,0)
    fracs[:, 0, 1] = np.nan  # nodata at (0,1)
    sp = np.full((1, 3), np.nan)
    sp[0, 0] = 1.0

    fd_da = _make_mfd_raster(fracs)
    sp_da = create_test_raster(sp.astype(np.float64))
    result = flow_path_mfd(fd_da, sp_da)

    assert result.data[0, 0] == 1.0
    assert result.data[0, 1] == 1.0  # reached before NaN check
    assert np.isnan(result.data[0, 2])


def test_path_stops_at_edge():
    fracs = np.zeros((8, 1, 3), dtype=np.float64)
    fracs[0, :, :] = 1.0  # all E
    sp = np.full((1, 3), np.nan)
    sp[0, 0] = 3.0

    fd_da = _make_mfd_raster(fracs)
    sp_da = create_test_raster(sp.astype(np.float64))
    result = flow_path_mfd(fd_da, sp_da)

    for c in range(3):
        assert result.data[0, c] == 3.0


def test_multiple_paths():
    fracs = _make_all_east(4, 4)
    sp = np.full((4, 4), np.nan)
    sp[0, 0] = 10.0
    sp[3, 0] = 20.0

    fd_da = _make_mfd_raster(fracs)
    sp_da = create_test_raster(sp.astype(np.float64))
    result = flow_path_mfd(fd_da, sp_da)

    for c in range(4):
        assert result.data[0, c] == 10.0
        assert result.data[3, c] == 20.0
    assert np.isnan(result.data[1, 0])


def test_paths_overlap_last_wins():
    # Both start points flow SE to (1,1)
    fracs = np.zeros((8, 3, 3), dtype=np.float64)
    fracs[1, :, :] = 1.0  # all SE
    fracs[:, -1, :] = 0.0  # bottom row pits
    fracs[:, :, -1] = 0.0  # right col pits

    sp = np.full((3, 3), np.nan)
    sp[0, 0] = 10.0  # -> (1,1) -> pit
    sp[0, 1] = 20.0  # -> (1,2) pit

    fd_da = _make_mfd_raster(fracs)
    sp_da = create_test_raster(sp.astype(np.float64))
    result = flow_path_mfd(fd_da, sp_da)

    assert result.data[0, 0] == 10.0
    assert result.data[0, 1] == 20.0
    assert result.data[1, 1] == 10.0


def test_output_dtype():
    fracs = _make_all_east(2, 3)
    sp = np.full((2, 3), np.nan)
    sp[0, 0] = 1.0

    fd_da = _make_mfd_raster(fracs)
    sp_da = create_test_raster(sp.astype(np.float64))
    result = flow_path_mfd(fd_da, sp_da)
    assert result.data.dtype == np.float64


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (1, 1), (2, 2), (3, 3), (1, 4), (4, 1),
])
def test_numpy_equals_dask(chunks):
    fracs = _make_all_east(4, 4)
    fracs[1, 2, :3] = 0.5  # add some SE flow in row 2
    fracs[0, 2, :3] = 0.5
    sp = np.full((4, 4), np.nan)
    sp[0, 0] = 1.0
    sp[3, 0] = 2.0

    fd_np = _make_mfd_raster(fracs, backend='numpy')
    sp_np = create_test_raster(sp.astype(np.float64), backend='numpy')
    fd_dk = _make_mfd_raster(fracs, backend='dask', chunks=chunks)
    sp_dk = create_test_raster(sp.astype(np.float64), backend='dask', chunks=chunks)

    np_result = flow_path_mfd(fd_np, sp_np)
    dk_result = flow_path_mfd(fd_dk, sp_dk)

    np.testing.assert_allclose(
        np_result.data, dk_result.data.compute(), equal_nan=True)


@cuda_and_cupy_available
def test_numpy_equals_cupy():
    fracs = _make_all_east(3, 4)
    sp = np.full((3, 4), np.nan)
    sp[0, 0] = 1.0
    sp[2, 0] = 2.0

    fd_np = _make_mfd_raster(fracs, backend='numpy')
    sp_np = create_test_raster(sp.astype(np.float64), backend='numpy')
    fd_cp = _make_mfd_raster(fracs, backend='cupy')
    sp_cp = create_test_raster(sp.astype(np.float64), backend='cupy')

    np_result = flow_path_mfd(fd_np, sp_np)
    cp_result = flow_path_mfd(fd_cp, sp_cp)

    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_numpy_equals_dask_cupy():
    fracs = _make_all_east(3, 4)
    sp = np.full((3, 4), np.nan)
    sp[0, 0] = 1.0
    sp[2, 0] = 2.0

    fd_np = _make_mfd_raster(fracs, backend='numpy')
    sp_np = create_test_raster(sp.astype(np.float64), backend='numpy')
    fd_dcp = _make_mfd_raster(fracs, backend='dask+cupy', chunks=(2, 2))
    sp_dcp = create_test_raster(sp.astype(np.float64), backend='dask+cupy', chunks=(2, 2))

    np_result = flow_path_mfd(fd_np, sp_np)
    dcp_result = flow_path_mfd(fd_dcp, sp_dcp)

    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)
