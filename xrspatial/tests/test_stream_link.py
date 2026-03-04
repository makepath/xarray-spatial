import numpy as np
import pytest
import xarray as xr

from xrspatial import stream_link
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ====================================================================
# Helpers
# ====================================================================

def _make_stream_link(flow_dir, flow_accum, threshold=0, **kwargs):
    """Shortcut: wrap raw arrays in DataArrays and call stream_link."""
    fd_da = create_test_raster(flow_dir)
    fa_da = create_test_raster(flow_accum)
    return stream_link(fd_da, fa_da, threshold=threshold, **kwargs)


# ====================================================================
# Basic functionality tests
# ====================================================================

def test_linear_chain():
    """Single stream with no junctions -> all one link_id."""
    flow_dir = np.array([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    result = _make_stream_link(flow_dir, flow_accum, threshold=1)
    vals = result.data
    # All cells should have the same link_id (headwater at (0,0))
    assert not np.any(np.isnan(vals))
    unique = np.unique(vals[~np.isnan(vals)])
    assert len(unique) == 1
    # Headwater is (0,0), width=5 -> ID = 0*5 + 0 + 1 = 1
    assert unique[0] == 1.0


def test_y_confluence():
    """Two headwaters merge at junction -> 3 distinct link_ids."""
    # H1=(0,0) flows SE(2) to (1,1), H2=(0,2) flows SW(8) to (1,1)
    # C=(1,1) flows S(4) to (2,1), P=(2,1) is pit
    # Non-stream cells use NaN flow_dir and accum=0 to stay below threshold.
    flow_dir = np.array([
        [2.0, np.nan, 8.0],
        [np.nan, 4.0, np.nan],
        [np.nan, 0.0, np.nan],
    ], dtype=np.float64)
    flow_accum = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float64)
    result = _make_stream_link(flow_dir, flow_accum, threshold=1)
    vals = result.data

    # Non-stream cells are NaN
    assert np.isnan(vals[0, 1])
    assert np.isnan(vals[1, 0])
    # (0,0) is headwater: link_id = 0*3 + 0 + 1 = 1
    assert vals[0, 0] == 1.0
    # (0,2) is headwater: link_id = 0*3 + 2 + 1 = 3
    assert vals[0, 2] == 3.0
    # (1,1) is junction (in_degree=2): link_id = 1*3 + 1 + 1 = 5
    assert vals[1, 1] == 5.0
    # (2,1) inherits from (1,1): link_id = 5
    assert vals[2, 1] == 5.0
    # 3 distinct link_ids among stream cells
    stream_vals = vals[~np.isnan(vals)]
    assert len(np.unique(stream_vals)) == 3


def test_cascade_junctions():
    """Multiple sequential junctions -> each segment has distinct ID.

    A(0,0)->E->B(0,1), C(1,0)->NE(128)->B(0,1) => B is junction
    B(0,1)->E->D(0,2), E(1,2)->N(64)->D(0,2)   => D is junction
    D(0,2) pit
    """
    flow_dir = np.array([
        [1.0, 1.0, 0.0],
        [128.0, 0.0, 64.0],
    ], dtype=np.float64)
    flow_accum = np.array([
        [1.0, 3.0, 4.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    result = _make_stream_link(flow_dir, flow_accum, threshold=1)
    vals = result.data

    # A(0,0): headwater, link_id = 0*3+0+1 = 1
    assert vals[0, 0] == 1.0
    # C(1,0): headwater, link_id = 1*3+0+1 = 4
    assert vals[1, 0] == 4.0
    # B(0,1): junction (2 inflows), link_id = 0*3+1+1 = 2
    assert vals[0, 1] == 2.0
    # E(1,2): headwater, link_id = 1*3+2+1 = 6
    assert vals[1, 2] == 6.0
    # D(0,2): junction (2 inflows), link_id = 0*3+2+1 = 3
    assert vals[0, 2] == 3.0


def test_non_stream_nan():
    """Cells below threshold are NaN."""
    flow_dir = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    result = _make_stream_link(flow_dir, flow_accum, threshold=2)
    assert np.isnan(result.data[0, 0])
    assert not np.isnan(result.data[0, 1])
    assert not np.isnan(result.data[0, 2])


def test_nan_flow_dir():
    """NaN flow_dir cells produce NaN."""
    flow_dir = np.array([[1.0, np.nan, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    result = _make_stream_link(flow_dir, flow_accum, threshold=1)
    assert np.isnan(result.data[0, 1])
    # (0,0) still valid stream cell
    assert not np.isnan(result.data[0, 0])


def test_threshold_filtering():
    """Only cells >= threshold get link_ids."""
    flow_dir = np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    result = _make_stream_link(flow_dir, flow_accum, threshold=3)
    assert np.isnan(result.data[0, 0])
    assert np.isnan(result.data[0, 1])
    assert not np.isnan(result.data[0, 2])
    assert not np.isnan(result.data[0, 3])


def test_output_dtype():
    """Result is float64."""
    flow_dir = np.array([[1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0]], dtype=np.float64)
    result = _make_stream_link(flow_dir, flow_accum, threshold=1)
    assert result.dtype == np.float64


def test_dataset_support():
    """Dataset input -> Dataset output."""
    flow_dir = np.array([[1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0]], dtype=np.float64)
    fd_da = xr.DataArray(flow_dir, dims=['y', 'x'], attrs={'res': (0.5, 0.5)})
    fd_da['y'] = np.linspace(0, 0, 1)
    fd_da['x'] = np.linspace(0, 0.5, 2)
    fa_da = xr.DataArray(flow_accum, dims=['y', 'x'],
                         attrs={'res': (0.5, 0.5)})
    fa_da['y'] = np.linspace(0, 0, 1)
    fa_da['x'] = np.linspace(0, 0.5, 2)
    ds = xr.Dataset({'fd1': fd_da, 'fd2': fd_da.copy()})
    result = stream_link(ds, fa_da, threshold=1)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'fd1', 'fd2'}


# ====================================================================
# Cross-backend tests
# ====================================================================

def _make_cross_backend_data():
    """Y-confluence network for cross-backend tests."""
    flow_dir = np.array([
        [2.0, 0.0, 8.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)
    flow_accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 4.0, 1.0],
    ], dtype=np.float64)
    return flow_dir, flow_accum


@dask_array_available
@pytest.mark.parametrize("chunks", [(3, 3), (2, 3), (3, 2)])
def test_stream_link_numpy_equals_dask(chunks):
    """Numpy vs dask across chunk configs."""
    flow_dir, flow_accum = _make_cross_backend_data()
    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_fa = create_test_raster(flow_accum, backend='numpy')
    da_fd = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    da_fa = create_test_raster(flow_accum, backend='dask', chunks=chunks)

    np_result = stream_link(np_fd, np_fa, threshold=1)
    da_result = stream_link(da_fd, da_fa, threshold=1)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True)


@dask_array_available
def test_stream_link_dask_random():
    """Random acyclic flow: dask matches numpy."""
    from xrspatial import flow_direction, flow_accumulation

    rng = np.random.default_rng(42)
    elev = rng.random((8, 10)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    fd = flow_direction(elev_da)
    fa = flow_accumulation(fd)

    np_result = stream_link(fd, fa, threshold=3)

    fd_data = fd.data
    fa_data = fa.data
    for chunks in [(3, 3), (4, 5), (8, 10)]:
        da_fd = create_test_raster(fd_data, backend='dask', chunks=chunks)
        da_fa = create_test_raster(fa_data, backend='dask', chunks=chunks)
        da_result = stream_link(da_fd, da_fa, threshold=3)
        np.testing.assert_allclose(
            np_result.data, da_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"


@dask_array_available
def test_stream_link_dask_temp_cleanup():
    """Verify no leaked temp files after dask computation."""
    import glob
    import tempfile
    import os

    tmpdir = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(tmpdir, 'xrs_bdry_*')))

    flow_dir, flow_accum = _make_cross_backend_data()
    da_fd = create_test_raster(flow_dir, backend='dask', chunks=(2, 2))
    da_fa = create_test_raster(flow_accum, backend='dask', chunks=(2, 2))
    result = stream_link(da_fd, da_fa, threshold=1)
    _ = result.data.compute()

    after = set(glob.glob(os.path.join(tmpdir, 'xrs_bdry_*')))
    leaked = after - before
    assert len(leaked) == 0, f"Leaked temp dirs: {leaked}"


@cuda_and_cupy_available
def test_stream_link_numpy_equals_cupy():
    """GPU matches CPU."""
    flow_dir, flow_accum = _make_cross_backend_data()
    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_fa = create_test_raster(flow_accum, backend='numpy')
    cp_fd = create_test_raster(flow_dir, backend='cupy')
    cp_fa = create_test_raster(flow_accum, backend='cupy')

    np_result = stream_link(np_fd, np_fa, threshold=1)
    cp_result = stream_link(cp_fd, cp_fa, threshold=1)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_stream_link_numpy_equals_dask_cupy():
    """Numpy vs dask+cupy."""
    flow_dir, flow_accum = _make_cross_backend_data()
    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_fa = create_test_raster(flow_accum, backend='numpy')
    dcp_fd = create_test_raster(flow_dir, backend='dask+cupy', chunks=(3, 3))
    dcp_fa = create_test_raster(flow_accum, backend='dask+cupy', chunks=(3, 3))

    np_result = stream_link(np_fd, np_fa, threshold=1)
    dcp_result = stream_link(dcp_fd, dcp_fa, threshold=1)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
def test_stream_link_dask_cupy_random():
    """Random acyclic flow: dask+cupy stream_link matches numpy."""
    from xrspatial import flow_direction, flow_accumulation

    rng = np.random.default_rng(952)
    elev = rng.random((8, 10)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    fd = flow_direction(elev_da)
    fa = flow_accumulation(fd)

    np_result = stream_link(fd, fa, threshold=3)

    fd_data = fd.data
    fa_data = fa.data
    for chunks in [(3, 3), (4, 5), (2, 2)]:
        dcp_fd = create_test_raster(fd_data, backend='dask+cupy', chunks=chunks)
        dcp_fa = create_test_raster(fa_data, backend='dask+cupy', chunks=chunks)
        dcp_result = stream_link(dcp_fd, dcp_fa, threshold=3)
        np.testing.assert_allclose(
            np_result.data, dcp_result.data.compute().get(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}"
