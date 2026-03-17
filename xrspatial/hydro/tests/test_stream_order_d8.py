import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro import stream_order
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ====================================================================
# Helpers
# ====================================================================

def _make_stream_order(flow_dir, flow_accum, threshold=0, **kwargs):
    """Shortcut: wrap raw arrays in DataArrays and call stream_order."""
    fd_da = create_test_raster(flow_dir)
    fa_da = create_test_raster(flow_accum)
    return stream_order(fd_da, fa_da, threshold=threshold, **kwargs)


# ====================================================================
# Strahler tests
# ====================================================================

def test_y_confluence():
    """Two order-1 streams merge -> order 2 downstream."""
    #  Row 0: two headwaters flowing south
    #  Row 1: confluence cell flowing south
    #  Row 2: pit
    #
    #  Layout (3x3):
    #    .  H1  .       H1 flows S (4), H2 flows S (4)
    #    .   C  .       C is the confluence (pit below)
    #    .   P  .       P is the pit
    #
    #  H1=(0,0) flows SE(2) to (1,1), H2=(0,2) flows SW(8) to (1,1)
    #  C=(1,1) flows S(4) to (2,1), P=(2,1) is pit
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
    result = _make_stream_order(flow_dir, flow_accum, threshold=1,
                                ordering='strahler')
    # (0,0) and (0,2) are headwaters: order 1
    assert result.data[0, 0] == 1.0
    assert result.data[0, 2] == 1.0
    # (1,1): two order-1 inflows merge -> order 2
    assert result.data[1, 1] == 2.0
    # (2,1): single order-2 inflow -> stays 2
    assert result.data[2, 1] == 2.0


def test_unequal_confluence():
    """Order 1 meets order 2 -> stays order 2.

    Network (3x3):
      H1(0,0)->SE  H2(0,2)->SW     -> merge at (1,1) = order 2
                    (1,1)->S(4)     -> (2,1) gets order-2 inflow
      H3(2,0)->E(1)                -> (2,1) gets order-1 inflow
                    (2,1) pit       -> max=2, cnt=1 -> order 2
    """
    flow_dir = np.array([
        [2.0, 0.0, 8.0],
        [0.0, 4.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)
    flow_accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 5.0, 1.0],
    ], dtype=np.float64)
    result = _make_stream_order(flow_dir, flow_accum, threshold=1,
                                ordering='strahler')
    assert result.data[0, 0] == 1.0  # headwater
    assert result.data[0, 2] == 1.0  # headwater
    assert result.data[1, 1] == 2.0  # two order-1 merge
    assert result.data[2, 0] == 1.0  # headwater
    # (2,1): order-2 from (1,1), order-1 from (2,0) -> max=2, cnt=1 -> 2
    assert result.data[2, 1] == 2.0


def test_order_3():
    """Two order-2 streams merge -> order 3.

    Network (3x5):
      H1(0,0)->SE  H2(0,2)->SW  => merge at (1,1) = order 2, flows E
      H3(0,4)->SW  H4(2,4)->NW  => merge at (1,3) = order 2, flows W
      (1,1)->E->(1,2), (1,3)->W(16)->(1,2) => two order-2 merge -> order 3
      (1,2) pit
    """
    flow_dir = np.array([
        [2.0, 0.0, 8.0, 0.0, 8.0],
        [0.0, 1.0, 0.0, 16.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 32.0],
    ], dtype=np.float64)
    flow_accum = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 3.0, 7.0, 3.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float64)
    result = _make_stream_order(flow_dir, flow_accum, threshold=1,
                                ordering='strahler')
    assert result.data[1, 1] == 2.0  # two order-1 merge
    assert result.data[1, 3] == 2.0  # two order-1 merge
    assert result.data[1, 2] == 3.0  # two order-2 merge


def test_order_3_equal_merge():
    """Two order-2 streams with equal merge -> order 3."""
    # Top branch: (0,0)->E->(0,1)->E->(0,2) = order 2
    # Bottom branch: (2,0)->E->(2,1)->E->(2,2) = order 2
    # (0,2)->S->(1,2), (2,2)->N(64)->(1,2), (1,2)=pit
    flow_dir = np.array([
        [1.0, 1.0, 4.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 64.0],
    ], dtype=np.float64)
    flow_accum = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 7.0],
        [1.0, 2.0, 3.0],
    ], dtype=np.float64)
    result = _make_stream_order(flow_dir, flow_accum, threshold=2,
                                ordering='strahler')
    # Only cells with accum >= 2 are stream cells
    assert np.isnan(result.data[0, 0])  # accum=1, not stream
    assert np.isnan(result.data[2, 0])  # accum=1, not stream
    assert np.isnan(result.data[1, 0])  # accum=1
    assert np.isnan(result.data[1, 1])  # accum=1
    # (0,1): headwater among stream cells (no stream inflow), order 1
    assert result.data[0, 1] == 1.0
    # (0,2): one order-1 inflow -> order 1
    assert result.data[0, 2] == 1.0
    # (2,1): headwater among stream cells, order 1
    assert result.data[2, 1] == 1.0
    # (2,2): one order-1 inflow -> order 1
    assert result.data[2, 2] == 1.0
    # (1,2): two order-1 inflows merge -> order 2
    assert result.data[1, 2] == 2.0


def test_linear_chain():
    """Single stream path -> all order 1."""
    flow_dir = np.array([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    result = _make_stream_order(flow_dir, flow_accum, threshold=1,
                                ordering='strahler')
    expected = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    np.testing.assert_array_equal(result.data, expected)


# ====================================================================
# Shreve tests
# ====================================================================

def test_shreve_y_confluence():
    """Two magnitude-1 streams merge -> magnitude 2."""
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
    result = _make_stream_order(flow_dir, flow_accum, threshold=1,
                                ordering='shreve')
    assert result.data[0, 0] == 1.0
    assert result.data[0, 2] == 1.0
    assert result.data[1, 1] == 2.0  # sum of two 1s
    assert result.data[2, 1] == 2.0  # propagated


def test_shreve_triple():
    """Three magnitude-1 streams merge -> magnitude 3."""
    # (0,0)->SE(2)->(1,1), (0,2)->SW(8)->(1,1), (0,1)->S(4)->(1,1)
    # (1,1) is pit
    flow_dir = np.array([
        [2.0, 4.0, 8.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)
    flow_accum = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 4.0, 1.0],
    ], dtype=np.float64)
    result = _make_stream_order(flow_dir, flow_accum, threshold=1,
                                ordering='shreve')
    assert result.data[0, 0] == 1.0
    assert result.data[0, 1] == 1.0
    assert result.data[0, 2] == 1.0
    assert result.data[1, 1] == 3.0


def test_shreve_cascade():
    """Cumulative sum through network.

    A(0,0)->E->B(0,1), C(1,0)->NE(128)->B(0,1) => B=2
    B(0,1)->E->D(0,2), E(1,2)->N(64)->D(0,2)   => D=3
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
    result = _make_stream_order(flow_dir, flow_accum, threshold=1,
                                ordering='shreve')
    assert result.data[0, 0] == 1.0  # headwater
    assert result.data[1, 0] == 1.0  # headwater
    assert result.data[0, 1] == 2.0  # A + C
    assert result.data[1, 2] == 1.0  # headwater
    assert result.data[0, 2] == 3.0  # B + E


# ====================================================================
# Threshold tests
# ====================================================================

def test_threshold_filters_cells():
    """High threshold -> fewer stream cells."""
    flow_dir = np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    result = _make_stream_order(flow_dir, flow_accum, threshold=3,
                                ordering='strahler')
    assert np.isnan(result.data[0, 0])
    assert np.isnan(result.data[0, 1])
    assert result.data[0, 2] == 1.0
    assert result.data[0, 3] == 1.0


def test_threshold_zero():
    """threshold=0 -> all cells are stream cells."""
    flow_dir = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    result = _make_stream_order(flow_dir, flow_accum, threshold=0,
                                ordering='strahler')
    assert not np.any(np.isnan(result.data))


# ====================================================================
# Edge cases
# ====================================================================

def test_no_streams():
    """Threshold higher than max accumulation -> all NaN."""
    flow_dir = np.array([[1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0]], dtype=np.float64)
    result = _make_stream_order(flow_dir, flow_accum, threshold=100,
                                ordering='strahler')
    assert np.all(np.isnan(result.data))


def test_nan_handling():
    """NaN in flow_dir -> NaN in output."""
    flow_dir = np.array([
        [1.0, np.nan, 0.0],
    ], dtype=np.float64)
    flow_accum = np.array([
        [1.0, 2.0, 3.0],
    ], dtype=np.float64)
    result = _make_stream_order(flow_dir, flow_accum, threshold=1,
                                ordering='strahler')
    assert np.isnan(result.data[0, 1])
    # (0,0) still valid stream cell
    assert result.data[0, 0] == 1.0


def test_pit_in_stream():
    """Pit cell (code 0) in stream -> order assigned, no downstream."""
    flow_dir = np.array([[1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0]], dtype=np.float64)
    result = _make_stream_order(flow_dir, flow_accum, threshold=1,
                                ordering='strahler')
    assert result.data[0, 0] == 1.0
    assert result.data[0, 1] == 1.0  # gets inflow from (0,0), single -> 1


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
    result = stream_order(ds, fa_da, threshold=1)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {'fd1', 'fd2'}


def test_invalid_method():
    """Invalid method raises ValueError."""
    flow_dir = np.array([[1.0, 0.0]], dtype=np.float64)
    flow_accum = np.array([[1.0, 2.0]], dtype=np.float64)
    fd_da = create_test_raster(flow_dir)
    fa_da = create_test_raster(flow_accum)
    with pytest.raises(ValueError, match="ordering must be"):
        stream_order(fd_da, fa_da, ordering='horton')


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
@pytest.mark.parametrize("method", ['strahler', 'shreve'])
def test_dask_equivalence(chunks, method):
    """Numpy vs dask across chunk configs."""
    flow_dir, flow_accum = _make_cross_backend_data()
    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_fa = create_test_raster(flow_accum, backend='numpy')
    da_fd = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    da_fa = create_test_raster(flow_accum, backend='dask', chunks=chunks)

    np_result = stream_order(np_fd, np_fa, threshold=1, ordering=method)
    da_result = stream_order(da_fd, da_fa, threshold=1, ordering=method)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True)


@dask_array_available
@pytest.mark.parametrize("method", ['strahler', 'shreve'])
def test_dask_cross_tile_confluence(method):
    """Confluence spanning tile boundary -> correct order."""
    # Larger grid: flow from top-left to bottom-right, merge at (3,3)
    flow_dir = np.array([
        [2.0, 0.0, 0.0, 8.0, 0.0],
        [0.0, 2.0, 0.0, 8.0, 0.0],
        [0.0, 0.0, 2.0, 8.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)
    flow_accum = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 3.0, 3.0, 1.0],
        [1.0, 1.0, 1.0, 7.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float64)

    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_fa = create_test_raster(flow_accum, backend='numpy')
    np_result = stream_order(np_fd, np_fa, threshold=1, ordering=method)

    for chunks in [(2, 2), (3, 3), (2, 5)]:
        da_fd = create_test_raster(flow_dir, backend='dask', chunks=chunks)
        da_fa = create_test_raster(flow_accum, backend='dask', chunks=chunks)
        da_result = stream_order(da_fd, da_fa, threshold=1, ordering=method)
        np.testing.assert_allclose(
            np_result.data, da_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}, method={method}"


@dask_array_available
@pytest.mark.parametrize("method", ['strahler', 'shreve'])
def test_dask_random(method):
    """Random acyclic flow: dask matches numpy."""
    from xrspatial.hydro import flow_direction, flow_accumulation

    rng = np.random.default_rng(123)
    elev = rng.random((8, 10)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    fd = flow_direction(elev_da)
    fa = flow_accumulation(fd)

    np_result = stream_order(fd, fa, threshold=3, ordering=method)

    fd_data = fd.data
    fa_data = fa.data
    for chunks in [(3, 3), (4, 5), (8, 10)]:
        da_fd = create_test_raster(fd_data, backend='dask', chunks=chunks)
        da_fa = create_test_raster(fa_data, backend='dask', chunks=chunks)
        da_result = stream_order(da_fd, da_fa, threshold=3, ordering=method)
        np.testing.assert_allclose(
            np_result.data, da_result.data.compute(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}, method={method}"


@cuda_and_cupy_available
@pytest.mark.parametrize("method", ['strahler', 'shreve'])
def test_gpu_equivalence(method):
    """Numpy vs cupy."""
    flow_dir, flow_accum = _make_cross_backend_data()
    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_fa = create_test_raster(flow_accum, backend='numpy')
    cp_fd = create_test_raster(flow_dir, backend='cupy')
    cp_fa = create_test_raster(flow_accum, backend='cupy')

    np_result = stream_order(np_fd, np_fa, threshold=1, ordering=method)
    cp_result = stream_order(cp_fd, cp_fa, threshold=1, ordering=method)
    np.testing.assert_allclose(
        np_result.data, cp_result.data.get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("method", ['strahler', 'shreve'])
def test_dask_cupy_equivalence(method):
    """Numpy vs dask+cupy."""
    flow_dir, flow_accum = _make_cross_backend_data()
    np_fd = create_test_raster(flow_dir, backend='numpy')
    np_fa = create_test_raster(flow_accum, backend='numpy')
    dcp_fd = create_test_raster(flow_dir, backend='dask+cupy', chunks=(3, 3))
    dcp_fa = create_test_raster(flow_accum, backend='dask+cupy', chunks=(3, 3))

    np_result = stream_order(np_fd, np_fa, threshold=1, ordering=method)
    dcp_result = stream_order(dcp_fd, dcp_fa, threshold=1, ordering=method)
    np.testing.assert_allclose(
        np_result.data, dcp_result.data.compute().get(), equal_nan=True)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("method", ['strahler', 'shreve'])
def test_dask_cupy_random(method):
    """Random acyclic flow: dask+cupy matches numpy."""
    from xrspatial.hydro import flow_direction, flow_accumulation

    rng = np.random.default_rng(952)
    elev = rng.random((8, 10)).astype(np.float64)
    elev_da = create_test_raster(elev, backend='numpy')
    fd = flow_direction(elev_da)
    fa = flow_accumulation(fd)

    np_result = stream_order(fd, fa, threshold=3, ordering=method)

    fd_data = fd.data
    fa_data = fa.data
    for chunks in [(3, 3), (4, 5), (2, 2)]:
        dcp_fd = create_test_raster(fd_data, backend='dask+cupy', chunks=chunks)
        dcp_fa = create_test_raster(fa_data, backend='dask+cupy', chunks=chunks)
        dcp_result = stream_order(dcp_fd, dcp_fa, threshold=3, ordering=method)
        np.testing.assert_allclose(
            np_result.data, dcp_result.data.compute().get(), equal_nan=True,
        ), f"Mismatch with chunks={chunks}, method={method}"
