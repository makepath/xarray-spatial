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
# Cross-tile connected components -- regression for #1394
# -------------------------------------------------------------------
#
# Per-tile CCL must be merged across tile boundaries so a connected
# sink that straddles a chunk gets one label, matching the numpy
# backend's behavior.

def _equiv_labels(arr_a, arr_b):
    """Return True if two label rasters describe the same partitioning.

    Two outputs are equivalent when they agree on which cells are sinks
    (NaN pattern matches) and when same-label-in-A pairs are also
    same-label-in-B.  We don't require literal label equality because
    the dask path uses position-based IDs that differ from the numpy
    path's IDs after cross-tile merging.
    """
    if arr_a.shape != arr_b.shape:
        return False
    nan_a = np.isnan(arr_a)
    nan_b = np.isnan(arr_b)
    if not np.array_equal(nan_a, nan_b):
        return False

    valid = ~nan_a
    a_vals = arr_a[valid]
    b_vals = arr_b[valid]
    # Build mapping a_label -> b_label and check it's consistent.
    mapping = {}
    for a, b in zip(a_vals.tolist(), b_vals.tolist()):
        if a in mapping:
            if mapping[a] != b:
                return False
        else:
            mapping[a] = b
    # And the reverse mapping must also be 1:1.
    rev = {}
    for a, b in mapping.items():
        if b in rev:
            return False
        rev[b] = a
    return True


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (1, 1), (1, 2), (1, 3),
])
def test_dask_horizontal_sink_across_tiles(chunks):
    """Horizontal connected sink straddling a chunk boundary."""
    flow_dir = np.array([[1.0, 0.0, 0.0, 16.0]], dtype=np.float64)
    np_agg = create_test_raster(flow_dir, backend='numpy')
    dk_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    np_result = sink(np_agg).data
    dk_result = sink(dk_agg).data.compute()
    assert _equiv_labels(np_result, dk_result), (
        f"chunks={chunks}: dask result {dk_result} does not match "
        f"numpy partitioning {np_result}"
    )
    # Specifically, the two sink cells should share a label
    assert dk_result[0, 1] == dk_result[0, 2]


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (1, 1), (2, 1), (3, 1),
])
def test_dask_vertical_sink_across_tiles(chunks):
    """Vertical connected sink straddling a chunk boundary."""
    flow_dir = np.array([
        [1.0, 1.0, 16.0],
        [1.0, 0.0, 16.0],
        [1.0, 0.0, 16.0],
        [1.0, 1.0, 16.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(flow_dir, backend='numpy')
    dk_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    np_result = sink(np_agg).data
    dk_result = sink(dk_agg).data.compute()
    assert _equiv_labels(np_result, dk_result)
    assert dk_result[1, 1] == dk_result[2, 1]


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (1, 1), (2, 2), (1, 2), (2, 1),
])
def test_dask_diagonal_sink_across_tiles(chunks):
    """Diagonal connected sink across a corner-shared tile boundary.

    8-connectivity means a sink chain along the NW-SE diagonal must
    survive a chunk boundary that separates the two cells corner-to-corner.
    """
    flow_dir = np.array([
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(flow_dir, backend='numpy')
    dk_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    np_result = sink(np_agg).data
    dk_result = sink(dk_agg).data.compute()
    assert _equiv_labels(np_result, dk_result)
    # All four diagonal cells are one connected sink under 8-connectivity
    sink_cells = [(0, 0), (1, 1), (2, 2), (3, 3)]
    first = dk_result[sink_cells[0]]
    for r, c in sink_cells[1:]:
        assert dk_result[r, c] == first


@dask_array_available
@pytest.mark.parametrize("chunks", [
    (1, 1), (2, 2), (1, 2), (2, 1), (3, 3),
])
def test_dask_block_sink_across_four_tiles(chunks):
    """Sink block spanning a 2x2 grid of tiles (all four corners meet)."""
    flow_dir = np.array([
        [1.0, 0.0, 0.0, 0.0, 16.0],
        [1.0, 0.0, 0.0, 0.0, 16.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(flow_dir, backend='numpy')
    dk_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    np_result = sink(np_agg).data
    dk_result = sink(dk_agg).data.compute()
    assert _equiv_labels(np_result, dk_result)
    # All six sink cells should share one label
    sink_cells = [(r, c) for r in range(2) for c in range(1, 4)]
    first = dk_result[sink_cells[0]]
    for r, c in sink_cells[1:]:
        assert dk_result[r, c] == first


@dask_array_available
def test_dask_separate_sinks_stay_separate():
    """Two distinct sinks separated by a non-sink chunk keep distinct labels."""
    flow_dir = np.array([
        [0.0, 1.0, 1.0, 1.0, 0.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(flow_dir, backend='numpy')
    dk_agg = create_test_raster(flow_dir, backend='dask', chunks=(1, 2))
    np_result = sink(np_agg).data
    dk_result = sink(dk_agg).data.compute()
    assert _equiv_labels(np_result, dk_result)
    # Different sinks -> different labels in both
    assert dk_result[0, 0] != dk_result[0, 4]


@dask_array_available
@pytest.mark.parametrize("chunks", [(1, 1), (2, 2), (3, 3)])
def test_dask_label_count_matches_numpy(chunks):
    """Number of unique labels in dask output equals numpy output."""
    flow_dir = np.array([
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
    ], dtype=np.float64)
    np_agg = create_test_raster(flow_dir, backend='numpy')
    dk_agg = create_test_raster(flow_dir, backend='dask', chunks=chunks)
    np_result = sink(np_agg).data
    dk_result = sink(dk_agg).data.compute()
    np_labels = np.unique(np_result[~np.isnan(np_result)])
    dk_labels = np.unique(dk_result[~np.isnan(dk_result)])
    assert len(np_labels) == len(dk_labels), (
        f"chunks={chunks}: numpy found {len(np_labels)} sinks, "
        f"dask found {len(dk_labels)}"
    )
    assert _equiv_labels(np_result, dk_result)


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


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

class TestMemoryGuard:
    """Memory guard on the eager numpy / cupy backends."""

    def test_numpy_huge_raster_raises(self):
        """Numpy backend raises MemoryError when projected RAM exceeds budget."""
        from unittest.mock import patch

        flow_dir = np.zeros((4, 4), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='numpy')

        with patch(
            "xrspatial.hydro.sink_d8._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="working memory"):
                sink(agg)

    def test_numpy_normal_input_succeeds(self):
        """Normal-size raster passes the guard with real memory."""
        flow_dir = np.zeros((10, 10), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='numpy')
        result = sink(agg)
        assert result.shape == (10, 10)

    @dask_array_available
    def test_dask_path_skips_guard(self):
        """Dask backend bypasses the guard -- per-tile allocations are bounded."""
        from unittest.mock import patch

        flow_dir = np.zeros((20, 20), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='dask+numpy', chunks=(5, 5))

        with patch(
            "xrspatial.hydro.sink_d8._available_memory_bytes",
            return_value=1,
        ):
            result = sink(agg)
            _ = result.data[:4, :4].compute()

    def test_error_message_mentions_dask(self):
        """The error message should suggest the dask alternative."""
        from unittest.mock import patch

        flow_dir = np.zeros((4, 4), dtype=np.float64)
        agg = create_test_raster(flow_dir, backend='numpy')

        with patch(
            "xrspatial.hydro.sink_d8._available_memory_bytes",
            return_value=1,
        ):
            with pytest.raises(MemoryError, match="dask"):
                sink(agg)

    def test_byte_per_pixel_constants(self):
        """Pin the documented per-pixel costs so refactors flag accidental changes."""
        import importlib
        mod = importlib.import_module("xrspatial.hydro.sink_d8")

        assert mod._BYTES_PER_PIXEL == 24
        assert mod._GPU_BYTES_PER_PIXEL == 8
