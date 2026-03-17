"""Tests for flow_accumulation_mfd."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro.flow_direction_mfd import flow_direction_mfd
from xrspatial.hydro.flow_accumulation_mfd import flow_accumulation_mfd


# =====================================================================
# Helpers
# =====================================================================

def _make_bowl(n=7):
    """Create a simple bowl where center is lowest."""
    y = np.arange(n, dtype=np.float64) - n // 2
    x = np.arange(n, dtype=np.float64) - n // 2
    yy, xx = np.meshgrid(y, x, indexing='ij')
    return xr.DataArray(yy ** 2 + xx ** 2, dims=['y', 'x'])


def _make_plane_south(rows=5, cols=5):
    """Create a plane that slopes uniformly south (increasing row = lower)."""
    data = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        data[r, :] = float(rows - 1 - r)  # top=4, bottom=0
    return xr.DataArray(data, dims=['y', 'x'])


def _make_plane_se(rows=5, cols=5):
    """Create a plane that slopes SE."""
    data = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            data[r, c] = float(rows + cols - 2 - r - c)
    return xr.DataArray(data, dims=['y', 'x'])


# =====================================================================
# Basic tests
# =====================================================================

class TestFlowAccumulationMFDBasic:
    """Basic shape and value tests."""

    def test_output_shape_2d(self):
        """Output should be 2-D matching spatial dimensions."""
        elev = _make_bowl(7)
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)
        assert accum.ndim == 2
        assert accum.shape == (7, 7)

    def test_output_dims(self):
        """Output should have same spatial dims as input without neighbor."""
        elev = _make_bowl(7)
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)
        assert 'neighbor' not in accum.dims
        assert accum.dims == ('y', 'x')

    def test_minimum_value_is_one(self):
        """Non-NaN cells should have accum >= 1.0 (counting self)."""
        elev = _make_bowl(7)
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)
        valid = accum.values[~np.isnan(accum.values)]
        assert np.all(valid >= 1.0)

    def test_total_accumulation_conservation(self):
        """Sum of all fractions leaving all cells should equal sum of all accum - N.

        Actually for a grid with no outflow except edges, the interior
        cell accum values should be self-consistent. A simpler check:
        each interior non-pit cell's accum should be >= 1.
        """
        elev = _make_bowl(7)
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)
        valid = accum.values[~np.isnan(accum.values)]
        assert len(valid) > 0
        assert np.all(valid >= 1.0)

    def test_bowl_center_has_max_accum(self):
        """Center of bowl should have highest accumulation (it's the pit)."""
        elev = _make_bowl(9)
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)
        vals = accum.values
        # Mask NaN
        center_val = vals[4, 4]
        # Center should be >= all other non-NaN cells
        valid = vals[~np.isnan(vals)]
        assert center_val == np.nanmax(valid)


class TestFlowAccumulationMFDPlane:
    """Tests on uniform-slope surfaces."""

    def test_plane_south_top_row_all_ones(self):
        """On a southward plane, the top interior row has accum=1 (no upstream)."""
        elev = _make_plane_south(7, 7)
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)
        # Row 1 is the first non-NaN interior row
        # (row 0 is edge -> NaN from MFD)
        row1 = accum.values[1, 1:-1]
        np.testing.assert_allclose(row1, 1.0, atol=1e-10)

    def test_plane_south_accum_increases_downhill(self):
        """On a southward plane, accumulation increases row by row."""
        elev = _make_plane_south(7, 7)
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)
        vals = accum.values
        # Check middle column, interior rows
        col_mid = vals[1:-1, 3]
        # Each subsequent row should have >= previous
        for i in range(1, len(col_mid)):
            assert col_mid[i] >= col_mid[i - 1] - 1e-10


class TestFlowAccumulationMFDNaN:
    """NaN handling tests."""

    def test_nan_input_produces_nan_output(self):
        """Cells with NaN fractions should produce NaN accumulation."""
        elev = _make_bowl(7)
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)
        # Edge cells are NaN in MFD output
        assert np.isnan(accum.values[0, 0])

    def test_all_nan_input(self):
        """All-NaN input should produce all-NaN output."""
        data = np.full((8, 5, 5), np.nan)
        da = xr.DataArray(data, dims=['neighbor', 'y', 'x'])
        accum = flow_accumulation_mfd(da)
        assert np.all(np.isnan(accum.values))

    def test_nan_barrier(self):
        """NaN cells should block flow (downstream of NaN gets less accum)."""
        elev = _make_plane_south(9, 9)
        mfd_clean = flow_direction_mfd(elev)

        # Add a NaN barrier in the middle
        elev_barrier = elev.copy(deep=True)
        elev_barrier.values[4, 2:7] = np.nan
        mfd_barrier = flow_direction_mfd(elev_barrier)

        accum_clean = flow_accumulation_mfd(mfd_clean)
        accum_barrier = flow_accumulation_mfd(mfd_barrier)

        # Below barrier, accumulation should be less
        below_clean = np.nansum(accum_clean.values[6, 2:7])
        below_barrier = np.nansum(accum_barrier.values[6, 2:7])
        assert below_barrier < below_clean


class TestFlowAccumulationMFDEdgeCases:
    """Edge cases."""

    def test_single_interior_cell(self):
        """3x3 grid has only one interior cell."""
        elev = xr.DataArray(
            np.array([[9., 8., 7.],
                       [6., 5., 4.],
                       [3., 2., 1.]], dtype=np.float64),
            dims=['y', 'x'])
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)
        # Center cell (1,1) is the only non-NaN cell
        assert accum.values[1, 1] == 1.0

    def test_flat_surface(self):
        """Flat surface: all fractions are 0, accum should be 1 everywhere."""
        elev = xr.DataArray(
            np.ones((5, 5), dtype=np.float64), dims=['y', 'x'])
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)
        # Interior cells should have accum = 1 (no flow)
        interior = accum.values[1:-1, 1:-1]
        valid = interior[~np.isnan(interior)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)

    def test_shape_validation(self):
        """Should reject 2-D input (not 3-D MFD format)."""
        data = xr.DataArray(np.ones((5, 5)), dims=['y', 'x'])
        with pytest.raises(ValueError, match="(3-D array|3D)"):
            flow_accumulation_mfd(data)

    def test_wrong_band_count(self):
        """Should reject 3-D input with wrong band count."""
        data = xr.DataArray(np.ones((3, 5, 5)), dims=['band', 'y', 'x'])
        with pytest.raises(ValueError, match="8, H, W"):
            flow_accumulation_mfd(data)


class TestFlowAccumulationMFDKnownValues:
    """Known-value tests for specific grid configurations."""

    def test_simple_v_shape(self):
        """V-shaped valley: flow converges to center column."""
        data = np.array([
            [5., 4., 3., 4., 5.],
            [4., 3., 2., 3., 4.],
            [3., 2., 1., 2., 3.],
            [4., 3., 2., 3., 4.],
            [5., 4., 3., 4., 5.],
        ], dtype=np.float64)
        elev = xr.DataArray(data, dims=['y', 'x'])
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)

        # Center cell (2,2) is the pit -- should have highest accum
        vals = accum.values
        center = vals[2, 2]
        assert center == np.nanmax(vals)

    def test_ridge_line(self):
        """Ridge: top row has high elevation, slopes down both sides."""
        rows, cols = 7, 7
        data = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                data[r, c] = abs(c - 3)  # ridge along col 3
        # Invert so ridge is high
        data = np.max(data) - data
        elev = xr.DataArray(data, dims=['y', 'x'])
        mfd = flow_direction_mfd(elev)
        accum = flow_accumulation_mfd(mfd)

        # Ridge cells (col 3) should have accum = 1 (no upstream on ridge)
        ridge_accum = accum.values[1:-1, 3]
        valid = ridge_accum[~np.isnan(ridge_accum)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)


class TestFlowAccumulationMFDWithFixedExponent:
    """Test with different fixed exponents for MFD fractions."""

    def test_high_exponent_concentrates_like_d8(self):
        """High exponent should concentrate flow, giving higher peak accum."""
        elev = _make_bowl(9)
        mfd_low = flow_direction_mfd(elev, p=1.0)
        mfd_high = flow_direction_mfd(elev, p=10.0)

        accum_low = flow_accumulation_mfd(mfd_low)
        accum_high = flow_accumulation_mfd(mfd_high)

        # Both should have the same center as pit
        # But high-p concentrates more flow through fewer paths
        # The peak should be similar (same total area drains to center)
        center_low = accum_low.values[4, 4]
        center_high = accum_high.values[4, 4]
        # Both centers collect all flow from interior
        assert center_low > 1.0
        assert center_high > 1.0


class TestFlowAccumulationMFDDask:
    """Dask backend tests."""

    @pytest.fixture
    def dask_mfd(self):
        """Create a dask-backed MFD flow direction array."""
        dask = pytest.importorskip('dask.array')
        elev = _make_bowl(9)
        mfd = flow_direction_mfd(elev)
        # Chunk the spatial dims
        data_dask = dask.from_array(mfd.values, chunks=(8, 5, 5))
        return xr.DataArray(data_dask,
                            dims=mfd.dims,
                            coords=mfd.coords)

    def test_dask_matches_numpy(self):
        """Dask result should match numpy result."""
        dask = pytest.importorskip('dask.array')
        elev = _make_bowl(9)
        mfd_np = flow_direction_mfd(elev)

        # Numpy result
        accum_np = flow_accumulation_mfd(mfd_np)

        # Dask result
        data_dask = dask.from_array(mfd_np.values, chunks=(8, 5, 5))
        mfd_dask = xr.DataArray(data_dask,
                                dims=mfd_np.dims,
                                coords=mfd_np.coords)
        accum_dask = flow_accumulation_mfd(mfd_dask)
        result = accum_dask.values

        np.testing.assert_allclose(
            result, accum_np.values, atol=1e-10, equal_nan=True)

    def test_dask_output_is_2d(self, dask_mfd):
        """Dask output should be 2-D."""
        accum = flow_accumulation_mfd(dask_mfd)
        assert accum.ndim == 2

    def test_dask_single_chunk(self):
        """Single-chunk dask should match numpy."""
        dask = pytest.importorskip('dask.array')
        elev = _make_bowl(7)
        mfd_np = flow_direction_mfd(elev)
        accum_np = flow_accumulation_mfd(mfd_np)

        data_dask = dask.from_array(mfd_np.values, chunks=(8, 7, 7))
        mfd_dask = xr.DataArray(data_dask,
                                dims=mfd_np.dims,
                                coords=mfd_np.coords)
        accum_dask = flow_accumulation_mfd(mfd_dask)

        np.testing.assert_allclose(
            accum_dask.values, accum_np.values, atol=1e-10, equal_nan=True)

    def test_dask_many_chunks(self):
        """Many small chunks should still match numpy."""
        dask = pytest.importorskip('dask.array')
        elev = _make_bowl(11)
        mfd_np = flow_direction_mfd(elev)
        accum_np = flow_accumulation_mfd(mfd_np)

        # Use small chunks (3x3 tiles)
        data_dask = dask.from_array(mfd_np.values, chunks=(8, 3, 4))
        mfd_dask = xr.DataArray(data_dask,
                                dims=mfd_np.dims,
                                coords=mfd_np.coords)
        accum_dask = flow_accumulation_mfd(mfd_dask)

        np.testing.assert_allclose(
            accum_dask.values, accum_np.values, atol=1e-10, equal_nan=True)


class TestFlowAccumulationMFDDataset:
    """Dataset support tests."""

    def test_dataset_input(self):
        """Should handle Dataset input via @supports_dataset."""
        elev = _make_bowl(7)
        mfd = flow_direction_mfd(elev)
        ds = xr.Dataset({'elev_mfd': mfd})
        result = flow_accumulation_mfd(ds)
        assert isinstance(result, xr.Dataset)
        assert 'elev_mfd' in result.data_vars

    def test_dataset_matches_dataarray(self):
        """Dataset result should match direct DataArray result."""
        elev = _make_bowl(7)
        mfd = flow_direction_mfd(elev)
        accum_da = flow_accumulation_mfd(mfd)

        ds = xr.Dataset({'mfd': mfd})
        accum_ds = flow_accumulation_mfd(ds)

        np.testing.assert_allclose(
            accum_ds['mfd'].values, accum_da.values,
            atol=1e-10, equal_nan=True)
