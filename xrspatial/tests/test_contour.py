import numpy as np
import pytest
import xarray as xr

from xrspatial.contour import contours, _contours_numpy, _stitch_segments
from xrspatial.tests.general_checks import (
    create_test_raster,
    cuda_and_cupy_available,
    dask_array_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ramp(ny=5, nx=6):
    """Simple left-to-right linear ramp: values 0..nx-1 repeated down rows."""
    return np.tile(np.arange(nx, dtype=np.float64), (ny, 1))


def _make_peak():
    """Small 5x5 raster with a peak in the center."""
    data = np.array([
        [0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 0.],
        [0., 1., 2., 1., 0.],
        [0., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0.],
    ], dtype=np.float64)
    return data


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestMarchingSquaresBasic:

    def test_no_contours_flat(self):
        """A flat raster produces no contour lines at any level outside it."""
        data = np.ones((4, 4), dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.5, 1.5])
        assert isinstance(result, list)
        # Level 0.5 is below entire surface, 1.5 is above -> no crossings.
        assert len(result) == 0

    def test_single_level_ramp(self):
        """A horizontal ramp should produce vertical contour lines."""
        data = _make_ramp(ny=5, nx=6)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[2.5])
        assert len(result) > 0
        # With res=0.5 x_coords = [0, 0.5, 1.0, 1.5, 2.0, 2.5].
        # The crossing between col indices 2 and 3 maps to x = 1.25.
        expected_x = 1.25
        for level, coords in result:
            assert level == 2.5
            np.testing.assert_allclose(coords[:, 1], expected_x, atol=1e-10)

    def test_multiple_levels_ramp(self):
        """Multiple levels on a ramp produce one line per level."""
        data = _make_ramp(ny=5, nx=6)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.5, 2.5, 3.5])
        levels_found = sorted(set(lvl for lvl, _ in result))
        assert levels_found == [1.5, 2.5, 3.5]

    def test_peak_contour_is_closed(self):
        """A contour fully inside the raster should form a closed ring."""
        # Use a larger peak so the 0.5 contour doesn't hit the boundary.
        data = np.array([
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 1., 1., 0., 0.],
            [0., 0., 1., 2., 1., 0., 0.],
            [0., 0., 1., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.5])
        assert len(result) >= 1
        for level, coords in result:
            # Closed ring: first point equals last point.
            np.testing.assert_allclose(coords[0], coords[-1], atol=1e-10)

    def test_peak_contour_at_1_5(self):
        """Contour at level 1.5 around a 2-valued peak."""
        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.5])
        assert len(result) >= 1
        y_min = float(agg.coords[agg.dims[0]].values.min())
        y_max = float(agg.coords[agg.dims[0]].values.max())
        x_min = float(agg.coords[agg.dims[1]].values.min())
        x_max = float(agg.coords[agg.dims[1]].values.max())
        for level, coords in result:
            assert level == 1.5
            # All points should be inside the raster coordinate bounds.
            assert np.all(coords[:, 0] >= y_min - 1e-10)
            assert np.all(coords[:, 0] <= y_max + 1e-10)
            assert np.all(coords[:, 1] >= x_min - 1e-10)
            assert np.all(coords[:, 1] <= x_max + 1e-10)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:

    def test_all_nan(self):
        """All-NaN raster produces no contours."""
        data = np.full((4, 4), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.0])
        assert result == []

    def test_partial_nan(self):
        """Contours skip quads with NaN corners."""
        data = _make_ramp(ny=5, nx=6)
        data[0, :] = np.nan  # top row is NaN
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[2.5])
        assert len(result) > 0
        # y_coords = [2.0, 1.5, 1.0, 0.5, 0.0] (decreasing with res=0.5).
        # NaN row is row 0 (y=2.0).  All contour points must stay at y <= 1.5.
        nan_row_y = agg.coords[agg.dims[0]].values[0]
        for level, coords in result:
            assert np.all(coords[:, 0] < nan_row_y + 1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_minimum_raster(self):
        """2x2 raster (one quad) with a crossing."""
        data = np.array([[0., 1.], [1., 2.]], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.5])
        assert len(result) >= 1

    def test_level_at_exact_value(self):
        """Level exactly matching cell values still works."""
        data = np.array([
            [0., 1., 2.],
            [1., 2., 3.],
            [2., 3., 4.],
        ], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[2.0])
        # Should not crash; some segments may be degenerate.
        assert isinstance(result, list)

    def test_auto_levels(self):
        """When levels is None, n_levels evenly spaced levels are chosen."""
        data = _make_ramp(ny=5, nx=10)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, n_levels=5)
        assert len(result) > 0
        levels_found = sorted(set(lvl for lvl, _ in result))
        assert len(levels_found) <= 5

    def test_invalid_ndim(self):
        """3D input raises ValueError."""
        data = np.ones((3, 3, 3), dtype=np.float64)
        agg = xr.DataArray(data, dims=['z', 'y', 'x'])
        with pytest.raises(ValueError, match="2D"):
            contours(agg, levels=[0.5])

    def test_too_small(self):
        """1-row raster raises ValueError."""
        data = np.ones((1, 5), dtype=np.float64)
        agg = xr.DataArray(data, dims=['y', 'x'])
        with pytest.raises(ValueError, match="at least 2"):
            contours(agg, levels=[0.5])


# ---------------------------------------------------------------------------
# Segment stitching
# ---------------------------------------------------------------------------

class TestStitching:

    def test_empty_segments(self):
        """No segments produces no lines."""
        seg_r = np.empty((0, 2), dtype=np.float64)
        seg_c = np.empty((0, 2), dtype=np.float64)
        lines = _stitch_segments(seg_r, seg_c, 0)
        assert lines == []

    def test_single_segment(self):
        """One segment produces one line with two points."""
        seg_r = np.array([[0.0, 1.0]], dtype=np.float64)
        seg_c = np.array([[0.5, 0.5]], dtype=np.float64)
        lines = _stitch_segments(seg_r, seg_c, 1)
        assert len(lines) == 1
        assert lines[0].shape == (2, 2)

    def test_connected_segments(self):
        """Three connected segments produce one line with four points."""
        seg_r = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
        ], dtype=np.float64)
        seg_c = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float64)
        lines = _stitch_segments(seg_r, seg_c, 3)
        assert len(lines) == 1
        assert lines[0].shape[0] == 4


# ---------------------------------------------------------------------------
# Backend equivalence
# ---------------------------------------------------------------------------

class TestBackendEquivalence:

    def _numpy_result(self, data, levels):
        agg = create_test_raster(data, backend='numpy')
        return contours(agg, levels=levels)

    def _collect_segments(self, results):
        """Collect all individual segments as a sorted set for comparison.

        Each polyline is decomposed into its constituent segments, each
        segment is canonicalized (smaller endpoint first), and the full
        set is sorted for stable comparison across backends.
        """
        from collections import defaultdict
        by_level = defaultdict(list)
        DECIMALS = 8
        for level, coords in results:
            for i in range(len(coords) - 1):
                p0 = (round(coords[i, 0], DECIMALS),
                      round(coords[i, 1], DECIMALS))
                p1 = (round(coords[i + 1, 0], DECIMALS),
                      round(coords[i + 1, 1], DECIMALS))
                seg = (min(p0, p1), max(p0, p1))
                by_level[level].append(seg)
        # Sort segments within each level for stable comparison.
        return {lvl: sorted(segs) for lvl, segs in by_level.items()}

    @dask_array_available
    def test_numpy_equals_dask(self, elevation_raster_no_nans):
        data = elevation_raster_no_nans
        levels = [300.0, 500.0, 700.0]
        numpy_agg = create_test_raster(data, backend='numpy')
        dask_agg = create_test_raster(data, backend='dask+numpy', chunks=(4, 3))

        np_result = contours(numpy_agg, levels=levels)
        dk_result = contours(dask_agg, levels=levels)

        np_segs = self._collect_segments(np_result)
        dk_segs = self._collect_segments(dk_result)

        assert set(np_segs.keys()) == set(dk_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == dk_segs[lvl], (
                f"Segment mismatch at level {lvl}")

    @cuda_and_cupy_available
    def test_numpy_equals_cupy(self, elevation_raster_no_nans):
        data = elevation_raster_no_nans
        levels = [300.0, 500.0, 700.0]
        numpy_agg = create_test_raster(data, backend='numpy')
        cupy_agg = create_test_raster(data, backend='cupy')

        np_result = contours(numpy_agg, levels=levels)
        cp_result = contours(cupy_agg, levels=levels)

        np_segs = self._collect_segments(np_result)
        cp_segs = self._collect_segments(cp_result)

        assert set(np_segs.keys()) == set(cp_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == cp_segs[lvl], (
                f"Segment mismatch at level {lvl}")

    @dask_array_available
    @cuda_and_cupy_available
    def test_numpy_equals_dask_cupy(self, elevation_raster_no_nans):
        data = elevation_raster_no_nans
        levels = [300.0, 500.0, 700.0]
        numpy_agg = create_test_raster(data, backend='numpy')
        dask_cupy_agg = create_test_raster(data, backend='dask+cupy',
                                           chunks=(4, 3))

        np_result = contours(numpy_agg, levels=levels)
        dc_result = contours(dask_cupy_agg, levels=levels)

        np_segs = self._collect_segments(np_result)
        dc_segs = self._collect_segments(dc_result)

        assert set(np_segs.keys()) == set(dc_segs.keys())
        for lvl in np_segs:
            assert np_segs[lvl] == dc_segs[lvl], (
                f"Segment mismatch at level {lvl}")


# ---------------------------------------------------------------------------
# Return type: GeoDataFrame
# ---------------------------------------------------------------------------

class TestGeoDataFrame:

    def test_geopandas_return(self):
        pytest.importorskip("geopandas")
        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')
        gdf = contours(agg, levels=[0.5, 1.5], return_type="geopandas")

        import geopandas as gpd
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert 'level' in gdf.columns
        assert 'geometry' in gdf.columns
        assert len(gdf) > 0

    def test_invalid_return_type(self):
        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')
        with pytest.raises(ValueError, match="Invalid return_type"):
            contours(agg, levels=[0.5], return_type="invalid")


# ---------------------------------------------------------------------------
# Accessor integration
# ---------------------------------------------------------------------------

class TestAccessor:

    def test_dataarray_accessor(self):
        import xrspatial  # noqa: F401 -- registers accessors
        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')
        result = agg.xrs.contours(levels=[0.5])
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Reference validation against skimage.measure.find_contours
# ---------------------------------------------------------------------------

class TestReferenceValidation:

    def test_matches_skimage(self):
        """Compare our contours against skimage as a reference."""
        skimage_measure = pytest.importorskip("skimage.measure")

        data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 1., 2., 1., 0.],
            [0., 2., 4., 2., 0.],
            [0., 1., 2., 1., 0.],
            [0., 0., 0., 0., 0.],
        ], dtype=np.float64)

        level = 1.5
        agg = create_test_raster(data, backend='numpy')
        our_result = contours(agg, levels=[level])

        sk_contours = skimage_measure.find_contours(data, level)

        # Both should find contour lines at this level.
        assert len(our_result) > 0
        assert len(sk_contours) > 0

        # Transform our coordinate-space points back to array indices for
        # comparison with skimage (which returns array index coordinates).
        y_coords = agg.coords[agg.dims[0]].values
        x_coords = agg.coords[agg.dims[1]].values
        y_idx = np.arange(len(y_coords), dtype=np.float64)
        x_idx = np.arange(len(x_coords), dtype=np.float64)

        our_points_idx = []
        for _, coords in our_result:
            pts = np.empty_like(coords)
            pts[:, 0] = np.interp(coords[:, 0], y_coords[::-1], y_idx[::-1]) \
                if y_coords[0] > y_coords[-1] \
                else np.interp(coords[:, 0], y_coords, y_idx)
            pts[:, 1] = np.interp(coords[:, 1], x_coords, x_idx)
            our_points_idx.append(pts)
        our_points = np.vstack(our_points_idx)

        # Every skimage contour point should be close to some point of ours.
        for sk_line in sk_contours:
            for pt in sk_line:
                dists = np.linalg.norm(our_points - pt, axis=1)
                assert np.min(dists) < 0.5, (
                    f"skimage point {pt} not near any of our contour points"
                )
