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

    def test_complex_dtype_rejected(self):
        """Complex input raises instead of silently dropping the imaginary part."""
        data = np.ones((3, 3), dtype=np.complex128)
        agg = xr.DataArray(data, dims=['y', 'x'])
        with pytest.raises(ValueError, match="real numeric"):
            contours(agg, levels=[0.5])

    def test_non_dataarray_rejected(self):
        """A plain ndarray raises a clear TypeError, not a late dispatch error."""
        data = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(TypeError, match="xarray.DataArray"):
            contours(data, levels=[0.5])


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

    def test_geopandas_propagates_crs(self):
        """A populated geopandas result carries the input raster's CRS."""
        pytest.importorskip("geopandas")
        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')  # attrs include a crs
        gdf = contours(agg, levels=[1.5], return_type="geopandas")
        assert len(gdf) > 0
        assert gdf.crs == agg.attrs['crs']

    def test_geopandas_empty_result_keeps_crs(self):
        """Levels with no crossings return an empty GeoDataFrame with the CRS.

        Regression for #2700: gpd.GeoDataFrame(records, crs=crs) raised
        ValueError when records was empty and crs was not None.
        """
        pytest.importorskip("geopandas")
        import geopandas as gpd
        data = np.ones((4, 4), dtype=np.float64)  # flat -> no crossings
        agg = create_test_raster(data, backend='numpy')  # attrs include a crs
        gdf = contours(agg, levels=[5.0], return_type="geopandas")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0
        assert 'level' in gdf.columns
        assert 'geometry' in gdf.columns
        assert gdf.crs == agg.attrs['crs']

    def test_geopandas_all_nan_keeps_crs(self):
        """All-NaN input with auto levels keeps the CRS on the empty frame.

        Regression for #2700: the all-NaN early-return path dropped the CRS.
        """
        pytest.importorskip("geopandas")
        import geopandas as gpd
        data = np.full((4, 4), np.nan, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')  # attrs include a crs
        gdf = contours(agg, return_type="geopandas")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0
        assert gdf.crs == agg.attrs['crs']

    def test_geopandas_empty_result_no_crs(self):
        """An empty result with no input CRS returns an empty frame, no crash."""
        pytest.importorskip("geopandas")
        import geopandas as gpd
        data = np.ones((4, 4), dtype=np.float64)
        agg = xr.DataArray(
            data, dims=['y', 'x'],
            coords={'y': np.linspace(2, 0, 4), 'x': np.linspace(0, 2, 4)},
        )
        gdf = contours(agg, levels=[5.0], return_type="geopandas")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0
        assert gdf.crs is None

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
# Memory guard (#1240)
# ---------------------------------------------------------------------------


class TestMemoryGuard:

    def test_rejects_oversize_numpy(self, monkeypatch):
        """contours() raises MemoryError when segment buffers exceed budget."""
        import xrspatial.contour as contour_mod

        # Force available memory to 1 MB so even a small raster trips the
        # guard.  A 1000x1000 raster needs ~32 MB per level's buffers.
        monkeypatch.setattr(
            contour_mod, '_available_memory_bytes', lambda: 1 * 1024 * 1024
        )

        data = np.zeros((1000, 1000), dtype=np.float64)
        agg = xr.DataArray(data, dims=['y', 'x'])

        with pytest.raises(MemoryError, match="segment buffers per level"):
            contours(agg, levels=[0.5])

    def test_allows_within_budget(self, monkeypatch):
        """A small raster stays under the guard and returns normally."""
        import xrspatial.contour as contour_mod

        monkeypatch.setattr(
            contour_mod, '_available_memory_bytes', lambda: 8 * 1024 ** 3
        )

        data = _make_peak()
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.5])
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


# ---------------------------------------------------------------------------
# Infinity handling (#2704)
# ---------------------------------------------------------------------------

class TestInfHandling:

    def _inf_peak(self, value):
        """5x5 raster of ones with a single +/-inf cell in the center."""
        data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., value, 1., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.],
        ], dtype=np.float64)
        return data

    def test_inf_far_level_no_crossing(self):
        """A level the inf cell sits above on all sides still traces the
        surrounding ring without touching the inf quad's interpolation."""
        data = self._inf_peak(np.inf)
        agg = create_test_raster(data, backend='numpy')
        # Level 0.5 crosses the outer 0/1 boundary.  The four quads that
        # touch the inf cell have all corners >= 0.5 (1 and inf), so they
        # are the all-above case (idx 15) and are skipped before any
        # interpolation runs.
        result = contours(agg, levels=[0.5])
        assert len(result) >= 1
        for level, coords in result:
            assert np.isfinite(coords).all(), (
                "level 0.5 ring should not include the inf quad")

    def test_inf_corner_no_nan_coords(self):
        """A finite level near a +inf cell must not leak NaN coordinates.

        Regression for #2704: the NaN-skip guard in the kernel used ``x != x``
        which does not catch infinity; fixed by using ``np.isfinite``.
        """
        data = self._inf_peak(np.inf)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.5])
        for level, coords in result:
            assert np.isfinite(coords).all(), (
                f"non-finite coordinate in contour at level {level}: {coords}")

    def test_neg_inf_corner_no_nan_coords(self):
        """A finite level near a -inf cell must not leak NaN coordinates.

        Regression for #2704: same fix as test_inf_corner_no_nan_coords.
        """
        data = self._inf_peak(-np.inf)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.5])
        for level, coords in result:
            assert np.isfinite(coords).all(), (
                f"non-finite coordinate in contour at level {level}: {coords}")

    def test_mixed_inf(self):
        """Multiple infinities of opposite signs must not produce NaN."""
        data = np.array([
            [0., 0., 0., 0., 0.],
            [0., np.inf, 1., -np.inf, 0.],
            [0., 1., 1., 1., 0.],
            [0., -np.inf, 1., np.inf, 0.],
            [0., 0., 0., 0., 0.],
        ], dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[0.5])
        for level, coords in result:
            assert np.isfinite(coords).all(), \
                f"Non-finite coordinates found at level {level}"

    def test_all_inf_quad(self):
        """A 2x2 raster with all corners infinite produces no contours."""
        data = np.full((2, 2), np.inf, dtype=np.float64)
        agg = create_test_raster(data, backend='numpy')
        result = contours(agg, levels=[1.0])
        assert result == []


# ---------------------------------------------------------------------------
# CRS propagation to GeoDataFrame output (#2704 audit, Cat 5)
# ---------------------------------------------------------------------------

class TestCRSPropagation:

    def test_geopandas_crs_from_attrs(self):
        """return_type='geopandas' copies agg.attrs['crs'] onto the gdf."""
        pytest.importorskip("geopandas")
        data = _make_peak()
        # create_test_raster sets attrs={'res': ..., 'crs': 'EPSG: 5070'}.
        agg = create_test_raster(data, backend='numpy')
        gdf = contours(agg, levels=[1.5], return_type="geopandas")
        # GeoPandas normalizes the CRS; check it resolves to EPSG:5070.
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 5070

    def test_geopandas_no_crs_attr(self):
        """A raster with no crs attr yields a GeoDataFrame with crs None."""
        gpd = pytest.importorskip("geopandas")
        data = _make_peak()
        agg = xr.DataArray(data, dims=['y', 'x'])
        agg['y'] = np.linspace(2.0, 0.0, data.shape[0])
        agg['x'] = np.linspace(0.0, 2.0, data.shape[1])
        gdf = contours(agg, levels=[1.5], return_type="geopandas")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs is None


# ---------------------------------------------------------------------------
# Non-default dim names: index -> coordinate transform (#2704 audit, Cat 5)
# ---------------------------------------------------------------------------

class TestNonDefaultDims:

    def test_lat_lon_dims_coordinate_transform(self):
        """Output coordinates map into the lat/lon coordinate space.

        contours() reads agg.dims[0]/[1] coords to convert array indices
        to coordinate values, so non-y/x dim names must still work and the
        coordinates must land inside the lat/lon ranges.
        """
        data = _make_peak()
        ny, nx = data.shape
        lat = np.linspace(40.0, 30.0, ny)   # decreasing, like a north-up DEM
        lon = np.linspace(-100.0, -90.0, nx)
        agg = xr.DataArray(data, dims=['lat', 'lon'])
        agg['lat'] = lat
        agg['lon'] = lon

        result = contours(agg, levels=[1.5])
        assert len(result) >= 1
        for level, coords in result:
            # coords are (lat, lon) in the input coordinate space.
            assert np.all(coords[:, 0] >= lat.min() - 1e-9)
            assert np.all(coords[:, 0] <= lat.max() + 1e-9)
            assert np.all(coords[:, 1] >= lon.min() - 1e-9)
            assert np.all(coords[:, 1] <= lon.max() + 1e-9)

    def test_lat_lon_matches_yx_equivalent(self):
        """Renaming y/x to lat/lon (same coord values) gives same output."""
        data = _make_peak()
        ny, nx = data.shape
        y = np.linspace(2.0, 0.0, ny)
        x = np.linspace(0.0, 2.5, nx)

        yx = xr.DataArray(data, dims=['y', 'x'])
        yx['y'] = y
        yx['x'] = x

        ll = xr.DataArray(data, dims=['lat', 'lon'])
        ll['lat'] = y
        ll['lon'] = x

        r_yx = contours(yx, levels=[1.5])
        r_ll = contours(ll, levels=[1.5])

        assert len(r_yx) == len(r_ll)
        for (lvl_a, c_a), (lvl_b, c_b) in zip(r_yx, r_ll):
            assert lvl_a == lvl_b
            np.testing.assert_allclose(c_a, c_b)
