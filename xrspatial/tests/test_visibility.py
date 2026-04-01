import numpy as np
import pytest
import xarray as xr

from xrspatial.visibility import _bresenham_line, _extract_transect


class TestBresenhamLine:
    def test_horizontal(self):
        cells = _bresenham_line(0, 0, 0, 4)
        assert cells == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    def test_vertical(self):
        cells = _bresenham_line(0, 0, 4, 0)
        assert cells == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]

    def test_diagonal(self):
        cells = _bresenham_line(0, 0, 3, 3)
        assert cells == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_single_cell(self):
        cells = _bresenham_line(2, 3, 2, 3)
        assert cells == [(2, 3)]

    def test_steep_negative(self):
        cells = _bresenham_line(4, 2, 0, 0)
        # Must start at (4, 2) and end at (0, 0)
        assert cells[0] == (4, 2)
        assert cells[-1] == (0, 0)
        assert len(cells) == 5

    def test_includes_endpoints(self):
        cells = _bresenham_line(1, 1, 5, 8)
        assert cells[0] == (1, 1)
        assert cells[-1] == (5, 8)


def _make_raster(data):
    """Module-level helper for creating test rasters."""
    h, w = data.shape
    return xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'y': np.arange(h, dtype=float),
                'x': np.arange(w, dtype=float)},
    )


class TestExtractTransect:
    def test_numpy_diagonal(self):
        data = np.arange(25, dtype=float).reshape(5, 5)
        raster = _make_raster(data)
        cells = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        elev, xs, ys = _extract_transect(raster, cells)
        np.testing.assert_array_equal(elev, [0, 6, 12, 18, 24])
        np.testing.assert_array_equal(xs, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(ys, [0, 1, 2, 3, 4])

    def test_dask_matches_numpy(self):
        import dask.array as da
        data = np.arange(25, dtype=float).reshape(5, 5)
        raster_np = _make_raster(data)
        raster_dask = raster_np.copy()
        raster_dask.data = da.from_array(data, chunks=(3, 3))
        cells = [(0, 0), (2, 3), (4, 4)]
        elev_np, _, _ = _extract_transect(raster_np, cells)
        elev_da, _, _ = _extract_transect(raster_dask, cells)
        np.testing.assert_array_equal(elev_np, elev_da)


from xrspatial.visibility import line_of_sight


class TestLineOfSight:
    def test_flat_terrain_all_visible(self):
        data = np.zeros((5, 10), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=2, x1=9, y1=2,
                               observer_elev=10, target_elev=10)
        assert isinstance(result, xr.Dataset)
        assert 'visible' in result
        assert 'elevation' in result
        assert 'los_height' in result
        assert 'distance' in result
        assert result['visible'].all()

    def test_obstruction_blocks_view(self):
        data = np.zeros((1, 10), dtype=float)
        data[0, 5] = 100  # tall wall in the middle
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=9, y1=0,
                               observer_elev=1, target_elev=0)
        vis = result['visible'].values
        # observer cell is visible
        assert vis[0]
        # cells before the wall are visible
        assert all(vis[:6])
        # at least some cells after the wall are blocked
        assert not all(vis[6:])

    def test_observer_equals_target(self):
        data = np.ones((5, 5), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=2, y0=2, x1=2, y1=2)
        assert len(result['sample']) == 1
        assert result['visible'].values[0]

    def test_elevation_offsets(self):
        data = np.zeros((1, 5), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=4, y1=0,
                               observer_elev=10, target_elev=20)
        los = result['los_height'].values
        # LOS starts at 10, ends at 20
        assert abs(los[0] - 10.0) < 1e-10
        assert abs(los[-1] - 20.0) < 1e-10

    def test_distance_monotonic(self):
        data = np.zeros((5, 10), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=9, y1=4)
        d = result['distance'].values
        assert all(d[i] <= d[i + 1] for i in range(len(d) - 1))

    def test_fresnel_zone(self):
        data = np.zeros((1, 11), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=10, y1=0,
                               observer_elev=50, target_elev=50,
                               frequency_mhz=900)
        assert 'fresnel_radius' in result
        assert 'fresnel_clear' in result
        # midpoint has largest Fresnel radius
        fr = result['fresnel_radius'].values
        mid = len(fr) // 2
        assert fr[mid] >= fr[1]
        assert fr[mid] >= fr[-2]
        # with 50m clearance and flat terrain, Fresnel should be clear
        assert result['fresnel_clear'].all()

    def test_no_fresnel_by_default(self):
        data = np.zeros((5, 5), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=4, y1=4)
        assert 'fresnel_radius' not in result
        assert 'fresnel_clear' not in result

    def test_xy_coords_in_output(self):
        data = np.zeros((5, 10), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=2, x1=9, y1=2)
        # first point should match observer
        assert abs(result['x'].values[0] - 0.0) < 1e-10
        assert abs(result['y'].values[0] - 2.0) < 1e-10
        # last point should match target
        assert abs(result['x'].values[-1] - 9.0) < 1e-10
        assert abs(result['y'].values[-1] - 2.0) < 1e-10
