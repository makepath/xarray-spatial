import numpy as np
import pytest
import xarray as xr

from xrspatial.tests.general_checks import cuda_and_cupy_available
from xrspatial.visibility import _bresenham_line, _extract_transect


class TestBresenhamLine:
    def test_horizontal(self):
        cells = _bresenham_line(0, 0, 0, 4)
        expected = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]])
        np.testing.assert_array_equal(cells, expected)

    def test_vertical(self):
        cells = _bresenham_line(0, 0, 4, 0)
        expected = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        np.testing.assert_array_equal(cells, expected)

    def test_diagonal(self):
        cells = _bresenham_line(0, 0, 3, 3)
        expected = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        np.testing.assert_array_equal(cells, expected)

    def test_single_cell(self):
        cells = _bresenham_line(2, 3, 2, 3)
        expected = np.array([[2, 3]])
        np.testing.assert_array_equal(cells, expected)

    def test_steep_negative(self):
        cells = _bresenham_line(4, 2, 0, 0)
        assert tuple(cells[0]) == (4, 2)
        assert tuple(cells[-1]) == (0, 0)
        assert len(cells) == 5

    def test_includes_endpoints(self):
        cells = _bresenham_line(1, 1, 5, 8)
        assert tuple(cells[0]) == (1, 1)
        assert tuple(cells[-1]) == (5, 8)


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
        cells = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
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
        cells = np.array([[0, 0], [2, 3], [4, 4]])
        elev_np, _, _ = _extract_transect(raster_np, cells)
        elev_da, _, _ = _extract_transect(raster_dask, cells)
        np.testing.assert_array_equal(elev_np, elev_da)

    @cuda_and_cupy_available
    def test_cupy_matches_numpy(self):
        import cupy
        data = np.arange(25, dtype=float).reshape(5, 5)
        raster_np = _make_raster(data)
        raster_cp = raster_np.copy()
        raster_cp.data = cupy.asarray(data)
        cells = np.array([[0, 0], [2, 3], [4, 4]])
        elev_np, _, _ = _extract_transect(raster_np, cells)
        elev_cp, _, _ = _extract_transect(raster_cp, cells)
        # cupy path pulls to numpy via .get(); result is a plain numpy array
        assert isinstance(elev_cp, np.ndarray)
        np.testing.assert_array_equal(elev_np, elev_cp)


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

    def test_nan_terrain_cell(self):
        """A NaN terrain cell drops out of the visibility sweep without
        raising; its elevation is carried through as NaN and downstream
        cells stay visible."""
        data = np.zeros((1, 10), dtype=float)
        data[0, 5] = np.nan
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=9, y1=0,
                               observer_elev=5)
        elev = result['elevation'].values
        vis = result['visible'].values
        # the NaN cell is carried through to the elevation profile
        assert np.isnan(elev[5])
        # the NaN cell itself is not counted as visible
        assert not vis[5]
        # a NaN cell does not poison the running max-angle: cells past it
        # remain visible on otherwise-flat terrain
        assert vis[6]
        assert vis[-1]

    def test_fresnel_blocked_by_obstruction(self):
        """When terrain intrudes into the first Fresnel zone, fresnel_clear
        is False at the affected samples (the non-default branch).

        Uses a long path and a low frequency so the first Fresnel zone is
        wide (~16 m at midpoint), then puts a ridge 1 m below the LOS so
        it sits inside the zone without blocking line of sight itself.
        """
        width = 101
        mid = width // 2
        data = np.zeros((1, width), dtype=float)
        data[0, mid] = 49.0  # 1 m below the flat 50 m LOS
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=width - 1, y1=0,
                               observer_elev=50, target_elev=50,
                               frequency_mhz=30)
        clear = result['fresnel_clear'].values
        fr = result['fresnel_radius'].values
        clearance = result['los_height'].values - result['elevation'].values
        # at the ridge the clearance is less than the Fresnel radius
        assert clearance[mid] < fr[mid]
        # so the Fresnel zone is reported blocked there
        assert not clear[mid]
        # endpoints (zero Fresnel radius) stay clear
        assert clear[0]
        assert clear[-1]

    @cuda_and_cupy_available
    def test_cupy_raster_matches_numpy(self):
        import cupy
        data = np.zeros((1, 10), dtype=float)
        data[0, 5] = 100.0
        raster_np = _make_raster(data)
        raster_cp = raster_np.copy()
        raster_cp.data = cupy.asarray(data)
        res_np = line_of_sight(raster_np, x0=0, y0=0, x1=9, y1=0,
                               observer_elev=1, target_elev=0)
        res_cp = line_of_sight(raster_cp, x0=0, y0=0, x1=9, y1=0,
                               observer_elev=1, target_elev=0)
        np.testing.assert_array_equal(res_np['elevation'].values,
                                      res_cp['elevation'].values)
        np.testing.assert_array_equal(res_np['visible'].values,
                                      res_cp['visible'].values)
        np.testing.assert_allclose(res_np['distance'].values,
                                   res_cp['distance'].values)


import dask.array as da
from xrspatial.visibility import cumulative_viewshed


class TestCumulativeViewshed:
    def test_flat_terrain_all_visible(self):
        """On flat terrain with elevated observers, every cell is visible."""
        data = np.zeros((10, 10), dtype=float)
        raster = _make_raster(data)
        observers = [
            {'x': 2.0, 'y': 2.0, 'observer_elev': 10},
            {'x': 7.0, 'y': 7.0, 'observer_elev': 10},
        ]
        result = cumulative_viewshed(raster, observers)
        assert result.dtype == np.int32
        # every cell should be seen by both observers
        assert (result.values == 2).all()

    def test_single_observer_matches_viewshed(self):
        """Single-observer cumulative should match binary viewshed."""
        from xrspatial import viewshed
        from xrspatial.viewshed import INVISIBLE
        data = np.random.RandomState(42).rand(15, 15).astype(float) * 100
        raster = _make_raster(data)
        obs = {'x': 7.0, 'y': 7.0, 'observer_elev': 50}
        result = cumulative_viewshed(raster, [obs])
        vs = viewshed(raster, x=7.0, y=7.0, observer_elev=50)
        expected = (vs.values != INVISIBLE).astype(np.int32)
        np.testing.assert_array_equal(result.values, expected)

    def test_wall_blocks_one_side(self):
        """A tall wall blocks visibility from the other side."""
        data = np.zeros((5, 11), dtype=float)
        data[:, 5] = 1000  # tall wall across all rows
        raster = _make_raster(data)
        obs_left = {'x': 0.0, 'y': 2.0, 'observer_elev': 1}
        obs_right = {'x': 10.0, 'y': 2.0, 'observer_elev': 1}
        result = cumulative_viewshed(raster, [obs_left, obs_right])
        # the wall cell itself is visible to both
        assert result.values[2, 5] == 2
        # cells far from wall visible to at least one observer
        assert result.values[2, 0] >= 1
        assert result.values[2, 10] >= 1

    def test_per_observer_max_distance(self):
        """Per-observer max_distance limits the analysis radius."""
        data = np.zeros((20, 20), dtype=float)
        raster = _make_raster(data)
        obs = {'x': 10.0, 'y': 10.0, 'observer_elev': 10, 'max_distance': 3}
        result = cumulative_viewshed(raster, [obs])
        # corners should be 0 (beyond max_distance)
        assert result.values[0, 0] == 0
        assert result.values[19, 19] == 0
        # center should be 1
        assert result.values[10, 10] == 1

    def test_empty_observers_raises(self):
        data = np.zeros((5, 5), dtype=float)
        raster = _make_raster(data)
        with pytest.raises(ValueError):
            cumulative_viewshed(raster, [])

    def test_dask_matches_numpy(self):
        """Dask backend should produce the same result as numpy."""
        data = np.random.RandomState(99).rand(15, 15).astype(float) * 50
        raster_np = _make_raster(data)
        raster_dask = raster_np.copy()
        raster_dask.data = da.from_array(data, chunks=(8, 8))
        observers = [
            {'x': 3.0, 'y': 3.0, 'observer_elev': 30},
            {'x': 12.0, 'y': 12.0, 'observer_elev': 30},
        ]
        result_np = cumulative_viewshed(raster_np, observers)
        result_dask = cumulative_viewshed(raster_dask, observers)
        np.testing.assert_array_equal(result_np.values, result_dask.values)

    def test_dask_source_computed_once(self):
        """No-max_distance dask path computes the source once, not per observer."""
        H = W = 16
        base = np.zeros((H, W), dtype=float)
        counter = {'n': 0}

        def _src(block_info=None):
            counter['n'] += 1
            return base.copy()

        source = da.map_blocks(_src, chunks=((H,), (W,)), dtype=float,
                               meta=np.array(()))
        raster = xr.DataArray(
            source, dims=['y', 'x'],
            coords={'y': np.arange(H, dtype=float),
                    'x': np.arange(W, dtype=float)},
        )
        observers = [{'x': float(i), 'y': float(i), 'observer_elev': 5}
                     for i in range(4)]
        counter['n'] = 0
        result = cumulative_viewshed(raster, observers)
        # source materialised exactly once despite four observers
        assert counter['n'] == 1
        # output stays dask-backed to match the dask input
        assert isinstance(result.data, da.Array)

    def test_dask_per_observer_max_distance_stays_lazy(self):
        """A per-observer max_distance keeps the dask windowing path."""
        data = np.zeros((20, 20), dtype=float)
        raster_np = _make_raster(data)
        raster_dask = raster_np.copy()
        raster_dask.data = da.from_array(data, chunks=(8, 8))
        observers = [{'x': 10.0, 'y': 10.0, 'observer_elev': 10,
                      'max_distance': 3}]
        result = cumulative_viewshed(raster_dask, observers)
        assert isinstance(result.data, da.Array)
        result_np = cumulative_viewshed(raster_np, observers)
        np.testing.assert_array_equal(result.values, result_np.values)

    def test_default_output_name(self):
        data = np.zeros((5, 5), dtype=float)
        raster = _make_raster(data)
        observers = [{'x': 2.0, 'y': 2.0, 'observer_elev': 10}]
        result = cumulative_viewshed(raster, observers)
        assert result.name == 'cumulative_viewshed'

    def test_custom_output_name(self):
        data = np.zeros((5, 5), dtype=float)
        raster = _make_raster(data)
        observers = [{'x': 2.0, 'y': 2.0, 'observer_elev': 10}]
        result = cumulative_viewshed(raster, observers, name='count')
        assert result.name == 'count'

    def test_preserves_coords_and_dims(self):
        data = np.zeros((5, 5), dtype=float)
        raster = _make_raster(data)
        raster.attrs['crs'] = 'EPSG:4326'
        observers = [{'x': 2.0, 'y': 2.0, 'observer_elev': 10}]
        result = cumulative_viewshed(raster, observers)
        assert result.dims == raster.dims
        np.testing.assert_array_equal(result.coords['x'].values,
                                      raster.coords['x'].values)
        np.testing.assert_array_equal(result.coords['y'].values,
                                      raster.coords['y'].values)
        assert result.attrs.get('crs') == 'EPSG:4326'


from xrspatial.visibility import visibility_frequency


class TestVisibilityFrequency:
    def test_flat_terrain_all_ones(self):
        data = np.zeros((10, 10), dtype=float)
        raster = _make_raster(data)
        observers = [
            {'x': 2.0, 'y': 2.0, 'observer_elev': 10},
            {'x': 7.0, 'y': 7.0, 'observer_elev': 10},
        ]
        result = visibility_frequency(raster, observers)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result.values, 1.0)

    def test_equals_cumulative_divided_by_n(self):
        data = np.random.RandomState(7).rand(15, 15).astype(float) * 100
        raster = _make_raster(data)
        observers = [
            {'x': 3.0, 'y': 3.0, 'observer_elev': 50},
            {'x': 10.0, 'y': 10.0, 'observer_elev': 50},
            {'x': 7.0, 'y': 2.0, 'observer_elev': 50},
        ]
        freq = visibility_frequency(raster, observers)
        cum = cumulative_viewshed(raster, observers)
        expected = cum.values.astype(np.float64) / 3.0
        np.testing.assert_allclose(freq.values, expected)

    def test_default_output_name(self):
        data = np.zeros((5, 5), dtype=float)
        raster = _make_raster(data)
        observers = [{'x': 2.0, 'y': 2.0, 'observer_elev': 10}]
        result = visibility_frequency(raster, observers)
        assert result.name == 'visibility_frequency'


from xrspatial import viewshed
from xrspatial.gpu_rtx import has_rtx
from xrspatial.utils import has_cuda_and_cupy
from xrspatial.viewshed import INVISIBLE

cupy_skip = pytest.mark.skipif(
    not (has_cuda_and_cupy() and has_rtx()),
    reason="cupy / rtxpy not available",
)


@cupy_skip
class TestCupyBackend:
    """cupy backend must return a cupy-backed DataArray with the same
    coords, dims, and attrs as the numpy backend (issue #3193)."""

    def _cupy_raster(self):
        import cupy as cp
        data = np.random.RandomState(1).rand(20, 20).astype(float) * 100
        raster = _make_raster(data)
        raster.attrs['crs'] = 'EPSG:4326'
        raster.data = cp.asarray(raster.data)
        return raster

    def test_cumulative_returns_cupy_with_metadata(self):
        import cupy as cp
        raster = self._cupy_raster()
        observers = [
            {'x': 5.0, 'y': 5.0, 'observer_elev': 50},
            {'x': 12.0, 'y': 12.0, 'observer_elev': 50},
        ]
        result = cumulative_viewshed(raster, observers)
        assert isinstance(result.data, cp.ndarray)
        assert result.dtype == np.int32
        assert result.dims == raster.dims
        np.testing.assert_array_equal(result.coords['x'].values,
                                      raster.coords['x'].values)
        np.testing.assert_array_equal(result.coords['y'].values,
                                      raster.coords['y'].values)
        assert result.attrs.get('crs') == 'EPSG:4326'

    def test_cumulative_matches_single_viewshed(self):
        import cupy as cp
        raster = self._cupy_raster()
        obs = {'x': 5.0, 'y': 5.0, 'observer_elev': 50}
        result = cumulative_viewshed(raster, [obs])
        vs = viewshed(raster, x=5.0, y=5.0, observer_elev=50)
        expected = (cp.asnumpy(vs.data) != INVISIBLE).astype(np.int32)
        np.testing.assert_array_equal(cp.asnumpy(result.data), expected)

    def test_frequency_returns_cupy_with_metadata(self):
        import cupy as cp
        raster = self._cupy_raster()
        observers = [
            {'x': 5.0, 'y': 5.0, 'observer_elev': 50},
            {'x': 12.0, 'y': 12.0, 'observer_elev': 50},
        ]
        result = visibility_frequency(raster, observers)
        assert isinstance(result.data, cp.ndarray)
        assert result.dtype == np.float64
        assert result.dims == raster.dims
        assert result.attrs.get('crs') == 'EPSG:4326'
