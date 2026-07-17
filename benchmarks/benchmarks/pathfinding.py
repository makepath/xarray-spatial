import numpy as np
import xarray as xr

from xrspatial.pathfinding import a_star_search, multi_stop_search

from .common import get_xr_dataarray


class MultiStopSearchDaskMemory:
    """Memory contract of the dask multi-stop path (issue #3660).

    Peak memory must scale with the chunk size and the explored
    corridor, not the full grid.  The grid is large (128 MB) but the
    waypoints are close together, so a regression back to eager
    full-grid stitching shows up as a step change in peak memory while
    the benchmark itself stays fast.
    """

    def setup(self):
        n = 4000  # 4000 x 4000 float64 = 128 MB, chunks 500 x 500 = 2 MB
        import dask.array as da
        data = da.ones((n, n), chunks=(500, 500), dtype='float64')
        self.agg = xr.DataArray(
            data, dims=['y', 'x'], attrs={'res': (1.0, 1.0)})
        self.agg['y'] = np.linspace(n - 1, 0, n)
        self.agg['x'] = np.linspace(0, n - 1, n)
        # 3 waypoints inside a 200-pixel corner neighbourhood
        self.waypoints = [
            (float(n - 1), 0.0),
            (float(n - 101), 100.0),
            (float(n - 201), 200.0),
        ]

    def peakmem_multi_stop_search(self):
        multi_stop_search(self.agg, self.waypoints)


class AStarSearch:
    params = ([100, 300, 1000], [4, 8], ["numpy", "cupy", "dask"])
    param_names = ("nx", "connectivity", "type")

    def setup(self, nx, connectivity, type):
        if type == "dask" and nx > 300:
            # The dask backend is a pure-Python sparse A* that loads
            # chunks on demand; at nx=1000 a single call takes ~4 s,
            # which would dominate the suite's runtime.
            raise NotImplementedError()
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type)
        self.start = self.agg.y[0], self.agg.x[0]
        self.goal = self.agg.y[-1], self.agg.x[-1]
        # snap_start/snap_goal raise on dask-backed arrays by design
        self.snap = type != "dask"

    def time_a_star_search(self, nx, connectivity, type):
        a_star_search(
            self.agg, self.start, self.goal,
            connectivity=connectivity,
            snap_start=self.snap, snap_goal=self.snap
        )


class MultiStopSearch:
    params = ([100, 300], ["numpy", "cupy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type)
        ys = self.agg.y.data
        xs = self.agg.x.data
        # 4 waypoints zigzagging across the grid (3 segments)
        self.waypoints = [
            (ys[0], xs[0]),
            (ys[-1], xs[nx // 3]),
            (ys[0], xs[2 * nx // 3]),
            (ys[-1], xs[-1]),
        ]

    def time_multi_stop_search(self, nx, type):
        multi_stop_search(self.agg, self.waypoints)

    def time_multi_stop_search_optimize_order(self, nx, type):
        multi_stop_search(self.agg, self.waypoints, optimize_order=True)
