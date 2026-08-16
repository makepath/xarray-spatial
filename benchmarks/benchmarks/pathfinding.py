import numpy as np
import xarray as xr

from xrspatial.pathfinding import a_star_search, multi_stop_search
from xrspatial.utils import has_cuda_and_cupy

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
    params = ([100, 300, 1000], [4, 8], ["numpy", "cupy", "dask", "dask+cupy"])
    param_names = ("nx", "connectivity", "type")

    def setup(self, nx, connectivity, type):
        if type in ("dask", "dask+cupy") and nx > 300:
            # The dask backends run a pure-Python sparse A* that loads
            # chunks on demand; at nx=1000 a single call takes ~4 s,
            # which would dominate the suite's runtime.
            raise NotImplementedError()
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type)
        self.start = self.agg.y[0], self.agg.x[0]
        self.goal = self.agg.y[-1], self.agg.x[-1]
        # snap_start/snap_goal raise on dask-backed arrays by design
        self.snap = type not in ("dask", "dask+cupy")

    def time_a_star_search(self, nx, connectivity, type):
        a_star_search(
            self.agg, self.start, self.goal,
            connectivity=connectivity,
            snap_start=self.snap, snap_goal=self.snap
        )


class AStarSearchObstacles:
    """A* through a barrier field with friction (issue #3705).

    The open-grid AStarSearch benchmark is A*'s best case: the heuristic
    walks almost straight to the goal.  Walls with alternating gaps force
    a serpentine route, so the frontier and visited set actually grow,
    and the friction surface exercises the friction-weighted cost path.
    Regressions in barrier handling or frontier bookkeeping are invisible
    to the open-grid benchmark but show up here.
    """

    params = ([100, 300], ["numpy", "cupy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        z = np.ones((ny, nx), dtype=np.float64)
        # Vertical walls (value 0) with a one-cell gap alternating
        # between the top and bottom row
        step = max(2, nx // 10)
        for wi, col in enumerate(range(step, nx - 1, step)):
            z[:, col] = 0.0
            if wi % 2 == 0:
                z[0, col] = 1.0
            else:
                z[-1, col] = 1.0
        # Left-to-right friction gradient
        f = 1.0 + np.linspace(0.0, 4.0, nx)[np.newaxis, :] * np.ones(
            (ny, 1), dtype=np.float64)

        if type == "cupy":
            if not has_cuda_and_cupy():
                raise NotImplementedError()
            import cupy
            z = cupy.asarray(z)
            f = cupy.asarray(f)
        elif type == "dask":
            import dask.array as da
            chunks = (max(1, ny // 2), max(1, nx // 2))
            z = da.from_array(z, chunks=chunks)
            f = da.from_array(f, chunks=chunks)

        y = np.linspace(ny - 1, 0, ny)
        x = np.linspace(0, nx - 1, nx)
        self.agg = xr.DataArray(z, coords=dict(y=y, x=x), dims=["y", "x"])
        self.friction = xr.DataArray(
            f, coords=dict(y=y, x=x), dims=["y", "x"])
        self.start = float(y[-1]), float(x[0])
        self.goal = float(y[0]), float(x[-1])

    def time_a_star_search_barriers(self, nx, type):
        a_star_search(self.agg, self.start, self.goal, barriers=[0])

    def time_a_star_search_barriers_friction(self, nx, type):
        a_star_search(self.agg, self.start, self.goal, barriers=[0],
                      friction=self.friction)


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
