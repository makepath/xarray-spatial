import numpy as np

from xrspatial.cost_distance import cost_distance

from .common import get_xr_dataarray


class CostDistance:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type, is_int=True)
        friction = get_xr_dataarray((ny, nx), type)
        # Ensure positive friction values
        if type == "dask":
            import dask.array as da
            friction.data = da.fabs(friction.data) + 0.1
        else:
            friction.data = np.abs(friction.data) + 0.1
        self.friction = friction

        # Pick a finite max_cost whose pixel radius stays well inside one
        # chunk (chunks are ny//2 x nx//2), so the dask path exercises the
        # bounded map_overlap branch rather than the unbounded iterative
        # one. radius_px = max_cost / (f_min * cellsize); friction >= 0.1
        # so using 0.1 keeps the true radius at or below the target.
        cellsize = min(360.0 / (nx - 1), 180.0 / (ny - 1))
        self.max_cost = 10 * 0.1 * cellsize

    def time_cost_distance(self, nx, type):
        cost_distance(self.agg, self.friction)

    def time_cost_distance_bounded(self, nx, type):
        cost_distance(self.agg, self.friction, max_cost=self.max_cost)
