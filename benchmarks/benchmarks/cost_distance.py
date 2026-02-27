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

    def time_cost_distance(self, nx, type):
        cost_distance(self.agg, self.friction)
