import numpy as np

from xrspatial.surface_distance import (
    surface_distance, surface_allocation, surface_direction,
)

from .common import get_xr_dataarray


class SurfaceDistance:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type, is_int=True)
        self.elev = get_xr_dataarray((ny, nx), type)

    def time_surface_distance(self, nx, type):
        surface_distance(self.agg, self.elev)

    def time_surface_allocation(self, nx, type):
        surface_allocation(self.agg, self.elev)

    def time_surface_direction(self, nx, type):
        surface_direction(self.agg, self.elev)
