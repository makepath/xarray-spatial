import numpy as np
import xarray as xr

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


class SurfaceDistanceDenseTargets:
    """Dijkstra with a mid-density target raster over rugged relief.

    This is the regime where the lazy-deletion heap holds more than one
    live entry per pixel (#3723).  The existing SurfaceDistance benchmark
    misses it: its integer source raster makes nearly every pixel a
    target, so almost no relaxation ever improves a distance.
    """

    params = ([200, 400], [0.05, 0.2, 0.4])
    param_names = ("nx", "target_fraction")

    def setup(self, nx, target_fraction):
        ny = nx // 2
        rng = np.random.default_rng(71942)
        source = np.zeros((ny, nx), dtype=np.float64)
        n_targets = max(1, int(ny * nx * target_fraction))
        source.flat[rng.choice(ny * nx, size=n_targets, replace=False)] = 1.0
        elev = rng.random((ny, nx)) * 200.0

        coords = dict(y=np.arange(ny, dtype=np.float64),
                      x=np.arange(nx, dtype=np.float64))
        self.agg = xr.DataArray(source, coords=coords, dims=["y", "x"],
                                attrs={"res": (1.0, 1.0)})
        self.elev = xr.DataArray(elev, coords=coords, dims=["y", "x"],
                                 attrs={"res": (1.0, 1.0)})

    def time_surface_distance(self, nx, target_fraction):
        surface_distance(self.agg, self.elev)

    def peakmem_surface_distance(self, nx, target_fraction):
        surface_distance(self.agg, self.elev)
