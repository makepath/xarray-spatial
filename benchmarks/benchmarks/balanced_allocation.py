import numpy as np
import xarray as xr

from xrspatial.balanced_allocation import balanced_allocation

from .common import get_xr_dataarray


class BalancedAllocation:
    # Memory-intensive: holds N_sources cost surfaces simultaneously.
    # Keep grids small to avoid OOM during benchmarking.
    params = ([100, 300], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        # Friction surface: positive values everywhere.
        friction = get_xr_dataarray((ny, nx), type)
        if type == "dask":
            import dask.array as da
            friction.data = da.fabs(friction.data) + 0.1
        else:
            friction.data = np.abs(friction.data) + 0.1
        self.friction = friction

        # Source raster: place 4 source points in corners.
        sources = np.zeros((ny, nx), dtype=np.float32)
        margin = max(1, nx // 10)
        sources[margin, margin] = 1
        sources[margin, nx - margin - 1] = 2
        sources[ny - margin - 1, margin] = 3
        sources[ny - margin - 1, nx - margin - 1] = 4

        x = np.linspace(-180, 180, nx)
        y = np.linspace(-90, 90, ny)

        if type == "dask":
            import dask.array as da
            data = da.from_array(sources, chunks=(max(1, ny // 2), max(1, nx // 2)))
        else:
            data = sources

        self.raster = xr.DataArray(data, coords=dict(y=y, x=x), dims=["y", "x"])

    def time_balanced_allocation(self, nx, type):
        balanced_allocation(self.raster, self.friction, max_iterations=10)
