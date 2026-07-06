import numpy as np
import xarray as xr

from xrspatial import bump


class Bump:
    # Two axes: raster width and backend.  The "dask" case times graph
    # construction only (no compute), which is where _partition_bumps runs;
    # it is finely chunked so the per-chunk bump partitioning is exercised.
    params = ([256, 1024, 2048], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx
        if type == "numpy":
            self.agg = xr.DataArray(np.zeros((ny, nx)), dims=["y", "x"])
        elif type == "dask":
            import dask.array as da
            self.agg = xr.DataArray(
                da.zeros((ny, nx), chunks=(64, 64), dtype=np.float64),
                dims=["y", "x"],
            )
        else:
            raise NotImplementedError()
        np.random.seed(71942)

    def time_bump(self, nx, type):
        # numpy: full eager render.  dask: lazy graph build (no compute).
        bump(agg=self.agg, count=50000, spread=1)
