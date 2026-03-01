from xrspatial import flow_direction, watershed, basins

from .common import get_xr_dataarray

import numpy as np


class Watershed:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        elev = get_xr_dataarray((ny, nx), "numpy")
        self.flow_dir = flow_direction(elev)
        fd_data = self.flow_dir.data

        # Create pour_points at pits (code 0)
        pp = np.full_like(fd_data, np.nan)
        pit_mask = fd_data == 0
        pp[pit_mask] = np.arange(1, pit_mask.sum() + 1, dtype=np.float64)

        if type == "dask":
            import dask.array as da
            self.flow_dir.data = da.from_array(
                fd_data,
                chunks=(max(1, ny // 2), max(1, nx // 2)),
            )
            self.pour_points = self.flow_dir.copy()
            self.pour_points.data = da.from_array(
                pp,
                chunks=(max(1, ny // 2), max(1, nx // 2)),
            )
        else:
            self.pour_points = self.flow_dir.copy()
            self.pour_points.data = pp

    def time_watershed(self, nx, type):
        result = watershed(self.flow_dir, self.pour_points)
        if hasattr(result.data, 'compute'):
            result.data.compute()


class Basins:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        elev = get_xr_dataarray((ny, nx), "numpy")
        self.flow_dir = flow_direction(elev)
        fd_data = self.flow_dir.data

        if type == "dask":
            import dask.array as da
            self.flow_dir.data = da.from_array(
                fd_data,
                chunks=(max(1, ny // 2), max(1, nx // 2)),
            )

    def time_basins(self, nx, type):
        result = basins(self.flow_dir)
        if hasattr(result.data, 'compute'):
            result.data.compute()
