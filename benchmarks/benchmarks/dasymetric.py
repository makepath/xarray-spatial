import numpy as np
import xarray as xr

from xrspatial.dasymetric import disaggregate

from .common import get_xr_dataarray


class Dasymetric:
    params = ([100, 300, 1000], ["numpy", "cupy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2

        # Zones: 4 rectangular blocks.
        zones_np = np.zeros((ny, nx), dtype=np.int32)
        zones_np[: ny // 2, : nx // 2] = 1
        zones_np[: ny // 2, nx // 2 :] = 2
        zones_np[ny // 2 :, : nx // 2] = 3
        zones_np[ny // 2 :, nx // 2 :] = 4

        x = np.linspace(-180, 180, nx)
        y = np.linspace(-90, 90, ny)

        if type == "dask":
            import dask.array as da
            zdata = da.from_array(zones_np, chunks=(max(1, ny // 2), max(1, nx // 2)))
        elif type == "cupy":
            from xrspatial.utils import has_cuda_and_cupy
            if not has_cuda_and_cupy:
                raise NotImplementedError()
            import cupy
            zdata = cupy.asarray(zones_np)
        else:
            zdata = zones_np

        self.zones = xr.DataArray(zdata, coords=dict(y=y, x=x), dims=["y", "x"])

        # Values: one total per zone.
        self.values = {1: 1000.0, 2: 2000.0, 3: 1500.0, 4: 2500.0}

        # Weight surface: use the standard Gaussian bump.
        weight = get_xr_dataarray((ny, nx), type)
        # Make weights non-negative.
        if type == "dask":
            import dask.array as da
            weight.data = da.fabs(weight.data) + 0.01
        elif type == "cupy":
            import cupy
            weight.data = cupy.abs(weight.data) + 0.01
        else:
            weight.data = np.abs(weight.data) + 0.01
        self.weight = weight

    def time_disaggregate_weighted(self, nx, type):
        disaggregate(self.zones, self.values, self.weight, method="weighted")

    def time_disaggregate_binary(self, nx, type):
        disaggregate(self.zones, self.values, self.weight, method="binary")
