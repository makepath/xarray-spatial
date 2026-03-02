from xrspatial import flow_direction, flow_accumulation, slope, twi

from .common import get_xr_dataarray


class TWI:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        elev = get_xr_dataarray((ny, nx), "numpy")
        flow_dir = flow_direction(elev)
        self.flow_accum = flow_accumulation(flow_dir)
        self.slope_agg = slope(elev)

        if type == "dask":
            import dask.array as da
            chunks = (max(1, ny // 2), max(1, nx // 2))
            self.flow_accum.data = da.from_array(
                self.flow_accum.data, chunks=chunks,
            )
            self.slope_agg.data = da.from_array(
                self.slope_agg.data, chunks=chunks,
            )

    def time_twi(self, nx, type):
        result = twi(self.flow_accum, self.slope_agg)
        if hasattr(result.data, 'compute'):
            result.data.compute()
