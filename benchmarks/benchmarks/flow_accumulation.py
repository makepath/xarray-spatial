from xrspatial import flow_direction, flow_accumulation

from .common import Benchmarking, get_xr_dataarray


class FlowAccumulation:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        elev = get_xr_dataarray((ny, nx), type)
        # Compute flow_dir from elevation (materialise for benchmarking)
        self.flow_dir = flow_direction(elev)
        if hasattr(self.flow_dir.data, 'compute'):
            self.flow_dir.data = self.flow_dir.data.compute()
            # Re-chunk for dask benchmark
            if type == "dask":
                import dask.array as da
                self.flow_dir.data = da.from_array(
                    self.flow_dir.data,
                    chunks=(max(1, ny // 2), max(1, nx // 2)),
                )

    def time_flow_accumulation(self, nx, type):
        result = flow_accumulation(self.flow_dir)
        if hasattr(result.data, 'compute'):
            result.data.compute()
