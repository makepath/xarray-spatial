from xrspatial import flow_direction, flow_length

from .common import get_xr_dataarray


class FlowLength:
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

    def time_flow_length_downstream(self, nx, type):
        result = flow_length(self.flow_dir, direction='downstream')
        if hasattr(result.data, 'compute'):
            result.data.compute()

    def time_flow_length_upstream(self, nx, type):
        result = flow_length(self.flow_dir, direction='upstream')
        if hasattr(result.data, 'compute'):
            result.data.compute()
