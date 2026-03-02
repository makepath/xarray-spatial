from xrspatial import flow_direction, flow_accumulation, hand

from .common import get_xr_dataarray


class HAND:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.elev = get_xr_dataarray((ny, nx), "numpy")
        flow_dir = flow_direction(self.elev)
        self.flow_dir = flow_dir
        self.flow_accum = flow_accumulation(flow_dir)

        if type == "dask":
            import dask.array as da
            chunks = (max(1, ny // 2), max(1, nx // 2))
            fd_data = self.flow_dir.data
            fa_data = self.flow_accum.data
            elev_data = self.elev.data

            self.flow_dir.data = da.from_array(fd_data, chunks=chunks)
            self.flow_accum.data = da.from_array(fa_data, chunks=chunks)
            self.elev.data = da.from_array(elev_data, chunks=chunks)

    def time_hand(self, nx, type):
        result = hand(self.flow_dir, self.flow_accum, self.elev)
        if hasattr(result.data, 'compute'):
            result.data.compute()
