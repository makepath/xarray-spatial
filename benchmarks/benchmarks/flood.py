from xrspatial import (
    flow_direction, flow_accumulation, hand, slope, flow_length,
)
from xrspatial.flood import (
    flood_depth, inundation, curve_number_runoff, travel_time,
)

from .common import get_xr_dataarray


class FloodDepth:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        elev = get_xr_dataarray((ny, nx), "numpy")
        flow_dir = flow_direction(elev)
        flow_accum = flow_accumulation(flow_dir)
        self.hand_agg = hand(flow_dir, flow_accum, elev)

        if type == "dask":
            import dask.array as da
            chunks = (max(1, ny // 2), max(1, nx // 2))
            self.hand_agg.data = da.from_array(
                self.hand_agg.data, chunks=chunks,
            )

    def time_flood_depth(self, nx, type):
        result = flood_depth(self.hand_agg, water_level=5.0)
        if hasattr(result.data, 'compute'):
            result.data.compute()


class Inundation:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        elev = get_xr_dataarray((ny, nx), "numpy")
        flow_dir = flow_direction(elev)
        flow_accum = flow_accumulation(flow_dir)
        self.hand_agg = hand(flow_dir, flow_accum, elev)

        if type == "dask":
            import dask.array as da
            chunks = (max(1, ny // 2), max(1, nx // 2))
            self.hand_agg.data = da.from_array(
                self.hand_agg.data, chunks=chunks,
            )

    def time_inundation(self, nx, type):
        result = inundation(self.hand_agg, water_level=5.0)
        if hasattr(result.data, 'compute'):
            result.data.compute()


class CurveNumberRunoff:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.rainfall = get_xr_dataarray((ny, nx), type)
        # Make all values positive for rainfall
        import numpy as np
        if type == "numpy":
            self.rainfall.data = np.abs(self.rainfall.data) + 1.0
        else:
            import dask.array as da
            self.rainfall.data = da.abs(self.rainfall.data) + 1.0

    def time_curve_number_runoff(self, nx, type):
        result = curve_number_runoff(self.rainfall, curve_number=80.0)
        if hasattr(result.data, 'compute'):
            result.data.compute()


class TravelTime:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        elev = get_xr_dataarray((ny, nx), "numpy")
        flow_dir = flow_direction(elev)
        self.slope_agg = slope(elev)
        self.flow_length_agg = flow_length(flow_dir)

        if type == "dask":
            import dask.array as da
            chunks = (max(1, ny // 2), max(1, nx // 2))
            self.slope_agg.data = da.from_array(
                self.slope_agg.data, chunks=chunks,
            )
            self.flow_length_agg.data = da.from_array(
                self.flow_length_agg.data, chunks=chunks,
            )

    def time_travel_time(self, nx, type):
        result = travel_time(
            self.flow_length_agg, self.slope_agg, mannings_n=0.03,
        )
        if hasattr(result.data, 'compute'):
            result.data.compute()
