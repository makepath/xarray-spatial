import numpy as np

from xrspatial import slope

from .common import Benchmarking, get_xr_dataarray


class Slope(Benchmarking):
    def __init__(self):
        super().__init__(func=slope)

    def time_slope(self, nx, type):
        return self.time(nx, type)


class SlopeNaN(Benchmarking):
    # Speckled nodata: the planar CPU kernel's cost depends on whether the
    # NaN pattern is predictable, so time it separately from Slope.
    # get_xr_dataarray(include_nan=True) only sets the [0, 0] corner to NaN,
    # which is on the border the kernel never visits, so add 30% random NaN
    # over the interior on top of it.
    params = ([100, 300, 1000, 3000, 10000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def __init__(self):
        super().__init__(func=slope)

    def setup(self, nx, type):
        ny = nx // 2
        agg = get_xr_dataarray((ny, nx), type, include_nan=True)
        rng = np.random.default_rng(71942)
        speckle = rng.random((ny, nx), dtype=np.float32) < 0.3
        self.xr = agg.where(~speckle)

    def time_slope_nan(self, nx, type):
        # Force the compute so the dask case times the kernel rather than
        # graph construction. Slope.time_slope does not, so the dask numbers
        # of the two classes are not directly comparable.
        result = self.func(self.xr)
        if hasattr(result.data, "compute"):
            result.data.compute()
