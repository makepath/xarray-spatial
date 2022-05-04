import numpy as np

from xrspatial.convolution import custom_kernel
from xrspatial.focal import apply, focal_stats, hotspots, mean

from .common import get_xr_dataarray


class Focal:
    params = ([100, 300, 1000, 3000], [(5, 5), (25, 25)], ["numpy", "cupy"])
    param_names = ("nx", "kernelsize", "type")

    def setup(self, nx, kernelsize, type):
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type)
        kernel_w, kernel_h = kernelsize
        self.kernel = custom_kernel(np.ones((kernel_h, kernel_w)))


class FocalApply(Focal):
    params = ([100, 300, 1000, 3000], [(5, 5), (25, 25)], ["numpy"])

    def time_apply(self, nx, kernelsize, type):
        apply(self.agg, self.kernel)


class FocalHotspots(Focal):
    def time_hotspots(self, nx, kernelsize, type):
        hotspots(self.agg, self.kernel)


class FocalStats(Focal):
    params = ([100, 300, 1000, 3000], [(5, 5), (15, 15)], ["numpy", "cupy"])

    def time_focal_stats(self, nx, kernelsize, type):
        focal_stats(self.agg, self.kernel)


class FocalMean:
    params = ([100, 300, 1000, 3000, 10000], [1, 10], ["numpy", "cupy"])
    param_names = ("nx", "passes", "type")

    def setup(self, nx, passes, type):
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type)

    def time_mean(self, nx, passes, type):
        mean(self.agg, passes)
