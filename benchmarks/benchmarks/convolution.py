import numpy as np

from xrspatial.convolution import circle_kernel, convolve_2d

from .common import get_xr_dataarray


class Convolve2d:
    params = ([300, 1000, 3000], [(5, 5), (25, 25)], ["numpy", "cupy", "dask"])
    param_names = ("nx", "kernelsize", "type")

    def setup(self, nx, kernelsize, type):
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type)
        kernel_h, kernel_w = kernelsize
        self.kernel = np.ones((kernel_h, kernel_w), dtype=np.float64)

    def time_convolve_2d(self, nx, kernelsize, type):
        # convolve_2d takes the backing array, not the DataArray wrapper.
        convolve_2d(self.agg.data, self.kernel)


class CircleKernel:
    params = ([3, 25, 100],)
    param_names = ("radius",)

    def time_circle_kernel(self, radius):
        circle_kernel(1, 1, radius)
