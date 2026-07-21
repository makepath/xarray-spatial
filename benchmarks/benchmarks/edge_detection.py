from xrspatial.edge_detection import laplacian, prewitt_x, prewitt_y, sobel_x, sobel_y

from .common import get_xr_dataarray


class EdgeDetection:
    params = ([300, 3000], ["numpy", "cupy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type)

    def _run(self, func):
        result = func(self.agg)
        # dask returns a lazy array; force the compute so the benchmark
        # times the kernel, not just graph construction.
        if hasattr(result.data, 'compute'):
            result.data.compute()

    def time_sobel_x(self, nx, type):
        self._run(sobel_x)

    def time_sobel_y(self, nx, type):
        self._run(sobel_y)

    def time_prewitt_x(self, nx, type):
        self._run(prewitt_x)

    def time_prewitt_y(self, nx, type):
        self._run(prewitt_y)

    def time_laplacian(self, nx, type):
        self._run(laplacian)
