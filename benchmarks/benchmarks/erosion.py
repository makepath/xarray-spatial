from xrspatial.erosion import erode

from .common import get_xr_dataarray


class Erosion:
    # Erosion is a global operation that materialises dask arrays.
    # Keep grid sizes small and iteration counts low so benchmarks
    # finish in reasonable time.
    params = ([100, 300], ["numpy", "cupy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.xr = get_xr_dataarray((ny, nx), type)

    def time_erode_500(self, nx, type):
        erode(self.xr, iterations=500, seed=42)

    def time_erode_5000(self, nx, type):
        erode(self.xr, iterations=5000, seed=42)
