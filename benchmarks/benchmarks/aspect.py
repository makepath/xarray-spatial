from xrspatial import aspect
from .common import get_xr_dataarray


class Aspect:
    params = ([100, 300, 1000, 3000, 10000], ["numpy", "cupy"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.xr = get_xr_dataarray((ny, nx), type)

    def time_aspect(self, nx, type):
        aspect(self.xr)
