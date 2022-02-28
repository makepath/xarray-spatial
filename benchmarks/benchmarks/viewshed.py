from xrspatial import viewshed

from .common import get_xr_dataarray


class Viewshed:
    # Note there is no option available for cupy without rtxpy.
    params = ([100, 300, 1000, 3000], ["numpy", "rtxpy"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.xr = get_xr_dataarray(
            (ny, nx), type, different_each_call=(type == "rtxpy"))
        self.x = 100
        self.y = 50

    def time_viewshed(self, nx, type):
        viewshed(self.xr, x=self.x, y=self.y, observer_elev=1.0)
