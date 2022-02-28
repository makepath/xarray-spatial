from xrspatial import hillshade

from .common import get_xr_dataarray


class Hillshade:
    # Note that rtxpy hillshade includes shadow calculations so timings are
    # not comparable with numpy and cupy hillshade.
    params = ([100, 300, 1000, 3000], ["numpy", "cupy", "rtxpy"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.xr = get_xr_dataarray(
            (ny, nx), type, different_each_call=(type == "rtxpy"))

    def time_hillshade(self, nx, type):
        shadows = (type == "rtxpy")
        hillshade(self.xr, shadows=shadows)
