from .common import get_xr_dataarray

try:
    from xrspatial.reproject import reproject as _reproject
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False


class Reproject:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        if not _HAS_PYPROJ:
            raise NotImplementedError("pyproj required")

        ny = nx // 2
        self.xr = get_xr_dataarray((ny, nx), type)
        # Tag with WGS84 so reproject knows the source CRS.
        self.xr.attrs["crs"] = "EPSG:4326"

    def time_reproject_to_mercator(self, nx, type):
        _reproject(self.xr, "EPSG:3857")

    def time_reproject_to_utm(self, nx, type):
        _reproject(self.xr, "EPSG:32610")
