import numpy as np
import xarray as xr

from .common import get_xr_dataarray


def _has_pyproj():
    try:
        import pyproj  # noqa: F401
        return True
    except ImportError:
        return False


class Reproject:
    params = ([100, 300, 1000], ["numpy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        if not _has_pyproj():
            raise NotImplementedError("pyproj required")

        ny = nx // 2
        self.xr = get_xr_dataarray((ny, nx), type)
        # Tag with WGS84 so reproject knows the source CRS.
        self.xr.attrs["crs"] = "EPSG:4326"

    def time_reproject_to_mercator(self, nx, type):
        from xrspatial.reproject import reproject
        reproject(self.xr, "EPSG:3857")

    def time_reproject_to_utm(self, nx, type):
        from xrspatial.reproject import reproject
        reproject(self.xr, "EPSG:32610")
