import numpy as np

from xrspatial.interpolate import idw, kriging, spline

from .common import get_xr_dataarray


def _scattered_points(n_points):
    rng = np.random.default_rng(71942)
    x = rng.uniform(-180, 180, n_points)
    y = rng.uniform(-90, 90, n_points)
    z = rng.normal(0.0, 10.0, n_points)
    return x, y, z


class IDW:
    params = ([100, 300, 1000], ["numpy", "cupy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.template = get_xr_dataarray((ny, nx), type)
        self.x, self.y, self.z = _scattered_points(100)

    def time_idw(self, nx, type):
        result = idw(self.x, self.y, self.z, self.template)
        if hasattr(result.data, 'compute'):
            result.data.compute()


class Kriging:
    # Kriging materialises a (grid_pixels, n_points + 1) prediction
    # matrix per chunk, so keep the grid smaller than the other
    # interpolators to stay inside the runner's memory.
    params = ([100, 300], ["numpy", "cupy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.template = get_xr_dataarray((ny, nx), type)
        self.x, self.y, self.z = _scattered_points(100)

    def time_kriging(self, nx, type):
        result = kriging(self.x, self.y, self.z, self.template)
        if hasattr(result.data, 'compute'):
            result.data.compute()

    def time_kriging_variance(self, nx, type):
        pred, var = kriging(self.x, self.y, self.z, self.template,
                            return_variance=True)
        if hasattr(pred.data, 'compute'):
            import dask
            dask.compute(pred.data, var.data)


class Spline:
    params = ([100, 300, 1000], ["numpy", "cupy", "dask"])
    param_names = ("nx", "type")

    def setup(self, nx, type):
        ny = nx // 2
        self.template = get_xr_dataarray((ny, nx), type)
        self.x, self.y, self.z = _scattered_points(100)

    def time_spline(self, nx, type):
        result = spline(self.x, self.y, self.z, self.template)
        if hasattr(result.data, 'compute'):
            result.data.compute()
