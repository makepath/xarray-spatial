import numpy as np
import xarray as xr
from xrspatial.gpu_rtx import has_rtx
from xrspatial.utils import has_cuda, has_cupy


def get_xr_dataarray(shape, type, different_each_call=False):
    # Gaussian bump with noise.
    #
    # Valid types are "numpy", "cupy" and "rtxpy". Using "numpy" will return
    # a numpy-backed xarray DataArray. Using either of the other two will
    # return a cupy-backed DataArray but only if the required dependencies are
    # available, otherwise a NotImplementedError will be raised so that the
    # benchmark will not be run,
    #
    # Calling with different_each_call=True will ensure that each array
    # returned by this function is different by randomly changing the last
    # element. This is required for functions that create an rtxpy
    # triangulation to avoid them reusing a cached triangulation leading to
    # optimistically fast benchmark times.
    ny, nx = shape

    x = np.linspace(-1000, 1000, nx)
    y = np.linspace(-800, 800, ny)
    x2, y2 = np.meshgrid(x, y)
    z = 100.0*np.exp(-x2**2 / 5e5 - y2**2 / 2e5)

    rng = np.random.default_rng(71942)
    z += rng.normal(0.0, 2.0, (ny, nx))

    if different_each_call:
        z[-1, -1] = np.random.default_rng().normal(0.0, 2.0)

    if type == "numpy":
        pass
    elif type == "cupy":
        if not (has_cuda() and has_cupy()):
            raise NotImplementedError()
        import cupy
        z = cupy.asarray(z)
    elif type == "rtxpy":
        if not has_rtx():
            raise NotImplementedError()
        import cupy
        z = cupy.asarray(z)
    else:
        raise RuntimeError(f"Unrecognised type {type}")

    return xr.DataArray(z, coords=dict(x=x, y=y), dims=["y", "x"])


class Benchmarking:
    params = ([100, 300, 1000, 3000, 10000], ["numpy", "cupy"])
    param_names = ("nx", "type")

    def __init__(self, func):
        self.func = func

    def setup(self, nx, type):
        ny = nx // 2
        self.xr = get_xr_dataarray((ny, nx), type)

    def time(self, nx, type):
        self.func(self.xr)
