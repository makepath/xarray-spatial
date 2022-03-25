import numpy as np
import xarray as xr

from xrspatial.gpu_rtx import has_rtx
from xrspatial.utils import has_cuda_and_cupy


def get_xr_dataarray(
    shape, type, different_each_call=False, seed=71942, is_int=False, include_nan=False
):
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

    x = np.linspace(-180, 180, nx)
    y = np.linspace(-90, 90, ny)
    x2, y2 = np.meshgrid(x, y)
    rng = np.random.default_rng(seed)

    if is_int:
        z = rng.integers(-nx, nx, size=shape).astype(np.float32)
    else:
        z = 100.0*np.exp(-x2**2 / 5e5 - y2**2 / 2e5)
        z += rng.normal(0.0, 2.0, (ny, nx))

    if different_each_call:
        if is_int:
            z[-1, -1] = np.random.default_rng().integers(-nx, nx)
        else:
            z[-1, -1] = np.random.default_rng().normal(0.0, 2.0)

    if include_nan:
        z[0, 0] = np.nan

    if type == "numpy":
        pass
    elif type == "cupy":
        if not has_cuda_and_cupy:
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

    return xr.DataArray(z, coords=dict(y=y, x=x), dims=["y", "x"])


class Benchmarking:
    params = ([100, 300, 1000, 3000, 10000], ["numpy", "cupy"])
    param_names = ("nx", "type")

    def __init__(self, func=None):
        self.func = func

    def setup(self, nx, type):
        ny = nx // 2
        self.xr = get_xr_dataarray((ny, nx), type)

    def time(self, nx, type):
        if self.func is not None:
            self.func(self.xr)
