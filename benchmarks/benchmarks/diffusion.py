from xrspatial.diffusion import diffuse

from .common import Benchmarking


class Diffusion(Benchmarking):
    params = ([100, 300, 1000], ["numpy", "cupy", "dask"])
    param_names = ("nx", "type")

    def __init__(self):
        super().__init__(func=None)

    def time_diffuse_1step(self, nx, type):
        diffuse(self.xr, diffusivity=1.0, steps=1)

    def time_diffuse_10steps(self, nx, type):
        diffuse(self.xr, diffusivity=1.0, steps=10)
