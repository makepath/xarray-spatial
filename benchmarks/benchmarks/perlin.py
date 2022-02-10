from xrspatial.perlin import perlin

from .common import Benchmarking


class Perlin(Benchmarking):
    def __init__(self):
        super().__init__(func=perlin)

    def time_perlin(self, nx, type):
        return self.time(nx, type)
