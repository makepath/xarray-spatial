from xrspatial import curvature

from .common import Benchmarking


class Curvature(Benchmarking):
    def __init__(self):
        super().__init__(func=curvature)

    def time_curvature(self, nx, type):
        return self.time(nx, type)
