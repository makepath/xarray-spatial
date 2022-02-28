from xrspatial import slope

from .common import Benchmarking


class Slope(Benchmarking):
    def __init__(self):
        super().__init__(func=slope)

    def time_slope(self, nx, type):
        return self.time(nx, type)
