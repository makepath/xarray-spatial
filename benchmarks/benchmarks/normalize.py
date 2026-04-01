from xrspatial.normalize import rescale, standardize

from .common import Benchmarking


class Rescale(Benchmarking):
    def __init__(self):
        super().__init__(func=rescale)

    def time_rescale(self, nx, type):
        return self.time(nx, type)


class Standardize(Benchmarking):
    def __init__(self):
        super().__init__(func=standardize)

    def time_standardize(self, nx, type):
        return self.time(nx, type)
