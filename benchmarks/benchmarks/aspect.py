from xrspatial import aspect

from .common import Benchmarking


class Aspect(Benchmarking):
    def __init__(self):
        super().__init__(func=aspect)

    def time_aspect(self, nx, type):
        return self.time(nx, type)
