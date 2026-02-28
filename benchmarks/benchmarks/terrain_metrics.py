from xrspatial import tri, tpi, roughness

from .common import Benchmarking


class TRI(Benchmarking):
    def __init__(self):
        super().__init__(func=tri)

    def time_tri(self, nx, type):
        return self.time(nx, type)


class TPI(Benchmarking):
    def __init__(self):
        super().__init__(func=tpi)

    def time_tpi(self, nx, type):
        return self.time(nx, type)


class Roughness(Benchmarking):
    def __init__(self):
        super().__init__(func=roughness)

    def time_roughness(self, nx, type):
        return self.time(nx, type)
