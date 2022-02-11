from xrspatial.terrain import generate_terrain

from .common import Benchmarking


class GenerateTerrain(Benchmarking):
    def __init__(self):
        super().__init__(func=generate_terrain)

    def time_generate_terrain(self, nx, type):
        return self.time(nx, type)
