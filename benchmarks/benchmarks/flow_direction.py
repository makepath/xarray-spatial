from xrspatial import flow_direction

from .common import Benchmarking


class FlowDirection(Benchmarking):
    def __init__(self):
        super().__init__(func=flow_direction)

    def time_flow_direction(self, nx, type):
        return self.time(nx, type)
