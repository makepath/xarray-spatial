import numpy as np

from xrspatial.proximity import allocation, direction, proximity

from .common import get_xr_dataarray


class Base:
    params = (
        [100, 1000],
        [1, 10, 100],
        ["EUCLIDEAN", "GREAT_CIRCLE", "MANHATTAN"],
        ["numpy"]
    )
    param_names = ("nx", "n_target_values", "distance_metric", "type")

    def setup(self, nx, n_target_values, distance_metric, type):
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type, is_int=True)
        unique_values = np.unique(self.agg.data)
        self.target_values = unique_values[:n_target_values]


class Proximity(Base):
    def time_proximity(self, nx, n_target_values, distance_metric, type):
        proximity(self.agg, target_values=self.target_values, distance_metric=distance_metric)


class Allocation(Base):
    def time_allocation(self, nx, n_target_values, distance_metric, type):
        allocation(self.agg, target_values=self.target_values, distance_metric=distance_metric)


class Direction(Base):
    def time_direction(self, nx, n_target_values, distance_metric, type):
        direction(self.agg, target_values=self.target_values, distance_metric=distance_metric)
