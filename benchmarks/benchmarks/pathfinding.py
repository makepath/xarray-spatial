from xrspatial.pathfinding import a_star_search

from .common import get_xr_dataarray


class AStarSearch:
    params = ([10, 100, 300], [4, 8], ["numpy"])
    param_names = ("nx", "connectivity", "type")

    def setup(self, nx, connectivity, type):
        ny = nx // 2
        self.agg = get_xr_dataarray((ny, nx), type)
        self.start = self.agg.y[0], self.agg.x[0]
        self.goal = self.agg.y[-1], self.agg.x[-1]

    def time_a_star_search(self, nx, connectivity, type):
        a_star_search(
            self.agg, self.start, self.goal,
            connectivity=connectivity,
            snap_start=True, snap_goal=True
        )
