import numpy as np
import collections
import heapq


class Queue:
    def __init__(self):
        self.elements = collections.deque()

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.popleft()


class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.walls

    def neighbors(self, id):
        (x, y) = id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results


class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}

    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


# Sample code from https://www.redblobgames.com/pathfinding/a-star/
# Copyright 2014 Red Blob Games <redblobgames@gmail.com>
#
# Feel free to use this code in your own projects, including commercial projects
# License: Apache v2.0 <http://www.apache.org/licenses/LICENSE-2.0.html>
def _a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    return came_from, cost_so_far


def _find_pixel_idx(x, y, xs, ys):

    cellsize_y = ys[1] - ys[0]
    cellsize_x = xs[1] - xs[0]

    py = int((y - ys[0]) / cellsize_y)
    px = int((x - xs[0]) / cellsize_x)
    return py, px


def _reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def a_star_search(surface, start, goal, barriers=[], x='x', y='y'):
    """
    Calculate distance from a starting point to a goal through a surface graph.

    A* is a modification of Dijkstra’s Algorithm that is optimized for
    a single destination. Dijkstra’s Algorithm can find paths to all locations;
    A* finds paths to one location, or the closest of several locations.
    It prioritizes paths that seem to be leading closer to a goal.

    Parameters
    ----------
    surface : xarray.DataArray
        xarray.DataArray of values to bin
    start: array like object (tuple, list, array, ...) of 2 numeric elements
        (x, y) or (lon, lat) coordinates of the starting point
    goal: array like object (tuple, list, array, ...) of 2 numeric elements
        (x, y) or (lon, lat) coordinates of the goal location
    barriers: array like object
        list of values inside the surface which are barriers (cannot cross)

    Returns
    -------
    distance: distance from start location to goal location
    Return -1 if no path found

    Algorithm References:
    - https://www.redblobgames.com/pathfinding/a-star/implementation.html
    - https://www.redblobgames.com/pathfinding/a-star/implementation.py
    """

    if surface.ndim != 2:
        raise ValueError("surface must be 2D")

    if surface.dims != (y, x):
        raise ValueError("surface.coords should be named as coordinates:"
                         "(%s, %s)".format(y, x))

    y_coords = surface.coords[y].data
    x_coords = surface.coords[x].data

    # convert starting and ending point from geo coords to pixel coords
    py0, px0 = _find_pixel_idx(start[0], start[1], x_coords, y_coords)
    py1, px1 = _find_pixel_idx(goal[0], goal[1], x_coords, y_coords)

    # create a diagram/graph
    height, width = surface.shape
    graph = GridWithWeights(height, width)
    # all cells have same weight
    graph.weights = {}

    # find all barrier/wall cells
    graph.walls = []
    for b in barriers:
        bys, bxs = np.where(surface.data == b)
        for (y, x) in zip(bys, bxs):
            graph.walls.append((y, x))

    came_from, cost_so_far = _a_star_search(graph, (py0, px0), (py1, px1))
    path = _reconstruct_path(came_from, (py0, px0), (py1, px1))
    return path
