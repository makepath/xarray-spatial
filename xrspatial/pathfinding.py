import xarray as xr
import numpy as np
import collections
import heapq


def _heuristic(x1, y1, x2, y2):
    # function to calculate distance between 2 point
    # TODO: what if we want to use another distance metric?
    return abs(x1 - x2) + abs(y1 - y2)


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
        # TODO: should we consider 8 connectivity?
        (x, y) = id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

    def cost(self, x_coords, y_coords, from_node, to_node):
        return _heuristic(x_coords[from_node[1]], y_coords[from_node[0]],
                          x_coords[to_node[1]], y_coords[to_node[0]])


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def _a_star_search(graph, x_coords, y_coords, start, goal):
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
            new_cost = cost_so_far[current] + graph.cost(x_coords, y_coords, current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + _heuristic(x_coords[goal[1]],
                                                 y_coords[goal[0]],
                                                 x_coords[next[1]],
                                                 y_coords[next[0]])
                frontier.put(next, priority)
                came_from[next] = current
    return came_from, cost_so_far


def _find_pixel_id(x, y, xs, ys):

    cellsize_y = ys[1] - ys[0]
    cellsize_x = xs[1] - xs[0]

    py = int((y - ys[0]) / cellsize_y)
    px = int((x - xs[0]) / cellsize_x)
    return py, px


def _find_valid_pixels(data, barriers):
    # get valid pixel values in an input image
    valid_values = set(np.unique(data)) - set(barriers)
    # idx of all valid pixels
    valid_pixels = []
    for v in valid_values:
        pixel_ys, pixel_xs = np.where(data == v)
        for i in range(len(pixel_xs)):
            valid_pixels.append((pixel_ys[i], pixel_xs[i]))

    valid_pixels = np.asarray(valid_pixels)
    return valid_pixels


def _find_nearest_pixel(valid_pixels, py, px):
    if valid_pixels.size > 0:
        # there at least some valid pixels that not barriers
        # pixel id of the input location
        # TODO: distance by xcoords and ycoords
        distances = np.sqrt((valid_pixels[:, 0] - py) ** 2 +
                            (valid_pixels[:, 1] - px) ** 2)
        nearest_index = np.argmin(distances)
        return valid_pixels[nearest_index]
    return (-1, -1)


def _reconstruct_path(path_img, came_from, cost_so_far, start, goal):
    # construct path output image as a 2d array with NaNs for non-path pixels,
    # and the value of the path pixels being the current cost up to that point
    current = goal
    if current in came_from:
        # add cost at start
        y, x = start
        path_img[y, x] = cost_so_far[start]
        # add cost along the path
        while current != start:
            y, x = current
            # value of a path pixel is the cost up to that point
            path_img[y, x] = cost_so_far[current]
            current = came_from[current]
    return


def _is_inside(point, xmin, xmax, epsilon_x, ymin, ymax, epsilon_y):
    # check if a point at (x, y) is within
    # range from (xmin - epsilon_x, ymin - epsilon_y)
    #       to   (xmax + epsilon_x, ymax + epsilon_y)

    x, y = point
    if (x < xmin - epsilon_x) or x > xmax + epsilon_x:
        return False
    if (y < ymin - epsilon_y) or y > ymax + epsilon_y:
        return False
    return True


def a_star_search(surface, start, goal, barriers=[], x='x', y='y', snap=False):
    """
    Calculate distance from a starting point to a goal through a surface graph.
    Starting location and goal location should be within the graph.

    A* is a modification of Dijkstra’s Algorithm that is optimized for
    a single destination. Dijkstra’s Algorithm can find paths to all locations;
    A* finds paths to one location, or the closest of several locations.
    It prioritizes paths that seem to be leading closer to a goal.

    The output is an equal sized Xarray.DataArray with NaNs for non-path pixels,
    and the value of the path pixels being the current cost up to that point.

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
    x: string
        name of the x coordinate in input surface raster
    y: string
        name of the y coordinate in input surface raster
    snap: bool
        snap the start and goal locations to the nearest valid value before
        beginning path finding
    Returns
    -------
    path_agg: Xarray.DataArray with same size as input surface raster.

    Algorithm References:
    - https://www.redblobgames.com/pathfinding/a-star/implementation.html
    - https://www.redblobgames.com/pathfinding/a-star/implementation.py
    """

    if surface.ndim != 2:
        raise ValueError("surface must be 2D")

    if surface.dims != (y, x):
        raise ValueError("surface.coords should be named as coordinates:"
                         "({}, {})".format(y, x))

    y_coords = surface.coords[y].data
    x_coords = surface.coords[x].data

    epsilon_x = (x_coords[1] - x_coords[0]) / 2
    epsilon_y = (y_coords[1] - y_coords[0]) / 2
    # validate start and goal locations are in the graph
    if not _is_inside(start, x_coords[0], x_coords[-1], epsilon_x,
                      y_coords[0], y_coords[-1], epsilon_y):
        raise ValueError("start location outside the surface graph.")

    if not _is_inside(goal, x_coords[0], x_coords[-1], epsilon_x,
                      y_coords[0], y_coords[-1], epsilon_y):
        raise ValueError("goal location outside the surface graph.")

    # 2d output image that stores the path
    path_img = np.zeros_like(surface, dtype=np.float64)
    # first, initialize all cells as np.nans
    path_img[:, :] = np.nan

    # convert starting and ending point from geo coords to pixel coords
    py0, px0 = _find_pixel_id(start[0], start[1], x_coords, y_coords)
    py1, px1 = _find_pixel_id(goal[0], goal[1], x_coords, y_coords)

    if snap:
        valid_pixels = _find_valid_pixels(surface.data, barriers)
        # find nearest pixel with a valid value
        py0, px0 = _find_nearest_pixel(valid_pixels, py0, px0)
        py1, px1 = _find_nearest_pixel(valid_pixels, py1, px1)

    if py0 != -1 or py1 != -1:
        # TODO: what if start and goal are in same cell in image raster?
        #       Currently, cost = 0 and path is the cell itself
        # TODO: what if they are in same cell and value in the cell is a barrier?

        # create a diagram/graph
        height, width = surface.shape
        graph = SquareGrid(height, width)
        # all cells have same weight
        graph.weights = {}

        # find all barrier/wall cells
        graph.walls = []
        for b in barriers:
            bys, bxs = np.where(surface.data == b)
            for (y, x) in zip(bys, bxs):
                graph.walls.append((y, x))

        came_from, cost_so_far = _a_star_search(graph, x_coords, y_coords,
                                                (py0, px0), (py1, px1))

        if (py1, px1) in came_from:
            # a path found
            _reconstruct_path(path_img, came_from, cost_so_far,
                              (py0, px0), (py1, px1))

    path_agg = xr.DataArray(path_img,
                            coords=surface.coords,
                            dims=surface.dims,
                            attrs=surface.attrs)

    return path_agg
