import xarray as xr
import numpy as np


def _heuristic(x1, y1, x2, y2):
    # function to calculate distance between 2 point
    # TODO: what if we want to use another distance metric?
    return abs(x1 - x2) + abs(y1 - y2)


def _find_min_cost_pixel(cost, is_open):
    height, width = cost.shape
    # set min cost to a very big number
    min_cost = 1000000
    py = None
    px = None
    for i in range(height):
        for j in range(width):
            if is_open[i, j] and cost[i, j] < min_cost:
                min_cost = cost[i, j]
                py = i
                px = j
    return py, px


def astar(data, path_img, start_py, start_px, goal_py, goal_px, barriers):
    # parent of the (i, j) pixel is the pixel at (parent_ys[i, j], parent_xs[i, j])
    parent_ys = np.ones(data.shape, dtype=int) * -1
    parent_xs = np.ones(data.shape, dtype=int) * -1
    # parent of start is itself
    parent_ys[start_py, start_px] = start_py
    parent_xs[start_py, start_px] = start_px

    # distance between the current node and the start node
    d_from_start = np.zeros_like(data, dtype=float)
    # total cost of the node: cost = d_from_start + estimated_d_to_goal
    # heuristic — estimated distance from the current node to the end node
    cost = np.zeros_like(data, dtype=float)

    # initialize both open and closed list all False
    is_open = np.zeros(data.shape, dtype=bool)
    is_closed = np.zeros(data.shape, dtype=bool)

    # add the start node to open list
    is_open[start_py, start_px] = True
    # init cost at start location
    d_from_start[start_py, start_px] = 0
    estimated_distance = _heuristic(start_px, start_py, goal_px, goal_py)
    cost[start_py, start_px] = d_from_start[start_py, start_px] + estimated_distance

    # 8-connectivity
    neighbor_xs = [-1, -1, -1, 0, 0, 1, 1, 1]
    neighbor_ys = [-1, 0, 1, -1, 1, -1, 0, 1]
    #     neighbor_ys = [0, -1, 1, 0]
    #     neighbor_xs = [-1, 0, 0, 1]

    height, width = data.shape
    num_open = np.sum(is_open)
    while num_open > 0:
        py, px = _find_min_cost_pixel(cost, is_open)
        # pop current node off open list, add it to closed list
        is_open[py, px] = False
        is_closed[py, px] = True

        # found the goal
        if (py, px) == (goal_py, goal_px):
            # reconstruct path
            reconstruct_path(path_img, parent_ys, parent_xs,
                             d_from_start, start_py, start_px,
                             goal_py, goal_px)
            return

        # visit neighborhood
        for y, x in zip(neighbor_ys, neighbor_xs):
            neighbor_y = py + y
            neighbor_x = px + x

            # neighbor is within the surface image
            if neighbor_y > height - 1 or neighbor_y < 0 \
                    or neighbor_x > width - 1 or neighbor_x < 0:
                continue

            # walkable
            if data[neighbor_y][neighbor_x] in barriers:
                continue

            # check if neighbor is in the closed list
            if is_closed[neighbor_y, neighbor_x]:
                continue

            # distance from start to this neighbor
            d = d_from_start[py, px] + 1
            # if neighbor is already in the open list
            if is_open[neighbor_y, neighbor_x] and d > d_from_start[neighbor_y, neighbor_x]:
                continue

            # calculate cost
            d_from_start[neighbor_y, neighbor_x] = d
            estimated_d_to_goal = _heuristic(neighbor_x, neighbor_y, goal_px,
                                             goal_py)
            cost[neighbor_y, neighbor_x] = d_from_start[neighbor_y, neighbor_x] + \
                estimated_d_to_goal
            # add neighbor to the open list
            is_open[neighbor_y, neighbor_x] = True
            parent_ys[neighbor_y, neighbor_x] = py
            parent_xs[neighbor_y, neighbor_x] = px

        num_open = np.sum(is_open)
    return


def _find_pixel_id(x, y, xs, ys):
    cellsize_y = ys[1] - ys[0]
    cellsize_x = xs[1] - xs[0]

    py = int((y - ys[0]) / cellsize_y)
    px = int((x - xs[0]) / cellsize_x)
    return py, px


def _find_valid_pixels(data, barriers):
    # get valid pixel values in an input image
    valid_values = set(np.unique(data[~np.isnan(data)])) - set(barriers)
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


def reconstruct_path(path_img, parent_ys, parent_xs, cost, start_py, start_px,
                     goal_py, goal_px):
    # construct path output image as a 2d array with NaNs for non-path pixels,
    # and the value of the path pixels being the current cost up to that point
    current_x = goal_px
    current_y = goal_py

    if parent_xs[current_y, current_x] != -1 and parent_ys[current_y, current_x] != -1:
        # exist path from start to goal
        # add cost at start
        path_img[start_py, start_px] = cost[start_py, start_px]
        # add cost along the path
        while current_x != start_px or current_y != start_py:
            # value of a path pixel is the cost up to that point
            path_img[current_y, current_x] = cost[current_y, current_x]
            parent_y = parent_ys[current_y, current_x]
            parent_x = parent_xs[current_y, current_x]
            current_y = parent_y
            current_x = parent_x
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
        astar(surface.data, path_img, py0, px0, py1, px1, barriers)

    path_agg = xr.DataArray(path_img,
                            coords=surface.coords,
                            dims=surface.dims,
                            attrs=surface.attrs)

    return path_agg
