import warnings
from typing import Optional, Union

import numpy as np
import xarray as xr

from xrspatial.utils import get_dataarray_resolution, ngjit

NONE = -1


def _get_pixel_id(point, raster, xdim=None, ydim=None):
    # get location in `raster` pixel space for `point` in y-x coordinate space
    # point: (y, x) - coordinates of the point
    # xdim: name of the x coordinate dimension in input `raster`.
    # ydim: name of the x coordinate dimension in input `raster`

    if ydim is None:
        ydim = raster.dims[-2]
    if xdim is None:
        xdim = raster.dims[-1]
    y_coords = raster.coords[ydim].data
    x_coords = raster.coords[xdim].data

    cellsize_x, cellsize_y = get_dataarray_resolution(raster, xdim, ydim)
    py = int(abs(point[0] - y_coords[0]) / cellsize_y)
    px = int(abs(point[1] - x_coords[0]) / cellsize_x)

    # return index of row and column where the `point` located.
    return py, px


@ngjit
def _is_not_crossable(cell_value, barriers):
    # nan cell is not walkable
    if np.isnan(cell_value):
        return True

    for i in barriers:
        if cell_value == i:
            return True
    return False


@ngjit
def _is_inside(py, px, h, w):
    inside = True
    if px < 0 or px >= w:
        inside = False
    if py < 0 or py >= h:
        inside = False
    return inside


@ngjit
def _distance(x1, y1, x2, y2):
    # euclidean distance in pixel space from (y1, x1) to (y2, x2)
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


@ngjit
def _heuristic(x1, y1, x2, y2):
    # heuristic to estimate distance between 2 point
    # TODO: what if we want to use another distance metric?
    return _distance(x1, y1, x2, y2)


@ngjit
def _min_cost_pixel_id(cost, is_open):
    height, width = cost.shape
    py = NONE
    px = NONE
    # set min cost to a very big number
    # this value is only an estimation
    min_cost = (height + width) ** 2
    for i in range(height):
        for j in range(width):
            if is_open[i, j] and cost[i, j] < min_cost:
                min_cost = cost[i, j]
                py = i
                px = j
    return py, px


@ngjit
def _find_nearest_pixel(py, px, data, barriers):
    # if the cell is already valid, return itself
    if not _is_not_crossable(data[py, px], barriers):
        return py, px

    height, width = data.shape
    # init min distance as max possible distance
    min_distance = _distance(0, 0, height - 1, width - 1)
    # return of the function
    nearest_y = NONE
    nearest_x = NONE
    for y in range(height):
        for x in range(width):
            if not _is_not_crossable(data[y, x], barriers):
                d = _distance(x, y, px, py)
                if d < min_distance:
                    min_distance = d
                    nearest_y = y
                    nearest_x = x

    return nearest_y, nearest_x


@ngjit
def _reconstruct_path(path_img, parent_ys, parent_xs, cost,
                      start_py, start_px, goal_py, goal_px):
    # construct path output image as a 2d array with NaNs for non-path pixels,
    # and the value of the path pixels being the current cost up to that point
    current_x = goal_px
    current_y = goal_py

    if parent_xs[current_y, current_x] != NONE and \
            parent_ys[current_y, current_x] != NONE:
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


def _neighborhood_structure(connectivity=8):
    if connectivity == 8:
        # 8-connectivity
        neighbor_xs = [-1, -1, -1, 0, 0, 1, 1, 1]
        neighbor_ys = [-1, 0, 1, -1, 1, -1, 0, 1]
    else:
        # 4-connectivity
        neighbor_ys = [0, -1, 1, 0]
        neighbor_xs = [-1, 0, 0, 1]
    return np.array(neighbor_ys), np.array(neighbor_xs)


@ngjit
def _a_star_search(data, path_img, start_py, start_px, goal_py, goal_px,
                   barriers, neighbor_ys, neighbor_xs):

    height, width = data.shape
    # parent of the (i, j) pixel is the pixel at
    # (parent_ys[i, j], parent_xs[i, j])
    # first initialize parent of all cells as invalid (NONE, NONE)
    parent_ys = np.ones((height, width), dtype=np.int64) * NONE
    parent_xs = np.ones((height, width), dtype=np.int64) * NONE

    # parent of start is itself
    parent_ys[start_py, start_px] = start_py
    parent_xs[start_py, start_px] = start_px

    # distance from start to the current node
    d_from_start = np.zeros_like(data, dtype=np.float64)
    # total cost of the node: cost = d_from_start + d_to_goal
    # heuristic — estimated distance from the current node to the end node
    cost = np.zeros_like(data, dtype=np.float64)

    # initialize both open and closed list all False
    is_open = np.zeros(data.shape, dtype=np.bool_)
    is_closed = np.zeros(data.shape, dtype=np.bool_)

    if not _is_not_crossable(data[start_py, start_px], barriers):
        # if start node is crossable
        # add the start node to open list
        is_open[start_py, start_px] = True
        # init cost at start location
        d_from_start[start_py, start_px] = 0
        cost[start_py, start_px] = d_from_start[start_py, start_px] + \
            _heuristic(start_px, start_py, goal_px, goal_py)

    num_open = np.sum(is_open)
    while num_open > 0:
        py, px = _min_cost_pixel_id(cost, is_open)
        # pop current node off open list, add it to closed list
        is_open[py][px] = 0
        is_closed[py][px] = True
        # found the goal
        if (py, px) == (goal_py, goal_px):
            # reconstruct path
            _reconstruct_path(path_img, parent_ys, parent_xs,
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
            if _is_not_crossable(data[neighbor_y][neighbor_x], barriers):
                continue

            # check if neighbor is in the closed list
            if is_closed[neighbor_y, neighbor_x]:
                continue

            # distance from start to this neighbor
            d = d_from_start[py, px] + _distance(px, py,
                                                 neighbor_x, neighbor_y)
            # if neighbor is already in the open list
            if is_open[neighbor_y, neighbor_x] and \
                    d > d_from_start[neighbor_y, neighbor_x]:
                continue

            # calculate cost
            d_from_start[neighbor_y, neighbor_x] = d
            d_to_goal = _heuristic(neighbor_x, neighbor_y, goal_px, goal_py)
            cost[neighbor_y, neighbor_x] = \
                d_from_start[neighbor_y, neighbor_x] + d_to_goal
            # add neighbor to the open list
            is_open[neighbor_y, neighbor_x] = True
            parent_ys[neighbor_y, neighbor_x] = py
            parent_xs[neighbor_y, neighbor_x] = px

        num_open = np.sum(is_open)
    return


def a_star_search(surface: xr.DataArray,
                  start: Union[tuple, list, np.array],
                  goal: Union[tuple, list, np.array],
                  barriers: list = [],
                  x: Optional[str] = 'x',
                  y: Optional[str] = 'y',
                  connectivity: int = 8,
                  snap_start: bool = False,
                  snap_goal: bool = False) -> xr.DataArray:
    """
    Calculate distance from a starting point to a goal through a
    surface graph. Starting location and goal location should be within
    the graph.

    A* is a modification of Dijkstra’s Algorithm that is optimized for
    a single destination. Dijkstra’s Algorithm can find paths to all
    locations; A* finds paths to one location, or the closest of several
    locations. It prioritizes paths that seem to be leading closer to
    a goal.

    The output is an equal sized Xarray.DataArray with NaNs for non-path
    pixels, and the value of the path pixels being the current cost up
    to that point.

    Parameters
    ----------
    surface : xr.DataArray
        2D array of values to bin.
    start : array-like object of 2 numeric elements
        (y, x) or (lat, lon) coordinates of the starting point.
    goal : array like object of 2 numeric elements
        (y, x) or (lat, lon) coordinates of the goal location.
    barriers : array like object, default=[]
        List of values inside the surface which are barriers
        (cannot cross).
    x : str, default='x'
        Name of the x coordinate in input surface raster.
    y: str, default='x'
        Name of the y coordinate in input surface raster.
    connectivity : int, default=8
    snap_start: bool, default=False
        Snap the start location to the nearest valid value before
        beginning pathfinding.
    snap_goal: bool, default=False
        Snap the goal location to the nearest valid value before
        beginning pathfinding.

    Returns
    -------
    path_agg: xr.DataArray of the same type as `surface`.
        2D array of pathfinding values.
        All other input attributes are preserved.

    References
    ----------
        - Red Blob Games: https://www.redblobgames.com/pathfinding/a-star/implementation.html  # noqa
        - Nicholas Swift: https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2  # noqa

    Examples
    --------
    ... sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import a_star_search
        >>> agg = xr.DataArray(np.array([
        ...     [0, 1, 0, 0],
        ...     [1, 1, 0, 0],
        ...     [0, 1, 2, 2],
        ...     [1, 0, 2, 0],
        ...     [0, 2, 2, 2]
        ... ]), dims=['lat', 'lon'])
        >>> height, width = agg.shape
        >>> _lon = np.linspace(0, width - 1, width)
        >>> _lat = np.linspace(height - 1, 0, height)
        >>> agg['lon'] = _lon
        >>> agg['lat'] = _lat

        >>> barriers = [0]  # set pixels with value 0 as barriers
        >>> start = (3, 0)
        >>> goal = (0, 1)
        >>> path_agg = a_star_search(agg, start, goal, barriers, 'lon', 'lat')
        >>> print(path_agg)
        <xarray.DataArray (lat: 5, lon: 4)>
        array([[       nan,        nan,        nan,        nan],
               [0.        ,        nan,        nan,        nan],
               [       nan, 1.41421356,        nan,        nan],
               [       nan,        nan, 2.82842712,        nan],
               [       nan, 4.24264069,        nan,        nan]])
        Coordinates:
          * lon      (lon) float64 0.0 1.0 2.0 3.0
          * lat      (lat) float64 4.0 3.0 2.0 1.0 0.0
    """

    if surface.ndim != 2:
        raise ValueError("input `surface` must be 2D")

    if surface.dims != (y, x):
        raise ValueError("`surface.coords` should be named as coordinates:"
                         "({}, {})".format(y, x))

    if connectivity != 4 and connectivity != 8:
        raise ValueError("Use either 4 or 8-connectivity.")

    # convert starting and ending point from geo coords to pixel coords
    start_py, start_px = _get_pixel_id(start, surface, x, y)
    goal_py, goal_px = _get_pixel_id(goal, surface, x, y)

    h, w = surface.shape
    # validate start and goal locations are in the graph
    if not _is_inside(start_py, start_px, h, w):
        raise ValueError("start location outside the surface graph.")

    if not _is_inside(goal_py, goal_px, h, w):
        raise ValueError("goal location outside the surface graph.")

    barriers = np.array(barriers)

    if snap_start:
        # find nearest valid pixel to the start location
        start_py, start_px = _find_nearest_pixel(
            start_py, start_px, surface.data, barriers
        )
    if _is_not_crossable(surface.data[start_py, start_px], barriers):
        warnings.warn("Start at a non crossable location", Warning)

    if snap_goal:
        # find nearest valid pixel to the goal location
        goal_py, goal_px = _find_nearest_pixel(
            goal_py, goal_px, surface.data, barriers
        )
    if _is_not_crossable(surface.data[goal_py, goal_px], barriers):
        warnings.warn("End at a non crossable location", Warning)

    # 2d output image that stores the path
    path_img = np.zeros_like(surface, dtype=np.float64)
    # first, initialize all cells as np.nans
    path_img[:] = np.nan

    if start_py != NONE:
        neighbor_ys, neighbor_xs = _neighborhood_structure(connectivity)
        _a_star_search(surface.data, path_img, start_py, start_px,
                       goal_py, goal_px, barriers, neighbor_ys, neighbor_xs)

    path_agg = xr.DataArray(path_img,
                            coords=surface.coords,
                            dims=surface.dims,
                            attrs=surface.attrs)

    return path_agg
