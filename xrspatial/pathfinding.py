import warnings
from math import sqrt
from typing import Optional, Union

import numpy as np
import xarray as xr

from xrspatial.cost_distance import _heap_push, _heap_pop
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
def _find_nearest_pixel(py, px, data, barriers):
    # if the cell is already valid, return itself
    if not _is_not_crossable(data[py, px], barriers):
        return py, px

    height, width = data.shape
    # init min distance as max possible distance (pixel space)
    min_distance = np.sqrt(float((height - 1) ** 2 + (width - 1) ** 2))
    # return of the function
    nearest_y = NONE
    nearest_x = NONE
    for y in range(height):
        for x in range(width):
            if not _is_not_crossable(data[y, x], barriers):
                d = np.sqrt(float((x - px) ** 2 + (y - py) ** 2))
                if d < min_distance:
                    min_distance = d
                    nearest_y = y
                    nearest_x = x

    return nearest_y, nearest_x


def _neighborhood_structure(cellsize_x, cellsize_y, connectivity=8):
    """Return (dy, dx, dd) with cellsize-scaled geometric distances."""
    if connectivity == 8:
        dy = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
        dx = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
        dd = np.array([
            sqrt(cellsize_y ** 2 + cellsize_x ** 2),   # (-1,-1)
            cellsize_y,                                  # (-1, 0)
            sqrt(cellsize_y ** 2 + cellsize_x ** 2),   # (-1,+1)
            cellsize_x,                                  # ( 0,-1)
            cellsize_x,                                  # ( 0,+1)
            sqrt(cellsize_y ** 2 + cellsize_x ** 2),   # (+1,-1)
            cellsize_y,                                  # (+1, 0)
            sqrt(cellsize_y ** 2 + cellsize_x ** 2),   # (+1,+1)
        ], dtype=np.float64)
    else:
        dy = np.array([0, -1, 1, 0], dtype=np.int64)
        dx = np.array([-1, 0, 0, 1], dtype=np.int64)
        dd = np.array([cellsize_x, cellsize_y, cellsize_y, cellsize_x],
                      dtype=np.float64)
    return dy, dx, dd


@ngjit
def _reconstruct_path(path_img, parent_ys, parent_xs, g_cost,
                      start_py, start_px, goal_py, goal_px):
    # construct path output image as a 2d array with NaNs for non-path pixels,
    # and the value of the path pixels being the g-cost up to that point
    current_x = goal_px
    current_y = goal_py

    if parent_xs[current_y, current_x] != NONE and \
            parent_ys[current_y, current_x] != NONE:
        # exist path from start to goal
        # add cost at start
        path_img[start_py, start_px] = g_cost[start_py, start_px]
        # add cost along the path
        while current_x != start_px or current_y != start_py:
            # value of a path pixel is the cost up to that point
            path_img[current_y, current_x] = g_cost[current_y, current_x]
            parent_y = parent_ys[current_y, current_x]
            parent_x = parent_xs[current_y, current_x]
            current_y = parent_y
            current_x = parent_x
    return


@ngjit
def _a_star_search(data, path_img, start_py, start_px, goal_py, goal_px,
                   barriers, dy, dx, dd, friction, f_min, use_friction,
                   cellsize_x, cellsize_y):

    height, width = data.shape
    n_neighbors = len(dy)

    # parent of the (i, j) pixel is the pixel at
    # (parent_ys[i, j], parent_xs[i, j])
    parent_ys = np.ones((height, width), dtype=np.int64) * NONE
    parent_xs = np.ones((height, width), dtype=np.int64) * NONE

    # parent of start is itself
    parent_ys[start_py, start_px] = start_py
    parent_xs[start_py, start_px] = start_px

    # g-cost: distance from start to the current node
    g_cost = np.full((height, width), np.inf, dtype=np.float64)

    visited = np.zeros((height, width), dtype=np.int8)

    # Heap arrays
    max_heap = height * width
    h_keys = np.empty(max_heap, dtype=np.float64)
    h_rows = np.empty(max_heap, dtype=np.int64)
    h_cols = np.empty(max_heap, dtype=np.int64)
    h_size = 0

    if not _is_not_crossable(data[start_py, start_px], barriers):
        # Check friction at start when using friction
        if use_friction:
            f_start_val = friction[start_py, start_px]
            if not (np.isfinite(f_start_val) and f_start_val > 0.0):
                return

        g_cost[start_py, start_px] = 0.0

        # Compute heuristic for start
        dy_goal = abs(start_py - goal_py) * cellsize_y
        dx_goal = abs(start_px - goal_px) * cellsize_x
        h = np.sqrt(dy_goal ** 2 + dx_goal ** 2)
        if use_friction:
            h *= f_min

        h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                            h, start_py, start_px)

    while h_size > 0:
        f_u, py, px, h_size = _heap_pop(h_keys, h_rows, h_cols, h_size)

        if visited[py, px]:
            continue
        visited[py, px] = 1

        # found the goal
        if py == goal_py and px == goal_px:
            _reconstruct_path(path_img, parent_ys, parent_xs,
                              g_cost, start_py, start_px,
                              goal_py, goal_px)
            return

        g_u = g_cost[py, px]

        # visit neighborhood
        for i in range(n_neighbors):
            ny = py + dy[i]
            nx = px + dx[i]

            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if visited[ny, nx]:
                continue
            if _is_not_crossable(data[ny, nx], barriers):
                continue

            # Compute edge cost
            if use_friction:
                f_u_val = friction[py, px]
                f_v_val = friction[ny, nx]
                # impassable if friction is NaN or non-positive
                if not (np.isfinite(f_v_val) and f_v_val > 0.0):
                    continue
                edge_cost = dd[i] * (f_u_val + f_v_val) * 0.5
            else:
                edge_cost = dd[i]

            new_g = g_u + edge_cost

            if new_g < g_cost[ny, nx]:
                g_cost[ny, nx] = new_g
                parent_ys[ny, nx] = py
                parent_xs[ny, nx] = px

                # Compute heuristic
                dy_goal = abs(ny - goal_py) * cellsize_y
                dx_goal = abs(nx - goal_px) * cellsize_x
                h = np.sqrt(dy_goal ** 2 + dx_goal ** 2)
                if use_friction:
                    h *= f_min

                f_val = new_g + h
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    f_val, ny, nx)

    return


def a_star_search(surface: xr.DataArray,
                  start: Union[tuple, list, np.array],
                  goal: Union[tuple, list, np.array],
                  barriers: list = [],
                  x: Optional[str] = 'x',
                  y: Optional[str] = 'y',
                  connectivity: int = 8,
                  snap_start: bool = False,
                  snap_goal: bool = False,
                  friction: xr.DataArray = None) -> xr.DataArray:
    """
    Calculate the least-cost path from a starting point to a goal through
    a surface graph, optionally weighted by a friction surface.

    A* is a modification of Dijkstra's Algorithm that is optimized for
    a single destination. It prioritizes paths that seem to be leading
    closer to a goal using an admissible heuristic.

    When a friction surface is provided, edge costs are
    ``geometric_distance * mean_friction_of_endpoints``, matching the
    cost model used by :func:`cost_distance`.  The heuristic is scaled
    by the minimum friction value to remain admissible.

    The output is an equal-sized ``xr.DataArray`` with NaN for non-path
    pixels and the accumulated cost at each path pixel.

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
    y: str, default='y'
        Name of the y coordinate in input surface raster.
    connectivity : int, default=8
    snap_start: bool, default=False
        Snap the start location to the nearest valid value before
        beginning pathfinding.
    snap_goal: bool, default=False
        Snap the goal location to the nearest valid value before
        beginning pathfinding.
    friction : xr.DataArray, optional
        2-D friction (cost) surface.  Must have the same shape as
        *surface*.  Values must be positive and finite for passable
        cells; NaN or ``<= 0`` marks impassable barriers.  When
        provided, edge costs become
        ``geometric_distance * mean_friction_of_endpoints``.

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
    """

    if surface.ndim != 2:
        raise ValueError("input `surface` must be 2D")

    if surface.dims != (y, x):
        raise ValueError("`surface.coords` should be named as coordinates:"
                         "({}, {})".format(y, x))

    if connectivity != 4 and connectivity != 8:
        raise ValueError("Use either 4 or 8-connectivity.")

    # compute cellsize
    cellsize_x, cellsize_y = get_dataarray_resolution(surface, x, y)
    cellsize_x = abs(float(cellsize_x))
    cellsize_y = abs(float(cellsize_y))

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

    # Handle friction
    if friction is not None:
        if friction.shape != surface.shape:
            raise ValueError("friction must have the same shape as surface")
        use_friction = True
        friction_data = np.asarray(friction.data, dtype=np.float64)
        # Compute f_min: minimum positive finite friction
        mask = np.isfinite(friction_data) & (friction_data > 0)
        if not np.any(mask):
            raise ValueError("friction has no positive finite values")
        f_min = float(np.min(friction_data[mask]))
    else:
        use_friction = False
        friction_data = np.ones((h, w), dtype=np.float64)
        f_min = 1.0

    # Build neighborhood with cellsize-scaled distances
    dy, dx, dd = _neighborhood_structure(cellsize_x, cellsize_y, connectivity)

    # 2d output image that stores the path
    path_img = np.zeros_like(surface.data, dtype=np.float64)
    # first, initialize all cells as np.nans
    path_img[:] = np.nan

    if start_py != NONE:
        _a_star_search(surface.data, path_img, start_py, start_px,
                       goal_py, goal_px, barriers, dy, dx, dd,
                       friction_data, f_min, use_friction,
                       cellsize_x, cellsize_y)

    path_agg = xr.DataArray(path_img,
                            coords=surface.coords,
                            dims=surface.dims,
                            attrs=surface.attrs)

    return path_agg
