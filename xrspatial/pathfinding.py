import heapq
import warnings
from collections import OrderedDict
from math import sqrt
from typing import Optional, Union

import numpy as np
import xarray as xr

try:
    import dask.array as da
    import dask
except ImportError:
    da = None
    dask = None

from xrspatial.cost_distance import _heap_push, _heap_pop
from xrspatial.utils import (
    get_dataarray_resolution, ngjit,
    has_cuda_and_cupy, is_cupy_array, is_dask_cupy, has_dask_array,
)

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


def _is_not_crossable_py(cell_value, barriers):
    """Pure Python version of _is_not_crossable for the dask A* loop."""
    if np.isnan(cell_value):
        return True
    for b in barriers:
        if cell_value == b:
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


# ---------------------------------------------------------------------------
# LRU chunk cache for dask A*
# ---------------------------------------------------------------------------

class _ChunkCache:
    """OrderedDict-based LRU cache for dask chunks."""

    def __init__(self, maxsize=128):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def get(self, key, loader):
        """Return cached chunk or call *loader()*, evicting oldest if full."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        value = loader()
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = value
        return value


# ---------------------------------------------------------------------------
# Sparse dask A*
# ---------------------------------------------------------------------------

def _a_star_dask(surface_da, friction_da, start_py, start_px,
                 goal_py, goal_px, barriers, dy, dx, dd,
                 f_min, use_friction, cellsize_x, cellsize_y, is_cupy):
    """Run A* on a dask-backed array, loading chunks on demand.

    Returns a list of (row, col, cost) tuples for path pixels,
    or [] if no path exists.
    """
    height, width = surface_da.shape
    n_neighbors = len(dy)

    # Chunk boundaries (cumulative sums of chunk sizes)
    row_chunks = np.array(surface_da.chunks[0])
    col_chunks = np.array(surface_da.chunks[1])
    row_bounds = np.cumsum(row_chunks)
    col_bounds = np.cumsum(col_chunks)

    surface_cache = _ChunkCache()
    friction_cache = _ChunkCache() if use_friction else None

    def _load_chunk(da_arr, cache, iy, ix):
        """Load and cache a single chunk, converting cupy->numpy."""
        def loader():
            block = da_arr.blocks[iy, ix].compute()
            if is_cupy:
                block = block.get()
            return np.asarray(block, dtype=np.float64)
        return cache.get((iy, ix), loader)

    def _get_value(da_arr, cache, r, c):
        """Get a scalar value at global (r, c) via chunk cache."""
        iy = int(np.searchsorted(row_bounds, r, side='right'))
        ix = int(np.searchsorted(col_bounds, c, side='right'))
        chunk = _load_chunk(da_arr, cache, iy, ix)
        local_r = r - (int(row_bounds[iy - 1]) if iy > 0 else 0)
        local_c = c - (int(col_bounds[ix - 1]) if ix > 0 else 0)
        return float(chunk[local_r, local_c])

    # Check start
    start_val = _get_value(surface_da, surface_cache, start_py, start_px)
    if _is_not_crossable_py(start_val, barriers):
        return []

    if use_friction:
        f_start = _get_value(friction_da, friction_cache, start_py, start_px)
        if not (np.isfinite(f_start) and f_start > 0.0):
            return []

    # A* data structures (sparse — dict/set, not full arrays)
    g_cost = {(start_py, start_px): 0.0}
    parent = {}
    visited = set()

    counter = 0  # tie-breaker for stable heap ordering

    # Heuristic for start
    dy_goal = abs(start_py - goal_py) * cellsize_y
    dx_goal = abs(start_px - goal_px) * cellsize_x
    h = sqrt(dy_goal ** 2 + dx_goal ** 2)
    if use_friction:
        h *= f_min

    heap = [(h, counter, start_py, start_px)]

    while heap:
        f_u, _, py, px = heapq.heappop(heap)

        if (py, px) in visited:
            continue
        visited.add((py, px))

        # Found goal — reconstruct path
        if py == goal_py and px == goal_px:
            path = []
            cr, cc = goal_py, goal_px
            path.append((cr, cc, g_cost[(cr, cc)]))
            while (cr, cc) in parent:
                cr, cc = parent[(cr, cc)]
                path.append((cr, cc, g_cost[(cr, cc)]))
            return path

        g_u = g_cost[(py, px)]

        for i in range(n_neighbors):
            ny = py + int(dy[i])
            nx = px + int(dx[i])

            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if (ny, nx) in visited:
                continue

            n_val = _get_value(surface_da, surface_cache, ny, nx)
            if _is_not_crossable_py(n_val, barriers):
                continue

            if use_friction:
                f_u_val = _get_value(friction_da, friction_cache, py, px)
                f_v_val = _get_value(friction_da, friction_cache, ny, nx)
                if not (np.isfinite(f_v_val) and f_v_val > 0.0):
                    continue
                edge_cost = float(dd[i]) * (f_u_val + f_v_val) * 0.5
            else:
                edge_cost = float(dd[i])

            new_g = g_u + edge_cost

            if new_g < g_cost.get((ny, nx), float('inf')):
                g_cost[(ny, nx)] = new_g
                parent[(ny, nx)] = (py, px)

                dy_goal = abs(ny - goal_py) * cellsize_y
                dx_goal = abs(nx - goal_px) * cellsize_x
                h = sqrt(dy_goal ** 2 + dx_goal ** 2)
                if use_friction:
                    h *= f_min

                counter += 1
                heapq.heappush(heap, (new_g + h, counter, ny, nx))

    return []  # no path


# ---------------------------------------------------------------------------
# Sparse path → lazy dask output
# ---------------------------------------------------------------------------

def _path_to_dask_array(path_pixels, shape, chunks, is_cupy):
    """Convert sparse path list to a lazy dask array of the original shape.

    *path_pixels* is a list of ``(row, col, cost)`` tuples.
    Non-path pixels are NaN.
    """
    height, width = shape
    row_chunks = chunks[0]
    col_chunks = chunks[1]
    row_bounds = np.cumsum(row_chunks)
    col_bounds = np.cumsum(col_chunks)

    # Group path pixels by chunk
    chunk_pixels = {}  # {(iy, ix): [(local_r, local_c, cost), ...]}
    for r, c, cost in path_pixels:
        iy = int(np.searchsorted(row_bounds, r, side='right'))
        ix = int(np.searchsorted(col_bounds, c, side='right'))
        local_r = r - (int(row_bounds[iy - 1]) if iy > 0 else 0)
        local_c = c - (int(col_bounds[ix - 1]) if ix > 0 else 0)
        chunk_pixels.setdefault((iy, ix), []).append(
            (local_r, local_c, cost))

    n_row_chunks = len(row_chunks)
    n_col_chunks = len(col_chunks)

    if is_cupy:
        import cupy

        @dask.delayed
        def _make_block_cupy(ch, cw, pixels):
            block = np.full((ch, cw), np.nan, dtype=np.float64)
            for lr, lc, cost in pixels:
                block[lr, lc] = cost
            return cupy.asarray(block)

        blocks = []
        for iy in range(n_row_chunks):
            row = []
            for ix in range(n_col_chunks):
                ch = int(row_chunks[iy])
                cw = int(col_chunks[ix])
                pixels = chunk_pixels.get((iy, ix), [])
                row.append(
                    da.from_delayed(
                        _make_block_cupy(ch, cw, pixels),
                        shape=(ch, cw),
                        dtype=np.float64,
                        meta=cupy.array((), dtype=np.float64),
                    )
                )
            blocks.append(row)
    else:
        @dask.delayed
        def _make_block(ch, cw, pixels):
            block = np.full((ch, cw), np.nan, dtype=np.float64)
            for lr, lc, cost in pixels:
                block[lr, lc] = cost
            return block

        blocks = []
        for iy in range(n_row_chunks):
            row = []
            for ix in range(n_col_chunks):
                ch = int(row_chunks[iy])
                cw = int(col_chunks[ix])
                pixels = chunk_pixels.get((iy, ix), [])
                row.append(
                    da.from_delayed(
                        _make_block(ch, cw, pixels),
                        shape=(ch, cw),
                        dtype=np.float64,
                        meta=np.array((), dtype=np.float64),
                    )
                )
            blocks.append(row)

    return da.block(blocks)


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

    **Backend support**

    =============  ===========================================================
    Backend        Strategy
    =============  ===========================================================
    NumPy          Numba-jitted kernel (fast, in-memory)
    Dask           Sparse Python A* with LRU chunk cache — loads chunks on
                   demand so the full grid never needs to fit in RAM
    CuPy           CPU fallback (transfers to numpy, runs numba kernel,
                   transfers back)
    Dask + CuPy    Same sparse A* as Dask, with cupy→numpy chunk conversion
    =============  ===========================================================

    ``snap_start`` and ``snap_goal`` are not supported with Dask-backed
    arrays (raises ``ValueError``).

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

    # Detect backend
    surface_data = surface.data
    _is_dask = da is not None and isinstance(surface_data, da.Array)
    _is_cupy_backend = (
        not _is_dask
        and has_cuda_and_cupy()
        and is_cupy_array(surface_data)
    )
    _is_dask_cupy = _is_dask and has_cuda_and_cupy() and is_dask_cupy(surface)

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

    # --- Snap / crossability checks ---
    if _is_dask:
        # Snapping requires O(h*w) scan — not supported for dask
        if snap_start:
            raise ValueError(
                "snap_start is not supported with dask-backed arrays; "
                "ensure the start pixel is valid before calling a_star_search"
            )
        if snap_goal:
            raise ValueError(
                "snap_goal is not supported with dask-backed arrays; "
                "ensure the goal pixel is valid before calling a_star_search"
            )
        # Single-pixel crossability check via .compute()
        start_val = float(surface_data[start_py, start_px].compute())
        if _is_not_crossable_py(start_val, barriers):
            warnings.warn("Start at a non crossable location", Warning)
        goal_val = float(surface_data[goal_py, goal_px].compute())
        if _is_not_crossable_py(goal_val, barriers):
            warnings.warn("End at a non crossable location", Warning)
    elif _is_cupy_backend:
        # CuPy: use .get() for scalar access in numpy-land
        surface_np = surface_data.get()
        if snap_start:
            start_py, start_px = _find_nearest_pixel(
                start_py, start_px, surface_np, barriers
            )
        if _is_not_crossable(surface_np[start_py, start_px], barriers):
            warnings.warn("Start at a non crossable location", Warning)
        if snap_goal:
            goal_py, goal_px = _find_nearest_pixel(
                goal_py, goal_px, surface_np, barriers
            )
        if _is_not_crossable(surface_np[goal_py, goal_px], barriers):
            warnings.warn("End at a non crossable location", Warning)
    else:
        # numpy path
        if snap_start:
            start_py, start_px = _find_nearest_pixel(
                start_py, start_px, surface_data, barriers
            )
        if _is_not_crossable(surface_data[start_py, start_px], barriers):
            warnings.warn("Start at a non crossable location", Warning)
        if snap_goal:
            goal_py, goal_px = _find_nearest_pixel(
                goal_py, goal_px, surface_data, barriers
            )
        if _is_not_crossable(surface_data[goal_py, goal_px], barriers):
            warnings.warn("End at a non crossable location", Warning)

    # Build neighborhood with cellsize-scaled distances
    dy, dx, dd = _neighborhood_structure(cellsize_x, cellsize_y, connectivity)

    # --- Handle friction ---
    if friction is not None:
        if friction.shape != surface.shape:
            raise ValueError("friction must have the same shape as surface")
        use_friction = True
    else:
        use_friction = False

    # --- Backend dispatch ---
    if _is_dask:
        # Dask or dask+cupy path
        if use_friction:
            friction_data = friction.data
            # Rechunk friction to match surface if needed
            if isinstance(friction_data, da.Array):
                friction_data = friction_data.rechunk(surface_data.chunks)
            else:
                friction_data = da.from_array(
                    friction_data, chunks=surface_data.chunks)
            # f_min via dask (same pattern as cost_distance)
            positive_friction = da.where(
                friction_data > 0, friction_data, np.inf)
            f_min = float(da.nanmin(positive_friction).compute())
            if not (np.isfinite(f_min) and f_min > 0):
                raise ValueError("friction has no positive finite values")
        else:
            friction_data = None
            f_min = 1.0

        path_pixels = _a_star_dask(
            surface_data, friction_data,
            start_py, start_px, goal_py, goal_px,
            barriers, dy, dx, dd,
            f_min, use_friction, cellsize_x, cellsize_y,
            _is_dask_cupy,
        )
        path_data = _path_to_dask_array(
            path_pixels, surface.shape, surface_data.chunks, _is_dask_cupy)

    elif _is_cupy_backend:
        import cupy
        # Transfer to CPU, run numpy kernel, transfer back
        if 'surface_np' not in dir():
            surface_np = surface_data.get()
        if use_friction:
            friction_np = np.asarray(friction.data.get(), dtype=np.float64)
            mask = np.isfinite(friction_np) & (friction_np > 0)
            if not np.any(mask):
                raise ValueError("friction has no positive finite values")
            f_min = float(np.min(friction_np[mask]))
        else:
            friction_np = np.ones((h, w), dtype=np.float64)
            f_min = 1.0

        path_img = np.full(surface.shape, np.nan, dtype=np.float64)
        if start_py != NONE:
            _a_star_search(surface_np, path_img, start_py, start_px,
                           goal_py, goal_px, barriers, dy, dx, dd,
                           friction_np, f_min, use_friction,
                           cellsize_x, cellsize_y)
        path_data = cupy.asarray(path_img)

    else:
        # numpy path (existing, unchanged)
        if use_friction:
            friction_data = np.asarray(friction.data, dtype=np.float64)
            mask = np.isfinite(friction_data) & (friction_data > 0)
            if not np.any(mask):
                raise ValueError("friction has no positive finite values")
            f_min = float(np.min(friction_data[mask]))
        else:
            friction_data = np.ones((h, w), dtype=np.float64)
            f_min = 1.0

        path_img = np.full(surface.shape, np.nan, dtype=np.float64)
        if start_py != NONE:
            _a_star_search(surface_data, path_img, start_py, start_px,
                           goal_py, goal_px, barriers, dy, dx, dd,
                           friction_data, f_min, use_friction,
                           cellsize_x, cellsize_y)
        path_data = path_img

    path_agg = xr.DataArray(path_data,
                            coords=surface.coords,
                            dims=surface.dims,
                            attrs=surface.attrs)

    return path_agg
