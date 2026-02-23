"""Cost-distance (weighted proximity) via multi-source Dijkstra.

Computes the minimum accumulated traversal cost through a friction surface
to reach the nearest target pixel.  This is the raster equivalent of
GRASS ``r.cost`` / ArcGIS *Cost Distance*.

Algorithm
---------
Multi-source Dijkstra with a numba-friendly binary min-heap:

1. All source (target) pixels are seeded at cost 0.
2. Pop the minimum-cost pixel, relax 4- or 8-connected neighbours.
3. Edge cost = geometric_distance * average_friction of the two endpoints.
4. Repeat until the heap is empty or ``max_cost`` is exceeded.

Dask strategy
-------------
For finite ``max_cost``, the maximum pixel radius any cost-path can reach
is ``max_cost / (f_min * cellsize)`` where *f_min* is the global minimum
friction (a tiny ``.compute()``).  This radius becomes the ``depth``
parameter to ``dask.array.map_overlap``, giving **exact** results within
the cost budget.

If ``max_cost`` is infinite or the implied radius exceeds half the raster,
fall back to single-chunk mode (same trade-off as ``proximity()``).
"""

from __future__ import annotations

from functools import partial
from math import sqrt

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.utils import get_dataarray_resolution, ngjit
from xrspatial.dataset_support import supports_dataset

# ---------------------------------------------------------------------------
# Numba binary min-heap (three parallel arrays: keys, rows, cols)
# ---------------------------------------------------------------------------

@ngjit
def _heap_push(keys, rows, cols, size, key, row, col):
    """Push (key, row, col) onto the heap.  Returns new size."""
    pos = size
    keys[pos] = key
    rows[pos] = row
    cols[pos] = col
    size += 1
    # sift up
    while pos > 0:
        parent = (pos - 1) >> 1
        if keys[parent] > keys[pos]:
            # swap
            keys[parent], keys[pos] = keys[pos], keys[parent]
            rows[parent], rows[pos] = rows[pos], rows[parent]
            cols[parent], cols[pos] = cols[pos], cols[parent]
            pos = parent
        else:
            break
    return size


@ngjit
def _heap_pop(keys, rows, cols, size):
    """Pop minimum element.  Returns (key, row, col, new_size)."""
    key = keys[0]
    row = rows[0]
    col = cols[0]
    size -= 1
    # move last to root
    keys[0] = keys[size]
    rows[0] = rows[size]
    cols[0] = cols[size]
    # sift down
    pos = 0
    while True:
        child = 2 * pos + 1
        if child >= size:
            break
        # pick smaller child
        if child + 1 < size and keys[child + 1] < keys[child]:
            child += 1
        if keys[child] < keys[pos]:
            keys[pos], keys[child] = keys[child], keys[pos]
            rows[pos], rows[child] = rows[child], rows[pos]
            cols[pos], cols[child] = cols[child], cols[pos]
            pos = child
        else:
            break
    return key, row, col, size


# ---------------------------------------------------------------------------
# Multi-source Dijkstra kernel
# ---------------------------------------------------------------------------

@ngjit
def _cost_distance_kernel(
    source_data,
    friction_data,
    height,
    width,
    cellsize_x,
    cellsize_y,
    max_cost,
    target_values,
    dy,
    dx,
    dd,
):
    """Run multi-source Dijkstra and return float32 cost-distance array.

    Parameters
    ----------
    source_data : 2-D array
        Source raster (targets are non-zero finite, or in *target_values*).
    friction_data : 2-D array
        Friction surface.  NaN or <= 0 means impassable.
    height, width : int
    cellsize_x, cellsize_y : float
    max_cost : float
    target_values : 1-D array
        Specific pixel values to treat as targets (empty ⇒ all non-zero
        finite pixels).
    dy, dx : 1-D int arrays
        Neighbour offsets (length = connectivity).
    dd : 1-D float array
        Geometric distance for each neighbour direction.
    """
    n_values = len(target_values)
    n_neighbors = len(dy)

    # output: initialise to NaN (unreachable)
    dist = np.full((height, width), np.inf, dtype=np.float64)

    # Heap arrays — worst-case each pixel is pushed once per neighbour
    # but practically much less.  We allocate height*width which is
    # sufficient for an exact Dijkstra (each pixel settled at most once).
    max_heap = height * width
    h_keys = np.empty(max_heap, dtype=np.float64)
    h_rows = np.empty(max_heap, dtype=np.int64)
    h_cols = np.empty(max_heap, dtype=np.int64)
    h_size = 0

    visited = np.zeros((height, width), dtype=np.int8)

    # Seed all source pixels
    for r in range(height):
        for c in range(width):
            val = source_data[r, c]
            is_target = False
            if n_values == 0:
                if val != 0.0 and np.isfinite(val):
                    is_target = True
            else:
                for k in range(n_values):
                    if val == target_values[k]:
                        is_target = True
                        break
            if is_target:
                # source must also be passable
                f = friction_data[r, c]
                if np.isfinite(f) and f > 0.0:
                    dist[r, c] = 0.0
                    h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                        0.0, r, c)

    # Dijkstra main loop
    while h_size > 0:
        cost_u, ur, uc, h_size = _heap_pop(h_keys, h_rows, h_cols, h_size)

        if visited[ur, uc]:
            continue
        visited[ur, uc] = 1

        if cost_u > max_cost:
            break

        f_u = friction_data[ur, uc]

        for i in range(n_neighbors):
            vr = ur + dy[i]
            vc = uc + dx[i]
            if vr < 0 or vr >= height or vc < 0 or vc >= width:
                continue
            if visited[vr, vc]:
                continue

            f_v = friction_data[vr, vc]
            # impassable if NaN or non-positive friction
            if not (np.isfinite(f_v) and f_v > 0.0):
                continue

            edge_cost = dd[i] * (f_u + f_v) * 0.5
            new_cost = cost_u + edge_cost

            if new_cost < dist[vr, vc]:
                dist[vr, vc] = new_cost
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    new_cost, vr, vc)

    # Convert unreachable / over-budget to NaN, cast to float32
    out = np.empty((height, width), dtype=np.float32)
    for r in range(height):
        for c in range(width):
            d = dist[r, c]
            if d == np.inf or d > max_cost:
                out[r, c] = np.nan
            else:
                out[r, c] = np.float32(d)
    return out


# ---------------------------------------------------------------------------
# NumPy wrapper
# ---------------------------------------------------------------------------

def _cost_distance_numpy(source_data, friction_data, cellsize_x, cellsize_y,
                         max_cost, target_values, dy, dx, dd):
    height, width = source_data.shape
    return _cost_distance_kernel(
        source_data, friction_data, height, width,
        cellsize_x, cellsize_y, max_cost,
        target_values, dy, dx, dd,
    )


# ---------------------------------------------------------------------------
# Dask wrapper
# ---------------------------------------------------------------------------

def _make_chunk_func(cellsize_x, cellsize_y, max_cost, target_values,
                     dy, dx, dd):
    """Return a function suitable for ``da.map_overlap`` over two arrays."""

    def _chunk(source_block, friction_block):
        h, w = source_block.shape
        return _cost_distance_kernel(
            source_block, friction_block, h, w,
            cellsize_x, cellsize_y, max_cost,
            target_values, dy, dx, dd,
        )

    return _chunk


def _cost_distance_dask(source_da, friction_da, cellsize_x, cellsize_y,
                        max_cost, target_values, dy, dx, dd):
    """Dask path: use map_overlap with depth derived from max_cost."""

    # We need the global minimum friction to compute max pixel radius.
    # This is a tiny scalar .compute().
    # Use da.where to avoid boolean indexing (which creates unknown chunks).
    positive_friction = da.where(friction_da > 0, friction_da, np.inf)
    f_min = da.nanmin(positive_friction).compute()
    if not np.isfinite(f_min) or f_min <= 0:
        # All friction is non-positive or NaN — nothing reachable
        return da.full(source_da.shape, np.nan, dtype=np.float32,
                       chunks=source_da.chunks)

    min_cellsize = min(abs(cellsize_x), abs(cellsize_y))
    max_radius = max_cost / (float(f_min) * min_cellsize)

    height, width = source_da.shape
    max_dim = max(height, width)

    pad = int(max_radius + 1) if np.isfinite(max_radius) else max_dim

    if not np.isfinite(max_radius) or pad >= height or pad >= width:
        # Fall back to single-chunk when depth would exceed array size
        source_da = source_da.rechunk({0: height, 1: width})
        friction_da = friction_da.rechunk({0: height, 1: width})
        pad_y = pad_x = 0
    else:
        pad_y = pad
        pad_x = pad

    chunk_func = _make_chunk_func(
        cellsize_x, cellsize_y, max_cost, target_values, dy, dx, dd,
    )

    out = da.map_overlap(
        chunk_func,
        source_da, friction_da,
        depth=(pad_y, pad_x),
        boundary=np.nan,
        dtype=np.float32,
        meta=np.array((), dtype=np.float32),
    )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@supports_dataset
def cost_distance(
    raster: xr.DataArray,
    friction: xr.DataArray,
    x: str = "x",
    y: str = "y",
    target_values: list = [],
    max_cost: float = np.inf,
    connectivity: int = 8,
) -> xr.DataArray:
    """Compute accumulated cost-distance through a friction surface.

    For every pixel, computes the minimum accumulated traversal cost
    to reach the nearest target pixel, where traversal cost along each
    edge equals ``geometric_distance * mean_friction_of_endpoints``.

    Parameters
    ----------
    raster : xr.DataArray or xr.Dataset
        2-D source raster.  Target pixels are identified by non-zero
        finite values (or values in *target_values*).
    friction : xr.DataArray
        2-D friction (cost) surface.  Must have the same shape and
        coordinates as *raster*.  Values must be positive and finite
        for passable cells; NaN or ``<= 0`` marks impassable barriers.
    x : str, default='x'
        Name of the x coordinate.
    y : str, default='y'
        Name of the y coordinate.
    target_values : list, optional
        Specific pixel values in *raster* to treat as sources.
        If empty, all non-zero finite pixels are sources.
    max_cost : float, default=np.inf
        Maximum accumulated cost.  Pixels whose least-cost path exceeds
        this budget are set to NaN.  A finite value enables efficient
        Dask parallelisation via ``map_overlap``.
    connectivity : int, default=8
        Pixel connectivity: 4 (cardinal only) or 8 (cardinal + diagonal).

    Returns
    -------
    xr.DataArray or xr.Dataset
        2-D array of accumulated cost-distance values (float32).
        Source pixels have cost 0.  Unreachable pixels are NaN.
    """
    # --- validation ---
    if raster.ndim != 2:
        raise ValueError("raster must be 2-D")
    if friction.ndim != 2:
        raise ValueError("friction must be 2-D")
    if raster.shape != friction.shape:
        raise ValueError("raster and friction must have the same shape")
    if raster.dims != (y, x):
        raise ValueError(
            f"raster.dims should be ({y!r}, {x!r}), got {raster.dims}"
        )
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    cellsize_x, cellsize_y = get_dataarray_resolution(raster)
    cellsize_x = abs(float(cellsize_x))
    cellsize_y = abs(float(cellsize_y))

    target_values = np.asarray(target_values, dtype=np.float64)
    max_cost_f = float(max_cost)

    # Build neighbour offsets and geometric distances
    if connectivity == 8:
        dy = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
        dx = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
        dd = np.array([
            sqrt(cellsize_y**2 + cellsize_x**2),   # (-1,-1)
            cellsize_y,                              # (-1, 0)
            sqrt(cellsize_y**2 + cellsize_x**2),   # (-1,+1)
            cellsize_x,                              # ( 0,-1)
            cellsize_x,                              # ( 0,+1)
            sqrt(cellsize_y**2 + cellsize_x**2),   # (+1,-1)
            cellsize_y,                              # (+1, 0)
            sqrt(cellsize_y**2 + cellsize_x**2),   # (+1,+1)
        ], dtype=np.float64)
    else:
        dy = np.array([0, -1, 1, 0], dtype=np.int64)
        dx = np.array([-1, 0, 0, 1], dtype=np.int64)
        dd = np.array([cellsize_x, cellsize_y, cellsize_y, cellsize_x],
                      dtype=np.float64)

    # Ensure friction chunks match raster chunks for dask
    source_data = raster.data
    friction_data = friction.data

    if da is not None and isinstance(source_data, da.Array):
        # Rechunk friction to match raster
        if isinstance(friction_data, da.Array):
            friction_data = friction_data.rechunk(source_data.chunks)
        else:
            friction_data = da.from_array(friction_data,
                                          chunks=source_data.chunks)

    if isinstance(source_data, np.ndarray):
        if isinstance(friction_data, np.ndarray):
            result_data = _cost_distance_numpy(
                source_data, friction_data,
                cellsize_x, cellsize_y, max_cost_f,
                target_values, dy, dx, dd,
            )
        else:
            raise TypeError("friction must be numpy-backed when raster is")
    elif da is not None and isinstance(source_data, da.Array):
        result_data = _cost_distance_dask(
            source_data, friction_data,
            cellsize_x, cellsize_y, max_cost_f,
            target_values, dy, dx, dd,
        )
    else:
        raise TypeError(f"Unsupported array type: {type(source_data)}")

    return xr.DataArray(
        result_data,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )
