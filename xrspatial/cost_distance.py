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

If ``max_cost`` is infinite or the implied radius exceeds chunk dimensions,
an iterative boundary-only Dijkstra is used that processes tiles one at a
time, keeping memory usage bounded regardless of raster size.
"""

from __future__ import annotations

import math as _math
from functools import partial
from math import sqrt

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from numba import cuda

try:
    import cupy
except ImportError:
    class cupy:  # type: ignore[no-redef]
        ndarray = False

from xrspatial.utils import (
    _validate_raster,
    cuda_args, get_dataarray_resolution, ngjit,
    has_cuda_and_cupy, is_cupy_array, is_dask_cupy,
)
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
# CuPy GPU backend — iterative parallel relaxation (parallel Bellman-Ford)
# ---------------------------------------------------------------------------

@cuda.jit
def _cost_distance_relax_kernel(friction, dist, changed,
                                height, width,
                                dy, dx, dd, n_neighbors,
                                max_cost):
    """One relaxation pass: each pixel checks all neighbours for shorter paths.

    Iterate until *changed* stays 0.  Convergence is guaranteed because
    dist values only decrease and the graph has finite edge weights.
    """
    iy, ix = cuda.grid(2)
    if iy >= height or ix >= width:
        return

    f_u = friction[iy, ix]
    if not (_math.isfinite(f_u) and f_u > 0.0):
        return

    current = dist[iy, ix]
    best = current

    for k in range(n_neighbors):
        vy = iy + dy[k]
        vx = ix + dx[k]
        if vy < 0 or vy >= height or vx < 0 or vx >= width:
            continue

        d_v = dist[vy, vx]
        if d_v >= best:
            continue

        f_v = friction[vy, vx]
        if not (_math.isfinite(f_v) and f_v > 0.0):
            continue

        edge_cost = dd[k] * (f_u + f_v) * 0.5
        new_cost = d_v + edge_cost

        if new_cost < best:
            best = new_cost

    if best < current and best <= max_cost:
        dist[iy, ix] = best
        changed[0] = 1


def _cost_distance_cupy(source_data, friction_data, cellsize_x, cellsize_y,
                        max_cost, target_values, dy, dx, dd):
    """GPU cost-distance via iterative parallel relaxation.

    Each CUDA thread processes one pixel per iteration, checking all
    neighbours for shorter paths (parallel Bellman-Ford).  The wavefront
    advances at least one pixel per iteration, so convergence takes at
    most O(height + width) iterations.
    """
    import cupy as cp

    height, width = source_data.shape
    src = source_data.astype(cp.float64) if source_data.dtype != cp.float64 \
        else source_data
    fric = friction_data.astype(cp.float64) if friction_data.dtype != cp.float64 \
        else friction_data

    # Initialize all distances to inf
    dist = cp.full((height, width), cp.inf, dtype=cp.float64)

    # Find source pixels
    if len(target_values) == 0:
        source_mask = cp.isfinite(src) & (src != 0)
    else:
        source_mask = cp.isin(src, cp.asarray(target_values, dtype=cp.float64))
        source_mask &= cp.isfinite(src)

    # Only seed sources on passable terrain
    passable = cp.isfinite(fric) & (fric > 0)
    source_mask &= passable
    dist[source_mask] = 0.0

    if not cp.any(source_mask):
        return cp.full((height, width), cp.nan, dtype=cp.float32)

    # Transfer neighbor offsets to device
    dy_d = cp.asarray(dy, dtype=cp.int64)
    dx_d = cp.asarray(dx, dtype=cp.int64)
    dd_d = cp.asarray(dd, dtype=cp.float64)
    n_neighbors = len(dy)

    changed = cp.zeros(1, dtype=cp.int32)
    griddim, blockdim = cuda_args((height, width))

    max_iterations = height + width
    for _ in range(max_iterations):
        changed[0] = 0
        _cost_distance_relax_kernel[griddim, blockdim](
            fric, dist, changed,
            height, width,
            dy_d, dx_d, dd_d, n_neighbors,
            np.float64(max_cost),
        )
        if int(changed[0]) == 0:
            break

    # Convert: inf or over-budget -> NaN, else float32
    out = cp.where(
        cp.isinf(dist) | (dist > max_cost), cp.nan, dist,
    ).astype(cp.float32)
    return out


def _cost_distance_dask_cupy(source_da, friction_da,
                              cellsize_x, cellsize_y, max_cost,
                              target_values, dy, dx, dd):
    """Dask+CuPy cost distance.

    Bounded max_cost: ``da.map_overlap`` with per-chunk GPU relaxation.
    Unbounded / large radius: convert to dask+numpy, use CPU iterative path
    (Dijkstra is O(N log N) vs parallel relaxation's O(N * diameter), so CPU
    is more efficient for unbounded global shortest paths).
    """
    import cupy as cp

    height, width = source_da.shape

    use_map_overlap = False
    if np.isfinite(max_cost):
        positive_friction = da.where(friction_da > 0, friction_da, np.inf)
        f_min = float(da.nanmin(positive_friction).compute())
        if np.isfinite(f_min) and f_min > 0:
            min_cellsize = min(cellsize_x, cellsize_y)
            max_radius = max_cost / (f_min * min_cellsize)
            pad = int(max_radius + 1)
            chunks_y, chunks_x = source_da.chunks
            if pad < max(chunks_y) and pad < max(chunks_x):
                use_map_overlap = True

    if use_map_overlap:
        pad_y = int(max_cost / (f_min * cellsize_y) + 1)
        pad_x = int(max_cost / (f_min * cellsize_x) + 1)

        # Closure captures the scalar parameters
        tv = target_values
        mc = max_cost
        cx, cy = cellsize_x, cellsize_y
        _dy, _dx, _dd = dy, dx, dd

        def _chunk_func(source_block, friction_block):
            return _cost_distance_cupy(
                source_block, friction_block,
                cx, cy, mc, tv, _dy, _dx, _dd,
            )

        return da.map_overlap(
            _chunk_func,
            source_da, friction_da,
            depth=(pad_y, pad_x),
            boundary=np.nan,
            dtype=np.float32,
            meta=cp.array((), dtype=cp.float32),
        )

    # Unbounded or padding too large: convert to dask+numpy, use CPU path
    source_np = source_da.map_blocks(
        lambda b: b.get(), dtype=source_da.dtype,
        meta=np.array((), dtype=source_da.dtype),
    )
    friction_np = friction_da.map_blocks(
        lambda b: b.get(), dtype=friction_da.dtype,
        meta=np.array((), dtype=friction_da.dtype),
    )
    result = _cost_distance_dask(
        source_np, friction_np,
        cellsize_x, cellsize_y, max_cost,
        target_values, dy, dx, dd,
    )
    # Convert back to dask+cupy
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# ---------------------------------------------------------------------------
# Tile kernel for iterative boundary Dijkstra
# ---------------------------------------------------------------------------

@ngjit
def _cost_distance_tile_kernel(
    source_data, friction_data, height, width,
    cellsize_x, cellsize_y, max_cost, target_values,
    dy, dx, dd,
    seed_top, seed_bottom, seed_left, seed_right,
    seed_tl, seed_tr, seed_bl, seed_br,
):
    """Seeded multi-source Dijkstra.  Returns float64 dist array.

    Like ``_cost_distance_kernel`` but additionally seeds boundary pixels
    from neighbouring-tile distance arrays.  Seeds already include the
    edge-crossing cost.
    """
    n_values = len(target_values)
    n_neighbors = len(dy)

    dist = np.full((height, width), np.inf, dtype=np.float64)

    max_heap = height * width
    h_keys = np.empty(max_heap, dtype=np.float64)
    h_rows = np.empty(max_heap, dtype=np.int64)
    h_cols = np.empty(max_heap, dtype=np.int64)
    h_size = 0

    visited = np.zeros((height, width), dtype=np.int8)

    # Phase 1: seed source pixels at cost 0
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
                f = friction_data[r, c]
                if np.isfinite(f) and f > 0.0:
                    dist[r, c] = 0.0
                    h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                        0.0, r, c)

    # Phase 2: seed boundary pixels from neighbour tiles
    # Top edge
    for c in range(width):
        s = seed_top[c]
        if s < dist[0, c]:
            f = friction_data[0, c]
            if np.isfinite(f) and f > 0.0:
                dist[0, c] = s
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    s, 0, c)
    # Bottom edge
    for c in range(width):
        s = seed_bottom[c]
        if s < dist[height - 1, c]:
            f = friction_data[height - 1, c]
            if np.isfinite(f) and f > 0.0:
                dist[height - 1, c] = s
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    s, height - 1, c)
    # Left edge
    for r in range(height):
        s = seed_left[r]
        if s < dist[r, 0]:
            f = friction_data[r, 0]
            if np.isfinite(f) and f > 0.0:
                dist[r, 0] = s
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    s, r, 0)
    # Right edge
    for r in range(height):
        s = seed_right[r]
        if s < dist[r, width - 1]:
            f = friction_data[r, width - 1]
            if np.isfinite(f) and f > 0.0:
                dist[r, width - 1] = s
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    s, r, width - 1)
    # Diagonal corner seeds (8-connectivity; caller sets to inf for 4-conn)
    # Top-left
    s = seed_tl
    if s < dist[0, 0]:
        f = friction_data[0, 0]
        if np.isfinite(f) and f > 0.0:
            dist[0, 0] = s
            h_size = _heap_push(h_keys, h_rows, h_cols, h_size, s, 0, 0)
    # Top-right
    s = seed_tr
    if s < dist[0, width - 1]:
        f = friction_data[0, width - 1]
        if np.isfinite(f) and f > 0.0:
            dist[0, width - 1] = s
            h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                s, 0, width - 1)
    # Bottom-left
    s = seed_bl
    if s < dist[height - 1, 0]:
        f = friction_data[height - 1, 0]
        if np.isfinite(f) and f > 0.0:
            dist[height - 1, 0] = s
            h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                s, height - 1, 0)
    # Bottom-right
    s = seed_br
    if s < dist[height - 1, width - 1]:
        f = friction_data[height - 1, width - 1]
        if np.isfinite(f) and f > 0.0:
            dist[height - 1, width - 1] = s
            h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                s, height - 1, width - 1)

    # Phase 3: Dijkstra main loop (identical to _cost_distance_kernel)
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
            if not (np.isfinite(f_v) and f_v > 0.0):
                continue

            edge_cost = dd[i] * (f_u + f_v) * 0.5
            new_cost = cost_u + edge_cost

            if new_cost < dist[vr, vc]:
                dist[vr, vc] = new_cost
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    new_cost, vr, vc)

    return dist


@ngjit
def _dist_to_float32(dist, height, width, max_cost):
    """Convert float64 dist -> float32, mapping inf / over-budget to NaN."""
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
# Iterative boundary-only Dijkstra for dask arrays
# ---------------------------------------------------------------------------

def _preprocess_tiles(source_da, friction_da, chunks_y, chunks_x,
                      target_values):
    """Extract friction boundary strips and identify source tiles.

    Streams one tile at a time so only one chunk is in RAM.
    """
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    n_values = len(target_values)

    friction_bdry = {
        side: [[None] * n_tile_x for _ in range(n_tile_y)]
        for side in ('top', 'bottom', 'left', 'right')
    }
    has_source = [[False] * n_tile_x for _ in range(n_tile_y)]

    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            fchunk = friction_da.blocks[iy, ix].compute()
            friction_bdry['top'][iy][ix] = fchunk[0, :].astype(np.float32)
            friction_bdry['bottom'][iy][ix] = fchunk[-1, :].astype(np.float32)
            friction_bdry['left'][iy][ix] = fchunk[:, 0].astype(np.float32)
            friction_bdry['right'][iy][ix] = fchunk[:, -1].astype(np.float32)

            schunk = source_da.blocks[iy, ix].compute()
            if n_values == 0:
                has_source[iy][ix] = bool(
                    np.any((schunk != 0) & np.isfinite(schunk))
                )
            else:
                for tv in target_values:
                    if np.any(schunk == tv):
                        has_source[iy][ix] = True
                        break

    return friction_bdry, has_source


def _init_boundaries(chunks_y, chunks_x):
    """Create boundary distance arrays, all initialised to inf (float32)."""
    n_y = len(chunks_y)
    n_x = len(chunks_x)
    return {
        'top': [
            [np.full(chunks_x[ix], np.inf, dtype=np.float32)
             for ix in range(n_x)]
            for _ in range(n_y)
        ],
        'bottom': [
            [np.full(chunks_x[ix], np.inf, dtype=np.float32)
             for ix in range(n_x)]
            for _ in range(n_y)
        ],
        'left': [
            [np.full(chunks_y[iy], np.inf, dtype=np.float32)
             for _ in range(n_x)]
            for iy in range(n_y)
        ],
        'right': [
            [np.full(chunks_y[iy], np.inf, dtype=np.float32)
             for _ in range(n_x)]
            for iy in range(n_y)
        ],
    }


def _compute_seeds(iy, ix, boundaries, friction_bdry,
                   cellsize_x, cellsize_y, chunks_y, chunks_x,
                   n_tile_y, n_tile_x, connectivity):
    """Compute seed arrays for tile (iy, ix) from neighbour boundaries.

    Returns (seed_top, seed_bottom, seed_left, seed_right,
             seed_tl, seed_tr, seed_bl, seed_br).
    Cardinal seeds are 1-D float64 arrays; corner seeds are float64 scalars.
    """
    tile_h = chunks_y[iy]
    tile_w = chunks_x[ix]
    diag_dist = sqrt(cellsize_x ** 2 + cellsize_y ** 2)

    seed_top = np.full(tile_w, np.inf)
    seed_bottom = np.full(tile_w, np.inf)
    seed_left = np.full(tile_h, np.inf)
    seed_right = np.full(tile_h, np.inf)
    seed_tl = np.inf
    seed_tr = np.inf
    seed_bl = np.inf
    seed_br = np.inf

    my_top = friction_bdry['top'][iy][ix].astype(np.float64)
    my_bottom = friction_bdry['bottom'][iy][ix].astype(np.float64)
    my_left = friction_bdry['left'][iy][ix].astype(np.float64)
    my_right = friction_bdry['right'][iy][ix].astype(np.float64)

    def _edge_seeds(nb_dist, nb_fric, my_fric, cardinal_dist):
        """Minimum-cost seed per boundary pixel (cardinal + diagonal)."""
        d = nb_dist.astype(np.float64)
        nf = nb_fric.astype(np.float64)
        mf = my_fric
        n = len(d)
        # Cardinal: pixel c <- c
        cost = d + cardinal_dist * (nf + mf) * 0.5
        valid = (np.isfinite(d) & np.isfinite(nf) & (nf > 0)
                 & np.isfinite(mf) & (mf > 0))
        seed = np.where(valid, cost, np.inf)
        if connectivity == 8 and n > 1:
            # Diagonal: pixel c <- c-1
            dl = d[:-1]
            nl = nf[:-1]
            cl = dl + diag_dist * (nl + mf[1:]) * 0.5
            vl = (np.isfinite(dl) & np.isfinite(nl) & (nl > 0)
                  & np.isfinite(mf[1:]) & (mf[1:] > 0))
            seed[1:] = np.minimum(seed[1:], np.where(vl, cl, np.inf))
            # Diagonal: pixel c <- c+1
            dr = d[1:]
            nr = nf[1:]
            cr = dr + diag_dist * (nr + mf[:-1]) * 0.5
            vr = (np.isfinite(dr) & np.isfinite(nr) & (nr > 0)
                  & np.isfinite(mf[:-1]) & (mf[:-1] > 0))
            seed[:-1] = np.minimum(seed[:-1], np.where(vr, cr, np.inf))
        return seed

    # Edge neighbours (cardinal + diagonal along shared boundary)
    if iy > 0:
        seed_top = _edge_seeds(
            boundaries['bottom'][iy - 1][ix],
            friction_bdry['bottom'][iy - 1][ix],
            my_top, cellsize_y,
        )
    if iy < n_tile_y - 1:
        seed_bottom = _edge_seeds(
            boundaries['top'][iy + 1][ix],
            friction_bdry['top'][iy + 1][ix],
            my_bottom, cellsize_y,
        )
    if ix > 0:
        seed_left = _edge_seeds(
            boundaries['right'][iy][ix - 1],
            friction_bdry['right'][iy][ix - 1],
            my_left, cellsize_x,
        )
    if ix < n_tile_x - 1:
        seed_right = _edge_seeds(
            boundaries['left'][iy][ix + 1],
            friction_bdry['left'][iy][ix + 1],
            my_right, cellsize_x,
        )

    # Diagonal corner seeds (8-connectivity only)
    if connectivity == 8:
        def _corner(nb_d, nb_f, my_f):
            nb_d = float(nb_d)
            nb_f = float(nb_f)
            my_f = float(my_f)
            if (np.isfinite(nb_d) and np.isfinite(nb_f) and nb_f > 0
                    and np.isfinite(my_f) and my_f > 0):
                return nb_d + diag_dist * (nb_f + my_f) * 0.5
            return np.inf

        if iy > 0 and ix > 0:
            seed_tl = _corner(
                boundaries['bottom'][iy - 1][ix - 1][-1],
                friction_bdry['bottom'][iy - 1][ix - 1][-1],
                my_top[0],
            )
        if iy > 0 and ix < n_tile_x - 1:
            seed_tr = _corner(
                boundaries['bottom'][iy - 1][ix + 1][0],
                friction_bdry['bottom'][iy - 1][ix + 1][0],
                my_top[-1],
            )
        if iy < n_tile_y - 1 and ix > 0:
            seed_bl = _corner(
                boundaries['top'][iy + 1][ix - 1][-1],
                friction_bdry['top'][iy + 1][ix - 1][-1],
                my_bottom[0],
            )
        if iy < n_tile_y - 1 and ix < n_tile_x - 1:
            seed_br = _corner(
                boundaries['top'][iy + 1][ix + 1][0],
                friction_bdry['top'][iy + 1][ix + 1][0],
                my_bottom[-1],
            )

    return (seed_top, seed_bottom, seed_left, seed_right,
            seed_tl, seed_tr, seed_bl, seed_br)


def _can_skip(iy, ix, has_source, boundaries,
              n_tile_y, n_tile_x, connectivity):
    """True when a tile cannot possibly receive any cost information."""
    if has_source[iy][ix]:
        return False
    # Cardinal neighbours
    if iy > 0 and np.any(np.isfinite(boundaries['bottom'][iy - 1][ix])):
        return False
    if (iy < n_tile_y - 1
            and np.any(np.isfinite(boundaries['top'][iy + 1][ix]))):
        return False
    if ix > 0 and np.any(np.isfinite(boundaries['right'][iy][ix - 1])):
        return False
    if (ix < n_tile_x - 1
            and np.any(np.isfinite(boundaries['left'][iy][ix + 1]))):
        return False
    # Diagonal corners
    if connectivity == 8:
        if (iy > 0 and ix > 0
                and np.isfinite(boundaries['bottom'][iy - 1][ix - 1][-1])):
            return False
        if (iy > 0 and ix < n_tile_x - 1
                and np.isfinite(boundaries['bottom'][iy - 1][ix + 1][0])):
            return False
        if (iy < n_tile_y - 1 and ix > 0
                and np.isfinite(boundaries['top'][iy + 1][ix - 1][-1])):
            return False
        if (iy < n_tile_y - 1 and ix < n_tile_x - 1
                and np.isfinite(boundaries['top'][iy + 1][ix + 1][0])):
            return False
    return True


def _process_tile(iy, ix, source_da, friction_da,
                  boundaries, friction_bdry,
                  cellsize_x, cellsize_y, max_cost, target_values,
                  dy, dx, dd, chunks_y, chunks_x,
                  n_tile_y, n_tile_x, connectivity):
    """Run seeded Dijkstra on one tile; update boundaries in-place.

    Returns the maximum absolute boundary change (float).
    """
    source_chunk = source_da.blocks[iy, ix].compute()
    friction_chunk = friction_da.blocks[iy, ix].compute()
    h, w = source_chunk.shape

    seeds = _compute_seeds(
        iy, ix, boundaries, friction_bdry,
        cellsize_x, cellsize_y, chunks_y, chunks_x,
        n_tile_y, n_tile_x, connectivity,
    )

    dist = _cost_distance_tile_kernel(
        source_chunk, friction_chunk, h, w,
        cellsize_x, cellsize_y, max_cost, target_values,
        dy, dx, dd, *seeds,
    )

    # Extract new boundary strips (float32)
    new_top = dist[0, :].astype(np.float32)
    new_bottom = dist[-1, :].astype(np.float32)
    new_left = dist[:, 0].astype(np.float32)
    new_right = dist[:, -1].astype(np.float32)

    # Compute max absolute change versus old boundaries
    change = 0.0
    for old, new in ((boundaries['top'][iy][ix], new_top),
                     (boundaries['bottom'][iy][ix], new_bottom),
                     (boundaries['left'][iy][ix], new_left),
                     (boundaries['right'][iy][ix], new_right)):
        with np.errstate(invalid='ignore'):
            diff = np.abs(new.astype(np.float64) - old.astype(np.float64))
        # inf - inf -> nan: treat as no change
        diff = np.where(np.isnan(diff), 0.0, diff)
        m = float(np.max(diff))
        if m > change:
            change = m

    # Store updated boundaries
    boundaries['top'][iy][ix] = new_top
    boundaries['bottom'][iy][ix] = new_bottom
    boundaries['left'][iy][ix] = new_left
    boundaries['right'][iy][ix] = new_right

    return change


def _cost_distance_dask_iterative(source_da, friction_da,
                                   cellsize_x, cellsize_y,
                                   max_cost, target_values,
                                   dy, dx, dd):
    """Iterative boundary-only Dijkstra for arbitrarily large dask arrays.

    Memory usage is O(sqrt(N)) for inter-iteration storage.
    """
    connectivity = len(dy)
    chunks_y = source_da.chunks[0]
    chunks_x = source_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    # Phase 0: pre-extract friction boundaries & source tile flags
    friction_bdry, has_source = _preprocess_tiles(
        source_da, friction_da, chunks_y, chunks_x, target_values,
    )

    # Phase 1: initialise distance boundaries to inf
    boundaries = _init_boundaries(chunks_y, chunks_x)

    # Phase 2: iterative forward/backward sweeps
    max_iterations = max(n_tile_y, n_tile_x) + 10
    args = (source_da, friction_da, boundaries, friction_bdry,
            cellsize_x, cellsize_y, max_cost, target_values,
            dy, dx, dd, chunks_y, chunks_x,
            n_tile_y, n_tile_x, connectivity)

    for _iteration in range(max_iterations):
        max_change = 0.0

        # Forward sweep (top-left -> bottom-right)
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                if _can_skip(iy, ix, has_source, boundaries,
                             n_tile_y, n_tile_x, connectivity):
                    continue
                c = _process_tile(iy, ix, *args)
                if c > max_change:
                    max_change = c

        # Backward sweep (bottom-right -> top-left)
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                if _can_skip(iy, ix, has_source, boundaries,
                             n_tile_y, n_tile_x, connectivity):
                    continue
                c = _process_tile(iy, ix, *args)
                if c > max_change:
                    max_change = c

        if max_change == 0.0:
            break

    # Phase 3: lazy final assembly via da.map_blocks
    return _assemble_result(
        source_da, friction_da, boundaries, friction_bdry,
        cellsize_x, cellsize_y, max_cost, target_values,
        dy, dx, dd, chunks_y, chunks_x,
        n_tile_y, n_tile_x, connectivity,
    )


def _assemble_result(source_da, friction_da, boundaries, friction_bdry,
                     cellsize_x, cellsize_y, max_cost, target_values,
                     dy, dx, dd, chunks_y, chunks_x,
                     n_tile_y, n_tile_x, connectivity):
    """Build a lazy dask array by re-running each tile with converged seeds."""

    def _tile_fn(source_block, friction_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(source_block.shape, np.nan, dtype=np.float32)
        iy, ix = block_info[0]['chunk-location']
        h, w = source_block.shape
        seeds = _compute_seeds(
            iy, ix, boundaries, friction_bdry,
            cellsize_x, cellsize_y, chunks_y, chunks_x,
            n_tile_y, n_tile_x, connectivity,
        )
        dist = _cost_distance_tile_kernel(
            source_block, friction_block, h, w,
            cellsize_x, cellsize_y, max_cost, target_values,
            dy, dx, dd, *seeds,
        )
        return _dist_to_float32(dist, h, w, max_cost)

    return da.map_blocks(
        _tile_fn,
        source_da, friction_da,
        dtype=np.float32,
        meta=np.array((), dtype=np.float32),
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
        # Use iterative tile Dijkstra — bounded memory, no single-chunk rechunk
        import warnings
        warnings.warn(
            "cost_distance: max_cost is infinite or the implied radius "
            "exceeds chunk dimensions; using iterative tile Dijkstra. "
            "Setting a finite max_cost enables faster single-pass "
            "processing.",
            UserWarning,
            stacklevel=4,
        )
        return _cost_distance_dask_iterative(
            source_da, friction_da, cellsize_x, cellsize_y,
            max_cost, target_values, dy, dx, dd,
        )
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
    _validate_raster(raster, func_name='cost_distance', name='raster')
    _validate_raster(friction, func_name='cost_distance', name='friction')
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

    _is_dask = da is not None and isinstance(source_data, da.Array)
    _is_cupy_backend = (
        not _is_dask
        and has_cuda_and_cupy()
        and is_cupy_array(source_data)
    )
    _is_dask_cupy = _is_dask and has_cuda_and_cupy() and is_dask_cupy(raster)

    if _is_dask:
        # Rechunk friction to match raster
        if isinstance(friction_data, da.Array):
            friction_data = friction_data.rechunk(source_data.chunks)
        else:
            friction_data = da.from_array(friction_data,
                                          chunks=source_data.chunks)

    if _is_cupy_backend:
        result_data = _cost_distance_cupy(
            source_data, friction_data,
            cellsize_x, cellsize_y, max_cost_f,
            target_values, dy, dx, dd,
        )
    elif _is_dask_cupy:
        result_data = _cost_distance_dask_cupy(
            source_data, friction_data,
            cellsize_x, cellsize_y, max_cost_f,
            target_values, dy, dx, dd,
        )
    elif isinstance(source_data, np.ndarray):
        if isinstance(friction_data, np.ndarray):
            result_data = _cost_distance_numpy(
                source_data, friction_data,
                cellsize_x, cellsize_y, max_cost_f,
                target_values, dy, dx, dd,
            )
        else:
            raise TypeError("friction must be numpy-backed when raster is")
    elif _is_dask:
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
