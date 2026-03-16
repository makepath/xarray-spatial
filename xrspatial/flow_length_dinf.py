"""D-infinity flow length: proportion-weighted distance from each cell to
outlet (downstream) or longest path from divide to cell (upstream).

Algorithm
---------
CPU : Kahn's BFS topological sort on the D-inf DAG -- O(N).
      Downstream: reverse pass from outlets, accumulating weighted step
      distance.  Upstream: forward pass from divides, propagating max
      distance.
GPU : CuPy-via-CPU (same strategy as flow_length.py).
Dask: iterative tile sweep with BoundaryStore boundary propagation.
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial._boundary_store import BoundaryStore
from xrspatial.flow_accumulation import _preprocess_tiles
from xrspatial.flow_accumulation_dinf import _angle_to_neighbors
from xrspatial.utils import (
    _validate_raster,
    get_dataarray_resolution,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)
from xrspatial.dataset_support import supports_dataset


# =====================================================================
# Step-distance helper
# =====================================================================

@ngjit
def _neighbor_dist(dy, dx, cellsize_x, cellsize_y):
    """Return the travel distance for a D-inf neighbor offset."""
    ady = abs(dy)
    adx = abs(dx)
    if ady == 0 and adx == 0:
        return 0.0
    if ady == 0:
        return cellsize_x
    if adx == 0:
        return cellsize_y
    return math.sqrt(cellsize_x * cellsize_x + cellsize_y * cellsize_y)


# =====================================================================
# CPU kernels
# =====================================================================

@ngjit
def _flow_length_dinf_downstream_cpu(flow_dir, H, W, cellsize_x, cellsize_y):
    """Downstream D-inf flow length: proportion-weighted average distance
    from each cell to its outlet(s).

    Two-pass O(N):
    1. Kahn's BFS builds topological order (divides first).
    2. Reverse pass (outlets first): flow_len[cell] =
       w1*(dist1 + flow_len[nb1]) + w2*(dist2 + flow_len[nb2])
    """
    in_degree = np.zeros((H, W), dtype=np.int32)
    valid = np.zeros((H, W), dtype=np.int8)
    flow_len = np.empty((H, W), dtype=np.float64)

    # Init
    for r in range(H):
        for c in range(W):
            v = flow_dir[r, c]
            if v == v:  # not NaN
                valid[r, c] = 1
                flow_len[r, c] = 0.0
            else:
                flow_len[r, c] = np.nan

    # In-degrees
    for r in range(H):
        for c in range(W):
            if valid[r, c] == 0:
                continue
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
            if w1 > 0.0:
                nr, nc = r + dy1, c + dx1
                if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1
            if w2 > 0.0:
                nr, nc = r + dy2, c + dx2
                if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1

    # BFS topological order (divides first)
    order_r = np.empty(H * W, dtype=np.int64)
    order_c = np.empty(H * W, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for r in range(H):
        for c in range(W):
            if valid[r, c] == 1 and in_degree[r, c] == 0:
                order_r[tail] = r
                order_c[tail] = c
                tail += 1

    while head < tail:
        r = order_r[head]
        c = order_c[head]
        head += 1
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
        if w1 > 0.0:
            nr, nc = r + dy1, c + dx1
            if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    order_r[tail] = nr
                    order_c[tail] = nc
                    tail += 1
        if w2 > 0.0:
            nr, nc = r + dy2, c + dx2
            if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    order_r[tail] = nr
                    order_c[tail] = nc
                    tail += 1

    # Reverse pass: outlets -> divides
    for i in range(tail - 1, -1, -1):
        r = order_r[i]
        c = order_c[i]
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
        total = 0.0

        if w1 > 0.0:
            d1 = _neighbor_dist(dy1, dx1, cellsize_x, cellsize_y)
            nr, nc = r + dy1, c + dx1
            if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                total += w1 * (d1 + flow_len[nr, nc])
            else:
                total += w1 * d1  # grid edge

        if w2 > 0.0:
            d2 = _neighbor_dist(dy2, dx2, cellsize_x, cellsize_y)
            nr, nc = r + dy2, c + dx2
            if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                total += w2 * (d2 + flow_len[nr, nc])
            else:
                total += w2 * d2  # grid edge

        flow_len[r, c] = total

    return flow_len


@ngjit
def _flow_length_dinf_upstream_cpu(flow_dir, H, W, cellsize_x, cellsize_y):
    """Upstream D-inf flow length: longest path from any divide to cell.

    Kahn's BFS from divides downstream, propagating max distance.
    """
    in_degree = np.zeros((H, W), dtype=np.int32)
    valid = np.zeros((H, W), dtype=np.int8)
    flow_len = np.empty((H, W), dtype=np.float64)

    for r in range(H):
        for c in range(W):
            v = flow_dir[r, c]
            if v == v:
                valid[r, c] = 1
                flow_len[r, c] = 0.0
            else:
                flow_len[r, c] = np.nan

    for r in range(H):
        for c in range(W):
            if valid[r, c] == 0:
                continue
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
            if w1 > 0.0:
                nr, nc = r + dy1, c + dx1
                if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1
            if w2 > 0.0:
                nr, nc = r + dy2, c + dx2
                if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1

    # BFS queue
    queue_r = np.empty(H * W, dtype=np.int64)
    queue_c = np.empty(H * W, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for r in range(H):
        for c in range(W):
            if valid[r, c] == 1 and in_degree[r, c] == 0:
                queue_r[tail] = r
                queue_c[tail] = c
                tail += 1

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1

        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])

        if w1 > 0.0:
            nr, nc = r + dy1, c + dx1
            if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                d1 = _neighbor_dist(dy1, dx1, cellsize_x, cellsize_y)
                candidate = flow_len[r, c] + d1
                if candidate > flow_len[nr, nc]:
                    flow_len[nr, nc] = candidate
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

        if w2 > 0.0:
            nr, nc = r + dy2, c + dx2
            if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                d2 = _neighbor_dist(dy2, dx2, cellsize_x, cellsize_y)
                candidate = flow_len[r, c] + d2
                if candidate > flow_len[nr, nc]:
                    flow_len[nr, nc] = candidate
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return flow_len


# =====================================================================
# CuPy backend (via CPU)
# =====================================================================

def _flow_length_dinf_cupy(flow_dir_data, direction, cellsize_x, cellsize_y):
    import cupy as cp
    fd_np = flow_dir_data.get().astype(np.float64)
    H, W = fd_np.shape
    if direction == 'downstream':
        out = _flow_length_dinf_downstream_cpu(fd_np, H, W,
                                                cellsize_x, cellsize_y)
    else:
        out = _flow_length_dinf_upstream_cpu(fd_np, H, W,
                                              cellsize_x, cellsize_y)
    return cp.asarray(out)


# =====================================================================
# Dask tile kernels
# =====================================================================

@ngjit
def _flow_length_dinf_downstream_tile(flow_dir, h, w, cellsize_x, cellsize_y,
                                       seed_top, seed_bottom,
                                       seed_left, seed_right,
                                       seed_tl, seed_tr, seed_bl, seed_br):
    """Downstream D-inf flow length for one tile with exit seeds."""
    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)
    flow_len = np.empty((h, w), dtype=np.float64)

    # Init
    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v == v:
                valid[r, c] = 1
                flow_len[r, c] = 0.0
            else:
                flow_len[r, c] = np.nan

    # In-degrees
    for r in range(h):
        for c in range(w):
            if valid[r, c] == 0:
                continue
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
            if w1 > 0.0:
                nr, nc = r + dy1, c + dx1
                if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1
            if w2 > 0.0:
                nr, nc = r + dy2, c + dx2
                if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1

    # BFS topological order
    order_r = np.empty(h * w, dtype=np.int64)
    order_c = np.empty(h * w, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for r in range(h):
        for c in range(w):
            if valid[r, c] == 1 and in_degree[r, c] == 0:
                order_r[tail] = r
                order_c[tail] = c
                tail += 1

    while head < tail:
        r = order_r[head]
        c = order_c[head]
        head += 1
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
        if w1 > 0.0:
            nr, nc = r + dy1, c + dx1
            if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    order_r[tail] = nr
                    order_c[tail] = nc
                    tail += 1
        if w2 > 0.0:
            nr, nc = r + dy2, c + dx2
            if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    order_r[tail] = nr
                    order_c[tail] = nc
                    tail += 1

    # Reverse pass
    for i in range(tail - 1, -1, -1):
        r = order_r[i]
        c = order_c[i]
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
        total = 0.0

        if w1 > 0.0:
            d1 = _neighbor_dist(dy1, dx1, cellsize_x, cellsize_y)
            nr, nc = r + dy1, c + dx1
            if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                total += w1 * (d1 + flow_len[nr, nc])
            else:
                # Flows out of tile -- look up exit seed
                exit_val = _get_exit_seed(nr, nc, h, w,
                                          seed_top, seed_bottom,
                                          seed_left, seed_right,
                                          seed_tl, seed_tr, seed_bl, seed_br)
                if exit_val == exit_val:  # not NaN
                    total += w1 * (d1 + exit_val)
                else:
                    total += w1 * d1  # grid edge

        if w2 > 0.0:
            d2 = _neighbor_dist(dy2, dx2, cellsize_x, cellsize_y)
            nr, nc = r + dy2, c + dx2
            if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                total += w2 * (d2 + flow_len[nr, nc])
            else:
                exit_val = _get_exit_seed(nr, nc, h, w,
                                          seed_top, seed_bottom,
                                          seed_left, seed_right,
                                          seed_tl, seed_tr, seed_bl, seed_br)
                if exit_val == exit_val:
                    total += w2 * (d2 + exit_val)
                else:
                    total += w2 * d2

        flow_len[r, c] = total

    return flow_len


@ngjit
def _get_exit_seed(nr, nc, h, w,
                   seed_top, seed_bottom, seed_left, seed_right,
                   seed_tl, seed_tr, seed_bl, seed_br):
    """Look up the exit seed for a cell flowing out of the tile."""
    if nr < 0 and nc < 0:
        return seed_tl
    if nr < 0 and nc >= w:
        return seed_tr
    if nr >= h and nc < 0:
        return seed_bl
    if nr >= h and nc >= w:
        return seed_br
    if nr < 0:
        return seed_top[nc]
    if nr >= h:
        return seed_bottom[nc]
    if nc < 0:
        return seed_left[nr]
    if nc >= w:
        return seed_right[nr]
    return np.nan


@ngjit
def _flow_length_dinf_upstream_tile(flow_dir, h, w, cellsize_x, cellsize_y,
                                     seed_top, seed_bottom,
                                     seed_left, seed_right,
                                     seed_tl, seed_tr, seed_bl, seed_br):
    """Upstream D-inf flow length for one tile with entry seeds."""
    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)
    flow_len = np.empty((h, w), dtype=np.float64)

    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v == v:
                valid[r, c] = 1
                flow_len[r, c] = 0.0
            else:
                flow_len[r, c] = np.nan

    # Apply entry seeds (max with existing)
    for c in range(w):
        if valid[0, c] == 1 and seed_top[c] > flow_len[0, c]:
            flow_len[0, c] = seed_top[c]
        if valid[h - 1, c] == 1 and seed_bottom[c] > flow_len[h - 1, c]:
            flow_len[h - 1, c] = seed_bottom[c]
    for r in range(h):
        if valid[r, 0] == 1 and seed_left[r] > flow_len[r, 0]:
            flow_len[r, 0] = seed_left[r]
        if valid[r, w - 1] == 1 and seed_right[r] > flow_len[r, w - 1]:
            flow_len[r, w - 1] = seed_right[r]

    # Corner seeds
    if valid[0, 0] == 1 and seed_tl > flow_len[0, 0]:
        flow_len[0, 0] = seed_tl
    if valid[0, w - 1] == 1 and seed_tr > flow_len[0, w - 1]:
        flow_len[0, w - 1] = seed_tr
    if valid[h - 1, 0] == 1 and seed_bl > flow_len[h - 1, 0]:
        flow_len[h - 1, 0] = seed_bl
    if valid[h - 1, w - 1] == 1 and seed_br > flow_len[h - 1, w - 1]:
        flow_len[h - 1, w - 1] = seed_br

    # In-degrees
    for r in range(h):
        for c in range(w):
            if valid[r, c] == 0:
                continue
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
            if w1 > 0.0:
                nr, nc = r + dy1, c + dx1
                if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1
            if w2 > 0.0:
                nr, nc = r + dy2, c + dx2
                if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1

    # BFS from divides
    queue_r = np.empty(h * w, dtype=np.int64)
    queue_c = np.empty(h * w, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for r in range(h):
        for c in range(w):
            if valid[r, c] == 1 and in_degree[r, c] == 0:
                queue_r[tail] = r
                queue_c[tail] = c
                tail += 1

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1

        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])

        if w1 > 0.0:
            nr, nc = r + dy1, c + dx1
            if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                d1 = _neighbor_dist(dy1, dx1, cellsize_x, cellsize_y)
                candidate = flow_len[r, c] + d1
                if candidate > flow_len[nr, nc]:
                    flow_len[nr, nc] = candidate
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

        if w2 > 0.0:
            nr, nc = r + dy2, c + dx2
            if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                d2 = _neighbor_dist(dy2, dx2, cellsize_x, cellsize_y)
                candidate = flow_len[r, c] + d2
                if candidate > flow_len[nr, nc]:
                    flow_len[nr, nc] = candidate
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return flow_len


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _compute_exit_seeds_downstream(iy, ix, boundaries, flow_bdry,
                                    chunks_y, chunks_x,
                                    n_tile_y, n_tile_x):
    """For downstream: provide flow_length values at destination cells
    in adjacent tiles. Same approach as flow_length_mfd -- seed arrays
    mirror the adjacent tile's facing boundary."""
    tile_h = chunks_y[iy]
    tile_w = chunks_x[ix]

    seed_top = np.full(tile_w, np.nan)
    seed_bottom = np.full(tile_w, np.nan)
    seed_left = np.full(tile_h, np.nan)
    seed_right = np.full(tile_h, np.nan)
    seed_tl = np.nan
    seed_tr = np.nan
    seed_bl = np.nan
    seed_br = np.nan

    if iy > 0:
        nb = boundaries.get('bottom', iy - 1, ix)
        seed_top[:len(nb)] = nb
    if iy < n_tile_y - 1:
        nb = boundaries.get('top', iy + 1, ix)
        seed_bottom[:len(nb)] = nb
    if ix > 0:
        nb = boundaries.get('right', iy, ix - 1)
        seed_left[:len(nb)] = nb
    if ix < n_tile_x - 1:
        nb = boundaries.get('left', iy, ix + 1)
        seed_right[:len(nb)] = nb

    # Diagonal corners
    if iy > 0 and ix > 0:
        seed_tl = float(boundaries.get('bottom', iy - 1, ix - 1)[-1])
    if iy > 0 and ix < n_tile_x - 1:
        seed_tr = float(boundaries.get('bottom', iy - 1, ix + 1)[0])
    if iy < n_tile_y - 1 and ix > 0:
        seed_bl = float(boundaries.get('top', iy + 1, ix - 1)[-1])
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        seed_br = float(boundaries.get('top', iy + 1, ix + 1)[0])

    return (seed_top, seed_bottom, seed_left, seed_right,
            seed_tl, seed_tr, seed_bl, seed_br)


def _compute_entry_seeds_upstream(iy, ix, boundaries, flow_bdry,
                                   chunks_y, chunks_x,
                                   n_tile_y, n_tile_x,
                                   cellsize_x, cellsize_y):
    """For upstream: check which adjacent cells flow INTO this tile,
    seed with max(existing, neighbor_upstream + step_dist)."""
    tile_h = chunks_y[iy]
    tile_w = chunks_x[ix]

    seed_top = np.zeros(tile_w, dtype=np.float64)
    seed_bottom = np.zeros(tile_w, dtype=np.float64)
    seed_left = np.zeros(tile_h, dtype=np.float64)
    seed_right = np.zeros(tile_h, dtype=np.float64)
    seed_tl = 0.0
    seed_tr = 0.0
    seed_bl = 0.0
    seed_br = 0.0

    # Top edge: bottom row of tile above flows south into this tile
    if iy > 0:
        nb_fdir = flow_bdry.get('bottom', iy - 1, ix)
        nb_val = boundaries.get('bottom', iy - 1, ix)
        for c in range(len(nb_fdir)):
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(nb_fdir[c])
            if w1 > 0.0 and dy1 == 1:
                d = _neighbor_dist(dy1, dx1, cellsize_x, cellsize_y)
                tc = c + dx1
                if 0 <= tc < tile_w:
                    v = nb_val[c] + d
                    if v > seed_top[tc]:
                        seed_top[tc] = v
            if w2 > 0.0 and dy2 == 1:
                d = _neighbor_dist(dy2, dx2, cellsize_x, cellsize_y)
                tc = c + dx2
                if 0 <= tc < tile_w:
                    v = nb_val[c] + d
                    if v > seed_top[tc]:
                        seed_top[tc] = v

    # Bottom edge: top row of tile below flows north
    if iy < n_tile_y - 1:
        nb_fdir = flow_bdry.get('top', iy + 1, ix)
        nb_val = boundaries.get('top', iy + 1, ix)
        for c in range(len(nb_fdir)):
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(nb_fdir[c])
            if w1 > 0.0 and dy1 == -1:
                d = _neighbor_dist(dy1, dx1, cellsize_x, cellsize_y)
                tc = c + dx1
                if 0 <= tc < tile_w:
                    v = nb_val[c] + d
                    if v > seed_bottom[tc]:
                        seed_bottom[tc] = v
            if w2 > 0.0 and dy2 == -1:
                d = _neighbor_dist(dy2, dx2, cellsize_x, cellsize_y)
                tc = c + dx2
                if 0 <= tc < tile_w:
                    v = nb_val[c] + d
                    if v > seed_bottom[tc]:
                        seed_bottom[tc] = v

    # Left edge: right col of tile to the left flows east
    if ix > 0:
        nb_fdir = flow_bdry.get('right', iy, ix - 1)
        nb_val = boundaries.get('right', iy, ix - 1)
        for r in range(len(nb_fdir)):
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(nb_fdir[r])
            if w1 > 0.0 and dx1 == 1:
                d = _neighbor_dist(dy1, dx1, cellsize_x, cellsize_y)
                tr = r + dy1
                if 0 <= tr < tile_h:
                    v = nb_val[r] + d
                    if v > seed_left[tr]:
                        seed_left[tr] = v
            if w2 > 0.0 and dx2 == 1:
                d = _neighbor_dist(dy2, dx2, cellsize_x, cellsize_y)
                tr = r + dy2
                if 0 <= tr < tile_h:
                    v = nb_val[r] + d
                    if v > seed_left[tr]:
                        seed_left[tr] = v

    # Right edge: left col of tile to the right flows west
    if ix < n_tile_x - 1:
        nb_fdir = flow_bdry.get('left', iy, ix + 1)
        nb_val = boundaries.get('left', iy, ix + 1)
        for r in range(len(nb_fdir)):
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(nb_fdir[r])
            if w1 > 0.0 and dx1 == -1:
                d = _neighbor_dist(dy1, dx1, cellsize_x, cellsize_y)
                tr = r + dy1
                if 0 <= tr < tile_h:
                    v = nb_val[r] + d
                    if v > seed_right[tr]:
                        seed_right[tr] = v
            if w2 > 0.0 and dx2 == -1:
                d = _neighbor_dist(dy2, dx2, cellsize_x, cellsize_y)
                tr = r + dy2
                if 0 <= tr < tile_h:
                    v = nb_val[r] + d
                    if v > seed_right[tr]:
                        seed_right[tr] = v

    # Diagonal corners
    diag = math.sqrt(cellsize_x ** 2 + cellsize_y ** 2)

    if iy > 0 and ix > 0:
        fv = flow_bdry.get('bottom', iy - 1, ix - 1)[-1]
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(fv)
        av = float(boundaries.get('bottom', iy - 1, ix - 1)[-1])
        if w1 > 0.0 and dy1 == 1 and dx1 == 1:
            seed_tl = max(seed_tl, av + diag)
        if w2 > 0.0 and dy2 == 1 and dx2 == 1:
            seed_tl = max(seed_tl, av + diag)

    if iy > 0 and ix < n_tile_x - 1:
        fv = flow_bdry.get('bottom', iy - 1, ix + 1)[0]
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(fv)
        av = float(boundaries.get('bottom', iy - 1, ix + 1)[0])
        if w1 > 0.0 and dy1 == 1 and dx1 == -1:
            seed_tr = max(seed_tr, av + diag)
        if w2 > 0.0 and dy2 == 1 and dx2 == -1:
            seed_tr = max(seed_tr, av + diag)

    if iy < n_tile_y - 1 and ix > 0:
        fv = flow_bdry.get('top', iy + 1, ix - 1)[-1]
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(fv)
        av = float(boundaries.get('top', iy + 1, ix - 1)[-1])
        if w1 > 0.0 and dy1 == -1 and dx1 == 1:
            seed_bl = max(seed_bl, av + diag)
        if w2 > 0.0 and dy2 == -1 and dx2 == 1:
            seed_bl = max(seed_bl, av + diag)

    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        fv = flow_bdry.get('top', iy + 1, ix + 1)[0]
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(fv)
        av = float(boundaries.get('top', iy + 1, ix + 1)[0])
        if w1 > 0.0 and dy1 == -1 and dx1 == -1:
            seed_br = max(seed_br, av + diag)
        if w2 > 0.0 and dy2 == -1 and dx2 == -1:
            seed_br = max(seed_br, av + diag)

    return (seed_top, seed_bottom, seed_left, seed_right,
            seed_tl, seed_tr, seed_bl, seed_br)


def _process_tile_downstream(iy, ix, flow_dir_da, boundaries, flow_bdry,
                              chunks_y, chunks_x, n_tile_y, n_tile_x,
                              cellsize_x, cellsize_y):
    chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    h, w = chunk.shape

    seeds = _compute_exit_seeds_downstream(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    result = _flow_length_dinf_downstream_tile(
        chunk, h, w, cellsize_x, cellsize_y, *seeds)

    change = 0.0
    for side, arr in (('top', result[0, :]), ('bottom', result[-1, :]),
                      ('left', result[:, 0]), ('right', result[:, -1])):
        new = arr.copy()
        old = boundaries.get(side, iy, ix)
        with np.errstate(invalid='ignore'):
            diff = np.abs(new - old)
        diff = np.where(np.isnan(diff), 0.0, diff)
        m = float(np.max(diff))
        if m > change:
            change = m
        boundaries.set(side, iy, ix, new)

    return change


def _process_tile_upstream(iy, ix, flow_dir_da, boundaries, flow_bdry,
                            chunks_y, chunks_x, n_tile_y, n_tile_x,
                            cellsize_x, cellsize_y):
    chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    h, w = chunk.shape

    seeds = _compute_entry_seeds_upstream(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x,
        cellsize_x, cellsize_y)

    result = _flow_length_dinf_upstream_tile(
        chunk, h, w, cellsize_x, cellsize_y, *seeds)

    change = 0.0
    for side, arr in (('top', result[0, :]), ('bottom', result[-1, :]),
                      ('left', result[:, 0]), ('right', result[:, -1])):
        new = np.where(np.isnan(arr), 0.0, arr).copy()
        old = boundaries.get(side, iy, ix)
        with np.errstate(invalid='ignore'):
            diff = np.abs(new - old)
        diff = np.where(np.isnan(diff), 0.0, diff)
        m = float(np.max(diff))
        if m > change:
            change = m
        boundaries.set(side, iy, ix, new)

    return change


def _flow_length_dinf_dask_iterative(flow_dir_da, direction,
                                      cellsize_x, cellsize_y):
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry = _preprocess_tiles(flow_dir_da, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()

    fill = np.nan if direction == 'downstream' else 0.0
    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=fill)

    process_fn = (_process_tile_downstream if direction == 'downstream'
                  else _process_tile_upstream)

    max_iterations = max(n_tile_y, n_tile_x) * 2 + 10

    for _iteration in range(max_iterations):
        max_change = 0.0

        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = process_fn(iy, ix, flow_dir_da, boundaries, flow_bdry,
                               chunks_y, chunks_x, n_tile_y, n_tile_x,
                               cellsize_x, cellsize_y)
                if c > max_change:
                    max_change = c

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = process_fn(iy, ix, flow_dir_da, boundaries, flow_bdry,
                               chunks_y, chunks_x, n_tile_y, n_tile_x,
                               cellsize_x, cellsize_y)
                if c > max_change:
                    max_change = c

        if max_change == 0.0:
            break

    boundaries = boundaries.snapshot()

    return _assemble_result(flow_dir_da, boundaries, flow_bdry,
                             chunks_y, chunks_x, n_tile_y, n_tile_x,
                             direction, cellsize_x, cellsize_y)


def _assemble_result(flow_dir_da, boundaries, flow_bdry,
                      chunks_y, chunks_x, n_tile_y, n_tile_x,
                      direction, cellsize_x, cellsize_y):
    rows = []
    for iy in range(n_tile_y):
        row = []
        for ix in range(n_tile_x):
            chunk = np.asarray(
                flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
            h, w = chunk.shape

            if direction == 'downstream':
                seeds = _compute_exit_seeds_downstream(
                    iy, ix, boundaries, flow_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                tile = _flow_length_dinf_downstream_tile(
                    chunk, h, w, cellsize_x, cellsize_y, *seeds)
            else:
                seeds = _compute_entry_seeds_upstream(
                    iy, ix, boundaries, flow_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x,
                    cellsize_x, cellsize_y)
                tile = _flow_length_dinf_upstream_tile(
                    chunk, h, w, cellsize_x, cellsize_y, *seeds)

            row.append(da.from_array(tile, chunks=tile.shape))
        rows.append(row)

    return da.block(rows)


def _flow_length_dinf_dask_cupy(flow_dir_da, direction,
                                 cellsize_x, cellsize_y):
    import cupy as cp

    flow_dir_np = flow_dir_da.map_blocks(
        lambda b: b.get(), dtype=flow_dir_da.dtype,
        meta=np.array((), dtype=flow_dir_da.dtype),
    )
    result = _flow_length_dinf_dask_iterative(
        flow_dir_np, direction, cellsize_x, cellsize_y)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_length_dinf(flow_dir: xr.DataArray,
                      direction: str = 'downstream',
                      name: str = 'flow_length_dinf') -> xr.DataArray:
    """Compute flow length from a D-infinity flow direction grid.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2-D D-infinity flow direction grid as returned by
        ``flow_direction_dinf``.  Values are angles in radians
        ``[0, 2*pi)``, ``-1.0`` for pits/flats, ``NaN`` for nodata.
        Supported backends: NumPy, CuPy, NumPy-backed Dask,
        CuPy-backed Dask.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    direction : str, default 'downstream'
        ``'downstream'``: proportion-weighted average distance from
        each cell to its outlet.  Flow is split between two neighbors
        following Tarboton (1997) angle decomposition.
        ``'upstream'``: longest path from any divide to each cell.
    name : str, default 'flow_length_dinf'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2-D float64 array of flow length values in coordinate units.
        NaN where flow_dir is NaN.

    References
    ----------
    Tarboton, D.G. (1997). A new method for the determination of flow
    directions and upslope areas in grid digital elevation models.
    Water Resources Research, 33(2), 309-319.
    """
    _validate_raster(flow_dir, func_name='flow_length_dinf', name='flow_dir')

    if direction not in ('downstream', 'upstream'):
        raise ValueError(
            f"direction must be 'downstream' or 'upstream', got {direction!r}")

    cellsize_x, cellsize_y = get_dataarray_resolution(flow_dir)
    cellsize_x = abs(cellsize_x)
    cellsize_y = abs(cellsize_y)

    data = flow_dir.data

    if isinstance(data, np.ndarray):
        fd = data.astype(np.float64)
        H, W = fd.shape
        if direction == 'downstream':
            out = _flow_length_dinf_downstream_cpu(
                fd, H, W, cellsize_x, cellsize_y)
        else:
            out = _flow_length_dinf_upstream_cpu(
                fd, H, W, cellsize_x, cellsize_y)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _flow_length_dinf_cupy(data, direction, cellsize_x, cellsize_y)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _flow_length_dinf_dask_cupy(data, direction,
                                           cellsize_x, cellsize_y)

    elif da is not None and isinstance(data, da.Array):
        out = _flow_length_dinf_dask_iterative(data, direction,
                                                cellsize_x, cellsize_y)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
