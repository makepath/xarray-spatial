"""MFD flow length: proportion-weighted distance from each cell to outlet
(downstream) or longest path from divide to cell (upstream).

Algorithm
---------
CPU : Kahn's BFS topological sort on the MFD DAG -- O(N).
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

from xrspatial.hydrology._boundary_store import BoundaryStore
from xrspatial.utils import (
    _validate_raster,
    get_dataarray_resolution,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)
from xrspatial.dataset_support import supports_dataset

# Neighbor offsets: E, SE, S, SW, W, NW, N, NE
_DY = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)


# =====================================================================
# Step-distance helper
# =====================================================================

@ngjit
def _step_distances(cellsize_x, cellsize_y):
    """Return array of 8 step distances for each neighbor direction."""
    diag = math.sqrt(cellsize_x * cellsize_x + cellsize_y * cellsize_y)
    dists = np.empty(8, dtype=np.float64)
    # E, SE, S, SW, W, NW, N, NE
    dists[0] = cellsize_x
    dists[1] = diag
    dists[2] = cellsize_y
    dists[3] = diag
    dists[4] = cellsize_x
    dists[5] = diag
    dists[6] = cellsize_y
    dists[7] = diag
    return dists


# =====================================================================
# CPU kernels
# =====================================================================

@ngjit
def _flow_length_mfd_downstream_cpu(fractions, H, W, cellsize_x, cellsize_y):
    """Downstream MFD flow length: proportion-weighted average distance
    from each cell to its outlet(s).

    Two-pass O(N):
    1. Kahn's BFS builds topological order (divides first).
    2. Reverse pass (outlets first): flow_len[cell] =
       sum_k(frac[k] * (step_dist[k] + flow_len[neighbor_k]))
    """
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    dists = _step_distances(cellsize_x, cellsize_y)

    in_degree = np.zeros((H, W), dtype=np.int32)
    valid = np.zeros((H, W), dtype=np.int8)
    flow_len = np.empty((H, W), dtype=np.float64)

    # Init
    for r in range(H):
        for c in range(W):
            v = fractions[0, r, c]
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
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
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
        for k in range(8):
            if fractions[k, r, c] > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
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
        total = 0.0
        for k in range(8):
            frac = fractions[k, r, c]
            if frac > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                    total += frac * (dists[k] + flow_len[nr, nc])
                else:
                    # Flows off grid: distance = 0 at destination
                    total += frac * dists[k]
        flow_len[r, c] = total

    return flow_len


@ngjit
def _flow_length_mfd_upstream_cpu(fractions, H, W, cellsize_x, cellsize_y):
    """Upstream MFD flow length: longest path from any divide to cell.

    Kahn's BFS from divides downstream, propagating max distance.
    """
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    dists = _step_distances(cellsize_x, cellsize_y)

    in_degree = np.zeros((H, W), dtype=np.int32)
    valid = np.zeros((H, W), dtype=np.int8)
    flow_len = np.empty((H, W), dtype=np.float64)

    for r in range(H):
        for c in range(W):
            v = fractions[0, r, c]
            if v == v:
                valid[r, c] = 1
                flow_len[r, c] = 0.0
            else:
                flow_len[r, c] = np.nan

    for r in range(H):
        for c in range(W):
            if valid[r, c] == 0:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
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

        for k in range(8):
            frac = fractions[k, r, c]
            if frac > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                    candidate = flow_len[r, c] + dists[k]
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

def _flow_length_mfd_cupy(fractions_data, direction, cellsize_x, cellsize_y):
    import cupy as cp
    frac_np = fractions_data.get().astype(np.float64)
    _, H, W = frac_np.shape
    if direction == 'downstream':
        out = _flow_length_mfd_downstream_cpu(frac_np, H, W,
                                               cellsize_x, cellsize_y)
    else:
        out = _flow_length_mfd_upstream_cpu(frac_np, H, W,
                                             cellsize_x, cellsize_y)
    return cp.asarray(out)


# =====================================================================
# Dask tile kernels
# =====================================================================

@ngjit
def _flow_length_mfd_downstream_tile(fractions, h, w, cellsize_x, cellsize_y,
                                      seed_top, seed_bottom,
                                      seed_left, seed_right,
                                      seed_tl, seed_tr, seed_bl, seed_br):
    """Downstream MFD flow length for one tile with exit seeds.

    Boundary cells that flow out of the tile use seed values as the
    known downstream flow_length at their destination.
    """
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    dists = _step_distances(cellsize_x, cellsize_y)

    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)
    flow_len = np.empty((h, w), dtype=np.float64)

    # Init
    for r in range(h):
        for c in range(w):
            v = fractions[0, r, c]
            if v == v:
                valid[r, c] = 1
                flow_len[r, c] = 0.0
            else:
                flow_len[r, c] = np.nan

    # In-degrees (only for edges within tile)
    for r in range(h):
        for c in range(w):
            if valid[r, c] == 0:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
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
        for k in range(8):
            if fractions[k, r, c] > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
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
        total = 0.0
        for k in range(8):
            frac = fractions[k, r, c]
            if frac > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                    total += frac * (dists[k] + flow_len[nr, nc])
                else:
                    # Flows out of tile -- look up exit seed
                    exit_val = _get_exit_seed(
                        r, c, nr, nc, h, w,
                        seed_top, seed_bottom, seed_left, seed_right,
                        seed_tl, seed_tr, seed_bl, seed_br)
                    if exit_val == exit_val:  # not NaN
                        total += frac * (dists[k] + exit_val)
                    else:
                        # Grid edge -- distance = step only
                        total += frac * dists[k]
        flow_len[r, c] = total

    return flow_len


@ngjit
def _get_exit_seed(r, c, nr, nc, h, w,
                   seed_top, seed_bottom, seed_left, seed_right,
                   seed_tl, seed_tr, seed_bl, seed_br):
    """Look up the exit seed for a cell flowing out of the tile."""
    # Corner cases first
    if nr < 0 and nc < 0:
        return seed_tl
    if nr < 0 and nc >= w:
        return seed_tr
    if nr >= h and nc < 0:
        return seed_bl
    if nr >= h and nc >= w:
        return seed_br
    # Edge cases
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
def _flow_length_mfd_upstream_tile(fractions, h, w, cellsize_x, cellsize_y,
                                    seed_top, seed_bottom,
                                    seed_left, seed_right,
                                    seed_tl, seed_tr, seed_bl, seed_br):
    """Upstream MFD flow length for one tile with entry seeds."""
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    dists = _step_distances(cellsize_x, cellsize_y)

    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)
    flow_len = np.empty((h, w), dtype=np.float64)

    for r in range(h):
        for c in range(w):
            v = fractions[0, r, c]
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
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
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
        for k in range(8):
            frac = fractions[k, r, c]
            if frac > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                    candidate = flow_len[r, c] + dists[k]
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

def _preprocess_mfd_tiles(fractions_da, chunks_y, chunks_x):
    """Extract boundary fraction strips into a dict."""
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    frac_bdry = {}

    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            chunk = fractions_da[
                :,
                sum(chunks_y[:iy]):sum(chunks_y[:iy + 1]),
                sum(chunks_x[:ix]):sum(chunks_x[:ix + 1]),
            ].compute()
            chunk = np.asarray(chunk, dtype=np.float64)

            frac_bdry[('top', iy, ix)] = chunk[:, 0, :].copy()
            frac_bdry[('bottom', iy, ix)] = chunk[:, -1, :].copy()
            frac_bdry[('left', iy, ix)] = chunk[:, :, 0].copy()
            frac_bdry[('right', iy, ix)] = chunk[:, :, -1].copy()

    return frac_bdry


def _compute_exit_seeds_downstream(iy, ix, boundaries, frac_bdry,
                                    chunks_y, chunks_x,
                                    n_tile_y, n_tile_x):
    """For downstream: provide flow_length values at destination cells
    in adjacent tiles.

    The tile kernel looks up ``seed_top[nc]`` for a cell flowing to
    destination column ``nc`` in the tile above, so each seed array
    is simply a copy of the adjacent tile's facing boundary.
    """
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

    # Top: destinations are in the bottom row of tile (iy-1, ix)
    if iy > 0:
        nb = boundaries.get('bottom', iy - 1, ix)
        seed_top[:len(nb)] = nb

    # Bottom: destinations are in the top row of tile (iy+1, ix)
    if iy < n_tile_y - 1:
        nb = boundaries.get('top', iy + 1, ix)
        seed_bottom[:len(nb)] = nb

    # Left: destinations are in the right col of tile (iy, ix-1)
    if ix > 0:
        nb = boundaries.get('right', iy, ix - 1)
        seed_left[:len(nb)] = nb

    # Right: destinations are in the left col of tile (iy, ix+1)
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


def _compute_entry_seeds_upstream(iy, ix, boundaries, frac_bdry,
                                   chunks_y, chunks_x,
                                   n_tile_y, n_tile_x,
                                   cellsize_x, cellsize_y):
    """For upstream: check which adjacent cells flow INTO this tile,
    seed with max(existing, neighbor_upstream + step_dist)."""
    dy_arr = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx_arr = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    dists = _step_distances(cellsize_x, cellsize_y)

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
        nb_frac = frac_bdry[('bottom', iy - 1, ix)]  # (8, w)
        nb_val = boundaries.get('bottom', iy - 1, ix)
        w = nb_frac.shape[1]
        for c in range(w):
            for k in range(8):
                if not (nb_frac[k, c] > 0.0):  # rejects NaN
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndy != 1:  # must flow south
                    continue
                tc = c + ndx
                if 0 <= tc < tile_w:
                    v = nb_val[c] + dists[k]
                    if v > seed_top[tc]:
                        seed_top[tc] = v

    # Bottom edge: top row of tile below flows north into this tile
    if iy < n_tile_y - 1:
        nb_frac = frac_bdry[('top', iy + 1, ix)]
        nb_val = boundaries.get('top', iy + 1, ix)
        w = nb_frac.shape[1]
        for c in range(w):
            for k in range(8):
                if not (nb_frac[k, c] > 0.0):  # rejects NaN
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndy != -1:
                    continue
                tc = c + ndx
                if 0 <= tc < tile_w:
                    v = nb_val[c] + dists[k]
                    if v > seed_bottom[tc]:
                        seed_bottom[tc] = v

    # Left edge: right col of tile to the left flows east into this tile
    if ix > 0:
        nb_frac = frac_bdry[('right', iy, ix - 1)]
        nb_val = boundaries.get('right', iy, ix - 1)
        h = nb_frac.shape[1]
        for r in range(h):
            for k in range(8):
                if not (nb_frac[k, r] > 0.0):  # rejects NaN
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndx != 1:
                    continue
                tr = r + ndy
                if 0 <= tr < tile_h:
                    v = nb_val[r] + dists[k]
                    if v > seed_left[tr]:
                        seed_left[tr] = v

    # Right edge: left col of tile to the right flows west into this tile
    if ix < n_tile_x - 1:
        nb_frac = frac_bdry[('left', iy, ix + 1)]
        nb_val = boundaries.get('left', iy, ix + 1)
        h = nb_frac.shape[1]
        for r in range(h):
            for k in range(8):
                if not (nb_frac[k, r] > 0.0):  # rejects NaN
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndx != -1:
                    continue
                tr = r + ndy
                if 0 <= tr < tile_h:
                    v = nb_val[r] + dists[k]
                    if v > seed_right[tr]:
                        seed_right[tr] = v

    # Diagonal corner seeds
    diag = dists[1]  # diagonal distance

    # TL: bottom-right cell of (iy-1, ix-1) flows SE (k=1)
    if iy > 0 and ix > 0:
        nb_frac = frac_bdry[('bottom', iy - 1, ix - 1)]
        if nb_frac[1, -1] > 0.0:
            val = boundaries.get('bottom', iy - 1, ix - 1)[-1]
            if val == val:
                seed_tl = val + diag

    # TR: bottom-left cell of (iy-1, ix+1) flows SW (k=3)
    if iy > 0 and ix < n_tile_x - 1:
        nb_frac = frac_bdry[('bottom', iy - 1, ix + 1)]
        if nb_frac[3, 0] > 0.0:
            val = boundaries.get('bottom', iy - 1, ix + 1)[0]
            if val == val:
                seed_tr = val + diag

    # BL: top-right cell of (iy+1, ix-1) flows NE (k=7)
    if iy < n_tile_y - 1 and ix > 0:
        nb_frac = frac_bdry[('top', iy + 1, ix - 1)]
        if nb_frac[7, -1] > 0.0:
            val = boundaries.get('top', iy + 1, ix - 1)[-1]
            if val == val:
                seed_bl = val + diag

    # BR: top-left cell of (iy+1, ix+1) flows NW (k=5)
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        nb_frac = frac_bdry[('top', iy + 1, ix + 1)]
        if nb_frac[5, 0] > 0.0:
            val = boundaries.get('top', iy + 1, ix + 1)[0]
            if val == val:
                seed_br = val + diag

    return (seed_top, seed_bottom, seed_left, seed_right,
            seed_tl, seed_tr, seed_bl, seed_br)


def _process_tile_downstream(iy, ix, fractions_da, boundaries, frac_bdry,
                              chunks_y, chunks_x, n_tile_y, n_tile_x,
                              cellsize_x, cellsize_y):
    """Process one tile for downstream MFD flow length."""
    y_start = sum(chunks_y[:iy])
    y_end = y_start + chunks_y[iy]
    x_start = sum(chunks_x[:ix])
    x_end = x_start + chunks_x[ix]

    chunk = np.asarray(
        fractions_da[:, y_start:y_end, x_start:x_end].compute(),
        dtype=np.float64)
    _, h, w = chunk.shape

    seeds = _compute_exit_seeds_downstream(
        iy, ix, boundaries, frac_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    result = _flow_length_mfd_downstream_tile(
        chunk, h, w, cellsize_x, cellsize_y, *seeds)

    new_top = np.where(np.isnan(result[0, :]), np.nan, result[0, :])
    new_bottom = np.where(np.isnan(result[-1, :]), np.nan, result[-1, :])
    new_left = np.where(np.isnan(result[:, 0]), np.nan, result[:, 0])
    new_right = np.where(np.isnan(result[:, -1]), np.nan, result[:, -1])

    change = 0.0
    for side, new in (('top', new_top), ('bottom', new_bottom),
                      ('left', new_left), ('right', new_right)):
        old = boundaries.get(side, iy, ix)
        with np.errstate(invalid='ignore'):
            diff = np.abs(new - old)
        diff = np.where(np.isnan(diff), 0.0, diff)
        m = float(np.max(diff))
        if m > change:
            change = m

    boundaries.set('top', iy, ix, new_top)
    boundaries.set('bottom', iy, ix, new_bottom)
    boundaries.set('left', iy, ix, new_left)
    boundaries.set('right', iy, ix, new_right)

    return change


def _process_tile_upstream(iy, ix, fractions_da, boundaries, frac_bdry,
                            chunks_y, chunks_x, n_tile_y, n_tile_x,
                            cellsize_x, cellsize_y):
    """Process one tile for upstream MFD flow length."""
    y_start = sum(chunks_y[:iy])
    y_end = y_start + chunks_y[iy]
    x_start = sum(chunks_x[:ix])
    x_end = x_start + chunks_x[ix]

    chunk = np.asarray(
        fractions_da[:, y_start:y_end, x_start:x_end].compute(),
        dtype=np.float64)
    _, h, w = chunk.shape

    seeds = _compute_entry_seeds_upstream(
        iy, ix, boundaries, frac_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x,
        cellsize_x, cellsize_y)

    result = _flow_length_mfd_upstream_tile(
        chunk, h, w, cellsize_x, cellsize_y, *seeds)

    new_top = np.where(np.isnan(result[0, :]), 0.0, result[0, :])
    new_bottom = np.where(np.isnan(result[-1, :]), 0.0, result[-1, :])
    new_left = np.where(np.isnan(result[:, 0]), 0.0, result[:, 0])
    new_right = np.where(np.isnan(result[:, -1]), 0.0, result[:, -1])

    change = 0.0
    for side, new in (('top', new_top), ('bottom', new_bottom),
                      ('left', new_left), ('right', new_right)):
        old = boundaries.get(side, iy, ix)
        with np.errstate(invalid='ignore'):
            diff = np.abs(new - old)
        diff = np.where(np.isnan(diff), 0.0, diff)
        m = float(np.max(diff))
        if m > change:
            change = m

    boundaries.set('top', iy, ix, new_top)
    boundaries.set('bottom', iy, ix, new_bottom)
    boundaries.set('left', iy, ix, new_left)
    boundaries.set('right', iy, ix, new_right)

    return change


def _flow_length_mfd_dask_iterative(fractions_da, direction,
                                      cellsize_x, cellsize_y,
                                      chunks_y, chunks_x):
    """Iterative boundary-propagation for MFD flow length on dask arrays."""
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    frac_bdry = _preprocess_mfd_tiles(fractions_da, chunks_y, chunks_x)

    fill = np.nan if direction == 'downstream' else 0.0
    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=fill)

    if direction == 'downstream':
        process_fn = _process_tile_downstream
    else:
        process_fn = _process_tile_upstream

    max_iterations = max(n_tile_y, n_tile_x) * 2 + 10

    for _iteration in range(max_iterations):
        max_change = 0.0

        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = process_fn(iy, ix, fractions_da, boundaries, frac_bdry,
                               chunks_y, chunks_x, n_tile_y, n_tile_x,
                               cellsize_x, cellsize_y)
                if c > max_change:
                    max_change = c

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = process_fn(iy, ix, fractions_da, boundaries, frac_bdry,
                               chunks_y, chunks_x, n_tile_y, n_tile_x,
                               cellsize_x, cellsize_y)
                if c > max_change:
                    max_change = c

        if max_change == 0.0:
            break

    boundaries = boundaries.snapshot()

    return _assemble_result(fractions_da, boundaries, frac_bdry,
                             chunks_y, chunks_x, n_tile_y, n_tile_x,
                             direction, cellsize_x, cellsize_y)


def _assemble_result(fractions_da, boundaries, frac_bdry,
                      chunks_y, chunks_x, n_tile_y, n_tile_x,
                      direction, cellsize_x, cellsize_y):
    """Build dask array by re-running tiles with converged boundaries."""
    rows = []
    for iy in range(n_tile_y):
        row = []
        for ix in range(n_tile_x):
            y_start = sum(chunks_y[:iy])
            y_end = y_start + chunks_y[iy]
            x_start = sum(chunks_x[:ix])
            x_end = x_start + chunks_x[ix]

            chunk = np.asarray(
                fractions_da[:, y_start:y_end, x_start:x_end].compute(),
                dtype=np.float64)
            _, h, w = chunk.shape

            if direction == 'downstream':
                seeds = _compute_exit_seeds_downstream(
                    iy, ix, boundaries, frac_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                tile = _flow_length_mfd_downstream_tile(
                    chunk, h, w, cellsize_x, cellsize_y, *seeds)
            else:
                seeds = _compute_entry_seeds_upstream(
                    iy, ix, boundaries, frac_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x,
                    cellsize_x, cellsize_y)
                tile = _flow_length_mfd_upstream_tile(
                    chunk, h, w, cellsize_x, cellsize_y, *seeds)

            row.append(da.from_array(tile, chunks=tile.shape))
        rows.append(row)

    return da.block(rows)


def _flow_length_mfd_dask_cupy(fractions_da, direction,
                                cellsize_x, cellsize_y,
                                chunks_y, chunks_x):
    """Dask+CuPy: convert to numpy, run CPU iterative, convert back."""
    import cupy as cp

    fractions_np = fractions_da.map_blocks(
        lambda b: b.get(), dtype=fractions_da.dtype,
        meta=np.array((), dtype=fractions_da.dtype),
    )
    result = _flow_length_mfd_dask_iterative(
        fractions_np, direction, cellsize_x, cellsize_y,
        chunks_y, chunks_x)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_length_mfd(flow_dir_mfd: xr.DataArray,
                     direction: str = 'downstream',
                     name: str = 'flow_length_mfd') -> xr.DataArray:
    """Compute flow length from an MFD flow direction grid.

    Parameters
    ----------
    flow_dir_mfd : xarray.DataArray or xr.Dataset
        3-D MFD flow direction array of shape ``(8, H, W)`` as returned
        by ``flow_direction_mfd``.  Values are flow fractions in
        ``[0, 1]`` that sum to 1.0 at each cell (0.0 at pits/flats,
        NaN at edges or nodata cells).
        Supported backends: NumPy, CuPy, NumPy-backed Dask,
        CuPy-backed Dask.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    direction : str, default 'downstream'
        ``'downstream'``: proportion-weighted average distance from
        each cell to its outlet.  Each cell gets the expected distance
        across all MFD flow paths.
        ``'upstream'``: longest path from any divide to each cell.
    name : str, default 'flow_length_mfd'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2-D float64 array of flow length values in coordinate units.
        NaN where the input has NaN.

    References
    ----------
    Qin, C., Zhu, A.X., Pei, T., Li, B., Zhou, C., and Yang, L.
    (2007). An adaptive approach to selecting a flow-partition
    exponent for a multiple-flow-direction algorithm. International
    Journal of Geographical Information Science, 21(4), 443-458.

    Quinn, P., Beven, K., Chevallier, P., and Planchon, O. (1991).
    The prediction of hillslope flow paths for distributed
    hydrological modelling using digital terrain models.
    Hydrological Processes, 5(1), 59-79.
    """
    _validate_raster(flow_dir_mfd, func_name='flow_length_mfd',
                     name='flow_dir_mfd', ndim=3)

    if direction not in ('downstream', 'upstream'):
        raise ValueError(
            f"direction must be 'downstream' or 'upstream', got {direction!r}")

    data = flow_dir_mfd.data

    if data.ndim != 3 or data.shape[0] != 8:
        raise ValueError(
            "flow_dir_mfd must be a 3-D array of shape (8, H, W), "
            f"got shape {data.shape}"
        )

    cellsize_x, cellsize_y = get_dataarray_resolution(flow_dir_mfd)
    cellsize_x = abs(cellsize_x)
    cellsize_y = abs(cellsize_y)

    if isinstance(data, np.ndarray):
        frac = data.astype(np.float64)
        _, H, W = frac.shape
        if direction == 'downstream':
            out = _flow_length_mfd_downstream_cpu(
                frac, H, W, cellsize_x, cellsize_y)
        else:
            out = _flow_length_mfd_upstream_cpu(
                frac, H, W, cellsize_x, cellsize_y)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _flow_length_mfd_cupy(data, direction, cellsize_x, cellsize_y)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir_mfd):
        chunks_y = data.chunks[1]
        chunks_x = data.chunks[2]
        out = _flow_length_mfd_dask_cupy(
            data, direction, cellsize_x, cellsize_y, chunks_y, chunks_x)

    elif da is not None and isinstance(data, da.Array):
        chunks_y = data.chunks[1]
        chunks_x = data.chunks[2]
        out = _flow_length_mfd_dask_iterative(
            data, direction, cellsize_x, cellsize_y, chunks_y, chunks_x)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    # Build 2-D output coords (drop 'neighbor' dim)
    spatial_dims = flow_dir_mfd.dims[1:]
    coords = {k: v for k, v in flow_dir_mfd.coords.items()
              if k != 'neighbor' and k not in flow_dir_mfd.dims[:1]}
    for d in spatial_dims:
        if d in flow_dir_mfd.coords:
            coords[d] = flow_dir_mfd.coords[d]

    return xr.DataArray(out,
                        name=name,
                        coords=coords,
                        dims=spatial_dims,
                        attrs=flow_dir_mfd.attrs)
