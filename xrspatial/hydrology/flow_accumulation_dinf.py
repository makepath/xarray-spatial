"""Flow accumulation for D-infinity (continuous angle) flow direction grids.

Takes the continuous-angle output from ``flow_direction_dinf`` and
accumulates upstream area, splitting flow proportionally between two
neighbors following Tarboton (1997).

Algorithm
---------
CPU : Kahn's BFS topological sort -- O(N).
GPU : iterative frontier peeling with pull-based kernels.
Dask: iterative tile sweep with boundary propagation (one tile in
      RAM at a time), following the ``flow_accumulation.py`` pattern.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numba import cuda

try:
    import cupy
except ImportError:
    class cupy:  # type: ignore[no-redef]
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.utils import (
    _validate_raster,
    cuda_args,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)
from xrspatial.hydrology._boundary_store import BoundaryStore
from xrspatial.dataset_support import supports_dataset
from xrspatial.hydrology.flow_accumulation import (
    _find_ready_and_finalize,
    _preprocess_tiles,
)


def _to_numpy_f64(arr):
    """Convert *arr* to a contiguous numpy float64 array.

    Handles CuPy arrays transparently via ``.get()``.
    """
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


# =====================================================================
# Dinf helpers
# =====================================================================

@ngjit
def _angle_to_neighbors(angle):
    """Decompose Dinf angle into two neighbors and weights.

    Returns (dy1, dx1, w1, dy2, dx2, w2).
    Pit (angle < 0) or NaN returns all zeros.
    """
    if angle < 0.0 or angle != angle:  # pit or NaN
        return 0, 0, 0.0, 0, 0, 0.0

    pi_over_4 = 0.7853981633974483  # np.pi / 4
    k = int(angle / pi_over_4)
    if k > 7:
        k = 7
    alpha = angle - k * pi_over_4
    w1 = 1.0 - alpha / pi_over_4
    w2 = alpha / pi_over_4

    if k == 0:
        dy1, dx1 = 0, 1      # E
        dy2, dx2 = -1, 1     # NE
    elif k == 1:
        dy1, dx1 = -1, 1     # NE
        dy2, dx2 = -1, 0     # N
    elif k == 2:
        dy1, dx1 = -1, 0     # N
        dy2, dx2 = -1, -1    # NW
    elif k == 3:
        dy1, dx1 = -1, -1    # NW
        dy2, dx2 = 0, -1     # W
    elif k == 4:
        dy1, dx1 = 0, -1     # W
        dy2, dx2 = 1, -1     # SW
    elif k == 5:
        dy1, dx1 = 1, -1     # SW
        dy2, dx2 = 1, 0      # S
    elif k == 6:
        dy1, dx1 = 1, 0      # S
        dy2, dx2 = 1, 1      # SE
    else:  # k == 7
        dy1, dx1 = 1, 1      # SE
        dy2, dx2 = 0, 1      # E

    return dy1, dx1, w1, dy2, dx2, w2


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _flow_accum_dinf_cpu(flow_dir, height, width):
    """Kahn's BFS topological sort for Dinf flow accumulation."""
    accum = np.empty((height, width), dtype=np.float64)
    in_degree = np.zeros((height, width), dtype=np.int32)
    valid = np.zeros((height, width), dtype=np.int8)

    for r in range(height):
        for c in range(width):
            v = flow_dir[r, c]
            if v == v:  # not NaN
                valid[r, c] = 1
                accum[r, c] = 1.0
            else:
                accum[r, c] = np.nan

    for r in range(height):
        for c in range(width):
            if valid[r, c] == 0:
                continue
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
            if w1 > 0.0:
                nr, nc = r + dy1, c + dx1
                if 0 <= nr < height and 0 <= nc < width and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1
            if w2 > 0.0:
                nr, nc = r + dy2, c + dx2
                if 0 <= nr < height and 0 <= nc < width and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1

    queue_r = np.empty(height * width, dtype=np.int64)
    queue_c = np.empty(height * width, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for r in range(height):
        for c in range(width):
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
            if 0 <= nr < height and 0 <= nc < width and valid[nr, nc] == 1:
                accum[nr, nc] += accum[r, c] * w1
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

        if w2 > 0.0:
            nr, nc = r + dy2, c + dx2
            if 0 <= nr < height and 0 <= nc < width and valid[nr, nc] == 1:
                accum[nr, nc] += accum[r, c] * w2
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return accum


# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit
def _init_accum_indegree_dinf(flow_dir, accum, in_degree, state, H, W):
    """Initialise accum/in_degree/state for Dinf on GPU."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    v = flow_dir[i, j]
    if v != v:  # NaN
        state[i, j] = 0
        accum[i, j] = 0.0
        return

    state[i, j] = 1
    accum[i, j] = 1.0

    if v < 0.0:  # pit
        return

    pi_over_4 = 0.7853981633974483
    k = int(v / pi_over_4)
    if k > 7:
        k = 7
    alpha = v - k * pi_over_4
    w1 = 1.0 - alpha / pi_over_4
    w2 = alpha / pi_over_4

    # Facet offsets (inline)
    if k == 0:
        dy1, dx1 = 0, 1
        dy2, dx2 = -1, 1
    elif k == 1:
        dy1, dx1 = -1, 1
        dy2, dx2 = -1, 0
    elif k == 2:
        dy1, dx1 = -1, 0
        dy2, dx2 = -1, -1
    elif k == 3:
        dy1, dx1 = -1, -1
        dy2, dx2 = 0, -1
    elif k == 4:
        dy1, dx1 = 0, -1
        dy2, dx2 = 1, -1
    elif k == 5:
        dy1, dx1 = 1, -1
        dy2, dx2 = 1, 0
    elif k == 6:
        dy1, dx1 = 1, 0
        dy2, dx2 = 1, 1
    else:
        dy1, dx1 = 1, 1
        dy2, dx2 = 0, 1

    if w1 > 0.0:
        ni, nj = i + dy1, j + dx1
        if 0 <= ni < H and 0 <= nj < W:
            cuda.atomic.add(in_degree, (ni, nj), 1)
    if w2 > 0.0:
        ni, nj = i + dy2, j + dx2
        if 0 <= ni < H and 0 <= nj < W:
            cuda.atomic.add(in_degree, (ni, nj), 1)


@cuda.jit
def _pull_from_frontier_dinf(flow_dir, accum, in_degree, state, H, W):
    """Active Dinf cells pull accumulation from frontier neighbours."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if state[i, j] != 1:
        return

    pi_over_4 = 0.7853981633974483

    for nbr in range(8):
        if nbr == 0:
            dy, dx = 0, 1
        elif nbr == 1:
            dy, dx = 1, 1
        elif nbr == 2:
            dy, dx = 1, 0
        elif nbr == 3:
            dy, dx = 1, -1
        elif nbr == 4:
            dy, dx = 0, -1
        elif nbr == 5:
            dy, dx = -1, -1
        elif nbr == 6:
            dy, dx = -1, 0
        else:
            dy, dx = -1, 1

        ni = i + dy
        nj = j + dx
        if ni < 0 or ni >= H or nj < 0 or nj >= W:
            continue
        if state[ni, nj] != 2:
            continue

        nv = flow_dir[ni, nj]
        if nv < 0.0 or nv != nv:
            continue

        nk = int(nv / pi_over_4)
        if nk > 7:
            nk = 7
        nalpha = nv - nk * pi_over_4
        nw1 = 1.0 - nalpha / pi_over_4
        nw2 = nalpha / pi_over_4

        # Inline facet offsets for neighbor's direction
        if nk == 0:
            ndy1, ndx1 = 0, 1
            ndy2, ndx2 = -1, 1
        elif nk == 1:
            ndy1, ndx1 = -1, 1
            ndy2, ndx2 = -1, 0
        elif nk == 2:
            ndy1, ndx1 = -1, 0
            ndy2, ndx2 = -1, -1
        elif nk == 3:
            ndy1, ndx1 = -1, -1
            ndy2, ndx2 = 0, -1
        elif nk == 4:
            ndy1, ndx1 = 0, -1
            ndy2, ndx2 = 1, -1
        elif nk == 5:
            ndy1, ndx1 = 1, -1
            ndy2, ndx2 = 1, 0
        elif nk == 6:
            ndy1, ndx1 = 1, 0
            ndy2, ndx2 = 1, 1
        else:
            ndy1, ndx1 = 1, 1
            ndy2, ndx2 = 0, 1

        if nw1 > 0.0 and ni + ndy1 == i and nj + ndx1 == j:
            accum[i, j] += accum[ni, nj] * nw1
            in_degree[i, j] -= 1
        if nw2 > 0.0 and ni + ndy2 == i and nj + ndx2 == j:
            accum[i, j] += accum[ni, nj] * nw2
            in_degree[i, j] -= 1


def _flow_accum_dinf_cupy(flow_dir_data):
    """GPU driver: iterative frontier peeling for Dinf."""
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)

    accum = cp.zeros((H, W), dtype=cp.float64)
    in_degree = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _init_accum_indegree_dinf[griddim, blockdim](
        flow_dir_f64, accum, in_degree, state, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _find_ready_and_finalize[griddim, blockdim](
            in_degree, state, changed, H, W)

        if int(changed[0]) == 0:
            break

        _pull_from_frontier_dinf[griddim, blockdim](
            flow_dir_f64, accum, in_degree, state, H, W)

    accum = cp.where(state == 0, cp.nan, accum)
    return accum


def _flow_accum_dinf_tile_cupy(flow_dir_data,
                                seed_top, seed_bottom, seed_left, seed_right,
                                seed_tl, seed_tr, seed_bl, seed_br):
    """GPU seeded Dinf flow accumulation for a single tile."""
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)

    accum = cp.zeros((H, W), dtype=cp.float64)
    in_degree = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _init_accum_indegree_dinf[griddim, blockdim](
        flow_dir_f64, accum, in_degree, state, H, W)

    accum[0, :] += cp.asarray(seed_top)
    accum[H - 1, :] += cp.asarray(seed_bottom)
    accum[:, 0] += cp.asarray(seed_left)
    accum[:, W - 1] += cp.asarray(seed_right)
    accum[0, 0] += seed_tl
    accum[0, W - 1] += seed_tr
    accum[H - 1, 0] += seed_bl
    accum[H - 1, W - 1] += seed_br

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _find_ready_and_finalize[griddim, blockdim](
            in_degree, state, changed, H, W)

        if int(changed[0]) == 0:
            break

        _pull_from_frontier_dinf[griddim, blockdim](
            flow_dir_f64, accum, in_degree, state, H, W)

    accum = cp.where(state == 0, cp.nan, accum)
    return accum


# =====================================================================
# Dinf tile kernel for dask
# =====================================================================

@ngjit
def _flow_accum_dinf_tile_kernel(flow_dir, h, w,
                                  seed_top, seed_bottom,
                                  seed_left, seed_right,
                                  seed_tl, seed_tr, seed_bl, seed_br):
    """Seeded BFS Dinf flow accumulation for a single tile."""
    accum = np.empty((h, w), dtype=np.float64)
    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)

    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v == v:
                valid[r, c] = 1
                accum[r, c] = 1.0
            else:
                accum[r, c] = np.nan

    # Add external seeds
    for c in range(w):
        if valid[0, c] == 1:
            accum[0, c] += seed_top[c]
        if valid[h - 1, c] == 1:
            accum[h - 1, c] += seed_bottom[c]
    for r in range(h):
        if valid[r, 0] == 1:
            accum[r, 0] += seed_left[r]
        if valid[r, w - 1] == 1:
            accum[r, w - 1] += seed_right[r]
    if valid[0, 0] == 1:
        accum[0, 0] += seed_tl
    if valid[0, w - 1] == 1:
        accum[0, w - 1] += seed_tr
    if valid[h - 1, 0] == 1:
        accum[h - 1, 0] += seed_bl
    if valid[h - 1, w - 1] == 1:
        accum[h - 1, w - 1] += seed_br

    # In-degrees via angle decomposition
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
                accum[nr, nc] += accum[r, c] * w1
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

        if w2 > 0.0:
            nr, nc = r + dy2, c + dx2
            if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                accum[nr, nc] += accum[r, c] * w2
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return accum


# =====================================================================
# Dinf dask iterative tile sweep
# =====================================================================

def _compute_seeds_dinf(iy, ix, boundaries, flow_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute seed arrays for tile (iy, ix) using Dinf angle decomposition."""
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

    # --- Top edge: bottom row of tile above flows south (dy=1) ---
    if iy > 0:
        nb_fdir = flow_bdry.get('bottom', iy - 1, ix)
        nb_accum = boundaries.get('bottom', iy - 1, ix)
        for c in range(len(nb_fdir)):
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(nb_fdir[c])
            if w1 > 0.0 and dy1 == 1:
                tc = c + dx1
                if 0 <= tc < tile_w:
                    seed_top[tc] += nb_accum[c] * w1
            if w2 > 0.0 and dy2 == 1:
                tc = c + dx2
                if 0 <= tc < tile_w:
                    seed_top[tc] += nb_accum[c] * w2

    # --- Bottom edge: top row of tile below flows north (dy=-1) ---
    if iy < n_tile_y - 1:
        nb_fdir = flow_bdry.get('top', iy + 1, ix)
        nb_accum = boundaries.get('top', iy + 1, ix)
        for c in range(len(nb_fdir)):
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(nb_fdir[c])
            if w1 > 0.0 and dy1 == -1:
                tc = c + dx1
                if 0 <= tc < tile_w:
                    seed_bottom[tc] += nb_accum[c] * w1
            if w2 > 0.0 and dy2 == -1:
                tc = c + dx2
                if 0 <= tc < tile_w:
                    seed_bottom[tc] += nb_accum[c] * w2

    # --- Left edge: right column of tile to the left flows east (dx=1) ---
    if ix > 0:
        nb_fdir = flow_bdry.get('right', iy, ix - 1)
        nb_accum = boundaries.get('right', iy, ix - 1)
        for r in range(len(nb_fdir)):
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(nb_fdir[r])
            if w1 > 0.0 and dx1 == 1:
                tr = r + dy1
                if 0 <= tr < tile_h:
                    seed_left[tr] += nb_accum[r] * w1
            if w2 > 0.0 and dx2 == 1:
                tr = r + dy2
                if 0 <= tr < tile_h:
                    seed_left[tr] += nb_accum[r] * w2

    # --- Right edge: left column of tile to the right flows west (dx=-1) ---
    if ix < n_tile_x - 1:
        nb_fdir = flow_bdry.get('left', iy, ix + 1)
        nb_accum = boundaries.get('left', iy, ix + 1)
        for r in range(len(nb_fdir)):
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(nb_fdir[r])
            if w1 > 0.0 and dx1 == -1:
                tr = r + dy1
                if 0 <= tr < tile_h:
                    seed_right[tr] += nb_accum[r] * w1
            if w2 > 0.0 and dx2 == -1:
                tr = r + dy2
                if 0 <= tr < tile_h:
                    seed_right[tr] += nb_accum[r] * w2

    # --- Diagonal corner seeds ---
    # TL: bottom-right of (iy-1, ix-1) flows SE (dy=1, dx=1)
    if iy > 0 and ix > 0:
        fv = flow_bdry.get('bottom', iy - 1, ix - 1)[-1]
        av = float(boundaries.get('bottom', iy - 1, ix - 1)[-1])
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(fv)
        if w1 > 0.0 and dy1 == 1 and dx1 == 1:
            seed_tl += av * w1
        if w2 > 0.0 and dy2 == 1 and dx2 == 1:
            seed_tl += av * w2

    # TR: bottom-left of (iy-1, ix+1) flows SW (dy=1, dx=-1)
    if iy > 0 and ix < n_tile_x - 1:
        fv = flow_bdry.get('bottom', iy - 1, ix + 1)[0]
        av = float(boundaries.get('bottom', iy - 1, ix + 1)[0])
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(fv)
        if w1 > 0.0 and dy1 == 1 and dx1 == -1:
            seed_tr += av * w1
        if w2 > 0.0 and dy2 == 1 and dx2 == -1:
            seed_tr += av * w2

    # BL: top-right of (iy+1, ix-1) flows NE (dy=-1, dx=1)
    if iy < n_tile_y - 1 and ix > 0:
        fv = flow_bdry.get('top', iy + 1, ix - 1)[-1]
        av = float(boundaries.get('top', iy + 1, ix - 1)[-1])
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(fv)
        if w1 > 0.0 and dy1 == -1 and dx1 == 1:
            seed_bl += av * w1
        if w2 > 0.0 and dy2 == -1 and dx2 == 1:
            seed_bl += av * w2

    # BR: top-left of (iy+1, ix+1) flows NW (dy=-1, dx=-1)
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        fv = flow_bdry.get('top', iy + 1, ix + 1)[0]
        av = float(boundaries.get('top', iy + 1, ix + 1)[0])
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(fv)
        if w1 > 0.0 and dy1 == -1 and dx1 == -1:
            seed_br += av * w1
        if w2 > 0.0 and dy2 == -1 and dx2 == -1:
            seed_br += av * w2

    return (seed_top, seed_bottom, seed_left, seed_right,
            seed_tl, seed_tr, seed_bl, seed_br)


def _process_tile_dinf(iy, ix, flow_dir_da, boundaries, flow_bdry,
                       chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded Dinf BFS on one tile; update boundaries in-place."""
    chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    h, w = chunk.shape

    seeds = _compute_seeds_dinf(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    accum = _flow_accum_dinf_tile_kernel(chunk, h, w, *seeds)

    new_top = accum[0, :].copy()
    new_bottom = accum[-1, :].copy()
    new_left = accum[:, 0].copy()
    new_right = accum[:, -1].copy()

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


def _flow_accum_dinf_dask_iterative(flow_dir_da):
    """Iterative boundary-propagation for Dinf dask arrays."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry = _preprocess_tiles(flow_dir_da, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()

    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    max_iterations = max(n_tile_y, n_tile_x) + 10

    for _iteration in range(max_iterations):
        max_change = 0.0

        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_tile_dinf(iy, ix, flow_dir_da, boundaries,
                                       flow_bdry, chunks_y, chunks_x,
                                       n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile_dinf(iy, ix, flow_dir_da, boundaries,
                                       flow_bdry, chunks_y, chunks_x,
                                       n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c

        if max_change == 0.0:
            break

    boundaries = boundaries.snapshot()

    return _assemble_result_dinf(flow_dir_da, boundaries, flow_bdry,
                                 chunks_y, chunks_x, n_tile_y, n_tile_x)


def _assemble_result_dinf(flow_dir_da, boundaries, flow_bdry,
                          chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Build lazy dask array by re-running each Dinf tile with converged seeds."""

    def _tile_fn(flow_dir_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(flow_dir_block.shape, np.nan, dtype=np.float64)
        iy, ix = block_info[0]['chunk-location']
        h, w = flow_dir_block.shape
        seeds = _compute_seeds_dinf(
            iy, ix, boundaries, flow_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _flow_accum_dinf_tile_kernel(
            np.asarray(flow_dir_block, dtype=np.float64), h, w, *seeds)

    return da.map_blocks(
        _tile_fn,
        flow_dir_da,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _process_tile_dinf_cupy(iy, ix, flow_dir_da, boundaries, flow_bdry,
                             chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded GPU Dinf flow accumulation on one tile."""
    import cupy as cp

    chunk = cp.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=cp.float64)

    seeds = _compute_seeds_dinf(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    accum = _flow_accum_dinf_tile_cupy(chunk, *seeds)

    new_top = accum[0, :].get().copy()
    new_bottom = accum[-1, :].get().copy()
    new_left = accum[:, 0].get().copy()
    new_right = accum[:, -1].get().copy()

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


def _assemble_result_dinf_cupy(flow_dir_da, boundaries, flow_bdry,
                                chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Build lazy dask+cupy array using GPU Dinf tile kernel."""
    import cupy as cp

    def _tile_fn(flow_dir_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return cp.full(flow_dir_block.shape, cp.nan, dtype=cp.float64)
        iy, ix = block_info[0]['chunk-location']
        seeds = _compute_seeds_dinf(
            iy, ix, boundaries, flow_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _flow_accum_dinf_tile_cupy(
            cp.asarray(flow_dir_block, dtype=cp.float64), *seeds)

    return da.map_blocks(
        _tile_fn,
        flow_dir_da,
        dtype=np.float64,
        meta=cp.array((), dtype=cp.float64),
    )


def _flow_accum_dinf_dask_cupy(flow_dir_da):
    """Dask+CuPy Dinf: native GPU processing per tile."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry = _preprocess_tiles(flow_dir_da, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()

    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    max_iterations = max(n_tile_y, n_tile_x) + 10

    for _iteration in range(max_iterations):
        max_change = 0.0

        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_tile_dinf_cupy(
                    iy, ix, flow_dir_da, boundaries,
                    flow_bdry, chunks_y, chunks_x,
                    n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile_dinf_cupy(
                    iy, ix, flow_dir_da, boundaries,
                    flow_bdry, chunks_y, chunks_x,
                    n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c

        if max_change == 0.0:
            break

    boundaries = boundaries.snapshot()

    return _assemble_result_dinf_cupy(flow_dir_da, boundaries, flow_bdry,
                                       chunks_y, chunks_x, n_tile_y, n_tile_x)


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_accumulation_dinf(flow_dir: xr.DataArray,
                           name: str = 'flow_accumulation') -> xr.DataArray:
    """Compute flow accumulation from a D-infinity flow direction grid.

    Takes a continuous-angle flow direction grid (as produced by
    ``flow_direction_dinf``) and accumulates upstream contributing
    area.  Flow is split proportionally between two neighbors
    following Tarboton (1997).

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2-D D-infinity flow direction grid.  Values are continuous
        angles in radians ``[0, 2*pi)``, with ``-1.0`` for pits and
        NaN for nodata (as produced by ``flow_direction_dinf``).
        Supported backends: NumPy, CuPy, NumPy-backed Dask,
        CuPy-backed Dask.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='flow_accumulation'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2-D float64 array of flow accumulation values.  Each cell
        holds the total upstream contributing area (including itself)
        that drains through it, weighted by D-inf proportional
        splitting.  NaN where the input has NaN.

    References
    ----------
    Tarboton, D.G. (1997). A new method for the determination of flow
    directions and upslope areas in grid digital elevation models.
    Water Resources Research, 33(2), 309-319.
    """
    _validate_raster(flow_dir, func_name='flow_accumulation_dinf',
                     name='flow_dir')

    data = flow_dir.data

    if isinstance(data, np.ndarray):
        out = _flow_accum_dinf_cpu(data.astype(np.float64), *data.shape)
    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _flow_accum_dinf_cupy(data)
    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _flow_accum_dinf_dask_cupy(data)
    elif da is not None and isinstance(data, da.Array):
        out = _flow_accum_dinf_dask_iterative(data)
    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
