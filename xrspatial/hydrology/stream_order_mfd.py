"""Stream order extraction for MFD (Multiple Flow Direction) grids.

Assigns hierarchical order values to stream cells derived from an MFD
flow direction grid (8, H, W) and a flow accumulation grid.  Cells with
accumulation below a user-defined threshold are non-stream and receive
NaN.  Two methods are supported:

* **Strahler**: headwaters = 1; when two streams of equal order meet
  the downstream order increments by 1; otherwise the higher order
  propagates.
* **Shreve**: headwaters = 1; at each confluence the downstream
  magnitude equals the sum of all incoming magnitudes.

Algorithm
---------
CPU : Kahn's BFS topological sort among stream cells -- O(N_stream).
GPU : iterative frontier peeling with pull-based kernels.
Dask: iterative tile sweep with boundary propagation, same pattern
      as ``flow_accumulation_mfd.py``.
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

# Neighbor offsets: E, SE, S, SW, W, NW, N, NE
_DY = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

# Opposite neighbor index (who points back at me?)
# E(0)->W(4), SE(1)->NW(5), S(2)->N(6), SW(3)->NE(7), ...
_OPPOSITE = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)


def _to_numpy_f64(arr):
    """Convert *arr* to a contiguous numpy float64 array (handles CuPy)."""
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


# =====================================================================
# CPU kernels
# =====================================================================

@ngjit
def _strahler_mfd_cpu(fractions, stream_mask, height, width):
    """Kahn's BFS Strahler ordering among stream cells (MFD topology)."""
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    order = np.empty((height, width), dtype=np.float64)
    in_degree = np.zeros((height, width), dtype=np.int32)
    max_in = np.zeros((height, width), dtype=np.float64)
    cnt_max = np.zeros((height, width), dtype=np.int32)

    # Initialise
    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 0:
                order[r, c] = np.nan
            else:
                order[r, c] = 0.0

    # Compute in-degrees (only among stream cells)
    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 0:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
                    if 0 <= nr < height and 0 <= nc < width:
                        if stream_mask[nr, nc] == 1:
                            in_degree[nr, nc] += 1

    # BFS queue
    queue_r = np.empty(height * width, dtype=np.int64)
    queue_c = np.empty(height * width, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    # Enqueue headwaters (stream cells with in_degree == 0)
    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 1 and in_degree[r, c] == 0:
                order[r, c] = 1.0
                queue_r[tail] = r
                queue_c[tail] = c
                tail += 1

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1

        cur_ord = order[r, c]

        for k in range(8):
            if fractions[k, r, c] > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if not (0 <= nr < height and 0 <= nc < width
                        and stream_mask[nr, nc] == 1):
                    continue

                if cur_ord > max_in[nr, nc]:
                    max_in[nr, nc] = cur_ord
                    cnt_max[nr, nc] = 1
                elif cur_ord == max_in[nr, nc]:
                    cnt_max[nr, nc] += 1

                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    if cnt_max[nr, nc] >= 2:
                        order[nr, nc] = max_in[nr, nc] + 1.0
                    else:
                        order[nr, nc] = max_in[nr, nc]
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return order


@ngjit
def _shreve_mfd_cpu(fractions, stream_mask, height, width):
    """Kahn's BFS Shreve ordering among stream cells (MFD topology)."""
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    order = np.empty((height, width), dtype=np.float64)
    in_degree = np.zeros((height, width), dtype=np.int32)

    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 0:
                order[r, c] = np.nan
            else:
                order[r, c] = 0.0

    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 0:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
                    if 0 <= nr < height and 0 <= nc < width:
                        if stream_mask[nr, nc] == 1:
                            in_degree[nr, nc] += 1

    queue_r = np.empty(height * width, dtype=np.int64)
    queue_c = np.empty(height * width, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 1 and in_degree[r, c] == 0:
                order[r, c] = 1.0
                queue_r[tail] = r
                queue_c[tail] = c
                tail += 1

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1

        for k in range(8):
            if fractions[k, r, c] > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if not (0 <= nr < height and 0 <= nc < width
                        and stream_mask[nr, nc] == 1):
                    continue

                order[nr, nc] += order[r, c]
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return order


# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit
def _stream_order_mfd_init_gpu(fractions, stream_mask, in_degree, state,
                                order, max_in, cnt_max, H, W):
    """Initialise GPU arrays for MFD stream order computation."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if stream_mask[i, j] == 0:
        state[i, j] = 0
        order[i, j] = 0.0
        max_in[i, j] = 0.0
        cnt_max[i, j] = 0
        return

    state[i, j] = 1
    order[i, j] = 0.0
    max_in[i, j] = 0.0
    cnt_max[i, j] = 0

    # Count in-degree: iterate over 8 directions, check fraction > 0
    for k in range(8):
        frac = fractions[k, i, j]
        if frac <= 0.0:
            continue

        if k == 0:
            dy, dx = 0, 1
        elif k == 1:
            dy, dx = 1, 1
        elif k == 2:
            dy, dx = 1, 0
        elif k == 3:
            dy, dx = 1, -1
        elif k == 4:
            dy, dx = 0, -1
        elif k == 5:
            dy, dx = -1, -1
        elif k == 6:
            dy, dx = -1, 0
        else:
            dy, dx = -1, 1

        ni = i + dy
        nj = j + dx
        if 0 <= ni < H and 0 <= nj < W and stream_mask[ni, nj] == 1:
            cuda.atomic.add(in_degree, (ni, nj), 1)


@cuda.jit
def _stream_order_mfd_find_ready(in_degree, state, order, changed, H, W):
    """Finalize previous frontier (2->3), mark new frontier (1->2).

    Identical logic to D8 ``_stream_order_find_ready``: headwater cells
    get order=1 when in_degree reaches 0.
    """
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if state[i, j] == 2:
        state[i, j] = 3

    if state[i, j] == 1 and in_degree[i, j] == 0:
        state[i, j] = 2
        if order[i, j] == 0.0:
            order[i, j] = 1.0  # headwater
        cuda.atomic.add(changed, 0, 1)


@cuda.jit
def _stream_order_mfd_pull_strahler(fractions, stream_mask, in_degree, state,
                                     order, max_in, cnt_max, H, W):
    """Active cells pull Strahler info from frontier neighbours (MFD).

    For each of 8 neighbours, check if the neighbour is on the frontier
    (state==2) and flows to us (fractions[opposite[k], ni, nj] > 0).
    """
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    if state[i, j] != 1:
        return

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
        if stream_mask[ni, nj] == 0:
            continue

        # Opposite of nbr: the direction from neighbor back to me
        if nbr == 0:
            opp = 4
        elif nbr == 1:
            opp = 5
        elif nbr == 2:
            opp = 6
        elif nbr == 3:
            opp = 7
        elif nbr == 4:
            opp = 0
        elif nbr == 5:
            opp = 1
        elif nbr == 6:
            opp = 2
        else:
            opp = 3

        frac = fractions[opp, ni, nj]
        if frac > 0.0:
            nb_ord = order[ni, nj]
            if nb_ord > max_in[i, j]:
                max_in[i, j] = nb_ord
                cnt_max[i, j] = 1
            elif nb_ord == max_in[i, j]:
                cnt_max[i, j] += 1
            in_degree[i, j] -= 1

    if in_degree[i, j] == 0:
        if cnt_max[i, j] >= 2:
            order[i, j] = max_in[i, j] + 1.0
        else:
            order[i, j] = max_in[i, j]


@cuda.jit
def _stream_order_mfd_pull_shreve(fractions, stream_mask, in_degree, state,
                                   order, H, W):
    """Active cells pull Shreve magnitudes from frontier neighbours (MFD)."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    if state[i, j] != 1:
        return

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
        if stream_mask[ni, nj] == 0:
            continue

        if nbr == 0:
            opp = 4
        elif nbr == 1:
            opp = 5
        elif nbr == 2:
            opp = 6
        elif nbr == 3:
            opp = 7
        elif nbr == 4:
            opp = 0
        elif nbr == 5:
            opp = 1
        elif nbr == 6:
            opp = 2
        else:
            opp = 3

        frac = fractions[opp, ni, nj]
        if frac > 0.0:
            order[i, j] += order[ni, nj]
            in_degree[i, j] -= 1


# =====================================================================
# CuPy driver
# =====================================================================

def _stream_order_mfd_cupy(fractions_data, stream_mask_data, method):
    """GPU driver for MFD stream order computation."""
    import cupy as cp

    _, H, W = fractions_data.shape
    fractions_f64 = fractions_data.astype(cp.float64)
    stream_mask_i8 = stream_mask_data.astype(cp.int8)

    in_degree = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    order = cp.zeros((H, W), dtype=cp.float64)
    max_in = cp.zeros((H, W), dtype=cp.float64)
    cnt_max = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _stream_order_mfd_init_gpu[griddim, blockdim](
        fractions_f64, stream_mask_i8, in_degree, state,
        order, max_in, cnt_max, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _stream_order_mfd_find_ready[griddim, blockdim](
            in_degree, state, order, changed, H, W)

        if int(changed[0]) == 0:
            break

        if method == 'strahler':
            _stream_order_mfd_pull_strahler[griddim, blockdim](
                fractions_f64, stream_mask_i8, in_degree, state,
                order, max_in, cnt_max, H, W)
        else:
            _stream_order_mfd_pull_shreve[griddim, blockdim](
                fractions_f64, stream_mask_i8, in_degree, state,
                order, H, W)

    order = cp.where(stream_mask_i8 == 0, cp.nan, order)
    return order


# =====================================================================
# CPU tile kernels for dask
# =====================================================================

@ngjit
def _strahler_mfd_tile_kernel(fractions, stream_mask, h, w,
                               seed_max_top, seed_cnt_top,
                               seed_max_bottom, seed_cnt_bottom,
                               seed_max_left, seed_cnt_left,
                               seed_max_right, seed_cnt_right):
    """Seeded Strahler BFS for a single MFD tile."""
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    order = np.empty((h, w), dtype=np.float64)
    in_degree = np.zeros((h, w), dtype=np.int32)
    max_in = np.zeros((h, w), dtype=np.float64)
    cnt_max = np.zeros((h, w), dtype=np.int32)

    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] == 0:
                order[r, c] = np.nan
            else:
                order[r, c] = 0.0

    # Apply seeds: set max_in / cnt_max from boundary info
    for c in range(w):
        if stream_mask[0, c] == 1 and seed_max_top[c] > 0:
            if seed_max_top[c] > max_in[0, c]:
                max_in[0, c] = seed_max_top[c]
                cnt_max[0, c] = int(seed_cnt_top[c])
            elif seed_max_top[c] == max_in[0, c]:
                cnt_max[0, c] += int(seed_cnt_top[c])
        if stream_mask[h - 1, c] == 1 and seed_max_bottom[c] > 0:
            if seed_max_bottom[c] > max_in[h - 1, c]:
                max_in[h - 1, c] = seed_max_bottom[c]
                cnt_max[h - 1, c] = int(seed_cnt_bottom[c])
            elif seed_max_bottom[c] == max_in[h - 1, c]:
                cnt_max[h - 1, c] += int(seed_cnt_bottom[c])
    for r in range(h):
        if stream_mask[r, 0] == 1 and seed_max_left[r] > 0:
            if seed_max_left[r] > max_in[r, 0]:
                max_in[r, 0] = seed_max_left[r]
                cnt_max[r, 0] = int(seed_cnt_left[r])
            elif seed_max_left[r] == max_in[r, 0]:
                cnt_max[r, 0] += int(seed_cnt_left[r])
        if stream_mask[r, w - 1] == 1 and seed_max_right[r] > 0:
            if seed_max_right[r] > max_in[r, w - 1]:
                max_in[r, w - 1] = seed_max_right[r]
                cnt_max[r, w - 1] = int(seed_cnt_right[r])
            elif seed_max_right[r] == max_in[r, w - 1]:
                cnt_max[r, w - 1] += int(seed_cnt_right[r])

    # Compute in-degrees among stream cells within tile
    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] == 0:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
                    if 0 <= nr < h and 0 <= nc < w and stream_mask[nr, nc] == 1:
                        in_degree[nr, nc] += 1

    # BFS
    queue_r = np.empty(h * w, dtype=np.int64)
    queue_c = np.empty(h * w, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] != 1:
                continue
            if in_degree[r, c] != 0:
                continue
            # Headwater or seeded cell
            if max_in[r, c] > 0:
                if cnt_max[r, c] >= 2:
                    order[r, c] = max_in[r, c] + 1.0
                else:
                    order[r, c] = max_in[r, c]
            else:
                order[r, c] = 1.0
            queue_r[tail] = r
            queue_c[tail] = c
            tail += 1

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1

        cur_ord = order[r, c]

        for k in range(8):
            if fractions[k, r, c] > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if not (0 <= nr < h and 0 <= nc < w
                        and stream_mask[nr, nc] == 1):
                    continue

                if cur_ord > max_in[nr, nc]:
                    max_in[nr, nc] = cur_ord
                    cnt_max[nr, nc] = 1
                elif cur_ord == max_in[nr, nc]:
                    cnt_max[nr, nc] += 1

                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    if cnt_max[nr, nc] >= 2:
                        order[nr, nc] = max_in[nr, nc] + 1.0
                    else:
                        order[nr, nc] = max_in[nr, nc]
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    # Fix headwater cells: represent (order=1, no inputs) as (max=1, cnt=1)
    # so that the boundary (max, cnt) reconstruction works correctly.
    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] == 1 and max_in[r, c] == 0.0:
                max_in[r, c] = order[r, c]
                cnt_max[r, c] = 1

    return order, max_in, cnt_max


@ngjit
def _shreve_mfd_tile_kernel(fractions, stream_mask, h, w,
                             seed_top, seed_bottom,
                             seed_left, seed_right):
    """Seeded Shreve BFS for a single MFD tile."""
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    order = np.empty((h, w), dtype=np.float64)
    in_degree = np.zeros((h, w), dtype=np.int32)

    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] == 0:
                order[r, c] = np.nan
            else:
                order[r, c] = 0.0

    # Apply additive seeds
    for c in range(w):
        if stream_mask[0, c] == 1:
            order[0, c] += seed_top[c]
        if stream_mask[h - 1, c] == 1:
            order[h - 1, c] += seed_bottom[c]
    for r in range(h):
        if stream_mask[r, 0] == 1:
            order[r, 0] += seed_left[r]
        if stream_mask[r, w - 1] == 1:
            order[r, w - 1] += seed_right[r]

    # In-degrees
    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] == 0:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
                    if 0 <= nr < h and 0 <= nc < w and stream_mask[nr, nc] == 1:
                        in_degree[nr, nc] += 1

    queue_r = np.empty(h * w, dtype=np.int64)
    queue_c = np.empty(h * w, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] == 1 and in_degree[r, c] == 0:
                if order[r, c] == 0.0:
                    order[r, c] = 1.0  # headwater
                queue_r[tail] = r
                queue_c[tail] = c
                tail += 1

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1

        for k in range(8):
            if fractions[k, r, c] > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if not (0 <= nr < h and 0 <= nc < w
                        and stream_mask[nr, nc] == 1):
                    continue

                order[nr, nc] += order[r, c]
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return order


# =====================================================================
# Dask preprocessing
# =====================================================================

def _preprocess_mfd_stream_tiles(fractions_da, accum_da, threshold,
                                  chunks_y, chunks_x):
    """Extract boundary fraction strips and stream masks into dicts.

    For MFD we need the full 8-band fractions at each boundary cell,
    so we store them as (8, length) arrays.  Stream masks are stored
    as 1-D float64 arrays (1.0 = stream, 0.0 = not stream).
    """
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    frac_bdry = {}
    mask_bdry = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            y_start = sum(chunks_y[:iy])
            y_end = y_start + chunks_y[iy]
            x_start = sum(chunks_x[:ix])
            x_end = x_start + chunks_x[ix]

            frac_chunk = np.asarray(
                fractions_da[:, y_start:y_end, x_start:x_end].compute(),
                dtype=np.float64)
            ac_chunk = _to_numpy_f64(
                accum_da[y_start:y_end, x_start:x_end].compute())

            # Build stream mask for this tile
            sm = np.where(ac_chunk >= threshold, 1.0, 0.0)
            sm = np.where(np.isnan(ac_chunk), 0.0, sm)
            # NaN fractions -> not stream
            frac0 = frac_chunk[0]
            sm = np.where(frac0 != frac0, 0.0, sm)  # NaN check

            # Store fraction boundary strips: (8, length)
            frac_bdry[('top', iy, ix)] = frac_chunk[:, 0, :].copy()
            frac_bdry[('bottom', iy, ix)] = frac_chunk[:, -1, :].copy()
            frac_bdry[('left', iy, ix)] = frac_chunk[:, :, 0].copy()
            frac_bdry[('right', iy, ix)] = frac_chunk[:, :, -1].copy()

            # Store stream mask boundary strips
            for side, row_data_sm in [
                ('top', sm[0, :]),
                ('bottom', sm[-1, :]),
                ('left', sm[:, 0]),
                ('right', sm[:, -1]),
            ]:
                mask_bdry.set(side, iy, ix,
                              np.asarray(row_data_sm, dtype=np.float64))

    return frac_bdry, mask_bdry


# =====================================================================
# Dask seed computation
# =====================================================================

def _compute_shreve_seeds_mfd(iy, ix, boundaries, frac_bdry, mask_bdry,
                               chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute additive Shreve seeds from neighbours for MFD tile (iy, ix).

    For MFD, a neighbour cell flows into the current tile if its fraction
    for the direction pointing into our tile is > 0.  The seed contribution
    is ``boundary_order[cell] * fraction``.
    """
    dy_arr = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx_arr = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    tile_h = chunks_y[iy]
    tile_w = chunks_x[ix]

    seed_top = np.zeros(tile_w, dtype=np.float64)
    seed_bottom = np.zeros(tile_w, dtype=np.float64)
    seed_left = np.zeros(tile_h, dtype=np.float64)
    seed_right = np.zeros(tile_h, dtype=np.float64)

    # --- Top edge: bottom row of tile above ---
    if iy > 0:
        nb_frac = frac_bdry[('bottom', iy - 1, ix)]  # (8, tile_w)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix)
        nb_order = boundaries.get('bottom', iy - 1, ix)
        w = nb_frac.shape[1]
        for c in range(w):
            if nb_mask[c] == 0:
                continue
            for k in range(8):
                if not (nb_frac[k, c] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndy == 1:  # flows south into our tile
                    tc = c + ndx
                    if 0 <= tc < tile_w:
                        seed_top[tc] += nb_order[c] * nb_frac[k, c]

    # --- Bottom edge: top row of tile below ---
    if iy < n_tile_y - 1:
        nb_frac = frac_bdry[('top', iy + 1, ix)]  # (8, tile_w)
        nb_mask = mask_bdry.get('top', iy + 1, ix)
        nb_order = boundaries.get('top', iy + 1, ix)
        w = nb_frac.shape[1]
        for c in range(w):
            if nb_mask[c] == 0:
                continue
            for k in range(8):
                if not (nb_frac[k, c] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndy == -1:  # flows north into our tile
                    tc = c + ndx
                    if 0 <= tc < tile_w:
                        seed_bottom[tc] += nb_order[c] * nb_frac[k, c]

    # --- Left edge: right column of tile to the left ---
    if ix > 0:
        nb_frac = frac_bdry[('right', iy, ix - 1)]  # (8, tile_h)
        nb_mask = mask_bdry.get('right', iy, ix - 1)
        nb_order = boundaries.get('right', iy, ix - 1)
        h = nb_frac.shape[1]
        for r in range(h):
            if nb_mask[r] == 0:
                continue
            for k in range(8):
                if not (nb_frac[k, r] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndx == 1:  # flows east into our tile
                    tr = r + ndy
                    if 0 <= tr < tile_h:
                        seed_left[tr] += nb_order[r] * nb_frac[k, r]

    # --- Right edge: left column of tile to the right ---
    if ix < n_tile_x - 1:
        nb_frac = frac_bdry[('left', iy, ix + 1)]  # (8, tile_h)
        nb_mask = mask_bdry.get('left', iy, ix + 1)
        nb_order = boundaries.get('left', iy, ix + 1)
        h = nb_frac.shape[1]
        for r in range(h):
            if nb_mask[r] == 0:
                continue
            for k in range(8):
                if not (nb_frac[k, r] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndx == -1:  # flows west into our tile
                    tr = r + ndy
                    if 0 <= tr < tile_h:
                        seed_right[tr] += nb_order[r] * nb_frac[k, r]

    # --- Diagonal corner seeds ---
    # TL: bottom-right cell of (iy-1, ix-1) flows SE (dy=1, dx=1 -> k=1)
    if iy > 0 and ix > 0:
        nb_frac = frac_bdry[('bottom', iy - 1, ix - 1)]  # (8, w)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix - 1)
        if nb_mask[-1] == 1:
            frac_se = nb_frac[1, -1]  # SE direction
            if frac_se > 0.0:
                av = float(boundaries.get('bottom', iy - 1, ix - 1)[-1])
                seed_top[0] += av * frac_se

    # TR: bottom-left cell of (iy-1, ix+1) flows SW (dy=1, dx=-1 -> k=3)
    if iy > 0 and ix < n_tile_x - 1:
        nb_frac = frac_bdry[('bottom', iy - 1, ix + 1)]  # (8, w)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix + 1)
        if nb_mask[0] == 1:
            frac_sw = nb_frac[3, 0]  # SW direction
            if frac_sw > 0.0:
                av = float(boundaries.get('bottom', iy - 1, ix + 1)[0])
                seed_top[tile_w - 1] += av * frac_sw

    # BL: top-right cell of (iy+1, ix-1) flows NE (dy=-1, dx=1 -> k=7)
    if iy < n_tile_y - 1 and ix > 0:
        nb_frac = frac_bdry[('top', iy + 1, ix - 1)]  # (8, w)
        nb_mask = mask_bdry.get('top', iy + 1, ix - 1)
        if nb_mask[-1] == 1:
            frac_ne = nb_frac[7, -1]  # NE direction
            if frac_ne > 0.0:
                av = float(boundaries.get('top', iy + 1, ix - 1)[-1])
                seed_bottom[0] += av * frac_ne

    # BR: top-left cell of (iy+1, ix+1) flows NW (dy=-1, dx=-1 -> k=5)
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        nb_frac = frac_bdry[('top', iy + 1, ix + 1)]  # (8, w)
        nb_mask = mask_bdry.get('top', iy + 1, ix + 1)
        if nb_mask[0] == 1:
            frac_nw = nb_frac[5, 0]  # NW direction
            if frac_nw > 0.0:
                av = float(boundaries.get('top', iy + 1, ix + 1)[0])
                seed_bottom[tile_w - 1] += av * frac_nw

    return seed_top, seed_bottom, seed_left, seed_right


def _compute_strahler_seeds_mfd(iy, ix, bdry_max, bdry_cnt,
                                 frac_bdry, mask_bdry,
                                 chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute Strahler (max, cnt) seeds from neighbours for MFD tile.

    For Strahler ordering, the seed is the order value of the boundary
    cell (not multiplied by fraction).
    """
    dy_arr = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx_arr = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    tile_h = chunks_y[iy]
    tile_w = chunks_x[ix]

    smax_top = np.zeros(tile_w, dtype=np.float64)
    scnt_top = np.zeros(tile_w, dtype=np.float64)
    smax_bottom = np.zeros(tile_w, dtype=np.float64)
    scnt_bottom = np.zeros(tile_w, dtype=np.float64)
    smax_left = np.zeros(tile_h, dtype=np.float64)
    scnt_left = np.zeros(tile_h, dtype=np.float64)
    smax_right = np.zeros(tile_h, dtype=np.float64)
    scnt_right = np.zeros(tile_h, dtype=np.float64)

    def _update_max_cnt(cur_max, cur_cnt, new_val, idx):
        if new_val > cur_max[idx]:
            cur_max[idx] = new_val
            cur_cnt[idx] = 1.0
        elif new_val == cur_max[idx] and new_val > 0:
            cur_cnt[idx] += 1.0

    # --- Top edge: bottom row of tile above ---
    if iy > 0:
        nb_frac = frac_bdry[('bottom', iy - 1, ix)]  # (8, tile_w)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix)
        nb_max = bdry_max.get('bottom', iy - 1, ix)
        nb_cnt = bdry_cnt.get('bottom', iy - 1, ix)
        w = nb_frac.shape[1]
        for c in range(w):
            if nb_mask[c] == 0 or nb_max[c] == 0:
                continue
            for k in range(8):
                if not (nb_frac[k, c] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndy == 1:  # flows south into our tile
                    tc = c + ndx
                    if 0 <= tc < tile_w:
                        # Reconstruct the order of the boundary cell
                        if nb_cnt[c] >= 2:
                            val = nb_max[c] + 1.0
                        else:
                            val = nb_max[c]
                        _update_max_cnt(smax_top, scnt_top, val, tc)

    # --- Bottom edge: top row of tile below ---
    if iy < n_tile_y - 1:
        nb_frac = frac_bdry[('top', iy + 1, ix)]  # (8, tile_w)
        nb_mask = mask_bdry.get('top', iy + 1, ix)
        nb_max = bdry_max.get('top', iy + 1, ix)
        nb_cnt = bdry_cnt.get('top', iy + 1, ix)
        w = nb_frac.shape[1]
        for c in range(w):
            if nb_mask[c] == 0 or nb_max[c] == 0:
                continue
            for k in range(8):
                if not (nb_frac[k, c] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndy == -1:  # flows north into our tile
                    tc = c + ndx
                    if 0 <= tc < tile_w:
                        if nb_cnt[c] >= 2:
                            val = nb_max[c] + 1.0
                        else:
                            val = nb_max[c]
                        _update_max_cnt(smax_bottom, scnt_bottom, val, tc)

    # --- Left edge: right column of tile to the left ---
    if ix > 0:
        nb_frac = frac_bdry[('right', iy, ix - 1)]  # (8, tile_h)
        nb_mask = mask_bdry.get('right', iy, ix - 1)
        nb_max = bdry_max.get('right', iy, ix - 1)
        nb_cnt = bdry_cnt.get('right', iy, ix - 1)
        h = nb_frac.shape[1]
        for r in range(h):
            if nb_mask[r] == 0 or nb_max[r] == 0:
                continue
            for k in range(8):
                if not (nb_frac[k, r] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndx == 1:  # flows east into our tile
                    tr = r + ndy
                    if 0 <= tr < tile_h:
                        if nb_cnt[r] >= 2:
                            val = nb_max[r] + 1.0
                        else:
                            val = nb_max[r]
                        _update_max_cnt(smax_left, scnt_left, val, tr)

    # --- Right edge: left column of tile to the right ---
    if ix < n_tile_x - 1:
        nb_frac = frac_bdry[('left', iy, ix + 1)]  # (8, tile_h)
        nb_mask = mask_bdry.get('left', iy, ix + 1)
        nb_max = bdry_max.get('left', iy, ix + 1)
        nb_cnt = bdry_cnt.get('left', iy, ix + 1)
        h = nb_frac.shape[1]
        for r in range(h):
            if nb_mask[r] == 0 or nb_max[r] == 0:
                continue
            for k in range(8):
                if not (nb_frac[k, r] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndx == -1:  # flows west into our tile
                    tr = r + ndy
                    if 0 <= tr < tile_h:
                        if nb_cnt[r] >= 2:
                            val = nb_max[r] + 1.0
                        else:
                            val = nb_max[r]
                        _update_max_cnt(smax_right, scnt_right, val, tr)

    # --- Diagonal corner seeds ---
    # TL: bottom-right cell of (iy-1, ix-1) flows SE (k=1)
    if iy > 0 and ix > 0:
        nb_frac = frac_bdry[('bottom', iy - 1, ix - 1)]
        nb_mask = mask_bdry.get('bottom', iy - 1, ix - 1)
        if nb_mask[-1] == 1:
            frac_se = nb_frac[1, -1]
            if frac_se > 0.0:
                nm = bdry_max.get('bottom', iy - 1, ix - 1)[-1]
                nc = bdry_cnt.get('bottom', iy - 1, ix - 1)[-1]
                if nm > 0:
                    val = nm + 1.0 if nc >= 2 else nm
                    _update_max_cnt(smax_top, scnt_top, val, 0)

    # TR: bottom-left cell of (iy-1, ix+1) flows SW (k=3)
    if iy > 0 and ix < n_tile_x - 1:
        nb_frac = frac_bdry[('bottom', iy - 1, ix + 1)]
        nb_mask = mask_bdry.get('bottom', iy - 1, ix + 1)
        if nb_mask[0] == 1:
            frac_sw = nb_frac[3, 0]
            if frac_sw > 0.0:
                nm = bdry_max.get('bottom', iy - 1, ix + 1)[0]
                nc = bdry_cnt.get('bottom', iy - 1, ix + 1)[0]
                if nm > 0:
                    val = nm + 1.0 if nc >= 2 else nm
                    _update_max_cnt(smax_top, scnt_top, val, tile_w - 1)

    # BL: top-right cell of (iy+1, ix-1) flows NE (k=7)
    if iy < n_tile_y - 1 and ix > 0:
        nb_frac = frac_bdry[('top', iy + 1, ix - 1)]
        nb_mask = mask_bdry.get('top', iy + 1, ix - 1)
        if nb_mask[-1] == 1:
            frac_ne = nb_frac[7, -1]
            if frac_ne > 0.0:
                nm = bdry_max.get('top', iy + 1, ix - 1)[-1]
                nc = bdry_cnt.get('top', iy + 1, ix - 1)[-1]
                if nm > 0:
                    val = nm + 1.0 if nc >= 2 else nm
                    _update_max_cnt(smax_bottom, scnt_bottom, val, 0)

    # BR: top-left cell of (iy+1, ix+1) flows NW (k=5)
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        nb_frac = frac_bdry[('top', iy + 1, ix + 1)]
        nb_mask = mask_bdry.get('top', iy + 1, ix + 1)
        if nb_mask[0] == 1:
            frac_nw = nb_frac[5, 0]
            if frac_nw > 0.0:
                nm = bdry_max.get('top', iy + 1, ix + 1)[0]
                nc = bdry_cnt.get('top', iy + 1, ix + 1)[0]
                if nm > 0:
                    val = nm + 1.0 if nc >= 2 else nm
                    _update_max_cnt(smax_bottom, scnt_bottom, val,
                                    tile_w - 1)

    return (smax_top, scnt_top, smax_bottom, scnt_bottom,
            smax_left, scnt_left, smax_right, scnt_right)


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _make_stream_mask_mfd_np(ac_chunk, frac_chunk, threshold):
    """Build stream mask as numpy int8 from accumulation and MFD fractions."""
    sm = np.where(ac_chunk >= threshold, 1, 0).astype(np.int8)
    sm = np.where(np.isnan(ac_chunk), 0, sm).astype(np.int8)
    # NaN fractions -> not stream
    frac0 = frac_chunk[0]
    sm = np.where(np.isnan(frac0), 0, sm).astype(np.int8)
    return sm


def _process_strahler_tile_mfd(iy, ix, fractions_da, accum_da, threshold,
                                bdry_max, bdry_cnt, frac_bdry, mask_bdry,
                                chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded Strahler BFS on one MFD tile; update boundary stores."""
    y_start = sum(chunks_y[:iy])
    y_end = y_start + chunks_y[iy]
    x_start = sum(chunks_x[:ix])
    x_end = x_start + chunks_x[ix]

    frac_chunk = np.asarray(
        fractions_da[:, y_start:y_end, x_start:x_end].compute(),
        dtype=np.float64)
    ac_chunk = _to_numpy_f64(
        accum_da[y_start:y_end, x_start:x_end].compute())
    sm = _make_stream_mask_mfd_np(ac_chunk, frac_chunk, threshold)
    _, h, w = frac_chunk.shape

    seeds = _compute_strahler_seeds_mfd(
        iy, ix, bdry_max, bdry_cnt, frac_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    order, ki_max_in, ki_cnt_max = _strahler_mfd_tile_kernel(
        frac_chunk, sm, h, w, *seeds)

    # Extract boundary max_in/cnt_max values (not final order) so that
    # the seed reconstruction (cnt>=2 -> order+1) works at tile borders.
    change = 0.0
    bdry_slices = [
        ('top', order[0, :], ki_max_in[0, :], ki_cnt_max[0, :]),
        ('bottom', order[-1, :], ki_max_in[-1, :], ki_cnt_max[-1, :]),
        ('left', order[:, 0], ki_max_in[:, 0], ki_cnt_max[:, 0]),
        ('right', order[:, -1], ki_max_in[:, -1], ki_cnt_max[:, -1]),
    ]
    for side, order_strip, mi_strip, cm_strip in bdry_slices:
        is_nan = np.isnan(order_strip)
        new_max = np.where(is_nan, 0.0, mi_strip.astype(np.float64))
        new_cnt = np.where(is_nan, 0.0, cm_strip.astype(np.float64))

        old_max = bdry_max.get(side, iy, ix).copy()
        with np.errstate(invalid='ignore'):
            diff = np.abs(new_max - old_max)
        diff = np.where(np.isnan(diff), 0.0, diff)
        m = float(np.max(diff))
        if m > change:
            change = m

        bdry_max.set(side, iy, ix, new_max)
        bdry_cnt.set(side, iy, ix, new_cnt)

    return change


def _process_shreve_tile_mfd(iy, ix, fractions_da, accum_da, threshold,
                              boundaries, frac_bdry, mask_bdry,
                              chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded Shreve BFS on one MFD tile; update boundaries."""
    y_start = sum(chunks_y[:iy])
    y_end = y_start + chunks_y[iy]
    x_start = sum(chunks_x[:ix])
    x_end = x_start + chunks_x[ix]

    frac_chunk = np.asarray(
        fractions_da[:, y_start:y_end, x_start:x_end].compute(),
        dtype=np.float64)
    ac_chunk = _to_numpy_f64(
        accum_da[y_start:y_end, x_start:x_end].compute())
    sm = _make_stream_mask_mfd_np(ac_chunk, frac_chunk, threshold)
    _, h, w = frac_chunk.shape

    seeds = _compute_shreve_seeds_mfd(
        iy, ix, boundaries, frac_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    order = _shreve_mfd_tile_kernel(frac_chunk, sm, h, w, *seeds)

    change = 0.0
    for side, strip in [('top', order[0, :]),
                        ('bottom', order[-1, :]),
                        ('left', order[:, 0]),
                        ('right', order[:, -1])]:
        new_vals = strip.copy()
        new_vals = np.where(np.isnan(new_vals), 0.0, new_vals)
        old = boundaries.get(side, iy, ix).copy()
        with np.errstate(invalid='ignore'):
            diff = np.abs(new_vals - old)
        diff = np.where(np.isnan(diff), 0.0, diff)
        m = float(np.max(diff))
        if m > change:
            change = m
        boundaries.set(side, iy, ix, new_vals)

    return change


def _stream_order_mfd_dask_strahler(fractions_da, accum_da, threshold):
    """Dask iterative sweep for MFD Strahler ordering."""
    chunks_y = fractions_da.chunks[1]
    chunks_x = fractions_da.chunks[2]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    frac_bdry, mask_bdry = _preprocess_mfd_stream_tiles(
        fractions_da, accum_da, threshold, chunks_y, chunks_x)
    mask_bdry = mask_bdry.snapshot()

    bdry_max = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)
    bdry_cnt = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    max_iterations = max(n_tile_y, n_tile_x) + 10
    for _ in range(max_iterations):
        max_change = 0.0
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_strahler_tile_mfd(
                    iy, ix, fractions_da, accum_da, threshold,
                    bdry_max, bdry_cnt, frac_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_strahler_tile_mfd(
                    iy, ix, fractions_da, accum_da, threshold,
                    bdry_max, bdry_cnt, frac_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        if max_change == 0.0:
            break

    _bdry_max = bdry_max.snapshot()
    _bdry_cnt = bdry_cnt.snapshot()
    _frac_bdry = frac_bdry
    _mask_bdry = mask_bdry
    _threshold = threshold

    # Assemble result by re-running each tile with converged seeds
    rows = []
    for iy in range(n_tile_y):
        row = []
        for ix in range(n_tile_x):
            y_start = sum(chunks_y[:iy])
            y_end = y_start + chunks_y[iy]
            x_start = sum(chunks_x[:ix])
            x_end = x_start + chunks_x[ix]

            frac_chunk = np.asarray(
                fractions_da[:, y_start:y_end, x_start:x_end].compute(),
                dtype=np.float64)
            ac_chunk = _to_numpy_f64(
                accum_da[y_start:y_end, x_start:x_end].compute())
            sm = _make_stream_mask_mfd_np(ac_chunk, frac_chunk, _threshold)
            _, h, w = frac_chunk.shape

            seeds = _compute_strahler_seeds_mfd(
                iy, ix, _bdry_max, _bdry_cnt, _frac_bdry, _mask_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)

            tile_order, _, _ = _strahler_mfd_tile_kernel(
                frac_chunk, sm, h, w, *seeds)
            row.append(da.from_array(tile_order, chunks=tile_order.shape))
        rows.append(row)

    return da.block(rows)


def _stream_order_mfd_dask_shreve(fractions_da, accum_da, threshold):
    """Dask iterative sweep for MFD Shreve ordering."""
    chunks_y = fractions_da.chunks[1]
    chunks_x = fractions_da.chunks[2]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    frac_bdry, mask_bdry = _preprocess_mfd_stream_tiles(
        fractions_da, accum_da, threshold, chunks_y, chunks_x)
    mask_bdry = mask_bdry.snapshot()

    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    max_iterations = max(n_tile_y, n_tile_x) + 10
    for _ in range(max_iterations):
        max_change = 0.0
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_shreve_tile_mfd(
                    iy, ix, fractions_da, accum_da, threshold,
                    boundaries, frac_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_shreve_tile_mfd(
                    iy, ix, fractions_da, accum_da, threshold,
                    boundaries, frac_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        if max_change == 0.0:
            break

    _boundaries = boundaries.snapshot()
    _frac_bdry = frac_bdry
    _mask_bdry = mask_bdry
    _threshold = threshold

    # Assemble result
    rows = []
    for iy in range(n_tile_y):
        row = []
        for ix in range(n_tile_x):
            y_start = sum(chunks_y[:iy])
            y_end = y_start + chunks_y[iy]
            x_start = sum(chunks_x[:ix])
            x_end = x_start + chunks_x[ix]

            frac_chunk = np.asarray(
                fractions_da[:, y_start:y_end, x_start:x_end].compute(),
                dtype=np.float64)
            ac_chunk = _to_numpy_f64(
                accum_da[y_start:y_end, x_start:x_end].compute())
            sm = _make_stream_mask_mfd_np(ac_chunk, frac_chunk, _threshold)
            _, h, w = frac_chunk.shape

            seeds = _compute_shreve_seeds_mfd(
                iy, ix, _boundaries, _frac_bdry, _mask_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)

            tile_order = _shreve_mfd_tile_kernel(
                frac_chunk, sm, h, w, *seeds)
            row.append(da.from_array(tile_order, chunks=tile_order.shape))
        rows.append(row)

    return da.block(rows)


# =====================================================================
# Dask+CuPy
# =====================================================================

def _stream_order_mfd_dask_cupy(fractions_da, accum_da, threshold, method):
    """Dask+CuPy MFD: convert to numpy, run iterative, convert back."""
    import cupy as cp

    # Convert dask+cupy to dask+numpy for processing
    fractions_np = fractions_da.map_blocks(
        lambda b: b.get(), dtype=fractions_da.dtype,
        meta=np.array((), dtype=fractions_da.dtype),
    )
    accum_np = accum_da.map_blocks(
        lambda b: b.get(), dtype=accum_da.dtype,
        meta=np.array((), dtype=accum_da.dtype),
    )

    if method == 'strahler':
        result = _stream_order_mfd_dask_strahler(
            fractions_np, accum_np, threshold)
    else:
        result = _stream_order_mfd_dask_shreve(
            fractions_np, accum_np, threshold)

    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def stream_order_mfd(fractions: xr.DataArray,
                     flow_accum: xr.DataArray,
                     threshold: float = 100,
                     method: str = 'strahler',
                     name: str = 'stream_order_mfd') -> xr.DataArray:
    """Compute stream order from MFD flow direction and accumulation grids.

    Parameters
    ----------
    fractions : xarray.DataArray or xr.Dataset
        3-D MFD flow direction array of shape ``(8, H, W)`` as returned
        by ``flow_direction_mfd``.  Values are flow fractions in
        ``[0, 1]`` that sum to 1.0 at each cell (0.0 at pits/flats,
        NaN at edges or nodata cells).
        Supported backends: NumPy, CuPy, NumPy-backed Dask,
        CuPy-backed Dask.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    flow_accum : xarray.DataArray
        2-D flow accumulation grid.  Cells with
        ``flow_accum >= threshold`` are considered stream cells.
    threshold : float, default 100
        Minimum accumulation to classify a cell as part of the
        stream network.
    method : str, default 'strahler'
        ``'strahler'`` for Strahler branching hierarchy or
        ``'shreve'`` for Shreve cumulative magnitude.
    name : str, default 'stream_order_mfd'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2-D float64 array of stream order values.  Non-stream cells
        (accumulation below threshold) are NaN.

    References
    ----------
    Strahler, A.N. (1957). Quantitative analysis of watershed
    geomorphology. Transactions of the American Geophysical Union,
    38(6), 913-920.

    Shreve, R.L. (1966). Statistical law of stream numbers. Journal
    of Geology, 74(1), 17-37.
    """
    _validate_raster(fractions, func_name='stream_order_mfd',
                     name='fractions', ndim=3)

    method = method.lower()
    if method not in ('strahler', 'shreve'):
        raise ValueError(
            f"method must be 'strahler' or 'shreve', got {method!r}")

    frac_data = fractions.data
    fa_data = flow_accum.data

    if frac_data.ndim != 3 or frac_data.shape[0] != 8:
        raise ValueError(
            "fractions must be a 3-D array of shape (8, H, W), "
            f"got shape {frac_data.shape}"
        )

    if isinstance(frac_data, np.ndarray):
        frac = frac_data.astype(np.float64)
        fa = np.asarray(fa_data, dtype=np.float64)
        stream_mask = np.where(fa >= threshold, 1, 0).astype(np.int8)
        stream_mask = np.where(np.isnan(fa), 0, stream_mask).astype(np.int8)
        # NaN fractions -> not stream
        stream_mask = np.where(
            np.isnan(frac[0]), 0, stream_mask).astype(np.int8)
        h, w = frac.shape[1], frac.shape[2]
        if method == 'strahler':
            out = _strahler_mfd_cpu(frac, stream_mask, h, w)
        else:
            out = _shreve_mfd_cpu(frac, stream_mask, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(frac_data):
        import cupy as cp
        fa_cp = cp.asarray(fa_data, dtype=cp.float64)
        frac_cp = frac_data.astype(cp.float64)
        stream_mask = cp.where(fa_cp >= threshold, 1, 0).astype(cp.int8)
        stream_mask = cp.where(
            cp.isnan(fa_cp), 0, stream_mask).astype(cp.int8)
        stream_mask = cp.where(
            cp.isnan(frac_cp[0]), 0, stream_mask).astype(cp.int8)
        out = _stream_order_mfd_cupy(frac_cp, stream_mask, method)

    elif has_cuda_and_cupy() and is_dask_cupy(fractions):
        out = _stream_order_mfd_dask_cupy(
            frac_data, fa_data, threshold, method)

    elif da is not None and isinstance(frac_data, da.Array):
        if method == 'strahler':
            out = _stream_order_mfd_dask_strahler(
                frac_data, fa_data, threshold)
        else:
            out = _stream_order_mfd_dask_shreve(
                frac_data, fa_data, threshold)

    else:
        raise TypeError(f"Unsupported array type: {type(frac_data)}")

    # Build 2-D output coords (drop 'neighbor' dim)
    spatial_dims = fractions.dims[1:]
    coords = {k: v for k, v in fractions.coords.items()
              if k != 'neighbor' and k not in fractions.dims[:1]}
    for d in spatial_dims:
        if d in fractions.coords:
            coords[d] = fractions.coords[d]

    return xr.DataArray(out,
                        name=name,
                        coords=coords,
                        dims=spatial_dims,
                        attrs=fractions.attrs)
