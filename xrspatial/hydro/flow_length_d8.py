"""D8 flow length: distance from each cell to outlet (downstream) or
longest path from divide to cell (upstream).

Algorithm
---------
CPU : Kahn's BFS topological sort — O(N).
      Downstream: reverse pass from outlets, accumulating step distance.
      Upstream: forward pass from divides, propagating max distance.
GPU : CuPy-via-CPU (same as flow_path.py).
Dask: iterative tile sweep with BoundaryStore boundary propagation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.hydro.flow_accumulation_d8 import _code_to_offset
from xrspatial.hydro.watershed_d8 import _code_to_offset_py
from xrspatial.hydro._boundary_store import BoundaryStore
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
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for ``_flow_length_*_cpu``:
#   flow_len   : float64 -> 8
#   in_degree  : int32   -> 4
#   valid      : int8    -> 1
#   order_r    : int64   -> 8
#   order_c    : int64   -> 8
# Total ~29 bytes/pixel.  The caller-provided ``flow_dir`` array already
# lives in RAM before the kernel runs and is not double-counted here.
_BYTES_PER_PIXEL = 29

# GPU peak working set per pixel for ``_flow_length_cupy``: that path
# copies fd to host via ``.get()`` then runs the CPU kernel and copies
# the output back via ``cp.asarray``.  Device-side residency at peak is
# the input float64 (8 B/px) plus the output float64 (8 B/px); host-side
# matches the 29 B/px CPU budget.  Use 32 B/px as a conservative GPU
# budget covering both copies plus headroom.
_GPU_BYTES_PER_PIXEL = 32


def _available_memory_bytes():
    """Best-effort estimate of available host memory in bytes."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    return 2 * 1024 ** 3


def _available_gpu_memory_bytes():
    """Best-effort estimate of free GPU memory in bytes.

    Returns 0 if CuPy / CUDA is unavailable or the query fails -- callers
    use that as a sentinel meaning "no GPU info, skip the guard".
    """
    try:
        import cupy as _cp
        free, _total = _cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return 0


def _check_memory(height, width):
    """Raise MemoryError if the flow_length kernel would exceed 50% of RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"flow_length_d8 on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of working memory but only "
            f"~{available / 1e9:.1f} GB is available.  Use a "
            f"dask-backed DataArray for out-of-core processing."
        )


def _check_gpu_memory(height, width):
    """Raise MemoryError if the CuPy kernel would exceed 50% of free GPU RAM.

    Skips the check (returns silently) when ``_available_gpu_memory_bytes``
    cannot determine the free memory -- e.g. on hosts without CUDA, where
    the kernel will fail at the cupy.asarray boundary anyway.
    """
    available = _available_gpu_memory_bytes()
    if available <= 0:
        return
    required = int(height) * int(width) * _GPU_BYTES_PER_PIXEL
    if required > 0.5 * available:
        raise MemoryError(
            f"flow_length_d8 on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


# =====================================================================
# Direction → step distance helper
# =====================================================================

@ngjit
def _step_distance(code, cellsize_x, cellsize_y, diag):
    """Return the travel distance for a D8 direction code."""
    c = int(code)
    if c == 1 or c == 16:     # E / W
        return cellsize_x
    elif c == 4 or c == 64:   # S / N
        return cellsize_y
    elif c == 2 or c == 8 or c == 32 or c == 128:  # diagonals
        return diag
    return 0.0


# =====================================================================
# CPU kernels
# =====================================================================

@ngjit
def _flow_length_downstream_cpu(flow_dir, H, W, cellsize_x, cellsize_y, diag):
    """Downstream flow length: distance from cell to outlet/edge-exit.

    Two-pass O(N):
    1. Kahn's BFS builds topological order (divides first).
    2. Reverse pass (outlets first): flow_len[cell] =
       flow_len[downstream] + step_dist.
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
            dy, dx = _code_to_offset(flow_dir[r, c])
            if dy == 0 and dx == 0:
                continue
            nr, nc = r + dy, c + dx
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
        dy, dx = _code_to_offset(flow_dir[r, c])
        if dy == 0 and dx == 0:
            continue
        nr, nc = r + dy, c + dx
        if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
            in_degree[nr, nc] -= 1
            if in_degree[nr, nc] == 0:
                order_r[tail] = nr
                order_c[tail] = nc
                tail += 1

    # Reverse pass: outlets → divides
    for i in range(tail - 1, -1, -1):
        r = order_r[i]
        c = order_c[i]
        dy, dx = _code_to_offset(flow_dir[r, c])
        if dy == 0 and dx == 0:
            flow_len[r, c] = 0.0
            continue
        nr, nc = r + dy, c + dx
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            # Edge exit
            flow_len[r, c] = 0.0
            continue
        if valid[nr, nc] == 0:
            flow_len[r, c] = 0.0
            continue
        sd = _step_distance(flow_dir[r, c], cellsize_x, cellsize_y, diag)
        flow_len[r, c] = flow_len[nr, nc] + sd

    return flow_len


@ngjit
def _flow_length_upstream_cpu(flow_dir, H, W, cellsize_x, cellsize_y, diag):
    """Upstream flow length: longest path from any divide to cell.

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
            dy, dx = _code_to_offset(flow_dir[r, c])
            if dy == 0 and dx == 0:
                continue
            nr, nc = r + dy, c + dx
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

        dy, dx = _code_to_offset(flow_dir[r, c])
        if dy == 0 and dx == 0:
            continue
        nr, nc = r + dy, c + dx
        if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
            sd = _step_distance(flow_dir[r, c], cellsize_x, cellsize_y, diag)
            candidate = flow_len[r, c] + sd
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

def _flow_length_cupy(flow_dir_data, direction, cellsize_x, cellsize_y, diag):
    import cupy as cp
    fd_np = flow_dir_data.get().astype(np.float64)
    H, W = fd_np.shape
    if direction == 'downstream':
        out = _flow_length_downstream_cpu(fd_np, H, W, cellsize_x, cellsize_y, diag)
    else:
        out = _flow_length_upstream_cpu(fd_np, H, W, cellsize_x, cellsize_y, diag)
    return cp.asarray(out)


# =====================================================================
# Dask tile kernels
# =====================================================================

@ngjit
def _flow_length_downstream_tile(flow_dir, h, w, cellsize_x, cellsize_y, diag,
                                  exit_top, exit_bottom, exit_left, exit_right,
                                  exit_tl, exit_tr, exit_bl, exit_br):
    """Downstream flow length for a single tile with exit-label seeds.

    Boundary cells that flow out of the tile use exit values as the
    known downstream flow_length at their destination.
    """
    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)
    flow_len = np.empty((h, w), dtype=np.float64)
    known = np.zeros((h, w), dtype=np.int8)

    # Init
    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v == v:
                valid[r, c] = 1
                flow_len[r, c] = 0.0
            else:
                flow_len[r, c] = np.nan

    # Apply exit labels: cells that flow OUT of tile get known downstream length
    # Top row
    for c in range(w):
        if valid[0, c] == 1:
            dy, dx = _code_to_offset(flow_dir[0, c])
            nr = 0 + dy
            if nr < 0:  # flows out top
                el = exit_top[c]
                if el == el:  # not NaN
                    sd = _step_distance(flow_dir[0, c], cellsize_x, cellsize_y, diag)
                    flow_len[0, c] = el + sd
                    known[0, c] = 1
                else:
                    flow_len[0, c] = 0.0
                    known[0, c] = 1
    # Bottom row
    for c in range(w):
        if valid[h - 1, c] == 1:
            dy, dx = _code_to_offset(flow_dir[h - 1, c])
            nr = h - 1 + dy
            if nr >= h:
                el = exit_bottom[c]
                if el == el:
                    sd = _step_distance(flow_dir[h - 1, c], cellsize_x, cellsize_y, diag)
                    flow_len[h - 1, c] = el + sd
                    known[h - 1, c] = 1
                else:
                    flow_len[h - 1, c] = 0.0
                    known[h - 1, c] = 1
    # Left column
    for r in range(h):
        if valid[r, 0] == 1:
            dy, dx = _code_to_offset(flow_dir[r, 0])
            nc = 0 + dx
            if nc < 0:
                el = exit_left[r]
                if el == el:
                    sd = _step_distance(flow_dir[r, 0], cellsize_x, cellsize_y, diag)
                    flow_len[r, 0] = el + sd
                    known[r, 0] = 1
                else:
                    flow_len[r, 0] = 0.0
                    known[r, 0] = 1
    # Right column
    for r in range(h):
        if valid[r, w - 1] == 1:
            dy, dx = _code_to_offset(flow_dir[r, w - 1])
            nc = w - 1 + dx
            if nc >= w:
                el = exit_right[r]
                if el == el:
                    sd = _step_distance(flow_dir[r, w - 1], cellsize_x, cellsize_y, diag)
                    flow_len[r, w - 1] = el + sd
                    known[r, w - 1] = 1
                else:
                    flow_len[r, w - 1] = 0.0
                    known[r, w - 1] = 1

    # Corner overrides
    if valid[0, 0] == 1:
        dy, dx = _code_to_offset(flow_dir[0, 0])
        if 0 + dy < 0 and 0 + dx < 0:
            if exit_tl == exit_tl:
                sd = _step_distance(flow_dir[0, 0], cellsize_x, cellsize_y, diag)
                flow_len[0, 0] = exit_tl + sd
                known[0, 0] = 1
    if valid[0, w - 1] == 1:
        dy, dx = _code_to_offset(flow_dir[0, w - 1])
        if 0 + dy < 0 and w - 1 + dx >= w:
            if exit_tr == exit_tr:
                sd = _step_distance(flow_dir[0, w - 1], cellsize_x, cellsize_y, diag)
                flow_len[0, w - 1] = exit_tr + sd
                known[0, w - 1] = 1
    if valid[h - 1, 0] == 1:
        dy, dx = _code_to_offset(flow_dir[h - 1, 0])
        if h - 1 + dy >= h and 0 + dx < 0:
            if exit_bl == exit_bl:
                sd = _step_distance(flow_dir[h - 1, 0], cellsize_x, cellsize_y, diag)
                flow_len[h - 1, 0] = exit_bl + sd
                known[h - 1, 0] = 1
    if valid[h - 1, w - 1] == 1:
        dy, dx = _code_to_offset(flow_dir[h - 1, w - 1])
        if h - 1 + dy >= h and w - 1 + dx >= w:
            if exit_br == exit_br:
                sd = _step_distance(flow_dir[h - 1, w - 1], cellsize_x, cellsize_y, diag)
                flow_len[h - 1, w - 1] = exit_br + sd
                known[h - 1, w - 1] = 1

    # In-degrees (only for edges within tile)
    for r in range(h):
        for c in range(w):
            if valid[r, c] == 0 or known[r, c] == 1:
                continue
            dy, dx = _code_to_offset(flow_dir[r, c])
            if dy == 0 and dx == 0:
                continue
            nr, nc = r + dy, c + dx
            if 0 <= nr < h and 0 <= nc < w:
                if valid[nr, nc] == 1 and known[nr, nc] == 0:
                    in_degree[nr, nc] += 1

    # BFS topological order
    order_r = np.empty(h * w, dtype=np.int64)
    order_c = np.empty(h * w, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for r in range(h):
        for c in range(w):
            if valid[r, c] == 1 and known[r, c] == 0 and in_degree[r, c] == 0:
                order_r[tail] = r
                order_c[tail] = c
                tail += 1

    while head < tail:
        r = order_r[head]
        c = order_c[head]
        head += 1
        dy, dx = _code_to_offset(flow_dir[r, c])
        if dy == 0 and dx == 0:
            continue
        nr, nc = r + dy, c + dx
        if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1 and known[nr, nc] == 0:
            in_degree[nr, nc] -= 1
            if in_degree[nr, nc] == 0:
                order_r[tail] = nr
                order_c[tail] = nc
                tail += 1

    # Reverse pass
    for i in range(tail - 1, -1, -1):
        r = order_r[i]
        c = order_c[i]
        dy, dx = _code_to_offset(flow_dir[r, c])
        if dy == 0 and dx == 0:
            flow_len[r, c] = 0.0
            continue
        nr, nc = r + dy, c + dx
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            # Exits tile but no exit label (grid edge with no neighbor tile)
            flow_len[r, c] = 0.0
            continue
        if valid[nr, nc] == 0:
            flow_len[r, c] = 0.0
            continue
        sd = _step_distance(flow_dir[r, c], cellsize_x, cellsize_y, diag)
        flow_len[r, c] = flow_len[nr, nc] + sd

    return flow_len


@ngjit
def _flow_length_upstream_tile(flow_dir, h, w, cellsize_x, cellsize_y, diag,
                                seed_top, seed_bottom, seed_left, seed_right,
                                seed_tl, seed_tr, seed_bl, seed_br):
    """Upstream flow length for a single tile with entry seeds.

    Boundary cells that receive flow from outside the tile get seeded
    with the upstream length from the adjacent tile.
    """
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
            dy, dx = _code_to_offset(flow_dir[r, c])
            if dy == 0 and dx == 0:
                continue
            nr, nc = r + dy, c + dx
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
        dy, dx = _code_to_offset(flow_dir[r, c])
        if dy == 0 and dx == 0:
            continue
        nr, nc = r + dy, c + dx
        if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
            sd = _step_distance(flow_dir[r, c], cellsize_x, cellsize_y, diag)
            candidate = flow_len[r, c] + sd
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

def _preprocess_tiles(flow_dir_da, chunks_y, chunks_x):
    """Extract boundary flow-direction strips into a BoundaryStore."""
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    flow_bdry = BoundaryStore(chunks_y, chunks_x, fill_value=np.nan)

    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            chunk = flow_dir_da.blocks[iy, ix].compute()
            flow_bdry.set('top', iy, ix,
                          np.asarray(chunk[0, :], dtype=np.float64))
            flow_bdry.set('bottom', iy, ix,
                          np.asarray(chunk[-1, :], dtype=np.float64))
            flow_bdry.set('left', iy, ix,
                          np.asarray(chunk[:, 0], dtype=np.float64))
            flow_bdry.set('right', iy, ix,
                          np.asarray(chunk[:, -1], dtype=np.float64))
    return flow_bdry


def _compute_exit_labels_downstream(iy, ix, boundaries, flow_bdry,
                                     chunks_y, chunks_x, n_tile_y, n_tile_x):
    """For downstream: look up the flow_length of the cell each
    boundary cell flows TO in the adjacent tile."""
    tile_h = chunks_y[iy]
    tile_w = chunks_x[ix]

    exit_top = np.full(tile_w, np.nan)
    exit_bottom = np.full(tile_w, np.nan)
    exit_left = np.full(tile_h, np.nan)
    exit_right = np.full(tile_h, np.nan)
    exit_tl = np.nan
    exit_tr = np.nan
    exit_bl = np.nan
    exit_br = np.nan

    # Top row: cells flowing north out of tile
    if iy > 0:
        fdir_top = flow_bdry.get('top', iy, ix)
        nb_labels = boundaries.get('bottom', iy - 1, ix)
        for j in range(tile_w):
            d = _code_to_offset_py(fdir_top[j])
            if d[0] == -1:
                dj = j + d[1]
                if d[1] == 0:
                    if 0 <= dj < len(nb_labels):
                        exit_top[j] = nb_labels[dj]
                elif d[1] == -1:
                    if 0 <= dj < len(nb_labels):
                        exit_top[j] = nb_labels[dj]
                    elif dj < 0 and ix > 0:
                        exit_top[j] = boundaries.get('bottom', iy - 1, ix - 1)[-1]
                elif d[1] == 1:
                    if 0 <= dj < len(nb_labels):
                        exit_top[j] = nb_labels[dj]
                    elif dj >= len(nb_labels) and ix < n_tile_x - 1:
                        exit_top[j] = boundaries.get('bottom', iy - 1, ix + 1)[0]

    # Bottom row: cells flowing south out of tile
    if iy < n_tile_y - 1:
        fdir_bot = flow_bdry.get('bottom', iy, ix)
        nb_labels = boundaries.get('top', iy + 1, ix)
        for j in range(tile_w):
            d = _code_to_offset_py(fdir_bot[j])
            if d[0] == 1:
                dj = j + d[1]
                if d[1] == 0:
                    if 0 <= dj < len(nb_labels):
                        exit_bottom[j] = nb_labels[dj]
                elif d[1] == 1:
                    if 0 <= dj < len(nb_labels):
                        exit_bottom[j] = nb_labels[dj]
                    elif dj >= len(nb_labels) and ix < n_tile_x - 1:
                        exit_bottom[j] = boundaries.get('top', iy + 1, ix + 1)[0]
                elif d[1] == -1:
                    if 0 <= dj < len(nb_labels):
                        exit_bottom[j] = nb_labels[dj]
                    elif dj < 0 and ix > 0:
                        exit_bottom[j] = boundaries.get('top', iy + 1, ix - 1)[-1]

    # Left column: cells flowing west out of tile
    if ix > 0:
        fdir_left = flow_bdry.get('left', iy, ix)
        nb_labels = boundaries.get('right', iy, ix - 1)
        for r in range(tile_h):
            d = _code_to_offset_py(fdir_left[r])
            if d[1] == -1:
                dr = r + d[0]
                if d[0] == 0:
                    if 0 <= dr < len(nb_labels):
                        exit_left[r] = nb_labels[dr]
                elif d[0] == -1:
                    if r == 0:
                        continue
                    if 0 <= dr < len(nb_labels):
                        exit_left[r] = nb_labels[dr]
                elif d[0] == 1:
                    if r == tile_h - 1:
                        continue
                    if 0 <= dr < len(nb_labels):
                        exit_left[r] = nb_labels[dr]

    # Right column: cells flowing east out of tile
    if ix < n_tile_x - 1:
        fdir_right = flow_bdry.get('right', iy, ix)
        nb_labels = boundaries.get('left', iy, ix + 1)
        for r in range(tile_h):
            d = _code_to_offset_py(fdir_right[r])
            if d[1] == 1:
                dr = r + d[0]
                if d[0] == 0:
                    if 0 <= dr < len(nb_labels):
                        exit_right[r] = nb_labels[dr]
                elif d[0] == -1:
                    if r == 0:
                        continue
                    if 0 <= dr < len(nb_labels):
                        exit_right[r] = nb_labels[dr]
                elif d[0] == 1:
                    if r == tile_h - 1:
                        continue
                    if 0 <= dr < len(nb_labels):
                        exit_right[r] = nb_labels[dr]

    # Edge-of-grid exits (flow off grid → distance 0, exit = NaN means "no neighbor")
    if iy == 0:
        fdir_top = flow_bdry.get('top', iy, ix)
        for j in range(tile_w):
            d = _code_to_offset_py(fdir_top[j])
            if d[0] == -1:
                exit_top[j] = np.nan
    if iy == n_tile_y - 1:
        fdir_bot = flow_bdry.get('bottom', iy, ix)
        for j in range(tile_w):
            d = _code_to_offset_py(fdir_bot[j])
            if d[0] == 1:
                exit_bottom[j] = np.nan
    if ix == 0:
        fdir_left = flow_bdry.get('left', iy, ix)
        for r in range(tile_h):
            d = _code_to_offset_py(fdir_left[r])
            if d[1] == -1:
                exit_left[r] = np.nan
    if ix == n_tile_x - 1:
        fdir_right = flow_bdry.get('right', iy, ix)
        for r in range(tile_h):
            d = _code_to_offset_py(fdir_right[r])
            if d[1] == 1:
                exit_right[r] = np.nan

    # Diagonal corners
    fdir_tl = flow_bdry.get('top', iy, ix)[0]
    d = _code_to_offset_py(fdir_tl)
    if d == (-1, -1):
        if iy > 0 and ix > 0:
            exit_tl = boundaries.get('bottom', iy - 1, ix - 1)[-1]
        else:
            exit_tl = np.nan

    fdir_tr = flow_bdry.get('top', iy, ix)[-1]
    d = _code_to_offset_py(fdir_tr)
    if d == (-1, 1):
        if iy > 0 and ix < n_tile_x - 1:
            exit_tr = boundaries.get('bottom', iy - 1, ix + 1)[0]
        else:
            exit_tr = np.nan

    fdir_bl = flow_bdry.get('bottom', iy, ix)[0]
    d = _code_to_offset_py(fdir_bl)
    if d == (1, -1):
        if iy < n_tile_y - 1 and ix > 0:
            exit_bl = boundaries.get('top', iy + 1, ix - 1)[-1]
        else:
            exit_bl = np.nan

    fdir_br = flow_bdry.get('bottom', iy, ix)[-1]
    d = _code_to_offset_py(fdir_br)
    if d == (1, 1):
        if iy < n_tile_y - 1 and ix < n_tile_x - 1:
            exit_br = boundaries.get('top', iy + 1, ix + 1)[0]
        else:
            exit_br = np.nan

    return (exit_top, exit_bottom, exit_left, exit_right,
            exit_tl, exit_tr, exit_bl, exit_br)


def _compute_seeds_upstream(iy, ix, boundaries, flow_bdry,
                             chunks_y, chunks_x, n_tile_y, n_tile_x,
                             cellsize_x, cellsize_y, diag):
    """For upstream: check which adjacent cells flow INTO this tile,
    and seed with max(existing, neighbor_upstream + step_dist)."""
    from xrspatial.hydro.flow_accumulation_d8 import _compute_seeds as _compute_accum_seeds

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

    # We need the same neighbor-flows-into-me pattern as flow_accumulation,
    # but instead of sum, we take max, and add step distance.

    # Top edge: bottom of tile above
    if iy > 0:
        nb_fdir = flow_bdry.get('bottom', iy - 1, ix)
        nb_val = boundaries.get('bottom', iy - 1, ix)
        w = len(nb_fdir)
        # Cardinal S (code 4)
        for j in range(w):
            if nb_fdir[j] == 4 and nb_val[j] == nb_val[j]:
                v = nb_val[j] + cellsize_y
                if v > seed_top[j]:
                    seed_top[j] = v
        if w > 1:
            # SE (code 2)
            for j in range(w - 1):
                if nb_fdir[j] == 2 and nb_val[j] == nb_val[j]:
                    v = nb_val[j] + diag
                    if v > seed_top[j + 1]:
                        seed_top[j + 1] = v
            # SW (code 8)
            for j in range(1, w):
                if nb_fdir[j] == 8 and nb_val[j] == nb_val[j]:
                    v = nb_val[j] + diag
                    if v > seed_top[j - 1]:
                        seed_top[j - 1] = v

    # Bottom edge: top of tile below
    if iy < n_tile_y - 1:
        nb_fdir = flow_bdry.get('top', iy + 1, ix)
        nb_val = boundaries.get('top', iy + 1, ix)
        w = len(nb_fdir)
        # Cardinal N (code 64)
        for j in range(w):
            if nb_fdir[j] == 64 and nb_val[j] == nb_val[j]:
                v = nb_val[j] + cellsize_y
                if v > seed_bottom[j]:
                    seed_bottom[j] = v
        if w > 1:
            # NE (code 128)
            for j in range(w - 1):
                if nb_fdir[j] == 128 and nb_val[j] == nb_val[j]:
                    v = nb_val[j] + diag
                    if v > seed_bottom[j + 1]:
                        seed_bottom[j + 1] = v
            # NW (code 32)
            for j in range(1, w):
                if nb_fdir[j] == 32 and nb_val[j] == nb_val[j]:
                    v = nb_val[j] + diag
                    if v > seed_bottom[j - 1]:
                        seed_bottom[j - 1] = v

    # Left edge: right of tile to the left
    if ix > 0:
        nb_fdir = flow_bdry.get('right', iy, ix - 1)
        nb_val = boundaries.get('right', iy, ix - 1)
        h = len(nb_fdir)
        # Cardinal E (code 1)
        for r in range(h):
            if nb_fdir[r] == 1 and nb_val[r] == nb_val[r]:
                v = nb_val[r] + cellsize_x
                if v > seed_left[r]:
                    seed_left[r] = v
        if h > 1:
            # SE (code 2)
            for r in range(h - 1):
                if nb_fdir[r] == 2 and nb_val[r] == nb_val[r]:
                    v = nb_val[r] + diag
                    if v > seed_left[r + 1]:
                        seed_left[r + 1] = v
            # NE (code 128)
            for r in range(1, h):
                if nb_fdir[r] == 128 and nb_val[r] == nb_val[r]:
                    v = nb_val[r] + diag
                    if v > seed_left[r - 1]:
                        seed_left[r - 1] = v

    # Right edge: left of tile to the right
    if ix < n_tile_x - 1:
        nb_fdir = flow_bdry.get('left', iy, ix + 1)
        nb_val = boundaries.get('left', iy, ix + 1)
        h = len(nb_fdir)
        # Cardinal W (code 16)
        for r in range(h):
            if nb_fdir[r] == 16 and nb_val[r] == nb_val[r]:
                v = nb_val[r] + cellsize_x
                if v > seed_right[r]:
                    seed_right[r] = v
        if h > 1:
            # SW (code 8)
            for r in range(h - 1):
                if nb_fdir[r] == 8 and nb_val[r] == nb_val[r]:
                    v = nb_val[r] + diag
                    if v > seed_right[r + 1]:
                        seed_right[r + 1] = v
            # NW (code 32)
            for r in range(1, h):
                if nb_fdir[r] == 32 and nb_val[r] == nb_val[r]:
                    v = nb_val[r] + diag
                    if v > seed_right[r - 1]:
                        seed_right[r - 1] = v

    # Diagonal corner seeds
    if iy > 0 and ix > 0:
        fdir = flow_bdry.get('bottom', iy - 1, ix - 1)[-1]
        if fdir == 2:
            val = boundaries.get('bottom', iy - 1, ix - 1)[-1]
            if val == val:
                seed_tl = val + diag
    if iy > 0 and ix < n_tile_x - 1:
        fdir = flow_bdry.get('bottom', iy - 1, ix + 1)[0]
        if fdir == 8:
            val = boundaries.get('bottom', iy - 1, ix + 1)[0]
            if val == val:
                seed_tr = val + diag
    if iy < n_tile_y - 1 and ix > 0:
        fdir = flow_bdry.get('top', iy + 1, ix - 1)[-1]
        if fdir == 128:
            val = boundaries.get('top', iy + 1, ix - 1)[-1]
            if val == val:
                seed_bl = val + diag
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        fdir = flow_bdry.get('top', iy + 1, ix + 1)[0]
        if fdir == 32:
            val = boundaries.get('top', iy + 1, ix + 1)[0]
            if val == val:
                seed_br = val + diag

    return (seed_top, seed_bottom, seed_left, seed_right,
            seed_tl, seed_tr, seed_bl, seed_br)


def _process_tile_downstream(iy, ix, flow_dir_da, boundaries, flow_bdry,
                              chunks_y, chunks_x, n_tile_y, n_tile_x,
                              cellsize_x, cellsize_y, diag):
    """Process one tile for downstream flow length; update boundaries."""
    chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    h, w = chunk.shape

    exits = _compute_exit_labels_downstream(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    result = _flow_length_downstream_tile(
        chunk, h, w, cellsize_x, cellsize_y, diag, *exits)

    new_top = result[0, :].copy()
    new_bottom = result[-1, :].copy()
    new_left = result[:, 0].copy()
    new_right = result[:, -1].copy()

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


def _process_tile_upstream(iy, ix, flow_dir_da, boundaries, flow_bdry,
                            chunks_y, chunks_x, n_tile_y, n_tile_x,
                            cellsize_x, cellsize_y, diag):
    """Process one tile for upstream flow length; update boundaries."""
    chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    h, w = chunk.shape

    seeds = _compute_seeds_upstream(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x,
        cellsize_x, cellsize_y, diag)

    result = _flow_length_upstream_tile(
        chunk, h, w, cellsize_x, cellsize_y, diag, *seeds)

    new_top = result[0, :].copy()
    new_bottom = result[-1, :].copy()
    new_left = result[:, 0].copy()
    new_right = result[:, -1].copy()

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


def _flow_length_dask_iterative(flow_dir_da, direction,
                                 cellsize_x, cellsize_y, diag):
    """Iterative boundary-propagation for flow length on dask arrays."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry = _preprocess_tiles(flow_dir_da, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()

    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    process_fn = (_process_tile_downstream if direction == 'downstream'
                  else _process_tile_upstream)

    max_iterations = max(n_tile_y, n_tile_x) * 2 + 10

    for _iteration in range(max_iterations):
        max_change = 0.0

        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = process_fn(iy, ix, flow_dir_da, boundaries, flow_bdry,
                               chunks_y, chunks_x, n_tile_y, n_tile_x,
                               cellsize_x, cellsize_y, diag)
                if c > max_change:
                    max_change = c

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = process_fn(iy, ix, flow_dir_da, boundaries, flow_bdry,
                               chunks_y, chunks_x, n_tile_y, n_tile_x,
                               cellsize_x, cellsize_y, diag)
                if c > max_change:
                    max_change = c

        if max_change == 0.0:
            break

    boundaries = boundaries.snapshot()

    return _assemble_result(flow_dir_da, boundaries, flow_bdry,
                            chunks_y, chunks_x, n_tile_y, n_tile_x,
                            direction, cellsize_x, cellsize_y, diag)


def _assemble_result(flow_dir_da, boundaries, flow_bdry,
                     chunks_y, chunks_x, n_tile_y, n_tile_x,
                     direction, cellsize_x, cellsize_y, diag):
    """Build lazy dask array by re-running tiles with converged boundaries."""

    if direction == 'downstream':
        def _tile_fn(flow_dir_block, block_info=None):
            if block_info is None or 0 not in block_info:
                return np.full(flow_dir_block.shape, np.nan, dtype=np.float64)
            iy, ix = block_info[0]['chunk-location']
            h, w = flow_dir_block.shape
            exits = _compute_exit_labels_downstream(
                iy, ix, boundaries, flow_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)
            return _flow_length_downstream_tile(
                np.asarray(flow_dir_block, dtype=np.float64),
                h, w, cellsize_x, cellsize_y, diag, *exits)
    else:
        def _tile_fn(flow_dir_block, block_info=None):
            if block_info is None or 0 not in block_info:
                return np.full(flow_dir_block.shape, np.nan, dtype=np.float64)
            iy, ix = block_info[0]['chunk-location']
            h, w = flow_dir_block.shape
            seeds = _compute_seeds_upstream(
                iy, ix, boundaries, flow_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x,
                cellsize_x, cellsize_y, diag)
            return _flow_length_upstream_tile(
                np.asarray(flow_dir_block, dtype=np.float64),
                h, w, cellsize_x, cellsize_y, diag, *seeds)

    return da.map_blocks(
        _tile_fn,
        flow_dir_da,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _flow_length_dask_cupy(flow_dir_da, direction,
                            cellsize_x, cellsize_y, diag):
    """Dask+CuPy: convert to numpy, run CPU iterative path, convert back."""
    import cupy as cp

    flow_dir_np = flow_dir_da.map_blocks(
        lambda b: b.get(), dtype=flow_dir_da.dtype,
        meta=np.array((), dtype=flow_dir_da.dtype),
    )
    result = _flow_length_dask_iterative(
        flow_dir_np, direction, cellsize_x, cellsize_y, diag)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_length_d8(flow_dir: xr.DataArray,
                direction: str = 'downstream',
                name: str = 'flow_length') -> xr.DataArray:
    """Compute D8 flow length from a flow direction grid.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2D D8 flow direction grid (codes 0/1/2/4/8/16/32/64/128;
        NaN for nodata).
    direction : str, default 'downstream'
        ``'downstream'``: distance from each cell to its outlet.
        ``'upstream'``: longest path from any divide to each cell.
    name : str, default 'flow_length'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array of flow length values in coordinate units.
        NaN where flow_dir is NaN.
    """
    _validate_raster(flow_dir, func_name='flow_length', name='flow_dir')
    if direction not in ('downstream', 'upstream'):
        raise ValueError(
            f"direction must be 'downstream' or 'upstream', got {direction!r}")

    cellsize_x, cellsize_y = get_dataarray_resolution(flow_dir)
    if not (np.isfinite(cellsize_x) and cellsize_x != 0
            and np.isfinite(cellsize_y) and cellsize_y != 0):
        raise ValueError(
            f"flow_length(): cellsize must be finite and non-zero "
            f"(got cellsize_x={cellsize_x}, cellsize_y={cellsize_y}).  "
            f"Ensure flow_dir has at least 2 cells per spatial dimension "
            f"with finite coords."
        )
    cellsize_x = abs(cellsize_x)
    cellsize_y = abs(cellsize_y)
    diag = np.sqrt(cellsize_x ** 2 + cellsize_y ** 2)

    data = flow_dir.data

    if isinstance(data, np.ndarray):
        _check_memory(*data.shape)
        fd = data.astype(np.float64)
        H, W = fd.shape
        if direction == 'downstream':
            out = _flow_length_downstream_cpu(fd, H, W, cellsize_x, cellsize_y, diag)
        else:
            out = _flow_length_upstream_cpu(fd, H, W, cellsize_x, cellsize_y, diag)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        _check_gpu_memory(*data.shape)
        _check_memory(*data.shape)
        out = _flow_length_cupy(data, direction, cellsize_x, cellsize_y, diag)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _flow_length_dask_cupy(data, direction, cellsize_x, cellsize_y, diag)

    elif da is not None and isinstance(data, da.Array):
        out = _flow_length_dask_iterative(data, direction, cellsize_x, cellsize_y, diag)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
