"""Stream order extraction: Strahler and Shreve ordering of drainage networks.

Assigns hierarchical order values to stream cells derived from a D8
flow direction grid and a flow accumulation grid.  Cells with
accumulation below a user-defined threshold are non-stream and receive
NaN.  Two methods are supported:

* **Strahler**: headwaters = 1; when two streams of equal order meet
  the downstream order increments by 1; otherwise the higher order
  propagates.
* **Shreve**: headwaters = 1; at each confluence the downstream
  magnitude equals the sum of all incoming magnitudes.

Algorithm
---------
CPU : Kahn's BFS topological sort among stream cells — O(N_stream).
GPU : iterative frontier peeling with pull-based kernels.
Dask: iterative tile sweep with boundary propagation, same pattern
      as ``flow_accumulation.py``.
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
from xrspatial.hydro._boundary_store import BoundaryStore
from xrspatial.dataset_support import supports_dataset
from xrspatial.hydro.flow_accumulation_d8 import _code_to_offset, _code_to_offset_py


# =====================================================================
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for the eager Strahler/Shreve kernels:
#   order     : float64 -> 8
#   in_degree : int32   -> 4
#   max_in    : float64 -> 8   (Strahler only; Shreve omits it)
#   cnt_max   : int32   -> 4   (Strahler only)
#   queue_r   : int64   -> 8
#   queue_c   : int64   -> 8
# Total ~40 bytes/pixel for Strahler, ~32 for Shreve.  We budget for the
# worst case.  Caller-provided ``flow_dir`` and ``flow_accum`` already
# live in RAM before the kernel runs and are not double-counted here.
_BYTES_PER_PIXEL = 40

# GPU peak working set per pixel for ``_stream_order_cupy``:
#   flow_dir_f64   : float64 -> 8
#   stream_mask_i8 : int8    -> 1
#   in_degree      : int32   -> 4
#   state          : int32   -> 4
#   order          : float64 -> 8
#   max_in         : float64 -> 8
#   cnt_max        : int32   -> 4
# Total ~37 bytes/pixel.  The ``fa`` input copy adds 8 B/px on the
# device but is created from the caller's CuPy array.  Use 40 B/px as
# a conservative budget.
_GPU_BYTES_PER_PIXEL = 40


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
    """Raise MemoryError if the kernel would exceed 50% of RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"stream_order_d8 on a {height}x{width} grid requires "
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
            f"stream_order_d8 on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


def _to_numpy_f64(arr):
    """Convert *arr* to a contiguous numpy float64 array (handles CuPy)."""
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


# =====================================================================
# CPU kernels
# =====================================================================

@ngjit
def _strahler_cpu(flow_dir, stream_mask, height, width):
    """Kahn's BFS Strahler ordering among stream cells."""
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
            v = flow_dir[r, c]
            if v != v:  # NaN
                continue
            dy, dx = _code_to_offset(v)
            if dy == 0 and dx == 0:
                continue
            nr = r + dy
            nc = c + dx
            if 0 <= nr < height and 0 <= nc < width and stream_mask[nr, nc] == 1:
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

        v = flow_dir[r, c]
        if v != v:
            continue
        dy, dx = _code_to_offset(v)
        if dy == 0 and dx == 0:
            continue
        nr = r + dy
        nc = c + dx
        if not (0 <= nr < height and 0 <= nc < width and stream_mask[nr, nc] == 1):
            continue

        cur_ord = order[r, c]
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
def _shreve_cpu(flow_dir, stream_mask, height, width):
    """Kahn's BFS Shreve ordering among stream cells."""
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
            v = flow_dir[r, c]
            if v != v:
                continue
            dy, dx = _code_to_offset(v)
            if dy == 0 and dx == 0:
                continue
            nr = r + dy
            nc = c + dx
            if 0 <= nr < height and 0 <= nc < width and stream_mask[nr, nc] == 1:
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

        v = flow_dir[r, c]
        if v != v:
            continue
        dy, dx = _code_to_offset(v)
        if dy == 0 and dx == 0:
            continue
        nr = r + dy
        nc = c + dx
        if not (0 <= nr < height and 0 <= nc < width and stream_mask[nr, nc] == 1):
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
def _stream_order_init_gpu(flow_dir, stream_mask, in_degree, state,
                           order, max_in, cnt_max, H, W):
    """Initialise GPU arrays for stream order computation."""
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

    v = flow_dir[i, j]
    if v != v:
        return

    code = int(v)
    dy = 0
    dx = 0
    if code == 1:
        dy, dx = 0, 1
    elif code == 2:
        dy, dx = 1, 1
    elif code == 4:
        dy, dx = 1, 0
    elif code == 8:
        dy, dx = 1, -1
    elif code == 16:
        dy, dx = 0, -1
    elif code == 32:
        dy, dx = -1, -1
    elif code == 64:
        dy, dx = -1, 0
    elif code == 128:
        dy, dx = -1, 1

    if dy == 0 and dx == 0:
        return

    ni = i + dy
    nj = j + dx
    if 0 <= ni < H and 0 <= nj < W and stream_mask[ni, nj] == 1:
        cuda.atomic.add(in_degree, (ni, nj), 1)


@cuda.jit
def _stream_order_find_ready(in_degree, state, order, changed, H, W):
    """Finalize previous frontier, mark new frontier cells."""
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
def _stream_order_pull_strahler(flow_dir, stream_mask, in_degree, state,
                                order, max_in, cnt_max, H, W):
    """Active cells pull Strahler info from frontier neighbours."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    if state[i, j] != 1:
        return

    for k in range(8):
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
        if ni < 0 or ni >= H or nj < 0 or nj >= W:
            continue
        if state[ni, nj] != 2:
            continue
        if stream_mask[ni, nj] == 0:
            continue

        nv = flow_dir[ni, nj]
        ncode = int(nv)
        ndy = 0
        ndx = 0
        if ncode == 1:
            ndy, ndx = 0, 1
        elif ncode == 2:
            ndy, ndx = 1, 1
        elif ncode == 4:
            ndy, ndx = 1, 0
        elif ncode == 8:
            ndy, ndx = 1, -1
        elif ncode == 16:
            ndy, ndx = 0, -1
        elif ncode == 32:
            ndy, ndx = -1, -1
        elif ncode == 64:
            ndy, ndx = -1, 0
        elif ncode == 128:
            ndy, ndx = -1, 1

        if ni + ndy == i and nj + ndx == j:
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
def _stream_order_pull_shreve(flow_dir, stream_mask, in_degree, state,
                              order, H, W):
    """Active cells pull Shreve magnitudes from frontier neighbours."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    if state[i, j] != 1:
        return

    for k in range(8):
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
        if ni < 0 or ni >= H or nj < 0 or nj >= W:
            continue
        if state[ni, nj] != 2:
            continue
        if stream_mask[ni, nj] == 0:
            continue

        nv = flow_dir[ni, nj]
        ncode = int(nv)
        ndy = 0
        ndx = 0
        if ncode == 1:
            ndy, ndx = 0, 1
        elif ncode == 2:
            ndy, ndx = 1, 1
        elif ncode == 4:
            ndy, ndx = 1, 0
        elif ncode == 8:
            ndy, ndx = 1, -1
        elif ncode == 16:
            ndy, ndx = 0, -1
        elif ncode == 32:
            ndy, ndx = -1, -1
        elif ncode == 64:
            ndy, ndx = -1, 0
        elif ncode == 128:
            ndy, ndx = -1, 1

        if ni + ndy == i and nj + ndx == j:
            order[i, j] += order[ni, nj]
            in_degree[i, j] -= 1


def _stream_order_cupy(flow_dir_data, stream_mask_data, method):
    """GPU driver for stream order computation."""
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)
    stream_mask_i8 = stream_mask_data.astype(cp.int8)

    in_degree = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    order = cp.zeros((H, W), dtype=cp.float64)
    max_in = cp.zeros((H, W), dtype=cp.float64)
    cnt_max = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _stream_order_init_gpu[griddim, blockdim](
        flow_dir_f64, stream_mask_i8, in_degree, state,
        order, max_in, cnt_max, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _stream_order_find_ready[griddim, blockdim](
            in_degree, state, order, changed, H, W)

        if int(changed[0]) == 0:
            break

        if method == 'strahler':
            _stream_order_pull_strahler[griddim, blockdim](
                flow_dir_f64, stream_mask_i8, in_degree, state,
                order, max_in, cnt_max, H, W)
        else:
            _stream_order_pull_shreve[griddim, blockdim](
                flow_dir_f64, stream_mask_i8, in_degree, state,
                order, H, W)

    order = cp.where(stream_mask_i8 == 0, cp.nan, order)
    return order


# =====================================================================
# CPU tile kernels for dask
# =====================================================================

@ngjit
def _strahler_tile_kernel(flow_dir, stream_mask, h, w,
                          seed_max_top, seed_cnt_top,
                          seed_max_bottom, seed_cnt_bottom,
                          seed_max_left, seed_cnt_left,
                          seed_max_right, seed_cnt_right):
    """Seeded Strahler BFS for a single tile."""
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
            v = flow_dir[r, c]
            if v != v:
                continue
            dy, dx = _code_to_offset(v)
            if dy == 0 and dx == 0:
                continue
            nr = r + dy
            nc = c + dx
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

        v = flow_dir[r, c]
        if v != v:
            continue
        dy, dx = _code_to_offset(v)
        if dy == 0 and dx == 0:
            continue
        nr = r + dy
        nc = c + dx
        if not (0 <= nr < h and 0 <= nc < w and stream_mask[nr, nc] == 1):
            continue

        cur_ord = order[r, c]
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
def _shreve_tile_kernel(flow_dir, stream_mask, h, w,
                        seed_top, seed_bottom, seed_left, seed_right):
    """Seeded Shreve BFS for a single tile."""
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
            v = flow_dir[r, c]
            if v != v:
                continue
            dy, dx = _code_to_offset(v)
            if dy == 0 and dx == 0:
                continue
            nr = r + dy
            nc = c + dx
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

        v = flow_dir[r, c]
        if v != v:
            continue
        dy, dx = _code_to_offset(v)
        if dy == 0 and dx == 0:
            continue
        nr = r + dy
        nc = c + dx
        if not (0 <= nr < h and 0 <= nc < w and stream_mask[nr, nc] == 1):
            continue

        order[nr, nc] += order[r, c]
        in_degree[nr, nc] -= 1
        if in_degree[nr, nc] == 0:
            queue_r[tail] = nr
            queue_c[tail] = nc
            tail += 1

    return order


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _preprocess_stream_tiles(flow_dir_da, accum_da, threshold,
                             chunks_y, chunks_x):
    """Extract boundary strips for flow dir and stream mask."""
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry = BoundaryStore(chunks_y, chunks_x, fill_value=np.nan)
    mask_bdry = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            fd_chunk = _to_numpy_f64(
                flow_dir_da.blocks[iy, ix].compute())
            ac_chunk = _to_numpy_f64(
                accum_da.blocks[iy, ix].compute())
            sm = np.where(ac_chunk >= threshold, 1.0, 0.0)
            sm = np.where(np.isnan(ac_chunk), 0.0, sm)
            sm = np.where(np.isnan(fd_chunk), 0.0, sm)

            for side, row_data_fd, row_data_sm in [
                ('top', fd_chunk[0, :], sm[0, :]),
                ('bottom', fd_chunk[-1, :], sm[-1, :]),
                ('left', fd_chunk[:, 0], sm[:, 0]),
                ('right', fd_chunk[:, -1], sm[:, -1]),
            ]:
                flow_bdry.set(side, iy, ix,
                              np.asarray(row_data_fd, dtype=np.float64))
                mask_bdry.set(side, iy, ix,
                              np.asarray(row_data_sm, dtype=np.float64))

    return flow_bdry, mask_bdry


def _compute_shreve_seeds(iy, ix, boundaries, flow_bdry, mask_bdry,
                          chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute additive Shreve seeds from neighbours (same as flow_accum)."""
    tile_h = chunks_y[iy]
    tile_w = chunks_x[ix]

    seed_top = np.zeros(tile_w, dtype=np.float64)
    seed_bottom = np.zeros(tile_w, dtype=np.float64)
    seed_left = np.zeros(tile_h, dtype=np.float64)
    seed_right = np.zeros(tile_h, dtype=np.float64)

    # Top edge: bottom of tile above
    if iy > 0:
        nb_fdir = flow_bdry.get('bottom', iy - 1, ix)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix)
        nb_order = boundaries.get('bottom', iy - 1, ix)
        w = len(nb_fdir)
        for j in range(w):
            if nb_mask[j] == 0:
                continue
            code = int(nb_fdir[j])
            dy, dx = _code_to_offset_py(code)
            if dy == 1 and dx == 0 and j < tile_w:  # S
                seed_top[j] += nb_order[j]
            elif dy == 1 and dx == 1 and j + 1 < tile_w:  # SE
                seed_top[j + 1] += nb_order[j]
            elif dy == 1 and dx == -1 and j - 1 >= 0:  # SW
                seed_top[j - 1] += nb_order[j]

    # Bottom edge: top of tile below
    if iy < n_tile_y - 1:
        nb_fdir = flow_bdry.get('top', iy + 1, ix)
        nb_mask = mask_bdry.get('top', iy + 1, ix)
        nb_order = boundaries.get('top', iy + 1, ix)
        w = len(nb_fdir)
        for j in range(w):
            if nb_mask[j] == 0:
                continue
            code = int(nb_fdir[j])
            dy, dx = _code_to_offset_py(code)
            if dy == -1 and dx == 0 and j < tile_w:  # N
                seed_bottom[j] += nb_order[j]
            elif dy == -1 and dx == 1 and j + 1 < tile_w:  # NE
                seed_bottom[j + 1] += nb_order[j]
            elif dy == -1 and dx == -1 and j - 1 >= 0:  # NW
                seed_bottom[j - 1] += nb_order[j]

    # Left edge: right column of tile to the left
    if ix > 0:
        nb_fdir = flow_bdry.get('right', iy, ix - 1)
        nb_mask = mask_bdry.get('right', iy, ix - 1)
        nb_order = boundaries.get('right', iy, ix - 1)
        h = len(nb_fdir)
        for i in range(h):
            if nb_mask[i] == 0:
                continue
            code = int(nb_fdir[i])
            dy, dx = _code_to_offset_py(code)
            if dx == 1 and dy == 0 and i < tile_h:  # E
                seed_left[i] += nb_order[i]
            elif dx == 1 and dy == 1 and i + 1 < tile_h:  # SE
                seed_left[i + 1] += nb_order[i]
            elif dx == 1 and dy == -1 and i - 1 >= 0:  # NE
                seed_left[i - 1] += nb_order[i]

    # Right edge: left column of tile to the right
    if ix < n_tile_x - 1:
        nb_fdir = flow_bdry.get('left', iy, ix + 1)
        nb_mask = mask_bdry.get('left', iy, ix + 1)
        nb_order = boundaries.get('left', iy, ix + 1)
        h = len(nb_fdir)
        for i in range(h):
            if nb_mask[i] == 0:
                continue
            code = int(nb_fdir[i])
            dy, dx = _code_to_offset_py(code)
            if dx == -1 and dy == 0 and i < tile_h:  # W
                seed_right[i] += nb_order[i]
            elif dx == -1 and dy == 1 and i + 1 < tile_h:  # SW
                seed_right[i + 1] += nb_order[i]
            elif dx == -1 and dy == -1 and i - 1 >= 0:  # NW
                seed_right[i - 1] += nb_order[i]

    # Corner seeds from diagonal neighbours
    if iy > 0 and ix > 0:
        fdir = flow_bdry.get('bottom', iy - 1, ix - 1)[-1]
        mask = mask_bdry.get('bottom', iy - 1, ix - 1)[-1]
        if mask == 1 and int(fdir) == 2:  # SE
            seed_top[0] += boundaries.get('bottom', iy - 1, ix - 1)[-1]
    if iy > 0 and ix < n_tile_x - 1:
        fdir = flow_bdry.get('bottom', iy - 1, ix + 1)[0]
        mask = mask_bdry.get('bottom', iy - 1, ix + 1)[0]
        if mask == 1 and int(fdir) == 8:  # SW
            seed_top[tile_w - 1] += boundaries.get(
                'bottom', iy - 1, ix + 1)[0]
    if iy < n_tile_y - 1 and ix > 0:
        fdir = flow_bdry.get('top', iy + 1, ix - 1)[-1]
        mask = mask_bdry.get('top', iy + 1, ix - 1)[-1]
        if mask == 1 and int(fdir) == 128:  # NE
            seed_bottom[0] += boundaries.get('top', iy + 1, ix - 1)[-1]
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        fdir = flow_bdry.get('top', iy + 1, ix + 1)[0]
        mask = mask_bdry.get('top', iy + 1, ix + 1)[0]
        if mask == 1 and int(fdir) == 32:  # NW
            seed_bottom[tile_w - 1] += boundaries.get(
                'top', iy + 1, ix + 1)[0]

    return seed_top, seed_bottom, seed_left, seed_right


def _compute_strahler_seeds(iy, ix, bdry_max, bdry_cnt,
                            flow_bdry, mask_bdry,
                            chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute Strahler (max, cnt) seeds from neighbours."""
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

    # Top edge
    if iy > 0:
        nb_fdir = flow_bdry.get('bottom', iy - 1, ix)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix)
        nb_max = bdry_max.get('bottom', iy - 1, ix)
        nb_cnt = bdry_cnt.get('bottom', iy - 1, ix)
        w = len(nb_fdir)
        for j in range(w):
            if nb_mask[j] == 0 or nb_max[j] == 0:
                continue
            code = int(nb_fdir[j])
            dy, dx = _code_to_offset_py(code)
            target = -1
            if dy == 1 and dx == 0:
                target = j
            elif dy == 1 and dx == 1:
                target = j + 1
            elif dy == 1 and dx == -1:
                target = j - 1
            if 0 <= target < tile_w:
                # Reconstruct the order that this cell had
                if nb_cnt[j] >= 2:
                    val = nb_max[j] + 1.0
                else:
                    val = nb_max[j]
                _update_max_cnt(smax_top, scnt_top, val, target)

    # Bottom edge
    if iy < n_tile_y - 1:
        nb_fdir = flow_bdry.get('top', iy + 1, ix)
        nb_mask = mask_bdry.get('top', iy + 1, ix)
        nb_max = bdry_max.get('top', iy + 1, ix)
        nb_cnt = bdry_cnt.get('top', iy + 1, ix)
        w = len(nb_fdir)
        for j in range(w):
            if nb_mask[j] == 0 or nb_max[j] == 0:
                continue
            code = int(nb_fdir[j])
            dy, dx = _code_to_offset_py(code)
            target = -1
            if dy == -1 and dx == 0:
                target = j
            elif dy == -1 and dx == 1:
                target = j + 1
            elif dy == -1 and dx == -1:
                target = j - 1
            if 0 <= target < tile_w:
                if nb_cnt[j] >= 2:
                    val = nb_max[j] + 1.0
                else:
                    val = nb_max[j]
                _update_max_cnt(smax_bottom, scnt_bottom, val, target)

    # Left edge
    if ix > 0:
        nb_fdir = flow_bdry.get('right', iy, ix - 1)
        nb_mask = mask_bdry.get('right', iy, ix - 1)
        nb_max = bdry_max.get('right', iy, ix - 1)
        nb_cnt = bdry_cnt.get('right', iy, ix - 1)
        h = len(nb_fdir)
        for i in range(h):
            if nb_mask[i] == 0 or nb_max[i] == 0:
                continue
            code = int(nb_fdir[i])
            dy, dx = _code_to_offset_py(code)
            target = -1
            if dx == 1 and dy == 0:
                target = i
            elif dx == 1 and dy == 1:
                target = i + 1
            elif dx == 1 and dy == -1:
                target = i - 1
            if 0 <= target < tile_h:
                if nb_cnt[i] >= 2:
                    val = nb_max[i] + 1.0
                else:
                    val = nb_max[i]
                _update_max_cnt(smax_left, scnt_left, val, target)

    # Right edge
    if ix < n_tile_x - 1:
        nb_fdir = flow_bdry.get('left', iy, ix + 1)
        nb_mask = mask_bdry.get('left', iy, ix + 1)
        nb_max = bdry_max.get('left', iy, ix + 1)
        nb_cnt = bdry_cnt.get('left', iy, ix + 1)
        h = len(nb_fdir)
        for i in range(h):
            if nb_mask[i] == 0 or nb_max[i] == 0:
                continue
            code = int(nb_fdir[i])
            dy, dx = _code_to_offset_py(code)
            target = -1
            if dx == -1 and dy == 0:
                target = i
            elif dx == -1 and dy == 1:
                target = i + 1
            elif dx == -1 and dy == -1:
                target = i - 1
            if 0 <= target < tile_h:
                if nb_cnt[i] >= 2:
                    val = nb_max[i] + 1.0
                else:
                    val = nb_max[i]
                _update_max_cnt(smax_right, scnt_right, val, target)

    # Corners
    def _corner_seed(nb_side, nb_iy, nb_ix, nb_idx, code_expected,
                     s_max, s_cnt, target):
        fdir = flow_bdry.get(nb_side, nb_iy, nb_ix)[nb_idx]
        mask = mask_bdry.get(nb_side, nb_iy, nb_ix)[nb_idx]
        if mask == 1 and int(fdir) == code_expected:
            nm = bdry_max.get(nb_side, nb_iy, nb_ix)[nb_idx]
            nc = bdry_cnt.get(nb_side, nb_iy, nb_ix)[nb_idx]
            if nm > 0:
                val = nm + 1.0 if nc >= 2 else nm
                _update_max_cnt(s_max, s_cnt, val, target)

    if iy > 0 and ix > 0:
        _corner_seed('bottom', iy - 1, ix - 1, -1, 2,
                     smax_top, scnt_top, 0)
    if iy > 0 and ix < n_tile_x - 1:
        _corner_seed('bottom', iy - 1, ix + 1, 0, 8,
                     smax_top, scnt_top, tile_w - 1)
    if iy < n_tile_y - 1 and ix > 0:
        _corner_seed('top', iy + 1, ix - 1, -1, 128,
                     smax_bottom, scnt_bottom, 0)
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        _corner_seed('top', iy + 1, ix + 1, 0, 32,
                     smax_bottom, scnt_bottom, tile_w - 1)

    return (smax_top, scnt_top, smax_bottom, scnt_bottom,
            smax_left, scnt_left, smax_right, scnt_right)


def _process_strahler_tile(iy, ix, flow_dir_da, accum_da, threshold,
                           bdry_max, bdry_cnt, flow_bdry, mask_bdry,
                           chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded Strahler BFS on one tile; update boundary stores."""
    fd_chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    ac_chunk = np.asarray(
        accum_da.blocks[iy, ix].compute(), dtype=np.float64)
    sm = np.where(ac_chunk >= threshold, 1, 0).astype(np.int8)
    sm = np.where(np.isnan(ac_chunk), 0, sm).astype(np.int8)
    sm = np.where(np.isnan(fd_chunk), 0, sm).astype(np.int8)
    h, w = fd_chunk.shape

    seeds = _compute_strahler_seeds(
        iy, ix, bdry_max, bdry_cnt, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    order = _strahler_tile_kernel(fd_chunk, sm, h, w, *seeds)

    # Extract boundary order values and recompute max/cnt for boundaries
    change = 0.0
    for side, strip in [('top', order[0, :]),
                        ('bottom', order[-1, :]),
                        ('left', order[:, 0]),
                        ('right', order[:, -1])]:
        new_vals = strip.copy()
        # For Strahler boundaries we store the raw order value as max,
        # with cnt=1 (represents the cell's own order for seed computation).
        # The seed computation will use these as incoming orders.
        new_max = np.where(np.isnan(new_vals), 0.0, new_vals)
        new_cnt = np.where(new_max > 0, 1.0, 0.0)

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


def _process_shreve_tile(iy, ix, flow_dir_da, accum_da, threshold,
                         boundaries, flow_bdry, mask_bdry,
                         chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded Shreve BFS on one tile; update boundaries."""
    fd_chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    ac_chunk = np.asarray(
        accum_da.blocks[iy, ix].compute(), dtype=np.float64)
    sm = np.where(ac_chunk >= threshold, 1, 0).astype(np.int8)
    sm = np.where(np.isnan(ac_chunk), 0, sm).astype(np.int8)
    sm = np.where(np.isnan(fd_chunk), 0, sm).astype(np.int8)
    h, w = fd_chunk.shape

    seeds = _compute_shreve_seeds(
        iy, ix, boundaries, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    order = _shreve_tile_kernel(fd_chunk, sm, h, w, *seeds)

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


def _stream_order_dask_strahler(flow_dir_da, accum_da, threshold):
    """Dask iterative sweep for Strahler ordering."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry, mask_bdry = _preprocess_stream_tiles(
        flow_dir_da, accum_da, threshold, chunks_y, chunks_x)
    # Read-only from here; release temp files
    flow_bdry = flow_bdry.snapshot()
    mask_bdry = mask_bdry.snapshot()

    bdry_max = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)
    bdry_cnt = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    max_iterations = max(n_tile_y, n_tile_x) + 10
    for _ in range(max_iterations):
        max_change = 0.0
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_strahler_tile(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    bdry_max, bdry_cnt, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_strahler_tile(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    bdry_max, bdry_cnt, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        if max_change == 0.0:
            break

    # Snapshot converged boundaries before assembly (releases temp files)
    _bdry_max = bdry_max.snapshot()
    _bdry_cnt = bdry_cnt.snapshot()
    _flow_bdry = flow_bdry
    _mask_bdry = mask_bdry
    _threshold = threshold

    def _tile_fn(block, accum_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(block.shape, np.nan, dtype=np.float64)
        iy, ix = block_info[0]['chunk-location']
        fd = np.asarray(block, dtype=np.float64)
        ac = np.asarray(accum_block, dtype=np.float64)
        sm = np.where(ac >= _threshold, 1, 0).astype(np.int8)
        sm = np.where(np.isnan(ac), 0, sm).astype(np.int8)
        sm = np.where(np.isnan(fd), 0, sm).astype(np.int8)
        h, w = fd.shape
        seeds = _compute_strahler_seeds(
            iy, ix, _bdry_max, _bdry_cnt, _flow_bdry, _mask_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _strahler_tile_kernel(fd, sm, h, w, *seeds)

    return da.map_blocks(
        _tile_fn, flow_dir_da, accum_da,
        dtype=np.float64, meta=np.array((), dtype=np.float64))


def _stream_order_dask_shreve(flow_dir_da, accum_da, threshold):
    """Dask iterative sweep for Shreve ordering."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry, mask_bdry = _preprocess_stream_tiles(
        flow_dir_da, accum_da, threshold, chunks_y, chunks_x)
    # Read-only from here; release temp files
    flow_bdry = flow_bdry.snapshot()
    mask_bdry = mask_bdry.snapshot()

    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    max_iterations = max(n_tile_y, n_tile_x) + 10
    for _ in range(max_iterations):
        max_change = 0.0
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_shreve_tile(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    boundaries, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_shreve_tile(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    boundaries, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        if max_change == 0.0:
            break

    # Snapshot converged boundaries before assembly (releases temp files)
    _boundaries = boundaries.snapshot()
    _flow_bdry = flow_bdry
    _mask_bdry = mask_bdry
    _threshold = threshold

    def _tile_fn(block, accum_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(block.shape, np.nan, dtype=np.float64)
        iy, ix = block_info[0]['chunk-location']
        fd = np.asarray(block, dtype=np.float64)
        ac = np.asarray(accum_block, dtype=np.float64)
        sm = np.where(ac >= _threshold, 1, 0).astype(np.int8)
        sm = np.where(np.isnan(ac), 0, sm).astype(np.int8)
        sm = np.where(np.isnan(fd), 0, sm).astype(np.int8)
        h, w = fd.shape
        seeds = _compute_shreve_seeds(
            iy, ix, _boundaries, _flow_bdry, _mask_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _shreve_tile_kernel(fd, sm, h, w, *seeds)

    return da.map_blocks(
        _tile_fn, flow_dir_da, accum_da,
        dtype=np.float64, meta=np.array((), dtype=np.float64))


def _stream_order_tile_cupy(flow_dir_data, stream_mask_data, method,
                             seeds):
    """GPU seeded stream order for a single tile.

    Uses GPU frontier peeling with seeds injected after initialisation.
    For Strahler, seeds are (max_top, cnt_top, max_bot, cnt_bot, ...).
    For Shreve, seeds are (top, bot, left, right).
    """
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)
    stream_mask_i8 = stream_mask_data.astype(cp.int8)

    in_degree = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    order = cp.zeros((H, W), dtype=cp.float64)
    max_in = cp.zeros((H, W), dtype=cp.float64)
    cnt_max = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _stream_order_init_gpu[griddim, blockdim](
        flow_dir_f64, stream_mask_i8, in_degree, state,
        order, max_in, cnt_max, H, W)

    if method == 'strahler':
        (smax_top, scnt_top, smax_bot, scnt_bot,
         smax_left, scnt_left, smax_right, scnt_right) = seeds
        # Inject Strahler seeds at boundaries.
        # After init, max_in is 0.0 everywhere.  Where seed_max > 0,
        # set max_in and cnt_max.
        for r_idx, s_max, s_cnt in [
            ((0, slice(None)), smax_top, scnt_top),
            ((H - 1, slice(None)), smax_bot, scnt_bot),
            ((slice(None), 0), smax_left, scnt_left),
            ((slice(None), W - 1), smax_right, scnt_right),
        ]:
            sm_cp = cp.asarray(s_max)
            sc_cp = cp.asarray(s_cnt).astype(cp.int32)
            is_stream = stream_mask_i8[r_idx] == 1
            has_seed = sm_cp > 0
            mask = is_stream & has_seed
            # Since max_in starts at 0, seed_max > 0 always wins
            max_in[r_idx] = cp.where(mask, sm_cp, max_in[r_idx])
            cnt_max[r_idx] = cp.where(mask, sc_cp, cnt_max[r_idx])
    else:
        (seed_top, seed_bot, seed_left, seed_right) = seeds
        # Inject Shreve seeds: additive, same as flow_accumulation
        order[0, :] += cp.asarray(seed_top)
        order[H - 1, :] += cp.asarray(seed_bot)
        order[:, 0] += cp.asarray(seed_left)
        order[:, W - 1] += cp.asarray(seed_right)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _stream_order_find_ready[griddim, blockdim](
            in_degree, state, order, changed, H, W)

        if int(changed[0]) == 0:
            break

        if method == 'strahler':
            _stream_order_pull_strahler[griddim, blockdim](
                flow_dir_f64, stream_mask_i8, in_degree, state,
                order, max_in, cnt_max, H, W)
        else:
            _stream_order_pull_shreve[griddim, blockdim](
                flow_dir_f64, stream_mask_i8, in_degree, state,
                order, H, W)

    order = cp.where(stream_mask_i8 == 0, cp.nan, order)
    return order


def _make_stream_mask_np(ac_chunk, fd_chunk, threshold):
    """Build stream mask as numpy int8 from accumulation chunk."""
    sm = np.where(ac_chunk >= threshold, 1, 0).astype(np.int8)
    sm = np.where(np.isnan(ac_chunk), 0, sm).astype(np.int8)
    sm = np.where(np.isnan(fd_chunk), 0, sm).astype(np.int8)
    return sm


def _process_strahler_tile_cupy(iy, ix, flow_dir_da, accum_da, threshold,
                                  bdry_max, bdry_cnt, flow_bdry, mask_bdry,
                                  chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded GPU Strahler on one tile; update boundary stores."""
    import cupy as cp

    fd_chunk = cp.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=cp.float64)
    ac_chunk = _to_numpy_f64(accum_da.blocks[iy, ix].compute())
    fd_np = fd_chunk.get()
    sm_np = _make_stream_mask_np(ac_chunk, fd_np, threshold)
    sm_cp = cp.asarray(sm_np)

    seeds = _compute_strahler_seeds(
        iy, ix, bdry_max, bdry_cnt, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    order = _stream_order_tile_cupy(fd_chunk, sm_cp, 'strahler', seeds)

    change = 0.0
    for side, strip_cp in [('top', order[0, :]),
                           ('bottom', order[-1, :]),
                           ('left', order[:, 0]),
                           ('right', order[:, -1])]:
        new_vals = strip_cp.get().copy()
        new_max = np.where(np.isnan(new_vals), 0.0, new_vals)
        new_cnt = np.where(new_max > 0, 1.0, 0.0)

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


def _process_shreve_tile_cupy(iy, ix, flow_dir_da, accum_da, threshold,
                                boundaries, flow_bdry, mask_bdry,
                                chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded GPU Shreve on one tile; update boundaries."""
    import cupy as cp

    fd_chunk = cp.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=cp.float64)
    ac_chunk = _to_numpy_f64(accum_da.blocks[iy, ix].compute())
    fd_np = fd_chunk.get()
    sm_np = _make_stream_mask_np(ac_chunk, fd_np, threshold)
    sm_cp = cp.asarray(sm_np)

    seeds = _compute_shreve_seeds(
        iy, ix, boundaries, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    order = _stream_order_tile_cupy(fd_chunk, sm_cp, 'shreve', seeds)

    change = 0.0
    for side, strip_cp in [('top', order[0, :]),
                           ('bottom', order[-1, :]),
                           ('left', order[:, 0]),
                           ('right', order[:, -1])]:
        new_vals = strip_cp.get().copy()
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


def _stream_order_dask_cupy(flow_dir_da, accum_da, threshold, method):
    """Dask+CuPy: native GPU processing per tile."""
    import cupy as cp

    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry, mask_bdry = _preprocess_stream_tiles(
        flow_dir_da, accum_da, threshold, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()
    mask_bdry = mask_bdry.snapshot()

    max_iterations = max(n_tile_y, n_tile_x) + 10

    if method == 'strahler':
        bdry_max = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)
        bdry_cnt = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

        for _ in range(max_iterations):
            max_change = 0.0
            for iy in range(n_tile_y):
                for ix in range(n_tile_x):
                    c = _process_strahler_tile_cupy(
                        iy, ix, flow_dir_da, accum_da, threshold,
                        bdry_max, bdry_cnt, flow_bdry, mask_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x)
                    if c > max_change:
                        max_change = c
            for iy in reversed(range(n_tile_y)):
                for ix in reversed(range(n_tile_x)):
                    c = _process_strahler_tile_cupy(
                        iy, ix, flow_dir_da, accum_da, threshold,
                        bdry_max, bdry_cnt, flow_bdry, mask_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x)
                    if c > max_change:
                        max_change = c
            if max_change == 0.0:
                break

        _bdry_max = bdry_max.snapshot()
        _bdry_cnt = bdry_cnt.snapshot()

        def _tile_fn(block, accum_block, block_info=None):
            if block_info is None or 0 not in block_info:
                return cp.full(block.shape, cp.nan, dtype=cp.float64)
            iy, ix = block_info[0]['chunk-location']
            fd = cp.asarray(block, dtype=cp.float64)
            ac_np = _to_numpy_f64(accum_block)
            fd_np = fd.get()
            sm = cp.asarray(_make_stream_mask_np(ac_np, fd_np, threshold))
            seeds = _compute_strahler_seeds(
                iy, ix, _bdry_max, _bdry_cnt, flow_bdry, mask_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)
            return _stream_order_tile_cupy(fd, sm, 'strahler', seeds)

    else:  # shreve
        boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

        for _ in range(max_iterations):
            max_change = 0.0
            for iy in range(n_tile_y):
                for ix in range(n_tile_x):
                    c = _process_shreve_tile_cupy(
                        iy, ix, flow_dir_da, accum_da, threshold,
                        boundaries, flow_bdry, mask_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x)
                    if c > max_change:
                        max_change = c
            for iy in reversed(range(n_tile_y)):
                for ix in reversed(range(n_tile_x)):
                    c = _process_shreve_tile_cupy(
                        iy, ix, flow_dir_da, accum_da, threshold,
                        boundaries, flow_bdry, mask_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x)
                    if c > max_change:
                        max_change = c
            if max_change == 0.0:
                break

        _boundaries = boundaries.snapshot()

        def _tile_fn(block, accum_block, block_info=None):
            if block_info is None or 0 not in block_info:
                return cp.full(block.shape, cp.nan, dtype=cp.float64)
            iy, ix = block_info[0]['chunk-location']
            fd = cp.asarray(block, dtype=cp.float64)
            ac_np = _to_numpy_f64(accum_block)
            fd_np = fd.get()
            sm = cp.asarray(_make_stream_mask_np(ac_np, fd_np, threshold))
            seeds = _compute_shreve_seeds(
                iy, ix, _boundaries, flow_bdry, mask_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)
            return _stream_order_tile_cupy(fd, sm, 'shreve', seeds)

    return da.map_blocks(
        _tile_fn, flow_dir_da, accum_da,
        dtype=np.float64, meta=cp.array((), dtype=cp.float64))


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def stream_order_d8(flow_dir: xr.DataArray,
                    flow_accum: xr.DataArray,
                    threshold: float = 100,
                    ordering: str = 'strahler',
                    name: str = 'stream_order') -> xr.DataArray:
    """Compute stream order from D8 flow direction and accumulation grids.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2D D8 flow direction grid (codes 0/1/2/4/8/16/32/64/128;
        NaN for nodata).
    flow_accum : xarray.DataArray
        2D flow accumulation grid.  Cells with
        ``flow_accum >= threshold`` are considered stream cells.
    threshold : float, default 100
        Minimum accumulation to classify a cell as part of the
        stream network.
    ordering : str, default 'strahler'
        ``'strahler'`` for Strahler branching hierarchy or
        ``'shreve'`` for Shreve cumulative magnitude.
    name : str, default 'stream_order'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array of stream order values.  Non-stream cells
        (accumulation below threshold) are NaN.

    References
    ----------
    Strahler, A.N. (1957). Quantitative analysis of watershed
    geomorphology. Transactions of the American Geophysical Union,
    38(6), 913-920.

    Shreve, R.L. (1966). Statistical law of stream numbers. Journal
    of Geology, 74(1), 17-37.
    """
    _validate_raster(flow_dir, func_name='stream_order', name='flow_dir')

    method = ordering.lower()
    if method not in ('strahler', 'shreve'):
        raise ValueError(
            f"ordering must be 'strahler' or 'shreve', got {ordering!r}")

    fd_data = flow_dir.data
    fa_data = flow_accum.data

    if isinstance(fd_data, np.ndarray):
        _check_memory(*fd_data.shape)
        fd = fd_data.astype(np.float64)
        fa = np.asarray(fa_data, dtype=np.float64)
        stream_mask = np.where(fa >= threshold, 1, 0).astype(np.int8)
        stream_mask = np.where(np.isnan(fa), 0, stream_mask).astype(np.int8)
        # NaN flow_dir → not stream
        stream_mask = np.where(np.isnan(fd), 0, stream_mask).astype(np.int8)
        h, w = fd.shape
        if method == 'strahler':
            out = _strahler_cpu(fd, stream_mask, h, w)
        else:
            out = _shreve_cpu(fd, stream_mask, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(fd_data):
        _check_gpu_memory(*fd_data.shape)
        import cupy as cp
        fa_cp = cp.asarray(fa_data, dtype=cp.float64)
        fd_cp = fd_data.astype(cp.float64)
        stream_mask = cp.where(fa_cp >= threshold, 1, 0).astype(cp.int8)
        stream_mask = cp.where(cp.isnan(fa_cp), 0, stream_mask).astype(cp.int8)
        stream_mask = cp.where(cp.isnan(fd_cp), 0, stream_mask).astype(cp.int8)
        out = _stream_order_cupy(fd_cp, stream_mask, method)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _stream_order_dask_cupy(fd_data, fa_data, threshold, method)

    elif da is not None and isinstance(fd_data, da.Array):
        if method == 'strahler':
            out = _stream_order_dask_strahler(fd_data, fa_data, threshold)
        else:
            out = _stream_order_dask_shreve(fd_data, fa_data, threshold)

    else:
        raise TypeError(f"Unsupported array type: {type(fd_data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
