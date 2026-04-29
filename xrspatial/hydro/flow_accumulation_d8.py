"""Flow accumulation: count of upstream cells draining through each cell.

Supports D8 (integer codes) flow direction grids.  For D-infinity
(continuous angles), see ``flow_accumulation_dinf``.

For D8 input, each cell drains to exactly one downstream neighbor.

Algorithm
---------
CPU : Kahn's BFS topological sort -- O(N).
GPU : iterative frontier peeling with pull-based kernels.
Dask: iterative tile sweep with boundary propagation (one tile in
      RAM at a time), following the ``cost_distance.py`` pattern.
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


# =====================================================================
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for ``_flow_accum_cpu``:
#   accum    : float64 -> 8
#   in_degree: int32   -> 4
#   valid    : int8    -> 1
#   queue_r  : int64   -> 8
#   queue_c  : int64   -> 8
# Total ~29 bytes/pixel.  The caller-provided ``flow_dir`` array already
# lives in RAM before the kernel runs and is not double-counted here.
_BYTES_PER_PIXEL = 29

# GPU peak working set per pixel for ``_flow_accum_cupy``:
#   accum     : float64 -> 8
#   in_degree : int32   -> 4
#   state     : int32   -> 4
# Total ~16 bytes/pixel.  ``flow_dir_data`` already lives on the device
# before the kernel runs and is not double-counted here.
_GPU_BYTES_PER_PIXEL = 16


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
    """Raise MemoryError if the BFS kernel would exceed 50% of RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"flow_accumulation on a {height}x{width} grid requires "
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
            f"flow_accumulation on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


def _to_numpy_f64(arr):
    """Convert *arr* to a contiguous numpy float64 array.

    Handles CuPy arrays transparently via ``.get()``.
    """
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


# =====================================================================
# Direction helpers
# =====================================================================

@ngjit
def _code_to_offset(code):
    """Return (dy, dx) row/col offset for a D8 direction code."""
    c = int(code)
    if c == 1:
        return 0, 1
    elif c == 2:
        return 1, 1
    elif c == 4:
        return 1, 0
    elif c == 8:
        return 1, -1
    elif c == 16:
        return 0, -1
    elif c == 32:
        return -1, -1
    elif c == 64:
        return -1, 0
    elif c == 128:
        return -1, 1
    return 0, 0


def _code_to_offset_py(code):
    """Pure-Python version for non-numba contexts."""
    c = int(code)
    _map = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
            16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
    return _map.get(c, (0, 0))


# =====================================================================
# Flow type detection (kept for backward compat)
# =====================================================================

@ngjit
def _classify_flow_block(flow_dir, height, width):
    """Classify a block of flow direction values.

    Returns 1 for definite Dinf, -1 for definite D8, 0 for ambiguous
    (all NaN or only zeros, which are valid in both conventions).
    """
    for r in range(height):
        for c in range(width):
            v = flow_dir[r, c]
            if v != v:  # NaN
                continue
            iv = int(v)
            if float(iv) != v:
                return 1  # non-integer -> Dinf
            if iv == 0:
                continue  # ambiguous: D8 pit or Dinf east
            if (iv == 1 or iv == 2 or iv == 4 or iv == 8
                    or iv == 16 or iv == 32 or iv == 64 or iv == 128):
                return -1  # definite D8
            return 1  # integer but not a D8 code (e.g. -1) -> Dinf
    return 0  # ambiguous


def _detect_flow_type(data):
    """Return 'd8' or 'dinf' based on flow direction values."""
    if da is not None and isinstance(data, da.Array):
        for iy in range(len(data.chunks[0])):
            for ix in range(len(data.chunks[1])):
                block = data.blocks[iy, ix].compute()
                sample = np.asarray(
                    block.get() if hasattr(block, 'get') else block)
                result = _classify_flow_block(
                    sample, sample.shape[0], sample.shape[1])
                if result == 1:
                    return "dinf"
                elif result == -1:
                    return "d8"
        return "d8"  # all ambiguous -> default D8
    elif hasattr(data, 'get'):  # cupy
        sample = np.asarray(data.get())
    elif isinstance(data, np.ndarray):
        sample = data
    else:
        return "d8"
    result = _classify_flow_block(sample, sample.shape[0], sample.shape[1])
    return "dinf" if result == 1 else "d8"


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _flow_accum_cpu(flow_dir, height, width):
    """Kahn's BFS topological sort for flow accumulation."""
    accum = np.empty((height, width), dtype=np.float64)
    in_degree = np.zeros((height, width), dtype=np.int32)
    valid = np.zeros((height, width), dtype=np.int8)

    # Pass 1: initialise
    for r in range(height):
        for c in range(width):
            v = flow_dir[r, c]
            if v == v:  # not NaN
                valid[r, c] = 1
                accum[r, c] = 1.0
            else:
                accum[r, c] = np.nan

    # Pass 2: compute in-degrees
    for r in range(height):
        for c in range(width):
            if valid[r, c] == 0:
                continue
            dy, dx = _code_to_offset(flow_dir[r, c])
            if dy == 0 and dx == 0:
                continue
            nr = r + dy
            nc = c + dx
            if 0 <= nr < height and 0 <= nc < width and valid[nr, nc] == 1:
                in_degree[nr, nc] += 1

    # BFS queue (flat arrays with head/tail pointers)
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

        dy, dx = _code_to_offset(flow_dir[r, c])
        if dy == 0 and dx == 0:
            continue
        nr = r + dy
        nc = c + dx
        if 0 <= nr < height and 0 <= nc < width and valid[nr, nc] == 1:
            accum[nr, nc] += accum[r, c]
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
def _init_accum_indegree(flow_dir, accum, in_degree, state, H, W):
    """Initialise accum, in_degree and state arrays on GPU."""
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

    # Decode direction (inline -- can't call @ngjit from @cuda.jit)
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
        return  # pit

    ni = i + dy
    nj = j + dx
    if 0 <= ni < H and 0 <= nj < W:
        cuda.atomic.add(in_degree, (ni, nj), 1)


@cuda.jit
def _find_ready_and_finalize(in_degree, state, changed, H, W):
    """Finalize previous frontier (2->3), mark new frontier (1->2)."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if state[i, j] == 2:
        state[i, j] = 3

    if state[i, j] == 1 and in_degree[i, j] == 0:
        state[i, j] = 2
        cuda.atomic.add(changed, 0, 1)


@cuda.jit
def _pull_from_frontier(flow_dir, accum, in_degree, state, H, W):
    """Active cells pull accumulation from frontier neighbours."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if state[i, j] != 1:
        return

    # Check all 8 neighbours
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

        # Check if neighbour's flow_dir points to me
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
            accum[i, j] += accum[ni, nj]
            in_degree[i, j] -= 1


def _flow_accum_cupy(flow_dir_data):
    """GPU driver: iterative frontier peeling."""
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)

    accum = cp.zeros((H, W), dtype=cp.float64)
    in_degree = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _init_accum_indegree[griddim, blockdim](
        flow_dir_f64, accum, in_degree, state, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _find_ready_and_finalize[griddim, blockdim](
            in_degree, state, changed, H, W)

        if int(changed[0]) == 0:
            break

        _pull_from_frontier[griddim, blockdim](
            flow_dir_f64, accum, in_degree, state, H, W)

    # Convert invalid cells to NaN
    accum = cp.where(state == 0, cp.nan, accum)
    return accum


def _flow_accum_tile_cupy(flow_dir_data,
                           seed_top, seed_bottom, seed_left, seed_right,
                           seed_tl, seed_tr, seed_bl, seed_br):
    """GPU seeded flow accumulation for a single tile.

    Same algorithm as ``_flow_accum_cupy`` but injects external seed
    values at boundary cells before frontier peeling.  Seeds are
    NumPy arrays; they are transferred to GPU inside this function.
    """
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)

    accum = cp.zeros((H, W), dtype=cp.float64)
    in_degree = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _init_accum_indegree[griddim, blockdim](
        flow_dir_f64, accum, in_degree, state, H, W)

    # Inject seeds at boundary cells.  Invalid cells (state==0) are
    # masked to NaN at the end and never enter frontier peeling, so
    # adding seeds to them is harmless.
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

        _pull_from_frontier[griddim, blockdim](
            flow_dir_f64, accum, in_degree, state, H, W)

    accum = cp.where(state == 0, cp.nan, accum)
    return accum


# =====================================================================
# Tile kernel for dask iterative path
# =====================================================================

@ngjit
def _flow_accum_tile_kernel(flow_dir, h, w,
                            seed_top, seed_bottom, seed_left, seed_right,
                            seed_tl, seed_tr, seed_bl, seed_br):
    """Seeded BFS flow accumulation for a single tile.

    Same as ``_flow_accum_cpu`` but adds external seeds to boundary
    cells before BFS.
    """
    accum = np.empty((h, w), dtype=np.float64)
    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)

    # Initialise
    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v == v:
                valid[r, c] = 1
                accum[r, c] = 1.0
            else:
                accum[r, c] = np.nan

    # Add external seeds to boundary cells
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

    # Corner seeds
    if valid[0, 0] == 1:
        accum[0, 0] += seed_tl
    if valid[0, w - 1] == 1:
        accum[0, w - 1] += seed_tr
    if valid[h - 1, 0] == 1:
        accum[h - 1, 0] += seed_bl
    if valid[h - 1, w - 1] == 1:
        accum[h - 1, w - 1] += seed_br

    # Compute in-degrees
    for r in range(h):
        for c in range(w):
            if valid[r, c] == 0:
                continue
            dy, dx = _code_to_offset(flow_dir[r, c])
            if dy == 0 and dx == 0:
                continue
            nr = r + dy
            nc = c + dx
            if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                in_degree[nr, nc] += 1

    # BFS
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
        nr = r + dy
        nc = c + dx
        if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
            accum[nr, nc] += accum[r, c]
            in_degree[nr, nc] -= 1
            if in_degree[nr, nc] == 0:
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1

    return accum


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
            flow_bdry.set('top', iy, ix, _to_numpy_f64(chunk[0, :]))
            flow_bdry.set('bottom', iy, ix, _to_numpy_f64(chunk[-1, :]))
            flow_bdry.set('left', iy, ix, _to_numpy_f64(chunk[:, 0]))
            flow_bdry.set('right', iy, ix, _to_numpy_f64(chunk[:, -1]))

    return flow_bdry


def _compute_seeds(iy, ix, boundaries, flow_bdry,
                   chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute seed arrays for tile (iy, ix) from neighbour boundaries.

    For each boundary cell of the current tile, checks whether cells
    in adjacent tiles flow INTO this tile.  If so, adds the adjacent
    cell's boundary accum value as a seed.

    Returns (seed_top, seed_bottom, seed_left, seed_right,
             seed_tl, seed_tr, seed_bl, seed_br).
    """
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

    # --- Top edge: bottom of tile above ---
    if iy > 0:
        nb_fdir = flow_bdry.get('bottom', iy - 1, ix)
        nb_accum = boundaries.get('bottom', iy - 1, ix)
        w = len(nb_fdir)
        # Cardinal S (code 4): nb cell j -> our (0, j)
        mask = (nb_fdir == 4)
        seed_top += np.where(mask, nb_accum, 0.0)
        if w > 1:
            # Diagonal SE (code 2): nb cell j -> our (0, j+1)
            mask = (nb_fdir[:-1] == 2)
            seed_top[1:] += np.where(mask, nb_accum[:-1], 0.0)
            # Diagonal SW (code 8): nb cell j -> our (0, j-1)
            mask = (nb_fdir[1:] == 8)
            seed_top[:-1] += np.where(mask, nb_accum[1:], 0.0)

    # --- Bottom edge: top of tile below ---
    if iy < n_tile_y - 1:
        nb_fdir = flow_bdry.get('top', iy + 1, ix)
        nb_accum = boundaries.get('top', iy + 1, ix)
        w = len(nb_fdir)
        # Cardinal N (code 64)
        mask = (nb_fdir == 64)
        seed_bottom += np.where(mask, nb_accum, 0.0)
        if w > 1:
            # Diagonal NE (code 128): nb cell j -> our (h-1, j+1)
            mask = (nb_fdir[:-1] == 128)
            seed_bottom[1:] += np.where(mask, nb_accum[:-1], 0.0)
            # Diagonal NW (code 32): nb cell j -> our (h-1, j-1)
            mask = (nb_fdir[1:] == 32)
            seed_bottom[:-1] += np.where(mask, nb_accum[1:], 0.0)

    # --- Left edge: right column of tile to the left ---
    if ix > 0:
        nb_fdir = flow_bdry.get('right', iy, ix - 1)
        nb_accum = boundaries.get('right', iy, ix - 1)
        h = len(nb_fdir)
        # Cardinal E (code 1)
        mask = (nb_fdir == 1)
        seed_left += np.where(mask, nb_accum, 0.0)
        if h > 1:
            # Diagonal SE (code 2): nb cell i -> our (i+1, 0)
            mask = (nb_fdir[:-1] == 2)
            seed_left[1:] += np.where(mask, nb_accum[:-1], 0.0)
            # Diagonal NE (code 128): nb cell i -> our (i-1, 0)
            mask = (nb_fdir[1:] == 128)
            seed_left[:-1] += np.where(mask, nb_accum[1:], 0.0)

    # --- Right edge: left column of tile to the right ---
    if ix < n_tile_x - 1:
        nb_fdir = flow_bdry.get('left', iy, ix + 1)
        nb_accum = boundaries.get('left', iy, ix + 1)
        h = len(nb_fdir)
        # Cardinal W (code 16)
        mask = (nb_fdir == 16)
        seed_right += np.where(mask, nb_accum, 0.0)
        if h > 1:
            # Diagonal SW (code 8): nb cell i -> our (i+1, w-1)
            mask = (nb_fdir[:-1] == 8)
            seed_right[1:] += np.where(mask, nb_accum[:-1], 0.0)
            # Diagonal NW (code 32): nb cell i -> our (i-1, w-1)
            mask = (nb_fdir[1:] == 32)
            seed_right[:-1] += np.where(mask, nb_accum[1:], 0.0)

    # --- Diagonal corner seeds ---
    # TL: bottom-right of (iy-1, ix-1) flows SE (code 2)
    if iy > 0 and ix > 0:
        fdir = flow_bdry.get('bottom', iy - 1, ix - 1)[-1]
        if fdir == 2:
            seed_tl = float(boundaries.get('bottom', iy - 1, ix - 1)[-1])
    # TR: bottom-left of (iy-1, ix+1) flows SW (code 8)
    if iy > 0 and ix < n_tile_x - 1:
        fdir = flow_bdry.get('bottom', iy - 1, ix + 1)[0]
        if fdir == 8:
            seed_tr = float(boundaries.get('bottom', iy - 1, ix + 1)[0])
    # BL: top-right of (iy+1, ix-1) flows NE (code 128)
    if iy < n_tile_y - 1 and ix > 0:
        fdir = flow_bdry.get('top', iy + 1, ix - 1)[-1]
        if fdir == 128:
            seed_bl = float(boundaries.get('top', iy + 1, ix - 1)[-1])
    # BR: top-left of (iy+1, ix+1) flows NW (code 32)
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        fdir = flow_bdry.get('top', iy + 1, ix + 1)[0]
        if fdir == 32:
            seed_br = float(boundaries.get('top', iy + 1, ix + 1)[0])

    return (seed_top, seed_bottom, seed_left, seed_right,
            seed_tl, seed_tr, seed_bl, seed_br)


def _process_tile(iy, ix, flow_dir_da, boundaries, flow_bdry,
                  chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded BFS on one tile; update boundaries in-place.

    Returns the maximum absolute boundary change (float).
    """
    chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    h, w = chunk.shape

    seeds = _compute_seeds(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    accum = _flow_accum_tile_kernel(chunk, h, w, *seeds)

    # Extract new boundary strips
    new_top = accum[0, :].copy()
    new_bottom = accum[-1, :].copy()
    new_left = accum[:, 0].copy()
    new_right = accum[:, -1].copy()

    # Compute max absolute change
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

    # Store updated boundaries
    boundaries.set('top', iy, ix, new_top)
    boundaries.set('bottom', iy, ix, new_bottom)
    boundaries.set('left', iy, ix, new_left)
    boundaries.set('right', iy, ix, new_right)

    return change


def _flow_accum_dask_iterative(flow_dir_da):
    """Iterative boundary-propagation for arbitrarily large dask arrays.

    Memory usage is O(tile_size + boundary_strips) per iteration.
    """
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    # Phase 0: extract boundary flow dirs
    flow_bdry = _preprocess_tiles(flow_dir_da, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()  # read-only from here; release temp files

    # Phase 1: initialise boundary accum to 0
    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    # Phase 2: iterative forward/backward sweeps
    max_iterations = max(n_tile_y, n_tile_x) + 10

    for _iteration in range(max_iterations):
        max_change = 0.0

        # Forward sweep (top-left -> bottom-right)
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_tile(iy, ix, flow_dir_da, boundaries,
                                  flow_bdry, chunks_y, chunks_x,
                                  n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c

        # Backward sweep (bottom-right -> top-left)
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile(iy, ix, flow_dir_da, boundaries,
                                  flow_bdry, chunks_y, chunks_x,
                                  n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c

        if max_change == 0.0:
            break

    # Snapshot converged boundaries before assembly (releases temp files)
    boundaries = boundaries.snapshot()

    # Phase 3: lazy assembly via da.map_blocks
    return _assemble_result(flow_dir_da, boundaries, flow_bdry,
                            chunks_y, chunks_x, n_tile_y, n_tile_x)


def _assemble_result(flow_dir_da, boundaries, flow_bdry,
                     chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Build a lazy dask array by re-running each tile with converged seeds."""

    def _tile_fn(flow_dir_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(flow_dir_block.shape, np.nan, dtype=np.float64)
        iy, ix = block_info[0]['chunk-location']
        h, w = flow_dir_block.shape
        seeds = _compute_seeds(
            iy, ix, boundaries, flow_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _flow_accum_tile_kernel(
            np.asarray(flow_dir_block, dtype=np.float64), h, w, *seeds)

    return da.map_blocks(
        _tile_fn,
        flow_dir_da,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _process_tile_cupy(iy, ix, flow_dir_da, boundaries, flow_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded GPU flow accumulation on one tile; update boundaries."""
    import cupy as cp

    chunk = cp.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=cp.float64)

    seeds = _compute_seeds(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    accum = _flow_accum_tile_cupy(chunk, *seeds)

    # Extract boundaries to CPU (small 1-D strips)
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


def _assemble_result_cupy(flow_dir_da, boundaries, flow_bdry,
                           chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Build a lazy dask+cupy array using GPU tile kernel."""
    import cupy as cp

    def _tile_fn(flow_dir_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return cp.full(flow_dir_block.shape, cp.nan, dtype=cp.float64)
        iy, ix = block_info[0]['chunk-location']
        seeds = _compute_seeds(
            iy, ix, boundaries, flow_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _flow_accum_tile_cupy(
            cp.asarray(flow_dir_block, dtype=cp.float64), *seeds)

    return da.map_blocks(
        _tile_fn,
        flow_dir_da,
        dtype=np.float64,
        meta=cp.array((), dtype=cp.float64),
    )


def _flow_accum_dask_cupy(flow_dir_da):
    """Dask+CuPy D8: native GPU processing per tile."""
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
                c = _process_tile_cupy(iy, ix, flow_dir_da, boundaries,
                                        flow_bdry, chunks_y, chunks_x,
                                        n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile_cupy(iy, ix, flow_dir_da, boundaries,
                                        flow_bdry, chunks_y, chunks_x,
                                        n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c

        if max_change == 0.0:
            break

    boundaries = boundaries.snapshot()

    return _assemble_result_cupy(flow_dir_da, boundaries, flow_bdry,
                                  chunks_y, chunks_x, n_tile_y, n_tile_x)


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_accumulation_d8(flow_dir: xr.DataArray,
                      name: str = 'flow_accumulation') -> xr.DataArray:
    """Compute flow accumulation from a D8 flow direction grid.

    Each cell drains to exactly one downstream neighbor based on
    integer D8 direction codes.  For D-infinity (continuous angle)
    grids, use ``flow_accumulation_dinf`` instead.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2D flow direction grid with D8 codes:
        0/1/2/4/8/16/32/64/128 (NaN for nodata).
        Supported backends: NumPy, CuPy, NumPy-backed Dask,
        CuPy-backed Dask.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='flow_accumulation'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array of flow accumulation values.  Each cell
        contains the count of upstream cells (including itself) that
        drain through it.  Cells with NaN flow direction produce NaN.

    References
    ----------
    Jenson, S.K. and Domingue, J.O. (1988). Extracting Topographic
    Structure from Digital Elevation Data for Geographic Information
    System Analysis. Photogrammetric Engineering and Remote Sensing,
    54(11), 1593-1600.
    """
    _validate_raster(flow_dir, func_name='flow_accumulation', name='flow_dir')

    data = flow_dir.data

    if isinstance(data, np.ndarray):
        _check_memory(*data.shape)
        out = _flow_accum_cpu(data.astype(np.float64), *data.shape)
    elif has_cuda_and_cupy() and is_cupy_array(data):
        _check_gpu_memory(*data.shape)
        out = _flow_accum_cupy(data)
    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _flow_accum_dask_cupy(data)
    elif da is not None and isinstance(data, da.Array):
        out = _flow_accum_dask_iterative(data)
    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
