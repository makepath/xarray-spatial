"""Watershed delineation and drainage basin labeling.

Two complementary functions:
- ``watershed(flow_dir, pour_points)`` — labels each cell with the
  pour point it drains to; cells not reaching any pour point → NaN.
- ``basins(flow_dir)`` — automatically identifies all outlets (pits +
  edge-exit cells) and labels every valid cell; no pour points needed.

Both use **downstream tracing with path compression** on CPU — follow
each cell's flow_dir downstream until hitting a labeled cell, then
label the entire traced path.  O(N) amortized.

GPU uses iterative label propagation (one hop per iteration).
Dask uses iterative tile sweep with exit-label propagation.
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


def _to_numpy_f64(arr):
    """Convert *arr* to a contiguous numpy float64 array (handles CuPy)."""
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


# =====================================================================
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for the numpy dispatch + ``_watershed_cpu``:
#   fd (float64 cast)    -> 8
#   labels (float64)     -> 8
#   state  (int8)        -> 1
#   path_r (int64)       -> 8
#   path_c (int64)       -> 8
# Total ~33 bytes/pixel.  The caller's ``flow_dir`` and ``pour_points``
# arrays already live in RAM before dispatch and are not double-counted.
_BYTES_PER_PIXEL = 33

# GPU peak working set per pixel for ``_watershed_cupy``:
#   flow_dir_f64 (float64) -> 8
#   pp_f64       (float64) -> 8
#   labels       (float64) -> 8
#   state        (int32)   -> 4
# Total 28 bytes/pixel on the device.
_GPU_BYTES_PER_PIXEL = 28


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
    """Raise MemoryError if the watershed kernel would exceed 50% of RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"watershed_d8 on a {height}x{width} grid requires "
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
            f"watershed_d8 on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


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
    import math
    if isinstance(code, float) and math.isnan(code):
        return (0, 0)
    c = int(code)
    _map = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
            16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
    return _map.get(c, (0, 0))


# =====================================================================
# CPU kernels
# =====================================================================

@ngjit
def _watershed_cpu(flow_dir, labels, state, h, w):
    """Downstream tracing with path compression for watershed.

    Uses a separate ``state`` array to track cell status, so that
    pour-point labels can be any float value (including negative).

    State values: 0=nodata, 1=unresolved, 2=in-trace, 3=resolved.
    On return every reachable cell has state 3 and the label of its
    pour point; unreachable cells have state 0 and NaN.
    """
    path_r = np.empty(h * w, dtype=np.int64)
    path_c = np.empty(h * w, dtype=np.int64)

    for r in range(h):
        for c in range(w):
            if state[r, c] != 1:
                continue  # already resolved, nodata, or in-trace

            # Trace downstream, collecting path
            path_len = 0
            cr, cc = r, c
            found_label = np.nan
            found = False

            while True:
                s = state[cr, cc]
                if s == 3:
                    # Hit a resolved cell (pour point or previously resolved)
                    found_label = labels[cr, cc]
                    found = True
                    break
                if s != 1:
                    # nodata (0) or in-trace (2) → cycle or dead end
                    break

                path_r[path_len] = cr
                path_c[path_len] = cc
                path_len += 1
                state[cr, cc] = 2  # in-trace marker

                v = flow_dir[cr, cc]
                if v != v:  # NaN
                    break
                dy, dx = _code_to_offset(v)
                if dy == 0 and dx == 0:
                    break  # pit with no pour point
                nr, nc = cr + dy, cc + dx
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    break  # exits grid
                cr, cc = nr, nc

            # Assign label to entire traced path
            for i in range(path_len):
                if found:
                    labels[path_r[i], path_c[i]] = found_label
                    state[path_r[i], path_c[i]] = 3
                else:
                    labels[path_r[i], path_c[i]] = np.nan
                    state[path_r[i], path_c[i]] = 0

    return labels




# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit
def _init_watershed_gpu(flow_dir, pour_points, labels, state, H, W):
    """Pour points → labeled + frontier. NaN → state 0. Others → state 1."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    v = flow_dir[i, j]
    if v != v:  # NaN
        state[i, j] = 0
        labels[i, j] = 0.0
        return

    pp = pour_points[i, j]
    if pp == pp:  # not NaN → pour point
        labels[i, j] = pp
        state[i, j] = 2  # frontier
    else:
        labels[i, j] = 0.0
        state[i, j] = 1  # active




@cuda.jit
def _propagate_labels_gpu(flow_dir, labels, state, changed, H, W):
    """Each active cell follows flow_dir one hop. If downstream is frontier → take label."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if state[i, j] != 1:
        return

    v = flow_dir[i, j]
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
    if ni < 0 or ni >= H or nj < 0 or nj >= W:
        return

    if state[ni, nj] == 2:  # downstream is frontier
        labels[i, j] = labels[ni, nj]
        state[i, j] = 3  # newly labeled
        cuda.atomic.add(changed, 0, 1)


@cuda.jit
def _advance_frontier_gpu(state, H, W):
    """state 2→4 (done), state 3→2 (new frontier)."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    s = state[i, j]
    if s == 2:
        state[i, j] = 4
    elif s == 3:
        state[i, j] = 2


def _watershed_cupy(flow_dir_data, pour_points_data):
    """GPU driver for watershed."""
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)
    pp_f64 = pour_points_data.astype(cp.float64)

    labels = cp.zeros((H, W), dtype=cp.float64)
    state = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _init_watershed_gpu[griddim, blockdim](
        flow_dir_f64, pp_f64, labels, state, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _propagate_labels_gpu[griddim, blockdim](
            flow_dir_f64, labels, state, changed, H, W)
        if int(changed[0]) == 0:
            break
        _advance_frontier_gpu[griddim, blockdim](state, H, W)

    # Unresolved (state=1) and invalid (state=0) → NaN
    labels = cp.where((state == 1) | (state == 0), cp.nan, labels)
    return labels




# =====================================================================
# Tile kernel for dask iterative path
# =====================================================================

@ngjit
def _watershed_tile_kernel(flow_dir, h, w, pour_points,
                           exit_top, exit_bottom, exit_left, exit_right,
                           exit_tl, exit_tr, exit_bl, exit_br):
    """Seeded downstream tracing for a single tile.

    Uses a separate state array so pour-point labels can be any float
    value (including negative).  State: 0=nodata, 1=unresolved,
    2=in-trace, 3=resolved.
    """
    labels = np.empty((h, w), dtype=np.float64)
    state = np.empty((h, w), dtype=np.int8)

    # Initialise labels and state
    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v != v:  # NaN
                labels[r, c] = np.nan
                state[r, c] = 0
                continue
            pp = pour_points[r, c]
            if pp == pp:  # not NaN → pour point
                labels[r, c] = pp
                state[r, c] = 3
                continue
            labels[r, c] = np.nan
            state[r, c] = 1  # unresolved

    # Apply exit labels to boundary cells that flow OUT of tile
    # Top row: cells flowing north
    for c in range(w):
        if state[0, c] == 1:
            el = exit_top[c]
            if el == el:  # not NaN → resolved
                labels[0, c] = el
                state[0, c] = 3
    # Bottom row
    for c in range(w):
        if state[h - 1, c] == 1:
            el = exit_bottom[c]
            if el == el:
                labels[h - 1, c] = el
                state[h - 1, c] = 3
    # Left column
    for r in range(h):
        if state[r, 0] == 1:
            el = exit_left[r]
            if el == el:
                labels[r, 0] = el
                state[r, 0] = 3
    # Right column
    for r in range(h):
        if state[r, w - 1] == 1:
            el = exit_right[r]
            if el == el:
                labels[r, w - 1] = el
                state[r, w - 1] = 3

    # Corners
    if state[0, 0] == 1 and exit_tl == exit_tl:
        labels[0, 0] = exit_tl
        state[0, 0] = 3
    if state[0, w - 1] == 1 and exit_tr == exit_tr:
        labels[0, w - 1] = exit_tr
        state[0, w - 1] = 3
    if state[h - 1, 0] == 1 and exit_bl == exit_bl:
        labels[h - 1, 0] = exit_bl
        state[h - 1, 0] = 3
    if state[h - 1, w - 1] == 1 and exit_br == exit_br:
        labels[h - 1, w - 1] = exit_br
        state[h - 1, w - 1] = 3

    # Downstream tracing with path compression
    path_r = np.empty(h * w, dtype=np.int64)
    path_c = np.empty(h * w, dtype=np.int64)

    for r in range(h):
        for c in range(w):
            if state[r, c] != 1:
                continue

            path_len = 0
            cr, cc = r, c
            found_label = np.nan
            found = False
            exit_tile = False

            while True:
                s = state[cr, cc]
                if s == 3:
                    found_label = labels[cr, cc]
                    found = True
                    break
                if s != 1:
                    break

                path_r[path_len] = cr
                path_c[path_len] = cc
                path_len += 1
                state[cr, cc] = 2

                v = flow_dir[cr, cc]
                if v != v:
                    break
                dy, dx = _code_to_offset(v)
                if dy == 0 and dx == 0:
                    break
                nr, nc = cr + dy, cc + dx
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    # Exits tile — leave as unresolved
                    exit_tile = True
                    break
                cr, cc = nr, nc

            for i in range(path_len):
                if found:
                    labels[path_r[i], path_c[i]] = found_label
                    state[path_r[i], path_c[i]] = 3
                elif exit_tile:
                    state[path_r[i], path_c[i]] = 1  # still unresolved
                else:
                    labels[path_r[i], path_c[i]] = np.nan
                    state[path_r[i], path_c[i]] = 0  # dead end

    return labels


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


def _compute_exit_labels(iy, ix, boundaries, flow_bdry,
                         chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute exit labels for tile (iy, ix).

    For each boundary cell of the current tile, check if its flow_dir
    points OUTSIDE the tile.  If so, look up the destination cell's
    resolved label in the adjacent tile's boundary data.

    This is the reverse of flow_accumulation's seed computation:
    - flow_accum: "who flows INTO my boundary?" (entry seeds)
    - watershed: "where does my boundary cell flow TO?" (exit labels)
    """
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

    # --- Top row: cells that flow north/NW/NE out of tile ---
    if iy > 0:
        fdir_top = flow_bdry.get('top', iy, ix)
        nb_labels = boundaries.get('bottom', iy - 1, ix)
        for j in range(tile_w):
            d = _code_to_offset_py(fdir_top[j])
            if d[0] == -1:  # flows north
                # Destination column in adjacent tile
                dj = j + d[1]
                if d[1] == 0:
                    # Cardinal N (64): dest is bottom[iy-1][ix][j]
                    if 0 <= dj < len(nb_labels):
                        exit_top[j] = nb_labels[dj]
                elif d[1] == -1:
                    # NW (32): dest is bottom[iy-1][ix][j-1] or corner
                    if 0 <= dj < len(nb_labels):
                        exit_top[j] = nb_labels[dj]
                    elif dj < 0 and ix > 0:
                        exit_top[j] = boundaries.get('bottom', iy - 1, ix - 1)[-1]
                elif d[1] == 1:
                    # NE (128): dest is bottom[iy-1][ix][j+1] or corner
                    if 0 <= dj < len(nb_labels):
                        exit_top[j] = nb_labels[dj]
                    elif dj >= len(nb_labels) and ix < n_tile_x - 1:
                        exit_top[j] = boundaries.get('bottom', iy - 1, ix + 1)[0]

    # --- Bottom row: cells that flow south/SW/SE out of tile ---
    if iy < n_tile_y - 1:
        fdir_bot = flow_bdry.get('bottom', iy, ix)
        nb_labels = boundaries.get('top', iy + 1, ix)
        for j in range(tile_w):
            d = _code_to_offset_py(fdir_bot[j])
            if d[0] == 1:  # flows south
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

    # --- Left column: cells that flow west/NW/SW out of tile ---
    if ix > 0:
        fdir_left = flow_bdry.get('left', iy, ix)
        nb_labels = boundaries.get('right', iy, ix - 1)
        for r in range(tile_h):
            d = _code_to_offset_py(fdir_left[r])
            if d[1] == -1:  # flows west
                dr = r + d[0]
                if d[0] == 0:
                    if 0 <= dr < len(nb_labels):
                        exit_left[r] = nb_labels[dr]
                elif d[0] == -1:
                    if r == 0:
                        continue  # handled by top-left corner
                    if 0 <= dr < len(nb_labels):
                        exit_left[r] = nb_labels[dr]
                elif d[0] == 1:
                    if r == tile_h - 1:
                        continue
                    if 0 <= dr < len(nb_labels):
                        exit_left[r] = nb_labels[dr]

    # --- Right column: cells that flow east/NE/SE out of tile ---
    if ix < n_tile_x - 1:
        fdir_right = flow_bdry.get('right', iy, ix)
        nb_labels = boundaries.get('left', iy, ix + 1)
        for r in range(tile_h):
            d = _code_to_offset_py(fdir_right[r])
            if d[1] == 1:  # flows east
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

    # --- Also handle edge-of-grid cells that flow off grid ---
    # Top row with no tile above
    if iy == 0:
        fdir_top = flow_bdry.get('top', iy, ix)
        for j in range(tile_w):
            d = _code_to_offset_py(fdir_top[j])
            if d[0] == -1:
                exit_top[j] = np.nan  # flows off grid
    # Bottom row with no tile below
    if iy == n_tile_y - 1:
        fdir_bot = flow_bdry.get('bottom', iy, ix)
        for j in range(tile_w):
            d = _code_to_offset_py(fdir_bot[j])
            if d[0] == 1:
                exit_bottom[j] = np.nan
    # Left col with no tile left
    if ix == 0:
        fdir_left = flow_bdry.get('left', iy, ix)
        for r in range(tile_h):
            d = _code_to_offset_py(fdir_left[r])
            if d[1] == -1:
                exit_left[r] = np.nan
    # Right col with no tile right
    if ix == n_tile_x - 1:
        fdir_right = flow_bdry.get('right', iy, ix)
        for r in range(tile_h):
            d = _code_to_offset_py(fdir_right[r])
            if d[1] == 1:
                exit_right[r] = np.nan

    # --- Diagonal corners ---
    # TL corner of this tile (0,0) flows to tile (iy-1, ix-1)?
    fdir_tl = flow_bdry.get('top', iy, ix)[0]
    d = _code_to_offset_py(fdir_tl)
    if d == (-1, -1):  # NW
        if iy > 0 and ix > 0:
            exit_tl = boundaries.get('bottom', iy - 1, ix - 1)[-1]
        else:
            exit_tl = np.nan

    # TR corner (0, w-1)
    fdir_tr = flow_bdry.get('top', iy, ix)[-1]
    d = _code_to_offset_py(fdir_tr)
    if d == (-1, 1):  # NE
        if iy > 0 and ix < n_tile_x - 1:
            exit_tr = boundaries.get('bottom', iy - 1, ix + 1)[0]
        else:
            exit_tr = np.nan

    # BL corner (h-1, 0)
    fdir_bl = flow_bdry.get('bottom', iy, ix)[0]
    d = _code_to_offset_py(fdir_bl)
    if d == (1, -1):  # SW
        if iy < n_tile_y - 1 and ix > 0:
            exit_bl = boundaries.get('top', iy + 1, ix - 1)[-1]
        else:
            exit_bl = np.nan

    # BR corner (h-1, w-1)
    fdir_br = flow_bdry.get('bottom', iy, ix)[-1]
    d = _code_to_offset_py(fdir_br)
    if d == (1, 1):  # SE
        if iy < n_tile_y - 1 and ix < n_tile_x - 1:
            exit_br = boundaries.get('top', iy + 1, ix + 1)[0]
        else:
            exit_br = np.nan

    return (exit_top, exit_bottom, exit_left, exit_right,
            exit_tl, exit_tr, exit_bl, exit_br)


def _process_tile_watershed(iy, ix, flow_dir_da, pour_points_da,
                            boundaries, flow_bdry,
                            chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded tracing on one tile; update boundaries in-place.

    Returns whether any boundary label changed (bool).
    """
    chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    pp_chunk = np.asarray(
        pour_points_da.blocks[iy, ix].compute(), dtype=np.float64)
    h, w = chunk.shape

    exits = _compute_exit_labels(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    result = _watershed_tile_kernel(chunk, h, w, pp_chunk, *exits)

    # Extract new boundary labels
    new_top = result[0, :].copy()
    new_bottom = result[-1, :].copy()
    new_left = result[:, 0].copy()
    new_right = result[:, -1].copy()

    # Check for changes
    changed = False
    for side, new in (('top', new_top), ('bottom', new_bottom),
                      ('left', new_left), ('right', new_right)):
        old = boundaries.get(side, iy, ix).copy()
        with np.errstate(invalid='ignore'):
            # Changed if any value differs (considering NaN==NaN as same)
            mask = ~(np.isnan(old) & np.isnan(new))
            if mask.any():
                diff = old[mask] != new[mask]
                if np.any(diff):
                    changed = True
                    break

    boundaries.set('top', iy, ix, new_top)
    boundaries.set('bottom', iy, ix, new_bottom)
    boundaries.set('left', iy, ix, new_left)
    boundaries.set('right', iy, ix, new_right)

    return changed


def _watershed_dask_iterative(flow_dir_da, pour_points_da):
    """Iterative boundary-propagation for watershed on dask arrays."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry = _preprocess_tiles(flow_dir_da, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()  # read-only from here; release temp files

    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=np.nan)

    max_iterations = max(n_tile_y, n_tile_x) * 2 + 10

    for _iteration in range(max_iterations):
        any_changed = False

        # Forward sweep
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_tile_watershed(
                    iy, ix, flow_dir_da, pour_points_da,
                    boundaries, flow_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True

        # Backward sweep
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile_watershed(
                    iy, ix, flow_dir_da, pour_points_da,
                    boundaries, flow_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True

        if not any_changed:
            break

    # Snapshot converged boundaries before assembly (releases temp files)
    boundaries = boundaries.snapshot()

    return _assemble_watershed(flow_dir_da, pour_points_da,
                               boundaries, flow_bdry,
                               chunks_y, chunks_x, n_tile_y, n_tile_x)


def _assemble_watershed(flow_dir_da, pour_points_da,
                        boundaries, flow_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Build lazy dask array by re-running tiles with converged exit labels."""

    def _tile_fn(flow_dir_block, pp_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(flow_dir_block.shape, np.nan, dtype=np.float64)
        iy, ix = block_info[0]['chunk-location']
        h, w = flow_dir_block.shape
        exits = _compute_exit_labels(
            iy, ix, boundaries, flow_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        result = _watershed_tile_kernel(
            np.asarray(flow_dir_block, dtype=np.float64),
            h, w,
            np.asarray(pp_block, dtype=np.float64),
            *exits)
        return result

    return da.map_blocks(
        _tile_fn,
        flow_dir_da, pour_points_da,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )




def _watershed_tile_cupy(flow_dir_data, pour_points_data,
                          exit_top, exit_bottom, exit_left, exit_right,
                          exit_tl, exit_tr, exit_bl, exit_br):
    """GPU seeded watershed for a single tile.

    Uses GPU label propagation with exit labels injected at boundary
    cells before iteration.
    """
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)
    pp_f64 = pour_points_data.astype(cp.float64)

    labels = cp.zeros((H, W), dtype=cp.float64)
    state = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _init_watershed_gpu[griddim, blockdim](
        flow_dir_f64, pp_f64, labels, state, H, W)

    # Inject exit labels at boundary cells where active (state==1)
    # and exit label is resolved (not NaN, >= 0).
    exit_top_cp = cp.asarray(exit_top)
    m = (state[0, :] == 1) & ~cp.isnan(exit_top_cp)
    labels[0, :] = cp.where(m, exit_top_cp, labels[0, :])
    state[0, :] = cp.where(m, 2, state[0, :])

    exit_bot_cp = cp.asarray(exit_bottom)
    m = (state[H - 1, :] == 1) & ~cp.isnan(exit_bot_cp)
    labels[H - 1, :] = cp.where(m, exit_bot_cp, labels[H - 1, :])
    state[H - 1, :] = cp.where(m, 2, state[H - 1, :])

    exit_left_cp = cp.asarray(exit_left)
    m = (state[:, 0] == 1) & ~cp.isnan(exit_left_cp)
    labels[:, 0] = cp.where(m, exit_left_cp, labels[:, 0])
    state[:, 0] = cp.where(m, 2, state[:, 0])

    exit_right_cp = cp.asarray(exit_right)
    m = (state[:, W - 1] == 1) & ~cp.isnan(exit_right_cp)
    labels[:, W - 1] = cp.where(m, exit_right_cp, labels[:, W - 1])
    state[:, W - 1] = cp.where(m, 2, state[:, W - 1])

    # Corner exit labels
    for r, c, val in [(0, 0, exit_tl), (0, W - 1, exit_tr),
                       (H - 1, 0, exit_bl), (H - 1, W - 1, exit_br)]:
        if val == val and int(state[r, c]) == 1:
            labels[r, c] = val
            state[r, c] = 2

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _propagate_labels_gpu[griddim, blockdim](
            flow_dir_f64, labels, state, changed, H, W)
        if int(changed[0]) == 0:
            break
        _advance_frontier_gpu[griddim, blockdim](state, H, W)

    labels = cp.where((state == 1) | (state == 0), cp.nan, labels)
    return labels


def _process_tile_watershed_cupy(iy, ix, flow_dir_da, pour_points_da,
                                  boundaries, flow_bdry,
                                  chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded GPU watershed on one tile; update boundaries."""
    import cupy as cp

    chunk = cp.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=cp.float64)
    pp_chunk = cp.asarray(
        pour_points_da.blocks[iy, ix].compute(), dtype=cp.float64)

    exits = _compute_exit_labels(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    result = _watershed_tile_cupy(chunk, pp_chunk, *exits)

    new_top = result[0, :].get().copy()
    new_bottom = result[-1, :].get().copy()
    new_left = result[:, 0].get().copy()
    new_right = result[:, -1].get().copy()

    changed = False
    for side, new in (('top', new_top), ('bottom', new_bottom),
                      ('left', new_left), ('right', new_right)):
        old = boundaries.get(side, iy, ix).copy()
        with np.errstate(invalid='ignore'):
            mask = ~(np.isnan(old) & np.isnan(new))
            if mask.any():
                diff = old[mask] != new[mask]
                if np.any(diff):
                    changed = True
                    break

    boundaries.set('top', iy, ix, new_top)
    boundaries.set('bottom', iy, ix, new_bottom)
    boundaries.set('left', iy, ix, new_left)
    boundaries.set('right', iy, ix, new_right)

    return changed


def _assemble_watershed_cupy(flow_dir_da, pour_points_da,
                              boundaries, flow_bdry,
                              chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Build lazy dask+cupy array using GPU watershed tile kernel."""
    import cupy as cp

    def _tile_fn(flow_dir_block, pp_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return cp.full(flow_dir_block.shape, cp.nan, dtype=cp.float64)
        iy, ix = block_info[0]['chunk-location']
        exits = _compute_exit_labels(
            iy, ix, boundaries, flow_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _watershed_tile_cupy(
            cp.asarray(flow_dir_block, dtype=cp.float64),
            cp.asarray(pp_block, dtype=cp.float64),
            *exits)

    return da.map_blocks(
        _tile_fn,
        flow_dir_da, pour_points_da,
        dtype=np.float64,
        meta=cp.array((), dtype=cp.float64),
    )


def _watershed_dask_cupy(flow_dir_da, pour_points_da):
    """Dask+CuPy watershed: native GPU processing per tile."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry = _preprocess_tiles(flow_dir_da, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()

    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=np.nan)

    max_iterations = max(n_tile_y, n_tile_x) * 2 + 10

    for _iteration in range(max_iterations):
        any_changed = False

        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_tile_watershed_cupy(
                    iy, ix, flow_dir_da, pour_points_da,
                    boundaries, flow_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile_watershed_cupy(
                    iy, ix, flow_dir_da, pour_points_da,
                    boundaries, flow_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True

        if not any_changed:
            break

    boundaries = boundaries.snapshot()

    return _assemble_watershed_cupy(flow_dir_da, pour_points_da,
                                     boundaries, flow_bdry,
                                     chunks_y, chunks_x, n_tile_y, n_tile_x)




# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def watershed_d8(flow_dir: xr.DataArray,
              pour_points: xr.DataArray,
              name: str = 'watershed') -> xr.DataArray:
    """Label each cell with the pour point it drains to.

    Follows each cell downstream through the D8 flow direction grid
    until it reaches a pour point.  The cell is then labeled with that
    pour point's value.  Cells that do not reach any pour point are
    assigned NaN.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2D D8 flow direction grid
        (codes 0/1/2/4/8/16/32/64/128; NaN for nodata).
    pour_points : xarray.DataArray
        2D raster where non-NaN cells are pour points and their
        values become the labels.  Must have the same shape as
        ``flow_dir``.
    name : str, default='watershed'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array where each cell = label of its pour point.
        NaN for nodata or cells not reaching any pour point.
    """
    _validate_raster(flow_dir, func_name='watershed', name='flow_dir')
    _validate_raster(pour_points, func_name='watershed', name='pour_points')

    data = flow_dir.data
    pp_data = pour_points.data

    if isinstance(data, np.ndarray):
        _check_memory(*data.shape)
        fd = data.astype(np.float64)
        pp = np.asarray(pp_data, dtype=np.float64)
        h, w = fd.shape
        # Init labels and state: pour points → resolved (state 3),
        # NaN flow_dir → nodata (state 0), others → unresolved (state 1)
        labels = np.full((h, w), np.nan, dtype=np.float64)
        state = np.zeros((h, w), dtype=np.int8)
        for r in range(h):
            for c in range(w):
                if fd[r, c] != fd[r, c]:  # NaN
                    pass  # state 0, label NaN
                elif pp[r, c] == pp[r, c]:  # not NaN → pour point
                    labels[r, c] = pp[r, c]
                    state[r, c] = 3
                else:
                    state[r, c] = 1  # unresolved
        out = _watershed_cpu(fd, labels, state, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        _check_gpu_memory(*data.shape)
        out = _watershed_cupy(data, pp_data)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _watershed_dask_cupy(data, pp_data)

    elif da is not None and isinstance(data, da.Array):
        out = _watershed_dask_iterative(data, pp_data)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)


def basins_d8(flow_dir, name='basins'):
    """Backward-compatible wrapper; use :func:`basin` instead."""
    from .basin_d8 import basin_d8
    return basin_d8(flow_dir, name=name)
