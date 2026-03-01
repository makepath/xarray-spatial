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
from xrspatial._boundary_store import BoundaryStore
from xrspatial.dataset_support import supports_dataset


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
def _watershed_cpu(flow_dir, labels, h, w):
    """Downstream tracing with path compression for watershed.

    ``labels`` is pre-initialised: pour points >= 0, NaN for nodata,
    -1 for unresolved.  On return every reachable cell has the label
    of its pour point, unreachable cells are NaN.
    """
    path_r = np.empty(h * w, dtype=np.int64)
    path_c = np.empty(h * w, dtype=np.int64)

    for r in range(h):
        for c in range(w):
            if labels[r, c] != -1.0:
                continue  # already resolved or NaN

            # Trace downstream, collecting path
            path_len = 0
            cr, cc = r, c
            found_label = np.nan

            while True:
                lbl = labels[cr, cc]
                if lbl >= 0.0:
                    # Hit a labeled cell (pour point or previously resolved)
                    found_label = lbl
                    break
                if lbl != -1.0:
                    # NaN or in-trace marker (-2) → cycle or dead end
                    break

                path_r[path_len] = cr
                path_c[path_len] = cc
                path_len += 1
                labels[cr, cc] = -2.0  # in-trace marker

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
                labels[path_r[i], path_c[i]] = found_label

    return labels


@ngjit
def _basins_init_labels(flow_dir, h, w, total_h, total_w, row_off, col_off):
    """Initialize labels for basins mode.

    Pits (code 0) and cells that exit the **global** grid get unique
    IDs.  Unique ID = (row_off + r) * total_w + (col_off + c) + 1.
    Other valid cells = -1.  NaN cells = NaN.

    The global boundary check (total_h × total_w) ensures that cells
    flowing into an adjacent dask tile are NOT treated as edge-exits.
    """
    labels = np.empty((h, w), dtype=np.float64)

    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v != v:  # NaN
                labels[r, c] = np.nan
                continue

            dy, dx = _code_to_offset(v)
            if dy == 0 and dx == 0:
                # Pit → assign unique ID
                labels[r, c] = float((row_off + r) * total_w +
                                     (col_off + c) + 1)
                continue

            # Check against GLOBAL grid boundaries
            gr = row_off + r + dy
            gc = col_off + c + dx
            if gr < 0 or gr >= total_h or gc < 0 or gc >= total_w:
                # Global edge-exit → assign unique ID
                labels[r, c] = float((row_off + r) * total_w +
                                     (col_off + c) + 1)
                continue

            # Check if flows into NaN within this tile
            nr, nc = r + dy, c + dx
            if 0 <= nr < h and 0 <= nc < w:
                nv = flow_dir[nr, nc]
                if nv != nv:  # flows into NaN
                    labels[r, c] = float((row_off + r) * total_w +
                                         (col_off + c) + 1)
                    continue

            labels[r, c] = -1.0  # unresolved

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
def _init_basins_gpu(flow_dir, labels, state, H, W):
    """Pits/edge-exits → labeled + frontier. NaN → state 0. Others → state 1."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    v = flow_dir[i, j]
    if v != v:  # NaN
        state[i, j] = 0
        labels[i, j] = 0.0
        return

    # Decode direction inline
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

    is_outlet = False
    if dy == 0 and dx == 0:
        is_outlet = True  # pit
    else:
        ni = i + dy
        nj = j + dx
        if ni < 0 or ni >= H or nj < 0 or nj >= W:
            is_outlet = True  # edge-exit
        else:
            nv = flow_dir[ni, nj]
            if nv != nv:  # flows into NaN
                is_outlet = True

    if is_outlet:
        labels[i, j] = float(i * W + j + 1)
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


def _basins_cupy(flow_dir_data):
    """GPU driver for basins."""
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)

    labels = cp.zeros((H, W), dtype=cp.float64)
    state = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _init_basins_gpu[griddim, blockdim](
        flow_dir_f64, labels, state, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _propagate_labels_gpu[griddim, blockdim](
            flow_dir_f64, labels, state, changed, H, W)
        if int(changed[0]) == 0:
            break
        _advance_frontier_gpu[griddim, blockdim](state, H, W)

    # Invalid (state=0) → NaN; unresolved should not exist for basins
    labels = cp.where(state == 0, cp.nan, labels)
    return labels


# =====================================================================
# Tile kernel for dask iterative path
# =====================================================================

@ngjit
def _watershed_tile_kernel(flow_dir, h, w, pour_points,
                           exit_top, exit_bottom, exit_left, exit_right,
                           exit_tl, exit_tr, exit_bl, exit_br):
    """Seeded downstream tracing for a single tile.

    Labels are initialised from pour_points and exit labels (resolved
    labels of the destination cell in adjacent tiles).  Downstream
    tracing with path compression resolves as many cells as possible
    within the tile.
    """
    labels = np.empty((h, w), dtype=np.float64)

    # Initialise labels
    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v != v:  # NaN
                labels[r, c] = np.nan
                continue
            pp = pour_points[r, c]
            if pp == pp:  # not NaN → pour point
                labels[r, c] = pp
                continue
            labels[r, c] = -1.0  # unresolved

    # Apply exit labels to boundary cells that flow OUT of tile
    # Top row: cells flowing north
    for c in range(w):
        if labels[0, c] == -1.0:
            el = exit_top[c]
            if el == el and el >= 0.0:  # not NaN and resolved
                labels[0, c] = el
    # Bottom row
    for c in range(w):
        if labels[h - 1, c] == -1.0:
            el = exit_bottom[c]
            if el == el and el >= 0.0:
                labels[h - 1, c] = el
    # Left column
    for r in range(h):
        if labels[r, 0] == -1.0:
            el = exit_left[r]
            if el == el and el >= 0.0:
                labels[r, 0] = el
    # Right column
    for r in range(h):
        if labels[r, w - 1] == -1.0:
            el = exit_right[r]
            if el == el and el >= 0.0:
                labels[r, w - 1] = el

    # Corners
    if labels[0, 0] == -1.0 and exit_tl == exit_tl and exit_tl >= 0.0:
        labels[0, 0] = exit_tl
    if labels[0, w - 1] == -1.0 and exit_tr == exit_tr and exit_tr >= 0.0:
        labels[0, w - 1] = exit_tr
    if labels[h - 1, 0] == -1.0 and exit_bl == exit_bl and exit_bl >= 0.0:
        labels[h - 1, 0] = exit_bl
    if labels[h - 1, w - 1] == -1.0 and exit_br == exit_br and exit_br >= 0.0:
        labels[h - 1, w - 1] = exit_br

    # Downstream tracing with path compression
    path_r = np.empty(h * w, dtype=np.int64)
    path_c = np.empty(h * w, dtype=np.int64)

    for r in range(h):
        for c in range(w):
            if labels[r, c] != -1.0:
                continue

            path_len = 0
            cr, cc = r, c
            found_label = np.nan

            while True:
                lbl = labels[cr, cc]
                if lbl >= 0.0:
                    found_label = lbl
                    break
                if lbl != -1.0:
                    break

                path_r[path_len] = cr
                path_c[path_len] = cc
                path_len += 1
                labels[cr, cc] = -2.0

                v = flow_dir[cr, cc]
                if v != v:
                    break
                dy, dx = _code_to_offset(v)
                if dy == 0 and dx == 0:
                    break
                nr, nc = cr + dy, cc + dx
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    # Exits tile — leave as unresolved (-1)
                    found_label = -1.0
                    break
                cr, cc = nr, nc

            for i in range(path_len):
                labels[path_r[i], path_c[i]] = found_label

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
            flow_bdry.set('top', iy, ix,
                          np.asarray(chunk[0, :], dtype=np.float64))
            flow_bdry.set('bottom', iy, ix,
                          np.asarray(chunk[-1, :], dtype=np.float64))
            flow_bdry.set('left', iy, ix,
                          np.asarray(chunk[:, 0], dtype=np.float64))
            flow_bdry.set('right', iy, ix,
                          np.asarray(chunk[:, -1], dtype=np.float64))

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
        # After convergence, any remaining unresolved cells → NaN
        result = np.where((result == -1.0) | (result == -2.0),
                          np.nan, result)
        return result

    return da.map_blocks(
        _tile_fn,
        flow_dir_da, pour_points_da,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _basins_dask_iterative(flow_dir_da):
    """Iterative boundary-propagation for basins on dask arrays."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    total_h = sum(chunks_y)
    total_w = sum(chunks_x)

    flow_bdry = _preprocess_tiles(flow_dir_da, chunks_y, chunks_x)
    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=np.nan)

    # Build basins pour_points lazily via map_blocks (never holds the
    # full array in memory).
    def _basins_make_pp_block(flow_dir_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(flow_dir_block.shape, np.nan, dtype=np.float64)
        row_off = block_info[0]['array-location'][0][0]
        col_off = block_info[0]['array-location'][1][0]
        h, w = flow_dir_block.shape
        chunk = np.asarray(flow_dir_block, dtype=np.float64)
        pp = _basins_init_labels(chunk, h, w, total_h, total_w,
                                 row_off, col_off)
        return np.where(pp >= 0, pp, np.nan)

    pour_points_da = da.map_blocks(
        _basins_make_pp_block, flow_dir_da,
        dtype=np.float64, meta=np.array((), dtype=np.float64))

    return _watershed_dask_iterative(flow_dir_da, pour_points_da)


def _watershed_dask_cupy(flow_dir_da, pour_points_da):
    """Dask+CuPy: convert to numpy, run CPU iterative path, convert back."""
    import cupy as cp

    flow_dir_np = flow_dir_da.map_blocks(
        lambda b: b.get(), dtype=flow_dir_da.dtype,
        meta=np.array((), dtype=flow_dir_da.dtype),
    )
    pp_np = pour_points_da.map_blocks(
        lambda b: b.get(), dtype=pour_points_da.dtype,
        meta=np.array((), dtype=pour_points_da.dtype),
    )
    result = _watershed_dask_iterative(flow_dir_np, pp_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


def _basins_dask_cupy(flow_dir_da):
    """Dask+CuPy basins: convert to numpy, run CPU iterative, convert back."""
    import cupy as cp

    flow_dir_np = flow_dir_da.map_blocks(
        lambda b: b.get(), dtype=flow_dir_da.dtype,
        meta=np.array((), dtype=flow_dir_da.dtype),
    )
    result = _basins_dask_iterative(flow_dir_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def watershed(flow_dir: xr.DataArray,
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

    data = flow_dir.data
    pp_data = pour_points.data

    if isinstance(data, np.ndarray):
        fd = data.astype(np.float64)
        pp = np.asarray(pp_data, dtype=np.float64)
        h, w = fd.shape
        # Init labels: pour points get their value, NaN flow_dir → NaN,
        # others → -1
        labels = np.full((h, w), -1.0, dtype=np.float64)
        for r in range(h):
            for c in range(w):
                if fd[r, c] != fd[r, c]:  # NaN
                    labels[r, c] = np.nan
                elif pp[r, c] == pp[r, c]:  # not NaN → pour point
                    labels[r, c] = pp[r, c]
        out = _watershed_cpu(fd, labels, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(data):
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


@supports_dataset
def basins(flow_dir: xr.DataArray,
           name: str = 'basins') -> xr.DataArray:
    """Delineate drainage basins: every cell labeled with its outlet ID.

    Automatically identifies all outlets (pits and edge-exit cells)
    and assigns each a unique ID.  Every valid cell is then labeled
    with the ID of the outlet it drains to.  NaN flow_dir cells
    produce NaN.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2D D8 flow direction grid.
    name : str, default='basins'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array where each cell = unique ID of its outlet.
        NaN for nodata cells.
    """
    _validate_raster(flow_dir, func_name='basins', name='flow_dir')

    data = flow_dir.data

    if isinstance(data, np.ndarray):
        fd = data.astype(np.float64)
        h, w = fd.shape
        labels = _basins_init_labels(fd, h, w, h, w, 0, 0)
        out = _watershed_cpu(fd, labels, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _basins_cupy(data)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _basins_dask_cupy(data)

    elif da is not None and isinstance(data, da.Array):
        out = _basins_dask_iterative(data)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
