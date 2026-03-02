"""Height Above Nearest Drainage (HAND).

For each cell, follows the D8 flow direction downstream until reaching
a stream cell (flow_accum >= threshold), then computes
HAND = elevation - drain_elevation.

Algorithm
---------
CPU : Kahn's BFS topological sort — O(N), same two-pass structure as
      downstream flow_length but propagating drain_elev instead of
      distance.
GPU : CuPy-via-CPU.
Dask: iterative tile sweep with BoundaryStore exit-label propagation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.flow_accumulation import _code_to_offset
from xrspatial.watershed import _code_to_offset_py
from xrspatial._boundary_store import BoundaryStore
from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _hand_cpu(flow_dir, flow_accum, elevation, H, W, threshold):
    """Compute HAND via Kahn's BFS + reverse propagation of drain_elev.

    Stream cells (flow_accum >= threshold): drain_elev = own elevation.
    Non-stream: drain_elev = drain_elev[downstream_neighbor].
    HAND = elevation - drain_elev.
    """
    in_degree = np.zeros((H, W), dtype=np.int32)
    valid = np.zeros((H, W), dtype=np.int8)
    is_stream = np.zeros((H, W), dtype=np.int8)
    drain_elev = np.empty((H, W), dtype=np.float64)
    hand_out = np.empty((H, W), dtype=np.float64)

    # Init
    for r in range(H):
        for c in range(W):
            v = flow_dir[r, c]
            if v == v:  # not NaN
                valid[r, c] = 1
                fa = flow_accum[r, c]
                if fa == fa and fa >= threshold:
                    is_stream[r, c] = 1
                    drain_elev[r, c] = elevation[r, c]
                else:
                    drain_elev[r, c] = np.nan
            else:
                drain_elev[r, c] = np.nan
                hand_out[r, c] = np.nan

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

    # BFS topological order
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

    # Reverse pass: outlets → divides, propagate drain_elev
    for i in range(tail - 1, -1, -1):
        r = order_r[i]
        c = order_c[i]
        if is_stream[r, c] == 1:
            # Stream cell: drain_elev already set
            continue
        dy, dx = _code_to_offset(flow_dir[r, c])
        if dy == 0 and dx == 0:
            # Pit not on stream: drain to self
            drain_elev[r, c] = elevation[r, c]
            continue
        nr, nc = r + dy, c + dx
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            # Edge exit not on stream: drain to self
            drain_elev[r, c] = elevation[r, c]
            continue
        if valid[nr, nc] == 0:
            drain_elev[r, c] = elevation[r, c]
            continue
        de = drain_elev[nr, nc]
        if de == de:  # not NaN
            drain_elev[r, c] = de
        else:
            drain_elev[r, c] = elevation[r, c]

    # Compute HAND
    for r in range(H):
        for c in range(W):
            if valid[r, c] == 1:
                hand_out[r, c] = elevation[r, c] - drain_elev[r, c]
            else:
                hand_out[r, c] = np.nan

    return hand_out


# =====================================================================
# CuPy backend (via CPU)
# =====================================================================

def _hand_cupy(fd_data, fa_data, elev_data, threshold):
    import cupy as cp
    fd_np = fd_data.get().astype(np.float64)
    fa_np = fa_data.get().astype(np.float64)
    el_np = elev_data.get().astype(np.float64)
    H, W = fd_np.shape
    out = _hand_cpu(fd_np, fa_np, el_np, H, W, threshold)
    return cp.asarray(out)


# =====================================================================
# Dask tile kernel
# =====================================================================

@ngjit
def _hand_tile_kernel(flow_dir, flow_accum, elevation, h, w, threshold,
                       exit_top, exit_bottom, exit_left, exit_right,
                       exit_tl, exit_tr, exit_bl, exit_br):
    """HAND tile kernel with exit-label seeds for drain_elev."""
    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)
    is_stream = np.zeros((h, w), dtype=np.int8)
    drain_elev = np.empty((h, w), dtype=np.float64)
    known = np.zeros((h, w), dtype=np.int8)

    # Init
    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v == v:
                valid[r, c] = 1
                fa = flow_accum[r, c]
                if fa == fa and fa >= threshold:
                    is_stream[r, c] = 1
                    drain_elev[r, c] = elevation[r, c]
                    known[r, c] = 1
                else:
                    drain_elev[r, c] = np.nan
            else:
                drain_elev[r, c] = np.nan

    # Apply exit labels: cells flowing OUT of tile get drain_elev from neighbor
    # Top row
    for c in range(w):
        if valid[0, c] == 1 and known[0, c] == 0:
            dy, dx = _code_to_offset(flow_dir[0, c])
            if 0 + dy < 0:
                el = exit_top[c]
                if el == el:  # not NaN
                    drain_elev[0, c] = el
                    known[0, c] = 1
                else:
                    # Edge of grid exit, drain to self
                    drain_elev[0, c] = elevation[0, c]
                    known[0, c] = 1
    # Bottom row
    for c in range(w):
        if valid[h - 1, c] == 1 and known[h - 1, c] == 0:
            dy, dx = _code_to_offset(flow_dir[h - 1, c])
            if h - 1 + dy >= h:
                el = exit_bottom[c]
                if el == el:
                    drain_elev[h - 1, c] = el
                    known[h - 1, c] = 1
                else:
                    drain_elev[h - 1, c] = elevation[h - 1, c]
                    known[h - 1, c] = 1
    # Left col
    for r in range(h):
        if valid[r, 0] == 1 and known[r, 0] == 0:
            dy, dx = _code_to_offset(flow_dir[r, 0])
            if 0 + dx < 0:
                el = exit_left[r]
                if el == el:
                    drain_elev[r, 0] = el
                    known[r, 0] = 1
                else:
                    drain_elev[r, 0] = elevation[r, 0]
                    known[r, 0] = 1
    # Right col
    for r in range(h):
        if valid[r, w - 1] == 1 and known[r, w - 1] == 0:
            dy, dx = _code_to_offset(flow_dir[r, w - 1])
            if w - 1 + dx >= w:
                el = exit_right[r]
                if el == el:
                    drain_elev[r, w - 1] = el
                    known[r, w - 1] = 1
                else:
                    drain_elev[r, w - 1] = elevation[r, w - 1]
                    known[r, w - 1] = 1

    # Corner overrides
    if valid[0, 0] == 1 and known[0, 0] == 0:
        dy, dx = _code_to_offset(flow_dir[0, 0])
        if 0 + dy < 0 and 0 + dx < 0:
            if exit_tl == exit_tl:
                drain_elev[0, 0] = exit_tl
                known[0, 0] = 1
    if valid[0, w - 1] == 1 and known[0, w - 1] == 0:
        dy, dx = _code_to_offset(flow_dir[0, w - 1])
        if 0 + dy < 0 and w - 1 + dx >= w:
            if exit_tr == exit_tr:
                drain_elev[0, w - 1] = exit_tr
                known[0, w - 1] = 1
    if valid[h - 1, 0] == 1 and known[h - 1, 0] == 0:
        dy, dx = _code_to_offset(flow_dir[h - 1, 0])
        if h - 1 + dy >= h and 0 + dx < 0:
            if exit_bl == exit_bl:
                drain_elev[h - 1, 0] = exit_bl
                known[h - 1, 0] = 1
    if valid[h - 1, w - 1] == 1 and known[h - 1, w - 1] == 0:
        dy, dx = _code_to_offset(flow_dir[h - 1, w - 1])
        if h - 1 + dy >= h and w - 1 + dx >= w:
            if exit_br == exit_br:
                drain_elev[h - 1, w - 1] = exit_br
                known[h - 1, w - 1] = 1

    # In-degrees (only non-known cells)
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

    # Reverse pass: propagate drain_elev
    for i in range(tail - 1, -1, -1):
        r = order_r[i]
        c = order_c[i]
        dy, dx = _code_to_offset(flow_dir[r, c])
        if dy == 0 and dx == 0:
            drain_elev[r, c] = elevation[r, c]
            continue
        nr, nc = r + dy, c + dx
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            # Exits tile with no exit label
            drain_elev[r, c] = elevation[r, c]
            continue
        if valid[nr, nc] == 0:
            drain_elev[r, c] = elevation[r, c]
            continue
        de = drain_elev[nr, nc]
        if de == de:
            drain_elev[r, c] = de
        else:
            drain_elev[r, c] = elevation[r, c]

    # Build output: HAND = elevation - drain_elev
    out = np.empty((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            if valid[r, c] == 1:
                out[r, c] = elevation[r, c] - drain_elev[r, c]
            else:
                out[r, c] = np.nan

    return out


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _preprocess_tiles(flow_dir_da, chunks_y, chunks_x):
    """Extract boundary flow-direction strips."""
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
    """Same exit-label pattern as watershed/flow_length downstream:
    look up drain_elev at the destination cell in the adjacent tile."""
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

    # Top row
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

    # Bottom row
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

    # Left column
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

    # Right column
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

    # Edge-of-grid exits
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


def _process_tile_hand(iy, ix, flow_dir_da, flow_accum_da, elev_da,
                        boundaries, flow_bdry, threshold,
                        chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run HAND tile kernel; update boundary drain_elev values."""
    fd_chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    fa_chunk = np.asarray(
        flow_accum_da.blocks[iy, ix].compute(), dtype=np.float64)
    el_chunk = np.asarray(
        elev_da.blocks[iy, ix].compute(), dtype=np.float64)
    h, w = fd_chunk.shape

    exits = _compute_exit_labels(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    # We need drain_elev, not HAND, at boundaries for propagation.
    # Run the tile kernel to get drain_elev, then extract boundaries.
    # We can't directly get drain_elev from the HAND kernel, so
    # run a modified internal pass.
    drain_elev = _hand_drain_elev_tile(
        fd_chunk, fa_chunk, el_chunk, h, w, threshold, *exits)

    new_top = drain_elev[0, :].copy()
    new_bottom = drain_elev[-1, :].copy()
    new_left = drain_elev[:, 0].copy()
    new_right = drain_elev[:, -1].copy()

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


@ngjit
def _hand_drain_elev_tile(flow_dir, flow_accum, elevation, h, w, threshold,
                            exit_top, exit_bottom, exit_left, exit_right,
                            exit_tl, exit_tr, exit_bl, exit_br):
    """Compute drain_elev for a tile (used for boundary propagation)."""
    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)
    is_stream = np.zeros((h, w), dtype=np.int8)
    drain_elev = np.empty((h, w), dtype=np.float64)
    known = np.zeros((h, w), dtype=np.int8)

    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v == v:
                valid[r, c] = 1
                fa = flow_accum[r, c]
                if fa == fa and fa >= threshold:
                    is_stream[r, c] = 1
                    drain_elev[r, c] = elevation[r, c]
                    known[r, c] = 1
                else:
                    drain_elev[r, c] = np.nan
            else:
                drain_elev[r, c] = np.nan

    # Apply exit labels
    for c in range(w):
        if valid[0, c] == 1 and known[0, c] == 0:
            dy, dx = _code_to_offset(flow_dir[0, c])
            if 0 + dy < 0:
                el = exit_top[c]
                if el == el:
                    drain_elev[0, c] = el
                    known[0, c] = 1
                else:
                    drain_elev[0, c] = elevation[0, c]
                    known[0, c] = 1
    for c in range(w):
        if valid[h - 1, c] == 1 and known[h - 1, c] == 0:
            dy, dx = _code_to_offset(flow_dir[h - 1, c])
            if h - 1 + dy >= h:
                el = exit_bottom[c]
                if el == el:
                    drain_elev[h - 1, c] = el
                    known[h - 1, c] = 1
                else:
                    drain_elev[h - 1, c] = elevation[h - 1, c]
                    known[h - 1, c] = 1
    for r in range(h):
        if valid[r, 0] == 1 and known[r, 0] == 0:
            dy, dx = _code_to_offset(flow_dir[r, 0])
            if 0 + dx < 0:
                el = exit_left[r]
                if el == el:
                    drain_elev[r, 0] = el
                    known[r, 0] = 1
                else:
                    drain_elev[r, 0] = elevation[r, 0]
                    known[r, 0] = 1
    for r in range(h):
        if valid[r, w - 1] == 1 and known[r, w - 1] == 0:
            dy, dx = _code_to_offset(flow_dir[r, w - 1])
            if w - 1 + dx >= w:
                el = exit_right[r]
                if el == el:
                    drain_elev[r, w - 1] = el
                    known[r, w - 1] = 1
                else:
                    drain_elev[r, w - 1] = elevation[r, w - 1]
                    known[r, w - 1] = 1

    # Corners
    if valid[0, 0] == 1 and known[0, 0] == 0:
        dy, dx = _code_to_offset(flow_dir[0, 0])
        if 0 + dy < 0 and 0 + dx < 0:
            if exit_tl == exit_tl:
                drain_elev[0, 0] = exit_tl
                known[0, 0] = 1
    if valid[0, w - 1] == 1 and known[0, w - 1] == 0:
        dy, dx = _code_to_offset(flow_dir[0, w - 1])
        if 0 + dy < 0 and w - 1 + dx >= w:
            if exit_tr == exit_tr:
                drain_elev[0, w - 1] = exit_tr
                known[0, w - 1] = 1
    if valid[h - 1, 0] == 1 and known[h - 1, 0] == 0:
        dy, dx = _code_to_offset(flow_dir[h - 1, 0])
        if h - 1 + dy >= h and 0 + dx < 0:
            if exit_bl == exit_bl:
                drain_elev[h - 1, 0] = exit_bl
                known[h - 1, 0] = 1
    if valid[h - 1, w - 1] == 1 and known[h - 1, w - 1] == 0:
        dy, dx = _code_to_offset(flow_dir[h - 1, w - 1])
        if h - 1 + dy >= h and w - 1 + dx >= w:
            if exit_br == exit_br:
                drain_elev[h - 1, w - 1] = exit_br
                known[h - 1, w - 1] = 1

    # In-degrees
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

    # BFS
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
            drain_elev[r, c] = elevation[r, c]
            continue
        nr, nc = r + dy, c + dx
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            drain_elev[r, c] = elevation[r, c]
            continue
        if valid[nr, nc] == 0:
            drain_elev[r, c] = elevation[r, c]
            continue
        de = drain_elev[nr, nc]
        if de == de:
            drain_elev[r, c] = de
        else:
            drain_elev[r, c] = elevation[r, c]

    return drain_elev


def _hand_dask_iterative(flow_dir_da, flow_accum_da, elev_da, threshold):
    """Iterative boundary propagation for HAND on dask arrays."""
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
                c = _process_tile_hand(
                    iy, ix, flow_dir_da, flow_accum_da, elev_da,
                    boundaries, flow_bdry, threshold,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile_hand(
                    iy, ix, flow_dir_da, flow_accum_da, elev_da,
                    boundaries, flow_bdry, threshold,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True

        if not any_changed:
            break

    boundaries = boundaries.snapshot()

    return _assemble_hand(flow_dir_da, flow_accum_da, elev_da,
                          boundaries, flow_bdry, threshold,
                          chunks_y, chunks_x, n_tile_y, n_tile_x)


def _assemble_hand(flow_dir_da, flow_accum_da, elev_da,
                   boundaries, flow_bdry, threshold,
                   chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Build lazy dask array for HAND with converged boundaries."""

    def _tile_fn(fd_block, fa_block, el_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(fd_block.shape, np.nan, dtype=np.float64)
        iy, ix = block_info[0]['chunk-location']
        h, w = fd_block.shape

        exits = _compute_exit_labels(
            iy, ix, boundaries, flow_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)

        return _hand_tile_kernel(
            np.asarray(fd_block, dtype=np.float64),
            np.asarray(fa_block, dtype=np.float64),
            np.asarray(el_block, dtype=np.float64),
            h, w, threshold, *exits)

    return da.map_blocks(
        _tile_fn,
        flow_dir_da, flow_accum_da, elev_da,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _hand_dask_cupy(flow_dir_da, flow_accum_da, elev_da, threshold):
    """Dask+CuPy: convert to numpy, run CPU iterative path, convert back."""
    import cupy as cp

    fd_np = flow_dir_da.map_blocks(
        lambda b: b.get(), dtype=flow_dir_da.dtype,
        meta=np.array((), dtype=flow_dir_da.dtype),
    )
    fa_np = flow_accum_da.map_blocks(
        lambda b: b.get(), dtype=flow_accum_da.dtype,
        meta=np.array((), dtype=flow_accum_da.dtype),
    )
    el_np = elev_da.map_blocks(
        lambda b: b.get(), dtype=elev_da.dtype,
        meta=np.array((), dtype=elev_da.dtype),
    )
    result = _hand_dask_iterative(fd_np, fa_np, el_np, threshold)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

def hand(flow_dir: xr.DataArray,
         flow_accum: xr.DataArray,
         elevation: xr.DataArray,
         threshold: float = 100,
         name: str = 'hand') -> xr.DataArray:
    """Compute Height Above Nearest Drainage (HAND).

    For each cell, follows the D8 flow direction downstream to the
    nearest stream cell (flow_accum >= threshold), then computes
    HAND = elevation - drain_elevation.

    Parameters
    ----------
    flow_dir : xarray.DataArray
        2D D8 flow direction grid.
    flow_accum : xarray.DataArray
        2D flow accumulation grid.
    elevation : xarray.DataArray
        2D elevation grid.
    threshold : float, default 100
        Minimum flow accumulation to define a stream cell.
    name : str, default 'hand'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 HAND grid. Stream cells have HAND = 0.
        NaN where flow_dir is NaN.
    """
    _validate_raster(flow_dir, func_name='hand', name='flow_dir')
    _validate_raster(flow_accum, func_name='hand', name='flow_accum')
    _validate_raster(elevation, func_name='hand', name='elevation')

    fd_data = flow_dir.data
    fa_data = flow_accum.data
    el_data = elevation.data

    if isinstance(fd_data, np.ndarray):
        fd = fd_data.astype(np.float64)
        fa = np.asarray(fa_data, dtype=np.float64)
        el = np.asarray(el_data, dtype=np.float64)
        H, W = fd.shape
        out = _hand_cpu(fd, fa, el, H, W, float(threshold))

    elif has_cuda_and_cupy() and is_cupy_array(fd_data):
        out = _hand_cupy(fd_data, fa_data, el_data, float(threshold))

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _hand_dask_cupy(fd_data, fa_data, el_data, float(threshold))

    elif da is not None and isinstance(fd_data, da.Array):
        out = _hand_dask_iterative(fd_data, fa_data, el_data, float(threshold))

    else:
        raise TypeError(f"Unsupported array type: {type(fd_data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
