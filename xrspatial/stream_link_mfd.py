"""Stream link segmentation for MFD (Multiple Flow Direction) grids.

Assigns a unique positive integer ID to each stream "link" -- the
contiguous segment of stream cells between junctions, headwaters, and
outlets.  This is the MFD counterpart to the D8 ``stream_link`` module.

A **link-start cell** is either a headwater (in-degree 0 among stream
cells) or a junction (in-degree >= 2).  Each link-start cell gets a
position-based ID: ``row * width + col + 1``.  Downstream non-junction
cells inherit their upstream neighbor's link_id via Kahn's BFS.

Because MFD allows a cell to flow to multiple downstream neighbors,
a non-junction downstream cell may receive link_id from more than one
upstream cell.  In that case it uses the first one that arrives in BFS
order.

Algorithm
---------
CPU : Kahn's BFS topological sort among stream cells -- O(N_stream).
GPU : iterative frontier peeling with pull-based kernels.
Dask: iterative tile sweep with boundary propagation, same pattern
      as ``stream_link.py``.
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
from xrspatial.stream_order import _to_numpy_f64


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _stream_link_mfd_cpu(fractions, stream_mask, height, width):
    """Kahn's BFS link ID assignment among stream cells (MFD).

    Parameters
    ----------
    fractions : (8, H, W) float64 -- MFD flow fractions
    stream_mask : (H, W) int8
    height, width : int

    Returns
    -------
    link_id : (H, W) float64
    """
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    link_id = np.empty((height, width), dtype=np.float64)
    in_degree = np.zeros((height, width), dtype=np.int32)

    # Initialise
    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 0:
                link_id[r, c] = np.nan
            else:
                link_id[r, c] = 0.0

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

    # Store original in-degree to identify junctions
    orig_indeg = np.empty((height, width), dtype=np.int32)
    for r in range(height):
        for c in range(width):
            orig_indeg[r, c] = in_degree[r, c]

    # BFS queue
    queue_r = np.empty(height * width, dtype=np.int64)
    queue_c = np.empty(height * width, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    # Enqueue headwaters (stream cells with in_degree == 0)
    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 1 and in_degree[r, c] == 0:
                link_id[r, c] = float(r * width + c + 1)
                queue_r[tail] = r
                queue_c[tail] = c
                tail += 1

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1

        for k in range(8):
            frac = fractions[k, r, c]
            if frac <= 0.0:
                continue
            nr = r + dy[k]
            nc = c + dx[k]
            if not (0 <= nr < height and 0 <= nc < width
                    and stream_mask[nr, nc] == 1):
                continue

            if orig_indeg[nr, nc] >= 2:
                # Junction: decrement in-degree but don't propagate link_id.
                # Junction gets its own position-based ID when ready.
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    link_id[nr, nc] = float(nr * width + nc + 1)
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1
            else:
                # Non-junction: inherit upstream link_id (first arrival wins)
                if link_id[nr, nc] == 0.0:
                    link_id[nr, nc] = link_id[r, c]
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return link_id


# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit
def _stream_link_mfd_init_gpu(fractions, stream_mask, in_degree, state,
                               link_id, H, W):
    """Initialise GPU arrays for MFD stream link computation."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if stream_mask[i, j] == 0:
        state[i, j] = 0
        link_id[i, j] = 0.0
        return

    state[i, j] = 1
    link_id[i, j] = 0.0

    # Compute in-degree from fractions
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
def _stream_link_mfd_save_indeg(in_degree, orig_indeg, stream_mask, H, W):
    """Copy in_degree to orig_indeg after init."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    if stream_mask[i, j] == 1:
        orig_indeg[i, j] = in_degree[i, j]


@cuda.jit
def _stream_link_mfd_find_ready(in_degree, orig_indeg, state, link_id,
                                 changed, H, W):
    """Finalize previous frontier, mark new frontier cells."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if state[i, j] == 2:
        state[i, j] = 3

    if state[i, j] == 1 and in_degree[i, j] == 0:
        state[i, j] = 2
        # Link-start cells get position-based ID
        if orig_indeg[i, j] == 0 or orig_indeg[i, j] >= 2:
            link_id[i, j] = float(i * W + j + 1)
        cuda.atomic.add(changed, 0, 1)


@cuda.jit
def _stream_link_mfd_pull(fractions, stream_mask, in_degree, orig_indeg,
                           state, link_id, H, W):
    """Active cells pull link_id from frontier neighbours (MFD)."""
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

        # Check if the neighbor flows to us via the opposite direction
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
            in_degree[i, j] -= 1
            # Non-junction cells inherit the upstream link_id
            if orig_indeg[i, j] < 2 and link_id[i, j] == 0.0:
                link_id[i, j] = link_id[ni, nj]


def _stream_link_mfd_cupy(fractions_data, stream_mask_data):
    """GPU driver for MFD stream link computation."""
    import cupy as cp

    _, H, W = fractions_data.shape
    fractions_f64 = fractions_data.astype(cp.float64)
    stream_mask_i8 = stream_mask_data.astype(cp.int8)

    in_degree = cp.zeros((H, W), dtype=cp.int32)
    orig_indeg = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    link_id = cp.zeros((H, W), dtype=cp.float64)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _stream_link_mfd_init_gpu[griddim, blockdim](
        fractions_f64, stream_mask_i8, in_degree, state, link_id, H, W)

    # Save orig_indeg after atomic adds complete
    _stream_link_mfd_save_indeg[griddim, blockdim](
        in_degree, orig_indeg, stream_mask_i8, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _stream_link_mfd_find_ready[griddim, blockdim](
            in_degree, orig_indeg, state, link_id, changed, H, W)

        if int(changed[0]) == 0:
            break

        _stream_link_mfd_pull[griddim, blockdim](
            fractions_f64, stream_mask_i8, in_degree, orig_indeg,
            state, link_id, H, W)

    link_id = cp.where(stream_mask_i8 == 0, cp.nan, link_id)
    return link_id


# =====================================================================
# CPU tile kernel for dask
# =====================================================================

@ngjit
def _stream_link_mfd_tile_kernel(fractions, stream_mask, h, w,
                                  seed_id_top, seed_id_bottom,
                                  seed_id_left, seed_id_right,
                                  seed_ext_top, seed_ext_bottom,
                                  seed_ext_left, seed_ext_right,
                                  row_offset, col_offset, total_width):
    """Seeded BFS link assignment for a single MFD tile.

    Parameters
    ----------
    fractions : (8, h, w) float64 -- MFD flow fractions for this tile
    stream_mask : (h, w) int8
    seed_id_*  : link_id values flowing in from neighbouring tiles.
    seed_ext_* : count of external stream cells flowing into each
                 boundary cell (to compute correct orig_indeg).
    row_offset, col_offset : global pixel offsets of this tile.
    total_width : total number of columns in the full raster.
    """
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    link_id = np.empty((h, w), dtype=np.float64)
    in_degree = np.zeros((h, w), dtype=np.int32)

    # Initialise
    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] == 0:
                link_id[r, c] = np.nan
            else:
                link_id[r, c] = 0.0

    # Compute within-tile in-degrees among stream cells
    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] == 0:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
                    if 0 <= nr < h and 0 <= nc < w:
                        if stream_mask[nr, nc] == 1:
                            in_degree[nr, nc] += 1

    # orig_indeg = within-tile + external (for junction detection only).
    # Do NOT add external counts to in_degree -- external inflows are
    # already "delivered" via seed_id values.
    orig_indeg = np.empty((h, w), dtype=np.int32)
    for r in range(h):
        for c in range(w):
            orig_indeg[r, c] = in_degree[r, c]

    for c in range(w):
        if stream_mask[0, c] == 1:
            orig_indeg[0, c] += int(seed_ext_top[c])
        if stream_mask[h - 1, c] == 1:
            orig_indeg[h - 1, c] += int(seed_ext_bottom[c])
    for r in range(h):
        if stream_mask[r, 0] == 1:
            orig_indeg[r, 0] += int(seed_ext_left[r])
        if stream_mask[r, w - 1] == 1:
            orig_indeg[r, w - 1] += int(seed_ext_right[r])

    # BFS queue
    queue_r = np.empty(h * w, dtype=np.int64)
    queue_c = np.empty(h * w, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    # Assign initial link_ids and enqueue cells with in_degree == 0
    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] != 1:
                continue
            if in_degree[r, c] != 0:
                continue
            # Headwater (orig_indeg == 0) or junction (orig_indeg >= 2):
            # get position-based global ID
            if orig_indeg[r, c] == 0 or orig_indeg[r, c] >= 2:
                gr = row_offset + r
                gc = col_offset + c
                link_id[r, c] = float(gr * total_width + gc + 1)
            queue_r[tail] = r
            queue_c[tail] = c
            tail += 1

    # Seed non-junction, non-headwater boundary cells that receive
    # a link_id from their external upstream neighbor.
    # These cells have in_degree > 0 so they are NOT yet in the queue.
    # We apply the seed_id so that when their in_degree drops to 0
    # during BFS, they already have the right inherited link_id.
    for c in range(w):
        if stream_mask[0, c] == 1 and seed_id_top[c] > 0:
            if orig_indeg[0, c] < 2:
                link_id[0, c] = seed_id_top[c]
        if stream_mask[h - 1, c] == 1 and seed_id_bottom[c] > 0:
            if orig_indeg[h - 1, c] < 2:
                link_id[h - 1, c] = seed_id_bottom[c]
    for r in range(h):
        if stream_mask[r, 0] == 1 and seed_id_left[r] > 0:
            if orig_indeg[r, 0] < 2:
                link_id[r, 0] = seed_id_left[r]
        if stream_mask[r, w - 1] == 1 and seed_id_right[r] > 0:
            if orig_indeg[r, w - 1] < 2:
                link_id[r, w - 1] = seed_id_right[r]

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1

        for k in range(8):
            frac = fractions[k, r, c]
            if frac <= 0.0:
                continue
            nr = r + dy[k]
            nc = c + dx[k]
            if not (0 <= nr < h and 0 <= nc < w
                    and stream_mask[nr, nc] == 1):
                continue

            if orig_indeg[nr, nc] >= 2:
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    gnr = row_offset + nr
                    gnc = col_offset + nc
                    link_id[nr, nc] = float(gnr * total_width + gnc + 1)
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1
            else:
                if link_id[nr, nc] == 0.0:
                    link_id[nr, nc] = link_id[r, c]
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return link_id


# =====================================================================
# Dask preprocessing: extract boundary fraction strips and stream masks
# =====================================================================

def _preprocess_mfd_stream_tiles(fractions_da, accum_da, threshold,
                                  chunks_y, chunks_x):
    """Extract boundary fraction strips (8-band) and stream masks.

    Returns
    -------
    frac_bdry : dict mapping (side, iy, ix) -> (8, length) float64 array
    mask_bdry : BoundaryStore with stream mask boundary strips
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

            chunk = np.asarray(
                fractions_da[:, y_start:y_end, x_start:x_end].compute(),
                dtype=np.float64)
            ac_chunk = _to_numpy_f64(
                accum_da[y_start:y_end, x_start:x_end].compute())

            sm = np.where(ac_chunk >= threshold, 1.0, 0.0)
            sm = np.where(np.isnan(ac_chunk), 0.0, sm)
            # Also mark as non-stream if fractions are NaN
            sm = np.where(np.isnan(chunk[0]), 0.0, sm)

            # Store fraction strips: (8, length)
            frac_bdry[('top', iy, ix)] = chunk[:, 0, :].copy()
            frac_bdry[('bottom', iy, ix)] = chunk[:, -1, :].copy()
            frac_bdry[('left', iy, ix)] = chunk[:, :, 0].copy()
            frac_bdry[('right', iy, ix)] = chunk[:, :, -1].copy()

            # Store stream mask strips
            for side, strip in [('top', sm[0, :]),
                                ('bottom', sm[-1, :]),
                                ('left', sm[:, 0]),
                                ('right', sm[:, -1])]:
                mask_bdry.set(side, iy, ix,
                              np.asarray(strip, dtype=np.float64))

    return frac_bdry, mask_bdry


# =====================================================================
# Dask seed computation
# =====================================================================

def _compute_link_seeds_mfd(iy, ix, boundaries, frac_bdry, mask_bdry,
                             chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute seed arrays for tile (iy, ix) from neighbour boundaries.

    Returns (seed_id_top, seed_id_bottom, seed_id_left, seed_id_right,
             seed_ext_top, seed_ext_bottom, seed_ext_left, seed_ext_right).

    seed_id_*  : link_id propagated from the neighbouring tile.
    seed_ext_* : count of external stream cells flowing into each
                 boundary cell (needed to compute correct orig_indeg).
    """
    # Neighbor offsets: E(0), SE(1), S(2), SW(3), W(4), NW(5), N(6), NE(7)
    dy_arr = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx_arr = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    tile_h = chunks_y[iy]
    tile_w = chunks_x[ix]

    seed_id_top = np.zeros(tile_w, dtype=np.float64)
    seed_id_bottom = np.zeros(tile_w, dtype=np.float64)
    seed_id_left = np.zeros(tile_h, dtype=np.float64)
    seed_id_right = np.zeros(tile_h, dtype=np.float64)

    seed_ext_top = np.zeros(tile_w, dtype=np.float64)
    seed_ext_bottom = np.zeros(tile_w, dtype=np.float64)
    seed_ext_left = np.zeros(tile_h, dtype=np.float64)
    seed_ext_right = np.zeros(tile_h, dtype=np.float64)

    # --- Top edge: bottom row of tile above ---
    if iy > 0:
        nb_frac = frac_bdry[('bottom', iy - 1, ix)]  # (8, tile_w)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix)
        nb_link = boundaries.get('bottom', iy - 1, ix)
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
                        seed_ext_top[tc] += 1.0
                        lid = nb_link[c]
                        if lid > 0 and (seed_id_top[tc] == 0
                                        or lid < seed_id_top[tc]):
                            seed_id_top[tc] = lid

    # --- Bottom edge: top row of tile below ---
    if iy < n_tile_y - 1:
        nb_frac = frac_bdry[('top', iy + 1, ix)]  # (8, tile_w)
        nb_mask = mask_bdry.get('top', iy + 1, ix)
        nb_link = boundaries.get('top', iy + 1, ix)
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
                        seed_ext_bottom[tc] += 1.0
                        lid = nb_link[c]
                        if lid > 0 and (seed_id_bottom[tc] == 0
                                        or lid < seed_id_bottom[tc]):
                            seed_id_bottom[tc] = lid

    # --- Left edge: right column of tile to the left ---
    if ix > 0:
        nb_frac = frac_bdry[('right', iy, ix - 1)]  # (8, tile_h)
        nb_mask = mask_bdry.get('right', iy, ix - 1)
        nb_link = boundaries.get('right', iy, ix - 1)
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
                        seed_ext_left[tr] += 1.0
                        lid = nb_link[r]
                        if lid > 0 and (seed_id_left[tr] == 0
                                        or lid < seed_id_left[tr]):
                            seed_id_left[tr] = lid

    # --- Right edge: left column of tile to the right ---
    if ix < n_tile_x - 1:
        nb_frac = frac_bdry[('left', iy, ix + 1)]  # (8, tile_h)
        nb_mask = mask_bdry.get('left', iy, ix + 1)
        nb_link = boundaries.get('left', iy, ix + 1)
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
                        seed_ext_right[tr] += 1.0
                        lid = nb_link[r]
                        if lid > 0 and (seed_id_right[tr] == 0
                                        or lid < seed_id_right[tr]):
                            seed_id_right[tr] = lid

    # --- Diagonal corner seeds ---
    # TL corner: bottom-right cell of (iy-1, ix-1) flows SE (k=1)
    if iy > 0 and ix > 0:
        nb_frac = frac_bdry[('bottom', iy - 1, ix - 1)]  # (8, w)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix - 1)
        nb_link = boundaries.get('bottom', iy - 1, ix - 1)
        if nb_mask[-1] == 1:
            frac_se = nb_frac[1, -1]  # SE direction
            if frac_se > 0.0:
                seed_ext_top[0] += 1.0
                lid = float(nb_link[-1])
                if lid > 0 and (seed_id_top[0] == 0
                                or lid < seed_id_top[0]):
                    seed_id_top[0] = lid

    # TR corner: bottom-left cell of (iy-1, ix+1) flows SW (k=3)
    if iy > 0 and ix < n_tile_x - 1:
        nb_frac = frac_bdry[('bottom', iy - 1, ix + 1)]  # (8, w)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix + 1)
        nb_link = boundaries.get('bottom', iy - 1, ix + 1)
        if nb_mask[0] == 1:
            frac_sw = nb_frac[3, 0]  # SW direction
            if frac_sw > 0.0:
                seed_ext_top[tile_w - 1] += 1.0
                lid = float(nb_link[0])
                if lid > 0 and (seed_id_top[tile_w - 1] == 0
                                or lid < seed_id_top[tile_w - 1]):
                    seed_id_top[tile_w - 1] = lid

    # BL corner: top-right cell of (iy+1, ix-1) flows NE (k=7)
    if iy < n_tile_y - 1 and ix > 0:
        nb_frac = frac_bdry[('top', iy + 1, ix - 1)]  # (8, w)
        nb_mask = mask_bdry.get('top', iy + 1, ix - 1)
        nb_link = boundaries.get('top', iy + 1, ix - 1)
        if nb_mask[-1] == 1:
            frac_ne = nb_frac[7, -1]  # NE direction
            if frac_ne > 0.0:
                seed_ext_bottom[0] += 1.0
                lid = float(nb_link[-1])
                if lid > 0 and (seed_id_bottom[0] == 0
                                or lid < seed_id_bottom[0]):
                    seed_id_bottom[0] = lid

    # BR corner: top-left cell of (iy+1, ix+1) flows NW (k=5)
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        nb_frac = frac_bdry[('top', iy + 1, ix + 1)]  # (8, w)
        nb_mask = mask_bdry.get('top', iy + 1, ix + 1)
        nb_link = boundaries.get('top', iy + 1, ix + 1)
        if nb_mask[0] == 1:
            frac_nw = nb_frac[5, 0]  # NW direction
            if frac_nw > 0.0:
                seed_ext_bottom[tile_w - 1] += 1.0
                lid = float(nb_link[0])
                if lid > 0 and (seed_id_bottom[tile_w - 1] == 0
                                or lid < seed_id_bottom[tile_w - 1]):
                    seed_id_bottom[tile_w - 1] = lid

    return (seed_id_top, seed_id_bottom, seed_id_left, seed_id_right,
            seed_ext_top, seed_ext_bottom, seed_ext_left, seed_ext_right)


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _process_link_tile_mfd(iy, ix, fractions_da, accum_da, threshold,
                            boundaries, frac_bdry, mask_bdry,
                            chunks_y, chunks_x, n_tile_y, n_tile_x,
                            row_offsets, col_offsets, total_width):
    """Run seeded BFS on one MFD tile; update boundary stores."""
    y_start = sum(chunks_y[:iy])
    y_end = y_start + chunks_y[iy]
    x_start = sum(chunks_x[:ix])
    x_end = x_start + chunks_x[ix]

    frac_chunk = np.asarray(
        fractions_da[:, y_start:y_end, x_start:x_end].compute(),
        dtype=np.float64)
    ac_chunk = _to_numpy_f64(
        accum_da[y_start:y_end, x_start:x_end].compute())

    sm = np.where(ac_chunk >= threshold, 1, 0).astype(np.int8)
    sm = np.where(np.isnan(ac_chunk), 0, sm).astype(np.int8)
    sm = np.where(np.isnan(frac_chunk[0]), 0, sm).astype(np.int8)
    _, h, w = frac_chunk.shape

    seeds = _compute_link_seeds_mfd(
        iy, ix, boundaries, frac_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    link = _stream_link_mfd_tile_kernel(
        frac_chunk, sm, h, w, *seeds,
        row_offsets[iy], col_offsets[ix], total_width)

    change = 0.0
    for side, strip in [('top', link[0, :]),
                        ('bottom', link[-1, :]),
                        ('left', link[:, 0]),
                        ('right', link[:, -1])]:
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


def _stream_link_mfd_dask(fractions_da, accum_da, threshold):
    """Iterative boundary-propagation for MFD dask arrays."""
    chunks_y = fractions_da.chunks[1]
    chunks_x = fractions_da.chunks[2]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    total_width = sum(chunks_x)

    # Precompute global pixel offsets for each tile
    row_offsets = np.zeros(n_tile_y, dtype=np.int64)
    col_offsets = np.zeros(n_tile_x, dtype=np.int64)
    if n_tile_y > 1:
        np.cumsum(chunks_y[:-1], out=row_offsets[1:])
    if n_tile_x > 1:
        np.cumsum(chunks_x[:-1], out=col_offsets[1:])

    # Phase 0: extract boundary fraction strips and stream masks
    frac_bdry, mask_bdry = _preprocess_mfd_stream_tiles(
        fractions_da, accum_da, threshold, chunks_y, chunks_x)
    mask_bdry = mask_bdry.snapshot()

    # Phase 1: initialise boundary link_ids
    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    # Phase 2: iterative forward/backward sweeps
    max_iterations = max(n_tile_y, n_tile_x) + 10
    for _ in range(max_iterations):
        max_change = 0.0
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_link_tile_mfd(
                    iy, ix, fractions_da, accum_da, threshold,
                    boundaries, frac_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x,
                    row_offsets, col_offsets, total_width)
                if c > max_change:
                    max_change = c
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_link_tile_mfd(
                    iy, ix, fractions_da, accum_da, threshold,
                    boundaries, frac_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x,
                    row_offsets, col_offsets, total_width)
                if c > max_change:
                    max_change = c
        if max_change == 0.0:
            break

    # Snapshot converged boundaries before assembly
    _boundaries = boundaries.snapshot()
    _frac_bdry = frac_bdry
    _mask_bdry = mask_bdry
    _threshold = threshold

    # Assemble result via da.block
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

            sm = np.where(ac_chunk >= _threshold, 1, 0).astype(np.int8)
            sm = np.where(np.isnan(ac_chunk), 0, sm).astype(np.int8)
            sm = np.where(np.isnan(frac_chunk[0]), 0, sm).astype(np.int8)
            _, h, w = frac_chunk.shape

            seeds = _compute_link_seeds_mfd(
                iy, ix, _boundaries, _frac_bdry, _mask_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)

            tile_link = _stream_link_mfd_tile_kernel(
                frac_chunk, sm, h, w, *seeds,
                row_offsets[iy], col_offsets[ix], total_width)
            row.append(da.from_array(tile_link, chunks=tile_link.shape))
        rows.append(row)

    return da.block(rows)


def _stream_link_mfd_dask_cupy(fractions_da, accum_da, threshold):
    """Dask+CuPy MFD: convert to numpy, run CPU dask path, convert back."""
    import cupy as cp

    fractions_np = fractions_da.map_blocks(
        lambda b: b.get(), dtype=fractions_da.dtype,
        meta=np.array((), dtype=fractions_da.dtype),
    )
    accum_np = accum_da.map_blocks(
        lambda b: b.get(), dtype=accum_da.dtype,
        meta=np.array((), dtype=accum_da.dtype),
    )
    result = _stream_link_mfd_dask(fractions_np, accum_np, threshold)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def stream_link_mfd(fractions: xr.DataArray,
                     flow_accum: xr.DataArray,
                     threshold: float = 100,
                     name: str = 'stream_link_mfd') -> xr.DataArray:
    """Assign unique IDs to stream link segments (MFD).

    Each contiguous stream segment between junctions, headwaters, and
    outlets receives a distinct positive integer ID.  Non-stream cells
    are NaN.

    Parameters
    ----------
    fractions : xarray.DataArray or xr.Dataset
        3-D MFD flow direction array of shape ``(8, H, W)`` as returned
        by ``flow_direction_mfd``.  Values are flow fractions in
        ``[0, 1]`` that sum to 1.0 at each cell (NaN at nodata cells).
        The neighbor order is:
        E(0), SE(1), S(2), SW(3), W(4), NW(5), N(6), NE(7).
    flow_accum : xarray.DataArray
        2-D flow accumulation grid.  Cells with
        ``flow_accum >= threshold`` are stream cells.
    threshold : float, default 100
        Minimum accumulation for stream classification.
    name : str, default 'stream_link_mfd'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2-D float64 grid where each stream link has a unique positive
        integer ID.  Non-stream cells are NaN.
    """
    _validate_raster(fractions, func_name='stream_link_mfd',
                     name='fractions', ndim=3)

    data = fractions.data

    if data.ndim != 3 or data.shape[0] != 8:
        raise ValueError(
            "fractions must be a 3-D array of shape (8, H, W), "
            f"got shape {data.shape}"
        )

    fa_data = flow_accum.data

    if isinstance(data, np.ndarray):
        frac = data.astype(np.float64)
        fa = np.asarray(fa_data, dtype=np.float64)
        stream_mask = np.where(fa >= threshold, 1, 0).astype(np.int8)
        stream_mask = np.where(np.isnan(fa), 0, stream_mask).astype(np.int8)
        stream_mask = np.where(
            np.isnan(frac[0]), 0, stream_mask).astype(np.int8)
        h, w = frac.shape[1], frac.shape[2]
        out = _stream_link_mfd_cpu(frac, stream_mask, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        import cupy as cp
        fa_cp = cp.asarray(fa_data, dtype=cp.float64)
        frac_cp = data.astype(cp.float64)
        stream_mask = cp.where(fa_cp >= threshold, 1, 0).astype(cp.int8)
        stream_mask = cp.where(
            cp.isnan(fa_cp), 0, stream_mask).astype(cp.int8)
        stream_mask = cp.where(
            cp.isnan(frac_cp[0]), 0, stream_mask).astype(cp.int8)
        out = _stream_link_mfd_cupy(frac_cp, stream_mask)

    elif has_cuda_and_cupy() and is_dask_cupy(fractions):
        out = _stream_link_mfd_dask_cupy(data, fa_data, threshold)

    elif da is not None and isinstance(data, da.Array):
        out = _stream_link_mfd_dask(data, fa_data, threshold)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

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
