"""Stream link segmentation: unique IDs for each stream segment.

Assigns a unique positive integer ID to each stream "link" -- the
contiguous segment of stream cells between junctions, headwaters, and
outlets.  The result matches the behavior of ArcGIS's Stream Link tool.

A **link-start cell** is either a headwater (in-degree 0 among stream
cells) or a junction (in-degree >= 2).  Each link-start cell gets a
position-based ID: ``row * width + col + 1``.  Downstream non-junction
cells inherit their upstream neighbor's link_id via Kahn's BFS.

Algorithm
---------
CPU : Kahn's BFS topological sort among stream cells -- O(N_stream).
GPU : iterative frontier peeling with pull-based kernels.
Dask: iterative tile sweep with boundary propagation, same pattern
      as ``stream_order.py``.
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
from xrspatial.flow_accumulation import _code_to_offset, _code_to_offset_py
from xrspatial.stream_order import _preprocess_stream_tiles


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _stream_link_cpu(flow_dir, stream_mask, height, width):
    """Kahn's BFS link ID assignment among stream cells."""
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

        v = flow_dir[r, c]
        if v != v:
            continue
        dy, dx = _code_to_offset(v)
        if dy == 0 and dx == 0:
            continue
        nr = r + dy
        nc = c + dx
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
            # Non-junction: inherit upstream link_id
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
def _stream_link_init_gpu(flow_dir, stream_mask, in_degree, orig_indeg,
                           state, link_id, H, W):
    """Initialise GPU arrays for stream link computation."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if stream_mask[i, j] == 0:
        state[i, j] = 0
        link_id[i, j] = 0.0
        orig_indeg[i, j] = 0
        return

    state[i, j] = 1
    link_id[i, j] = 0.0

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
def _stream_link_save_indeg(in_degree, orig_indeg, stream_mask, H, W):
    """Copy in_degree to orig_indeg after init."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    if stream_mask[i, j] == 1:
        orig_indeg[i, j] = in_degree[i, j]


@cuda.jit
def _stream_link_find_ready(in_degree, orig_indeg, state, link_id,
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
def _stream_link_pull(flow_dir, stream_mask, in_degree, orig_indeg,
                       state, link_id, H, W):
    """Active cells pull link_id from frontier neighbours."""
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
            in_degree[i, j] -= 1
            # Non-junction cells inherit the upstream link_id
            if orig_indeg[i, j] < 2:
                link_id[i, j] = link_id[ni, nj]


def _stream_link_cupy(flow_dir_data, stream_mask_data):
    """GPU driver for stream link computation."""
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)
    stream_mask_i8 = stream_mask_data.astype(cp.int8)

    in_degree = cp.zeros((H, W), dtype=cp.int32)
    orig_indeg = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    link_id = cp.zeros((H, W), dtype=cp.float64)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _stream_link_init_gpu[griddim, blockdim](
        flow_dir_f64, stream_mask_i8, in_degree, orig_indeg,
        state, link_id, H, W)

    # Save orig_indeg after atomic adds complete
    _stream_link_save_indeg[griddim, blockdim](
        in_degree, orig_indeg, stream_mask_i8, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _stream_link_find_ready[griddim, blockdim](
            in_degree, orig_indeg, state, link_id, changed, H, W)

        if int(changed[0]) == 0:
            break

        _stream_link_pull[griddim, blockdim](
            flow_dir_f64, stream_mask_i8, in_degree, orig_indeg,
            state, link_id, H, W)

    link_id = cp.where(stream_mask_i8 == 0, cp.nan, link_id)
    return link_id


# =====================================================================
# CPU tile kernel for dask
# =====================================================================

@ngjit
def _stream_link_tile_kernel(flow_dir, stream_mask, h, w,
                              seed_id_top, seed_id_bottom,
                              seed_id_left, seed_id_right,
                              seed_ext_top, seed_ext_bottom,
                              seed_ext_left, seed_ext_right,
                              row_offset, col_offset, total_width):
    """Seeded BFS link assignment for a single tile.

    Parameters
    ----------
    seed_id_*  : link_id values flowing in from neighbouring tiles.
    seed_ext_* : count of external stream cells flowing into each
                 boundary cell (to compute correct orig_indeg).
    row_offset, col_offset : global pixel offsets of this tile.
    total_width : total number of columns in the full raster.
    """
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
            link_id[nr, nc] = link_id[r, c]
            in_degree[nr, nc] -= 1
            if in_degree[nr, nc] == 0:
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1

    return link_id


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _compute_link_seeds(iy, ix, boundaries, indeg_bdry,
                        flow_bdry, mask_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute seed arrays for tile (iy, ix) from neighbour boundaries.

    Returns (seed_id_top, seed_id_bottom, seed_id_left, seed_id_right,
             seed_ext_top, seed_ext_bottom, seed_ext_left, seed_ext_right).

    seed_id_*  : link_id propagated from the neighbouring tile.
    seed_ext_* : count of external stream neighbours flowing into each
                 boundary cell (needed to compute correct orig_indeg).
    """
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

    # Top edge: bottom of tile above
    if iy > 0:
        nb_fdir = flow_bdry.get('bottom', iy - 1, ix)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix)
        nb_link = boundaries.get('bottom', iy - 1, ix)
        w = len(nb_fdir)
        for j in range(w):
            if nb_mask[j] == 0:
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
                seed_ext_top[target] += 1.0
                lid = nb_link[j]
                if lid > 0 and (seed_id_top[target] == 0
                                or lid < seed_id_top[target]):
                    seed_id_top[target] = lid

    # Bottom edge: top of tile below
    if iy < n_tile_y - 1:
        nb_fdir = flow_bdry.get('top', iy + 1, ix)
        nb_mask = mask_bdry.get('top', iy + 1, ix)
        nb_link = boundaries.get('top', iy + 1, ix)
        w = len(nb_fdir)
        for j in range(w):
            if nb_mask[j] == 0:
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
                seed_ext_bottom[target] += 1.0
                lid = nb_link[j]
                if lid > 0 and (seed_id_bottom[target] == 0
                                or lid < seed_id_bottom[target]):
                    seed_id_bottom[target] = lid

    # Left edge: right column of tile to the left
    if ix > 0:
        nb_fdir = flow_bdry.get('right', iy, ix - 1)
        nb_mask = mask_bdry.get('right', iy, ix - 1)
        nb_link = boundaries.get('right', iy, ix - 1)
        h = len(nb_fdir)
        for i in range(h):
            if nb_mask[i] == 0:
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
                seed_ext_left[target] += 1.0
                lid = nb_link[i]
                if lid > 0 and (seed_id_left[target] == 0
                                or lid < seed_id_left[target]):
                    seed_id_left[target] = lid

    # Right edge: left column of tile to the right
    if ix < n_tile_x - 1:
        nb_fdir = flow_bdry.get('left', iy, ix + 1)
        nb_mask = mask_bdry.get('left', iy, ix + 1)
        nb_link = boundaries.get('left', iy, ix + 1)
        h = len(nb_fdir)
        for i in range(h):
            if nb_mask[i] == 0:
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
                seed_ext_right[target] += 1.0
                lid = nb_link[i]
                if lid > 0 and (seed_id_right[target] == 0
                                or lid < seed_id_right[target]):
                    seed_id_right[target] = lid

    # Corner seeds
    if iy > 0 and ix > 0:
        fdir = flow_bdry.get('bottom', iy - 1, ix - 1)[-1]
        mask = mask_bdry.get('bottom', iy - 1, ix - 1)[-1]
        if mask == 1 and int(fdir) == 2:  # SE
            seed_ext_top[0] += 1.0
            lid = float(boundaries.get('bottom', iy - 1, ix - 1)[-1])
            if lid > 0 and (seed_id_top[0] == 0 or lid < seed_id_top[0]):
                seed_id_top[0] = lid
    if iy > 0 and ix < n_tile_x - 1:
        fdir = flow_bdry.get('bottom', iy - 1, ix + 1)[0]
        mask = mask_bdry.get('bottom', iy - 1, ix + 1)[0]
        if mask == 1 and int(fdir) == 8:  # SW
            seed_ext_top[tile_w - 1] += 1.0
            lid = float(boundaries.get('bottom', iy - 1, ix + 1)[0])
            if lid > 0 and (seed_id_top[tile_w - 1] == 0
                            or lid < seed_id_top[tile_w - 1]):
                seed_id_top[tile_w - 1] = lid
    if iy < n_tile_y - 1 and ix > 0:
        fdir = flow_bdry.get('top', iy + 1, ix - 1)[-1]
        mask = mask_bdry.get('top', iy + 1, ix - 1)[-1]
        if mask == 1 and int(fdir) == 128:  # NE
            seed_ext_bottom[0] += 1.0
            lid = float(boundaries.get('top', iy + 1, ix - 1)[-1])
            if lid > 0 and (seed_id_bottom[0] == 0
                            or lid < seed_id_bottom[0]):
                seed_id_bottom[0] = lid
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        fdir = flow_bdry.get('top', iy + 1, ix + 1)[0]
        mask = mask_bdry.get('top', iy + 1, ix + 1)[0]
        if mask == 1 and int(fdir) == 32:  # NW
            seed_ext_bottom[tile_w - 1] += 1.0
            lid = float(boundaries.get('top', iy + 1, ix + 1)[0])
            if lid > 0 and (seed_id_bottom[tile_w - 1] == 0
                            or lid < seed_id_bottom[tile_w - 1]):
                seed_id_bottom[tile_w - 1] = lid

    return (seed_id_top, seed_id_bottom, seed_id_left, seed_id_right,
            seed_ext_top, seed_ext_bottom, seed_ext_left, seed_ext_right)


def _process_link_tile(iy, ix, flow_dir_da, accum_da, threshold,
                       boundaries, indeg_bdry, flow_bdry, mask_bdry,
                       chunks_y, chunks_x, n_tile_y, n_tile_x,
                       row_offsets, col_offsets, total_width):
    """Run seeded BFS on one tile; update boundary stores."""
    fd_chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    ac_chunk = np.asarray(
        accum_da.blocks[iy, ix].compute(), dtype=np.float64)
    sm = np.where(ac_chunk >= threshold, 1, 0).astype(np.int8)
    sm = np.where(np.isnan(ac_chunk), 0, sm).astype(np.int8)
    sm = np.where(np.isnan(fd_chunk), 0, sm).astype(np.int8)
    h, w = fd_chunk.shape

    seeds = _compute_link_seeds(
        iy, ix, boundaries, indeg_bdry, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    link = _stream_link_tile_kernel(
        fd_chunk, sm, h, w, *seeds,
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


def _stream_link_dask(flow_dir_da, accum_da, threshold):
    """Iterative boundary-propagation for dask arrays."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    total_width = sum(chunks_x)

    # Precompute global pixel offsets for each tile
    row_offsets = np.zeros(n_tile_y, dtype=np.int64)
    col_offsets = np.zeros(n_tile_x, dtype=np.int64)
    np.cumsum(chunks_y[:-1], out=row_offsets[1:]) if n_tile_y > 1 else None
    np.cumsum(chunks_x[:-1], out=col_offsets[1:]) if n_tile_x > 1 else None

    # Phase 0: extract boundary flow dirs and stream masks
    flow_bdry, mask_bdry = _preprocess_stream_tiles(
        flow_dir_da, accum_da, threshold, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()
    mask_bdry = mask_bdry.snapshot()

    # Phase 1: initialise boundary link_ids and in-degree info
    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)
    indeg_bdry = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    # Phase 2: iterative forward/backward sweeps
    max_iterations = max(n_tile_y, n_tile_x) + 10
    for _ in range(max_iterations):
        max_change = 0.0
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_link_tile(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    boundaries, indeg_bdry, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x,
                    row_offsets, col_offsets, total_width)
                if c > max_change:
                    max_change = c
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_link_tile(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    boundaries, indeg_bdry, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x,
                    row_offsets, col_offsets, total_width)
                if c > max_change:
                    max_change = c
        if max_change == 0.0:
            break

    # Snapshot converged boundaries before assembly (releases temp files)
    _boundaries = boundaries.snapshot()
    _indeg_bdry = indeg_bdry.snapshot()
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
        seeds = _compute_link_seeds(
            iy, ix, _boundaries, _indeg_bdry, _flow_bdry, _mask_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _stream_link_tile_kernel(
            fd, sm, h, w, *seeds,
            row_offsets[iy], col_offsets[ix], total_width)

    return da.map_blocks(
        _tile_fn, flow_dir_da, accum_da,
        dtype=np.float64, meta=np.array((), dtype=np.float64))


def _stream_link_dask_cupy(flow_dir_da, accum_da, threshold):
    """Dask+CuPy: convert to numpy, run CPU dask path, convert back."""
    import cupy as cp

    fd_np = flow_dir_da.map_blocks(
        lambda b: b.get(), dtype=flow_dir_da.dtype,
        meta=np.array((), dtype=flow_dir_da.dtype))
    ac_np = accum_da.map_blocks(
        lambda b: b.get(), dtype=accum_da.dtype,
        meta=np.array((), dtype=accum_da.dtype))

    result = _stream_link_dask(fd_np, ac_np, threshold)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype))


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def stream_link(flow_dir: xr.DataArray,
                flow_accum: xr.DataArray,
                threshold: float = 100,
                name: str = 'stream_link') -> xr.DataArray:
    """Assign unique IDs to stream link segments.

    Each contiguous stream segment between junctions, headwaters, and
    outlets receives a distinct positive integer ID.  Non-stream cells
    are NaN.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2D D8 flow direction grid (codes 0/1/2/4/8/16/32/64/128;
        NaN for nodata).
    flow_accum : xarray.DataArray
        2D flow accumulation grid.  Cells with
        ``flow_accum >= threshold`` are stream cells.
    threshold : float, default 100
        Minimum accumulation for stream classification.
    name : str, default 'stream_link'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 grid where each stream link has a unique positive
        integer ID.  Non-stream cells are NaN.
    """
    _validate_raster(flow_dir, func_name='stream_link', name='flow_dir')

    fd_data = flow_dir.data
    fa_data = flow_accum.data

    if isinstance(fd_data, np.ndarray):
        fd = fd_data.astype(np.float64)
        fa = np.asarray(fa_data, dtype=np.float64)
        stream_mask = np.where(fa >= threshold, 1, 0).astype(np.int8)
        stream_mask = np.where(np.isnan(fa), 0, stream_mask).astype(np.int8)
        stream_mask = np.where(np.isnan(fd), 0, stream_mask).astype(np.int8)
        h, w = fd.shape
        out = _stream_link_cpu(fd, stream_mask, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(fd_data):
        import cupy as cp
        fa_cp = cp.asarray(fa_data, dtype=cp.float64)
        fd_cp = fd_data.astype(cp.float64)
        stream_mask = cp.where(fa_cp >= threshold, 1, 0).astype(cp.int8)
        stream_mask = cp.where(cp.isnan(fa_cp), 0, stream_mask).astype(cp.int8)
        stream_mask = cp.where(cp.isnan(fd_cp), 0, stream_mask).astype(cp.int8)
        out = _stream_link_cupy(fd_cp, stream_mask)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _stream_link_dask_cupy(fd_data, fa_data, threshold)

    elif da is not None and isinstance(fd_data, da.Array):
        out = _stream_link_dask(fd_data, fa_data, threshold)

    else:
        raise TypeError(f"Unsupported array type: {type(fd_data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
