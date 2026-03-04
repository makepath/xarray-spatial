"""Flow accumulation for Multiple Flow Direction (MFD) grids.

Takes the (8, H, W) fractional flow direction output from
``flow_direction_mfd`` and accumulates upstream area through all
downslope paths simultaneously.

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
from xrspatial._boundary_store import BoundaryStore
from xrspatial.dataset_support import supports_dataset

# Neighbor offsets: E, SE, S, SW, W, NW, N, NE
_DY = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

# Opposite neighbor index (who points back at me?)
# E(0)->W(4), SE(1)->NW(5), S(2)->N(6), SW(3)->NE(7), ...
_OPPOSITE = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _flow_accum_mfd_cpu(fractions, height, width):
    """Kahn's BFS topological sort for MFD flow accumulation.

    Parameters
    ----------
    fractions : (8, H, W) float64 array of flow fractions
    height, width : int

    Returns
    -------
    accum : (H, W) float64 array
    """
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    accum = np.empty((height, width), dtype=np.float64)
    in_degree = np.zeros((height, width), dtype=np.int32)
    valid = np.zeros((height, width), dtype=np.int8)

    # Pass 1: initialise
    for r in range(height):
        for c in range(width):
            v = fractions[0, r, c]
            if v != v:  # NaN
                accum[r, c] = np.nan
            else:
                valid[r, c] = 1
                accum[r, c] = 1.0

    # Pass 2: compute in-degrees
    for r in range(height):
        for c in range(width):
            if valid[r, c] == 0:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
                    if 0 <= nr < height and 0 <= nc < width:
                        if valid[nr, nc] == 1:
                            in_degree[nr, nc] += 1

    # BFS queue
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

        for k in range(8):
            frac = fractions[k, r, c]
            if frac > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if 0 <= nr < height and 0 <= nc < width:
                    if valid[nr, nc] == 1:
                        accum[nr, nc] += accum[r, c] * frac
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
def _init_accum_indegree_mfd(fractions, accum, in_degree, state, H, W):
    """Initialise accum, in_degree and state for MFD on GPU."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    v = fractions[0, i, j]
    if v != v:  # NaN
        state[i, j] = 0
        accum[i, j] = 0.0
        return

    state[i, j] = 1
    accum[i, j] = 1.0

    # Neighbor offsets: E, SE, S, SW, W, NW, N, NE
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
        if 0 <= ni < H and 0 <= nj < W:
            cuda.atomic.add(in_degree, (ni, nj), 1)


@cuda.jit
def _find_ready_and_finalize_mfd(in_degree, state, changed, H, W):
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
def _pull_from_frontier_mfd(fractions, accum, in_degree, state, H, W):
    """Active MFD cells pull accumulation from frontier neighbours."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if state[i, j] != 1:
        return

    # Opposite direction index: if neighbor k sent flow in direction k,
    # I am the opposite direction from them.
    # E(0)->W(4), SE(1)->NW(5), S(2)->N(6), SW(3)->NE(7), etc.
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
            accum[i, j] += accum[ni, nj] * frac
            in_degree[i, j] -= 1


def _flow_accum_mfd_cupy(fractions_data):
    """GPU driver: iterative frontier peeling for MFD."""
    import cupy as cp

    _, H, W = fractions_data.shape
    fractions_f64 = fractions_data.astype(cp.float64)

    accum = cp.zeros((H, W), dtype=cp.float64)
    in_degree = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _init_accum_indegree_mfd[griddim, blockdim](
        fractions_f64, accum, in_degree, state, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _find_ready_and_finalize_mfd[griddim, blockdim](
            in_degree, state, changed, H, W)

        if int(changed[0]) == 0:
            break

        _pull_from_frontier_mfd[griddim, blockdim](
            fractions_f64, accum, in_degree, state, H, W)

    accum = cp.where(state == 0, cp.nan, accum)
    return accum


# =====================================================================
# Dask tile kernel
# =====================================================================

@ngjit
def _flow_accum_mfd_tile_kernel(fractions, h, w,
                                 seed_top, seed_bottom,
                                 seed_left, seed_right,
                                 seed_tl, seed_tr, seed_bl, seed_br):
    """Seeded BFS MFD flow accumulation for a single tile.

    Parameters
    ----------
    fractions : (8, h, w) float64 -- MFD flow fractions for this tile
    """
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    accum = np.empty((h, w), dtype=np.float64)
    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)

    # Initialise
    for r in range(h):
        for c in range(w):
            v = fractions[0, r, c]
            if v == v:  # not NaN
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

    # Compute in-degrees
    for r in range(h):
        for c in range(w):
            if valid[r, c] == 0:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
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

        for k in range(8):
            frac = fractions[k, r, c]
            if frac > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1:
                    accum[nr, nc] += accum[r, c] * frac
                    in_degree[nr, nc] -= 1
                    if in_degree[nr, nc] == 0:
                        queue_r[tail] = nr
                        queue_c[tail] = nc
                        tail += 1

    return accum


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _preprocess_mfd_tiles(fractions_da, chunks_y, chunks_x):
    """Extract boundary fraction strips into a dict.

    For MFD we need the full 8-band fractions at each boundary cell,
    so we store them as (8, length) arrays.
    """
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    # Store fraction strips keyed by (side, iy, ix)
    frac_bdry = {}

    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            # fractions_da is (8, H, W) dask array
            # Each tile's fractions: shape (8, tile_h, tile_w)
            chunk = fractions_da[:, sum(chunks_y[:iy]):sum(chunks_y[:iy+1]),
                                    sum(chunks_x[:ix]):sum(chunks_x[:ix+1])].compute()
            chunk = np.asarray(chunk, dtype=np.float64)

            # top row: (8, tile_w)
            frac_bdry[('top', iy, ix)] = chunk[:, 0, :].copy()
            # bottom row: (8, tile_w)
            frac_bdry[('bottom', iy, ix)] = chunk[:, -1, :].copy()
            # left col: (8, tile_h)
            frac_bdry[('left', iy, ix)] = chunk[:, :, 0].copy()
            # right col: (8, tile_h)
            frac_bdry[('right', iy, ix)] = chunk[:, :, -1].copy()

    return frac_bdry


def _compute_seeds_mfd(iy, ix, boundaries, frac_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute seed arrays for tile (iy, ix) from neighbour boundaries.

    For MFD, a neighbor cell flows into the current tile if its fraction
    for the direction pointing into our tile is > 0.
    """
    # Neighbor offsets: E(0), SE(1), S(2), SW(3), W(4), NW(5), N(6), NE(7)
    # Opposite:         W(4), NW(5), N(6), NE(7), E(0), SE(1), S(2), SE(3)

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

    dy_arr = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx_arr = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    # --- Top edge: bottom row of tile above ---
    if iy > 0:
        nb_frac = frac_bdry[('bottom', iy - 1, ix)]  # (8, tile_w)
        nb_accum = boundaries.get('bottom', iy - 1, ix)
        w = nb_frac.shape[1]
        for c in range(w):
            for k in range(8):
                if not (nb_frac[k, c] > 0.0):
                    continue
                # Direction k from neighbor: dy_arr[k], dx_arr[k]
                # Neighbor is in row above, so dy must be +1 to enter our tile
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndy == 1:  # flows south into our tile
                    tc = c + ndx
                    if 0 <= tc < tile_w:
                        seed_top[tc] += nb_accum[c] * nb_frac[k, c]

    # --- Bottom edge: top row of tile below ---
    if iy < n_tile_y - 1:
        nb_frac = frac_bdry[('top', iy + 1, ix)]  # (8, tile_w)
        nb_accum = boundaries.get('top', iy + 1, ix)
        w = nb_frac.shape[1]
        for c in range(w):
            for k in range(8):
                if not (nb_frac[k, c] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndy == -1:  # flows north into our tile
                    tc = c + ndx
                    if 0 <= tc < tile_w:
                        seed_bottom[tc] += nb_accum[c] * nb_frac[k, c]

    # --- Left edge: right column of tile to the left ---
    if ix > 0:
        nb_frac = frac_bdry[('right', iy, ix - 1)]  # (8, tile_h)
        nb_accum = boundaries.get('right', iy, ix - 1)
        h = nb_frac.shape[1]
        for r in range(h):
            for k in range(8):
                if not (nb_frac[k, r] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndx == 1:  # flows east into our tile
                    tr = r + ndy
                    if 0 <= tr < tile_h:
                        seed_left[tr] += nb_accum[r] * nb_frac[k, r]

    # --- Right edge: left column of tile to the right ---
    if ix < n_tile_x - 1:
        nb_frac = frac_bdry[('left', iy, ix + 1)]  # (8, tile_h)
        nb_accum = boundaries.get('left', iy, ix + 1)
        h = nb_frac.shape[1]
        for r in range(h):
            for k in range(8):
                if not (nb_frac[k, r] > 0.0):
                    continue
                ndy = dy_arr[k]
                ndx = dx_arr[k]
                if ndx == -1:  # flows west into our tile
                    tr = r + ndy
                    if 0 <= tr < tile_h:
                        seed_right[tr] += nb_accum[r] * nb_frac[k, r]

    # --- Diagonal corner seeds ---
    # TL: bottom-right cell of (iy-1, ix-1) flows SE (dy=1, dx=1 -> k=1)
    if iy > 0 and ix > 0:
        nb_frac = frac_bdry[('bottom', iy - 1, ix - 1)]  # (8, w)
        av = float(boundaries.get('bottom', iy - 1, ix - 1)[-1])
        frac_se = nb_frac[1, -1]  # SE direction
        if frac_se > 0.0:
            seed_tl += av * frac_se

    # TR: bottom-left cell of (iy-1, ix+1) flows SW (dy=1, dx=-1 -> k=3)
    if iy > 0 and ix < n_tile_x - 1:
        nb_frac = frac_bdry[('bottom', iy - 1, ix + 1)]  # (8, w)
        av = float(boundaries.get('bottom', iy - 1, ix + 1)[0])
        frac_sw = nb_frac[3, 0]  # SW direction
        if frac_sw > 0.0:
            seed_tr += av * frac_sw

    # BL: top-right cell of (iy+1, ix-1) flows NE (dy=-1, dx=1 -> k=7)
    if iy < n_tile_y - 1 and ix > 0:
        nb_frac = frac_bdry[('top', iy + 1, ix - 1)]  # (8, w)
        av = float(boundaries.get('top', iy + 1, ix - 1)[-1])
        frac_ne = nb_frac[7, -1]  # NE direction
        if frac_ne > 0.0:
            seed_bl += av * frac_ne

    # BR: top-left cell of (iy+1, ix+1) flows NW (dy=-1, dx=-1 -> k=5)
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        nb_frac = frac_bdry[('top', iy + 1, ix + 1)]  # (8, w)
        av = float(boundaries.get('top', iy + 1, ix + 1)[0])
        frac_nw = nb_frac[5, 0]  # NW direction
        if frac_nw > 0.0:
            seed_br += av * frac_nw

    return (seed_top, seed_bottom, seed_left, seed_right,
            seed_tl, seed_tr, seed_bl, seed_br)


def _process_tile_mfd(iy, ix, fractions_da, boundaries, frac_bdry,
                       chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded MFD BFS on one tile; update boundaries in-place."""
    # Extract this tile's fractions: (8, tile_h, tile_w)
    y_start = sum(chunks_y[:iy])
    y_end = y_start + chunks_y[iy]
    x_start = sum(chunks_x[:ix])
    x_end = x_start + chunks_x[ix]

    chunk = np.asarray(
        fractions_da[:, y_start:y_end, x_start:x_end].compute(),
        dtype=np.float64)
    _, h, w = chunk.shape

    seeds = _compute_seeds_mfd(
        iy, ix, boundaries, frac_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    accum = _flow_accum_mfd_tile_kernel(chunk, h, w, *seeds)

    # NaN cells don't contribute flow; replace with 0 for boundary storage
    new_top = np.where(np.isnan(accum[0, :]), 0.0, accum[0, :])
    new_bottom = np.where(np.isnan(accum[-1, :]), 0.0, accum[-1, :])
    new_left = np.where(np.isnan(accum[:, 0]), 0.0, accum[:, 0])
    new_right = np.where(np.isnan(accum[:, -1]), 0.0, accum[:, -1])

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


def _flow_accum_mfd_dask_iterative(fractions_da, chunks_y, chunks_x):
    """Iterative boundary-propagation for MFD dask arrays.

    Parameters
    ----------
    fractions_da : dask array of shape (8, H, W)
    chunks_y, chunks_x : tuples of chunk sizes for the spatial dims
    """
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    # Phase 0: extract boundary fraction strips
    frac_bdry = _preprocess_mfd_tiles(fractions_da, chunks_y, chunks_x)

    # Phase 1: initialise boundary accum to 0
    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    # Phase 2: iterative forward/backward sweeps
    max_iterations = max(n_tile_y, n_tile_x) + 10

    for _iteration in range(max_iterations):
        max_change = 0.0

        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_tile_mfd(iy, ix, fractions_da, boundaries,
                                       frac_bdry, chunks_y, chunks_x,
                                       n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile_mfd(iy, ix, fractions_da, boundaries,
                                       frac_bdry, chunks_y, chunks_x,
                                       n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c

        if max_change == 0.0:
            break

    boundaries = boundaries.snapshot()

    return _assemble_result_mfd(fractions_da, boundaries, frac_bdry,
                                 chunks_y, chunks_x, n_tile_y, n_tile_x)


def _assemble_result_mfd(fractions_da, boundaries, frac_bdry,
                          chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Build lazy dask array by re-running each MFD tile with converged seeds.

    fractions_da is (8, H, W).  We build a 2-D dask result by using
    da.block from individually computed tiles.
    """
    rows = []
    for iy in range(n_tile_y):
        row = []
        for ix in range(n_tile_x):
            y_start = sum(chunks_y[:iy])
            y_end = y_start + chunks_y[iy]
            x_start = sum(chunks_x[:ix])
            x_end = x_start + chunks_x[ix]

            chunk = np.asarray(
                fractions_da[:, y_start:y_end, x_start:x_end].compute(),
                dtype=np.float64)
            _, h, w = chunk.shape

            seeds = _compute_seeds_mfd(
                iy, ix, boundaries, frac_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)

            tile_accum = _flow_accum_mfd_tile_kernel(chunk, h, w, *seeds)
            row.append(da.from_array(tile_accum, chunks=tile_accum.shape))
        rows.append(row)

    return da.block(rows)


def _flow_accum_mfd_dask_cupy(fractions_da, chunks_y, chunks_x):
    """Dask+CuPy MFD: convert to numpy, run iterative, convert back."""
    import cupy as cp

    fractions_np = fractions_da.map_blocks(
        lambda b: b.get(), dtype=fractions_da.dtype,
        meta=np.array((), dtype=fractions_da.dtype),
    )
    result = _flow_accum_mfd_dask_iterative(fractions_np, chunks_y, chunks_x)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_accumulation_mfd(flow_dir_mfd: xr.DataArray,
                           name: str = 'flow_accumulation_mfd') -> xr.DataArray:
    """Compute flow accumulation from an MFD flow direction grid.

    Takes the 3-D fractional output of ``flow_direction_mfd`` and
    accumulates upstream contributing area through all downslope
    paths simultaneously.  Each cell starts with a value of 1 (itself)
    and passes fractions of its accumulated value to each downstream
    neighbor.

    Parameters
    ----------
    flow_dir_mfd : xarray.DataArray or xr.Dataset
        3-D MFD flow direction array of shape ``(8, H, W)`` as returned
        by ``flow_direction_mfd``.  Values are flow fractions in
        ``[0, 1]`` that sum to 1.0 at each cell (0.0 at pits/flats,
        NaN at edges or nodata cells).
        Supported backends: NumPy, CuPy, NumPy-backed Dask,
        CuPy-backed Dask.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='flow_accumulation_mfd'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2-D float64 array of flow accumulation values.  Each cell
        holds the total upstream contributing area (including itself)
        that drains through it, weighted by MFD fractions.
        NaN where the input has NaN.

    References
    ----------
    Qin, C., Zhu, A.X., Pei, T., Li, B., Zhou, C., and Yang, L.
    (2007). An adaptive approach to selecting a flow-partition
    exponent for a multiple-flow-direction algorithm. International
    Journal of Geographical Information Science, 21(4), 443-458.

    Quinn, P., Beven, K., Chevallier, P., and Planchon, O. (1991).
    The prediction of hillslope flow paths for distributed
    hydrological modelling using digital terrain models.
    Hydrological Processes, 5(1), 59-79.
    """
    _validate_raster(flow_dir_mfd, func_name='flow_accumulation_mfd',
                     name='flow_dir_mfd', ndim=3)

    data = flow_dir_mfd.data

    if data.ndim != 3 or data.shape[0] != 8:
        raise ValueError(
            "flow_dir_mfd must be a 3-D array of shape (8, H, W), "
            f"got shape {data.shape}"
        )

    if isinstance(data, np.ndarray):
        out = _flow_accum_mfd_cpu(
            data.astype(np.float64), data.shape[1], data.shape[2])
    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _flow_accum_mfd_cupy(data)
    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir_mfd):
        # Spatial chunk sizes from dims 1 and 2
        chunks_y = data.chunks[1]
        chunks_x = data.chunks[2]
        out = _flow_accum_mfd_dask_cupy(data, chunks_y, chunks_x)
    elif da is not None and isinstance(data, da.Array):
        chunks_y = data.chunks[1]
        chunks_x = data.chunks[2]
        out = _flow_accum_mfd_dask_iterative(data, chunks_y, chunks_x)
    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    # Build 2-D output coords (drop 'neighbor' dim)
    spatial_dims = flow_dir_mfd.dims[1:]
    coords = {k: v for k, v in flow_dir_mfd.coords.items()
              if k != 'neighbor' and k not in flow_dir_mfd.dims[:1]}
    # Copy spatial coordinate arrays
    for d in spatial_dims:
        if d in flow_dir_mfd.coords:
            coords[d] = flow_dir_mfd.coords[d]

    return xr.DataArray(out,
                        name=name,
                        coords=coords,
                        dims=spatial_dims,
                        attrs=flow_dir_mfd.attrs)
