"""Stream link segmentation for D-infinity flow direction grids.

Assigns a unique positive integer ID to each stream "link" -- the
contiguous segment of stream cells between junctions, headwaters, and
outlets.  This is the D-inf counterpart to the D8 ``stream_link`` module.

A **link-start cell** is either a headwater (in-degree 0 among stream
cells) or a junction (in-degree >= 2).  Each link-start cell gets a
position-based ID: ``row * width + col + 1``.  Downstream non-junction
cells inherit their upstream neighbor's link_id via Kahn's BFS.

Because D-inf angles can direct flow to two neighbours (the pair
bracketing the continuous angle), a cell may have up to two downstream
neighbours rather than exactly one as in D8.

Algorithm
---------
CPU : Kahn's BFS topological sort among stream cells -- O(N_stream).
GPU : iterative frontier peeling with pull-based kernels.
Dask: iterative tile sweep with boundary propagation, same pattern
      as ``stream_link.py``.
"""

from __future__ import annotations

import math

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
from xrspatial.hydro.stream_order_d8 import _to_numpy_f64
from xrspatial.hydro.stream_order_dinf import (
    _dinf_downstream,
    _NB_DY,
    _NB_DX,
    _preprocess_stream_tiles_dinf,
    _make_stream_mask_np_dinf,
)


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _stream_link_dinf_cpu(angles, stream_mask, height, width):
    """Kahn's BFS link ID assignment among stream cells (D-inf).

    Parameters
    ----------
    angles : (H, W) float64 -- D-inf flow direction angles
    stream_mask : (H, W) int8
    height, width : int

    Returns
    -------
    link_id : (H, W) float64
    """
    nb_dy = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int64)
    nb_dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    pi_over_4 = math.pi / 4.0

    link_id = np.empty((height, width), dtype=np.float64)
    in_degree = np.zeros((height, width), dtype=np.int32)

    # Initialise
    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 0:
                link_id[r, c] = np.nan
            else:
                link_id[r, c] = 0.0

    # Compute in-degrees from D-inf angles
    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 0:
                continue
            theta = angles[r, c]
            if theta != theta:  # NaN
                continue
            if theta < 0.0:  # pit
                continue

            k = int(theta / pi_over_4)
            if k >= 8:
                k = 7
            k2 = (k + 1) % 8
            alpha = theta - k * pi_over_4
            frac1 = (pi_over_4 - alpha) / pi_over_4
            frac2 = alpha / pi_over_4

            if frac1 > 0.0:
                nr = r + nb_dy[k]
                nc = c + nb_dx[k]
                if 0 <= nr < height and 0 <= nc < width:
                    if stream_mask[nr, nc] == 1:
                        in_degree[nr, nc] += 1

            if frac2 > 0.0:
                nr2 = r + nb_dy[k2]
                nc2 = c + nb_dx[k2]
                if 0 <= nr2 < height and 0 <= nc2 < width:
                    if stream_mask[nr2, nc2] == 1:
                        in_degree[nr2, nc2] += 1

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

        theta = angles[r, c]
        if theta != theta:
            continue
        if theta < 0.0:
            continue

        k = int(theta / pi_over_4)
        if k >= 8:
            k = 7
        k2 = (k + 1) % 8
        alpha = theta - k * pi_over_4
        frac1 = (pi_over_4 - alpha) / pi_over_4
        frac2 = alpha / pi_over_4

        # Propagate to neighbor k
        if frac1 > 0.0:
            nr = r + nb_dy[k]
            nc = c + nb_dx[k]
            if 0 <= nr < height and 0 <= nc < width:
                if stream_mask[nr, nc] == 1:
                    if orig_indeg[nr, nc] >= 2:
                        in_degree[nr, nc] -= 1
                        if in_degree[nr, nc] == 0:
                            link_id[nr, nc] = float(nr * width + nc + 1)
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

        # Propagate to neighbor k2
        if frac2 > 0.0:
            nr2 = r + nb_dy[k2]
            nc2 = c + nb_dx[k2]
            if 0 <= nr2 < height and 0 <= nc2 < width:
                if stream_mask[nr2, nc2] == 1:
                    if orig_indeg[nr2, nc2] >= 2:
                        in_degree[nr2, nc2] -= 1
                        if in_degree[nr2, nc2] == 0:
                            link_id[nr2, nc2] = float(
                                nr2 * width + nc2 + 1)
                            queue_r[tail] = nr2
                            queue_c[tail] = nc2
                            tail += 1
                    else:
                        if link_id[nr2, nc2] == 0.0:
                            link_id[nr2, nc2] = link_id[r, c]
                        in_degree[nr2, nc2] -= 1
                        if in_degree[nr2, nc2] == 0:
                            queue_r[tail] = nr2
                            queue_c[tail] = nc2
                            tail += 1

    return link_id


# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit(device=True)
def _dinf_nb_dy(k):
    if k == 0:
        return 0
    elif k == 1:
        return -1
    elif k == 2:
        return -1
    elif k == 3:
        return -1
    elif k == 4:
        return 0
    elif k == 5:
        return 1
    elif k == 6:
        return 1
    else:
        return 1


@cuda.jit(device=True)
def _dinf_nb_dx(k):
    if k == 0:
        return 1
    elif k == 1:
        return 1
    elif k == 2:
        return 0
    elif k == 3:
        return -1
    elif k == 4:
        return -1
    elif k == 5:
        return -1
    elif k == 6:
        return 0
    else:
        return 1


@cuda.jit(device=True)
def _dinf_angle_to_k(theta):
    if theta != theta:  # NaN
        return -1
    if theta < 0.0:
        return -1
    pi_over_4 = 0.7853981633974483
    k = int(theta / pi_over_4)
    if k >= 8:
        k = 7
    return k


@cuda.jit
def _stream_link_dinf_init_gpu(angles, stream_mask, in_degree, state,
                                link_id, H, W):
    """Initialise GPU arrays and compute in-degrees from D-inf angles."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if stream_mask[i, j] == 0:
        state[i, j] = 0
        link_id[i, j] = 0.0
        return

    state[i, j] = 1
    link_id[i, j] = 0.0

    theta = angles[i, j]
    k = _dinf_angle_to_k(theta)
    if k < 0:
        return

    pi_over_4 = 0.7853981633974483
    k2 = (k + 1) % 8
    alpha = theta - k * pi_over_4
    frac1 = (pi_over_4 - alpha) / pi_over_4
    frac2 = alpha / pi_over_4

    if frac1 > 0.0:
        ni = i + _dinf_nb_dy(k)
        nj = j + _dinf_nb_dx(k)
        if 0 <= ni < H and 0 <= nj < W and stream_mask[ni, nj] == 1:
            cuda.atomic.add(in_degree, (ni, nj), 1)

    if frac2 > 0.0:
        ni2 = i + _dinf_nb_dy(k2)
        nj2 = j + _dinf_nb_dx(k2)
        if 0 <= ni2 < H and 0 <= nj2 < W and stream_mask[ni2, nj2] == 1:
            cuda.atomic.add(in_degree, (ni2, nj2), 1)


@cuda.jit
def _stream_link_dinf_save_indeg(in_degree, orig_indeg, stream_mask, H, W):
    """Copy in_degree to orig_indeg after init."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    if stream_mask[i, j] == 1:
        orig_indeg[i, j] = in_degree[i, j]


@cuda.jit
def _stream_link_dinf_find_ready(in_degree, orig_indeg, state, link_id,
                                  changed, H, W):
    """Finalize previous frontier, mark new frontier cells."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if state[i, j] == 2:
        state[i, j] = 3

    if state[i, j] == 1 and in_degree[i, j] == 0:
        state[i, j] = 2
        if orig_indeg[i, j] == 0 or orig_indeg[i, j] >= 2:
            link_id[i, j] = float(i * W + j + 1)
        cuda.atomic.add(changed, 0, 1)


@cuda.jit
def _stream_link_dinf_pull(angles, stream_mask, in_degree, orig_indeg,
                            state, link_id, H, W):
    """Active cells pull link_id from frontier D-inf neighbours."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    if state[i, j] != 1:
        return

    pi_over_4 = 0.7853981633974483

    for nb in range(8):
        dy = _dinf_nb_dy(nb)
        dx = _dinf_nb_dx(nb)
        ni = i + dy
        nj = j + dx
        if ni < 0 or ni >= H or nj < 0 or nj >= W:
            continue
        if state[ni, nj] != 2:
            continue
        if stream_mask[ni, nj] == 0:
            continue

        nb_theta = angles[ni, nj]
        nb_k = _dinf_angle_to_k(nb_theta)
        if nb_k < 0:
            continue

        nb_k2 = (nb_k + 1) % 8
        nb_alpha = nb_theta - nb_k * pi_over_4
        nb_frac1 = (pi_over_4 - nb_alpha) / pi_over_4
        nb_frac2 = nb_alpha / pi_over_4

        flows_to_us = False
        if nb_frac1 > 0.0:
            ti = ni + _dinf_nb_dy(nb_k)
            tj = nj + _dinf_nb_dx(nb_k)
            if ti == i and tj == j:
                flows_to_us = True

        if not flows_to_us and nb_frac2 > 0.0:
            ti2 = ni + _dinf_nb_dy(nb_k2)
            tj2 = nj + _dinf_nb_dx(nb_k2)
            if ti2 == i and tj2 == j:
                flows_to_us = True

        if flows_to_us:
            in_degree[i, j] -= 1
            if orig_indeg[i, j] < 2 and link_id[i, j] == 0.0:
                link_id[i, j] = link_id[ni, nj]


def _stream_link_dinf_cupy(angles_data, stream_mask_data):
    """GPU driver for D-inf stream link computation."""
    import cupy as cp

    H, W = angles_data.shape
    angles_f64 = angles_data.astype(cp.float64)
    stream_mask_i8 = stream_mask_data.astype(cp.int8)

    in_degree = cp.zeros((H, W), dtype=cp.int32)
    orig_indeg = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    link_id = cp.zeros((H, W), dtype=cp.float64)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _stream_link_dinf_init_gpu[griddim, blockdim](
        angles_f64, stream_mask_i8, in_degree, state, link_id, H, W)

    _stream_link_dinf_save_indeg[griddim, blockdim](
        in_degree, orig_indeg, stream_mask_i8, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _stream_link_dinf_find_ready[griddim, blockdim](
            in_degree, orig_indeg, state, link_id, changed, H, W)

        if int(changed[0]) == 0:
            break

        _stream_link_dinf_pull[griddim, blockdim](
            angles_f64, stream_mask_i8, in_degree, orig_indeg,
            state, link_id, H, W)

    link_id = cp.where(stream_mask_i8 == 0, cp.nan, link_id)
    return link_id


# =====================================================================
# GPU tile kernel for dask+cupy
# =====================================================================

@cuda.jit
def _stream_link_dinf_find_ready_tile(in_degree, orig_indeg, state, link_id,
                                       changed, H, W,
                                       row_off, col_off, total_w):
    """Tile-aware find_ready: uses global coordinates for link IDs."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if state[i, j] == 2:
        state[i, j] = 3

    if state[i, j] == 1 and in_degree[i, j] == 0:
        state[i, j] = 2
        if orig_indeg[i, j] == 0 or orig_indeg[i, j] >= 2:
            gi = row_off + i
            gj = col_off + j
            link_id[i, j] = float(gi * total_w + gj + 1)
        cuda.atomic.add(changed, 0, 1)


def _stream_link_dinf_tile_cupy(angles_data, stream_mask_data,
                                 seed_id_top, seed_id_bottom,
                                 seed_id_left, seed_id_right,
                                 seed_ext_top, seed_ext_bottom,
                                 seed_ext_left, seed_ext_right,
                                 row_offset, col_offset, total_width):
    """GPU seeded D-inf stream link for a single tile."""
    import cupy as cp

    H, W = angles_data.shape
    angles_f64 = angles_data.astype(cp.float64)
    stream_mask_i8 = stream_mask_data.astype(cp.int8)

    in_degree = cp.zeros((H, W), dtype=cp.int32)
    orig_indeg = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    link_id = cp.zeros((H, W), dtype=cp.float64)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _stream_link_dinf_init_gpu[griddim, blockdim](
        angles_f64, stream_mask_i8, in_degree, state, link_id, H, W)

    _stream_link_dinf_save_indeg[griddim, blockdim](
        in_degree, orig_indeg, stream_mask_i8, H, W)

    # Add external in-degree counts to orig_indeg (for junction detection)
    for r_idx, ext in [
        ((0, slice(None)), seed_ext_top),
        ((H - 1, slice(None)), seed_ext_bottom),
        ((slice(None), 0), seed_ext_left),
        ((slice(None), W - 1), seed_ext_right),
    ]:
        ext_cp = cp.asarray(ext).astype(cp.int32)
        is_stream = stream_mask_i8[r_idx] == 1
        orig_indeg[r_idx] += cp.where(is_stream, ext_cp, 0)

    # Pre-set link_ids for non-junction boundary cells with seed values
    for r_idx, s_id in [
        ((0, slice(None)), seed_id_top),
        ((H - 1, slice(None)), seed_id_bottom),
        ((slice(None), 0), seed_id_left),
        ((slice(None), W - 1), seed_id_right),
    ]:
        sid_cp = cp.asarray(s_id)
        is_stream = stream_mask_i8[r_idx] == 1
        non_junction = orig_indeg[r_idx] < 2
        has_seed = sid_cp > 0
        mask = is_stream & non_junction & has_seed
        link_id[r_idx] = cp.where(mask, sid_cp, link_id[r_idx])

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _stream_link_dinf_find_ready_tile[griddim, blockdim](
            in_degree, orig_indeg, state, link_id, changed,
            H, W, row_offset, col_offset, total_width)

        if int(changed[0]) == 0:
            break

        _stream_link_dinf_pull[griddim, blockdim](
            angles_f64, stream_mask_i8, in_degree, orig_indeg,
            state, link_id, H, W)

    link_id = cp.where(stream_mask_i8 == 0, cp.nan, link_id)
    return link_id


# =====================================================================
# CPU tile kernel for dask
# =====================================================================

@ngjit
def _stream_link_dinf_tile_kernel(angles, stream_mask, h, w,
                                   seed_id_top, seed_id_bottom,
                                   seed_id_left, seed_id_right,
                                   seed_ext_top, seed_ext_bottom,
                                   seed_ext_left, seed_ext_right,
                                   row_offset, col_offset, total_width):
    """Seeded BFS link assignment for a single D-inf tile.

    Parameters
    ----------
    angles : (h, w) float64 -- D-inf flow direction angles for this tile
    stream_mask : (h, w) int8
    seed_id_*  : link_id values flowing in from neighbouring tiles.
    seed_ext_* : count of external stream cells flowing into each
                 boundary cell (to compute correct orig_indeg).
    row_offset, col_offset : global pixel offsets of this tile.
    total_width : total number of columns in the full raster.
    """
    nb_dy = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int64)
    nb_dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    pi_over_4 = math.pi / 4.0

    link_id = np.empty((h, w), dtype=np.float64)
    in_degree = np.zeros((h, w), dtype=np.int32)

    # Initialise
    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] == 0:
                link_id[r, c] = np.nan
            else:
                link_id[r, c] = 0.0

    # Compute within-tile in-degrees
    for r in range(h):
        for c in range(w):
            if stream_mask[r, c] == 0:
                continue
            theta = angles[r, c]
            if theta != theta:
                continue
            if theta < 0.0:
                continue

            k = int(theta / pi_over_4)
            if k >= 8:
                k = 7
            k2 = (k + 1) % 8
            alpha = theta - k * pi_over_4
            frac1 = (pi_over_4 - alpha) / pi_over_4
            frac2 = alpha / pi_over_4

            if frac1 > 0.0:
                nr = r + nb_dy[k]
                nc = c + nb_dx[k]
                if 0 <= nr < h and 0 <= nc < w:
                    if stream_mask[nr, nc] == 1:
                        in_degree[nr, nc] += 1

            if frac2 > 0.0:
                nr2 = r + nb_dy[k2]
                nc2 = c + nb_dx[k2]
                if 0 <= nr2 < h and 0 <= nc2 < w:
                    if stream_mask[nr2, nc2] == 1:
                        in_degree[nr2, nc2] += 1

    # orig_indeg = within-tile + external (for junction detection only)
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
            if orig_indeg[r, c] == 0 or orig_indeg[r, c] >= 2:
                gr = row_offset + r
                gc = col_offset + c
                link_id[r, c] = float(gr * total_width + gc + 1)
            queue_r[tail] = r
            queue_c[tail] = c
            tail += 1

    # Seed non-junction, non-headwater boundary cells
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

        theta = angles[r, c]
        if theta != theta:
            continue
        if theta < 0.0:
            continue

        k = int(theta / pi_over_4)
        if k >= 8:
            k = 7
        k2 = (k + 1) % 8
        alpha = theta - k * pi_over_4
        frac1 = (pi_over_4 - alpha) / pi_over_4
        frac2 = alpha / pi_over_4

        if frac1 > 0.0:
            nr = r + nb_dy[k]
            nc = c + nb_dx[k]
            if 0 <= nr < h and 0 <= nc < w:
                if stream_mask[nr, nc] == 1:
                    if orig_indeg[nr, nc] >= 2:
                        in_degree[nr, nc] -= 1
                        if in_degree[nr, nc] == 0:
                            gnr = row_offset + nr
                            gnc = col_offset + nc
                            link_id[nr, nc] = float(
                                gnr * total_width + gnc + 1)
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

        if frac2 > 0.0:
            nr2 = r + nb_dy[k2]
            nc2 = c + nb_dx[k2]
            if 0 <= nr2 < h and 0 <= nc2 < w:
                if stream_mask[nr2, nc2] == 1:
                    if orig_indeg[nr2, nc2] >= 2:
                        in_degree[nr2, nc2] -= 1
                        if in_degree[nr2, nc2] == 0:
                            gnr2 = row_offset + nr2
                            gnc2 = col_offset + nc2
                            link_id[nr2, nc2] = float(
                                gnr2 * total_width + gnc2 + 1)
                            queue_r[tail] = nr2
                            queue_c[tail] = nc2
                            tail += 1
                    else:
                        if link_id[nr2, nc2] == 0.0:
                            link_id[nr2, nc2] = link_id[r, c]
                        in_degree[nr2, nc2] -= 1
                        if in_degree[nr2, nc2] == 0:
                            queue_r[tail] = nr2
                            queue_c[tail] = nc2
                            tail += 1

    return link_id


# =====================================================================
# Dask seed computation
# =====================================================================

def _compute_link_seeds_dinf(iy, ix, boundaries, flow_bdry, mask_bdry,
                              chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute seed arrays for tile (iy, ix) from neighbour boundaries.

    Returns (seed_id_top, seed_id_bottom, seed_id_left, seed_id_right,
             seed_ext_top, seed_ext_bottom, seed_ext_left, seed_ext_right).

    seed_id_*  : link_id propagated from the neighbouring tile.
    seed_ext_* : count of external stream cells flowing into each
                 boundary cell (needed to compute correct orig_indeg).
    """
    nb_dy = _NB_DY
    nb_dx = _NB_DX

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

    # Top edge: bottom row of tile above
    if iy > 0:
        nb_fdir = flow_bdry.get('bottom', iy - 1, ix)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix)
        nb_link = boundaries.get('bottom', iy - 1, ix)
        w = len(nb_fdir)
        for j in range(w):
            if nb_mask[j] == 0:
                continue
            theta = nb_fdir[j]
            if theta != theta or theta < 0.0:
                continue
            k1, f1, k2, f2 = _dinf_downstream(theta)
            if k1 < 0:
                continue
            for kk, frac in [(k1, f1), (k2, f2)]:
                if frac <= 0.0:
                    continue
                dy = int(nb_dy[kk])
                dx = int(nb_dx[kk])
                if dy == 1:  # flows south into our tile
                    tc = j + dx
                    if 0 <= tc < tile_w:
                        seed_ext_top[tc] += 1.0
                        lid = nb_link[j]
                        if lid > 0 and (seed_id_top[tc] == 0
                                        or lid < seed_id_top[tc]):
                            seed_id_top[tc] = lid

    # Bottom edge: top row of tile below
    if iy < n_tile_y - 1:
        nb_fdir = flow_bdry.get('top', iy + 1, ix)
        nb_mask = mask_bdry.get('top', iy + 1, ix)
        nb_link = boundaries.get('top', iy + 1, ix)
        w = len(nb_fdir)
        for j in range(w):
            if nb_mask[j] == 0:
                continue
            theta = nb_fdir[j]
            if theta != theta or theta < 0.0:
                continue
            k1, f1, k2, f2 = _dinf_downstream(theta)
            if k1 < 0:
                continue
            for kk, frac in [(k1, f1), (k2, f2)]:
                if frac <= 0.0:
                    continue
                dy = int(nb_dy[kk])
                dx = int(nb_dx[kk])
                if dy == -1:  # flows north into our tile
                    tc = j + dx
                    if 0 <= tc < tile_w:
                        seed_ext_bottom[tc] += 1.0
                        lid = nb_link[j]
                        if lid > 0 and (seed_id_bottom[tc] == 0
                                        or lid < seed_id_bottom[tc]):
                            seed_id_bottom[tc] = lid

    # Left edge: right column of tile to the left
    if ix > 0:
        nb_fdir = flow_bdry.get('right', iy, ix - 1)
        nb_mask = mask_bdry.get('right', iy, ix - 1)
        nb_link = boundaries.get('right', iy, ix - 1)
        h = len(nb_fdir)
        for i in range(h):
            if nb_mask[i] == 0:
                continue
            theta = nb_fdir[i]
            if theta != theta or theta < 0.0:
                continue
            k1, f1, k2, f2 = _dinf_downstream(theta)
            if k1 < 0:
                continue
            for kk, frac in [(k1, f1), (k2, f2)]:
                if frac <= 0.0:
                    continue
                dy = int(nb_dy[kk])
                dx = int(nb_dx[kk])
                if dx == 1:  # flows east into our tile
                    tr = i + dy
                    if 0 <= tr < tile_h:
                        seed_ext_left[tr] += 1.0
                        lid = nb_link[i]
                        if lid > 0 and (seed_id_left[tr] == 0
                                        or lid < seed_id_left[tr]):
                            seed_id_left[tr] = lid

    # Right edge: left column of tile to the right
    if ix < n_tile_x - 1:
        nb_fdir = flow_bdry.get('left', iy, ix + 1)
        nb_mask = mask_bdry.get('left', iy, ix + 1)
        nb_link = boundaries.get('left', iy, ix + 1)
        h = len(nb_fdir)
        for i in range(h):
            if nb_mask[i] == 0:
                continue
            theta = nb_fdir[i]
            if theta != theta or theta < 0.0:
                continue
            k1, f1, k2, f2 = _dinf_downstream(theta)
            if k1 < 0:
                continue
            for kk, frac in [(k1, f1), (k2, f2)]:
                if frac <= 0.0:
                    continue
                dy = int(nb_dy[kk])
                dx = int(nb_dx[kk])
                if dx == -1:  # flows west into our tile
                    tr = i + dy
                    if 0 <= tr < tile_h:
                        seed_ext_right[tr] += 1.0
                        lid = nb_link[i]
                        if lid > 0 and (seed_id_right[tr] == 0
                                        or lid < seed_id_right[tr]):
                            seed_id_right[tr] = lid

    # Corner seeds from diagonal neighbour tiles
    def _corner_link(nb_side, nb_iy, nb_ix, nb_idx,
                     required_dy, required_dx,
                     s_id, s_ext, target):
        fdir = flow_bdry.get(nb_side, nb_iy, nb_ix)[nb_idx]
        mask = mask_bdry.get(nb_side, nb_iy, nb_ix)[nb_idx]
        if mask == 0:
            return
        if fdir != fdir or fdir < 0.0:
            return
        k1, f1, k2, f2 = _dinf_downstream(fdir)
        if k1 < 0:
            return
        for kk, frac in [(k1, f1), (k2, f2)]:
            if frac <= 0.0:
                continue
            dy = int(nb_dy[kk])
            dx = int(nb_dx[kk])
            if dy == required_dy and dx == required_dx:
                s_ext[target] += 1.0
                lid = float(boundaries.get(nb_side, nb_iy, nb_ix)[nb_idx])
                if lid > 0 and (s_id[target] == 0
                                or lid < s_id[target]):
                    s_id[target] = lid

    # Top-left corner: bottom-right cell of (iy-1, ix-1)
    if iy > 0 and ix > 0:
        _corner_link('bottom', iy - 1, ix - 1, -1,
                     1, 1, seed_id_top, seed_ext_top, 0)
    # Top-right corner: bottom-left cell of (iy-1, ix+1)
    if iy > 0 and ix < n_tile_x - 1:
        _corner_link('bottom', iy - 1, ix + 1, 0,
                     1, -1, seed_id_top, seed_ext_top, tile_w - 1)
    # Bottom-left corner: top-right cell of (iy+1, ix-1)
    if iy < n_tile_y - 1 and ix > 0:
        _corner_link('top', iy + 1, ix - 1, -1,
                     -1, 1, seed_id_bottom, seed_ext_bottom, 0)
    # Bottom-right corner: top-left cell of (iy+1, ix+1)
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        _corner_link('top', iy + 1, ix + 1, 0,
                     -1, -1, seed_id_bottom, seed_ext_bottom, tile_w - 1)

    return (seed_id_top, seed_id_bottom, seed_id_left, seed_id_right,
            seed_ext_top, seed_ext_bottom, seed_ext_left, seed_ext_right)


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _process_link_tile_dinf(iy, ix, flow_dir_da, accum_da, threshold,
                             boundaries, flow_bdry, mask_bdry,
                             chunks_y, chunks_x, n_tile_y, n_tile_x,
                             row_offsets, col_offsets, total_width):
    """Run seeded BFS on one D-inf tile; update boundary stores."""
    fd_chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    ac_chunk = np.asarray(
        accum_da.blocks[iy, ix].compute(), dtype=np.float64)
    sm = _make_stream_mask_np_dinf(ac_chunk, fd_chunk, threshold)
    h, w = fd_chunk.shape

    seeds = _compute_link_seeds_dinf(
        iy, ix, boundaries, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    link = _stream_link_dinf_tile_kernel(
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


def _stream_link_dinf_dask(flow_dir_da, accum_da, threshold):
    """Iterative boundary-propagation for D-inf dask arrays."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    total_width = sum(chunks_x)

    row_offsets = np.zeros(n_tile_y, dtype=np.int64)
    col_offsets = np.zeros(n_tile_x, dtype=np.int64)
    if n_tile_y > 1:
        np.cumsum(chunks_y[:-1], out=row_offsets[1:])
    if n_tile_x > 1:
        np.cumsum(chunks_x[:-1], out=col_offsets[1:])

    flow_bdry, mask_bdry = _preprocess_stream_tiles_dinf(
        flow_dir_da, accum_da, threshold, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()
    mask_bdry = mask_bdry.snapshot()

    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    max_iterations = max(n_tile_y, n_tile_x) + 10
    for _ in range(max_iterations):
        max_change = 0.0
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_link_tile_dinf(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    boundaries, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x,
                    row_offsets, col_offsets, total_width)
                if c > max_change:
                    max_change = c
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_link_tile_dinf(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    boundaries, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x,
                    row_offsets, col_offsets, total_width)
                if c > max_change:
                    max_change = c
        if max_change == 0.0:
            break

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
        sm = _make_stream_mask_np_dinf(ac, fd, _threshold)
        h, w = fd.shape
        seeds = _compute_link_seeds_dinf(
            iy, ix, _boundaries, _flow_bdry, _mask_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _stream_link_dinf_tile_kernel(
            fd, sm, h, w, *seeds,
            row_offsets[iy], col_offsets[ix], total_width)

    return da.map_blocks(
        _tile_fn, flow_dir_da, accum_da,
        dtype=np.float64, meta=np.array((), dtype=np.float64))


# =====================================================================
# Dask + CuPy
# =====================================================================

def _process_link_tile_dinf_cupy(iy, ix, flow_dir_da, accum_da, threshold,
                                  boundaries, flow_bdry, mask_bdry,
                                  chunks_y, chunks_x, n_tile_y, n_tile_x,
                                  row_offsets, col_offsets, total_width):
    """Run seeded GPU D-inf stream link on one tile; update boundaries."""
    import cupy as cp

    fd_chunk = cp.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=cp.float64)
    ac_chunk = _to_numpy_f64(accum_da.blocks[iy, ix].compute())
    fd_np = fd_chunk.get()
    sm_np = _make_stream_mask_np_dinf(ac_chunk, fd_np, threshold)
    sm_cp = cp.asarray(sm_np)

    seeds = _compute_link_seeds_dinf(
        iy, ix, boundaries, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    link = _stream_link_dinf_tile_cupy(
        fd_chunk, sm_cp, *seeds,
        row_offsets[iy], col_offsets[ix], total_width)

    change = 0.0
    for side, strip_cp in [('top', link[0, :]),
                           ('bottom', link[-1, :]),
                           ('left', link[:, 0]),
                           ('right', link[:, -1])]:
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


def _stream_link_dinf_dask_cupy(flow_dir_da, accum_da, threshold):
    """Dask+CuPy: native GPU processing per tile for D-inf stream link."""
    import cupy as cp

    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    total_width = sum(chunks_x)

    row_offsets = np.zeros(n_tile_y, dtype=np.int64)
    col_offsets = np.zeros(n_tile_x, dtype=np.int64)
    if n_tile_y > 1:
        np.cumsum(chunks_y[:-1], out=row_offsets[1:])
    if n_tile_x > 1:
        np.cumsum(chunks_x[:-1], out=col_offsets[1:])

    flow_bdry, mask_bdry = _preprocess_stream_tiles_dinf(
        flow_dir_da, accum_da, threshold, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()
    mask_bdry = mask_bdry.snapshot()

    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    max_iterations = max(n_tile_y, n_tile_x) + 10
    for _ in range(max_iterations):
        max_change = 0.0
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_link_tile_dinf_cupy(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    boundaries, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x,
                    row_offsets, col_offsets, total_width)
                if c > max_change:
                    max_change = c
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_link_tile_dinf_cupy(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    boundaries, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x,
                    row_offsets, col_offsets, total_width)
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
        sm = cp.asarray(_make_stream_mask_np_dinf(ac_np, fd_np, threshold))
        seeds = _compute_link_seeds_dinf(
            iy, ix, _boundaries, flow_bdry, mask_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _stream_link_dinf_tile_cupy(
            fd, sm, *seeds,
            row_offsets[iy], col_offsets[ix], total_width)

    return da.map_blocks(
        _tile_fn, flow_dir_da, accum_da,
        dtype=np.float64, meta=cp.array((), dtype=cp.float64))


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def stream_link_dinf(flow_dir_dinf: xr.DataArray,
                      flow_accum: xr.DataArray,
                      threshold: float = 100,
                      name: str = 'stream_link_dinf') -> xr.DataArray:
    """Assign unique IDs to stream link segments (D-infinity).

    Each contiguous stream segment between junctions, headwaters, and
    outlets receives a distinct positive integer ID.  Non-stream cells
    are NaN.

    Parameters
    ----------
    flow_dir_dinf : xarray.DataArray or xr.Dataset
        2D D-infinity flow direction grid.  Values are continuous
        angles in radians in the range ``[0, 2*pi)``.  ``-1.0``
        indicates a pit or flat.  ``NaN`` indicates nodata.
    flow_accum : xarray.DataArray
        2D flow accumulation grid.  Cells with
        ``flow_accum >= threshold`` are stream cells.
    threshold : float, default 100
        Minimum accumulation for stream classification.
    name : str, default 'stream_link_dinf'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 grid where each stream link has a unique positive
        integer ID.  Non-stream cells are NaN.

    References
    ----------
    Tarboton, D.G. (1997). A new method for the determination of flow
    directions and upslope areas in grid digital elevation models.
    Water Resources Research, 33(2), 309-319.
    """
    _validate_raster(flow_dir_dinf,
                     func_name='stream_link_dinf',
                     name='flow_dir_dinf')

    fd_data = flow_dir_dinf.data
    fa_data = flow_accum.data

    if isinstance(fd_data, np.ndarray):
        fd = fd_data.astype(np.float64)
        fa = np.asarray(fa_data, dtype=np.float64)
        stream_mask = np.where(fa >= threshold, 1, 0).astype(np.int8)
        stream_mask = np.where(np.isnan(fa), 0, stream_mask).astype(np.int8)
        stream_mask = np.where(np.isnan(fd), 0, stream_mask).astype(np.int8)
        h, w = fd.shape
        out = _stream_link_dinf_cpu(fd, stream_mask, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(fd_data):
        import cupy as cp
        fa_cp = cp.asarray(fa_data, dtype=cp.float64)
        fd_cp = fd_data.astype(cp.float64)
        stream_mask = cp.where(fa_cp >= threshold, 1, 0).astype(cp.int8)
        stream_mask = cp.where(
            cp.isnan(fa_cp), 0, stream_mask).astype(cp.int8)
        stream_mask = cp.where(
            cp.isnan(fd_cp), 0, stream_mask).astype(cp.int8)
        out = _stream_link_dinf_cupy(fd_cp, stream_mask)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir_dinf):
        out = _stream_link_dinf_dask_cupy(fd_data, fa_data, threshold)

    elif da is not None and isinstance(fd_data, da.Array):
        out = _stream_link_dinf_dask(fd_data, fa_data, threshold)

    else:
        raise TypeError(f"Unsupported array type: {type(fd_data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir_dinf.coords,
                        dims=flow_dir_dinf.dims,
                        attrs=flow_dir_dinf.attrs)
