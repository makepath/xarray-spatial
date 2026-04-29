"""D-infinity Height Above Nearest Drainage (HAND).

Uses D-inf angle decomposition for downstream tracing.  At each cell,
the dominant neighbor (higher weight) is followed to find the nearest
stream cell.  HAND = elevation - drain_elevation.

Algorithm
---------
CPU : Kahn's BFS topological sort with reverse propagation of drain_elev.
      In-degrees use both D-inf neighbors; reverse pass follows dominant.
GPU : CuPy-via-CPU.
Dask: iterative tile sweep with BoundaryStore exit-label propagation.
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.hydro.flow_accumulation_dinf import _angle_to_neighbors
from xrspatial.hydro.flow_path_dinf import _angle_to_neighbors_py
from xrspatial.hydro._boundary_store import BoundaryStore
from xrspatial.hydro.watershed_dinf import (
    _dominant_offset_py,
    _preprocess_tiles,
    _to_numpy_f64,
)
from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)


# =====================================================================
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for ``_hand_dinf_cpu``:
#   in_degree  : int32   -> 4
#   valid      : int8    -> 1
#   is_stream  : int8    -> 1
#   drain_elev : float64 -> 8
#   hand_out   : float64 -> 8
#   order_r    : int64   -> 8
#   order_c    : int64   -> 8
# Total ~38 bytes/pixel.  Caller-provided ``flow_dir``, ``flow_accum``,
# and ``elevation`` arrays already live in RAM before the kernel runs
# and are not double-counted here.
_BYTES_PER_PIXEL = 38

# GPU peak working set per pixel for ``_hand_dinf_cupy``: that path
# copies fd/fa/elev to host via ``.get()`` then runs ``_hand_dinf_cpu``.
# Host working set is dominated by the same 38 B/px as the numpy path;
# on the device we keep the three input arrays (3 * float64 = 24 B/px)
# and the output (float64 = 8 B/px) -- 32 B/px total on the GPU side,
# but the input copies already exist before dispatch, so the marginal
# device allocation is 8 B/px.  Use 32 B/px as a conservative budget
# mirroring the d8 and mfd siblings.
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
    """Raise MemoryError if the HAND kernel would exceed 50% of RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"hand_dinf on a {height}x{width} grid requires "
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
            f"hand_dinf on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _hand_dinf_cpu(flow_dir, flow_accum, elevation, H, W, threshold):
    """Compute HAND via Kahn's BFS with D-inf angle decomposition."""
    in_degree = np.zeros((H, W), dtype=np.int32)
    valid = np.zeros((H, W), dtype=np.int8)
    is_stream = np.zeros((H, W), dtype=np.int8)
    drain_elev = np.empty((H, W), dtype=np.float64)
    hand_out = np.empty((H, W), dtype=np.float64)

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

    # In-degrees: both D-inf neighbors contribute
    for r in range(H):
        for c in range(W):
            if valid[r, c] == 0:
                continue
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
            if w1 > 0.0:
                nr, nc = r + dy1, c + dx1
                if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                    in_degree[nr, nc] += 1
            if w2 > 0.0:
                nr, nc = r + dy2, c + dx2
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
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
        if w1 > 0.0:
            nr, nc = r + dy1, c + dx1
            if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    order_r[tail] = nr
                    order_c[tail] = nc
                    tail += 1
        if w2 > 0.0:
            nr, nc = r + dy2, c + dx2
            if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    order_r[tail] = nr
                    order_c[tail] = nc
                    tail += 1

    # Reverse pass: propagate drain_elev via dominant neighbor
    for i in range(tail - 1, -1, -1):
        r = order_r[i]
        c = order_c[i]
        if is_stream[r, c] == 1:
            continue
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
        if w1 <= 0.0 and w2 <= 0.0:
            drain_elev[r, c] = elevation[r, c]
            continue
        if w1 >= w2:
            ddy, ddx = dy1, dx1
        else:
            ddy, ddx = dy2, dx2
        nr, nc = r + ddy, c + ddx
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
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

    for r in range(H):
        for c in range(W):
            if valid[r, c] == 1:
                hand_out[r, c] = elevation[r, c] - drain_elev[r, c]
            else:
                hand_out[r, c] = np.nan

    return hand_out


# =====================================================================
# CuPy backend
# =====================================================================

def _hand_dinf_cupy(fd_data, fa_data, elev_data, threshold):
    import cupy as cp
    fd_np = fd_data.get().astype(np.float64)
    fa_np = fa_data.get().astype(np.float64)
    el_np = elev_data.get().astype(np.float64)
    H, W = fd_np.shape
    out = _hand_dinf_cpu(fd_np, fa_np, el_np, H, W, threshold)
    return cp.asarray(out)


# =====================================================================
# Dask tile kernel
# =====================================================================

@ngjit
def _hand_dinf_drain_elev_tile(flow_dir, flow_accum, elevation, h, w,
                                threshold,
                                exit_top, exit_bottom, exit_left, exit_right,
                                exit_tl, exit_tr, exit_bl, exit_br):
    """Compute drain_elev for a D-inf tile (for boundary propagation)."""
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

    # Apply exit labels at boundaries where dominant neighbor exits tile
    for c in range(w):
        if valid[0, c] == 1 and known[0, c] == 0:
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[0, c])
            if w1 <= 0.0 and w2 <= 0.0:
                continue
            if w1 >= w2:
                ddy = dy1
            else:
                ddy = dy2
            if 0 + ddy < 0:
                el = exit_top[c]
                if el == el:
                    drain_elev[0, c] = el
                    known[0, c] = 1
                else:
                    drain_elev[0, c] = elevation[0, c]
                    known[0, c] = 1

    for c in range(w):
        if valid[h - 1, c] == 1 and known[h - 1, c] == 0:
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[h - 1, c])
            if w1 <= 0.0 and w2 <= 0.0:
                continue
            if w1 >= w2:
                ddy = dy1
            else:
                ddy = dy2
            if h - 1 + ddy >= h:
                el = exit_bottom[c]
                if el == el:
                    drain_elev[h - 1, c] = el
                    known[h - 1, c] = 1
                else:
                    drain_elev[h - 1, c] = elevation[h - 1, c]
                    known[h - 1, c] = 1

    for r in range(h):
        if valid[r, 0] == 1 and known[r, 0] == 0:
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, 0])
            if w1 <= 0.0 and w2 <= 0.0:
                continue
            if w1 >= w2:
                ddx = dx1
            else:
                ddx = dx2
            if 0 + ddx < 0:
                el = exit_left[r]
                if el == el:
                    drain_elev[r, 0] = el
                    known[r, 0] = 1
                else:
                    drain_elev[r, 0] = elevation[r, 0]
                    known[r, 0] = 1

    for r in range(h):
        if valid[r, w - 1] == 1 and known[r, w - 1] == 0:
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, w - 1])
            if w1 <= 0.0 and w2 <= 0.0:
                continue
            if w1 >= w2:
                ddx = dx1
            else:
                ddx = dx2
            if w - 1 + ddx >= w:
                el = exit_right[r]
                if el == el:
                    drain_elev[r, w - 1] = el
                    known[r, w - 1] = 1
                else:
                    drain_elev[r, w - 1] = elevation[r, w - 1]
                    known[r, w - 1] = 1

    # Corner overrides
    if valid[0, 0] == 1 and known[0, 0] == 0:
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[0, 0])
        if not (w1 <= 0.0 and w2 <= 0.0):
            if w1 >= w2:
                ddy, ddx = dy1, dx1
            else:
                ddy, ddx = dy2, dx2
            if 0 + ddy < 0 and 0 + ddx < 0:
                if exit_tl == exit_tl:
                    drain_elev[0, 0] = exit_tl
                    known[0, 0] = 1
    if valid[0, w - 1] == 1 and known[0, w - 1] == 0:
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[0, w - 1])
        if not (w1 <= 0.0 and w2 <= 0.0):
            if w1 >= w2:
                ddy, ddx = dy1, dx1
            else:
                ddy, ddx = dy2, dx2
            if 0 + ddy < 0 and w - 1 + ddx >= w:
                if exit_tr == exit_tr:
                    drain_elev[0, w - 1] = exit_tr
                    known[0, w - 1] = 1
    if valid[h - 1, 0] == 1 and known[h - 1, 0] == 0:
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[h - 1, 0])
        if not (w1 <= 0.0 and w2 <= 0.0):
            if w1 >= w2:
                ddy, ddx = dy1, dx1
            else:
                ddy, ddx = dy2, dx2
            if h - 1 + ddy >= h and 0 + ddx < 0:
                if exit_bl == exit_bl:
                    drain_elev[h - 1, 0] = exit_bl
                    known[h - 1, 0] = 1
    if valid[h - 1, w - 1] == 1 and known[h - 1, w - 1] == 0:
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[h - 1, w - 1])
        if not (w1 <= 0.0 and w2 <= 0.0):
            if w1 >= w2:
                ddy, ddx = dy1, dx1
            else:
                ddy, ddx = dy2, dx2
            if h - 1 + ddy >= h and w - 1 + ddx >= w:
                if exit_br == exit_br:
                    drain_elev[h - 1, w - 1] = exit_br
                    known[h - 1, w - 1] = 1

    # In-degrees (non-known cells only)
    for r in range(h):
        for c in range(w):
            if valid[r, c] == 0 or known[r, c] == 1:
                continue
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
            if w1 > 0.0:
                nr, nc = r + dy1, c + dx1
                if 0 <= nr < h and 0 <= nc < w:
                    if valid[nr, nc] == 1 and known[nr, nc] == 0:
                        in_degree[nr, nc] += 1
            if w2 > 0.0:
                nr, nc = r + dy2, c + dx2
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
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
        if w1 > 0.0:
            nr, nc = r + dy1, c + dx1
            if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1 and known[nr, nc] == 0:
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    order_r[tail] = nr
                    order_c[tail] = nc
                    tail += 1
        if w2 > 0.0:
            nr, nc = r + dy2, c + dx2
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
        dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(flow_dir[r, c])
        if w1 <= 0.0 and w2 <= 0.0:
            drain_elev[r, c] = elevation[r, c]
            continue
        if w1 >= w2:
            ddy, ddx = dy1, dx1
        else:
            ddy, ddx = dy2, dx2
        nr, nc = r + ddy, c + ddx
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


@ngjit
def _hand_dinf_tile_kernel(flow_dir, flow_accum, elevation, h, w, threshold,
                            exit_top, exit_bottom, exit_left, exit_right,
                            exit_tl, exit_tr, exit_bl, exit_br):
    """HAND tile kernel: returns HAND values (elevation - drain_elev)."""
    drain_elev = _hand_dinf_drain_elev_tile(
        flow_dir, flow_accum, elevation, h, w, threshold,
        exit_top, exit_bottom, exit_left, exit_right,
        exit_tl, exit_tr, exit_bl, exit_br)

    out = np.empty((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v == v:
                out[r, c] = elevation[r, c] - drain_elev[r, c]
            else:
                out[r, c] = np.nan
    return out


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _compute_exit_labels(iy, ix, boundaries, flow_bdry,
                          chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Same exit-label pattern as watershed_dinf but propagating drain_elev."""
    from xrspatial.hydro.watershed_dinf import _compute_exit_labels as _ws_compute
    return _ws_compute(iy, ix, boundaries, flow_bdry,
                       chunks_y, chunks_x, n_tile_y, n_tile_x)


def _process_tile_hand(iy, ix, flow_dir_da, flow_accum_da, elev_da,
                        boundaries, flow_bdry, threshold,
                        chunks_y, chunks_x, n_tile_y, n_tile_x):
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

    drain_elev = _hand_dinf_drain_elev_tile(
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


def _hand_dinf_dask(flow_dir_da, flow_accum_da, elev_da, threshold):
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

    def _tile_fn(fd_block, fa_block, el_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(fd_block.shape, np.nan, dtype=np.float64)
        iy, ix = block_info[0]['chunk-location']
        h, w = fd_block.shape
        exits = _compute_exit_labels(
            iy, ix, boundaries, flow_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _hand_dinf_tile_kernel(
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


def _hand_dinf_dask_cupy(flow_dir_da, flow_accum_da, elev_da, threshold):
    import cupy as cp
    fd_np = flow_dir_da.map_blocks(
        lambda b: b.get(), dtype=flow_dir_da.dtype,
        meta=np.array((), dtype=flow_dir_da.dtype))
    fa_np = flow_accum_da.map_blocks(
        lambda b: b.get(), dtype=flow_accum_da.dtype,
        meta=np.array((), dtype=flow_accum_da.dtype))
    el_np = elev_da.map_blocks(
        lambda b: b.get(), dtype=elev_da.dtype,
        meta=np.array((), dtype=elev_da.dtype))
    result = _hand_dinf_dask(fd_np, fa_np, el_np, threshold)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype))


# =====================================================================
# Public API
# =====================================================================

def hand_dinf(flow_dir_dinf: xr.DataArray,
              flow_accum: xr.DataArray,
              elevation: xr.DataArray,
              threshold: float = 100,
              name: str = 'hand_dinf') -> xr.DataArray:
    """Compute HAND using D-infinity flow direction.

    Parameters
    ----------
    flow_dir_dinf : xarray.DataArray
        2D D-infinity flow direction grid.
    flow_accum : xarray.DataArray
        2D flow accumulation grid.
    elevation : xarray.DataArray
        2D elevation grid.
    threshold : float, default 100
        Minimum flow accumulation to define a stream cell.
    name : str, default 'hand_dinf'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 HAND grid. Stream cells have HAND = 0.
    """
    _validate_raster(flow_dir_dinf, func_name='hand_dinf', name='flow_dir_dinf')
    _validate_raster(flow_accum, func_name='hand_dinf', name='flow_accum')
    _validate_raster(elevation, func_name='hand_dinf', name='elevation')

    fd_data = flow_dir_dinf.data
    fa_data = flow_accum.data
    el_data = elevation.data

    if isinstance(fd_data, np.ndarray):
        _check_memory(*fd_data.shape)
        fd = fd_data.astype(np.float64)
        fa = np.asarray(fa_data, dtype=np.float64)
        el = np.asarray(el_data, dtype=np.float64)
        H, W = fd.shape
        out = _hand_dinf_cpu(fd, fa, el, H, W, float(threshold))

    elif has_cuda_and_cupy() and is_cupy_array(fd_data):
        _check_gpu_memory(*fd_data.shape)
        _check_memory(*fd_data.shape)
        out = _hand_dinf_cupy(fd_data, fa_data, el_data, float(threshold))

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir_dinf):
        out = _hand_dinf_dask_cupy(fd_data, fa_data, el_data, float(threshold))

    elif da is not None and isinstance(fd_data, da.Array):
        out = _hand_dinf_dask(fd_data, fa_data, el_data, float(threshold))

    else:
        raise TypeError(f"Unsupported array type: {type(fd_data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir_dinf.coords,
                        dims=flow_dir_dinf.dims,
                        attrs=flow_dir_dinf.attrs)
