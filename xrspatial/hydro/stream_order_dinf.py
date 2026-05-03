"""Stream order extraction for D-infinity flow direction grids.

Assigns hierarchical order values to stream cells derived from a
D-infinity (continuous angle) flow direction grid and a flow
accumulation grid.  Cells with accumulation below a user-defined
threshold are non-stream and receive NaN.  Two methods are supported:

* **Strahler**: headwaters = 1; when two streams of equal order meet
  the downstream order increments by 1; otherwise the higher order
  propagates.
* **Shreve**: headwaters = 1; at each confluence the downstream
  magnitude equals the sum of all incoming magnitudes.

Unlike the D8 variant, D-inf flow directions are continuous angles in
[0, 2*pi).  Each cell flows to at most two downstream neighbours
(the pair bracketing the angle), with proportional fractions.  The
topology (in-degree, upstream/downstream relationship) is derived from
these angles rather than from discrete D8 codes.

Algorithm
---------
CPU : Kahn's BFS topological sort among stream cells -- O(N_stream).
GPU : iterative frontier peeling with pull-based kernels.
Dask: iterative tile sweep with boundary propagation, same pattern
      as ``stream_order.py``.
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


# =====================================================================
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for the eager Strahler/Shreve D-inf
# kernels:
#   order     : float64 -> 8
#   in_degree : int32   -> 4
#   max_in    : float64 -> 8   (Strahler only; Shreve omits it)
#   cnt_max   : int32   -> 4   (Strahler only)
#   queue_r   : int64   -> 8
#   queue_c   : int64   -> 8
# Total ~40 bytes/pixel for Strahler, ~32 for Shreve.  D-inf encodes one
# continuous downstream angle per cell, not an 8-channel weight buffer
# like MFD, so the working set matches the d8 budget.  We budget for the
# worst case.  Caller-provided ``flow_dir`` and ``flow_accum`` already
# live in RAM before the kernel runs and are not double-counted here.
_BYTES_PER_PIXEL = 40

# GPU peak working set per pixel for ``_stream_order_dinf_cupy``:
#   angles_f64     : float64 -> 8
#   stream_mask_i8 : int8    -> 1
#   in_degree      : int32   -> 4
#   state          : int32   -> 4
#   order          : float64 -> 8
#   max_in         : float64 -> 8
#   cnt_max        : int32   -> 4
# Total ~37 bytes/pixel.  The ``fa_cp`` input copy adds 8 B/px on the
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
    """Raise MemoryError if the kernel would exceed 50% of available RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"stream_order_dinf on a {height}x{width} grid requires "
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
            f"stream_order_dinf on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


# Neighbor offsets: counterclockwise from East
# E, NE, N, NW, W, SW, S, SE
_NB_DY = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int64)
_NB_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

_PI_OVER_4 = math.pi / 4.0
_TWO_PI = 2.0 * math.pi


# =====================================================================
# Helper: angle -> two downstream neighbors + fractions
# =====================================================================

def _dinf_downstream(theta):
    """Return (k1, frac1, k2, frac2) for a D-inf angle theta.

    k1/k2 are neighbor indices (0..7 into the E,NE,N,...,SE convention).
    frac1 is the proportion of flow to neighbor k1, frac2 to k2.
    Returns (-1, 0, -1, 0) if theta is NaN or < 0 (pit/nodata).
    """
    if theta != theta:  # NaN
        return -1, 0.0, -1, 0.0
    if theta < 0.0:
        return -1, 0.0, -1, 0.0

    k = int(theta / _PI_OVER_4)
    if k >= 8:
        k = 7
    k2 = (k + 1) % 8
    alpha = theta - k * _PI_OVER_4
    frac1 = (_PI_OVER_4 - alpha) / _PI_OVER_4
    frac2 = alpha / _PI_OVER_4
    return k, frac1, k2, frac2


# =====================================================================
# CPU kernels
# =====================================================================

@ngjit
def _strahler_dinf_cpu(angles, stream_mask, height, width):
    """Kahn's BFS Strahler ordering for D-inf flow directions."""
    nb_dy = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int64)
    nb_dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    pi_over_4 = math.pi / 4.0

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

    # Compute in-degrees: each stream cell's angle gives 2 downstream
    # neighbors; increment their in-degree if they are stream cells.
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
        cur_ord = order[r, c]

        # Propagate to neighbor k
        if frac1 > 0.0:
            nr = r + nb_dy[k]
            nc = c + nb_dx[k]
            if 0 <= nr < height and 0 <= nc < width:
                if stream_mask[nr, nc] == 1:
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

        # Propagate to neighbor k2
        if frac2 > 0.0:
            nr2 = r + nb_dy[k2]
            nc2 = c + nb_dx[k2]
            if 0 <= nr2 < height and 0 <= nc2 < width:
                if stream_mask[nr2, nc2] == 1:
                    if cur_ord > max_in[nr2, nc2]:
                        max_in[nr2, nc2] = cur_ord
                        cnt_max[nr2, nc2] = 1
                    elif cur_ord == max_in[nr2, nc2]:
                        cnt_max[nr2, nc2] += 1

                    in_degree[nr2, nc2] -= 1
                    if in_degree[nr2, nc2] == 0:
                        if cnt_max[nr2, nc2] >= 2:
                            order[nr2, nc2] = max_in[nr2, nc2] + 1.0
                        else:
                            order[nr2, nc2] = max_in[nr2, nc2]
                        queue_r[tail] = nr2
                        queue_c[tail] = nc2
                        tail += 1

    return order


@ngjit
def _shreve_dinf_cpu(angles, stream_mask, height, width):
    """Kahn's BFS Shreve ordering for D-inf flow directions."""
    nb_dy = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int64)
    nb_dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    pi_over_4 = math.pi / 4.0

    order = np.empty((height, width), dtype=np.float64)
    in_degree = np.zeros((height, width), dtype=np.int32)

    for r in range(height):
        for c in range(width):
            if stream_mask[r, c] == 0:
                order[r, c] = np.nan
            else:
                order[r, c] = 0.0

    # Compute in-degrees
    for r in range(height):
        for c in range(width):
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
                if 0 <= nr < height and 0 <= nc < width:
                    if stream_mask[nr, nc] == 1:
                        in_degree[nr, nc] += 1

            if frac2 > 0.0:
                nr2 = r + nb_dy[k2]
                nc2 = c + nb_dx[k2]
                if 0 <= nr2 < height and 0 <= nc2 < width:
                    if stream_mask[nr2, nc2] == 1:
                        in_degree[nr2, nc2] += 1

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
        cur_ord = order[r, c]

        if frac1 > 0.0:
            nr = r + nb_dy[k]
            nc = c + nb_dx[k]
            if 0 <= nr < height and 0 <= nc < width:
                if stream_mask[nr, nc] == 1:
                    order[nr, nc] += cur_ord
                    in_degree[nr, nc] -= 1
                    if in_degree[nr, nc] == 0:
                        queue_r[tail] = nr
                        queue_c[tail] = nc
                        tail += 1

        if frac2 > 0.0:
            nr2 = r + nb_dy[k2]
            nc2 = c + nb_dx[k2]
            if 0 <= nr2 < height and 0 <= nc2 < width:
                if stream_mask[nr2, nc2] == 1:
                    order[nr2, nc2] += cur_ord
                    in_degree[nr2, nc2] -= 1
                    if in_degree[nr2, nc2] == 0:
                        queue_r[tail] = nr2
                        queue_c[tail] = nc2
                        tail += 1

    return order


# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit(device=True)
def _dinf_nb_dy(k):
    """Return row offset for neighbor k (0..7)."""
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
    """Return column offset for neighbor k (0..7)."""
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
    """Convert angle to primary neighbor index k (0..7).

    Returns -1 if theta is NaN or < 0 (pit/nodata).
    """
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
def _stream_order_dinf_init_gpu(angles, stream_mask, in_degree, state,
                                 order, max_in, cnt_max, H, W):
    """Initialise GPU arrays and compute in-degrees from D-inf angles."""
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
def _stream_order_dinf_find_ready(in_degree, state, order, changed, H, W):
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
def _stream_order_dinf_pull_strahler(angles, stream_mask, in_degree, state,
                                      order, max_in, cnt_max, H, W):
    """Active cells pull Strahler info from frontier D-inf neighbours."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    if state[i, j] != 1:
        return

    pi_over_4 = 0.7853981633974483

    # Check each of the 8 neighbors to see if it is on the frontier
    # and flows to us.
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

        # Check if the neighbor's angle points to us.
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
def _stream_order_dinf_pull_shreve(angles, stream_mask, in_degree, state,
                                    order, H, W):
    """Active cells pull Shreve magnitudes from frontier D-inf neighbours."""
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
            order[i, j] += order[ni, nj]
            in_degree[i, j] -= 1


# =====================================================================
# CuPy driver
# =====================================================================

def _stream_order_dinf_cupy(angles_data, stream_mask_data, method):
    """GPU driver for D-inf stream order computation."""
    import cupy as cp

    H, W = angles_data.shape
    angles_f64 = angles_data.astype(cp.float64)
    stream_mask_i8 = stream_mask_data.astype(cp.int8)

    in_degree = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    order = cp.zeros((H, W), dtype=cp.float64)
    max_in = cp.zeros((H, W), dtype=cp.float64)
    cnt_max = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _stream_order_dinf_init_gpu[griddim, blockdim](
        angles_f64, stream_mask_i8, in_degree, state,
        order, max_in, cnt_max, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _stream_order_dinf_find_ready[griddim, blockdim](
            in_degree, state, order, changed, H, W)

        if int(changed[0]) == 0:
            break

        if method == 'strahler':
            _stream_order_dinf_pull_strahler[griddim, blockdim](
                angles_f64, stream_mask_i8, in_degree, state,
                order, max_in, cnt_max, H, W)
        else:
            _stream_order_dinf_pull_shreve[griddim, blockdim](
                angles_f64, stream_mask_i8, in_degree, state,
                order, H, W)

    order = cp.where(stream_mask_i8 == 0, cp.nan, order)
    return order


# =====================================================================
# CPU tile kernels for Dask
# =====================================================================

@ngjit
def _strahler_dinf_tile_kernel(angles, stream_mask, h, w,
                                seed_max_top, seed_cnt_top,
                                seed_max_bottom, seed_cnt_bottom,
                                seed_max_left, seed_cnt_left,
                                seed_max_right, seed_cnt_right):
    """Seeded Strahler BFS for a single tile (D-inf)."""
    nb_dy = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int64)
    nb_dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    pi_over_4 = math.pi / 4.0

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
        cur_ord = order[r, c]

        if frac1 > 0.0:
            nr = r + nb_dy[k]
            nc = c + nb_dx[k]
            if 0 <= nr < h and 0 <= nc < w:
                if stream_mask[nr, nc] == 1:
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

        if frac2 > 0.0:
            nr2 = r + nb_dy[k2]
            nc2 = c + nb_dx[k2]
            if 0 <= nr2 < h and 0 <= nc2 < w:
                if stream_mask[nr2, nc2] == 1:
                    if cur_ord > max_in[nr2, nc2]:
                        max_in[nr2, nc2] = cur_ord
                        cnt_max[nr2, nc2] = 1
                    elif cur_ord == max_in[nr2, nc2]:
                        cnt_max[nr2, nc2] += 1

                    in_degree[nr2, nc2] -= 1
                    if in_degree[nr2, nc2] == 0:
                        if cnt_max[nr2, nc2] >= 2:
                            order[nr2, nc2] = max_in[nr2, nc2] + 1.0
                        else:
                            order[nr2, nc2] = max_in[nr2, nc2]
                        queue_r[tail] = nr2
                        queue_c[tail] = nc2
                        tail += 1

    return order


@ngjit
def _shreve_dinf_tile_kernel(angles, stream_mask, h, w,
                              seed_top, seed_bottom,
                              seed_left, seed_right):
    """Seeded Shreve BFS for a single tile (D-inf)."""
    nb_dy = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int64)
    nb_dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    pi_over_4 = math.pi / 4.0

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
        cur_ord = order[r, c]

        if frac1 > 0.0:
            nr = r + nb_dy[k]
            nc = c + nb_dx[k]
            if 0 <= nr < h and 0 <= nc < w:
                if stream_mask[nr, nc] == 1:
                    order[nr, nc] += cur_ord
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
                    order[nr2, nc2] += cur_ord
                    in_degree[nr2, nc2] -= 1
                    if in_degree[nr2, nc2] == 0:
                        queue_r[tail] = nr2
                        queue_c[tail] = nc2
                        tail += 1

    return order


# =====================================================================
# Dask preprocessing and seed computation
# =====================================================================

def _preprocess_stream_tiles_dinf(flow_dir_da, accum_da, threshold,
                                   chunks_y, chunks_x):
    """Extract boundary strips for D-inf angles and stream mask."""
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


def _compute_shreve_seeds_dinf(iy, ix, boundaries, flow_bdry, mask_bdry,
                                chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute additive Shreve seeds from D-inf neighbour tiles."""
    nb_dy = _NB_DY
    nb_dx = _NB_DX

    tile_h = chunks_y[iy]
    tile_w = chunks_x[ix]

    seed_top = np.zeros(tile_w, dtype=np.float64)
    seed_bottom = np.zeros(tile_w, dtype=np.float64)
    seed_left = np.zeros(tile_h, dtype=np.float64)
    seed_right = np.zeros(tile_h, dtype=np.float64)

    # Top edge: bottom row of tile above flows south into our top row
    if iy > 0:
        nb_fdir = flow_bdry.get('bottom', iy - 1, ix)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix)
        nb_order = boundaries.get('bottom', iy - 1, ix)
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
                        seed_top[tc] += nb_order[j] * frac

    # Bottom edge: top row of tile below flows north into our bottom row
    if iy < n_tile_y - 1:
        nb_fdir = flow_bdry.get('top', iy + 1, ix)
        nb_mask = mask_bdry.get('top', iy + 1, ix)
        nb_order = boundaries.get('top', iy + 1, ix)
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
                        seed_bottom[tc] += nb_order[j] * frac

    # Left edge: right column of tile to the left flows east into us
    if ix > 0:
        nb_fdir = flow_bdry.get('right', iy, ix - 1)
        nb_mask = mask_bdry.get('right', iy, ix - 1)
        nb_order = boundaries.get('right', iy, ix - 1)
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
                        seed_left[tr] += nb_order[i] * frac

    # Right edge: left column of tile to the right flows west into us
    if ix < n_tile_x - 1:
        nb_fdir = flow_bdry.get('left', iy, ix + 1)
        nb_mask = mask_bdry.get('left', iy, ix + 1)
        nb_order = boundaries.get('left', iy, ix + 1)
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
                        seed_right[tr] += nb_order[i] * frac

    # Corner seeds from diagonal neighbour tiles.
    def _corner_shreve(nb_side, nb_iy, nb_ix, nb_idx,
                       required_dy, required_dx,
                       s_arr, target):
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
                order_val = boundaries.get(nb_side, nb_iy, nb_ix)[nb_idx]
                s_arr[target] += order_val * frac

    # Top-left corner: bottom-right cell of tile (iy-1, ix-1)
    if iy > 0 and ix > 0:
        _corner_shreve('bottom', iy - 1, ix - 1, -1,
                       1, 1, seed_top, 0)
    # Top-right corner: bottom-left cell of tile (iy-1, ix+1)
    if iy > 0 and ix < n_tile_x - 1:
        _corner_shreve('bottom', iy - 1, ix + 1, 0,
                       1, -1, seed_top, tile_w - 1)
    # Bottom-left corner: top-right cell of tile (iy+1, ix-1)
    if iy < n_tile_y - 1 and ix > 0:
        _corner_shreve('top', iy + 1, ix - 1, -1,
                       -1, 1, seed_bottom, 0)
    # Bottom-right corner: top-left cell of tile (iy+1, ix+1)
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        _corner_shreve('top', iy + 1, ix + 1, 0,
                       -1, -1, seed_bottom, tile_w - 1)

    return seed_top, seed_bottom, seed_left, seed_right


def _compute_strahler_seeds_dinf(iy, ix, bdry_max, bdry_cnt,
                                  flow_bdry, mask_bdry,
                                  chunks_y, chunks_x,
                                  n_tile_y, n_tile_x):
    """Compute Strahler (max, cnt) seeds from D-inf neighbour tiles."""
    nb_dy = _NB_DY
    nb_dx = _NB_DX

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

    def _reconstruct_order(nm, nc):
        """Reconstruct the order from stored max/cnt."""
        if nc >= 2:
            return nm + 1.0
        return nm

    # Top edge: bottom row of tile above
    if iy > 0:
        nb_fdir = flow_bdry.get('bottom', iy - 1, ix)
        nb_mask = mask_bdry.get('bottom', iy - 1, ix)
        nb_max = bdry_max.get('bottom', iy - 1, ix)
        nb_cnt = bdry_cnt.get('bottom', iy - 1, ix)
        w = len(nb_fdir)
        for j in range(w):
            if nb_mask[j] == 0 or nb_max[j] == 0:
                continue
            theta = nb_fdir[j]
            if theta != theta or theta < 0.0:
                continue
            k1, f1, k2, f2 = _dinf_downstream(theta)
            if k1 < 0:
                continue
            val = _reconstruct_order(nb_max[j], nb_cnt[j])
            for kk, frac in [(k1, f1), (k2, f2)]:
                if frac <= 0.0:
                    continue
                dy = int(nb_dy[kk])
                dx = int(nb_dx[kk])
                if dy == 1:  # flows south
                    tc = j + dx
                    if 0 <= tc < tile_w:
                        _update_max_cnt(smax_top, scnt_top, val, tc)

    # Bottom edge: top row of tile below
    if iy < n_tile_y - 1:
        nb_fdir = flow_bdry.get('top', iy + 1, ix)
        nb_mask = mask_bdry.get('top', iy + 1, ix)
        nb_max = bdry_max.get('top', iy + 1, ix)
        nb_cnt = bdry_cnt.get('top', iy + 1, ix)
        w = len(nb_fdir)
        for j in range(w):
            if nb_mask[j] == 0 or nb_max[j] == 0:
                continue
            theta = nb_fdir[j]
            if theta != theta or theta < 0.0:
                continue
            k1, f1, k2, f2 = _dinf_downstream(theta)
            if k1 < 0:
                continue
            val = _reconstruct_order(nb_max[j], nb_cnt[j])
            for kk, frac in [(k1, f1), (k2, f2)]:
                if frac <= 0.0:
                    continue
                dy = int(nb_dy[kk])
                dx = int(nb_dx[kk])
                if dy == -1:  # flows north
                    tc = j + dx
                    if 0 <= tc < tile_w:
                        _update_max_cnt(smax_bottom, scnt_bottom, val, tc)

    # Left edge: right column of tile to the left
    if ix > 0:
        nb_fdir = flow_bdry.get('right', iy, ix - 1)
        nb_mask = mask_bdry.get('right', iy, ix - 1)
        nb_max = bdry_max.get('right', iy, ix - 1)
        nb_cnt = bdry_cnt.get('right', iy, ix - 1)
        h = len(nb_fdir)
        for i in range(h):
            if nb_mask[i] == 0 or nb_max[i] == 0:
                continue
            theta = nb_fdir[i]
            if theta != theta or theta < 0.0:
                continue
            k1, f1, k2, f2 = _dinf_downstream(theta)
            if k1 < 0:
                continue
            val = _reconstruct_order(nb_max[i], nb_cnt[i])
            for kk, frac in [(k1, f1), (k2, f2)]:
                if frac <= 0.0:
                    continue
                dy = int(nb_dy[kk])
                dx = int(nb_dx[kk])
                if dx == 1:  # flows east
                    tr = i + dy
                    if 0 <= tr < tile_h:
                        _update_max_cnt(smax_left, scnt_left, val, tr)

    # Right edge: left column of tile to the right
    if ix < n_tile_x - 1:
        nb_fdir = flow_bdry.get('left', iy, ix + 1)
        nb_mask = mask_bdry.get('left', iy, ix + 1)
        nb_max = bdry_max.get('left', iy, ix + 1)
        nb_cnt = bdry_cnt.get('left', iy, ix + 1)
        h = len(nb_fdir)
        for i in range(h):
            if nb_mask[i] == 0 or nb_max[i] == 0:
                continue
            theta = nb_fdir[i]
            if theta != theta or theta < 0.0:
                continue
            k1, f1, k2, f2 = _dinf_downstream(theta)
            if k1 < 0:
                continue
            val = _reconstruct_order(nb_max[i], nb_cnt[i])
            for kk, frac in [(k1, f1), (k2, f2)]:
                if frac <= 0.0:
                    continue
                dy = int(nb_dy[kk])
                dx = int(nb_dx[kk])
                if dx == -1:  # flows west
                    tr = i + dy
                    if 0 <= tr < tile_h:
                        _update_max_cnt(smax_right, scnt_right, val, tr)

    # Corner seeds
    def _corner_strahler(nb_side, nb_iy, nb_ix, nb_idx,
                         required_dy, required_dx,
                         s_max, s_cnt, target):
        fdir = flow_bdry.get(nb_side, nb_iy, nb_ix)[nb_idx]
        mask = mask_bdry.get(nb_side, nb_iy, nb_ix)[nb_idx]
        if mask == 0:
            return
        nm = bdry_max.get(nb_side, nb_iy, nb_ix)[nb_idx]
        if nm == 0:
            return
        if fdir != fdir or fdir < 0.0:
            return
        k1, f1, k2, f2 = _dinf_downstream(fdir)
        if k1 < 0:
            return
        nc = bdry_cnt.get(nb_side, nb_iy, nb_ix)[nb_idx]
        val = _reconstruct_order(nm, nc)
        for kk, frac in [(k1, f1), (k2, f2)]:
            if frac <= 0.0:
                continue
            dy = int(nb_dy[kk])
            dx = int(nb_dx[kk])
            if dy == required_dy and dx == required_dx:
                _update_max_cnt(s_max, s_cnt, val, target)

    # Top-left corner
    if iy > 0 and ix > 0:
        _corner_strahler('bottom', iy - 1, ix - 1, -1,
                         1, 1, smax_top, scnt_top, 0)
    # Top-right corner
    if iy > 0 and ix < n_tile_x - 1:
        _corner_strahler('bottom', iy - 1, ix + 1, 0,
                         1, -1, smax_top, scnt_top, tile_w - 1)
    # Bottom-left corner
    if iy < n_tile_y - 1 and ix > 0:
        _corner_strahler('top', iy + 1, ix - 1, -1,
                         -1, 1, smax_bottom, scnt_bottom, 0)
    # Bottom-right corner
    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        _corner_strahler('top', iy + 1, ix + 1, 0,
                         -1, -1, smax_bottom, scnt_bottom, tile_w - 1)

    return (smax_top, scnt_top, smax_bottom, scnt_bottom,
            smax_left, scnt_left, smax_right, scnt_right)


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _make_stream_mask_np_dinf(ac_chunk, fd_chunk, threshold):
    """Build stream mask as numpy int8 from accumulation and D-inf angles."""
    sm = np.where(ac_chunk >= threshold, 1, 0).astype(np.int8)
    sm = np.where(np.isnan(ac_chunk), 0, sm).astype(np.int8)
    sm = np.where(np.isnan(fd_chunk), 0, sm).astype(np.int8)
    return sm


def _process_strahler_tile_dinf(iy, ix, flow_dir_da, accum_da, threshold,
                                 bdry_max, bdry_cnt, flow_bdry, mask_bdry,
                                 chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded Strahler BFS on one D-inf tile; update boundary stores."""
    fd_chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    ac_chunk = np.asarray(
        accum_da.blocks[iy, ix].compute(), dtype=np.float64)
    sm = _make_stream_mask_np_dinf(ac_chunk, fd_chunk, threshold)
    h, w = fd_chunk.shape

    seeds = _compute_strahler_seeds_dinf(
        iy, ix, bdry_max, bdry_cnt, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    order = _strahler_dinf_tile_kernel(fd_chunk, sm, h, w, *seeds)

    change = 0.0
    for side, strip in [('top', order[0, :]),
                        ('bottom', order[-1, :]),
                        ('left', order[:, 0]),
                        ('right', order[:, -1])]:
        new_vals = strip.copy()
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


def _process_shreve_tile_dinf(iy, ix, flow_dir_da, accum_da, threshold,
                               boundaries, flow_bdry, mask_bdry,
                               chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run seeded Shreve BFS on one D-inf tile; update boundaries."""
    fd_chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    ac_chunk = np.asarray(
        accum_da.blocks[iy, ix].compute(), dtype=np.float64)
    sm = _make_stream_mask_np_dinf(ac_chunk, fd_chunk, threshold)
    h, w = fd_chunk.shape

    seeds = _compute_shreve_seeds_dinf(
        iy, ix, boundaries, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    order = _shreve_dinf_tile_kernel(fd_chunk, sm, h, w, *seeds)

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


def _stream_order_dinf_dask_strahler(flow_dir_da, accum_da, threshold):
    """Dask iterative sweep for D-inf Strahler ordering."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry, mask_bdry = _preprocess_stream_tiles_dinf(
        flow_dir_da, accum_da, threshold, chunks_y, chunks_x)
    flow_bdry = flow_bdry.snapshot()
    mask_bdry = mask_bdry.snapshot()

    bdry_max = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)
    bdry_cnt = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

    max_iterations = max(n_tile_y, n_tile_x) + 10
    for _ in range(max_iterations):
        max_change = 0.0
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_strahler_tile_dinf(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    bdry_max, bdry_cnt, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_strahler_tile_dinf(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    bdry_max, bdry_cnt, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        if max_change == 0.0:
            break

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
        sm = _make_stream_mask_np_dinf(ac, fd, _threshold)
        h, w = fd.shape
        seeds = _compute_strahler_seeds_dinf(
            iy, ix, _bdry_max, _bdry_cnt, _flow_bdry, _mask_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _strahler_dinf_tile_kernel(fd, sm, h, w, *seeds)

    return da.map_blocks(
        _tile_fn, flow_dir_da, accum_da,
        dtype=np.float64, meta=np.array((), dtype=np.float64))


def _stream_order_dinf_dask_shreve(flow_dir_da, accum_da, threshold):
    """Dask iterative sweep for D-inf Shreve ordering."""
    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

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
                c = _process_shreve_tile_dinf(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    boundaries, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c > max_change:
                    max_change = c
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_shreve_tile_dinf(
                    iy, ix, flow_dir_da, accum_da, threshold,
                    boundaries, flow_bdry, mask_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
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
        seeds = _compute_shreve_seeds_dinf(
            iy, ix, _boundaries, _flow_bdry, _mask_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _shreve_dinf_tile_kernel(fd, sm, h, w, *seeds)

    return da.map_blocks(
        _tile_fn, flow_dir_da, accum_da,
        dtype=np.float64, meta=np.array((), dtype=np.float64))


# =====================================================================
# Dask + CuPy
# =====================================================================

def _stream_order_dinf_tile_cupy(angles_data, stream_mask_data, method,
                                  seeds):
    """GPU seeded D-inf stream order for a single tile.

    Uses GPU frontier peeling with seeds injected after initialisation.
    For Strahler, seeds are (max_top, cnt_top, max_bot, cnt_bot, ...).
    For Shreve, seeds are (top, bot, left, right).
    """
    import cupy as cp

    H, W = angles_data.shape
    angles_f64 = angles_data.astype(cp.float64)
    stream_mask_i8 = stream_mask_data.astype(cp.int8)

    in_degree = cp.zeros((H, W), dtype=cp.int32)
    state = cp.zeros((H, W), dtype=cp.int32)
    order = cp.zeros((H, W), dtype=cp.float64)
    max_in = cp.zeros((H, W), dtype=cp.float64)
    cnt_max = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _stream_order_dinf_init_gpu[griddim, blockdim](
        angles_f64, stream_mask_i8, in_degree, state,
        order, max_in, cnt_max, H, W)

    if method == 'strahler':
        (smax_top, scnt_top, smax_bot, scnt_bot,
         smax_left, scnt_left, smax_right, scnt_right) = seeds
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
            max_in[r_idx] = cp.where(mask, sm_cp, max_in[r_idx])
            cnt_max[r_idx] = cp.where(mask, sc_cp, cnt_max[r_idx])
    else:
        (seed_top, seed_bot, seed_left, seed_right) = seeds
        order[0, :] += cp.asarray(seed_top)
        order[H - 1, :] += cp.asarray(seed_bot)
        order[:, 0] += cp.asarray(seed_left)
        order[:, W - 1] += cp.asarray(seed_right)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _stream_order_dinf_find_ready[griddim, blockdim](
            in_degree, state, order, changed, H, W)

        if int(changed[0]) == 0:
            break

        if method == 'strahler':
            _stream_order_dinf_pull_strahler[griddim, blockdim](
                angles_f64, stream_mask_i8, in_degree, state,
                order, max_in, cnt_max, H, W)
        else:
            _stream_order_dinf_pull_shreve[griddim, blockdim](
                angles_f64, stream_mask_i8, in_degree, state,
                order, H, W)

    order = cp.where(stream_mask_i8 == 0, cp.nan, order)
    return order


def _process_strahler_tile_dinf_cupy(iy, ix, flow_dir_da, accum_da,
                                      threshold,
                                      bdry_max, bdry_cnt,
                                      flow_bdry, mask_bdry,
                                      chunks_y, chunks_x,
                                      n_tile_y, n_tile_x):
    """Run seeded GPU D-inf Strahler on one tile; update boundaries."""
    import cupy as cp

    fd_chunk = cp.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=cp.float64)
    ac_chunk = _to_numpy_f64(accum_da.blocks[iy, ix].compute())
    fd_np = fd_chunk.get()
    sm_np = _make_stream_mask_np_dinf(ac_chunk, fd_np, threshold)
    sm_cp = cp.asarray(sm_np)

    seeds = _compute_strahler_seeds_dinf(
        iy, ix, bdry_max, bdry_cnt, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    order = _stream_order_dinf_tile_cupy(fd_chunk, sm_cp, 'strahler', seeds)

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


def _process_shreve_tile_dinf_cupy(iy, ix, flow_dir_da, accum_da,
                                    threshold,
                                    boundaries, flow_bdry, mask_bdry,
                                    chunks_y, chunks_x,
                                    n_tile_y, n_tile_x):
    """Run seeded GPU D-inf Shreve on one tile; update boundaries."""
    import cupy as cp

    fd_chunk = cp.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=cp.float64)
    ac_chunk = _to_numpy_f64(accum_da.blocks[iy, ix].compute())
    fd_np = fd_chunk.get()
    sm_np = _make_stream_mask_np_dinf(ac_chunk, fd_np, threshold)
    sm_cp = cp.asarray(sm_np)

    seeds = _compute_shreve_seeds_dinf(
        iy, ix, boundaries, flow_bdry, mask_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    order = _stream_order_dinf_tile_cupy(fd_chunk, sm_cp, 'shreve', seeds)

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


def _stream_order_dinf_dask_cupy(flow_dir_da, accum_da, threshold, method):
    """Dask+CuPy: native GPU processing per tile for D-inf stream order."""
    import cupy as cp

    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    flow_bdry, mask_bdry = _preprocess_stream_tiles_dinf(
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
                    c = _process_strahler_tile_dinf_cupy(
                        iy, ix, flow_dir_da, accum_da, threshold,
                        bdry_max, bdry_cnt, flow_bdry, mask_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x)
                    if c > max_change:
                        max_change = c
            for iy in reversed(range(n_tile_y)):
                for ix in reversed(range(n_tile_x)):
                    c = _process_strahler_tile_dinf_cupy(
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
            sm = cp.asarray(_make_stream_mask_np_dinf(ac_np, fd_np, threshold))
            seeds = _compute_strahler_seeds_dinf(
                iy, ix, _bdry_max, _bdry_cnt, flow_bdry, mask_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)
            return _stream_order_dinf_tile_cupy(
                fd, sm, 'strahler', seeds)

    else:  # shreve
        boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=0.0)

        for _ in range(max_iterations):
            max_change = 0.0
            for iy in range(n_tile_y):
                for ix in range(n_tile_x):
                    c = _process_shreve_tile_dinf_cupy(
                        iy, ix, flow_dir_da, accum_da, threshold,
                        boundaries, flow_bdry, mask_bdry,
                        chunks_y, chunks_x, n_tile_y, n_tile_x)
                    if c > max_change:
                        max_change = c
            for iy in reversed(range(n_tile_y)):
                for ix in reversed(range(n_tile_x)):
                    c = _process_shreve_tile_dinf_cupy(
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
            sm = cp.asarray(_make_stream_mask_np_dinf(ac_np, fd_np, threshold))
            seeds = _compute_shreve_seeds_dinf(
                iy, ix, _boundaries, flow_bdry, mask_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)
            return _stream_order_dinf_tile_cupy(
                fd, sm, 'shreve', seeds)

    return da.map_blocks(
        _tile_fn, flow_dir_da, accum_da,
        dtype=np.float64, meta=cp.array((), dtype=cp.float64))


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def stream_order_dinf(flow_dir_dinf: xr.DataArray,
                      flow_accum: xr.DataArray,
                      threshold: float = 100,
                      method: str = 'strahler',
                      name: str = 'stream_order_dinf') -> xr.DataArray:
    """Compute stream order from D-infinity flow direction and accumulation.

    Parameters
    ----------
    flow_dir_dinf : xarray.DataArray or xr.Dataset
        2D D-infinity flow direction grid.  Values are continuous
        angles in radians in the range ``[0, 2*pi)``.  ``-1.0``
        indicates a pit or flat.  ``NaN`` indicates nodata.
    flow_accum : xarray.DataArray
        2D flow accumulation grid.  Cells with
        ``flow_accum >= threshold`` are considered stream cells.
    threshold : float, default 100
        Minimum accumulation to classify a cell as part of the
        stream network.
    method : str, default 'strahler'
        ``'strahler'`` for Strahler branching hierarchy or
        ``'shreve'`` for Shreve cumulative magnitude.
    name : str, default 'stream_order_dinf'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array of stream order values.  Non-stream cells
        (accumulation below threshold) are NaN.

    References
    ----------
    Tarboton, D.G. (1997). A new method for the determination of flow
    directions and upslope areas in grid digital elevation models.
    Water Resources Research, 33(2), 309-319.

    Strahler, A.N. (1957). Quantitative analysis of watershed
    geomorphology. Transactions of the American Geophysical Union,
    38(6), 913-920.

    Shreve, R.L. (1966). Statistical law of stream numbers. Journal
    of Geology, 74(1), 17-37.
    """
    _validate_raster(flow_dir_dinf,
                     func_name='stream_order_dinf',
                     name='flow_dir_dinf')
    _validate_raster(flow_accum,
                     func_name='stream_order_dinf',
                     name='flow_accum')

    method = method.lower()
    if method not in ('strahler', 'shreve'):
        raise ValueError(
            f"method must be 'strahler' or 'shreve', got {method!r}")

    fd_data = flow_dir_dinf.data
    fa_data = flow_accum.data

    if isinstance(fd_data, np.ndarray):
        _check_memory(*fd_data.shape)
        fd = fd_data.astype(np.float64)
        fa = np.asarray(fa_data, dtype=np.float64)
        stream_mask = np.where(fa >= threshold, 1, 0).astype(np.int8)
        stream_mask = np.where(np.isnan(fa), 0, stream_mask).astype(np.int8)
        stream_mask = np.where(np.isnan(fd), 0, stream_mask).astype(np.int8)
        h, w = fd.shape
        if method == 'strahler':
            out = _strahler_dinf_cpu(fd, stream_mask, h, w)
        else:
            out = _shreve_dinf_cpu(fd, stream_mask, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(fd_data):
        _check_gpu_memory(*fd_data.shape)
        import cupy as cp
        fa_cp = cp.asarray(fa_data, dtype=cp.float64)
        fd_cp = fd_data.astype(cp.float64)
        stream_mask = cp.where(fa_cp >= threshold, 1, 0).astype(cp.int8)
        stream_mask = cp.where(
            cp.isnan(fa_cp), 0, stream_mask).astype(cp.int8)
        stream_mask = cp.where(
            cp.isnan(fd_cp), 0, stream_mask).astype(cp.int8)
        out = _stream_order_dinf_cupy(fd_cp, stream_mask, method)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir_dinf):
        out = _stream_order_dinf_dask_cupy(
            fd_data, fa_data, threshold, method)

    elif da is not None and isinstance(fd_data, da.Array):
        if method == 'strahler':
            out = _stream_order_dinf_dask_strahler(
                fd_data, fa_data, threshold)
        else:
            out = _stream_order_dinf_dask_shreve(
                fd_data, fa_data, threshold)

    else:
        raise TypeError(f"Unsupported array type: {type(fd_data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir_dinf.coords,
                        dims=flow_dir_dinf.dims,
                        attrs=flow_dir_dinf.attrs)
