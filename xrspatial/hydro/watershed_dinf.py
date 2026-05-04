"""D-infinity watershed delineation.

Labels each cell with the pour point it drains to, using D-inf
dominant-neighbor downstream tracing with path compression.

Algorithm
---------
CPU : downstream tracing with path compression, using the dominant
      neighbor from D-inf angle decomposition.
GPU : CuPy-via-CPU.
Dask: iterative tile sweep with exit-label propagation.
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

try:
    import cupy
except ImportError:
    class cupy:  # type: ignore[no-redef]
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.hydro.flow_accumulation_dinf import _angle_to_neighbors
from xrspatial.hydro.flow_path_dinf import _angle_to_neighbors_py
from xrspatial.hydro._boundary_store import BoundaryStore
from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)
from xrspatial.dataset_support import supports_dataset


def _to_numpy_f64(arr):
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


# =====================================================================
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for the numpy dispatch + the
# ``_watershed_dinf_cpu`` kernel:
#   fd (float64 cast)    -> 8
#   labels (float64)     -> 8
#   state  (int8)        -> 1
#   path_r (int64)       -> 8
#   path_c (int64)       -> 8
# Total ~33 bytes/pixel.  D-inf encodes its downstream direction as a
# single real-valued angle, so the per-pixel footprint matches D8 rather
# than MFD's eight-channel fractions buffer.
_BYTES_PER_PIXEL = 33

# GPU peak working set per pixel for ``_watershed_dinf_cupy``.  The
# function copies the device flow_dir to the host, runs the CPU kernel,
# and ships the result back.  Device-resident peak is the caller's
# float64 flow_dir input (8) plus the caller's float64 pour_points (8)
# plus the final ``cp.asarray(out)`` (8) -> 24 bytes/pixel.
_GPU_BYTES_PER_PIXEL = 24


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
    """Raise MemoryError if the watershed_dinf kernel would exceed 50% of RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"watershed_dinf on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of working memory but only "
            f"~{available / 1e9:.1f} GB is available.  Use a "
            f"dask-backed DataArray for out-of-core processing."
        )


def _check_gpu_memory(height, width):
    """Raise MemoryError if the cupy path would exceed 50% of free GPU RAM.

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
            f"watershed_dinf on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


def _dominant_offset_py(angle):
    """Return (dy, dx) for the dominant D-inf neighbor, or (0,0) for pit/NaN."""
    dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors_py(angle)
    if w1 <= 0.0 and w2 <= 0.0:
        return (0, 0)
    if w1 >= w2:
        return (dy1, dx1)
    return (dy2, dx2)


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _watershed_dinf_cpu(flow_dir, labels, state, h, w):
    """Downstream tracing with path compression for D-inf watershed.

    State: 0=nodata, 1=unresolved, 2=in-trace, 3=resolved.
    """
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

                angle = flow_dir[cr, cc]
                if angle != angle:  # NaN
                    break
                dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(angle)
                if w1 <= 0.0 and w2 <= 0.0:
                    break
                if w1 >= w2:
                    dy, dx = dy1, dx1
                else:
                    dy, dx = dy2, dx2
                nr, nc = cr + dy, cc + dx
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    break
                cr, cc = nr, nc

            for i in range(path_len):
                if found:
                    labels[path_r[i], path_c[i]] = found_label
                    state[path_r[i], path_c[i]] = 3
                else:
                    labels[path_r[i], path_c[i]] = np.nan
                    state[path_r[i], path_c[i]] = 0

    return labels


# =====================================================================
# CuPy backend
# =====================================================================

def _watershed_dinf_cupy(flow_dir_data, pour_points_data):
    import cupy as cp
    fd_np = _to_numpy_f64(flow_dir_data)
    pp_np = _to_numpy_f64(pour_points_data)
    h, w = fd_np.shape
    labels = np.full((h, w), np.nan, dtype=np.float64)
    state = np.zeros((h, w), dtype=np.int8)
    for r in range(h):
        for c in range(w):
            if fd_np[r, c] != fd_np[r, c]:
                pass
            elif pp_np[r, c] == pp_np[r, c]:
                labels[r, c] = pp_np[r, c]
                state[r, c] = 3
            else:
                state[r, c] = 1
    out = _watershed_dinf_cpu(fd_np, labels, state, h, w)
    return cp.asarray(out)


# =====================================================================
# Dask tile kernel
# =====================================================================

@ngjit
def _watershed_dinf_tile_kernel(flow_dir, h, w, pour_points,
                                 exit_top, exit_bottom, exit_left, exit_right,
                                 exit_tl, exit_tr, exit_bl, exit_br):
    """Seeded downstream tracing for a D-inf tile."""
    labels = np.empty((h, w), dtype=np.float64)
    state = np.empty((h, w), dtype=np.int8)

    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v != v:
                labels[r, c] = np.nan
                state[r, c] = 0
                continue
            pp = pour_points[r, c]
            if pp == pp:
                labels[r, c] = pp
                state[r, c] = 3
                continue
            labels[r, c] = np.nan
            state[r, c] = 1

    # Apply exit labels at boundaries
    for c in range(w):
        if state[0, c] == 1:
            el = exit_top[c]
            if el == el:
                labels[0, c] = el
                state[0, c] = 3
    for c in range(w):
        if state[h - 1, c] == 1:
            el = exit_bottom[c]
            if el == el:
                labels[h - 1, c] = el
                state[h - 1, c] = 3
    for r in range(h):
        if state[r, 0] == 1:
            el = exit_left[r]
            if el == el:
                labels[r, 0] = el
                state[r, 0] = 3
    for r in range(h):
        if state[r, w - 1] == 1:
            el = exit_right[r]
            if el == el:
                labels[r, w - 1] = el
                state[r, w - 1] = 3

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

                angle = flow_dir[cr, cc]
                if angle != angle:
                    break
                dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(angle)
                if w1 <= 0.0 and w2 <= 0.0:
                    break
                if w1 >= w2:
                    dy, dx = dy1, dx1
                else:
                    dy, dx = dy2, dx2
                nr, nc = cr + dy, cc + dx
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    exit_tile = True
                    break
                cr, cc = nr, nc

            for i in range(path_len):
                if found:
                    labels[path_r[i], path_c[i]] = found_label
                    state[path_r[i], path_c[i]] = 3
                elif exit_tile:
                    state[path_r[i], path_c[i]] = 1
                else:
                    labels[path_r[i], path_c[i]] = np.nan
                    state[path_r[i], path_c[i]] = 0

    return labels


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _preprocess_tiles(flow_dir_da, chunks_y, chunks_x):
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
    """Compute exit labels for D-inf tile using dominant neighbor."""
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
            d = _dominant_offset_py(float(fdir_top[j]))
            if d[0] == -1:
                dj = j + d[1]
                if d[1] == 0 and 0 <= dj < len(nb_labels):
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
            d = _dominant_offset_py(float(fdir_bot[j]))
            if d[0] == 1:
                dj = j + d[1]
                if d[1] == 0 and 0 <= dj < len(nb_labels):
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
            d = _dominant_offset_py(float(fdir_left[r]))
            if d[1] == -1:
                dr = r + d[0]
                if d[0] == 0 and 0 <= dr < len(nb_labels):
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
            d = _dominant_offset_py(float(fdir_right[r]))
            if d[1] == 1:
                dr = r + d[0]
                if d[0] == 0 and 0 <= dr < len(nb_labels):
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
            d = _dominant_offset_py(float(fdir_top[j]))
            if d[0] == -1:
                exit_top[j] = np.nan
    if iy == n_tile_y - 1:
        fdir_bot = flow_bdry.get('bottom', iy, ix)
        for j in range(tile_w):
            d = _dominant_offset_py(float(fdir_bot[j]))
            if d[0] == 1:
                exit_bottom[j] = np.nan
    if ix == 0:
        fdir_left = flow_bdry.get('left', iy, ix)
        for r in range(tile_h):
            d = _dominant_offset_py(float(fdir_left[r]))
            if d[1] == -1:
                exit_left[r] = np.nan
    if ix == n_tile_x - 1:
        fdir_right = flow_bdry.get('right', iy, ix)
        for r in range(tile_h):
            d = _dominant_offset_py(float(fdir_right[r]))
            if d[1] == 1:
                exit_right[r] = np.nan

    # Diagonal corners
    fdir_tl = flow_bdry.get('top', iy, ix)[0]
    d = _dominant_offset_py(float(fdir_tl))
    if d == (-1, -1):
        if iy > 0 and ix > 0:
            exit_tl = boundaries.get('bottom', iy - 1, ix - 1)[-1]
        else:
            exit_tl = np.nan

    fdir_tr = flow_bdry.get('top', iy, ix)[-1]
    d = _dominant_offset_py(float(fdir_tr))
    if d == (-1, 1):
        if iy > 0 and ix < n_tile_x - 1:
            exit_tr = boundaries.get('bottom', iy - 1, ix + 1)[0]
        else:
            exit_tr = np.nan

    fdir_bl = flow_bdry.get('bottom', iy, ix)[0]
    d = _dominant_offset_py(float(fdir_bl))
    if d == (1, -1):
        if iy < n_tile_y - 1 and ix > 0:
            exit_bl = boundaries.get('top', iy + 1, ix - 1)[-1]
        else:
            exit_bl = np.nan

    fdir_br = flow_bdry.get('bottom', iy, ix)[-1]
    d = _dominant_offset_py(float(fdir_br))
    if d == (1, 1):
        if iy < n_tile_y - 1 and ix < n_tile_x - 1:
            exit_br = boundaries.get('top', iy + 1, ix + 1)[0]
        else:
            exit_br = np.nan

    return (exit_top, exit_bottom, exit_left, exit_right,
            exit_tl, exit_tr, exit_bl, exit_br)


def _process_tile(iy, ix, flow_dir_da, pour_points_da,
                   boundaries, flow_bdry,
                   chunks_y, chunks_x, n_tile_y, n_tile_x):
    chunk = np.asarray(
        flow_dir_da.blocks[iy, ix].compute(), dtype=np.float64)
    pp_chunk = np.asarray(
        pour_points_da.blocks[iy, ix].compute(), dtype=np.float64)
    h, w = chunk.shape

    exits = _compute_exit_labels(
        iy, ix, boundaries, flow_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    result = _watershed_dinf_tile_kernel(chunk, h, w, pp_chunk, *exits)

    new_top = result[0, :].copy()
    new_bottom = result[-1, :].copy()
    new_left = result[:, 0].copy()
    new_right = result[:, -1].copy()

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


def _watershed_dinf_dask(flow_dir_da, pour_points_da):
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
                c = _process_tile(
                    iy, ix, flow_dir_da, pour_points_da,
                    boundaries, flow_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile(
                    iy, ix, flow_dir_da, pour_points_da,
                    boundaries, flow_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True

        if not any_changed:
            break

    boundaries = boundaries.snapshot()

    def _tile_fn(flow_dir_block, pp_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(flow_dir_block.shape, np.nan, dtype=np.float64)
        iy, ix = block_info[0]['chunk-location']
        h, w = flow_dir_block.shape
        exits = _compute_exit_labels(
            iy, ix, boundaries, flow_bdry,
            chunks_y, chunks_x, n_tile_y, n_tile_x)
        return _watershed_dinf_tile_kernel(
            np.asarray(flow_dir_block, dtype=np.float64),
            h, w,
            np.asarray(pp_block, dtype=np.float64),
            *exits)

    return da.map_blocks(
        _tile_fn,
        flow_dir_da, pour_points_da,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


# =====================================================================
# Dask+CuPy backend
# =====================================================================

def _watershed_dinf_dask_cupy(flow_dir_da, pour_points_da):
    import cupy as cp
    fd_np = flow_dir_da.map_blocks(
        lambda b: b.get(), dtype=flow_dir_da.dtype,
        meta=np.array((), dtype=flow_dir_da.dtype),
    )
    pp_np = pour_points_da.map_blocks(
        lambda b: b.get(), dtype=pour_points_da.dtype,
        meta=np.array((), dtype=pour_points_da.dtype),
    )
    result = _watershed_dinf_dask(fd_np, pp_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def watershed_dinf(flow_dir_dinf: xr.DataArray,
                   pour_points: xr.DataArray,
                   name: str = 'watershed_dinf') -> xr.DataArray:
    """Label each cell with the pour point it drains to (D-inf).

    Parameters
    ----------
    flow_dir_dinf : xarray.DataArray or xr.Dataset
        2D D-infinity flow direction grid.
    pour_points : xarray.DataArray
        2D raster where non-NaN cells are pour points.
    name : str, default='watershed_dinf'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array where each cell = label of its pour point.
        NaN for nodata or unreachable cells.
    """
    _validate_raster(flow_dir_dinf, func_name='watershed_dinf',
                     name='flow_dir_dinf')
    _validate_raster(pour_points, func_name='watershed_dinf',
                     name='pour_points')

    data = flow_dir_dinf.data
    pp_data = pour_points.data

    if isinstance(data, np.ndarray):
        _check_memory(*data.shape)
        fd = data.astype(np.float64)
        pp = np.asarray(pp_data, dtype=np.float64)
        h, w = fd.shape
        labels = np.full((h, w), np.nan, dtype=np.float64)
        state = np.zeros((h, w), dtype=np.int8)
        for r in range(h):
            for c in range(w):
                if fd[r, c] != fd[r, c]:
                    pass
                elif pp[r, c] == pp[r, c]:
                    labels[r, c] = pp[r, c]
                    state[r, c] = 3
                else:
                    state[r, c] = 1
        out = _watershed_dinf_cpu(fd, labels, state, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        _check_gpu_memory(*data.shape)
        out = _watershed_dinf_cupy(data, pp_data)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir_dinf):
        out = _watershed_dinf_dask_cupy(data, pp_data)

    elif da is not None and isinstance(data, da.Array):
        out = _watershed_dinf_dask(data, pp_data)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir_dinf.coords,
                        dims=flow_dir_dinf.dims,
                        attrs=flow_dir_dinf.attrs)
