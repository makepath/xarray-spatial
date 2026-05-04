"""Trace downstream flow paths from start points through an MFD fraction grid.

Uses the dominant-neighbor approach: at each cell, the neighbor with the
highest fraction is followed. Returns a single path per start point.

Algorithm
---------
For each non-NaN cell in ``start_points``:
1. Find the neighbor direction k with the highest fraction.
2. Follow that neighbor at each step.
3. Write the start cell's label to every visited cell.
4. Stop at NaN, pit (all fractions zero), out-of-bounds, or grid edge.
"""

from __future__ import annotations

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

from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)
from xrspatial.dataset_support import supports_dataset

# Neighbor offsets: E, SE, S, SW, W, NW, N, NE
_DY_NP = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
_DX_NP = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)


# =====================================================================
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for ``flow_path_mfd`` numpy dispatch:
#   data.astype(float64) copy of (8, H, W) fractions -> 64
#   np.asarray(sp_data, dtype=float64) copy          -> 8
#   out (H, W) float64                               -> 8
# Total ~80 B/px.
_BYTES_PER_PIXEL = 80

# GPU peak working set per pixel for ``_flow_path_mfd_cupy``:
#   host fr_np = data.get()       -> 64
#   host sp_np                    -> 8
#   host out (H, W) float64       -> 8
#   device output                 -> 8
# Total ~88 B/px (conservative, treating host residency as the bound).
_GPU_BYTES_PER_PIXEL = 88


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
            f"flow_path_mfd on a {height}x{width} grid requires "
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
            f"flow_path_mfd on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


def _dominant_neighbor_py(fractions, r, c):
    """Pure-Python: find dominant neighbor direction and offset.

    Returns (dy, dx, frac) or (0, 0, 0.0) if pit/nodata.
    """
    dy_arr = [0, 1, 1, 1, 0, -1, -1, -1]
    dx_arr = [1, 1, 0, -1, -1, -1, 0, 1]
    best_k = -1
    best_frac = 0.0
    for k in range(8):
        f = float(fractions[k, r, c])
        if f > best_frac:
            best_frac = f
            best_k = k
    if best_k == -1:
        return 0, 0, 0.0
    return dy_arr[best_k], dx_arr[best_k], best_frac


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _flow_path_mfd_cpu(fractions, start_points, H, W):
    """Trace downstream paths using MFD dominant neighbor."""
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    out = np.empty((H, W), dtype=np.float64)
    out[:] = np.nan

    for r in range(H):
        for c in range(W):
            v = start_points[r, c]
            if v != v:  # NaN
                continue
            label = v
            cr, cc = r, c
            while True:
                out[cr, cc] = label
                # Check if cell is valid
                chk = fractions[0, cr, cc]
                if chk != chk:  # NaN → nodata
                    break
                # Find dominant neighbor
                best_k = -1
                best_frac = 0.0
                for k in range(8):
                    f = fractions[k, cr, cc]
                    if f > best_frac:
                        best_frac = f
                        best_k = k
                if best_k == -1:
                    break  # pit
                nr = cr + dy[best_k]
                nc = cc + dx[best_k]
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                cr, cc = nr, nc

    return out


# =====================================================================
# CuPy backend
# =====================================================================

def _flow_path_mfd_cupy(fractions_data, start_points_data):
    """CuPy: convert to numpy, run CPU kernel, convert back."""
    import cupy as cp

    fr_np = fractions_data.get() if hasattr(fractions_data, 'get') else np.asarray(fractions_data)
    sp_np = start_points_data.get() if hasattr(start_points_data, 'get') else np.asarray(start_points_data)
    fr_np = fr_np.astype(np.float64)
    sp_np = sp_np.astype(np.float64)
    _, H, W = fr_np.shape
    out = _flow_path_mfd_cpu(fr_np, sp_np, H, W)
    return cp.asarray(out)


# =====================================================================
# Dask backend
# =====================================================================

def _flow_path_mfd_dask(fractions_data, start_points_data, chunks_y, chunks_x):
    """Dask: sparse start-point extraction, LRU-cached tracing, lazy assembly."""
    from xrspatial.hydro.flow_path_d8 import _group_cells_by_chunk
    from functools import lru_cache

    _, H, W = fractions_data.shape

    # Phase 1: identify chunks with start points
    sp_chunks_y = start_points_data.chunks[0]
    sp_chunks_x = start_points_data.chunks[1]

    def _has_sp(block):
        return np.array(
            [[np.any(~np.isnan(np.asarray(block))).item()]],
            dtype=np.int8,
        )

    flags = da.map_blocks(
        _has_sp, start_points_data,
        dtype=np.int8,
        chunks=tuple((1,) * len(c) for c in start_points_data.chunks),
    ).compute()

    # Phase 2: extract start points
    points = []
    row_off = 0
    for iy, cy in enumerate(sp_chunks_y):
        col_off = 0
        for ix, cx in enumerate(sp_chunks_x):
            if flags[iy, ix]:
                chunk = np.asarray(
                    start_points_data.blocks[iy, ix].compute(),
                    dtype=np.float64,
                )
                rs, cs = np.where(~np.isnan(chunk))
                for k in range(len(rs)):
                    points.append((
                        row_off + int(rs[k]),
                        col_off + int(cs[k]),
                        float(chunk[rs[k], cs[k]]),
                    ))
            col_off += cx
        row_off += cy

    # Phase 3: trace paths with LRU cache
    fd_row_offsets = np.zeros(len(chunks_y) + 1, dtype=np.int64)
    for i, cy in enumerate(chunks_y):
        fd_row_offsets[i + 1] = fd_row_offsets[i] + cy
    fd_col_offsets = np.zeros(len(chunks_x) + 1, dtype=np.int64)
    for i, cx in enumerate(chunks_x):
        fd_col_offsets[i + 1] = fd_col_offsets[i] + cx

    max_chunk_bytes = max(
        8 * int(cy) * int(cx) * 8  # 8 bands * tile * 8 bytes
        for cy in chunks_y for cx in chunks_x
    )
    cache_size = max(4, (512 * 1024 * 1024) // max(max_chunk_bytes, 1))

    @lru_cache(maxsize=cache_size)
    def _get_chunk(iy, ix):
        y_start = int(fd_row_offsets[iy])
        y_end = int(fd_row_offsets[iy + 1])
        x_start = int(fd_col_offsets[ix])
        x_end = int(fd_col_offsets[ix + 1])
        return np.asarray(
            fractions_data[:, y_start:y_end, x_start:x_end].compute(),
            dtype=np.float64,
        )

    def _find_chunk(r, c):
        iy = int(np.searchsorted(fd_row_offsets[1:], r, side='right'))
        ix = int(np.searchsorted(fd_col_offsets[1:], c, side='right'))
        return iy, ix, r - int(fd_row_offsets[iy]), c - int(fd_col_offsets[ix])

    _init_cap = max(1024, len(points) * 4)
    _buf_rows = np.empty(_init_cap, dtype=np.int64)
    _buf_cols = np.empty(_init_cap, dtype=np.int64)
    _buf_labels = np.empty(_init_cap, dtype=np.float64)
    _buf_len = 0

    dy_arr = [0, 1, 1, 1, 0, -1, -1, -1]
    dx_arr = [1, 1, 0, -1, -1, -1, 0, 1]

    for r, c, label in points:
        cr, cc = r, c
        while True:
            if _buf_len >= len(_buf_rows):
                new_cap = len(_buf_rows) * 2
                _new_rows = np.empty(new_cap, dtype=np.int64)
                _new_rows[:_buf_len] = _buf_rows[:_buf_len]
                _buf_rows = _new_rows
                _new_cols = np.empty(new_cap, dtype=np.int64)
                _new_cols[:_buf_len] = _buf_cols[:_buf_len]
                _buf_cols = _new_cols
                _new_labels = np.empty(new_cap, dtype=np.float64)
                _new_labels[:_buf_len] = _buf_labels[:_buf_len]
                _buf_labels = _new_labels

            _buf_rows[_buf_len] = cr
            _buf_cols[_buf_len] = cc
            _buf_labels[_buf_len] = label
            _buf_len += 1

            iy, ix, lr, lc = _find_chunk(cr, cc)
            chunk = _get_chunk(iy, ix)
            # Check valid
            if np.isnan(chunk[0, lr, lc]):
                break
            # Find dominant neighbor
            best_k = -1
            best_frac = 0.0
            for k in range(8):
                f = float(chunk[k, lr, lc])
                if f > best_frac:
                    best_frac = f
                    best_k = k
            if best_k == -1:
                break
            nr = cr + dy_arr[best_k]
            nc = cc + dx_arr[best_k]
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                break
            cr, cc = nr, nc

    path_rows = _buf_rows[:_buf_len]
    path_cols = _buf_cols[:_buf_len]
    path_labels = _buf_labels[:_buf_len]

    _get_chunk.cache_clear()

    # Phase 4: assemble via map_blocks
    _grouped = _group_cells_by_chunk(
        path_rows, path_cols, path_labels,
        chunks_y, chunks_x,
    )

    def _assemble_block(block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(block.shape, np.nan, dtype=np.float64)
        loc = block_info[0]['chunk-location']
        out = np.full(block.shape, np.nan, dtype=np.float64)
        group = _grouped.get((loc[0], loc[1]))
        if group is not None:
            local_r, local_c, lbls = group
            out[local_r, local_c] = lbls
        return out

    dummy = da.zeros((H, W), chunks=(chunks_y, chunks_x), dtype=np.float64)
    return da.map_blocks(
        _assemble_block, dummy,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


# =====================================================================
# Dask+CuPy backend
# =====================================================================

def _flow_path_mfd_dask_cupy(fractions_data, start_points_data, chunks_y, chunks_x):
    """Dask+CuPy: convert to numpy dask, run dask path, convert back."""
    import cupy as cp

    fr_np = fractions_data.map_blocks(
        lambda b: b.get(), dtype=fractions_data.dtype,
        meta=np.array((), dtype=fractions_data.dtype),
    )
    sp_np = start_points_data.map_blocks(
        lambda b: b.get(), dtype=start_points_data.dtype,
        meta=np.array((), dtype=start_points_data.dtype),
    )

    result = _flow_path_mfd_dask(fr_np, sp_np, chunks_y, chunks_x)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_path_mfd(flow_dir_mfd: xr.DataArray,
                  start_points: xr.DataArray,
                  name: str = 'flow_path_mfd') -> xr.DataArray:
    """Trace downstream flow paths using MFD dominant neighbor.

    Parameters
    ----------
    flow_dir_mfd : xarray.DataArray or xr.Dataset
        3D MFD flow direction array of shape (8, H, W) as returned
        by flow_direction_mfd.
    start_points : xarray.DataArray
        2D raster where non-NaN cells are path starting locations.
    name : str, default 'flow_path_mfd'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D grid where each cell on a traced path carries the label
        of its originating start point. All other cells are NaN.
    """
    _validate_raster(flow_dir_mfd, func_name='flow_path_mfd',
                     name='flow_dir_mfd', ndim=3)
    _validate_raster(start_points, func_name='flow_path_mfd',
                     name='start_points')

    data = flow_dir_mfd.data
    sp_data = start_points.data

    if data.ndim != 3 or data.shape[0] != 8:
        raise ValueError(
            f"flow_dir_mfd must have shape (8, H, W), got {data.shape}")

    _, H, W = data.shape

    if isinstance(data, np.ndarray):
        _check_memory(H, W)
        fr = data.astype(np.float64)
        sp = np.asarray(sp_data, dtype=np.float64)
        out = _flow_path_mfd_cpu(fr, sp, H, W)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        _check_gpu_memory(H, W)
        _check_memory(H, W)
        out = _flow_path_mfd_cupy(data, sp_data)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir_mfd):
        chunks_y = data.chunks[1]
        chunks_x = data.chunks[2]
        out = _flow_path_mfd_dask_cupy(data, sp_data, chunks_y, chunks_x)

    elif da is not None and isinstance(data, da.Array):
        chunks_y = data.chunks[1]
        chunks_x = data.chunks[2]
        out = _flow_path_mfd_dask(data, sp_data, chunks_y, chunks_x)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    # Build 2D output coords
    spatial_dims = flow_dir_mfd.dims[1:]
    coords = {}
    for d in spatial_dims:
        if d in flow_dir_mfd.coords:
            coords[d] = flow_dir_mfd.coords[d]

    return xr.DataArray(out,
                        name=name,
                        coords=coords,
                        dims=spatial_dims,
                        attrs=flow_dir_mfd.attrs)
