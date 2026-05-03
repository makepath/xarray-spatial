"""Trace downstream flow paths from start points through a D-inf direction grid.

Uses the dominant-neighbor approach: at each cell, the D-inf angle
decomposes into two neighbors with proportional weights; the path
follows whichever neighbor receives the higher weight.

Algorithm
---------
For each non-NaN cell in ``start_points``:
1. Decompose the D-inf angle into two neighbors and weights.
2. Follow the dominant neighbor (higher weight) at each step.
3. Write the start cell's label to every visited cell.
4. Stop at NaN, pit (angle < 0), out-of-bounds, or grid edge.
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
from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)
from xrspatial.dataset_support import supports_dataset


# =====================================================================
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for the eager ``flow_path_dinf`` branch:
#   fd float64 cast : 8
#   sp float64 cast : 8
#   out float64     : 8
# Total ~24 bytes/pixel.  The caller-provided ``flow_dir`` and
# ``start_points`` arrays already live in RAM before the kernel runs and
# are not double-counted here.
_BYTES_PER_PIXEL = 24

# GPU peak working set per pixel for ``_flow_path_dinf_cupy``: that path
# copies ``flow_dir`` and ``start_points`` to host via
# ``.get().astype()`` and runs the CPU kernel before converting the
# float64 output back to device via ``cp.asarray``.  Device-side
# residency at peak is the input float64 (8 B/px) plus the output
# float64 (8 B/px); host-side matches the 24 B/px CPU budget.  Use
# 32 B/px as a conservative GPU budget covering both copies plus
# headroom.
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
    """Raise MemoryError if the kernel would exceed 50% of available RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"flow_path_dinf on a {height}x{width} grid requires "
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
            f"flow_path_dinf on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


# =====================================================================
# Pure-Python angle decomposition (for dask tracing loop)
# =====================================================================

def _angle_to_neighbors_py(angle):
    """Pure-Python version of _angle_to_neighbors for non-numba contexts."""
    if isinstance(angle, float) and (math.isnan(angle) or angle < 0.0):
        return (0, 0, 0.0, 0, 0, 0.0)
    if angle < 0.0 or angle != angle:
        return (0, 0, 0.0, 0, 0, 0.0)

    pi_over_4 = math.pi / 4
    k = int(angle / pi_over_4)
    if k > 7:
        k = 7
    alpha = angle - k * pi_over_4
    w1 = 1.0 - alpha / pi_over_4
    w2 = alpha / pi_over_4

    facets = [
        ((0, 1), (-1, 1)),    # k=0: E, NE
        ((-1, 1), (-1, 0)),   # k=1: NE, N
        ((-1, 0), (-1, -1)),  # k=2: N, NW
        ((-1, -1), (0, -1)),  # k=3: NW, W
        ((0, -1), (1, -1)),   # k=4: W, SW
        ((1, -1), (1, 0)),    # k=5: SW, S
        ((1, 0), (1, 1)),     # k=6: S, SE
        ((1, 1), (0, 1)),     # k=7: SE, E
    ]
    (dy1, dx1), (dy2, dx2) = facets[k]
    return (dy1, dx1, w1, dy2, dx2, w2)


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _flow_path_dinf_cpu(flow_dir, start_points, H, W):
    """Trace downstream paths using D-inf dominant neighbor."""
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
                angle = flow_dir[cr, cc]
                if angle != angle:  # NaN
                    break
                dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(angle)
                if w1 <= 0.0 and w2 <= 0.0:
                    break  # pit
                if w1 >= w2:
                    dy, dx = dy1, dx1
                else:
                    dy, dx = dy2, dx2
                nr = cr + dy
                nc = cc + dx
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                cr, cc = nr, nc

    return out


# =====================================================================
# CuPy backend
# =====================================================================

def _flow_path_dinf_cupy(flow_dir_data, start_points_data):
    """CuPy: convert to numpy, run CPU kernel, convert back."""
    import cupy as cp

    fd_np = flow_dir_data.get() if hasattr(flow_dir_data, 'get') else np.asarray(flow_dir_data)
    sp_np = start_points_data.get() if hasattr(start_points_data, 'get') else np.asarray(start_points_data)
    fd_np = fd_np.astype(np.float64)
    sp_np = sp_np.astype(np.float64)
    H, W = fd_np.shape
    out = _flow_path_dinf_cpu(fd_np, sp_np, H, W)
    return cp.asarray(out)


# =====================================================================
# Dask backend
# =====================================================================

def _flow_path_dinf_dask(flow_dir_data, start_points_data):
    """Dask: sparse start-point extraction, LRU-cached tracing, lazy assembly."""
    from xrspatial.hydro.flow_path_d8 import _group_cells_by_chunk
    from functools import lru_cache

    H, W = flow_dir_data.shape
    chunks_y = start_points_data.chunks[0]
    chunks_x = start_points_data.chunks[1]

    # Phase 1: identify chunks with start points
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

    # Phase 2: extract start point coordinates
    points = []
    row_off = 0
    for iy, cy in enumerate(chunks_y):
        col_off = 0
        for ix, cx in enumerate(chunks_x):
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
    fd_chunks_y = flow_dir_data.chunks[0]
    fd_chunks_x = flow_dir_data.chunks[1]

    fd_row_offsets = np.zeros(len(fd_chunks_y) + 1, dtype=np.int64)
    for i, cy in enumerate(fd_chunks_y):
        fd_row_offsets[i + 1] = fd_row_offsets[i] + cy
    fd_col_offsets = np.zeros(len(fd_chunks_x) + 1, dtype=np.int64)
    for i, cx in enumerate(fd_chunks_x):
        fd_col_offsets[i + 1] = fd_col_offsets[i] + cx

    max_chunk_bytes = max(
        int(cy) * int(cx) * 8
        for cy in fd_chunks_y for cx in fd_chunks_x
    )
    cache_size = max(4, (512 * 1024 * 1024) // max(max_chunk_bytes, 1))

    @lru_cache(maxsize=cache_size)
    def _get_chunk(iy, ix):
        return np.asarray(
            flow_dir_data.blocks[iy, ix].compute(), dtype=np.float64)

    def _find_chunk(r, c):
        iy = int(np.searchsorted(fd_row_offsets[1:], r, side='right'))
        ix = int(np.searchsorted(fd_col_offsets[1:], c, side='right'))
        return iy, ix, r - int(fd_row_offsets[iy]), c - int(fd_col_offsets[ix])

    _init_cap = max(1024, len(points) * 4)
    _buf_rows = np.empty(_init_cap, dtype=np.int64)
    _buf_cols = np.empty(_init_cap, dtype=np.int64)
    _buf_labels = np.empty(_init_cap, dtype=np.float64)
    _buf_len = 0

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
            angle = chunk[lr, lc]
            if np.isnan(angle):
                break
            dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors_py(float(angle))
            if w1 <= 0.0 and w2 <= 0.0:
                break
            if w1 >= w2:
                dy, dx = dy1, dx1
            else:
                dy, dx = dy2, dx2
            nr = cr + dy
            nc = cc + dx
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
        flow_dir_data.chunks[0], flow_dir_data.chunks[1],
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

    dummy = da.zeros((H, W), chunks=flow_dir_data.chunks, dtype=np.float64)
    return da.map_blocks(
        _assemble_block, dummy,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


# =====================================================================
# Dask+CuPy backend
# =====================================================================

def _flow_path_dinf_dask_cupy(flow_dir_data, start_points_data):
    """Dask+CuPy: convert to numpy dask, run dask path, convert back."""
    import cupy as cp

    fd_np = flow_dir_data.map_blocks(
        lambda b: b.get(), dtype=flow_dir_data.dtype,
        meta=np.array((), dtype=flow_dir_data.dtype),
    )
    sp_np = start_points_data.map_blocks(
        lambda b: b.get(), dtype=start_points_data.dtype,
        meta=np.array((), dtype=start_points_data.dtype),
    )

    result = _flow_path_dinf_dask(fd_np, sp_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_path_dinf(flow_dir_dinf: xr.DataArray,
                   start_points: xr.DataArray,
                   name: str = 'flow_path_dinf') -> xr.DataArray:
    """Trace downstream flow paths using D-infinity dominant neighbor.

    Parameters
    ----------
    flow_dir_dinf : xarray.DataArray or xr.Dataset
        2D D-infinity flow direction grid. Values are continuous
        angles in radians [0, 2*pi), with -1.0 for pits and NaN
        for nodata.
    start_points : xarray.DataArray
        2D raster where non-NaN cells are path starting locations.
        Values are preserved as labels along the traced path.
    name : str, default 'flow_path_dinf'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        Same-shape grid where each cell on a traced path carries
        the label of its originating start point. All other cells
        are NaN. If paths overlap, the last start point in
        raster-scan order wins.
    """
    _validate_raster(flow_dir_dinf, func_name='flow_path_dinf',
                     name='flow_dir_dinf')
    _validate_raster(start_points, func_name='flow_path_dinf',
                     name='start_points')

    fd_data = flow_dir_dinf.data
    sp_data = start_points.data

    if isinstance(fd_data, np.ndarray):
        _check_memory(*fd_data.shape)
        fd = fd_data.astype(np.float64)
        sp = np.asarray(sp_data, dtype=np.float64)
        H, W = fd.shape
        out = _flow_path_dinf_cpu(fd, sp, H, W)

    elif has_cuda_and_cupy() and is_cupy_array(fd_data):
        _check_gpu_memory(*fd_data.shape)
        _check_memory(*fd_data.shape)
        out = _flow_path_dinf_cupy(fd_data, sp_data)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir_dinf):
        out = _flow_path_dinf_dask_cupy(fd_data, sp_data)

    elif da is not None and isinstance(fd_data, da.Array):
        out = _flow_path_dinf_dask(fd_data, sp_data)

    else:
        raise TypeError(f"Unsupported array type: {type(fd_data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir_dinf.coords,
                        dims=flow_dir_dinf.dims,
                        attrs=flow_dir_dinf.attrs)
