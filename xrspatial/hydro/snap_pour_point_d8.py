"""Snap pour points to the highest-accumulation cell within a search radius.

Users typically place pour points manually, but these often land a cell or
two off from the actual drainage channel.  This module moves each pour point
to the highest flow-accumulation cell within a circular search radius so
that subsequent ``watershed()`` calls delineate correctly.

Algorithm
---------
For each non-NaN cell in ``pour_points``:
1. Search all cells within ``search_radius`` pixels (Euclidean distance).
2. Among valid (non-NaN) ``flow_accum`` cells in that radius, find the one
   with maximum accumulation.
3. Move the pour point label to that cell.

If multiple pour points snap to the same cell, the last one in raster-scan
order wins (deterministic across all backends).
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
from xrspatial.dataset_support import supports_dataset


# =====================================================================
# Memory guards
# =====================================================================
#
# CPU peak working set per pixel for the numpy dispatch + ``_snap_pour_point_cpu``:
#   fa  (float64 cast)   -> 8
#   pp  (float64 cast)   -> 8
#   out (float64)        -> 8
# Total 24 bytes/pixel.  The caller's input arrays already live in RAM
# before dispatch and are not double-counted here.
_BYTES_PER_PIXEL = 24

# GPU peak working set per pixel for ``_snap_pour_point_cupy``:
#   fa  (float64 cast)   -> 8
#   pp  (float64 cast)   -> 8
#   out (float64)        -> 8
# Total 24 bytes/pixel on the device.  The sparse pour-point coordinate
# arrays scale with the number of pour points (typically << H*W) and are
# not counted here.
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
    """Raise MemoryError if the snap kernel would exceed 50% of RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"snap_pour_point_d8 on a {height}x{width} grid requires "
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
            f"snap_pour_point_d8 on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


def _to_numpy_f64(arr):
    """Convert *arr* to a contiguous numpy float64 array (handles CuPy)."""
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _snap_pour_point_cpu(flow_accum, pour_points, search_radius, H, W):
    """Snap each pour point to the max-accumulation cell within *radius*."""
    out = np.empty((H, W), dtype=np.float64)
    out[:] = np.nan
    radius_sq = search_radius * search_radius

    for r in range(H):
        for c in range(W):
            v = pour_points[r, c]
            if v != v:  # NaN
                continue
            label = v

            best_r = r
            best_c = c
            fa_val = flow_accum[r, c]
            if fa_val == fa_val:  # not NaN
                best_accum = fa_val
            else:
                best_accum = -1e308  # ~-inf

            r_lo = r - search_radius
            r_hi = r + search_radius
            c_lo = c - search_radius
            c_hi = c + search_radius

            if r_lo < 0:
                r_lo = 0
            if r_hi >= H:
                r_hi = H - 1
            if c_lo < 0:
                c_lo = 0
            if c_hi >= W:
                c_hi = W - 1

            for nr in range(r_lo, r_hi + 1):
                for nc in range(c_lo, c_hi + 1):
                    dr = nr - r
                    dc = nc - c
                    if dr * dr + dc * dc > radius_sq:
                        continue
                    fa_n = flow_accum[nr, nc]
                    if fa_n != fa_n:  # NaN
                        continue
                    if fa_n > best_accum:
                        best_accum = fa_n
                        best_r = nr
                        best_c = nc

            out[best_r, best_c] = label

    return out


# =====================================================================
# CuPy backend
# =====================================================================

@cuda.jit
def _snap_pour_point_gpu(flow_accum, pp_rows, pp_cols,
                          snap_rows, snap_cols, search_radius,
                          n_pp, H, W):
    """Each thread handles one pour point: windowed max search on GPU."""
    k = cuda.grid(1)
    if k >= n_pp:
        return

    r = pp_rows[k]
    c = pp_cols[k]

    best_r = r
    best_c = c
    fa_val = flow_accum[r, c]
    if fa_val == fa_val:  # not NaN
        best_accum = fa_val
    else:
        best_accum = -1e308

    radius_sq = search_radius * search_radius

    r_lo = r - search_radius
    if r_lo < 0:
        r_lo = 0
    r_hi = r + search_radius
    if r_hi >= H:
        r_hi = H - 1
    c_lo = c - search_radius
    if c_lo < 0:
        c_lo = 0
    c_hi = c + search_radius
    if c_hi >= W:
        c_hi = W - 1

    for nr in range(r_lo, r_hi + 1):
        for nc in range(c_lo, c_hi + 1):
            dr = nr - r
            dc = nc - c
            if dr * dr + dc * dc > radius_sq:
                continue
            fa_n = flow_accum[nr, nc]
            if fa_n != fa_n:  # NaN
                continue
            if fa_n > best_accum:
                best_accum = fa_n
                best_r = nr
                best_c = nc

    snap_rows[k] = best_r
    snap_cols[k] = best_c


def _snap_pour_point_cupy(flow_accum_data, pour_points_data, search_radius):
    """Native CuPy: CUDA kernel for windowed max search per pour point.

    Flow accumulation data stays on GPU.  Only pour point coordinates
    (sparse, typically < 100) are transferred CPU/GPU.
    """
    import cupy as cp

    H, W = flow_accum_data.shape
    fa = flow_accum_data.astype(cp.float64)
    pp = pour_points_data.astype(cp.float64)
    out = cp.full((H, W), cp.nan, dtype=cp.float64)

    mask = ~cp.isnan(pp)
    if not cp.any(mask):
        return out

    rows, cols = cp.where(mask)
    labels = pp[rows, cols]
    n_pp = len(rows)

    snap_rows = cp.empty(n_pp, dtype=cp.int64)
    snap_cols = cp.empty(n_pp, dtype=cp.int64)

    threads = min(256, n_pp)
    blocks = (n_pp + threads - 1) // threads

    _snap_pour_point_gpu[blocks, threads](
        fa, rows, cols, snap_rows, snap_cols,
        search_radius, n_pp, H, W)

    # Write labels in raster-scan order (last wins if overlap)
    # Sort by raster-scan position to ensure deterministic ordering
    raster_pos = rows * W + cols
    sort_idx = cp.argsort(raster_pos)

    for k in sort_idx.get():
        out[int(snap_rows[k]), int(snap_cols[k])] = float(labels[k])

    return out


# =====================================================================
# Dask backend
# =====================================================================

def _snap_pour_point_dask(flow_accum_data, pour_points_data, search_radius):
    """Dask: extract sparse pour points chunk-by-chunk, windowed search, lazy assembly.

    Pour points are sparse (typically < 100 in a multi-million-cell raster).
    We never materialize the full pour_points grid: a ``map_blocks`` pass
    reduces each chunk to a 1-byte flag, then only the (few) flagged chunks
    are loaded to extract coordinates.  Small windows of ``flow_accum`` are
    sliced for each pour point, and the output is assembled lazily.
    """
    H, W = flow_accum_data.shape
    chunks_y = pour_points_data.chunks[0]
    chunks_x = pour_points_data.chunks[1]

    # --- Phase 1: identify which chunks contain pour points --------
    # Single dask pass; each chunk is reduced to a scalar flag.
    # The scheduler parallelizes reads and releases each chunk after
    # the reduction, so peak memory is bounded by thread count × chunk size.
    def _has_pp(block):
        return np.array(
            [[np.any(~np.isnan(np.asarray(block))).item()]],
            dtype=np.int8,
        )

    flags = da.map_blocks(
        _has_pp, pour_points_data,
        dtype=np.int8,
        chunks=tuple((1,) * len(c) for c in pour_points_data.chunks),
    ).compute()  # tiny array: one byte per chunk

    # --- Phase 2: load only flagged chunks, extract coordinates ----
    points = []  # list of (global_row, global_col, label)
    row_off = 0
    for iy, cy in enumerate(chunks_y):
        col_off = 0
        for ix, cx in enumerate(chunks_x):
            if flags[iy, ix]:
                chunk = np.asarray(
                    pour_points_data.blocks[iy, ix].compute(),
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

    # --- Phase 3: snap each pour point via windowed search ---------
    snapped = []  # list of (snap_r, snap_c, label)
    radius_sq = search_radius * search_radius

    for r, c, label in points:
        r_lo = max(0, r - search_radius)
        r_hi = min(H - 1, r + search_radius)
        c_lo = max(0, c - search_radius)
        c_hi = min(W - 1, c + search_radius)

        # Small window; dask handles cross-chunk slicing
        window = np.asarray(
            flow_accum_data[r_lo:r_hi + 1, c_lo:c_hi + 1].compute(),
            dtype=np.float64,
        )

        best_r, best_c = r, c
        fa_val = window[r - r_lo, c - c_lo]
        best_accum = fa_val if not np.isnan(fa_val) else -np.inf

        for wr in range(window.shape[0]):
            for wc in range(window.shape[1]):
                nr = r_lo + wr
                nc = c_lo + wc
                dr = nr - r
                dc = nc - c
                if dr * dr + dc * dc > radius_sq:
                    continue
                fa_n = window[wr, wc]
                if np.isnan(fa_n):
                    continue
                if fa_n > best_accum:
                    best_accum = fa_n
                    best_r = nr
                    best_c = nc

        snapped.append((best_r, best_c, label))

    # --- Phase 4: lazy output assembly via map_blocks --------------
    snap_rows = np.array([s[0] for s in snapped], dtype=np.int64) if snapped else np.array([], dtype=np.int64)
    snap_cols = np.array([s[1] for s in snapped], dtype=np.int64) if snapped else np.array([], dtype=np.int64)
    snap_labels = np.array([s[2] for s in snapped], dtype=np.float64) if snapped else np.array([], dtype=np.float64)

    _snap_rows = snap_rows
    _snap_cols = snap_cols
    _snap_labels = snap_labels

    def _assemble_block(block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(block.shape, np.nan, dtype=np.float64)
        row_start, row_end = block_info[0]['array-location'][0]
        col_start, col_end = block_info[0]['array-location'][1]
        h, w = block.shape
        out = np.full((h, w), np.nan, dtype=np.float64)
        for k in range(len(_snap_rows)):
            sr = _snap_rows[k]
            sc = _snap_cols[k]
            if row_start <= sr < row_end and col_start <= sc < col_end:
                out[sr - row_start, sc - col_start] = _snap_labels[k]
        return out

    dummy = da.zeros((H, W), chunks=flow_accum_data.chunks, dtype=np.float64)
    return da.map_blocks(
        _assemble_block, dummy,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


# =====================================================================
# Dask+CuPy backend
# =====================================================================

def _snap_pour_point_dask_cupy(flow_accum_data, pour_points_data, search_radius):
    """Dask+CuPy: sparse pour-point processing with GPU-resident flow_accum.

    Extracts sparse pour point coordinates from CuPy chunks (small
    transfers), performs windowed search via dask slicing, and assembles
    output as dask+cupy.
    """
    import cupy as cp

    H, W = flow_accum_data.shape
    chunks_y = pour_points_data.chunks[0]
    chunks_x = pour_points_data.chunks[1]

    # Phase 1: identify chunks with pour points
    def _has_pp(block):
        b = block.get() if hasattr(block, 'get') else np.asarray(block)
        return np.array([[np.any(~np.isnan(b)).item()]], dtype=np.int8)

    flags = da.map_blocks(
        _has_pp, pour_points_data,
        dtype=np.int8,
        chunks=tuple((1,) * len(c) for c in pour_points_data.chunks),
    ).compute()

    # Phase 2: extract coordinates from flagged chunks
    points = []
    row_off = 0
    for iy, cy in enumerate(chunks_y):
        col_off = 0
        for ix, cx in enumerate(chunks_x):
            if flags[iy, ix]:
                chunk = _to_numpy_f64(
                    pour_points_data.blocks[iy, ix].compute())
                rs, cs = np.where(~np.isnan(chunk))
                for k in range(len(rs)):
                    points.append((
                        row_off + int(rs[k]),
                        col_off + int(cs[k]),
                        float(chunk[rs[k], cs[k]]),
                    ))
            col_off += cx
        row_off += cy

    # Phase 3: windowed search
    snapped = []
    radius_sq = search_radius * search_radius

    for r, c, label in points:
        r_lo = max(0, r - search_radius)
        r_hi = min(H - 1, r + search_radius)
        c_lo = max(0, c - search_radius)
        c_hi = min(W - 1, c + search_radius)

        window = _to_numpy_f64(
            flow_accum_data[r_lo:r_hi + 1, c_lo:c_hi + 1].compute())

        best_r, best_c = r, c
        fa_val = window[r - r_lo, c - c_lo]
        best_accum = fa_val if not np.isnan(fa_val) else -np.inf

        for wr in range(window.shape[0]):
            for wc in range(window.shape[1]):
                nr = r_lo + wr
                nc = c_lo + wc
                dr = nr - r
                dc = nc - c
                if dr * dr + dc * dc > radius_sq:
                    continue
                fa_n = window[wr, wc]
                if np.isnan(fa_n):
                    continue
                if fa_n > best_accum:
                    best_accum = fa_n
                    best_r = nr
                    best_c = nc

        snapped.append((best_r, best_c, label))

    # Phase 4: lazy output assembly as dask+cupy
    snap_rows = np.array([s[0] for s in snapped], dtype=np.int64) \
        if snapped else np.array([], dtype=np.int64)
    snap_cols = np.array([s[1] for s in snapped], dtype=np.int64) \
        if snapped else np.array([], dtype=np.int64)
    snap_labels = np.array([s[2] for s in snapped], dtype=np.float64) \
        if snapped else np.array([], dtype=np.float64)

    _snap_rows = snap_rows
    _snap_cols = snap_cols
    _snap_labels = snap_labels

    def _assemble_block(block, block_info=None):
        if block_info is None or 0 not in block_info:
            return cp.full(block.shape, cp.nan, dtype=cp.float64)
        row_start, row_end = block_info[0]['array-location'][0]
        col_start, col_end = block_info[0]['array-location'][1]
        h, w = block.shape
        out = np.full((h, w), np.nan, dtype=np.float64)
        for k in range(len(_snap_rows)):
            sr = _snap_rows[k]
            sc = _snap_cols[k]
            if row_start <= sr < row_end and col_start <= sc < col_end:
                out[sr - row_start, sc - col_start] = _snap_labels[k]
        return cp.asarray(out)

    dummy = da.zeros((H, W), chunks=flow_accum_data.chunks, dtype=np.float64)
    return da.map_blocks(
        _assemble_block, dummy,
        dtype=np.float64,
        meta=cp.array((), dtype=cp.float64),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def snap_pour_point_d8(flow_accum: xr.DataArray,
                    pour_points: xr.DataArray,
                    search_radius: int = 5,
                    name: str = 'snapped_pour_points') -> xr.DataArray:
    """Snap pour points to the highest-accumulation cell within a radius.

    Parameters
    ----------
    flow_accum : xarray.DataArray or xr.Dataset
        2D flow accumulation grid.
    pour_points : xarray.DataArray
        2D raster where non-NaN cells mark pour points (same format
        as ``watershed()`` expects).  Values are preserved as labels.
    search_radius : int, default 5
        Maximum search distance in pixels (Euclidean).
    name : str, default 'snapped_pour_points'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        Same-shape grid with pour point labels moved to their snapped
        locations.  Non-pour-point cells are NaN.
    """
    _validate_raster(flow_accum, func_name='snap_pour_point', name='flow_accum')

    if not isinstance(search_radius, (int, np.integer)) or search_radius < 1:
        raise ValueError(
            "search_radius must be a positive integer, got %r"
            % (search_radius,)
        )

    fa_data = flow_accum.data
    pp_data = pour_points.data

    if isinstance(fa_data, np.ndarray):
        _check_memory(*fa_data.shape)
        fa = fa_data.astype(np.float64)
        pp = np.asarray(pp_data, dtype=np.float64)
        H, W = fa.shape
        out = _snap_pour_point_cpu(fa, pp, search_radius, H, W)

    elif has_cuda_and_cupy() and is_cupy_array(fa_data):
        _check_gpu_memory(*fa_data.shape)
        out = _snap_pour_point_cupy(fa_data, pp_data, search_radius)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_accum):
        out = _snap_pour_point_dask_cupy(fa_data, pp_data, search_radius)

    elif da is not None and isinstance(fa_data, da.Array):
        out = _snap_pour_point_dask(fa_data, pp_data, search_radius)

    else:
        raise TypeError(f"Unsupported array type: {type(fa_data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_accum.coords,
                        dims=flow_accum.dims,
                        attrs=flow_accum.attrs)
