"""Trace downstream flow paths from start points through a D8 direction grid.

Given a set of start cells and a D8 flow direction grid, this module follows
the direction codes from each start cell to the outlet, writing the start
cell's label to every cell along the way.  If two paths share downstream
cells, the last start point in raster-scan order overwrites earlier labels
(deterministic, matches the snap_pour_point convention).

Algorithm
---------
For each non-NaN cell in ``start_points``:
1. Walk the D8 flow direction grid: read the code at the current cell,
   move to the neighbor it points to.
2. Write the start cell's label to each visited cell in the output.
3. Stop when the path hits NaN, code 0 (pit), out-of-bounds, or the
   grid edge.
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

from xrspatial.hydro.flow_accumulation_d8 import _code_to_offset, _code_to_offset_py
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
# CPU peak working set per pixel for ``flow_path_d8`` numpy path:
#   fd float64 copy   -> 8
#   sp float64 copy   -> 8
#   out float64       -> 8
# Measured peak via tracemalloc on a 2000x2000 grid: ~21 B/pixel.
# Use 24 B/pixel as a small safety margin.
_BYTES_PER_PIXEL = 24

# GPU peak working set per pixel for ``_flow_path_cupy``: that path
# copies fd and sp to host via ``.get()`` then runs the CPU kernel and
# copies the output back via ``cp.asarray``.  Device-side residency at
# peak is the input float64 (8 B/px) plus the output float64 (8 B/px);
# host-side matches the 24 B/px CPU budget.  Use 32 B/px as a
# conservative GPU budget covering both copies plus headroom.
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
    """Raise MemoryError if the flow_path kernel would exceed 50% of RAM."""
    required = int(height) * int(width) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"flow_path_d8 on a {height}x{width} grid requires "
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
            f"flow_path_d8 on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _flow_path_cpu(flow_dir, start_points, H, W):
    """Trace downstream paths from every non-NaN start point."""
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
                code = flow_dir[cr, cc]
                if code != code:  # NaN
                    break
                dy, dx = _code_to_offset(code)
                if dy == 0 and dx == 0:  # pit
                    break
                nr = cr + dy
                nc = cc + dx
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                cr, cc = nr, nc

    return out


# =====================================================================
# CuPy backend
# =====================================================================

def _flow_path_cupy(flow_dir_data, start_points_data):
    """CuPy: convert to numpy, run CPU kernel, convert back."""
    import cupy as cp

    fd_np = flow_dir_data.get() if hasattr(flow_dir_data, 'get') else np.asarray(flow_dir_data)
    sp_np = start_points_data.get() if hasattr(start_points_data, 'get') else np.asarray(start_points_data)
    fd_np = fd_np.astype(np.float64)
    sp_np = sp_np.astype(np.float64)
    H, W = fd_np.shape
    out = _flow_path_cpu(fd_np, sp_np, H, W)
    return cp.asarray(out)


# =====================================================================
# Dask backend
# =====================================================================

def _group_cells_by_chunk(rows, cols, labels, chunks_y, chunks_x):
    """Group path cells by destination chunk using vectorized operations.

    Parameters
    ----------
    rows, cols : np.ndarray[int64]
        Global row/col coordinates of path cells.
    labels : np.ndarray[float64]
        Label for each path cell.
    chunks_y, chunks_x : tuple[int, ...]
        Chunk sizes along each axis.

    Returns
    -------
    dict[(iy, ix)] -> (local_rows, local_cols, labels)
        Cells grouped by chunk, with coordinates converted to chunk-local.
    """
    if len(rows) == 0:
        return {}

    # Cumulative offsets for chunk boundary lookup
    row_offsets = np.cumsum(chunks_y)
    col_offsets = np.cumsum(chunks_x)
    n_tile_x = len(chunks_x)

    # Map each cell to its chunk index
    chunk_iy = np.searchsorted(row_offsets, rows, side='right')
    chunk_ix = np.searchsorted(col_offsets, cols, side='right')

    # Composite key for grouping; stable sort preserves raster-scan order
    keys = chunk_iy * n_tile_x + chunk_ix
    order = np.argsort(keys, kind='stable')

    sorted_keys = keys[order]
    sorted_rows = rows[order]
    sorted_cols = cols[order]
    sorted_labels = labels[order]

    # Find group boundaries
    breaks = np.nonzero(np.diff(sorted_keys))[0] + 1
    starts = np.empty(len(breaks) + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = breaks
    ends = np.empty(len(breaks) + 1, dtype=np.intp)
    ends[:-1] = breaks
    ends[-1] = len(sorted_keys)

    # Row/col offsets with a leading zero for local coord conversion
    row_off = np.empty(len(chunks_y), dtype=np.int64)
    row_off[0] = 0
    row_off[1:] = row_offsets[:-1]
    col_off = np.empty(len(chunks_x), dtype=np.int64)
    col_off[0] = 0
    col_off[1:] = col_offsets[:-1]

    result = {}
    for g in range(len(starts)):
        s, e = starts[g], ends[g]
        iy = int(chunk_iy[order[s]])
        ix = int(chunk_ix[order[s]])
        local_r = sorted_rows[s:e] - row_off[iy]
        local_c = sorted_cols[s:e] - col_off[ix]
        result[(iy, ix)] = (local_r, local_c, sorted_labels[s:e])
    return result


def _flow_path_dask(flow_dir_data, start_points_data):
    """Dask: extract sparse start points, trace paths, lazy assembly.

    Start points are sparse.  A ``map_blocks`` pass reduces each chunk
    of ``start_points`` to a 1-byte flag, then only flagged chunks are
    loaded to extract coordinates.  Paths are traced through
    ``flow_dir`` with an LRU cache bounding memory use.  The output is
    assembled lazily via ``map_blocks``.
    """
    from functools import lru_cache

    H, W = flow_dir_data.shape
    chunks_y = start_points_data.chunks[0]
    chunks_x = start_points_data.chunks[1]

    # --- Phase 1: identify which chunks contain start points ----------
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

    # --- Phase 2: load only flagged chunks, extract coordinates -------
    points = []  # list of (global_row, global_col, label)
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

    # --- Phase 3: trace paths through flow_dir with LRU cache ---------
    fd_chunks_y = flow_dir_data.chunks[0]
    fd_chunks_x = flow_dir_data.chunks[1]

    # Precompute cumulative offsets for chunk lookups
    fd_row_offsets = np.zeros(len(fd_chunks_y) + 1, dtype=np.int64)
    for i, cy in enumerate(fd_chunks_y):
        fd_row_offsets[i + 1] = fd_row_offsets[i] + cy
    fd_col_offsets = np.zeros(len(fd_chunks_x) + 1, dtype=np.int64)
    for i, cx in enumerate(fd_chunks_x):
        fd_col_offsets[i + 1] = fd_col_offsets[i] + cx

    # Adaptive LRU: cap cached chunks at ~512 MB total
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
        """Return (chunk_iy, chunk_ix, local_r, local_c)."""
        iy = int(np.searchsorted(fd_row_offsets[1:], r, side='right'))
        ix = int(np.searchsorted(fd_col_offsets[1:], c, side='right'))
        return iy, ix, r - int(fd_row_offsets[iy]), c - int(fd_col_offsets[ix])

    # Growable numpy buffers instead of list-of-tuples (~24 bytes/cell
    # vs ~164 bytes/cell for Python tuples).
    _init_cap = max(1024, len(points) * 4)
    _buf_rows = np.empty(_init_cap, dtype=np.int64)
    _buf_cols = np.empty(_init_cap, dtype=np.int64)
    _buf_labels = np.empty(_init_cap, dtype=np.float64)
    _buf_len = 0

    for r, c, label in points:
        cr, cc = r, c
        while True:
            # Grow buffers if needed (doubling strategy)
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
            code = chunk[lr, lc]
            if np.isnan(code):
                break
            dy, dx = _code_to_offset_py(code)
            if dy == 0 and dx == 0:
                break
            nr = cr + dy
            nc = cc + dx
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                break
            cr, cc = nr, nc

    # Trim to actual length (slice view, no copy)
    path_rows = _buf_rows[:_buf_len]
    path_cols = _buf_cols[:_buf_len]
    path_labels = _buf_labels[:_buf_len]

    # Release cached flow_dir chunks before assembly
    _get_chunk.cache_clear()

    # --- Phase 4: chunk-indexed assembly via map_blocks ---------------
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

def _flow_path_dask_cupy(flow_dir_data, start_points_data):
    """Dask+CuPy: convert cupy chunks to numpy, run dask path, convert back."""
    import cupy as cp

    fd_np = flow_dir_data.map_blocks(
        lambda b: b.get(), dtype=flow_dir_data.dtype,
        meta=np.array((), dtype=flow_dir_data.dtype),
    )
    sp_np = start_points_data.map_blocks(
        lambda b: b.get(), dtype=start_points_data.dtype,
        meta=np.array((), dtype=start_points_data.dtype),
    )

    result = _flow_path_dask(fd_np, sp_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def flow_path_d8(flow_dir: xr.DataArray,
              start_points: xr.DataArray,
              name: str = 'flow_path') -> xr.DataArray:
    """Trace downstream flow paths from start points through a D8 grid.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2D D8 flow direction grid (codes 0/1/2/4/8/16/32/64/128;
        NaN for nodata).
    start_points : xarray.DataArray
        2D raster where non-NaN cells are path starting locations.
        Values are preserved as labels along the traced path.
    name : str, default 'flow_path'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        Same-shape grid where each cell on a traced path carries
        the label of its originating start point.  All other cells
        are NaN.  If paths overlap, the last start point in
        raster-scan order wins.
    """
    _validate_raster(flow_dir, func_name='flow_path', name='flow_dir')
    _validate_raster(start_points, func_name='flow_path',
                     name='start_points')

    fd_data = flow_dir.data
    sp_data = start_points.data

    if isinstance(fd_data, np.ndarray):
        _check_memory(*fd_data.shape)
        fd = fd_data.astype(np.float64)
        sp = np.asarray(sp_data, dtype=np.float64)
        H, W = fd.shape
        out = _flow_path_cpu(fd, sp, H, W)

    elif has_cuda_and_cupy() and is_cupy_array(fd_data):
        _check_gpu_memory(*fd_data.shape)
        _check_memory(*fd_data.shape)
        out = _flow_path_cupy(fd_data, sp_data)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _flow_path_dask_cupy(fd_data, sp_data)

    elif da is not None and isinstance(fd_data, da.Array):
        out = _flow_path_dask(fd_data, sp_data)

    else:
        raise TypeError(f"Unsupported array type: {type(fd_data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
