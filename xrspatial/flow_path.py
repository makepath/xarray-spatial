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

from xrspatial.flow_accumulation import _code_to_offset, _code_to_offset_py
from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)
from xrspatial.dataset_support import supports_dataset


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

    @lru_cache(maxsize=32)
    def _get_chunk(iy, ix):
        return np.asarray(
            flow_dir_data.blocks[iy, ix].compute(), dtype=np.float64)

    def _find_chunk(r, c):
        """Return (chunk_iy, chunk_ix, local_r, local_c)."""
        iy = int(np.searchsorted(fd_row_offsets[1:], r, side='right'))
        ix = int(np.searchsorted(fd_col_offsets[1:], c, side='right'))
        return iy, ix, r - int(fd_row_offsets[iy]), c - int(fd_col_offsets[ix])

    path_cells = []  # list of (r, c, label)
    for r, c, label in points:
        cr, cc = r, c
        while True:
            path_cells.append((cr, cc, label))
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

    # --- Phase 4: lazy output assembly via map_blocks -----------------
    path_rows = np.array([p[0] for p in path_cells], dtype=np.int64) if path_cells else np.array([], dtype=np.int64)
    path_cols = np.array([p[1] for p in path_cells], dtype=np.int64) if path_cells else np.array([], dtype=np.int64)
    path_labels = np.array([p[2] for p in path_cells], dtype=np.float64) if path_cells else np.array([], dtype=np.float64)

    _path_rows = path_rows
    _path_cols = path_cols
    _path_labels = path_labels

    def _assemble_block(block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(block.shape, np.nan, dtype=np.float64)
        row_start, row_end = block_info[0]['array-location'][0]
        col_start, col_end = block_info[0]['array-location'][1]
        h, w = block.shape
        out = np.full((h, w), np.nan, dtype=np.float64)
        for k in range(len(_path_rows)):
            pr = _path_rows[k]
            pc = _path_cols[k]
            if row_start <= pr < row_end and col_start <= pc < col_end:
                out[pr - row_start, pc - col_start] = _path_labels[k]
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
def flow_path(flow_dir: xr.DataArray,
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

    fd_data = flow_dir.data
    sp_data = start_points.data

    if isinstance(fd_data, np.ndarray):
        fd = fd_data.astype(np.float64)
        sp = np.asarray(sp_data, dtype=np.float64)
        H, W = fd.shape
        out = _flow_path_cpu(fd, sp, H, W)

    elif has_cuda_and_cupy() and is_cupy_array(fd_data):
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
