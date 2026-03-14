"""Drainage basin delineation from a D8 flow direction grid.

Automatically identifies all outlets (pits and edge-exit cells),
assigns each a unique ID, and labels every valid cell with the ID
of the outlet it drains to.  NaN flow direction cells produce NaN.

CPU uses downstream tracing with path compression.
GPU uses iterative label propagation.
Dask delegates to the watershed tile-sweep infrastructure.
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
# CPU kernels
# =====================================================================

@ngjit
def _basins_init_labels(flow_dir, h, w, total_h, total_w, row_off, col_off):
    """Initialize labels for basins mode.

    Pits (code 0) and cells that exit the **global** grid get unique
    IDs.  Unique ID = (row_off + r) * total_w + (col_off + c) + 1.
    Other valid cells = -1.  NaN cells = NaN.

    The global boundary check (total_h x total_w) ensures that cells
    flowing into an adjacent dask tile are NOT treated as edge-exits.
    """
    labels = np.empty((h, w), dtype=np.float64)

    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v != v:  # NaN
                labels[r, c] = np.nan
                continue

            dy, dx = _code_to_offset(v)
            if dy == 0 and dx == 0:
                # Pit -> assign unique ID
                labels[r, c] = float((row_off + r) * total_w +
                                     (col_off + c) + 1)
                continue

            # Check against GLOBAL grid boundaries
            gr = row_off + r + dy
            gc = col_off + c + dx
            if gr < 0 or gr >= total_h or gc < 0 or gc >= total_w:
                # Global edge-exit -> assign unique ID
                labels[r, c] = float((row_off + r) * total_w +
                                     (col_off + c) + 1)
                continue

            # Check if flows into NaN within this tile
            nr, nc = r + dy, c + dx
            if 0 <= nr < h and 0 <= nc < w:
                nv = flow_dir[nr, nc]
                if nv != nv:  # flows into NaN
                    labels[r, c] = float((row_off + r) * total_w +
                                         (col_off + c) + 1)
                    continue

            labels[r, c] = -1.0  # unresolved

    return labels


@ngjit
def _code_to_offset(code):
    """Return (dy, dx) row/col offset for a D8 direction code."""
    c = int(code)
    if c == 1:
        return 0, 1
    elif c == 2:
        return 1, 1
    elif c == 4:
        return 1, 0
    elif c == 8:
        return 1, -1
    elif c == 16:
        return 0, -1
    elif c == 32:
        return -1, -1
    elif c == 64:
        return -1, 0
    elif c == 128:
        return -1, 1
    return 0, 0


# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit
def _init_basins_gpu(flow_dir, labels, state, H, W):
    """Pits/edge-exits -> labeled + frontier. NaN -> state 0. Others -> state 1."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    v = flow_dir[i, j]
    if v != v:  # NaN
        state[i, j] = 0
        labels[i, j] = 0.0
        return

    # Decode direction inline
    code = int(v)
    dy = 0
    dx = 0
    if code == 1:
        dy, dx = 0, 1
    elif code == 2:
        dy, dx = 1, 1
    elif code == 4:
        dy, dx = 1, 0
    elif code == 8:
        dy, dx = 1, -1
    elif code == 16:
        dy, dx = 0, -1
    elif code == 32:
        dy, dx = -1, -1
    elif code == 64:
        dy, dx = -1, 0
    elif code == 128:
        dy, dx = -1, 1

    is_outlet = False
    if dy == 0 and dx == 0:
        is_outlet = True  # pit
    else:
        ni = i + dy
        nj = j + dx
        if ni < 0 or ni >= H or nj < 0 or nj >= W:
            is_outlet = True  # edge-exit
        else:
            nv = flow_dir[ni, nj]
            if nv != nv:  # flows into NaN
                is_outlet = True

    if is_outlet:
        labels[i, j] = float(i * W + j + 1)
        state[i, j] = 2  # frontier
    else:
        labels[i, j] = 0.0
        state[i, j] = 1  # active


@cuda.jit
def _propagate_basins_gpu(flow_dir, labels, state, changed, H, W):
    """Each active cell follows flow_dir one hop. If downstream is frontier -> take label."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    if state[i, j] != 1:
        return

    v = flow_dir[i, j]
    code = int(v)
    dy = 0
    dx = 0
    if code == 1:
        dy, dx = 0, 1
    elif code == 2:
        dy, dx = 1, 1
    elif code == 4:
        dy, dx = 1, 0
    elif code == 8:
        dy, dx = 1, -1
    elif code == 16:
        dy, dx = 0, -1
    elif code == 32:
        dy, dx = -1, -1
    elif code == 64:
        dy, dx = -1, 0
    elif code == 128:
        dy, dx = -1, 1

    if dy == 0 and dx == 0:
        return

    ni = i + dy
    nj = j + dx
    if ni < 0 or ni >= H or nj < 0 or nj >= W:
        return

    if state[ni, nj] == 2:  # downstream is frontier
        labels[i, j] = labels[ni, nj]
        state[i, j] = 3  # newly labeled
        cuda.atomic.add(changed, 0, 1)


@cuda.jit
def _advance_frontier_gpu(state, H, W):
    """state 2->4 (done), state 3->2 (new frontier)."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    s = state[i, j]
    if s == 2:
        state[i, j] = 4
    elif s == 3:
        state[i, j] = 2


def _basins_cupy(flow_dir_data):
    """GPU driver for basins."""
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)

    labels = cp.zeros((H, W), dtype=cp.float64)
    state = cp.zeros((H, W), dtype=cp.int32)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))

    _init_basins_gpu[griddim, blockdim](
        flow_dir_f64, labels, state, H, W)

    max_iter = H * W
    for _ in range(max_iter):
        changed[0] = 0
        _propagate_basins_gpu[griddim, blockdim](
            flow_dir_f64, labels, state, changed, H, W)
        if int(changed[0]) == 0:
            break
        _advance_frontier_gpu[griddim, blockdim](state, H, W)

    # Invalid (state=0) -> NaN; unresolved should not exist for basins
    labels = cp.where(state == 0, cp.nan, labels)
    return labels


# =====================================================================
# Dask paths
# =====================================================================

def _basins_dask_iterative(flow_dir_da):
    """Iterative boundary-propagation for basins on dask arrays.

    Constructs basin pour_points lazily, then delegates to the
    watershed dask infrastructure.
    """
    from xrspatial.watershed import _watershed_dask_iterative

    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    total_h = sum(chunks_y)
    total_w = sum(chunks_x)

    def _basins_make_pp_block(flow_dir_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(flow_dir_block.shape, np.nan, dtype=np.float64)
        row_off = block_info[0]['array-location'][0][0]
        col_off = block_info[0]['array-location'][1][0]
        h, w = flow_dir_block.shape
        chunk = np.asarray(flow_dir_block, dtype=np.float64)
        pp = _basins_init_labels(chunk, h, w, total_h, total_w,
                                 row_off, col_off)
        return np.where(pp >= 0, pp, np.nan)

    pour_points_da = da.map_blocks(
        _basins_make_pp_block, flow_dir_da,
        dtype=np.float64, meta=np.array((), dtype=np.float64))

    return _watershed_dask_iterative(flow_dir_da, pour_points_da)


def _basins_dask_cupy(flow_dir_da):
    """Dask+CuPy basins: native GPU via watershed infrastructure."""
    import cupy as cp
    from xrspatial.watershed import _watershed_dask_cupy

    chunks_y = flow_dir_da.chunks[0]
    chunks_x = flow_dir_da.chunks[1]
    total_h = sum(chunks_y)
    total_w = sum(chunks_x)

    def _basins_make_pp_block(flow_dir_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return cp.full(flow_dir_block.shape, cp.nan, dtype=cp.float64)
        row_off = block_info[0]['array-location'][0][0]
        col_off = block_info[0]['array-location'][1][0]
        h, w = flow_dir_block.shape
        chunk_np = flow_dir_block.get() if hasattr(flow_dir_block, 'get') \
            else np.asarray(flow_dir_block)
        chunk_np = np.asarray(chunk_np, dtype=np.float64)
        pp = _basins_init_labels(chunk_np, h, w, total_h, total_w,
                                  row_off, col_off)
        return cp.asarray(np.where(pp >= 0, pp, np.nan))

    pour_points_da = da.map_blocks(
        _basins_make_pp_block, flow_dir_da,
        dtype=np.float64, meta=cp.array((), dtype=cp.float64))

    return _watershed_dask_cupy(flow_dir_da, pour_points_da)


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def basin(flow_dir: xr.DataArray,
          name: str = 'basin') -> xr.DataArray:
    """Delineate drainage basins: every cell labeled with its outlet ID.

    Automatically identifies all outlets (pits and edge-exit cells)
    and assigns each a unique ID.  Every valid cell is then labeled
    with the ID of the outlet it drains to.  NaN flow_dir cells
    produce NaN.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2D D8 flow direction grid
        (codes 0/1/2/4/8/16/32/64/128; NaN for nodata).
    name : str, default='basin'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array where each cell = unique ID of its outlet.
        NaN for nodata cells.
    """
    _validate_raster(flow_dir, func_name='basin', name='flow_dir')

    data = flow_dir.data

    if isinstance(data, np.ndarray):
        from xrspatial.watershed import _watershed_cpu
        fd = data.astype(np.float64)
        h, w = fd.shape
        labels = _basins_init_labels(fd, h, w, h, w, 0, 0)
        # Build state array: 0=nodata(NaN), 1=unresolved(-1), 3=resolved
        state = np.where(np.isnan(labels), 0,
                         np.where(labels == -1.0, 1, 3)).astype(np.int8)
        out = _watershed_cpu(fd, labels, state, h, w)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _basins_cupy(data)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _basins_dask_cupy(data)

    elif da is not None and isinstance(data, da.Array):
        out = _basins_dask_iterative(data)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
