"""Sink identification: find and label depression cells in a D8 flow direction grid.

Identifies cells with direction code 0 (pit/flat with no downhill neighbor)
and labels connected groups using 8-connected BFS.
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
# CPU kernel
# =====================================================================

@ngjit
def _sink_cpu(flow_dir, h, w, row_off, col_off, total_w):
    """8-connected BFS flood-fill CCL for sink cells (code 0).

    Labels each connected group of code-0 cells with a unique ID
    based on position: (row_off + r) * total_w + (col_off + c) + 1.
    """
    labels = np.empty((h, w), dtype=np.float64)
    labels[:] = np.nan

    dy = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dx = np.array([-1, 0, 1, -1, 1, -1, 0, 1])

    queue_r = np.empty(h * w, dtype=np.int64)
    queue_c = np.empty(h * w, dtype=np.int64)

    for r in range(h):
        for c in range(w):
            v = flow_dir[r, c]
            if v != v:  # NaN
                continue
            if v != 0.0:
                continue
            if labels[r, c] == labels[r, c]:  # already labeled
                continue

            label = float((row_off + r) * total_w + (col_off + c) + 1)
            labels[r, c] = label
            head = np.int64(0)
            tail = np.int64(0)
            queue_r[tail] = r
            queue_c[tail] = c
            tail += 1

            while head < tail:
                cr = queue_r[head]
                cc = queue_c[head]
                head += 1

                for k in range(8):
                    nr = cr + dy[k]
                    nc = cc + dx[k]
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    nv = flow_dir[nr, nc]
                    if nv != nv:
                        continue
                    if nv != 0.0:
                        continue
                    if labels[nr, nc] == labels[nr, nc]:
                        continue
                    labels[nr, nc] = label
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    return labels


# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit
def _sink_init_gpu(flow_dir, labels, H, W):
    """Pits (code 0) get position-based ID, others get 0."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    v = flow_dir[i, j]
    if v != v:  # NaN
        labels[i, j] = 0.0
        return
    if v == 0.0:
        labels[i, j] = float(i * W + j + 1)
    else:
        labels[i, j] = 0.0


@cuda.jit
def _sink_propagate_gpu(labels, changed, H, W):
    """Min-label propagation: each sink cell takes minimum neighbor label."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    my_label = labels[i, j]
    if my_label <= 0.0:
        return

    min_label = my_label
    for k in range(8):
        if k == 0:
            dy, dx = -1, -1
        elif k == 1:
            dy, dx = -1, 0
        elif k == 2:
            dy, dx = -1, 1
        elif k == 3:
            dy, dx = 0, -1
        elif k == 4:
            dy, dx = 0, 1
        elif k == 5:
            dy, dx = 1, -1
        elif k == 6:
            dy, dx = 1, 0
        else:
            dy, dx = 1, 1

        ni = i + dy
        nj = j + dx
        if ni < 0 or ni >= H or nj < 0 or nj >= W:
            continue
        nb = labels[ni, nj]
        if nb > 0.0 and nb < min_label:
            min_label = nb

    if min_label < my_label:
        labels[i, j] = min_label
        cuda.atomic.add(changed, 0, 1)


def _sink_cupy(flow_dir_data):
    """GPU driver for sink identification."""
    import cupy as cp

    H, W = flow_dir_data.shape
    flow_dir_f64 = flow_dir_data.astype(cp.float64)
    labels = cp.zeros((H, W), dtype=cp.float64)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))
    _sink_init_gpu[griddim, blockdim](flow_dir_f64, labels, H, W)

    max_iter = max(H, W)
    for _ in range(max_iter):
        changed[0] = 0
        _sink_propagate_gpu[griddim, blockdim](labels, changed, H, W)
        if int(changed[0]) == 0:
            break

    return cp.where(labels > 0, labels, cp.nan)


# =====================================================================
# Backend wrappers
# =====================================================================

def _run_numpy(data):
    h, w = data.shape
    return _sink_cpu(data.astype(np.float64), h, w, 0, 0, w)


def _run_dask_numpy(data):
    total_w = data.shape[1]

    def _tile_fn(block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(block.shape, np.nan, dtype=np.float64)
        row_off = block_info[0]['array-location'][0][0]
        col_off = block_info[0]['array-location'][1][0]
        h, w = block.shape
        return _sink_cpu(np.asarray(block, dtype=np.float64),
                         h, w, row_off, col_off, total_w)

    return da.map_blocks(
        _tile_fn, data,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _run_dask_cupy(data):
    """Dask+CuPy: convert to numpy dask, run CPU path, convert back."""
    import cupy as cp

    data_np = data.map_blocks(
        lambda b: b.get(), dtype=data.dtype,
        meta=np.array((), dtype=data.dtype),
    )
    result = _run_dask_numpy(data_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def sink_d8(flow_dir: xr.DataArray,
         name: str = 'sink') -> xr.DataArray:
    """Identify and label depression cells in a D8 flow direction grid.

    Finds cells with direction code 0 (pit/flat with no downhill
    neighbor) and labels connected groups using 8-connected
    component labeling.

    Parameters
    ----------
    flow_dir : xarray.DataArray or xr.Dataset
        2D D8 flow direction grid
        (codes 0/1/2/4/8/16/32/64/128; NaN for nodata).
    name : str, default='sink'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array where each sink cell is labeled with a unique
        group ID.  Non-sink cells and NaN cells are NaN.
    """
    _validate_raster(flow_dir, func_name='sink', name='flow_dir')

    data = flow_dir.data

    if isinstance(data, np.ndarray):
        out = _run_numpy(data)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _sink_cupy(data)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir):
        out = _run_dask_cupy(data)

    elif da is not None and isinstance(data, da.Array):
        out = _run_dask_numpy(data)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=flow_dir.coords,
                        dims=flow_dir.dims,
                        attrs=flow_dir.attrs)
