"""MFD watershed delineation.

Labels each cell with the pour point it drains to, using MFD
dominant-neighbor downstream tracing with path compression.

Algorithm
---------
CPU : downstream tracing with path compression, following the neighbor
      with the highest fraction at each step.
GPU : CuPy-via-CPU.
Dask: iterative tile sweep with exit-label propagation.
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

from xrspatial.hydrology._boundary_store import BoundaryStore
from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)
from xrspatial.dataset_support import supports_dataset

_DY_LIST = [0, 1, 1, 1, 0, -1, -1, -1]
_DX_LIST = [1, 1, 0, -1, -1, -1, 0, 1]


def _to_numpy_f64(arr):
    if hasattr(arr, 'get'):
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


def _dominant_offset_mfd_py(fractions_8):
    """Return (dy, dx) of dominant MFD neighbor, or (0,0) for pit/nodata."""
    best_k = -1
    best_f = 0.0
    for k in range(8):
        f = float(fractions_8[k])
        if f > best_f:
            best_f = f
            best_k = k
    if best_k == -1:
        return (0, 0)
    return (_DY_LIST[best_k], _DX_LIST[best_k])


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _watershed_mfd_cpu(fractions, labels, state, h, w):
    """Downstream tracing with path compression for MFD watershed."""
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

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

                chk = fractions[0, cr, cc]
                if chk != chk:  # NaN
                    break

                best_k = -1
                best_frac = 0.0
                for k in range(8):
                    f = fractions[k, cr, cc]
                    if f > best_frac:
                        best_frac = f
                        best_k = k
                if best_k == -1:
                    break

                nr, nc = cr + dy[best_k], cc + dx[best_k]
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

def _watershed_mfd_cupy(fractions_data, pour_points_data):
    import cupy as cp
    fr_np = _to_numpy_f64(fractions_data)
    pp_np = _to_numpy_f64(pour_points_data)
    _, h, w = fr_np.shape
    labels = np.full((h, w), np.nan, dtype=np.float64)
    state = np.zeros((h, w), dtype=np.int8)
    for r in range(h):
        for c in range(w):
            if fr_np[0, r, c] != fr_np[0, r, c]:
                pass
            elif pp_np[r, c] == pp_np[r, c]:
                labels[r, c] = pp_np[r, c]
                state[r, c] = 3
            else:
                state[r, c] = 1
    out = _watershed_mfd_cpu(fr_np, labels, state, h, w)
    return cp.asarray(out)


# =====================================================================
# Dask tile kernel
# =====================================================================

@ngjit
def _watershed_mfd_tile_kernel(fractions, h, w, pour_points,
                                exit_top, exit_bottom, exit_left, exit_right,
                                exit_tl, exit_tr, exit_bl, exit_br):
    """Seeded downstream tracing for an MFD tile."""
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    labels = np.empty((h, w), dtype=np.float64)
    state = np.empty((h, w), dtype=np.int8)

    for r in range(h):
        for c in range(w):
            v = fractions[0, r, c]
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

    # Apply exit labels
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

    # Downstream tracing
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

                chk = fractions[0, cr, cc]
                if chk != chk:
                    break

                best_k = -1
                best_frac = 0.0
                for k in range(8):
                    f = fractions[k, cr, cc]
                    if f > best_frac:
                        best_frac = f
                        best_k = k
                if best_k == -1:
                    break

                nr, nc = cr + dy[best_k], cc + dx[best_k]
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

def _preprocess_mfd_tiles(fractions_da, chunks_y, chunks_x):
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    frac_bdry = {}
    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            chunk = fractions_da[:, sum(chunks_y[:iy]):sum(chunks_y[:iy+1]),
                                    sum(chunks_x[:ix]):sum(chunks_x[:ix+1])].compute()
            chunk = np.asarray(chunk, dtype=np.float64)
            frac_bdry[('top', iy, ix)] = chunk[:, 0, :].copy()
            frac_bdry[('bottom', iy, ix)] = chunk[:, -1, :].copy()
            frac_bdry[('left', iy, ix)] = chunk[:, :, 0].copy()
            frac_bdry[('right', iy, ix)] = chunk[:, :, -1].copy()
    return frac_bdry


def _compute_exit_labels_mfd(iy, ix, boundaries, frac_bdry,
                              chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Compute exit labels for MFD tile using dominant neighbor."""
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
    fdir_top = frac_bdry.get(('top', iy, ix))
    if fdir_top is not None and iy > 0:
        nb_labels = boundaries.get('bottom', iy - 1, ix)
        for j in range(tile_w):
            d = _dominant_offset_mfd_py(fdir_top[:, j])
            if d[0] == -1:
                dj = j + d[1]
                if 0 <= dj < len(nb_labels):
                    exit_top[j] = nb_labels[dj]
                elif dj < 0 and ix > 0:
                    exit_top[j] = boundaries.get('bottom', iy - 1, ix - 1)[-1]
                elif dj >= len(nb_labels) and ix < n_tile_x - 1:
                    exit_top[j] = boundaries.get('bottom', iy - 1, ix + 1)[0]

    # Bottom row
    fdir_bot = frac_bdry.get(('bottom', iy, ix))
    if fdir_bot is not None and iy < n_tile_y - 1:
        nb_labels = boundaries.get('top', iy + 1, ix)
        for j in range(tile_w):
            d = _dominant_offset_mfd_py(fdir_bot[:, j])
            if d[0] == 1:
                dj = j + d[1]
                if 0 <= dj < len(nb_labels):
                    exit_bottom[j] = nb_labels[dj]
                elif dj < 0 and ix > 0:
                    exit_bottom[j] = boundaries.get('top', iy + 1, ix - 1)[-1]
                elif dj >= len(nb_labels) and ix < n_tile_x - 1:
                    exit_bottom[j] = boundaries.get('top', iy + 1, ix + 1)[0]

    # Left column
    fdir_left = frac_bdry.get(('left', iy, ix))
    if fdir_left is not None and ix > 0:
        nb_labels = boundaries.get('right', iy, ix - 1)
        for r in range(tile_h):
            d = _dominant_offset_mfd_py(fdir_left[:, r])
            if d[1] == -1:
                dr = r + d[0]
                if 0 <= dr < len(nb_labels):
                    exit_left[r] = nb_labels[dr]

    # Right column
    fdir_right = frac_bdry.get(('right', iy, ix))
    if fdir_right is not None and ix < n_tile_x - 1:
        nb_labels = boundaries.get('left', iy, ix + 1)
        for r in range(tile_h):
            d = _dominant_offset_mfd_py(fdir_right[:, r])
            if d[1] == 1:
                dr = r + d[0]
                if 0 <= dr < len(nb_labels):
                    exit_right[r] = nb_labels[dr]

    # Edge-of-grid exits
    if iy == 0 and fdir_top is not None:
        for j in range(tile_w):
            d = _dominant_offset_mfd_py(fdir_top[:, j])
            if d[0] == -1:
                exit_top[j] = np.nan
    if iy == n_tile_y - 1 and fdir_bot is not None:
        for j in range(tile_w):
            d = _dominant_offset_mfd_py(fdir_bot[:, j])
            if d[0] == 1:
                exit_bottom[j] = np.nan
    if ix == 0 and fdir_left is not None:
        for r in range(tile_h):
            d = _dominant_offset_mfd_py(fdir_left[:, r])
            if d[1] == -1:
                exit_left[r] = np.nan
    if ix == n_tile_x - 1 and fdir_right is not None:
        for r in range(tile_h):
            d = _dominant_offset_mfd_py(fdir_right[:, r])
            if d[1] == 1:
                exit_right[r] = np.nan

    # Diagonal corners
    if fdir_top is not None:
        d = _dominant_offset_mfd_py(fdir_top[:, 0])
        if d == (-1, -1):
            if iy > 0 and ix > 0:
                exit_tl = boundaries.get('bottom', iy - 1, ix - 1)[-1]
            else:
                exit_tl = np.nan
        d = _dominant_offset_mfd_py(fdir_top[:, -1])
        if d == (-1, 1):
            if iy > 0 and ix < n_tile_x - 1:
                exit_tr = boundaries.get('bottom', iy - 1, ix + 1)[0]
            else:
                exit_tr = np.nan
    if fdir_bot is not None:
        d = _dominant_offset_mfd_py(fdir_bot[:, 0])
        if d == (1, -1):
            if iy < n_tile_y - 1 and ix > 0:
                exit_bl = boundaries.get('top', iy + 1, ix - 1)[-1]
            else:
                exit_bl = np.nan
        d = _dominant_offset_mfd_py(fdir_bot[:, -1])
        if d == (1, 1):
            if iy < n_tile_y - 1 and ix < n_tile_x - 1:
                exit_br = boundaries.get('top', iy + 1, ix + 1)[0]
            else:
                exit_br = np.nan

    return (exit_top, exit_bottom, exit_left, exit_right,
            exit_tl, exit_tr, exit_bl, exit_br)


def _process_tile_mfd(iy, ix, fractions_da, pour_points_da,
                       boundaries, frac_bdry,
                       chunks_y, chunks_x, n_tile_y, n_tile_x):
    y_start = sum(chunks_y[:iy])
    y_end = y_start + chunks_y[iy]
    x_start = sum(chunks_x[:ix])
    x_end = x_start + chunks_x[ix]

    chunk = np.asarray(
        fractions_da[:, y_start:y_end, x_start:x_end].compute(),
        dtype=np.float64)
    pp_chunk = np.asarray(
        pour_points_da.blocks[iy, ix].compute(), dtype=np.float64)
    _, h, w = chunk.shape

    exits = _compute_exit_labels_mfd(
        iy, ix, boundaries, frac_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    result = _watershed_mfd_tile_kernel(chunk, h, w, pp_chunk, *exits)

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


def _watershed_mfd_dask(fractions_da, pour_points_da, chunks_y, chunks_x):
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    frac_bdry = _preprocess_mfd_tiles(fractions_da, chunks_y, chunks_x)
    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=np.nan)

    max_iterations = max(n_tile_y, n_tile_x) * 2 + 10

    for _iteration in range(max_iterations):
        any_changed = False

        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_tile_mfd(
                    iy, ix, fractions_da, pour_points_da,
                    boundaries, frac_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True

        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile_mfd(
                    iy, ix, fractions_da, pour_points_da,
                    boundaries, frac_bdry,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True

        if not any_changed:
            break

    boundaries = boundaries.snapshot()

    # Assemble final result
    rows = []
    for iy in range(n_tile_y):
        row = []
        for ix in range(n_tile_x):
            y_start = sum(chunks_y[:iy])
            y_end = y_start + chunks_y[iy]
            x_start = sum(chunks_x[:ix])
            x_end = x_start + chunks_x[ix]

            chunk = np.asarray(
                fractions_da[:, y_start:y_end, x_start:x_end].compute(),
                dtype=np.float64)
            pp_chunk = np.asarray(
                pour_points_da.blocks[iy, ix].compute(), dtype=np.float64)
            _, h, w = chunk.shape

            exits = _compute_exit_labels_mfd(
                iy, ix, boundaries, frac_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)

            tile = _watershed_mfd_tile_kernel(chunk, h, w, pp_chunk, *exits)
            row.append(da.from_array(tile, chunks=tile.shape))
        rows.append(row)

    return da.block(rows)


# =====================================================================
# Dask+CuPy backend
# =====================================================================

def _watershed_mfd_dask_cupy(fractions_da, pour_points_da, chunks_y, chunks_x):
    import cupy as cp
    fr_np = fractions_da.map_blocks(
        lambda b: b.get(), dtype=fractions_da.dtype,
        meta=np.array((), dtype=fractions_da.dtype),
    )
    pp_np = pour_points_da.map_blocks(
        lambda b: b.get(), dtype=pour_points_da.dtype,
        meta=np.array((), dtype=pour_points_da.dtype),
    )
    result = _watershed_mfd_dask(fr_np, pp_np, chunks_y, chunks_x)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def watershed_mfd(flow_dir_mfd: xr.DataArray,
                  pour_points: xr.DataArray,
                  name: str = 'watershed_mfd') -> xr.DataArray:
    """Label each cell with the pour point it drains to (MFD).

    Parameters
    ----------
    flow_dir_mfd : xarray.DataArray or xr.Dataset
        3D MFD flow direction array of shape (8, H, W).
    pour_points : xarray.DataArray
        2D raster where non-NaN cells are pour points.
    name : str, default='watershed_mfd'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D float64 array where each cell = label of its pour point.
        NaN for nodata or unreachable cells.
    """
    _validate_raster(flow_dir_mfd, func_name='watershed_mfd',
                     name='flow_dir_mfd', ndim=3)

    data = flow_dir_mfd.data
    pp_data = pour_points.data

    if data.ndim != 3 or data.shape[0] != 8:
        raise ValueError(
            f"flow_dir_mfd must have shape (8, H, W), got {data.shape}")

    _, H, W = data.shape

    if isinstance(data, np.ndarray):
        fr = data.astype(np.float64)
        pp = np.asarray(pp_data, dtype=np.float64)
        labels = np.full((H, W), np.nan, dtype=np.float64)
        state = np.zeros((H, W), dtype=np.int8)
        for r in range(H):
            for c in range(W):
                if fr[0, r, c] != fr[0, r, c]:
                    pass
                elif pp[r, c] == pp[r, c]:
                    labels[r, c] = pp[r, c]
                    state[r, c] = 3
                else:
                    state[r, c] = 1
        out = _watershed_mfd_cpu(fr, labels, state, H, W)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _watershed_mfd_cupy(data, pp_data)

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir_mfd):
        chunks_y = data.chunks[1]
        chunks_x = data.chunks[2]
        out = _watershed_mfd_dask_cupy(data, pp_data, chunks_y, chunks_x)

    elif da is not None and isinstance(data, da.Array):
        chunks_y = data.chunks[1]
        chunks_x = data.chunks[2]
        out = _watershed_mfd_dask(data, pp_data, chunks_y, chunks_x)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

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
