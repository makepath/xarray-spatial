"""MFD Height Above Nearest Drainage (HAND).

Uses MFD dominant-neighbor (highest fraction) for downstream tracing.
HAND = elevation - drain_elevation.

Algorithm
---------
CPU : Kahn's BFS topological sort with reverse propagation of drain_elev.
GPU : CuPy-via-CPU.
Dask: iterative tile sweep with BoundaryStore exit-label propagation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.hydro._boundary_store import BoundaryStore
from xrspatial.hydro.watershed_mfd import (
    _dominant_offset_mfd_py,
    _preprocess_mfd_tiles,
    _to_numpy_f64,
)
from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)


# =====================================================================
# CPU kernel
# =====================================================================

@ngjit
def _hand_mfd_cpu(fractions, flow_accum, elevation, H, W, threshold):
    """Compute HAND via Kahn's BFS with MFD dominant-neighbor tracing."""
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    in_degree = np.zeros((H, W), dtype=np.int32)
    valid = np.zeros((H, W), dtype=np.int8)
    is_stream = np.zeros((H, W), dtype=np.int8)
    drain_elev = np.empty((H, W), dtype=np.float64)
    hand_out = np.empty((H, W), dtype=np.float64)

    for r in range(H):
        for c in range(W):
            v = fractions[0, r, c]
            if v == v:
                valid[r, c] = 1
                fa = flow_accum[r, c]
                if fa == fa and fa >= threshold:
                    is_stream[r, c] = 1
                    drain_elev[r, c] = elevation[r, c]
                else:
                    drain_elev[r, c] = np.nan
            else:
                drain_elev[r, c] = np.nan
                hand_out[r, c] = np.nan

    # In-degrees: all MFD neighbors with frac > 0 contribute
    for r in range(H):
        for c in range(W):
            if valid[r, c] == 0:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
                    if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                        in_degree[nr, nc] += 1

    # BFS topological order
    order_r = np.empty(H * W, dtype=np.int64)
    order_c = np.empty(H * W, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for r in range(H):
        for c in range(W):
            if valid[r, c] == 1 and in_degree[r, c] == 0:
                order_r[tail] = r
                order_c[tail] = c
                tail += 1

    while head < tail:
        r = order_r[head]
        c = order_c[head]
        head += 1
        for k in range(8):
            if fractions[k, r, c] > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if 0 <= nr < H and 0 <= nc < W and valid[nr, nc] == 1:
                    in_degree[nr, nc] -= 1
                    if in_degree[nr, nc] == 0:
                        order_r[tail] = nr
                        order_c[tail] = nc
                        tail += 1

    # Reverse pass: propagate drain_elev via dominant neighbor
    for i in range(tail - 1, -1, -1):
        r = order_r[i]
        c = order_c[i]
        if is_stream[r, c] == 1:
            continue
        best_k = -1
        best_frac = 0.0
        for k in range(8):
            f = fractions[k, r, c]
            if f > best_frac:
                best_frac = f
                best_k = k
        if best_k == -1:
            drain_elev[r, c] = elevation[r, c]
            continue
        nr, nc = r + dy[best_k], c + dx[best_k]
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            drain_elev[r, c] = elevation[r, c]
            continue
        if valid[nr, nc] == 0:
            drain_elev[r, c] = elevation[r, c]
            continue
        de = drain_elev[nr, nc]
        if de == de:
            drain_elev[r, c] = de
        else:
            drain_elev[r, c] = elevation[r, c]

    for r in range(H):
        for c in range(W):
            if valid[r, c] == 1:
                hand_out[r, c] = elevation[r, c] - drain_elev[r, c]
            else:
                hand_out[r, c] = np.nan

    return hand_out


# =====================================================================
# CuPy backend
# =====================================================================

def _hand_mfd_cupy(fr_data, fa_data, elev_data, threshold):
    import cupy as cp
    fr_np = fr_data.get().astype(np.float64)
    fa_np = fa_data.get().astype(np.float64)
    el_np = elev_data.get().astype(np.float64)
    _, H, W = fr_np.shape
    out = _hand_mfd_cpu(fr_np, fa_np, el_np, H, W, threshold)
    return cp.asarray(out)


# =====================================================================
# Dask tile kernel
# =====================================================================

@ngjit
def _hand_mfd_drain_elev_tile(fractions, flow_accum, elevation, h, w,
                                threshold,
                                exit_top, exit_bottom, exit_left, exit_right,
                                exit_tl, exit_tr, exit_bl, exit_br):
    """Compute drain_elev for an MFD tile (for boundary propagation)."""
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)

    in_degree = np.zeros((h, w), dtype=np.int32)
    valid = np.zeros((h, w), dtype=np.int8)
    is_stream = np.zeros((h, w), dtype=np.int8)
    drain_elev = np.empty((h, w), dtype=np.float64)
    known = np.zeros((h, w), dtype=np.int8)

    for r in range(h):
        for c in range(w):
            v = fractions[0, r, c]
            if v == v:
                valid[r, c] = 1
                fa = flow_accum[r, c]
                if fa == fa and fa >= threshold:
                    is_stream[r, c] = 1
                    drain_elev[r, c] = elevation[r, c]
                    known[r, c] = 1
                else:
                    drain_elev[r, c] = np.nan
            else:
                drain_elev[r, c] = np.nan

    # Apply exit labels at boundaries where dominant neighbor exits tile
    # Top row
    for c in range(w):
        if valid[0, c] == 1 and known[0, c] == 0:
            best_k = -1
            best_frac = 0.0
            for k in range(8):
                f = fractions[k, 0, c]
                if f > best_frac:
                    best_frac = f
                    best_k = k
            if best_k >= 0 and 0 + dy[best_k] < 0:
                el = exit_top[c]
                if el == el:
                    drain_elev[0, c] = el
                    known[0, c] = 1
                else:
                    drain_elev[0, c] = elevation[0, c]
                    known[0, c] = 1

    # Bottom row
    for c in range(w):
        if valid[h - 1, c] == 1 and known[h - 1, c] == 0:
            best_k = -1
            best_frac = 0.0
            for k in range(8):
                f = fractions[k, h - 1, c]
                if f > best_frac:
                    best_frac = f
                    best_k = k
            if best_k >= 0 and h - 1 + dy[best_k] >= h:
                el = exit_bottom[c]
                if el == el:
                    drain_elev[h - 1, c] = el
                    known[h - 1, c] = 1
                else:
                    drain_elev[h - 1, c] = elevation[h - 1, c]
                    known[h - 1, c] = 1

    # Left column
    for r in range(h):
        if valid[r, 0] == 1 and known[r, 0] == 0:
            best_k = -1
            best_frac = 0.0
            for k in range(8):
                f = fractions[k, r, 0]
                if f > best_frac:
                    best_frac = f
                    best_k = k
            if best_k >= 0 and 0 + dx[best_k] < 0:
                el = exit_left[r]
                if el == el:
                    drain_elev[r, 0] = el
                    known[r, 0] = 1
                else:
                    drain_elev[r, 0] = elevation[r, 0]
                    known[r, 0] = 1

    # Right column
    for r in range(h):
        if valid[r, w - 1] == 1 and known[r, w - 1] == 0:
            best_k = -1
            best_frac = 0.0
            for k in range(8):
                f = fractions[k, r, w - 1]
                if f > best_frac:
                    best_frac = f
                    best_k = k
            if best_k >= 0 and w - 1 + dx[best_k] >= w:
                el = exit_right[r]
                if el == el:
                    drain_elev[r, w - 1] = el
                    known[r, w - 1] = 1
                else:
                    drain_elev[r, w - 1] = elevation[r, w - 1]
                    known[r, w - 1] = 1

    # Corners
    if valid[0, 0] == 1 and known[0, 0] == 0:
        best_k = -1
        best_frac = 0.0
        for k in range(8):
            f = fractions[k, 0, 0]
            if f > best_frac:
                best_frac = f
                best_k = k
        if best_k >= 0 and 0 + dy[best_k] < 0 and 0 + dx[best_k] < 0:
            if exit_tl == exit_tl:
                drain_elev[0, 0] = exit_tl
                known[0, 0] = 1

    if valid[0, w - 1] == 1 and known[0, w - 1] == 0:
        best_k = -1
        best_frac = 0.0
        for k in range(8):
            f = fractions[k, 0, w - 1]
            if f > best_frac:
                best_frac = f
                best_k = k
        if best_k >= 0 and 0 + dy[best_k] < 0 and w - 1 + dx[best_k] >= w:
            if exit_tr == exit_tr:
                drain_elev[0, w - 1] = exit_tr
                known[0, w - 1] = 1

    if valid[h - 1, 0] == 1 and known[h - 1, 0] == 0:
        best_k = -1
        best_frac = 0.0
        for k in range(8):
            f = fractions[k, h - 1, 0]
            if f > best_frac:
                best_frac = f
                best_k = k
        if best_k >= 0 and h - 1 + dy[best_k] >= h and 0 + dx[best_k] < 0:
            if exit_bl == exit_bl:
                drain_elev[h - 1, 0] = exit_bl
                known[h - 1, 0] = 1

    if valid[h - 1, w - 1] == 1 and known[h - 1, w - 1] == 0:
        best_k = -1
        best_frac = 0.0
        for k in range(8):
            f = fractions[k, h - 1, w - 1]
            if f > best_frac:
                best_frac = f
                best_k = k
        if best_k >= 0 and h - 1 + dy[best_k] >= h and w - 1 + dx[best_k] >= w:
            if exit_br == exit_br:
                drain_elev[h - 1, w - 1] = exit_br
                known[h - 1, w - 1] = 1

    # In-degrees
    for r in range(h):
        for c in range(w):
            if valid[r, c] == 0 or known[r, c] == 1:
                continue
            for k in range(8):
                if fractions[k, r, c] > 0.0:
                    nr = r + dy[k]
                    nc = c + dx[k]
                    if 0 <= nr < h and 0 <= nc < w:
                        if valid[nr, nc] == 1 and known[nr, nc] == 0:
                            in_degree[nr, nc] += 1

    # BFS
    order_r = np.empty(h * w, dtype=np.int64)
    order_c = np.empty(h * w, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)
    for r in range(h):
        for c in range(w):
            if valid[r, c] == 1 and known[r, c] == 0 and in_degree[r, c] == 0:
                order_r[tail] = r
                order_c[tail] = c
                tail += 1
    while head < tail:
        r = order_r[head]
        c = order_c[head]
        head += 1
        for k in range(8):
            if fractions[k, r, c] > 0.0:
                nr = r + dy[k]
                nc = c + dx[k]
                if 0 <= nr < h and 0 <= nc < w and valid[nr, nc] == 1 and known[nr, nc] == 0:
                    in_degree[nr, nc] -= 1
                    if in_degree[nr, nc] == 0:
                        order_r[tail] = nr
                        order_c[tail] = nc
                        tail += 1

    # Reverse pass
    for i in range(tail - 1, -1, -1):
        r = order_r[i]
        c = order_c[i]
        best_k = -1
        best_frac = 0.0
        for k in range(8):
            f = fractions[k, r, c]
            if f > best_frac:
                best_frac = f
                best_k = k
        if best_k == -1:
            drain_elev[r, c] = elevation[r, c]
            continue
        nr, nc = r + dy[best_k], c + dx[best_k]
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            drain_elev[r, c] = elevation[r, c]
            continue
        if valid[nr, nc] == 0:
            drain_elev[r, c] = elevation[r, c]
            continue
        de = drain_elev[nr, nc]
        if de == de:
            drain_elev[r, c] = de
        else:
            drain_elev[r, c] = elevation[r, c]

    return drain_elev


@ngjit
def _hand_mfd_tile_kernel(fractions, flow_accum, elevation, h, w, threshold,
                           exit_top, exit_bottom, exit_left, exit_right,
                           exit_tl, exit_tr, exit_bl, exit_br):
    """HAND tile kernel: returns HAND values."""
    drain_elev = _hand_mfd_drain_elev_tile(
        fractions, flow_accum, elevation, h, w, threshold,
        exit_top, exit_bottom, exit_left, exit_right,
        exit_tl, exit_tr, exit_bl, exit_br)

    out = np.empty((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            v = fractions[0, r, c]
            if v == v:
                out[r, c] = elevation[r, c] - drain_elev[r, c]
            else:
                out[r, c] = np.nan
    return out


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _compute_exit_labels_mfd(iy, ix, boundaries, frac_bdry,
                              chunks_y, chunks_x, n_tile_y, n_tile_x):
    from xrspatial.hydro.watershed_mfd import _compute_exit_labels_mfd as _ws_compute
    return _ws_compute(iy, ix, boundaries, frac_bdry,
                       chunks_y, chunks_x, n_tile_y, n_tile_x)


def _process_tile_hand_mfd(iy, ix, fractions_da, flow_accum_da, elev_da,
                             boundaries, frac_bdry, threshold,
                             chunks_y, chunks_x, n_tile_y, n_tile_x):
    y_start = sum(chunks_y[:iy])
    y_end = y_start + chunks_y[iy]
    x_start = sum(chunks_x[:ix])
    x_end = x_start + chunks_x[ix]

    fr_chunk = np.asarray(
        fractions_da[:, y_start:y_end, x_start:x_end].compute(),
        dtype=np.float64)
    fa_chunk = np.asarray(
        flow_accum_da.blocks[iy, ix].compute(), dtype=np.float64)
    el_chunk = np.asarray(
        elev_da.blocks[iy, ix].compute(), dtype=np.float64)
    _, h, w = fr_chunk.shape

    exits = _compute_exit_labels_mfd(
        iy, ix, boundaries, frac_bdry,
        chunks_y, chunks_x, n_tile_y, n_tile_x)

    drain_elev = _hand_mfd_drain_elev_tile(
        fr_chunk, fa_chunk, el_chunk, h, w, threshold, *exits)

    new_top = drain_elev[0, :].copy()
    new_bottom = drain_elev[-1, :].copy()
    new_left = drain_elev[:, 0].copy()
    new_right = drain_elev[:, -1].copy()

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


def _hand_mfd_dask(fractions_da, flow_accum_da, elev_da, threshold,
                     chunks_y, chunks_x):
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    frac_bdry = _preprocess_mfd_tiles(fractions_da, chunks_y, chunks_x)
    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=np.nan)

    max_iterations = max(n_tile_y, n_tile_x) * 2 + 10

    for _iteration in range(max_iterations):
        any_changed = False
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                c = _process_tile_hand_mfd(
                    iy, ix, fractions_da, flow_accum_da, elev_da,
                    boundaries, frac_bdry, threshold,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                c = _process_tile_hand_mfd(
                    iy, ix, fractions_da, flow_accum_da, elev_da,
                    boundaries, frac_bdry, threshold,
                    chunks_y, chunks_x, n_tile_y, n_tile_x)
                if c:
                    any_changed = True
        if not any_changed:
            break

    boundaries = boundaries.snapshot()

    # Assemble
    rows = []
    for iy in range(n_tile_y):
        row = []
        for ix in range(n_tile_x):
            y_start = sum(chunks_y[:iy])
            y_end = y_start + chunks_y[iy]
            x_start = sum(chunks_x[:ix])
            x_end = x_start + chunks_x[ix]

            fr_chunk = np.asarray(
                fractions_da[:, y_start:y_end, x_start:x_end].compute(),
                dtype=np.float64)
            fa_chunk = np.asarray(
                flow_accum_da.blocks[iy, ix].compute(), dtype=np.float64)
            el_chunk = np.asarray(
                elev_da.blocks[iy, ix].compute(), dtype=np.float64)
            _, h, w = fr_chunk.shape

            exits = _compute_exit_labels_mfd(
                iy, ix, boundaries, frac_bdry,
                chunks_y, chunks_x, n_tile_y, n_tile_x)

            tile = _hand_mfd_tile_kernel(
                fr_chunk, fa_chunk, el_chunk, h, w, threshold, *exits)
            row.append(da.from_array(tile, chunks=tile.shape))
        rows.append(row)

    return da.block(rows)


def _hand_mfd_dask_cupy(fractions_da, flow_accum_da, elev_da, threshold,
                          chunks_y, chunks_x):
    import cupy as cp
    fr_np = fractions_da.map_blocks(
        lambda b: b.get(), dtype=fractions_da.dtype,
        meta=np.array((), dtype=fractions_da.dtype))
    fa_np = flow_accum_da.map_blocks(
        lambda b: b.get(), dtype=flow_accum_da.dtype,
        meta=np.array((), dtype=flow_accum_da.dtype))
    el_np = elev_da.map_blocks(
        lambda b: b.get(), dtype=elev_da.dtype,
        meta=np.array((), dtype=elev_da.dtype))
    result = _hand_mfd_dask(fr_np, fa_np, el_np, threshold,
                             chunks_y, chunks_x)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype))


# =====================================================================
# Public API
# =====================================================================

def hand_mfd(flow_dir_mfd: xr.DataArray,
             flow_accum: xr.DataArray,
             elevation: xr.DataArray,
             threshold: float = 100,
             name: str = 'hand_mfd') -> xr.DataArray:
    """Compute HAND using MFD flow direction.

    Parameters
    ----------
    flow_dir_mfd : xarray.DataArray
        3D MFD flow direction array of shape (8, H, W).
    flow_accum : xarray.DataArray
        2D flow accumulation grid.
    elevation : xarray.DataArray
        2D elevation grid.
    threshold : float, default 100
        Minimum flow accumulation to define a stream cell.
    name : str, default 'hand_mfd'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray
        2D float64 HAND grid. Stream cells have HAND = 0.
    """
    _validate_raster(flow_dir_mfd, func_name='hand_mfd',
                     name='flow_dir_mfd', ndim=3)
    _validate_raster(flow_accum, func_name='hand_mfd', name='flow_accum')
    _validate_raster(elevation, func_name='hand_mfd', name='elevation')

    data = flow_dir_mfd.data
    fa_data = flow_accum.data
    el_data = elevation.data

    if data.ndim != 3 or data.shape[0] != 8:
        raise ValueError(
            f"flow_dir_mfd must have shape (8, H, W), got {data.shape}")

    _, H, W = data.shape

    if isinstance(data, np.ndarray):
        fr = data.astype(np.float64)
        fa = np.asarray(fa_data, dtype=np.float64)
        el = np.asarray(el_data, dtype=np.float64)
        out = _hand_mfd_cpu(fr, fa, el, H, W, float(threshold))

    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _hand_mfd_cupy(data, fa_data, el_data, float(threshold))

    elif has_cuda_and_cupy() and is_dask_cupy(flow_dir_mfd):
        chunks_y = data.chunks[1]
        chunks_x = data.chunks[2]
        out = _hand_mfd_dask_cupy(data, fa_data, el_data,
                                    float(threshold), chunks_y, chunks_x)

    elif da is not None and isinstance(data, da.Array):
        chunks_y = data.chunks[1]
        chunks_x = data.chunks[2]
        out = _hand_mfd_dask(data, fa_data, el_data,
                              float(threshold), chunks_y, chunks_x)

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
