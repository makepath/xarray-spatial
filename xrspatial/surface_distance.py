"""Surface distance along 3D terrain via multi-source Dijkstra.

Computes the minimum accumulated distance along the terrain surface
from each pixel to the nearest target pixel, accounting for elevation
relief.  A steep hillside has more surface distance than its flat
map projection.

Algorithm
---------
Multi-source Dijkstra with edge costs derived from elevation::

    edge_cost = sqrt(horizontal_dist² + (elev_v − elev_u)²)

NaN elevation marks impassable pixels (barriers).

Dask strategy
-------------
For finite ``max_distance``, the maximum pixel radius any path can
reach is ``max_distance / min_cellsize`` (since surface distance >=
horizontal distance).  This becomes the ``depth`` parameter to
``dask.array.map_overlap``, giving exact results.

If ``max_distance`` is infinite or the implied radius exceeds chunk
dimensions, an iterative boundary-only Dijkstra is used (same
tile-sweep pattern as ``cost_distance``).
"""

from __future__ import annotations

import math as _math
import warnings
from math import sqrt

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from numba import cuda

try:
    import cupy
except ImportError:
    class cupy:  # type: ignore[no-redef]
        ndarray = False

from xrspatial.cost_distance import _heap_push, _heap_pop
from xrspatial.proximity import _vectorized_calc_direction
from xrspatial.utils import (
    _validate_raster,
    cuda_args, get_dataarray_resolution, ngjit,
    has_cuda_and_cupy, is_cupy_array, is_dask_cupy,
)
from xrspatial.dataset_support import supports_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISTANCE = 0
ALLOCATION = 1
DIRECTION = 2


# ---------------------------------------------------------------------------
# Memory guards
# ---------------------------------------------------------------------------
# Peak working set per pixel for the eager numpy backend:
#   dist (float64)              8
#   alloc (float64)             8
#   src_row (int64)             8
#   src_col (int64)             8
#   visited (int8)              1
#   h_keys (float64)            8
#   h_rows (int64)              8
#   h_cols (int64)              8
#   output (float32)            4
#   direction-mode temps        ~16
# Total ~80 bytes/pixel.  A 50000x50000 raster needs ~200 GB.
_BYTES_PER_PIXEL = 80

# CuPy backend skips the explicit binary heap (parallel relaxation instead)
# but still allocates dist, alloc, srow, scol, src cast, elev cast, mask,
# row_idx/col_idx, output.  ~72 bytes/pixel.
_GPU_BYTES_PER_PIXEL = 72


def _available_memory_bytes():
    """Best-effort estimate of available host memory in bytes."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil

        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    return 2 * 1024**3


def _available_gpu_memory_bytes():
    """Best-effort estimate of free GPU memory in bytes.

    Returns 0 when CuPy / CUDA is unavailable or the query fails.
    Callers treat that as "no GPU info, skip the guard".
    """
    try:
        import cupy as _cp

        free, _total = _cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return 0


def _check_memory(rows, cols):
    """Raise MemoryError if the eager numpy pass would exceed 50% of RAM."""
    required = int(rows) * int(cols) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"surface_distance() on a {rows}x{cols} raster needs "
            f"~{required / 1e9:.1f} GB of working memory but only "
            f"~{available / 1e9:.1f} GB is available.  Set a finite "
            f"`max_distance=` to bound the search, or use a dask-backed "
            f"DataArray for out-of-core processing."
        )


def _check_gpu_memory(rows, cols):
    """Raise MemoryError when the cupy allocation would not fit.

    Checks host memory first because the input may already be staged on
    the host before transfer.  Skips silently when the GPU query fails.
    """
    _check_memory(rows, cols)
    available = _available_gpu_memory_bytes()
    if available <= 0:
        return
    required = int(rows) * int(cols) * _GPU_BYTES_PER_PIXEL
    if required > 0.5 * available:
        raise MemoryError(
            f"surface_distance() on a {rows}x{cols} cupy raster needs "
            f"~{required / 1e9:.1f} GB of GPU memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Set a finite `max_distance=` to bound the search, or use a "
            f"dask+cupy DataArray for out-of-core processing."
        )


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------


@ngjit
def _seed_sources(source_data, elev_data, target_values,
                  dist, alloc, src_row, src_col):
    """Seed source pixels into pre-allocated output arrays.

    Source pixels: dist=0, alloc=pixel_value, src_row/col=pixel position.
    Only seeds if elevation is finite (passable).
    """
    height, width = source_data.shape
    n_values = len(target_values)

    for r in range(height):
        for c in range(width):
            val = source_data[r, c]
            is_target = False
            if n_values == 0:
                if val != 0.0 and np.isfinite(val):
                    is_target = True
            else:
                for k in range(n_values):
                    if val == target_values[k]:
                        is_target = True
                        break
            if is_target:
                if np.isfinite(elev_data[r, c]):
                    dist[r, c] = 0.0
                    alloc[r, c] = val
                    src_row[r, c] = r
                    src_col[r, c] = c


@ngjit
def _dijkstra(elev_data, height, width, max_distance,
              dy, dx, dd, dist, alloc, src_row, src_col):
    """Planar multi-source Dijkstra.  Modifies arrays in-place.

    Pre-seeded pixels (dist < inf) are added to the heap.
    Edge cost = sqrt(dd[i]^2 + dz^2).
    """
    n_neighbors = len(dy)

    max_heap = height * width
    h_keys = np.empty(max_heap, dtype=np.float64)
    h_rows = np.empty(max_heap, dtype=np.int64)
    h_cols = np.empty(max_heap, dtype=np.int64)
    h_size = 0

    visited = np.zeros((height, width), dtype=np.int8)

    # Add all pre-seeded pixels to the heap
    for r in range(height):
        for c in range(width):
            if dist[r, c] < np.inf:
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    dist[r, c], r, c)

    # Dijkstra main loop
    while h_size > 0:
        cost_u, ur, uc, h_size = _heap_pop(h_keys, h_rows, h_cols, h_size)

        if visited[ur, uc]:
            continue
        visited[ur, uc] = 1

        if cost_u > max_distance:
            break

        elev_u = elev_data[ur, uc]

        for i in range(n_neighbors):
            vr = ur + dy[i]
            vc = uc + dx[i]
            if vr < 0 or vr >= height or vc < 0 or vc >= width:
                continue
            if visited[vr, vc]:
                continue

            elev_v = elev_data[vr, vc]
            if not np.isfinite(elev_v):
                continue

            dz = elev_v - elev_u
            edge_cost = np.sqrt(dd[i] * dd[i] + dz * dz)
            new_cost = cost_u + edge_cost

            if new_cost < dist[vr, vc]:
                dist[vr, vc] = new_cost
                alloc[vr, vc] = alloc[ur, uc]
                src_row[vr, vc] = src_row[ur, uc]
                src_col[vr, vc] = src_col[ur, uc]
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    new_cost, vr, vc)


@ngjit
def _dijkstra_geodesic(elev_data, height, width, max_distance,
                       dy, dx, dd_grid, dist, alloc, src_row, src_col):
    """Geodesic variant with per-pixel horizontal distances.

    dd_grid[i, r, c] = great-circle horizontal distance to neighbour i
    from pixel (r, c).
    """
    n_neighbors = len(dy)

    max_heap = height * width
    h_keys = np.empty(max_heap, dtype=np.float64)
    h_rows = np.empty(max_heap, dtype=np.int64)
    h_cols = np.empty(max_heap, dtype=np.int64)
    h_size = 0

    visited = np.zeros((height, width), dtype=np.int8)

    for r in range(height):
        for c in range(width):
            if dist[r, c] < np.inf:
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    dist[r, c], r, c)

    while h_size > 0:
        cost_u, ur, uc, h_size = _heap_pop(h_keys, h_rows, h_cols, h_size)

        if visited[ur, uc]:
            continue
        visited[ur, uc] = 1

        if cost_u > max_distance:
            break

        elev_u = elev_data[ur, uc]

        for i in range(n_neighbors):
            vr = ur + dy[i]
            vc = uc + dx[i]
            if vr < 0 or vr >= height or vc < 0 or vc >= width:
                continue
            if visited[vr, vc]:
                continue

            elev_v = elev_data[vr, vc]
            if not np.isfinite(elev_v):
                continue

            hdist = dd_grid[i, ur, uc]
            dz = elev_v - elev_u
            edge_cost = np.sqrt(hdist * hdist + dz * dz)
            new_cost = cost_u + edge_cost

            if new_cost < dist[vr, vc]:
                dist[vr, vc] = new_cost
                alloc[vr, vc] = alloc[ur, uc]
                src_row[vr, vc] = src_row[ur, uc]
                src_col[vr, vc] = src_col[ur, uc]
                h_size = _heap_push(h_keys, h_rows, h_cols, h_size,
                                    new_cost, vr, vc)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _init_arrays(H, W):
    """Create and initialize output arrays for the Dijkstra kernel."""
    dist = np.full((H, W), np.inf, dtype=np.float64)
    alloc = np.full((H, W), np.nan, dtype=np.float64)
    src_row = np.full((H, W), -1, dtype=np.int64)
    src_col = np.full((H, W), -1, dtype=np.int64)
    return dist, alloc, src_row, src_col


def _finalize_dist(dist, max_distance):
    """Convert float64 dist to float32; inf / over-budget -> NaN."""
    out = np.where(
        np.isinf(dist) | (dist > max_distance), np.nan, dist,
    ).astype(np.float32)
    return out


def _finalize_alloc(alloc, dist, max_distance):
    """Float32 allocation; NaN where unreachable."""
    out = np.where(
        np.isinf(dist) | (dist > max_distance), np.nan, alloc,
    ).astype(np.float32)
    return out


def _finalize_direction(src_row, src_col, dist, cellsize_x, cellsize_y,
                        max_distance):
    """Compute compass bearing from each pixel to its allocated source.

    Uses pixel index differences scaled by cell size.
    """
    H, W = dist.shape
    row_idx, col_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Coordinate differences (source - pixel)
    dx = (src_col.astype(np.float64) - col_idx) * cellsize_x
    dy = (src_row.astype(np.float64) - row_idx) * cellsize_y

    result = _vectorized_calc_direction(
        np.zeros((H, W), dtype=np.float64), dx,
        np.zeros((H, W), dtype=np.float64), dy,
    )

    # Mask unreachable and no-source pixels
    mask = np.isinf(dist) | (dist > max_distance) | (src_row < 0)
    result[mask] = np.nan
    return result


def _extract_output(dist, alloc, src_row, src_col,
                    cellsize_x, cellsize_y, max_distance, mode):
    """Select and finalize the requested output from raw Dijkstra arrays."""
    if mode == DISTANCE:
        return _finalize_dist(dist, max_distance)
    elif mode == ALLOCATION:
        return _finalize_alloc(alloc, dist, max_distance)
    else:
        return _finalize_direction(src_row, src_col, dist,
                                   cellsize_x, cellsize_y, max_distance)


# ---------------------------------------------------------------------------
# Geodesic dd_grid precomputation
# ---------------------------------------------------------------------------

EARTH_RADIUS = 6378137.0  # meters


def _precompute_dd_grid(lat_2d, lon_2d, dy, dx):
    """Precompute per-pixel great-circle horizontal distances.

    Returns dd_grid[n_neighbors, H, W] in meters.
    """
    H, W = lat_2d.shape
    n = len(dy)
    # Memory guard: dd_grid is (n_neighbors, H, W) float64
    estimated = n * H * W * 8
    try:
        from xrspatial.zonal import _available_memory_bytes
        avail = _available_memory_bytes()
    except ImportError:
        avail = 2 * 1024**3
    if estimated > 0.8 * avail:
        raise MemoryError(
            f"Geodesic dd_grid needs ~{estimated / 1e9:.1f} GB "
            f"({n} neighbors x {H}x{W} x 8 bytes) but only "
            f"~{avail / 1e9:.1f} GB available.  Use planar mode "
            f"or downsample the raster."
        )
    dd_grid = np.zeros((n, H, W), dtype=np.float64)

    for i in range(n):
        dr, dc = int(dy[i]), int(dx[i])
        # Source region
        r0 = max(0, -dr)
        r1 = H - max(0, dr)
        c0 = max(0, -dc)
        c1 = W - max(0, dc)
        # Neighbour region
        nr0 = max(0, dr)
        nr1 = H - max(0, -dr)
        nc0 = max(0, dc)
        nc1 = W - max(0, -dc)

        lat1 = np.radians(lat_2d[r0:r1, c0:c1])
        lon1 = np.radians(lon_2d[r0:r1, c0:c1])
        lat2 = np.radians(lat_2d[nr0:nr1, nc0:nc1])
        lon2 = np.radians(lon_2d[nr0:nr1, nc0:nc1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
        dd_grid[i, r0:r1, c0:c1] = (
            EARTH_RADIUS * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        )

    return dd_grid


# ---------------------------------------------------------------------------
# NumPy wrapper
# ---------------------------------------------------------------------------


def _surface_distance_numpy(source_data, elev_data, cellsize_x, cellsize_y,
                            max_distance, target_values, dy, dx, dd,
                            dd_grid, use_geodesic, mode):
    """NumPy backend: run Dijkstra and extract requested output."""
    H, W = source_data.shape
    _check_memory(H, W)
    dist, alloc, src_row, src_col = _init_arrays(H, W)

    _seed_sources(source_data, elev_data, target_values,
                  dist, alloc, src_row, src_col)

    if use_geodesic:
        _dijkstra_geodesic(elev_data, H, W, max_distance,
                           dy, dx, dd_grid, dist, alloc, src_row, src_col)
    else:
        _dijkstra(elev_data, H, W, max_distance,
                  dy, dx, dd, dist, alloc, src_row, src_col)

    return _extract_output(dist, alloc, src_row, src_col,
                           cellsize_x, cellsize_y, max_distance, mode)


# ---------------------------------------------------------------------------
# CuPy GPU backend — iterative parallel relaxation
# ---------------------------------------------------------------------------


@cuda.jit
def _sd_relax_kernel(elev, dist, alloc, src_row, src_col, changed,
                     height, width, dy, dx, dd, n_neighbors,
                     max_distance):
    """One relaxation pass for surface distance.

    Each pixel checks all neighbours for shorter 3D paths.
    Iterate until *changed* stays 0.
    """
    iy, ix = cuda.grid(2)
    if iy >= height or ix >= width:
        return

    elev_u = elev[iy, ix]
    if not _math.isfinite(elev_u):
        return

    current = dist[iy, ix]
    best = current
    best_alloc = alloc[iy, ix]
    best_srow = src_row[iy, ix]
    best_scol = src_col[iy, ix]

    for k in range(n_neighbors):
        vy = iy + dy[k]
        vx = ix + dx[k]
        if vy < 0 or vy >= height or vx < 0 or vx >= width:
            continue

        d_v = dist[vy, vx]
        if d_v >= best:
            continue

        elev_v = elev[vy, vx]
        if not _math.isfinite(elev_v):
            continue

        dz = elev_u - elev_v
        edge_cost = _math.sqrt(dd[k] * dd[k] + dz * dz)
        new_cost = d_v + edge_cost

        if new_cost < best:
            best = new_cost
            best_alloc = alloc[vy, vx]
            best_srow = src_row[vy, vx]
            best_scol = src_col[vy, vx]

    if best < current and best <= max_distance:
        dist[iy, ix] = best
        alloc[iy, ix] = best_alloc
        src_row[iy, ix] = best_srow
        src_col[iy, ix] = best_scol
        changed[0] = 1


def _surface_distance_cupy(source_data, elev_data, cellsize_x, cellsize_y,
                           max_distance, target_values, dy, dx, dd, mode):
    """GPU surface distance via iterative parallel relaxation."""
    import cupy as cp

    H, W = source_data.shape
    _check_gpu_memory(H, W)
    src = source_data.astype(cp.float64)
    elev = elev_data.astype(cp.float64)

    dist = cp.full((H, W), cp.inf, dtype=cp.float64)
    alloc_arr = cp.full((H, W), cp.nan, dtype=cp.float64)
    srow = cp.full((H, W), -1, dtype=cp.int64)
    scol = cp.full((H, W), -1, dtype=cp.int64)

    # Seed sources
    if len(target_values) == 0:
        mask = cp.isfinite(src) & (src != 0) & cp.isfinite(elev)
    else:
        tv = cp.asarray(target_values, dtype=cp.float64)
        mask = cp.isin(src, tv) & cp.isfinite(elev)

    dist[mask] = 0.0
    alloc_arr[mask] = src[mask]
    rows_g, cols_g = cp.meshgrid(cp.arange(H, dtype=cp.int64),
                                 cp.arange(W, dtype=cp.int64),
                                 indexing='ij')
    srow[mask] = rows_g[mask]
    scol[mask] = cols_g[mask]

    if not cp.any(mask):
        out = cp.full((H, W), cp.nan, dtype=cp.float32)
        return out

    dy_d = cp.asarray(dy, dtype=cp.int64)
    dx_d = cp.asarray(dx, dtype=cp.int64)
    dd_d = cp.asarray(dd, dtype=cp.float64)
    n_neighbors = len(dy)

    changed = cp.zeros(1, dtype=cp.int32)
    griddim, blockdim = cuda_args((H, W))

    max_iterations = H + W
    for _ in range(max_iterations):
        changed[0] = 0
        _sd_relax_kernel[griddim, blockdim](
            elev, dist, alloc_arr, srow, scol, changed,
            H, W,
            dy_d, dx_d, dd_d, n_neighbors,
            np.float64(max_distance),
        )
        if int(changed[0]) == 0:
            break

    # Extract output
    if mode == DISTANCE:
        out = cp.where(cp.isinf(dist) | (dist > max_distance),
                       cp.nan, dist).astype(cp.float32)
    elif mode == ALLOCATION:
        out = cp.where(cp.isinf(dist) | (dist > max_distance),
                       cp.nan, alloc_arr).astype(cp.float32)
    else:  # DIRECTION
        row_idx, col_idx = cp.meshgrid(cp.arange(H, dtype=cp.float64),
                                       cp.arange(W, dtype=cp.float64),
                                       indexing='ij')
        dx_coord = (scol.astype(cp.float64) - col_idx) * cellsize_x
        dy_coord = (srow.astype(cp.float64) - row_idx) * cellsize_y
        # Compute direction on CPU (uses numpy-based vectorized function)
        dx_np = cp.asnumpy(dx_coord)
        dy_np = cp.asnumpy(dy_coord)
        zeros = np.zeros((H, W), dtype=np.float64)
        result = _vectorized_calc_direction(zeros, dx_np, zeros, dy_np)
        mask_np = cp.asnumpy(cp.isinf(dist) | (dist > max_distance)
                             | (srow < 0))
        result[mask_np] = np.nan
        out = cp.asarray(result)

    return out


# ---------------------------------------------------------------------------
# Dask bounded — map_overlap
# ---------------------------------------------------------------------------


def _make_sd_chunk_func(cellsize_x, cellsize_y, max_distance,
                        target_values, dy, dx, dd, mode):
    """Return a function for ``da.map_overlap`` over source + elev."""

    def _chunk(source_block, elev_block):
        return _surface_distance_numpy(
            source_block, elev_block,
            cellsize_x, cellsize_y, max_distance,
            target_values, dy, dx, dd,
            None, False, mode,
        )

    return _chunk


def _surface_distance_dask_bounded(source_da, elev_da,
                                   cellsize_x, cellsize_y,
                                   max_distance, target_values,
                                   dy, dx, dd, mode):
    """Dask bounded path via map_overlap."""
    min_cellsize = min(abs(cellsize_x), abs(cellsize_y))
    pad = int(max_distance / min_cellsize) + 1

    chunk_func = _make_sd_chunk_func(
        cellsize_x, cellsize_y, max_distance,
        target_values, dy, dx, dd, mode,
    )

    return da.map_overlap(
        chunk_func,
        source_da, elev_da,
        depth=(pad, pad),
        boundary=np.nan,
        dtype=np.float32,
        meta=np.array((), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Iterative boundary-only Dijkstra for dask arrays
# ---------------------------------------------------------------------------


def _preprocess_tiles_sd(source_da, elev_da, chunks_y, chunks_x,
                         target_values):
    """Extract elevation boundary strips and identify source tiles."""
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    n_values = len(target_values)

    elev_bdry = {
        side: [[None] * n_tile_x for _ in range(n_tile_y)]
        for side in ('top', 'bottom', 'left', 'right')
    }
    has_source = [[False] * n_tile_x for _ in range(n_tile_y)]

    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            echunk = elev_da.blocks[iy, ix].compute()
            elev_bdry['top'][iy][ix] = echunk[0, :].astype(np.float64)
            elev_bdry['bottom'][iy][ix] = echunk[-1, :].astype(np.float64)
            elev_bdry['left'][iy][ix] = echunk[:, 0].astype(np.float64)
            elev_bdry['right'][iy][ix] = echunk[:, -1].astype(np.float64)

            schunk = source_da.blocks[iy, ix].compute()
            if n_values == 0:
                has_source[iy][ix] = bool(
                    np.any((schunk != 0) & np.isfinite(schunk)
                           & np.isfinite(echunk))
                )
            else:
                for tv in target_values:
                    if np.any((schunk == tv) & np.isfinite(echunk)):
                        has_source[iy][ix] = True
                        break

    return elev_bdry, has_source


def _init_boundaries_sd(chunks_y, chunks_x):
    """Create boundary arrays: dist, alloc, src_row, src_col."""
    n_y = len(chunks_y)
    n_x = len(chunks_x)

    def _make(side_sizes, fill, dtype):
        return [[np.full(side_sizes(iy, ix), fill, dtype=dtype)
                 for ix in range(n_x)]
                for iy in range(n_y)]

    horiz = lambda iy, ix: chunks_x[ix]  # noqa: E731
    vert = lambda iy, ix: chunks_y[iy]  # noqa: E731

    boundaries = {}
    for key, fill, dtype in [
        ('dist', np.inf, np.float64),
        ('alloc', np.nan, np.float64),
        ('src_row', np.inf, np.float64),
        ('src_col', np.inf, np.float64),
    ]:
        boundaries[key] = {
            'top': _make(horiz, fill, dtype),
            'bottom': _make(horiz, fill, dtype),
            'left': _make(vert, fill, dtype),
            'right': _make(vert, fill, dtype),
        }
    return boundaries


def _compute_seeds_sd(iy, ix, boundaries, elev_bdry,
                      cellsize_x, cellsize_y, chunks_y, chunks_x,
                      n_tile_y, n_tile_x, connectivity):
    """Compute seed arrays for tile (iy, ix) from neighbour boundaries.

    Returns dict with keys 'dist', 'alloc', 'src_row', 'src_col',
    each containing (top, bottom, left, right, tl, tr, bl, br).
    Cardinal seeds are 1-D float64 arrays; corner seeds are float64 scalars.
    """
    tile_h = chunks_y[iy]
    tile_w = chunks_x[ix]
    diag_dist = sqrt(cellsize_x ** 2 + cellsize_y ** 2)

    # Initialize seeds
    seeds = {}
    for key in ('dist', 'alloc', 'src_row', 'src_col'):
        fill = np.inf if key == 'dist' else (np.nan if key == 'alloc'
                                             else np.inf)
        seeds[key] = {
            'top': np.full(tile_w, fill),
            'bottom': np.full(tile_w, fill),
            'left': np.full(tile_h, fill),
            'right': np.full(tile_h, fill),
            'tl': fill, 'tr': fill, 'bl': fill, 'br': fill,
        }

    my_top = elev_bdry['top'][iy][ix]
    my_bottom = elev_bdry['bottom'][iy][ix]
    my_left = elev_bdry['left'][iy][ix]
    my_right = elev_bdry['right'][iy][ix]

    def _edge_seeds(nb_dist, nb_elev, my_elev,
                    nb_alloc, nb_srow, nb_scol,
                    cardinal_dist):
        """Compute min-cost seed per boundary pixel, tracking alloc/src."""
        n = len(nb_dist)

        # Cardinal: pixel c <- pixel c in neighbour
        dz = my_elev - nb_elev
        cost = nb_dist + np.sqrt(cardinal_dist ** 2 + dz ** 2)
        valid = (np.isfinite(nb_dist) & np.isfinite(nb_elev)
                 & np.isfinite(my_elev))

        s_dist = np.where(valid, cost, np.inf)
        s_alloc = np.where(valid, nb_alloc, np.nan)
        s_srow = np.where(valid, nb_srow, np.inf)
        s_scol = np.where(valid, nb_scol, np.inf)

        if connectivity == 8 and n > 1:
            # Diagonal left: pixel c <- pixel c-1 in neighbour
            dz_l = my_elev[1:] - nb_elev[:-1]
            cost_l = nb_dist[:-1] + np.sqrt(diag_dist ** 2 + dz_l ** 2)
            valid_l = (np.isfinite(nb_dist[:-1]) & np.isfinite(nb_elev[:-1])
                       & np.isfinite(my_elev[1:]))
            cost_l = np.where(valid_l, cost_l, np.inf)
            better = cost_l < s_dist[1:]
            s_dist[1:] = np.where(better, cost_l, s_dist[1:])
            s_alloc[1:] = np.where(better, nb_alloc[:-1], s_alloc[1:])
            s_srow[1:] = np.where(better, nb_srow[:-1], s_srow[1:])
            s_scol[1:] = np.where(better, nb_scol[:-1], s_scol[1:])

            # Diagonal right: pixel c <- pixel c+1 in neighbour
            dz_r = my_elev[:-1] - nb_elev[1:]
            cost_r = nb_dist[1:] + np.sqrt(diag_dist ** 2 + dz_r ** 2)
            valid_r = (np.isfinite(nb_dist[1:]) & np.isfinite(nb_elev[1:])
                       & np.isfinite(my_elev[:-1]))
            cost_r = np.where(valid_r, cost_r, np.inf)
            better_r = cost_r < s_dist[:-1]
            s_dist[:-1] = np.where(better_r, cost_r, s_dist[:-1])
            s_alloc[:-1] = np.where(better_r, nb_alloc[1:], s_alloc[:-1])
            s_srow[:-1] = np.where(better_r, nb_srow[1:], s_srow[:-1])
            s_scol[:-1] = np.where(better_r, nb_scol[1:], s_scol[:-1])

        return s_dist, s_alloc, s_srow, s_scol

    def _get_bdry(key, side, iy_, ix_):
        return boundaries[key][side][iy_][ix_]

    # Edge neighbours
    if iy > 0:
        result = _edge_seeds(
            _get_bdry('dist', 'bottom', iy - 1, ix),
            elev_bdry['bottom'][iy - 1][ix],
            my_top,
            _get_bdry('alloc', 'bottom', iy - 1, ix),
            _get_bdry('src_row', 'bottom', iy - 1, ix),
            _get_bdry('src_col', 'bottom', iy - 1, ix),
            cellsize_y,
        )
        for i, key in enumerate(('dist', 'alloc', 'src_row', 'src_col')):
            seeds[key]['top'] = result[i]

    if iy < n_tile_y - 1:
        result = _edge_seeds(
            _get_bdry('dist', 'top', iy + 1, ix),
            elev_bdry['top'][iy + 1][ix],
            my_bottom,
            _get_bdry('alloc', 'top', iy + 1, ix),
            _get_bdry('src_row', 'top', iy + 1, ix),
            _get_bdry('src_col', 'top', iy + 1, ix),
            cellsize_y,
        )
        for i, key in enumerate(('dist', 'alloc', 'src_row', 'src_col')):
            seeds[key]['bottom'] = result[i]

    if ix > 0:
        result = _edge_seeds(
            _get_bdry('dist', 'right', iy, ix - 1),
            elev_bdry['right'][iy][ix - 1],
            my_left,
            _get_bdry('alloc', 'right', iy, ix - 1),
            _get_bdry('src_row', 'right', iy, ix - 1),
            _get_bdry('src_col', 'right', iy, ix - 1),
            cellsize_x,
        )
        for i, key in enumerate(('dist', 'alloc', 'src_row', 'src_col')):
            seeds[key]['left'] = result[i]

    if ix < n_tile_x - 1:
        result = _edge_seeds(
            _get_bdry('dist', 'left', iy, ix + 1),
            elev_bdry['left'][iy][ix + 1],
            my_right,
            _get_bdry('alloc', 'left', iy, ix + 1),
            _get_bdry('src_row', 'left', iy, ix + 1),
            _get_bdry('src_col', 'left', iy, ix + 1),
            cellsize_x,
        )
        for i, key in enumerate(('dist', 'alloc', 'src_row', 'src_col')):
            seeds[key]['right'] = result[i]

    # Diagonal corner seeds (8-connectivity only)
    if connectivity == 8:
        def _corner(nb_d, nb_e, my_e, nb_a, nb_sr, nb_sc):
            nb_d = float(nb_d)
            nb_e = float(nb_e)
            my_e = float(my_e)
            if (np.isfinite(nb_d) and np.isfinite(nb_e)
                    and np.isfinite(my_e)):
                dz = my_e - nb_e
                cost = nb_d + sqrt(diag_dist ** 2 + dz ** 2)
                return cost, float(nb_a), float(nb_sr), float(nb_sc)
            return np.inf, np.nan, np.inf, np.inf

        corners = [
            # (tile_offset_y, tile_offset_x, nb_side, nb_col_idx, my_elev_val,
            #  seed_key)
            (iy - 1, ix - 1, 'bottom', -1, my_top[0], 'tl'),
            (iy - 1, ix + 1, 'bottom', 0, my_top[-1], 'tr'),
            (iy + 1, ix - 1, 'top', -1, my_bottom[0], 'bl'),
            (iy + 1, ix + 1, 'top', 0, my_bottom[-1], 'br'),
        ]
        for niy, nix, nb_side, nb_idx, my_e, skey in corners:
            if 0 <= niy < n_tile_y and 0 <= nix < n_tile_x:
                nb_d = boundaries['dist'][nb_side][niy][nix][nb_idx]
                nb_e = elev_bdry[nb_side][niy][nix][nb_idx]
                nb_a = boundaries['alloc'][nb_side][niy][nix][nb_idx]
                nb_sr = boundaries['src_row'][nb_side][niy][nix][nb_idx]
                nb_sc = boundaries['src_col'][nb_side][niy][nix][nb_idx]
                cd, ca, csr, csc = _corner(nb_d, nb_e, my_e,
                                           nb_a, nb_sr, nb_sc)
                seeds['dist'][skey] = cd
                seeds['alloc'][skey] = ca
                seeds['src_row'][skey] = csr
                seeds['src_col'][skey] = csc

    return seeds


def _can_skip_sd(iy, ix, has_source, boundaries,
                 n_tile_y, n_tile_x, connectivity):
    """True when a tile cannot possibly receive any distance information."""
    if has_source[iy][ix]:
        return False
    bdist = boundaries['dist']
    if iy > 0 and np.any(np.isfinite(bdist['bottom'][iy - 1][ix])):
        return False
    if (iy < n_tile_y - 1
            and np.any(np.isfinite(bdist['top'][iy + 1][ix]))):
        return False
    if ix > 0 and np.any(np.isfinite(bdist['right'][iy][ix - 1])):
        return False
    if (ix < n_tile_x - 1
            and np.any(np.isfinite(bdist['left'][iy][ix + 1]))):
        return False
    if connectivity == 8:
        if (iy > 0 and ix > 0
                and np.isfinite(bdist['bottom'][iy - 1][ix - 1][-1])):
            return False
        if (iy > 0 and ix < n_tile_x - 1
                and np.isfinite(bdist['bottom'][iy - 1][ix + 1][0])):
            return False
        if (iy < n_tile_y - 1 and ix > 0
                and np.isfinite(bdist['top'][iy + 1][ix - 1][-1])):
            return False
        if (iy < n_tile_y - 1 and ix < n_tile_x - 1
                and np.isfinite(bdist['top'][iy + 1][ix + 1][0])):
            return False
    return True


def _run_tile(source_chunk, elev_chunk, seeds, target_values,
              max_distance, dy, dx, dd, row_offset, col_offset):
    """Run Dijkstra on one tile with boundary seeds.

    Returns (dist, alloc, src_row, src_col) as raw float64/int64 arrays.
    src_row/src_col are global indices.
    """
    h, w = source_chunk.shape
    dist, alloc, src_row, src_col = _init_arrays(h, w)

    # Seed source pixels (using global coords for src)
    n_values = len(target_values)
    for r in range(h):
        for c in range(w):
            val = source_chunk[r, c]
            is_target = False
            if n_values == 0:
                if val != 0.0 and np.isfinite(val):
                    is_target = True
            else:
                for k in range(n_values):
                    if val == target_values[k]:
                        is_target = True
                        break
            if is_target and np.isfinite(elev_chunk[r, c]):
                dist[r, c] = 0.0
                alloc[r, c] = val
                src_row[r, c] = r + row_offset
                src_col[r, c] = c + col_offset

    # Seed boundary pixels from neighbour tiles
    sd = seeds['dist']
    sa = seeds['alloc']
    ssr = seeds['src_row']
    ssc = seeds['src_col']

    # Top edge
    for c in range(w):
        if sd['top'][c] < dist[0, c] and np.isfinite(elev_chunk[0, c]):
            dist[0, c] = sd['top'][c]
            alloc[0, c] = sa['top'][c]
            src_row[0, c] = int(ssr['top'][c])
            src_col[0, c] = int(ssc['top'][c])
    # Bottom edge
    for c in range(w):
        if sd['bottom'][c] < dist[h - 1, c] and np.isfinite(
                elev_chunk[h - 1, c]):
            dist[h - 1, c] = sd['bottom'][c]
            alloc[h - 1, c] = sa['bottom'][c]
            src_row[h - 1, c] = int(ssr['bottom'][c])
            src_col[h - 1, c] = int(ssc['bottom'][c])
    # Left edge
    for r in range(h):
        if sd['left'][r] < dist[r, 0] and np.isfinite(elev_chunk[r, 0]):
            dist[r, 0] = sd['left'][r]
            alloc[r, 0] = sa['left'][r]
            src_row[r, 0] = int(ssr['left'][r])
            src_col[r, 0] = int(ssc['left'][r])
    # Right edge
    for r in range(h):
        if sd['right'][r] < dist[r, w - 1] and np.isfinite(
                elev_chunk[r, w - 1]):
            dist[r, w - 1] = sd['right'][r]
            alloc[r, w - 1] = sa['right'][r]
            src_row[r, w - 1] = int(ssr['right'][r])
            src_col[r, w - 1] = int(ssc['right'][r])

    # Corner seeds
    _corners = [
        (0, 0, 'tl'),
        (0, w - 1, 'tr'),
        (h - 1, 0, 'bl'),
        (h - 1, w - 1, 'br'),
    ]
    for cr, cc, skey in _corners:
        sv = sd[skey]
        if sv < dist[cr, cc] and np.isfinite(elev_chunk[cr, cc]):
            dist[cr, cc] = sv
            alloc[cr, cc] = sa[skey]
            src_row[cr, cc] = int(ssr[skey])
            src_col[cr, cc] = int(ssc[skey])

    # Run Dijkstra
    _dijkstra(elev_chunk, h, w, max_distance,
              dy, dx, dd, dist, alloc, src_row, src_col)

    return dist, alloc, src_row, src_col


def _process_tile_sd(iy, ix, source_da, elev_da,
                     boundaries, elev_bdry,
                     cellsize_x, cellsize_y, max_distance, target_values,
                     dy, dx, dd, chunks_y, chunks_x,
                     n_tile_y, n_tile_x, connectivity,
                     cumul_rows, cumul_cols):
    """Run seeded Dijkstra on one tile; update boundaries in-place.

    Returns the maximum absolute boundary distance change.
    """
    source_chunk = source_da.blocks[iy, ix].compute()
    elev_chunk = elev_da.blocks[iy, ix].compute()
    h, w = source_chunk.shape

    row_offset = cumul_rows[iy]
    col_offset = cumul_cols[ix]

    seeds = _compute_seeds_sd(
        iy, ix, boundaries, elev_bdry,
        cellsize_x, cellsize_y, chunks_y, chunks_x,
        n_tile_y, n_tile_x, connectivity,
    )

    dist, alloc_arr, srow, scol = _run_tile(
        source_chunk, elev_chunk, seeds, target_values,
        max_distance, dy, dx, dd, row_offset, col_offset,
    )

    # Extract new boundary strips
    change = 0.0
    for side, sl in [
        ('top', (0, slice(None))),
        ('bottom', (-1, slice(None))),
        ('left', (slice(None), 0)),
        ('right', (slice(None), -1)),
    ]:
        new_d = dist[sl]
        old_d = boundaries['dist'][side][iy][ix]
        with np.errstate(invalid='ignore'):
            diff = np.abs(new_d - old_d)
        diff = np.where(np.isnan(diff), 0.0, diff)
        m = float(np.max(diff))
        if m > change:
            change = m

        boundaries['dist'][side][iy][ix] = new_d.copy()
        boundaries['alloc'][side][iy][ix] = alloc_arr[sl].copy()
        boundaries['src_row'][side][iy][ix] = srow[sl].astype(
            np.float64).copy()
        boundaries['src_col'][side][iy][ix] = scol[sl].astype(
            np.float64).copy()

    return change


def _sd_dask_iterative(source_da, elev_da,
                       cellsize_x, cellsize_y,
                       max_distance, target_values,
                       dy, dx, dd, mode):
    """Iterative boundary-only Dijkstra for arbitrarily large dask arrays."""
    connectivity = len(dy)
    chunks_y = source_da.chunks[0]
    chunks_x = source_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    # Cumulative row/col offsets for global indexing
    cumul_rows = [0]
    for cy in chunks_y[:-1]:
        cumul_rows.append(cumul_rows[-1] + cy)
    cumul_cols = [0]
    for cx in chunks_x[:-1]:
        cumul_cols.append(cumul_cols[-1] + cx)

    # Phase 0: pre-extract elevation boundaries & source flags
    elev_bdry, has_source = _preprocess_tiles_sd(
        source_da, elev_da, chunks_y, chunks_x, target_values,
    )

    # Phase 1: initialise boundaries
    boundaries = _init_boundaries_sd(chunks_y, chunks_x)

    # Phase 2: iterative sweeps
    max_iterations = max(n_tile_y, n_tile_x) + 10
    args = (source_da, elev_da, boundaries, elev_bdry,
            cellsize_x, cellsize_y, max_distance, target_values,
            dy, dx, dd, chunks_y, chunks_x,
            n_tile_y, n_tile_x, connectivity, cumul_rows, cumul_cols)

    for _iteration in range(max_iterations):
        max_change = 0.0

        # Forward sweep
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                if _can_skip_sd(iy, ix, has_source, boundaries,
                                n_tile_y, n_tile_x, connectivity):
                    continue
                c = _process_tile_sd(iy, ix, *args)
                if c > max_change:
                    max_change = c

        # Backward sweep
        for iy in reversed(range(n_tile_y)):
            for ix in reversed(range(n_tile_x)):
                if _can_skip_sd(iy, ix, has_source, boundaries,
                                n_tile_y, n_tile_x, connectivity):
                    continue
                c = _process_tile_sd(iy, ix, *args)
                if c > max_change:
                    max_change = c

        if max_change == 0.0:
            break

    # Phase 3: lazy final assembly
    return _assemble_sd(
        source_da, elev_da, boundaries, elev_bdry,
        cellsize_x, cellsize_y, max_distance, target_values,
        dy, dx, dd, chunks_y, chunks_x,
        n_tile_y, n_tile_x, connectivity,
        cumul_rows, cumul_cols, mode,
    )


def _assemble_sd(source_da, elev_da, boundaries, elev_bdry,
                 cellsize_x, cellsize_y, max_distance, target_values,
                 dy, dx, dd, chunks_y, chunks_x,
                 n_tile_y, n_tile_x, connectivity,
                 cumul_rows, cumul_cols, mode):
    """Build lazy dask array by re-running each tile with converged seeds."""

    def _tile_fn(source_block, elev_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(source_block.shape, np.nan, dtype=np.float32)
        iy, ix = block_info[0]['chunk-location']
        h, w = source_block.shape

        row_offset = cumul_rows[iy]
        col_offset = cumul_cols[ix]

        seeds = _compute_seeds_sd(
            iy, ix, boundaries, elev_bdry,
            cellsize_x, cellsize_y, chunks_y, chunks_x,
            n_tile_y, n_tile_x, connectivity,
        )

        dist, alloc_arr, srow, scol = _run_tile(
            source_block, elev_block, seeds, target_values,
            max_distance, dy, dx, dd, row_offset, col_offset,
        )

        return _extract_output(dist, alloc_arr, srow, scol,
                               cellsize_x, cellsize_y, max_distance, mode)

    return da.map_blocks(
        _tile_fn,
        source_da, elev_da,
        dtype=np.float32,
        meta=np.array((), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Dask wrapper
# ---------------------------------------------------------------------------


def _surface_distance_dask(source_da, elev_da, cellsize_x, cellsize_y,
                           max_distance, target_values, dy, dx, dd, mode):
    """Dask path: use map_overlap for bounded, iterative for unbounded."""
    min_cellsize = min(abs(cellsize_x), abs(cellsize_y))
    height, width = source_da.shape

    use_overlap = False
    if np.isfinite(max_distance):
        pad = int(max_distance / min_cellsize) + 1
        chunks_y, chunks_x = source_da.chunks
        if pad < max(chunks_y) and pad < max(chunks_x):
            use_overlap = True

    if use_overlap:
        return _surface_distance_dask_bounded(
            source_da, elev_da, cellsize_x, cellsize_y,
            max_distance, target_values, dy, dx, dd, mode,
        )

    warnings.warn(
        "surface_distance: max_distance is infinite or the implied radius "
        "exceeds chunk dimensions; using iterative tile Dijkstra. "
        "Setting a finite max_distance enables faster single-pass "
        "processing.",
        UserWarning,
        stacklevel=4,
    )
    return _sd_dask_iterative(
        source_da, elev_da, cellsize_x, cellsize_y,
        max_distance, target_values, dy, dx, dd, mode,
    )


# ---------------------------------------------------------------------------
# Dask + CuPy wrapper
# ---------------------------------------------------------------------------


def _surface_distance_dask_cupy(source_da, elev_da,
                                cellsize_x, cellsize_y,
                                max_distance, target_values,
                                dy, dx, dd, mode):
    """Dask+CuPy surface distance.

    Bounded max_distance: map_overlap with per-chunk GPU relaxation.
    Unbounded: convert to dask+numpy, use CPU iterative path.
    """
    import cupy as cp

    min_cellsize = min(abs(cellsize_x), abs(cellsize_y))
    use_overlap = False
    if np.isfinite(max_distance):
        pad = int(max_distance / min_cellsize) + 1
        chunks_y, chunks_x = source_da.chunks
        if pad < max(chunks_y) and pad < max(chunks_x):
            use_overlap = True

    if use_overlap:
        _dy, _dx, _dd = dy, dx, dd
        _mode = mode
        cx, cy = cellsize_x, cellsize_y
        md, tv = max_distance, target_values

        def _chunk_func(source_block, elev_block):
            return _surface_distance_cupy(
                source_block, elev_block, cx, cy,
                md, tv, _dy, _dx, _dd, _mode,
            )

        return da.map_overlap(
            _chunk_func,
            source_da, elev_da,
            depth=(pad, pad),
            boundary=np.nan,
            dtype=np.float32,
            meta=cp.array((), dtype=cp.float32),
        )

    # Unbounded: convert to dask+numpy, use CPU path
    source_np = source_da.map_blocks(
        lambda b: b.get(), dtype=source_da.dtype,
        meta=np.array((), dtype=source_da.dtype),
    )
    elev_np = elev_da.map_blocks(
        lambda b: b.get(), dtype=elev_da.dtype,
        meta=np.array((), dtype=elev_da.dtype),
    )
    result = _surface_distance_dask(
        source_np, elev_np, cellsize_x, cellsize_y,
        max_distance, target_values, dy, dx, dd, mode,
    )
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# ---------------------------------------------------------------------------
# Core dispatcher
# ---------------------------------------------------------------------------


def _compute(raster, elevation, x, y, target_values, max_distance,
             connectivity, method, mode):
    """Core dispatcher for surface_distance / allocation / direction."""
    _validate_raster(raster, func_name='surface_distance', name='raster')
    _validate_raster(elevation, func_name='surface_distance',
                     name='elevation')
    if raster.shape != elevation.shape:
        raise ValueError("raster and elevation must have the same shape")
    if raster.dims != (y, x):
        raise ValueError(
            f"raster.dims should be ({y!r}, {x!r}), got {raster.dims}"
        )
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if method not in ('planar', 'geodesic'):
        raise ValueError("method must be 'planar' or 'geodesic'")

    cellsize_x, cellsize_y = get_dataarray_resolution(raster)
    cellsize_x = abs(float(cellsize_x))
    cellsize_y = abs(float(cellsize_y))

    target_values = np.asarray(target_values, dtype=np.float64)
    max_distance_f = float(max_distance)

    # Build neighbour offsets
    if connectivity == 8:
        dy_arr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
        dx_arr = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
        dd_arr = np.array([
            sqrt(cellsize_y ** 2 + cellsize_x ** 2),
            cellsize_y,
            sqrt(cellsize_y ** 2 + cellsize_x ** 2),
            cellsize_x,
            cellsize_x,
            sqrt(cellsize_y ** 2 + cellsize_x ** 2),
            cellsize_y,
            sqrt(cellsize_y ** 2 + cellsize_x ** 2),
        ], dtype=np.float64)
    else:
        dy_arr = np.array([0, -1, 1, 0], dtype=np.int64)
        dx_arr = np.array([-1, 0, 0, 1], dtype=np.int64)
        dd_arr = np.array(
            [cellsize_x, cellsize_y, cellsize_y, cellsize_x],
            dtype=np.float64)

    # Geodesic dd_grid precomputation (numpy only for now)
    use_geodesic = (method == 'geodesic')
    dd_grid = None
    if use_geodesic:
        from xrspatial.utils import _extract_latlon_coords
        lat_2d, lon_2d = _extract_latlon_coords(raster)
        dd_grid = _precompute_dd_grid(lat_2d, lon_2d, dy_arr, dx_arr)

    source_data = raster.data
    elev_data = elevation.data

    _is_dask = da is not None and isinstance(source_data, da.Array)
    _is_cupy = (
        not _is_dask
        and has_cuda_and_cupy()
        and is_cupy_array(source_data)
    )
    _is_dask_cupy_flag = _is_dask and has_cuda_and_cupy() and is_dask_cupy(
        raster)

    if _is_dask:
        # Ensure chunks match
        if isinstance(elev_data, da.Array):
            elev_data = elev_data.rechunk(source_data.chunks)
        else:
            elev_data = da.from_array(elev_data, chunks=source_data.chunks)

    if _is_cupy:
        if use_geodesic:
            raise NotImplementedError(
                "geodesic mode is not yet supported for CuPy arrays")
        result_data = _surface_distance_cupy(
            source_data, elev_data, cellsize_x, cellsize_y,
            max_distance_f, target_values, dy_arr, dx_arr, dd_arr, mode,
        )
    elif _is_dask_cupy_flag:
        if use_geodesic:
            raise NotImplementedError(
                "geodesic mode is not yet supported for Dask+CuPy arrays")
        result_data = _surface_distance_dask_cupy(
            source_data, elev_data, cellsize_x, cellsize_y,
            max_distance_f, target_values, dy_arr, dx_arr, dd_arr, mode,
        )
    elif isinstance(source_data, np.ndarray):
        if isinstance(elev_data, np.ndarray):
            result_data = _surface_distance_numpy(
                source_data, elev_data, cellsize_x, cellsize_y,
                max_distance_f, target_values, dy_arr, dx_arr, dd_arr,
                dd_grid, use_geodesic, mode,
            )
        else:
            elev_np = np.asarray(elev_data)
            result_data = _surface_distance_numpy(
                source_data, elev_np, cellsize_x, cellsize_y,
                max_distance_f, target_values, dy_arr, dx_arr, dd_arr,
                dd_grid, use_geodesic, mode,
            )
    elif _is_dask:
        if use_geodesic:
            raise NotImplementedError(
                "geodesic mode is not yet supported for Dask arrays")
        result_data = _surface_distance_dask(
            source_data, elev_data, cellsize_x, cellsize_y,
            max_distance_f, target_values, dy_arr, dx_arr, dd_arr, mode,
        )
    else:
        raise TypeError(f"Unsupported array type: {type(source_data)}")

    return result_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@supports_dataset
def surface_distance(
    raster: xr.DataArray,
    elevation: xr.DataArray,
    x: str = "x",
    y: str = "y",
    target_values: list = [],
    max_distance: float = np.inf,
    connectivity: int = 8,
    method: str = 'planar',
) -> xr.DataArray:
    """Compute surface distance along terrain to nearest target pixel.

    For every pixel, computes the minimum accumulated distance along
    the 3D terrain surface to reach the nearest target pixel.
    Edge cost accounts for both horizontal distance and elevation
    change: ``sqrt(horizontal_dist^2 + dz^2)``.

    Parameters
    ----------
    raster : xr.DataArray or xr.Dataset
        2-D source raster.  Target pixels are identified by non-zero
        finite values (or values in *target_values*).
    elevation : xr.DataArray
        2-D elevation surface.  Must have the same shape as *raster*.
        NaN marks impassable barriers.
    x : str, default='x'
        Name of the x coordinate.
    y : str, default='y'
        Name of the y coordinate.
    target_values : list, optional
        Specific pixel values in *raster* to treat as sources.
        If empty, all non-zero finite pixels are sources.
    max_distance : float, default=np.inf
        Maximum surface distance.  Pixels beyond this budget are NaN.
        A finite value enables efficient Dask parallelisation.
    connectivity : int, default=8
        Pixel connectivity: 4 (cardinal) or 8 (cardinal + diagonal).
    method : str, default='planar'
        ``'planar'`` uses cell sizes in map units.
        ``'geodesic'`` computes great-circle horizontal distances
        from lat/lon coordinates (elevation in meters).

    Returns
    -------
    xr.DataArray or xr.Dataset
        2-D array of surface distance values (float32).
        Source pixels have distance 0.  Unreachable pixels are NaN.
    """
    result_data = _compute(
        raster, elevation, x, y, target_values, max_distance,
        connectivity, method, DISTANCE,
    )
    return xr.DataArray(
        result_data,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )


@supports_dataset
def surface_allocation(
    raster: xr.DataArray,
    elevation: xr.DataArray,
    x: str = "x",
    y: str = "y",
    target_values: list = [],
    max_distance: float = np.inf,
    connectivity: int = 8,
    method: str = 'planar',
) -> xr.DataArray:
    """Compute nearest-target allocation along terrain surface.

    For each pixel, returns the value of the nearest target pixel by
    surface distance through the elevation model.

    Parameters
    ----------
    raster : xr.DataArray or xr.Dataset
        2-D source raster with target pixels.
    elevation : xr.DataArray
        2-D elevation surface (same shape as *raster*).
    x, y, target_values, max_distance, connectivity, method
        See :func:`surface_distance`.

    Returns
    -------
    xr.DataArray or xr.Dataset
        2-D array of allocation values (float32).
    """
    result_data = _compute(
        raster, elevation, x, y, target_values, max_distance,
        connectivity, method, ALLOCATION,
    )
    return xr.DataArray(
        result_data,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )


@supports_dataset
def surface_direction(
    raster: xr.DataArray,
    elevation: xr.DataArray,
    x: str = "x",
    y: str = "y",
    target_values: list = [],
    max_distance: float = np.inf,
    connectivity: int = 8,
    method: str = 'planar',
) -> xr.DataArray:
    """Compute compass direction to nearest target along terrain surface.

    For each pixel, returns the compass direction (in degrees) to the
    nearest target pixel by surface distance.  0 = source pixel,
    90 = east, 180 = south, 270 = west, 360 = north.

    Parameters
    ----------
    raster : xr.DataArray or xr.Dataset
        2-D source raster with target pixels.
    elevation : xr.DataArray
        2-D elevation surface (same shape as *raster*).
    x, y, target_values, max_distance, connectivity, method
        See :func:`surface_distance`.

    Returns
    -------
    xr.DataArray or xr.Dataset
        2-D array of direction values (float32, degrees).
    """
    result_data = _compute(
        raster, elevation, x, y, target_values, max_distance,
        connectivity, method, DIRECTION,
    )
    return xr.DataArray(
        result_data,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )
