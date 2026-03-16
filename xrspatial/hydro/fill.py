"""Fill depressions in a DEM using Planchon-Darboux iterative flooding.

Initializes interior cells to +inf, boundary cells to DEM elevation,
then iterates fill[r,c] = max(dem[r,c], min(fill[neighbors])) with
forward and backward scans until convergence.

References
----------
Planchon, O. and Darboux, F. (2001). A fast, simple and versatile
algorithm to fill the depressions of digital elevation models.
Catena, 46(2-3), 159-176.
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
from xrspatial.hydro._boundary_store import BoundaryStore
from xrspatial.dataset_support import supports_dataset


# =====================================================================
# CPU tile kernel
# =====================================================================

@ngjit
def _fill_tile_kernel(dem, h, w, ring):
    """Planchon-Darboux fill for a single tile.

    Parameters
    ----------
    dem : ndarray, shape (h, w)
        Elevation values.
    ring : ndarray, shape (h+2, w+2)
        Constraint ring.  ring[r+1, c+1] maps to dem[r, c].
        Border values hold external constraints:
        -1e308 for global boundary (water escapes at DEM elevation),
        adjacent tile's fill value for tile boundary.
    """
    fill = np.empty((h, w), dtype=np.float64)

    for r in range(h):
        for c in range(w):
            v = dem[r, c]
            if v != v:  # NaN
                fill[r, c] = np.nan
            else:
                fill[r, c] = np.inf

    dy = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dx = np.array([-1, 0, 1, -1, 1, -1, 0, 1])

    changed = True
    while changed:
        changed = False

        # Forward scan (top-left to bottom-right)
        for r in range(h):
            for c in range(w):
                f = fill[r, c]
                if f != f:  # NaN
                    continue
                d = dem[r, c]

                min_nb = np.inf
                for k in range(8):
                    nr = r + dy[k]
                    nc = c + dx[k]
                    if 0 <= nr < h and 0 <= nc < w:
                        nbv = fill[nr, nc]
                    else:
                        nbv = ring[nr + 1, nc + 1]
                    if nbv != nbv:  # NaN
                        continue
                    if nbv < min_nb:
                        min_nb = nbv

                new_val = d if d > min_nb else min_nb
                if new_val < f:
                    fill[r, c] = new_val
                    changed = True

        # Backward scan (bottom-right to top-left)
        for r in range(h - 1, -1, -1):
            for c in range(w - 1, -1, -1):
                f = fill[r, c]
                if f != f:
                    continue
                d = dem[r, c]

                min_nb = np.inf
                for k in range(8):
                    nr = r + dy[k]
                    nc = c + dx[k]
                    if 0 <= nr < h and 0 <= nc < w:
                        nbv = fill[nr, nc]
                    else:
                        nbv = ring[nr + 1, nc + 1]
                    if nbv != nbv:
                        continue
                    if nbv < min_nb:
                        min_nb = nbv

                new_val = d if d > min_nb else min_nb
                if new_val < f:
                    fill[r, c] = new_val
                    changed = True

    # Revert cells still at inf to DEM (isolated cells surrounded by NaN)
    for r in range(h):
        for c in range(w):
            if fill[r, c] == np.inf:
                fill[r, c] = dem[r, c]

    return fill


# =====================================================================
# GPU kernels
# =====================================================================

@cuda.jit
def _fill_init_gpu(dem, fill, H, W):
    """Boundary cells = DEM, interior cells = large sentinel."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return
    v = dem[i, j]
    if v != v:  # NaN
        fill[i, j] = v
        return
    if i == 0 or i == H - 1 or j == 0 or j == W - 1:
        fill[i, j] = v
    else:
        fill[i, j] = 1e308


@cuda.jit
def _fill_iterate_gpu(dem, fill, changed, H, W):
    """One P-D iteration: fill[i,j] = max(dem, min(fill[neighbors]))."""
    i, j = cuda.grid(2)
    if i >= H or j >= W:
        return

    # Boundary cells are fixed at DEM elevation
    if i == 0 or i == H - 1 or j == 0 or j == W - 1:
        return

    f = fill[i, j]
    if f != f:  # NaN
        return

    d = dem[i, j]

    min_nb = 1e308
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
        nb = fill[ni, nj]
        if nb != nb:  # NaN
            continue
        if nb < min_nb:
            min_nb = nb

    new_val = d if d > min_nb else min_nb
    if new_val < f:
        fill[i, j] = new_val
        cuda.atomic.add(changed, 0, 1)


def _fill_cupy(dem_data):
    """GPU driver for Planchon-Darboux fill."""
    import cupy as cp

    H, W = dem_data.shape
    dem_f64 = dem_data.astype(cp.float64)
    fill = cp.empty((H, W), dtype=cp.float64)
    changed = cp.zeros(1, dtype=cp.int32)

    griddim, blockdim = cuda_args((H, W))
    _fill_init_gpu[griddim, blockdim](dem_f64, fill, H, W)

    max_iter = max(H, W) * 2
    for _ in range(max_iter):
        changed[0] = 0
        _fill_iterate_gpu[griddim, blockdim](dem_f64, fill, changed, H, W)
        if int(changed[0]) == 0:
            break

    # Revert cells still at sentinel to DEM (isolated cells)
    return cp.where(fill > 1e307, dem_f64, fill)


# =====================================================================
# Dask iterative tile sweep
# =====================================================================

def _build_constraint_ring(iy, ix, boundaries, chunks_y, chunks_x,
                            n_tile_y, n_tile_x):
    """Build (h+2, w+2) constraint ring for tile (iy, ix).

    Grid boundaries use -1e308 (water escapes at DEM elevation).
    Tile boundaries use the adjacent tile's boundary fill values.
    """
    h = chunks_y[iy]
    w = chunks_x[ix]
    ring = np.full((h + 2, w + 2), np.inf, dtype=np.float64)

    # Edges (not including corners)
    if iy > 0:
        ring[0, 1:-1] = boundaries.get('bottom', iy - 1, ix)
    else:
        ring[0, 1:-1] = -1e308

    if iy < n_tile_y - 1:
        ring[-1, 1:-1] = boundaries.get('top', iy + 1, ix)
    else:
        ring[-1, 1:-1] = -1e308

    if ix > 0:
        ring[1:-1, 0] = boundaries.get('right', iy, ix - 1)
    else:
        ring[1:-1, 0] = -1e308

    if ix < n_tile_x - 1:
        ring[1:-1, -1] = boundaries.get('left', iy, ix + 1)
    else:
        ring[1:-1, -1] = -1e308

    # Corners
    if iy > 0 and ix > 0:
        ring[0, 0] = boundaries.get('bottom', iy - 1, ix - 1)[-1]
    else:
        ring[0, 0] = -1e308

    if iy > 0 and ix < n_tile_x - 1:
        ring[0, -1] = boundaries.get('bottom', iy - 1, ix + 1)[0]
    else:
        ring[0, -1] = -1e308

    if iy < n_tile_y - 1 and ix > 0:
        ring[-1, 0] = boundaries.get('top', iy + 1, ix - 1)[-1]
    else:
        ring[-1, 0] = -1e308

    if iy < n_tile_y - 1 and ix < n_tile_x - 1:
        ring[-1, -1] = boundaries.get('top', iy + 1, ix + 1)[0]
    else:
        ring[-1, -1] = -1e308

    return ring


def _process_fill_tile(iy, ix, dem_da, boundaries,
                        chunks_y, chunks_x, n_tile_y, n_tile_x):
    """Run P-D on one tile; update boundaries in-place.

    Returns the maximum absolute boundary change (float).
    """
    chunk = np.asarray(
        dem_da.blocks[iy, ix].compute(), dtype=np.float64)
    h, w = chunk.shape

    ring = _build_constraint_ring(iy, ix, boundaries,
                                   chunks_y, chunks_x,
                                   n_tile_y, n_tile_x)

    result = _fill_tile_kernel(chunk, h, w, ring)

    new_top = result[0, :].copy()
    new_bottom = result[-1, :].copy()
    new_left = result[:, 0].copy()
    new_right = result[:, -1].copy()

    change = 0.0
    for side, new in (('top', new_top), ('bottom', new_bottom),
                      ('left', new_left), ('right', new_right)):
        old = boundaries.get(side, iy, ix)
        with np.errstate(invalid='ignore'):
            diff = np.abs(new - old)
        diff = np.where(np.isnan(diff), 0.0, diff)
        m = float(np.max(diff))
        if m > change:
            change = m

    boundaries.set('top', iy, ix, new_top)
    boundaries.set('bottom', iy, ix, new_bottom)
    boundaries.set('left', iy, ix, new_left)
    boundaries.set('right', iy, ix, new_right)

    return change


def _tile_neighbors(iy, ix, n_tile_y, n_tile_x):
    """Yield valid (iy, ix) neighbors of a tile (8-connected)."""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = iy + dy, ix + dx
            if 0 <= ny < n_tile_y and 0 <= nx < n_tile_x:
                yield (ny, nx)


def _fill_dask_iterative(dem_da):
    """Iterative boundary-propagation for P-D fill on dask arrays.

    Uses a dirty-tile set to avoid reprocessing tiles whose
    neighborhood hasn't changed.  Only grid-edge tiles are dirty
    initially (they're the only ones that can produce finite fill
    values on the first pass).  When a tile's boundary values change,
    its neighbors are marked dirty for the next round.

    This reduces total tile reads from O(n_tiles * diameter) to
    O(n_tiles * k) where k is a small constant for typical DEMs.
    """
    chunks_y = dem_da.chunks[0]
    chunks_x = dem_da.chunks[1]
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)

    boundaries = BoundaryStore(chunks_y, chunks_x, fill_value=np.inf)

    # Seed: grid-edge tiles are the only ones that can drain on the
    # first pass (their ring contains -1e308 grid-boundary sentinels).
    dirty = set()
    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            if iy == 0 or iy == n_tile_y - 1 or ix == 0 or ix == n_tile_x - 1:
                dirty.add((iy, ix))

    max_iterations = (n_tile_y + n_tile_x) * 2 + 10

    for _iteration in range(max_iterations):
        if not dirty:
            break

        next_dirty = set()

        # Forward pass over dirty tiles (sorted for cache-friendly order)
        for iy, ix in sorted(dirty):
            c = _process_fill_tile(iy, ix, dem_da, boundaries,
                                   chunks_y, chunks_x,
                                   n_tile_y, n_tile_x)
            if c > 0.0:
                for nb in _tile_neighbors(iy, ix, n_tile_y, n_tile_x):
                    next_dirty.add(nb)

        # Backward pass over newly-dirty tiles
        backward_dirty = set()
        for iy, ix in sorted(next_dirty, reverse=True):
            c = _process_fill_tile(iy, ix, dem_da, boundaries,
                                   chunks_y, chunks_x,
                                   n_tile_y, n_tile_x)
            if c > 0.0:
                for nb in _tile_neighbors(iy, ix, n_tile_y, n_tile_x):
                    backward_dirty.add(nb)

        dirty = backward_dirty

    # Snapshot converged boundaries before assembly (releases temp files)
    boundaries = boundaries.snapshot()

    return _assemble_fill_result(dem_da, boundaries,
                                  chunks_y, chunks_x,
                                  n_tile_y, n_tile_x)


def _assemble_fill_result(dem_da, boundaries,
                           chunks_y, chunks_x,
                           n_tile_y, n_tile_x):
    """Build lazy dask array by re-running tiles with converged constraints."""

    def _tile_fn(dem_block, block_info=None):
        if block_info is None or 0 not in block_info:
            return np.full(dem_block.shape, np.nan, dtype=np.float64)
        iy, ix = block_info[0]['chunk-location']
        h, w = dem_block.shape
        ring = _build_constraint_ring(iy, ix, boundaries,
                                       chunks_y, chunks_x,
                                       n_tile_y, n_tile_x)
        return _fill_tile_kernel(
            np.asarray(dem_block, dtype=np.float64), h, w, ring)

    return da.map_blocks(
        _tile_fn, dem_da,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _fill_dask_cupy(dem_data):
    """Dask+CuPy: convert to numpy, run CPU iterative path, convert back."""
    import cupy as cp

    dem_np = dem_data.map_blocks(
        lambda b: b.get(), dtype=dem_data.dtype,
        meta=np.array((), dtype=dem_data.dtype),
    )
    result = _fill_dask_iterative(dem_np)
    return result.map_blocks(
        cp.asarray, dtype=result.dtype,
        meta=cp.array((), dtype=result.dtype),
    )


# =====================================================================
# Public API
# =====================================================================

@supports_dataset
def fill(dem: xr.DataArray,
         z_limit=None,
         name: str = 'fill') -> xr.DataArray:
    """Fill depressions in a DEM using Planchon-Darboux iterative flooding.

    Raises each depression cell to the elevation of its pour point
    (the lowest point on the rim through which water can escape).

    Parameters
    ----------
    dem : xarray.DataArray or xr.Dataset
        2D elevation raster (NumPy, CuPy, Dask+NumPy, or Dask+CuPy).
    z_limit : float, optional
        Maximum allowed fill depth per cell.  Cells where
        ``filled - dem > z_limit`` revert to their original DEM value.
    name : str, default='fill'
        Name of output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        Filled DEM with depressions removed.

    References
    ----------
    Planchon, O. and Darboux, F. (2001). A fast, simple and versatile
    algorithm to fill the depressions of digital elevation models.
    Catena, 46(2-3), 159-176.
    """
    _validate_raster(dem, func_name='fill', name='dem')

    data = dem.data

    if isinstance(data, np.ndarray):
        dem_f64 = data.astype(np.float64)
        h, w = dem_f64.shape
        ring = np.full((h + 2, w + 2), -1e308, dtype=np.float64)
        out = _fill_tile_kernel(dem_f64, h, w, ring)
        if z_limit is not None:
            out = np.where(out - dem_f64 > z_limit, dem_f64, out)

    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _fill_cupy(data)
        if z_limit is not None:
            import cupy as cp
            dem_f64 = data.astype(cp.float64)
            out = cp.where(out - dem_f64 > z_limit, dem_f64, out)

    elif has_cuda_and_cupy() and is_dask_cupy(dem):
        out = _fill_dask_cupy(data)
        if z_limit is not None:
            import cupy as cp
            dem_f64 = data.astype(cp.float64)
            out = da.where(out - dem_f64 > z_limit, dem_f64, out)

    elif da is not None and isinstance(data, da.Array):
        out = _fill_dask_iterative(data)
        if z_limit is not None:
            dem_f64 = data.astype(np.float64)
            out = da.where(out - dem_f64 > z_limit, dem_f64, out)

    else:
        raise TypeError(f"Unsupported array type: {type(data)}")

    return xr.DataArray(out,
                        name=name,
                        coords=dem.coords,
                        dims=dem.dims,
                        attrs=dem.attrs)
