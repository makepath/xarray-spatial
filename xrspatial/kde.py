"""Kernel density estimation (KDE) for point-to-raster conversion.

Produces a continuous density surface from point (or line) data.  Each
output pixel accumulates weighted kernel contributions from all nearby
features, yielding a smooth density field.

Supports numpy, cupy, dask+numpy, and dask+cupy backends.
"""
from __future__ import annotations

import math
from math import pi, sqrt
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr

from xrspatial.utils import ngjit, has_cuda_and_cupy, has_dask_array

try:
    import cupy
except ImportError:
    cupy = None

try:
    import dask
    import dask.array as da
except ImportError:
    da = None

from numba import cuda


# ---------------------------------------------------------------------------
# Kernel constants
# ---------------------------------------------------------------------------
_KERNEL_NAMES = ('gaussian', 'epanechnikov', 'quartic')

# Normalisation constants for 2-D product kernels.
# Gaussian  : (1/(2pi))
# Epanechnikov: (2/pi)  (product of (3/4)*(1-u^2) marginals, integrated)
# Quartic   : (3/pi)    (product of (15/16)*(1-u^2)^2 marginals, integrated)
_NORM_GAUSSIAN = 1.0 / (2.0 * pi)
_NORM_EPANECHNIKOV = 2.0 / pi
_NORM_QUARTIC = 3.0 / pi


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------

def _silverman_bandwidth(x, y):
    """Silverman's rule of thumb for 2-D data.

    h = n^(-1/6) * mean(sigma_x, sigma_y)
    where sigma uses the robust scale estimator min(std, IQR/1.34).
    """
    n = len(x)
    if n < 2:
        return 1.0

    def _robust_scale(v):
        s = float(np.std(v))
        q75, q25 = float(np.percentile(v, 75)), float(np.percentile(v, 25))
        iqr = (q75 - q25) / 1.34
        return min(s, iqr) if iqr > 0 else s

    sx = _robust_scale(x)
    sy = _robust_scale(y)
    sigma = (sx + sy) / 2.0
    if sigma == 0:
        sigma = 1.0
    return sigma * (n ** (-1.0 / 6.0))


# ---------------------------------------------------------------------------
# CPU kernels (numba-jitted)
# ---------------------------------------------------------------------------

@ngjit
def _kde_cpu(xs, ys, ws, out, x0, y0, dx, dy, bw, kernel_id):
    """Populate *out* with kernel density values.

    Parameters
    ----------
    xs, ys : 1-D float64 arrays of point coordinates.
    ws : 1-D float64 array of weights (same length as xs).
    out : 2-D float64 output array (rows, cols), pre-zeroed.
    x0, y0 : origin (lower-left corner of first pixel centre).
    dx, dy : pixel spacing in x and y.
    bw : bandwidth (same units as coordinates).
    kernel_id : 0 = gaussian, 1 = epanechnikov, 2 = quartic.
    """
    rows, cols = out.shape
    n_pts = xs.shape[0]
    inv_bw = 1.0 / bw
    inv_bw2 = inv_bw * inv_bw

    # Pre-compute normalisation
    if kernel_id == 0:
        norm = inv_bw2 / (2.0 * pi)
    elif kernel_id == 1:
        norm = inv_bw2 * 2.0 / pi
    else:
        norm = inv_bw2 * 3.0 / pi

    # Cutoff radius in pixels for compact kernels.
    # Gaussian uses 4*bw; compact kernels use exactly bw.
    if kernel_id == 0:
        cutoff = 4.0 * bw
    else:
        cutoff = bw

    for p in range(n_pts):
        px = xs[p]
        py = ys[p]
        w = ws[p]

        # Pixel range that falls within the cutoff.
        # Compute both endpoints and use min/max so that negative
        # spacing (descending coordinates) still produces lo <= hi.
        c_a = int((px - cutoff - x0) / dx)
        c_b = int((px + cutoff - x0) / dx)
        col_lo = max(0, min(c_a, c_b))
        col_hi = min(cols - 1, max(c_a, c_b)) + 1
        r_a = int((py - cutoff - y0) / dy)
        r_b = int((py + cutoff - y0) / dy)
        row_lo = max(0, min(r_a, r_b))
        row_hi = min(rows - 1, max(r_a, r_b)) + 1

        for r in range(row_lo, row_hi):
            cy = y0 + r * dy
            uy = (cy - py) * inv_bw
            uy2 = uy * uy
            for c in range(col_lo, col_hi):
                cx = x0 + c * dx
                ux = (cx - px) * inv_bw
                u2 = ux * ux + uy2

                if kernel_id == 0:
                    # Gaussian
                    val = norm * w * np.exp(-0.5 * u2)
                elif kernel_id == 1:
                    # Epanechnikov
                    if u2 <= 1.0:
                        val = norm * w * (1.0 - u2)
                    else:
                        val = 0.0
                else:
                    # Quartic
                    if u2 <= 1.0:
                        t = 1.0 - u2
                        val = norm * w * t * t
                    else:
                        val = 0.0

                out[r, c] += val


@ngjit
def _line_density_cpu(x1s, y1s, x2s, y2s, ws, out,
                      x0, y0, dx, dy, bw, kernel_id):
    """Compute line density on *out*.

    Each line segment is sampled at sub-segment intervals (step = bw/4)
    and each sample acts as a weighted point where the weight is
    proportional to the step length.
    """
    rows, cols = out.shape
    n_segs = x1s.shape[0]
    inv_bw = 1.0 / bw
    inv_bw2 = inv_bw * inv_bw

    if kernel_id == 0:
        norm = inv_bw2 / (2.0 * pi)
    elif kernel_id == 1:
        norm = inv_bw2 * 2.0 / pi
    else:
        norm = inv_bw2 * 3.0 / pi

    if kernel_id == 0:
        cutoff = 4.0 * bw
    else:
        cutoff = bw

    step = bw / 4.0
    if step < 1e-12:
        return

    for s in range(n_segs):
        ax = x1s[s]
        ay = y1s[s]
        bx = x2s[s]
        by = y2s[s]
        seg_len = sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        if seg_len < 1e-12:
            continue

        w = ws[s]
        n_steps = max(1, int(seg_len / step))
        sub_w = w * (seg_len / n_steps)

        for i in range(n_steps):
            t = (i + 0.5) / n_steps
            px = ax + t * (bx - ax)
            py = ay + t * (by - ay)

            c_a = int((px - cutoff - x0) / dx)
            c_b = int((px + cutoff - x0) / dx)
            col_lo = max(0, min(c_a, c_b))
            col_hi = min(cols - 1, max(c_a, c_b)) + 1
            r_a = int((py - cutoff - y0) / dy)
            r_b = int((py + cutoff - y0) / dy)
            row_lo = max(0, min(r_a, r_b))
            row_hi = min(rows - 1, max(r_a, r_b)) + 1

            for r in range(row_lo, row_hi):
                cy = y0 + r * dy
                uy = (cy - py) * inv_bw
                uy2 = uy * uy
                for c in range(col_lo, col_hi):
                    cx = x0 + c * dx
                    ux = (cx - px) * inv_bw
                    u2 = ux * ux + uy2

                    if kernel_id == 0:
                        val = norm * sub_w * np.exp(-0.5 * u2)
                    elif kernel_id == 1:
                        if u2 <= 1.0:
                            val = norm * sub_w * (1.0 - u2)
                        else:
                            val = 0.0
                    else:
                        if u2 <= 1.0:
                            tt = 1.0 - u2
                            val = norm * sub_w * tt * tt
                        else:
                            val = 0.0

                    out[r, c] += val


# ---------------------------------------------------------------------------
# GPU kernels (CUDA)
# ---------------------------------------------------------------------------

@cuda.jit
def _kde_cuda(xs, ys, ws, out, x0, y0, dx, dy, bw, kernel_id, n_pts):
    """Each thread computes one output pixel."""
    r, c = cuda.grid(2)
    rows = out.shape[0]
    cols = out.shape[1]
    if r >= rows or c >= cols:
        return

    cx = x0[0] + c * dx[0]
    cy = y0[0] + r * dy[0]
    inv_bw = 1.0 / bw[0]
    inv_bw2 = inv_bw * inv_bw
    kid = kernel_id[0]

    if kid == 0:
        norm = inv_bw2 / (2.0 * 3.141592653589793)
    elif kid == 1:
        norm = inv_bw2 * 2.0 / 3.141592653589793
    else:
        norm = inv_bw2 * 3.0 / 3.141592653589793

    total = 0.0
    for p in range(n_pts[0]):
        ux = (cx - xs[p]) * inv_bw
        uy = (cy - ys[p]) * inv_bw
        u2 = ux * ux + uy * uy

        if kid == 0:
            # No hard cutoff; exp decays fast enough and each thread
            # loops independently so the extra iterations are cheap.
            total += norm * ws[p] * math.exp(-0.5 * u2)
        elif kid == 1:
            if u2 <= 1.0:
                total += norm * ws[p] * (1.0 - u2)
        else:
            if u2 <= 1.0:
                t = 1.0 - u2
                total += norm * ws[p] * t * t

    out[r, c] = total


# ---------------------------------------------------------------------------
# Backend wrappers
# ---------------------------------------------------------------------------

def _kernel_id(kernel: str) -> int:
    if kernel == 'gaussian':
        return 0
    elif kernel == 'epanechnikov':
        return 1
    elif kernel == 'quartic':
        return 2
    raise ValueError(
        f"kernel must be one of {_KERNEL_NAMES}, got {kernel!r}"
    )


def _run_kde_numpy(xs, ys, ws, shape, x0, y0, dx, dy, bw, kernel_id):
    out = np.zeros(shape, dtype=np.float64)
    _kde_cpu(xs, ys, ws, out, x0, y0, dx, dy, bw, kernel_id)
    return out


def _run_kde_cupy(xs, ys, ws, shape, x0, y0, dx, dy, bw, kernel_id):
    out = cupy.zeros(shape, dtype=cupy.float64)
    n_pts = cupy.array([len(xs)], dtype=cupy.int64)
    x0_d = cupy.array([x0], dtype=cupy.float64)
    y0_d = cupy.array([y0], dtype=cupy.float64)
    dx_d = cupy.array([dx], dtype=cupy.float64)
    dy_d = cupy.array([dy], dtype=cupy.float64)
    bw_d = cupy.array([bw], dtype=cupy.float64)
    kid_d = cupy.array([kernel_id], dtype=cupy.int64)

    xs_d = cupy.asarray(xs, dtype=cupy.float64)
    ys_d = cupy.asarray(ys, dtype=cupy.float64)
    ws_d = cupy.asarray(ws, dtype=cupy.float64)

    tpb = (16, 16)
    bpg = (
        (shape[0] + tpb[0] - 1) // tpb[0],
        (shape[1] + tpb[1] - 1) // tpb[1],
    )
    _kde_cuda[bpg, tpb](xs_d, ys_d, ws_d, out,
                         x0_d, y0_d, dx_d, dy_d, bw_d, kid_d, n_pts)
    return out


def _filter_points_to_tile(xs, ys, ws, tile_x0, tile_y0, dx, dy,
                           tile_rows, tile_cols, cutoff):
    """Return (xs, ys, ws) subset that could affect this tile.

    Points whose cutoff circle doesn't overlap the tile extent are
    excluded, reducing serialization and speeding up the kernel.
    """
    tile_x1 = tile_x0 + tile_cols * dx
    tile_y1 = tile_y0 + tile_rows * dy
    mask = ((xs >= tile_x0 - cutoff) & (xs <= tile_x1 + cutoff) &
            (ys >= tile_y0 - cutoff) & (ys <= tile_y1 + cutoff))
    if mask.all():
        return xs, ys, ws
    return xs[mask], ys[mask], ws[mask]


def _run_kde_dask_numpy(xs, ys, ws, shape, x0, y0, dx, dy, bw, kernel_id,
                        chunks):
    """Dask-backed KDE: each chunk computes its own tile independently.

    Points are pre-filtered per tile so each delayed task receives only
    the relevant subset, reducing serialization from O(n_tiles * n_points)
    to O(n_tiles * points_per_tile).
    """
    # Determine chunk layout
    if chunks is None:
        chunks = (min(256, shape[0]), min(256, shape[1]))
    row_splits = _split_sizes(shape[0], chunks[0])
    col_splits = _split_sizes(shape[1], chunks[1])

    # Cutoff radius matching the kernel implementation
    cutoff = 4.0 * bw if kernel_id == 0 else bw

    blocks = []
    row_off = 0
    for rs in row_splits:
        row_blocks = []
        col_off = 0
        for cs in col_splits:
            tile_y0 = y0 + row_off * dy
            tile_x0 = x0 + col_off * dx
            tile_shape = (rs, cs)
            # Pre-filter points to this tile's extent + cutoff
            txs, tys, tws = _filter_points_to_tile(
                xs, ys, ws, tile_x0, tile_y0, dx, dy, rs, cs, cutoff)
            block = dask.delayed(_run_kde_numpy)(
                txs, tys, tws, tile_shape,
                tile_x0, tile_y0, dx, dy, bw, kernel_id,
            )
            row_blocks.append(
                da.from_delayed(block, shape=tile_shape, dtype=np.float64)
            )
            col_off += cs
        blocks.append(row_blocks)
        row_off += rs

    return da.block(blocks)


def _run_kde_dask_cupy(xs, ys, ws, shape, x0, y0, dx, dy, bw, kernel_id,
                       chunks):
    """Dask+CuPy KDE: each chunk uses the GPU kernel.

    Points are pre-filtered per tile (same as the numpy dask path)
    so each delayed task serializes only the relevant subset.
    """
    if chunks is None:
        chunks = (min(256, shape[0]), min(256, shape[1]))
    row_splits = _split_sizes(shape[0], chunks[0])
    col_splits = _split_sizes(shape[1], chunks[1])

    cutoff = 4.0 * bw if kernel_id == 0 else bw

    blocks = []
    row_off = 0
    for rs in row_splits:
        row_blocks = []
        col_off = 0
        for cs in col_splits:
            tile_y0 = y0 + row_off * dy
            tile_x0 = x0 + col_off * dx
            tile_shape = (rs, cs)
            txs, tys, tws = _filter_points_to_tile(
                xs, ys, ws, tile_x0, tile_y0, dx, dy, rs, cs, cutoff)
            block = dask.delayed(_run_kde_cupy)(
                txs, tys, tws, tile_shape,
                tile_x0, tile_y0, dx, dy, bw, kernel_id,
            )
            row_blocks.append(
                da.from_delayed(block, shape=tile_shape,
                                dtype=np.float64,
                                meta=cupy.array((), dtype=cupy.float64))
            )
            col_off += cs
        blocks.append(row_blocks)
        row_off += rs

    return da.block(blocks)


def _split_sizes(total, chunk):
    """Return a list of chunk sizes that sum to *total*."""
    full, rem = divmod(total, chunk)
    sizes = [chunk] * full
    if rem:
        sizes.append(rem)
    return sizes


# ---------------------------------------------------------------------------
# Public API -- kde
# ---------------------------------------------------------------------------

def kde(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    *,
    weights: Optional[Union[np.ndarray, list]] = None,
    bandwidth: Union[float, str] = 'silverman',
    kernel: str = 'gaussian',
    template: Optional[xr.DataArray] = None,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    width: int = 256,
    height: int = 256,
    name: str = 'kde',
) -> xr.DataArray:
    """Compute 2-D kernel density estimation from point data.

    Each output pixel accumulates weighted kernel contributions from all
    input points, producing a smooth continuous density surface.

    Parameters
    ----------
    x, y : array-like
        1-D arrays of point coordinates.
    weights : array-like, optional
        Per-point weights.  Defaults to uniform weights of 1.
    bandwidth : float or ``'silverman'``
        Kernel bandwidth in the same units as *x*/*y*.
        ``'silverman'`` (default) uses Silverman's rule of thumb.
    kernel : ``{'gaussian', 'epanechnikov', 'quartic'}``
        Kernel shape.
    template : xr.DataArray, optional
        If provided, the output matches this array's shape, extent, and
        coordinates.  *x_range*, *y_range*, *width*, and *height* are
        ignored when *template* is given.
    x_range, y_range : (min, max), optional
        Spatial extent of the output grid.  Defaults to the data extent
        with 10 %% padding on each side.
    width, height : int
        Number of columns and rows in the output grid.  Ignored when
        *template* is provided.
    name : str
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        2-D density surface.
    """
    # -- Validate and coerce inputs ----------------------------------------
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length")
    n = x_arr.shape[0]

    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64).ravel()
        if w_arr.shape[0] != n:
            raise ValueError("weights must have the same length as x and y")
    else:
        w_arr = np.ones(n, dtype=np.float64)

    kid = _kernel_id(kernel)

    # -- Bandwidth ---------------------------------------------------------
    if isinstance(bandwidth, str):
        if bandwidth != 'silverman':
            raise ValueError(
                "bandwidth must be a positive number or 'silverman', "
                f"got {bandwidth!r}"
            )
        bw = _silverman_bandwidth(x_arr, y_arr)
    else:
        bw = float(bandwidth)
        if bw <= 0:
            raise ValueError(f"bandwidth must be positive, got {bw}")

    # -- Output grid -------------------------------------------------------
    if template is not None:
        _validate_template(template)
        y_coords = template.coords[template.dims[0]].values
        x_coords = template.coords[template.dims[1]].values
        rows, cols = template.shape
        # Pixel spacing
        dy = float(y_coords[1] - y_coords[0]) if rows > 1 else 1.0
        dx = float(x_coords[1] - x_coords[0]) if cols > 1 else 1.0
        x0 = float(x_coords[0])
        y0 = float(y_coords[0])
        use_dask = has_dask_array() and isinstance(template.data, da.Array)
        use_cupy = (has_cuda_and_cupy() and cupy is not None
                    and _is_cupy_backed(template))
        out_chunks = template.data.chunksize if use_dask else None
    else:
        if x_range is None:
            pad = max(bw, (float(x_arr.max()) - float(x_arr.min())) * 0.1)
            x_range = (float(x_arr.min()) - pad, float(x_arr.max()) + pad)
        if y_range is None:
            pad = max(bw, (float(y_arr.max()) - float(y_arr.min())) * 0.1)
            y_range = (float(y_arr.min()) - pad, float(y_arr.max()) + pad)
        rows, cols = height, width
        dx = (x_range[1] - x_range[0]) / max(cols - 1, 1)
        dy = (y_range[1] - y_range[0]) / max(rows - 1, 1)
        x0 = x_range[0]
        y0 = y_range[0]
        x_coords = np.linspace(x_range[0], x_range[1], cols)
        y_coords = np.linspace(y_range[0], y_range[1], rows)
        use_dask = False
        use_cupy = False
        out_chunks = None

    shape = (rows, cols)

    # -- Dispatch -----------------------------------------------------------
    if use_dask and use_cupy:
        data = _run_kde_dask_cupy(
            x_arr, y_arr, w_arr, shape, x0, y0, dx, dy, bw, kid, out_chunks,
        )
    elif use_dask:
        data = _run_kde_dask_numpy(
            x_arr, y_arr, w_arr, shape, x0, y0, dx, dy, bw, kid, out_chunks,
        )
    elif use_cupy:
        data = _run_kde_cupy(
            x_arr, y_arr, w_arr, shape, x0, y0, dx, dy, bw, kid,
        )
    else:
        data = _run_kde_numpy(
            x_arr, y_arr, w_arr, shape, x0, y0, dx, dy, bw, kid,
        )

    # -- Build output DataArray --------------------------------------------
    if template is not None:
        return xr.DataArray(
            data, name=name,
            coords=template.coords, dims=template.dims,
            attrs=template.attrs,
        )
    return xr.DataArray(
        data, name=name,
        dims=['y', 'x'],
        coords={'y': y_coords, 'x': x_coords},
    )


# ---------------------------------------------------------------------------
# Public API -- line_density
# ---------------------------------------------------------------------------

def line_density(
    x1: Union[np.ndarray, list],
    y1: Union[np.ndarray, list],
    x2: Union[np.ndarray, list],
    y2: Union[np.ndarray, list],
    *,
    weights: Optional[Union[np.ndarray, list]] = None,
    bandwidth: Union[float, str] = 'silverman',
    kernel: str = 'gaussian',
    template: Optional[xr.DataArray] = None,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    width: int = 256,
    height: int = 256,
    name: str = 'line_density',
) -> xr.DataArray:
    """Compute line density from line-segment data.

    Each segment is uniformly sampled and the samples are convolved with
    the chosen kernel, producing a smooth density surface that represents
    the concentration of linear features.

    Parameters
    ----------
    x1, y1, x2, y2 : array-like
        Start and end coordinates of each line segment.
    weights : array-like, optional
        Per-segment weights.  Defaults to uniform weights of 1.
    bandwidth : float or ``'silverman'``
        Kernel bandwidth.  ``'silverman'`` uses an automatic estimate
        based on all segment endpoints.
    kernel : ``{'gaussian', 'epanechnikov', 'quartic'}``
        Kernel shape.
    template : xr.DataArray, optional
        Output grid specification (same as :func:`kde`).
    x_range, y_range : (min, max), optional
        Spatial extent.
    width, height : int
        Grid dimensions.
    name : str
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        2-D line-density surface.
    """
    x1a = np.asarray(x1, dtype=np.float64).ravel()
    y1a = np.asarray(y1, dtype=np.float64).ravel()
    x2a = np.asarray(x2, dtype=np.float64).ravel()
    y2a = np.asarray(y2, dtype=np.float64).ravel()
    n = x1a.shape[0]
    if not (y1a.shape[0] == n and x2a.shape[0] == n and y2a.shape[0] == n):
        raise ValueError("x1, y1, x2, y2 must all have the same length")

    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64).ravel()
        if w_arr.shape[0] != n:
            raise ValueError(
                "weights must have the same length as the segment arrays"
            )
    else:
        w_arr = np.ones(n, dtype=np.float64)

    kid = _kernel_id(kernel)

    # Bandwidth from all endpoints
    all_x = np.concatenate([x1a, x2a])
    all_y = np.concatenate([y1a, y2a])

    if isinstance(bandwidth, str):
        if bandwidth != 'silverman':
            raise ValueError(
                "bandwidth must be a positive number or 'silverman', "
                f"got {bandwidth!r}"
            )
        bw = _silverman_bandwidth(all_x, all_y)
    else:
        bw = float(bandwidth)
        if bw <= 0:
            raise ValueError(f"bandwidth must be positive, got {bw}")

    # Grid
    if template is not None:
        _validate_template(template)
        y_coords = template.coords[template.dims[0]].values
        x_coords = template.coords[template.dims[1]].values
        rows, cols = template.shape
        dy = float(y_coords[1] - y_coords[0]) if rows > 1 else 1.0
        dx = float(x_coords[1] - x_coords[0]) if cols > 1 else 1.0
        x0 = float(x_coords[0])
        y0 = float(y_coords[0])
    else:
        if x_range is None:
            pad = max(bw, (float(all_x.max()) - float(all_x.min())) * 0.1)
            x_range = (float(all_x.min()) - pad, float(all_x.max()) + pad)
        if y_range is None:
            pad = max(bw, (float(all_y.max()) - float(all_y.min())) * 0.1)
            y_range = (float(all_y.min()) - pad, float(all_y.max()) + pad)
        rows, cols = height, width
        dx = (x_range[1] - x_range[0]) / max(cols - 1, 1)
        dy = (y_range[1] - y_range[0]) / max(rows - 1, 1)
        x0 = x_range[0]
        y0 = y_range[0]
        x_coords = np.linspace(x_range[0], x_range[1], cols)
        y_coords = np.linspace(y_range[0], y_range[1], rows)

    shape = (rows, cols)
    out = np.zeros(shape, dtype=np.float64)
    _line_density_cpu(x1a, y1a, x2a, y2a, w_arr, out,
                      x0, y0, dx, dy, bw, kid)

    if template is not None:
        return xr.DataArray(
            out, name=name,
            coords=template.coords, dims=template.dims,
            attrs=template.attrs,
        )
    return xr.DataArray(
        out, name=name,
        dims=['y', 'x'],
        coords={'y': y_coords, 'x': x_coords},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_template(template):
    if not isinstance(template, xr.DataArray):
        raise TypeError(
            "template must be an xr.DataArray, "
            f"got {type(template).__qualname__}"
        )
    if template.ndim != 2:
        raise ValueError(
            f"template must be 2-D, got {template.ndim}-D"
        )


def _is_cupy_backed(agg):
    """Check if a DataArray is backed by cupy (plain or via dask)."""
    try:
        meta = agg.data._meta
        return type(meta).__module__.split('.')[0] == 'cupy'
    except AttributeError:
        if cupy is not None:
            return isinstance(agg.data, cupy.ndarray)
    return False
