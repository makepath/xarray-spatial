"""Sky-view factor (SVF) for digital elevation models.

SVF measures the fraction of the sky hemisphere visible from each cell,
ranging from 0 (fully obstructed) to 1 (flat open terrain).  For each
cell, rays are cast at *n_directions* evenly spaced azimuths out to
*max_radius* cells.  Along each ray the maximum elevation angle to the
horizon is recorded and SVF is computed as:

    SVF(i,j) = 1 - mean(sin(max_horizon_angle(d)) for d in directions)

References
----------
Zakek, K., Ostir, K., Kokalj, Z. (2011).  Sky-View Factor as a Relief
Visualization Technique.  Remote Sensing, 3(2), 398-415.

Yokoyama, R., Sidle, R.C., Noguchi, S. (2002).  Application of
Topographic Shading to Terrain Visualization.  International Journal
of Geographical Information Science, 16(5), 489-502.
"""
from __future__ import annotations

from functools import partial
from math import atan2 as _atan2, cos as _cos, pi as _pi, sin as _sin, sqrt as _sqrt
from typing import Optional

import numpy as np
import xarray as xr
from numba import cuda

try:
    import cupy
except ImportError:
    class cupy:
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.dataset_support import supports_dataset
from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _boundary_to_dask,
    _validate_raster,
    _validate_scalar,
    cuda_args,
    get_dataarray_resolution,
    ngjit,
)


# ---------------------------------------------------------------------------
# Memory guards
# ---------------------------------------------------------------------------
# Peak working set per pixel:
#   float64 input cast    8 bytes
#   float64 output array  8 bytes
# Total ~16 bytes/pixel.  Same budget pattern as resample / kde / sieve.
_BYTES_PER_PIXEL = 16


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
            f"sky_view_factor() on a {rows}x{cols} raster needs "
            f"~{required / 1e9:.1f} GB of working memory but only "
            f"~{available / 1e9:.1f} GB is available.  Use a dask-backed "
            f"DataArray for out-of-core processing or downsample the input."
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
    required = int(rows) * int(cols) * _BYTES_PER_PIXEL
    if required > 0.5 * available:
        raise MemoryError(
            f"sky_view_factor() on a {rows}x{cols} cupy raster needs "
            f"~{required / 1e9:.1f} GB of GPU memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing or "
            f"downsample the input."
        )


# ---------------------------------------------------------------------------
# CPU kernel
# ---------------------------------------------------------------------------

@ngjit
def _svf_cpu(data, max_radius, n_directions, cellsize_x, cellsize_y):
    """Compute SVF over an entire 2-D array on the CPU.

    *cellsize_x* and *cellsize_y* are the ground distance per cell along
    the x and y axes, in the same units as the elevation values.  They
    are used to convert the integer ray step *r* into a true ground
    distance for the horizon angle calculation.
    """
    rows, cols = data.shape
    out = np.empty((rows, cols), dtype=np.float64)
    out[:] = np.nan

    for y in range(rows):
        for x in range(cols):
            center = data[y, x]
            if center != center:  # NaN check
                continue

            svf_sum = 0.0
            for d in range(n_directions):
                angle = 2.0 * _pi * d / n_directions
                dx = _cos(angle)
                dy = _sin(angle)

                max_elev_angle = 0.0
                for r in range(1, max_radius + 1):
                    sx = x + int(round(r * dx))
                    sy = y + int(round(r * dy))
                    if sx < 0 or sx >= cols or sy < 0 or sy >= rows:
                        break
                    elev = data[sy, sx]
                    if elev != elev:  # NaN
                        break
                    dz = elev - center
                    gx = (sx - x) * cellsize_x
                    gy = (sy - y) * cellsize_y
                    dist = _sqrt(gx * gx + gy * gy)
                    elev_angle = _atan2(dz, dist)
                    if elev_angle > max_elev_angle:
                        max_elev_angle = elev_angle

                svf_sum += _sin(max_elev_angle)

            out[y, x] = 1.0 - svf_sum / n_directions
    return out


# ---------------------------------------------------------------------------
# GPU kernels
# ---------------------------------------------------------------------------

@cuda.jit
def _svf_gpu(data, out, max_radius, n_directions, cellsize_x, cellsize_y):
    """CUDA global kernel: one thread per cell."""
    y, x = cuda.grid(2)
    rows, cols = data.shape

    if y >= rows or x >= cols:
        return

    center = data[y, x]
    if center != center:  # NaN
        return

    svf_sum = 0.0
    pi2 = 2.0 * 3.141592653589793

    for d in range(n_directions):
        angle = pi2 * d / n_directions
        dx = _cos(angle)
        dy = _sin(angle)

        max_elev_angle = 0.0
        for r in range(1, max_radius + 1):
            sx = x + int(round(r * dx))
            sy = y + int(round(r * dy))
            if sx < 0 or sx >= cols or sy < 0 or sy >= rows:
                break
            elev = data[sy, sx]
            if elev != elev:  # NaN
                break
            dz = elev - center
            gx = (sx - x) * cellsize_x
            gy = (sy - y) * cellsize_y
            dist = _sqrt(gx * gx + gy * gy)
            elev_angle = _atan2(dz, dist)
            if elev_angle > max_elev_angle:
                max_elev_angle = elev_angle

        svf_sum += _sin(max_elev_angle)

    out[y, x] = 1.0 - svf_sum / n_directions


# ---------------------------------------------------------------------------
# Backend wrappers
# ---------------------------------------------------------------------------

def _run_numpy(data, max_radius, n_directions, cellsize_x, cellsize_y):
    _check_memory(data.shape[0], data.shape[1])
    data = data.astype(np.float64)
    return _svf_cpu(data, max_radius, n_directions, cellsize_x, cellsize_y)


def _run_cupy(data, max_radius, n_directions, cellsize_x, cellsize_y):
    _check_gpu_memory(data.shape[0], data.shape[1])
    data = data.astype(cupy.float64)
    out = cupy.full(data.shape, cupy.nan, dtype=cupy.float64)
    griddim, blockdim = cuda_args(data.shape)
    _svf_gpu[griddim, blockdim](
        data, out, max_radius, n_directions, cellsize_x, cellsize_y
    )
    return out


def _run_dask_numpy(data, max_radius, n_directions, cellsize_x, cellsize_y):
    data = data.astype(np.float64)
    _func = partial(
        _svf_cpu,
        max_radius=max_radius,
        n_directions=n_directions,
        cellsize_x=cellsize_x,
        cellsize_y=cellsize_y,
    )
    out = data.map_overlap(
        _func,
        depth=(max_radius, max_radius),
        boundary=np.nan,
        meta=np.array(()),
    )
    return out


def _run_dask_cupy(data, max_radius, n_directions, cellsize_x, cellsize_y):
    data = data.astype(cupy.float64)
    _func = partial(
        _run_cupy,
        max_radius=max_radius,
        n_directions=n_directions,
        cellsize_x=cellsize_x,
        cellsize_y=cellsize_y,
    )
    out = data.map_overlap(
        _func,
        depth=(max_radius, max_radius),
        boundary=cupy.nan,
        meta=cupy.array(()),
    )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@supports_dataset
def sky_view_factor(
    agg: xr.DataArray,
    max_radius: int = 10,
    n_directions: int = 16,
    cellsize_x: Optional[float] = None,
    cellsize_y: Optional[float] = None,
    name: Optional[str] = 'sky_view_factor',
) -> xr.DataArray:
    """Compute the sky-view factor for each cell of a DEM.

    SVF measures the fraction of the sky hemisphere visible from each
    cell on a scale from 0 (fully obstructed) to 1 (flat open terrain).
    Rays are cast at *n_directions* evenly spaced azimuths out to
    *max_radius* cells, and the maximum elevation angle along each
    ray determines the horizon obstruction.

    The horizon angle along each ray uses true ground distance
    (``cell_step * cellsize``), so cell size must be in the same unit
    as the elevation values (e.g. meters for both).

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask
        xarray DataArray of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    max_radius : int, default=10
        Maximum search distance in cells along each ray direction.
    n_directions : int, default=16
        Number of azimuth directions to sample, evenly spaced
        around 360 degrees.
    cellsize_x : float, optional
        Ground distance per cell along the x axis, in the same unit
        as the elevation values.  If not provided, it is read from
        ``agg.attrs['res']`` or computed from the x coordinate.
    cellsize_y : float, optional
        Ground distance per cell along the y axis, in the same unit
        as the elevation values.  If not provided, it is read from
        ``agg.attrs['res']`` or computed from the y coordinate.
    name : str, default='sky_view_factor'
        Name of the output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D array of SVF values in [0, 1] with the same shape, coords,
        dims, and attrs as the input.  Edge cells use truncated rays
        that stop at the raster boundary.

    References
    ----------
    Zakek, K., Ostir, K., Kokalj, Z. (2011).  Sky-View Factor as a
    Relief Visualization Technique.  Remote Sensing, 3(2), 398-415.

    Examples
    --------
    >>> from xrspatial import sky_view_factor
    >>> svf = sky_view_factor(dem, max_radius=100, n_directions=16)
    """
    _validate_raster(agg, func_name='sky_view_factor', name='agg')
    _validate_scalar(max_radius, func_name='sky_view_factor',
                     name='max_radius', dtype=int, min_val=1)
    _validate_scalar(n_directions, func_name='sky_view_factor',
                     name='n_directions', dtype=int, min_val=1)

    if cellsize_x is None or cellsize_y is None:
        try:
            res_x, res_y = get_dataarray_resolution(agg)
        except Exception:
            res_x, res_y = 1.0, 1.0
        if cellsize_x is None:
            cellsize_x = float(abs(res_x)) if res_x else 1.0
        if cellsize_y is None:
            cellsize_y = float(abs(res_y)) if res_y else 1.0
    else:
        cellsize_x = float(abs(cellsize_x))
        cellsize_y = float(abs(cellsize_y))

    if cellsize_x <= 0:
        cellsize_x = 1.0
    if cellsize_y <= 0:
        cellsize_y = 1.0

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        cupy_func=_run_cupy,
        dask_func=_run_dask_numpy,
        dask_cupy_func=_run_dask_cupy,
    )
    out = mapper(agg)(
        agg.data, max_radius, n_directions, cellsize_x, cellsize_y
    )
    return xr.DataArray(
        out,
        name=name,
        coords=agg.coords,
        dims=agg.dims,
        attrs=agg.attrs,
    )
