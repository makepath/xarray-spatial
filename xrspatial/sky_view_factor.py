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
    ngjit,
)


# ---------------------------------------------------------------------------
# CPU kernel
# ---------------------------------------------------------------------------

@ngjit
def _svf_cpu(data, max_radius, n_directions):
    """Compute SVF over an entire 2-D array on the CPU."""
    rows, cols = data.shape
    out = np.empty((rows, cols), dtype=np.float64)
    out[:] = np.nan

    for y in range(max_radius, rows - max_radius):
        for x in range(max_radius, cols - max_radius):
            center = data[y, x]
            if center != center:  # NaN check
                continue

            svf_sum = 0.0
            valid_dirs = 0
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
                    dist = float(r)
                    elev_angle = _atan2(dz, dist)
                    if elev_angle > max_elev_angle:
                        max_elev_angle = elev_angle

                svf_sum += _sin(max_elev_angle)
                valid_dirs += 1

            if valid_dirs > 0:
                out[y, x] = 1.0 - svf_sum / valid_dirs
    return out


# ---------------------------------------------------------------------------
# GPU kernels
# ---------------------------------------------------------------------------

@cuda.jit
def _svf_gpu(data, out, max_radius, n_directions):
    """CUDA global kernel: one thread per cell."""
    y, x = cuda.grid(2)
    rows, cols = data.shape

    if y < max_radius or y >= rows - max_radius:
        return
    if x < max_radius or x >= cols - max_radius:
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
            dist = float(r)
            elev_angle = _atan2(dz, dist)
            if elev_angle > max_elev_angle:
                max_elev_angle = elev_angle

        svf_sum += _sin(max_elev_angle)

    out[y, x] = 1.0 - svf_sum / n_directions


# ---------------------------------------------------------------------------
# Backend wrappers
# ---------------------------------------------------------------------------

def _run_numpy(data, max_radius, n_directions):
    data = data.astype(np.float64)
    return _svf_cpu(data, max_radius, n_directions)


def _run_cupy(data, max_radius, n_directions):
    data = data.astype(cupy.float64)
    out = cupy.full(data.shape, cupy.nan, dtype=cupy.float64)
    griddim, blockdim = cuda_args(data.shape)
    _svf_gpu[griddim, blockdim](data, out, max_radius, n_directions)
    return out


def _run_dask_numpy(data, max_radius, n_directions):
    data = data.astype(np.float64)
    _func = partial(_svf_cpu, max_radius=max_radius, n_directions=n_directions)
    out = data.map_overlap(
        _func,
        depth=(max_radius, max_radius),
        boundary=np.nan,
        meta=np.array(()),
    )
    return out


def _run_dask_cupy(data, max_radius, n_directions):
    data = data.astype(cupy.float64)
    _func = partial(_run_cupy, max_radius=max_radius, n_directions=n_directions)
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
    name: Optional[str] = 'sky_view_factor',
) -> xr.DataArray:
    """Compute the sky-view factor for each cell of a DEM.

    SVF measures the fraction of the sky hemisphere visible from each
    cell on a scale from 0 (fully obstructed) to 1 (flat open terrain).
    Rays are cast at *n_directions* evenly spaced azimuths out to
    *max_radius* cells, and the maximum elevation angle along each
    ray determines the horizon obstruction.

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask
        xarray DataArray of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    max_radius : int, default=10
        Maximum search distance in cells along each ray direction.
        Cells within *max_radius* of the raster edge will be NaN.
    n_directions : int, default=16
        Number of azimuth directions to sample, evenly spaced
        around 360 degrees.
    name : str, default='sky_view_factor'
        Name of the output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D array of SVF values in [0, 1] with the same shape, coords,
        dims, and attrs as the input.  Edge cells within *max_radius*
        of the boundary are NaN.

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

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        cupy_func=_run_cupy,
        dask_func=_run_dask_numpy,
        dask_cupy_func=_run_dask_cupy,
    )
    out = mapper(agg)(agg.data, max_radius, n_directions)
    return xr.DataArray(
        out,
        name=name,
        coords=agg.coords,
        dims=agg.dims,
        attrs=agg.attrs,
    )
