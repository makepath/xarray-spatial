"""Bilateral filter for feature-preserving smoothing.

Smooths a raster while preserving edges by weighting neighbors based on
both spatial distance and value similarity.

Reference
---------
Tomasi & Manduchi, "Bilateral Filtering for Gray and Color Images," ICCV 1998.
"""
from __future__ import annotations

from functools import partial
from math import ceil, exp, isnan

import numpy as np
import xarray as xr
from numba import cuda
from xarray import DataArray

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _boundary_to_dask,
    _pad_array,
    _validate_boundary,
    _validate_raster,
    _validate_scalar,
    cuda_args,
    ngjit,
)
from xrspatial.dataset_support import supports_dataset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _kernel_radius(sigma_spatial):
    """Derive kernel radius from sigma_spatial: ceil(2 * sigma)."""
    return int(ceil(2.0 * sigma_spatial))


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

@ngjit
def _bilateral_numpy(data, radius, sigma_spatial, sigma_range):
    rows, cols = data.shape
    out = np.empty_like(data)
    inv_2_ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_2_sr = 1.0 / (2.0 * sigma_range * sigma_range)

    for y in range(rows):
        for x in range(cols):
            center = data[y, x]
            if isnan(center):
                out[y, x] = np.nan
                continue

            w_sum = 0.0
            v_sum = 0.0

            y0 = max(y - radius, 0)
            y1 = min(y + radius + 1, rows)
            x0 = max(x - radius, 0)
            x1 = min(x + radius + 1, cols)

            for ny in range(y0, y1):
                for nx in range(x0, x1):
                    val = data[ny, nx]
                    if isnan(val):
                        continue
                    dy = ny - y
                    dx = nx - x
                    dist2 = dy * dy + dx * dx
                    diff = val - center
                    range2 = diff * diff

                    w = exp(-dist2 * inv_2_ss - range2 * inv_2_sr)
                    w_sum += w
                    v_sum += w * val

            if w_sum > 0.0:
                out[y, x] = v_sum / w_sum
            else:
                out[y, x] = np.nan

    return out


def _bilateral_numpy_boundary(data, radius, sigma_spatial, sigma_range,
                              boundary='nan'):
    if boundary == 'nan':
        return _bilateral_numpy(data, radius, sigma_spatial, sigma_range)
    padded = _pad_array(data, radius, boundary)
    result = _bilateral_numpy(padded, radius, sigma_spatial, sigma_range)
    return result[radius:-radius, radius:-radius]


# ---------------------------------------------------------------------------
# Dask + NumPy backend
# ---------------------------------------------------------------------------

def _bilateral_dask_numpy(data, radius, sigma_spatial, sigma_range,
                          boundary='nan'):
    _func = partial(
        _bilateral_numpy, radius=radius,
        sigma_spatial=sigma_spatial, sigma_range=sigma_range,
    )
    out = data.map_overlap(
        _func,
        depth=(radius, radius),
        boundary=_boundary_to_dask(boundary),
        meta=np.array(()),
    )
    return out


# ---------------------------------------------------------------------------
# CuPy (GPU) backend
# ---------------------------------------------------------------------------

@cuda.jit
def _bilateral_gpu(data, radius, inv_2_ss, inv_2_sr, out):
    x, y = cuda.grid(2)
    rows, cols = data.shape

    if 0 <= x < cols and 0 <= y < rows:
        center = data[y, x]
        if isnan(center):
            out[y, x] = center
            return

        w_sum = 0.0
        v_sum = 0.0

        y0 = max(y - radius, 0)
        y1 = min(y + radius + 1, rows)
        x0 = max(x - radius, 0)
        x1 = min(x + radius + 1, cols)

        for ny in range(y0, y1):
            for nx in range(x0, x1):
                val = data[ny, nx]
                if not isnan(val):
                    dy = ny - y
                    dx = nx - x
                    dist2 = dy * dy + dx * dx
                    diff = val - center
                    range2 = diff * diff
                    w = exp(-dist2 * inv_2_ss - range2 * inv_2_sr)
                    w_sum += w
                    v_sum += w * val

        if w_sum > 0.0:
            out[y, x] = v_sum / w_sum
        else:
            out[y, x] = center


def _bilateral_cupy(data, radius, sigma_spatial, sigma_range):
    data_cu = cupy.asarray(data, dtype=cupy.float64)
    inv_2_ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_2_sr = 1.0 / (2.0 * sigma_range * sigma_range)

    griddim, blockdim = cuda_args(data_cu.shape)
    out = cupy.empty_like(data_cu)

    _bilateral_gpu[griddim, blockdim](
        data_cu, radius, inv_2_ss, inv_2_sr, out,
    )
    return out


# ---------------------------------------------------------------------------
# Dask + CuPy backend
# ---------------------------------------------------------------------------

def _bilateral_dask_cupy(data, radius, sigma_spatial, sigma_range,
                         boundary='nan'):
    _func = partial(
        _bilateral_cupy, radius=radius,
        sigma_spatial=sigma_spatial, sigma_range=sigma_range,
    )
    out = data.map_overlap(
        _func,
        depth=(radius, radius),
        boundary=_boundary_to_dask(boundary, is_cupy=True),
        meta=cupy.array(()),
    )
    return out


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def _bilateral(data, radius, sigma_spatial, sigma_range, boundary='nan'):
    agg = xr.DataArray(data)
    mapper = ArrayTypeFunctionMapping(
        numpy_func=partial(
            _bilateral_numpy_boundary,
            boundary=boundary,
        ),
        cupy_func=_bilateral_cupy,
        dask_func=partial(
            _bilateral_dask_numpy,
            boundary=boundary,
        ),
        dask_cupy_func=partial(
            _bilateral_dask_cupy,
            boundary=boundary,
        ),
    )
    out = mapper(agg)(
        agg.data,
        radius=radius,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
    )
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

@supports_dataset
def bilateral(agg, sigma_spatial=1.0, sigma_range=10.0,
              name='bilateral', boundary='nan'):
    """Apply a bilateral filter for feature-preserving smoothing.

    Smooths a raster while preserving edges.  Each pixel is replaced by
    a weighted average of its neighbours, where the weight depends on
    both the spatial distance and the value difference between the
    neighbour and the center pixel.  Neighbours that are far away *or*
    very different in value contribute little, so edges stay sharp.

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D (or 3D multi-band) array of input values.
        Supports NumPy, CuPy, Dask+NumPy, and Dask+CuPy backends.
    sigma_spatial : float, default 1.0
        Standard deviation of the spatial Gaussian.  Controls the size
        of the neighbourhood: kernel radius = ceil(2 * sigma_spatial).
        Must be > 0.
    sigma_range : float, default 10.0
        Standard deviation of the range (value-similarity) Gaussian.
        Larger values allow more smoothing across value differences;
        smaller values preserve more edges.  Must be > 0.
    name : str, default 'bilateral'
        Name for the output DataArray.
    boundary : str, default 'nan'
        How to handle edges where the kernel extends beyond the raster.
        ``'nan'``     -- fill missing neighbours with NaN (default).
        ``'nearest'`` -- repeat edge values.
        ``'reflect'`` -- mirror at boundary.
        ``'wrap'``    -- periodic / toroidal.

    Returns
    -------
    out : xarray.DataArray or xr.Dataset
        Filtered array of the same shape, dtype, dims, and coords as
        the input.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import bilateral
        >>> data = np.array([
        ...     [0., 0., 0., 100., 100.],
        ...     [0., 0., 0., 100., 100.],
        ...     [0., 0., 0., 100., 100.],
        ...     [0., 0., 0., 100., 100.],
        ...     [0., 0., 0., 100., 100.]])
        >>> raster = xr.DataArray(data)
        >>> smoothed = bilateral(raster, sigma_spatial=1.0, sigma_range=5.0)

    References
    ----------
    Tomasi & Manduchi, "Bilateral Filtering for Gray and Color Images,"
    ICCV 1998.
    """
    _validate_raster(agg, func_name='bilateral', name='agg', ndim=(2, 3))
    _validate_scalar(sigma_spatial, func_name='bilateral',
                     name='sigma_spatial', dtype=(int, float),
                     min_val=0, min_exclusive=True)
    _validate_scalar(sigma_range, func_name='bilateral',
                     name='sigma_range', dtype=(int, float),
                     min_val=0, min_exclusive=True)
    _validate_boundary(boundary)

    sigma_spatial = float(sigma_spatial)
    sigma_range = float(sigma_range)
    radius = _kernel_radius(sigma_spatial)

    if agg.ndim == 3:
        from xrspatial.focal import _apply_per_band
        return _apply_per_band(
            bilateral, agg,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range,
            name=name,
            boundary=boundary,
        )

    out = _bilateral(
        agg.data.astype(float),
        radius, sigma_spatial, sigma_range, boundary,
    )

    return DataArray(
        out,
        name=name,
        dims=agg.dims,
        coords=agg.coords,
        attrs=agg.attrs,
    )
