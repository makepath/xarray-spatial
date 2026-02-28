from __future__ import annotations

from functools import partial
from math import isnan, sqrt
from typing import Optional

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import xarray as xr
from numba import cuda

from xrspatial.utils import ArrayTypeFunctionMapping
from xrspatial.utils import _boundary_to_dask
from xrspatial.utils import _pad_array
from xrspatial.utils import _validate_boundary
from xrspatial.utils import _validate_raster
from xrspatial.utils import cuda_args
from xrspatial.utils import ngjit
from xrspatial.dataset_support import supports_dataset


# ---------------------------------------------------------------------------
# CPU kernels
# ---------------------------------------------------------------------------

@ngjit
def _tri_cpu(data):
    out = np.empty(data.shape, np.float64)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            center = data[y, x]
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    diff = data[y + dy, x + dx] - center
                    total += diff * diff
            out[y, x] = sqrt(total)
    return out


@ngjit
def _tpi_cpu(data):
    out = np.empty(data.shape, np.float64)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            center = data[y, x]
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    total += data[y + dy, x + dx]
            out[y, x] = center - total / 8.0
    return out


@ngjit
def _roughness_cpu(data):
    out = np.empty(data.shape, np.float64)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            has_nan = False
            vmin = data[y - 1, x - 1]
            vmax = vmin
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    v = data[y + dy, x + dx]
                    if isnan(v):
                        has_nan = True
                        break
                    if v < vmin:
                        vmin = v
                    if v > vmax:
                        vmax = v
                if has_nan:
                    break
            if not has_nan:
                out[y, x] = vmax - vmin
    return out


# ---------------------------------------------------------------------------
# GPU kernels
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _tri_gpu(arr):
    center = arr[1, 1]
    total = 0.0
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            diff = arr[dy, dx] - center
            total += diff * diff
    return sqrt(total)


@cuda.jit(device=True)
def _tpi_gpu(arr):
    center = arr[1, 1]
    total = 0.0
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            total += arr[dy, dx]
    return center - total / 8.0


@cuda.jit(device=True)
def _roughness_gpu(arr):
    vmin = arr[0, 0]
    vmax = arr[0, 0]
    for dy in range(3):
        for dx in range(3):
            v = arr[dy, dx]
            # NaN check: v != v is True only for NaN
            if v != v:
                return v
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
    return vmax - vmin


@cuda.jit
def _tri_run_gpu(data, out):
    i, j = cuda.grid(2)
    if (i - 1 >= 0 and i + 1 <= out.shape[0] - 1 and
            j - 1 >= 0 and j + 1 <= out.shape[1] - 1):
        out[i, j] = _tri_gpu(data[i - 1:i + 2, j - 1:j + 2])


@cuda.jit
def _tpi_run_gpu(data, out):
    i, j = cuda.grid(2)
    if (i - 1 >= 0 and i + 1 <= out.shape[0] - 1 and
            j - 1 >= 0 and j + 1 <= out.shape[1] - 1):
        out[i, j] = _tpi_gpu(data[i - 1:i + 2, j - 1:j + 2])


@cuda.jit
def _roughness_run_gpu(data, out):
    i, j = cuda.grid(2)
    if (i - 1 >= 0 and i + 1 <= out.shape[0] - 1 and
            j - 1 >= 0 and j + 1 <= out.shape[1] - 1):
        out[i, j] = _roughness_gpu(data[i - 1:i + 2, j - 1:j + 2])


# ---------------------------------------------------------------------------
# Backend wrapper factory
# ---------------------------------------------------------------------------

def _make_backends(cpu_kernel, gpu_global_kernel):
    """Return (run_numpy, run_cupy, run_dask_numpy, run_dask_cupy) tuple."""

    def _run_numpy(data: np.ndarray,
                   boundary: str = 'nan') -> np.ndarray:
        data = data.astype(np.float64)
        if boundary == 'nan':
            return cpu_kernel(data)
        padded = _pad_array(data, 1, boundary)
        result = cpu_kernel(padded)
        return result[1:-1, 1:-1]

    def _run_dask_numpy(data: da.Array,
                        boundary: str = 'nan') -> da.Array:
        data = data.astype(np.float64)
        _func = cpu_kernel
        out = data.map_overlap(_func,
                               depth=(1, 1),
                               boundary=_boundary_to_dask(boundary),
                               meta=np.array(()))
        return out

    def _run_cupy(data: cupy.ndarray,
                  boundary: str = 'nan') -> cupy.ndarray:
        if boundary != 'nan':
            padded = _pad_array(data, 1, boundary)
            result = _run_cupy(padded)
            return result[1:-1, 1:-1]

        data = data.astype(cupy.float64)
        griddim, blockdim = cuda_args(data.shape)
        out = cupy.empty(data.shape, dtype='f8')
        out[:] = cupy.nan
        gpu_global_kernel[griddim, blockdim](data, out)
        return out

    def _run_dask_cupy(data: da.Array,
                       boundary: str = 'nan') -> da.Array:
        data = data.astype(cupy.float64)
        _func = partial(_run_cupy, boundary=boundary)
        out = data.map_overlap(_func,
                               depth=(1, 1),
                               boundary=_boundary_to_dask(boundary, is_cupy=True),
                               meta=cupy.array(()))
        return out

    return _run_numpy, _run_cupy, _run_dask_numpy, _run_dask_cupy


_tri_backends = _make_backends(_tri_cpu, _tri_run_gpu)
_tpi_backends = _make_backends(_tpi_cpu, _tpi_run_gpu)
_roughness_backends = _make_backends(_roughness_cpu, _roughness_run_gpu)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@supports_dataset
def tri(agg: xr.DataArray,
        name: Optional[str] = 'tri',
        boundary: str = 'nan') -> xr.DataArray:
    """Compute the Terrain Ruggedness Index (TRI) for each cell.

    TRI quantifies local elevation variation as the square root of
    the sum of squared differences between the center cell and its
    8 neighbors in a 3x3 window (Riley et al. 1999).

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask
        xarray DataArray of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='tri'
        Name of output DataArray.
    boundary : str, default='nan'
        How to handle edges where the kernel extends beyond the raster.
        ``'nan'``     - fill missing neighbours with NaN (default).
        ``'nearest'`` - repeat edge values.
        ``'reflect'`` - mirror at boundary.
        ``'wrap'``    - periodic / toroidal.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D array of TRI values with the same shape, coords, dims,
        and attrs as the input.

    References
    ----------
    Riley, S.J., DeGloria, S.D. and Elliot, R. (1999). A Terrain
    Ruggedness Index that Quantifies Topographic Heterogeneity.
    Intermountain Journal of Sciences, 5(1-4), 23-27.
    """
    _validate_raster(agg, func_name='tri', name='agg')
    _validate_boundary(boundary)

    run_np, run_cupy, run_dask_np, run_dask_cupy = _tri_backends
    mapper = ArrayTypeFunctionMapping(
        numpy_func=run_np,
        cupy_func=run_cupy,
        dask_func=run_dask_np,
        dask_cupy_func=run_dask_cupy,
    )
    out = mapper(agg)(agg.data, boundary)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)


@supports_dataset
def tpi(agg: xr.DataArray,
        name: Optional[str] = 'tpi',
        boundary: str = 'nan') -> xr.DataArray:
    """Compute the Topographic Position Index (TPI) for each cell.

    TPI is the difference between the elevation of the center cell
    and the mean elevation of its 8 neighbors in a 3x3 window
    (Weiss 2001). Positive values indicate ridgetops, negative
    values indicate valleys, and near-zero values indicate flat
    areas or mid-slope positions.

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask
        xarray DataArray of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='tpi'
        Name of output DataArray.
    boundary : str, default='nan'
        How to handle edges where the kernel extends beyond the raster.
        ``'nan'``     - fill missing neighbours with NaN (default).
        ``'nearest'`` - repeat edge values.
        ``'reflect'`` - mirror at boundary.
        ``'wrap'``    - periodic / toroidal.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D array of TPI values with the same shape, coords, dims,
        and attrs as the input.

    References
    ----------
    Weiss, A. (2001). Topographic Position and Landforms Analysis.
    Poster presentation, ESRI User Conference, San Diego, CA.
    """
    _validate_raster(agg, func_name='tpi', name='agg')
    _validate_boundary(boundary)

    run_np, run_cupy, run_dask_np, run_dask_cupy = _tpi_backends
    mapper = ArrayTypeFunctionMapping(
        numpy_func=run_np,
        cupy_func=run_cupy,
        dask_func=run_dask_np,
        dask_cupy_func=run_dask_cupy,
    )
    out = mapper(agg)(agg.data, boundary)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)


@supports_dataset
def roughness(agg: xr.DataArray,
              name: Optional[str] = 'roughness',
              boundary: str = 'nan') -> xr.DataArray:
    """Compute the roughness for each cell.

    Roughness is the difference between the maximum and minimum
    elevation values in a 3x3 window (including the center cell).
    This is the same definition used by GDAL's ``gdaldem roughness``.

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask
        xarray DataArray of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='roughness'
        Name of output DataArray.
    boundary : str, default='nan'
        How to handle edges where the kernel extends beyond the raster.
        ``'nan'``     - fill missing neighbours with NaN (default).
        ``'nearest'`` - repeat edge values.
        ``'reflect'`` - mirror at boundary.
        ``'wrap'``    - periodic / toroidal.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        2D array of roughness values with the same shape, coords, dims,
        and attrs as the input.

    References
    ----------
    GDAL/OGR contributors (2024). GDAL/OGR Geospatial Data Abstraction
    software Library. Open Source Geospatial Foundation.
    https://gdal.org/programs/gdaldem.html
    """
    _validate_raster(agg, func_name='roughness', name='agg')
    _validate_boundary(boundary)

    run_np, run_cupy, run_dask_np, run_dask_cupy = _roughness_backends
    mapper = ArrayTypeFunctionMapping(
        numpy_func=run_np,
        cupy_func=run_cupy,
        dask_func=run_dask_np,
        dask_cupy_func=run_dask_cupy,
    )
    out = mapper(agg)(agg.data, boundary)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
