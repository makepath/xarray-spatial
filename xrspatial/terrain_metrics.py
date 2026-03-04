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
from xrspatial.slope import slope as _slope_func


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
    Poster presentation, International User Conference, San Diego, CA.
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


# ---------------------------------------------------------------------------
# TPI at arbitrary radius (helpers for landforms)
# ---------------------------------------------------------------------------

def _circular_kernel(radius):
    """Circular boolean kernel with center excluded."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    k = (x * x + y * y <= radius * radius).astype(np.float64)
    k[radius, radius] = 0.0
    return k


def _tpi_radius_np(data, kernel):
    """TPI via NaN-aware circular convolution (numpy)."""
    from scipy.ndimage import convolve
    data = data.astype(np.float64)
    valid = np.isfinite(data)
    filled = np.where(valid, data, 0.0)
    s = convolve(filled, kernel, mode='constant', cval=0.0)
    c = convolve(valid.astype(np.float64), kernel, mode='constant', cval=0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(c > 0, s / c, np.nan)
    return data - mean


def _tpi_radius_cupy(data, kernel_np):
    """TPI via NaN-aware circular convolution (cupy)."""
    from cupyx.scipy.ndimage import convolve
    kernel = cupy.asarray(kernel_np)
    data = data.astype(cupy.float64)
    valid = cupy.isfinite(data)
    filled = cupy.where(valid, data, 0.0)
    s = convolve(filled, kernel, mode='constant', cval=0.0)
    c = convolve(valid.astype(cupy.float64), kernel, mode='constant', cval=0.0)
    mean = cupy.where(c > 0, s / c, cupy.nan)
    return data - mean


def _tpi_radius_dask_np(data, radius, kernel):
    """TPI via map_overlap (dask + numpy)."""
    _func = partial(_tpi_radius_np, kernel=kernel)
    return data.map_overlap(
        _func, depth=(radius, radius),
        boundary=np.nan, meta=np.array(()))


def _tpi_radius_dask_cupy(data, radius, kernel_np):
    """TPI via map_overlap (dask + cupy)."""
    _func = partial(_tpi_radius_cupy, kernel_np=kernel_np)
    return data.map_overlap(
        _func, depth=(radius, radius),
        boundary=cupy.nan, meta=cupy.array(()))


def _compute_tpi_at_radius(agg, radius):
    """Dispatch TPI-at-radius to the appropriate backend."""
    kernel = _circular_kernel(radius)
    mapper = ArrayTypeFunctionMapping(
        numpy_func=partial(_tpi_radius_np, kernel=kernel),
        cupy_func=partial(_tpi_radius_cupy, kernel_np=kernel),
        dask_func=partial(_tpi_radius_dask_np,
                          radius=radius, kernel=kernel),
        dask_cupy_func=partial(_tpi_radius_dask_cupy,
                               radius=radius, kernel_np=kernel),
    )
    out = mapper(agg)(agg.data)
    return xr.DataArray(out, coords=agg.coords, dims=agg.dims,
                        attrs=agg.attrs)


# ---------------------------------------------------------------------------
# Weiss (2001) landform classification
# ---------------------------------------------------------------------------

LANDFORM_CLASSES = {
    1: 'Canyon / deeply incised stream',
    2: 'Midslope drainage / shallow valley',
    3: 'Upland drainage / headwater',
    4: 'U-shaped valley',
    5: 'Plain',
    6: 'Open slope',
    7: 'Upper slope / mesa',
    8: 'Local ridge / hill in valley',
    9: 'Midslope ridge / small hill',
    10: 'Mountain top / high ridge',
}


@supports_dataset
def landforms(agg: xr.DataArray,
              inner_radius: int = 3,
              outer_radius: int = 15,
              slope_threshold: float = 5.0,
              name: Optional[str] = 'landforms') -> xr.DataArray:
    """Classify terrain into landform types using the Weiss (2001) scheme.

    Computes TPI at two neighborhood scales, standardizes to z-scores,
    and classifies each cell into one of 10 landform categories based
    on its relative position at both scales and its slope.

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or CuPy-backed Dask
        xarray DataArray of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    inner_radius : int, default=3
        Radius in cells for the small-scale TPI neighborhood.
    outer_radius : int, default=15
        Radius in cells for the large-scale TPI neighborhood.
    slope_threshold : float, default=5.0
        Slope in degrees separating plains (class 5) from open
        slopes (class 6).
    name : str, default='landforms'
        Name of the output DataArray.

    Returns
    -------
    xarray.DataArray or xr.Dataset
        Integer-coded raster of landform classes:

        ==  =================================
        1   Canyon / deeply incised stream
        2   Midslope drainage / shallow valley
        3   Upland drainage / headwater
        4   U-shaped valley
        5   Plain
        6   Open slope
        7   Upper slope / mesa
        8   Local ridge / hill in valley
        9   Midslope ridge / small hill
        10  Mountain top / high ridge
        ==  =================================

    References
    ----------
    Weiss, A. (2001). Topographic Position and Landforms Analysis.
    Poster presentation, ESRI International User Conference,
    San Diego, CA.
    """
    _validate_raster(agg, func_name='landforms', name='agg')

    if inner_radius < 1:
        raise ValueError(
            f"inner_radius must be >= 1, got {inner_radius}")
    if outer_radius <= inner_radius:
        raise ValueError(
            f"outer_radius ({outer_radius}) must be greater than "
            f"inner_radius ({inner_radius})")

    # 1. TPI at two scales
    tpi_s = _compute_tpi_at_radius(agg, inner_radius)
    tpi_l = _compute_tpi_at_radius(agg, outer_radius)

    # 2. Standardize to z-scores.
    #    Global mean/std are needed, so for dask we compute all four
    #    reductions in one call to share the task graph.
    if da is not None and isinstance(agg.data, da.Array):
        import dask
        s_mean, s_std, l_mean, l_std = dask.compute(
            da.nanmean(tpi_s.data), da.nanstd(tpi_s.data),
            da.nanmean(tpi_l.data), da.nanstd(tpi_l.data),
        )
        s_mean, s_std = float(s_mean), float(s_std)
        l_mean, l_std = float(l_mean), float(l_std)
    else:
        s_mean = float(tpi_s.mean(skipna=True))
        s_std = float(tpi_s.std(skipna=True))
        l_mean = float(tpi_l.mean(skipna=True))
        l_std = float(tpi_l.std(skipna=True))

    if s_std > 0:
        tpi_s_z = (tpi_s - s_mean) / s_std
    else:
        tpi_s_z = tpi_s * 0

    if l_std > 0:
        tpi_l_z = (tpi_l - l_mean) / l_std
    else:
        tpi_l_z = tpi_l * 0

    # 3. Slope (used only to split plains from open slopes)
    slp = _slope_func(agg)

    # 4. Classify into 10 Weiss landform classes
    s_lo = tpi_s_z < -1
    s_hi = tpi_s_z > 1
    s_mid = ~s_lo & ~s_hi
    l_lo = tpi_l_z < -1
    l_hi = tpi_l_z > 1
    l_mid = ~l_lo & ~l_hi

    out = xr.full_like(agg, np.nan, dtype=np.float64)
    out = xr.where(s_lo & l_lo, 1.0, out)
    out = xr.where(s_lo & l_mid, 2.0, out)
    out = xr.where(s_lo & l_hi, 3.0, out)
    out = xr.where(s_mid & l_lo, 4.0, out)
    out = xr.where(s_mid & l_mid & (slp <= slope_threshold), 5.0, out)
    out = xr.where(s_mid & l_mid & (slp > slope_threshold), 6.0, out)
    out = xr.where(s_mid & l_hi, 7.0, out)
    out = xr.where(s_hi & l_lo, 8.0, out)
    out = xr.where(s_hi & l_mid, 9.0, out)
    out = xr.where(s_hi & l_hi, 10.0, out)

    # Preserve original NaN
    out = xr.where(agg.isnull(), np.nan, out)

    return xr.DataArray(out.data,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
