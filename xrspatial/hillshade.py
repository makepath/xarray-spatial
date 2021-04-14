# std lib
from functools import partial
from math import sqrt

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

import dask.array as da

from numba import cuda

import numpy as np
import xarray as xr

# local modules
from xrspatial.utils import cuda_args
from xrspatial.utils import has_cuda
from xrspatial.utils import is_cupy_backed

from typing import Optional


def _run_numpy(data, azimuth=225, angle_altitude=25):
    azimuth = 360.0 - azimuth
    x, y = np.gradient(data)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.
    shaded = np.sin(altituderad) * np.sin(slope) + \
        np.cos(altituderad) * np.cos(slope) * \
        np.cos((azimuthrad - np.pi/2.) - aspect)
    result = (shaded + 1) / 2
    result[(0, -1), :] = np.nan
    result[:, (0, -1)] = np.nan
    return result


def _run_dask_numpy(data, azimuth, angle_altitude):
    _func = partial(_run_numpy, azimuth=azimuth, angle_altitude=angle_altitude)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


@cuda.jit
def _gpu_calc(x, y, out):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        out[i, j] = sqrt(x[i, j] * x[i, j] + y[i, j] * y[i, j])


@cuda.jit
def _gpu_cos_part(cos_altituderad, cos_slope, cos_aspect, out):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        out[i, j] = cos_altituderad * cos_slope[i, j] * cos_aspect[i, j]


def _run_cupy(data, azimuth, angle_altitude):
    x, y = np.gradient(data.get())
    x = cupy.asarray(x, dtype=x.dtype)
    y = cupy.asarray(y, dtype=y.dtype)

    altituderad = angle_altitude * np.pi / 180.
    sin_altituderad = np.sin(altituderad)
    cos_altituderad = np.cos(altituderad)

    griddim, blockdim = cuda_args(data.shape)
    arctan_part = cupy.empty(data.shape, dtype='f4')
    _gpu_calc[griddim, blockdim](x, y, arctan_part)

    slope = np.pi / 2. - np.arctan(arctan_part)
    sin_slope = np.sin(slope)
    sin_part = sin_altituderad * sin_slope

    azimuthrad = (360.0 - azimuth) * np.pi / 180.
    aspect = (azimuthrad - np.pi / 2.) - np.arctan2(-x, y)
    cos_aspect = np.cos(aspect)
    cos_slope = np.cos(slope)

    cos_part = cupy.empty(data.shape, dtype='f4')
    _gpu_cos_part[griddim, blockdim](cos_altituderad, cos_slope,
                                     cos_aspect, cos_part)
    shaded = sin_part + cos_part
    out = (shaded + 1) / 2

    out[0, :] = cupy.nan
    out[-1, :] = cupy.nan
    out[:, 0] = cupy.nan
    out[:, -1] = cupy.nan

    return out


def _run_dask_cupy(data, azimuth, angle_altitude):
    msg = 'Upstream bug in dask prevents cupy backed arrays'
    raise NotImplementedError(msg)


def hillshade(agg: xr.DataArray,
              azimuth: int = 225,
              angle_altitude: int = 25,
              name: Optional[str] = 'hillshade') -> xr.DataArray:
    """
    Calculates, for all cells in the array, an illumination
    value of each cell based on illumination from a specific
    azimuth and altitude.

    Parameters:
    ----------
    agg: xarray.DataArray
        2D array of elevation values:
        NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array.
    altitude: int (default: 30)
        Altitude angle of the sun specified in degrees.
    azimuth: int (default: 315)
        The angle between the north vector and the perpendicular projection
        of the light source down onto the horizon specified in degrees.
    name: str, optional (default = "hillshade")
        Name of output DataArray.

    Returns:
    ----------
    data: xarray.DataArray
        2D array, of the same type as the input of calculated illumination
    values.

    Notes:
    ----------
    Algorithm References:
    - http://geoexamples.blogspot.com/2014/03/shaded-relief-images-using-gdal-python.html # noqa

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrspatial

    Create Initial DataArray
    >>> agg = xr.DataArray(np.array([[0, 1, 0, 0],
    >>>                              [1, 1, 0, 0],
    >>>                              [0, 1, 2, 2],
    >>>                              [1, 0, 2, 0],
    >>>                              [0, 2, 2, 2]]),
    >>>                       dims = ["lat", "lon"])
    >>> height, width = agg.shape
    >>> _lon = np.linspace(0, width - 1, width)
    >>> _lat = np.linspace(0, height - 1, height)
    >>> agg["lon"] = _lon
    >>> agg["lat"] = _lat
    >>> print(agg)
    <xarray.DataArray (lat: 5, lon: 4)>
    array([[0, 1, 0, 0],
           [1, 1, 0, 0],
           [0, 1, 2, 2],
           [1, 0, 2, 0],
           [0, 2, 2, 2]])
    Coordinates:
      * lon      (lon) float64 0.0 1.0 2.0 3.0
      * lat      (lat) float64 0.0 1.0 2.0 3.0 4.0

    Create Hillshade DataArray
    >>> hillshade = xrspatial.hillshade(agg)
    >>> print(hillshade)
    <xarray.DataArray 'hillshade' (lat: 5, lon: 4)>
    array([[       nan,        nan,        nan,        nan],
           [       nan, 0.54570079, 0.32044456,        nan],
           [       nan, 0.96130094, 0.53406336,        nan],
           [       nan, 0.67253318, 0.71130913,        nan],
           [       nan,        nan,        nan,        nan]])
    Coordinates:
      * lon      (lon) float64 0.0 1.0 2.0 3.0
      * lat      (lat) float64 0.0 1.0 2.0 3.0 4.0
    """

    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy(agg.data, azimuth, angle_altitude)

    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        out = _run_cupy(agg.data, azimuth, angle_altitude)

    # dask + cupy case
    elif has_cuda() and isinstance(agg.data, da.Array) and is_cupy_backed(agg):
        out = _run_dask_cupy(agg.data, azimuth, angle_altitude)

    # dask + numpy case
    elif isinstance(agg.data, da.Array):
        out = _run_dask_numpy(agg.data, azimuth, angle_altitude)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
