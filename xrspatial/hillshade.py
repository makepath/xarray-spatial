import math
from functools import partial
from typing import Optional

import dask.array as da
import numpy as np
import xarray as xr
from numba import cuda

from .gpu_rtx import has_rtx
from .utils import calc_cuda_dims, has_cuda_and_cupy, is_cupy_array, is_cupy_backed


def _run_numpy(data, azimuth=225, angle_altitude=25):
    data = data.astype(np.float32)

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
    data = data.astype(np.float32)

    _func = partial(_run_numpy, azimuth=azimuth, angle_altitude=angle_altitude)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


@cuda.jit
def _gpu_calc_numba(
    data,
    output,
    sin_altituderad,
    cos_altituderad,
    azimuthrad
):

    i, j = cuda.grid(2)
    if i > 0 and i < data.shape[0]-1 and j > 0 and j < data.shape[1] - 1:
        x = (data[i+1, j]-data[i-1, j])/2
        y = (data[i, j+1]-data[i, j-1])/2

        len = math.sqrt(x*x + y*y)
        slope = 1.57079632679 - math.atan(len)
        aspect = (azimuthrad - 1.57079632679) - math.atan2(-x, y)

        sin_slope = math.sin(slope)
        sin_part = sin_altituderad * sin_slope

        cos_aspect = math.cos(aspect)
        cos_slope = math.cos(slope)
        cos_part = cos_altituderad * cos_slope * cos_aspect

        res = sin_part + cos_part
        output[i, j] = (res + 1) * 0.5


def _run_cupy(d_data, azimuth, angle_altitude):
    # Precompute constant values shared between all threads
    altituderad = angle_altitude * np.pi / 180.
    sin_altituderad = np.sin(altituderad)
    cos_altituderad = np.cos(altituderad)
    azimuthrad = (360.0 - azimuth) * np.pi / 180.

    # Allocate output buffer and launch kernel with appropriate dimensions
    import cupy
    d_data = d_data.astype(cupy.float32)
    output = cupy.empty(d_data.shape, np.float32)
    griddim, blockdim = calc_cuda_dims(d_data.shape)
    _gpu_calc_numba[griddim, blockdim](
        d_data, output, sin_altituderad, cos_altituderad, azimuthrad
    )

    # Fill borders with nans.
    output[0, :] = cupy.nan
    output[-1, :] = cupy.nan
    output[:,  0] = cupy.nan
    output[:, -1] = cupy.nan

    return output


def hillshade(agg: xr.DataArray,
              azimuth: int = 225,
              angle_altitude: int = 25,
              name: Optional[str] = 'hillshade',
              shadows: bool = False) -> xr.DataArray:
    """
    Calculates, for all cells in the array, an illumination value of
    each cell based on illumination from a specific azimuth and
    altitude.

    Parameters
    ----------
    agg : xarray.DataArray
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of elevation values.
    angle_altitude : int, default=25
        Altitude angle of the sun specified in degrees.
    azimuth : int, default=225
        The angle between the north vector and the perpendicular
        projection of the light source down onto the horizon
        specified in degrees.
    name : str, default='hillshade'
        Name of output DataArray.
    shadows : bool, default=False
        Whether to calculate shadows or not. Shadows are available
        only for Cupy-backed Dask arrays and only if rtxpy is
        installed and appropriate graphics hardware is available.

    Returns
    -------
    hillshade_agg : xarray.DataArray, of same type as `agg`
        2D aggregate array of illumination values.

    References
    ----------
        - GeoExamples: http://geoexamples.blogspot.com/2014/03/shaded-relief-images-using-gdal-python.html # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import hillshade
        >>> data = np.array([
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 1., 0., 2., 0.],
        ...    [0., 0., 3., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.]])
        >>> n, m = data.shape
        >>> raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
        >>> raster['y'] = np.arange(n)[::-1]
        >>> raster['x'] = np.arange(m)
        >>> hillshade_agg = hillshade(raster)
        >>> print(hillshade_agg)
        <xarray.DataArray 'hillshade' (y: 5, x: 5)>
        array([[       nan,        nan,        nan,        nan,        nan],
               [       nan, 0.71130913, 0.44167341, 0.71130913,        nan],
               [       nan, 0.95550163, 0.71130913, 0.52478473,        nan],
               [       nan, 0.71130913, 0.88382559, 0.71130913,        nan],
               [       nan,        nan,        nan,        nan,        nan]])
        Coordinates:
          * y        (y) int32 4 3 2 1 0
          * x        (x) int32 0 1 2 3 4
    """

    if shadows and not has_rtx():
        raise RuntimeError(
            "Can only calculate shadows if cupy and rtxpy are available")

    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy(agg.data, azimuth, angle_altitude)

    # cupy/numba case
    elif has_cuda_and_cupy() and is_cupy_array(agg.data):
        if shadows and has_rtx():
            from .gpu_rtx.hillshade import hillshade_rtx
            out = hillshade_rtx(agg, azimuth, angle_altitude, shadows=shadows)
        else:
            out = _run_cupy(agg.data, azimuth, angle_altitude)

    # dask + cupy case
    elif (has_cuda_and_cupy() and isinstance(agg.data, da.Array) and
            is_cupy_backed(agg)):
        raise NotImplementedError("Dask/CuPy hillshade not implemented")

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
