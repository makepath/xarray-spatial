import math
from functools import partial
from typing import Optional

import numpy as np

try:
    import dask.array as da
except ImportError:
    da = None

import xarray as xr
from numba import cuda

from .gpu_rtx import has_rtx
from .utils import (calc_cuda_dims, get_dataarray_resolution,
                    has_cuda_and_cupy, is_cupy_array, is_cupy_backed)
from .dataset_support import supports_dataset


def _run_numpy(data, azimuth=225, angle_altitude=25,
               cellsize_x=1.0, cellsize_y=1.0):
    data = data.astype(np.float32)

    az_rad = azimuth * np.pi / 180.
    alt_rad = angle_altitude * np.pi / 180.
    sin_alt = np.sin(alt_rad)
    cos_alt = np.cos(alt_rad)
    sin_az = np.sin(az_rad)
    cos_az = np.cos(az_rad)

    # Gradient with actual cell spacing (matches GDAL Horn method)
    dy, dx = np.gradient(data, cellsize_y, cellsize_x)
    xx_plus_yy = dx * dx + dy * dy

    # GDAL-equivalent hillshade formula (simplified from the original
    # trig-heavy version; see issue #748 and GDAL gdaldem_lib.cpp):
    #   shaded = (sin(alt) + cos(alt) * sqrt(xx+yy) * sin(aspect - az))
    #            / sqrt(1 + xx+yy)
    # where aspect = atan2(dy, dx), expanded inline:
    #   sin(aspect - az) = (dy*cos(az) - dx*sin(az)) / sqrt(xx+yy)
    # so sqrt(xx+yy) cancels, giving:
    shaded = (sin_alt + cos_alt * (dy * cos_az - dx * sin_az)) \
        / np.sqrt(1.0 + xx_plus_yy)

    # Clamp negatives (shadow) then scale to [0, 1]
    result = np.clip(shaded, 0.0, 1.0)
    result[(0, -1), :] = np.nan
    result[:, (0, -1)] = np.nan
    return result


def _run_dask_numpy(data, azimuth, angle_altitude,
                    cellsize_x=1.0, cellsize_y=1.0):
    data = data.astype(np.float32)

    _func = partial(_run_numpy, azimuth=azimuth,
                    angle_altitude=angle_altitude,
                    cellsize_x=cellsize_x, cellsize_y=cellsize_y)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


@cuda.jit
def _gpu_calc_numba(
    data,
    output,
    sin_alt,
    cos_alt,
    sin_az,
    cos_az,
    cellsize_x,
    cellsize_y,
):

    i, j = cuda.grid(2)
    if i > 0 and i < data.shape[0]-1 and j > 0 and j < data.shape[1] - 1:
        dx = (data[i, j+1] - data[i, j-1]) / (2.0 * cellsize_x)
        dy = (data[i+1, j] - data[i-1, j]) / (2.0 * cellsize_y)

        xx_plus_yy = dx * dx + dy * dy
        shaded = (sin_alt + cos_alt * (dy * cos_az - dx * sin_az)) \
            / math.sqrt(1.0 + xx_plus_yy)

        if shaded < 0.0:
            shaded = 0.0
        output[i, j] = shaded


def _run_dask_cupy(data, azimuth, angle_altitude,
                   cellsize_x=1.0, cellsize_y=1.0):
    import cupy
    data = data.astype(cupy.float32)

    _func = partial(_run_cupy, azimuth=azimuth,
                    angle_altitude=angle_altitude,
                    cellsize_x=cellsize_x, cellsize_y=cellsize_y)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=cupy.nan,
                           meta=cupy.array(()))
    return out


def _run_cupy(d_data, azimuth, angle_altitude,
              cellsize_x=1.0, cellsize_y=1.0):
    altituderad = angle_altitude * np.pi / 180.
    azimuthrad = azimuth * np.pi / 180.
    sin_alt = np.sin(altituderad)
    cos_alt = np.cos(altituderad)
    sin_az = np.sin(azimuthrad)
    cos_az = np.cos(azimuthrad)

    import cupy
    d_data = d_data.astype(cupy.float32)
    output = cupy.empty(d_data.shape, np.float32)
    griddim, blockdim = calc_cuda_dims(d_data.shape)
    _gpu_calc_numba[griddim, blockdim](
        d_data, output, sin_alt, cos_alt, sin_az, cos_az,
        float(cellsize_x), float(cellsize_y),
    )

    # Fill borders with nans.
    output[0, :] = cupy.nan
    output[-1, :] = cupy.nan
    output[:,  0] = cupy.nan
    output[:, -1] = cupy.nan

    return output


@supports_dataset
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
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask, or Cupy-backed Dask array
        of elevation values.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
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
    hillshade_agg : xarray.DataArray or xr.Dataset
        If `agg` is a DataArray, returns a DataArray of the same type.
        If `agg` is a Dataset, returns a Dataset with hillshade computed
        for each data variable.
        2D aggregate array of illumination values.

    References
    ----------
        - GDAL gdaldem hillshade: https://gdal.org/programs/gdaldem.html
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
    """

    if shadows and not has_rtx():
        raise RuntimeError(
            "Can only calculate shadows if cupy and rtxpy are available")

    cellsize_x, cellsize_y = get_dataarray_resolution(agg)

    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _run_numpy(agg.data, azimuth, angle_altitude,
                         cellsize_x, cellsize_y)

    # cupy/numba case
    elif has_cuda_and_cupy() and is_cupy_array(agg.data):
        if shadows and has_rtx():
            from .gpu_rtx.hillshade import hillshade_rtx
            out = hillshade_rtx(agg, azimuth, angle_altitude, shadows=shadows)
        else:
            out = _run_cupy(agg.data, azimuth, angle_altitude,
                            cellsize_x, cellsize_y)

    # dask + cupy case
    elif (has_cuda_and_cupy() and da is not None and isinstance(agg.data, da.Array) and
            is_cupy_backed(agg)):
        out = _run_dask_cupy(agg.data, azimuth, angle_altitude,
                             cellsize_x, cellsize_y)

    # dask + numpy case
    elif da is not None and isinstance(agg.data, da.Array):
        out = _run_dask_numpy(agg.data, azimuth, angle_altitude,
                              cellsize_x, cellsize_y)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
