from __future__ import division, absolute_import
from math import sqrt

import numpy as np

from xarray import DataArray
from numba import cuda

from xrspatial.utils import has_cuda
from xrspatial.utils import cuda_args


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


def hillshade(agg, azimuth=225, angle_altitude=25, name='hillshade',
              use_cuda=True, use_cupy=True):
    """
    Illuminates 2D DataArray from specific azimuth and altitude.

    Parameters
    ----------
    agg : DataArray
    angle_altitude : int, optional (default: 25)
        Altitude angle of the sun specified in degrees.
    azimuth : int, optional (default: 225)
        The angle between the north vector and the perpendicular projection
        of the light source down onto the horizon specified in degrees.
    name: str, name of output aggregate
    use_cuda: bool
    use_cupy: bool

    Returns
    -------
    Datashader Image

    Notes:
    ------
    Algorithm References:
     - http://geoexamples.blogspot.com/2014/03/shaded-relief-images-using-gdal-python.html
    """

    data = agg.data
    x, y = np.gradient(data)

    altituderad = angle_altitude * np.pi / 180.
    sin_altituderad = np.sin(altituderad)
    cos_altituderad = np.cos(altituderad)

    if has_cuda() and use_cuda:
        griddim, blockdim = cuda_args(data.shape)
        arctan_part = np.empty(data.shape, dtype='f4')
        arctan_part[:] = np.nan

        if use_cupy:
            import cupy
            arctan_part = cupy.asarray(arctan_part)

        _gpu_calc[griddim, blockdim](x, y, arctan_part)
    else:
        arctan_part = np.sqrt(x * x + y * y)

    slope = np.pi / 2. - np.arctan(arctan_part)
    sin_slope = np.sin(slope)
    sin_part = sin_altituderad * sin_slope

    azimuthrad = (360.0 - azimuth) * np.pi / 180.
    aspect = (azimuthrad - np.pi / 2.) - np.arctan2(-x, y)
    cos_aspect = np.cos(aspect)
    cos_slope = np.cos(slope)

    if has_cuda() and use_cuda:
        griddim, blockdim = cuda_args(data.shape)
        cos_part = np.empty(data.shape, dtype='f4')
        cos_part[:] = np.nan
        if use_cupy:
            cos_part = cupy.asarray(cos_part)

        _gpu_cos_part[griddim, blockdim](cos_altituderad, cos_slope,
                                         cos_aspect, cos_part)
    else:
        cos_part = cos_altituderad * cos_slope * cos_aspect

    shaded = sin_part + cos_part
    out = (shaded + 1) / 2

    return DataArray(out,
                     name=name,
                     coords=agg.coords,
                     dims=agg.dims,
                     attrs=agg.attrs)
