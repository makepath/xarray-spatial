from __future__ import division, absolute_import

import numpy as np

from xarray import DataArray


def hillshade(agg, azimuth=225, angle_altitude=25, name='hillshade'):
    """Illuminates 2D DataArray from specific azimuth and altitude.

    Parameters
    ----------
    agg : DataArray
    angle_altitude : int, optional (default: 30)
        Altitude angle of the sun specified in degrees.
    azimuth : int, optional (default: 315)
        The angle between the north vector and the perpendicular projection
        of the light source down onto the horizon specified in degrees.
    name: str, name of output aggregate

    Returns
    -------
    Datashader Image

    Notes:
    ------
    Algorithm References:
     - http://geoexamples.blogspot.com/2014/03/shaded-relief-images-using-gdal-python.html
    """
    azimuth = 360.0 - azimuth
    x, y = np.gradient(agg.data)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.
    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)
    data = (shaded + 1) / 2
    return DataArray(data, name=name, dims=agg.dims,
                     coords=agg.coords, attrs=agg.attrs)
