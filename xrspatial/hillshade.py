from functools import partial

import numpy as np

from xarray import DataArray

import dask.array as da


def _hillshade(data, azimuth=225, angle_altitude=25):
    azimuth = 360.0 - azimuth
    x, y = np.gradient(data)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.
    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)
    result = (shaded + 1) / 2
    result[(0, -1), :] = np.nan
    result[:, (0, -1)] = np.nan
    return data


def hillshade(agg, azimuth=225, angle_altitude=25, name='hillshade'):
    """Illuminates 2D DataArray from specific azimuth and altitude.

    Parameters
    ----------
    agg : DataArray
    altitude : int, optional (default: 30)
        Altitude angle of the sun specified in degrees.
    azimuth : int, optional (default: 315)
        The angle between the north vector and the perpendicular projection
        of the light source down onto the horizon specified in degrees.
    cmap : list of colors or matplotlib.colors.Colormap, optional
        The colormap to use. Can be either a list of colors (in any of the
        formats described above), or a matplotlib colormap object.
        Default is `["lightgray", "black"]`
    alpha : int, optional
        Value between 0 - 255 representing the alpha value of pixels which contain
        data (i.e. non-nan values). Regardless of this value, `NaN` values are
        set to fully transparent.

    Returns
    -------
    Datashader Image

    Notes:
    ------
    Algorithm References:
     - http://geoexamples.blogspot.com/2014/03/shaded-relief-images-using-gdal-python.html
    """

    if isinstance(agg.data, da.Array):
        _func = partial(_hillshade, azimuth=azimuth, angle_altitude=angle_altitude)
        out = agg.data.map_overlap(_func,
                                   depth=(1, 1),
                                   boundary=np.nan,
                                   meta=np.array(()))
    else:
        out = _hillshade(agg.data, azimuth, angle_altitude)

    return DataArray(out, name=name, dims=agg.dims,
                     coords=agg.coords, attrs=agg.attrs)
