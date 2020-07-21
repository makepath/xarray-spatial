import numba as nb
import numpy as np


ngjit = nb.jit(nopython=True, nogil=True)


def lnglat_to_meters(longitude, latitude):
    """
    Projects the given (longitude, latitude) values into Web Mercator
    coordinates (meters East of Greenwich and meters North of the Equator).

    Longitude and latitude can be provided as scalars, Pandas columns,
    or Numpy arrays, and will be returned in the same form.  Lists
    or tuples will be converted to Numpy arrays.

    Examples:
    easting, northing = lnglat_to_meters(-40.71,74)

    easting, northing = lnglat_to_meters(np.array([-74]),np.array([40.71]))

    df=pandas.DataFrame(dict(longitude=np.array([-74]),latitude=np.array([40.71])))
    df.loc[:, 'longitude'], df.loc[:, 'latitude'] = lnglat_to_meters(df.longitude,df.latitude)
    """
    if isinstance(longitude, (list, tuple)):
        longitude = np.array(longitude)
    if isinstance(latitude, (list, tuple)):
        latitude = np.array(latitude)

    origin_shift = np.pi * 6378137
    easting = longitude * origin_shift / 180.0
    northing = np.log(np.tan((90 + latitude) * np.pi / 360.0)) * origin_shift / np.pi
    return (easting, northing)


def height_implied_by_aspect_ratio(W, X, Y):
    """
    Utility function for calculating height (in pixels)
    which is implied by a width, x-range, and y-range.
    Simple ratios are used to maintain aspect ratio.

    Parameters
    ----------
    W: int
      width in pixel
    X: tuple(xmin, xmax)
      x-range in data units
    Y: tuple(xmin, xmax)
      x-range in data units

    Returns
    -------
    H: int
      height in pixels

    Example
    -------
    plot_width = 1000
    x_range = (0,35
    y_range = (0, 70)
    plot_height = height_implied_by_aspect_ratio(plot_width, x_range, y_range)
    """
    return int((W * (Y[1] - Y[0])) / (X[1] - X[0]))
