'''Focal Related Utilities'''
import re
import warnings

from numba import prange
import numpy as np
from xarray import DataArray

from xrspatial.utils import ngjit
from xrspatial.convolution import convolve_2d

warnings.simplefilter('default')

# TODO: Make convolution more generic with numba first-class functions.


DEFAULT_UNIT = 'meter'
METER = 1
FOOT = 0.3048
KILOMETER = 1000
MILE = 1609.344
UNITS = {'meter': METER, 'meters': METER, 'm': METER,
         'feet': FOOT, 'foot': FOOT, 'ft': FOOT,
         'miles': MILE, 'mls': MILE, 'ml': MILE,
         'kilometer': KILOMETER, 'kilometers': KILOMETER, 'km': KILOMETER}


def _is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _to_meters(d, unit):
    return d * UNITS[unit]


def _get_distance(distance_str):
    # return distance in meters

    # spit string into numbers and text
    splits = [x for x in re.split(r'(-?\d*\.?\d+)', distance_str) if x != '']
    if len(splits) not in [1, 2]:
        raise ValueError("Invalid distance.")

    unit = DEFAULT_UNIT
    if len(splits) == 1:
        warnings.warn('Raster distance unit not provided. '
                      'Use meter as default.', Warning)
    elif len(splits) == 2:
        unit = splits[1]

    number = splits[0]
    if not _is_numeric(number):
        raise ValueError(
            "Invalid value.\n"
            "Distance should be a possitive numeric value.\n")

    distance = float(number)
    if distance <= 0:
        raise ValueError(
            "Invalid value.\n"
            "Distance should be a possitive.\n")

    unit = unit.lower()
    unit = unit.replace(' ', '')
    if unit not in UNITS:
        raise ValueError(
            "Invalid value.\n"
            "Distance unit should be one of the following: \n"
            "meter (meter, meters, m),\n"
            "kilometer (kilometer, kilometers, km),\n"
            "foot (foot, feet, ft),\n"
            "mile (mile, miles, ml, mls)")

    # convert distance to meters
    meters = _to_meters(distance, unit)
    return meters


def calc_cellsize(raster, x='x', y='y'):
    """
Calculates cell size of an array based on its attributes. Default = meters. If lat-lon units are converted to meters.
Parameters:
----------
    raster: xarray.DataArray
        - 2D array of input values.
    x: String (Default = "x")
        - Name of input x-axis.
    y: String (Default = "y")
        - Name of input y-axis

Returns:
----------
    cellsize_x: Int
        - Size of cells in x direction.
    cellsize_y: Int
        - Size of cells in y direction.

Notes:
----------

Examples:
----------

    """
    if 'unit' in raster.attrs:
        unit = raster.attrs['unit']
    else:
        unit = DEFAULT_UNIT
        warnings.warn('Raster distance unit not provided. '
                      'Use meter as default.', Warning)

    # TODO: check coordinate system
    #       if in lat-lon, need to convert to meter, lnglat_to_meters
    cellsize_x = raster.coords[x].data[1] - raster.coords[x].data[0]
    cellsize_y = raster.coords[y].data[1] - raster.coords[y].data[0]
    cellsize_x = _to_meters(cellsize_x, unit)
    cellsize_y = _to_meters(cellsize_y, unit)

    # When converting from lnglat_to_meters, could have negative cellsize in y
    return cellsize_x, np.abs(cellsize_y)


def _gen_ellipse_kernel(half_w, half_h):
    # x values of interest
    x = np.linspace(-half_w, half_w, 2 * half_w + 1)
    # y values of interest, as a "column" array
    y = np.linspace(-half_h, half_h, 2 * half_h + 1)[:, None]

    # True for points inside the ellipse
    # (x / a)^2 + (y / b)^2 <= 1, avoid division to avoid rounding issue
    ellipse = (x * half_h) ** 2 + (y * half_w) ** 2 <= (half_w * half_h) ** 2
    return ellipse.astype(float)


def _validate_kernel(kernel):
    """Validatetes that the kernel is a numpy array and has odd dimensions."""

    if not isinstance(kernel, np.ndarray):
        raise ValueError(
            "Received a custom kernel that is not a Numpy array.",
            """The kernel received was of type {} and needs to be of type `ndarray`
            """.format(type(kernel))
        )
    else:
        rows, cols = kernel.shape

    if (rows % 2 == 0 or cols % 2 == 0):
        raise ValueError(
            "Received custom kernel with improper dimensions.",
            """A custom kernel needs to have an odd shape, the
            supplied kernel has {} rows and {} columns.
            """.format(rows, cols)
        )


def circle_kernel(cellsize_x, cellsize_y, radius):
    # validate radius, convert radius to meters
    r = _get_distance(str(radius))

    kernel_half_w = int(r / cellsize_x)
    kernel_half_h = int(r / cellsize_y)

    kernel = _gen_ellipse_kernel(kernel_half_w, kernel_half_h)
    return kernel


def annulus_kernel(cellsize_x, cellsize_y, outer_radius, inner_radius):

    # validate radii, convert to meters
    r2 = _get_distance(str(outer_radius))
    r1 = _get_distance(str(inner_radius))

    # Validate that outer radius is indeed outer radius
    if r2 > r1:
        r_outer = r2
        r_inner = r1
    else:
        r_outer = r1
        r_inner = r2

    if r_outer - r_inner < np.sqrt((cellsize_x / 2)**2 + \
                                   (cellsize_y / 2)**2):
        warnings.warn('Annulus radii are closer than cellsize distance.',
                      Warning)

    # Get the two circular kernels for the annulus
    kernel_outer = circle_kernel(cellsize_x, cellsize_y, outer_radius)
    kernel_inner = circle_kernel(cellsize_x, cellsize_y, inner_radius)

    # Need to pad kernel_inner to get it the same shape and centered
    # in kernel_outer
    pad_vals = np.array(kernel_outer.shape) - np.array(kernel_inner.shape)
    pad_kernel = np.pad(kernel_inner,
                        # Pad ((before_rows, after_rows),
                        #      (before_cols, after_cols))
                        pad_width=((pad_vals[0] // 2, pad_vals[0] // 2),
                                   (pad_vals[1] // 2, pad_vals[1] // 2)),
                        mode='constant',
                        constant_values=0)
    # Get annulus by subtracting inner from outer
    kernel = kernel_outer - pad_kernel
    return kernel


def custom_kernel(kernel):
    """Validates a custom kernel. If the kernel is valid, returns itself."""
    _validate_kernel(kernel)
    return kernel


@ngjit
def _mean(data, excludes):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):

            exclude = False
            for ex in excludes:
                if data[y, x] == ex:
                    exclude = True
                    break

            if not exclude:
                a, b, c, d, e, f, g, h, i = [data[y - 1, x - 1],
                                             data[y, x - 1],
                                             data[y + 1, x - 1],
                                             data[y - 1, x], data[y, x],
                                             data[y + 1, x],
                                             data[y - 1, x + 1],
                                             data[y, x + 1],
                                             data[y + 1, x + 1]]
                out[y, x] = (a + b + c + d + e + f + g + h + i) / 9
            else:
                out[y, x] = data[y, x]
    return out


def mean(agg, passes=1, excludes=[np.nan], name='mean'):
    """
    Returns Mean filtered array using a 3x3 window

    Parameters
    ----------
    agg : DataArray
    passes : int
      number of times to run mean
    name : str
      output xr.DataArray.name property

    Returns
    -------
    data: DataArray
    """
    out = None
    for i in range(passes):
        if out is None:
            out = _mean(agg.data, tuple(excludes))
        else:
            out = _mean(out, tuple(excludes))

    return DataArray(out, name=name, dims=agg.dims,
                     coords=agg.coords, attrs=agg.attrs)


@ngjit
def calc_mean(array):
    return np.nanmean(array)


@ngjit
def calc_sum(array):
    return np.nansum(array)


@ngjit
def upper_bound_p_value(zscore):
    if abs(zscore) >= 2.33:
        return 0.0099
    if abs(zscore) >= 1.65:
        return 0.0495
    if abs(zscore) >= 1.29:
        return 0.0985
    return 1


@ngjit
def _hot_cold(zscore):
    if zscore > 0:
        return 1
    if zscore < 0:
        return -1
    return 0


@ngjit
def _confidence(zscore):
    p_value = upper_bound_p_value(zscore)
    if abs(zscore) > 2.58 and p_value < 0.01:
        return 99
    if abs(zscore) > 1.96 and p_value < 0.05:
        return 95
    if abs(zscore) > 1.65 and p_value < 0.1:
        return 90
    return 0


@ngjit
def _apply(data, kernel_array, func):
    out = np.zeros_like(data)
    rows, cols = data.shape
    krows, kcols = kernel_array.shape
    hrows, hcols = int(krows / 2), int(kcols / 2)
    kernel_values = np.zeros_like(kernel_array, dtype=data.dtype)

    for y in prange(rows):
        for x in prange(cols):
            # kernel values are all nans at the beginning of each step
            kernel_values.fill(np.nan)
            for ky in range(y - hrows, y + hrows + 1):
                for kx in range(x - hcols, x + hcols + 1):
                    if ky >= 0 and kx >= 0:
                        if ky >= 0 and ky < rows and kx >= 0 and kx < cols:
                            kyidx, kxidx = ky - (y - hrows), kx - (x - hcols)
                            if kernel_array[kyidx, kxidx] == 1:
                                kernel_values[kyidx, kxidx] = data[ky, kx]
            out[y, x] = func(kernel_values)
    return out


def apply(raster, kernel, x='x', y='y', func=calc_mean):
    """
    """

    # validate raster
    if not isinstance(raster, DataArray):
        raise TypeError("`raster` must be instance of DataArray")

    if raster.ndim != 2:
        raise ValueError("`raster` must be 2D")

    if not (issubclass(raster.values.dtype.type, np.integer) or
            issubclass(raster.values.dtype.type, np.floating)):
        raise ValueError(
            "`raster` must be an array of integers or float")

    raster_dims = raster.dims
    if raster_dims != (y, x):
        raise ValueError("raster.coords should be named as coordinates:"
                         "(%s, %s)".format(y, x))

    # Validate the kernel
    _validate_kernel(kernel)

    # apply kernel to raster values
    out = _apply(raster.values.astype(float), kernel, func)

    result = DataArray(out,
                       coords=raster.coords,
                       dims=raster.dims,
                       attrs=raster.attrs)

    return result


@ngjit
def _hotspots(z_array):
    out = np.zeros_like(z_array, dtype=np.int8)
    rows, cols = z_array.shape
    for y in prange(rows):
        for x in prange(cols):
            out[y, x] = _hot_cold(z_array[y, x]) * _confidence(z_array[y, x])
    return out


def hotspots(raster, kernel, x='x', y='y'):
    """Identify statistically significant hot spots and cold spots in an input
    raster. To be a statistically significant hot spot, a feature will have a
    high value and be surrounded by other features with high values as well.
    Neighborhood of a feature defined by the input kernel, which currently
    support a shape of circle, annulus, or custom kernel.

    The result should be a raster with the following 7 values:
    90 for 90% confidence high value cluster
    95 for 95% confidence high value cluster
    99 for 99% confidence high value cluster
    -90 for 90% confidence low value cluster
    -95 for 95% confidence low value cluster
    -99 for 99% confidence low value cluster
    0 for no significance

    Parameters
    ----------
    raster: xarray.DataArray
        Input raster image with shape=(height, width)
    kernel: Kernel

    Returns
    -------
    hotspots: xarray.DataArray
    """

    # validate raster
    if not isinstance(raster, DataArray):
        raise TypeError("`raster` must be instance of DataArray")

    if raster.ndim != 2:
        raise ValueError("`raster` must be 2D")

    if not (issubclass(raster.values.dtype.type, np.integer) or
            issubclass(raster.values.dtype.type, np.floating)):
        raise ValueError(
            "`raster` must be an array of integers or float")

    raster_dims = raster.dims
    if raster_dims != (y, x):
        raise ValueError("raster.coords should be named as coordinates:"
                         "(%s, %s)".format(y, x))

    # apply kernel to raster values
    mean_array = convolve_2d(raster.values, kernel / kernel.sum(), pad=True)

    # calculate z-scores
    global_mean = np.nanmean(raster.values)
    global_std = np.nanstd(raster.values)
    if global_std == 0:
        raise ZeroDivisionError("Standard deviation "
                                "of the input raster values is 0.")
    z_array = (mean_array - global_mean) / global_std

    out = _hotspots(z_array)

    result = DataArray(out,
                       coords=raster.coords,
                       dims=raster.dims,
                       attrs=raster.attrs)

    return result
