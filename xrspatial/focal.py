'''Focal Related Utilities'''
import re
import warnings

from numba import prange
import numpy as np
from xarray import DataArray

from xrspatial.utils import ngjit
from xrspatial.utils import lnglat_to_meters


warnings.simplefilter('default')

DEFAULT_UNIT = 'meter'


# TODO: Make convolution more generic with numba first-class functions.

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# modified from https://stackoverflow.com/questions/3943752/the-dateutil-parser-parse-of-distance-strings
class Distance(object):
    METER = 1
    FOOT = 0.3048
    KILOMETER = 1000
    MILE = 1609.344
    UNITS = {'meter': METER,
             'meters': METER,
             'm': METER,
             'feet': FOOT,
             'foot': FOOT,
             'ft': FOOT,
             'miles': MILE,
             'mls': MILE,
             'ml': MILE,
             'kilometer': KILOMETER,
             'kilometers': KILOMETER,
             'km': KILOMETER,
             }

    def __init__(self, s):
        self.number, unit = self._get_distance_unit(s)
        self._convert(unit)

    def _get_distance_unit(self, s):
        # spit string into numbers and text
        splits = [x for x in re.split(r'(-?\d*\.?\d+)', s) if x != '']
        if len(splits) not in [1, 2]:
            raise ValueError("Invalid distance.")

        number = splits[0]
        unit = DEFAULT_UNIT
        if len(splits) == 1:
            warnings.warn('Raster distance unit not provided. '
                          'Use meter as default.', Warning)
        elif len(splits) == 2:
            unit = splits[1]

        unit = unit.lower()
        unit = unit.replace(' ', '')
        if unit not in self.UNITS:
            raise ValueError(
                "Invalid value.\n"
                "Distance unit should be one of the following: \n"
                "meter (meter, meters, m),\n"
                "kilometer (kilometer, kilometers, km),\n"
                "foot (foot, feet, ft),\n"
                "mile (mile, miles, ml, mls)")
        return number, unit

    def _convert(self, unit):
        self.number = float(self.number)
        if self.UNITS[unit] != 1:
            self.number *= self.UNITS[unit]

    @property
    def meters(self):
        return self.number

    @meters.setter
    def meters(self, v):
        self.number = float(v)

    @property
    def miles(self):
        return self.number / self.MILE

    @miles.setter
    def miles(self, v):
        self.number = v
        self._convert('miles')

    @property
    def feet(self):
        return self.number / self.FOOT

    @feet.setter
    def feet(self, v):
        self.number = v
        self._convert('feet')

    @property
    def kilometers(self):
        return self.number / self.KILOMETER

    @kilometers.setter
    def kilometers(self, v):
        self.number = v
        self._convert('KILOMETER')


def _calc_cell_size(raster):
    if 'unit' in raster.attrs:
        unit = raster.attrs['unit']
    else:
        unit = DEFAULT_UNIT
        warnings.warn('Raster distance unit not provided. '
                      'Use meter as default.', Warning)

    cell_size_x = 1
    cell_size_y = 1

    # calculate cell size from input `raster`
    for dim in raster.dims:
        if (dim.lower().count('x')) > 0:
            # dimension of x-coordinates
            if len(raster[dim]) > 1:
                cell_size_x = raster[dim].values[1] - raster[dim].values[0]
        elif (dim.lower().count('y')) > 0:
            # dimension of y-coordinates
            if len(raster[dim]) > 1:
                cell_size_y = raster[dim].values[1] - raster[dim].values[0]

    lon0, lon1, lat0, lat1 = None, None, None, None
    for dim in raster.dims:
        if (dim.lower().count('lon')) > 0:
            # dimension of x-coordinates
            if len(raster[dim]) > 1:
                lon0, lon1 = raster[dim].values[0], raster[dim].values[1]
        elif (dim.lower().count('lat')) > 0:
            # dimension of y-coordinates
            if len(raster[dim]) > 1:
                lat0, lat1 = raster[dim].values[0], raster[dim].values[1]

    # convert lat-lon to meters
    if (lon0, lon1, lat0, lat1) != (None, None, None, None):
        mx0, my0 = lnglat_to_meters(lon0, lat0)
        mx1, my1 = lnglat_to_meters(lon1, lat1)
        cell_size_x = mx1 - mx0
        cell_size_y = my1 - my0
        unit = DEFAULT_UNIT

    sx = Distance(str(cell_size_x) + unit)
    sy = Distance(str(cell_size_y) + unit)
    return sx, sy


def _gen_ellipse_kernel(half_w, half_h):
    # x values of interest
    x = np.linspace(-half_w, half_w, 2 * half_w + 1)
    # y values of interest, as a "column" array
    y = np.linspace(-half_h, half_h, 2 * half_h + 1)[:, None]

    # True for points inside the ellipse
    # (x / a)^2 + (y / b)^2 <= 1, avoid division to avoid rounding issue
    ellipse = (x * half_h) ** 2 + (y * half_w) ** 2 <= (half_w * half_h) ** 2

    return ellipse.astype(float)


class Kernel:
    def __init__(self, shape='circle', radius=10000):
        self.shape = shape
        self.radius = radius
        self._validate_shape()
        self._validate_radius()

    def _validate_shape(self):
        # validate shape
        if self.shape not in ['circle']:
            raise ValueError(
                "Kernel shape must be \'circle\'")

    def _validate_radius(self):
        # try to convert into Distance object
        d = Distance(str(self.radius))
        print(d)

    def to_array(self, raster):
        # calculate cell size over the x and y axis
        sx, sy = _calc_cell_size(raster)
        # create Distance object of radius
        sr = Distance(str(self.radius))
        if self.shape == 'circle':
            # convert radius (meter) to pixel
            kernel_half_w = int(sr.meters / sx.meters)
            kernel_half_h = int(sr.meters / sy.meters)
            kernel = _gen_ellipse_kernel(kernel_half_w, kernel_half_h)
        return kernel


@ngjit
def _mean(data, excludes):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):

            exclude = False
            for ex in excludes:
                if data[y, x] == ex:
                    exclude = True
                    break

            if not exclude:
                a,b,c,d,e,f,g,h,i = [data[y-1, x-1], data[y, x-1], data[y+1, x-1],
                                     data[y-1, x],   data[y, x],   data[y+1, x],
                                     data[y-1, x+1], data[y, x+1], data[y+1, x+1]]
                out[y, x] = (a+b+c+d+e+f+g+h+i) / 9
            else:
                out[y, x] = data[y, x]
    return out


# TODO: add optional name parameter `name='mean'`
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


def apply(raster, kernel, func=calc_mean):
    # validate raster
    if not isinstance(raster, DataArray):
        raise TypeError("`raster` must be instance of DataArray")

    if raster.ndim != 2:
        raise ValueError("`raster` must be 2D")

    if not (issubclass(raster.values.dtype.type, np.integer) or
            issubclass(raster.values.dtype.type, np.float)):
        raise ValueError(
            "`raster` must be an array of integers or float")

    # create kernel mask array
    kernel_values = kernel.to_array(raster)
    # apply kernel to raster values
    out = _apply(raster.values.astype(float), kernel_values, func)

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


def hotspots(raster, kernel):
    """Identify statistically significant hot spots and cold spots in an input
    raster. To be a statistically significant hot spot, a feature will have a
    high value and be surrounded by other features with high values as well.
    Neighborhood of a feature defined by the input kernel, which currently
    support a shape of circle and a radius in meters.

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
            issubclass(raster.values.dtype.type, np.float)):
        raise ValueError(
            "`raster` must be an array of integers or float")

    # create kernel mask array
    kernel_values = kernel.to_array(raster)
    # apply kernel to raster values
    mean_array = _apply(raster.values.astype(float), kernel_values, calc_mean)

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
