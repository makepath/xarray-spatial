import numpy as np
from xarray import DataArray
from xrspatial.utils import ngjit
from numba import stencil
import re

DEFAULT_UNIT = 'meter'

# TODO: Make convolution more generic with numba first-class functions.


@ngjit
def _mean(data, excludes):
    out = np.zeros_like(data)
    rows, cols = data.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):

            exclude = False
            for ex in excludes:
                if data[y,x] == ex:
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
def mean(agg, passes=1, excludes=[np.nan]):
    """
    Returns Mean filtered array using a 3x3 window

    Parameters
    ----------
    agg : DataArray
    passes : int, number of times to run mean

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

    return DataArray(out, name='mean',
                     dims=agg.dims, coords=agg.coords, attrs=agg.attrs)


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
            print("Distance unit not provided. Use meter as default.")
        elif len(splits) == 2:
            unit = splits[1]

        unit = unit.lower()
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


def _zscores(array):
    mean = np.nanmean(array)
    std = np.nanstd(array)
    return (array - mean) / std


def _gen_ellipse_kernel(half_w, half_h):
    # x values of interest
    x = np.linspace(-half_w, half_w, 2 * half_w + 1)
    # y values of interest, as a "column" array
    y = np.linspace(-half_h, half_w, 2 * half_h + 1)[:, None]

    # True for points inside the ellipse
    # (x / a)^2 + (y / b)^2 <= 1, avoid division to avoid rounding issue
    ellipse = (x * half_h) ** 2 + (y * half_w) ** 2 <= (half_w * half_h) ** 2

    return ellipse.astype(int)


def _apply_convolution(array, kernel):
    kernel_half_h, kernel_half_w = kernel.shape
    h = int(kernel_half_h / 2)
    w = int(kernel_half_w / 2)

    # number of pixels inside the kernel
    num_pixels = 0

    # return of the function
    res = 0

    # row id of the kernel
    k_row = 0
    for i in range(-h, h + 1):
        # column id of the kernel
        k_col = 0
        for j in range(-w, w + 1):
            res += array[i, j] * kernel[k_row, k_col]
            if (kernel[k_row, k_col] == 1):
                num_pixels += 1
            k_col += 1
        k_row += 1

    return res / num_pixels


def focal_analysis(raster, shape='circle', radius=1):
    # check raster
    if not isinstance(raster, DataArray):
        raise TypeError("`raster` must be instance of DataArray")

    if raster.ndim != 2:
        raise ValueError("`raster` must be 2D")

    if not (issubclass(raster.values.dtype.type, np.integer) or
            issubclass(raster.values.dtype.type, np.float)):
        raise ValueError(
            "`raster` must be an array of integers or float")

    cell_size_x = 1
    cell_size_y = 1

    # calculate cell size from input `raster`
    for dim in raster.dims:
        if (dim.lower().count('x')) > 0 or (dim.lower().count('lon')) > 0:
            # dimension of x-coordinates
            if len(raster[dim]) > 1:
                cell_size_x = raster[dim].values[1] - raster[dim].values[0]
        elif (dim.lower().count('y')) > 0 or (dim.lower().count('lat')) > 0:
            # dimension of y-coordinates
            if len(raster[dim]) > 1:
                cell_size_y = raster[dim].values[1] - raster[dim].values[0]

    # TODO: check coordinate unit, convert from lat-lon to meters
    if 'unit' in raster.attrs:
        unit = raster.attrs['unit']
    else:
        unit = DEFAULT_UNIT
        print("Raster distance unit not provided. Use meter as default.")

    sx = Distance(str(cell_size_x) + unit)
    sy = Distance(str(cell_size_y) + unit)
    sr = Distance(str(radius))

    # create kernel
    if shape == 'circle':
        # convert radius (meter) to pixel
        kernel_half_w = int(sr.meters / sx.meters)
        kernel_half_h = int(sr.meters / sy.meters)
        kernel = _gen_ellipse_kernel(kernel_half_w, kernel_half_h)

    # zero padding
    height, width = raster.shape
    padded_raster_val = np.zeros((height + 2*kernel_half_h,
                                  width + 2*kernel_half_w))
    padded_raster_val[kernel_half_h:height + kernel_half_h,
                      kernel_half_w:width + kernel_half_w] = raster.values

    # apply kernel to raster values
    padded_res = stencil(_apply_convolution,
                         standard_indexing=("kernel",),
                         neighborhood=((-kernel_half_h, kernel_half_h),
                                       (-kernel_half_w, kernel_half_w)))(padded_raster_val, kernel)

    result = DataArray(padded_res[kernel_half_h:height + kernel_half_h,
                                  kernel_half_w:width + kernel_half_w],
                       coords=raster.coords,
                       dims=raster.dims,
                       attrs=raster.attrs)

    return result
