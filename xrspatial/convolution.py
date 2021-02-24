import re
import numpy as np

import warnings
warnings.simplefilter('default')

from numba import cuda, float32, prange, jit

from xrspatial.utils import has_cuda
from xrspatial.utils import cuda_args
from xrspatial.utils import get_dataarray_resolution

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False


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
        raise ValueError("Distance should be a positive numeric value.\n")

    distance = float(number)
    if distance <= 0:
        raise ValueError("Distance should be a positive.\n")

    unit = unit.lower()
    unit = unit.replace(' ', '')
    if unit not in UNITS:
        raise ValueError(
            "Distance unit should be one of the following: \n"
            "meter (meter, meters, m),\n"
            "kilometer (kilometer, kilometers, km),\n"
            "foot (foot, feet, ft),\n"
            "mile (mile, miles, ml, mls)")

    # convert distance to meters
    meters = _to_meters(distance, unit)
    return meters


def cellsize(raster):
    if 'unit' in raster.attrs:
        unit = raster.attrs['unit']
    else:
        unit = DEFAULT_UNIT
        warnings.warn('Raster distance unit not provided. '
                      'Use meter as default.', Warning)

    cellsize_x, cellsize_y = get_dataarray_resolution(raster)
    cellsize_x = _to_meters(cellsize_x, unit)
    cellsize_y = _to_meters(cellsize_y, unit)

    # When converting from lnglat_to_meters, could have negative cellsize in y
    return cellsize_x, np.abs(cellsize_y)


def _ellipse_kernel(half_w, half_h):
    # x values of interest
    x = np.linspace(-half_w, half_w, 2 * half_w + 1)
    # y values of interest, as a "column" array
    y = np.linspace(-half_h, half_h, 2 * half_h + 1)[:, None]

    # True for points inside the ellipse
    # (x / a)^2 + (y / b)^2 <= 1, avoid division to avoid rounding issue
    ellipse = (x * half_h) ** 2 + (y * half_w) ** 2 <= (half_w * half_h) ** 2
    return ellipse.astype(float)


def circle_kernel(cellsize_x, cellsize_y, radius):
    # validate radius, convert radius to meters
    r = _get_distance(str(radius))

    kernel_half_w = int(r / cellsize_x)
    kernel_half_h = int(r / cellsize_y)

    kernel = _ellipse_kernel(kernel_half_w, kernel_half_h)
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

    if r_outer - r_inner < np.sqrt((cellsize_x / 2)**2 + (cellsize_y / 2)**2):
        warnings.warn('Annulus radii are closer than cellsize distance.', Warning)

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

    if not isinstance(kernel, np.ndarray):
        raise ValueError(
            "Received a custom kernel that is not a Numpy array.",
            "The kernel received was of type {} and needs to be of type `ndarray`".format(type(kernel))
        )
    else:
        rows, cols = kernel.shape

    if (rows % 2 == 0 or cols % 2 == 0):
        raise ValueError(
            "Received custom kernel with improper dimensions.",
            "A custom kernel needs to have an odd shape, the supplied kernel has {} rows and {} columns.".format(rows, cols)
        )
    return kernel


@jit(nopython=True, nogil=True, parallel=True)
def _convolve_2d_cpu(data, kernel):
    """Apply kernel to data image."""

    # TODO: handle nan

    nx = data.shape[0]
    ny = data.shape[1]
    nkx = kernel.shape[0]
    nky = kernel.shape[1]
    wkx = nkx // 2
    wky = nky // 2

    out = np.zeros(data.shape, dtype=float32)
    out[:, :] = np.nan
    for i in prange(wkx, nx-wkx):
        iimin = max(i - wkx, 0)
        iimax = min(i + wkx + 1, nx)
        for j in prange(wky, ny-wky):
            jjmin = max(j - wky, 0)
            jjmax = min(j + wky + 1, ny)
            num = 0.0
            for ii in range(iimin, iimax, 1):
                iii = wkx + ii - i
                for jj in range(jjmin, jjmax, 1):
                    jjj = wky + jj - j
                    num += kernel[iii, jjj] * data[ii, jj]
            out[i, j] = num

    return out


# https://www.vincent-lunot.com/post/an-introduction-to-cuda-in-python-part-3/
@cuda.jit
def _convolve_2d_cuda(data, kernel, out):
    # expect a 2D grid and 2D blocks,
    # a kernel with odd numbers of rows and columns, (-1-)
    # a grayscale image

    # (-2-) 2D coordinates of the current thread:
    i, j = cuda.grid(2)

    # To compute the out at coordinates (i, j), we need to use delta_rows rows of the array
    # before and after the i_th row,
    # as well as delta_cols columns of the array before and after the j_th column:
    delta_rows = kernel.shape[0] // 2
    delta_cols = kernel.shape[1] // 2

    data_rows, data_cols = data.shape
    # (-3-) if the thread coordinates are outside of the data image, we ignore the thread
    # currently, if the thread coordinates are in the edges, we ignore the thread
    if i < delta_rows or i >= data_rows - delta_rows or j < delta_cols or j >= data_cols - delta_cols:
        return

    # The out at coordinates (i, j) is equal to
    # sum_{k, l} kernel[k, l] * data[i - k + delta_rows, j - l + delta_cols]
    # with k and l going through the whole kernel array:
    s = 0
    for k in range(kernel.shape[0]):
        for l in range(kernel.shape[1]):
            i_k = i - k + delta_rows
            j_l = j - l + delta_cols
            # (-4-) Check if (i_k, j_l) coordinates are inside the array:
            if (i_k >= 0) and (i_k < data_rows) and (j_l >= 0) and (j_l < data_cols):
                s += kernel[k, l] * data[i_k, j_l]
    out[i, j] = s


def _convolve_2d_gpu(data, kernel):
    out = cupy.empty(data.shape, dtype='f4')
    out[:, :] = cupy.nan
    griddim, blockdim = cuda_args(data.shape)
    _convolve_2d_cuda[griddim, blockdim](data, kernel, cupy.asarray(out))
    return out


def convolve_2d(data, kernel):
    """Function to call the 2D convolution via Numba.
    The Numba convolution function does not account for an edge so
    if we wish to take this into account, will pad the data array.
    """

    # numpy case
    if isinstance(data, np.ndarray):
        out = _convolve_2d_cpu(data, kernel)

    # cupy case
    elif has_cuda() and isinstance(data, cupy.ndarray):
        out = _convolve_2d_gpu(data, kernel)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(data)))

    return out
