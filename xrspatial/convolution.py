from functools import partial

import re
import warnings

import numpy as np
import dask.array as da

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
        with warnings.catch_warnings():
            warnings.simplefilter('default')
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


def calc_cellsize(raster):
    """
    Calculates cell size of an array based on its attributes.
    Default = meters. If lat-lon units are converted to meters.
    Parameters:
    ----------
    raster: xarray.DataArray
        2D array of input values.
    Returns:
    ----------
    cellsize_x: float
        Size of cells in x direction.
    cellsize_y: float
        Size of cells in y direction.
    Notes:
    ----------
    Examples:
    -----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial import focal
    Create Data Array
    >>> np.random.seed(0)
    >>> agg = xr.DataArray(np.random.rand(4,4),
                               dims = ["lat", "lon"])
    >>> height, width = nir_agg.shape
    >>> _lat = np.linspace(0, height - 1, height)
    >>> _lon = np.linspace(0, width - 1, width)
    >>> nir_agg["lat"] = _lat
    >>> nir_agg["lon"] = _lon
    Calculate Cell Size
    >>> focal.calc_cellsize(agg, 'lon', 'lat')
    (1, 1)
    """

    if 'unit' in raster.attrs:
        unit = raster.attrs['unit']
    else:
        unit = DEFAULT_UNIT
        with warnings.catch_warnings():
            warnings.simplefilter('default')
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
    """
    Generates a circular kernel of a given cellsize and radius.

    Parameters:
    ----------
    cellsize_x: int
        Cell size of output kernel in x direction.
    cellsize_y: int
        Cell size of output kernel in y direction.
    radius: int
        Radius of output kernel.

    Returns:
    ----------
    kernel: NumPy Array
        2D array where values of 1 indicate the kernel.

    Examples:
    ----------
        Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial import focal

        Create Kernels
    >>> focal.circle_kernel(1, 1, 3)
    array([[0., 0., 0., 1., 0., 0., 0.],
           [0., 1., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 1., 0.],
           [0., 0., 0., 1., 0., 0., 0.]])

    >>> focal.circle_kernel(1, 2, 3)
    array([[0., 0., 0., 1., 0., 0., 0.],
           [1., 1., 1., 1., 1., 1., 1.],
           [0., 0., 0., 1., 0., 0., 0.]])
    """

    # validate radius, convert radius to meters
    r = _get_distance(str(radius))

    kernel_half_w = int(r / cellsize_x)
    kernel_half_h = int(r / cellsize_y)

    kernel = _ellipse_kernel(kernel_half_w, kernel_half_h)
    return kernel


def annulus_kernel(cellsize_x, cellsize_y, outer_radius, inner_radius):
    """
    Generates a annulus (ring-shaped) kernel of a given cellsize and radius.

    Parameters:
    ----------
    cellsize_x: int
        Cell size of output kernel in x direction.
    cellsize_y: int
        Cell size of output kernel in y direction.
    outer_radius: int
        Outer ring radius of output kernel.
    inner_radius: int
        Inner circle radius of output kernel.

    Returns:
    ----------
    kernel: NumPy Array
        2D array of 0s and 1s where values of 1 indicate the kernel.

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial import focal

    Create Kernels
    >>> focal.annulus_kernel(1, 1, 3, 1)
    array([[0., 0., 0., 1., 0., 0., 0.],
           [0., 1., 1., 1., 1., 1., 0.],
           [0., 1., 1., 0., 1., 1., 0.],
           [1., 1., 0., 0., 0., 1., 1.],
           [0., 1., 1., 0., 1., 1., 0.],
           [0., 1., 1., 1., 1., 1., 0.],
           [0., 0., 0., 1., 0., 0., 0.]])

    >>> focal.annulus_kernel(1, 2, 5, 2)
    array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0.],
           [1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.],
           [0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])
    """

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
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.warn(
                'Annulus radii are closer than cellsize distance.', Warning
            )

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
    """
    Validates a custom kernel. If the kernel is valid, returns itself.
    """

    if not isinstance(kernel, np.ndarray):
        raise ValueError(
            "Received a custom kernel that is not a Numpy array.",
            "The kernel received was of type {} and needs to be "
            "of type `ndarray`".format(type(kernel))
        )
    else:
        rows, cols = kernel.shape

    if (rows % 2 == 0 or cols % 2 == 0):
        raise ValueError(
            "Received custom kernel with improper dimensions.",
            "A custom kernel needs to have an odd shape, the supplied kernel "
            "has {} rows and {} columns.".format(rows, cols)
        )
    return kernel


@jit(nopython=True, nogil=True, parallel=True)
def _convolve_2d_numpy(data, kernel):
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


def _convolve_2d_dask_numpy(data, kernel):
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    _func = partial(_convolve_2d_numpy, kernel=kernel)
    out = data.map_overlap(_func,
                           depth=(pad_h, pad_w),
                           boundary=np.nan,
                           meta=np.array(()))
    return out


# https://www.vincent-lunot.com/post/an-introduction-to-cuda-in-python-part-3/
@cuda.jit
def _convolve_2d_cuda(data, kernel, out):
    # expect a 2D grid and 2D blocks,
    # a kernel with odd numbers of rows and columns, (-1-)
    # a grayscale image

    # (-2-) 2D coordinates of the current thread:
    i, j = cuda.grid(2)

    # To compute the out at coordinates (i, j), we need to use delta_rows rows
    # of the array before and after the i_th row, as well as delta_cols columns
    # of the array before and after the j_th column:
    delta_rows = kernel.shape[0] // 2
    delta_cols = kernel.shape[1] // 2

    data_rows, data_cols = data.shape
    # (-3-) if the thread coordinates are outside of the data image,
    # we ignore the thread
    # currently, if the thread coordinates are in the edges,
    # we ignore the thread
    if i < delta_rows or i >= data_rows - delta_rows or \
            j < delta_cols or j >= data_cols - delta_cols:
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
            if (i_k >= 0) and (i_k < data_rows) and \
                    (j_l >= 0) and (j_l < data_cols):
                s += kernel[k, l] * data[i_k, j_l]
    out[i, j] = s


def _convolve_2d_cupy(data, kernel):
    out = cupy.empty(data.shape, dtype='f4')
    out[:, :] = cupy.nan
    griddim, blockdim = cuda_args(data.shape)
    _convolve_2d_cuda[griddim, blockdim](data, kernel, cupy.asarray(out))
    return out


def _convolve_2d_dask_cupy(data, kernel):
    msg = 'Upstream bug in dask prevents cupy backed arrays'
    raise NotImplementedError(msg)


def convolve_2d(data, kernel):
    """
    Calculates, for all inner cells of an array, the 2D convolution of
    each cell via Numba. To account for edge cells, a pad can be added
    to the image array. Convolution is frequently used for image
    processing, such as smoothing, sharpening, and edge detection of
    images by elimatig spurious data or enhancing features in the data.

    Parameters:
    ----------
    image: xarray.DataArray
        2D array of values to processed and padded.
    kernel: array-like object
        Impulse kernel, determines area to apply
        impulse function for each cell.
    pad: Boolean
        To compute edges set to True.
    use-cuda: Boolean
        For parallel computing set to True.

    Returns:
    ----------
    convolve_agg: xarray.DataArray
        2D array representation of the impulse function.
        All other input attributes are preserverd.

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial import convolution, focal

    Create Data Array
    >>> agg = xr.DataArray(np.array([[0, 0, 0, 0, 0, 0, 0],
    >>>                              [0, 0, 2, 4, 0, 8, 0],
    >>>                              [0, 2, 2, 4, 6, 8, 0],
    >>>                              [0, 4, 4, 4, 6, 8, 0],
    >>>                              [0, 6, 6, 6, 6, 8, 0],
    >>>                              [0, 8, 8, 8, 8, 8, 0],
    >>>                              [0, 0, 0, 0, 0, 0, 0]]),
    >>>                     dims = ["lat", "lon"],
    >>>                     attrs = dict(res = 1))
    >>> height, width = agg.shape
    >>> _lon = np.linspace(0, width - 1, width)
    >>> _lat = np.linspace(0, height - 1, height)
    >>> agg["lon"] = _lon
    >>> agg["lat"] = _lat

        Create Kernel
    >>> kernel = focal.circle_kernel(1, 1, 1)

        Create Convolution Data Array
    >>> print(convolution.convolve_2d(agg, kernel))
    [[ 0.  0.  4.  8.  0. 16.  0.]
     [ 0.  4.  8. 10. 18. 16. 16.]
     [ 4.  8. 14. 20. 24. 30. 16.]
     [ 8. 16. 20. 24. 30. 30. 16.]
     [12. 24. 30. 30. 34. 30. 16.]
     [16. 22. 30. 30. 30. 24. 16.]
     [ 0. 16. 16. 16. 16. 16.  0.]]
    """

    # numpy case
    if isinstance(data, np.ndarray):
        out = _convolve_2d_numpy(data, kernel)

    # cupy case
    elif has_cuda() and isinstance(data, cupy.ndarray):
        out = _convolve_2d_cupy(data, kernel)

    # dask + cupy case
    elif has_cuda() and isinstance(data, da.Array) and \
            type(data._meta).__module__.split('.')[0] == 'cupy':
        out = _convolve_2d_dask_cupy(data, kernel)

    # dask + numpy case
    elif isinstance(data, da.Array):
        out = _convolve_2d_dask_numpy(data, kernel)

    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(data)))

    return out
