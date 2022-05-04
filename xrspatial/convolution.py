import re
from functools import partial

import numpy as np
import xarray as xr
from numba import cuda, jit, prange

from xrspatial.utils import (ArrayTypeFunctionMapping, cuda_args, get_dataarray_resolution,
                             not_implemented_func)

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

    if len(splits) == 2:
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
    Supported units are: meter, kelometer, foot, and mile.
    Cellsize will be converted to meters.

    Parameters
    ----------
    raster : xarray.DataArray
        2D array of input values.

    Returns
    -------
    cellsize : tuple
        Tuple of (cellsize_x, cellsize_y).
    Where cellsize_x is the size of cells in x-direction,
    and cellsize_y is the size of cells in y-direction.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> h, w = 100, 200
        >>> data = np.ones((h, w))
        >>> from xrspatial.convolution import calc_cellsize
        >>> # cellsize that already specified as an attribute of input raster
        >>> raster_1 = xr.DataArray(data, attrs={'res': (0.5, 0.5)})
        >>> calc_cellsize(raster_1)
        (0.5, 0.5)
        >>> # if no unit specified, default to meters
        >>> raster_2 = xr.DataArray(data, dims=['y', 'x'])
        >>> raster_2['y'] = np.linspace(1, h, h)
        >>> raster_2['x'] = np.linspace(1, w, w)
        >>> calc_cellsize(raster_2)
        (1.0, 1.0)
        # convert cellsize to meters
        >>> raster_3 = xr.DataArray(
        ...     data, dims=['y', 'x'], attrs={'unit': 'km'})
        >>> raster_3['y'] = np.linspace(1, h, h)
        >>> raster_3['x'] = np.linspace(1, w, w)
        >>> calc_cellsize(raster_3)
        >>> (1000.0, 1000.0)
    """

    if 'unit' in raster.attrs:
        unit = raster.attrs['unit']
    else:
        unit = DEFAULT_UNIT

    cellsize_x, cellsize_y = get_dataarray_resolution(raster)
    cellsize_x = _to_meters(cellsize_x, unit)
    cellsize_y = _to_meters(cellsize_y, unit)

    # avoid negative cellsize in y
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

    Parameters
    ----------
    cellsize_x : int
        Cell size of output kernel in x-direction.
    cellsize_y : int
        Cell size of output kernel in y-direction.
    radius : int
        Radius of output kernel.

    Returns
    -------
    kernel : NumPy Array of float values
        2D array where values of 1 indicate the kernel.

    Examples
    --------
    .. sourcecode:: python

        >>> import xarray as xr
        >>> from xrspatial.convolution import circle_kernel
        >>> # Create Kernel
        >>> kernel = circle_kernel(1, 1, 3)
        >>> print(kernel)
        [[0. 0. 0. 1. 0. 0. 0.]
        [0. 1. 1. 1. 1. 1. 0.]
        [0. 1. 1. 1. 1. 1. 0.]
        [1. 1. 1. 1. 1. 1. 1.]
        [0. 1. 1. 1. 1. 1. 0.]
        [0. 1. 1. 1. 1. 1. 0.]
        [0. 0. 0. 1. 0. 0. 0.]]
        >>> kernel = circle_kernel(1, 2, 3)
        >>> print(kernel)
        [[0. 0. 0. 1. 0. 0. 0.]
         [1. 1. 1. 1. 1. 1. 1.]
         [0. 0. 0. 1. 0. 0. 0.]]
    """
    # validate radius, convert radius to meters
    r = _get_distance(str(radius))

    kernel_half_w = int(r / cellsize_x)
    kernel_half_h = int(r / cellsize_y)

    kernel = _ellipse_kernel(kernel_half_w, kernel_half_h)
    return kernel


def annulus_kernel(cellsize_x, cellsize_y, outer_radius, inner_radius):
    """
    Generates an annulus (ring-shaped) kernel of a given cellsize and radius.

    Parameters
    ----------
    cellsize_x : int
        Cell size of output kernel in x direction.
    cellsize_y : int
        Cell size of output kernel in y direction.
    outer_radius : int
        Outer ring radius of output kernel.
    inner_radius : int
        Inner circle radius of output kernel.

    Returns
    -------
    kernel : NumPy Array of float values.
        2D array of 0s and 1s where values of 1 indicate the kernel.

    Examples
    --------
    .. sourcecode:: python

        >>> import xarray as xr
        >>> from xrspatial.convolution import annulus_kernel
        >>> # Create Kernel
        >>> kernel = annulus_kernel(1, 1, 3, 1)
        >>> print(kernel)
        [[0., 0., 0., 1., 0., 0., 0.],
         [0., 1., 1., 1., 1., 1., 0.],
         [0., 1., 1., 0., 1., 1., 0.],
         [1., 1., 0., 0., 0., 1., 1.],
         [0., 1., 1., 0., 1., 1., 0.],
         [0., 1., 1., 1., 1., 1., 0.],
         [0., 0., 0., 1., 0., 0., 0.]]
        >>> kernel = annulus_kernel(1, 2, 5, 2)
        >>> print(kernel)
        [[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0.],
         [1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.],
         [0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])
    """
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


@jit(nopython=True, nogil=True)
def _convolve_2d_numpy(data, kernel):
    # apply kernel to data image.
    # TODO: handle nan
    data = data.astype(np.float32)
    nx = data.shape[0]
    ny = data.shape[1]
    nkx = kernel.shape[0]
    nky = kernel.shape[1]
    wkx = nkx // 2
    wky = nky // 2

    out = np.zeros(data.shape, dtype=np.float32)
    out[:] = np.nan
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
    data = data.astype(np.float32)
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
    # sum_{k, h} kernel[k, h] * data[i + k - delta_rows, j + h - delta_cols]
    # with k and h going through the whole kernel array:
    s = 0
    for k in range(kernel.shape[0]):
        for h in range(kernel.shape[1]):
            i_k = i + k - delta_rows
            j_h = j + h - delta_cols
            # (-4-) Check if (i_k, j_h) coordinates are inside the array:
            if (i_k >= 0) and (i_k < data_rows) and \
                    (j_h >= 0) and (j_h < data_cols):
                s += kernel[k, h] * data[i_k, j_h]
    out[i, j] = s


def _convolve_2d_cupy(data, kernel):
    data = data.astype(cupy.float32)
    out = cupy.empty(data.shape, dtype='f4')
    out[:, :] = cupy.nan
    griddim, blockdim = cuda_args(data.shape)
    _convolve_2d_cuda[griddim, blockdim](data, kernel, cupy.asarray(out))
    return out


def convolve_2d(data, kernel):
    mapper = ArrayTypeFunctionMapping(
        numpy_func=_convolve_2d_numpy,
        cupy_func=_convolve_2d_cupy,
        dask_func=_convolve_2d_dask_numpy,
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='convolution_2d() does not support dask with cupy backed xr.DataArray'  # noqa
        )
    )
    out = mapper(xr.DataArray(data))(data, kernel)
    return out


def convolution_2d(agg, kernel, name='convolution_2d'):
    """
    Calculates, for all inner cells of an array, the 2D convolution of
    each cell. Convolution is frequently used for image
    processing, such as smoothing, sharpening, and edge detection of
    images by eliminating spurious data or enhancing features in the
    data. Note that edges of output data array are filled with NaNs.

    Parameters
    ----------
    agg : xarray.DataArray
        2D array of values to processed. Can be NumPy backed, CuPybacked,
        or Dask with NumPy backed DataArray.
    kernel : array-like object
        Impulse kernel, determines area to apply impulse function for
        each cell.

    Returns
    -------
    convolve_agg : xarray.DataArray
        2D array representation of the impulse function.
        The backend array type is the same as of input.

    Examples
    --------
    convolution_2d() works with NumPy backed DataArray.
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.convolution import circle_kernel
        >>> kernel = circle_kernel(1, 1, 1)
        >>> kernel
        array([[0., 1., 0.],
               [1., 1., 1.],
               [0., 1., 0.]])
        >>> h, w = 4, 6
        >>> data = np.arange(h*w).reshape(h, w)
        >>> raster = xr.DataArray(data)
        >>> raster
        <xarray.DataArray (dim_0: 4, dim_1: 6)>
        array([[ 0,  1,  2,  3,  4,  5],
               [ 6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17],
               [18, 19, 20, 21, 22, 23]])
        Dimensions without coordinates: dim_0, dim_1
        >>> from xrspatial.convolution import convolution_2d
        >>> convolved_agg = convolution_2d(raster, kernel)
        >>> convolved_agg
        <xarray.DataArray 'convolution_2d' (dim_0: 4, dim_1: 6)>
        array([[nan, nan, nan, nan, nan, nan],
               [nan, 35., 40., 45., 50., nan],
               [nan, 65., 70., 75., 80., nan],
               [nan, nan, nan, nan, nan, nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1

    convolution_2d() works with Dask with NumPy backed DataArray.
    .. sourcecode:: python

        >>> from xrspatial.convolution import annulus_kernel
        >>> kernel = annulus_kernel(1, 1, 1.5, 0.5)
        >>> kernel
        array([[0., 1., 0.],
               [1., 0., 1.],
               [0., 1., 0.]])
        >>> import dask.array as da
        >>> data_da = da.from_array(np.ones((h, w)), chunks=(2, 2))
        >>> raster_da = xr.DataArray(data_da, name='raster_da')
        >>> raster_da
        <xarray.DataArray 'raster_da' (dim_0: 4, dim_1: 6)>
        dask.array<array, shape=(4, 6), dtype=float64, chunksize=(2, 2), chunktype=numpy.ndarray>  # noqa
        Dimensions without coordinates: dim_0, dim_1
        >>> convolved_agg = convolution_2d(raster_da, kernel)
        >>> convolved_agg
        <xarray.DataArray 'convolution_2d' (dim_0: 4, dim_1: 6)>
        dask.array<_trim, shape=(4, 6), dtype=float64, chunksize=(2, 2), chunktype=numpy.ndarray>  # noqa
        Dimensions without coordinates: dim_0, dim_1
        >>> convolved_agg.compute()
        <xarray.DataArray 'convolution_2d' (dim_0: 4, dim_1: 6)>
        array([[nan, nan, nan, nan, nan, nan],
               [nan,  4.,  4.,  4.,  4., nan],
               [nan,  4.,  4.,  4.,  4., nan],
               [nan, nan, nan, nan, nan, nan]], dtype=float32)

    convolution_2d() works with CuPy backed DataArray.
    .. sourcecode:: python

        >>> from xrspatial.convolution import custom_kernel
        >>> kernel = custom_kernel(np.array([
        ...    [1, 0, 0],
        ...    [1, 1, 0],
        ...    [1, 0, 0]
        ... ]))
        >>> import cupy
        >>> data_cupy = cupy.arange(0, w * h * 2, 2).reshape(h, w)
        >>> raster_cupy = xr.DataArray(data_cupy, name='raster_cupy')
        >>> print(raster_cupy)
        <xarray.DataArray 'raster_cupy' (dim_0: 4, dim_1: 6)>
        array([[ 0,  2,  4,  6,  8, 10],
               [12, 14, 16, 18, 20, 22],
               [24, 26, 28, 30, 32, 34],
               [36, 38, 40, 42, 44, 46]])
        Dimensions without coordinates: dim_0, dim_1
        >>> convolved_agg = convolution_2d(raster_cupy, kernel)
        >>> type(convolved_agg.data)
        <class 'cupy.core.core.ndarray'>
        >>> convolved_agg
        <xarray.DataArray 'convolution_2d' (dim_0: 4, dim_1: 6)>
        array([[ nan,  nan,  nan,  nan,  nan,  nan],
               [ nan,  50.,  58.,  66.,  74.,  nan],
               [ nan,  98., 106., 114., 122.,  nan],
               [ nan,  nan,  nan,  nan,  nan,  nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
    """

    # wrapper of convolve_2d
    out = convolve_2d(agg.data, kernel)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
