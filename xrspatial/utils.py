from math import ceil

import dask.array as da
import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import xarray as xr
from datashader.colors import rgb
from numba import cuda, jit

try:
    import cupy

    if cupy.result_type is np.result_type:
        # hack until cupy release of https://github.com/cupy/cupy/pull/2249
        # Without this, cupy.histogram raises an error that cupy.result_type
        # is not defined.
        cupy.result_type = lambda *args: np.result_type(
            *[
                arg.dtype if isinstance(arg, cupy.ndarray)
                else arg for arg in args
            ]
        )
except ImportError:
    cupy = None

ngjit = jit(nopython=True, nogil=True)


def has_cuda_and_cupy():
    return _has_cuda() and _has_cupy()


def _has_cupy():
    return cupy is not None


def is_cupy_array(arr):
    return _has_cupy() and isinstance(arr, cupy.ndarray)


def _has_cuda():
    """Check for supported CUDA device. If none found, return False"""
    local_cuda = False
    try:
        cuda.cudadrv.devices.gpus.current
        local_cuda = True
    except cuda.cudadrv.error.CudaSupportError:
        local_cuda = False

    return local_cuda


def cuda_args(shape):
    """
    Compute the blocks-per-grid and threads-per-block parameters for
    use when invoking cuda kernels

    Parameters
    ----------
    shape: int or tuple of ints
        The shape of the input array that the kernel will parallelize
        over.

    Returns
    -------
    bpg, tpb : tuple
        Tuple of (blocks_per_grid, threads_per_block).
    """
    if isinstance(shape, int):
        shape = (shape,)

    max_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    # Note: We divide max_threads by 2.0 to leave room for the registers
    threads_per_block = int(ceil(max_threads / 2.0) ** (1.0 / len(shape)))
    tpb = (threads_per_block,) * len(shape)
    bpg = tuple(int(ceil(d / threads_per_block)) for d in shape)
    return bpg, tpb


def calc_cuda_dims(shape):
    threadsperblock = (32, 32)
    blockspergrid = (
        (shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0],
        (shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]
    )
    return blockspergrid, threadsperblock


def is_cupy_backed(agg: xr.DataArray):
    try:
        return type(agg.data._meta).__module__.split(".")[0] == "cupy"
    except AttributeError:
        return False


def is_dask_cupy(agg: xr.DataArray):
    return isinstance(agg.data, da.Array) and is_cupy_backed(agg)


def not_implemented_func(agg, *args, messages='Not yet implemented.'):
    raise NotImplementedError(messages)


class ArrayTypeFunctionMapping(object):
    def __init__(self, numpy_func, cupy_func, dask_func, dask_cupy_func):
        self.numpy_func = numpy_func
        self.cupy_func = cupy_func
        self.dask_func = dask_func
        self.dask_cupy_func = dask_cupy_func

    def __call__(self, arr):

        # numpy case
        if isinstance(arr.data, np.ndarray):
            return self.numpy_func

        # cupy case
        elif has_cuda_and_cupy() and is_cupy_array(arr.data):
            return self.cupy_func

        # dask + cupy case
        elif has_cuda_and_cupy() and is_dask_cupy(arr):
            return self.dask_cupy_func

        # dask + numpy case
        elif isinstance(arr.data, da.Array):
            return self.dask_func

        else:
            raise TypeError("Unsupported Array Type: {}".format(type(arr)))


def validate_arrays(*arrays):
    if len(arrays) < 2:
        raise ValueError(
            "validate_arrays() input must contain 2 or more arrays"
        )

    first_array = arrays[0]
    for i in range(1, len(arrays)):

        if not first_array.data.shape == arrays[i].data.shape:
            raise ValueError("input arrays must have equal shapes")

        if not isinstance(first_array.data, type(arrays[i].data)):
            raise ValueError("input arrays must have same type")

    # ensure dask chunksizes of all arrays are the same
    if isinstance(first_array.data, da.Array):
        for i in range(1, len(arrays)):
            if first_array.chunks != arrays[i].chunks:
                arrays[i].data = arrays[i].data.rechunk(first_array.chunks)


def get_xy_range(raster, xdim=None, ydim=None):
    """
    Compute xrange and yrange for input `raster`

    Parameters
    ----------
    raster: xarray.DataArray
    xdim: str, default = None
        Name of the x coordinate dimension in input `raster`.
        If not provided, assume xdim is `raster.dims[-1]`
    ydim: str, default = None
        Name of the y coordinate dimension in input `raster`
        If not provided, assume ydim is `raster.dims[-2]`

    Returns
    ----------
    xrange, yrange
        Tuple of tuples: (x, y-range).
        xrange: tuple of (xmin, xmax)
        yrange: tuple of (ymin, ymax)
    """

    if ydim is None:
        ydim = raster.dims[-2]
    if xdim is None:
        xdim = raster.dims[-1]

    xmin = raster[xdim].min().item()
    xmax = raster[xdim].max().item()
    ymin = raster[ydim].min().item()
    ymax = raster[ydim].max().item()

    xrange = (xmin, xmax)
    yrange = (ymin, ymax)

    return xrange, yrange


def calc_res(raster, xdim=None, ydim=None):
    """
    Calculate the resolution of xarray.DataArray raster and return it
    as thetwo-tuple (xres, yres).

    Parameters
    ----------
    raster: xr.DataArray
        Input raster.
    xdim: str, default = None
        Name of the x coordinate dimension in input `raster`.
        If not provided, assume xdim is `raster.dims[-1]`
    ydim: str, default = None
        Name of the y coordinate dimension in input `raster`
        If not provided, assume ydim is `raster.dims[-2]`

    Returns
    -------
    xres, yres: tuple
        Tuple of (x-resolution, y-resolution).
    """

    h, w = raster.shape[-2:]
    xrange, yrange = get_xy_range(raster, xdim, ydim)
    xres = (xrange[-1] - xrange[0]) / (w - 1)
    yres = (yrange[-1] - yrange[0]) / (h - 1)
    return xres, yres


def get_dataarray_resolution(
    agg: xr.DataArray,
    xdim: str = None,
    ydim: str = None,
):
    """
    Calculate resolution of xarray.DataArray.

    Parameters
    ----------
    agg: xarray.DataArray
        Input raster.
    xdim: str, default = None
        Name of the x coordinate dimension in input `raster`.
        If not provided, assume xdim is `raster.dims[-1]`
    ydim: str, default = None
        Name of the y coordinate dimension in input `raster`
        If not provided, assume ydim is `raster.dims[-2]`

    Returns
    -------
    cellsize_x, cellsize_y: tuple
        Tuple of (x cell size, y cell size).
    """

    # get cellsize out from 'res' attribute
    try:
        cellsize = agg.attrs.get("res")
        if (
            isinstance(cellsize, (tuple, np.ndarray, list))
            and len(cellsize) == 2
            and isinstance(cellsize[0], (int, float))
            and isinstance(cellsize[1], (int, float))
        ):
            cellsize_x, cellsize_y = cellsize
        elif isinstance(cellsize, (int, float)):
            cellsize_x = cellsize
            cellsize_y = cellsize
        else:
            cellsize_x, cellsize_y = calc_res(agg, xdim, ydim)

    except Exception:
        cellsize_x, cellsize_y = calc_res(agg, xdim, ydim)

    return cellsize_x, cellsize_y


def lnglat_to_meters(longitude, latitude):
    """
    Projects the given (longitude, latitude) values into Web Mercator
    coordinates (meters East of Greenwich and meters North of the
    Equator).

    Longitude and latitude can be provided as scalars, Pandas columns,
    or Numpy arrays, and will be returned in the same form.  Lists
    or tuples will be converted to Numpy arrays.

    Parameters
    ----------
    latitude: float
        Input latitude.
    longitude: float
        Input longitude.

    Returns
    -------
    easting, northing : tuple
        Tuple of (easting, northing).

    Examples
    --------
    .. sourcecode:: python

        >>> easting, northing = lnglat_to_meters(-40.71,74)
        >>> easting, northing = lnglat_to_meters(np.array([-74]),
        >>>                                      np.array([40.71]))
        >>> df = pandas.DataFrame(dict(longitude=np.array([-74]),
        >>>                            latitude=np.array([40.71])))
        >>> df.loc[:, 'longitude'], df.loc[:, 'latitude'] = lnglat_to_meters(
        >>>     df.longitude, df.latitude)
    """
    if isinstance(longitude, (list, tuple)):
        longitude = np.array(longitude)
    if isinstance(latitude, (list, tuple)):
        latitude = np.array(latitude)

    origin_shift = np.pi * 6378137
    easting = longitude * origin_shift / 180.0
    northing = np.log(
        np.tan((90 + latitude) * np.pi / 360.0)
        ) * origin_shift / np.pi
    return (easting, northing)


def height_implied_by_aspect_ratio(W, X, Y):
    """
    Utility function for calculating height (in pixels) which is implied
    by a width, x-range, and y-range. Simple ratios are used to maintain
    aspect ratio.

    Parameters
    ----------
    W: int
        Width in pixel.
    X: tuple
        X-range in data units.
    Y: tuple
        X-range in data units.

    Returns
    -------
    height : int
        height in pixels

    Examples
    --------
    .. sourcecode:: python

        >>> plot_width = 1000
        >>> x_range = (0,35
        >>> y_range = (0, 70)
        >>> plot_height = height_implied_by_aspect_ratio(
                plot_width,
                x_range,
                y_range,
            )
    """
    return int((W * (Y[1] - Y[0])) / (X[1] - X[0]))


def bands_to_img(r, g, b, nodata=1):
    h, w = r.shape
    data = np.zeros((h, w, 4), dtype=np.uint8)
    data[:, :, 0] = (r).astype(np.uint8)
    data[:, :, 1] = (g).astype(np.uint8)
    data[:, :, 2] = (b).astype(np.uint8)
    a = np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)
    data[:, :, 3] = a.astype(np.uint8)
    return tf.Image.fromarray(data, "RGBA")


def canvas_like(
    raster,
    width=512,
    height=None,
    x_range=None,
    y_range=None,
    **kwargs
):

    """
    Resample a xarray.DataArray by canvas width and bounds.
    Height of the resampled raster is implied from the canvas width
    using aspect ratio of original raster.

    This function uses of datashader.Canvas.raster internally.
    Most of the docstrings are copied from Datashader.

    Handles 2D or 3D xarray.DataArray, assuming that the last two
    array dimensions are the y-axis and x-axis that are to be
    resampled. If a 3D array is supplied a layer may be specified
    to resample to select the layer along the first dimension to
    resample.

    Parameters
    ----------
    raster : xarray.DataArray
        2D or 3D labeled data array.
    layer : float, optional
        For a 3D array, value along the z dimension.
    width : int, default=512
        Width of the output aggregate in pixels.
    height : int, default=None
        Height of the output aggregate in pixels.
        If not provided, height will be implied from `width`
        using aspect ratio of input raster.
    x_range : tuple of int, optional
        A tuple representing the bounds inclusive space ``[min, max]``
        along the x-axis.
    y_range : tuple of int, optional
        A tuple representing the bounds inclusive space ``[min, max]``
        along the y-axis.

    References
    ----------
        - https://datashader.org/_modules/datashader/core.html#Canvas
    """

    # get ranges
    if x_range is None:
        x_range = (
            raster.coords["x"].min().item(),
            raster.coords["x"].max().item()
        )
    if y_range is None:
        y_range = (
            raster.coords["y"].min().item(),
            raster.coords["y"].max().item()
        )

    if height is None:
        # set width and height
        height = height_implied_by_aspect_ratio(width, x_range, y_range)

    cvs = ds.Canvas(
        plot_width=width, plot_height=height, x_range=x_range, y_range=y_range
    )
    out = cvs.raster(raster, **kwargs)

    return out


def color_values(agg, color_key, alpha=255):
    def _convert_color(c):
        r, g, b = rgb(c)
        return np.array([r, g, b, alpha]).astype(np.uint8).view(np.uint32)[0]

    _converted_colors = {k: _convert_color(v) for k, v in color_key.items()}
    f = np.vectorize(lambda v: _converted_colors.get(v, 0))
    return tf.Image(f(agg.data))
