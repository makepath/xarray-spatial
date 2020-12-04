from math import ceil
import numba as nb
import numpy as np
import xarray as xr

from numba import cuda

try:
    import cupy
    if cupy.result_type is np.result_type:
        # Workaround until cupy release of https://github.com/cupy/cupy/pull/2249
        # Without this, cupy.histogram raises an error that cupy.result_type
        # is not defined.
        cupy.result_type = lambda *args: np.result_type(
            *[arg.dtype if isinstance(arg, cupy.ndarray) else arg
              for arg in args]
        )
except:
    cupy = None


ngjit = nb.jit(nopython=True, nogil=True)


def has_cuda():
    """Check for supported CUDA device. If none found, return False"""
    local_cuda = False
    try:
        cuda.cudadrv.devices.gpus.current
        local_cuda = True
    except cuda.cudadrv.error.CudaSupportError:
        local_cuda = False

    return local_cuda


def doesnt_have_cuda():
    return not has_cuda()


def cuda_args(shape):
    """
    Compute the blocks-per-grid and threads-per-block parameters for use when
    invoking cuda kernels
    Parameters
    ----------
    shape: int or tuple of ints
        The shape of the input array that the kernel will parallelize over
    Returns
    -------
    tuple
        Tuple of (blocks_per_grid, threads_per_block)
    """
    if isinstance(shape, int):
        shape = (shape,)

    max_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    # Note: We divide max_threads by 2.0 to leave room for the registers
    threads_per_block = int(ceil(max_threads / 2.0) ** (1.0 / len(shape)))
    tpb = (threads_per_block,) * len(shape)
    bpg = tuple(int(ceil(d / threads_per_block)) for d in shape)
    return bpg, tpb


def is_cupy_backed(agg: xr.DataArray):
    return type(agg.data._meta).__module__.split('.')[0] == 'cupy'


def calc_res(raster):
    """Calculate the resolution of xarray.DataArray raster and return it as the
    two-tuple (xres, yres).

    Notes
    -----

    Sourced from datashader.utils
    """
    h, w = raster.shape[-2:]
    ydim, xdim = raster.dims[-2:]
    xcoords = raster[xdim].values
    ycoords = raster[ydim].values
    xres = (xcoords[-1] - xcoords[0]) / (w - 1)
    yres = (ycoords[0] - ycoords[-1]) / (h - 1)
    return xres, yres


def get_dataarray_resolution(agg: xr.DataArray):

    if not agg.attrs.get('res'):
        raise ValueError('input xarray must have `res` attr.')

    # get cellsize out from 'res' attribute
    cellsize = agg.attrs.get('res')
    if isinstance(cellsize, tuple) and len(cellsize) == 2 \
            and isinstance(cellsize[0], (int, float)) \
            and isinstance(cellsize[1], (int, float)):
        cellsize_x, cellsize_y = cellsize
    elif isinstance(cellsize, (int, float)):
        cellsize_x = cellsize
        cellsize_y = cellsize
    else:
        cellsize_x, cellsize_y = calc_res(agg)
    
    return cellsize_x, cellsize_y


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
