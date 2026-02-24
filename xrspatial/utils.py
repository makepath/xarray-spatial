from __future__ import annotations

from math import ceil
import warnings

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import xarray as xr
from datashader.colors import rgb
from numba import cuda, jit

try:
    import cupy
except ImportError:
    cupy = None


try:
    import dask.array as da
except ImportError:
    da = None


try:
    import dask.dataframe as dd
except ImportError:
    dd = None


ngjit = jit(nopython=True, nogil=True)


# ---------- Boundary mode utilities ----------
VALID_BOUNDARY_MODES = ('nan', 'nearest', 'reflect', 'wrap')


def _validate_boundary(boundary):
    """Raise ValueError if *boundary* is not a recognised mode."""
    if boundary not in VALID_BOUNDARY_MODES:
        raise ValueError(
            f"boundary must be one of {VALID_BOUNDARY_MODES}, "
            f"got {boundary!r}"
        )


def _boundary_to_dask(boundary, is_cupy=False):
    """Convert a boundary mode string to the value expected by
    ``dask.array.map_overlap``'s *boundary* parameter."""
    if boundary == 'nan':
        if is_cupy:
            import cupy as _cp
            return _cp.nan
        return np.nan
    _mode_map = {
        'nearest': 'nearest',
        'reflect': 'reflect',
        'wrap': 'periodic',
    }
    return _mode_map[boundary]


def _pad_array(data, depth, boundary):
    """Pad a 2-D numpy or cupy array according to *boundary* mode.

    Parameters
    ----------
    data : array-like
        2-D array to pad.
    depth : int or tuple of int
        Number of cells to pad on each side.  An int pads all axes
        equally; a tuple ``(d0, d1)`` pads each axis independently.
    boundary : str
        One of ``'nearest'``, ``'reflect'``, ``'wrap'``.
        ``'nan'`` should be handled before calling this function.
    """
    # numpy.pad 'symmetric' matches dask map_overlap 'reflect'
    # (both include the edge element in the reflection)
    _np_mode_map = {
        'nearest': 'edge',
        'reflect': 'symmetric',
        'wrap': 'wrap',
    }
    mode = _np_mode_map[boundary]
    if isinstance(depth, int):
        pad_width = ((depth, depth), (depth, depth))
    else:
        pad_width = tuple((d, d) for d in depth)
    if is_cupy_array(data):
        return cupy.pad(data, pad_width, mode=mode)
    return np.pad(data, pad_width, mode=mode)


def has_cuda_and_cupy():
    return _has_cuda() and _has_cupy()


def _has_cupy():
    return cupy is not None


def is_cupy_array(arr):
    return _has_cupy() and isinstance(arr, cupy.ndarray)


def has_dask_array():
    return da is not None


def has_dask_dataframe():
    return dd is not None


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
        elif has_dask_array() and isinstance(arr.data, da.Array):
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
    if has_dask_array() and isinstance(first_array.data, da.Array):
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
        If not provided, assume ydim is `raturns
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


def _infer_coord_unit_type(coord: xr.DataArray, cellsize: float) -> str:
    """
    Heuristic to classify a spatial coordinate axis as:
    - 'degrees'
    - 'linear'   (meters/feet/etc)
    - 'unknown'

    Parameters
    ----------
    coord : xr.DataArray
        1D coordinate variable (x or y).
    cellsize : float
        Mean spacing along this coordinate.

    Returns
    -------
    str
    """
    units = str(coord.attrs.get("units", "")).lower()

    # 1) Explicit units, if present
    if "degree" in units or units in ("deg", "degrees"):
        return "degrees"
    if units in ("m", "meter", "metre", "meters", "metres",
                 "km", "kilometer", "kilometre", "kilometers", "kilometres",
                 "ft", "foot", "feet"):
        return "linear"

    # 2) Numeric heuristics (very conservative)
    vals = coord.values
    if vals.size < 2 or not np.issubdtype(vals.dtype, np.number):
        return "unknown"

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    span = abs(vmax - vmin)
    dx = abs(float(cellsize))

    # Typical global geographic axes: span <= 360, spacing ~1e-5–0.5 deg
    if -360.0 <= vmin <= 360.0 and -360.0 <= vmax <= 360.0:
        if 1e-5 <= dx <= 0.5:
            return "degrees"

    # Typical projected axes in meters: span >> 1, spacing > ~0.1
    # (e.g. UTM / national grids)
    if span > 1000.0 and dx >= 0.1:
        return "linear"

    return "unknown"


def _infer_vertical_unit_type(agg):
    units = str(agg.attrs.get("units", "")).lower()

    # Cheap / reliable first
    if any(k in units for k in ("degree", "deg")) or "rad" in units:
        return "angle"
    if units in ("m", "meter", "metre", "meters", "metres",
                 "km", "kilometer", "kilometre", "kilometers", "kilometres",
                 "ft", "foot", "feet"):
        return "elevation"

    # Numeric fallback: sample only (never full compute)
    data = agg.data
    try:
        vmin, vmax = _sample_windows_min_max(data, max_window_elems=65536, windows=5)
    except Exception:
        return "unknown"

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return "unknown"

    span = vmax - vmin

    # Elevation-ish heuristic
    if 10.0 <= span <= 20000.0 and vmin > -500.0:
        return "elevation"

    # Angle-ish heuristic
    if -360.0 <= vmin <= 360.0 and -360.0 <= vmax <= 360.0 and span <= 720.0:
        return "angle"

    return "unknown"


def warn_if_unit_mismatch(agg: xr.DataArray) -> None:
    """
    Heuristic check for horizontal vs vertical unit mismatch.

    Intended to catch the common case of:
    - coordinates in degrees (lon/lat)
    - elevation values in meters/feet

    Emits a UserWarning if a likely mismatch is detected.
    """
    try:
        cellsize_x, cellsize_y = get_dataarray_resolution(agg)
    except Exception:
        # If we can't even get a resolution, we also can't say much
        return

    # pick "x" and "y" coords in a generic way:
    #   - typically dims are ('y', 'x') or ('lat', 'lon')
    #   - fall back to last two dims
    if len(agg.dims) < 2:
        return

    dim_y, dim_x = agg.dims[-2], agg.dims[-1]
    coord_x = agg.coords.get(dim_x, None)
    coord_y = agg.coords.get(dim_y, None)

    if coord_x is None or coord_y is None:
        # Can't infer spatial types without coords
        return

    horiz_x = _infer_coord_unit_type(coord_x, cellsize_x)
    horiz_y = _infer_coord_unit_type(coord_y, cellsize_y)
    vert = _infer_vertical_unit_type(agg)

    horiz_types = {horiz_x, horiz_y} - {"unknown"}

    # Only act if we have some signal about horizontal AND vertical
    if not horiz_types or vert == "unknown":
        return

    # If any axis looks like degrees and vertical looks like elevation,
    # it's almost certainly "lat/lon degrees + meter elevations"
    if "degrees" in horiz_types and vert == "elevation":
        warnings.warn(
            "xrspatial: input DataArray appears to have coordinates in degrees "
            "but elevation values in a linear unit (e.g. meters/feet). "
            "Slope/aspect operations expect horizontal distances in the same "
            "units as vertical. Consider reprojecting to a projected CRS with "
            "meter-based coordinates before calling `slope`.",
            UserWarning,
        )


# ---------- Z-unit conversion for geodesic methods ----------
Z_UNITS = {
    'meter': 1.0, 'meters': 1.0, 'm': 1.0,
    'foot': 0.3048, 'feet': 0.3048, 'ft': 0.3048,
    'kilometer': 1000.0, 'kilometers': 1000.0, 'km': 1000.0,
    'mile': 1609.344, 'miles': 1609.344, 'mi': 1609.344,
}


# ---------- Lat/lon coordinate extraction for geodesic methods ----------
# Known dimension / coordinate names (lower-cased for matching)
_LAT_NAMES = {'lat', 'latitude', 'y'}
_LON_NAMES = {'lon', 'longitude', 'x'}


def _extract_latlon_coords(agg: xr.DataArray):
    """
    Extract 2-D latitude and longitude arrays from a DataArray.

    Supports:
    - 1-D coordinates on the last two dims (regular geographic grid).
    - 2-D coordinates that vary per cell (curvilinear grid).

    Returns
    -------
    lat_2d, lon_2d : numpy.ndarray
        Always 2-D float64 numpy arrays of shape ``(H, W)``.

    Raises
    ------
    ValueError
        If coordinates are missing, non-numeric, or outside geographic
        ranges (lat not in [-90, 90], lon not in [-180, 360]).
    """
    if agg.ndim < 2:
        raise ValueError(
            "geodesic method requires a 2-D DataArray, "
            f"got {agg.ndim}-D"
        )

    dim_y, dim_x = agg.dims[-2], agg.dims[-1]

    # --- locate lat coordinate ---
    lat_coord = _find_coord(agg, dim_y, _LAT_NAMES, 'latitude')
    # --- locate lon coordinate ---
    lon_coord = _find_coord(agg, dim_x, _LON_NAMES, 'longitude')

    lat_vals = np.asarray(lat_coord.values, dtype=np.float64)
    lon_vals = np.asarray(lon_coord.values, dtype=np.float64)

    # Build 2-D arrays
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        # Regular grid: broadcast to 2-D
        lat_2d = np.broadcast_to(lat_vals[:, np.newaxis],
                                 (agg.sizes[dim_y], agg.sizes[dim_x])).copy()
        lon_2d = np.broadcast_to(lon_vals[np.newaxis, :],
                                 (agg.sizes[dim_y], agg.sizes[dim_x])).copy()
    elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
        lat_2d = lat_vals
        lon_2d = lon_vals
    else:
        raise ValueError(
            f"lat/lon coordinates must be both 1-D or both 2-D, "
            f"got lat={lat_vals.ndim}-D and lon={lon_vals.ndim}-D"
        )

    # --- validate ranges ---
    _validate_geographic_range(lat_2d, lon_2d)

    return lat_2d, lon_2d


def _find_coord(agg, dim_name, known_names, label):
    """Find a coordinate matching *dim_name* or one of *known_names*."""
    # 1) Try the dimension name directly
    if dim_name in agg.coords:
        coord = agg.coords[dim_name]
        if np.issubdtype(coord.dtype, np.number):
            return coord

    # 2) Scan all coords for a known name
    for name in agg.coords:
        if str(name).lower() in known_names:
            coord = agg.coords[name]
            if np.issubdtype(coord.dtype, np.number):
                return coord

    raise ValueError(
        f"geodesic method requires {label} coordinates on the DataArray. "
        f"No numeric coordinate found for dim '{dim_name}' or any of "
        f"{sorted(known_names)}."
    )


def _validate_geographic_range(lat_2d, lon_2d):
    """Raise ValueError if lat/lon values look non-geographic."""
    lat_min = np.nanmin(lat_2d)
    lat_max = np.nanmax(lat_2d)
    lon_min = np.nanmin(lon_2d)
    lon_max = np.nanmax(lon_2d)

    if lat_min < -90 or lat_max > 90:
        raise ValueError(
            f"Latitude values must be in [-90, 90], "
            f"got [{lat_min}, {lat_max}]. "
            f"Are your coordinates in a projected CRS?"
        )
    if lon_min < -180 or lon_max > 360:
        raise ValueError(
            f"Longitude values must be in [-180, 360], "
            f"got [{lon_min}, {lon_max}]. "
            f"Are your coordinates in a projected CRS?"
        )
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span > 180 or lon_span > 360:
        raise ValueError(
            f"Coordinate span too large for geographic coordinates "
            f"(lat span={lat_span}, lon span={lon_span}). "
            f"Are your coordinates in a projected CRS?"
        )


def _to_float_scalar(x) -> float:
    """Convert numpy/cupy scalar or 0-d array to python float safely."""
    if cupy is not None:
        # cupy.ndarray scalar
        if isinstance(x, cupy.ndarray):
            return float(cupy.asnumpy(x).item())
        # cupy scalar type
        if x.__class__.__module__.startswith("cupy") and hasattr(x, "item"):
            return float(x.item())

    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _sample_windows_min_max(
    data,
    *,
    max_window_elems: int = 65536,   # e.g. 256x256
    windows: int = 5,                # corners + center default
) -> tuple[float, float]:
    """
    Estimate (nanmin, nanmax) from a small sample of windows.

    Works for numpy, cupy, dask+numpy, dask+cupy. Only computes on the sampled
    windows, not the full array.
    """
    # Normalize to last-2D sampling (y,x). For higher dims, sample first index.
    if hasattr(data, "ndim") and data.ndim >= 3:
        prefix = (0,) * (data.ndim - 2)
    else:
        prefix = ()

    # Determine y/x sizes
    shape = data.shape
    ny, nx = shape[-2], shape[-1]

    if ny == 0 or nx == 0:
        return np.nan, np.nan

    # Choose a square-ish window size bounded by array shape
    w = int(np.sqrt(max_window_elems))
    w = max(1, min(w, ny, nx))

    # Define window anchor positions: (top-left), (top-right), (bottom-left), (bottom-right), (center)
    anchors = [
        (0, 0),
        (0, max(0, nx - w)),
        (max(0, ny - w), 0),
        (max(0, ny - w), max(0, nx - w)),
    ]
    if windows >= 5:
        anchors.append((max(0, ny // 2 - w // 2), max(0, nx // 2 - w // 2)))

    # If windows > 5, sprinkle additional evenly-spaced anchors (optional)
    if windows > 5:
        extra = windows - 5
        ys = np.linspace(0, max(0, ny - w), extra + 2, dtype=int)[1:-1]
        xs = np.linspace(0, max(0, nx - w), extra + 2, dtype=int)[1:-1]
        for y0, x0 in zip(ys, xs):
            anchors.append((int(y0), int(x0)))

    # Reduce min/max across sampled windows
    mins = []
    maxs = []

    for y0, x0 in anchors:
        sl = prefix + (slice(y0, y0 + w), slice(x0, x0 + w))
        win = data[sl]

        if da is not None and isinstance(win, da.Array):
            # Compute scalars only on this window
            mins.append(da.nanmin(win))
            maxs.append(da.nanmax(win))
        elif cupy is not None and isinstance(win, cupy.ndarray):
            mins.append(cupy.nanmin(win))
            maxs.append(cupy.nanmax(win))
        else:
            mins.append(np.nanmin(win))
            maxs.append(np.nanmax(win))

    # Finalize: if dask, compute the scalar graph now (still tiny)
    if da is not None and any(isinstance(m, da.Array) for m in mins):
        mn = da.nanmin(da.stack(mins)).compute()
        mx = da.nanmax(da.stack(maxs)).compute()
        return _to_float_scalar(mn), _to_float_scalar(mx)

    # If cupy scalars, convert safely
    if cupy is not None and (any(isinstance(m, cupy.ndarray) for m in mins) or
                             any(getattr(m.__class__, "__module__", "").startswith("cupy") for m in mins)):
        mn = mins[0]
        mx = maxs[0]
        # reduce on device
        for m in mins[1:]:
            mn = cupy.minimum(mn, m)
        for m in maxs[1:]:
            mx = cupy.maximum(mx, m)
        return _to_float_scalar(mn), _to_float_scalar(mx)

    # numpy scalars
    return float(np.nanmin(np.array(mins, dtype=float))), float(np.nanmax(np.array(maxs, dtype=float)))
