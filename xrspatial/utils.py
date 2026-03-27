from __future__ import annotations

from math import ceil
import warnings

import numpy as np
import xarray as xr
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


def _validate_raster(
    agg,
    *,
    func_name: str,
    name: str = 'raster',
    ndim: int | tuple[int, ...] | None = 2,
    numeric: bool = True,
    integer_only: bool = False,
):
    """Validate that *agg* is an xarray.DataArray with expected properties.

    Parameters
    ----------
    agg : object
        Value to validate.
    func_name : str
        Name of the calling function (for error messages).
    name : str
        Parameter name (for error messages).
    ndim : int, tuple of int, or None
        Allowed number of dimensions.  ``None`` skips the check.
    numeric : bool
        If True, require a numeric dtype (int or float).
    integer_only : bool
        If True, require an integer dtype specifically.

    Raises
    ------
    TypeError
        If *agg* is not an ``xr.DataArray``.
    ValueError
        If the dimensionality or dtype is wrong.
    """
    if not isinstance(agg, xr.DataArray):
        raise TypeError(
            f"{func_name}(): `{name}` must be an xarray.DataArray, "
            f"got {type(agg).__module__}.{type(agg).__qualname__}"
        )

    if ndim is not None:
        allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
        if agg.ndim not in allowed:
            expected = 'or '.join(f'{d}D' for d in allowed)
            raise ValueError(
                f"{func_name}(): `{name}` must be {expected}, "
                f"got {agg.ndim}D"
            )

    if numeric:
        if integer_only:
            if not np.issubdtype(agg.dtype, np.integer):
                raise ValueError(
                    f"{func_name}(): `{name}` must have an integer dtype, "
                    f"got {agg.dtype}"
                )
        else:
            if not np.issubdtype(agg.dtype, np.number):
                raise ValueError(
                    f"{func_name}(): `{name}` must have a numeric dtype "
                    f"(integer or float), got {agg.dtype}"
                )


def _validate_scalar(
    value,
    *,
    func_name: str,
    name: str,
    dtype: type | tuple = (int, float),
    min_val=None,
    max_val=None,
    min_exclusive: bool = False,
):
    """Validate that *value* is a scalar of the expected type and range.

    Parameters
    ----------
    value : object
        Value to validate.
    func_name : str
        Name of the calling function (for error messages).
    name : str
        Parameter name (for error messages).
    dtype : type or tuple of types
        Allowed Python types (checked with ``isinstance``).
    min_val, max_val : numeric or None
        Inclusive bounds (or exclusive lower bound if *min_exclusive*).
    min_exclusive : bool
        If True, the lower bound is exclusive (``>`` instead of ``>=``).

    Raises
    ------
    TypeError
        If *value* is not an instance of *dtype*.
    ValueError
        If *value* is outside the allowed range.
    """
    # Expand dtype to also accept numpy scalar equivalents so that
    # e.g. np.int64(5) passes a check for dtype=int.
    _dtype = dtype if isinstance(dtype, tuple) else (dtype,)
    _expanded = list(_dtype)
    if int in _dtype:
        _expanded.append(np.integer)
    if float in _dtype:
        _expanded.append(np.floating)
    _expanded = tuple(_expanded)

    if not isinstance(value, _expanded):
        expected = dtype.__name__ if isinstance(dtype, type) else \
            ' or '.join(t.__name__ for t in _dtype)
        raise TypeError(
            f"{func_name}(): `{name}` must be {expected}, "
            f"got {type(value).__name__}"
        )

    if min_val is not None:
        if min_exclusive:
            if value <= min_val:
                raise ValueError(
                    f"{func_name}(): `{name}` must be > {min_val}, "
                    f"got {value}"
                )
        else:
            if value < min_val:
                raise ValueError(
                    f"{func_name}(): `{name}` must be >= {min_val}, "
                    f"got {value}"
                )

    if max_val is not None:
        if value > max_val:
            raise ValueError(
                f"{func_name}(): `{name}` must be <= {max_val}, "
                f"got {value}"
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
    from PIL import Image

    h, w = r.shape
    data = np.zeros((h, w, 4), dtype=np.uint8)
    data[:, :, 0] = (r).astype(np.uint8)
    data[:, :, 1] = (g).astype(np.uint8)
    data[:, :, 2] = (b).astype(np.uint8)
    a = np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)
    data[:, :, 3] = a.astype(np.uint8)
    return Image.fromarray(data, "RGBA")


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

    try:
        import datashader as ds
    except ImportError:
        raise ImportError(
            "canvas_like requires datashader: pip install datashader"
        )

    cvs = ds.Canvas(
        plot_width=width, plot_height=height, x_range=x_range, y_range=y_range
    )
    out = cvs.raster(raster, **kwargs)

    return out


def _hex_to_rgb(c):
    """Convert a hex color string (e.g. '#ff0000' or 'ff0000') to (r, g, b)."""
    c = c.lstrip('#')
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def color_values(agg, color_key, alpha=255):
    from PIL import Image

    def _convert_color(c):
        r, g, b = _hex_to_rgb(c)
        return np.array([r, g, b, alpha]).astype(np.uint8).view(np.uint32)[0]

    _converted_colors = {k: _convert_color(v) for k, v in color_key.items()}
    f = np.vectorize(lambda v: _converted_colors.get(v, 0))
    return Image.fromarray(f(agg.data).astype(np.uint32).view(np.uint8).reshape(
        agg.data.shape + (4,)), "RGBA")


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
                                 (agg.sizes[dim_y], agg.sizes[dim_x]))
        lon_2d = np.broadcast_to(lon_vals[np.newaxis, :],
                                 (agg.sizes[dim_y], agg.sizes[dim_x]))
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


def _no_shuffle_chunks(chunks, dtype, dims, target_mb):
    """Compute target chunk dict that is an exact multiple of *chunks*.

    Returns a ``{dim: size}`` dict, or ``None`` when the current
    chunks already meet or exceed the target.
    """
    base = tuple(c[0] for c in chunks)

    current_bytes = dtype.itemsize
    for b in base:
        current_bytes *= b

    target_bytes = target_mb * 1024 * 1024

    if current_bytes >= target_bytes:
        return None

    ndim = len(base)
    ratio = target_bytes / current_bytes
    multiplier = max(1, int(ratio ** (1.0 / ndim)))

    if multiplier <= 1:
        return None

    return {dim: b * multiplier for dim, b in zip(dims, base)}


def rechunk_no_shuffle(agg, target_mb=128):
    """Rechunk a dask-backed DataArray or Dataset without triggering a shuffle.

    Computes an integer multiplier per dimension so that each new chunk
    is an exact multiple of the original chunk size.  This lets dask
    merge whole source chunks in-place instead of splitting and
    recombining partial blocks (which is effectively a shuffle).

    Parameters
    ----------
    agg : xr.DataArray or xr.Dataset
        Input raster(s).  If not backed by a dask array the input is
        returned unchanged.  For Datasets, each variable is rechunked
        independently.
    target_mb : int or float
        Target chunk size in megabytes.  The actual chunk size will be
        the closest multiple of the source chunk that does not exceed
        this target.  Default 128.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Rechunked object.  Coordinates and attributes are preserved.

    Raises
    ------
    TypeError
        If *agg* is not an ``xr.DataArray`` or ``xr.Dataset``.
    ValueError
        If *target_mb* is not positive.

    Examples
    --------
    >>> import dask.array as da
    >>> import xarray as xr
    >>> arr = xr.DataArray(da.zeros((4096, 4096), chunks=256))
    >>> big = rechunk_no_shuffle(arr, target_mb=64)
    >>> big.chunks  # multiples of 256
    """
    if isinstance(agg, xr.Dataset):
        return _rechunk_dataset_no_shuffle(agg, target_mb)
    if not isinstance(agg, xr.DataArray):
        raise TypeError(
            f"rechunk_no_shuffle(): expected xr.DataArray or xr.Dataset, "
            f"got {type(agg).__name__}"
        )
    if target_mb <= 0:
        raise ValueError(
            f"rechunk_no_shuffle(): target_mb must be > 0, got {target_mb}"
        )

    if not has_dask_array() or not isinstance(agg.data, da.Array):
        return agg

    new_chunks = _no_shuffle_chunks(
        agg.chunks, agg.dtype, agg.dims, target_mb,
    )
    if new_chunks is None:
        return agg
    return agg.chunk(new_chunks)


def _rechunk_dataset_no_shuffle(ds, target_mb):
    """Rechunk every variable in a Dataset without triggering a shuffle."""
    if target_mb <= 0:
        raise ValueError(
            f"rechunk_no_shuffle(): target_mb must be > 0, got {target_mb}"
        )

    if not has_dask_array():
        return ds

    # Compute target chunks from the first dask-backed variable.
    # This assumes all variables share the same chunk layout and dtype;
    # for mixed-dtype Datasets the budget may overshoot on smaller types.
    new_chunks = None
    for var in ds.data_vars.values():
        if isinstance(var.data, da.Array):
            new_chunks = _no_shuffle_chunks(
                var.chunks, var.dtype, var.dims, target_mb,
            )
            break

    if new_chunks is None:
        return ds

    return ds.chunk(new_chunks)


def _normalize_depth(depth, ndim):
    """Normalize depth to a dict {axis: int}.

    Accepts int, tuple, or dict.  Validates completeness and
    non-negativity.
    """
    if isinstance(depth, dict):
        expected = set(range(ndim))
        got = set(depth.keys())
        missing = expected - got
        extra = got - expected
        if missing:
            raise ValueError(
                f"_normalize_depth: missing axes {sorted(missing)} "
                f"for ndim={ndim}"
            )
        if extra:
            raise ValueError(
                f"_normalize_depth: extra axes {sorted(extra)} "
                f"for ndim={ndim}"
            )
        for v in depth.values():
            if v < 0:
                raise ValueError(
                    f"_normalize_depth: depth must be non-negative, got {v}"
                )
        return dict(depth)

    if isinstance(depth, int):
        if depth < 0:
            raise ValueError(
                f"_normalize_depth: depth must be non-negative, got {depth}"
            )
        return {ax: depth for ax in range(ndim)}

    if isinstance(depth, tuple):
        if len(depth) != ndim:
            raise ValueError(
                f"_normalize_depth: tuple length {len(depth)} != ndim {ndim}"
            )
        for v in depth:
            if v < 0:
                raise ValueError(
                    f"_normalize_depth: depth must be non-negative, got {v}"
                )
        return {ax: d for ax, d in enumerate(depth)}

    raise TypeError(
        f"_normalize_depth: expected int, tuple, or dict, got {type(depth).__name__}"
    )


def _pad_nan(data, depth):
    """Pad a 2-D numpy or cupy array with NaN on each side.

    Parameters
    ----------
    data : numpy or cupy array
    depth : tuple of int
        ``(d0, d1)`` cells to pad per axis.
    """
    pad_width = tuple((d, d) for d in depth)
    if is_cupy_array(data):
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(cupy.float64)
        out = cupy.pad(data, pad_width, mode='constant',
                       constant_values=np.nan)
    else:
        # Promote integer dtypes so NaN fill works
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float64)
        out = np.pad(data, pad_width, mode='constant',
                     constant_values=np.nan)
    return out


def fused_overlap(agg, *stages, boundary='nan'):
    """Run multiple overlap operations in a single map_overlap call.

    Each stage is a ``(func, depth)`` pair. ``func`` receives a padded
    chunk and returns the unpadded interior result.  Stages are fused
    into one ``map_overlap`` call with the sum of all depths, producing
    one blockwise graph layer instead of N.

    Parameters
    ----------
    agg : xr.DataArray
        Input raster.
    *stages : tuple of (callable, depth)
        Each ``func`` takes array ``(H+2*d, W+2*d)`` -> ``(H, W)``.
        ``depth`` is int, tuple, or dict.
    boundary : str
        Must be ``'nan'``.

    Returns
    -------
    xr.DataArray
    """
    if not isinstance(agg, xr.DataArray):
        raise TypeError(
            f"fused_overlap(): expected xr.DataArray, "
            f"got {type(agg).__name__}"
        )
    if not stages:
        raise ValueError("fused_overlap(): need at least one stage")
    if boundary != 'nan':
        raise ValueError(
            f"fused_overlap(): boundary must be 'nan', got {boundary!r}"
        )

    ndim = agg.ndim

    # Normalize and sum depths
    stage_depths = [_normalize_depth(d, ndim) for _, d in stages]
    total_depth = {ax: sum(sd[ax] for sd in stage_depths)
                   for ax in range(ndim)}

    # --- non-dask path ---
    if not has_dask_array() or not isinstance(agg.data, da.Array):
        result = agg.data
        for i, (func, _) in enumerate(stages):
            depth_tuple = tuple(stage_depths[i][ax] for ax in range(ndim))
            padded = _pad_nan(result, depth_tuple)
            result = func(padded)
        return agg.copy(data=result)

    # --- dask path ---
    # Validate chunk sizes
    for ax, d in total_depth.items():
        for cs in agg.chunks[ax]:
            if cs < d:
                raise ValueError(
                    f"Chunk size {cs} on axis {ax} is smaller than "
                    f"total depth {d}. Rechunk first."
                )

    funcs = [f for f, _ in stages]

    def _fused_wrapper(block):
        result = block
        for func in funcs:
            result = func(result)
        return result

    out = agg.data.map_overlap(
        _fused_wrapper,
        depth=total_depth,
        boundary=np.nan,
        trim=False,
        meta=np.array(()),
    )

    return agg.copy(data=out)


def multi_overlap(agg, func, n_outputs, depth, boundary='nan', dtype=None):
    """Run a multi-output kernel via a single overlap + map_blocks call.

    ``func`` receives a padded 2-D chunk and returns
    ``(n_outputs, H, W)`` -- the unpadded interior for each output band.
    The result is a 3-D DataArray with a leading ``band`` dimension.

    Parameters
    ----------
    agg : xr.DataArray
        2-D input raster.
    func : callable
        ``(H+2*dy, W+2*dx) -> (n_outputs, H, W)``
    n_outputs : int
        Number of output bands (>= 1).
    depth : int or tuple of int
        Per-axis overlap (>= 1 on each axis).
    boundary : str
        Boundary mode: 'nan', 'nearest', 'reflect', or 'wrap'.
    dtype : numpy dtype, optional
        Output dtype.  Defaults to input dtype.

    Returns
    -------
    xr.DataArray
        Shape ``(n_outputs, H, W)`` with ``band`` leading dimension.
    """
    if not isinstance(agg, xr.DataArray):
        raise TypeError(
            f"multi_overlap(): expected xr.DataArray, "
            f"got {type(agg).__name__}"
        )
    if agg.ndim != 2:
        raise ValueError(
            f"multi_overlap(): input must be 2-D, got {agg.ndim}-D"
        )
    if n_outputs < 1:
        raise ValueError(
            f"multi_overlap(): n_outputs must be >= 1, got {n_outputs}"
        )

    _validate_boundary(boundary)

    depth_dict = _normalize_depth(depth, agg.ndim)
    for ax, d in depth_dict.items():
        if d < 1:
            raise ValueError(
                f"multi_overlap(): depth must be >= 1, got {d} on axis {ax}"
            )

    dtype = dtype or agg.dtype

    # --- non-dask path ---
    if not has_dask_array() or not isinstance(agg.data, da.Array):
        if boundary == 'nan':
            depth_tuple = tuple(depth_dict[ax] for ax in range(agg.ndim))
            padded = _pad_nan(agg.data, depth_tuple)
        else:
            depth_tuple = tuple(depth_dict[ax] for ax in range(agg.ndim))
            padded = _pad_array(agg.data, depth_tuple, boundary)
        result_data = func(padded).astype(dtype)
        return xr.DataArray(
            result_data,
            dims=['band'] + list(agg.dims),
            coords=agg.coords,
            attrs=agg.attrs,
        )

    # --- dask path ---
    import dask.array.overlap as _dask_overlap

    boundary_val = _boundary_to_dask(boundary, is_cupy=is_cupy_backed(agg))

    # Validate chunk sizes
    for ax, d in depth_dict.items():
        for cs in agg.chunks[ax]:
            if cs < d:
                raise ValueError(
                    f"Chunk size {cs} on axis {ax} is smaller than "
                    f"depth {d}. Rechunk first."
                )

    # Step 1: pad with overlap
    padded = _dask_overlap.overlap(
        agg.data, depth=depth_dict, boundary=boundary_val
    )

    # Step 2: map_blocks -- func returns (n_outputs, H, W) per block
    out = da.map_blocks(
        func,
        padded,
        dtype=dtype,
        new_axis=0,
        chunks=((n_outputs,),) + agg.data.chunks,
    )

    return xr.DataArray(
        out,
        dims=['band'] + list(agg.dims),
        coords=agg.coords,
        attrs=agg.attrs,
    )
