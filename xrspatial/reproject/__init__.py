"""Out-of-core CRS reprojection and multi-raster merge.

Public API
----------
reproject(raster, target_crs, ...)
    Reproject a DataArray to a new coordinate reference system.
merge(rasters, ...)
    Merge multiple DataArrays into a single mosaic.
"""
from __future__ import annotations

import math

import numpy as np
import xarray as xr

from xrspatial.utils import _validate_raster

from ._crs_utils import _detect_nodata, _detect_source_crs, _resolve_crs
from ._grid import (
    _chunk_bounds,
    _compute_chunk_layout,
    _compute_output_grid,
    _make_output_coords,
    _validate_grid_params,
)
from ._interpolate import (
    _resample_cupy,
    _resample_cupy_native,
    _resample_numpy,
    _validate_resampling,
)
from ._merge import _merge_arrays_cupy, _merge_arrays_numpy, _validate_strategy
from ._transform import ApproximateTransform

from ._vertical import (
    geoid_height,
    geoid_height_raster,
    ellipsoidal_to_orthometric,
    orthometric_to_ellipsoidal,
    depth_to_ellipsoidal,
    ellipsoidal_to_depth,
)
from ._itrf import itrf_transform, list_frames as itrf_frames

__all__ = [
    'reproject', 'merge',
    'geoid_height', 'geoid_height_raster',
    'ellipsoidal_to_orthometric', 'orthometric_to_ellipsoidal',
    'depth_to_ellipsoidal', 'ellipsoidal_to_depth',
    'itrf_transform', 'itrf_frames',
]


# ---------------------------------------------------------------------------
# Source geometry helpers
# ---------------------------------------------------------------------------

_Y_NAMES = {'y', 'lat', 'latitude', 'Y', 'Lat', 'Latitude'}
_X_NAMES = {'x', 'lon', 'longitude', 'X', 'Lon', 'Longitude'}

# Map friendly vertical datum tokens to EPSG codes so attrs['vertical_crs']
# from reproject output matches the convention used by xrspatial.geotiff,
# which also writes EPSG ints to attrs['vertical_crs'].
_VERTICAL_DATUM_EPSG = {
    'EGM96': 5773,        # EGM96 height
    'EGM2008': 3855,      # EGM2008 height
    'ellipsoidal': 4979,  # WGS 84 (3D, ellipsoidal height)
}


def _find_spatial_dims(raster):
    """Find the y and x dimension names, handling multi-band rasters.

    Returns (ydim, xdim).  Checks dim names first, falls back to
    assuming the last two non-band dims are spatial.
    """
    dims = raster.dims
    ydim = xdim = None
    for d in dims:
        if d in _Y_NAMES:
            ydim = d
        elif d in _X_NAMES:
            xdim = d
    if ydim is not None and xdim is not None:
        return ydim, xdim
    # Fallback: last two dims
    return dims[-2], dims[-1]


def _source_bounds(raster):
    """Extract (left, bottom, right, top) from a DataArray's coordinates."""
    ydim, xdim = _find_spatial_dims(raster)
    y = raster.coords[ydim].values
    x = raster.coords[xdim].values
    # Compute pixel-edge bounds from pixel-center coords
    if len(y) > 1:
        res_y = abs(float(y[1] - y[0]))
    else:
        res_y = 1.0
    if len(x) > 1:
        res_x = abs(float(x[1] - x[0]))
    else:
        res_x = 1.0
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    left = x_min - res_x / 2
    right = x_max + res_x / 2
    bottom = y_min - res_y / 2
    top = y_max + res_y / 2
    return (left, bottom, right, top)


def _is_y_descending(raster):
    """Check if Y axis goes from top (large) to bottom (small)."""
    ydim, _ = _find_spatial_dims(raster)
    y = raster.coords[ydim].values
    if len(y) < 2:
        return True
    return float(y[0]) > float(y[-1])


# ---------------------------------------------------------------------------
# Per-chunk coordinate transform
# ---------------------------------------------------------------------------

def _transform_coords(transformer, chunk_bounds, chunk_shape,
                      transform_precision, src_crs=None, tgt_crs=None):
    """Compute source CRS coordinates for every output pixel.

    When *transform_precision* is 0, every pixel is transformed through
    pyproj exactly (same strategy as GDAL/rasterio).  Otherwise an
    approximate bilinear control-grid interpolation is used.

    For common CRS pairs (WGS84/NAD83 <-> UTM, WGS84 <-> Web Mercator),
    a Numba JIT fast path bypasses pyproj entirely for ~30x speedup.

    Returns
    -------
    src_y, src_x : ndarray (height, width)
    """
    # Try Numba fast path for common projections
    if src_crs is not None and tgt_crs is not None:
        try:
            from ._projections import try_numba_transform
            result = try_numba_transform(
                src_crs, tgt_crs, chunk_bounds, chunk_shape,
            )
            if result is not None:
                return result
        except (ImportError, ModuleNotFoundError):
            pass  # fall through to pyproj

    height, width = chunk_shape
    left, bottom, right, top = chunk_bounds
    res_x = (right - left) / width
    res_y = (top - bottom) / height

    if transform_precision == 0:
        # Exact per-pixel transform via pyproj bulk API.
        # Process in row strips to keep memory bounded and improve
        # cache locality for large rasters.
        out_x_1d = left + (np.arange(width, dtype=np.float64) + 0.5) * res_x
        src_x_out = np.empty((height, width), dtype=np.float64)
        src_y_out = np.empty((height, width), dtype=np.float64)
        strip = 256
        for r0 in range(0, height, strip):
            r1 = min(r0 + strip, height)
            n_rows = r1 - r0
            out_y_strip = top - (np.arange(r0, r1, dtype=np.float64) + 0.5) * res_y
            # Broadcast to (n_rows, width) without allocating a full copy
            sx, sy = transformer.transform(
                np.tile(out_x_1d, n_rows),
                np.repeat(out_y_strip, width),
            )
            src_x_out[r0:r1] = np.asarray(sx, dtype=np.float64).reshape(n_rows, width)
            src_y_out[r0:r1] = np.asarray(sy, dtype=np.float64).reshape(n_rows, width)
        return src_y_out, src_x_out

    # Approximate: bilinear interpolation on a coarse control grid.
    approx = ApproximateTransform(
        transformer, chunk_bounds, chunk_shape,
        precision=transform_precision,
    )
    row_grid = np.arange(height, dtype=np.float64)[:, np.newaxis]
    col_grid = np.arange(width, dtype=np.float64)[np.newaxis, :]
    row_grid = np.broadcast_to(row_grid, (height, width))
    col_grid = np.broadcast_to(col_grid, (height, width))
    return approx(row_grid, col_grid)


# ---------------------------------------------------------------------------
# Per-chunk worker functions
# ---------------------------------------------------------------------------

def _reproject_chunk_numpy(
    source_data, source_bounds_tuple, source_shape, source_y_desc,
    src_wkt, tgt_wkt,
    chunk_bounds_tuple, chunk_shape,
    resampling, nodata, transform_precision,
):
    """Reproject a single output chunk (numpy backend).

    Called inside ``dask.delayed`` for the dask path, or directly for numpy.
    CRS objects are passed as WKT strings for pickle safety.
    """
    from ._crs_utils import _crs_from_wkt

    src_crs = _crs_from_wkt(src_wkt)
    tgt_crs = _crs_from_wkt(tgt_wkt)

    # Try Numba fast path first (avoids creating pyproj Transformer)
    numba_result = None
    try:
        from ._projections import try_numba_transform
        numba_result = try_numba_transform(
            src_crs, tgt_crs, chunk_bounds_tuple, chunk_shape,
        )
    except (ImportError, ModuleNotFoundError):
        pass

    if numba_result is not None:
        src_y, src_x = numba_result
    else:
        # Fallback: create pyproj Transformer (expensive)
        from ._crs_utils import _require_pyproj
        pyproj = _require_pyproj()
        transformer = pyproj.Transformer.from_crs(
            tgt_crs, src_crs, always_xy=True
        )
        src_y, src_x = _transform_coords(
            transformer, chunk_bounds_tuple, chunk_shape, transform_precision,
            src_crs=src_crs, tgt_crs=tgt_crs,
        )

    # Convert source CRS coordinates to source pixel coordinates
    src_left, src_bottom, src_right, src_top = source_bounds_tuple
    src_h, src_w = source_shape
    src_res_x = (src_right - src_left) / src_w
    src_res_y = (src_top - src_bottom) / src_h

    src_col_px = (src_x - src_left) / src_res_x - 0.5
    if source_y_desc:
        src_row_px = (src_top - src_y) / src_res_y - 0.5
    else:
        src_row_px = (src_y - src_bottom) / src_res_y - 0.5

    # Determine source window needed
    r_min = np.nanmin(src_row_px)
    r_max = np.nanmax(src_row_px)
    c_min = np.nanmin(src_col_px)
    c_max = np.nanmax(src_col_px)

    if not np.isfinite(r_min) or not np.isfinite(r_max):
        return np.full(chunk_shape, nodata, dtype=np.float64)
    if not np.isfinite(c_min) or not np.isfinite(c_max):
        return np.full(chunk_shape, nodata, dtype=np.float64)

    r_min = int(np.floor(r_min)) - 2
    r_max = int(np.ceil(r_max)) + 3
    c_min = int(np.floor(c_min)) - 2
    c_max = int(np.ceil(c_max)) + 3

    # Check overlap
    if r_min >= src_h or r_max <= 0 or c_min >= src_w or c_max <= 0:
        return np.full(chunk_shape, nodata, dtype=np.float64)

    # Clip to source bounds
    r_min_clip = max(0, r_min)
    r_max_clip = min(src_h, r_max)
    c_min_clip = max(0, c_min)
    c_max_clip = min(src_w, c_max)

    # Guard: cap source window to prevent OOM if projection maps a small
    # output chunk to a huge source area (e.g. polar stereographic edges).
    _MAX_WINDOW_PIXELS = 64 * 1024 * 1024  # 64 Mpix (~512 MB for float64)
    win_pixels = (r_max_clip - r_min_clip) * (c_max_clip - c_min_clip)
    if win_pixels > _MAX_WINDOW_PIXELS:
        return np.full(chunk_shape, nodata, dtype=np.float64)

    # Extract source window
    window = source_data[r_min_clip:r_max_clip, c_min_clip:c_max_clip]
    if hasattr(window, 'compute'):
        window = window.compute()
    window = np.asarray(window)
    orig_dtype = window.dtype

    # Adjust coordinates relative to window
    local_row = src_row_px - r_min_clip
    local_col = src_col_px - c_min_clip

    # Multi-band: reproject each band separately, share coordinates
    if window.ndim == 3:
        n_bands = window.shape[2]
        bands = []
        for b in range(n_bands):
            band_data = window[:, :, b].astype(np.float64)
            if not np.isnan(nodata):
                band_data[band_data == nodata] = np.nan
            band_result = _resample_numpy(band_data, local_row, local_col,
                                          resampling=resampling, nodata=nodata)
            if np.issubdtype(orig_dtype, np.integer):
                info = np.iinfo(orig_dtype)
                band_result = np.clip(np.round(band_result), info.min, info.max).astype(orig_dtype)
            bands.append(band_result)
        return np.stack(bands, axis=-1)

    # Single-band path
    window = window.astype(np.float64)

    # Convert sentinel nodata to NaN so numba kernels can detect it
    if not np.isnan(nodata):
        window[window == nodata] = np.nan

    result = _resample_numpy(window, local_row, local_col,
                             resampling=resampling, nodata=nodata)

    # Clamp and cast back for integer source dtypes
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        result = np.clip(np.round(result), info.min, info.max).astype(orig_dtype)

    return result


def _reproject_chunk_cupy(
    source_data, source_bounds_tuple, source_shape, source_y_desc,
    src_wkt, tgt_wkt,
    chunk_bounds_tuple, chunk_shape,
    resampling, nodata, transform_precision,
):
    """CuPy variant of ``_reproject_chunk_numpy``."""
    import cupy as cp

    from ._crs_utils import _crs_from_wkt

    src_crs = _crs_from_wkt(src_wkt)
    tgt_crs = _crs_from_wkt(tgt_wkt)

    # Try CUDA transform first (keeps coordinates on-device)
    cuda_result = None
    if src_crs is not None and tgt_crs is not None:
        try:
            from ._projections_cuda import try_cuda_transform
            cuda_result = try_cuda_transform(
                src_crs, tgt_crs, chunk_bounds_tuple, chunk_shape,
            )
        except (ImportError, ModuleNotFoundError):
            pass

    if cuda_result is not None:
        src_y, src_x = cuda_result  # cupy arrays
        src_left, src_bottom, src_right, src_top = source_bounds_tuple
        src_h, src_w = source_shape
        src_res_x = (src_right - src_left) / src_w
        src_res_y = (src_top - src_bottom) / src_h
        # Pixel coordinate math stays on GPU via cupy operators
        src_col_px = (src_x - src_left) / src_res_x - 0.5
        if source_y_desc:
            src_row_px = (src_top - src_y) / src_res_y - 0.5
        else:
            src_row_px = (src_y - src_bottom) / src_res_y - 0.5
        # Need min/max on CPU for window selection.
        # Stack the four reductions and pull them across in one device-to-host
        # transfer to avoid four separate synchronous syncs.
        mins_maxes = cp.stack([
            cp.nanmin(src_row_px), cp.nanmax(src_row_px),
            cp.nanmin(src_col_px), cp.nanmax(src_col_px),
        ])
        r_min_val, r_max_val, c_min_val, c_max_val = (
            float(v) for v in mins_maxes.get()
        )
        if not (np.isfinite(r_min_val) and np.isfinite(r_max_val)
                and np.isfinite(c_min_val) and np.isfinite(c_max_val)):
            return cp.full(chunk_shape, nodata, dtype=cp.float64)
        r_min = int(np.floor(r_min_val)) - 2
        r_max = int(np.ceil(r_max_val)) + 3
        c_min = int(np.floor(c_min_val)) - 2
        c_max = int(np.ceil(c_max_val)) + 3
        # Keep coordinates as CuPy arrays for native CUDA resampling
        _use_native_cuda = True
    else:
        # CPU fallback (Numba JIT or pyproj)
        from ._crs_utils import _require_pyproj
        pyproj = _require_pyproj()
        transformer = pyproj.Transformer.from_crs(
            tgt_crs, src_crs, always_xy=True
        )
        src_y, src_x = _transform_coords(
            transformer, chunk_bounds_tuple, chunk_shape, transform_precision,
            src_crs=src_crs, tgt_crs=tgt_crs,
        )

        src_left, src_bottom, src_right, src_top = source_bounds_tuple
        src_h, src_w = source_shape
        src_res_x = (src_right - src_left) / src_w
        src_res_y = (src_top - src_bottom) / src_h

        src_col_px = (src_x - src_left) / src_res_x - 0.5
        if source_y_desc:
            src_row_px = (src_top - src_y) / src_res_y - 0.5
        else:
            src_row_px = (src_y - src_bottom) / src_res_y - 0.5

        r_min = np.nanmin(src_row_px)
        r_max = np.nanmax(src_row_px)
        c_min = np.nanmin(src_col_px)
        c_max = np.nanmax(src_col_px)
        if not np.isfinite(r_min) or not np.isfinite(r_max):
            return cp.full(chunk_shape, nodata, dtype=cp.float64)
        if not np.isfinite(c_min) or not np.isfinite(c_max):
            return cp.full(chunk_shape, nodata, dtype=cp.float64)
        r_min = int(np.floor(r_min)) - 2
        r_max = int(np.ceil(r_max)) + 3
        c_min = int(np.floor(c_min)) - 2
        c_max = int(np.ceil(c_max)) + 3
        _use_native_cuda = False

    if r_min >= src_h or r_max <= 0 or c_min >= src_w or c_max <= 0:
        return cp.full(chunk_shape, nodata, dtype=cp.float64)

    r_min_clip = max(0, r_min)
    r_max_clip = min(src_h, r_max)
    c_min_clip = max(0, c_min)
    c_max_clip = min(src_w, c_max)

    # Guard: cap source window to prevent GPU OOM if projection maps a
    # small output chunk to a huge source area (matches numpy path).
    _MAX_WINDOW_PIXELS = 64 * 1024 * 1024  # 64 Mpix (~512 MB for float64)
    win_pixels = (r_max_clip - r_min_clip) * (c_max_clip - c_min_clip)
    if win_pixels > _MAX_WINDOW_PIXELS:
        return cp.full(chunk_shape, nodata, dtype=cp.float64)

    window = source_data[r_min_clip:r_max_clip, c_min_clip:c_max_clip]
    if hasattr(window, 'compute'):
        window = window.compute()
    if not isinstance(window, cp.ndarray):
        window = cp.asarray(window)
    orig_dtype = window.dtype
    window = window.astype(cp.float64)

    # Adjust coordinates relative to window (stays on GPU if CuPy)
    local_row = src_row_px - r_min_clip
    local_col = src_col_px - c_min_clip

    if _use_native_cuda:
        # Coordinates are already CuPy arrays -- use native CUDA kernels
        # (nodata->NaN conversion is handled inside _resample_cupy_native)
        result = _resample_cupy_native(window, local_row, local_col,
                                       resampling=resampling, nodata=nodata)
    else:
        # CPU coordinates -- convert sentinel nodata to NaN before map_coordinates
        if not np.isnan(nodata):
            window[window == nodata] = cp.nan

        result = _resample_cupy(window, local_row, local_col,
                                resampling=resampling, nodata=nodata)

    # Clamp and cast back for integer source dtypes (parity with numpy path)
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        result = cp.clip(cp.round(result), info.min, info.max).astype(orig_dtype)
    return result


# ---------------------------------------------------------------------------
# reproject()
# ---------------------------------------------------------------------------

def reproject(
    raster,
    target_crs,
    *,
    source_crs=None,
    resolution=None,
    bounds=None,
    width=None,
    height=None,
    resampling='bilinear',
    nodata=None,
    transform_precision=16,
    chunk_size=None,
    name=None,
    max_memory=None,
    src_vertical_crs=None,
    tgt_vertical_crs=None,
):
    """Reproject a raster DataArray to a new coordinate reference system.

    Supports numpy, cupy, dask+numpy, and dask+cupy backends. For dask
    inputs, the computation is fully lazy: each output chunk independently
    reads only the source pixels it needs.

    Parameters
    ----------
    raster : xr.DataArray
        Input raster with y/x coordinates.
    target_crs
        Target CRS in any format accepted by ``pyproj.CRS()``.
    source_crs : optional
        Source CRS. Auto-detected from *raster* if None.
    resolution : float or (float, float) or None
        Output pixel size in target CRS units.
    bounds : (left, bottom, right, top) or None
        Explicit output extent in target CRS.
    width, height : int or None
        Explicit output grid dimensions.
    resampling : str
        One of 'nearest', 'bilinear', 'cubic'.
    nodata : float or None
        Nodata value. Auto-detected if None.
    transform_precision : int
        Control-grid subdivisions for the coordinate transform (default 16).
        Higher values increase accuracy at the cost of more pyproj calls.
        Set to 0 for exact per-pixel transforms matching GDAL/rasterio.
    chunk_size : int or (int, int) or None
        Output chunk size for dask. If None, defaults to 512 for the
        standard dask path and 2048 for the in-memory streaming and
        dask+cupy paths (chosen to amortize kernel launch overhead).
    name : str or None
        Name for the output DataArray.
    max_memory : int or str or None
        Maximum memory budget for the reprojection working set.
        Accepts bytes (int) or human-readable strings like ``'4GB'``,
        ``'512MB'``.  Controls how many output tiles are processed
        in parallel for large-dataset streaming mode.  Default None
        uses 1GB.  Has no effect for small datasets that fit in memory.
    src_vertical_crs : str or None
        Source vertical datum for height values. One of:

        - ``'EGM96'`` -- orthometric heights relative to EGM96 geoid (MSL)
        - ``'EGM2008'`` -- orthometric heights relative to EGM2008 geoid
        - ``'ellipsoidal'`` -- heights relative to the WGS84 ellipsoid
        - ``None`` -- no vertical transformation (default)
    tgt_vertical_crs : str or None
        Target vertical datum. Same options as *src_vertical_crs*.
        Both must be set to trigger a vertical transformation.

    Returns
    -------
    xr.DataArray
        The output ``attrs['crs']`` is in WKT format.
        Whenever *tgt_vertical_crs* is set, ``attrs['vertical_crs']``
        records the target vertical datum's EPSG code (5773 for EGM96,
        3855 for EGM2008, 4979 for ellipsoidal WGS84) to match the
        convention used by ``xrspatial.geotiff``. The friendly string
        token (``'EGM96'`` etc.) is preserved under ``attrs['vertical_datum']``.
        Both attrs are written even when no shift is applied (e.g. when
        *src_vertical_crs* equals *tgt_vertical_crs*, or when only the
        target is given), so the output's vertical reference is always
        explicit.

        The output y coordinate is always emitted in descending order
        (top-down, north-up) regardless of the input direction. This
        matches the standard raster convention and the output of common
        GIS libraries.

        Non-spatial coords from the input (such as a scalar ``time``
        coord or a non-dimension coord that is not aligned to the
        spatial dims) are carried through to the output. Coords that
        are aligned to the input y or x dims are dropped because their
        values do not apply to the rebuilt grid.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from xrspatial.reproject import reproject
    >>> raster = xr.DataArray(
    ...     np.random.rand(64, 64),
    ...     dims=['y', 'x'],
    ...     coords={'y': np.linspace(50, 40, 64),
    ...             'x': np.linspace(-5, 5, 64)},
    ...     attrs={'crs': 'EPSG:4326'},
    ... )
    >>> result = reproject(raster, 'EPSG:3857')
    >>> result.attrs['crs'].startswith(('PROJCRS', 'PROJCS'))
    True
    """
    _validate_raster(raster, func_name='reproject', name='raster',
                     ndim=(2, 3))

    _validate_grid_params(
        resolution=resolution,
        bounds=bounds,
        width=width,
        height=height,
        transform_precision=transform_precision,
        func_name='reproject',
    )

    _validate_resampling(resampling)

    # Reject unknown vertical-datum tokens at the API boundary so we never
    # write None into attrs['vertical_crs'] for typos / unsupported values.
    for _name, _val in (('src_vertical_crs', src_vertical_crs),
                        ('tgt_vertical_crs', tgt_vertical_crs)):
        if _val is not None and _val not in _VERTICAL_DATUM_EPSG:
            raise ValueError(
                f"Unknown {_name}={_val!r}; expected one of "
                f"{sorted(_VERTICAL_DATUM_EPSG)} or None."
            )

    # Resolve CRS
    src_crs = _resolve_crs(source_crs)
    if src_crs is None:
        src_crs = _detect_source_crs(raster)
    if src_crs is None:
        raise ValueError(
            "Could not detect source CRS. Pass source_crs explicitly."
        )
    tgt_crs = _resolve_crs(target_crs)

    # Detect nodata
    nd = _detect_nodata(raster, nodata)

    # Source geometry
    src_bounds = _source_bounds(raster)
    _ydim, _xdim = _find_spatial_dims(raster)
    src_shape = (raster.sizes[_ydim], raster.sizes[_xdim])
    y_desc = _is_y_descending(raster)

    # Compute output grid
    grid = _compute_output_grid(
        src_bounds, src_shape, src_crs, tgt_crs,
        resolution=resolution, bounds=bounds,
        width=width, height=height,
    )
    out_bounds = grid['bounds']
    out_shape = grid['shape']

    # Output coordinates
    y_coords, x_coords = _make_output_coords(out_bounds, out_shape)

    # Detect backend
    from ..utils import has_dask_array, is_cupy_array

    data = raster.data
    is_dask = False
    if has_dask_array():
        import dask.array as _da
        is_dask = isinstance(data, _da.Array)
    is_cupy = False
    if is_dask:
        # Check underlying type
        try:
            from ..utils import is_cupy_backed
            is_cupy = is_cupy_backed(raster)
        except (ImportError, ModuleNotFoundError):
            pass
    else:
        is_cupy = is_cupy_array(data)

    # For large in-memory datasets, wrap in dask for chunked processing.
    # map_blocks generates an O(1) HighLevelGraph (single blockwise layer)
    # so graph metadata is no longer a concern -- the streaming fallback
    # is only needed when dask itself is unavailable.
    _use_streaming = False
    if not is_dask and not is_cupy:
        nbytes = src_shape[0] * src_shape[1] * data.dtype.itemsize
        if data.ndim == 3:
            nbytes *= data.shape[2]
        _OOM_THRESHOLD = 512 * 1024 * 1024  # 512 MB
        if nbytes > _OOM_THRESHOLD:
            cs = chunk_size or 2048
            if isinstance(cs, int):
                cs = (cs, cs)
            try:
                import dask.array as _da
                data = _da.from_array(data, chunks=cs)
                raster = xr.DataArray(
                    data, dims=raster.dims, coords=raster.coords,
                    name=raster.name, attrs=raster.attrs,
                )
                is_dask = True
            except ImportError:
                # dask not available -- fall back to streaming
                _use_streaming = True

    # Serialize CRS for pickle safety
    src_wkt = src_crs.to_wkt()
    tgt_wkt = tgt_crs.to_wkt()

    if _use_streaming:
        result_data = _reproject_streaming(
            raster, src_bounds, src_shape, y_desc,
            src_wkt, tgt_wkt,
            out_bounds, out_shape,
            resampling, nd, transform_precision,
            chunk_size or 2048,
            _parse_max_memory(max_memory),
        )
    elif is_dask and is_cupy:
        result_data = _reproject_dask_cupy(
            raster, src_bounds, src_shape, y_desc,
            src_wkt, tgt_wkt,
            out_bounds, out_shape,
            resampling, nd, transform_precision,
            chunk_size,
        )
    elif is_dask:
        result_data = _reproject_dask(
            raster, src_bounds, src_shape, y_desc,
            src_wkt, tgt_wkt,
            out_bounds, out_shape,
            resampling, nd, transform_precision,
            chunk_size, False,
        )
    elif is_cupy:
        result_data = _reproject_inmemory_cupy(
            raster, src_bounds, src_shape, y_desc,
            src_wkt, tgt_wkt,
            out_bounds, out_shape,
            resampling, nd, transform_precision,
        )
    else:
        result_data = _reproject_inmemory_numpy(
            raster, src_bounds, src_shape, y_desc,
            src_wkt, tgt_wkt,
            out_bounds, out_shape,
            resampling, nd, transform_precision,
        )

    # Vertical datum transformation (if requested)
    if src_vertical_crs is not None and tgt_vertical_crs is not None:
        if src_vertical_crs != tgt_vertical_crs:
            result_data = _apply_vertical_shift(
                result_data, y_coords, x_coords,
                src_vertical_crs, tgt_vertical_crs, nd,
                tgt_crs_wkt=tgt_wkt,
            )

    ydim, xdim = _find_spatial_dims(raster)
    # Carry input attrs forward so units, long_name, scale_factor, etc.
    # survive the transform. Re-emit `transform` and `res` for the new
    # output grid (rasterio 6-tuple convention). Drop `crs_wkt` since it
    # would duplicate (or contradict) the canonical `crs` we re-emit.
    out_left, _out_bottom, _out_right, out_top = grid['bounds']
    out_res_x = grid['res_x']
    out_res_y = grid['res_y']
    out_attrs = {**raster.attrs}
    out_attrs.pop('crs_wkt', None)
    out_attrs['crs'] = tgt_wkt
    out_attrs['nodata'] = nd
    out_attrs['res'] = (out_res_x, out_res_y)
    out_attrs['transform'] = (
        out_res_x, 0.0, out_left, 0.0, -out_res_y, out_top,
    )
    # If the input used `_FillValue`, propagate the resolved nodata to
    # both keys so downstream consumers reading either find a consistent
    # value. If `_FillValue` was absent, leave it absent.
    if '_FillValue' in raster.attrs:
        out_attrs['_FillValue'] = nd
    if tgt_vertical_crs is not None:
        # Align with xrspatial.geotiff: attrs['vertical_crs'] holds the
        # EPSG integer code. The friendly string token is preserved under
        # attrs['vertical_datum'] so the human-readable name is not lost.
        # See GH issue #1570.
        out_attrs['vertical_crs'] = _VERTICAL_DATUM_EPSG.get(tgt_vertical_crs)
        out_attrs['vertical_datum'] = tgt_vertical_crs

    # Handle multi-band output (3D result from multi-band source)
    if result_data.ndim == 3:
        # Find the band dimension name from the source
        band_dims = [d for d in raster.dims if d not in (ydim, xdim)]
        band_dim = band_dims[0] if band_dims else 'band'
        out_dims = [ydim, xdim, band_dim]
        out_coords = {ydim: y_coords, xdim: x_coords}
        if band_dim in raster.coords:
            out_coords[band_dim] = raster.coords[band_dim]
        # Carry forward non-spatial coords (e.g. scalar 'time' coord).
        # Skip coords aligned to the rebuilt spatial dims because their
        # values do not apply to the new grid.
        for cname, cval in raster.coords.items():
            if cname in (ydim, xdim, band_dim):
                continue
            if ydim in cval.dims or xdim in cval.dims:
                continue
            out_coords[cname] = cval
    else:
        out_dims = [ydim, xdim]
        out_coords = {ydim: y_coords, xdim: x_coords}
        # Carry forward non-spatial coords (e.g. scalar 'time' coord).
        # Skip coords aligned to the rebuilt spatial dims because their
        # values do not apply to the new grid.
        for cname, cval in raster.coords.items():
            if cname in (ydim, xdim):
                continue
            if ydim in cval.dims or xdim in cval.dims:
                continue
            out_coords[cname] = cval

    result = xr.DataArray(
        result_data,
        dims=out_dims,
        coords=out_coords,
        name=name or raster.name,
        attrs=out_attrs,
    )
    return result


def _apply_vertical_shift(data, y_coords, x_coords,
                          src_vcrs, tgt_vcrs, nodata,
                          tgt_crs_wkt=None):
    """Apply vertical datum shift to reprojected height values.

    The geoid undulation grid is in geographic (lon/lat) coordinates.
    If the output CRS is projected, coordinates are inverse-projected
    to geographic before the geoid lookup.

    Supported vertical CRS:
    - 'EGM96', 'EGM2008': orthometric heights (above geoid/MSL)
    - 'ellipsoidal': heights above WGS84 ellipsoid
    """
    from ._vertical import _load_geoid, _interp_geoid_2d

    # Determine direction
    geoid_models = []
    signs = []

    if src_vcrs in ('EGM96', 'EGM2008') and tgt_vcrs == 'ellipsoidal':
        geoid_models.append(src_vcrs)
        signs.append(1.0)  # H + N = h
    elif src_vcrs == 'ellipsoidal' and tgt_vcrs in ('EGM96', 'EGM2008'):
        geoid_models.append(tgt_vcrs)
        signs.append(-1.0)  # h - N = H
    elif src_vcrs in ('EGM96', 'EGM2008') and tgt_vcrs in ('EGM96', 'EGM2008'):
        geoid_models.extend([src_vcrs, tgt_vcrs])
        signs.extend([1.0, -1.0])  # H1 + N1 - N2
    else:
        return data

    # Determine if we need inverse projection (output CRS is projected)
    need_inverse = False
    transformer = None
    if tgt_crs_wkt is not None:
        try:
            from ._crs_utils import _require_pyproj
            pyproj = _require_pyproj()
            tgt_crs = pyproj.CRS.from_wkt(tgt_crs_wkt)
            if not tgt_crs.is_geographic:
                need_inverse = True
                geo_crs = pyproj.CRS.from_epsg(4326)
                transformer = pyproj.Transformer.from_crs(
                    tgt_crs, geo_crs, always_xy=True
                )
        except Exception:
            pass

    x_arr = np.asarray(x_coords, dtype=np.float64)
    y_arr = np.asarray(y_coords, dtype=np.float64)
    out_h, out_w = data.shape[:2] if hasattr(data, 'shape') else (len(y_arr), len(x_arr))

    # Load geoid grids once
    geoids = []
    for gm in geoid_models:
        geoids.append(_load_geoid(gm))

    # Process in row strips to bound memory (128 rows at a time)
    result = data.copy() if hasattr(data, 'copy') else np.array(data)
    is_nan_nodata = np.isnan(nodata) if isinstance(nodata, float) else False
    strip = 128

    for r0 in range(0, out_h, strip):
        r1 = min(r0 + strip, out_h)
        n_rows = r1 - r0

        # Build strip coordinate grid
        xx_strip = np.tile(x_arr, n_rows).reshape(n_rows, out_w)
        yy_strip = np.repeat(y_arr[r0:r1], out_w).reshape(n_rows, out_w)

        # Inverse project if needed
        if need_inverse and transformer is not None:
            lon_s, lat_s = transformer.transform(xx_strip.ravel(), yy_strip.ravel())
            xx_strip = np.asarray(lon_s, dtype=np.float64).reshape(n_rows, out_w)
            yy_strip = np.asarray(lat_s, dtype=np.float64).reshape(n_rows, out_w)

        # Guard against non-finite output coords (projection singularities,
        # antimeridian, polar regions). Hand NaN to the JIT batch so the
        # longitude wrap loop in _interp_geoid_point does not see inf and
        # spin forever.
        finite_coord = np.isfinite(xx_strip) & np.isfinite(yy_strip)
        if not finite_coord.all():
            xx_strip = np.where(finite_coord, xx_strip, np.nan)
            yy_strip = np.where(finite_coord, yy_strip, np.nan)

        # Apply each geoid shift
        strip_data = result[r0:r1]
        if is_nan_nodata:
            is_valid = np.isfinite(strip_data)
        else:
            is_valid = strip_data != nodata
        is_valid = is_valid & finite_coord

        for (grid_data, g_left, g_top, g_rx, g_ry, g_h, g_w), sign in zip(geoids, signs):
            N_strip = np.empty((n_rows, out_w), dtype=np.float64)
            _interp_geoid_2d(xx_strip, yy_strip, N_strip,
                             grid_data, g_left, g_top, g_rx, g_ry, g_h, g_w)
            strip_data[is_valid] += sign * N_strip[is_valid]

    return result


def _reproject_inmemory_numpy(
    raster, src_bounds, src_shape, y_desc,
    src_wkt, tgt_wkt,
    out_bounds, out_shape,
    resampling, nodata, precision,
):
    """Single-chunk numpy reproject."""
    return _reproject_chunk_numpy(
        raster.values,
        src_bounds, src_shape, y_desc,
        src_wkt, tgt_wkt,
        out_bounds, out_shape,
        resampling, nodata, precision,
    )


def _reproject_inmemory_cupy(
    raster, src_bounds, src_shape, y_desc,
    src_wkt, tgt_wkt,
    out_bounds, out_shape,
    resampling, nodata, precision,
):
    """Single-chunk cupy reproject."""
    return _reproject_chunk_cupy(
        raster.data,
        src_bounds, src_shape, y_desc,
        src_wkt, tgt_wkt,
        out_bounds, out_shape,
        resampling, nodata, precision,
    )


def _parse_max_memory(max_memory):
    """Parse max_memory parameter to bytes.  Accepts int, '4GB', '512MB'."""
    if max_memory is None:
        return 1024 * 1024 * 1024  # 1GB default
    if isinstance(max_memory, (int, float)):
        return int(max_memory)
    s = str(max_memory).strip().upper()
    for suffix, factor in [('TB', 1024**4), ('GB', 1024**3), ('MB', 1024**2), ('KB', 1024)]:
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * factor)
    return int(s)


def _process_tile_batch(batch, source_data, src_bounds, src_shape, y_desc,
                        src_wkt, tgt_wkt, resampling, nodata, precision,
                        max_memory_bytes, tile_mem):
    """Process a batch of tiles within a single worker.

    Uses ThreadPoolExecutor for intra-worker parallelism (Numba
    releases the GIL).  Memory bounded by max_memory_bytes.

    Returns list of (row_offset, col_offset, tile_data) tuples.
    """
    max_concurrent = max(1, max_memory_bytes // max(tile_mem, 1))

    def _do_one(job):
        _, _, rchunk, cchunk, cb = job
        return _reproject_chunk_numpy(
            source_data,
            src_bounds, src_shape, y_desc,
            src_wkt, tgt_wkt,
            cb, (rchunk, cchunk),
            resampling, nodata, precision,
        )

    results = []
    if max_concurrent >= 2 and len(batch) > 1:
        import os
        from concurrent.futures import ThreadPoolExecutor
        n_threads = min(max_concurrent, len(batch), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for sub_start in range(0, len(batch), n_threads):
                sub = batch[sub_start:sub_start + n_threads]
                tiles = list(pool.map(_do_one, sub))
                for job, tile in zip(sub, tiles):
                    ro, co, rchunk, cchunk, _ = job
                    results.append((ro, co, tile))
                del tiles
    else:
        for job in batch:
            ro, co, rchunk, cchunk, _ = job
            tile = _do_one(job)
            results.append((ro, co, tile))
            del tile

    return results


def _reproject_streaming(
    raster, src_bounds, src_shape, y_desc,
    src_wkt, tgt_wkt,
    out_bounds, out_shape,
    resampling, nodata, precision,
    tile_size, max_memory_bytes,
):
    """Streaming reproject for datasets too large for dask's graph.

    Two modes:
    1. **Local** (no dask.distributed): ThreadPoolExecutor within one
       process, bounded by max_memory.
    2. **Distributed** (dask.distributed active): creates a dask.bag
       with one partition per worker, each partition processes its
       tile batch using threads.  Graph size: O(n_workers), not
       O(n_tiles).

    Memory usage per worker: bounded by max_memory.
    """
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)

    row_chunks, col_chunks = _compute_chunk_layout(out_shape, tile_size)

    tile_mem = tile_size[0] * tile_size[1] * 8 * 4  # ~4 arrays per tile

    # Build tile job list
    jobs = []
    row_offset = 0
    for rchunk in row_chunks:
        col_offset = 0
        for cchunk in col_chunks:
            cb = _chunk_bounds(
                out_bounds, out_shape,
                row_offset, row_offset + rchunk,
                col_offset, col_offset + cchunk,
            )
            jobs.append((row_offset, col_offset, rchunk, cchunk, cb))
            col_offset += cchunk
        row_offset += rchunk

    # Check if dask.distributed is active
    _use_distributed = False
    try:
        from dask.distributed import get_client
        client = get_client()
        n_distributed_workers = len(client.scheduler_info()['workers'])
        if n_distributed_workers > 0:
            _use_distributed = True
    except (ImportError, ValueError):
        pass

    if _use_distributed and len(jobs) > n_distributed_workers:
        # Distributed: partition tiles across workers via dask.bag
        import dask.bag as db

        # Split jobs into N partitions (one per worker)
        n_parts = min(n_distributed_workers, len(jobs))
        batch_size = math.ceil(len(jobs) / n_parts)
        batches = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]

        # Create bag and map the batch processor
        bag = db.from_sequence(batches, npartitions=len(batches))
        results_bag = bag.map(
            _process_tile_batch,
            source_data=raster.data,
            src_bounds=src_bounds, src_shape=src_shape, y_desc=y_desc,
            src_wkt=src_wkt, tgt_wkt=tgt_wkt,
            resampling=resampling, nodata=nodata, precision=precision,
            max_memory_bytes=max_memory_bytes, tile_mem=tile_mem,
        )

        # Compute all partitions and assemble result
        result = np.full(out_shape, nodata, dtype=np.float64)
        for batch_results in results_bag.compute():
            for ro, co, tile in batch_results:
                result[ro:ro + tile.shape[0], co:co + tile.shape[1]] = tile
        return result

    # Local: ThreadPoolExecutor within one process
    result = np.full(out_shape, nodata, dtype=np.float64)
    batch_results = _process_tile_batch(
        jobs, raster.data,
        src_bounds, src_shape, y_desc,
        src_wkt, tgt_wkt,
        resampling, nodata, precision,
        max_memory_bytes, tile_mem,
    )
    for ro, co, tile in batch_results:
        result[ro:ro + tile.shape[0], co:co + tile.shape[1]] = tile

    return result


def _reproject_dask_cupy(
    raster, src_bounds, src_shape, y_desc,
    src_wkt, tgt_wkt,
    out_bounds, out_shape,
    resampling, nodata, precision,
    chunk_size,
):
    """Dask+CuPy backend: process output chunks on GPU.

    Two modes based on available GPU memory:

    **Fast path** (output fits in GPU VRAM): pre-allocates the full output
    on GPU and fills it chunk-by-chunk.  ~22x faster than the map_blocks
    path because CRS/transformer objects are created once and CUDA kernels
    run with minimal launch overhead.

    **Chunked fallback** (output exceeds GPU VRAM): delegates to
    ``_reproject_dask(is_cupy=True)`` which uses ``map_blocks`` and
    processes one chunk at a time with O(chunk_size) GPU memory.
    """
    import cupy as cp

    from ._crs_utils import _crs_from_wkt

    src_crs = _crs_from_wkt(src_wkt)
    tgt_crs = _crs_from_wkt(tgt_wkt)

    # Use larger chunks for GPU to amortize kernel launch overhead
    gpu_chunk = chunk_size or 2048
    if isinstance(gpu_chunk, int):
        gpu_chunk = (gpu_chunk, gpu_chunk)

    row_chunks, col_chunks = _compute_chunk_layout(out_shape, gpu_chunk)
    out_h, out_w = out_shape
    src_left, src_bottom, src_right, src_top = src_bounds
    src_h, src_w = src_shape
    src_res_x = (src_right - src_left) / src_w
    src_res_y = (src_top - src_bottom) / src_h

    # Memory check: if the full output doesn't fit in GPU memory,
    # fall back to the map_blocks path which is O(chunk_size) memory.
    estimated = out_shape[0] * out_shape[1] * 8  # float64
    try:
        free_gpu, _total = cp.cuda.Device().mem_info
        fits_in_gpu = estimated < 0.5 * free_gpu
    except (AttributeError, RuntimeError):
        fits_in_gpu = False

    if not fits_in_gpu:
        import warnings
        warnings.warn(
            f"Output ({estimated / 1e9:.1f} GB) exceeds GPU memory; "
            f"falling back to chunked map_blocks path.",
            stacklevel=3,
        )
        return _reproject_dask(
            raster, src_bounds, src_shape, y_desc,
            src_wkt, tgt_wkt,
            out_bounds, out_shape,
            resampling, nodata, precision,
            chunk_size or 2048, True,  # is_cupy=True
        )

    result = cp.full(out_shape, nodata, dtype=cp.float64)

    row_offset = 0
    for i, rchunk in enumerate(row_chunks):
        col_offset = 0
        for j, cchunk in enumerate(col_chunks):
            cb = _chunk_bounds(
                out_bounds, out_shape,
                row_offset, row_offset + rchunk,
                col_offset, col_offset + cchunk,
            )
            chunk_shape = (rchunk, cchunk)

            # CUDA coordinate transform (reuses cached CRS objects)
            try:
                from ._projections_cuda import try_cuda_transform
                cuda_coords = try_cuda_transform(
                    src_crs, tgt_crs, cb, chunk_shape,
                )
            except (ImportError, ModuleNotFoundError):
                cuda_coords = None

            if cuda_coords is not None:
                src_y, src_x = cuda_coords
                src_col_px = (src_x - src_left) / src_res_x - 0.5
                if y_desc:
                    src_row_px = (src_top - src_y) / src_res_y - 0.5
                else:
                    src_row_px = (src_y - src_bottom) / src_res_y - 0.5

                # Batch the four reductions into a single device-to-host
                # transfer instead of four separate synchronous .get() calls.
                mins_maxes = cp.stack([
                    cp.nanmin(src_row_px), cp.nanmax(src_row_px),
                    cp.nanmin(src_col_px), cp.nanmax(src_col_px),
                ])
                r_min_val, r_max_val, c_min_val, c_max_val = (
                    float(v) for v in mins_maxes.get()
                )
                if not (np.isfinite(r_min_val) and np.isfinite(r_max_val)
                        and np.isfinite(c_min_val) and np.isfinite(c_max_val)):
                    col_offset += cchunk
                    continue
                r_min = int(np.floor(r_min_val)) - 2
                r_max = int(np.ceil(r_max_val)) + 3
                c_min = int(np.floor(c_min_val)) - 2
                c_max = int(np.ceil(c_max_val)) + 3
            else:
                # CPU fallback for this chunk
                from ._crs_utils import _require_pyproj
                pyproj = _require_pyproj()
                transformer = pyproj.Transformer.from_crs(
                    tgt_crs, src_crs, always_xy=True
                )
                src_y, src_x = _transform_coords(
                    transformer, cb, chunk_shape, precision,
                    src_crs=src_crs, tgt_crs=tgt_crs,
                )
                src_col_px = (src_x - src_left) / src_res_x - 0.5
                if y_desc:
                    src_row_px = (src_top - src_y) / src_res_y - 0.5
                else:
                    src_row_px = (src_y - src_bottom) / src_res_y - 0.5
                r_min = np.nanmin(src_row_px)
                r_max = np.nanmax(src_row_px)
                c_min = np.nanmin(src_col_px)
                c_max = np.nanmax(src_col_px)
                if not np.isfinite(r_min) or not np.isfinite(r_max):
                    col_offset += cchunk
                    continue
                if not np.isfinite(c_min) or not np.isfinite(c_max):
                    col_offset += cchunk
                    continue
                r_min = int(np.floor(r_min)) - 2
                r_max = int(np.ceil(r_max)) + 3
                c_min = int(np.floor(c_min)) - 2
                c_max = int(np.ceil(c_max)) + 3

            # Check overlap
            if r_min >= src_h or r_max <= 0 or c_min >= src_w or c_max <= 0:
                col_offset += cchunk
                continue

            r_min_clip = max(0, r_min)
            r_max_clip = min(src_h, r_max)
            c_min_clip = max(0, c_min)
            c_max_clip = min(src_w, c_max)

            # Guard: cap source window to prevent GPU OOM (matches numpy path)
            _MAX_WINDOW_PIXELS = 64 * 1024 * 1024  # 64 Mpix
            win_pixels = (r_max_clip - r_min_clip) * (c_max_clip - c_min_clip)
            if win_pixels > _MAX_WINDOW_PIXELS:
                col_offset += cchunk
                continue

            # Fetch only the needed source window from dask
            window = raster.data[r_min_clip:r_max_clip, c_min_clip:c_max_clip]
            if hasattr(window, 'compute'):
                window = window.compute()
            if not isinstance(window, cp.ndarray):
                window = cp.asarray(window)
            window = window.astype(cp.float64)

            if not np.isnan(nodata):
                window[window == nodata] = cp.nan

            local_row = src_row_px - r_min_clip
            local_col = src_col_px - c_min_clip

            if cuda_coords is not None:
                chunk_data = _resample_cupy_native(
                    window, local_row, local_col,
                    resampling=resampling, nodata=nodata,
                )
            else:
                chunk_data = _resample_cupy(
                    window, local_row, local_col,
                    resampling=resampling, nodata=nodata,
                )

            result[row_offset:row_offset + rchunk,
                   col_offset:col_offset + cchunk] = chunk_data
            col_offset += cchunk
        row_offset += rchunk

    return result


def _source_footprint_in_target(src_bounds, src_wkt, tgt_wkt):
    """Compute approximate bounding box of source raster in target CRS."""
    try:
        from ._crs_utils import _crs_from_wkt, _resolve_crs
        try:
            src_crs = _crs_from_wkt(src_wkt)
        except Exception:
            src_crs = _resolve_crs(src_wkt)
        try:
            tgt_crs = _crs_from_wkt(tgt_wkt)
        except Exception:
            tgt_crs = _resolve_crs(tgt_wkt)
    except Exception:
        return None

    sl, sb, sr, st = src_bounds
    mx = (sl + sr) / 2
    my = (sb + st) / 2
    xs = np.array([sl, mx, sr, sl, mx, sr, sl, mx, sr, sl, sr, mx])
    ys = np.array([sb, sb, sb, my, my, my, st, st, st, mx, mx, sb])

    try:
        from ._projections import transform_points
        result = transform_points(src_crs, tgt_crs, xs, ys)
        if result is not None:
            tx, ty = result
            tx = [v for v in tx if np.isfinite(v)]
            ty = [v for v in ty if np.isfinite(v)]
            if not tx or not ty:
                return None
            return (min(tx), min(ty), max(tx), max(ty))
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        from ._crs_utils import _require_pyproj
        pyproj = _require_pyproj()
        transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
        tx, ty = transformer.transform(xs.tolist(), ys.tolist())
        tx = [v for v in tx if np.isfinite(v)]
        ty = [v for v in ty if np.isfinite(v)]
        if not tx or not ty:
            return None
        return (min(tx), min(ty), max(tx), max(ty))
    except Exception:
        return None


def _bounds_overlap(a, b):
    """Return True if bounding boxes *a* and *b* overlap."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _reproject_block_adapter(
    block, block_info, source_data,
    src_bounds, src_shape, y_desc,
    src_wkt, tgt_wkt,
    out_bounds, out_shape,
    resampling, nodata, precision,
    is_cupy, src_footprint_tgt,
):
    """``map_blocks`` adapter for reprojection.

    Derives chunk bounds from *block_info* and delegates to the
    per-chunk worker.
    """
    info = block_info[0]
    (row_start, row_end), (col_start, col_end) = info['array-location']
    chunk_shape = (row_end - row_start, col_end - col_start)
    cb = _chunk_bounds(out_bounds, out_shape,
                       row_start, row_end, col_start, col_end)

    # Skip chunks that don't overlap the source footprint
    if src_footprint_tgt is not None and not _bounds_overlap(cb, src_footprint_tgt):
        return np.full(chunk_shape, nodata, dtype=np.float64)

    chunk_fn = _reproject_chunk_cupy if is_cupy else _reproject_chunk_numpy
    return chunk_fn(
        source_data, src_bounds, src_shape, y_desc,
        src_wkt, tgt_wkt,
        cb, chunk_shape,
        resampling, nodata, precision,
    )


def _reproject_dask(
    raster, src_bounds, src_shape, y_desc,
    src_wkt, tgt_wkt,
    out_bounds, out_shape,
    resampling, nodata, precision,
    chunk_size, is_cupy,
):
    """Dask+NumPy backend: ``map_blocks`` over a template array.

    Uses a single ``blockwise`` layer in the HighLevelGraph instead of
    O(N) ``dask.delayed`` nodes, keeping graph metadata O(1).

    The source dask array is bound to the adapter via ``functools.partial``
    rather than passed as a ``map_blocks`` kwarg.  This prevents dask from
    adding the full source as a dependency of every output block (which
    would cause a MemoryError on distributed schedulers when the source
    exceeds the worker memory limit).
    """
    import functools

    import dask.array as da

    row_chunks, col_chunks = _compute_chunk_layout(out_shape, chunk_size)

    # Precompute source footprint in target CRS for empty-chunk skipping
    src_footprint_tgt = _source_footprint_in_target(
        src_bounds, src_wkt, tgt_wkt
    )

    # Bind the source dask array and all scalar params via partial so
    # map_blocks doesn't detect them as dask Array kwargs (which would
    # add the full source as a dependency of every output block).
    bound_adapter = functools.partial(
        _reproject_block_adapter,
        source_data=raster.data,
        src_bounds=src_bounds,
        src_shape=src_shape,
        y_desc=y_desc,
        src_wkt=src_wkt,
        tgt_wkt=tgt_wkt,
        out_bounds=out_bounds,
        out_shape=out_shape,
        resampling=resampling,
        nodata=nodata,
        precision=precision,
        is_cupy=is_cupy,
        src_footprint_tgt=src_footprint_tgt,
    )

    # Pick the template dtype to match the eager path: integer sources
    # round-trip back to their original dtype after clamping; floats stay
    # float64. Without this, dask claims float64 meta but the chunks
    # actually return the integer dtype, producing inconsistent output.
    src_dtype = np.dtype(raster.dtype)
    if np.issubdtype(src_dtype, np.integer):
        out_dtype = src_dtype
    else:
        out_dtype = np.dtype(np.float64)

    template = da.empty(
        out_shape, dtype=out_dtype, chunks=(row_chunks, col_chunks)
    )

    return da.map_blocks(
        bound_adapter,
        template,
        dtype=out_dtype,
        meta=np.array((), dtype=out_dtype),
    )


# ---------------------------------------------------------------------------
# merge()
# ---------------------------------------------------------------------------

def merge(
    rasters,
    *,
    target_crs=None,
    resolution=None,
    bounds=None,
    resampling='bilinear',
    nodata=None,
    strategy='first',
    chunk_size=None,
    transform_precision=16,
    name=None,
):
    """Merge multiple rasters into a single mosaic.

    Each input is reprojected to the target CRS (if needed) and placed
    into a unified output grid. Overlapping regions are resolved using
    the selected *strategy*.

    Parameters
    ----------
    rasters : list of xr.DataArray
        Input rasters to merge.
    target_crs : optional
        Target CRS. Defaults to the CRS of the first raster.
    resolution : float or (float, float) or None
        Output resolution in target CRS units.
    bounds : (left, bottom, right, top) or None
        Explicit output extent.
    resampling : str
        Interpolation method: 'nearest', 'bilinear', 'cubic'.
    nodata : float or None
        Nodata value for the output.
    strategy : str
        Merge strategy: 'first', 'last', 'mean', 'max', 'min'.
    chunk_size : int or (int, int) or None
        Output chunk size for dask. If None, defaults to 512 for the
        standard dask path and 2048 for the in-memory streaming and
        dask+cupy paths (chosen to amortize kernel launch overhead).
    transform_precision : int
        Control-grid subdivisions for the coordinate transform (default 16).
        Higher values increase accuracy at the cost of more pyproj calls.
        Set to 0 for exact per-pixel transforms matching GDAL/rasterio.
    name : str or None
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
        The output y coordinate is always emitted in descending order
        (top-down, north-up) regardless of the input direction. This
        matches the standard raster convention and the output of common
        GIS libraries.

        Non-spatial coords from the first raster (such as a scalar
        ``time`` coord) are carried through to the output. Coords
        aligned to the spatial dims are dropped because their values
        do not apply to the merged grid.

    Notes
    -----
    There is no streaming in-memory path; for very large output mosaics,
    pass dask-backed inputs (or rely on the automatic promotion to the
    dask path) so that each output chunk is computed independently.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from xrspatial.reproject import merge
    >>> tile_a = xr.DataArray(
    ...     np.full((32, 32), 1.0), dims=['y', 'x'],
    ...     coords={'y': np.linspace(50, 45, 32),
    ...             'x': np.linspace(-5, 0, 32)},
    ...     attrs={'crs': 'EPSG:4326'},
    ... )
    >>> tile_b = xr.DataArray(
    ...     np.full((32, 32), 2.0), dims=['y', 'x'],
    ...     coords={'y': np.linspace(50, 45, 32),
    ...             'x': np.linspace(0, 5, 32)},
    ...     attrs={'crs': 'EPSG:4326'},
    ... )
    >>> mosaic = merge([tile_a, tile_b], resolution=0.5)
    >>> mosaic.shape[1] >= 32
    True
    """
    if not rasters:
        raise ValueError("merge(): rasters list must not be empty")

    for i, r in enumerate(rasters):
        _validate_raster(r, func_name='merge', name=f'rasters[{i}]',
                         ndim=(2, 3))

    _validate_grid_params(
        resolution=resolution,
        bounds=bounds,
        width=None,
        height=None,
        transform_precision=transform_precision,
        func_name='merge',
    )

    _validate_resampling(resampling)
    _validate_strategy(strategy)

    # Resolve target CRS
    tgt_crs = _resolve_crs(target_crs)
    if tgt_crs is None:
        tgt_crs = _detect_source_crs(rasters[0])
    if tgt_crs is None:
        raise ValueError(
            "Could not detect target CRS. Pass target_crs explicitly."
        )

    # Detect output nodata (the sentinel the user asked for)
    nd = nodata if nodata is not None else _detect_nodata(rasters[0], nodata)

    # Gather source info for each raster
    raster_infos = []
    for r in rasters:
        src_crs = _detect_source_crs(r)
        if src_crs is None:
            raise ValueError(
                f"Could not detect CRS for raster '{r.name}'. "
                "Ensure all rasters have CRS metadata."
            )
        sb = _source_bounds(r)
        r_ydim, r_xdim = _find_spatial_dims(r)
        ss = (r.sizes[r_ydim], r.sizes[r_xdim])
        yd = _is_y_descending(r)
        # Per-raster input nodata sentinel. Detected independently of the
        # user-supplied output nodata so that mixed-sentinel inputs are
        # canonicalized correctly during merge.
        r_nd = _detect_nodata(r, None)
        raster_infos.append({
            'raster': r,
            'src_crs': src_crs,
            'src_bounds': sb,
            'src_shape': ss,
            'y_desc': yd,
            'src_wkt': src_crs.to_wkt(),
            'raster_nodata': r_nd,
        })

    # Compute unified output grid
    if bounds is None:
        # Union of all raster bounds in target CRS
        all_bounds = []
        for info in raster_infos:
            grid = _compute_output_grid(
                info['src_bounds'], info['src_shape'],
                info['src_crs'], tgt_crs,
                resolution=resolution,
            )
            all_bounds.append(grid['bounds'])
        left = min(b[0] for b in all_bounds)
        bottom = min(b[1] for b in all_bounds)
        right = max(b[2] for b in all_bounds)
        top = max(b[3] for b in all_bounds)
        merged_bounds = (left, bottom, right, top)
    else:
        merged_bounds = bounds

    # Use first raster's info for resolution estimation if needed
    info0 = raster_infos[0]
    grid = _compute_output_grid(
        info0['src_bounds'], info0['src_shape'],
        info0['src_crs'], tgt_crs,
        resolution=resolution, bounds=merged_bounds,
    )
    out_bounds = grid['bounds']
    out_shape = grid['shape']
    tgt_wkt = tgt_crs.to_wkt()

    # Detect if any input is dask, or if total size exceeds memory threshold
    from ..utils import has_dask_array

    any_dask = False
    if has_dask_array():
        import dask.array as _da
        any_dask = any(isinstance(r.data, _da.Array) for r in rasters)

    # Auto-promote to dask path if output would be too large for in-memory merge
    if not any_dask:
        out_nbytes = out_shape[0] * out_shape[1] * 8 * len(rasters)  # float64 per tile
        _OOM_THRESHOLD = 512 * 1024 * 1024
        if out_nbytes > _OOM_THRESHOLD:
            any_dask = True

    if any_dask:
        result_data = _merge_dask(
            raster_infos, tgt_wkt, out_bounds, out_shape,
            resampling, nd, strategy, chunk_size, transform_precision,
        )
    else:
        result_data = _merge_inmemory(
            raster_infos, tgt_wkt, out_bounds, out_shape,
            resampling, nd, strategy, transform_precision,
        )

    y_coords, x_coords = _make_output_coords(out_bounds, out_shape)
    ydim, xdim = _find_spatial_dims(rasters[0])

    out_coords = {ydim: y_coords, xdim: x_coords}
    # Carry forward non-spatial coords from the first raster (e.g. scalar
    # 'time' coord). Skip coords aligned to the rebuilt spatial dims
    # because their values do not apply to the new grid.
    for cname, cval in rasters[0].coords.items():
        if cname in (ydim, xdim):
            continue
        if ydim in cval.dims or xdim in cval.dims:
            continue
        out_coords[cname] = cval

    # Carry the first raster's attrs forward (matches the default
    # strategy='first'). Drop the duplicate `crs_wkt` and re-emit
    # `transform` and `res` for the new output grid (rasterio 6-tuple).
    out_left, _out_bottom, _out_right, out_top = grid['bounds']
    out_res_x = grid['res_x']
    out_res_y = grid['res_y']
    out_attrs = {**rasters[0].attrs}
    out_attrs.pop('crs_wkt', None)
    out_attrs['crs'] = tgt_wkt
    out_attrs['nodata'] = nd
    out_attrs['res'] = (out_res_x, out_res_y)
    out_attrs['transform'] = (
        out_res_x, 0.0, out_left, 0.0, -out_res_y, out_top,
    )
    # If the first raster used `_FillValue`, propagate the resolved
    # nodata to both keys for consistent round-trip. Leave absent
    # otherwise.
    if '_FillValue' in rasters[0].attrs:
        out_attrs['_FillValue'] = nd

    result = xr.DataArray(
        result_data,
        dims=[ydim, xdim],
        coords=out_coords,
        name=name or rasters[0].name or 'merged',
        attrs=out_attrs,
    )
    return result


def _place_same_crs(src_data, src_bounds, src_shape, y_desc,
                    out_bounds, out_shape, nodata):
    """Place a same-CRS tile into the output grid by coordinate alignment.

    No reprojection needed -- just index the output rows/columns that
    overlap with the source tile and copy the data.
    """
    out_h, out_w = out_shape
    src_h, src_w = src_shape
    o_left, o_bottom, o_right, o_top = out_bounds
    s_left, s_bottom, s_right, s_top = src_bounds

    o_res_x = (o_right - o_left) / out_w
    o_res_y = (o_top - o_bottom) / out_h
    s_res_x = (s_right - s_left) / src_w
    s_res_y = (s_top - s_bottom) / src_h

    # Output pixel range that this tile covers
    col_start = int(round((s_left - o_left) / o_res_x))
    col_end = int(round((s_right - o_left) / o_res_x))
    row_start = int(round((o_top - s_top) / o_res_y))
    row_end = int(round((o_top - s_bottom) / o_res_y))

    # Clip to output bounds
    col_start_clip = max(0, col_start)
    col_end_clip = min(out_w, col_end)
    row_start_clip = max(0, row_start)
    row_end_clip = min(out_h, row_end)

    if col_start_clip >= col_end_clip or row_start_clip >= row_end_clip:
        return np.full(out_shape, nodata, dtype=np.float64)

    # Source pixel range (handle offset if tile extends beyond output)
    src_col_start = col_start_clip - col_start
    src_row_start = row_start_clip - row_start

    # Resolutions may differ slightly; if close enough, do direct copy
    res_ratio_x = s_res_x / o_res_x
    res_ratio_y = s_res_y / o_res_y
    if abs(res_ratio_x - 1.0) > 0.01 or abs(res_ratio_y - 1.0) > 0.01:
        return None  # resolutions too different, fall back to reproject

    out_data = np.full(out_shape, nodata, dtype=np.float64)
    n_rows = row_end_clip - row_start_clip
    n_cols = col_end_clip - col_start_clip

    # Clamp source window
    src_r_end = min(src_row_start + n_rows, src_h)
    src_c_end = min(src_col_start + n_cols, src_w)
    actual_rows = src_r_end - src_row_start
    actual_cols = src_c_end - src_col_start

    if actual_rows <= 0 or actual_cols <= 0:
        return out_data

    src_window = np.asarray(src_data[src_row_start:src_r_end,
                                     src_col_start:src_c_end],
                            dtype=np.float64)
    out_data[row_start_clip:row_start_clip + actual_rows,
             col_start_clip:col_start_clip + actual_cols] = src_window
    return out_data


def _merge_inmemory(
    raster_infos, tgt_wkt, out_bounds, out_shape,
    resampling, nodata, strategy, transform_precision,
):
    """In-memory merge using numpy.

    Detects same-CRS tiles and uses fast direct placement instead
    of reprojection.

    Each raster is reprojected using its own nodata sentinel and then
    canonicalized to NaN before the strategy merge so that mixed-sentinel
    inputs do not leak invalid pixels into the mosaic. After merging, the
    NaN canonical sentinel is converted back to the user-requested
    output ``nodata``.
    """
    from ._crs_utils import _crs_from_wkt
    tgt_crs = _crs_from_wkt(tgt_wkt)

    arrays = []
    for info in raster_infos:
        r_nd = info.get('raster_nodata', float('nan'))
        # Check if source CRS matches target (no reprojection needed)
        placed = None
        if info['src_crs'] == tgt_crs:
            placed = _place_same_crs(
                info['raster'].values,
                info['src_bounds'], info['src_shape'], info['y_desc'],
                out_bounds, out_shape, r_nd,
            )
        if placed is not None:
            arr = placed
        else:
            arr = _reproject_chunk_numpy(
                info['raster'].values,
                info['src_bounds'], info['src_shape'], info['y_desc'],
                info['src_wkt'], tgt_wkt,
                out_bounds, out_shape,
                resampling, r_nd, transform_precision,
            )
        # Canonicalize this raster's sentinel to NaN before the merge so
        # rasters with different sentinels merge correctly.
        if not np.isnan(r_nd):
            arr = np.asarray(arr, dtype=np.float64)
            arr = np.where(arr == r_nd, np.nan, arr)
        arrays.append(arr)

    merged = _merge_arrays_numpy(arrays, float('nan'), strategy)
    if not np.isnan(nodata):
        merged = np.where(np.isnan(merged), nodata, merged)
    return merged


def _merge_block_adapter(
    block, block_info,
    raster_data_list, src_bounds_list, src_shape_list, y_desc_list,
    src_wkt_list, tgt_wkt,
    out_bounds, out_shape,
    resampling, nodata, strategy, precision,
    src_footprints_tgt, raster_nodata_list, same_crs_list,
):
    """``map_blocks`` adapter for merge.

    Each raster is reprojected with its own input nodata sentinel and
    canonicalized to NaN before the strategy merge. The final result is
    converted back to the user-requested output ``nodata``.
    """
    info = block_info[0]
    (row_start, row_end), (col_start, col_end) = info['array-location']
    chunk_shape = (row_end - row_start, col_end - col_start)
    cb = _chunk_bounds(out_bounds, out_shape,
                       row_start, row_end, col_start, col_end)

    # Only reproject rasters whose footprint overlaps this chunk
    arrays = []
    for i in range(len(raster_data_list)):
        if (src_footprints_tgt[i] is not None
                and not _bounds_overlap(cb, src_footprints_tgt[i])):
            continue
        r_nd = raster_nodata_list[i]
        placed = None
        if same_crs_list[i]:
            # Same-CRS path: direct pixel placement (no resampling).
            # Mirrors the eager merge so dask matches numpy bit-for-bit.
            src_data = raster_data_list[i]
            if hasattr(src_data, 'compute'):
                src_data = src_data.compute()
            placed = _place_same_crs(
                np.asarray(src_data),
                src_bounds_list[i], src_shape_list[i], y_desc_list[i],
                cb, chunk_shape, r_nd,
            )
        if placed is not None:
            arr = placed
        else:
            arr = _reproject_chunk_numpy(
                raster_data_list[i],
                src_bounds_list[i], src_shape_list[i], y_desc_list[i],
                src_wkt_list[i], tgt_wkt,
                cb, chunk_shape,
                resampling, r_nd, precision,
            )
        if not np.isnan(r_nd):
            arr = np.asarray(arr, dtype=np.float64)
            arr = np.where(arr == r_nd, np.nan, arr)
        arrays.append(arr)

    if not arrays:
        return np.full(chunk_shape, nodata, dtype=np.float64)
    merged = _merge_arrays_numpy(arrays, float('nan'), strategy)
    if not np.isnan(nodata):
        merged = np.where(np.isnan(merged), nodata, merged)
    return merged


def _merge_dask(
    raster_infos, tgt_wkt, out_bounds, out_shape,
    resampling, nodata, strategy, chunk_size, transform_precision,
):
    """Dask merge backend using ``map_blocks``."""
    import functools

    import dask.array as da

    row_chunks, col_chunks = _compute_chunk_layout(out_shape, chunk_size)

    # Prepare lists for the worker
    data_list = [info['raster'].data for info in raster_infos]
    bounds_list = [info['src_bounds'] for info in raster_infos]
    shape_list = [info['src_shape'] for info in raster_infos]
    ydesc_list = [info['y_desc'] for info in raster_infos]
    wkt_list = [info['src_wkt'] for info in raster_infos]
    rnodata_list = [
        info.get('raster_nodata', float('nan')) for info in raster_infos
    ]

    # Precompute source footprints in target CRS
    footprints = [
        _source_footprint_in_target(bounds_list[i], wkt_list[i], tgt_wkt)
        for i in range(len(raster_infos))
    ]

    # Precompute CRS-equality flags so per-block adapters can shortcut to
    # direct pixel placement (matches the eager _merge_inmemory path).
    from ._crs_utils import _crs_from_wkt
    tgt_crs = _crs_from_wkt(tgt_wkt)
    same_crs_list = [
        bool(_crs_from_wkt(wkt_list[i]) == tgt_crs)
        for i in range(len(raster_infos))
    ]

    # Bind via partial to prevent map_blocks from adding dask arrays
    # in data_list as whole-array dependencies.
    bound_adapter = functools.partial(
        _merge_block_adapter,
        raster_data_list=data_list,
        src_bounds_list=bounds_list,
        src_shape_list=shape_list,
        y_desc_list=ydesc_list,
        src_wkt_list=wkt_list,
        tgt_wkt=tgt_wkt,
        out_bounds=out_bounds,
        out_shape=out_shape,
        resampling=resampling,
        nodata=nodata,
        strategy=strategy,
        precision=transform_precision,
        src_footprints_tgt=footprints,
        raster_nodata_list=rnodata_list,
        same_crs_list=same_crs_list,
    )

    template = da.empty(
        out_shape, dtype=np.float64, chunks=(row_chunks, col_chunks)
    )

    return da.map_blocks(
        bound_adapter,
        template,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )
