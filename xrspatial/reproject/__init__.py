"""Out-of-core CRS reprojection and multi-raster merge.

Public API
----------
reproject(raster, target_crs, ...)
    Reproject a DataArray to a new coordinate reference system.
merge(rasters, ...)
    Merge multiple DataArrays into a single mosaic.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ._crs_utils import _detect_nodata, _detect_source_crs, _resolve_crs
from ._grid import (
    _chunk_bounds,
    _compute_chunk_layout,
    _compute_output_grid,
    _make_output_coords,
)
from ._interpolate import _resample_cupy, _resample_numpy, _validate_resampling
from ._merge import _merge_arrays_cupy, _merge_arrays_numpy, _validate_strategy
from ._transform import ApproximateTransform

__all__ = ['reproject', 'merge']


# ---------------------------------------------------------------------------
# Source geometry helpers
# ---------------------------------------------------------------------------

def _source_bounds(raster):
    """Extract (left, bottom, right, top) from a DataArray's coordinates."""
    ydim = raster.dims[-2]
    xdim = raster.dims[-1]
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
    ydim = raster.dims[-2]
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
        except Exception:
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
    from ._crs_utils import _require_pyproj

    pyproj = _require_pyproj()
    src_crs = pyproj.CRS.from_wkt(src_wkt)
    tgt_crs = pyproj.CRS.from_wkt(tgt_wkt)

    # Build inverse transformer: target -> source
    transformer = pyproj.Transformer.from_crs(
        tgt_crs, src_crs, always_xy=True
    )

    # Source CRS coordinates for each output pixel
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
    r_min = int(np.floor(np.nanmin(src_row_px))) - 2
    r_max = int(np.ceil(np.nanmax(src_row_px))) + 3
    c_min = int(np.floor(np.nanmin(src_col_px))) - 2
    c_max = int(np.ceil(np.nanmax(src_col_px))) + 3

    # Check overlap
    if r_min >= src_h or r_max <= 0 or c_min >= src_w or c_max <= 0:
        return np.full(chunk_shape, nodata, dtype=np.float64)

    # Clip to source bounds
    r_min_clip = max(0, r_min)
    r_max_clip = min(src_h, r_max)
    c_min_clip = max(0, c_min)
    c_max_clip = min(src_w, c_max)

    # Extract source window
    window = source_data[r_min_clip:r_max_clip, c_min_clip:c_max_clip]
    if hasattr(window, 'compute'):
        window = window.compute()
    window = np.asarray(window, dtype=np.float64)

    # Convert sentinel nodata to NaN so numba kernels can detect it
    if not np.isnan(nodata):
        window = window.copy()
        window[window == nodata] = np.nan

    # Adjust coordinates relative to window
    local_row = src_row_px - r_min_clip
    local_col = src_col_px - c_min_clip

    return _resample_numpy(window, local_row, local_col,
                           resampling=resampling, nodata=nodata)


def _reproject_chunk_cupy(
    source_data, source_bounds_tuple, source_shape, source_y_desc,
    src_wkt, tgt_wkt,
    chunk_bounds_tuple, chunk_shape,
    resampling, nodata, transform_precision,
):
    """CuPy variant of ``_reproject_chunk_numpy``."""
    import cupy as cp

    from ._crs_utils import _require_pyproj

    pyproj = _require_pyproj()
    src_crs = pyproj.CRS.from_wkt(src_wkt)
    tgt_crs = pyproj.CRS.from_wkt(tgt_wkt)

    transformer = pyproj.Transformer.from_crs(
        tgt_crs, src_crs, always_xy=True
    )

    # Try CUDA transform first (keeps coordinates on-device)
    cuda_result = None
    if src_crs is not None and tgt_crs is not None:
        try:
            from ._projections_cuda import try_cuda_transform
            cuda_result = try_cuda_transform(
                src_crs, tgt_crs, chunk_bounds_tuple, chunk_shape,
            )
        except Exception:
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
        # Need min/max on CPU for window selection
        r_min = int(cp.floor(cp.nanmin(src_row_px)).get()) - 2
        r_max = int(cp.ceil(cp.nanmax(src_row_px)).get()) + 3
        c_min = int(cp.floor(cp.nanmin(src_col_px)).get()) - 2
        c_max = int(cp.ceil(cp.nanmax(src_col_px)).get()) + 3
        # Convert to numpy for downstream resampling
        src_row_px = cp.asnumpy(src_row_px)
        src_col_px = cp.asnumpy(src_col_px)
    else:
        # CPU fallback (Numba JIT or pyproj)
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

        r_min = int(np.floor(np.nanmin(src_row_px))) - 2
        r_max = int(np.ceil(np.nanmax(src_row_px))) + 3
        c_min = int(np.floor(np.nanmin(src_col_px))) - 2
        c_max = int(np.ceil(np.nanmax(src_col_px))) + 3

    if r_min >= src_h or r_max <= 0 or c_min >= src_w or c_max <= 0:
        return cp.full(chunk_shape, nodata, dtype=cp.float64)

    r_min_clip = max(0, r_min)
    r_max_clip = min(src_h, r_max)
    c_min_clip = max(0, c_min)
    c_max_clip = min(src_w, c_max)

    window = source_data[r_min_clip:r_max_clip, c_min_clip:c_max_clip]
    if hasattr(window, 'compute'):
        window = window.compute()
    if not isinstance(window, cp.ndarray):
        window = cp.asarray(window)
    window = window.astype(cp.float64)

    # Convert sentinel nodata to NaN
    if not np.isnan(nodata):
        window = window.copy()
        window[window == nodata] = cp.nan

    local_row = src_row_px - r_min_clip
    local_col = src_col_px - c_min_clip

    return _resample_cupy(window, local_row, local_col,
                          resampling=resampling, nodata=nodata)


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
        Output chunk size for dask. Defaults to 512.
    name : str or None
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
    """
    from ._crs_utils import _require_pyproj

    if not isinstance(raster, xr.DataArray):
        raise TypeError(
            f"reproject(): raster must be an xr.DataArray, "
            f"got {type(raster).__name__}"
        )

    _validate_resampling(resampling)
    _require_pyproj()

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
    src_shape = (raster.sizes[raster.dims[-2]], raster.sizes[raster.dims[-1]])
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
        except (ImportError, Exception):
            pass
    else:
        is_cupy = is_cupy_array(data)

    # Serialize CRS for pickle safety
    src_wkt = src_crs.to_wkt()
    tgt_wkt = tgt_crs.to_wkt()

    if is_dask:
        result_data = _reproject_dask(
            raster, src_bounds, src_shape, y_desc,
            src_wkt, tgt_wkt,
            out_bounds, out_shape,
            resampling, nd, transform_precision,
            chunk_size, is_cupy,
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

    ydim = raster.dims[-2]
    xdim = raster.dims[-1]
    result = xr.DataArray(
        result_data,
        dims=[ydim, xdim],
        coords={ydim: y_coords, xdim: x_coords},
        name=name or raster.name,
        attrs={
            'crs': tgt_wkt,
            'nodata': nd,
        },
    )
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


def _reproject_dask(
    raster, src_bounds, src_shape, y_desc,
    src_wkt, tgt_wkt,
    out_bounds, out_shape,
    resampling, nodata, precision,
    chunk_size, is_cupy,
):
    """Dask backend: build output as ``da.block`` of delayed chunks."""
    import dask
    import dask.array as da

    row_chunks, col_chunks = _compute_chunk_layout(out_shape, chunk_size)
    n_row = len(row_chunks)
    n_col = len(col_chunks)

    chunk_fn = _reproject_chunk_cupy if is_cupy else _reproject_chunk_numpy
    dtype = np.float64

    blocks = [[None] * n_col for _ in range(n_row)]

    row_offset = 0
    for i in range(n_row):
        col_offset = 0
        for j in range(n_col):
            rchunk = row_chunks[i]
            cchunk = col_chunks[j]
            cb = _chunk_bounds(
                out_bounds, out_shape,
                row_offset, row_offset + rchunk,
                col_offset, col_offset + cchunk,
            )
            delayed_chunk = dask.delayed(chunk_fn)(
                raster.data,
                src_bounds, src_shape, y_desc,
                src_wkt, tgt_wkt,
                cb, (rchunk, cchunk),
                resampling, nodata, precision,
            )
            blocks[i][j] = da.from_delayed(
                delayed_chunk, shape=(rchunk, cchunk), dtype=dtype
            )
            col_offset += cchunk
        row_offset += rchunk

    return da.block(blocks)


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
        Chunk size for dask output.
    name : str or None
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
    """
    from ._crs_utils import _require_pyproj

    if not rasters:
        raise ValueError("merge(): rasters list must not be empty")

    _validate_resampling(resampling)
    _validate_strategy(strategy)
    pyproj = _require_pyproj()

    # Resolve target CRS
    tgt_crs = _resolve_crs(target_crs)
    if tgt_crs is None:
        tgt_crs = _detect_source_crs(rasters[0])
    if tgt_crs is None:
        raise ValueError(
            "Could not detect target CRS. Pass target_crs explicitly."
        )

    # Detect nodata
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
        ss = (r.sizes[r.dims[-2]], r.sizes[r.dims[-1]])
        yd = _is_y_descending(r)
        raster_infos.append({
            'raster': r,
            'src_crs': src_crs,
            'src_bounds': sb,
            'src_shape': ss,
            'y_desc': yd,
            'src_wkt': src_crs.to_wkt(),
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

    # Detect if any input is dask
    from ..utils import has_dask_array

    any_dask = False
    if has_dask_array():
        import dask.array as _da
        any_dask = any(isinstance(r.data, _da.Array) for r in rasters)

    if any_dask:
        result_data = _merge_dask(
            raster_infos, tgt_wkt, out_bounds, out_shape,
            resampling, nd, strategy, chunk_size,
        )
    else:
        result_data = _merge_inmemory(
            raster_infos, tgt_wkt, out_bounds, out_shape,
            resampling, nd, strategy,
        )

    y_coords, x_coords = _make_output_coords(out_bounds, out_shape)
    ydim = rasters[0].dims[-2]
    xdim = rasters[0].dims[-1]

    result = xr.DataArray(
        result_data,
        dims=[ydim, xdim],
        coords={ydim: y_coords, xdim: x_coords},
        name=name or 'merged',
        attrs={
            'crs': tgt_wkt,
            'nodata': nd,
        },
    )
    return result


def _merge_inmemory(
    raster_infos, tgt_wkt, out_bounds, out_shape,
    resampling, nodata, strategy,
):
    """In-memory merge using numpy."""
    arrays = []
    for info in raster_infos:
        reprojected = _reproject_chunk_numpy(
            info['raster'].values,
            info['src_bounds'], info['src_shape'], info['y_desc'],
            info['src_wkt'], tgt_wkt,
            out_bounds, out_shape,
            resampling, nodata, 16,
        )
        arrays.append(reprojected)
    return _merge_arrays_numpy(arrays, nodata, strategy)


def _merge_chunk_worker(
    raster_data_list, src_bounds_list, src_shape_list, y_desc_list,
    src_wkt_list, tgt_wkt,
    chunk_bounds_tuple, chunk_shape,
    resampling, nodata, strategy, precision,
):
    """Worker for a single merge chunk."""
    arrays = []
    for i in range(len(raster_data_list)):
        reprojected = _reproject_chunk_numpy(
            raster_data_list[i],
            src_bounds_list[i], src_shape_list[i], y_desc_list[i],
            src_wkt_list[i], tgt_wkt,
            chunk_bounds_tuple, chunk_shape,
            resampling, nodata, precision,
        )
        arrays.append(reprojected)
    return _merge_arrays_numpy(arrays, nodata, strategy)


def _merge_dask(
    raster_infos, tgt_wkt, out_bounds, out_shape,
    resampling, nodata, strategy, chunk_size,
):
    """Dask merge backend."""
    import dask
    import dask.array as da

    row_chunks, col_chunks = _compute_chunk_layout(out_shape, chunk_size)
    n_row = len(row_chunks)
    n_col = len(col_chunks)

    # Prepare lists for the worker
    data_list = [info['raster'].data for info in raster_infos]
    bounds_list = [info['src_bounds'] for info in raster_infos]
    shape_list = [info['src_shape'] for info in raster_infos]
    ydesc_list = [info['y_desc'] for info in raster_infos]
    wkt_list = [info['src_wkt'] for info in raster_infos]

    dtype = np.float64
    blocks = [[None] * n_col for _ in range(n_row)]

    row_offset = 0
    for i in range(n_row):
        col_offset = 0
        for j in range(n_col):
            rchunk = row_chunks[i]
            cchunk = col_chunks[j]
            cb = _chunk_bounds(
                out_bounds, out_shape,
                row_offset, row_offset + rchunk,
                col_offset, col_offset + cchunk,
            )
            delayed_chunk = dask.delayed(_merge_chunk_worker)(
                data_list, bounds_list, shape_list, ydesc_list,
                wkt_list, tgt_wkt,
                cb, (rchunk, cchunk),
                resampling, nodata, strategy, 16,
            )
            blocks[i][j] = da.from_delayed(
                delayed_chunk, shape=(rchunk, cchunk), dtype=dtype
            )
            col_offset += cchunk
        row_offset += rchunk

    return da.block(blocks)
