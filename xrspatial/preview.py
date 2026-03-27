"""Memory-safe raster preview via downsampling."""

import numpy as np
import xarray as xr

from xrspatial.dataset_support import supports_dataset
from xrspatial.utils import (
    _validate_raster,
    has_cuda_and_cupy,
    is_cupy_array,
)

_COARSEN_METHODS = ('mean', 'median', 'max', 'min')
_METHODS = (*_COARSEN_METHODS, 'nearest', 'bilinear')

# Fallback chunk budget when no distributed client is available.
_DEFAULT_PREVIEW_CHUNK_BYTES = 512 * 1024 * 1024


def _preview_chunk_budget():
    """Max bytes per preview chunk, based on the active dask cluster.

    If a ``dask.distributed`` client is connected, returns
    ``worker_memory * 0.7 / nthreads`` so that concurrent tasks on
    the same worker stay under the memory-pause threshold.  Otherwise
    falls back to ``_DEFAULT_PREVIEW_CHUNK_BYTES`` (512 MB).
    """
    try:
        from dask.distributed import get_client
        client = get_client()
        info = client.scheduler_info()
        workers = info.get('workers', {})
        if workers:
            w = next(iter(workers.values()))
            mem = w.get('memory_limit', 0)
            nthreads = w.get('nthreads', 1) or 1
            if mem > 0:
                return int(mem * 0.7 / nthreads)
    except Exception:
        pass
    return _DEFAULT_PREVIEW_CHUNK_BYTES


def _nan_full(oh, ow, block):
    """NaN-filled ``(oh, ow)`` array matching *block*'s type and dtype."""
    try:
        import cupy
        if isinstance(block, cupy.ndarray):
            return cupy.full((oh, ow), np.nan, dtype=block.dtype)
    except ImportError:
        pass
    return np.full((oh, ow), np.nan, dtype=block.dtype)


def _is_all_nan(block):
    """Fast all-NaN check for float arrays.  Works with numpy and cupy.

    Uses ``nanmax`` (single pass, no intermediate boolean array) after
    a near-free first-element guard that exits immediately for the
    common non-NaN case.
    """
    if block.dtype.kind != 'f' or block.size == 0:
        return False
    first = block.flat[0]
    if first == first:          # not NaN → definitely not all-NaN
        return False
    # nanmax returns NaN iff every element is NaN.
    try:
        import cupy
        if isinstance(block, cupy.ndarray):
            with np.errstate(invalid='ignore'):
                return bool(cupy.isnan(cupy.nanmax(block)))
    except ImportError:
        pass
    with np.errstate(invalid='ignore'):
        return bool(np.isnan(np.nanmax(block)))


# ---------------------------------------------------------------------------
# Block reduction (numpy / cupy)
# ---------------------------------------------------------------------------

def _block_reduce(data, factor_y, factor_x, method):
    """Trim-reshape-reduce a 2D array.  Works with numpy and cupy."""
    oh = data.shape[0] // factor_y
    ow = data.shape[1] // factor_x
    if oh == 0 or ow == 0:
        if has_cuda_and_cupy() and is_cupy_array(data):
            import cupy
            return cupy.empty((oh, ow), dtype=data.dtype)
        return np.empty((oh, ow), dtype=data.dtype)
    if _is_all_nan(data):
        return _nan_full(oh, ow, data)
    trimmed = data[:oh * factor_y, :ow * factor_x]
    blocks = trimmed.reshape(oh, factor_y, ow, factor_x)
    if method == 'median':
        flat = blocks.transpose(0, 2, 1, 3).reshape(oh, ow, -1)
        if has_cuda_and_cupy() and is_cupy_array(data):
            import cupy
            return cupy.median(flat, axis=2)
        return np.median(flat, axis=2).astype(data.dtype)
    return getattr(blocks, method)(axis=(1, 3))


def _reduce_local(agg, factor_y, factor_x, method, y_dim, x_dim):
    """Block reduction for in-memory (numpy / cupy) DataArrays."""
    out_data = _block_reduce(agg.data, factor_y, factor_x, method)
    oh, ow = out_data.shape
    coords = {}
    if y_dim in agg.coords:
        coords[y_dim] = _interpolate_coords(agg.coords[y_dim], oh)
    if x_dim in agg.coords:
        coords[x_dim] = _interpolate_coords(agg.coords[x_dim], ow)
    return xr.DataArray(
        out_data, dims=[y_dim, x_dim], coords=coords, attrs=agg.attrs,
    )


# ---------------------------------------------------------------------------
# Dask block reduction via map_blocks
# ---------------------------------------------------------------------------

def _snap_factor(chunk_size, factor):
    """Return the divisor of *chunk_size* closest to *factor*.

    When the reduction factor evenly divides every chunk, no rechunking
    is needed and the dask graph stays minimal.  The output dimensions
    may overshoot the target; a cheap in-memory second pass corrects
    that afterwards.
    """
    if chunk_size % factor == 0:
        return factor
    best = 1          # 1 always divides; guarantees a result
    best_dist = abs(1 - factor)
    for d in range(2, int(chunk_size ** 0.5) + 1):
        if chunk_size % d == 0:
            for candidate in (d, chunk_size // d):
                dist = abs(candidate - factor)
                if dist < best_dist:
                    best_dist = dist
                    best = candidate
    return best


def _reduce_dask(agg, factor_y, factor_x, method, y_dim, x_dim):
    """Block reduction for dask-backed DataArrays.

    Uses ``dask.array.map_blocks`` so each chunk is independently
    trim-reshape-reduced in a single task.  This produces one graph
    layer on top of the input instead of the five layers that
    ``xarray.coarsen`` generates (reshape, mean_chunk, mean_agg, …).
    """
    import dask.array as da

    data = agg.data

    out_chunks_y = tuple(c // factor_y for c in data.chunks[0])
    out_chunks_x = tuple(c // factor_x for c in data.chunks[1])

    # Captured by the closure; serialised into the task graph.
    _fy, _fx, _m = factor_y, factor_x, method

    def _reduce_block(block):
        oh = block.shape[0] // _fy
        ow = block.shape[1] // _fx
        if oh == 0 or ow == 0:
            return np.empty((oh, ow), dtype=block.dtype)
        if _is_all_nan(block):
            return _nan_full(oh, ow, block)
        trimmed = block[:oh * _fy, :ow * _fx]
        blocks = trimmed.reshape(oh, _fy, ow, _fx)
        if _m == 'median':
            flat = blocks.transpose(0, 2, 1, 3).reshape(oh, ow, -1)
            return np.median(flat, axis=2).astype(block.dtype)
        return getattr(blocks, _m)(axis=(1, 3))

    result_data = da.map_blocks(
        _reduce_block, data,
        dtype=agg.dtype,
        chunks=(out_chunks_y, out_chunks_x),
    )

    out_h = sum(out_chunks_y)
    out_w = sum(out_chunks_x)
    coords = {}
    if y_dim in agg.coords:
        coords[y_dim] = _interpolate_coords(agg.coords[y_dim], out_h)
    if x_dim in agg.coords:
        coords[x_dim] = _interpolate_coords(agg.coords[x_dim], out_w)

    return xr.DataArray(
        result_data, dims=[y_dim, x_dim], coords=coords, attrs=agg.attrs,
    )


# ---------------------------------------------------------------------------
# Bilinear helpers
# ---------------------------------------------------------------------------

def _bilinear_numpy(data, out_h, out_w):
    """Bilinear interpolation on a 2D numpy array."""
    from scipy.ndimage import zoom
    return zoom(data, (out_h / data.shape[0], out_w / data.shape[1]), order=1)


def _bilinear_cupy(data, out_h, out_w):
    """Bilinear interpolation on a 2D cupy array."""
    from cupyx.scipy.ndimage import zoom
    return zoom(data, (out_h / data.shape[0], out_w / data.shape[1]), order=1)


def _bilinear_dask(agg, out_h, out_w, y_dim, x_dim):
    """Memory-safe bilinear interpolation for dask-backed arrays.

    Each chunk is independently zoomed via ``map_blocks``, keeping peak
    memory bounded by the largest input chunk.
    """
    import dask.array as da

    h, w = agg.shape
    in_chunks_y = agg.data.chunks[0]
    in_chunks_x = agg.data.chunks[1]

    # Integer output chunk sizes that sum to exactly out_h / out_w.
    cum_y = np.cumsum([0] + list(in_chunks_y))
    edges_y = np.round(cum_y * (out_h / h)).astype(int)
    out_chunks_y = tuple(int(v) for v in np.diff(edges_y))

    cum_x = np.cumsum([0] + list(in_chunks_x))
    edges_x = np.round(cum_x * (out_w / w)).astype(int)
    out_chunks_x = tuple(int(v) for v in np.diff(edges_x))

    _ocy = out_chunks_y
    _ocx = out_chunks_x

    def _zoom_block(block, block_info=None):
        from scipy.ndimage import zoom

        if block_info is None or block.size == 0:
            return block
        yi, xi = block_info[0]['chunk-location']
        th, tw = _ocy[yi], _ocx[xi]
        if th == 0 or tw == 0:
            return np.empty((th, tw), dtype=block.dtype)
        return zoom(
            block,
            (th / block.shape[0], tw / block.shape[1]),
            order=1,
        )

    result_data = da.map_blocks(
        _zoom_block, agg.data,
        dtype=agg.dtype,
        chunks=(out_chunks_y, out_chunks_x),
    )

    coords = {}
    if y_dim in agg.coords:
        coords[y_dim] = _interpolate_coords(agg.coords[y_dim], out_h)
    if x_dim in agg.coords:
        coords[x_dim] = _interpolate_coords(agg.coords[x_dim], out_w)

    return xr.DataArray(
        result_data, dims=[y_dim, x_dim], coords=coords, attrs=agg.attrs,
    )


def _preview_bilinear(agg, out_h, out_w, y_dim, x_dim):
    """Dispatch bilinear interpolation across backends."""
    import dask.array as da

    if isinstance(agg.data, da.Array):
        return _bilinear_dask(agg, out_h, out_w, y_dim, x_dim)
    elif has_cuda_and_cupy() and is_cupy_array(agg.data):
        out_data = _bilinear_cupy(agg.data, out_h, out_w)
    else:
        out_data = _bilinear_numpy(agg.data, out_h, out_w)

    coords = {}
    if y_dim in agg.coords:
        coords[y_dim] = _interpolate_coords(agg.coords[y_dim], out_h)
    if x_dim in agg.coords:
        coords[x_dim] = _interpolate_coords(agg.coords[x_dim], out_w)

    return xr.DataArray(
        out_data, dims=[y_dim, x_dim], coords=coords, attrs=agg.attrs,
    )


# ---------------------------------------------------------------------------
# Coordinate interpolation
# ---------------------------------------------------------------------------

def _interpolate_coords(coords, n_out):
    """Interpolate coordinate values to *n_out* evenly-spaced index positions.

    Works for both increasing and decreasing coordinates because
    interpolation is done in index-space.
    """
    vals = coords.values
    if len(vals) <= 1 or n_out <= 1:
        return vals[:max(n_out, 1)]
    indices = np.linspace(0, len(vals) - 1, n_out)
    return np.interp(indices, np.arange(len(vals)), vals.astype(np.float64))


# ---------------------------------------------------------------------------
# Second-pass refinement
# ---------------------------------------------------------------------------

def _refine_to_target(result, target_h, target_w, y_dim, x_dim):
    """Subsample a small in-memory result to exact target dimensions.

    When snap-based dask reduction overshoots the requested size (e.g.
    1680 instead of 1000), this picks evenly-spaced rows/columns to
    hit the target exactly.  The intermediate is always small, so this
    is negligible.
    """
    rh = result.sizes[y_dim]
    rw = result.sizes[x_dim]
    out_h = min(rh, target_h)
    out_w = min(rw, target_w)
    if out_h == rh and out_w == rw:
        return result
    idx_y = np.linspace(0, rh - 1, out_h, dtype=int)
    idx_x = np.linspace(0, rw - 1, out_w, dtype=int)
    out_data = result.data[np.ix_(idx_y, idx_x)]
    coords = {}
    if y_dim in result.coords:
        coords[y_dim] = _interpolate_coords(result.coords[y_dim], out_h)
    if x_dim in result.coords:
        coords[x_dim] = _interpolate_coords(result.coords[x_dim], out_w)
    return xr.DataArray(
        out_data, dims=[y_dim, x_dim], coords=coords, attrs=result.attrs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _reopen_preview_chunks(agg):
    """Re-open a zarr-backed DataArray with memory-safe chunks.

    Computes the largest chunk size that is an exact multiple of the
    zarr storage chunks and fits under the per-task memory budget
    (derived from the active dask cluster configuration).  This keeps
    the task graph small (far fewer chunks than storage granularity)
    while keeping peak memory per task well within worker limits even
    when ``threads_per_worker > 1``.

    Returns a new DataArray or *None* if the source isn't available.
    When the input is a spatial subset (``.sel()``), the returned
    array covers the same coordinate range.
    """
    source = agg.encoding.get('_xrs_zarr_source')
    pref = agg.encoding.get('preferred_chunks')
    if source is None or pref is None or agg.name is None:
        return None
    try:
        budget = _preview_chunk_budget()
        # Compute the largest multiple of storage chunks that fits
        # under the per-task budget.
        base = tuple(pref[d] for d in agg.dims if d in pref)
        if not base or len(base) != 2:
            return None
        base_bytes = agg.dtype.itemsize * base[0] * base[1]
        if base_bytes >= budget:
            # Storage chunks already exceed the budget; use them as-is.
            chunks = pref
        else:
            ratio = budget / base_bytes
            multiplier = max(1, int(ratio ** (1.0 / len(base))))
            chunks = {d: pref[d] * multiplier for d in agg.dims if d in pref}

        ds = xr.open_zarr(source, chunks=chunks)
        if agg.name not in ds:
            return None
        da_full = ds[agg.name]
        # Select to match the current DataArray's coordinate extent.
        sel = {}
        for dim in agg.dims:
            if dim in agg.coords and dim in da_full.coords:
                c = agg.coords[dim].values
                if len(c) > 0:
                    sel[dim] = slice(c[0], c[-1])
        if sel:
            da_full = da_full.sel(sel)
        return da_full
    except Exception:
        return None


@supports_dataset
def preview(agg, width=1000, height=None, method='mean', name='preview'):
    """Downsample a raster to target pixel dimensions.

    For dask-backed arrays, the operation is lazy: each chunk is reduced
    independently, so peak memory is bounded by the largest chunk plus
    the small output array.  A 30 TB raster can be previewed at
    1000x1000 with only a few MB of RAM.

    Parameters
    ----------
    agg : xr.DataArray
        Input raster (2D).
    width : int, default 1000
        Target width in pixels.
    height : int, optional
        Target height in pixels.  If not provided, computed from *width*
        preserving the aspect ratio of *agg*.
    method : str, default 'mean'
        Downsampling method.  One of:

        - ``'mean'``: block averaging.
        - ``'median'``: block median.
        - ``'max'``: block maximum.
        - ``'min'``: block minimum.
        - ``'nearest'``: stride-based subsampling (fastest, no smoothing).
        - ``'bilinear'``: bilinear interpolation via ``scipy.ndimage.zoom``.
    name : str, default 'preview'
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
        Downsampled raster with updated coordinates.
    """
    _validate_raster(agg, func_name='preview', ndim=2)

    if method not in _METHODS:
        raise ValueError(
            f"method must be one of {_METHODS!r}, got {method!r}"
        )

    # If chunks are too large for a single worker task, re-open from
    # the zarr source with memory-safe chunks.  The budget accounts
    # for threads_per_worker so concurrent tasks don't collectively
    # exceed the worker's memory-pause threshold.
    try:
        import dask.array as _da
        if isinstance(agg.data, _da.Array):
            chunk_bytes = (agg.dtype.itemsize
                          * agg.data.chunksize[0] * agg.data.chunksize[1])
            if chunk_bytes > _preview_chunk_budget():
                safe = _reopen_preview_chunks(agg)
                if safe is not None:
                    agg = safe
    except ImportError:
        pass

    h = agg.sizes[agg.dims[0]]
    w = agg.sizes[agg.dims[1]]

    if height is None:
        height = max(1, round(width * h / w))

    factor_y = max(1, h // height)
    factor_x = max(1, w // width)

    if factor_y <= 1 and factor_x <= 1:
        return agg

    y_dim = agg.dims[0]
    x_dim = agg.dims[1]

    # Save the original targets before snap may widen them.
    target_h, target_w = height, width

    # For dask arrays, snap each factor to the nearest divisor of the
    # chunk size so that every chunk divides evenly and no rechunking
    # is needed.  The output dimensions may overshoot the target; a
    # cheap second pass corrects that below.
    try:
        import dask.array as da
        if isinstance(agg.data, da.Array):
            factor_y = _snap_factor(agg.data.chunksize[0], factor_y)
            factor_x = _snap_factor(agg.data.chunksize[1], factor_x)
            height = h // factor_y
            width = w // factor_x
    except ImportError:
        pass

    # Pre-trim the input to an exact multiple of the factors so the
    # reduce output is exactly (height, width) without a post-reduce
    # trim.  On dask arrays this only touches boundary chunks, avoiding
    # two extra getitem layers over the (much larger) output grid.
    trim_h = height * factor_y
    trim_w = width * factor_x
    trim = {}
    if trim_h < h:
        trim[y_dim] = slice(0, trim_h)
    if trim_w < w:
        trim[x_dim] = slice(0, trim_w)
    if trim:
        agg = agg.isel(trim)

    if method == 'nearest':
        result = agg.isel(
            {y_dim: slice(None, None, factor_y),
             x_dim: slice(None, None, factor_x)}
        )
    elif method == 'bilinear':
        result = _preview_bilinear(agg, height, width, y_dim, x_dim)
    else:
        # mean / median / max / min
        try:
            import dask.array as da
            is_dask = isinstance(agg.data, da.Array)
        except ImportError:
            is_dask = False

        if is_dask:
            result = _reduce_dask(
                agg, factor_y, factor_x, method, y_dim, x_dim,
            )
        else:
            result = _reduce_local(
                agg, factor_y, factor_x, method, y_dim, x_dim,
            )

    result.name = name
    result.attrs = agg.attrs

    # Second pass: if snap overshot the target, compute the small
    # intermediate and subsample to exact dimensions.
    if (result.sizes[y_dim] > target_h or result.sizes[x_dim] > target_w):
        try:
            result = result.compute()
        except (AttributeError, TypeError):
            pass
        result = _refine_to_target(
            result, target_h, target_w, y_dim, x_dim,
        )
        result.name = name

    return result
