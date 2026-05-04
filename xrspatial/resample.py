"""Raster resampling -- resolution change without reprojection.

Provides :func:`resample` for changing raster cell size using
interpolation or block-aggregation methods.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates as _scipy_map_coords

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import cupy
except ImportError:
    cupy = None

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _validate_raster,
    calc_res,
    ngjit,
)
from xrspatial.dataset_support import supports_dataset


# -- Constants ---------------------------------------------------------------

INTERP_METHODS = {'nearest': 0, 'bilinear': 1, 'cubic': 3}
AGGREGATE_METHODS = {'average', 'min', 'max', 'median', 'mode'}
ALL_METHODS = set(INTERP_METHODS) | AGGREGATE_METHODS

# Overlap depth (input pixels) each interpolation kernel needs from
# neighbouring chunks when processing dask arrays.  Cubic requires extra
# depth because the B-spline prefilter is a global IIR filter whose
# boundary transient decays as ~0.268^n; depth 10 brings the boundary
# transient to ~1e-7 (sufficient for float32 output and well under
# typical interpolation error).
_INTERP_DEPTH = {'nearest': 1, 'bilinear': 1, 'cubic': 10}

# Approximate working-set size per output cell for the eager backends:
# one float64 working buffer (8 B) plus a float32 output cell (4 B).
# scipy.ndimage.map_coordinates also allocates a temporary of the same
# size during higher-order spline evaluation; the 0.5 * available bound
# below leaves room for that.
_BYTES_PER_OUTPUT_CELL = 12


# -- Memory guard ------------------------------------------------------------

def _available_memory_bytes():
    """Best-effort estimate of available host memory in bytes."""
    # Try /proc/meminfo (Linux)
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    # Try psutil
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    # Fallback: 2 GB
    return 2 * 1024 ** 3


def _available_gpu_memory_bytes():
    """Best-effort estimate of free GPU memory in bytes.

    Returns 0 when CuPy / CUDA is unavailable or the query fails -- callers
    treat that as a sentinel meaning "no GPU info, skip the guard".
    """
    try:
        import cupy as _cp
        free, _total = _cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return 0


def _check_resample_memory(out_h, out_w):
    """Raise MemoryError if the eager output buffer would exceed RAM.

    The numpy and cupy-eager backends allocate a single (out_h, out_w)
    float64 working buffer plus a float32 output before any actual work.
    A user passing a huge ``scale_factor`` (or a tiny ``target_resolution``)
    would otherwise OOM the process before this function returns.
    """
    required = int(out_h) * int(out_w) * _BYTES_PER_OUTPUT_CELL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"resample output of {out_h}x{out_w} would need "
            f"~{required / 1e9:.1f} GB of working memory but only "
            f"~{available / 1e9:.1f} GB is available. "
            f"Use a smaller scale_factor / larger target_resolution, "
            f"or pass a dask-backed DataArray for out-of-core processing."
        )


def _check_resample_gpu_memory(out_h, out_w):
    """Raise MemoryError if the cupy-eager output buffer would exceed VRAM.

    Skips the check (returns silently) when free GPU memory cannot be
    queried -- the kernel will fail later at the cupy.empty boundary
    anyway.
    """
    available = _available_gpu_memory_bytes()
    if available <= 0:
        return
    required = int(out_h) * int(out_w) * _BYTES_PER_OUTPUT_CELL
    if required > 0.5 * available:
        raise MemoryError(
            f"resample output of {out_h}x{out_w} would need "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device. "
            f"Use a smaller scale_factor / larger target_resolution, "
            f"or pass a dask+cupy DataArray for out-of-core processing."
        )


# -- Output-geometry helpers -------------------------------------------------

def _output_shape(in_h, in_w, scale_y, scale_x):
    return max(1, round(in_h * scale_y)), max(1, round(in_w * scale_x))


def _output_chunks(in_chunks, scale):
    """Compute per-chunk output sizes via cumulative rounding.

    Guarantees ``sum(result) == round(sum(in_chunks) * scale)``.
    """
    cum = np.cumsum([0] + list(in_chunks))
    out_cum = np.round(cum * scale).astype(int)
    return tuple(int(max(1, out_cum[i + 1] - out_cum[i]))
                 for i in range(len(in_chunks)))


# -- Block-centered coordinate mapping ---------------------------------------

def _block_centered_coords(n_in, n_out):
    """Return input coordinates for each output pixel using block-centered mapping.

    Maps output pixel ``o`` to input pixel ``(o + 0.5) * (n_in / n_out) - 0.5``.
    This places each output pixel at the center of its spatial footprint,
    matching the convention used by ``_new_coords`` for output coordinate
    metadata.
    """
    o = np.arange(n_out, dtype=np.float64)
    return (o + 0.5) * (n_in / n_out) - 0.5


# -- NaN-aware interpolation (NumPy) ----------------------------------------

def _nan_aware_interp_np(data, out_h, out_w, order):
    """Interpolate *data* to *(out_h, out_w)* with NaN-aware weighting.

    Uses ``scipy.ndimage.map_coordinates`` with block-centered coordinate
    mapping so that sample positions match the output coordinate metadata.

    For *order* 0 (nearest-neighbour) NaN propagates naturally.
    For higher orders the zero-fill / weight-mask trick is used so that
    NaN pixels do not corrupt their neighbours.
    """
    iy = _block_centered_coords(data.shape[0], out_h)
    ix = _block_centered_coords(data.shape[1], out_w)
    yy, xx = np.meshgrid(iy, ix, indexing='ij')
    coords = np.array([yy.ravel(), xx.ravel()])

    if order == 0:
        result = _scipy_map_coords(data, coords, order=0, mode='nearest')
        return result.reshape(out_h, out_w)

    mask = np.isnan(data)
    if not mask.any():
        result = _scipy_map_coords(data, coords, order=order, mode='nearest')
        return result.reshape(out_h, out_w)

    filled = np.where(mask, 0.0, data)
    weights = (~mask).astype(data.dtype)

    z_data = _scipy_map_coords(filled, coords, order=order, mode='nearest')
    z_wt = _scipy_map_coords(weights, coords, order=order, mode='nearest')

    # Gate on majority weight: an output pixel is valid only when more
    # than half of the resampling kernel weight came from valid input
    # pixels.  This rejects pixels lit only by cubic-kernel sidelobes
    # leaking small positive weight from a single neighbour.
    result = np.where(z_wt > 0.5,
                      z_data / np.maximum(z_wt, 1e-10),
                      np.nan)
    return result.reshape(out_h, out_w)


# -- NaN-aware interpolation (CuPy) -----------------------------------------

def _nan_aware_interp_cupy(data, out_h, out_w, order):
    """CuPy variant of :func:`_nan_aware_interp_np`."""
    from cupyx.scipy.ndimage import map_coordinates as _cupy_map_coords

    iy = cupy.asarray(_block_centered_coords(data.shape[0], out_h))
    ix = cupy.asarray(_block_centered_coords(data.shape[1], out_w))
    yy, xx = cupy.meshgrid(iy, ix, indexing='ij')
    coords = cupy.array([yy.ravel(), xx.ravel()])

    if order == 0:
        result = _cupy_map_coords(data, coords, order=0, mode='nearest')
        return result.reshape(out_h, out_w)

    mask = cupy.isnan(data)
    if not mask.any():
        result = _cupy_map_coords(data, coords, order=order, mode='nearest')
        return result.reshape(out_h, out_w)

    filled = cupy.where(mask, 0.0, data)
    weights = (~mask).astype(data.dtype)

    z_data = _cupy_map_coords(filled, coords, order=order, mode='nearest')
    z_wt = _cupy_map_coords(weights, coords, order=order, mode='nearest')

    # Majority-weight gate (see _nan_aware_interp_np for rationale).
    result = cupy.where(z_wt > 0.5,
                        z_data / cupy.maximum(z_wt, 1e-10),
                        cupy.nan)
    return result.reshape(out_h, out_w)


# -- Block-aggregation kernels (NumPy, numba) --------------------------------

@ngjit
def _agg_mean(data, out_h, out_w):
    h, w = data.shape
    out = np.empty((out_h, out_w), dtype=np.float64)
    for oy in range(out_h):
        y0 = int(oy * h / out_h)
        y1 = max(y0 + 1, int((oy + 1) * h / out_h))
        for ox in range(out_w):
            x0 = int(ox * w / out_w)
            x1 = max(x0 + 1, int((ox + 1) * w / out_w))
            total = 0.0
            count = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    v = data[y, x]
                    if not np.isnan(v):
                        total += v
                        count += 1
            out[oy, ox] = total / count if count > 0 else np.nan
    return out


@ngjit
def _agg_min(data, out_h, out_w):
    h, w = data.shape
    out = np.empty((out_h, out_w), dtype=np.float64)
    for oy in range(out_h):
        y0 = int(oy * h / out_h)
        y1 = max(y0 + 1, int((oy + 1) * h / out_h))
        for ox in range(out_w):
            x0 = int(ox * w / out_w)
            x1 = max(x0 + 1, int((ox + 1) * w / out_w))
            best = np.inf
            found = False
            for y in range(y0, y1):
                for x in range(x0, x1):
                    v = data[y, x]
                    if not np.isnan(v) and v < best:
                        best = v
                        found = True
            out[oy, ox] = best if found else np.nan
    return out


@ngjit
def _agg_max(data, out_h, out_w):
    h, w = data.shape
    out = np.empty((out_h, out_w), dtype=np.float64)
    for oy in range(out_h):
        y0 = int(oy * h / out_h)
        y1 = max(y0 + 1, int((oy + 1) * h / out_h))
        for ox in range(out_w):
            x0 = int(ox * w / out_w)
            x1 = max(x0 + 1, int((ox + 1) * w / out_w))
            best = -np.inf
            found = False
            for y in range(y0, y1):
                for x in range(x0, x1):
                    v = data[y, x]
                    if not np.isnan(v) and v > best:
                        best = v
                        found = True
            out[oy, ox] = best if found else np.nan
    return out


@ngjit
def _agg_median(data, out_h, out_w):
    h, w = data.shape
    out = np.empty((out_h, out_w), dtype=np.float64)
    for oy in range(out_h):
        y0 = int(oy * h / out_h)
        y1 = max(y0 + 1, int((oy + 1) * h / out_h))
        for ox in range(out_w):
            x0 = int(ox * w / out_w)
            x1 = max(x0 + 1, int((ox + 1) * w / out_w))
            buf = np.empty((y1 - y0) * (x1 - x0), dtype=np.float64)
            n = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    v = data[y, x]
                    if not np.isnan(v):
                        buf[n] = v
                        n += 1
            if n == 0:
                out[oy, ox] = np.nan
            else:
                s = np.sort(buf[:n])
                if n % 2 == 1:
                    out[oy, ox] = s[n // 2]
                else:
                    out[oy, ox] = (s[n // 2 - 1] + s[n // 2]) / 2.0
    return out


@ngjit
def _agg_mode(data, out_h, out_w):
    h, w = data.shape
    out = np.empty((out_h, out_w), dtype=np.float64)
    for oy in range(out_h):
        y0 = int(oy * h / out_h)
        y1 = max(y0 + 1, int((oy + 1) * h / out_h))
        for ox in range(out_w):
            x0 = int(ox * w / out_w)
            x1 = max(x0 + 1, int((ox + 1) * w / out_w))
            buf = np.empty((y1 - y0) * (x1 - x0), dtype=np.float64)
            n = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    v = data[y, x]
                    if not np.isnan(v):
                        buf[n] = v
                        n += 1
            if n == 0:
                out[oy, ox] = np.nan
                continue
            s = np.sort(buf[:n])
            best_val = s[0]
            best_cnt = 1
            cur_val = s[0]
            cur_cnt = 1
            for i in range(1, n):
                if s[i] == cur_val:
                    cur_cnt += 1
                else:
                    if cur_cnt > best_cnt:
                        best_cnt = cur_cnt
                        best_val = cur_val
                    cur_val = s[i]
                    cur_cnt = 1
            if cur_cnt > best_cnt:
                best_val = cur_val
            out[oy, ox] = best_val
    return out


_AGG_FUNCS = {
    'average': _agg_mean,
    'min': _agg_min,
    'max': _agg_max,
    'median': _agg_median,
    'mode': _agg_mode,
}


# -- Dask block helpers ------------------------------------------------------
#
# Interpolation uses map_coordinates with *global* coordinate mapping so
# that results are identical regardless of chunk layout.  Each block
# receives the cumulative chunk boundaries and computes which global
# output pixels it is responsible for, maps them back to global input
# coordinates, then converts to local (within-block) coordinates.


def _interp_block_np(block, global_in_h, global_in_w,
                     global_out_h, global_out_w,
                     cum_in_y, cum_in_x, cum_out_y, cum_out_x,
                     depth, order, block_info=None):
    """Interpolate one (possibly overlapped) numpy block."""
    yi, xi = block_info[0]['chunk-location']
    target_h = int(cum_out_y[yi + 1] - cum_out_y[yi])
    target_w = int(cum_out_x[xi + 1] - cum_out_x[xi])

    block = block.astype(np.float64)

    # Global output pixel indices for this chunk
    oy = np.arange(cum_out_y[yi], cum_out_y[yi + 1], dtype=np.float64)
    ox = np.arange(cum_out_x[xi], cum_out_x[xi + 1], dtype=np.float64)

    # Map to global input coordinates using block-centered formula
    iy = (oy + 0.5) * (global_in_h / global_out_h) - 0.5
    ix = (ox + 0.5) * (global_in_w / global_out_w) - 0.5

    # Convert to local block coordinates (overlap shifts the origin)
    iy_local = iy - (cum_in_y[yi] - depth)
    ix_local = ix - (cum_in_x[xi] - depth)

    yy, xx = np.meshgrid(iy_local, ix_local, indexing='ij')
    coords = np.array([yy.ravel(), xx.ravel()])

    # NaN-aware interpolation
    mask = np.isnan(block)
    if order == 0 or not mask.any():
        result = _scipy_map_coords(block, coords, order=order, mode='nearest')
    else:
        filled = np.where(mask, 0.0, block)
        weights = (~mask).astype(np.float64)
        z_data = _scipy_map_coords(filled, coords, order=order, mode='nearest')
        z_wt = _scipy_map_coords(weights, coords, order=order, mode='nearest')
        # Majority-weight gate (see _nan_aware_interp_np for rationale).
        result = np.where(z_wt > 0.5,
                          z_data / np.maximum(z_wt, 1e-10), np.nan)

    return result.reshape(target_h, target_w).astype(np.float32)


def _interp_block_cupy(block, global_in_h, global_in_w,
                       global_out_h, global_out_w,
                       cum_in_y, cum_in_x, cum_out_y, cum_out_x,
                       depth, order, block_info=None):
    """CuPy variant of :func:`_interp_block_np`."""
    from cupyx.scipy.ndimage import map_coordinates as _cupy_map_coords

    yi, xi = block_info[0]['chunk-location']
    target_h = int(cum_out_y[yi + 1] - cum_out_y[yi])
    target_w = int(cum_out_x[xi + 1] - cum_out_x[xi])

    block = block.astype(cupy.float64)

    oy = cupy.arange(int(cum_out_y[yi]), int(cum_out_y[yi + 1]),
                     dtype=cupy.float64)
    ox = cupy.arange(int(cum_out_x[xi]), int(cum_out_x[xi + 1]),
                     dtype=cupy.float64)

    # Map to global input coordinates using block-centered formula
    iy = (oy + 0.5) * (global_in_h / global_out_h) - 0.5
    ix = (ox + 0.5) * (global_in_w / global_out_w) - 0.5

    iy_local = iy - float(cum_in_y[yi] - depth)
    ix_local = ix - float(cum_in_x[xi] - depth)

    yy, xx = cupy.meshgrid(iy_local, ix_local, indexing='ij')
    coords = cupy.array([yy.ravel(), xx.ravel()])

    mask = cupy.isnan(block)
    if order == 0 or not mask.any():
        result = _cupy_map_coords(block, coords, order=order, mode='nearest')
    else:
        filled = cupy.where(mask, 0.0, block)
        weights = (~mask).astype(cupy.float64)
        z_data = _cupy_map_coords(filled, coords, order=order, mode='nearest')
        z_wt = _cupy_map_coords(weights, coords, order=order, mode='nearest')
        # Majority-weight gate (see _nan_aware_interp_np for rationale).
        result = cupy.where(z_wt > 0.5,
                            z_data / cupy.maximum(z_wt, 1e-10), cupy.nan)

    return result.reshape(target_h, target_w).astype(cupy.float32)


def _agg_block_np(block, method, global_in_h, global_in_w,
                  global_out_h, global_out_w,
                  cum_in_y, cum_in_x, cum_out_y, cum_out_x,
                  depth_y, depth_x, block_info=None):
    """Block-aggregate one (possibly overlapped) numpy chunk."""
    yi, xi = block_info[0]['chunk-location']
    target_h = int(cum_out_y[yi + 1] - cum_out_y[yi])
    target_w = int(cum_out_x[xi + 1] - cum_out_x[xi])

    block = block.astype(np.float64)
    # The overlapped block starts depth pixels before the original chunk
    in_y0 = int(cum_in_y[yi]) - depth_y
    in_x0 = int(cum_in_x[xi]) - depth_x
    func = _AGG_FUNCS[method]

    out = np.empty((target_h, target_w), dtype=np.float64)
    for lo_y in range(target_h):
        go_y = int(cum_out_y[yi]) + lo_y
        gy0 = int(go_y * global_in_h / global_out_h) - in_y0
        gy1 = max(gy0 + 1, int((go_y + 1) * global_in_h / global_out_h) - in_y0)
        for lo_x in range(target_w):
            go_x = int(cum_out_x[xi]) + lo_x
            gx0 = int(go_x * global_in_w / global_out_w) - in_x0
            gx1 = max(gx0 + 1, int((go_x + 1) * global_in_w / global_out_w) - in_x0)
            sub = block[gy0:gy1, gx0:gx1]
            out[lo_y, lo_x] = func(sub, 1, 1)[0, 0]

    return out.astype(np.float32)


def _agg_block_cupy(block, method, global_in_h, global_in_w,
                    global_out_h, global_out_w,
                    cum_in_y, cum_in_x, cum_out_y, cum_out_x,
                    depth_y, depth_x, block_info=None):
    """Block-aggregate one cupy chunk (falls back to CPU)."""
    cpu = cupy.asnumpy(block)
    result = _agg_block_np(
        cpu, method, global_in_h, global_in_w,
        global_out_h, global_out_w,
        cum_in_y, cum_in_x, cum_out_y, cum_out_x,
        depth_y, depth_x, block_info=block_info,
    )
    return cupy.asarray(result)


# -- Per-backend runners -----------------------------------------------------

def _run_numpy(data, scale_y, scale_x, method):
    data = data.astype(np.float64)
    out_h, out_w = _output_shape(*data.shape, scale_y, scale_x)

    if method in INTERP_METHODS:
        return _nan_aware_interp_np(data, out_h, out_w,
                                    INTERP_METHODS[method]).astype(np.float32)

    return _AGG_FUNCS[method](data, out_h, out_w).astype(np.float32)


def _run_cupy(data, scale_y, scale_x, method):
    data = data.astype(cupy.float64)
    out_h, out_w = _output_shape(*data.shape, scale_y, scale_x)

    if method in INTERP_METHODS:
        return _nan_aware_interp_cupy(data, out_h, out_w,
                                      INTERP_METHODS[method]).astype(cupy.float32)

    # Aggregate: GPU reshape+reduce for integer factors, CPU fallback otherwise
    fy, fx = data.shape[0] / out_h, data.shape[1] / out_w
    if (fy == int(fy) and fx == int(fx)
            and method in ('average', 'min', 'max')):
        fy, fx = int(fy), int(fx)
        trimmed = data[:out_h * fy, :out_w * fx]
        reshaped = trimmed.reshape(out_h, fy, out_w, fx)
        reducer = {'average': cupy.nanmean,
                   'min': cupy.nanmin,
                   'max': cupy.nanmax}[method]
        return reducer(reshaped, axis=(1, 3)).astype(cupy.float32)

    cpu = cupy.asnumpy(data)
    return cupy.asarray(
        _AGG_FUNCS[method](cpu, out_h, out_w).astype(np.float32)
    )


def _min_chunksize_for_scale(scale):
    """Minimum input chunk size so that no output chunk is zero after rounding."""
    if scale >= 1.0:
        return 1
    # c > 1/s guarantees round((k+1)*c*s) - round(k*c*s) >= 1 for all k.
    return int(1.0 / scale) + 1


def _ensure_min_chunksize(data, min_size):
    """Rechunk *data* so every chunk is at least *min_size* pixels wide."""
    import math
    new = {}
    for ax in range(data.ndim):
        if any(c < min_size for c in data.chunks[ax]):
            total = sum(data.chunks[ax])
            # Find chunk size where ALL chunks (including last) >= min_size
            n = max(1, total // min_size)
            cs = math.ceil(total / n)
            while n > 1:
                remainder = total - cs * (total // cs)
                if remainder == 0 or remainder >= min_size:
                    break
                n -= 1
                cs = math.ceil(total / n)
            new[ax] = cs
    return data.rechunk(new) if new else data


def _run_dask_numpy(data, scale_y, scale_x, method):
    data = data.astype(np.float64)
    meta = np.array((), dtype=np.float32)

    if method in INTERP_METHODS:
        order = INTERP_METHODS[method]
        depth = _INTERP_DEPTH[method]

        min_size = max(2 * depth + 1,
                       _min_chunksize_for_scale(scale_y),
                       _min_chunksize_for_scale(scale_x))
        data = _ensure_min_chunksize(data, min_size)

        global_in_h = int(sum(data.chunks[0]))
        global_in_w = int(sum(data.chunks[1]))
        global_out_h, global_out_w = _output_shape(
            global_in_h, global_in_w, scale_y, scale_x)
        out_y = _output_chunks(data.chunks[0], scale_y)
        out_x = _output_chunks(data.chunks[1], scale_x)

        cum_in_y = np.cumsum([0] + list(data.chunks[0]))
        cum_in_x = np.cumsum([0] + list(data.chunks[1]))
        cum_out_y = np.cumsum([0] + list(out_y))
        cum_out_x = np.cumsum([0] + list(out_x))

        src = data
        if depth > 0:
            from dask.array.overlap import overlap as _add_overlap
            src = _add_overlap(data, depth={0: depth, 1: depth},
                               boundary='nearest')

        fn = partial(_interp_block_np,
                     global_in_h=global_in_h, global_in_w=global_in_w,
                     global_out_h=global_out_h, global_out_w=global_out_w,
                     cum_in_y=cum_in_y, cum_in_x=cum_in_x,
                     cum_out_y=cum_out_y, cum_out_x=cum_out_x,
                     depth=depth, order=order)
        return da.map_blocks(fn, src, chunks=(out_y, out_x),
                             dtype=np.float32, meta=meta)

    import math
    min_size = max(_min_chunksize_for_scale(scale_y),
                   _min_chunksize_for_scale(scale_x))
    data = _ensure_min_chunksize(data, min_size)

    global_in_h = int(sum(data.chunks[0]))
    global_in_w = int(sum(data.chunks[1]))
    global_out_h, global_out_w = _output_shape(
        global_in_h, global_in_w, scale_y, scale_x)
    out_y = _output_chunks(data.chunks[0], scale_y)
    out_x = _output_chunks(data.chunks[1], scale_x)
    cum_in_y = np.cumsum([0] + list(data.chunks[0]))
    cum_in_x = np.cumsum([0] + list(data.chunks[1]))
    cum_out_y = np.cumsum([0] + list(out_y))
    cum_out_x = np.cumsum([0] + list(out_x))

    # Aggregate windows can cross chunk boundaries; add overlap.
    depth_y = math.ceil(global_in_h / global_out_h)
    depth_x = math.ceil(global_in_w / global_out_w)
    data = _ensure_min_chunksize(data, max(2 * depth_y + 1, 2 * depth_x + 1))
    # Recompute in case rechunk changed layout
    if data.chunks[0] != tuple(cum_in_y[1:] - cum_in_y[:-1]):
        cum_in_y = np.cumsum([0] + list(data.chunks[0]))
        cum_in_x = np.cumsum([0] + list(data.chunks[1]))
        out_y = _output_chunks(data.chunks[0], scale_y)
        out_x = _output_chunks(data.chunks[1], scale_x)
        cum_out_y = np.cumsum([0] + list(out_y))
        cum_out_x = np.cumsum([0] + list(out_x))

    from dask.array.overlap import overlap as _add_overlap
    src = _add_overlap(data, depth={0: depth_y, 1: depth_x},
                       boundary='nearest')

    fn = partial(_agg_block_np, method=method,
                 global_in_h=global_in_h, global_in_w=global_in_w,
                 global_out_h=global_out_h, global_out_w=global_out_w,
                 cum_in_y=cum_in_y, cum_in_x=cum_in_x,
                 cum_out_y=cum_out_y, cum_out_x=cum_out_x,
                 depth_y=depth_y, depth_x=depth_x)
    return da.map_blocks(fn, src, chunks=(out_y, out_x),
                         dtype=np.float32, meta=meta)


def _run_dask_cupy(data, scale_y, scale_x, method):
    data = data.astype(cupy.float64)
    meta = cupy.array((), dtype=cupy.float32)

    if method in INTERP_METHODS:
        order = INTERP_METHODS[method]
        depth = _INTERP_DEPTH[method]

        min_size = max(2 * depth + 1,
                       _min_chunksize_for_scale(scale_y),
                       _min_chunksize_for_scale(scale_x))
        data = _ensure_min_chunksize(data, min_size)

        global_in_h = int(sum(data.chunks[0]))
        global_in_w = int(sum(data.chunks[1]))
        global_out_h, global_out_w = _output_shape(
            global_in_h, global_in_w, scale_y, scale_x)
        out_y = _output_chunks(data.chunks[0], scale_y)
        out_x = _output_chunks(data.chunks[1], scale_x)

        cum_in_y = np.cumsum([0] + list(data.chunks[0]))
        cum_in_x = np.cumsum([0] + list(data.chunks[1]))
        cum_out_y = np.cumsum([0] + list(out_y))
        cum_out_x = np.cumsum([0] + list(out_x))

        src = data
        if depth > 0:
            from dask.array.overlap import overlap as _add_overlap
            src = _add_overlap(data, depth={0: depth, 1: depth},
                               boundary='nearest')

        fn = partial(_interp_block_cupy,
                     global_in_h=global_in_h, global_in_w=global_in_w,
                     global_out_h=global_out_h, global_out_w=global_out_w,
                     cum_in_y=cum_in_y, cum_in_x=cum_in_x,
                     cum_out_y=cum_out_y, cum_out_x=cum_out_x,
                     depth=depth, order=order)
        return da.map_blocks(fn, src, chunks=(out_y, out_x),
                             dtype=cupy.float32, meta=meta)

    import math
    min_size = max(_min_chunksize_for_scale(scale_y),
                   _min_chunksize_for_scale(scale_x))
    data = _ensure_min_chunksize(data, min_size)

    global_in_h = int(sum(data.chunks[0]))
    global_in_w = int(sum(data.chunks[1]))
    global_out_h, global_out_w = _output_shape(
        global_in_h, global_in_w, scale_y, scale_x)

    depth_y = math.ceil(global_in_h / global_out_h)
    depth_x = math.ceil(global_in_w / global_out_w)
    data = _ensure_min_chunksize(data, max(2 * depth_y + 1, 2 * depth_x + 1))

    out_y = _output_chunks(data.chunks[0], scale_y)
    out_x = _output_chunks(data.chunks[1], scale_x)
    cum_in_y = np.cumsum([0] + list(data.chunks[0]))
    cum_in_x = np.cumsum([0] + list(data.chunks[1]))
    cum_out_y = np.cumsum([0] + list(out_y))
    cum_out_x = np.cumsum([0] + list(out_x))

    from dask.array.overlap import overlap as _add_overlap
    src = _add_overlap(data, depth={0: depth_y, 1: depth_x},
                       boundary='nearest')

    fn = partial(_agg_block_cupy, method=method,
                 global_in_h=global_in_h, global_in_w=global_in_w,
                 global_out_h=global_out_h, global_out_w=global_out_w,
                 cum_in_y=cum_in_y, cum_in_x=cum_in_x,
                 cum_out_y=cum_out_y, cum_out_x=cum_out_x,
                 depth_y=depth_y, depth_x=depth_x)
    return da.map_blocks(fn, src, chunks=(out_y, out_x),
                         dtype=cupy.float32, meta=meta)


# -- Public API --------------------------------------------------------------

@supports_dataset
def resample(
    agg,
    scale_factor=None,
    target_resolution=None,
    method='nearest',
    name='resample',
):
    """Change raster resolution without changing its CRS.

    Exactly one of *scale_factor* or *target_resolution* must be given.

    Parameters
    ----------
    agg : xarray.DataArray
        Input raster (2-D).
    scale_factor : float or (float, float), optional
        Multiplicative factor applied to the number of pixels.
        ``0.5`` halves the pixel count (doubles the cell size);
        ``2.0`` doubles the pixel count (halves the cell size).
        A two-element tuple sets ``(scale_y, scale_x)`` independently.
    target_resolution : float, optional
        Desired cell size in the same units as the raster coordinates.
        Both axes are set to this resolution.
    method : str, default ``'nearest'``
        Resampling algorithm.  Interpolation methods (``'nearest'``,
        ``'bilinear'``, ``'cubic'``) work for both upsampling and
        downsampling.  Aggregation methods (``'average'``, ``'min'``,
        ``'max'``, ``'median'``, ``'mode'``) only support downsampling
        (scale_factor <= 1).
    name : str, default ``'resample'``
        Name for the output DataArray.

    Returns
    -------
    xarray.DataArray
        Resampled raster with updated coordinates, ``res`` attribute,
        and float32 dtype.
    """
    _validate_raster(agg, func_name='resample', name='agg')

    if method not in ALL_METHODS:
        raise ValueError(
            f"method must be one of {sorted(ALL_METHODS)}, got {method!r}"
        )

    # -- resolve scale factors -----------------------------------------------
    if (scale_factor is None) == (target_resolution is None):
        raise ValueError(
            "Exactly one of scale_factor or target_resolution must be given"
        )

    if target_resolution is not None:
        if agg.shape[-2] < 2 or agg.shape[-1] < 2:
            raise ValueError(
                "target_resolution requires at least 2 pixels per dimension"
            )
        res_x, res_y = calc_res(agg)
        if isinstance(target_resolution, (tuple, list)):
            scale_y = abs(res_y) / target_resolution[0]
            scale_x = abs(res_x) / target_resolution[1]
        else:
            scale_y = abs(res_y) / target_resolution
            scale_x = abs(res_x) / target_resolution
    elif isinstance(scale_factor, (tuple, list)):
        scale_y, scale_x = float(scale_factor[0]), float(scale_factor[1])
    else:
        scale_y = scale_x = float(scale_factor)

    if scale_y <= 0 or scale_x <= 0:
        raise ValueError(
            f"Scale factors must be positive, got ({scale_y}, {scale_x})"
        )

    if method in AGGREGATE_METHODS and (scale_y > 1.0 or scale_x > 1.0):
        raise ValueError(
            f"Aggregate method {method!r} only supports downsampling "
            f"(scale_factor <= 1.0)"
        )

    # -- fast path: identity -------------------------------------------------
    if scale_y == 1.0 and scale_x == 1.0:
        out = agg.copy()
        out.name = name
        return out

    # -- memory guard for eager backends ------------------------------------
    # Dask paths build per-chunk allocations lazily (chunk size already
    # bounds peak memory). The eager numpy and cupy paths allocate the
    # full (out_h, out_w) buffer up front and need an explicit guard.
    in_h, in_w = agg.shape[-2:]
    out_h, out_w = _output_shape(in_h, in_w, scale_y, scale_x)

    is_dask = da is not None and isinstance(agg.data, da.Array)
    is_cupy = cupy is not None and isinstance(agg.data, cupy.ndarray)
    if not is_dask:
        if is_cupy:
            _check_resample_gpu_memory(out_h, out_w)
        else:
            _check_resample_memory(out_h, out_w)

    # -- dispatch to backend -------------------------------------------------
    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        cupy_func=_run_cupy,
        dask_func=_run_dask_numpy,
        dask_cupy_func=_run_dask_cupy,
    )
    result_data = mapper(agg)(agg.data, scale_y, scale_x, method)

    # -- build output coordinates -------------------------------------------
    ydim, xdim = agg.dims[-2], agg.dims[-1]
    y_vals = np.asarray(agg[ydim].values, dtype=np.float64)
    x_vals = np.asarray(agg[xdim].values, dtype=np.float64)

    def _new_coords(vals, n_out):
        if len(vals) > 1:
            half_first = (vals[1] - vals[0]) / 2
            half_last = (vals[-1] - vals[-2]) / 2
        else:
            half_first = half_last = 0.5
        edge_start = vals[0] - half_first
        edge_end = vals[-1] + half_last
        px = (edge_end - edge_start) / n_out
        return np.linspace(edge_start + px / 2, edge_end - px / 2, n_out), px

    new_y, py = _new_coords(y_vals, out_h)
    new_x, px = _new_coords(x_vals, out_w)

    new_attrs = dict(agg.attrs)
    new_attrs['res'] = (abs(px), abs(py))

    result = xr.DataArray(
        result_data,
        name=name,
        dims=agg.dims,
        coords={ydim: new_y, xdim: new_x},
        attrs=new_attrs,
    )

    for dim in (ydim, xdim):
        if dim in agg.coords and agg[dim].attrs:
            result[dim].attrs = dict(agg[dim].attrs)

    return result
