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
# one float64 working buffer (8 B) plus a float64 output cell (8 B) in
# the worst case. scipy.ndimage.map_coordinates also allocates a
# temporary of the same size during higher-order spline evaluation; the
# 0.5 * available bound below leaves room for that.
_BYTES_PER_OUTPUT_CELL = 16


# -- Working / output dtype selection ----------------------------------------

def _working_dtype(input_dtype):
    """Pick the working float dtype for resampling.

    float64 inputs stay in float64 to preserve precision; everything else
    (smaller floats, integers, bool) uses float32.
    """
    dt = np.dtype(input_dtype)
    if dt.kind == 'f' and dt.itemsize >= 8:
        return np.float64
    return np.float32


def _output_dtype(input_dtype):
    """Pick the output dtype for resampling.

    Float inputs keep their dtype. Integer / bool inputs return float32
    because NaN-sentinel resampling needs a float type.
    """
    dt = np.dtype(input_dtype)
    if dt.kind == 'f':
        return dt.type
    return np.float32


def _maybe_astype(arr, dtype):
    """astype copy that no-ops when already at the requested dtype."""
    return arr if arr.dtype == np.dtype(dtype) else arr.astype(dtype)


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


# -- Block-aggregation kernels for dask chunks -------------------------------
#
# These mirror the eager `_agg_mean / _agg_min / ...` family but compute
# per-pixel windows from the *global* input/output geometry and a chunk
# offset, rather than from the local block shape.  The whole chunk runs
# inside a single jitted call, instead of one numba dispatch per output
# pixel as the previous `func(sub, 1, 1)[0, 0]` loop did.
#
# Window bounds for output pixel `go` (a *global* output index):
#     gy0 = int(go * global_in_h / global_out_h) - in_y0
#     gy1 = max(gy0 + 1,
#               int((go + 1) * global_in_h / global_out_h) - in_y0)
# where `in_y0` is the global input index of the chunk's first row
# (negative if `_add_overlap` extended the chunk past the input edge).

@ngjit
def _agg_block_mean_nb(data, target_h, target_w,
                       go_y0, go_x0,
                       global_in_h, global_in_w,
                       global_out_h, global_out_w,
                       in_y0, in_x0):
    out = np.empty((target_h, target_w), dtype=np.float64)
    for lo_y in range(target_h):
        go_y = go_y0 + lo_y
        gy0 = int(go_y * global_in_h / global_out_h) - in_y0
        gy1 = int((go_y + 1) * global_in_h / global_out_h) - in_y0
        if gy1 < gy0 + 1:
            gy1 = gy0 + 1
        for lo_x in range(target_w):
            go_x = go_x0 + lo_x
            gx0 = int(go_x * global_in_w / global_out_w) - in_x0
            gx1 = int((go_x + 1) * global_in_w / global_out_w) - in_x0
            if gx1 < gx0 + 1:
                gx1 = gx0 + 1
            total = 0.0
            count = 0
            for y in range(gy0, gy1):
                for x in range(gx0, gx1):
                    v = data[y, x]
                    if not np.isnan(v):
                        total += v
                        count += 1
            out[lo_y, lo_x] = total / count if count > 0 else np.nan
    return out


@ngjit
def _agg_block_min_nb(data, target_h, target_w,
                      go_y0, go_x0,
                      global_in_h, global_in_w,
                      global_out_h, global_out_w,
                      in_y0, in_x0):
    out = np.empty((target_h, target_w), dtype=np.float64)
    for lo_y in range(target_h):
        go_y = go_y0 + lo_y
        gy0 = int(go_y * global_in_h / global_out_h) - in_y0
        gy1 = int((go_y + 1) * global_in_h / global_out_h) - in_y0
        if gy1 < gy0 + 1:
            gy1 = gy0 + 1
        for lo_x in range(target_w):
            go_x = go_x0 + lo_x
            gx0 = int(go_x * global_in_w / global_out_w) - in_x0
            gx1 = int((go_x + 1) * global_in_w / global_out_w) - in_x0
            if gx1 < gx0 + 1:
                gx1 = gx0 + 1
            best = np.inf
            found = False
            for y in range(gy0, gy1):
                for x in range(gx0, gx1):
                    v = data[y, x]
                    if not np.isnan(v) and v < best:
                        best = v
                        found = True
            out[lo_y, lo_x] = best if found else np.nan
    return out


@ngjit
def _agg_block_max_nb(data, target_h, target_w,
                      go_y0, go_x0,
                      global_in_h, global_in_w,
                      global_out_h, global_out_w,
                      in_y0, in_x0):
    out = np.empty((target_h, target_w), dtype=np.float64)
    for lo_y in range(target_h):
        go_y = go_y0 + lo_y
        gy0 = int(go_y * global_in_h / global_out_h) - in_y0
        gy1 = int((go_y + 1) * global_in_h / global_out_h) - in_y0
        if gy1 < gy0 + 1:
            gy1 = gy0 + 1
        for lo_x in range(target_w):
            go_x = go_x0 + lo_x
            gx0 = int(go_x * global_in_w / global_out_w) - in_x0
            gx1 = int((go_x + 1) * global_in_w / global_out_w) - in_x0
            if gx1 < gx0 + 1:
                gx1 = gx0 + 1
            best = -np.inf
            found = False
            for y in range(gy0, gy1):
                for x in range(gx0, gx1):
                    v = data[y, x]
                    if not np.isnan(v) and v > best:
                        best = v
                        found = True
            out[lo_y, lo_x] = best if found else np.nan
    return out


@ngjit
def _agg_block_median_nb(data, target_h, target_w,
                         go_y0, go_x0,
                         global_in_h, global_in_w,
                         global_out_h, global_out_w,
                         in_y0, in_x0):
    out = np.empty((target_h, target_w), dtype=np.float64)
    for lo_y in range(target_h):
        go_y = go_y0 + lo_y
        gy0 = int(go_y * global_in_h / global_out_h) - in_y0
        gy1 = int((go_y + 1) * global_in_h / global_out_h) - in_y0
        if gy1 < gy0 + 1:
            gy1 = gy0 + 1
        for lo_x in range(target_w):
            go_x = go_x0 + lo_x
            gx0 = int(go_x * global_in_w / global_out_w) - in_x0
            gx1 = int((go_x + 1) * global_in_w / global_out_w) - in_x0
            if gx1 < gx0 + 1:
                gx1 = gx0 + 1
            buf = np.empty((gy1 - gy0) * (gx1 - gx0), dtype=np.float64)
            n = 0
            for y in range(gy0, gy1):
                for x in range(gx0, gx1):
                    v = data[y, x]
                    if not np.isnan(v):
                        buf[n] = v
                        n += 1
            if n == 0:
                out[lo_y, lo_x] = np.nan
            else:
                s = np.sort(buf[:n])
                if n % 2 == 1:
                    out[lo_y, lo_x] = s[n // 2]
                else:
                    out[lo_y, lo_x] = (s[n // 2 - 1] + s[n // 2]) / 2.0
    return out


@ngjit
def _agg_block_mode_nb(data, target_h, target_w,
                       go_y0, go_x0,
                       global_in_h, global_in_w,
                       global_out_h, global_out_w,
                       in_y0, in_x0):
    out = np.empty((target_h, target_w), dtype=np.float64)
    for lo_y in range(target_h):
        go_y = go_y0 + lo_y
        gy0 = int(go_y * global_in_h / global_out_h) - in_y0
        gy1 = int((go_y + 1) * global_in_h / global_out_h) - in_y0
        if gy1 < gy0 + 1:
            gy1 = gy0 + 1
        for lo_x in range(target_w):
            go_x = go_x0 + lo_x
            gx0 = int(go_x * global_in_w / global_out_w) - in_x0
            gx1 = int((go_x + 1) * global_in_w / global_out_w) - in_x0
            if gx1 < gx0 + 1:
                gx1 = gx0 + 1
            buf = np.empty((gy1 - gy0) * (gx1 - gx0), dtype=np.float64)
            n = 0
            for y in range(gy0, gy1):
                for x in range(gx0, gx1):
                    v = data[y, x]
                    if not np.isnan(v):
                        buf[n] = v
                        n += 1
            if n == 0:
                out[lo_y, lo_x] = np.nan
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
            out[lo_y, lo_x] = best_val
    return out


_AGG_BLOCK_FUNCS = {
    'average': _agg_block_mean_nb,
    'min': _agg_block_min_nb,
    'max': _agg_block_max_nb,
    'median': _agg_block_median_nb,
    'mode': _agg_block_mode_nb,
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
                     depth, order, work_dtype, out_dtype, block_info=None):
    """Interpolate one (possibly overlapped) numpy block."""
    yi, xi = block_info[0]['chunk-location']
    target_h = int(cum_out_y[yi + 1] - cum_out_y[yi])
    target_w = int(cum_out_x[xi + 1] - cum_out_x[xi])

    block = _maybe_astype(block, work_dtype)

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
        weights = (~mask).astype(block.dtype)
        z_data = _scipy_map_coords(filled, coords, order=order, mode='nearest')
        z_wt = _scipy_map_coords(weights, coords, order=order, mode='nearest')
        # Majority-weight gate (see _nan_aware_interp_np for rationale).
        result = np.where(z_wt > 0.5,
                          z_data / np.maximum(z_wt, 1e-10), np.nan)

    return _maybe_astype(result.reshape(target_h, target_w), out_dtype)


def _interp_block_cupy(block, global_in_h, global_in_w,
                       global_out_h, global_out_w,
                       cum_in_y, cum_in_x, cum_out_y, cum_out_x,
                       depth, order, work_dtype, out_dtype, block_info=None):
    """CuPy variant of :func:`_interp_block_np`."""
    from cupyx.scipy.ndimage import map_coordinates as _cupy_map_coords

    yi, xi = block_info[0]['chunk-location']
    target_h = int(cum_out_y[yi + 1] - cum_out_y[yi])
    target_w = int(cum_out_x[xi + 1] - cum_out_x[xi])

    if block.dtype != cupy.dtype(work_dtype):
        block = block.astype(work_dtype)

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
        weights = (~mask).astype(block.dtype)
        z_data = _cupy_map_coords(filled, coords, order=order, mode='nearest')
        z_wt = _cupy_map_coords(weights, coords, order=order, mode='nearest')
        # Majority-weight gate (see _nan_aware_interp_np for rationale).
        result = cupy.where(z_wt > 0.5,
                            z_data / cupy.maximum(z_wt, 1e-10), cupy.nan)

    result = result.reshape(target_h, target_w)
    if result.dtype != cupy.dtype(out_dtype):
        result = result.astype(out_dtype)
    return result


def _agg_block_np(block, method, global_in_h, global_in_w,
                  global_out_h, global_out_w,
                  cum_in_y, cum_in_x, cum_out_y, cum_out_x,
                  depth_y, depth_x, out_dtype, block_info=None):
    """Block-aggregate one (possibly overlapped) numpy chunk.

    Runs the entire chunk inside one numba dispatch via the
    `_agg_block_*_nb` kernels.  Earlier versions called a 1x1 jitted
    aggregate per output pixel, which scaled badly for large rasters.
    """
    yi, xi = block_info[0]['chunk-location']
    target_h = int(cum_out_y[yi + 1] - cum_out_y[yi])
    target_w = int(cum_out_x[xi + 1] - cum_out_x[xi])

    # _AGG_FUNCS kernels are @ngjit-compiled with hard-coded float64
    # working buffers; cast accordingly so numba dispatch matches.
    block = _maybe_astype(block, np.float64)
    # The overlapped block starts depth pixels before the original chunk
    in_y0 = int(cum_in_y[yi]) - depth_y
    in_x0 = int(cum_in_x[xi]) - depth_x
    go_y0 = int(cum_out_y[yi])
    go_x0 = int(cum_out_x[xi])

    kernel = _AGG_BLOCK_FUNCS[method]
    out = kernel(block, target_h, target_w,
                 go_y0, go_x0,
                 int(global_in_h), int(global_in_w),
                 int(global_out_h), int(global_out_w),
                 in_y0, in_x0)
    return _maybe_astype(out, out_dtype)


def _agg_block_cupy(block, method, global_in_h, global_in_w,
                    global_out_h, global_out_w,
                    cum_in_y, cum_in_x, cum_out_y, cum_out_x,
                    depth_y, depth_x, out_dtype, block_info=None):
    """Block-aggregate one cupy chunk (falls back to CPU)."""
    cpu = cupy.asnumpy(block)
    result = _agg_block_np(
        cpu, method, global_in_h, global_in_w,
        global_out_h, global_out_w,
        cum_in_y, cum_in_x, cum_out_y, cum_out_x,
        depth_y, depth_x, out_dtype, block_info=block_info,
    )
    return cupy.asarray(result)


# -- Per-backend runners -----------------------------------------------------

def _run_numpy(data, scale_y, scale_x, method):
    work_dt = _working_dtype(data.dtype)
    out_dt = _output_dtype(data.dtype)
    data = _maybe_astype(data, work_dt)
    out_h, out_w = _output_shape(*data.shape, scale_y, scale_x)

    if method in INTERP_METHODS:
        result = _nan_aware_interp_np(data, out_h, out_w,
                                      INTERP_METHODS[method])
        return _maybe_astype(result, out_dt)

    result = _AGG_FUNCS[method](data, out_h, out_w)
    return _maybe_astype(result, out_dt)


def _run_cupy(data, scale_y, scale_x, method):
    work_dt = _working_dtype(data.dtype)
    out_dt = _output_dtype(data.dtype)
    data = data if data.dtype == cupy.dtype(work_dt) else data.astype(work_dt)
    out_h, out_w = _output_shape(*data.shape, scale_y, scale_x)

    if method in INTERP_METHODS:
        result = _nan_aware_interp_cupy(data, out_h, out_w,
                                        INTERP_METHODS[method])
        return result if result.dtype == cupy.dtype(out_dt) else result.astype(out_dt)

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
        result = reducer(reshaped, axis=(1, 3))
        return result if result.dtype == cupy.dtype(out_dt) else result.astype(out_dt)

    cpu = cupy.asnumpy(data)
    return cupy.asarray(
        _maybe_astype(_AGG_FUNCS[method](cpu, out_h, out_w), out_dt)
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
    work_dt = _working_dtype(data.dtype)
    out_dt = _output_dtype(data.dtype)
    if data.dtype != np.dtype(work_dt):
        data = data.astype(work_dt)
    meta = np.array((), dtype=out_dt)

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
                     depth=depth, order=order,
                     work_dtype=work_dt, out_dtype=out_dt)
        return da.map_blocks(fn, src, chunks=(out_y, out_x),
                             dtype=out_dt, meta=meta)

    import math
    # Aggregate windows can cross chunk boundaries; size chunks to satisfy
    # both the scale-driven minimum and the depth-driven minimum in one pass,
    # then build the cumulative arrays once.
    global_in_h = int(sum(data.chunks[0]))
    global_in_w = int(sum(data.chunks[1]))
    global_out_h, global_out_w = _output_shape(
        global_in_h, global_in_w, scale_y, scale_x)
    depth_y = math.ceil(global_in_h / global_out_h)
    depth_x = math.ceil(global_in_w / global_out_w)
    min_size = max(_min_chunksize_for_scale(scale_y),
                   _min_chunksize_for_scale(scale_x),
                   2 * depth_y + 1, 2 * depth_x + 1)
    data = _ensure_min_chunksize(data, min_size)

    out_y = _output_chunks(data.chunks[0], scale_y)
    out_x = _output_chunks(data.chunks[1], scale_x)
    cum_in_y = np.cumsum([0] + list(data.chunks[0]))
    cum_in_x = np.cumsum([0] + list(data.chunks[1]))
    cum_out_y = np.cumsum([0] + list(out_y))
    cum_out_x = np.cumsum([0] + list(out_x))

    # boundary=np.nan keeps overlap padding from contaminating the aggregate
    # at the global edges. The kernels skip NaN inputs and return NaN for
    # empty windows, so the padded region is ignored naturally.
    from dask.array.overlap import overlap as _add_overlap
    src = _add_overlap(data, depth={0: depth_y, 1: depth_x},
                       boundary=np.nan)

    fn = partial(_agg_block_np, method=method,
                 global_in_h=global_in_h, global_in_w=global_in_w,
                 global_out_h=global_out_h, global_out_w=global_out_w,
                 cum_in_y=cum_in_y, cum_in_x=cum_in_x,
                 cum_out_y=cum_out_y, cum_out_x=cum_out_x,
                 depth_y=depth_y, depth_x=depth_x,
                 out_dtype=out_dt)
    return da.map_blocks(fn, src, chunks=(out_y, out_x),
                         dtype=out_dt, meta=meta)


def _run_dask_cupy(data, scale_y, scale_x, method):
    work_dt = _working_dtype(data.dtype)
    out_dt = _output_dtype(data.dtype)
    if data.dtype != cupy.dtype(work_dt):
        data = data.astype(work_dt)
    meta = cupy.array((), dtype=out_dt)

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
                     depth=depth, order=order,
                     work_dtype=work_dt, out_dtype=out_dt)
        return da.map_blocks(fn, src, chunks=(out_y, out_x),
                             dtype=out_dt, meta=meta)

    import math
    # Aggregate windows can cross chunk boundaries; size chunks to satisfy
    # both the scale-driven minimum and the depth-driven minimum in one pass,
    # then build the cumulative arrays once.
    global_in_h = int(sum(data.chunks[0]))
    global_in_w = int(sum(data.chunks[1]))
    global_out_h, global_out_w = _output_shape(
        global_in_h, global_in_w, scale_y, scale_x)
    depth_y = math.ceil(global_in_h / global_out_h)
    depth_x = math.ceil(global_in_w / global_out_w)
    min_size = max(_min_chunksize_for_scale(scale_y),
                   _min_chunksize_for_scale(scale_x),
                   2 * depth_y + 1, 2 * depth_x + 1)
    data = _ensure_min_chunksize(data, min_size)

    out_y = _output_chunks(data.chunks[0], scale_y)
    out_x = _output_chunks(data.chunks[1], scale_x)
    cum_in_y = np.cumsum([0] + list(data.chunks[0]))
    cum_in_x = np.cumsum([0] + list(data.chunks[1]))
    cum_out_y = np.cumsum([0] + list(out_y))
    cum_out_x = np.cumsum([0] + list(out_x))

    # boundary=np.nan keeps overlap padding from contaminating the aggregate
    # at the global edges. The kernels skip NaN inputs and return NaN for
    # empty windows, so the padded region is ignored naturally.
    from dask.array.overlap import overlap as _add_overlap
    src = _add_overlap(data, depth={0: depth_y, 1: depth_x},
                       boundary=cupy.nan)

    fn = partial(_agg_block_cupy, method=method,
                 global_in_h=global_in_h, global_in_w=global_in_w,
                 global_out_h=global_out_h, global_out_w=global_out_w,
                 cum_in_y=cum_in_y, cum_in_x=cum_in_x,
                 cum_out_y=cum_out_y, cum_out_x=cum_out_x,
                 depth_y=depth_y, depth_x=depth_x,
                 out_dtype=out_dt)
    return da.map_blocks(fn, src, chunks=(out_y, out_x),
                         dtype=out_dt, meta=meta)


# -- Public API --------------------------------------------------------------

def _resolve_nodata(agg, nodata):
    """Resolve the input-side nodata sentinel.

    Explicit *nodata* wins. Otherwise fall back to ``_FillValue`` then
    ``nodata`` in ``agg.attrs``. Returns ``None`` when no sentinel was
    found (the caller skips the masking step).

    NaN sentinels are returned as NaN so the caller can branch on
    ``np.isnan`` rather than ``==`` (which never matches NaN).
    """
    if nodata is None:
        for key in ('_FillValue', 'nodata'):
            v = agg.attrs.get(key)
            if v is not None:
                nodata = v
                break
    if nodata is None:
        return None
    nd = float(nodata)
    if np.isinf(nd):
        raise ValueError(f"nodata must be finite or NaN, got {nodata!r}")
    return nd


def _apply_nodata_mask(agg, nodata):
    """Return a float copy of *agg* with sentinel pixels replaced by NaN.

    Works for numpy, cupy, dask+numpy, and dask+cupy backings via
    xarray's ``.where`` (which dispatches per backend).
    """
    if nodata is None:
        return agg
    # Promote to float so NaN can be stored. xr.where keeps the backend.
    # Integer / bool inputs become float32 (consistent with _output_dtype).
    if not np.issubdtype(agg.dtype, np.floating):
        agg = agg.astype(np.float32)
    if np.isnan(nodata):
        return agg  # already-NaN sentinels need no replacement
    return agg.where(agg != nodata)


@supports_dataset
def resample(
    agg,
    scale_factor=None,
    target_resolution=None,
    method='nearest',
    nodata=None,
    name='resample',
):
    """Change raster resolution without changing its CRS.

    Exactly one of *scale_factor* or *target_resolution* must be given.

    Parameters
    ----------
    agg : xarray.DataArray
        Input raster. 2-D ``(y, x)`` or 3-D ``(band, y, x)``. For 3-D
        inputs each band is resampled independently and the leading
        non-spatial coordinate is preserved.
    scale_factor : float or (float, float), optional
        Multiplicative factor applied to the number of pixels.
        ``0.5`` halves the pixel count (doubles the cell size);
        ``2.0`` doubles the pixel count (halves the cell size).
        A two-element tuple sets ``(scale_y, scale_x)`` independently.
    target_resolution : float or (float, float), optional
        Desired cell size in the same units as the raster coordinates.
        A scalar sets both axes to the same resolution; a 2-tuple sets
        ``(res_y, res_x)`` independently.
    method : str, default ``'nearest'``
        Resampling algorithm.  Interpolation methods (``'nearest'``,
        ``'bilinear'``, ``'cubic'``) work for both upsampling and
        downsampling.  Aggregation methods (``'average'``, ``'min'``,
        ``'max'``, ``'median'``, ``'mode'``) only support downsampling
        (scale_factor <= 1).
    nodata : float, optional
        Sentinel value in the input that should be treated as missing.
        Input pixels equal to *nodata* are replaced with NaN before
        resampling. When ``None``, falls back to ``agg.attrs['_FillValue']``
        then ``agg.attrs['nodata']``. The output uses NaN as the sentinel
        regardless of the input convention.
    name : str, default ``'resample'``
        Name for the output DataArray.

    Returns
    -------
    xarray.DataArray
        Resampled raster with updated coordinates and ``res`` attribute.
        Output dtype matches the input float dtype (float32 or float64);
        integer inputs return float32 since NaN-sentinel resampling
        requires a float type.
    """
    _validate_raster(agg, func_name='resample', name='agg', ndim=(2, 3))

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

    # -- nodata: replace sentinels with NaN before resampling ----------------
    nd_resolved = _resolve_nodata(agg, nodata)
    has_nodata = nd_resolved is not None
    if has_nodata:
        agg = _apply_nodata_mask(agg, nd_resolved)

    # -- fast path: identity -------------------------------------------------
    if scale_y == 1.0 and scale_x == 1.0:
        out = agg.copy()
        out.name = name
        # When nodata was applied, advertise NaN as the new sentinel.
        if has_nodata:
            out.attrs['_FillValue'] = float('nan')
        return out

    # -- 3D: dispatch per band ----------------------------------------------
    if agg.ndim == 3:
        leading_dim = agg.dims[0]
        bands = []
        for i in range(agg.sizes[leading_dim]):
            band_2d = agg.isel({leading_dim: i})
            band_out = resample(
                band_2d,
                scale_factor=scale_factor,
                target_resolution=target_resolution,
                method=method,
                # Pass NaN so the recursive call short-circuits masking
                # (we already applied the mask on the 3D input above) and
                # ignores the original attrs sentinel.
                nodata=float('nan'),
                name=name,
            )
            bands.append(band_out)
        # Stack along the leading dim. concat preserves the per-band
        # coordinate when each input has it.
        result = xr.concat(bands, dim=leading_dim)
        # concat may reorder dims; transpose to the original layout.
        result = result.transpose(*agg.dims)
        result.name = name
        # Carry across input attrs (concat picks the first; merge with input).
        new_attrs = dict(agg.attrs)
        new_attrs.update(bands[0].attrs)  # res from per-band resample
        if has_nodata:
            new_attrs['_FillValue'] = float('nan')
        result.attrs = new_attrs
        # Preserve the leading-dim coordinate if it was on the input.
        if leading_dim in agg.coords:
            result = result.assign_coords({leading_dim: agg.coords[leading_dim]})
        return result

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
        coords = np.linspace(edge_start + px / 2, edge_end - px / 2, n_out)
        return coords, px, edge_start, edge_end

    new_y, py, y_edge_start, y_edge_end = _new_coords(y_vals, out_h)
    new_x, px, x_edge_start, x_edge_end = _new_coords(x_vals, out_w)

    new_attrs = dict(agg.attrs)
    new_attrs['res'] = (abs(px), abs(py))
    if has_nodata:
        new_attrs['_FillValue'] = float('nan')

    # Refresh `transform` if the input had one. The rasterio 6-tuple is
    # (res_x, 0.0, left, 0.0, -res_y, top). `top` is the upper edge of
    # the first row, which is `y_edge_start` when y is descending and
    # `y_edge_end` when y is ascending. `left` is the lower edge of the
    # first column, which is `x_edge_start` when x is ascending and
    # `x_edge_end` when x is descending.
    if 'transform' in agg.attrs:
        out_res_x = abs(px)
        out_res_y = abs(py)
        top = y_edge_start if y_vals[0] > y_vals[-1] else y_edge_end
        left = x_edge_start if x_vals[0] < x_vals[-1] else x_edge_end
        new_attrs['transform'] = (
            out_res_x, 0.0, left, 0.0, -out_res_y, top,
        )

    # Resample currently emits float32 with NaN as the missing-data
    # sentinel regardless of input dtype. If the input declared a
    # different sentinel via `_FillValue` or `nodatavals`, replace the
    # value with NaN so the metadata matches the actual data. Leave the
    # keys absent when the input did not have them.
    if '_FillValue' in agg.attrs:
        new_attrs['_FillValue'] = float('nan')
    if 'nodatavals' in agg.attrs:
        old = agg.attrs['nodatavals']
        new_attrs['nodatavals'] = tuple(float('nan') for _ in old)

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
