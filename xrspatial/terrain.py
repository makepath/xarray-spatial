from __future__ import annotations

# std lib
import math
import threading
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

# 3rd-party
import numpy as np
import xarray as xr
from numba import njit, prange

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

try:
    import dask
    import dask.array as da
except ImportError:
    dask = None
    da = None

# local modules
from xrspatial.utils import (ArrayTypeFunctionMapping, _validate_raster, cuda_args,
                             get_dataarray_resolution, not_implemented_func)

# Scratch-buffer budget: 5-7 same-shape float32 arrays plus headroom.
# 24 bytes/pixel covers (data, x, y, warp_x, warp_y, weight) at 4 B each.
_SCRATCH_BYTES_PER_PIXEL = 24
_GPU_SCRATCH_BYTES_PER_PIXEL = 24


def _available_memory_bytes():
    """Best-effort estimate of available host memory in bytes."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    return 2 * 1024 ** 3


def _available_gpu_memory_bytes():
    """Free GPU memory in bytes, or 0 when CUDA is unavailable."""
    try:
        import cupy as _cp
        free, _total = _cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return 0


def _check_memory(height, width):
    """Raise MemoryError if scratch buffers would exceed 50% of RAM."""
    required = int(height) * int(width) * _SCRATCH_BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"generate_terrain on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of scratch memory but only "
            f"~{available / 1e9:.1f} GB is available.  Use a "
            f"dask-backed DataArray for out-of-core processing."
        )


def _check_gpu_memory(height, width):
    """Raise MemoryError if scratch buffers would exceed 50% of free VRAM."""
    available = _available_gpu_memory_bytes()
    if available <= 0:
        return
    required = int(height) * int(width) * _GPU_SCRATCH_BYTES_PER_PIXEL
    if required > 0.5 * available:
        raise MemoryError(
            f"generate_terrain on a {height}x{width} grid requires "
            f"~{required / 1e9:.1f} GB of GPU scratch memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )

from .perlin import _make_perm_table, _perlin, _perlin_gpu, _perlin_gpu_xy
from .worley import _worley_cpu, _worley_numpy_xy, _worley_gpu, _worley_gpu_xy
from xrspatial.utils import _dask_task_name_kwargs


def _scale(value, old_range, new_range):
    d = (value - old_range[0]) / (old_range[1] - old_range[0])
    return d * (new_range[1] - new_range[0]) + new_range[0]


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------

# Fused, multithreaded reimplementation of the perlin octave loop.  The plain
# numpy version in _gen_terrain calls _perlin once per octave, and each _perlin
# call allocates ~20 full-size temporaries (floor, gather, fade, lerp ...).  At
# 16 octaves that is hundreds of array allocations.  The kernel below collapses
# warp + fbm/ridged + cube into one per-pixel scalar computation parallelised
# over rows with prange, so there are no per-octave temporaries.  Output matches
# the numpy path to float32 epsilon.  Worley blending is not handled here (it
# needs a global min/max pass); _gen_terrain keeps the numpy path for that.

# Numba parallel=True kernels must not be launched concurrently from multiple
# Python threads: the default 'workqueue' threading layer is not threadsafe and
# aborts the process (SIGABRT on macOS) when two host threads enter a parallel
# region at once.  _terrain_dask_numpy calls _gen_terrain_fast per chunk under
# dask's threaded scheduler, so the kernel launch is serialized behind this
# lock.  Same hazard and fix as the reproject kernels (#3141).
_PARALLEL_KERNEL_LOCK = threading.Lock()


@njit(nogil=True, inline='always')
def _fade_scalar(t):
    t3 = t * t * t
    t4 = t3 * t
    t5 = t4 * t
    return 6 * t5 - 15 * t4 + 10 * t3


@njit(nogil=True, inline='always')
def _grad_scalar(h, x, y):
    hv = h & 3
    sel = (hv >> 1) & 1  # 0 -> y axis, 1 -> x axis
    u = x * sel + y * (1 - sel)
    return u * (1 - 2 * (hv & 1))


@njit(nogil=True, inline='always')
def _perlin_point(p, x, y):
    xfl = np.float32(math.floor(x))
    yfl = np.float32(math.floor(y))
    xi = np.int32(xfl) & 255
    yi = np.int32(yfl) & 255
    xf = x - xfl
    yf = y - yfl
    u = _fade_scalar(xf)
    v = _fade_scalar(yf)
    one = np.float32(1.0)
    n00 = _grad_scalar(p[p[xi] + yi], xf, yf)
    n01 = _grad_scalar(p[p[xi] + yi + 1], xf, yf - one)
    n11 = _grad_scalar(p[p[xi + 1] + yi + 1], xf - one, yf - one)
    n10 = _grad_scalar(p[p[xi + 1] + yi], xf - one, yf)
    x1 = n00 + u * (n10 - n00)
    x2 = n01 + u * (n11 - n01)
    return x1 + v * (x2 - x1)


@njit(parallel=True, nogil=True, cache=True)
def _fused_terrain_numpy(out, linx, liny, perms, octaves,
                         persistence, lacunarity, norm, is_ridged,
                         do_warp, warp_px, warp_py, warp_octaves,
                         warp_norm, warp_strength):
    height = out.shape[0]
    width = out.shape[1]
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for i in prange(height):
        yb0 = liny[i]
        for j in range(width):
            xb = linx[j]
            yb = yb0

            # --- domain warping ---
            if do_warp:
                wx = zero
                wy = zero
                amp = one
                freq = one
                for wi in range(warp_octaves):
                    wx += _perlin_point(warp_px[wi], xb * freq, yb * freq) * amp
                    wy += _perlin_point(warp_py[wi], xb * freq, yb * freq) * amp
                    amp *= persistence
                    freq *= lacunarity
                xb = xb + (wx / warp_norm) * warp_strength
                yb = yb + (wy / warp_norm) * warp_strength

            # --- octave noise loop ---
            acc = zero
            amp = one
            freq = one
            if is_ridged:
                weight = one
                for o in range(octaves):
                    nval = _perlin_point(perms[o], xb * freq, yb * freq)
                    nval = one - abs(nval)
                    nval = nval * nval
                    nval = nval * weight
                    weight = min(max(nval, zero), one)
                    acc += nval * amp
                    amp *= persistence
                    freq *= lacunarity
            else:  # fbm
                for o in range(octaves):
                    acc += _perlin_point(perms[o], xb * freq, yb * freq) * amp
                    amp *= persistence
                    freq *= lacunarity

            val = acc / norm
            out[i, j] = val * val * val


def _gen_terrain_fast(height, width, seed, x_range, y_range, octaves,
                      lacunarity, persistence, noise_mode,
                      warp_strength, warp_octaves):
    """Fast path for _gen_terrain when worley blending is off (the default)."""
    linx = np.linspace(
        x_range[0], x_range[1], width, endpoint=False, dtype=np.float32
    )
    liny = np.linspace(
        y_range[0], y_range[1], height, endpoint=False, dtype=np.float32
    )

    perms = np.stack([_make_perm_table(seed + i) for i in range(octaves)])
    norm = float(sum(persistence ** i for i in range(octaves)))

    do_warp = warp_strength > 0
    if do_warp:
        warp_px = np.stack(
            [_make_perm_table(seed + 100 + wi) for wi in range(warp_octaves)]
        )
        warp_py = np.stack(
            [_make_perm_table(seed + 200 + wi) for wi in range(warp_octaves)]
        )
        warp_norm = float(sum(persistence ** i for i in range(warp_octaves)))
    else:
        # numba needs concrete array types even on the unused branch
        warp_px = np.zeros((1, 512), dtype=np.int32)
        warp_py = np.zeros((1, 512), dtype=np.int32)
        warp_norm = 1.0

    out = np.empty((height, width), dtype=np.float32)
    with _PARALLEL_KERNEL_LOCK:
        _fused_terrain_numpy(
            out, linx, liny, perms, octaves,
            np.float32(persistence), np.float32(lacunarity), np.float32(norm),
            noise_mode == 'ridged', do_warp, warp_px, warp_py,
            warp_octaves, np.float32(warp_norm), np.float32(warp_strength),
        )
    return out


def _gen_terrain(height_map, seed, x_range=(0, 1), y_range=(0, 1),
                 octaves=16, lacunarity=2.0, persistence=0.5,
                 noise_mode='fbm', warp_strength=0.0, warp_octaves=4,
                 worley_blend=0.0, worley_seed=None,
                 worley_norm_range=None):
    height, width = height_map.shape

    # Fast path: when worley blending is off (the default), the whole warp +
    # octave + cube pipeline runs in one fused, multithreaded kernel instead of
    # the per-octave numpy loop below.  Worley needs a global min/max pass, so
    # it stays on the numpy path.
    if worley_blend <= 0:
        return _gen_terrain_fast(
            height, width, seed, x_range, y_range, octaves,
            lacunarity, persistence, noise_mode, warp_strength, warp_octaves,
        )

    linx = np.linspace(
        x_range[0], x_range[1], width, endpoint=False, dtype=np.float32
    )
    liny = np.linspace(
        y_range[0], y_range[1], height, endpoint=False, dtype=np.float32
    )
    x, y = np.meshgrid(linx, liny)

    # --- domain warping ---
    if warp_strength > 0:
        warp_x = np.zeros_like(x)
        warp_y = np.zeros_like(y)
        for wi in range(warp_octaves):
            w_amp = persistence ** wi
            w_freq = lacunarity ** wi
            p_wx = _make_perm_table(seed + 100 + wi)
            p_wy = _make_perm_table(seed + 200 + wi)
            warp_x += _perlin(p_wx, x * w_freq, y * w_freq) * w_amp
            warp_y += _perlin(p_wy, x * w_freq, y * w_freq) * w_amp
        warp_norm = sum(persistence ** i for i in range(warp_octaves))
        warp_x /= warp_norm
        warp_y /= warp_norm
        x = x + warp_x * warp_strength
        y = y + warp_y * warp_strength

    # --- octave noise loop ---
    norm = sum(persistence ** i for i in range(octaves))

    if noise_mode == 'ridged':
        weight = np.ones((height, width), dtype=np.float32)
        for i in range(octaves):
            amp = persistence ** i
            freq = lacunarity ** i
            p = _make_perm_table(seed + i)
            noise = _perlin(p, x * freq, y * freq)
            noise = 1.0 - np.abs(noise)
            noise = noise * noise
            noise *= weight
            weight = np.clip(noise, 0, 1)
            height_map += noise * amp
    else:  # fbm
        for i in range(octaves):
            amp = persistence ** i
            freq = lacunarity ** i
            p = _make_perm_table(seed + i)
            noise = _perlin(p, x * freq, y * freq) * amp
            height_map += noise

    height_map /= norm

    # --- worley blending ---
    if worley_blend > 0:
        if worley_seed is None:
            worley_seed = seed + 1000
        w_p = _make_perm_table(worley_seed)
        w_noise = _worley_numpy_xy(w_p, x, y)
        if worley_norm_range is not None:
            w_min, w_max = worley_norm_range
            w_ptp = w_max - w_min
        else:
            w_min = w_noise.min()
            w_ptp = w_noise.max() - w_min
        if w_ptp > 0:
            w_noise = (w_noise - w_min) / w_ptp
        height_map = height_map * (1 - worley_blend) + w_noise * worley_blend

    height_map = height_map ** 3
    return height_map


def _terrain_numpy(data, seed, x_range_scaled, y_range_scaled, zfactor,
                   octaves, lacunarity, persistence, noise_mode,
                   warp_strength, warp_octaves,
                   worley_blend, worley_seed):
    _check_memory(data.shape[0], data.shape[1])
    # zeros_like (not data * 0) so an all-NaN template (e.g. from
    # from_template()) is overwritten instead of propagating NaN
    # everywhere (NaN * 0 == NaN).
    data = np.zeros_like(data)
    data[:] = _gen_terrain(
        data, seed, x_range=x_range_scaled, y_range=y_range_scaled,
        octaves=octaves, lacunarity=lacunarity, persistence=persistence,
        noise_mode=noise_mode, warp_strength=warp_strength,
        warp_octaves=warp_octaves, worley_blend=worley_blend,
        worley_seed=worley_seed,
    )

    data = np.clip(data, -1, 1)
    data = (data + 1) / 2
    data[data < 0.3] = 0  # create water
    data *= zfactor

    return data


# ---------------------------------------------------------------------------
# dask + numpy backend
# ---------------------------------------------------------------------------

def _terrain_dask_numpy(data, seed, x_range_scaled, y_range_scaled, zfactor,
                        octaves, lacunarity, persistence, noise_mode,
                        warp_strength, warp_octaves,
                        worley_blend, worley_seed):
    """Inline the entire terrain computation into a single map_blocks call.

    Each chunk computes its own warp + octave + worley pipeline independently
    using _gen_terrain.  This keeps the dask graph shallow (one task per chunk)
    instead of building per-octave intermediate arrays that blow up memory.
    """
    height, width = data.shape

    # The chunk functions below regenerate every cell from coordinates and
    # never read the incoming block values.  Map over an uninitialised
    # skeleton with the same shape/chunking so the (NaN-filled) template the
    # caller passed in — e.g. from_template()'s da.full — drops out of the
    # graph instead of being memset and then thrown away.
    skeleton = da.empty_like(data, dtype=np.float32)

    # --- worley pre-pass: get global min/max to avoid per-chunk seams ---
    w_norm_range = None
    if worley_blend > 0:
        _w_seed = worley_seed if worley_seed is not None else seed + 1000

        def _chunk_worley_raw(block, block_info=None):
            info = block_info[0]
            y_start, y_end = info['array-location'][0]
            x_start, x_end = info['array-location'][1]
            x0 = x_range_scaled[0] + (x_range_scaled[1] - x_range_scaled[0]) * x_start / width
            x1 = x_range_scaled[0] + (x_range_scaled[1] - x_range_scaled[0]) * x_end / width
            y0 = y_range_scaled[0] + (y_range_scaled[1] - y_range_scaled[0]) * y_start / height
            y1 = y_range_scaled[0] + (y_range_scaled[1] - y_range_scaled[0]) * y_end / height

            h, w = block.shape
            linx = np.linspace(x0, x1, w, endpoint=False, dtype=np.float32)
            liny = np.linspace(y0, y1, h, endpoint=False, dtype=np.float32)
            x_arr, y_arr = np.meshgrid(linx, liny)

            if warp_strength > 0:
                warp_x = np.zeros((h, w), dtype=np.float32)
                warp_y = np.zeros((h, w), dtype=np.float32)
                for wi in range(warp_octaves):
                    w_amp = persistence ** wi
                    w_freq = lacunarity ** wi
                    p_wx = _make_perm_table(seed + 100 + wi)
                    p_wy = _make_perm_table(seed + 200 + wi)
                    warp_x += _perlin(p_wx, x_arr * w_freq, y_arr * w_freq) * w_amp
                    warp_y += _perlin(p_wy, x_arr * w_freq, y_arr * w_freq) * w_amp
                warp_norm = sum(persistence ** i for i in range(warp_octaves))
                warp_x /= warp_norm
                warp_y /= warp_norm
                x_arr = x_arr + warp_x * warp_strength
                y_arr = y_arr + warp_y * warp_strength

            w_p = _make_perm_table(_w_seed)
            return _worley_numpy_xy(w_p, x_arr, y_arr)

        raw_worley = da.map_blocks(
            _chunk_worley_raw, skeleton, dtype=np.float32,
            meta=np.array((), dtype=np.float32),
        )
        (raw_worley,) = dask.persist(raw_worley)
        w_min, w_max = dask.compute(da.min(raw_worley), da.max(raw_worley))
        w_norm_range = (float(w_min), float(w_max))
        del raw_worley

    # _chunk_terrain ignores the input block values — it creates its own
    # np.zeros internally.  We pass `data` only so map_blocks inherits its
    # chunk structure and block_info coordinates.
    def _chunk_terrain(block, block_info=None):
        info = block_info[0]
        y_start, y_end = info['array-location'][0]
        x_start, x_end = info['array-location'][1]
        x0 = x_range_scaled[0] + (x_range_scaled[1] - x_range_scaled[0]) * x_start / width
        x1 = x_range_scaled[0] + (x_range_scaled[1] - x_range_scaled[0]) * x_end / width
        y0 = y_range_scaled[0] + (y_range_scaled[1] - y_range_scaled[0]) * y_start / height
        y1 = y_range_scaled[0] + (y_range_scaled[1] - y_range_scaled[0]) * y_end / height

        out = np.zeros(block.shape, dtype=np.float32)
        out[:] = _gen_terrain(
            out, seed, x_range=(x0, x1), y_range=(y0, y1),
            octaves=octaves, lacunarity=lacunarity,
            persistence=persistence, noise_mode=noise_mode,
            warp_strength=warp_strength, warp_octaves=warp_octaves,
            worley_blend=worley_blend, worley_seed=worley_seed,
            worley_norm_range=w_norm_range,
        )
        np.clip(out, -1, 1, out=out)
        out = (out + 1) / 2
        out[out < 0.3] = 0
        out *= zfactor
        return out

    return da.map_blocks(_chunk_terrain, skeleton, dtype=np.float32,
                         meta=np.array((), dtype=np.float32),
                         **_dask_task_name_kwargs('xrspatial.terrain'))


# ---------------------------------------------------------------------------
# cupy (GPU) backend
# ---------------------------------------------------------------------------

def _terrain_gpu(height_map, seed, x_range=(0, 1), y_range=(0, 1),
                 octaves=16, lacunarity=2.0, persistence=0.5,
                 noise_mode='fbm', warp_strength=0.0, warp_octaves=4,
                 worley_blend=0.0, worley_seed=None,
                 worley_norm_range=None):

    h, w = height_map.shape
    griddim, blockdim = cuda_args(height_map.shape)
    noise = cupy.empty_like(height_map, dtype=cupy.float32)

    # coordinate arrays (needed if warping or worley with xy kernels)
    use_xy_kernel = (warp_strength > 0)
    x_arr = None
    y_arr = None

    if use_xy_kernel or worley_blend > 0:
        linx = cupy.linspace(x_range[0], x_range[1], w,
                             endpoint=False, dtype=cupy.float32)
        liny = cupy.linspace(y_range[0], y_range[1], h,
                             endpoint=False, dtype=cupy.float32)
        y_arr, x_arr = cupy.meshgrid(liny, linx, indexing='ij')

    # --- domain warping ---
    # pre-allocate reusable buffers for scaled coordinates (GPU)
    if use_xy_kernel:
        scaled_x = cupy.empty_like(x_arr)
        scaled_y = cupy.empty_like(y_arr)

    if warp_strength > 0:
        warp_x = cupy.zeros((h, w), dtype=cupy.float32)
        warp_y = cupy.zeros((h, w), dtype=cupy.float32)
        tmp = cupy.empty_like(noise)

        for wi in range(warp_octaves):
            w_amp = persistence ** wi
            w_freq = lacunarity ** wi
            p_wx = cupy.asarray(_make_perm_table(seed + 100 + wi))
            p_wy = cupy.asarray(_make_perm_table(seed + 200 + wi))

            cupy.multiply(x_arr, w_freq, out=scaled_x)
            cupy.multiply(y_arr, w_freq, out=scaled_y)

            _perlin_gpu_xy[griddim, blockdim](
                p_wx, scaled_x, scaled_y, 1.0, tmp
            )
            warp_x += tmp * w_amp

            _perlin_gpu_xy[griddim, blockdim](
                p_wy, scaled_x, scaled_y, 1.0, tmp
            )
            warp_y += tmp * w_amp

        warp_norm = sum(persistence ** i for i in range(warp_octaves))
        warp_x /= warp_norm
        warp_y /= warp_norm
        x_arr = x_arr + warp_x * warp_strength
        y_arr = y_arr + warp_y * warp_strength

    # --- octave loop ---
    norm = sum(persistence ** i for i in range(octaves))

    if noise_mode == 'ridged':
        weight = cupy.ones((h, w), dtype=cupy.float32)
        for i in range(octaves):
            amp = persistence ** i
            freq = lacunarity ** i
            p = cupy.asarray(_make_perm_table(seed + i))

            if use_xy_kernel:
                cupy.multiply(x_arr, freq, out=scaled_x)
                cupy.multiply(y_arr, freq, out=scaled_y)
                _perlin_gpu_xy[griddim, blockdim](
                    p, scaled_x, scaled_y, 1.0, noise
                )
            else:
                _perlin_gpu[griddim, blockdim](
                    p, x_range[0] * freq, x_range[1] * freq,
                    y_range[0] * freq, y_range[1] * freq, 1.0, noise
                )

            noise_val = 1.0 - cupy.abs(noise)
            noise_val = noise_val * noise_val
            noise_val *= weight
            weight = cupy.clip(noise_val, 0, 1)
            height_map += noise_val * amp
    else:  # fbm
        for i in range(octaves):
            amp = persistence ** i
            freq = lacunarity ** i
            p = cupy.asarray(_make_perm_table(seed + i))

            if use_xy_kernel:
                cupy.multiply(x_arr, freq, out=scaled_x)
                cupy.multiply(y_arr, freq, out=scaled_y)
                _perlin_gpu_xy[griddim, blockdim](
                    p, scaled_x, scaled_y, amp, noise
                )
            else:
                _perlin_gpu[griddim, blockdim](
                    p, x_range[0] * freq, x_range[1] * freq,
                    y_range[0] * freq, y_range[1] * freq, amp, noise
                )

            height_map += noise

    height_map /= norm

    # --- worley blending ---
    if worley_blend > 0:
        if worley_seed is None:
            worley_seed = seed + 1000
        w_p = cupy.asarray(_make_perm_table(worley_seed))
        w_noise = cupy.empty_like(height_map)

        if x_arr is not None:
            _worley_gpu_xy[griddim, blockdim](w_p, x_arr, y_arr, w_noise)
        else:
            _worley_gpu[griddim, blockdim](
                w_p, x_range[0], x_range[1], y_range[0], y_range[1], w_noise
            )

        if worley_norm_range is not None:
            w_min, w_max = worley_norm_range
            w_ptp = w_max - w_min
        else:
            w_min = cupy.amin(w_noise)
            w_ptp = cupy.amax(w_noise) - w_min
        if float(w_ptp) > 0:
            w_noise = (w_noise - w_min) / w_ptp
        height_map = height_map * (1 - worley_blend) + w_noise * worley_blend

    height_map = height_map ** 3
    return height_map


def _terrain_cupy(data, seed, x_range_scaled, y_range_scaled, zfactor,
                  octaves, lacunarity, persistence, noise_mode,
                  warp_strength, warp_octaves,
                  worley_blend, worley_seed):
    _check_gpu_memory(data.shape[0], data.shape[1])
    # zeros_like (not data * 0) so an all-NaN template (e.g. from
    # from_template()) is overwritten instead of propagating NaN
    # everywhere (NaN * 0 == NaN).
    data = cupy.zeros_like(data)
    data[:] = _terrain_gpu(
        data, seed, x_range=x_range_scaled, y_range=y_range_scaled,
        octaves=octaves, lacunarity=lacunarity, persistence=persistence,
        noise_mode=noise_mode, warp_strength=warp_strength,
        warp_octaves=warp_octaves, worley_blend=worley_blend,
        worley_seed=worley_seed,
    )
    data = cupy.clip(data, -1, 1)
    data[:] = (data + 1) / 2
    data[data < 0.3] = 0  # create water
    data *= zfactor
    return data


# ---------------------------------------------------------------------------
# dask + cupy backend
# ---------------------------------------------------------------------------

def _terrain_dask_cupy(data, seed, x_range_scaled, y_range_scaled, zfactor,
                       octaves, lacunarity, persistence, noise_mode,
                       warp_strength, warp_octaves,
                       worley_blend, worley_seed):
    """Inline the entire terrain computation into a single map_blocks call.

    Each chunk computes its own warp + octave + worley pipeline independently
    using the GPU kernels.  When worley blending is enabled, a pre-pass
    computes global worley noise range so normalization is consistent across
    chunk boundaries.
    """
    height, width = data.shape

    # See _terrain_dask_numpy: the chunk functions regenerate every cell from
    # coordinates and never read block values, so map over an uninitialised
    # skeleton instead of materialising (and discarding) the template's fill.
    skeleton = da.empty_like(data, dtype=cupy.float32)

    # --- worley pre-pass: get global min/max to avoid per-chunk seams ---
    w_norm_range = None
    if worley_blend > 0:
        _w_seed = worley_seed if worley_seed is not None else seed + 1000

        def _chunk_worley_raw(block, block_info=None):
            """Compute raw worley noise for this chunk (with warp)."""
            info = block_info[0]
            y_start, y_end = info['array-location'][0]
            x_start, x_end = info['array-location'][1]
            x0 = x_range_scaled[0] + (x_range_scaled[1] - x_range_scaled[0]) * x_start / width
            x1 = x_range_scaled[0] + (x_range_scaled[1] - x_range_scaled[0]) * x_end / width
            y0 = y_range_scaled[0] + (y_range_scaled[1] - y_range_scaled[0]) * y_start / height
            y1 = y_range_scaled[0] + (y_range_scaled[1] - y_range_scaled[0]) * y_end / height

            h, w = block.shape
            griddim, blockdim = cuda_args(block.shape)

            linx = cupy.linspace(x0, x1, w, endpoint=False, dtype=cupy.float32)
            liny = cupy.linspace(y0, y1, h, endpoint=False, dtype=cupy.float32)
            y_arr, x_arr = cupy.meshgrid(liny, linx, indexing='ij')

            # apply the same warp as _terrain_gpu
            if warp_strength > 0:
                warp_x = cupy.zeros((h, w), dtype=cupy.float32)
                warp_y = cupy.zeros((h, w), dtype=cupy.float32)
                tmp = cupy.empty((h, w), dtype=cupy.float32)
                scaled_x = cupy.empty_like(x_arr)
                scaled_y = cupy.empty_like(y_arr)

                for wi in range(warp_octaves):
                    w_amp = persistence ** wi
                    w_freq = lacunarity ** wi
                    p_wx = cupy.asarray(_make_perm_table(seed + 100 + wi))
                    p_wy = cupy.asarray(_make_perm_table(seed + 200 + wi))
                    cupy.multiply(x_arr, w_freq, out=scaled_x)
                    cupy.multiply(y_arr, w_freq, out=scaled_y)
                    _perlin_gpu_xy[griddim, blockdim](p_wx, scaled_x, scaled_y, 1.0, tmp)
                    warp_x += tmp * w_amp
                    _perlin_gpu_xy[griddim, blockdim](p_wy, scaled_x, scaled_y, 1.0, tmp)
                    warp_y += tmp * w_amp

                warp_norm = sum(persistence ** i for i in range(warp_octaves))
                warp_x /= warp_norm
                warp_y /= warp_norm
                x_arr = x_arr + warp_x * warp_strength
                y_arr = y_arr + warp_y * warp_strength

            w_p = cupy.asarray(_make_perm_table(_w_seed))
            w_noise = cupy.empty((h, w), dtype=cupy.float32)
            _worley_gpu_xy[griddim, blockdim](w_p, x_arr, y_arr, w_noise)
            return w_noise

        raw_worley = da.map_blocks(
            _chunk_worley_raw, skeleton, dtype=cupy.float32,
            meta=cupy.array((), dtype=cupy.float32),
        )
        (raw_worley,) = dask.persist(raw_worley)
        w_min, w_max = dask.compute(da.min(raw_worley), da.max(raw_worley))
        w_norm_range = (float(w_min), float(w_max))
        del raw_worley

    def _chunk_terrain(block, block_info=None):
        info = block_info[0]
        y_start, y_end = info['array-location'][0]
        x_start, x_end = info['array-location'][1]
        x0 = x_range_scaled[0] + (x_range_scaled[1] - x_range_scaled[0]) * x_start / width
        x1 = x_range_scaled[0] + (x_range_scaled[1] - x_range_scaled[0]) * x_end / width
        y0 = y_range_scaled[0] + (y_range_scaled[1] - y_range_scaled[0]) * y_start / height
        y1 = y_range_scaled[0] + (y_range_scaled[1] - y_range_scaled[0]) * y_end / height

        out = cupy.zeros(block.shape, dtype=cupy.float32)
        out[:] = _terrain_gpu(
            out, seed, x_range=(x0, x1), y_range=(y0, y1),
            octaves=octaves, lacunarity=lacunarity,
            persistence=persistence, noise_mode=noise_mode,
            warp_strength=warp_strength, warp_octaves=warp_octaves,
            worley_blend=worley_blend, worley_seed=worley_seed,
            worley_norm_range=w_norm_range,
        )
        return out

    data = da.map_blocks(_chunk_terrain, skeleton, dtype=cupy.float32,
                         meta=cupy.array((), dtype=cupy.float32),
                         **_dask_task_name_kwargs('xrspatial.terrain'))

    data = da.clip(data, -1, 1)
    data = (data + 1) / 2
    data = da.where(data < 0.3, 0, data)
    data *= zfactor
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_terrain(agg: xr.DataArray,
                     x_range: tuple = (0, 500),
                     y_range: tuple = (0, 500),
                     seed: int = 10,
                     zfactor: int = 4000,
                     full_extent: Optional[Union[Tuple, List]] = None,
                     name: str = 'terrain',
                     # enhanced terrain parameters
                     octaves: Optional[int] = 16,
                     lacunarity: float = 2.0,
                     persistence: float = 0.5,
                     noise_mode: str = 'fbm',
                     warp_strength: float = 0.0,
                     warp_octaves: int = 4,
                     worley_blend: float = 0.0,
                     worley_seed: Optional[int] = None,
                     erode: bool = False,
                     erosion_iterations: int = 50000,
                     erosion_params: Optional[Dict] = None,
                     ) -> xr.DataArray:
    """
    Generate pseudo-random terrain for testing raster functions.

    Parameters
    ----------
    agg : xr.DataArray
        2D template array (determines height, width, and backend).
        If it carries its own ``y``/``x`` coordinates and attrs (e.g.
        ``crs``, ``res``, ``units``), they are preserved on the output and
        ``x_range``/``y_range`` only drive the noise field. A bare template
        with no coordinates gets a synthetic grid from ``x_range``/``y_range``.
    x_range : tuple, default=(0, 500)
        Range of x values.
    y_range : tuple, default=(0, 500)
        Range of y values.
    seed : int, default=10
        Seed for random number generator.
    zfactor : int, default=4000
        Multiplier for z values.
    full_extent : tuple or list, optional
        bbox (xmin, ymin, xmax, ymax). Full extent of coordinate system.
    name : str, default='terrain'
        Name for the output DataArray.
    octaves : int or None, default=16
        Number of noise octaves. None = adaptive based on raster size.
    lacunarity : float, default=2.0
        Frequency multiplier per octave.
    persistence : float, default=0.5
        Amplitude multiplier per octave.
    noise_mode : str, default='fbm'
        Noise algorithm: 'fbm' (fractal Brownian motion) or 'ridged'
        (ridged multifractal for sharp mountain ridges).
    warp_strength : float, default=0.0
        Domain warping intensity. 0 disables warping;
        ~0.5 produces organic flowing features.
    warp_octaves : int, default=4
        Octaves used for the warp displacement fields.
    worley_blend : float, default=0.0
        Blend factor for Worley (cellular) noise. 0 = none;
        0.1-0.3 adds rocky micro-texture.
    worley_seed : int or None, default=None
        Seed for Worley noise. None = seed + 1000.
    erode : bool, default=False
        Apply hydraulic erosion post-pass.
    erosion_iterations : int, default=50000
        Number of water droplets for erosion.
    erosion_params : dict, optional
        Override default erosion constants.

    Returns
    -------
    terrain : xr.DataArray
        2D array of generated terrain values.

    References
    ----------
        - Michael McHugh: https://www.youtube.com/watch?v=O33YV4ooHSo
        - Red Blob Games: https://www.redblobgames.com/maps/terrain-from-noise/
    """
    _validate_raster(agg, func_name='generate_terrain', name='agg')

    height, width = agg.shape

    # --- validate noise_mode ---
    if noise_mode not in ('fbm', 'ridged'):
        raise ValueError(
            f"noise_mode must be 'fbm' or 'ridged', got {noise_mode!r}"
        )

    # --- adaptive octaves ---
    if octaves is None:
        octaves = max(1, int(np.ceil(np.log2(min(height, width)))))

    if octaves < 1:
        raise ValueError(f"octaves must be >= 1, got {octaves}")

    if not (np.isfinite(lacunarity) and lacunarity > 0):
        raise ValueError(
            f"lacunarity must be a positive finite number, got {lacunarity!r}"
        )

    if not (np.isfinite(persistence) and persistence > 0):
        raise ValueError(
            f"persistence must be a positive finite number, got {persistence!r}"
        )

    if full_extent is None:
        full_extent = (x_range[0], y_range[0],
                       x_range[1], y_range[1])

    elif not isinstance(full_extent, (list, tuple)) and len(full_extent) != 4:
        raise TypeError('full_extent must be tuple(4)')

    full_xrange = (full_extent[0], full_extent[2])
    full_yrange = (full_extent[1], full_extent[3])

    x_range_scaled = (_scale(x_range[0], full_xrange, (0.0, 1.0)),
                      _scale(x_range[1], full_xrange, (0.0, 1.0)))

    y_range_scaled = (_scale(y_range[0], full_yrange, (0.0, 1.0)),
                      _scale(y_range[1], full_yrange, (0.0, 1.0)))

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_terrain_numpy,
        cupy_func=_terrain_cupy,
        dask_func=_terrain_dask_numpy,
        dask_cupy_func=_terrain_dask_cupy
    )
    out = mapper(agg)(agg.data, seed, x_range_scaled, y_range_scaled, zfactor,
                      octaves, lacunarity, persistence, noise_mode,
                      warp_strength, warp_octaves, worley_blend, worley_seed)

    # When the caller's array carries its own georeference (coords / attrs
    # such as crs, res, units), preserve it so generate_terrain fills the
    # caller's grid instead of replacing it with a synthetic (x_range,
    # y_range) one.  A bare template with no coords keeps the old behaviour
    # of synthesizing pixel-center coordinates from x_range / y_range.
    if 'x' in agg.coords and 'y' in agg.coords:
        coords = {'y': agg.coords['y'], 'x': agg.coords['x']}
    else:
        # pixel-center coordinates (standard pixel-center grid convention)
        dx = (x_range[1] - x_range[0]) / width
        dy = (y_range[1] - y_range[0]) / height
        xs = np.linspace(x_range[0] + dx / 2, x_range[1] - dx / 2, width)
        ys = np.linspace(y_range[0] + dy / 2, y_range[1] - dy / 2, height)
        coords = {'y': ys, 'x': xs}

    # Carry the caller's attrs (crs, units, ...) verbatim.  Only synthesize a
    # res when the caller did not provide one, deriving it from the output
    # coord spacing so it stays consistent with the coordinates above.
    attrs = dict(agg.attrs)
    if 'res' not in attrs:
        xs_arr = np.asarray(coords['x'])
        ys_arr = np.asarray(coords['y'])
        res_x = (float(xs_arr[1] - xs_arr[0]) if xs_arr.size > 1
                 else float(x_range[1] - x_range[0]))
        res_y = (float(ys_arr[1] - ys_arr[0]) if ys_arr.size > 1
                 else float(y_range[1] - y_range[0]))
        attrs['res'] = (res_x, res_y)

    result = xr.DataArray(out,
                          name=name,
                          coords=coords,
                          dims=['y', 'x'],
                          attrs=attrs)

    # --- hydraulic erosion ---
    if erode:
        from xrspatial.erosion import erode as _erode
        result = _erode(result, iterations=erosion_iterations,
                        seed=seed, params=erosion_params)
        result.name = name

    return result
