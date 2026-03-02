from __future__ import annotations

# std lib
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import datashader as ds
# 3rd-party
import numpy as np
import pandas as pd
import xarray as xr

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

from .perlin import _make_perm_table, _perlin, _perlin_gpu, _perlin_gpu_xy
from .worley import _worley_cpu, _worley_numpy_xy, _worley_gpu, _worley_gpu_xy


def _scale(value, old_range, new_range):
    d = (value - old_range[0]) / (old_range[1] - old_range[0])
    return d * (new_range[1] - new_range[0]) + new_range[0]


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------

def _gen_terrain(height_map, seed, x_range=(0, 1), y_range=(0, 1),
                 octaves=16, lacunarity=2.0, persistence=0.5,
                 noise_mode='fbm', warp_strength=0.0, warp_octaves=4,
                 worley_blend=0.0, worley_seed=None):
    height, width = height_map.shape

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
    data = data * 0
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
    data = data * 0
    height, width = data.shape

    linx = da.linspace(
        x_range_scaled[0], x_range_scaled[1], width, endpoint=False,
        dtype=np.float32, chunks=data.chunks[1][0]
    )
    liny = da.linspace(
        y_range_scaled[0], y_range_scaled[1], height, endpoint=False,
        dtype=np.float32, chunks=data.chunks[0][0]
    )
    x, y = da.meshgrid(linx, liny)

    # --- domain warping ---
    if warp_strength > 0:
        warp_x = da.zeros_like(x)
        warp_y = da.zeros_like(y)
        for wi in range(warp_octaves):
            w_amp = persistence ** wi
            w_freq = lacunarity ** wi
            p_wx = _make_perm_table(seed + 100 + wi)
            p_wy = _make_perm_table(seed + 200 + wi)
            _fx = partial(_perlin, p_wx)
            _fy = partial(_perlin, p_wy)
            warp_x += da.map_blocks(
                _fx, x * w_freq, y * w_freq,
                meta=np.array((), dtype=np.float32)
            ) * w_amp
            warp_y += da.map_blocks(
                _fy, x * w_freq, y * w_freq,
                meta=np.array((), dtype=np.float32)
            ) * w_amp
        warp_norm = sum(persistence ** i for i in range(warp_octaves))
        warp_x /= warp_norm
        warp_y /= warp_norm
        x = x + warp_x * warp_strength
        y = y + warp_y * warp_strength
        # persist warped coords so the octave loop doesn't rebuild the
        # warp subgraph on every iteration
        (x, y) = dask.persist(x, y)

    # --- octave noise loop ---
    norm = sum(persistence ** i for i in range(octaves))

    if noise_mode == 'ridged':
        weight = da.ones((height, width), dtype=np.float32,
                         chunks=data.chunks)
        for i in range(octaves):
            amp = persistence ** i
            freq = lacunarity ** i
            p = _make_perm_table(seed + i)
            _func = partial(_perlin, p)
            noise = da.map_blocks(
                _func, x * freq, y * freq,
                meta=np.array((), dtype=np.float32)
            )
            noise = 1.0 - da.abs(noise)
            noise = noise * noise
            noise = noise * weight
            weight = da.clip(noise, 0, 1)
            data += noise * amp
    else:  # fbm
        for i in range(octaves):
            amp = persistence ** i
            freq = lacunarity ** i
            p = _make_perm_table(seed + i)
            _func = partial(_perlin, p)
            noise = da.map_blocks(
                _func, x * freq, y * freq,
                meta=np.array((), dtype=np.float32)
            )
            data += noise * amp

    data /= norm

    # --- worley blending ---
    if worley_blend > 0:
        if worley_seed is None:
            worley_seed = seed + 1000
        w_p = _make_perm_table(worley_seed)
        _wfunc = partial(_worley_numpy_xy, w_p)
        w_noise = da.map_blocks(
            _wfunc, x, y, meta=np.array((), dtype=np.float32)
        )
        # persist so min/max don't recompute worley (and warped coords)
        (w_noise,) = dask.persist(w_noise)
        w_min, w_max = dask.compute(da.min(w_noise), da.max(w_noise))
        w_ptp = w_max - w_min
        if w_ptp > 0:
            w_noise = (w_noise - w_min) / w_ptp
        data = data * (1 - worley_blend) + w_noise * worley_blend

    data = data ** 3

    data = da.clip(data, -1, 1)
    data = (data + 1) / 2
    data = da.where(data < 0.3, 0, data)
    data *= zfactor

    return data


# ---------------------------------------------------------------------------
# cupy (GPU) backend
# ---------------------------------------------------------------------------

def _terrain_gpu(height_map, seed, x_range=(0, 1), y_range=(0, 1),
                 octaves=16, lacunarity=2.0, persistence=0.5,
                 noise_mode='fbm', warp_strength=0.0, warp_octaves=4,
                 worley_blend=0.0, worley_seed=None):

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
    data = data * 0
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
    using the GPU kernels.
    """
    data = data * 0
    height, width = data.shape

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
        )
        return out

    data = da.map_blocks(_chunk_terrain, data, dtype=cupy.float32,
                         meta=cupy.array((), dtype=cupy.float32))

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

    canvas = ds.Canvas(
        plot_width=width, plot_height=height, x_range=x_range, y_range=y_range
    )

    # DataArray coords were coming back different from cvs.points...
    hack_agg = canvas.points(pd.DataFrame({'x': [], 'y': []}), 'x', 'y')
    res = get_dataarray_resolution(hack_agg)
    result = xr.DataArray(out,
                          name=name,
                          coords=hack_agg.coords,
                          dims=hack_agg.dims,
                          attrs={'res': res})

    # --- hydraulic erosion ---
    if erode:
        from xrspatial.erosion import erode as _erode
        result = _erode(result, iterations=erosion_iterations,
                        seed=seed, params=erosion_params)
        result.name = name

    return result
