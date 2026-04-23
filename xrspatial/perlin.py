from __future__ import annotations

# std lib
import math
from functools import partial

# 3rd-party
import numpy as np
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

import numba as nb
from numba import cuda, jit

# local modules
from xrspatial.utils import (ArrayTypeFunctionMapping, _validate_raster, cuda_args,
                             not_implemented_func)


def _make_perm_table(seed):
    """Build a 512-element permutation table (256 unique values, doubled for wrap-around).

    Uses RandomState to avoid mutating the global numpy RNG.
    """
    rs = np.random.RandomState(seed)
    p = rs.permutation(256).astype(np.int32)
    return np.append(p, p)


@jit(nopython=True, nogil=True)
def _lerp(a, b, x):
    return a + x * (b - a)


@jit(nopython=True, nogil=True)
def _fade(t):
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


@jit(nopython=True, nogil=True)
def _gradient(h, x, y):
    out = np.zeros(h.shape, dtype=np.float32)
    for j in nb.prange(h.shape[1]):
        for i in nb.prange(h.shape[0]):
            hv = h[i, j] & 3
            sel = (hv >> 1) & 1  # 0 -> y axis, 1 -> x axis
            u = x[i, j] * sel + y[i, j] * (1 - sel)
            out[i, j] = u * (1 - 2 * (hv & 1))
    return out


def _perlin(p, x, y):
    # coordinates of the top-left (floor, not truncate, so negatives work)
    x_floor = np.floor(x)
    y_floor = np.floor(y)
    xi = x_floor.astype(np.int32)
    yi = y_floor.astype(np.int32)

    # mask to 0-255 range for 512-element table
    xi = xi & 255
    yi = yi & 255

    # internal coordinates
    xf = x - x_floor
    yf = y - y_floor

    # fade factors
    u = _fade(xf)
    v = _fade(yf)

    # noise components
    n00 = _gradient(p[p[xi] + yi], xf, yf)
    n01 = _gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = _gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = _gradient(p[p[xi + 1] + yi], xf - 1, yf)

    # combine noises
    x1 = _lerp(n00, n10, u)
    x2 = _lerp(n01, n11, u)
    a = _lerp(x1, x2, v)
    return a


def _perlin_numpy(data: np.ndarray,
                  freq: tuple,
                  seed: int) -> np.ndarray:
    p = _make_perm_table(seed)

    height, width = data.shape
    linx = np.linspace(0, freq[0], width, endpoint=False, dtype=np.float32)
    liny = np.linspace(0, freq[1], height, endpoint=False, dtype=np.float32)
    x, y = np.meshgrid(linx, liny)

    data[:] = _perlin(p, x, y)
    data[:] = (data - np.min(data)) / np.ptp(data)
    return data


def _perlin_dask_numpy(data: da.Array,
                       freq: tuple,
                       seed: int) -> da.Array:
    p = _make_perm_table(seed)

    height, width = data.shape
    linx = da.linspace(0, freq[0], width, endpoint=False, dtype=np.float32,
                       chunks=data.chunks[1][0])
    liny = da.linspace(0, freq[1], height, endpoint=False, dtype=np.float32,
                       chunks=data.chunks[0][0])
    x, y = da.meshgrid(linx, liny)

    _func = partial(_perlin, p)
    data = da.map_blocks(_func, x, y, meta=np.array((), dtype=np.float32))

    # persist so min/ptp don't recompute the noise from scratch
    (data,) = dask.persist(data)
    min_val, ptp_val = dask.compute(da.min(data), da.ptp(data))
    data = (data - min_val) / ptp_val
    return data


@cuda.jit(device=True)
def _lerp_gpu(a, b, x):
    return a + x * (b - a)


@cuda.jit(device=True)
def _fade_gpu(t):
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


@cuda.jit(device=True)
def _gradient_gpu(h, x, y):
    hv = h & 3
    sel = (hv >> 1) & 1
    u = x * sel + y * (1 - sel)
    return u * (1 - 2 * (hv & 1))


@cuda.jit(fastmath=True, opt=True)
def _perlin_gpu(p, x0, x1, y0, y1, m, out):
    # cooperative load of permutation table into shared memory
    sp = cuda.shared.array(512, nb.int32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    tid = ty * bw + tx
    block_size = bw * bh
    for k in range(tid, 512, block_size):
        sp[k] = p[k]
    cuda.syncthreads()

    i, j = nb.cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        # coordinates of the top-left
        y = y0 + i * (y1 - y0) / out.shape[0]
        x = x0 + j * (x1 - x0) / out.shape[1]

        # integer coordinates masked to table range
        # floor, not truncate, so negative coords work correctly
        x_int = int(math.floor(x)) & 255
        y_int = int(math.floor(y)) & 255

        # internal coordinates
        xf = x - math.floor(x)
        yf = y - math.floor(y)

        # fade factors
        u = _fade_gpu(xf)
        v = _fade_gpu(yf)

        # noise components (all reads from shared memory)
        n00 = _gradient_gpu(sp[sp[x_int] + y_int], xf, yf)
        n01 = _gradient_gpu(sp[sp[x_int] + y_int + 1], xf, yf - 1)
        n11 = _gradient_gpu(sp[sp[x_int + 1] + y_int + 1], xf - 1, yf - 1)
        n10 = _gradient_gpu(sp[sp[x_int + 1] + y_int], xf - 1, yf)

        # combine noises
        x1 = _lerp_gpu(n00, n10, u)
        x2 = _lerp_gpu(n01, n11, u)
        out[i, j] = m * _lerp_gpu(x1, x2, v)


@cuda.jit(fastmath=True, opt=True)
def _perlin_gpu_xy(p, x_arr, y_arr, m, out):
    """Like _perlin_gpu but takes 2D coordinate arrays instead of linear ranges.

    Needed for domain warping where per-pixel coordinates are non-linear.
    """
    sp = cuda.shared.array(512, nb.int32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    tid = ty * bw + tx
    block_size = bw * bh
    for k in range(tid, 512, block_size):
        sp[k] = p[k]
    cuda.syncthreads()

    i, j = nb.cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        x = x_arr[i, j]
        y = y_arr[i, j]

        x_int = int(math.floor(x)) & 255
        y_int = int(math.floor(y)) & 255

        xf = x - math.floor(x)
        yf = y - math.floor(y)

        u = _fade_gpu(xf)
        v = _fade_gpu(yf)

        n00 = _gradient_gpu(sp[sp[x_int] + y_int], xf, yf)
        n01 = _gradient_gpu(sp[sp[x_int] + y_int + 1], xf, yf - 1)
        n11 = _gradient_gpu(sp[sp[x_int + 1] + y_int + 1], xf - 1, yf - 1)
        n10 = _gradient_gpu(sp[sp[x_int + 1] + y_int], xf - 1, yf)

        x1 = _lerp_gpu(n00, n10, u)
        x2 = _lerp_gpu(n01, n11, u)
        out[i, j] = m * _lerp_gpu(x1, x2, v)


def _perlin_cupy(data: cupy.ndarray,
                 freq: tuple,
                 seed: int) -> cupy.ndarray:

    p = cupy.asarray(_make_perm_table(seed))

    griddim, blockdim = cuda_args(data.shape)
    _perlin_gpu[griddim, blockdim](p, 0, freq[0], 0, freq[1], 1, data)

    minimum = cupy.amin(data)
    maximum = cupy.amax(data)
    data[:] = (data - minimum) / (maximum - minimum)
    return data


def _perlin_dask_cupy(data: da.Array,
                      freq: tuple,
                      seed: int) -> da.Array:
    p = cupy.asarray(_make_perm_table(seed))

    height, width = data.shape

    def _chunk_perlin(block, block_info=None):
        info = block_info[0]
        y_start, y_end = info['array-location'][0]
        x_start, x_end = info['array-location'][1]
        x0 = freq[0] * x_start / width
        x1 = freq[0] * x_end / width
        y0 = freq[1] * y_start / height
        y1 = freq[1] * y_end / height

        out = cupy.empty(block.shape, dtype=cupy.float32)
        griddim, blockdim = cuda_args(block.shape)
        _perlin_gpu[griddim, blockdim](p, x0, x1, y0, y1, 1.0, out)
        return out

    data = da.map_blocks(_chunk_perlin, data, dtype=cupy.float32,
                         meta=cupy.array((), dtype=cupy.float32))

    # persist so min/max don't recompute the noise from scratch
    (data,) = dask.persist(data)
    min_val, max_val = dask.compute(da.min(data), da.max(data))
    data = (data - min_val) / (max_val - min_val)
    return data


def perlin(agg: xr.DataArray,
           freq: tuple = (1, 1),
           seed: int = 5,
           name: str = 'perlin') -> xr.DataArray:
    """
    Generate perlin noise aggregate.

    Parameters
    ----------
    agg : xr.DataArray
        2D array of size width x height, will be used to determine
        height/ width and which platform to use for calculation.
    freq : tuple, default=(1,1)
        (x, y) frequency multipliers.
    seed : int, default=5
        Seed for random number generator.

    Returns
    -------
    perlin_agg : xarray.DataArray
        2D array of perlin noise values.

    References
    ----------
        - Paul Panzer: https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy # noqa
        - ICA: http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import perlin

        >>> W = 4
        >>> H = 3
        >>> data = np.zeros((H, W), dtype=np.float32)
        >>> raster = xr.DataArray(data, dims=['y', 'x'])

        >>> perlin_noise = perlin(raster)
        >>> print(perlin_noise)
        <xarray.DataArray 'perlin' (y: 3, x: 4)>
        array([[0.        , 0.58311844, 0.96620274, 0.58311844],
               [0.        , 0.5796199 , 1.        , 0.7346118 ],
               [0.        , 0.5172636 , 0.83896613, 0.697184  ]], dtype=float32)
        Dimensions without coordinates: y, x
    """
    _validate_raster(agg, func_name='perlin', name='agg')

    # perlin writes float noise into the raster in place, then normalizes
    # by ptp.  With an integer buffer the float values cast to 0, ptp is 0,
    # and the normalization divides by zero, corrupting every pixel to
    # INT_MIN.  Reject non-float dtypes up front with a clear error.
    if not np.issubdtype(agg.dtype, np.floating):
        raise ValueError(
            f"perlin(): `agg` must have a floating-point dtype "
            f"(float32 or float64), got {agg.dtype}"
        )

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_perlin_numpy,
        cupy_func=_perlin_cupy,
        dask_func=_perlin_dask_numpy,
        dask_cupy_func=_perlin_dask_cupy
    )
    out = mapper(agg)(agg.data, freq, seed)
    result = xr.DataArray(out,
                          dims=agg.dims,
                          attrs=agg.attrs,
                          name=name)
    return result
