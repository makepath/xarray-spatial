"""GLCM (Gray-Level Co-occurrence Matrix) texture metrics.

Computes Haralick texture features over a sliding window on a raster.
Supports numpy, cupy, dask+numpy, and dask+cupy backends.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import cupy
except ImportError:
    class cupy:
        ndarray = False

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _validate_raster,
    _validate_scalar,
    is_cupy_array,
    ngjit,
    not_implemented_func,
)

VALID_METRICS = ('contrast', 'dissimilarity', 'homogeneity',
                 'energy', 'correlation', 'entropy')

# Offset vectors for the four standard GLCM angles (row_offset, col_offset).
_ANGLE_OFFSETS = {
    0:   (0, 1),
    45:  (-1, 1),
    90:  (-1, 0),
    135: (-1, -1),
}


def glcm_texture(
    agg,
    metric='contrast',
    window_size=7,
    levels=64,
    distance=1,
    angle=None,
):
    """Compute GLCM texture metrics over a sliding window.

    Parameters
    ----------
    agg : xr.DataArray
        2-D input raster with numeric dtype.
    metric : str or list of str
        One or more of: 'contrast', 'dissimilarity', 'homogeneity',
        'energy', 'correlation', 'entropy'.
    window_size : int
        Side length of the sliding window (must be odd, >= 3).
    levels : int
        Number of gray levels for quantization (2-256).
    distance : int
        Pixel pair distance (>= 1).
    angle : int or None
        Co-occurrence angle in degrees: 0, 45, 90, or 135.
        If None, averages over all four angles.

    Returns
    -------
    xr.DataArray
        If a single metric is requested, returns a 2-D DataArray.
        If multiple metrics are requested, returns a 3-D DataArray
        with a leading 'metric' dimension.
    """
    _validate_raster(agg, func_name='glcm_texture', ndim=2)
    _validate_scalar(window_size, func_name='glcm_texture', name='window_size',
                     dtype=int, min_val=3)
    if window_size % 2 == 0:
        raise ValueError("glcm_texture(): `window_size` must be odd, "
                         f"got {window_size}")
    _validate_scalar(levels, func_name='glcm_texture', name='levels',
                     dtype=int, min_val=2, max_val=256)
    _validate_scalar(distance, func_name='glcm_texture', name='distance',
                     dtype=int, min_val=1)

    if angle is not None:
        if angle not in _ANGLE_OFFSETS:
            raise ValueError(
                f"glcm_texture(): `angle` must be one of "
                f"{list(_ANGLE_OFFSETS.keys())} or None, got {angle}"
            )

    single_metric = isinstance(metric, str)
    if single_metric:
        metrics = [metric]
    else:
        metrics = list(metric)

    for m in metrics:
        if m not in VALID_METRICS:
            raise ValueError(
                f"glcm_texture(): unknown metric {m!r}, "
                f"must be one of {VALID_METRICS}"
            )

    # Sort metrics to match the kernel's output order (VALID_METRICS order).
    # Without this, coordinate labels would be wrong when the user requests
    # metrics in a different order (e.g. ['entropy', 'contrast']).
    metrics = _sorted_metrics(metrics)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_glcm_numpy,
        cupy_func=_glcm_cupy,
        dask_func=_glcm_dask_numpy,
        dask_cupy_func=_glcm_dask_cupy,
    )
    func = mapper(agg)
    result = func(agg, metrics, window_size, levels, distance, angle)

    if single_metric:
        result = result.isel(metric=0, drop=True)

    return result


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def _quantize(data, levels, dmin=None, dmax=None):
    """Quantize data to integer levels [0, levels-1]. NaN maps to -1."""
    xp = np
    if is_cupy_array(data):
        xp = cupy

    result = xp.full(data.shape, -1, dtype=np.int32)
    valid = ~xp.isnan(data)
    if not xp.any(valid):
        return result

    if dmin is None:
        dmin = float(xp.nanmin(data))
    if dmax is None:
        dmax = float(xp.nanmax(data))

    if dmin == dmax:
        result[valid] = 0
        return result

    scale = (levels - 1) / (dmax - dmin)
    result[valid] = xp.clip(
        ((data[valid] - dmin) * scale).astype(np.int32),
        0, levels - 1,
    )
    return result


# ---------------------------------------------------------------------------
# Numba kernel (shared by all backends)
# ---------------------------------------------------------------------------

@ngjit
def _glcm_numba_kernel(quantized, out, metric_flags, levels, half, dy, dx):
    """Numba-jitted GLCM kernel for a single angle.

    metric_flags: length-6 bool array in VALID_METRICS order.
    """
    h = quantized.shape[0]
    w = quantized.shape[1]

    for r in range(h):
        for c in range(w):
            glcm = np.zeros((levels, levels), dtype=np.float64)
            count = 0.0

            for wy in range(r - half, r + half + 1):
                for wx in range(c - half, c + half + 1):
                    ny = wy + dy
                    nx = wx + dx
                    if (0 <= wy < h and 0 <= wx < w and
                            0 <= ny < h and 0 <= nx < w):
                        i_val = quantized[wy, wx]
                        j_val = quantized[ny, nx]
                        if i_val >= 0 and j_val >= 0:
                            glcm[i_val, j_val] += 1.0
                            count += 1.0

            if count == 0.0:
                continue

            for i in range(levels):
                for j in range(levels):
                    glcm[i, j] /= count

            need_corr = metric_flags[4]
            mu_i = 0.0
            mu_j = 0.0
            std_i = 0.0
            std_j = 0.0
            if need_corr:
                for i in range(levels):
                    row_sum = 0.0
                    col_sum = 0.0
                    for j in range(levels):
                        row_sum += glcm[i, j]
                        col_sum += glcm[j, i]
                    mu_i += i * row_sum
                    mu_j += i * col_sum
                var_i = 0.0
                var_j = 0.0
                for i in range(levels):
                    row_sum = 0.0
                    col_sum = 0.0
                    for j in range(levels):
                        row_sum += glcm[i, j]
                        col_sum += glcm[j, i]
                    var_i += (i - mu_i) ** 2 * row_sum
                    var_j += (i - mu_j) ** 2 * col_sum
                std_i = var_i ** 0.5
                std_j = var_j ** 0.5

            v_contrast = 0.0
            v_dissimilarity = 0.0
            v_homogeneity = 0.0
            v_energy = 0.0
            v_correlation = 0.0
            v_entropy = 0.0

            for i in range(levels):
                for j in range(levels):
                    p = glcm[i, j]
                    if p == 0.0:
                        continue
                    diff = i - j
                    if diff < 0:
                        diff = -diff
                    if metric_flags[0]:
                        v_contrast += p * diff * diff
                    if metric_flags[1]:
                        v_dissimilarity += p * diff
                    if metric_flags[2]:
                        v_homogeneity += p / (1.0 + diff * diff)
                    if metric_flags[3]:
                        v_energy += p * p
                    if need_corr and std_i > 0 and std_j > 0:
                        v_correlation += (
                            p * (i - mu_i) * (j - mu_j) / (std_i * std_j)
                        )
                    if metric_flags[5]:
                        v_entropy -= p * np.log(p)

            idx = 0
            if metric_flags[0]:
                out[idx, r, c] = v_contrast
                idx += 1
            if metric_flags[1]:
                out[idx, r, c] = v_dissimilarity
                idx += 1
            if metric_flags[2]:
                out[idx, r, c] = v_homogeneity
                idx += 1
            if metric_flags[3]:
                out[idx, r, c] = v_energy
                idx += 1
            if metric_flags[4]:
                if std_i > 0 and std_j > 0:
                    out[idx, r, c] = v_correlation
                else:
                    out[idx, r, c] = np.nan
                idx += 1
            if metric_flags[5]:
                out[idx, r, c] = v_entropy
                idx += 1


def _metric_flags(metrics):
    """Convert metric names to a boolean flag array in VALID_METRICS order."""
    flags = np.zeros(len(VALID_METRICS), dtype=np.bool_)
    for m in metrics:
        flags[VALID_METRICS.index(m)] = True
    return flags


def _sorted_metrics(metrics):
    """Return *metrics* sorted in VALID_METRICS order.

    The numba kernel always writes output slots in VALID_METRICS order,
    so coordinate labels must follow the same ordering.
    """
    order = {m: i for i, m in enumerate(VALID_METRICS)}
    return sorted(metrics, key=lambda m: order[m])


def _run_glcm_on_quantized(quantized, metrics, window_size, levels,
                            distance, angle):
    """Run GLCM computation on pre-quantized int32 data.

    Returns (n_metrics, H, W) float64 array.
    """
    h, w = quantized.shape
    n_metrics = len(metrics)
    half = window_size // 2
    flags = _metric_flags(metrics)

    if angle is not None:
        out = np.full((n_metrics, h, w), np.nan, dtype=np.float64)
        dy, dx = _ANGLE_OFFSETS[angle]
        dy *= distance
        dx *= distance
        _glcm_numba_kernel(quantized, out, flags, levels, half, dy, dx)
    else:
        out = np.zeros((n_metrics, h, w), dtype=np.float64)
        for a in _ANGLE_OFFSETS:
            tmp = np.full((n_metrics, h, w), np.nan, dtype=np.float64)
            dy, dx = _ANGLE_OFFSETS[a]
            dy *= distance
            dx *= distance
            _glcm_numba_kernel(quantized, tmp, flags, levels, half, dy, dx)
            nan_mask = np.isnan(tmp)
            tmp[nan_mask] = 0.0
            out += tmp
        out /= 4.0

    return out


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

def _glcm_numpy(agg, metrics, window_size, levels, distance, angle):
    data = agg.values.astype(np.float64)
    quantized = _quantize(data, levels)
    result = _run_glcm_on_quantized(quantized, metrics, window_size,
                                    levels, distance, angle)
    coords = dict(agg.coords)
    dims = ('metric',) + agg.dims
    return xr.DataArray(
        result, dims=dims,
        coords={'metric': list(metrics), **coords},
        attrs=agg.attrs,
    )


# ---------------------------------------------------------------------------
# Dask + NumPy backend
# ---------------------------------------------------------------------------

def _glcm_dask_numpy(agg, metrics, window_size, levels, distance, angle):
    if da is None:
        raise ImportError("dask is required for the dask+numpy backend")

    data = agg.data.astype(np.float64)
    depth = window_size // 2 + distance

    # Global min/max for consistent quantization across chunks
    dmin = float(da.nanmin(data))
    dmax = float(da.nanmax(data))

    quantized = _dask_quantize(data, levels, dmin, dmax)

    # Compute each metric individually via map_overlap then stack
    layers = []
    for m in metrics:
        single = [m]

        def _chunk_func(block, _single=single):
            return _run_glcm_on_quantized(block, _single, window_size,
                                          levels, distance, angle)[0]

        layer = da.map_overlap(
            _chunk_func, quantized,
            depth=depth, boundary=-1, dtype=np.float64,
        )
        layers.append(layer)

    result = da.stack(layers, axis=0)

    coords = dict(agg.coords)
    dims = ('metric',) + agg.dims
    return xr.DataArray(
        result, dims=dims,
        coords={'metric': list(metrics), **coords},
        attrs=agg.attrs,
    )


def _dask_quantize(data, levels, dmin, dmax):
    """Quantize a dask array to int32 levels. NaN maps to -1."""
    if dmin == dmax:
        return da.where(da.isnan(data), -1, 0).astype(np.int32)
    scale = (levels - 1) / (dmax - dmin)
    result = da.clip(((data - dmin) * scale).astype(np.int32),
                     0, levels - 1)
    return da.where(da.isnan(data), -1, result).astype(np.int32)


# ---------------------------------------------------------------------------
# CuPy backend (CPU fallback via cupy.asnumpy)
# ---------------------------------------------------------------------------

def _glcm_cupy(agg, metrics, window_size, levels, distance, angle):
    data = cupy.asnumpy(agg.data).astype(np.float64)
    quantized = _quantize(data, levels)
    result_np = _run_glcm_on_quantized(quantized, metrics, window_size,
                                       levels, distance, angle)
    result_cp = cupy.asarray(result_np)
    coords = dict(agg.coords)
    dims = ('metric',) + agg.dims
    return xr.DataArray(
        result_cp, dims=dims,
        coords={'metric': list(metrics), **coords},
        attrs=agg.attrs,
    )


# ---------------------------------------------------------------------------
# Dask + CuPy backend (CPU fallback per chunk)
# ---------------------------------------------------------------------------

def _glcm_dask_cupy(agg, metrics, window_size, levels, distance, angle):
    if da is None:
        raise ImportError("dask is required for the dask+cupy backend")

    data = agg.data
    depth = window_size // 2 + distance

    dmin = float(da.nanmin(data))
    dmax = float(da.nanmax(data))

    quantized = _dask_quantize(data, levels, dmin, dmax)

    layers = []
    for m in metrics:
        single = [m]

        def _chunk_func(block, _single=single):
            block_np = cupy.asnumpy(block)
            result = _run_glcm_on_quantized(block_np, _single, window_size,
                                            levels, distance, angle)[0]
            return cupy.asarray(result)

        layer = da.map_overlap(
            _chunk_func, quantized,
            depth=depth, boundary=-1, dtype=np.float64,
        )
        layers.append(layer)

    result = da.stack(layers, axis=0)

    coords = dict(agg.coords)
    dims = ('metric',) + agg.dims
    return xr.DataArray(
        result, dims=dims,
        coords={'metric': list(metrics), **coords},
        attrs=agg.attrs,
    )
