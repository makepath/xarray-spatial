"""Emerging hot spot analysis: time-series hot/cold spot trend detection.

Combines the Getis-Ord Gi* statistic (computed per time step) with the
Mann-Kendall non-parametric trend test to classify each pixel into one
of 17 trend categories describing how spatial clusters evolve over time.

References
----------
Getis, A. and Ord, J. K. (1992). The analysis of spatial association by
    use of distance statistics. *Geographical Analysis*, 24(3), 189-206.
Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica*,
    13(3), 245-259.
Kendall, M. G. (1975). *Rank Correlation Methods*. 4th ed. London:
    Charles Griffin.
"""
from __future__ import annotations

from functools import partial
from math import sqrt

import numpy as np
import xarray as xr
from numba import prange

from xrspatial.convolution import convolve_2d, _convolve_2d_numpy, _convolve_2d_cupy
from xrspatial.focal import _calc_hotspots_numpy
from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _boundary_to_dask,
    _validate_boundary,
    ngjit,
    not_implemented_func,
)

try:
    import cupy
except ImportError:
    cupy = None

try:
    import dask.array as da
except ImportError:
    da = None


# -- Trend category codes ---------------------------------------------------
#  positive = hot spot variant, negative = cold spot mirror, 0 = no pattern
CATEGORY_NEW = 1
CATEGORY_CONSECUTIVE = 2
CATEGORY_INTENSIFYING = 3
CATEGORY_PERSISTENT = 4
CATEGORY_DIMINISHING = 5
CATEGORY_SPORADIC = 6
CATEGORY_OSCILLATING = 7
CATEGORY_HISTORICAL = 8

_MK_ALPHA = 0.05  # significance level for Mann-Kendall trend test


# ---------------------------------------------------------------------------
# Mann-Kendall helpers (Numba-JIT for use inside pixel loops)
# ---------------------------------------------------------------------------

@ngjit
def _norm_cdf_approx(x):
    """Standard normal CDF using Abramowitz & Stegun 26.2.17.

    Accurate to ~1.5e-7 for all x.
    """
    if x < 0.0:
        return 1.0 - _norm_cdf_approx(-x)
    # constants for the rational approximation
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    t = 1.0 / (1.0 + p * x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    pdf = (1.0 / sqrt(2.0 * 3.141592653589793)) * np.exp(-0.5 * x * x)
    return 1.0 - pdf * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5)


@ngjit
def _mk_pvalue(z_mk):
    """Two-sided p-value for Mann-Kendall Z statistic."""
    if z_mk == 0.0:
        return 1.0
    return 2.0 * (1.0 - _norm_cdf_approx(abs(z_mk)))


@ngjit
def _mann_kendall_statistic_numpy(series, n):
    """NaN-aware Mann-Kendall statistic for a 1-D series of length *n*.

    Parameters
    ----------
    series : 1-D float array (may contain NaNs)
    n : int, length of *series*

    Returns
    -------
    z_mk : float  – standardised Mann-Kendall Z
    s     : float  – raw S statistic
    var_s : float  – variance of S (with tie correction)
    """
    # Filter out NaN values
    valid = np.empty(n, dtype=series.dtype)
    m = 0
    for i in range(n):
        if not np.isnan(series[i]):
            valid[m] = series[i]
            m += 1

    if m < 3:
        return 0.0, 0.0, 0.0

    # Compute S
    s = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            diff = valid[j] - valid[i]
            if diff > 0.0:
                s += 1.0
            elif diff < 0.0:
                s -= 1.0

    # Count ties for variance correction
    # Sort valid values to find runs of ties
    # Simple insertion sort (m is small, typically 10-50)
    sorted_v = valid[:m].copy()
    for i in range(1, m):
        key = sorted_v[i]
        j = i - 1
        while j >= 0 and sorted_v[j] > key:
            sorted_v[j + 1] = sorted_v[j]
            j -= 1
        sorted_v[j + 1] = key

    # Compute tie correction sum: sum of t_k*(t_k-1)*(2*t_k+5) for each
    # group of t_k tied values
    tie_correction = 0.0
    i = 0
    while i < m:
        t_k = 1
        while i + t_k < m and sorted_v[i + t_k] == sorted_v[i]:
            t_k += 1
        if t_k > 1:
            tie_correction += t_k * (t_k - 1.0) * (2.0 * t_k + 5.0)
        i += t_k

    var_s = (m * (m - 1.0) * (2.0 * m + 5.0) - tie_correction) / 18.0

    if var_s <= 0.0:
        return 0.0, s, 0.0

    # Continuity-corrected Z
    if s > 0.0:
        z_mk = (s - 1.0) / sqrt(var_s)
    elif s < 0.0:
        z_mk = (s + 1.0) / sqrt(var_s)
    else:
        z_mk = 0.0

    return z_mk, s, var_s


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

@ngjit
def _classify_one_side(bins, n_times, n_sig, n_opposite, is_sig_final,
                       mk_z, mk_p, mk_alpha, threshold_90, sign):
    """Classify trend for one side (hot or cold).

    Parameters
    ----------
    bins : 1-D int8 array of confidence bins (90/95/99 * sign)
    n_times : int
    n_sig : int – number of significant time steps on *this* side
    n_opposite : int – number of significant time steps on opposite side
    is_sig_final : bool – whether final step is significant on this side
    mk_z : float – Mann-Kendall Z
    mk_p : float – Mann-Kendall p-value
    mk_alpha : float – significance threshold
    threshold_90 : int – 90% of n_times (precomputed)
    sign : int – +1 for hot, -1 for cold

    Returns
    -------
    category : int8 (0 if no match on this side)
    """
    if not is_sig_final and n_sig < threshold_90:
        # Not significant at final step and not >=90% of steps -> check
        # only Historical
        if n_sig >= threshold_90:
            return sign * CATEGORY_HISTORICAL
        return 0

    if is_sig_final:
        # Check from most specific to least specific
        # New: hot ONLY at the final time step
        if n_sig == 1:
            return sign * CATEGORY_NEW

        # Categories requiring >=90% significance (check before Consecutive)
        if n_sig >= threshold_90:
            # Intensifying: significant increasing trend (same sign as side)
            if mk_p < mk_alpha and mk_z * sign > 0:
                return sign * CATEGORY_INTENSIFYING
            # Diminishing: significant decreasing trend (opposite sign)
            if mk_p < mk_alpha and mk_z * sign < 0:
                return sign * CATEGORY_DIMINISHING
            # Persistent: no significant MK trend
            return sign * CATEGORY_PERSISTENT

        # Consecutive: hot for final N consecutive steps (N>=2), never before
        consecutive = 0
        for t in range(n_times - 1, -1, -1):
            if bins[t] * sign > 0:
                consecutive += 1
            else:
                break
        if consecutive >= 2 and consecutive == n_sig:
            return sign * CATEGORY_CONSECUTIVE

        # Oscillating: hot at final, was cold at least once
        if n_opposite > 0:
            return sign * CATEGORY_OSCILLATING

        # Sporadic: hot at final, <90% of steps, never cold
        return sign * CATEGORY_SPORADIC

    # Not significant at final step but >=90% of steps -> Historical
    if n_sig >= threshold_90:
        return sign * CATEGORY_HISTORICAL

    return 0


@ngjit
def _classify_single_pixel(bins, n_times, mk_z, mk_p, mk_alpha):
    """Assign one of 17 trend categories to a single pixel.

    Parameters
    ----------
    bins : 1-D int8 array of length n_times (confidence * sign)
    n_times : int
    mk_z, mk_p : float – Mann-Kendall results
    mk_alpha : float – significance threshold

    Returns
    -------
    category : int8  (-8..8)
    """
    threshold_90 = max(int(n_times * 0.9), 1)

    # Count significant time steps on each side
    n_hot = 0
    n_cold = 0
    for t in range(n_times):
        if bins[t] > 0:
            n_hot += 1
        elif bins[t] < 0:
            n_cold += 1

    is_hot_final = bins[n_times - 1] > 0
    is_cold_final = bins[n_times - 1] < 0

    # Try hot side first, then cold
    cat = _classify_one_side(bins, n_times, n_hot, n_cold, is_hot_final,
                             mk_z, mk_p, mk_alpha, threshold_90, 1)
    if cat != 0:
        return cat

    cat = _classify_one_side(bins, n_times, n_cold, n_hot, is_cold_final,
                             mk_z, mk_p, mk_alpha, threshold_90, -1)
    return cat


# ---------------------------------------------------------------------------
# Per-pixel Mann-Kendall + classification loop
# ---------------------------------------------------------------------------

@ngjit
def _mk_classify_numpy(gi_zscore, gi_bin, n_times, ny, nx):
    """Apply Mann-Kendall trend test and classification to every pixel.

    Parameters
    ----------
    gi_zscore : 3-D float32 array (n_times, ny, nx)
    gi_bin : 3-D int8 array (n_times, ny, nx)
    n_times, ny, nx : int

    Returns
    -------
    category : 2-D int8 (ny, nx)
    trend_z  : 2-D float32 (ny, nx)
    trend_p  : 2-D float32 (ny, nx)
    """
    category = np.zeros((ny, nx), dtype=np.int8)
    trend_z = np.zeros((ny, nx), dtype=np.float32)
    trend_p = np.ones((ny, nx), dtype=np.float32)

    for y in prange(ny):
        for x in range(nx):
            series = gi_zscore[:, y, x]
            bins = gi_bin[:, y, x]

            # Check if all NaN
            all_nan = True
            for t in range(n_times):
                if not np.isnan(series[t]):
                    all_nan = False
                    break
            if all_nan:
                trend_z[y, x] = np.nan
                trend_p[y, x] = np.nan
                continue

            z_mk, s, var_s = _mann_kendall_statistic_numpy(series, n_times)
            p_mk = _mk_pvalue(z_mk)

            trend_z[y, x] = np.float32(z_mk)
            trend_p[y, x] = np.float32(p_mk)

            category[y, x] = _classify_single_pixel(
                bins, n_times, z_mk, p_mk, _MK_ALPHA
            )

    return category, trend_z, trend_p


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

def _emerging_hotspots_numpy(raster, kernel, boundary='nan'):
    data = raster.data.astype(np.float32)
    n_times, ny, nx = data.shape

    # Global statistics across entire space-time cube
    global_mean = np.nanmean(data)
    global_std = np.nanstd(data)
    if global_std == 0:
        raise ZeroDivisionError(
            "Standard deviation of the input raster values is 0."
        )

    norm_kernel = (kernel / kernel.sum()).astype(np.float32)

    # Compute Gi* z-scores and confidence bins per time step
    gi_zscore = np.empty((n_times, ny, nx), dtype=np.float32)
    gi_bin = np.empty((n_times, ny, nx), dtype=np.int8)

    for t in range(n_times):
        step_data = data[t]
        convolved = convolve_2d(step_data, norm_kernel, boundary)
        z = (convolved - global_mean) / global_std
        gi_zscore[t] = z
        gi_bin[t] = _calc_hotspots_numpy(z)

    # Mann-Kendall + classification
    category, trend_z, trend_p = _mk_classify_numpy(
        gi_zscore, gi_bin, n_times, ny, nx
    )

    return gi_zscore, gi_bin, category, trend_z, trend_p


# ---------------------------------------------------------------------------
# CuPy backend (GPU convolution, CPU Mann-Kendall)
# ---------------------------------------------------------------------------

def _emerging_hotspots_cupy(raster, kernel, boundary='nan'):
    data = raster.data.astype(cupy.float32)
    n_times, ny, nx = data.shape

    global_mean = cupy.nanmean(data)
    global_std = cupy.nanstd(data)
    if float(global_std) == 0:
        raise ZeroDivisionError(
            "Standard deviation of the input raster values is 0."
        )

    norm_kernel = (kernel / kernel.sum()).astype(np.float32)

    # GPU: convolution + z-score per time step
    gi_zscore_gpu = cupy.empty((n_times, ny, nx), dtype=cupy.float32)
    for t in range(n_times):
        convolved = convolve_2d(data[t], norm_kernel, boundary)
        gi_zscore_gpu[t] = (convolved - global_mean) / global_std

    # Transfer to CPU for Mann-Kendall (branching-heavy, better on CPU)
    gi_zscore_cpu = cupy.asnumpy(gi_zscore_gpu)
    gi_bin = np.empty((n_times, ny, nx), dtype=np.int8)
    for t in range(n_times):
        gi_bin[t] = _calc_hotspots_numpy(gi_zscore_cpu[t])

    category, trend_z, trend_p = _mk_classify_numpy(
        gi_zscore_cpu, gi_bin, n_times, ny, nx
    )

    # Return with CuPy-backed arrays
    return (
        gi_zscore_gpu,                    # already on GPU
        cupy.asarray(gi_bin),
        cupy.asarray(category),
        cupy.asarray(trend_z),
        cupy.asarray(trend_p),
    )


# ---------------------------------------------------------------------------
# Dask + NumPy backend
# ---------------------------------------------------------------------------

def _convolve_and_zscore_chunk(chunk, kernel, global_mean, global_std):
    """Dask chunk function: convolve -> z-score."""
    convolved = _convolve_2d_numpy(chunk, kernel)
    return ((convolved - global_mean) / global_std).astype(np.float32)


def _mk_classify_dask_chunk(block):
    """Dask chunk function: Mann-Kendall + classification on a (T, Y, X) block.

    ``block`` is a single 3-D numpy array whose first axis spans the
    full time dimension (after rechunking).
    """
    n_times, ny, nx = block.shape
    gi_bin = np.empty_like(block, dtype=np.int8)
    for t in range(n_times):
        gi_bin[t] = _calc_hotspots_numpy(block[t])
    category, trend_z, trend_p = _mk_classify_numpy(
        block, gi_bin, n_times, ny, nx
    )
    # Stack into (4, ny, nx): gi_bin is 3-D, rest are 2-D -> we need to
    # return results separately. Use a 4-layer 2-D block that map_blocks
    # can handle: category, trend_z, trend_p, plus a dummy.
    # Actually, just pack into a single (3, ny, nx) float32 block.
    out = np.empty((3, ny, nx), dtype=np.float32)
    out[0] = category.astype(np.float32)
    out[1] = trend_z
    out[2] = trend_p
    return out


def _gi_bin_chunk(block):
    """Dask chunk function: compute confidence bins from z-scores.

    Input and output are both (T, Y, X) blocks.
    """
    n_times = block.shape[0]
    out = np.empty_like(block, dtype=np.int8)
    for t in range(n_times):
        out[t] = _calc_hotspots_numpy(block[t])
    return out


def _emerging_hotspots_dask_numpy(raster, kernel, boundary='nan'):
    data = raster.data
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    n_times = data.shape[0]

    # Pass 1: eagerly compute global statistics (two scalars)
    global_mean, global_std = da.compute(da.nanmean(data), da.nanstd(data))
    global_mean = np.float32(global_mean)
    global_std = np.float32(global_std)

    if global_std == 0:
        raise ZeroDivisionError(
            "Standard deviation of the input raster values is 0."
        )

    norm_kernel = (kernel / kernel.sum()).astype(np.float32)
    pad_h = norm_kernel.shape[0] // 2
    pad_w = norm_kernel.shape[1] // 2

    # Pass 2: per time step, convolve + z-score via map_overlap, then stack
    zscore_slices = []
    for t in range(n_times):
        _func = partial(
            _convolve_and_zscore_chunk,
            kernel=norm_kernel,
            global_mean=global_mean,
            global_std=global_std,
        )
        z_t = data[t].map_overlap(
            _func,
            depth=(pad_h, pad_w),
            boundary=_boundary_to_dask(boundary),
            meta=np.array((), dtype=np.float32),
        )
        zscore_slices.append(z_t)

    gi_zscore = da.stack(zscore_slices, axis=0)
    # Rechunk so time axis is in a single chunk (time dim is small)
    gi_zscore = gi_zscore.rechunk({0: n_times})

    # gi_bin via map_blocks (element-wise on the z-scores, no overlap needed)
    gi_bin = da.map_blocks(
        _gi_bin_chunk,
        gi_zscore,
        dtype=np.int8,
        meta=np.array((), dtype=np.int8),
    )

    # Pass 3: Mann-Kendall + classification via map_blocks
    # Input: (T, Y, X) -> Output: (3, Y, X) = [category, trend_z, trend_p]
    mk_result = da.map_blocks(
        _mk_classify_dask_chunk,
        gi_zscore,
        dtype=np.float32,
        chunks=((3,), *gi_zscore.chunks[1:]),
        drop_axis=0,
        new_axis=0,
        meta=np.array((), dtype=np.float32),
    )

    return gi_zscore, gi_bin, mk_result


# ---------------------------------------------------------------------------
# Dask + CuPy backend (GPU convolution, CPU Mann-Kendall)
# ---------------------------------------------------------------------------

def _convolve_and_zscore_cupy_chunk(chunk, kernel, global_mean, global_std):
    """Dask chunk function: GPU convolve -> z-score."""
    convolved = _convolve_2d_cupy(chunk, kernel)
    return ((convolved - global_mean) / global_std).astype(cupy.float32)


def _gi_bin_cupy_chunk(block):
    """Dask chunk function: z-scores -> confidence bins (CPU classify, CuPy I/O)."""
    block_cpu = cupy.asnumpy(block)
    n_times = block_cpu.shape[0]
    out = np.empty_like(block_cpu, dtype=np.int8)
    for t in range(n_times):
        out[t] = _calc_hotspots_numpy(block_cpu[t])
    return cupy.asarray(out)


def _mk_classify_dask_cupy_chunk(block):
    """Dask chunk function: Mann-Kendall + classification (CPU, CuPy I/O)."""
    block_cpu = cupy.asnumpy(block)
    n_times, ny, nx = block_cpu.shape
    gi_bin_cpu = np.empty((n_times, ny, nx), dtype=np.int8)
    for t in range(n_times):
        gi_bin_cpu[t] = _calc_hotspots_numpy(block_cpu[t])
    category, trend_z, trend_p = _mk_classify_numpy(
        block_cpu, gi_bin_cpu, n_times, ny, nx
    )
    out = np.empty((3, ny, nx), dtype=np.float32)
    out[0] = category.astype(np.float32)
    out[1] = trend_z
    out[2] = trend_p
    return cupy.asarray(out)


def _emerging_hotspots_dask_cupy(raster, kernel, boundary='nan'):
    data = raster.data
    if not cupy.issubdtype(data.dtype, cupy.floating):
        data = data.astype(cupy.float32)

    n_times = data.shape[0]

    # Pass 1: eagerly compute global statistics (two scalars)
    global_mean, global_std = da.compute(da.nanmean(data), da.nanstd(data))
    global_mean = np.float32(float(global_mean))
    global_std = np.float32(float(global_std))

    if global_std == 0:
        raise ZeroDivisionError(
            "Standard deviation of the input raster values is 0."
        )

    norm_kernel = (kernel / kernel.sum()).astype(np.float32)
    pad_h = norm_kernel.shape[0] // 2
    pad_w = norm_kernel.shape[1] // 2

    # Pass 2: per time step, GPU convolve + z-score via map_overlap, then stack
    _func = partial(
        _convolve_and_zscore_cupy_chunk,
        kernel=norm_kernel,
        global_mean=global_mean,
        global_std=global_std,
    )
    zscore_slices = []
    for t in range(n_times):
        z_t = data[t].map_overlap(
            _func,
            depth=(pad_h, pad_w),
            boundary=_boundary_to_dask(boundary, is_cupy=True),
            meta=cupy.array((), dtype=cupy.float32),
        )
        zscore_slices.append(z_t)

    gi_zscore = da.stack(zscore_slices, axis=0)
    gi_zscore = gi_zscore.rechunk({0: n_times})

    # gi_bin via map_blocks (element-wise, no overlap needed)
    gi_bin = da.map_blocks(
        _gi_bin_cupy_chunk,
        gi_zscore,
        dtype=np.int8,
        meta=cupy.array((), dtype=cupy.int8),
    )

    # Pass 3: Mann-Kendall + classification via map_blocks
    mk_result = da.map_blocks(
        _mk_classify_dask_cupy_chunk,
        gi_zscore,
        dtype=np.float32,
        chunks=((3,), *gi_zscore.chunks[1:]),
        drop_axis=0,
        new_axis=0,
        meta=cupy.array((), dtype=cupy.float32),
    )

    return gi_zscore, gi_bin, mk_result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def emerging_hotspots(raster, kernel, boundary='nan'):
    """Classify time-series hot spot and cold spot trends.

    Combines the Getis-Ord Gi* statistic (per time step) with the
    Mann-Kendall non-parametric trend test to assign each pixel one
    of 17 categories describing how spatial clusters evolve over time.

    Parameters
    ----------
    raster : xarray.DataArray, 3-D ``(time, y, x)``
        Input values. Can be NumPy-backed, CuPy-backed, or
        Dask+NumPy-backed.
    kernel : numpy.ndarray, 2-D
        Spatial neighbourhood weights for the Gi* statistic.
    boundary : str, default ``'nan'``
        How to handle edges where the kernel extends beyond the
        raster. One of ``'nan'``, ``'nearest'``, ``'reflect'``,
        ``'wrap'``.

    Returns
    -------
    xarray.Dataset
        ``category``     : ``(y, x)`` int8 – trend code in [-8, 8].
        ``gi_zscore``    : ``(time, y, x)`` float32 – Gi* z-score per step.
        ``gi_bin``       : ``(time, y, x)`` int8 – confidence bin per step.
        ``trend_zscore`` : ``(y, x)`` float32 – Mann-Kendall Z.
        ``trend_pvalue`` : ``(y, x)`` float32 – Mann-Kendall p-value.

    Category codes
    --------------

    ======  ============  ======================================
    Code    Name          Rule
    ======  ============  ======================================
    1       New           Hot only at the final time step
    2       Consecutive   Hot for final N consecutive steps
                          (N>=2), never before
    3       Intensifying  Hot >=90% incl. final, MK up
    4       Persistent    Hot >=90% incl. final, MK flat
    5       Diminishing   Hot >=90% incl. final, MK down
    6       Sporadic      Hot at final, <90%, never cold
    7       Oscillating   Hot at final, was cold at least once
    8       Historical    Hot >=90% of steps, NOT at final
    -1..-8                Cold spot mirrors of 1-8
    0       No Pattern    None of the above
    ======  ============  ======================================
    """
    if not isinstance(raster, xr.DataArray):
        raise TypeError("`raster` must be an xarray.DataArray")

    if raster.ndim != 3:
        raise ValueError(
            "`raster` must be 3-D (time, y, x), "
            f"got {raster.ndim}-D"
        )

    if raster.shape[0] < 2:
        raise ValueError(
            "Need at least 2 time steps, got "
            f"{raster.shape[0]}"
        )

    _validate_boundary(boundary)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=partial(_emerging_hotspots_numpy, boundary=boundary),
        cupy_func=partial(_emerging_hotspots_cupy, boundary=boundary),
        dask_func=partial(_emerging_hotspots_dask_numpy, boundary=boundary),
        dask_cupy_func=partial(
            _emerging_hotspots_dask_cupy, boundary=boundary,
        ),
    )

    result = mapper(raster)(raster, kernel)

    dims_3d = raster.dims
    dims_2d = raster.dims[1:]
    coords_2d = {k: v for k, v in raster.coords.items()
                 if k in dims_2d or set(v.dims).issubset(set(dims_2d))}
    coords_3d = raster.coords

    if da is not None and isinstance(raster.data, da.Array):
        gi_zscore, gi_bin, mk_result = result
        category = mk_result[0].astype(np.int8)
        trend_zscore = mk_result[1]
        trend_pvalue = mk_result[2]
    else:
        gi_zscore, gi_bin, category, trend_zscore, trend_pvalue = result

    ds = xr.Dataset(
        {
            'category': xr.DataArray(category, dims=dims_2d,
                                     coords=coords_2d),
            'gi_zscore': xr.DataArray(gi_zscore, dims=dims_3d,
                                      coords=coords_3d),
            'gi_bin': xr.DataArray(gi_bin, dims=dims_3d,
                                   coords=coords_3d),
            'trend_zscore': xr.DataArray(trend_zscore, dims=dims_2d,
                                         coords=coords_2d),
            'trend_pvalue': xr.DataArray(trend_pvalue, dims=dims_2d,
                                         coords=coords_2d),
        }
    )
    return ds
