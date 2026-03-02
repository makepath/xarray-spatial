"""Fire behavior, burn severity, and fire danger functions.

Provides per-cell raster operations for wildfire analysis:

Burn severity
    - :func:`dnbr` -- differenced Normalized Burn Ratio
    - :func:`rdnbr` -- relative differenced Normalized Burn Ratio
    - :func:`burn_severity_class` -- USGS 7-class burn severity

Fire behavior
    - :func:`fireline_intensity` -- Byram's fireline intensity (kW/m)
    - :func:`flame_length` -- flame length from intensity (m)
    - :func:`rate_of_spread` -- simplified Rothermel spread rate (m/min)

Fire danger
    - :func:`kbdi` -- Keetch-Byram Drought Index (single time-step)
"""
from __future__ import annotations

from math import exp, log, sqrt, tan

import numpy as np
import xarray as xr
from numba import cuda

from xrspatial.dataset_support import supports_dataset
from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _validate_raster,
    _validate_scalar,
    cuda_args,
    ngjit,
    validate_arrays,
)

# 3rd-party (optional)
try:
    import cupy
except ImportError:
    class cupy:
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None


# ============================================================================
# Anderson 13 fuel model table (metric units)
# ============================================================================
# Columns: fuel_load (kg/m^2), SAV (1/m), bed_depth (m), moisture_ext
#           (fraction), heat_content (kJ/kg), S_T, S_e, rho_p (kg/m^3)
#
# Converted from Anderson (1982) "Aids to Determining Fuel Models for
# Estimating Fire Behavior", USDA Forest Service Gen. Tech. Rep. INT-122.
# Original imperial values converted: tons/acre -> kg/m^2 (* 0.22417),
# ft -> m (* 0.3048), Btu/lb -> kJ/kg (* 2.326).
ANDERSON_13 = np.array([
    # Model  1: short grass (1 ft)
    [0.1662, 11483., 0.3048, 0.12, 18608., 0.0555, 0.010, 513.],
    # Model  2: timber (grass and understory)
    [0.8983, 9186., 0.3048, 0.15, 18608., 0.0555, 0.010, 513.],
    # Model  3: tall grass (2.5 ft)
    [0.6730, 4921., 0.7620, 0.25, 18608., 0.0555, 0.010, 513.],
    # Model  4: chaparral (6 ft)
    [2.6810, 6562., 1.8288, 0.20, 18608., 0.0555, 0.010, 513.],
    # Model  5: brush (2 ft)
    [0.2242, 6562., 0.6096, 0.20, 18608., 0.0555, 0.010, 513.],
    # Model  6: dormant brush / hardwood slash
    [0.3362, 5741., 0.7620, 0.25, 18608., 0.0555, 0.010, 513.],
    # Model  7: southern rough
    [0.2578, 5741., 0.7620, 0.40, 18608., 0.0555, 0.010, 513.],
    # Model  8: closed timber litter
    [0.2242, 6562., 0.0610, 0.30, 18608., 0.0555, 0.010, 513.],
    # Model  9: hardwood litter
    [0.1541, 8202., 0.0610, 0.25, 18608., 0.0555, 0.010, 513.],
    # Model 10: timber (litter and understory)
    [0.6730, 6562., 0.3048, 0.25, 18608., 0.0555, 0.010, 513.],
    # Model 11: light logging slash
    [0.3362, 4921., 0.3048, 0.15, 18608., 0.0555, 0.010, 513.],
    # Model 12: medium logging slash
    [0.8983, 4921., 0.7010, 0.20, 18608., 0.0555, 0.010, 513.],
    # Model 13: heavy logging slash
    [1.5714, 4921., 0.9144, 0.25, 18608., 0.0555, 0.010, 513.],
], dtype=np.float64)


# ============================================================================
# Rothermel pre-computation (scalar fuel-model params -> derived constants)
# ============================================================================
def _rothermel_fuel_constants(fuel_model: int):
    """Pre-compute Rothermel constants from an Anderson-13 fuel model row.

    Returns a tuple of 12 float64 scalars that get passed into the per-pixel
    kernel so only moisture/wind/slope vary per cell.
    """
    row = ANDERSON_13[fuel_model - 1]
    w_0 = row[0]       # fuel load, kg/m^2
    sigma = row[1]      # SAV ratio, 1/m
    delta = row[2]      # bed depth, m
    M_x = row[3]        # moisture of extinction, fraction
    h = row[4]          # heat content, kJ/kg
    S_T = row[5]        # total mineral content
    S_e = row[6]        # effective mineral content
    rho_p = row[7]      # oven-dry particle density, kg/m^3

    # packing ratio
    rho_b = w_0 / delta if delta > 0 else 0.0
    beta = rho_b / rho_p if rho_p > 0 else 0.0
    beta_op = 3.348 * sigma ** (-0.8189)
    ratio = beta / beta_op if beta_op > 0 else 0.0

    # maximum reaction velocity
    A = 133.0 * sigma ** (-0.7913)
    Gamma_max = sigma ** 1.5 / (495.0 + 0.0594 * sigma ** 1.5)
    Gamma = Gamma_max * (ratio ** A) * exp(A * (1.0 - ratio))

    # mineral damping
    eta_s = max(0.174 * S_e ** (-0.19), 0.0)
    if eta_s > 1.0:
        eta_s = 1.0

    # propagating flux ratio
    xi = exp((0.792 + 0.681 * sigma ** 0.5) * (beta + 0.1)) / \
        (192.0 + 0.2595 * sigma)

    # heat of pre-ignition
    epsilon = exp(-138.0 / sigma) if sigma > 0 else 0.0

    # wind factor exponents
    C_w = 7.47 * exp(-0.133 * sigma ** 0.55)
    B_w = 0.02526 * sigma ** 0.54
    E_w = 0.715 * exp(-3.59e-4 * sigma)

    return (w_0, h, M_x, beta, rho_b, Gamma, eta_s, xi, epsilon,
            C_w, B_w, E_w)


# ============================================================================
# dNBR
# ============================================================================
@ngjit
def _dnbr_cpu(pre_data, post_data):
    out = np.full(pre_data.shape, np.nan, dtype=np.float32)
    rows, cols = pre_data.shape
    for y in range(rows):
        for x in range(cols):
            pre = pre_data[y, x]
            post = post_data[y, x]
            if pre != pre or post != post:
                continue
            out[y, x] = pre - post
    return out


@cuda.jit
def _dnbr_gpu(pre_data, post_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        pre = pre_data[y, x]
        post = post_data[y, x]
        if pre == pre and post == post:
            out[y, x] = pre - post


def _dnbr_dask(pre_data, post_data):
    return da.map_blocks(_dnbr_cpu, pre_data, post_data, meta=np.array(()))


def _dnbr_cupy(pre_data, post_data):
    griddim, blockdim = cuda_args(pre_data.shape)
    out = cupy.empty(pre_data.shape, dtype='f4')
    out[:] = cupy.nan
    _dnbr_gpu[griddim, blockdim](pre_data, post_data, out)
    return out


def _dnbr_dask_cupy(pre_data, post_data):
    return da.map_blocks(_dnbr_cupy, pre_data, post_data,
                         dtype=cupy.float32, meta=cupy.array(()))


def dnbr(pre_nbr_agg: xr.DataArray,
         post_nbr_agg: xr.DataArray,
         name: str = 'dnbr') -> xr.DataArray:
    """Differenced Normalized Burn Ratio (dNBR).

    Computes ``pre_nbr - post_nbr``.  Higher values indicate greater
    burn severity.

    Parameters
    ----------
    pre_nbr_agg : xr.DataArray
        Pre-fire NBR values.
    post_nbr_agg : xr.DataArray
        Post-fire NBR values.
    name : str, default='dnbr'
        Name of output DataArray.

    Returns
    -------
    xr.DataArray
        dNBR values (float32).
    """
    _validate_raster(pre_nbr_agg, func_name='dnbr', name='pre_nbr_agg')
    _validate_raster(post_nbr_agg, func_name='dnbr', name='post_nbr_agg')
    validate_arrays(pre_nbr_agg, post_nbr_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_dnbr_cpu,
        cupy_func=_dnbr_cupy,
        dask_func=_dnbr_dask,
        dask_cupy_func=_dnbr_dask_cupy,
    )
    out = mapper(pre_nbr_agg)(
        pre_nbr_agg.data.astype('f4'),
        post_nbr_agg.data.astype('f4'),
    )
    return xr.DataArray(out, name=name,
                        coords=pre_nbr_agg.coords,
                        dims=pre_nbr_agg.dims,
                        attrs=pre_nbr_agg.attrs)


# ============================================================================
# RdNBR
# ============================================================================
@ngjit
def _rdnbr_cpu(dnbr_data, pre_data):
    out = np.full(dnbr_data.shape, np.nan, dtype=np.float32)
    rows, cols = dnbr_data.shape
    for y in range(rows):
        for x in range(cols):
            d = dnbr_data[y, x]
            p = pre_data[y, x]
            if d != d or p != p:
                continue
            denom = abs(p / 1000.0)
            if denom < 1e-10:
                continue
            out[y, x] = d / sqrt(denom)
    return out


@cuda.jit
def _rdnbr_gpu(dnbr_data, pre_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        d = dnbr_data[y, x]
        p = pre_data[y, x]
        if d == d and p == p:
            denom = abs(p / 1000.0)
            if denom >= 1e-10:
                out[y, x] = d / (denom ** 0.5)


def _rdnbr_dask(dnbr_data, pre_data):
    return da.map_blocks(_rdnbr_cpu, dnbr_data, pre_data, meta=np.array(()))


def _rdnbr_cupy(dnbr_data, pre_data):
    griddim, blockdim = cuda_args(dnbr_data.shape)
    out = cupy.empty(dnbr_data.shape, dtype='f4')
    out[:] = cupy.nan
    _rdnbr_gpu[griddim, blockdim](dnbr_data, pre_data, out)
    return out


def _rdnbr_dask_cupy(dnbr_data, pre_data):
    return da.map_blocks(_rdnbr_cupy, dnbr_data, pre_data,
                         dtype=cupy.float32, meta=cupy.array(()))


def rdnbr(dnbr_agg: xr.DataArray,
          pre_nbr_agg: xr.DataArray,
          name: str = 'rdnbr') -> xr.DataArray:
    """Relative differenced Normalized Burn Ratio (RdNBR).

    Computes ``dNBR / sqrt(abs(pre_NBR / 1000))``.  Normalises dNBR by
    pre-fire vegetation density so that burn severity is comparable
    across different vegetation types.

    Parameters
    ----------
    dnbr_agg : xr.DataArray
        dNBR values (e.g. from :func:`dnbr`).
    pre_nbr_agg : xr.DataArray
        Pre-fire NBR values.
    name : str, default='rdnbr'
        Name of output DataArray.

    Returns
    -------
    xr.DataArray
        RdNBR values (float32).  Pixels where ``abs(pre_NBR) < 1e-7``
        are set to NaN to avoid division by near-zero.
    """
    _validate_raster(dnbr_agg, func_name='rdnbr', name='dnbr_agg')
    _validate_raster(pre_nbr_agg, func_name='rdnbr', name='pre_nbr_agg')
    validate_arrays(dnbr_agg, pre_nbr_agg)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_rdnbr_cpu,
        cupy_func=_rdnbr_cupy,
        dask_func=_rdnbr_dask,
        dask_cupy_func=_rdnbr_dask_cupy,
    )
    out = mapper(dnbr_agg)(
        dnbr_agg.data.astype('f4'),
        pre_nbr_agg.data.astype('f4'),
    )
    return xr.DataArray(out, name=name,
                        coords=dnbr_agg.coords,
                        dims=dnbr_agg.dims,
                        attrs=dnbr_agg.attrs)


# ============================================================================
# Burn severity classification (USGS 7-class)
# ============================================================================
# Class thresholds from USGS:
#   1: Enhanced regrowth (high)  dNBR < -0.251
#   2: Enhanced regrowth (low)   -0.251 <= dNBR < -0.101
#   3: Unburned                  -0.101 <= dNBR <  0.099
#   4: Low severity               0.099 <= dNBR <  0.269
#   5: Moderate-low severity      0.269 <= dNBR <  0.439
#   6: Moderate-high severity     0.439 <= dNBR <  0.659
#   7: High severity              0.659 <= dNBR
#   0: nodata (NaN input)

_BSC_THRESHOLDS = np.array([-0.251, -0.101, 0.099, 0.269, 0.439, 0.659],
                           dtype=np.float64)


@ngjit
def _bsc_cpu(data):
    out = np.zeros(data.shape, dtype=np.int8)
    rows, cols = data.shape
    for y in range(rows):
        for x in range(cols):
            v = data[y, x]
            if v != v:
                continue  # 0 = nodata
            if v < -0.251:
                out[y, x] = 1
            elif v < -0.101:
                out[y, x] = 2
            elif v < 0.099:
                out[y, x] = 3
            elif v < 0.269:
                out[y, x] = 4
            elif v < 0.439:
                out[y, x] = 5
            elif v < 0.659:
                out[y, x] = 6
            else:
                out[y, x] = 7
    return out


@cuda.jit
def _bsc_gpu(data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        v = data[y, x]
        if v == v:
            if v < -0.251:
                out[y, x] = 1
            elif v < -0.101:
                out[y, x] = 2
            elif v < 0.099:
                out[y, x] = 3
            elif v < 0.269:
                out[y, x] = 4
            elif v < 0.439:
                out[y, x] = 5
            elif v < 0.659:
                out[y, x] = 6
            else:
                out[y, x] = 7


def _bsc_dask(data):
    return da.map_blocks(_bsc_cpu, data, dtype=np.int8, meta=np.array(()))


def _bsc_cupy(data):
    griddim, blockdim = cuda_args(data.shape)
    out = cupy.zeros(data.shape, dtype=np.int8)
    _bsc_gpu[griddim, blockdim](data, out)
    return out


def _bsc_dask_cupy(data):
    return da.map_blocks(_bsc_cupy, data,
                         dtype=np.int8, meta=cupy.array(()))


@supports_dataset
def burn_severity_class(dnbr_agg: xr.DataArray,
                        name: str = 'burn_severity_class') -> xr.DataArray:
    """Classify dNBR into USGS 7-class burn severity.

    Classes:
        1 = enhanced regrowth (high), 2 = enhanced regrowth (low),
        3 = unburned, 4 = low severity, 5 = moderate-low,
        6 = moderate-high, 7 = high severity.  0 = nodata.

    Parameters
    ----------
    dnbr_agg : xr.DataArray or xr.Dataset
        dNBR values (e.g. from :func:`dnbr`).
    name : str, default='burn_severity_class'
        Name of output DataArray.

    Returns
    -------
    xr.DataArray
        int8 class labels (0-7).
    """
    _validate_raster(dnbr_agg, func_name='burn_severity_class',
                     name='dnbr_agg')

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_bsc_cpu,
        cupy_func=_bsc_cupy,
        dask_func=_bsc_dask,
        dask_cupy_func=_bsc_dask_cupy,
    )
    out = mapper(dnbr_agg)(dnbr_agg.data.astype('f4'))
    return xr.DataArray(out, name=name,
                        coords=dnbr_agg.coords,
                        dims=dnbr_agg.dims,
                        attrs=dnbr_agg.attrs)


# ============================================================================
# Fireline intensity (Byram)
# ============================================================================
@ngjit
def _fli_cpu(fuel_data, spread_data, heat_content):
    out = np.full(fuel_data.shape, np.nan, dtype=np.float32)
    rows, cols = fuel_data.shape
    for y in range(rows):
        for x in range(cols):
            w = fuel_data[y, x]
            r = spread_data[y, x]
            if w != w or r != r:
                continue
            out[y, x] = heat_content * w * r
    return out


@cuda.jit
def _fli_gpu(fuel_data, spread_data, heat_content, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        w = fuel_data[y, x]
        r = spread_data[y, x]
        if w == w and r == r:
            out[y, x] = heat_content * w * r


def _fli_dask(fuel_data, spread_data, heat_content):
    return da.map_blocks(_fli_cpu, fuel_data, spread_data, heat_content,
                         meta=np.array(()))


def _fli_cupy(fuel_data, spread_data, heat_content):
    griddim, blockdim = cuda_args(fuel_data.shape)
    out = cupy.empty(fuel_data.shape, dtype='f4')
    out[:] = cupy.nan
    _fli_gpu[griddim, blockdim](fuel_data, spread_data, heat_content, out)
    return out


def _fli_dask_cupy(fuel_data, spread_data, heat_content):
    return da.map_blocks(_fli_cupy, fuel_data, spread_data, heat_content,
                         dtype=cupy.float32, meta=cupy.array(()))


def fireline_intensity(fuel_consumed_agg: xr.DataArray,
                       spread_rate_agg: xr.DataArray,
                       heat_content: float = 18000.0,
                       name: str = 'fireline_intensity') -> xr.DataArray:
    """Byram's fireline intensity.

    ``I = H * w * R`` where *H* is heat content (kJ/kg), *w* is fuel
    consumed (kg/m^2), and *R* is rate of spread (m/s).

    Parameters
    ----------
    fuel_consumed_agg : xr.DataArray
        Fuel consumed per unit area (kg/m^2).
    spread_rate_agg : xr.DataArray
        Rate of spread (m/s).
    heat_content : float, default=18000
        Heat content of fuel (kJ/kg).
    name : str, default='fireline_intensity'
        Name of output DataArray.

    Returns
    -------
    xr.DataArray
        Fireline intensity in kW/m (float32).
    """
    _validate_raster(fuel_consumed_agg, func_name='fireline_intensity',
                     name='fuel_consumed_agg')
    _validate_raster(spread_rate_agg, func_name='fireline_intensity',
                     name='spread_rate_agg')
    validate_arrays(fuel_consumed_agg, spread_rate_agg)
    _validate_scalar(heat_content, func_name='fireline_intensity',
                     name='heat_content', dtype=(int, float), min_val=0)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_fli_cpu,
        cupy_func=_fli_cupy,
        dask_func=_fli_dask,
        dask_cupy_func=_fli_dask_cupy,
    )
    out = mapper(fuel_consumed_agg)(
        fuel_consumed_agg.data.astype('f4'),
        spread_rate_agg.data.astype('f4'),
        float(heat_content),
    )
    return xr.DataArray(out, name=name,
                        coords=fuel_consumed_agg.coords,
                        dims=fuel_consumed_agg.dims,
                        attrs=fuel_consumed_agg.attrs)


# ============================================================================
# Flame length
# ============================================================================
@ngjit
def _fl_cpu(intensity_data):
    out = np.full(intensity_data.shape, np.nan, dtype=np.float32)
    rows, cols = intensity_data.shape
    for y in range(rows):
        for x in range(cols):
            v = intensity_data[y, x]
            if v != v:
                continue
            if v <= 0.0:
                out[y, x] = 0.0
            else:
                out[y, x] = 0.0775 * v ** 0.46
    return out


@cuda.jit
def _fl_gpu(intensity_data, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        v = intensity_data[y, x]
        if v == v:
            if v <= 0.0:
                out[y, x] = 0.0
            else:
                out[y, x] = 0.0775 * v ** 0.46


def _fl_dask(intensity_data):
    return da.map_blocks(_fl_cpu, intensity_data, meta=np.array(()))


def _fl_cupy(intensity_data):
    griddim, blockdim = cuda_args(intensity_data.shape)
    out = cupy.empty(intensity_data.shape, dtype='f4')
    out[:] = cupy.nan
    _fl_gpu[griddim, blockdim](intensity_data, out)
    return out


def _fl_dask_cupy(intensity_data):
    return da.map_blocks(_fl_cupy, intensity_data,
                         dtype=cupy.float32, meta=cupy.array(()))


@supports_dataset
def flame_length(intensity_agg: xr.DataArray,
                 name: str = 'flame_length') -> xr.DataArray:
    """Flame length from fireline intensity.

    Uses Byram's equation: ``L = 0.0775 * I^0.46``.
    Negative or zero intensity yields zero flame length.

    Parameters
    ----------
    intensity_agg : xr.DataArray or xr.Dataset
        Fireline intensity (kW/m).
    name : str, default='flame_length'
        Name of output DataArray.

    Returns
    -------
    xr.DataArray
        Flame length in metres (float32).
    """
    _validate_raster(intensity_agg, func_name='flame_length',
                     name='intensity_agg')

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_fl_cpu,
        cupy_func=_fl_cupy,
        dask_func=_fl_dask,
        dask_cupy_func=_fl_dask_cupy,
    )
    out = mapper(intensity_agg)(intensity_agg.data.astype('f4'))
    return xr.DataArray(out, name=name,
                        coords=intensity_agg.coords,
                        dims=intensity_agg.dims,
                        attrs=intensity_agg.attrs)


# ============================================================================
# Rate of spread (simplified Rothermel)
# ============================================================================
@ngjit
def _ros_cpu(slope_data, wind_data, moisture_data,
             w_0, h, M_x, beta, rho_b, Gamma, eta_s, xi, epsilon,
             C_w, B_w, E_w):
    out = np.full(slope_data.shape, np.nan, dtype=np.float32)
    rows, cols = slope_data.shape

    # pre-ignition heat (constant)
    Q_ig = 250.0 + 1116.0 * 0.0  # moisture term added per-pixel

    for y in range(rows):
        for x in range(cols):
            slp = slope_data[y, x]
            wind = wind_data[y, x]
            moist = moisture_data[y, x]
            if slp != slp or wind != wind or moist != moist:
                continue

            # moisture ratio
            rm = moist / M_x if M_x > 0 else 0.0
            if rm < 0.0:
                rm = 0.0

            # moisture damping coefficient
            eta_M = 1.0 - 2.59 * rm + 5.11 * rm * rm - 3.52 * rm * rm * rm
            if eta_M < 0.0:
                eta_M = 0.0
            if eta_M > 1.0:
                eta_M = 1.0

            # reaction intensity (kJ/m^2/min)
            I_R = Gamma * w_0 * h * eta_M * eta_s

            # wind factor (convert km/h to m/min: * 1000/60)
            U_mmin = wind * 1000.0 / 60.0
            if U_mmin < 0.0:
                U_mmin = 0.0
            phi_w = C_w * U_mmin ** B_w
            if beta > 0.0:
                phi_w = phi_w * beta ** (-E_w)

            # slope factor
            tan_slp = tan(slp * 3.141592653589793 / 180.0)
            phi_s = 5.275 * (beta ** (-0.3)) * tan_slp * tan_slp if beta > 0 else 0.0

            # heat of pre-ignition
            Q_ig = 250.0 + 1116.0 * moist

            # rate of spread (m/min)
            denom = rho_b * epsilon * Q_ig
            if denom > 0.0:
                R = I_R * xi * (1.0 + phi_w + phi_s) / denom
            else:
                R = 0.0

            if R < 0.0:
                R = 0.0
            out[y, x] = R
    return out


@cuda.jit
def _ros_gpu(slope_data, wind_data, moisture_data,
             w_0, h, M_x, beta, rho_b, Gamma, eta_s, xi, epsilon,
             C_w, B_w, E_w, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        slp = slope_data[y, x]
        wind = wind_data[y, x]
        moist = moisture_data[y, x]
        if slp != slp or wind != wind or moist != moist:
            return

        rm = moist / M_x if M_x > 0 else 0.0
        if rm < 0.0:
            rm = 0.0

        eta_M = 1.0 - 2.59 * rm + 5.11 * rm * rm - 3.52 * rm * rm * rm
        if eta_M < 0.0:
            eta_M = 0.0
        if eta_M > 1.0:
            eta_M = 1.0

        I_R = Gamma * w_0 * h * eta_M * eta_s

        U_mmin = wind * 1000.0 / 60.0
        if U_mmin < 0.0:
            U_mmin = 0.0
        phi_w = C_w * U_mmin ** B_w
        if beta > 0.0:
            phi_w = phi_w * beta ** (-E_w)

        tan_slp = tan(slp * 3.141592653589793 / 180.0)
        phi_s = 5.275 * (beta ** (-0.3)) * tan_slp * tan_slp if beta > 0 else 0.0

        Q_ig = 250.0 + 1116.0 * moist

        denom = rho_b * epsilon * Q_ig
        if denom > 0.0:
            R = I_R * xi * (1.0 + phi_w + phi_s) / denom
        else:
            R = 0.0

        if R < 0.0:
            R = 0.0
        out[y, x] = R


def _ros_dask(slope_data, wind_data, moisture_data,
              w_0, h, M_x, beta, rho_b, Gamma, eta_s, xi, epsilon,
              C_w, B_w, E_w):
    return da.map_blocks(
        _ros_cpu, slope_data, wind_data, moisture_data,
        w_0, h, M_x, beta, rho_b, Gamma, eta_s, xi, epsilon,
        C_w, B_w, E_w,
        meta=np.array(()),
    )


def _ros_cupy(slope_data, wind_data, moisture_data,
              w_0, h, M_x, beta, rho_b, Gamma, eta_s, xi, epsilon,
              C_w, B_w, E_w):
    griddim, blockdim = cuda_args(slope_data.shape)
    out = cupy.empty(slope_data.shape, dtype='f4')
    out[:] = cupy.nan
    _ros_gpu[griddim, blockdim](
        slope_data, wind_data, moisture_data,
        w_0, h, M_x, beta, rho_b, Gamma, eta_s, xi, epsilon,
        C_w, B_w, E_w, out,
    )
    return out


def _ros_dask_cupy(slope_data, wind_data, moisture_data,
                   w_0, h, M_x, beta, rho_b, Gamma, eta_s, xi, epsilon,
                   C_w, B_w, E_w):
    return da.map_blocks(
        _ros_cupy, slope_data, wind_data, moisture_data,
        w_0, h, M_x, beta, rho_b, Gamma, eta_s, xi, epsilon,
        C_w, B_w, E_w,
        dtype=cupy.float32, meta=cupy.array(()),
    )


def rate_of_spread(slope_agg: xr.DataArray,
                   wind_speed_agg: xr.DataArray,
                   fuel_moisture_agg: xr.DataArray,
                   fuel_model: int = 1,
                   name: str = 'rate_of_spread') -> xr.DataArray:
    """Rate of spread using a simplified Rothermel model.

    Uses the Anderson 13 fuel model lookup table and computes per-pixel
    spread rate from slope, wind speed, and fuel moisture.

    Parameters
    ----------
    slope_agg : xr.DataArray
        Terrain slope in degrees.
    wind_speed_agg : xr.DataArray
        Mid-flame wind speed in km/h.
    fuel_moisture_agg : xr.DataArray
        Dead fuel moisture content as a fraction (0-1).
    fuel_model : int, default=1
        Anderson 13 fuel model number (1-13).
    name : str, default='rate_of_spread'
        Name of output DataArray.

    Returns
    -------
    xr.DataArray
        Rate of spread in m/min (float32).
    """
    _validate_raster(slope_agg, func_name='rate_of_spread',
                     name='slope_agg')
    _validate_raster(wind_speed_agg, func_name='rate_of_spread',
                     name='wind_speed_agg')
    _validate_raster(fuel_moisture_agg, func_name='rate_of_spread',
                     name='fuel_moisture_agg')
    validate_arrays(slope_agg, wind_speed_agg, fuel_moisture_agg)
    _validate_scalar(fuel_model, func_name='rate_of_spread',
                     name='fuel_model', dtype=int, min_val=1, max_val=13)

    constants = _rothermel_fuel_constants(fuel_model)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_ros_cpu,
        cupy_func=_ros_cupy,
        dask_func=_ros_dask,
        dask_cupy_func=_ros_dask_cupy,
    )
    out = mapper(slope_agg)(
        slope_agg.data.astype('f4'),
        wind_speed_agg.data.astype('f4'),
        fuel_moisture_agg.data.astype('f4'),
        *constants,
    )
    return xr.DataArray(out, name=name,
                        coords=slope_agg.coords,
                        dims=slope_agg.dims,
                        attrs=slope_agg.attrs)


# ============================================================================
# KBDI (Keetch-Byram Drought Index, single time-step)
# ============================================================================
@ngjit
def _kbdi_cpu(kbdi_prev_data, max_temp_data, precip_data, annual_precip):
    out = np.full(kbdi_prev_data.shape, np.nan, dtype=np.float32)
    rows, cols = kbdi_prev_data.shape
    for y in range(rows):
        for x in range(cols):
            Q_prev = kbdi_prev_data[y, x]
            T = max_temp_data[y, x]
            P = precip_data[y, x]
            if Q_prev != Q_prev or T != T or P != P:
                continue

            # net precipitation (subtract 5.08 mm canopy interception)
            net_P = P - 5.08
            if net_P < 0.0:
                net_P = 0.0

            # reduce drought index by net precip
            Q = Q_prev - net_P
            if Q < 0.0:
                Q = 0.0

            # drought factor (only accumulates when T > 50 F = 10 C)
            if T > 10.0:
                # convert Celsius to Fahrenheit for empirical formula
                T_f = T * 9.0 / 5.0 + 32.0
                R = annual_precip  # mm
                # KBDI drought factor (metric form)
                numer = (800.0 - Q) * (0.968 * exp(0.0486 * T_f) - 8.30)
                denom = 1.0 + 10.46 * exp(-0.0116 * R)
                dQ = numer * 0.001 / denom
                if dQ < 0.0:
                    dQ = 0.0
                Q = Q + dQ

            # clamp to 0-800
            if Q < 0.0:
                Q = 0.0
            if Q > 800.0:
                Q = 800.0

            out[y, x] = Q
    return out


@cuda.jit
def _kbdi_gpu(kbdi_prev_data, max_temp_data, precip_data, annual_precip, out):
    y, x = cuda.grid(2)
    if y < out.shape[0] and x < out.shape[1]:
        Q_prev = kbdi_prev_data[y, x]
        T = max_temp_data[y, x]
        P = precip_data[y, x]
        if Q_prev != Q_prev or T != T or P != P:
            return

        net_P = P - 5.08
        if net_P < 0.0:
            net_P = 0.0

        Q = Q_prev - net_P
        if Q < 0.0:
            Q = 0.0

        if T > 10.0:
            T_f = T * 9.0 / 5.0 + 32.0
            R = annual_precip
            numer = (800.0 - Q) * (0.968 * exp(0.0486 * T_f) - 8.30)
            denom = 1.0 + 10.46 * exp(-0.0116 * R)
            dQ = numer * 0.001 / denom
            if dQ < 0.0:
                dQ = 0.0
            Q = Q + dQ

        if Q < 0.0:
            Q = 0.0
        if Q > 800.0:
            Q = 800.0

        out[y, x] = Q


def _kbdi_dask(kbdi_prev_data, max_temp_data, precip_data, annual_precip):
    return da.map_blocks(
        _kbdi_cpu, kbdi_prev_data, max_temp_data, precip_data, annual_precip,
        meta=np.array(()),
    )


def _kbdi_cupy(kbdi_prev_data, max_temp_data, precip_data, annual_precip):
    griddim, blockdim = cuda_args(kbdi_prev_data.shape)
    out = cupy.empty(kbdi_prev_data.shape, dtype='f4')
    out[:] = cupy.nan
    _kbdi_gpu[griddim, blockdim](
        kbdi_prev_data, max_temp_data, precip_data, annual_precip, out,
    )
    return out


def _kbdi_dask_cupy(kbdi_prev_data, max_temp_data, precip_data,
                    annual_precip):
    return da.map_blocks(
        _kbdi_cupy, kbdi_prev_data, max_temp_data, precip_data,
        annual_precip,
        dtype=cupy.float32, meta=cupy.array(()),
    )


def kbdi(kbdi_prev_agg: xr.DataArray,
         max_temp_agg: xr.DataArray,
         precip_agg: xr.DataArray,
         annual_precip: float,
         name: str = 'kbdi') -> xr.DataArray:
    """Keetch-Byram Drought Index (single time-step update).

    Advances the KBDI by one day given previous KBDI, daily max
    temperature, and daily precipitation.

    Parameters
    ----------
    kbdi_prev_agg : xr.DataArray
        Previous day's KBDI values (0-800 mm deficit).
    max_temp_agg : xr.DataArray
        Daily maximum temperature in degrees Celsius.
    precip_agg : xr.DataArray
        Daily precipitation in mm.
    annual_precip : float
        Mean annual precipitation in mm (scalar, constant across domain).
    name : str, default='kbdi'
        Name of output DataArray.

    Returns
    -------
    xr.DataArray
        Updated KBDI values (float32), clamped to 0-800.
    """
    _validate_raster(kbdi_prev_agg, func_name='kbdi', name='kbdi_prev_agg')
    _validate_raster(max_temp_agg, func_name='kbdi', name='max_temp_agg')
    _validate_raster(precip_agg, func_name='kbdi', name='precip_agg')
    validate_arrays(kbdi_prev_agg, max_temp_agg, precip_agg)
    _validate_scalar(annual_precip, func_name='kbdi',
                     name='annual_precip', dtype=(int, float),
                     min_val=0, min_exclusive=True)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_kbdi_cpu,
        cupy_func=_kbdi_cupy,
        dask_func=_kbdi_dask,
        dask_cupy_func=_kbdi_dask_cupy,
    )
    out = mapper(kbdi_prev_agg)(
        kbdi_prev_agg.data.astype('f4'),
        max_temp_agg.data.astype('f4'),
        precip_agg.data.astype('f4'),
        float(annual_precip),
    )
    return xr.DataArray(out, name=name,
                        coords=kbdi_prev_agg.coords,
                        dims=kbdi_prev_agg.dims,
                        attrs=kbdi_prev_agg.attrs)
