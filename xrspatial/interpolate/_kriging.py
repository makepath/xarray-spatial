"""Ordinary Kriging interpolation."""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _validate_raster,
    _validate_scalar,
)

from ._validation import extract_grid_coords, validate_points

try:
    import cupy
except ImportError:
    cupy = None

try:
    import dask.array as da
except ImportError:
    da = None


def _get_xp(arr):
    """Return the array module (numpy or cupy) for *arr*."""
    if cupy is not None and isinstance(arr, cupy.ndarray):
        return cupy
    return np


# ---------------------------------------------------------------------------
# Variogram models
# ---------------------------------------------------------------------------

def _spherical(h, c0, c, a):
    xp = _get_xp(h)
    return xp.where(
        h < a,
        c0 + c * (1.5 * h / a - 0.5 * (h / a) ** 3),
        c0 + c,
    )


def _exponential(h, c0, c, a):
    xp = _get_xp(h)
    return c0 + c * (1.0 - xp.exp(-3.0 * h / a))


def _gaussian(h, c0, c, a):
    xp = _get_xp(h)
    return c0 + c * (1.0 - xp.exp(-3.0 * (h / a) ** 2))


_VARIOGRAM_MODELS = {
    'spherical': _spherical,
    'exponential': _exponential,
    'gaussian': _gaussian,
}


# ---------------------------------------------------------------------------
# Experimental variogram
# ---------------------------------------------------------------------------

def _experimental_variogram(x, y, z, nlags):
    """Compute binned experimental variogram from point data."""
    n = len(x)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dists = np.sqrt(dx ** 2 + dy ** 2)
    semivar = 0.5 * (z[i_idx] - z[j_idx]) ** 2

    max_dist = dists.max() / 2.0
    if max_dist <= 0:
        return np.array([]), np.array([])

    bins = np.linspace(0, max_dist, nlags + 1)
    lag_centers = 0.5 * (bins[:-1] + bins[1:])
    lag_sv = np.zeros(nlags, dtype=np.float64)
    lag_count = np.zeros(nlags, dtype=np.int64)

    bin_idx = np.digitize(dists, bins) - 1
    for k in range(nlags):
        mask = bin_idx == k
        if mask.any():
            lag_sv[k] = semivar[mask].mean()
            lag_count[k] = mask.sum()

    valid = lag_count > 0
    return lag_centers[valid], lag_sv[valid]


# ---------------------------------------------------------------------------
# Variogram fitting
# ---------------------------------------------------------------------------

def _fit_variogram(lag_h, lag_sv, model_func):
    """Fit variogram model parameters via curve_fit.

    Returns (c0, c, a).
    """
    from scipy.optimize import curve_fit

    c0_init = float(lag_sv[0]) if len(lag_sv) > 0 else 0.0
    c_init = float(lag_sv.max() - c0_init) if len(lag_sv) > 0 else 1.0
    a_init = float(lag_h[-1]) if len(lag_h) > 0 else 1.0

    c_init = max(c_init, 1e-10)
    a_init = max(a_init, 1e-10)

    try:
        popt, _ = curve_fit(
            model_func, lag_h, lag_sv,
            p0=[c0_init, c_init, a_init],
            bounds=([0, 0, 1e-12], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
    except (RuntimeError, ValueError):
        popt = np.array([c0_init, c_init, a_init])

    return popt


# ---------------------------------------------------------------------------
# Kriging matrix
# ---------------------------------------------------------------------------

def _build_kriging_matrix(x, y, vario_func):
    """Build the (N+1)x(N+1) ordinary-kriging matrix and return K_inv."""
    n = len(x)
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    D = np.sqrt(dx ** 2 + dy ** 2)

    K = np.zeros((n + 1, n + 1), dtype=np.float64)
    K[:n, :n] = vario_func(D)
    K[:n, n] = 1.0
    K[n, :n] = 1.0

    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        K[:n, :n] += 1e-10 * np.eye(n)
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            warnings.warn(
                "kriging(): kriging matrix is singular even after "
                "regularisation; predictions will be NaN."
            )
            return None

    return K_inv


# ---------------------------------------------------------------------------
# Numpy prediction
# ---------------------------------------------------------------------------

def _kriging_predict(x_pts, y_pts, z_pts, x_grid, y_grid,
                     vario_func, K_inv, return_variance):
    """Vectorised kriging prediction for a grid chunk."""
    n = len(x_pts)
    gx, gy = np.meshgrid(x_grid, y_grid)
    gx_flat = gx.ravel()
    gy_flat = gy.ravel()

    dx = gx_flat[:, np.newaxis] - x_pts[np.newaxis, :]
    dy = gy_flat[:, np.newaxis] - y_pts[np.newaxis, :]
    dists = np.sqrt(dx ** 2 + dy ** 2)

    k0 = np.empty((len(gx_flat), n + 1), dtype=np.float64)
    k0[:, :n] = vario_func(dists)
    k0[:, n] = 1.0

    # w = K_inv @ k0 for each pixel (K_inv is symmetric)
    w = k0 @ K_inv

    prediction = (w[:, :n] * z_pts[np.newaxis, :]).sum(axis=1)
    prediction = prediction.reshape(len(y_grid), len(x_grid))

    variance = None
    if return_variance:
        variance = np.sum(w * k0, axis=1)
        variance = variance.reshape(len(y_grid), len(x_grid))

    return prediction, variance


# ---------------------------------------------------------------------------
# Numpy backend wrapper
# ---------------------------------------------------------------------------

def _kriging_numpy(x_pts, y_pts, z_pts, x_grid, y_grid,
                   vario_func, K_inv, return_variance, template_data):
    return _kriging_predict(x_pts, y_pts, z_pts, x_grid, y_grid,
                            vario_func, K_inv, return_variance)


# ---------------------------------------------------------------------------
# Dask + numpy backend
# ---------------------------------------------------------------------------

def _kriging_dask_numpy(x_pts, y_pts, z_pts, x_grid, y_grid,
                        vario_func, K_inv, return_variance, template_data):

    def _chunk_pred(block, block_info=None):
        if block_info is None:
            return block
        loc = block_info[0]['array-location']
        y_sl = y_grid[loc[0][0]:loc[0][1]]
        x_sl = x_grid[loc[1][0]:loc[1][1]]
        pred, _ = _kriging_predict(x_pts, y_pts, z_pts, x_sl, y_sl,
                                   vario_func, K_inv, False)
        return pred

    prediction = da.map_blocks(_chunk_pred, template_data, dtype=np.float64)

    variance = None
    if return_variance:
        def _chunk_var(block, block_info=None):
            if block_info is None:
                return block
            loc = block_info[0]['array-location']
            y_sl = y_grid[loc[0][0]:loc[0][1]]
            x_sl = x_grid[loc[1][0]:loc[1][1]]
            _, var = _kriging_predict(x_pts, y_pts, z_pts, x_sl, y_sl,
                                     vario_func, K_inv, True)
            return var

        variance = da.map_blocks(_chunk_var, template_data, dtype=np.float64)

    return prediction, variance


# ---------------------------------------------------------------------------
# CuPy prediction
# ---------------------------------------------------------------------------

def _kriging_predict_cupy(x_pts, y_pts, z_pts, x_grid, y_grid,
                          vario_func, K_inv, return_variance):
    """Vectorised kriging prediction on GPU via CuPy."""
    n = len(x_pts)

    x_gpu = cupy.asarray(x_pts)
    y_gpu = cupy.asarray(y_pts)
    z_gpu = cupy.asarray(z_pts)
    xg_gpu = cupy.asarray(x_grid)
    yg_gpu = cupy.asarray(y_grid)
    K_inv_gpu = cupy.asarray(K_inv)

    gx, gy = cupy.meshgrid(xg_gpu, yg_gpu)
    gx_flat = gx.ravel()
    gy_flat = gy.ravel()

    dx = gx_flat[:, None] - x_gpu[None, :]
    dy = gy_flat[:, None] - y_gpu[None, :]
    dists = cupy.sqrt(dx ** 2 + dy ** 2)

    k0 = cupy.empty((len(gx_flat), n + 1), dtype=np.float64)
    k0[:, :n] = vario_func(dists)
    k0[:, n] = 1.0

    w = k0 @ K_inv_gpu

    prediction = (w[:, :n] * z_gpu[None, :]).sum(axis=1)
    prediction = prediction.reshape(len(y_grid), len(x_grid))

    variance = None
    if return_variance:
        variance = cupy.sum(w * k0, axis=1)
        variance = variance.reshape(len(y_grid), len(x_grid))

    return prediction, variance


# ---------------------------------------------------------------------------
# CuPy backend wrapper
# ---------------------------------------------------------------------------

def _kriging_cupy(x_pts, y_pts, z_pts, x_grid, y_grid,
                  vario_func, K_inv, return_variance, template_data):
    return _kriging_predict_cupy(x_pts, y_pts, z_pts, x_grid, y_grid,
                                 vario_func, K_inv, return_variance)


# ---------------------------------------------------------------------------
# Dask + CuPy backend
# ---------------------------------------------------------------------------

def _kriging_dask_cupy(x_pts, y_pts, z_pts, x_grid, y_grid,
                       vario_func, K_inv, return_variance, template_data):

    def _chunk_pred(block, block_info=None):
        if block_info is None:
            return block
        loc = block_info[0]['array-location']
        y_sl = y_grid[loc[0][0]:loc[0][1]]
        x_sl = x_grid[loc[1][0]:loc[1][1]]
        pred, _ = _kriging_predict_cupy(x_pts, y_pts, z_pts, x_sl, y_sl,
                                        vario_func, K_inv, False)
        return pred

    prediction = da.map_blocks(
        _chunk_pred, template_data, dtype=np.float64,
        meta=cupy.array((), dtype=np.float64),
    )

    variance = None
    if return_variance:
        def _chunk_var(block, block_info=None):
            if block_info is None:
                return block
            loc = block_info[0]['array-location']
            y_sl = y_grid[loc[0][0]:loc[0][1]]
            x_sl = x_grid[loc[1][0]:loc[1][1]]
            _, var = _kriging_predict_cupy(x_pts, y_pts, z_pts, x_sl, y_sl,
                                          vario_func, K_inv, True)
            return var

        variance = da.map_blocks(
            _chunk_var, template_data, dtype=np.float64,
            meta=cupy.array((), dtype=np.float64),
        )

    return prediction, variance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def kriging(x, y, z, template, variogram_model='spherical', nlags=15,
            return_variance=False, name='kriging'):
    """Ordinary Kriging interpolation.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates and values of scattered input points.
    template : xr.DataArray
        2-D DataArray whose grid defines the output raster.
    variogram_model : str, default 'spherical'
        Variogram model: ``'spherical'``, ``'exponential'``, or
        ``'gaussian'``.
    nlags : int, default 15
        Number of lag bins for the experimental variogram.
    return_variance : bool, default False
        If True, return ``(prediction, variance)`` tuple.
    name : str, default 'kriging'
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray or tuple of xr.DataArray
        Prediction raster, or ``(prediction, variance)`` if
        *return_variance* is True.
    """
    _validate_raster(template, func_name='kriging', name='template')
    x_arr, y_arr, z_arr = validate_points(x, y, z, func_name='kriging')

    if variogram_model not in _VARIOGRAM_MODELS:
        raise ValueError(
            f"kriging(): variogram_model must be one of "
            f"{sorted(_VARIOGRAM_MODELS)}, got {variogram_model!r}"
        )

    _validate_scalar(nlags, func_name='kriging', name='nlags',
                     dtype=int, min_val=1)

    # Experimental variogram
    lag_h, lag_sv = _experimental_variogram(x_arr, y_arr, z_arr, nlags)

    if len(lag_h) < 3:
        warnings.warn(
            "kriging(): fewer than 3 non-empty lag bins; variogram fit "
            "may be unreliable. Consider using more points or fewer lags."
        )
        if len(lag_h) == 0:
            lag_h = np.array([1.0])
            lag_sv = np.array([1.0])

    # Fit variogram model
    model_func = _VARIOGRAM_MODELS[variogram_model]
    params = _fit_variogram(lag_h, lag_sv, model_func)

    def vario_func(h):
        return model_func(h, *params)

    # Build kriging matrix
    K_inv = _build_kriging_matrix(x_arr, y_arr, vario_func)

    x_grid, y_grid = extract_grid_coords(template, func_name='kriging')

    if K_inv is None:
        pred = np.full(template.shape, np.nan)
        prediction = xr.DataArray(
            pred, name=name,
            coords=template.coords,
            dims=template.dims,
            attrs=template.attrs,
        )
        if return_variance:
            return prediction, prediction.copy()
        return prediction

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_kriging_numpy,
        cupy_func=_kriging_cupy,
        dask_func=_kriging_dask_numpy,
        dask_cupy_func=_kriging_dask_cupy,
    )

    pred_arr, var_arr = mapper(template)(
        x_arr, y_arr, z_arr, x_grid, y_grid,
        vario_func, K_inv, return_variance, template.data,
    )

    prediction = xr.DataArray(
        pred_arr, name=name,
        coords=template.coords,
        dims=template.dims,
        attrs=template.attrs,
    )

    if return_variance:
        variance = xr.DataArray(
            var_arr, name=f'{name}_variance',
            coords=template.coords,
            dims=template.dims,
            attrs=template.attrs,
        )
        return prediction, variance

    return prediction
