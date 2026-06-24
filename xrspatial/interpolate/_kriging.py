"""Ordinary Kriging interpolation."""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from xrspatial.utils import (ArrayTypeFunctionMapping, _dask_task_name_kwargs, _validate_raster,
                             _validate_scalar)

from ._validation import extract_grid_coords, validate_points
from ._vector import resolve_xyz

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

    if dists.size == 0:
        # Fewer than two points: no pairs, so no spatial structure to bin.
        return np.array([]), np.array([])

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
    G = vario_func(D)
    # The semivariogram has gamma(0) = 0 by definition; vario_func(0)
    # returns the nugget c0, which is the one-sided limit as h -> 0+,
    # not the value at h = 0.  Force the diagonal to 0 so a non-zero
    # nugget is not placed on the matrix diagonal (which would force
    # exact interpolation and bias the kriging variance downward).
    np.fill_diagonal(G, 0.0)
    K[:n, :n] = G
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

def _dual_z_aug(xp, z_pts):
    """Return ``[z, 0]`` for the dual kriging form on array module *xp*."""
    z_aug = xp.zeros(len(z_pts) + 1, dtype=np.float64)
    z_aug[:len(z_pts)] = z_pts
    return z_aug


def _kriging_predict(x_pts, y_pts, z_pts, x_grid, y_grid,
                     vario_func, K_inv, return_variance, dual_w=None):
    """Vectorised kriging prediction for a grid chunk.

    ``dual_w`` is an optional precomputed dual weight vector
    ``K_inv @ [z, 0]`` used by the no-variance path; the dask backend
    passes it in so the O(N^2) matvec runs once instead of per chunk.
    """
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

    variance = None
    if return_variance:
        # w = K_inv @ k0 for each pixel (K_inv is symmetric)
        w = k0 @ K_inv
        prediction = (w[:, :n] * z_pts[np.newaxis, :]).sum(axis=1)
        variance = np.sum(w * k0, axis=1)
        variance = variance.reshape(len(y_grid), len(x_grid))
    else:
        # Prediction only: the weights are needed solely to dot against
        # z, so fold K_inv into z up front (dual kriging form).
        # k0 @ (K_inv @ z_aug) equals (k0 @ K_inv)[:, :n] @ z because
        # z_aug[n] == 0, and it replaces the
        # (grid_pixels, N+1) x (N+1, N+1) matmul and the
        # (grid_pixels, N+1) `w` temporary with a single matvec.
        if dual_w is None:
            dual_w = K_inv @ _dual_z_aug(np, z_pts)
        prediction = k0 @ dual_w
    prediction = prediction.reshape(len(y_grid), len(x_grid))

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

    if not return_variance:
        # The dual weight vector is chunk-invariant; compute it once
        # here rather than once per chunk.
        dual_w = K_inv @ _dual_z_aug(np, z_pts)

        def _chunk_pred(block, block_info=None):
            if block_info is None:
                return block
            loc = block_info[0]['array-location']
            y_sl = y_grid[loc[0][0]:loc[0][1]]
            x_sl = x_grid[loc[1][0]:loc[1][1]]
            pred, _ = _kriging_predict(x_pts, y_pts, z_pts, x_sl, y_sl,
                                       vario_func, K_inv, False,
                                       dual_w=dual_w)
            return pred

        prediction = da.map_blocks(_chunk_pred, template_data,
                                   dtype=np.float64,
                                   **_dask_task_name_kwargs('xrspatial.kriging'))
        return prediction, None

    # Prediction and variance fall out of the same per-chunk pipeline
    # (dists, k0, weights), so compute both in one map_blocks pass and
    # slice the two channels.  Two separate passes would run that
    # pipeline twice per chunk.
    def _chunk_both(block, block_info=None):
        if block_info is None:
            return np.stack([block, block])
        loc = block_info[0]['array-location']
        y_sl = y_grid[loc[0][0]:loc[0][1]]
        x_sl = x_grid[loc[1][0]:loc[1][1]]
        pred, var = _kriging_predict(x_pts, y_pts, z_pts, x_sl, y_sl,
                                     vario_func, K_inv, True)
        return np.stack([pred, var])

    stacked = da.map_blocks(
        _chunk_both, template_data, dtype=np.float64,
        new_axis=0, chunks=((2,),) + template_data.chunks,
        **_dask_task_name_kwargs('xrspatial.kriging_variance'),
    )
    return stacked[0], stacked[1]


# ---------------------------------------------------------------------------
# CuPy prediction
# ---------------------------------------------------------------------------

def _kriging_predict_cupy(x_pts, y_pts, z_pts, x_grid, y_grid,
                          vario_func, K_inv, return_variance,
                          dual_w=None):
    """Vectorised kriging prediction on GPU via CuPy.

    The point arrays and ``K_inv`` may arrive either on the host or
    already on the device; ``cupy.asarray`` is a no-op for the latter.
    ``dual_w`` is the optional precomputed device-resident dual weight
    vector for the no-variance path (see ``_kriging_predict``).
    """
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

    variance = None
    if return_variance:
        w = k0 @ K_inv_gpu
        prediction = (w[:, :n] * z_gpu[None, :]).sum(axis=1)
        variance = cupy.sum(w * k0, axis=1)
        variance = variance.reshape(len(y_grid), len(x_grid))
    else:
        # Dual kriging form; see _kriging_predict for the derivation.
        if dual_w is None:
            dual_w = K_inv_gpu @ _dual_z_aug(cupy, z_gpu)
        prediction = k0 @ dual_w
    prediction = prediction.reshape(len(y_grid), len(x_grid))

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

    # The point arrays and K_inv are the same for every chunk, so
    # upload them to the device once instead of once per chunk.
    # _kriging_predict_cupy passes them through cupy.asarray, which is
    # a no-op for device-resident arrays.  K_inv is (N+1) x (N+1), so
    # re-uploading it per chunk is the expensive part for large N.
    # Under the threaded/synchronous scheduler the per-chunk closure
    # shares these device buffers by reference; a distributed scheduler
    # would re-serialise them per task, which is no worse than the
    # previous per-chunk upload.
    x_gpu = cupy.asarray(x_pts)
    y_gpu = cupy.asarray(y_pts)
    z_gpu = cupy.asarray(z_pts)
    K_inv_gpu = cupy.asarray(K_inv)

    if not return_variance:
        # The dual weight vector is chunk-invariant; compute it once
        # here rather than once per chunk.
        dual_w = K_inv_gpu @ _dual_z_aug(cupy, z_gpu)

        def _chunk_pred(block, block_info=None):
            if block_info is None:
                return block
            loc = block_info[0]['array-location']
            y_sl = y_grid[loc[0][0]:loc[0][1]]
            x_sl = x_grid[loc[1][0]:loc[1][1]]
            pred, _ = _kriging_predict_cupy(x_gpu, y_gpu, z_gpu, x_sl, y_sl,
                                            vario_func, K_inv_gpu, False,
                                            dual_w=dual_w)
            return pred

        prediction = da.map_blocks(
            _chunk_pred, template_data, dtype=np.float64,
            meta=cupy.array((), dtype=np.float64),
            **_dask_task_name_kwargs('xrspatial.kriging'),
        )
        return prediction, None

    # Single pass for prediction and variance; see _kriging_dask_numpy.
    def _chunk_both(block, block_info=None):
        if block_info is None:
            return cupy.stack([block, block])
        loc = block_info[0]['array-location']
        y_sl = y_grid[loc[0][0]:loc[0][1]]
        x_sl = x_grid[loc[1][0]:loc[1][1]]
        pred, var = _kriging_predict_cupy(x_gpu, y_gpu, z_gpu, x_sl, y_sl,
                                          vario_func, K_inv_gpu, True)
        return cupy.stack([pred, var])

    stacked = da.map_blocks(
        _chunk_both, template_data, dtype=np.float64,
        new_axis=0, chunks=((2,),) + template_data.chunks,
        meta=cupy.array((), dtype=np.float64),
        **_dask_task_name_kwargs('xrspatial.kriging_variance'),
    )
    return stacked[0], stacked[1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _check_kriging_memory(n_points, grid_pixels, is_dask=False):
    """Raise MemoryError if kriging() would exceed available memory.

    Three allocations dominate kriging memory use:

    * Experimental variogram pair arrays.  ``np.triu_indices`` produces
      two int64 index arrays of length ``N*(N-1)/2``, plus float64
      ``dists`` and ``semivar`` of the same length.  Roughly
      ``4 * N*(N-1)/2 * 8`` bytes.
    * Kriging matrix.  ``K`` and ``K_inv`` are both ``(N+1, N+1)``
      float64, plus an N x N intermediate distance matrix.  About
      ``3 * (N+1)**2 * 8`` bytes.
    * Prediction ``k0`` matrix of shape ``(grid_pixels, N+1)`` float64,
      plus matching ``dists`` and ``w`` of similar size.  About
      ``3 * grid_pixels * (N+1) * 8`` bytes.

    Worst case is the maximum of these three.  The variogram and matrix
    builds run sequentially, and ``k0`` is built later, so peak usage
    is bounded by the largest single allocation.

    When ``is_dask`` is True the prediction ``k0`` matrix is built one
    chunk at a time by ``map_blocks``, so its peak size scales with the
    chunk rather than ``grid_pixels``.  ``grid_pixels`` is not a valid
    bound for that path, so the ``k0`` term is dropped.  The variogram
    and matrix terms are point-based and materialised on the host
    regardless of backend, so they still apply.
    """
    n = int(n_points)
    g = int(grid_pixels)

    pair_bytes = 4 * (n * (n - 1) // 2) * 8 if n > 1 else 0
    matrix_bytes = 3 * (n + 1) * (n + 1) * 8
    k0_bytes = 0 if is_dask else 3 * g * (n + 1) * 8

    estimate = max(pair_bytes, matrix_bytes, k0_bytes)

    try:
        from xrspatial.zonal import _available_memory_bytes
        avail = _available_memory_bytes()
    except ImportError:
        avail = 2 * 1024 ** 3

    if estimate > 0.8 * avail:
        if estimate == k0_bytes:
            culprit = (
                f"prediction matrix of shape ({g}, {n + 1}) "
                f"(grid_pixels x N+1)"
            )
            advice = (
                "Reduce the template grid size or the number of input "
                "points, or use a chunked dask-backed template."
            )
        elif estimate == matrix_bytes:
            culprit = f"kriging matrix of shape ({n + 1}, {n + 1})"
            advice = "Reduce the number of input points."
        else:
            culprit = (
                f"variogram pair arrays of length {n * (n - 1) // 2} "
                f"(N*(N-1)/2 for N={n})"
            )
            advice = "Reduce the number of input points."

        raise MemoryError(
            f"kriging() needs ~{estimate / 1e9:.1f} GB to allocate the "
            f"{culprit}, but only ~{avail / 1e9:.1f} GB is available. "
            f"{advice}"
        )


def kriging(x, y=None, z=None, template=None, variogram_model='spherical',
            nlags=15, return_variance=False, name='kriging', *, column=None):
    """Ordinary Kriging interpolation.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates and values of scattered input points.  Alternatively,
        pass a GeoDataFrame of Point geometries as the first argument and
        leave *y*/*z* unset; *template* and *column* must then be keywords.
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
    column : str, optional
        When the first argument is a GeoDataFrame, the column whose values
        are interpolated.  Defaults to the first numeric column.

    Returns
    -------
    xr.DataArray or tuple of xr.DataArray
        Prediction raster, or ``(prediction, variance)`` if
        *return_variance* is True.

    Raises
    ------
    MemoryError
        If the worst-case allocation (variogram pair arrays, kriging
        matrix, or prediction matrix) would exceed 80% of available
        memory.
    """
    x, y, z = resolve_xyz(x, y, z, column=column, func_name='kriging')
    _validate_raster(template, func_name='kriging', name='template')
    x_arr, y_arr, z_arr = validate_points(x, y, z, func_name='kriging')

    if variogram_model not in _VARIOGRAM_MODELS:
        raise ValueError(
            f"kriging(): variogram_model must be one of "
            f"{sorted(_VARIOGRAM_MODELS)}, got {variogram_model!r}"
        )

    _validate_scalar(nlags, func_name='kriging', name='nlags',
                     dtype=int, min_val=1)

    # Memory guard.  Runs after input validation so we know N and the
    # template grid size, but before any large allocation.
    grid_pixels = int(np.prod(template.shape))
    is_dask = da is not None and isinstance(template.data, da.Array)
    _check_kriging_memory(len(x_arr), grid_pixels, is_dask=is_dask)

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
        # Build the all-NaN fallback on the same backend as the template:
        # lazy per-chunk for dask templates, and np.full_like dispatches to
        # cupy via __array_function__ for cupy-backed blocks.
        if da is not None and isinstance(template.data, da.Array):
            pred = template.data.map_blocks(
                lambda block: np.full_like(block, np.nan, dtype=np.float64),
                dtype=np.float64,
            )
        else:
            pred = np.full_like(template.data, np.nan, dtype=np.float64)
        prediction = xr.DataArray(
            pred, name=name,
            coords=template.coords,
            dims=template.dims,
            attrs=template.attrs,
        )
        if return_variance:
            variance = prediction.copy()
            variance.name = f'{name}_variance'
            return prediction, variance
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
