"""Thin Plate Spline (TPS) interpolation."""

from __future__ import annotations

import math

import numpy as np
import xarray as xr
from numba import cuda

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _validate_raster,
    _validate_scalar,
    cuda_args,
    ngjit,
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


# ---------------------------------------------------------------------------
# TPS system build (always CPU)
# ---------------------------------------------------------------------------

def _tps_radial(r2):
    """U(r) = r^2 ln(r) = 0.5 * r^2 * ln(r^2) for r > 0, else 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(r2 > 0, 0.5 * r2 * np.log(r2), 0.0)
    return result


def _tps_build_and_solve(x_pts, y_pts, z_pts, smoothing):
    """Build and solve the (N+3)x(N+3) TPS system.

    Returns the weight vector of length N+3:
        [w_0, ..., w_{N-1}, a_0, a_1, a_2]
    """
    n = len(x_pts)

    # With fewer than 3 points the full affine+radial system is
    # underdetermined.  Fall back to a simple constant/affine fit.
    if n < 3:
        weights = np.zeros(n + 3, dtype=np.float64)
        if n == 1:
            weights[n] = z_pts[0]  # constant surface
        else:
            # n == 2: fit z = a0 + a1*x + a2*y via least-squares
            P = np.ones((n, 3), dtype=np.float64)
            P[:, 1] = x_pts
            P[:, 2] = y_pts
            a, _, _, _ = np.linalg.lstsq(P, z_pts, rcond=None)
            weights[n:] = a
        return weights

    # K block: K[i,j] = U(||p_i - p_j||)
    dx = x_pts[:, np.newaxis] - x_pts[np.newaxis, :]
    dy = y_pts[:, np.newaxis] - y_pts[np.newaxis, :]
    r2 = dx ** 2 + dy ** 2
    K = _tps_radial(r2)

    # Regularisation
    K += (smoothing + 1e-10) * np.eye(n)

    # P block: [1, x_i, y_i]
    P = np.ones((n, 3), dtype=np.float64)
    P[:, 1] = x_pts
    P[:, 2] = y_pts

    # Assemble full system
    A = np.zeros((n + 3, n + 3), dtype=np.float64)
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T

    b = np.zeros(n + 3, dtype=np.float64)
    b[:n] = z_pts

    try:
        weights = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Degenerate geometry (e.g. collinear points)
        weights, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return weights


# ---------------------------------------------------------------------------
# CPU evaluation kernel (numba JIT)
# ---------------------------------------------------------------------------

@ngjit
def _tps_evaluate_cpu(x_pts, y_pts, weights, n_pts, x_grid, y_grid):
    ny = y_grid.shape[0]
    nx = x_grid.shape[0]
    out = np.empty((ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            gx = x_grid[j]
            gy = y_grid[i]
            val = (weights[n_pts]
                   + weights[n_pts + 1] * gx
                   + weights[n_pts + 2] * gy)

            for p in range(n_pts):
                dx = gx - x_pts[p]
                dy = gy - y_pts[p]
                r2 = dx * dx + dy * dy
                if r2 > 0.0:
                    val += weights[p] * r2 * math.log(r2) * 0.5

            out[i, j] = val

    return out


# ---------------------------------------------------------------------------
# Numpy backend
# ---------------------------------------------------------------------------

def _spline_numpy(x_pts, y_pts, z_pts, x_grid, y_grid,
                  smoothing, weights, template_data):
    n = len(x_pts)
    return _tps_evaluate_cpu(x_pts, y_pts, weights, n, x_grid, y_grid)


# ---------------------------------------------------------------------------
# CUDA evaluation kernel
# ---------------------------------------------------------------------------

@cuda.jit
def _tps_cuda_kernel(x_pts, y_pts, weights, n_pts, x_grid, y_grid, out):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        gx = x_grid[j]
        gy = y_grid[i]
        val = (weights[n_pts]
               + weights[n_pts + 1] * gx
               + weights[n_pts + 2] * gy)

        for p in range(n_pts):
            dx = gx - x_pts[p]
            dy = gy - y_pts[p]
            r2 = dx * dx + dy * dy
            if r2 > 0.0:
                val += weights[p] * r2 * math.log(r2) * 0.5

        out[i, j] = val


# ---------------------------------------------------------------------------
# CuPy backend (CPU solve + GPU evaluate)
# ---------------------------------------------------------------------------

def _spline_cupy(x_pts, y_pts, z_pts, x_grid, y_grid,
                 smoothing, weights, template_data):
    n = len(x_pts)
    x_gpu = cupy.asarray(x_pts)
    y_gpu = cupy.asarray(y_pts)
    w_gpu = cupy.asarray(weights)
    xg_gpu = cupy.asarray(x_grid)
    yg_gpu = cupy.asarray(y_grid)

    ny, nx = len(y_grid), len(x_grid)
    out = cupy.empty((ny, nx), dtype=np.float64)

    griddim, blockdim = cuda_args((ny, nx))
    _tps_cuda_kernel[griddim, blockdim](
        x_gpu, y_gpu, w_gpu, n, xg_gpu, yg_gpu, out,
    )
    return out


# ---------------------------------------------------------------------------
# Dask + numpy backend
# ---------------------------------------------------------------------------

def _spline_dask_numpy(x_pts, y_pts, z_pts, x_grid, y_grid,
                       smoothing, weights, template_data):

    def _chunk(block, block_info=None):
        if block_info is None:
            return block
        loc = block_info[0]['array-location']
        y_sl = y_grid[loc[0][0]:loc[0][1]]
        x_sl = x_grid[loc[1][0]:loc[1][1]]
        return _spline_numpy(x_pts, y_pts, z_pts, x_sl, y_sl,
                             smoothing, weights, None)

    return da.map_blocks(_chunk, template_data, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dask + cupy backend
# ---------------------------------------------------------------------------

def _spline_dask_cupy(x_pts, y_pts, z_pts, x_grid, y_grid,
                      smoothing, weights, template_data):

    def _chunk(block, block_info=None):
        if block_info is None:
            return block
        loc = block_info[0]['array-location']
        y_sl = y_grid[loc[0][0]:loc[0][1]]
        x_sl = x_grid[loc[1][0]:loc[1][1]]
        return _spline_cupy(x_pts, y_pts, z_pts, x_sl, y_sl,
                            smoothing, weights, None)

    return da.map_blocks(
        _chunk, template_data, dtype=np.float64,
        meta=cupy.array((), dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _check_spline_memory(n_points, grid_pixels):
    """Raise MemoryError if spline() would exceed available memory.

    Two allocations dominate TPS memory use:

    * Kernel build.  Lines 67-70 construct ``dx``, ``dy``, ``r2`` and
      ``K`` each as N x N float64.  About ``4 * N**2 * 8`` bytes during
      construction.
    * Augmented system.  Line 81 builds ``A`` of shape ``(N+3, N+3)``
      float64, and ``np.linalg.solve`` copies it during LU
      factorization.  About ``2 * (N+3)**2 * 8`` bytes peak.

    The kernel build runs first and is freed before the solve, so peak
    usage is bounded by the larger of the two.  The grid is iterated
    point-by-point inside a numba kernel, so prediction does not
    materialize a grid x N matrix.
    """
    n = int(n_points)

    if n < 3:
        return  # short-circuit path, no big allocations

    kernel_bytes = 4 * n * n * 8
    solve_bytes = 2 * (n + 3) * (n + 3) * 8

    estimate = max(kernel_bytes, solve_bytes)

    try:
        from xrspatial.zonal import _available_memory_bytes
        avail = _available_memory_bytes()
    except ImportError:
        avail = 2 * 1024 ** 3

    # Match the kriging guard's 0.8 fraction (PR #1309) so users get
    # consistent behaviour across interpolators.  The TPS solve and the
    # kernel build do not run concurrently with other large arrays in
    # this function, so 80% of available RAM is the right ceiling.
    if estimate > 0.8 * avail:
        if estimate == solve_bytes:
            culprit = (
                f"augmented system A of shape ({n + 3}, {n + 3}) "
                f"plus its LU factorization copy"
            )
        else:
            culprit = (
                f"radial-basis kernel block K of shape ({n}, {n}) "
                f"(plus dx, dy, r2 of the same shape)"
            )

        raise MemoryError(
            f"spline() needs ~{estimate / 1e9:.1f} GB to allocate the "
            f"{culprit}, but only ~{avail / 1e9:.1f} GB is available. "
            f"Reduce the number of input points."
        )


def spline(x, y, z, template, smoothing=0.0, name='spline'):
    """Thin Plate Spline interpolation.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates and values of scattered input points.
    template : xr.DataArray
        2-D DataArray whose grid defines the output raster.
    smoothing : float, default 0.0
        Smoothing parameter.  0 forces exact interpolation through
        the data points.
    name : str, default 'spline'
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray

    Raises
    ------
    MemoryError
        If the worst-case allocation (kernel block K or augmented
        system A) would exceed 80% of available memory.
    """
    _validate_raster(template, func_name='spline', name='template')
    x_arr, y_arr, z_arr = validate_points(x, y, z, func_name='spline')
    _validate_scalar(smoothing, func_name='spline', name='smoothing',
                     min_val=0.0)

    x_grid, y_grid = extract_grid_coords(template, func_name='spline')

    # Memory guard.  Runs after input validation so we know N, but
    # before the TPS system is built.
    grid_pixels = int(np.prod(template.shape))
    _check_spline_memory(len(x_arr), grid_pixels)

    # Solve TPS system once on CPU
    weights = _tps_build_and_solve(x_arr, y_arr, z_arr, smoothing)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_spline_numpy,
        cupy_func=_spline_cupy,
        dask_func=_spline_dask_numpy,
        dask_cupy_func=_spline_dask_cupy,
    )

    out = mapper(template)(
        x_arr, y_arr, z_arr, x_grid, y_grid,
        smoothing, weights, template.data,
    )

    return xr.DataArray(
        out, name=name,
        coords=template.coords,
        dims=template.dims,
        attrs=template.attrs,
    )
