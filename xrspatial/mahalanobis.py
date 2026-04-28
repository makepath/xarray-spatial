"""Mahalanobis distance for multi-band rasters.

Computes the Mahalanobis distance at each pixel from a reference
distribution, accounting for correlations between bands.

    D(x) = sqrt((x - mu)^T * Sigma^-1 * (x - mu))

Supports numpy, cupy, dask+numpy, and dask+cupy backends.
"""

from typing import List, Optional

import numpy as np
import xarray as xr

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    has_cuda_and_cupy,
    has_dask_array,
    is_cupy_array,
    is_dask_cupy,
    validate_arrays,
)

try:
    import dask
    import dask.array as da
except ImportError:
    da = None
    dask = None

try:
    import cupy
except ImportError:
    cupy = None


# ---------------------------------------------------------------------------
# Memory guards
# ---------------------------------------------------------------------------
#
# Both the statistics phase and the per-pixel phase materialise float64
# working buffers of shape (n_bands, H*W).  A conservative count of the
# live copies for each eager backend:
#
#   stack          (float64) : 8 bytes/cell * n_bands
#   reshape/.T     (float64) : 8 bytes/cell * n_bands  (transpose forces a
#                                                       contiguous copy when
#                                                       passed to BLAS)
#   centered/diff  (float64) : 8 bytes/cell * n_bands
#   diff @ inv_cov (float64) : 8 bytes/cell * n_bands
#
# Total ~32 bytes/cell * n_bands.  The (n_bands, n_bands) inverse covariance
# is negligible by comparison.
_BYTES_PER_CELL_PER_BAND = 32


def _available_memory_bytes():
    """Best-effort estimate of available memory in bytes."""
    # Try /proc/meminfo (Linux)
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024  # kB -> bytes
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

    Returns 0 if CuPy / CUDA is unavailable or the query fails -- callers
    use that as a sentinel meaning "no GPU info, skip the guard".
    """
    try:
        import cupy as _cp
        free, _total = _cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return 0


def _projected_bytes(n_bands, height, width):
    return int(n_bands) * int(height) * int(width) * _BYTES_PER_CELL_PER_BAND


def _check_memory(n_bands, height, width):
    """Raise MemoryError if the host working set would exceed 50% of RAM."""
    required = _projected_bytes(n_bands, height, width)
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"mahalanobis on {n_bands} bands of shape "
            f"({height}, {width}) needs ~{required / 1e9:.1f} GB of "
            f"working memory but only ~{available / 1e9:.1f} GB is "
            f"available.  Use a dask-backed DataArray for out-of-core "
            f"processing, or pass smaller inputs."
        )


def _check_gpu_memory(n_bands, height, width):
    """Raise MemoryError if the GPU working set would exceed 50% of free VRAM.

    Returns silently when ``_available_gpu_memory_bytes`` cannot determine
    the free memory -- e.g. on hosts without CUDA, where the kernel will
    fail at the cupy boundary anyway.
    """
    available = _available_gpu_memory_bytes()
    if available <= 0:
        return
    required = _projected_bytes(n_bands, height, width)
    if required > 0.5 * available:
        raise MemoryError(
            f"mahalanobis on {n_bands} bands of shape "
            f"({height}, {width}) needs ~{required / 1e9:.1f} GB of "
            f"GPU working memory but only ~{available / 1e9:.1f} GB is "
            f"free on the active device.  Use a dask+cupy DataArray for "
            f"out-of-core processing, or pass smaller inputs."
        )


# ---------------------------------------------------------------------------
# Phase 1: compute statistics (mean vector, inverse covariance)
# ---------------------------------------------------------------------------

def _compute_stats_numpy(bands_data):
    """Compute (mu, inv_cov) from a list of numpy 2D arrays."""
    n_bands = len(bands_data)
    stacked = np.stack([b.astype(np.float64) for b in bands_data], axis=0)
    flat = stacked.reshape(n_bands, -1)

    # only pixels where ALL bands are finite
    valid = np.all(np.isfinite(flat), axis=0)
    n_valid = int(np.sum(valid))
    if n_valid < n_bands + 1:
        raise ValueError(
            f"Not enough valid pixels ({n_valid}) to compute statistics "
            f"for {n_bands} bands. Need at least {n_bands + 1}."
        )

    flat_valid = flat[:, valid]  # (N, n_valid)
    mu = np.mean(flat_valid, axis=1)  # (N,)

    centered = flat_valid - mu[:, np.newaxis]
    cov = (centered @ centered.T) / (n_valid - 1)  # (N, N)

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Covariance matrix is singular. Bands may be linearly "
            "dependent or have zero variance."
        )

    return mu, inv_cov


def _compute_stats_cupy(bands_data):
    """Compute (mu, inv_cov) from a list of cupy 2D arrays.

    Returns numpy arrays so the small statistics live on host.
    """
    n_bands = len(bands_data)
    stacked = cupy.stack(
        [b.astype(cupy.float64) for b in bands_data], axis=0
    )
    flat = stacked.reshape(n_bands, -1)

    valid = cupy.all(cupy.isfinite(flat), axis=0)
    n_valid = int(cupy.sum(valid).get())
    if n_valid < n_bands + 1:
        raise ValueError(
            f"Not enough valid pixels ({n_valid}) to compute statistics "
            f"for {n_bands} bands. Need at least {n_bands + 1}."
        )

    flat_valid = flat[:, valid]
    mu = cupy.mean(flat_valid, axis=1)

    centered = flat_valid - mu[:, cupy.newaxis]
    cov = (centered @ centered.T) / (n_valid - 1)

    try:
        inv_cov = cupy.linalg.inv(cov)
    except cupy.linalg.LinAlgError:
        raise ValueError(
            "Covariance matrix is singular. Bands may be linearly "
            "dependent or have zero variance."
        )

    return mu.get(), inv_cov.get()


def _compute_stats_dask(bands_data):
    """Compute (mu, inv_cov) from a list of dask 2D arrays.

    Uses lazy reductions materialized in a single ``dask.compute()`` call.
    Returns numpy arrays.
    """
    n_bands = len(bands_data)
    stacked = da.stack(
        [b.astype(np.float64) for b in bands_data], axis=0
    ).rechunk({0: n_bands})
    flat = stacked.reshape(n_bands, -1)

    valid = da.all(da.isfinite(flat), axis=0)
    n_valid_lazy = da.sum(valid)

    flat_masked = da.where(valid[np.newaxis, :], flat, 0.0)
    mu_sum = da.sum(flat_masked, axis=1)

    # materialize in one pass
    n_valid, mu_sum_val = dask.compute(n_valid_lazy, mu_sum)
    n_valid = int(n_valid)

    if n_valid < n_bands + 1:
        raise ValueError(
            f"Not enough valid pixels ({n_valid}) to compute statistics "
            f"for {n_bands} bands. Need at least {n_bands + 1}."
        )

    mu = mu_sum_val / n_valid  # small numpy array

    # center and compute covariance lazily
    centered = da.where(
        valid[np.newaxis, :],
        flat - mu[:, np.newaxis],
        0.0,
    )
    cov_lazy = (centered @ centered.T) / (n_valid - 1)

    cov = cov_lazy.compute()

    # bring to numpy if cupy-backed
    if is_cupy_array(cov):
        mu = cupy.asnumpy(mu)
        cov = cupy.asnumpy(cov)

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Covariance matrix is singular. Bands may be linearly "
            "dependent or have zero variance."
        )

    return mu.astype(np.float64), inv_cov.astype(np.float64)


# ---------------------------------------------------------------------------
# Phase 2: per-pixel distance
# ---------------------------------------------------------------------------

def _mahalanobis_pixel_numpy(stacked, mu, inv_cov):
    """Per-pixel Mahalanobis distance for a (N, H, W) numpy array.

    Parameters
    ----------
    stacked : numpy array, shape (N, H, W)
    mu : numpy array, shape (N,)
    inv_cov : numpy array, shape (N, N)

    Returns
    -------
    numpy array, shape (H, W), float64
    """
    n_bands, h, w = stacked.shape
    flat = stacked.reshape(n_bands, -1).T  # (pixels, N)

    # NaN mask: any band NaN → output NaN
    nan_mask = ~np.all(np.isfinite(flat), axis=1)  # (pixels,)

    diff = flat - mu[np.newaxis, :]  # (pixels, N)
    transformed = diff @ inv_cov  # (pixels, N)
    dist_sq = np.sum(transformed * diff, axis=1)  # (pixels,)
    dist_sq = np.maximum(dist_sq, 0.0)  # numerical guard
    result = np.sqrt(dist_sq)
    result[nan_mask] = np.nan

    return result.reshape(h, w)


def _mahalanobis_pixel_cupy(stacked, mu, inv_cov):
    """Per-pixel Mahalanobis distance for a (N, H, W) cupy array."""
    mu_gpu = cupy.asarray(mu)
    inv_cov_gpu = cupy.asarray(inv_cov)

    n_bands, h, w = stacked.shape
    flat = stacked.reshape(n_bands, -1).T  # (pixels, N)

    nan_mask = ~cupy.all(cupy.isfinite(flat), axis=1)

    diff = flat - mu_gpu[cupy.newaxis, :]
    transformed = diff @ inv_cov_gpu
    dist_sq = cupy.sum(transformed * diff, axis=1)
    dist_sq = cupy.maximum(dist_sq, 0.0)
    result = cupy.sqrt(dist_sq)
    result[nan_mask] = cupy.nan

    return result.reshape(h, w)


# ---------------------------------------------------------------------------
# Per-backend entry points
# ---------------------------------------------------------------------------

def _mahalanobis_numpy(bands_data, mu, inv_cov):
    stacked = np.stack([b.astype(np.float64) for b in bands_data], axis=0)
    return _mahalanobis_pixel_numpy(stacked, mu, inv_cov)


def _mahalanobis_cupy(bands_data, mu, inv_cov):
    stacked = cupy.stack(
        [b.astype(cupy.float64) for b in bands_data], axis=0
    )
    return _mahalanobis_pixel_cupy(stacked, mu, inv_cov)


def _mahalanobis_dask_numpy(bands_data, mu, inv_cov):
    stacked = da.stack(
        [b.astype(np.float64) for b in bands_data], axis=0
    ).rechunk({0: len(bands_data)})

    def _block_fn(block):
        return _mahalanobis_pixel_numpy(block, mu, inv_cov)

    return da.map_blocks(
        _block_fn,
        stacked,
        drop_axis=0,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64),
    )


def _mahalanobis_dask_cupy(bands_data, mu, inv_cov):
    stacked = da.stack(
        [b.astype(np.float64) for b in bands_data], axis=0
    ).rechunk({0: len(bands_data)})

    def _block_fn(block):
        return _mahalanobis_pixel_cupy(block, mu, inv_cov)

    return da.map_blocks(
        _block_fn,
        stacked,
        drop_axis=0,
        dtype=np.float64,
        meta=cupy.array((), dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mahalanobis(
    bands: List[xr.DataArray],
    mean: Optional[np.ndarray] = None,
    inv_cov: Optional[np.ndarray] = None,
    name: str = 'mahalanobis',
) -> xr.DataArray:
    """Compute per-pixel Mahalanobis distance from a multi-band reference.

    Parameters
    ----------
    bands : list of xr.DataArray
        N 2-D rasters (same shape, dtype family, and chunk structure).
        Must contain at least 2 bands.
    mean : numpy array, shape (N,), optional
        Mean vector of the reference distribution.  Must be provided
        together with *inv_cov*, or both omitted (auto-computed).
    inv_cov : numpy array, shape (N, N), optional
        Inverse covariance matrix.  Must be provided together with
        *mean*, or both omitted (auto-computed).
    name : str
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
        2-D float64 raster with same coords/dims/attrs as ``bands[0]``.
    """
    # --- input validation ---
    if len(bands) < 2:
        raise ValueError("At least 2 bands are required.")

    validate_arrays(*bands)

    if (mean is None) != (inv_cov is None):
        raise ValueError(
            "Provide both `mean` and `inv_cov`, or neither."
        )

    n_bands = len(bands)
    ref = bands[0]
    bands_data = [b.data for b in bands]

    # --- memory guard (eager backends only) ---
    # Dask paths process bounded chunks, so the per-task working set is
    # capped by the user's chunk size rather than the full raster shape.
    height, width = ref.shape
    is_dask = has_dask_array() and isinstance(ref.data, da.Array)
    is_cupy = has_cuda_and_cupy() and is_cupy_array(ref.data)
    if not is_dask:
        if is_cupy:
            _check_gpu_memory(n_bands, height, width)
        else:
            _check_memory(n_bands, height, width)

    # --- cast to float64 ---
    # (handled inside each backend function)

    # --- compute or validate statistics ---
    if mean is not None:
        mu = np.asarray(mean, dtype=np.float64)
        icov = np.asarray(inv_cov, dtype=np.float64)
        if mu.shape != (n_bands,):
            raise ValueError(
                f"`mean` shape {mu.shape} does not match "
                f"number of bands ({n_bands},)."
            )
        if icov.shape != (n_bands, n_bands):
            raise ValueError(
                f"`inv_cov` shape {icov.shape} does not match "
                f"expected ({n_bands}, {n_bands})."
            )
    else:
        # auto-compute stats — dispatch by backend
        if is_dask:
            mu, icov = _compute_stats_dask(bands_data)
        elif is_cupy:
            mu, icov = _compute_stats_cupy(bands_data)
        else:
            mu, icov = _compute_stats_numpy(bands_data)

    # --- per-pixel distance — dispatch by backend ---
    mapper = ArrayTypeFunctionMapping(
        numpy_func=_mahalanobis_numpy,
        cupy_func=_mahalanobis_cupy,
        dask_func=_mahalanobis_dask_numpy,
        dask_cupy_func=_mahalanobis_dask_cupy,
    )

    out = mapper(ref)(bands_data, mu, icov)

    return xr.DataArray(
        out,
        name=name,
        dims=ref.dims,
        coords=ref.coords,
        attrs=ref.attrs,
    )
