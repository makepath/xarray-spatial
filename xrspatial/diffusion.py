"""Scalar diffusion solver for raster fields.

Solves the 2D diffusion (heat) equation using an explicit forward-Euler
finite-difference scheme with a 5-point Laplacian stencil::

    du/dt = alpha * laplacian(u)

Supports uniform or spatially varying diffusivity and all four backends
(numpy, cupy, dask+numpy, dask+cupy).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import xarray as xr
from xarray import DataArray

from numba import cuda, prange

try:
    import cupy
except ImportError:
    class cupy:
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _boundary_to_dask,
    _pad_array,
    _validate_boundary,
    _validate_raster,
    _validate_scalar,
    calc_cuda_dims,
    has_cuda_and_cupy,
    is_cupy_array,
    ngjit,
    not_implemented_func,
)


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------
#
# Per-step working set on the eager numpy / cupy paths:
#   * u           : rows*cols  float64 (8 B)
#   * out         : rows*cols  float64 (8 B)
#   * alpha_arr   : rows*cols  float64 (8 B)   (only when materialised)
#   * padded copy : rows*cols  float64 (8 B)   (rough upper bound for the
#                                              padded buffer, ignoring the
#                                              1-cell border)
# Total ~32 bytes per pixel.  For dask paths the equivalent footprint is
# per-chunk and is bounded by chunk size, so the public-API guard targets
# the eager paths only.
_BYTES_PER_PIXEL = 32

# Hard cap on the number of explicit Euler iterations a single diffuse()
# call will run.  100k steps on a modest 1024x1024 raster is already minutes
# of CPU work; anything beyond this is almost certainly a misuse.
_MAX_STEPS = 100_000


def _available_memory_bytes():
    """Best-effort estimate of available memory in bytes."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    return 2 * 1024 ** 3  # 2 GB fallback


def _available_gpu_memory_bytes():
    """Best-effort estimate of free GPU memory in bytes.

    Returns 0 when the query fails -- callers treat that as a sentinel
    meaning "no GPU info, skip the guard".
    """
    try:
        import cupy as _cp
        free, _total = _cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return 0


def _check_memory(rows, cols):
    """Raise MemoryError if the eager numpy diffuse buffers exceed 50% of RAM."""
    required = int(rows) * int(cols) * _BYTES_PER_PIXEL
    available = _available_memory_bytes()
    if required > 0.5 * available:
        raise MemoryError(
            f"diffuse() on a {rows}x{cols} raster needs "
            f"~{required / 1e9:.1f} GB of working memory, but only "
            f"~{available / 1e9:.1f} GB is available. "
            f"Use a dask-backed DataArray for out-of-core processing."
        )


def _check_gpu_memory(rows, cols):
    """Raise MemoryError if the CuPy diffuse buffers exceed 50% of free GPU RAM.

    Skipped (returns silently) when the free-memory query fails -- the
    kernel will then fail at the cupy.asarray boundary anyway.
    """
    available = _available_gpu_memory_bytes()
    if available <= 0:
        return
    required = int(rows) * int(cols) * _BYTES_PER_PIXEL
    if required > 0.5 * available:
        raise MemoryError(
            f"diffuse() on a {rows}x{cols} raster needs "
            f"~{required / 1e9:.1f} GB of GPU working memory, but only "
            f"~{available / 1e9:.1f} GB is free on the active device. "
            f"Use a dask+cupy DataArray for out-of-core processing."
        )


# ---- numpy backend ----

@ngjit
def _diffuse_step_numpy(u, alpha_arr, dt_over_dx2, rows, cols):
    """One forward-Euler step on a padded array.

    *u* has shape ``(rows+2, cols+2)`` (1-cell pad on each side).
    Writes the updated interior back into *u* in-place.
    """
    out = np.empty((rows, cols), dtype=u.dtype)
    for i in prange(rows):
        ip = i + 1  # padded index
        for j in range(cols):
            jp = j + 1
            val = u[ip, jp]
            if np.isnan(val):
                out[i, j] = np.nan
                continue
            n = u[ip - 1, jp]
            s = u[ip + 1, jp]
            w = u[ip, jp - 1]
            e = u[ip, jp + 1]
            lap = n + s + w + e - 4.0 * val
            out[i, j] = val + dt_over_dx2 * alpha_arr[i, j] * lap
    return out


def _diffuse_numpy(data, alpha_arr, steps, dt_over_dx2, boundary):
    rows, cols = data.shape
    _check_memory(rows, cols)
    u = data.astype(np.float64)
    for _ in range(steps):
        if boundary == 'nan':
            padded = np.pad(u, 1, mode='constant', constant_values=np.nan)
        else:
            padded = _pad_array(u, 1, boundary)
        u = _diffuse_step_numpy(padded, alpha_arr, dt_over_dx2, rows, cols)
    return u


# ---- cupy backend ----

@cuda.jit
def _diffuse_step_gpu(u, alpha_arr, dt_over_dx2, out):
    """One forward-Euler step.  *u* is padded (rows+2, cols+2)."""
    i, j = cuda.grid(2)
    rows = out.shape[0]
    cols = out.shape[1]
    if i < rows and j < cols:
        ip = i + 1
        jp = j + 1
        val = u[ip, jp]
        if val != val:  # NaN check in device code
            out[i, j] = val
            return
        n = u[ip - 1, jp]
        s = u[ip + 1, jp]
        w = u[ip, jp - 1]
        e = u[ip, jp + 1]
        lap = n + s + w + e - 4.0 * val
        out[i, j] = val + dt_over_dx2 * alpha_arr[i, j] * lap


def _diffuse_cupy(data, alpha_arr, steps, dt_over_dx2, boundary):
    import cupy as cp

    rows, cols = data.shape
    _check_gpu_memory(rows, cols)
    u = cp.asarray(data, dtype=cp.float64)
    alpha_dev = cp.asarray(alpha_arr, dtype=cp.float64)
    bpg, tpb = calc_cuda_dims((rows, cols))

    for _ in range(steps):
        if boundary == 'nan':
            padded = cp.pad(u, 1, mode='constant', constant_values=cp.nan)
        else:
            padded = _pad_array(u, 1, boundary)
        out = cp.empty_like(u)
        _diffuse_step_gpu[bpg, tpb](padded, alpha_dev, dt_over_dx2, out)
        u = out
    return u


# ---- dask + numpy backend ----

def _diffuse_chunk_numpy(chunk, alpha_chunk, steps, dt_over_dx2, block_info=None):
    """Process a single dask chunk (already overlapped by 1 cell).

    ``alpha_chunk`` may be a scalar (uniform diffusivity) or a 2-D array
    matching the overlapped chunk shape.
    """
    rows = chunk.shape[0] - 2
    cols = chunk.shape[1] - 2
    if np.ndim(alpha_chunk) == 0:
        # Scalar diffusivity — broadcast to interior shape
        interior_alpha = np.broadcast_to(float(alpha_chunk),
                                         (rows, cols))
    else:
        interior_alpha = alpha_chunk[1:-1, 1:-1]

    u = chunk.copy()
    for _ in range(steps):
        interior = _diffuse_step_numpy(u, interior_alpha, dt_over_dx2, rows, cols)
        u[1:-1, 1:-1] = interior
    return u


def _diffuse_dask_numpy(data, alpha, steps, dt_over_dx2, boundary):
    """Dask+numpy backend.

    ``alpha`` is either a Python float (scalar diffusivity) or a dask
    array matching data's shape (spatially varying diffusivity).
    """
    if isinstance(alpha, (int, float, np.floating)):
        # Scalar: pass directly — no full-raster allocation, tiny closure.
        _func = partial(
            _diffuse_chunk_numpy,
            alpha_chunk=float(alpha),
            steps=1,
            dt_over_dx2=dt_over_dx2,
        )
    else:
        # Spatially varying: alpha is a dask array.  map_overlap will
        # feed matching chunks automatically.
        _func = partial(
            _diffuse_chunk_numpy,
            steps=1,
            dt_over_dx2=dt_over_dx2,
        )
    u = data.astype(np.float64)
    for _ in range(steps):
        if isinstance(alpha, (int, float, np.floating)):
            u = u.map_overlap(
                _func,
                depth=(1, 1),
                boundary=_boundary_to_dask(boundary),
                meta=np.array(()),
            )
        else:
            # Pass alpha as a second dask argument to map_overlap
            u = da.map_overlap(
                _diffuse_chunk_numpy,
                u, alpha,
                depth=(1, 1),
                boundary=_boundary_to_dask(boundary),
                meta=np.array(()),
                steps=1,
                dt_over_dx2=dt_over_dx2,
            )
    return u


# ---- dask + cupy backend ----

def _diffuse_chunk_cupy(chunk, alpha_chunk, dt_over_dx2, block_info=None):
    import cupy as cp

    rows = chunk.shape[0] - 2
    cols = chunk.shape[1] - 2
    if np.ndim(alpha_chunk) == 0:
        alpha_dev = cp.full((rows, cols), float(alpha_chunk), dtype=cp.float64)
    else:
        alpha_dev = cp.asarray(alpha_chunk[1:-1, 1:-1], dtype=cp.float64)
    bpg, tpb = calc_cuda_dims((rows, cols))
    out = cp.empty((rows, cols), dtype=cp.float64)
    _diffuse_step_gpu[bpg, tpb](chunk, alpha_dev, dt_over_dx2, out)
    # rebuild padded shape expected by map_overlap
    result = chunk.copy()
    result[1:-1, 1:-1] = out
    return result


def _diffuse_dask_cupy(data, alpha, steps, dt_over_dx2, boundary):
    import cupy as cp

    u = data.astype(cp.float64)
    for _ in range(steps):
        if isinstance(alpha, (int, float, np.floating)):
            _func = partial(
                _diffuse_chunk_cupy,
                alpha_chunk=float(alpha),
                dt_over_dx2=dt_over_dx2,
            )
            u = u.map_overlap(
                _func,
                depth=(1, 1),
                boundary=_boundary_to_dask(boundary, is_cupy=True),
                meta=cp.array(()),
            )
        else:
            u = da.map_overlap(
                _diffuse_chunk_cupy,
                u, alpha,
                depth=(1, 1),
                boundary=_boundary_to_dask(boundary, is_cupy=True),
                meta=cp.array(()),
                dt_over_dx2=dt_over_dx2,
            )
    return u


# ---- public API ----

def diffuse(
    agg,
    diffusivity=1.0,
    steps=1,
    dt=None,
    boundary='nan',
    name='diffuse',
):
    """Run forward-Euler diffusion on a 2D scalar field.

    Solves ``du/dt = alpha * laplacian(u)`` using an explicit 5-point
    stencil for *steps* time steps.

    Parameters
    ----------
    agg : xarray.DataArray
        2D raster representing the initial scalar field (temperature,
        concentration, etc.).
    diffusivity : float or xarray.DataArray
        Thermal/mass diffusivity.  A scalar applies uniformly; a
        DataArray of the same shape allows spatially varying
        diffusivity.
    steps : int, default 1
        Number of time steps to run.  Capped at 100,000 to prevent a
        single call from pinning a CPU indefinitely.
    dt : float or None
        Time step size.  When ``None`` the largest stable step is
        chosen automatically: ``dt = 0.25 * dx**2 / max(alpha)``.
    boundary : str, default ``'nan'``
        Edge handling: ``'nan'``, ``'nearest'``, ``'reflect'``, or
        ``'wrap'``.
    name : str, default ``'diffuse'``
        Name for the output DataArray.

    Returns
    -------
    xarray.DataArray
        Scalar field after *steps* diffusion steps.
    """
    _validate_raster(agg, func_name='diffuse', name='agg', ndim=2)
    _validate_scalar(
        steps,
        func_name='diffuse',
        name='steps',
        dtype=int,
        min_val=1,
        max_val=_MAX_STEPS,
    )
    _validate_boundary(boundary)

    # resolve diffusivity
    #   - scalar: keep as float; only materialised to a full raster if the
    #     dispatched backend is eager (numpy/cupy) and actually needs it
    #   - DataArray: keep as .data (numpy/cupy/dask) for backend dispatch
    if isinstance(diffusivity, xr.DataArray):
        _validate_raster(diffusivity, func_name='diffuse', name='diffusivity', ndim=2)
        if diffusivity.shape != agg.shape:
            raise ValueError(
                f"diffuse(): diffusivity shape {diffusivity.shape} "
                f"does not match agg shape {agg.shape}"
            )
        alpha_scalar = None
        alpha_data = diffusivity.data  # may be numpy, cupy, or dask
        # For numpy/cupy eager paths, materialize to numpy
        if da is not None and isinstance(alpha_data, da.Array):
            alpha_arr_eager = None  # deferred — only built if needed
        else:
            if hasattr(alpha_data, 'get'):
                alpha_arr_eager = alpha_data.get().astype(np.float64)
            else:
                alpha_arr_eager = np.asarray(alpha_data, dtype=np.float64)
    elif isinstance(diffusivity, (int, float)):
        if diffusivity <= 0:
            raise ValueError(
                f"diffuse(): diffusivity must be > 0, got {diffusivity}"
            )
        alpha_scalar = float(diffusivity)
        alpha_data = None
        # Defer the np.full allocation: dask paths can use the scalar
        # directly, and the eager paths build their own broadcast view
        # below after passing the memory guard.
        alpha_arr_eager = None
    else:
        raise TypeError(
            f"diffuse(): diffusivity must be a float or xr.DataArray, "
            f"got {type(diffusivity).__name__}"
        )

    # resolve cell size — assume square cells, use x dimension
    res = agg.attrs.get('res')
    if isinstance(res, (tuple, list, np.ndarray)) and len(res) >= 1:
        dx = float(res[0]) if isinstance(res[0], (int, float)) else 1.0
    elif isinstance(res, (int, float)):
        dx = float(res)
    else:
        dx = 1.0

    if alpha_scalar is not None:
        alpha_max = alpha_scalar
    elif alpha_arr_eager is not None:
        alpha_max = float(np.nanmax(alpha_arr_eager))
    else:
        # dask DataArray diffusivity — compute max lazily
        alpha_max = float(da.nanmax(alpha_data).compute())
    if alpha_max <= 0:
        raise ValueError("diffuse(): all diffusivity values must be > 0")

    # auto dt from CFL stability condition
    if dt is None:
        dt = 0.25 * dx * dx / alpha_max
    else:
        _validate_scalar(dt, func_name='diffuse', name='dt', dtype=(int, float),
                         min_val=0, min_exclusive=True)

    dt_over_dx2 = float(dt) / (dx * dx)

    # Decide which backend will run before materialising any full-raster
    # alpha buffer.  This matters for the dask + scalar case: we want to
    # leave alpha as a scalar all the way through to the per-chunk worker
    # rather than allocating an N*M float64 raster up front.
    is_eager_numpy = isinstance(agg.data, np.ndarray)
    is_eager_cupy = (
        has_cuda_and_cupy() and is_cupy_array(agg.data)
    )
    needs_eager_alpha = is_eager_numpy or is_eager_cupy

    if needs_eager_alpha and alpha_arr_eager is None:
        if alpha_scalar is not None:
            # Run the memory guard *before* allocating the full raster.
            # The guard already accounts for alpha_arr in its budget.
            rows_, cols_ = agg.shape
            if is_eager_cupy:
                _check_gpu_memory(rows_, cols_)
            else:
                _check_memory(rows_, cols_)
            alpha_arr_eager = np.full(agg.shape, alpha_scalar, dtype=np.float64)
        elif alpha_data is not None:
            # Dask DataArray diffusivity but the input array itself is eager.
            # This is unusual but supported: materialise alpha now.
            alpha_arr_eager = alpha_data.compute()
            if hasattr(alpha_arr_eager, 'get'):
                alpha_arr_eager = alpha_arr_eager.get()
            alpha_arr_eager = np.asarray(alpha_arr_eager, dtype=np.float64)

    dask_alpha = alpha_scalar if alpha_scalar is not None else alpha_data

    # dispatch to backend
    mapper = ArrayTypeFunctionMapping(
        numpy_func=partial(_diffuse_numpy, alpha_arr=alpha_arr_eager,
                           steps=steps, dt_over_dx2=dt_over_dx2,
                           boundary=boundary),
        cupy_func=partial(_diffuse_cupy, alpha_arr=alpha_arr_eager,
                          steps=steps, dt_over_dx2=dt_over_dx2,
                          boundary=boundary),
        dask_func=partial(_diffuse_dask_numpy, alpha=dask_alpha,
                          steps=steps, dt_over_dx2=dt_over_dx2,
                          boundary=boundary),
        dask_cupy_func=partial(_diffuse_dask_cupy, alpha=dask_alpha,
                               steps=steps, dt_over_dx2=dt_over_dx2,
                               boundary=boundary),
    )

    out = mapper(agg)(agg.data)

    return DataArray(
        out,
        name=name,
        dims=agg.dims,
        coords=agg.coords,
        attrs=agg.attrs,
    )
