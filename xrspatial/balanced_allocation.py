"""Balanced service area partitioning.

Partitions a cost surface into territories of roughly equal cost-weighted
area.  Each cell is assigned to the source whose biased cost-distance is
lowest, where biases are iteratively adjusted so that no source's territory
is disproportionately large or small.

Algorithm
---------
1. Run ``cost_distance`` once per source to get N cost surfaces.
2. Assign each cell to ``argmin(cost[i] + bias[i])``.
3. Compute each territory's cost-weighted area (sum of friction values).
4. Adjust biases proportionally: increase for over-large territories,
   decrease for under-served ones.
5. Repeat 2-4 until convergence or ``max_iterations``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xrspatial.cost_distance import cost_distance
from xrspatial.utils import _validate_raster

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import cupy
except ImportError:
    class cupy:  # type: ignore[no-redef]
        ndarray = False


def _to_numpy(arr):
    """Convert any array (numpy, cupy, dask) to a numpy array."""
    if da is not None and isinstance(arr, da.Array):
        arr = arr.compute()
    if hasattr(arr, 'get'):  # cupy
        return arr.get()
    return np.asarray(arr)


def _as_numpy(arr):
    """Convert a computed array (numpy or cupy) to numpy."""
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def _extract_sources(raster, target_values):
    """Return sorted array of unique source IDs from the raster.

    For dask arrays, uses ``da.unique`` (per-chunk reduction) so the full
    raster is never pulled into RAM just to discover source IDs.
    """
    if len(target_values) > 0:
        ids = np.asarray(target_values, dtype=np.float64)
        return ids[np.isfinite(ids)]

    data = raster.data
    if da is not None and isinstance(data, da.Array):
        uniq = da.unique(data).compute()  # small result array
        if hasattr(uniq, 'get'):  # cupy
            uniq = uniq.get()
        mask = np.isfinite(uniq) & (uniq != 0)
        return np.sort(uniq[mask])
    data_np = _to_numpy(data)
    mask = np.isfinite(data_np) & (data_np != 0)
    return np.unique(data_np[mask])


def _make_single_source_raster(raster, source_id):
    """Create a raster with only one source value, rest zero."""
    data = raster.data
    # Build mask for this source
    if da is not None and isinstance(data, da.Array):
        single = da.where(data == source_id, source_id, 0.0)
    elif hasattr(data, 'get'):  # cupy
        import cupy as cp
        single = cp.where(data == source_id, source_id, 0.0)
    else:
        single = np.where(data == source_id, source_id, 0.0)

    return xr.DataArray(
        single,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )


def _sum_where(friction_data, alloc_data, source_id):
    """Sum friction values where allocation matches source_id.

    Returns a Python float.  Works with numpy, cupy, and dask arrays.
    """
    if da is not None and isinstance(friction_data, da.Array):
        masked = da.where(alloc_data == source_id, friction_data, 0.0)
        return float(masked.sum().compute())
    elif hasattr(friction_data, 'get'):  # cupy
        import cupy as cp
        masked = cp.where(alloc_data == source_id, friction_data, 0.0)
        return float(cp.asnumpy(masked.sum()))
    else:
        masked = np.where(alloc_data == source_id, friction_data, 0.0)
        return float(masked.sum())


def _allocate_from_costs(cost_stack, source_ids, fill_value=np.nan):
    """Assign each cell to the source with lowest cost.

    Parameters
    ----------
    cost_stack : list of arrays
        Per-source cost-distance arrays (same shape).
    source_ids : 1-D numpy array
        Source ID for each layer in cost_stack.
    fill_value : float
        Value for cells unreachable from any source.

    Returns
    -------
    allocation : array
        2-D array of source IDs.
    """
    # Stack along axis 0 -> shape (N, H, W)
    first = cost_stack[0]
    if da is not None and isinstance(first, da.Array):
        stacked = da.stack(cost_stack, axis=0)
        # Replace NaN with inf for argmin
        stacked_clean = da.where(da.isnan(stacked), np.inf, stacked)
        best_idx = da.argmin(stacked_clean, axis=0).compute()
        best_idx = _as_numpy(best_idx)
    elif hasattr(first, 'get'):  # cupy
        import cupy as cp
        stacked = cp.stack(cost_stack, axis=0)
        stacked_clean = cp.where(cp.isnan(stacked), cp.inf, stacked)
        best_idx = cp.asnumpy(cp.argmin(stacked_clean, axis=0))
    else:
        stacked = np.stack(cost_stack, axis=0)
        stacked_clean = np.where(np.isnan(stacked), np.inf, stacked)
        best_idx = np.argmin(stacked_clean, axis=0)

    # Map index back to source ID
    alloc = source_ids[best_idx]

    # Mark cells that are unreachable from all sources
    if da is not None and isinstance(first, da.Array):
        all_nan = da.all(da.isnan(da.stack(cost_stack, axis=0)), axis=0)
        all_nan = _as_numpy(all_nan.compute())
    elif hasattr(first, 'get'):
        import cupy as cp
        all_nan = cp.asnumpy(
            cp.all(cp.isnan(cp.stack(cost_stack, axis=0)), axis=0)
        )
    else:
        all_nan = np.all(np.isnan(np.stack(cost_stack, axis=0)), axis=0)

    alloc = alloc.astype(np.float64)
    alloc[all_nan] = fill_value

    return alloc


def _allocate_biased(cost_stack, biases, source_ids, fill_value=np.nan):
    """Assign each cell to source with lowest (cost + bias).

    Like _allocate_from_costs but adds per-source bias before argmin.
    """
    first = cost_stack[0]
    n = len(cost_stack)

    if da is not None and isinstance(first, da.Array):
        layers = []
        for i in range(n):
            layer = da.where(da.isnan(cost_stack[i]), np.inf,
                             cost_stack[i] + biases[i])
            layers.append(layer)
        stacked = da.stack(layers, axis=0)
        best_idx = da.argmin(stacked, axis=0).compute()
        best_idx = _as_numpy(best_idx)
    elif hasattr(first, 'get'):
        import cupy as cp
        layers = []
        for i in range(n):
            layer = cp.where(cp.isnan(cost_stack[i]), cp.inf,
                             cost_stack[i] + biases[i])
            layers.append(layer)
        stacked = cp.stack(layers, axis=0)
        best_idx = cp.asnumpy(cp.argmin(stacked, axis=0))
    else:
        layers = []
        for i in range(n):
            layer = np.where(np.isnan(cost_stack[i]), np.inf,
                             cost_stack[i] + biases[i])
            layers.append(layer)
        stacked = np.stack(layers, axis=0)
        best_idx = np.argmin(stacked, axis=0)

    alloc = source_ids[best_idx].astype(np.float64)

    # Mark unreachable cells
    if da is not None and isinstance(first, da.Array):
        all_nan = da.all(da.isnan(da.stack(cost_stack, axis=0)), axis=0)
        all_nan = _as_numpy(all_nan.compute())
    elif hasattr(first, 'get'):
        import cupy as cp
        all_nan = cp.asnumpy(
            cp.all(cp.isnan(cp.stack(cost_stack, axis=0)), axis=0)
        )
    else:
        all_nan = np.all(np.isnan(np.stack(cost_stack, axis=0)), axis=0)

    alloc[all_nan] = fill_value
    return alloc


def balanced_allocation(
    raster: xr.DataArray,
    friction: xr.DataArray,
    x: str = "x",
    y: str = "y",
    target_values: list = [],
    max_cost: float = np.inf,
    connectivity: int = 8,
    tolerance: float = 0.05,
    max_iterations: int = 100,
    learning_rate: float = 0.5,
) -> xr.DataArray:
    """Partition a cost surface into balanced service territories.

    Assigns each cell to a source such that all territories have roughly
    equal cost-weighted area (sum of friction values).  This extends
    standard cost-distance allocation by iteratively adjusting per-source
    biases until the workload is balanced.

    Parameters
    ----------
    raster : xr.DataArray
        2-D source raster.  Source pixels are identified by non-zero
        finite values (or values in *target_values*).  Each unique value
        is treated as a separate source.
    friction : xr.DataArray
        2-D friction (cost) surface.  Must have the same shape and
        coordinates as *raster*.
    x : str, default='x'
        Name of the x coordinate.
    y : str, default='y'
        Name of the y coordinate.
    target_values : list, optional
        Specific pixel values in *raster* to treat as sources.
        If empty, all non-zero finite pixels are sources.
    max_cost : float, default=np.inf
        Maximum accumulated cost passed to ``cost_distance``.
    connectivity : int, default=8
        Pixel connectivity (4 or 8) passed to ``cost_distance``.
    tolerance : float, default=0.05
        Convergence threshold.  The loop stops when every territory's
        cost-weighted area is within ``tolerance`` of the mean (as a
        fraction of the mean).
    max_iterations : int, default=100
        Maximum number of bias-adjustment iterations.
    learning_rate : float, default=0.5
        Controls how aggressively biases are updated each iteration.
        Smaller values are more stable; larger values converge faster.

    Returns
    -------
    xr.DataArray
        2-D array of source IDs (float32).  Each cell contains the ID
        of the source it is assigned to.  Unreachable cells are NaN.
    """
    _validate_raster(raster, func_name='balanced_allocation', name='raster')
    _validate_raster(friction, func_name='balanced_allocation',
                     name='friction')
    if raster.shape != friction.shape:
        raise ValueError("raster and friction must have the same shape")
    if raster.dims != (y, x):
        raise ValueError(
            f"raster.dims should be ({y!r}, {x!r}), got {raster.dims}"
        )
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    source_ids = _extract_sources(raster, target_values)
    n_sources = len(source_ids)

    if n_sources == 0:
        out = np.full(raster.shape, np.nan, dtype=np.float32)
        return xr.DataArray(out, coords=raster.coords, dims=raster.dims,
                            attrs=raster.attrs)

    if n_sources == 1:
        # Only one source: every reachable cell goes to it
        cd = cost_distance(raster, friction, x=x, y=y,
                           target_values=list(target_values),
                           max_cost=max_cost, connectivity=connectivity)
        cd_np = _to_numpy(cd.data)
        out = np.where(np.isfinite(cd_np), source_ids[0], np.nan)
        return xr.DataArray(out.astype(np.float32), coords=raster.coords,
                            dims=raster.dims, attrs=raster.attrs)

    # Memory guard: we hold N cost surfaces + friction simultaneously.
    # Estimate total footprint before doing any expensive work.
    array_bytes = np.prod(raster.shape) * 8  # float64
    # N cost surfaces + friction + allocation + stacked intermediate
    total_estimate = array_bytes * (n_sources + 3)
    try:
        from xrspatial.zonal import _available_memory_bytes
        avail = _available_memory_bytes()
    except ImportError:
        avail = 2 * 1024**3
    if total_estimate > 0.8 * avail:
        raise MemoryError(
            f"balanced_allocation with {n_sources} sources needs "
            f"~{total_estimate / 1e9:.1f} GB ({n_sources} cost surfaces "
            f"+ friction + intermediates) but only ~{avail / 1e9:.1f} GB "
            f"available.  Reduce the number of sources, downsample the "
            f"raster, or increase available memory."
        )

    # Step 1: compute per-source cost-distance surfaces
    cost_surfaces = []  # list of raw data arrays (numpy/cupy/dask)
    for sid in source_ids:
        single = _make_single_source_raster(raster, sid)
        cd = cost_distance(single, friction, x=x, y=y,
                           max_cost=max_cost, connectivity=connectivity)
        cost_surfaces.append(cd.data)

    # Step 2: get friction data for weighting
    fric_data = friction.data
    if da is not None and isinstance(fric_data, da.Array):
        fric_np = fric_data.compute()
        if hasattr(fric_np, 'get'):
            fric_np = fric_np.get()
    elif hasattr(fric_data, 'get'):
        fric_np = fric_data.get()
    else:
        fric_np = np.asarray(fric_data)

    # Replace non-positive and NaN friction with 0 for weighting
    fric_weight = np.where(np.isfinite(fric_np) & (fric_np > 0), fric_np, 0.0)

    # Step 3: iterative balancing
    biases = np.zeros(n_sources, dtype=np.float64)

    for iteration in range(max_iterations):
        # Allocate with current biases
        alloc = _allocate_biased(cost_surfaces, biases, source_ids)

        # Compute per-territory cost-weighted area
        weights = np.array([
            float(np.sum(fric_weight[alloc == sid]))
            for sid in source_ids
        ])

        # Handle sources with no reachable cells
        total = weights.sum()
        if total == 0:
            break

        target_weight = total / n_sources

        # Check convergence: max relative deviation
        nonzero = weights > 0
        if np.any(nonzero):
            max_dev = np.max(np.abs(weights - target_weight)) / target_weight
            if max_dev <= tolerance:
                break

        # Update biases proportionally
        for i in range(n_sources):
            if target_weight > 0:
                deviation = (weights[i] - target_weight) / target_weight
                biases[i] += learning_rate * deviation

    # Build output DataArray
    return xr.DataArray(
        alloc.astype(np.float32),
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )
