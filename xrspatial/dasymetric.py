"""Raster-based dasymetric mapping.

Dasymetric mapping redistributes aggregate zone-level data (e.g. population
per census tract) onto a finer raster grid using ancillary weight information
(land cover, nighttime lights, etc.).  This is the spatial inverse of
``zonal.stats``.

Three redistribution methods are supported via :func:`disaggregate`:

* ``'binary'`` -- binarize weights (nonzero -> 1), split value equally among
  nonzero-weight pixels in each zone.
* ``'weighted'`` (default) -- distribute proportionally:
  ``pixel = zone_value * pixel_weight / zone_weight_sum``.
* ``'limiting_variable'`` -- three-class dasymetric with per-class density
  caps and iterative redistribution (numpy-only).

Additionally, :func:`pycnophylactic` implements Tobler's smooth
pycnophylactic interpolation, which preserves zone totals through iterative
Laplacian smoothing with mass correction.

:func:`validate_disaggregation` checks that zone totals are preserved within
tolerance after any redistribution.

All four backends (numpy, cupy, dask+numpy, dask+cupy) are supported for the
binary and weighted methods of ``disaggregate`` and for
``validate_disaggregation``.  ``pycnophylactic`` supports numpy and cupy
(via CPU fallback).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

try:
    import dask
    import dask.array as da
    from dask import delayed
except ImportError:
    dask = None
    da = None
    delayed = None

try:
    import cupy
except ImportError:
    class cupy:
        ndarray = False

from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    has_cuda_and_cupy,
    has_dask_array,
    is_cupy_array,
    is_dask_cupy,
)

VALID_METHODS = ('binary', 'weighted', 'limiting_variable')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _normalize_values(values):
    """Convert *values* (dict or pd.Series) to ``dict[int, float]``."""
    if isinstance(values, pd.Series):
        return {int(k): float(v) for k, v in values.items()}
    if isinstance(values, dict):
        return {int(k): float(v) for k, v in values.items()}
    raise TypeError(
        f"'values' must be a dict or pd.Series, got {type(values).__name__}"
    )


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------

def _disaggregate_numpy(zones, weight, values_dict, method, nodata_zone):
    """Core numpy implementation.  Returns a float64 ndarray."""
    zones_arr = np.asarray(zones, dtype=np.float64)
    weight_arr = np.asarray(weight, dtype=np.float64)

    # clamp negative weights to 0
    weight_arr = np.where(weight_arr < 0, 0.0, weight_arr)

    # binarize for binary method
    if method == 'binary':
        weight_arr = np.where(weight_arr != 0, 1.0, 0.0)

    # mask: NaN in zones or weight, or nodata_zone
    invalid = np.isnan(zones_arr) | np.isnan(weight_arr)
    if nodata_zone is not None:
        invalid |= (zones_arr == nodata_zone)

    # compute zone weight sums
    zone_ids = np.array(sorted(values_dict.keys()), dtype=np.float64)
    zone_sums = {}
    for zid in zone_ids:
        mask = (~invalid) & (zones_arr == zid)
        zone_sums[int(zid)] = float(np.nansum(weight_arr[mask]))

    return _disaggregate_numpy_with_sums(
        zones_arr, weight_arr, values_dict, zone_sums, invalid,
    )


def _disaggregate_numpy_with_sums(
    zones_arr, weight_arr, values_dict, zone_sums, invalid,
):
    """Distribute using precomputed zone weight sums.

    Used directly by numpy and also called per-chunk by the dask backend.
    """
    result = np.full(zones_arr.shape, np.nan, dtype=np.float64)

    for zid, zval in values_dict.items():
        mask = (~invalid) & (zones_arr == zid)
        wsum = zone_sums.get(int(zid), 0.0)
        if wsum == 0.0:
            # zero total weight -> NaN
            result[mask] = np.nan
        else:
            result[mask] = zval * weight_arr[mask] / wsum

    return result


# ---------------------------------------------------------------------------
# cupy backend -- CPU fallback
# ---------------------------------------------------------------------------

def _disaggregate_cupy(zones, weight, values_dict, method, nodata_zone):
    """CuPy fallback: transfer to CPU, run numpy, transfer back."""
    zones_np = zones.get()
    weight_np = weight.get()
    result_np = _disaggregate_numpy(zones_np, weight_np, values_dict,
                                    method, nodata_zone)
    return cupy.asarray(result_np)


# ---------------------------------------------------------------------------
# dask+numpy backend
# ---------------------------------------------------------------------------

def _compute_zone_sums_dask(zones_da, weight_da, zone_ids, method,
                            nodata_zone):
    """Eagerly compute per-zone weight sums across all dask chunks.

    Returns a plain ``dict[int, float]``.
    """
    zone_ids_set = set(int(z) for z in zone_ids)

    def _chunk_sums(z_block, w_block):
        z = np.asarray(z_block, dtype=np.float64)
        w = np.asarray(w_block, dtype=np.float64)
        w = np.where(w < 0, 0.0, w)
        if method == 'binary':
            w = np.where(w != 0, 1.0, 0.0)
        invalid = np.isnan(z) | np.isnan(w)
        if nodata_zone is not None:
            invalid |= (z == nodata_zone)
        sums = {}
        for zid in zone_ids_set:
            mask = (~invalid) & (z == zid)
            sums[zid] = float(np.nansum(w[mask]))
        return sums

    # collect delayed chunk sums
    z_blocks = zones_da.to_delayed().ravel()
    w_blocks = weight_da.to_delayed().ravel()
    chunk_results = dask.compute(*[
        delayed(_chunk_sums)(zb, wb) for zb, wb in zip(z_blocks, w_blocks)
    ])

    # merge across chunks
    merged = {int(zid): 0.0 for zid in zone_ids_set}
    for cr in chunk_results:
        for zid, s in cr.items():
            merged[int(zid)] += s
    return merged


def _distribute_chunk(z_block, w_block, values_dict, zone_sums, method,
                      nodata_zone):
    """Map-blocks worker: distribute within a single chunk."""
    z = np.asarray(z_block, dtype=np.float64)
    w = np.asarray(w_block, dtype=np.float64)
    w = np.where(w < 0, 0.0, w)
    if method == 'binary':
        w = np.where(w != 0, 1.0, 0.0)
    invalid = np.isnan(z) | np.isnan(w)
    if nodata_zone is not None:
        invalid |= (z == nodata_zone)
    return _disaggregate_numpy_with_sums(z, w, values_dict, zone_sums,
                                         invalid)


def _disaggregate_dask_numpy(zones_da, weight_da, values_dict, method,
                             nodata_zone):
    """Dask+numpy: eagerly compute zone sums, lazily distribute."""
    zone_ids = list(values_dict.keys())
    zone_sums = _compute_zone_sums_dask(zones_da, weight_da, zone_ids,
                                        method, nodata_zone)

    result = da.map_blocks(
        _distribute_chunk,
        zones_da,
        weight_da,
        values_dict=values_dict,
        zone_sums=zone_sums,
        method=method,
        nodata_zone=nodata_zone,
        dtype=np.float64,
    )
    return result


# ---------------------------------------------------------------------------
# dask+cupy backend
# ---------------------------------------------------------------------------

def _cupy_to_numpy_block(block):
    """Convert a cupy array block to numpy."""
    if hasattr(block, 'get'):
        return block.get()
    return np.asarray(block)


def _disaggregate_dask_cupy(zones_da, weight_da, values_dict, method,
                            nodata_zone):
    """Dask+cupy: convert chunks to numpy, delegate to dask+numpy path."""
    zones_np = da.map_blocks(_cupy_to_numpy_block, zones_da,
                             dtype=zones_da.dtype,
                             meta=np.array((), dtype=zones_da.dtype))
    weight_np = da.map_blocks(_cupy_to_numpy_block, weight_da,
                              dtype=weight_da.dtype,
                              meta=np.array((), dtype=weight_da.dtype))
    return _disaggregate_dask_numpy(zones_np, weight_np, values_dict,
                                   method, nodata_zone)


# ---------------------------------------------------------------------------
# limiting variable (numpy-only)
# ---------------------------------------------------------------------------

def _disaggregate_limiting_numpy(zones, weight, values_dict, nodata_zone,
                                 class_breaks=(0.0,), density_caps=None):
    """Three-class dasymetric with density caps and iterative redistribution.

    Parameters
    ----------
    zones, weight : numpy arrays
    values_dict : dict mapping zone_id -> value
    nodata_zone : int or None
    class_breaks : tuple of floats
        Thresholds that split weight values into classes.  For example
        ``(0.0,)`` creates two classes: zero-weight and nonzero-weight.
    density_caps : sequence of floats or None
        Maximum density (value per pixel) for each class.  If None,
        no capping is applied and the result equals the weighted method.
    """
    zones_arr = np.asarray(zones, dtype=np.float64)
    weight_arr = np.asarray(weight, dtype=np.float64)
    weight_arr = np.where(weight_arr < 0, 0.0, weight_arr)

    invalid = np.isnan(zones_arr) | np.isnan(weight_arr)
    if nodata_zone is not None:
        invalid |= (zones_arr == nodata_zone)

    # classify pixels
    breaks = sorted(class_breaks)
    n_classes = len(breaks) + 1
    pixel_class = np.zeros(zones_arr.shape, dtype=np.int32)
    for i, b in enumerate(breaks):
        pixel_class[weight_arr > b] = i + 1

    if density_caps is None:
        # Class 0 captures zero-weight (uninhabitable) pixels, so its cap
        # must be 0 to prevent population landing on water/parks/etc.
        density_caps = [0.0] + [np.inf] * (n_classes - 1)

    result = np.full(zones_arr.shape, np.nan, dtype=np.float64)

    for zid, zval in values_dict.items():
        zmask = (~invalid) & (zones_arr == zid)
        remaining = zval

        for cls in range(n_classes):
            cls_mask = zmask & (pixel_class == cls)
            n_pixels = int(cls_mask.sum())
            if n_pixels == 0:
                continue
            cap = density_caps[cls]
            allocated = min(remaining, n_pixels * cap)
            if n_pixels > 0:
                result[cls_mask] = allocated / n_pixels
            remaining -= allocated
            if remaining <= 0:
                break

        # distribute any leftover to habitable pixels only (cap > 0)
        if remaining > 0:
            overflow_mask = zmask.copy()
            for cls in range(n_classes):
                if density_caps[cls] <= 0:
                    overflow_mask &= ~(pixel_class == cls)
            n_overflow = int(overflow_mask.sum())
            if n_overflow > 0:
                result[overflow_mask] += remaining / n_overflow

    return result


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def disaggregate(
    zones: xr.DataArray,
    values: Union[dict, pd.Series],
    weight: xr.DataArray,
    method: str = 'weighted',
    nodata_zone: Union[int, None] = None,
    class_breaks: Union[tuple, None] = None,
    density_caps: Union[list, None] = None,
    name: str = 'disaggregate',
) -> xr.DataArray:
    """Redistribute zone-level values onto a raster using ancillary weights.

    This is the spatial inverse of :func:`~xrspatial.zonal.stats`: given
    aggregate values per zone (e.g. population per census tract) and a
    weight surface (e.g. land cover), ``disaggregate`` distributes each
    zone's total across its pixels proportionally to their weights.

    Parameters
    ----------
    zones : xr.DataArray
        2-D integer raster identifying zone membership for each pixel.
    values : dict or pd.Series
        Mapping of ``zone_id -> value`` to redistribute.
    weight : xr.DataArray
        2-D float raster of ancillary weights (e.g. land-cover suitability).
        Negative weights are clamped to zero.
    method : str, default ``'weighted'``
        Redistribution method:

        * ``'binary'`` -- binarize weights (nonzero -> 1), split equally.
        * ``'weighted'`` -- proportional: ``pixel = zone_value *
          pixel_weight / zone_weight_sum``.
        * ``'limiting_variable'`` -- three-class dasymetric with per-class
          density caps (numpy-only).
    nodata_zone : int or None, default None
        Zone ID to treat as no-data (output NaN).
    class_breaks : tuple of float or None, default None
        Only used when ``method='limiting_variable'``.  Thresholds that
        split weight values into classes.  For example ``(0.0,)`` creates
        two classes: zero-weight and nonzero-weight.  Defaults to
        ``(0.0,)`` if None.
    density_caps : list of float or None, default None
        Only used when ``method='limiting_variable'``.  Maximum density
        (value per pixel) for each class.  Must have ``len(class_breaks)+1``
        entries.  If None, the first class (zero-weight pixels) gets
        cap 0 and all others get unlimited capacity.
    name : str, default ``'disaggregate'``
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
        Float raster with the same shape, coordinates, and attributes as
        *zones*.  The sum of output pixels within each zone equals the
        input zone value (conservation property), except where all weights
        are zero (result is NaN).

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> from xrspatial import disaggregate
    >>> zones = xr.DataArray(np.array([[1, 1], [2, 2]]), dims=['y', 'x'])
    >>> weight = xr.DataArray(np.array([[1.0, 3.0], [2.0, 2.0]]), dims=['y', 'x'])
    >>> result = disaggregate(zones, {1: 100, 2: 50}, weight)
    >>> result.values
    array([[25., 75.],
           [25., 25.]])
    """
    # --- validation ---
    if not isinstance(zones, xr.DataArray):
        raise TypeError(
            f"'zones' must be an xr.DataArray, got {type(zones).__name__}"
        )
    if not isinstance(weight, xr.DataArray):
        raise TypeError(
            f"'weight' must be an xr.DataArray, got {type(weight).__name__}"
        )
    if zones.ndim != 2:
        raise ValueError(
            f"'zones' must be 2-D, got {zones.ndim}-D"
        )
    if weight.ndim != 2:
        raise ValueError(
            f"'weight' must be 2-D, got {weight.ndim}-D"
        )
    if zones.shape != weight.shape:
        raise ValueError(
            f"'zones' and 'weight' must have the same shape, "
            f"got {zones.shape} and {weight.shape}"
        )

    method = method.lower()
    if method not in VALID_METHODS:
        raise ValueError(
            f"Invalid method {method!r}. Must be one of {VALID_METHODS}"
        )

    values_dict = _normalize_values(values)

    # --- limiting_variable: numpy-only ---
    if method == 'limiting_variable':
        _data = zones.data
        if has_dask_array() and isinstance(_data, da.Array):
            raise NotImplementedError(
                "method='limiting_variable' is not supported for dask arrays"
            )
        if has_cuda_and_cupy() and is_cupy_array(_data):
            raise NotImplementedError(
                "method='limiting_variable' is not supported for cupy arrays"
            )
        lv_breaks = class_breaks if class_breaks is not None else (0.0,)
        result_data = _disaggregate_limiting_numpy(
            zones.data, weight.data, values_dict, nodata_zone,
            class_breaks=lv_breaks, density_caps=density_caps,
        )
        return xr.DataArray(
            result_data, dims=zones.dims, coords=zones.coords,
            attrs=zones.attrs, name=name,
        )

    # --- dispatch by backend ---
    mapper = ArrayTypeFunctionMapping(
        numpy_func=_disaggregate_numpy,
        cupy_func=_disaggregate_cupy,
        dask_func=_disaggregate_dask_numpy,
        dask_cupy_func=_disaggregate_dask_cupy,
    )
    func = mapper(zones)
    result_data = func(zones.data, weight.data, values_dict, method,
                       nodata_zone)

    return xr.DataArray(
        result_data, dims=zones.dims, coords=zones.coords,
        attrs=zones.attrs, name=name,
    )


# ---------------------------------------------------------------------------
# pycnophylactic interpolation
# ---------------------------------------------------------------------------

def _pycnophylactic_numpy(zones_arr, values_dict, nodata_zone,
                          max_iterations, convergence):
    """Tobler's pycnophylactic interpolation (numpy).

    Algorithm
    ---------
    1. Initialise: each pixel gets ``zone_value / zone_pixel_count``.
    2. Smooth: replace each pixel with the mean of its 4-connected neighbours
       (edge pixels use available neighbours only).
    3. Correct: scale every pixel in each zone so the zone sum matches the
       original value exactly.
    4. Repeat 2-3 until the maximum per-pixel change is below *convergence*.
    """
    zones = np.asarray(zones_arr, dtype=np.float64)
    nrows, ncols = zones.shape

    # identify valid pixels per zone and build initial surface
    invalid = np.isnan(zones)
    if nodata_zone is not None:
        invalid |= (zones == nodata_zone)

    surface = np.full(zones.shape, np.nan, dtype=np.float64)
    zone_masks = {}
    zone_counts = {}
    zone_vals = {}

    for zid, zval in values_dict.items():
        mask = (~invalid) & (zones == zid)
        n = int(mask.sum())
        if n == 0:
            continue
        zone_masks[zid] = mask
        zone_counts[zid] = n
        zone_vals[zid] = zval
        surface[mask] = zval / n

    # pixels that belong to some zone (valid for smoothing)
    valid = ~np.isnan(surface)

    for _ in range(max_iterations):
        # Laplacian smoothing: mean of 4-connected neighbours
        smoothed = np.full_like(surface, np.nan)
        neighbour_sum = np.zeros_like(surface)
        neighbour_count = np.zeros_like(surface)

        # shift in each direction, accumulate
        if nrows > 1:
            neighbour_sum[1:, :] += np.where(valid[:-1, :], surface[:-1, :], 0)
            neighbour_count[1:, :] += valid[:-1, :].astype(np.float64)
            neighbour_sum[:-1, :] += np.where(valid[1:, :], surface[1:, :], 0)
            neighbour_count[:-1, :] += valid[1:, :].astype(np.float64)
        if ncols > 1:
            neighbour_sum[:, 1:] += np.where(valid[:, :-1], surface[:, :-1], 0)
            neighbour_count[:, 1:] += valid[:, :-1].astype(np.float64)
            neighbour_sum[:, :-1] += np.where(valid[:, 1:], surface[:, 1:], 0)
            neighbour_count[:, :-1] += valid[:, 1:].astype(np.float64)

        has_neighbours = neighbour_count > 0
        smoothed[valid & has_neighbours] = (
            neighbour_sum[valid & has_neighbours]
            / neighbour_count[valid & has_neighbours]
        )
        # pixels with no valid neighbours keep their current value
        smoothed[valid & ~has_neighbours] = surface[valid & ~has_neighbours]

        # mass correction: rescale each zone to preserve total
        for zid in zone_masks:
            mask = zone_masks[zid]
            current_sum = np.nansum(smoothed[mask])
            if current_sum != 0:
                smoothed[mask] *= zone_vals[zid] / current_sum
            else:
                # all smoothed to zero -- revert to uniform
                smoothed[mask] = zone_vals[zid] / zone_counts[zid]

        # convergence check
        max_change = np.nanmax(np.abs(smoothed[valid] - surface[valid]))
        surface = smoothed
        if max_change < convergence:
            break

    return surface


def pycnophylactic(
    zones: xr.DataArray,
    values: Union[dict, pd.Series],
    max_iterations: int = 100,
    convergence: float = 1e-5,
    nodata_zone: Union[int, None] = None,
    name: str = 'pycnophylactic',
) -> xr.DataArray:
    """Tobler's pycnophylactic interpolation preserving zone totals.

    Produces a smooth surface by iteratively applying Laplacian smoothing
    and then rescaling each zone so its pixel sum matches the original
    value.  Unlike :func:`disaggregate`, no ancillary weight raster is
    needed -- smoothness alone drives the redistribution.

    Parameters
    ----------
    zones : xr.DataArray
        2-D integer raster identifying zone membership for each pixel.
    values : dict or pd.Series
        Mapping of ``zone_id -> value`` to redistribute.
    max_iterations : int, default 100
        Maximum number of smoothing-correction iterations.
    convergence : float, default 1e-5
        Stop when the largest per-pixel change is below this threshold.
    nodata_zone : int or None, default None
        Zone ID to treat as no-data (output NaN).
    name : str, default ``'pycnophylactic'``
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
        Float raster with the same shape, coordinates, and attributes as
        *zones*.  Zone totals are preserved (pycnophylactic property).

    Notes
    -----
    Currently supports numpy and cupy (via CPU fallback) backends.
    Dask arrays raise ``NotImplementedError`` because the algorithm is
    inherently iterative and requires global zone sums each iteration.

    References
    ----------
    Tobler, W. R. (1979). "Smooth Pycnophylactic Interpolation for
    Geographical Regions". *Journal of the American Statistical
    Association*, 74(367), 519-530.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> from xrspatial.dasymetric import pycnophylactic
    >>> zones = xr.DataArray(np.array([[1, 1], [2, 2]]), dims=['y', 'x'])
    >>> result = pycnophylactic(zones, {1: 100, 2: 50})
    >>> float(np.nansum(result.values[:1, :]))  # zone 1 sum
    100.0
    """
    # --- validation ---
    if not isinstance(zones, xr.DataArray):
        raise TypeError(
            f"'zones' must be an xr.DataArray, got {type(zones).__name__}"
        )
    if zones.ndim != 2:
        raise ValueError(
            f"'zones' must be 2-D, got {zones.ndim}-D"
        )

    values_dict = _normalize_values(values)

    _data = zones.data

    # dask not supported (iterative algorithm needs global state each step)
    if has_dask_array() and isinstance(_data, da.Array):
        raise NotImplementedError(
            "pycnophylactic is not supported for dask arrays. "
            "Compute zones first with zones.compute()."
        )

    # cupy: CPU fallback
    if has_cuda_and_cupy() and is_cupy_array(_data):
        zones_np = _data.get()
        result_data = _pycnophylactic_numpy(
            zones_np, values_dict, nodata_zone, max_iterations, convergence,
        )
        result_data = cupy.asarray(result_data)
    else:
        result_data = _pycnophylactic_numpy(
            _data, values_dict, nodata_zone, max_iterations, convergence,
        )

    return xr.DataArray(
        result_data, dims=zones.dims, coords=zones.coords,
        attrs=zones.attrs, name=name,
    )


# ---------------------------------------------------------------------------
# validation helper
# ---------------------------------------------------------------------------

def validate_disaggregation(
    result: xr.DataArray,
    zones: xr.DataArray,
    values: Union[dict, pd.Series],
    tolerance: float = 1e-6,
) -> bool:
    """Check that zone totals in *result* match *values* within tolerance.

    Parameters
    ----------
    result : xr.DataArray
        Output of :func:`disaggregate` or :func:`pycnophylactic`.
    zones : xr.DataArray
        The zone raster used to produce *result*.
    values : dict or pd.Series
        The original ``zone_id -> value`` mapping.
    tolerance : float, default 1e-6
        Maximum allowed relative error per zone.  Zones whose total
        weight is zero (result all NaN) are skipped.

    Returns
    -------
    bool
        ``True`` if all zone totals are within tolerance.

    Raises
    ------
    ValueError
        If any zone total exceeds the tolerance, with details on
        which zones failed.

    Examples
    --------
    >>> from xrspatial.dasymetric import disaggregate, validate_disaggregation
    >>> result = disaggregate(zones, values, weight)
    >>> validate_disaggregation(result, zones, values)
    True
    """
    values_dict = _normalize_values(values)

    # extract numpy arrays from any backend
    result_np = _to_numpy(result.data)
    zones_np = _to_numpy(zones.data)

    failures = []
    for zid, expected in values_dict.items():
        mask = zones_np == zid
        if not np.any(mask):
            continue
        actual = float(np.nansum(result_np[mask]))
        # skip zones that are all NaN (zero-weight zones)
        if np.all(np.isnan(result_np[mask])):
            continue
        if expected == 0:
            if abs(actual) > tolerance:
                failures.append((zid, expected, actual))
        else:
            rel_err = abs(actual - expected) / abs(expected)
            if rel_err > tolerance:
                failures.append((zid, expected, actual))

    if failures:
        lines = [f"  zone {z}: expected={e}, actual={a}" for z, e, a in failures]
        raise ValueError(
            f"Zone totals not preserved (tolerance={tolerance}):\n"
            + "\n".join(lines)
        )
    return True


def _to_numpy(data):
    """Convert array data from any backend to a numpy ndarray."""
    if has_dask_array() and isinstance(data, da.Array):
        data = data.compute()
    if has_cuda_and_cupy() and is_cupy_array(data):
        data = data.get()
    return np.asarray(data)
