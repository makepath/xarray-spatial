"""Combination methods for MCDA criterion layers.

All functions take an ``xr.Dataset`` of standardized (0-1) criterion
layers and return a single composite ``xr.DataArray``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _validate_criteria(criteria: xr.Dataset) -> None:
    if not isinstance(criteria, xr.Dataset):
        raise TypeError(
            f"criteria must be an xr.Dataset, got {type(criteria).__name__}"
        )
    if len(criteria.data_vars) == 0:
        raise ValueError("criteria Dataset has no variables")


def _check_finite_weights(weights: dict[str, float], label: str) -> None:
    """Reject weight dicts that contain NaN or infinite values.

    Without this guard, a single NaN weight slips past the sum-to-1 check
    (``abs(NaN - 1.0) > 0.01`` is False) and propagates through every
    pixel, returning an all-NaN raster with no error.
    """
    bad = []
    for name, value in weights.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{label}[{name!r}] must be a real number, got {value!r}"
            )
        if not np.isfinite(v):
            bad.append((name, v))
    if bad:
        details = ", ".join(f"{n!r}={v}" for n, v in bad)
        raise ValueError(
            f"{label} must be finite; got non-finite value(s): {details}"
        )


def _validate_weights(
    weights: dict[str, float], criteria: xr.Dataset
) -> None:
    criteria_names = set(criteria.data_vars)
    weight_names = set(weights)
    missing = criteria_names - weight_names
    if missing:
        raise ValueError(f"Missing weights for criteria: {missing}")
    extra = weight_names - criteria_names
    if extra:
        raise ValueError(
            f"Weights contain keys not in criteria Dataset: {extra}"
        )
    _check_finite_weights(weights, "weights")
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Weights must sum to ~1.0, got {total:.4f}"
        )


def wlc(
    criteria: xr.Dataset,
    weights: dict[str, float],
    name: str = "wlc",
) -> xr.DataArray:
    """Weighted Linear Combination.

    Fully compensatory: per-pixel ``sum(w_i * x_i)``.

    Parameters
    ----------
    criteria : xr.Dataset
        Standardized criterion layers (0-1).
    weights : dict
        ``{criterion_name: weight}``, must sum to ~1.0.
    name : str
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        Composite suitability surface.
    """
    _validate_criteria(criteria)
    _validate_weights(weights, criteria)

    result = None
    for var_name in criteria.data_vars:
        w = weights[var_name]
        term = criteria[var_name] * w
        result = term if result is None else result + term

    result.name = name
    return result


def _check_wpm_positive(criteria: xr.Dataset) -> None:
    """Reject criterion layers with zero or negative values.

    ``wpm`` computes ``prod(x_i ** w_i)``. For non-integer weights this is
    only well-defined when ``x_i > 0``: a single zero collapses the
    product to zero (masking upstream bugs) and a negative base produces
    NaN with no error. NaN values are allowed through so the documented
    NaN-propagation behaviour is preserved.
    """
    bad = []
    for var_name in criteria.data_vars:
        arr = criteria[var_name].data
        # Mask NaN so they pass through; we only want to flag <= 0.
        try:
            import dask.array as da
            if isinstance(arr, da.Array):
                # Compute once; cheap relative to the full product pass.
                min_val = float(da.nanmin(arr).compute())
            else:
                min_val = float(np.nanmin(arr))
        except ImportError:
            min_val = float(np.nanmin(arr))
        except ValueError:
            # All-NaN slice; nothing to flag.
            continue
        if not np.isnan(min_val) and min_val <= 0.0:
            bad.append((var_name, min_val))
    if bad:
        details = ", ".join(f"{n!r} (min={v})" for n, v in bad)
        raise ValueError(
            "wpm requires strictly positive criterion values; got "
            f"non-positive value(s) in: {details}"
        )


def wpm(
    criteria: xr.Dataset,
    weights: dict[str, float],
    name: str = "wpm",
) -> xr.DataArray:
    """Weighted Product Model.

    Multiplicative combination: per-pixel ``product(x_i ^ w_i)``.
    More sensitive to low scores than WLC.

    Parameters
    ----------
    criteria : xr.Dataset
        Standardized criterion layers. Values must be strictly positive
        (``> 0``); zero or negative inputs raise ``ValueError`` because
        ``x ** w`` is undefined or collapses to zero for non-integer
        weights. NaN values propagate through to the output.
    weights : dict
        ``{criterion_name: weight}``, must sum to ~1.0.
    name : str
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        Composite suitability surface.
    """
    _validate_criteria(criteria)
    _validate_weights(weights, criteria)
    _check_wpm_positive(criteria)

    result = None
    for var_name in criteria.data_vars:
        w = weights[var_name]
        term = criteria[var_name] ** w
        result = term if result is None else result * term

    result.name = name
    return result


def owa(
    criteria: xr.Dataset,
    criterion_weights: dict[str, float],
    order_weights: list[float],
    name: str = "owa",
) -> xr.DataArray:
    """Ordered Weighted Averaging.

    Generalizes WLC by adding position-based order weights that control
    risk attitude. Order weights are applied to criterion values sorted
    by magnitude at each pixel, not by criterion identity.

    Parameters
    ----------
    criteria : xr.Dataset
        Standardized criterion layers (0-1).
    criterion_weights : dict
        ``{criterion_name: weight}``, must sum to ~1.0.
    order_weights : list of float
        Weights applied by rank position (index 0 = highest value).
        Must have the same length as the number of criteria and
        sum to ~1.0.
    name : str
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        Composite suitability surface.
    """
    _validate_criteria(criteria)
    _validate_weights(criterion_weights, criteria)

    n = len(criteria.data_vars)
    if len(order_weights) != n:
        raise ValueError(
            f"order_weights length ({len(order_weights)}) must match "
            f"number of criteria ({n})"
        )
    # Reject NaN/Inf entries before the sum check, since
    # ``abs(NaN - 1.0) > 0.01`` is False and would let the bad value through.
    bad_ow = []
    for i, value in enumerate(order_weights):
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"order_weights[{i}] must be a real number, got {value!r}"
            )
        if not np.isfinite(v):
            bad_ow.append((i, v))
    if bad_ow:
        details = ", ".join(f"[{i}]={v}" for i, v in bad_ow)
        raise ValueError(
            f"order_weights must be finite; got non-finite value(s): "
            f"{details}"
        )
    ow_sum = sum(order_weights)
    if abs(ow_sum - 1.0) > 0.01:
        raise ValueError(
            f"order_weights must sum to ~1.0, got {ow_sum:.4f}"
        )

    order_weights_arr = np.array(order_weights, dtype=np.float64)

    # First apply criterion weights
    weighted_layers = []
    for var_name in criteria.data_vars:
        w = criterion_weights[var_name]
        weighted_layers.append(criteria[var_name] * w * n)

    # Stack, sort descending along criterion axis, apply order weights
    stacked = xr.concat(weighted_layers, dim="__mcda_criterion")
    sorted_data = _sort_descending(stacked.data, axis=0)

    # Apply order weights along the criterion axis
    # Reshape order weights for broadcasting
    shape = [n] + [1] * (sorted_data.ndim - 1)

    try:
        import dask.array as da
        if isinstance(sorted_data, da.Array):
            ow = da.from_array(order_weights_arr.reshape(shape),
                               chunks=-1)
        else:
            ow = order_weights_arr.reshape(shape)
    except ImportError:
        ow = order_weights_arr.reshape(shape)

    result_data = (sorted_data * ow).sum(axis=0)

    # Pick dims/coords from first data var
    first_var = list(criteria.data_vars)[0]
    template = criteria[first_var]
    return xr.DataArray(
        result_data, name=name, dims=template.dims,
        coords=template.coords, attrs=template.attrs,
    )


def _sort_descending(data, axis):
    """Sort array descending along axis, supporting dask."""
    try:
        import dask.array as da
        if isinstance(data, da.Array):
            # Negate, sort, negate back to get descending order
            return -da.sort(-data, axis=axis)
    except ImportError:
        pass

    try:
        import cupy
        if isinstance(data, cupy.ndarray):
            return -cupy.sort(-data, axis=axis)
    except ImportError:
        pass

    return -np.sort(-data, axis=axis)


def fuzzy_overlay(
    criteria: xr.Dataset,
    operator: str = "and",
    gamma: float = 0.9,
    name: str = "fuzzy_overlay",
) -> xr.DataArray:
    """Combine criteria using fuzzy set operators.

    No explicit weights. All criteria contribute equally through the
    chosen fuzzy operator.

    Parameters
    ----------
    criteria : xr.Dataset
        Standardized criterion layers (0-1).
    operator : str
        Fuzzy operator: ``"and"`` (min), ``"or"`` (max),
        ``"sum"``, ``"product"``, or ``"gamma"``.
    gamma : float
        Gamma parameter for the ``"gamma"`` operator (0 = product,
        1 = sum). Only used when ``operator="gamma"``.
    name : str
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        Composite suitability surface.
    """
    _validate_criteria(criteria)

    var_names = list(criteria.data_vars)
    layers = [criteria[v] for v in var_names]

    if operator == "and":
        result = layers[0]
        for layer in layers[1:]:
            result = np.minimum(result, layer)
    elif operator == "or":
        result = layers[0]
        for layer in layers[1:]:
            result = np.maximum(result, layer)
    elif operator == "product":
        result = layers[0]
        for layer in layers[1:]:
            result = result * layer
    elif operator == "sum":
        # Fuzzy algebraic sum: 1 - product(1 - x_i)
        complement = layers[0] * 0.0 + 1.0
        for layer in layers:
            complement = complement * (1.0 - layer)
        result = 1.0 - complement
    elif operator == "gamma":
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        # product part
        prod = layers[0]
        for layer in layers[1:]:
            prod = prod * layer
        # sum part
        complement = layers[0] * 0.0 + 1.0
        for layer in layers:
            complement = complement * (1.0 - layer)
        fuzzy_sum = 1.0 - complement
        result = (fuzzy_sum ** gamma) * (prod ** (1.0 - gamma))
    else:
        raise ValueError(
            f"Unknown operator {operator!r}. Choose from "
            "'and', 'or', 'sum', 'product', 'gamma'"
        )

    result.name = name
    return result


def boolean_overlay(
    criteria: dict[str, xr.DataArray],
    operator: str = "and",
    name: str = "boolean_overlay",
) -> xr.DataArray:
    """Combine binary (boolean) criterion masks.

    Parameters
    ----------
    criteria : dict of str to xr.DataArray
        Named boolean/binary criterion masks.
    operator : str
        ``"and"`` (intersection) or ``"or"`` (union).
    name : str
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        Binary suitability mask.
    """
    if not criteria:
        raise ValueError("criteria dict is empty")

    # Cast to bool to handle numeric input (0/1, float thresholds)
    layers = [v.astype(bool) for v in criteria.values()]
    if operator == "and":
        result = layers[0]
        for layer in layers[1:]:
            result = result & layer
    elif operator == "or":
        result = layers[0]
        for layer in layers[1:]:
            result = result | layer
    else:
        raise ValueError(
            f"Unknown operator {operator!r}. Choose from 'and', 'or'"
        )

    result.name = name
    return result
