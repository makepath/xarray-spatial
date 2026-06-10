"""Sensitivity analysis for MCDA weight stability."""

from __future__ import annotations

import numpy as np
import xarray as xr

from xrspatial.mcda.combine import (
    _check_wpm_positive,
    _validate_criteria,
    _validate_weights,
    wlc,
    wpm,
)
from xrspatial.mcda.standardize import _get_xp


def sensitivity(
    criteria: xr.Dataset,
    weights: dict[str, float],
    method: str = "one_at_a_time",
    combine_method: str = "wlc",
    delta: float = 0.05,
    n_samples: int = 1000,
    seed: int = 42,
    name: str = "sensitivity",
) -> xr.DataArray | xr.Dataset:
    """Assess how weight changes affect the suitability surface.

    Parameters
    ----------
    criteria : xr.Dataset
        Standardized criterion layers.
    weights : dict
        Base weights (must sum to ~1.0).
    method : str
        ``"one_at_a_time"`` or ``"monte_carlo"``.
    combine_method : str
        Combination function: ``"wlc"`` or ``"wpm"``.
    delta : float
        Perturbation magnitude for one-at-a-time (default 0.05).
    n_samples : int
        Number of random weight vectors for Monte Carlo (default 1000).
        Must be >= 1 when ``method="monte_carlo"``.
    seed : int
        Random seed for Monte Carlo reproducibility.
    name : str
        Name prefix for output arrays.

    Returns
    -------
    xr.Dataset or xr.DataArray
        For ``"one_at_a_time"``: Dataset with per-criterion DataArrays
        showing the absolute score difference under perturbation.
        For ``"monte_carlo"``: DataArray of per-pixel coefficient of
        variation across random weight samples. Dask-backed criteria
        produce a lazy dask-backed result; the sample loop runs inside
        each chunk, so memory stays bounded by chunk size.
    """
    combine_fn = {"wlc": wlc, "wpm": wpm}.get(combine_method)
    if combine_fn is None:
        raise ValueError(
            f"Unknown combine_method {combine_method!r}. "
            "Choose 'wlc' or 'wpm'"
        )

    if method == "one_at_a_time":
        return _oat(criteria, weights, combine_fn, delta, name)
    elif method == "monte_carlo":
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError(
                f"n_samples must be a positive integer >= 1, got {n_samples!r}"
            )
        return _monte_carlo(
            criteria, weights, combine_method, int(n_samples), seed, name
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. "
            "Choose 'one_at_a_time' or 'monte_carlo'"
        )


def _perturb_weights(
    weights: dict[str, float], target: str, delta: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Create +delta and -delta weight variants for one criterion.

    The target criterion's weight shifts by delta; remaining weights
    are rescaled proportionally to maintain a sum of 1.0.
    """
    names = list(weights.keys())
    base_w = weights[target]
    others = [k for k in names if k != target]
    other_sum = sum(weights[k] for k in others)

    # Single criterion: weight must stay at 1.0, perturbation is a no-op
    if not others:
        copy = dict(weights)
        return copy, dict(copy)

    results = []
    for sign in (+1, -1):
        new_target = max(0.0, min(1.0, base_w + sign * delta))
        remainder = 1.0 - new_target
        perturbed = {}
        for k in names:
            if k == target:
                perturbed[k] = new_target
            else:
                if other_sum > 0:
                    perturbed[k] = weights[k] / other_sum * remainder
                else:
                    perturbed[k] = remainder / len(others)
        results.append(perturbed)

    return results[0], results[1]


def _oat(criteria, weights, combine_fn, delta, name):
    """One-at-a-time sensitivity: perturb each weight +/- delta."""
    result_vars = {}

    for criterion in weights:
        w_plus, w_minus = _perturb_weights(weights, criterion, delta)
        score_plus = combine_fn(criteria, w_plus)
        score_minus = combine_fn(criteria, w_minus)
        diff = abs(score_plus - score_minus)
        diff.name = f"{name}_{criterion}"
        result_vars[criterion] = diff

    return xr.Dataset(result_vars)


def _is_dask_dataset(criteria):
    """Check if any variable in the Dataset is backed by dask."""
    try:
        import dask.array as da
        return any(
            isinstance(criteria[v].data, da.Array)
            for v in criteria.data_vars
        )
    except ImportError:
        return False


def _mc_block(block, weight_matrix, combine_method):
    """Run the Monte Carlo sample loop on one in-memory block.

    ``block`` has shape ``(n_criteria,) + grid`` with weight columns in
    the same criterion order. Welford's online algorithm keeps only the
    running mean and M2 buffers alive, so peak memory per block is a
    few times the block size regardless of ``n_samples``. Works on
    numpy and cupy blocks alike.
    """
    xp = _get_xp(block)
    n_samples = weight_matrix.shape[0]
    n_criteria = block.shape[0]
    shape = block.shape[1:]
    running_mean = xp.zeros(shape, dtype=np.float64)
    running_m2 = xp.zeros(shape, dtype=np.float64)

    for i in range(n_samples):
        w = weight_matrix[i]
        # Same per-pixel arithmetic as combine.wlc / combine.wpm (the
        # canonical definitions), inlined on raw arrays so the loop
        # runs inside a single dask chunk. Keep in sync with those.
        if combine_method == "wpm":
            score = block[0] ** float(w[0])
            for j in range(1, n_criteria):
                score = score * block[j] ** float(w[j])
        else:
            score = block[0] * float(w[0])
            for j in range(1, n_criteria):
                score = score + block[j] * float(w[j])

        delta_val = score - running_mean
        running_mean = running_mean + delta_val / (i + 1)
        delta2 = score - running_mean
        running_m2 = running_m2 + delta_val * delta2

    variance = running_m2 / n_samples
    std_dev = xp.sqrt(variance)
    # Coefficient of variation
    with np.errstate(divide="ignore", invalid="ignore"):
        return xp.where(
            running_mean != 0, std_dev / xp.abs(running_mean), 0.0
        )


def _monte_carlo(criteria, weights, combine_method, n_samples, seed, name):
    """Monte Carlo sensitivity: random weight vectors, measure CV."""
    rng = np.random.default_rng(seed)
    n_criteria = len(weights)
    criteria_names = list(weights.keys())

    # Generate random weight vectors using Dirichlet distribution
    # Use base weights as concentration parameters (scaled up)
    alpha = np.array([weights[k] for k in criteria_names]) * n_criteria * 5
    alpha = np.maximum(alpha, 0.5)  # floor to avoid zero concentrations

    # Draw sequentially so a given seed reproduces the sample stream of
    # the previous per-iteration implementation.
    weight_matrix = np.stack(
        [rng.dirichlet(alpha) for _ in range(n_samples)]
    )

    # Validate once up front with the first sampled vector (it is
    # finite and sums to 1, so only key-set mismatches can fire) the
    # way the first combine call used to.
    _validate_criteria(criteria)
    probe = {k: float(weight_matrix[0][j])
             for j, k in enumerate(criteria_names)}
    _validate_weights(probe, criteria)
    if combine_method == "wpm":
        _check_wpm_positive(criteria)

    first_var = list(criteria.data_vars)[0]
    template = criteria[first_var]

    # Stack criterion layers in data_vars order (the order wlc/wpm
    # combine in) and permute the weight columns to match.
    var_order = list(criteria.data_vars)
    col = {k: j for j, k in enumerate(criteria_names)}
    ordered_weights = weight_matrix[:, [col[v] for v in var_order]]

    if _is_dask_dataset(criteria):
        import dask.array as da

        base_chunks = next(
            criteria[v].data.chunks for v in var_order
            if isinstance(criteria[v].data, da.Array)
        )
        layers = []
        for v in var_order:
            arr = criteria[v].data
            if isinstance(arr, da.Array):
                arr = arr.rechunk(base_chunks)
            else:
                arr = da.from_array(arr, chunks=base_chunks)
            layers.append(arr)
        # The sample loop runs inside one chunk function, so the graph
        # stays at one task per chunk and nothing materializes the full
        # criteria stack.
        stacked = da.stack(layers).rechunk({0: -1})
        cv_vals = da.map_blocks(
            _mc_block, stacked,
            weight_matrix=ordered_weights, combine_method=combine_method,
            drop_axis=0, dtype=np.float64,
            meta=stacked._meta.sum(axis=0),
        )
    else:
        stacked = np.stack([criteria[v].data for v in var_order])
        cv_vals = _mc_block(stacked, ordered_weights, combine_method)

    result = xr.DataArray(
        cv_vals, name=name, dims=template.dims,
        coords=template.coords, attrs=template.attrs,
    )
    return result
