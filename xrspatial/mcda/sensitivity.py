"""Sensitivity analysis for MCDA weight stability."""

from __future__ import annotations

import numpy as np
import xarray as xr

from xrspatial.mcda.combine import wlc, wpm


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
        variation across random weight samples.
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
        return _monte_carlo(
            criteria, weights, combine_fn, n_samples, seed, name
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


def _monte_carlo(criteria, weights, combine_fn, n_samples, seed, name):
    """Monte Carlo sensitivity: random weight vectors, measure CV."""
    rng = np.random.default_rng(seed)
    n_criteria = len(weights)
    criteria_names = list(weights.keys())

    # Generate random weight vectors using Dirichlet distribution
    # Use base weights as concentration parameters (scaled up)
    alpha = np.array([weights[k] for k in criteria_names]) * n_criteria * 5
    alpha = np.maximum(alpha, 0.5)  # floor to avoid zero concentrations

    first_var = list(criteria.data_vars)[0]
    template = criteria[first_var]

    use_dask = _is_dask_dataset(criteria)

    # Accumulate running mean and M2 for Welford's online algorithm
    # to avoid storing all n_samples surfaces in memory
    running_mean = template.values * 0.0 if not use_dask else np.zeros(template.shape)
    running_m2 = running_mean.copy()

    # For dask inputs: compute each score eagerly to avoid graph
    # explosion (each lazy iteration would chain ~35 graph nodes;
    # 1000 iterations = 35000 chained tasks with no parallelism).
    # The criterion layers are small enough per-pixel that the
    # bottleneck is the combination, not I/O.
    if use_dask:
        criteria_np = criteria.compute()
    else:
        criteria_np = criteria

    for i in range(n_samples):
        raw = rng.dirichlet(alpha)
        sample_weights = {
            k: float(raw[j]) for j, k in enumerate(criteria_names)
        }
        score = combine_fn(criteria_np, sample_weights)
        score_vals = score.values

        delta_val = score_vals - running_mean
        running_mean = running_mean + delta_val / (i + 1)
        delta2 = score_vals - running_mean
        running_m2 = running_m2 + delta_val * delta2

    variance = running_m2 / n_samples
    std_dev = np.sqrt(variance)
    # Coefficient of variation
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_vals = np.where(running_mean != 0, std_dev / np.abs(running_mean), 0.0)

    result = xr.DataArray(
        cv_vals, name=name, dims=template.dims,
        coords=template.coords, attrs=template.attrs,
    )
    return result
