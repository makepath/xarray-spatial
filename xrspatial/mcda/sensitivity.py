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
    baseline = combine_fn(criteria, weights)
    result_vars = {}

    for criterion in weights:
        w_plus, w_minus = _perturb_weights(weights, criterion, delta)
        score_plus = combine_fn(criteria, w_plus)
        score_minus = combine_fn(criteria, w_minus)
        diff = abs(score_plus - score_minus)
        diff.name = f"{name}_{criterion}"
        result_vars[criterion] = diff

    return xr.Dataset(result_vars)


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

    # Accumulate running mean and M2 for Welford's online algorithm
    # to avoid storing all n_samples surfaces in memory
    running_mean = template * 0.0
    running_m2 = template * 0.0

    for i in range(n_samples):
        raw = rng.dirichlet(alpha)
        sample_weights = {
            k: float(raw[j]) for j, k in enumerate(criteria_names)
        }
        score = combine_fn(criteria, sample_weights)
        delta_val = score - running_mean
        running_mean = running_mean + delta_val / (i + 1)
        delta2 = score - running_mean
        running_m2 = running_m2 + delta_val * delta2

    variance = running_m2 / n_samples
    std_dev = variance ** 0.5
    # Coefficient of variation
    cv = xr.where(running_mean != 0, std_dev / abs(running_mean), 0.0)
    cv.name = name
    return cv
