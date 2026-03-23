"""Weight derivation methods for MCDA.

Provides AHP (Analytical Hierarchy Process) and rank-order weighting.
These operate on small metadata (criteria names and comparisons), not
on raster data, so they always use numpy regardless of backend.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


# Random consistency index (Saaty) for matrices of size 1..15
_RI = [0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49,
       1.51, 1.48, 1.56, 1.57, 1.59]


@dataclass
class ConsistencyResult:
    """AHP consistency check results."""
    ratio: float
    index: float
    is_consistent: bool
    lambda_max: float


def ahp_weights(
    criteria: list[str],
    comparisons: dict[tuple[str, str], float],
) -> tuple[dict[str, float], ConsistencyResult]:
    """Derive criterion weights from pairwise comparisons using AHP.

    Uses the standard Saaty eigenvector method. Input pairwise
    comparisons on a 1-9 scale (or reciprocals). The function builds
    the full comparison matrix, computes the principal eigenvector for
    weights, and derives the consistency ratio.

    Parameters
    ----------
    criteria : list of str
        Criterion names in order.
    comparisons : dict
        Pairwise comparisons as ``{(criterion_a, criterion_b): value}``.
        Only provide each pair once; the reciprocal is inferred.
        Values follow the Saaty scale (1 = equal, 9 = extreme preference
        of a over b, 1/9 = extreme preference of b over a).

    Returns
    -------
    weights : dict of str to float
        Normalized weights summing to 1.0.
    consistency : ConsistencyResult
        Consistency ratio and related metrics. ``is_consistent`` is
        True when ratio < 0.10.

    Raises
    ------
    ValueError
        If criteria list has fewer than 2 items or comparisons are
        incomplete.
    """
    n = len(criteria)
    if n < 2:
        raise ValueError("Need at least 2 criteria")

    expected = n * (n - 1) // 2
    if len(comparisons) < expected:
        warnings.warn(
            f"Only {len(comparisons)} of {expected} pairwise comparisons "
            f"provided for {n} criteria. Missing pairs default to 1 "
            f"(equal importance).",
            UserWarning,
            stacklevel=2,
        )

    idx = {name: i for i, name in enumerate(criteria)}
    matrix = np.ones((n, n), dtype=np.float64)

    for (a, b), val in comparisons.items():
        if a not in idx:
            raise ValueError(f"Unknown criterion {a!r}")
        if b not in idx:
            raise ValueError(f"Unknown criterion {b!r}")
        if a == b:
            raise ValueError(
                f"Self-comparison ({a!r}, {b!r}) is not allowed; "
                f"diagonal entries are always 1"
            )
        if val <= 0:
            raise ValueError(
                f"Comparison value must be positive, got {val} "
                f"for ({a!r}, {b!r})"
            )
        i, j = idx[a], idx[b]
        matrix[i, j] = val
        matrix[j, i] = 1.0 / val

    # Principal eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # Find the largest real eigenvalue
    real_parts = eigenvalues.real
    max_idx = np.argmax(real_parts)
    lambda_max = float(real_parts[max_idx])

    # Corresponding eigenvector (take real part)
    raw_weights = eigenvectors[:, max_idx].real
    raw_weights = np.abs(raw_weights)
    normalized = raw_weights / raw_weights.sum()

    # Consistency
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = _RI[n - 1] if n <= len(_RI) else _RI[-1]
    cr = ci / ri if ri > 0 else 0.0

    weights = {name: float(normalized[i]) for i, name in enumerate(criteria)}
    consistency = ConsistencyResult(
        ratio=cr,
        index=ci,
        is_consistent=(cr < 0.10),
        lambda_max=lambda_max,
    )
    return weights, consistency


def rank_weights(
    ranking: list[str],
    method: str = "roc",
) -> dict[str, float]:
    """Derive weights from a rank ordering of criteria.

    Parameters
    ----------
    ranking : list of str
        Criteria ordered from most to least important.
    method : str
        Weighting scheme: ``"roc"`` (rank-order centroid),
        ``"rs"`` (rank sum), or ``"rr"`` (reciprocal of ranks).

    Returns
    -------
    weights : dict of str to float
        Normalized weights summing to 1.0.
    """
    n = len(ranking)
    if n < 1:
        raise ValueError("Need at least 1 criterion in ranking")

    if method == "roc":
        # ROC: w_i = (1/n) * sum(1/k for k in range(i+1, n+1))
        raw = np.array([
            sum(1.0 / k for k in range(i + 1, n + 1)) / n
            for i in range(n)
        ])
    elif method == "rs":
        # Rank sum: w_i = (n - rank + 1) / sum
        raw = np.array([n - i for i in range(n)], dtype=np.float64)
    elif method == "rr":
        # Reciprocal of rank: w_i = (1/rank) / sum
        raw = np.array([1.0 / (i + 1) for i in range(n)])
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose from 'roc', 'rs', 'rr'"
        )

    normalized = raw / raw.sum()
    return {name: float(normalized[i]) for i, name in enumerate(ranking)}
