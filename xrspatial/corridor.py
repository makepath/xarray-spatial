"""Least-cost corridor analysis.

Identifies zones of low cumulative traversal cost between two (or more)
source locations on a friction surface.  Unlike a single-cell path, a
corridor shows all cells within a cost threshold of the optimal route.

Algorithm
---------
1. Compute ``cost_distance(friction, source_a)`` and
   ``cost_distance(friction, source_b)``.
2. Sum the two surfaces: ``corridor = cd_a + cd_b``.
3. Normalize: ``corridor - corridor.min()``.
4. Optionally threshold to produce a binary corridor mask.

Because the function is pure arithmetic on ``cost_distance`` outputs,
all four backends (NumPy, CuPy, Dask+NumPy, Dask+CuPy) are supported
natively with no extra kernel code.
"""

from __future__ import annotations

import itertools

import numpy as np
import xarray as xr

from xrspatial.cost_distance import cost_distance
from xrspatial.utils import _validate_raster


def _scalar_to_float(da_scalar):
    """Extract a Python float from a scalar DataArray (numpy/cupy/dask)."""
    data = da_scalar.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'get'):
        data = data.get()
    return float(data)


def _compute_corridor(
    cd_a: xr.DataArray,
    cd_b: xr.DataArray,
    threshold: float | None,
    relative: bool,
) -> xr.DataArray:
    """Sum two cost-distance surfaces, normalize, and optionally threshold."""
    corridor = cd_a + cd_b
    corridor_min = _scalar_to_float(corridor.min())

    if not np.isfinite(corridor_min):
        # Sources are mutually unreachable -- return all-NaN.
        return corridor * np.nan

    normalized = corridor - corridor_min

    if threshold is not None:
        if relative:
            cutoff = threshold * corridor_min
        else:
            cutoff = threshold
        data = normalized.data
        try:
            import dask.array as _da
            if isinstance(data, _da.Array):
                data = _da.where(data <= cutoff, data, np.nan)
            else:
                raise ImportError
        except ImportError:
            if hasattr(data, 'get'):  # cupy
                import cupy as cp
                data = cp.where(data <= cutoff, data, cp.nan)
            else:
                data = np.where(data <= cutoff, data, np.nan)
        normalized = normalized.copy(data=data)

    return normalized


def least_cost_corridor(
    friction: xr.DataArray,
    source_a: xr.DataArray | None = None,
    source_b: xr.DataArray | None = None,
    *,
    sources: list[xr.DataArray] | None = None,
    pairwise: bool = False,
    threshold: float | None = None,
    relative: bool = False,
    precomputed: bool = False,
    x: str = "x",
    y: str = "y",
    connectivity: int = 8,
    max_cost: float = np.inf,
) -> xr.DataArray | xr.Dataset:
    """Compute a least-cost corridor between two source regions.

    A corridor surface is the sum of two cost-distance fields (one from
    each source), normalized by the minimum corridor cost.  Cells where
    the normalized value falls within *threshold* form the corridor.

    Parameters
    ----------
    friction : xr.DataArray
        2-D friction (cost) surface.  Positive finite values are
        passable; NaN or ``<= 0`` marks barriers.  Ignored when
        *precomputed* is True.
    source_a, source_b : xr.DataArray, optional
        Source rasters identifying start and end regions.  Non-zero
        finite values mark source cells (same convention as
        ``cost_distance``).  Required unless *sources* is given.
        When *precomputed* is True these are treated as pre-computed
        cost-distance surfaces.
    sources : list of xr.DataArray, optional
        Multiple source rasters for pairwise corridor computation.
        Mutually exclusive with *source_a* / *source_b*.
    pairwise : bool, default False
        When True and *sources* has more than two entries, compute
        corridors for every unique pair and return an ``xr.Dataset``
        with one variable per pair (named ``corridor_i_j``).
    threshold : float, optional
        Cost threshold for masking the corridor.  Cells whose
        normalized corridor cost exceeds this value are set to NaN.
    relative : bool, default False
        If True, *threshold* is a fraction of the minimum corridor
        cost (e.g. 0.10 means within 10 % of optimal).  If False,
        *threshold* is in absolute cost units.
    precomputed : bool, default False
        If True, *source_a* / *source_b* (or entries in *sources*)
        are already cost-distance surfaces.  Skips the internal
        ``cost_distance`` calls.
    x, y : str
        Coordinate names forwarded to ``cost_distance``.
    connectivity : int, default 8
        Pixel connectivity (4 or 8), forwarded to ``cost_distance``.
    max_cost : float, default np.inf
        Maximum cost budget, forwarded to ``cost_distance``.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Normalized corridor surface (float).  Values start at 0 along
        the optimal corridor and increase with deviation from it.
        When *threshold* is set, cells beyond the cutoff are NaN.
        With *pairwise=True* and multiple sources, returns a Dataset
        keyed by pair indices (``corridor_0_1``, ``corridor_0_2``, ...).
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if sources is not None and (source_a is not None or source_b is not None):
        raise ValueError(
            "Provide either (source_a, source_b) or sources, not both"
        )

    if sources is not None:
        if len(sources) < 2:
            raise ValueError("sources must contain at least 2 entries")
        for i, s in enumerate(sources):
            _validate_raster(
                s, func_name="least_cost_corridor", name=f"sources[{i}]"
            )
    else:
        if source_a is None or source_b is None:
            raise ValueError(
                "source_a and source_b are required when sources is not given"
            )
        _validate_raster(
            source_a, func_name="least_cost_corridor", name="source_a"
        )
        _validate_raster(
            source_b, func_name="least_cost_corridor", name="source_b"
        )
        sources = [source_a, source_b]

    if not precomputed:
        _validate_raster(
            friction, func_name="least_cost_corridor", name="friction"
        )

    if threshold is not None and threshold < 0:
        raise ValueError("threshold must be non-negative")

    # ------------------------------------------------------------------
    # Compute cost-distance surfaces (or use precomputed ones)
    # ------------------------------------------------------------------
    if precomputed:
        cd_surfaces = list(sources)
    else:
        cd_surfaces = [
            cost_distance(
                src, friction,
                x=x, y=y, connectivity=connectivity, max_cost=max_cost,
            )
            for src in sources
        ]

    # ------------------------------------------------------------------
    # Two-source case: return a single DataArray
    # ------------------------------------------------------------------
    if len(cd_surfaces) == 2 and not pairwise:
        return _compute_corridor(
            cd_surfaces[0], cd_surfaces[1], threshold, relative
        )

    # ------------------------------------------------------------------
    # Multi-source pairwise: return a Dataset
    # ------------------------------------------------------------------
    corridors = {}
    for i, j in itertools.combinations(range(len(cd_surfaces)), 2):
        name = f"corridor_{i}_{j}"
        corridors[name] = _compute_corridor(
            cd_surfaces[i], cd_surfaces[j], threshold, relative
        )

    return xr.Dataset(corridors)
