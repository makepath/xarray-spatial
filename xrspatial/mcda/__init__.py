"""Multi-Criteria Decision Analysis (MCDA) for raster suitability modeling.

Provides four stages of spatial MCDA:

1. **Standardize** -- convert criterion layers to a common 0-1 scale
2. **Weight** -- derive criterion weights (AHP, rank-order, or direct)
3. **Combine** -- merge layers into a composite suitability surface
4. **Validate** -- constrain exclusion zones and check weight stability
"""

from xrspatial.mcda.standardize import standardize
from xrspatial.mcda.weights import ahp_weights, rank_weights
from xrspatial.mcda.combine import (
    boolean_overlay,
    fuzzy_overlay,
    owa,
    wlc,
    wpm,
)
from xrspatial.mcda.constrain import constrain
from xrspatial.mcda.sensitivity import sensitivity

__all__ = [
    "standardize",
    "ahp_weights",
    "rank_weights",
    "wlc",
    "wpm",
    "owa",
    "fuzzy_overlay",
    "boolean_overlay",
    "constrain",
    "sensitivity",
]
