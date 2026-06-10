"""Constraint masking for MCDA suitability surfaces."""

from __future__ import annotations

import numpy as np
import xarray as xr


def constrain(
    suitability: xr.DataArray,
    exclude: list[xr.DataArray],
    fill: float = np.nan,
    name: str | None = None,
) -> xr.DataArray:
    """Mask out areas that are categorically unsuitable.

    Pixels where any exclusion mask is True (nonzero) are set to *fill*.

    Parameters
    ----------
    suitability : xr.DataArray
        Input suitability surface.
    exclude : list of xr.DataArray
        Boolean or binary masks. True/nonzero marks excluded areas.
    fill : float
        Value to assign to excluded pixels. Default ``np.nan``.
    name : str, optional
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        Constrained suitability surface. Keeps the input's dims,
        coords, and attrs (``res``, ``crs``, ``nodatavals``, ...).
    """
    if name is None:
        name = suitability.name

    # Shallow copy: xr.where returns new objects and nothing below
    # mutates the data, so a deep copy would only duplicate the full
    # raster. The copy still keeps the .name assignment from leaking
    # into the caller's object when exclude is empty.
    result = suitability.copy(deep=False)
    for mask in exclude:
        result = xr.where(mask, fill, result)

    # xr.where takes attrs from its first value argument (the scalar
    # ``fill``), which strips res/crs/nodatavals from the output.
    # Restore the input's attrs so the constrained surface stays
    # georeferenced (#3147).
    result.attrs = dict(suitability.attrs)
    result.name = name
    return result
