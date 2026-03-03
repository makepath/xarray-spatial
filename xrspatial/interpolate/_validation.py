"""Shared input validation for interpolation functions."""

from __future__ import annotations

import numpy as np

from xrspatial.utils import _validate_raster


def validate_points(x, y, z, *, func_name):
    """Coerce *x*, *y*, *z* to float64 1-D arrays and drop NaN rows.

    Returns
    -------
    x, y, z : numpy.ndarray
        Cleaned float64 arrays with NaN/inf rows removed.

    Raises
    ------
    ValueError
        If lengths don't match or no valid points remain.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()

    if not (x.shape[0] == y.shape[0] == z.shape[0]):
        raise ValueError(
            f"{func_name}(): x, y, z must have the same length, "
            f"got {x.shape[0]}, {y.shape[0]}, {z.shape[0]}"
        )

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]

    if len(x) == 0:
        raise ValueError(
            f"{func_name}(): no valid (non-NaN) points remain after filtering"
        )

    return x, y, z


def extract_grid_coords(template, *, func_name):
    """Extract 1-D coordinate vectors from a 2-D template DataArray.

    Returns
    -------
    x_grid, y_grid : numpy.ndarray
        Float64 coordinate arrays for the last two dims.
    """
    _validate_raster(template, func_name=func_name, name='template')

    y_dim = template.dims[-2]
    x_dim = template.dims[-1]

    y_grid = np.asarray(template.coords[y_dim].values, dtype=np.float64)
    x_grid = np.asarray(template.coords[x_dim].values, dtype=np.float64)

    return x_grid, y_grid
