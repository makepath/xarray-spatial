"""Extract interpolation point inputs from geospatial vector data.

The interpolation functions (``idw``/``kriging``/``spline`` and ``kde``)
historically only accept raw ``x, y, z`` array-likes.  These helpers let
them also accept a GeoDataFrame of Point geometries, mirroring how
``rasterize`` reads a burn value from a ``column``.  geopandas stays an
optional dependency: it is imported lazily and a guarded ImportError keeps
the array path working without it.
"""

from __future__ import annotations

import numpy as np


def is_geodataframe(obj):
    """True if *obj* is a geopandas GeoDataFrame (geopandas optional)."""
    try:
        import geopandas as gpd
    except ImportError:
        return False
    return isinstance(obj, gpd.GeoDataFrame)


def _resolve_value_column(gdf, *, column, value_required, func_name):
    """Resolve the per-point value array from a GeoDataFrame.

    Mirrors ``rasterize._parse_input``: an explicit *column* wins, else the
    first numeric column is used when a value is required, else ``None``.
    """
    if column is not None:
        if column not in gdf.columns:
            raise ValueError(
                f"{func_name}(): column {column!r} is not in the GeoDataFrame"
            )
        try:
            return np.asarray(gdf[column].values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"{func_name}(): column {column!r} is not numeric"
            ) from e

    if not value_required:
        return None

    numeric_cols = gdf.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        raise ValueError(
            f"{func_name}(): GeoDataFrame has no numeric column to "
            f"interpolate. Pass a 'column' name explicitly."
        )
    return np.asarray(gdf[numeric_cols[0]].values, dtype=np.float64)


def points_from_geodataframe(gdf, *, column, value_required, func_name):
    """Return ``(x, y, value, crs)`` from a GeoDataFrame of Point geometries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input points.  Only single-part ``Point`` geometries are accepted;
        any other geometry type raises (interpolation is point-based, unlike
        ``rasterize`` which fills polygons and lines).
    column : str or None
        Column holding the per-point value.  When ``None`` and
        *value_required*, the first numeric column is used; when ``None`` and
        not required, *value* is ``None``.
    value_required : bool
        Whether a per-point value must be resolved (``z`` for
        idw/kriging/spline; optional weights for kde).
    func_name : str
        Caller name, used in error messages.

    Returns
    -------
    x, y : numpy.ndarray
        Float64 point coordinates.
    value : numpy.ndarray or None
        Per-point values, or ``None`` when not required and no column given.
    crs : object or None
        ``gdf.crs``, passed through as-is.
    """
    geom = gdf.geometry
    geom_types = set(geom.geom_type.dropna().unique())
    other = geom_types - {'Point'}
    if other:
        raise ValueError(
            f"{func_name}(): interpolation needs single Point geometries, "
            f"got {sorted(other)}"
        )

    x = np.asarray(geom.x.values, dtype=np.float64)
    y = np.asarray(geom.y.values, dtype=np.float64)
    value = _resolve_value_column(
        gdf, column=column, value_required=value_required, func_name=func_name
    )
    return x, y, value, gdf.crs


def resolve_xyz(x, y, z, *, column, func_name):
    """Resolve ``(x, y, z)`` for the value-interpolation functions.

    If *x* is a GeoDataFrame, pull coordinates and the value column from it
    (*y*/*z* must be unset); otherwise pass the arrays through unchanged.
    ``column`` is only meaningful for the GeoDataFrame form.
    """
    if is_geodataframe(x):
        if y is not None or z is not None:
            raise ValueError(
                f"{func_name}(): when the first argument is a GeoDataFrame, "
                f"pass template and column as keywords, not positional y/z"
            )
        gx, gy, value, _crs = points_from_geodataframe(
            x, column=column, value_required=True, func_name=func_name
        )
        return gx, gy, value
    if column is not None:
        raise ValueError(
            f"{func_name}(): 'column' is only valid when the first argument "
            f"is a GeoDataFrame"
        )
    return x, y, z
