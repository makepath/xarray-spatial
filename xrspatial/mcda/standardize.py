"""Value-function standardization for MCDA criterion layers.

Converts raw criterion rasters from native units to a common 0-1
suitability scale using configurable value functions.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def standardize(
    agg: xr.DataArray,
    method: str = "linear",
    *,
    direction: str = "higher_is_better",
    bounds: tuple[float, float] | None = None,
    midpoint: float = 0.0,
    spread: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    low: float = 0.0,
    peak: float = 0.5,
    high: float = 1.0,
    breakpoints: list[float] | None = None,
    values: list[float] | None = None,
    mapping: dict | None = None,
    name: str | None = None,
) -> xr.DataArray:
    """Standardize a criterion raster to 0-1 suitability scale.

    Parameters
    ----------
    agg : xr.DataArray
        Input criterion raster (numpy, cupy, dask+numpy, or dask+cupy).
    method : str
        Value function type. One of ``"linear"``, ``"sigmoidal"``,
        ``"gaussian"``, ``"triangular"``, ``"piecewise"``, or
        ``"categorical"``.
    direction : str
        For monotonic methods (``"linear"``): ``"higher_is_better"`` or
        ``"lower_is_better"``.
    bounds : tuple of float, optional
        ``(low, high)`` for linear scaling. Values outside bounds are
        clipped. If *None*, uses data min/max.
    midpoint : float
        Inflection point for sigmoidal method.
    spread : float
        Steepness for sigmoidal method. Negative values produce a
        decreasing sigmoid.
    mean : float
        Center of the bell curve for gaussian method.
    std : float
        Width of the bell curve for gaussian method. Must be > 0.
    low, peak, high : float
        Vertices of the triangle for triangular method.
    breakpoints : list of float, optional
        X-coordinates for piecewise linear interpolation.
    values : list of float, optional
        Corresponding suitability values for each breakpoint.
    mapping : dict, optional
        ``{class_value: suitability}`` lookup for categorical method.
    name : str, optional
        Name of the output DataArray. Defaults to the input name.

    Returns
    -------
    xr.DataArray
        Standardized raster with values in [0, 1]. NaN values pass through.
    """
    if name is None:
        name = agg.name

    _methods = {
        "linear": _linear,
        "sigmoidal": _sigmoidal,
        "gaussian": _gaussian,
        "triangular": _triangular,
        "piecewise": _piecewise,
        "categorical": _categorical,
    }
    if method not in _methods:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {list(_methods)}"
        )

    func = _methods[method]

    if method == "linear":
        data = func(agg.data, direction=direction, bounds=bounds)
    elif method == "sigmoidal":
        data = func(agg.data, midpoint=midpoint, spread=spread)
    elif method == "gaussian":
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        data = func(agg.data, mean=mean, std=std)
    elif method == "triangular":
        if not (low <= peak <= high):
            raise ValueError(
                f"Need low <= peak <= high, got {low}, {peak}, {high}"
            )
        data = func(agg.data, low=low, peak=peak, high=high)
    elif method == "piecewise":
        if breakpoints is None or values is None:
            raise ValueError(
                "piecewise method requires breakpoints and values"
            )
        if len(breakpoints) != len(values):
            raise ValueError(
                "breakpoints and values must have the same length"
            )
        if len(breakpoints) < 2:
            raise ValueError("piecewise requires at least 2 breakpoints")
        data = func(agg.data, breakpoints=breakpoints, values=values)
    elif method == "categorical":
        if mapping is None:
            raise ValueError("categorical method requires mapping dict")
        data = func(agg.data, mapping=mapping)

    return xr.DataArray(
        data, name=name, dims=agg.dims, coords=agg.coords, attrs=agg.attrs,
    )


def _get_xp(data):
    """Return the array module (numpy or cupy) for the underlying data."""
    try:
        import cupy
        if isinstance(data, cupy.ndarray):
            return cupy
    except ImportError:
        pass

    try:
        import dask.array as da
        if isinstance(data, da.Array):
            # Check if chunks are cupy
            meta = getattr(data, '_meta', None)
            if meta is not None:
                try:
                    import cupy
                    if isinstance(meta, cupy.ndarray):
                        return cupy
                except ImportError:
                    pass
    except ImportError:
        pass

    return np


def _linear(data, *, direction, bounds):
    xp = _get_xp(data)

    if bounds is not None:
        lo, hi = float(bounds[0]), float(bounds[1])
    else:
        # Compute from data, ignoring NaN
        try:
            import dask.array as da
            if isinstance(data, da.Array):
                lo = float(da.nanmin(data))
                hi = float(da.nanmax(data))
            else:
                raise ImportError
        except (ImportError, TypeError):
            lo = float(np.nanmin(data))
            hi = float(np.nanmax(data))

    rng = hi - lo
    if rng == 0:
        # Constant surface -- return 0.5
        out = xp.where(xp.isfinite(data), 0.5, xp.nan)
        return out

    # Clip and scale to 0-1
    clipped = xp.clip(data, lo, hi)
    result = (clipped - lo) / rng

    if direction == "lower_is_better":
        result = 1.0 - result
    elif direction != "higher_is_better":
        raise ValueError(
            f"direction must be 'higher_is_better' or 'lower_is_better', "
            f"got {direction!r}"
        )

    # Preserve NaN
    result = xp.where(xp.isfinite(data), result, xp.nan)
    return result


def _sigmoidal(data, *, midpoint, spread):
    xp = _get_xp(data)
    exponent = -spread * (data - midpoint)
    result = 1.0 / (1.0 + xp.exp(exponent))
    result = xp.where(xp.isfinite(data), result, xp.nan)
    return result


def _gaussian(data, *, mean, std):
    xp = _get_xp(data)
    result = xp.exp(-0.5 * ((data - mean) / std) ** 2)
    result = xp.where(xp.isfinite(data), result, xp.nan)
    return result


def _triangular(data, *, low, peak, high):
    xp = _get_xp(data)
    # Rising edge: low to peak
    # Falling edge: peak to high
    # Outside: 0
    rising = (data - low) / (peak - low) if peak != low else 0.0
    falling = (high - data) / (high - peak) if high != peak else 0.0

    result = xp.minimum(rising, falling)
    result = xp.clip(result, 0.0, 1.0)
    result = xp.where(xp.isfinite(data), result, xp.nan)
    return result


def _piecewise(data, *, breakpoints, values):
    xp = _get_xp(data)
    breakpoints = np.asarray(breakpoints, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    # Sort by breakpoint
    order = np.argsort(breakpoints)
    breakpoints = breakpoints[order]
    values = values[order]

    # xp.interp works for numpy and cupy; for dask we use map_blocks
    try:
        import dask.array as da
        if isinstance(data, da.Array):
            result = da.map_blocks(
                lambda block: np.interp(block, breakpoints, values),
                data,
                dtype=np.float64,
            )
            result = da.where(da.isfinite(data), result, np.nan)
            return result
    except ImportError:
        pass

    result = xp.interp(data, breakpoints, values)
    result = xp.where(xp.isfinite(data), result, xp.nan)
    return result


def _categorical(data, *, mapping):
    xp = _get_xp(data)

    # Build output: default NaN, fill from mapping
    keys = np.array(list(mapping.keys()), dtype=np.float64)
    vals = np.array(list(mapping.values()), dtype=np.float64)

    try:
        import dask.array as da
        if isinstance(data, da.Array):
            def _apply_mapping(block):
                out = np.full(block.shape, np.nan, dtype=np.float64)
                for k, v in zip(keys, vals):
                    out[block == k] = v
                return out
            result = da.map_blocks(_apply_mapping, data, dtype=np.float64)
            return result
    except ImportError:
        pass

    out = xp.full(data.shape, xp.nan, dtype=np.float64)
    for k, v in zip(keys, vals):
        out[data == k] = v
    return out
