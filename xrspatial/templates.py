"""Empty study-area DataArrays for common regions.

:func:`from_template` turns a region name into an empty (NaN-filled) raster you
can start an analysis from, instead of looking up a bounding box, choosing a
CRS, and assembling coordinates by hand. The result follows the xarray-spatial
array contract (2-D ``['y', 'x']`` grid with pixel-center coordinates and
``res``/``crs`` attributes), so it feeds straight into ``slope``, ``hillshade``,
``rasterize``, and the rest of the library.
"""
import numpy as np
import xarray as xr

from xrspatial._template_data import (
    _COUNTRY_BBOXES,
    _COUNTRY_DEFAULT_RESOLUTION,
    _REGIONS,
)
from xrspatial.reproject._grid import _make_output_coords

# Guard against a stray fine resolution allocating an enormous array. Applied to
# every backend, including dask: a lazy grid this large is almost always a typo,
# and the uniform cap keeps the error identical across backends.
_MAX_CELLS = 500_000_000


def _resolve(name):
    """Resolve a name to (bounds, crs, default_resolution, key)."""
    if not isinstance(name, str):
        raise TypeError(
            f"name must be a string region name or country code, "
            f"got {type(name).__name__}"
        )

    region = _REGIONS.get(name.lower())
    if region is not None:
        return (region["bounds"], region["crs"],
                region["default_resolution"], name.lower())

    code = name.upper()
    bbox = _COUNTRY_BBOXES.get(code)
    if bbox is not None:
        return bbox, 4326, _COUNTRY_DEFAULT_RESOLUTION, code

    regions = ", ".join(sorted(_REGIONS))
    raise ValueError(
        f"Unknown template {name!r}. Available named regions: {regions}. "
        f"Countries must be an ISO-3166 / GADM alpha-3 code "
        f"(e.g. 'USA', 'FRA', 'JPN'); {len(_COUNTRY_BBOXES)} are supported."
    )


def _normalize_resolution(resolution, default):
    """Return a positive (res_x, res_y) tuple from a scalar/tuple/None."""
    if resolution is None:
        resolution = default
    if isinstance(resolution, (tuple, list)):
        if len(resolution) != 2:
            raise ValueError(
                "resolution tuple must be (res_x, res_y), "
                f"got {len(resolution)} values"
            )
        res_x, res_y = resolution
    else:
        res_x = res_y = resolution
    res_x, res_y = float(res_x), float(res_y)
    if res_x <= 0 or res_y <= 0:
        raise ValueError(f"resolution must be positive, got {(res_x, res_y)}")
    return res_x, res_y


def _make_data(shape, fill, backend, chunks):
    """Build a NaN-filled array of ``shape`` for the requested backend."""
    backend = backend.lower()
    if backend in ("dask", "dask+numpy"):
        import dask.array as da
        return da.full(shape, fill, dtype="float32", chunks=chunks)
    if backend == "numpy":
        return np.full(shape, fill, dtype="float32")
    if backend == "cupy":
        import cupy
        return cupy.full(shape, fill, dtype="float32")
    if backend == "dask+cupy":
        import cupy
        import dask.array as da
        meta = cupy.empty((0, 0), dtype="float32")
        return da.full(shape, fill, dtype="float32",
                       chunks=chunks).map_blocks(cupy.asarray, meta=meta)
    raise ValueError(
        f"Unknown backend {backend!r}; choose from 'numpy', 'dask+numpy', "
        f"'cupy', 'dask+cupy'."
    )


def from_template(name, resolution=None, *, backend="numpy", fill=np.nan,
                  chunks="auto"):
    """Create an empty DataArray for a common study area.

    The returned raster is NaN-filled and obeys the xarray-spatial array
    contract: a 2-D ``['y', 'x']`` grid with pixel-center 1-D coordinates
    (north-up, descending ``y``) and ``res``/``crs`` attributes. It covers the
    study area's rectangular bounding box and is meant as a starting canvas.

    Parameters
    ----------
    name : str
        A curated region name (case-insensitive), e.g. ``'conus'``, ``'nyc'``,
        ``'europe'``, ``'world'``; or an ISO-3166 / GADM alpha-3 country code,
        e.g. ``'USA'``, ``'FRA'``, ``'JPN'``. Curated regions come back in a
        projected CRS; country codes come back in EPSG:4326.
    resolution : float or tuple of float, optional
        Cell size in the template's CRS units (metres for projected regions,
        degrees for country codes). A scalar gives square cells; a
        ``(res_x, res_y)`` tuple sets each axis. Defaults to a per-template
        value so a bare ``from_template('conus')`` works.
    backend : str, default='numpy'
        Array backend: ``'numpy'``, ``'dask+numpy'`` (alias ``'dask'``),
        ``'cupy'``, or ``'dask+cupy'``.
    fill : float, default=numpy.nan
        Value the grid is filled with. The dtype is always ``float32``.
    chunks : int, str, or tuple, default='auto'
        Dask chunk specification; only used by the dask backends.

    Returns
    -------
    template : xarray.DataArray
        Empty 2-D raster with ``dims=('y', 'x')``, pixel-center coordinates,
        and ``attrs`` carrying ``res`` and ``crs``.

    Examples
    --------
    .. sourcecode:: python

        >>> from xrspatial import from_template
        >>> agg = from_template("conus")            # Albers, default 5 km cells
        >>> agg.attrs["crs"]
        5070
        >>> agg = from_template("conus", resolution=1000)   # 1 km cells
        >>> agg = from_template("FRA")              # France bbox in EPSG:4326
        >>> agg.attrs["crs"]
        4326
    """
    bounds, crs, default_res, key = _resolve(name)
    left, bottom, right, top = bounds
    res_x, res_y = _normalize_resolution(resolution, default_res)

    width = max(1, int(round((right - left) / res_x)))
    height = max(1, int(round((top - bottom) / res_y)))

    n_cells = width * height
    if n_cells > _MAX_CELLS:
        raise ValueError(
            f"resolution {(res_x, res_y)} produces a {height} x {width} grid "
            f"({n_cells:,} cells), exceeding the {_MAX_CELLS:,}-cell limit. "
            f"Use a coarser resolution."
        )

    ys, xs = _make_output_coords(bounds, (height, width))
    # Realized cell size from the integer grid (may differ slightly from input).
    actual_res_x = (right - left) / width
    actual_res_y = (top - bottom) / height

    data = _make_data((height, width), fill, backend, chunks)
    unit = "degree" if crs == 4326 else "m"

    template = xr.DataArray(
        data,
        name=key,
        coords={"y": ys, "x": xs},
        dims=["y", "x"],
        attrs={"res": (actual_res_x, actual_res_y), "crs": crs},
    )
    template["x"].attrs["units"] = unit
    template["y"].attrs["units"] = unit
    return template
