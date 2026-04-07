"""Clip a raster to an arbitrary polygon geometry.

Unlike :func:`~xrspatial.zonal.crop`, which extracts a rectangular bounding
box, ``clip_polygon`` masks pixels that fall outside the polygon boundary to
a configurable nodata value (default ``np.nan``).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.utils import (
    _validate_raster, has_cuda_and_cupy, has_dask_array, is_cupy_array,
    is_dask_cupy,
)


def _resolve_geometry(geometry):
    """Return a list of ``(shapely_geom, 1.0)`` pairs for rasterize().

    Accepts a single shapely geometry, an iterable of shapely geometries,
    a GeoDataFrame/GeoSeries, or a coordinate array ``[(x, y), ...]``.
    """
    # GeoDataFrame / GeoSeries — check FIRST because GeoDataFrame has a
    # ``geom_type`` property that returns a *Series*, which would break the
    # scalar ``in`` check below.
    try:
        import geopandas as gpd
        if isinstance(geometry, gpd.GeoDataFrame):
            from shapely.ops import unary_union
            merged = unary_union(geometry.geometry)
            return [(merged, 1.0)]
        if isinstance(geometry, gpd.GeoSeries):
            from shapely.ops import unary_union
            merged = unary_union(geometry)
            return [(merged, 1.0)]
    except ImportError:
        pass

    # Single shapely geometry
    _base = getattr(geometry, 'geom_type', None)
    if isinstance(_base, str) and _base in ('Polygon', 'MultiPolygon'):
        return [(geometry, 1.0)]

    # Iterable of shapely geometries or coordinate pairs
    if hasattr(geometry, '__iter__'):
        items = list(geometry)
        if len(items) == 0:
            raise ValueError("geometry is empty")

        first = items[0]

        # List of shapely geometries
        ft = getattr(first, 'geom_type', None)
        if isinstance(ft, str) and ft in ('Polygon', 'MultiPolygon'):
            from shapely.ops import unary_union
            merged = unary_union(items)
            return [(merged, 1.0)]

        # Coordinate array: [(x, y), ...] — build a polygon
        if (isinstance(first, (list, tuple, np.ndarray))
                and len(first) == 2):
            from shapely.geometry import Polygon as ShapelyPolygon
            poly = ShapelyPolygon(items)
            return [(poly, 1.0)]

    raise TypeError(
        f"geometry must be a shapely Polygon/MultiPolygon, an iterable of "
        f"geometries, a GeoDataFrame, or a list of (x, y) coordinates. "
        f"Got {type(geometry)}"
    )


def _crop_to_bbox(raster, geom_pairs):
    """Slice the raster to the bounding box of the geometry.

    Returns the sliced DataArray and the geometry pairs (unchanged).
    """
    from shapely.ops import unary_union
    merged = unary_union([g for g, _ in geom_pairs])
    minx, miny, maxx, maxy = merged.bounds

    y_coords = raster.coords[raster.dims[-2]].values
    x_coords = raster.coords[raster.dims[-1]].values

    # Determine coordinate ordering (ascending or descending)
    y_ascending = y_coords[-1] >= y_coords[0]
    x_ascending = x_coords[-1] >= x_coords[0]

    if y_ascending:
        y_mask = (y_coords >= miny) & (y_coords <= maxy)
    else:
        y_mask = (y_coords >= miny) & (y_coords <= maxy)

    if x_ascending:
        x_mask = (x_coords >= minx) & (x_coords <= maxx)
    else:
        x_mask = (x_coords >= minx) & (x_coords <= maxx)

    y_idx = np.where(y_mask)[0]
    x_idx = np.where(x_mask)[0]

    if len(y_idx) == 0 or len(x_idx) == 0:
        raise ValueError(
            "Clipping geometry does not overlap the raster extent."
        )

    y_slice = slice(int(y_idx[0]), int(y_idx[-1]) + 1)
    x_slice = slice(int(x_idx[0]), int(x_idx[-1]) + 1)

    return raster[..., y_slice, x_slice]


def clip_polygon(
    raster: xr.DataArray,
    geometry,
    nodata: float = np.nan,
    crop: bool = True,
    all_touched: bool = False,
    rasterize_kw: Optional[dict] = None,
    name: Optional[str] = None,
) -> xr.DataArray:
    """Clip a raster to an arbitrary polygon geometry.

    Pixels outside the polygon are set to *nodata*. When *crop* is True
    (the default), the output is also trimmed to the polygon's bounding
    box so the result is smaller than the input.

    Parameters
    ----------
    raster : xr.DataArray
        Input raster to clip.  Must be at least 2-D with named ``y``
        and ``x`` dimensions (last two dimensions).
    geometry : shapely geometry, list of geometries, GeoDataFrame, or coordinate array
        The clipping polygon(s).  Accepts:

        * A single ``shapely.geometry.Polygon`` or ``MultiPolygon``.
        * An iterable of shapely polygon geometries (merged via
          ``unary_union``).
        * A ``GeoDataFrame`` or ``GeoSeries`` (merged via
          ``unary_union``).
        * A list of ``(x, y)`` coordinate pairs defining a single
          polygon ring.
    nodata : float, default np.nan
        Value to assign to pixels outside the polygon.
    crop : bool, default True
        If True, also trim the output to the bounding box of the
        polygon.  This reduces memory usage for small clip regions
        within large rasters.
    all_touched : bool, default False
        If True, all pixels touched by the polygon boundary are
        included.  If False (default), only pixels whose centre falls
        inside the polygon are included.
    rasterize_kw : dict, optional
        Extra keyword arguments forwarded to :func:`~xrspatial.rasterize.rasterize`
        when creating the polygon mask.
    name : str, optional
        Name for the output DataArray.  Defaults to the input name.

    Returns
    -------
    xr.DataArray
        Clipped raster with the same dtype and attributes as *raster*.

    Examples
    --------
    .. sourcecode:: python

        >>> from shapely.geometry import Polygon
        >>> import xrspatial
        >>> poly = Polygon([(1, 1), (1, 3), (3, 3), (3, 1)])
        >>> clipped = xrspatial.clip_polygon(raster, poly)
    """
    _validate_raster(raster, func_name='clip_polygon', name='raster')

    # Resolve geometry into [(shapely_geom, 1.0)] pairs
    geom_pairs = _resolve_geometry(geometry)

    # Optionally crop to bounding box first (reduces rasterize cost)
    if crop:
        raster = _crop_to_bbox(raster, geom_pairs)

    # Build a binary mask via rasterize, aligned to the (possibly cropped)
    # raster grid.  Always produce a plain numpy mask first, then convert
    # to the raster's backend so xarray's .where() sees matching types.
    from .rasterize import rasterize

    kw = dict(rasterize_kw or {})
    kw['like'] = raster
    kw['fill'] = 0.0
    kw['dtype'] = np.uint8
    kw['all_touched'] = all_touched

    mask = rasterize(geom_pairs, **kw)

    # Get mask as a plain numpy array regardless of what rasterize returned
    mask_np = mask.data
    if has_dask_array() and isinstance(mask_np, da.Array):
        mask_np = mask_np.compute()
    if has_cuda_and_cupy() and is_cupy_array(mask_np):
        mask_np = mask_np.get()

    # Apply the mask.  For CuPy backends we operate on the raw array
    # to avoid an xarray/cupy incompatibility in ``DataArray.where()``.
    cond_np = mask_np == 1

    if has_cuda_and_cupy() and is_cupy_array(raster.data):
        import cupy
        cond_cp = cupy.asarray(cond_np)
        out = raster.data.copy()
        out[~cond_cp] = nodata
        result = xr.DataArray(out, dims=raster.dims, coords=raster.coords)
    elif has_cuda_and_cupy() and is_dask_cupy(raster):
        import cupy
        cond_cp = cupy.asarray(cond_np)

        def _apply_mask(block, block_info=None):
            if block_info is not None:
                loc = block_info[0]['array-location']
                y_sl = slice(loc[-2][0], loc[-2][1])
                x_sl = slice(loc[-1][0], loc[-1][1])
                c = cond_cp[y_sl, x_sl]
            else:
                c = cond_cp
            out = block.copy()
            out[~c] = nodata
            return out

        out = raster.data.map_blocks(_apply_mask, dtype=raster.dtype)
        result = xr.DataArray(out, dims=raster.dims, coords=raster.coords)
    elif has_dask_array() and isinstance(raster.data, da.Array):
        cond_da = da.from_array(cond_np, chunks=raster.data.chunks[-2:])
        result = raster.where(cond_da, other=nodata)
    else:
        result = raster.where(cond_np, other=nodata)

    result.attrs = raster.attrs
    result.name = name if name is not None else raster.name
    return result
