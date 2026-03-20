"""Lightweight GeoTIFF/COG reader and writer.

No GDAL dependency -- uses only numpy, numba, xarray, and the standard library.

Public API
----------
read_geotiff(source, ...)
    Read a GeoTIFF file to an xarray.DataArray.
write_geotiff(data, path, ...)
    Write an xarray.DataArray as a GeoTIFF or COG.
open_cog(url, ...)
    Read a Cloud Optimized GeoTIFF from an HTTP URL.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ._geotags import GeoTransform, RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT
from ._reader import read_to_array
from ._writer import write

__all__ = ['read_geotiff', 'write_geotiff', 'open_cog', 'read_geotiff_dask']


def _wkt_to_epsg(wkt_or_proj: str) -> int | None:
    """Try to extract an EPSG code from a WKT or PROJ string.

    Returns None if pyproj is not installed or the string can't be parsed.
    """
    try:
        from pyproj import CRS
        crs = CRS.from_user_input(wkt_or_proj)
        epsg = crs.to_epsg()
        return epsg
    except Exception:
        return None


def _geo_to_coords(geo_info, height: int, width: int) -> dict:
    """Build y/x coordinate arrays from GeoInfo.

    For PixelIsArea (default): origin is the edge of pixel (0,0), so pixel
    centers are at origin + 0.5*pixel_size.
    For PixelIsPoint: origin (tiepoint) is already the center of pixel (0,0),
    so no half-pixel offset is needed.
    """
    t = geo_info.transform
    if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
        # Tiepoint is pixel center -- no offset needed
        x = np.arange(width, dtype=np.float64) * t.pixel_width + t.origin_x
        y = np.arange(height, dtype=np.float64) * t.pixel_height + t.origin_y
    else:
        # Tiepoint is pixel edge -- shift to center
        x = np.arange(width, dtype=np.float64) * t.pixel_width + t.origin_x + t.pixel_width * 0.5
        y = np.arange(height, dtype=np.float64) * t.pixel_height + t.origin_y + t.pixel_height * 0.5
    return {'y': y, 'x': x}


def _coords_to_transform(da: xr.DataArray) -> GeoTransform | None:
    """Infer GeoTransform from DataArray coordinates.

    Coordinates are always pixel-center values. The transform origin depends
    on raster_type:
    - PixelIsArea (default): origin = center - half_pixel  (edge of pixel 0)
    - PixelIsPoint: origin = center  (center of pixel 0)
    """
    ydim = da.dims[-2]
    xdim = da.dims[-1]

    if xdim not in da.coords or ydim not in da.coords:
        return None

    x = da.coords[xdim].values
    y = da.coords[ydim].values

    if len(x) < 2 or len(y) < 2:
        return None

    pixel_width = float(x[1] - x[0])
    pixel_height = float(y[1] - y[0])

    is_point = da.attrs.get('raster_type') == 'point'
    if is_point:
        # PixelIsPoint: tiepoint is at the pixel center
        origin_x = float(x[0])
        origin_y = float(y[0])
    else:
        # PixelIsArea: tiepoint is at the edge (center - half pixel)
        origin_x = float(x[0]) - pixel_width * 0.5
        origin_y = float(y[0]) - pixel_height * 0.5

    return GeoTransform(
        origin_x=origin_x,
        origin_y=origin_y,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )


def read_geotiff(source: str, *, window=None,
                 overview_level: int | None = None,
                 band: int | None = None,
                 name: str | None = None) -> xr.DataArray:
    """Read a GeoTIFF file into an xarray.DataArray.

    Parameters
    ----------
    source : str
        File path or HTTP URL.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop) for windowed reading.
    overview_level : int or None
        Overview level to read (0 = full resolution). None reads full res.
    band : int
        Band index (0-based) for multi-band files.
    name : str or None
        Name for the DataArray. Defaults to filename stem.

    Returns
    -------
    xr.DataArray
        2D DataArray with y/x coordinates and geo attributes.
    """
    arr, geo_info = read_to_array(
        source, window=window,
        overview_level=overview_level, band=band,
    )

    height, width = arr.shape[:2]
    coords = _geo_to_coords(geo_info, height, width)

    if window is not None:
        # Adjust coordinates for windowed read
        r0, c0, r1, c1 = window
        t = geo_info.transform
        full_x = np.arange(c0, c1, dtype=np.float64) * t.pixel_width + t.origin_x + t.pixel_width * 0.5
        full_y = np.arange(r0, r1, dtype=np.float64) * t.pixel_height + t.origin_y + t.pixel_height * 0.5
        coords = {'y': full_y, 'x': full_x}

    if name is None:
        # Derive from source path
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    attrs = {}
    if geo_info.crs_epsg is not None:
        attrs['crs'] = geo_info.crs_epsg
    if geo_info.crs_wkt is not None:
        attrs['crs_wkt'] = geo_info.crs_wkt
    if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
        attrs['raster_type'] = 'point'

    # CRS description fields
    if geo_info.crs_name is not None:
        attrs['crs_name'] = geo_info.crs_name
    if geo_info.geog_citation is not None:
        attrs['geog_citation'] = geo_info.geog_citation
    if geo_info.datum_code is not None:
        attrs['datum_code'] = geo_info.datum_code
    if geo_info.angular_units is not None:
        attrs['angular_units'] = geo_info.angular_units
    if geo_info.linear_units is not None:
        attrs['linear_units'] = geo_info.linear_units
    if geo_info.semi_major_axis is not None:
        attrs['semi_major_axis'] = geo_info.semi_major_axis
    if geo_info.inv_flattening is not None:
        attrs['inv_flattening'] = geo_info.inv_flattening
    if geo_info.projection_code is not None:
        attrs['projection_code'] = geo_info.projection_code
    # Vertical CRS
    if geo_info.vertical_epsg is not None:
        attrs['vertical_crs'] = geo_info.vertical_epsg
    if geo_info.vertical_citation is not None:
        attrs['vertical_citation'] = geo_info.vertical_citation
    if geo_info.vertical_units is not None:
        attrs['vertical_units'] = geo_info.vertical_units

    # GDAL metadata (tag 42112)
    if geo_info.gdal_metadata is not None:
        attrs['gdal_metadata'] = geo_info.gdal_metadata
    if geo_info.gdal_metadata_xml is not None:
        attrs['gdal_metadata_xml'] = geo_info.gdal_metadata_xml

    # Extra (non-managed) TIFF tags for pass-through
    if geo_info.extra_tags is not None:
        attrs['extra_tags'] = geo_info.extra_tags

    # Resolution / DPI metadata
    if geo_info.x_resolution is not None:
        attrs['x_resolution'] = geo_info.x_resolution
    if geo_info.y_resolution is not None:
        attrs['y_resolution'] = geo_info.y_resolution
    if geo_info.resolution_unit is not None:
        _unit_names = {1: 'none', 2: 'inch', 3: 'centimeter'}
        attrs['resolution_unit'] = _unit_names.get(
            geo_info.resolution_unit, str(geo_info.resolution_unit))

    # Attach palette colormap for indexed-color TIFFs
    if geo_info.colormap is not None:
        try:
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(geo_info.colormap, name='tiff_palette')
            attrs['cmap'] = cmap
            attrs['colormap_rgba'] = geo_info.colormap
        except ImportError:
            # matplotlib not available -- store raw RGBA tuples only
            attrs['colormap_rgba'] = geo_info.colormap

    # Apply nodata mask: replace nodata sentinel values with NaN
    nodata = geo_info.nodata
    if nodata is not None:
        attrs['nodata'] = nodata
        if arr.dtype.kind == 'f':
            if not np.isnan(nodata):
                arr = arr.copy()
                arr[arr == arr.dtype.type(nodata)] = np.nan
        elif arr.dtype.kind in ('u', 'i'):
            # Integer arrays: convert to float to represent NaN
            nodata_int = int(nodata)
            mask = arr == arr.dtype.type(nodata_int)
            if mask.any():
                arr = arr.astype(np.float64)
                arr[mask] = np.nan

    if arr.ndim == 3:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(arr.shape[2])
    else:
        dims = ['y', 'x']

    da = xr.DataArray(
        arr,
        dims=dims,
        coords=coords,
        name=name,
        attrs=attrs,
    )
    return da


def write_geotiff(data: xr.DataArray | np.ndarray, path: str, *,
                  crs: int | str | None = None,
                  nodata=None,
                  compression: str = 'deflate',
                  tiled: bool = True,
                  tile_size: int = 256,
                  predictor: bool = False,
                  cog: bool = False,
                  overview_levels: list[int] | None = None,
                  overview_resampling: str = 'mean',
                  bigtiff: bool | None = None) -> None:
    """Write data as a GeoTIFF or Cloud Optimized GeoTIFF.

    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        2D raster data.
    path : str
        Output file path.
    crs : int, str, or None
        EPSG code (int), WKT string, or PROJ string. If None and data
        is a DataArray, tries to read from attrs ('crs' for EPSG,
        'crs_wkt' for WKT).
    nodata : float, int, or None
        NoData value.
    compression : str
        'none', 'deflate', or 'lzw'.
    tiled : bool
        Use tiled layout (default True).
    tile_size : int
        Tile size in pixels (default 256).
    predictor : bool
        Use horizontal differencing predictor.
    cog : bool
        Write as Cloud Optimized GeoTIFF.
    overview_levels : list[int] or None
        Overview decimation factors. Only used when cog=True.
    overview_resampling : str
        Resampling method for overviews: 'mean' (default), 'nearest',
        'min', 'max', 'median', 'mode', or 'cubic'.
    """
    geo_transform = None
    epsg = None
    raster_type = RASTER_PIXEL_IS_AREA
    x_res = None
    y_res = None
    res_unit = None
    gdal_meta_xml = None
    extra_tags_list = None

    # Resolve crs argument: can be int (EPSG) or str (WKT/PROJ)
    if isinstance(crs, int):
        epsg = crs
    elif isinstance(crs, str):
        epsg = _wkt_to_epsg(crs)  # try to extract EPSG from WKT/PROJ

    if isinstance(data, xr.DataArray):
        arr = data.values
        if geo_transform is None:
            geo_transform = _coords_to_transform(data)
        if epsg is None and crs is None:
            epsg = data.attrs.get('crs')
            if epsg is None:
                # Try resolving EPSG from a WKT string in attrs
                wkt = data.attrs.get('crs_wkt')
                if isinstance(wkt, str):
                    epsg = _wkt_to_epsg(wkt)
        if nodata is None:
            nodata = data.attrs.get('nodata')
        if data.attrs.get('raster_type') == 'point':
            raster_type = RASTER_PIXEL_IS_POINT
        # GDAL metadata from attrs (prefer raw XML, fall back to dict)
        gdal_meta_xml = data.attrs.get('gdal_metadata_xml')
        if gdal_meta_xml is None:
            gdal_meta_dict = data.attrs.get('gdal_metadata')
            if isinstance(gdal_meta_dict, dict):
                from ._geotags import _build_gdal_metadata_xml
                gdal_meta_xml = _build_gdal_metadata_xml(gdal_meta_dict)
        # Extra tags for pass-through
        extra_tags_list = data.attrs.get('extra_tags')
        # Resolution / DPI from attrs
        x_res = data.attrs.get('x_resolution')
        y_res = data.attrs.get('y_resolution')
        unit_str = data.attrs.get('resolution_unit')
        if unit_str is not None:
            _unit_ids = {'none': 1, 'inch': 2, 'centimeter': 3}
            res_unit = _unit_ids.get(str(unit_str), None)
    else:
        arr = np.asarray(data)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")

    write(
        arr, path,
        geo_transform=geo_transform,
        crs_epsg=epsg,
        nodata=nodata,
        compression=compression,
        tiled=tiled,
        tile_size=tile_size,
        predictor=predictor,
        cog=cog,
        overview_levels=overview_levels,
        overview_resampling=overview_resampling,
        raster_type=raster_type,
        x_resolution=x_res,
        y_resolution=y_res,
        resolution_unit=res_unit,
        gdal_metadata_xml=gdal_meta_xml,
        extra_tags=extra_tags_list,
        bigtiff=bigtiff,
    )


def open_cog(url: str, *,
             overview_level: int | None = None) -> xr.DataArray:
    """Read a Cloud Optimized GeoTIFF from an HTTP URL.

    Uses range requests so only the needed tiles are fetched.

    Parameters
    ----------
    url : str
        HTTP(S) URL to the COG.
    overview_level : int or None
        Overview level (0 = full resolution).

    Returns
    -------
    xr.DataArray
    """
    return read_geotiff(url, overview_level=overview_level)


def read_geotiff_dask(source: str, *, chunks: int | tuple = 512,
                      overview_level: int | None = None,
                      name: str | None = None) -> xr.DataArray:
    """Read a GeoTIFF as a dask-backed DataArray for out-of-core processing.

    Each chunk is loaded lazily via windowed reads.

    Parameters
    ----------
    source : str
        File path.
    chunks : int or (row_chunk, col_chunk) tuple
        Chunk size in pixels. Default 512.
    overview_level : int or None
        Overview level (0 = full resolution).
    name : str or None
        Name for the DataArray.

    Returns
    -------
    xr.DataArray
        Dask-backed DataArray with y/x coordinates.
    """
    import dask.array as da

    # First, do a metadata-only read to get shape, dtype, coords, attrs
    arr, geo_info = read_to_array(source, overview_level=overview_level)
    full_h, full_w = arr.shape[:2]
    n_bands = arr.shape[2] if arr.ndim == 3 else 0
    dtype = arr.dtype

    coords = _geo_to_coords(geo_info, full_h, full_w)

    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    attrs = {}
    if geo_info.crs_epsg is not None:
        attrs['crs'] = geo_info.crs_epsg
    if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
        attrs['raster_type'] = 'point'
    if geo_info.nodata is not None:
        attrs['nodata'] = geo_info.nodata

    if isinstance(chunks, int):
        ch_h = ch_w = chunks
    else:
        ch_h, ch_w = chunks

    # Build dask array from delayed windowed reads
    rows = list(range(0, full_h, ch_h))
    cols = list(range(0, full_w, ch_w))

    # For multi-band, each window read returns (h, w, bands); for single-band (h, w)
    # read_to_array with band=0 extracts a single band, band=None returns all
    band_arg = None  # return all bands (or 2D if single-band)

    dask_rows = []
    for r0 in rows:
        r1 = min(r0 + ch_h, full_h)
        dask_cols = []
        for c0 in cols:
            c1 = min(c0 + ch_w, full_w)
            if n_bands > 0:
                block_shape = (r1 - r0, c1 - c0, n_bands)
            else:
                block_shape = (r1 - r0, c1 - c0)
            block = da.from_delayed(
                _delayed_read_window(source, r0, c0, r1, c1,
                                     overview_level, geo_info.nodata,
                                     dtype, band_arg),
                shape=block_shape,
                dtype=dtype,
            )
            dask_cols.append(block)
        dask_rows.append(da.concatenate(dask_cols, axis=1))

    dask_arr = da.concatenate(dask_rows, axis=0)

    if n_bands > 0:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(n_bands)
    else:
        dims = ['y', 'x']

    return xr.DataArray(
        dask_arr, dims=dims, coords=coords, name=name, attrs=attrs,
    )


def _delayed_read_window(source, r0, c0, r1, c1, overview_level, nodata,
                         dtype, band):
    """Dask-delayed function to read a single window."""
    import dask
    @dask.delayed
    def _read():
        arr, _ = read_to_array(source, window=(r0, c0, r1, c1),
                               overview_level=overview_level, band=band)
        if nodata is not None:
            if arr.dtype.kind == 'f' and not np.isnan(nodata):
                arr = arr.copy()
                arr[arr == arr.dtype.type(nodata)] = np.nan
            elif arr.dtype.kind in ('u', 'i'):
                mask = arr == arr.dtype.type(int(nodata))
                if mask.any():
                    arr = arr.astype(np.float64)
                    arr[mask] = np.nan
        return arr
    return _read()


def plot_geotiff(da: xr.DataArray, **kwargs):
    """Plot a DataArray using its embedded colormap if present.

    Deprecated: use ``da.xrs.plot()`` instead.
    """
    return da.xrs.plot(**kwargs)
