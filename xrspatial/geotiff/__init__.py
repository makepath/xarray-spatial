"""Lightweight GeoTIFF/COG reader and writer.

No GDAL dependency -- uses only numpy, numba, xarray, and the standard library.

Public API
----------
open_geotiff(source, ...)
    Read a GeoTIFF file to an xarray.DataArray.
to_geotiff(data, path, ...)
    Write an xarray.DataArray as a GeoTIFF or COG.
write_vrt(vrt_path, source_files, ...)
    Generate a VRT mosaic XML from a list of GeoTIFF files.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ._geotags import GeoTransform, RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT
from ._reader import read_to_array
from ._writer import write

__all__ = ['open_geotiff', 'to_geotiff', 'write_vrt']


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


def _validate_dtype_cast(source_dtype, target_dtype):
    """Validate that casting source_dtype to target_dtype is allowed.

    Raises ValueError for float-to-int casts (lossy in a way users
    often don't intend).  All other casts are permitted -- the user
    asked for them explicitly.
    """
    src = np.dtype(source_dtype)
    tgt = np.dtype(target_dtype)
    if src.kind == 'f' and tgt.kind in ('u', 'i'):
        raise ValueError(
            f"Cannot cast float ({src}) to int ({tgt}). "
            f"This loses fractional data and is usually unintentional. "
            f"Cast explicitly after reading if you really want this.")


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


def _read_geo_info(source: str, *, overview_level: int | None = None):
    """Read only the geographic metadata and image dimensions from a GeoTIFF.

    Returns (geo_info, height, width, dtype, n_bands) without reading pixel
    data.  Uses mmap for header-only access -- O(1) memory regardless of file
    size.

    Parameters
    ----------
    overview_level : int or None
        Overview IFD index (0 = full resolution).
    """
    from ._dtypes import tiff_dtype_to_numpy
    from ._geotags import extract_geo_info
    from ._header import parse_all_ifds, parse_header

    with open(source, 'rb') as f:
        import mmap
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        ifd_idx = 0
        if overview_level is not None:
            ifd_idx = min(overview_level, len(ifds) - 1)
        ifd = ifds[ifd_idx]
        geo_info = extract_geo_info(ifd, data, header.byte_order)
        bps = ifd.bits_per_sample
        if isinstance(bps, tuple):
            bps = bps[0]
        file_dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
        n_bands = ifd.samples_per_pixel if ifd.samples_per_pixel > 1 else 0
        return geo_info, ifd.height, ifd.width, file_dtype, n_bands
    finally:
        data.close()


def _extent_to_window(transform, file_height, file_width,
                      y_min, y_max, x_min, x_max):
    """Convert geographic extent to pixel window (row_start, col_start, row_stop, col_stop).

    Clamps to file bounds.
    """
    # Pixel coords from geographic coords
    col_start = (x_min - transform.origin_x) / transform.pixel_width
    col_stop = (x_max - transform.origin_x) / transform.pixel_width

    row_start = (y_max - transform.origin_y) / transform.pixel_height
    row_stop = (y_min - transform.origin_y) / transform.pixel_height

    # pixel_height is typically negative, so row_start/row_stop may be swapped
    if row_start > row_stop:
        row_start, row_stop = row_stop, row_start
    if col_start > col_stop:
        col_start, col_stop = col_stop, col_start

    row_start = max(0, int(np.floor(row_start)))
    col_start = max(0, int(np.floor(col_start)))
    row_stop = min(file_height, int(np.ceil(row_stop)))
    col_stop = min(file_width, int(np.ceil(col_stop)))

    return (row_start, col_start, row_stop, col_stop)




def open_geotiff(source: str, *, dtype=None, window=None,
                 overview_level: int | None = None,
                 band: int | None = None,
                 name: str | None = None,
                 chunks: int | tuple | None = None,
                 gpu: bool = False,
                 max_pixels: int | None = None) -> xr.DataArray:
    """Read a GeoTIFF, COG, or VRT file into an xarray.DataArray.

    Automatically dispatches to the best backend:
    - ``gpu=True``: GPU-accelerated read via nvCOMP (returns CuPy)
    - ``chunks=N``: Dask lazy read via windowed chunks
    - ``gpu=True, chunks=N``: Dask+CuPy for out-of-core GPU pipelines
    - Default: NumPy eager read

    VRT files are auto-detected by extension.

    Parameters
    ----------
    source : str
        File path, HTTP URL, or cloud URI (s3://, gs://, az://).
    dtype : str, numpy.dtype, or None
        Cast the result to this dtype after reading. None keeps the
        file's native dtype. Float-to-int casts raise ValueError to
        prevent accidental data loss.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop) for windowed reading.
    overview_level : int or None
        Overview level (0 = full resolution).
    band : int or None
        Band index (0-based). None returns all bands.
    name : str or None
        Name for the DataArray.
    chunks : int, tuple, or None
        Chunk size for Dask lazy reading.
    gpu : bool
        Use GPU-accelerated decompression (requires cupy + nvCOMP).
    max_pixels : int or None
        Maximum allowed pixel count (width * height * samples). None
        uses the default (~1 billion). Raise to read legitimately
        large files.

    Returns
    -------
    xr.DataArray
        NumPy, Dask, CuPy, or Dask+CuPy backed depending on options.
    """
    # VRT files
    if source.lower().endswith('.vrt'):
        return read_vrt(source, dtype=dtype, window=window, band=band,
                        name=name, chunks=chunks, gpu=gpu,
                        max_pixels=max_pixels)

    # GPU path
    if gpu:
        return read_geotiff_gpu(source, dtype=dtype,
                                overview_level=overview_level,
                                name=name, chunks=chunks,
                                max_pixels=max_pixels)

    # Dask path (CPU)
    if chunks is not None:
        return read_geotiff_dask(source, dtype=dtype, chunks=chunks,
                                 overview_level=overview_level, name=name)

    kwargs = {}
    if max_pixels is not None:
        kwargs['max_pixels'] = max_pixels
    arr, geo_info = read_to_array(
        source, window=window,
        overview_level=overview_level, band=band,
        **kwargs,
    )

    height, width = arr.shape[:2]
    coords = _geo_to_coords(geo_info, height, width)

    if window is not None:
        # Adjust coordinates for windowed read
        r0, c0, r1, c1 = window
        t = geo_info.transform
        if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
            full_x = np.arange(c0, c1, dtype=np.float64) * t.pixel_width + t.origin_x
            full_y = np.arange(r0, r1, dtype=np.float64) * t.pixel_height + t.origin_y
        else:
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

    if dtype is not None:
        target = np.dtype(dtype)
        _validate_dtype_cast(arr.dtype, target)
        arr = arr.astype(target)

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


def _is_gpu_data(data) -> bool:
    """Check if data is CuPy-backed (raw array or DataArray)."""
    try:
        import cupy
        _cupy_type = cupy.ndarray
    except ImportError:
        return False

    if isinstance(data, xr.DataArray):
        raw = data.data
        if hasattr(raw, 'compute'):
            meta = getattr(raw, '_meta', None)
            return isinstance(meta, _cupy_type)
        return isinstance(raw, _cupy_type)
    return isinstance(data, _cupy_type)


_LEVEL_RANGES = {
    'deflate': (1, 9),
    'zstd': (1, 22),
    'lz4': (0, 16),
}


def to_geotiff(data: xr.DataArray | np.ndarray, path: str, *,
               crs: int | str | None = None,
               nodata=None,
               compression: str = 'zstd',
               compression_level: int | None = None,
               tiled: bool = True,
               tile_size: int = 256,
               predictor: bool = False,
               cog: bool = False,
               overview_levels: list[int] | None = None,
               overview_resampling: str = 'mean',
               bigtiff: bool | None = None,
               gpu: bool | None = None) -> None:
    """Write data as a GeoTIFF or Cloud Optimized GeoTIFF.

    Dask-backed DataArrays are written in streaming mode: one tile-row
    at a time, without materialising the full array into RAM.  Peak
    memory is roughly ``tile_size * width * bytes_per_sample``.  COG
    output (``cog=True``) still materialises because overviews need the
    full array.

    Automatically dispatches to GPU compression when:
    - ``gpu=True`` is passed, or
    - The input data is CuPy-backed (auto-detected)

    GPU write uses nvCOMP batch compression (deflate/ZSTD) and keeps
    the array on device. Falls back to CPU if nvCOMP is not available.

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
        'none', 'deflate', 'lzw', 'jpeg', 'packbits', or 'zstd'.
        JPEG is lossy and only supports uint8 data (1 or 3 bands).
        With ``gpu=True``, JPEG uses nvJPEG for GPU-accelerated
        encode/decode when available, falling back to Pillow on CPU.
    compression_level : int or None
        Compression effort level. None uses each codec's default (6 for
        deflate/zstd). Valid ranges: deflate 1-9, zstd 1-22, lz4 0-16.
        Codecs without a level concept (lzw, packbits, jpeg) accept any
        value and ignore it.
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
    gpu : bool or None
        Force GPU compression. None (default) auto-detects CuPy data.
    """
    # VRT tiled output
    if path.lower().endswith('.vrt'):
        if cog:
            raise ValueError(
                "cog=True is not compatible with VRT output. "
                "VRT writes tiled GeoTIFFs, not a single COG.")
        if overview_levels is not None:
            raise ValueError(
                "overview_levels is not compatible with VRT output. "
                "VRT tiles do not include overviews.")
        _write_vrt_tiled(data, path,
                         crs=crs, nodata=nodata,
                         compression=compression,
                         compression_level=compression_level,
                         tile_size=tile_size,
                         predictor=predictor,
                         bigtiff=bigtiff)
        return

    # Auto-detect GPU data and dispatch to write_geotiff_gpu
    use_gpu = gpu if gpu is not None else _is_gpu_data(data)
    if use_gpu:
        try:
            write_geotiff_gpu(data, path, crs=crs, nodata=nodata,
                              compression=compression,
                              compression_level=compression_level,
                              tile_size=tile_size,
                              predictor=predictor,
                              cog=cog,
                              overview_levels=overview_levels,
                              overview_resampling=overview_resampling)
            return
        except (ImportError, Exception):
            pass  # fall through to CPU path

    geo_transform = None
    epsg = None
    wkt_fallback = None  # WKT string when EPSG is not available
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
        if epsg is None:
            wkt_fallback = crs

    if isinstance(data, xr.DataArray):
        raw = data.data

        # Extract metadata from DataArray attrs (no materialisation needed)
        if geo_transform is None:
            geo_transform = _coords_to_transform(data)
        if epsg is None and crs is None:
            crs_attr = data.attrs.get('crs')
            if isinstance(crs_attr, str):
                epsg = _wkt_to_epsg(crs_attr)
                if epsg is None and wkt_fallback is None:
                    wkt_fallback = crs_attr
            elif crs_attr is not None:
                epsg = int(crs_attr)
            if epsg is None:
                wkt = data.attrs.get('crs_wkt')
                if isinstance(wkt, str):
                    epsg = _wkt_to_epsg(wkt)
                    if epsg is None and wkt_fallback is None:
                        wkt_fallback = wkt
        if nodata is None:
            nodata = data.attrs.get('nodata')
        if data.attrs.get('raster_type') == 'point':
            raster_type = RASTER_PIXEL_IS_POINT
        gdal_meta_xml = data.attrs.get('gdal_metadata_xml')
        if gdal_meta_xml is None:
            gdal_meta_dict = data.attrs.get('gdal_metadata')
            if isinstance(gdal_meta_dict, dict):
                from ._geotags import _build_gdal_metadata_xml
                gdal_meta_xml = _build_gdal_metadata_xml(gdal_meta_dict)
        extra_tags_list = data.attrs.get('extra_tags')
        x_res = data.attrs.get('x_resolution')
        y_res = data.attrs.get('y_resolution')
        unit_str = data.attrs.get('resolution_unit')
        if unit_str is not None:
            _unit_ids = {'none': 1, 'inch': 2, 'centimeter': 3}
            res_unit = _unit_ids.get(str(unit_str), None)

        # Dask-backed: stream tiles to avoid materialising the full array.
        # COG requires overviews from the full array, so it falls through
        # to the eager path.
        if hasattr(raw, 'dask') and not cog:
            dask_arr = raw
            # Handle band-first dimension order (band, y, x) -> (y, x, band)
            if raw.ndim == 3 and data.dims[0] in ('band', 'bands', 'channel'):
                import dask.array as da
                dask_arr = da.moveaxis(raw, 0, -1)
            if dask_arr.ndim not in (2, 3):
                raise ValueError(
                    f"Expected 2D or 3D array, got {dask_arr.ndim}D")
            # Validate compression_level
            if compression_level is not None:
                level_range = _LEVEL_RANGES.get(compression.lower())
                if level_range is not None:
                    lo, hi = level_range
                    if not (lo <= compression_level <= hi):
                        raise ValueError(
                            f"compression_level={compression_level} out of "
                            f"range for {compression} (valid: {lo}-{hi})")
            from ._writer import write_streaming
            write_streaming(
                dask_arr, path,
                geo_transform=geo_transform,
                crs_epsg=epsg,
                crs_wkt=wkt_fallback if epsg is None else None,
                nodata=nodata,
                compression=compression,
                compression_level=compression_level,
                tiled=tiled,
                tile_size=tile_size,
                predictor=predictor,
                raster_type=raster_type,
                x_resolution=x_res,
                y_resolution=y_res,
                resolution_unit=res_unit,
                gdal_metadata_xml=gdal_meta_xml,
                extra_tags=extra_tags_list,
                bigtiff=bigtiff,
            )
            return

        # Eager compute (numpy, CuPy, or dask+COG)
        if hasattr(raw, 'get'):
            arr = raw.get()  # CuPy -> numpy
        elif hasattr(raw, 'compute'):
            arr = raw.compute()  # Dask -> numpy
            if hasattr(arr, 'get'):
                arr = arr.get()  # Dask+CuPy -> numpy
        else:
            arr = np.asarray(raw)
        # Handle band-first dimension order (band, y, x) -> (y, x, band)
        if arr.ndim == 3 and data.dims[0] in ('band', 'bands', 'channel'):
            arr = np.moveaxis(arr, 0, -1)
    else:
        if hasattr(data, 'get'):
            arr = data.get()  # CuPy -> numpy
        else:
            arr = np.asarray(data)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")

    # Auto-promote unsupported dtypes
    if arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)

    # Restore NaN pixels to the nodata sentinel value so the written file
    # has sentinel values matching the GDAL_NODATA tag.
    if nodata is not None and arr.dtype.kind == 'f' and not np.isnan(nodata):
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr = arr.copy()
            arr[nan_mask] = arr.dtype.type(nodata)

    # Validate compression_level against codec-specific range
    if compression_level is not None:
        level_range = _LEVEL_RANGES.get(compression.lower())
        if level_range is not None:
            lo, hi = level_range
            if not (lo <= compression_level <= hi):
                raise ValueError(
                    f"compression_level={compression_level} out of range "
                    f"for {compression} (valid: {lo}-{hi})")

    write(
        arr, path,
        geo_transform=geo_transform,
        crs_epsg=epsg,
        crs_wkt=wkt_fallback if epsg is None else None,
        nodata=nodata,
        compression=compression,
        compression_level=compression_level,
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


def _write_single_tile(chunk_data, path, geo_transform, epsg, wkt,
                       nodata, compression, compression_level,
                       tile_size, predictor, bigtiff):
    """Write a single tile GeoTIFF. Used by _write_vrt_tiled."""
    if hasattr(chunk_data, 'compute'):
        chunk_data = chunk_data.compute()
    if hasattr(chunk_data, 'get'):
        chunk_data = chunk_data.get()  # CuPy -> numpy

    arr = np.asarray(chunk_data)

    # Auto-promote unsupported dtypes
    if arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)

    # Restore NaN to nodata sentinel
    if nodata is not None and arr.dtype.kind == 'f' and not np.isnan(nodata):
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr = arr.copy()
            arr[nan_mask] = arr.dtype.type(nodata)

    write(arr, path,
          geo_transform=geo_transform,
          crs_epsg=epsg,
          crs_wkt=wkt if epsg is None else None,
          nodata=nodata,
          compression=compression,
          tiled=True,
          tile_size=tile_size,
          predictor=predictor,
          compression_level=compression_level,
          bigtiff=bigtiff)


def _write_vrt_tiled(data, vrt_path, *, crs=None, nodata=None,
                     compression='zstd', compression_level=None,
                     tile_size=256, predictor=False, bigtiff=None):
    """Write a DataArray as a directory of tiled GeoTIFFs with a VRT index.

    This enables streaming dask arrays to disk without materializing the
    full array in RAM.
    """
    import os

    # Validate compression_level against codec-specific range
    if compression_level is not None:
        level_range = _LEVEL_RANGES.get(compression.lower())
        if level_range is not None:
            lo, hi = level_range
            if not (lo <= compression_level <= hi):
                raise ValueError(
                    f"compression_level={compression_level} out of range "
                    f"for {compression} (valid: {lo}-{hi})")

    # Derive tiles directory from VRT path stem
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    stem = os.path.splitext(os.path.basename(vrt_path))[0]
    tiles_dir_name = stem + '_tiles'
    tiles_dir = os.path.join(vrt_dir, tiles_dir_name)

    # Validate tiles directory
    if os.path.isdir(tiles_dir) and os.listdir(tiles_dir):
        raise FileExistsError(
            f"Tiles directory already contains files: {tiles_dir}")
    os.makedirs(tiles_dir, exist_ok=True)

    # Resolve CRS
    epsg = None
    wkt_fallback = None
    if isinstance(crs, int):
        epsg = crs
    elif isinstance(crs, str):
        epsg = _wkt_to_epsg(crs)
        if epsg is None:
            wkt_fallback = crs

    geo_transform = None

    if isinstance(data, xr.DataArray):
        raw = data.data
        if epsg is None and crs is None:
            crs_attr = data.attrs.get('crs')
            if isinstance(crs_attr, str):
                epsg = _wkt_to_epsg(crs_attr)
                if epsg is None and wkt_fallback is None:
                    wkt_fallback = crs_attr
            elif crs_attr is not None:
                epsg = int(crs_attr)
            if epsg is None:
                wkt = data.attrs.get('crs_wkt')
                if isinstance(wkt, str):
                    epsg = _wkt_to_epsg(wkt)
                    if epsg is None and wkt_fallback is None:
                        wkt_fallback = wkt
        if nodata is None:
            nodata = data.attrs.get('nodata')
        geo_transform = _coords_to_transform(data)
    else:
        raw = data

    # Check for dask backing
    is_dask = hasattr(raw, 'dask')

    if is_dask:
        if raw.ndim != 2:
            raise ValueError(
                "VRT tiled output currently supports 2D arrays only, "
                f"got {raw.ndim}D. Squeeze or select a band first.")
        # Use dask chunk grid
        import dask
        row_chunks = raw.chunks[0]  # tuple of chunk sizes along y
        col_chunks = raw.chunks[1]  # tuple of chunk sizes along x
        n_row_tiles = len(row_chunks)
        n_col_tiles = len(col_chunks)
    else:
        # Numpy: tile using tile_size
        if hasattr(raw, 'get'):
            np_arr = raw.get()  # CuPy
        elif hasattr(raw, 'compute'):
            np_arr = raw.compute()
        else:
            np_arr = np.asarray(raw)
        if np_arr.ndim != 2:
            raise ValueError(
                "VRT tiled output currently supports 2D arrays only, "
                f"got {np_arr.ndim}D. Squeeze or select a band first.")
        height, width = np_arr.shape[:2]
        n_row_tiles = (height + tile_size - 1) // tile_size
        n_col_tiles = (width + tile_size - 1) // tile_size

    # Zero-padding width for tile names
    pad_width = max(2, len(str(max(n_row_tiles, n_col_tiles) - 1)))

    tile_paths = []
    delayed_tasks = []

    row_offset = 0
    for ri in range(n_row_tiles):
        if is_dask:
            chunk_h = row_chunks[ri]
        else:
            chunk_h = min(tile_size, height - row_offset)

        col_offset = 0
        for ci in range(n_col_tiles):
            if is_dask:
                chunk_w = col_chunks[ci]
            else:
                chunk_w = min(tile_size, width - col_offset)

            tile_name = f'tile_{ri:0{pad_width}d}_{ci:0{pad_width}d}.tif'
            tile_path = os.path.join(tiles_dir, tile_name)
            tile_paths.append(tile_path)

            # Compute per-tile geo_transform
            tile_gt = None
            if geo_transform is not None:
                tile_gt = GeoTransform(
                    origin_x=geo_transform.origin_x + col_offset * geo_transform.pixel_width,
                    origin_y=geo_transform.origin_y + row_offset * geo_transform.pixel_height,
                    pixel_width=geo_transform.pixel_width,
                    pixel_height=geo_transform.pixel_height,
                )

            if is_dask:
                # Slice the dask array for this chunk
                r_end = row_offset + chunk_h
                c_end = col_offset + chunk_w
                chunk_data = raw[row_offset:r_end, col_offset:c_end]

                task = dask.delayed(_write_single_tile)(
                    chunk_data, tile_path, tile_gt, epsg, wkt_fallback,
                    nodata, compression, compression_level,
                    tile_size, predictor, bigtiff)
                delayed_tasks.append(task)
            else:
                # Numpy: slice and write directly
                chunk_data = np_arr[row_offset:row_offset + chunk_h,
                                    col_offset:col_offset + chunk_w]
                _write_single_tile(
                    chunk_data, tile_path, tile_gt, epsg, wkt_fallback,
                    nodata, compression, compression_level,
                    tile_size, predictor, bigtiff)

            col_offset += chunk_w
        row_offset += chunk_h

    # Execute all dask tasks
    if delayed_tasks:
        import dask
        dask.compute(*delayed_tasks, scheduler='synchronous')

    # Write VRT index with relative paths
    from ._vrt import write_vrt as _write_vrt_fn
    _write_vrt_fn(vrt_path, tile_paths, relative=True, nodata=nodata)


def read_geotiff_dask(source: str, *, dtype=None, chunks: int | tuple = 512,
                      overview_level: int | None = None,
                      name: str | None = None) -> xr.DataArray:
    """Read a GeoTIFF as a dask-backed DataArray for out-of-core processing.

    Each chunk is loaded lazily via windowed reads.

    Parameters
    ----------
    source : str
        File path.
    dtype : str, numpy.dtype, or None
        Cast each chunk to this dtype after reading. None keeps the
        file's native dtype. Float-to-int casts raise ValueError.
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

    # VRT files: delegate to read_vrt which handles chunks
    if source.lower().endswith('.vrt'):
        return read_vrt(source, dtype=dtype, name=name, chunks=chunks)

    # Metadata-only read: O(1) memory via mmap, no pixel decompression
    geo_info, full_h, full_w, file_dtype, n_bands = _read_geo_info(
        source, overview_level=overview_level)
    nodata = geo_info.nodata

    # Nodata masking promotes integer arrays to float64 (for NaN).
    # Validate against the effective dtype, not the raw file dtype.
    if nodata is not None and file_dtype.kind in ('u', 'i'):
        effective_dtype = np.dtype('float64')
    else:
        effective_dtype = file_dtype

    if dtype is not None:
        target_dtype = np.dtype(dtype)
        _validate_dtype_cast(effective_dtype, target_dtype)
    else:
        target_dtype = effective_dtype

    coords = _geo_to_coords(geo_info, full_h, full_w)

    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    attrs = {}
    if geo_info.crs_epsg is not None:
        attrs['crs'] = geo_info.crs_epsg
    if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
        attrs['raster_type'] = 'point'
    if nodata is not None:
        attrs['nodata'] = nodata

    if isinstance(chunks, int):
        ch_h = ch_w = chunks
    else:
        ch_h, ch_w = chunks

    # Graph-size guard. Each chunk becomes a delayed task whose Python graph
    # entry retains ~1KB. At very large chunk counts the graph itself OOMs
    # the driver before any read executes (30TB at chunks=256 => ~500M tasks
    # => ~500GB graph on host). Auto-scale chunks up to cap total task count.
    _MAX_DASK_CHUNKS = 1_000_000
    n_chunks = ((full_h + ch_h - 1) // ch_h) * ((full_w + ch_w - 1) // ch_w)
    if n_chunks > _MAX_DASK_CHUNKS:
        import math
        scale = math.sqrt(n_chunks / _MAX_DASK_CHUNKS)
        new_ch_h = int(math.ceil(ch_h * scale))
        new_ch_w = int(math.ceil(ch_w * scale))
        import warnings
        warnings.warn(
            f"read_geotiff_dask: requested chunks=({ch_h}, {ch_w}) on a "
            f"{full_h}x{full_w} image would produce {n_chunks} dask tasks, "
            f"exceeding the {_MAX_DASK_CHUNKS}-task cap. Auto-scaling to "
            f"chunks=({new_ch_h}, {new_ch_w}).",
            stacklevel=2,
        )
        ch_h, ch_w = new_ch_h, new_ch_w

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
                                     overview_level, nodata,
                                     band_arg,
                                     target_dtype=target_dtype if dtype is not None else None),
                shape=block_shape,
                dtype=target_dtype,
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
                         band, *, target_dtype=None):
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
        if target_dtype is not None:
            arr = arr.astype(target_dtype)
        return arr
    return _read()


def read_geotiff_gpu(source: str, *,
                     dtype=None,
                     overview_level: int | None = None,
                     name: str | None = None,
                     chunks: int | tuple | None = None,
                     max_pixels: int | None = None) -> xr.DataArray:
    """Read a GeoTIFF with GPU-accelerated decompression via Numba CUDA.

    Decompresses all tiles in parallel on the GPU and returns a
    CuPy-backed DataArray that stays on device memory. No CPU->GPU
    transfer needed for downstream xrspatial GPU operations.

    With ``chunks=``, returns a Dask+CuPy DataArray for out-of-core
    GPU pipelines.

    Requires: cupy, numba with CUDA support.

    Parameters
    ----------
    source : str
        File path.
    overview_level : int or None
        Overview level (0 = full resolution).
    chunks : int, tuple, or None
        If set, return a Dask-chunked CuPy DataArray. int for square
        chunks, (row, col) tuple for rectangular.
    name : str or None
        Name for the DataArray.
    max_pixels : int or None
        Maximum allowed pixel count (width * height * samples). None
        uses the default (~1 billion).

    Returns
    -------
    xr.DataArray
        CuPy-backed DataArray on GPU device.
    """
    try:
        import cupy
    except ImportError:
        raise ImportError(
            "cupy is required for GPU reads. "
            "Install it with: pip install cupy-cuda12x")

    from ._reader import _FileSource, _check_dimensions, MAX_PIXELS_DEFAULT
    from ._header import parse_header, parse_all_ifds
    from ._dtypes import tiff_dtype_to_numpy
    from ._geotags import extract_geo_info
    from ._gpu_decode import gpu_decode_tiles

    if max_pixels is None:
        max_pixels = MAX_PIXELS_DEFAULT

    # Parse metadata on CPU (fast, <1ms)
    src = _FileSource(source)
    data = src.read_all()

    try:
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)

        if len(ifds) == 0:
            raise ValueError("No IFDs found in TIFF file")

        ifd_idx = 0
        if overview_level is not None:
            ifd_idx = min(overview_level, len(ifds) - 1)
        ifd = ifds[ifd_idx]

        bps = ifd.bits_per_sample
        if isinstance(bps, tuple):
            bps = bps[0]
        file_dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
        geo_info = extract_geo_info(ifd, data, header.byte_order)

        if not ifd.is_tiled:
            # Fall back to CPU for stripped files
            src.close()
            arr_cpu, _ = read_to_array(source, overview_level=overview_level)
            arr_gpu = cupy.asarray(arr_cpu)
            coords = _geo_to_coords(geo_info, arr_gpu.shape[0], arr_gpu.shape[1])
            if name is None:
                import os
                name = os.path.splitext(os.path.basename(source))[0]
            attrs = {}
            if geo_info.crs_epsg is not None:
                attrs['crs'] = geo_info.crs_epsg
            if dtype is not None:
                target = np.dtype(dtype)
                _validate_dtype_cast(np.dtype(str(arr_gpu.dtype)), target)
                arr_gpu = arr_gpu.astype(target)
            return xr.DataArray(arr_gpu, dims=['y', 'x'],
                                coords=coords, name=name, attrs=attrs)

        offsets = ifd.tile_offsets
        byte_counts = ifd.tile_byte_counts
        compression = ifd.compression
        predictor = ifd.predictor
        samples = ifd.samples_per_pixel
        tw = ifd.tile_width
        th = ifd.tile_height
        width = ifd.width
        height = ifd.height

        if tw <= 0 or th <= 0:
            raise ValueError(
                f"Invalid tile dimensions: TileWidth={tw}, TileLength={th}")

        _check_dimensions(width, height, samples, max_pixels)
        # A single tile's decoded bytes must also fit under the pixel budget.
        _check_dimensions(tw, th, samples, max_pixels)

    finally:
        src.close()

    # GPU decode: try GDS (SSD→GPU direct) first, then CPU mmap path
    from ._gpu_decode import gpu_decode_tiles_from_file
    arr_gpu = None

    try:
        arr_gpu = gpu_decode_tiles_from_file(
            source, offsets, byte_counts,
            tw, th, width, height,
            compression, predictor, file_dtype, samples,
        )
    except Exception:
        pass

    if arr_gpu is None:
        # Fallback: extract tiles via CPU mmap, then GPU decode
        src2 = _FileSource(source)
        data2 = src2.read_all()
        try:
            compressed_tiles = [
                bytes(data2[offsets[i]:offsets[i] + byte_counts[i]])
                for i in range(len(offsets))
            ]
        finally:
            src2.close()

    if arr_gpu is None:
        try:
            arr_gpu = gpu_decode_tiles(
                compressed_tiles,
                tw, th, width, height,
                compression, predictor, file_dtype, samples,
            )
        except (ValueError, Exception):
            # Unsupported compression -- fall back to CPU then transfer
            arr_cpu, _ = read_to_array(source, overview_level=overview_level)
            arr_gpu = cupy.asarray(arr_cpu)

    if dtype is not None:
        target = np.dtype(dtype)
        _validate_dtype_cast(np.dtype(str(arr_gpu.dtype)), target)
        arr_gpu = arr_gpu.astype(target)

    # Build DataArray
    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    coords = _geo_to_coords(geo_info, height, width)

    attrs = {}
    if geo_info.crs_epsg is not None:
        attrs['crs'] = geo_info.crs_epsg
    if geo_info.crs_wkt is not None:
        attrs['crs_wkt'] = geo_info.crs_wkt

    if arr_gpu.ndim == 3:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(arr_gpu.shape[2])
    else:
        dims = ['y', 'x']

    result = xr.DataArray(arr_gpu, dims=dims, coords=coords,
                          name=name, attrs=attrs)

    if chunks is not None:
        if isinstance(chunks, int):
            chunk_dict = {'y': chunks, 'x': chunks}
        else:
            chunk_dict = {'y': chunks[0], 'x': chunks[1]}
        result = result.chunk(chunk_dict)

    return result


def write_geotiff_gpu(data, path: str, *,
                      crs: int | str | None = None,
                      nodata=None,
                      compression: str = 'zstd',
                      compression_level: int | None = None,
                      tile_size: int = 256,
                      predictor: bool = False,
                      cog: bool = False,
                      overview_levels: list[int] | None = None,
                      overview_resampling: str = 'mean') -> None:
    """Write a CuPy-backed DataArray as a GeoTIFF with GPU compression.

    Tiles are extracted and compressed on the GPU via nvCOMP, then
    assembled into a TIFF file on CPU. The CuPy array stays on device
    throughout compression -- only the compressed bytes transfer to CPU
    for file writing.

    When ``cog=True``, generates overview pyramids on GPU and writes a
    Cloud Optimized GeoTIFF with all IFDs at the file start for
    efficient range-request access.

    Falls back to CPU compression if nvCOMP is not available.

    Parameters
    ----------
    data : xr.DataArray (CuPy-backed) or cupy.ndarray
        2D raster on GPU.
    path : str
        Output file path.
    crs : int, str, or None
        EPSG code or WKT string.
    nodata : float, int, or None
        NoData value.
    compression : str
        'zstd' (default, fastest on GPU), 'deflate', 'jpeg', or 'none'.
        JPEG uses nvJPEG when available, falling back to Pillow.
    compression_level : int or None
        Compression effort level. Accepted for API compatibility but
        currently ignored -- nvCOMP does not expose level control.
    tile_size : int
        Tile size in pixels (default 256).
    predictor : bool
        Apply horizontal differencing predictor.
    cog : bool
        Write as Cloud Optimized GeoTIFF with overviews.
    overview_levels : list[int] or None
        Overview decimation factors (e.g. [2, 4, 8]). Only used when
        cog=True. If None and cog=True, auto-generates levels by
        halving until the smallest overview fits in a single tile.
    overview_resampling : str
        Resampling method for overviews: 'mean' (default), 'nearest',
        'min', 'max', 'median', or 'mode'.
    """
    try:
        import cupy
    except ImportError:
        raise ImportError("cupy is required for GPU writes")

    from ._gpu_decode import gpu_compress_tiles, make_overview_gpu
    from ._writer import (
        _compression_tag, _assemble_tiff, _write_bytes,
        GeoTransform as _GT,
    )
    from ._dtypes import numpy_to_tiff_dtype

    # Extract array and metadata
    geo_transform = None
    epsg = None
    raster_type = 1

    if isinstance(crs, int):
        epsg = crs
    elif isinstance(crs, str):
        epsg = _wkt_to_epsg(crs)

    if isinstance(data, xr.DataArray):
        arr = data.data
        # Handle Dask arrays: compute to materialize
        if hasattr(arr, 'compute'):
            arr = arr.compute()
        # Now arr should be CuPy or numpy
        if hasattr(arr, 'get'):
            pass  # CuPy array, already on GPU
        else:
            arr = cupy.asarray(np.asarray(arr))  # numpy -> GPU

        geo_transform = _coords_to_transform(data)
        if epsg is None:
            epsg = data.attrs.get('crs')
        if nodata is None:
            nodata = data.attrs.get('nodata')
        if data.attrs.get('raster_type') == 'point':
            raster_type = RASTER_PIXEL_IS_POINT
    else:
        if hasattr(data, 'compute'):
            data = data.compute()  # Dask -> CuPy or numpy
        if hasattr(data, 'device'):
            arr = data  # already CuPy
        elif hasattr(data, 'get'):
            arr = data  # CuPy
        else:
            arr = cupy.asarray(np.asarray(data))  # numpy/list -> GPU

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")

    height, width = arr.shape[:2]
    samples = arr.shape[2] if arr.ndim == 3 else 1
    np_dtype = np.dtype(str(arr.dtype))  # cupy dtype -> numpy dtype

    comp_tag = _compression_tag(compression)
    pred_val = 2 if predictor else 1

    def _gpu_compress_to_part(gpu_arr, w, h, spp):
        """Compress a GPU array into a (stub, w, h, offsets, counts, tiles) tuple."""
        compressed = gpu_compress_tiles(
            gpu_arr, tile_size, tile_size, w, h,
            comp_tag, pred_val, np_dtype, spp)
        rel_off = []
        bc = []
        off = 0
        for tile in compressed:
            rel_off.append(off)
            bc.append(len(tile))
            off += len(tile)
        stub = np.empty((1, 1, spp) if spp > 1 else (1, 1), dtype=np_dtype)
        return (stub, w, h, rel_off, bc, compressed)

    # Full resolution
    parts = [_gpu_compress_to_part(arr, width, height, samples)]

    # Overview generation
    if cog:
        if overview_levels is None:
            overview_levels = []
            oh, ow = height, width
            while oh > tile_size and ow > tile_size:
                oh //= 2
                ow //= 2
                if oh > 0 and ow > 0:
                    overview_levels.append(len(overview_levels) + 1)

        current = arr
        for _ in overview_levels:
            current = make_overview_gpu(current, method=overview_resampling)
            oh, ow = current.shape[:2]
            parts.append(_gpu_compress_to_part(current, ow, oh, samples))

    file_bytes = _assemble_tiff(
        width, height, np_dtype, comp_tag, predictor, True, tile_size,
        parts, geo_transform, epsg, nodata,
        is_cog=(cog and len(parts) > 1),
        raster_type=raster_type)

    _write_bytes(file_bytes, path)


def read_vrt(source: str, *, dtype=None, window=None,
             band: int | None = None,
             name: str | None = None,
             chunks: int | tuple | None = None,
             gpu: bool = False,
             max_pixels: int | None = None) -> xr.DataArray:
    """Read a GDAL Virtual Raster Table (.vrt) into an xarray.DataArray.

    The VRT's source GeoTIFFs are read via windowed reads and assembled
    into a single array.

    Parameters
    ----------
    source : str
        Path to the .vrt file.
    dtype : str, numpy.dtype, or None
        Cast the result to this dtype after reading. None keeps the
        file's native dtype. Float-to-int casts raise ValueError.
    window : tuple or None
        (row_start, col_start, row_stop, col_stop) for windowed reading.
    band : int or None
        Band index (0-based). None returns all bands.
    name : str or None
        Name for the DataArray.
    chunks : int, tuple, or None
        If set, return a Dask-chunked DataArray. int for square chunks,
        (row, col) tuple for rectangular.
    gpu : bool
        If True, return a CuPy-backed DataArray on GPU.

    Returns
    -------
    xr.DataArray
        NumPy, Dask, CuPy, or Dask+CuPy backed depending on options.
    """
    from ._vrt import read_vrt as _read_vrt_internal

    arr, vrt = _read_vrt_internal(source, window=window, band=band,
                                   max_pixels=max_pixels)

    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    # Build coordinates from GeoTransform
    gt = vrt.geo_transform
    if gt is not None:
        origin_x, res_x, _, origin_y, _, res_y = gt
        if window is not None:
            r0, c0, r1, c1 = window
            r0 = max(0, r0)
            c0 = max(0, c0)
        else:
            r0, c0 = 0, 0
        height, width = arr.shape[:2]
        x = np.arange(width, dtype=np.float64) * res_x + origin_x + (c0 + 0.5) * res_x
        y = np.arange(height, dtype=np.float64) * res_y + origin_y + (r0 + 0.5) * res_y
        coords = {'y': y, 'x': x}
    else:
        coords = {}

    attrs = {}
    if vrt.crs_wkt:
        epsg = _wkt_to_epsg(vrt.crs_wkt)
        if epsg is not None:
            attrs['crs'] = epsg
        attrs['crs_wkt'] = vrt.crs_wkt
    if vrt.bands:
        nodata = vrt.bands[0].nodata
        if nodata is not None:
            attrs['nodata'] = nodata

    # Transfer to GPU if requested
    if gpu:
        import cupy
        arr = cupy.asarray(arr)

    if dtype is not None:
        target = np.dtype(dtype)
        _validate_dtype_cast(np.dtype(str(arr.dtype)), target)
        arr = arr.astype(target)

    if arr.ndim == 3:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(arr.shape[2])
    else:
        dims = ['y', 'x']

    result = xr.DataArray(arr, dims=dims, coords=coords, name=name, attrs=attrs)

    # Chunk for Dask (or Dask+CuPy if gpu=True)
    if chunks is not None:
        if isinstance(chunks, int):
            chunk_dict = {'y': chunks, 'x': chunks}
        else:
            chunk_dict = {'y': chunks[0], 'x': chunks[1]}
        result = result.chunk(chunk_dict)

    return result


def write_vrt(vrt_path: str, source_files: list[str], **kwargs) -> str:
    """Generate a VRT file that mosaics multiple GeoTIFF tiles.

    Parameters
    ----------
    vrt_path : str
        Output .vrt file path.
    source_files : list of str
        Paths to the source GeoTIFF files.
    **kwargs
        relative, crs_wkt, nodata -- see _vrt.write_vrt.

    Returns
    -------
    str
        Path to the written VRT file.
    """
    from ._vrt import write_vrt as _write_vrt_internal
    return _write_vrt_internal(vrt_path, source_files, **kwargs)


def plot_geotiff(da: xr.DataArray, **kwargs):
    """Plot a DataArray using its embedded colormap if present.

    Deprecated: use ``da.xrs.plot()`` instead.
    """
    return da.xrs.plot(**kwargs)
