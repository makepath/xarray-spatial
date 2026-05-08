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

    Returned coords are pixel-center values in either raster type, matching
    xarray convention. The raw GeoTransform (origin and pixel size) is
    preserved separately on the DataArray as a rasterio-style 6-tuple in
    ``attrs['transform']``: ``(pixel_width, 0, origin_x, 0, pixel_height,
    origin_y)``. ``to_geotiff`` prefers that attr over recomputing the
    transform from the coord arrays, which avoids float drift on
    fractional-precision rasters.

    When the file carries no GeoTIFF tags (``has_georef=False``), fall back
    to integer pixel coordinates 0..N-1 instead of inventing fractional
    values from the default unit transform.
    """
    if not getattr(geo_info, 'has_georef', True):
        return {
            'y': np.arange(height, dtype=np.int64),
            'x': np.arange(width, dtype=np.int64),
        }
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


def _transform_tuple(geo_info) -> tuple | None:
    """Return the rasterio-style 6-tuple for a GeoInfo's transform.

    Format: ``(pixel_width, 0.0, origin_x, 0.0, pixel_height, origin_y)``.

    This matches ``rasterio.Affine.to_gdal()``-adjacent ordering used by
    rioxarray's ``rio.transform()`` output. Storing the tuple on the
    DataArray lets ``to_geotiff`` reproduce the source GeoTransform
    byte-for-byte, side-stepping float drift in the y/x coord arrays.
    """
    if geo_info is None:
        return None
    t = geo_info.transform
    if t is None:
        return None
    return (
        float(t.pixel_width), 0.0, float(t.origin_x),
        0.0, float(t.pixel_height), float(t.origin_y),
    )


def _transform_from_attr(attr_val) -> 'GeoTransform | None':
    """Build a GeoTransform from an ``attrs['transform']`` value.

    Accepts a 6-tuple ``(a, b, c, d, e, f)`` (rasterio Affine ordering;
    ``b`` and ``d`` are ignored, only axis-aligned affines round-trip),
    a 6-tuple GDAL ordering ``(c, a, b, f, d, e)`` is NOT accepted, or
    a ``GeoTransform`` instance. Returns None for anything else.
    """
    if attr_val is None:
        return None
    if isinstance(attr_val, GeoTransform):
        return attr_val
    try:
        seq = tuple(attr_val)
    except TypeError:
        return None
    if len(seq) != 6:
        return None
    try:
        a, _b, c, _d, e, f = (float(x) for x in seq)
    except (TypeError, ValueError):
        return None
    return GeoTransform(
        origin_x=c, origin_y=f, pixel_width=a, pixel_height=e,
    )


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


def _read_geo_info(source, *, overview_level: int | None = None):
    """Read only the geographic metadata and image dimensions from a GeoTIFF.

    Returns (geo_info, height, width, dtype, n_bands) without reading pixel
    data.  Uses mmap for header-only access on string paths; for file-like
    inputs it reads the bytes directly. O(1) memory regardless of file size
    when a path is supplied.

    Parameters
    ----------
    source : str or binary file-like
        Path or any object with ``read``/``seek``.
    overview_level : int or None
        Overview IFD index (0 = full resolution).
    """
    from ._dtypes import tiff_dtype_to_numpy
    from ._geotags import extract_geo_info
    from ._header import parse_all_ifds, parse_header, select_overview_ifd
    from ._reader import _coerce_path, _is_file_like

    source = _coerce_path(source)
    if _is_file_like(source):
        # File-like: read its full bytes; we don't try to mmap arbitrary
        # buffers because they may not back a real file descriptor.
        try:
            cur = source.tell()
        except (OSError, AttributeError):
            cur = 0
        source.seek(0)
        data = source.read()
        try:
            source.seek(cur)
        except (OSError, AttributeError):
            pass
        close_data = False
    elif isinstance(source, str):
        with open(source, 'rb') as f:
            import mmap
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        close_data = True
    else:
        raise TypeError(
            "source must be a str path or binary file-like, "
            f"got {type(source).__name__}")
    try:
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        if not ifds:
            raise ValueError("No IFDs found in TIFF file")
        ifd = select_overview_ifd(ifds, overview_level)
        geo_info = extract_geo_info(ifd, data, header.byte_order)
        bps = ifd.bits_per_sample
        if isinstance(bps, tuple):
            bps = bps[0]
        file_dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
        n_bands = ifd.samples_per_pixel if ifd.samples_per_pixel > 1 else 0
        return geo_info, ifd.height, ifd.width, file_dtype, n_bands
    finally:
        if close_data:
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




def open_geotiff(source, *, dtype=None, window=None,
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
    source : str or binary file-like
        File path, HTTP URL, cloud URI (s3://, gs://, az://), or a
        binary file-like object (e.g. ``io.BytesIO``) with read+seek.
        VRT, dask-chunked, GPU, and remote-URL paths require a string;
        in-memory file-like buffers go through the eager numpy reader.
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

    Notes
    -----
    The CRS is stored as an int EPSG code in ``attrs['crs']`` whenever the
    file's GeoKeys carry a recognized EPSG. Files whose CRS can only be
    expressed as WKT keep the WKT in ``attrs['crs_wkt']`` and leave
    ``attrs['crs']`` unset. ``to_geotiff`` accepts either an int EPSG or a
    WKT string in ``attrs['crs']`` for backward compatibility.

    The file's GeoTransform is also surfaced as ``attrs['transform']``,
    a rasterio-style 6-tuple
    ``(pixel_width, 0, origin_x, 0, pixel_height, origin_y)``. ``to_geotiff``
    uses this attr verbatim when present, falling back to recomputing the
    transform from the y/x coord arrays only when it is missing. The attr
    is what makes write -> read -> write -> read round-trips bit-stable for
    rasters with fractional pixel sizes or origins.

    Integer rasters with a nodata sentinel are silently promoted to
    ``float64`` with NaN replacing the sentinel so downstream NaN-aware
    code works uniformly. Pass ``dtype=...`` to keep the source dtype
    (the cast will fail with ``ValueError`` for float-to-int because that
    is lossy in a way users rarely intend; cast explicitly after read if
    you need it).
    """
    from ._reader import _coerce_path

    source = _coerce_path(source)

    # VRT files (string paths only -- VRT XML references other files on disk)
    if isinstance(source, str) and source.lower().endswith('.vrt'):
        return read_vrt(source, dtype=dtype, window=window, band=band,
                        name=name, chunks=chunks, gpu=gpu,
                        max_pixels=max_pixels)

    # File-like buffers don't support the GPU or dask code paths because
    # those re-open the source by path from worker tasks or device-side
    # readers. Reject early with a clear message.
    if not isinstance(source, str):
        if gpu:
            raise ValueError(
                "gpu=True is not supported for file-like sources. "
                "Pass a path string instead.")
        if chunks is not None:
            raise ValueError(
                "chunks=... (dask) is not supported for file-like sources. "
                "Pass a path string instead.")

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
        # Derive from source path. File-like buffers don't have a path,
        # so leave name unset rather than fabricating one.
        if isinstance(source, str):
            import os
            name = os.path.splitext(os.path.basename(source))[0]

    attrs = {}
    if geo_info.crs_epsg is not None:
        attrs['crs'] = geo_info.crs_epsg
    if geo_info.crs_wkt is not None:
        attrs['crs_wkt'] = geo_info.crs_wkt
    if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
        attrs['raster_type'] = 'point'

    # Preserve the source GeoTransform verbatim. For a windowed read the
    # origin shifts to the window's top-left pixel so the transform stays
    # consistent with the returned y/x coords.
    src_t = geo_info.transform
    if src_t is not None:
        if window is not None:
            r0, c0, _r1, _c1 = window
            origin_x_w = float(src_t.origin_x) + c0 * float(src_t.pixel_width)
            origin_y_w = float(src_t.origin_y) + r0 * float(src_t.pixel_height)
            attrs['transform'] = (
                float(src_t.pixel_width), 0.0, origin_x_w,
                0.0, float(src_t.pixel_height), origin_y_w,
            )
        else:
            attrs['transform'] = _transform_tuple(geo_info)

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

    # Friendly accessors for a few common pass-through tags. The raw
    # entry stays in attrs['extra_tags'] so the writer can re-emit the
    # exact bytes; users who tweak these convenience attrs can rely on
    # to_geotiff to fold the new value into extra_tags before write.
    if geo_info.image_description is not None:
        attrs['image_description'] = geo_info.image_description
    if geo_info.extra_samples is not None:
        attrs['extra_samples'] = geo_info.extra_samples

    # Resolution / DPI metadata
    if geo_info.x_resolution is not None:
        attrs['x_resolution'] = geo_info.x_resolution
    if geo_info.y_resolution is not None:
        attrs['y_resolution'] = geo_info.y_resolution
    if geo_info.resolution_unit is not None:
        _unit_names = {1: 'none', 2: 'inch', 3: 'centimeter'}
        attrs['resolution_unit'] = _unit_names.get(
            geo_info.resolution_unit, str(geo_info.resolution_unit))

    # Attach palette colormap for indexed-color TIFFs. The normalized
    # RGBA triples drive matplotlib display; the raw uint16 ColorMap
    # tag value lives in attrs['extra_tags'] for round-trip and is
    # exposed here as attrs['colormap'] for convenience.
    if geo_info.colormap is not None:
        try:
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(geo_info.colormap, name='tiff_palette')
            attrs['cmap'] = cmap
            attrs['colormap_rgba'] = geo_info.colormap
        except ImportError:
            # matplotlib not available -- store raw RGBA tuples only
            attrs['colormap_rgba'] = geo_info.colormap

    # Raw uint16 ColorMap tag value (3 * 2**bps entries, R-then-G-then-B)
    if geo_info.extra_tags is not None:
        for _tag_id, _tt, _tc, _tv in geo_info.extra_tags:
            if _tag_id == 320:  # TAG_COLORMAP
                attrs['colormap'] = _tv
                break

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

# Names accepted by ``compression=`` in :func:`to_geotiff`.  Kept in sync with
# ``_compression_tag`` in ``_writer.py``.  Validated up-front so users see a
# friendly error rather than the deeper traceback from ``_compression_tag``.
_VALID_COMPRESSIONS = (
    'none', 'deflate', 'lzw', 'jpeg', 'packbits', 'zstd', 'lz4',
    'jpeg2000', 'j2k', 'lerc',
)


# TIFF type ids needed when synthesizing extra_tags entries from attrs.
_TIFF_BYTE = 1
_TIFF_ASCII = 2
_TIFF_SHORT = 3


def _merge_friendly_extra_tags(extra_tags_list, attrs: dict) -> list | None:
    """Combine ``attrs['extra_tags']`` with friendly tag attrs.

    Synthesizes ``(tag_id, type_id, count, value)`` entries from
    ``attrs['image_description']`` (270, ASCII),
    ``attrs['extra_samples']`` (338, SHORT) and ``attrs['colormap']``
    (320, SHORT). An entry already present in ``extra_tags`` wins, so
    a verbatim round-trip stays byte-identical.
    """
    existing = list(extra_tags_list) if extra_tags_list else []
    seen_ids = {t[0] for t in existing}

    img_desc = attrs.get('image_description')
    if img_desc is not None and 270 not in seen_ids:
        s = str(img_desc)
        existing.append((270, _TIFF_ASCII, len(s) + 1, s))
        seen_ids.add(270)

    extra_samples = attrs.get('extra_samples')
    if extra_samples is not None and 338 not in seen_ids:
        try:
            vals = tuple(int(x) for x in extra_samples)
        except (TypeError, ValueError):
            vals = None
        if vals:
            value = vals if len(vals) > 1 else vals[0]
            existing.append((338, _TIFF_SHORT, len(vals), value))
            seen_ids.add(338)

    colormap = attrs.get('colormap')
    if colormap is not None and 320 not in seen_ids:
        try:
            cmap_vals = tuple(int(x) for x in colormap)
        except (TypeError, ValueError):
            cmap_vals = None
        if cmap_vals:
            value = cmap_vals if len(cmap_vals) > 1 else cmap_vals[0]
            existing.append((320, _TIFF_SHORT, len(cmap_vals), value))
            seen_ids.add(320)

    return existing or None


def to_geotiff(data: xr.DataArray | np.ndarray, path, *,
               crs: int | str | None = None,
               nodata=None,
               compression: str = 'zstd',
               compression_level: int | None = None,
               tiled: bool = True,
               tile_size: int = 256,
               predictor: bool | int = False,
               cog: bool = False,
               overview_levels: list[int] | None = None,
               overview_resampling: str = 'mean',
               bigtiff: bool | None = None,
               gpu: bool | None = None,
               streaming_buffer_bytes: int = 256 * 1024 * 1024,
               max_z_error: float = 0.0) -> None:
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
    path : str or binary file-like
        Output file path, or any object exposing a ``write`` method
        (e.g. ``io.BytesIO``). When a file-like is passed, the encoded
        TIFF bytes are written to that object once assembly completes.
        ``cog=True`` and ``.vrt`` outputs require a string path.
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
        Tile size in pixels (default 256). Ignored when ``tiled=False``;
        a warning is emitted if a non-default value is passed alongside
        strip mode.
    predictor : bool or int
        TIFF predictor. Accepted values:

        * ``False``, ``0``, or ``1`` -> no predictor.
        * ``True`` or ``2`` -> horizontal differencing (good for integer
          data; ``True`` and ``2`` are exactly equivalent).
        * ``3`` -> floating-point predictor (float dtypes only; typically
          gives better deflate/zstd ratios on float data than predictor 2).
    cog : bool
        Write as Cloud Optimized GeoTIFF.
    overview_levels : list[int] or None
        Overview decimation factors. Only used when cog=True.
    overview_resampling : str
        Resampling method for overviews: 'mean' (default), 'nearest',
        'min', 'max', 'median', 'mode', or 'cubic'.
    gpu : bool or None
        Force GPU compression. None (default) auto-detects CuPy data.
    streaming_buffer_bytes : int
        Soft cap on bytes materialised per dask compute call when
        streaming a dask-backed DataArray. Defaults to 256 MB. Wide
        rasters whose tile-row exceeds this budget are split into
        horizontal segments. Ignored for numpy / CuPy / COG paths.
    max_z_error : float
        Per-pixel error budget for LERC compression. ``0.0`` (default)
        is lossless; larger values let the encoder approximate values
        within the bound, producing smaller files at the cost of accuracy
        bounded by ``abs(decoded - original) <= max_z_error``. Only used
        when ``compression='lerc'``; passing a non-zero value with any
        other codec raises ``ValueError``.
    """
    from ._reader import _coerce_path

    path = _coerce_path(path)

    # Up-front validation: catch bad compression names before they reach
    # any of the deeper write paths (streaming, GPU, VRT, COG) where the
    # error surfaces from _compression_tag with a less obvious traceback.
    if isinstance(compression, str):
        if compression.lower() not in _VALID_COMPRESSIONS:
            raise ValueError(
                f"Unknown compression {compression!r}. "
                f"Valid options: {list(_VALID_COMPRESSIONS)}.")
        # JPEG-in-TIFF (compression=7) requires the JPEGTables tag (347)
        # carrying the abbreviated quantization/Huffman tables. The
        # current encoder writes a self-contained JFIF stream per
        # tile/strip and omits JPEGTables, which makes the resulting
        # files unreadable by libtiff / GDAL / rasterio: they reject the
        # tile data with "TIFFReadEncodedStrip() failed". The internal
        # reader round-trips because Pillow re-decodes the JFIF stream
        # directly, masking the interop break. Refuse the write rather
        # than emit files no other tool can decode. See issue tracking
        # the proper JPEGTables fix for re-enabling this codec.
        if compression.lower() == 'jpeg':
            raise ValueError(
                "compression='jpeg' is not supported: the encoder writes "
                "self-contained JFIF streams without the required "
                "JPEGTables tag (347), so other readers (libtiff, GDAL, "
                "rasterio) reject the file. Use 'deflate', 'zstd', or "
                "'lzw' instead.")

    # max_z_error only applies to LERC; reject negative values and reject
    # non-zero values paired with any other codec so the caller learns the
    # parameter was ignored before bytes hit disk.
    if max_z_error < 0:
        raise ValueError(
            f"max_z_error must be >= 0, got {max_z_error}")
    if max_z_error != 0 and (
            not isinstance(compression, str)
            or compression.lower() != 'lerc'):
        raise ValueError(
            "max_z_error is only valid with compression='lerc'")

    # File-like (BytesIO etc.) destinations: the streaming, GPU, COG, and
    # VRT writers all need a real filesystem path (atomic rename, overview
    # passes, sidecar writes). Reject those combos up front so the user
    # gets a clear error instead of a deep traceback.
    _path_is_file_like = (not isinstance(path, str)) and hasattr(path, 'write')
    if _path_is_file_like:
        if cog:
            raise ValueError(
                "cog=True is not supported for file-like destinations. "
                "Pass a string path or write to BytesIO without cog=True.")
    elif not isinstance(path, str):
        raise TypeError(
            f"path must be a str or a binary file-like with a write() "
            f"method, got {type(path).__name__}")

    # tile_size only applies to tiled output; warn if the caller passed a
    # non-default size alongside strip mode (it would otherwise be silently
    # ignored).
    if not tiled and tile_size != 256:
        import warnings
        warnings.warn(
            f"tile_size={tile_size} is ignored when tiled=False "
            "(strip layout). Pass tiled=True to use tile_size, or drop "
            "tile_size to silence this warning.",
            stacklevel=2,
        )

    # VRT tiled output (string paths only -- VRT writes a real .vrt file
    # plus per-tile GeoTIFFs to a directory)
    if isinstance(path, str) and path.lower().endswith('.vrt'):
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
                         bigtiff=bigtiff,
                         max_z_error=max_z_error)
        return

    # Auto-detect GPU data and dispatch to write_geotiff_gpu
    use_gpu = gpu if gpu is not None else _is_gpu_data(data)
    if use_gpu and _path_is_file_like:
        # write_geotiff_gpu's nvCOMP path materialises tile parts and then
        # calls _write_bytes(path), which would write at the buffer's
        # current cursor without truncating. More importantly, the GPU
        # path was never tested with file-like destinations; refuse rather
        # than silently produce something untested.
        raise ValueError(
            "gpu=True is not supported for file-like destinations. "
            "Pass a string path (or set gpu=False).")
    if use_gpu:
        # GPU writer uses nvCOMP and does not support LERC; refuse rather
        # than silently dropping the requested error budget.
        if max_z_error != 0:
            raise ValueError(
                "max_z_error is not supported on the GPU writer "
                "(nvCOMP has no LERC backend). Use the CPU path "
                "(gpu=False) or omit max_z_error.")
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

        # Extract metadata from DataArray attrs (no materialisation needed).
        # Prefer attrs['transform'] (from open_geotiff) over the coord-derived
        # transform: that path is bit-stable across round-trips, while
        # _coords_to_transform can drift on fractional pixel sizes because
        # x[1] - x[0] is computed in float64 from already-rounded coords.
        if geo_transform is None:
            geo_transform = _transform_from_attr(data.attrs.get('transform'))
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
        # Fold friendly attrs into extra_tags so a user-edited
        # attrs['image_description'] / ['extra_samples'] / ['colormap']
        # actually reaches the file. Existing entries with the same tag id
        # win, which keeps verbatim round-trips byte-stable.
        extra_tags_list = _merge_friendly_extra_tags(
            extra_tags_list, data.attrs)
        x_res = data.attrs.get('x_resolution')
        y_res = data.attrs.get('y_resolution')
        unit_str = data.attrs.get('resolution_unit')
        if unit_str is not None:
            _unit_ids = {'none': 1, 'inch': 2, 'centimeter': 3}
            res_unit = _unit_ids.get(str(unit_str), None)

        # Dask-backed: stream tiles to avoid materialising the full array.
        # COG requires overviews from the full array, so it falls through
        # to the eager path. Streaming write needs a real filesystem path
        # (it builds a temp file then atomic-renames); for file-like
        # destinations we materialise eagerly and assemble in-memory.
        if hasattr(raw, 'dask') and not cog and not _path_is_file_like:
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
                streaming_buffer_bytes=streaming_buffer_bytes,
                max_z_error=max_z_error,
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
        max_z_error=max_z_error,
    )


def _write_single_tile(chunk_data, path, geo_transform, epsg, wkt,
                       nodata, compression, compression_level,
                       tile_size, predictor, bigtiff,
                       max_z_error: float = 0.0):
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
          bigtiff=bigtiff,
          max_z_error=max_z_error)


def _write_vrt_tiled(data, vrt_path, *, crs=None, nodata=None,
                     compression='zstd', compression_level=None,
                     tile_size=256, predictor: bool | int = False,
                     bigtiff=None, max_z_error: float = 0.0):
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
        geo_transform = _transform_from_attr(data.attrs.get('transform'))
        if geo_transform is None:
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
                    tile_size, predictor, bigtiff, max_z_error)
                delayed_tasks.append(task)
            else:
                # Numpy: slice and write directly
                chunk_data = np_arr[row_offset:row_offset + chunk_h,
                                    col_offset:col_offset + chunk_w]
                _write_single_tile(
                    chunk_data, tile_path, tile_gt, epsg, wkt_fallback,
                    nodata, compression, compression_level,
                    tile_size, predictor, bigtiff, max_z_error)

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

    from ._reader import _coerce_path

    source = _coerce_path(source)

    # ``read_geotiff`` already routes ``.vrt`` to ``read_vrt`` before
    # reaching here, so this branch is only hit when ``read_geotiff_dask``
    # is called directly with a VRT path. Keep it as a defensive fallback
    # rather than letting the windowed-read path try to parse VRT XML as
    # TIFF bytes. ``read_vrt`` is the single source of truth for VRT.
    if isinstance(source, str) and source.lower().endswith('.vrt'):
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
    transform_tuple = _transform_tuple(geo_info)
    if transform_tuple is not None:
        attrs['transform'] = transform_tuple

    if isinstance(chunks, int):
        ch_h = ch_w = chunks
    else:
        ch_h, ch_w = chunks

    # Graph-size guard. Each chunk becomes a delayed task whose Python graph
    # entry retains ~1KB. At very large chunk counts the graph itself OOMs
    # the driver before any read executes (30TB at chunks=256 => ~500M tasks
    # => ~500GB graph on host). Refuse anything past the cap and ask the
    # caller to pick a chunk size, rather than silently rescaling -- the
    # rescaled chunks may not align with the user's downstream pipeline.
    _MAX_DASK_CHUNKS = 50_000
    n_chunks = ((full_h + ch_h - 1) // ch_h) * ((full_w + ch_w - 1) // ch_w)
    if n_chunks > _MAX_DASK_CHUNKS:
        import math
        scale = math.sqrt(n_chunks / _MAX_DASK_CHUNKS)
        suggested_h = int(math.ceil(ch_h * scale))
        suggested_w = int(math.ceil(ch_w * scale))
        raise ValueError(
            f"read_geotiff_dask: chunks=({ch_h}, {ch_w}) on a "
            f"{full_h}x{full_w} image would produce {n_chunks:,} dask "
            f"tasks, exceeding the {_MAX_DASK_CHUNKS:,}-task cap. Pass a "
            f"larger chunks=... value explicitly (e.g. chunks="
            f"({suggested_h}, {suggested_w}) keeps the task count under "
            "the cap)."
        )

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

    from ._reader import (
        _FileSource, _check_dimensions, MAX_PIXELS_DEFAULT, _coerce_path,
    )
    from ._header import (
        parse_header, parse_all_ifds, select_overview_ifd, validate_tile_layout,
    )
    from ._dtypes import tiff_dtype_to_numpy
    from ._geotags import extract_geo_info
    from ._gpu_decode import gpu_decode_tiles

    source = _coerce_path(source)

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

        # Skip mask IFDs (NewSubfileType bit 2)
        ifd = select_overview_ifd(ifds, overview_level)

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
            t_tuple = _transform_tuple(geo_info)
            if t_tuple is not None:
                attrs['transform'] = t_tuple
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

        # Reject malformed TIFFs whose declared tile grid exceeds the
        # supplied TileOffsets length. The GPU tile-assembly kernel would
        # read OOB otherwise. See issue #1219.
        validate_tile_layout(ifd)

    finally:
        src.close()

    # GPU decode: try GDS (SSD→GPU direct) first, then CPU mmap path.
    # Sparse tiles (byte_count == 0) are unsupported on the GPU pipeline;
    # the CPU reader fills them with nodata and copies onto the GPU.
    has_sparse_tile = any(bc == 0 for bc in byte_counts)
    if has_sparse_tile:
        arr_cpu, _ = read_to_array(source, overview_level=overview_level)
        arr_gpu = cupy.asarray(arr_cpu)
    else:
        from ._gpu_decode import gpu_decode_tiles_from_file
        arr_gpu = None

        try:
            arr_gpu = gpu_decode_tiles_from_file(
                source, offsets, byte_counts,
                tw, th, width, height,
                compression, predictor, file_dtype, samples,
                byte_order=header.byte_order,
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
                byte_order=header.byte_order,
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
    t_tuple = _transform_tuple(geo_info)
    if t_tuple is not None:
        attrs['transform'] = t_tuple

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
                      predictor: bool | int = False,
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
    predictor : bool or int
        TIFF predictor. ``False``/``0``/``1`` -> none, ``True``/``2`` ->
        horizontal differencing, ``3`` -> floating-point predictor
        (float dtypes only).
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
        normalize_predictor,
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
    pred_val = normalize_predictor(predictor, np_dtype, comp_tag)

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

    # Overview generation -- mirrors the CPU writer's 8-level cap.
    if cog:
        if overview_levels is None:
            from ._writer import _MAX_OVERVIEW_LEVELS
            overview_levels = []
            oh, ow = height, width
            while (oh > tile_size and ow > tile_size and
                   len(overview_levels) < _MAX_OVERVIEW_LEVELS):
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
        width, height, np_dtype, comp_tag, pred_val, True, tile_size,
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

    Notes
    -----
    Like ``open_geotiff``, the CRS lands as an int EPSG in
    ``attrs['crs']`` when the VRT's WKT resolves to a known EPSG code.
    Otherwise ``attrs['crs']`` stays unset and ``attrs['crs_wkt']`` carries
    the original WKT. The source GeoTransform is preserved as a
    rasterio-style 6-tuple in ``attrs['transform']``.
    """
    from ._reader import _coerce_path
    from ._vrt import read_vrt as _read_vrt_internal

    source = _coerce_path(source)

    arr, vrt = _read_vrt_internal(source, window=window, band=band,
                                   max_pixels=max_pixels)

    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    # Build coordinates from GeoTransform.
    #
    # GDAL's convention: when AREA_OR_POINT=Area (default) the
    # GeoTransform origin is the top-left corner of pixel (0, 0) and
    # pixel centers need a half-pixel shift.  When AREA_OR_POINT=Point
    # the origin already *is* the center of pixel (0, 0) and no shift
    # is applied.  This mirrors ``_geo_to_coords`` for non-VRT reads.
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
        if vrt.raster_type == 'point':
            x_shift = c0 * res_x
            y_shift = r0 * res_y
        else:
            x_shift = (c0 + 0.5) * res_x
            y_shift = (r0 + 0.5) * res_y
        x = np.arange(width, dtype=np.float64) * res_x + origin_x + x_shift
        y = np.arange(height, dtype=np.float64) * res_y + origin_y + y_shift
        coords = {'y': y, 'x': x}
    else:
        coords = {}

    attrs = {}
    if vrt.crs_wkt:
        epsg = _wkt_to_epsg(vrt.crs_wkt)
        if epsg is not None:
            attrs['crs'] = epsg
        attrs['crs_wkt'] = vrt.crs_wkt
    if vrt.raster_type == 'point':
        attrs['raster_type'] = 'point'
    if vrt.bands:
        nodata = vrt.bands[0].nodata
        if nodata is not None:
            attrs['nodata'] = nodata

    # Surface the source GeoTransform in the same rasterio ordering used
    # by open_geotiff: (pixel_width, 0, origin_x, 0, pixel_height, origin_y).
    # vrt.geo_transform is GDAL ordering, so reorder. For a windowed read
    # the origin shifts by (col_offset * res_x, row_offset * res_y).
    if gt is not None:
        if window is not None:
            r0w, c0w, _r1w, _c1w = window
            r0w = max(0, r0w)
            c0w = max(0, c0w)
        else:
            r0w = c0w = 0
        origin_x_out = float(origin_x) + c0w * float(res_x)
        origin_y_out = float(origin_y) + r0w * float(res_y)
        attrs['transform'] = (
            float(res_x), 0.0, origin_x_out,
            0.0, float(res_y), origin_y_out,
        )

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
    relative : bool, optional
        Store source paths relative to the VRT file (default True).
    crs_wkt : str or None, optional
        CRS as a WKT string. If None, the CRS is taken from the first
        source GeoTIFF.
    nodata : float or None, optional
        NoData value. If None, taken from the first source GeoTIFF.

    Returns
    -------
    str
        Path to the written VRT file.

    Notes
    -----
    Only the keyword arguments listed above are accepted. Passing any
    other keyword raises ``TypeError`` from the underlying writer.
    """
    from ._vrt import write_vrt as _write_vrt_internal
    return _write_vrt_internal(vrt_path, source_files, **kwargs)


def plot_geotiff(da: xr.DataArray, **kwargs):
    """Plot a DataArray using its embedded colormap if present.

    Deprecated: use ``da.xrs.plot()`` instead.
    """
    return da.xrs.plot(**kwargs)
