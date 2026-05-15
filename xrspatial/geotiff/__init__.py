"""Lightweight GeoTIFF/COG reader and writer.

No GDAL dependency -- uses only numpy, numba, xarray, and the standard library.

Public API
----------
open_geotiff(source, ...)
    Read a GeoTIFF, COG, or VRT file to an xarray.DataArray. Auto-dispatches
    to the GPU, dask, or numpy backend based on the ``gpu`` and ``chunks``
    kwargs.
read_geotiff_gpu(source, ...)
    GPU-only read returning a CuPy-backed DataArray. ``open_geotiff(...,
    gpu=True)`` calls this internally; use the explicit name when you want
    the strict-mode failure semantics (``on_gpu_failure='strict'``) or want
    to bypass auto-dispatch.
read_geotiff_dask(source, ...)
    Dask-only read returning a windowed lazy DataArray. ``open_geotiff(...,
    chunks=N)`` calls this internally.
read_vrt(source, ...)
    Read a GDAL Virtual Raster Table (.vrt). ``open_geotiff`` routes ``.vrt``
    paths here automatically; the explicit entry point is useful for
    callers that already know they have a VRT.
to_geotiff(data, path, ...)
    Write an xarray.DataArray as a GeoTIFF or COG. Auto-dispatches to GPU
    when the data is CuPy-backed.
write_geotiff_gpu(data, path, ...)
    GPU-only writer using nvCOMP. ``to_geotiff(..., gpu=True)`` calls this
    internally.
write_vrt(vrt_path, source_files, ...)
    Generate a VRT mosaic XML from a list of GeoTIFF files.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from typing import BinaryIO

from ._coords import (
    _BAND_DIM_NAMES,
    coords_from_geo_info as _coords_from_geo_info,
    coords_from_pixel_geometry as _coords_from_pixel_geometry,
    transform_tuple_from_pixel_geometry as _transform_tuple_from_pixel_geometry,
    geo_to_coords as _geo_to_coords,
    transform_tuple as _transform_tuple,
    transform_from_attr as _transform_from_attr,
    coords_to_transform as _coords_to_transform,
)
from ._geotags import GeoTransform, RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT
from ._reader import UnsafeURLError
# ``read_to_array`` is internal: it is used by ``open_geotiff`` and the
# GPU fallback below but is not in ``__all__`` or the module-level
# Public API docstring. Bind it under a leading-underscore name so it
# does not leak into ``xrspatial.geotiff``'s public namespace. Tests
# and internal callers that genuinely need it can import directly from
# ``xrspatial.geotiff._reader``. See issue #1708.
from ._crs import _resolve_crs_to_wkt, _wkt_to_epsg
from ._reader import read_to_array as _read_to_array
from ._runtime import (
    GeoTIFFFallbackWarning,
    _CRS_WKT_DEPRECATED_SENTINEL,
    _GPU_DEPRECATED_SENTINEL,
    _MISSING_SOURCES_SENTINEL,
    _ON_GPU_FAILURE_SENTINEL,
    _X_DIM_NAMES,
    _Y_DIM_NAMES,
    _geotiff_strict_mode,
    _gpu_fallback_warning_message,
)
from ._writer import write

# All names below are part of the supported public API. ``plot_geotiff``
# is intentionally omitted: it is deprecated in favour of ``da.xrs.plot()``
# and emits a ``DeprecationWarning`` when called.
__all__ = [
    'GeoTIFFFallbackWarning',
    'UnsafeURLError',
    'open_geotiff',
    'read_geotiff_gpu',
    'read_geotiff_dask',
    'read_vrt',
    'to_geotiff',
    'write_geotiff_gpu',
    'write_vrt',
]


def _validate_3d_writer_dims(dims) -> None:
    """Reject ambiguous 3D writer inputs (issue #1812).

    The writer interprets a 3D DataArray as either ``(band, y, x)`` or
    ``(y, x, band)``. ``data.dims[0] in _BAND_DIM_NAMES`` decides which
    branch fires the ``moveaxis``. Anything else (e.g. ``('time', 'y', 'x')``)
    used to fall through silently: the writer kept the leading axis as
    the spatial ``y`` axis and the result was a TIFF with the leading
    axis values laid out along ``y`` (silent data corruption -- on
    read-back the array round-tripped with a swapped shape).

    Refuse the ambiguous case at the entry point. The message tells the
    caller exactly how to fix the input (rename to one of
    ``_BAND_DIM_NAMES`` or transpose to ``(y, x, band)``).
    """
    if len(dims) != 3:
        return
    d0, d1, d2 = dims
    band_layout = (d0 in _BAND_DIM_NAMES
                   and d1 in _Y_DIM_NAMES
                   and d2 in _X_DIM_NAMES)
    yxb_layout = (d0 in _Y_DIM_NAMES
                  and d1 in _X_DIM_NAMES
                  and d2 in _BAND_DIM_NAMES)
    if band_layout or yxb_layout:
        return
    # Bare (y, x, *) or (*, y, x) where the third dim is unnamed but
    # spatial -- the writer's old behaviour treats the non-spatial axis
    # as bands. Accept that only when the unknown dim is in the band
    # position (last), which matches how raw numpy callers typically
    # build a band-last array.
    if d0 in _Y_DIM_NAMES and d1 in _X_DIM_NAMES:
        return
    raise ValueError(
        f"3D writer input has ambiguous dims {dims!r}. Expected "
        f"(band, y, x) or (y, x, band); accepted band-dim aliases are "
        f"{_BAND_DIM_NAMES} and spatial aliases are y={_Y_DIM_NAMES} / "
        f"x={_X_DIM_NAMES}. Rename the non-spatial dim to 'band' or "
        f"transpose the array so spatial dims come first (e.g. "
        f"``da.transpose('y', 'x', {dims[0]!r})``). The writer cannot "
        f"infer which axis is the band axis from arbitrary dim names "
        f"and would otherwise silently treat the leading axis as the "
        f"spatial y axis (issue #1812)."
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
    from ._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
    from ._geotags import extract_geo_info_with_overview_inheritance
    from ._header import parse_all_ifds, parse_header, select_overview_ifd
    from ._reader import (
        _CloudSource, _coerce_path, _is_file_like, _is_fsspec_uri,
        _parse_cog_http_meta,
    )

    source = _coerce_path(source)
    if isinstance(source, str) and _is_fsspec_uri(source):
        # fsspec URI (s3://, gs://, az://, memory://, ...): use the
        # bounded-prefetch metadata parser instead of downloading the
        # full remote object. ``_parse_cog_http_meta`` only needs
        # ``read_range`` on the source, which ``_CloudSource`` provides;
        # it grows a small range buffer until the IFD chain resolves
        # (capped by ``MAX_HTTP_HEADER_BYTES``). Avoids the
        # whole-file fetch that would otherwise happen on every
        # ``open_geotiff(..., chunks=...)`` graph build for a large COG.
        _src = _CloudSource(source)
        try:
            _header, _ifd, geo_info, _ = _parse_cog_http_meta(
                _src, overview_level=overview_level)
        finally:
            _src.close()
        bps = resolve_bits_per_sample(_ifd.bits_per_sample)
        file_dtype = tiff_dtype_to_numpy(bps, _ifd.sample_format)
        n_bands = (
            _ifd.samples_per_pixel if _ifd.samples_per_pixel > 1 else 0
        )
        # Stash photometric + samples_per_pixel so the dask graph builder
        # can detect MinIsWhite and invert ``geo_info.nodata`` before
        # binding it into the chunk closure (#1809).
        geo_info._ifd_photometric = _ifd.photometric
        geo_info._ifd_samples_per_pixel = _ifd.samples_per_pixel
        return geo_info, _ifd.height, _ifd.width, file_dtype, n_bands
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
        # Inherit georef from the level-0 IFD when the overview itself
        # has no geokeys (issue #1640). Pass-through for level 0.
        geo_info = extract_geo_info_with_overview_inheritance(
            ifd, ifds, data, header.byte_order)
        bps = resolve_bits_per_sample(ifd.bits_per_sample)
        file_dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
        n_bands = ifd.samples_per_pixel if ifd.samples_per_pixel > 1 else 0
        # Stash photometric + samples_per_pixel so the dask graph builder
        # can detect MinIsWhite and invert ``geo_info.nodata`` before
        # binding it into the chunk closure (#1809).
        geo_info._ifd_photometric = ifd.photometric
        geo_info._ifd_samples_per_pixel = ifd.samples_per_pixel
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


def _populate_attrs_from_geo_info(attrs: dict, geo_info, *, window=None) -> None:
    """Populate ``attrs`` with all GeoTIFF metadata from ``geo_info``.

    Centralised so the eager numpy, dask, and GPU read paths emit the
    same attrs keys for the same input file. Mutates ``attrs`` in place.

    The ``nodata`` attr is intentionally NOT set here because each caller
    sets it next to its own nodata-masking step (the value's presence in
    attrs signals "this array has been NaN-masked").

    ``window`` is a ``(r0, c0, r1, c1)`` tuple for windowed reads; when
    set, the emitted ``attrs['transform']`` shifts the origin to the
    window's top-left. The eager path and the dask path (since #1561,
    which threads ``window=`` through ``read_geotiff_dask``) both pass
    the outer window through this helper so the resulting DataArray
    advertises the windowed transform. The GPU path does not currently
    expose a windowed read, so it passes ``window=None``.
    """
    if geo_info.crs_epsg is not None:
        attrs['crs'] = geo_info.crs_epsg
    if geo_info.crs_wkt is not None:
        attrs['crs_wkt'] = geo_info.crs_wkt
    if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
        attrs['raster_type'] = 'point'

    src_t = geo_info.transform
    # Skip the transform attr for files where no GeoTIFF transform tags
    # (ModelTransformation, ModelPixelScale, or ModelTiepoint) are
    # present, signalled by ``has_georef=False``. GeoKeys / CRS metadata
    # can still be present in that case. The default unit
    # ``GeoTransform`` is a struct placeholder, not real georef --
    # emitting it leaks an identity transform into attrs and confuses
    # downstream code that expects ``'transform' in attrs`` to mean
    # "this raster has a georef transform" (#1710).
    has_georef = getattr(geo_info, 'has_georef', True)
    if src_t is not None and has_georef:
        attrs['transform'] = _transform_tuple_from_pixel_geometry(
            src_t.origin_x, src_t.origin_y,
            src_t.pixel_width, src_t.pixel_height,
            window=window,
        )

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
    if geo_info.vertical_epsg is not None:
        attrs['vertical_crs'] = geo_info.vertical_epsg
    if geo_info.vertical_citation is not None:
        attrs['vertical_citation'] = geo_info.vertical_citation
    if geo_info.vertical_units is not None:
        attrs['vertical_units'] = geo_info.vertical_units

    if geo_info.gdal_metadata is not None:
        attrs['gdal_metadata'] = geo_info.gdal_metadata
    if geo_info.gdal_metadata_xml is not None:
        attrs['gdal_metadata_xml'] = geo_info.gdal_metadata_xml

    if geo_info.extra_tags is not None:
        attrs['extra_tags'] = geo_info.extra_tags
    if geo_info.image_description is not None:
        attrs['image_description'] = geo_info.image_description
    if geo_info.extra_samples is not None:
        attrs['extra_samples'] = geo_info.extra_samples

    if geo_info.x_resolution is not None:
        attrs['x_resolution'] = geo_info.x_resolution
    if geo_info.y_resolution is not None:
        attrs['y_resolution'] = geo_info.y_resolution
    if geo_info.resolution_unit is not None:
        _unit_names = {1: 'none', 2: 'inch', 3: 'centimeter'}
        attrs['resolution_unit'] = _unit_names.get(
            geo_info.resolution_unit, str(geo_info.resolution_unit))

    if geo_info.colormap is not None:
        try:
            from matplotlib.colors import ListedColormap
            attrs['cmap'] = ListedColormap(
                geo_info.colormap, name='tiff_palette')
            attrs['colormap_rgba'] = geo_info.colormap
        except ImportError:
            attrs['colormap_rgba'] = geo_info.colormap

    if geo_info.extra_tags is not None:
        for _tag_id, _tt, _tc, _tv in geo_info.extra_tags:
            if _tag_id == 320:  # TAG_COLORMAP
                attrs['colormap'] = _tv
                break


def open_geotiff(source: str | BinaryIO, *,
                 dtype: str | np.dtype | None = None,
                 window: tuple | None = None,
                 overview_level: int | None = None,
                 band: int | None = None,
                 name: str | None = None,
                 chunks: int | tuple | None = None,
                 gpu: bool = False,
                 max_pixels: int | None = None,
                 on_gpu_failure: str = _ON_GPU_FAILURE_SENTINEL,
                 missing_sources: str = _MISSING_SOURCES_SENTINEL,
                 ) -> xr.DataArray:
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
    on_gpu_failure : {'auto', 'strict'}, optional
        Forwarded to ``read_geotiff_gpu`` when ``gpu=True``. Controls
        whether GPU decode failures fall back to CPU (``'auto'``,
        default) or re-raise the original exception (``'strict'``).
        Passing this kwarg with ``gpu=False`` raises ``ValueError``
        because the policy only applies to the GPU pipeline. See
        ``read_geotiff_gpu`` for the full description.
    missing_sources : {'raise', 'warn'}, optional
        Forwarded to ``read_vrt`` when the source is a ``.vrt`` file.
        When the caller does not pass this kwarg, the public
        ``read_vrt`` default applies (``'raise'`` since #1860).
        ``'raise'`` fails immediately on an unreadable backing source.
        ``'warn'`` is the opt-in lenient mode: emit
        ``GeoTIFFFallbackWarning``, record ``attrs['vrt_holes']``, and
        return a partial mosaic. Passing this kwarg with a non-VRT
        source raises ``ValueError`` because the policy only applies to
        the VRT pipeline. See ``read_vrt`` for the full description.

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

    # ``on_gpu_failure`` is GPU-only. Reject it up front for CPU/dask paths
    # rather than silently dropping it once dispatch is decided -- callers
    # otherwise have no way to learn that the policy is being ignored.
    # ``gpu=False`` (the default) on a ``.vrt`` source still routes through
    # ``read_vrt`` below which has no GPU-failure concept, so the same
    # rejection rule applies there.
    if on_gpu_failure is not _ON_GPU_FAILURE_SENTINEL and not gpu:
        raise ValueError(
            "on_gpu_failure only applies when gpu=True. "
            "Pass gpu=True to enable the GPU pipeline, or drop "
            "on_gpu_failure to keep the default CPU path.")

    # ``missing_sources`` is VRT-only. Reject it up front when the source
    # is not a ``.vrt`` file so callers learn the policy is being ignored
    # instead of getting a silent drop -- same pattern ``on_gpu_failure``
    # uses above for the GPU-only kwarg, and the same class of dispatcher
    # silently-drops-backend-kwarg bug #1561 / #1605 / #1685 / #1795 fixed
    # for the other VRT/GPU kwargs. See issue #1810.
    missing_sources_passed = (
        missing_sources is not _MISSING_SOURCES_SENTINEL)
    _is_vrt_source = (
        isinstance(source, str) and source.lower().endswith('.vrt'))
    if missing_sources_passed and not _is_vrt_source:
        raise ValueError(
            "missing_sources only applies to VRT sources. "
            "Pass a .vrt path to enable the VRT pipeline, or drop "
            "missing_sources to keep the default GeoTIFF path.")

    # VRT files (string paths only -- VRT XML references other files on disk)
    if _is_vrt_source:
        # ``read_vrt`` does not accept ``overview_level`` (the VRT XML
        # references its own source files; overview selection would need
        # to apply to each one). Silently dropping the kwarg was the same
        # class of bug issue #1561 fixed for the dask and GPU dispatchers,
        # so refuse the combination up front rather than handing the
        # caller a full-resolution mosaic with no warning. See issue #1685.
        # ``overview_level=0`` is documented as "full resolution" (the
        # default), so treat it as a no-op the same as ``None`` rather
        # than rejecting a kwarg value the caller could have omitted.
        if overview_level not in (None, 0):
            raise ValueError(
                "overview_level is not supported for VRT sources. "
                "VRT references its own source files; pass overview_level "
                "to open_geotiff on a .tif source, or drop the kwarg.")
        # ``on_gpu_failure`` only routes through ``read_geotiff_gpu``.
        # ``read_vrt`` has no analogous failure policy, so any value the
        # caller supplied alongside a VRT source would be silently lost.
        # The ``gpu=False`` branch is already rejected above; this catches
        # the ``gpu=True, source.endswith('.vrt')`` case the earlier check
        # lets through.
        if on_gpu_failure is not _ON_GPU_FAILURE_SENTINEL:
            raise ValueError(
                "on_gpu_failure is not supported for VRT sources. "
                "VRT reads do not go through the GPU decoder pipeline; "
                "drop the kwarg or call read_geotiff_gpu directly on a "
                ".tif source.")
        vrt_kwargs = {}
        if missing_sources_passed:
            vrt_kwargs['missing_sources'] = missing_sources
        return read_vrt(source, dtype=dtype, window=window, band=band,
                        name=name, chunks=chunks, gpu=gpu,
                        max_pixels=max_pixels, **vrt_kwargs)

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
        gpu_kwargs = {}
        if on_gpu_failure is not _ON_GPU_FAILURE_SENTINEL:
            gpu_kwargs['on_gpu_failure'] = on_gpu_failure
        return read_geotiff_gpu(source, dtype=dtype,
                                overview_level=overview_level,
                                window=window, band=band,
                                name=name, chunks=chunks,
                                max_pixels=max_pixels,
                                **gpu_kwargs)

    # Dask path (CPU)
    if chunks is not None:
        return read_geotiff_dask(source, dtype=dtype, chunks=chunks,
                                 overview_level=overview_level,
                                 window=window, band=band,
                                 max_pixels=max_pixels, name=name)

    kwargs = {}
    if max_pixels is not None:
        kwargs['max_pixels'] = max_pixels

    # ``read_to_array`` validates ``window`` against the selected IFD's
    # extent and raises ``ValueError`` for out-of-bounds windows with
    # the same message format as the dask path's pre-flight validator
    # in :func:`read_geotiff_dask`. That keeps the two backends in sync
    # on the contract without forcing a second metadata parse here. See
    # issue #1634.
    arr, geo_info = _read_to_array(
        source, window=window,
        overview_level=overview_level, band=band,
        **kwargs,
    )

    height, width = arr.shape[:2]
    coords = _geo_to_coords(geo_info, height, width)

    if window is not None:
        # Adjust coordinates for windowed read. ``read_to_array`` rejected
        # out-of-bounds windows above, so ``r0/c0/r1/c1`` are guaranteed
        # in-range here (no clamp needed). For files with no GeoTIFF
        # tags (``has_georef=False``), the default unit transform is
        # not real, so fall back to integer pixel coords matching the
        # ``_geo_to_coords`` shortcut taken on full reads. See #1710.
        coords = _coords_from_geo_info(
            geo_info, height, width, window=window,
        )

    if name is None:
        # Derive from source path. File-like buffers don't have a path,
        # so leave name unset rather than fabricating one.
        if isinstance(source, str):
            import os
            name = os.path.splitext(os.path.basename(source))[0]

    attrs = {}
    _populate_attrs_from_geo_info(attrs, geo_info, window=window)

    # Apply nodata mask: replace nodata sentinel values with NaN.
    # ``arr`` came from ``read_to_array``, which returns a freshly
    # allocated buffer from ``_read_tiles`` / ``_read_strips`` (and is
    # ``np.ascontiguousarray``-wrapped if the orientation tag triggered
    # a remap). Nothing else holds a reference here, so the in-place
    # write is safe; an extra ``arr.copy()`` would just double peak
    # memory for a multi-MB raster.
    nodata = geo_info.nodata
    if nodata is not None:
        attrs['nodata'] = nodata
        # When the reader applied MinIsWhite, the sentinel-equality mask
        # must compare against the inverted sentinel value (issue #1809).
        # ``read_to_array`` / ``_read_cog_http`` stash that value on
        # ``geo_info._mask_nodata``; fall back to the original sentinel
        # on non-MinIsWhite files.
        mask_nodata = getattr(geo_info, '_mask_nodata', nodata)
        if arr.dtype.kind == 'f':
            if mask_nodata is not None and not np.isnan(mask_nodata):
                arr[arr == arr.dtype.type(mask_nodata)] = np.nan
        elif arr.dtype.kind in ('u', 'i'):
            # Integer arrays: convert to float to represent NaN.
            # An out-of-range sentinel (e.g. uint16 file with
            # GDAL_NODATA="-9999") cannot match any decoded pixel, so the
            # mask would be all-False -- skip the cast that would otherwise
            # raise OverflowError and leave the array unchanged. A
            # non-finite sentinel ("NaN" / "Inf" GDAL_NODATA strings) also
            # cannot match an integer pixel, so the ``int(nodata)`` cast
            # below would raise ValueError; gate on ``np.isfinite`` first
            # to mirror ``_resolve_masked_fill`` / ``_sparse_fill_value``
            # in ``_reader.py`` (#1774). A fractional sentinel (e.g.
            # ``GDAL_NODATA="3.5"`` on a ``uint16`` file) also cannot match
            # an integer pixel; ``int(3.5)`` would truncate to 3 and
            # silently mask a real pixel value, so gate on
            # ``float(nodata).is_integer()`` as well (mirrors the
            # ``_writer.py`` / ``_vrt.py`` pattern used for #1564 / #1616).
            # attrs['nodata'] still carries the raw sentinel so a write
            # round-trip preserves the tag.
            if (mask_nodata is not None
                    and np.isfinite(mask_nodata)
                    and float(mask_nodata).is_integer()):
                nodata_int = int(mask_nodata)
                info = np.iinfo(arr.dtype)
                if info.min <= nodata_int <= info.max:
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


def _apply_nodata_mask_gpu(arr_gpu, nodata):
    """Replace nodata sentinel values with NaN on a CuPy array.

    Mirrors the CPU eager path in :func:`open_geotiff` (lines around the
    ``# Apply nodata mask`` comment). Float arrays get NaN substituted in
    place of the sentinel; integer arrays are promoted to float64 with NaN
    where the sentinel was. NaN nodata on a float array is a no-op (the
    sentinel already matches NaN-aware code).

    Returns the (possibly promoted, possibly nodata-masked) CuPy array.
    The caller is responsible for setting ``attrs['nodata']`` so the
    sentinel is still discoverable downstream.
    """
    import cupy

    if nodata is None:
        return arr_gpu
    arr_dtype = np.dtype(str(arr_gpu.dtype))
    if arr_dtype.kind == 'f':
        if not np.isnan(nodata):
            sentinel = arr_dtype.type(nodata)
            arr_gpu = cupy.where(arr_gpu == sentinel,
                                 arr_dtype.type('nan'), arr_gpu)
        return arr_gpu
    if arr_dtype.kind in ('u', 'i'):
        # Out-of-range sentinels (e.g. uint16 + GDAL_NODATA="-9999") cannot
        # match any decoded pixel; skip the cast that would otherwise raise
        # OverflowError. A non-finite sentinel ("NaN" / "Inf" GDAL_NODATA
        # strings) also cannot match an integer pixel and would raise
        # ValueError on ``int(nodata)``; gate on ``np.isfinite`` first to
        # mirror ``_resolve_masked_fill`` in ``_reader.py`` (#1774). A
        # fractional sentinel (e.g. ``"3.5"`` on a ``uint16`` file) also
        # cannot match an integer pixel and ``int(3.5)`` would truncate
        # to 3, silently masking a real pixel value; gate on
        # ``float(nodata).is_integer()`` as well (mirrors the
        # ``_writer.py`` / ``_vrt.py`` pattern used for #1564 / #1616).
        # attrs['nodata'] is still set by the caller so the original
        # sentinel survives a write round-trip.
        if not (np.isfinite(nodata) and float(nodata).is_integer()):
            return arr_gpu
        nodata_int = int(nodata)
        info = np.iinfo(arr_dtype)
        if not (info.min <= nodata_int <= info.max):
            return arr_gpu
        sentinel = arr_dtype.type(nodata_int)
        mask = arr_gpu == sentinel
        if bool(mask.any().item()):
            arr_gpu = arr_gpu.astype(cupy.float64)
            arr_gpu = cupy.where(mask, cupy.float64('nan'), arr_gpu)
        return arr_gpu
    return arr_gpu


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


def _resolve_nodata_attr(attrs: dict):
    """Resolve a NoData sentinel from DataArray attrs.

    xrspatial's own readers always emit ``attrs['nodata']`` (a scalar),
    so that key is checked first for a clean intra-library round-trip.
    Falls back to two ecosystem conventions on miss:

    * ``attrs['nodatavals']`` -- rioxarray's per-band tuple. Returns
      the first entry that is not None, not non-numeric, and not NaN.
      In practice this is band 0 for almost every real file; the skip
      logic only matters when band 0 is missing a sentinel (NaN /
      None) while a later band declares one. Bands with mixed concrete
      sentinels are uncommon and would need an explicit ``nodata=``
      argument anyway.
    * ``attrs['_FillValue']`` -- CF-style xarray pipelines.

    Returns ``None`` when none of the keys carry a usable value. NaN
    entries in ``nodatavals`` are skipped rather than treated as a
    sentinel (NaN means "the float NaN is the sentinel", which is
    already the default and doesn't need a GDAL_NODATA tag).
    """
    nodata = attrs.get('nodata')
    if nodata is not None:
        return nodata

    vals = attrs.get('nodatavals')
    if vals is not None:
        try:
            seq = list(vals)
        except TypeError:
            seq = [vals]
        for v in seq:
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(fv):
                continue
            return v

    fill = attrs.get('_FillValue')
    if fill is not None:
        try:
            ffv = float(fill)
        except (TypeError, ValueError):
            return fill  # non-numeric -- pass through verbatim
        if np.isnan(ffv):
            return None
        return fill

    return None


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


# String identifiers (used in xrspatial attrs) -> TIFF ResolutionUnit tag ids.
_RESOLUTION_UNIT_IDS = {'none': 1, 'inch': 2, 'centimeter': 3}


def _extract_rich_tags(attrs: dict) -> dict:
    """Extract the rich-tag set forwarded by the writers to ``write(...)``.

    Centralises the bookkeeping shared by :func:`to_geotiff`,
    :func:`_write_vrt_tiled`, and :func:`write_geotiff_gpu`:

    * ``raster_type`` -- mapped from ``attrs['raster_type']`` ('point'
      becomes :data:`RASTER_PIXEL_IS_POINT`; everything else stays
      :data:`RASTER_PIXEL_IS_AREA`).
    * ``gdal_metadata_xml`` -- prefers ``attrs['gdal_metadata_xml']``;
      falls back to building XML from ``attrs['gdal_metadata']`` when
      it is a dict.
    * ``extra_tags`` -- ``attrs['extra_tags']`` folded with the friendly
      tag attrs (image_description / extra_samples / colormap) via
      :func:`_merge_friendly_extra_tags`.
    * ``x_resolution`` / ``y_resolution`` -- pass-through.
    * ``resolution_unit`` -- string label mapped to the integer tag id.

    Returns a kwargs dict ready to splat into ``write(...)``: every key
    matches the corresponding parameter name on
    :func:`xrspatial.geotiff._writer.write`.
    """
    raster_type = (RASTER_PIXEL_IS_POINT
                   if attrs.get('raster_type') == 'point'
                   else RASTER_PIXEL_IS_AREA)

    gdal_meta_xml = attrs.get('gdal_metadata_xml')
    if gdal_meta_xml is None:
        gdal_meta_dict = attrs.get('gdal_metadata')
        if isinstance(gdal_meta_dict, dict):
            from ._geotags import _build_gdal_metadata_xml
            gdal_meta_xml = _build_gdal_metadata_xml(gdal_meta_dict)

    extra_tags_list = _merge_friendly_extra_tags(
        attrs.get('extra_tags'), attrs)

    res_unit = None
    unit_str = attrs.get('resolution_unit')
    if unit_str is not None:
        res_unit = _RESOLUTION_UNIT_IDS.get(str(unit_str), None)

    return {
        'raster_type': raster_type,
        'gdal_metadata_xml': gdal_meta_xml,
        'extra_tags': extra_tags_list,
        'x_resolution': attrs.get('x_resolution'),
        'y_resolution': attrs.get('y_resolution'),
        'resolution_unit': res_unit,
    }


def _validate_tile_size(tile_size) -> None:
    """Validate ``tile_size`` for the tiled GeoTIFF writers.

    Shared by ``to_geotiff`` (when ``tiled=True``) and
    ``write_geotiff_gpu`` (always tiled) so the accepted types, the
    non-positive rejection, and the multiple-of-16 hint stay in lockstep.
    The tiled writer computes the tile grid as
    ``math.ceil(width / tile_size)``; ``tile_size=0`` hits
    ``ZeroDivisionError`` deep inside the writer, and negative values
    produce a nonsensical tile grid. The TIFF 6 spec also requires
    ``TileWidth`` and ``TileLength`` to be positive multiples of 16
    for broad interoperability with libtiff / GDAL strict readers; a
    value like 17 would otherwise round-trip through the in-repo
    reader but be rejected elsewhere.
    """
    if not isinstance(tile_size, (int, np.integer)) or isinstance(
            tile_size, bool):
        raise ValueError(
            f"tile_size must be a positive int, got "
            f"{tile_size!r} (type {type(tile_size).__name__}).")
    if tile_size <= 0:
        raise ValueError(
            f"tile_size must be a positive int, got tile_size={tile_size}.")
    if tile_size % 16 != 0:
        lower = (int(tile_size) // 16) * 16
        upper = lower + 16
        # ``lower`` is 0 for tile_size < 16; suppress it from the hint
        # because 0 is not a valid tile size on its own.
        if lower <= 0:
            hint = f"try tile_size={upper}"
        else:
            hint = f"try tile_size={lower} or tile_size={upper}"
        raise ValueError(
            f"tile_size must be a positive multiple of 16 (TIFF 6 "
            f"spec requirement for TileWidth/TileLength), got "
            f"tile_size={tile_size}; {hint}.")


def to_geotiff(data: xr.DataArray | np.ndarray,
               path: str | BinaryIO, *,
               crs: int | str | None = None,
               nodata: float | int | None = None,
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
               max_z_error: float = 0.0,
               photometric: str | int = 'auto') -> None:
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

        EPSG codes are strongly preferred for interop. The WKT-only
        path writes ``ProjectedCSType`` / ``GeographicType`` = 32767
        with the WKT stored in ``GTCitationGeoKey`` -- libgeotiff and
        GDAL can round-trip this but many other GeoTIFF readers treat
        the citation as a free-form name and lose the CRS. A
        ``UserWarning`` is emitted when the WKT-only path is taken.
        See issue #1768.
    nodata : float, int, or None
        NoData value.
    compression : str
        Codec name. One of ``'none'``, ``'deflate'``, ``'lzw'``,
        ``'jpeg'``, ``'packbits'``, ``'zstd'``, ``'lz4'``,
        ``'jpeg2000'`` (alias ``'j2k'``), or ``'lerc'``.
        ``'jpeg'`` is currently rejected on write because the encoder
        omits the JPEGTables tag and produced files do not round-trip
        through libtiff / GDAL / rasterio. Use ``'deflate'``, ``'zstd'``,
        or ``'lzw'`` instead. ``'lerc'`` accepts ``max_z_error`` for
        lossy compression with a bounded per-pixel error.
    compression_level : int or None
        Compression effort level. None uses each codec's default (6 for
        deflate/zstd). Valid ranges: deflate 1-9, zstd 1-22, lz4 0-16.
        Codecs without a level concept (lzw, packbits, jpeg) accept any
        value and ignore it.
    tiled : bool
        Use tiled layout (default True).
    tile_size : int
        Tile size in pixels (default 256). Must be a positive multiple
        of 16 when ``tiled=True``; this is a TIFF 6 spec requirement
        on TileWidth and TileLength for broad reader compatibility.
        Ignored when ``tiled=False``; a warning is emitted if a
        non-default value is passed alongside strip mode.
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
        Overview decimation factors relative to full resolution.
        Each entry must be a power-of-two integer >= 2, and the list
        must be strictly increasing (e.g. ``[2, 4, 8]`` writes
        overviews at 1/2, 1/4 and 1/8 of the full resolution).
        Invalid values raise ``ValueError``. Only used when ``cog=True``.
        If None and ``cog=True``, levels auto-generate as
        ``[2, 4, 8, ...]`` until the next halving would fall below
        ``tile_size`` (capped at 8 levels).
    overview_resampling : str
        Resampling method for overviews: 'mean' (default), 'nearest',
        'min', 'max', 'median', 'mode', or 'cubic'.
    bigtiff : bool or None
        Force BigTIFF (64-bit offsets). None (default) auto-promotes
        when the estimated file size would exceed the classic-TIFF 4 GB
        limit. Matches the same kwarg on ``write_geotiff_gpu``.
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
    photometric : str or int
        Photometric interpretation for the TIFF Photometric tag (262).

        * ``'auto'`` (default) -- MinIsBlack (1) for any band count.
          ExtraSamples for every band beyond the first is tagged ``0``
          (unspecified). Multispectral rasters (e.g. R, G, B, NIR)
          round-trip through this default without being silently
          labelled as RGB+alpha. Prior versions treated any 3+ band
          array as RGB and the 4th band as unassociated alpha -- the
          behaviour change is intentional (issue #1769).
        * ``'rgb'`` -- RGB (Photometric=2). Three colour bands; any
          additional bands are tagged ``0`` (unspecified).
        * ``'rgba'`` -- RGB with the 4th band tagged as unassociated
          alpha (TIFF ExtraSamples=2). Requires at least 4 bands.
        * ``'minisblack'`` or ``'miniswhite'`` -- grayscale; multi-band
          extras tagged ``0``.
        * An ``int`` -- written verbatim into Photometric for advanced
          callers (e.g. ``3`` for Palette, ``5`` for CMYK).

        A user-supplied ``extra_tags`` entry of ``(TAG_PHOTOMETRIC,
        ...)`` or ``(TAG_EXTRA_SAMPLES, ...)`` overrides the writer's
        chosen value; only these two tag ids are overridable so other
        auto-emitted tags such as ``ImageWidth`` or ``StripOffsets``
        remain protected.

    Raises
    ------
    ValueError
        If ``data.attrs['transform']`` is a rotated or skewed affine
        (``b != 0`` or ``d != 0`` in rasterio ``Affine`` ordering). The
        on-disk GeoTIFF is axis-aligned; reproject onto an axis-aligned
        grid first.
    ValueError
        If ``data`` is a 3D DataArray whose ``dims`` is not
        ``(band, y, x)`` or ``(y, x, band)`` (accepting the band-name
        aliases ``bands`` / ``channel`` and spatial-name aliases
        ``lat`` / ``lon`` / ``latitude`` / ``longitude`` / ``row`` /
        ``col``). A leading non-band dim such as ``time`` is rejected
        because the writer cannot infer the band axis from arbitrary
        names and used to silently treat the leading axis as ``y``
        (issue #1812).
    """
    from ._reader import _coerce_path

    path = _coerce_path(path)

    # tiled=False ignores tile_size, so only validate when tiled output
    # is requested. Shared with write_geotiff_gpu via
    # _validate_tile_size_arg so both writers keep identical validation.
    if tiled:
        _validate_tile_size_arg(tile_size)

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

    _is_vrt_path = (
        isinstance(path, str) and path.lower().endswith('.vrt'))

    # tile_size only applies to tiled output; warn if the caller passed a
    # non-default size alongside strip mode (it would otherwise be silently
    # ignored). The VRT path always tiles, so the warning would be
    # misleading there -- the VRT branch below rejects tiled=False up front
    # instead.
    if not tiled and tile_size != 256 and not _is_vrt_path:
        warnings.warn(
            f"tile_size={tile_size} is ignored when tiled=False "
            "(strip layout). Pass tiled=True to use tile_size, or drop "
            "tile_size to silence this warning.",
            stacklevel=2,
        )

    # VRT tiled output (string paths only -- VRT writes a real .vrt file
    # plus per-tile GeoTIFFs to a directory)
    if _is_vrt_path:
        if not tiled:
            raise ValueError(
                "tiled=False is not compatible with VRT output. "
                "VRT writes a directory of tiled GeoTIFFs; pass "
                "tiled=True or omit it.")
        # The early ``if tiled: _validate_tile_size_arg(tile_size)`` check
        # above already validates tile_size when tiled=True, but call it
        # here as well so the VRT path stays self-contained against future
        # changes to the early-validation gate (no-op on re-entry; either
        # raises or returns).
        _validate_tile_size_arg(tile_size)
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
                         max_z_error=max_z_error,
                         photometric=photometric)
        return

    # Auto-detect GPU data and dispatch to write_geotiff_gpu. ``gpu is
    # None`` is the implicit "use whatever fits the data" path; preserve
    # that distinction in the fallback warning below so users who never
    # set ``gpu=True`` are not told their explicit request was dropped.
    auto_detected_gpu = gpu is None
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
        # Strip output is not implemented on the GPU path; reject up
        # front rather than silently producing a tiled file.
        if not tiled:
            raise ValueError(
                "tiled=False is not supported on the GPU writer. "
                "Pass gpu=False or omit tiled=False.")
        try:
            write_geotiff_gpu(data, path, crs=crs, nodata=nodata,
                              compression=compression,
                              compression_level=compression_level,
                              tiled=tiled,
                              tile_size=tile_size,
                              predictor=predictor,
                              cog=cog,
                              overview_levels=overview_levels,
                              overview_resampling=overview_resampling,
                              bigtiff=bigtiff,
                              streaming_buffer_bytes=streaming_buffer_bytes,
                              photometric=photometric)
            return
        except ImportError as e:
            # ``write_geotiff_gpu`` raises ImportError when cupy itself
            # can't be imported. nvCOMP absence doesn't surface here:
            # ``_try_nvcomp_from_device_bufs`` returns None when the
            # library can't load, and the writer drops to CPU
            # compression internally instead of re-raising. Fall back
            # to the CPU writer with a typed warning so callers see
            # that gpu=True (or auto-detected CuPy data) didn't go
            # through. Strict mode re-raises so CI can fail loudly on
            # missing GPU stacks.
            if _geotiff_strict_mode():
                raise
            warnings.warn(
                _gpu_fallback_warning_message(auto_detected_gpu, e),
                GeoTIFFFallbackWarning,
                stacklevel=2,
            )
        except RuntimeError as e:
            # Only fall back when the message names a GPU-availability
            # problem; any other RuntimeError is a real bug in the GPU
            # writer and the broad ``except (ImportError, Exception)``
            # used to hide it from the user. Keep the keyword list
            # tight: nvCOMP / CUDA / no device / no GPU / cuInit cover
            # the realistic "no GPU present" failure modes without
            # masking, e.g., a CRS or compression error that happens to
            # raise RuntimeError. Strict mode re-raises in either case.
            _gpu_unavail_tokens = (
                'nvcomp', 'cuda', 'no device', 'no gpu', 'cuinit',
            )
            msg = str(e).lower()
            if not any(tok in msg for tok in _gpu_unavail_tokens):
                raise
            if _geotiff_strict_mode():
                raise
            warnings.warn(
                _gpu_fallback_warning_message(auto_detected_gpu, e),
                GeoTIFFFallbackWarning,
                stacklevel=2,
            )

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
            nodata = _resolve_nodata_attr(data.attrs)
        # Pull raster_type, gdal_metadata_xml, extra_tags (folded with
        # the friendly image_description / extra_samples / colormap
        # attrs), x/y_resolution, and resolution_unit via the shared
        # helper so all three writers stay in lockstep.
        _rich = _extract_rich_tags(data.attrs)
        raster_type = _rich['raster_type']
        gdal_meta_xml = _rich['gdal_metadata_xml']
        extra_tags_list = _rich['extra_tags']
        x_res = _rich['x_resolution']
        y_res = _rich['y_resolution']
        res_unit = _rich['resolution_unit']

        # Dask-backed: stream tiles to avoid materialising the full array.
        # COG requires overviews from the full array, so it falls through
        # to the eager path. Streaming write needs a real filesystem path
        # (it builds a temp file then atomic-renames); for file-like
        # destinations we materialise eagerly and assemble in-memory.
        if hasattr(raw, 'dask') and not cog and not _path_is_file_like:
            dask_arr = raw
            # Reject ambiguous 3D layouts at the entry point so a leading
            # non-band dim like ``('time', 'y', 'x')`` cannot silently
            # round-trip as a TIFF whose ``y`` axis carries the time
            # values (issue #1812).
            if raw.ndim == 3:
                _validate_3d_writer_dims(data.dims)
            # Handle band-first dimension order (band, y, x) -> (y, x, band)
            if raw.ndim == 3 and data.dims[0] in _BAND_DIM_NAMES:
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
                photometric=photometric,
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
        # Reject ambiguous 3D layouts (issue #1812). The validator runs
        # on ``data.dims`` (the original DataArray's dim names) rather
        # than on ``arr`` so the error fires before the move-axis even
        # for COG and file-like destinations that fall through here.
        if arr.ndim == 3:
            _validate_3d_writer_dims(data.dims)
        # Handle band-first dimension order (band, y, x) -> (y, x, band)
        if arr.ndim == 3 and data.dims[0] in _BAND_DIM_NAMES:
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
    #
    # The defensive ``arr.copy()`` here is intentional: ``arr`` may be
    # ``np.asarray(raw)`` of a caller-owned numpy DataArray (a view,
    # not a copy) or ``np.moveaxis(arr, 0, -1)`` of one (also a view).
    # Mutating without a copy would corrupt the user's input buffer.
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
        photometric=photometric,
    )


def _write_single_tile(chunk_data, path, geo_transform, epsg, wkt,
                       nodata, compression, compression_level,
                       tile_size, predictor, bigtiff,
                       max_z_error: float = 0.0,
                       raster_type: int = RASTER_PIXEL_IS_AREA,
                       x_resolution=None,
                       y_resolution=None,
                       resolution_unit=None,
                       gdal_metadata_xml=None,
                       extra_tags=None,
                       photometric: str | int = 'auto'):
    """Write a single tile GeoTIFF. Used by _write_vrt_tiled.

    Forwards the same rich-tag set that ``to_geotiff`` passes through to
    ``write`` (raster_type, x/y resolution, GDAL metadata, extra tags) so
    every per-tile file under a VRT carries the same metadata it would
    have received from a single-file ``to_geotiff(..., out.tif)`` write.
    Without this, ``to_geotiff(da, "out.vrt")`` silently drops everything
    except the per-tile geo_transform / crs / nodata. See issue #1606.
    """
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

    # Restore NaN to nodata sentinel.
    #
    # The defensive ``arr.copy()`` here is intentional: ``arr`` came
    # from ``np.asarray(chunk_data)`` where ``chunk_data`` may be a
    # caller-owned numpy buffer. Mutating without a copy would corrupt
    # the user's input.
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
          raster_type=raster_type,
          x_resolution=x_resolution,
          y_resolution=y_resolution,
          resolution_unit=resolution_unit,
          gdal_metadata_xml=gdal_metadata_xml,
          extra_tags=extra_tags,
          bigtiff=bigtiff,
          max_z_error=max_z_error,
          photometric=photometric)


def _write_vrt_tiled(data, vrt_path, *, crs=None, nodata=None,
                     compression='zstd', compression_level=None,
                     tile_size=256, predictor: bool | int = False,
                     bigtiff=None, max_z_error: float = 0.0,
                     photometric: str | int = 'auto'):
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
    raster_type = RASTER_PIXEL_IS_AREA
    x_res = None
    y_res = None
    res_unit = None
    gdal_meta_xml = None
    extra_tags_list = None

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
            # Use the same alias-aware resolver that to_geotiff /
            # write_geotiff_gpu apply so a rioxarray-style DataArray
            # (``attrs['nodatavals']``) or a CF-style one
            # (``attrs['_FillValue']``) round-trips through ``.vrt``
            # the same way it does through ``.tif``. Before this fix
            # the VRT path used ``attrs.get('nodata')`` directly and
            # silently dropped both aliases (issue #1606).
            nodata = _resolve_nodata_attr(data.attrs)
        geo_transform = _transform_from_attr(data.attrs.get('transform'))
        if geo_transform is None:
            geo_transform = _coords_to_transform(data)
        # Pull the same rich-tag set that to_geotiff forwards to
        # ``write`` so per-tile files under the VRT carry it too.
        _rich = _extract_rich_tags(data.attrs)
        raster_type = _rich['raster_type']
        gdal_meta_xml = _rich['gdal_metadata_xml']
        extra_tags_list = _rich['extra_tags']
        x_res = _rich['x_resolution']
        y_res = _rich['y_resolution']
        res_unit = _rich['resolution_unit']
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
                    tile_size, predictor, bigtiff, max_z_error,
                    raster_type=raster_type,
                    x_resolution=x_res,
                    y_resolution=y_res,
                    resolution_unit=res_unit,
                    gdal_metadata_xml=gdal_meta_xml,
                    extra_tags=extra_tags_list,
                    photometric=photometric)
                delayed_tasks.append(task)
            else:
                # Numpy: slice and write directly
                chunk_data = np_arr[row_offset:row_offset + chunk_h,
                                    col_offset:col_offset + chunk_w]
                _write_single_tile(
                    chunk_data, tile_path, tile_gt, epsg, wkt_fallback,
                    nodata, compression, compression_level,
                    tile_size, predictor, bigtiff, max_z_error,
                    raster_type=raster_type,
                    x_resolution=x_res,
                    y_resolution=y_res,
                    resolution_unit=res_unit,
                    gdal_metadata_xml=gdal_meta_xml,
                    extra_tags=extra_tags_list,
                    photometric=photometric)

            col_offset += chunk_w
        row_offset += chunk_h

    # Execute all dask tasks.
    #
    # Each delayed task is an independent ``_write_single_tile`` call on
    # a distinct output path, with no shared mutable Python state, so
    # the writes are embarrassingly parallel. Using ``scheduler='threads'``
    # lets zlib / zstd / LZW release the GIL during compression and the
    # OS coalesce concurrent writes; in a 256-tile zstd write on a
    # 4096x4096 dask DataArray the wall time drops ~33% versus the
    # ``synchronous`` scheduler this used to call (issue #1714).
    if delayed_tasks:
        import dask
        dask.compute(*delayed_tasks, scheduler='threads')

    # Write VRT index with relative paths
    from ._vrt import write_vrt as _write_vrt_fn
    _write_vrt_fn(vrt_path, tile_paths, relative=True, nodata=nodata)


def _validate_chunks_arg(chunks, *, allow_none=False):
    """Validate the ``chunks`` kwarg shared across the dask read entry points.

    Centralises the rejection rule that ``read_geotiff_dask`` already
    runs so ``read_geotiff_gpu`` and ``read_vrt`` can share the same
    error format. With ``allow_none=True`` a ``None`` value passes
    through unchanged (used by entry points whose default is
    ``chunks=None``, e.g. ``read_geotiff_gpu`` and ``read_vrt``).
    With ``allow_none=False`` (default, matches ``read_geotiff_dask``)
    a ``None`` is rejected with the same ``ValueError`` format as any
    other non-int / non-tuple value, so callers see a clear
    parameter-named error instead of a downstream ``TypeError`` from
    the chunk-unpacking math.
    Otherwise ``chunks`` must be a positive int or a 2-tuple of
    positive ints. Booleans are rejected because ``True``/``False``
    are int subclasses that would otherwise sneak through the integer
    check. Returns the coerced int when given an ``np.integer`` scalar
    so downstream ``isinstance(chunks, int)`` checks stay accurate.

    Mirrors the chunks-validation #1752 added to ``read_geotiff_dask``;
    extends it to the GPU read and VRT read entry points per #1776.
    """
    if chunks is None:
        if allow_none:
            return chunks
        raise ValueError(
            f"chunks must be a positive int or (row, col) tuple of "
            f"positive ints, got chunks=None.")
    if (isinstance(chunks, (int, np.integer))
            and not isinstance(chunks, bool)):
        if chunks <= 0:
            raise ValueError(
                f"chunks must be a positive int or (row, col) tuple of "
                f"positive ints, got chunks={chunks}.")
        return int(chunks)
    if isinstance(chunks, tuple):
        if len(chunks) != 2:
            raise ValueError(
                f"chunks tuple must have length 2 (row, col), got "
                f"chunks={chunks!r} with length {len(chunks)}.")
        for _v in chunks:
            if (not isinstance(_v, (int, np.integer))
                    or isinstance(_v, bool)
                    or _v <= 0):
                raise ValueError(
                    f"chunks must be a positive int or (row, col) tuple "
                    f"of positive ints, got chunks={chunks!r}.")
        return chunks
    raise ValueError(
        f"chunks must be a positive int or (row, col) tuple of "
        f"positive ints, got chunks={chunks!r} "
        f"(type {type(chunks).__name__}).")


def _validate_tile_size_arg(tile_size):
    """Validate the ``tile_size`` kwarg for the tiled writer entry points.

    Wrapper kept for backwards internal compatibility; delegates to
    ``_validate_tile_size`` so to_geotiff/write_geotiff_gpu share one
    validation path (positive int + multiple-of-16 for tiled output).
    """
    _validate_tile_size(tile_size)


def read_geotiff_dask(source: str, *,
                      dtype: str | np.dtype | None = None,
                      chunks: int | tuple = 512,
                      overview_level: int | None = None,
                      window: tuple | None = None,
                      band: int | None = None,
                      max_pixels: int | None = None,
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
    window : tuple or None
        ``(row_start, col_start, row_stop, col_stop)`` to restrict
        chunking to a sub-region of the file. Chunks are laid out
        relative to the window origin. None reads the full raster.
    band : int or None
        Zero-based band index. None returns all bands (3D for
        multi-band files, 2D for single-band). Selecting a single band
        produces a 2D DataArray.
    max_pixels : int or None
        Maximum allowed pixel count (width * height * samples) for the
        windowed region. None uses the reader default (~1 billion).
        The cap is checked once up-front against the lazy region; each
        chunk task also re-checks against ``max_pixels`` so windowed
        reads stay bounded even when ``read_to_array`` is invoked
        directly.
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

    # Reject non-positive chunk sizes up front. ``chunks=0`` and negative
    # values otherwise propagate into dask chunk math (``range(0, N, 0)``
    # ValueError, or empty chunk grids) with no indication that ``chunks``
    # was the problem. Shared with ``read_geotiff_gpu`` / ``read_vrt`` via
    # ``_validate_chunks_arg`` so all three entry points emit the same
    # error format (#1752 / #1776). ``allow_none=False`` (the default)
    # rejects ``chunks=None`` with the same ValueError; this entry point
    # requires a concrete chunk size since the chunk-unpacking math below
    # would otherwise fail with a confusing TypeError.
    chunks = _validate_chunks_arg(chunks)

    # ``open_geotiff`` already routes ``.vrt`` to ``read_vrt`` before
    # reaching here, so this branch is only hit when ``read_geotiff_dask``
    # is called directly with a VRT path. Keep it as a defensive fallback
    # rather than letting the windowed-read path try to parse VRT XML as
    # TIFF bytes. ``read_vrt`` is the single source of truth for VRT.
    if isinstance(source, str) and source.lower().endswith('.vrt'):
        return read_vrt(
            source, dtype=dtype, window=window, band=band, name=name,
            chunks=chunks, max_pixels=max_pixels,
        )

    # P5: HTTP COG sources used to fire one IFD/header GET per chunk
    # task. Parse metadata once here so every delayed task can reuse it.
    # The same prefetch path also covers fsspec URIs (s3://, gs://, ...);
    # ``_parse_cog_http_meta`` only needs a ``read_range``-having source,
    # and ``_CloudSource`` satisfies that contract. Going through it
    # bounds metadata reads to ``MAX_HTTP_HEADER_BYTES`` instead of
    # fetching the whole remote object up front. See PR #1755 review.
    is_http = (
        isinstance(source, str)
        and source.startswith(('http://', 'https://'))
    )
    from ._reader import _is_fsspec_uri
    is_fsspec = isinstance(source, str) and _is_fsspec_uri(source)
    http_meta = None
    http_meta_key = None
    if is_http or is_fsspec:
        import dask
        from ._reader import _parse_cog_http_meta
        if is_http:
            from ._reader import _HTTPSource
            _src = _HTTPSource(source)
        else:
            from ._reader import _CloudSource
            _src = _CloudSource(source)
        try:
            http_header, http_ifd, http_geo, _ = _parse_cog_http_meta(
                _src, overview_level=overview_level)
        finally:
            _src.close()
        http_meta = (http_header, http_ifd)
        if http_ifd.orientation != 1:
            raise ValueError(
                f"Orientation tag (274) is {http_ifd.orientation}; "
                f"dask-chunked reads (chunks=...) are not supported for "
                f"non-default orientation on remote GeoTIFF sources. Read "
                f"the full array first, then slice/chunk it."
            )
        # Wrap the parsed metadata in a single dask Delayed so every
        # window task takes it as a graph input, not a Python closure.
        # Without this, the (TIFFHeader, IFD) pair -- which can carry
        # multi-million-entry TileOffsets/TileByteCounts tuples on
        # large COGs -- would be embedded in each chunk task and
        # serialised N times under distributed/process schedulers.
        http_meta_key = dask.delayed(http_meta, pure=True)
        geo_info = http_geo
        full_h = http_ifd.height
        full_w = http_ifd.width
        from ._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
        bps = resolve_bits_per_sample(http_ifd.bits_per_sample)
        file_dtype = tiff_dtype_to_numpy(bps, http_ifd.sample_format)
        n_bands = (
            http_ifd.samples_per_pixel
            if http_ifd.samples_per_pixel > 1 else 0
        )
        # Stash IFD photometric for the MinIsWhite nodata-inversion check below.
        geo_info._ifd_photometric = http_ifd.photometric
        geo_info._ifd_samples_per_pixel = http_ifd.samples_per_pixel
    else:
        # Metadata-only read: O(1) memory via mmap, no pixel decompression
        geo_info, full_h, full_w, file_dtype, n_bands = _read_geo_info(
            source, overview_level=overview_level)
    nodata = geo_info.nodata
    nodata_attr = nodata  # original sentinel preserved for attrs['nodata']
    # When the source is MinIsWhite (photometric == 0, samples_per_pixel == 1),
    # the per-chunk reader inverts pixel values before this closure's nodata
    # mask runs. Track the inverted sentinel so the mask compares against the
    # post-inversion value, not the original (#1809).
    if nodata is not None:
        _phm = getattr(geo_info, '_ifd_photometric', None)
        _spp = getattr(geo_info, '_ifd_samples_per_pixel', None)
        if _phm == 0 and _spp == 1:
            if file_dtype.kind == 'u' and np.isfinite(nodata) and \
                    float(nodata).is_integer():
                vi = int(nodata)
                info = np.iinfo(file_dtype)
                if info.min <= vi <= info.max:
                    nodata = info.max - vi
            elif file_dtype.kind == 'f' and not np.isnan(nodata):
                nodata = -float(nodata)

    # Nodata masking promotes integer arrays to float64 (for NaN).
    # Validate against the effective dtype, not the raw file dtype.
    # An out-of-range sentinel (e.g. uint16 file + nodata=-9999) is a
    # no-op for masking and leaves the file dtype unchanged. A
    # non-finite sentinel ("NaN" / "Inf" GDAL_NODATA strings) cannot
    # match an integer pixel either and is short-circuited via the
    # ``np.isfinite`` gate so the ``int(...)`` cast never sees NaN
    # (#1774). A fractional sentinel (e.g. ``"3.5"`` on a ``uint16``
    # file) also cannot match an integer pixel and ``int(3.5)`` would
    # truncate to 3, silently flagging a real pixel value as nodata;
    # gate on ``float(nodata).is_integer()`` as well so fractional
    # tags stay on the no-op path. The try/except keeps callers that
    # pass an exotic ``nodata`` type (e.g. complex) on the no-op path
    # rather than surfacing an opaque error here.
    effective_dtype = file_dtype
    if (nodata is not None
            and file_dtype.kind in ('u', 'i')
            and np.isfinite(nodata)
            and float(nodata).is_integer()):
        try:
            _nd_int = int(nodata)
            _info = np.iinfo(file_dtype)
            if _info.min <= _nd_int <= _info.max:
                effective_dtype = np.dtype('float64')
        except (TypeError, ValueError):
            pass

    if dtype is not None:
        target_dtype = np.dtype(dtype)
        _validate_dtype_cast(effective_dtype, target_dtype)
    else:
        target_dtype = effective_dtype

    # Window clipping: restrict the lazy region to the requested
    # sub-rectangle. ``read_to_array`` already accepts ``window=`` per
    # chunk; we only need to remap the chunk grid so its origin moves to
    # ``(win_r0, win_c0)`` and its extent shrinks to the window.
    win_r0 = win_c0 = 0
    if window is not None:
        win_r0, win_c0, win_r1, win_c1 = window
        if (win_r0 < 0 or win_c0 < 0
                or win_r1 > full_h or win_c1 > full_w
                or win_r0 >= win_r1 or win_c0 >= win_c1):
            raise ValueError(
                f"window={window} is outside the source extent "
                f"({full_h}x{full_w}) or has non-positive size.")
        # Mirror the eager-path windowed coord computation in open_geotiff,
        # including the ``has_georef=False`` shortcut to integer pixel
        # coords for non-georef files (#1710).
        coords = _coords_from_geo_info(
            geo_info, win_r1 - win_r0, win_c1 - win_c0, window=window,
        )
        full_h = win_r1 - win_r0
        full_w = win_c1 - win_c0
    else:
        coords = _geo_to_coords(geo_info, full_h, full_w)

    if band is not None:
        # Reject ``bool`` and ``np.bool_`` up front; ``isinstance(True, int)``
        # is True in Python so ``True < n_bands`` evaluates without raising
        # and silently reads band 1. ``np.bool_`` is not a subclass of
        # ``bool`` so it needs its own check to match the VRT path's
        # rejection. See #1786.
        if isinstance(band, (bool, np.bool_)):
            raise ValueError(
                f"band must be a non-negative int, got {band!r}")
        if n_bands == 0:
            if band != 0:
                raise IndexError(
                    f"band={band} requested on a single-band file.")
        elif not 0 <= band < n_bands:
            raise IndexError(
                f"band={band} out of range for {n_bands}-band file.")

    # Up-front pixel-count guard against the windowed extent. Chunk
    # tasks re-check via read_to_array's own ``max_pixels`` (which we
    # forward through ``_delayed_read_window``), but catching an
    # oversized request before any task is scheduled saves the caller
    # from a misleading "tile size exceeds max_pixels" error in a
    # chunk that happens to align with the file's tile grid.
    # ``max_pixels=None`` substitutes the module default to match the
    # eager (``read_to_array``) and VRT chunked paths. Without the
    # substitution the guard would skip entirely on ``None`` and a
    # caller could build a lazy graph over a region far larger than the
    # documented safety cap. See issue #1838.
    from ._reader import MAX_PIXELS_DEFAULT as _MAX_PIXELS_DEFAULT
    effective_max_pixels = (max_pixels if max_pixels is not None
                            else _MAX_PIXELS_DEFAULT)
    eff_bands = (1 if band is not None
                 else (n_bands if n_bands > 0 else 1))
    if full_h * full_w * eff_bands > effective_max_pixels:
        raise ValueError(
            f"Requested region {full_h}x{full_w}x{eff_bands} "
            f"exceeds max_pixels={effective_max_pixels:,}.")

    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    attrs = {}
    _populate_attrs_from_geo_info(attrs, geo_info, window=window)
    if nodata_attr is not None:
        attrs['nodata'] = nodata_attr

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
    band_arg = band  # None => all bands (or 2D for single-band file)

    # When ``band`` is set, each chunk reads a 2D slice -- collapse the
    # output dims so the returned DataArray is 2D regardless of file band
    # count.
    out_has_band_axis = band is None and n_bands > 0

    dask_rows = []
    for r0 in rows:
        r1 = min(r0 + ch_h, full_h)
        dask_cols = []
        for c0 in cols:
            c1 = min(c0 + ch_w, full_w)
            if out_has_band_axis:
                block_shape = (r1 - r0, c1 - c0, n_bands)
            else:
                block_shape = (r1 - r0, c1 - c0)
            # Translate window-relative chunk coords back to file-relative
            # coords for ``read_to_array``. ``win_r0`` / ``win_c0`` are 0
            # when no window was requested.
            # Always thread ``target_dtype`` so each delayed chunk lands
            # in the same dtype that the dask array declared. Without
            # this, an integer raster with an in-range nodata sentinel
            # would have ``effective_dtype=float64`` declared on the
            # dask graph but per-chunk arrays only promoted when a
            # chunk happened to contain a sentinel pixel. Dask
            # concatenation then preallocates from the first chunk's
            # actual dtype (uint16), silently casting later float64
            # chunks back to int and converting their NaNs to 0. See
            # issue #1597.
            block = da.from_delayed(
                _delayed_read_window(source,
                                     r0 + win_r0, c0 + win_c0,
                                     r1 + win_r0, c1 + win_c0,
                                     overview_level, nodata,
                                     band_arg,
                                     target_dtype=target_dtype,
                                     http_meta_key=http_meta_key,
                                     max_pixels=max_pixels),
                shape=block_shape,
                dtype=target_dtype,
            )
            dask_cols.append(block)
        dask_rows.append(da.concatenate(dask_cols, axis=1))

    dask_arr = da.concatenate(dask_rows, axis=0)

    if out_has_band_axis:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(n_bands)
    else:
        dims = ['y', 'x']

    return xr.DataArray(
        dask_arr, dims=dims, coords=coords, name=name, attrs=attrs,
    )


def _delayed_read_window(source, r0, c0, r1, c1, overview_level, nodata,
                         band, *, target_dtype=None, http_meta_key=None,
                         max_pixels=None):
    """Dask-delayed function to read a single window.

    *http_meta_key* is an optional ``Delayed[(TIFFHeader, IFD)]`` parsed
    once by :func:`read_geotiff_dask` and wrapped via ``dask.delayed``.
    Passing it as a function argument (rather than a closure capture)
    makes the metadata a single graph input that all window tasks
    depend on, so distributed/process schedulers serialise it once
    instead of once per chunk. For local sources it is ``None``.
    """
    import dask

    @dask.delayed
    def _read(http_meta):
        # The prefetched-metadata fast path covers both HTTP COGs and
        # fsspec-addressable remotes (s3://, gs://, az://, memory://, ...).
        # Both source classes expose ``read_range``, which is all
        # ``_fetch_decode_cog_http_tiles`` needs.
        _is_http_src = isinstance(source, str) and source.startswith(
            ('http://', 'https://'))
        _is_fsspec_src = False
        if http_meta is not None and isinstance(source, str) and \
                not _is_http_src:
            from ._reader import _is_fsspec_uri as _ifs
            _is_fsspec_src = _ifs(source)
        if http_meta is not None and (_is_http_src or _is_fsspec_src):
            from ._reader import (
                _fetch_decode_cog_http_tiles,
                MAX_PIXELS_DEFAULT,
                _apply_photometric_miniswhite,
            )
            header, ifd = http_meta
            if _is_http_src:
                from ._reader import _HTTPSource
                src = _HTTPSource(source)
            else:
                from ._reader import _CloudSource
                src = _CloudSource(source)
            try:
                arr = _fetch_decode_cog_http_tiles(
                    src, header, ifd,
                    max_pixels=(max_pixels if max_pixels is not None
                                else MAX_PIXELS_DEFAULT),
                    window=(r0, c0, r1, c1))
            finally:
                src.close()
            if (arr.ndim == 3 and ifd.samples_per_pixel > 1
                    and band is not None):
                arr = arr[:, :, band]
            arr = _apply_photometric_miniswhite(arr, ifd)
        else:
            _r2a_kwargs = {}
            if max_pixels is not None:
                _r2a_kwargs['max_pixels'] = max_pixels
            arr, _ = _read_to_array(source, window=(r0, c0, r1, c1),
                                    overview_level=overview_level,
                                    band=band, **_r2a_kwargs)
        if nodata is not None:
            # ``arr`` was just decoded by ``_fetch_decode_cog_http_tiles``
            # or ``read_to_array``; both return freshly-allocated buffers
            # that nothing else references, so the in-place sentinel
            # rewrite is safe. Skip the defensive ``arr.copy()`` to
            # avoid a peak-memory doubler on every dask chunk.
            if arr.dtype.kind == 'f' and not np.isnan(nodata):
                arr[arr == arr.dtype.type(nodata)] = np.nan
            elif (arr.dtype.kind in ('u', 'i')
                  and np.isfinite(nodata)
                  and float(nodata).is_integer()):
                # Out-of-range sentinels (e.g. uint16 + nodata=-9999)
                # cannot match any pixel; skip the cast that would
                # otherwise raise OverflowError and leave arr unchanged.
                # Non-finite sentinels ("NaN" / "Inf" GDAL_NODATA strings)
                # also cannot match an integer pixel and would raise
                # ValueError on ``int(nodata)``; the ``np.isfinite`` gate
                # mirrors ``_resolve_masked_fill`` in ``_reader.py``
                # (#1774). Fractional sentinels (e.g. ``"3.5"`` on a
                # ``uint16`` file) also cannot match an integer pixel and
                # ``int(3.5)`` would truncate to 3 and silently mask
                # pixel value 3; the ``float(nodata).is_integer()`` gate
                # short-circuits them too.
                nodata_int = int(nodata)
                info = np.iinfo(arr.dtype)
                if info.min <= nodata_int <= info.max:
                    mask = arr == arr.dtype.type(nodata_int)
                    if mask.any():
                        arr = arr.astype(np.float64)
                        arr[mask] = np.nan
        if target_dtype is not None and arr.dtype != target_dtype:
            # Skip the cast when dtype already matches. ``numpy.astype``
            # defaults to ``copy=True`` and would otherwise allocate a
            # full chunk-sized buffer and memcpy on every read just to
            # land in the same dtype the array already has. The int->
            # float64 promotion above (sentinel-hit branch) keeps the
            # contract that every chunk lands in the dask-declared
            # dtype; this guard only elides no-op casts. See #1624.
            arr = arr.astype(target_dtype)
        return arr
    return _read(http_meta_key)


def _gpu_decode_single_band_tiles(
    source, lazy_data, offsets, byte_counts,
    tw, th, width, height,
    compression, predictor, file_dtype,
    *,
    byte_order: str,
    gpu: str,
):
    """Decode one band's tile sequence into a 2-D ``(H, W)`` cupy array.

    Helper for the ``PlanarConfiguration=2`` GPU path: the existing
    ``gpu_decode_tiles`` / ``gpu_decode_tiles_from_file`` kernels assume
    a single chunky tile sequence with ``bytes_per_pixel = itemsize *
    samples``. For planar=2 each band has its own list of tiles, so we
    invoke those kernels once per band with ``samples=1`` and stack the
    resulting 2-D arrays into ``(H, W, samples)`` afterwards.

    Mirrors the two-stage GPU pipeline in ``read_geotiff_gpu`` -- GDS
    first, then CPU-extracted-tiles GPU decode. ``lazy_data`` is a
    zero-arg callable that returns the full file bytes; it caches its
    result so the first band that needs the stage-2 fallback pays the
    ``read_all()``, and subsequent bands reuse the same buffer. When
    every band's GDS path succeeds the file is never read at all.
    Sparse tiles are not expected here; the caller routes sparse files
    to the CPU path.

    Auto-mode semantics: a stage-1 GDS failure warns and falls through
    to stage 2; a stage-2 failure warns and returns ``None`` so the
    caller can trigger a whole-image CPU fallback (a per-band CPU
    decode would require a single-band CPU path keyed off planar
    config, which doesn't exist). Strict mode re-raises from either
    stage.
    """
    from ._gpu_decode import gpu_decode_tiles, gpu_decode_tiles_from_file

    arr_gpu = None
    try:
        arr_gpu = gpu_decode_tiles_from_file(
            source, offsets, byte_counts,
            tw, th, width, height,
            compression, predictor, file_dtype, 1,
            byte_order=byte_order,
        )
    except Exception as e:
        if gpu == 'strict' or _geotiff_strict_mode():
            raise
        warnings.warn(
            f"read_geotiff_gpu: GPU decode failed "
            f"({type(e).__name__}: {e}); falling back to CPU.",
            RuntimeWarning,
            stacklevel=3,
        )
        arr_gpu = None

    if arr_gpu is None:
        shared_data = lazy_data()
        compressed_tiles = [
            bytes(shared_data[offsets[i]:offsets[i] + byte_counts[i]])
            for i in range(len(offsets))
        ]
        try:
            arr_gpu = gpu_decode_tiles(
                compressed_tiles,
                tw, th, width, height,
                compression, predictor, file_dtype, 1,
                byte_order=byte_order,
            )
        except Exception as e:
            if gpu == 'strict' or _geotiff_strict_mode():
                raise
            warnings.warn(
                f"read_geotiff_gpu: GPU decode failed "
                f"({type(e).__name__}: {e}); falling back to CPU.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None

    if arr_gpu.ndim != 2 or arr_gpu.shape != (height, width):
        raise RuntimeError(
            f"single-band GPU tile decode produced shape "
            f"{arr_gpu.shape}, expected ({height}, {width})"
        )
    return arr_gpu


def _apply_orientation_gpu(arr_gpu, orientation: int):
    """cupy-side mirror of :func:`._reader._apply_orientation`.

    The CPU reader applies the TIFF Orientation tag (274) post-decode so
    pixel (0, 0) is always the visual top-left. The GPU read path used
    to skip this remap, so reads of any file with orientation != 1
    returned different pixel buffers than the CPU reader (#1540).

    Same eight orientations the CPU helper handles. Operates on a cupy
    ndarray and returns a cupy ndarray; ``cupy.ascontiguousarray`` is
    applied so downstream views (DataArray.data) work without surprise
    re-strides on the GPU.
    """
    import cupy
    if orientation == 1:
        return arr_gpu
    if orientation == 2:
        return cupy.ascontiguousarray(arr_gpu[:, ::-1])
    if orientation == 3:
        return cupy.ascontiguousarray(arr_gpu[::-1, ::-1])
    if orientation == 4:
        return cupy.ascontiguousarray(arr_gpu[::-1, :])
    if arr_gpu.ndim == 3:
        if orientation == 5:
            return cupy.ascontiguousarray(arr_gpu.transpose(1, 0, 2))
        if orientation == 6:
            return cupy.ascontiguousarray(arr_gpu.transpose(1, 0, 2)[:, ::-1])
        if orientation == 7:
            return cupy.ascontiguousarray(
                arr_gpu.transpose(1, 0, 2)[::-1, ::-1])
        if orientation == 8:
            return cupy.ascontiguousarray(arr_gpu.transpose(1, 0, 2)[::-1, :])
    else:
        if orientation == 5:
            return cupy.ascontiguousarray(arr_gpu.T)
        if orientation == 6:
            return cupy.ascontiguousarray(arr_gpu.T[:, ::-1])
        if orientation == 7:
            return cupy.ascontiguousarray(arr_gpu.T[::-1, ::-1])
        if orientation == 8:
            return cupy.ascontiguousarray(arr_gpu.T[::-1, :])
    raise ValueError(
        f"Invalid TIFF Orientation tag value: {orientation} "
        f"(must be 1-8 per TIFF 6.0)"
    )


def _apply_orientation_geo_info(geo_info, orientation: int,
                                file_h: int, file_w: int):
    """Mirror the transform updates `_reader.read_to_array` does post-flip.

    Centralised so both ``read_to_array`` (CPU) and ``read_geotiff_gpu``
    (this module) update the GeoTransform consistently. Operates only
    on ``geo_info.transform``; the rest of the GeoInfo struct stays as
    parsed.
    """
    if orientation == 1 or geo_info is None or geo_info.transform is None:
        return geo_info
    t = geo_info.transform
    if orientation in (2, 3, 4):
        if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
            x_shift = file_w - 1
            y_shift = file_h - 1
        else:
            x_shift = file_w
            y_shift = file_h
        new_origin_x = t.origin_x
        new_origin_y = t.origin_y
        new_px_w = t.pixel_width
        new_px_h = t.pixel_height
        if orientation in (2, 3):
            new_origin_x = t.origin_x + x_shift * t.pixel_width
            new_px_w = -t.pixel_width
        if orientation in (3, 4):
            new_origin_y = t.origin_y + y_shift * t.pixel_height
            new_px_h = -t.pixel_height
        geo_info.transform = GeoTransform(
            origin_x=new_origin_x,
            origin_y=new_origin_y,
            pixel_width=new_px_w,
            pixel_height=new_px_h,
        )
    elif orientation in (5, 6, 7, 8):
        # Match the CPU reader's #1765 refusal: a pixel-size swap alone
        # cannot express the per-orientation origin shift plus rotation
        # these orientations require, so the x/y coords would be wrong.
        # ``has_georef`` is True for any file carrying ModelTransformation,
        # ModelPixelScale, or ModelTiepoint, with or without a CRS tag, so
        # gate on that flag rather than CRS presence.
        if getattr(geo_info, 'has_georef', False):
            raise NotImplementedError(
                f"TIFF Orientation {orientation} on a georeferenced file "
                f"requires a per-orientation origin shift plus a rotation "
                f"that the axis-aligned GeoTransform used here cannot "
                f"represent, so the returned x/y coords would be wrong. "
                f"Reproject the file with another tool (e.g. GDAL) or "
                f"strip the Orientation tag before reading. See issue "
                f"#1765."
            )
        # Non-georeferenced file: swap pixel sizes to match the
        # transposed array shape. No geographic claim to violate.
        geo_info.transform = GeoTransform(
            origin_x=t.origin_x,
            origin_y=t.origin_y,
            pixel_width=t.pixel_height,
            pixel_height=t.pixel_width,
        )
    return geo_info


def _gpu_apply_window_band(arr_gpu, geo_info, *, window, band):
    """Slice a fully-decoded GPU array down to a window and/or band.

    Used by ``read_geotiff_gpu`` to keep the public surface in line with
    ``open_geotiff`` and ``read_geotiff_dask``: callers can pass ``window``
    and ``band``, and the returned DataArray covers exactly that subset.

    The current implementation slices on device after the full-image GPU
    decode is complete. That preserves correctness but does no I/O
    savings -- a future PR can short-circuit tile decode for partial
    windows. For ``band`` selection, the savings are also post-decode
    because the planar=1 (chunky) tile assembly returns all bands in a
    single GPU buffer.

    Returns ``(arr_gpu, coords)`` where ``coords`` is a dict with
    ``y`` / ``x`` numpy arrays sized to the output array. The caller is
    responsible for setting ``attrs['transform']`` to the windowed origin
    via ``_populate_attrs_from_geo_info(..., window=window)`` so the array
    and the transform agree.
    """
    if window is not None:
        r0, c0, r1, c1 = window
        arr_gpu = arr_gpu[r0:r1, c0:c1]
        out_h = r1 - r0
        out_w = c1 - c0
        # Mirror the eager-numpy windowed coord computation in
        # open_geotiff so the GPU-windowed coords carry the same
        # absolute pixel-center values as the CPU path. For files
        # with no GeoTIFF tags (``has_georef=False``), fall back to
        # integer pixel coords matching ``_geo_to_coords`` (#1710).
        coords = _coords_from_geo_info(
            geo_info, out_h, out_w, window=window,
        )
    else:
        out_h = arr_gpu.shape[0]
        out_w = arr_gpu.shape[1]
        coords = _geo_to_coords(geo_info, out_h, out_w)

    if band is not None and arr_gpu.ndim == 3:
        arr_gpu = arr_gpu[:, :, band]

    return arr_gpu, coords


def read_geotiff_gpu(source: str, *,
                     dtype: str | np.dtype | None = None,
                     overview_level: int | None = None,
                     window: tuple | None = None,
                     band: int | None = None,
                     name: str | None = None,
                     chunks: int | tuple | None = None,
                     max_pixels: int | None = None,
                     on_gpu_failure: str = _ON_GPU_FAILURE_SENTINEL,
                     gpu: str = _GPU_DEPRECATED_SENTINEL,
                     ) -> xr.DataArray:
    """Read a GeoTIFF with GPU-accelerated decompression via Numba CUDA.

    Decompresses all tiles in parallel on the GPU and returns a
    CuPy-backed DataArray that stays on device memory. No CPU->GPU
    transfer needed for downstream xrspatial GPU operations.

    With ``chunks=``, returns a Dask+CuPy DataArray with real
    out-of-core memory bounds: each chunk reads only the tiles for its
    window (via the CPU dask path) and uploads the result to the
    device, so peak GPU memory is one chunk rather than the whole
    raster. The trade-off is per-chunk CPU decode rather than bulk GPU
    decode; for rasters that fit on the device, ``chunks=None`` keeps
    the full GPU-decode fast path.

    Requires: cupy, numba with CUDA support.

    Parameters
    ----------
    source : str
        File path.
    dtype : str, numpy.dtype, or None
        Cast the result to this dtype after reading. None keeps the
        file's native dtype. Float-to-int casts raise ValueError, mirroring
        ``open_geotiff`` / ``read_geotiff_dask``.
    overview_level : int or None
        Overview level (0 = full resolution).
    window : tuple or None
        ``(row_start, col_start, row_stop, col_stop)`` for windowed
        reading. None reads the full raster. The GPU pipeline currently
        decodes all tiles and slices on device after assembly, so the
        kwarg restores API parity with ``open_geotiff`` and
        ``read_geotiff_dask`` but does not yet skip I/O for partial
        windows. The returned coords, ``attrs['transform']``, and
        shape match the eager numpy path.
    band : int or None
        Zero-based band index. None returns all bands (3D output for
        multi-band files, 2D for single-band). Selecting a single band
        yields a 2D DataArray.
    chunks : int, tuple, or None
        If set, return a Dask-chunked CuPy DataArray decoded one chunk
        at a time. int for square chunks, (row, col) tuple for
        rectangular. Each chunk task reads only the tiles overlapping
        its window (CPU decode) and uploads the result to the device,
        so peak GPU memory is bounded by chunk size. ``chunks=None``
        (default) decodes the full raster on the GPU in one pass.
    name : str or None
        Name for the DataArray.
    max_pixels : int or None
        Maximum allowed pixel count (width * height * samples). None
        uses the default (~1 billion).
    on_gpu_failure : {'auto', 'strict'}, default 'auto'
        Behaviour when any GPU decode stage raises an exception.

        The GPU pipeline has two stages: first ``gpu_decode_tiles_from_file``
        (GDS-style direct read), then ``gpu_decode_tiles`` over CPU-mmap
        extracted tile bytes. Both stages still run on the GPU. The CPU
        fallback (``read_to_array`` + ``cupy.asarray``) only fires after
        both GPU stages have failed.

        - ``'auto'``: each GPU-stage failure emits a ``RuntimeWarning``
          reporting the original exception type and message, then falls
          through to the next stage (CPU mmap re-decode for the first
          failure, full CPU decode + GPU transfer for the second). This
          preserves backward-compatible behaviour while making GPU
          regressions visible.
        - ``'strict'``: re-raise the original exception from either stage
          so GPU bugs surface immediately. Useful in tests and CI for the
          GPU fast path.

        Stripped layouts and sparse-tile files route directly to the CPU
        reader before either GPU decode stage runs, so the ``on_gpu_failure``
        kwarg does not affect them. A failure inside the subsequent
        ``cupy.asarray(...)`` upload propagates unchanged in both modes.
    gpu : str, optional
        Deprecated alias for ``on_gpu_failure``. Emits ``DeprecationWarning``
        when used. Passing both ``gpu`` and ``on_gpu_failure`` raises
        ``TypeError``. The old name shipped with values ``'auto'`` /
        ``'strict'`` and was easy to confuse with the boolean ``gpu=``
        kwarg on ``open_geotiff`` / ``to_geotiff`` / ``read_vrt``.

    Returns
    -------
    xr.DataArray
        CuPy-backed DataArray on GPU device.
    """
    new_passed = on_gpu_failure is not _ON_GPU_FAILURE_SENTINEL
    old_passed = gpu is not _GPU_DEPRECATED_SENTINEL
    if new_passed and old_passed:
        # Both supplied is ambiguous regardless of which values were
        # chosen (including the matching ``on_gpu_failure='auto',
        # gpu='auto'`` pair). Refuse rather than silently picking one.
        raise TypeError(
            "read_geotiff_gpu: pass either 'on_gpu_failure' or the "
            "deprecated 'gpu' alias, not both.")
    if old_passed:
        warnings.warn(
            "read_geotiff_gpu(..., gpu=...) is deprecated; use "
            "on_gpu_failure=... instead. The kwarg was renamed because "
            "'gpu' on open_geotiff/to_geotiff/read_vrt is a bool that "
            "selects the GPU backend, while here it selects the failure "
            "policy when the GPU path raises.",
            DeprecationWarning,
            stacklevel=2,
        )
        on_gpu_failure = gpu
    elif not new_passed:
        on_gpu_failure = 'auto'
    gpu = on_gpu_failure
    if gpu not in ('auto', 'strict'):
        raise ValueError(
            f"on_gpu_failure must be 'auto' or 'strict', got {gpu!r}")
    # Reject non-positive chunk sizes up front so the GPU dask+cupy path
    # surfaces the same error as ``read_geotiff_dask`` (#1776). Previously
    # ``chunks=0`` raised ``ZeroDivisionError`` deep in cupy/dask, and
    # ``chunks=-1`` was silently accepted (negative chunks fall out of
    # the dask chunk grid as a no-op). ``chunks=None`` is the default
    # (eager read), so allow it through here.
    chunks = _validate_chunks_arg(chunks, allow_none=True)
    try:
        import cupy
    except ImportError:
        raise ImportError(
            "cupy is required for GPU reads. "
            "Install it with: pip install cupy-cuda12x")

    # When ``chunks=`` is set, bound peak GPU memory to chunk size by
    # building a Dask+CuPy graph that decodes one chunk at a time. The
    # CPU dask path already lays out a window-per-chunk delayed graph
    # (parses TIFF metadata once, decodes only the tiles overlapping
    # each chunk window, handles HTTP/fsspec/local/sparse/planar=2/
    # MinIsWhite/nodata/orientation). Reusing it and uploading each
    # block to the device via ``map_blocks(cupy.asarray)`` gives real
    # out-of-core behaviour for the read; the trade-off is per-chunk
    # CPU decode rather than the eager path's bulk GPU decode. Users
    # who want full GPU-side decode (and have device memory for the
    # whole image) pass ``chunks=None``. See issue #1876.
    if chunks is not None:
        return _read_geotiff_gpu_chunked(
            source, dtype=dtype, chunks=chunks,
            overview_level=overview_level, window=window, band=band,
            name=name, max_pixels=max_pixels,
        )

    from ._reader import (
        _FileSource, _check_dimensions, MAX_PIXELS_DEFAULT, _coerce_path,
        _resolve_masked_fill,
    )
    from ._compression import COMPRESSION_LERC
    from ._header import (
        parse_header, parse_all_ifds, select_overview_ifd, validate_tile_layout,
    )
    from ._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
    from ._geotags import extract_geo_info_with_overview_inheritance
    from ._gpu_decode import gpu_decode_tiles

    source = _coerce_path(source)

    if max_pixels is None:
        max_pixels = MAX_PIXELS_DEFAULT

    # Window basic shape check happens here; full bounds-vs-file validation
    # runs after the IFD parse below so we can compare against the chosen
    # overview level's actual height/width. ``band`` is similarly validated
    # against ``ifd.samples_per_pixel`` after the header parse.
    if window is not None:
        if len(window) != 4:
            raise ValueError(
                f"window must be a 4-tuple (r0, c0, r1, c1), got {window!r}")
        w_r0, w_c0, w_r1, w_c1 = window
        if w_r0 >= w_r1 or w_c0 >= w_c1 or w_r0 < 0 or w_c0 < 0:
            raise ValueError(
                f"window={window} has non-positive size or negative origin.")

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

        bps = resolve_bits_per_sample(ifd.bits_per_sample)
        file_dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
        # Inherit georef from the level-0 IFD when the overview itself
        # has no geokeys (issue #1640); pass-through for level 0.
        geo_info = extract_geo_info_with_overview_inheritance(
            ifd, ifds, data, header.byte_order)
        # Capture the Orientation tag (274) once so the post-decode flip
        # below picks it up for both the stripped fallback and the tiled
        # GPU pipelines. CPU read_to_array applies the array remap +
        # transform update for stripped reads, so for that branch we
        # only need to copy the post-flip geo_info back here.
        orientation = ifd.orientation

        # Orientation tag (274): values 2-8 mean the stored pixel order
        # differs from display order. A windowed read against a non-default
        # orientation has ambiguous semantics (does the window refer to
        # file pixels or display pixels?), so the CPU reader
        # ``_reader.read_to_array`` rejects ``window=`` for orientation != 1.
        # Mirror that here so the GPU path agrees with the CPU path and
        # ``read_geotiff_dask``. Use the same error wording so the failure
        # message is identical across backends.
        if orientation != 1 and window is not None:
            raise ValueError(
                f"Orientation tag (274) is {orientation}; windowed reads "
                f"(window=...) and dask-chunked reads (chunks=...) are not "
                f"supported for non-default orientation. Read the full "
                f"array first, then slice."
            )

        # Validate band against the selected IFD's sample count.
        # ``samples_per_pixel`` is at least 1 for any valid TIFF; we treat
        # ``band=0`` as "first band" for single-band files too so the
        # behaviour mirrors ``read_geotiff_dask``.
        ifd_samples = ifd.samples_per_pixel
        if band is not None:
            # Reject ``bool`` and ``np.bool_`` up front;
            # ``isinstance(True, int)`` is True in Python so
            # ``True < ifd_samples`` evaluates without raising and silently
            # reads band 1. ``np.bool_`` is not a subclass of ``bool`` so it
            # needs its own check to match the VRT path's rejection.
            # See #1786.
            if isinstance(band, (bool, np.bool_)):
                raise ValueError(
                    f"band must be a non-negative int, got {band!r}")
            if ifd_samples <= 1:
                if band != 0:
                    raise IndexError(
                        f"band={band} requested on a single-band file.")
            elif not 0 <= band < ifd_samples:
                raise IndexError(
                    f"band={band} out of range for {ifd_samples}-band file.")

        # Validate window upper bounds against the selected IFD's extent.
        if window is not None:
            w_r0, w_c0, w_r1, w_c1 = window
            ifd_h, ifd_w = ifd.height, ifd.width
            if w_r1 > ifd_h or w_c1 > ifd_w:
                raise ValueError(
                    f"window={window} is outside the source extent "
                    f"({ifd_h}x{ifd_w}).")

        if not ifd.is_tiled:
            # Fall back to CPU for stripped files. read_to_array remaps
            # the array but only updates geo_info.transform for orientations
            # 5-8 today (the 2/3/4 fix in #1539 is in a sibling PR). Discard
            # its geo_info and apply our own transform update below so the
            # result is correct regardless of merge order.
            #
            # Forward ``max_pixels``, ``window``, and ``band`` so the
            # caller's safety cap is honoured, windowed reads avoid
            # decoding the full image, and single-band selection on a
            # multi-band source skips the unused channels. Without this,
            # the stripped GPU path bypassed all three (issue #1732).
            # Orientation != 1 + window is already rejected at line 2495,
            # so ``window`` is None whenever ``geo_info`` will be remapped
            # below.
            src.close()
            arr_cpu, _stripped_geo = _read_to_array(
                source, overview_level=overview_level,
                window=window, band=band, max_pixels=max_pixels)
            arr_gpu = cupy.asarray(arr_cpu)
            if orientation != 1:
                geo_info = _apply_orientation_geo_info(
                    geo_info, orientation,
                    file_h=ifd.height, file_w=ifd.width)
            if name is None:
                import os
                name = os.path.splitext(os.path.basename(source))[0]
            attrs = {}
            _populate_attrs_from_geo_info(attrs, geo_info, window=window)
            # Apply nodata mask + record sentinel so the GPU read agrees
            # with the CPU eager path. Without this, integer rasters keep
            # the literal sentinel value and float rasters keep the
            # sentinel rather than NaN -- a silent backend divergence.
            # ``read_to_array`` stashes the post-MinIsWhite sentinel on
            # ``_mask_nodata`` when applicable; fall back to the original
            # sentinel otherwise (#1809).
            nodata = geo_info.nodata
            if nodata is not None:
                attrs['nodata'] = nodata
                mask_value = getattr(_stripped_geo, '_mask_nodata', nodata)
                arr_gpu = _apply_nodata_mask_gpu(arr_gpu, mask_value)
            if dtype is not None:
                target = np.dtype(dtype)
                _validate_dtype_cast(np.dtype(str(arr_gpu.dtype)), target)
                arr_gpu = arr_gpu.astype(target)
            # ``read_to_array`` already applied window + band slicing, so
            # ``arr_gpu`` is at output shape. Compute coords for that
            # shape without re-slicing. Mirror the eager-numpy /
            # ``read_geotiff_dask`` / ``_gpu_apply_window_band`` checks
            # against ``has_georef``: a non-georef TIFF carries a
            # default ``GeoTransform()`` placeholder (``t is None`` is
            # never true here) so a transform-based coord path would
            # emit synthetic ``[-0.5, -1.5, ...]`` floats instead of
            # the integer pixel coords every other backend produces
            # (#1753 / regression of #1710).
            coords = _coords_from_geo_info(
                geo_info, arr_gpu.shape[0], arr_gpu.shape[1], window=window,
            )
            # Multi-band stripped reads come back as (y, x, band); mirror
            # the tiled branch so dims line up with ndim. Single-band stays
            # 2-D ('y', 'x').
            if arr_gpu.ndim == 3:
                dims = ['y', 'x', 'band']
                coords['band'] = np.arange(arr_gpu.shape[2])
            else:
                dims = ['y', 'x']
            result = xr.DataArray(arr_gpu, dims=dims,
                                  coords=coords, name=name, attrs=attrs)
            # ``chunks`` was previously honoured only on the tiled path,
            # so stripped TIFFs returned an unchunked DataArray even when
            # the caller asked for a Dask+CuPy result. Mirror the tiled
            # branch's chunking step so behaviour is consistent across
            # layouts.
            if chunks is not None:
                if isinstance(chunks, int):
                    chunk_dict = {'y': chunks, 'x': chunks}
                else:
                    chunk_dict = {'y': chunks[0], 'x': chunks[1]}
                result = result.chunk(chunk_dict)
            return result

        offsets = ifd.tile_offsets
        byte_counts = ifd.tile_byte_counts
        compression = ifd.compression
        predictor = ifd.predictor
        samples = ifd.samples_per_pixel
        planar = ifd.planar_config
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
    # LERC tiles can carry a per-pixel valid mask that GDAL writes
    # zero-filled in the data array.  Compute the nodata fill the same
    # way the CPU reader does so the GPU decode path can restore it
    # post-assembly (mirrors PR #1529 for the CPU path). Only the
    # chunky (planar=1) GPU path threads masked_fill into its kernel
    # call below; the planar=2 per-band branch falls back to the CPU
    # reader for masked pixels (rare in practice -- LERC files
    # typically use chunky layout).
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, file_dtype)
                   if compression == COMPRESSION_LERC else None)

    # Track whether the array we end up with was already orientation-flipped
    # by `read_to_array`. Any path that falls back to CPU decode picks up
    # the orientation remap from PR #1521 + #1537 for free; the pure GPU
    # paths still need the explicit remap added in #1540.
    arr_was_cpu_decoded = False
    # When a CPU fallback runs, ``read_to_array`` has already applied the
    # MinIsWhite inversion and stashed the post-inversion sentinel on
    # ``_mask_nodata``. Keep that geo_info alongside the pre-extracted one
    # so the downstream nodata mask compares against the correct value
    # (Copilot review of #1817).
    _cpu_fallback_geo = None

    # PlanarConfiguration=2 (separate bands): each band has its own list
    # of tiles back-to-back in TileOffsets / TileByteCounts. The GPU
    # tile-assembly kernel assumes a single chunky tile sequence with
    # bytes_per_pixel = itemsize * samples, so it cannot handle planar=2
    # directly. Decode each band's tile slab as a single-band image, then
    # stack into (H, W, samples). For planar=1 (chunky) the existing
    # single-pass kernel is correct. Sparse-tile files always route to
    # the CPU reader regardless of planar config.
    if planar == 2 and samples > 1 and not has_sparse_tile:
        tiles_across = math.ceil(width / tw)
        tiles_down = math.ceil(height / th)
        tiles_per_band = tiles_across * tiles_down
        # validate_tile_layout already requires len(offsets) >= the grid;
        # accept extra trailing entries (some writers emit padding) and
        # only consume the first tiles_per_band * samples.
        expected_min = tiles_per_band * samples
        if len(offsets) < expected_min:
            raise ValueError(
                f"PlanarConfiguration=2 expects at least {expected_min} "
                f"TileOffsets entries ({tiles_across} x {tiles_down} x "
                f"{samples} bands), got {len(offsets)}"
            )
        # Lazy shared file read for the per-band stage-2 fallback. When
        # every band's GDS path succeeds, _read_once is never called
        # and we skip the read_all() entirely; when any band falls
        # back, the first call materialises the bytes and subsequent
        # bands reuse the same buffer (so N bands cost at most one
        # read_all(), not N).
        _shared_data_cache: list = []

        def _read_once():
            if not _shared_data_cache:
                src2 = _FileSource(source)
                try:
                    _shared_data_cache.append(src2.read_all())
                finally:
                    src2.close()
            return _shared_data_cache[0]

        band_arrays = []
        cpu_fallback_needed = False
        for band_idx in range(samples):
            b0 = band_idx * tiles_per_band
            b1 = b0 + tiles_per_band
            band_offsets = list(offsets[b0:b1])
            band_byte_counts = list(byte_counts[b0:b1])
            band_arr = _gpu_decode_single_band_tiles(
                source, _read_once, band_offsets, band_byte_counts,
                tw, th, width, height,
                compression, predictor, file_dtype,
                byte_order=header.byte_order,
                gpu=gpu,
            )
            if band_arr is None:
                # Auto-mode signal: stage-2 GPU decode failed for this
                # band. There's no per-band CPU decode path, so fall
                # back to a whole-image CPU read + GPU upload, matching
                # the chunky path's auto-mode semantics.
                cpu_fallback_needed = True
                break
            band_arrays.append(band_arr)
        if cpu_fallback_needed:
            # Drop read_to_array's geo_info for orientation transform
            # handling (below operates on our pre-extracted geo_info so the
            # 2/3/4 case is covered regardless of #1539's merge state), but
            # keep it on ``_cpu_fallback_geo`` so the MinIsWhite-aware nodata
            # mask below sees ``_mask_nodata``.
            arr_cpu, _cpu_fallback_geo = _read_to_array(
                source, overview_level=overview_level)
            arr_gpu = cupy.asarray(arr_cpu)
            arr_was_cpu_decoded = True
        else:
            arr_gpu = cupy.stack(band_arrays, axis=2)
            if arr_gpu.shape != (height, width, samples):
                raise RuntimeError(
                    f"planar=2 GPU assembly produced shape "
                    f"{arr_gpu.shape}, expected "
                    f"({height}, {width}, {samples})"
                )
    elif has_sparse_tile:
        arr_cpu, _cpu_fallback_geo = _read_to_array(
            source, overview_level=overview_level)
        arr_gpu = cupy.asarray(arr_cpu)
        arr_was_cpu_decoded = True
    else:
        from ._gpu_decode import gpu_decode_tiles_from_file
        arr_gpu = None

        try:
            arr_gpu = gpu_decode_tiles_from_file(
                source, offsets, byte_counts,
                tw, th, width, height,
                compression, predictor, file_dtype, samples,
                byte_order=header.byte_order,
                masked_fill=masked_fill,
            )
        except Exception as e:
            if gpu == 'strict' or _geotiff_strict_mode():
                raise
            warnings.warn(
                f"read_geotiff_gpu: GPU decode failed "
                f"({type(e).__name__}: {e}); falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            arr_gpu = None

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
                masked_fill=masked_fill,
            )
        except Exception as e:
            if gpu == 'strict' or _geotiff_strict_mode():
                raise
            warnings.warn(
                f"read_geotiff_gpu: GPU decode failed "
                f"({type(e).__name__}: {e}); falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            arr_cpu, _cpu_fallback_geo = _read_to_array(
                source, overview_level=overview_level)
            arr_gpu = cupy.asarray(arr_cpu)
            arr_was_cpu_decoded = True

    # Multi-band tiled output must be (H, W, samples) regardless of planar
    # config -- catch any shape regression in the kernels before we attach
    # dims/coords below. Plain `raise` rather than `assert` so the check
    # survives `python -O`.
    if samples > 1:
        if (arr_gpu.shape[:2] != (height, width)
                or arr_gpu.shape[2] != samples):
            raise RuntimeError(
                f"GPU multi-band tile assembly produced shape "
                f"{arr_gpu.shape}, expected "
                f"({height}, {width}, {samples})"
            )

    # Apply the TIFF Orientation tag (274). The pure GPU paths land here
    # with a raw stored-order buffer; the CPU-fallback paths land here
    # with arr_gpu already remapped (read_to_array does the data flip)
    # but with their pre-orientation geo_info (we discarded the one
    # read_to_array returned because it does not handle 2/3/4 today).
    # Skip the GPU array remap on CPU-decoded paths to avoid a double
    # flip, but always apply the geo_info update so coords match.
    if orientation != 1:
        if not arr_was_cpu_decoded:
            arr_gpu = _apply_orientation_gpu(arr_gpu, orientation)
        geo_info = _apply_orientation_geo_info(
            geo_info, orientation, file_h=height, file_w=width)

    _mw_mask_nodata = None
    if (ifd.photometric == 0 and samples == 1 and not arr_was_cpu_decoded):
        from ._reader import _miniswhite_inverted_nodata as _miw_inv_nd
        gpu_dtype = np.dtype(str(arr_gpu.dtype))
        # Compute the post-MinIsWhite sentinel BEFORE inverting the array,
        # so the downstream ``_apply_nodata_mask_gpu`` call compares
        # against the right value (#1809).
        _mw_mask_nodata = _miw_inv_nd(geo_info.nodata, ifd, gpu_dtype)
        if gpu_dtype.kind == 'u':
            arr_gpu = np.iinfo(gpu_dtype).max - arr_gpu
        elif gpu_dtype.kind == 'f':
            arr_gpu = -arr_gpu

    # Apply nodata mask + record sentinel so the GPU read agrees with the
    # CPU eager path (issue #1542). Without this, integer rasters keep the
    # literal sentinel value and float rasters keep the sentinel rather
    # than NaN -- a silent backend divergence. Apply before the optional
    # dtype cast so the float promotion for masked integer rasters doesn't
    # surprise a user-supplied dtype.
    nodata = geo_info.nodata
    if nodata is not None:
        # When MinIsWhite was applied, the mask must use the inverted
        # sentinel; otherwise the original sentinel. The pure GPU path
        # records the inverted sentinel in ``_mw_mask_nodata`` above; the
        # CPU-fallback paths (sparse-tile, planar=2 auto-fallback, and
        # post-decode CPU fallback) get it from ``read_to_array`` via
        # ``_cpu_fallback_geo._mask_nodata`` (Copilot review of #1817).
        if _mw_mask_nodata is not None:
            _gpu_mask_value = _mw_mask_nodata
        elif _cpu_fallback_geo is not None:
            _gpu_mask_value = getattr(
                _cpu_fallback_geo, '_mask_nodata', nodata)
        else:
            _gpu_mask_value = nodata
        arr_gpu = _apply_nodata_mask_gpu(arr_gpu, _gpu_mask_value)

    if dtype is not None:
        target = np.dtype(dtype)
        _validate_dtype_cast(np.dtype(str(arr_gpu.dtype)), target)
        arr_gpu = arr_gpu.astype(target)

    # Build DataArray
    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    attrs = {}
    _populate_attrs_from_geo_info(attrs, geo_info, window=window)
    if nodata is not None:
        attrs['nodata'] = nodata

    # Apply window/band slicing post-decode. Coords are derived from the
    # sliced array so the (y, x) labels line up with the user's requested
    # subrectangle. This mirrors the ``open_geotiff`` / ``read_geotiff_dask``
    # contract: ``attrs['transform']`` always carries the full-source
    # GeoTransform shifted to the window origin (via
    # ``_populate_attrs_from_geo_info(..., window=window)``), while
    # ``coords['y']`` / ``coords['x']`` cover only the windowed cells.
    arr_gpu, coords = _gpu_apply_window_band(
        arr_gpu, geo_info, window=window, band=band)

    if arr_gpu.ndim == 3:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(arr_gpu.shape[2])
    else:
        dims = ['y', 'x']

    result = xr.DataArray(arr_gpu, dims=dims, coords=coords,
                          name=name, attrs=attrs)

    # ``chunks=`` is handled at function entry via
    # ``_read_geotiff_gpu_chunked`` for real out-of-core support; this
    # eager path always returns a non-chunked CuPy-backed DataArray.

    return result


def _gds_chunk_path_available(source, ifd, has_sparse_tile, orientation):
    """Return True iff a direct-to-GPU per-chunk decode is possible.

    The disk->GPU per-chunk path requires:

    - KvikIO present (so ``_try_kvikio_read_tiles`` can DMA tiles to VRAM).
    - A local file path (no HTTP/fsspec source).
    - A tiled layout (no strip-only file).
    - PlanarConfiguration=1 (chunky); planar=2 would need per-band tile
      grids and per-band crops.
    - No sparse tiles, since the GPU decoders skip the bytes-zero-fill
      handling the CPU reader does for them.
    - Orientation == 1, since a non-default orientation needs the full
      array on hand to apply the transform.
    - PhotometricInterpretation != 0 (MinIsWhite needs an inversion
      pass that lives in the eager path).
    """
    if not isinstance(source, str):
        return False
    if source.startswith(('http://', 'https://')):
        return False
    try:
        from ._reader import _is_fsspec_uri
        if _is_fsspec_uri(source):
            return False
    except Exception:
        pass
    try:
        import importlib.util
        if importlib.util.find_spec('kvikio') is None:
            return False
    except Exception:
        return False
    if not ifd.is_tiled:
        return False
    if has_sparse_tile:
        return False
    if ifd.planar_config != 1:
        return False
    if orientation != 1:
        return False
    if ifd.photometric == 0:
        return False
    return True


def _decode_window_gpu_direct(file_path, all_offsets, all_byte_counts,
                              tw, th, full_w, compression, predictor,
                              file_dtype, samples, byte_order,
                              r0, c0, r1, c1):
    """Decode a window's tile subset disk->GPU.

    Picks just the tiles overlapping ``(r0..r1, c0..c1)`` from the full
    tile sequence, runs them through ``gpu_decode_tiles_from_file``
    (which tries KvikIO GDS first, then a CPU mmap + ``gpu_decode_tiles``
    fallback), and crops the assembled sub-image to the requested window.
    Peak device memory is the sub-tile-grid bounding box, not the full
    raster.

    Called from inside a ``dask.delayed`` per-chunk task, so it runs
    once per chunk and only pulls the tiles that chunk needs from disk.
    """
    from ._gpu_decode import gpu_decode_tiles, gpu_decode_tiles_from_file

    tiles_across = (full_w + tw - 1) // tw

    ty_start = r0 // th
    ty_end = (r1 - 1) // th + 1
    tx_start = c0 // tw
    tx_end = (c1 - 1) // tw + 1

    sub_tiles_across = tx_end - tx_start
    sub_tiles_down = ty_end - ty_start
    sub_h = sub_tiles_down * th
    sub_w = sub_tiles_across * tw

    indices = [ty * tiles_across + tx
               for ty in range(ty_start, ty_end)
               for tx in range(tx_start, tx_end)]
    sub_offsets = [all_offsets[i] for i in indices]
    sub_byte_counts = [all_byte_counts[i] for i in indices]

    arr_gpu = gpu_decode_tiles_from_file(
        file_path, sub_offsets, sub_byte_counts,
        tw, th, sub_w, sub_h,
        compression, predictor, file_dtype, samples,
        byte_order=byte_order,
    )

    if arr_gpu is None:
        # ``gpu_decode_tiles_from_file`` returns None when KvikIO is not
        # usable on the host. Open the file via mmap, slice out just the
        # bytes for these tiles, and run the GPU decoder on those.
        from ._reader import _FileSource
        src = _FileSource(file_path)
        try:
            data = src.read_all()
            compressed_tiles = [
                bytes(data[sub_offsets[i]:sub_offsets[i] + sub_byte_counts[i]])
                for i in range(len(sub_offsets))
            ]
        finally:
            src.close()
        arr_gpu = gpu_decode_tiles(
            compressed_tiles, tw, th, sub_w, sub_h,
            compression, predictor, file_dtype, samples,
            byte_order=byte_order,
        )

    crop_r0 = r0 - ty_start * th
    crop_c0 = c0 - tx_start * tw
    crop_r1 = crop_r0 + (r1 - r0)
    crop_c1 = crop_c0 + (c1 - c0)
    if samples > 1:
        return arr_gpu[crop_r0:crop_r1, crop_c0:crop_c1, :]
    return arr_gpu[crop_r0:crop_r1, crop_c0:crop_c1]


def _read_geotiff_gpu_chunked(source, *, dtype, chunks, overview_level,
                              window, band, name, max_pixels):
    """Lazy Dask+CuPy backend for ``read_geotiff_gpu(chunks=...)``.

    Two paths produce the same shape of dask graph:

    1. **Direct disk->GPU** when KvikIO is available and the file is a
       local, tiled, chunky GeoTIFF with no sparse tiles and a trivial
       orientation. Each chunk task picks the tile subset for its
       window, DMA's those tiles to the device via GDS, decodes on the
       GPU, and crops. Peak GPU memory is one chunk and the file bytes
       never traverse host memory.

    2. **CPU decode + GPU upload** for everything else (HTTP / fsspec,
       no KvikIO, planar=2, sparse, MinIsWhite, non-trivial orientation,
       stripped layouts). Reuses ``read_geotiff_dask`` to build the
       per-chunk windowed delayed graph and ``map_blocks(cupy.asarray)``
       to upload each block. Peak GPU memory is still one chunk; the
       cost is per-chunk CPU decode rather than GDS DMA.

    Both paths are real out-of-core for device memory.
    """
    import cupy
    import dask
    import dask.array as da_mod

    from ._reader import _FileSource, _coerce_path
    from ._header import parse_header, parse_all_ifds, select_overview_ifd
    from ._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
    from ._geotags import extract_geo_info_with_overview_inheritance

    src_path = _coerce_path(source)

    # Try the disk->GPU path. Parse metadata once; if the file does not
    # qualify, fall through to the CPU-decode path. Any unexpected
    # exception during the qualification probe also falls through so we
    # never lose the ability to return a result.
    try:
        if isinstance(src_path, str) and not src_path.startswith(
                ('http://', 'https://')):
            fs = _FileSource(src_path)
            try:
                raw = fs.read_all()
            finally:
                fs.close()
            header = parse_header(raw)
            ifds = parse_all_ifds(raw, header)
            if not ifds:
                raise ValueError("No IFDs found in TIFF file")
            ifd = select_overview_ifd(ifds, overview_level)
            geo_info = extract_geo_info_with_overview_inheritance(
                ifd, ifds, raw, header.byte_order,
            )
            orientation = ifd.orientation
            has_sparse_tile = (
                ifd.tile_byte_counts is not None
                and any(bc == 0 for bc in ifd.tile_byte_counts)
            )
            if _gds_chunk_path_available(
                    src_path, ifd, has_sparse_tile, orientation):
                return _read_geotiff_gpu_chunked_gds(
                    src_path, ifd, geo_info, header,
                    dtype=dtype, chunks=chunks, window=window, band=band,
                    name=name, max_pixels=max_pixels,
                )
    except Exception:
        # GDS qualification failed; fall back to the CPU path. The
        # error would otherwise be unrelated to what the user asked
        # for (the CPU path re-parses metadata anyway).
        pass

    cpu_da = read_geotiff_dask(
        source, dtype=dtype, chunks=chunks,
        overview_level=overview_level, window=window, band=band,
        max_pixels=max_pixels, name=name,
    )

    cpu_dask_arr = cpu_da.data

    def _upload(block):
        return cupy.asarray(block)

    gpu_dask_arr = cpu_dask_arr.map_blocks(
        _upload,
        dtype=cpu_dask_arr.dtype,
        meta=cupy.empty((0,) * cpu_dask_arr.ndim, dtype=cpu_dask_arr.dtype),
    )

    return xr.DataArray(
        gpu_dask_arr, dims=cpu_da.dims, coords=cpu_da.coords,
        name=cpu_da.name, attrs=dict(cpu_da.attrs),
    )


def _read_geotiff_gpu_chunked_gds(source, ifd, geo_info, header, *,
                                  dtype, chunks, window, band, name,
                                  max_pixels):
    """Build a Dask+CuPy graph that decodes each chunk disk->GPU.

    Caller must have verified that the source qualifies via
    ``_gds_chunk_path_available``. Each chunk task pulls only the tile
    subset overlapping its window via KvikIO GDS (or an mmap fallback
    inside ``gpu_decode_tiles_from_file``) and crops on device.
    """
    import cupy
    import dask
    import dask.array as da_mod

    from ._reader import _check_dimensions, MAX_PIXELS_DEFAULT
    from ._header import validate_tile_layout
    from ._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy

    if max_pixels is None:
        max_pixels = MAX_PIXELS_DEFAULT

    full_h = ifd.height
    full_w = ifd.width
    samples = ifd.samples_per_pixel
    bps = resolve_bits_per_sample(ifd.bits_per_sample)
    file_dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
    tw = ifd.tile_width
    th = ifd.tile_height
    compression = ifd.compression
    predictor = ifd.predictor
    byte_order = header.byte_order
    offsets = list(ifd.tile_offsets)
    byte_counts = list(ifd.tile_byte_counts)

    _check_dimensions(full_w, full_h, samples, max_pixels)
    _check_dimensions(tw, th, samples, max_pixels)
    validate_tile_layout(ifd)

    # Window restricts the visible region; offsets are computed relative
    # to the windowed origin so chunks line up with the user's request.
    if window is not None:
        w_r0, w_c0, w_r1, w_c1 = window
        if (w_r0 < 0 or w_c0 < 0 or w_r1 > full_h or w_c1 > full_w
                or w_r0 >= w_r1 or w_c0 >= w_c1):
            raise ValueError(
                f"window={window} is out of bounds for image "
                f"{full_w}x{full_h}.")
        out_h, out_w = w_r1 - w_r0, w_c1 - w_c0
        win_r0, win_c0 = w_r0, w_c0
    else:
        out_h, out_w = full_h, full_w
        win_r0, win_c0 = 0, 0

    if isinstance(chunks, int):
        ch_h = ch_w = chunks
    else:
        ch_h, ch_w = chunks
    if ch_h <= 0 or ch_w <= 0:
        raise ValueError(f"Invalid chunks: {chunks}")

    # Validate band kwarg against the file's band count.
    n_bands_out = samples if samples > 1 else 0
    if band is not None:
        if n_bands_out == 0:
            if band != 0:
                raise IndexError(
                    f"band={band} requested but file is single-band.")
        elif band < 0 or band >= n_bands_out:
            raise IndexError(
                f"band={band} out of range for {n_bands_out}-band file.")

    # Wrap the big tile-offset/byte-count tuples in a single Delayed so
    # every chunk task takes them as one graph input rather than burning
    # them into every task's pickled closure.
    meta_key = dask.delayed(
        (offsets, byte_counts), pure=True,
    )

    nodata = geo_info.nodata

    @dask.delayed
    def _chunk_task(meta, r0, c0, r1, c1):
        all_offsets, all_byte_counts = meta
        arr = _decode_window_gpu_direct(
            source, all_offsets, all_byte_counts,
            tw, th, full_w, compression, predictor,
            file_dtype, samples, byte_order,
            r0, c0, r1, c1,
        )
        if nodata is not None:
            arr = _apply_nodata_mask_gpu(arr, nodata)
        if dtype is not None:
            target = np.dtype(dtype)
            _validate_dtype_cast(np.dtype(str(arr.dtype)), target)
            arr = arr.astype(target)
        if band is not None and arr.ndim == 3:
            arr = arr[:, :, band]
        return arr

    # Determine declared dtype for the dask graph. Nodata masking
    # promotes integer rasters to float64; mirror the CPU dask path.
    declared_dtype = file_dtype
    if nodata is not None and file_dtype.kind in ('u', 'i'):
        if np.isfinite(nodata) and float(nodata).is_integer():
            info = np.iinfo(file_dtype)
            if info.min <= int(nodata) <= info.max:
                declared_dtype = np.dtype('float64')
    if dtype is not None:
        declared_dtype = np.dtype(dtype)

    out_has_band_axis = band is None and n_bands_out > 0

    blocks_rows = []
    for r0 in range(0, out_h, ch_h):
        r1 = min(r0 + ch_h, out_h)
        blocks_cols = []
        for c0 in range(0, out_w, ch_w):
            c1 = min(c0 + ch_w, out_w)
            if out_has_band_axis:
                block_shape = (r1 - r0, c1 - c0, n_bands_out)
            else:
                block_shape = (r1 - r0, c1 - c0)
            # Convert chunk coords to file-space coords.
            block = da_mod.from_delayed(
                _chunk_task(meta_key,
                            r0 + win_r0, c0 + win_c0,
                            r1 + win_r0, c1 + win_c0),
                shape=block_shape,
                dtype=declared_dtype,
                meta=cupy.empty((0,) * len(block_shape),
                                dtype=declared_dtype),
            )
            blocks_cols.append(block)
        blocks_rows.append(da_mod.concatenate(blocks_cols, axis=1))
    dask_arr = da_mod.concatenate(blocks_rows, axis=0)

    # Build coords/attrs that match read_geotiff_dask's output.
    coords = _coords_from_geo_info(geo_info, out_h, out_w, window=window)
    if out_has_band_axis:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(n_bands_out)
    else:
        dims = ['y', 'x']

    attrs = {}
    _populate_attrs_from_geo_info(attrs, geo_info, window=window)
    if nodata is not None:
        attrs['nodata'] = nodata

    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    return xr.DataArray(
        dask_arr, dims=dims, coords=coords, name=name, attrs=attrs,
    )


def write_geotiff_gpu(data: xr.DataArray | cupy.ndarray | np.ndarray,
                      path: str | BinaryIO, *,
                      crs: int | str | None = None,
                      nodata: float | int | None = None,
                      compression: str = 'zstd',
                      compression_level: int | None = None,
                      tiled: bool = True,
                      tile_size: int = 256,
                      predictor: bool | int = False,
                      cog: bool = False,
                      overview_levels: list[int] | None = None,
                      overview_resampling: str = 'mean',
                      bigtiff: bool | None = None,
                      max_z_error: float = 0.0,
                      streaming_buffer_bytes: int = 256 * 1024 * 1024,
                      photometric: str | int = 'auto',
                      allow_internal_only_jpeg: bool = False) -> None:
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
    data : xr.DataArray (CuPy- or NumPy-backed), cupy.ndarray, or np.ndarray
        2D or 3D raster. CuPy-backed inputs stay on device; NumPy/Dask
        inputs are uploaded via ``cupy.asarray(np.asarray(data))``
        before compression (matches ``to_geotiff`` parity).
    path : str or binary file-like
        Output file path or any object with a ``write`` method
        (e.g. ``io.BytesIO``). ``cog=True`` requires a string path:
        the auto-dispatch path through ``to_geotiff(gpu=True, cog=True)``
        rejects file-like destinations, and the explicit GPU writer
        mirrors that rule (issue #1652).
    crs : int, str, or None
        EPSG code or WKT string. EPSG codes are strongly preferred for
        interop; the WKT-only path emits a user-defined CRS (32767) with
        the WKT stored in ``GTCitationGeoKey``, which many non-libgeotiff
        readers ignore. A ``UserWarning`` is emitted when the WKT-only
        path is taken. See issue #1768.
    nodata : float, int, or None
        NoData value.
    compression : str
        Codec name. Accepts the same set ``to_geotiff`` lists in its
        own signature: ``'none'``, ``'deflate'``, ``'lzw'``, ``'jpeg'``,
        ``'packbits'``, ``'zstd'``, ``'lz4'``, ``'jpeg2000'`` (alias
        ``'j2k'``), or ``'lerc'``.

        Routing per codec:

        - ``'zstd'`` (default) and ``'deflate'``: nvCOMP batch
          compression on the GPU -- the fastest paths and the reason to
          use this entry point.
        - ``'jpeg'``: rejected by default for parity with
          ``to_geotiff``. Both writers emit self-contained JFIF tiles
          and skip the required TIFF JPEGTables tag (347), so the
          resulting files are unreadable by libtiff, GDAL, and
          rasterio. Pass ``allow_internal_only_jpeg=True`` to opt in
          to the experimental encode path; the writer then routes to
          nvJPEG when libnvjpeg is loadable and falls back to Pillow
          otherwise, and emits a ``GeoTIFFFallbackWarning`` so the
          caller knows the output will not round-trip through external
          readers. See issue #1845.
        - ``'jpeg2000'`` and ``'j2k'``: nvJPEG2K GPU encode when
          available, glymur CPU encode otherwise. The two paths are
          not byte-for-byte identical (different libraries, different
          default parameters); use ``to_geotiff`` if you need exact
          CPU-writer parity.
        - ``'lerc'``, ``'lzw'``, ``'packbits'``, and ``'lz4'``: no
          nvCOMP/CUDA accelerator, so these fall through to the CPU
          encoder for byte-stable parity with ``to_geotiff``.
    compression_level : int or None
        Compression effort level. Accepted for API compatibility but
        currently ignored -- nvCOMP does not expose level control.
    tiled : bool
        Must be True (default). The GPU writer is tiled-only because
        nvCOMP batch compression operates on per-tile streams; passing
        ``tiled=False`` raises ``ValueError`` rather than silently
        producing a tiled file. Accepted for API parity with
        ``to_geotiff``.
    tile_size : int
        Tile size in pixels (default 256). Must be a positive multiple
        of 16; this is a TIFF 6 spec requirement on TileWidth and
        TileLength for broad reader compatibility. ``write_geotiff_gpu``
        is always tiled, so the check fires for every call.
    predictor : bool or int
        TIFF predictor. ``False``/``0``/``1`` -> none, ``True``/``2`` ->
        horizontal differencing, ``3`` -> floating-point predictor
        (float dtypes only).
    cog : bool
        Write as Cloud Optimized GeoTIFF with overviews.
    overview_levels : list[int] or None
        Overview decimation factors relative to full resolution.
        Each entry must be a power-of-two integer >= 2, and the list
        must be strictly increasing (e.g. ``[2, 4, 8]`` writes
        overviews at 1/2, 1/4 and 1/8 of the full resolution).
        Invalid values raise ``ValueError``. Only used when ``cog=True``.
        If None and ``cog=True``, auto-generates ``[2, 4, 8, ...]`` by
        halving until the smallest overview fits in a single tile.
    overview_resampling : str
        Resampling method for overviews: 'mean' (default), 'nearest',
        'min', 'max', 'median', 'mode', or 'cubic'. ``mode`` and
        ``cubic`` fall back to the CPU implementation in
        ``xrspatial.geotiff._writer`` so the GPU writer produces the
        same overview bytes as the CPU writer.
    bigtiff : bool or None
        Force BigTIFF (64-bit offsets). None auto-promotes when the
        estimated file size would exceed the classic-TIFF 4 GB limit.
    max_z_error : float
        Per-pixel error budget for LERC compression. The GPU writer
        does not implement LERC (nvCOMP has no LERC backend), so any
        non-zero value raises ``ValueError``. Accepted at the signature
        level for API parity with ``to_geotiff``.
    streaming_buffer_bytes : int
        Accepted for API parity with ``to_geotiff``. The GPU writer
        materialises the entire array on device and has no streaming
        concept, so this kwarg is a no-op. Default matches
        ``to_geotiff`` (256 MB) so callers passing the same kwargs to
        either entry point see the same default and the same type.
    photometric : str or int
        Photometric interpretation for the TIFF Photometric tag (262).
        See :func:`to_geotiff` for the full set of accepted values; the
        GPU writer forwards this kwarg unchanged. Default ``'auto'``
        writes MinIsBlack for any band count, so a 4-band raster is
        not silently tagged as RGB+alpha (issue #1769).
    allow_internal_only_jpeg : bool
        Opt in to the experimental ``compression='jpeg'`` encode path
        (default ``False``). The encoder emits self-contained JFIF
        tiles without the TIFF JPEGTables tag (347); the file decodes
        through this library's reader but not through libtiff, GDAL,
        or rasterio. With the flag set, the write proceeds and a
        ``GeoTIFFFallbackWarning`` is emitted at call time. Without
        the flag, ``compression='jpeg'`` raises ``ValueError`` for
        parity with ``to_geotiff``. See issue #1845.

    Raises
    ------
    ValueError
        If ``data`` is a 3D DataArray whose ``dims`` is not
        ``(band, y, x)`` or ``(y, x, band)`` (accepting band-name
        aliases ``bands`` / ``channel`` and spatial-name aliases
        ``lat`` / ``lon`` / ``row`` / ``col``). A leading non-band
        dim such as ``time`` would otherwise silently round-trip with
        the leading axis treated as ``y`` (issue #1812).
    """
    if not tiled:
        raise ValueError(
            "write_geotiff_gpu requires tiled=True. nvCOMP batch "
            "compression is tile-based; the strip layout is not "
            "implemented on the GPU path. Use to_geotiff(..., gpu=False, "
            "tiled=False) for strip output on CPU.")
    # JPEG-in-TIFF parity with to_geotiff (issue #1845). The GPU encode
    # path writes self-contained JFIF tiles without the TIFF JPEGTables
    # tag (347), matching the broken CPU encoder. ``to_geotiff`` refuses
    # the codec outright; this writer offered no rejection at all, so
    # callers could produce GeoTIFFs that decoded through xrspatial but
    # broke in libtiff/GDAL/rasterio. Reject by default with the same
    # wording so both entry points agree, and surface an opt-in flag for
    # users who explicitly want the internal-only path.
    if (isinstance(compression, str)
            and compression.lower() == 'jpeg'
            and not allow_internal_only_jpeg):
        raise ValueError(
            "compression='jpeg' is not supported: the encoder writes "
            "self-contained JFIF streams without the required "
            "JPEGTables tag (347), so other readers (libtiff, GDAL, "
            "rasterio) reject the file. Use 'deflate', 'zstd', or "
            "'lzw' instead. Pass allow_internal_only_jpeg=True to opt "
            "in to the experimental internal-reader-only path (issue "
            "#1845).")
    if (isinstance(compression, str)
            and compression.lower() == 'jpeg'
            and allow_internal_only_jpeg):
        warnings.warn(
            "write_geotiff_gpu(compression='jpeg', "
            "allow_internal_only_jpeg=True) writes JFIF tiles without "
            "the TIFF JPEGTables tag (347); the file decodes through "
            "xrspatial but may fail in libtiff, GDAL, or rasterio. "
            "See issue #1845.",
            GeoTIFFFallbackWarning,
            stacklevel=2,
        )
    # MinIsWhite pre-inversion (issue #1836) runs in the eager CPU writer.
    # The GPU writer assembles tile bytes directly on device; threading
    # the pixel + nodata-sentinel transform through that pipeline is out
    # of scope for the round-trip fix. Refuse the combination so callers
    # do not silently get inverted on-disk values. Move the array to the
    # CPU and call the eager ``write`` path for MinIsWhite output.
    from ._writer import _resolve_photometric as _resolve_photo_gpu
    _gpu_samples_hint = (data.shape[2] if hasattr(data, 'shape')
                         and data.ndim == 3 else 1)
    _gpu_resolved_photo, _ = _resolve_photo_gpu(
        photometric, _gpu_samples_hint)
    if _gpu_resolved_photo == 0 and _gpu_samples_hint == 1:
        raise NotImplementedError(
            "photometric='miniswhite' on the GPU writer is not "
            "supported: the writer-side pixel inversion that mirrors "
            "the reader's unconditional MinIsWhite inversion (issue "
            "#1836) is only wired into the eager CPU ``write`` path. "
            "Move the array to host memory and call to_geotiff with "
            "gpu=False, or write with photometric='minisblack' / "
            "'auto'.")
    # write_geotiff_gpu is always tiled, so validate tile_size here and
    # keep parity with the public to_geotiff entry point.
    _validate_tile_size_arg(tile_size)
    if max_z_error < 0:
        raise ValueError(
            f"max_z_error must be >= 0, got {max_z_error}")
    if max_z_error != 0:
        raise ValueError(
            "max_z_error is not supported on the GPU writer "
            "(nvCOMP has no LERC backend). Use to_geotiff(..., gpu=False) "
            "or omit max_z_error.")
    # Mirror to_geotiff's path-type + cog=True gating verbatim so callers
    # see identical errors from the two entry points. The auto-dispatch
    # path through ``to_geotiff(gpu=True, cog=True, path=BytesIO)`` raises
    # before reaching here; the explicit GPU writer mirrors the same gate
    # so callers cannot bypass it (issue #1652). Non-cog file-like writes
    # remain supported on this entry point.
    _path_is_file_like = (
        not isinstance(path, str)) and hasattr(path, 'write')
    if _path_is_file_like:
        if cog:
            raise ValueError(
                "cog=True is not supported for file-like destinations. "
                "Pass a string path or write to BytesIO without cog=True.")
    elif not isinstance(path, str):
        raise TypeError(
            f"path must be a str or a binary file-like with a write() "
            f"method, got {type(path).__name__}")
    # streaming_buffer_bytes is intentionally a no-op on the GPU path;
    # the kwarg exists for API parity with to_geotiff so callers can pass
    # the same kwargs to both entry points without filtering.
    del streaming_buffer_bytes
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
    wkt_fallback = None  # WKT string when EPSG is not available
    raster_type = 1
    gdal_meta_xml = None
    extra_tags_list = None
    x_res = None
    y_res = None
    res_unit = None

    if isinstance(crs, int):
        epsg = crs
    elif isinstance(crs, str):
        epsg = _wkt_to_epsg(crs)
        if epsg is None:
            wkt_fallback = crs

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

        # Reject ambiguous 3D layouts (issue #1812). Mirrors the gate
        # in ``to_geotiff``: a leading non-band dim like ``time`` would
        # otherwise round-trip with the leading axis silently treated
        # as ``y``.
        if arr.ndim == 3:
            _validate_3d_writer_dims(data.dims)
        # Handle band-first dimension order (band, y, x) -> (y, x, band).
        # rioxarray and CF-style multi-band rasters land here; without
        # this remap the writer treats arr.shape[2] as the band axis and
        # produces a transposed file (issue #1580). The CPU writer does
        # the same remap at the matching step in to_geotiff().
        if arr.ndim == 3 and data.dims[0] in _BAND_DIM_NAMES:
            arr = cupy.ascontiguousarray(cupy.moveaxis(arr, 0, -1))

        # Prefer attrs['transform'] over the coord-derived transform: it
        # is bit-stable across round-trips, while _coords_to_transform
        # can drift on fractional pixel sizes (the same reasoning the
        # CPU to_geotiff path applies for issue #1484).
        geo_transform = _transform_from_attr(data.attrs.get('transform'))
        if geo_transform is None:
            geo_transform = _coords_to_transform(data)
        # Resolve CRS the same way the CPU writer does. attrs['crs'] may
        # be an int EPSG or a WKT string; attrs['crs_wkt'] only carries
        # WKT. Without the WKT branch the GPU writer silently drops CRS
        # on files whose original CRS only resolves to WKT (no recognized
        # EPSG).
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
            nodata = _resolve_nodata_attr(data.attrs)
        # Mirror the CPU writer's pass-through of GDAL metadata, the
        # extra_tags list, the friendly image_description / extra_samples
        # / colormap synthesis, and the resolution tags. Without these,
        # a GPU write -> CPU read round-trip silently drops every rich
        # tag (#1563).
        _rich = _extract_rich_tags(data.attrs)
        raster_type = _rich['raster_type']
        gdal_meta_xml = _rich['gdal_metadata_xml']
        extra_tags_list = _rich['extra_tags']
        x_res = _rich['x_resolution']
        y_res = _rich['y_resolution']
        res_unit = _rich['resolution_unit']
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

    # Mirror the CPU writer's NaN-to-sentinel substitution (issue #1599).
    # Without this step the GPU writer emits raw NaN bytes interleaved
    # with valid data even when ``nodata=<finite>`` is supplied; the
    # GDAL_NODATA tag still advertises the sentinel but external readers
    # (rasterio / GDAL / QGIS) mask only on the sentinel value and
    # therefore see the NaN pixels as valid data. The CPU writer does
    # the equivalent rewrite at ``to_geotiff`` (lines around
    # ``arr.copy(); arr[nan_mask] = arr.dtype.type(nodata)``); both
    # paths must produce byte-equivalent files for the same input.
    # We always copy before the in-place sentinel write. Some upstream
    # branches above already produce a fresh buffer (``cupy.asarray``
    # from numpy/dask, ``ascontiguousarray`` from the band-first
    # moveaxis); others (a CuPy-backed DataArray taking the no-moveaxis
    # path, or a plain CuPy positional ``data``) hand ``arr`` back as
    # the caller's buffer. Rather than tracking provenance across that
    # branch tree, copy unconditionally when we are about to mutate --
    # the cost is one GPU array allocation, only on the NaN-present
    # path, and it guarantees the CPU writer's defensive-copy semantics
    # in every case.
    if (nodata is not None
            and np_dtype.kind == 'f'
            and not np.isnan(float(nodata))):
        nan_mask = cupy.isnan(arr)
        if bool(nan_mask.any()):
            arr = arr.copy()
            arr[nan_mask] = np_dtype.type(nodata)

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
            # Auto-generated lists hold actual decimation factors (2,
            # 4, 8, ...) so the loop below treats auto-generated and
            # user-supplied lists identically (issue #1766).
            overview_levels = []
            oh, ow = height, width
            factor = 2
            while (oh > tile_size and ow > tile_size and
                   len(overview_levels) < _MAX_OVERVIEW_LEVELS):
                oh //= 2
                ow //= 2
                if oh > 0 and ow > 0:
                    overview_levels.append(factor)
                    factor *= 2
        else:
            # Validate explicit lists: power-of-two factors >= 2,
            # strictly increasing, feasible for the input shape.
            # Previously the values were ignored and only the list
            # length mattered (issue #1766).
            from ._writer import _validate_overview_levels
            overview_levels = _validate_overview_levels(
                overview_levels, height=height, width=width)

        # Pass ``nodata`` so the GPU reducer masks the sentinel back to
        # NaN before averaging. Without this, the NaN->sentinel rewrite
        # done above on ``arr`` leaks the sentinel into the overview
        # reduction and poisons the pyramid (issue #1613). Rewrite any
        # all-sentinel cell NaN back to the sentinel after each level
        # so the on-disk overview tiles still carry the sentinel value
        # external readers expect.
        current = arr
        cumulative_factor = 1
        for target_factor in overview_levels:
            # Halve repeatedly until the cumulative decimation matches
            # the requested factor. Validation has already established
            # that ``target_factor`` is a power of two and strictly
            # greater than ``cumulative_factor``.
            while cumulative_factor < target_factor:
                current = make_overview_gpu(current, method=overview_resampling,
                                            nodata=nodata)
                cumulative_factor *= 2
                if (nodata is not None
                        and np.dtype(str(current.dtype)).kind == 'f'
                        and not np.isnan(float(nodata))):
                    nan_mask = cupy.isnan(current)
                    if bool(nan_mask.any().item()):
                        current = current.copy()
                        current[nan_mask] = np.dtype(
                            str(current.dtype)).type(nodata)
            oh, ow = current.shape[:2]
            parts.append(_gpu_compress_to_part(current, ow, oh, samples))

    file_bytes = _assemble_tiff(
        width, height, np_dtype, comp_tag, pred_val, True, tile_size,
        parts, geo_transform, epsg, nodata,
        is_cog=(cog and len(parts) > 1),
        raster_type=raster_type,
        crs_wkt=wkt_fallback if epsg is None else None,
        gdal_metadata_xml=gdal_meta_xml,
        extra_tags=extra_tags_list,
        x_resolution=x_res,
        y_resolution=y_res,
        resolution_unit=res_unit,
        force_bigtiff=bigtiff,
        photometric=photometric,
    )

    _write_bytes(file_bytes, path)


def read_vrt(source: str, *,
             dtype: str | np.dtype | None = None,
             window: tuple | None = None,
             band: int | None = None,
             name: str | None = None,
             chunks: int | tuple | None = None,
             gpu: bool = False,
             max_pixels: int | None = None,
             missing_sources: str = 'raise') -> xr.DataArray:
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
    max_pixels : int or None
        Maximum allowed pixel count (width * height * samples) for the
        assembled VRT region. None uses the reader default (~1 billion).
        Matches ``open_geotiff`` / ``read_geotiff_dask`` /
        ``read_geotiff_gpu``.
    missing_sources : {'raise', 'warn'}, default 'raise'
        Policy for unreadable source files referenced by the VRT.
        ``'raise'`` (the default since #1860) fails immediately on an
        unreadable backing source so a partial mosaic never surfaces
        silently. This matches the internal ``_vrt.read_vrt`` default
        and the rest of the geotiff module's up-front rejection of
        malformed input. Prior to #1860 the public default was
        ``'warn'``; callers that relied on the lenient behaviour pass
        ``missing_sources='warn'`` explicitly.
        ``'warn'`` is the opt-in escape hatch for partial mosaics: it
        emits ``GeoTIFFFallbackWarning``, records ``attrs['vrt_holes']``,
        and returns the mosaic with holes left as the band's nodata
        sentinel (or zero on integer bands without a sentinel).
        ``XRSPATIAL_GEOTIFF_STRICT=1`` forces a raise across the whole
        module regardless of this kwarg.

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

    Source-path containment (issue #1671): every ``<SourceFilename>`` in
    the VRT must resolve (after canonicalising ``..`` segments and
    symlinks) to a path under the VRT's own directory.  Absolute paths
    pointing elsewhere are rejected with ``ValueError`` by default.
    Operators that legitimately need to mosaic files from outside the
    VRT directory can opt in by setting the
    ``XRSPATIAL_VRT_ALLOWED_ROOTS`` environment variable to a
    ``os.pathsep``-separated list of trusted directory roots; sources
    resolving under any listed root are then accepted.  A
    ``relativeToVRT='1'`` source that escapes the VRT directory (e.g.
    ``../../etc/passwd`` or a symlink to a file outside the directory)
    is rejected regardless of the allowlist.

    Lazy chunked reads (issue #1814): when ``chunks=`` is set, the
    returned DataArray wraps a dask graph that decodes one chunk
    window per task.  Construction does not materialise any pixels;
    only the VRT XML is parsed.  The eager read populates
    ``attrs['vrt_holes']`` from skipped sources; the chunked path does
    not aggregate per-task hole records, so that attribute is not set
    when ``chunks=`` is used.  Each worker still emits
    ``GeoTIFFFallbackWarning`` for missing sources.
    """
    from ._reader import _coerce_path
    from ._vrt import (
        read_vrt as _read_vrt_internal,
        _apply_integer_sentinel_mask as _vrt_apply_integer_sentinel_mask,
    )

    source = _coerce_path(source)

    # Reject non-positive chunk sizes up front so the VRT dask path
    # surfaces the same error as ``read_geotiff_dask`` (#1776). Without
    # this check ``chunks=0`` raised ``ZeroDivisionError`` deep in dask
    # and ``chunks=-1`` was silently accepted. ``chunks=None`` is the
    # default (eager read), so allow it through here.
    chunks = _validate_chunks_arg(chunks, allow_none=True)

    if missing_sources not in ('warn', 'raise'):
        raise ValueError(
            f"missing_sources must be 'warn' or 'raise', got "
            f"{missing_sources!r}")

    # Lazy chunked path (issue #1814). The eager call below materialises
    # the full mosaic on host RAM and then wraps the array via
    # ``.chunk()``, so chunks= gave no memory protection and gpu=True +
    # chunks= still assembled the full mosaic on the CPU before moving to
    # the device. When chunks= is set, dispatch to a delayed-per-window
    # builder so each task decodes only the sources intersecting its
    # destination window.
    if chunks is not None:
        return _read_vrt_chunked(
            source,
            window=window,
            band=band,
            name=name,
            chunks=chunks,
            gpu=gpu,
            dtype=dtype,
            max_pixels=max_pixels,
            missing_sources=missing_sources,
        )

    arr, vrt = _read_vrt_internal(
        source, window=window, band=band, max_pixels=max_pixels,
        missing_sources=missing_sources,
    )

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
        height, width = arr.shape[:2]
        if window is not None:
            r0 = max(0, window[0])
            c0 = max(0, window[1])
            coord_window = (r0, c0, r0 + height, c0 + width)
        else:
            coord_window = None
        coords = _coords_from_pixel_geometry(
            origin_x, origin_y, res_x, res_y, height, width,
            is_point=vrt.raster_type == 'point',
            window=coord_window,
        )
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
    # Surface skipped-source records as ``attrs['vrt_holes']`` so
    # callers can detect a partial mosaic by attribute lookup. Under
    # lenient mode (the default), the underlying ``_vrt.read_vrt``
    # already warned per skipped source but the warning is easy to
    # miss in a pipeline; the attr lets downstream code branch on
    # ``"vrt_holes" in da.attrs`` instead of monitoring the warnings
    # stream. Empty list is omitted so the attr only appears when
    # there is actually a hole. See issue #1734.
    if vrt.holes:
        attrs['vrt_holes'] = list(vrt.holes)
    # When a specific band is selected, source its nodata from that
    # band's <NoDataValue> instead of band 0's. Otherwise multi-band
    # VRTs with per-band sentinels would mis-mask the read: attrs would
    # advertise band 0's sentinel, the integer-promotion block below
    # would mask against band 0's sentinel, and band N's actual nodata
    # pixels would survive as literal integers. See issue #1598.
    # ``band`` has already been validated by ``_vrt.read_vrt`` as
    # 0 <= band < len(vrt.bands), so a simple lookup is safe here.
    nodata = None
    if vrt.bands:
        band_idx_for_nodata = band if band is not None else 0
        nodata = vrt.bands[band_idx_for_nodata].nodata
        if nodata is not None:
            attrs['nodata'] = nodata

    # Mirror the integer-with-nodata promotion that open_geotiff /
    # read_geotiff_dask / read_geotiff_gpu apply post-decode. The VRT
    # internal reader NaN-masks float source arrays inline (see
    # ``_vrt._read_data``) but leaves integer sentinels untouched. Without
    # this branch, ``attrs['nodata']`` would be set while the array still
    # carried the literal sentinel value, breaking the convention that
    # downstream code follows (``attrs['nodata']`` is present iff the
    # array has already been NaN-masked).
    #
    # The helper handles both per-band masking (multi-band reads where
    # each band has its own ``<NoDataValue>``) and single-band masking,
    # promoting ``arr`` to float64 on the first sentinel hit and writing
    # NaNs in place on the promoted view. Shared with the chunked path
    # (issue #1825) so behaviour stays in lockstep. See issue #1611.
    arr = _vrt_apply_integer_sentinel_mask(arr, vrt, band)

    # Surface the source GeoTransform in the same rasterio ordering used
    # by open_geotiff: (pixel_width, 0, origin_x, 0, pixel_height, origin_y).
    # vrt.geo_transform is GDAL ordering, so reorder. For a windowed read
    # the origin shifts by (col_offset * res_x, row_offset * res_y).
    if gt is not None:
        if window is not None:
            tt_window = (max(0, window[0]), max(0, window[1]), 0, 0)
        else:
            tt_window = None
        attrs['transform'] = _transform_tuple_from_pixel_geometry(
            origin_x, origin_y, res_x, res_y, window=tt_window,
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

    # ``chunks is not None`` is handled by ``_read_vrt_chunked`` higher up
    # in this function (issue #1814); reaching this point implies the
    # eager path, so no post-decode chunking is needed.
    return result


# Hard cap on the per-VRT chunk task count. Matches the
# ``_MAX_DASK_CHUNKS`` value used by ``read_geotiff_dask`` so the two
# entry points refuse the same scheduler-busting chunk grids. See
# issue #1814.
_MAX_VRT_DASK_CHUNKS = 50_000


def _vrt_chunk_read(source, r0, c0, r1, c1, *,
                    band, max_pixels, missing_sources,
                    declared_dtype, gpu, parsed_vrt):
    """Decode a single chunk window from a VRT.

    Called by ``dask.delayed`` from :func:`_read_vrt_chunked`. The
    function reads only the destination window via the existing VRT
    internal reader, applies the same integer-sentinel masking the
    eager :func:`read_vrt` does post-decode, casts to the dtype the
    dask graph declared up front, and optionally moves the block to
    the GPU.

    ``parsed_vrt`` is the parent dispatcher's already-parsed
    :class:`VRTDataset`; the internal reader skips the XML parse and
    source-path containment check when this is supplied, removing the
    N+1 parse cost an earlier implementation had (issue #1825).

    Returning a ``numpy.ndarray`` (or ``cupy.ndarray`` when ``gpu`` is
    set) whose shape and dtype match the ``shape=`` / ``dtype=`` kwargs
    of the surrounding ``dask.array.from_delayed`` is the contract; a
    mismatch would silently produce a wrong-shape dask array.
    """
    from ._vrt import (
        read_vrt as _read_vrt_internal,
        _apply_integer_sentinel_mask,
    )

    arr, vrt = _read_vrt_internal(
        source, window=(r0, c0, r1, c1), band=band,
        max_pixels=max_pixels, missing_sources=missing_sources,
        parsed=parsed_vrt,
    )

    # Mirror the eager post-decode integer-sentinel masking via the
    # shared helper. The internal reader NaN-masks float source arrays
    # inline but leaves integer sentinels untouched, so the eager path
    # promotes to float64 when sentinels hit. The surrounding dask graph
    # already declared float64 when any band has a representable integer
    # sentinel, so any chunk that actually fires the mask returns a
    # buffer whose dtype matches the declared one.
    arr = _apply_integer_sentinel_mask(arr, vrt, band)

    if declared_dtype is not None and arr.dtype != declared_dtype:
        arr = arr.astype(declared_dtype)

    if gpu:
        import cupy
        arr = cupy.asarray(arr)

    return arr


def _read_vrt_chunked(source, *, window, band, name, chunks, gpu, dtype,
                      max_pixels, missing_sources):
    """Lazy ``read_vrt`` dispatch when ``chunks=`` is set (issue #1814).

    Parses the VRT XML once to recover the extent, CRS, GeoTransform,
    and per-band metadata, then builds a dask graph with one task per
    chunk window. Each task calls into the existing VRT internal reader
    with its own ``window=`` so only the sources intersecting the
    chunk's destination rectangle are decoded.

    ``attrs['vrt_holes']`` is populated from a parse-time
    ``os.path.exists`` sweep over every source referenced by the parsed
    VRT; this preserves the eager-path contract documented in #1734 so
    callers switching from eager to chunked can still detect partial
    mosaics by attribute lookup (rather than monitoring the
    ``GeoTIFFFallbackWarning`` stream). The check is a static
    approximation of the eager path's per-source decode-time exception
    handling: it catches the dominant "missing file" case but does not
    detect decode-time codec failures, which surface as per-task
    ``GeoTIFFFallbackWarning`` from each worker.
    """
    import os as _os
    import dask
    import dask.array as da

    from ._reader import MAX_PIXELS_DEFAULT
    from ._vrt import (
        parse_vrt,
        _read_vrt_xml,
        _effective_dtype_for_bands,
        _sentinel_for_dtype,
    )

    # Parse the VRT XML up-front (cheap; no pixel decode). Route through
    # ``_read_vrt_xml`` so the 64 MiB ``XRSPATIAL_VRT_MAX_XML_BYTES`` cap
    # added in #1818 applies to the chunked dispatcher too; a raw
    # ``open().read()`` here would let a multi-GB attacker-supplied VRT
    # exhaust memory before any parser-side guard fires (issue #1831).
    # The parsed VRTDataset is plumbed into every per-chunk task so each
    # task can skip the redundant XML parse and source-path allowlist
    # validation the internal reader otherwise performs (issue #1825).
    xml_str = _read_vrt_xml(source)
    vrt_dir = _os.path.dirname(_os.path.abspath(source))
    vrt = parse_vrt(xml_str, vrt_dir)

    # Validate ``band`` against the parsed band count, matching the
    # internal reader's contract so the failure mode is the same whether
    # the user reads eagerly or chunked.
    if band is not None:
        if not isinstance(band, (int, np.integer)) or isinstance(band, bool):
            raise ValueError(
                f"band must be a non-negative int, got {band!r}")
        if band < 0 or band >= len(vrt.bands):
            raise ValueError(
                f"band index {band} out of range for VRT with "
                f"{len(vrt.bands)} band(s)")

    # Resolve the windowed extent against the VRT.
    if window is not None:
        r0, c0, r1, c1 = window
        if (r0 < 0 or c0 < 0
                or r1 > vrt.height or c1 > vrt.width
                or r0 >= r1 or c0 >= c1):
            raise ValueError(
                f"window={window} is outside the VRT extent "
                f"({vrt.height}x{vrt.width}) or has non-positive size.")
        win_r0, win_c0 = r0, c0
        full_h, full_w = r1 - r0, c1 - c0
    else:
        win_r0, win_c0 = 0, 0
        full_h, full_w = vrt.height, vrt.width

    max_pixels_effective = (
        max_pixels if max_pixels is not None else MAX_PIXELS_DEFAULT
    )

    # Up-front pixel-count guard against the windowed extent. Mirrors
    # the eager ``_vrt.read_vrt`` (which calls ``_check_dimensions`` on
    # the full output shape) and ``read_geotiff_dask`` (which guards
    # ``full_h * full_w * eff_bands`` before scheduling any task). Each
    # chunk task additionally re-checks via ``max_pixels`` through the
    # internal reader, but catching an oversized request up front saves
    # the caller from a misleading per-chunk error.
    eff_bands = 1 if band is not None else max(1, len(vrt.bands))
    if full_h * full_w * eff_bands > max_pixels_effective:
        raise ValueError(
            f"Requested region {full_h}x{full_w}x{eff_bands} exceeds "
            f"max_pixels={max_pixels_effective:,}.")

    if isinstance(chunks, int):
        ch_h = ch_w = chunks
    else:
        ch_h, ch_w = chunks

    # Refuse chunk grids that would build more tasks than the scheduler
    # can hold without OOMing the driver. ``read_geotiff_dask`` uses the
    # same cap with the same suggestion logic (see issue #1814 and the
    # ``_MAX_DASK_CHUNKS`` guard upstream).
    n_chunks = ((full_h + ch_h - 1) // ch_h) * ((full_w + ch_w - 1) // ch_w)
    if n_chunks > _MAX_VRT_DASK_CHUNKS:
        scale = math.sqrt(n_chunks / _MAX_VRT_DASK_CHUNKS)
        suggested_h = int(math.ceil(ch_h * scale))
        suggested_w = int(math.ceil(ch_w * scale))
        raise ValueError(
            f"read_vrt: chunks=({ch_h}, {ch_w}) on a "
            f"{full_h}x{full_w} VRT region would produce {n_chunks:,} "
            f"dask tasks, exceeding the {_MAX_VRT_DASK_CHUNKS:,}-task "
            f"cap. Pass a larger chunks=... value explicitly (e.g. "
            f"chunks=({suggested_h}, {suggested_w}) keeps the task "
            f"count under the cap)."
        )

    # Select bands for shape/dtype declaration.
    if band is not None:
        selected_bands = [vrt.bands[band]]
    else:
        selected_bands = vrt.bands

    if not selected_bands:
        raise ValueError(
            "VRT has no <VRTRasterBand> elements; cannot determine "
            "output dtype")

    # Compute the declared dtype. Share the per-band effective-dtype
    # rule (ComplexSource scale/offset promotes to float64) with the
    # eager path via ``_effective_dtype_for_bands`` so both paths agree
    # on the result_type (issue #1825). Then widen to float64 if any
    # selected band declares an integer nodata sentinel that round-trips
    # through the band's dtype.
    #
    # The eager path defers the promotion to runtime: it scans every
    # band's pixels and promotes only if at least one sentinel hits.
    # The chunked path cannot afford that scan up front (it would
    # require decoding the mosaic the dask graph was constructed to
    # defer), so this is a parse-time approximation. The trade-off:
    #   * if a band declares nodata and no chunk contains the
    #     sentinel, the chunked output is float64 where the eager
    #     output would have stayed integer (acceptable: the user
    #     asked the source for nodata, so they expect NaN masking);
    #   * if a band does not declare nodata, both paths keep the
    #     source integer dtype (handled by the ``promotes is False``
    #     fall-through below).
    # See also Copilot review on PR #1822.
    declared_dtype = _effective_dtype_for_bands(selected_bands)

    if declared_dtype.kind in ('u', 'i'):
        promotes = False
        for vrt_band in selected_bands:
            if _sentinel_for_dtype(vrt_band.nodata,
                                   declared_dtype) is not None:
                promotes = True
                break
        if promotes:
            declared_dtype = np.dtype(np.float64)

    out_has_band_axis = band is None and len(vrt.bands) > 1
    n_out_bands = len(selected_bands)

    # Build the dask graph: one ``from_delayed`` per chunk window. The
    # destination coordinate space is the VRT's full extent (or the
    # windowed extent), so chunk windows are computed relative to that
    # space and translated to absolute VRT coords before being passed
    # into the per-chunk reader.
    rows = list(range(0, full_h, ch_h))
    cols = list(range(0, full_w, ch_w))

    delayed_read = dask.delayed(_vrt_chunk_read)

    if gpu:
        import cupy
        meta = cupy.empty((0,) * (3 if out_has_band_axis else 2),
                          dtype=declared_dtype)
    else:
        meta = np.empty((0,) * (3 if out_has_band_axis else 2),
                        dtype=declared_dtype)

    dask_rows = []
    for r0c in rows:
        r1c = min(r0c + ch_h, full_h)
        dask_cols = []
        for c0c in cols:
            c1c = min(c0c + ch_w, full_w)
            if out_has_band_axis:
                block_shape = (r1c - r0c, c1c - c0c, n_out_bands)
            else:
                block_shape = (r1c - r0c, c1c - c0c)
            d = delayed_read(
                source,
                r0c + win_r0, c0c + win_c0,
                r1c + win_r0, c1c + win_c0,
                band=band,
                max_pixels=max_pixels_effective,
                missing_sources=missing_sources,
                declared_dtype=declared_dtype,
                gpu=gpu,
                parsed_vrt=vrt,
            )
            block = da.from_delayed(d, shape=block_shape,
                                    dtype=declared_dtype, meta=meta)
            dask_cols.append(block)
        dask_rows.append(da.concatenate(dask_cols, axis=1))

    dask_arr = da.concatenate(dask_rows, axis=0)

    # Optional user-requested dtype cast happens lazily on the dask
    # array so the per-chunk decode dtype stays predictable.
    if dtype is not None:
        target = np.dtype(dtype)
        _validate_dtype_cast(declared_dtype, target)
        dask_arr = dask_arr.astype(target)
        final_dtype = target
    else:
        final_dtype = declared_dtype

    # Coordinates: derive from the VRT GeoTransform and the windowed
    # extent. Mirrors the eager branch in ``read_vrt`` so chunked and
    # eager reads share the same x/y arrays.
    gt = vrt.geo_transform
    coords = {}
    attrs = {}
    if gt is not None:
        origin_x, res_x, _, origin_y, _, res_y = gt
        coord_window = (win_r0, win_c0, win_r0 + full_h, win_c0 + full_w)
        coords = _coords_from_pixel_geometry(
            origin_x, origin_y, res_x, res_y, full_h, full_w,
            is_point=vrt.raster_type == 'point',
            window=coord_window,
        )
        attrs['transform'] = _transform_tuple_from_pixel_geometry(
            origin_x, origin_y, res_x, res_y,
            window=(win_r0, win_c0, 0, 0),
        )

    if vrt.crs_wkt:
        epsg = _wkt_to_epsg(vrt.crs_wkt)
        if epsg is not None:
            attrs['crs'] = epsg
        attrs['crs_wkt'] = vrt.crs_wkt
    if vrt.raster_type == 'point':
        attrs['raster_type'] = 'point'

    # Surface the nodata sentinel for the selected band.
    nodata_meta = None
    if vrt.bands:
        band_idx_for_nodata = band if band is not None else 0
        nodata_meta = vrt.bands[band_idx_for_nodata].nodata
        if nodata_meta is not None:
            attrs['nodata'] = nodata_meta

    # Static hole detection: mirror the eager-path ``attrs['vrt_holes']``
    # contract (#1734) by scanning every source referenced in the parsed
    # VRT and recording the ones whose backing file does not exist on
    # disk. The eager path discovers holes at decode time (per-source
    # OSError / codec error) and aggregates them onto ``vrt.holes``;
    # under chunked dispatch each per-task decode catches its own
    # missing source and warns, but those records cannot be reduced
    # back onto the parent DataArray without an extra synchronisation
    # pass. The parse-time existence sweep catches the dominant
    # missing-file case before scheduling and lets callers branch on
    # ``"vrt_holes" in da.attrs`` exactly as with the eager reader.
    # Empty list is omitted so the attr only appears when a hole is
    # actually present. Each entry mirrors the eager schema:
    # ``{'source', 'band', 'dst_rect', 'error'}``.
    chunked_holes: list[dict] = []
    for vrt_band in vrt.bands:
        for src in vrt_band.sources:
            if not _os.path.exists(src.filename):
                chunked_holes.append({
                    'source': src.filename,
                    'band': vrt_band.band_num,
                    'dst_rect': (src.dst_rect.x_off, src.dst_rect.y_off,
                                 src.dst_rect.x_size, src.dst_rect.y_size),
                    'error': 'FileNotFoundError: source file not found',
                })
    if chunked_holes:
        attrs['vrt_holes'] = chunked_holes

    if out_has_band_axis:
        dims = ['y', 'x', 'band']
        coords['band'] = np.arange(n_out_bands)
    else:
        dims = ['y', 'x']

    if name is None:
        name = _os.path.splitext(_os.path.basename(source))[0]

    result = xr.DataArray(
        dask_arr, dims=dims, coords=coords, name=name, attrs=attrs,
    )
    # Sanity: the declared dtype on the dask array is what we return.
    assert result.dtype == final_dtype, (
        f"internal: result dtype {result.dtype} != declared {final_dtype}"
    )
    return result


def write_vrt(vrt_path: str, source_files: list[str], *,
              relative: bool = True,
              crs: int | str | None = None,
              crs_wkt: str | None = _CRS_WKT_DEPRECATED_SENTINEL,
              nodata: float | int | None = None) -> str:
    """Generate a VRT file that mosaics multiple GeoTIFF tiles.

    Parameters
    ----------
    vrt_path : str
        Output .vrt file path.
    source_files : list of str
        Paths to the source GeoTIFF files.
    relative : bool, optional
        Store source paths relative to the VRT file (default True).
    crs : int, str, or None, optional
        EPSG code (int), WKT string, or PROJ string. If None, the CRS
        is taken from the first source GeoTIFF. Mirrors the ``crs``
        kwarg on ``to_geotiff`` and ``write_geotiff_gpu`` so the same
        value can be forwarded to whichever writer the caller picked
        without per-writer special-casing (issue #1715).
    crs_wkt : str or None, optional
        Deprecated alias for ``crs``. Emits ``DeprecationWarning`` when
        supplied (including ``crs_wkt=None``); passing both ``crs`` and
        ``crs_wkt`` raises ``TypeError``. The value is forwarded through
        the same ``_resolve_crs_to_wkt`` path as ``crs``, so any string
        the resolver accepts (WKT root keyword, PROJ string,
        ``"EPSG:NNNN"``) and ``None`` work here. The historic
        ``str | None`` surface is preserved; new code should use ``crs``
        instead, which additionally accepts ``int`` EPSG codes.
    nodata : float, int, or None, optional
        NoData value. If None, taken from the first source GeoTIFF.
        Integer sentinels (e.g. ``65535`` for uint16, ``-9999`` for
        int32) are accepted so the surface lines up with the
        ``nodata`` kwarg on ``to_geotiff`` and ``write_geotiff_gpu``.

    Returns
    -------
    str
        Path to the written VRT file.
    """
    # Explicit signature (previously ``**kwargs``) so ``inspect.signature``,
    # IDE autocomplete, and ``mypy --strict`` can see the accepted kwargs
    # without parsing the docstring. Mirrors ``_vrt.write_vrt`` for the
    # historic ``crs_wkt`` path; the new ``crs`` path normalises through
    # ``_resolve_crs_to_wkt`` before forwarding because the internal
    # writer still only speaks WKT.
    crs_wkt_passed = crs_wkt is not _CRS_WKT_DEPRECATED_SENTINEL
    if crs is not None and crs_wkt_passed:
        # Both supplied is ambiguous regardless of whether the WKT happens
        # to encode the same CRS as the int. Refuse rather than silently
        # picking one.
        raise TypeError(
            "write_vrt: pass either 'crs' or the deprecated 'crs_wkt' "
            "alias, not both.")
    if crs_wkt_passed:
        warnings.warn(
            "write_vrt(..., crs_wkt=...) is deprecated; use crs=... "
            "instead. The kwarg was renamed for parity with to_geotiff "
            "and write_geotiff_gpu, which already accept 'crs' as either "
            "an int EPSG code or a WKT string.",
            DeprecationWarning,
            stacklevel=2,
        )
        crs = crs_wkt

    resolved_wkt = _resolve_crs_to_wkt(crs)

    from ._vrt import write_vrt as _write_vrt_internal
    return _write_vrt_internal(
        vrt_path, source_files,
        relative=relative,
        crs_wkt=resolved_wkt,
        nodata=nodata,
    )


def plot_geotiff(da: xr.DataArray, **kwargs):
    """Plot a DataArray using its embedded colormap if present.

    .. deprecated:: 0.10.0
        Use ``da.xrs.plot()`` instead. ``plot_geotiff`` is a thin wrapper
        kept for backward compatibility and will be removed in a future
        release.
    """
    warnings.warn(
        "plot_geotiff is deprecated and will be removed in a future "
        "release. Use ``da.xrs.plot()`` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return da.xrs.plot(**kwargs)
