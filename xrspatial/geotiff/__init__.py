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
write_vrt(path, source_files, ...)
    Generate a VRT mosaic XML from a list of GeoTIFF files. ``vrt_path``
    is kept as a deprecated alias for ``path``; passing both ``path`` and
    ``vrt_path`` raises ``TypeError`` (#1946).
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
    require_transform_for_georeferenced as _require_transform_for_georeferenced,
)
from ._errors import (
    ConflictingCRSError,
    ConflictingNodataError,
    GeoTIFFAmbiguousMetadataError,
    InvalidCRSCodeError,
    MixedBandMetadataError,
    NonUniformCoordsError,
    RotatedTransformError,
    UnparseableCRSError,
)
from ._geotags import GeoTransform, RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT
from ._reader import UnsafeURLError
# ``read_to_array`` is internal: it is used by ``open_geotiff`` and the
# GPU fallback below but is not in ``__all__`` or the module-level
# Public API docstring. Bind it under a leading-underscore name so it
# does not leak into ``xrspatial.geotiff``'s public namespace. Tests
# and internal callers that genuinely need it can import directly from
# ``xrspatial.geotiff._reader``. See issue #1708.
from ._attrs import (
    _LEVEL_RANGES,
    _VALID_COMPRESSIONS,
    _extent_to_window,
    _extract_rich_tags,
    _populate_attrs_from_geo_info,
    _validate_read_geo_info,
    _resolve_nodata_attr,
    _set_nodata_attrs,
)
from ._backends._gpu_helpers import _is_gpu_data
# Re-export only; called by xrspatial/geotiff/tests/test_nodata_*.py.
from ._backends._gpu_helpers import _apply_nodata_mask_gpu  # noqa: F401
from ._backends.dask import read_geotiff_dask
from ._backends.gpu import read_geotiff_gpu
from ._backends.vrt import read_vrt
from ._crs import _resolve_crs_to_wkt, _wkt_to_epsg
from ._reader import (
    _MAX_CLOUD_BYTES_SENTINEL,
    read_to_array as _read_to_array,
)
from ._runtime import (
    GeoTIFFFallbackWarning,
    _CRS_WKT_DEPRECATED_SENTINEL,
    _GPU_DEPRECATED_SENTINEL,
    _MISSING_SOURCES_SENTINEL,
    _ON_GPU_FAILURE_SENTINEL,
    _geotiff_strict_mode,
    _gpu_fallback_warning_message,
)
from ._validation import (
    _validate_3d_writer_dims,
    _validate_chunks_arg,
    _validate_dtype_cast,
    _validate_tile_size_arg,
)
from ._writer import write
from ._writers.eager import to_geotiff
# Re-export only; called by xrspatial/geotiff/tests/test_nodata_no_extra_copy_1553.py.
from ._writers.eager import _write_single_tile  # noqa: F401
from ._writers.gpu import write_geotiff_gpu
from ._writers.vrt import write_vrt

# All names below are part of the supported public API. ``plot_geotiff``
# is intentionally omitted: it is deprecated in favour of ``da.xrs.plot()``
# and emits a ``DeprecationWarning`` when called.
__all__ = [
    'ConflictingCRSError',
    'ConflictingNodataError',
    'GeoTIFFAmbiguousMetadataError',
    'GeoTIFFFallbackWarning',
    'InvalidCRSCodeError',
    'MixedBandMetadataError',
    'NonUniformCoordsError',
    'RotatedTransformError',
    'UnparseableCRSError',
    'UnsafeURLError',
    'open_geotiff',
    'read_geotiff_gpu',
    'read_geotiff_dask',
    'read_vrt',
    'to_geotiff',
    'write_geotiff_gpu',
    'write_vrt',
]


def _read_geo_info(source, *, overview_level: int | None = None,
                   allow_rotated: bool = False):
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
    allow_rotated : bool, optional
        Forwarded to the geotag parser. When True, a rotated
        ``ModelTransformationTag`` reads as an ungeoreferenced pixel
        grid instead of raising ``NotImplementedError`` (issue #2115).
    """
    from ._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
    from ._geotags import extract_geo_info_with_overview_inheritance
    from ._header import parse_all_ifds, parse_header, select_overview_ifd
    from ._reader import (
        _CloudSource, _coerce_path, _is_file_like, _is_fsspec_uri,
        _parse_cog_http_meta,
    )
    from ._validation import _validate_predictor_sample_format

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
                _src, overview_level=overview_level,
                allow_rotated=allow_rotated)
        finally:
            _src.close()
        bps = resolve_bits_per_sample(_ifd.bits_per_sample)
        file_dtype = tiff_dtype_to_numpy(bps, _ifd.sample_format)
        _validate_predictor_sample_format(_ifd.predictor, _ifd.sample_format)
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
    sidecar = None
    try:
        header = parse_header(data)
        ifds = parse_all_ifds(data, header)
        if not ifds:
            raise ValueError("No IFDs found in TIFF file")
        # Append sibling `.tif.ovr` sidecar IFDs onto the pyramid list
        # so ``overview_level`` indexes both internal and external
        # overviews (issue #2112). Local file paths only.
        from ._sidecar import (
            attach_sidecar_origin, find_sidecar, load_sidecar,
        )
        sidecar_path = find_sidecar(source)
        if sidecar_path is not None:
            sidecar = load_sidecar(sidecar_path)
            # Metadata-only path: drop the origin mapping. The reader
            # only needs the merged IFD list to resolve the requested
            # ``overview_level`` against; strip/tile bytes are sliced by
            # ``read_to_array`` on the actual read.
            attach_sidecar_origin(
                sidecar.ifds, sidecar.data, sidecar.header)
            ifds = ifds + sidecar.ifds
        ifd = select_overview_ifd(ifds, overview_level)
        # Inherit georef from the level-0 IFD when the overview itself
        # has no geokeys (issue #1640). Pass-through for level 0. The
        # sidecar IFDs typically lack geokeys so the inheritance pulls
        # from the base file's full-resolution IFD as GDAL does.
        geo_info = extract_geo_info_with_overview_inheritance(
            ifd, ifds, data, header.byte_order,
            allow_rotated=allow_rotated)
        bps = resolve_bits_per_sample(ifd.bits_per_sample)
        file_dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
        _validate_predictor_sample_format(ifd.predictor, ifd.sample_format)
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
        from ._sidecar import close_sidecar
        close_sidecar(sidecar)


def open_geotiff(source: str | BinaryIO, *,
                 dtype: str | np.dtype | None = None,
                 window: tuple | None = None,
                 overview_level: int | None = None,
                 band: int | None = None,
                 name: str | None = None,
                 chunks: int | tuple | None = None,
                 gpu: bool = False,
                 max_pixels: int | None = None,
                 max_cloud_bytes: int | None = _MAX_CLOUD_BYTES_SENTINEL,  # type: ignore[assignment]
                 on_gpu_failure: str = _ON_GPU_FAILURE_SENTINEL,
                 missing_sources: str = _MISSING_SOURCES_SENTINEL,
                 allow_rotated: bool = False,
                 allow_unparseable_crs: bool = False,
                 band_nodata: str | None = None,
                 mask_nodata: bool = True,
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
        Overview level (0 = full resolution). Must be a non-negative int
        or ``None``; passing ``bool`` or any other type raises
        ``TypeError``.
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
    max_cloud_bytes : int or None, optional
        Byte ceiling for eager reads from fsspec sources (``s3://``,
        ``gs://``, ``az://``, ``abfs://``, ``memory://``, ...). The
        compressed object size is checked against this budget before
        any bytes are downloaded. Default is 256 MiB, overridable via
        the ``XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES`` env var. Pass
        ``None`` to skip the check entirely. The HTTP path already
        reads only what it needs via range requests and is not subject
        to this limit. Has no effect on local file or file-like
        sources. Passing this kwarg with ``gpu=True``, ``chunks=...``,
        or a ``.vrt`` source raises ``ValueError`` because those
        backends do not apply the cloud-byte budget. See issue #1928
        (eager path) and issue #1974 (rejection guard).
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
    band_nodata : {'first', None}, optional
        VRT-only. Opt-out for the fail-closed check that rejects VRT
        sources whose bands declare disagreeing per-band nodata
        sentinels (issue #1987 PR 5). When ``None`` (the default), a VRT
        that mosaics bands with different sentinels raises
        ``MixedBandMetadataError``; flattening to one value would let
        one band's valid pixels collide with another band's sentinel.
        Pass ``band_nodata='first'`` to keep the legacy behaviour of
        using band 0's sentinel for the whole mosaic. Passing this
        kwarg with a non-VRT source raises ``ValueError`` because the
        policy only applies to the VRT pipeline.
    mask_nodata : bool, default True
        If True (the default), replace the nodata sentinel with ``NaN``;
        integer rasters get promoted to ``float64`` first so NaN can be
        represented. If False, skip the sentinel-to-NaN step and keep
        the source dtype. ``attrs['nodata']`` still carries the raw
        sentinel either way, so downstream code can mask explicitly.
        Pass ``mask_nodata=False`` when you want to preserve an integer
        source dtype via ``dtype=``: the default ``mask_nodata=True``
        promotes to ``float64`` whenever the sentinel matches an actual
        pixel, and ``dtype=<integer>`` then raises ``ValueError`` on the
        float-to-int cast.
    allow_rotated : bool, default False
        Read-side opt-in for rotated / sheared ``ModelTransformationTag``
        files. By default the reader raises ``NotImplementedError``
        because the rest of xrspatial assumes an axis-aligned grid.
        ``allow_rotated=True`` reads the pixel grid without the
        geospatial assumption: the result has integer pixel coords on
        ``x`` / ``y`` and both ``attrs['crs']`` and ``attrs['crs_wkt']``
        are dropped. The CRS attrs are dropped together with the
        transform because keeping them while the axis-aligned transform
        is gone misleads downstream code that gates on
        ``"crs" in da.attrs`` to mean the array is spatially usable
        (issue #2126). The contract is read-only -- ``to_geotiff`` does
        not currently emit rotated transforms, so a read-then-write
        round-trip writes an identity-affine output and silently drops
        the rotation (issue #2115).

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
    code works uniformly. To keep the source dtype on a file whose
    sentinel matches actual pixels, pass ``mask_nodata=False``; the raw
    sentinel stays in the data and ``attrs['nodata']`` still carries it.
    Passing ``dtype=<integer>`` on its own is not enough: the
    sentinel-to-NaN promotion runs first and the subsequent integer cast
    then raises ``ValueError`` (float-to-int is lossy in a way users
    rarely intend). When the file has no in-range sentinel match, the
    promotion is skipped and ``dtype=<integer>`` works either way.
    """
    from ._reader import _coerce_path

    source = _coerce_path(source)

    # Reject bool and non-int ``overview_level`` up front (issue #2074).
    # Without this guard, ``overview_level=True`` is coerced to ``1`` and
    # silently returns the first overview level, and non-int types leak
    # raw ``TypeError`` messages from the internal numeric comparison or
    # list indexing.
    from ._validation import _validate_overview_level_arg
    _validate_overview_level_arg(overview_level)

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

    # ``band_nodata`` is the #1987 PR 5 opt-out for the mixed-band
    # metadata fail-closed check. It only has meaning on the VRT pipeline
    # (a plain GeoTIFF has one nodata sentinel per file, not per band),
    # so reject the kwarg up front on non-VRT sources rather than letting
    # it leak into ``read_vrt`` and confuse the caller about what the
    # opt-out actually controls.
    if band_nodata is not None and not _is_vrt_source:
        raise ValueError(
            "band_nodata only applies to VRT sources. "
            "Pass a .vrt path to enable the VRT pipeline, or drop "
            "band_nodata to keep the default GeoTIFF path. "
            "See issue #1987.")

    # ``max_cloud_bytes`` is the eager fsspec-read budget. Only
    # ``_read_to_array`` on the eager non-VRT, non-GPU, non-dask branch
    # consumes it; the GPU (``read_geotiff_gpu``), dask
    # (``read_geotiff_dask``), and VRT (``read_vrt``) branches all ignore
    # the kwarg silently. Reject it up front on those paths so callers
    # learn the budget is being dropped, matching the
    # ``on_gpu_failure`` / ``missing_sources`` guards above and the
    # silently-drops-backend-kwarg fixes in #1561 / #1605 / #1685 / #1810.
    # See issue #1974.
    if max_cloud_bytes is not _MAX_CLOUD_BYTES_SENTINEL:
        if _is_vrt_source:
            raise ValueError(
                "max_cloud_bytes is not supported for VRT sources. "
                "The VRT reader does not apply the cloud-byte budget; "
                "drop the kwarg, or call open_geotiff on the underlying "
                ".tif source.")
        if gpu:
            raise ValueError(
                "max_cloud_bytes is not supported when gpu=True. "
                "The GPU reader does not apply the cloud-byte budget; "
                "drop the kwarg, or pass gpu=False to use the eager "
                "CPU path.")
        if chunks is not None:
            raise ValueError(
                "max_cloud_bytes is not supported when chunks=... (dask). "
                "The dask reader does not apply the cloud-byte budget; "
                "drop the kwarg, or drop chunks to use the eager path.")

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
                        max_pixels=max_pixels,
                        allow_rotated=allow_rotated,
                        allow_unparseable_crs=allow_unparseable_crs,
                        band_nodata=band_nodata,
                        mask_nodata=mask_nodata,
                        **vrt_kwargs)

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
                                allow_rotated=allow_rotated,
                                allow_unparseable_crs=allow_unparseable_crs,
                                mask_nodata=mask_nodata,
                                **gpu_kwargs)

    # Dask path (CPU)
    if chunks is not None:
        return read_geotiff_dask(source, dtype=dtype, chunks=chunks,
                                 overview_level=overview_level,
                                 window=window, band=band,
                                 max_pixels=max_pixels, name=name,
                                 allow_rotated=allow_rotated,
                                 allow_unparseable_crs=allow_unparseable_crs,
                                 mask_nodata=mask_nodata)

    kwargs = {}
    if max_pixels is not None:
        kwargs['max_pixels'] = max_pixels
    if max_cloud_bytes is not _MAX_CLOUD_BYTES_SENTINEL:
        kwargs['max_cloud_bytes'] = max_cloud_bytes

    # ``read_to_array`` validates ``window`` against the selected IFD's
    # extent and raises ``ValueError`` for out-of-bounds windows with
    # the same message format as the dask path's pre-flight validator
    # in :func:`read_geotiff_dask`. That keeps the two backends in sync
    # on the contract without forcing a second metadata parse here. See
    # issue #1634.
    arr, geo_info = _read_to_array(
        source, window=window,
        overview_level=overview_level, band=band,
        allow_rotated=allow_rotated,
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

    # Issue #1987 ambiguous-metadata checks. Run before attrs population
    # so a rejected file does not leak a partly-populated attrs dict.
    _validate_read_geo_info(
        geo_info, window=window,
        allow_rotated=allow_rotated,
        allow_unparseable_crs=allow_unparseable_crs,
    )

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
    if nodata is not None and mask_nodata:
        # When the reader applied MinIsWhite, the sentinel-equality mask
        # must compare against the inverted sentinel value (issue #1809).
        # ``read_to_array`` / ``_read_cog_http`` stash that value on
        # ``geo_info._mask_nodata``; fall back to the original sentinel
        # on non-MinIsWhite files. Renamed from ``mask_nodata`` to
        # ``nodata_sentinel`` so the local does not shadow the
        # ``mask_nodata`` opt-out kwarg (#2052).
        nodata_sentinel = getattr(geo_info, '_mask_nodata', nodata)
        if arr.dtype.kind == 'f':
            if nodata_sentinel is not None and not np.isnan(nodata_sentinel):
                arr[arr == arr.dtype.type(nodata_sentinel)] = np.nan
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
            if (nodata_sentinel is not None
                    and np.isfinite(nodata_sentinel)
                    and float(nodata_sentinel).is_integer()):
                nodata_int = int(nodata_sentinel)
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

    # Set ``attrs['nodata']`` + ``attrs['masked_nodata']`` after the
    # mask + optional dtype cast so ``masked_nodata`` reflects the
    # final array dtype (issue #1988).
    _set_nodata_attrs(attrs, nodata, array_dtype=arr.dtype)

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
