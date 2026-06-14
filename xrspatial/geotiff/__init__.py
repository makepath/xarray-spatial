"""Lightweight GeoTIFF/COG reader and writer.

No GDAL dependency -- uses only numpy, numba, xarray, and the standard library.

Public API
----------
open_geotiff(source, ...)
    Read a GeoTIFF, COG, or VRT file to an xarray.DataArray. The backend is
    chosen from the parameters: ``gpu=True`` returns a CuPy-backed array,
    ``chunks=N`` returns a windowed lazy dask array, a ``.vrt`` source reads
    a GDAL Virtual Raster Table, and the default is an eager numpy read.
to_geotiff(data, path, ...)
    Write an xarray.DataArray as a GeoTIFF or COG. The backend is chosen
    from the data and parameters: CuPy-backed data or ``gpu=True`` writes
    through the GPU (nvCOMP) path, a ``.vrt`` output path writes a directory
    of tiled GeoTIFFs plus a VRT index, and the default is an eager CPU
    write.

VRT mosaics are written by passing a ``.vrt`` path to ``to_geotiff``; the
underlying index emitter (``_build_vrt``) is internal and not part of the
public surface.

The backend functions ``_read_geotiff_gpu``, ``_read_geotiff_dask``,
``_read_vrt``, and ``_write_geotiff_gpu`` are private. ``open_geotiff`` and
``to_geotiff`` dispatch to them. They are bound on the package
(``xrspatial.geotiff._read_geotiff_gpu``) and also importable from their
backend modules; reach for them only to bypass auto-dispatch.
"""
from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from typing import BinaryIO

# Re-exports only; consumers import these as ``xrspatial.geotiff._coords_from_pixel_geometry``
# ``read_to_array`` is internal: it is used by ``open_geotiff`` and the
# GPU fallback below but is not in ``__all__`` or the module-level
# Public API docstring. Bind it under a leading-underscore name so it
# does not leak into ``xrspatial.geotiff``'s public namespace. Tests
# and internal callers that genuinely need it can import directly from
# ``xrspatial.geotiff._reader``.
from ._attrs import (_LEVEL_RANGES, _VALID_COMPRESSIONS, GEOREF_STATUS_CRS_ONLY,  # noqa: F401
                     GEOREF_STATUS_FULL, GEOREF_STATUS_NONE, GEOREF_STATUS_ROTATED_DROPPED,
                     GEOREF_STATUS_TRANSFORM_ONLY, GEOREF_STATUS_VALUES, _extent_to_window,
                     _extract_rich_tags, _finalize_eager_read, _resolve_nodata_attr)
# Re-export only; called by xrspatial/geotiff/tests/test_nodata_*.py.
from ._backends._gpu_helpers import _apply_nodata_mask_gpu  # noqa: F401
from ._backends._gpu_helpers import _is_gpu_data  # noqa: F401
from ._backends.dask import _read_geotiff_dask
from ._backends.gpu import _read_geotiff_gpu
from ._backends.vrt import _read_vrt
from ._coords import _BAND_DIM_NAMES  # noqa: F401
from ._coords import coords_from_pixel_geometry as _coords_from_pixel_geometry  # noqa: F401
from ._coords import coords_to_transform as _coords_to_transform  # noqa: F401
from ._coords import \
    require_transform_for_georeferenced as _require_transform_for_georeferenced  # noqa: F401
from ._coords import transform_from_attr as _transform_from_attr  # noqa: F401
from ._coords import transform_tuple as _transform_tuple  # noqa: F401
from ._coords import \
    transform_tuple_from_pixel_geometry as _transform_tuple_from_pixel_geometry  # noqa: F401
from ._crs import _resolve_crs_to_wkt, _wkt_to_epsg  # noqa: F401
from ._errors import (ConflictingCRSError, ConflictingNodataError, DuplicateIFDTagError,
                      GeoTIFFAmbiguousMetadataError, InconsistentGeoKeysError, InvalidCRSCodeError,
                      InvalidIntegerNodataError, MalformedScaleOffsetError, MixedBandMetadataError,
                      NonRepresentableEPSGCRSError, NonUniformCoordsError,
                      RemoteStableSourcesOnlyError, RotatedTransformError, UnknownCRSModelTypeError,
                      UnparseableCRSError, UnsupportedGeoTIFFFeatureError,
                      VRTStableSourcesOnlyError, VRTUnsupportedError)
from ._geotags import RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT, GeoTransform  # noqa: F401
# ``PixelSafetyLimitError`` is defined in ``_layout`` and re-exported by
# ``_reader`` (the historical import surface); bind it here so callers can
# catch the ``max_pixels`` rejection without a private-module import.
from ._reader import (_MAX_CLOUD_BYTES_SENTINEL, CloudSizeLimitError, PixelSafetyLimitError,
                      UnsafeURLError)
from ._reader import read_to_array as _read_to_array
from ._runtime import (_CRS_WKT_DEPRECATED_SENTINEL, _GPU_DEPRECATED_SENTINEL,  # noqa: F401
                       _MASK_AND_SCALE_DEPRECATED_SENTINEL, _MASK_NODATA_DEPRECATED_SENTINEL,
                       _MISSING_SOURCES_SENTINEL, _NAME_DEPRECATED_SENTINEL,
                       _ON_GPU_FAILURE_SENTINEL, GeoTIFFFallbackWarning, _geotiff_strict_mode,
                       _gpu_fallback_warning_message)
from ._validation import (_validate_3d_writer_dims, _validate_chunks_arg,  # noqa: F401
                          _validate_tile_size_arg)
# Re-export only; called by xrspatial/geotiff/tests/test_nodata_no_extra_copy_1553.py.
# ``_writer.write`` (alias for ``_writer._write``) is module-private;
# see the ``_writer.py`` docstring. The public eager write surface is
# :func:`to_geotiff`; do not re-export the array-level entry point here.
# The dotted path ``xrspatial.geotiff._writer._write`` still works for
# the handful of internal call sites that need it.
from ._writers.eager import _write_single_tile  # noqa: F401
from ._writers.eager import to_geotiff
# Re-export only: bound on the package so ``xrspatial.geotiff._write_geotiff_gpu``
# resolves for tests that monkeypatch it and callers bypassing auto-dispatch.
# ``to_geotiff`` reaches it via ``_writers.eager``; not called here directly.
from ._writers.gpu import _write_geotiff_gpu  # noqa: F401
# Re-export only: the internal VRT-index emitter. ``to_geotiff``'s ``.vrt``
# path reaches it via ``_writers.eager``; bound here so tests and internal
# callers can import ``xrspatial.geotiff._build_vrt``. Not public API.
from ._writers.vrt import _build_vrt  # noqa: F401

# All names below are part of the supported public API. ``plot_geotiff``
# is intentionally omitted: it is deprecated in favour of ``da.xrs.plot()``
# and emits a ``DeprecationWarning`` when called.
__all__ = [
    'CloudSizeLimitError',
    'ConflictingCRSError',
    'ConflictingNodataError',
    'DuplicateIFDTagError',
    'GeoTIFFAmbiguousMetadataError',
    'GeoTIFFFallbackWarning',
    'GEOREF_STATUS_CRS_ONLY',
    'GEOREF_STATUS_FULL',
    'GEOREF_STATUS_NONE',
    'GEOREF_STATUS_ROTATED_DROPPED',
    'GEOREF_STATUS_TRANSFORM_ONLY',
    'GEOREF_STATUS_VALUES',
    'InconsistentGeoKeysError',
    'InvalidCRSCodeError',
    'InvalidIntegerNodataError',
    'MalformedScaleOffsetError',
    'MixedBandMetadataError',
    'NonRepresentableEPSGCRSError',
    'NonUniformCoordsError',
    'PixelSafetyLimitError',
    'RemoteStableSourcesOnlyError',
    'RotatedTransformError',
    'SUPPORTED_FEATURES',
    'UnknownCRSModelTypeError',
    'UnparseableCRSError',
    'UnsafeURLError',
    'UnsupportedGeoTIFFFeatureError',
    'VRTStableSourcesOnlyError',
    'VRTUnsupportedError',
    'open_geotiff',
    'to_geotiff',
]


# ``SUPPORTED_FEATURES`` and its derived ``_EXPERIMENTAL_CODECS`` set
# live in ``_attrs.py`` so the writers can import them at module scope
# without a circular dependency (this ``__init__`` already imports the
# writers, so the writers cannot import from ``..`` at module scope).
# The names are re-exported below to keep the public API at
# ``xrspatial.geotiff.SUPPORTED_FEATURES``.
#
# Tier semantics
# --------------
# - ``"stable"`` -- the path a new user should be on. Local file in,
#   local file out, lossless codec, axis-aligned grid. Covered by the
#   cross-backend parity matrix.
# - ``"advanced"`` -- works and is tested, but the caller should know
#   what they are signing up for (cloud cost, partial VRT mosaics,
#   rotated transforms dropping on write, BigTIFF promotion, etc.). No
#   kwarg gate; the docstring carries an ``Advanced:`` marker.
# - ``"experimental"`` -- works in our tests, no claim about external
#   interop or numerical parity across backends. Tier 3 codecs
#   (``lerc``, ``jpeg2000`` / ``j2k``, ``lz4``) require
#   ``allow_experimental_codecs=True`` on the writers; the GPU paths
#   use ``gpu=True`` as the explicit opt-in.
# - ``"internal_only"`` -- the strictest tier. Already gated behind
#   its own dedicated flag because the output does not round-trip
#   through libtiff / GDAL / rasterio. ``codec.jpeg`` requires
#   ``allow_internal_only_jpeg=True``;
#   ``allow_experimental_codecs`` does NOT cover it.
#
# Tests in ``xrspatial/geotiff/tests/release_gates/test_features.py``
# walk the mapping and assert that every Tier 3 codec rejects without
# the opt-in flag and every Tier 4 codec rejects without its own
# dedicated flag. The user-guide notebook
# (``examples/user_guide/39_GeoTIFF_IO.ipynb``) renders the same
# mapping as a table so the documentation cannot drift from the code.
from ._attrs import SUPPORTED_FEATURES  # noqa: E402


def _read_geo_info(source, *, overview_level: int | None = None,
                   allow_rotated: bool = False,
                   allow_invalid_nodata: bool = False):
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
        grid instead of raising ``RotatedTransformError``.
    allow_invalid_nodata : bool, optional
        Forwarded to the geotag parser. When True, restores the legacy
        no-op handling of non-finite / fractional ``GDAL_NODATA`` on
        integer sources.
    """
    # ``_parse_cog_http_meta`` is imported from ``_cog_http`` directly
    # rather than re-routed through ``_reader`` because the
    # ``open_geotiff(..., chunks=...)`` fsspec metadata path is not part
    # of the ``_reader.*`` monkeypatch surface (no test patches
    # ``_reader._parse_cog_http_meta`` and then exercises this branch).
    # The eager / dask HTTP paths that ARE patched route through
    # ``_cog_http._read_cog_http`` and ``_backends/dask.py``'s
    # ``_HTTPSource`` construction, both of which still go through
    # ``_reader`` for the patchable names.
    from ._cog_http import _parse_cog_http_meta
    from ._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
    from ._geotags import extract_geo_info_with_overview_inheritance
    from ._header import parse_all_ifds, parse_header, select_overview_ifd
    from ._sources import _CloudSource, _coerce_path, _is_file_like, _is_fsspec_uri
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
        #
        # ``source_path=source`` opts the parser into external
        # ``.tif.ovr`` sidecar discovery. Without it,
        # ``open_geotiff(uri, chunks=..., overview_level=1)`` on a
        # GDAL external-overview file raised out-of-range or picked a
        # different overview than the eager read of the same URI.
        # ``return_sidecar=False`` (the default) makes the helper
        # close the sidecar buffer for us before returning -- the
        # metadata-only path here only needs ``geo_info`` and the
        # IFD's dimensions, both populated before the buffer is freed.
        _src = _CloudSource(source)
        try:
            _header, _ifd, geo_info, _ = _parse_cog_http_meta(
                _src, overview_level=overview_level,
                allow_rotated=allow_rotated,
                allow_invalid_nodata=allow_invalid_nodata,
                source_path=source)
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
        # binding it into the chunk closure.
        geo_info._ifd_photometric = _ifd.photometric
        geo_info._ifd_samples_per_pixel = _ifd.samples_per_pixel
        geo_info._ifd_compression = _ifd.compression
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
        # overviews. Local file paths only.
        #
        # A broken sidecar must not break the base read. The release
        # contract puts ``reader.local_file`` at the stable tier and
        # ``reader.sidecar_ovr`` at advanced; a stale or corrupt
        # ``.ovr`` written by an external tool falls back to base-only
        # behaviour with a warning. Mirrors the eager CPU path in
        # ``_reader._read_to_array`` and the dask metadata helper
        # ``_sidecar.discover_remote_sidecar``.
        from ._sidecar import (attach_sidecar_origin, find_sidecar, handle_sidecar_parse_failure,
                               load_sidecar)
        sidecar_origin: dict[int, tuple] = {}
        sidecar_path = find_sidecar(source)
        if sidecar_path is not None:
            try:
                sidecar = load_sidecar(sidecar_path)
            except CloudSizeLimitError:
                # Re-raised for symmetry with ``_reader._read_to_array``;
                # the byte budget is a caller-set contract. In practice
                # this branch is local-file-only (the cloud / HTTP cases
                # are handled in the earlier ``_parse_cog_http_meta`` /
                # ``_CloudSource`` branch above) so the exception cannot
                # fire from a local mmap today, but keeping the explicit
                # re-raise prevents the symmetry breaking if a future
                # patch routes a cloud-source path through here.
                raise
            except Exception as exc:
                # Shared policy: surface the parse error when the
                # caller asked for a level the base file alone cannot
                # serve; warn and fall back otherwise. See
                # ``_sidecar.handle_sidecar_parse_failure`` for the
                # rationale. Issue #2484.
                handle_sidecar_parse_failure(
                    exc, sidecar_path, overview_level,
                    base_ifd_count=len(ifds),
                )
                sidecar = None
            if sidecar is not None:
                # The origin mapping is consumed below for georef extraction
                # only -- strip/tile bytes are sliced by ``read_to_array`` on
                # the actual read. A sidecar IFD that carries its own
                # GeoKeyDirectory / ModelPixelScale / ModelTiepoint /
                # ModelTransformation needs the sidecar's byte order to
                # parse cleanly; without the mapping the helper falls back
                # to the base file's bytes (today's default, correct under
                # the usual GDAL convention).
                sidecar_origin = attach_sidecar_origin(
                    sidecar.ifds, sidecar.data, sidecar.header)
                ifds = ifds + sidecar.ifds
        ifd = select_overview_ifd(ifds, overview_level)
        # Inherit georef from the level-0 IFD when the overview itself
        # has no geokeys. Pass-through for level 0. The sidecar IFDs
        # typically lack geokeys so the inheritance pulls from the base
        # file's full-resolution IFD as GDAL does. When a sidecar IFD
        # does declare its own georef payload, ``georef_origin`` routes
        # the parse to the sidecar's bytes / byte order so the sidecar's
        # georef wins.
        georef_origin = (
            {iid: (od, oh.byte_order)
             for iid, (od, oh) in sidecar_origin.items()}
            if sidecar_origin else None
        )
        geo_info = extract_geo_info_with_overview_inheritance(
            ifd, ifds, data, header.byte_order,
            allow_rotated=allow_rotated,
            allow_invalid_nodata=allow_invalid_nodata,
            sidecar_origin=georef_origin)
        bps = resolve_bits_per_sample(ifd.bits_per_sample)
        file_dtype = tiff_dtype_to_numpy(bps, ifd.sample_format)
        _validate_predictor_sample_format(ifd.predictor, ifd.sample_format)
        n_bands = ifd.samples_per_pixel if ifd.samples_per_pixel > 1 else 0
        # Stash photometric + samples_per_pixel so the dask graph builder
        # can detect MinIsWhite and invert ``geo_info.nodata`` before
        # binding it into the chunk closure.
        geo_info._ifd_photometric = ifd.photometric
        geo_info._ifd_samples_per_pixel = ifd.samples_per_pixel
        # Stash compression so the dask graph builder can fire the
        # experimental / internal-only codec opt-in gate at graph build
        # rather than waiting for the per-chunk task to fail.
        geo_info._ifd_compression = ifd.compression
        return geo_info, ifd.height, ifd.width, file_dtype, n_bands
    finally:
        if close_data:
            data.close()
        from ._sidecar import close_sidecar
        close_sidecar(sidecar)


def _bbox_to_window(source, bbox, *, overview_level=None,
                    allow_rotated=False, allow_invalid_nodata=False):
    """Resolve a geographic ``bbox`` to a pixel ``window`` for the source.

    ``bbox`` is ``(x_min, y_min, x_max, y_max)`` in the source's CRS.
    The returned tuple is ``(row_start, col_start, row_stop, col_stop)``
    clamped to the file's extent and ready to forward as the existing
    ``window=`` kwarg through the backend dispatch.

    Uses ``_read_geo_info`` which already supports local files,
    BytesIO, HTTP, and fsspec URIs via header-only reads, so this is
    an O(1)-memory metadata pass rather than a full decode.

    Raises ``ValueError`` if ``bbox`` is malformed, the source is not
    georeferenced, or the transform is rotated. Rotated-affine files
    are rejected because ``_extent_to_window`` assumes an
    axis-aligned grid; the caller can pass ``allow_rotated=True`` to
    drop the rotation upstream and then re-call with ``bbox=``.
    """
    geo_info, height, width, _dtype, _nbands = _read_geo_info(
        source, overview_level=overview_level,
        allow_rotated=allow_rotated,
        allow_invalid_nodata=allow_invalid_nodata)
    return _geo_info_to_window(geo_info, height, width, bbox)


def _vrt_bbox_to_window(source, bbox):
    """Resolve a geographic ``bbox`` to a pixel ``window`` for a VRT.

    The TIFF-only :func:`_bbox_to_window` cannot read a ``.vrt`` source:
    its ``_read_geo_info`` call parses a TIFF header and chokes on the
    VRT's XML. This variant parses the VRT XML instead and reuses the
    same synthesised ``GeoInfo`` the VRT read path builds, so the
    bbox-to-window math (and the georef / rotated-affine rejection) lands
    on the same code as the TIFF path via :func:`_geo_info_to_window`.

    Only the VRT XML is read here (a header-only, O(1)-memory parse); no
    source tiles are decoded. The bbox is resolved purely from the VRT's
    GeoTransform, so CRS parseability does not matter at this stage.
    """
    import os as _os

    from ._backends.vrt import _vrt_to_synthetic_geo_info
    from ._vrt import _read_vrt_xml
    from ._vrt import parse_vrt as _parse_vrt
    xml_str = _read_vrt_xml(source)
    vrt_dir = _os.path.dirname(_os.path.abspath(source))
    parsed = _parse_vrt(xml_str, vrt_dir)
    geo_info = _vrt_to_synthetic_geo_info(parsed)
    return _geo_info_to_window(geo_info, parsed.height, parsed.width, bbox)


def _geo_info_to_window(geo_info, height, width, bbox):
    """Resolve a validated ``bbox`` against a ``GeoInfo`` to a pixel window.

    Shared by the TIFF (:func:`_bbox_to_window`) and VRT
    (:func:`_vrt_bbox_to_window`) bbox resolvers so both paths run the
    same bbox validation, rotated-affine / no-georef rejection, and
    extent clamping. ``geo_info`` carries the axis-aligned transform;
    ``height`` / ``width`` are the source's pixel dimensions.
    """
    if (not isinstance(bbox, (tuple, list)) or len(bbox) != 4):
        raise ValueError(
            "open_geotiff: bbox must be a 4-tuple "
            "(x_min, y_min, x_max, y_max), "
            f"got {bbox!r}.")
    x_min, y_min, x_max, y_max = bbox
    # ``NaN >= NaN`` is False, so NaN coordinates would slip past the
    # ordering check below and only surface later as an unhelpful
    # integer-cast error inside ``_extent_to_window``. Reject upfront.
    if not all(np.isfinite(v) for v in (x_min, y_min, x_max, y_max)):
        raise ValueError(
            f"open_geotiff: bbox must contain finite coordinates, "
            f"got bbox={bbox!r}.")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"open_geotiff: bbox has non-positive size "
            f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, "
            f"y_max={y_max}). Expected x_min < x_max and y_min < y_max.")

    # ``allow_rotated=True`` clears a rotated affine and stashes the
    # original 6-tuple on ``transform.rotated_affine`` while setting
    # ``has_georef=False`` (the dropped-rotation marker). Check
    # ``rotated_affine`` first so this case gets the more specific
    # message that names the recovery path; the plain-no-georef
    # error then covers everything else.
    if (geo_info.transform is not None
            and geo_info.transform.rotated_affine is not None):
        raise ValueError(
            "open_geotiff: bbox= requires an axis-aligned transform, "
            "but this file has a rotated affine. The rotation cannot "
            "be expressed as a 4-tuple bbox in the file's CRS; pass "
            "window= for pixel-space windowing instead.")
    if not geo_info.has_georef:
        raise ValueError(
            "open_geotiff: bbox= requires a georeferenced source, "
            "but this source has no georeferencing (no GeoTIFF tags or "
            "VRT GeoTransform). Pass window= instead for pixel-space "
            "windowing.")

    pixel_window = _extent_to_window(
        geo_info.transform, height, width,
        y_min, y_max, x_min, x_max)
    row_start, col_start, row_stop, col_stop = pixel_window
    if row_start >= row_stop or col_start >= col_stop:
        raise ValueError(
            f"open_geotiff: bbox={bbox!r} does not overlap the file's "
            f"extent (height={height}, width={width}). Resolved pixel "
            f"window={pixel_window} has non-positive size.")
    return pixel_window


def open_geotiff(source: str | BinaryIO, *,
                 dtype: str | np.dtype | None = None,
                 window: tuple | None = None,
                 bbox: tuple | None = None,
                 overview_level: int | None = None,
                 band: int | None = None,
                 default_name: str | None = None,
                 name: str | None = _NAME_DEPRECATED_SENTINEL,  # type: ignore[assignment]
                 chunks: int | tuple | None = None,
                 gpu: bool = False,
                 max_pixels: int | None = None,
                 max_cloud_bytes: int | None = (
                     _MAX_CLOUD_BYTES_SENTINEL),  # type: ignore[assignment]
                 on_gpu_failure: str = _ON_GPU_FAILURE_SENTINEL,
                 missing_sources: str = _MISSING_SOURCES_SENTINEL,
                 allow_rotated: bool = False,
                 allow_unparseable_crs: bool = False,
                 allow_invalid_nodata: bool = False,
                 stable_only: bool = False,
                 allow_experimental_codecs: bool = False,
                 allow_internal_only_jpeg: bool = False,
                 band_nodata: str | None = None,
                 masked: bool = False,
                 mask_nodata: bool = _MASK_NODATA_DEPRECATED_SENTINEL,  # type: ignore[assignment]
                 unpack: bool = False,
                 mask_and_scale: bool = (
                     _MASK_AND_SCALE_DEPRECATED_SENTINEL),  # type: ignore[assignment]
                 parse_coordinates: bool = True,
                 lock: object | None = None,
                 cache: bool = True,
                 ) -> xr.DataArray:
    """Read a GeoTIFF, COG, or VRT file into an xarray.DataArray.

    Release-contract tier (see
    ``docs/source/reference/release_gate_geotiff.rst`` for the audited
    matrix and ``docs/source/reference/geotiff_release_contract.rst``
    for the prose contract once that page lands):

    * [stable] Local-file reads on axis-aligned grids with an EPSG CRS
      in ``attrs['crs']``; Tier 1 codecs (``none`` / ``deflate`` /
      ``lzw`` / ``packbits`` / ``zstd``); windowed reads via ``window=``.
    * [advanced] Cloud / fsspec URIs, HTTP range reads, ``.vrt``
      mosaics, external ``.tif.ovr`` sidecars, ``allow_rotated=True``,
      ``allow_unparseable_crs=True``, ``overview_level=`` selection.
      These paths work and are tested, but each carries a specific
      failure mode named on the parameter doc.
    * [experimental] ``gpu=True``; LERC / JPEG2000 / J2K / LZ4 decode.
      No cross-backend numerical parity claim. JPEG-in-TIFF on the
      read side decodes best-effort with no parity claim against
      libtiff / GDAL / rasterio; the write side is ``[internal-only]``
      (the encoder omits the required JPEGTables tag, so round-trips
      hold only for files this library itself wrote).
    * Out of scope for this release (allowed to raise): full GDAL VRT
      parity, warped / reprojection VRTs, rotated/sheared write
      support.

    See :data:`xrspatial.geotiff.SUPPORTED_FEATURES` for the full tier
    map. Per-parameter tier markers below describe the
    tier the parameter itself carries; a parameter's effective tier
    is bounded by the function-level surface above (e.g. ``[stable]``
    ``masked`` is still only stable when combined with a
    ``[stable]`` source, codec, and options).

    The read/masking parameters mostly match rioxarray's
    ``open_rasterio`` (``masked``, ``default_name``, ``parse_coordinates``,
    ``lock``, ``cache``) so callers can move between the two with minimal
    edits. ``masked`` defaults to ``False`` (no sentinel-to-NaN promotion),
    matching ``open_rasterio``. The scale/offset option is named ``unpack``
    here; rioxarray's ``mask_and_scale`` is kept as a deprecated alias.

    Automatically dispatches to the best backend:
    - ``gpu=True``: GPU-accelerated read via nvCOMP (returns CuPy)
    - ``chunks=N``: Dask lazy read via windowed chunks
    - ``gpu=True, chunks=N``: Dask+CuPy for out-of-core GPU pipelines
    - Default: NumPy eager read

    VRT files are auto-detected by extension. The supported VRT subset
    is narrow on purpose. See the
    "VRT support matrix" section in ``docs/source/reference/geotiff.rst``
    and the audited matrix in
    ``docs/source/reference/release_gate_geotiff.rst`` for the
    canonical contract. In short:

    * Supported: simple GDAL VRT mosaics over GeoTIFF sources;
      compatible CRS, transform orientation, pixel size, dtype, and
      band count across sources; clean windowed reads; lazy / dask
      reads over the same subset; explicit nodata with mixed-band
      rejection by default; ``missing_sources='raise'`` as the
      default.
    * Non-goals (allowed to raise): warped / reprojection VRTs,
      arbitrary resampling beyond the tested subset, mixed CRS /
      resolution / dtype / band metadata without an opt-in, nested
      VRTs, complex source / mask band / alpha band structures, full
      GDAL VRT parity.

    Parameters
    ----------
    source : str or binary file-like
        [stable for local file paths; advanced for HTTP/fsspec URIs,
        ``.vrt`` paths, and in-memory file-like buffers (the file-like
        path is restricted to the eager numpy reader -- dask, GPU,
        VRT, and remote-URL paths require a string)] File path, HTTP
        URL, cloud URI (s3://, gs://, az://), or a binary file-like
        object (e.g. ``io.BytesIO``) with read+seek.
    dtype : str, numpy.dtype, or None
        [stable] Cast the result to this dtype after reading. None
        keeps the file's native dtype. Float-to-int casts raise
        ValueError to prevent accidental data loss.
    window : tuple or None
        [stable] ``(row_start, col_start, row_stop, col_stop)`` for
        windowed reading. Mutually exclusive with ``bbox=``.
    bbox : tuple or None
        [stable] ``(x_min, y_min, x_max, y_max)`` in the file's CRS.
        Resolved to a pixel ``window=`` via a header-only metadata read
        and clamped to the file's extent. Requires the source to be
        georeferenced with an axis-aligned transform; rotated affines
        require ``allow_rotated=True`` to clear the rotation first.
        Mutually exclusive with ``window=``.
    overview_level : int or None
        [advanced] Overview level (0 = full resolution). Must be a
        non-negative int or ``None``; passing ``bool`` or any other
        type raises ``TypeError``. External ``.tif.ovr`` sidecars are
        also [advanced] and are tested but not load-bearing for
        release-gate parity.
    band : int or None
        [stable] Band index (0-based). None returns all bands.
    default_name : str or None
        [stable] Name for the DataArray. None derives it from the source
        file name. Matches rioxarray's ``open_rasterio`` parameter.
    name : str or None
        [deprecated] Deprecated alias of ``default_name``; emits a
        ``DeprecationWarning``. Passing both ``default_name`` and ``name``
        raises ``TypeError``.
    chunks : int, tuple, or None
        [stable] Chunk size for Dask lazy reading. Dask reads are
        gated against the eager reader by the cross-backend parity
        suite for the Tier 1 codec set.
    gpu : bool
        [experimental] Use GPU-accelerated decompression. Requires
        cupy + numba CUDA plus optional nvCOMP / nvJPEG / nvJPEG2K
        libraries for codec-specific acceleration. The reader falls
        back to CPU when those libraries are unavailable unless
        ``on_gpu_failure='strict'`` is also set. No cross-backend
        numerical parity claim outside the Tier 1 codec set.
    max_pixels : int or None
        [stable] Maximum allowed pixel count per materialised buffer.
        Without ``chunks=`` the cap bounds the full windowed region
        (width * height * samples); with ``chunks=`` the cap bounds
        each chunk's decode buffer instead, so a small ``max_pixels``
        no longer rejects a large lazy raster up front. None uses the
        default (~1 billion). Raise it to read legitimately large
        files. Exceeding the cap raises
        :class:`~xrspatial.geotiff.PixelSafetyLimitError` (a
        ``ValueError`` subclass).
    max_cloud_bytes : int or None, optional
        [advanced] fsspec cloud reads can run up cost on large objects;
        the budget defends against accidental large downloads but the
        eager path still pulls the full object once the budget allows.
        Byte ceiling for eager reads from fsspec sources (``s3://``,
        ``gs://``, ``az://``, ``abfs://``, ``memory://``, ...). The
        compressed object size is checked against this budget before
        any bytes are downloaded; a breach raises
        :class:`~xrspatial.geotiff.CloudSizeLimitError` (a
        ``ValueError`` subclass). Default is 256 MiB, overridable via
        the ``XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES`` env var. Pass
        ``None`` to skip the check entirely. The HTTP path already
        reads only what it needs via range requests and is not subject
        to this limit. Has no effect on local file or file-like
        sources. Passing this kwarg with ``gpu=True``, ``chunks=...``,
        or a ``.vrt`` source raises ``ValueError`` because those
        backends do not apply the cloud-byte budget.
    on_gpu_failure : {'auto', 'strict'}, optional
        [experimental] Forwarded to ``_read_geotiff_gpu`` when
        ``gpu=True``. Controls whether GPU decode failures fall back
        to CPU (``'auto'``, default) or re-raise the original exception
        (``'strict'``). Passing this kwarg with ``gpu=False`` raises
        ``ValueError`` because the policy only applies to the GPU
        pipeline. See ``_read_geotiff_gpu`` for the full description.
    missing_sources : {'raise', 'warn'}, optional
        [advanced] VRT mosaics can return partial output under
        ``missing_sources='warn'`` when a backing source is unreadable;
        the ``attrs['vrt_holes']`` entry records which sources were
        skipped so downstream code can detect the partial mosaic.
        Forwarded to ``_read_vrt`` when the source is a ``.vrt`` file.
        When the caller does not pass this kwarg, the public
        ``_read_vrt`` default applies (``'raise'``).
        ``'raise'`` fails immediately on an unreadable backing source.
        ``'warn'`` is the opt-in lenient mode: emit
        ``GeoTIFFFallbackWarning``, record ``attrs['vrt_holes']``, and
        return a partial mosaic. Passing this kwarg with a non-VRT
        source raises ``ValueError`` because the policy only applies to
        the VRT pipeline. See ``_read_vrt`` for the full description.
    band_nodata : {'first', None}, optional
        [advanced] VRT-only. Opt-out for the fail-closed check that
        rejects VRT sources whose bands declare disagreeing per-band
        nodata sentinels. When ``None`` (the default), a VRT
        that mosaics bands with different sentinels raises
        ``MixedBandMetadataError``; flattening to one value would let
        one band's valid pixels collide with another band's sentinel.
        Pass ``band_nodata='first'`` to keep the legacy behaviour of
        using band 0's sentinel for the whole mosaic. Passing this
        kwarg with a non-VRT source raises ``ValueError`` because the
        policy only applies to the VRT pipeline.
    masked : bool, default False
        [stable] If True, replace the nodata sentinel with ``NaN``;
        integer rasters get promoted to ``float64`` first so NaN can be
        represented. If False (the default), skip the sentinel-to-NaN
        step and keep the source dtype. ``attrs['nodata']`` still
        carries the raw sentinel either way, so downstream code can mask
        explicitly. The default matches rioxarray's ``open_rasterio``
        (``masked=False``); note that earlier xrspatial releases masked
        by default (``mask_nodata=True``), so a bare ``open_geotiff(path)``
        no longer promotes the sentinel to NaN. Pass ``masked=True`` and
        ``dtype=<integer>`` together on a source with a maskable sentinel
        and the read raises ``ValueError``, because the unconditional
        float64 promotion (issue #2990) makes the integer cast lossy
        whether or not a sentinel pixel is present.
    mask_nodata : bool
        [deprecated] Deprecated alias of ``masked``; emits a
        ``DeprecationWarning``. Passing both ``masked`` and ``mask_nodata``
        raises ``TypeError``. Note the default also changed from
        ``mask_nodata=True`` to ``masked=False``.
    unpack : bool, default False
        [experimental] If True, read the source's GDAL ``SCALE`` / ``OFFSET``
        metadata and return ``data * scale + offset``, masking the nodata
        sentinel to NaN as well. This unpacks CF-packed data (integers
        stored with a scale / offset that recover floats) and is the
        inverse of the writer's ``pack`` option.
        The applied values are recorded on ``attrs['scale_factor']`` /
        ``attrs['add_offset']``. A source without scale / offset metadata
        skips the scaling step, but the sentinel-to-NaN mask still runs:
        a declared nodata sentinel is replaced with NaN and an integer
        source is promoted to ``float64`` (matching rioxarray's
        ``mask_and_scale``). Only a source with neither scale / offset
        metadata nor a nodata sentinel reads unchanged.
        A dataset-level scale / offset, or per-band values that
        agree across bands, applies to the whole array. A source whose
        per-band scale / offset differ raises ``MixedBandMetadataError``
        unless ``band=`` selects a single band, in which case that band's
        scale / offset is applied. Supported on the CPU eager, dask, GPU
        (``gpu=True``), and dask+GPU (``gpu=True, chunks=``) paths;
        combining it with a ``.vrt`` source raises ``ValueError``.
        On the dask+GPU path, ``unpack=True`` reads through the
        CPU-decode-then-upload route rather than the direct disk->GPU GDS
        fast path (the GDS path has no scale step), so a local tiled COG
        that would otherwise stream straight to the device decodes on CPU
        first.
        Round-trip caveat: the source's ``SCALE`` / ``OFFSET`` tags stay on
        ``attrs['gdal_metadata']`` / ``attrs['gdal_metadata_xml']`` after the
        read, so writing an ``unpack=True`` result back out with
        ``to_geotiff`` re-embeds them, and reading that file again with
        ``unpack=True`` applies the scale a second time. Drop those
        tags (and ``attrs['scale_factor']`` / ``attrs['add_offset']``) before
        writing if you need a clean round-trip.
    mask_and_scale : bool
        [deprecated] Deprecated alias of ``unpack``; emits a
        ``DeprecationWarning``. Named after rioxarray's ``open_rasterio``.
        Passing both ``unpack`` and ``mask_and_scale`` raises ``TypeError``.
    parse_coordinates : bool, default True
        [stable] If True (the default), build ``x`` / ``y`` coordinate
        arrays from the transform. If False, skip them and return a
        DataArray with only dimensions (matching rioxarray's
        ``open_rasterio``); ``attrs['transform']`` and ``attrs['crs']``
        still carry the georeferencing, and the ``band`` coord is kept.
        Supported on the CPU eager and dask paths; combining
        ``parse_coordinates=False`` with ``gpu=True`` or a ``.vrt`` source
        raises ``ValueError``.
    lock : object or None
        [advanced] Accepted for ``open_rasterio`` signature compatibility
        but has no effect: xrspatial's reader re-opens the source per
        window, so there is no shared GDAL handle to lock. Passing a
        non-default value emits a ``GeoTIFFFallbackWarning``.
    cache : bool
        [advanced] Accepted for ``open_rasterio`` signature compatibility
        but has no effect: xrspatial has no caching backend to toggle.
        Passing a non-default value emits a ``GeoTIFFFallbackWarning``.
    allow_rotated : bool, default False
        [advanced] Read-only opt-in. ``to_geotiff`` does not currently
        emit ``rotated_affine``; it rejects DataArrays that carry the
        attr (``ValueError`` naming the attr) unless the caller passes
        ``drop_rotation=True`` to accept the loss explicitly.
        Read-side opt-in for rotated / sheared ``ModelTransformationTag``
        files. By default the reader raises ``RotatedTransformError``
        (a ``GeoTIFFAmbiguousMetadataError`` / ``ValueError`` subclass)
        because the rest of xrspatial assumes an axis-aligned grid.
        ``allow_rotated=True`` reads the pixel grid without the
        geospatial assumption: the result has integer pixel coords on
        ``x`` / ``y`` and both ``attrs['crs']`` and ``attrs['crs_wkt']``
        are dropped. The CRS attrs are dropped together with the
        transform because keeping them while the axis-aligned transform
        is gone misleads downstream code that gates on
        ``"crs" in da.attrs`` to mean the array is spatially usable.
        The rotated 6-tuple itself is surfaced on
        ``attrs['rotated_affine']`` as ``(a, b, c, d, e, f)`` (rasterio
        ``Affine`` ordering) so consumers that know how to handle
        rotated rasters can recover the mapping. The
        contract is read-only -- writes must either reproject onto an
        axis-aligned grid first, or pass ``drop_rotation=True`` to
        ``to_geotiff`` / ``_write_geotiff_gpu`` to accept the loss; the
        ``ModelTransformationTag`` emit path is tracked separately.
    allow_unparseable_crs : bool, default False
        [advanced] Read-side opt-in for CRS strings that pyproj cannot
        resolve and that do not parse as WKT. When ``False`` (the
        default), an unrecognised CRS payload raises
        ``UnparseableCRSError`` instead of landing in ``attrs['crs_wkt']``
        verbatim. Set to ``True`` to keep the permissive
        behaviour where the citation field passes through unchanged.
        Matches the same kwarg on ``to_geotiff`` / ``_write_geotiff_gpu``
        so a value the reader accepted can survive a round-trip.
    allow_invalid_nodata : bool, default False
        [advanced] Read-side opt-in for integer-dtype sources whose
        ``GDAL_NODATA`` tag is non-finite (``"NaN"``, ``"Inf"``,
        ``"-Inf"``) or fractional (e.g. ``"3.5"`` on a ``uint16``
        file). The legacy reader parsed the value into
        ``attrs['nodata']`` and silently skipped the masking step, so
        callers had no way to tell a silently-ignored sentinel from a
        missing one. When ``False`` (the default), the read raises
        ``InvalidIntegerNodataError``. Set to ``True`` to keep the
        pre-rejection no-op behaviour for files known to carry such
        sentinels (e.g. external tooling that writes ``"nan"`` on
        integer outputs).
    stable_only : bool, default False
        [advanced] Read-side opt-in that restricts the read to the
        stable-tier local-file path. When ``True``, advanced-tier
        sources are rejected: a ``.vrt`` source raises
        :class:`VRTStableSourcesOnlyError` because ``reader.vrt`` and
        the VRT child-source pipeline sit at the ``advanced`` /
        ``experimental`` tiers in
        :data:`xrspatial.geotiff.SUPPORTED_FEATURES`, and HTTP /
        fsspec sources (``http(s)://``, ``s3://``, etc.) are rejected
        too because ``reader.http`` and ``reader.fsspec`` are also
        ``advanced``. Only a local-file source riding the stable
        ``reader.local_file`` path and the per-source codec gate is
        accepted. The rejection names the offending source and the
        ``allow_experimental_codecs`` opt-in so the caller can unlock
        the broader tier set explicitly when needed. See
        ``docs/source/reference/release_gate_geotiff.rst``. The VRT
        rejection is enforced today; the HTTP / fsspec rejection is the
        documented contract being rolled out and may not yet fire on
        every read path (tracked in issue #2820).
    allow_experimental_codecs : bool, default False
        Read-side opt-in for sources compressed with the Tier 3
        experimental codecs (``lerc``, ``jpeg2000`` / ``j2k``, ``lz4``).
        Default ``False`` rejects the read with ``ValueError`` naming
        the flag; cross-backend numerical parity is not claimed and
        reader support across GDAL versions is uneven. Matches the
        same kwarg on the writers so a round-trip through a Tier 3
        codec stays opt-in on both sides. See SUPPORTED_FEATURES tier
        ``'experimental'``.
    allow_internal_only_jpeg : bool, default False
        Read-side opt-in for JPEG-in-TIFF sources. The encoder writes
        self-contained JFIF tiles without the TIFF JPEGTables tag
        (347), so the read path is not interoperable with libtiff /
        GDAL / rasterio. ``allow_experimental_codecs=True`` does NOT
        cover this codec; the dedicated flag is its only gate. See
        SUPPORTED_FEATURES tier ``'internal_only'`` for ``codec.jpeg``.

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

    With ``masked=True``, integer rasters with a nodata sentinel are
    promoted to ``float64`` with NaN replacing the sentinel so downstream
    NaN-aware code works uniformly. The default ``masked=False`` keeps the
    source dtype and leaves the raw sentinel in the data;
    ``attrs['nodata']`` still carries it either way. With ``masked=True``,
    passing ``dtype=<integer>`` as well is not enough to keep an integer
    dtype: the sentinel-to-NaN promotion runs first and the subsequent
    integer cast then raises ``ValueError`` (float-to-int is lossy in a
    way users rarely intend). The promotion runs whenever the sentinel is
    maskable (finite, integer, in-range), whether or not any pixel matches
    it, so the eager and dask paths return the same float64 dtype for the
    same input (issue #2990). A sentinel that cannot match (out-of-range,
    non-finite, or fractional) leaves the source dtype, so
    ``dtype=<integer>`` works in that case.

    Examples
    --------
    Safe VRT usage. Write a ``.vrt`` mosaic with ``to_geotiff`` and read
    it back with the fail-closed defaults:

    >>> from xrspatial.geotiff import open_geotiff, to_geotiff
    >>> to_geotiff(data, 'mosaic.vrt')  # doctest: +SKIP
    >>> da = open_geotiff('mosaic.vrt')  # doctest: +SKIP

    Intentionally raises. A VRT whose source tiles disagree on their
    per-band nodata sentinels is rejected by the default
    ``band_nodata=None``:

    >>> from xrspatial.geotiff import MixedBandMetadataError
    >>> try:  # doctest: +SKIP
    ...     open_geotiff('mixed_nodata.vrt')
    ... except MixedBandMetadataError:
    ...     pass  # pass band_nodata='first' to opt back into the
    ...           # legacy flatten-to-band-0 semantics, or fix the
    ...           # source tiles.
    """
    from ._reader import _coerce_path

    source = _coerce_path(source)

    # Resolve the rioxarray-compatible renames. ``masked`` / ``default_name``
    # are the canonical names; ``mask_nodata`` / ``name`` are deprecated
    # aliases kept for back-compat. Mirrors the sentinel-based deprecation in
    # ``read_geotiff_gpu`` (gpu -> on_gpu_failure): passing both the old and
    # new name is ambiguous and raises, passing the old name alone warns.
    if mask_nodata is not _MASK_NODATA_DEPRECATED_SENTINEL:
        # ``masked`` carries a real default of False, so an explicit
        # ``masked=False`` cannot be told apart from the default here; that
        # one combination (``masked=False`` + ``mask_nodata=True``) does not
        # raise and resolves to the ``mask_nodata`` value. This matches the
        # documented stance on ``read_geotiff_gpu``'s gpu/on_gpu_failure pair.
        if masked is not False:
            raise TypeError(
                "open_geotiff: pass either 'masked' or the deprecated "
                "'mask_nodata' alias, not both.")
        warnings.warn(
            "open_geotiff(..., mask_nodata=...) is deprecated; use "
            "masked=... instead. Note the default also changed from "
            "mask_nodata=True to masked=False to match rioxarray's "
            "open_rasterio: a bare open_geotiff(path) no longer promotes "
            "the nodata sentinel to NaN.",
            DeprecationWarning, stacklevel=2)
        masked = mask_nodata
    if name is not _NAME_DEPRECATED_SENTINEL:
        if default_name is not None:
            raise TypeError(
                "open_geotiff: pass either 'default_name' or the deprecated "
                "'name' alias, not both.")
        warnings.warn(
            "open_geotiff(..., name=...) is deprecated; use default_name=... "
            "instead to match rioxarray's open_rasterio.",
            DeprecationWarning, stacklevel=2)
        default_name = name
    if mask_and_scale is not _MASK_AND_SCALE_DEPRECATED_SENTINEL:
        # ``unpack`` is the canonical name; ``mask_and_scale`` is the
        # deprecated rioxarray-compatible alias. ``unpack`` carries a real
        # default of False, so an explicit ``unpack=False`` cannot be told
        # apart from the default; that one combination resolves to the
        # ``mask_and_scale`` value, matching the ``masked`` / ``mask_nodata``
        # stance above.
        if unpack is not False:
            raise TypeError(
                "open_geotiff: pass either 'unpack' or the deprecated "
                "'mask_and_scale' alias, not both.")
        warnings.warn(
            "open_geotiff(..., mask_and_scale=...) is deprecated; use "
            "unpack=... instead. The behaviour is unchanged: it applies the "
            "source's GDAL SCALE / OFFSET and masks the nodata sentinel.",
            DeprecationWarning, stacklevel=2)
        unpack = mask_and_scale

    # ``lock`` / ``cache`` are accepted for open_rasterio signature
    # compatibility. xrspatial's dask reader re-opens the source per window,
    # so there is no shared GDAL handle to lock and no caching backend to
    # toggle. Warn rather than silently ignore so a porting caller is not
    # surprised by a no-op.
    if lock is not None or cache is not True:
        warnings.warn(
            "open_geotiff: 'lock' and 'cache' are accepted for rioxarray "
            "open_rasterio compatibility but have no effect; xrspatial's "
            "reader re-opens the source per window, so there is no shared "
            "GDAL handle to lock and no caching layer to toggle.",
            GeoTIFFFallbackWarning, stacklevel=2)

    # ``unpack`` and ``parse_coordinates=False`` build their DataArrays
    # through code the cross-backend parity suite does not cover on every
    # backend, so refuse the unsupported combinations up front rather than
    # silently ignoring the kwarg -- the same per-backend rejection contract
    # the dispatcher already applies to on_gpu_failure / missing_sources /
    # max_cloud_bytes. ``unpack`` now runs on the GPU and dask+GPU paths
    # (issue #3071), so it only rejects ``.vrt`` sources. ``parse_coordinates``
    # is still CPU/dask-only, so it rejects both gpu=True and ``.vrt``.
    _is_vrt_source = (
        isinstance(source, str) and source.lower().endswith('.vrt'))
    if not parse_coordinates:
        if gpu:
            raise ValueError(
                "parse_coordinates=False is not supported with gpu=True; it "
                "is implemented on the CPU eager and dask paths. Drop "
                "gpu=True or the kwarg.")
        if _is_vrt_source:
            raise ValueError(
                "parse_coordinates=False is not supported for .vrt sources; "
                "it is implemented on the CPU eager and dask paths over .tif "
                "sources. Drop the kwarg.")
    if unpack and _is_vrt_source:
        raise ValueError(
            "unpack=True is not supported for .vrt sources; it is "
            "implemented on the CPU, dask, GPU, and dask+GPU paths over .tif "
            "sources. Drop the kwarg.")

    # All dispatcher-level kwarg rejection lives in
    # ``_validate_dispatch_kwargs`` so the three direct backends
    # (``_read_geotiff_dask``, ``_read_geotiff_gpu``, ``_read_vrt``)
    # surface the same errors when called directly. The single call
    # runs ``_validate_overview_level_arg``, the ``on_gpu_failure``
    # GPU-only guard, the ``missing_sources`` VRT-only guard, the
    # ``band_nodata`` VRT-only guard, the ``max_cloud_bytes`` non-VRT
    # non-GPU non-dask guard, and the file-like source restrictions
    # for gpu/chunks.
    from ._validation import _validate_dispatch_kwargs
    _validate_dispatch_kwargs(
        source=source,
        gpu=gpu,
        chunks=chunks,
        overview_level=overview_level,
        on_gpu_failure=on_gpu_failure,
        missing_sources=missing_sources,
        band_nodata=band_nodata,
        max_cloud_bytes=max_cloud_bytes,
    )

    missing_sources_passed = (
        missing_sources is not _MISSING_SOURCES_SENTINEL)
    # ``_is_vrt_source`` was resolved above for the unpack /
    # parse_coordinates gate.

    # Gate ``stable_only=True`` BEFORE resolving ``bbox=``. The bbox
    # resolver reads source geo metadata first (the TIFF path reads a
    # header, which is a range GET for an HTTP / fsspec source; the VRT
    # path parses the VRT XML), so the stable-only rejection has to run
    # ahead of it or a stable-only request would trigger remote I/O or a
    # VRT parse before it is refused. ``_validate_stable_only_remote``
    # documents exactly this ordering contract ("before any range GET or
    # decode work"). ``_validate_stable_only_vrt`` is the matching gate
    # for ``.vrt`` sources; ``_read_vrt`` runs it again on the direct-call
    # path, so this is defence in depth, not the only gate. Each helper is
    # a no-op for the wrong source type, but branching keeps the intent
    # obvious. Running ahead of the bbox block also puts this gate ahead
    # of the ``window=``/``bbox=`` mutual-exclusion check below, so a
    # stable-only remote/VRT source is refused on the tier gate before
    # that kwarg conflict is reported -- the tier rejection wins, which
    # matches refusing the unsupported source before validating kwargs
    # that would not be honoured anyway.
    from ._validation import _validate_stable_only_remote, _validate_stable_only_vrt
    if _is_vrt_source:
        _validate_stable_only_vrt(
            source,
            stable_only=stable_only,
            allow_experimental_codecs=allow_experimental_codecs,
        )
    else:
        _validate_stable_only_remote(
            source,
            stable_only=stable_only,
            allow_experimental_codecs=allow_experimental_codecs,
        )

    # Resolve ``bbox=`` to a pixel ``window=`` via a header-only
    # metadata read. Done at the dispatcher so every backend
    # (eager / dask / GPU / VRT) sees a uniform pixel window without
    # learning a new kwarg. VRT sources route through the VRT XML parser
    # (``_vrt_bbox_to_window``); the TIFF resolver's ``_read_geo_info``
    # would otherwise try to read the VRT's XML as a TIFF header and
    # fail. Both resolvers share the same bbox validation and extent math
    # via ``_geo_info_to_window``.
    if bbox is not None:
        if window is not None:
            raise ValueError(
                "open_geotiff: window= and bbox= are mutually exclusive; "
                "pass one or the other.")
        if _is_vrt_source:
            window = _vrt_bbox_to_window(source, bbox)
        else:
            window = _bbox_to_window(
                source, bbox, overview_level=overview_level,
                allow_rotated=allow_rotated,
                allow_invalid_nodata=allow_invalid_nodata)

    # VRT files (string paths only -- VRT XML references other files on disk)
    if _is_vrt_source:
        # ``_read_vrt`` does not accept ``overview_level`` (the VRT XML
        # references its own source files; overview selection would need
        # to apply to each one). Silently dropping the kwarg is the same
        # class of bug the dask and GPU dispatchers already guard against,
        # so refuse the combination up front rather than handing the
        # caller a full-resolution mosaic with no warning.
        # ``overview_level=0`` is documented as "full resolution" (the
        # default), so treat it as a no-op the same as ``None`` rather
        # than rejecting a kwarg value the caller could have omitted.
        # Mirrored at ``_backends/vrt.py`` (the direct-call entry point)
        # so callers see the same rejection through both paths. The
        # value-level rejection stays here rather than in
        # ``_validate_dispatch_kwargs`` so the helper does not need a
        # special-case for ``overview_level=0`` vs ``overview_level=N``.
        if overview_level not in (None, 0):
            raise ValueError(
                "overview_level is not supported for VRT sources. "
                "VRT references its own source files; pass overview_level "
                "to open_geotiff on a .tif source, or drop the kwarg.")
        # ``on_gpu_failure`` only routes through ``_read_geotiff_gpu``.
        # ``_read_vrt`` has no analogous failure policy, so any value the
        # caller supplied alongside a VRT source would be silently lost.
        # The ``gpu=False`` branch is already rejected above; this catches
        # the ``gpu=True, source.endswith('.vrt')`` case the earlier check
        # lets through.
        if on_gpu_failure is not _ON_GPU_FAILURE_SENTINEL:
            raise ValueError(
                "on_gpu_failure is not supported for VRT sources. "
                "VRT reads do not go through the GPU decoder pipeline; "
                "drop the kwarg or call _read_geotiff_gpu directly on a "
                ".tif source.")
        vrt_kwargs = {}
        if missing_sources_passed:
            vrt_kwargs['missing_sources'] = missing_sources
        return _read_vrt(source, dtype=dtype, window=window, band=band,
                         name=default_name, chunks=chunks, gpu=gpu,
                         max_pixels=max_pixels,
                         allow_rotated=allow_rotated,
                         allow_unparseable_crs=allow_unparseable_crs,
                         allow_invalid_nodata=allow_invalid_nodata,
                         stable_only=stable_only,
                         allow_experimental_codecs=allow_experimental_codecs,
                         allow_internal_only_jpeg=allow_internal_only_jpeg,
                         band_nodata=band_nodata,
                         mask_nodata=masked,
                         **vrt_kwargs)

    # File-like buffer rejections for ``gpu=True`` / ``chunks=...`` already
    # fired inside ``_validate_dispatch_kwargs`` above; the non-VRT branches
    # below run with a string source or an eager file-like. The remote /
    # VRT ``stable_only=True`` gate ran ahead of bbox resolution above.

    # GPU path
    if gpu:
        gpu_kwargs = {}
        if on_gpu_failure is not _ON_GPU_FAILURE_SENTINEL:
            gpu_kwargs['on_gpu_failure'] = on_gpu_failure
        return _read_geotiff_gpu(source, dtype=dtype,
                                 overview_level=overview_level,
                                 window=window, band=band,
                                 name=default_name, chunks=chunks,
                                 max_pixels=max_pixels,
                                 allow_rotated=allow_rotated,
                                 allow_unparseable_crs=allow_unparseable_crs,
                                 allow_invalid_nodata=allow_invalid_nodata,
                                 stable_only=stable_only,
                                 allow_experimental_codecs=(
                                    allow_experimental_codecs),
                                 allow_internal_only_jpeg=(
                                    allow_internal_only_jpeg),
                                 mask_nodata=masked,
                                 mask_and_scale=unpack,
                                 **gpu_kwargs)

    # Dask path (CPU)
    if chunks is not None:
        return _read_geotiff_dask(source, dtype=dtype, chunks=chunks,
                                  overview_level=overview_level,
                                  window=window, band=band,
                                  max_pixels=max_pixels, name=default_name,
                                  allow_rotated=allow_rotated,
                                  allow_unparseable_crs=allow_unparseable_crs,
                                  allow_invalid_nodata=allow_invalid_nodata,
                                  stable_only=stable_only,
                                  allow_experimental_codecs=(
                                     allow_experimental_codecs),
                                  allow_internal_only_jpeg=(
                                     allow_internal_only_jpeg),
                                  mask_nodata=masked,
                                  mask_and_scale=unpack,
                                  parse_coordinates=parse_coordinates)

    kwargs = {}
    if max_pixels is not None:
        kwargs['max_pixels'] = max_pixels
    if max_cloud_bytes is not _MAX_CLOUD_BYTES_SENTINEL:
        kwargs['max_cloud_bytes'] = max_cloud_bytes

    # ``read_to_array`` validates ``window`` against the selected IFD's
    # extent and raises ``ValueError`` for out-of-bounds windows with
    # the same message format as the dask path's pre-flight validator
    # in :func:`_read_geotiff_dask`. That keeps the two backends in sync
    # on the contract without forcing a second metadata parse here.
    arr, geo_info = _read_to_array(
        source, window=window,
        overview_level=overview_level, band=band,
        allow_rotated=allow_rotated,
        allow_invalid_nodata=allow_invalid_nodata,
        allow_experimental_codecs=allow_experimental_codecs,
        allow_internal_only_jpeg=allow_internal_only_jpeg,
        **kwargs,
    )

    if default_name is None:
        # Derive from source path. File-like buffers don't have a path,
        # so leave name unset rather than fabricating one.
        if isinstance(source, str):
            default_name = os.path.splitext(os.path.basename(source))[0]

    # Hand the post-decode buffer to the shared eager finalizer. The
    # helper runs the same validate -> populate attrs -> mask -> cast
    # -> set_nodata_attrs -> build DataArray pipeline the inline block
    # used to do. ``mask_sentinel`` honours the post-MinIsWhite
    # inversion stashed on ``geo_info._mask_nodata`` by
    # ``read_to_array`` / ``_read_cog_http``; on non-MinIsWhite files
    # it falls back to the raw declared sentinel.
    nodata = geo_info.nodata
    mask_sentinel = (
        getattr(geo_info, '_mask_nodata', nodata)
        if nodata is not None else None
    )
    return _finalize_eager_read(
        arr,
        geo_info=geo_info,
        nodata=nodata,
        mask_sentinel=mask_sentinel,
        mask_nodata=masked,
        dtype=dtype,
        window=window,
        name=default_name,
        allow_rotated=allow_rotated,
        allow_unparseable_crs=allow_unparseable_crs,
        mask_and_scale=unpack,
        parse_coordinates=parse_coordinates,
        band=band,
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
