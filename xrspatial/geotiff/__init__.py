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
# ``xrspatial.geotiff._reader``. See issue #1708.
from ._attrs import (_LEVEL_RANGES, _VALID_COMPRESSIONS, GEOREF_STATUS_CRS_ONLY,  # noqa: F401
                     GEOREF_STATUS_FULL, GEOREF_STATUS_NONE, GEOREF_STATUS_ROTATED_DROPPED,
                     GEOREF_STATUS_TRANSFORM_ONLY, GEOREF_STATUS_VALUES, _extent_to_window,
                     _extract_rich_tags, _finalize_eager_read, _resolve_nodata_attr)
# Re-export only; called by xrspatial/geotiff/tests/test_nodata_*.py.
from ._backends._gpu_helpers import _apply_nodata_mask_gpu  # noqa: F401
from ._backends._gpu_helpers import _is_gpu_data  # noqa: F401
from ._backends.dask import read_geotiff_dask
from ._backends.gpu import read_geotiff_gpu
from ._backends.vrt import read_vrt
# etc. See the ``# noqa: F401`` pattern at lines 89 and 119 for the same convention.
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
from ._errors import (ConflictingCRSError, ConflictingNodataError, GeoTIFFAmbiguousMetadataError,
                      InvalidCRSCodeError, MixedBandMetadataError, NonUniformCoordsError,
                      RotatedTransformError, UnknownCRSModelTypeError, UnparseableCRSError)
from ._geotags import RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT, GeoTransform  # noqa: F401
from ._reader import _MAX_CLOUD_BYTES_SENTINEL, UnsafeURLError
from ._reader import read_to_array as _read_to_array
from ._runtime import (_CRS_WKT_DEPRECATED_SENTINEL, _GPU_DEPRECATED_SENTINEL,  # noqa: F401
                       _MISSING_SOURCES_SENTINEL, _ON_GPU_FAILURE_SENTINEL, GeoTIFFFallbackWarning,
                       _geotiff_strict_mode, _gpu_fallback_warning_message)
from ._validation import (_validate_3d_writer_dims, _validate_chunks_arg,  # noqa: F401
                          _validate_tile_size_arg)
# Re-export only; called by xrspatial/geotiff/tests/test_nodata_no_extra_copy_1553.py.
# ``_writer.write`` (alias for ``_writer._write``) is module-private;
# see ``_writer.py`` docstring and issue #2138. The public eager write
# surface is :func:`to_geotiff`; do not re-export the array-level
# entry point here. The dotted path ``xrspatial.geotiff._writer._write``
# still works for the handful of internal call sites that need it.
from ._writers.eager import _write_single_tile  # noqa: F401
from ._writers.eager import to_geotiff
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
    'GEOREF_STATUS_CRS_ONLY',
    'GEOREF_STATUS_FULL',
    'GEOREF_STATUS_NONE',
    'GEOREF_STATUS_ROTATED_DROPPED',
    'GEOREF_STATUS_TRANSFORM_ONLY',
    'GEOREF_STATUS_VALUES',
    'InvalidCRSCodeError',
    'MixedBandMetadataError',
    'NonUniformCoordsError',
    'RotatedTransformError',
    'SUPPORTED_FEATURES',
    'UnknownCRSModelTypeError',
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
#   ``allow_internal_only_jpeg=True`` (issue #1845);
#   ``allow_experimental_codecs`` does NOT cover it.
#
# Tests in ``xrspatial/geotiff/tests/test_supported_features_tiers_2137.py``
# walk the mapping and assert that every Tier 3 codec rejects without
# the opt-in flag and every Tier 4 codec rejects without its own
# dedicated flag. The user-guide notebook
# (``examples/user_guide/39_GeoTIFF_IO.ipynb``) renders the same
# mapping as a table so the documentation cannot drift from the code.
#
# See issue #2137.
from ._attrs import SUPPORTED_FEATURES  # noqa: E402


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
        grid instead of raising ``RotatedTransformError`` (issues #2115,
        #2267).
    """
    # ``_parse_cog_http_meta`` is imported from ``_cog_http`` directly
    # rather than re-routed through ``_reader`` because the
    # ``open_geotiff(..., chunks=...)`` fsspec metadata path is not part
    # of the ``_reader.*`` monkeypatch surface (no test patches
    # ``_reader._parse_cog_http_meta`` and then exercises this branch).
    # The eager / dask HTTP paths that ARE patched route through
    # ``_cog_http._read_cog_http`` and ``_backends/dask.py``'s
    # ``_HTTPSource`` construction, both of which still go through
    # ``_reader`` for the patchable names. See PR-J / #2258.
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
        # ``.tif.ovr`` sidecar discovery (issue #2239). Without it,
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
        from ._sidecar import attach_sidecar_origin, find_sidecar, load_sidecar
        sidecar_origin: dict[int, tuple] = {}
        sidecar_path = find_sidecar(source)
        if sidecar_path is not None:
            sidecar = load_sidecar(sidecar_path)
            # The origin mapping is consumed below for georef extraction
            # only -- strip/tile bytes are sliced by ``read_to_array`` on
            # the actual read. A sidecar IFD that carries its own
            # GeoKeyDirectory / ModelPixelScale / ModelTiepoint /
            # ModelTransformation needs the sidecar's byte order to
            # parse cleanly; without the mapping the helper falls back
            # to the base file's bytes (today's default, correct under
            # the usual GDAL convention). See issue #2315.
            sidecar_origin = attach_sidecar_origin(
                sidecar.ifds, sidecar.data, sidecar.header)
            ifds = ifds + sidecar.ifds
        ifd = select_overview_ifd(ifds, overview_level)
        # Inherit georef from the level-0 IFD when the overview itself
        # has no geokeys (issue #1640). Pass-through for level 0. The
        # sidecar IFDs typically lack geokeys so the inheritance pulls
        # from the base file's full-resolution IFD as GDAL does. When a
        # sidecar IFD does declare its own georef payload, ``georef_origin``
        # routes the parse to the sidecar's bytes / byte order so the
        # sidecar's georef wins. See issue #2315.
        georef_origin = (
            {iid: (od, oh.byte_order)
             for iid, (od, oh) in sidecar_origin.items()}
            if sidecar_origin else None
        )
        geo_info = extract_geo_info_with_overview_inheritance(
            ifd, ifds, data, header.byte_order,
            allow_rotated=allow_rotated,
            sidecar_origin=georef_origin)
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
                 max_cloud_bytes: int | None = (
                     _MAX_CLOUD_BYTES_SENTINEL),  # type: ignore[assignment]
                 on_gpu_failure: str = _ON_GPU_FAILURE_SENTINEL,
                 missing_sources: str = _MISSING_SOURCES_SENTINEL,
                 allow_rotated: bool = False,
                 allow_unparseable_crs: bool = False,
                 band_nodata: str | None = None,
                 mask_nodata: bool = True,
                 ) -> xr.DataArray:
    """Read a GeoTIFF, COG, or VRT file into an xarray.DataArray.

    Release-contract tier (epic #2340; see
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
      No cross-backend numerical parity claim. JPEG-in-TIFF is
      ``[internal-only]`` and the read path only decodes files this
      library itself wrote.
    * Out of scope for this release (allowed to raise): full GDAL VRT
      parity, warped / reprojection VRTs, rotated/sheared write
      support.

    See :data:`xrspatial.geotiff.SUPPORTED_FEATURES` for the full tier
    map (issue #2137).

    Automatically dispatches to the best backend:
    - ``gpu=True``: GPU-accelerated read via nvCOMP (returns CuPy)
    - ``chunks=N``: Dask lazy read via windowed chunks
    - ``gpu=True, chunks=N``: Dask+CuPy for out-of-core GPU pipelines
    - Default: NumPy eager read

    VRT files are auto-detected by extension. The supported VRT subset
    is narrow on purpose (issue #2321; epic #2340). See the "VRT
    support matrix" section in ``docs/source/reference/geotiff.rst``
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
        [stable for local file paths; advanced for HTTP/fsspec URIs;
        advanced for ``.vrt`` paths] File path, HTTP URL, cloud URI
        (s3://, gs://, az://), or a binary file-like object
        (e.g. ``io.BytesIO``) with read+seek. VRT, dask-chunked, GPU,
        and remote-URL paths require a string; in-memory file-like
        buffers go through the eager numpy reader.
    dtype : str, numpy.dtype, or None
        [stable] Cast the result to this dtype after reading. None
        keeps the file's native dtype. Float-to-int casts raise
        ValueError to prevent accidental data loss.
    window : tuple or None
        [stable] ``(row_start, col_start, row_stop, col_stop)`` for
        windowed reading.
    overview_level : int or None
        [advanced] Overview level (0 = full resolution). Must be a
        non-negative int or ``None``; passing ``bool`` or any other
        type raises ``TypeError``. External ``.tif.ovr`` sidecars are
        also [advanced] and are tested but not load-bearing for
        release-gate parity.
    band : int or None
        [stable] Band index (0-based). None returns all bands.
    name : str or None
        [stable] Name for the DataArray.
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
        [stable] Maximum allowed pixel count (width * height *
        samples). None uses the default (~1 billion). Raise to read
        legitimately large files.
    max_cloud_bytes : int or None, optional
        [advanced] fsspec cloud reads can run up cost on large objects;
        the budget defends against accidental large downloads but the
        eager path still pulls the full object once the budget allows.
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
        [experimental] Forwarded to ``read_geotiff_gpu`` when
        ``gpu=True``. Controls whether GPU decode failures fall back
        to CPU (``'auto'``, default) or re-raise the original exception
        (``'strict'``). Passing this kwarg with ``gpu=False`` raises
        ``ValueError`` because the policy only applies to the GPU
        pipeline. See ``read_geotiff_gpu`` for the full description.
    missing_sources : {'raise', 'warn'}, optional
        [advanced] VRT mosaics can return partial output under
        ``missing_sources='warn'`` when a backing source is unreadable;
        the ``attrs['vrt_holes']`` entry records which sources were
        skipped so downstream code can detect the partial mosaic.
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
        [advanced] VRT-only. Opt-out for the fail-closed check that
        rejects VRT sources whose bands declare disagreeing per-band
        nodata sentinels (issue #1987 PR 5). When ``None`` (the
        default), a VRT
        that mosaics bands with different sentinels raises
        ``MixedBandMetadataError``; flattening to one value would let
        one band's valid pixels collide with another band's sentinel.
        Pass ``band_nodata='first'`` to keep the legacy behaviour of
        using band 0's sentinel for the whole mosaic. Passing this
        kwarg with a non-VRT source raises ``ValueError`` because the
        policy only applies to the VRT pipeline.
    mask_nodata : bool, default True
        [stable] If True (the default), replace the nodata sentinel
        with ``NaN``; integer rasters get promoted to ``float64`` first
        so NaN can be represented. If False, skip the sentinel-to-NaN
        step and keep the source dtype. ``attrs['nodata']`` still
        carries the raw sentinel either way, so downstream code can
        mask explicitly. Pass ``mask_nodata=False`` when you want to
        preserve an integer source dtype via ``dtype=``: the default
        ``mask_nodata=True`` promotes to ``float64`` whenever the
        sentinel matches an actual pixel, and ``dtype=<integer>`` then
        raises ``ValueError`` on the float-to-int cast.
    allow_rotated : bool, default False
        [advanced] Read-only opt-in. ``to_geotiff`` does not currently
        emit ``rotated_affine``; it rejects DataArrays that carry the
        attr (``ValueError`` naming the attr) unless the caller passes
        ``drop_rotation=True`` to accept the loss explicitly (#2216).
        Read-side opt-in for rotated / sheared ``ModelTransformationTag``
        files. By default the reader raises ``RotatedTransformError``
        (a ``GeoTIFFAmbiguousMetadataError`` / ``ValueError`` subclass;
        previously a bare ``NotImplementedError`` -- see #2267) because
        the rest of xrspatial assumes an axis-aligned grid.
        ``allow_rotated=True`` reads the pixel grid without the
        geospatial assumption: the result has integer pixel coords on
        ``x`` / ``y`` and both ``attrs['crs']`` and ``attrs['crs_wkt']``
        are dropped. The CRS attrs are dropped together with the
        transform because keeping them while the axis-aligned transform
        is gone misleads downstream code that gates on
        ``"crs" in da.attrs`` to mean the array is spatially usable
        (issue #2126). The rotated 6-tuple itself is surfaced on
        ``attrs['rotated_affine']`` as ``(a, b, c, d, e, f)`` (rasterio
        ``Affine`` ordering) so consumers that know how to handle
        rotated rasters can recover the mapping (issue #2129). The
        contract is read-only -- writes must either reproject onto an
        axis-aligned grid first, or pass ``drop_rotation=True`` to
        ``to_geotiff`` / ``write_geotiff_gpu`` to accept the loss; the
        ``ModelTransformationTag`` emit path is tracked separately
        (issue #2115).
    allow_unparseable_crs : bool, default False
        [advanced] Read-side opt-in for CRS strings that pyproj cannot
        resolve and that do not parse as WKT. When ``False`` (the
        default since #1929), an unrecognised CRS payload raises
        ``UnparseableCRSError`` instead of landing in ``attrs['crs_wkt']``
        verbatim. Set to ``True`` to keep the pre-#1929 permissive
        behaviour where the citation field passes through unchanged.
        Matches the same kwarg on ``to_geotiff`` / ``write_geotiff_gpu``
        so a value the reader accepted can survive a round-trip.

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

    Examples
    --------
    Safe VRT usage. Mosaic two compatible tiles and read with the
    fail-closed defaults:

    >>> from xrspatial.geotiff import open_geotiff, write_vrt
    >>> vrt_path = write_vrt(  # doctest: +SKIP
    ...     'mosaic.vrt',
    ...     source_files=['tile_west.tif', 'tile_east.tif'],
    ... )
    >>> da = open_geotiff(vrt_path)  # doctest: +SKIP

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

    # All dispatcher-level kwarg rejection lives in
    # ``_validate_dispatch_kwargs`` so the three direct backends
    # (``read_geotiff_dask``, ``read_geotiff_gpu``, ``read_vrt``)
    # surface the same errors when called directly (issue #2175,
    # parent #2162). The previous inline block at this line is now
    # a single call that runs ``_validate_overview_level_arg`` (issue
    # #2074), the ``on_gpu_failure`` GPU-only guard (issue #1615),
    # the ``missing_sources`` VRT-only guard (issue #1810), the
    # ``band_nodata`` VRT-only guard (issue #1987), the
    # ``max_cloud_bytes`` non-VRT non-GPU non-dask guard (issue #1974),
    # and the file-like source restrictions for gpu/chunks.
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
    _is_vrt_source = (
        isinstance(source, str) and source.lower().endswith('.vrt'))

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

    # File-like buffer rejections for ``gpu=True`` / ``chunks=...`` already
    # fired inside ``_validate_dispatch_kwargs`` above; the non-VRT branches
    # below run with a string source or an eager file-like.

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

    if name is None:
        # Derive from source path. File-like buffers don't have a path,
        # so leave name unset rather than fabricating one.
        if isinstance(source, str):
            name = os.path.splitext(os.path.basename(source))[0]

    # Hand the post-decode buffer to the shared eager finalizer
    # (issue #2179). The helper runs the same validate -> populate attrs
    # -> mask -> cast -> set_nodata_attrs -> build DataArray pipeline
    # the inline block used to do. ``mask_sentinel`` honours the
    # post-MinIsWhite inversion stashed on ``geo_info._mask_nodata`` by
    # ``read_to_array`` / ``_read_cog_http`` (#1809); on non-MinIsWhite
    # files it falls back to the raw declared sentinel.
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
        mask_nodata=mask_nodata,
        dtype=dtype,
        window=window,
        name=name,
        allow_rotated=allow_rotated,
        allow_unparseable_crs=allow_unparseable_crs,
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
