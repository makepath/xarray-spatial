"""Eager (CPU) writer entry point and helpers.

Step 10 of issue #1813. Holds ``to_geotiff`` (the public eager writer),
``_write_single_tile`` (per-tile worker used by ``_write_vrt_tiled``),
and ``_write_vrt_tiled`` (the deprecated ``vrt_tiled=True`` path on
``to_geotiff``). Companion modules ``_writers/gpu.py`` and
``_writers/vrt.py`` hold the GPU writer and the public ``write_vrt``;
``to_geotiff`` dispatches to them when the caller asks for a GPU
output or a ``.vrt`` path.
"""
from __future__ import annotations

import numbers
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from typing import BinaryIO

from .._attrs import (
    _EXPERIMENTAL_CODECS,
    _LEVEL_RANGES,
    _VALID_COMPRESSIONS,
    _extract_rich_tags,
    _resolve_nodata_attr,
    _should_restore_nan_sentinel,
)
from .._backends._gpu_helpers import _is_gpu_data
from .._coords import (
    _BAND_DIM_NAMES,
    _has_no_georef_marker,
    coords_to_transform as _coords_to_transform,
    require_transform_for_georeferenced as _require_transform_for_georeferenced,
    transform_from_attr as _transform_from_attr,
)
from .._crs import _validate_crs_arg, _validate_crs_fallback, _wkt_to_epsg
from .._geotags import GeoTransform, RASTER_PIXEL_IS_AREA
from .._runtime import (
    GeoTIFFFallbackWarning,
    _geotiff_strict_mode,
    _gpu_fallback_warning_message,
)
from .._validation import (
    _validate_3d_writer_dims,
    _validate_nodata_arg,
    _validate_tile_size_arg,
    _validate_writer_spatial_shape,
    validate_write_metadata,
)
from .._writer import write
from .gpu import write_geotiff_gpu


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
               photometric: str | int = 'auto',
               allow_internal_only_jpeg: bool = False,
               allow_experimental_codecs: bool = False,
               allow_unparseable_crs: bool = False) -> str | BinaryIO:
    """Write data as a GeoTIFF or Cloud Optimized GeoTIFF.

    Tier: Stable for local-file output with ``compression`` in
    ``{'none', 'deflate', 'lzw', 'packbits', 'zstd'}`` on an axis-aligned
    grid. ``cog=True`` / overviews / BigTIFF are Advanced (work, but the
    caller should know the failure modes). GPU output, GDAL XML metadata
    pass-through, and ``extra_tags`` are Experimental. ``compression`` in
    ``{'lerc', 'jpeg2000', 'j2k', 'lz4'}`` is Experimental and requires
    ``allow_experimental_codecs=True``. ``compression='jpeg'`` is
    Internal-only and requires the dedicated ``allow_internal_only_jpeg``
    flag. See :data:`xrspatial.geotiff.SUPPORTED_FEATURES` for the full
    tier map (issue #2137).

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
    crs : int, numpy.integer, str, or None
        EPSG code (int or numpy integer scalar), WKT string, or PROJ
        string. If None and data is a DataArray, tries to read from
        attrs ('crs' for EPSG, 'crs_wkt' for WKT).

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

        Stable codecs (Tier 1, lossless, byte-for-byte round-trip):
        ``'none'``, ``'deflate'``, ``'lzw'``, ``'packbits'``,
        ``'zstd'``.

        Experimental codecs (Tier 3): ``'lerc'``, ``'jpeg2000'`` /
        ``'j2k'``, ``'lz4'``. Rejected by default; pass
        ``allow_experimental_codecs=True`` to opt in. The opt-in emits
        ``GeoTIFFFallbackWarning`` once per call so the caller knows
        the chosen codec carries no cross-backend numerical parity
        claim and uneven reader support across GDAL versions. ``'lerc'``
        accepts ``max_z_error`` for lossy compression with a bounded
        per-pixel error.

        Internal-only codec (Tier 4): ``'jpeg'``. Rejected on write by
        default because the encoder omits the JPEGTables tag and the
        produced files do not round-trip through libtiff / GDAL /
        rasterio. Pass ``allow_internal_only_jpeg=True`` to opt in to
        the internal-reader-only path (see that parameter for details).
        ``allow_experimental_codecs=True`` does NOT cover ``'jpeg'``:
        internal-only is a stricter tier than experimental, and the two
        flags do not collapse into one switch.
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
        Advanced: COG output materialises the full array because
        overview pyramids need it, and the all-IFDs-at-file-start layout
        only round-trips through readers that honour the COG layout
        contract. Write as Cloud Optimized GeoTIFF.
    overview_levels : list[int] or None
        Advanced: overview pyramids are an optional COG feature; the
        decimation factors and resampling choice affect downstream
        analytics in ways that are not byte-for-byte reproducible
        across backends. Overview decimation factors relative to full
        resolution.
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
        Advanced: BigTIFF uses 64-bit offsets; older readers that only
        speak classic TIFF cannot open the output. Force BigTIFF
        (64-bit offsets). None (default) auto-promotes when the
        estimated file size would exceed the classic-TIFF 4 GB limit.
        Matches the same kwarg on ``write_geotiff_gpu``.
    gpu : bool or None
        Experimental: requires cupy + numba CUDA, plus the optional
        nvCOMP / nvJPEG / nvJPEG2K libraries for codec-specific
        acceleration; backend parity with the CPU writer is tested for
        the Tier 1 codec set only. Force GPU compression. None
        (default) auto-detects CuPy data.
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
    allow_experimental_codecs : bool
        Opt in to the Tier 3 experimental codecs ``'lerc'``,
        ``'jpeg2000'`` / ``'j2k'``, and ``'lz4'`` (default ``False``).
        Setting ``compression=`` to one of those codecs without this
        flag raises ``ValueError`` whose message names the flag. With
        the flag set, the write proceeds and a
        ``GeoTIFFFallbackWarning`` is emitted once per call so the
        caller knows the chosen codec carries no cross-backend
        numerical parity claim and uneven reader support across GDAL
        versions. Does NOT cover ``compression='jpeg'``: the
        internal-only JPEG path keeps its own dedicated
        ``allow_internal_only_jpeg`` flag because internal-only is a
        stricter tier than experimental. The kwarg is forwarded
        unchanged to ``write_geotiff_gpu`` on the GPU dispatch path.
        See issue #2137.
    allow_internal_only_jpeg : bool
        Opt in to the experimental ``compression='jpeg'`` encode path
        (default ``False``). The encoder writes self-contained JFIF
        tiles without the TIFF JPEGTables tag (347); the file decodes
        through this library's reader but not through libtiff, GDAL,
        or rasterio. With the flag set, the write proceeds and a
        ``GeoTIFFFallbackWarning`` is emitted at call time. Without
        the flag, ``compression='jpeg'`` raises ``ValueError``. The
        kwarg is forwarded unchanged to ``write_geotiff_gpu`` on the
        GPU dispatch path so callers can reach the same experimental
        encode via ``to_geotiff(..., gpu=True)``. See issue #1845.
    allow_unparseable_crs : bool
        Opt in to writing an unvalidatable CRS string into
        ``GTCitationGeoKey`` (default ``False``). When ``False`` (the
        default since #1929), a ``crs=`` value that is neither an EPSG
        int nor a string that pyproj can resolve and is not
        structurally WKT (no ``PROJCS`` / ``GEOGCS`` / ``PROJCRS`` /
        ``GEOGCRS`` root) raises ``ValueError`` instead of landing
        verbatim in the citation field. Set to ``True`` to keep the
        pre-#1929 permissive behaviour. The fail-closed default
        protects against files where a typo'd PROJ string or an
        ``"EPSG:4326"`` token on a host without pyproj produces a
        citation that most readers cannot interpret. See issue #1929.

    Returns
    -------
    str or binary file-like
        The ``path`` argument (a string for filesystem paths, the
        file-like object for BytesIO destinations). Returning the path
        lines up with ``write_vrt`` and lets callers chain a write into
        a read without round-tripping through a variable; existing
        callers that discarded the previous ``None`` return are
        unaffected. See issue #1938.

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
    from .._reader import _coerce_path

    path = _coerce_path(path)

    # Reject bool / np.bool_ nodata up front. ``bool`` is a subclass of
    # ``int`` in Python, so a typo like ``nodata=True`` slips past every
    # downstream ``isinstance(nodata, (int, float))`` guard. The geotag
    # builder then writes ``str(True)`` -> ``"True"`` into GDAL_NODATA,
    # which no reader parses as numeric, so the round-trip drops the
    # sentinel without warning. Refuse rather than coerce ``True`` to 1:
    # the caller almost certainly meant a real numeric sentinel. See
    # issue #1911.
    if isinstance(nodata, (bool, np.bool_)):
        raise TypeError(
            f"nodata must be numeric (int or float), got {nodata!r}")

    # tiled=False ignores tile_size, so only validate when tiled output
    # is requested. Shared with write_geotiff_gpu via
    # _validate_tile_size_arg so both writers keep identical validation.
    if tiled:
        _validate_tile_size_arg(tile_size)

    _validate_nodata_arg(nodata)

    # Issue #2075: reject zero-height / zero-width inputs before any
    # dispatch decision. Clip / window pipelines naturally produce empty
    # rasters and the writers used to accept them, produce a TIFF whose
    # IFD claimed shape ``(0, N)`` / ``(N, 0)``, and surface a generic
    # "Invalid image dimensions" only at read time. Fail closed at the
    # entry point with a message that names the offending dim.
    _shape = getattr(data, 'shape', None)
    _dims = getattr(data, 'dims', None)
    _validate_writer_spatial_shape(_shape, _dims)

    # Issue #1987 ambiguous-metadata checks. The hook is a no-op
    # when no check is registered, so this call is safe even if every
    # check is later unregistered for a specific entry point.
    _attrs = getattr(data, 'attrs', None) or {}
    _coords = getattr(data, 'coords', None)
    _coord_y = _coords['y'].values if (
        _coords is not None and 'y' in _coords
    ) else None
    _coord_x = _coords['x'].values if (
        _coords is not None and 'x' in _coords
    ) else None
    validate_write_metadata({
        'crs_kwarg': crs,
        'attrs_crs': _attrs.get('crs'),
        'attrs_crs_wkt': _attrs.get('crs_wkt'),
        'nodata_kwarg': nodata,
        'attrs_nodata': _attrs.get('nodata'),
        'attrs_nodatavals': _attrs.get('nodatavals'),
        'coord_y': _coord_y,
        'coord_x': _coord_x,
        'no_georef_marker': _has_no_georef_marker(data),
    })

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
        # directly, masking the interop break. Refuse the write by
        # default and surface the same ``allow_internal_only_jpeg=True``
        # opt-in that ``write_geotiff_gpu`` already accepts, so the
        # auto-dispatch entry point can reach the experimental
        # internal-reader-only path the explicit GPU entry point
        # exposes (issue #1845).
        if compression.lower() == 'jpeg' and not allow_internal_only_jpeg:
            raise ValueError(
                "compression='jpeg' is not supported: the encoder writes "
                "self-contained JFIF streams without the required "
                "JPEGTables tag (347), so other readers (libtiff, GDAL, "
                "rasterio) reject the file. Use 'deflate', 'zstd', or "
                "'lzw' instead. Pass allow_internal_only_jpeg=True to "
                "opt in to the experimental internal-reader-only path "
                "(issue #1845).")
        # The JPEG opt-in warning is emitted below once we know the
        # dispatch decision: ``write_geotiff_gpu`` emits its own warning
        # on the GPU path, so emitting here would double-warn callers
        # of ``to_geotiff(gpu=True, compression='jpeg',
        # allow_internal_only_jpeg=True)``.

        # Tier 3 experimental-codec gate (issue #2137). Lerc, jpeg2000 /
        # j2k, and lz4 sit in ``_VALID_COMPRESSIONS`` for wire-format
        # reasons but their cross-backend numerical parity, reader
        # support across GDAL versions, and (for lerc) bounded lossy
        # behaviour all carry caveats the default writer should not
        # silently accept. Mirror the ``allow_internal_only_jpeg`` shape
        # so callers learn the opt-in name from the rejection message
        # and can fix the call site in one line. The opt-in warning is
        # emitted below once the GPU dispatch decision is known so the
        # GPU path does not double-warn (``write_geotiff_gpu`` emits its
        # own warning on the GPU path).
        if (compression.lower() in _EXPERIMENTAL_CODECS
                and not allow_experimental_codecs):
            raise ValueError(
                f"compression={compression!r} is experimental: cross-backend "
                "numerical parity is not claimed and reader support across "
                "GDAL versions is uneven. Pass allow_experimental_codecs=True "
                "to opt in, or use 'deflate', 'zstd', or 'lzw' for a "
                "stable lossless codec (issue #2137).")

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

    # Resolve GPU dispatch up front so the JPEG opt-in warning fires
    # exactly once. ``write_geotiff_gpu`` emits its own warning on the
    # GPU path; emitting here as well would double-warn callers of
    # ``to_geotiff(gpu=True, compression='jpeg',
    # allow_internal_only_jpeg=True)``. VRT and CPU paths receive the
    # warning here. On GPU-to-CPU fallback the GPU writer has already
    # warned before raising, so the CPU fallback does not warn twice.
    auto_detected_gpu = gpu is None
    use_gpu = gpu if gpu is not None else _is_gpu_data(data)
    if (isinstance(compression, str)
            and compression.lower() == 'jpeg'
            and allow_internal_only_jpeg
            and not use_gpu):
        warnings.warn(
            "to_geotiff(compression='jpeg', "
            "allow_internal_only_jpeg=True) writes JFIF tiles "
            "without the TIFF JPEGTables tag (347); the file decodes "
            "through xrspatial but may fail in libtiff, GDAL, or "
            "rasterio. See issue #1845.",
            GeoTIFFFallbackWarning,
            stacklevel=2,
        )
    # Tier 3 experimental-codec opt-in warning (issue #2137). Mirrors
    # the JPEG flag's "warn once, after dispatch is resolved" shape:
    # ``write_geotiff_gpu`` emits its own warning on the GPU path with
    # a backend-specific caveat, so the CPU dispatcher only warns when
    # the write is staying on CPU.
    if (isinstance(compression, str)
            and compression.lower() in _EXPERIMENTAL_CODECS
            and allow_experimental_codecs
            and not use_gpu):
        warnings.warn(
            f"to_geotiff(compression={compression!r}, "
            "allow_experimental_codecs=True): experimental codec, "
            "no cross-backend parity claim and uneven reader support "
            "across GDAL versions. See issue #2137.",
            GeoTIFFFallbackWarning,
            stacklevel=2,
        )

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
                         photometric=photometric,
                         allow_unparseable_crs=allow_unparseable_crs)
        return path

    # Dispatch to write_geotiff_gpu when GPU was selected (explicit
    # ``gpu=True`` or auto-detected CuPy data). ``auto_detected_gpu``
    # and ``use_gpu`` were computed above to gate the JPEG opt-in
    # warning; reuse them so the call sites stay in sync.
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
            write_geotiff_gpu(
                data, path, crs=crs, nodata=nodata,
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
                photometric=photometric,
                allow_internal_only_jpeg=allow_internal_only_jpeg,
                allow_experimental_codecs=allow_experimental_codecs,
                allow_unparseable_crs=allow_unparseable_crs,
            )
            return path
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
    # Default for the issue #1988 gate. Overwritten with the actual
    # ``attrs['masked_nodata']`` value below when ``data`` is a
    # DataArray; bare numpy / cupy positional inputs have no attrs
    # so they keep the pre-#1988 NaN->sentinel rewrite.
    restore_sentinel = True

    # Resolve crs argument: can be int (EPSG) or str (WKT/PROJ).
    # ``numbers.Integral`` covers plain ``int`` and numpy integer scalars
    # (``np.int32``, ``np.int64``); ``_validate_crs_arg`` already rejects
    # bool. Coerce to plain ``int`` so downstream ``epsg`` is a real int.
    _validate_crs_arg(crs)
    if isinstance(crs, numbers.Integral):
        epsg = int(crs)
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
        # Fail closed when coords are present but no transform could be
        # derived (e.g. 1x1 without ``attrs['transform']``) instead of
        # silently writing a non-georeferenced TIFF that round-trips back
        # with integer pixel coords (#1945).
        _require_transform_for_georeferenced(data, geo_transform)
        if epsg is None and crs is None:
            crs_attr = data.attrs.get('crs')
            if isinstance(crs_attr, str):
                epsg = _wkt_to_epsg(crs_attr)
                if epsg is None and wkt_fallback is None:
                    wkt_fallback = crs_attr
            elif crs_attr is not None:
                # Same gate as the kwarg path: reject bool / non-int
                # types and confirm the EPSG resolves before writing it
                # to disk. Without this, ``attrs={'crs': True}`` round-
                # trips as EPSG=1 (issue #1971 follow-up).
                _validate_crs_arg(crs_attr)
                epsg = int(crs_attr)
            if epsg is None:
                wkt = data.attrs.get('crs_wkt')
                if isinstance(wkt, str):
                    epsg = _wkt_to_epsg(wkt)
                    if epsg is None and wkt_fallback is None:
                        wkt_fallback = wkt
        if nodata is None:
            nodata = _resolve_nodata_attr(data.attrs)
        # Issue #1988: ``attrs['masked_nodata']`` records whether the
        # reader promoted the sentinel to NaN. Writers reverse that
        # rewrite only when the read side did the forward step. Default
        # (attr absent) is True, which preserves pre-#1988 behaviour for
        # external DataArrays.
        restore_sentinel = _should_restore_nan_sentinel(data.attrs)
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
            from .._writer import write_streaming
            # Issue #1929: refuse to write an unvalidatable CRS string
            # into GTCitationGeoKey unless the caller opts in. ``epsg``
            # is set when ``_wkt_to_epsg`` succeeded; only the fallback
            # branch needs the guard.
            if epsg is None:
                _validate_crs_fallback(wkt_fallback, allow_unparseable_crs)
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
                restore_sentinel=restore_sentinel,
            )
            return path

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
    if (nodata is not None and arr.dtype.kind == 'f'
            and not np.isnan(nodata) and restore_sentinel):
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

    # Issue #1929: refuse to write an unvalidatable CRS string into
    # GTCitationGeoKey unless the caller opts in.
    if epsg is None:
        _validate_crs_fallback(wkt_fallback, allow_unparseable_crs)
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
        restore_sentinel=restore_sentinel,
    )
    return path


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
                       photometric: str | int = 'auto',
                       restore_sentinel: bool = True):
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
    if (nodata is not None and arr.dtype.kind == 'f'
            and not np.isnan(nodata) and restore_sentinel):
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
          photometric=photometric,
          restore_sentinel=restore_sentinel)


def _write_vrt_tiled(data, vrt_path, *, crs=None, nodata=None,
                     compression='zstd', compression_level=None,
                     tile_size=256, predictor: bool | int = False,
                     bigtiff=None, max_z_error: float = 0.0,
                     photometric: str | int = 'auto',
                     allow_unparseable_crs: bool = False):
    """Write a DataArray as a directory of tiled GeoTIFFs with a VRT index.

    This enables streaming dask arrays to disk without materializing the
    full array in RAM.
    """
    _validate_nodata_arg(nodata)

    # Issue #1987 ambiguous-metadata checks; mirrors the call in
    # ``to_geotiff`` so the dask-VRT write path enforces the same
    # crs/crs_wkt / nodata / coord rules.
    _attrs = getattr(data, 'attrs', None) or {}
    _coords = getattr(data, 'coords', None)
    _coord_y = _coords['y'].values if (
        _coords is not None and 'y' in _coords
    ) else None
    _coord_x = _coords['x'].values if (
        _coords is not None and 'x' in _coords
    ) else None
    validate_write_metadata({
        'crs_kwarg': crs,
        'attrs_crs': _attrs.get('crs'),
        'attrs_crs_wkt': _attrs.get('crs_wkt'),
        'nodata_kwarg': nodata,
        'attrs_nodata': _attrs.get('nodata'),
        'attrs_nodatavals': _attrs.get('nodatavals'),
        'coord_y': _coord_y,
        'coord_x': _coord_x,
        'no_georef_marker': _has_no_georef_marker(data),
    })

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

    # Resolve CRS. ``numbers.Integral`` covers numpy integer scalars
    # (``np.int32``, ``np.int64``) so ``crs=np.int64(4326)`` does not
    # silently fall through to ``epsg=None``. Validator already
    # rejects bool.
    _validate_crs_arg(crs)
    epsg = None
    wkt_fallback = None
    if isinstance(crs, numbers.Integral):
        epsg = int(crs)
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
                # Same gate as the kwarg path: reject bool / non-int
                # types and confirm the EPSG resolves before writing it
                # to disk. Without this, ``attrs={'crs': True}`` round-
                # trips as EPSG=1 (issue #1971 follow-up).
                _validate_crs_arg(crs_attr)
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
        # Issue #1988: mirror the ``to_geotiff`` gate so per-tile
        # writes only NaN-rewrite when the read side promoted the
        # sentinel. See ``_should_restore_nan_sentinel`` for the
        # semantics; default True keeps existing behaviour.
        restore_sentinel = _should_restore_nan_sentinel(data.attrs)
        geo_transform = _transform_from_attr(data.attrs.get('transform'))
        if geo_transform is None:
            geo_transform = _coords_to_transform(data)
        # Match the to_geotiff fail-closed guard so VRT writes don't
        # silently produce non-georeferenced tiles either (#1945).
        _require_transform_for_georeferenced(data, geo_transform)
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

    # Issue #1929: refuse to write an unvalidatable CRS string into the
    # per-tile GTCitationGeoKey fields unless the caller opts in.
    # ``_write_single_tile`` forwards ``wkt_fallback`` to the geokey
    # builder once per tile, so validate once up front rather than
    # per-tile.
    if epsg is None:
        _validate_crs_fallback(wkt_fallback, allow_unparseable_crs)

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
                    photometric=photometric,
                    restore_sentinel=restore_sentinel)
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
                    photometric=photometric,
                    restore_sentinel=restore_sentinel)

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
    from .._vrt import write_vrt as _write_vrt_fn
    _write_vrt_fn(vrt_path, tile_paths, relative=True, nodata=nodata)

