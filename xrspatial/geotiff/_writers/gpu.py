"""GPU writer entry point.

Holds ``_write_geotiff_gpu``, which compresses tiles on the device via
nvCOMP (when available) and falls back to the
eager CPU writer when nvCOMP is missing or the device path raises
under ``on_gpu_failure='auto'``.
"""
from __future__ import annotations

import numbers
import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from typing import BinaryIO

    import cupy

from .._attrs import (_EXPERIMENTAL_CODECS, _LEVEL_RANGES, _extract_rich_tags, _resolve_nodata_attr,
                      _should_restore_nan_sentinel)
from .._coords import _BAND_DIM_NAMES, _has_no_georef_marker
from .._coords import require_transform_for_georeferenced as _require_transform_for_georeferenced
from .._coords import resolve_georef as _resolve_georef
from .._crs import _validate_crs_arg, _validate_crs_fallback, _wkt_to_epsg
from .._nodata import NodataLifecycle as _NL
from .._runtime import GeoTIFFFallbackWarning, _resolve_spatial_coords
from .._validation import (_validate_3d_writer_dims, _validate_no_rotated_affine,
                           _validate_nodata_arg, _validate_tile_size_arg,
                           _validate_writer_spatial_shape, validate_write_metadata)


def _warn_dask_materialised() -> None:
    """Warn that a dask-backed input is about to be fully materialised.

    Non-COG dask input streams through the GPU writer one tile-row
    band at a time (issue #3166), so this fires only from the
    ``cog=True`` ``.compute()`` sites in ``_write_geotiff_gpu``:
    overview generation needs the full-resolution array on device.
    Emitted from both the DataArray and positional compute sites so
    explicit and auto-detected GPU writes warn the same way.

    Deliberately does NOT participate in the
    ``XRSPATIAL_GEOTIFF_STRICT`` promotion that most
    ``GeoTIFFFallbackWarning`` sites apply: there is no opt-out flag
    for the materialisation, so promotion would turn every dask+cupy
    COG write into a hard failure. Same shape as the JPEG /
    experimental-codec opt-in warnings, which also stay warnings under
    strict mode.

    The warning stays truthful even when the GPU write later falls
    back to CPU (e.g. nvCOMP raises after dispatch): the ``.compute()``
    has already run by the time the fallback fires.
    """
    warnings.warn(
        "Dask-backed input with cog=True is materialised in full on "
        "device before encoding: GPU overview generation needs the "
        "whole array, so streaming_buffer_bytes is ignored. Pass "
        "cog=False to stream per tile-row band. See issue #3166.",
        GeoTIFFFallbackWarning,
        stacklevel=3,
    )


def _compute_gpu_samples_hint(data) -> int:
    """Return the band count using the same convention the GPU writer's
    band-first -> band-last remap uses.

    The remap below in ``_write_geotiff_gpu`` moves bands from
    ``shape[0]`` to ``shape[2]`` for band-first DataArrays. The
    MinIsWhite single-band guard runs *before* that remap, so reading
    ``data.shape[2]`` blindly would treat a band-first ``(1, H, W)``
    array as having ``W`` samples and miss the single-band rejection.
    Picking the band axis the same way the remap does keeps the guard
    and the actual on-device band placement in sync.

    Returns 1 for non-3D inputs. For 3D inputs without ``.dims`` (raw
    arrays), defaults to band-last (``shape[2]``); the writer's other
    entry-point validators reject ambiguous-dim DataArrays separately.
    """
    if getattr(data, 'ndim', None) != 3:
        return 1
    dims = getattr(data, 'dims', None)
    if (dims is not None
            and len(dims) == 3
            and dims[0] in _BAND_DIM_NAMES):
        return int(data.shape[0])
    return int(data.shape[2])


def _write_geotiff_gpu(data: xr.DataArray | cupy.ndarray | np.ndarray,
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
                       streaming_buffer_bytes: int = 256 * 1024 * 1024,
                       max_z_error: float = 0.0,
                       photometric: str | int = 'auto',
                       allow_internal_only_jpeg: bool = False,
                       allow_experimental_codecs: bool = False,
                       allow_unparseable_crs: bool = False,
                       drop_rotation: bool = False,
                       pack: bool = False,
                       ) -> str | BinaryIO:
    """Write a CuPy-backed DataArray as a GeoTIFF with GPU compression.

    Release-contract tier (see
    ``docs/source/reference/release_gate_geotiff.rst`` and
    ``docs/source/reference/geotiff_release_contract.rst``): the
    entire entry point is [experimental]. The surface may shift
    without a deprecation window and ``to_geotiff`` is the canonical
    writer. Requires cupy + numba CUDA plus optional nvCOMP / nvJPEG /
    nvJPEG2K libraries for codec-specific acceleration; cross-backend
    numerical parity with ``to_geotiff`` is tested for the Tier 1
    codec set only. Tier 3 codecs (``'lerc'``, ``'jpeg2000'`` /
    ``'j2k'``, ``'lz4'``) require the explicit
    ``allow_experimental_codecs=True`` opt-in; the [internal-only]
    ``'jpeg'`` codec keeps its own dedicated
    ``allow_internal_only_jpeg`` flag. See
    :data:`xrspatial.geotiff.SUPPORTED_FEATURES` for the full tier map.

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
        [experimental] 2D or 3D raster. CuPy-backed inputs stay on
        device; NumPy inputs are uploaded via
        ``cupy.asarray(np.asarray(data))`` before compression (matches
        ``to_geotiff`` parity). Dask-backed inputs stream one tile-row
        band at a time unless ``cog=True`` (see
        ``streaming_buffer_bytes``).
    path : str or binary file-like
        [experimental] Output file path or any object with a ``write``
        method (e.g. ``io.BytesIO``). ``cog=True`` requires a string
        path: the auto-dispatch path through
        ``to_geotiff(gpu=True, cog=True)`` rejects file-like
        destinations, and the explicit GPU writer mirrors that rule.
    crs : int, numpy.integer, str, or None
        [experimental] EPSG code (int or numpy integer scalar) or WKT
        string. EPSG codes are strongly preferred for interop; the
        WKT-only path emits a user-defined CRS (32767) with the WKT
        stored in ``GTCitationGeoKey``, which many non-libgeotiff
        readers ignore. A ``UserWarning`` is emitted when the WKT-only
        path is taken.
    nodata : float, int, or None
        [experimental] NoData value.
    compression : str
        [experimental for Tier 1 codecs on this path; experimental
        gated by ``allow_experimental_codecs=True`` for Tier 3 codecs;
        internal-only gated by ``allow_internal_only_jpeg=True`` for
        ``'jpeg'``] Codec name. Accepts the same set ``to_geotiff``
        lists in its own signature: ``'none'``, ``'deflate'``,
        ``'lzw'``, ``'jpeg'``, ``'packbits'``, ``'zstd'``, ``'lz4'``,
        ``'jpeg2000'`` (alias ``'j2k'``), or ``'lerc'``.

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
          readers.
        - ``'jpeg2000'`` and ``'j2k'``: nvJPEG2K GPU encode when
          available, glymur CPU encode otherwise. The two paths are
          not byte-for-byte identical (different libraries, different
          default parameters); use ``to_geotiff`` if you need exact
          CPU-writer parity.
        - ``'lerc'``, ``'lzw'``, ``'packbits'``, and ``'lz4'``: no
          nvCOMP/CUDA accelerator, so these fall through to the CPU
          encoder for byte-stable parity with ``to_geotiff``.
    compression_level : int or None
        [experimental] Compression effort level. Validated against the
        same codec ranges as ``to_geotiff`` (deflate 1-9, zstd 1-22,
        lz4 0-16); out-of-range values raise ``ValueError``. Tiles
        compressed through the CPU codecs (lzw / packbits / lerc, and
        deflate / zstd / lz4 when nvCOMP is unavailable) honor the
        level. The nvCOMP device encoder for deflate / zstd does not
        expose level control: when it handles the tiles, an explicit
        level is ignored and a ``UserWarning`` is emitted.
    tiled : bool
        [experimental] Must be True (default). The GPU writer is
        tiled-only because nvCOMP batch compression operates on
        per-tile streams; passing ``tiled=False`` raises ``ValueError``
        rather than silently producing a tiled file. Accepted for API
        parity with ``to_geotiff``.
    tile_size : int
        [experimental] Tile size in pixels (default 256). Must be a
        positive multiple of 16; this is a TIFF 6 spec requirement on
        TileWidth and TileLength for broad reader compatibility.
        ``_write_geotiff_gpu`` is always tiled, so the check fires for
        every call.
    predictor : bool or int
        [experimental] TIFF predictor. ``False``/``0``/``1`` -> none,
        ``True``/``2`` -> horizontal differencing, ``3`` ->
        floating-point predictor (float dtypes only).
    cog : bool
        [experimental] Write as Cloud Optimized GeoTIFF with
        overviews.
    overview_levels : list[int] or None
        [experimental] Overview decimation factors relative to full
        resolution.
        Each entry must be a power-of-two integer >= 2, and the list
        must be strictly increasing (e.g. ``[2, 4, 8]`` writes
        overviews at 1/2, 1/4 and 1/8 of the full resolution).
        Invalid values raise ``ValueError``. Only used when ``cog=True``.
        If None and ``cog=True``, auto-generates ``[2, 4, 8, ...]`` by
        halving until the smallest overview fits in a single tile.
    overview_resampling : str
        [experimental] Resampling method for overviews: 'mean'
        (default), 'nearest', 'min', 'max', 'median', 'mode', or
        'cubic'. ``mode`` and ``cubic`` fall back to the CPU
        implementation in ``xrspatial.geotiff._writer`` so the GPU
        writer produces the same overview bytes as the CPU writer.
    bigtiff : bool or None
        [experimental] Force BigTIFF (64-bit offsets). None
        auto-promotes when the estimated file size would exceed the
        classic-TIFF 4 GB limit.
    streaming_buffer_bytes : int
        [experimental] Soft cap on the bytes materialised on device
        per dask compute call when the input is dask-backed and
        ``cog=False`` (issue #3166). Default matches ``to_geotiff``
        (256 MB). The writer computes and compresses one tile-row band
        at a time, grouping consecutive tile-rows by the source
        chunk-row span under this budget the same way the CPU
        streaming writer does; the floor is one full-width tile-row,
        so the cap is soft, and the tile-extraction buffer adds about
        one band-sized allocation on top. The cap bounds device
        memory only: the compressed tile bytes accumulate in host RAM
        until the file is assembled and written, matching the
        non-streaming GPU write. In-memory (cupy / numpy)
        input ignores the kwarg. ``cog=True`` still materialises
        dask input in full on device (overview generation needs the
        whole array) and emits a ``GeoTIFFFallbackWarning``.
    max_z_error : float
        [internal-only] Per-pixel error budget for LERC compression.
        The GPU writer does not implement LERC (nvCOMP has no LERC
        backend), so any non-zero value raises ``ValueError``.
        Accepted at the signature level for API parity with
        ``to_geotiff``.
    photometric : str or int
        [experimental] Photometric interpretation for the TIFF
        Photometric tag (262). See :func:`to_geotiff` for the full set
        of accepted values; the GPU writer forwards this kwarg
        unchanged. Default ``'auto'`` writes MinIsBlack for any band
        count, so a 4-band raster is not silently tagged as RGB+alpha.
    allow_experimental_codecs : bool
        [experimental] Opt in to the Tier 3 experimental codecs
        ``'lerc'``,
        ``'jpeg2000'`` / ``'j2k'``, and ``'lz4'`` (default ``False``).
        Mirrors the same kwarg on ``to_geotiff`` so the two writers
        expose a consistent surface; the GPU dispatch path through
        ``to_geotiff`` forwards the kwarg unchanged. Setting
        ``compression=`` to one of those codecs without this flag
        raises ``ValueError`` whose message names the flag. With the
        flag set, the write proceeds and a ``GeoTIFFFallbackWarning``
        is emitted once per call. Does NOT cover ``compression='jpeg'``:
        the internal-only JPEG path keeps its own dedicated
        ``allow_internal_only_jpeg`` flag.
    allow_internal_only_jpeg : bool
        [internal-only] Opt in to the ``compression='jpeg'`` encode
        path (default ``False``). The encoder emits self-contained
        JFIF tiles without the TIFF JPEGTables tag (347); the file
        decodes through this library's reader but not through libtiff,
        GDAL, or rasterio. With the flag set, the write proceeds and a
        ``GeoTIFFFallbackWarning`` is emitted at call time. Without
        the flag, ``compression='jpeg'`` raises ``ValueError`` for
        parity with ``to_geotiff``.
    allow_unparseable_crs : bool
        [experimental] Opt in to writing an unvalidatable CRS string
        into ``GTCitationGeoKey`` (default ``False``). See
        :func:`to_geotiff` for the full description; the GPU writer
        applies the same fail-closed default.
    drop_rotation : bool, default False
        [experimental] Opt in to writing a DataArray that carries
        ``attrs['rotated_affine']``. Mirrors the same kwarg on
        ``to_geotiff`` so the two writers share one gate. Default
        ``False`` refuses the write with ``ValueError``; the GPU
        writer does not emit a ``ModelTransformationTag`` either, so
        the silent-loss surface is identical on both backends.
    pack : bool, default False
        [experimental] No-op on the GPU writer: it exists for signature
        parity with ``to_geotiff``, which applies the ``pack`` re-pack
        transform before dispatching here. See ``to_geotiff`` for the
        behaviour.

    Returns
    -------
    str or binary file-like
        The ``path`` argument (a string for filesystem paths, the
        file-like object for BytesIO destinations). Returning the path
        mirrors ``to_geotiff`` and ``_build_vrt`` so callers can handle
        the three writers uniformly.

    Raises
    ------
    ValueError
        If ``data`` is a 3D DataArray whose ``dims`` is not
        ``(band, y, x)`` or ``(y, x, band)`` (accepting band-name
        aliases ``bands`` / ``channel`` and spatial-name aliases
        ``lat`` / ``lon`` / ``row`` / ``col``). A leading non-band
        dim such as ``time`` would otherwise silently round-trip with
        the leading axis treated as ``y``.
    """
    if not tiled:
        raise ValueError(
            "_write_geotiff_gpu requires tiled=True. nvCOMP batch "
            "compression is tile-based; the strip layout is not "
            "implemented on the GPU path. Use to_geotiff(..., gpu=False, "
            "tiled=False) for strip output on CPU.")
    # JPEG-in-TIFF parity with to_geotiff. The GPU encode path writes
    # self-contained JFIF tiles without the TIFF JPEGTables
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
            "_write_geotiff_gpu(compression='jpeg', "
            "allow_internal_only_jpeg=True) writes JFIF tiles without "
            "the TIFF JPEGTables tag (347); the file decodes through "
            "xrspatial but may fail in libtiff, GDAL, or rasterio. "
            "See issue #1845.",
            GeoTIFFFallbackWarning,
            stacklevel=2,
        )
    # Tier 3 experimental-codec gate. Lerc, jpeg2000 / j2k, and lz4
    # require ``allow_experimental_codecs=True``; the GPU
    # writer mirrors the same gate ``to_geotiff`` enforces so the two
    # entry points agree.  The GPU dispatch path through
    # ``to_geotiff(gpu=True, compression='lerc', ...)`` forwards the
    # kwarg, so the warning is emitted once at the CPU dispatcher and a
    # second time here when the GPU writer re-runs the same check; that
    # mirrors ``allow_internal_only_jpeg`` (which already double-warns
    # under that codepath) and keeps the explicit GPU entry point usable
    # standalone.
    if isinstance(compression, str):
        _gpu_codec = compression.lower()
        if (_gpu_codec in _EXPERIMENTAL_CODECS
                and not allow_experimental_codecs):
            raise ValueError(
                f"compression={compression!r} is experimental: cross-backend "
                "numerical parity is not claimed and reader support across "
                "GDAL versions is uneven. Pass allow_experimental_codecs=True "
                "to opt in, or use 'deflate', 'zstd', or 'lzw' for a "
                "stable lossless codec (issue #2137).")
        # Mirror to_geotiff's compression_level range gate so direct
        # callers of _write_geotiff_gpu get the same rejection the
        # dispatcher applies upstream (#3167). Runs before the
        # experimental-codec opt-in warning below so a rejected call
        # raises without warning first, matching to_geotiff's order.
        # to_geotiff also rejects non-int / bool compression_level
        # upstream (#3321), so the range comparison here is type-safe on
        # the normal dispatch path.
        if compression_level is not None:
            _level_range = _LEVEL_RANGES.get(_gpu_codec)
            if _level_range is not None:
                _lo, _hi = _level_range
                if not (_lo <= compression_level <= _hi):
                    raise ValueError(
                        f"compression_level={compression_level} out of "
                        f"range for {compression} "
                        f"(valid: {_lo}-{_hi})")
        if (_gpu_codec in _EXPERIMENTAL_CODECS
                and allow_experimental_codecs):
            warnings.warn(
                f"_write_geotiff_gpu(compression={compression!r}, "
                "allow_experimental_codecs=True): experimental codec, "
                "GPU encode path is not byte-identical to the CPU writer "
                "(different backend libraries). See issue #2137.",
                GeoTIFFFallbackWarning,
                stacklevel=2,
            )
    # MinIsWhite pre-inversion runs in the eager CPU writer.
    # The GPU writer assembles tile bytes directly on device; threading
    # the pixel + nodata-sentinel transform through that pipeline is out
    # of scope for the round-trip fix. Refuse the combination so callers
    # do not silently get inverted on-disk values. Move the array to the
    # CPU and call the eager ``write`` path for MinIsWhite output.
    from .._writer import _resolve_photometric as _resolve_photo_gpu
    _gpu_samples_hint = _compute_gpu_samples_hint(data)
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
    # _write_geotiff_gpu is always tiled, so validate tile_size here and
    # keep parity with the public to_geotiff entry point.
    _validate_tile_size_arg(tile_size)
    _validate_nodata_arg(nodata)

    # Refuse to silently drop ``attrs['rotated_affine']``. Mirror the
    # gate ``to_geotiff`` runs upstream so direct callers of
    # ``_write_geotiff_gpu`` get the same rejection.
    _drop_rotation_attrs = getattr(data, 'attrs', None) or {}
    _validate_no_rotated_affine(
        _drop_rotation_attrs,
        drop_rotation=drop_rotation,
        entry_point="_write_geotiff_gpu",
    )

    # Reject empty spatial shapes. ``_write_geotiff_gpu`` is a public
    # entry point and direct callers (with cupy.ndarray or raw
    # numpy) do not flow through ``to_geotiff``'s guard, so check here
    # before any GPU work starts.
    _validate_writer_spatial_shape(
        getattr(data, 'shape', None),
        getattr(data, 'dims', None),
        entry_point="_write_geotiff_gpu",
    )

    # Reject ``gdal_metadata_xml`` / ``extra_tags`` pass-through writes
    # unless the caller opted in via ``allow_experimental_codecs=True``.
    # Mirrors ``to_geotiff`` so the two writers expose the same surface.
    from .._attrs import _validate_write_rich_tag_optin
    _attrs_for_optin = getattr(data, 'attrs', None) or {}
    _validate_write_rich_tag_optin(
        _attrs_for_optin,
        allow_experimental_codecs=allow_experimental_codecs,
        entry_point="_write_geotiff_gpu",
    )

    # Ambiguous-metadata checks; mirrors ``to_geotiff`` so the GPU
    # writer enforces the same crs/crs_wkt consistency rule. Alias
    # resolution keeps the validator consistent across alias-named
    # coord arrays (lat/lon, latitude/longitude, row/col).
    _attrs = getattr(data, 'attrs', None) or {}
    _coord_y, _coord_x = _resolve_spatial_coords(data)
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
    # so callers cannot bypass it. Non-cog file-like writes remain
    # supported on this entry point.
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
    # ``pack`` is a no-op here: ``to_geotiff`` applies the
    # re-pack transform before dispatching to this writer, so the kwarg
    # only needs to exist for signature parity (#3064).
    del pack
    try:
        import cupy
    except ImportError:
        raise ImportError("cupy is required for GPU writes")

    from .._gpu_decode import gpu_compress_tiles, make_overview_gpu
    from .._writer import _assemble_tiff, _compression_tag, _write_bytes, normalize_predictor

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

    # ``numbers.Integral`` covers numpy integer scalars (``np.int32``,
    # ``np.int64``) so they hit the EPSG branch instead of falling
    # through to ``epsg=None``. Validator already rejects bool.
    _validate_crs_arg(crs)
    if isinstance(crs, numbers.Integral):
        epsg = int(crs)
    elif isinstance(crs, str):
        epsg = _wkt_to_epsg(crs)
        if epsg is None:
            wkt_fallback = crs

    # ``attrs['masked_nodata']`` records whether the read side promoted
    # the sentinel to NaN. Default True preserves the NaN->sentinel
    # rewrite for external DataArrays and bare cupy / numpy positional
    # arrays that have no attrs to read from.
    restore_sentinel = True
    if isinstance(data, xr.DataArray):
        restore_sentinel = _should_restore_nan_sentinel(data.attrs)
        arr = data.data
        # Dask arrays: only cog=True materialises here (overview
        # generation needs the full-resolution array on device).
        # Otherwise the array stays lazy and the streaming loop below
        # computes one tile-row band at a time (issue #3166).
        if hasattr(arr, 'compute') and cog:
            _warn_dask_materialised()
            arr = arr.compute()
        # Now arr should be CuPy, numpy, or a still-lazy dask array
        if hasattr(arr, 'get') or hasattr(arr, 'compute'):
            pass  # CuPy (on GPU) or dask (uploaded per band below)
        else:
            arr = cupy.asarray(np.asarray(arr))  # numpy -> GPU

        # Reject ambiguous 3D layouts. Mirrors the gate in
        # ``to_geotiff``: a leading non-band dim like ``time`` would
        # otherwise round-trip with the leading axis silently treated
        # as ``y``.
        if arr.ndim == 3:
            _validate_3d_writer_dims(data.dims)
        # Handle band-first dimension order (band, y, x) -> (y, x, band).
        # rioxarray and CF-style multi-band rasters land here; without
        # this remap the writer treats arr.shape[2] as the band axis and
        # produces a transposed file. The CPU writer does the same remap
        # at the matching step in to_geotiff(). Dask input remaps
        # lazily; the per-band ``ascontiguousarray`` in the streaming
        # loop replaces the eager one here.
        if arr.ndim == 3 and data.dims[0] in _BAND_DIM_NAMES:
            if hasattr(arr, 'compute'):
                import dask.array as _da
                arr = _da.moveaxis(arr, 0, -1)
            else:
                arr = cupy.ascontiguousarray(cupy.moveaxis(arr, 0, -1))

        # Resolve via the centralised resolver. Same precedence as the
        # CPU writer: prefer attrs['transform'] (bit-stable on
        # round-trip) over the coord-derived transform, which drifts on
        # fractional pixel sizes.
        geo_transform = _resolve_georef(data).transform
        # Match the CPU writer's fail-closed guard: an array with spatial
        # coords but no derivable transform (e.g. 1x1 without
        # ``attrs['transform']``) must not silently round-trip as a
        # non-georeferenced TIFF.
        _require_transform_for_georeferenced(data, geo_transform)
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
                # Same gate as the kwarg path: reject bool / non-int
                # types and confirm the EPSG resolves before writing it
                # to disk. Without this, ``attrs={'crs': True}`` round-
                # trips as EPSG=1.
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
        # Mirror the CPU writer's pass-through of GDAL metadata, the
        # extra_tags list, the friendly image_description / extra_samples
        # / colormap synthesis, and the resolution tags. Without these,
        # a GPU write -> CPU read round-trip silently drops every rich
        # tag.
        _rich = _extract_rich_tags(data.attrs)
        raster_type = _rich['raster_type']
        gdal_meta_xml = _rich['gdal_metadata_xml']
        extra_tags_list = _rich['extra_tags']
        x_res = _rich['x_resolution']
        y_res = _rich['y_resolution']
        res_unit = _rich['resolution_unit']
    else:
        if hasattr(data, 'compute') and cog:
            # Overview generation needs the full array on device.
            _warn_dask_materialised()
            data = data.compute()  # Dask -> CuPy or numpy
        if hasattr(data, 'compute'):
            arr = data  # still-lazy dask; streamed per band below
        elif hasattr(data, 'device'):
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

    # Dask input that survived the cog gates above streams below: the
    # full-resolution part is compressed one tile-row band at a time
    # instead of materialising the whole array on device (issue #3166).
    stream_dask = hasattr(arr, 'compute')

    # Mirror the CPU writer's NaN-to-sentinel substitution.
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
    # Shared NodataLifecycle gate for NaN->sentinel restore. The
    # streaming path applies the same rewrite per tile-row band inside
    # ``_gpu_stream_compress_to_part`` below, after each band is
    # computed onto the device.
    rewrite_nodata = _NL(declared=nodata, dtype_in=np_dtype).writer_restore_sentinel(
        buffer_dtype=np_dtype,
        restore_sentinel=restore_sentinel,
    )
    if rewrite_nodata and not stream_dask:
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
            comp_tag, pred_val, np_dtype, spp,
            compression_level=compression_level)
        rel_off = []
        bc = []
        off = 0
        for tile in compressed:
            rel_off.append(off)
            bc.append(len(tile))
            off += len(tile)
        stub = np.empty((1, 1, spp) if spp > 1 else (1, 1), dtype=np_dtype)
        return (stub, w, h, rel_off, bc, compressed)

    def _gpu_stream_compress_to_part(lazy_arr, w, h, spp):
        """Stream-compress a dask array into the same part tuple as
        ``_gpu_compress_to_part`` without materialising it in full.

        Computes one tile-row band at a time, grouping consecutive
        tile-rows by the source chunk-row span under
        ``streaming_buffer_bytes`` via the CPU streaming writer's
        ``_stream_row_bands`` helper (so a source chunked taller than
        the tile is not recomputed once per tile-row, issue #3117 /
        #3007). The floor is one full-width tile-row, so the budget is
        a soft cap.

        Per-band compression is byte-identical to one whole-image
        ``gpu_compress_tiles`` call: bands are tile-row aligned, tiles
        are independent in the TIFF layout, and the tile-extraction
        kernel zero-pads edge tiles per band exactly as it does for
        the full image. The NaN->sentinel rewrite runs per band after
        the compute, mirroring the full-array rewrite above.
        """
        from .._writer import _stream_row_bands
        bytes_per_src_row = max(1, w * np_dtype.itemsize * spp)
        max_src_rows = max(1, streaming_buffer_bytes // bytes_per_src_row)
        row_chunks = getattr(lazy_arr, 'chunks', None)
        try:
            row_bands = _stream_row_bands(
                row_chunks[0] if row_chunks else None,
                tile_size, h, max_src_rows)
        except (TypeError, ValueError):
            row_bands = _stream_row_bands(None, tile_size, h, 0)
        compressed = []
        for r0, r1 in row_bands:
            band = lazy_arr[r0:r1].compute()
            if hasattr(band, 'get'):
                # CuPy band. ``da.moveaxis`` chunks may compute to
                # transposed views; the tile-extraction kernel needs a
                # contiguous device buffer (no-op when already
                # contiguous).
                band = cupy.ascontiguousarray(band)
            else:
                band = cupy.asarray(np.asarray(band))  # numpy -> GPU
            if rewrite_nodata:
                nan_mask = cupy.isnan(band)
                if bool(nan_mask.any()):
                    band = band.copy()
                    band[nan_mask] = np_dtype.type(nodata)
            compressed.extend(gpu_compress_tiles(
                band, tile_size, tile_size, w, r1 - r0,
                comp_tag, pred_val, np_dtype, spp,
                compression_level=compression_level))
        rel_off = []
        bc = []
        off = 0
        for tile in compressed:
            rel_off.append(off)
            bc.append(len(tile))
            off += len(tile)
        stub = np.empty((1, 1, spp) if spp > 1 else (1, 1), dtype=np_dtype)
        return (stub, w, h, rel_off, bc, compressed)

    # Full resolution. Dask input streams (cog=True materialised it
    # above, so ``stream_dask`` is False on the overview path).
    if stream_dask:
        parts = [_gpu_stream_compress_to_part(arr, width, height, samples)]
    else:
        parts = [_gpu_compress_to_part(arr, width, height, samples)]

    # Overview generation mirrors the CPU writer's 8-level cap.
    if cog:
        if overview_levels is None:
            from .._overview import _MAX_OVERVIEW_LEVELS

            # Auto-generated lists hold actual decimation factors (2,
            # 4, 8, ...) so the loop below treats auto-generated and
            # user-supplied lists identically.
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
            # length mattered.
            from .._overview import _validate_overview_levels
            overview_levels = _validate_overview_levels(
                overview_levels, height=height, width=width)

        # Pass ``nodata`` so the GPU reducer masks the sentinel back to
        # NaN before averaging. Without this, the NaN->sentinel rewrite
        # done above on ``arr`` leaks the sentinel into the overview
        # reduction and poisons the pyramid. Rewrite any all-sentinel
        # cell NaN back to the sentinel after each level
        # so the on-disk overview tiles still carry the sentinel value
        # external readers expect.
        #
        # The sentinel rewrite uses ``cupy.putmask`` rather than
        # ``current.copy(); current[nan_mask] = sentinel`` because
        # ``make_overview_gpu`` always returns a freshly allocated
        # buffer (``_block_reduce_2d_gpu`` ends in ``cupy.nan*`` /
        # ``cupy.around(...).astype(...)`` / ``cropped[::2, ::2].copy()``,
        # and the 3-D path ends in ``cupy.stack``) so nothing else
        # aliases the buffer between the return and the rewrite.
        # Dropping the redundant ``copy()`` skips one chunk-sized GPU
        # allocation per overview level. Mirrors the in-place sentinel
        # rewrite ``_apply_nodata_mask_gpu`` uses.
        current = arr
        cumulative_factor = 1
        # ``make_overview_gpu`` preserves dtype, so the sentinel cast is
        # loop-invariant. Hoist it (and the float/finite gate) out of the
        # inner ``while`` to skip redundant per-level scalar work.
        # The lifecycle's writer_restore_sentinel mirrors the gate used
        # for the full-resolution rewrite above so the overview loop
        # stays in sync with the base rewrite.
        rewrite_nodata = _NL(
            declared=nodata, dtype_in=np_dtype,
        ).writer_restore_sentinel(
            buffer_dtype=np_dtype,
            restore_sentinel=restore_sentinel,
        )
        sentinel_scalar = (
            np_dtype.type(nodata) if rewrite_nodata else None
        )
        for target_factor in overview_levels:
            # Halve repeatedly until the cumulative decimation matches
            # the requested factor. Validation has already established
            # that ``target_factor`` is a power of two and strictly
            # greater than ``cumulative_factor``.
            while cumulative_factor < target_factor:
                current = make_overview_gpu(current, method=overview_resampling,
                                            nodata=nodata)
                cumulative_factor *= 2
                if rewrite_nodata:
                    nan_mask = cupy.isnan(current)
                    if bool(nan_mask.any().item()):
                        cupy.putmask(current, nan_mask, sentinel_scalar)
            oh, ow = current.shape[:2]
            parts.append(_gpu_compress_to_part(current, ow, oh, samples))

    # Refuse to write an unvalidatable CRS string into GTCitationGeoKey
    # unless the caller opts in.
    if epsg is None:
        _validate_crs_fallback(wkt_fallback, allow_unparseable_crs)
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
    return path
