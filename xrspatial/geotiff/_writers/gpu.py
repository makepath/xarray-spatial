"""GPU writer entry point.

Step 10 of issue #1813. Holds ``write_geotiff_gpu``, which compresses
tiles on the device via nvCOMP (when available) and falls back to the
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

from .._attrs import (
    _EXPERIMENTAL_CODECS,
    _extract_rich_tags,
    _resolve_nodata_attr,
    _should_restore_nan_sentinel,
)
from .._coords import (
    _BAND_DIM_NAMES,
    coords_to_transform as _coords_to_transform,
    require_transform_for_georeferenced as _require_transform_for_georeferenced,
    transform_from_attr as _transform_from_attr,
)
from .._crs import _validate_crs_arg, _validate_crs_fallback, _wkt_to_epsg
from .._runtime import GeoTIFFFallbackWarning
from .._validation import (
    _validate_3d_writer_dims,
    _validate_nodata_arg,
    _validate_tile_size_arg,
    _validate_writer_spatial_shape,
    validate_write_metadata,
)


def _compute_gpu_samples_hint(data) -> int:
    """Return the band count using the same convention the GPU writer's
    band-first -> band-last remap uses (issue #2097).

    The remap below in ``write_geotiff_gpu`` moves bands from
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
                      streaming_buffer_bytes: int = 256 * 1024 * 1024,
                      max_z_error: float = 0.0,
                      photometric: str | int = 'auto',
                      allow_internal_only_jpeg: bool = False,
                      allow_experimental_codecs: bool = False,
                      allow_unparseable_crs: bool = False
                      ) -> str | BinaryIO:
    """Write a CuPy-backed DataArray as a GeoTIFF with GPU compression.

    Tier: Experimental (issue #2137). The GPU writer requires cupy +
    numba CUDA plus optional nvCOMP / nvJPEG / nvJPEG2K libraries for
    codec-specific acceleration; cross-backend numerical parity with
    ``to_geotiff`` is tested for the Tier 1 codec set only. Tier 3
    codecs (``'lerc'``, ``'jpeg2000'`` / ``'j2k'``, ``'lz4'``) require
    the explicit ``allow_experimental_codecs=True`` opt-in; the
    internal-only ``'jpeg'`` codec keeps its own dedicated
    ``allow_internal_only_jpeg`` flag. See
    :data:`xrspatial.geotiff.SUPPORTED_FEATURES` for the full tier
    map.

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
    crs : int, numpy.integer, str, or None
        EPSG code (int or numpy integer scalar) or WKT string. EPSG
        codes are strongly preferred for interop; the WKT-only path
        emits a user-defined CRS (32767) with the WKT stored in
        ``GTCitationGeoKey``, which many non-libgeotiff readers
        ignore. A ``UserWarning`` is emitted when the WKT-only path
        is taken. See issue #1768.
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
    streaming_buffer_bytes : int
        Accepted for API parity with ``to_geotiff``. The GPU writer
        materialises the entire array on device and has no streaming
        concept, so this kwarg is a no-op. Default matches
        ``to_geotiff`` (256 MB) so callers passing the same kwargs to
        either entry point see the same default and the same type.
    max_z_error : float
        Per-pixel error budget for LERC compression. The GPU writer
        does not implement LERC (nvCOMP has no LERC backend), so any
        non-zero value raises ``ValueError``. Accepted at the signature
        level for API parity with ``to_geotiff``.
    photometric : str or int
        Photometric interpretation for the TIFF Photometric tag (262).
        See :func:`to_geotiff` for the full set of accepted values; the
        GPU writer forwards this kwarg unchanged. Default ``'auto'``
        writes MinIsBlack for any band count, so a 4-band raster is
        not silently tagged as RGB+alpha (issue #1769).
    allow_experimental_codecs : bool
        Opt in to the Tier 3 experimental codecs ``'lerc'``,
        ``'jpeg2000'`` / ``'j2k'``, and ``'lz4'`` (default ``False``).
        Mirrors the same kwarg on ``to_geotiff`` so the two writers
        expose a consistent surface; the GPU dispatch path through
        ``to_geotiff`` forwards the kwarg unchanged. Setting
        ``compression=`` to one of those codecs without this flag
        raises ``ValueError`` whose message names the flag. With the
        flag set, the write proceeds and a ``GeoTIFFFallbackWarning``
        is emitted once per call. Does NOT cover ``compression='jpeg'``:
        the internal-only JPEG path keeps its own dedicated
        ``allow_internal_only_jpeg`` flag. See issue #2137.
    allow_internal_only_jpeg : bool
        Opt in to the experimental ``compression='jpeg'`` encode path
        (default ``False``). The encoder emits self-contained JFIF
        tiles without the TIFF JPEGTables tag (347); the file decodes
        through this library's reader but not through libtiff, GDAL,
        or rasterio. With the flag set, the write proceeds and a
        ``GeoTIFFFallbackWarning`` is emitted at call time. Without
        the flag, ``compression='jpeg'`` raises ``ValueError`` for
        parity with ``to_geotiff``. See issue #1845.
    allow_unparseable_crs : bool
        Opt in to writing an unvalidatable CRS string into
        ``GTCitationGeoKey`` (default ``False``). See
        :func:`to_geotiff` for the full description; the GPU writer
        applies the same fail-closed default. See issue #1929.

    Returns
    -------
    str or binary file-like
        The ``path`` argument (a string for filesystem paths, the
        file-like object for BytesIO destinations). Returning the path
        mirrors ``to_geotiff`` and ``write_vrt`` so callers can handle
        the three writers uniformly. See issue #1938.

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
    # Tier 3 experimental-codec gate (issue #2137). Lerc, jpeg2000 /
    # j2k, and lz4 require ``allow_experimental_codecs=True``; the GPU
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
        if (_gpu_codec in _EXPERIMENTAL_CODECS
                and allow_experimental_codecs):
            warnings.warn(
                f"write_geotiff_gpu(compression={compression!r}, "
                "allow_experimental_codecs=True): experimental codec, "
                "GPU encode path is not byte-identical to the CPU writer "
                "(different backend libraries). See issue #2137.",
                GeoTIFFFallbackWarning,
                stacklevel=2,
            )
    # MinIsWhite pre-inversion (issue #1836) runs in the eager CPU writer.
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
    # write_geotiff_gpu is always tiled, so validate tile_size here and
    # keep parity with the public to_geotiff entry point.
    _validate_tile_size_arg(tile_size)
    _validate_nodata_arg(nodata)

    # Issue #2075: reject empty spatial shapes. ``write_geotiff_gpu`` is
    # a public entry point and direct callers (with cupy.ndarray or raw
    # numpy) do not flow through ``to_geotiff``'s guard, so check here
    # before any GPU work starts.
    _validate_writer_spatial_shape(
        getattr(data, 'shape', None),
        getattr(data, 'dims', None),
        entry_point="write_geotiff_gpu",
    )

    # Issue #1987 ambiguous-metadata checks; mirrors ``to_geotiff`` so the
    # GPU writer enforces the same crs/crs_wkt consistency rule.
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

    from .._gpu_decode import gpu_compress_tiles, make_overview_gpu
    from .._writer import (
        _compression_tag, _assemble_tiff, _write_bytes,
        normalize_predictor,
    )

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

    # Issue #1988: ``attrs['masked_nodata']`` records whether the read
    # side promoted the sentinel to NaN. Default True preserves the
    # pre-#1988 NaN->sentinel rewrite for external DataArrays and bare
    # cupy / numpy positional arrays that have no attrs to read from.
    restore_sentinel = True
    if isinstance(data, xr.DataArray):
        restore_sentinel = _should_restore_nan_sentinel(data.attrs)
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
        # Match the CPU writer's fail-closed guard: an array with spatial
        # coords but no derivable transform (e.g. 1x1 without
        # ``attrs['transform']``) must not silently round-trip as a
        # non-georeferenced TIFF (#1945).
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
            and not np.isnan(float(nodata))
            and restore_sentinel):
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
            from .._writer import _MAX_OVERVIEW_LEVELS
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
            from .._writer import _validate_overview_levels
            overview_levels = _validate_overview_levels(
                overview_levels, height=height, width=width)

        # Pass ``nodata`` so the GPU reducer masks the sentinel back to
        # NaN before averaging. Without this, the NaN->sentinel rewrite
        # done above on ``arr`` leaks the sentinel into the overview
        # reduction and poisons the pyramid (issue #1613). Rewrite any
        # all-sentinel cell NaN back to the sentinel after each level
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
        # rewrite ``_apply_nodata_mask_gpu`` adopted in #1934. See
        # issue #1948.
        current = arr
        cumulative_factor = 1
        # ``make_overview_gpu`` preserves dtype, so the sentinel cast is
        # loop-invariant. Hoist it (and the float/finite gate) out of the
        # inner ``while`` to skip redundant per-level scalar work.
        rewrite_nodata = (
            nodata is not None
            and np_dtype.kind == 'f'
            and not np.isnan(float(nodata))
            and restore_sentinel
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

    # Issue #1929: refuse to write an unvalidatable CRS string into
    # GTCitationGeoKey unless the caller opts in.
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


