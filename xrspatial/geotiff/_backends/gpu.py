"""GPU read backend: ``read_geotiff_gpu`` and its private helpers.

Step 7 of issue #1813. With the leaf helpers in ``_backends/_gpu_helpers``
already extracted (#1884), validators in ``_validation`` (#1882), attrs
helpers in ``_attrs`` (#1883), and sentinels in ``_runtime`` (#1880),
moving the entry-point body is a near-mechanical lift; the body stays
unchanged except for adjusted relative imports.

``_read_geotiff_gpu_chunked`` calls into ``read_geotiff_dask`` for its
CPU-decode fallback path. That import was originally lazy because
``read_geotiff_dask`` still lived in ``__init__.py``; once it moved
to a sibling module in #1886, the import was promoted to a static
top-of-module ``from .dask import read_geotiff_dask``.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import xarray as xr

from .._attrs import _populate_attrs_from_geo_info
from .._coords import (
    coords_from_geo_info as _coords_from_geo_info,
)
from .._reader import read_to_array as _read_to_array
from .._runtime import (
    _GPU_DEPRECATED_SENTINEL,
    _ON_GPU_FAILURE_SENTINEL,
    _geotiff_strict_mode,
)
from .._validation import _validate_chunks_arg, _validate_dtype_cast
from ._gpu_helpers import (
    _apply_nodata_mask_gpu,
    _apply_orientation_geo_info,
    _apply_orientation_gpu,
    _gpu_apply_window_band,
    _gpu_decode_single_band_tiles,
)
from .dask import read_geotiff_dask


def _preflight_cuda_runtime(cupy) -> None:
    """Verify the CUDA runtime is usable before the GPU pipeline runs.

    ``cupy`` can import successfully on a machine whose driver is older
    than the build expects, was uninstalled, or belongs to a suspended
    VM. Without this check the failure surfaces as a
    ``cudaErrorInsufficientDriver`` from a deep ``cupy.asarray(...)``
    call in the CPU-fallback path (issue #1903). Raises a clean
    ``RuntimeError`` that names the underlying CUDA error so the user
    can fix the driver, switch CuPy builds, or pass ``gpu=False``.
    """
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as e:
        raise RuntimeError(
            f"read_geotiff_gpu: CUDA runtime is not usable "
            f"({type(e).__name__}: {e}). Check the GPU driver matches "
            f"the installed cupy build, or pass gpu=False."
        ) from e
    if device_count == 0:
        raise RuntimeError(
            "read_geotiff_gpu: cupy reports 0 CUDA devices. Check "
            "the GPU driver and CUDA_VISIBLE_DEVICES, or pass gpu=False."
        )


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
        kwarg does not affect them. The function preflights the CUDA
        runtime via ``cupy.cuda.runtime.getDeviceCount()`` immediately
        after importing cupy and raises ``RuntimeError`` if the driver
        is unusable (#1903); transient errors inside a later
        ``cupy.asarray(...)`` upload (e.g. device OOM) still propagate
        unchanged in both modes.
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

    # Preflight CUDA. ``cupy`` can import on machines whose driver is
    # older than the build expects or whose GPU is offline; the error
    # otherwise surfaces as a low-level CUDA failure from
    # ``cupy.asarray(...)`` deep in the CPU-fallback path (#1903).
    _preflight_cuda_runtime(cupy)

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

    from .._reader import (
        _FileSource, _check_dimensions, MAX_PIXELS_DEFAULT, _coerce_path,
        _resolve_masked_fill,
    )
    from .._compression import COMPRESSION_LERC
    from .._header import (
        parse_header, parse_all_ifds, select_overview_ifd, validate_tile_layout,
    )
    from .._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy
    from .._geotags import extract_geo_info_with_overview_inheritance
    from .._gpu_decode import gpu_decode_tiles

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
        from .._gpu_decode import gpu_decode_tiles_from_file
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
        from .._reader import _miniswhite_inverted_nodata as _miw_inv_nd
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
        from .._reader import _is_fsspec_uri
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
                              r0, c0, r1, c1, masked_fill=None):
    """Decode a window's tile subset disk->GPU.

    Picks just the tiles overlapping ``(r0..r1, c0..c1)`` from the full
    tile sequence, runs them through ``gpu_decode_tiles_from_file``
    (which tries KvikIO GDS first, then a CPU mmap + ``gpu_decode_tiles``
    fallback), and crops the assembled sub-image to the requested window.
    Peak device memory is the sub-tile-grid bounding box, not the full
    raster.

    Called from inside a ``dask.delayed`` per-chunk task, so it runs
    once per chunk and only pulls the tiles that chunk needs from disk.

    ``masked_fill`` is forwarded to both GPU decoders for LERC files
    with a per-pixel valid mask, matching the eager-GPU path (#1896).
    Without it, masked pixels read back as LERC's zero fill instead of
    the file's nodata sentinel.
    """
    from .._gpu_decode import gpu_decode_tiles, gpu_decode_tiles_from_file

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
        masked_fill=masked_fill,
    )

    if arr_gpu is None:
        # ``gpu_decode_tiles_from_file`` returns None when KvikIO is not
        # usable on the host. Open the file via mmap, slice out just the
        # bytes for these tiles, and run the GPU decoder on those.
        from .._reader import _FileSource
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
            masked_fill=masked_fill,
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

    from .._reader import _FileSource, _coerce_path
    from .._header import parse_header, parse_all_ifds, select_overview_ifd
    from .._geotags import extract_geo_info_with_overview_inheritance

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

    from .._reader import (
        _check_dimensions, MAX_PIXELS_DEFAULT, _resolve_masked_fill,
    )
    from .._compression import COMPRESSION_LERC
    from .._header import validate_tile_layout
    from .._dtypes import resolve_bits_per_sample, tiff_dtype_to_numpy

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
        # Reject ``bool`` / ``np.bool_`` up front; ``isinstance(True, int)``
        # is True in Python so ``True < n_bands_out`` would silently read
        # band 1. The eager GPU path and the dask path already reject
        # bools here (#1786); mirror them so the GDS chunked path agrees
        # (#1896).
        if isinstance(band, (bool, np.bool_)):
            raise ValueError(
                f"band must be a non-negative int, got {band!r}")
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

    # LERC tiles can carry a per-pixel valid mask that GDAL writes
    # zero-filled in the data array. Resolve the nodata fill the same way
    # the eager GPU path does so each chunk task restores it inside the
    # GPU decode kernels (#1896). Without this, masked pixels read back
    # at LERC's zero fill on the chunked path while the eager path
    # restores the sentinel.
    masked_fill = (_resolve_masked_fill(ifd.nodata_str, file_dtype)
                   if compression == COMPRESSION_LERC else None)

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

    @dask.delayed
    def _chunk_task(meta, r0, c0, r1, c1):
        all_offsets, all_byte_counts = meta
        arr = _decode_window_gpu_direct(
            source, all_offsets, all_byte_counts,
            tw, th, full_w, compression, predictor,
            file_dtype, samples, byte_order,
            r0, c0, r1, c1,
            masked_fill=masked_fill,
        )
        if nodata is not None:
            arr = _apply_nodata_mask_gpu(arr, nodata)
        if dtype is not None:
            target = np.dtype(dtype)
            _validate_dtype_cast(np.dtype(str(arr.dtype)), target)
            arr = arr.astype(target)
        if band is not None and arr.ndim == 3:
            arr = arr[:, :, band]
        # Cast to the declared graph dtype so every chunk matches the
        # dask array's promised dtype. ``_apply_nodata_mask_gpu`` only
        # promotes integer arrays to float64 when at least one sentinel
        # pixel hits, mirroring the eager GPU semantics; the dask graph
        # always-promotes (matching the CPU dask path's #1597 contract),
        # so chunks with no sentinel hit need an explicit cast here.
        # Without this, a uint16 file + declared nodata sentinel
        # produces a graph that advertises float64 but emits uint16
        # buffers from any chunk that didn't hit the sentinel -- a
        # silent declared/actual dtype mismatch (issue #1909). Skip
        # the cast when the dtype already matches to avoid the
        # ``astype(copy=True)`` per-chunk allocation that #1624 fixed
        # for the CPU dask path.
        if arr.dtype != declared_dtype:
            arr = arr.astype(declared_dtype)
        return arr

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
