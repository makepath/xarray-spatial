"""GPU-only helpers consumed by ``read_geotiff_gpu`` and the GPU write path.

Lazy-imports cupy at call time so the module imports cleanly on
numpy-only environments. Every public entry point that touches GPU
data routes through one of these helpers, so a single canonical copy
keeps nodata-mask, orientation-flip, and window/band-slice semantics
in lockstep with the CPU reader.

Extracted in step 6 of issue #1813 so the next PR (step 7) can move
``read_geotiff_gpu`` itself into ``_backends/gpu.py`` as a near-
mechanical lift.
"""
from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from .._coords import (
    coords_from_geo_info as _coords_from_geo_info,
    geo_to_coords as _geo_to_coords,
)
from .._geotags import GeoTransform, RASTER_PIXEL_IS_POINT
from .._runtime import _geotiff_strict_mode


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


def _apply_nodata_mask_gpu_with_presence(arr_gpu, nodata):
    """Same as :func:`_apply_nodata_mask_gpu` but also reports presence.

    Returns ``(arr_gpu, pixels_present)`` where ``pixels_present`` is a
    bool indicating whether any pixel in ``arr_gpu`` matched the
    declared sentinel before masking. Lets the eager GPU read path
    surface ``attrs['nodata_pixels_present']`` (issue #2135) without
    rescanning the buffer.

    ``pixels_present`` is ``False`` when ``nodata`` is ``None`` (no
    declared sentinel), when the sentinel is out of range for an
    integer buffer, or when the sentinel is fractional / non-finite
    and cannot match an integer pixel -- the same fast-paths the
    masking branch takes. On a NaN-only sentinel against a float
    buffer, ``pixels_present`` is whether the buffer already contains
    NaN.
    """
    import cupy

    if nodata is None:
        return arr_gpu, False
    arr_dtype = np.dtype(str(arr_gpu.dtype))
    if arr_dtype.kind == 'f':
        if np.isnan(nodata):
            # NaN-only sentinel: no replacement happens, but report
            # whether any NaN already lives in the buffer so the attr
            # stays informative.
            return arr_gpu, bool(cupy.isnan(arr_gpu).any().item())
        sentinel = arr_dtype.type(nodata)
        mask = arr_gpu == sentinel
        present = bool(mask.any().item())
        if present:
            cupy.putmask(arr_gpu, mask, arr_dtype.type('nan'))
        return arr_gpu, present
    if arr_dtype.kind in ('u', 'i'):
        if not (np.isfinite(nodata) and float(nodata).is_integer()):
            return arr_gpu, False
        nodata_int = int(nodata)
        info = np.iinfo(arr_dtype)
        if not (info.min <= nodata_int <= info.max):
            return arr_gpu, False
        sentinel = arr_dtype.type(nodata_int)
        mask = arr_gpu == sentinel
        present = bool(mask.any().item())
        if present:
            arr_gpu = arr_gpu.astype(cupy.float64)
            cupy.putmask(arr_gpu, mask, cupy.float64('nan'))
        return arr_gpu, present
    return arr_gpu, False


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

    The sentinel-replacement step writes NaN into the existing buffer with
    ``cupy.putmask`` rather than allocating a fresh output via
    ``cupy.where``; every call site passes a freshly decoded GPU buffer
    that no caller-visible state aliases, so the mutation is safe and
    drops one chunk-sized device allocation per call (#1934).
    """
    import cupy

    if nodata is None:
        return arr_gpu
    arr_dtype = np.dtype(str(arr_gpu.dtype))
    if arr_dtype.kind == 'f':
        if not np.isnan(nodata):
            sentinel = arr_dtype.type(nodata)
            cupy.putmask(arr_gpu, arr_gpu == sentinel,
                         arr_dtype.type('nan'))
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
            # ``astype`` allocates the float64 buffer; write NaN into it
            # in place instead of running it through another ``cupy.where``
            # that would allocate again (#1934).
            arr_gpu = arr_gpu.astype(cupy.float64)
            cupy.putmask(arr_gpu, mask, cupy.float64('nan'))
        return arr_gpu
    return arr_gpu


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
    from .._gpu_decode import gpu_decode_tiles, gpu_decode_tiles_from_file

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
