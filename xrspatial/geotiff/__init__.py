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

from ._geotags import GeoTransform, RASTER_PIXEL_IS_AREA, RASTER_PIXEL_IS_POINT
from ._reader import UnsafeURLError
# ``read_to_array`` is internal: it is used by ``open_geotiff`` and the
# GPU fallback below but is not in ``__all__`` or the module-level
# Public API docstring. Bind it under a leading-underscore name so it
# does not leak into ``xrspatial.geotiff``'s public namespace. Tests
# and internal callers that genuinely need it can import directly from
# ``xrspatial.geotiff._reader``. See issue #1708.
from ._reader import read_to_array as _read_to_array
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


# Sentinels distinguishing "user passed this kwarg explicitly" from "user
# passed nothing". A plain default of None does not work because None is
# itself a value a caller could supply. ``read_geotiff_gpu`` needs both
# sentinels so it can tell whether the deprecated ``gpu=`` and the new
# ``on_gpu_failure=`` were *each* supplied, and refuse the ambiguous
# both-supplied case regardless of which values were chosen.
# ``open_geotiff`` also uses ``_ON_GPU_FAILURE_SENTINEL`` to distinguish
# "caller never set on_gpu_failure" (default sentinel: skip forwarding so
# the read_geotiff_gpu signature default applies) from "caller set
# on_gpu_failure=<value>" (forward verbatim).
_GPU_DEPRECATED_SENTINEL = object()
_ON_GPU_FAILURE_SENTINEL = object()
# ``write_vrt`` needs to distinguish "user passed crs_wkt= explicitly"
# (deprecation path) from "user passed nothing" (no warning, pick CRS
# from the first source). A plain default of None does not work because
# None is itself a value a caller could supply alongside crs=. See
# issue #1715.
_CRS_WKT_DEPRECATED_SENTINEL = object()

# Names of dims that ``to_geotiff`` / ``write_geotiff_gpu`` treat as the
# non-spatial band axis. Used both to remap ``(band, y, x)`` inputs to
# ``(y, x, band)`` before writing and to skip the band axis when inferring
# a GeoTransform from coords (see ``_coords_to_transform`` and issue #1643).
_BAND_DIM_NAMES = ('band', 'bands', 'channel')


class GeoTIFFFallbackWarning(UserWarning):
    """Warning emitted when a geotiff helper falls back to a slower path.

    Raised in the same call sites that would silently return ``None`` under
    the historic ``except Exception: return None`` pattern. See issue #1662
    for the audit and the ``XRSPATIAL_GEOTIFF_STRICT=1`` env var that
    promotes these warnings to exceptions.
    """


def _geotiff_strict_mode() -> bool:
    """Return True when ``XRSPATIAL_GEOTIFF_STRICT`` is set to a truthy value.

    Strict mode promotes the silent fallbacks audited in issue #1662 into
    raised exceptions. Useful in CI to catch GPU-path or VRT regressions
    that would otherwise hide behind a CPU fallback or a missing tile.
    """
    return os.environ.get(
        'XRSPATIAL_GEOTIFF_STRICT', '').lower() in ('1', 'true', 'yes')


def _gpu_fallback_warning_message(auto_detected: bool, exc: BaseException) -> str:
    """Build the ``to_geotiff`` GPU-to-CPU fallback warning text.

    ``to_geotiff`` reaches the GPU writer two ways: an explicit
    ``gpu=True`` argument, or the auto-detect branch when ``gpu is
    None`` and the data lives on a CuPy device. The wording differs
    because blaming the fallback on a flag the caller never set sends
    them to fix the wrong thing. Both routes share the exception
    payload format so callers can grep ``type(e).__name__: e`` either
    way.
    """
    suffix = f"({type(exc).__name__}: {exc})."
    if auto_detected:
        return (
            "Data is on the GPU and was routed to the GPU writer, but "
            "the writer is unavailable; falling back to CPU and copying "
            "the array to host. " + suffix
        )
    return (
        "to_geotiff(gpu=True) was requested but the GPU writer is "
        "unavailable; falling back to CPU. " + suffix
    )


def _wkt_to_epsg(wkt_or_proj: str) -> int | None:
    """Try to extract an EPSG code from a WKT or PROJ string.

    Returns None if pyproj is not installed or the string can't be parsed.

    Under ``XRSPATIAL_GEOTIFF_STRICT=1`` the underlying exception is
    re-raised instead of being swallowed. In the default mode a
    ``GeoTIFFFallbackWarning`` is emitted so callers can tell
    pyproj-missing from pyproj-broken-input.
    """
    try:
        from pyproj import CRS
        crs = CRS.from_user_input(wkt_or_proj)
        epsg = crs.to_epsg()
        return epsg
    except Exception as e:
        if _geotiff_strict_mode():
            raise
        warnings.warn(
            f"_wkt_to_epsg failed ({type(e).__name__}: {e}); returning None.",
            GeoTIFFFallbackWarning,
            stacklevel=2,
        )
        return None


def _resolve_crs_to_wkt(crs) -> str | None:
    """Normalise a CRS argument to a WKT string for downstream writers.

    Mirrors ``to_geotiff`` / ``write_geotiff_gpu``'s ``crs`` kwarg semantics
    so callers can pass an int EPSG code, a WKT string, or a PROJ string
    interchangeably. Returns the canonical WKT string (or ``None`` if
    ``crs`` is ``None``) for forwarding to ``_vrt.write_vrt``, which only
    speaks WKT.

    Used by ``write_vrt`` (see issue #1715) to close the parameter-naming
    drift versus the eager and GPU writer entry points.

    Parameters
    ----------
    crs : int, str, or None
        EPSG code (int), WKT string, or PROJ string. ``None`` returns
        ``None`` (the downstream writer falls back to the first source
        file's CRS).

    Returns
    -------
    str or None
        Canonical WKT string, or ``None`` if ``crs`` is ``None``.

    Raises
    ------
    TypeError
        If ``crs`` is not an int, str, or ``None``.
    ValueError
        If ``crs`` is an int that pyproj cannot resolve to a known CRS,
        or a string that pyproj cannot parse.
    ImportError
        If pyproj is not installed and ``crs`` is supplied as something
        other than a string. (A string is passed through verbatim so the
        WKT-only path keeps working without pyproj.)
    """
    if crs is None:
        return None
    if not isinstance(crs, (int, str)):
        raise TypeError(
            f"crs must be int (EPSG code), str (WKT or PROJ), or None; "
            f"got {type(crs).__name__}")
    if isinstance(crs, str):
        # Empty string is a common "no CRS" sentinel from upstream
        # GeoTIFFs; preserve the existing _vrt.write_vrt semantics (it
        # falls back to the first source's CRS for empty strings too).
        if not crs:
            return None
        # If the caller already handed us a WKT, return it untouched.
        # PROJCS/GEOGCS/PROJCRS/GEOGCRS are the standard WKT root
        # keywords; anything else (EPSG:NNNN, +proj=...) gets normalised
        # through pyproj so the downstream XML sees a canonical WKT.
        if crs.lstrip().startswith(('PROJCS', 'GEOGCS', 'PROJCRS', 'GEOGCRS',
                                     'COMPD_CS', 'COMPOUNDCRS')):
            return crs
        try:
            from pyproj import CRS
        except ImportError as exc:
            raise ImportError(
                "pyproj is required to convert non-WKT CRS strings (got "
                f"{crs!r}). Pass a WKT string directly, or install pyproj."
            ) from exc
        try:
            return CRS.from_user_input(crs).to_wkt()
        except Exception as exc:
            raise ValueError(
                f"Could not parse crs={crs!r} as an EPSG/PROJ/WKT string: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    # int branch: convert EPSG -> WKT via pyproj.
    try:
        from pyproj import CRS
    except ImportError as exc:
        raise ImportError(
            f"pyproj is required to convert crs={crs} (EPSG int) to WKT. "
            "Install pyproj, or pass crs as a WKT string."
        ) from exc
    try:
        return CRS.from_epsg(crs).to_wkt()
    except Exception as exc:
        raise ValueError(
            f"Could not resolve EPSG:{crs}: {type(exc).__name__}: {exc}"
        ) from exc


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

    Accepts a 6-tuple ``(a, b, c, d, e, f)`` in rasterio ``Affine``
    ordering, or a ``GeoTransform`` instance. Returns None for anything
    that isn't a recognisable 6-tuple. GDAL ordering
    ``(c, a, b, f, d, e)`` is NOT accepted.

    Rotated or skewed affines (``b != 0`` or ``d != 0``, beyond a
    1e-12 tolerance for float noise) are rejected with ``ValueError``.
    The on-disk GeoTIFF representation written by this package is
    axis-aligned, so silently dropping ``b`` and ``d`` would place the
    raster at the wrong location. Reproject onto an axis-aligned grid
    before writing.
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
        a, b, c, d, e, f = (float(x) for x in seq)
    except (TypeError, ValueError):
        return None
    _ROT_TOL = 1e-12
    if abs(b) > _ROT_TOL or abs(d) > _ROT_TOL:
        raise ValueError(
            f"attrs['transform'] has non-zero rotation/shear "
            f"(b={b!r}, d={d!r}); rotated or skewed affines are not "
            f"supported by the GeoTIFF writers in this module because "
            f"the on-disk GeoTIFF representation is axis-aligned and "
            f"would be written at the wrong location. Reproject the "
            f"raster to an axis-aligned grid before writing."
        )
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

    For 3D arrays the spatial dims are the two non-band dims. The helper
    filters out any dim named ``band`` / ``bands`` / ``channel`` (see
    ``_BAND_DIM_NAMES``) regardless of position, so a ``(y, x, band)``,
    ``(band, y, x)``, or ``(y, band, x)`` DataArray returns the y/x
    transform rather than picking up the band axis spacing as a pixel
    size. ``to_geotiff`` itself remaps ``(band, y, x)`` arrays to
    ``(y, x, band)`` before writing pixel bytes, but it calls
    :func:`_coords_to_transform` against the original DataArray, so the
    helper must handle both layouts to keep the geo-transform consistent
    with the file's coord arrays. See issue #1643.
    """
    if da.ndim == 3:
        # Drop the band-like dim and keep the two spatial dims in their
        # original (y, x) order. Position-based fallback covers the case
        # where none of the dims are named like a band axis.
        spatial = tuple(d for d in da.dims if d not in _BAND_DIM_NAMES)
        if len(spatial) == 2:
            ydim, xdim = spatial[0], spatial[1]
        else:
            # No identifiable band dim; fall back to dims[-2:] so the
            # original 2-D-style behaviour applies. This branch only
            # triggers for unusual 3D layouts callers built by hand.
            ydim = da.dims[-2]
            xdim = da.dims[-1]
    else:
        ydim = da.dims[-2]
        xdim = da.dims[-1]

    if xdim not in da.coords or ydim not in da.coords:
        return None

    x = da.coords[xdim].values
    y = da.coords[ydim].values

    if len(x) < 2 or len(y) < 2:
        return None

    # GeoTIFF only supports an affine transform; non-uniform spacing
    # cannot be expressed faithfully. Validate up-front instead of
    # silently writing a transform that only matches the first step.
    def _is_regular(coord, name):
        diffs = np.diff(coord)
        # Use median (not mean) so a single bad sample doesn't shift
        # the reference step. The 1e-6 relative tolerance is forgiving
        # for float artifacts in otherwise-uniform coords.
        step = float(np.median(diffs))
        if step == 0:
            raise ValueError(
                f"{name} coords are constant; cannot infer pixel size"
            )
        rel = float(np.max(np.abs(diffs - step)) / abs(step))
        if rel > 1e-6:
            raise ValueError(
                f"{name} coords are not uniformly spaced "
                f"(max relative deviation {rel:.3e} exceeds 1e-6); "
                f"GeoTIFF requires an affine transform."
            )

    _is_regular(x, "x")
    _is_regular(y, "y")

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
        if window is not None:
            r0, c0, _r1, _c1 = window
            origin_x_w = float(src_t.origin_x) + c0 * float(src_t.pixel_width)
            origin_y_w = float(src_t.origin_y) + r0 * float(src_t.pixel_height)
            attrs['transform'] = (
                float(src_t.pixel_width), 0.0, origin_x_w,
                0.0, float(src_t.pixel_height), origin_y_w,
            )
        else:
            t_tuple = _transform_tuple(geo_info)
            if t_tuple is not None:
                attrs['transform'] = t_tuple

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

    # VRT files (string paths only -- VRT XML references other files on disk)
    if isinstance(source, str) and source.lower().endswith('.vrt'):
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
        r0, c0, r1, c1 = window
        if not getattr(geo_info, 'has_georef', True):
            full_x = np.arange(c0, c1, dtype=np.int64)
            full_y = np.arange(r0, r1, dtype=np.int64)
        else:
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
        if arr.dtype.kind == 'f':
            if not np.isnan(nodata):
                arr[arr == arr.dtype.type(nodata)] = np.nan
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
            if np.isfinite(nodata) and float(nodata).is_integer():
                nodata_int = int(nodata)
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

    # tile_size only applies to tiled output; warn if the caller passed a
    # non-default size alongside strip mode (it would otherwise be silently
    # ignored).
    if not tiled and tile_size != 256:
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
                       extra_tags=None):
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
                    extra_tags=extra_tags_list)
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
                    extra_tags=extra_tags_list)

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
        return read_vrt(source, dtype=dtype, name=name, chunks=chunks)

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
    else:
        # Metadata-only read: O(1) memory via mmap, no pixel decompression
        geo_info, full_h, full_w, file_dtype, n_bands = _read_geo_info(
            source, overview_level=overview_level)
    nodata = geo_info.nodata

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
        if not getattr(geo_info, 'has_georef', True):
            win_x = np.arange(win_c0, win_c1, dtype=np.int64)
            win_y = np.arange(win_r0, win_r1, dtype=np.int64)
        else:
            t = geo_info.transform
            if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
                win_x = (np.arange(win_c0, win_c1, dtype=np.float64)
                         * t.pixel_width + t.origin_x)
                win_y = (np.arange(win_r0, win_r1, dtype=np.float64)
                         * t.pixel_height + t.origin_y)
            else:
                win_x = (np.arange(win_c0, win_c1, dtype=np.float64)
                         * t.pixel_width + t.origin_x
                         + t.pixel_width * 0.5)
                win_y = (np.arange(win_r0, win_r1, dtype=np.float64)
                         * t.pixel_height + t.origin_y
                         + t.pixel_height * 0.5)
        coords = {'y': win_y, 'x': win_x}
        full_h = win_r1 - win_r0
        full_w = win_c1 - win_c0
    else:
        coords = _geo_to_coords(geo_info, full_h, full_w)

    if band is not None:
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
    if max_pixels is not None:
        eff_bands = (1 if band is not None
                     else (n_bands if n_bands > 0 else 1))
        if full_h * full_w * eff_bands > max_pixels:
            raise ValueError(
                f"Requested region {full_h}x{full_w}x{eff_bands} "
                f"exceeds max_pixels={max_pixels:,}.")

    if name is None:
        import os
        name = os.path.splitext(os.path.basename(source))[0]

    attrs = {}
    _populate_attrs_from_geo_info(attrs, geo_info, window=window)
    if nodata is not None:
        attrs['nodata'] = nodata

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
        t = geo_info.transform
        if t is None or not getattr(geo_info, 'has_georef', True):
            coords = {
                'y': np.arange(r0, r1, dtype=np.int64),
                'x': np.arange(c0, c1, dtype=np.int64),
            }
        else:
            if geo_info.raster_type == RASTER_PIXEL_IS_POINT:
                full_x = (np.arange(c0, c1, dtype=np.float64)
                          * t.pixel_width + t.origin_x)
                full_y = (np.arange(r0, r1, dtype=np.float64)
                          * t.pixel_height + t.origin_y)
            else:
                full_x = (np.arange(c0, c1, dtype=np.float64)
                          * t.pixel_width + t.origin_x
                          + t.pixel_width * 0.5)
                full_y = (np.arange(r0, r1, dtype=np.float64)
                          * t.pixel_height + t.origin_y
                          + t.pixel_height * 0.5)
            coords = {'y': full_y, 'x': full_x}
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

    With ``chunks=``, returns a Dask+CuPy DataArray for out-of-core
    GPU pipelines.

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
        If set, return a Dask-chunked CuPy DataArray. int for square
        chunks, (row, col) tuple for rectangular.
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
            arr_cpu, _ = _read_to_array(
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
            nodata = geo_info.nodata
            if nodata is not None:
                attrs['nodata'] = nodata
                arr_gpu = _apply_nodata_mask_gpu(arr_gpu, nodata)
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
            if window is not None:
                r0, c0, r1, c1 = window
                t = geo_info.transform
                if t is None or not getattr(geo_info, 'has_georef', True):
                    coords = {
                        'y': np.arange(r0, r1, dtype=np.int64),
                        'x': np.arange(c0, c1, dtype=np.int64),
                    }
                elif geo_info.raster_type == RASTER_PIXEL_IS_POINT:
                    coords = {
                        'x': (np.arange(c0, c1, dtype=np.float64)
                              * t.pixel_width + t.origin_x),
                        'y': (np.arange(r0, r1, dtype=np.float64)
                              * t.pixel_height + t.origin_y),
                    }
                else:
                    coords = {
                        'x': (np.arange(c0, c1, dtype=np.float64)
                              * t.pixel_width + t.origin_x
                              + t.pixel_width * 0.5),
                        'y': (np.arange(r0, r1, dtype=np.float64)
                              * t.pixel_height + t.origin_y
                              + t.pixel_height * 0.5),
                    }
            else:
                coords = _geo_to_coords(
                    geo_info, arr_gpu.shape[0], arr_gpu.shape[1])
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
            # Drop read_to_array's geo_info: orientation transform handling
            # below operates on our pre-extracted geo_info so the 2/3/4 case
            # is covered regardless of #1539's merge state.
            arr_cpu, _ = _read_to_array(
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
        arr_cpu, _ = _read_to_array(
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
            arr_cpu, _ = _read_to_array(
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

    if (ifd.photometric == 0 and samples == 1 and not arr_was_cpu_decoded):
        gpu_dtype = np.dtype(str(arr_gpu.dtype))
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
        arr_gpu = _apply_nodata_mask_gpu(arr_gpu, nodata)

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

    if chunks is not None:
        if isinstance(chunks, int):
            chunk_dict = {'y': chunks, 'x': chunks}
        else:
            chunk_dict = {'y': chunks[0], 'x': chunks[1]}
        result = result.chunk(chunk_dict)

    return result


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
                      photometric: str | int = 'auto') -> None:
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
        - ``'jpeg'``: nvJPEG when libnvjpeg is loadable, Pillow
          otherwise. Note that ``to_geotiff`` rejects
          ``compression='jpeg'`` at runtime because its CPU encoder
          omits the required TIFF JPEGTables tag (347); this GPU entry
          point instead emits self-contained JFIF tiles. The two
          writers therefore disagree about JPEG-in-TIFF interop. Files
          produced here decode through this library's own reader but
          may not round-trip through GDAL, rasterio, or libtiff
          readers that require the JPEGTables tag. Treat the JPEG path
          as experimental and internal-reader-only until the
          JPEGTables fix lands.
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
    """
    if not tiled:
        raise ValueError(
            "write_geotiff_gpu requires tiled=True. nvCOMP batch "
            "compression is tile-based; the strip layout is not "
            "implemented on the GPU path. Use to_geotiff(..., gpu=False, "
            "tiled=False) for strip output on CPU.")
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
    max_pixels : int or None
        Maximum allowed pixel count (width * height * samples) for the
        assembled VRT region. None uses the reader default (~1 billion).
        Matches ``open_geotiff`` / ``read_geotiff_dask`` /
        ``read_geotiff_gpu``.

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
    """
    from ._reader import _coerce_path
    from ._vrt import read_vrt as _read_vrt_internal

    source = _coerce_path(source)

    # Reject non-positive chunk sizes up front so the VRT dask path
    # surfaces the same error as ``read_geotiff_dask`` (#1776). Without
    # this check ``chunks=0`` raised ``ZeroDivisionError`` deep in dask
    # and ``chunks=-1`` was silently accepted. ``chunks=None`` is the
    # default (eager read), so allow it through here.
    chunks = _validate_chunks_arg(chunks, allow_none=True)

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
    # For multi-band reads (``band is None`` and ``arr.ndim == 3``), each
    # band can declare its own ``<NoDataValue>``. The float-VRT path masks
    # per-band inline in ``_vrt._read_data``; mirror that here by walking
    # ``vrt.bands`` and masking each ``arr[..., i]`` slice against its own
    # sentinel. Before this branch, only band 0's sentinel was applied and
    # bands 1+ left their integer sentinels as literal finite values in
    # the returned float64 array. See issue #1611.
    def _sentinel_for_dtype(nodata_val, dtype):
        """Return ``dtype``-cast sentinel for ``nodata_val`` or ``None``
        if the value can't be represented in ``dtype`` (non-integer
        dtype, out-of-range, non-finite, or fractional). Mirrors the
        gating PR #1583 added to other read paths via
        ``_int_nodata_in_range``.

        A plain Python ``int`` ``nodata_val`` is handled without going
        through ``float`` first, so 64-bit sentinels such as
        ``2**64 - 1`` (``UInt64`` max) and ``-2**63`` (``Int64`` min)
        round-trip without the float64 rounding that pushes them just
        past the dtype's representable range.  ``_parse_band_nodata``
        in ``_vrt.py`` parses integer-band ``<NoDataValue>`` directly
        as ``int`` to feed this path.  See issue #1783 follow-up.
        """
        if nodata_val is None or dtype.kind not in ('u', 'i'):
            return None
        info = np.iinfo(dtype)
        # Fast/exact path: ``nodata_val`` is already an integer.  Avoids
        # the float64 round-trip that loses precision near the int64 /
        # uint64 extremes.  ``bool`` is a subclass of ``int`` -- treat
        # True/False as a 1/0 sentinel rather than rejecting outright,
        # matching the existing ``int(float(...))`` behaviour.
        if isinstance(nodata_val, (int, np.integer)) and not isinstance(
                nodata_val, bool):
            nodata_int = int(nodata_val)
            if info.min <= nodata_int <= info.max:
                return dtype.type(nodata_int)
            return None
        try:
            nodata_f = float(nodata_val)
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(nodata_f) and nodata_f.is_integer()
                and info.min <= nodata_f <= info.max):
            return None
        return dtype.type(int(nodata_f))

    if arr.dtype.kind in ('u', 'i'):
        if arr.ndim == 3 and band is None and vrt.bands:
            # Per-band masking: walk ``vrt.bands`` once and stream each
            # band's mask. The first band with a sentinel hit promotes
            # ``arr`` to float64 in place; ``int_arr`` keeps the original
            # integer view alive so subsequent bands still compare against
            # the exact sentinel dtype (the post-promotion float64 view
            # works too, but staying on the integer dtype avoids any
            # rounding edge case on extreme sentinels). Peak boolean-mask
            # memory is O(H * W), not O(bands * H * W) like the earlier
            # collect-then-apply implementation.
            int_arr = arr
            int_dtype = int_arr.dtype
            for i, vrt_band in enumerate(vrt.bands):
                if i >= int_arr.shape[-1]:
                    break
                sentinel = _sentinel_for_dtype(vrt_band.nodata, int_dtype)
                if sentinel is None:
                    continue
                mask = int_arr[..., i] == sentinel
                if not mask.any():
                    continue
                if arr.dtype != np.float64:
                    arr = arr.astype(np.float64)
                arr[..., i][mask] = np.nan
        elif nodata is not None:
            sentinel = _sentinel_for_dtype(nodata, arr.dtype)
            if sentinel is not None:
                mask = arr == sentinel
                if mask.any():
                    arr = arr.astype(np.float64)
                    arr[mask] = np.nan

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
