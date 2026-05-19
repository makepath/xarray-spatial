"""Coordinate / transform helpers shared across the geotiff backends.

Internal module — symbols are re-exported from ``xrspatial.geotiff`` for
backwards-compatible private imports used in tests and other backends.
Extracted from ``__init__.py`` as part of issue #1813 to remove the
duplicated GeoTransform-to-(y, x) inline code that lived in each
backend's eager / dask / GPU / VRT read path.

Two coord conventions are emitted:

* ``has_georef=True`` (file carries GeoTIFF transform tags) → float64
  pixel-center coords in projected units. ``raster_type='point'`` skips
  the half-pixel shift (the tiepoint already sits at the pixel center);
  ``raster_type='area'`` adds the half-pixel shift.
* ``has_georef=False`` → integer pixel coords ``0..N-1``. Files without
  real georef tags carry a placeholder unit transform that would emit
  synthetic ``[-0.5, -1.5, ...]`` floats; the integer-pixel fallback
  keeps the no-georef path consistent across backends (#1710, #1753).

A ``window=(r0, c0, r1, c1)`` parameter shifts the output coord arrays
to the windowed sub-rectangle, so callers don't have to special-case
windowed vs full reads.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ._geotags import GeoTransform, RASTER_PIXEL_IS_POINT


# Names of dims that ``to_geotiff`` / ``write_geotiff_gpu`` treat as the
# non-spatial band axis. Used both to remap ``(band, y, x)`` inputs to
# ``(y, x, band)`` before writing and to skip the band axis when inferring
# a GeoTransform from coords (see :func:`coords_to_transform` / #1643).
_BAND_DIM_NAMES = ('band', 'bands', 'channel')


# Stamped on reads from files that carry no GeoTIFF transform tags
# (ModelTransformation, ModelPixelScale, or ModelTiepoint). The
# reader emits ``np.arange(start, stop, dtype=int64)`` placeholder y/x
# coords in that case, and the writer needs an unambiguous signal to
# distinguish those from user-authored int64 step-1 coord grids that
# happen to match the same shape (issue #2120). Treating any int64
# ascending-step-1 grid as the placeholder silently stripped georef
# from real user data; treating the marker as the signal makes that
# round-trip safe again.
_NO_GEOREF_KEY = '_xrspatial_no_georef'


def _has_no_georef_marker(da: xr.DataArray) -> bool:
    """True iff ``da`` was stamped by the reader as carrying no georef.

    The reader sets ``attrs[_NO_GEOREF_KEY] = True`` whenever it emits
    the placeholder int64 step-1 y/x coords for files without GeoTIFF
    transform tags. The writer checks the same marker before deciding
    that an int64 step-1 grid is the placeholder and skipping transform
    synthesis. See issue #2120 for the silent-strip regression this
    replaced.
    """
    return da.attrs.get(_NO_GEOREF_KEY) is True


def _is_no_georef_sentinel(coord: np.ndarray) -> bool:
    """True iff ``coord`` matches the read-side no-georef placeholder shape.

    ``coords_from_pixel_geometry`` emits ``np.arange(start, stop,
    dtype=np.int64)`` for the y/x coords whenever the source file
    carries no GeoTIFF transform tags -- both for full reads
    (``start=0``) and windowed reads (``start=window_offset``). See
    issues #1710, #1753, #1949.

    The writer no longer treats coord shape alone as the no-georef
    signal: it checks ``attrs[_NO_GEOREF_KEY]`` instead. See
    :func:`_has_no_georef_marker` and issue #2120. This helper remains
    available as a diagnostic for the placeholder shape, and the
    existing tests in ``test_int_coord_sentinel_2087.py`` still pin
    the predicate. It is no longer called from the writer path, so
    user-authored int64 step-1 grids that match this pattern but lack
    the marker keep their georef on round-trip.
    """
    if coord.dtype != np.int64:
        return False
    n = len(coord)
    if n < 1:
        return False
    return bool(np.array_equal(
        coord, np.arange(coord[0], coord[0] + n, dtype=np.int64)
    ))


def coords_from_pixel_geometry(
    origin_x: float,
    origin_y: float,
    pixel_width: float,
    pixel_height: float,
    height: int,
    width: int,
    *,
    is_point: bool = False,
    window: tuple | None = None,
    has_georef: bool = True,
) -> dict:
    """Build y/x coordinate arrays from raw pixel geometry.

    ``window``, when given, is ``(r0, c0, r1, c1)`` in source-pixel
    coordinates; the returned arrays describe rows ``r0..r1-1`` and
    columns ``c0..c1-1`` rather than ``0..height-1`` and ``0..width-1``.
    ``height`` and ``width`` should match the windowed extent in that
    case (caller already sliced the data).

    For ``has_georef=False`` the result is integer pixel indices instead
    of projected coordinates; this matches the no-georef fallback every
    backend agreed on (#1710, #1753).
    """
    if window is not None:
        r0, c0, r1, c1 = window
        if not has_georef:
            return {
                'y': np.arange(r0, r1, dtype=np.int64),
                'x': np.arange(c0, c1, dtype=np.int64),
            }
        x_base = np.arange(c0, c1, dtype=np.float64) * pixel_width + origin_x
        y_base = np.arange(r0, r1, dtype=np.float64) * pixel_height + origin_y
    else:
        if not has_georef:
            return {
                'y': np.arange(height, dtype=np.int64),
                'x': np.arange(width, dtype=np.int64),
            }
        x_base = np.arange(width, dtype=np.float64) * pixel_width + origin_x
        y_base = np.arange(height, dtype=np.float64) * pixel_height + origin_y

    if is_point:
        # Tiepoint is at pixel center; no shift.
        return {'y': y_base, 'x': x_base}
    # Tiepoint is at pixel edge; shift to center.
    return {
        'y': y_base + pixel_height * 0.5,
        'x': x_base + pixel_width * 0.5,
    }


def transform_tuple_from_pixel_geometry(
    origin_x: float,
    origin_y: float,
    pixel_width: float,
    pixel_height: float,
    *,
    window: tuple | None = None,
) -> tuple:
    """Return a rasterio-style 6-tuple for the given pixel geometry.

    Format: ``(pixel_width, 0.0, origin_x, 0.0, pixel_height, origin_y)``.
    When ``window=(r0, c0, ...)`` is given, the origin shifts to the
    top-left of the windowed sub-rectangle.
    """
    if window is not None:
        r0, c0 = window[0], window[1]
        ox = float(origin_x) + c0 * float(pixel_width)
        oy = float(origin_y) + r0 * float(pixel_height)
    else:
        ox = float(origin_x)
        oy = float(origin_y)
    return (
        float(pixel_width), 0.0, ox,
        0.0, float(pixel_height), oy,
    )


def coords_from_geo_info(
    geo_info,
    height: int,
    width: int,
    *,
    window: tuple | None = None,
) -> dict:
    """Build y/x coordinate arrays for a GeoInfo, optionally windowed.

    Thin wrapper around :func:`coords_from_pixel_geometry` that pulls the
    origin / pixel size off ``geo_info.transform`` and the
    ``raster_type`` / ``has_georef`` flags off ``geo_info``. A missing
    ``transform`` (``None``) is treated as ``has_georef=False`` so the
    no-georef integer-pixel fallback path runs (#1710 / #1753).
    """
    has_georef = getattr(geo_info, 'has_georef', True)
    t = geo_info.transform
    if not has_georef or t is None:
        return coords_from_pixel_geometry(
            0.0, 0.0, 1.0, 1.0, height, width,
            window=window, has_georef=False,
        )
    is_point = geo_info.raster_type == RASTER_PIXEL_IS_POINT
    return coords_from_pixel_geometry(
        t.origin_x, t.origin_y, t.pixel_width, t.pixel_height,
        height, width,
        is_point=is_point, window=window, has_georef=True,
    )


def geo_to_coords(geo_info, height: int, width: int) -> dict:
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
    return coords_from_geo_info(geo_info, height, width)


def transform_tuple(geo_info) -> tuple | None:
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
    return transform_tuple_from_pixel_geometry(
        t.origin_x, t.origin_y, t.pixel_width, t.pixel_height,
    )


def transform_from_attr(attr_val) -> 'GeoTransform | None':
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


def require_transform_for_georeferenced(
    da: xr.DataArray, geo_transform
) -> None:
    """Raise if ``da`` carries spatial coords but no transform was derived.

    Used by the writer entry points (#1945). A DataArray whose spatial
    dim names appear in ``da.coords`` is an explicit caller request for a
    georeferenced output. Silently falling through to a non-georeferenced
    TIFF -- which is what the old code did for 1x1 inputs and inputs with
    a degenerate axis -- corrupted round-trips. If the writer cannot
    recover a transform from coords *and* the caller did not supply
    ``attrs['transform']``, fail closed instead.

    ``geo_transform`` is the value the writer has already resolved (from
    ``attrs['transform']`` first, then from coord arrays). If it's not
    ``None`` we have a transform and there's nothing to check.
    """
    if geo_transform is not None:
        return
    if da.ndim == 3:
        spatial = tuple(d for d in da.dims if d not in _BAND_DIM_NAMES)
        if len(spatial) == 2:
            ydim, xdim = spatial[0], spatial[1]
        else:
            ydim = da.dims[-2]
            xdim = da.dims[-1]
    else:
        ydim = da.dims[-2]
        xdim = da.dims[-1]
    if xdim in da.coords and ydim in da.coords:
        # The reader stamps ``attrs[_NO_GEOREF_KEY] = True`` when it
        # emits its int64 step-1 placeholder coords for files without
        # GeoTIFF transform tags (#1710, #1753, #1949). The writer
        # checks that marker rather than the coord shape itself so a
        # user-authored int64 grid that happens to match the placeholder
        # pattern (e.g. ``x=[500,501,502], y=[1000,1001]``) keeps its
        # georef on round-trip. Pre-#2120 the writer detected the
        # placeholder by shape alone and silently stripped georef from
        # real user data with that shape.
        if _has_no_georef_marker(da):
            return
        raise ValueError(
            f"Cannot infer GeoTIFF transform from a "
            f"{tuple(da.sizes.values())} array with spatial coords on "
            f"both axes: neither coord array could yield a pixel size "
            f"(1x1 inputs, or coords spaced non-uniformly). Supply the "
            f"affine transform explicitly via ``attrs['transform']`` "
            f"(rasterio 6-tuple ``(px, 0, ox, 0, py, oy)``) or drop the "
            f"coords if a non-georeferenced TIFF is desired."
        )


def coords_to_transform(da: xr.DataArray) -> 'GeoTransform | None':
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
    :func:`coords_to_transform` against the original DataArray, so the
    helper must handle both layouts to keep the geo-transform consistent
    with the file's coord arrays. See issue #1643.

    DataArrays carrying ``attrs[_NO_GEOREF_KEY] = True`` (stamped by
    the reader for files without GeoTIFF transform tags) return
    ``None`` before the uniformity check runs so the placeholder
    round-trips without inventing a fake unit transform (#1949,
    #2120). Pre-#2120 the placeholder was detected by coord shape
    alone, which silently stripped georef from user-authored int64
    step-1 grids that matched the same arange pattern.
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

    # 1x1 has no pixel-size signal on either axis. The caller must supply
    # ``attrs['transform']`` (handled by the writer before calling us).
    # Returning ``None`` lets the writer detect this and raise rather than
    # silently writing a non-georeferenced TIFF (#1945).
    if len(x) < 2 and len(y) < 2:
        return None

    # No-georef path: the reader stamps
    # ``attrs[_NO_GEOREF_KEY] = True`` whenever it emits the int64
    # step-1 placeholder y/x coords for files without GeoTIFF transform
    # tags (#1710, #1753, #1949). Synthesising a GeoTransform from those
    # arrays would inject a fake unit transform (``pixel_width=1.0``,
    # origin derived from ``coord[0]``) into the written file's
    # ModelPixelScale / ModelTiepoint tags. The next read would then
    # take the georef branch and the coord dtype silently flip to
    # ``float64`` with ``attrs['transform']`` present, breaking the
    # no-georef contract that downstream code branches on.
    #
    # Older revisions of this code detected the placeholder by coord
    # shape: first ``dtype.kind in ('i', 'u')`` (broad, then tightened
    # in #2087 to the exact ``arange(start, start+n, dtype=int64)``
    # pattern). Both shape-based checks misclassified user-authored
    # int64 step-1 grids (e.g. ``x=[500,501,502], y=[1000,1001]``) as
    # the placeholder and silently stripped their georef on write
    # (#2120). The marker is now the only signal: shape-matching coords
    # without the marker fall through to the regular transform
    # synthesis below.
    if _has_no_georef_marker(da):
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

    # Degenerate-axis fallback (#1945). When one axis has length 1, we
    # can't read a step off it (``coord[1] - coord[0]`` is undefined),
    # so we recover the per-axis pixel size from the non-degenerate
    # axis and assume square pixels for the degenerate one. That matches
    # how every other geospatial reader handles 1xN / Nx1 strips. The
    # earlier behaviour — bailing out and silently writing a
    # non-georeferenced TIFF — broke round-trips for legitimate
    # single-scanline / single-profile rasters.
    if len(x) >= 2:
        _is_regular(x, "x")
        pixel_width = float(x[1] - x[0])
    else:
        pixel_width = None
    if len(y) >= 2:
        _is_regular(y, "y")
        pixel_height = float(y[1] - y[0])
    else:
        pixel_height = None

    if pixel_width is None:
        # Borrow magnitude from y; x increases left-to-right by convention.
        pixel_width = abs(pixel_height)
    if pixel_height is None:
        # Borrow magnitude from x; y decreases top-to-bottom by convention,
        # so flip sign.
        pixel_height = -abs(pixel_width)

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
