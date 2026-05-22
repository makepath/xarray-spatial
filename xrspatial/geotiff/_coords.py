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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr

from ._errors import NonUniformCoordsError
from ._geotags import _NO_GEOREF_KEY, RASTER_PIXEL_IS_POINT, GeoTransform

# Tolerance for rotation/shear detection in affine transforms. ``b`` and
# ``d`` terms below this magnitude are treated as float noise and pass
# through as axis-aligned. Shared between ``transform_from_attr`` (6-tuple
# attrs) and ``to_geotiff`` (rasterio ``Affine`` attrs) so the two gates
# stay in lockstep on future tolerance tweaks (#2301).
ROTATION_SHEAR_TOL = 1e-12

# Canonical georef_status string values (mirrored in ``_attrs.py`` as
# ``GEOREF_STATUS_*`` constants). Issue #2225 centralises the
# transform/georef contract here; ``_attrs.py`` aliases these names so
# the public attrs surface keeps its existing identifiers.
GEOREF_STATUS_NONE = 'none'
GEOREF_STATUS_COORDS = 'coords'
GEOREF_STATUS_TRANSFORM_ONLY = 'transform_only'
GEOREF_STATUS_CRS_ONLY = 'crs_only'
GEOREF_STATUS_ROTATED_DROPPED = 'rotated_dropped'
GEOREF_STATUS_FULL = 'full'

GeorefStatus = Literal[
    'none',
    'coords',
    'transform_only',
    'crs_only',
    'rotated_dropped',
    'full',
]


@dataclass(frozen=True)
class GeorefResolution:
    """Typed result returned by :func:`resolve_georef` (issue #2225).

    Centralises the three georef decisions every backend used to make
    inline: how to derive the affine transform, which of the canonical
    ``georef_status`` buckets the input falls into, and whether the
    no-georef marker was applied.

    Fields:

    * ``transform`` -- the resolved :class:`GeoTransform`, or ``None``
      when no transform could be derived (no coords, no
      ``attrs['transform']``, no inbound ``GeoInfo`` transform). Rotated
      / sheared affines surface here as ``None`` because the on-disk
      GeoTIFF representation is axis-aligned; the ``dropped_reason``
      field records the original reason for callers that care.
    * ``georef_status`` -- one of ``GeorefStatus``. The bucket the
      resolver landed in; mirrors the canonical attr emitted by
      :mod:`xrspatial.geotiff._attrs`. ``'coords'`` is the writer-side
      bucket for inputs whose only georef signal is the coord arrays
      themselves (no CRS, no inbound transform). Reader-side callers
      that distinguish ``transform_only`` and ``crs_only`` use those.
    * ``dropped_reason`` -- short string explaining why ``transform``
      is ``None`` (e.g. ``'rotated_affine_dropped'``,
      ``'degenerate_axis'``, ``'no_coords'``), or ``None`` when the
      resolver did derive a transform.
    * ``applied_no_georef_marker`` -- True iff the resolver routed the
      input through the no-georef placeholder path. Mirrors the
      reader's ``attrs[_NO_GEOREF_KEY] = True`` marker so writer call
      sites can branch the same way.
    """

    transform: 'GeoTransform | None' = None
    georef_status: GeorefStatus = GEOREF_STATUS_NONE
    dropped_reason: str | None = None
    applied_no_georef_marker: bool = False


# Names of dims that ``to_geotiff`` / ``write_geotiff_gpu`` treat as the
# non-spatial band axis. Used both to remap ``(band, y, x)`` inputs to
# ``(y, x, band)`` before writing and to skip the band axis when inferring
# a GeoTransform from coords (see :func:`coords_to_transform` / #1643).
_BAND_DIM_NAMES = ('band', 'bands', 'channel')


# Attribute key callers set to True to opt in to the borrow-from-other-axis
# pixel-size fallback for 1xN / Nx1 writes. See ``coords_to_transform`` and
# ``require_transform_for_georeferenced`` for the contract. Issue #2214
# replaced the unconditional borrow with this explicit opt-in so callers
# whose source raster has non-square pixels don't get a silently wrong
# transform written for a degenerate strip.
_ASSUME_SQUARE_DEGENERATE_KEY = 'assume_square_pixels_for_degenerate_axis'


def _assume_square_for_degenerate(da: Any) -> bool:
    """True iff ``da.attrs`` opts in to the square-pixel degenerate fallback.

    Identity check on ``True`` (rather than plain truthiness) so a stray
    string like ``attrs['assume_square_pixels_for_degenerate_axis'] = 'no'``
    doesn't quietly enable the borrow path. Non-mapping ``attrs`` (raw
    numpy/cupy arrays) return False.
    """
    attrs = getattr(da, 'attrs', None)
    if not isinstance(attrs, Mapping):
        return False
    return attrs.get(_ASSUME_SQUARE_DEGENERATE_KEY) is True


def _has_no_georef_marker(da: Any) -> bool:
    """True iff ``da`` was stamped by the reader as carrying no georef.

    The reader sets ``attrs[_NO_GEOREF_KEY] = True`` whenever it emits
    the placeholder int64 step-1 y/x coords for files without GeoTIFF
    transform tags. The writer checks the same marker before deciding
    that an int64 step-1 grid is the placeholder and skipping transform
    synthesis. See issue #2120 for the silent-strip regression this
    replaced.

    The identity check (``is True``) is deliberate: only the exact
    boolean ``True`` flips the writer into no-georef mode. A stray
    third-party stamp like ``attrs['_xrspatial_no_georef'] = 'yes'``
    should not be treated as truthy and silently drop a transform.

    Inputs that are not xarray DataArrays (e.g. a raw ``numpy.ndarray``
    or ``cupy.ndarray`` passed directly to ``write_geotiff_gpu``) carry
    no attrs and therefore no marker; return ``False`` rather than
    raise so callers can use this as a plain predicate.
    """
    attrs = getattr(da, 'attrs', None)
    if not isinstance(attrs, Mapping):
        return False
    return attrs.get(_NO_GEOREF_KEY) is True


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
    if abs(b) > ROTATION_SHEAR_TOL or abs(d) > ROTATION_SHEAR_TOL:
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
            f"(1x1 inputs, 1xN / Nx1 strips, or coords spaced "
            f"non-uniformly). Supply the affine transform explicitly via "
            f"``attrs['transform']`` (rasterio 6-tuple "
            f"``(px, 0, ox, 0, py, oy)``); for 1xN / Nx1 strips known to "
            f"have square pixels, opt in to the borrow-from-other-axis "
            f"fallback with "
            f"``attrs['{_ASSUME_SQUARE_DEGENERATE_KEY}'] = True``. Drop "
            f"the coords if a non-georeferenced TIFF is desired."
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
    #
    # Issue #2215: raise NonUniformCoordsError (subclass of ValueError,
    # so existing ``except ValueError`` callers keep working) rather
    # than plain ValueError. This keeps the exception type consistent
    # with the upstream ambiguous-metadata validator, which already
    # raises NonUniformCoordsError for the same condition. Before this,
    # the validator caught the y/x case and this path caught the
    # alias-named case, leaving callers with two different exception
    # types depending on which dim name they used.
    def _is_regular(coord, name):
        diffs = np.diff(coord)
        # Use median (not mean) so a single bad sample doesn't shift
        # the reference step. The 1e-6 relative tolerance is forgiving
        # for float artifacts in otherwise-uniform coords.
        step = float(np.median(diffs))
        if step == 0:
            raise NonUniformCoordsError(
                f"{name} coords are constant; cannot infer pixel size"
            )
        rel = float(np.max(np.abs(diffs - step)) / abs(step))
        if rel > 1e-6:
            raise NonUniformCoordsError(
                f"{name} coords are not uniformly spaced "
                f"(max relative deviation {rel:.3e} exceeds 1e-6); "
                f"GeoTIFF requires an affine transform."
            )

    # Degenerate-axis handling (#1945, #2214). When one axis has length
    # 1, we can't read a step off it (``coord[1] - coord[0]`` is
    # undefined). #1945 first taught the writer to borrow the pixel size
    # from the non-degenerate axis on the assumption that the raster was
    # square. That assumption is unsafe: a 30 m by 10 m source raster
    # served as a 1xN strip silently wrote out with 30 m by 30 m pixels,
    # and downstream slope / proximity / zonal math then trusted the
    # wrong transform (#2214).
    #
    # We now return ``None`` for the degenerate case by default, which
    # routes to ``require_transform_for_georeferenced`` and raises a
    # ValueError naming both escape hatches. Callers who know their
    # source is square can opt in explicitly via
    # ``attrs['assume_square_pixels_for_degenerate_axis'] = True``;
    # callers who know the true pixel geometry should pass it on
    # ``attrs['transform']`` (the writer prefers that path anyway because
    # it round-trips bit-exactly).
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

    if pixel_width is None or pixel_height is None:
        if not _assume_square_for_degenerate(da):
            # Fail closed; the writer will raise via
            # require_transform_for_georeferenced.
            return None
        if pixel_width is None:
            # Borrow magnitude from y; x increases left-to-right by
            # convention.
            pixel_width = abs(pixel_height)
        if pixel_height is None:
            # Borrow magnitude from x; y decreases top-to-bottom by
            # convention, so flip sign.
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


def _has_crs_signal(crs_epsg, crs_wkt) -> bool:
    """True iff at least one CRS field is populated.

    Lifted out so the read-side (``GeoInfo``) and write-side
    (``attrs['crs']`` / ``attrs['crs_wkt']``) call sites of
    :func:`resolve_georef` share the same decision rule. An empty
    string ``crs_wkt`` (which the VRT parser emits when the XML has
    a ``<SRS>`` element with no recognised body) is treated as
    absent so the resolver does not bucket those into ``crs_only``.
    """
    if crs_epsg is not None:
        return True
    if isinstance(crs_wkt, str) and crs_wkt:
        return True
    return False


def resolve_georef(
    source: Any = None,
    *,
    geo_info: Any = None,
    crs_epsg: int | None = None,
    crs_wkt: str | None = None,
    transform_hint: Any = None,
) -> GeorefResolution:
    """Centralised transform/georef resolver shared across backends.

    Issue #2225. One function owns the four decisions every backend
    used to make inline:

    1. Coord-to-transform inference (via :func:`coords_to_transform`).
    2. The no-georef marker behaviour
       (``attrs[_NO_GEOREF_KEY] = True`` -> placeholder coords).
    3. Degenerate-axis policy (1xN / Nx1 / 1x1 inputs).
    4. Rotated-affine drop policy (rotated transforms are not
       expressible as axis-aligned 6-tuples and are dropped).

    Inputs:

    * ``source`` -- either an :class:`xarray.DataArray` (writer entry
      points) or ``None``. When a DataArray is supplied, the resolver
      reads ``attrs['transform']``, ``attrs['crs']`` / ``attrs['crs_wkt']``,
      and the ``_NO_GEOREF_KEY`` marker off the array's attrs.
    * ``geo_info`` -- a reader-side :class:`GeoInfo`-like object (the
      eager / dask / GPU read paths feed this). When supplied, the
      resolver derives the transform from ``geo_info.transform`` and
      bucketises the status from ``geo_info.has_georef`` /
      ``geo_info.crs_epsg`` / ``geo_info.crs_wkt`` /
      ``geo_info.transform.rotated_affine``. ``crs_epsg`` and
      ``crs_wkt`` overrides take precedence over the ``GeoInfo`` values
      when both are passed (used by the VRT inline path which holds CRS
      state outside the synthesised ``GeoInfo``).
    * ``crs_epsg`` / ``crs_wkt`` -- direct CRS signals for callers that
      do not have a DataArray or a ``GeoInfo`` (the VRT path threads
      these from the parsed XML).
    * ``transform_hint`` -- a :class:`GeoTransform` or rasterio-style
      6-tuple to use when neither ``source`` nor ``geo_info`` is given.

    The returned :class:`GeorefResolution` contract is documented on
    the dataclass.
    """
    # Reader path: a GeoInfo (or VRT-synthesised stand-in) carries
    # has_georef + transform + (optional) rotated_affine. The resolver
    # mirrors the decision table from ``_compute_georef_status`` so the
    # eager / dask / GPU / VRT read sites all land on the same bucket.
    if geo_info is not None:
        gi_transform = getattr(geo_info, 'transform', None)
        rotated_affine = (
            getattr(gi_transform, 'rotated_affine', None)
            if gi_transform is not None else None
        )
        has_georef = bool(getattr(geo_info, 'has_georef', False))
        # CRS overrides win so VRT call sites can pass the parsed
        # ``vrt.crs_wkt`` without rebuilding the GeoInfo. When the
        # override is left at ``None`` we read from the GeoInfo.
        resolved_crs_epsg = (
            crs_epsg if crs_epsg is not None
            else getattr(geo_info, 'crs_epsg', None)
        )
        resolved_crs_wkt = (
            crs_wkt if crs_wkt is not None
            else getattr(geo_info, 'crs_wkt', None)
        )
        has_crs = _has_crs_signal(resolved_crs_epsg, resolved_crs_wkt)

        if rotated_affine is not None:
            # ``allow_rotated=True`` path: the reader stamped the
            # rotated 6-tuple onto the transform but reported
            # ``has_georef=False``. The axis-aligned transform we would
            # surface here cannot represent the rotation, so drop it.
            return GeorefResolution(
                transform=None,
                georef_status=GEOREF_STATUS_ROTATED_DROPPED,
                dropped_reason='rotated_affine_dropped',
                applied_no_georef_marker=True,
            )

        if has_georef and gi_transform is not None:
            status: GeorefStatus = (
                GEOREF_STATUS_FULL if has_crs else GEOREF_STATUS_TRANSFORM_ONLY
            )
            return GeorefResolution(
                transform=gi_transform,
                georef_status=status,
                dropped_reason=None,
                applied_no_georef_marker=False,
            )

        # No transform on the reader path -> placeholder coords + marker.
        if has_crs:
            return GeorefResolution(
                transform=None,
                georef_status=GEOREF_STATUS_CRS_ONLY,
                dropped_reason='no_transform_tags',
                applied_no_georef_marker=True,
            )
        return GeorefResolution(
            transform=None,
            georef_status=GEOREF_STATUS_NONE,
            dropped_reason='no_transform_tags',
            applied_no_georef_marker=True,
        )

    # Writer path: ``source`` is a DataArray. Prefer attrs['transform']
    # over coord-derived transforms (the same precedence the inline
    # writer code applied for issue #1484 -- attrs['transform']
    # round-trips bit-exactly while coord subtraction can drift on
    # fractional pixel sizes).
    if isinstance(source, xr.DataArray):
        attrs = source.attrs
        attr_transform = attrs.get('transform') if attrs is not None else None
        a_crs_epsg = crs_epsg
        a_crs_wkt = crs_wkt
        if attrs is not None:
            # Mirror ``attrs_to_metadata`` precedence: string ``crs``
            # values fold into ``crs_wkt`` (some pipelines stash the
            # WKT string under ``attrs['crs']``), bool values are
            # ignored at the boundary so the resolver does not flag a
            # bogus ``has_crs`` signal on ``crs=True`` (issue #1971).
            crs_val = attrs.get('crs')
            if a_crs_wkt is None and isinstance(crs_val, str) and crs_val:
                a_crs_wkt = crs_val
            if (a_crs_epsg is None
                    and crs_val is not None
                    and not isinstance(crs_val, (bool, str))):
                try:
                    a_crs_epsg = int(crs_val)
                except (TypeError, ValueError):
                    a_crs_epsg = None
            if a_crs_wkt is None:
                wkt_val = attrs.get('crs_wkt')
                if isinstance(wkt_val, str) and wkt_val:
                    a_crs_wkt = wkt_val
        has_crs = _has_crs_signal(a_crs_epsg, a_crs_wkt)

        # ``transform_from_attr`` raises on rotated 6-tuples (b/d != 0)
        # so the resolver propagates the same diagnostic. A ``None``
        # return means the attr was absent or unparseable.
        transform = transform_from_attr(attr_transform)
        transform_source = 'attr' if transform is not None else None
        if transform is None:
            transform = coords_to_transform(source)
            if transform is not None:
                transform_source = 'coords'

        no_georef_marker = _has_no_georef_marker(source)

        if transform is None and no_georef_marker:
            # ``attrs[_NO_GEOREF_KEY] = True`` -> reader-stamped
            # placeholder. Treat as no-georef regardless of coords.
            status = (
                GEOREF_STATUS_CRS_ONLY if has_crs else GEOREF_STATUS_NONE
            )
            return GeorefResolution(
                transform=None,
                georef_status=status,
                dropped_reason='no_georef_marker',
                applied_no_georef_marker=True,
            )

        if transform is not None:
            # ``coords`` bucket signals "transform derived from coord
            # arrays"; ``transform_only`` signals "transform supplied
            # via attrs['transform']". Either bucket with a CRS
            # rolls up to ``full``.
            if has_crs:
                status = GEOREF_STATUS_FULL
            elif transform_source == 'attr':
                status = GEOREF_STATUS_TRANSFORM_ONLY
            else:
                status = GEOREF_STATUS_COORDS
            return GeorefResolution(
                transform=transform,
                georef_status=status,
                dropped_reason=None,
                applied_no_georef_marker=False,
            )

        # No transform and no marker. The writer entry points run
        # :func:`require_transform_for_georeferenced` after the
        # resolver and raise when spatial coords are present but no
        # transform could be inferred (#1945). We return ``None`` /
        # ``GEOREF_STATUS_NONE`` here and let that guard handle the
        # failure path.
        status = GEOREF_STATUS_CRS_ONLY if has_crs else GEOREF_STATUS_NONE
        return GeorefResolution(
            transform=None,
            georef_status=status,
            dropped_reason='no_transform_inferred',
            applied_no_georef_marker=False,
        )

    # Bare-input path (no DataArray, no GeoInfo). Used by callers that
    # only have CRS and an optional transform hint (e.g. test fixtures
    # exercising the resolver shape).
    transform = transform_from_attr(transform_hint)
    has_crs = _has_crs_signal(crs_epsg, crs_wkt)
    if transform is not None:
        status = GEOREF_STATUS_FULL if has_crs else GEOREF_STATUS_TRANSFORM_ONLY
        return GeorefResolution(
            transform=transform,
            georef_status=status,
            dropped_reason=None,
            applied_no_georef_marker=False,
        )
    status = GEOREF_STATUS_CRS_ONLY if has_crs else GEOREF_STATUS_NONE
    return GeorefResolution(
        transform=None,
        georef_status=status,
        dropped_reason='no_inputs' if not has_crs else 'no_transform_inferred',
        applied_no_georef_marker=False,
    )
