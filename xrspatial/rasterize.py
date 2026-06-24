"""Vector geometry rasterization (polygons, lines, points).

Converts vector geometries (GeoDataFrame or list of (geometry, value) pairs)
to a 2D xr.DataArray.  No GDAL dependency.

- Polygons/MultiPolygons: scanline fill
- Lines/LinearRings/MultiLineStrings: Bresenham line rasterization
- Points/MultiPoints: direct pixel burn

Supports numpy, cupy, dask+numpy, and dask+cupy backends.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from xrspatial.utils import ngjit

try:
    import cupy
except ImportError:
    cupy = None

try:
    import cuspatial  # noqa: F401  -- reserved for future GPU geometry parsing
except ImportError:
    cuspatial = None

#: Cached shapely module, populated by :func:`_require_shapely` on first use.
#: shapely is an optional dependency (the ``vector`` extra). It is imported
#: lazily so ``import xrspatial`` does not pull in shapely (and GEOS) for users
#: who never rasterize vector geometry.
_shapely = None


def _require_shapely():
    """Import shapely or raise a helpful error, caching the module.

    rasterize and polygonize are the only paths that need shapely. Every
    function here that touches the shapely array API calls this and binds the
    return value to a local ``shapely`` name, so the error surfaces clearly
    (including inside dask workers, which call the tile helpers directly rather
    than through :func:`rasterize`).
    """
    global _shapely
    if _shapely is None:
        try:
            import shapely as _s
        except ImportError as e:
            raise ImportError(
                "shapely is required for rasterize/polygonize but is not "
                "installed. Install it with: pip install xarray-spatial[vector]"
            ) from e
        _shapely = _s
    return _shapely


# ---------------------------------------------------------------------------
# Allocation guard: reject output dimensions that would exhaust memory
# ---------------------------------------------------------------------------

#: Default maximum total output pixel count (width * height).
#: ~1 billion pixels, which is ~8 GB for float64 single-band plus an int8
#: ``written`` mask.  Override per call via the ``max_pixels`` keyword.
MAX_PIXELS_DEFAULT = 1_000_000_000


def _check_output_dimensions(width, height, max_pixels):
    """Raise ValueError if the requested output raster exceeds *max_pixels*.

    Called before any host or device allocation so a hostile ``width``,
    ``height``, or ``resolution`` cannot trigger a multi-gigabyte
    ``np.full`` / ``cupy.full`` before the error surfaces.
    """
    total = int(width) * int(height)
    if total > max_pixels:
        raise ValueError(
            f"rasterize output dimensions ({width} x {height} = "
            f"{total:,} pixels) exceed the safety limit of "
            f"{max_pixels:,} pixels.  Pass a larger max_pixels value to "
            f"rasterize() if this size is intentional."
        )


# ---------------------------------------------------------------------------
# Merge functions (CPU, numba-jitted)
#
# Public merge_fn signature (user-supplied callables): see below.
#
# For built-in ``'first'`` / ``'last'`` modes, input order is the source of
# truth: the geometry with the smallest / largest global input index wins
# regardless of which type bucket it falls into.  That decision is made by a
# companion predicate ``should_write(is_first, new_idx, cur_idx)`` which
# gates the call to ``merge_fn``.  When the gate fires for an "ordered
# overwrite" merge, the merge function is just an overwrite, so first and
# last share ``_merge_overwrite``.
#
# Public merge_fn signature: ``merge_fn(pixel, props, is_first) -> float64``
#   pixel    : current pixel value (fill value on first write)
#   props    : 1D float64 array of property values for the geometry
#   is_first : 1 if first write to this pixel, 0 otherwise
# ---------------------------------------------------------------------------

@ngjit
def _merge_overwrite(pixel, props, is_first):
    return props[0]


# Kept for back-compat in case anything imports them by name.
_merge_last = _merge_overwrite
_merge_first = _merge_overwrite


@ngjit
def _merge_max(pixel, props, is_first):
    val = props[0]
    # NaN burn poisons max regardless of order.  Without this branch the
    # outcome would depend on whether the NaN burn happened to land before
    # or after the finite burns ('NaN > finite' is False, so a NaN burn
    # arriving second is silently dropped).  Issue #2255.
    if val != val:
        return val
    if is_first:
        return val
    # If pixel is already NaN from an earlier burn, 'val > NaN' is False
    # below and we fall through to 'return pixel', keeping NaN sticky.
    if val > pixel:
        return val
    return pixel


@ngjit
def _merge_min(pixel, props, is_first):
    val = props[0]
    if val != val:
        return val
    if is_first:
        return val
    if val < pixel:
        return val
    return pixel


@ngjit
def _merge_sum(pixel, props, is_first):
    if is_first:
        return props[0]
    return pixel + props[0]


@ngjit
def _merge_count(pixel, props, is_first):
    if is_first:
        return 1.0
    return pixel + 1.0


# ---------------------------------------------------------------------------
# Ordering predicates: decide whether to call merge_fn given the new
# geometry's global input index and the index that currently owns the pixel.
# ---------------------------------------------------------------------------

@ngjit
def _should_write_any(is_first, new_idx, cur_idx):
    return True


@ngjit
def _should_write_first(is_first, new_idx, cur_idx):
    # 'first' wins on the smallest input index
    return is_first or new_idx < cur_idx


@ngjit
def _should_write_last(is_first, new_idx, cur_idx):
    # 'last' wins on the largest input index
    return is_first or new_idx > cur_idx


# (merge_fn, should_write_fn) tuples, dispatched by merge name.
_MERGE_FUNCTIONS = {
    'last':  (_merge_overwrite, _should_write_last),
    'first': (_merge_overwrite, _should_write_first),
    'max':   (_merge_max,       _should_write_any),
    'min':   (_merge_min,       _should_write_any),
    'sum':   (_merge_sum,       _should_write_any),
    'count': (_merge_count,     _should_write_any),
}


# ---------------------------------------------------------------------------
# Merge pixel helper (CPU)
# ---------------------------------------------------------------------------

@ngjit
def _apply_merge(out, written, order, r, c, props, new_idx,
                 merge_fn, should_write):
    """Write a value into ``out[r, c]`` using the given merge function.

    ``order`` is an int64 array tracking which input-index currently owns
    each pixel.  For order-sensitive merges, ``should_write`` decides
    whether the new geometry beats the current owner; commutative merges
    pass ``_should_write_any`` and behave as before.
    """
    is_first = np.int64(written[r, c] == 0)
    cur_idx = order[r, c]
    if not should_write(is_first, new_idx, cur_idx):
        return
    out[r, c] = merge_fn(out[r, c], props, is_first)
    written[r, c] = 1
    order[r, c] = new_idx


# ---------------------------------------------------------------------------
# Geometry classification (single pass)
# ---------------------------------------------------------------------------

#: Human-readable names for the shapely type ids the fast path reports.
_TYPE_ID_NAMES = {
    0: 'Point', 1: 'LineString', 2: 'LinearRing', 3: 'Polygon',
    4: 'MultiPoint', 5: 'MultiLineString', 6: 'MultiPolygon',
    7: 'GeometryCollection',
}


def _warn_dropped_geometries(types):
    """Warn that non-empty geometries of unsupported types are being dropped.

    ``types`` is an iterable of either shapely type ids (ints, from the
    vectorized fast path) or ``geom_type`` names (strings, from the recursive
    slow path).  Names are resolved and de-duplicated so the warning lists each
    dropped type once.
    """
    names = sorted({
        _TYPE_ID_NAMES.get(int(t), f'type id {int(t)}')
        if isinstance(t, (int, np.integer)) else str(t)
        for t in types
    })
    warnings.warn(
        f"rasterize: dropping unsupported non-empty geometr"
        f"{'y' if len(names) == 1 else 'ies'} of type "
        f"{', '.join(names)}; only points, lines, and polygons are "
        f"rasterized.",
        stacklevel=2,
    )


def _warn_nonfinite_geometries(count):
    """Warn that geometries with non-finite coordinates are being dropped.

    A NaN or infinite coordinate has no defined location on the raster
    grid; letting it through poisons the pixel-space int casts downstream
    (issue #3295).
    """
    warnings.warn(
        f"rasterize: dropping {count} geometr"
        f"{'y' if count == 1 else 'ies'} with non-finite coordinates "
        f"(NaN or infinity); they have no defined location on the "
        f"raster grid.",
        stacklevel=2,
    )


def _classify_geometries(geometries, props_array):
    """Classify geometries by type in a single pass.

    Two internal paths: a vectorized fast path that uses shapely 2.0
    array ops (``get_type_id`` / ``is_empty``) for the common case of
    no ``GeometryCollection`` inputs, and a recursive slow path that
    unpacks nested collections so their contents are rasterized rather
    than silently dropped. The slow path runs only when at least one
    input is a ``GeometryCollection``; both paths produce identical
    output shapes.

    Also tracks each polygon's input index so the scanline fill can
    process geometries in input order (needed for first/last merge).

    Parameters
    ----------
    geometries : list of shapely geometries
    props_array : (N, P) float64 array of property values

    Returns
    -------
    (poly_geoms, poly_props, poly_ids, poly_global_idx),
    (line_geoms, line_props, line_global_idx),
    (point_geoms, point_props, point_global_idx)

    Where ``*_props`` is an (N, P) float64 array, ``poly_ids`` is the
    local-to-edges integer keyed by scanline (still 0..N_poly-1), and
    each ``*_global_idx`` is an int64 array mapping the per-type local
    index back to the original input position.  The global index is what
    decides ordered merges (`first`, `last`) across geometry types.
    """
    n_props = props_array.shape[1] if props_array.ndim == 2 else 1
    geom_arr = np.array(geometries, dtype=object)
    n = len(geom_arr)

    if n == 0:
        empty_props = np.empty((0, n_props), dtype=np.float64)
        empty_idx = np.empty(0, dtype=np.int64)
        return (([], empty_props, [], empty_idx),
                ([], empty_props.copy(), empty_idx.copy()),
                ([], empty_props.copy(), empty_idx.copy()))

    shapely = _require_shapely()
    type_ids = shapely.get_type_id(geom_arr)
    empty = shapely.is_empty(geom_arr)
    # ``get_type_id`` returns -1 for None/missing geometries (which report as
    # non-empty); treat those as empty so they are skipped like the slow path's
    # ``geom is None`` guard rather than warned about as an unsupported type.
    valid = ~empty & (type_ids >= 0)

    # Drop geometries containing non-finite coordinates (NaN / inf).
    # They have no defined raster location, and letting them through
    # poisons the pixel-space float -> int casts downstream: a NaN
    # vertex survives the horizontal-edge filter (NaN != NaN) and turns
    # into a phantom full-height edge, while an inf vertex burns pixels
    # outside the polygon's real extent (issue #3295).  GEOS skips NaN
    # coordinates when computing bounds, so a bbox check cannot detect
    # these; inspect the coordinates directly.  GeometryCollections are
    # excluded here -- their members are checked per-leaf in the slow
    # path below, which is the only path that unpacks them.  When any
    # GeometryCollection is present the slow path runs over every input
    # and owns both the per-leaf check and the warning, so the
    # vectorized check is skipped entirely to avoid warning twice for
    # the same geometry.
    if not np.any(valid & (type_ids == 7)):
        coords_all, coord_owner = shapely.get_coordinates(
            geom_arr, return_index=True)
        nonfinite_coord = ~np.all(np.isfinite(coords_all), axis=1)
        if np.any(nonfinite_coord):
            nonfinite_geom = np.zeros(n, dtype=bool)
            nonfinite_geom[coord_owner[nonfinite_coord]] = True
            nonfinite_geom &= valid
            if np.any(nonfinite_geom):
                _warn_nonfinite_geometries(
                    int(np.count_nonzero(nonfinite_geom)))
                valid = valid & ~nonfinite_geom

    # Type ID mapping:
    # 0=Point, 1=LineString, 2=LinearRing, 3=Polygon,
    # 4=MultiPoint, 5=MultiLineString, 6=MultiPolygon,
    # 7=GeometryCollection
    # LinearRing (2) is a closed line; rasterize it like any other line.
    poly_mask = valid & ((type_ids == 3) | (type_ids == 6))
    line_mask = valid & ((type_ids == 1) | (type_ids == 2) | (type_ids == 5))
    point_mask = valid & ((type_ids == 0) | (type_ids == 4))
    gc_mask = valid & (type_ids == 7)

    # Warn before dropping any non-empty geometry that matches no bucket, so a
    # caller never loses data silently.
    dropped_mask = valid & ~(poly_mask | line_mask | point_mask | gc_mask)
    if np.any(dropped_mask):
        _warn_dropped_geometries(np.unique(type_ids[dropped_mask]))

    # Fast path: no GeometryCollections (common case)
    if not np.any(gc_mask):
        poly_idx = np.where(poly_mask)[0]
        line_idx = np.where(line_mask)[0]
        point_idx = np.where(point_mask)[0]

        poly_geoms = [geometries[i] for i in poly_idx]
        poly_ids = list(range(len(poly_idx)))
        poly_props = (props_array[poly_idx] if len(poly_idx) > 0
                      else np.empty((0, n_props), dtype=np.float64))
        poly_global = poly_idx.astype(np.int64)

        line_geoms = [geometries[i] for i in line_idx]
        line_props = (props_array[line_idx] if len(line_idx) > 0
                      else np.empty((0, n_props), dtype=np.float64))
        line_global = line_idx.astype(np.int64)

        point_geoms = [geometries[i] for i in point_idx]
        point_props = (props_array[point_idx] if len(point_idx) > 0
                       else np.empty((0, n_props), dtype=np.float64))
        point_global = point_idx.astype(np.int64)

        return ((poly_geoms, poly_props, poly_ids, poly_global),
                (line_geoms, line_props, line_global),
                (point_geoms, point_props, point_global))

    # Slow path: recursively unpack GeometryCollections before classifying.
    # Only taken when at least one input is a GeometryCollection.
    poly_geoms, poly_prop_rows, poly_ids = [], [], []
    poly_global, line_global, point_global = [], [], []
    line_geoms, line_prop_rows = [], []
    point_geoms, point_prop_rows = [], []
    poly_counter = 0
    nonfinite_count = 0

    def _classify_one(geom, prop_row, global_idx):
        nonlocal poly_counter, nonfinite_count
        if geom is None or geom.is_empty:
            return
        gt = geom.geom_type
        # Per-leaf counterpart of the fast path's non-finite coordinate
        # drop (issue #3295).  Collections recurse below, so each member
        # is checked individually rather than dropping the whole
        # collection for one bad leaf.
        if gt != 'GeometryCollection' and not np.all(
                np.isfinite(shapely.get_coordinates(geom))):
            nonfinite_count += 1
            return
        if gt in ('Polygon', 'MultiPolygon'):
            poly_geoms.append(geom)
            poly_prop_rows.append(prop_row)
            poly_ids.append(poly_counter)
            poly_global.append(global_idx)
            poly_counter += 1
        elif gt in ('LineString', 'LinearRing', 'MultiLineString'):
            line_geoms.append(geom)
            line_prop_rows.append(prop_row)
            line_global.append(global_idx)
        elif gt in ('Point', 'MultiPoint'):
            point_geoms.append(geom)
            point_prop_rows.append(prop_row)
            point_global.append(global_idx)
        elif gt == 'GeometryCollection':
            for sub in geom.geoms:
                _classify_one(sub, prop_row, global_idx)
        else:
            _warn_dropped_geometries([gt])

    for idx, geom in enumerate(geometries):
        _classify_one(geom, props_array[idx], idx)

    if nonfinite_count:
        _warn_nonfinite_geometries(nonfinite_count)

    def _to_2d(rows):
        if rows:
            return np.array(rows, dtype=np.float64)
        return np.empty((0, n_props), dtype=np.float64)

    def _to_i64(lst):
        return np.array(lst, dtype=np.int64) if lst else np.empty(0, dtype=np.int64)

    return ((poly_geoms, _to_2d(poly_prop_rows), poly_ids, _to_i64(poly_global)),
            (line_geoms, _to_2d(line_prop_rows), _to_i64(line_global)),
            (point_geoms, _to_2d(point_prop_rows), _to_i64(point_global)))


# ---------------------------------------------------------------------------
# Edge table construction
# ---------------------------------------------------------------------------

_EMPTY_EDGES = (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64), np.empty(0, np.float64),
                np.empty(0, np.int32))


def _extract_edges(geometries, geom_ids, bounds, height, width,
                   all_touched=False):
    """Build the edge table for polygon scanline fill.

    Returns
    -------
    edge_y_min, edge_y_max : int32 arrays
    edge_x_at_ymin, edge_inv_slope : float64 arrays
    edge_geom_id : int32 array -- input geometry index for ordering
    """
    if not geometries:
        return _EMPTY_EDGES
    return _extract_edges_vectorized(
        geometries, geom_ids, bounds, height, width, all_touched)


def _extract_edges_vectorized(geometries, geom_ids, bounds,
                              height, width, all_touched):
    """Vectorized edge extraction using shapely 2.0 array ops."""
    shapely = _require_shapely()
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    geom_arr = np.array(geometries, dtype=object)
    id_arr = np.array(geom_ids, dtype=np.int32)

    # Explode MultiPolygons to individual Polygons
    parts, part_idx = shapely.get_parts(geom_arr, return_index=True)
    part_ids = id_arr[part_idx]

    # Get all rings (exterior + interior)
    rings, ring_idx = shapely.get_rings(parts, return_index=True)
    ring_ids = part_ids[ring_idx]

    if len(rings) == 0:
        return _EMPTY_EDGES

    # Get all vertex coordinates with ring membership
    coords, coord_ring_idx = shapely.get_coordinates(
        rings, return_index=True)
    n_coords = len(coords)
    if n_coords < 2:
        return _EMPTY_EDGES

    # Mark last coordinate of each ring (don't form cross-ring edges)
    is_last = np.zeros(n_coords, dtype=bool)
    changes = np.nonzero(np.diff(coord_ring_idx))[0]
    is_last[changes] = True
    is_last[-1] = True

    # Edges: from each non-last coordinate to its successor
    start_idx = np.nonzero(~is_last)[0]
    end_idx = start_idx + 1

    # Geometry id for each edge
    edge_ids = ring_ids[coord_ring_idx[start_idx]]

    # Convert to pixel space with half-pixel offset so that integer
    # positions correspond to pixel *centers* (not edges).  Without
    # this shift the scanline fill samples at pixel boundaries, which
    # causes an off-by-one asymmetry: the top/left edges of the
    # raster lose a row/column compared to the bottom/right.
    sr = (ymax - coords[start_idx, 1]) / py - 0.5
    sc = (coords[start_idx, 0] - xmin) / px - 0.5
    er = (ymax - coords[end_idx, 1]) / py - 0.5
    ec = (coords[end_idx, 0] - xmin) / px - 0.5

    # Drop horizontal edges (filter in-place)
    not_horiz = sr != er
    sr = sr[not_horiz]
    sc = sc[not_horiz]
    er = er[not_horiz]
    ec = ec[not_horiz]
    edge_ids = edge_ids[not_horiz]

    if len(sr) == 0:
        return _EMPTY_EDGES

    # Orient edges so top_r < bot_r, compute derived values, then
    # filter.  We reuse short names and delete intermediates early
    # to keep peak memory down for large edge counts.
    swap = sr > er
    top_r = np.where(swap, er, sr)
    top_c = np.where(swap, ec, sc)
    bot_r = np.where(swap, sr, er)
    bot_c = np.where(swap, sc, ec)
    del sr, sc, er, ec, swap

    # Inverse slope and row clamping (compute before filtering so
    # the valid mask can be applied once at the end).
    dr = bot_r - top_r  # guaranteed != 0
    inv_slope = (bot_c - top_c) / dr
    del bot_c

    # Clamp the cast inputs to a narrow band around the raster before
    # converting to int32.  Row values beyond +/-2**31 otherwise
    # overflow the cast (numpy lands them on INT_MIN, and the ``- 1``
    # below wraps INT_MIN to INT_MAX), turning an edge that lies
    # entirely off-raster into a phantom full-height edge instead of
    # being filtered by the ``ry_min <= ry_max`` check (issue #3295).
    # Clamping is exact for the visible region: any value at or beyond
    # the band edges produces the same ry_min / ry_max as the original
    # out-of-range value would have, without the overflow.  Only the
    # cast inputs are clamped -- ``x_at_ymin`` below extrapolates from
    # the original ``top_r`` so x intersections stay correct for edges
    # entering the raster from far away.
    top_r_cast = np.clip(top_r, -2.0, float(height) + 2.0)
    bot_r_cast = np.clip(bot_r, -2.0, float(height) + 2.0)
    if all_touched:
        ry_min = np.maximum(np.floor(top_r_cast - 0.5).astype(np.int32), 0)
        ry_max = np.minimum(
            np.ceil(bot_r_cast + 0.5).astype(np.int32) - 1, height - 1)
    else:
        ry_min = np.maximum(np.ceil(top_r_cast).astype(np.int32), 0)
        ry_max = np.minimum(
            np.ceil(bot_r_cast).astype(np.int32) - 1, height - 1)
    del bot_r, top_r_cast, bot_r_cast

    x_at_ymin = top_c + (ry_min.astype(np.float64) - top_r) * inv_slope
    del top_c, top_r

    # Single filter pass at the end
    valid = ry_min <= ry_max
    return (ry_min[valid],
            ry_max[valid],
            x_at_ymin[valid],
            inv_slope[valid],
            edge_ids[valid])


def _sort_edges(edge_arrays):
    """Sort edge table by y_min for scanline early termination."""
    if len(edge_arrays[0]) == 0:
        return edge_arrays
    order = np.argsort(edge_arrays[0], kind='stable')
    return tuple(arr[order] for arr in edge_arrays)


# ---------------------------------------------------------------------------
# Point extraction (always on host)
# ---------------------------------------------------------------------------

def _extract_points(geometries, bounds, height, width):
    """Parse Point/MultiPoint geometries into pixel coordinate arrays.

    Returns (rows, cols, geom_idx) where geom_idx is int32 indices into
    the geometry list (and thus into the per-type props table).
    """
    if not geometries:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.int32))
    return _extract_points_vectorized(
        geometries, bounds, height, width)


def _extract_points_vectorized(geometries, bounds, height, width):
    """Vectorized point extraction using shapely 2.0 array ops."""
    shapely = _require_shapely()
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    geom_arr = np.array(geometries, dtype=object)
    idx_arr = np.arange(len(geometries), dtype=np.int32)

    # Explode MultiPoints to individual Points
    parts, part_idx = shapely.get_parts(geom_arr, return_index=True)
    part_geom_idx = idx_arr[part_idx]

    if len(parts) == 0:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.int32))

    # Extract coordinates with index back to each point
    coords, coord_idx = shapely.get_coordinates(
        parts, return_index=True)
    pt_geom_idx = part_geom_idx[coord_idx]

    cols = np.floor((coords[:, 0] - xmin) / px).astype(np.int32)
    rows = np.floor((ymax - coords[:, 1]) / py).astype(np.int32)

    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return (rows[valid], cols[valid], pt_geom_idx[valid])


# ---------------------------------------------------------------------------
# Line segment extraction (always on host)
# ---------------------------------------------------------------------------

_EMPTY_LINES = (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.int32))

# Float-coordinate variant used by the supercover boundary burn.
# Endpoints are kept in subpixel space so grid traversal can pick up
# every cell a segment crosses, not just the dominant-axis steps.
_EMPTY_LINES_FLOAT = (np.empty(0, np.float64), np.empty(0, np.float64),
                      np.empty(0, np.float64), np.empty(0, np.float64),
                      np.empty(0, np.int32))


def _extract_line_segments(geometries, bounds, height, width):
    """Parse LineString/MultiLineString geometries into pixel-space segments.

    Segments are clipped to the raster extent before conversion to pixel
    coordinates, so Bresenham never iterates over out-of-bounds pixels.

    Returns (r0, c0, r1, c1, geom_idx) where geom_idx is int32 indices
    into the geometry list (and thus into the per-type props table).
    """
    if not geometries:
        return _EMPTY_LINES
    return _extract_lines_vectorized(
        geometries, bounds, height, width)


def _extract_lines_vectorized(geometries, bounds, height, width):
    """Vectorized line extraction with Liang-Barsky clipping."""
    shapely = _require_shapely()
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    geom_arr = np.array(geometries, dtype=object)
    idx_arr = np.arange(len(geometries), dtype=np.int32)

    # Explode MultiLineStrings to individual LineStrings
    parts, part_idx = shapely.get_parts(geom_arr, return_index=True)
    part_geom_idx = idx_arr[part_idx]

    if len(parts) == 0:
        return _EMPTY_LINES

    # Get all vertex coordinates with line membership
    coords, coord_line_idx = shapely.get_coordinates(
        parts, return_index=True)
    n_coords = len(coords)
    if n_coords < 2:
        return _EMPTY_LINES

    # Mark last coordinate of each line (don't form cross-line segments)
    is_last = np.zeros(n_coords, dtype=bool)
    changes = np.nonzero(np.diff(coord_line_idx))[0]
    is_last[changes] = True
    is_last[-1] = True

    # Segments: from each non-last coordinate to its successor
    start_idx = np.nonzero(~is_last)[0]
    end_idx = start_idx + 1
    seg_geom_idx = part_geom_idx[coord_line_idx[start_idx]]

    # World-space segment endpoints
    x0 = coords[start_idx, 0]
    y0 = coords[start_idx, 1]
    x1 = coords[end_idx, 0]
    y1 = coords[end_idx, 1]

    # Vectorized Liang-Barsky clip to raster bounds
    dx = x1 - x0
    dy = y1 - y0

    # p and q arrays: shape (4, n_segments)
    p = np.array([-dx, dx, -dy, dy])
    q = np.array([x0 - xmin, xmax - x0, y0 - ymin, ymax - y0])

    t0 = np.zeros(len(x0))
    t1 = np.ones(len(x0))
    valid = np.ones(len(x0), dtype=bool)

    for i in range(4):
        parallel = p[i] == 0.0
        outside = parallel & (q[i] < 0.0)
        valid &= ~outside

        neg = (~parallel) & (p[i] < 0.0)
        pos = (~parallel) & (p[i] > 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            t_neg = np.where(neg, q[i] / p[i], 0.0)
            t_pos = np.where(pos, q[i] / p[i], 1.0)

        t0 = np.where(neg, np.maximum(t0, t_neg), t0)
        t1 = np.where(pos, np.minimum(t1, t_pos), t1)

    valid &= (t0 <= t1)

    # Apply clipping
    cx0 = x0 + t0 * dx
    cy0 = y0 + t0 * dy
    cx1 = x0 + t1 * dx
    cy1 = y0 + t1 * dy

    # Convert to pixel space and floor to int32
    r0 = np.floor((ymax - cy0) / py).astype(np.int32)
    c0 = np.floor((cx0 - xmin) / px).astype(np.int32)
    r1 = np.floor((ymax - cy1) / py).astype(np.int32)
    c1 = np.floor((cx1 - xmin) / px).astype(np.int32)

    # Clamp edge cases (clipping guarantees in-bounds but float rounding
    # at exact boundaries can produce height or width)
    np.clip(r0, 0, height - 1, out=r0)
    np.clip(c0, 0, width - 1, out=c0)
    np.clip(r1, 0, height - 1, out=r1)
    np.clip(c1, 0, width - 1, out=c1)

    v = valid
    return (r0[v], c0[v], r1[v], c1[v], seg_geom_idx[v])


def _extract_line_segments_float(geometries, bounds, height, width):
    """LineString/MultiLineString segments in float pixel space.

    Variant of :func:`_extract_line_segments` that keeps sub-pixel
    precision so the supercover grid walker can pick up every cell a
    segment crosses (issue #3102, the line counterpart to the polygon
    boundary path).  World-space clipping uses the same Liang-Barsky
    pass; the result is converted to pixel coordinates
    (``cx`` = x in pixels, ``cy`` = y in pixels, increasing downward)
    without rounding.

    Returns ``(cx0, cy0, cx1, cy1, geom_idx)`` as float64 arrays (and
    int32 ids into the per-type props table).
    """
    if not geometries:
        return _EMPTY_LINES_FLOAT

    shapely = _require_shapely()
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    geom_arr = np.array(geometries, dtype=object)
    idx_arr = np.arange(len(geometries), dtype=np.int32)

    # Explode MultiLineStrings to individual LineStrings
    parts, part_idx = shapely.get_parts(geom_arr, return_index=True)
    part_geom_idx = idx_arr[part_idx]

    if len(parts) == 0:
        return _EMPTY_LINES_FLOAT

    coords, coord_line_idx = shapely.get_coordinates(
        parts, return_index=True)
    n_coords = len(coords)
    if n_coords < 2:
        return _EMPTY_LINES_FLOAT

    # Mark last coordinate of each line (don't form cross-line segments)
    is_last = np.zeros(n_coords, dtype=bool)
    changes = np.nonzero(np.diff(coord_line_idx))[0]
    is_last[changes] = True
    is_last[-1] = True

    start_idx = np.nonzero(~is_last)[0]
    end_idx = start_idx + 1
    seg_geom_idx = part_geom_idx[coord_line_idx[start_idx]].astype(np.int32)

    x0 = coords[start_idx, 0].astype(np.float64)
    y0 = coords[start_idx, 1].astype(np.float64)
    x1 = coords[end_idx, 0].astype(np.float64)
    y1 = coords[end_idx, 1].astype(np.float64)

    # Vectorized Liang-Barsky clip to raster world-space bounds.
    dx = x1 - x0
    dy = y1 - y0
    p = np.array([-dx, dx, -dy, dy])
    q = np.array([x0 - xmin, xmax - x0, y0 - ymin, ymax - y0])

    t0 = np.zeros(len(x0))
    t1 = np.ones(len(x0))
    valid = np.ones(len(x0), dtype=bool)

    for i in range(4):
        parallel = p[i] == 0.0
        valid &= ~(parallel & (q[i] < 0.0))
        neg = (~parallel) & (p[i] < 0.0)
        pos = (~parallel) & (p[i] > 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_neg = np.where(neg, q[i] / p[i], 0.0)
            t_pos = np.where(pos, q[i] / p[i], 1.0)
        t0 = np.where(neg, np.maximum(t0, t_neg), t0)
        t1 = np.where(pos, np.minimum(t1, t_pos), t1)

    valid &= (t0 <= t1)

    cx0_w = x0 + t0 * dx
    cy0_w = y0 + t0 * dy
    cx1_w = x0 + t1 * dx
    cy1_w = y0 + t1 * dy

    # World -> float pixel coordinates. ymax is the top of the raster,
    # row 0 is at the top, so y increases downward in pixel space.
    cx0 = (cx0_w - xmin) / px
    cy0 = (ymax - cy0_w) / py
    cx1 = (cx1_w - xmin) / px
    cy1 = (ymax - cy1_w) / py

    v = valid
    return (cx0[v], cy0[v], cx1[v], cy1[v], seg_geom_idx[v])


# ---------------------------------------------------------------------------
# Polygon boundary segments (for all_touched mode)
# ---------------------------------------------------------------------------

def _extract_polygon_boundary_segments(geometries, geom_ids, bounds,
                                       height, width):
    """Extract polygon ring edges as line segments for Bresenham drawing.

    Used by all_touched mode: drawing the boundary ensures every pixel
    the polygon touches is burned, without expanding scanline edge
    y-ranges (which breaks edge pairing).

    Extracts ring coordinates directly (no intermediate LineString objects)
    and runs vectorized Liang-Barsky clipping to produce pixel-space
    segments.

    Returns (r0, c0, r1, c1, geom_idx) where geom_idx maps each segment
    back to the polygon's index in geom_ids (for props table lookup).
    """
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    # Collect all ring vertex arrays and the polygon id for each ring
    all_coords = []   # list of (N, 2) arrays
    all_ids = []      # polygon id repeated per segment in each ring
    for geom, gid in zip(geometries, geom_ids):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'Polygon':
            parts = [geom]
        elif geom.geom_type == 'MultiPolygon':
            parts = list(geom.geoms)
        else:
            continue
        for poly in parts:
            coords = np.asarray(poly.exterior.coords)
            n = len(coords) - 1  # segments in this ring
            if n > 0:
                all_coords.append(coords)
                all_ids.append(np.full(n, gid, dtype=np.int32))
            for interior in poly.interiors:
                coords = np.asarray(interior.coords)
                n = len(coords) - 1
                if n > 0:
                    all_coords.append(coords)
                    all_ids.append(np.full(n, gid, dtype=np.int32))

    if not all_coords:
        return _EMPTY_LINES

    # Build segment arrays: consecutive vertex pairs within each ring
    seg_x0, seg_y0, seg_x1, seg_y1 = [], [], [], []
    for coords in all_coords:
        seg_x0.append(coords[:-1, 0])
        seg_y0.append(coords[:-1, 1])
        seg_x1.append(coords[1:, 0])
        seg_y1.append(coords[1:, 1])

    x0 = np.concatenate(seg_x0)
    y0 = np.concatenate(seg_y0)
    x1 = np.concatenate(seg_x1)
    y1 = np.concatenate(seg_y1)
    seg_ids = np.concatenate(all_ids)

    # Vectorized Liang-Barsky clip to raster bounds
    dx = x1 - x0
    dy = y1 - y0
    p = np.array([-dx, dx, -dy, dy])
    q = np.array([x0 - xmin, xmax - x0, y0 - ymin, ymax - y0])

    t0 = np.zeros(len(x0))
    t1 = np.ones(len(x0))
    valid = np.ones(len(x0), dtype=bool)

    for i in range(4):
        parallel = p[i] == 0.0
        valid &= ~(parallel & (q[i] < 0.0))
        neg = (~parallel) & (p[i] < 0.0)
        pos = (~parallel) & (p[i] > 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_neg = np.where(neg, q[i] / p[i], 0.0)
            t_pos = np.where(pos, q[i] / p[i], 1.0)
        t0 = np.where(neg, np.maximum(t0, t_neg), t0)
        t1 = np.where(pos, np.minimum(t1, t_pos), t1)

    valid &= (t0 <= t1)

    cx0 = x0 + t0 * dx
    cy0 = y0 + t0 * dy
    cx1 = x0 + t1 * dx
    cy1 = y0 + t1 * dy

    r0 = np.floor((ymax - cy0) / py).astype(np.int32)
    c0 = np.floor((cx0 - xmin) / px).astype(np.int32)
    r1 = np.floor((ymax - cy1) / py).astype(np.int32)
    c1 = np.floor((cx1 - xmin) / px).astype(np.int32)

    np.clip(r0, 0, height - 1, out=r0)
    np.clip(c0, 0, width - 1, out=c0)
    np.clip(r1, 0, height - 1, out=r1)
    np.clip(c1, 0, width - 1, out=c1)

    v = valid
    return (r0[v], c0[v], r1[v], c1[v], seg_ids[v])


def _extract_polygon_boundary_segments_float(geometries, geom_ids, bounds,
                                             height, width):
    """Polygon ring edges as float pixel-space segments.

    Variant of :func:`_extract_polygon_boundary_segments` that keeps
    sub-pixel precision so a supercover line walker can pick up every
    cell crossed by the segment. World-space clipping uses Liang-Barsky
    against the raster bounds; the result is converted to pixel
    coordinates (col = x in pixels, row = y in pixels) without rounding.

    Returns
    -------
    (cx0, cy0, cx1, cy1, seg_ids) as float64 arrays (and int32 ids).
    ``cy`` increases downward (matches the raster row axis); the y-flip
    against world coordinates is applied here so the grid traversal
    kernel does not need to know about it.
    """
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    all_coords = []
    all_ids = []
    for geom, gid in zip(geometries, geom_ids):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'Polygon':
            parts = [geom]
        elif geom.geom_type == 'MultiPolygon':
            parts = list(geom.geoms)
        else:
            continue
        for poly in parts:
            coords = np.asarray(poly.exterior.coords)
            n = len(coords) - 1
            if n > 0:
                all_coords.append(coords)
                all_ids.append(np.full(n, gid, dtype=np.int32))
            for interior in poly.interiors:
                coords = np.asarray(interior.coords)
                n = len(coords) - 1
                if n > 0:
                    all_coords.append(coords)
                    all_ids.append(np.full(n, gid, dtype=np.int32))

    if not all_coords:
        return _EMPTY_LINES_FLOAT

    seg_x0, seg_y0, seg_x1, seg_y1 = [], [], [], []
    for coords in all_coords:
        seg_x0.append(coords[:-1, 0])
        seg_y0.append(coords[:-1, 1])
        seg_x1.append(coords[1:, 0])
        seg_y1.append(coords[1:, 1])

    x0 = np.concatenate(seg_x0).astype(np.float64)
    y0 = np.concatenate(seg_y0).astype(np.float64)
    x1 = np.concatenate(seg_x1).astype(np.float64)
    y1 = np.concatenate(seg_y1).astype(np.float64)
    seg_ids = np.concatenate(all_ids)

    # Vectorized Liang-Barsky clip to raster world-space bounds.
    dx = x1 - x0
    dy = y1 - y0
    p = np.array([-dx, dx, -dy, dy])
    q = np.array([x0 - xmin, xmax - x0, y0 - ymin, ymax - y0])

    t0 = np.zeros(len(x0))
    t1 = np.ones(len(x0))
    valid = np.ones(len(x0), dtype=bool)

    for i in range(4):
        parallel = p[i] == 0.0
        valid &= ~(parallel & (q[i] < 0.0))
        neg = (~parallel) & (p[i] < 0.0)
        pos = (~parallel) & (p[i] > 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_neg = np.where(neg, q[i] / p[i], 0.0)
            t_pos = np.where(pos, q[i] / p[i], 1.0)
        t0 = np.where(neg, np.maximum(t0, t_neg), t0)
        t1 = np.where(pos, np.minimum(t1, t_pos), t1)

    valid &= (t0 <= t1)

    cx0_w = x0 + t0 * dx
    cy0_w = y0 + t0 * dy
    cx1_w = x0 + t1 * dx
    cy1_w = y0 + t1 * dy

    # World -> float pixel coordinates. ymax is the top of the raster,
    # row 0 is at the top, so y increases downward in pixel space.
    cx0 = (cx0_w - xmin) / px
    cy0 = (ymax - cy0_w) / py
    cx1 = (cx1_w - xmin) / px
    cy1 = (ymax - cy1_w) / py

    v = valid
    return (cx0[v], cy0[v], cx1[v], cy1[v], seg_ids[v])


def _extract_polygon_boundary_segments_float_global(
        geometries, geom_ids, full_bounds, full_height, full_width,
        tile_bounds, row_off, col_off):
    """Polygon ring edges as float pixel-space segments in tile-local space,
    computed in the *global* grid frame so floor() ties match the eager path.

    This is the dask counterpart of
    :func:`_extract_polygon_boundary_segments_float`.  The eager backend
    converts world coordinates to pixel coordinates against the full
    raster origin; a dask tile that re-derives px/py and the origin from
    its own ``tile_bounds`` reintroduces a different floating-point
    rounding, so a boundary segment landing on an exact integer pixel
    row in the full grid (e.g. ``(10.0 - 4.0)/0.4 == 15.0``) can land at
    ``1.9999999999999996`` in tile-local space and ``floor`` to the wrong
    cell.  The supercover walk then burns a different row/column than the
    eager backend, breaking the documented pixel-for-pixel parity under
    ``all_touched=True`` (issue #3384).

    To match the eager backend bit-for-bit, the world->pixel conversion
    uses the full raster's ``px`` / ``py`` and origin (so the float
    pixel coordinates are identical to what the eager
    :func:`_extract_polygon_boundary_segments_float` produces for the
    same vertex), then subtracts the *integer* tile offset
    (``row_off`` / ``col_off``).  Integer translation is exact, so the
    fractional part -- and therefore ``floor`` -- is preserved.

    Clipping is still done in world space against ``tile_bounds`` so only
    the ring segments overlapping this tile are walked; the clipped
    endpoints are then mapped through the full-grid conversion.

    Returns ``(cx0, cy0, cx1, cy1, seg_ids)`` in tile-local float pixel
    coordinates, matching the signature of
    :func:`_extract_polygon_boundary_segments_float`.
    """
    fxmin, fymin, fxmax, fymax = full_bounds
    px = (fxmax - fxmin) / full_width
    py = (fymax - fymin) / full_height
    txmin, tymin, txmax, tymax = tile_bounds

    all_coords = []
    all_ids = []
    for geom, gid in zip(geometries, geom_ids):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'Polygon':
            parts = [geom]
        elif geom.geom_type == 'MultiPolygon':
            parts = list(geom.geoms)
        else:
            continue
        for poly in parts:
            coords = np.asarray(poly.exterior.coords)
            n = len(coords) - 1
            if n > 0:
                all_coords.append(coords)
                all_ids.append(np.full(n, gid, dtype=np.int32))
            for interior in poly.interiors:
                coords = np.asarray(interior.coords)
                n = len(coords) - 1
                if n > 0:
                    all_coords.append(coords)
                    all_ids.append(np.full(n, gid, dtype=np.int32))

    if not all_coords:
        return _EMPTY_LINES_FLOAT

    seg_x0, seg_y0, seg_x1, seg_y1 = [], [], [], []
    for coords in all_coords:
        seg_x0.append(coords[:-1, 0])
        seg_y0.append(coords[:-1, 1])
        seg_x1.append(coords[1:, 0])
        seg_y1.append(coords[1:, 1])

    x0 = np.concatenate(seg_x0).astype(np.float64)
    y0 = np.concatenate(seg_y0).astype(np.float64)
    x1 = np.concatenate(seg_x1).astype(np.float64)
    y1 = np.concatenate(seg_y1).astype(np.float64)
    seg_ids = np.concatenate(all_ids)

    # Vectorized Liang-Barsky clip to this tile's world-space bounds.
    dx = x1 - x0
    dy = y1 - y0
    p = np.array([-dx, dx, -dy, dy])
    q = np.array([x0 - txmin, txmax - x0, y0 - tymin, tymax - y0])

    t0 = np.zeros(len(x0))
    t1 = np.ones(len(x0))
    valid = np.ones(len(x0), dtype=bool)

    for i in range(4):
        parallel = p[i] == 0.0
        valid &= ~(parallel & (q[i] < 0.0))
        neg = (~parallel) & (p[i] < 0.0)
        pos = (~parallel) & (p[i] > 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_neg = np.where(neg, q[i] / p[i], 0.0)
            t_pos = np.where(pos, q[i] / p[i], 1.0)
        t0 = np.where(neg, np.maximum(t0, t_neg), t0)
        t1 = np.where(pos, np.minimum(t1, t_pos), t1)

    valid &= (t0 <= t1)

    cx0_w = x0 + t0 * dx
    cy0_w = y0 + t0 * dy
    cx1_w = x0 + t1 * dx
    cy1_w = y0 + t1 * dy

    # World -> float pixel coordinates in the FULL grid frame (identical
    # to the eager path), then shift to tile-local space by the integer
    # tile offset.  The integer subtraction does not perturb the
    # fractional part, so floor() ties match the eager walk.
    cx0 = (cx0_w - fxmin) / px - col_off
    cy0 = (fymax - cy0_w) / py - row_off
    cx1 = (cx1_w - fxmin) / px - col_off
    cy1 = (fymax - cy1_w) / py - row_off

    v = valid
    return (cx0[v], cy0[v], cx1[v], cy1[v], seg_ids[v])


# ---------------------------------------------------------------------------
# CPU burn kernels (numba)
# ---------------------------------------------------------------------------

@ngjit
def _burn_points_cpu(out, written, rows, cols, geom_idx, geom_props,
                     geom_global_idx, merge_fn, should_write, order):
    for i in range(len(rows)):
        r = rows[i]
        c = cols[i]
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            gi = geom_idx[i]
            _apply_merge(out, written, order, r, c, geom_props[gi],
                         geom_global_idx[gi], merge_fn, should_write)


@ngjit
def _burn_lines_cpu(out, written, r0_arr, c0_arr, r1_arr, c1_arr, geom_idx,
                    geom_props, height, width, geom_global_idx,
                    merge_fn, should_write, order):
    for i in range(len(r0_arr)):
        r0 = r0_arr[i]
        c0 = c0_arr[i]
        r1 = r1_arr[i]
        c1 = c1_arr[i]
        gi = geom_idx[i]
        props = geom_props[gi]
        new_idx = geom_global_idx[gi]

        dr = r1 - r0
        dc = c1 - c0
        sr = 1 if dr >= 0 else -1
        sc = 1 if dc >= 0 else -1
        dr = dr * sr
        dc = dc * sc

        if dr >= dc:
            err = dc - dr
            r, c = r0, c0
            for _ in range(dr + 1):
                if 0 <= r < height and 0 <= c < width:
                    _apply_merge(out, written, order, r, c, props,
                                 new_idx, merge_fn, should_write)
                if err >= 0:
                    c += sc
                    err -= dr
                r += sr
                err += dc
        else:
            err = dr - dc
            r, c = r0, c0
            for _ in range(dc + 1):
                if 0 <= r < height and 0 <= c < width:
                    _apply_merge(out, written, order, r, c, props,
                                 new_idx, merge_fn, should_write)
                if err >= 0:
                    r += sr
                    err -= dc
                c += sc
                err += dr


# ---------------------------------------------------------------------------
# CPU supercover line burn (numba) -- used by all_touched polygon
# boundaries to match rasterio.features.rasterize(all_touched=True).
#
# Amanatides & Woo grid traversal: walks the segment cell-by-cell in
# pixel space, advancing along whichever axis hits the next cell
# boundary first. Every cell the mathematical segment crosses is
# visited, including the off-axis cells Bresenham would skip on a
# shallow diagonal.
# ---------------------------------------------------------------------------

@ngjit
def _burn_lines_supercover_cpu(out, written, x0_arr, y0_arr, x1_arr, y1_arr,
                               geom_idx, geom_props, height, width,
                               geom_global_idx, merge_fn, should_write,
                               order):
    n = len(x0_arr)
    for i in range(n):
        x0 = x0_arr[i]
        y0 = y0_arr[i]
        x1 = x1_arr[i]
        y1 = y1_arr[i]
        gi = geom_idx[i]
        props = geom_props[gi]
        new_idx = geom_global_idx[gi]

        # Starting cell. floor() gives the cell index containing the
        # endpoint; integer endpoints land on cell boundaries and we
        # break ties by stepping into the cell in the segment direction.
        cx = int(np.floor(x0))
        cy = int(np.floor(y0))

        end_cx = int(np.floor(x1))
        end_cy = int(np.floor(y1))

        dx = x1 - x0
        dy = y1 - y0

        # Step direction along each axis. 0 dx/dy -> step is irrelevant
        # because we will never advance that axis (t_delta is +inf).
        if dx > 0.0:
            step_x = 1
        elif dx < 0.0:
            step_x = -1
        else:
            step_x = 0

        if dy > 0.0:
            step_y = 1
        elif dy < 0.0:
            step_y = -1
        else:
            step_y = 0

        # t to traverse one full cell along each axis (parametric t in
        # [0, 1] along the segment).
        if dx != 0.0:
            t_delta_x = 1.0 / (dx if dx > 0.0 else -dx)
        else:
            t_delta_x = np.inf

        if dy != 0.0:
            t_delta_y = 1.0 / (dy if dy > 0.0 else -dy)
        else:
            t_delta_y = np.inf

        # t at which the segment first crosses the next x / y cell
        # boundary in its direction of travel.
        if dx > 0.0:
            next_x_boundary = float(cx + 1)
            t_max_x = (next_x_boundary - x0) / dx
        elif dx < 0.0:
            next_x_boundary = float(cx)
            t_max_x = (next_x_boundary - x0) / dx
        else:
            t_max_x = np.inf

        if dy > 0.0:
            next_y_boundary = float(cy + 1)
            t_max_y = (next_y_boundary - y0) / dy
        elif dy < 0.0:
            next_y_boundary = float(cy)
            t_max_y = (next_y_boundary - y0) / dy
        else:
            t_max_y = np.inf

        # Walk cells until we have stepped through the cell containing
        # the segment endpoint. A small step cap guards against any
        # pathological floating point case where t_max never crosses 1.
        max_steps = (abs(end_cx - cx) + abs(end_cy - cy) + 2)
        for _ in range(max_steps):
            if 0 <= cy < height and 0 <= cx < width:
                _apply_merge(out, written, order, cy, cx, props,
                             new_idx, merge_fn, should_write)

            if cx == end_cx and cy == end_cy:
                break

            if t_max_x < t_max_y:
                t_max_x += t_delta_x
                cx += step_x
            elif t_max_y < t_max_x:
                t_max_y += t_delta_y
                cy += step_y
            else:
                # Exact corner crossing. Stepping just one axis at a
                # time misses the diagonal-neighbour cells the segment
                # technically only grazes; matching rasterio's
                # all_touched semantics here means advancing both
                # axes in a single tick so we land in the next cell
                # along the diagonal without burning the two off-axis
                # neighbours.
                t_max_x += t_delta_x
                t_max_y += t_delta_y
                cx += step_x
                cy += step_y


# ---------------------------------------------------------------------------
# Per-geometry write deduplication for merge='sum' / merge='count'
#
# Those two merges are the only built-ins where burning the same pixel
# twice for the same geometry changes the result (min/max are
# idempotent, first/last gate repeats through the per-pixel owner
# index).  Two code paths used to do exactly that (issue #3304):
#
#   * all_touched=True polygons: the scanline fill burns every cell
#     whose center is inside, then the supercover boundary pass burns
#     every cell the rings cross -- including the ones scanline just
#     burned -- and consecutive ring segments re-burn their shared
#     vertex cell.
#   * lines: each Bresenham segment burns both endpoints, so the
#     connecting vertex of consecutive segments is burned once per
#     segment.
#
# For sum/count the line / boundary cells are therefore enumerated
# host-side (the segment extraction is host-side already), deduplicated
# per (geometry, row, col), filtered against the geometry's own
# scanline coverage, and burned through the point kernels, which write
# each surviving cell exactly once per geometry on every backend.
# GDAL behaves the same way: rasterio's MergeAlg.add yields 1 per
# geometry per pixel, even for self-crossing lines.  Points are left
# alone -- GDAL burns once per point, so a MultiPoint with duplicated
# coordinates adding twice already matches.
#
# Memory tradeoff: line cells are materialized as O(pixels traversed)
# int32 arrays instead of O(vertices) segment endpoints.  That is the
# same order as the work the burn kernel does anyway, but it is
# allocated up front (per tile on the dask backends), so very long
# lines on very large eager rasters cost a few extra bytes per
# traversed pixel while the burn runs.
# ---------------------------------------------------------------------------

@ngjit
def _bresenham_cells(r0_arr, c0_arr, r1_arr, c1_arr, geom_idx,
                     height, width):
    """Enumerate the in-bounds cells each Bresenham segment visits.

    Walks the exact same path as :func:`_burn_lines_cpu` (and the GPU
    twin) so the covered cell set is identical; only the write is
    replaced by collection.  Returns (rows, cols, gids) int32 arrays
    with one entry per visited in-bounds cell (duplicates included --
    the caller dedups).
    """
    n = len(r0_arr)
    total = 0
    for i in range(n):
        dr = r1_arr[i] - r0_arr[i]
        if dr < 0:
            dr = -dr
        dc = c1_arr[i] - c0_arr[i]
        if dc < 0:
            dc = -dc
        total += (dr if dr >= dc else dc) + 1

    rows = np.empty(total, np.int32)
    cols = np.empty(total, np.int32)
    gids = np.empty(total, np.int32)
    m = 0
    for i in range(n):
        r0 = r0_arr[i]
        c0 = c0_arr[i]
        r1 = r1_arr[i]
        c1 = c1_arr[i]
        gi = geom_idx[i]

        dr = r1 - r0
        dc = c1 - c0
        sr = 1 if dr >= 0 else -1
        sc = 1 if dc >= 0 else -1
        dr = dr * sr
        dc = dc * sc

        if dr >= dc:
            err = dc - dr
            r, c = r0, c0
            for _ in range(dr + 1):
                if 0 <= r < height and 0 <= c < width:
                    rows[m] = r
                    cols[m] = c
                    gids[m] = gi
                    m += 1
                if err >= 0:
                    c += sc
                    err -= dr
                r += sr
                err += dc
        else:
            err = dr - dc
            r, c = r0, c0
            for _ in range(dc + 1):
                if 0 <= r < height and 0 <= c < width:
                    rows[m] = r
                    cols[m] = c
                    gids[m] = gi
                    m += 1
                if err >= 0:
                    r += sr
                    err -= dc
                c += sc
                err += dr
    return rows[:m].copy(), cols[:m].copy(), gids[:m].copy()


@ngjit
def _supercover_cells(x0_arr, y0_arr, x1_arr, y1_arr, geom_idx,
                      height, width):
    """Enumerate the in-bounds cells each supercover segment visits.

    Same Amanatides & Woo walk as :func:`_burn_lines_supercover_cpu`
    (including the diagonal corner tick), collecting instead of
    writing.  Returns (rows, cols, gids) int32 arrays; duplicates are
    the caller's problem.
    """
    n = len(x0_arr)
    total = 0
    for i in range(n):
        cx = int(np.floor(x0_arr[i]))
        cy = int(np.floor(y0_arr[i]))
        ex = int(np.floor(x1_arr[i]))
        ey = int(np.floor(y1_arr[i]))
        total += abs(ex - cx) + abs(ey - cy) + 2

    rows = np.empty(total, np.int32)
    cols = np.empty(total, np.int32)
    gids = np.empty(total, np.int32)
    m = 0
    for i in range(n):
        x0 = x0_arr[i]
        y0 = y0_arr[i]
        x1 = x1_arr[i]
        y1 = y1_arr[i]
        gi = geom_idx[i]

        cx = int(np.floor(x0))
        cy = int(np.floor(y0))
        end_cx = int(np.floor(x1))
        end_cy = int(np.floor(y1))

        dx = x1 - x0
        dy = y1 - y0

        if dx > 0.0:
            step_x = 1
        elif dx < 0.0:
            step_x = -1
        else:
            step_x = 0

        if dy > 0.0:
            step_y = 1
        elif dy < 0.0:
            step_y = -1
        else:
            step_y = 0

        if dx != 0.0:
            t_delta_x = 1.0 / (dx if dx > 0.0 else -dx)
        else:
            t_delta_x = np.inf

        if dy != 0.0:
            t_delta_y = 1.0 / (dy if dy > 0.0 else -dy)
        else:
            t_delta_y = np.inf

        if dx > 0.0:
            t_max_x = (float(cx + 1) - x0) / dx
        elif dx < 0.0:
            t_max_x = (float(cx) - x0) / dx
        else:
            t_max_x = np.inf

        if dy > 0.0:
            t_max_y = (float(cy + 1) - y0) / dy
        elif dy < 0.0:
            t_max_y = (float(cy) - y0) / dy
        else:
            t_max_y = np.inf

        max_steps = abs(end_cx - cx) + abs(end_cy - cy) + 2
        for _ in range(max_steps):
            if 0 <= cy < height and 0 <= cx < width:
                rows[m] = cy
                cols[m] = cx
                gids[m] = gi
                m += 1
            if cx == end_cx and cy == end_cy:
                break
            if t_max_x < t_max_y:
                t_max_x += t_delta_x
                cx += step_x
            elif t_max_y < t_max_x:
                t_max_y += t_delta_y
                cy += step_y
            else:
                t_max_x += t_delta_x
                t_max_y += t_delta_y
                cx += step_x
                cy += step_y
    return rows[:m].copy(), cols[:m].copy(), gids[:m].copy()


def _dedup_cells(rows, cols, gids, height, width):
    """Drop duplicate (gid, row, col) triplets.

    Packs the triplet into a single int64 key (all components are
    non-negative and bounded by the raster dimensions / geometry
    count), uniques it, and unpacks.  Output order is irrelevant to
    the commutative merges this feeds.

    The key only overflows when n_geometries * height * width exceeds
    2**63, by which point the cell arrays being deduplicated could not
    have been allocated in the first place.
    """
    if len(rows) == 0:
        return rows, cols, gids
    h = np.int64(height)
    w = np.int64(width)
    key = (gids.astype(np.int64) * h + rows) * w + cols
    uniq = np.unique(key)
    hw = h * w
    g = (uniq // hw).astype(np.int32)
    rem = uniq % hw
    r = (rem // w).astype(np.int32)
    c = (rem % w).astype(np.int32)
    return r, c, g


@ngjit
def _cells_not_scanline_covered(rows, cols, gids, e_y_min, e_y_max,
                                e_x_at_ymin, e_inv_slope, gid_starts):
    """Boolean keep-mask: True where the cell is NOT burned by its own
    geometry's scanline fill.

    Replays the scanline arithmetic for the cell's row: an edge of the
    same geometry covers column ``c`` when ``ceil(x_at_row) <= c``, and
    the fill burns ``c`` exactly when an odd number of the geometry's
    active edges do (the span pairs in :func:`_scanline_fill_cpu` are
    consecutive sorted intersections, so [ceil(x_k), ceil(x_{k+1})-1]
    membership and crossing parity agree).  Using the same float
    expression and the same ceil keeps the test bit-identical to what
    the fill actually burned.

    Edge arrays must be pre-sorted by geometry id; ``gid_starts`` maps
    gid -> [start, end) slice of the sorted arrays.
    """
    n = len(rows)
    keep = np.ones(n, np.bool_)
    for i in range(n):
        g = gids[i]
        r = rows[i]
        c = cols[i]
        cnt = 0
        for e in range(gid_starts[g], gid_starts[g + 1]):
            if e_y_min[e] <= r and r <= e_y_max[e]:
                x = e_x_at_ymin[e] + (r - e_y_min[e]) * e_inv_slope[e]
                if int(np.ceil(x)) <= c:
                    cnt += 1
        if cnt % 2 == 1:
            keep[i] = False
    return keep


def _boundary_cells_sum_count(poly_geoms, poly_ids, bounds, height, width,
                              edge_arrays, boundary_segments=None):
    """all_touched boundary cells for sum/count, deduplicated per geometry.

    Enumerates the supercover cells of every ring segment, drops
    duplicate (geometry, cell) pairs (shared vertices, ring closure),
    then drops cells the same geometry's scanline fill already burns.
    The survivors are exactly the cells all_touched adds on top of the
    center-inside fill, once per geometry.

    ``edge_arrays`` is the 5-tuple the scanline fill consumes; passing
    the same arrays keeps the coverage test in the same float
    arithmetic as the fill itself.

    ``boundary_segments`` optionally supplies pre-computed tile-local
    boundary float segments ``(bx0, by0, bx1, by1, bidx)``.  The dask
    tile path passes segments derived in the global grid frame (see
    :func:`_extract_polygon_boundary_segments_float_global`) so the
    supercover walk lands on the same cells as the eager backend
    (issue #3384); the eager path leaves it ``None`` and extracts from
    ``bounds`` directly.
    """
    empty = np.empty(0, np.int32)
    if boundary_segments is None:
        bx0, by0, bx1, by1, bidx = _extract_polygon_boundary_segments_float(
            poly_geoms, poly_ids, bounds, height, width)
    else:
        bx0, by0, bx1, by1, bidx = boundary_segments
    if len(bx0) == 0:
        return empty, empty.copy(), empty.copy()

    rows, cols, gids = _supercover_cells(
        bx0, by0, bx1, by1, bidx, height, width)
    rows, cols, gids = _dedup_cells(rows, cols, gids, height, width)
    if len(rows) == 0:
        return rows, cols, gids

    edge_y_min, edge_y_max, edge_x_at_ymin, edge_inv_slope, \
        edge_geom_id = edge_arrays
    if len(edge_y_min) > 0:
        order = np.argsort(edge_geom_id, kind='stable')
        sorted_gid = edge_geom_id[order]
        n_poly = len(poly_geoms)
        gid_starts = np.searchsorted(
            sorted_gid, np.arange(n_poly + 1, dtype=sorted_gid.dtype))
        keep = _cells_not_scanline_covered(
            rows, cols, gids,
            edge_y_min[order], edge_y_max[order],
            edge_x_at_ymin[order], edge_inv_slope[order],
            gid_starts)
        rows, cols, gids = rows[keep], cols[keep], gids[keep]
    return rows, cols, gids


def _is_dedup_merge_fn(merge_fn):
    """True when *merge_fn* is one of the built-in CPU merges that needs
    per-geometry write deduplication (sum / count).  Identity-based so
    user callables never match; the built-ins are module-level numba
    dispatchers, so identity survives pickling into dask workers.
    """
    return merge_fn is _merge_sum or merge_fn is _merge_count


# ---------------------------------------------------------------------------
# CPU scanline fill (numba) -- edges must be sorted by y_min
# ---------------------------------------------------------------------------

@ngjit
def _scanline_fill_cpu(out, written, edge_y_min, edge_y_max, edge_x_at_ymin,
                       edge_inv_slope, edge_geom_id,
                       geom_props, height, width, geom_global_idx,
                       merge_fn, should_write, order):
    """Scanline fill with active-edge-list for O(active) work per row.

    Instead of scanning all edges up to the binary-search cutoff (which
    wastes >99% of checks on dead edges for many-polygon inputs), this
    maintains a compact list of currently-active edge indices.  For each
    row we remove expired edges and add newly-active ones, keeping total
    work proportional to the sum of active-edge counts across rows.
    """
    n_edges = len(edge_y_min)

    # Active edge list: indices into the edge arrays
    active = np.empty(n_edges, dtype=np.int32)
    n_active = 0
    add_ptr = 0  # next edge to consider adding (y_min sorted)

    # Scratch arrays for intersections
    xs = np.empty(n_edges, dtype=np.float64)
    gs = np.empty(n_edges, dtype=np.int32)

    for row in range(height):
        # 1. Remove expired edges (y_max < row)
        write_pos = 0
        for i in range(n_active):
            if edge_y_max[active[i]] >= row:
                active[write_pos] = active[i]
                write_pos += 1
        n_active = write_pos

        # 2. Add newly-active edges whose y_min <= row
        while add_ptr < n_edges and edge_y_min[add_ptr] <= row:
            active[n_active] = add_ptr
            n_active += 1
            add_ptr += 1

        if n_active == 0:
            continue

        # 3. Compute x-intersections for active edges only
        for i in range(n_active):
            e = active[i]
            xs[i] = (edge_x_at_ymin[e]
                     + (row - edge_y_min[e]) * edge_inv_slope[e])
            gs[i] = edge_geom_id[e]

        # 4. Shell sort by (geom_id, x) so each geometry's edges pair
        # correctly and geometries are processed in input order.
        # Shell sort is O(n^(4/3)) worst-case vs insertion sort's O(n²),
        # while staying in-place with no allocation.  The final gap=1
        # pass is a standard insertion sort, which is fast when the data
        # is already nearly sorted (common between consecutive rows).
        gap = n_active >> 1
        while gap > 0:
            for i in range(gap, n_active):
                kx = xs[i]
                kg = gs[i]
                j = i - gap
                while j >= 0 and (gs[j] > kg or (gs[j] == kg and xs[j] > kx)):
                    xs[j + gap] = xs[j]
                    gs[j + gap] = gs[j]
                    j -= gap
                xs[j + gap] = kx
                gs[j + gap] = kg
            gap >>= 1

        # 5. Fill between edge pairs per geometry
        i = 0
        while i < n_active - 1:
            gid = gs[i]
            new_idx = geom_global_idx[gid]
            j = i
            while j < n_active and gs[j] == gid:
                j += 1
            k = i
            while k + 1 < j:
                x_start = xs[k]
                x_end = xs[k + 1]
                col_start = max(int(np.ceil(x_start)), 0)
                col_end = min(int(np.ceil(x_end)) - 1, width - 1)
                for c in range(col_start, col_end + 1):
                    _apply_merge(out, written, order, row, c,
                                 geom_props[gid], new_idx,
                                 merge_fn, should_write)
                k += 2
            i = j


def _alloc_order(height, width, should_write):
    """Allocate the per-pixel owner-index array for the CPU kernels.

    ``order`` tracks the input index of the current owner per pixel.  -1
    sentinel is fine: written[r,c]==0 means cur_idx is never consulted
    for the "first write" branch of the predicates.

    Only the ordered predicates (``first`` / ``last``) ever read the
    stored values.  For ``_should_write_any`` merges (max/min/sum/count
    and user callables) the kernels still store the int64 owner index,
    but nothing reads it back, so an int8 buffer (1 byte/pixel instead
    of 8) is enough -- numba wraps the store and the wrapped values are
    dead.  Issue #3107.

    Caveat: the wrap is a compiled-code behavior.  Under
    ``NUMBA_DISABLE_JIT=1`` (pure-Python debugging) NumPy raises
    ``OverflowError`` on the first store of an owner index above 127;
    if that bites, force the int64 branch here.
    """
    if should_write is _should_write_any:
        return np.zeros((height, width), dtype=np.int8)
    return np.full((height, width), -1, dtype=np.int64)


def _run_numpy(geometries, props_array, bounds, height, width, fill, dtype,
               all_touched, merge_fn, should_write):
    """NumPy backend for rasterize."""
    out = np.full((height, width), fill, dtype=np.float64)
    written = np.zeros((height, width), dtype=np.int8)
    order = _alloc_order(height, width, should_write)

    (poly_geoms, poly_props, poly_ids, poly_global), \
        (line_geoms, line_props, line_global), \
        (point_geoms, point_props, point_global) = _classify_geometries(
            geometries, props_array)

    # 1. Polygons -- always use normal edge ranges for scanline fill
    #    (all_touched y-expansion breaks edge pairing, so we handle
    #    all_touched by drawing polygon boundaries separately below).
    edge_arrays = _extract_edges(
        poly_geoms, poly_ids, bounds, height, width)
    edge_arrays = _sort_edges(edge_arrays)
    if len(edge_arrays[0]) > 0:
        _scanline_fill_cpu(out, written, *edge_arrays,
                           poly_props, height, width, poly_global,
                           merge_fn, should_write, order)

    # 1b. all_touched: draw polygon boundaries with a supercover
    #     grid traversal so every pixel a ring segment crosses is
    #     burned (matches rasterio.features.rasterize all_touched).
    #     sum/count burn the deduplicated cell list through the point
    #     kernel instead, so each geometry contributes at most once
    #     per pixel (issue #3304).
    dedup = _is_dedup_merge_fn(merge_fn)
    if all_touched and poly_geoms:
        if dedup:
            brows, bcols, bgids = _boundary_cells_sum_count(
                poly_geoms, poly_ids, bounds, height, width, edge_arrays)
            if len(brows) > 0:
                _burn_points_cpu(out, written, brows, bcols, bgids,
                                 poly_props, poly_global,
                                 merge_fn, should_write, order)
        else:
            bx0, by0, bx1, by1, bidx = (
                _extract_polygon_boundary_segments_float(
                    poly_geoms, poly_ids, bounds, height, width))
            if len(bx0) > 0:
                _burn_lines_supercover_cpu(
                    out, written, bx0, by0, bx1, by1, bidx,
                    poly_props, height, width, poly_global,
                    merge_fn, should_write, order)

    # 2. Lines.  all_touched burns every cell a segment crosses via the
    #    supercover walk (issue #3102), matching the polygon-boundary
    #    path and rasterio; otherwise the Bresenham burner runs.
    #    sum/count: enumerate + dedup cells so the connecting vertex of
    #    consecutive segments is burned once (issue #3304).
    if all_touched and line_geoms:
        lx0, ly0, lx1, ly1, line_idx = _extract_line_segments_float(
            line_geoms, bounds, height, width)
        if len(lx0) > 0:
            if dedup:
                lrows, lcols, lgids = _supercover_cells(
                    lx0, ly0, lx1, ly1, line_idx, height, width)
                lrows, lcols, lgids = _dedup_cells(
                    lrows, lcols, lgids, height, width)
                if len(lrows) > 0:
                    _burn_points_cpu(out, written, lrows, lcols, lgids,
                                     line_props, line_global,
                                     merge_fn, should_write, order)
            else:
                _burn_lines_supercover_cpu(
                    out, written, lx0, ly0, lx1, ly1, line_idx,
                    line_props, height, width, line_global,
                    merge_fn, should_write, order)
    else:
        r0, c0, r1, c1, line_idx = _extract_line_segments(
            line_geoms, bounds, height, width)
        if len(r0) > 0:
            if dedup:
                lrows, lcols, lgids = _bresenham_cells(
                    r0, c0, r1, c1, line_idx, height, width)
                lrows, lcols, lgids = _dedup_cells(
                    lrows, lcols, lgids, height, width)
                if len(lrows) > 0:
                    _burn_points_cpu(out, written, lrows, lcols, lgids,
                                     line_props, line_global,
                                     merge_fn, should_write, order)
            else:
                _burn_lines_cpu(out, written, r0, c0, r1, c1, line_idx,
                                line_props, height, width, line_global,
                                merge_fn, should_write, order)

    # 3. Points
    prows, pcols, pt_idx = _extract_points(
        point_geoms, bounds, height, width)
    if len(prows) > 0:
        _burn_points_cpu(out, written, prows, pcols, pt_idx,
                         point_props, point_global,
                         merge_fn, should_write, order)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        # copy=False: the work buffer is local, so when dtype is already
        # float64 (the default) the cast is a no-op and skips a full-
        # raster copy.  Issue #3107.
        return out.astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# GPU kernels -- compiled lazily to avoid importing numba.cuda at module
# level (~160ms + CUDA driver init even when not using GPU).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GPU merge functions -- @cuda.jit(device=True) equivalents of the CPU
# merge functions above.  Defined lazily to avoid importing numba.cuda
# at module level.
# ---------------------------------------------------------------------------

_gpu_merge_fns = None


def _get_gpu_merge_fns():
    """Lazily define and cache built-in GPU merge device functions and the
    matching ordering predicates.  Mirrors the CPU dispatch table.
    """
    global _gpu_merge_fns
    if _gpu_merge_fns is not None:
        return _gpu_merge_fns

    from numba import cuda

    @cuda.jit(device=True)
    def _merge_overwrite_gpu(pixel, props, is_first):
        return props[0]

    @cuda.jit(device=True)
    def _merge_max_gpu(pixel, props, is_first):
        if is_first or props[0] > pixel:
            return props[0]
        return pixel

    @cuda.jit(device=True)
    def _merge_min_gpu(pixel, props, is_first):
        if is_first or props[0] < pixel:
            return props[0]
        return pixel

    @cuda.jit(device=True)
    def _merge_sum_gpu(pixel, props, is_first):
        if is_first:
            return props[0]
        return pixel + props[0]

    @cuda.jit(device=True)
    def _merge_count_gpu(pixel, props, is_first):
        if is_first:
            return 1.0
        return pixel + 1.0

    @cuda.jit(device=True)
    def _should_write_any_gpu(is_first, new_idx, cur_idx):
        return True

    @cuda.jit(device=True)
    def _should_write_first_gpu(is_first, new_idx, cur_idx):
        return is_first or new_idx < cur_idx

    @cuda.jit(device=True)
    def _should_write_last_gpu(is_first, new_idx, cur_idx):
        return is_first or new_idx > cur_idx

    _gpu_merge_fns = {
        'last':  (_merge_overwrite_gpu, _should_write_last_gpu),
        'first': (_merge_overwrite_gpu, _should_write_first_gpu),
        'max':   (_merge_max_gpu,       _should_write_any_gpu),
        'min':   (_merge_min_gpu,       _should_write_any_gpu),
        'sum':   (_merge_sum_gpu,       _should_write_any_gpu),
        'count': (_merge_count_gpu,     _should_write_any_gpu),
    }
    return _gpu_merge_fns


# ---------------------------------------------------------------------------
# GPU kernels -- compiled per (merge_fn, should_write) pair and cached.
# ---------------------------------------------------------------------------

_gpu_kernel_cache = {}

# Must stay in lock-step with the cuda.local.array(2048, ...) allocations in
# _scanline_fill_gpu.  Raise both together if this is ever tuned.
_GPU_MAX_ISECT = 2048


def _check_gpu_edge_cap(row_ptr):
    """Raise if any raster row needs more active edges than the GPU kernel
    can hold in its fixed-size local array.  Without this guard the CUDA
    scanline kernel would silently truncate and emit wrong pixels.
    """
    max_edges_per_row = int(np.diff(row_ptr).max())
    if max_edges_per_row > _GPU_MAX_ISECT:
        raise ValueError(
            f"rasterize CUDA kernel: {max_edges_per_row} active edges in a "
            f"single row exceeds the limit of {_GPU_MAX_ISECT}. The kernel "
            f"would silently drop excess edges and produce wrong output. "
            f"Use the numpy/dask+numpy backend, or reduce polygon density."
        )


#: Built-in merges whose GPU path uses CUDA atomics so overlapping
#: writes are deterministic.  User-supplied merge callables fall back to
#: the non-atomic closure path (the public ``merge_fn`` signature has no
#: slot for atomicity, and we cannot synthesise atomics for an arbitrary
#: Python function).
_GPU_ATOMIC_MERGES = ('sum', 'count', 'min', 'max', 'first', 'last')


def _ensure_gpu_kernels(merge_fn, should_write, merge_name=None):
    """Compile CUDA kernels for the given merge + predicate pair and cache.

    Each unique pair produces a separate set of kernels because CUDA kernels
    cannot accept function arguments; both functions are captured by closure
    at compile time.

    When ``merge_name`` is one of the built-in merges (``sum``, ``count``,
    ``min``, ``max``, ``first``, ``last``), the returned kernel set uses
    CUDA atomics on ``out`` / ``order`` so that overlapping point writes,
    crossing line segments, and shared polygon-boundary pixels produce
    deterministic results matching the CPU backend.  Otherwise (callable
    or unknown merge) the kernels keep the legacy non-atomic
    read-modify-write path, which is racy on overlap but preserved for
    user callables whose signature has no atomic equivalent.
    """
    key = (id(merge_fn), id(should_write), merge_name)
    if key in _gpu_kernel_cache:
        return _gpu_kernel_cache[key]

    from numba import cuda

    atomic_mode = merge_name if merge_name in _GPU_ATOMIC_MERGES else None

    # The per-pixel write strategy depends on ``atomic_mode``.  Numba
    # treats free Python variables in a ``cuda.jit`` closure as compile-
    # time constants, so the if/elif ladder below is folded away and
    # each compiled kernel contains only the relevant atomic call.
    #
    # Initialisation contract (launcher fills ``out`` / ``order`` before
    # the kernel runs):
    #   sum, count    -> out = 0.0           , order = MIN_INT64
    #   min           -> out = +inf          , order = MIN_INT64
    #   max           -> out = -inf          , order = MIN_INT64
    #   first         -> out = fill          , order = MAX_INT64
    #   last          -> out = fill          , order = MIN_INT64
    #   None (user)   -> out = fill          , order = -1 (legacy)
    #
    # For sum/count/min/max, ``order`` is repurposed as a touched mask:
    # any non-sentinel value means the pixel was written by at least
    # one geometry.  A host-side post-pass blends in ``fill`` where
    # the pixel is still untouched, and substitutes ``fill`` for the
    # numerical sentinels used by min/max init.
    #
    # For first/last, the kernel is two-pass: pass 1 resolves the
    # winning input index atomically into ``order``; pass 2 stamps the
    # winner's ``props[0]`` value into ``out``.  Pass 1 and pass 2 are
    # separate compiled kernels (``_..._gpu`` and ``_..._gpu_pass2``);
    # ``_run_cupy`` / ``_rasterize_tile_cupy`` skip the pass-2 launch
    # for all other modes.

    @cuda.jit(device=True)
    def _apply_merge_gpu(out, written, order, r, c, props, new_idx):
        if atomic_mode == 'sum':
            cuda.atomic.add(out, (r, c), props[0])
            cuda.atomic.max(order, (r, c), new_idx)
        elif atomic_mode == 'count':
            cuda.atomic.add(out, (r, c), 1.0)
            cuda.atomic.max(order, (r, c), new_idx)
        elif atomic_mode == 'min':
            val = props[0]
            # CUDA atomicMin uses a < comparison, which returns False for
            # any NaN operand, so a NaN burn would silently leave the +inf
            # initialiser in place.  Track NaN burns in ``written`` (sized
            # H*W int32 for min/max, see _gpu_init_buffers) and stamp NaN
            # into ``out`` at finalize.  Issue #2255.
            if val == val:
                cuda.atomic.min(out, (r, c), val)
            else:
                cuda.atomic.max(written, (r, c), 1)
            cuda.atomic.max(order, (r, c), new_idx)
        elif atomic_mode == 'max':
            val = props[0]
            # See min branch above; the -inf initialiser would otherwise
            # win against any NaN burn.  Issue #2255.
            if val == val:
                cuda.atomic.max(out, (r, c), val)
            else:
                cuda.atomic.max(written, (r, c), 1)
            cuda.atomic.max(order, (r, c), new_idx)
        elif atomic_mode == 'first':
            # Pass 1: resolve winning input index per pixel.
            cuda.atomic.min(order, (r, c), new_idx)
        elif atomic_mode == 'last':
            cuda.atomic.max(order, (r, c), new_idx)
        else:
            # Legacy non-atomic path for user callables.
            is_first = np.int64(written[r, c] == 0)
            cur_idx = order[r, c]
            if not should_write(is_first, new_idx, cur_idx):
                return
            out[r, c] = merge_fn(out[r, c], props, is_first)
            written[r, c] = 1
            order[r, c] = new_idx

    # Pass-2 device function for first/last: stamp the winner's value
    # into ``out``.  Only the geometry whose index won pass 1 writes.
    # Multiple threads from the same geometry may write the same pixel,
    # but they all write ``props[0]`` so the outcome is deterministic.
    @cuda.jit(device=True)
    def _apply_merge_gpu_pass2(out, written, order, r, c, props, new_idx):
        if atomic_mode == 'first' or atomic_mode == 'last':
            if order[r, c] == new_idx:
                out[r, c] = props[0]
        # else: pass 2 is a no-op for all other modes; the launcher
        # simply skips it.

    @cuda.jit
    def _scanline_fill_gpu(out, written, order, edge_y_min, edge_x_at_ymin,
                           edge_inv_slope, edge_geom_id,
                           geom_props, geom_global_idx,
                           row_ptr, col_idx, width):
        """CUDA kernel: one thread per raster row, CSR-indexed active edges."""
        row = cuda.grid(1)
        if row >= out.shape[0]:
            return

        start = row_ptr[row]
        end = row_ptr[row + 1]
        count = end - start

        if count == 0:
            return

        MAX_ISECT = 2048
        if count > MAX_ISECT:
            count = MAX_ISECT

        xs = cuda.local.array(2048, dtype=np.float64)
        gs = cuda.local.array(2048, dtype=np.int32)

        actual = 0
        for k in range(start, end):
            if actual >= MAX_ISECT:
                break
            e = col_idx[k]
            xs[actual] = (edge_x_at_ymin[e]
                          + (row - edge_y_min[e]) * edge_inv_slope[e])
            gs[actual] = edge_geom_id[e]
            actual += 1

        # Shell sort by (geom_id, x)
        gap = actual >> 1
        while gap > 0:
            for i in range(gap, actual):
                kx = xs[i]
                kg = gs[i]
                j = i - gap
                while j >= 0 and (gs[j] > kg or (gs[j] == kg and xs[j] > kx)):
                    xs[j + gap] = xs[j]
                    gs[j + gap] = gs[j]
                    j -= gap
                xs[j + gap] = kx
                gs[j + gap] = kg
            gap >>= 1

        # Fill between pairs per geometry
        i = 0
        while i < actual - 1:
            gid = gs[i]
            new_idx = geom_global_idx[gid]
            j = i
            while j < actual and gs[j] == gid:
                j += 1
            k = i
            while k + 1 < j:
                x_start = xs[k]
                x_end = xs[k + 1]
                # Proper ceil: int() truncates toward zero, so for
                # positive fractions we add 1; for exact ints and
                # negative fractions int() already rounds up.
                ix = int(x_start)
                col_start = ix + 1 if x_start > ix else ix
                if col_start < 0:
                    col_start = 0
                # Half-open interval: ceil(x_end) - 1 excludes
                # boundary pixels whose centers are outside.
                ix_end = int(x_end)
                col_end = ix_end if x_end > ix_end else ix_end - 1
                if col_end >= width:
                    col_end = width - 1
                for c in range(col_start, col_end + 1):
                    _apply_merge_gpu(out, written, order, row, c,
                                     geom_props[gid], new_idx)
                k += 2
            i = j

    @cuda.jit
    def _burn_points_gpu(out, written, order, rows, cols, geom_idx,
                         geom_props, geom_global_idx, n_points):
        i = cuda.grid(1)
        if i >= n_points:
            return
        r = rows[i]
        c = cols[i]
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            gi = geom_idx[i]
            _apply_merge_gpu(out, written, order, r, c,
                             geom_props[gi], geom_global_idx[gi])

    @cuda.jit
    def _burn_lines_gpu(out, written, order, r0_arr, c0_arr, r1_arr, c1_arr,
                        geom_idx, geom_props, geom_global_idx,
                        n_segs, height, width):
        i = cuda.grid(1)
        if i >= n_segs:
            return
        r0 = r0_arr[i]
        c0 = c0_arr[i]
        r1 = r1_arr[i]
        c1 = c1_arr[i]
        gi = geom_idx[i]
        props = geom_props[gi]
        new_idx = geom_global_idx[gi]

        dr = r1 - r0
        dc = c1 - c0
        sr = 1 if dr >= 0 else -1
        sc = 1 if dc >= 0 else -1
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc

        if dr >= dc:
            err = dc - dr
            r, c = r0, c0
            for _ in range(dr + 1):
                if 0 <= r < height and 0 <= c < width:
                    _apply_merge_gpu(out, written, order, r, c,
                                     props, new_idx)
                if err >= 0:
                    c += sc
                    err -= dr
                r += sr
                err += dc
        else:
            err = dr - dc
            r, c = r0, c0
            for _ in range(dc + 1):
                if 0 <= r < height and 0 <= c < width:
                    _apply_merge_gpu(out, written, order, r, c,
                                     props, new_idx)
                if err >= 0:
                    r += sr
                    err -= dc
                c += sc
                err += dr

    @cuda.jit
    def _burn_lines_supercover_gpu(
            out, written, order,
            x0_arr, y0_arr, x1_arr, y1_arr,
            geom_idx, geom_props, geom_global_idx,
            n_segs, height, width):
        """Amanatides & Woo grid traversal on the GPU.

        One thread per segment; walks cells along the float-pixel
        segment so every cell crossed is burned. Used only by the
        all_touched polygon-boundary path; the regular line and
        point kernels still call ``_burn_lines_gpu`` so their
        behaviour is unchanged.
        """
        i = cuda.grid(1)
        if i >= n_segs:
            return
        x0 = x0_arr[i]
        y0 = y0_arr[i]
        x1 = x1_arr[i]
        y1 = y1_arr[i]
        gi = geom_idx[i]
        props = geom_props[gi]
        new_idx = geom_global_idx[gi]

        cx = int(math.floor(x0))
        cy = int(math.floor(y0))
        end_cx = int(math.floor(x1))
        end_cy = int(math.floor(y1))

        dx = x1 - x0
        dy = y1 - y0

        if dx > 0.0:
            step_x = 1
        elif dx < 0.0:
            step_x = -1
        else:
            step_x = 0

        if dy > 0.0:
            step_y = 1
        elif dy < 0.0:
            step_y = -1
        else:
            step_y = 0

        if dx != 0.0:
            t_delta_x = 1.0 / (dx if dx > 0.0 else -dx)
        else:
            t_delta_x = math.inf

        if dy != 0.0:
            t_delta_y = 1.0 / (dy if dy > 0.0 else -dy)
        else:
            t_delta_y = math.inf

        if dx > 0.0:
            t_max_x = (float(cx + 1) - x0) / dx
        elif dx < 0.0:
            t_max_x = (float(cx) - x0) / dx
        else:
            t_max_x = math.inf

        if dy > 0.0:
            t_max_y = (float(cy + 1) - y0) / dy
        elif dy < 0.0:
            t_max_y = (float(cy) - y0) / dy
        else:
            t_max_y = math.inf

        max_steps = abs(end_cx - cx) + abs(end_cy - cy) + 2
        for _ in range(max_steps):
            if 0 <= cy < height and 0 <= cx < width:
                _apply_merge_gpu(out, written, order, cy, cx,
                                 props, new_idx)
            if cx == end_cx and cy == end_cy:
                break
            if t_max_x < t_max_y:
                t_max_x += t_delta_x
                cx += step_x
            elif t_max_y < t_max_x:
                t_max_y += t_delta_y
                cy += step_y
            else:
                t_max_x += t_delta_x
                t_max_y += t_delta_y
                cx += step_x
                cy += step_y

    # ---- Pass-2 kernels (only meaningful for first / last) -----------
    # These mirror the pass-1 kernels but call ``_apply_merge_gpu_pass2``
    # to stamp the winner's value once ``order`` holds the resolved
    # input index.  Compiled unconditionally so the kernels dict shape
    # stays uniform; for non-first/last modes the launcher does not
    # invoke them.

    @cuda.jit
    def _scanline_fill_gpu_pass2(out, written, order, edge_y_min,
                                 edge_x_at_ymin, edge_inv_slope,
                                 edge_geom_id, geom_props, geom_global_idx,
                                 row_ptr, col_idx, width):
        row = cuda.grid(1)
        if row >= out.shape[0]:
            return
        start = row_ptr[row]
        end = row_ptr[row + 1]
        if end - start == 0:
            return
        MAX_ISECT = 2048
        xs = cuda.local.array(2048, dtype=np.float64)
        gs = cuda.local.array(2048, dtype=np.int32)
        actual = 0
        for k in range(start, end):
            if actual >= MAX_ISECT:
                break
            e = col_idx[k]
            xs[actual] = (edge_x_at_ymin[e]
                          + (row - edge_y_min[e]) * edge_inv_slope[e])
            gs[actual] = edge_geom_id[e]
            actual += 1
        gap = actual >> 1
        while gap > 0:
            for i in range(gap, actual):
                kx = xs[i]
                kg = gs[i]
                j = i - gap
                while j >= 0 and (gs[j] > kg or (gs[j] == kg and xs[j] > kx)):
                    xs[j + gap] = xs[j]
                    gs[j + gap] = gs[j]
                    j -= gap
                xs[j + gap] = kx
                gs[j + gap] = kg
            gap >>= 1
        i = 0
        while i < actual - 1:
            gid = gs[i]
            new_idx = geom_global_idx[gid]
            j = i
            while j < actual and gs[j] == gid:
                j += 1
            k = i
            while k + 1 < j:
                x_start = xs[k]
                x_end = xs[k + 1]
                ix = int(x_start)
                col_start = ix + 1 if x_start > ix else ix
                if col_start < 0:
                    col_start = 0
                ix_end = int(x_end)
                col_end = ix_end if x_end > ix_end else ix_end - 1
                if col_end >= width:
                    col_end = width - 1
                for c in range(col_start, col_end + 1):
                    _apply_merge_gpu_pass2(out, written, order, row, c,
                                           geom_props[gid], new_idx)
                k += 2
            i = j

    @cuda.jit
    def _burn_points_gpu_pass2(out, written, order, rows, cols, geom_idx,
                               geom_props, geom_global_idx, n_points):
        i = cuda.grid(1)
        if i >= n_points:
            return
        r = rows[i]
        c = cols[i]
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            gi = geom_idx[i]
            _apply_merge_gpu_pass2(out, written, order, r, c,
                                   geom_props[gi], geom_global_idx[gi])

    @cuda.jit
    def _burn_lines_gpu_pass2(out, written, order, r0_arr, c0_arr, r1_arr,
                              c1_arr, geom_idx, geom_props,
                              geom_global_idx, n_segs, height, width):
        i = cuda.grid(1)
        if i >= n_segs:
            return
        r0 = r0_arr[i]
        c0 = c0_arr[i]
        r1 = r1_arr[i]
        c1 = c1_arr[i]
        gi = geom_idx[i]
        props = geom_props[gi]
        new_idx = geom_global_idx[gi]
        dr = r1 - r0
        dc = c1 - c0
        sr = 1 if dr >= 0 else -1
        sc = 1 if dc >= 0 else -1
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc
        if dr >= dc:
            err = dc - dr
            r, c = r0, c0
            for _ in range(dr + 1):
                if 0 <= r < height and 0 <= c < width:
                    _apply_merge_gpu_pass2(out, written, order, r, c,
                                           props, new_idx)
                if err >= 0:
                    c += sc
                    err -= dr
                r += sr
                err += dc
        else:
            err = dr - dc
            r, c = r0, c0
            for _ in range(dc + 1):
                if 0 <= r < height and 0 <= c < width:
                    _apply_merge_gpu_pass2(out, written, order, r, c,
                                           props, new_idx)
                if err >= 0:
                    r += sr
                    err -= dc
                c += sc
                err += dr

    @cuda.jit
    def _burn_lines_supercover_gpu_pass2(
            out, written, order,
            x0_arr, y0_arr, x1_arr, y1_arr,
            geom_idx, geom_props, geom_global_idx,
            n_segs, height, width):
        """Pass-2 supercover boundary burn for first/last modes.

        Mirrors ``_burn_lines_supercover_gpu`` but calls
        ``_apply_merge_gpu_pass2`` so the winning-index resolved in
        pass 1 gets stamped exactly once per pixel.
        """
        i = cuda.grid(1)
        if i >= n_segs:
            return
        x0 = x0_arr[i]
        y0 = y0_arr[i]
        x1 = x1_arr[i]
        y1 = y1_arr[i]
        gi = geom_idx[i]
        props = geom_props[gi]
        new_idx = geom_global_idx[gi]

        cx = int(math.floor(x0))
        cy = int(math.floor(y0))
        end_cx = int(math.floor(x1))
        end_cy = int(math.floor(y1))

        dx = x1 - x0
        dy = y1 - y0

        if dx > 0.0:
            step_x = 1
        elif dx < 0.0:
            step_x = -1
        else:
            step_x = 0

        if dy > 0.0:
            step_y = 1
        elif dy < 0.0:
            step_y = -1
        else:
            step_y = 0

        if dx != 0.0:
            t_delta_x = 1.0 / (dx if dx > 0.0 else -dx)
        else:
            t_delta_x = math.inf

        if dy != 0.0:
            t_delta_y = 1.0 / (dy if dy > 0.0 else -dy)
        else:
            t_delta_y = math.inf

        if dx > 0.0:
            t_max_x = (float(cx + 1) - x0) / dx
        elif dx < 0.0:
            t_max_x = (float(cx) - x0) / dx
        else:
            t_max_x = math.inf

        if dy > 0.0:
            t_max_y = (float(cy + 1) - y0) / dy
        elif dy < 0.0:
            t_max_y = (float(cy) - y0) / dy
        else:
            t_max_y = math.inf

        max_steps = abs(end_cx - cx) + abs(end_cy - cy) + 2
        for _ in range(max_steps):
            if 0 <= cy < height and 0 <= cx < width:
                _apply_merge_gpu_pass2(out, written, order, cy, cx,
                                       props, new_idx)
            if cx == end_cx and cy == end_cy:
                break
            if t_max_x < t_max_y:
                t_max_x += t_delta_x
                cx += step_x
            elif t_max_y < t_max_x:
                t_max_y += t_delta_y
                cy += step_y
            else:
                t_max_x += t_delta_x
                t_max_y += t_delta_y
                cx += step_x
                cy += step_y

    kernels = {
        'scanline_fill': _scanline_fill_gpu,
        'burn_points': _burn_points_gpu,
        'burn_lines': _burn_lines_gpu,
        'burn_lines_supercover': _burn_lines_supercover_gpu,
        'scanline_fill_pass2': _scanline_fill_gpu_pass2,
        'burn_points_pass2': _burn_points_gpu_pass2,
        'burn_lines_pass2': _burn_lines_gpu_pass2,
        'burn_lines_supercover_pass2': _burn_lines_supercover_gpu_pass2,
        'atomic_mode': atomic_mode,
    }
    _gpu_kernel_cache[key] = kernels
    return kernels


@ngjit
def _build_row_csr_numba(edge_y_min, edge_y_max, height):
    """Build a CSR structure mapping each raster row to its active edges.

    Uses a difference-array approach so Pass 1 is O(n_edges + height)
    instead of O(n_edges * avg_edge_height).  Pass 2 still needs to
    place each edge into every row it spans, but the row_ptr / offsets
    are already computed so no redundant counting occurs.
    """
    n_edges = len(edge_y_min)
    if n_edges == 0:
        return (np.zeros(height + 1, dtype=np.int64),
                np.empty(0, dtype=np.int32))

    # Pass 1: difference-array counting — O(n_edges + height)
    # diff[r] += 1 when an edge starts, diff[r+1] -= 1 when it ends.
    # row_ptr / diff / offsets are int64 so the cumulative edge-count
    # (sum over rows of edges_per_row) cannot overflow int32 on tall
    # rasters with many long edges. col_idx values stay int32 since
    # they hold edge indices, not cumulative counts.
    diff = np.zeros(height + 1, dtype=np.int64)
    for e in range(n_edges):
        y_lo = edge_y_min[e]
        y_hi = edge_y_max[e]
        if y_hi >= height:
            y_hi = height - 1
        if y_lo <= y_hi:
            diff[y_lo] += 1
            diff[y_hi + 1] -= 1

    # Prefix sum on diff -> counts, and build row_ptr simultaneously
    row_ptr = np.empty(height + 1, dtype=np.int64)
    row_ptr[0] = 0
    running = np.int64(0)
    for r in range(height):
        running += diff[r]
        row_ptr[r + 1] = row_ptr[r] + running

    # Pass 2: fill col_idx — each edge placed into its row range
    total = row_ptr[height]
    col_idx = np.empty(total, dtype=np.int32)
    offsets = np.empty(height, dtype=np.int64)
    for r in range(height):
        offsets[r] = row_ptr[r]

    for e in range(n_edges):
        y_lo = edge_y_min[e]
        y_hi = edge_y_max[e]
        if y_hi >= height:
            y_hi = height - 1
        for r in range(y_lo, y_hi + 1):
            col_idx[offsets[r]] = e
            offsets[r] += 1

    return row_ptr, col_idx


#: Sentinels used to mark "untouched" pixels in the GPU ``order`` array.
#: Numba's ``cuda.atomic.{min,max}`` on int64 accepts standard limits.
_GPU_ORDER_SENTINEL_MIN = np.int64(np.iinfo(np.int64).min)
_GPU_ORDER_SENTINEL_MAX = np.int64(np.iinfo(np.int64).max)


def _gpu_init_buffers(height, width, fill, merge_name):
    """Allocate ``out`` / ``written`` / ``order`` with the right initial
    values for the selected atomic merge.  Returns the trio plus the
    sentinel value used in ``order`` so the caller can blend ``fill``
    back in afterwards.

    Most atomic kernels never consult ``written`` (the legacy
    user-callable path is the only consumer of the H*W int8 mask), so
    they get a placeholder ``(1, 1)`` buffer to satisfy the kernel
    signature without spending ``H*W`` bytes on dead storage.

    ``min`` / ``max`` repurpose ``written`` as a NaN-burn mask: the
    +/-inf initialisers used by the atomic min/max kernels would
    otherwise outrank any NaN burn ('NaN > finite' is False), silently
    dropping NaN-valued geometries.  The mask records which pixels saw
    a NaN burn so finalize can stamp NaN back in.  Issue #2255.
    ``cuda.atomic.max`` requires int32 / int64 / uint32 / uint64 (int8
    is not supported), hence the dtype bump for these two modes.
    """
    if merge_name in ('min', 'max'):
        written = cupy.zeros((height, width), dtype=cupy.int32)
    elif merge_name in _GPU_ATOMIC_MERGES:
        written = cupy.zeros((1, 1), dtype=cupy.int8)
    else:
        written = cupy.zeros((height, width), dtype=cupy.int8)
    if merge_name in ('sum', 'count'):
        out = cupy.zeros((height, width), dtype=cupy.float64)
        order = cupy.full((height, width), _GPU_ORDER_SENTINEL_MIN,
                          dtype=cupy.int64)
        order_sentinel = _GPU_ORDER_SENTINEL_MIN
    elif merge_name == 'min':
        out = cupy.full((height, width), cupy.inf, dtype=cupy.float64)
        order = cupy.full((height, width), _GPU_ORDER_SENTINEL_MIN,
                          dtype=cupy.int64)
        order_sentinel = _GPU_ORDER_SENTINEL_MIN
    elif merge_name == 'max':
        out = cupy.full((height, width), -cupy.inf, dtype=cupy.float64)
        order = cupy.full((height, width), _GPU_ORDER_SENTINEL_MIN,
                          dtype=cupy.int64)
        order_sentinel = _GPU_ORDER_SENTINEL_MIN
    elif merge_name == 'first':
        out = cupy.full((height, width), fill, dtype=cupy.float64)
        order = cupy.full((height, width), _GPU_ORDER_SENTINEL_MAX,
                          dtype=cupy.int64)
        order_sentinel = _GPU_ORDER_SENTINEL_MAX
    elif merge_name == 'last':
        out = cupy.full((height, width), fill, dtype=cupy.float64)
        order = cupy.full((height, width), _GPU_ORDER_SENTINEL_MIN,
                          dtype=cupy.int64)
        order_sentinel = _GPU_ORDER_SENTINEL_MIN
    else:
        # User callable path: legacy initialisation, non-atomic semantics.
        out = cupy.full((height, width), fill, dtype=cupy.float64)
        order = cupy.full((height, width), -1, dtype=cupy.int64)
        order_sentinel = None
    return out, written, order, order_sentinel


def _gpu_finalize_buffers(out, written, order, order_sentinel, fill,
                          merge_name):
    """Apply the post-pass blend that restores ``fill`` for untouched
    pixels (sum/count/min/max/first/last) and replaces the +/- inf
    initialisers used by min/max.

    For ``min`` / ``max``, also stamps NaN into pixels that received at
    least one NaN burn -- ``written`` is repurposed as the NaN-mask for
    those modes (see ``_gpu_init_buffers``).  Issue #2255.

    Returns ``out`` (modified in-place where convenient).
    """
    if merge_name is None or order_sentinel is None:
        return out
    touched = order != order_sentinel
    if merge_name in ('sum', 'count'):
        # Pixels with no contribution should show ``fill``, not 0.
        out = cupy.where(touched, out, cupy.float64(fill))
    elif merge_name in ('min', 'max'):
        # Any pixel that saw a NaN burn must show NaN, otherwise the
        # +/-inf initialiser would leak through (or the finite max/min
        # of the non-NaN burns would silently win).  Issue #2255.
        nan_burned = written > 0
        out = cupy.where(nan_burned, cupy.float64(cupy.nan), out)
        # +/-inf initialiser leaks into untouched pixels; restore fill.
        out = cupy.where(touched, out, cupy.float64(fill))
    # first/last already initialise ``out`` to ``fill``; pass-2 only
    # writes for touched pixels, so no blend is needed.
    return out


def _run_cupy(geometries, props_array, bounds, height, width, fill, dtype,
              all_touched, merge_fn, should_write, merge_name=None):
    """CuPy backend for rasterize."""
    kernels = _ensure_gpu_kernels(merge_fn, should_write, merge_name)
    atomic_mode = kernels['atomic_mode']
    two_pass = atomic_mode in ('first', 'last')
    # sum/count need per-geometry write dedup (issue #3304); their
    # boundary / line cells are enumerated host-side and burned through
    # the point kernel instead of the segment kernels.
    dedup = atomic_mode in ('sum', 'count')
    extra_point_launches = []

    out, written, order, order_sentinel = _gpu_init_buffers(
        height, width, fill, atomic_mode)

    (poly_geoms, poly_props, poly_ids, poly_global), \
        (line_geoms, line_props, line_global), \
        (point_geoms, point_props, point_global) = _classify_geometries(
            geometries, props_array)

    # 1. Polygons -- always use normal edge ranges (see _run_numpy comment)
    edge_arrays = _extract_edges(
        poly_geoms, poly_ids, bounds, height, width)
    edge_arrays = _sort_edges(edge_arrays)
    edge_y_min, edge_y_max, edge_x_at_ymin, edge_inv_slope, \
        edge_geom_id = edge_arrays

    # Stage the per-launch inputs once; first/last reuses these tensors
    # for the pass-2 stamp launches.
    #
    # ``poly_props`` and ``poly_global`` are referenced by both the
    # scanline ``poly_launch`` and the supercover ``boundary_launch``
    # under ``all_touched=True``. Hoist the host-to-device transfer above
    # the conditional so the two launches share the same device buffers;
    # without the hoist the props/global tables would be uploaded twice
    # per call. Skip the upload when neither launch will consume it
    # (``len(edge_y_min) == 0`` and ``not all_touched``), so the hoist
    # never costs more than the prior code. See issue #2506.
    poly_props_gpu = None
    poly_global_gpu = None
    if poly_geoms and (len(edge_y_min) > 0 or all_touched):
        poly_props_gpu = cupy.asarray(poly_props)
        poly_global_gpu = cupy.asarray(poly_global)

    poly_launch = None
    if len(edge_y_min) > 0:
        row_ptr, col_idx = _build_row_csr_numba(
            edge_y_min, edge_y_max, height)
        # The CUDA scanline kernel allocates a fixed-size local array; rows
        # exceeding _GPU_MAX_ISECT would have edges silently dropped and
        # produce wrong output.  Fail fast instead.
        _check_gpu_edge_cap(row_ptr)
        poly_launch = (
            cupy.asarray(edge_y_min), cupy.asarray(edge_x_at_ymin),
            cupy.asarray(edge_inv_slope), cupy.asarray(edge_geom_id),
            poly_props_gpu, poly_global_gpu,
            cupy.asarray(row_ptr), cupy.asarray(col_idx))

    # all_touched boundaries.  ``poly_geoms`` and ``poly_ids`` come
    # from ``_classify_geometries`` above and are always defined here,
    # so this block does not depend on the polygon scanline path
    # having allocated ``poly_launch``.  The supercover boundary kernel
    # walks every cell a ring segment crosses (float pixel endpoints),
    # replacing the older Bresenham boundary burn.
    boundary_launch = None
    if all_touched and poly_geoms:
        if dedup:
            brows, bcols, bgids = _boundary_cells_sum_count(
                poly_geoms, poly_ids, bounds, height, width, edge_arrays)
            if len(brows) > 0:
                extra_point_launches.append((
                    cupy.asarray(brows), cupy.asarray(bcols),
                    cupy.asarray(bgids), poly_props_gpu,
                    poly_global_gpu, len(brows)))
        else:
            bx0, by0, bx1, by1, bidx = (
                _extract_polygon_boundary_segments_float(
                    poly_geoms, poly_ids, bounds, height, width))
            if len(bx0) > 0:
                boundary_launch = (
                    cupy.asarray(bx0), cupy.asarray(by0),
                    cupy.asarray(bx1), cupy.asarray(by1),
                    cupy.asarray(bidx), poly_props_gpu,
                    poly_global_gpu, len(bx0))

    # all_touched routes lines through the supercover walk (issue #3102),
    # the same float-segment kernel the polygon boundaries use.
    line_launch = None
    line_supercover_launch = None
    if all_touched and line_geoms:
        lx0, ly0, lx1, ly1, line_idx = _extract_line_segments_float(
            line_geoms, bounds, height, width)
        if len(lx0) > 0:
            if dedup:
                lrows, lcols, lgids = _supercover_cells(
                    lx0, ly0, lx1, ly1, line_idx, height, width)
                lrows, lcols, lgids = _dedup_cells(
                    lrows, lcols, lgids, height, width)
                if len(lrows) > 0:
                    extra_point_launches.append((
                        cupy.asarray(lrows), cupy.asarray(lcols),
                        cupy.asarray(lgids), cupy.asarray(line_props),
                        cupy.asarray(line_global), len(lrows)))
            else:
                line_supercover_launch = (
                    cupy.asarray(lx0), cupy.asarray(ly0),
                    cupy.asarray(lx1), cupy.asarray(ly1),
                    cupy.asarray(line_idx), cupy.asarray(line_props),
                    cupy.asarray(line_global), len(lx0))
    else:
        r0, c0, r1, c1, line_idx = _extract_line_segments(
            line_geoms, bounds, height, width)
        if len(r0) > 0:
            if dedup:
                lrows, lcols, lgids = _bresenham_cells(
                    r0, c0, r1, c1, line_idx, height, width)
                lrows, lcols, lgids = _dedup_cells(
                    lrows, lcols, lgids, height, width)
                if len(lrows) > 0:
                    extra_point_launches.append((
                        cupy.asarray(lrows), cupy.asarray(lcols),
                        cupy.asarray(lgids), cupy.asarray(line_props),
                        cupy.asarray(line_global), len(lrows)))
            else:
                line_launch = (
                    cupy.asarray(r0), cupy.asarray(c0),
                    cupy.asarray(r1), cupy.asarray(c1),
                    cupy.asarray(line_idx), cupy.asarray(line_props),
                    cupy.asarray(line_global), len(r0))

    prows, pcols, pt_idx = _extract_points(
        point_geoms, bounds, height, width)
    point_launch = None
    if len(prows) > 0:
        point_launch = (
            cupy.asarray(prows), cupy.asarray(pcols),
            cupy.asarray(pt_idx), cupy.asarray(point_props),
            cupy.asarray(point_global), len(prows))

    def _launch_phase(k_scan, k_lines, k_boundary, k_points):
        tpb = 256
        if poly_launch is not None:
            blocks = (height + tpb - 1) // tpb
            k_scan[blocks, tpb](out, written, order,
                                poly_launch[0], poly_launch[1],
                                poly_launch[2], poly_launch[3],
                                poly_launch[4], poly_launch[5],
                                poly_launch[6], poly_launch[7], width)
        if boundary_launch is not None:
            n_bsegs = boundary_launch[7]
            bpg = (n_bsegs + tpb - 1) // tpb
            k_boundary[bpg, tpb](out, written, order,
                                 boundary_launch[0], boundary_launch[1],
                                 boundary_launch[2], boundary_launch[3],
                                 boundary_launch[4], boundary_launch[5],
                                 boundary_launch[6], n_bsegs, height, width)
        if line_launch is not None:
            n_segs = line_launch[7]
            bpg = (n_segs + tpb - 1) // tpb
            k_lines[bpg, tpb](out, written, order,
                              line_launch[0], line_launch[1],
                              line_launch[2], line_launch[3],
                              line_launch[4], line_launch[5],
                              line_launch[6], n_segs, height, width)
        if line_supercover_launch is not None:
            n_lsegs = line_supercover_launch[7]
            bpg = (n_lsegs + tpb - 1) // tpb
            k_boundary[bpg, tpb](out, written, order,
                                 line_supercover_launch[0],
                                 line_supercover_launch[1],
                                 line_supercover_launch[2],
                                 line_supercover_launch[3],
                                 line_supercover_launch[4],
                                 line_supercover_launch[5],
                                 line_supercover_launch[6],
                                 n_lsegs, height, width)
        if point_launch is not None:
            n_pts = point_launch[5]
            bpg = (n_pts + tpb - 1) // tpb
            k_points[bpg, tpb](out, written, order,
                               point_launch[0], point_launch[1],
                               point_launch[2], point_launch[3],
                               point_launch[4], n_pts)
        # Deduplicated sum/count line / boundary cells (issue #3304).
        # Only populated when ``dedup`` is true, which never coincides
        # with the two-pass first/last modes.
        for pl in extra_point_launches:
            n_pts = pl[5]
            bpg = (n_pts + tpb - 1) // tpb
            k_points[bpg, tpb](out, written, order,
                               pl[0], pl[1], pl[2], pl[3], pl[4], n_pts)

    # Pass 1: write values (atomics) or, for first/last, resolve the
    # per-pixel winning input index into ``order``.
    _launch_phase(kernels['scanline_fill'],
                  kernels['burn_lines'],
                  kernels['burn_lines_supercover'],
                  kernels['burn_points'])

    # Pass 2 (first / last only): stamp the winner's value into ``out``.
    if two_pass:
        _launch_phase(kernels['scanline_fill_pass2'],
                      kernels['burn_lines_pass2'],
                      kernels['burn_lines_supercover_pass2'],
                      kernels['burn_points_pass2'])

    final_out = _gpu_finalize_buffers(out, written, order, order_sentinel,
                                      fill, atomic_mode)
    # copy=False skips a full-raster device copy when dtype is already
    # float64 (the default).  The buffer is local to this call.  #3107.
    return final_out.astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# Dask tile-based rasterization
# ---------------------------------------------------------------------------

def _geometry_bboxes(geometries):
    """Return (N, 4) float64 array of [xmin, ymin, xmax, ymax] per geometry."""
    if len(geometries) == 0:
        return np.empty((0, 4), dtype=np.float64)
    shapely = _require_shapely()
    return shapely.bounds(np.asarray(geometries, dtype=object))


def _tile_grid(bounds, height, width, row_chunks, col_chunks):
    """Compute tile specs from output grid and chunk sizes.

    Returns list of (row_start, row_end, col_start, col_end, tile_bounds)
    where tile_bounds is (xmin, ymin, xmax, ymax) in geographic coords.
    """
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    tiles = []
    r = 0
    for rchunk in row_chunks:
        c = 0
        for cchunk in col_chunks:
            r_end = r + rchunk
            c_end = c + cchunk
            tile_xmin = xmin + c * px
            tile_xmax = xmin + c_end * px
            # y axis is top-down in pixel space: row 0 = ymax
            tile_ymax = ymax - r * py
            tile_ymin = ymax - r_end * py
            tiles.append((r, r_end, c, c_end,
                          (tile_xmin, tile_ymin, tile_xmax, tile_ymax)))
            c = c_end
        r = r_end
    return tiles


def _filter_geoms_to_tile(geom_bboxes, tile_bounds):
    """Return boolean mask of geometries whose bbox intersects tile_bounds.

    Uses strict ``<`` (not ``<=``) so that geometries touching the tile
    boundary are included.  The scanline fill uses ``floor()`` on
    x-intersections, so a polygon edge exactly at the tile boundary can
    still produce a pixel inside the tile.
    """
    if len(geom_bboxes) == 0:
        return np.empty(0, dtype=bool)
    txmin, tymin, txmax, tymax = tile_bounds
    return ~(
        (geom_bboxes[:, 2] < txmin) | (geom_bboxes[:, 0] > txmax) |
        (geom_bboxes[:, 3] < tymin) | (geom_bboxes[:, 1] > tymax)
    )


def _normalize_chunks(chunks, height, width):
    """Convert chunks parameter to (row_chunk_sizes, col_chunk_sizes) tuples."""
    if isinstance(chunks, int):
        rchunk = cchunk = chunks
    else:
        rchunk, cchunk = chunks

    # Both axes must have positive chunk sizes. A zero would loop forever
    # below; a negative would diverge.
    if rchunk <= 0 or cchunk <= 0:
        raise ValueError(
            f"chunks must be positive, got ({rchunk}, {cchunk})")

    row_chunks = []
    remaining = height
    while remaining > 0:
        row_chunks.append(min(rchunk, remaining))
        remaining -= row_chunks[-1]

    col_chunks = []
    remaining = width
    while remaining > 0:
        col_chunks.append(min(cchunk, remaining))
        remaining -= col_chunks[-1]

    return tuple(row_chunks), tuple(col_chunks)


def _segment_bboxes(r0, c0, r1, c1):
    """Pre-compute per-segment bounding boxes (once, before the tile loop)."""
    return (np.minimum(r0, r1), np.maximum(r0, r1),
            np.minimum(c0, c1), np.maximum(c0, c1))


def _segments_for_tile(r0, c0, r1, c1, geom_idx, seg_bboxes,
                       r_start, r_end, c_start, c_end):
    """Filter segments whose pixel bbox overlaps the tile, offset to local.

    Returns segments in tile-local coordinates (r_start/c_start subtracted).
    Endpoints can be negative or exceed tile dimensions -- the Bresenham
    bounds check in ``_burn_lines_cpu`` handles this, and the pixel path
    is translation-invariant so the result is exact.

    seg_bboxes is a (rmin, rmax, cmin, cmax) tuple from _segment_bboxes(),
    computed once before the tile loop to avoid redundant min/max per tile.

    geom_idx values are indices into the per-type props table and do not
    need adjustment when filtering.
    """
    if len(r0) == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty, empty, np.empty(0, dtype=np.int32)
    seg_rmin, seg_rmax, seg_cmin, seg_cmax = seg_bboxes
    mask = ((seg_rmax >= r_start) & (seg_rmin < r_end) &
            (seg_cmax >= c_start) & (seg_cmin < c_end))
    return (r0[mask] - r_start, c0[mask] - c_start,
            r1[mask] - r_start, c1[mask] - c_start,
            geom_idx[mask])


def _float_segments_for_tile(x0, y0, x1, y1, seg_bboxes,
                             r_start, r_end, c_start, c_end):
    """Filter float line segments with the same mask as integer segments.

    Uses the integer ``seg_bboxes`` (computed from the Bresenham
    endpoints) so the selection matches :func:`_segments_for_tile`
    exactly, keeping the float supercover segments aligned with the
    integer ``seg_geom_idx`` for the same tile (issue #3102).  Float
    endpoints are offset to tile-local pixel space; the supercover walk's
    bounds check handles endpoints landing outside the tile.
    """
    if len(x0) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    seg_rmin, seg_rmax, seg_cmin, seg_cmax = seg_bboxes
    mask = ((seg_rmax >= r_start) & (seg_rmin < r_end) &
            (seg_cmax >= c_start) & (seg_cmin < c_end))
    return (x0[mask] - c_start, y0[mask] - r_start,
            x1[mask] - c_start, y1[mask] - r_start)


def _points_for_tile(rows, cols, geom_idx, r_start, r_end, c_start, c_end):
    """Filter points within the tile, offset to tile-local coordinates.

    geom_idx values are indices into the per-type props table and do not
    need adjustment when filtering.
    """
    if len(rows) == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, np.empty(0, dtype=np.int32)
    mask = ((rows >= r_start) & (rows < r_end) &
            (cols >= c_start) & (cols < c_end))
    return (rows[mask] - r_start, cols[mask] - c_start, geom_idx[mask])


def _slice_props_for_tile(geom_idx, props):
    """Slice a per-type props table to only entries referenced by *geom_idx*.

    Returns ``(local_idx, sliced_props)`` where ``local_idx`` is a remapped
    int32 array of the same length as ``geom_idx`` but with values in
    ``range(len(sliced_props))``, and ``sliced_props`` is the
    contiguous subset of ``props`` that those local indices index into.

    This is the line / point counterpart to the polygon path's
    ``poly_wkb_arr[pmask]`` / ``poly_props[pmask]`` slicing.  Without it,
    every dask tile task would embed the full ``line_props`` /
    ``point_props`` table in its closure, even when the tile sees only
    a handful of segments.  At a few thousand geometries this is a
    multi-MB scheduler overhead per task; at 1 M geometries x 100 tiles
    the difference is ~800 MB of redundant graph state.

    If ``geom_idx`` is empty, returns an empty ``(0,)`` int32 array and
    a zero-row slice of ``props`` (preserving column count).
    """
    if len(geom_idx) == 0:
        return geom_idx.astype(np.int32, copy=False), props[:0]
    # np.unique returns sorted unique values and an inverse map that
    # points each entry in geom_idx to its position in unique_idx.
    unique_idx, local_idx = np.unique(geom_idx, return_inverse=True)
    return local_idx.astype(np.int32), props[unique_idx]


def _slice_per_tile(geom_idx, props, global_idx):
    """Same as ``_slice_props_for_tile`` but also slices the per-type
    global-input-index array so the tile worker can resolve ordered
    merges.  Returns ``(local_idx, sliced_props, sliced_global_idx)``.
    """
    if len(geom_idx) == 0:
        empty = global_idx[:0] if len(global_idx) else np.empty(0, dtype=np.int64)
        return (geom_idx.astype(np.int32, copy=False), props[:0], empty)
    unique_idx, local_idx = np.unique(geom_idx, return_inverse=True)
    return (local_idx.astype(np.int32), props[unique_idx],
            global_idx[unique_idx])


def _polys_to_wkb(geoms):
    """Pre-serialize polygon geometries to WKB for cheap pickling."""
    if not geoms:
        return []
    shapely = _require_shapely()
    return shapely.to_wkb(np.asarray(geoms, dtype=object)).tolist()


def _polys_from_wkb(wkb_list):
    """Deserialize WKB back to shapely geometries."""
    shapely = _require_shapely()
    geoms = shapely.from_wkb(wkb_list)
    if not isinstance(geoms, (list, np.ndarray)):
        geoms = [geoms]
    return list(geoms)


def _rasterize_tile_numpy(poly_wkb, poly_props_2d, poly_global_2d, tile_bounds,
                          tile_h, tile_w, fill, dtype, all_touched,
                          merge_fn, should_write,
                          seg_r0, seg_c0, seg_r1, seg_c1, seg_geom_idx,
                          line_props, line_global,
                          pt_rows, pt_cols, pt_geom_idx,
                          point_props, point_global,
                          lseg_x0, lseg_y0, lseg_x1, lseg_y1,
                          full_bounds, full_height, full_width,
                          row_off, col_off):
    """Rasterize a single tile.

    Polygons are passed as WKB bytes (cheap to pickle) and deserialized
    inside the worker.  Line segments and points are passed in tile-local
    pixel coordinates with geometry indices into their respective props
    tables.  Each per-type global-input-index array carries the original
    input position so order-sensitive merges work across types.

    Under ``all_touched`` the line segments arrive as float tile-local
    pixel coordinates (``lseg_*``) so the supercover walk can pick up
    every cell crossed, matching the eager path (issue #3102); the
    integer ``seg_*`` arrays carry the Bresenham endpoints used
    otherwise.  ``seg_geom_idx`` indexes ``line_props`` for both.
    """
    out = np.full((tile_h, tile_w), fill, dtype=np.float64)
    written = np.zeros((tile_h, tile_w), dtype=np.int8)
    # int8 for order-insensitive merges, int64 for first/last; the
    # decision runs worker-side, keyed on the unpickled predicate.
    # ``_should_write_any`` is a module-level dispatcher, so pickling
    # preserves identity.  Issue #3107.
    order = _alloc_order(tile_h, tile_w, should_write)

    dedup = _is_dedup_merge_fn(merge_fn)

    # 1. Polygons (deserialize WKB, then scanline fill)
    if poly_wkb:
        poly_geoms = _polys_from_wkb(poly_wkb)
        poly_ids = np.arange(len(poly_geoms), dtype=np.int32)
        edge_arrays = _extract_edges(
            poly_geoms, poly_ids, tile_bounds,
            tile_h, tile_w)
        edge_arrays = _sort_edges(edge_arrays)
        if len(edge_arrays[0]) > 0:
            _scanline_fill_cpu(out, written, *edge_arrays,
                               poly_props_2d, tile_h, tile_w,
                               poly_global_2d, merge_fn, should_write, order)

        # 1b. all_touched: draw polygon boundaries with the
        # supercover grid traversal so the dask path matches the
        # eager numpy path pixel-for-pixel under all_touched=True.
        # sum/count burn the deduplicated cell list instead, exactly
        # like the eager path (issue #3304).
        if all_touched:
            # Extract boundary float segments in the GLOBAL grid frame
            # then shift to tile-local space by the integer tile offset,
            # so the supercover walk lands on the same cells as the eager
            # backend even when a boundary segment falls on a pixel-grid
            # line (issue #3384).
            bx0, by0, bx1, by1, bidx = (
                _extract_polygon_boundary_segments_float_global(
                    poly_geoms, poly_ids, full_bounds,
                    full_height, full_width,
                    tile_bounds, row_off, col_off))
            if dedup:
                brows, bcols, bgids = _boundary_cells_sum_count(
                    poly_geoms, poly_ids, tile_bounds, tile_h, tile_w,
                    edge_arrays,
                    boundary_segments=(bx0, by0, bx1, by1, bidx))
                if len(brows) > 0:
                    _burn_points_cpu(out, written, brows, bcols, bgids,
                                     poly_props_2d, poly_global_2d,
                                     merge_fn, should_write, order)
            else:
                if len(bx0) > 0:
                    _burn_lines_supercover_cpu(
                        out, written, bx0, by0, bx1, by1, bidx,
                        poly_props_2d, tile_h, tile_w,
                        poly_global_2d, merge_fn, should_write, order)

    # 2. Lines (tile-local segments).  all_touched walks float segments
    #    with the supercover traversal (issue #3102); otherwise Bresenham
    #    runs on the integer endpoints.  sum/count dedup the visited cells
    #    first (issue #3304); cells partition across tiles, so per-tile
    #    dedup equals global dedup.
    if all_touched:
        if len(lseg_x0) > 0:
            if dedup:
                lrows, lcols, lgids = _supercover_cells(
                    lseg_x0, lseg_y0, lseg_x1, lseg_y1, seg_geom_idx,
                    tile_h, tile_w)
                lrows, lcols, lgids = _dedup_cells(
                    lrows, lcols, lgids, tile_h, tile_w)
                if len(lrows) > 0:
                    _burn_points_cpu(out, written, lrows, lcols, lgids,
                                     line_props, line_global,
                                     merge_fn, should_write, order)
            else:
                _burn_lines_supercover_cpu(
                    out, written, lseg_x0, lseg_y0, lseg_x1, lseg_y1,
                    seg_geom_idx, line_props, tile_h, tile_w,
                    line_global, merge_fn, should_write, order)
    elif len(seg_r0) > 0:
        if dedup:
            lrows, lcols, lgids = _bresenham_cells(
                seg_r0, seg_c0, seg_r1, seg_c1, seg_geom_idx,
                tile_h, tile_w)
            lrows, lcols, lgids = _dedup_cells(
                lrows, lcols, lgids, tile_h, tile_w)
            if len(lrows) > 0:
                _burn_points_cpu(out, written, lrows, lcols, lgids,
                                 line_props, line_global,
                                 merge_fn, should_write, order)
        else:
            _burn_lines_cpu(out, written, seg_r0, seg_c0, seg_r1, seg_c1,
                            seg_geom_idx, line_props, tile_h, tile_w,
                            line_global, merge_fn, should_write, order)

    # 3. Points (tile-local)
    if len(pt_rows) > 0:
        _burn_points_cpu(out, written, pt_rows, pt_cols, pt_geom_idx,
                         point_props, point_global,
                         merge_fn, should_write, order)

    # copy=False: no per-tile copy when dtype is already float64.  #3107.
    return out.astype(dtype, copy=False)


def _run_dask_numpy(geometries, props_array, bounds, height, width, fill,
                    dtype, all_touched, merge_fn, should_write,
                    row_chunks, col_chunks):
    """Dask + NumPy backend: tile-based parallel rasterization."""
    import dask
    import dask.array as da

    # Classify geometries once
    (poly_geoms, poly_props, _poly_ids, poly_global), \
        (line_geoms, line_props, line_global), \
        (point_geoms, point_props, point_global) = _classify_geometries(
            geometries, props_array)

    # Compact representations in full-raster pixel space
    seg_r0, seg_c0, seg_r1, seg_c1, seg_geom_idx = _extract_line_segments(
        line_geoms, bounds, height, width)
    pt_rows, pt_cols, pt_geom_idx = _extract_points(
        point_geoms, bounds, height, width)

    # Float line segments for the supercover walk under all_touched
    # (issue #3102).  Shares the clipping logic of _extract_line_segments
    # so it produces the same number of segments in the same order, which
    # keeps it aligned with seg_geom_idx and the per-tile filter mask.
    if all_touched:
        lseg_x0, lseg_y0, lseg_x1, lseg_y1, _ = _extract_line_segments_float(
            line_geoms, bounds, height, width)
    else:
        lseg_x0 = lseg_y0 = lseg_x1 = lseg_y1 = None

    # Pre-serialize polygons to WKB (20x cheaper to pickle than shapely).
    # Store as object array for single-pass boolean indexing per tile.
    if poly_geoms:
        poly_bboxes = _geometry_bboxes(poly_geoms)
        poly_wkb_arr = np.array(_polys_to_wkb(poly_geoms), dtype=object)
    else:
        poly_bboxes = np.empty((0, 4), dtype=np.float64)
        poly_wkb_arr = np.empty(0, dtype=object)

    # Pre-compute segment bboxes once (avoids redundant min/max per tile)
    seg_bboxes = _segment_bboxes(seg_r0, seg_c0, seg_r1, seg_c1)

    tiles = _tile_grid(bounds, height, width, row_chunks, col_chunks)
    n_row_chunks = len(row_chunks)
    n_col_chunks = len(col_chunks)
    blocks = [[None] * n_col_chunks for _ in range(n_row_chunks)]

    ri = 0
    for i in range(n_row_chunks):
        for j in range(n_col_chunks):
            r_start, r_end, c_start, c_end, tile_bounds = tiles[ri]
            tile_h = r_end - r_start
            tile_w = c_end - c_start

            # Filter polygons by tile geo bbox (single boolean index)
            pmask = _filter_geoms_to_tile(poly_bboxes, tile_bounds)
            if len(poly_wkb_arr) > 0:
                tile_wkb = poly_wkb_arr[pmask].tolist()
                tile_poly_props = poly_props[pmask]
                tile_poly_global = poly_global[pmask]
            else:
                tile_wkb = []
                tile_poly_props = poly_props[:0]  # empty (0, P)
                tile_poly_global = np.empty(0, dtype=np.int64)

            # Filter segments and points by tile pixel range
            ts = _segments_for_tile(seg_r0, seg_c0, seg_r1, seg_c1,
                                    seg_geom_idx, seg_bboxes,
                                    r_start, r_end, c_start, c_end)
            tp = _points_for_tile(pt_rows, pt_cols, pt_geom_idx,
                                  r_start, r_end, c_start, c_end)

            # all_touched: float supercover segments, filtered by the
            # identical mask so they stay aligned with ts geom indices.
            if all_touched:
                lts = _float_segments_for_tile(
                    lseg_x0, lseg_y0, lseg_x1, lseg_y1, seg_bboxes,
                    r_start, r_end, c_start, c_end)
            else:
                _empty_f = np.empty(0, dtype=np.float64)
                lts = (_empty_f, _empty_f, _empty_f, _empty_f)

            # Slice line_props / point_props (and their global-idx
            # companions) to only the geometries this tile actually
            # references, remapping the geom_idx arrays to local indices.
            ts_local_idx, tile_line_props, tile_line_global = \
                _slice_per_tile(ts[4], line_props, line_global)
            ts = (*ts[:4], ts_local_idx)
            tp_local_idx, tile_point_props, tile_point_global = \
                _slice_per_tile(tp[2], point_props, point_global)
            tp = (*tp[:2], tp_local_idx)

            delayed_tile = dask.delayed(_rasterize_tile_numpy)(
                tile_wkb, tile_poly_props, tile_poly_global, tile_bounds,
                tile_h, tile_w, fill, dtype, all_touched,
                merge_fn, should_write,
                *ts, tile_line_props, tile_line_global,
                *tp, tile_point_props, tile_point_global,
                *lts,
                bounds, height, width, r_start, c_start)
            blocks[i][j] = da.from_delayed(
                delayed_tile, shape=(tile_h, tile_w), dtype=dtype)
            ri += 1

    return da.block(blocks)


def _ensure_cupy(arr):
    """Transfer to GPU only if not already a CuPy array."""
    if isinstance(arr, cupy.ndarray):
        return arr
    return cupy.asarray(arr)


def _rasterize_tile_cupy(poly_wkb, poly_props_2d, poly_global_2d, tile_bounds,
                         tile_h, tile_w, fill, dtype, all_touched,
                         merge_fn, should_write,
                         seg_r0, seg_c0, seg_r1, seg_c1, seg_geom_idx,
                         line_props, line_global,
                         pt_rows, pt_cols, pt_geom_idx,
                         point_props, point_global,
                         lseg_x0, lseg_y0, lseg_x1, lseg_y1,
                         full_bounds, full_height, full_width,
                         row_off, col_off,
                         merge_name=None):
    """GPU tile rasterization: polygons as WKB, lines/points as segments.

    When ``merge_name`` selects an atomic built-in merge, the per-tile
    kernels use CUDA atomics so overlapping writes inside the tile are
    deterministic.  Cross-tile pixels do not overlap (the tiling is a
    partition of the output grid), so per-tile atomicity is sufficient.
    """
    kernels = _ensure_gpu_kernels(merge_fn, should_write, merge_name)
    atomic_mode = kernels['atomic_mode']
    two_pass = atomic_mode in ('first', 'last')
    # Per-geometry write dedup for sum/count (issue #3304), mirroring
    # _run_cupy: boundary / line cells enumerated host-side, burned via
    # the point kernel.
    dedup = atomic_mode in ('sum', 'count')
    extra_point_launches = []

    out, written, order, order_sentinel = _gpu_init_buffers(
        tile_h, tile_w, fill, atomic_mode)

    # Stage launches (same pattern as _run_cupy).  Stage once so the
    # first/last pass-2 sweep reuses the device buffers.  ``poly_geoms``
    # and ``poly_ids`` are bound inside the ``if poly_wkb:`` block, so
    # ``boundary_launch`` is staged inside the same block to keep the
    # dependency local.
    poly_launch = None
    boundary_launch = None
    if poly_wkb:
        poly_geoms = _polys_from_wkb(poly_wkb)
        poly_ids = np.arange(len(poly_geoms), dtype=np.int32)
        edge_arrays = _extract_edges(
            poly_geoms, poly_ids, tile_bounds, tile_h, tile_w)
        edge_arrays = _sort_edges(edge_arrays)
        edge_y_min = edge_arrays[0]
        # Upload ``poly_props_2d`` and ``poly_global_2d`` once; both the
        # scanline ``poly_launch`` and the supercover ``boundary_launch``
        # under ``all_touched=True`` reference these tables, and without
        # the hoist they would be transferred twice per tile. Skip the
        # upload when neither launch will consume it (no edges and not
        # ``all_touched``) so the hoist never costs more than the prior
        # code. Issue #2506.
        poly_props_2d_gpu = None
        poly_global_2d_gpu = None
        if len(edge_y_min) > 0 or all_touched:
            poly_props_2d_gpu = cupy.asarray(poly_props_2d)
            poly_global_2d_gpu = cupy.asarray(poly_global_2d)
        if len(edge_y_min) > 0:
            edge_y_max, edge_x_at_ymin, edge_inv_slope, edge_geom_id = \
                edge_arrays[1:]
            row_ptr, col_idx = _build_row_csr_numba(
                edge_y_min, edge_y_max, tile_h)
            _check_gpu_edge_cap(row_ptr)
            poly_launch = (
                cupy.asarray(edge_y_min), cupy.asarray(edge_x_at_ymin),
                cupy.asarray(edge_inv_slope), cupy.asarray(edge_geom_id),
                poly_props_2d_gpu, poly_global_2d_gpu,
                cupy.asarray(row_ptr), cupy.asarray(col_idx))

        # all_touched: stage the supercover boundary burn through
        # ``boundary_launch`` so the two-pass first/last path can fire
        # both pass-1 and pass-2 kernels.  sum/count stage the
        # deduplicated cell list as a point launch instead (#3304).
        if all_touched:
            # Boundary float segments in the GLOBAL grid frame, shifted
            # to tile-local space by the integer tile offset, so the
            # supercover walk matches the eager backend on pixel-grid
            # boundary lines (issue #3384).
            bx0, by0, bx1, by1, bidx = (
                _extract_polygon_boundary_segments_float_global(
                    poly_geoms, poly_ids, full_bounds,
                    full_height, full_width,
                    tile_bounds, row_off, col_off))
            if dedup:
                brows, bcols, bgids = _boundary_cells_sum_count(
                    poly_geoms, poly_ids, tile_bounds, tile_h, tile_w,
                    edge_arrays,
                    boundary_segments=(bx0, by0, bx1, by1, bidx))
                if len(brows) > 0:
                    extra_point_launches.append((
                        cupy.asarray(brows), cupy.asarray(bcols),
                        cupy.asarray(bgids), poly_props_2d_gpu,
                        poly_global_2d_gpu, len(brows)))
            else:
                if len(bx0) > 0:
                    boundary_launch = (
                        cupy.asarray(bx0), cupy.asarray(by0),
                        cupy.asarray(bx1), cupy.asarray(by1),
                        cupy.asarray(bidx), poly_props_2d_gpu,
                        poly_global_2d_gpu, len(bx0))

    # all_touched routes lines through the supercover walk on float
    # segments (issue #3102); otherwise Bresenham runs on the integer
    # endpoints.
    line_launch = None
    line_supercover_launch = None
    if all_touched:
        if len(lseg_x0) > 0:
            if dedup:
                lrows, lcols, lgids = _supercover_cells(
                    lseg_x0, lseg_y0, lseg_x1, lseg_y1, seg_geom_idx,
                    tile_h, tile_w)
                lrows, lcols, lgids = _dedup_cells(
                    lrows, lcols, lgids, tile_h, tile_w)
                if len(lrows) > 0:
                    extra_point_launches.append((
                        cupy.asarray(lrows), cupy.asarray(lcols),
                        cupy.asarray(lgids), _ensure_cupy(line_props),
                        _ensure_cupy(line_global), len(lrows)))
            else:
                line_supercover_launch = (
                    cupy.asarray(lseg_x0), cupy.asarray(lseg_y0),
                    cupy.asarray(lseg_x1), cupy.asarray(lseg_y1),
                    cupy.asarray(seg_geom_idx), _ensure_cupy(line_props),
                    _ensure_cupy(line_global), len(lseg_x0))
    elif len(seg_r0) > 0:
        if dedup:
            lrows, lcols, lgids = _bresenham_cells(
                seg_r0, seg_c0, seg_r1, seg_c1, seg_geom_idx,
                tile_h, tile_w)
            lrows, lcols, lgids = _dedup_cells(
                lrows, lcols, lgids, tile_h, tile_w)
            if len(lrows) > 0:
                extra_point_launches.append((
                    cupy.asarray(lrows), cupy.asarray(lcols),
                    cupy.asarray(lgids), _ensure_cupy(line_props),
                    _ensure_cupy(line_global), len(lrows)))
        else:
            line_launch = (
                cupy.asarray(seg_r0), cupy.asarray(seg_c0),
                cupy.asarray(seg_r1), cupy.asarray(seg_c1),
                cupy.asarray(seg_geom_idx), _ensure_cupy(line_props),
                _ensure_cupy(line_global), len(seg_r0))

    point_launch = None
    if len(pt_rows) > 0:
        point_launch = (
            cupy.asarray(pt_rows), cupy.asarray(pt_cols),
            cupy.asarray(pt_geom_idx), _ensure_cupy(point_props),
            _ensure_cupy(point_global), len(pt_rows))

    def _launch_phase(k_scan, k_lines, k_boundary, k_points):
        tpb = 256
        if poly_launch is not None:
            blocks = (tile_h + tpb - 1) // tpb
            k_scan[blocks, tpb](out, written, order,
                                poly_launch[0], poly_launch[1],
                                poly_launch[2], poly_launch[3],
                                poly_launch[4], poly_launch[5],
                                poly_launch[6], poly_launch[7], tile_w)
        if boundary_launch is not None:
            n_bsegs = boundary_launch[7]
            bpg = (n_bsegs + tpb - 1) // tpb
            k_boundary[bpg, tpb](out, written, order,
                                 boundary_launch[0], boundary_launch[1],
                                 boundary_launch[2], boundary_launch[3],
                                 boundary_launch[4], boundary_launch[5],
                                 boundary_launch[6],
                                 n_bsegs, tile_h, tile_w)
        if line_launch is not None:
            n_segs = line_launch[7]
            bpg = (n_segs + tpb - 1) // tpb
            k_lines[bpg, tpb](out, written, order,
                              line_launch[0], line_launch[1],
                              line_launch[2], line_launch[3],
                              line_launch[4], line_launch[5],
                              line_launch[6], n_segs, tile_h, tile_w)
        if line_supercover_launch is not None:
            n_lsegs = line_supercover_launch[7]
            bpg = (n_lsegs + tpb - 1) // tpb
            k_boundary[bpg, tpb](out, written, order,
                                 line_supercover_launch[0],
                                 line_supercover_launch[1],
                                 line_supercover_launch[2],
                                 line_supercover_launch[3],
                                 line_supercover_launch[4],
                                 line_supercover_launch[5],
                                 line_supercover_launch[6],
                                 n_lsegs, tile_h, tile_w)
        if point_launch is not None:
            n_pts = point_launch[5]
            bpg = (n_pts + tpb - 1) // tpb
            k_points[bpg, tpb](out, written, order,
                               point_launch[0], point_launch[1],
                               point_launch[2], point_launch[3],
                               point_launch[4], n_pts)
        # Deduplicated sum/count line / boundary cells (issue #3304).
        for pl in extra_point_launches:
            n_pts = pl[5]
            bpg = (n_pts + tpb - 1) // tpb
            k_points[bpg, tpb](out, written, order,
                               pl[0], pl[1], pl[2], pl[3], pl[4], n_pts)

    _launch_phase(kernels['scanline_fill'],
                  kernels['burn_lines'],
                  kernels['burn_lines_supercover'],
                  kernels['burn_points'])
    if two_pass:
        _launch_phase(kernels['scanline_fill_pass2'],
                      kernels['burn_lines_pass2'],
                      kernels['burn_lines_supercover_pass2'],
                      kernels['burn_points_pass2'])

    final_out = _gpu_finalize_buffers(out, written, order, order_sentinel,
                                      fill, atomic_mode)
    # copy=False: no per-tile device copy when dtype is already float64.
    # Issue #3107.
    return final_out.astype(dtype, copy=False)


def _run_dask_cupy(geometries, props_array, bounds, height, width, fill,
                   dtype, all_touched, merge_fn, should_write,
                   row_chunks, col_chunks, merge_name=None):
    """Dask + CuPy backend: tile-based parallel GPU rasterization."""
    import dask
    import dask.array as da

    # Classify geometries once
    (poly_geoms, poly_props, _poly_ids, poly_global), \
        (line_geoms, line_props, line_global), \
        (point_geoms, point_props, point_global) = _classify_geometries(
            geometries, props_array)

    # Compact representations in full-raster pixel space
    seg_r0, seg_c0, seg_r1, seg_c1, seg_geom_idx = _extract_line_segments(
        line_geoms, bounds, height, width)
    pt_rows, pt_cols, pt_geom_idx = _extract_points(
        point_geoms, bounds, height, width)

    # Float line segments for the supercover walk under all_touched
    # (issue #3102); aligned with seg_geom_idx via the shared clip logic.
    if all_touched:
        lseg_x0, lseg_y0, lseg_x1, lseg_y1, _ = _extract_line_segments_float(
            line_geoms, bounds, height, width)
    else:
        lseg_x0 = lseg_y0 = lseg_x1 = lseg_y1 = None

    # Pre-serialize polygons to WKB (20x cheaper to pickle than shapely).
    if poly_geoms:
        poly_bboxes = _geometry_bboxes(poly_geoms)
        poly_wkb_arr = np.array(_polys_to_wkb(poly_geoms), dtype=object)
    else:
        poly_bboxes = np.empty((0, 4), dtype=np.float64)
        poly_wkb_arr = np.empty(0, dtype=object)

    # Pre-compute segment bboxes once (avoids redundant min/max per tile)
    seg_bboxes = _segment_bboxes(seg_r0, seg_c0, seg_r1, seg_c1)

    # Per-tile props slicing keeps PCIe traffic and graph payload small:
    # each tile only references a subset of geometries, so we slice
    # line_props / point_props per tile (as np arrays in the graph) and
    # transfer only that subset to the GPU inside the worker.  Mirrors
    # the polygon path's poly_props[pmask] pattern.  Tradeoff: for
    # sprawling-line workloads where every tile references most rows,
    # the GPU now sees N_tiles small uploads instead of one shared
    # driver-side upload, shifting cost from graph payload to PCIe
    # traffic.  Localized geometries (the realistic case) still win.

    tiles = _tile_grid(bounds, height, width, row_chunks, col_chunks)
    n_row_chunks = len(row_chunks)
    n_col_chunks = len(col_chunks)
    blocks = [[None] * n_col_chunks for _ in range(n_row_chunks)]

    ri = 0
    for i in range(n_row_chunks):
        for j in range(n_col_chunks):
            r_start, r_end, c_start, c_end, tile_bounds = tiles[ri]
            tile_h = r_end - r_start
            tile_w = c_end - c_start

            # Filter polygons by tile geo bbox (single boolean index)
            pmask = _filter_geoms_to_tile(poly_bboxes, tile_bounds)
            if len(poly_wkb_arr) > 0:
                tile_wkb = poly_wkb_arr[pmask].tolist()
                tile_poly_props = poly_props[pmask]
                tile_poly_global = poly_global[pmask]
            else:
                tile_wkb = []
                tile_poly_props = poly_props[:0]  # empty (0, P)
                tile_poly_global = np.empty(0, dtype=np.int64)

            ts = _segments_for_tile(seg_r0, seg_c0, seg_r1, seg_c1,
                                    seg_geom_idx, seg_bboxes,
                                    r_start, r_end, c_start, c_end)
            tp = _points_for_tile(pt_rows, pt_cols, pt_geom_idx,
                                  r_start, r_end, c_start, c_end)

            # all_touched: float supercover segments, same mask as ts.
            if all_touched:
                lts = _float_segments_for_tile(
                    lseg_x0, lseg_y0, lseg_x1, lseg_y1, seg_bboxes,
                    r_start, r_end, c_start, c_end)
            else:
                _empty_f = np.empty(0, dtype=np.float64)
                lts = (_empty_f, _empty_f, _empty_f, _empty_f)

            ts_local_idx, tile_line_props, tile_line_global = \
                _slice_per_tile(ts[4], line_props, line_global)
            ts = (*ts[:4], ts_local_idx)
            tp_local_idx, tile_point_props, tile_point_global = \
                _slice_per_tile(tp[2], point_props, point_global)
            tp = (*tp[:2], tp_local_idx)

            delayed_tile = dask.delayed(_rasterize_tile_cupy)(
                tile_wkb, tile_poly_props, tile_poly_global, tile_bounds,
                tile_h, tile_w, fill, dtype, all_touched,
                merge_fn, should_write,
                *ts, tile_line_props, tile_line_global,
                *tp, tile_point_props, tile_point_global,
                *lts,
                bounds, height, width, r_start, c_start,
                merge_name)
            blocks[i][j] = da.from_delayed(
                delayed_tile, shape=(tile_h, tile_w), dtype=dtype,
                meta=cupy.empty(()))
            ri += 1

    return da.block(blocks)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _is_categorical_like(series):
    """True when a GeoDataFrame column should be label-encoded.

    Catches pandas ``CategoricalDtype`` and any non-numeric dtype
    (object / string).  Bool stays numeric (it burns as 0/1), matching
    the historical ``.astype(np.float64)`` behaviour.
    """
    import pandas as pd
    return (isinstance(series.dtype, pd.CategoricalDtype)
            or not pd.api.types.is_numeric_dtype(series.dtype))


def _encode_categorical(series):
    """Label-encode a column to integer codes plus an ordered name list.

    An existing ``CategoricalDtype`` keeps its declared category order;
    a plain string/object column is converted with ``astype('category')``,
    whose categories come out lexically sorted.  Missing values keep the
    pandas ``-1`` code, which the caller maps to the fill value.
    """
    import pandas as pd
    cat = series if isinstance(series.dtype, pd.CategoricalDtype) \
        else series.astype('category')
    codes = cat.cat.codes.to_numpy()
    names = [str(c) for c in cat.cat.categories]
    return codes, names


def _categorical_colors(n):
    """Generate ``n`` distinct opaque RGBA colors from an HSV spread.

    Evenly spaced hues at fixed saturation/value give visually separable
    classes without pulling in matplotlib.  Returns a list of
    ``(r, g, b, a)`` int tuples with components in 0-255.
    """
    import colorsys
    colors = []
    for i in range(n):
        h = i / n if n else 0.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        colors.append((round(r * 255), round(g * 255), round(b * 255), 255))
    return colors


def _parse_input(geometries, column=None, columns=None):
    """Normalise input to (geometry_list, props_array, bounds, crs, cats).

    Returns
    -------
    geom_list : list of shapely geometries
    props_array : (N, P) float64 array of property values
    bounds : tuple or None
    crs : the GeoDataFrame's ``.crs`` (any pyproj-parseable value) or
        ``None``.  Only a GeoDataFrame exposes a CRS; the
        ``(geometry, value)`` iterable path always returns ``None``.
    category_names : ordered list of label strings when ``column`` named a
        string/categorical field, else ``None``.  When set, ``props_array``
        holds the integer codes (``-1`` for missing).
    """
    # Handle dask-geopandas by materializing eagerly.  Geometry data is
    # typically much smaller than the output raster, so this is fine.
    try:
        import dask_geopandas
        if isinstance(geometries, dask_geopandas.GeoDataFrame):
            geometries = geometries.compute()
    except ImportError:
        pass

    try:
        import geopandas as gpd
        if isinstance(geometries, gpd.GeoDataFrame):
            geom_list = geometries.geometry.tolist()
            total_bounds = tuple(geometries.total_bounds)
            # GeoPandas returns (nan, nan, nan, nan) for an empty frame.
            # Treat as "no inferred bounds" so the caller's explicit-bounds
            # guard fires instead of producing a raster with nan coords.
            if any(not np.isfinite(v) for v in total_bounds):
                total_bounds = None
            category_names = None
            if columns is not None:
                props_array = geometries[columns].values.astype(np.float64)
            else:
                if column is None:
                    numeric_cols = geometries.select_dtypes(
                        include='number').columns
                    if len(numeric_cols) == 0:
                        raise ValueError(
                            "GeoDataFrame has no numeric columns to burn. "
                            "Pass a 'column' name explicitly.")
                    column = numeric_cols[0]
                col = geometries[column]
                if _is_categorical_like(col):
                    codes, category_names = _encode_categorical(col)
                    props_array = codes.astype(np.float64).reshape(-1, 1)
                else:
                    props_array = col.values.astype(
                        np.float64).reshape(-1, 1)
            return (geom_list, props_array, total_bounds,
                    geometries.crs, category_names)
    except ImportError:
        pass

    if not hasattr(geometries, '__iter__'):
        raise TypeError(
            "geometries must be a GeoDataFrame or iterable of "
            "(geometry, value) pairs")
    geom_list = []
    value_list = []
    for item in geometries:
        geom_list.append(item[0])
        value_list.append(float(item[1]))

    if not geom_list:
        props_array = np.empty((0, 1), dtype=np.float64)
        return geom_list, props_array, None, None, None

    props_array = np.array(value_list, dtype=np.float64).reshape(-1, 1)

    # Bounds computation is deferred: return None here and let the
    # caller compute bboxes only when bounds are actually needed.
    return geom_list, props_array, None, None, None


def _check_uniform_axis(axis_name, coords, expected_step):
    """Raise ``ValueError`` if ``coords`` is not uniformly spaced.

    ``coords`` is a 1-D float64 array of dim-coord values for the named
    axis (``'x'`` or ``'y'``).  ``expected_step`` is the magnitude of the
    first interval (already computed by the caller).  Axes with fewer
    than three points cannot be non-uniform in a way this check would
    catch, so they pass trivially.

    The comparison uses the *signed* ``np.diff`` against the signed
    first interval so the validation does not care whether the axis
    is ascending or descending -- the sign of the first interval
    carries the direction.  This also rejects zig-zag /
    duplicate-coord patterns like ``[0.5, 1.5, 0.5, 1.5]`` whose
    ``abs(diff)`` is uniform but whose signed diffs alternate.
    ``np.allclose`` is used (rather than strict equality) because
    affine-transform-derived coords drift by a few ulps in practice.
    """
    if coords.size < 3:
        return

    # A non-finite ``expected_step`` (NaN / inf) is a separate kind of
    # broken; let the downstream rasterizer surface that rather than
    # masking it with a misleading "non-uniform spacing" message here.
    if not np.isfinite(expected_step):
        return

    # An all-equal coord vector gives expected_step==0 and diffs all
    # zero; allclose would pass and the rasterizer would later trip on
    # a zero-sized pixel.  Reject up front with the same "non-uniform"
    # framing so users get one diagnostic, not two.
    if expected_step == 0:
        raise ValueError(
            f"'like' DataArray has zero-width pixels along the "
            f"{axis_name!r} axis (consecutive coords are equal). "
            "rasterize() requires a regular grid with non-zero "
            "spacing; resample 'like' to a uniform grid (e.g. with "
            "xarray's ``interp`` or ``reindex``) before passing it."
        )

    # Compare signed diffs, not magnitudes.  Comparing only
    # ``abs(diff)`` against ``abs(expected_step)`` accepts zig-zag
    # patterns like ``[0.5, 1.5, 0.5, 1.5]`` whose magnitudes are
    # uniform but whose coords are non-monotonic with duplicate
    # values -- which then poisons ``.sel`` and any other coord-aware
    # lookup on the output.
    signed_step = float(coords[1] - coords[0])
    signed_diffs = np.diff(coords)
    if not np.allclose(signed_diffs, signed_step, rtol=1e-5, atol=1e-8):
        max_dev = float(np.max(np.abs(signed_diffs - signed_step)))
        raise ValueError(
            "'like' DataArray has non-uniform spacing along the "
            f"{axis_name!r} axis (expected step {signed_step}, "
            f"largest deviation {max_dev}). rasterize() requires a "
            "regular, strictly monotonic grid; resample 'like' to a "
            "uniform grid (e.g. with xarray's ``interp`` or "
            "``reindex``) before passing it."
        )


class _LikeGrid(NamedTuple):
    """Grid attributes extracted from a template DataArray.

    Returned by :func:`_extract_grid_from_like`.  Using a named tuple
    instead of a bare tuple keeps the call site readable as more
    template-derived attributes accrete over time.
    """
    width: int
    height: int
    bounds: Tuple[float, float, float, float]
    dtype: np.dtype
    x_coord: xr.DataArray
    y_coord: xr.DataArray
    extra_coords: dict
    attrs: dict
    y_ascending: bool
    x_descending: bool


def _extract_grid_from_like(like):
    """Extract width, height, bounds, dtype from a template DataArray.

    Also returns the input coords and attrs so the caller can propagate
    them onto the output raster.  Reusing ``like.coords`` directly (rather
    than rebuilding with ``linspace``) keeps the output bit-identical to
    the template and so preserves ``xr.align`` compatibility.
    """
    if not isinstance(like, xr.DataArray):
        raise TypeError("'like' must be an xr.DataArray")
    if like.ndim != 2 or 'y' not in like.dims or 'x' not in like.dims:
        raise ValueError(
            "'like' DataArray must be 2D with 'y' and 'x' dimensions")

    height = like.sizes['y']
    width = like.sizes['x']
    dt = like.dtype

    x = like.coords['x'].values.astype(np.float64)
    y = like.coords['y'].values.astype(np.float64)

    if width > 1:
        px = abs(float(x[1] - x[0]))
    else:
        px = 1.0
    if height > 1:
        py = abs(float(y[0] - y[1]))
    else:
        py = 1.0

    # The rasterizer assumes a uniform grid.  If ``like`` has non-uniform
    # spacing on either axis, ``px``/``py`` (taken from the first interval)
    # will not describe the rest of the grid, and reusing ``like.coords``
    # on the output (see ``rasterize`` below) would mislabel where each
    # pixel lives.  Validate uniform spacing here so the rasterizer never
    # produces a DataArray whose coords disagree with its data layout.
    #
    # Compare *signed* diffs against the signed first interval so the
    # check accepts ascending and descending axes (the sign of the
    # first interval carries the direction) but rejects zig-zag /
    # duplicate-coord patterns whose abs(diff) happens to be uniform.
    # Use ``np.allclose`` rather than strict equality because affine-
    # transform-derived coords drift by a few ulps.
    _check_uniform_axis('x', x, px)
    _check_uniform_axis('y', y, py)

    xmin = float(np.min(x)) - px / 2
    xmax = float(np.max(x)) + px / 2
    ymin = float(np.min(y)) - py / 2
    ymax = float(np.max(y)) + py / 2

    # Detect axis orientation.  The rasterizer always burns with row 0
    # at ymax and column 0 at xmin (standard image convention).  If the
    # template's y is ascending (low-to-high) or its x is descending
    # (high-to-low), the burned array has to be flipped along that axis
    # before we hand back the coords, or downstream coord-aware ops line
    # up against the wrong cells.  Only the first and last samples are
    # inspected; the ``_check_uniform_axis`` call above has already
    # rejected non-monotonic or duplicate-valued coords on both axes,
    # so this is safe.  Single-row / single-column templates have no
    # direction to detect, so the ``> 1`` guard short-circuits to
    # ``False`` (no flip) for those.
    y_ascending = height > 1 and float(y[-1]) > float(y[0])
    x_descending = width > 1 and float(x[-1]) < float(x[0])

    # Carry through any non-dim coords (e.g. rioxarray's ``spatial_ref``
    # CRS coord).  The y/x dim coords are returned separately because the
    # caller decides whether to reuse them (bit-identical grid) or build
    # fresh ones (resized grid).
    extra_coords = {
        k: v for k, v in like.coords.items() if k not in ('x', 'y')
    }

    return _LikeGrid(
        width=width,
        height=height,
        bounds=(xmin, ymin, xmax, ymax),
        dtype=dt,
        x_coord=like.coords['x'],
        y_coord=like.coords['y'],
        extra_coords=extra_coords,
        attrs=dict(like.attrs),
        y_ascending=y_ascending,
        x_descending=x_descending,
    )


def _like_crs(like):
    """Detect the CRS of a ``like`` template DataArray, or ``None``.

    Mirrors the resolution order in
    ``xrspatial.polygonize._detect_raster_crs`` so a template carries the
    same CRS into rasterize that polygonize would read back out:

    1. ``like.attrs['crs']`` (xrspatial.geotiff convention)
    2. ``like.attrs['crs_wkt']``
    3. the ``spatial_ref`` non-dim coord's ``crs_wkt`` / ``spatial_ref``
       attr (rioxarray's georeference coord)
    4. ``like.rio.crs`` (rioxarray, if installed)
    5. ``None``

    The raw value is returned (string / int / WKT / pyproj.CRS) so the
    caller normalizes it the same way the geometry CRS is normalized.
    """
    crs_attr = like.attrs.get('crs')
    if crs_attr is not None:
        return crs_attr

    crs_wkt = like.attrs.get('crs_wkt')
    if crs_wkt is not None:
        return crs_wkt

    if 'spatial_ref' in like.coords:
        sr_attrs = like.coords['spatial_ref'].attrs
        sr_crs = sr_attrs.get('crs_wkt') or sr_attrs.get('spatial_ref')
        if sr_crs is not None:
            return sr_crs

    try:
        rio_crs = like.rio.crs
        if rio_crs is not None:
            return rio_crs
    except Exception:
        # Broad on purpose (matches polygonize._detect_raster_crs):
        # ``.rio`` raises AttributeError when rioxarray is not installed,
        # and rio.crs can raise on a malformed georeference.  Either way
        # we treat the template as having no detectable CRS and let the
        # earlier attrs/spatial_ref paths be the source of truth.
        pass

    return None


def _check_crs_match(geom_crs, like_crs):
    """Raise ``ValueError`` if ``geom_crs`` and ``like_crs`` disagree.

    Both values are normalized through ``pyproj.CRS`` so an int EPSG
    code, an ``"EPSG:xxxx"`` string, a WKT string, and a ``pyproj.CRS``
    instance describing the same system all compare equal.  When either
    side is ``None`` (no CRS to compare) the check is a no-op -- this
    keeps CRS-less geometry lists and templates working unchanged.  An
    unparseable value on either side raises ``ValueError`` rather than
    silently disabling the guard.
    """
    if geom_crs is None or like_crs is None:
        return

    from pyproj import CRS as _PyprojCRS
    from pyproj.exceptions import CRSError

    def _norm(value, label):
        try:
            return _PyprojCRS(value)
        except CRSError as e:
            raise ValueError(
                f"{label} CRS {value!r} is not a valid CRS: {e}"
            ) from e

    g = _norm(geom_crs, 'geometry')
    t = _norm(like_crs, "'like' template")
    if not g.equals(t):
        # Prefer a compact "EPSG:xxxx" label over the raw value, whose
        # repr is a multi-line WKT block for a pyproj/geopandas CRS.
        # Fall back to the CRS name when no EPSG code is available.
        def _label(crs):
            epsg = crs.to_epsg()
            return f"EPSG:{epsg}" if epsg is not None else crs.name
        raise ValueError(
            f"CRS mismatch: geometries are in {_label(g)} but the "
            f"'like' template is in {_label(t)}. Reproject the "
            f"geometries to match the template, or pass check_crs=False "
            f"to rasterize onto the template grid anyway (the output "
            f"will inherit the template CRS without reprojection)."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rasterize(
    geometries: Any,
    width: Optional[int] = None,
    height: Optional[int] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    column: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    fill: float = np.nan,
    dtype: Optional[Union[str, np.dtype, type]] = None,
    all_touched: bool = False,
    gpu: bool = False,
    name: str = 'rasterize',
    resolution: Optional[Union[float, Tuple[float, float]]] = None,
    like: Optional[xr.DataArray] = None,
    merge: Union[str, Callable] = 'last',
    chunks: Optional[Union[int, Tuple[int, int]]] = None,
    max_pixels: int = MAX_PIXELS_DEFAULT,
    check_crs: bool = True,
    use_cuda: Optional[bool] = None,
) -> xr.DataArray:
    """Rasterize vector geometries into a 2D DataArray.

    Converts geometries from a GeoDataFrame or a list of
    ``(geometry, value)`` pairs into a regularly gridded raster.
    No GDAL dependency.

    Supported geometry types:

    - **Polygon / MultiPolygon** -- scanline fill
    - **LineString / LinearRing / MultiLineString** -- Bresenham line
      rasterization (a LinearRing burns its closed boundary)
    - **Point / MultiPoint** -- direct pixel burn
    - **GeometryCollection** -- recursively unpacked

    A non-empty geometry of any other type is dropped with a warning rather
    than silently discarded.  Likewise, a geometry containing non-finite
    coordinates (NaN or infinity) has no defined location on the raster
    grid and is dropped with a warning.

    Parameters
    ----------
    geometries : GeoDataFrame or iterable of (geometry, value)
        Input vector data.  If a GeoDataFrame, the ``column`` parameter
        selects which attribute to burn into the raster (defaults to the
        first numeric column).  If an iterable, each element must be a
        ``(shapely.geometry, numeric_value)`` pair.
    width : int, optional
        Number of columns in the output raster.  Required unless
        ``resolution`` or ``like`` is given.  Must be passed together
        with ``height``: a partial override (only one of the two) is
        rejected with ``ValueError`` rather than silently filling the
        missing dimension from ``like`` or ``resolution``.
    height : int, optional
        Number of rows in the output raster.  Required unless
        ``resolution`` or ``like`` is given.  Must be passed together
        with ``width`` (see above).
    bounds : tuple of (xmin, ymin, xmax, ymax), optional
        Geographic extent of the output raster.  Inferred from the
        geometries (or ``like``) if omitted.
    column : str, optional
        Name of the GeoDataFrame column whose values are burned into
        the raster.  Ignored when ``geometries`` is a list of pairs.
        Mutually exclusive with ``columns``.

        A string or categorical column is label-encoded: each distinct
        label gets an integer code ``0..N-1`` and the result is an
        ``int32`` band with a ``-1`` nodata sentinel (unless ``dtype`` /
        ``fill`` are passed explicitly).  Plain string/object columns are
        ordered lexically; an existing pandas ``Categorical`` keeps its
        declared order.  The label map is stored on the result as
        ``attrs['category_names']`` (index == pixel code) plus an
        auto-generated ``attrs['category_colors']`` (one RGBA per class).
        ``to_geotiff`` writes these to a PAM ``<file>.aux.xml`` sidecar
        so GDAL/QGIS display the class names, and ``open_geotiff`` reads
        them back.
    columns : list of str, optional
        Names of multiple GeoDataFrame columns to pass as a properties
        array to the merge function.  Mutually exclusive with ``column``.
        When given, the merge function receives a 1D float64 array of
        length ``len(columns)`` as its ``props`` argument.
    fill : float, default np.nan
        Value for pixels not covered by any geometry.  When ``dtype``
        resolves to an integer or boolean type (either via the ``dtype``
        argument or via ``like``), a fill the dtype cannot represent
        exactly is rejected with ``ValueError``.  This covers NaN into an
        integer dtype (which would land on a platform sentinel), an
        out-of-range integer (``fill=-9999`` into ``uint8`` wraps to
        241), and any non-False value into ``bool`` (which collapses to
        ``True``).  Without the guard the burned array and its emitted
        ``nodata`` / ``_FillValue`` attrs would disagree.  Pass a fill the
        dtype can hold (e.g. ``fill=0`` or ``fill=-9999`` for integers,
        ``fill=False`` for bool) or use a floating dtype.
    dtype : numpy dtype, dtype name, or scalar type, optional
        Data type of the output array.  Accepts anything ``np.dtype()``
        can parse: an ``np.dtype`` instance, a dtype name string
        (``'int32'``), or a numpy scalar type (``np.int32``).  Defaults
        to np.float64, or to the dtype of ``like`` if provided.  When
        this resolves to an integer type, burn values are validated
        against the float64 safe
        integer range: the rasterizer computes in float64, so a value
        with magnitude above ``2**53 - 1`` cannot be cast back to an
        exact integer (e.g. ``2**53 + 1`` would land on ``2**53``).  Such
        a value is rejected with ``ValueError`` rather than silently
        rounded; use a floating dtype to burn identifiers larger than
        that.  Non-finite burn values (NaN, inf) are likewise rejected
        for integer and bool dtypes -- the final cast would otherwise
        land them on a platform sentinel (NaN becomes ``-2147483648``
        for int32, ``True`` for bool).  ``merge='count'`` is exempt
        because it burns overlap counts, never the property values.
    all_touched : bool, default False
        If True, every pixel a geometry passes through is burned, using
        a supercover (Amanatides & Woo) grid traversal: polygon
        boundaries in addition to the normal center-fill, and every
        cell a line crosses rather than only the Bresenham path. Output
        matches ``rasterio.features.rasterize(..., all_touched=True)``
        pixel-for-pixel up to rasterization tie-breaking on shared
        edges and exact cell-corner crossings. If False, only pixels
        whose centers fall inside a polygon (or on the Bresenham line)
        are burned.
    gpu : bool, default False
        If True, use the CuPy/CUDA backend.  Same convention as
        ``open_geotiff(gpu=True)``; combine with ``chunks`` for the
        dask+cupy backend.
    name : str, default 'rasterize'
        Name for the output DataArray.
    resolution : float or (x_res, y_res), optional
        Pixel size.  Must be finite and ``> 0``.  When given with
        ``bounds``, computes ``width`` and ``height`` automatically.
        A single float uses the same resolution for both axes.
        The requested cell size is honored exactly.  When the bounds
        don't divide evenly by the resolution, the output grid extends
        past the original ``xmax`` / ``ymin`` (anchored at ``xmin`` /
        ``ymax``) rather than shrinking the cell to fit.  Matches the
        ``rasterio.transform.from_origin`` convention.
    like : xr.DataArray, optional
        Template raster.  Width, height, bounds, and dtype are copied
        from this array.  Bounds and dtype can be overridden one at a
        time; width and height must be overridden together (passing
        only one raises ``ValueError``).
        Must have uniformly spaced ``x`` and ``y`` dim coords -- the
        rasterizer only writes to a regular grid, so a non-uniform
        ``like`` is rejected with ``ValueError`` rather than silently
        producing pixel labels that don't match the data.  Both
        descending (top-down, ymax first) and ascending (bottom-up,
        ymin first) y coords are supported -- the burned rows are
        flipped to match so ``result.sel(y=...)`` lines up with the
        geometry in world coordinates either way, and ``result.y``
        always equals ``like.y`` exactly.
    merge : str or callable, default 'last'
        How to combine values when geometries overlap.

        Built-in modes (pass as string):

        - ``'last'`` -- last geometry in input order wins
        - ``'first'`` -- first geometry wins
        - ``'max'`` / ``'min'`` -- keep the larger / smaller value
        - ``'sum'`` -- add values together
        - ``'count'`` -- count overlapping geometries

        For ``'sum'`` and ``'count'``, each geometry contributes at
        most once per pixel: the connecting vertex of consecutive line
        segments and the overlap between a polygon's interior fill and
        its ``all_touched`` boundary are deduplicated, matching
        rasterio's ``MergeAlg.add`` (which yields 1 per geometry per
        pixel even for a self-crossing line).  Individual points are
        not deduplicated -- a MultiPoint with repeated coordinates
        burns once per point, also matching rasterio.

        Custom merge function (pass a callable):

        For CPU backends, pass a ``@ngjit``-decorated function.  For GPU
        backends (``gpu=True``), pass a
        ``@numba.cuda.jit(device=True)`` function.  Signature::

            merge_fn(pixel, props, is_first) -> float64

        - *pixel*: current pixel value (fill value on first write)
        - *props*: 1D float64 array of property values for the geometry
        - *is_first*: 1 on first write to this pixel, 0 otherwise

        .. warning::

           On the GPU backends (``gpu=True``, with or without
           ``chunks``) a custom callable does not use CUDA atomics.  Its
           per-pixel update is a non-atomic read-modify-write, so when
           geometries overlap, several threads may update the same pixel
           at once and the result for those pixels is nondeterministic --
           it can vary between runs and need not match the CPU backend.
           The built-in string merges (``'sum'``, ``'count'``, ``'min'``,
           ``'max'``, ``'first'``, ``'last'``) do use atomics and stay
           deterministic over overlap; pass one of those if you need a
           stable result where geometries overlap on the GPU.  Calling
           ``rasterize`` with a callable ``merge`` and ``gpu=True``
           emits a ``UserWarning`` to this effect.

    chunks : int or (int, int), optional
        If given, use the dask backend and split the output raster into
        tiles of this size ``(row_chunk, col_chunk)``.  Both axes must be
        ``> 0``.  A single int uses the same chunk size for both axes.
        Combined with ``gpu`` to select dask+numpy vs dask+cupy.
    max_pixels : int, default 1_000_000_000
        Safety cap on the resolved output size (``width * height``).  The
        function raises ``ValueError`` before any host or device
        allocation if the cap is exceeded.  Raise this explicitly when
        rasterizing a legitimately large grid.
    check_crs : bool, default True
        When ``geometries`` is a GeoDataFrame with a ``.crs`` and ``like``
        is a template that carries a CRS (via ``attrs['crs']``,
        ``attrs['crs_wkt']``, a ``spatial_ref`` coord, or ``rio.crs``),
        compare the two and raise ``ValueError`` if they disagree.  This
        prevents silently burning, say, an EPSG:4326 GeoDataFrame onto an
        EPSG:3857 template and handing back a raster mislabeled with the
        template CRS.  The geometries are never reprojected; reproject
        them yourself to match the template.  Pass ``check_crs=False`` to
        skip the comparison (the output still inherits the template CRS).
        The check is a no-op when either side lacks a CRS.
    use_cuda : bool, optional
        Deprecated alias for ``gpu``; emits a ``DeprecationWarning``
        when passed.  Passing both ``gpu=True`` and ``use_cuda`` raises
        ``TypeError``.

    Returns
    -------
    xr.DataArray
        2D raster with dims ``('y', 'x')``.  When ``geometries`` is a
        GeoDataFrame with a ``.crs`` and neither ``like`` nor its
        ``spatial_ref`` coord supplies a CRS, the geometry CRS is
        emitted on the output (``attrs['crs']`` as an EPSG int when one
        resolves, else ``attrs['crs_wkt']`` as WKT).

    Examples
    --------
    .. sourcecode:: python

        >>> from shapely.geometry import box
        >>> result = rasterize([(box(0, 0, 5, 5), 1.0)],
        ...                    width=10, height=10)

        >>> # Using resolution instead of width/height:
        >>> result = rasterize(gdf, resolution=0.5,
        ...                    bounds=(0, 0, 10, 10), column='value')

        >>> # Match an existing raster grid:
        >>> zones = rasterize(gdf, like=elevation, column='zone')

        >>> # Sum overlapping values:
        >>> density = rasterize(gdf, width=100, height=100,
        ...                     column='pop', merge='sum', fill=0)
    """
    # Deprecation shim: ``use_cuda`` was renamed to ``gpu`` so the GPU
    # opt-in matches ``open_geotiff(gpu=True)`` (issue #3089).  The old
    # keyword still works but warns; asking for both is ambiguous.
    if use_cuda is not None:
        if gpu:
            raise TypeError(
                "rasterize() got both 'gpu' and its deprecated alias "
                "'use_cuda'; pass only 'gpu'")
        warnings.warn(
            "rasterize(use_cuda=...) is deprecated; use gpu=... instead "
            "(same convention as open_geotiff).",
            DeprecationWarning,
            stacklevel=2,
        )
        gpu = use_cuda

    # Fail early with a clear message if the optional ``vector`` extra
    # (shapely) is not installed, rather than deep inside a helper.
    _require_shapely()

    if column is not None and columns is not None:
        raise ValueError(
            "'column' and 'columns' are mutually exclusive; use one or "
            "the other")
    # Resolve merge: string -> (merge_fn, should_write_fn), or pass callable
    # through.  User-supplied callables are treated as order-insensitive (the
    # public signature is (pixel, props, is_first) -> float64, with no slot
    # for input order); they keep the pre-2064 behaviour where the last
    # call wins per-pixel.  Built-in ``'first'`` / ``'last'`` are
    # order-sensitive: the predicate gates writes by global input index so
    # the bug-2064 example (Point before Polygon, ``merge='last'``) returns
    # the polygon's value rather than letting the points-burned-last phase
    # win.
    if callable(merge):
        merge_fn = merge
        should_write_cpu = _should_write_any
        _merge_fn_gpu = merge  # same object for GPU path
    elif isinstance(merge, str):
        if merge not in _MERGE_FUNCTIONS:
            raise ValueError(
                f"merge must be one of {set(_MERGE_FUNCTIONS)} or a "
                f"callable, got {merge!r}")
        merge_fn, should_write_cpu = _MERGE_FUNCTIONS[merge]
        _merge_fn_gpu = merge  # resolved lazily in GPU path by name
    else:
        raise TypeError(
            f"merge must be a string or callable, got {type(merge).__name__}")

    # Reject partial width/height before any geometry or template work.
    # Passing only one of the two has no well-defined meaning here.
    # When ``like`` is given, the bounds also come from the template, so
    # deriving the missing dimension from aspect ratio would make the x
    # and y pixel resolutions diverge and the output coords would no
    # longer match ``like``.  When ``like`` is not given, there's
    # nothing to derive from at all.  Either way, the old code silently
    # fell through to the ``resolution`` or ``like`` branch and
    # discarded the explicit dimension without warning.
    if (width is None) != (height is None):
        missing = 'height' if width is not None else 'width'
        given = 'width' if width is not None else 'height'
        raise ValueError(
            f"{given} was provided but {missing} was not. Pass both "
            f"width and height together, or omit both and supply "
            f"resolution or like to size the output.")

    # Extract defaults from template raster
    like_width = like_height = like_bounds = like_dtype = None
    like_x_coord = like_y_coord = None
    like_extra_coords = {}
    like_attrs = None
    like_y_ascending = False
    like_x_descending = False
    bounds_explicit = bounds is not None
    if like is not None:
        grid = _extract_grid_from_like(like)
        like_width = grid.width
        like_height = grid.height
        like_bounds = grid.bounds
        like_dtype = grid.dtype
        like_x_coord = grid.x_coord
        like_y_coord = grid.y_coord
        like_extra_coords = grid.extra_coords
        like_attrs = grid.attrs
        like_y_ascending = grid.y_ascending
        like_x_descending = grid.x_descending

    # Parse input geometries
    geom_list, props_array, inferred_bounds, geom_crs, category_names = \
        _parse_input(geometries, column=column, columns=columns)

    # Categorical column: burn integer codes onto an integer band with a
    # -1 nodata sentinel (pandas' own missing code) so the result renders
    # as discrete classes in QGIS rather than a float ramp.  Explicit
    # ``dtype`` / ``fill`` always win.  ``like`` only supplies the grid
    # here, not the dtype -- a float template must not silently demote the
    # codes back to a float ramp.
    category_colors = None
    if category_names is not None:
        if dtype is None:
            dtype = np.int32
        try:
            fill_is_nan = np.isnan(float(fill))
        except (TypeError, ValueError):
            fill_is_nan = False
        if fill_is_nan:
            fill = -1
        # Map the pandas missing code (-1) to the resolved fill so
        # geometries with no category become nodata.
        if fill != -1:
            props_array = np.where(props_array == -1, fill, props_array)
        category_colors = _categorical_colors(len(category_names))

    # Guard against silently burning geometries onto a template in a
    # different CRS.  The output inherits the template CRS (attrs /
    # spatial_ref coord) but the geometry coords are never reprojected,
    # so a mismatch yields an authoritative-looking but wrong raster.
    if check_crs and geom_crs is not None and like is not None:
        _check_crs_match(geom_crs, _like_crs(like))

    # Resolve bounds: explicit > like > inferred from geometries
    final_bounds = bounds
    if final_bounds is None and like_bounds is not None:
        final_bounds = like_bounds
    if final_bounds is None:
        final_bounds = inferred_bounds
    if final_bounds is None and geom_list:
        # Compute bounds lazily only when not supplied by the caller.
        # Drop empty/invalid geoms whose bbox is nan so they don't poison
        # the inferred extent.
        geom_bboxes = _geometry_bboxes(geom_list)
        if len(geom_bboxes) > 0:
            finite = np.all(np.isfinite(geom_bboxes), axis=1)
            geom_bboxes = geom_bboxes[finite]
            if len(geom_bboxes) > 0:
                final_bounds = (geom_bboxes[:, 0].min(),
                                geom_bboxes[:, 1].min(),
                                geom_bboxes[:, 2].max(),
                                geom_bboxes[:, 3].max())
    if final_bounds is None:
        raise ValueError(
            "bounds must be provided when geometries are empty or have "
            "no spatial extent")

    xmin, ymin, xmax, ymax = final_bounds
    if not all(np.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
        raise ValueError(
            f"Invalid bounds: all of (xmin, ymin, xmax, ymax) must be "
            f"finite, got {(xmin, ymin, xmax, ymax)!r}")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            f"Invalid bounds: xmin ({xmin}) must be < xmax ({xmax}) and "
            f"ymin ({ymin}) must be < ymax ({ymax})")

    # Resolve width/height: explicit > resolution > like
    if width is not None and height is not None:
        final_width, final_height = int(width), int(height)
    elif resolution is not None:
        # Validate shape and element type up front so bad inputs surface a
        # single clean ValueError naming the offending value, instead of
        # leaking IndexError (length-1 sequences would crash at
        # resolution[1]), KeyError (dicts), or a raw float() conversion
        # error (strings iterate character-by-character into
        # resolution[0]/[1]).  A 3+-element sequence was previously
        # silently truncated to the first two elements -- reject it here
        # too.  numpy scalars (np.float32, np.int64, ...) and 1-D numpy
        # arrays of size 2 are accepted alongside Python int/float and
        # list/tuple, since geospatial pipelines routinely produce them.
        is_scalar = (
            isinstance(resolution, (int, float, np.number))
            and not isinstance(resolution, (bool, np.bool_))
        )
        is_sequence = isinstance(resolution, (tuple, list, np.ndarray))
        if not (is_scalar or is_sequence):
            raise ValueError(
                f"resolution must be a number or a length-2 sequence of "
                f"numbers (x_res, y_res), got {resolution!r}")
        if is_scalar:
            x_res = y_res = float(resolution)
        else:
            # numpy arrays expose .ndim; require 1-D for the sequence form
            # so e.g. a (2, 2) array does not slip past the length-2 check.
            if isinstance(resolution, np.ndarray) and resolution.ndim != 1:
                raise ValueError(
                    f"resolution array must be 1-D with length 2 "
                    f"(x_res, y_res), got shape {resolution.shape}: "
                    f"{resolution!r}")
            if len(resolution) != 2:
                raise ValueError(
                    f"resolution sequence must have length 2 (x_res, y_res), "
                    f"got length {len(resolution)}: {resolution!r}")
            try:
                x_res = float(resolution[0])
                y_res = float(resolution[1])
            except (TypeError, ValueError):
                raise ValueError(
                    f"resolution sequence elements must be numbers, "
                    f"got {resolution!r}")
        # Reject non-finite or non-positive resolution before dimension math.
        # Without this, inf/-1 quietly produce a 1x1 raster, 0 raises an
        # opaque ZeroDivisionError, and nan raises an int-conversion error.
        for r in (x_res, y_res):
            if not np.isfinite(r) or r <= 0:
                raise ValueError(
                    f"resolution must be finite and > 0, got {resolution!r}")
        final_width = max(int(np.ceil((xmax - xmin) / x_res)), 1)
        final_height = max(int(np.ceil((ymax - ymin) / y_res)), 1)
        # Honor the requested resolution exactly: rebuild final_bounds so
        # cell size is ``x_res`` / ``y_res`` instead of letting the
        # rasterizer compute ``(xmax-xmin)/width`` and silently shrink the
        # pixel to fit non-divisible bounds (issue #2573).  Anchor the
        # upper-left corner (xmin, ymax) and extend xmax / ymin outward,
        # matching rasterio's ``from_origin(west, north, x_res, y_res)``
        # convention.  When bounds already divide evenly, the new values
        # equal the originals to within float precision so behaviour is
        # unchanged.
        xmax = xmin + final_width * x_res
        ymin = ymax - final_height * y_res
        final_bounds = (xmin, ymin, xmax, ymax)
    elif like_width is not None:
        final_width, final_height = like_width, like_height
    else:
        raise ValueError(
            "Must specify width/height, resolution, or like")

    if final_width < 1 or final_height < 1:
        raise ValueError(
            f"width and height must be >= 1, got width={final_width}, "
            f"height={final_height}")

    # Reject oversize outputs before any host or device allocation.  Covers
    # the explicit width/height path and the resolution-derived path.
    _check_output_dimensions(final_width, final_height, max_pixels)

    # Resolve dtype: explicit > like > default
    if dtype is not None:
        final_dtype = dtype
    elif like_dtype is not None:
        final_dtype = like_dtype
    else:
        final_dtype = np.float64

    # Reject a fill that the output dtype cannot represent exactly.  Every
    # backend casts its float64 work buffer with ``astype(final_dtype)`` at
    # the end (``_run_numpy`` etc.), and the attrs block below stores the
    # original ``fill`` verbatim.  When the cast changes the value, the
    # array and its ``nodata`` / ``_FillValue`` / ``nodatavals`` attrs
    # disagree, so downstream consumers (geotiff writer, rioxarray masks)
    # mask the wrong pixels or treat unwritten cells as valid data -- a
    # metadata-propagation failure surfaced by the metadata sweep
    # (issues #2504, #3054).  Failure modes guarded here:
    #   * NaN fill into an integer dtype -> coerced to 0 / INT_MIN.
    #   * out-of-range integer fill (fill=-9999 into uint8 -> 241).
    #   * any non-False fill into bool (fill=NaN -> True everywhere).
    # ``np.issubdtype(bool_, np.integer)`` is False, so bool needs the
    # explicit round-trip test below rather than the old integer-only
    # check.
    #
    # Checked before any host / device allocation so the error surfaces
    # cleanly regardless of backend (numpy, cupy, dask+numpy, dask+cupy).
    # It runs after ``_check_output_dimensions`` because the
    # width/height/resolution guard reports a more actionable diagnostic
    # for oversized grids; both checks land before the allocator either
    # way.  Float dtypes are left alone: NaN is representable and ordinary
    # float rounding is expected, not a metadata lie.
    final_dtype_np = np.dtype(final_dtype)
    if (np.issubdtype(final_dtype_np, np.integer)
            or final_dtype_np == np.bool_):
        try:
            fill_arr = np.array(fill)
            with warnings.catch_warnings():
                # NaN -> int casts warn; we are probing for exactly that.
                warnings.simplefilter('ignore', RuntimeWarning)
                cast_back = fill_arr.astype(
                    final_dtype_np).astype(fill_arr.dtype)
            representable = bool(np.array_equal(fill_arr, cast_back))
        except (TypeError, ValueError, OverflowError):
            # A fill too large for the dtype's C type (e.g. an int wider
            # than the platform long) overflows the cast rather than
            # round-tripping; that is exactly a value the dtype cannot
            # hold, so treat it as non-representable.
            representable = False
        if not representable:
            raise ValueError(
                f"fill={fill!r} cannot be represented exactly in output "
                f"dtype {final_dtype_np}: the final cast would silently "
                f"change it (e.g. NaN -> 0/INT_MIN, an out-of-range int -> "
                f"a wrapped value, any non-False value -> True for bool), "
                f"leaving the burned array and its nodata/_FillValue attrs "
                f"in disagreement. Pass a fill the dtype can hold (e.g. "
                f"fill=0 or fill=-9999 for integers, fill=False for bool) "
                f"or use a floating dtype.")

    # Reject burn values that float64 cannot hold exactly when the output
    # dtype is integer.  ``_parse_input`` casts every burn value to float64
    # (props_array is float64, GeoDataFrame columns go through
    # ``.astype(np.float64)``, and ``(geom, value)`` pairs go through
    # ``float()``), and the whole rasterize pipeline computes in float64
    # before the final ``astype(int_dtype)``.  Integers with magnitude above
    # 2**53 - 1 (the IEEE-754 "max safe integer") are not exactly
    # representable, so an ID like ``2**53 + 1`` silently lands on
    # ``2**53`` -- off by one -- by the time it reaches the output.  Zone
    # IDs, parcel IDs, and uint64 identifiers are exactly the values that
    # exceed this range, so a silent round is a correctness failure rather
    # than a rounding nicety.  Reject up front with the offending value
    # named, mirroring the NaN-fill guard above.  Float dtypes are
    # unaffected: the float64 value is what the user asked to store.
    #
    # Checked before any host / device allocation so the error surfaces
    # cleanly across all four backends (numpy, cupy, dask+numpy, dask+cupy).
    if (np.issubdtype(final_dtype_np, np.integer)
            and props_array.size > 0):
        max_safe_int = 2.0 ** 53 - 1
        finite = np.isfinite(props_array)
        unsafe = finite & (np.abs(props_array) > max_safe_int)
        if unsafe.any():
            bad_value = float(props_array[unsafe].flat[0])
            raise ValueError(
                f"burn value {bad_value!r} exceeds the range of integers "
                f"that float64 can represent exactly (|value| > 2**53 - 1 "
                f"= {int(max_safe_int)}). rasterize() computes in float64, "
                f"so casting to integer dtype {final_dtype_np} would "
                f"silently round it (e.g. 2**53 + 1 lands on 2**53). Use a "
                f"floating output dtype, or pass values within the safe "
                f"integer range.")

    # Reject non-finite burn values when the output dtype cannot represent
    # them.  The pipeline computes in float64 and casts with
    # ``astype(final_dtype)`` at the end, so a NaN or +/-inf burn value
    # (e.g. a GeoDataFrame column with missing data) against an integer
    # dtype lands on a platform sentinel (NaN -> -2147483648 for int32) and
    # against bool collapses to True -- silently on the numpy backend,
    # whose final cast suppresses the RuntimeWarning.  This mirrors the
    # NaN-fill guard (#2504) and the unsafe-integer guard (#3056) above;
    # the non-finite burn value was the remaining gap (#3085).
    # ``merge='count'`` never reads property values (the burned value is
    # the overlap count), so NaN attributes stay usable there.  Float
    # dtypes are unaffected: NaN/inf are representable.
    if ((np.issubdtype(final_dtype_np, np.integer)
            or final_dtype_np == np.bool_)
            and props_array.size > 0
            and merge != 'count'):
        nonfinite_props = ~np.isfinite(props_array)
        if nonfinite_props.any():
            bad_value = float(props_array[nonfinite_props].flat[0])
            raise ValueError(
                f"burn value {bad_value!r} is not finite and cannot be "
                f"represented in output dtype {final_dtype_np}: the final "
                f"cast would silently turn it into a platform sentinel "
                f"(e.g. NaN -> -2147483648 for int32, NaN -> True for "
                f"bool). Remove or fill the non-finite values, or use a "
                f"floating output dtype.")

    # For GPU paths, resolve string merge names to GPU (merge_fn,
    # should_write_fn) pairs.  User callables are paired with the
    # always-write predicate (matches the public 3-arg signature).
    # ``gpu_merge_name`` selects the atomic kernel path in
    # ``_ensure_gpu_kernels``; ``None`` falls back to the non-atomic
    # closure path used for user callables.
    gpu_merge_name = None
    if gpu:
        if cupy is None:
            raise ImportError(
                "CuPy is required for gpu=True but is not installed")
        gpu_fns = _get_gpu_merge_fns()
        if isinstance(_merge_fn_gpu, str):
            gpu_merge_fn, should_write_gpu = gpu_fns[_merge_fn_gpu]
            gpu_merge_name = _merge_fn_gpu
        else:
            gpu_merge_fn = _merge_fn_gpu
            # Need an always-true predicate compiled for cuda; reuse any
            # _should_write_any_gpu from the cache.  The cached dict
            # always includes a sum/count entry that uses it.
            _, should_write_gpu = gpu_fns['sum']
            # A callable keeps gpu_merge_name=None, i.e. the non-atomic
            # read-modify-write path in _apply_merge_gpu.  Overlapping
            # geometries then race and the overlap pixels are
            # nondeterministic.  Warn here (after the CuPy check so a
            # missing-dependency caller only sees the ImportError); this
            # covers both the cupy and dask+cupy paths.  A built-in string
            # merge stays deterministic over overlap.
            warnings.warn(
                "A custom callable merge on the GPU backend "
                "(gpu=True) uses a non-atomic read-modify-write, so "
                "values for pixels where geometries overlap are "
                "nondeterministic and may not match the CPU backend. Use a "
                "built-in string merge ('sum', 'count', 'min', 'max', "
                "'first', 'last') if you need a deterministic result over "
                "overlap.",
                UserWarning,
                stacklevel=2,
            )

    if chunks is not None:
        row_chunks, col_chunks = _normalize_chunks(
            chunks, final_height, final_width)
        if gpu:
            out = _run_dask_cupy(
                geom_list, props_array, final_bounds,
                final_height, final_width, fill, final_dtype,
                all_touched, gpu_merge_fn, should_write_gpu,
                row_chunks, col_chunks, gpu_merge_name)
        else:
            out = _run_dask_numpy(
                geom_list, props_array, final_bounds,
                final_height, final_width, fill, final_dtype,
                all_touched, merge_fn, should_write_cpu,
                row_chunks, col_chunks)
    elif gpu:
        out = _run_cupy(geom_list, props_array, final_bounds,
                        final_height, final_width, fill, final_dtype,
                        all_touched, gpu_merge_fn, should_write_gpu,
                        gpu_merge_name)
    else:
        out = _run_numpy(geom_list, props_array, final_bounds,
                         final_height, final_width, fill, final_dtype,
                         all_touched, merge_fn, should_write_cpu)

    # Build coordinates.  When the caller didn't override the grid (no
    # explicit width/height/bounds/resolution that resizes the output),
    # reuse like.coords directly so the output is bit-identical to the
    # template and xr.align keeps working.  Float-equal bound comparison
    # would be fragile, so key the reuse on size + "bounds weren't
    # overridden" instead.
    reuse_like_coords = (
        like_x_coord is not None
        and like_x_coord.sizes['x'] == final_width
        and like_y_coord.sizes['y'] == final_height
        and not bounds_explicit
        and resolution is None
    )
    if reuse_like_coords:
        x_coords = like_x_coord
        y_coords = like_y_coord
        # The rasterizer always burns with row 0 = ymax and column 0 =
        # xmin (top-down image convention).  If the template's y axis is
        # ascending or its x axis is descending, the burned array has to
        # be flipped along that axis before assigning the template's
        # coords, so world-coord selection still lines up with the
        # geometry.  Works for numpy, cupy, dask+numpy, and dask+cupy
        # alike -- they all expose the same slicing semantics.
        if like_y_ascending:
            out = out[::-1, :]
        if like_x_descending:
            out = out[:, ::-1]
    else:
        px = (xmax - xmin) / final_width
        py = (ymax - ymin) / final_height
        x_coords = np.linspace(xmin + px / 2, xmax - px / 2, final_width)
        y_coords = np.linspace(ymax - py / 2, ymin + py / 2, final_height)

    # Build attrs.  Start from like.attrs if given so chained spatial
    # pipelines (slope(rasterize(gdf, like=elevation))) see the same res,
    # crs, transform, etc. as the template.  Strip any inherited nodata
    # keys first -- the template's old fill value almost certainly
    # disagrees with this call's fill, and the geotiff writer's
    # _resolve_nodata_attr would otherwise tag pixels with a stale
    # sentinel.  Then re-emit a consistent triplet keyed off the actual
    # fill: nodata (xrspatial's primary key), _FillValue (CF), and
    # nodatavals (rioxarray's per-band tuple).
    out_attrs = like_attrs if like_attrs is not None else {}
    for k in ('nodata', '_FillValue', 'nodatavals'):
        out_attrs.pop(k, None)
    # Grid-shape attrs (res, transform) describe the template's grid.
    # When the caller overrides bounds/width/height/resolution so the
    # output grid no longer matches ``like``, leaving the inherited
    # values in place lies to downstream consumers: get_dataarray_
    # resolution() prefers ``attrs['res']`` over computing from coords,
    # so a stale res silently poisons slope/aspect/proximity callers
    # with the template's cellsize instead of the actual one.  Drop them
    # when the grid was reshaped; keep them when the output reuses the
    # template's coords bit-identically.
    if like_attrs is not None and not reuse_like_coords:
        for k in ('res', 'transform'):
            out_attrs.pop(k, None)
    try:
        fill_as_float = float(fill)
        fill_is_nan = np.isnan(fill_as_float)
    except (TypeError, ValueError):
        fill_is_nan = False
    if not fill_is_nan:
        out_attrs['nodata'] = fill
        out_attrs['_FillValue'] = fill
        out_attrs['nodatavals'] = (fill,)

    # Carry the label map so to_geotiff can emit a PAM sidecar and QGIS
    # shows class names.  Index == pixel code; colors are one RGBA per
    # category.  These are pass-through attrs (xrspatial/geotiff/_attrs.py).
    if category_names is not None:
        out_attrs['category_names'] = category_names
        out_attrs['category_colors'] = category_colors

    # Emit the geometry CRS when the output would otherwise carry none.
    # The grid is laid out in the geometry's coordinate system (bounds
    # come from the geometry coords), so a CRS-carrying GeoDataFrame
    # rasterized without ``like`` should produce a raster labeled with
    # that CRS -- otherwise to_geotiff writes an un-georeferenced file
    # and polygonize(rasterize(gdf)) returns crs=None (issue #3087).
    # A template CRS always wins: if ``like`` supplied attrs['crs'],
    # attrs['crs_wkt'], or a spatial_ref coord, those are left alone
    # (the check_crs guard has already compared them to geom_crs).
    # Follow the geotiff attrs convention (xrspatial/geotiff/_attrs.py):
    # an EPSG int under 'crs' when one resolves, else WKT under
    # 'crs_wkt'.  geom_crs is only non-None on the GeoDataFrame path,
    # where pyproj is available (geopandas requires it).
    if (geom_crs is not None
            and 'crs' not in out_attrs
            and 'crs_wkt' not in out_attrs
            and 'spatial_ref' not in like_extra_coords):
        from pyproj import CRS as _PyprojCRS
        crs_obj = _PyprojCRS(geom_crs)
        epsg = crs_obj.to_epsg()
        if epsg is not None:
            out_attrs['crs'] = epsg
        else:
            out_attrs['crs_wkt'] = crs_obj.to_wkt()

    # Combine y/x dim coords with any non-dim coords carried from the
    # template (e.g. rioxarray's spatial_ref CRS coord).
    out_coords = {'y': y_coords, 'x': x_coords}
    for k, v in like_extra_coords.items():
        if k not in out_coords:
            out_coords[k] = v

    return xr.DataArray(
        out,
        name=name,
        dims=['y', 'x'],
        coords=out_coords,
        attrs=out_attrs,
    )
