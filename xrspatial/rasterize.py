"""Vector geometry rasterization (polygons, lines, points).

Converts vector geometries (GeoDataFrame or list of (geometry, value) pairs)
to a 2D xr.DataArray.  No GDAL dependency.

- Polygons/MultiPolygons: scanline fill
- Lines/MultiLineStrings: Bresenham line rasterization
- Points/MultiPoints: direct pixel burn

Supports numpy, cupy, dask+numpy, and dask+cupy backends.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import shapely
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
    if is_first or props[0] > pixel:
        return props[0]
    return pixel


@ngjit
def _merge_min(pixel, props, is_first):
    if is_first or props[0] < pixel:
        return props[0]
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

    type_ids = shapely.get_type_id(geom_arr)
    empty = shapely.is_empty(geom_arr)
    valid = ~empty

    # Type ID mapping:
    # 0=Point, 1=LineString, 2=LinearRing, 3=Polygon,
    # 4=MultiPoint, 5=MultiLineString, 6=MultiPolygon,
    # 7=GeometryCollection
    poly_mask = valid & ((type_ids == 3) | (type_ids == 6))
    line_mask = valid & ((type_ids == 1) | (type_ids == 5))
    point_mask = valid & ((type_ids == 0) | (type_ids == 4))
    gc_mask = valid & (type_ids == 7)

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

    def _classify_one(geom, prop_row, global_idx):
        nonlocal poly_counter
        if geom is None or geom.is_empty:
            return
        gt = geom.geom_type
        if gt in ('Polygon', 'MultiPolygon'):
            poly_geoms.append(geom)
            poly_prop_rows.append(prop_row)
            poly_ids.append(poly_counter)
            poly_global.append(global_idx)
            poly_counter += 1
        elif gt in ('LineString', 'MultiLineString'):
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

    for idx, geom in enumerate(geometries):
        _classify_one(geom, props_array[idx], idx)

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

    if all_touched:
        ry_min = np.maximum(np.floor(top_r - 0.5).astype(np.int32), 0)
        ry_max = np.minimum(
            np.ceil(bot_r + 0.5).astype(np.int32) - 1, height - 1)
    else:
        ry_min = np.maximum(np.ceil(top_r).astype(np.int32), 0)
        ry_max = np.minimum(
            np.ceil(bot_r).astype(np.int32) - 1, height - 1)
    del bot_r

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


def _run_numpy(geometries, props_array, bounds, height, width, fill, dtype,
               all_touched, merge_fn, should_write):
    """NumPy backend for rasterize."""
    out = np.full((height, width), fill, dtype=np.float64)
    written = np.zeros((height, width), dtype=np.int8)
    # Order tracks the input index of the current owner per pixel.  -1
    # sentinel is fine: written[r,c]==0 means cur_idx is never consulted
    # for the "first write" branch of the predicates.
    order = np.full((height, width), -1, dtype=np.int64)

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

    # 1b. all_touched: draw polygon boundaries via Bresenham so every
    #     pixel the boundary passes through is burned.  This guarantees
    #     all_touched is a superset of the normal fill.
    if all_touched and poly_geoms:
        br0, bc0, br1, bc1, bidx = _extract_polygon_boundary_segments(
            poly_geoms, poly_ids, bounds, height, width)
        if len(br0) > 0:
            _burn_lines_cpu(out, written, br0, bc0, br1, bc1, bidx,
                            poly_props, height, width, poly_global,
                            merge_fn, should_write, order)

    # 2. Lines
    r0, c0, r1, c1, line_idx = _extract_line_segments(
        line_geoms, bounds, height, width)
    if len(r0) > 0:
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

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return out.astype(dtype)


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
            cuda.atomic.min(out, (r, c), props[0])
            cuda.atomic.max(order, (r, c), new_idx)
        elif atomic_mode == 'max':
            cuda.atomic.max(out, (r, c), props[0])
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

    kernels = {
        'scanline_fill': _scanline_fill_gpu,
        'burn_points': _burn_points_gpu,
        'burn_lines': _burn_lines_gpu,
        'scanline_fill_pass2': _scanline_fill_gpu_pass2,
        'burn_points_pass2': _burn_points_gpu_pass2,
        'burn_lines_pass2': _burn_lines_gpu_pass2,
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

    The atomic kernels never consult ``written`` (the legacy
    user-callable path is the only consumer), so atomic modes get a
    placeholder ``(1, 1)`` buffer to satisfy the kernel signature
    without spending ``H*W`` bytes on dead storage.
    """
    if merge_name in _GPU_ATOMIC_MERGES:
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


def _gpu_finalize_buffers(out, order, order_sentinel, fill, merge_name):
    """Apply the post-pass blend that restores ``fill`` for untouched
    pixels (sum/count/min/max/first/last) and replaces the +/- inf
    initialisers used by min/max.

    Returns ``out`` (modified in-place where convenient).
    """
    if merge_name is None or order_sentinel is None:
        return out
    touched = order != order_sentinel
    if merge_name in ('sum', 'count'):
        # Pixels with no contribution should show ``fill``, not 0.
        out = cupy.where(touched, out, cupy.float64(fill))
    elif merge_name == 'min':
        # +inf initialiser leaks into untouched pixels.
        out = cupy.where(touched, out, cupy.float64(fill))
    elif merge_name == 'max':
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
            cupy.asarray(poly_props), cupy.asarray(poly_global),
            cupy.asarray(row_ptr), cupy.asarray(col_idx))

    # all_touched boundaries.  ``poly_geoms`` and ``poly_ids`` come
    # from ``_classify_geometries`` above and are always defined here,
    # so this block does not depend on the polygon scanline path
    # having allocated ``poly_launch``.
    boundary_launch = None
    if all_touched and poly_geoms:
        br0, bc0, br1, bc1, bidx = _extract_polygon_boundary_segments(
            poly_geoms, poly_ids, bounds, height, width)
        if len(br0) > 0:
            boundary_launch = (
                cupy.asarray(br0), cupy.asarray(bc0),
                cupy.asarray(br1), cupy.asarray(bc1),
                cupy.asarray(bidx), cupy.asarray(poly_props),
                cupy.asarray(poly_global), len(br0))

    r0, c0, r1, c1, line_idx = _extract_line_segments(
        line_geoms, bounds, height, width)
    line_launch = None
    if len(r0) > 0:
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

    def _launch_phase(k_scan, k_lines, k_points):
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
            k_lines[bpg, tpb](out, written, order,
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
        if point_launch is not None:
            n_pts = point_launch[5]
            bpg = (n_pts + tpb - 1) // tpb
            k_points[bpg, tpb](out, written, order,
                               point_launch[0], point_launch[1],
                               point_launch[2], point_launch[3],
                               point_launch[4], n_pts)

    # Pass 1: write values (atomics) or, for first/last, resolve the
    # per-pixel winning input index into ``order``.
    _launch_phase(kernels['scanline_fill'],
                  kernels['burn_lines'],
                  kernels['burn_points'])

    # Pass 2 (first / last only): stamp the winner's value into ``out``.
    if two_pass:
        _launch_phase(kernels['scanline_fill_pass2'],
                      kernels['burn_lines_pass2'],
                      kernels['burn_points_pass2'])

    final_out = _gpu_finalize_buffers(out, order, order_sentinel, fill,
                                      atomic_mode)
    return final_out.astype(dtype)


# ---------------------------------------------------------------------------
# Dask tile-based rasterization
# ---------------------------------------------------------------------------

def _geometry_bboxes(geometries):
    """Return (N, 4) float64 array of [xmin, ymin, xmax, ymax] per geometry."""
    if len(geometries) == 0:
        return np.empty((0, 4), dtype=np.float64)
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
    return shapely.to_wkb(np.asarray(geoms, dtype=object)).tolist()


def _polys_from_wkb(wkb_list):
    """Deserialize WKB back to shapely geometries."""
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
                          point_props, point_global):
    """Rasterize a single tile.

    Polygons are passed as WKB bytes (cheap to pickle) and deserialized
    inside the worker.  Line segments and points are passed in tile-local
    pixel coordinates with geometry indices into their respective props
    tables.  Each per-type global-input-index array carries the original
    input position so order-sensitive merges work across types.
    """
    out = np.full((tile_h, tile_w), fill, dtype=np.float64)
    written = np.zeros((tile_h, tile_w), dtype=np.int8)
    order = np.full((tile_h, tile_w), -1, dtype=np.int64)

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

        # 1b. all_touched: draw polygon boundaries via Bresenham
        if all_touched:
            br0, bc0, br1, bc1, bidx = _extract_polygon_boundary_segments(
                poly_geoms, poly_ids, tile_bounds, tile_h, tile_w)
            if len(br0) > 0:
                _burn_lines_cpu(out, written, br0, bc0, br1, bc1, bidx,
                                poly_props_2d, tile_h, tile_w,
                                poly_global_2d, merge_fn, should_write,
                                order)

    # 2. Lines (tile-local segments, Bresenham with bounds check)
    if len(seg_r0) > 0:
        _burn_lines_cpu(out, written, seg_r0, seg_c0, seg_r1, seg_c1,
                        seg_geom_idx, line_props, tile_h, tile_w,
                        line_global, merge_fn, should_write, order)

    # 3. Points (tile-local)
    if len(pt_rows) > 0:
        _burn_points_cpu(out, written, pt_rows, pt_cols, pt_geom_idx,
                         point_props, point_global,
                         merge_fn, should_write, order)

    return out.astype(dtype)


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
                *tp, tile_point_props, tile_point_global)
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
                         point_props, point_global, merge_name=None):
    """GPU tile rasterization: polygons as WKB, lines/points as segments.

    When ``merge_name`` selects an atomic built-in merge, the per-tile
    kernels use CUDA atomics so overlapping writes inside the tile are
    deterministic.  Cross-tile pixels do not overlap (the tiling is a
    partition of the output grid), so per-tile atomicity is sufficient.
    """
    kernels = _ensure_gpu_kernels(merge_fn, should_write, merge_name)
    atomic_mode = kernels['atomic_mode']
    two_pass = atomic_mode in ('first', 'last')

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
        if len(edge_y_min) > 0:
            edge_y_max, edge_x_at_ymin, edge_inv_slope, edge_geom_id = \
                edge_arrays[1:]
            row_ptr, col_idx = _build_row_csr_numba(
                edge_y_min, edge_y_max, tile_h)
            _check_gpu_edge_cap(row_ptr)
            poly_launch = (
                cupy.asarray(edge_y_min), cupy.asarray(edge_x_at_ymin),
                cupy.asarray(edge_inv_slope), cupy.asarray(edge_geom_id),
                cupy.asarray(poly_props_2d), cupy.asarray(poly_global_2d),
                cupy.asarray(row_ptr), cupy.asarray(col_idx))

        if all_touched:
            br0, bc0, br1, bc1, bidx = _extract_polygon_boundary_segments(
                poly_geoms, poly_ids, tile_bounds, tile_h, tile_w)
            if len(br0) > 0:
                boundary_launch = (
                    cupy.asarray(br0), cupy.asarray(bc0),
                    cupy.asarray(br1), cupy.asarray(bc1),
                    cupy.asarray(bidx), cupy.asarray(poly_props_2d),
                    cupy.asarray(poly_global_2d), len(br0))

    line_launch = None
    if len(seg_r0) > 0:
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

    def _launch_phase(k_scan, k_lines, k_points):
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
            k_lines[bpg, tpb](out, written, order,
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
        if point_launch is not None:
            n_pts = point_launch[5]
            bpg = (n_pts + tpb - 1) // tpb
            k_points[bpg, tpb](out, written, order,
                               point_launch[0], point_launch[1],
                               point_launch[2], point_launch[3],
                               point_launch[4], n_pts)

    _launch_phase(kernels['scanline_fill'],
                  kernels['burn_lines'],
                  kernels['burn_points'])
    if two_pass:
        _launch_phase(kernels['scanline_fill_pass2'],
                      kernels['burn_lines_pass2'],
                      kernels['burn_points_pass2'])

    final_out = _gpu_finalize_buffers(out, order, order_sentinel, fill,
                                      atomic_mode)
    return final_out.astype(dtype)


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
                merge_name)
            blocks[i][j] = da.from_delayed(
                delayed_tile, shape=(tile_h, tile_w), dtype=dtype,
                meta=cupy.empty(()))
            ri += 1

    return da.block(blocks)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _parse_input(geometries, column=None, columns=None):
    """Normalise input to (geometry_list, props_array, bounds).

    Returns
    -------
    geom_list : list of shapely geometries
    props_array : (N, P) float64 array of property values
    bounds : tuple or None
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
                props_array = geometries[column].values.astype(
                    np.float64).reshape(-1, 1)
            return geom_list, props_array, total_bounds
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
        return geom_list, props_array, None

    props_array = np.array(value_list, dtype=np.float64).reshape(-1, 1)

    # Bounds computation is deferred: return None here and let the
    # caller compute bboxes only when bounds are actually needed.
    return geom_list, props_array, None


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

    xmin = float(np.min(x)) - px / 2
    xmax = float(np.max(x)) + px / 2
    ymin = float(np.min(y)) - py / 2
    ymax = float(np.max(y)) + py / 2

    # Carry through any non-dim coords (e.g. rioxarray's ``spatial_ref``
    # CRS coord).  The y/x dim coords are returned separately because the
    # caller decides whether to reuse them (bit-identical grid) or build
    # fresh ones (resized grid).
    extra_coords = {
        k: v for k, v in like.coords.items() if k not in ('x', 'y')
    }

    return (width, height, (xmin, ymin, xmax, ymax), dt,
            like.coords['x'], like.coords['y'],
            extra_coords, dict(like.attrs))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rasterize(
    geometries,
    width: Optional[int] = None,
    height: Optional[int] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    column: Optional[str] = None,
    columns=None,
    fill: float = np.nan,
    dtype: Optional[np.dtype] = None,
    all_touched: bool = False,
    use_cuda: bool = False,
    name: str = 'rasterize',
    resolution: Optional[Union[float, Tuple[float, float]]] = None,
    like: Optional[xr.DataArray] = None,
    merge='last',
    chunks: Optional[Union[int, Tuple[int, int]]] = None,
    max_pixels: int = MAX_PIXELS_DEFAULT,
) -> xr.DataArray:
    """Rasterize vector geometries into a 2D DataArray.

    Converts geometries from a GeoDataFrame or a list of
    ``(geometry, value)`` pairs into a regularly gridded raster.
    No GDAL dependency.

    Supported geometry types:

    - **Polygon / MultiPolygon** -- scanline fill
    - **LineString / MultiLineString** -- Bresenham line rasterization
    - **Point / MultiPoint** -- direct pixel burn
    - **GeometryCollection** -- recursively unpacked

    Parameters
    ----------
    geometries : GeoDataFrame or iterable of (geometry, value)
        Input vector data.  If a GeoDataFrame, the ``column`` parameter
        selects which attribute to burn into the raster (defaults to the
        first numeric column).  If an iterable, each element must be a
        ``(shapely.geometry, numeric_value)`` pair.
    width : int, optional
        Number of columns in the output raster.  Required unless
        ``resolution`` or ``like`` is given.
    height : int, optional
        Number of rows in the output raster.  Required unless
        ``resolution`` or ``like`` is given.
    bounds : tuple of (xmin, ymin, xmax, ymax), optional
        Geographic extent of the output raster.  Inferred from the
        geometries (or ``like``) if omitted.
    column : str, optional
        Name of the GeoDataFrame column whose values are burned into
        the raster.  Ignored when ``geometries`` is a list of pairs.
        Mutually exclusive with ``columns``.
    columns : list of str, optional
        Names of multiple GeoDataFrame columns to pass as a properties
        array to the merge function.  Mutually exclusive with ``column``.
        When given, the merge function receives a 1D float64 array of
        length ``len(columns)`` as its ``props`` argument.
    fill : float, default np.nan
        Value for pixels not covered by any geometry.
    dtype : numpy dtype, optional
        Data type of the output array.  Defaults to np.float64, or
        to the dtype of ``like`` if provided.
    all_touched : bool, default False
        If True, all pixels touched by a geometry are burned, not just
        those whose center falls inside a polygon.
    use_cuda : bool, default False
        If True, use the CuPy/CUDA backend.
    name : str, default 'rasterize'
        Name for the output DataArray.
    resolution : float or (x_res, y_res), optional
        Pixel size.  Must be finite and ``> 0``.  When given with
        ``bounds``, computes ``width`` and ``height`` automatically.
        A single float uses the same resolution for both axes.
    like : xr.DataArray, optional
        Template raster.  Width, height, bounds, and dtype are copied
        from this array (any can still be overridden explicitly).
    merge : str or callable, default 'last'
        How to combine values when geometries overlap.

        Built-in modes (pass as string):

        - ``'last'`` -- last geometry in input order wins
        - ``'first'`` -- first geometry wins
        - ``'max'`` / ``'min'`` -- keep the larger / smaller value
        - ``'sum'`` -- add values together
        - ``'count'`` -- count overlapping geometries

        Custom merge function (pass a callable):

        For CPU backends, pass a ``@ngjit``-decorated function.  For GPU
        backends (``use_cuda=True``), pass a
        ``@numba.cuda.jit(device=True)`` function.  Signature::

            merge_fn(pixel, props, is_first) -> float64

        - *pixel*: current pixel value (fill value on first write)
        - *props*: 1D float64 array of property values for the geometry
        - *is_first*: 1 on first write to this pixel, 0 otherwise

    chunks : int or (int, int), optional
        If given, use the dask backend and split the output raster into
        tiles of this size ``(row_chunk, col_chunk)``.  Both axes must be
        ``> 0``.  A single int uses the same chunk size for both axes.
        Combined with ``use_cuda`` to select dask+numpy vs dask+cupy.
    max_pixels : int, default 1_000_000_000
        Safety cap on the resolved output size (``width * height``).  The
        function raises ``ValueError`` before any host or device
        allocation if the cap is exceeded.  Raise this explicitly when
        rasterizing a legitimately large grid.

    Returns
    -------
    xr.DataArray
        2D raster with dims ``('y', 'x')``.

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

    # Extract defaults from template raster
    like_width = like_height = like_bounds = like_dtype = None
    like_x_coord = like_y_coord = None
    like_extra_coords = {}
    like_attrs = None
    bounds_explicit = bounds is not None
    if like is not None:
        (like_width, like_height, like_bounds, like_dtype,
         like_x_coord, like_y_coord, like_extra_coords, like_attrs) = \
            _extract_grid_from_like(like)

    # Parse input geometries
    geom_list, props_array, inferred_bounds = _parse_input(
        geometries, column=column, columns=columns)

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
        if isinstance(resolution, (int, float)):
            x_res = y_res = float(resolution)
        else:
            x_res, y_res = float(resolution[0]), float(resolution[1])
        # Reject non-finite or non-positive resolution before dimension math.
        # Without this, inf/-1 quietly produce a 1x1 raster, 0 raises an
        # opaque ZeroDivisionError, and nan raises an int-conversion error.
        for r in (x_res, y_res):
            if not np.isfinite(r) or r <= 0:
                raise ValueError(
                    f"resolution must be finite and > 0, got {resolution!r}")
        final_width = max(int(np.ceil((xmax - xmin) / x_res)), 1)
        final_height = max(int(np.ceil((ymax - ymin) / y_res)), 1)
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

    # For GPU paths, resolve string merge names to GPU (merge_fn,
    # should_write_fn) pairs.  User callables are paired with the
    # always-write predicate (matches the public 3-arg signature).
    # ``gpu_merge_name`` selects the atomic kernel path in
    # ``_ensure_gpu_kernels``; ``None`` falls back to the non-atomic
    # closure path used for user callables.
    gpu_merge_name = None
    if use_cuda:
        if cupy is None:
            raise ImportError(
                "CuPy is required for use_cuda=True but is not installed")
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

    if chunks is not None:
        row_chunks, col_chunks = _normalize_chunks(
            chunks, final_height, final_width)
        if use_cuda:
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
    elif use_cuda:
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
    try:
        fill_as_float = float(fill)
        fill_is_nan = np.isnan(fill_as_float)
    except (TypeError, ValueError):
        fill_is_nan = False
    if not fill_is_nan:
        out_attrs['nodata'] = fill
        out_attrs['_FillValue'] = fill
        out_attrs['nodatavals'] = (fill,)

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
