"""Vector geometry rasterization (polygons, lines, points).

Converts vector geometries (GeoDataFrame or list of (geometry, value) pairs)
to a 2D xr.DataArray.  No GDAL dependency.

- Polygons/MultiPolygons: scanline fill
- Lines/MultiLineStrings: Bresenham line rasterization
- Points/MultiPoints: direct pixel burn

Supports numpy, cupy, dask+numpy, and dask+cupy backends.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

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

# Detect shapely 2.0+ for vectorized extraction
try:
    import shapely as _shapely_mod
    _HAS_SHAPELY2 = hasattr(_shapely_mod, 'get_parts')
except ImportError:
    _HAS_SHAPELY2 = False


# ---------------------------------------------------------------------------
# Merge functions (CPU, numba-jitted)
#
# Signature: merge_fn(pixel, props, is_first) -> float64
#   pixel    : current pixel value (fill value on first write)
#   props    : 1D float64 array of property values for the geometry
#   is_first : 1 if first write to this pixel, 0 otherwise
# ---------------------------------------------------------------------------

@ngjit
def _merge_last(pixel, props, is_first):
    return props[0]


@ngjit
def _merge_first(pixel, props, is_first):
    if is_first:
        return props[0]
    return pixel


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


_MERGE_FUNCTIONS = {
    'last': _merge_last, 'first': _merge_first,
    'max': _merge_max, 'min': _merge_min,
    'sum': _merge_sum, 'count': _merge_count,
}


# ---------------------------------------------------------------------------
# Merge pixel helper (CPU)
# ---------------------------------------------------------------------------

@ngjit
def _apply_merge(out, written, r, c, props, merge_fn):
    """Write a value into ``out[r, c]`` using the given merge function.

    *props* is a 1D float64 array of property values for the geometry.
    A separate ``written`` array (int8) tracks which pixels have been
    touched.
    """
    is_first = np.int64(written[r, c] == 0)
    out[r, c] = merge_fn(out[r, c], props, is_first)
    written[r, c] = 1


# ---------------------------------------------------------------------------
# Geometry classification (single pass)
# ---------------------------------------------------------------------------

def _classify_geometries(geometries, props_array):
    """Classify geometries by type in a single pass.

    Also tracks each polygon's input index so the scanline fill can
    process geometries in input order (needed for first/last merge).

    GeometryCollections are recursively unpacked so their contents are
    rasterized rather than silently dropped.

    Parameters
    ----------
    geometries : list of shapely geometries
    props_array : (N, P) float64 array of property values

    Returns
    -------
    (poly_geoms, poly_props, poly_ids),
    (line_geoms, line_props),
    (point_geoms, point_props)

    Where poly_props is (N_poly, P), line_props is (N_line, P),
    point_props is (N_point, P) float64 arrays.
    """
    n_props = props_array.shape[1] if props_array.ndim == 2 else 1
    poly_geoms, poly_prop_rows, poly_ids = [], [], []
    line_geoms, line_prop_rows = [], []
    point_geoms, point_prop_rows = [], []

    # poly_counter tracks per-type indices that both preserve input order
    # and serve as valid indices into the poly_props table.
    poly_counter = [0]

    def _classify_one(geom, prop_row, global_idx):
        if geom is None or geom.is_empty:
            return
        gt = geom.geom_type
        if gt in ('Polygon', 'MultiPolygon'):
            poly_geoms.append(geom)
            poly_prop_rows.append(prop_row)
            poly_ids.append(poly_counter[0])
            poly_counter[0] += 1
        elif gt in ('LineString', 'MultiLineString'):
            line_geoms.append(geom)
            line_prop_rows.append(prop_row)
        elif gt in ('Point', 'MultiPoint'):
            point_geoms.append(geom)
            point_prop_rows.append(prop_row)
        elif gt == 'GeometryCollection':
            for sub in geom.geoms:
                _classify_one(sub, prop_row, global_idx)

    for idx, geom in enumerate(geometries):
        _classify_one(geom, props_array[idx], idx)

    def _to_2d(rows):
        if rows:
            return np.array(rows, dtype=np.float64)
        return np.empty((0, n_props), dtype=np.float64)

    return ((poly_geoms, _to_2d(poly_prop_rows), poly_ids),
            (line_geoms, _to_2d(line_prop_rows)),
            (point_geoms, _to_2d(point_prop_rows)))


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
    if _HAS_SHAPELY2:
        return _extract_edges_vectorized(
            geometries, geom_ids, bounds, height, width, all_touched)
    return _extract_edges_loop(
        geometries, geom_ids, bounds, height, width, all_touched)


def _extract_edges_vectorized(geometries, geom_ids, bounds,
                              height, width, all_touched):
    """Vectorized edge extraction using shapely 2.0 array ops."""
    import shapely

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

    # Convert to pixel space (reuse arrays through the pipeline to
    # cut peak memory from ~12 temporaries to ~6).
    sr = (ymax - coords[start_idx, 1]) / py
    sc = (coords[start_idx, 0] - xmin) / px
    er = (ymax - coords[end_idx, 1]) / py
    ec = (coords[end_idx, 0] - xmin) / px

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


def _extract_edges_loop(geometries, geom_ids, bounds, height, width,
                        all_touched):
    """Loop-based edge extraction (shapely < 2.0 fallback).

    Pre-allocates output arrays sized to the total vertex count (an upper
    bound on edge count) to avoid per-edge Python list appends and
    np.int32() scalar wrapping overhead.
    """
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    # Upper bound: each ring vertex pair can produce at most one edge.
    # Sum of (len(ring.coords) - 1) across all rings.
    est = 0
    ring_data = []  # (coords_array, gid) pairs
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
            rings = [poly.exterior] + list(poly.interiors)
            for ring in rings:
                coords = np.asarray(ring.coords)
                n = len(coords) - 1
                if n > 0:
                    ring_data.append((coords, gid))
                    est += n

    if est == 0:
        return _EMPTY_EDGES

    # Pre-allocate arrays
    buf_ymin = np.empty(est, dtype=np.int32)
    buf_ymax = np.empty(est, dtype=np.int32)
    buf_xmin = np.empty(est, dtype=np.float64)
    buf_inv = np.empty(est, dtype=np.float64)
    buf_gid = np.empty(est, dtype=np.int32)
    pos = 0

    for coords, gid in ring_data:
        row = (ymax - coords[:, 1]) / py
        col = (coords[:, 0] - xmin) / px
        n = len(row) - 1
        for i in range(n):
            r0, c0 = row[i], col[i]
            r1, c1 = row[i + 1], col[i + 1]
            if r0 == r1:
                continue
            if r0 > r1:
                r0, c0, r1, c1 = r1, c1, r0, c0
            if all_touched:
                ry_min = max(int(np.floor(r0 - 0.5)), 0)
                ry_max = min(int(np.ceil(r1 + 0.5)) - 1, height - 1)
            else:
                ry_min = max(int(np.ceil(r0)), 0)
                ry_max = min(int(np.ceil(r1)) - 1, height - 1)
            if ry_min > ry_max:
                continue
            inv_slope = (c1 - c0) / (r1 - r0)
            buf_ymin[pos] = ry_min
            buf_ymax[pos] = ry_max
            buf_xmin[pos] = c0 + (ry_min - r0) * inv_slope
            buf_inv[pos] = inv_slope
            buf_gid[pos] = gid
            pos += 1

    if pos == 0:
        return _EMPTY_EDGES

    return (buf_ymin[:pos], buf_ymax[:pos], buf_xmin[:pos],
            buf_inv[:pos], buf_gid[:pos])


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
    if _HAS_SHAPELY2:
        return _extract_points_vectorized(
            geometries, bounds, height, width)
    return _extract_points_loop(
        geometries, bounds, height, width)


def _extract_points_vectorized(geometries, bounds, height, width):
    """Vectorized point extraction using shapely 2.0 array ops."""
    import shapely

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


def _extract_points_loop(geometries, bounds, height, width):
    """Loop-based point extraction (shapely < 2.0 fallback)."""
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    all_rows, all_cols, all_idx = [], [], []

    for gi, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'Point':
            pts = [geom]
        elif geom.geom_type == 'MultiPoint':
            pts = list(geom.geoms)
        else:
            continue
        for pt in pts:
            col = int(np.floor((pt.x - xmin) / px))
            row = int(np.floor((ymax - pt.y) / py))
            if 0 <= row < height and 0 <= col < width:
                all_rows.append(row)
                all_cols.append(col)
                all_idx.append(gi)

    if not all_rows:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.int32))
    return (np.array(all_rows, np.int32),
            np.array(all_cols, np.int32),
            np.array(all_idx, np.int32))


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
    if _HAS_SHAPELY2:
        return _extract_lines_vectorized(
            geometries, bounds, height, width)
    return _extract_lines_loop(
        geometries, bounds, height, width)


def _liang_barsky_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """Liang-Barsky line clipping.  Returns clipped (x0,y0,x1,y1) or None."""
    dx = x1 - x0
    dy = y1 - y0
    p = np.array([-dx, dx, -dy, dy])
    q = np.array([x0 - xmin, xmax - x0, y0 - ymin, ymax - y0])

    t0, t1 = 0.0, 1.0
    for i in range(4):
        if p[i] == 0.0:
            if q[i] < 0.0:
                return None
        elif p[i] < 0.0:
            t = q[i] / p[i]
            if t > t1:
                return None
            if t > t0:
                t0 = t
        else:
            t = q[i] / p[i]
            if t < t0:
                return None
            if t < t1:
                t1 = t

    cx0 = x0 + t0 * dx
    cy0 = y0 + t0 * dy
    cx1 = x0 + t1 * dx
    cy1 = y0 + t1 * dy
    return cx0, cy0, cx1, cy1


def _extract_lines_vectorized(geometries, bounds, height, width):
    """Vectorized line extraction with Liang-Barsky clipping."""
    import shapely

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


def _extract_lines_loop(geometries, bounds, height, width):
    """Loop-based line extraction with Liang-Barsky clipping (fallback)."""
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    all_r0, all_c0, all_r1, all_c1, all_idx = [], [], [], [], []

    for gi, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'LineString':
            lines = [geom]
        elif geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        else:
            continue
        for line in lines:
            coords = np.asarray(line.coords)
            for i in range(len(coords) - 1):
                clipped = _liang_barsky_clip(
                    coords[i, 0], coords[i, 1],
                    coords[i + 1, 0], coords[i + 1, 1],
                    xmin, ymin, xmax, ymax)
                if clipped is None:
                    continue
                cx0, cy0, cx1, cy1 = clipped
                r0 = min(max(int(np.floor((ymax - cy0) / py)), 0), height - 1)
                c0 = min(max(int(np.floor((cx0 - xmin) / px)), 0), width - 1)
                r1 = min(max(int(np.floor((ymax - cy1) / py)), 0), height - 1)
                c1 = min(max(int(np.floor((cx1 - xmin) / px)), 0), width - 1)
                all_r0.append(r0)
                all_c0.append(c0)
                all_r1.append(r1)
                all_c1.append(c1)
                all_idx.append(gi)

    if not all_r0:
        return _EMPTY_LINES
    return (np.array(all_r0, np.int32), np.array(all_c0, np.int32),
            np.array(all_r1, np.int32), np.array(all_c1, np.int32),
            np.array(all_idx, np.int32))


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
                     merge_fn):
    for i in range(len(rows)):
        r = rows[i]
        c = cols[i]
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            _apply_merge(out, written, r, c, geom_props[geom_idx[i]],
                         merge_fn)


@ngjit
def _burn_lines_cpu(out, written, r0_arr, c0_arr, r1_arr, c1_arr, geom_idx,
                    geom_props, height, width, merge_fn):
    for i in range(len(r0_arr)):
        r0 = r0_arr[i]
        c0 = c0_arr[i]
        r1 = r1_arr[i]
        c1 = c1_arr[i]
        props = geom_props[geom_idx[i]]

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
                    _apply_merge(out, written, r, c, props, merge_fn)
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
                    _apply_merge(out, written, r, c, props, merge_fn)
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
                       geom_props, height, width, merge_fn):
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
                    _apply_merge(out, written, row, c,
                                 geom_props[gid], merge_fn)
                k += 2
            i = j


def _run_numpy(geometries, props_array, bounds, height, width, fill, dtype,
               all_touched, merge_fn):
    """NumPy backend for rasterize."""
    out = np.full((height, width), fill, dtype=np.float64)
    written = np.zeros((height, width), dtype=np.int8)

    (poly_geoms, poly_props, poly_ids), (line_geoms, line_props), \
        (point_geoms, point_props) = _classify_geometries(
            geometries, props_array)

    # 1. Polygons -- always use normal edge ranges for scanline fill
    #    (all_touched y-expansion breaks edge pairing, so we handle
    #    all_touched by drawing polygon boundaries separately below).
    edge_arrays = _extract_edges(
        poly_geoms, poly_ids, bounds, height, width)
    edge_arrays = _sort_edges(edge_arrays)
    if len(edge_arrays[0]) > 0:
        _scanline_fill_cpu(out, written, *edge_arrays,
                           poly_props, height, width, merge_fn)

    # 1b. all_touched: draw polygon boundaries via Bresenham so every
    #     pixel the boundary passes through is burned.  This guarantees
    #     all_touched is a superset of the normal fill.
    if all_touched and poly_geoms:
        br0, bc0, br1, bc1, bidx = _extract_polygon_boundary_segments(
            poly_geoms, poly_ids, bounds, height, width)
        if len(br0) > 0:
            _burn_lines_cpu(out, written, br0, bc0, br1, bc1, bidx,
                            poly_props, height, width, merge_fn)

    # 2. Lines
    r0, c0, r1, c1, line_idx = _extract_line_segments(
        line_geoms, bounds, height, width)
    if len(r0) > 0:
        _burn_lines_cpu(out, written, r0, c0, r1, c1, line_idx,
                        line_props, height, width, merge_fn)

    # 3. Points
    prows, pcols, pt_idx = _extract_points(
        point_geoms, bounds, height, width)
    if len(prows) > 0:
        _burn_points_cpu(out, written, prows, pcols, pt_idx,
                         point_props, merge_fn)

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
    """Lazily define and cache built-in GPU merge device functions."""
    global _gpu_merge_fns
    if _gpu_merge_fns is not None:
        return _gpu_merge_fns

    from numba import cuda

    @cuda.jit(device=True)
    def _merge_last_gpu(pixel, props, is_first):
        return props[0]

    @cuda.jit(device=True)
    def _merge_first_gpu(pixel, props, is_first):
        if is_first:
            return props[0]
        return pixel

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

    _gpu_merge_fns = {
        'last': _merge_last_gpu, 'first': _merge_first_gpu,
        'max': _merge_max_gpu, 'min': _merge_min_gpu,
        'sum': _merge_sum_gpu, 'count': _merge_count_gpu,
    }
    return _gpu_merge_fns


# ---------------------------------------------------------------------------
# GPU kernels -- compiled per merge function and cached.
# ---------------------------------------------------------------------------

_gpu_kernel_cache = {}


def _ensure_gpu_kernels(merge_fn):
    """Compile CUDA kernels for the given merge device function and cache.

    Each unique merge function produces a separate set of kernels because
    CUDA kernels cannot accept function arguments -- the merge function
    is captured by closure at compile time.
    """
    key = id(merge_fn)
    if key in _gpu_kernel_cache:
        return _gpu_kernel_cache[key]

    from numba import cuda

    @cuda.jit(device=True)
    def _apply_merge_gpu(out, written, r, c, props):
        is_first = np.int64(written[r, c] == 0)
        out[r, c] = merge_fn(out[r, c], props, is_first)
        written[r, c] = 1

    @cuda.jit
    def _scanline_fill_gpu(out, written, edge_y_min, edge_x_at_ymin,
                           edge_inv_slope, edge_geom_id,
                           geom_props, row_ptr, col_idx, width):
        """CUDA kernel: one thread per raster row, CSR-indexed active edges."""
        row = cuda.grid(1)
        if row >= out.shape[0]:
            return

        start = row_ptr[row]
        end = row_ptr[row + 1]
        count = end - start

        if count == 0:
            return

        MAX_ISECT = 512
        if count > MAX_ISECT:
            count = MAX_ISECT

        xs = cuda.local.array(512, dtype=np.float64)
        gs = cuda.local.array(512, dtype=np.int32)

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
                    _apply_merge_gpu(out, written, row, c,
                                     geom_props[gid])
                k += 2
            i = j

    @cuda.jit
    def _burn_points_gpu(out, written, rows, cols, geom_idx, geom_props,
                         n_points):
        i = cuda.grid(1)
        if i >= n_points:
            return
        r = rows[i]
        c = cols[i]
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            _apply_merge_gpu(out, written, r, c,
                             geom_props[geom_idx[i]])

    @cuda.jit
    def _burn_lines_gpu(out, written, r0_arr, c0_arr, r1_arr, c1_arr,
                        geom_idx, geom_props, n_segs, height, width):
        i = cuda.grid(1)
        if i >= n_segs:
            return
        r0 = r0_arr[i]
        c0 = c0_arr[i]
        r1 = r1_arr[i]
        c1 = c1_arr[i]
        props = geom_props[geom_idx[i]]

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
                    _apply_merge_gpu(out, written, r, c, props)
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
                    _apply_merge_gpu(out, written, r, c, props)
                if err >= 0:
                    r += sr
                    err -= dc
                c += sc
                err += dr

    kernels = {
        'scanline_fill': _scanline_fill_gpu,
        'burn_points': _burn_points_gpu,
        'burn_lines': _burn_lines_gpu,
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
        return (np.zeros(height + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32))

    # Pass 1: difference-array counting — O(n_edges + height)
    # diff[r] += 1 when an edge starts, diff[r+1] -= 1 when it ends.
    diff = np.zeros(height + 1, dtype=np.int32)
    for e in range(n_edges):
        y_lo = edge_y_min[e]
        y_hi = edge_y_max[e]
        if y_hi >= height:
            y_hi = height - 1
        if y_lo <= y_hi:
            diff[y_lo] += 1
            diff[y_hi + 1] -= 1

    # Prefix sum on diff -> counts, and build row_ptr simultaneously
    row_ptr = np.empty(height + 1, dtype=np.int32)
    row_ptr[0] = 0
    running = np.int32(0)
    for r in range(height):
        running += diff[r]
        row_ptr[r + 1] = row_ptr[r] + running

    # Pass 2: fill col_idx — each edge placed into its row range
    total = row_ptr[height]
    col_idx = np.empty(total, dtype=np.int32)
    offsets = np.empty(height, dtype=np.int32)
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


def _run_cupy(geometries, props_array, bounds, height, width, fill, dtype,
              all_touched, merge_fn):
    """CuPy backend for rasterize."""
    kernels = _ensure_gpu_kernels(merge_fn)

    out = cupy.full((height, width), fill, dtype=cupy.float64)
    written = cupy.zeros((height, width), dtype=cupy.int8)

    (poly_geoms, poly_props, poly_ids), (line_geoms, line_props), \
        (point_geoms, point_props) = _classify_geometries(
            geometries, props_array)

    # 1. Polygons -- always use normal edge ranges (see _run_numpy comment)
    edge_arrays = _extract_edges(
        poly_geoms, poly_ids, bounds, height, width)
    edge_arrays = _sort_edges(edge_arrays)
    edge_y_min, edge_y_max, edge_x_at_ymin, edge_inv_slope, \
        edge_geom_id = edge_arrays

    if len(edge_y_min) > 0:
        # Build CSR structure on CPU, then transfer to GPU
        row_ptr, col_idx = _build_row_csr_numba(
            edge_y_min, edge_y_max, height)

        d_y_min = cupy.asarray(edge_y_min)
        d_x_at_ymin = cupy.asarray(edge_x_at_ymin)
        d_inv_slope = cupy.asarray(edge_inv_slope)
        d_geom_id = cupy.asarray(edge_geom_id)
        d_geom_props = cupy.asarray(poly_props)
        d_row_ptr = cupy.asarray(row_ptr)
        d_col_idx = cupy.asarray(col_idx)

        tpb = 256
        blocks = (height + tpb - 1) // tpb
        kernels['scanline_fill'][blocks, tpb](
            out, written, d_y_min, d_x_at_ymin, d_inv_slope,
            d_geom_id, d_geom_props, d_row_ptr, d_col_idx, width)

    # 1b. all_touched: draw polygon boundaries via Bresenham (GPU)
    if all_touched and poly_geoms:
        br0, bc0, br1, bc1, bidx = _extract_polygon_boundary_segments(
            poly_geoms, poly_ids, bounds, height, width)
        if len(br0) > 0:
            n_bsegs = len(br0)
            tpb = 256
            bpg = (n_bsegs + tpb - 1) // tpb
            kernels['burn_lines'][bpg, tpb](
                out, written, cupy.asarray(br0), cupy.asarray(bc0),
                cupy.asarray(br1), cupy.asarray(bc1),
                cupy.asarray(bidx), cupy.asarray(poly_props),
                n_bsegs, height, width)

    # 2. Lines
    r0, c0, r1, c1, line_idx = _extract_line_segments(
        line_geoms, bounds, height, width)
    if len(r0) > 0:
        n_segs = len(r0)
        tpb = 256
        bpg = (n_segs + tpb - 1) // tpb
        kernels['burn_lines'][bpg, tpb](
            out, written, cupy.asarray(r0), cupy.asarray(c0),
            cupy.asarray(r1), cupy.asarray(c1),
            cupy.asarray(line_idx), cupy.asarray(line_props),
            n_segs, height, width)

    # 3. Points
    prows, pcols, pt_idx = _extract_points(
        point_geoms, bounds, height, width)
    if len(prows) > 0:
        n_pts = len(prows)
        tpb = 256
        bpg = (n_pts + tpb - 1) // tpb
        kernels['burn_points'][bpg, tpb](
            out, written, cupy.asarray(prows), cupy.asarray(pcols),
            cupy.asarray(pt_idx), cupy.asarray(point_props), n_pts)

    return out.astype(dtype)


# ---------------------------------------------------------------------------
# Dask tile-based rasterization
# ---------------------------------------------------------------------------

def _geometry_bboxes(geometries):
    """Return (N, 4) float64 array of [xmin, ymin, xmax, ymax] per geometry."""
    if len(geometries) == 0:
        return np.empty((0, 4), dtype=np.float64)
    if _HAS_SHAPELY2:
        import shapely
        return shapely.bounds(np.asarray(geometries, dtype=object))
    return np.array([g.bounds for g in geometries], dtype=np.float64)


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


def _polys_to_wkb(geoms):
    """Pre-serialize polygon geometries to WKB for cheap pickling."""
    return [g.wkb for g in geoms]


def _polys_from_wkb(wkb_list):
    """Deserialize WKB back to shapely geometries."""
    from shapely import from_wkb
    geoms = from_wkb(wkb_list)
    if not isinstance(geoms, (list, np.ndarray)):
        geoms = [geoms]
    return list(geoms)


def _rasterize_tile_numpy(poly_wkb, poly_props_2d, tile_bounds, tile_h,
                          tile_w, fill, dtype, all_touched, merge_fn,
                          seg_r0, seg_c0, seg_r1, seg_c1, seg_geom_idx,
                          line_props,
                          pt_rows, pt_cols, pt_geom_idx,
                          point_props):
    """Rasterize a single tile.

    Polygons are passed as WKB bytes (cheap to pickle) and deserialized
    inside the worker.  Line segments and points are passed in tile-local
    pixel coordinates with geometry indices into their respective props
    tables.
    """
    out = np.full((tile_h, tile_w), fill, dtype=np.float64)
    written = np.zeros((tile_h, tile_w), dtype=np.int8)

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
                               poly_props_2d, tile_h, tile_w, merge_fn)

        # 1b. all_touched: draw polygon boundaries via Bresenham
        if all_touched:
            br0, bc0, br1, bc1, bidx = _extract_polygon_boundary_segments(
                poly_geoms, poly_ids, tile_bounds, tile_h, tile_w)
            if len(br0) > 0:
                _burn_lines_cpu(out, written, br0, bc0, br1, bc1, bidx,
                                poly_props_2d, tile_h, tile_w, merge_fn)

    # 2. Lines (tile-local segments, Bresenham with bounds check)
    if len(seg_r0) > 0:
        _burn_lines_cpu(out, written, seg_r0, seg_c0, seg_r1, seg_c1,
                        seg_geom_idx, line_props, tile_h, tile_w, merge_fn)

    # 3. Points (tile-local)
    if len(pt_rows) > 0:
        _burn_points_cpu(out, written, pt_rows, pt_cols, pt_geom_idx,
                         point_props, merge_fn)

    return out.astype(dtype)


def _run_dask_numpy(geometries, props_array, bounds, height, width, fill,
                    dtype, all_touched, merge_fn, row_chunks, col_chunks):
    """Dask + NumPy backend: tile-based parallel rasterization."""
    import dask
    import dask.array as da

    # Classify geometries once
    (poly_geoms, poly_props, _poly_ids), (line_geoms, line_props), \
        (point_geoms, point_props) = _classify_geometries(
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
        poly_wkb_arr = np.array([g.wkb for g in poly_geoms], dtype=object)
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
            else:
                tile_wkb = []
                tile_poly_props = poly_props[:0]  # empty (0, P)

            # Filter segments and points by tile pixel range
            ts = _segments_for_tile(seg_r0, seg_c0, seg_r1, seg_c1,
                                    seg_geom_idx, seg_bboxes,
                                    r_start, r_end, c_start, c_end)
            tp = _points_for_tile(pt_rows, pt_cols, pt_geom_idx,
                                  r_start, r_end, c_start, c_end)

            delayed_tile = dask.delayed(_rasterize_tile_numpy)(
                tile_wkb, tile_poly_props, tile_bounds,
                tile_h, tile_w, fill, dtype, all_touched, merge_fn,
                *ts, line_props, *tp, point_props)
            blocks[i][j] = da.from_delayed(
                delayed_tile, shape=(tile_h, tile_w), dtype=dtype)
            ri += 1

    return da.block(blocks)


def _ensure_cupy(arr):
    """Transfer to GPU only if not already a CuPy array."""
    if isinstance(arr, cupy.ndarray):
        return arr
    return cupy.asarray(arr)


def _rasterize_tile_cupy(poly_wkb, poly_props_2d, tile_bounds, tile_h,
                         tile_w, fill, dtype, all_touched, merge_fn,
                         seg_r0, seg_c0, seg_r1, seg_c1, seg_geom_idx,
                         line_props,
                         pt_rows, pt_cols, pt_geom_idx,
                         point_props):
    """GPU tile rasterization: polygons as WKB, lines/points as segments."""
    kernels = _ensure_gpu_kernels(merge_fn)

    out = cupy.full((tile_h, tile_w), fill, dtype=cupy.float64)
    written = cupy.zeros((tile_h, tile_w), dtype=cupy.int8)

    # 1. Polygons (deserialize WKB, then scanline fill on GPU)
    if poly_wkb:
        poly_geoms = _polys_from_wkb(poly_wkb)
        poly_ids = np.arange(len(poly_geoms), dtype=np.int32)
        edge_arrays = _extract_edges(
            poly_geoms, poly_ids, tile_bounds,
            tile_h, tile_w)
        edge_arrays = _sort_edges(edge_arrays)
        edge_y_min = edge_arrays[0]
        if len(edge_y_min) > 0:
            edge_y_max, edge_x_at_ymin, edge_inv_slope, \
                edge_geom_id = edge_arrays[1:]
            row_ptr, col_idx = _build_row_csr_numba(
                edge_y_min, edge_y_max, tile_h)
            tpb = 256
            blocks = (tile_h + tpb - 1) // tpb
            kernels['scanline_fill'][blocks, tpb](
                out, written,
                cupy.asarray(edge_y_min), cupy.asarray(edge_x_at_ymin),
                cupy.asarray(edge_inv_slope), cupy.asarray(edge_geom_id),
                cupy.asarray(poly_props_2d), cupy.asarray(row_ptr),
                cupy.asarray(col_idx), tile_w)

        # 1b. all_touched: draw polygon boundaries via Bresenham (GPU)
        if all_touched:
            br0, bc0, br1, bc1, bidx = _extract_polygon_boundary_segments(
                poly_geoms, poly_ids, tile_bounds, tile_h, tile_w)
            if len(br0) > 0:
                n_bsegs = len(br0)
                tpb_b = 256
                bpg_b = (n_bsegs + tpb_b - 1) // tpb_b
                kernels['burn_lines'][bpg_b, tpb_b](
                    out, written, cupy.asarray(br0), cupy.asarray(bc0),
                    cupy.asarray(br1), cupy.asarray(bc1),
                    cupy.asarray(bidx), cupy.asarray(poly_props_2d),
                    n_bsegs, tile_h, tile_w)

    # 2. Lines (tile-local segments, GPU Bresenham)
    if len(seg_r0) > 0:
        n_segs = len(seg_r0)
        tpb = 256
        bpg = (n_segs + tpb - 1) // tpb
        kernels['burn_lines'][bpg, tpb](
            out, written,
            cupy.asarray(seg_r0), cupy.asarray(seg_c0),
            cupy.asarray(seg_r1), cupy.asarray(seg_c1),
            cupy.asarray(seg_geom_idx), _ensure_cupy(line_props),
            n_segs, tile_h, tile_w)

    # 3. Points (tile-local)
    if len(pt_rows) > 0:
        n_pts = len(pt_rows)
        tpb = 256
        bpg = (n_pts + tpb - 1) // tpb
        kernels['burn_points'][bpg, tpb](
            out, written,
            cupy.asarray(pt_rows), cupy.asarray(pt_cols),
            cupy.asarray(pt_geom_idx), _ensure_cupy(point_props), n_pts)

    return out.astype(dtype)


def _run_dask_cupy(geometries, props_array, bounds, height, width, fill,
                   dtype, all_touched, merge_fn, row_chunks, col_chunks):
    """Dask + CuPy backend: tile-based parallel GPU rasterization."""
    import dask
    import dask.array as da

    # Classify geometries once
    (poly_geoms, poly_props, _poly_ids), (line_geoms, line_props), \
        (point_geoms, point_props) = _classify_geometries(
            geometries, props_array)

    # Compact representations in full-raster pixel space
    seg_r0, seg_c0, seg_r1, seg_c1, seg_geom_idx = _extract_line_segments(
        line_geoms, bounds, height, width)
    pt_rows, pt_cols, pt_geom_idx = _extract_points(
        point_geoms, bounds, height, width)

    # Pre-serialize polygons to WKB (20x cheaper to pickle than shapely).
    if poly_geoms:
        poly_bboxes = _geometry_bboxes(poly_geoms)
        poly_wkb_arr = np.array([g.wkb for g in poly_geoms], dtype=object)
    else:
        poly_bboxes = np.empty((0, 4), dtype=np.float64)
        poly_wkb_arr = np.empty(0, dtype=object)

    # Pre-compute segment bboxes once (avoids redundant min/max per tile)
    seg_bboxes = _segment_bboxes(seg_r0, seg_c0, seg_r1, seg_c1)

    # Transfer shared props tables to GPU once (avoids repeated PCIe
    # transfers in each tile worker).
    d_line_props = cupy.asarray(line_props)
    d_point_props = cupy.asarray(point_props)

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
            else:
                tile_wkb = []
                tile_poly_props = poly_props[:0]  # empty (0, P)

            ts = _segments_for_tile(seg_r0, seg_c0, seg_r1, seg_c1,
                                    seg_geom_idx, seg_bboxes,
                                    r_start, r_end, c_start, c_end)
            tp = _points_for_tile(pt_rows, pt_cols, pt_geom_idx,
                                  r_start, r_end, c_start, c_end)

            delayed_tile = dask.delayed(_rasterize_tile_cupy)(
                tile_wkb, tile_poly_props, tile_bounds,
                tile_h, tile_w, fill, dtype, all_touched, merge_fn,
                *ts, d_line_props, *tp, d_point_props)
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

    # Compute per-geometry bboxes (reused by dask tile filtering later)
    geom_bboxes = _geometry_bboxes(geom_list)
    if len(geom_bboxes) == 0:
        return geom_list, props_array, None
    total_bounds = (geom_bboxes[:, 0].min(), geom_bboxes[:, 1].min(),
                    geom_bboxes[:, 2].max(), geom_bboxes[:, 3].max())
    return geom_list, props_array, total_bounds


def _extract_grid_from_like(like):
    """Extract width, height, bounds, dtype from a template DataArray."""
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

    return width, height, (xmin, ymin, xmax, ymax), dt


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
        Pixel size.  When given with ``bounds``, computes ``width`` and
        ``height`` automatically.  A single float uses the same
        resolution for both axes.
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
        tiles of this size ``(row_chunk, col_chunk)``.  A single int
        uses the same chunk size for both axes.  Combined with
        ``use_cuda`` to select dask+numpy vs dask+cupy.

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
    # Resolve merge: string -> built-in function, or pass callable through.
    if callable(merge):
        merge_fn = merge
        # For GPU, the caller provides a @cuda.jit(device=True) function
        # directly.  For CPU, a @ngjit function.
        _merge_fn_gpu = merge  # same object for GPU path
    elif isinstance(merge, str):
        if merge not in _MERGE_FUNCTIONS:
            raise ValueError(
                f"merge must be one of {set(_MERGE_FUNCTIONS)} or a "
                f"callable, got {merge!r}")
        merge_fn = _MERGE_FUNCTIONS[merge]
        _merge_fn_gpu = merge  # resolved lazily in GPU path by name
    else:
        raise TypeError(
            f"merge must be a string or callable, got {type(merge).__name__}")

    # Extract defaults from template raster
    like_width = like_height = like_bounds = like_dtype = None
    if like is not None:
        like_width, like_height, like_bounds, like_dtype = \
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
    if final_bounds is None:
        raise ValueError(
            "bounds must be provided when geometries are empty or have "
            "no spatial extent")

    xmin, ymin, xmax, ymax = final_bounds
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

    # Resolve dtype: explicit > like > default
    if dtype is not None:
        final_dtype = dtype
    elif like_dtype is not None:
        final_dtype = like_dtype
    else:
        final_dtype = np.float64

    # For GPU paths, resolve string merge names to GPU device functions.
    if use_cuda:
        if cupy is None:
            raise ImportError(
                "CuPy is required for use_cuda=True but is not installed")
        if isinstance(_merge_fn_gpu, str):
            gpu_fns = _get_gpu_merge_fns()
            gpu_merge_fn = gpu_fns[_merge_fn_gpu]
        else:
            gpu_merge_fn = _merge_fn_gpu

    if chunks is not None:
        row_chunks, col_chunks = _normalize_chunks(
            chunks, final_height, final_width)
        if use_cuda:
            out = _run_dask_cupy(
                geom_list, props_array, final_bounds,
                final_height, final_width, fill, final_dtype,
                all_touched, gpu_merge_fn, row_chunks, col_chunks)
        else:
            out = _run_dask_numpy(
                geom_list, props_array, final_bounds,
                final_height, final_width, fill, final_dtype,
                all_touched, merge_fn, row_chunks, col_chunks)
    elif use_cuda:
        out = _run_cupy(geom_list, props_array, final_bounds,
                        final_height, final_width, fill, final_dtype,
                        all_touched, gpu_merge_fn)
    else:
        out = _run_numpy(geom_list, props_array, final_bounds,
                         final_height, final_width, fill, final_dtype,
                         all_touched, merge_fn)

    # Build coordinates
    px = (xmax - xmin) / final_width
    py = (ymax - ymin) / final_height
    x_coords = np.linspace(xmin + px / 2, xmax - px / 2, final_width)
    y_coords = np.linspace(ymax - py / 2, ymin + py / 2, final_height)

    return xr.DataArray(
        out,
        name=name,
        dims=['y', 'x'],
        coords={'y': y_coords, 'x': x_coords},
    )
