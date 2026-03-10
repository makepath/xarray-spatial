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
# Merge mode constants
# ---------------------------------------------------------------------------

_MERGE_LAST = 0
_MERGE_FIRST = 1
_MERGE_MAX = 2
_MERGE_MIN = 3
_MERGE_SUM = 4
_MERGE_COUNT = 5
_MERGE_MODES = {
    'last': _MERGE_LAST, 'first': _MERGE_FIRST,
    'max': _MERGE_MAX, 'min': _MERGE_MIN,
    'sum': _MERGE_SUM, 'count': _MERGE_COUNT,
}


# ---------------------------------------------------------------------------
# Merge pixel helper (CPU)
# ---------------------------------------------------------------------------

@ngjit
def _merge_pixel(out, written, r, c, val, mode):
    """Write *val* into ``out[r, c]`` using the given merge strategy.

    A separate ``written`` array (int8) tracks which pixels have been
    touched, replacing the previous NaN-sentinel approach which failed
    when the caller intentionally burned NaN values.
    """
    if mode == 0:  # last -- unconditional overwrite, written not read
        out[r, c] = val
    elif mode == 1:  # first
        if written[r, c] == 0:
            out[r, c] = val
            written[r, c] = 1
    elif mode == 2:  # max
        if written[r, c] == 0 or val > out[r, c]:
            out[r, c] = val
            written[r, c] = 1
    elif mode == 3:  # min
        if written[r, c] == 0 or val < out[r, c]:
            out[r, c] = val
            written[r, c] = 1
    elif mode == 4:  # sum
        if written[r, c] == 0:
            out[r, c] = val
            written[r, c] = 1
        else:
            out[r, c] = out[r, c] + val
    else:  # count
        if written[r, c] == 0:
            out[r, c] = 1.0
            written[r, c] = 1
        else:
            out[r, c] = out[r, c] + 1.0


# ---------------------------------------------------------------------------
# Geometry classification (single pass)
# ---------------------------------------------------------------------------

def _classify_geometries(geometries, values):
    """Classify geometries by type in a single pass.

    Also tracks each polygon's input index so the scanline fill can
    process geometries in input order (needed for first/last merge).

    GeometryCollections are recursively unpacked so their contents are
    rasterized rather than silently dropped.

    Returns
    -------
    (poly_geoms, poly_vals, poly_ids),
    (line_geoms, line_vals),
    (point_geoms, point_vals)
    """
    poly_geoms, poly_vals, poly_ids = [], [], []
    line_geoms, line_vals = [], []
    point_geoms, point_vals = [], []

    def _classify_one(geom, val, idx):
        if geom is None or geom.is_empty:
            return
        gt = geom.geom_type
        if gt in ('Polygon', 'MultiPolygon'):
            poly_geoms.append(geom)
            poly_vals.append(val)
            poly_ids.append(idx)
        elif gt in ('LineString', 'MultiLineString'):
            line_geoms.append(geom)
            line_vals.append(val)
        elif gt in ('Point', 'MultiPoint'):
            point_geoms.append(geom)
            point_vals.append(val)
        elif gt == 'GeometryCollection':
            for sub in geom.geoms:
                _classify_one(sub, val, idx)

    for idx, (geom, val) in enumerate(zip(geometries, values)):
        _classify_one(geom, val, idx)

    return ((poly_geoms, poly_vals, poly_ids),
            (line_geoms, line_vals),
            (point_geoms, point_vals))


# ---------------------------------------------------------------------------
# Edge table construction
# ---------------------------------------------------------------------------

_EMPTY_EDGES = (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64), np.empty(0, np.float64),
                np.empty(0, np.float64), np.empty(0, np.int32))


def _extract_edges(geometries, values, geom_ids, bounds, height, width,
                   all_touched=False):
    """Build the edge table for polygon scanline fill.

    Returns
    -------
    edge_y_min, edge_y_max : int32 arrays
    edge_x_at_ymin, edge_inv_slope, edge_value : float64 arrays
    edge_geom_id : int32 array -- input geometry index for ordering
    """
    if not geometries:
        return _EMPTY_EDGES
    if _HAS_SHAPELY2:
        return _extract_edges_vectorized(
            geometries, values, geom_ids, bounds, height, width, all_touched)
    return _extract_edges_loop(
        geometries, values, geom_ids, bounds, height, width, all_touched)


def _extract_edges_vectorized(geometries, values, geom_ids, bounds,
                              height, width, all_touched):
    """Vectorized edge extraction using shapely 2.0 array ops."""
    import shapely

    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    geom_arr = np.array(geometries, dtype=object)
    val_arr = np.array(values, dtype=np.float64)
    id_arr = np.array(geom_ids, dtype=np.int32)

    # Explode MultiPolygons to individual Polygons
    parts, part_idx = shapely.get_parts(geom_arr, return_index=True)
    part_vals = val_arr[part_idx]
    part_ids = id_arr[part_idx]

    # Get all rings (exterior + interior)
    rings, ring_idx = shapely.get_rings(parts, return_index=True)
    ring_vals = part_vals[ring_idx]
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

    # Burn value and geometry id for each edge
    edge_vals = ring_vals[coord_ring_idx[start_idx]]
    edge_ids = ring_ids[coord_ring_idx[start_idx]]

    # Convert to pixel space
    start_row = (ymax - coords[start_idx, 1]) / py
    start_col = (coords[start_idx, 0] - xmin) / px
    end_row = (ymax - coords[end_idx, 1]) / py
    end_col = (coords[end_idx, 0] - xmin) / px

    # Drop horizontal edges
    not_horiz = start_row != end_row
    start_row = start_row[not_horiz]
    start_col = start_col[not_horiz]
    end_row = end_row[not_horiz]
    end_col = end_col[not_horiz]
    edge_vals = edge_vals[not_horiz]
    edge_ids = edge_ids[not_horiz]

    if len(start_row) == 0:
        return _EMPTY_EDGES

    # Orient edges so top_r < bot_r
    swap = start_row > end_row
    top_r = np.where(swap, end_row, start_row)
    top_c = np.where(swap, end_col, start_col)
    bot_r = np.where(swap, start_row, end_row)
    bot_c = np.where(swap, start_col, end_col)

    # Clamp to raster rows
    if all_touched:
        ry_min = np.maximum(np.floor(top_r - 0.5).astype(np.int32), 0)
        ry_max = np.minimum(
            np.ceil(bot_r + 0.5).astype(np.int32) - 1, height - 1)
    else:
        ry_min = np.maximum(np.ceil(top_r).astype(np.int32), 0)
        ry_max = np.minimum(
            np.ceil(bot_r).astype(np.int32) - 1, height - 1)

    # Only keep edges that span at least one row
    valid = ry_min <= ry_max

    # Inverse slope and x at first active row
    dr = bot_r - top_r  # guaranteed != 0
    inv_slope = (bot_c - top_c) / dr
    x_at_ymin = top_c + (ry_min.astype(np.float64) - top_r) * inv_slope

    return (ry_min[valid],
            ry_max[valid],
            x_at_ymin[valid],
            inv_slope[valid],
            edge_vals[valid],
            edge_ids[valid])


def _extract_edges_loop(geometries, values, geom_ids, bounds, height, width,
                        all_touched):
    """Loop-based edge extraction (shapely < 2.0 fallback)."""
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    all_y_min = []
    all_y_max = []
    all_x_at_ymin = []
    all_inv_slope = []
    all_value = []
    all_geom_id = []

    for geom, val, gid in zip(geometries, values, geom_ids):
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
                        ry_max = min(
                            int(np.ceil(r1 + 0.5)) - 1, height - 1)
                    else:
                        ry_min = max(int(np.ceil(r0)), 0)
                        ry_max = min(
                            int(np.ceil(r1)) - 1, height - 1)
                    if ry_min > ry_max:
                        continue
                    inv_slope = (c1 - c0) / (r1 - r0)
                    x_at_ymin = c0 + (ry_min - r0) * inv_slope
                    all_y_min.append(np.int32(ry_min))
                    all_y_max.append(np.int32(ry_max))
                    all_x_at_ymin.append(x_at_ymin)
                    all_inv_slope.append(inv_slope)
                    all_value.append(np.float64(val))
                    all_geom_id.append(np.int32(gid))

    if not all_y_min:
        return _EMPTY_EDGES

    return (np.array(all_y_min, np.int32),
            np.array(all_y_max, np.int32),
            np.array(all_x_at_ymin, np.float64),
            np.array(all_inv_slope, np.float64),
            np.array(all_value, np.float64),
            np.array(all_geom_id, np.int32))


def _sort_edges(edge_arrays):
    """Sort edge table by y_min for scanline early termination."""
    if len(edge_arrays[0]) == 0:
        return edge_arrays
    order = np.argsort(edge_arrays[0], kind='stable')
    return tuple(arr[order] for arr in edge_arrays)


# ---------------------------------------------------------------------------
# Point extraction (always on host)
# ---------------------------------------------------------------------------

def _extract_points(geometries, values, bounds, height, width):
    """Parse Point/MultiPoint geometries into pixel coordinate arrays."""
    if not geometries:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64))
    if _HAS_SHAPELY2:
        return _extract_points_vectorized(
            geometries, values, bounds, height, width)
    return _extract_points_loop(
        geometries, values, bounds, height, width)


def _extract_points_vectorized(geometries, values, bounds, height, width):
    """Vectorized point extraction using shapely 2.0 array ops."""
    import shapely

    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    geom_arr = np.array(geometries, dtype=object)
    val_arr = np.array(values, dtype=np.float64)

    # Explode MultiPoints to individual Points
    parts, part_idx = shapely.get_parts(geom_arr, return_index=True)
    part_vals = val_arr[part_idx]

    if len(parts) == 0:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64))

    # Extract coordinates with index back to each point
    coords, coord_idx = shapely.get_coordinates(
        parts, return_index=True)
    pt_vals = part_vals[coord_idx]

    cols = np.floor((coords[:, 0] - xmin) / px).astype(np.int32)
    rows = np.floor((ymax - coords[:, 1]) / py).astype(np.int32)

    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return (rows[valid], cols[valid], pt_vals[valid])


def _extract_points_loop(geometries, values, bounds, height, width):
    """Loop-based point extraction (shapely < 2.0 fallback)."""
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    all_rows, all_cols, all_vals = [], [], []

    for geom, val in zip(geometries, values):
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
                all_rows.append(np.int32(row))
                all_cols.append(np.int32(col))
                all_vals.append(np.float64(val))

    if not all_rows:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64))
    return (np.array(all_rows, np.int32),
            np.array(all_cols, np.int32),
            np.array(all_vals, np.float64))


# ---------------------------------------------------------------------------
# Line segment extraction (always on host)
# ---------------------------------------------------------------------------

_EMPTY_LINES = (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64))


def _extract_line_segments(geometries, values, bounds, height, width):
    """Parse LineString/MultiLineString geometries into pixel-space segments.

    Segments are clipped to the raster extent before conversion to pixel
    coordinates, so Bresenham never iterates over out-of-bounds pixels.
    """
    if not geometries:
        return _EMPTY_LINES
    if _HAS_SHAPELY2:
        return _extract_lines_vectorized(
            geometries, values, bounds, height, width)
    return _extract_lines_loop(
        geometries, values, bounds, height, width)


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


def _extract_lines_vectorized(geometries, values, bounds, height, width):
    """Vectorized line extraction with Liang-Barsky clipping."""
    import shapely

    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    geom_arr = np.array(geometries, dtype=object)
    val_arr = np.array(values, dtype=np.float64)

    # Explode MultiLineStrings to individual LineStrings
    parts, part_idx = shapely.get_parts(geom_arr, return_index=True)
    part_vals = val_arr[part_idx]

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
    seg_vals = part_vals[coord_line_idx[start_idx]]

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
    return (r0[v], c0[v], r1[v], c1[v], seg_vals[v])


def _extract_lines_loop(geometries, values, bounds, height, width):
    """Loop-based line extraction with Liang-Barsky clipping (fallback)."""
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    all_r0, all_c0, all_r1, all_c1, all_vals = [], [], [], [], []

    for geom, val in zip(geometries, values):
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
                all_r0.append(np.int32(r0))
                all_c0.append(np.int32(c0))
                all_r1.append(np.int32(r1))
                all_c1.append(np.int32(c1))
                all_vals.append(np.float64(val))

    if not all_r0:
        return _EMPTY_LINES
    return (np.array(all_r0, np.int32), np.array(all_c0, np.int32),
            np.array(all_r1, np.int32), np.array(all_c1, np.int32),
            np.array(all_vals, np.float64))


# ---------------------------------------------------------------------------
# CPU burn kernels (numba)
# ---------------------------------------------------------------------------

@ngjit
def _burn_points_cpu(out, written, rows, cols, vals, mode):
    for i in range(len(rows)):
        r = rows[i]
        c = cols[i]
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            _merge_pixel(out, written, r, c, vals[i], mode)


@ngjit
def _burn_lines_cpu(out, written, r0_arr, c0_arr, r1_arr, c1_arr, vals,
                    height, width, mode):
    for i in range(len(r0_arr)):
        r0 = r0_arr[i]
        c0 = c0_arr[i]
        r1 = r1_arr[i]
        c1 = c1_arr[i]
        val = vals[i]

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
                    _merge_pixel(out, written, r, c, val, mode)
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
                    _merge_pixel(out, written, r, c, val, mode)
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
                       edge_inv_slope, edge_value, edge_geom_id,
                       height, width, mode):
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
    vs = np.empty(n_edges, dtype=np.float64)
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
            vs[i] = edge_value[e]
            gs[i] = edge_geom_id[e]

        # 4. Insertion sort by (geom_id, x) so each geometry's edges pair
        # correctly and geometries are processed in input order.
        for i in range(1, n_active):
            kx = xs[i]
            kv = vs[i]
            kg = gs[i]
            j = i - 1
            while j >= 0 and (gs[j] > kg or (gs[j] == kg and xs[j] > kx)):
                xs[j + 1] = xs[j]
                vs[j + 1] = vs[j]
                gs[j + 1] = gs[j]
                j -= 1
            xs[j + 1] = kx
            vs[j + 1] = kv
            gs[j + 1] = kg

        # 5. Fill between edge pairs per geometry
        i = 0
        while i < n_active - 1:
            gid = gs[i]
            val = vs[i]
            j = i
            while j < n_active and gs[j] == gid:
                j += 1
            k = i
            while k + 1 < j:
                x_start = xs[k]
                x_end = xs[k + 1]
                col_start = max(int(np.ceil(x_start)), 0)
                col_end = min(int(np.floor(x_end)), width - 1)
                for c in range(col_start, col_end + 1):
                    _merge_pixel(out, written, row, c, val, mode)
                k += 2
            i = j


def _run_numpy(geometries, values, bounds, height, width, fill, dtype,
               all_touched, merge_mode):
    """NumPy backend for rasterize."""
    out = np.full((height, width), fill, dtype=np.float64)

    # For non-'last' modes we need a written mask to track which pixels
    # have been touched (replacing the old NaN-sentinel approach).
    if merge_mode != _MERGE_LAST:
        written = np.zeros((height, width), dtype=np.int8)
    else:
        # Dummy -- never indexed, but numba needs a typed array argument
        written = np.empty((0, 0), dtype=np.int8)

    (poly_geoms, poly_vals, poly_ids), (line_geoms, line_vals), \
        (point_geoms, point_vals) = _classify_geometries(geometries, values)

    # 1. Polygons
    edge_arrays = _extract_edges(
        poly_geoms, poly_vals, poly_ids, bounds, height, width, all_touched)
    edge_arrays = _sort_edges(edge_arrays)
    if len(edge_arrays[0]) > 0:
        _scanline_fill_cpu(out, written, *edge_arrays, height, width,
                           merge_mode)

    # 2. Lines
    r0, c0, r1, c1, lvals = _extract_line_segments(
        line_geoms, line_vals, bounds, height, width)
    if len(r0) > 0:
        _burn_lines_cpu(out, written, r0, c0, r1, c1, lvals, height, width,
                        merge_mode)

    # 3. Points
    prows, pcols, pvals = _extract_points(
        point_geoms, point_vals, bounds, height, width)
    if len(prows) > 0:
        _burn_points_cpu(out, written, prows, pcols, pvals, merge_mode)

    return out.astype(dtype)


# ---------------------------------------------------------------------------
# GPU kernels -- compiled lazily to avoid importing numba.cuda at module
# level (~160ms + CUDA driver init even when not using GPU).
# ---------------------------------------------------------------------------

_gpu_kernels = None


def _ensure_gpu_kernels():
    """Compile CUDA kernels on first use and cache them."""
    global _gpu_kernels
    if _gpu_kernels is not None:
        return _gpu_kernels

    from numba import cuda

    @cuda.jit(device=True)
    def _merge_pixel_gpu(out, written, r, c, val, mode):
        if mode == 0:  # last
            out[r, c] = val
        elif mode == 1:  # first
            if written[r, c] == 0:
                out[r, c] = val
                written[r, c] = 1
        elif mode == 2:  # max
            if written[r, c] == 0 or val > out[r, c]:
                out[r, c] = val
                written[r, c] = 1
        elif mode == 3:  # min
            if written[r, c] == 0 or val < out[r, c]:
                out[r, c] = val
                written[r, c] = 1
        elif mode == 4:  # sum
            if written[r, c] == 0:
                out[r, c] = val
                written[r, c] = 1
            else:
                out[r, c] = out[r, c] + val
        else:  # count
            if written[r, c] == 0:
                out[r, c] = 1.0
                written[r, c] = 1
            else:
                out[r, c] = out[r, c] + 1.0

    @cuda.jit
    def _scanline_fill_gpu(out, written, edge_y_min, edge_x_at_ymin,
                           edge_inv_slope, edge_value, edge_geom_id,
                           row_ptr, col_idx, width, mode):
        """CUDA kernel: one thread per raster row, CSR-indexed active edges.

        Instead of binary-searching the sorted edge table and scanning
        through dead edges, each thread reads its active edge list
        directly from the precomputed CSR structure (row_ptr, col_idx).
        """
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
        vs = cuda.local.array(512, dtype=np.float64)
        gs = cuda.local.array(512, dtype=np.int32)

        actual = 0
        for k in range(start, end):
            if actual >= MAX_ISECT:
                break
            e = col_idx[k]
            xs[actual] = (edge_x_at_ymin[e]
                          + (row - edge_y_min[e]) * edge_inv_slope[e])
            vs[actual] = edge_value[e]
            gs[actual] = edge_geom_id[e]
            actual += 1

        # Insertion sort by (geom_id, x)
        for i in range(1, actual):
            kx = xs[i]
            kv = vs[i]
            kg = gs[i]
            j = i - 1
            while j >= 0 and (gs[j] > kg or (gs[j] == kg and xs[j] > kx)):
                xs[j + 1] = xs[j]
                vs[j + 1] = vs[j]
                gs[j + 1] = gs[j]
                j -= 1
            xs[j + 1] = kx
            vs[j + 1] = kv
            gs[j + 1] = kg

        # Fill between pairs per geometry
        i = 0
        while i < actual - 1:
            gid = gs[i]
            val = vs[i]
            j = i
            while j < actual and gs[j] == gid:
                j += 1
            k = i
            while k + 1 < j:
                x_start = xs[k]
                x_end = xs[k + 1]
                col_start = int(x_start + 0.999999)
                if col_start < 0:
                    col_start = 0
                col_end = int(x_end)
                if col_end >= width:
                    col_end = width - 1
                for c in range(col_start, col_end + 1):
                    _merge_pixel_gpu(out, written, row, c, val, mode)
                k += 2
            i = j

    @cuda.jit
    def _burn_points_gpu(out, written, rows, cols, vals, n_points, mode):
        i = cuda.grid(1)
        if i >= n_points:
            return
        r = rows[i]
        c = cols[i]
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            _merge_pixel_gpu(out, written, r, c, vals[i], mode)

    @cuda.jit
    def _burn_lines_gpu(out, written, r0_arr, c0_arr, r1_arr, c1_arr, vals,
                        n_segs, height, width, mode):
        i = cuda.grid(1)
        if i >= n_segs:
            return
        r0 = r0_arr[i]
        c0 = c0_arr[i]
        r1 = r1_arr[i]
        c1 = c1_arr[i]
        val = vals[i]

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
                    _merge_pixel_gpu(out, written, r, c, val, mode)
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
                    _merge_pixel_gpu(out, written, r, c, val, mode)
                if err >= 0:
                    r += sr
                    err -= dc
                c += sc
                err += dr

    _gpu_kernels = {
        'scanline_fill': _scanline_fill_gpu,
        'burn_points': _burn_points_gpu,
        'burn_lines': _burn_lines_gpu,
    }
    return _gpu_kernels


@ngjit
def _build_row_csr_numba(edge_y_min, edge_y_max, height):
    """Numba-accelerated CSR builder for GPU scanline precomputation."""
    n_edges = len(edge_y_min)

    # Pass 1: count active edges per row
    counts = np.zeros(height, dtype=np.int32)
    for e in range(n_edges):
        y_lo = edge_y_min[e]
        y_hi = edge_y_max[e]
        if y_hi >= height:
            y_hi = height - 1
        for r in range(y_lo, y_hi + 1):
            counts[r] += 1

    # Build row_ptr (prefix sum)
    row_ptr = np.empty(height + 1, dtype=np.int32)
    row_ptr[0] = 0
    for r in range(height):
        row_ptr[r + 1] = row_ptr[r] + counts[r]

    # Pass 2: fill col_idx
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


def _run_cupy(geometries, values, bounds, height, width, fill, dtype,
              all_touched, merge_mode):
    """CuPy backend for rasterize."""
    kernels = _ensure_gpu_kernels()

    out = cupy.full((height, width), fill, dtype=cupy.float64)
    if merge_mode != _MERGE_LAST:
        written = cupy.zeros((height, width), dtype=cupy.int8)
    else:
        written = cupy.empty((0, 0), dtype=cupy.int8)

    (poly_geoms, poly_vals, poly_ids), (line_geoms, line_vals), \
        (point_geoms, point_vals) = _classify_geometries(geometries, values)

    # 1. Polygons
    edge_arrays = _extract_edges(
        poly_geoms, poly_vals, poly_ids, bounds, height, width, all_touched)
    edge_arrays = _sort_edges(edge_arrays)
    edge_y_min, edge_y_max, edge_x_at_ymin, edge_inv_slope, \
        edge_value, edge_geom_id = edge_arrays

    if len(edge_y_min) > 0:
        # Build CSR structure on CPU, then transfer to GPU
        row_ptr, col_idx = _build_row_csr_numba(
            edge_y_min, edge_y_max, height)

        d_y_min = cupy.asarray(edge_y_min)
        d_x_at_ymin = cupy.asarray(edge_x_at_ymin)
        d_inv_slope = cupy.asarray(edge_inv_slope)
        d_value = cupy.asarray(edge_value)
        d_geom_id = cupy.asarray(edge_geom_id)
        d_row_ptr = cupy.asarray(row_ptr)
        d_col_idx = cupy.asarray(col_idx)

        tpb = 256
        blocks = (height + tpb - 1) // tpb
        kernels['scanline_fill'][blocks, tpb](
            out, written, d_y_min, d_x_at_ymin, d_inv_slope,
            d_value, d_geom_id, d_row_ptr, d_col_idx, width, merge_mode)

    # 2. Lines
    r0, c0, r1, c1, lvals = _extract_line_segments(
        line_geoms, line_vals, bounds, height, width)
    if len(r0) > 0:
        n_segs = len(r0)
        tpb = 256
        bpg = (n_segs + tpb - 1) // tpb
        kernels['burn_lines'][bpg, tpb](
            out, written, cupy.asarray(r0), cupy.asarray(c0),
            cupy.asarray(r1), cupy.asarray(c1),
            cupy.asarray(lvals), n_segs, height, width, merge_mode)

    # 3. Points
    prows, pcols, pvals = _extract_points(
        point_geoms, point_vals, bounds, height, width)
    if len(prows) > 0:
        n_pts = len(prows)
        tpb = 256
        bpg = (n_pts + tpb - 1) // tpb
        kernels['burn_points'][bpg, tpb](
            out, written, cupy.asarray(prows), cupy.asarray(pcols),
            cupy.asarray(pvals), n_pts, merge_mode)

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


def _segments_for_tile(r0, c0, r1, c1, vals, r_start, r_end, c_start, c_end):
    """Filter segments whose pixel bbox overlaps the tile, offset to local.

    Returns segments in tile-local coordinates (r_start/c_start subtracted).
    Endpoints can be negative or exceed tile dimensions — the Bresenham
    bounds check in ``_burn_lines_cpu`` handles this, and the pixel path
    is translation-invariant so the result is exact.
    """
    if len(r0) == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty, empty, np.empty(0, dtype=np.float64)
    seg_rmin = np.minimum(r0, r1)
    seg_rmax = np.maximum(r0, r1)
    seg_cmin = np.minimum(c0, c1)
    seg_cmax = np.maximum(c0, c1)
    mask = ((seg_rmax >= r_start) & (seg_rmin < r_end) &
            (seg_cmax >= c_start) & (seg_cmin < c_end))
    return (r0[mask] - r_start, c0[mask] - c_start,
            r1[mask] - r_start, c1[mask] - c_start,
            vals[mask])


def _points_for_tile(rows, cols, vals, r_start, r_end, c_start, c_end):
    """Filter points within the tile, offset to tile-local coordinates."""
    if len(rows) == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, np.empty(0, dtype=np.float64)
    mask = ((rows >= r_start) & (rows < r_end) &
            (cols >= c_start) & (cols < c_end))
    return (rows[mask] - r_start, cols[mask] - c_start, vals[mask])


def _polys_to_wkb(geoms, vals):
    """Pre-serialize polygon geometries to WKB for cheap pickling."""
    wkb_list = [g.wkb for g in geoms]
    return wkb_list, list(vals)


def _polys_from_wkb(wkb_list, vals):
    """Deserialize WKB back to shapely geometries."""
    from shapely import from_wkb
    geoms = from_wkb(wkb_list)
    if not isinstance(geoms, (list, np.ndarray)):
        geoms = [geoms]
    return list(geoms), vals


def _rasterize_tile_numpy(poly_wkb, poly_vals, tile_bounds, tile_h, tile_w,
                          fill, dtype, all_touched, merge_mode,
                          seg_r0, seg_c0, seg_r1, seg_c1, seg_vals,
                          pt_rows, pt_cols, pt_vals):
    """Rasterize a single tile.

    Polygons are passed as WKB bytes (cheap to pickle) and deserialized
    inside the worker.  Line segments and points are passed in tile-local
    pixel coordinates.
    """
    out = np.full((tile_h, tile_w), fill, dtype=np.float64)

    if merge_mode != _MERGE_LAST:
        written = np.zeros((tile_h, tile_w), dtype=np.int8)
    else:
        written = np.empty((0, 0), dtype=np.int8)

    # 1. Polygons (deserialize WKB, then scanline fill)
    if poly_wkb:
        poly_geoms, poly_vals = _polys_from_wkb(poly_wkb, poly_vals)
        poly_ids = list(range(len(poly_geoms)))
        edge_arrays = _extract_edges(
            poly_geoms, poly_vals, poly_ids, tile_bounds,
            tile_h, tile_w, all_touched)
        edge_arrays = _sort_edges(edge_arrays)
        if len(edge_arrays[0]) > 0:
            _scanline_fill_cpu(out, written, *edge_arrays,
                               tile_h, tile_w, merge_mode)

    # 2. Lines (tile-local segments, Bresenham with bounds check)
    if len(seg_r0) > 0:
        _burn_lines_cpu(out, written, seg_r0, seg_c0, seg_r1, seg_c1,
                        seg_vals, tile_h, tile_w, merge_mode)

    # 3. Points (tile-local)
    if len(pt_rows) > 0:
        _burn_points_cpu(out, written, pt_rows, pt_cols, pt_vals,
                         merge_mode)

    return out.astype(dtype)


def _run_dask_numpy(geometries, values, bounds, height, width, fill, dtype,
                    all_touched, merge_mode, row_chunks, col_chunks):
    """Dask + NumPy backend: tile-based parallel rasterization."""
    import dask
    import dask.array as da

    # Classify geometries once
    (poly_geoms, poly_vals, _poly_ids), (line_geoms, line_vals), \
        (point_geoms, point_vals) = _classify_geometries(geometries, values)

    # Compact representations: segments (5 arrays) and points (3 arrays)
    # in full-raster pixel space.  No pixel expansion here.
    seg_r0, seg_c0, seg_r1, seg_c1, seg_vals = _extract_line_segments(
        line_geoms, line_vals, bounds, height, width)
    pt_rows, pt_cols, pt_vals = _extract_points(
        point_geoms, point_vals, bounds, height, width)

    # Pre-serialize polygons to WKB (20x cheaper to pickle than shapely)
    if poly_geoms:
        poly_bboxes = _geometry_bboxes(poly_geoms)
        poly_wkb = [g.wkb for g in poly_geoms]
    else:
        poly_bboxes = np.empty((0, 4), dtype=np.float64)
        poly_wkb = []
    poly_val_arr = np.asarray(poly_vals, dtype=np.float64)

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

            # Filter polygons by tile geo bbox
            pmask = _filter_geoms_to_tile(poly_bboxes, tile_bounds)
            if len(poly_wkb) > 0:
                tile_wkb = [poly_wkb[k] for k in np.nonzero(pmask)[0]]
                tile_vals = poly_val_arr[pmask].tolist()
            else:
                tile_wkb = []
                tile_vals = []

            # Filter segments and points by tile pixel range
            ts = _segments_for_tile(seg_r0, seg_c0, seg_r1, seg_c1,
                                    seg_vals,
                                    r_start, r_end, c_start, c_end)
            tp = _points_for_tile(pt_rows, pt_cols, pt_vals,
                                  r_start, r_end, c_start, c_end)

            delayed_tile = dask.delayed(_rasterize_tile_numpy)(
                tile_wkb, tile_vals, tile_bounds,
                tile_h, tile_w, fill, dtype, all_touched, merge_mode,
                *ts, *tp)
            blocks[i][j] = da.from_delayed(
                delayed_tile, shape=(tile_h, tile_w), dtype=dtype)
            ri += 1

    return da.block(blocks)


def _rasterize_tile_cupy(poly_wkb, poly_vals, tile_bounds, tile_h, tile_w,
                         fill, dtype, all_touched, merge_mode,
                         seg_r0, seg_c0, seg_r1, seg_c1, seg_vals,
                         pt_rows, pt_cols, pt_vals):
    """GPU tile rasterization: polygons as WKB, lines/points as segments."""
    kernels = _ensure_gpu_kernels()

    out = cupy.full((tile_h, tile_w), fill, dtype=cupy.float64)
    if merge_mode != _MERGE_LAST:
        written = cupy.zeros((tile_h, tile_w), dtype=cupy.int8)
    else:
        written = cupy.empty((0, 0), dtype=cupy.int8)

    # 1. Polygons (deserialize WKB, then scanline fill on GPU)
    if poly_wkb:
        poly_geoms, poly_vals = _polys_from_wkb(poly_wkb, poly_vals)
        poly_ids = list(range(len(poly_geoms)))
        edge_arrays = _extract_edges(
            poly_geoms, poly_vals, poly_ids, tile_bounds,
            tile_h, tile_w, all_touched)
        edge_arrays = _sort_edges(edge_arrays)
        edge_y_min = edge_arrays[0]
        if len(edge_y_min) > 0:
            edge_y_max, edge_x_at_ymin, edge_inv_slope, \
                edge_value, edge_geom_id = edge_arrays[1:]
            row_ptr, col_idx = _build_row_csr_numba(
                edge_y_min, edge_y_max, tile_h)
            tpb = 256
            blocks = (tile_h + tpb - 1) // tpb
            kernels['scanline_fill'][blocks, tpb](
                out, written,
                cupy.asarray(edge_y_min), cupy.asarray(edge_x_at_ymin),
                cupy.asarray(edge_inv_slope), cupy.asarray(edge_value),
                cupy.asarray(edge_geom_id), cupy.asarray(row_ptr),
                cupy.asarray(col_idx), tile_w, merge_mode)

    # 2. Lines (tile-local segments, GPU Bresenham)
    if len(seg_r0) > 0:
        n_segs = len(seg_r0)
        tpb = 256
        bpg = (n_segs + tpb - 1) // tpb
        kernels['burn_lines'][bpg, tpb](
            out, written,
            cupy.asarray(seg_r0), cupy.asarray(seg_c0),
            cupy.asarray(seg_r1), cupy.asarray(seg_c1),
            cupy.asarray(seg_vals), n_segs, tile_h, tile_w, merge_mode)

    # 3. Points (tile-local)
    if len(pt_rows) > 0:
        n_pts = len(pt_rows)
        tpb = 256
        bpg = (n_pts + tpb - 1) // tpb
        kernels['burn_points'][bpg, tpb](
            out, written,
            cupy.asarray(pt_rows), cupy.asarray(pt_cols),
            cupy.asarray(pt_vals), n_pts, merge_mode)

    return out.astype(dtype)


def _run_dask_cupy(geometries, values, bounds, height, width, fill, dtype,
                   all_touched, merge_mode, row_chunks, col_chunks):
    """Dask + CuPy backend: tile-based parallel GPU rasterization."""
    import dask
    import dask.array as da

    # Classify geometries once
    (poly_geoms, poly_vals, _poly_ids), (line_geoms, line_vals), \
        (point_geoms, point_vals) = _classify_geometries(geometries, values)

    # Compact representations in full-raster pixel space
    seg_r0, seg_c0, seg_r1, seg_c1, seg_vals = _extract_line_segments(
        line_geoms, line_vals, bounds, height, width)
    pt_rows, pt_cols, pt_vals = _extract_points(
        point_geoms, point_vals, bounds, height, width)

    # Pre-serialize polygons to WKB (20x cheaper to pickle than shapely)
    if poly_geoms:
        poly_bboxes = _geometry_bboxes(poly_geoms)
        poly_wkb = [g.wkb for g in poly_geoms]
    else:
        poly_bboxes = np.empty((0, 4), dtype=np.float64)
        poly_wkb = []
    poly_val_arr = np.asarray(poly_vals, dtype=np.float64)

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

            # Filter polygons by tile geo bbox
            pmask = _filter_geoms_to_tile(poly_bboxes, tile_bounds)
            if len(poly_wkb) > 0:
                tile_wkb = [poly_wkb[k] for k in np.nonzero(pmask)[0]]
                tile_vals = poly_val_arr[pmask].tolist()
            else:
                tile_wkb = []
                tile_vals = []

            ts = _segments_for_tile(seg_r0, seg_c0, seg_r1, seg_c1,
                                    seg_vals,
                                    r_start, r_end, c_start, c_end)
            tp = _points_for_tile(pt_rows, pt_cols, pt_vals,
                                  r_start, r_end, c_start, c_end)

            delayed_tile = dask.delayed(_rasterize_tile_cupy)(
                tile_wkb, tile_vals, tile_bounds,
                tile_h, tile_w, fill, dtype, all_touched, merge_mode,
                *ts, *tp)
            blocks[i][j] = da.from_delayed(
                delayed_tile, shape=(tile_h, tile_w), dtype=dtype,
                meta=cupy.empty(()))
            ri += 1

    return da.block(blocks)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _parse_input(geometries, column=None):
    """Normalise input to (geometry_list, value_list, bounds)."""
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
            if column is None:
                numeric_cols = geometries.select_dtypes(
                    include='number').columns
                if len(numeric_cols) == 0:
                    raise ValueError(
                        "GeoDataFrame has no numeric columns to burn. "
                        "Pass a 'column' name explicitly.")
                column = numeric_cols[0]
            geom_list = geometries.geometry.tolist()
            value_list = geometries[column].values.astype(np.float64).tolist()
            total_bounds = geometries.total_bounds
            return geom_list, value_list, tuple(total_bounds)
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
        return geom_list, value_list, None

    all_bounds = np.array([g.bounds for g in geom_list if g is not None])
    if len(all_bounds) == 0:
        return geom_list, value_list, None
    total_bounds = (all_bounds[:, 0].min(), all_bounds[:, 1].min(),
                    all_bounds[:, 2].max(), all_bounds[:, 3].max())
    return geom_list, value_list, total_bounds


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
    fill: float = np.nan,
    dtype: Optional[np.dtype] = None,
    all_touched: bool = False,
    use_cuda: bool = False,
    name: str = 'rasterize',
    resolution: Optional[Union[float, Tuple[float, float]]] = None,
    like: Optional[xr.DataArray] = None,
    merge: str = 'last',
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
    merge : str, default 'last'
        How to combine values when geometries overlap:

        - ``'last'`` -- last geometry in input order wins
        - ``'first'`` -- first geometry wins
        - ``'max'`` / ``'min'`` -- keep the larger / smaller value
        - ``'sum'`` -- add values together
        - ``'count'`` -- count overlapping geometries

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
    # Validate merge mode
    if merge not in _MERGE_MODES:
        raise ValueError(
            f"merge must be one of {set(_MERGE_MODES)}, got {merge!r}")
    merge_mode = _MERGE_MODES[merge]

    # Extract defaults from template raster
    like_width = like_height = like_bounds = like_dtype = None
    if like is not None:
        like_width, like_height, like_bounds, like_dtype = \
            _extract_grid_from_like(like)

    # Parse input geometries
    geom_list, value_list, inferred_bounds = _parse_input(geometries, column)

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

    if chunks is not None:
        row_chunks, col_chunks = _normalize_chunks(
            chunks, final_height, final_width)
        if use_cuda:
            if cupy is None:
                raise ImportError(
                    "CuPy is required for use_cuda=True but is not installed")
            out = _run_dask_cupy(
                geom_list, value_list, final_bounds,
                final_height, final_width, fill, final_dtype,
                all_touched, merge_mode, row_chunks, col_chunks)
        else:
            out = _run_dask_numpy(
                geom_list, value_list, final_bounds,
                final_height, final_width, fill, final_dtype,
                all_touched, merge_mode, row_chunks, col_chunks)
    elif use_cuda:
        if cupy is None:
            raise ImportError(
                "CuPy is required for use_cuda=True but is not installed")
        out = _run_cupy(geom_list, value_list, final_bounds,
                        final_height, final_width, fill, final_dtype,
                        all_touched, merge_mode)
    else:
        out = _run_numpy(geom_list, value_list, final_bounds,
                         final_height, final_width, fill, final_dtype,
                         all_touched, merge_mode)

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
