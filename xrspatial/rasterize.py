"""Vector geometry rasterization (polygons, lines, points).

Converts vector geometries (GeoDataFrame or list of (geometry, value) pairs)
to a 2D xr.DataArray.  No GDAL dependency.

- Polygons/MultiPolygons: scanline fill
- Lines/MultiLineStrings: Bresenham line rasterization
- Points/MultiPoints: direct pixel burn

Supports numpy and cupy backends.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from xrspatial.utils import ngjit

try:
    import cupy
except ImportError:
    cupy = None

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

def _extract_line_segments(geometries, values, bounds, height, width):
    """Parse LineString/MultiLineString geometries into pixel-space segments."""
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
            rows = (ymax - coords[:, 1]) / py
            cols = (coords[:, 0] - xmin) / px
            for i in range(len(coords) - 1):
                all_r0.append(np.int32(int(np.floor(rows[i]))))
                all_c0.append(np.int32(int(np.floor(cols[i]))))
                all_r1.append(np.int32(int(np.floor(rows[i + 1]))))
                all_c1.append(np.int32(int(np.floor(cols[i + 1]))))
                all_vals.append(np.float64(val))

    if not all_r0:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64))
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
    def _scanline_fill_gpu(out, written, edge_y_min, edge_y_max,
                           edge_x_at_ymin, edge_inv_slope, edge_value,
                           edge_geom_id, n_edges, width, mode):
        """CUDA kernel: one thread per raster row."""
        row = cuda.grid(1)
        if row >= out.shape[0]:
            return

        lo, hi = 0, n_edges
        while lo < hi:
            mid = (lo + hi) // 2
            if edge_y_min[mid] <= row:
                lo = mid + 1
            else:
                hi = mid

        count = 0
        for e in range(hi):
            if edge_y_max[e] >= row:
                count += 1

        if count == 0:
            return

        MAX_ISECT = 512
        if count > MAX_ISECT:
            count = MAX_ISECT

        xs = cuda.local.array(512, dtype=np.float64)
        vs = cuda.local.array(512, dtype=np.float64)
        gs = cuda.local.array(512, dtype=np.int32)

        idx = 0
        for e in range(hi):
            if idx >= MAX_ISECT:
                break
            if edge_y_max[e] >= row:
                xs[idx] = (edge_x_at_ymin[e]
                           + (row - edge_y_min[e]) * edge_inv_slope[e])
                vs[idx] = edge_value[e]
                gs[idx] = edge_geom_id[e]
                idx += 1

        actual = idx

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
        d_y_min = cupy.asarray(edge_y_min)
        d_y_max = cupy.asarray(edge_y_max)
        d_x_at_ymin = cupy.asarray(edge_x_at_ymin)
        d_inv_slope = cupy.asarray(edge_inv_slope)
        d_value = cupy.asarray(edge_value)
        d_geom_id = cupy.asarray(edge_geom_id)

        tpb = 256
        blocks = (height + tpb - 1) // tpb
        kernels['scanline_fill'][blocks, tpb](
            out, written, d_y_min, d_y_max, d_x_at_ymin, d_inv_slope,
            d_value, d_geom_id, len(edge_y_min), width, merge_mode)

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
# Input parsing
# ---------------------------------------------------------------------------

def _parse_input(geometries, column=None):
    """Normalise input to (geometry_list, value_list, bounds)."""
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

    if use_cuda:
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
