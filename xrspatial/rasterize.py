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
from numba import cuda

from xrspatial.utils import ngjit

try:
    import cupy
except ImportError:
    cupy = None


# ---------------------------------------------------------------------------
# Edge table construction (always on host)
# ---------------------------------------------------------------------------

def _extract_edges(geometries, values, bounds, height, width):
    """Parse geometries into a flat edge table.

    Returns
    -------
    edge_y_min : ndarray, shape (n_edges,), int32
        Row index of the edge's top (smallest y pixel row).
    edge_y_max : ndarray, shape (n_edges,), int32
        Row index of the edge's bottom (largest y pixel row).
    edge_x_at_ymin : ndarray, shape (n_edges,), float64
        x-coordinate (in pixel space) at ``edge_y_min``.
    edge_inv_slope : ndarray, shape (n_edges,), float64
        1 / slope in pixel space (dx per unit dy).
    edge_value : ndarray, shape (n_edges,), float64
        Burn value for the polygon that owns this edge.
    """
    import shapely

    xmin, ymin, xmax, ymax = bounds
    # pixel size
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    all_y_min = []
    all_y_max = []
    all_x_at_ymin = []
    all_inv_slope = []
    all_value = []

    for geom, val in zip(geometries, values):
        if geom is None or geom.is_empty:
            continue

        # Normalise to MultiPolygon
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
                # Convert to pixel space.
                # Row 0 is at ymax (top), row (height-1) is at ymin (bottom).
                row = (ymax - coords[:, 1]) / py
                col = (coords[:, 0] - xmin) / px

                n = len(row) - 1  # ring is closed, skip last==first
                for i in range(n):
                    r0, c0 = row[i], col[i]
                    r1, c1 = row[i + 1], col[i + 1]
                    if r0 == r1:
                        continue  # horizontal edge, skip
                    if r0 > r1:
                        r0, c0, r1, c1 = r1, c1, r0, c0
                    # Clamp to raster
                    ry_min = max(int(np.ceil(r0)), 0)
                    ry_max = min(int(np.floor(r1)), height) - 1
                    if ry_min > ry_max:
                        continue
                    inv_slope = (c1 - c0) / (r1 - r0)
                    x_at_ymin = c0 + (ry_min - r0) * inv_slope
                    all_y_min.append(np.int32(ry_min))
                    all_y_max.append(np.int32(ry_max))
                    all_x_at_ymin.append(x_at_ymin)
                    all_inv_slope.append(inv_slope)
                    all_value.append(np.float64(val))

    if not all_y_min:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64), np.empty(0, np.float64),
                np.empty(0, np.float64))

    return (np.array(all_y_min, np.int32),
            np.array(all_y_max, np.int32),
            np.array(all_x_at_ymin, np.float64),
            np.array(all_inv_slope, np.float64),
            np.array(all_value, np.float64))


def _extract_edges_all_touched(geometries, values, bounds, height, width):
    """Parse geometries into a flat edge table for all_touched mode.

    In all_touched mode we use sub-pixel edge positions and fill any
    pixel whose center falls within half a pixel of an intersection,
    which requires different clamping.  The scanline kernel handles
    the half-pixel expansion.

    Returns the same arrays as ``_extract_edges`` but edges are NOT
    clamped to integer rows -- the scanline kernel iterates over all
    integer rows in [floor(r0), ceil(r1)-1].
    """
    import shapely

    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    all_y_min = []
    all_y_max = []
    all_x_at_ymin = []
    all_inv_slope = []
    all_value = []

    for geom, val in zip(geometries, values):
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
                    # Expand by half a pixel so edge-adjacent pixels are touched
                    ry_min = max(int(np.floor(r0 - 0.5)), 0)
                    ry_max = min(int(np.ceil(r1 + 0.5)), height) - 1
                    if ry_min > ry_max:
                        continue
                    inv_slope = (c1 - c0) / (r1 - r0)
                    x_at_ymin = c0 + (ry_min - r0) * inv_slope
                    all_y_min.append(np.int32(ry_min))
                    all_y_max.append(np.int32(ry_max))
                    all_x_at_ymin.append(x_at_ymin)
                    all_inv_slope.append(inv_slope)
                    all_value.append(np.float64(val))

    if not all_y_min:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64), np.empty(0, np.float64),
                np.empty(0, np.float64))

    return (np.array(all_y_min, np.int32),
            np.array(all_y_max, np.int32),
            np.array(all_x_at_ymin, np.float64),
            np.array(all_inv_slope, np.float64),
            np.array(all_value, np.float64))


# ---------------------------------------------------------------------------
# Point extraction (always on host)
# ---------------------------------------------------------------------------

def _extract_points(geometries, values, bounds, height, width):
    """Parse Point/MultiPoint geometries into pixel coordinate arrays.

    Returns
    -------
    rows : ndarray, int32 -- row indices
    cols : ndarray, int32 -- column indices
    vals : ndarray, float64 -- burn values
    """
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    all_rows = []
    all_cols = []
    all_vals = []

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
    """Parse LineString/MultiLineString geometries into pixel-space segments.

    Returns
    -------
    r0, c0, r1, c1 : ndarray, int32 -- endpoint pixel coordinates
    vals : ndarray, float64 -- burn values
    """
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height

    all_r0 = []
    all_c0 = []
    all_r1 = []
    all_c1 = []
    all_vals = []

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
                r_a = int(np.floor(rows[i]))
                c_a = int(np.floor(cols[i]))
                r_b = int(np.floor(rows[i + 1]))
                c_b = int(np.floor(cols[i + 1]))
                all_r0.append(np.int32(r_a))
                all_c0.append(np.int32(c_a))
                all_r1.append(np.int32(r_b))
                all_c1.append(np.int32(c_b))
                all_vals.append(np.float64(val))

    if not all_r0:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float64))

    return (np.array(all_r0, np.int32), np.array(all_c0, np.int32),
            np.array(all_r1, np.int32), np.array(all_c1, np.int32),
            np.array(all_vals, np.float64))


# ---------------------------------------------------------------------------
# CPU point burn (numba)
# ---------------------------------------------------------------------------

@ngjit
def _burn_points_cpu(out, rows, cols, vals):
    """Burn point values into the output raster."""
    for i in range(len(rows)):
        r = rows[i]
        c = cols[i]
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = vals[i]


# ---------------------------------------------------------------------------
# CPU Bresenham line rasterization (numba)
# ---------------------------------------------------------------------------

@ngjit
def _burn_lines_cpu(out, r0_arr, c0_arr, r1_arr, c1_arr, vals, height, width):
    """Burn line segments using Bresenham's algorithm."""
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
        dr = dr * sr  # abs
        dc = dc * sc  # abs

        if dr >= dc:
            # Row-major iteration
            err = dc - dr
            r = r0
            c = c0
            for _ in range(dr + 1):
                if 0 <= r < height and 0 <= c < width:
                    out[r, c] = val
                if err >= 0:
                    c += sc
                    err -= dr
                r += sr
                err += dc
        else:
            # Column-major iteration
            err = dr - dc
            r = r0
            c = c0
            for _ in range(dc + 1):
                if 0 <= r < height and 0 <= c < width:
                    out[r, c] = val
                if err >= 0:
                    r += sr
                    err -= dc
                c += sc
                err += dr


# ---------------------------------------------------------------------------
# CPU scanline fill (numba)
# ---------------------------------------------------------------------------

@ngjit
def _scanline_fill_cpu(out, edge_y_min, edge_y_max, edge_x_at_ymin,
                       edge_inv_slope, edge_value, height, width):
    """Fill output raster row by row using scanline algorithm.

    For each row, collect x-intersections from all edges that span
    the row, group by polygon value, sort, and fill between pairs.
    Last-writer-wins when polygons overlap.
    """
    n_edges = len(edge_y_min)
    # Temporary arrays sized to max possible intersections per row
    xs = np.empty(n_edges, dtype=np.float64)
    vs = np.empty(n_edges, dtype=np.float64)

    for row in range(height):
        # Collect intersections for this row
        count = 0
        for e in range(n_edges):
            if edge_y_min[e] <= row <= edge_y_max[e]:
                x = edge_x_at_ymin[e] + (row - edge_y_min[e]) * edge_inv_slope[e]
                xs[count] = x
                vs[count] = edge_value[e]
                count += 1

        if count == 0:
            continue

        # Group by value: get unique values
        # Simple insertion sort of (value, x) pairs
        for i in range(1, count):
            kx = xs[i]
            kv = vs[i]
            j = i - 1
            while j >= 0 and (vs[j] > kv or (vs[j] == kv and xs[j] > kx)):
                xs[j + 1] = xs[j]
                vs[j + 1] = vs[j]
                j -= 1
            xs[j + 1] = kx
            vs[j + 1] = kv

        # Fill between pairs of intersections for each value
        i = 0
        while i < count - 1:
            val = vs[i]
            # Find extent of this value
            j = i
            while j < count and vs[j] == val:
                j += 1
            # Fill pairs within this value group
            k = i
            while k + 1 < j:
                x_start = xs[k]
                x_end = xs[k + 1]
                col_start = max(int(np.ceil(x_start)), 0)
                col_end = min(int(np.floor(x_end)), width - 1)
                for c in range(col_start, col_end + 1):
                    out[row, c] = val
                k += 2
            i = j


def _run_numpy(geometries, values, bounds, height, width, fill, dtype,
               all_touched):
    """NumPy backend for rasterize."""
    out = np.full((height, width), fill, dtype=np.float64)

    # 1. Polygons (scanline fill)
    if all_touched:
        edge_arrays = _extract_edges_all_touched(
            geometries, values, bounds, height, width)
    else:
        edge_arrays = _extract_edges(
            geometries, values, bounds, height, width)
    edge_y_min, edge_y_max, edge_x_at_ymin, edge_inv_slope, edge_value = \
        edge_arrays
    if len(edge_y_min) > 0:
        _scanline_fill_cpu(out, edge_y_min, edge_y_max, edge_x_at_ymin,
                           edge_inv_slope, edge_value, height, width)

    # 2. Lines (Bresenham)
    r0, c0, r1, c1, lvals = _extract_line_segments(
        geometries, values, bounds, height, width)
    if len(r0) > 0:
        _burn_lines_cpu(out, r0, c0, r1, c1, lvals, height, width)

    # 3. Points (direct burn)
    prows, pcols, pvals = _extract_points(
        geometries, values, bounds, height, width)
    if len(prows) > 0:
        _burn_points_cpu(out, prows, pcols, pvals)

    return out.astype(dtype)


# ---------------------------------------------------------------------------
# GPU scanline fill (cupy + numba.cuda)
# ---------------------------------------------------------------------------

@cuda.jit
def _scanline_fill_gpu(out, edge_y_min, edge_y_max, edge_x_at_ymin,
                       edge_inv_slope, edge_value, n_edges, width):
    """CUDA kernel: one thread per raster row.

    Each thread walks the edge table, collects intersections for its row,
    sorts them by (value, x), and fills between pairs.
    """
    row = cuda.grid(1)
    if row >= out.shape[0]:
        return

    # Count intersections for this row
    count = 0
    for e in range(n_edges):
        if edge_y_min[e] <= row <= edge_y_max[e]:
            count += 1

    if count == 0:
        return

    # Allocate local arrays via dynamic shared memory is not practical,
    # so we use a fixed-size local buffer.  For most real-world polygons
    # the number of edges crossing any single row is small.
    # We use device-side allocation via cuda.local.array.
    MAX_ISECT = 512
    if count > MAX_ISECT:
        count = MAX_ISECT  # truncate (safety)

    xs = cuda.local.array(512, dtype=np.float64)
    vs = cuda.local.array(512, dtype=np.float64)

    idx = 0
    for e in range(n_edges):
        if idx >= MAX_ISECT:
            break
        if edge_y_min[e] <= row <= edge_y_max[e]:
            x = edge_x_at_ymin[e] + (row - edge_y_min[e]) * edge_inv_slope[e]
            xs[idx] = x
            vs[idx] = edge_value[e]
            idx += 1

    actual = idx

    # Insertion sort by (value, x)
    for i in range(1, actual):
        kx = xs[i]
        kv = vs[i]
        j = i - 1
        while j >= 0 and (vs[j] > kv or (vs[j] == kv and xs[j] > kx)):
            xs[j + 1] = xs[j]
            vs[j + 1] = vs[j]
            j -= 1
        xs[j + 1] = kx
        vs[j + 1] = kv

    # Fill between pairs per value
    i = 0
    while i < actual - 1:
        val = vs[i]
        j = i
        while j < actual and vs[j] == val:
            j += 1
        k = i
        while k + 1 < j:
            x_start = xs[k]
            x_end = xs[k + 1]
            col_start = int(x_start + 0.999999)  # ceil without math import
            if col_start < 0:
                col_start = 0
            col_end = int(x_end)  # floor
            if col_end >= width:
                col_end = width - 1
            for c in range(col_start, col_end + 1):
                out[row, c] = val
            k += 2
        i = j


@cuda.jit
def _burn_points_gpu(out, rows, cols, vals, n_points):
    """CUDA kernel: one thread per point."""
    i = cuda.grid(1)
    if i >= n_points:
        return
    r = rows[i]
    c = cols[i]
    if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
        out[r, c] = vals[i]


@cuda.jit
def _burn_lines_gpu(out, r0_arr, c0_arr, r1_arr, c1_arr, vals,
                    n_segs, height, width):
    """CUDA kernel: one thread per line segment, Bresenham."""
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
        r = r0
        c = c0
        for _ in range(dr + 1):
            if 0 <= r < height and 0 <= c < width:
                out[r, c] = val
            if err >= 0:
                c += sc
                err -= dr
            r += sr
            err += dc
    else:
        err = dr - dc
        r = r0
        c = c0
        for _ in range(dc + 1):
            if 0 <= r < height and 0 <= c < width:
                out[r, c] = val
            if err >= 0:
                r += sr
                err -= dc
            c += sc
            err += dr


def _run_cupy(geometries, values, bounds, height, width, fill, dtype,
              all_touched):
    """CuPy backend for rasterize."""
    out = cupy.full((height, width), fill, dtype=cupy.float64)

    # 1. Polygons (scanline fill)
    if all_touched:
        edge_arrays = _extract_edges_all_touched(
            geometries, values, bounds, height, width)
    else:
        edge_arrays = _extract_edges(
            geometries, values, bounds, height, width)
    edge_y_min, edge_y_max, edge_x_at_ymin, edge_inv_slope, edge_value = \
        edge_arrays

    if len(edge_y_min) > 0:
        d_y_min = cupy.asarray(edge_y_min)
        d_y_max = cupy.asarray(edge_y_max)
        d_x_at_ymin = cupy.asarray(edge_x_at_ymin)
        d_inv_slope = cupy.asarray(edge_inv_slope)
        d_value = cupy.asarray(edge_value)

        threads_per_block = 256
        blocks = (height + threads_per_block - 1) // threads_per_block
        _scanline_fill_gpu[blocks, threads_per_block](
            out, d_y_min, d_y_max, d_x_at_ymin, d_inv_slope, d_value,
            len(edge_y_min), width)

    # 2. Lines (Bresenham)
    r0, c0, r1, c1, lvals = _extract_line_segments(
        geometries, values, bounds, height, width)
    if len(r0) > 0:
        n_segs = len(r0)
        tpb = 256
        bpg = (n_segs + tpb - 1) // tpb
        _burn_lines_gpu[bpg, tpb](
            out, cupy.asarray(r0), cupy.asarray(c0),
            cupy.asarray(r1), cupy.asarray(c1),
            cupy.asarray(lvals), n_segs, height, width)

    # 3. Points (direct burn)
    prows, pcols, pvals = _extract_points(
        geometries, values, bounds, height, width)
    if len(prows) > 0:
        n_pts = len(prows)
        tpb = 256
        bpg = (n_pts + tpb - 1) // tpb
        _burn_points_gpu[bpg, tpb](
            out, cupy.asarray(prows), cupy.asarray(pcols),
            cupy.asarray(pvals), n_pts)

    return out.astype(dtype)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _parse_input(geometries, column=None):
    """Normalise input to (geometry_list, value_list, bounds).

    Accepts:
    - GeoDataFrame with a ``column`` attribute to burn
    - List of (geometry, value) pairs
    """
    try:
        import geopandas as gpd
        if isinstance(geometries, gpd.GeoDataFrame):
            if column is None:
                # Use first numeric column
                numeric_cols = geometries.select_dtypes(
                    include='number').columns
                if len(numeric_cols) == 0:
                    raise ValueError(
                        "GeoDataFrame has no numeric columns to burn. "
                        "Pass a 'column' name explicitly.")
                column = numeric_cols[0]
            geom_list = geometries.geometry.tolist()
            value_list = geometries[column].values.astype(np.float64).tolist()
            total_bounds = geometries.total_bounds  # (xmin, ymin, xmax, ymax)
            return geom_list, value_list, tuple(total_bounds)
    except ImportError:
        pass

    # Assume list of (geometry, value) pairs
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

    # Compute bounds from geometries
    all_bounds = np.array([g.bounds for g in geom_list if g is not None])
    if len(all_bounds) == 0:
        return geom_list, value_list, None
    total_bounds = (all_bounds[:, 0].min(), all_bounds[:, 1].min(),
                    all_bounds[:, 2].max(), all_bounds[:, 3].max())
    return geom_list, value_list, total_bounds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rasterize(
    geometries,
    width: int,
    height: int,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    column: Optional[str] = None,
    fill: float = np.nan,
    dtype: np.dtype = np.float64,
    all_touched: bool = False,
    use_cuda: bool = False,
    name: str = 'rasterize',
) -> xr.DataArray:
    """Rasterize vector geometries into a 2D DataArray.

    Converts geometries from a GeoDataFrame or a list of
    ``(geometry, value)`` pairs into a regularly gridded raster.
    No GDAL dependency.

    Supported geometry types:

    - **Polygon / MultiPolygon** -- scanline fill
    - **LineString / MultiLineString** -- Bresenham line rasterization
    - **Point / MultiPoint** -- direct pixel burn

    Parameters
    ----------
    geometries : GeoDataFrame or iterable of (geometry, value)
        Input vector data.  If a GeoDataFrame, the ``column`` parameter
        selects which attribute to burn into the raster (defaults to the
        first numeric column).  If an iterable, each element must be a
        ``(shapely.geometry, numeric_value)`` pair.
    width : int
        Number of columns in the output raster.
    height : int
        Number of rows in the output raster.
    bounds : tuple of (xmin, ymin, xmax, ymax), optional
        Geographic extent of the output raster.  Inferred from the
        geometries if omitted.
    column : str, optional
        Name of the GeoDataFrame column whose values are burned into
        the raster.  Ignored when ``geometries`` is a list of pairs.
    fill : float, default np.nan
        Value for pixels not covered by any geometry.
    dtype : numpy dtype, default np.float64
        Data type of the output array.
    all_touched : bool, default False
        If True, all pixels touched by a geometry are burned, not just
        those whose center falls inside a polygon.
    use_cuda : bool, default False
        If True, use the CuPy/CUDA backend.  The output DataArray will
        be backed by a cupy array (stays on device for downstream ops).
    name : str, default 'rasterize'
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
        2D raster with dims ``('y', 'x')`` and coordinates derived from
        ``bounds`` and resolution.

    Notes
    -----
    Geometry types are burned in priority order: polygons first, then
    lines, then points.  Within each type, last-writer-wins when
    geometries overlap.  Unsupported geometry types (e.g.
    GeometryCollection) are silently skipped.

    The GPU path launches one CUDA thread per raster row (polygons),
    per line segment, or per point.  All data is transferred to device
    once and the output stays on device for downstream ops.

    Dependencies: ``shapely`` (vertex extraction), ``geopandas``
    (if passing a GeoDataFrame).  No GDAL or rasterio.

    Examples
    --------
    .. sourcecode:: python

        >>> import geopandas as gpd
        >>> from shapely.geometry import box
        >>> gdf = gpd.GeoDataFrame(
        ...     {'value': [1.0, 2.0]},
        ...     geometry=[box(0, 0, 5, 5), box(3, 3, 8, 8)])
        >>> from xrspatial.rasterize import rasterize
        >>> result = rasterize(gdf, width=10, height=10, column='value')
        >>> result.shape
        (10, 10)
    """
    if width < 1 or height < 1:
        raise ValueError(
            f"width and height must be >= 1, got width={width}, "
            f"height={height}")

    geom_list, value_list, inferred_bounds = _parse_input(geometries, column)

    if bounds is None:
        bounds = inferred_bounds
    if bounds is None:
        raise ValueError(
            "bounds must be provided when geometries are empty or have "
            "no spatial extent")

    xmin, ymin, xmax, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            f"Invalid bounds: xmin ({xmin}) must be < xmax ({xmax}) and "
            f"ymin ({ymin}) must be < ymax ({ymax})")

    if use_cuda:
        if cupy is None:
            raise ImportError(
                "CuPy is required for use_cuda=True but is not installed")
        out = _run_cupy(geom_list, value_list, bounds, height, width,
                        fill, dtype, all_touched)
    else:
        out = _run_numpy(geom_list, value_list, bounds, height, width,
                         fill, dtype, all_touched)

    # Build coordinates
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height
    x_coords = np.linspace(xmin + px / 2, xmax - px / 2, width)
    # y goes from top (ymax) to bottom (ymin) -- row 0 is at ymax
    y_coords = np.linspace(ymax - py / 2, ymin + py / 2, height)

    return xr.DataArray(
        out,
        name=name,
        dims=['y', 'x'],
        coords={'y': y_coords, 'x': x_coords},
    )
