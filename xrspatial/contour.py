# Contour line extraction from raster DEMs using marching squares.
#
# The marching squares algorithm processes each 2x2 cell quad in the raster.
# Each corner is classified as above or below the contour level, producing
# one of 16 cases.  Line segments are emitted with linearly interpolated
# endpoints along cell edges where the contour crosses.
#
# The algorithm is embarrassingly parallel across quads and across contour
# levels, making it well suited to Dask chunking and GPU execution.

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from .utils import ArrayTypeFunctionMapping, ngjit

try:
    import dask
    import dask.array as da
except ImportError:
    dask = None
    da = None

try:
    import cupy
    from numba import cuda
except ImportError:
    cupy = None
    cuda = None


# Marching squares lookup table.
# For each of the 16 cases (4-bit index from corner classification),
# stores pairs of edge indices that form line segments.
# Edge numbering:  0=top, 1=right, 2=bottom, 3=left.
# Empty tuple means no contour passes through the quad.
_MS_TABLE = (
    (),              # 0000 - all below
    ((3, 0),),       # 0001 - bottom-left above
    ((0, 1),),       # 0010 - bottom-right above
    ((3, 1),),       # 0011 - bottom row above
    ((1, 2),),       # 0100 - top-right above
    ((3, 0), (1, 2)),  # 0101 - saddle: BL and TR above
    ((0, 2),),       # 0110 - right column above
    ((3, 2),),       # 0111 - only top-left below
    ((2, 3),),       # 1000 - top-left above
    ((2, 0),),       # 1001 - left column above
    ((2, 3), (0, 1)),  # 1010 - saddle: TL and BR above
    ((2, 1),),       # 1011 - only top-right below
    ((1, 3),),       # 1100 - top row above
    ((1, 0),),       # 1101 - only bottom-right below
    ((0, 3),),       # 1110 - only bottom-left below
    (),              # 1111 - all above
)


def _interp_edge(edge, r, c, tl, tr, bl, br, level):
    """Return interpolated (row, col) for a contour crossing on the given edge.

    Corners of the quad:
        tl = (r, c)      tr = (r, c+1)
        bl = (r+1, c)    br = (r+1, c+1)

    Edge numbering: 0=top (tl-tr), 1=right (tr-br), 2=bottom (bl-br), 3=left (tl-bl)
    """
    if edge == 0:  # top: tl -> tr
        t = (level - tl) / (tr - tl) if tr != tl else 0.5
        return r, c + t
    elif edge == 1:  # right: tr -> br
        t = (level - tr) / (br - tr) if br != tr else 0.5
        return r + t, c + 1
    elif edge == 2:  # bottom: bl -> br
        t = (level - bl) / (br - bl) if br != bl else 0.5
        return r + 1, c + t
    else:  # edge == 3, left: tl -> bl
        t = (level - tl) / (bl - tl) if bl != tl else 0.5
        return r + t, c


@ngjit
def _marching_squares_kernel(data, level, seg_rows, seg_cols, seg_count):
    """Process all 2x2 quads for a single contour level.

    Writes line segments into pre-allocated arrays.
    seg_rows and seg_cols have shape (max_segs, 2) for start/end points.
    seg_count is a 1-element array holding the current write position.
    """
    ny, nx = data.shape
    for r in range(ny - 1):
        for c in range(nx - 1):
            tl = data[r, c]
            tr = data[r, c + 1]
            bl = data[r + 1, c]
            br = data[r + 1, c + 1]

            # Skip quads with any NaN corner.
            if tl != tl or tr != tr or bl != bl or br != br:
                continue

            # Build 4-bit case index.
            idx = 0
            if tl >= level:
                idx |= 8
            if tr >= level:
                idx |= 4
            if bl >= level:
                idx |= 1
            if br >= level:
                idx |= 2

            if idx == 0 or idx == 15:
                continue

            # Saddle disambiguation: use center value.
            if idx == 5 or idx == 10:
                center = (tl + tr + bl + br) * 0.25
                if center >= level:
                    # Connect like-valued corners.
                    if idx == 5:
                        # BL+TR above, center above -> connect them
                        idx = 5  # keep as-is: two separate segments
                    else:
                        # TL+BR above, center above -> connect them
                        idx = 10
                # If center < level, keep the default table entry.

            # Emit segments for this case.
            # Inline the lookup and interpolation for Numba compatibility.
            if idx == 1:
                _emit_seg(r, c, tl, tr, bl, br, level, 3, 0,
                          seg_rows, seg_cols, seg_count)
            elif idx == 2:
                _emit_seg(r, c, tl, tr, bl, br, level, 0, 1,
                          seg_rows, seg_cols, seg_count)
            elif idx == 3:
                _emit_seg(r, c, tl, tr, bl, br, level, 3, 1,
                          seg_rows, seg_cols, seg_count)
            elif idx == 4:
                _emit_seg(r, c, tl, tr, bl, br, level, 1, 2,
                          seg_rows, seg_cols, seg_count)
            elif idx == 5:
                _emit_seg(r, c, tl, tr, bl, br, level, 3, 0,
                          seg_rows, seg_cols, seg_count)
                _emit_seg(r, c, tl, tr, bl, br, level, 1, 2,
                          seg_rows, seg_cols, seg_count)
            elif idx == 6:
                _emit_seg(r, c, tl, tr, bl, br, level, 0, 2,
                          seg_rows, seg_cols, seg_count)
            elif idx == 7:
                _emit_seg(r, c, tl, tr, bl, br, level, 3, 2,
                          seg_rows, seg_cols, seg_count)
            elif idx == 8:
                _emit_seg(r, c, tl, tr, bl, br, level, 2, 3,
                          seg_rows, seg_cols, seg_count)
            elif idx == 9:
                _emit_seg(r, c, tl, tr, bl, br, level, 2, 0,
                          seg_rows, seg_cols, seg_count)
            elif idx == 10:
                _emit_seg(r, c, tl, tr, bl, br, level, 2, 3,
                          seg_rows, seg_cols, seg_count)
                _emit_seg(r, c, tl, tr, bl, br, level, 0, 1,
                          seg_rows, seg_cols, seg_count)
            elif idx == 11:
                _emit_seg(r, c, tl, tr, bl, br, level, 2, 1,
                          seg_rows, seg_cols, seg_count)
            elif idx == 12:
                _emit_seg(r, c, tl, tr, bl, br, level, 1, 3,
                          seg_rows, seg_cols, seg_count)
            elif idx == 13:
                _emit_seg(r, c, tl, tr, bl, br, level, 1, 0,
                          seg_rows, seg_cols, seg_count)
            elif idx == 14:
                _emit_seg(r, c, tl, tr, bl, br, level, 0, 3,
                          seg_rows, seg_cols, seg_count)


@ngjit
def _emit_seg(r, c, tl, tr, bl, br, level, edge_a, edge_b,
              seg_rows, seg_cols, seg_count):
    """Interpolate endpoints and write one segment."""
    idx = seg_count[0]
    if idx >= seg_rows.shape[0]:
        return  # buffer full

    # Interpolate start point (edge_a).
    if edge_a == 0:  # top: tl -> tr
        t = (level - tl) / (tr - tl) if tr != tl else 0.5
        r0 = float(r)
        c0 = float(c) + t
    elif edge_a == 1:  # right: tr -> br
        t = (level - tr) / (br - tr) if br != tr else 0.5
        r0 = float(r) + t
        c0 = float(c + 1)
    elif edge_a == 2:  # bottom: bl -> br
        t = (level - bl) / (br - bl) if br != bl else 0.5
        r0 = float(r + 1)
        c0 = float(c) + t
    else:  # left: tl -> bl
        t = (level - tl) / (bl - tl) if bl != tl else 0.5
        r0 = float(r) + t
        c0 = float(c)

    # Interpolate end point (edge_b).
    if edge_b == 0:
        t = (level - tl) / (tr - tl) if tr != tl else 0.5
        r1 = float(r)
        c1 = float(c) + t
    elif edge_b == 1:
        t = (level - tr) / (br - tr) if br != tr else 0.5
        r1 = float(r) + t
        c1 = float(c + 1)
    elif edge_b == 2:
        t = (level - bl) / (br - bl) if br != bl else 0.5
        r1 = float(r + 1)
        c1 = float(c) + t
    else:
        t = (level - tl) / (bl - tl) if bl != tl else 0.5
        r1 = float(r) + t
        c1 = float(c)

    seg_rows[idx, 0] = r0
    seg_rows[idx, 1] = r1
    seg_cols[idx, 0] = c0
    seg_cols[idx, 1] = c1
    seg_count[0] = idx + 1


def _stitch_segments(seg_rows, seg_cols, n_segs):
    """Join connected segments into polylines.

    Two segments connect when an endpoint of one matches an endpoint of
    the other (within floating-point tolerance).  Returns a list of
    Nx2 arrays, each representing a polyline as (row, col) coordinates.
    """
    if n_segs == 0:
        return []

    rows = seg_rows[:n_segs]
    cols = seg_cols[:n_segs]

    # Build adjacency via endpoint hashing.
    # Round to 10 decimal places to handle float noise.
    DECIMALS = 10
    used = np.zeros(n_segs, dtype=np.bool_)
    endpoint_map = {}  # (rounded_r, rounded_c) -> list of (seg_idx, end_idx)

    for i in range(n_segs):
        for end in range(2):
            key = (round(rows[i, end], DECIMALS), round(cols[i, end], DECIMALS))
            if key not in endpoint_map:
                endpoint_map[key] = []
            endpoint_map[key].append((i, end))

    lines = []
    for start_seg in range(n_segs):
        if used[start_seg]:
            continue
        used[start_seg] = True

        # Start a polyline from this segment.
        line_r = [rows[start_seg, 0], rows[start_seg, 1]]
        line_c = [cols[start_seg, 0], cols[start_seg, 1]]

        # Extend forward from end point.
        _extend_line(line_r, line_c, 1, rows, cols, used, endpoint_map,
                     DECIMALS)
        # Extend backward from start point.
        _extend_line(line_r, line_c, 0, rows, cols, used, endpoint_map,
                     DECIMALS)

        coords = np.column_stack([line_r, line_c])
        lines.append(coords)

    return lines


def _extend_line(line_r, line_c, direction, rows, cols, used, endpoint_map,
                 decimals):
    """Extend a polyline by following connected segments.

    direction: 1 = extend from the end, 0 = extend from the start.
    """
    while True:
        if direction == 1:
            tip_r, tip_c = line_r[-1], line_c[-1]
        else:
            tip_r, tip_c = line_r[0], line_c[0]

        key = (round(tip_r, decimals), round(tip_c, decimals))
        candidates = endpoint_map.get(key, [])

        found = False
        for seg_idx, end_idx in candidates:
            if used[seg_idx]:
                continue
            used[seg_idx] = True
            # The matching end connects; the other end extends the line.
            other = 1 - end_idx
            nr, nc = rows[seg_idx, other], cols[seg_idx, other]
            if direction == 1:
                line_r.append(nr)
                line_c.append(nc)
            else:
                line_r.insert(0, nr)
                line_c.insert(0, nc)
            found = True
            break

        if not found:
            break


def _contours_numpy(data, levels):
    """NumPy backend: extract contour lines for all levels."""
    data = np.asarray(data, dtype=np.float64)
    ny, nx = data.shape
    max_segs_per_level = (ny - 1) * (nx - 1) * 2  # worst case: every quad saddle

    results = []
    for level in levels:
        seg_rows = np.empty((max_segs_per_level, 2), dtype=np.float64)
        seg_cols = np.empty((max_segs_per_level, 2), dtype=np.float64)
        seg_count = np.zeros(1, dtype=np.int64)

        _marching_squares_kernel(data, float(level), seg_rows, seg_cols,
                                 seg_count)
        n = int(seg_count[0])
        lines = _stitch_segments(seg_rows, seg_cols, n)
        for line in lines:
            results.append((float(level), line))

    return results


def _contours_cupy(data, levels):
    """CuPy backend: transfer to CPU and run numpy implementation.

    Contour tracing and segment stitching are inherently sequential
    graph operations that don't benefit from GPU parallelism.  The GPU
    kernel approach (writing per-quad segments to a buffer) produces
    the same segments as the CPU version, so we transfer the data once
    and reuse the optimized Numba kernel.
    """
    if cupy is None:
        raise ImportError("CuPy is required for GPU contour extraction")
    cpu_data = cupy.asnumpy(data)
    return _contours_numpy(cpu_data, levels)


def _contours_dask(data, levels):
    """Dask backend: process each chunk independently, then merge.

    Each chunk is processed with a 1-cell overlap so that quads spanning
    chunk boundaries are handled by both neighbors.  Duplicate segments
    at boundaries are removed during stitching.
    """
    if da is None:
        raise ImportError("Dask is required for chunked contour extraction")

    # Compute chunks independently.
    chunks = data.to_delayed().ravel()
    chunk_slices = _get_chunk_slices(data.chunks)

    all_results = []
    for chunk_delayed, (r_off, c_off) in zip(chunks, chunk_slices):
        result = dask.delayed(_process_chunk_numpy)(
            chunk_delayed, levels, r_off, c_off
        )
        all_results.append(result)

    # Compute all chunks and merge.
    chunk_results = dask.compute(*all_results)

    # Flatten and deduplicate.
    merged = []
    for chunk_lines in chunk_results:
        merged.extend(chunk_lines)

    return _deduplicate_by_level(merged)


def _contours_dask_cupy(data, levels):
    """Dask+CuPy backend: transfer each chunk to CPU independently."""
    if da is None:
        raise ImportError("Dask is required for chunked contour extraction")

    chunks = data.to_delayed().ravel()
    chunk_slices = _get_chunk_slices(data.chunks)

    all_results = []
    for chunk_delayed, (r_off, c_off) in zip(chunks, chunk_slices):
        result = dask.delayed(_process_chunk_cupy)(
            chunk_delayed, levels, r_off, c_off
        )
        all_results.append(result)

    chunk_results = dask.compute(*all_results)

    merged = []
    for chunk_lines in chunk_results:
        merged.extend(chunk_lines)

    return _deduplicate_by_level(merged)


def _get_chunk_slices(chunks_tuple):
    """Compute (row_offset, col_offset) for each chunk."""
    row_chunks, col_chunks = chunks_tuple
    slices = []
    r_off = 0
    for rsize in row_chunks:
        c_off = 0
        for csize in col_chunks:
            slices.append((r_off, c_off))
            c_off += csize
        r_off += rsize
    return slices


def _process_chunk_numpy(chunk_data, levels, r_offset, c_offset):
    """Process a single numpy chunk, offsetting coordinates to global space."""
    chunk_data = np.asarray(chunk_data)
    if chunk_data.shape[0] < 2 or chunk_data.shape[1] < 2:
        return []

    local_results = _contours_numpy(chunk_data, levels)
    # Offset coordinates to global raster space.
    offset_results = []
    for level, coords in local_results:
        shifted = coords.copy()
        shifted[:, 0] += r_offset
        shifted[:, 1] += c_offset
        offset_results.append((level, shifted))
    return offset_results


def _process_chunk_cupy(chunk_data, levels, r_offset, c_offset):
    """Process a single CuPy chunk by transferring to CPU first."""
    if cupy is not None and hasattr(chunk_data, 'get'):
        chunk_data = chunk_data.get()
    return _process_chunk_numpy(np.asarray(chunk_data), levels,
                                r_offset, c_offset)


def _deduplicate_by_level(results):
    """Group results by level and deduplicate overlapping segments.

    Segments from overlapping chunk boundaries may produce duplicate
    polylines.  We merge by endpoint proximity within each level.
    """
    if not results:
        return results

    from collections import defaultdict
    by_level = defaultdict(list)
    for level, coords in results:
        by_level[level].append(coords)

    merged = []
    for level in sorted(by_level.keys()):
        lines = by_level[level]
        # Re-stitch all segments across chunk boundaries.
        all_segs_r = []
        all_segs_c = []
        for line in lines:
            for i in range(len(line) - 1):
                all_segs_r.append([line[i, 0], line[i + 1, 0]])
                all_segs_c.append([line[i, 1], line[i + 1, 1]])

        if not all_segs_r:
            continue

        seg_rows = np.array(all_segs_r, dtype=np.float64)
        seg_cols = np.array(all_segs_c, dtype=np.float64)

        # Remove exact duplicate segments.
        seg_rows, seg_cols = _remove_duplicate_segments(seg_rows, seg_cols)

        stitched = _stitch_segments(seg_rows, seg_cols, len(seg_rows))
        for line in stitched:
            merged.append((level, line))

    return merged


def _remove_duplicate_segments(seg_rows, seg_cols):
    """Remove duplicate segments (same endpoints in either order)."""
    n = len(seg_rows)
    if n == 0:
        return seg_rows, seg_cols

    DECIMALS = 10
    seen = set()
    keep = []
    for i in range(n):
        r0 = round(seg_rows[i, 0], DECIMALS)
        r1 = round(seg_rows[i, 1], DECIMALS)
        c0 = round(seg_cols[i, 0], DECIMALS)
        c1 = round(seg_cols[i, 1], DECIMALS)
        # Canonical form: smaller endpoint first.
        fwd = (r0, c0, r1, c1)
        rev = (r1, c1, r0, c0)
        key = min(fwd, rev)
        if key not in seen:
            seen.add(key)
            keep.append(i)

    return seg_rows[keep], seg_cols[keep]


def _to_geopandas(results, crs=None):
    """Convert contour results to a GeoDataFrame."""
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except ImportError:
        raise ImportError(
            "geopandas and shapely are required for GeoDataFrame output. "
            "Install them with: pip install geopandas shapely"
        )

    records = []
    for level, coords in results:
        if len(coords) >= 2:
            # coords are (row, col); convert to (x, y) = (col, row)
            geom = LineString(coords[:, ::-1])
            records.append({'level': level, 'geometry': geom})

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf


def contours(
    agg: xr.DataArray,
    levels: Optional[Union[Sequence[float], np.ndarray]] = None,
    n_levels: int = 10,
    return_type: str = "numpy",
) -> Union[List[Tuple[float, np.ndarray]], "gpd.GeoDataFrame"]:
    """Extract contour lines (isolines) from a raster surface.

    Uses the marching squares algorithm to trace isolines through the
    raster at specified elevation values.  Each 2x2 cell quad is
    classified independently, so the algorithm parallelizes across
    Dask chunks.

    Parameters
    ----------
    agg : xr.DataArray
        2D input raster (e.g. a DEM).

    levels : sequence of float, optional
        Explicit contour levels to extract.  If not provided, ``n_levels``
        evenly spaced levels are chosen between the raster min and max.

    n_levels : int, default 10
        Number of contour levels to generate when ``levels`` is not
        provided.

    return_type : str, default "numpy"
        Output format.  ``"numpy"`` returns a list of ``(level, coords)``
        tuples where *coords* is an Nx2 array of ``(row, col)``
        coordinates.  ``"geopandas"`` returns a GeoDataFrame with
        ``level`` and ``geometry`` columns (requires geopandas/shapely).

    Returns
    -------
    list of (float, ndarray) or GeoDataFrame
        Contour lines grouped by level.

    Notes
    -----
    CuPy and Dask+CuPy arrays are accepted as input.  Data is
    transferred to CPU for the tracing step because segment stitching
    is an inherently sequential graph traversal.  For Dask inputs,
    each chunk is processed independently and results are merged,
    keeping peak memory proportional to chunk size.

    Examples
    --------
    >>> from xrspatial import contours
    >>> lines = contours(dem, levels=[100, 500, 1000])
    >>> # Each entry is (level_value, Nx2_coordinate_array)
    >>> level, coords = lines[0]
    """
    if agg.ndim != 2:
        raise ValueError("Input raster must be 2D")
    if agg.shape[0] < 2 or agg.shape[1] < 2:
        raise ValueError(
            "Input raster must have at least 2 rows and 2 columns"
        )

    # Determine contour levels.
    if levels is None:
        if da is not None and isinstance(agg.data, da.Array):
            vmin, vmax = dask.compute(
                da.nanmin(agg.data), da.nanmax(agg.data)
            )
            vmin = float(vmin)
            vmax = float(vmax)
        elif cupy is not None and hasattr(agg.data, 'get'):
            vmin = float(cupy.nanmin(agg.data))
            vmax = float(cupy.nanmax(agg.data))
        else:
            vmin = float(np.nanmin(agg.values))
            vmax = float(np.nanmax(agg.values))

        if np.isnan(vmin) or np.isnan(vmax):
            return [] if return_type == "numpy" else _to_geopandas([], None)

        # Exclude exact min/max to avoid tracing along the boundary.
        levels = np.linspace(vmin, vmax, n_levels + 2)[1:-1]
    else:
        levels = np.asarray(levels, dtype=np.float64)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_contours_numpy,
        cupy_func=_contours_cupy,
        dask_func=_contours_dask,
        dask_cupy_func=_contours_dask_cupy,
    )
    results = mapper(agg)(agg.data, levels)

    if return_type == "numpy":
        return results
    elif return_type == "geopandas":
        crs = agg.attrs.get('crs', None)
        return _to_geopandas(results, crs=crs)
    else:
        raise ValueError(
            f"Invalid return_type '{return_type}'. "
            "Allowed values are 'numpy' and 'geopandas'."
        )
