import warnings
from functools import partial
from math import sqrt

try:
    import dask.array as da
except ImportError:
    da = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

import math as _math

import numpy as np
import xarray as xr
from numba import cuda, prange

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

from xrspatial.pathfinding import _available_memory_bytes
from xrspatial.utils import (
    _validate_raster,
    cuda_args, get_dataarray_resolution, has_cuda_and_cupy,
    is_cupy_array, is_dask_cupy, ngjit,
)
from xrspatial.dataset_support import supports_dataset

EUCLIDEAN = 0
GREAT_CIRCLE = 1
MANHATTAN = 2

PROXIMITY = 0
ALLOCATION = 1
DIRECTION = 2


def _distance_metric_mapping():
    DISTANCE_METRICS = {}
    DISTANCE_METRICS["EUCLIDEAN"] = EUCLIDEAN
    DISTANCE_METRICS["GREAT_CIRCLE"] = GREAT_CIRCLE
    DISTANCE_METRICS["MANHATTAN"] = MANHATTAN

    return DISTANCE_METRICS


# create dictionary to map distance metric presented by string and the
# corresponding metric presented by integer.
DISTANCE_METRICS = _distance_metric_mapping()


@ngjit
def euclidean_distance(x1: float, x2: float, y1: float, y2: float) -> float:
    """
    Calculates Euclidean (straight line) distance between (x1, y1) and
    (x2, y2).

    Parameters
    ----------
    x1 : float
        x-coordinate of the first point.
    x2 : float
        x-coordinate of the second point.
    y1 : float
        y-coordinate of the first point.
    y2 : float
        y-coordinate of the second point.

    Returns
    -------
    distance : float
        Euclidean distance between two points.

    References
    ----------
        - Wikipedia: https://en.wikipedia.org/wiki/Euclidean_distance#:~:text=In%20mathematics%2C%20the%20Euclidean%20distance,being%20called%20the%20Pythagorean%20distance. # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> # Imports
        >>> from xrspatial import euclidean_distance
        >>> point_a = (142.32, 23.23)
        >>> point_b = (312.54, 432.01)
        >>> # Calculate Euclidean Distance
        >>> dist = euclidean_distance(
        ...     point_a[0],
        ...     point_b[0],
        ...     point_a[1],
        ...     point_b[1])
        >>> print(dist)
        442.80462599209596
    """

    x = x1 - x2
    y = y1 - y2
    return np.sqrt(x * x + y * y)


@ngjit
def manhattan_distance(x1: float, x2: float, y1: float, y2: float) -> float:
    """
    Calculates Manhattan distance (sum of distance in x and y directions)
    between (x1, y1) and (x2, y2).

    Parameters
    ----------
    x1 : float
        x-coordinate of the first point.
    x2 : float
        x-coordinate of the second point.
    y1 : float
        y-coordinate of the first point.
    y2 : float
        y-coordinate of the second point.

    Returns
    -------
    distance : float
        Manhattan distance between two points.

    References
    ----------
        - Wikipedia: https://en.wikipedia.org/wiki/Taxicab_geometry

    Examples
    --------
    .. sourcecode:: python

        >>> from xrspatial import manhattan_distance
        >>> point_a = (142.32, 23.23)
        >>> point_b = (312.54, 432.01)
        >>> # Calculate Manhattan Distance
        >>> dist = manhattan_distance(
        ...     point_a[0],
        ...     point_b[0],
        ...     point_a[1],
        ...     point_b[1])
        >>> print(dist)
        579.0
    """

    x = x1 - x2
    y = y1 - y2
    return abs(x) + abs(y)


@ngjit
def great_circle_distance(
    x1: float, x2: float, y1: float, y2: float, radius: float = 6378137
) -> float:
    """
    Calculates great-circle (orthodromic/spherical) distance between
    (x1, y1) and (x2, y2), assuming each point is a longitude,
    latitude pair.

    Parameters
    ----------
    x1 : float
        x-coordinate (longitude) between -180 and 180 of the first point.
    x2: float
        x-coordinate (longitude) between -180 and 180 of the second point.
    y1: float
        y-coordinate (latitude) between -90 and 90 of the first point.
    y2: float
        y-coordinate (latitude) between -90 and 90 of the second point.
    radius: float, default=6378137
        Radius of sphere (earth).

    Returns
    -------
    distance : float
        Great-Circle distance between two points.

    References
    ----------
        - Wikipedia: https://en.wikipedia.org/wiki/Great-circle_distance#:~:text=The%20great%2Dcircle%20distance%2C%20orthodromic,line%20through%20the%20sphere's%20interior). # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> from xrspatial import great_circle_distance
        >>> point_a = (123.2, 82.32)
        >>> point_b = (178.0, 65.09)
        >>> # Calculate Great Circle Distance
        >>> dist = great_circle_distance(
        ...     point_a[0],
        ...     point_b[0],
        ...     point_a[1],
        ...     point_b[1])
        >>> print(dist)
        2378290.489801402
    """

    if x1 > 180 or x1 < -180:
        raise ValueError(
            "Invalid x-coordinate of the first point."
            "Must be in the range [-180, 180]"
        )

    if x2 > 180 or x2 < -180:
        raise ValueError(
            "Invalid x-coordinate of the second point."
            "Must be in the range [-180, 180]"
        )

    if y1 > 90 or y1 < -90:
        raise ValueError(
            "Invalid y-coordinate of the first point."
            "Must be in the range [-90, 90]"
        )

    if y2 > 90 or y2 < -90:
        raise ValueError(
            "Invalid y-coordinate of the second point."
            "Must be in the range [-90, 90]"
        )

    lat1, lon1, lat2, lon2 = (
        np.radians(y1),
        np.radians(x1),
        np.radians(y2),
        np.radians(x2),
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # earth radius: 6378137
    return radius * 2 * np.arcsin(np.sqrt(a))


@ngjit
def _distance(x1, x2, y1, y2, metric):

    if metric == EUCLIDEAN:
        d = euclidean_distance(x1, x2, y1, y2)

    elif metric == GREAT_CIRCLE:
        d = great_circle_distance(x1, x2, y1, y2)

    else:
        # metric == MANHATTAN:
        d = manhattan_distance(x1, x2, y1, y2)

    return np.float32(d)


@ngjit
def _calc_direction(x1, x2, y1, y2):
    # Calculate direction from (x1, y1) to a source cell (x2, y2).
    # The output values are based on compass directions,
    # 90 to the east, 180 to the south, 270 to the west, and 360 to the north,
    # with 0 reserved for the source cell itself

    if x1 == x2 and y1 == y2:
        return 0

    x = x2 - x1
    y = y2 - y1
    d = np.arctan2(-y, x) * 57.29578
    if d < 0:
        d = 90.0 - d
    elif d > 90.0:
        d = 360.0 - d + 90.0
    else:
        d = 90.0 - d

    return np.float32(d)


def _vectorized_calc_direction(x1, x2, y1, y2):
    """Array-based compass direction from (x1, y1) to (x2, y2).

    Uses the same conversion constant (57.29578) as _calc_direction
    to ensure identical floating-point behaviour.
    """
    dx = x2 - x1
    dy = y2 - y1
    d = np.arctan2(-dy, dx) * 57.29578
    result = np.where(d < 0, 90.0 - d,
             np.where(d > 90.0, 360.0 - d + 90.0, 90.0 - d))
    result[(x1 == x2) & (y1 == y2)] = 0.0
    return result.astype(np.float32)


# =====================================================================
# GPU (CuPy / CUDA) backend
# =====================================================================

@cuda.jit(device=True)
def _gpu_euclidean_distance(x1, x2, y1, y2):
    dx = x1 - x2
    dy = y1 - y2
    return _math.sqrt(dx * dx + dy * dy)


@cuda.jit(device=True)
def _gpu_manhattan_distance(x1, x2, y1, y2):
    return abs(x1 - x2) + abs(y1 - y2)


@cuda.jit(device=True)
def _gpu_great_circle_distance(x1, x2, y1, y2):
    if x1 == x2 and y1 == y2:
        return 0.0
    lat1 = y1 * 0.017453292519943295
    lon1 = x1 * 0.017453292519943295
    lat2 = y2 * 0.017453292519943295
    lon2 = x2 * 0.017453292519943295
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (_math.sin(dlat / 2.0) ** 2
         + _math.cos(lat1) * _math.cos(lat2)
         * _math.sin(dlon / 2.0) ** 2)
    return 6378137.0 * 2.0 * _math.asin(_math.sqrt(a))


@cuda.jit(device=True)
def _gpu_distance(x1, x2, y1, y2, metric):
    if metric == EUCLIDEAN:
        return _gpu_euclidean_distance(x1, x2, y1, y2)
    elif metric == GREAT_CIRCLE:
        return _gpu_great_circle_distance(x1, x2, y1, y2)
    else:
        return _gpu_manhattan_distance(x1, x2, y1, y2)


@cuda.jit(device=True)
def _gpu_calc_direction(x1, x2, y1, y2):
    if x1 == x2 and y1 == y2:
        return 0.0
    dx = x2 - x1
    dy = y2 - y1
    d = _math.atan2(-dy, dx) * 57.29578
    if d < 0.0:
        d = 90.0 - d
    elif d > 90.0:
        d = 360.0 - d + 90.0
    else:
        d = 90.0 - d
    return d


@cuda.jit
def _proximity_cuda_kernel(target_xs, target_ys, target_vals, n_targets,
                           y_coords, x_coords, max_distance,
                           distance_metric, process_mode, out):
    iy, ix = cuda.grid(2)
    if iy >= out.shape[0] or ix >= out.shape[1]:
        return

    px = x_coords[ix]
    py = y_coords[iy]

    best_dist = 1.0e38
    best_idx = -1

    for k in range(n_targets):
        d = _gpu_distance(px, target_xs[k], py, target_ys[k], distance_metric)
        if d < best_dist:
            best_dist = d
            best_idx = k

    if best_idx >= 0 and best_dist <= max_distance:
        if process_mode == PROXIMITY:
            out[iy, ix] = best_dist
        elif process_mode == ALLOCATION:
            out[iy, ix] = target_vals[best_idx]
        else:
            out[iy, ix] = _gpu_calc_direction(
                px, target_xs[best_idx], py, target_ys[best_idx])


def _process_cupy(raster_data, x_coords, y_coords, target_values,
                  max_distance, distance_metric, process_mode):
    """GPU proximity using CUDA brute-force nearest-target kernel."""
    import cupy as cp

    # Find target pixels on GPU
    if len(target_values) == 0:
        mask = cp.isfinite(raster_data) & (raster_data != 0)
    else:
        mask = cp.isin(raster_data, cp.asarray(target_values))
        mask &= cp.isfinite(raster_data)

    target_rows, target_cols = cp.where(mask)
    n_targets = int(target_rows.shape[0])

    if n_targets == 0:
        return cp.full(raster_data.shape, cp.nan, dtype=cp.float32)

    # Collect target world-coordinates and values
    y_dev = cp.asarray(y_coords, dtype=cp.float64)
    x_dev = cp.asarray(x_coords, dtype=cp.float64)
    target_ys = y_dev[target_rows]
    target_xs = x_dev[target_cols]
    target_vals = raster_data[target_rows, target_cols].astype(cp.float32)

    # Pre-fill output with NaN (pixels with no target within range stay NaN)
    out = cp.full(raster_data.shape, cp.nan, dtype=cp.float32)

    griddim, blockdim = cuda_args(raster_data.shape)
    _proximity_cuda_kernel[griddim, blockdim](
        target_xs, target_ys, target_vals, n_targets,
        y_dev, x_dev,
        np.float64(max_distance),
        np.int32(distance_metric),
        np.int32(process_mode),
        out,
    )

    return out


def _process_dask_cupy(raster, x_coords, y_coords, target_values,
                       max_distance, distance_metric, process_mode):
    """Dask+CuPy bounded proximity via map_overlap with per-chunk GPU kernel.

    Each chunk (plus overlap padding of ``max_distance / cellsize`` pixels)
    is processed on GPU independently.  Only valid for finite max_distance
    where the padding guarantees all relevant targets are visible within
    each overlapped chunk.
    """
    import cupy as cp

    cellsize_x, cellsize_y = get_dataarray_resolution(raster)
    pad_y = int(max_distance / abs(cellsize_y) + 0.5)
    pad_x = int(max_distance / abs(cellsize_x) + 0.5)

    # Build 2D coordinate grids as dask+cupy arrays matching raster chunks.
    # Each chunk is small (chunk_h x chunk_w x 8 bytes); the full grid is
    # never materialised.
    x_cp = cp.asarray(x_coords, dtype=cp.float64)
    y_cp = cp.asarray(y_coords, dtype=cp.float64)
    x_da = da.from_array(x_cp, chunks=(x_cp.shape[0],))
    y_da = da.from_array(y_cp, chunks=(y_cp.shape[0],))
    xs = da.tile(x_da, (raster.shape[0], 1)).rechunk(raster.data.chunks)
    ys = da.repeat(y_da, raster.shape[1]).reshape(
        raster.shape).rechunk(raster.data.chunks)

    # Capture closure vars for the chunk function
    tv = target_values
    md = max_distance
    dm = distance_metric
    pm = process_mode

    def _chunk_func(data_chunk, xs_chunk, ys_chunk):
        # Use middle row/col to avoid NaN from boundary padding
        x_1d = xs_chunk[xs_chunk.shape[0] // 2, :]
        y_1d = ys_chunk[:, ys_chunk.shape[1] // 2]
        return _process_cupy(data_chunk, x_1d, y_1d, tv, md, dm, pm)

    return da.map_overlap(
        _chunk_func,
        raster.data, xs, ys,
        depth=(pad_y, pad_x),
        boundary=np.nan,
        meta=cp.array((), dtype=cp.float32),
    )


@ngjit
def _process_proximity_line(
    source_line,
    xs,
    ys,
    pan_near_x,
    pan_near_y,
    is_forward,
    line_id,
    width,
    max_distance,
    line_proximity,
    nearest_xs,
    nearest_ys,
    values,
    distance_metric,
):
    """
    Process proximity for a line of pixels in an image.

    Parameters
    ----------
    source_line : numpy.array
        Input data.
    pan_near_x : numpy.array
    pan_near_y : numpy.array
    is_forward : boolean
        Will we loop forward through pixel.
    line_id : np.int64
        Index of the source_line in the image.
    width : np.int64
        Image width.
        It is the number of pixels in the `source_line`.
    max_distance : np.float32, maximum distance considered.
    line_proximity : numpy.array
        1d numpy array of type np.float32, calculated proximity from
        source_line.
    values : numpy.array
        1d numpy array. A list of target pixel values
        to measure the distance from. If this option is not provided
        proximity will be computed from non-zero pixel values.

    Returns
    -------
    self: numpy.array
        1d numpy array of type np.float32. Corresponding proximity of
        source_line.
    """
    start = width - 1
    end = -1
    step = -1
    if is_forward:
        start = 0
        end = width
        step = 1

    n_values = len(values)
    for pixel in prange(start, end, step):
        is_target = False
        # Is the current pixel a target pixel?
        if n_values == 0:
            if source_line[pixel] != 0 and np.isfinite(source_line[pixel]):
                is_target = True
        else:
            for i in prange(n_values):
                if source_line[pixel] == values[i]:
                    is_target = True

        if is_target:
            line_proximity[pixel] = 0.0
            nearest_xs[pixel] = pixel
            nearest_ys[pixel] = line_id
            pan_near_x[pixel] = pixel
            pan_near_y[pixel] = line_id
            continue

        # Are we near(er) to the closest target to the above (below) pixel?
        near_distance_square = max_distance ** 2 * 2.0
        if pan_near_x[pixel] != -1:
            # distance_square
            x1 = xs[pan_near_y[pixel], pan_near_x[pixel]]
            y1 = ys[pan_near_y[pixel], pan_near_x[pixel]]
            x2 = xs[line_id, pixel]
            y2 = ys[line_id, pixel]

            dist = _distance(x1, x2, y1, y2, distance_metric)
            dist_sqr = dist ** 2
            if dist_sqr < near_distance_square:
                near_distance_square = dist_sqr
            else:
                pan_near_x[pixel] = -1
                pan_near_y[pixel] = -1

        # Are we near(er) to the closest target to the left (right) pixel?
        last = pixel - step
        if pixel != start and pan_near_x[last] != -1:
            x1 = xs[pan_near_y[last], pan_near_x[last]]
            y1 = ys[pan_near_y[last], pan_near_x[last]]
            x2 = xs[line_id, pixel]
            y2 = ys[line_id, pixel]

            dist = _distance(x1, x2, y1, y2, distance_metric)
            dist_sqr = dist ** 2
            if dist_sqr < near_distance_square:
                near_distance_square = dist_sqr
                pan_near_x[pixel] = pan_near_x[last]
                pan_near_y[pixel] = pan_near_y[last]

        #  Are we near(er) to the closest target to the
        #  topright (bottom left) pixel?
        tr = pixel + step
        if tr != end and pan_near_x[tr] != -1:
            x1 = xs[pan_near_y[tr], pan_near_x[tr]]
            y1 = ys[pan_near_y[tr], pan_near_x[tr]]
            x2 = xs[line_id, pixel]
            y2 = ys[line_id, pixel]

            dist = _distance(x1, x2, y1, y2, distance_metric)
            dist_sqr = dist ** 2
            if dist_sqr < near_distance_square:
                near_distance_square = dist_sqr
                pan_near_x[pixel] = pan_near_x[tr]
                pan_near_y[pixel] = pan_near_y[tr]

        # Update our proximity value.
        if (
            pan_near_x[pixel] != -1
            and max_distance * max_distance >= near_distance_square
            and (
                line_proximity[pixel] < 0
                or near_distance_square < line_proximity[pixel]
                * line_proximity[pixel]
            )
        ):
            line_proximity[pixel] = sqrt(near_distance_square)
            nearest_xs[pixel] = pan_near_x[pixel]
            nearest_ys[pixel] = pan_near_y[pixel]
    return


def _kdtree_chunk_fn(block, y_coords_1d, x_coords_1d,
                     tree, block_info, max_distance, p,
                     process_mode, target_vals, target_coords):
    """Query k-d tree for nearest target for every pixel in block."""
    if block_info is None or block_info == []:
        return np.full(block.shape, np.nan, dtype=np.float32)

    y_start = block_info[0]['array-location'][0][0]
    x_start = block_info[0]['array-location'][1][0]
    h, w = block.shape

    chunk_ys = y_coords_1d[y_start:y_start + h]
    chunk_xs = x_coords_1d[x_start:x_start + w]
    yy, xx = np.meshgrid(chunk_ys, chunk_xs, indexing='ij')
    query_pts = np.column_stack([yy.ravel(), xx.ravel()])

    dists, indices = tree.query(query_pts, p=p,
                                distance_upper_bound=max_distance)

    n_targets = len(target_vals)
    oob = indices >= n_targets
    safe_idx = np.where(oob, 0, indices)

    if process_mode == PROXIMITY:
        result = dists.astype(np.float32)
        result[result == np.inf] = np.nan
    elif process_mode == ALLOCATION:
        result = target_vals[safe_idx].astype(np.float32)
        result[oob] = np.nan
    else:  # DIRECTION
        query_x = xx.ravel()
        query_y = yy.ravel()
        target_x = target_coords[safe_idx, 1]
        target_y = target_coords[safe_idx, 0]
        result = _vectorized_calc_direction(
            query_x, target_x, query_y, target_y)
        result[oob] = np.nan
        result[dists == 0] = 0.0

    return result.reshape(h, w)


def _target_mask(chunk_data, target_values):
    """Boolean mask of target pixels in *chunk_data*."""
    if len(target_values) == 0:
        return np.isfinite(chunk_data) & (chunk_data != 0)
    return np.isin(chunk_data, target_values) & np.isfinite(chunk_data)


def _stream_target_counts(raster, target_values, y_coords, x_coords,
                          chunks_y, chunks_x):
    """Stream all dask chunks, counting targets per chunk.

    Caches per-chunk coordinate arrays and pixel values within a 25%
    memory budget to reduce re-reads in later phases.

    Returns
    -------
    target_counts : ndarray, shape (n_tile_y, n_tile_x), dtype int64
    total_targets : int
    coords_cache : dict  (iy, ix) -> ndarray shape (N, 2)
    values_cache : dict  (iy, ix) -> ndarray shape (N,), dtype float32
    """
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    target_counts = np.zeros((n_tile_y, n_tile_x), dtype=np.int64)
    coords_cache = {}
    values_cache = {}
    cache_bytes = 0
    budget = int(0.25 * _available_memory_bytes())

    y_offsets = np.zeros(n_tile_y + 1, dtype=np.int64)
    np.cumsum(chunks_y, out=y_offsets[1:])
    x_offsets = np.zeros(n_tile_x + 1, dtype=np.int64)
    np.cumsum(chunks_x, out=x_offsets[1:])

    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            chunk_data = raster.data.blocks[iy, ix].compute()
            mask = _target_mask(chunk_data, target_values)
            rows, cols = np.where(mask)
            n = len(rows)
            target_counts[iy, ix] = n
            if n > 0:
                coords = np.column_stack([
                    y_coords[y_offsets[iy] + rows],
                    x_coords[x_offsets[ix] + cols],
                ])
                vals = chunk_data[rows, cols].astype(np.float32)
                entry_bytes = coords.nbytes + vals.nbytes
                if cache_bytes + entry_bytes <= budget:
                    coords_cache[(iy, ix)] = coords
                    values_cache[(iy, ix)] = vals
                    cache_bytes += entry_bytes

    total_targets = int(target_counts.sum())
    return target_counts, total_targets, coords_cache, values_cache


def _chunk_offsets(chunks):
    """Return cumulative offset array of length len(chunks)+1."""
    offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    np.cumsum(chunks, out=offsets[1:])
    return offsets


def _collect_region_targets(raster, jy_lo, jy_hi, jx_lo, jx_hi,
                            target_values, target_counts,
                            y_coords, x_coords,
                            y_offsets, x_offsets,
                            coords_cache, values_cache):
    """Collect target (y, x) coords and pixel values from chunks.

    Uses cache where available, re-reads uncached chunks via .compute().
    Returns (coords ndarray (N, 2), vals ndarray (N,)) or (None, None).
    """
    coord_parts = []
    val_parts = []
    for iy in range(jy_lo, jy_hi):
        for ix in range(jx_lo, jx_hi):
            if target_counts[iy, ix] == 0:
                continue
            if (iy, ix) in coords_cache:
                coord_parts.append(coords_cache[(iy, ix)])
                val_parts.append(values_cache[(iy, ix)])
            else:
                chunk_data = raster.data.blocks[iy, ix].compute()
                mask = _target_mask(chunk_data, target_values)
                rows, cols = np.where(mask)
                if len(rows) > 0:
                    coords = np.column_stack([
                        y_coords[y_offsets[iy] + rows],
                        x_coords[x_offsets[ix] + cols],
                    ])
                    coord_parts.append(coords)
                    val_parts.append(
                        chunk_data[rows, cols].astype(np.float32)
                    )
    if not coord_parts:
        return None, None
    return np.concatenate(coord_parts), np.concatenate(val_parts)


def _min_boundary_distance(iy, ix, y_coords, x_coords,
                           y_offsets, x_offsets,
                           jy_lo, jy_hi, jx_lo, jx_hi,
                           n_tile_y, n_tile_x):
    """Lower bound on distance from any pixel in chunk (iy, ix) to any point
    outside the search region [jy_lo:jy_hi, jx_lo:jx_hi].

    For each of the 4 sides where the search region doesn't reach the raster
    edge, compute the gap between the chunk's edge pixel coordinate and the
    first pixel outside the search region.  The minimum of these gaps is
    a valid lower bound for both L1 and L2 norms.

    Returns float (inf if search covers the full raster).
    """
    gaps = []

    # Top boundary
    if jy_lo > 0:
        # chunk's top-edge row in pixel space
        chunk_top_row = y_offsets[iy]
        # first row outside region (above)
        outside_row = y_offsets[jy_lo] - 1
        gap = abs(float(y_coords[chunk_top_row]) - float(y_coords[outside_row]))
        gaps.append(gap)

    # Bottom boundary
    if jy_hi < n_tile_y:
        chunk_bot_row = y_offsets[iy + 1] - 1
        outside_row = y_offsets[jy_hi]
        gap = abs(float(y_coords[chunk_bot_row]) - float(y_coords[outside_row]))
        gaps.append(gap)

    # Left boundary
    if jx_lo > 0:
        chunk_left_col = x_offsets[ix]
        outside_col = x_offsets[jx_lo] - 1
        gap = abs(float(x_coords[chunk_left_col]) - float(x_coords[outside_col]))
        gaps.append(gap)

    # Right boundary
    if jx_hi < n_tile_x:
        chunk_right_col = x_offsets[ix + 1] - 1
        outside_col = x_offsets[jx_hi]
        gap = abs(float(x_coords[chunk_right_col]) - float(x_coords[outside_col]))
        gaps.append(gap)

    return min(gaps) if gaps else np.inf


def _tiled_chunk_query(raster, iy, ix, y_coords, x_coords,
                       y_offsets, x_offsets,
                       target_values, target_counts,
                       coords_cache, values_cache,
                       max_distance, p,
                       n_tile_y, n_tile_x, process_mode):
    """Expanding-ring local KDTree for one output chunk.

    Returns ndarray shape (h, w), dtype float32.
    """
    h = int(y_offsets[iy + 1] - y_offsets[iy])
    w = int(x_offsets[ix + 1] - x_offsets[ix])

    # Build query points for this chunk
    chunk_ys = y_coords[y_offsets[iy]:y_offsets[iy + 1]]
    chunk_xs = x_coords[x_offsets[ix]:x_offsets[ix + 1]]
    yy, xx = np.meshgrid(chunk_ys, chunk_xs, indexing='ij')
    query_pts = np.column_stack([yy.ravel(), xx.ravel()])

    ring = 0
    while True:
        jy_lo = max(iy - ring, 0)
        jy_hi = min(iy + 1 + ring, n_tile_y)
        jx_lo = max(ix - ring, 0)
        jx_hi = min(ix + 1 + ring, n_tile_x)

        covers_full = (jy_lo == 0 and jy_hi == n_tile_y
                       and jx_lo == 0 and jx_hi == n_tile_x)

        target_coords, target_vals = _collect_region_targets(
            raster, jy_lo, jy_hi, jx_lo, jx_hi,
            target_values, target_counts,
            y_coords, x_coords, y_offsets, x_offsets,
            coords_cache, values_cache,
        )

        if target_coords is None:
            if covers_full:
                # No targets in entire raster
                return np.full((h, w), np.nan, dtype=np.float32)
            ring += 1
            continue

        tree = cKDTree(target_coords)
        ub = max_distance if np.isfinite(max_distance) else np.inf
        dists, indices = tree.query(query_pts, p=p, distance_upper_bound=ub)

        n_targets = len(target_vals)
        oob = indices >= n_targets
        safe_idx = np.where(oob, 0, indices)

        # Always compute dists for convergence check
        dist_result = dists.reshape(h, w).astype(np.float32)
        dist_result[dist_result == np.inf] = np.nan

        def _converged():
            if covers_full:
                return True
            max_nearest = (np.nanmax(dist_result)
                           if not np.all(np.isnan(dist_result)) else 0.0)
            min_bd = _min_boundary_distance(
                iy, ix, y_coords, x_coords, y_offsets, x_offsets,
                jy_lo, jy_hi, jx_lo, jx_hi, n_tile_y, n_tile_x,
            )
            return max_nearest < min_bd

        if _converged():
            if process_mode == PROXIMITY:
                return dist_result
            elif process_mode == ALLOCATION:
                result = target_vals[safe_idx].astype(np.float32)
                result[oob] = np.nan
                return result.reshape(h, w)
            else:  # DIRECTION
                query_x = xx.ravel()
                query_y = yy.ravel()
                target_x = target_coords[safe_idx, 1]
                target_y = target_coords[safe_idx, 0]
                result = _vectorized_calc_direction(
                    query_x, target_x, query_y, target_y)
                result[oob] = np.nan
                result[dists == 0] = 0.0
                return result.reshape(h, w)

        ring += 1


def _build_tiled_kdtree(raster, y_coords, x_coords, target_values,
                        max_distance, p, target_counts,
                        coords_cache, values_cache,
                        chunks_y, chunks_x, process_mode):
    """Tiled (eager) KDTree query — memory-safe fallback."""
    H, W = raster.shape
    result_bytes = H * W * 4  # float32
    avail = _available_memory_bytes()
    if result_bytes > 0.8 * avail:
        raise MemoryError(
            f"Proximity result array ({H}x{W}, {result_bytes / 1e9:.1f} GB) "
            f"exceeds 80% of available memory ({avail / 1e9:.1f} GB)."
        )

    warnings.warn(
        "proximity: target coordinates exceed 50% of available memory; "
        "using tiled KDTree fallback (slower but memory-safe).",
        ResourceWarning,
        stacklevel=4,
    )

    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    y_offsets = _chunk_offsets(chunks_y)
    x_offsets = _chunk_offsets(chunks_x)

    result = np.full((H, W), np.nan, dtype=np.float32)

    for iy in range(n_tile_y):
        for ix in range(n_tile_x):
            chunk_result = _tiled_chunk_query(
                raster, iy, ix, y_coords, x_coords,
                y_offsets, x_offsets,
                target_values, target_counts,
                coords_cache, values_cache,
                max_distance, p, n_tile_y, n_tile_x, process_mode,
            )
            r0 = int(y_offsets[iy])
            r1 = int(y_offsets[iy + 1])
            c0 = int(x_offsets[ix])
            c1 = int(x_offsets[ix + 1])
            result[r0:r1, c0:c1] = chunk_result

    return da.from_array(result, chunks=raster.data.chunks)


def _build_global_kdtree(raster, y_coords, x_coords, target_values,
                         max_distance, p, target_counts,
                         coords_cache, values_cache,
                         chunks_y, chunks_x, process_mode):
    """Global KDTree query — fast, lazy via da.map_blocks."""
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    y_offsets = _chunk_offsets(chunks_y)
    x_offsets = _chunk_offsets(chunks_x)

    target_coords, target_vals = _collect_region_targets(
        raster, 0, n_tile_y, 0, n_tile_x,
        target_values, target_counts,
        y_coords, x_coords, y_offsets, x_offsets,
        coords_cache, values_cache,
    )

    tree = cKDTree(target_coords)

    chunk_fn = partial(
        _kdtree_chunk_fn,
        y_coords_1d=y_coords,
        x_coords_1d=x_coords,
        tree=tree,
        max_distance=max_distance if np.isfinite(max_distance) else np.inf,
        p=p,
        process_mode=process_mode,
        target_vals=target_vals,
        target_coords=target_coords,
    )

    return da.map_blocks(
        chunk_fn,
        raster.data,
        dtype=np.float32,
        meta=np.array((), dtype=np.float32),
    )


def _process_dask_kdtree(raster, x_coords, y_coords,
                         target_values, max_distance, distance_metric,
                         process_mode):
    """Memory-guarded k-d tree query for dask arrays.

    Phase 0: stream through chunks counting targets (with caching).
    Then choose global tree (fast, lazy) or tiled tree (memory-safe, eager)
    based on estimated memory usage.
    """
    p = 2 if distance_metric == EUCLIDEAN else 1  # Manhattan: p=1

    chunks_y, chunks_x = raster.data.chunks

    # Phase 0: streaming count pass
    target_counts, total_targets, coords_cache, values_cache = \
        _stream_target_counts(
            raster, target_values, y_coords, x_coords, chunks_y, chunks_x,
        )

    if total_targets == 0:
        return da.full(raster.shape, np.nan, dtype=np.float32,
                       chunks=raster.data.chunks)

    # Memory decision: 16 bytes coords + 4 bytes value + ~32 bytes tree overhead
    estimate = total_targets * 52
    avail = _available_memory_bytes()

    if estimate < 0.5 * avail:
        return _build_global_kdtree(
            raster, y_coords, x_coords, target_values,
            max_distance, p, target_counts,
            coords_cache, values_cache,
            chunks_y, chunks_x, process_mode,
        )
    else:
        return _build_tiled_kdtree(
            raster, y_coords, x_coords, target_values,
            max_distance, p, target_counts,
            coords_cache, values_cache,
            chunks_y, chunks_x, process_mode,
        )


def _process(
    raster,
    x,
    y,
    target_values,
    max_distance,
    distance_metric,
    process_mode
):

    raster_dims = raster.dims
    if raster_dims != (y, x):
        raise ValueError(
            "raster.coords should be named as coordinates:"
            "({0}, {1})".format(y, x)
        )

    distance_metric = DISTANCE_METRICS.get(distance_metric, None)
    if distance_metric is None:
        distance_metric = DISTANCE_METRICS["EUCLIDEAN"]

    target_values = np.asarray(target_values)

    if max_distance is None:
        max_distance = np.inf

    # Get 1D coordinate arrays (these are small, just the axis coordinates)
    x_coords = raster[x].data
    y_coords = raster[y].data

    # Ensure 1D coords are numpy arrays for max_possible_distance calculation
    if da is not None and isinstance(x_coords, da.Array):
        x_coords = x_coords.compute()
    if da is not None and isinstance(y_coords, da.Array):
        y_coords = y_coords.compute()

    # Compute max_possible_distance using coordinate endpoints directly
    max_possible_distance = _distance(
        x_coords[0], x_coords[-1], y_coords[0], y_coords[-1], distance_metric
    )

    @ngjit
    def _process_numpy(img, x_coords, y_coords):
        height, width = img.shape
        pan_near_x = np.zeros(width, dtype=np.int64)
        pan_near_y = np.zeros(width, dtype=np.int64)

        # output of the function
        output_img = np.full((height, width), np.nan, dtype=np.float32)
        img_distance = np.zeros(shape=(height, width), dtype=np.float32)

        # Loop from top to bottom of the image.
        for i in prange(width):
            pan_near_x[i] = -1
            pan_near_y[i] = -1

        # a single line of the input image img
        scan_line = np.zeros(width, dtype=img.dtype)

        # indexes of nearest pixels of current line scan_line
        nearest_xs = np.zeros(width, dtype=np.int64)
        nearest_ys = np.zeros(width, dtype=np.int64)

        for line in prange(height):
            # Read for target values.
            for i in prange(width):
                scan_line[i] = img[line][i]

            line_proximity = np.zeros(width, dtype=np.float32)

            for i in prange(width):
                line_proximity[i] = -1.0
                nearest_xs[i] = -1
                nearest_ys[i] = -1

            # left to right
            _process_proximity_line(
                scan_line, x_coords, y_coords,
                pan_near_x, pan_near_y, True,
                line, width, max_distance,
                line_proximity, nearest_xs, nearest_ys,
                target_values, distance_metric,
            )
            for i in prange(width):
                if nearest_xs[i] != -1 and line_proximity[i] >= 0:
                    if process_mode == ALLOCATION:
                        output_img[line][i] = img[nearest_ys[i], nearest_xs[i]]
                    elif process_mode == DIRECTION:
                        output_img[line][i] = _calc_direction(
                            x_coords[line, i],
                            x_coords[nearest_ys[i], nearest_xs[i]],
                            y_coords[line, i],
                            y_coords[nearest_ys[i], nearest_xs[i]],
                        )

            # right to left
            for i in prange(width):
                nearest_xs[i] = -1
                nearest_ys[i] = -1

            _process_proximity_line(
                scan_line, x_coords, y_coords,
                pan_near_x, pan_near_y, False,
                line, width, max_distance,
                line_proximity, nearest_xs, nearest_ys,
                target_values, distance_metric,
            )

            for i in prange(width):
                img_distance[line][i] = line_proximity[i]

                if nearest_xs[i] != -1 and line_proximity[i] >= 0:
                    if process_mode == ALLOCATION:
                        output_img[line][i] = img[nearest_ys[i], nearest_xs[i]]

                    elif process_mode == DIRECTION:
                        output_img[line][i] = _calc_direction(
                            x_coords[line, i],
                            x_coords[nearest_ys[i], nearest_xs[i]],
                            y_coords[line, i],
                            y_coords[nearest_ys[i], nearest_xs[i]],
                        )

        # Loop from bottom to top of the image.
        for i in prange(width):
            pan_near_x[i] = -1
            pan_near_y[i] = -1

        for line in prange(height - 1, -1, -1):
            # Read first pass proximity.
            for i in prange(width):
                line_proximity[i] = img_distance[line][i]

            # Read pixel target_values.
            for i in prange(width):
                scan_line[i] = img[line][i]

            # Right to left
            for i in prange(width):
                nearest_xs[i] = -1
                nearest_ys[i] = -1

            _process_proximity_line(
                scan_line, x_coords, y_coords,
                pan_near_x, pan_near_y, False,
                line, width, max_distance,
                line_proximity, nearest_xs, nearest_ys,
                target_values, distance_metric,
            )

            for i in prange(width):
                if nearest_xs[i] != -1 and line_proximity[i] >= 0:
                    if process_mode == ALLOCATION:
                        output_img[line][i] = img[nearest_ys[i], nearest_xs[i]]

                    elif process_mode == DIRECTION:
                        output_img[line][i] = _calc_direction(
                            x_coords[line, i],
                            x_coords[nearest_ys[i], nearest_xs[i]],
                            y_coords[line, i],
                            y_coords[nearest_ys[i], nearest_xs[i]],
                        )

            # Left to right
            for i in prange(width):
                nearest_xs[i] = -1
                nearest_ys[i] = -1

            _process_proximity_line(
                scan_line, x_coords, y_coords,
                pan_near_x, pan_near_y, True,
                line, width, max_distance,
                line_proximity, nearest_xs, nearest_ys,
                target_values, distance_metric,
            )

            # final post processing of distances
            for i in prange(width):
                if line_proximity[i] < 0:
                    line_proximity[i] = np.nan
                else:
                    if nearest_xs[i] != -1 and line_proximity[i] >= 0:
                        if process_mode == ALLOCATION:
                            output_img[line][i] = img[
                                nearest_ys[i], nearest_xs[i]]

                        elif process_mode == DIRECTION:
                            output_img[line][i] = _calc_direction(
                                x_coords[line, i],
                                x_coords[nearest_ys[i], nearest_xs[i]],
                                y_coords[line, i],
                                y_coords[nearest_ys[i], nearest_xs[i]],
                            )

            for i in prange(width):
                img_distance[line][i] = line_proximity[i]

        if process_mode == PROXIMITY:
            return img_distance
        else:
            return output_img

    def _process_dask(raster, xs, ys):

        if max_distance >= max_possible_distance:
            # consider all targets in the whole raster
            # the data array is computed at once,
            # make sure your data fit your memory
            height, width = raster.shape
            raster.data = raster.data.rechunk({0: height, 1: width})
            xs = xs.rechunk({0: height, 1: width})
            ys = ys.rechunk({0: height, 1: width})
            pad_y = pad_x = 0
        else:
            cellsize_x, cellsize_y = get_dataarray_resolution(raster)
            # calculate padding for each chunk
            pad_y = int(max_distance / cellsize_y + 0.5)
            pad_x = int(max_distance / cellsize_x + 0.5)

        out = da.map_overlap(
            _process_numpy,
            raster.data, xs, ys,
            depth=(pad_y, pad_x),
            boundary=np.nan,
            meta=np.array(()),
        )
        return out

    if isinstance(raster.data, np.ndarray):
        # numpy case - create full coordinate arrays as numpy
        xs = np.tile(x_coords, raster.shape[0]).reshape(raster.shape)
        ys = np.repeat(y_coords, raster.shape[1]).reshape(raster.shape)
        result = _process_numpy(raster.data, xs, ys)

    elif has_cuda_and_cupy() and is_cupy_array(raster.data):
        result = _process_cupy(
            raster.data, x_coords, y_coords,
            target_values, max_distance, distance_metric, process_mode,
        )

    elif da is not None and isinstance(raster.data, da.Array):
        if (has_cuda_and_cupy() and is_dask_cupy(raster)
                and max_distance < max_possible_distance):
            # Bounded dask+cupy: out-of-core GPU via map_overlap
            result = _process_dask_cupy(
                raster, x_coords, y_coords,
                target_values, max_distance, distance_metric, process_mode,
            )
        else:
            # dask+numpy path (or unbounded dask+cupy → convert first)
            was_dask_cupy = has_cuda_and_cupy() and is_dask_cupy(raster)
            if was_dask_cupy:
                import cupy as cp
                # Unbounded: convert to dask+numpy for KDTree/line-sweep
                # (KDTree is CPU-only; O(N log T) beats brute-force O(NT))
                original_chunks = raster.data.chunks
                raster = raster.copy(
                    data=raster.data.map_blocks(
                        lambda x: x.get(), dtype=raster.dtype,
                        meta=np.array(()),
                    )
                )

            use_kdtree = (
                cKDTree is not None
                and distance_metric in (EUCLIDEAN, MANHATTAN)
                and max_distance >= max_possible_distance
            )
            if use_kdtree:
                result = _process_dask_kdtree(
                    raster, x_coords, y_coords,
                    target_values, max_distance, distance_metric,
                    process_mode,
                )
            else:
                # Memory guard: unbounded distance on large rasters can OOM
                if max_distance >= max_possible_distance:
                    H, W = raster.shape
                    required = H * W * 4 * 3  # raster + xs + ys, float32
                    avail = _available_memory_bytes()
                    if required > 0.8 * avail:
                        if cKDTree is None:
                            raise MemoryError(
                                "Raster too large for single-chunk processing "
                                "and scipy is not installed for memory-safe "
                                "KDTree path. Install scipy or set a finite "
                                "max_distance."
                            )
                        else:  # must be GREAT_CIRCLE
                            raise MemoryError(
                                "GREAT_CIRCLE with unbounded max_distance on "
                                "this raster would exceed available memory. "
                                "Set a finite max_distance."
                            )

                # Existing path: build 2D coordinate arrays as dask arrays
                x_coords_da = da.from_array(x_coords, chunks=x_coords.shape[0])
                y_coords_da = da.from_array(y_coords, chunks=y_coords.shape[0])
                xs = da.tile(x_coords_da, (raster.shape[0], 1))
                ys = da.repeat(y_coords_da, raster.shape[1]).reshape(
                    raster.shape)
                xs = xs.rechunk(raster.chunks)
                ys = ys.rechunk(raster.chunks)
                result = _process_dask(raster, xs, ys)

            # Convert result back to dask+cupy if input was dask+cupy
            if was_dask_cupy:
                result = result.map_blocks(
                    cp.asarray, dtype=result.dtype,
                    meta=cp.array((), dtype=result.dtype),
                )

    else:
        raise TypeError(
            f"Unsupported array type {type(raster.data).__name__} "
            f"for proximity/allocation/direction"
        )

    return result


# ported from
# https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp
@supports_dataset
def proximity(
    raster: xr.DataArray,
    x: str = "x",
    y: str = "y",
    target_values: list = [],
    max_distance: float = np.inf,
    distance_metric: str = "EUCLIDEAN",
) -> xr.DataArray:
    """
    Computes the proximity of all pixels in the image to a set of pixels
    in the source image based on a distance metric.

    This function attempts to compute the proximity of all pixels in the
    image to a set of pixels in the source image. The following options
    are used to define the behavior of the function. By default all
    non-zero pixels in `raster.values` will be considered the "target",
    and all proximities will be computed in pixels. Note that target
    pixels are set to the value corresponding to a distance of zero.

    Proximity support NumPy backed, and Dask with NumPy backed
    xarray DataArray. The return values of proximity are of the same type as
    the input type.
    If input raster is a NumPy-backed DataArray, the result is NumPy-backed.
    If input raster is a Dask-backed DataArray, the result is Dask-backed.

    The implementation for NumPy-backed is ported from GDAL, which uses
    a dynamic programming approach to identify nearest target of a pixel from
    its surrounding neighborhood in a 3x3 window.
    The implementation for Dask-backed uses `dask.map_overlap` to compute
    proximity chunk by chunk by expanding the chunk's borders to cover
    the `max_distance`.

    Parameters
    ----------
    raster : xr.DataArray or xr.Dataset
        2D array image with `raster.shape` = (height, width).
        If a Dataset is passed, the function is applied to each
        data variable independently, returning a Dataset.

    x : str, default='x'
        Name of x-coordinates.

    y : str, default='y'
        Name of y-coordinates.

    target_values: list
        Target pixel values to measure the distance from. If this option
        is not provided, proximity will be computed from non-zero pixel
        values.

    max_distance: float, default=np.inf
        The maximum distance to search. Proximity distances greater than
        this value will be set to NaN.
        Should be given in the same distance unit as input.
        For example, if input raster is in lat-lon and distances between points
        within the raster is calculated using Euclidean distance metric,
        `max_distance` should also be provided in lat-lon unit.
        If using Great Circle distance metric, and thus all distances is in km,
        `max_distance` should also be provided in kilometer unit.

        When scaling with Dask, whether the function scales well depends on
        the `max_distance` value. If `max_distance` is infinite by default,
        this function only works on a single machine.
        It should scale well, however, if `max_distance` is relatively small
        compared to the maximum possible distance in two arbitrary points
        in the input raster. Note that if `max_distance` is equal or larger
        than the max possible distance between 2 arbitrary points in the input
        raster, the input data array will be rechunked.

    distance_metric: str, default='EUCLIDEAN'
        The metric for calculating distance between 2 points.
        Valid distance metrics are:
        'EUCLIDEAN', 'GREAT_CIRCLE', and 'MANHATTAN'.

    Returns
    -------
    xr.DataArray or xr.Dataset
        If ``raster`` is a DataArray, returns a DataArray.
        If ``raster`` is a Dataset, returns a Dataset with each
        variable processed independently.
        2D array of proximity values.
        All other input attributes are preserved.

    References
    ----------
        - OSGeo: https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        >>> n, m = data.shape
        >>> raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
        >>> raster['y'] = np.arange(n)[::-1]
        >>> raster['x'] = np.arange(m)

        >>> from xrspatial import proximity
        >>> proximity_agg = proximity(raster)
        >>> proximity_agg
        <xarray.DataArray (y: 5, x: 5)>
        array([[3.1622777, 2.236068 , 1.4142135, 1.       , 1.4142135],
               [3.       , 2.       , 1.       , 0.       , 1.       ],
               [3.1622777, 2.236068 , 1.4142135, 1.       , 1.4142135],
               [3.6055512, 2.828427 , 2.236068 , 2.       , 2.236068 ],
               [4.2426405, 3.6055512, 3.1622777, 3.       , 3.1622777]],
              dtype=float32)
        Coordinates:
          * y        (y) int64 4 3 2 1 0
          * x        (x) int64 0 1 2 3 4
    """

    _validate_raster(raster, func_name='proximity', name='raster')

    proximity_img = _process(
        raster,
        x=x,
        y=y,
        target_values=target_values,
        max_distance=max_distance,
        distance_metric=distance_metric,
        process_mode=PROXIMITY,
    )

    result = xr.DataArray(
        proximity_img,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs
    )

    return result


@supports_dataset
def allocation(
    raster: xr.DataArray,
    x: str = "x",
    y: str = "y",
    target_values: list = [],
    max_distance: float = np.inf,
    distance_metric: str = "EUCLIDEAN",
):
    """
    Calculates, for all pixels in the input raster, the nearest source
    based on a set of target values and a distance metric.

    This function attempts to produce the value of nearest feature of all
    pixels in the image to a set of pixels in the source image. The
    following options are used to define the behavior of the function.
    By default all non-zero pixels in `raster.values` will be considered
    as"target", and all allocation will be computed in pixels.

    Allocation supports NumPy backed, and Dask with NumPy backed
    xarray DataArray. The return values of `allocation` are of the same type as
    the input type.
    If input raster is a NumPy-backed DataArray, the result is NumPy-backed.
    If input raster is a Dask-backed DataArray, the result is Dask-backed.

    `allocation` uses the same approach as `proximity`, which is ported
    from GDAL. A dynamic programming approach is used for identifying nearest
    target of a pixel from its surrounding neighborhood in a 3x3 window.
    The implementation for Dask-backed uses `dask.map_overlap` to compute
    `allocation` chunk by chunk by expanding the chunk's borders to cover
    the `max_distance`.

    Parameters
    ----------
    raster : xr.DataArray or xr.Dataset
        2D array of target data.
        If a Dataset is passed, the function is applied to each
        data variable independently, returning a Dataset.

    x : str, default='x'
        Name of x-coordinates.

    y : str, default='y'
        Name of y-coordinates.

    target_values : list
        Target pixel values to measure the distance from. If this option
        is not provided, allocation will be computed from non-zero pixel
        values.

    max_distance: float, default=np.inf
        The maximum distance to search. Proximity distances greater than
        this value will be set to NaN.
        Should be given in the same distance unit as input.
        For example, if input raster is in lat-lon and distances between points
        within the raster is calculated using Euclidean distance metric,
        `max_distance` should also be provided in lat-lon unit.
        If using Great Circle distance metric, and thus all distances is in km,
        `max_distance` should also be provided in kilometer unit.

        When scaling with Dask, whether the function scales well depends on
        the `max_distance` value. If `max_distance` is infinite by default,
        this function only works on a single machine.
        It should scale well, however, if `max_distance` is relatively small
        compared to the maximum possible distance in two arbitrary points
        in the input raster. Note that if `max_distance` is equal or larger
        than the max possible distance between 2 arbitrary points in the input
        raster, the input data array will be rechunked.

    distance_metric : str, default='EUCLIDEAN'
        The metric for calculating distance between 2 points. Valid
        distance metrics are: 'EUCLIDEAN', 'GREAT_CIRCLE', and 'MANHATTAN'.

    Returns
    -------
    xr.DataArray or xr.Dataset
        If ``raster`` is a DataArray, returns a DataArray.
        If ``raster`` is a Dataset, returns a Dataset with each
        variable processed independently.
        2D array of allocation values.
        All other input attributes are preserved.

    References
    ----------
        - OSGeo: https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 2., 0.],
            [0., 0., 3., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        >>> n, m = data.shape
        >>> raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
        >>> raster['y'] = np.arange(n)[::-1]
        >>> raster['x'] = np.arange(m)

        >>> from xrspatial import allocation
        >>> allocation_agg = allocation(raster)
        >>> allocation_agg
        <xarray.DataArray (y: 5, x: 5)>
        array([[1., 1., 2., 2., 2.],
               [1., 1., 1., 2., 2.],
               [1., 1., 3., 2., 2.],
               [1., 3., 3., 3., 2.],
               [3., 3., 3., 3., 3.]])
        Coordinates:
          * y        (y) int64 4 3 2 1 0
          * x        (x) int64 0 1 2 3 4
    """

    _validate_raster(raster, func_name='allocation', name='raster')

    allocation_img = _process(
        raster,
        x=x,
        y=y,
        target_values=target_values,
        max_distance=max_distance,
        distance_metric=distance_metric,
        process_mode=ALLOCATION,
    )

    # convert to have same type as of input @raster
    result = xr.DataArray(
        allocation_img,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )
    return result


@supports_dataset
def direction(
    raster: xr.DataArray,
    x: str = "x",
    y: str = "y",
    target_values: list = [],
    max_distance: float = np.inf,
    distance_metric: str = "EUCLIDEAN",
):
    """
    Calculates, for all cells in the array, the downward slope direction
    Calculates, for all pixels in the input raster, the direction to
    nearest source based on a set of target values and a distance metric.

    This function attempts to calculate for each cell, the the direction,
    in degrees, to the nearest source. The output values are based on
    compass directions, where 90 is for the east, 180 for the south,
    270 for the west, 360 for the north, and 0 for the source cell
    itself. The following options are used to define the behavior of
    the function. By default all non-zero pixels in `raster.values`
    will be considered as "target", and all direction will be computed
    in pixels.

    Direction support NumPy backed, and Dask with NumPy backed
    xarray DataArray. The return values of `direction` are of the same type as
    the input type.
    If input raster is a NumPy-backed DataArray, the result is NumPy-backed.
    If input raster is a Dask-backed DataArray, the result is Dask-backed.

    Similar to `proximity`, the implementation for NumPy-backed is ported
    from GDAL, which uses a dynamic programming approach to identify
    nearest target of a pixel from its surrounding neighborhood in a 3x3 window
    The implementation for Dask-backed uses `dask.map_overlap` to compute
    proximity direction chunk by chunk by expanding the chunk's borders
    to cover the `max_distance`.

    Parameters
    ----------
    raster : xr.DataArray or xr.Dataset
        2D array image with `raster.shape` = (height, width).
        If a Dataset is passed, the function is applied to each
        data variable independently, returning a Dataset.

    x : str, default='x'
        Name of x-coordinates.

    y : str, default='y'
        Name of y-coordinates.

    target_values: list
        Target pixel values to measure the distance from. If this
        option is not provided, proximity will be computed from
        non-zero pixel values.

    max_distance: float, default=np.inf
        The maximum distance to search. Proximity distances greater than
        this value will be set to NaN.
        Should be given in the same distance unit as input.
        For example, if input raster is in lat-lon and distances between points
        within the raster is calculated using Euclidean distance metric,
        `max_distance` should also be provided in lat-lon unit.
        If using Great Circle distance metric, and thus all distances is in km,
        `max_distance` should also be provided in kilometer unit.

        When scaling with Dask, whether the function scales well depends on
        the `max_distance` value. If `max_distance` is infinite by default,
        this function only works on a single machine.
        It should scale well, however, if `max_distance` is relatively small
        compared to the maximum possible distance in two arbitrary points
        in the input raster. Note that if `max_distance` is equal or larger
        than the max possible distance between 2 arbitrary points in the input
        raster, the input data array will be rechunked.

    distance_metric: str, default='EUCLIDEAN'
        The metric for calculating distance between 2 points.
        Valid distance_metrics are:
        'EUCLIDEAN', 'GREAT_CIRCLE', and 'MANHATTAN'.

    Returns
    -------
    xr.DataArray or xr.Dataset
        If ``raster`` is a DataArray, returns a DataArray.
        If ``raster`` is a Dataset, returns a Dataset with each
        variable processed independently.
        2D array of direction values.
        All other input attributes are preserved.

    References
    ----------
        - OSGeo: https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.]
        ])
        >>> n, m = data.shape
        >>> raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
        >>> raster['y'] = np.arange(n)[::-1]
        >>> raster['x'] = np.arange(m)

        >>> from xrspatial import direction
        >>> direction_agg = direction(raster)
        >>> direction_agg
        <xarray.DataArray (y: 5, x: 5)>
        array([[ 45.      ,  26.56505 , 360.      , 333.43494 , 315.      ],
               [ 63.434948,  45.      , 360.      , 315.      , 296.56506 ],
               [ 90.      ,  90.      ,   0.      , 270.      , 270.      ],
               [360.      , 135.      , 180.      , 225.      , 243.43495 ],
               [  0.      , 270.      , 180.      , 206.56505 , 225.      ]],
              dtype=float32)
        Coordinates:
          * y        (y) int64 4 3 2 1 0
          * x        (x) int64 0 1 2 3 4
    """

    _validate_raster(raster, func_name='direction', name='raster')

    direction_img = _process(
        raster,
        x=x,
        y=y,
        target_values=target_values,
        max_distance=max_distance,
        distance_metric=distance_metric,
        process_mode=DIRECTION,
    )

    result = xr.DataArray(
        direction_img,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs
    )
    return result
