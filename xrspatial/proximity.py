import threading
import warnings
from functools import partial

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
from numba import cuda, jit, prange

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

from xrspatial.dataset_support import supports_dataset
from xrspatial.pathfinding import _available_memory_bytes
from xrspatial.utils import (_dask_task_name_kwargs, _validate_raster, cuda_args, has_cuda_and_cupy,
                             is_cupy_array, is_dask_cupy, ngjit)

EUCLIDEAN = 0
GREAT_CIRCLE = 1
MANHATTAN = 2

PROXIMITY = 0
ALLOCATION = 1
DIRECTION = 2

# Map the runtime process_mode to the public function name so the shared
# _process compute layers label their dask tasks under the op that called them.
_PROCESS_MODE_TASK_NAMES = {
    PROXIMITY: 'xrspatial.proximity',
    ALLOCATION: 'xrspatial.allocation',
    DIRECTION: 'xrspatial.direction',
}


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
        Radius of sphere (earth), in meters. The default is the WGS84
        equatorial radius, so the returned distance is in meters.

    Returns
    -------
    distance : float
        Great-Circle distance between two points, in the same unit as
        ``radius`` (meters by default).

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


def _check_monotonic_coords(x_coords, y_coords, x, y):
    """Reject non-monotonic 1D coordinates.

    Every backend in this module assumes the 1D axis coordinates are
    monotonic: ``max_possible_distance`` is taken from the endpoints, the
    dask halo treats array adjacency as spatial adjacency, and the tiled
    KDTree convergence check lower-bounds the
    out-of-region distance with chunk-boundary coordinate gaps. None of
    those hold when a coordinate axis is not monotonic, so a non-monotonic
    axis silently yields wrong proximity/allocation/direction. Reject it up
    front with a clear message instead.

    A single-element axis has no order to violate and is allowed.
    """
    for coords, name in ((x_coords, x), (y_coords, y)):
        if len(coords) < 2:
            continue
        diffs = np.diff(coords)
        ascending = np.all(diffs > 0)
        descending = np.all(diffs < 0)
        if not (ascending or descending):
            raise ValueError(
                "proximity/allocation/direction require strictly monotonic "
                "(strictly increasing or strictly decreasing, no duplicate or "
                "NaN values) 1D coordinates, but the {0!r} axis does not "
                "qualify. Sort the raster along {0!r} before calling.".format(
                    name)
            )


def _great_circle_col_halo(x_coords, y_coords, max_distance):
    """Column halo depth (in pixels) for the GREAT_CIRCLE metric.

    Great-circle distance is periodic in longitude (haversine takes the
    short way around the sphere) and its chords shorten toward the poles,
    so a linear sum of per-column step distances is not a lower bound on
    the spherical distance between two grid points. Use the chord bound

        dist(p, t) >= 2 * R * asin(cos(lat_max) * |sin(dlon / 2)|)

    which holds for every pair of grid points (``lat_max`` is the largest
    absolute latitude on the raster). Inverting it for ``max_distance``
    gives ``dlon_max``: any pair separated by more than ``dlon_max``
    degrees of longitude (the short way around) is farther apart than
    ``max_distance`` no matter the latitudes.

    When the bound cannot exclude anything -- ``max_distance`` reaches the
    180-degree chord at ``lat_max``, or targets across the +/-180 seam are
    within reach (seam gap <= ``dlon_max``) -- no array-space halo can
    cover the wrap. Return a depth one larger than the axis so
    ``_fit_halo_to_chunks`` folds the x axis into a single chunk and every
    chunk sees all columns.
    """
    width = len(x_coords)
    if width < 2:
        return 0
    fold = width + 1

    radius = 6378137.0
    half_angle = max_distance / (2.0 * radius)
    if half_angle >= np.pi / 2.0:
        # max_distance spans half the circumference: everything is in reach.
        return fold

    cos_lat_max = np.cos(np.radians(np.abs(np.asarray(y_coords)).max()))
    sin_half = np.sin(half_angle)
    if sin_half >= cos_lat_max:
        # Even a 180-degree longitude gap at the worst-case latitude stays
        # within max_distance, so no longitude separation excludes a target.
        return fold

    dlon_max = np.degrees(2.0 * np.arcsin(sin_half / cos_lat_max))

    # Wrap check: the smallest longitude separation through the +/-180 seam
    # is between the first and last columns. If that is within dlon_max, a
    # target near one edge of the array can be the nearest target of a pixel
    # near the opposite edge, which no per-chunk halo can express.
    span = abs(float(x_coords[-1]) - float(x_coords[0]))
    seam_gap = 360.0 - span
    if seam_gap <= dlon_max:
        return fold

    # No wrap in reach: a target k columns away is at least k * min_step
    # degrees of longitude away (monotonic coords), so columns beyond
    # dlon_max / min_step are excluded by the chord bound.
    min_step = np.abs(np.diff(np.asarray(x_coords, dtype=np.float64))).min()
    return int(np.ceil(dlon_max / min_step))


def _halo_depth(x_coords, y_coords, max_distance, distance_metric):
    """Overlap depth in pixels for the bounded dask map_overlap call.

    ``max_distance`` is expressed in the same unit as the chosen
    distance_metric, so the pixel pitch is measured with that same metric.
    Using the raw degree cellsize for GREAT_CIRCLE (where max_distance is in
    metres) would yield a meaningless depth.

    The depth is sized from the *densest* (smallest positive) spacing along
    each axis rather than only the first coordinate pair. On irregular
    coordinates the first gap can be much larger than later gaps; sizing the
    halo from the first gap alone leaves it too thin and chunks then miss
    valid targets just past the boundary.

    An axis with a single coordinate has no spacing and therefore contributes
    no halo along that axis (depth 0), so (1, N) and (N, 1) rasters do not
    crash on the missing second coordinate.

    For GREAT_CIRCLE the column depth comes from the spherical chord bound
    in ``_great_circle_col_halo``: longitude is periodic and chords shorten
    toward the poles, so a per-column linear step sum is not a valid lower
    bound there. When targets across the +/-180 seam are within
    ``max_distance``, the returned column depth exceeds the axis length so
    ``_fit_halo_to_chunks`` folds the axis. The row depth stays linear: the
    great-circle distance between two points is never smaller than their
    meridian (north-south) separation, so the per-row step sum is a valid
    lower bound for every metric.
    """
    def _min_step_distance(coords, x_ref, y_ref, along):
        if len(coords) < 2:
            return None
        smallest = None
        for i in range(len(coords) - 1):
            if along == "row":
                d = _distance(
                    x_ref, x_ref, coords[i], coords[i + 1], distance_metric)
            else:
                d = _distance(
                    coords[i], coords[i + 1], y_ref, y_ref, distance_metric)
            if d > 0 and (smallest is None or d < smallest):
                smallest = d
        return smallest

    dist_per_row = _min_step_distance(y_coords, x_coords[0], None, "row")
    pad_y = 0 if dist_per_row is None else int(max_distance / dist_per_row + 0.5)

    if distance_metric == GREAT_CIRCLE:
        pad_x = _great_circle_col_halo(x_coords, y_coords, max_distance)
    else:
        dist_per_col = _min_step_distance(x_coords, None, y_coords[0], "col")
        pad_x = 0 if dist_per_col is None else int(
            max_distance / dist_per_col + 0.5)
    return pad_y, pad_x


def _fit_halo_to_chunks(pad_y, pad_x, *arrays):
    """Make a bounded halo depth that ``da.map_overlap`` will accept.

    ``map_overlap`` rejects a depth that is larger than the smallest chunk
    along that axis. The pixel halo from ``_halo_depth`` can exceed that on
    skinny rasters (e.g. a 3-row raster with a 10-pixel halo), or on
    great-circle rasters where the pixel pitch shrinks toward the poles. When
    a halo is too deep for the chunking, fold that whole axis into a single
    chunk and drop its depth to zero: every chunk then sees the full axis, so
    no target within ``max_distance`` is missed and the result still matches
    the NumPy backend. The fold deliberately trades chunking on that axis for
    correctness; clamping the depth while keeping multiple chunks would
    silently drop targets that fall in a non-adjacent chunk, so do not replace
    it with a bare depth clamp.

    ``arrays`` are the dask arrays passed to the same ``map_overlap`` call
    (the raster and the coordinate grids); they all share the raster's
    chunking and are rechunked together so the call stays aligned.

    Returns the adjusted ``(pad_y, pad_x)`` and the (possibly rechunked)
    arrays in the same order they were given.
    """
    arrays = list(arrays)
    height, width = arrays[0].shape
    chunks_y, chunks_x = arrays[0].chunks
    rechunk = {}
    if pad_y > min(chunks_y):
        rechunk[0] = height
        pad_y = 0
    if pad_x > min(chunks_x):
        rechunk[1] = width
        pad_x = 0
    if rechunk:
        arrays = [a.rechunk(rechunk) for a in arrays]
    return pad_y, pad_x, arrays


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


@ngjit
def _is_target_value(v, target_values):
    # A pixel is a target if it matches one of target_values, or (when no
    # target_values are given) if it is non-zero and finite. NaN padding from
    # dask's boundary=np.nan is excluded either way.
    if len(target_values) == 0:
        return v != 0 and np.isfinite(v)
    for k in range(len(target_values)):
        if v == target_values[k]:
            return True
    return False


# Numba parallel=True kernels must not be launched concurrently from multiple
# Python threads: the default 'workqueue' threading layer is not threadsafe and
# aborts the process (SIGABRT on macOS) when two host threads enter a parallel
# region at once. _process_dask maps _process_numpy over chunks under dask's
# threaded scheduler, and that reaches the brute-force kernel for GREAT_CIRCLE,
# ALLOCATION, DIRECTION and the no-scipy PROXIMITY fallback, so the kernel
# launch is serialized behind this lock. Same hazard and fix as the
# convolution, terrain and reproject kernels (#3141).
_PARALLEL_KERNEL_LOCK = threading.Lock()


@ngjit
def _collect_targets(img, target_values):
    """Row/col indices of every target pixel, in flat (row-major) order."""
    height, width = img.shape
    n_targets = 0
    for line in range(height):
        for col in range(width):
            if _is_target_value(img[line, col], target_values):
                n_targets += 1

    target_rows = np.empty(n_targets, dtype=np.int64)
    target_cols = np.empty(n_targets, dtype=np.int64)
    t = 0
    for line in range(height):
        for col in range(width):
            if _is_target_value(img[line, col], target_values):
                target_rows[t] = line
                target_cols[t] = col
                t += 1
    return target_rows, target_cols


# The three inner loops below each scan every target for one pixel and return
# (index, float32 distance) of the nearest one, or (-1, inf) with no targets.
#
# They compare a cheap proxy that is monotone in the distance (squared
# distance, |dx| + |dy|, the haversine term) and only evaluate the sqrt /
# arcsin and the float32 rounding when the proxy beats the running best. The
# strict ``<`` that decides the winner still runs on the float32 distance, the
# same value ``_distance`` returns, so the documented tie-break is unchanged:
# two targets whose float64 distances differ only past the float32 mantissa
# are a tie and the lowest flat index wins (issue #3689, and the matching
# comment in ``_proximity_cuda_kernel``). A candidate whose proxy does not
# beat the running best has a float32 distance >= the running best and could
# never have won under that rule, so skipping it changes nothing.
# The running best is updated with a select rather than a branch, and only
# when the candidate wins, so a NaN distance (say a haversine term rounded a
# hair past 1.0) is skipped instead of poisoning every later comparison.

@ngjit
def _nearest_euclidean(px, py, txs, tys):
    best_proxy = np.inf
    best_dist = np.float32(np.inf)
    best_idx = -1
    for k in range(len(txs)):
        dx = px - txs[k]
        dy = py - tys[k]
        proxy = dx * dx + dy * dy
        if proxy < best_proxy:
            d = np.float32(np.sqrt(proxy))
            better = d < best_dist
            best_idx = k if better else best_idx
            best_dist = d if better else best_dist
            best_proxy = proxy
    return best_idx, best_dist


@ngjit
def _nearest_manhattan(px, py, txs, tys):
    best_proxy = np.inf
    best_dist = np.float32(np.inf)
    best_idx = -1
    for k in range(len(txs)):
        dx = px - txs[k]
        dy = py - tys[k]
        proxy = abs(dx) + abs(dy)
        if proxy < best_proxy:
            d = np.float32(proxy)
            better = d < best_dist
            best_idx = k if better else best_idx
            best_dist = d if better else best_dist
            best_proxy = proxy
    return best_idx, best_dist


@ngjit
def _nearest_great_circle(px, py, tlons, tlats, tcoslats):
    # Same arithmetic, in the same order, as great_circle_distance with the
    # default radius, so the float32 distance is bit-identical to _distance.
    lat1 = np.radians(py)
    lon1 = np.radians(px)
    coslat1 = np.cos(lat1)
    best_proxy = np.inf
    best_dist = np.float32(np.inf)
    best_idx = -1
    for k in range(len(tlons)):
        dlon = tlons[k] - lon1
        dlat = tlats[k] - lat1
        proxy = np.sin(dlat / 2.0) ** 2 + \
            coslat1 * tcoslats[k] * np.sin(dlon / 2.0) ** 2
        if proxy < best_proxy:
            d = np.float32(6378137 * 2 * np.arcsin(np.sqrt(proxy)))
            better = d < best_dist
            best_idx = k if better else best_idx
            best_dist = d if better else best_dist
            best_proxy = proxy
    return best_idx, best_dist


@ngjit
def _great_circle_target_terms(txs, tys):
    # Precompute the per-target radians and cos(lat) with numba's np.radians /
    # np.cos rather than numpy's, so they match what the per-pixel side of
    # _nearest_great_circle (and great_circle_distance) computes bit for bit.
    n = len(txs)
    tlons = np.empty(n, dtype=np.float64)
    tlats = np.empty(n, dtype=np.float64)
    tcoslats = np.empty(n, dtype=np.float64)
    for k in range(n):
        tlons[k] = np.radians(txs[k])
        tlats[k] = np.radians(tys[k])
        tcoslats[k] = np.cos(tlats[k])
    return tlons, tlats, tcoslats


@ngjit
def _great_circle_range_violation(xs, ys, txs, tys):
    # Report the first out-of-range coordinate in the order the per-pair
    # guards in great_circle_distance would have hit it when the pixel loop
    # called it pair by pair: pixel (0, 0) against every target, then the
    # remaining pixels. 0 = no violation, otherwise the guard number (1: x of
    # the first point, 2: x of the second, 3: y of the first, 4: y of the
    # second). NaN coordinates (dask halo padding) fail no comparison, as
    # before.
    px = xs[0, 0]
    py = ys[0, 0]
    if px > 180 or px < -180:
        return 1
    if txs[0] > 180 or txs[0] < -180:
        return 2
    if py > 90 or py < -90:
        return 3
    if tys[0] > 90 or tys[0] < -90:
        return 4
    for k in range(1, len(txs)):
        if txs[k] > 180 or txs[k] < -180:
            return 2
        if tys[k] > 90 or tys[k] < -90:
            return 4
    height, width = xs.shape
    for line in range(height):
        for col in range(width):
            if xs[line, col] > 180 or xs[line, col] < -180:
                return 1
            if ys[line, col] > 90 or ys[line, col] < -90:
                return 3
    return 0


_GREAT_CIRCLE_RANGE_MESSAGES = {
    1: "Invalid x-coordinate of the first point."
       "Must be in the range [-180, 180]",
    2: "Invalid x-coordinate of the second point."
       "Must be in the range [-180, 180]",
    3: "Invalid y-coordinate of the first point."
       "Must be in the range [-90, 90]",
    4: "Invalid y-coordinate of the second point."
       "Must be in the range [-90, 90]",
}


@jit(nopython=True, nogil=True, parallel=True)
def _bruteforce_kernel(
    img, xs, ys, target_rows, target_cols, txs, tys, tlons, tlats, tcoslats,
    max_distance, distance_metric, process_mode, output
):
    # The metric branch sits per pixel, outside the target loop, and the
    # metric stays a runtime value so one compiled specialization serves all
    # three. Rows are independent, so prange over them.
    height, width = img.shape
    for line in prange(height):
        for col in range(width):
            px = xs[line, col]
            py = ys[line, col]
            if distance_metric == EUCLIDEAN:
                best_idx, best_dist = _nearest_euclidean(px, py, txs, tys)
            elif distance_metric == GREAT_CIRCLE:
                best_idx, best_dist = _nearest_great_circle(
                    px, py, tlons, tlats, tcoslats)
            else:
                best_idx, best_dist = _nearest_manhattan(px, py, txs, tys)
            if best_idx >= 0 and best_dist <= max_distance:
                if process_mode == PROXIMITY:
                    output[line, col] = best_dist
                elif process_mode == ALLOCATION:
                    output[line, col] = img[
                        target_rows[best_idx], target_cols[best_idx]]
                else:
                    output[line, col] = _calc_direction(
                        px, txs[best_idx], py, tys[best_idx])


def _process_numpy_bruteforce(
    img, xs, ys, target_values, max_distance, distance_metric, process_mode
):
    """Exact nearest-target proximity / allocation / direction on the CPU.

    For every pixel, scan all target pixels and keep the closest one under the
    chosen distance metric. This is the same brute-force search the CUDA kernel
    runs (see ``_proximity_cuda_kernel``). It covers what the cKDTree path
    cannot: GREAT_CIRCLE (not a Minkowski metric), the tie-break-sensitive
    ALLOCATION/DIRECTION modes, and PROXIMITY when scipy is missing.

    ``xs`` and ``ys`` are the per-pixel 2D coordinate grids built by the caller.

    The target coordinates are gathered into flat arrays once, the per-metric
    inner loops live in ``_nearest_*`` and the pixel loop runs in parallel over
    rows in ``_bruteforce_kernel``, serialized behind ``_PARALLEL_KERNEL_LOCK``
    because the dask path calls this per chunk from worker threads.
    """
    target_rows, target_cols = _collect_targets(img, target_values)

    output = np.full(img.shape, np.nan, dtype=np.float32)
    if len(target_rows) == 0:
        return output

    txs = xs[target_rows, target_cols]
    tys = ys[target_rows, target_cols]

    if distance_metric == GREAT_CIRCLE:
        # The per-pair guards in great_circle_distance no longer run inside
        # the loop; check the grids once up front and raise the same message.
        violation = _great_circle_range_violation(xs, ys, txs, tys)
        if violation:
            raise ValueError(_GREAT_CIRCLE_RANGE_MESSAGES[violation])
        tlons, tlats, tcoslats = _great_circle_target_terms(txs, tys)
    else:
        # Placeholders: the kernel only reads these under GREAT_CIRCLE.
        tlons = tlats = tcoslats = np.empty(0, dtype=np.float64)

    with _PARALLEL_KERNEL_LOCK:
        _bruteforce_kernel(
            img, xs, ys, target_rows, target_cols, txs, tys,
            tlons, tlats, tcoslats,
            max_distance, distance_metric, process_mode, output,
        )
    return output


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

    # Round each candidate distance to float32 before comparing, matching the
    # CPU brute-force path where _distance returns np.float32(d). This makes
    # both the argmin and the max_distance range test float32 on every
    # backend: two targets whose float64 distances differ only past the
    # float32 mantissa are a tie, and the strict < keeps the first (lowest
    # flat-index) target, the tie-break documented on allocation/direction.
    # Comparing float64 here instead let the float64-closer target win on the
    # GPU while the CPU called it a tie and picked the lowest index.
    for k in range(n_targets):
        d = np.float32(_gpu_distance(
            px, target_xs[k], py, target_ys[k], distance_metric))
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

    Each chunk (plus an overlap padding of ``max_distance`` converted to
    pixels using the active distance metric) is processed on GPU
    independently.  Only valid for finite max_distance
    where the padding guarantees all relevant targets are visible within
    each overlapped chunk.
    """
    import cupy as cp

    # Overlap depth in pixels, sized from the densest coordinate spacing and
    # measured with the active distance_metric. See _halo_depth.
    pad_y, pad_x = _halo_depth(
        x_coords, y_coords, max_distance, distance_metric)

    # Build 2D coordinate grids as dask+cupy arrays matching raster chunks.
    # Each chunk is small (chunk_h x chunk_w x 8 bytes); the full grid is
    # never materialised. Chunk the 1D coords to the raster's chunking and
    # broadcast: tile/repeat + rechunk built the same grids but cost ~100
    # graph tasks per chunk (the repeat term scales with raster height),
    # while a chunk-aligned broadcast is ~1 task per chunk (issue #3132).
    x_cp = cp.asarray(x_coords, dtype=cp.float64)
    y_cp = cp.asarray(y_coords, dtype=cp.float64)
    x_da = da.from_array(x_cp, chunks=(raster.data.chunks[1],))
    y_da = da.from_array(y_cp, chunks=(raster.data.chunks[0],))
    xs = da.broadcast_to(
        x_da[None, :], raster.shape, chunks=raster.data.chunks)
    ys = da.broadcast_to(
        y_da[:, None], raster.shape, chunks=raster.data.chunks)

    # Keep the overlap depth within what map_overlap accepts on skinny rasters.
    pad_y, pad_x, (raster_data, xs, ys) = _fit_halo_to_chunks(
        pad_y, pad_x, raster.data, xs, ys)

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
        raster_data, xs, ys,
        depth=(pad_y, pad_x),
        boundary=np.nan,
        meta=cp.array((), dtype=cp.float32),
        **_dask_task_name_kwargs(_PROCESS_MODE_TASK_NAMES[process_mode]),
    )


def _inclusive_upper_bound(max_distance):
    """Exclusive cKDTree ``distance_upper_bound`` matching the float32 keep test.

    The brute-force and CUDA kernels round each candidate distance to float32
    and keep it when ``np.float32(dist) <= max_distance`` (see
    ``_process_numpy_bruteforce`` and ``_proximity_cuda_kernel``). cKDTree
    instead compares *float64* distances against an *exclusive* bound, so to
    reproduce the kernels' keep decision the bound has to account for both
    differences.

    * Float32 rounding: a target whose true float64 distance rounds *down*
      across a float32 step to a value ``<= max_distance`` is kept by the
      kernels but, against a float64-exact bound, dropped by cKDTree to NaN.
      This surfaces when ``max_distance`` sits in the float32 ulp gap just
      below a target's distance (issue #3392). The bound is therefore widened
      to the midpoint between the largest float32 ``<= max_distance`` and the
      next float32 up: the supremum of float64 distances that still round to a
      float32 value ``<= max_distance``. Nothing beyond that midpoint is pulled
      in, so no target the kernels exclude is admitted.

    * Inclusive boundary for ``p=2``: cKDTree compares *squared* distances
      internally, and the square of ``nextafter(0.0, inf)`` (the smallest
      subnormal) underflows back to 0.0, collapsing the bound to exactly
      ``max_distance`` and dropping a target sitting on the bound -- most
      visibly a target pixel itself at ``max_distance=0``. A relative bump with
      an absolute floor stays large enough to survive squaring for ``p=2`` (and
      is harmless for ``p=1``). At small ``max_distance`` the float32 midpoint
      underflows toward 0, so this floor is what keeps the boundary inclusive.
    """
    if not np.isfinite(max_distance):
        return np.inf
    # Largest float32 value <= max_distance (np.float32 rounds to nearest, so it
    # can land just above; step down one float32 ulp when it does).
    f = np.float32(max_distance)
    if f > max_distance:
        f = np.nextafter(f, np.float32(-np.inf))
    nf = np.nextafter(f, np.float32(np.inf))
    f32_bound = (float(f) + float(nf)) / 2.0
    bump = 4.0 * np.finfo(np.float64).eps * max(abs(max_distance), 1.0)
    return max(f32_bound, max_distance + bump)


def _process_numpy_kdtree(img, xs, ys, target_values, max_distance, p,
                          workers=1):
    """Exact nearest-target PROXIMITY on the CPU via scipy's cKDTree.

    ``workers`` is forwarded to ``cKDTree.query``. The eager numpy backend
    passes -1 (all cores) because it runs a single query over the whole
    raster; the dask chunk path keeps the default 1 because concurrent
    chunks already fill the cores and -1 would oversubscribe them.

    Replaces the GDAL-ported 4-pass line-sweep for EUCLIDEAN/MANHATTAN: the
    sweep propagated one nearest-target candidate between adjacent pixels,
    and on some target layouts that chain never delivers the true nearest
    target, overstating the distance by a fraction of a pixel (issue #3121).

    This function is also the bounded-dask chunk function. map_overlap pads
    edge chunks with NaN (boundary=np.nan) in both the data and the
    coordinate grids, so:

    * pad pixels are excluded as targets. Their coordinates are NaN; on
      integer rasters the NaN data pad casts to a garbage finite value that
      passes the target test, so the coordinate check is what drops them --
      the same effect the NaN coordinate distances had in the kernels.
    * pad pixels are not queried (cKDTree rejects non-finite query points);
      dask trims them from the output anyway.
    """
    finite_coords = np.isfinite(xs) & np.isfinite(ys)
    target_mask = _target_mask(img, target_values) & finite_coords

    output = np.full(img.shape, np.nan, dtype=np.float32)
    if not target_mask.any():
        return output

    tree = cKDTree(np.column_stack([ys[target_mask], xs[target_mask]]))
    query_pts = np.column_stack([ys[finite_coords], xs[finite_coords]])

    # cKDTree's distance_upper_bound is exclusive, while the brute-force and
    # CUDA kernels keep dist <= max_distance. Widen the bound so the exclusive
    # check becomes the inclusive one for both p=1 and p=2 (see
    # _inclusive_upper_bound).
    upper = _inclusive_upper_bound(max_distance)
    dists, _ = tree.query(query_pts, p=p, distance_upper_bound=upper,
                          workers=workers)

    dists = dists.astype(np.float32)
    dists[~np.isfinite(dists)] = np.nan
    output[finite_coords] = dists
    return output


def _kdtree_query_lowest_index(tree, query_pts, p, max_distance):
    """Nearest-target query that breaks ties by lowest target index.

    ``cKDTree.query`` does not promise which of several equidistant targets
    it returns, so allocation and direction can disagree with the brute-force
    and CUDA backends on a tie. Target coordinates are sorted into row-major
    (flat-index) order by ``_collect_region_targets`` before the tree is
    built -- chunk-by-chunk collection alone does not produce that order
    (issue #3090) -- so the lowest target index is the lowest flat index,
    the tie-break policy documented on ``allocation``/``direction``.

    Query the two nearest targets; wherever they are equidistant, keep the one
    with the smaller index. "Equidistant" is evaluated at float32 precision,
    matching the brute-force and CUDA kernels, which compare float32-rounded
    distances (see ``_distance``). Detecting ties with exact float64 equality
    instead let this path pick the float64-closer target on grids whose
    coordinates are not exact float64 lattice points (e.g. res 0.1), where a
    geometric tie shows up as float64 distances separated by rounding noise
    but identical in float32 -- diverging from the kernels' lowest-index pick.
    This resolves 2-way ties, which is what grid geometry produces in
    practice. A pixel tied with three or more targets relies on the two
    float64-smallest of them (the pair cKDTree returns) including the
    lowest-index one, which holds for exact ties under the row-major target
    order used here but is not strictly promised. Concretely: a 3-way float32
    tie whose lowest-flat-index member is the float64-largest of the three
    resolves to a different target here than on the brute-force/CUDA kernels.
    That takes three targets within one float32 ulp of the same distance.
    """
    # Match the inclusive dist <= max_distance check of the brute-force/CUDA
    # backends; cKDTree's bound is exclusive (see _inclusive_upper_bound).
    upper = _inclusive_upper_bound(max_distance)

    n_targets = tree.n
    if n_targets < 2:
        return tree.query(query_pts, p=p, distance_upper_bound=upper)

    dists2, idx2 = tree.query(query_pts, k=2, p=p,
                              distance_upper_bound=upper)
    dists = dists2[:, 0]
    indices = idx2[:, 0]
    # A tie exists where both neighbours are finite and equidistant at float32
    # precision. Prefer the smaller index in that case so the result is
    # independent of cKDTree's internal traversal order and of float64
    # rounding noise the kernels' float32 comparison cannot see.
    tied = (np.isfinite(dists2[:, 1])
            & (dists2[:, 1].astype(np.float32) == dists.astype(np.float32)))
    if tied.any():
        indices = np.where(tied, np.minimum(idx2[:, 0], idx2[:, 1]), indices)
    return dists, indices


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

    dists, indices = _kdtree_query_lowest_index(
        tree, query_pts, p, max_distance)

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


def _global_flat_indices(iy, ix, rows, cols, y_offsets, x_offsets, width):
    """Global flat (row-major) pixel indices of chunk-local target positions.

    The flat index is the tie-break key documented on
    ``allocation``/``direction``: among equidistant targets, the lowest one
    wins (see ``_kdtree_query_lowest_index``).
    """
    return (y_offsets[iy] + rows) * width + (x_offsets[ix] + cols)


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
    flat_cache : dict  (iy, ix) -> ndarray shape (N,), dtype int64
        Global flat (row-major) pixel index of each cached target. Used by
        ``_collect_region_targets`` to restore row-major target order so the
        documented lowest-flat-index tie-break holds across chunkings.
    """
    n_tile_y = len(chunks_y)
    n_tile_x = len(chunks_x)
    target_counts = np.zeros((n_tile_y, n_tile_x), dtype=np.int64)
    coords_cache = {}
    values_cache = {}
    flat_cache = {}
    cache_bytes = 0
    budget = int(0.25 * _available_memory_bytes())

    y_offsets = np.zeros(n_tile_y + 1, dtype=np.int64)
    np.cumsum(chunks_y, out=y_offsets[1:])
    x_offsets = np.zeros(n_tile_x + 1, dtype=np.int64)
    np.cumsum(chunks_x, out=x_offsets[1:])
    width = int(x_offsets[-1])

    for iy in range(n_tile_y):
        # Compute one chunk-row at a time so the scheduler can read the
        # row's chunks in parallel instead of one blocking .compute() per
        # chunk. Peak driver memory stays bounded to a single row of
        # chunks, preserving the streaming guarantee from gh-879.
        row_chunks = da.compute(
            *(raster.data.blocks[iy, ix] for ix in range(n_tile_x))
        )
        for ix in range(n_tile_x):
            chunk_data = row_chunks[ix]
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
                flat = _global_flat_indices(
                    iy, ix, rows, cols, y_offsets, x_offsets, width)
                entry_bytes = coords.nbytes + vals.nbytes + flat.nbytes
                if cache_bytes + entry_bytes <= budget:
                    coords_cache[(iy, ix)] = coords
                    values_cache[(iy, ix)] = vals
                    flat_cache[(iy, ix)] = flat
                    cache_bytes += entry_bytes

    total_targets = int(target_counts.sum())
    return target_counts, total_targets, coords_cache, values_cache, flat_cache


def _chunk_offsets(chunks):
    """Return cumulative offset array of length len(chunks)+1."""
    offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    np.cumsum(chunks, out=offsets[1:])
    return offsets


def _collect_region_targets(raster, jy_lo, jy_hi, jx_lo, jx_hi,
                            target_values, target_counts,
                            y_coords, x_coords,
                            y_offsets, x_offsets,
                            coords_cache, values_cache, flat_cache):
    """Collect target (y, x) coords and pixel values from chunks.

    Uses cache where available, re-reads uncached chunks via .compute().
    Returns (coords ndarray (N, 2), vals ndarray (N,)) or (None, None).

    The returned targets are sorted by global flat (row-major) pixel index.
    Chunks are visited chunk-row by chunk-column, so with more than one chunk
    column the concatenation order is chunk-major, not row-major. The KDTree
    tie-break in ``_kdtree_query_lowest_index`` keeps the lowest *tree* index
    on a tie, which only matches the documented lowest-flat-index policy of
    ``allocation``/``direction`` if the tree's target order is global
    row-major. Sorting here restores that order for both the global and the
    tiled KDTree paths.
    """
    coord_parts = []
    val_parts = []
    flat_parts = []
    width = int(x_offsets[-1])
    for iy in range(jy_lo, jy_hi):
        for ix in range(jx_lo, jx_hi):
            if target_counts[iy, ix] == 0:
                continue
            if (iy, ix) in coords_cache:
                coord_parts.append(coords_cache[(iy, ix)])
                val_parts.append(values_cache[(iy, ix)])
                flat_parts.append(flat_cache[(iy, ix)])
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
                    flat_parts.append(_global_flat_indices(
                        iy, ix, rows, cols, y_offsets, x_offsets, width))
    if not coord_parts:
        return None, None
    coords = np.concatenate(coord_parts)
    vals = np.concatenate(val_parts)
    flat = np.concatenate(flat_parts)
    order = np.argsort(flat)
    return coords[order], vals[order]


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
                       coords_cache, values_cache, flat_cache,
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
            coords_cache, values_cache, flat_cache,
        )

        if target_coords is None:
            if covers_full:
                # No targets in entire raster
                return np.full((h, w), np.nan, dtype=np.float32)
            ring += 1
            continue

        tree = cKDTree(target_coords)
        ub = max_distance if np.isfinite(max_distance) else np.inf
        dists, indices = _kdtree_query_lowest_index(tree, query_pts, p, ub)

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
                        coords_cache, values_cache, flat_cache,
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
                coords_cache, values_cache, flat_cache,
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
                         coords_cache, values_cache, flat_cache,
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
        coords_cache, values_cache, flat_cache,
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
        **_dask_task_name_kwargs(_PROCESS_MODE_TASK_NAMES[process_mode]),
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
    target_counts, total_targets, coords_cache, values_cache, flat_cache = \
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
            coords_cache, values_cache, flat_cache,
            chunks_y, chunks_x, process_mode,
        )
    else:
        return _build_tiled_kdtree(
            raster, y_coords, x_coords, target_values,
            max_distance, p, target_counts,
            coords_cache, values_cache, flat_cache,
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

    if distance_metric not in DISTANCE_METRICS:
        valid = ", ".join(sorted(DISTANCE_METRICS))
        raise ValueError(
            "Invalid distance_metric {0!r}. "
            "Valid options are: {1}.".format(distance_metric, valid)
        )
    distance_metric = DISTANCE_METRICS[distance_metric]

    target_values = np.asarray(target_values)

    # Reject non-finite explicit target_values. On numpy a pixel holding inf
    # matched target_values=[inf] and ran the search, while dask/cupy mask
    # non-finite pixels out and returned all-NaN for the same input. nan can
    # never match a pixel anyway (nan == nan is False). Fail fast instead of
    # producing backend-dependent output (issue #2850). A non-numeric dtype
    # (e.g. strings) can't index a raster either, so it gets the same error
    # rather than a downstream TypeError from np.isfinite.
    if target_values.size and (
        target_values.dtype.kind not in "iuf"
        or not np.isfinite(target_values).all()
    ):
        raise ValueError(
            "target_values must all be finite numbers, got {0!r}.".format(
                target_values.tolist()
            )
        )

    if max_distance is None:
        max_distance = np.inf

    if np.isnan(max_distance) or max_distance < 0:
        raise ValueError(
            "max_distance must be non-negative, got {0!r}.".format(max_distance)
        )

    # Get 1D coordinate arrays (these are small, just the axis coordinates)
    x_coords = raster[x].data
    y_coords = raster[y].data

    # Ensure 1D coords are numpy arrays for max_possible_distance calculation
    if da is not None and isinstance(x_coords, da.Array):
        x_coords = x_coords.compute()
    if da is not None and isinstance(y_coords, da.Array):
        y_coords = y_coords.compute()

    # The endpoint-based max distance, the dask halo, and the tiled KDTree
    # convergence check all assume monotonic 1D coords.
    _check_monotonic_coords(x_coords, y_coords, x, y)

    # Compute max_possible_distance using coordinate endpoints directly
    max_possible_distance = _distance(
        x_coords[0], x_coords[-1], y_coords[0], y_coords[-1], distance_metric
    )

    def _process_numpy(img, x_coords, y_coords, workers=1):
        # GREAT_CIRCLE is not a Minkowski metric, so the cKDTree query below
        # cannot answer it. Use an exact brute-force search instead (matches
        # the GPU kernel). The branch lives in plain Python rather than
        # inside the numba kernel so the metric choice is a runtime value,
        # never frozen into a compiled specialization.
        if distance_metric == GREAT_CIRCLE:
            return _process_numpy_bruteforce(
                img, x_coords, y_coords, target_values,
                np.float32(max_distance), distance_metric, process_mode,
            )
        # ALLOCATION and DIRECTION pick a single target per pixel, so a tie
        # between two equidistant targets must resolve the same way on every
        # backend. cKDTree.query makes no promise about which of several
        # equidistant targets it returns, so route those modes through the
        # brute-force search, which keeps the lowest-flat-index target on a
        # tie (see the `allocation`/`direction` docstrings).
        if process_mode in (ALLOCATION, DIRECTION):
            return _process_numpy_bruteforce(
                img, x_coords, y_coords, target_values,
                np.float32(max_distance), distance_metric, process_mode,
            )
        # PROXIMITY with EUCLIDEAN/MANHATTAN: exact cKDTree query. The
        # GDAL-ported line-sweep that used to run here could miss the true
        # nearest target by a fraction of a pixel (issue #3121). Without
        # scipy, fall back to the brute-force kernel, which is also exact.
        if cKDTree is None:
            warnings.warn(
                "proximity: scipy is not installed; falling back to an "
                "exact brute-force search, which is slower on large "
                "rasters.",
                UserWarning,
                stacklevel=2,
            )
            return _process_numpy_bruteforce(
                img, x_coords, y_coords, target_values,
                np.float32(max_distance), distance_metric, process_mode,
            )
        return _process_numpy_kdtree(
            img, x_coords, y_coords, target_values, max_distance,
            2 if distance_metric == EUCLIDEAN else 1, workers=workers,
        )

    def _process_dask(raster, xs, ys):

        # Rechunk into a local variable instead of reassigning raster.data,
        # which would mutate the caller's input DataArray (issue #2847).
        data = raster.data
        if max_distance >= max_possible_distance:
            # consider all targets in the whole raster
            # the data array is computed at once,
            # make sure your data fit your memory
            height, width = raster.shape
            data = data.rechunk({0: height, 1: width})
            xs = xs.rechunk({0: height, 1: width})
            ys = ys.rechunk({0: height, 1: width})
            pad_y = pad_x = 0
        else:
            pad_y, pad_x = _halo_depth(
                x_coords, y_coords, max_distance, distance_metric)
            # Fold into the same local ``data`` the map_overlap call uses, not
            # ``raster.data``: assigning back to ``raster.data`` would both
            # mutate the caller's input (issue #2847) and leave map_overlap
            # reading the stale, un-folded array while xs/ys are folded, so
            # chunking disagrees and in-range targets in adjacent chunks drop
            # to NaN (issue #2908).
            pad_y, pad_x, (data, xs, ys) = _fit_halo_to_chunks(
                pad_y, pad_x, data, xs, ys)

        out = da.map_overlap(
            _process_numpy,
            data, xs, ys,
            depth=(pad_y, pad_x),
            boundary=np.nan,
            meta=np.array((), dtype=np.float32),
            **_dask_task_name_kwargs(_PROCESS_MODE_TASK_NAMES[process_mode]),
        )
        return out

    if isinstance(raster.data, np.ndarray):
        # numpy case - create full coordinate arrays as numpy
        xs = np.tile(x_coords, raster.shape[0]).reshape(raster.shape)
        ys = np.repeat(y_coords, raster.shape[1]).reshape(raster.shape)
        result = _process_numpy(raster.data, xs, ys, workers=-1)

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

                # Unbounded: convert to dask+numpy for the KDTree
                # (KDTree is CPU-only; O(N log T) beats brute-force O(NT))
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

                # Build 2D coordinate grids as dask arrays. Chunk the 1D
                # coords to the raster's chunking and broadcast: tile/repeat
                # + rechunk built the same grids but cost ~100 graph tasks
                # per chunk (the repeat term scales with raster height),
                # while a chunk-aligned broadcast is ~1 task per chunk
                # (issue #3132).
                x_coords_da = da.from_array(
                    x_coords, chunks=(raster.data.chunks[1],))
                y_coords_da = da.from_array(
                    y_coords, chunks=(raster.data.chunks[0],))
                xs = da.broadcast_to(
                    x_coords_da[None, :], raster.shape,
                    chunks=raster.data.chunks)
                ys = da.broadcast_to(
                    y_coords_da[:, None], raster.shape,
                    chunks=raster.data.chunks)
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
    target_values: list = None,
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

    Proximity supports NumPy, CuPy, Dask with NumPy, and Dask with CuPy
    backed xarray DataArray. The return values of proximity are of the same
    type as the input type: a NumPy-backed input gives a NumPy-backed result,
    a CuPy-backed input gives a CuPy-backed result, and a Dask-backed input
    gives a Dask-backed result.

    The NumPy-backed implementation answers EUCLIDEAN and MANHATTAN with an
    exact nearest-target k-d tree query (scipy's cKDTree), falling back to an
    exact brute-force search when scipy is not installed. GREAT_CIRCLE is not
    a Minkowski metric, so it always uses the brute-force search. Earlier
    versions used a GDAL-ported 3x3 line-sweep here, which could overstate
    the distance by a fraction of a pixel on some target layouts
    (issue #3121).
    The Dask-backed implementation depends on `max_distance`: when it is
    smaller than the maximum possible distance within the raster, proximity
    is computed chunk by chunk with `dask.map_overlap`, expanding each
    chunk's borders to cover `max_distance`; otherwise the nearest targets
    are found with a KDTree query over all target pixels.

    Parameters
    ----------
    raster : xr.DataArray or xr.Dataset
        2D array image with `raster.shape` = (height, width).
        If a Dataset is passed, the function is applied to each
        data variable independently, returning a Dataset.
        The 1D ``x`` and ``y`` coordinates must be monotonic (strictly
        increasing or strictly decreasing); a non-monotonic axis raises
        a ValueError.

    x : str, default='x'
        Name of x-coordinates.

    y : str, default='y'
        Name of y-coordinates.

    target_values: list
        Target pixel values to measure the distance from. If this option
        is not provided, proximity will be computed from non-zero pixel
        values. All entries must be finite; a non-finite value (inf or
        nan) raises ValueError.

    max_distance: float, default=np.inf
        The maximum distance to search. Proximity distances greater than
        this value will be set to NaN. Must be a non-negative, non-NaN
        number; a negative or NaN value raises a ValueError.
        Should be given in the same distance unit as input.
        For example, if input raster is in lat-lon and distances between points
        within the raster is calculated using Euclidean distance metric,
        `max_distance` should also be provided in lat-lon unit.
        If using Great Circle distance metric, and thus all distances is in
        meters, `max_distance` should also be provided in meters.

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
        An unrecognized value raises ValueError.

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

    if target_values is None:
        target_values = []

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
        attrs=raster.attrs,
    )
    # Drop any internal dask op name so all backends agree on .name=None.
    result.name = None

    return result


@supports_dataset
def allocation(
    raster: xr.DataArray,
    x: str = "x",
    y: str = "y",
    target_values: list = None,
    max_distance: float = np.inf,
    distance_metric: str = "EUCLIDEAN",
) -> xr.DataArray:
    """
    Calculates, for all pixels in the input raster, the nearest source
    based on a set of target values and a distance metric.

    This function attempts to produce the value of nearest feature of all
    pixels in the image to a set of pixels in the source image. The
    following options are used to define the behavior of the function.
    By default all non-zero pixels in `raster.values` will be considered
    as "target", and all allocation will be computed in pixels.

    Allocation supports NumPy, CuPy, Dask with NumPy, and Dask with CuPy
    backed xarray DataArray. The return values of `allocation` are of the
    same type as the input type: a NumPy-backed input gives a NumPy-backed
    result, a CuPy-backed input gives a CuPy-backed result, and a
    Dask-backed input gives a Dask-backed result.

    The NumPy-backed implementation uses an exact brute-force nearest-target
    search: `allocation` must pick a single target per pixel, so its
    tie-break has to match every other backend (see Tie-breaking below).
    The Dask-backed implementation depends on `max_distance`: when it is
    smaller than the maximum possible distance within the raster,
    `allocation` is computed chunk by chunk with `dask.map_overlap`,
    expanding each chunk's borders to cover `max_distance`; otherwise the
    nearest targets are found with a KDTree query over all target pixels.

    Tie-breaking: when two or more targets are equidistant from a pixel, the
    target with the lowest flat (row-major) index wins, i.e. the first target
    encountered when scanning the raster top-to-bottom and left-to-right.
    Distances are compared at float32 precision (the precision of the
    output), so targets whose distances differ only beyond the float32
    mantissa count as equidistant. This policy is identical across all
    backends (numpy, cupy, dask+numpy, dask+cupy), so the allocated value is
    deterministic regardless of which backend computes it.

    Parameters
    ----------
    raster : xr.DataArray or xr.Dataset
        2D array of target data.
        If a Dataset is passed, the function is applied to each
        data variable independently, returning a Dataset.
        The 1D ``x`` and ``y`` coordinates must be monotonic (strictly
        increasing or strictly decreasing); a non-monotonic axis raises
        a ValueError.

    x : str, default='x'
        Name of x-coordinates.

    y : str, default='y'
        Name of y-coordinates.

    target_values : list
        Target pixel values to measure the distance from. If this option
        is not provided, allocation will be computed from non-zero pixel
        values. All entries must be finite; a non-finite value (inf or
        nan) raises ValueError.

    max_distance: float, default=np.inf
        The maximum distance to search. Proximity distances greater than
        this value will be set to NaN. Must be a non-negative, non-NaN
        number; a negative or NaN value raises a ValueError.
        Should be given in the same distance unit as input.
        For example, if input raster is in lat-lon and distances between points
        within the raster is calculated using Euclidean distance metric,
        `max_distance` should also be provided in lat-lon unit.
        If using Great Circle distance metric, and thus all distances is in
        meters, `max_distance` should also be provided in meters.

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
        An unrecognized value raises ValueError.

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
        array([[1., 1., 1., 2., 2.],
               [1., 1., 1., 2., 2.],
               [1., 1., 3., 2., 2.],
               [1., 3., 3., 3., 2.],
               [3., 3., 3., 3., 3.]], dtype=float32)
        Coordinates:
          * y        (y) int64 4 3 2 1 0
          * x        (x) int64 0 1 2 3 4
    """

    if target_values is None:
        target_values = []

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

    result = xr.DataArray(
        allocation_img,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )
    # Drop any internal dask op name so all backends agree on .name=None.
    result.name = None
    return result


@supports_dataset
def direction(
    raster: xr.DataArray,
    x: str = "x",
    y: str = "y",
    target_values: list = None,
    max_distance: float = np.inf,
    distance_metric: str = "EUCLIDEAN",
) -> xr.DataArray:
    """
    Calculates, for all pixels in the input raster, the direction to
    nearest source based on a set of target values and a distance metric.

    This function attempts to calculate for each cell, the direction,
    in degrees, to the nearest source. The output values are based on
    compass directions, where 90 is for the east, 180 for the south,
    270 for the west, 360 for the north, and 0 for the source cell
    itself. The following options are used to define the behavior of
    the function. By default all non-zero pixels in `raster.values`
    will be considered as "target", and all direction will be computed
    in pixels.

    Direction supports NumPy, CuPy, Dask with NumPy, and Dask with CuPy
    backed xarray DataArray. The return values of `direction` are of the
    same type as the input type: a NumPy-backed input gives a NumPy-backed
    result, a CuPy-backed input gives a CuPy-backed result, and a
    Dask-backed input gives a Dask-backed result.

    The NumPy-backed implementation uses an exact brute-force nearest-target
    search: `direction` must pick a single target per pixel, so its
    tie-break has to match every other backend (see Tie-breaking below).
    The Dask-backed implementation depends on `max_distance`: when it is
    smaller than the maximum possible distance within the raster,
    `direction` is computed chunk by chunk with `dask.map_overlap`,
    expanding each chunk's borders to cover `max_distance`; otherwise the
    nearest targets are found with a KDTree query over all target pixels.

    Tie-breaking: when two or more targets are equidistant from a pixel, the
    direction is computed toward the target with the lowest flat (row-major)
    index, i.e. the first target encountered when scanning the raster
    top-to-bottom and left-to-right. Distances are compared at float32
    precision (the precision of the output), so targets whose distances
    differ only beyond the float32 mantissa count as equidistant. This policy
    is identical across all backends (numpy, cupy, dask+numpy, dask+cupy), so
    the reported direction is deterministic regardless of which backend
    computes it.

    Parameters
    ----------
    raster : xr.DataArray or xr.Dataset
        2D array image with `raster.shape` = (height, width).
        If a Dataset is passed, the function is applied to each
        data variable independently, returning a Dataset.
        The 1D ``x`` and ``y`` coordinates must be monotonic (strictly
        increasing or strictly decreasing); a non-monotonic axis raises
        a ValueError.

    x : str, default='x'
        Name of x-coordinates.

    y : str, default='y'
        Name of y-coordinates.

    target_values: list
        Target pixel values to measure the distance from. If this
        option is not provided, proximity will be computed from
        non-zero pixel values. All entries must be finite; a non-finite
        value (inf or nan) raises ValueError.

    max_distance: float, default=np.inf
        The maximum distance to search. Proximity distances greater than
        this value will be set to NaN. Must be a non-negative, non-NaN
        number; a negative or NaN value raises a ValueError.
        Should be given in the same distance unit as input.
        For example, if input raster is in lat-lon and distances between points
        within the raster is calculated using Euclidean distance metric,
        `max_distance` should also be provided in lat-lon unit.
        If using Great Circle distance metric, and thus all distances is in
        meters, `max_distance` should also be provided in meters.

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
        An unrecognized value raises ValueError.

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

    if target_values is None:
        target_values = []

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
        attrs=raster.attrs,
    )
    # Drop any internal dask op name so all backends agree on .name=None.
    result.name = None
    return result
