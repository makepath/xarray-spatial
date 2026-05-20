# Polygonize algorithm creates vector polygons for connected regions of pixels
# that share the same pixel value in a raster.  It is a raster to vector
# converter.
#
# Algorithm here uses compass directions for clarity, so +x direction is East
# and +y direction is North.  2D arrays are flattened to 1D to make maths of
# moving through the grid easier/faster to calculate.  2D array have shape
# (ny, nx) so 1D flattened array has length n = nx*ny.  Position within a 1D
# array is denoted by index ij = i + j*nx, so i = ij % nx and j = ij // nx.
# Direction of motion through the grid is represented by forward and left
# which are the values to add to ij to move one pixel in the forward and left
# direction respectively.  forward is +1 for E, -1 for W, +nx for N and -nx
# for S.
#
# There are two main stages in the algorithm.  Firstly the raster is divided
# into connected regions that contain adjacent pixels of the same value.  Each
# region is labelled with a unique integer ID starting at 1.  Regions
# corresponding to masked out pixels are all given the same region ID of 0.
#
# The second stage identifies where polygon exteriors and holes start, and
# follows each of these boundaries around the raster, keeping to the edge of
# the region.  Holes are grouped together with their enclosing exterior
# boundary.
#
# The points of exterior boundaries are ordered in an anticlockwise manner,
# those of hole boundaries in a clockwise manner.  This assumes that both the
# x and y coordinates are monotonically increasing or decreasing.

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numba as nb
import numpy as np
import xarray as xr

try:
    import dask
    import dask.array as da
except ImportError:
    dask = None
    da = None

try:
    import cupy
except ImportError:
    cupy = None

from .utils import (
    ArrayTypeFunctionMapping,
    _validate_raster,
    is_cupy_array,
    ngjit,
)

_regions_dtype = np.uint32
_visited_dtype = np.uint8


def generated_jit(function=None, cache=False,
                  pipeline_class=None, **options):
    """
    This decorator allows flexible type-based compilation
    of a jitted function.  It works as `@jit`, except that the decorated
    function is called at compile-time with the *types* of the arguments
    and should return an implementation function for those types.
    """
    from numba.extending import overload
    jit_options = dict()
    if pipeline_class is not None:
        jit_options['pipeline_class'] = pipeline_class
    jit_options['cache'] = cache
    jit_options |= options

    if function is not None:
        overload(function, jit_options=jit_options,
                    strict=False)(function)
        return function
    else:
        def wrapper(func):
            overload(func, jit_options=jit_options,
                        strict=False)(func)
            return func
        return wrapper


class Turn(Enum):
    Left = -1
    Straight = 0
    Right = 1


@ngjit
def _diff_row(ij0: int, ij1: int, nx: int) -> bool:
    return (ij0 // nx) != (ij1 // nx)


@ngjit
def _outside_domain(ij: int, n: int) -> bool:
    return ij < 0 or ij >= n


@ngjit
def _min_and_max(value0, value1):
    if value0 < value1:
        return value0, value1
    else:
        return value1, value0


# Follow the boundary of a polygon around the raster starting from the
# specified ij pixel.  If hole is True the start is on the N edge of the pixel
# facing W, and if hole is False the start is on the S edge of the pixel
# facing E.
#
# There are two passes.  First pass determines the number of points and
# allocates a numpy array big enough to take the points, the second pass saves
# the points.  This is better than a single pass that repeatedly reallocates
# space for the increasing number of points.
#
# Returns the region ID and the 2D array of boundary points.  The last
# boundary point is the same as the first.
@ngjit
def _follow(
    regions: np.ndarray,  # _regions_dtype, shape (nx*ny,)
    visited: np.ndarray,  # _visited_dtype, shape (nx*ny,)
    nx: int,
    ny: int,
    ij: int,
    hole: bool,
) -> Tuple[int, np.ndarray]:
    region = regions[ij]
    n = nx*ny
    points = None  # Declare before loop for numba

    for pass_ in range(2):
        prev_forward = 0  # Start with an invalid direction.
        if hole:
            forward = -1  # Facing W along N edge.
            left = -nx
        else:
            forward = 1  # Facing E along S edge.
            left = nx

        start_forward = forward
        start_ij = ij
        npoints = 0

        while True:
            if pass_ == 1:
                if forward == 1 and not hole:
                    # Mark pixel so that it will not be considered a future
                    # non-hole starting pixel.
                    visited[ij] |= 1
                elif forward == -1 and ij+nx < n:
                    # Mark pixel so that is will not be considered a future
                    # hole starting pixel.
                    visited[ij+nx] |= 2

            if prev_forward != forward:
                if pass_ == 1:
                    # Add point.
                    i = ij % nx
                    j = ij // nx
                    if forward == -1:
                        i += 1
                        j += 1
                    elif forward == nx:
                        i += 1
                    elif forward == -nx:
                        j += 1
                    points[2*npoints] = i
                    points[2*npoints+1] = j
                npoints += 1

            prev_forward = forward
            ijnext = ij + forward
            ijnext_right = ijnext - left

            # Determine direction of turn.
            if abs(forward) == 1:  # Facing E (forward == 1) or W (forward -1)
                if _diff_row(ij, ijnext, nx):
                    turn = Turn.Left
                elif (not _outside_domain(ijnext_right, n) and
                        regions[ijnext_right] == region):
                    turn = Turn.Right
                elif regions[ijnext] == region:
                    turn = Turn.Straight
                else:
                    turn = Turn.Left
            else:  # Facing N (forward == nx) or S (forward == -nx)
                if _outside_domain(ijnext, n):
                    turn = Turn.Left
                elif (not _diff_row(ijnext, ijnext_right, nx) and
                        regions[ijnext_right] == region):
                    turn = Turn.Right
                elif regions[ijnext] == region:
                    turn = Turn.Straight
                else:
                    turn = Turn.Left

            # Apply turn.
            if turn == Turn.Straight:
                ij = ijnext
            elif turn == Turn.Left:
                prev_forward = forward
                forward = left
                left = -prev_forward
            else:  # Turn.Right
                prev_forward = forward
                forward = -left
                left = prev_forward
                ij = ijnext_right

            # Finished boundary when returned to start.
            if ij == start_ij and forward == start_forward:
                break

        if pass_ == 0:
            # End of first pass.
            points = np.empty(2*(npoints+1))  # Note extra point at end.

    points = points.reshape((-1, 2))
    points[-1] = points[0]  # End point the same as start point.
    return region, points


# Generator of numba-compatible comparison functions for values.
# If both values are integers use a fast equality operator, otherwise use a
# slower floating-point comparison like numpy.isclose.
@generated_jit(nogil=True, nopython=True)
def _is_close(
    reference: Union[int, float],
    value: Union[int, float],
) -> bool:
    if (isinstance(reference, nb.types.Integer) and
            isinstance(value, nb.types.Integer)):
        return lambda reference, value: value == reference
    else:
        atol = 1e-8
        rtol = 1e-5
        return lambda reference, value: \
            abs(value - reference) <= (atol + rtol*abs(reference))


# Calculate region connectivity for the specified values raster and optional
# mask raster.  Each region is labelled with a unique integer ID starting at
# 1.  Regions corresponding to masked out pixels are all given the same region
# ID of 0.
#
# Algorithm walks through the raster from ij = 0 to ij = n-1.  If connectivity
# is 4 it uses the values and region IDs of the pixels to the W and S to
# determine the region ID of the current pixel.  This may involve using a new
# region ID, or the region ID of the W or S pixel, or merging the region IDs
# of the W and S pixels together if they are joined together as a single
# region.  Merging information is stored in a region_lookup dict for resolving
# later on.
#
# For a connectivity of 8, the values and region IDs of the pixels to the SW
# and SE are also considered.
#
# If the algorithm requires too many unique region IDs it will raise a
# RuntimeError advising the user to split up their raster into chunks, e.g.
# using dask.
@ngjit
def _calculate_regions(
    values: np.ndarray,          # Could be ints or floats, shape (nx*ny,)
    mask: Optional[np.ndarray],  # shape (nx*ny,)
    connectivity_8: bool,
    nx: int,
    ny: int,
) -> np.ndarray:  # _regions_dtype, shape (nx*ny,)
    # Array of regions to return, integers starting at zero.
    regions = np.zeros_like(values, dtype=_regions_dtype)

    # Non-zero entries of region_lookup refer to region of first pixel (lowest
    # ij) in region.
    lookup_size = max(64, nx, ny)  # Guess reasonable initial size.
    region_lookup = np.zeros(lookup_size, dtype=_regions_dtype)

    max_region = np.iinfo(_regions_dtype).max

    region = 0
    for ij in range(nx*ny):
        if mask is not None and not mask[ij]:
            regions[ij] = 0  # Masked out pixels are always region 0.
        else:
            # Is pixel in same region as pixel to W?
            matches_W = \
                (ij % nx > 0 and                         # i > 0
                    (mask is None or mask[ij-1]) and     # W pixel in mask
                    _is_close(values[ij], values[ij-1]))

            if matches_W:
                region_W = regions[ij-1]

            # Is pixel in same region as pixel to S?
            matches_S = \
                (ij >= nx and                             # j > 0
                    (mask is None or mask[ij-nx]) and     # S pixel in mask
                    _is_close(values[ij], values[ij-nx]))

            if matches_S:
                region_S = regions[ij-nx]

            # If connectivity is 8, need to consider pixels to SW and SE.
            # Only need to consider SW pixel if it is in a different region to
            # W pixel; similar applies to SE and S pixels.
            if connectivity_8 and ij >= nx:
                if (not matches_W and ij % nx > 0 and
                        (mask is None or mask[ij-nx-1]) and
                        _is_close(values[ij], values[ij-nx-1])):
                    matches_W = True
                    region_W = regions[ij-nx-1]

                if (not matches_S and ij % nx < nx-1 and
                        (mask is None or mask[ij-nx+1]) and
                        _is_close(values[ij], values[ij-nx+1])):
                    matches_S = True
                    region_S = regions[ij-nx+1]

            # Set region for this pixel, and merge regions if necessary.
            if matches_W and matches_S:
                lower_region, upper_region = _min_and_max(region_W, region_S)
                regions[ij] = lower_region
                if lower_region != upper_region:
                    region_lookup = _merge_regions(
                        region_lookup, lower_region, upper_region)
            elif matches_W:
                regions[ij] = region_W
            elif matches_S:
                regions[ij] = region_S
            else:
                if region == max_region:
                    raise RuntimeError(
                        "polygonize generates too many polygons, "
                        "split your raster into smaller chunks."
                    )
                region += 1
                regions[ij] = region

    # A number of region IDs may refer to the same region via region_lookup.
    # Here create new region_lookup to map from current region ID to the
    # region ID of the first pixel (lowest ij) in the region.
    max_region = region + 1
    new_region_lookup = np.empty(max_region, dtype=_regions_dtype)
    n_region_lookup = len(region_lookup)
    new_region = 0
    for i in range(max_region):
        target = region_lookup[i] if i < n_region_lookup else 0
        if target == 0:
            new_region_lookup[i] = new_region
            new_region += 1
        else:
            new_region_lookup[i] = new_region_lookup[target]

    region_lookup = new_region_lookup

    # Update regions using region_lookup.
    for ij in range(nx*ny):
        regions[ij] = region_lookup[regions[ij]]

    return regions


@ngjit
def _merge_regions(
    region_lookup: Dict[int, int],
    lower_region: int,
    upper_region: int,
) -> Dict[int, int]:
    if upper_region >= len(region_lookup):
        old_size = len(region_lookup)
        new_size = max(upper_region + 1, 2*old_size)

        # numba-compatible resize of region_lookup.
        old_region_lookup = region_lookup
        region_lookup = np.empty(new_size, dtype=_regions_dtype)
        region_lookup[:old_size] = old_region_lookup
        region_lookup[old_size:] = 0

    # Will be setting region_lookup[upper_region].  If this already has a
    # non-zero value, need to ensure that overwriting it does not result in
    # the region being left unconnected, hence the while-loop.
    while True:
        prev = region_lookup[upper_region]
        repeat = (prev != 0 and prev != lower_region)
        if repeat:
            lower_region, prev = _min_and_max(lower_region, prev)

        region_lookup[upper_region] = lower_region

        if not repeat:
            break

        upper_region = prev

    return region_lookup


# For debugging purposes only.
@ngjit
def _print_regions(regions, region_lookup, nx, ny):
    print("---------- regions ----------")
    print(regions.reshape((ny, nx))[::-1])
    print("----------------------------------")
    print("lookup", region_lookup)
    print("---------------------------")


@ngjit
def _transform_points(
    pts: np.ndarray,        # float64, shape (?, 2)
    transform: np.ndarray,  # float64, shape (6,)
):
    # Apply transform in place.
    for i in range(len(pts)):
        x = transform[0]*pts[i, 0] + transform[1]*pts[i, 1] + transform[2]
        y = transform[3]*pts[i, 0] + transform[4]*pts[i, 1] + transform[5]
        pts[i, 0] = x
        pts[i, 1] = y


@ngjit
def _scan(
    regions: np.ndarray,              # _regions_dtype, shape (nx*ny,)
    values: np.ndarray,               # shape (nx*ny,)
    mask: Optional[np.ndarray],       # shape (nx*ny,)
    connectivity_8: bool,
    transform: Optional[np.ndarray],  # shape (6,)
    nx: int,
    ny: int,
) -> Tuple[List[Union[int, float]], List[List[np.ndarray]]]:
    # Visited flags used to denote where boundaries have already been
    # followed and hence are not future start positions.
    visited = np.zeros_like(values, dtype=_visited_dtype)

    region_done = 0   # Always consider regions in increasing order.
    column = []       # Pixel values corresponding to regions > 0.
    polygons = []     # Polygons corresponding to regions > 0.

    # Identify start pixels and follow their region boundaries, adding them to
    # the correct polygons as exterior boundaries or holes.
    for ij in range(nx*ny):
        if not (visited[ij] & 1) and regions[ij] == region_done+1:
            # Follow exterior of polygon on S side of pixel ij facing E.
            region, points = _follow(regions, visited, nx, ny, ij, False)
            if transform is not None:
                _transform_points(points, transform)
            column.append(values[ij])
            polygons.append([points])
            # Now len(polygons) == region
            region_done = region

        if (ij >= nx and not (visited[ij] & 2) and
                regions[ij] != regions[ij-nx] and regions[ij-nx] != 0):
            # Follow hole of polygon on N side of pixel ij-nx facing W.
            region, points = _follow(regions, visited, nx, ny, ij-nx, True)
            if transform is not None:
                _transform_points(points, transform)
            # Polygon index is one less than region as region 0 has no
            # polygons.
            polygons[region-1].append(points)

    return column, polygons


def _to_awkward(
    column: List[Union[int, float]],
    polygon_points: List[np.ndarray],
):
    import awkward as ak
    ak_array = ak.Array(polygon_points)
    return column, ak_array


def _detect_raster_crs(raster: xr.DataArray):
    """Detect the CRS of an input raster for output georeferencing.

    Mirrors the resolution order in
    ``xrspatial.reproject._crs_utils._detect_source_crs`` without
    pulling pyproj as a hard dependency: returns the raw attribute
    value (string / int / pyproj.CRS / rioxarray CRS) so the caller can
    hand it straight to ``GeoDataFrame.set_crs``, which performs its own
    parsing.

    Resolution order:

    1. ``raster.attrs['crs']`` (xrspatial.geotiff convention)
    2. ``raster.attrs['crs_wkt']``
    3. ``raster.rio.crs`` (rioxarray, if installed)
    4. ``None``
    """
    crs_attr = raster.attrs.get('crs')
    if crs_attr is not None:
        return crs_attr

    crs_wkt = raster.attrs.get('crs_wkt')
    if crs_wkt is not None:
        return crs_wkt

    try:
        rio_crs = raster.rio.crs
        if rio_crs is not None:
            return rio_crs
    except Exception:
        pass

    return None


def _to_geopandas(
    column: List[Union[int, float]],
    polygon_points: List[np.ndarray],
    column_name: str,
    crs=None,
):
    import geopandas as gpd
    import shapely
    from shapely.geometry import Polygon

    # Batch-construct hole-free polygons via the shapely 2.0
    # linearrings -> polygons pipeline (both are C-level batch ops).
    # The pre-#2060 fallback to ``[Polygon(...) for pts in ...]`` for
    # shapely < 2 is gone now that the package pins ``shapely>=2.0``.
    no_holes = [i for i, pts in enumerate(polygon_points)
                if len(pts) == 1]

    if len(no_holes) == len(polygon_points):
        # All hole-free: batch create LinearRings then Polygons.
        rings = [shapely.linearrings(pts[0])
                 for pts in polygon_points]
        polygons = list(shapely.polygons(rings))
    else:
        polygons = [None] * len(polygon_points)
        if no_holes:
            rings = [shapely.linearrings(polygon_points[i][0])
                     for i in no_holes]
            batch = shapely.polygons(rings)
            for idx, poly in zip(no_holes, batch):
                polygons[idx] = poly
        for i, pts in enumerate(polygon_points):
            if polygons[i] is None:
                polygons[i] = Polygon(pts[0], pts[1:])

    df = gpd.GeoDataFrame({column_name: column, "geometry": polygons})
    if crs is not None:
        # GeoPandas accepts pyproj.CRS, EPSG ints, "EPSG:XXXX" strings,
        # WKT strings, and the CRS objects rioxarray exposes; let it
        # parse whatever the raster carried.
        try:
            df = df.set_crs(crs)
        except Exception:
            # Don't fail the whole call if the CRS value is unparseable;
            # GeoDataFrame is still functional, just without CRS.
            pass
    return df


def _to_spatialpandas(
    column: List[Union[int, float]],
    polygon_points: List[np.ndarray],
    column_name: str,
):
    from spatialpandas import GeoDataFrame
    from spatialpandas.geometry import PolygonArray

    # spatialpandas expects 1d numpy arrays.
    for i, arrays in enumerate(polygon_points):
        polygon_points[i] = \
            list(map(lambda array: np.reshape(array, -1), arrays))

    df = GeoDataFrame({
        column_name: column, "geometry": PolygonArray(polygon_points)})
    return df


def _to_geojson(
    column: List[Union[int, float]],
    polygon_points: List[List[np.ndarray]],
    column_name: str,
):
    """Convert to GeoJSON FeatureCollection dict."""
    features = []
    for value, rings in zip(column, polygon_points):
        coords = [ring.tolist() for ring in rings]
        features.append({
            "type": "Feature",
            "properties": {column_name: value},
            "geometry": {"type": "Polygon", "coordinates": coords},
        })
    return {"type": "FeatureCollection", "features": features}


def _polygonize_numpy(
    values: np.ndarray,
    mask: Optional[np.ndarray],
    connectivity_8: bool,
    transform: Optional[np.ndarray],
) -> Tuple[List[Union[int, float]], List[List[np.ndarray]]]:

    ny, nx = values.shape

    # Mask NaN pixels for float arrays, matching the CuPy backend.
    if np.issubdtype(values.dtype, np.floating):
        nan_mask = ~np.isnan(values)
        if mask is not None:
            mask = mask & nan_mask
        else:
            mask = nan_mask

    if nx == 1:
        # Algorithm requires nx > 1 to differentiate between facing E
        # (forward == 1) and facing N (forward == nx), so add extra column to
        # values array and mask the column out.
        nx = 2
        values = np.hstack((values, np.empty_like(values)))
        if mask is not None:
            mask = np.hstack((mask, np.zeros_like(mask)))
        else:
            mask = np.zeros_like(values, dtype=bool)
            mask[:, 0] = True

    values_flat = values.ravel()
    mask_flat = mask.ravel() if mask is not None else None

    regions = _calculate_regions(values_flat, mask_flat, connectivity_8, nx, ny)
    column, polygon_points = _scan(
        regions, values_flat, mask_flat, connectivity_8, transform, nx, ny)

    return column, polygon_points


# CW angle order for grid-aligned directions (y increases downward).
_DIR_ANGLE = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}


def _calculate_regions_cupy(data, mask_data, connectivity_8):
    """CuPy GPU backend for connected-component labeling.

    Uses cupyx.scipy.ndimage.label per unique value to produce a regions
    array compatible with _scan.  Returns a cupy uint32 2D array.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import label as cp_label

    if connectivity_8:
        structure = cp.ones((3, 3), dtype=cp.int32)
    else:
        structure = cp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=cp.int32)

    regions = cp.zeros(data.shape, dtype=cp.uint32)

    # Build valid mask (unmask + handle float NaN).
    is_float = cp.issubdtype(data.dtype, cp.floating)
    if mask_data is not None:
        valid = cp.asarray(mask_data, dtype=bool)
        if is_float:
            valid &= ~cp.isnan(data)
    else:
        valid = ~cp.isnan(data) if is_float else None

    unique_vals = data[valid] if valid is not None else data.ravel()
    unique_vals = cp.unique(unique_vals)

    uid = 1
    for v in unique_vals:
        bin_mask = (data == v)
        if valid is not None:
            bin_mask &= valid
        labeled, n_features = cp_label(bin_mask, structure=structure)
        if n_features == 0:
            continue
        # Vectorized assignment: offset labeled region IDs by (uid - 1) so
        # label 1 → uid, label 2 → uid+1, etc.  Single kernel, no Python loop.
        where = labeled > 0
        regions[where] = (labeled[where].astype(cp.uint32) +
                          cp.uint32(uid - 1))
        uid += n_features

    return regions


@ngjit
def _renumber_regions(regions, nx, ny):
    """Renumber regions so IDs are sequential in raster-scan order.

    _scan expects region 1 to have the smallest ij, region 2 the next, etc.
    GPU CCL numbers regions per-value, not spatially.  This pass assigns
    new sequential IDs in the order regions are first encountered.
    """
    n = nx * ny
    max_old = 0
    for i in range(n):
        if regions[i] > max_old:
            max_old = regions[i]

    # Map from old region ID to new sequential ID.
    remap = np.zeros(max_old + 1, dtype=_regions_dtype)
    new_id = _regions_dtype(0)
    for ij in range(n):
        old = regions[ij]
        if old == 0:
            continue
        if remap[old] == 0:
            new_id += 1
            remap[old] = new_id
        regions[ij] = remap[old]

    return regions


def _polygonize_cupy(data, mask_data, connectivity_8, transform):
    """Hybrid GPU/CPU: GPU CCL, CPU boundary tracing."""
    np_data = cupy.asnumpy(data)
    np_mask = cupy.asnumpy(mask_data) if mask_data is not None else None
    ny, nx = np_data.shape
    if nx == 1:
        # Edge case: fall back to full numpy path (pads array).
        return _polygonize_numpy(np_data, np_mask, connectivity_8, transform)
    regions_gpu = _calculate_regions_cupy(data, mask_data, connectivity_8)
    regions = cupy.asnumpy(regions_gpu).ravel()
    # Renumber into raster-scan order for _scan compatibility.
    regions = _renumber_regions(regions, nx, ny)
    column, polygon_points = _scan(
        regions, np_data.ravel(),
        np_mask.ravel() if np_mask is not None else None,
        connectivity_8, transform, nx, ny)
    return column, polygon_points


def _to_numpy(arr):
    """Convert array to numpy, handling cupy arrays."""
    if cupy is not None and isinstance(arr, cupy.ndarray):
        return cupy.asnumpy(arr)
    return np.asarray(arr)


def _polygonize_chunk(block, mask_block, connectivity_8, row_offset, col_offset,
                      ny_total, nx_total):
    """Run _polygonize_numpy on a single chunk, offset coords to global space.

    Polygons are classified as "interior" (no vertex on an inter-chunk
    boundary, already final) or "boundary" (touches an inter-chunk edge,
    needs merge with neighbours).  Raster-edge boundaries are NOT counted
    as inter-chunk boundaries.
    """
    block = _to_numpy(block)
    if mask_block is not None:
        mask_block = _to_numpy(mask_block)
    ny, nx = block.shape
    column, polygon_points = _polygonize_numpy(
        block, mask_block, connectivity_8, transform=None)

    interior = []  # (value, [ring, ...])
    boundary = []  # (value, [ring, ...])

    for val, rings in zip(column, polygon_points):
        offset_rings = []
        for ring in rings:
            ring = ring.copy()
            ring[:, 0] += col_offset
            ring[:, 1] += row_offset
            offset_rings.append(ring)

        # Check if any vertex touches an INTERNAL chunk boundary.
        # Internal boundaries are chunk edges that are not raster edges.
        exterior = offset_rings[0]
        xs = exterior[:, 0]
        ys = exterior[:, 1]
        on_boundary = False
        if col_offset > 0:
            on_boundary |= np.any(xs == col_offset)
        if col_offset + nx < nx_total:
            on_boundary |= np.any(xs == col_offset + nx)
        if row_offset > 0:
            on_boundary |= np.any(ys == row_offset)
        if row_offset + ny < ny_total:
            on_boundary |= np.any(ys == row_offset + ny)

        if on_boundary:
            boundary.append((val, offset_rings))
        else:
            interior.append((val, offset_rings))

    return interior, boundary


def _add_or_cancel_edge(edge_set, x1, y1, x2, y2):
    """Add a directed unit edge, canceling with its reverse if present."""
    reverse = (x2, y2, x1, y1)
    if reverse in edge_set:
        edge_set[reverse] -= 1
        if edge_set[reverse] == 0:
            del edge_set[reverse]
    else:
        edge = (x1, y1, x2, y2)
        edge_set[edge] = edge_set.get(edge, 0) + 1


def _rings_to_unit_edges(polys_list, edge_set):
    """Split polygon ring edges into unit-length directed segments.

    Edges longer than 1 pixel (collinear segments) are decomposed into
    individual unit edges so that partial overlaps cancel correctly.
    """
    for rings in polys_list:
        for ring in rings:
            for k in range(len(ring) - 1):
                x1, y1 = int(ring[k, 0]), int(ring[k, 1])
                x2, y2 = int(ring[k + 1, 0]), int(ring[k + 1, 1])
                if x1 == x2:  # vertical
                    step = 1 if y2 > y1 else -1
                    y = y1
                    while y != y2:
                        _add_or_cancel_edge(edge_set, x1, y, x1, y + step)
                        y += step
                else:  # horizontal
                    step = 1 if x2 > x1 else -1
                    x = x1
                    while x != x2:
                        _add_or_cancel_edge(edge_set, x, y1, x + step, y1)
                        x += step


def _pick_next_edge(adj, prev_vertex, current_vertex):
    """Pick the next outgoing edge using the rightmost-turn rule.

    At a vertex with multiple outgoing edges, picks the first edge clockwise
    from the incoming direction (= smallest right turn).  This correctly
    traces individual polygon rings even when separate same-value polygons
    share a vertex, because it follows the ring that keeps the polygon
    interior to the left.
    """
    targets = adj[current_vertex]
    if len(targets) == 1:
        return next(iter(targets))

    dx_in = current_vertex[0] - prev_vertex[0]
    dy_in = current_vertex[1] - prev_vertex[1]
    incoming_angle = _DIR_ANGLE[(dx_in, dy_in)]

    best = None
    best_rel = 5
    for target in targets:
        dx = target[0] - current_vertex[0]
        dy = target[1] - current_vertex[1]
        out_angle = _DIR_ANGLE[(dx, dy)]
        rel = (out_angle - incoming_angle) % 4
        if rel == 0:
            rel = 4  # straight ahead → last priority (u-turn equivalent)
        if rel < best_rel:
            best_rel = rel
            best = target
    return best


def _remove_directed_edge(adj, from_v, to_v):
    """Remove one instance of directed edge from adj."""
    targets = adj[from_v]
    targets[to_v] -= 1
    if targets[to_v] == 0:
        del targets[to_v]


def _trace_rings(edge_set):
    """Trace directed unit edges into closed rings.

    Uses CW planar face ordering to correctly handle vertices with degree > 2
    (e.g. where two same-value regions share a corner vertex).
    """
    # Build adjacency: vertex -> {successor_vertex: count}.
    adj = {}
    for (x1, y1, x2, y2), count in edge_set.items():
        v = (x1, y1)
        t = (x2, y2)
        entry = adj.setdefault(v, {})
        entry[t] = entry.get(t, 0) + count

    rings = []
    while adj:
        start = next(iter(adj))
        first_target = next(iter(adj[start]))

        ring = [start]
        _remove_directed_edge(adj, start, first_target)
        prev = start
        current = first_target

        while current != start:
            ring.append(current)
            next_v = _pick_next_edge(adj, prev, current)
            _remove_directed_edge(adj, current, next_v)
            prev = current
            current = next_v

        ring.append(start)  # close the ring
        rings.append(np.array(ring, dtype=np.float64))

        # Clean up empty vertices.
        for v in list(adj.keys()):
            if not adj[v]:
                del adj[v]

    return rings


@ngjit
def _simplify_ring(ring):
    """Remove collinear (redundant) vertices from a closed ring."""
    n = len(ring) - 1  # exclude closing point
    if n <= 3:
        return ring
    # Single pass into pre-allocated output.
    out = np.empty((n + 1, 2), dtype=np.float64)
    count = 0
    for i in range(n):
        pi = (i - 1) % n
        ni = (i + 1) % n
        if not ((ring[pi, 0] == ring[i, 0] == ring[ni, 0]) or
                (ring[pi, 1] == ring[i, 1] == ring[ni, 1])):
            out[count, 0] = ring[i, 0]
            out[count, 1] = ring[i, 1]
            count += 1
    if count < n:
        out[count, 0] = out[0, 0]
        out[count, 1] = out[0, 1]
        count += 1
        return out[:count].copy()
    return ring


@ngjit
def _signed_ring_area(ring):
    """Shoelace formula for signed area of a closed ring."""
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * (np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


@ngjit
def _point_in_ring(px, py, ring):
    """Ray-casting point-in-polygon test."""
    n = len(ring) - 1
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i, 0], ring[i, 1]
        xj, yj = ring[j, 0], ring[j, 1]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _group_rings_into_polygons(rings):
    """Classify rings as exteriors/holes and assign holes to exteriors.

    Returns list of [exterior_ring, *hole_rings].
    """
    exteriors = []
    holes = []
    for ring in rings:
        area = _signed_ring_area(ring)
        if area > 0:
            exteriors.append(ring)
        elif area < 0:
            holes.append(ring)

    result = [[ext] for ext in exteriors]
    for hole in holes:
        px, py = hole[0, 0], hole[0, 1]
        for i, ext in enumerate(exteriors):
            if _point_in_ring(px, py, ext):
                result[i].append(hole)
                break
    return result


@ngjit
def _perpendicular_distance(px, py, ax, ay, bx, by):
    """Perpendicular distance from point (px,py) to line (ax,ay)-(bx,by)."""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0.0:
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


@ngjit
def _douglas_peucker(coords, tolerance):
    """Douglas-Peucker line simplification on an Nx2 float64 array.

    Endpoints are always preserved. Returns a new Nx2 array with
    only the retained vertices.
    """
    n = len(coords)
    if n <= 2:
        return coords.copy()

    # Iterative DP using an explicit numpy array stack to avoid recursion
    # depth issues and stay compatible with numba nopython mode.
    keep = np.zeros(n, dtype=np.bool_)
    keep[0] = True
    keep[n - 1] = True

    # Stack of (start, end) index pairs stored as a pre-allocated array.
    stack_arr = np.empty((n, 2), dtype=np.int64)
    stack_arr[0, 0] = np.int64(0)
    stack_arr[0, 1] = np.int64(n - 1)
    stack_top = 1

    while stack_top > 0:
        stack_top -= 1
        start = stack_arr[stack_top, 0]
        end = stack_arr[stack_top, 1]

        if end - start < 2:
            continue

        ax, ay = coords[start, 0], coords[start, 1]
        bx, by = coords[end, 0], coords[end, 1]

        max_dist = 0.0
        max_idx = start
        for i in range(start + 1, end):
            d = _perpendicular_distance(
                coords[i, 0], coords[i, 1], ax, ay, bx, by)
            if d > max_dist:
                max_dist = d
                max_idx = i

        if max_dist > tolerance:
            keep[max_idx] = True
            stack_arr[stack_top, 0] = start
            stack_arr[stack_top, 1] = max_idx
            stack_top += 1
            stack_arr[stack_top, 0] = max_idx
            stack_arr[stack_top, 1] = end
            stack_top += 1

    count = 0
    for i in range(n):
        if keep[i]:
            count += 1

    result = np.empty((count, 2), dtype=np.float64)
    j = 0
    for i in range(n):
        if keep[i]:
            result[j, 0] = coords[i, 0]
            result[j, 1] = coords[i, 1]
            j += 1

    return result


@ngjit
def _visvalingam_whyatt(coords, tolerance):
    """Visvalingam-Whyatt area-based line simplification.

    Iteratively removes the vertex that forms the smallest triangle
    area with its neighbors, until no triangle area is below tolerance.
    Endpoints are always preserved.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Input coordinate array.
    tolerance : float
        Minimum triangle area threshold. Vertices forming triangles
        with area below this value are removed.

    Returns
    -------
    np.ndarray, shape (M, 2)
        Simplified coordinate array.
    """
    n = len(coords)
    if n <= 2:
        return coords.copy()

    # Use a doubly-linked list via prev/next index arrays.
    prev_idx = np.empty(n, dtype=np.int64)
    next_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        prev_idx[i] = i - 1
        next_idx[i] = i + 1
    # Endpoints are never removed (sentinel values).
    prev_idx[0] = -1
    next_idx[n - 1] = -1

    # Compute triangle areas for interior vertices.
    areas = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, n - 1):
        ax, ay = coords[prev_idx[i], 0], coords[prev_idx[i], 1]
        bx, by = coords[i, 0], coords[i, 1]
        cx, cy = coords[next_idx[i], 0], coords[next_idx[i], 1]
        areas[i] = abs((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2.0)

    removed = np.zeros(n, dtype=np.bool_)
    remaining = n

    while remaining > 2:
        # Find vertex with minimum area.
        min_area = np.inf
        min_idx = -1
        for i in range(1, n - 1):
            if not removed[i] and areas[i] < min_area:
                min_area = areas[i]
                min_idx = i

        if min_idx == -1 or min_area >= tolerance:
            break

        # Remove vertex.
        removed[min_idx] = True
        remaining -= 1

        # Update linked list.
        p = prev_idx[min_idx]
        nx_i = next_idx[min_idx]
        if p >= 0:
            next_idx[p] = nx_i
        if nx_i >= 0 and nx_i < n:
            prev_idx[nx_i] = p

        # Recompute areas for affected neighbors.
        if p > 0 and prev_idx[p] >= 0:
            ax, ay = coords[prev_idx[p], 0], coords[prev_idx[p], 1]
            bx, by = coords[p, 0], coords[p, 1]
            cx, cy = coords[next_idx[p], 0], coords[next_idx[p], 1]
            new_area = abs((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2.0)
            # Enforce monotonicity: area can only increase.
            areas[p] = max(new_area, min_area)

        if nx_i >= 0 and nx_i < n - 1 and next_idx[nx_i] >= 0:
            ax, ay = coords[prev_idx[nx_i], 0], coords[prev_idx[nx_i], 1]
            bx, by = coords[nx_i, 0], coords[nx_i, 1]
            cx, cy = coords[next_idx[nx_i], 0], coords[next_idx[nx_i], 1]
            new_area = abs((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2.0)
            areas[nx_i] = max(new_area, min_area)

    # Collect remaining vertices.
    count = 0
    for i in range(n):
        if not removed[i]:
            count += 1

    result = np.empty((count, 2), dtype=np.float64)
    j = 0
    for i in range(n):
        if not removed[i]:
            result[j, 0] = coords[i, 0]
            result[j, 1] = coords[i, 1]
            j += 1

    return result


def _find_junctions(all_rings):
    """Find junction vertices that must be pinned during simplification.

    A junction is any vertex where the topology of shared edges changes.
    This includes:
    - Vertices where 3+ distinct rings meet (true topological junctions).
    - Endpoints of shared edge chains between pairs of rings (where a
      shared boundary starts or ends).

    These vertices are pinned so that shared edges are simplified
    identically for all rings that use them.

    Parameters
    ----------
    all_rings : list of list of np.ndarray
        polygon_points structure: list of polygons, each polygon is
        a list of rings (Nx2 arrays, closed).

    Returns
    -------
    set of (float, float)
    """
    # Build a set of directed edges per ring, and track which rings
    # each vertex belongs to.
    vertex_ring_count = {}  # (x, y) -> set of ring identifiers
    # Track directed edges: (pt_a, pt_b) -> set of ring_ids
    edge_rings = {}
    ring_id = 0
    for rings in all_rings:
        for ring in rings:
            n = len(ring) - 1  # unique vertices (ring is closed)
            for k in range(n):
                pt = (ring[k, 0], ring[k, 1])
                if pt not in vertex_ring_count:
                    vertex_ring_count[pt] = set()
                vertex_ring_count[pt].add(ring_id)

                pt_next = (ring[k + 1, 0], ring[k + 1, 1])
                edge = (pt, pt_next)
                if edge not in edge_rings:
                    edge_rings[edge] = set()
                edge_rings[edge].add(ring_id)
            ring_id += 1

    # An edge is "shared" if its reverse also exists (in a different ring).
    # Shared edges connect two polygons along a boundary.
    shared_edges = set()
    for (a, b), rids in edge_rings.items():
        rev = (b, a)
        if rev in edge_rings:
            # The forward and reverse edges exist (potentially in different rings).
            shared_edges.add((min(a, b), max(a, b)))

    # Build set of vertices on shared edges.
    shared_vertices = set()
    for a, b in shared_edges:
        shared_vertices.add(a)
        shared_vertices.add(b)

    # Find junctions: vertices in 3+ rings, OR shared vertices that
    # are adjacent to at least one non-shared edge in some ring.
    junctions = set()

    # Type 1: vertices in 3+ rings.
    for pt, ids in vertex_ring_count.items():
        if len(ids) >= 3:
            junctions.add(pt)

    # Type 2: endpoints of shared chains. A shared vertex is a chain
    # endpoint if, in any ring, it has an adjacent edge that is NOT shared.
    ring_id = 0
    for rings in all_rings:
        for ring in rings:
            n = len(ring) - 1
            for k in range(n):
                pt = (ring[k, 0], ring[k, 1])
                if pt not in shared_vertices:
                    continue
                # Check the edge leaving this vertex.
                pt_next = (ring[(k + 1) % n, 0], ring[(k + 1) % n, 1])
                edge_fwd = (min(pt, pt_next), max(pt, pt_next))
                # Check the edge arriving at this vertex.
                pt_prev = (ring[(k - 1) % n, 0], ring[(k - 1) % n, 1])
                edge_bwd = (min(pt, pt_prev), max(pt, pt_prev))

                if edge_fwd not in shared_edges or edge_bwd not in shared_edges:
                    junctions.add(pt)
            ring_id += 1

    return junctions


def _split_ring_at_junctions(ring, junctions):
    """Split a closed ring into chains at junction vertices.

    Each chain starts and ends at a junction vertex (endpoints included
    in the chain).  If the ring contains no junctions, the entire ring
    is returned as a single chain.

    Parameters
    ----------
    ring : np.ndarray, shape (N, 2)
        Closed ring (first == last vertex).
    junctions : set of (float, float)

    Returns
    -------
    list of np.ndarray
        Each array is an Mx2 chain.  Consecutive chains share their
        endpoint/startpoint.
    """
    n = len(ring) - 1  # number of unique vertices

    # Find indices of junction vertices within this ring.
    junction_indices = []
    for k in range(n):
        if (ring[k, 0], ring[k, 1]) in junctions:
            junction_indices.append(k)

    if len(junction_indices) == 0:
        # No junctions: return the whole ring as a single chain.
        return [ring.copy()]

    # Rotate ring so that the first junction is at index 0.
    first = junction_indices[0]
    if first > 0:
        # Rotate unique vertices, then re-close.
        rotated = np.empty_like(ring)
        rotated[:n - first] = ring[first:n]
        rotated[n - first:n] = ring[:first]
        rotated[n] = rotated[0]
        ring = rotated
        junction_indices = [(ji - first) % n for ji in junction_indices]
        junction_indices.sort()

    # Split at each junction.
    chains = []
    for i in range(len(junction_indices)):
        start = junction_indices[i]
        if i + 1 < len(junction_indices):
            end = junction_indices[i + 1]
        else:
            end = n  # wrap back to first junction (index 0 after rotation)
        chains.append(ring[start:end + 1].copy())

    return chains


def _chain_key(chain):
    """Canonical key for deduplicating shared edge chains.

    Two chains that connect the same pair of junctions but are traversed
    in opposite directions should map to the same key.  We use the sorted
    endpoint pair plus the frozenset of interior vertices.
    """
    start = (chain[0, 0], chain[0, 1])
    end = (chain[-1, 0], chain[-1, 1])
    if start > end:
        start, end = end, start
    # Include interior points for disambiguation.
    interior = tuple(
        (chain[k, 0], chain[k, 1]) for k in range(1, len(chain) - 1))
    interior_rev = interior[::-1]
    interior = min(interior, interior_rev)
    return (start, end, interior)


def _simplify_polygons(column, polygon_points, tolerance,
                       method="douglas-peucker"):
    """Topology-preserving simplification of all polygons.

    Uses shared-edge decomposition: finds junction vertices, splits
    rings into chains at junctions, simplifies each unique chain once
    with the chosen algorithm, then reassembles rings.

    Parameters
    ----------
    column : list
        Pixel values corresponding to each polygon.
    polygon_points : list of list of np.ndarray
        Output of polygonize backend: list of polygons, each polygon
        is [exterior_ring, *hole_rings].
    tolerance : float
        Simplification tolerance in coordinate units.
    method : str, default="douglas-peucker"
        Simplification algorithm: ``"douglas-peucker"`` or
        ``"visvalingam-whyatt"``.

    Returns
    -------
    (list, list of list of np.ndarray)
        Filtered column and simplified polygon_points.  Polygons whose
        exterior ring collapses below 4 vertices are dropped from both.
    """
    if tolerance <= 0:
        return column, polygon_points

    # Step 1: Find junctions.
    junctions = _find_junctions(polygon_points)

    # Step 2 & 3: Split rings into chains, deduplicate, simplify.
    simplified_chains = {}  # chain_key -> simplified np.ndarray

    # Track how to reassemble each ring.
    # ring_info[poly_idx][ring_idx] = list of (chain_key, is_reversed)
    ring_info = []

    for poly_idx, rings in enumerate(polygon_points):
        poly_info = []
        for ring in rings:
            chains = _split_ring_at_junctions(ring, junctions)
            chain_refs = []
            for chain in chains:
                key = _chain_key(chain)
                if key not in simplified_chains:
                    if method == "douglas-peucker":
                        simplified_chains[key] = _douglas_peucker(chain, tolerance)
                    else:
                        simplified_chains[key] = _visvalingam_whyatt(chain, tolerance)
                # Determine if this chain was reversed relative to canonical.
                start = (chain[0, 0], chain[0, 1])
                canonical_start = (simplified_chains[key][0, 0],
                                   simplified_chains[key][0, 1])
                is_reversed = (start != canonical_start)
                chain_refs.append((key, is_reversed))
            poly_info.append(chain_refs)
        ring_info.append(poly_info)

    # Step 4: Reassemble rings.
    result_column = []
    result = []
    for poly_idx, rings in enumerate(polygon_points):
        new_rings = []
        for ring_idx, chain_refs in enumerate(ring_info[poly_idx]):
            if len(chain_refs) == 1:
                key, is_reversed = chain_refs[0]
                simplified = simplified_chains[key]
                if is_reversed:
                    simplified = simplified[::-1].copy()
                # Ensure ring is closed.
                if not (simplified[0, 0] == simplified[-1, 0] and
                        simplified[0, 1] == simplified[-1, 1]):
                    simplified = np.vstack([simplified, simplified[:1]])
                new_rings.append(simplified)
            else:
                # Multiple chains: concatenate (drop duplicate junction points).
                parts = []
                for key, is_reversed in chain_refs:
                    simplified = simplified_chains[key]
                    if is_reversed:
                        simplified = simplified[::-1].copy()
                    if parts:
                        # Skip first point (same as last of previous chain).
                        parts.append(simplified[1:])
                    else:
                        parts.append(simplified)
                assembled = np.vstack(parts)
                # Ensure ring is closed.
                if not (assembled[0, 0] == assembled[-1, 0] and
                        assembled[0, 1] == assembled[-1, 1]):
                    assembled = np.vstack([assembled, assembled[:1]])
                new_rings.append(assembled)

        # Drop degenerate rings (fewer than 4 vertices = triangle minimum).
        # If the exterior ring is degenerate, drop the entire polygon.
        if len(new_rings) == 0 or len(new_rings[0]) < 4:
            continue
        filtered = [new_rings[0]]  # exterior is valid
        for ring in new_rings[1:]:
            if len(ring) >= 4:
                filtered.append(ring)
        result_column.append(column[poly_idx])
        result.append(filtered)

    return result_column, result


def _merge_polygon_rings(polys_list):
    """Merge polygon ring sets that share chunk-boundary edges.

    Uses edge cancellation: splits all rings into unit-length directed edges,
    cancels opposing edges (which occur at chunk boundaries where the same
    value continues across), and traces the remaining edges into closed rings.

    polys_list: list of [exterior_ring, *hole_rings] lists (same pixel value)
    Returns: list of [exterior_ring, *hole_rings] lists (merged)
    """
    edge_set = {}
    _rings_to_unit_edges(polys_list, edge_set)

    if not edge_set:
        return []

    raw_rings = _trace_rings(edge_set)
    simplified = [_simplify_ring(r) for r in raw_rings]
    return _group_rings_into_polygons(simplified)


def _merge_chunk_polygons(chunk_results, transform):
    """Merge polygons from all chunks and return final output."""
    all_interior = []
    boundary_by_value = {}

    for interior, boundary in chunk_results:
        all_interior.extend(interior)
        for val, rings in boundary:
            boundary_by_value.setdefault(val, []).append(rings)

    # Merge boundary polygons per value using edge cancellation.
    merged = []
    for val, polys_list in boundary_by_value.items():
        if len(polys_list) == 1:
            # Single polygon set for this value — nothing to merge.
            merged.append((val, polys_list[0]))
        else:
            merged_polys = _merge_polygon_rings(polys_list)
            for rings in merged_polys:
                merged.append((val, rings))

    # Combine interior and merged boundary polygons.
    all_polys = all_interior + merged

    # Sort by min (y, x) of exterior to approximate numpy row-major order.
    def sort_key(item):
        ext = item[1][0]
        min_y = np.min(ext[:, 1])
        min_x = np.min(ext[ext[:, 1] == min_y, 0])
        return (min_y, min_x, item[0])

    all_polys.sort(key=sort_key)

    # Apply transform and format output.
    column = []
    polygon_points = []
    for val, rings in all_polys:
        if transform is not None:
            for ring in rings:
                _transform_points(ring, transform)
        column.append(val)
        polygon_points.append(rings)

    return column, polygon_points


def _merge_from_separated(all_interior, boundary_by_value, transform):
    """Merge pre-separated interior/boundary polygons into final output.

    Like _merge_chunk_polygons but takes already-separated data so the
    caller can accumulate incrementally (one chunk at a time) instead of
    holding all chunk_results in memory simultaneously.
    """
    # Merge boundary polygons per value using edge cancellation.
    merged = []
    for val, polys_list in boundary_by_value.items():
        if len(polys_list) == 1:
            merged.append((val, polys_list[0]))
        else:
            merged_polys = _merge_polygon_rings(polys_list)
            for rings in merged_polys:
                merged.append((val, rings))

    all_polys = all_interior + merged

    def sort_key(item):
        ext = item[1][0]
        min_y = np.min(ext[:, 1])
        min_x = np.min(ext[ext[:, 1] == min_y, 0])
        return (min_y, min_x, item[0])

    all_polys.sort(key=sort_key)

    column = []
    polygon_points = []
    for val, rings in all_polys:
        if transform is not None:
            for ring in rings:
                _transform_points(ring, transform)
        column.append(val)
        polygon_points.append(rings)

    return column, polygon_points


def _polygonize_dask(dask_data, mask_data, connectivity_8, transform):
    """Dask backend for polygonize: per-chunk polygonize + edge merge."""
    # Ensure mask chunks match raster chunks.
    if mask_data is not None and mask_data.chunks != dask_data.chunks:
        mask_data = mask_data.rechunk(dask_data.chunks)

    # Compute row/col offsets for each chunk.
    row_chunks = dask_data.chunks[0]
    col_chunks = dask_data.chunks[1]
    row_offsets = np.concatenate([[0], np.cumsum(row_chunks[:-1])])
    col_offsets = np.concatenate([[0], np.cumsum(col_chunks[:-1])])

    ny_total = int(sum(row_chunks))
    nx_total = int(sum(col_chunks))

    # Process chunks incrementally: compute one at a time so only boundary
    # polygons accumulate in memory.  Interior polygons (fully inside a
    # chunk, no merging needed) go straight to the output list.  This keeps
    # peak memory proportional to boundary_polygon_count rather than
    # total_polygon_count * n_chunks.
    all_interior = []
    boundary_by_value = {}

    for iy in range(len(row_chunks)):
        for ix in range(len(col_chunks)):
            block = dask_data.blocks[iy, ix]
            mask_block = (mask_data.blocks[iy, ix]
                          if mask_data is not None else None)
            interior, boundary = dask.compute(
                dask.delayed(_polygonize_chunk)(
                    block, mask_block, connectivity_8,
                    int(row_offsets[iy]), int(col_offsets[ix]),
                    ny_total, nx_total,
                ))[0]
            all_interior.extend(interior)
            for val, rings in boundary:
                boundary_by_value.setdefault(val, []).append(rings)

    return _merge_from_separated(all_interior, boundary_by_value, transform)


def polygonize(
    raster: xr.DataArray,                  # shape (ny, nx) integer or float
    mask: Optional[xr.DataArray] = None,   # shape (ny, nx) bool/integer/float
    connectivity: int = 4,                 # 4 or 8
    transform: Optional[np.ndarray] = None,  # shape (6,)
    column_name: str = "DN",
    return_type: str = "numpy",
    simplify_tolerance: Optional[float] = None,
    simplify_method: str = "douglas-peucker",
):
    """
    Polygonize creates vector polygons for connected regions of pixels in a
    raster that share the same pixel value.  It is a raster to vector
    converter.

    Parameters
    ----------
    raster: xr.DataArray
        Input raster.

    mask: xr.DataArray, optional
        Optional input mask.  Pixels to include should have mask values of 1
        or True, pixels to exclude should have 0 or False.  This is the
        opposite of a NumPy mask.

    connectivity: int, default=4
        Whether to use 4-connectivity (adjacent along long edge only) or
        8-connectivity (adjacent along long edge or diagonal) to determine
        which pixels are connected.  Connectivity of 4 returns valid polygons
        (by shapely's definition) provided both x and y are monotonically
        increasing or decreasing.  Connectivity of 8 does not necessarily
        return valid polygons.

        Note: when using Dask arrays, 8-connectivity may produce extra polygon
        splits at chunk corners where diagonal-only adjacency crosses a chunk
        boundary.  4-connectivity works perfectly with Dask chunking.

    transform: ndarray, optional
        Optional affine transform to apply to return polygon coordinates.

    column_name: str, default="DN"
        Name to use for column returned.  Only used if return_type is
        "geopandas" or "spatialpandas".

    return_type: str, default="numpy"
        Format of returned data.  Allowed values are "numpy", "spatialpandas",
        "geopandas", "awkward" and "geojson".  "numpy" and "geojson" are
        always available, the others require optional dependencies.

    simplify_tolerance: float, optional
        Simplification tolerance in coordinate units. When set, polygon
        boundaries are simplified using shared-edge decomposition to
        preserve topology between adjacent polygons. Default is None
        (no simplification).

        For ``"douglas-peucker"``, this is the maximum perpendicular
        distance a vertex may deviate from the simplified line.

        For ``"visvalingam-whyatt"``, this is the minimum triangle area
        threshold; vertices forming triangles smaller than this are removed.

    simplify_method: str, default="douglas-peucker"
        Simplification algorithm. Options are ``"douglas-peucker"``
        (distance-based, good for general use) and ``"visvalingam-whyatt"``
        (area-based, tends to produce better cartographic results).

    Returns
    -------
    Polygons and their corresponding values in a format determined by
    return_type.

    Notes
    -----
    CuPy and Dask+CuPy arrays are accepted as input.  Data is transferred
    to CPU for processing because boundary tracing is an inherently
    sequential graph traversal (each step depends on the previous turn
    direction), preventing GPU parallelism.  Output is always CPU-side
    numpy coordinate arrays regardless of input type.

    For Dask+CuPy, each chunk is transferred independently, keeping peak
    CPU memory proportional to chunk size rather than full raster size.

    When ``return_type="geopandas"``, the raster's CRS is propagated to
    the output ``GeoDataFrame``.  The resolution order is
    ``raster.attrs['crs']``, then ``raster.attrs['crs_wkt']``, then
    ``raster.rio.crs`` (if rioxarray is installed).  An unparseable CRS
    value is dropped rather than raised.  The ``spatialpandas`` and
    ``geojson`` return types do not carry CRS metadata: spatialpandas
    has no CRS slot, and GeoJSON (RFC 7946) is WGS84 only.
    """
    _validate_raster(raster, func_name='polygonize', name='raster', ndim=2)
    if raster.shape[0] < 1 or raster.shape[1] < 1:
        raise ValueError(
            "Raster array must be 2D with a shape of at least (1, 1)")

    # Check mask.
    if mask is not None:
        _validate_raster(mask, func_name='polygonize', name='mask',
                         ndim=2, numeric=False)
        if not (type(raster.data) is type(mask.data)):  # noqa: E721
            raise TypeError(
                "raster and mask have different underlying types: "
                f"{type(raster.data)} and {type(mask.data)}")
        if raster.shape != mask.shape:
            raise ValueError(
                f"raster and mask must have the same shape: {raster.shape} "
                f"{mask.shape}")

    mask_data = mask.data if mask is not None else None

    # Check connectivity.
    if connectivity not in (4, 8):
        raise ValueError(
            f"connectivity must be either 4 or 8, not {connectivity}")
    connectivity_8 = (connectivity == 8)

    # Check transform.
    if transform is not None:
        transform = np.asarray(transform)
        if len(transform) != 6:
            raise ValueError(
                f"Incorrect transform length of {len(transform)} instead of 6")

    # Check simplification parameters.
    if simplify_tolerance is not None and simplify_tolerance < 0:
        raise ValueError(
            "simplify_tolerance must be non-negative, "
            f"got {simplify_tolerance}")
    if simplify_method not in ("douglas-peucker", "visvalingam-whyatt"):
        raise ValueError(
            f"simplify_method must be 'douglas-peucker' or "
            f"'visvalingam-whyatt', got '{simplify_method}'")

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_polygonize_numpy,
        cupy_func=_polygonize_cupy,
        dask_func=_polygonize_dask,
        dask_cupy_func=_polygonize_dask,
    )
    column, polygon_points = mapper(raster)(
        raster.data, mask_data, connectivity_8, transform)

    # Apply simplification if requested.
    if simplify_tolerance is not None and simplify_tolerance > 0:
        column, polygon_points = _simplify_polygons(
            column, polygon_points, simplify_tolerance,
            method=simplify_method)

    # Convert to requested return_type.
    if return_type == "numpy":
        return column, polygon_points
    elif return_type == "awkward":
        return _to_awkward(column, polygon_points)
    elif return_type == "geopandas":
        # Propagate the raster's CRS to the GeoDataFrame so downstream
        # spatial joins / reprojections / file writes keep their
        # georeferencing.  spatialpandas has no CRS slot and GeoJSON
        # RFC 7946 is WGS84-only, so the propagation lives only here.
        crs = _detect_raster_crs(raster)
        return _to_geopandas(column, polygon_points, column_name, crs=crs)
    elif return_type == "spatialpandas":
        return _to_spatialpandas(column, polygon_points, column_name)
    elif return_type == "geojson":
        return _to_geojson(column, polygon_points, column_name)
    else:
        raise ValueError(f"Invalid return_type '{return_type}'")
