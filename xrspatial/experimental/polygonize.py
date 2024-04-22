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

from ..utils import ngjit

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
    values: np.ndarray,               # shape (nx*ny,)
    mask: Optional[np.ndarray],       # shape (nx*ny,)
    connectivity_8: bool,
    transform: Optional[np.ndarray],  # shape (6,)
    nx: int,
    ny: int,
) -> Tuple[List[Union[int, float]], List[List[np.ndarray]]]:
    regions = _calculate_regions(values, mask, connectivity_8, nx, ny)

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


def _to_geopandas(
    column: List[Union[int, float]],
    polygon_points: List[np.ndarray],
    column_name: str,
):
    import geopandas as gpd
    from shapely.geometry import Polygon

    # Convert list of point arrays to shapely Polygons.
    polygons = list(map(
        lambda points: Polygon(points[0], points[1:]), polygon_points))

    df = gpd.GeoDataFrame({column_name: column, "geometry": polygons})
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


def _polygonize_numpy(
    values: np.ndarray,
    mask: Optional[np.ndarray],
    connectivity_8: bool,
    transform: Optional[np.ndarray],
) -> Tuple[List[Union[int, float]], List[List[np.ndarray]]]:

    ny, nx = values.shape
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

    values = values.ravel()
    if mask is not None:
        mask = mask.ravel()

    column, polygon_points = _scan(
        values, mask, connectivity_8, transform, nx, ny)

    return column, polygon_points


def polygonize(
    raster: xr.DataArray,                  # shape (ny, nx) integer or float
    mask: Optional[xr.DataArray] = None,   # shape (ny, nx) bool/integer/float
    connectivity: int = 4,                 # 4 or 8
    transform: Optional[np.ndarray] = None,  # shape (6,)
    column_name: str = "DN",
    return_type: str = "numpy",
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

    transform: ndarray, optional
        Optional affine transform to apply to return polygon coordinates.

    column_name: str, default="DN"
        Name to use for column returned.  Only used if return_type is
        "geopandas" or "spatialpandas".

    return_type: str, default="numpy"
        Format of returned data.  Allowed values are "numpy", "spatialpandas",
        "geopandas" and "awkward".  Only "numpy" is always available, the
        others require optional dependencies.

    Returns
    -------
    Polygons and their corresponding values in a format determined by
    return_type.
    """
    if raster.ndim != 2 or raster.shape[0] < 1 or raster.shape[1] < 1:
        raise ValueError(
            "Raster array must be 2D with a shape of at least (1, 1)")

    # Check mask.
    if mask is not None:
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

    if isinstance(raster.data, np.ndarray):
        column, polygon_points = _polygonize_numpy(
            raster.data, mask_data, connectivity_8, transform)
    else:
        raise TypeError(f"Unsupported array type: {type(raster.data)}")

    # Convert to requested return_type.
    if return_type == "numpy":
        return column, polygon_points
    elif return_type == "awkward":
        return _to_awkward(column, polygon_points)
    elif return_type == "geopandas":
        return _to_geopandas(column, polygon_points, column_name)
    elif return_type == "spatialpandas":
        return _to_spatialpandas(column, polygon_points, column_name)
    else:
        raise ValueError(f"Invalid return_type '{return_type}'")
