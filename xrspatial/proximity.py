from math import sqrt

import xarray as xr
import numpy as np
import dask.array as da

from numba import njit, prange

from xrspatial.utils import ngjit

import sklearn.neighbors


EUCLIDEAN = 0
GREAT_CIRCLE = 1
MANHATTAN = 2

PROXIMITY = 0
ALLOCATION = 1
DIRECTION = 2


def _distance_metric_mapping():
    DISTANCE_METRICS = {}
    DISTANCE_METRICS['EUCLIDEAN'] = EUCLIDEAN
    DISTANCE_METRICS['GREAT_CIRCLE'] = GREAT_CIRCLE
    DISTANCE_METRICS['MANHATTAN'] = MANHATTAN

    return DISTANCE_METRICS


# create dictionary to map distance metric presented by string and the
# corresponding metric presented by integer.
DISTANCE_METRICS = _distance_metric_mapping()


@njit(nogil=True)
def euclidean_distance(x1: float,
                       x2: float,
                       y1: float,
                       y2: float) -> float:
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
                point_a[0],
                point_b[0],
                point_a[1],
                point_b[1],
            )
        >>> print(dist)
        442.80462599209596
    """
    x = x1 - x2
    y = y1 - y2
    return np.sqrt(x * x + y * y)


@njit(nogil=True)
def manhattan_distance(x1: float,
                       x2: float,
                       y1: float,
                       y2: float) -> float:
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

        >>> # Imports
        >>> from xrspatial import manhattan_distance

        >>> point_a = (142.32, 23.23)
        >>> point_b = (312.54, 432.01)

        >>> # Calculate Euclidean Distance
        >>> dist = manhattan_distance(
                point_a[0],
                point_b[0],
                point_a[1],
                point_b[1],
            )
        >>> print(dist)
        196075.9368
    """
    x = x1 - x2
    y = y1 - y2
    return abs(x) + abs(y)


@njit(nogil=True)
def great_circle_distance(x1: float,
                          x2: float,
                          y1: float,
                          y2: float,
                          radius: float = 6378137) -> float:
    """
    Calculates great-circle (orthodromic/spherical) distance between
    (x1, y1) and (x2, y2), assuming each point is a longitude,
    latitude pair.

    Parameters
    ----------
    x1 : float
        x-coordinate (latitude) between -180 and 180 of the first point.
    x2: float
        x-coordinate (latitude) between -180 and 180 of the second point.
    y1: float
        y-coordinate (longitude) between -90 and 90 of the first point.
    y2: float
        y-coordinate (longitude) between -90 and 90 of the second point.
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

        >>> # Imports
        >>> from xrspatial import great_circle_distance

        >>> point_a = (123.2, 82.32)
        >>> point_b = (178.0, 65.09)

        >>> # Calculate Euclidean Distance
        >>> dist = great_circle_distance(
                point_a[0],
                point_b[0],
                point_a[1],
                point_b[1],
            )
        >>> print(dist)
        2378290.489801402
    """
    if x1 > 180 or x1 < -180:
        raise ValueError('Invalid x-coordinate of the first point.'
                         'Must be in the range [-180, 180]')

    if x2 > 180 or x2 < -180:
        raise ValueError('Invalid x-coordinate of the second point.'
                         'Must be in the range [-180, 180]')

    if y1 > 90 or y1 < -90:
        raise ValueError('Invalid y-coordinate of the first point.'
                         'Must be in the range [-90, 90]')

    if y2 > 90 or y2 < -90:
        raise ValueError('Invalid y-coordinate of the second point.'
                         'Must be in the range [-90, 90]')

    lat1, lon1, lat2, lon2 = (np.radians(y1), np.radians(x1),
                              np.radians(y2), np.radians(x2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # earth radius: 6378137
    return radius * 2 * np.arcsin(np.sqrt(a))


@njit(nogil=True)
def _distance(x1, x2, y1, y2, metric):
    if metric == EUCLIDEAN:
        return euclidean_distance(x1, x2, y1, y2)

    if metric == GREAT_CIRCLE:
        return great_circle_distance(x1, x2, y1, y2)

    if metric == MANHATTAN:
        return manhattan_distance(x1, x2, y1, y2)

    return -1.0


@njit(nogil=True)
def _process_proximity_line(source_line, x_coords, y_coords,
                            pan_near_x, pan_near_y, is_forward,
                            line_id, width, max_distance, line_proximity,
                            nearest_xs, nearest_ys,
                            values, distance_metric):
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
    max_distance : np.float64, maximum distance considered.
    line_proximity : numpy.array
        1d numpy array of type np.float64, calculated proximity from
        source_line.
    values : numpy.array
        1d numpy array. A list of target pixel values
        to measure the distance from. If this option is not provided
        proximity will be computed from non-zero pixel values.

    Returns
    -------
    self: numpy.array
        1d numpy array of type np.float64. Corresponding proximity of
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
            dist = _distance(x_coords[pan_near_x[pixel]], x_coords[pixel],
                             y_coords[pan_near_y[pixel]], y_coords[line_id],
                             distance_metric)
            dist_sqr = dist ** 2
            if dist_sqr < near_distance_square:
                near_distance_square = dist_sqr
            else:
                pan_near_x[pixel] = -1
                pan_near_y[pixel] = -1

        # Are we near(er) to the closest target to the left (right) pixel?
        last = pixel - step
        if pixel != start and pan_near_x[last] != -1:
            dist = _distance(x_coords[pan_near_x[last]], x_coords[pixel],
                             y_coords[pan_near_y[last]], y_coords[line_id],
                             distance_metric)
            dist_sqr = dist ** 2
            if dist_sqr < near_distance_square:
                near_distance_square = dist_sqr
                pan_near_x[pixel] = pan_near_x[last]
                pan_near_y[pixel] = pan_near_y[last]

        #  Are we near(er) to the closest target to the
        #  topright (bottom left) pixel?
        tr = pixel + step
        if tr != end and pan_near_x[tr] != -1:
            dist = _distance(x_coords[pan_near_x[tr]], x_coords[pixel],
                             y_coords[pan_near_y[tr]], y_coords[line_id],
                             distance_metric)
            dist_sqr = dist ** 2
            if dist_sqr < near_distance_square:
                near_distance_square = dist_sqr
                pan_near_x[pixel] = pan_near_x[tr]
                pan_near_y[pixel] = pan_near_y[tr]

        # Update our proximity value.
        if pan_near_x[pixel] != -1 \
                and max_distance * max_distance >= near_distance_square \
                and (line_proximity[pixel] < 0 or
                     near_distance_square <
                     line_proximity[pixel] * line_proximity[pixel]):
            line_proximity[pixel] = sqrt(near_distance_square)
            nearest_xs[pixel] = pan_near_x[pixel]
            nearest_ys[pixel] = pan_near_y[pixel]
    return


@njit
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
    return d


@njit(nogil=True)
def _process_image(img, x_coords, y_coords, target_values,
                   max_distance, distance_metric, process_mode):
    # max_distance = _distance(x_coords[0], x_coords[-1],
    #                          y_coords[0], y_coords[-1],
    #                          distance_metric)

    height, width = img.shape

    pan_near_x = np.zeros(width, dtype=np.int64)
    pan_near_y = np.zeros(width, dtype=np.int64)

    # output of the function
    img_distance = np.zeros(shape=(height, width), dtype=np.float64)
    img_allocation = np.zeros(shape=(height, width), dtype=np.float64)
    img_direction = np.zeros(shape=(height, width), dtype=np.float64)

    # Loop from top to bottom of the image.
    for i in prange(width):
        pan_near_x[i] = -1
        pan_near_y[i] = -1

    # a single line of the input image @img
    scan_line = np.zeros(width, dtype=img.dtype)
    # indexes of nearest pixels of current line @scan_line
    nearest_xs = np.zeros(width, dtype=np.int64)
    nearest_ys = np.zeros(width, dtype=np.int64)

    for line in prange(height):
        # Read for target values.
        for i in prange(width):
            scan_line[i] = img[line][i]

        line_proximity = np.zeros(width, dtype=np.float64)

        for i in prange(width):
            line_proximity[i] = -1.0
            nearest_xs[i] = -1
            nearest_ys[i] = -1

        # left to right
        _process_proximity_line(scan_line, x_coords, y_coords,
                                pan_near_x, pan_near_y, True, line,
                                width, max_distance, line_proximity,
                                nearest_xs, nearest_ys,
                                target_values, distance_metric)
        for i in prange(width):
            if nearest_xs[i] != -1 and line_proximity[i] >= 0:
                img_allocation[line][i] = img[nearest_ys[i], nearest_xs[i]]
                d = _calc_direction(x_coords[i], x_coords[nearest_xs[i]],
                                    y_coords[line], y_coords[nearest_ys[i]])
                img_direction[line][i] = d

        # right to left
        for i in prange(width):
            nearest_xs[i] = -1
            nearest_ys[i] = -1

        _process_proximity_line(scan_line, x_coords, y_coords,
                                pan_near_x, pan_near_y, False, line,
                                width, max_distance, line_proximity,
                                nearest_xs, nearest_ys,
                                target_values, distance_metric)

        for i in prange(width):
            img_distance[line][i] = line_proximity[i]
            if nearest_xs[i] != -1 and line_proximity[i] >= 0:
                img_allocation[line][i] = img[nearest_ys[i], nearest_xs[i]]
                d = _calc_direction(x_coords[i], x_coords[nearest_xs[i]],
                                    y_coords[line], y_coords[nearest_ys[i]])
                img_direction[line][i] = d

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

        _process_proximity_line(scan_line, x_coords, y_coords,
                                pan_near_x, pan_near_y, False, line,
                                width, max_distance, line_proximity,
                                nearest_xs, nearest_ys,
                                target_values, distance_metric)

        for i in prange(width):
            if nearest_xs[i] != -1 and line_proximity[i] >= 0:
                img_allocation[line][i] = img[nearest_ys[i], nearest_xs[i]]
                d = _calc_direction(x_coords[i], x_coords[nearest_xs[i]],
                                    y_coords[line], y_coords[nearest_ys[i]])
                img_direction[line][i] = d

        # Left to right
        for i in prange(width):
            nearest_xs[i] = -1
            nearest_ys[i] = -1

        _process_proximity_line(scan_line, x_coords, y_coords,
                                pan_near_x, pan_near_y, True, line,
                                width, max_distance, line_proximity,
                                nearest_xs, nearest_ys,
                                target_values, distance_metric)

        # final post processing of distances
        for i in prange(width):
            if line_proximity[i] < 0:
                line_proximity[i] = np.nan
            else:
                if nearest_xs[i] != -1 and line_proximity[i] >= 0:
                    img_allocation[line][i] = img[nearest_ys[i],
                                                  nearest_xs[i]]
                    d = _calc_direction(x_coords[i],
                                        x_coords[nearest_xs[i]],
                                        y_coords[line],
                                        y_coords[nearest_ys[i]])
                    img_direction[line][i] = d

        for i in prange(width):
            img_distance[line][i] = line_proximity[i]

    if process_mode == PROXIMITY:
        return img_distance
    elif process_mode == ALLOCATION:
        return img_allocation
    elif process_mode == DIRECTION:
        return img_direction


def _process_numpy(raster, x, y, target_values, max_distance,
                   distance_metric, process_mode):

    distance_metric = DISTANCE_METRICS.get(distance_metric, None)

    if distance_metric is None:
        distance_metric = DISTANCE_METRICS['EUCLIDEAN']

    target_values = np.asarray(target_values)

    img = raster.values
    y_coords = raster.coords[y].values
    x_coords = raster.coords[x].values

    if max_distance is None:
        max_distance = np.inf
    output_img = _process_image(img, x_coords, y_coords, target_values,
                                max_distance, distance_metric, process_mode)
    return output_img


@ngjit
def _indicies_to_coords_numpy(indicies, coords):
    out = np.zeros_like(indicies, dtype=np.float64)
    n = indicies.shape[0]
    for i in range(n):
        out[i] = coords[indicies[i]]
    return out


def _indicies_to_coords_dask(indicies, coords):
    out = da.map_blocks(
        _indicies_to_coords_numpy,
        indicies,
        coords,
        meta=np.array(())
    )
    return out


@ngjit
def _compute_distance_numpy(nearest_targets,
                            flatten_coords,
                            target_xcoords,
                            target_ycoords,
                            metric):
    out = np.zeros_like(nearest_targets, dtype=np.float64)
    n = nearest_targets.shape[0]
    for i in range(n):
        x1 = flatten_coords[i][1]
        y1 = flatten_coords[i][0]
        x2 = target_xcoords[nearest_targets[i][0]]
        y2 = target_ycoords[nearest_targets[i][0]]
        out[i] = _distance(x1, x2, y1, y2, metric)
    return out


def _compute_distance_dask(nearest_targets,
                           flatten_coords,
                           target_xcoords,
                           target_ycoords,
                           metric):
    out = da.map_blocks(
        _compute_distance_numpy,
        nearest_targets,
        flatten_coords,
        target_xcoords,
        target_ycoords,
        metric,
        meta=np.array(())
    )
    return out


@ngjit
def _compute_direction_numpy(nearest_targets,
                             flatten_coords,
                             target_xcoords,
                             target_ycoords):
    out = np.zeros_like(nearest_targets, dtype=np.float64)
    n = nearest_targets.shape[0]
    for i in range(n):
        x1 = flatten_coords[i][1]
        y1 = flatten_coords[i][0]
        x2 = target_xcoords[nearest_targets[i][0]]
        y2 = target_ycoords[nearest_targets[i][0]]
        out[i] = _calc_direction(x1, x2, y1, y2)
    return out


def _compute_direction_dask(nearest_targets,
                            flatten_coords,
                            target_xcoords,
                            target_ycoords):
    out = da.map_blocks(
        _compute_direction_numpy,
        nearest_targets,
        flatten_coords,
        target_xcoords,
        target_ycoords,
        meta=np.array(())
    )
    return out


def _process_dask(raster, x, y, target_values, distance_metric, process_mode):

    if process_mode == ALLOCATION:
        raise NotImplementedError('allocation does not support Dask yet.')

    # find target pixels by target values
    if len(target_values):
        conditions = False
        for t in target_values:
            conditions |= (raster.data == t)
    else:
        conditions = (raster.data != 0)

    target_mask = da.ma.masked_where(
        (conditions & np.isfinite(raster.data)),
        raster.data
    )
    # indicies in pixel space of all targets
    target_ys = da.nonzero(da.ma.getmaskarray(target_mask))[0].compute()
    target_xs = da.nonzero(da.ma.getmaskarray(target_mask))[1].compute()

    # coords of all targets
    target_ycoords = _indicies_to_coords_dask(
        target_ys, raster[y].data).compute()
    target_xcoords = _indicies_to_coords_dask(
        target_xs, raster[x].data).compute()
    target_coords = np.array(
        [[y, x] for y, x in zip(target_ycoords, target_xcoords)]
    )

    # chunksize
    chunksize = max(*raster.shape, *target_coords.shape)

    # A 2-D array that has the x-y coordinates of each point.
    # flatten the coords of input raster
    xs = np.tile(raster[x], raster.shape[0])
    ys = np.repeat(raster[y], raster.shape[1])
    flatten_coords = da.stack([ys, xs]).T.rechunk(chunksize)

    # build the KDTree
    tree = sklearn.neighbors.KDTree(
        target_coords, metric=distance_metric.lower()
    )

    nearest_targets = flatten_coords.map_blocks(
        tree.query,
        return_distance=False,
        dtype=flatten_coords.dtype,
        chunks=(flatten_coords.chunks[0], (1,))
    )

    if process_mode == PROXIMITY:
        out = _compute_distance_dask(
            nearest_targets,
            flatten_coords,
            target_xcoords,
            target_ycoords,
            DISTANCE_METRICS[distance_metric],
        ).reshape(raster.shape)

    elif process_mode == DIRECTION:
        out = _compute_direction_dask(
            nearest_targets,
            flatten_coords,
            target_xcoords,
            target_ycoords,
        ).reshape(raster.shape)

    return out


def _process(raster, x, y, target_values, max_distance,
             distance_metric, process_mode):

    raster_dims = raster.dims
    if raster_dims != (y, x):
        raise ValueError("raster.coords should be named as coordinates:"
                         "({0}, {1})".format(y, x))

    if isinstance(raster.data, np.ndarray):
        # numpy case
        result = _process_numpy(
            raster, x=x, y=y, target_values=target_values,
            max_distance=max_distance, distance_metric=distance_metric,
            process_mode=process_mode
        )
    elif isinstance(raster.data, da.Array):
        # dask + numpy case
        result = _process_dask(
            raster, x=x, y=y, target_values=target_values,
            distance_metric=distance_metric, process_mode=process_mode
        )
    return result


# ported from
# https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp
def proximity(raster: xr.DataArray,
              x: str = 'x',
              y: str = 'y',
              target_values: list = [],
              max_distance: float = None,
              distance_metric: str = 'EUCLIDEAN') -> xr.DataArray:
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
    The implementation for Dask-backed uses `sklearn.sklearn.neighbors.KDTree`
    internally.

    Parameters
    ----------
    raster : xr.DataArray
        2D array image with `raster.shape` = (height, width).
    x : str, default='x'
        Name of x-coordinates.
    y : str, default='y'
        Name of y-coordinates.
    target_values: list
        Target pixel values to measure the distance from. If this option
        is not provided, proximity will be computed from non-zero pixel
        values.
    distance_metric: str, default='EUCLIDEAN'
        The metric for calculating distance between 2 points.
        Valid distance_metrics for Numpy-backed raster: 'EUCLIDEAN', 'GREAT_CIRCLE', and 'MANHATTAN'.  # noqa
        Valid distance_metrics for Dask-backed raster: 'EUCLIDEAN', and 'MANHATTAN'.  # noqa

    Returns
    -------
    proximity_agg: xr.DataArray of same type as `raster`
        2D array of proximity values.
        All other input attributes are preserved.

    References
    ----------
        - OSGeo: https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import numpy as np
        import xarray as xr
        from xrspatial import proximity

        data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
        raster['y'] = np.arange(n)[::-1]
        raster['x'] = np.arange(n)

        proximity_agg = proximity(raster)

    .. sourcecode:: python

        >>> raster
        <xarray.DataArray 'raster' (y: 5, x: 5)>
        array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [1., 0., 0., 0., 0.]])
        Coordinates:
          * y        (y) int64 4 3 2 1 0
          * x        (x) int64 0 1 2 3 4

        >>> proximity_agg
        <xarray.DataArray (y: 5, x: 5)>
        array([[3.1622777, 2.236068 , 1.4142135, 1.       , 1.4142135],
               [3.       , 2.       , 1.       , 0.       , 1.       ],
               [3.1622777, 2.236068 , 1.4142135, 1.       , 1.4142135],
               [3.6055512, 2.828427 , 2.236068 , 2.       , 2.236068 ],
               [4.2426405, 3.6055512, 3.1622777, 3.       , 3.1622777]],
              dtype=float64)
        Coordinates:
          * y        (y) int64 4 3 2 1 0
          * x        (x) int64 0 1 2 3 4
    """

    proximity_img = _process(raster,
                             x=x,
                             y=y,
                             target_values=target_values,
                             max_distance=max_distance,
                             distance_metric=distance_metric,
                             process_mode=PROXIMITY)

    result = xr.DataArray(proximity_img,
                          coords=raster.coords,
                          dims=raster.dims,
                          attrs=raster.attrs)

    return result


def allocation(raster: xr.DataArray,
               x: str = 'x',
               y: str = 'y',
               target_values: list = [],
               max_distance: float = None,
               distance_metric: str = 'EUCLIDEAN'):
    """
    Calculates, for all cells in the array, the downward slope direction
    Calculates, for all pixels in the input raster, the nearest source
    based on a set of target values and a distance metric.

    This function attempts to produce the value of nearest feature of all
    pixels in the image to a set of pixels in the source image. The
    following options are used to define the behavior of the function.
    By default all non-zero pixels in `raster.values` will be considered
    as"target", and all allocation will be computed in pixels.

    Allocation support Numpy-backed xarray DataArray currently.
    It uses the same approach as `proximity`, which is ported from GDAL.
    A dynamic programming approach is used for identifying nearest target
    of a pixel from its surrounding neighborhood in a 3x3 window.

    Parameters
    ----------
    raster : xr.DataArray
        2D array of target data.
    x : str, default='x'
        Name of x-coordinates.
    y : str, default='y'
        Name of y-coordinates.
    target_values : list
        Target pixel values to measure the distance from. If this option
        is not provided, allocation will be computed from non-zero pixel
        values.
    distance_metric : str, default='EUCLIDEAN'
        The metric for calculating distance between 2 points. Valid
        distance_metrics: 'EUCLIDEAN', 'GREAT_CIRCLE', and 'MANHATTAN'.

    Returns
    -------
    allocation_agg: xr.DataArray of same type as `raster`
        2D array of allocation values.
        All other input attributes are preserved.

    References
    ----------
        - OSGeo: https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import numpy as np
        import xarray as xr
        from xrspatial import allocation

        data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 2., 0.],
            [0., 0., 3., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
        raster['y'] = np.arange(n)[::-1]
        raster['x'] = np.arange(n)

        allocation_agg = allocation(raster)

    .. sourcecode:: python

        >>> raster
        <xarray.DataArray 'raster' (y: 5, x: 5)>
        array([[0., 0., 0., 0., 0.],
               [0., 1., 0., 2., 0.],
               [0., 0., 3., 0., 0.],
               [0., 0., 0., 0., 0.],
               [1., 0., 0., 0., 0.]])
        Coordinates:
          * y        (y) int64 4 3 2 1 0
          * x        (x) int64 0 1 2 3 4

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
    allocation_img = _process(raster,
                              x=x,
                              y=y,
                              target_values=target_values,
                              max_distance=max_distance,
                              distance_metric=distance_metric,
                              process_mode=ALLOCATION)

    # convert to have same type as of input @raster
    result = xr.DataArray((allocation_img).astype(raster.dtype),
                          coords=raster.coords,
                          dims=raster.dims,
                          attrs=raster.attrs)
    return result


def direction(raster: xr.DataArray,
              x: str = 'x',
              y: str = 'y',
              target_values: list = [],
              max_distance: float = None,
              distance_metric: str = 'EUCLIDEAN'):
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
    The implementation for Dask-backed uses `sklearn.sklearn.neighbors.KDTree`
    internally.

    Parameters
    ----------
    raster : xr.DataArray
        2D array image with `raster.shape` = (height, width).
    x : str, default='x'
        Name of x-coordinates.
    y : str, default='y'
        Name of y-coordinates.
    target_values: list
        Target pixel values to measure the distance from. If this
        option is not provided, proximity will be computed from
        non-zero pixel values.
    distance_metric: str, default='EUCLIDEAN'
        The metric for calculating distance between 2 points.
        Valid distance_metrics for NumPy-backed raster: 'EUCLIDEAN', 'GREAT_CIRCLE', and 'MANHATTAN'.  # noqa
        Valid distance_metrics for Dask-backed raster: 'EUCLIDEAN', and 'MANHATTAN'.  # noqa

    Returns
    -------
    direction_agg: xr.DataArray of same type as `raster`
        2D array of direction values.
        All other input attributes are preserved.

    References
    ----------
        - OSGeo: https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import numpy as np
        import xarray as xr
        from xrspatial import direction

        data = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.]
        ])
        raster = xr.DataArray(data, dims=['y', 'x'], name='raster')
        raster['y'] = np.arange(n)[::-1]
        raster['x'] = np.arange(n)

        direction_agg = direction(raster)

    ... sourcecode:: python

        >>> raster
        <xarray.DataArray 'raster' (y: 5, x: 5)>
        array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 0.],
               [1., 0., 0., 0., 0.]])
        Coordinates:
          * y        (y) int64 4 3 2 1 0
          * x        (x) int64 0 1 2 3 4

        >>> direction_agg
        <xarray.DataArray (y: 5, x: 5)>
        array([[ 45.      ,  26.56505 , 360.      , 333.43494 , 315.      ],
               [ 63.434948,  45.      , 360.      , 315.      , 296.56506 ],
               [ 90.      ,  90.      ,   0.      , 270.      , 270.      ],
               [360.      , 135.      , 180.      , 225.      , 243.43495 ],
               [  0.      , 270.      , 180.      , 206.56505 , 225.      ]],
              dtype=float64)
        Coordinates:
          * y        (y) int64 4 3 2 1 0
          * x        (x) int64 0 1 2 3 4
    """

    direction_img = _process(raster,
                             x=x,
                             y=y,
                             target_values=target_values,
                             max_distance=max_distance,
                             distance_metric=distance_metric,
                             process_mode=DIRECTION)

    result = xr.DataArray(direction_img,
                          coords=raster.coords,
                          dims=raster.dims,
                          attrs=raster.attrs)
    return result
