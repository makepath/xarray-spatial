from math import sqrt

import dask.array as da
import numpy as np
import xarray as xr
from numba import prange

from xrspatial.utils import get_dataarray_resolution, ngjit

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

    # x-y coordinates of each pixel.
    # flatten the coords of input raster and reshape to 2d
    xs = np.tile(raster[x].data, raster.shape[0]).reshape(raster.shape)
    ys = np.repeat(raster[y].data, raster.shape[1]).reshape(raster.shape)

    if max_distance is None:
        max_distance = np.inf

    max_possible_distance = _distance(
        xs[0][0], xs[-1][-1], ys[0][0], ys[-1][-1], distance_metric
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
        # numpy case
        result = _process_numpy(raster.data, xs, ys)

    elif isinstance(raster.data, da.Array):
        # dask + numpy case
        xs = da.from_array(xs, chunks=(raster.chunks))
        ys = da.from_array(ys, chunks=(raster.chunks))
        result = _process_dask(raster, xs, ys)

    return result


# ported from
# https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp
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
    proximity_agg: xr.DataArray of same type as `raster`
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
    allocation_agg: xr.DataArray of same type as `raster`
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
    direction_agg: xr.DataArray of same type as `raster`
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
