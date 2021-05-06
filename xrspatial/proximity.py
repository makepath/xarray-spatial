import xarray as xr
import numpy as np
from numba import njit, prange
from math import sqrt

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
        1d numpy array of type np.uint8. A list of target pixel values
        to measure the distance from. If this option is not provided
        proximity will be computed from non-zero pixel values.
        Currently pixel values are internally processed as integers.

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
                   distance_metric, process_mode):
    max_distance = _distance(x_coords[0], x_coords[-1],
                             y_coords[0], y_coords[-1],
                             distance_metric)

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


def _process(raster, x='x', y='y', target_values=[],
             distance_metric='EUCLIDEAN', process_mode=PROXIMITY):
    raster_dims = raster.dims
    if raster_dims != (y, x):
        raise ValueError("raster.coords should be named as coordinates:"
                         "({0}, {1})".format(y, x))

    # convert distance metric from string to integer, the correct type
    # of argument for function _distance()
    distance_metric = DISTANCE_METRICS.get(distance_metric, None)

    if distance_metric is None:
        distance_metric = DISTANCE_METRICS['EUCLIDEAN']

    target_values = np.asarray(target_values).astype(np.uint8)

    img = raster.values
    y_coords = raster.coords[y].values
    x_coords = raster.coords[x].values

    output_img = _process_image(img, x_coords, y_coords, target_values,
                                distance_metric, process_mode)
    return output_img


# ported from
# https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp
def proximity(raster: xr.DataArray,
              x: str = 'x',
              y: str = 'y',
              target_values: list = [],
              distance_metric: str = 'EUCLIDEAN') -> xr.DataArray:
    """
    Computes the proximity of all pixels in the image to a set of pixels
    in the source image based on Euclidean, Great-Circle or Manhattan
    distance.

    This function attempts to compute the proximity of all pixels in the
    image to a set of pixels in the source image. The following options
    are used to define the behavior of the function. By default all
    non-zero pixels in `raster.values` will be considered the "target",
    and all proximities will be computed in pixels. Note that target
    pixels are set to the value corresponding to a distance of zero.

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
        values. Currently pixel values are internally processed as
        integers.
    distance_metric: str, default='EUCLIDEAN'
        The metric for calculating distance between 2 points. Valid
        distance_metrics: 'EUCLIDEAN', 'GREAT_CIRCLE', and 'MANHATTAN'.

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

        import datashader as ds
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain, proximity

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))

        # Generate Example Terrain
        terrain_agg = generate_terrain(canvas = cvs)

        # Edit Attributes
        terrain_agg = terrain_agg.assign_attrs(
            {
                'Description': 'Example Terrain',
                'units': 'km',
                'Max Elevation': '4000',
            }
        )

        terrain_agg = terrain_agg.rename({'x': 'lon', 'y': 'lat'})
        terrain_agg = terrain_agg.rename('Elevation')

        # Generate a Target Aggregate Array
        volcano_agg = terrain_agg.copy(deep = True)
        volcano_agg = volcano_agg.where(volcano_agg.data > 2800)
        volcano_agg = volcano_agg.notnull()

        # Edit Attributes
        volcano_agg = volcano_agg.assign_attrs({'Description': 'Volcano'})
        volcano_agg = volcano_agg.rename('Volcano')

        # Create Proximity Aggregate Array
        proximity_agg = proximity(volcano_agg, x = 'lon', y = 'lat')
        proximity_agg = proximity_agg.where(
            (volcano_agg == 0) & (terrain_agg > 500)
        )

        # Edit Attributes
        proximity_agg = proximity_agg.assign_attrs(
            {
                'Description': 'Example Proximity',
                'units': 'px',
            }
        )
        proximity_agg = proximity_agg.rename('Distance')

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Volcano
        volcano_agg.plot(cmap = 'Pastel1', aspect = 2, size = 4)
        plt.title("Volcano")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Proximity
        proximity_agg.plot(cmap = 'autumn', aspect = 2, size = 4)
        plt.title("Proximity")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1264.02249454, 1261.94748873],
               [1285.37061171, 1282.48046696],
               [1306.02305679, 1303.40657515]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Terrain
            units:          km
            Max Elevation:  4000

        >>> print(volcano_agg[200:203, 200:202])
        <xarray.DataArray 'Volcano' (lat: 3, lon: 2)>
        array([[False, False],
               [False, False],
               [False, False]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            Description:  Volcano

        >>> print(proximity_agg[200:203, 200:202])
        <xarray.DataArray 'Distance' (lat: 3, lon: 2)>
        array([[4126101.19981456, 4153841.5954391 ],
               [4001421.96947258, 4025606.92456568],
               [3875484.19913922, 3897714.42999327]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            Description:  Example Proximity
            units:        px
    """
    proximity_img = _process(raster,
                             x=x,
                             y=y,
                             target_values=target_values,
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
        values. Currently pixel values are internally processed as
        integers.
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

        import datashader as ds
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import scatter
        from xrspatial import generate_terrain, allocation

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))

        # Generate Example Terrain
        terrain_agg = generate_terrain(canvas = cvs)

        # Edit Attributes
        terrain_agg = terrain_agg.assign_attrs(
            {
                'Description': 'Example Terrain',
                'units': 'km',
                'Max Elevation': '4000',
            }
        )

        terrain_agg = terrain_agg.rename({'x': 'lon', 'y': 'lat'})
        terrain_agg = terrain_agg.rename('Elevation')

        # Generate a Target Aggregate Array
        cities_df = pd.DataFrame({
            'lon': [
                -7475000,
                25000,
                15025000,
                -9975000,
                5025000,
                -14975000,
            ],
            'lat': [
                -9966666,
                6700000,
                13366666,
                3366666,
                13366666,
                13366666,
            ],
            'elevation': [
                306.5926712,
                352.50955382,
                347.20870554,
                324.11835519,
                686.31312024,
                319.34522171,
            ]
        })

        cities_da = cvs.points(cities_df,
                                x ='lon',
                                y ='lat',
                                agg = ds.max('elevation'))

        # Edit Attributes
        cities_da = cities_da.assign_attrs({'Description': 'Cities'})
        cities_da = cities_da.rename('Cities')

        # Create Allocation Aggregate Array
        allocation_agg = allocation(cities_da, x = 'lon', y = 'lat')
        allocation_agg = allocation_agg.where(terrain_agg > 500)

        # Edit Attributes
        allocation_agg = allocation_agg.assign_attrs(
            {
                'Description': 'Example Allocation',
            }
        )
        allocation_agg = allocation_agg.rename('Closest City')

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Cities
        cities_df.plot.scatter(x = 'lon', y = 'lat')
        plt.title("Cities")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Allocation
        allocation_agg.plot(cmap = 'prism', aspect = 2, size = 4)
        plt.title("Allocation")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1264.02249454, 1261.94748873],
               [1285.37061171, 1282.48046696],
               [1306.02305679, 1303.40657515]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Terrain
            units:          km
            Max Elevation:  4000

        >>> print(allocation_agg[200:203, 200:202])
        <xarray.DataArray 'Closest City' (lat: 3, lon: 2)>
        array([[352.50955382, 352.50955382],
               [352.50955382, 352.50955382],
               [352.50955382, 352.50955382]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            Description:  Example Allocation
    """
    allocation_img = _process(raster,
                              x=x,
                              y=y,
                              target_values=target_values,
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
    will be considered as "target", and all allocation will be computed
    in pixels.

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
        non-zero pixel values. Currently pixel values are
        internally processed as integers.
    distance_metric: str, default='EUCLIDEAN'
        The metric for calculating distance between 2 points. Valid
        distance_metrics: 'EUCLIDEAN', 'GREAT_CIRCLE', and 'MANHATTAN'.

    Returns
    -------
    direction_agg: xr.DataArray of same type as `raster`
        2D array of proximity values.
        All other input attributes are preserved.

    References
    ----------
        - OSGeo: https://github.com/OSGeo/gdal/blob/master/gdal/alg/gdalproximity.cpp # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import scatter
        from xrspatial import generate_terrain, direction

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))

        # Generate Example Terrain
        terrain_agg = generate_terrain(canvas = cvs)

        # Edit Attributes
        terrain_agg = terrain_agg.assign_attrs(
            {
                'Description': 'Example Terrain',
                'units': 'km',
                'Max Elevation': '4000',
            }
        )

        terrain_agg = terrain_agg.rename({'x': 'lon', 'y': 'lat'})
        terrain_agg = terrain_agg.rename('Elevation')

        # Generate a Target Aggregate Array
        cities_df = pd.DataFrame({
            'lon': [
                -7475000,
                25000,
                15025000,
                -9975000,
                5025000,
                -14975000,
            ],
            'lat': [
                -9966666,
                6700000,
                13366666,
                3366666,
                13366666,
                13366666,
            ],
            'elevation': [
                306.5926712,
                352.50955382,
                347.20870554,
                324.11835519,
                686.31312024,
                319.34522171,
            ]
        })

        cities_da = cvs.points(cities_df,
                                x ='lon',
                                y ='lat',
                                agg = ds.max('elevation'))

        # Edit Attributes
        cities_da = cities_da.assign_attrs({'Description': 'Cities'})
        cities_da = cities_da.rename('Cities')

        # Create Direction Aggregate Array
        direction_agg = direction(cities_da, x = 'lon', y = 'lat')
        direction_agg = direction_agg.where(terrain_agg > 500)

        # Edit Attributes
        direction_agg = direction_agg.assign_attrs(
            {
                'Description': 'Example Direction',
            }
        )
        direction_agg = direction_agg.rename('Cardinal Direction')

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Cities
        cities_df.plot.scatter(x = 'lon', y = 'lat')
        plt.title("Cities")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Allocation
        direction_agg.plot( aspect = 2, size = 4)
        plt.title("Direction")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    ... sourcecode:: python

        >>> print(terrain_agg[200:203, 200:202])
        <xarray.DataArray 'Elevation' (lat: 3, lon: 2)>
        array([[1264.02249454, 1261.94748873],
               [1285.37061171, 1282.48046696],
               [1306.02305679, 1303.40657515]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Terrain
            units:          km
            Max Elevation:  4000

        >>> print(direction_agg[200:203, 200:202])
        <xarray.DataArray 'Cardinal Direction' (lat: 3, lon: 2)>
        array([[90.        , 90.        ],
               [88.09084755, 88.05191498],
               [86.18592513, 86.10832367]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            Description:  Example Direction
    """
    direction_img = _process(raster,
                             x=x,
                             y=y,
                             target_values=target_values,
                             distance_metric=distance_metric,
                             process_mode=DIRECTION)

    result = xr.DataArray(direction_img,
                          coords=raster.coords,
                          dims=raster.dims,
                          attrs=raster.attrs)
    return result
