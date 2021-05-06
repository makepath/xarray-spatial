import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray

from xrspatial.utils import ngjit

from math import sqrt

from typing import Optional, Callable, Union


def stats(zones: xr.DataArray,
          values: xr.DataArray,
          stat_funcs=['mean', 'max', 'min', 'std', 'var', 'count']):
    """
    Calculate summary statistics for each zone defined by a zone
    dataset, based on values aggregate.

    A single output value is computed for every zone in the input zone
    dataset.

    Parameters
    ----------
    zones : xr.DataArray
        zones.values is a 2d array of integers.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input zone layer defines
        the shape, values, and locations of the zones. An integer field
        in the zone input is specified to define the zones.
    values : xr.DataArray
        values.values is a 2d array of integers or floats.
        The input value raster contains the input values used in
        calculating the output statistic for each zone.
    stat_funcs : list of string or dict, default=['mean', 'max', 'min',
        'std', 'var', 'count'])
        Which statistics to calculate for each zone. If a list, possible
        choices are subsets of ['mean', 'max', 'min', 'std', 'var',
        'count']. In the dictionary case, all of its values must be
        callable. Function takes only one argument that is the zone
        values. The key become the column name in the output DataFrame.

    Returns
    -------
    stats_df : pandas.DataFrame
        A pandas DataFrame where each column is a statistic and each
        row is a zone with zone id.

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain
        from xrspatial.classify import equal_interval
        from xrspatial.zonal import stats

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))


        # Generate Values
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

        # Create Zones
        equal_interval_agg = equal_interval(
            agg = terrain_agg,
            name = 'Elevation',
        )
        equal_interval_agg = equal_interval_agg.astype('int')

        # Edit Attributes
        equal_interval_agg = equal_interval_agg.assign_attrs(
            {
                'Description': 'Example Equal Interval',
            }
        )

        # Plot Terrain (Values)
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain (Values)")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Equal Interval (Zones)
        equal_interval_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Equal Interval (Zones)")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> # Calculate Stats
        >>> stats_agg = stats(zones = equal_interval_agg, values = terrain_agg)
        >>> print(stats_agg)
                  mean          max          min         std           var    count # noqa
        1  1346.099206  1599.980772  1200.014238  107.647012  11587.879265  52698.0 # noqa
        2  1867.613738  2399.949943  1600.049783  207.072933  42879.199507  22987.0 # noqa
        3  2716.967940  3199.499889  2400.079093  215.475764  46429.804756   4926.0 # noqa
        4  3491.072129  4000.000000  3200.057209  182.752194  33398.364467   1373.0 # noqa

        >>> # Custom Stats
        >>> custom_stats ={'sum': lambda val: val.sum()}
        >>> custom_stats_agg = stats(zones = equal_interval_agg,
                                     values = terrain_agg,
                                     stat_funcs=custom_stats)
        >>> print(custom_stats_agg)
                    sum
        1  7.093674e+07
        2  4.293084e+07
        3  1.338378e+07
        4  4.793242e+06
    """
    if zones.shape != values.shape:
        raise ValueError(
            "`zones` and `values` must have same shape")

    if not issubclass(zones.data.dtype.type, np.integer):
        raise ValueError("`zones` must be an array of integers")

    if not (issubclass(values.data.dtype.type, np.integer) or
            issubclass(values.data.dtype.type, np.floating)):
        raise ValueError(
            "`values` must be an array of integers or floats")

    # do not consider zone with 0s
    unique_zones = np.unique(zones.data[np.where(zones.data != 0)])

    # mask out all invalid values such as: nan, inf
    masked_values = np.ma.masked_invalid(values.data)

    if isinstance(stat_funcs, dict):
        stats_df = pd.DataFrame(columns=[*stat_funcs])
        for zone_id in unique_zones:
            # get zone values
            zone_values = np.ma.masked_where(zones.data != zone_id,
                                             masked_values)
            zone_stats = []
            for stat in stat_funcs:
                stat_func = stat_funcs.get(stat)
                if not callable(stat_func):
                    raise ValueError(stat)
                zone_stats.append(stat_func(zone_values))
            stats_df.loc[zone_id] = zone_stats
    else:
        stats_df = pd.DataFrame(columns=stat_funcs)

        for zone_id in unique_zones:
            # get zone values
            zone_values = np.ma.masked_where(zones.data != zone_id,
                                             masked_values)
            zone_stats = []
            for stat in stat_funcs:
                if stat == 'mean':
                    zone_stats.append(zone_values.mean())
                elif stat == 'max':
                    zone_stats.append(zone_values.max())
                elif stat == 'min':
                    zone_stats.append(zone_values.min())
                elif stat == 'std':
                    zone_stats.append(zone_values.std())
                elif stat == 'var':
                    zone_stats.append(zone_values.var())
                elif stat == 'count':
                    zone_stats.append(np.ma.count(zone_values))
                else:
                    err_str = 'Invalid stat name. ' \
                              + '\'' + stat + '\' option not supported.'
                    raise ValueError(err_str)
            stats_df.loc[zone_id] = zone_stats

    return stats_df


def _crosstab_2d(zones, values):
    # do not consider zone with 0s
    unique_zones = np.unique(zones.data[np.where(zones.data != 0)])

    # mask out all invalid values such as: nan, inf
    masked_values = np.ma.masked_invalid(values.data)

    # categories
    cats = np.unique(masked_values[masked_values.mask == False]).data # noqa

    # return of the function
    # columns are categories
    crosstab_df = pd.DataFrame(columns=cats)

    for zone_id in unique_zones:
        # get zone values
        zone_values = np.ma.masked_where(zones.data != zone_id, masked_values)
        zone_cat_counts = np.zeros((len(cats),))
        for i, cat in enumerate(cats):
            zone_cat_counts[i] = len(np.where(zone_values == cat)[0])
        if np.sum(zone_cat_counts) != 0:
            zone_cat_stats = zone_cat_counts / np.sum(zone_cat_counts)
        # percentage of each category over the zone
        crosstab_df.loc[zone_id] = zone_cat_stats

    return crosstab_df


def _crosstab_3d(zones, values, layer):
    if layer is None:
        cats = values.indexes[values.dims[-1]].values
    else:
        if layer not in values.dims:
            raise ValueError("`layer` does not exist in `values` agg.")
        cats = values[layer].values

    num_cats = len(cats)

    # do not consider zone with 0s
    unique_zones = np.unique(zones.data[np.where(zones.data != 0)])

    # mask out all invalid values such as: nan, inf
    masked_values = np.ma.masked_invalid(values.data)

    # return of the function
    # columns are categories
    crosstab_df = pd.DataFrame(columns=cats)

    for zone_id in unique_zones:
        # get all entries in zones with zone_id
        zone_entries = zones.data == zone_id
        zones_entries_3d = np.repeat(zone_entries[:, :, np.newaxis],
                                     num_cats, axis=-1)
        zone_values = zones_entries_3d * masked_values
        zone_cat_stats = [np.sum(zone_cat) for zone_cat in zone_values.T]
        sum_zone_cats = sum(zone_cat_stats)
        if sum_zone_cats != 0:
            zone_cat_stats = zone_cat_stats / sum_zone_cats
        # percentage of each category over the zone
        crosstab_df.loc[zone_id] = zone_cat_stats

    return crosstab_df


def crosstab(zones: xr.DataArray,
             values: xr.DataArray,
             layer: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate cross-tabulated (categorical stats) areas
    between two datasets: a zone dataset, a value dataset (a value
    raster). Outputs a pandas DataFrame.

    Requires a DataArray with a single data dimension, here called the
    "values", indexed using 3D coordinates.

    DataArrays with 3D coordinates are expected to contain values
    distributed over different categories that are indexed by the
    additional coordinate.  Such an array would reduce to the
    2D-coordinate case if collapsed across the categories (e.g. if one
    did ``aggc.sum(dim='cat')`` for a categorical dimension ``cat``).

    Parameters
    ----------
    zones : xr.DataArray
        zones.values is a 2d array of integers.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input zone layer defines
        the shape, values, and locations of the zones. An integer field
        in the zone input is specified to define the zones.
    values : xr.DataArray
        values.values is a 3d array of integers or floats.
        The input value raster contains the input values used in
        calculating the categorical statistic for each zone.
    layer: str, default=None
        name of the layer inside the `values` DataArray for getting
        the values.

    Returns
    -------
    crosstab_df : pandas.DataFrame
        A pandas DataFrame where each column is a categorical value
        and each row is a zone with zone id.
        Each entry presents the percentage of the category over the zone.

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain
        from xrspatial.classify import equal_interval
        from xrspatial.zonal import crosstab

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))


        # Generate Values
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

        # Create Zones
        equal_interval_agg = equal_interval(
            agg = terrain_agg,
            name = 'Elevation',
        )
        equal_interval_agg = equal_interval_agg.astype('int')

        # Edit Attributes
        equal_interval_agg = equal_interval_agg.assign_attrs(
            {
                'Description': 'Example Equal Interval',
            }
        )

        # Plot Terrain (Values)
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain (Values)")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Equal Interval (Zones)
        equal_interval_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Equal Interval (Zones)")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> # Calculate Crosstab
        >>> crosstab_agg = crosstab(
                zones=equal_interval_agg,
                values=terrain_agg,
            )
        >>> print(crosstab_agg)
           0.000000     1200.014238  1200.014864  1200.021077  1200.027001
        1          0.0     0.000019     0.000019     0.000019     0.000019
        2          0.0     0.000000     0.000000     0.000000     0.000000
        3          0.0     0.000000     0.000000     0.000000     0.000000
        4          0.0     0.000000     0.000000     0.000000     0.000000

           1200.030954  1200.034319  1200.038300  1200.042588  1200.042805
        1     0.000019     0.000019     0.000019     0.000019     0.000019
        2     0.000000     0.000000     0.000000     0.000000     0.000000
        3     0.000000     0.000000     0.000000     0.000000     0.000000
        4     0.000000     0.000000     0.000000     0.000000     0.000000

           3932.764078  3946.456024  3947.505026  3952.279254  3957.748147
        1     0.000000     0.000000     0.000000     0.000000     0.000000
        2     0.000000     0.000000     0.000000     0.000000     0.000000
        3     0.000000     0.000000     0.000000     0.000000     0.000000
        4     0.000728     0.000728     0.000728     0.000728     0.000728

           3968.600152  3973.840684  3989.509344  3998.678232  4000.000000
        1     0.000000     0.000000     0.000000     0.000000     0.000000
        2     0.000000     0.000000     0.000000     0.000000     0.000000
        3     0.000000     0.000000     0.000000     0.000000     0.000000
        4     0.000728     0.000728     0.000728     0.000728     0.000728

        [4 rows x 81984 columns]
    """
    if not isinstance(zones, xr.DataArray):
        raise TypeError("zones must be instance of DataArray")

    if not isinstance(values, xr.DataArray):
        raise TypeError("values must be instance of DataArray")

    if zones.ndim != 2:
        raise ValueError("zones must be 2D")

    if zones.shape != values.shape[:2]:
        raise ValueError(
            "Incompatible shapes between `zones` and `values`")

    if not issubclass(zones.data.dtype.type, np.integer):
        raise ValueError("`zones` must be an xarray of integers")

    if not issubclass(values.data.dtype.type, np.integer) and \
            not issubclass(values.data.dtype.type, np.floating):
        raise ValueError(
            "`values` must be an xarray of integers or floats")

    if values.ndim == 3:
        return _crosstab_3d(zones, values, layer)
    elif values.ndim == 2:
        return _crosstab_2d(zones, values)
    else:
        raise ValueError("`values` must use either 2D or 3D coordinates.")


def apply(zones: xr.DataArray,
          values: xr.DataArray,
          func: Callable):
    """
    Apply a function to the `values` agg within zones in `zones` agg.
    Change the agg content.

    Parameters
    ----------
    zones : xr.DataArray
        zones.values is a 2d array of integers. A zone is all the cells
        in a raster that have the same value, whether or not they are
        contiguous. The input zone layer defines the shape, values, and
        locations of the zones. An integer field in the zone input is
        specified to define the zones.
    agg : xr.DataArray
        agg.values is either a 2D or 3D array of integers or floats.
        The input value raster.
    func : callable function to apply.

    Examples
    --------
    .. sourcecode:: python

        >>> zones_val = np.array([[1, 1, 0, 2],
        >>>                       [0, 2, 1, 2]])
        >>> zones = xarray.DataArray(zones_val)
        >>> values_val = np.array([[2, -1, 5, 3],
        >>>                        [3, np.nan, 20, 10]])
        >>> agg = xarray.DataArray(values_val)
        >>> func = lambda x: 0
        >>> apply(zones, agg, func)
        >>> agg
        array([[0, 0, 5, 0],
               [3, 0, 0, 0]])
    """
    if not isinstance(zones, xr.DataArray):
        raise TypeError("zones must be instance of DataArray")

    if not isinstance(values, xr.DataArray):
        raise TypeError("values must be instance of DataArray")

    if zones.ndim != 2:
        raise ValueError("zones must be 2D")

    if values.ndim != 2 and values.ndim != 3:
        raise ValueError("values must be either 2D or 3D coordinates")

    if zones.shape != values.shape[:2]:
        raise ValueError(
            "Incompatible shapes between `zones` and `values`")

    if not issubclass(zones.values.dtype.type, np.integer):
        raise ValueError("`zones.values` must be an array of integers")

    if not (issubclass(values.values.dtype.type, np.integer) or
            issubclass(values.values.dtype.type, np.floating)):
        raise ValueError(
            "`values` must be an array of integers or float")

    # entries of zone 0 remain the same
    remain_entries = zones.data == 0

    # entries with a non-zero zone value
    zones_entries = zones.data != 0

    if len(values.shape) == 3:
        z = values.shape[-1]
        # add new z-dimension in case 3D `values` aggregate
        remain_entries = np.repeat(remain_entries[:, :, np.newaxis], z,
                                   axis=-1)
        zones_entries = np.repeat(zones_entries[:, :, np.newaxis], z, axis=-1)

    remain_mask = np.ma.masked_array(values.data, mask=remain_entries)
    zones_mask = np.ma.masked_array(values.data, mask=zones_entries)

    # apply func to corresponding `values` of `zones`
    vfunc = np.vectorize(func)
    values_func = vfunc(zones_mask)
    values.values = remain_mask.data * remain_mask.mask \
        + values_func.data * values_func.mask


def get_full_extent(crs):
    """
    Returns the full extent of a map projection, available projections
    are 'Mercator' and 'Geographic'.

    Parameters
    ----------
    crs : str
        Input projection.

    Returns
    -------
    extent : tuple
        Projection extent of form ((min_x, max_x), (min_y, max_y)).

    Examples
    --------
    .. sourcecode:: python

        >>> from xrspatial.zonal import get_full_extent

        >>> extent = get_full_extent('Mercator')
        >>> print(extent)
        ((-20000000.0, 20000000.0), (-20000000.0, 20000000.0))
    """
    Mercator = (-20e6, 20e6), (-20e6, 20e6)
    Geographic = (-180, 180), (-90, 90)

    def _crs_code_mapping():
        CRS_CODES = {}
        CRS_CODES['Mercator'] = Mercator
        CRS_CODES['Geographic'] = Geographic
        return CRS_CODES

    CRS_CODES = _crs_code_mapping()
    return CRS_CODES[crs]


def suggest_zonal_canvas(smallest_area: Union[int, float],
                         x_range: Union[tuple, list],
                         y_range: Union[tuple, list],
                         crs: str = 'Mercator',
                         min_pixels: int = 25) -> tuple:
    """
    Given a coordinate reference system (crs), a set of polygons with
    corresponding x range and y range, calculate the height and width
    of canvas so that the smallest polygon (polygon with smallest area)
    is rasterized with at least min pixels.

    Currently, we assume that the smallest polygon does not intersect
    others. One should note that a polygon can have different shapes
    when it is rasterized in canvases of different size. Thus, we cannot
    100% guarantee the actual number of pixels after rasterization.
    It is recommended to add an additional of 5% to @min_pixels parameter.

    Parameters
    ----------
    x_range : tuple or list of float or int
        The full x extent of the polygon GeoDataFrame.
    y_range : tuple or list of float or int
        The full y extent of the polygon GeoDataFrame.
    smallest_area : float or int
        Area of the smallest polygon.
    crs : str, default='Mercator'
        Name of the coordinate reference system.
    min_pixels : int, default=25
        Expected number of pixels of the polygon with smallest area
        when the whole dataframe is rasterized.

    Returns
    -------
    height, width: int
        Height and width of the canvas in pixel space.

    Examples
    --------
    .. sourcecode:: python

        >>> # Imports
        >>> from spatialpandas import GeoDataFrame
        >>> import geopandas as gpd
        >>> import datashader as ds

        >>> df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        >>> df = df.to_crs("EPSG:3857")
        >>> df = df[df.continent != 'Antarctica']
        >>> df['id'] = [i for i in range(len(df.index))]
        >>> xmin, ymin, xmax, ymax = (
                df.bounds.minx.min(),
                df.bounds.miny.min(),
                df.bounds.maxx.max(),
                df.bounds.maxy.max(),
            )
        >>> x_range = (xmin, xmax)
        >>> y_range = (ymin, ymax)
        >>> smallest_area = df.area.min()
        >>> min_pixels = 20
        >>> height, width = suggest_zonal_canvas(
                x_range=x_range,
                y_range=y_range,
                smallest_area=smallest_area,
                crs='Mercator',
                min_pixels=min_pixels,
            )
        >>> cvs = ds.Canvas(x_range=x_range, y_range=y_range,
        >>>             plot_height=height, plot_width=width)
        >>> spatial_df = GeoDataFrame(df, geometry='geometry')
        >>> agg = cvs.polygons(spatial_df, 'geometry', agg=ds.max('id'))
        >>> min_poly_id = df.area.argmin()
        >>> actual_min_pixels = len(np.where(agg.data==min_poly_id)[0])
    """
    full_xrange, full_yrange = get_full_extent(crs)
    xmin, xmax = full_xrange
    ymin, ymax = full_yrange
    aspect_ratio = (xmax - xmin) / (ymax - ymin)

    # area that a pixel stands for
    pixel_area = smallest_area / min_pixels
    # total_area of whole GeoDataFrame
    total_area = (xmax - xmin) * (ymax - ymin)
    # total pixels needed
    total_pixels = total_area / pixel_area
    # We have, h * w = total_pixels
    # and,     w / h = aspect_ratio
    # Thus,    aspect_ratio * h**2 = total_pixels
    h = sqrt(total_pixels / aspect_ratio)
    w = aspect_ratio * h
    canvas_h = int(h * (y_range[1] - y_range[0]) / (ymax - ymin))
    canvas_w = int(w * (x_range[1] - x_range[0]) / (xmax - xmin))

    return canvas_h, canvas_w


@ngjit
def _area_connectivity(data, n=4):
    out = np.zeros_like(data)
    rows, cols = data.shape
    uid = 1

    src_window = np.zeros(shape=(n,), dtype=data.dtype)
    area_window = np.zeros(shape=(n,), dtype=data.dtype)

    for y in range(0, rows):
        for x in range(0, cols):

            val = data[y, x]

            if np.isnan(val):
                out[y, x] = val
                continue

            if n == 8:
                src_window[0] = data[max(y-1, 0), max(x-1, 0)]
                src_window[1] = data[y, max(x-1, 0)]
                src_window[2] = data[min(y+1, rows-1), max(x-1, 0)]
                src_window[3] = data[max(y-1, 0), x]
                src_window[4] = data[min(y+1, rows-1), x]
                src_window[5] = data[max(y-1, 0), min(x+1, cols-1)]
                src_window[6] = data[y, min(x+1, cols-1)]
                src_window[7] = data[min(y+1, rows-1), min(x+1, cols-1)]

                area_window[0] = out[max(y-1, 0), max(x-1, 0)]
                area_window[1] = out[y, max(x-1, 0)]
                area_window[2] = out[min(y+1, rows-1), max(x-1, 0)]
                area_window[3] = out[max(y-1, 0), x]
                area_window[4] = out[min(y+1, rows-1), x]
                area_window[5] = out[max(y-1, 0), min(x+1, cols-1)]
                area_window[6] = out[y, min(x+1, cols-1)]
                area_window[7] = out[min(y+1, rows-1), min(x+1, cols-1)]

            else:
                src_window[0] = data[y, max(x-1, 0)]
                src_window[1] = data[max(y-1, 0), x]
                src_window[2] = data[min(y+1, rows-1), x]
                src_window[3] = data[y, min(x+1, cols-1)]

                area_window[0] = out[y, max(x-1, 0)]
                area_window[1] = out[max(y-1, 0), x]
                area_window[2] = out[min(y+1, rows-1), x]
                area_window[3] = out[y, min(x+1, cols-1)]

            # check in has matching value in neighborhood
            rtol = 1e-05
            atol = 1e-08
            is_close = np.abs(src_window - val) <= (atol + rtol * np.abs(val))
            neighbor_matches = np.where(is_close)[0]

            if len(neighbor_matches) > 0:

                # check in has area already assigned
                assigned_value = None
                for j in range(len(neighbor_matches)): # NOQA
                    area_val = area_window[neighbor_matches[j]]
                    if area_val > 0:
                        assigned_value = area_val
                        break

                if assigned_value is not None:
                    out[y, x] = assigned_value
                else:
                    out[y, x] = uid
                    uid += 1
            else:
                out[y, x] = uid
                uid += 1

    for y in range(0, rows):
        for x in range(0, cols):

            if n == 8:
                src_window[0] = data[max(y-1, 0), max(x-1, 0)]
                src_window[1] = data[y, max(x-1, 0)]
                src_window[2] = data[min(y+1, rows-1), max(x-1, 0)]
                src_window[3] = data[max(y-1, 0), x]
                src_window[4] = data[min(y+1, rows-1), x]
                src_window[5] = data[max(y-1, 0), min(x+1, cols-1)]
                src_window[6] = data[y, min(x+1, cols-1)]
                src_window[7] = data[min(y+1, rows-1), min(x+1, cols-1)]

                area_window[0] = out[max(y-1, 0), max(x-1, 0)]
                area_window[1] = out[y, max(x-1, 0)]
                area_window[2] = out[min(y+1, rows-1), max(x-1, 0)]
                area_window[3] = out[max(y-1, 0), x]
                area_window[4] = out[min(y+1, rows-1), x]
                area_window[5] = out[max(y-1, 0), min(x+1, cols-1)]
                area_window[6] = out[y, min(x+1, cols-1)]
                area_window[7] = out[min(y+1, rows-1), min(x+1, cols-1)]

            else:
                src_window[0] = data[y, max(x-1, 0)]
                src_window[1] = data[max(y-1, 0), x]
                src_window[2] = data[min(y+1, rows-1), x]
                src_window[3] = data[y, min(x+1, cols-1)]

                area_window[0] = out[y, max(x-1, 0)]
                area_window[1] = out[max(y-1, 0), x]
                area_window[2] = out[min(y+1, rows-1), x]
                area_window[3] = out[y, min(x+1, cols-1)]

            val = data[y, x]

            if np.isnan(val):
                continue

            # check in has matching value in neighborhood
            rtol = 1e-05
            atol = 1e-08
            is_close = np.abs(src_window - val) <= (atol + rtol * np.abs(val))
            neighbor_matches = np.where(is_close)[0]

            # check in has area already assigned
            assigned_values_min = None
            for j in range(len(neighbor_matches)):
                area_val = area_window[neighbor_matches[j]]
                nn = assigned_values_min is not None
                if nn and assigned_values_min != area_val:
                    if assigned_values_min > area_val:

                        # replace
                        for y1 in range(0, rows):
                            for x1 in range(0, cols):
                                if out[y1, x1] == assigned_values_min:
                                    out[y1, x1] = area_val

                        assigned_values_min = area_val

                    else:
                        # replace
                        for y1 in range(0, rows):
                            for x1 in range(0, cols):
                                if out[y1, x1] == area_val:
                                    out[y1, x1] = assigned_values_min

                elif assigned_values_min is None:
                    assigned_values_min = area_val

    return out


def regions(raster: xr.DataArray,
            neighborhood: int = 4,
            name: str = 'regions') -> xr.DataArray:
    """
    Create unique regions of raster based on pixel value connectivity.
    Connectivity can be based on either 4 or 8-pixel neighborhoods.
    Output raster contain a unique int for each connected region.

    Parameters
    ----------
    raster : xr.DataArray
    connections : int, default=4
        4 or 8 pixel-based connectivity.
    name: str, default='regions'
        output xr.DataArray.name property.

    Returns
    -------
    regions_agg : xarray.DataArray

    References
    ----------
        - Tomislav Hengl: http://spatial-analyst.net/ILWIS/htm/ilwisapp/areanumbering_algorithm.htm # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain
        from xrspatial.zonal import regions

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))


        # Generate Values
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

        # Create Regions
        regions_agg = regions(terrain_agg)

        # Edit Attributes
        regions_agg = regions_agg.assign_attrs({'Description': 'Example Regions',
                                                 'units': ''})
        regions_agg = regions_agg.rename('Region Value')

        # Plot Terrain (Values)
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain (Values)")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Regions
        regions_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Regions")
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

        >>> print(regions_agg[200:203, 200:202])
        <xarray.DataArray 'Region Value' (lat: 3, lon: 2)>
        array([[39557., 39558.],
               [39943., 39944.],
               [40327., 40328.]])
        Coordinates:
          * lon      (lon) float64 -3.96e+06 -3.88e+06
          * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            1
            Description:    Example Regions
            units:
            Max Elevation:  4000
    """
    if neighborhood not in (4, 8):
        raise ValueError('`neighborhood` value must be either 4 or 8)')

    out = _area_connectivity(raster.data, n=neighborhood)

    return DataArray(out, name=name,
                     dims=raster.dims,
                     coords=raster.coords, attrs=raster.attrs)


def _bool_crop(arr, rows_flags, cols_flags):
    top = np.argwhere(rows_flags).flatten()[0]
    bottom = np.argwhere(rows_flags).flatten()[-1]
    left = np.argwhere(cols_flags).flatten()[0]
    right = np.argwhere(cols_flags).flatten()[-1]
    return arr[top:bottom+1, left:right+1]


@ngjit
def _trim(data, excludes):

    rows, cols = data.shape

    # find empty top rows
    top = 0
    scan_complete = False
    for y in range(rows):

        if scan_complete:
            break

        top = y
        for x in range(cols):
            val = data[y, x]
            is_nodata = False
            for e in excludes:
                if e == val:
                    is_nodata = True
                    break

            if not is_nodata:
                scan_complete = True
                break

    # find empty bottom rows
    bottom = 0
    scan_complete = False
    for y in range(rows-1, -1, -1):
        if scan_complete:
            break
        bottom = y
        for x in range(cols):
            val = data[y, x]
            is_nodata = False
            for e in excludes:
                if e == val:
                    is_nodata = True
                    break
            if not is_nodata:
                scan_complete = True
                break

    # find empty left cols
    left = 0
    scan_complete = False
    for x in range(cols):
        if scan_complete:
            break
        left = x
        for y in range(rows):
            val = data[y, x]
            is_nodata = False
            for e in excludes:
                if e == val:
                    is_nodata = True
                    break
            if not is_nodata:
                scan_complete = True
                break

    # find empty right cols
    right = 0
    scan_complete = False
    for x in range(cols-1, -1, -1):
        if scan_complete:
            break
        right = x
        for y in range(rows):
            val = data[y, x]
            is_nodata = False
            for e in excludes:
                if e == val:
                    is_nodata = True
                    break
            if not is_nodata:
                scan_complete = True
                break

    return top, bottom, left, right


def trim(raster: xr.DataArray,
         values: Union[list, tuple] = (np.nan,),
         name: str = 'trim') -> xr.DataArray:
    """
    Trim scans from the edges and eliminates rows / cols which only
    contain the values supplied.

    Parameters
    ----------
    raster: xr.DataArray
    values: list or tuple, default=(np.nan)
        List of zone ids to trim from raster edge.
    name: str, default='trim'
        Output xr.DataArray.name property.

    Returns
    -------
    trim_agg : xarray.DataArray

    Notes
    -----
        - This operation will change the output size of the raster.

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import numpy as np
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain
        from xrspatial.zonal import trim

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))


        # Generate Terrain
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
        terrain_agg = terrain_agg.astype('int')

        # Trim Image
        trimmed_agg = trim(raster = terrain_agg, values = [0])

        # Edit Attributes
        trimmed_agg = trimmed_agg.assign_attrs({'Description': 'Example Trim'})

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Trimmed Terrain
        trimmed_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Trim")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg.shape)
        (300, 500)

        >>> print(terrain_agg.attrs)
        {
            'res': 1,
            'Description': 'Example Terrain',
            'units': 'km',
            'Max Elevation': '4000',
        }

        >>> print(trimmed_agg.shape)
        (268, 500)

        >>> print(trimmed_agg.attrs)
        {
            'res': 1,
            'Description': 'Example Trim',
            'units': 'km',
            'Max Elevation': '4000',
        }
    """
    top, bottom, left, right = _trim(raster.data, values)
    arr = raster[top:bottom+1, left:right+1]
    arr.name = name
    return arr


@ngjit
def _crop(data, values):

    rows, cols = data.shape

    top = -1
    bottom = -1
    left = -1
    right = -1

    # find empty top rows
    top = 0
    scan_complete = False
    for y in range(rows):

        if scan_complete:
            break

        top = y

        for x in range(cols):
            val = data[y, x]
            for v in values:
                if v == val:
                    scan_complete = True
                    break
                else:
                    continue

            if scan_complete:
                break

    # find empty bottom rows
    bottom = 0
    scan_complete = False
    for y in range(rows-1, -1, -1):

        if scan_complete:
            break

        bottom = y

        for x in range(cols):
            val = data[y, x]
            for e in values:
                if e == val:
                    scan_complete = True
                    break
                else:
                    continue

            if scan_complete:
                break

    # find empty left cols
    left = 0
    scan_complete = False
    for x in range(cols):

        if scan_complete:
            break

        left = x

        for y in range(rows):
            val = data[y, x]
            for e in values:
                if e == val:
                    scan_complete = True
                    break
                else:
                    continue

            if scan_complete:
                break

    # find empty right cols
    right = 0
    scan_complete = False
    for x in range(cols-1, -1, -1):
        if scan_complete:
            break
        right = x
        for y in range(rows):
            val = data[y, x]
            for e in values:
                if e == val:
                    scan_complete = True
                    break
                else:
                    continue

            if scan_complete:
                break

    return top, bottom, left, right


def crop(zones: xr.DataArray,
         values: xr.DataArray,
         zones_ids: Union[list, tuple],
         name: str = 'crop'):
    """
    Crop scans from edges and eliminates rows / cols until one of the
    input values is found.

    Parameters
    ----------
    zones : xr.DataArray
        Input zone raster.

    values: xr.DataArray
        Input values raster.

    zones_ids : list or tuple
        List of zone ids to crop raster.

    name: str, default='crop'
        Output xr.DataArray.name property.

    Returns
    -------
    crop_agg : xarray.DataArray

    Notes
    -----
        - This operation will change the output size of the raster.

    Examples
    --------
    .. plot::
       :include-source:

        import datashader as ds
        import matplotlib.pyplot as plt
        from xrspatial import generate_terrain
        from xrspatial.zonal import crop

        # Create Canvas
        W = 500
        H = 300
        cvs = ds.Canvas(plot_width = W,
                        plot_height = H,
                        x_range = (-20e6, 20e6),
                        y_range = (-20e6, 20e6))


        # Generate Zones
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

        # Crop Image
        values_agg = terrain_agg[0:300, 0:250]
        cropped_agg = crop(
            zones=terrain_agg,
            values=values_agg,
            zones_ids=[0],
        )

        # Edit Attributes
        cropped_agg = cropped_agg.assign_attrs({'Description': 'Example Crop'})

        # Plot Terrain
        terrain_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Terrain")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

        # Plot Cropped Terrain
        cropped_agg.plot(cmap = 'terrain', aspect = 2, size = 4)
        plt.title("Crop")
        plt.ylabel("latitude")
        plt.xlabel("longitude")

    .. sourcecode:: python

        >>> print(terrain_agg.shape)
        (300, 500)

        >>> print(terrain_agg.attrs)
        {
            'res': 1,
            'Description': 'Example Terrain',
            'units': 'km',
            'Max Elevation': '4000',
        }

        >>> print(cropped_agg.shape)
        (300, 250)

        >>> print(cropped_agg.attrs)
        {
            'res': 1,
            'Description': 'Example Crop',
            'units': 'km',
            'Max Elevation': '4000',
        }
    """
    top, bottom, left, right = _crop(zones.data, zones_ids)
    arr = values[top:bottom+1, left:right+1]
    arr.name = name
    return arr
