import warnings

import numpy as np
import pandas as pd
import xarray as xa
from xarray import DataArray

from xrspatial.utils import ngjit

from math import sqrt


warnings.simplefilter('default')


def zonal_stats(zones, values,
                stat_funcs=['mean', 'max', 'min', 'std', 'var']):
    warnings.warn('\'zonal_stats\' is deprecated. Use \'stats\' instead',
                  DeprecationWarning)

    return stats(zones, values, stat_funcs)


def stats(zones, values, stat_funcs=['mean', 'max', 'min', 'std',
                                     'var', 'count']):
    """Calculate summary statistics for each zone defined by a zone dataset,
    based on values aggregate.

    A single output value is computed for every zone in the input zone dataset.

    Parameters
    ----------
    zones: xarray.DataArray,
        zones.values is a 2d array of integers.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input zone layer defines
        the shape, values, and locations of the zones. An integer field
        in the zone input is specified to define the zones.

    values: xarray.DataArray,
        values.values is a 2d array of integers or floats.
        The input value raster contains the input values used in calculating
        the output statistic for each zone.

    stat_funcs: list of strings or dictionary<stat_name: func(zone_values)>.
        Which statistics to calculate for each zone.
        If a list, possible choices are subsets of
            ['mean', 'max', 'min', 'std', 'var', 'count']
        In the dictionary case, all of its values must be callable.
            Function takes only one argument that is the zone values.
            The key become the column name in the output DataFrame.

    Returns
    -------
    stats_df: pandas.DataFrame
        A pandas DataFrame where each column is a statistic
        and each row is a zone with zone id.

    Examples
    --------
    >>> zones_val = np.array([[1, 1, 0, 2],
    >>>                      [0, 2, 1, 2]])
    >>> zones = xarray.DataArray(zones_val)
    >>> values_val = np.array([[2, -1, 5, 3],
    >>>                       [3, np.nan, 20, 10]])
    >>> values = xarray.DataArray(values_val)

    # default setting
    >>> df = stats(zones, values)
    >>> df
        mean	max 	min 	std     	var
    1	7.0 	20.0	-1.0	9.273618	86.00
    2	6.5	    10.0   	3.0	    3.500000	12.25

    # custom stat
    >>> custom_stats ={'sum': lambda val: val.sum()}
    >>> df = stats(zones, values, stat_funcs=custom_stats)
    >>> df
        sum
    1	21.0
    2	13.0
    """

    if zones.shape != values.shape:
        raise ValueError(
            "`zones` and `values` must have same shape")

    if not issubclass(zones.data.dtype.type, np.integer):
        raise ValueError("`zones` must be an array of integers")

    if not (issubclass(values.data.dtype.type, np.integer) or
            issubclass(values.data.dtype.type, np.float)):
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
    cats = np.unique(masked_values[masked_values.mask==False]).data

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


def crosstab(zones, values, layer=None):
    """Calculate cross-tabulated (categorical stats) areas
    between two datasets: a zone dataset, a value dataset (a value raster).
    Outputs a pandas DataFrame.

    Requires a DataArray with a single data dimension, here called the
    "values", indexed using 3D coordinates.

    DataArrays with 3D coordinates are expected to contain values
    distributed over different categories that are indexed by the
    additional coordinate.  Such an array would reduce to the
    2D-coordinate case if collapsed across the categories (e.g. if one
    did ``aggc.sum(dim='cat')`` for a categorical dimension ``cat``).

    Parameters
    ----------
    zones: xarray.DataArray,
        zones.values is a 2d array of integers.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input zone layer defines
        the shape, values, and locations of the zones. An integer field
        in the zone input is specified to define the zones.

    values: xarray.DataArray,
        values.values is a 3d array of integers or floats.
        The input value raster contains the input values used in calculating
        the categorical statistic for each zone.

    layer: string (optional)
        name of the layer inside the `values` DataArray
        for getting the values
    Returns
    -------
    crosstab_df: pandas.DataFrame
        A pandas DataFrame where each column is a categorical value
        and each row is a zone with zone id.
        Each entry presents the percentage of the category over the zone.
    """

    if not isinstance(zones, xa.DataArray):
        raise TypeError("zones must be instance of DataArray")

    if not isinstance(values, xa.DataArray):
        raise TypeError("values must be instance of DataArray")

    if zones.ndim != 2:
        raise ValueError("zones must be 2D")

    if zones.shape != values.shape[:2]:
        raise ValueError(
            "Incompatible shapes between `zones` and `values`")

    if not issubclass(zones.data.dtype.type, np.integer):
        raise ValueError("`zones` must be an xarray of integers")

    if not issubclass(values.data.dtype.type, np.integer) and \
            not issubclass(values.data.dtype.type, np.float):
        raise ValueError(
            "`values` must be an xarray of integers or floats")

    if values.ndim == 3:
        return _crosstab_3d(zones, values, layer)
    elif values.ndim == 2:
        return _crosstab_2d(zones, values)
    else:
        raise ValueError("`values` must use either 2D or 3D coordinates.")


def apply(zones, values, func):
    """Apply a function to the `values` agg within zones in `zones` agg.
     Change the agg content.

    Parameters
    ----------
    zones: xarray.DataArray,
        zones.values is a 2d array of integers.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input zone layer defines
        the shape, values, and locations of the zones. An integer field
        in the zone input is specified to define the zones.
    agg: xarray.DataArray,
        agg.values is either a 2D or 3D array of integers or floats.
        The input value raster.
    func: callable function to apply.

    Returns
    -------

    Examples
    --------
    >>> zones_val = np.array([[1, 1, 0, 2],
    >>>                      [0, 2, 1, 2]])
    >>> zones = xarray.DataArray(zones_val)
    >>> values_val = np.array([[2, -1, 5, 3],
    >>>                       [3, np.nan, 20, 10]])
    >>> agg = xarray.DataArray(values_val)
    >>> func = lambda x: 0
    >>> apply(zones, agg, func)
    >>> agg
    >>> array([[0, 0, 5, 0],
    >>>        [3, 0, 0, 0]])
    """

    if not isinstance(zones, xa.DataArray):
        raise TypeError("zones must be instance of DataArray")

    if not isinstance(values, xa.DataArray):
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
            issubclass(values.values.dtype.type, np.float)):
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
    Mercator = (-20e6, 20e6), (-20e6, 20e6)
    Geographic = (-180, 180), (-90, 90)

    def _crs_code_mapping():
        CRS_CODES = {}
        CRS_CODES['Mercator'] = Mercator
        CRS_CODES['Geographic'] = Geographic
        return CRS_CODES

    CRS_CODES = _crs_code_mapping()
    return CRS_CODES[crs]


def suggest_zonal_canvas(smallest_area, x_range, y_range,
                         crs='Mercator', min_pixels=25):
    """Given a coordinate reference system (crs), a set of polygons with
    corresponding x range and y range, calculate the height and width of canvas
    so that the smallest polygon (polygon with smallest area) is rasterized
    with at least min pixels.

    Currently, we assume that the smallest polygon does not intersect others.
    One should note that a polygon can have different shapes when it is
    rasterized in canvases of different size. Thus, we cannot 100% guarantee
    the actual number of pixels after rasterization. It is recommended to add
    an additional of 5% to @min_pixels parameter.

    Parameters
    ----------
    x_range: tuple or list of 2 numeric elements,
        The full x extent of the polygon GeoDataFrame.
    y_range: tuple or list of 2 numeric elements,
        The full y extent of the polygon GeoDataFrame.
    smallest_area: numeric (float, int)
        Area of the smallest polygon.
    crs: string,
        Name of the coordinate reference system.
    min_pixels: int
        Expected number of pixels of the polygon with smallest area
        when the whole dataframe is rasterized.

    Returns
    -------
    height, width: int, int
        height and width of the canvas in pixel space

    Examples
    --------
    >>> from spatialpandas import GeoDataFrame
    >>> import geopandas as gpd
    >>> import datashader as ds
    >>> df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    >>> df = df.to_crs("EPSG:3857")
    >>> df = df[df.continent != 'Antarctica']
    >>> df['id'] = [i for i in range(len(df.index))]
    >>> xmin, ymin, xmax, ymax = (df.bounds.minx.min(), df.bounds.miny.min(),
    ...                           df.bounds.maxx.max(), df.bounds.maxy.max())
    >>> x_range = (xmin, xmax)
    >>> y_range = (ymin, ymax)
    >>> smallest_area = df.area.min()
    >>> min_pixels = 20
    >>> height, width = suggest_zonal_canvas(x_range=x_range, y_range=y_range,
    ...                                      smallest_area=smallest_area,
    ...                                      crs='Mercator',
    ...                                      min_pixels=min_pixels)
    >>> cvs = ds.Canvas(x_range=x_range, y_range=y_range,
    ...                 plot_height=height, plot_width=width)
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
    '''
    '''
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


def regions(raster, neighborhood=4, name='regions'):
    """
    Create unique regions of raster based on pixel value connectivity.
    Connectivity can be based on either 4 or 8-pixel neighborhoods.
    Output raster contain a unique int for each connected region.

    Parameters
    ----------
    raster : xr.DataArray
    connections : int
      4 or 8 pixel-based connectivity (default: 4)
    name : str
      output xr.DataArray.name property

    Returns
    -------
    data: DataArray

    Notes
    -----

    Area Numbering implementing based on:
      http://spatial-analyst.net/ILWIS/htm/ilwisapp/areanumbering_algorithm.htm

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


def trim(raster, values=(np.nan,), name='trim'):
    """
    Trim scans from the edges and eliminates rows / cols
    which only contain the values supplied.

    Parameters
    ----------
    raster : xr.DataArray
    values : list, tuple
       list of zone ids to trim from raster edge
    name : str
      output xr.DataArray.name property

    Returns
    -------
    data: DataArray

    Notes
    -----
    This operation will change the output size of the raster
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


def crop(zones, values, zones_ids, name='crop'):
    """
    Crop scans from edges and eliminates rows / cols
    until one of the input values is found.

    Parameters
    ----------
    zones : xr.DataArray
      input zone raster

    values : xr.DataArray
      input values raster

    zones_ids : list, tuple
       list of zone ids to crop raster

    name : str
      output xr.DataArray.name property

    Returns
    -------
    data: DataArray

    Notes
    -----
    This operation will change the output size of the raster

    """
    top, bottom, left, right = _crop(zones.data, zones_ids)
    arr = values[top:bottom+1, left:right+1]
    arr.name = name
    return arr
