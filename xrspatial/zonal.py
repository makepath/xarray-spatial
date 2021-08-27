from math import sqrt
from typing import Optional, Callable, Union, Dict, List

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray

from xrspatial.utils import ngjit


def _stats_count(data):
    if isinstance(data, np.ndarray):
        # numpy case
        stats_count = np.ma.count(data)
    else:
        # dask case
        stats_count = data.size - da.ma.getmaskarray(data).sum()
    return stats_count


_DEFAULT_STATS = dict(
    mean=lambda z: z.mean(),
    max=lambda z: z.max(),
    min=lambda z: z.min(),
    sum=lambda z: z.sum(),
    std=lambda z: z.std(),
    var=lambda z: z.var(),
    count=lambda z: _stats_count(z),
)


def _to_int(numeric_value):
    # convert an integer in float type to integer type
    # if not an integer, return the value itself
    if float(numeric_value).is_integer():
        return int(numeric_value)
    return numeric_value


def _zone_cat_data(
    zones,
    values,
    zone_id,
    nodata_values,
    cat=None,
    cat_id=None
):

    # array backend
    if isinstance(zones.data, np.ndarray):
        array_module = np
    elif isinstance(zones.data, da.Array):
        array_module = da

    if len(values.shape) == 2:
        # 2D case
        conditions = (
            (zones.data != zone_id)
            | ~np.isfinite(values.data)  # mask out nan, inf
            | (values.data == nodata_values)  # mask out nodata_values
        )

        if cat is not None:
            conditions |= values.data != cat

        zone_cat_data = array_module.ma.masked_where(conditions, values.data)

    else:
        # 3D case
        cat_data = values[cat_id].data
        cat_masked_data = array_module.ma.masked_invalid(cat_data)
        zone_cat_data = array_module.ma.masked_where(
            (
                (zones.data != zone_id)
                | (cat_data == nodata_values)
            ),
            cat_masked_data
        )
    return zone_cat_data


def _stats(
    zones: xr.DataArray,
    values: xr.DataArray,
    unique_zones: List[int],
    stats_funcs: List,
    nodata_values: Union[int, float],
) -> Dict:

    stats_dict = {}
    # zone column
    stats_dict["zone"] = unique_zones
    # stats columns
    for stats in stats_funcs:
        stats_dict[stats] = []

    for zone_id in unique_zones:
        # get zone values
        zone_values = _zone_cat_data(zones, values, zone_id, nodata_values)
        for stats in stats_funcs:
            stats_func = stats_funcs.get(stats)
            if not callable(stats_func):
                raise ValueError(stats)
            stats_dict[stats].append(stats_func(zone_values))

    unique_zones = list(map(_to_int, unique_zones))
    stats_dict["zone"] = unique_zones

    return stats_dict


def _stats_numpy(
    zones: xr.DataArray,
    values: xr.DataArray,
    zone_ids: List[Union[int, float]],
    stats_funcs: Dict,
    nodata_zones: Union[int, float],
    nodata_values: Union[int, float],
) -> pd.DataFrame:

    if zone_ids is None:
        # no zone_ids provided, find ids for all zones
        # do not consider zone with nodata values
        unique_zones = np.unique(zones.data[np.isfinite(zones.data)])
        unique_zones = sorted(list(set(unique_zones) - set([nodata_zones])))
    else:
        unique_zones = np.array(zone_ids)

    stats_dict = _stats(
        zones,
        values,
        unique_zones,
        stats_funcs,
        nodata_values
    )

    stats_df = pd.DataFrame(stats_dict)
    stats_df.set_index("zone")

    return stats_df


def _stats_dask(
    zones: xr.DataArray,
    values: xr.DataArray,
    zone_ids: List[Union[int, float]],
    stats_funcs: Dict,
    nodata_zones: Union[int, float],
    nodata_values: Union[int, float],
) -> pd.DataFrame:

    if zone_ids is None:
        # no zone_ids provided, find ids for all zones
        # precompute unique zones
        unique_zones = da.unique(zones.data[da.isfinite(zones.data)]).compute()
        # do not consider zone with nodata values
        unique_zones = sorted(list(set(unique_zones) - set([nodata_zones])))
    else:
        unique_zones = np.array(zone_ids)

    stats_dict = _stats(
        zones,
        values,
        unique_zones,
        stats_funcs,
        nodata_values
    )

    stats_dict = {
        stats: da.stack(zonal_stats, axis=0)
        for stats, zonal_stats in stats_dict.items()
    }

    # generate dask dataframe
    stats_df = dd.concat(
        [dd.from_dask_array(stats) for stats in stats_dict.values()], axis=1
    )
    # name columns
    stats_df.columns = stats_dict.keys()
    stats_df.set_index("zone")

    return stats_df


def stats(
    zones: xr.DataArray,
    values: xr.DataArray,
    zone_ids: Optional[List[Union[int, float]]] = None,
    stats_funcs: Union[Dict, List] = [
        "mean",
        "max",
        "min",
        "sum",
        "std",
        "var",
        "count",
    ],
    nodata_zones: Optional[Union[int, float]] = None,
    nodata_values: Union[int, float] = None,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Calculate summary statistics for each zone defined by a zone
    dataset, based on values aggregate.

    A single output value is computed for every zone in the input zone
    dataset.

    This function currently supports numpy backed, and dask with numpy backed
    xarray DataArray.

    Parameters
    ----------
    zones : xr.DataArray
        zones is a 2D xarray DataArray of numeric values.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input `zones` raster defines
        the shape, values, and locations of the zones. An integer field
        in the input `zones` DataArray defines a zone.

    values : xr.DataArray
        values is a 2D xarray DataArray of numeric values (integers or floats).
        The input `values` raster contains the input values used in
        calculating the output statistic for each zone.

    zone_ids : list of ints, or floats
        List of zones to be included in calculation. If no zone_ids provided,
        all zones will be used.

    stats_funcs : dict, or list of strings, default=['mean', 'max', 'min',
        'sum', 'std', 'var', 'count'])
        The statistics to calculate for each zone. If a list, possible
        choices are subsets of the default options.
        In the dictionary case, all of its values must be
        callable. Function takes only one argument that is the `values` raster.
        The key become the column name in the output DataFrame.

    nodata_zones: int, float, default=None
        Nodata value in `zones` raster.
        Cells with `nodata_zones` do not belong to any zone,
        and thus excluded from calculation.

    nodata_values: int, float, default=None
        Nodata value in `values` raster.
        Cells with `nodata_values` do not belong to any zone,
        and thus excluded from calculation.

    Returns
    -------
    stats_df : Union[pandas.DataFrame, dask.dataframe.DataFrame]
        A pandas DataFrame, or a dask DataFrame where each column
        is a statistic and each row is a zone with zone id.

    Examples
    --------
    .. plot::
       :include-source:

        import numpy as np
        import xarray as xr
        from xrspatial.zonal import stats

        height, width = 10, 10
        # Values raster
        values = xr.DataArray(np.arange(height * width).reshape(height, width))
        # Zones raster
        zones = xr.DataArray(np.zeros(height * width).reshape(height, width))
        zones[:5, :5] = 0
        zones[:5, 5:] = 10
        zones[5:, :5] = 20
        zones[5:, 5:] = 30

    .. sourcecode:: python

        >>> # Calculate Stats
        >>> stats_df = stats(zones=zones, values=values)
        >>> print(stats_df)
            zone  mean  max  min   sum       std    var  count
        0   0    22.0   44    0   550  14.21267  202.0     25
        1  10    27.0   49    5   675  14.21267  202.0     25
        2  20    72.0   94   50  1800  14.21267  202.0     25
        3  30    77.0   99   55  1925  14.21267  202.0     25

        >>> # Custom Stats
        >>> custom_stats ={'double_sum': lambda val: val.sum()*2}
        >>> custom_stats_df = stats(zones=zones,
                                    values=values,
                                    stats_funcs=custom_stats)
        >>> print(custom_stats_df)
            zone  double_sum
        0   0     1100
        1  10     1350
        2  20     3600
        3  30     3850

        >>> # Calculate Stats with dask backed xarray DataArrays
        >>> dask_stats_df = stats(zones=dask_zones, values=dask_values)
        >>> print(type(dask_stats_df))
        <class 'dask.dataframe.core.DataFrame'>
        >>> print(dask_stats_df.compute())
            zone  mean  max  min   sum       std    var  count
        0     0  22.0   44    0   550  14.21267  202.0     25
        1    10  27.0   49    5   675  14.21267  202.0     25
        2    20  72.0   94   50  1800  14.21267  202.0     25
        3    30  77.0   99   55  1925  14.21267  202.0     25

        >>> # Custom Stats with dask backed xarray DataArrays
        >>> dask_custom_stats ={'double_sum': lambda val: val.sum()*2}
        >>> dask_custom_stats_df = stats(
        >>>      zones=dask_zones, values=dask_values, stats_funcs=custom_stats
        >>> )
        >>> print(type(dask_custom_stats_df))
        <class 'dask.dataframe.core.DataFrame'>
        >>> print(dask_custom_stats_df.compute())
            zone  double_sum
        0     0        1100
        1    10        1350
        2    20        3600
        3    30        3850
    """

    if zones.shape != values.shape:
        raise ValueError("`zones` and `values` must have same shape.")

    if not (
        issubclass(zones.data.dtype.type, np.integer)
        or issubclass(zones.data.dtype.type, np.floating)
    ):
        raise ValueError("`zones` must be an array of integers.")

    if not (
        issubclass(values.data.dtype.type, np.integer)
        or issubclass(values.data.dtype.type, np.floating)
    ):
        raise ValueError("`values` must be an array of integers or floats.")

    if isinstance(stats_funcs, list):
        # create a dict of stats
        stats_funcs_dict = {}

        for stats in stats_funcs:
            func = _DEFAULT_STATS.get(stats, None)
            if func is None:
                err_str = f"Invalid stat name. {stats} option not supported."
                raise ValueError(err_str)

            stats_funcs_dict[stats] = func

    elif isinstance(stats_funcs, dict):
        stats_funcs_dict = stats_funcs.copy()

    if isinstance(values.data, np.ndarray):
        # numpy case
        stats_df = _stats_numpy(
            zones,
            values,
            zone_ids,
            stats_funcs_dict,
            nodata_zones,
            nodata_values
        )
    else:
        # dask case
        stats_df = _stats_dask(
            zones,
            values,
            zone_ids,
            stats_funcs_dict,
            nodata_zones,
            nodata_values
        )

    return stats_df


def _crosstab_dict(zones, values, unique_zones, cats, nodata_values, agg):

    crosstab_dict = {}

    unique_zones = list(map(_to_int, unique_zones))
    crosstab_dict["zone"] = unique_zones

    for i in cats:
        crosstab_dict[i] = []

    for cat_id, cat in enumerate(cats):
        for zone_id in unique_zones:
            # get category cat values in the selected zone
            zone_cat_data = _zone_cat_data(
                zones, values, zone_id, nodata_values, cat, cat_id
            )
            zone_cat_count = _stats_count(zone_cat_data)
            crosstab_dict[cat].append(zone_cat_count)

    if agg == "percentage":
        zone_counts = _stats(
            zones,
            values,
            unique_zones,
            {"count": _stats_count},
            nodata_values
        )["count"]
        for c, cat in enumerate(cats):
            for z in range(len(unique_zones)):
                crosstab_dict[cat][z] = (
                    crosstab_dict[cat][z] / zone_counts[z] * 100
                )  # noqa

    return crosstab_dict


def _crosstab_numpy(
    zones,
    values,
    zone_ids,
    cat_ids,
    nodata_zones,
    nodata_values,
    agg
):

    if cat_ids is not None:
        cats = np.array(cat_ids)
    else:
        # no categories provided, find all possible cats in values raster
        if len(values.shape) == 3:
            # 3D case
            cats = values.indexes[values.dims[0]].values
        else:
            # 2D case
            # mask out all invalid values such as: nan, inf
            cats = da.unique(values.data[da.isfinite(values.data)]).compute()
            cats = sorted(list(set(cats) - set([nodata_values])))

    if zone_ids is None:
        # do not consider zone with nodata values
        unique_zones = np.unique(zones.data[np.isfinite(zones.data)])
        unique_zones = sorted(list(set(unique_zones) - set([nodata_zones])))
    else:
        unique_zones = np.array(zone_ids)

    crosstab_dict = _crosstab_dict(
        zones, values, unique_zones, cats, nodata_values, agg
    )

    crosstab_df = pd.DataFrame(crosstab_dict)

    # name columns
    crosstab_df.columns = crosstab_dict.keys()

    return crosstab_df


def _crosstab_dask(
    zones,
    values,
    zone_ids,
    cat_ids,
    nodata_zones,
    nodata_values,
    agg
):

    if cat_ids is not None:
        cats = np.array(cat_ids)
    else:
        # no categories provided, find all possible cats in values raster
        if len(values.shape) == 3:
            # 3D case
            cats = values.indexes[values.dims[0]].values
        else:
            # 2D case
            # precompute categories
            cats = da.unique(values.data[da.isfinite(values.data)]).compute()
            cats = sorted(list(set(cats) - set([nodata_values])))

    if zone_ids is None:
        # precompute unique zones
        unique_zones = da.unique(zones.data[da.isfinite(zones.data)]).compute()
        # do not consider zone with nodata values
        unique_zones = sorted(list(set(unique_zones) - set([nodata_zones])))
    else:
        unique_zones = np.array(zone_ids)

    crosstab_dict = _crosstab_dict(
        zones, values, unique_zones, cats, nodata_values, agg
    )
    crosstab_dict = {
        stats: da.stack(zonal_stats, axis=0)
        for stats, zonal_stats in crosstab_dict.items()
    }

    # generate dask dataframe
    crosstab_df = dd.concat(
        [dd.from_dask_array(stats) for stats in crosstab_dict.values()], axis=1
    )

    # name columns
    crosstab_df.columns = crosstab_dict.keys()

    return crosstab_df


def crosstab(
    zones: xr.DataArray,
    values: xr.DataArray,
    zone_ids: List[Union[int, float]] = None,
    cat_ids: List[Union[int, float]] = None,
    layer: Optional[int] = None,
    agg: Optional[str] = "count",
    nodata_zones: Optional[Union[int, float]] = None,
    nodata_values: Optional[Union[int, float]] = None,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Calculate cross-tabulated (categorical stats) areas
    between two datasets: a zone dataset `zones`, a value dataset `values`
    (a value raster). Infinite and NaN values in `zones` and `values` will
    be ignored.

    Outputs a pandas DataFrame if `zones` and `values` are numpy backed.
    Outputs a dask DataFrame if `zones` and `values` are dask with
    numpy-backed xarray DataArrays.

    Requires a DataArray with a single data dimension, here called the
    "values", indexed using either 2D or 3D coordinates.

    DataArrays with 3D coordinates are expected to contain values
    distributed over different categories that are indexed by the
    additional coordinate.  Such an array would reduce to the
    2D-coordinate case if collapsed across the categories (e.g. if one
    did ``aggc.sum(dim='cat')`` for a categorical dimension ``cat``).

    Parameters
    ----------
    zones : xr.DataArray
        2D data array of integers or floats.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input `zones` raster defines
        the shape, values, and locations of the zones. An unique field
        in the zone input is specified to define the zones.

    values : xr.DataArray
        2D or 3D data array of integers or floats.
        The input value raster contains the input values used in
        calculating the categorical statistic for each zone.

    zone_ids: List of ints, or floats
        List of zones to be included in calculation. If no zone_ids provided,
        all zones will be used.

    cat_ids: List of ints, or floats
        List of categories to be included in calculation.
        If no cat_ids provided, all categories will be used.

    layer: int, default=0
        index of the categorical dimension layer inside the `values` DataArray.

    agg: str, default = 'count'
        Aggregation method.
        If the data is 2D, available options are: percentage, count.
        If the data is 3D, available option is: count.

    nodata_zones: int, float, default=None
        Nodata value in `zones` raster.
        Cells with `nodata` do not belong to any zone,
        and thus excluded from calculation.

    nodata_values: int, float, default=None
        Nodata value in `values` raster.
        Cells with `nodata` do not belong to any zone,
        and thus excluded from calculation.

    Returns
    -------
    crosstab_df : Union[pandas.DataFrame, dask.dataframe.DataFrame]
        A pandas DataFrame, or an uncomputed dask DataFrame,
        where each column is a categorical value and each row is a zone
        with zone id. Each entry presents the statistics, which computed
        using the specified aggregation method, of the category over the zone.

    Examples
    --------
    .. plot::
       :include-source:

        import dask.array as da
        import numpy as np
        import xarray as xr
        from xrspatial.zonal import crosstab

        values_data = np.asarray([[0, 0, 10, 20],
                                  [0, 0, 0, 10],
                                  [0, np.nan, 20, 50],
                                  [10, 30, 40, np.inf],
                                  [10, 10, 50, 0]])
        values = xr.DataArray(values_data)

        zones_data = np.asarray([[1, 1, 6, 6],
                                 [1, np.nan, 6, 6],
                                 [3, 5, 6, 6],
                                 [3, 5, 7, np.nan],
                                 [3, 7, 7, 0]])
        zones = xr.DataArray(zones_data)

        values_dask = xr.DataArray(da.from_array(values, chunks=(3, 3)))
        zones_dask = xr.DataArray(da.from_array(zones, chunks=(3, 3)))

    .. sourcecode:: python

        >>> # Calculate Crosstab, numpy case
        >>> df = crosstab(zones=zones, values=values)
        >>> print(df)
                zone  0.0  10.0  20.0  30.0  40.0  50.0
            0      0    1     0     0     0     0     0
            1      1    3     0     0     0     0     0
            2      3    1     2     0     0     0     0
            3      5    0     0     0     1     0     0
            4      6    1     2     2     0     0     1
            5      7    0     1     0     0     1     1

        >>> # Calculate Crosstab, dask case
        >>> df = crosstab(zones=zones_dask, values=values_dask)
        >>> print(df)
            Dask DataFrame Structure:
            zone	0.0	10.0	20.0	30.0	40.0	50.0
            npartitions=5
            0	float64	int64	int64	int64	int64	int64	int64
            1	...	...	...	...	...	...	...
            ...	...	...	...	...	...	...	...
            4	...	...	...	...	...	...	...
            5	...	...	...	...	...	...	...
            Dask Name: astype, 1186 tasks
        >>> print(dask_df.compute)
                zone  0.0  10.0  20.0  30.0  40.0  50.0
            0      0    1     0     0     0     0     0
            1      1    3     0     0     0     0     0
            2      3    1     2     0     0     0     0
            3      5    0     0     0     1     0     0
            4      6    1     2     2     0     0     1
            5      7    0     1     0     0     1     1
    """
    if not isinstance(zones, xr.DataArray):
        raise TypeError("zones must be instance of DataArray")

    if not isinstance(values, xr.DataArray):
        raise TypeError("values must be instance of DataArray")

    if zones.ndim != 2:
        raise ValueError("zones must be 2D")

    if not (
        issubclass(zones.data.dtype.type, np.integer)
        or issubclass(zones.data.dtype.type, np.floating)
    ):
        raise ValueError("`zones` must be an xarray of integers or floats")

    if not issubclass(values.data.dtype.type, np.integer) and not issubclass(
        values.data.dtype.type, np.floating
    ):
        raise ValueError("`values` must be an xarray of integers or floats")

    if values.ndim not in [2, 3]:
        raise ValueError("`values` must use either 2D or 3D coordinates.")

    agg_2d = ["percentage", "count"]
    agg_3d = ["count"]

    if values.ndim == 2 and agg not in agg_2d:
        raise ValueError(
            f"`agg` method for 2D data array must be one of following {agg_2d}"
        )

    if values.ndim == 3 and agg not in agg_3d:
        raise ValueError(
            f"`agg` method for 3D data array must be one of following {agg_3d}"
        )

    if len(values.shape) == 3:
        # 3D case
        if layer is None:
            layer = 0
        try:
            values.indexes[values.dims[layer]].values
        except (IndexError, KeyError):
            raise ValueError("Invalid `layer`")

        dims = values.dims
        reshape_dims = [dims[layer]] + [d for d in dims if d != dims[layer]]
        # transpose by that category dimension
        values = values.transpose(*reshape_dims)

        if zones.shape != values.shape[1:]:
            raise ValueError("Incompatible shapes")

    if isinstance(values.data, np.ndarray):
        # numpy case
        crosstab_df = _crosstab_numpy(
            zones, values, zone_ids, cat_ids, nodata_zones, nodata_values, agg
        )
    else:
        # dask case
        crosstab_df = _crosstab_dask(
            zones, values, zone_ids, cat_ids, nodata_zones, nodata_values, agg
        )

    return crosstab_df


def apply(
    zones: xr.DataArray,
    values: xr.DataArray,
    func: Callable,
    nodata: Optional[int] = 0
):
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

    nodata: int, default=None
        Nodata value in `zones` raster.
        Cells with `nodata` does not belong to any zone,
        and thus excluded from calculation.

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
        raise ValueError("Incompatible shapes between `zones` and `values`")

    if not issubclass(zones.values.dtype.type, np.integer):
        raise ValueError("`zones.values` must be an array of integers")

    if not (
        issubclass(values.values.dtype.type, np.integer)
        or issubclass(values.values.dtype.type, np.floating)
    ):
        raise ValueError("`values` must be an array of integers or float")

    # entries of nodata remain the same
    remain_entries = zones.data == nodata

    # entries with to be included in calculation
    zones_entries = zones.data != nodata

    if len(values.shape) == 3:
        z = values.shape[-1]
        # add new z-dimension in case 3D `values` aggregate
        remain_entries = np.repeat(
            remain_entries[:, :, np.newaxis],
            z,
            axis=-1
        )
        zones_entries = np.repeat(
            zones_entries[:, :, np.newaxis],
            z,
            axis=-1
        )

    remain_mask = np.ma.masked_array(values.data, mask=remain_entries)
    zones_mask = np.ma.masked_array(values.data, mask=zones_entries)

    # apply func to corresponding `values` of `zones`
    vfunc = np.vectorize(func)
    values_func = vfunc(zones_mask)
    values.values = (
        remain_mask.data
        * remain_mask.mask
        + values_func.data
        * values_func.mask
    )


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
        CRS_CODES["Mercator"] = Mercator
        CRS_CODES["Geographic"] = Geographic
        return CRS_CODES

    CRS_CODES = _crs_code_mapping()
    return CRS_CODES[crs]


def suggest_zonal_canvas(
    smallest_area: Union[int, float],
    x_range: Union[tuple, list],
    y_range: Union[tuple, list],
    crs: str = "Mercator",
    min_pixels: int = 25,
) -> tuple:
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
                src_window[0] = data[max(y - 1, 0), max(x - 1, 0)]
                src_window[1] = data[y, max(x - 1, 0)]
                src_window[2] = data[min(y + 1, rows - 1), max(x - 1, 0)]
                src_window[3] = data[max(y - 1, 0), x]
                src_window[4] = data[min(y + 1, rows - 1), x]
                src_window[5] = data[max(y - 1, 0), min(x + 1, cols - 1)]
                src_window[6] = data[y, min(x + 1, cols - 1)]
                src_window[7] = data[min(y + 1, rows - 1), min(x + 1, cols - 1)] # noqa

                area_window[0] = out[max(y - 1, 0), max(x - 1, 0)]
                area_window[1] = out[y, max(x - 1, 0)]
                area_window[2] = out[min(y + 1, rows - 1), max(x - 1, 0)]
                area_window[3] = out[max(y - 1, 0), x]
                area_window[4] = out[min(y + 1, rows - 1), x]
                area_window[5] = out[max(y - 1, 0), min(x + 1, cols - 1)]
                area_window[6] = out[y, min(x + 1, cols - 1)]
                area_window[7] = out[min(y + 1, rows - 1), min(x + 1, cols - 1)] # noqa

            else:
                src_window[0] = data[y, max(x - 1, 0)]
                src_window[1] = data[max(y - 1, 0), x]
                src_window[2] = data[min(y + 1, rows - 1), x]
                src_window[3] = data[y, min(x + 1, cols - 1)]

                area_window[0] = out[y, max(x - 1, 0)]
                area_window[1] = out[max(y - 1, 0), x]
                area_window[2] = out[min(y + 1, rows - 1), x]
                area_window[3] = out[y, min(x + 1, cols - 1)]

            # check in has matching value in neighborhood
            rtol = 1e-05
            atol = 1e-08
            is_close = np.abs(src_window - val) <= (atol + rtol * np.abs(val))
            neighbor_matches = np.where(is_close)[0]

            if len(neighbor_matches) > 0:

                # check in has area already assigned
                assigned_value = None
                for j in range(len(neighbor_matches)):
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
                src_window[0] = data[max(y - 1, 0), max(x - 1, 0)]
                src_window[1] = data[y, max(x - 1, 0)]
                src_window[2] = data[min(y + 1, rows - 1), max(x - 1, 0)]
                src_window[3] = data[max(y - 1, 0), x]
                src_window[4] = data[min(y + 1, rows - 1), x]
                src_window[5] = data[max(y - 1, 0), min(x + 1, cols - 1)]
                src_window[6] = data[y, min(x + 1, cols - 1)]
                src_window[7] = data[min(y + 1, rows - 1), min(x + 1, cols - 1)] # noqa

                area_window[0] = out[max(y - 1, 0), max(x - 1, 0)]
                area_window[1] = out[y, max(x - 1, 0)]
                area_window[2] = out[min(y + 1, rows - 1), max(x - 1, 0)]
                area_window[3] = out[max(y - 1, 0), x]
                area_window[4] = out[min(y + 1, rows - 1), x]
                area_window[5] = out[max(y - 1, 0), min(x + 1, cols - 1)]
                area_window[6] = out[y, min(x + 1, cols - 1)]
                area_window[7] = out[min(y + 1, rows - 1), min(x + 1, cols - 1)] # noqa

            else:
                src_window[0] = data[y, max(x - 1, 0)]
                src_window[1] = data[max(y - 1, 0), x]
                src_window[2] = data[min(y + 1, rows - 1), x]
                src_window[3] = data[y, min(x + 1, cols - 1)]

                area_window[0] = out[y, max(x - 1, 0)]
                area_window[1] = out[max(y - 1, 0), x]
                area_window[2] = out[min(y + 1, rows - 1), x]
                area_window[3] = out[y, min(x + 1, cols - 1)]

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


def regions(
    raster: xr.DataArray, neighborhood: int = 4, name: str = "regions"
) -> xr.DataArray:
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
        raise ValueError("`neighborhood` value must be either 4 or 8)")

    out = _area_connectivity(raster.data, n=neighborhood)

    return DataArray(
        out,
        name=name,
        dims=raster.dims,
        coords=raster.coords,
        attrs=raster.attrs
    )


def _bool_crop(arr, rows_flags, cols_flags):
    top = np.argwhere(rows_flags).flatten()[0]
    bottom = np.argwhere(rows_flags).flatten()[-1]
    left = np.argwhere(cols_flags).flatten()[0]
    right = np.argwhere(cols_flags).flatten()[-1]
    return arr[top: bottom + 1, left: right + 1]


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
    for y in range(rows - 1, -1, -1):
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
    for x in range(cols - 1, -1, -1):
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


def trim(
    raster: xr.DataArray,
    values: Union[list, tuple] = (np.nan,),
    name: str = "trim"
) -> xr.DataArray:
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
    arr = raster[top: bottom + 1, left: right + 1]
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
    for y in range(rows - 1, -1, -1):

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
    for x in range(cols - 1, -1, -1):
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


def crop(
    zones: xr.DataArray,
    values: xr.DataArray,
    zones_ids: Union[list, tuple],
    name: str = "crop",
):
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
    arr = values[top: bottom + 1, left: right + 1]
    arr.name = name
    return arr
