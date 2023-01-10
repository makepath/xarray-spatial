# standard library
import copy
from math import sqrt
from typing import Callable, Dict, List, Optional, Union

# 3rd-party
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed
from xarray import DataArray

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

# local modules
from xrspatial.utils import ArrayTypeFunctionMapping, ngjit, not_implemented_func, validate_arrays

TOTAL_COUNT = '_total_count'


def _stats_count(data):
    if isinstance(data, np.ndarray):
        # numpy case
        stats_count = np.ma.count(data)
    elif isinstance(data, cupy.ndarray):
        # cupy case
        stats_count = np.prod(data.shape)
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


_DASK_BLOCK_STATS = dict(
    max=lambda z: z.max(),
    min=lambda z: z.min(),
    sum=lambda z: z.sum(),
    count=lambda z: _stats_count(z),
    sum_squares=lambda z: (z**2).sum()
)


_DASK_STATS = dict(
    max=lambda block_maxes: np.nanmax(block_maxes, axis=0),
    min=lambda block_mins: np.nanmin(block_mins, axis=0),
    sum=lambda block_sums: np.nansum(block_sums, axis=0),
    count=lambda block_counts: np.nansum(block_counts, axis=0),
    sum_squares=lambda block_sum_squares: np.nansum(block_sum_squares, axis=0),
    squared_sum=lambda block_sums: np.nansum(block_sums, axis=0)**2,
)
def _dask_mean(sums, counts): return sums / counts  # noqa
def _dask_std(sum_squares, squared_sum, n): return np.sqrt((sum_squares - squared_sum/n) / n)  # noqa
def _dask_var(sum_squares, squared_sum, n): return (sum_squares - squared_sum/n) / n  # noqa


@ngjit
def _strides(flatten_zones, unique_zones):
    num_elements = flatten_zones.shape[0]
    num_zones = len(unique_zones)
    strides = np.zeros(len(unique_zones), dtype=np.int32)

    count = 0
    for i in range(num_zones):
        while (count < num_elements) and (
                flatten_zones[count] == unique_zones[i]):
            count += 1
        strides[i] = count

    return strides


def _sort_and_stride(zones, values, unique_zones):
    flatten_zones = zones.ravel()
    sorted_indices = np.argsort(flatten_zones)
    sorted_zones = flatten_zones[sorted_indices]

    values_shape = values.shape
    if len(values_shape) == 3:
        values_by_zones = copy.deepcopy(values).reshape(
            values_shape[0], values_shape[1] * values_shape[2])
        for i in range(values_shape[0]):
            values_by_zones[i] = values_by_zones[i][sorted_indices]
    else:
        values_by_zones = values.ravel()[sorted_indices]

    # exclude nans from calculation
    # flatten_zones is already sorted, NaN elements (if any) are at the end
    # of the array, removing them will not affect data before them
    sorted_zones = sorted_zones[np.isfinite(sorted_zones)]
    zone_breaks = _strides(sorted_zones, unique_zones)

    return sorted_indices, values_by_zones, zone_breaks


def _calc_stats(
    values_by_zones: np.array,
    zone_breaks: np.array,
    unique_zones: np.array,
    zone_ids: np.array,
    func: Callable,
    nodata_values: Union[int, float] = None,
):
    start = 0
    results = np.full(unique_zones.shape, np.nan)
    for i in range(len(unique_zones)):
        end = zone_breaks[i]
        if unique_zones[i] in zone_ids:
            zone_values = values_by_zones[start:end]
            # filter out non-finite and nodata_values
            zone_values = zone_values[np.isfinite(zone_values) & (zone_values != nodata_values)]
            if len(zone_values) > 0:
                results[i] = func(zone_values)
        start = end
    return results


@delayed
def _single_stats_func(
    zones_block: np.array,
    values_block: np.array,
    unique_zones: np.array,
    zone_ids: np.array,
    func: Callable,
    nodata_values: Union[int, float] = None,
) -> pd.DataFrame:

    _, values_by_zones, zone_breaks = _sort_and_stride(zones_block, values_block, unique_zones)
    results = _calc_stats(values_by_zones, zone_breaks, unique_zones, zone_ids, func, nodata_values)
    return results


def _stats_dask_numpy(
    zones: da.Array,
    values: da.Array,
    zone_ids: List[Union[int, float]],
    stats_funcs: Dict,
    nodata_values: Union[int, float],
) -> pd.DataFrame:

    # find ids for all zones
    unique_zones = np.unique(zones[np.isfinite(zones)])

    select_all_zones = False
    # selecte zones to do analysis
    if zone_ids is None:
        zone_ids = unique_zones
        select_all_zones = True

    zones_blocks = zones.to_delayed().ravel()
    values_blocks = values.to_delayed().ravel()

    stats_dict = {}
    stats_dict["zone"] = unique_zones  # zone column

    compute_sum_squares = False
    compute_sum = False
    compute_count = False

    if 'mean' or 'std' or 'var' in stats_funcs:
        compute_sum = True
        compute_count = True

    if 'std' or 'var' in stats_funcs:
        compute_sum_squares = True

    basis_stats = [s for s in _DASK_BLOCK_STATS if s in stats_funcs]
    if compute_count and 'count' not in basis_stats:
        basis_stats.append('count')
    if compute_sum and 'sum' not in basis_stats:
        basis_stats.append('sum')
    if compute_sum_squares:
        basis_stats.append('sum_squares')

    dask_dtypes = dict(
        max=values.dtype,
        min=values.dtype,
        sum=values.dtype,
        count=np.int64,
        sum_squares=values.dtype,
        squared_sum=values.dtype,
    )

    for s in basis_stats:
        if s == 'sum_squares' and not compute_sum_squares:
            continue
        stats_func = _DASK_BLOCK_STATS.get(s)
        stats_by_block = [
            da.from_delayed(
                delayed(_single_stats_func)(
                    z, v, unique_zones, zone_ids, stats_func, nodata_values
                ), shape=(np.nan,), dtype=dask_dtypes[s]
            )
            for z, v in zip(zones_blocks, values_blocks)
        ]
        zonal_stats = da.stack(stats_by_block, allow_unknown_chunksizes=True)
        stats_func_by_block = delayed(_DASK_STATS[s])
        stats_dict[s] = da.from_delayed(
            stats_func_by_block(zonal_stats), shape=(np.nan,), dtype=np.float64
        )

    if 'mean' in stats_funcs:
        stats_dict['mean'] = _dask_mean(stats_dict['sum'], stats_dict['count'])
    if 'std' in stats_funcs:
        stats_dict['std'] = _dask_std(
            stats_dict['sum_squares'], stats_dict['sum'] ** 2, stats_dict['count']
        )
    if 'var' in stats_funcs:
        stats_dict['var'] = _dask_var(
            stats_dict['sum_squares'], stats_dict['sum'] ** 2, stats_dict['count']
        )

    # generate dask dataframe
    stats_df = dd.concat([dd.from_dask_array(s) for s in stats_dict.values()], axis=1)
    # name columns
    stats_df.columns = stats_dict.keys()
    # select columns
    stats_df = stats_df[['zone'] + list(stats_funcs.keys())]

    if not select_all_zones:
        # only return zones specified in `zone_ids`
        selected_rows = []
        for index, row in stats_df.iterrows():
            if row['zone'] in zone_ids:
                selected_rows.append(stats_df.loc[index])
        stats_df = dd.concat(selected_rows)

    return stats_df


def _stats_numpy(
    zones: xr.DataArray,
    values: xr.DataArray,
    zone_ids: List[Union[int, float]],
    stats_funcs: Dict,
    nodata_values: Union[int, float],
    return_type: str,
) -> Union[pd.DataFrame, np.ndarray]:

    # find ids for all zones
    unique_zones = np.unique(zones[np.isfinite(zones)])
    # selected zones to do analysis
    if zone_ids is None:
        zone_ids = unique_zones
    else:
        zone_ids = np.unique(zone_ids)
        # remove zones that do not exist in `zones` raster
        zone_ids = [z for z in zone_ids if z in unique_zones]

    sorted_indices, values_by_zones, zone_breaks = _sort_and_stride(zones, values, unique_zones)
    if return_type == 'pandas.DataFrame':
        stats_dict = {}
        stats_dict["zone"] = zone_ids
        selected_indexes = [i for i, z in enumerate(unique_zones) if z in zone_ids]
        for stats in stats_funcs:
            func = stats_funcs.get(stats)
            stats_dict[stats] = _calc_stats(
                values_by_zones, zone_breaks,
                unique_zones, zone_ids, func, nodata_values
            )
            stats_dict[stats] = stats_dict[stats][selected_indexes]
        result = pd.DataFrame(stats_dict)

    else:
        result = np.full((len(stats_funcs), values.size), np.nan)
        zone_ids_map = {z: i for i, z in enumerate(unique_zones) if z in zone_ids}
        stats_id = 0
        for stats in stats_funcs:
            func = stats_funcs.get(stats)
            stats_results = _calc_stats(
                values_by_zones, zone_breaks,
                unique_zones, zone_ids, func, nodata_values
            )
            for zone in zone_ids:
                iz = zone_ids_map[zone]  # position of zone in unique_zones
                if iz == 0:
                    zs = sorted_indices[: zone_breaks[iz]]
                else:
                    zs = sorted_indices[zone_breaks[iz-1]: zone_breaks[iz]]
                result[stats_id][zs] = stats_results[iz]
            stats_id += 1
        result = result.reshape(len(stats_funcs), *values.shape)
    return result


def _stats_cupy(
    orig_zones: xr.DataArray,
    orig_values: xr.DataArray,
    zone_ids: List[Union[int, float]],
    stats_funcs: Dict,
    nodata_values: Union[int, float],
) -> pd.DataFrame:

    # TODO add support for 3D input
    if len(orig_values.shape) > 2:
        raise TypeError('3D inputs not supported for cupy backend')

    zones = cupy.ravel(orig_zones)
    values = cupy.ravel(orig_values)

    sorted_indices = cupy.argsort(zones)

    sorted_zones = zones[sorted_indices]
    values_by_zone = values[sorted_indices]

    # filter out values that are non-finite or values equal to nodata_values
    if nodata_values:
        filter_values = cupy.isfinite(values_by_zone) & (
            values_by_zone != nodata_values)
    else:
        filter_values = cupy.isfinite(values_by_zone)
    values_by_zone = values_by_zone[filter_values]
    sorted_zones = sorted_zones[filter_values]

    # Now I need to find the unique zones, and zone breaks
    unique_zones, unique_index, unique_counts = cupy.unique(
        sorted_zones, return_index=True, return_counts=True)

    # Transfer to the host
    unique_index = unique_index.get()
    unique_counts = unique_counts.get()
    unique_zones = unique_zones.get()

    if zone_ids is not None:
        # We need to extract the index and element count
        # only for the elements in zone_ids
        unique_index_lst = []
        unique_counts_lst = []
        unique_zones = list(unique_zones)
        for z in zone_ids:
            try:
                idx = unique_zones.index(z)
                unique_index_lst.append(unique_index[idx])
                unique_counts_lst.append(unique_counts[idx])
            except ValueError:
                continue
        unique_zones = zone_ids
        unique_counts = unique_counts_lst
        unique_index = unique_index_lst

    # stats columns
    stats_dict = {'zone': []}
    for stats in stats_funcs:
        stats_dict[stats] = []

    for i in range(len(unique_zones)):
        zone_id = unique_zones[i]
        # skip zone_id == nodata_zones, and non-finite zone ids
        if not np.isfinite(zone_id):
            continue

        stats_dict['zone'].append(zone_id)

        # extract zone_values
        zone_values = values_by_zone[unique_index[i]:unique_index[i]+unique_counts[i]]

        # apply stats on the zone data
        for j, stats in enumerate(stats_funcs):
            stats_func = stats_funcs.get(stats)
            if not callable(stats_func):
                raise ValueError(stats)
            result = stats_func(zone_values)

            assert(len(result.shape) == 0)

            stats_dict[stats].append(cupy.float_(result))

    stats_df = pd.DataFrame(stats_dict)
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
    nodata_values: Union[int, float] = None,
    return_type: str = 'pandas.DataFrame',
) -> Union[pd.DataFrame, dd.DataFrame, xr.DataArray]:
    """
    Calculate summary statistics for each zone defined by a `zones`
    dataset, based on `values` aggregate.

    A single output value is computed for every zone in the input `zones`
    dataset.

    This function currently supports numpy backed, and dask with numpy backed
    xarray DataArrays.

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
        calculating the output statistic for each zone. In dask case,
        the chunksizes of `zones` and `values` should be matching. If not,
        `values` will be rechunked to be the same as of `zones`.

    zone_ids : list of ints, or floats
        List of zones to be included in calculation. If no zone_ids provided,
        all zones will be used.

    stats_funcs : dict, or list of strings, default=['mean', 'max', 'min',
        'sum', 'std', 'var', 'count']
        The statistics to calculate for each zone. If a list, possible
        choices are subsets of the default options.
        In the dictionary case, all of its values must be
        callable. Function takes only one argument that is the `values` raster.
        The key become the column name in the output DataFrame.
        Note that if `zones` and `values` are dask backed DataArrays,
        `stats_funcs` must be provided as a list that is a subset of
        default supported stats.

    nodata_values: int, float, default=None
        Nodata value in `values` raster.
        Cells with `nodata_values` do not belong to any zone,
        and thus excluded from calculation.

    return_type: str, default='pandas.DataFrame'
        Format of returned data. If `zones` and `values` numpy backed xarray DataArray,
        allowed values are 'pandas.DataFrame', and 'xarray.DataArray'.
        Otherwise, only 'pandas.DataFrame' is supported.

    Returns
    -------
    stats_df : Union[pandas.DataFrame, dask.dataframe.DataFrame]
        A pandas DataFrame, or a dask DataFrame where each column
        is a statistic and each row is a zone with zone id.

    Examples
    --------
    stats() works with NumPy backed DataArray

    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.zonal import stats
        >>> height, width = 10, 10
        >>> values_data = np.array([
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
        >>> values = xr.DataArray(values_data)
        >>> zones_data = np.array([
            [ 0.,  0.,  0.,  0.,  0., 10., 10., 10., 10., 10.],
            [ 0.,  0.,  0.,  0.,  0., 10., 10., 10., 10., 10.],
            [ 0.,  0.,  0.,  0.,  0., 10., 10., 10., 10., 10.],
            [ 0.,  0.,  0.,  0.,  0., 10., 10., 10., 10., 10.],
            [ 0.,  0.,  0.,  0.,  0., 10., 10., 10., 10., 10.],
            [20., 20., 20., 20., 20., 30., 30., 30., 30., 30.],
            [20., 20., 20., 20., 20., 30., 30., 30., 30., 30.],
            [20., 20., 20., 20., 20., 30., 30., 30., 30., 30.],
            [20., 20., 20., 20., 20., 30., 30., 30., 30., 30.],
            [20., 20., 20., 20., 20., 30., 30., 30., 30., 30.]])
        >>> zones = xr.DataArray(zones_data)

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

    stats() works with Dask with NumPy backed DataArray
        >>> import dask.array as da
        >>> import dask.array as da
        >>> values_dask = xr.DataArray(da.from_array(values_data, chunks=(3, 3)))
        >>> zones_dask = xr.DataArray(da.from_array(zones_data, chunks=(3, 3)))
        >>> # Calculate Stats with dask backed xarray DataArrays
        >>> dask_stats_df = stats(zones=zones_dask, values=values_dask)
        >>> print(type(dask_stats_df))
        <class 'dask.dataframe.core.DataFrame'>
        >>> print(dask_stats_df.compute())
            zone  mean  max  min   sum       std    var  count
        0     0  22.0   44    0   550  14.21267  202.0     25
        1    10  27.0   49    5   675  14.21267  202.0     25
        2    20  72.0   94   50  1800  14.21267  202.0     25
        3    30  77.0   99   55  1925  14.21267  202.0     25
    """

    validate_arrays(zones, values)

    if not (
        issubclass(zones.data.dtype.type, np.integer)
        or issubclass(zones.data.dtype.type, np.floating)
    ):
        raise ValueError("`zones` must be an array of integers or floats.")

    if not (
        issubclass(values.data.dtype.type, np.integer)
        or issubclass(values.data.dtype.type, np.floating)
    ):
        raise ValueError("`values` must be an array of integers or floats.")

    # validate stats_funcs
    if isinstance(values.data, da.Array) and not isinstance(stats_funcs, list):
        raise ValueError(
            "Got dask-backed DataArray as `values` aggregate. "
            "`stats_funcs` must be a subset of default supported stats "
            "`[\'mean\', \'max\', \'min\', \'sum\', \'std\', \'var\', \'count\']`"
        )

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

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _stats_numpy(*args, return_type=return_type),
        dask_func=_stats_dask_numpy,
        cupy_func=_stats_cupy,
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='stats() does not support dask with cupy backed DataArray'  # noqa
        ),
    )
    result = mapper(values)(
        zones.data, values.data, zone_ids, stats_funcs_dict, nodata_values,
    )

    if return_type == 'xarray.DataArray':
        return xr.DataArray(
            result,
            coords={'stats': list(stats_funcs_dict.keys()), **values.coords},
            dims=('stats', *values.dims),
            attrs=values.attrs
        )
    return result


def _find_cats(values, cat_ids, nodata_values):
    if len(values.shape) == 2:
        # 2D case
        unique_cats = np.unique(values.data[
            np.isfinite(values.data) & (values.data != nodata_values)
        ])
    else:
        # 3D case
        unique_cats = values[values.dims[0]].data

    if cat_ids is None:
        cat_ids = unique_cats
    else:
        if isinstance(values.data, np.ndarray):
            # remove cats that do not exist in `values` raster
            cat_ids = [c for c in cat_ids if c in unique_cats]
        else:
            cat_ids = _select_ids(unique_cats, cat_ids)

    return unique_cats, cat_ids


def _get_zone_values(values_by_zones, start, end):
    if len(values_by_zones.shape) == 1:
        # 1D flatten, i.e, original data is 2D
        return values_by_zones[start:end]
    else:
        # 2D flatten, i.e, original data is 3D
        return values_by_zones[:, start:end]


def _single_zone_crosstab_2d(
    zone_values,
    unique_cats,
    cat_ids,
    nodata_values,
    crosstab_dict,
):
    # 1D flatten zone_values, i.e, original data is 2D
    # filter out non-finite and nodata_values
    zone_values = zone_values[
        np.isfinite(zone_values) & (zone_values != nodata_values)
    ]
    total_count = zone_values.shape[0]
    crosstab_dict[TOTAL_COUNT].append(total_count)

    sorted_zone_values = np.sort(zone_values)
    zone_cat_breaks = _strides(sorted_zone_values, unique_cats)

    cat_start = 0

    for j, cat in enumerate(unique_cats):
        if cat in cat_ids:
            count = zone_cat_breaks[j] - cat_start
            crosstab_dict[cat].append(count)
            cat_start = zone_cat_breaks[j]


def _single_zone_crosstab_3d(
    zone_values,
    unique_cats,
    cat_ids,
    nodata_values,
    crosstab_dict,
    stats_func
):
    # 2D flatten `zone_values`, i.e, original data is 3D
    for j, cat in enumerate(unique_cats):
        if cat in cat_ids:
            zone_cat_data = zone_values[j]
            # filter out non-finite and nodata_values
            zone_cat_data = zone_cat_data[
                np.isfinite(zone_cat_data)
                & (zone_cat_data != nodata_values)
            ]
            crosstab_dict[cat].append(stats_func(zone_cat_data))


def _crosstab_numpy(
    zones: np.ndarray,
    values: np.ndarray,
    zone_ids: List[Union[int, float]],
    unique_cats: np.ndarray,
    cat_ids: Union[List, np.ndarray],
    nodata_values: Union[int, float],
    agg: str,
) -> pd.DataFrame:

    # find ids for all zones
    unique_zones = np.unique(zones[np.isfinite(zones)])
    # selected zones to do analysis
    if zone_ids is None:
        zone_ids = unique_zones
    else:
        # remove zones that do not exist in `zones` raster
        zone_ids = [z for z in zone_ids if z in unique_zones]

    crosstab_dict = {}
    crosstab_dict["zone"] = zone_ids
    if len(values.shape) == 2:
        crosstab_dict[TOTAL_COUNT] = []
    for cat in cat_ids:
        crosstab_dict[cat] = []

    _, values_by_zones, zone_breaks = _sort_and_stride(
        zones, values, unique_zones
    )

    start = 0
    for i in range(len(unique_zones)):
        end = zone_breaks[i]
        if unique_zones[i] in zone_ids:
            # get data for zone unique_zones[i]
            zone_values = _get_zone_values(values_by_zones, start, end)
            if len(values.shape) == 2:
                _single_zone_crosstab_2d(
                    zone_values, unique_cats, cat_ids, nodata_values, crosstab_dict
                )
            else:
                _single_zone_crosstab_3d(
                    zone_values, unique_cats, cat_ids, nodata_values, crosstab_dict, _DEFAULT_STATS[agg]  # noqa
                )
        start = end

    if TOTAL_COUNT in crosstab_dict:
        crosstab_dict[TOTAL_COUNT] = np.array(
            crosstab_dict[TOTAL_COUNT], dtype=np.float32
        )
    for cat in cat_ids:
        crosstab_dict[cat] = np.array(crosstab_dict[cat])

    # construct output dataframe
    if agg == 'percentage':
        # replace 0s with nans to avoid dividing by 0 error
        crosstab_dict[TOTAL_COUNT][crosstab_dict[TOTAL_COUNT] == 0] = np.nan
        for cat in cat_ids:
            crosstab_dict[cat] = crosstab_dict[cat] / crosstab_dict[TOTAL_COUNT] * 100  # noqa

    crosstab_df = pd.DataFrame(crosstab_dict)
    crosstab_df = crosstab_df[['zone'] + list(cat_ids)]
    return crosstab_df


@delayed
def _single_chunk_crosstab(
    zones_block: np.array,
    values_block: np.array,
    unique_zones: np.array,
    zone_ids: np.array,
    unique_cats,
    cat_ids,
    nodata_values: Union[int, float],
):
    results = {}
    if len(values_block.shape) == 2:
        results[TOTAL_COUNT] = []
    for cat in cat_ids:
        results[cat] = []

    _, values_by_zones, zone_breaks = _sort_and_stride(
        zones_block, values_block, unique_zones
    )

    start = 0
    for i in range(len(unique_zones)):
        end = zone_breaks[i]
        if unique_zones[i] in zone_ids:
            # get data for zone unique_zones[i]
            zone_values = _get_zone_values(values_by_zones, start, end)
            if len(values_block.shape) == 2:
                _single_zone_crosstab_2d(
                    zone_values, unique_cats, cat_ids, nodata_values, results
                )
            else:
                _single_zone_crosstab_3d(
                    zone_values, unique_cats, cat_ids, nodata_values, results, _DEFAULT_STATS['count']  # noqa
                )

        start = end

    if TOTAL_COUNT in results:
        results[TOTAL_COUNT] = np.array(results[TOTAL_COUNT], dtype=np.float32)
    for cat in cat_ids:
        results[cat] = np.array(results[cat])

    return results


@delayed
def _select_ids(unique_ids, ids):
    selected_ids = []
    for i in ids:
        if i in unique_ids:
            selected_ids.append(i)
    return selected_ids


@delayed
def _crosstab_df_dask(crosstab_by_block, zone_ids, cat_ids, agg):
    result = crosstab_by_block[0]
    for i in range(1, len(crosstab_by_block)):
        for k in crosstab_by_block[i]:
            result[k] += crosstab_by_block[i][k]

    if agg == 'percentage':
        # replace 0s with nans to avoid dividing by 0 error
        result[TOTAL_COUNT][result[TOTAL_COUNT] == 0] = np.nan
        for cat in cat_ids:
            result[cat] = result[cat] / result[TOTAL_COUNT] * 100

    df = pd.DataFrame(result)
    df['zone'] = zone_ids
    columns = ['zone'] + list(cat_ids)
    df = df[columns]
    return df


def _crosstab_dask_numpy(
    zones: np.ndarray,
    values: np.ndarray,
    zone_ids: List[Union[int, float]],
    unique_cats: np.ndarray,
    cat_ids: Union[List, np.ndarray],
    nodata_values: Union[int, float],
    agg: str,
):
    # find ids for all zones
    unique_zones = np.unique(zones[np.isfinite(zones)])
    if zone_ids is None:
        zone_ids = unique_zones
    else:
        zone_ids = _select_ids(unique_zones, zone_ids)

    cat_ids = _select_ids(unique_cats, cat_ids)

    zones_blocks = zones.to_delayed().ravel()
    values_blocks = values.to_delayed().ravel()

    crosstab_by_block = [
        _single_chunk_crosstab(
            z, v, unique_zones, zone_ids,
            unique_cats, cat_ids, nodata_values
        )
        for z, v in zip(zones_blocks, values_blocks)
    ]

    crosstab_df = _crosstab_df_dask(
        crosstab_by_block, zone_ids, cat_ids, agg
    )
    return dd.from_delayed(crosstab_df)


def crosstab(
    zones: xr.DataArray,
    values: xr.DataArray,
    zone_ids: List[Union[int, float]] = None,
    cat_ids: List[Union[int, float]] = None,
    layer: Optional[int] = None,
    agg: Optional[str] = "count",
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
        If the `values` data is 2D, available options are: `percentage`, and `count`.
        If `values` is 3D and is numpy-backed, available options are:
        `min`, `max`, `mean`, `sum`, `std`, `var`, and `count`.
        If `values` is 3D and is dask with numpy-backed, the only available option is `count`.

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
    crosstab() works with NumPy backed DataArray.

    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.zonal import crosstab

        >>> values_data = np.asarray([
            [0, 0, 10, 20],
            [0, 0, 0, 10],
            [0, np.nan, 20, 50],
            [10, 30, 40, np.inf],
            [10, 10, 50, 0]])
        >>> values = xr.DataArray(values_data)
        >>> zones_data = np.asarray([
            [1, 1, 6, 6],
            [1, np.nan, 6, 6],
            [3, 5, 6, 6],
            [3, 5, 7, np.nan],
            [3, 7, 7, 0]])
        >>> zones = xr.DataArray(zones_data)
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

    crosstab() works with Dask with NumPy backed DataArray.

    .. sourcecode:: python

        >>> import dask.array as da
        >>> values_dask = xr.DataArray(da.from_array(values_data, chunks=(3, 3)))
        >>> zones_dask = xr.DataArray(da.from_array(zones_data, chunks=(3, 3)))
        >>> dask_df = crosstab(zones=zones_dask, values=values_dask)
        >>> print(dask_df)
            Dask DataFrame Structure:
            zone    0.0 10.0    20.0    30.0    40.0    50.0
            npartitions=5
            0   float64 int64   int64   int64   int64   int64   int64
            1   ... ... ... ... ... ... ...
            ... ... ... ... ... ... ... ...
            4   ... ... ... ... ... ... ...
            5   ... ... ... ... ... ... ...
            Dask Name: astype, 1186 tasks
        >>> print(dask_df.compute())
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
    agg_3d_numpy = _DEFAULT_STATS.keys()
    agg_3d_dask = ["count"]

    if values.ndim == 2 and agg not in agg_2d:
        raise ValueError(
            f"`agg` method for 2D data array must be one of following {agg_2d}"
        )

    if values.ndim == 3:
        if isinstance(values.data, np.ndarray) and agg not in agg_3d_numpy:
            raise ValueError(
                f"`agg` method for 3D numpy backed data array must be one of following {agg_3d_numpy}"  # noqa
            )
        if isinstance(values.data, da.Array) and agg not in agg_3d_dask:
            raise ValueError(
                f"`agg` method for 3D dask backed data array must be one of following {agg_3d_dask}"
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

        if isinstance(values.data, da.Array):
            # dask case, rechunk if necessary
            zones_chunks = zones.chunks
            expected_values_chunks = {
                0: (values.shape[0],),
                1: zones_chunks[0],
                2: zones_chunks[1],
            }
            actual_values_chunks = {
                i: values.chunks[i] for i in range(3)
            }
            if actual_values_chunks != expected_values_chunks:
                values.data = values.data.rechunk(expected_values_chunks)

    # find categories
    unique_cats, cat_ids = _find_cats(values, cat_ids, nodata_values)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_crosstab_numpy,
        dask_func=_crosstab_dask_numpy,
        cupy_func=lambda *args: not_implemented_func(
            *args, messages='crosstab() does not support cupy backed DataArray'
        ),
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages='crosstab() does not support dask with cupy backed DataArray'  # noqa
        ),
    )
    crosstab_df = mapper(values)(
        zones.data, values.data,
        zone_ids, unique_cats, cat_ids, nodata_values, agg
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

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.zonal import apply
        >>> zones_val = np.array([
            [1, 1, 0, 2],
            [0, 2, 1, 2]])
        >>> zones = xr.DataArray(zones_val)
        >>> values_val = np.array([
            [2, -1, 5, 3],
            [3, np.nan, 20, 10]])
        >>> agg = xr.DataArray(values_val)
        >>> func = lambda x: 0
        >>> apply(zones, agg, func)
        >>> agg
        array([[0, 0, 5, 0],
               [3, np.nan, 0, 0]])
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
        >>> from xrspatial.zonal import suggest_zonal_canvas

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
        >>> height, width
        (1537, 2376)
        >>> cvs = ds.Canvas(x_range=x_range, y_range=y_range,
        >>>             plot_height=height, plot_width=width)
        >>> spatial_df = GeoDataFrame(df, geometry='geometry')
        >>> agg = cvs.polygons(spatial_df, 'geometry', agg=ds.max('id'))
        >>> min_poly_id = df.area.argmin()
        >>> actual_min_pixels = len(np.where(agg.data==min_poly_id)[0])
        >>> actual_min_pixels
        22
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
                src_window[7] = data[min(y + 1, rows - 1), min(x + 1, cols - 1)]  # noqa

                area_window[0] = out[max(y - 1, 0), max(x - 1, 0)]
                area_window[1] = out[y, max(x - 1, 0)]
                area_window[2] = out[min(y + 1, rows - 1), max(x - 1, 0)]
                area_window[3] = out[max(y - 1, 0), x]
                area_window[4] = out[min(y + 1, rows - 1), x]
                area_window[5] = out[max(y - 1, 0), min(x + 1, cols - 1)]
                area_window[6] = out[y, min(x + 1, cols - 1)]
                area_window[7] = out[min(y + 1, rows - 1), min(x + 1, cols - 1)]  # noqa

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
                src_window[7] = data[min(y + 1, rows - 1), min(x + 1, cols - 1)]  # noqa

                area_window[0] = out[max(y - 1, 0), max(x - 1, 0)]
                area_window[1] = out[y, max(x - 1, 0)]
                area_window[2] = out[min(y + 1, rows - 1), max(x - 1, 0)]
                area_window[3] = out[max(y - 1, 0), x]
                area_window[4] = out[min(y + 1, rows - 1), x]
                area_window[5] = out[max(y - 1, 0), min(x + 1, cols - 1)]
                area_window[6] = out[y, min(x + 1, cols - 1)]
                area_window[7] = out[min(y + 1, rows - 1), min(x + 1, cols - 1)]  # noqa

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

        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr

        from xrspatial import generate_terrain
        from xrspatial.zonal import regions


        # Generate Example Terrain
        W = 500
        H = 300

        template_terrain = xr.DataArray(np.zeros((H, W)))
        x_range=(-20e6, 20e6)
        y_range=(-20e6, 20e6)

        terrain_agg = generate_terrain(
            template_terrain, x_range=x_range, y_range=y_range
        )

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
        array([[1264.02296597, 1261.947921  ],
               [1285.37105519, 1282.48079719],
               [1306.02339636, 1303.4069579 ]])
        Coordinates:
        * lon      (lon) float64 -3.96e+06 -3.88e+06
        * lat      (lat) float64 6.733e+06 6.867e+06 7e+06
        Attributes:
            res:            (80000.0, 133333.3333333333)
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
            res:            (80000.0, 133333.3333333333)
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

        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr

        from xrspatial import generate_terrain
        from xrspatial.zonal import trim


        # Generate Example Terrain
        W = 500
        H = 300

        template_terrain = xr.DataArray(np.zeros((H, W)))
        x_range=(-20e6, 20e6)
        y_range=(-20e6, 20e6)

        terrain_agg = generate_terrain(
            template_terrain, x_range=x_range, y_range=y_range
        )

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
            'res': (80000.0, 133333.3333333333),
            'Description': 'Example Terrain',
            'units': 'km',
            'Max Elevation': '4000',
        }

        >>> print(trimmed_agg.shape)
        (268, 500)

        >>> print(trimmed_agg.attrs)
        {
            'res': (80000.0, 133333.3333333333),
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

        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr

        from xrspatial import generate_terrain
        from xrspatial.zonal import crop


        # Generate Example Terrain
        W = 500
        H = 300

        template_terrain = xr.DataArray(np.zeros((H, W)))
        x_range=(-20e6, 20e6)
        y_range=(-20e6, 20e6)

        terrain_agg = generate_terrain(
            template_terrain, x_range=x_range, y_range=y_range
        )

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
            'res': (80000.0, 133333.3333333333),
            'Description': 'Example Terrain',
            'units': 'km',
            'Max Elevation': '4000',
        }

        >>> print(cropped_agg.shape)
        (300, 250)

        >>> print(cropped_agg.attrs)
        {
            'res': (80000.0, 133333.3333333333),
            'Description': 'Example Crop',
            'units': 'km',
            'Max Elevation': '4000',
        }
    """
    top, bottom, left, right = _crop(zones.data, zones_ids)
    arr = values[top: bottom + 1, left: right + 1]
    arr.name = name
    return arr
