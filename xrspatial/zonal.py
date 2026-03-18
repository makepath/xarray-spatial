from __future__ import annotations

# standard library
import copy
from math import sqrt
from typing import Callable, Dict, List, Optional, Union


# 3rd-party
try:
    import dask
    import dask.array as da
except ImportError:
    dask = None
    da = None

try:
    import dask.dataframe as dd
except ImportError:
    dd = None

try:
    from dask import delayed
except ImportError:
    delayed = lambda x: None  # noqa

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

# local modules
from xrspatial.utils import (
    ArrayTypeFunctionMapping, _validate_raster, cuda_args, has_cuda_and_cupy,
    is_cupy_array, is_dask_cupy,
    ngjit, not_implemented_func, validate_arrays,
)
from xrspatial.utils import has_dask_array

TOTAL_COUNT = '_total_count'


def _maybe_rasterize_zones(zones, values, column=None, rasterize_kw=None):
    """If *zones* is vector data, rasterize it using *values* as the template.

    Accepts:
    - ``GeoDataFrame`` (requires *column* to identify the zone-ID field)
    - list of ``(geometry, value)`` pairs

    Returns a 2-D ``xr.DataArray`` of zone IDs aligned to *values*.
    If *zones* is already a DataArray it is returned unchanged.
    """
    if isinstance(zones, xr.DataArray):
        return zones

    # list-of-pairs: [(geom, value), ...]
    is_pairs = (
        isinstance(zones, (list, tuple))
        and len(zones) > 0
        and isinstance(zones[0], (list, tuple))
        and len(zones[0]) == 2
    )

    # GeoDataFrame
    is_gdf = False
    try:
        import geopandas as gpd
        is_gdf = isinstance(zones, gpd.GeoDataFrame)
    except ImportError:
        pass

    if not is_pairs and not is_gdf:
        return zones

    from .rasterize import rasterize

    # Build the template from values (first 2D variable if Dataset)
    if isinstance(values, xr.Dataset):
        for var in values.data_vars:
            da_var = values[var]
            if da_var.ndim >= 2 and 'y' in da_var.dims and 'x' in da_var.dims:
                like = da_var
                break
        else:
            raise ValueError(
                "values Dataset has no 2D variable with 'y' and 'x' "
                "dimensions to use as rasterize template"
            )
    elif isinstance(values, xr.DataArray):
        if values.ndim >= 2:
            like = values
        else:
            raise ValueError(
                "values must be at least 2D to use as rasterize template"
            )
    else:
        raise TypeError(
            f"values must be an xr.DataArray or xr.Dataset, got {type(values)}"
        )

    kw = dict(rasterize_kw or {})
    kw['like'] = like

    if is_gdf:
        if column is None:
            raise ValueError(
                "column is required when zones is a GeoDataFrame. "
                "Specify which column contains zone IDs."
            )
        kw['column'] = column
    elif is_pairs:
        if column is not None:
            raise ValueError(
                "column should not be set when zones is a list of "
                "(geometry, value) pairs"
            )

    return rasterize(zones, **kw)


def _unique_finite_zones(arr):
    """Sorted unique finite values from *arr* without full materialisation.

    For dask arrays uses ``da.unique`` (per-chunk reduction) so the full
    array is never pulled into RAM.
    """
    if da is not None and isinstance(arr, da.Array):
        uniq = da.unique(arr).compute()
        return uniq[np.isfinite(uniq)]
    return np.unique(arr[np.isfinite(arr)])


def _unique_finite_cats(arr, nodata_values):
    """Sorted unique values excluding NaN, Inf, and *nodata_values*.

    Dask-safe: uses ``da.unique`` so the full array is never materialised.
    """
    if da is not None and isinstance(arr, da.Array):
        uniq = da.unique(arr).compute()
        mask = np.isfinite(uniq)
        if nodata_values is not None:
            mask &= (uniq != nodata_values)
        return uniq[mask]
    mask = np.isfinite(arr)
    if nodata_values is not None:
        mask &= (arr != nodata_values)
    return np.unique(arr[mask])


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


def _stats_majority(data):
    if isinstance(data, np.ndarray):
        # numpy case
        values, counts = np.unique(data, return_counts=True)
        return values[np.argmax(counts)]
    elif isinstance(data, cupy.ndarray):
        # cupy case
        values, counts = cupy.unique(data, return_counts=True)
        return values[cupy.argmax(counts)]
    else:
        # dask case
        values, counts = da.unique(data, return_counts=True)
        return values[da.argmax(counts)]


_DEFAULT_STATS = dict(
    mean=lambda z: z.mean(),
    max=lambda z: z.max(),
    min=lambda z: z.min(),
    sum=lambda z: z.sum(),
    std=lambda z: z.std(),
    var=lambda z: z.var(),
    count=lambda z: _stats_count(z),
    majority=lambda z: _stats_majority(z),
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
    unique_zones = _unique_finite_zones(zones)

    select_all_zones = False
    # selecte zones to do analysis
    if zone_ids is None:
        zone_ids = unique_zones
        select_all_zones = True

    zones_blocks = zones.to_delayed().ravel()
    values_blocks = values.to_delayed().ravel()

    stats_dict = {}
    stats_dict["zone"] = da.from_delayed(  # zone column
        delayed(lambda x: x)(unique_zones),
        shape=(np.nan,), dtype=unique_zones.dtype,
    )

    compute_sum_squares = False
    compute_sum = False
    compute_count = False

    if any(s in stats_funcs for s in ('mean', 'std', 'var')):
        compute_sum = True
        compute_count = True

    if any(s in stats_funcs for s in ('std', 'var')):
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
    stats_df = dd.concat([dd.from_dask_array(s) for s in stats_dict.values()], axis=1, ignore_unknown_divisions=True)
    # name columns
    stats_df.columns = stats_dict.keys()
    # select columns (only include stats that were actually computed)
    computed_stats = [s for s in stats_funcs.keys() if s in stats_dict]
    stats_df = stats_df[['zone'] + computed_stats]

    if not select_all_zones:
        # only return zones specified in `zone_ids`
        selected_rows = []
        for index, row in stats_df.iterrows():
            if row['zone'] in zone_ids:
                selected_rows.append(stats_df.loc[index])
        stats_df = dd.concat(selected_rows, ignore_unknown_divisions=True)

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
    unique_zones = _unique_finite_zones(zones)
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

            assert (len(result.shape) == 0)

            stats_dict[stats].append(cupy.float_(result))

    stats_df = pd.DataFrame(stats_dict)
    stats_df.set_index("zone")
    return stats_df


def _stats_dask_cupy(
    zones,
    values,
    zone_ids,
    stats_funcs,
    nodata_values,
):
    zones_cpu = zones.map_blocks(
        lambda x: x.get(), dtype=zones.dtype, meta=np.array(()),
    )
    values_cpu = values.map_blocks(
        lambda x: x.get(), dtype=values.dtype, meta=np.array(()),
    )
    return _stats_dask_numpy(
        zones_cpu, values_cpu, zone_ids, stats_funcs, nodata_values,
    )


def stats(
    zones,
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
        "majority",
    ],
    nodata_values: Union[int, float] = None,
    return_type: str = 'pandas.DataFrame',
    column: Optional[str] = None,
    rasterize_kw: Optional[dict] = None,
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
    zones : xr.DataArray, GeoDataFrame, or list of (geometry, value) pairs
        Zone definitions. Can be:

        - A 2D xarray DataArray of numeric zone IDs.
        - A ``geopandas.GeoDataFrame`` (requires *column*).
        - A list of ``(shapely geometry, zone_id)`` pairs.

        When vector input is provided, ``rasterize()`` is called internally
        using *values* as the template grid. Results depend on raster
        resolution.

    values : xr.DataArray or xr.Dataset
        values is a 2D xarray DataArray of numeric values (integers or floats).
        The input `values` raster contains the input values used in
        calculating the output statistic for each zone. In dask case,
        the chunksizes of `zones` and `values` should be matching. If not,
        `values` will be rechunked to be the same as of `zones`.
        When a Dataset is passed, stats are computed for each variable
        and columns are prefixed with the variable name
        (e.g. ``elevation_mean``).
        For 3D time-series DataArrays, convert to a Dataset first using
        ``.to_dataset(dim='time')`` and pass the resulting Dataset.

    zone_ids : list of ints, or floats
        List of zones to be included in calculation. If no zone_ids provided,
        all zones will be used.

    stats_funcs : dict, or list of strings, default=['mean', 'max', 'min',
        'sum', 'std', 'var', 'count', 'majority']
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

    column : str, optional
        Column name in the GeoDataFrame that contains zone IDs.
        Required when *zones* is a GeoDataFrame; must not be set
        for list-of-pairs or DataArray input.

    rasterize_kw : dict, optional
        Extra keyword arguments forwarded to ``rasterize()`` when
        *zones* is vector input (e.g. ``{'all_touched': True}``).

    Returns
    -------
    stats_df : Union[pandas.DataFrame, dask.dataframe.DataFrame]
        A pandas DataFrame, or a dask DataFrame where each column
        is a statistic and each row is a zone with zone id.
        When ``values`` is a Dataset, the returned DataFrame has
        columns prefixed by the variable name (e.g. ``elevation_mean``,
        ``elevation_max``), and ``return_type`` must be
        ``'pandas.DataFrame'``.

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

    stats() works with 3D time-series DataArrays via Dataset conversion

    .. sourcecode:: python

        >>> # Convert a 3D time-series DataArray to a Dataset,
        >>> # then pass to stats() to get per-timestep statistics.
        >>> values_3d = xr.DataArray(
        ...     np.random.rand(2, 10, 10),
        ...     dims=['time', 'dim_0', 'dim_1'],
        ...     coords={'time': [2020, 2021]})
        >>> ds = values_3d.to_dataset(dim='time')
        >>> stats_df = stats(zones=zones, values=ds)
        >>> # Columns: zone, 2020_mean, 2020_max, ..., 2021_mean, 2021_max, ...
    """

    zones = _maybe_rasterize_zones(zones, values, column=column,
                                   rasterize_kw=rasterize_kw)

    # Dataset support: run stats per variable and merge into one DataFrame
    if isinstance(values, xr.Dataset):
        if return_type != 'pandas.DataFrame':
            raise ValueError(
                "return_type must be 'pandas.DataFrame' when values is a Dataset"
            )
        dfs = []
        for var_name in values.data_vars:
            df = stats(
                zones, values[var_name], zone_ids, stats_funcs,
                nodata_values, 'pandas.DataFrame',
            )
            df = df.rename(
                columns={c: f'{var_name}_{c}' for c in df.columns if c != 'zone'}
            )
            dfs.append(df)
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on='zone', how='outer')
        return result

    _validate_raster(zones, func_name='stats', name='zones', ndim=2)
    _validate_raster(values, func_name='stats', name='values', ndim=(2, 3))

    validate_arrays(zones, values)

    # validate stats_funcs
    if has_dask_array() and isinstance(values.data, da.Array) and not isinstance(stats_funcs, list):
        raise ValueError(
            "Got dask-backed DataArray as `values` aggregate. "
            "`stats_funcs` must be a subset of default supported stats "
            "`[\'mean\', \'max\', \'min\', \'sum\', \'std\', \'var\', \'count\']`"
        )

    if isinstance(stats_funcs, list):
        # create a dict of stats
        stats_funcs_dict = {}
        for stat_name in stats_funcs:
            func = _DEFAULT_STATS.get(stat_name, None)
            if func is None:
                err_str = f"Invalid stat name. {stat_name} option not supported."
                raise ValueError(err_str)
            stats_funcs_dict[stat_name] = func

    elif isinstance(stats_funcs, dict):
        stats_funcs_dict = stats_funcs.copy()

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda *args: _stats_numpy(*args, return_type=return_type),
        dask_func=_stats_dask_numpy,
        cupy_func=_stats_cupy,
        dask_cupy_func=_stats_dask_cupy,
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
        unique_cats = _unique_finite_cats(values.data, nodata_values)
    else:
        # 3D case
        unique_cats = values[values.dims[0]].data

    if cat_ids is None:
        cat_ids = unique_cats
    else:
        if isinstance(values.data, np.ndarray) or is_cupy_array(values.data):
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
    unique_zones = _unique_finite_zones(zones)
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
    unique_zones = _unique_finite_zones(zones)
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


def _crosstab_cupy(
    zones: np.ndarray,
    values: np.ndarray,
    zone_ids,
    unique_cats,
    cat_ids,
    nodata_values,
    agg: str,
):
    # unique_cats / cat_ids may be cupy arrays from _find_cats
    if is_cupy_array(unique_cats):
        unique_cats = cupy.asnumpy(unique_cats)
    if is_cupy_array(cat_ids):
        cat_ids = cupy.asnumpy(cat_ids)
    return _crosstab_numpy(
        cupy.asnumpy(zones), cupy.asnumpy(values),
        zone_ids, unique_cats, cat_ids, nodata_values, agg,
    )


def _crosstab_dask_cupy(
    zones,
    values,
    zone_ids,
    unique_cats,
    cat_ids,
    nodata_values,
    agg: str,
):
    zones_cpu = zones.map_blocks(
        lambda x: x.get(), dtype=zones.dtype, meta=np.array(()),
    )
    values_cpu = values.map_blocks(
        lambda x: x.get(), dtype=values.dtype, meta=np.array(()),
    )
    # unique_cats / cat_ids may be cupy arrays from _find_cats
    if is_cupy_array(unique_cats):
        unique_cats = cupy.asnumpy(unique_cats)
    if is_cupy_array(cat_ids):
        cat_ids = cupy.asnumpy(cat_ids)
    return _crosstab_dask_numpy(
        zones_cpu, values_cpu, zone_ids, unique_cats, cat_ids,
        nodata_values, agg,
    )


def crosstab(
    zones,
    values: xr.DataArray,
    zone_ids: List[Union[int, float]] = None,
    cat_ids: List[Union[int, float]] = None,
    layer: Optional[int] = None,
    agg: Optional[str] = "count",
    nodata_values: Optional[Union[int, float]] = None,
    column: Optional[str] = None,
    rasterize_kw: Optional[dict] = None,
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
    zones : xr.DataArray, GeoDataFrame, or list of (geometry, value) pairs
        Zone definitions. Can be:

        - A 2D xarray DataArray of integers or floats.
        - A ``geopandas.GeoDataFrame`` (requires *column*).
        - A list of ``(shapely geometry, zone_id)`` pairs.

        When vector input is provided, ``rasterize()`` is called internally
        using *values* as the template grid.

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

    column : str, optional
        Column name in the GeoDataFrame that contains zone IDs.
        Required when *zones* is a GeoDataFrame.

    rasterize_kw : dict, optional
        Extra keyword arguments forwarded to ``rasterize()`` when
        *zones* is vector input (e.g. ``{'all_touched': True}``).

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

    zones = _maybe_rasterize_zones(zones, values, column=column,
                                   rasterize_kw=rasterize_kw)

    _validate_raster(zones, func_name='crosstab', name='zones', ndim=2)
    _validate_raster(values, func_name='crosstab', name='values', ndim=(2, 3))

    # For 2D values, validate and align chunks between zones and values
    # This is critical for dask arrays that may come from different sources
    # (e.g., xarray Datasets via to_array().sel())
    if values.ndim == 2:
        validate_arrays(zones, values)

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
        if has_dask_array() and isinstance(values.data, da.Array) and agg not in agg_3d_dask:
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

        if has_dask_array() and isinstance(values.data, da.Array):
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
        cupy_func=_crosstab_cupy,
        dask_cupy_func=_crosstab_dask_cupy,
    )
    crosstab_df = mapper(values)(
        zones.data, values.data,
        zone_ids, unique_cats, cat_ids, nodata_values, agg
    )
    return crosstab_df


def _apply_numpy(zones_data, values_data, func, nodata):
    out = values_data.copy()
    if nodata is not None:
        zone_mask = zones_data != nodata
    else:
        zone_mask = np.ones(zones_data.shape, dtype=bool)
    vfunc = np.vectorize(func)
    if values_data.ndim == 2:
        out[zone_mask] = vfunc(values_data[zone_mask])
    else:  # 3D
        for k in range(values_data.shape[2]):
            out[:, :, k][zone_mask] = vfunc(values_data[:, :, k][zone_mask])
    return out


def _make_apply_kernel(func):
    """Build a CUDA kernel that applies *func* element-wise."""
    from numba import cuda as nb_cuda

    device_func = nb_cuda.jit(device=True)(func)

    @nb_cuda.jit
    def _kernel(zones, values, out, nodata_val, has_nodata):
        y, x = nb_cuda.grid(2)
        if y < zones.shape[0] and x < zones.shape[1]:
            if has_nodata and zones[y, x] == nodata_val:
                return
            out[y, x] = device_func(values[y, x])

    return _kernel


def _apply_cupy_gpu(zones_data, values_data, kernel, nodata):
    """Run the CUDA apply kernel on cupy arrays."""
    out = values_data.copy()
    has_nodata = nodata is not None
    nodata_val = nodata if has_nodata else 0

    griddim, blockdim = cuda_args(values_data.shape[:2])

    if values_data.ndim == 2:
        kernel[griddim, blockdim](
            zones_data, values_data, out, nodata_val, has_nodata,
        )
    else:
        for k in range(values_data.shape[2]):
            kernel[griddim, blockdim](
                zones_data, values_data[:, :, k], out[:, :, k],
                nodata_val, has_nodata,
            )
    return out


def _apply_cupy(zones_data, values_data, func, nodata):
    try:
        kernel = _make_apply_kernel(func)
        return _apply_cupy_gpu(zones_data, values_data, kernel, nodata)
    except Exception:
        result_np = _apply_numpy(zones_data.get(), values_data.get(), func, nodata)
        return cupy.asarray(result_np)


def _apply_dask_numpy(zones_data, values_data, func, nodata):
    def _chunk_fn(zones_chunk, values_chunk):
        return _apply_numpy(zones_chunk, values_chunk, func, nodata)

    if values_data.ndim == 2:
        return da.map_blocks(
            _chunk_fn, zones_data, values_data,
            dtype=values_data.dtype, meta=np.array(()),
        )
    else:
        layers = []
        for k in range(values_data.shape[2]):
            layer = values_data[:, :, k].rechunk(zones_data.chunks)
            layers.append(da.map_blocks(
                _chunk_fn, zones_data, layer,
                dtype=values_data.dtype, meta=np.array(()),
            ))
        return da.stack(layers, axis=2)


def _apply_dask_cupy(zones_data, values_data, func, nodata):
    # Try GPU: build kernel once, reuse across all chunks
    try:
        kernel = _make_apply_kernel(func)
        gpu_ok = True
    except Exception:
        gpu_ok = False

    if gpu_ok:
        def _chunk_fn(zones_chunk, values_chunk):
            try:
                return _apply_cupy_gpu(zones_chunk, values_chunk, kernel, nodata)
            except Exception:
                result_np = _apply_numpy(
                    zones_chunk.get(), values_chunk.get(), func, nodata,
                )
                return cupy.asarray(result_np)
    else:
        def _chunk_fn(zones_chunk, values_chunk):
            result_np = _apply_numpy(
                zones_chunk.get(), values_chunk.get(), func, nodata,
            )
            return cupy.asarray(result_np)

    if values_data.ndim == 2:
        return da.map_blocks(
            _chunk_fn, zones_data, values_data,
            dtype=values_data.dtype, meta=cupy.array(()),
        )
    else:
        layers = []
        for k in range(values_data.shape[2]):
            layer = values_data[:, :, k].rechunk(zones_data.chunks)
            layers.append(da.map_blocks(
                _chunk_fn, zones_data, layer,
                dtype=values_data.dtype, meta=cupy.array(()),
            ))
        return da.stack(layers, axis=2)


def apply(
    zones,
    values: xr.DataArray,
    func: Callable,
    nodata: Optional[int] = 0,
    column: Optional[str] = None,
    rasterize_kw: Optional[dict] = None,
):
    """
    Apply a function to the `values` agg within zones in `zones` agg.
    Returns a new DataArray with the function applied.

    Parameters
    ----------
    zones : xr.DataArray, GeoDataFrame, or list of (geometry, value) pairs
        Zone definitions. Can be:

        - A 2D xarray DataArray of integers.
        - A ``geopandas.GeoDataFrame`` (requires *column*).
        - A list of ``(shapely geometry, zone_id)`` pairs.

        When vector input is provided, ``rasterize()`` is called internally
        using *values* as the template grid. Use ``rasterize_kw={'dtype': int}``
        to produce integer zones (required by ``apply``).

    values : xr.DataArray
        values.data is either a 2D or 3D array of integers or floats.
        The input value raster.

    func : callable function to apply.

    nodata: int, default=0
        Nodata value in `zones` raster.
        Cells with `nodata` does not belong to any zone,
        and thus excluded from calculation.
        Set to None to apply func to all cells.

    column : str, optional
        Column name in the GeoDataFrame that contains zone IDs.
        Required when *zones* is a GeoDataFrame.

    rasterize_kw : dict, optional
        Extra keyword arguments forwarded to ``rasterize()`` when
        *zones* is vector input.

    Returns
    -------
    result : xr.DataArray
        A new DataArray with the same shape, dims, coords, and attrs
        as `values`, with `func` applied to cells within zones.

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
        >>> result = apply(zones, agg, func)
        >>> result
        array([[0, 0, 5, 0],
               [3, np.nan, 0, 0]])
    """
    zones = _maybe_rasterize_zones(zones, values, column=column,
                                   rasterize_kw=rasterize_kw)

    _validate_raster(zones, func_name='apply', name='zones', ndim=2, integer_only=True)
    _validate_raster(values, func_name='apply', name='values', ndim=(2, 3))

    if zones.shape != values.shape[:2]:
        raise ValueError("Incompatible shapes between `zones` and `values`")

    # align chunks for 2D values
    if values.ndim == 2:
        validate_arrays(zones, values)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_apply_numpy,
        dask_func=_apply_dask_numpy,
        cupy_func=_apply_cupy,
        dask_cupy_func=_apply_dask_cupy,
    )
    out = mapper(values)(zones.data, values.data, func, nodata)

    return xr.DataArray(
        out, dims=values.dims, coords=values.coords, attrs=values.attrs,
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

        >>> # Imports (datashader is optional: pip install datashader)
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


def _regions_numpy(data, neighborhood):
    """Connected-component labeling using scipy.ndimage.label (union-find)."""
    from scipy.ndimage import label

    if neighborhood == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        structure = np.ones((3, 3), dtype=int)

    is_float = np.issubdtype(data.dtype, np.floating)
    valid = ~np.isnan(data) if is_float else np.ones(data.shape, dtype=bool)
    unique_vals = np.unique(data[valid])

    out = np.full(data.shape, np.nan, dtype=np.float64)
    uid = 1
    for v in unique_vals:
        mask = (data == v)
        if is_float:
            mask &= valid
        labeled, n_features = label(mask, structure=structure)
        for region_id in range(1, n_features + 1):
            out[labeled == region_id] = uid
            uid += 1

    return out


def _available_memory_bytes():
    """Best-effort estimate of available memory in bytes."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    return 2 * 1024 ** 3


def _regions_dask(data, neighborhood):
    """Dask backend: compute to numpy and run scipy label."""
    avail = _available_memory_bytes()
    nbytes = data.nbytes
    if nbytes * 5 > 0.5 * avail:
        raise MemoryError(
            f"regions() requires ~{nbytes * 5 / 1e9:.1f} GB but only "
            f"{avail / 1e9:.1f} GB available."
        )

    np_data = data.compute()
    result = _regions_numpy(np_data, neighborhood)
    return da.from_array(result, chunks=data.chunks)


def _regions_cupy(data, neighborhood):
    """CuPy GPU backend using cupyx.scipy.ndimage.label."""
    import cupy as cp
    from cupyx.scipy.ndimage import label as cp_label

    if neighborhood == 4:
        structure = cp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        structure = cp.ones((3, 3), dtype=int)

    is_float = cp.issubdtype(data.dtype, cp.floating)
    valid = ~cp.isnan(data) if is_float else cp.ones(data.shape, dtype=bool)
    unique_vals = cp.unique(data[valid])

    out = cp.full(data.shape, cp.nan, dtype=cp.float64)
    uid = 1
    for v in unique_vals:
        mask = (data == v)
        if is_float:
            mask &= valid
        labeled, n_features = cp_label(mask, structure=structure)
        for region_id in range(1, n_features + 1):
            out[labeled == region_id] = uid
            uid += 1

    return out


def _regions_dask_cupy(data, neighborhood):
    """Dask+CuPy backend: compute to cupy and run GPU label."""
    cp_data = data.compute()
    result = _regions_cupy(cp_data, neighborhood)
    return da.from_array(result, chunks=data.chunks)


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
    neighborhood : int, default=4
        4 or 8 pixel-based connectivity.
    name : str, default='regions'
        Output xr.DataArray.name property.

    Returns
    -------
    regions_agg : xarray.DataArray

    References
    ----------
        - Tomislav Hengl: http://spatial-analyst.net/ILWIS/htm/ilwisapp/areanumbering_algorithm.htm # noqa

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial.zonal import regions

        >>> # Create a raster with distinct value regions
        >>> arr = np.array([[1, 1, 0, 2, 2],
        ...                 [1, 1, 0, 2, 2],
        ...                 [0, 0, 0, 0, 0],
        ...                 [3, 3, 0, 3, 3],
        ...                 [3, 3, 0, 3, 3]], dtype=np.float64)
        >>> raster = xr.DataArray(arr, dims=['y', 'x'])
        >>> print(raster.values)
        [[1. 1. 0. 2. 2.]
         [1. 1. 0. 2. 2.]
         [0. 0. 0. 0. 0.]
         [3. 3. 0. 3. 3.]
         [3. 3. 0. 3. 3.]]

        >>> # With 4-connectivity, each group of connected same-value
        >>> # pixels becomes a unique region
        >>> result = regions(raster, neighborhood=4)
        >>> print(result.values)
        [[1. 1. 2. 3. 3.]
         [1. 1. 2. 3. 3.]
         [2. 2. 2. 2. 2.]
         [5. 5. 2. 6. 6.]
         [5. 5. 2. 6. 6.]]

        >>> # Note: The two bottom-corner 3-regions are separate because
        >>> # they are not connected (the zero-valued cross separates them)
        >>> print(f"Number of unique regions: {len(np.unique(result.values))}")
        Number of unique regions: 5

        >>> # With 8-connectivity, diagonal neighbors also connect regions
        >>> diagonal = np.array([[1, 0, 1],
        ...                      [0, 1, 0],
        ...                      [1, 0, 1]], dtype=np.float64)
        >>> raster_diag = xr.DataArray(diagonal, dims=['y', 'x'])
        >>> result_8 = regions(raster_diag, neighborhood=8)
        >>> print(result_8.values)
        [[1. 2. 1.]
         [2. 1. 2.]
         [1. 2. 1.]]

        >>> # All 1s are connected diagonally into one region,
        >>> # all 0s form another region
        >>> print(f"Number of unique regions: {len(np.unique(result_8.values))}")
        Number of unique regions: 2
    """
    _validate_raster(raster, func_name='regions', name='raster', ndim=2)

    if neighborhood not in (4, 8):
        raise ValueError("`neighborhood` value must be either 4 or 8)")

    data = raster.data

    if isinstance(data, np.ndarray):
        out = _regions_numpy(data, neighborhood)
    elif has_cuda_and_cupy() and is_cupy_array(data):
        out = _regions_cupy(data, neighborhood)
    elif da is not None and isinstance(data, da.Array):
        if is_dask_cupy(raster):
            out = _regions_dask_cupy(data, neighborhood)
        else:
            out = _regions_dask(data, neighborhood)
    else:
        raise TypeError(
            f"Unsupported array type {type(data).__name__} for regions()"
        )

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


def _trim_bounds_dask(data, excludes):
    """Find trim bounds using lazy dask reductions (O(rows+cols) memory)."""
    excluded = da.zeros_like(data, dtype=bool)
    for v in excludes:
        if isinstance(v, float) and np.isnan(v):
            excluded = excluded | da.isnan(data)
        else:
            excluded = excluded | (data == v)

    all_excl_rows = excluded.all(axis=1)
    all_excl_cols = excluded.all(axis=0)
    row_mask, col_mask = dask.compute(all_excl_rows, all_excl_cols)

    # dask+cupy computes to cupy arrays; move to numpy for np.where
    if is_cupy_array(row_mask):
        row_mask = row_mask.get()
    if is_cupy_array(col_mask):
        col_mask = col_mask.get()

    data_rows = np.where(~np.asarray(row_mask))[0]
    data_cols = np.where(~np.asarray(col_mask))[0]

    if len(data_rows) == 0 or len(data_cols) == 0:
        return 0, -1, 0, -1  # empty slice

    return (int(data_rows[0]), int(data_rows[-1]),
            int(data_cols[0]), int(data_cols[-1]))


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
    _validate_raster(raster, func_name='trim', name='raster', ndim=2)

    data = raster.data
    if has_dask_array() and isinstance(data, da.Array):
        top, bottom, left, right = _trim_bounds_dask(data, values)
    else:
        if is_cupy_array(data):
            data = data.get()
        top, bottom, left, right = _trim(data, values)

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


def _crop_bounds_dask(data, target_values):
    """Find crop bounds using lazy dask reductions (O(rows+cols) memory)."""
    matched = da.zeros_like(data, dtype=bool)
    for v in target_values:
        matched = matched | (data == v)

    any_match_rows = matched.any(axis=1)
    any_match_cols = matched.any(axis=0)
    row_mask, col_mask = dask.compute(any_match_rows, any_match_cols)

    # dask+cupy computes to cupy arrays; move to numpy for np.where
    if is_cupy_array(row_mask):
        row_mask = row_mask.get()
    if is_cupy_array(col_mask):
        col_mask = col_mask.get()

    match_rows = np.where(np.asarray(row_mask))[0]
    match_cols = np.where(np.asarray(col_mask))[0]

    if len(match_rows) == 0 or len(match_cols) == 0:
        return 0, data.shape[0] - 1, 0, data.shape[1] - 1

    return (int(match_rows[0]), int(match_rows[-1]),
            int(match_cols[0]), int(match_cols[-1]))


def crop(
    zones,
    values: xr.DataArray,
    zones_ids: Union[list, tuple],
    name: str = "crop",
    column: Optional[str] = None,
    rasterize_kw: Optional[dict] = None,
):
    """
    Crop scans from edges and eliminates rows / cols until one of the
    input values is found.

    Parameters
    ----------
    zones : xr.DataArray, GeoDataFrame, or list of (geometry, value) pairs
        Zone definitions. Can be a 2D DataArray, a GeoDataFrame
        (requires *column*), or a list of ``(geometry, zone_id)`` pairs.

    values: xr.DataArray
        Input values raster.

    zones_ids : list or tuple
        List of zone ids to crop raster.

    name: str, default='crop'
        Output xr.DataArray.name property.

    column : str, optional
        Column name in the GeoDataFrame that contains zone IDs.
        Required when *zones* is a GeoDataFrame.

    rasterize_kw : dict, optional
        Extra keyword arguments forwarded to ``rasterize()`` when
        *zones* is vector input.

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

        # Create a simple zone raster (0 = below median, 1 = above)
        zones_agg = (terrain_agg > terrain_agg.median()).astype(int)
        zones_agg.attrs = terrain_agg.attrs
        zones_agg = zones_agg.rename('Zone')

        # Crop to keep only the above-median zone
        values_agg = terrain_agg[0:300, 0:250]
        zones_sub = zones_agg[0:300, 0:250]
        cropped_agg = crop(
            zones=zones_sub,
            values=values_agg,
            zones_ids=[1],
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
    zones = _maybe_rasterize_zones(zones, values, column=column,
                                   rasterize_kw=rasterize_kw)

    _validate_raster(zones, func_name='crop', name='zones', ndim=2)
    _validate_raster(values, func_name='crop', name='values', ndim=2)

    data = zones.data
    if has_dask_array() and isinstance(data, da.Array):
        top, bottom, left, right = _crop_bounds_dask(data, zones_ids)
    else:
        if is_cupy_array(data):
            data = data.get()
        top, bottom, left, right = _crop(data, zones_ids)

    arr = values[top: bottom + 1, left: right + 1]
    arr.name = name
    return arr
