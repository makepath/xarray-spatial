import pytest
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import dask.dataframe as dd

from xrspatial import zonal_stats as stats
from xrspatial import zonal_apply as apply
from xrspatial import zonal_crosstab as crosstab
from xrspatial import suggest_zonal_canvas
from xrspatial import trim
from xrspatial import crop


from xrspatial.zonal import regions


def stats_create_zones_values(backend='numpy'):
    # create valid "zones" and "values" for testing stats()
    zones_val = np.array([[0, 1, 1, 2, 4, 0, 0],
                          [0, 0, 1, 1, 2, 1, 4],
                          [4, 2, 2, 4, 4, 4, 0]])
    zones = xr.DataArray(zones_val)

    values_val = np.array([[0, 12, 10, 2, 3.25, np.nan, np.nan],
                           [0, 0, -11, 4, -2.5, np.nan, 7],
                           [np.nan, 3.5, -9, 4, 2, 0, np.inf]])
    values = xr.DataArray(values_val)

    if 'dask' in backend:
        zones.data = da.from_array(zones.data, chunks=(3, 3))
        values.data = da.from_array(values.data, chunks=(3, 3))

    return zones, values


def test_stats_default():
    zones, values = stats_create_zones_values(backend='numpy')
    dask_zones, dask_values = stats_create_zones_values(backend='dask')

    unique_values = [0, 1, 2, 4]
    masked_values = np.ma.masked_invalid(values.values)
    zone_vals_0 = np.ma.masked_where(zones != 0, masked_values)
    zone_vals_1 = np.ma.masked_where(zones != 1, masked_values)
    zone_vals_2 = np.ma.masked_where(zones != 2, masked_values)
    zone_vals_3 = np.ma.masked_where(zones != 4, masked_values)

    zone_means = [
        zone_vals_0.mean(),
        zone_vals_1.mean(),
        zone_vals_2.mean(),
        zone_vals_3.mean()
    ]
    zone_maxes = [
        zone_vals_0.max(),
        zone_vals_1.max(),
        zone_vals_2.max(),
        zone_vals_3.max()
    ]
    zone_mins = [
        zone_vals_0.min(),
        zone_vals_1.min(),
        zone_vals_2.min(),
        zone_vals_3.min()
    ]
    zone_sums = [
        zone_vals_0.sum(),
        zone_vals_1.sum(),
        zone_vals_2.sum(),
        zone_vals_3.sum(),
    ]
    zone_stds = [
        zone_vals_0.std(),
        zone_vals_1.std(),
        zone_vals_2.std(),
        zone_vals_3.std()
    ]
    zone_vars = [
        zone_vals_0.var(),
        zone_vals_1.var(),
        zone_vals_2.var(),
        zone_vals_3.var()
    ]
    zone_counts = [
        np.ma.count(zone_vals_0),
        np.ma.count(zone_vals_1),
        np.ma.count(zone_vals_2),
        np.ma.count(zone_vals_3)
    ]

    # default stats_funcs=['mean', 'max', 'min', 'sum', 'std', 'var', 'count']
    df = stats(zones=zones, values=values)
    assert isinstance(df, pd.DataFrame)

    dask_df = stats(zones=dask_zones, values=dask_values)
    assert isinstance(dask_df, dd.DataFrame)

    dask_df = dask_df.compute()
    assert isinstance(dask_df, pd.DataFrame)

    assert (df.columns == dask_df.columns).all()
    for col in df.columns:
        assert np.isclose(df[col], dask_df[col], equal_nan=True).all()

    # indices of the output DataFrame matches the unique values in `zones`
    idx = df['zone']
    assert np.isclose(idx, unique_values).all()

    num_cols = len(df.columns)
    # 8 columns: 1 for zone id, and 7 statistics in default settings
    assert num_cols == 8

    assert zone_means == df['mean'].tolist()
    assert zone_maxes == df['max'].tolist()
    assert zone_mins == df['min'].tolist()
    assert zone_sums == df['sum'].tolist()
    assert zone_stds == df['std'].tolist()
    assert zone_vars == df['var'].tolist()
    assert zone_counts == df['count'].tolist()

    # custom stats
    def cal_sum(values):
        return values.sum()

    def cal_double_sum(values):
        return values.sum() * 2

    zone_sums = [
        cal_sum(zone_vals_0),
        cal_sum(zone_vals_1),
        cal_sum(zone_vals_2),
        cal_sum(zone_vals_3)
    ]
    zone_double_sums = [
        cal_double_sum(zone_vals_0),
        cal_double_sum(zone_vals_1),
        cal_double_sum(zone_vals_2),
        cal_double_sum(zone_vals_3)
    ]

    custom_stats = {'sum': cal_sum, 'double sum': cal_double_sum}
    df = stats(zones=zones, values=values, stats_funcs=custom_stats)

    assert isinstance(df, pd.DataFrame)
    # indices of the output DataFrame matches the unique values in `zones`
    idx = df['zone']
    assert np.isclose(idx, unique_values).all()
    num_cols = len(df.columns)
    # 3 columns: 1 zone, 2 statistics
    assert num_cols == 3
    assert zone_sums == df['sum'].tolist()
    assert zone_double_sums == df['double sum'].tolist()


# TODO: get this test passing
def _test_stats_invalid_custom_stat():
    zones, values = stats_create_zones_values()

    def cal_sum(values):
        return values.sum()

    custom_stats = {'sum': cal_sum}

    # custom stat only takes 1 argument. Thus, raise error
    with pytest.raises(Exception) as e_info:  # noqa
        stats(zones=zones, values=values, stats_funcs=custom_stats)


def test_stats_invalid_stat_input():
    zones, values = stats_create_zones_values()

    # invalid stats
    custom_stats = ['some_stat']
    with pytest.raises(Exception) as e_info:  # noqa
        stats(zones=zones, values=values, stats_funcs=custom_stats)

    # invalid values:
    zones = xr.DataArray(np.array([1, 2, 0], dtype=np.int))
    values = xr.DataArray(np.array(['apples', 'foobar', 'cowboy']))
    with pytest.raises(Exception) as e_info:  # noqa
        stats(zones=zones, values=values)

    # mismatch shape between zones and values:
    zones = xr.DataArray(np.array([1, 2, 0]))
    values = xr.DataArray(np.array([1, 2, 0, np.nan]))
    with pytest.raises(Exception) as e_info:  # noqa
        stats(zones=zones, values=values)


def test_crosstab_invalid_input():
    # invalid zones dims (must be 2d)
    zones = xr.DataArray(np.array([1, 2, 0]))
    values = xr.DataArray(np.array([[[1, 2, 0.5]]]),
                          dims=['lat', 'lon', 'race'])
    values['race'] = ['cat1', 'cat2', 'cat3']
    with pytest.raises(Exception) as e_info:
        crosstab(zones_agg=zones, values_agg=values)

    # invalid values
    zones = xr.DataArray(np.array([[1, 2, 0]], dtype=np.int))
    # values must be either int or float
    values = xr.DataArray(np.array([[['apples', 'foobar', 'cowboy']]]),
                          dims=['lat', 'lon', 'race'])
    values['race'] = ['cat1', 'cat2', 'cat3']
    with pytest.raises(Exception) as e_info:  # noqa
        crosstab(zones_agg=zones, values_agg=values)

    # mismatch shape zones and values
    zones = xr.DataArray(np.array([[1, 2]]))
    values = xr.DataArray(np.array([[[1, 2, np.nan]]]),
                          dims=['lat', 'lon', 'race'])
    values['race'] = ['cat1', 'cat2', 'cat3']
    with pytest.raises(Exception) as e_info:  # noqa
        crosstab(zones_agg=zones, values_agg=values)

    # invalid layer
    zones = xr.DataArray(np.array([[1, 2]]))
    values = xr.DataArray(np.array([[[1, 2, np.nan]]]),
                          dims=['lat', 'lon', 'race'])
    values['race'] = ['cat1', 'cat2', 'cat3']
    # this layer does not exist in values agg
    layer = 'cat'
    with pytest.raises(Exception) as e_info:  # noqa
        crosstab(zones_agg=zones, values_agg=values, layer=layer)


def test_crosstab_no_values():
    # create valid `values_agg` of 0s
    values_agg = xr.DataArray(np.zeros(24).reshape(2, 3, 4),
                              dims=['lat', 'lon', 'race'])
    values_agg['race'] = ['cat1', 'cat2', 'cat3', 'cat4']
    layer = -1

    # create a valid `zones_agg` with compatiable shape
    zones_arr = np.arange(6, dtype=np.int).reshape(2, 3)
    zones_agg = xr.DataArray(zones_arr)

    df = crosstab(zones_agg, values_agg, layer, nodata_values=0)

    num_cats = len(values_agg.dims[-1])
    # number of columns = number of categories + 1
    assert len(df.columns) == num_cats + 1

    zone_idx = np.unique(zones_arr)
    num_zones = len(zone_idx)
    # number of rows = number of zones
    assert len(df.index) == num_zones

    # values_agg are all 0s, so all 0 over categories
    for col in df.columns:
        if col != 'zone':
            assert np.isclose(df[col].unique(), [0])


def test_crosstab_3d():
    # create valid `values_agg` of np.nan and np.inf
    values_agg = xr.DataArray(np.ones(4*5*6).reshape(5, 6, 4),
                              dims=['lat', 'lon', 'race'])
    values_agg['race'] = ['cat1', 'cat2', 'cat3', 'cat4']
    layer = -1

    # create a valid `zones_agg` with compatiable shape
    zones_arr = np.arange(5*6, dtype=np.int).reshape(5, 6)
    zones_agg = xr.DataArray(zones_arr)

    # numpy case
    df = crosstab(zones_agg, values_agg, layer)
    assert isinstance(df, pd.DataFrame)

    # dask case
    values_agg_dask = xr.DataArray(
        da.from_array(values_agg.data, chunks=(3, 3, 1)),
        dims=['lat', 'lon', 'race']
    )
    values_agg_dask['race'] = ['cat1', 'cat2', 'cat3', 'cat4']
    zones_agg_dask = xr.DataArray(da.from_array(zones_agg.data, chunks=(3, 3)))
    dask_df = crosstab(zones_agg_dask, values_agg_dask, layer)
    assert isinstance(dask_df, dd.DataFrame)

    dask_df = dask_df.compute()
    assert isinstance(dask_df, pd.DataFrame)

    assert (df.columns == dask_df.columns).all()
    for col in df.columns:
        assert np.isclose(df[col], dask_df[col], equal_nan=True).all()

    num_cats = len(values_agg.dims[-1])
    # number of columns = number of categories
    assert len(df.columns) == num_cats + 1

    zone_idx = np.unique(zones_arr)
    num_zones = len(zone_idx)
    # number of rows = number of zones
    assert len(df.index) == num_zones

    num_nans = df.isnull().sum().sum()
    # no NaN
    assert num_nans == 0

    # values_agg are all 1s
    for col in df.columns:
        if col != 'zone':
            assert len(df[col].unique()) == 1


def test_crosstab_2d():
    values_val = np.asarray([[0, 0, 10, 20],
                             [0, 0, 0, 10],
                             [0, 30, 20, 50],
                             [10, 30, 40, 40],
                             [10, 10, 50, 0]])
    values_agg = xr.DataArray(values_val)
    values_agg_dask = xr.DataArray(da.from_array(values_val, chunks=(3, 3)))

    zones_val = np.asarray([[1, 1, 6, 6],
                            [1, 1, 6, 6],
                            [3, 5, 6, 6],
                            [3, 5, 7, 7],
                            [3, 7, 7, 0]])
    zones_agg = xr.DataArray(zones_val)
    zones_agg_dask = xr.DataArray(da.from_array(zones_val, chunks=(3, 3)))

    df = crosstab(zones_agg, values_agg)
    assert isinstance(df, pd.DataFrame)

    dask_df = crosstab(zones_agg_dask, values_agg_dask)
    assert isinstance(dask_df, dd.DataFrame)

    dask_df = dask_df.compute()
    assert isinstance(dask_df, pd.DataFrame)

    assert (df.columns == dask_df.columns).all()
    for col in df.columns:
        assert np.isclose(df[col], dask_df[col], equal_nan=True).all()

    num_cats = 6  # 0, 10, 20, 30, 40, 50
    # number of columns = number of categories + 1 (zone column)
    assert len(df.columns) == num_cats + 1

    zone_idx = np.unique(zones_agg.data)
    num_zones = len(zone_idx)
    # number of rows = number of zones
    assert len(df.index) == num_zones


def test_apply_invalid_input():
    def func(x):
        return 0

    # invalid dims (must be 2d)
    zones = xr.DataArray(np.array([1, 2, 0]))
    values = xr.DataArray(np.array([[[1, 2, 0.5]]]))
    with pytest.raises(Exception) as e_info:
        apply(zones, values, func)

    # invalid zones data dtype (must be int)
    zones = xr.DataArray(np.array([[1, 2, 0.5]]))
    values = xr.DataArray(np.array([[[1, 2, 0.5]]]))
    with pytest.raises(Exception) as e_info:
        apply(zones, values, func)

    # invalid values data dtype (must be int or float)
    values = xr.DataArray(np.array([['apples', 'foobar', 'cowboy']]))
    zones = xr.DataArray(np.array([[1, 2, 0]]))
    with pytest.raises(Exception) as e_info:
        apply(zones, values, func)

    # invalid values dim (must be 2d or 3d)
    values = xr.DataArray(np.array([1, 2, 0.5]))
    zones = xr.DataArray(np.array([[1, 2, 0]]))
    with pytest.raises(Exception) as e_info:
        apply(zones, values, func)

    zones = xr.DataArray(np.array([[1, 2, 0], [1, 2, 3]]))
    values = xr.DataArray(np.array([[1, 2, 0.5]]))
    # mis-match zones.shape and values.shape
    with pytest.raises(Exception) as e_info:  # noqa
        apply(zones, values, func)


def test_apply():

    def func(x):
        return 0

    zones_val = np.zeros((3, 3), dtype=np.int)
    # define some zones
    zones_val[1] = 1
    zones_val[2] = 2
    zones = xr.DataArray(zones_val)

    values_val = np.array([[0, 1, 2],
                           [3, 4, 5],
                           [6, 7, np.nan]])
    values = xr.DataArray(values_val)

    values_copy = values.copy()
    apply(zones, values, func, nodata=2)

    # agg.shape remains the same
    assert values.shape == values_copy.shape

    values_val = values.values
    # values within zones are all 0s
    assert (values_val[0] == [0, 0, 0]).all()
    assert (values_val[1] == [0, 0, 0]).all()
    # values outside zones remain
    assert np.isclose(
        values_val[2], values_copy.values[2], equal_nan=True
    ).all()


def test_suggest_zonal_canvas():
    # crs: Geographic
    x_range = (0, 20)
    y_range = (0, 10)
    smallest_area = 2
    min_pixels = 2
    height, width = suggest_zonal_canvas(x_range=x_range, y_range=y_range,
                                         smallest_area=smallest_area,
                                         crs='Geographic',
                                         min_pixels=min_pixels)
    assert height == 10
    assert width == 20

    # crs: Mercator
    x_range = (-1e6, 1e6)
    y_range = (0, 1e6)
    smallest_area = 2e9
    min_pixels = 20
    height, width = suggest_zonal_canvas(x_range=x_range, y_range=y_range,
                                         smallest_area=smallest_area,
                                         crs='Mercator',
                                         min_pixels=min_pixels)
    assert height == 100
    assert width == 200


def create_test_arr(arr):
    n, m = arr.shape
    raster = xr.DataArray(arr, dims=['y', 'x'])
    raster['y'] = np.linspace(0, n, n)
    raster['x'] = np.linspace(0, m, m)
    return raster


def test_regions_four_pixel_connectivity_int():
    arr = np.array([[0, 0, 0, 0],
                    [0, 4, 0, 0],
                    [1, 4, 4, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0]], dtype=np.int64)
    raster = create_test_arr(arr)
    raster_regions = regions(raster, neighborhood=4)
    assert len(np.unique(raster_regions.data)) == 3
    assert raster.shape == raster_regions.shape


def test_regions_four_pixel_connectivity_float():
    arr = np.array([[0, 0, 0, np.nan],
                    [0, 4, 0, 0],
                    [1, 4, 4, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0]], dtype=np.float64)
    raster = create_test_arr(arr)
    raster_regions = regions(raster, neighborhood=4)
    assert len(np.unique(raster_regions.data)) == 4
    assert raster.shape == raster_regions.shape


def test_regions_eight_pixel_connectivity_int():
    arr = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1]], dtype=np.int64)
    raster = create_test_arr(arr)
    raster_regions = regions(raster, neighborhood=8)
    assert len(np.unique(raster_regions.data)) == 2
    assert raster.shape == raster_regions.shape


def test_regions_eight_pixel_connectivity_float():
    arr = np.array([[1, 0, 0, np.nan],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1]], dtype=np.float64)
    raster = create_test_arr(arr)
    raster_regions = regions(raster, neighborhood=8)
    assert len(np.unique(raster_regions.data)) == 3
    assert raster.shape == raster_regions.shape


def test_trim():
    arr = np.array([[0, 0, 0, 0],
                    [0, 4, 0, 0],
                    [0, 4, 4, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0]], dtype=np.int64)
    raster = create_test_arr(arr)
    trimmed_raster = trim(raster, values=(0,))
    assert trimmed_raster.shape == (3, 2)

    trimmed_arr = np.array([[4, 0],
                            [4, 4],
                            [1, 1]], dtype=np.int64)

    compare = trimmed_arr == trimmed_raster.data
    assert compare.all()


def test_trim_left_top():
    arr = np.array([[0, 0, 0, 0],
                    [0, 4, 0, 3],
                    [0, 4, 4, 3],
                    [0, 1, 1, 3],
                    [0, 1, 1, 3]], dtype=np.int64)

    raster = create_test_arr(arr)
    trimmed_raster = trim(raster, values=(0,))
    assert trimmed_raster.shape == (4, 3)

    trimmed_arr = np.array([[4, 0, 3],
                            [4, 4, 3],
                            [1, 1, 3],
                            [1, 1, 3]], dtype=np.int64)

    compare = trimmed_arr == trimmed_raster.data
    assert compare.all()


def test_trim_right_top():
    arr = np.array([[0, 0, 0, 0],
                    [4, 0, 3, 0],
                    [4, 4, 3, 0],
                    [1, 1, 3, 0],
                    [1, 1, 3, 0]], dtype=np.int64)

    raster = create_test_arr(arr)
    trimmed_raster = trim(raster, values=(0,))
    assert trimmed_raster.shape == (4, 3)

    trimmed_arr = np.array([[4, 0, 3],
                            [4, 4, 3],
                            [1, 1, 3],
                            [1, 1, 3]], dtype=np.int64)

    compare = trimmed_arr == trimmed_raster.data
    assert compare.all()


def test_trim_left_bottom():
    arr = np.array([[4, 0, 3, 0],
                    [4, 4, 3, 0],
                    [1, 1, 3, 0],
                    [1, 1, 3, 0],
                    [0, 0, 0, 0]], dtype=np.int64)

    raster = create_test_arr(arr)
    trimmed_raster = trim(raster, values=(0,))
    assert trimmed_raster.shape == (4, 3)

    trimmed_arr = np.array([[4, 0, 3],
                            [4, 4, 3],
                            [1, 1, 3],
                            [1, 1, 3]], dtype=np.int64)

    compare = trimmed_arr == trimmed_raster.data
    assert compare.all()


def test_trim_right_bottom():
    arr = np.array([[0, 4, 0, 3],
                    [0, 4, 4, 3],
                    [0, 1, 1, 3],
                    [0, 1, 1, 3],
                    [0, 0, 0, 0]], dtype=np.int64)

    raster = create_test_arr(arr)
    trimmed_raster = trim(raster, values=(0,))
    assert trimmed_raster.shape == (4, 3)

    trimmed_arr = np.array([[4, 0, 3],
                            [4, 4, 3],
                            [1, 1, 3],
                            [1, 1, 3]], dtype=np.int64)

    compare = trimmed_arr == trimmed_raster.data
    assert compare.all()


def test_crop():
    arr = np.array([[0, 4, 0, 3],
                    [0, 4, 4, 3],
                    [0, 1, 1, 3],
                    [0, 1, 1, 3],
                    [0, 0, 0, 0]], dtype=np.int64)

    raster = create_test_arr(arr)
    result = crop(raster, raster, zones_ids=(1, 3))
    assert result.shape == (4, 3)

    trimmed_arr = np.array([[4, 0, 3],
                            [4, 4, 3],
                            [1, 1, 3],
                            [1, 1, 3]], dtype=np.int64)

    compare = trimmed_arr == result.data
    assert compare.all()


def test_crop_nothing_to_crop():
    arr = np.array([[0, 4, 0, 3],
                    [0, 4, 4, 3],
                    [0, 1, 1, 3],
                    [0, 1, 1, 3],
                    [0, 0, 0, 0]], dtype=np.int64)

    raster = create_test_arr(arr)
    result = crop(raster, raster, zones_ids=(0,))
    assert result.shape == arr.shape
    compare = arr == result.data
    assert compare.all()
