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


def create_zones_values(backend):
    zones_val = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                          [0, 0, 1, 1, 2, 2, 3, 3],
                          [0, 0, 1, 1, 2, np.nan, 3, 3]])
    zones = xr.DataArray(zones_val)

    values_val_2d = np.asarray([
        [0, 0, 1, 1, 2, 2, 3, np.inf],
        [0, 0, 1, 1, 2, np.nan, 3, 0],
        [np.inf, 0, 1, 1, 2, 2, 3, 3]
    ])
    values_2d = xr.DataArray(values_val_2d)

    values_val_3d = np.ones(4 * 3 * 6).reshape(3, 6, 4)
    values_3d = xr.DataArray(
        values_val_3d,
        dims=['lat', 'lon', 'race']
    )
    values_3d['race'] = ['cat1', 'cat2', 'cat3', 'cat4']

    if 'dask' in backend:
        zones.data = da.from_array(zones.data, chunks=(3, 3))
        values_2d.data = da.from_array(values_2d.data, chunks=(3, 3))
        values_3d.data = da.from_array(values_3d.data, chunks=(3, 3, 1))

    return zones, values_2d, values_3d


def test_stats():
    # expected results
    default_stats_results = {
        'zone':  [0, 1, 2, 3],
        'mean':  [0, 1, 2, 2.4],
        'max':   [0, 1, 2, 3],
        'min':   [0, 1, 2, 0],
        'sum':   [0, 6, 8, 12],
        'std':   [0, 0, 0, 1.2],
        'var':   [0, 0, 0, 1.44],
        'count': [5, 6, 4, 5]
    }

    # numpy case
    zones_np, values_np, _ = create_zones_values(backend='numpy')
    # default stats_funcs
    df_np = stats(zones=zones_np, values=values_np)
    assert isinstance(df_np, pd.DataFrame)
    assert len(df_np.columns) == len(default_stats_results)
    for col in df_np.columns:
        assert np.isclose(
            df_np[col], default_stats_results[col], equal_nan=True
        ).all()

    # dask case
    zones_da, values_da, _ = create_zones_values(backend='dask')
    df_da = stats(zones=zones_da, values=values_da)
    assert isinstance(df_da, dd.DataFrame)
    df_da = df_da.compute()
    assert isinstance(df_da, pd.DataFrame)
    assert (df_da.columns == df_np.columns).all()
    for col in df_da.columns:
        assert np.isclose(df_da[col], df_np[col], equal_nan=True).all()

    # ---- custom stats ----
    # expected results
    custom_stats_results = {
        'zone':       [1,   2,  3],
        'double_sum': [12, 16, 24],
        'range':      [0,   0,  0],
    }

    def _double_sum(values):
        return values.sum() * 2

    def _range(values):
        return values.max() - values.min()

    custom_stats = {
        'double_sum': _double_sum,
        'range': _range,
    }

    # numpy case
    df_np = stats(
        zones=zones_np, values=values_np, stats_funcs=custom_stats,
        nodata_zones=0, nodata_values=0
    )
    assert isinstance(df_np, pd.DataFrame)
    assert len(df_np.columns) == len(custom_stats_results)
    for col in df_np.columns:
        assert np.isclose(
            df_np[col], custom_stats_results[col], equal_nan=True
        ).all()

    # dask case
    df_da = stats(
        zones=zones_da, values=values_da, stats_funcs=custom_stats,
        nodata_zones=0, nodata_values=0
    )
    assert isinstance(df_da, dd.DataFrame)
    df_da = df_da.compute()
    assert isinstance(df_da, pd.DataFrame)
    assert (df_da.columns == df_np.columns).all()
    for col in df_da.columns:
        assert np.isclose(df_da[col], df_np[col], equal_nan=True).all()


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
