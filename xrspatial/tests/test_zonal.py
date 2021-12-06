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

    values_val_3d = np.ones(4*3*8).reshape(3, 8, 4)
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


def check_results(df_np, df_da, expected_results_dict):
    # numpy case
    assert isinstance(df_np, pd.DataFrame)
    assert len(df_np.columns) == len(expected_results_dict)

    # zone column
    assert (df_np['zone'] == expected_results_dict['zone']).all()

    for col in df_np.columns[1:]:
        assert np.isclose(
            df_np[col], expected_results_dict[col], equal_nan=True
        ).all()

    if df_da is not None:
        # dask case
        assert isinstance(df_da, dd.DataFrame)
        df_da = df_da.compute()
        assert isinstance(df_da, pd.DataFrame)

        # numpy results equal dask results, ignoring their indexes
        assert np.array_equal(df_np.values, df_da.values, equal_nan=True)


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

    # dask case
    zones_da, values_da, _ = create_zones_values(backend='dask')
    df_da = stats(zones=zones_da, values=values_da)
    check_results(df_np, df_da, default_stats_results)

    # expected results
    stats_results_zone_0_3 = {
        'zone':  [0, 3],
        'mean':  [0, 2.4],
        'max':   [0, 3],
        'min':   [0, 0],
        'sum':   [0, 12],
        'std':   [0, 1.2],
        'var':   [0, 1.44],
        'count': [5, 5]
    }

    # numpy case
    df_np_zone_0_3 = stats(zones=zones_np, values=values_np, zone_ids=[0, 3])

    # dask case
    df_da_zone_0_3 = stats(zones=zones_da, values=values_da, zone_ids=[0, 3])

    check_results(df_np_zone_0_3, df_da_zone_0_3, stats_results_zone_0_3)

    # ---- custom stats (NumPy only) ----
    # expected results
    custom_stats_results = {
        'zone':       [1, 2],
        'double_sum': [12, 16],
        'range':      [0,   0],
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
        zone_ids=[1, 2], nodata_values=0
    )
    # dask case
    df_da = None
    check_results(df_np, df_da, custom_stats_results)


def test_crosstab_2d():
    # count agg, expected results
    crosstab_2d_results = {
        'zone': [1, 2, 3],
        0:      [0, 0, 1],
        1:      [6, 0, 0],
        2:      [0, 4, 0],
    }

    # numpy case
    zones_np, values_np, _ = create_zones_values(backend='numpy')
    df_np = crosstab(
        zones=zones_np, values=values_np,
        zone_ids=[1, 2, 3], cat_ids=[0, 1, 2],
    )
    # dask case
    zones_da, values_da, _ = create_zones_values(backend='dask')
    df_da = crosstab(
        zones=zones_da, values=values_da, zone_ids=[1, 2, 3], nodata_values=3
    )
    check_results(df_np, df_da, crosstab_2d_results)

    # percentage agg, expected results

    crosstab_2d_percentage_results = {
        'zone': [1,   2],
        1:      [100, 0],
        2:      [0,   100],
    }

    # numpy case
    df_np = crosstab(
        zones=zones_np, values=values_np, zone_ids=[1, 2], cat_ids=[1, 2],
        nodata_values=3, agg='percentage'
    )
    # dask case
    df_da = crosstab(
        zones=zones_da, values=values_da, zone_ids=[1, 2], cat_ids=[1, 2],
        nodata_values=3, agg='percentage'
    )
    check_results(df_np, df_da, crosstab_2d_percentage_results)


def test_crosstab_3d():
    # expected results
    crosstab_3d_results = {
        'zone': [1, 2, 3],
        'cat1': [6, 5, 6],
        'cat2': [6, 5, 6],
        'cat3': [6, 5, 6],
        'cat4': [6, 5, 6],
    }

    # numpy case
    zones_np, _, values_np = create_zones_values(backend='numpy')
    df_np = crosstab(
        zones=zones_np, values=values_np, zone_ids=[1, 2, 3], layer=-1
    )
    # dask case
    zones_da, _, values_da = create_zones_values(backend='dask')
    df_da = crosstab(
        zones=zones_da, values=values_da, zone_ids=[1, 2, 3],
        cat_ids=['cat1', 'cat2', 'cat3', 'cat4'], layer=-1
    )
    check_results(df_np, df_da, crosstab_3d_results)

    # ----- no values case ------
    crosstab_3d_novalues_results = {
        'zone': [1, 2, 3],
        'cat1': [0, 0, 0],
        'cat2': [0, 0, 0],
        'cat3': [0, 0, 0],
        'cat4': [0, 0, 0],
    }

    # numpy case
    zones_np, _, values_np = create_zones_values(backend='numpy')
    df_np = crosstab(
        zones=zones_np, values=values_np, layer=-1,
        zone_ids=[1, 2, 3], nodata_values=1
    )
    # dask case
    zones_da, _, values_da = create_zones_values(backend='dask')
    df_da = crosstab(
        zones=zones_da, values=values_da, layer=-1,
        zone_ids=[1, 2, 3], nodata_values=1
    )
    check_results(df_np, df_da, crosstab_3d_novalues_results)


def test_apply():

    def func(x):
        return 0

    zones_val = np.zeros((3, 3), dtype=int)
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
