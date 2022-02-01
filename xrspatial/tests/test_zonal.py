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

from xrspatial.tests.general_checks import create_test_raster


@pytest.fixture
def data_zones(backend):
    data = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                     [0, 0, 1, 1, 2, 2, 3, 3],
                     [0, 0, 1, 1, 2, np.nan, 3, 3]])
    agg = create_test_raster(data, backend)
    return agg


@pytest.fixture
def data_values_2d(backend):
    data = np.asarray([
        [0, 0, 1, 1, 2, 2, 3, np.inf],
        [0, 0, 1, 1, 2, np.nan, 3, 0],
        [np.inf, 0, 1, 1, 2, 2, 3, 3]
    ])
    agg = create_test_raster(data, backend)
    return agg


@pytest.fixture
def data_values_3d(backend):
    data = np.ones(4*3*8).reshape(3, 8, 4)
    if 'dask' in backend:
        data = da.from_array(data, chunks=(3, 4, 2))

    agg = xr.DataArray(data, dims=['lat', 'lon', 'race'])
    agg['race'] = ['cat1', 'cat2', 'cat3', 'cat4']
    return agg


@pytest.fixture
def result_default_stats():
    expected_result = {
        'zone':  [0, 1, 2, 3],
        'mean':  [0, 1, 2, 2.4],
        'max':   [0, 1, 2, 3],
        'min':   [0, 1, 2, 0],
        'sum':   [0, 6, 8, 12],
        'std':   [0, 0, 0, 1.2],
        'var':   [0, 0, 0, 1.44],
        'count': [5, 6, 4, 5]
    }
    return expected_result


@pytest.fixture
def result_zone_ids_stats():
    zone_ids = [0, 3]
    expected_result = {
        'zone':  [0, 3],
        'mean':  [0, 2.4],
        'max':   [0, 3],
        'min':   [0, 0],
        'sum':   [0, 12],
        'std':   [0, 1.2],
        'var':   [0, 1.44],
        'count': [5, 5]
    }
    return zone_ids, expected_result


def _double_sum(values):
    return values.sum() * 2


def _range(values):
    return values.max() - values.min()


@pytest.fixture
def result_custom_stats():
    zone_ids = [1, 2]
    nodata_values = 0
    expected_result = {
        'zone':       [1, 2],
        'double_sum': [12, 16],
        'range':      [0,   0],
    }
    return nodata_values, zone_ids, expected_result


@pytest.fixture
def result_count_crosstab_2d():
    zone_ids = [1, 2, 3]
    cat_ids = [0, 1, 2]
    expected_result = {
        'zone': [1, 2, 3],
        0:      [0, 0, 1],
        1:      [6, 0, 0],
        2:      [0, 4, 0],
    }
    return zone_ids, cat_ids, expected_result


@pytest.fixture
def result_percentage_crosstab_2d():
    zone_ids = [1, 2]
    cat_ids = [1, 2]
    nodata_values = 3
    expected_result = {
        'zone': [1,   2],
        1:      [100, 0],
        2:      [0,   100],
    }
    return nodata_values, zone_ids, cat_ids, expected_result


@pytest.fixture
def result_crosstab_3d():
    zone_ids = [1, 2, 3]
    layer = -1
    expected_result = {
        'zone': [1, 2, 3],
        'cat1': [6, 5, 6],
        'cat2': [6, 5, 6],
        'cat3': [6, 5, 6],
        'cat4': [6, 5, 6],
    }
    return layer, zone_ids, expected_result


@pytest.fixture
def result_nodata_values_crosstab_3d():
    zone_ids = [1, 2, 3]
    layer = -1
    nodata_values = 1
    expected_result = {
        'zone': [1, 2, 3],
        'cat1': [0, 0, 0],
        'cat2': [0, 0, 0],
        'cat3': [0, 0, 0],
        'cat4': [0, 0, 0],
    }
    return nodata_values, layer, zone_ids, expected_result


def check_results(backend, df_result, expected_results_dict):
    if 'dask' in backend:
        # dask case, compute result
        assert isinstance(df_result, dd.DataFrame)
        df_result = df_result.compute()
        assert isinstance(df_result, pd.DataFrame)

    assert len(df_result.columns) == len(expected_results_dict)
    # zone column
    assert (df_result['zone'] == expected_results_dict['zone']).all()
    # stats columns
    for col in df_result.columns[1:]:
        np.testing.assert_allclose(df_result[col], expected_results_dict[col])


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_default_stats(backend, data_zones, data_values_2d, result_default_stats):
    df_result = stats(zones=data_zones, values=data_values_2d)
    check_results(backend, df_result, result_default_stats)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_zone_ids_stats(backend, data_zones, data_values_2d, result_zone_ids_stats):
    zone_ids, expected_result = result_zone_ids_stats
    df_result = stats(zones=data_zones, values=data_values_2d, zone_ids=zone_ids)
    check_results(backend, df_result, expected_result)


@pytest.mark.parametrize("backend", ['numpy'])
def test_custom_stats(backend, data_zones, data_values_2d, result_custom_stats):
    # ---- custom stats (NumPy only) ----
    custom_stats = {
        'double_sum': _double_sum,
        'range': _range,
    }
    nodata_values, zone_ids, expected_result = result_custom_stats
    df_result = stats(
        zones=data_zones, values=data_values_2d, stats_funcs=custom_stats,
        zone_ids=zone_ids, nodata_values=nodata_values
    )
    check_results(backend, df_result, expected_result)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_count_crosstab_2d(backend, data_zones, data_values_2d, result_count_crosstab_2d):
    zone_ids, cat_ids, expected_result = result_count_crosstab_2d
    df_result = crosstab(
        zones=data_zones, values=data_values_2d, zone_ids=zone_ids, cat_ids=cat_ids,
    )
    check_results(backend, df_result, expected_result)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_percentage_crosstab_2d(backend, data_zones, data_values_2d, result_percentage_crosstab_2d):
    nodata_values, zone_ids, cat_ids, expected_result = result_percentage_crosstab_2d
    df_result = crosstab(
        zones=data_zones, values=data_values_2d, zone_ids=zone_ids, cat_ids=cat_ids,
        nodata_values=nodata_values, agg='percentage'
    )
    check_results(backend, df_result, expected_result)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_crosstab_3d(backend, data_zones, data_values_3d, result_crosstab_3d):
    layer, zone_ids, expected_result = result_crosstab_3d
    df_result = crosstab(zones=data_zones, values=data_values_3d, zone_ids=zone_ids, layer=layer)
    check_results(backend, df_result, expected_result)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_nodata_values_crosstab_3d(
    backend,
    data_zones,
    data_values_3d,
    result_nodata_values_crosstab_3d
):
    nodata_values, layer, zone_ids, expected_result = result_nodata_values_crosstab_3d
    df_result = crosstab(
        zones=data_zones, values=data_values_3d, zone_ids=zone_ids,
        layer=layer, nodata_values=nodata_values
    )
    check_results(backend, df_result, expected_result)


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
