import numpy as np
import pytest
import xarray as xr

from xrspatial.local import (cell_stats, combine, equal_frequency, greater_frequency,
                             highest_position, lesser_frequency, lowest_position, popularity, rank)

arr = xr.DataArray([[2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2]], name='arr')

arr1 = xr.DataArray([[np.nan, 4, 2, 0],
                     [2, 3, np.nan, 1],
                     [5, 1, 2, 0],
                     [1, 3, 2, np.nan]], name='arr1')

arr2 = xr.DataArray([[3, 1, 1, 2],
                     [4, 1, 2, 5],
                     [0, 0, 0, 0],
                     [np.nan, 1, 1, 1]], name='arr2')

arr3 = xr.DataArray([[3, 3, 2, 0],
                     [4, 1, 3, 1],
                     [6, 1, 2, 2],
                     [0, 0, 1, 1]], name='arr3')

raster_ds = xr.merge([arr, arr1, arr2, arr3])


def test_cell_stats_all_data_vars():
    result_arr = cell_stats(raster_ds[['arr1', 'arr2', 'arr3']])
    expected_arr = xr.DataArray([[np.nan,  8,  5,  2],
                                 [10,  5, np.nan,  7],
                                 [11,  2,  4,  2],
                                 [np.nan,  4,  4, np.nan]])

    assert result_arr.equals(expected_arr)


def test_cell_stats_some_data_vars():
    result_arr = cell_stats(
        raster_ds[['arr1', 'arr2', 'arr3']],
        ['arr1', 'arr3'],
    )
    expected_arr = xr.DataArray([[np.nan,  7,  4,  0],
                                 [6,  4, np.nan,  2],
                                 [11,  2,  4,  2],
                                 [1,  3,  3, np.nan]])

    assert result_arr.equals(expected_arr)


def test_cell_stats_max():
    result_arr = cell_stats(raster_ds[['arr1', 'arr2', 'arr3']], func='max')
    expected_arr = xr.DataArray([[np.nan,  4,  2,  2],
                                 [4,  3, np.nan,  5],
                                 [6,  1,  2,  2],
                                 [np.nan,  3,  2, np.nan]])

    assert result_arr.equals(expected_arr)


def test_cell_stats_mean():
    result_arr = cell_stats(raster_ds[['arr1', 'arr2', 'arr3']], func='mean')
    expected_arr = xr.DataArray([[np.nan, 2.6666666666666665, 1.6666666666666667, 0.6666666666666666], # noqa
                                 [3.3333333333333335, 1.6666666666666667, np.nan, 2.3333333333333335],  # noqa
                                 [3.6666666666666665, 0.6666666666666666, 1.3333333333333333, 0.6666666666666666],  # noqa
                                 [np.nan, 1.3333333333333333, 1.3333333333333333, np.nan]])  # noqa

    assert result_arr.equals(expected_arr)


def test_cell_stats_median():
    result_arr = cell_stats(raster_ds[['arr1', 'arr2', 'arr3']], func='median')
    expected_arr = xr.DataArray([[np.nan,  3,  2,  0],
                                 [4,  1, np.nan,  1],
                                 [5,  1,  2,  0],
                                 [np.nan,  1,  1, np.nan]])

    assert result_arr.equals(expected_arr)


def test_cell_stats_min():
    result_arr = cell_stats(raster_ds[['arr1', 'arr2', 'arr3']], func='min')
    expected_arr = xr.DataArray([[np.nan,  1,  1,  0],
                                 [2,  1, np.nan,  1],
                                 [0,  0,  0,  0],
                                 [np.nan,  0,  1, np.nan]])

    assert result_arr.equals(expected_arr)


def test_cell_stats_std():
    result_arr = cell_stats(raster_ds[['arr1', 'arr2', 'arr3']], func='std')
    expected_arr = xr.DataArray([[np.nan, 1.247219128924647, 0.4714045207910317, 0.9428090415820634], # noqa
                                 [0.9428090415820634, 0.9428090415820634, np.nan, 1.8856180831641267],  # noqa
                                 [2.6246692913372702, 0.4714045207910317, 0.9428090415820634, 0.9428090415820634],  # noqa
                                 [np.nan, 1.247219128924647, 0.4714045207910317, np.nan]])  # noqa

    assert result_arr.equals(expected_arr)


def test_cell_stats_wrong_func():
    with pytest.raises(ValueError):
        cell_stats(raster_ds[['arr1', 'arr2', 'arr3']], func='med')


def test_cell_stats_raster_type_error():
    with pytest.raises(TypeError):
        cell_stats(arr1)


def test_cell_stats_data_vars_param_type_error():
    with pytest.raises(TypeError):
        cell_stats(raster_ds[['arr1', 'arr2', 'arr3']], data_vars='arr1')


def test_cell_stats_data_vars_elem_type_error():
    with pytest.raises(TypeError):
        cell_stats(raster_ds[['arr1', 'arr2', 'arr3']], data_vars=[0])


def test_cell_stats_wrong_var_name():
    with pytest.raises(ValueError):
        cell_stats(
            raster_ds[['arr1', 'arr2', 'arr3']],
            data_vars=['arr1', 'arr9'],
        )


def test_combine_all_data_vars():
    result_arr = combine(raster_ds[['arr1', 'arr2', 'arr3']])
    expected_arr = xr.DataArray([[np.nan,  1,  2,  3],
                                 [4,  5, np.nan,  6],
                                 [7,  8,  9, 10],
                                 [np.nan, 11, 12, np.nan]])

    assert result_arr.equals(expected_arr)


def test_combine_some_data_vars():
    result_arr = combine(raster_ds[['arr1', 'arr2', 'arr3']], ['arr1', 'arr3'])
    expected_arr = xr.DataArray([[np.nan,  1,  2,  3],
                                 [4,  5, np.nan,  6],
                                 [7,  6,  2,  8],
                                 [9, 10, 11, np.nan]])

    assert result_arr.equals(expected_arr)


def test_combine_raster_type_error():
    with pytest.raises(TypeError):
        combine(arr1)


def test_combine_data_vars_param_type_error():
    with pytest.raises(TypeError):
        combine(raster_ds[['arr1', 'arr2', 'arr3']], data_vars='arr1')


def test_combine_data_vars_elem_type_error():
    with pytest.raises(TypeError):
        combine(raster_ds[['arr1', 'arr2', 'arr3']], data_vars=[0])


def test_combine_wrong_var_name():
    with pytest.raises(ValueError):
        combine(
            raster_ds[['arr1', 'arr2', 'arr3']],
            data_vars=['arr1', 'arr9'],
        )


def test_lesser_frequency_all_data_vars():
    expected_arr = xr.DataArray([[np.nan,  1,  1,  2],
                                 [0,  2, np.nan,  2],
                                 [1,  3,  1,  2],
                                 [np.nan,  2,  2, np.nan]])
    result_arr = lesser_frequency(raster_ds, 'arr')

    assert result_arr.equals(expected_arr)


def test_lesser_frequency_some_data_vars():
    expected_arr = xr.DataArray([[np.nan,  1,  1,  1],
                                 [0,  1, np.nan,  1],
                                 [1,  2,  1,  2],
                                 [np.nan,  1,  1, np.nan]])
    result_arr = lesser_frequency(raster_ds, 'arr', ['arr1', 'arr2'])

    assert result_arr.equals(expected_arr)


def test_lesser_frequency_raster_type_error():
    with pytest.raises(TypeError):
        lesser_frequency(arr1, 'arr1')


def test_lesser_frequency_data_vars_param_type_error():
    with pytest.raises(TypeError):
        lesser_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars='arr2',
        )


def test_lesser_frequency_dim_ref_param_type_error():
    with pytest.raises(TypeError):
        lesser_frequency(raster_ds[['arr1', 'arr2', 'arr3']], ['arr1'])


def test_lesser_frequency_data_vars_elem_type_error():
    with pytest.raises(TypeError):
        lesser_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=[0],
        )


def test_lesser_frequency_wrong_var_name():
    with pytest.raises(ValueError):
        lesser_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=['arr2', 'arr9'],
        )


def test_lesser_frequency_data_vars_contain_ref_error():
    with pytest.raises(ValueError):
        lesser_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=['arr1', 'arr2'],
        )


def test_lesser_frequency_all_data_vars_not_contain_ref_error():
    with pytest.raises(ValueError):
        lesser_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr9',
            data_vars=['arr1', 'arr2'],
        )


def test_equal_frequency_all_data_vars():
    expected_arr = xr.DataArray([[np.nan,  0,  2,  1],
                                 [1,  0, np.nan,  0],
                                 [0,  0,  2,  1],
                                 [np.nan,  0,  1, np.nan]])
    result_arr = equal_frequency(raster_ds, 'arr')

    assert result_arr.equals(expected_arr)


def test_equal_frequency_some_data_vars():
    expected_arr = xr.DataArray([[np.nan,  0,  1,  1],
                                 [1,  0, np.nan,  0],
                                 [0,  0,  1,  0],
                                 [np.nan,  0,  1, np.nan]])
    result_arr = equal_frequency(raster_ds, 'arr', ['arr1', 'arr2'])

    assert result_arr.equals(expected_arr)


def test_equal_frequency_raster_type_error():
    with pytest.raises(TypeError):
        equal_frequency(arr1, 'arr1')


def test_equal_frequency_data_vars_param_type_error():
    with pytest.raises(TypeError):
        equal_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars='arr2',
        )


def test_equal_frequency_dim_ref_param_type_error():
    with pytest.raises(TypeError):
        equal_frequency(raster_ds[['arr1', 'arr2', 'arr3']], ['arr1'])


def test_equal_frequency_data_vars_elem_type_error():
    with pytest.raises(TypeError):
        equal_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=[0],
        )


def test_equal_frequency_wrong_var_name():
    with pytest.raises(ValueError):
        equal_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=['arr2', 'arr9'],
        )


def test_equal_frequency_data_vars_contain_ref_error():
    with pytest.raises(ValueError):
        equal_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=['arr1', 'arr2'],
        )


def test_equal_frequency_all_data_vars_not_contain_ref_error():
    with pytest.raises(ValueError):
        equal_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr9',
            data_vars=['arr1', 'arr2'],
        )


def test_greater_frequency_all_data_vars():
    expected_arr = xr.DataArray([[np.nan,  2,  0,  0],
                                 [2,  1, np.nan,  1],
                                 [2,  0,  0,  0],
                                 [np.nan,  1,  0, np.nan]])
    result_arr = greater_frequency(raster_ds, 'arr')

    assert result_arr.equals(expected_arr)


def test_greater_frequency_some_data_vars():
    expected_arr = xr.DataArray([[np.nan,  1,  0,  0],
                                 [1,  1, np.nan,  1],
                                 [1,  0,  0,  0],
                                 [np.nan,  1,  0, np.nan]])
    result_arr = greater_frequency(raster_ds, 'arr', ['arr1', 'arr2'])

    assert result_arr.equals(expected_arr)


def test_greater_frequency_raster_type_error():
    with pytest.raises(TypeError):
        greater_frequency(arr1, 'arr1')


def test_greater_frequency_data_vars_param_type_error():
    with pytest.raises(TypeError):
        greater_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars='arr2',
        )


def test_greater_frequency_dim_ref_param_type_error():
    with pytest.raises(TypeError):
        greater_frequency(raster_ds[['arr1', 'arr2', 'arr3']], ['arr1'])


def test_greater_frequency_data_vars_elem_type_error():
    with pytest.raises(TypeError):
        greater_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=[0],
        )


def test_greater_frequency_wrong_var_name():
    with pytest.raises(ValueError):
        greater_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=['arr2', 'arr9'],
        )


def test_greater_frequency_data_vars_contain_ref_error():
    with pytest.raises(ValueError):
        greater_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=['arr1', 'arr2'],
        )


def test_greater_frequency_all_data_vars_not_contain_ref_error():
    with pytest.raises(ValueError):
        greater_frequency(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr9',
            data_vars=['arr1', 'arr2'],
        )


def test_highest_position_all_data_vars():
    result_arr = highest_position(raster_ds[['arr1', 'arr2', 'arr3']])
    expected_arr = xr.DataArray([[np.nan,  1,  1,  2],
                                 [2,  1, np.nan,  2],
                                 [3,  1,  1,  3],
                                 [np.nan,  1,  1, np.nan]])

    assert result_arr.equals(expected_arr)


def test_highest_position_some_data_vars():
    result_arr = highest_position(
        raster_ds[['arr1', 'arr2', 'arr3']],
        ['arr1', 'arr3'],
    )
    expected_arr = xr.DataArray([[np.nan,  1,  1,  1],
                                 [2,  1, np.nan,  1],
                                 [2,  1,  1,  2],
                                 [1,  1,  1, np.nan]])

    assert result_arr.equals(expected_arr)


def test_highest_position_raster_type_error():
    with pytest.raises(TypeError):
        highest_position(arr1)


def test_highest_position_data_vars_param_type_error():
    with pytest.raises(TypeError):
        highest_position(raster_ds[['arr1', 'arr2', 'arr3']], data_vars='arr1')


def test_highest_position_data_vars_elem_type_error():
    with pytest.raises(TypeError):
        highest_position(raster_ds[['arr1', 'arr2', 'arr3']], data_vars=[0])


def test_highest_position_wrong_var_name():
    with pytest.raises(ValueError):
        highest_position(
            raster_ds[['arr1', 'arr2', 'arr3']], data_vars=['arr1', 'arr9'])


def test_lowest_position_all_data_vars():
    result_arr = lowest_position(raster_ds[['arr1', 'arr2', 'arr3']])
    expected_arr = xr.DataArray([[np.nan,  2,  2,  1],
                                 [1,  2, np.nan,  1.],
                                 [2.,  2.,  2.,  1.],
                                 [np.nan,  3.,  2., np.nan]])

    assert result_arr.equals(expected_arr)


def test_lowest_position_some_data_vars():
    result_arr = lowest_position(
        raster_ds[['arr1', 'arr2', 'arr3']],
        ['arr1', 'arr3'],
    )
    expected_arr = xr.DataArray([[np.nan,  2,  1,  1],
                                 [1,  2, np.nan,  1],
                                 [1,  1,  1,  1],
                                 [2,  2,  2, np.nan]])

    assert result_arr.equals(expected_arr)


def test_lowest_position_raster_type_error():
    with pytest.raises(TypeError):
        lowest_position(arr1)


def test_lowest_position_data_vars_param_type_error():
    with pytest.raises(TypeError):
        lowest_position(raster_ds[['arr1', 'arr2', 'arr3']], data_vars='arr1')


def test_lowest_position_data_vars_elem_type_error():
    with pytest.raises(TypeError):
        lowest_position(raster_ds[['arr1', 'arr2', 'arr3']], data_vars=[0])


def test_lowest_position_wrong_var_name():
    with pytest.raises(ValueError):
        lowest_position(
            raster_ds[['arr1', 'arr2', 'arr3']],
            data_vars=['arr1', 'arr9'],
        )


def test_popularity_all_data_vars():
    expected_arr = xr.DataArray([[np.nan, np.nan,  2,  2],
                                 [4,  3, np.nan,  5],
                                 [np.nan,  1,  2,  2],
                                 [np.nan, np.nan,  2, np.nan]])
    result_arr = popularity(raster_ds, 'arr')

    assert result_arr.equals(expected_arr)


def test_popularity_some_data_vars():
    expected_arr = xr.DataArray([[np.nan, np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan,  0],
                                 [np.nan, np.nan, np.nan, np.nan]])
    result_arr = popularity(raster_ds, 'arr', ['arr1', 'arr2'])

    assert result_arr.equals(expected_arr)


def test_popularity_raster_type_error():
    with pytest.raises(TypeError):
        popularity(arr1, 'arr1')


def test_popularity_data_vars_param_type_error():
    with pytest.raises(TypeError):
        popularity(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars='arr2',
        )


def test_popularity_dim_ref_param_type_error():
    with pytest.raises(TypeError):
        popularity(raster_ds[['arr1', 'arr2', 'arr3']], ['arr1'])


def test_popularity_data_vars_elem_type_error():
    with pytest.raises(TypeError):
        popularity(raster_ds[['arr1', 'arr2', 'arr3']], 'arr1', data_vars=[0])


def test_popularity_wrong_var_name():
    with pytest.raises(ValueError):
        popularity(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=['arr2', 'arr9'],
        )


def test_popularity_data_vars_contain_ref_error():
    with pytest.raises(ValueError):
        popularity(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=['arr1', 'arr2'],
        )


def test_popularity_all_data_vars_not_contain_ref_error():
    with pytest.raises(ValueError):
        popularity(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr9',
            data_vars=['arr1', 'arr2'],
        )


def test_rank_all_data_vars():
    expected_arr = xr.DataArray([[np.nan,  3,  2,  0],
                                 [4,  1, np.nan,  1],
                                 [5,  1,  2,  0],
                                 [np.nan,  1,  1, np.nan]])
    result_arr = rank(raster_ds, 'arr')

    assert result_arr.equals(expected_arr)


def test_rank_some_data_vars():
    expected_arr = xr.DataArray([[np.nan,  4,  2,  2],
                                 [4,  3, np.nan,  5],
                                 [5,  1,  2,  0],
                                 [np.nan,  3,  2, np.nan]])
    result_arr = rank(raster_ds, 'arr', ['arr1', 'arr2'])

    assert result_arr.equals(expected_arr)


def test_rank_raster_type_error():
    with pytest.raises(TypeError):
        rank(arr1, 'arr1')


def test_rank_data_vars_param_type_error():
    with pytest.raises(TypeError):
        rank(raster_ds[['arr1', 'arr2', 'arr3']], 'arr1', data_vars='arr2')


def test_rank_dim_ref_param_type_error():
    with pytest.raises(TypeError):
        rank(raster_ds[['arr1', 'arr2', 'arr3']], ['arr1'])


def test_rank_data_vars_elem_type_error():
    with pytest.raises(TypeError):
        rank(raster_ds[['arr1', 'arr2', 'arr3']], 'arr1', data_vars=[0])


def test_rank_wrong_var_name():
    with pytest.raises(ValueError):
        rank(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=['arr2', 'arr9'],
        )


def test_rank_data_vars_contain_ref_error():
    with pytest.raises(ValueError):
        rank(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr1',
            data_vars=['arr1', 'arr2'],
        )


def test_rank_all_data_vars_not_contain_ref_error():
    with pytest.raises(ValueError):
        rank(
            raster_ds[['arr1', 'arr2', 'arr3']],
            'arr9',
            data_vars=['arr1', 'arr2'],
        )
