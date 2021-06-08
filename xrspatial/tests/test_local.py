import numpy as np
import pytest
import xarray as xr

from xrspatial.local import (
    cell_statistics,
    combine,
    equal_frequency,
    greater_frequency,
    highest_position,
    lesser_frequency,
    lowest_position,
    popularity,
    rank,
)


arr1 = xr.DataArray([[1, 1, 0, 0],
                     [np.nan, 1, 2, 2],
                     [4, 0, 0, 2],
                     [4, 0, 1, 1]], name='arr1')

arr2 = xr.DataArray([[0, 1, 1, 0],
                     [3, 3, 1, 2],
                     [np.nan, 0, 0, 2],
                     [3, 2, 1, 0]], name='arr2')

arr3 = xr.DataArray([[np.nan, 1, 0, 0],
                     [2, 0, 3, 3],
                     [0, 0, 3, 2],
                     [1, 1, np.nan, 0]], name='arr3')

raster_ds = xr.merge([arr1, arr2, arr3])


def test_cell_statistics_all_dims():
    result = cell_statistics(raster_ds)

    expected_arr = xr.DataArray([[np.nan, 3, 1, 0],
                                 [np.nan, 4, 6, 7],
                                 [np.nan, 0, 3, 6],
                                 [8, 3, np.nan, 1]])

    assert result.equals(expected_arr)


def test_cell_statistics_some_dims():
    result = cell_statistics(raster_ds, ['arr1', 'arr3'])

    expected_arr = xr.DataArray([[np.nan, 2, 0, 0],
                                 [np.nan, 1, 5, 5],
                                 [4, 0, 3, 4],
                                 [5, 1, np.nan, 1]])

    assert result.equals(expected_arr)


def test_cell_statistics_max():
    result = cell_statistics(raster_ds, func='max')

    expected_arr = xr.DataArray([[np.nan,  1,  1,  0],
                                 [np.nan,  3,  3,  3],
                                 [np.nan,  0,  3,  2],
                                 [4,  2, np.nan,  1]])

    assert result.equals(expected_arr)


def test_cell_statistics_mean():
    result = cell_statistics(raster_ds, func='mean')

    expected_arr = xr.DataArray([[np.nan, 1., 0.3333333333333333, 0.],
                                 [np.nan, 1.3333333333333333,
                                     2., 2.3333333333333333],
                                 [np.nan, 0., 1., 2.],
                                 [2.6666666666666665, 1., np.nan, 0.3333333333333333]]) # noqa

    assert result.equals(expected_arr)


def test_cell_statistics_median():
    result = cell_statistics(raster_ds, func='median')

    expected_arr = xr.DataArray([[np.nan,  1,  0,  0],
                                 [np.nan,  1,  2,  2],
                                 [np.nan,  0,  0,  2],
                                 [3,  1, np.nan,  0]])

    assert result.equals(expected_arr)


def test_cell_statistics_min():
    result = cell_statistics(raster_ds, func='min')

    expected_arr = xr.DataArray([[np.nan,  1,  0,  0],
                                 [np.nan,  0,  1,  2],
                                 [np.nan,  0,  0,  2],
                                 [1,  0, np.nan,  0]])

    assert result.equals(expected_arr)


def test_cell_statistics_std():
    result = cell_statistics(raster_ds, func='std')

    expected_arr = xr.DataArray([[np.nan, 0., 0.4714045207910317, 0.],
                                 [np.nan, 1.247219128924647,
                                     0.816496580927726, 0.4714045207910317],
                                 [np.nan, 0., 1.4142135623730951, 0.],
                                 [1.247219128924647, 0.816496580927726, np.nan, 0.4714045207910317]]) # noqa

    assert result.equals(expected_arr)


def test_cell_statistics_wrong_func():
    with pytest.raises(ValueError):
        cell_statistics(raster_ds, func='med')


def test_cell_statistics_raster_type_error():
    with pytest.raises(TypeError):
        cell_statistics(arr1)


def test_cell_statistics_dims_param_type_error():
    with pytest.raises(TypeError):
        cell_statistics(raster_ds, dims='arr1')


def test_cell_statistics_dims_elem_type_error():
    with pytest.raises(TypeError):
        cell_statistics(raster_ds, dims=[0])


def test_cell_statistics_wrong_dim():
    with pytest.raises(ValueError):
        cell_statistics(raster_ds, dims=['arr1', 'arr9'])


def test_combine_all_dims():
    result = combine(raster_ds)

    expected_arr = xr.DataArray([[np.nan, 1, 2, 3],
                                 [np.nan, 4, 5, 6],
                                 [np.nan, 3, 7, 8],
                                 [9, 10, np.nan, 11]])

    assert result.equals(expected_arr)


def test_combine_some_dims():
    result = combine(raster_ds, ['arr1', 'arr3'])

    expected_arr = xr.DataArray([[np.nan, 1, 2, 2],
                                 [np.nan, 3, 4, 4],
                                 [5, 2, 6, 7],
                                 [8, 9, np.nan, 3]])

    assert result.equals(expected_arr)


def test_combine_raster_type_error():
    with pytest.raises(TypeError):
        combine(arr1)


def test_combine_dims_param_type_error():
    with pytest.raises(TypeError):
        combine(raster_ds, dims='arr1')


def test_combine_dims_elem_type_error():
    with pytest.raises(TypeError):
        combine(raster_ds, dims=[0])


def test_combine_wrong_dim():
    with pytest.raises(ValueError):
        combine(raster_ds, dims=['arr1', 'arr9'])


def test_equal_frequency_all_dims():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]], name='arr')

    input_arr = xr.merge([comp_arr, raster_ds])
    expected_arr = xr.DataArray([[np.nan, 0, 0, 0],
                                 [np.nan, 0, 1, 2],
                                 [np.nan, 0, 0, 3],
                                 [0, 1, np.nan, 0]])

    result = equal_frequency(input_arr, 'arr')

    assert result.equals(expected_arr)


def test_equal_frequency_some_dims():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]], name='arr')

    input_arr = xr.merge([comp_arr, raster_ds])
    expected_arr = xr.DataArray([[0, 0, 0, 0],
                                 [np.nan, 0, 1, 2],
                                 [np.nan, 0, 0, 2],
                                 [0, 1, 0, 0]])

    result = equal_frequency(input_arr, 'arr', ['arr1', 'arr2'])

    assert result.equals(expected_arr)


def test_equal_frequency_raster_type_error():
    with pytest.raises(TypeError):
        equal_frequency(arr1, 'arr1')


def test_equal_frequency_dims_param_type_error():
    with pytest.raises(TypeError):
        equal_frequency(raster_ds, 'arr1', dims='arr2')


def test_equal_frequency_dim_ref_param_type_error():
    with pytest.raises(TypeError):
        equal_frequency(raster_ds, ['arr1'])


def test_equal_frequency_dims_elem_type_error():
    with pytest.raises(TypeError):
        equal_frequency(raster_ds, 'arr1', dims=[0])


def test_equal_frequency_wrong_dim():
    with pytest.raises(ValueError):
        equal_frequency(raster_ds, 'arr1', dims=['arr2', 'arr9'])


def test_equal_frequency_dims_contain_ref_error():
    with pytest.raises(ValueError):
        equal_frequency(raster_ds, 'arr1', dims=['arr1', 'arr2'])


def test_equal_frequency_all_dims_not_contain_ref_error():
    with pytest.raises(ValueError):
        equal_frequency(raster_ds, 'arr9', dims=['arr1', 'arr2'])


def test_greater_frequency_all_dims():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]], name='arr')

    input_arr = xr.merge([comp_arr, raster_ds])
    expected_arr = xr.DataArray([[np.nan, 0, 0, 0],
                                 [np.nan, 1, 1, 1],
                                 [np.nan, 0, 1, 0],
                                 [2, 0, np.nan, 0]])

    result = greater_frequency(input_arr, 'arr')

    assert result.equals(expected_arr)


def test_greater_frequency_some_dims():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]], name='arr')

    input_arr = xr.merge([comp_arr, raster_ds])
    expected_arr = xr.DataArray([[0, 0, 0, 0],
                                 [np.nan, 1, 0, 0],
                                 [np.nan, 0, 0, 0],
                                 [2, 0, 0, 0]])

    result = greater_frequency(input_arr, 'arr', ['arr1', 'arr2'])

    assert result.equals(expected_arr)


def test_greater_frequency_raster_type_error():
    with pytest.raises(TypeError):
        greater_frequency(arr1, 'arr1')


def test_greater_frequency_dims_param_type_error():
    with pytest.raises(TypeError):
        greater_frequency(raster_ds, 'arr1', dims='arr2')


def test_greater_frequency_dim_ref_param_type_error():
    with pytest.raises(TypeError):
        greater_frequency(raster_ds, ['arr1'])


def test_greater_frequency_dims_elem_type_error():
    with pytest.raises(TypeError):
        greater_frequency(raster_ds, 'arr1', dims=[0])


def test_greater_frequency_wrong_dim():
    with pytest.raises(ValueError):
        greater_frequency(raster_ds, 'arr1', dims=['arr2', 'arr9'])


def test_greater_frequency_dims_contain_ref_error():
    with pytest.raises(ValueError):
        greater_frequency(raster_ds, 'arr1', dims=['arr1', 'arr2'])


def test_greater_frequency_all_dims_not_contain_ref_error():
    with pytest.raises(ValueError):
        greater_frequency(raster_ds, 'arr9', dims=['arr1', 'arr2'])


def test_highest_position_all_dims():
    result = highest_position(raster_ds)

    expected_arr = xr.DataArray([[np.nan, 1, 2, 1],
                                 [np.nan, 2, 3, 3],
                                 [np.nan, 1, 3, 1],
                                 [1, 2, np.nan, 1]])

    assert result.equals(expected_arr)


def test_highest_position_some_dims():
    result = highest_position(raster_ds, ['arr1', 'arr3'])

    expected_arr = xr.DataArray([[np.nan, 1, 1, 1],
                                 [np.nan, 1, 2, 2],
                                 [1, 1, 2, 1],
                                 [1, 2, np.nan, 1]])

    assert result.equals(expected_arr)


def test_highest_position_raster_type_error():
    with pytest.raises(TypeError):
        highest_position(arr1)


def test_highest_position_dims_param_type_error():
    with pytest.raises(TypeError):
        highest_position(raster_ds, dims='arr1')


def test_highest_position_dims_elem_type_error():
    with pytest.raises(TypeError):
        highest_position(raster_ds, dims=[0])


def test_highest_position_wrong_dim():
    with pytest.raises(ValueError):
        highest_position(raster_ds, dims=['arr1', 'arr9'])


def test_lesser_frequency_all_dims():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]], name='arr')

    input_arr = xr.merge([comp_arr, raster_ds])
    expected_arr = xr.DataArray([[np.nan, 3, 3, 3],
                                 [np.nan, 2, 1, 0],
                                 [np.nan, 3, 2, 0],
                                 [1, 2, np.nan, 3]])

    result = lesser_frequency(input_arr, 'arr')

    assert result.equals(expected_arr)


def test_lesser_frequency_some_dims():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]], name='arr')

    input_arr = xr.merge([comp_arr, raster_ds])
    expected_arr = xr.DataArray([[2, 2, 2, 2],
                                 [np.nan, 1, 1, 0],
                                 [np.nan, 2, 2, 0],
                                 [0, 1, 2, 2]])

    result = lesser_frequency(input_arr, 'arr', ['arr1', 'arr2'])

    assert result.equals(expected_arr)


def test_lesser_frequency_raster_type_error():
    with pytest.raises(TypeError):
        lesser_frequency(arr1, 'arr1')


def test_lesser_frequency_dims_param_type_error():
    with pytest.raises(TypeError):
        lesser_frequency(raster_ds, 'arr1', dims='arr2')


def test_lesser_frequency_dim_ref_param_type_error():
    with pytest.raises(TypeError):
        lesser_frequency(raster_ds, ['arr1'])


def test_lesser_frequency_dims_elem_type_error():
    with pytest.raises(TypeError):
        lesser_frequency(raster_ds, 'arr1', dims=[0])


def test_lesser_frequency_wrong_dim():
    with pytest.raises(ValueError):
        lesser_frequency(raster_ds, 'arr1', dims=['arr2', 'arr9'])


def test_lesser_frequency_dims_contain_ref_error():
    with pytest.raises(ValueError):
        lesser_frequency(raster_ds, 'arr1', dims=['arr1', 'arr2'])


def test_lesser_frequency_all_dims_not_contain_ref_error():
    with pytest.raises(ValueError):
        lesser_frequency(raster_ds, 'arr9', dims=['arr1', 'arr2'])


def test_lowest_position_all_dims():
    result = lowest_position(raster_ds)

    expected_arr = xr.DataArray([[np.nan, 1, 1, 1],
                                 [np.nan, 3, 2, 1],
                                 [np.nan, 1, 1, 1],
                                 [3, 1, np.nan, 2]])

    assert result.equals(expected_arr)


def test_lowest_position_some_dims():
    result = lowest_position(raster_ds, ['arr1', 'arr3'])

    expected_arr = xr.DataArray([[np.nan, 1, 1, 1],
                                 [np.nan, 2, 1, 1],
                                 [2, 1, 1, 1],
                                 [2, 1, np.nan, 2]])

    assert result.equals(expected_arr)


def test_lowest_position_raster_type_error():
    with pytest.raises(TypeError):
        lowest_position(arr1)


def test_lowest_position_dims_param_type_error():
    with pytest.raises(TypeError):
        lowest_position(raster_ds, dims='arr1')


def test_lowest_position_dims_elem_type_error():
    with pytest.raises(TypeError):
        lowest_position(raster_ds, dims=[0])


def test_lowest_position_wrong_dim():
    with pytest.raises(ValueError):
        lowest_position(raster_ds, dims=['arr1', 'arr9'])


def test_popularity_all_dims():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]], name='arr')

    input_arr = xr.merge([comp_arr, raster_ds])
    expected_arr = xr.DataArray([[np.nan, 1, 1, 0],
                                 [np.nan, np.nan, np.nan, 3],
                                 [np.nan, 0, 3, 2],
                                 [np.nan, np.nan, np.nan, 1]])

    result = popularity(input_arr, 'arr')

    assert result.equals(expected_arr)


def test_popularity_some_dims():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]], name='arr')

    input_arr = xr.merge([comp_arr, raster_ds])
    expected_arr = xr.DataArray([[np.nan, 1, np.nan, 0],
                                 [np.nan, np.nan, np.nan, 2],
                                 [np.nan, 0, 0, 2],
                                 [np.nan, np.nan, 1, np.nan]])

    result = popularity(input_arr, 'arr', ['arr1', 'arr2'])

    assert result.equals(expected_arr)


def test_popularity_raster_type_error():
    with pytest.raises(TypeError):
        popularity(arr1, 'arr1')


def test_popularity_dims_param_type_error():
    with pytest.raises(TypeError):
        popularity(raster_ds, 'arr1', dims='arr2')


def test_popularity_dim_ref_param_type_error():
    with pytest.raises(TypeError):
        popularity(raster_ds, ['arr1'])


def test_popularity_dims_elem_type_error():
    with pytest.raises(TypeError):
        popularity(raster_ds, 'arr1', dims=[0])


def test_popularity_wrong_dim():
    with pytest.raises(ValueError):
        popularity(raster_ds, 'arr1', dims=['arr2', 'arr9'])


def test_popularity_dims_contain_ref_error():
    with pytest.raises(ValueError):
        popularity(raster_ds, 'arr1', dims=['arr1', 'arr2'])


def test_popularity_all_dims_not_contain_ref_error():
    with pytest.raises(ValueError):
        popularity(raster_ds, 'arr9', dims=['arr1', 'arr2'])


def test_rank_all_dims():
    comp_arr = xr.DataArray([[3, 3, 3, 3],
                             [3, 3, 3, 3],
                             [3, 3, 3, 3],
                             [3, 3, 3, 3]], name='arr')

    input_arr = xr.merge([comp_arr, raster_ds])
    expected_arr = xr.DataArray([[np.nan, 1, 1, 0],
                                 [np.nan, 3, 3, 3],
                                 [np.nan, 0, 3, 2],
                                 [4, 2, np.nan, 1]])

    result = rank(input_arr, 'arr')

    assert result.equals(expected_arr)


def test_rank_some_dims():
    comp_arr = xr.DataArray([[3, 3, 3, 3],
                             [3, 3, 3, 3],
                             [3, 3, 3, 3],
                             [3, 3, 3, 3]], name='arr')

    input_arr = xr.merge([comp_arr, raster_ds])
    expected_arr = xr.DataArray([[np.nan, np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan, np.nan]])

    result = rank(input_arr, 'arr', ['arr1', 'arr2'])

    assert result.equals(expected_arr)


def test_rank_raster_type_error():
    with pytest.raises(TypeError):
        rank(arr1, 'arr1')


def test_rank_dims_param_type_error():
    with pytest.raises(TypeError):
        rank(raster_ds, 'arr1', dims='arr2')


def test_rank_dim_ref_param_type_error():
    with pytest.raises(TypeError):
        rank(raster_ds, ['arr1'])


def test_rank_dims_elem_type_error():
    with pytest.raises(TypeError):
        rank(raster_ds, 'arr1', dims=[0])


def test_rank_wrong_dim():
    with pytest.raises(ValueError):
        rank(raster_ds, 'arr1', dims=['arr2', 'arr9'])


def test_rank_dims_contain_ref_error():
    with pytest.raises(ValueError):
        rank(raster_ds, 'arr1', dims=['arr1', 'arr2'])


def test_rank_all_dims_not_contain_ref_error():
    with pytest.raises(ValueError):
        rank(raster_ds, 'arr9', dims=['arr1', 'arr2'])
