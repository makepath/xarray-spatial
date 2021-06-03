import numpy as np
import xarray as xr

from xrspatial.local import (
    combine_arrays,
    equal_frequency,
    greater_frequency,
    highest_array,
    lesser_frequency,
    lowest_array,
    popularity,
    rank,
)


arr1 = xr.DataArray([[1, 1, 0, 0],
                     [np.nan, 1, 2, 2],
                     [4, 0, 0, 2],
                     [4, 0, 1, 1]])

arr2 = xr.DataArray([[0, 1, 1, 0],
                     [3, 3, 1, 2],
                     [np.nan, 0, 0, 2],
                     [3, 2, 1, 0]])

arr3 = xr.DataArray([[np.nan, 1, 0, 0],
                     [2, 0, 3, 3],
                     [0, 0, 3, 2],
                     [1, 1, np.nan, 0]])


def test_combine_arrays():
    result = combine_arrays(arr1, arr2, arr3)

    expected_arr = xr.DataArray([[np.nan, 1, 2, 3],
                                 [np.nan, 4, 5, 6],
                                 [np.nan, 3, 7, 8],
                                 [9, 10, np.nan, 11]])

    assert result.equals(expected_arr)


def test_equal_frequency():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]])

    result = equal_frequency(comp_arr, [arr1, arr2, arr3])

    expected_arr = xr.DataArray([[np.nan, 0, 0, 0],
                                 [np.nan, 0, 1, 2],
                                 [np.nan, 0, 0, 3],
                                 [0, 1, np.nan, 0]])

    assert result.equals(expected_arr)


def test_greater_frequency():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]])

    result = greater_frequency(comp_arr, [arr1, arr2, arr3])

    expected_arr = xr.DataArray([[np.nan, 0, 0, 0],
                                 [np.nan, 1, 1, 1],
                                 [np.nan, 0, 1, 0],
                                 [2, 0, np.nan, 0]])

    assert result.equals(expected_arr)


def test_highest_array():
    result = highest_array(arr1, arr2, arr3)

    expected_arr = xr.DataArray([[np.nan, 1, 2, 1],
                                 [np.nan, 2, 3, 3],
                                 [np.nan, 1, 3, 1],
                                 [1, 2, np.nan, 1]])

    assert result.equals(expected_arr)


def test_lesser_frequency():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]])

    result = lesser_frequency(comp_arr, [arr1, arr2, arr3])

    expected_arr = xr.DataArray([[np.nan, 3, 3, 3],
                                 [np.nan, 2, 1, 0],
                                 [np.nan, 3, 2, 0],
                                 [1, 2, np.nan, 3]])

    assert result.equals(expected_arr)


def test_lowest_array():
    result = lowest_array(arr1, arr2, arr3)

    expected_arr = xr.DataArray([[np.nan, 1, 1, 1],
                                 [np.nan, 3, 2, 1],
                                 [np.nan, 1, 1, 1],
                                 [3, 1, np.nan, 2]])

    assert result.equals(expected_arr)


def test_popularity():
    comp_arr = xr.DataArray([[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]])

    result = popularity(comp_arr, [arr1, arr2, arr3])

    expected_arr = xr.DataArray([[np.nan, 1, 1, 0],
                                 [np.nan, np.nan, np.nan, 3],
                                 [np.nan, 0, 3, 2],
                                 [np.nan, np.nan, np.nan, 1]])

    assert result.equals(expected_arr)


def test_rank():
    comp_arr = xr.DataArray([[3, 3, 3, 3],
                             [3, 3, 3, 3],
                             [3, 3, 3, 3],
                             [3, 3, 3, 3]])

    result = rank(comp_arr, [arr1, arr2, arr3])

    expected_arr = xr.DataArray([[np.nan, 1, 1, 0],
                                 [np.nan, 3, 3, 3],
                                 [np.nan, 0, 3, 2],
                                 [4, 2, np.nan, 1]])

    assert result.equals(expected_arr)
