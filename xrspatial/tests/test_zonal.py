import copy

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import dask.dataframe as dd
except ImportError:
    dd = None

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrspatial import crop, suggest_zonal_canvas, trim
from xrspatial import zonal_apply as apply
from xrspatial import zonal_crosstab as crosstab
from xrspatial import zonal_stats as stats
from xrspatial.zonal import regions

from .general_checks import (
    assert_input_data_unmodified, create_test_raster, general_output_checks, has_cuda_and_cupy,
    dask_array_available, has_dask_array, has_dask_dataframe
)


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
    if has_dask_array() and 'dask' in backend:
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
        'count': [5, 6, 4, 5],
        'majority': [0, 1, 2, 3]
    }
    return expected_result


@pytest.fixture
def result_default_stats_no_majority():
    """Expected result for dask backend which doesn't support majority."""
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
def result_default_stats_dataarray():
    expected_result = np.array(
        [[[0., 0., 1., 1., 2., 2., 2.4, 2.4],
          [0., 0., 1., 1., 2., 2., 2.4, 2.4],
          [0., 0., 1., 1., 2., np.nan, 2.4, 2.4]],

         [[0., 0., 1., 1., 2., 2., 3., 3.],
          [0., 0., 1., 1., 2., 2., 3., 3.],
          [0., 0., 1., 1., 2., np.nan, 3., 3.]],

         [[0., 0., 1., 1., 2., 2., 0., 0.],
          [0., 0., 1., 1., 2., 2., 0., 0.],
          [0., 0., 1., 1., 2., np.nan, 0., 0.]],

         [[0., 0., 6., 6., 8., 8., 12., 12.],
          [0., 0., 6., 6., 8., 8., 12., 12.],
          [0., 0., 6., 6., 8., np.nan, 12., 12.]],

         [[0., 0., 0., 0., 0., 0., 1.2, 1.2],
          [0., 0., 0., 0., 0., 0., 1.2, 1.2],
          [0., 0., 0., 0., 0., np.nan, 1.2, 1.2]],

         [[0., 0., 0., 0., 0., 0., 1.44, 1.44],
          [0., 0., 0., 0., 0., 0., 1.44, 1.44],
          [0., 0., 0., 0., 0., np.nan, 1.44, 1.44]],

         [[5., 5., 6., 6., 4., 4., 5., 5.],
          [5., 5., 6., 6., 4., 4., 5., 5.],
          [5., 5., 6., 6., 4., np.nan, 5., 5.]],

         [[0., 0., 1., 1., 2., 2., 3., 3.],
          [0., 0., 1., 1., 2., 2., 3., 3.],
          [0., 0., 1., 1., 2., np.nan, 3., 3.]]]
    )
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
        'count': [5, 5],
        'majority': [0, 3]
    }
    return zone_ids, expected_result


@pytest.fixture
def result_zone_ids_stats_no_majority():
    """Expected result for dask backend which doesn't support majority."""
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


@pytest.fixture
def result_zone_ids_stats_dataarray():
    zone_ids = [0, 3]
    expected_result = np.array(
        [[[0., 0., np.nan, np.nan, np.nan, np.nan, 2.4, 2.4],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 2.4, 2.4],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 2.4, 2.4]],

         [[0., 0., np.nan, np.nan, np.nan, np.nan, 3., 3.],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 3., 3.],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 3., 3.]],

         [[0., 0., np.nan, np.nan, np.nan, np.nan, 0., 0.],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 0., 0.],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 0., 0.]],

         [[0., 0., np.nan, np.nan, np.nan, np.nan, 12., 12.],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 12., 12.],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 12., 12.]],

         [[0., 0., np.nan, np.nan, np.nan, np.nan, 1.2, 1.2],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 1.2, 1.2],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 1.2, 1.2]],

         [[0., 0., np.nan, np.nan, np.nan, np.nan, 1.44, 1.44],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 1.44, 1.44],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 1.44, 1.44]],

         [[5., 5., np.nan, np.nan, np.nan, np.nan, 5., 5.],
          [5., 5., np.nan, np.nan, np.nan, np.nan, 5., 5.],
          [5., 5., np.nan, np.nan, np.nan, np.nan, 5., 5.]],

         [[0., 0., np.nan, np.nan, np.nan, np.nan, 3., 3.],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 3., 3.],
          [0., 0., np.nan, np.nan, np.nan, np.nan, 3., 3.]]])

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
def result_custom_stats_dataarray():
    zone_ids = [1, 2]
    nodata_values = 0
    expected_result = np.array(
        [[[np.nan, np.nan, 12., 12., 16., 16., np.nan, np.nan],
          [np.nan, np.nan, 12., 12., 16., 16., np.nan, np.nan],
          [np.nan, np.nan, 12., 12., 16., np.nan, np.nan, np.nan]],

         [[np.nan, np.nan, 0., 0., 0., 0., np.nan, np.nan],
          [np.nan, np.nan, 0., 0., 0., 0., np.nan, np.nan],
          [np.nan, np.nan, 0., 0., 0., np.nan, np.nan, np.nan]]]
    )
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
        'mean': {
            'zone': [1, 2, 3],
            'cat1': [1., 1., 1.],
            'cat2': [1., 1., 1.],
            'cat3': [1., 1., 1.],
            'cat4': [1., 1., 1.]
        },
        'max': {
            'zone': [1, 2, 3],
            'cat1': [1., 1., 1.],
            'cat2': [1., 1., 1.],
            'cat3': [1., 1., 1.],
            'cat4': [1., 1., 1.]
        },
        'min': {
            'zone': [1, 2, 3],
            'cat1': [1., 1., 1.],
            'cat2': [1., 1., 1.],
            'cat3': [1., 1., 1.],
            'cat4': [1., 1., 1.]
        },
        'sum': {
            'zone': [1, 2, 3],
            'cat1': [6., 5., 6.],
            'cat2': [6., 5., 6.],
            'cat3': [6., 5., 6.],
            'cat4': [6., 5., 6.]
        },
        'std': {
            'zone': [1, 2, 3],
            'cat1': [0., 0., 0.],
            'cat2': [0., 0., 0.],
            'cat3': [0., 0., 0.],
            'cat4': [0., 0., 0.]
        },
        'var': {
            'zone': [1, 2, 3],
            'cat1': [0., 0., 0.],
            'cat2': [0., 0., 0.],
            'cat3': [0., 0., 0.],
            'cat4': [0., 0., 0.]
        },
        'count': {
            'zone': [1, 2, 3],
            'cat1': [6, 5, 6],
            'cat2': [6, 5, 6],
            'cat3': [6, 5, 6],
            'cat4': [6, 5, 6]
        }
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


@pytest.fixture
def qgis_zonal_stats():
    qgis_result = {
        'zone': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        'mean': [748.04910278,
                 619.62845612,
                 363.29403178,
                 582.55223301,
                 356.15832265,
                 730.03720856,
                 468.15884018,
                 388.61296272,
                 706.54189046,
                 677.92201742],
        'max': [999.14184570,
                859.26989746,
                752.95483398,
                845.27789307,
                704.23699951,
                977.11694336,
                870.53448486,
                721.16333008,
                990.82781982,
                984.69262695],
        'min': [496.95635986,
                76.49687195,
                151.49211121,
                290.60409546,
                51.21858978,
                447.12411499,
                49.61272812,
                32.27882004,
                468.97912598,
                242.24084473],
        'sum': [1496.09820557,
                3717.77073669,
                1089.88209534,
                4077.86563110,
                3205.42490387,
                2920.14883423,
                2340.79420090,
                2331.67777634,
                2119.62567139,
                2033.76605225],
        'count': [2, 6, 3, 7, 9, 4, 5, 6, 3, 3]
    }
    return qgis_result


def check_results(
        backend, df_result, expected_results_dict, rtol=1e-05, atol=1e-07, equal_nan=True
):
    if has_dask_dataframe() and 'dask' in backend:
        # dask case, compute result
        assert isinstance(df_result, dd.DataFrame)
        df_result = df_result.compute()
        assert isinstance(df_result, pd.DataFrame)

    assert len(df_result.columns) == len(expected_results_dict)
    # zone column
    assert (df_result['zone'] == expected_results_dict['zone']).all()
    # stats columns
    for col in df_result.columns[1:]:
        np.testing.assert_allclose(
            df_result[col], expected_results_dict[col], rtol=rtol, atol=atol, equal_nan=equal_nan
        )


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy'])
def test_default_stats(backend, data_zones, data_values_2d, result_default_stats,
                       result_default_stats_no_majority):
    if backend == 'cupy' and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)

    df_result = stats(zones=data_zones, values=data_values_2d)
    # dask doesn't support majority stat (can't be computed block-by-block)
    expected_result = result_default_stats_no_majority if 'dask' in backend else result_default_stats
    check_results(backend, df_result, expected_result)

    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_2d, copied_data_values_2d)


@pytest.mark.parametrize("backend", ['numpy'])
def test_default_stats_dataarray(
    backend, data_zones, data_values_2d, result_default_stats_dataarray
):
    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)

    dataarray_result = stats(
        zones=data_zones, values=data_values_2d, return_type='xarray.DataArray'
    )
    general_output_checks(
        data_values_2d,
        dataarray_result,
        result_default_stats_dataarray,
        verify_dtype=False,
        verify_attrs=False,
    )
    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_2d, copied_data_values_2d)

@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy'])
def test_zone_ids_stats(backend, data_zones, data_values_2d, result_zone_ids_stats,
                        result_zone_ids_stats_no_majority):
    if backend == 'cupy' and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)

    # dask doesn't support majority stat (can't be computed block-by-block)
    if 'dask' in backend:
        zone_ids, expected_result = result_zone_ids_stats_no_majority
    else:
        zone_ids, expected_result = result_zone_ids_stats
    df_result = stats(zones=data_zones, values=data_values_2d,
                      zone_ids=zone_ids)
    check_results(backend, df_result, expected_result)
    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_2d, copied_data_values_2d)


@pytest.mark.parametrize("backend", ['numpy'])
def test_zone_ids_stats_dataarray(
    backend, data_zones, data_values_2d, result_zone_ids_stats_dataarray
):
    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)

    zone_ids, expected_result = result_zone_ids_stats_dataarray
    dataarray_result = stats(
        zones=data_zones, values=data_values_2d, zone_ids=zone_ids, return_type='xarray.DataArray'
    )
    general_output_checks(
        data_values_2d, dataarray_result, expected_result, verify_dtype=False, verify_attrs=False
    )
    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_2d, copied_data_values_2d)


@pytest.mark.parametrize("backend", ['numpy', 'cupy'])
def test_custom_stats(backend, data_zones, data_values_2d, result_custom_stats):
    # ---- custom stats (NumPy and CuPy only) ----
    if backend == 'cupy' and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)

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
    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_2d, copied_data_values_2d)


@pytest.mark.parametrize("backend", ['numpy'])
def test_custom_stats_dataarray(backend, data_zones, data_values_2d, result_custom_stats_dataarray):
    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)
    # ---- custom stats returns a xr.DataArray (NumPy only) ----
    custom_stats = {
        'double_sum': _double_sum,
        'range': _range,
    }
    nodata_values, zone_ids, expected_result = result_custom_stats_dataarray
    dataarray_result = stats(
        zones=data_zones, values=data_values_2d, stats_funcs=custom_stats,
        zone_ids=zone_ids, nodata_values=nodata_values, return_type='xarray.DataArray'
    )
    general_output_checks(
        data_values_2d, dataarray_result, expected_result, verify_dtype=False, verify_attrs=False
    )
    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_2d, copied_data_values_2d)


@pytest.mark.parametrize("backend", ['numpy', 'cupy'])
def test_majority_stats(backend, data_zones, data_values_2d):
    """Test that majority stat returns the most frequent value in each zone."""
    if backend == 'cupy' and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)

    df_result = stats(zones=data_zones, values=data_values_2d, stats_funcs=['majority'])
    expected_result = {
        'zone': [0, 1, 2, 3],
        'majority': [0, 1, 2, 3]
    }
    check_results(backend, df_result, expected_result)
    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_2d, copied_data_values_2d)


@pytest.mark.parametrize("backend", ['numpy', 'cupy'])
def test_majority_with_ties(backend):
    """Test majority when there are ties - should return the smallest value."""
    if backend == 'cupy' and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    # Create test data with ties
    zones_data = np.array([[1, 1, 1, 1],
                           [1, 1, 2, 2],
                           [2, 2, 2, 2]])
    values_data = np.array([[1, 1, 2, 2],  # zone 1 has two 1s and two 2s - tie
                            [3, 3, 5, 5],  # zone 1 also has two 3s, zone 2 has two 5s
                            [5, 5, 6, 6]]) # zone 2 has two more 5s and two 6s

    zones = create_test_raster(zones_data, backend)
    values = create_test_raster(values_data, backend)

    df_result = stats(zones=zones, values=values, stats_funcs=['majority'])
    # Zone 1: values [1, 1, 2, 2, 3, 3] - three values with count 2, majority is 1 (smallest)
    # Zone 2: values [5, 5, 5, 5, 6, 6] - majority is 5 (count 4)
    expected_result = {
        'zone': [1, 2],
        'majority': [1, 5]
    }
    check_results(backend, df_result, expected_result)


@pytest.mark.parametrize("stats_funcs, expected_cols", [
    (['min', 'max'], ['zone', 'min', 'max']),
    (['mean'], ['zone', 'mean']),
    (['std'], ['zone', 'std']),
    (['var'], ['zone', 'var']),
    (['count'], ['zone', 'count']),
    (['sum'], ['zone', 'sum']),
    (['min', 'max', 'count'], ['zone', 'min', 'max', 'count']),
])
@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_stats_subset_columns(backend, data_zones, data_values_2d,
                              stats_funcs, expected_cols):
    """Requesting a subset of stats returns only those columns.

    Regression test for GH-899: the dask path had a boolean short-circuit
    bug (``if 'mean' or 'std' or 'var' in stats_funcs``) that always
    evaluated to True, causing unnecessary intermediate stats to be
    computed.  After the fix, each subset exercises a distinct code path
    for compute_sum / compute_count / compute_sum_squares flags.
    """
    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    df_result = stats(zones=data_zones, values=data_values_2d,
                      stats_funcs=stats_funcs)

    # Verify values are correct for the requested stats
    all_expected = {
        'zone':  [0, 1, 2, 3],
        'mean':  [0, 1, 2, 2.4],
        'max':   [0, 1, 2, 3],
        'min':   [0, 1, 2, 0],
        'sum':   [0, 6, 8, 12],
        'std':   [0, 0, 0, 1.2],
        'var':   [0, 0, 0, 1.44],
        'count': [5, 6, 4, 5],
    }
    expected = {k: all_expected[k] for k in expected_cols}
    check_results(backend, df_result, expected)


def test_zonal_stats_against_qgis(elevation_raster_no_nans, raster, qgis_zonal_stats):
    stats_funcs = list(set(qgis_zonal_stats.keys()) - set(['zone']))
    zones_agg = create_test_raster(raster)
    values_agg = create_test_raster(elevation_raster_no_nans)

    xrspatial_df_result = stats(
        zones=zones_agg, values=values_agg, stats_funcs=stats_funcs
    )
    check_results('numpy', xrspatial_df_result, qgis_zonal_stats, atol=1e-5)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy'])
def test_zonal_stats_inputs_unmodified(backend, data_zones, data_values_2d, result_default_stats):
    if backend == 'cupy' and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)

    _ = stats(zones=data_zones, values=data_values_2d)

    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_2d, copied_data_values_2d)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_stats_3d_timeseries_via_dataset(backend):
    """Convert a 3D time-series DataArray to a Dataset and verify per-timestep stats."""
    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    zones_data = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                            [0, 0, 1, 1, 2, 2, 3, 3],
                            [0, 0, 1, 1, 2, np.nan, 3, 3]])
    values_data = np.asarray([
        [0, 0, 1, 1, 2, 2, 3, np.inf],
        [0, 0, 1, 1, 2, np.nan, 3, 0],
        [np.inf, 0, 1, 1, 2, 2, 3, 3]
    ])

    # Stack original (t0) and doubled (t1) into a 3D DataArray
    values_3d = xr.DataArray(
        np.stack([values_data, values_data * 2], axis=0),
        dims=['time', 'y', 'x'],
        coords={'time': ['t0', 't1']},
    )

    if 'dask' in backend:
        zones = xr.DataArray(da.from_array(zones_data, chunks=(3, 4)), dims=['y', 'x'])
        values_3d = values_3d.chunk({'y': 3, 'x': 4})
    else:
        zones = xr.DataArray(zones_data, dims=['y', 'x'])

    ds = values_3d.to_dataset(dim='time')
    df_result = stats(zones=zones, values=ds)

    if 'dask' in backend:
        # dask doesn't support majority stat
        expected = {
            'zone':     [0, 1, 2, 3],
            't0_mean':  [0, 1, 2, 2.4],
            't0_max':   [0, 1, 2, 3],
            't0_min':   [0, 1, 2, 0],
            't0_sum':   [0, 6, 8, 12],
            't0_std':   [0, 0, 0, 1.2],
            't0_var':   [0, 0, 0, 1.44],
            't0_count': [5, 6, 4, 5],
            't1_mean':  [0, 2, 4, 4.8],
            't1_max':   [0, 2, 4, 6],
            't1_min':   [0, 2, 4, 0],
            't1_sum':   [0, 12, 16, 24],
            't1_std':   [0, 0, 0, 2.4],
            't1_var':   [0, 0, 0, 5.76],
            't1_count': [5, 6, 4, 5],
        }
    else:
        expected = {
            'zone':         [0, 1, 2, 3],
            't0_mean':      [0, 1, 2, 2.4],
            't0_max':       [0, 1, 2, 3],
            't0_min':       [0, 1, 2, 0],
            't0_sum':       [0, 6, 8, 12],
            't0_std':       [0, 0, 0, 1.2],
            't0_var':       [0, 0, 0, 1.44],
            't0_count':     [5, 6, 4, 5],
            't0_majority':  [0, 1, 2, 3],
            't1_mean':      [0, 2, 4, 4.8],
            't1_max':       [0, 2, 4, 6],
            't1_min':       [0, 2, 4, 0],
            't1_sum':       [0, 12, 16, 24],
            't1_std':       [0, 0, 0, 2.4],
            't1_var':       [0, 0, 0, 5.76],
            't1_count':     [5, 6, 4, 5],
            't1_majority':  [0, 2, 4, 6],
        }

    check_results(backend, df_result, expected)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.parametrize("backend", ['numpy'])
def test_stats_3d_timeseries_via_dataset_zone_ids(backend):
    """Zone filtering works with Dataset from 3D time-series DataArray."""
    zones_data = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                            [0, 0, 1, 1, 2, 2, 3, 3],
                            [0, 0, 1, 1, 2, np.nan, 3, 3]])
    values_data = np.asarray([
        [0, 0, 1, 1, 2, 2, 3, np.inf],
        [0, 0, 1, 1, 2, np.nan, 3, 0],
        [np.inf, 0, 1, 1, 2, 2, 3, 3]
    ])

    values_3d = xr.DataArray(
        np.stack([values_data, values_data * 2], axis=0),
        dims=['time', 'y', 'x'],
        coords={'time': ['t0', 't1']},
    )
    zones = xr.DataArray(zones_data, dims=['y', 'x'])
    ds = values_3d.to_dataset(dim='time')

    df_result = stats(zones=zones, values=ds, zone_ids=[0, 3])

    expected = {
        'zone':         [0, 3],
        't0_mean':      [0, 2.4],
        't0_max':       [0, 3],
        't0_min':       [0, 0],
        't0_sum':       [0, 12],
        't0_std':       [0, 1.2],
        't0_var':       [0, 1.44],
        't0_count':     [5, 5],
        't0_majority':  [0, 3],
        't1_mean':      [0, 4.8],
        't1_max':       [0, 6],
        't1_min':       [0, 0],
        't1_sum':       [0, 24],
        't1_std':       [0, 2.4],
        't1_var':       [0, 5.76],
        't1_count':     [5, 5],
        't1_majority':  [0, 6],
    }

    check_results(backend, df_result, expected)


@pytest.mark.parametrize("backend", ['numpy'])
def test_stats_3d_timeseries_via_dataset_custom_stats(backend):
    """Custom stats_funcs work with Dataset from 3D time-series DataArray."""
    zones_data = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                            [0, 0, 1, 1, 2, 2, 3, 3],
                            [0, 0, 1, 1, 2, np.nan, 3, 3]])
    values_data = np.asarray([
        [0, 0, 1, 1, 2, 2, 3, np.inf],
        [0, 0, 1, 1, 2, np.nan, 3, 0],
        [np.inf, 0, 1, 1, 2, 2, 3, 3]
    ])

    values_3d = xr.DataArray(
        np.stack([values_data, values_data * 2], axis=0),
        dims=['time', 'y', 'x'],
        coords={'time': ['t0', 't1']},
    )
    zones = xr.DataArray(zones_data, dims=['y', 'x'])
    ds = values_3d.to_dataset(dim='time')

    custom_stats = {
        'double_sum': _double_sum,
        'range': _range,
    }
    df_result = stats(
        zones=zones, values=ds, stats_funcs=custom_stats,
        zone_ids=[1, 2], nodata_values=0,
    )

    expected = {
        'zone':          [1, 2],
        't0_double_sum': [12, 16],
        't0_range':      [0, 0],
        't1_double_sum': [24, 32],
        't1_range':      [0, 0],
    }

    check_results(backend, df_result, expected)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_count_crosstab_2d(backend, data_zones, data_values_2d, result_count_crosstab_2d):
    # copy input data to verify they're unchanged after running the function

    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)

    zone_ids, cat_ids, expected_result = result_count_crosstab_2d
    df_result = crosstab(
        zones=data_zones, values=data_values_2d, zone_ids=zone_ids, cat_ids=cat_ids,
    )
    check_results(backend, df_result, expected_result)
    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_2d, copied_data_values_2d)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_percentage_crosstab_2d(backend, data_zones, data_values_2d, result_percentage_crosstab_2d):
    # copy input data to verify they're unchanged after running the function

    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)

    nodata_values, zone_ids, cat_ids, expected_result = result_percentage_crosstab_2d
    df_result = crosstab(
        zones=data_zones, values=data_values_2d, zone_ids=zone_ids, cat_ids=cat_ids,
        nodata_values=nodata_values, agg='percentage'
    )
    check_results(backend, df_result, expected_result)
    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_2d, copied_data_values_2d)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_crosstab_3d_count(backend, data_zones, data_values_3d, result_crosstab_3d):

    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_3d = copy.deepcopy(data_values_3d)

    layer, zone_ids, expected_result = result_crosstab_3d
    df_result = crosstab(zones=data_zones, values=data_values_3d,
                         zone_ids=zone_ids, layer=layer, agg='count')
    check_results(backend, df_result, expected_result['count'])
    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_3d, copied_data_values_3d)


@pytest.mark.parametrize("backend", ['numpy'])
def test_crosstab_3d_agg_method(backend, data_zones, data_values_3d, result_crosstab_3d):
    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_3d = copy.deepcopy(data_values_3d)

    layer, zone_ids, expected_result = result_crosstab_3d
    agg_methods = ['min', 'max', 'mean', 'sum', 'std', 'var', 'count']
    for agg in agg_methods:
        df_result = crosstab(zones=data_zones, values=data_values_3d,
                             zone_ids=zone_ids, layer=layer, agg=agg)
        check_results(backend, df_result, expected_result[agg])
        assert_input_data_unmodified(data_zones, copied_data_zones)
        assert_input_data_unmodified(data_values_3d, copied_data_values_3d)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_nodata_values_crosstab_3d(
    backend,
    data_zones,
    data_values_3d,
    result_nodata_values_crosstab_3d
):

    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    # copy input data to verify they're unchanged after running the function
    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_3d = copy.deepcopy(data_values_3d)

    nodata_values, layer, zone_ids, expected_result = result_nodata_values_crosstab_3d
    df_result = crosstab(
        zones=data_zones, values=data_values_3d, zone_ids=zone_ids,
        layer=layer, nodata_values=nodata_values
    )
    check_results(backend, df_result, expected_result)
    assert_input_data_unmodified(data_zones, copied_data_zones)
    assert_input_data_unmodified(data_values_3d, copied_data_values_3d)


@pytest.mark.skipif(not dask_array_available(), reason="Requires Dask")
def test_crosstab_dask_from_dataset():
    """
    Test crosstab with dask arrays originating from xarray Datasets.

    This is a regression test for issue #777 where dask arrays created via
    Dataset.to_array().sel() had misaligned chunks that caused IndexError.
    """
    # Simulate what happens with rioxarray band_as_variable=True
    data_band1 = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                           [0, 0, 1, 1, 2, 2, 3, 3],
                           [0, 0, 1, 1, 2, 2, 3, 3]], dtype=float)
    data_band2 = np.array([[1, 1, 2, 2, 3, 3, 0, 0],
                           [1, 1, 2, 2, 3, 3, 0, 0],
                           [1, 1, 2, 2, 3, 3, 0, 0]], dtype=float)

    # Use different chunk sizes to simulate real-world scenario
    dask_band1 = da.from_array(data_band1, chunks=(2, 3))
    dask_band2 = da.from_array(data_band2, chunks=(2, 3))

    ds = xr.Dataset({
        'band_1': (['y', 'x'], dask_band1),
        'band_2': (['y', 'x'], dask_band2),
    })

    # This is the pattern from issue #777: to_array().sel(variable='band_1', drop=True)
    values = ds.to_array().sel(variable='band_1', drop=True)

    # Create zones with different chunks
    zones_data = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                           [0, 0, 1, 1, 2, 2, 3, 3],
                           [0, 0, 1, 1, 2, 2, 3, 3]], dtype=float)
    zones_dask = da.from_array(zones_data, chunks=(3, 4))
    zones = xr.DataArray(zones_dask, dims=['y', 'x'])

    # This should not raise an error
    result = crosstab(zones, values)
    assert isinstance(result, dd.DataFrame)

    result_df = result.compute()
    expected = {
        'zone': [0.0, 1.0, 2.0, 3.0],
        0.0: [6, 0, 0, 0],
        1.0: [0, 6, 0, 0],
        2.0: [0, 0, 6, 0],
        3.0: [0, 0, 0, 6],
    }
    check_results('dask+numpy', result, expected)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_apply_2d(backend):
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip('cupy not available')
    if 'dask' in backend and not dask_array_available():
        pytest.skip('dask not available')

    zones_data = np.array([[1, 1, 0],
                           [0, 2, 2],
                           [3, 3, 3]], dtype=np.int32)
    values_data = np.array([[10.0, 20.0, 30.0],
                            [40.0, 50.0, 60.0],
                            [70.0, 80.0, 90.0]])
    zones = create_test_raster(zones_data, backend)
    values = create_test_raster(values_data, backend)

    result = apply(zones, values, lambda x: x * 2, nodata=0)

    expected = np.array([[20.0, 40.0, 30.0],
                         [40.0, 100.0, 120.0],
                         [140.0, 160.0, 180.0]])
    general_output_checks(values, result, expected, verify_attrs=True)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_apply_does_not_mutate_input(backend):
    if 'dask' in backend and not dask_array_available():
        pytest.skip('dask not available')

    zones_data = np.array([[1, 1], [2, 2]], dtype=np.int32)
    values_data = np.array([[10.0, 20.0], [30.0, 40.0]])
    zones = create_test_raster(zones_data, backend, chunks=(2, 2))
    values = create_test_raster(values_data, backend, chunks=(2, 2))
    values_before = values.copy(deep=True)

    apply(zones, values, lambda x: x * 0)

    assert_input_data_unmodified(values_before, values)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_apply_3d(backend):
    if 'dask' in backend and not dask_array_available():
        pytest.skip('dask not available')

    zones_data = np.array([[1, 0],
                           [0, 2]], dtype=np.int32)
    values_data = np.ones((2, 2, 3)) * 5.0

    zones = xr.DataArray(zones_data, dims=['y', 'x'])
    vals = xr.DataArray(values_data, dims=['y', 'x', 'band'])

    if 'dask' in backend:
        zones.data = da.from_array(zones.data, chunks=(2, 2))
        vals.data = da.from_array(vals.data, chunks=(2, 2, 3))

    result = apply(zones, vals, lambda x: x + 10, nodata=0)

    assert result.shape == vals.shape
    # zone 1 cell (0,0) and zone 2 cell (1,1) should be 15
    result_np = result.values if not hasattr(result.data, 'compute') else result.data.compute()
    np.testing.assert_equal(result_np[0, 0, :], [15.0, 15.0, 15.0])
    np.testing.assert_equal(result_np[1, 1, :], [15.0, 15.0, 15.0])
    # nodata cells (0,1) and (1,0) remain 5
    np.testing.assert_equal(result_np[0, 1, :], [5.0, 5.0, 5.0])
    np.testing.assert_equal(result_np[1, 0, :], [5.0, 5.0, 5.0])


def test_apply_nodata_none():
    zones_data = np.array([[0, 1], [2, 3]], dtype=np.int32)
    values_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    zones = xr.DataArray(zones_data, dims=['y', 'x'])
    values = xr.DataArray(values_data, dims=['y', 'x'])

    result = apply(zones, values, lambda x: x * 10, nodata=None)
    expected = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_apply_backward_compat():
    """Same scenario as original test, but with new return semantics."""
    zones_val = np.zeros((3, 3), dtype=np.int32)
    zones_val[1] = 1
    zones_val[2] = 2
    zones = xr.DataArray(zones_val, dims=['y', 'x'])

    values_val = np.array([[0.0, 1.0, 2.0],
                           [3.0, 4.0, 5.0],
                           [6.0, 7.0, np.nan]])
    values = xr.DataArray(values_val, dims=['y', 'x'])

    result = apply(zones, values, lambda x: 0, nodata=2)

    assert result.shape == values.shape
    result_np = result.values
    # zones 0 and 1 → func applied (all become 0)
    assert (result_np[0] == [0, 0, 0]).all()
    assert (result_np[1] == [0, 0, 0]).all()
    # zone 2 = nodata → values unchanged
    assert np.isclose(result_np[2], values_val[2], equal_nan=True).all()


def test_apply_validation_errors():
    zones_float = xr.DataArray(np.array([[1.0, 2.0]], dtype=np.float64), dims=['y', 'x'])
    values = xr.DataArray(np.array([[1.0, 2.0]]), dims=['y', 'x'])

    with pytest.raises(ValueError, match="integers"):
        apply(zones_float, values, lambda x: x)

    zones_ok = xr.DataArray(np.array([[1, 2]], dtype=np.int32), dims=['y', 'x'])
    values_wrong_shape = xr.DataArray(np.array([[1.0, 2.0, 3.0]]), dims=['y', 'x'])
    with pytest.raises(ValueError, match="Incompatible shapes"):
        apply(zones_ok, values_wrong_shape, lambda x: x)

    zones_3d = xr.DataArray(np.ones((2, 2, 2), dtype=np.int32), dims=['y', 'x', 'z'])
    with pytest.raises(ValueError, match="2D"):
        apply(zones_3d, values, lambda x: x)


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


def _make_regions_raster(arr, backend):
    """Create a test raster from *arr* for the given backend."""
    raster = create_test_raster(arr, backend)
    return raster


def _count_unique(raster_regions):
    """Count unique values in a regions result, computing dask if needed."""
    data = raster_regions.data
    if da is not None and isinstance(data, da.Array):
        data = data.compute()
    return len(np.unique(data))


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_four_pixel_connectivity_int(backend):
    arr = np.array([[0, 0, 0, 0],
                    [0, 4, 0, 0],
                    [1, 4, 4, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0]], dtype=np.int64)
    raster = _make_regions_raster(arr, backend)
    raster_regions = regions(raster, neighborhood=4)
    assert _count_unique(raster_regions) == 3
    assert raster.shape == raster_regions.shape


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_four_pixel_connectivity_float(backend):
    arr = np.array([[0, 0, 0, np.nan],
                    [0, 4, 0, 0],
                    [1, 4, 4, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0]], dtype=np.float64)
    raster = _make_regions_raster(arr, backend)
    raster_regions = regions(raster, neighborhood=4)
    assert _count_unique(raster_regions) == 4
    assert raster.shape == raster_regions.shape


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_eight_pixel_connectivity_int(backend):
    arr = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1]], dtype=np.int64)
    raster = _make_regions_raster(arr, backend)
    raster_regions = regions(raster, neighborhood=8)
    assert _count_unique(raster_regions) == 2
    assert raster.shape == raster_regions.shape


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_eight_pixel_connectivity_float(backend):
    arr = np.array([[1, 0, 0, np.nan],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1]], dtype=np.float64)
    raster = _make_regions_raster(arr, backend)
    raster_regions = regions(raster, neighborhood=8)
    assert _count_unique(raster_regions) == 3
    assert raster.shape == raster_regions.shape


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_single_pixel(backend):
    arr = np.array([[np.nan, np.nan],
                    [np.nan, 5.0]], dtype=np.float64)
    raster = _make_regions_raster(arr, backend)
    raster_regions = regions(raster, neighborhood=4)
    data = raster_regions.data
    if da is not None and isinstance(data, da.Array):
        data = data.compute()
    assert np.nansum(data > 0) == 1
    assert raster.shape == raster_regions.shape


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_all_same_value(backend):
    arr = np.full((4, 4), 7.0, dtype=np.float64)
    raster = _make_regions_raster(arr, backend)
    raster_regions = regions(raster, neighborhood=4)
    assert _count_unique(raster_regions) == 1
    assert raster.shape == raster_regions.shape


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_all_nan(backend):
    arr = np.full((3, 3), np.nan, dtype=np.float64)
    raster = _make_regions_raster(arr, backend)
    raster_regions = regions(raster, neighborhood=4)
    data = raster_regions.data
    if da is not None and isinstance(data, da.Array):
        data = data.compute()
    assert np.all(np.isnan(data))
    assert raster.shape == raster_regions.shape


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_numpy_dask_match(backend):
    """Verify numpy and dask backends produce identical results."""
    arr = np.array([[1, 1, 0, 2],
                    [1, 1, 0, 2],
                    [0, 0, 0, 0],
                    [3, 3, 0, 3]], dtype=np.float64)
    raster = _make_regions_raster(arr, backend)
    result = regions(raster, neighborhood=4)
    data = result.data
    if da is not None and isinstance(data, da.Array):
        data = data.compute()
    # 0-region is connected, 1-region, 2-region, and two separate 3-regions
    assert _count_unique(result) == 5
    assert result.shape == arr.shape


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


@pytest.mark.skipif(not dask_array_available(), reason="Requires Dask")
def test_dask_zonal_stats_no_concat_warnings():
    """Regression test for #774: dd.concat should not warn about unknown divisions."""
    import warnings

    zones_data = np.array([[0, 0, 1, 1],
                           [0, 0, 1, 1],
                           [2, 2, 3, 3]])
    values_data = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]], dtype=float)

    zones = xr.DataArray(da.from_array(zones_data, chunks=(3, 2)), dims=['y', 'x'])
    values = xr.DataArray(da.from_array(values_data, chunks=(3, 2)), dims=['y', 'x'])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        # all zones (exercises column-wise concat, line 262)
        result_all = stats(zones=zones, values=values)
        assert isinstance(result_all, dd.DataFrame)
        result_all.compute()

        # filtered zone_ids (exercises row-wise concat, line 275)
        result_filtered = stats(zones=zones, values=values, zone_ids=[0, 3])
        assert isinstance(result_filtered, dd.DataFrame)
        result_filtered.compute()

    division_warnings = [
        w for w in caught
        if "unknown divisions" in str(w.message).lower()
    ]
    assert division_warnings == [], (
        f"Expected no 'unknown divisions' warnings, got: "
        f"{[str(w.message) for w in division_warnings]}"
    )


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


# ---------------------------------------------------------------------------
# Regression tests for #881: np.unique / np.isfinite must not materialise
# the full dask array.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_dask_array(), reason="dask.array not available")
def test_stats_does_not_materialise_dask_zones():
    """stats() with dask backend must never pass a dask array to np.unique."""
    from unittest import mock

    zones_np = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                         [0, 0, 1, 1, 2, 2, 3, 3],
                         [0, 0, 1, 1, 2, np.nan, 3, 3]])
    values_np = np.array([[0, 0, 1, 1, 2, 2, 3, np.inf],
                          [0, 0, 1, 1, 2, np.nan, 3, 0],
                          [np.inf, 0, 1, 1, 2, 2, 3, 3]])

    zones = xr.DataArray(da.from_array(zones_np, chunks=(3, 4)), dims=['y', 'x'])
    values = xr.DataArray(da.from_array(values_np, chunks=(3, 4)), dims=['y', 'x'])

    _real_np_unique = np.unique

    def _guarded_unique(a, *args, **kwargs):
        if isinstance(a, da.Array):
            raise AssertionError("np.unique called with a dask array — would materialise")
        return _real_np_unique(a, *args, **kwargs)

    with mock.patch("xrspatial.zonal.np.unique", side_effect=_guarded_unique):
        result = stats(zones, values)

    # dask path returns a lazy dask DataFrame; compute to verify correctness
    if hasattr(result, 'compute'):
        result = result.compute()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


@pytest.mark.skipif(not has_dask_array(), reason="dask.array not available")
def test_crosstab_does_not_materialise_dask_zones():
    """crosstab() with dask backend must never pass a dask array to np.unique."""
    from unittest import mock

    zones_np = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                         [0, 0, 1, 1, 2, 2, 3, 3],
                         [0, 0, 1, 1, 2, np.nan, 3, 3]])
    values_np = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
                          [0, 0, 1, 1, 2, np.nan, 3, 0],
                          [0, 0, 1, 1, 2, 2, 3, 3]])

    zones = xr.DataArray(da.from_array(zones_np, chunks=(3, 4)), dims=['y', 'x'])
    values = xr.DataArray(da.from_array(values_np, chunks=(3, 4)), dims=['y', 'x'])

    _real_np_unique = np.unique

    def _guarded_unique(a, *args, **kwargs):
        if isinstance(a, da.Array):
            raise AssertionError("np.unique called with a dask array — would materialise")
        return _real_np_unique(a, *args, **kwargs)

    with mock.patch("xrspatial.zonal.np.unique", side_effect=_guarded_unique):
        result = crosstab(zones, values)

    # dask path returns a lazy dask DataFrame; compute to verify correctness
    if hasattr(result, 'compute'):
        result = result.compute()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
