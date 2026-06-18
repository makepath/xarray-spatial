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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_default_stats(backend, data_zones, data_values_2d, result_default_stats,
                       result_default_stats_no_majority):
    if 'cupy' in backend and not has_cuda_and_cupy():
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
@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_zone_ids_stats(backend, data_zones, data_values_2d, result_zone_ids_stats,
                        result_zone_ids_stats_no_majority):
    if 'cupy' in backend and not has_cuda_and_cupy():
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


# Regression tests for issue #2562: cupy backend used to drop all-NaN zones
# and could desync the zone/stats alignment when zone_ids contained IDs that
# don't appear in the raster.

@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in.*:RuntimeWarning")
@pytest.mark.parametrize("backend", ['numpy', 'cupy'])
def test_stats_all_nan_zone_preserved(backend):
    # Zone 5 exists in the zones raster but every value at zone 5 is NaN.
    # The numpy path returns zone 5 with NaN stats; the cupy path used to
    # drop the row entirely. After the fix both backends should agree.
    if backend == 'cupy' and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    zones_data = np.array([[1, 1, 2, 2],
                           [1, 5, 2, 5],
                           [3, 3, 5, 5]], dtype=np.float64)
    values_data = np.array([[1.0, 2.0, 3.0, 4.0],
                            [5.0, np.nan, 7.0, np.nan],
                            [9.0, 10.0, np.nan, np.nan]])

    zones = create_test_raster(zones_data, backend)
    values = create_test_raster(values_data, backend)

    df_result = stats(
        zones=zones, values=values, stats_funcs=['min', 'max', 'mean', 'count']
    )

    assert list(df_result.columns) == ['zone', 'min', 'max', 'mean', 'count']
    expected_result = {
        'zone': [1, 2, 3, 5],
        'min': [1.0, 3.0, 9.0, np.nan],
        'max': [5.0, 7.0, 10.0, np.nan],
        'mean': [(1.0 + 2.0 + 5.0) / 3, (3.0 + 4.0 + 7.0) / 3, 9.5, np.nan],
        # An empty zone (zone 5, all NaN) reports count == 0 because the
        # number of valid cells is zero. Other stats stay NaN (issue #2644).
        'count': [3, 3, 2, 0],
    }
    check_results(backend, df_result, expected_result)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.parametrize("backend", ['numpy', 'cupy'])
def test_stats_zone_ids_missing_from_raster(backend):
    # zone_ids contains 999 which does not appear in the zones raster.
    # The numpy path drops missing IDs so the result lists only zone 1.
    # The cupy path used to keep 999 in the zone column while indexing
    # into a found-only counts/index list, which desynced the rows.
    if backend == 'cupy' and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    zones_data = np.array([[1, 1, 2, 2],
                           [1, 1, 2, 2],
                           [3, 3, 3, 3]], dtype=np.float64)
    values_data = np.array([[10.0, 20.0, 1.0, 2.0],
                            [30.0, 40.0, 3.0, 4.0],
                            [100.0, 100.0, 100.0, 100.0]])

    zones = create_test_raster(zones_data, backend)
    values = create_test_raster(values_data, backend)

    df_result = stats(
        zones=zones, values=values, zone_ids=[1, 999],
        stats_funcs=['min', 'max', 'sum', 'count']
    )

    assert list(df_result.columns) == ['zone', 'min', 'max', 'sum', 'count']
    expected_result = {
        'zone': [1],
        'min': [10.0],
        'max': [40.0],
        'sum': [100.0],
        'count': [4],
    }
    check_results(backend, df_result, expected_result)

    # Also verify that zone_ids passed in reverse order (or with duplicates)
    # produces the same numpy-style ordering: np.unique sorts ascending and
    # dedupes, so [3, 1, 1] becomes [1, 3]. This pins the cupy path's
    # np.unique normalization that matches the numpy backend.
    df_result_reversed = stats(
        zones=zones, values=values, zone_ids=[3, 1, 1],
        stats_funcs=['min', 'max', 'sum', 'count'],
    )
    expected_reversed = {
        'zone': [1, 3],
        'min': [10.0, 100.0],
        'max': [40.0, 100.0],
        'sum': [100.0, 400.0],
        'count': [4, 4],
    }
    check_results(backend, df_result_reversed, expected_reversed)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in.*:RuntimeWarning")
def test_stats_cupy_matches_numpy_row_for_row():
    # Cross-backend parity check. Run numpy and cupy with the same inputs
    # (all-NaN zone + zone_ids containing a missing ID) and assert the
    # resulting DataFrames match row-for-row.
    if not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    zones_data = np.array([[1, 1, 2, 2, 5],
                           [1, 5, 2, 5, 5],
                           [3, 3, 5, 5, 4]], dtype=np.float64)
    values_data = np.array([[1.0, 2.0, 3.0, 4.0, np.nan],
                            [5.0, np.nan, 7.0, np.nan, np.nan],
                            [9.0, 10.0, np.nan, np.nan, 42.0]])

    zones_np = create_test_raster(zones_data, 'numpy')
    values_np = create_test_raster(values_data, 'numpy')
    zones_cp = create_test_raster(zones_data, 'cupy')
    values_cp = create_test_raster(values_data, 'cupy')

    stats_funcs = ['min', 'max', 'mean', 'sum', 'count']

    df_np = stats(zones=zones_np, values=values_np, stats_funcs=stats_funcs)
    df_cp = stats(zones=zones_cp, values=values_cp, stats_funcs=stats_funcs)

    assert list(df_np.columns) == list(df_cp.columns)
    assert (df_np['zone'].to_numpy() == df_cp['zone'].to_numpy()).all()
    for col in df_np.columns[1:]:
        np.testing.assert_allclose(
            df_np[col].to_numpy(), df_cp[col].to_numpy(),
            rtol=1e-05, atol=1e-07, equal_nan=True,
        )

    # And again with zone_ids requesting a zone that doesn't exist.
    df_np2 = stats(
        zones=zones_np, values=values_np, zone_ids=[1, 5, 999],
        stats_funcs=stats_funcs,
    )
    df_cp2 = stats(
        zones=zones_cp, values=values_cp, zone_ids=[1, 5, 999],
        stats_funcs=stats_funcs,
    )
    assert (df_np2['zone'].to_numpy() == df_cp2['zone'].to_numpy()).all()
    for col in df_np2.columns[1:]:
        np.testing.assert_allclose(
            df_np2[col].to_numpy(), df_cp2[col].to_numpy(),
            rtol=1e-05, atol=1e-07, equal_nan=True,
        )


def test_stats_cupy_zone_is_column_not_index():
    # Regression test for issue #2641: the cupy stats path used to call
    # stats_df.set_index("zone") without assigning the result, a no-op that
    # discarded its return value. The numpy and dask backends return "zone"
    # as a plain column with a default RangeIndex, so the cupy path must too.
    if not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    zones_data = np.array([[1, 1, 2, 2],
                           [3, 3, 2, 2]], dtype=np.float64)
    values_data = np.array([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0]])

    zones_np = create_test_raster(zones_data, 'numpy')
    values_np = create_test_raster(values_data, 'numpy')
    zones_cp = create_test_raster(zones_data, 'cupy')
    values_cp = create_test_raster(values_data, 'cupy')

    stats_funcs = ['min', 'max', 'sum', 'count']
    df_np = stats(zones=zones_np, values=values_np, stats_funcs=stats_funcs)
    df_cp = stats(zones=zones_cp, values=values_cp, stats_funcs=stats_funcs)

    # "zone" stays a column on both backends, and neither sets it as index.
    assert 'zone' in df_cp.columns
    assert df_cp.index.name is None
    assert df_cp.index.equals(df_np.index)
    assert list(df_cp.columns) == list(df_np.columns)


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


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Degrees of freedom:RuntimeWarning")
@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_stats_all_nan_zone(backend):
    """Zone where every value is NaN should not crash.

    All backends agree: the empty zone stays in the output. Its count is
    0 (zero valid cells) while every other stat is NaN (issue #2644). The
    cupy path used to drop the zone entirely (issue #2562); that is fixed
    and the per-backend branching is gone.
    """
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    zones_data = np.array([[1, 1],
                            [2, 2]])
    values_data = np.array([[np.nan, np.nan],   # zone 1: all NaN
                             [5.0,    7.0]])     # zone 2: normal

    zones = create_test_raster(zones_data, backend, chunks=(2, 2))
    values = create_test_raster(values_data, backend, chunks=(2, 2))

    funcs = ['mean', 'max', 'min', 'sum', 'count']
    df_result = stats(zones=zones, values=values, stats_funcs=funcs)

    expected = {
        'zone':  [1, 2],
        'mean':  [np.nan, 6.0],
        'max':   [np.nan, 7.0],
        'min':   [np.nan, 5.0],
        'sum':   [np.nan, 12.0],
        'count': [0, 2],
    }
    check_results(backend, df_result, expected)


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_stats_single_cell_zones(backend):
    """Each zone has exactly one cell — std and var must be 0, not NaN."""
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    zones_data = np.array([[1, 2, 3]])
    values_data = np.array([[10.0, 20.0, 30.0]])

    zones = create_test_raster(zones_data, backend, chunks=(1, 3))
    values = create_test_raster(values_data, backend, chunks=(1, 3))

    funcs = ['mean', 'max', 'min', 'std', 'var', 'count']
    df_result = stats(zones=zones, values=values, stats_funcs=funcs)

    expected = {
        'zone':  [1, 2, 3],
        'mean':  [10.0, 20.0, 30.0],
        'max':   [10.0, 20.0, 30.0],
        'min':   [10.0, 20.0, 30.0],
        'std':   [0.0, 0.0, 0.0],
        'var':   [0.0, 0.0, 0.0],
        'count': [1, 1, 1],
    }
    check_results(backend, df_result, expected)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_stats_negative_zone_ids(backend):
    """Negative integers are valid zone IDs."""
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    zones_data = np.array([[-1, -1, 2, 2],
                            [-1, -1, 2, 2]])
    values_data = np.array([[1.0, 3.0, 10.0, 20.0],
                             [5.0, 7.0, 30.0, 40.0]])

    zones = create_test_raster(zones_data, backend, chunks=(2, 2))
    values = create_test_raster(values_data, backend, chunks=(2, 2))

    funcs = ['mean', 'max', 'min', 'sum', 'count']
    df_result = stats(zones=zones, values=values, stats_funcs=funcs)

    expected = {
        'zone':  [-1, 2],
        'mean':  [4.0, 25.0],
        'max':   [7.0, 40.0],
        'min':   [1.0, 10.0],
        'sum':   [16.0, 100.0],
        'count': [4, 4],
    }
    check_results(backend, df_result, expected)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Degrees of freedom:RuntimeWarning")
@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_stats_nodata_wipes_zone(backend):
    """When nodata_values filters out every finite value in a zone, the zone
    is empty: count is 0 and the other stats are NaN.

    All backends agree after the issue #2562 fix. The empty-zone count is 0
    rather than NaN per issue #2644.
    """
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    zones_data = np.array([[1, 1],
                            [2, 2]])
    values_data = np.array([[5.0, 5.0],   # zone 1: all values equal to nodata
                             [3.0, 7.0]])  # zone 2: normal

    zones = create_test_raster(zones_data, backend, chunks=(2, 2))
    values = create_test_raster(values_data, backend, chunks=(2, 2))

    funcs = ['mean', 'max', 'min', 'sum', 'count']
    df_result = stats(zones=zones, values=values, stats_funcs=funcs, nodata_values=5)

    expected = {
        'zone':  [1, 2],
        'mean':  [np.nan, 5.0],
        'max':   [np.nan, 7.0],
        'min':   [np.nan, 3.0],
        'sum':   [np.nan, 10.0],
        'count': [0, 2],
    }
    check_results(backend, df_result, expected)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_stats_nodata_values_zero_across_backends_1227(backend):
    """Regression: nodata_values=0 must be filtered on every backend.

    The cupy path previously used `if nodata_values:` (truthiness), so passing
    nodata_values=0 silently skipped the filter and zeros were included in
    the per-zone statistics.  Every other backend used `is not None` and
    dropped the zeros correctly — that divergence is what this test pins.
    """
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    # Zone 1: one zero (should be dropped) and three 10s -> mean=10, count=3.
    # Zone 2: one zero (should be dropped) and three 20s -> mean=20, count=3.
    zones_data = np.array([[1, 1, 2, 2],
                           [1, 1, 2, 2]], dtype=float)
    values_data = np.array([[0.0, 10.0, 0.0, 20.0],
                            [10.0, 10.0, 20.0, 20.0]])

    zones = create_test_raster(zones_data, backend, chunks=(2, 2))
    values = create_test_raster(values_data, backend, chunks=(2, 2))

    funcs = ['mean', 'sum', 'count']
    df_result = stats(
        zones=zones, values=values,
        stats_funcs=funcs, nodata_values=0,
    )

    expected = {
        'zone':  [1, 2],
        'mean':  [10.0, 20.0],
        'sum':   [30.0, 60.0],
        'count': [3, 3],
    }
    check_results(backend, df_result, expected)


@pytest.mark.skipif(not dask_array_available(), reason="Requires Dask")
@pytest.mark.parametrize("backend", ['dask+numpy', 'dask+cupy'])
def test_stats_zone_in_subset_of_blocks(backend):
    """A zone present in only some dask blocks must still be reduced correctly."""
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    # 2x6 grid, chunked into two 2x3 blocks.
    # Zone 1 only in left block, zone 3 only in right block, zone 2 spans both.
    zones_data = np.array([[1, 1, 2, 2, 3, 3],
                            [1, 1, 2, 2, 3, 3]], dtype=float)
    values_data = np.array([[2.0, 4.0, 10.0, 20.0, 100.0, 200.0],
                             [6.0, 8.0, 30.0, 40.0, 300.0, 400.0]])

    zones = create_test_raster(zones_data, backend, chunks=(2, 3))
    values = create_test_raster(values_data, backend, chunks=(2, 3))

    funcs = ['mean', 'max', 'min', 'sum', 'count']
    df_result = stats(zones=zones, values=values, stats_funcs=funcs)

    expected = {
        'zone':  [1, 2, 3],
        'mean':  [5.0, 25.0, 250.0],
        'max':   [8.0, 40.0, 400.0],
        'min':   [2.0, 10.0, 100.0],
        'sum':   [20.0, 100.0, 1000.0],
        'count': [4, 4, 4],
    }
    check_results(backend, df_result, expected)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
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
def test_stats_variance_numerical_stability_1090(backend):
    """Dask std/var should match numpy for data with large mean, small spread.

    Regression test for #1090: the naive one-pass formula
    ``(Σx² − (Σx)²/n) / n`` loses precision through catastrophic
    cancellation.  The fix uses Chan-Golub-LeVeque parallel merge.
    """
    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    # Values near 1e8 with a spread of 1: the naive formula would lose
    # most of the significant digits in float64.
    zones_data = np.array([[1, 1, 1, 1, 1, 1]])
    values_data = np.array([[1e8, 1e8 + 1, 1e8 + 2,
                             1e8 + 3, 1e8 + 4, 1e8 + 5]], dtype=np.float64)

    zones = create_test_raster(zones_data, backend, chunks=(1, 3))
    values = create_test_raster(values_data, backend, chunks=(1, 3))

    df_result = stats(zones=zones, values=values,
                      stats_funcs=['mean', 'std', 'var'])

    if hasattr(df_result, 'compute'):
        df_result = df_result.compute()

    # Reference: population variance of [0,1,2,3,4,5] = 35/12 ≈ 2.9167
    expected_var = np.var(np.arange(6, dtype=np.float64))
    expected_std = np.std(np.arange(6, dtype=np.float64))

    actual_var = float(df_result['var'].iloc[0])
    actual_std = float(df_result['std'].iloc[0])

    assert abs(actual_var - expected_var) < 1e-6, (
        f"var={actual_var}, expected={expected_var}"
    )
    assert abs(actual_std - expected_std) < 1e-6, (
        f"std={actual_std}, expected={expected_std}"
    )


def test_stats_nodata_none_no_warning_1090():
    """Passing nodata_values=None (the default) should not trigger warnings.

    Regression test for #1090: ``zone_values != None`` triggered a numpy
    FutureWarning.
    """
    import warnings

    zones_data = np.array([[1, 1], [2, 2]], dtype=float)
    values_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    zones = xr.DataArray(zones_data)
    values = xr.DataArray(values_data)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = stats(zones=zones, values=values, nodata_values=None)

    assert len(df) == 2
    assert float(df['mean'].iloc[0]) == 1.5
    assert float(df['mean'].iloc[1]) == 3.5


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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_count_crosstab_2d(backend, data_zones, data_values_2d, result_count_crosstab_2d):
    # copy input data to verify they're unchanged after running the function

    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_percentage_crosstab_2d(backend, data_zones, data_values_2d, result_percentage_crosstab_2d):
    # copy input data to verify they're unchanged after running the function

    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_crosstab_2d_cat_ids_skips_earlier_category(backend, data_zones, data_values_2d):
    """Regression test for issue #2560.

    When cat_ids filters out a category that appears in the values raster,
    the count for later selected categories must not be inflated by cells
    that belong to the skipped category.

    Zone 3 in the test fixture contains values [3, 3, 0, 3, 3] after nodata
    filtering. Asking for cat_ids=[3] alone (skipping 0) should report 4
    threes, not 5.
    """
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and not dask_array_available():
        pytest.skip("Requires Dask")

    copied_data_zones = copy.deepcopy(data_zones)
    copied_data_values_2d = copy.deepcopy(data_values_2d)

    # cat_ids=[3] alone, with zone 3 containing 0s too. unique_cats is
    # [0, 1, 2, 3]; the buggy code would let cat_start stay at the start
    # of the sorted zone array and report 5 threes for zone 3.
    df_result = crosstab(
        zones=data_zones, values=data_values_2d,
        zone_ids=[0, 1, 2, 3], cat_ids=[3],
    )
    expected = {
        'zone': [0, 1, 2, 3],
        3:      [0, 0, 0, 4],
    }
    check_results(backend, df_result, expected)

    # cat_ids=[1, 3] skips category 2 between them. Zone 1 has six 1s
    # and no 3s; zone 3 has zero 1s and four 3s.
    df_result = crosstab(
        zones=data_zones, values=data_values_2d,
        zone_ids=[0, 1, 2, 3], cat_ids=[1, 3],
    )
    expected = {
        'zone': [0, 1, 2, 3],
        1:      [0, 6, 0, 0],
        3:      [0, 0, 0, 4],
    }
    check_results(backend, df_result, expected)

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


def test_sort_and_stride_3d_no_mutation_and_parity():
    # Regression for #3381: the 3D branch of _sort_and_stride used to
    # copy.deepcopy the values array and reindex it row by row in a Python
    # loop. It now reindexes in a single vectorized fancy-index. Pin both
    # properties: the input must be left untouched, and the reordered output
    # must match the layer-wise sort the loop produced.
    from xrspatial.zonal import _sort_and_stride

    rng = np.random.default_rng(3381)
    n_layers, h, w = 4, 5, 6
    values = rng.integers(0, 100, (n_layers, h, w)).astype(np.float64)
    zones = rng.integers(0, 3, (h, w)).astype(np.float64)
    unique_zones = np.unique(zones)

    values_before = values.copy()
    sorted_indices, values_by_zones, _ = _sort_and_stride(
        zones, values, unique_zones
    )

    # input array is not mutated by the reindex
    np.testing.assert_array_equal(values, values_before)

    # each layer is reordered by the same flat zone-sort permutation
    expected = values.reshape(n_layers, h * w)[:, sorted_indices]
    assert values_by_zones.shape == (n_layers, h * w)
    np.testing.assert_array_equal(values_by_zones, expected)


@pytest.mark.skipif(not dask_array_available(), reason="Requires Dask")
def test_crosstab_3d_mixed_backend_raises():
    # 3D crosstab() must reject mismatched backends up front with a clear
    # error, instead of crashing deep inside dask/numba (e.g. with
    # "'NoneType' object is not subscriptable"). See issue #2640.
    zones_data = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=float)
    values_data = np.ones(2 * 4 * 2).reshape(2, 4, 2)

    numpy_zones = xr.DataArray(zones_data, dims=['lat', 'lon'])
    dask_values = xr.DataArray(
        da.from_array(values_data, chunks=(2, 2, 1)),
        dims=['lat', 'lon', 'race'],
    )
    dask_values['race'] = ['c1', 'c2']

    # numpy zones + dask 3D values
    with pytest.raises(ValueError, match="same backend"):
        crosstab(zones=numpy_zones, values=dask_values, agg='count', layer=-1)

    # dask zones + numpy 3D values (reverse direction)
    dask_zones = xr.DataArray(
        da.from_array(zones_data, chunks=(2, 2)), dims=['lat', 'lon']
    )
    numpy_values = xr.DataArray(values_data, dims=['lat', 'lon', 'race'])
    numpy_values['race'] = ['c1', 'c2']
    with pytest.raises(ValueError, match="same backend"):
        crosstab(zones=dask_zones, values=numpy_values, agg='count', layer=-1)


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


@pytest.mark.skipif(not dask_array_available(), reason="Requires Dask")
def test_apply_dask_3d_axis2_rechunked_2526():
    """Regression for #2526: apply 3D dask path must not leave axis-2
    chunks at size 1 after da.stack.
    """
    shape = (4, 4, 3)
    zones_data = np.array([[1, 1, 0, 2],
                           [1, 0, 2, 2],
                           [0, 2, 2, 0],
                           [2, 0, 1, 1]], dtype=np.int32)
    values_data = np.arange(np.prod(shape)).reshape(shape).astype(np.float64)

    zones = xr.DataArray(
        da.from_array(zones_data, chunks=(2, 2)), dims=['y', 'x'],
    )
    values = xr.DataArray(
        da.from_array(values_data, chunks=(2, 2, 3)),
        dims=['y', 'x', 'band'],
    )

    result = apply(zones, values, lambda x: x * 2, nodata=0)

    # Axis-2 chunks should match the input chunking, not be split
    # into unit chunks by da.stack.
    assert result.data.chunks[2] == values.data.chunks[2], (
        f"axis-2 chunks should be {values.data.chunks[2]}, "
        f"got {result.data.chunks[2]}"
    )
    # Sanity: result is numerically correct on a non-nodata cell.
    out = result.data.compute()
    # cell (0, 0) is zone 1 (non-nodata) so func applied
    np.testing.assert_array_equal(
        out[0, 0, :], values_data[0, 0, :] * 2,
    )


@pytest.mark.skipif(not dask_array_available(), reason="Requires Dask")
def test_apply_3d_mixed_backend_raises():
    """Regression for #2639: 3D apply() with mixed dask/numpy zones and
    values must fail early with a clear backend error, not crash with an
    AttributeError or silently return eager numpy output.
    """
    zones_data = np.array([[1, 0],
                           [0, 2]], dtype=np.int32)
    values_data = np.ones((2, 2, 3)) * 5.0

    # numpy zones + dask values: dask backend would hit zones.chunks
    zones_np = xr.DataArray(zones_data, dims=['y', 'x'])
    values_dask = xr.DataArray(
        da.from_array(values_data, chunks=(2, 2, 3)),
        dims=['y', 'x', 'band'],
    )
    with pytest.raises(ValueError, match="same backend"):
        apply(zones_np, values_dask, lambda x: x + 10, nodata=0)

    # dask zones + numpy values: numpy backend would silently go eager
    zones_dask = xr.DataArray(
        da.from_array(zones_data, chunks=(2, 2)), dims=['y', 'x'],
    )
    values_np = xr.DataArray(values_data, dims=['y', 'x', 'band'])
    with pytest.raises(ValueError, match="same backend"):
        apply(zones_dask, values_np, lambda x: x + 10, nodata=0)


def test_apply_nodata_none():
    zones_data = np.array([[0, 1], [2, 3]], dtype=np.int32)
    values_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    zones = xr.DataArray(zones_data, dims=['y', 'x'])
    values = xr.DataArray(values_data, dims=['y', 'x'])

    result = apply(zones, values, lambda x: x * 10, nodata=None)
    expected = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_apply_name_consistent_across_backends(backend):
    # Regression for #2611: apply() left .name unset, so numpy/cupy returned
    # None while the dask backends inherited an internal task name.
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

    # default name is None on every backend
    result = apply(zones, values, lambda x: x * 2, nodata=0)
    assert result.name is None

    # explicit name is honored on every backend
    named = apply(zones, values, lambda x: x * 2, nodata=0, name='doubled')
    assert named.name == 'doubled'


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

    with pytest.raises(ValueError, match="integer dtype"):
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


def _regions_gpu_backends():
    backends = []
    if has_cuda_and_cupy():
        backends.append('cupy')
        if has_dask_array():
            backends.append('dask+cupy')
    return backends


@pytest.mark.parametrize("backend", _regions_gpu_backends())
def test_regions_gpu_matches_numpy(backend):
    """The cupy and dask+cupy backends must match the numpy backend.

    The ``_regions_cupy`` / ``_regions_dask_cupy`` paths (using
    ``cupyx.scipy.ndimage.label``) had no test exercising them, so a
    regression in either could ship undetected. This parametrizes the
    same 4-connectivity case used by ``test_regions_numpy_dask_match``
    over the GPU backends and asserts a cell-by-cell match against the
    numpy reference.
    """
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")
    if 'dask' in backend and not has_dask_array():
        pytest.skip("Requires Dask")

    arr = np.array([[1, 1, 0, 2],
                    [1, 1, 0, 2],
                    [0, 0, 0, 0],
                    [3, 3, 0, 3]], dtype=np.float64)

    numpy_result = regions(_make_regions_raster(arr, 'numpy'), neighborhood=4)

    gpu_raster = _make_regions_raster(arr, backend)
    gpu_result = regions(gpu_raster, neighborhood=4)

    general_output_checks(gpu_raster, gpu_result)

    gpu_data = gpu_result.data
    if da is not None and isinstance(gpu_data, da.Array):
        gpu_data = gpu_data.compute()
    if hasattr(gpu_data, 'get'):
        gpu_data = gpu_data.get()

    np.testing.assert_allclose(
        gpu_data, numpy_result.data, equal_nan=True
    )


def _regions_relabel_reference(data, neighborhood):
    """Per-region relabel: the original O(n_regions * n_cells) semantics.

    Kept here as the reference the vectorized backend remap must match
    byte-for-byte (issue #3207).
    """
    from scipy.ndimage import label

    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        if neighborhood == 4 else np.ones((3, 3), dtype=int)
    )
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


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
@pytest.mark.parametrize("neighborhood", [4, 8])
def test_regions_many_regions_matches_reference_3207(backend, neighborhood):
    """Vectorized relabel equals the per-region reference on a many-region input.

    Random 0/1 noise yields thousands of tiny connected components, the case
    the old per-region scan was slow on.  Every backend must produce the same
    labeling the original loop did: identical unique count, identical id at
    every cell, and distinct ids for distinct regions.
    """
    if 'cupy' in backend and not has_cuda_and_cupy():
        pytest.skip("CUDA and cupy required")

    rng = np.random.default_rng(3207)
    arr = rng.integers(0, 2, size=(48, 48)).astype(np.float64)

    expected = _regions_relabel_reference(arr, neighborhood)

    raster = _make_regions_raster(arr, backend)
    result = regions(raster, neighborhood=neighborhood)

    data = result.data
    if da is not None and isinstance(data, da.Array):
        data = data.compute()
    if not isinstance(data, np.ndarray):  # cupy
        data = data.get()

    # same id at every cell (NaN-aware) and same shape
    np.testing.assert_array_equal(
        np.nan_to_num(data, nan=-1.0), np.nan_to_num(expected, nan=-1.0)
    )
    assert result.shape == arr.shape

    # distinct regions get distinct ids: the labeled cells number 1..N with no
    # gaps, so the count of unique finite ids equals the max id.
    finite = data[~np.isnan(data)]
    assert finite.size > 1000  # this input really does produce many regions
    assert len(np.unique(finite)) == int(finite.max())


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_regions_dask_memory_guard():
    """_regions_dask should raise MemoryError before .compute() on huge arrays."""
    from unittest.mock import patch
    from xrspatial.zonal import _regions_dask

    # Create a dask array that claims to be enormous (shape implies ~80 GB)
    huge = da.zeros((100_000, 100_000), chunks=(1000, 1000), dtype=np.float64)

    # Mock available memory to 1 GB so the guard trips
    with patch('xrspatial.zonal._available_memory_bytes', return_value=1 * 1024**3):
        with pytest.raises(MemoryError, match="Connected-component labeling"):
            _regions_dask(huge.data if hasattr(huge, 'data') else huge, 4)


def test_stats_dataarray_return_type_memory_guard_2523():
    """stats(return_type='xarray.DataArray') should refuse to allocate
    an oversized (n_stats, H*W) float64 working buffer.

    Regression guard for issue #2523: the numpy backend's xarray.DataArray
    return path allocated np.full((n_stats, values.size), nan) with no
    memory check, scaling linearly with the user-supplied stats_funcs.
    """
    from unittest.mock import patch

    zones_arr = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    values_arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                          dtype=np.float64)
    zones = xr.DataArray(zones_arr)
    values = xr.DataArray(values_arr)

    # Mock available memory to a tiny budget so the (n_stats, 8) buffer
    # exceeds 50% of it.  The default 8 stats * 8 cells * 8 bytes = 512 B;
    # with avail = 100 B the guard must trip.
    with patch('xrspatial.zonal._available_memory_bytes', return_value=100):
        with pytest.raises(MemoryError, match="xarray.DataArray"):
            stats(zones=zones, values=values,
                  return_type='xarray.DataArray')

    # Sanity: with the normal memory budget the same call succeeds.
    out = stats(zones=zones, values=values, return_type='xarray.DataArray')
    assert isinstance(out, xr.DataArray)
    # n_stats=8 default, output shape = (8, *values.shape)
    assert out.shape[0] == 8
    assert out.shape[1:] == values_arr.shape


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_stats_dask_zone_filter():
    """stats() with zone_ids filter should return only requested zones."""
    zones = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 3, 3]], dtype=float)
    values = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=float)

    zones_da = xr.DataArray(da.from_array(zones, chunks=(3, 2)), dims=['y', 'x'])
    values_da = xr.DataArray(da.from_array(values, chunks=(3, 2)), dims=['y', 'x'])

    result = stats(zones_da, values_da, zone_ids=[1, 3])
    if hasattr(result, 'compute'):
        result = result.compute()

    assert set(result['zone'].values) == {1.0, 3.0}
    assert 2.0 not in result['zone'].values


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
    result = crop(raster, raster, zone_ids=(1, 3))
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
    result = crop(raster, raster, zone_ids=(0,))
    assert result.shape == arr.shape
    compare = arr == result.data
    assert compare.all()


# ---------------------------------------------------------------------------
# Regression tests for #2521: crop() should accept the canonical zone_ids
# kwarg (matching stats() and crosstab()), with zones_ids kept as a
# deprecated alias.
# ---------------------------------------------------------------------------

def test_crop_zone_ids_canonical_matches_zones_ids_legacy():
    """crop(..., zone_ids=...) produces the same result as the legacy alias."""
    import warnings

    arr = np.array([[0, 4, 0, 3],
                    [0, 4, 4, 3],
                    [0, 1, 1, 3],
                    [0, 1, 1, 3],
                    [0, 0, 0, 0]], dtype=np.int64)
    raster = create_test_arr(arr)

    new = crop(raster, raster, zone_ids=(1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = crop(raster, raster, zones_ids=(1, 3))

    assert new.shape == legacy.shape
    assert (new.data == legacy.data).all()


def test_crop_zones_ids_emits_deprecation_warning():
    """Passing zones_ids must emit a DeprecationWarning."""
    import warnings

    arr = np.array([[0, 4, 0, 3],
                    [0, 4, 4, 3],
                    [0, 1, 1, 3],
                    [0, 1, 1, 3],
                    [0, 0, 0, 0]], dtype=np.int64)
    raster = create_test_arr(arr)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crop(raster, raster, zones_ids=(1, 3))

    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)
           and "zones_ids" in str(w.message)]
    assert len(dep) >= 1, (
        f"Expected a DeprecationWarning mentioning zones_ids, got: "
        f"{[str(w.message) for w in caught]}"
    )


def test_crop_zone_ids_does_not_warn():
    """Passing zone_ids (canonical) must not emit a DeprecationWarning."""
    import warnings

    arr = np.array([[0, 4, 0, 3],
                    [0, 4, 4, 3],
                    [0, 1, 1, 3],
                    [0, 1, 1, 3],
                    [0, 0, 0, 0]], dtype=np.int64)
    raster = create_test_arr(arr)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crop(raster, raster, zone_ids=(1, 3))

    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)
           and "zones_ids" in str(w.message)]
    assert dep == [], (
        f"Expected no DeprecationWarning for zone_ids, got: "
        f"{[str(w.message) for w in dep]}"
    )


def test_crop_both_aliases_raises():
    """Passing both zone_ids and zones_ids must raise TypeError."""
    arr = np.array([[0, 4, 0, 3],
                    [0, 4, 4, 3],
                    [0, 1, 1, 3],
                    [0, 1, 1, 3],
                    [0, 0, 0, 0]], dtype=np.int64)
    raster = create_test_arr(arr)

    with pytest.raises(TypeError, match="zone_ids.*zones_ids|zones_ids.*zone_ids"):
        crop(raster, raster, zone_ids=(1,), zones_ids=(3,))


def test_crop_missing_zone_ids_raises():
    """crop() with neither zone_ids nor zones_ids must raise TypeError."""
    arr = np.array([[0, 4, 0, 3],
                    [0, 4, 4, 3]], dtype=np.int64)
    raster = create_test_arr(arr)

    with pytest.raises(TypeError, match="zone_ids"):
        crop(raster, raster)


def test_crop_mismatched_shapes_raises():
    """Regression test for #2638: crop() must raise when zones and values
    have incompatible shapes instead of silently returning a nonsense crop."""
    zones = create_test_arr(np.array([[1, 1, 2],
                                      [1, 1, 2],
                                      [3, 3, 3]], dtype=np.int64))
    values = create_test_arr(np.array([[10, 20],
                                       [30, 40]], dtype=np.int64))

    with pytest.raises(ValueError, match="equal shapes"):
        crop(zones, values, zone_ids=(1,))


# ---------------------------------------------------------------------------
# Regression tests for #2561: crop() must return the same shape across all
# backends when the requested zone_ids are absent (previously the dask path
# returned the full input raster while numpy/cupy returned an empty (0, 0)).
# ---------------------------------------------------------------------------

_CROP_ARR_2561 = np.array([[0, 4, 0, 3],
                           [0, 4, 4, 3],
                           [0, 1, 1, 3],
                           [0, 1, 1, 3],
                           [0, 0, 0, 0]], dtype=np.int64)


def _crop_backends_2561():
    backends = ['numpy']
    if has_dask_array():
        backends.append('dask+numpy')
    if has_cuda_and_cupy():
        backends.append('cupy')
        if has_dask_array():
            backends.append('dask+cupy')
    return backends


@pytest.mark.parametrize("backend", _crop_backends_2561())
def test_crop_absent_zone_ids_returns_empty_2561(backend):
    """All backends must return shape (0, 0) when no requested zone exists."""
    zones = create_test_raster(_CROP_ARR_2561, backend=backend)
    values = create_test_raster(_CROP_ARR_2561, backend=backend)

    # zone_id 99 is not present anywhere in the raster.
    result = crop(zones, values, zone_ids=(99,))

    # Output array type must match input type.
    assert isinstance(result.data, type(zones.data))
    assert result.shape == (0, 0)


@pytest.mark.parametrize("backend", _crop_backends_2561())
def test_crop_partial_zone_ids_2561(backend):
    """Mix of present and absent zone_ids crops to the present ones only."""
    zones = create_test_raster(_CROP_ARR_2561, backend=backend)
    values = create_test_raster(_CROP_ARR_2561, backend=backend)

    # Only zone_id 1 is actually present here. 77 and 88 are absent.
    result_partial = crop(zones, values, zone_ids=(77, 1, 88))
    result_present_only = crop(zones, values, zone_ids=(1,))

    # Pin the expected bounding box of zone_id=1 in _CROP_ARR_2561
    # (rows 2-3, cols 1-2) so a regression cannot pass by drifting
    # both code paths in the same direction.
    assert result_partial.shape == (2, 2)
    assert result_partial.shape == result_present_only.shape

    if backend.startswith('dask'):
        partial_np = result_partial.data.compute()
        present_np = result_present_only.data.compute()
    else:
        partial_np = result_partial.data
        present_np = result_present_only.data

    if 'cupy' in backend:
        partial_np = partial_np.get()
        present_np = present_np.get()

    np.testing.assert_array_equal(partial_np, present_np)


@pytest.mark.parametrize("backend", _crop_backends_2561())
def test_crop_happy_path_matches_numpy_2561(backend):
    """All backends agree on the standard crop result."""
    zones = create_test_raster(_CROP_ARR_2561, backend=backend)
    values = create_test_raster(_CROP_ARR_2561, backend=backend)

    result = crop(zones, values, zone_ids=(1, 3))
    assert result.shape == (4, 3)

    expected = np.array([[4, 0, 3],
                         [4, 4, 3],
                         [1, 1, 3],
                         [1, 1, 3]], dtype=np.int64)

    if backend.startswith('dask'):
        result_np = result.data.compute()
    else:
        result_np = result.data

    if 'cupy' in backend:
        result_np = result_np.get()

    np.testing.assert_array_equal(result_np, expected)


# ---------------------------------------------------------------------------
# Regression tests for #2528: dask backend must not silently drop 'majority'
# from the requested stats list.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_dask_array(), reason="dask.array not available")
def test_stats_dask_explicit_majority_raises_2528():
    """Explicit 'majority' on dask must raise instead of being silently dropped."""
    zones_data = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=float)
    values_data = np.array([[1.0, 1, 2, 2], [3, 3, 5, 5]])

    zones = xr.DataArray(da.from_array(zones_data, chunks=(2, 2)), dims=['y', 'x'])
    values = xr.DataArray(da.from_array(values_data, chunks=(2, 2)), dims=['y', 'x'])

    with pytest.raises(ValueError, match="majority"):
        stats(zones=zones, values=values, stats_funcs=['mean', 'majority'])

    # Single-stat majority request also raises.
    with pytest.raises(ValueError, match="majority"):
        stats(zones=zones, values=values, stats_funcs=['majority'])


@pytest.mark.skipif(not has_dask_array(), reason="dask.array not available")
def test_stats_dask_default_omits_majority_2528():
    """Bare stats() on dask should resolve to the dask default (no majority).

    Regression for #2528: the previous mutable default included 'majority'
    and the dask path silently dropped it.  After the fix the dask default
    resolves to the supported subset, so the result columns match the
    documented contract and no stat is silently filtered.
    """
    zones_data = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=float)
    values_data = np.array([[1.0, 1, 2, 2], [3, 3, 5, 5]])

    zones = xr.DataArray(da.from_array(zones_data, chunks=(2, 2)), dims=['y', 'x'])
    values = xr.DataArray(da.from_array(values_data, chunks=(2, 2)), dims=['y', 'x'])

    df = stats(zones=zones, values=values).compute()
    expected_cols = ['zone', 'mean', 'max', 'min', 'sum', 'std', 'var', 'count']
    assert df.columns.tolist() == expected_cols
    assert 'majority' not in df.columns


def test_stats_numpy_default_includes_majority_2528():
    """Bare stats() on numpy keeps 'majority' in the resolved default list."""
    zones = xr.DataArray(np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=float),
                         dims=['y', 'x'])
    values = xr.DataArray(np.array([[1.0, 1, 2, 2], [3, 3, 5, 5]]),
                          dims=['y', 'x'])

    df = stats(zones=zones, values=values)
    assert 'majority' in df.columns


def test_stats_default_list_is_not_mutated_2528():
    """Repeated calls with the default must not accumulate state.

    The previous implementation used a mutable list literal as the default
    argument.  After the switch to ``stats_funcs=None``, the resolved list
    is freshly constructed each call, so a caller that mutates it locally
    cannot leak state into the next caller.
    """
    zones = xr.DataArray(np.array([[1, 1], [2, 2]], dtype=float), dims=['y', 'x'])
    values = xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]]), dims=['y', 'x'])

    df1 = stats(zones=zones, values=values)
    df2 = stats(zones=zones, values=values)
    assert df1.columns.tolist() == df2.columns.tolist()
    assert 'majority' in df1.columns and 'majority' in df2.columns


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


# ---------------------------------------------------------------------------
# Vector zones (GeoDataFrame / list-of-pairs) — implicit rasterization
# ---------------------------------------------------------------------------

try:
    from shapely.geometry import box as shapely_box
    import geopandas as gpd
    _has_vector = True
except ImportError:
    _has_vector = False

skip_no_vector = pytest.mark.skipif(
    not _has_vector, reason="shapely/geopandas not installed"
)


def _make_grid(width=20, height=20, bounds=(0, 0, 100, 100)):
    """Build a template DataArray with pixel-centre coords."""
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height
    x = np.linspace(xmin + px / 2, xmax - px / 2, width)
    y = np.linspace(ymax - py / 2, ymin + py / 2, height)
    return xr.DataArray(
        np.ones((height, width), dtype=np.float64),
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
    )


@skip_no_vector
class TestVectorZones:
    """Verify that zonal functions accept vector zones directly."""

    def _zones_raster_and_gdf(self):
        """Two non-overlapping boxes covering different quadrants."""
        from xrspatial.rasterize import rasterize

        values = _make_grid()
        gdf = gpd.GeoDataFrame(
            {'zone_id': [1.0, 2.0]},
            geometry=[
                shapely_box(0, 50, 50, 100),   # top-left
                shapely_box(50, 0, 100, 50),    # bottom-right
            ],
        )
        zones_raster = rasterize(gdf, like=values, column='zone_id')
        return values, gdf, zones_raster

    # -- stats --

    def test_stats_gdf_matches_raster(self):
        values, gdf, zones_raster = self._zones_raster_and_gdf()
        expected = stats(zones_raster, values, stats_funcs=['mean', 'count'])
        result = stats(gdf, values, stats_funcs=['mean', 'count'],
                       column='zone_id')
        pd.testing.assert_frame_equal(result, expected)

    def test_stats_list_of_pairs(self):
        values, gdf, zones_raster = self._zones_raster_and_gdf()
        pairs = [
            (shapely_box(0, 50, 50, 100), 1.0),
            (shapely_box(50, 0, 100, 50), 2.0),
        ]
        expected = stats(zones_raster, values, stats_funcs=['mean', 'count'])
        result = stats(pairs, values, stats_funcs=['mean', 'count'])
        pd.testing.assert_frame_equal(result, expected)

    def test_stats_gdf_missing_column_raises(self):
        values = _make_grid()
        gdf = gpd.GeoDataFrame(
            {'zone_id': [1.0]},
            geometry=[shapely_box(0, 0, 50, 50)],
        )
        with pytest.raises(ValueError, match="column is required"):
            stats(gdf, values)

    def test_stats_pairs_with_column_raises(self):
        values = _make_grid()
        pairs = [(shapely_box(0, 0, 50, 50), 1.0)]
        with pytest.raises(ValueError, match="column should not be set"):
            stats(pairs, values, column='zone_id')

    def test_stats_accessor(self):
        import xrspatial  # noqa: F401  — registers accessor
        values, gdf, zones_raster = self._zones_raster_and_gdf()
        expected = stats(zones_raster, values, stats_funcs=['mean', 'count'])
        result = values.xrs.zonal_stats(gdf, stats_funcs=['mean', 'count'],
                                        column='zone_id')
        pd.testing.assert_frame_equal(result, expected)

    # -- crosstab --

    def test_crosstab_gdf(self):
        values, gdf, zones_raster = self._zones_raster_and_gdf()
        expected = crosstab(zones_raster, values)
        result = crosstab(gdf, values, column='zone_id')
        pd.testing.assert_frame_equal(result, expected)

    # -- apply --

    def test_apply_gdf(self):
        values, gdf, zones_raster = self._zones_raster_and_gdf()
        # rasterize produces float zones; apply needs int zones.
        # Substitute the default NaN fill with 0 (apply's default nodata)
        # before casting so the cast is well-defined across platforms;
        # rasterize(..., dtype=int) now requires an explicit integer fill
        # (#2504), so pass fill=0 to match.
        zones_float = zones_raster.copy(
            data=np.where(np.isnan(zones_raster.values),
                          0.0, zones_raster.values))
        zones_int = zones_float.copy(data=zones_float.values.astype(int))
        fn = lambda x: x * 2
        expected = apply(zones_int, values, fn)
        result = apply(gdf, values, fn, column='zone_id',
                       rasterize_kw={'dtype': int, 'fill': 0})
        xr.testing.assert_identical(result, expected)

    # -- crop --

    def test_crop_gdf(self):
        values, gdf, zones_raster = self._zones_raster_and_gdf()
        expected = crop(zones_raster, values, zone_ids=[1.0])
        result = crop(gdf, values, zone_ids=[1.0], column='zone_id')
        xr.testing.assert_identical(result, expected)

    # -- rasterize_kw forwarding --

    def test_stats_rasterize_kw_all_touched(self):
        values, gdf, _ = self._zones_raster_and_gdf()
        from xrspatial.rasterize import rasterize
        zones_at = rasterize(gdf, like=values, column='zone_id',
                             all_touched=True)
        expected = stats(zones_at, values, stats_funcs=['count'])
        result = stats(gdf, values, stats_funcs=['count'],
                       column='zone_id',
                       rasterize_kw={'all_touched': True})
        pd.testing.assert_frame_equal(result, expected)

    # -- raster zones still work --

    def test_raster_zones_unchanged(self):
        """Passing a DataArray directly should still work as before."""
        values, _, zones_raster = self._zones_raster_and_gdf()
        result = stats(zones_raster, values, stats_funcs=['count'])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# stats(return_type=...) validation -- issue #2558
# ---------------------------------------------------------------------------


@pytest.fixture
def small_zones_values_2558():
    zones = xr.DataArray(np.array([[0, 0, 1, 1],
                                   [0, 0, 1, 1]], dtype=np.int32))
    values = xr.DataArray(np.array([[1.0, 2.0, 3.0, 4.0],
                                    [5.0, 6.0, 7.0, 8.0]], dtype=np.float64))
    return zones, values


def test_stats_return_type_default_is_dataframe_2558(small_zones_values_2558):
    """Default return_type should still produce a pandas.DataFrame."""
    zones, values = small_zones_values_2558
    result = stats(zones=zones, values=values)
    assert isinstance(result, pd.DataFrame)
    assert 'zone' in result.columns


def test_stats_return_type_pandas_dataframe_2558(small_zones_values_2558):
    """Explicit 'pandas.DataFrame' should produce a pandas.DataFrame."""
    zones, values = small_zones_values_2558
    result = stats(zones=zones, values=values, return_type='pandas.DataFrame')
    assert isinstance(result, pd.DataFrame)


def test_stats_return_type_xarray_dataarray_2558(small_zones_values_2558):
    """Explicit 'xarray.DataArray' should produce an xarray.DataArray."""
    zones, values = small_zones_values_2558
    result = stats(zones=zones, values=values, return_type='xarray.DataArray')
    assert isinstance(result, xr.DataArray)
    assert result.dims[0] == 'stats'
    assert result.shape[1:] == values.shape


@pytest.mark.parametrize("bad", ['bogus', 'numpy.ndarray', '', 'DataFrame',
                                 'xarray.dataarray'])
def test_stats_return_type_invalid_raises_2558(small_zones_values_2558, bad):
    """Unknown return_type values should raise ValueError, not silently
    return a numpy.ndarray (regression for issue #2558)."""
    zones, values = small_zones_values_2558
    with pytest.raises(ValueError, match="return_type"):
        stats(zones=zones, values=values, return_type=bad)


def test_stats_return_type_invalid_message_lists_allowed_2558(
        small_zones_values_2558):
    """The ValueError message should enumerate the allowed values."""
    zones, values = small_zones_values_2558
    with pytest.raises(ValueError) as excinfo:
        stats(zones=zones, values=values, return_type='bogus')
    msg = str(excinfo.value)
    assert 'pandas.DataFrame' in msg
    assert 'xarray.DataArray' in msg


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_stats_return_type_dataarray_rejected_for_dask_2558():
    """'xarray.DataArray' is documented as numpy-only; dask input should
    raise ValueError instead of silently doing the wrong thing."""
    zones_arr = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    values_arr = np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    zones = xr.DataArray(da.from_array(zones_arr, chunks=(2, 2)))
    values = xr.DataArray(da.from_array(values_arr, chunks=(2, 2)))
    with pytest.raises(ValueError, match="xarray.DataArray"):
        stats(zones=zones, values=values, return_type='xarray.DataArray')


@pytest.mark.skipif(not has_cuda_and_cupy(), reason="Requires CUDA and CuPy")
def test_stats_return_type_dataarray_rejected_for_cupy_2558():
    """'xarray.DataArray' is documented as numpy-only; cupy input should
    raise ValueError instead of crashing inside the xr.DataArray wrapper."""
    import cupy
    zones_arr = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    values_arr = np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    zones = xr.DataArray(cupy.asarray(zones_arr))
    values = xr.DataArray(cupy.asarray(values_arr))
    with pytest.raises(ValueError, match="xarray.DataArray"):
        stats(zones=zones, values=values, return_type='xarray.DataArray')


def test_stats_return_type_invalid_rejected_for_dataset_2558(
        small_zones_values_2558):
    """Invalid return_type on Dataset input should raise ValueError
    (the entry-point validator runs before the Dataset branch)."""
    zones, values = small_zones_values_2558
    ds = xr.Dataset({'v': values})
    with pytest.raises(ValueError, match="return_type"):
        stats(zones=zones, values=ds, return_type='bogus')


def test_stats_empty_dataset_raises_value_error_2637(small_zones_values_2558):
    """An empty Dataset (no data variables) should raise a clear ValueError
    instead of leaking an IndexError from the dfs[0] access."""
    zones, _ = small_zones_values_2558
    empty = xr.Dataset()
    with pytest.raises(ValueError, match="no data variables"):
        stats(zones=zones, values=empty)


def test_strides_int64_no_overflow_2612():
    """`_strides` must accumulate counts in int64, not int32.

    The counter runs up to the flattened pixel count. In an int32 array
    inside the numba kernel, a count above 2**31-1 wraps silently to a
    negative value, corrupting the slice bounds that `_calc_stats` and
    crosstab derive from it. See issue #2612.
    """
    from xrspatial.zonal import _strides

    # A real >2**31-element input would need ~17 GB to allocate, so the
    # dtype assertion below stands in for the actual overflow case; a
    # small input is enough to lock in the int64 contract and correctness.
    flatten_zones = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    unique_zones = np.array([0, 1, 2], dtype=np.int64)

    strides = _strides(flatten_zones, unique_zones)

    # The result dtype is the load-bearing guarantee: int64 cannot wrap
    # at realistic raster sizes (~2.1 billion pixels) the way int32 does.
    assert strides.dtype == np.int64

    # Cumulative boundaries stay non-negative and non-decreasing, and the
    # final stride equals the number of finite elements.
    expected = np.array([3, 5, 6], dtype=np.int64)
    np.testing.assert_array_equal(strides, expected)
    assert np.all(strides >= 0)
    assert np.all(np.diff(strides) >= 0)
