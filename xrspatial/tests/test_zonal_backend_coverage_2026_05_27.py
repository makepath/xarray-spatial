"""Backend coverage tests for xrspatial.zonal.

Closes HIGH-severity backend-coverage gaps surfaced by the test-coverage
sweep on 2026-05-27 (deep-sweep-test-coverage-zonal-2026-05-27).

Module ``xrspatial/zonal.py`` registers four-backend dispatchers
(``ArrayTypeFunctionMapping``) for ``crosstab``, ``regions``, ``apply``
(3D), and also has dedicated cupy / dask branches inside ``trim`` and
``crop``. Prior to this file the existing ``test_zonal.py`` only
exercised the numpy + dask+numpy paths for crosstab/regions/trim/crop
and the 2D-only path for apply on cupy backends. A regression on the
``_crosstab_cupy``, ``_crosstab_dask_cupy``, ``_regions_cupy``,
``_regions_dask_cupy``, ``_trim_bounds_dask``, ``_crop_bounds_dask``,
or 3D cupy ``apply`` branches would ship undetected.

Tests in this file:

- Cat 1 HIGH: crosstab cupy + dask+cupy parity vs numpy
- Cat 1 HIGH: regions cupy + dask+cupy parity vs numpy
- Cat 1 HIGH: trim cupy + dask+numpy + dask+cupy parity vs numpy
- Cat 1 HIGH: crop cupy + dask+numpy + dask+cupy parity vs numpy
- Cat 1 HIGH: apply 3D cupy + dask+cupy parity vs numpy
- Cat 3 MEDIUM: 1x1 single-pixel raster on regions/trim/crop
- Cat 3 MEDIUM: Nx1 / 1xN strip on regions/trim
- Cat 4 LOW: regions invalid neighborhood ValueError pin
- Cat 4 LOW: suggest_zonal_canvas Geographic CRS pin
- Cat 5 MEDIUM: trim/crop attrs preservation

CUDA was available on the host when this file was authored (2026-05-27);
tests parametrized over ``cupy``/``dask+cupy`` execute live unless the
GPU-skip decorator skips them on a non-CUDA host.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial.zonal import (
    apply, crop, crosstab, regions, suggest_zonal_canvas, trim,
)

from .general_checks import (
    create_test_raster, cuda_and_cupy_available, dask_array_available,
    has_dask_array,
)

# Local alias kept for readability at decorator sites in this file.
dask_required = dask_array_available


def _to_numpy(arr):
    """Bring a DataArray-or-array result to a numpy array."""
    if hasattr(arr, 'data'):
        arr = arr.data
    if da is not None and isinstance(arr, da.Array):
        arr = arr.compute()
    if hasattr(arr, 'get'):  # cupy
        arr = arr.get()
    return np.asarray(arr)


def _to_pandas(df):
    """Bring a (dask) DataFrame to pandas."""
    if hasattr(df, 'compute'):
        return df.compute()
    return df


def _canonical_labels(a):
    """Re-label a region map so labels are assigned in raster-scan order.

    scipy.ndimage.label and cupyx.scipy.ndimage.label may emit labels in
    different orders for the same input.  Canonicalising lets the parity
    tests compare cell partitions, not raw label values.
    """
    out = np.full_like(a, -1, dtype=np.int64)
    seen = {}
    next_id = 0
    flat = a.ravel()
    for i, v in enumerate(flat):
        if not np.isfinite(v):
            continue
        key = float(v)
        if key not in seen:
            seen[key] = next_id
            next_id += 1
        out.ravel()[i] = seen[key]
    return out


# ---------------------------------------------------------------------------
# Cat 1 HIGH -- crosstab backend coverage (cupy / dask+cupy)
# ---------------------------------------------------------------------------

@pytest.fixture
def crosstab_zones_values():
    """Small zones + values fixture for crosstab parity tests."""
    zones = np.array(
        [[1, 1, 2, 2],
         [1, 1, 2, 2],
         [3, 3, 4, 4]],
        dtype=np.float64,
    )
    values = np.array(
        [[10, 10, 20, 20],
         [10, 20, 20, 30],
         [30, 30, 10, 10]],
        dtype=np.float64,
    )
    return zones, values


@cuda_and_cupy_available
def test_crosstab_cupy_matches_numpy(crosstab_zones_values):
    """_crosstab_cupy parity vs _crosstab_numpy (Cat 1 HIGH)."""
    import cupy as cp
    zones_np, values_np = crosstab_zones_values

    zones_n = xr.DataArray(zones_np, dims=['y', 'x'])
    values_n = xr.DataArray(values_np, dims=['y', 'x'])
    df_np = crosstab(zones_n, values_n)

    zones_c = xr.DataArray(cp.asarray(zones_np), dims=['y', 'x'])
    values_c = xr.DataArray(cp.asarray(values_np), dims=['y', 'x'])
    df_cp = crosstab(zones_c, values_c)

    # cupy path returns a pandas DataFrame after _crosstab_cupy converts back
    assert isinstance(df_cp, pd.DataFrame)
    pd.testing.assert_frame_equal(
        df_cp.sort_values('zone').reset_index(drop=True),
        df_np.sort_values('zone').reset_index(drop=True),
        check_dtype=False,
    )


@cuda_and_cupy_available
@dask_required
def test_crosstab_dask_cupy_matches_numpy(crosstab_zones_values):
    """_crosstab_dask_cupy parity vs _crosstab_numpy (Cat 1 HIGH)."""
    import cupy as cp
    zones_np, values_np = crosstab_zones_values

    zones_n = xr.DataArray(zones_np, dims=['y', 'x'])
    values_n = xr.DataArray(values_np, dims=['y', 'x'])
    df_np = crosstab(zones_n, values_n)

    zones_dc = xr.DataArray(
        da.from_array(cp.asarray(zones_np), chunks=(2, 2)),
        dims=['y', 'x'],
    )
    values_dc = xr.DataArray(
        da.from_array(cp.asarray(values_np), chunks=(2, 2)),
        dims=['y', 'x'],
    )
    df_dc = _to_pandas(crosstab(zones_dc, values_dc))

    pd.testing.assert_frame_equal(
        df_dc.sort_values('zone').reset_index(drop=True),
        df_np.sort_values('zone').reset_index(drop=True),
        check_dtype=False,
    )


@cuda_and_cupy_available
def test_crosstab_cupy_percentage(crosstab_zones_values):
    """crosstab cupy with agg='percentage' (Cat 4 MEDIUM)."""
    import cupy as cp
    zones_np, values_np = crosstab_zones_values

    zones_n = xr.DataArray(zones_np, dims=['y', 'x'])
    values_n = xr.DataArray(values_np, dims=['y', 'x'])
    df_np = crosstab(zones_n, values_n, agg='percentage')

    zones_c = xr.DataArray(cp.asarray(zones_np), dims=['y', 'x'])
    values_c = xr.DataArray(cp.asarray(values_np), dims=['y', 'x'])
    df_cp = crosstab(zones_c, values_c, agg='percentage')

    pd.testing.assert_frame_equal(
        df_cp.sort_values('zone').reset_index(drop=True),
        df_np.sort_values('zone').reset_index(drop=True),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Cat 1 HIGH -- regions backend coverage (cupy / dask+cupy)
# ---------------------------------------------------------------------------

@pytest.fixture
def regions_input():
    return np.array(
        [[1, 1, 0, 2, 2],
         [1, 1, 0, 2, 2],
         [0, 0, 0, 0, 0],
         [3, 3, 0, 3, 3],
         [3, 3, 0, 3, 3]],
        dtype=np.float64,
    )


@cuda_and_cupy_available
def test_regions_cupy_matches_numpy(regions_input):
    """_regions_cupy parity vs _regions_numpy (Cat 1 HIGH)."""
    import cupy as cp

    arr_np = xr.DataArray(regions_input, dims=['y', 'x'])
    arr_cp = xr.DataArray(cp.asarray(regions_input), dims=['y', 'x'])

    out_np = _to_numpy(regions(arr_np, neighborhood=4))
    out_cp = _to_numpy(regions(arr_cp, neighborhood=4))

    # Labels may differ between scipy / cupyx; partitions must match.
    np.testing.assert_array_equal(
        _canonical_labels(out_np), _canonical_labels(out_cp),
    )


@cuda_and_cupy_available
@dask_required
def test_regions_dask_cupy_matches_numpy(regions_input):
    """_regions_dask_cupy parity vs _regions_numpy (Cat 1 HIGH)."""
    import cupy as cp

    arr_np = xr.DataArray(regions_input, dims=['y', 'x'])
    arr_dc = xr.DataArray(
        da.from_array(cp.asarray(regions_input), chunks=(3, 3)),
        dims=['y', 'x'],
    )

    out_np = _to_numpy(regions(arr_np, neighborhood=4))
    out_dc = _to_numpy(regions(arr_dc, neighborhood=4))

    np.testing.assert_array_equal(
        _canonical_labels(out_np), _canonical_labels(out_dc),
    )


@cuda_and_cupy_available
def test_regions_cupy_eight_connectivity():
    """8-connectivity branch of _regions_cupy (Cat 4 MEDIUM)."""
    import cupy as cp

    diag = np.array(
        [[1, 0, 1],
         [0, 1, 0],
         [1, 0, 1]],
        dtype=np.float64,
    )
    arr_cp = xr.DataArray(cp.asarray(diag), dims=['y', 'x'])
    out = _to_numpy(regions(arr_cp, neighborhood=8))
    # 8-connected: all 1s merge into one region, all 0s into another
    finite_labels = out[np.isfinite(out)]
    assert len(np.unique(finite_labels)) == 2


def test_regions_invalid_neighborhood_raises():
    """Cat 4 LOW: regions(neighborhood=6) must raise ValueError."""
    arr = xr.DataArray(np.array([[1, 1], [1, 1]], dtype=np.float64), dims=['y', 'x'])
    with pytest.raises(ValueError, match="neighborhood"):
        regions(arr, neighborhood=6)


# ---------------------------------------------------------------------------
# Cat 1 HIGH -- trim backend coverage (cupy / dask+numpy / dask+cupy)
# ---------------------------------------------------------------------------

@pytest.fixture
def trim_input():
    return np.array(
        [[0, 0, 0, 0],
         [0, 4, 0, 0],
         [0, 4, 4, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        dtype=np.float64,
    )


@cuda_and_cupy_available
def test_trim_cupy_matches_numpy(trim_input):
    """trim cupy branch (data.get() + _trim) parity (Cat 1 HIGH)."""
    import cupy as cp

    arr_np = xr.DataArray(trim_input, dims=['y', 'x'])
    arr_cp = xr.DataArray(cp.asarray(trim_input), dims=['y', 'x'])

    out_np = trim(arr_np, values=(0.0,))
    out_cp = trim(arr_cp, values=(0.0,))

    assert out_cp.shape == out_np.shape == (3, 2)
    np.testing.assert_array_equal(_to_numpy(out_cp), _to_numpy(out_np))


@dask_required
def test_trim_dask_numpy_matches_numpy(trim_input):
    """trim dask path (_trim_bounds_dask) parity (Cat 1 HIGH)."""
    arr_np = xr.DataArray(trim_input, dims=['y', 'x'])
    arr_da = xr.DataArray(
        da.from_array(trim_input, chunks=(3, 2)), dims=['y', 'x'],
    )

    out_np = trim(arr_np, values=(0.0,))
    out_da = trim(arr_da, values=(0.0,))

    assert out_da.shape == out_np.shape == (3, 2)
    np.testing.assert_array_equal(_to_numpy(out_da), _to_numpy(out_np))


@cuda_and_cupy_available
@dask_required
def test_trim_dask_cupy_matches_numpy(trim_input):
    """trim dask+cupy parity (Cat 1 HIGH)."""
    import cupy as cp

    arr_np = xr.DataArray(trim_input, dims=['y', 'x'])
    arr_dc = xr.DataArray(
        da.from_array(cp.asarray(trim_input), chunks=(3, 2)),
        dims=['y', 'x'],
    )

    out_np = trim(arr_np, values=(0.0,))
    out_dc = trim(arr_dc, values=(0.0,))

    assert out_dc.shape == out_np.shape == (3, 2)
    np.testing.assert_array_equal(_to_numpy(out_dc), _to_numpy(out_np))


@dask_required
def test_trim_dask_nan_values(trim_input):
    """trim with a NaN sentinel agrees across numpy and dask backends.

    Originally pinned a numpy/dask asymmetry: ``_trim``'s ``e == val``
    check never matched NaN, so the numpy path left a NaN-framed
    raster unchanged while dask trimmed it. Fixed in #2559 by routing
    NaN sentinels through ``_trim_bounds_numpy`` in the wrapper.
    """
    arr_with_nan = np.where(trim_input == 0, np.nan, trim_input)

    arr_np = xr.DataArray(arr_with_nan, dims=['y', 'x'])
    arr_da = xr.DataArray(
        da.from_array(arr_with_nan, chunks=(3, 2)), dims=['y', 'x'],
    )

    # Bounding box covers rows 1-3, cols 1-2 of the input.
    # Interior NaNs (the original 0 in the middle of row 1) are
    # preserved.
    expected = np.array([[4.0, np.nan],
                         [4.0, 4.0],
                         [1.0, 1.0]])

    out_da = _to_numpy(trim(arr_da, values=(np.nan,)))
    out_np = _to_numpy(trim(arr_np, values=(np.nan,)))

    assert out_da.shape == (3, 2)
    assert out_np.shape == (3, 2)
    np.testing.assert_array_equal(out_da, expected)
    np.testing.assert_array_equal(out_np, expected)


def test_trim_preserves_name_attribute():
    """Cat 5 MEDIUM: trim should set name to 'trim' by default."""
    arr = xr.DataArray(
        np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=np.float64),
        dims=['y', 'x'],
        attrs={'res': (1.0, 1.0), 'crs': 'EPSG:4326'},
    )
    out = trim(arr, values=(0.0,))
    assert out.name == 'trim'
    # attrs propagated from input
    assert out.attrs.get('res') == (1.0, 1.0)
    assert out.attrs.get('crs') == 'EPSG:4326'


# ---------------------------------------------------------------------------
# trim() with default NaN sentinel -- cross-backend agreement (#2559)
# ---------------------------------------------------------------------------

@pytest.fixture
def trim_nan_input():
    return np.array(
        [[np.nan, np.nan, np.nan, np.nan],
         [np.nan, 4.0,    np.nan, np.nan],
         [np.nan, 4.0,    4.0,    np.nan],
         [np.nan, 1.0,    1.0,    np.nan],
         [np.nan, np.nan, np.nan, np.nan]],
        dtype=np.float64,
    )


def test_trim_numpy_default_nan(trim_nan_input):
    """Default values=(np.nan,) trims a NaN-framed numpy raster (#2559).

    Pre-fix this returned the input unchanged because ``_trim`` matched
    sentinels with ``e == val`` and ``NaN == NaN`` is False.
    """
    arr = xr.DataArray(trim_nan_input, dims=['y', 'x'])

    out = trim(arr)  # default values=(np.nan,)
    out_np = _to_numpy(out)

    expected = np.array([[4.0, np.nan],
                         [4.0, 4.0],
                         [1.0, 1.0]])
    assert out_np.shape == (3, 2)
    np.testing.assert_array_equal(out_np, expected)


def test_trim_numpy_explicit_nan_matches_default(trim_nan_input):
    """Passing values=(np.nan,) explicitly matches the default (#2559)."""
    arr = xr.DataArray(trim_nan_input, dims=['y', 'x'])

    out_default = _to_numpy(trim(arr))
    out_explicit = _to_numpy(trim(arr, values=(np.nan,)))

    np.testing.assert_array_equal(out_default, out_explicit)


@cuda_and_cupy_available
def test_trim_cupy_default_nan_matches_numpy(trim_nan_input):
    """cupy backend trims a NaN-framed raster identically to numpy (#2559)."""
    import cupy as cp

    arr_np = xr.DataArray(trim_nan_input, dims=['y', 'x'])
    arr_cp = xr.DataArray(cp.asarray(trim_nan_input), dims=['y', 'x'])

    out_np = _to_numpy(trim(arr_np))
    out_cp = _to_numpy(trim(arr_cp))

    assert out_cp.shape == out_np.shape == (3, 2)
    np.testing.assert_array_equal(out_cp, out_np)


@dask_required
def test_trim_dask_numpy_default_nan_matches_numpy(trim_nan_input):
    """dask+numpy backend trims a NaN-framed raster identically to numpy (#2559)."""
    arr_np = xr.DataArray(trim_nan_input, dims=['y', 'x'])
    arr_da = xr.DataArray(
        da.from_array(trim_nan_input, chunks=(3, 2)), dims=['y', 'x'],
    )

    out_np = _to_numpy(trim(arr_np))
    out_da = _to_numpy(trim(arr_da))

    assert out_da.shape == out_np.shape == (3, 2)
    np.testing.assert_array_equal(out_da, out_np)


@cuda_and_cupy_available
@dask_required
def test_trim_dask_cupy_default_nan_matches_numpy(trim_nan_input):
    """dask+cupy backend trims a NaN-framed raster identically to numpy (#2559)."""
    import cupy as cp

    arr_np = xr.DataArray(trim_nan_input, dims=['y', 'x'])
    arr_dc = xr.DataArray(
        da.from_array(cp.asarray(trim_nan_input), chunks=(3, 2)),
        dims=['y', 'x'],
    )

    out_np = _to_numpy(trim(arr_np))
    out_dc = _to_numpy(trim(arr_dc))

    assert out_dc.shape == out_np.shape == (3, 2)
    np.testing.assert_array_equal(out_dc, out_np)


def test_trim_numpy_mixed_nan_and_finite_sentinels():
    """Passing both NaN and a finite sentinel trims either kind of border (#2559)."""
    data = np.array(
        [[0.0,    0.0, 0.0,    0.0],
         [0.0,    5.0, np.nan, 0.0],
         [np.nan, 5.0, 5.0,    np.nan],
         [0.0,    0.0, 0.0,    0.0]],
        dtype=np.float64,
    )
    arr = xr.DataArray(data, dims=['y', 'x'])

    out = _to_numpy(trim(arr, values=(0.0, np.nan)))

    # Rows 0 and 3 are all-zero -> trimmed.
    # Row 1 col 2 is NaN, row 2 cols 0 and 3 are NaN -> all trimmable
    # via the mixed-sentinel rule. The first non-trimmable row is 1
    # (because of the 5.0), last is 2. Same for cols (1, 2).
    expected = np.array([[5.0, np.nan],
                         [5.0, 5.0]])
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, expected)


def test_trim_numpy_integer_dtype_non_nan_sentinel_still_works():
    """Integer-dtype input with a non-NaN sentinel goes through the numba
    kernel and trims correctly (#2559 regression guard)."""
    data = np.array(
        [[0, 0, 0, 0, 0],
         [0, 0, 7, 0, 0],
         [0, 0, 7, 7, 0],
         [0, 0, 0, 0, 0]],
        dtype=np.int32,
    )
    arr = xr.DataArray(data, dims=['y', 'x'])

    out = _to_numpy(trim(arr, values=(0,)))

    expected = np.array([[7, 0],
                         [7, 7]], dtype=np.int32)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, expected)


def test_trim_numpy_all_nan_input():
    """trim() of an all-NaN raster returns an empty slice on numpy (#2559)."""
    arr = xr.DataArray(np.full((4, 4), np.nan), dims=['y', 'x'])

    out = _to_numpy(trim(arr))

    assert out.size == 0


# ---------------------------------------------------------------------------
# Cat 1 HIGH -- crop backend coverage (cupy / dask+numpy / dask+cupy)
# ---------------------------------------------------------------------------

@pytest.fixture
def crop_input():
    arr = np.array(
        [[0, 4, 0, 3],
         [0, 4, 4, 3],
         [0, 1, 1, 3],
         [0, 1, 1, 3],
         [0, 0, 0, 0]],
        dtype=np.float64,
    )
    return arr


@cuda_and_cupy_available
def test_crop_cupy_matches_numpy(crop_input):
    """crop cupy branch (data.get() + _crop) parity (Cat 1 HIGH)."""
    import cupy as cp

    zones_np = xr.DataArray(crop_input, dims=['y', 'x'])
    zones_cp = xr.DataArray(cp.asarray(crop_input), dims=['y', 'x'])

    out_np = crop(zones_np, zones_np, zone_ids=(1.0, 3.0))
    out_cp = crop(zones_cp, zones_cp, zone_ids=(1.0, 3.0))

    assert out_cp.shape == out_np.shape == (4, 3)
    np.testing.assert_array_equal(_to_numpy(out_cp), _to_numpy(out_np))


@dask_required
def test_crop_dask_numpy_matches_numpy(crop_input):
    """crop dask path (_crop_bounds_dask) parity (Cat 1 HIGH)."""
    zones_np = xr.DataArray(crop_input, dims=['y', 'x'])
    zones_da = xr.DataArray(
        da.from_array(crop_input, chunks=(3, 2)), dims=['y', 'x'],
    )

    out_np = crop(zones_np, zones_np, zone_ids=(1.0, 3.0))
    out_da = crop(zones_da, zones_da, zone_ids=(1.0, 3.0))

    assert out_da.shape == out_np.shape == (4, 3)
    np.testing.assert_array_equal(_to_numpy(out_da), _to_numpy(out_np))


@cuda_and_cupy_available
@dask_required
def test_crop_dask_cupy_matches_numpy(crop_input):
    """crop dask+cupy parity (Cat 1 HIGH)."""
    import cupy as cp

    zones_np = xr.DataArray(crop_input, dims=['y', 'x'])
    zones_dc = xr.DataArray(
        da.from_array(cp.asarray(crop_input), chunks=(3, 2)),
        dims=['y', 'x'],
    )
    values_dc = xr.DataArray(
        da.from_array(cp.asarray(crop_input), chunks=(3, 2)),
        dims=['y', 'x'],
    )

    out_np = crop(zones_np, zones_np, zone_ids=(1.0, 3.0))
    out_dc = crop(zones_dc, values_dc, zone_ids=(1.0, 3.0))

    assert out_dc.shape == out_np.shape == (4, 3)
    np.testing.assert_array_equal(_to_numpy(out_dc), _to_numpy(out_np))


def test_crop_preserves_name_attribute(crop_input):
    """Cat 5 MEDIUM: crop should set name to 'crop' by default."""
    arr = xr.DataArray(
        crop_input,
        dims=['y', 'x'],
        attrs={'res': (1.0, 1.0), 'crs': 'EPSG:4326'},
    )
    out = crop(arr, arr, zone_ids=(1.0, 3.0))
    assert out.name == 'crop'
    # attrs propagated from input values
    assert out.attrs.get('res') == (1.0, 1.0)


# ---------------------------------------------------------------------------
# Cat 1 HIGH -- apply 3D cupy / dask+cupy backend coverage
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
def test_apply_3d_cupy():
    """apply on 3D values with cupy backend (Cat 1 HIGH).

    Exercises the 3D branch of _apply_cupy (zonal.py:1655-1660): per-layer
    kernel launch over the third axis.
    """
    import cupy as cp

    zones_data = np.array([[1, 0], [0, 2]], dtype=np.int32)
    values_data = np.ones((2, 2, 3), dtype=np.float64) * 5.0

    zones = xr.DataArray(cp.asarray(zones_data), dims=['y', 'x'])
    vals = xr.DataArray(cp.asarray(values_data), dims=['y', 'x', 'band'])

    result = apply(zones, vals, lambda x: x + 10, nodata=0)
    result_np = _to_numpy(result)

    # Zone-1 cell and zone-2 cell incremented
    np.testing.assert_array_equal(result_np[0, 0, :], [15.0, 15.0, 15.0])
    np.testing.assert_array_equal(result_np[1, 1, :], [15.0, 15.0, 15.0])
    # nodata cells stay at 5.0
    np.testing.assert_array_equal(result_np[0, 1, :], [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(result_np[1, 0, :], [5.0, 5.0, 5.0])


@cuda_and_cupy_available
@dask_required
def test_apply_3d_dask_cupy():
    """apply on 3D values with dask+cupy backend (Cat 1 HIGH).

    Exercises the 3D branch of _apply_dask_cupy (zonal.py:1722-1731): per-layer
    map_blocks + da.stack.
    """
    import cupy as cp

    zones_data = np.array([[1, 0], [0, 2]], dtype=np.int32)
    values_data = np.ones((2, 2, 3), dtype=np.float64) * 5.0

    zones = xr.DataArray(
        da.from_array(cp.asarray(zones_data), chunks=(2, 2)),
        dims=['y', 'x'],
    )
    vals = xr.DataArray(
        da.from_array(cp.asarray(values_data), chunks=(2, 2, 3)),
        dims=['y', 'x', 'band'],
    )

    result = apply(zones, vals, lambda x: x + 10, nodata=0)
    result_np = _to_numpy(result)

    np.testing.assert_array_equal(result_np[0, 0, :], [15.0, 15.0, 15.0])
    np.testing.assert_array_equal(result_np[1, 1, :], [15.0, 15.0, 15.0])
    np.testing.assert_array_equal(result_np[0, 1, :], [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(result_np[1, 0, :], [5.0, 5.0, 5.0])


# ---------------------------------------------------------------------------
# Cat 3 MEDIUM -- 1x1 single-pixel raster edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_trim_single_pixel(backend):
    """Cat 3 MEDIUM: trim on a 1x1 raster (no padding to trim)."""
    if 'dask' in backend and not has_dask_array():
        pytest.skip("Requires dask.array")

    arr = create_test_raster(np.array([[5.0]]), backend, chunks=(1, 1))
    out = trim(arr, values=(0.0,))
    assert out.shape == (1, 1)
    np.testing.assert_array_equal(_to_numpy(out), np.array([[5.0]]))


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_crop_single_pixel(backend):
    """Cat 3 MEDIUM: crop on a 1x1 raster."""
    if 'dask' in backend and not has_dask_array():
        pytest.skip("Requires dask.array")

    arr = create_test_raster(np.array([[3.0]]), backend, chunks=(1, 1))
    out = crop(arr, arr, zone_ids=(3.0,))
    assert out.shape == (1, 1)
    np.testing.assert_array_equal(_to_numpy(out), np.array([[3.0]]))


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_strip_1xN(backend):
    """Cat 3 MEDIUM: regions on a 1xN strip."""
    if 'dask' in backend and not has_dask_array():
        pytest.skip("Requires dask.array")

    arr = create_test_raster(
        np.array([[1.0, 1.0, 0.0, 2.0, 2.0]]), backend, chunks=(1, 3),
    )
    out = regions(arr, neighborhood=4)
    out_np = _to_numpy(out)
    # three connected regions: the two 1s, the zero, the two 2s
    assert len(np.unique(out_np[np.isfinite(out_np)])) == 3


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_strip_Nx1(backend):
    """Cat 3 MEDIUM: regions on an Nx1 strip."""
    if 'dask' in backend and not has_dask_array():
        pytest.skip("Requires dask.array")

    arr = create_test_raster(
        np.array([[1.0], [1.0], [0.0], [2.0], [2.0]]),
        backend, chunks=(3, 1),
    )
    out = regions(arr, neighborhood=4)
    out_np = _to_numpy(out)
    # three connected regions
    assert len(np.unique(out_np[np.isfinite(out_np)])) == 3


# ---------------------------------------------------------------------------
# Cat 4 LOW -- suggest_zonal_canvas Geographic CRS pin
# ---------------------------------------------------------------------------

def test_suggest_zonal_canvas_geographic_crs():
    """Cat 4 LOW: pin Geographic CRS branch of suggest_zonal_canvas.

    Geographic uses a 2:1 (x:y) aspect ratio (extent ±180 / ±90).  This
    test pins that ratio so a regression that changes the extent surfaces.
    """
    h_g, w_g = suggest_zonal_canvas(
        smallest_area=1.0,
        x_range=(-10.0, 10.0),
        y_range=(-5.0, 5.0),
        crs='Geographic',
        min_pixels=25,
    )
    assert isinstance(h_g, int) and isinstance(w_g, int)
    assert h_g > 0 and w_g > 0

    # Geographic has 2:1 (x:y) aspect ratio -> width/height should land
    # near 4x (because input x_range = 2*y_range and aspect_ratio = 2)
    # The exact math: full_aspect_ratio = (360/180) = 2,
    # h = sqrt(total_pixels / 2), w = 2*h.
    # Then canvas_h *= y_range/180 (= 10/180), canvas_w *= x_range/360 (= 20/360).
    # Both fractions are 1/18, so canvas_w / canvas_h = w/h = 2.
    assert w_g == 2 * h_g


def test_suggest_zonal_canvas_invalid_crs_raises():
    """Cat 4 LOW: invalid CRS triggers KeyError in CRS lookup."""
    with pytest.raises(KeyError):
        suggest_zonal_canvas(
            smallest_area=1.0,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            crs='NotAValidProjection',
        )


# ---------------------------------------------------------------------------
# Cat 4 MEDIUM -- crosstab parameter coverage on cupy backend
# ---------------------------------------------------------------------------

@cuda_and_cupy_available
def test_crosstab_cupy_with_zone_ids_filter():
    """crosstab cupy with zone_ids subset (Cat 4 MEDIUM)."""
    import cupy as cp

    zones_np = np.array(
        [[1, 1, 2, 2],
         [1, 1, 2, 2],
         [3, 3, 4, 4]],
        dtype=np.float64,
    )
    values_np = np.array(
        [[10, 10, 20, 20],
         [10, 20, 20, 30],
         [30, 30, 10, 10]],
        dtype=np.float64,
    )

    zones_c = xr.DataArray(cp.asarray(zones_np), dims=['y', 'x'])
    values_c = xr.DataArray(cp.asarray(values_np), dims=['y', 'x'])
    df_cp = crosstab(zones_c, values_c, zone_ids=[1.0, 3.0])

    # filtered to zones 1 and 3 only
    assert set(df_cp['zone'].tolist()) == {1.0, 3.0}


@cuda_and_cupy_available
def test_crosstab_cupy_with_cat_ids_filter():
    """crosstab cupy with cat_ids subset (Cat 4 MEDIUM)."""
    import cupy as cp

    zones_np = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.float64)
    values_np = np.array([[10, 20, 30, 40], [10, 20, 30, 40]], dtype=np.float64)

    zones_c = xr.DataArray(cp.asarray(zones_np), dims=['y', 'x'])
    values_c = xr.DataArray(cp.asarray(values_np), dims=['y', 'x'])
    df_cp = crosstab(zones_c, values_c, cat_ids=[10.0, 30.0])

    cols = set(df_cp.columns) - {'zone'}
    assert cols == {10.0, 30.0}


# ---------------------------------------------------------------------------
# Cat 1 -- 3D crosstab backend coverage (cupy / dask+cupy), issue #2619
#
# crosstab() accepts a 3D categorical `values` array (category dim picked by
# `layer=`).  The cupy / dask+cupy backends move the data to host memory and
# delegate to _crosstab_numpy.  The 3D GPU paths run but were untested: the
# existing 3D crosstab tests only parametrize numpy / dask+numpy.
# ---------------------------------------------------------------------------

@pytest.fixture
def crosstab_3d_input():
    """2D zones + 3D categorical values (y, x, cat) for 3D crosstab tests."""
    zones = np.array(
        [[1, 1, 2, 2, 3, 3],
         [1, 1, 2, 2, 3, 3],
         [1, 1, 2, 2, 3, 3]],
        dtype=np.float64,
    )
    # 3 rows x 6 cols x 2 categories; the two category layers differ so a
    # backend that mishandles the category axis would diverge from numpy.
    cat0 = np.ones((3, 6), dtype=np.float64)
    cat1 = np.full((3, 6), 2.0, dtype=np.float64)
    values = np.stack([cat0, cat1], axis=-1)  # shape (3, 6, 2)
    return zones, values


def _build_3d_values(values_np, backend):
    """Wrap a (y, x, cat) numpy array as a DataArray for the given backend."""
    import cupy as cp
    if backend == 'numpy':
        data = values_np
    elif backend == 'cupy':
        data = cp.asarray(values_np)
    elif backend == 'dask+cupy':
        data = da.from_array(cp.asarray(values_np), chunks=(3, 3, 2))
    else:
        raise ValueError(backend)
    agg = xr.DataArray(data, dims=['y', 'x', 'cat'])
    agg['cat'] = ['a', 'b']
    return agg


def _build_2d_zones(zones_np, backend):
    import cupy as cp
    if backend == 'numpy':
        data = zones_np
    elif backend == 'cupy':
        data = cp.asarray(zones_np)
    elif backend == 'dask+cupy':
        data = da.from_array(cp.asarray(zones_np), chunks=(3, 3))
    else:
        raise ValueError(backend)
    return xr.DataArray(data, dims=['y', 'x'])


@cuda_and_cupy_available
def test_crosstab_3d_count_cupy_matches_numpy(crosstab_3d_input):
    """3D crosstab(agg='count') on cupy matches numpy (issue #2619).

    Exercises the 3D branch of _crosstab_cupy: a 3D cupy values array is
    moved to host with cupy.asnumpy and the category coordinate flows
    through _find_cats / _crosstab_numpy.
    """
    zones_np, values_np = crosstab_3d_input

    zones_n = _build_2d_zones(zones_np, 'numpy')
    values_n = _build_3d_values(values_np, 'numpy')
    df_np = crosstab(zones_n, values_n, layer=-1, agg='count')

    zones_c = _build_2d_zones(zones_np, 'cupy')
    values_c = _build_3d_values(values_np, 'cupy')
    df_cp = _to_pandas(crosstab(zones_c, values_c, layer=-1, agg='count'))

    assert list(df_cp.columns) == list(df_np.columns)
    pd.testing.assert_frame_equal(
        df_cp.reset_index(drop=True), df_np.reset_index(drop=True),
        check_dtype=False,
    )


@cuda_and_cupy_available
@dask_required
def test_crosstab_3d_count_dask_cupy_matches_numpy(crosstab_3d_input):
    """3D crosstab(agg='count') on dask+cupy matches numpy (issue #2619).

    Exercises the 3D branch of _crosstab_dask_cupy: per-block .get() to
    host, then the dask+numpy 3D crosstab.  Only agg='count' is supported
    for 3D dask-backed input.
    """
    zones_np, values_np = crosstab_3d_input

    zones_n = _build_2d_zones(zones_np, 'numpy')
    values_n = _build_3d_values(values_np, 'numpy')
    df_np = crosstab(zones_n, values_n, layer=-1, agg='count')

    zones_dc = _build_2d_zones(zones_np, 'dask+cupy')
    values_dc = _build_3d_values(values_np, 'dask+cupy')
    df_dc = _to_pandas(crosstab(zones_dc, values_dc, layer=-1, agg='count'))

    assert list(df_dc.columns) == list(df_np.columns)
    pd.testing.assert_frame_equal(
        df_dc.reset_index(drop=True), df_np.reset_index(drop=True),
        check_dtype=False,
    )


@cuda_and_cupy_available
def test_crosstab_3d_nodata_cupy_matches_numpy(crosstab_3d_input):
    """3D crosstab with nodata_values on cupy matches numpy (issue #2619)."""
    zones_np, values_np = crosstab_3d_input

    zones_n = _build_2d_zones(zones_np, 'numpy')
    values_n = _build_3d_values(values_np, 'numpy')
    # nodata_values=2 drops every cell in the second category layer.
    df_np = crosstab(zones_n, values_n, layer=-1, agg='count', nodata_values=2)

    zones_c = _build_2d_zones(zones_np, 'cupy')
    values_c = _build_3d_values(values_np, 'cupy')
    df_cp = _to_pandas(
        crosstab(zones_c, values_c, layer=-1, agg='count', nodata_values=2)
    )

    pd.testing.assert_frame_equal(
        df_cp.reset_index(drop=True), df_np.reset_index(drop=True),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Cat 5 MEDIUM -- regions coords / attrs propagation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_regions_preserves_coords_and_attrs(backend, regions_input):
    """Cat 5 MEDIUM: regions must propagate coords + attrs."""
    if 'dask' in backend and not has_dask_array():
        pytest.skip("Requires dask.array")

    arr = create_test_raster(
        regions_input, backend,
        attrs={'res': (0.5, 0.5), 'crs': 'EPSG:4326', 'custom': 'tag'},
        chunks=(3, 3),
    )
    out = regions(arr, neighborhood=4)

    assert out.dims == arr.dims
    assert out.attrs == arr.attrs
    for coord in arr.coords:
        np.testing.assert_allclose(out[coord].data, arr[coord].data)
