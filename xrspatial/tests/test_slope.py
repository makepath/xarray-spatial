import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial import slope
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_nan_edges_effect, assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy, create_test_raster,
                                            cuda_and_cupy_available,
                                            dask_array_available, general_output_checks)


def input_data(data, backend):
    # Notes:
    # ------
    # The `elevation` data was run through QGIS slope function to
    # get values to compare against.  Xarray-Spatial currently handles
    # edges by padding with nan which is different than QGIS but acknowledged
    raster = create_test_raster(data, backend, attrs={'res': (1, 1)})
    return raster


@pytest.fixture
def qgis_slope():
    qgis_result = np.array([
        [   np.nan,    np.nan,    np.nan,    np.nan,    np.nan,    np.nan],
        [   np.nan,    np.nan,    np.nan,    np.nan,    np.nan,    np.nan],
        [89.707756, 88.56143 , 89.45366 , 89.50229 , 88.82584 , 89.782394],
        [89.78415 , 89.61588 , 89.47127 , 89.24196 , 88.385376, 89.67071 ],
        [89.7849  , 89.61132 , 89.59183 , 89.56854 , 88.90889 , 89.765114],
        [89.775246, 89.42886 , 89.25054 , 89.60963 , 89.71719 , 89.76396 ],
        [89.85427 , 89.75693 , 89.67336 , 89.502174, 89.24611 , 89.352   ],
        [89.87612 , 89.76542 , 89.269966, 89.78526 , 88.35767 , 89.764206]],
        dtype=np.float32)
    return qgis_result


def test_numpy_equals_qgis(elevation_raster, qgis_slope):
    # slope by xrspatial
    numpy_agg = input_data(elevation_raster, backend='numpy')
    xrspatial_slope_numpy = slope(numpy_agg, name='slope_numpy')
    # slope overrides attrs['units'] to 'degrees', so skip exact-attrs check
    general_output_checks(numpy_agg, xrspatial_slope_numpy, verify_attrs=False)
    assert xrspatial_slope_numpy.name == 'slope_numpy'
    print('numpy_agg', numpy_agg)
    print('xrspatial_slope_numpy', xrspatial_slope_numpy)
    xrspatial_vals = xrspatial_slope_numpy.data[1:-1, 1:-1]
    qgis_vals = qgis_slope[1:-1, 1:-1]
    print('xrspatial_vals', xrspatial_vals)

    np.testing.assert_allclose(xrspatial_vals, qgis_vals, rtol=1e-05, equal_nan=True)

    # nan border edges
    assert_nan_edges_effect(xrspatial_slope_numpy)


@dask_array_available
def test_numpy_equals_dask_qgis_data(elevation_raster):
    # compare using the data run through QGIS
    numpy_agg = input_data(elevation_raster, 'numpy')
    dask_agg = input_data(elevation_raster, 'dask+numpy')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, slope, verify_attrs=False)


@cuda_and_cupy_available
def test_numpy_equals_cupy_qgis_data(elevation_raster):
    # compare using the data run through QGIS
    numpy_agg = input_data(elevation_raster, 'numpy')
    cupy_agg = input_data(elevation_raster, 'cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, slope, verify_attrs=False)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_cupy_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, slope, atol=1e-6, rtol=1e-6,
                                  verify_attrs=False)


@dask_array_available
def test_boundary_modes(elevation_raster):
    numpy_agg = input_data(elevation_raster, 'numpy')
    dask_agg = input_data(elevation_raster, 'dask+numpy')
    assert_boundary_mode_correctness(numpy_agg, dask_agg, slope)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((6, 8), (3, 4)),      # chunks evenly divide array
    ((7, 9), (3, 3)),      # ragged last chunk
    ((10, 15), (5, 5)),    # larger array, medium chunks
    ((10, 15), (10, 15)),  # single chunk (no overlap needed)
    ((5, 5), (2, 2)),      # many small chunks
])
def test_boundary_numpy_equals_dask(boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float64) * 500
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=chunks)
    np_result = slope(numpy_agg, boundary=boundary)
    da_result = slope(dask_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@cuda_and_cupy_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
def test_boundary_numpy_equals_cupy(boundary):
    # The planar cupy backend has its own non-'nan' boundary branch
    # (_run_cupy pads, recurses, then crops). Pin it against the numpy
    # reference for every boundary mode so a GPU-only boundary regression
    # can't ship undetected.
    rng = np.random.default_rng(42)
    data = (rng.random((10, 15)).astype(np.float64) * 500)
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    cupy_agg = create_test_raster(data, backend='cupy', attrs={'res': (1, 1)})
    np_result = slope(numpy_agg, boundary=boundary)
    cupy_result = slope(cupy_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, cupy_result.data.get(), equal_nan=True, rtol=1e-5)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((10, 15), (5, 5)),    # multiple chunks → exercises overlap
    ((10, 15), (10, 15)),  # single chunk (no overlap needed)
    ((7, 9), (3, 3)),      # ragged last chunk under depth-1 overlap
])
def test_boundary_numpy_equals_dask_cupy(boundary, size, chunks):
    # The planar dask+cupy backend threads `boundary` through
    # `_boundary_to_dask(..., is_cupy=True)` and map_overlap. Pin it against
    # the numpy reference across chunkings so a GPU dask boundary regression
    # can't ship undetected.
    rng = np.random.default_rng(42)
    data = (rng.random(size).astype(np.float64) * 500)
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy',
                                       attrs={'res': (1, 1)}, chunks=chunks)
    np_result = slope(numpy_agg, boundary=boundary)
    dc_result = slope(dask_cupy_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, dc_result.data.compute().get(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_boundary_no_nan_edges(boundary, elevation_raster_no_nans):
    """Non-nan modes produce no NaN output when source has no NaN."""
    numpy_agg = create_test_raster(elevation_raster_no_nans, backend='numpy',
                                   attrs={'res': (1, 1)})
    dask_agg = create_test_raster(elevation_raster_no_nans, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=(4, 3))
    np_result = slope(numpy_agg, boundary=boundary)
    da_result = slope(dask_agg, boundary=boundary)
    assert not np.any(np.isnan(np_result.data))
    assert not np.any(np.isnan(da_result.data.compute()))
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
def test_boundary_constant_surface():
    """Constant surface should produce zero slope for all boundary modes."""
    data = np.full((8, 10), 42.0, dtype=np.float64)
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=(4, 5))
    for mode in ('nan', 'nearest', 'reflect', 'wrap'):
        np_result = slope(numpy_agg, boundary=mode)
        da_result = slope(dask_agg, boundary=mode)
        np_data = np_result.data
        da_data = da_result.data.compute()
        # Interior should be zero; edges depend on mode
        if mode != 'nan':
            np.testing.assert_allclose(np_data, 0.0, atol=1e-10)
        np.testing.assert_allclose(np_data, da_data, equal_nan=True, rtol=1e-6)


def test_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float32)
    agg = create_test_raster(data, attrs={'res': (1, 1)})
    with pytest.raises(ValueError, match="boundary must be one of"):
        slope(agg, boundary='invalid')


@dask_array_available
def test_dask_numpy_lazy_dtype_matches_computed():
    # The lazy dask dtype must match both the computed dtype and the
    # numpy backend (all float32). Regression for the planar dask meta
    # defaulting to float64 while chunks return float32.
    data = np.random.default_rng(0).random((8, 10)).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=(3, 4))
    np_result = slope(numpy_agg)
    da_result = slope(dask_agg)
    assert np_result.dtype == np.float32
    assert da_result.dtype == np.float32
    assert da_result.data.compute().dtype == np.float32


@cuda_and_cupy_available
def test_cupy_lazy_dtype_matches_computed():
    data = np.random.default_rng(0).random((8, 10)).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    cupy_agg = create_test_raster(data, backend='cupy', attrs={'res': (1, 1)})
    assert slope(numpy_agg).dtype == np.float32
    assert slope(cupy_agg).dtype == np.float32


@dask_array_available
@cuda_and_cupy_available
def test_dask_cupy_lazy_dtype_matches_computed():
    data = np.random.default_rng(0).random((8, 10)).astype(np.float64) * 100
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy',
                                       attrs={'res': (1, 1)}, chunks=(3, 4))
    da_result = slope(dask_cupy_agg)
    assert da_result.dtype == np.float32
    assert da_result.data.compute().dtype == np.float32


def test_name_annotation_matches_terrain_family():
    # slope's `name` annotation should match aspect/curvature/northness/
    # eastness, all of which use Optional[str] (None is a valid name).
    import typing

    from xrspatial import curvature
    from xrspatial.aspect import aspect, eastness, northness

    slope_func = getattr(slope, '__wrapped__', slope)
    expected = typing.get_type_hints(slope_func)['name']
    for func in (aspect, curvature, northness, eastness):
        unwrapped = getattr(func, '__wrapped__', func)
        assert typing.get_type_hints(unwrapped)['name'] == expected


def test_name_none_accepted():
    data = np.ones((4, 5), dtype=np.float32)
    agg = create_test_raster(data, attrs={'res': (1, 1)})
    result = slope(agg, name=None)
    assert result.name is None


# The output .name must agree across backends. xr.DataArray keeps a dask
# array's internal graph token (e.g. '_trim-<hash>' planar, 'getitem-<hash>'
# geodesic) as .name when name=None, so the dask backends used to disagree
# with numpy/cupy. Regression for the slope .name leak (same class as zonal
# #2611, focal #2733, proximity #2723, viewshed #2743).
def _assert_name_planar(backend, name):
    data = np.random.default_rng(0).random((8, 10)).astype(np.float64) * 100
    agg = create_test_raster(data, backend=backend, attrs={'res': (1, 1)},
                             chunks=(3, 4))
    assert slope(agg, name=name).name == name


def _geodesic_raster(backend):
    H, W = 8, 10
    lat = np.linspace(40.0, 41.0, H)
    lon = np.linspace(10.0, 11.0, W)
    data = np.random.default_rng(1).random((H, W)) * 100
    raster = xr.DataArray(
        data, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon},
    )
    if 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)
    if 'dask' in backend:
        import dask.array as da
        raster.data = da.from_array(raster.data, chunks=(4, 5))
    return raster


def _assert_name_geodesic(backend, name):
    result = slope(_geodesic_raster(backend), method='geodesic', name=name)
    assert result.name == name


@pytest.mark.parametrize("name", [None, 'slope'])
def test_name_consistent_numpy(name):
    _assert_name_planar('numpy', name)
    _assert_name_geodesic('numpy', name)


@dask_array_available
@pytest.mark.parametrize("name", [None, 'slope'])
def test_name_consistent_dask_numpy(name):
    _assert_name_planar('dask+numpy', name)
    _assert_name_geodesic('dask+numpy', name)


@cuda_and_cupy_available
@pytest.mark.parametrize("name", [None, 'slope'])
def test_name_consistent_cupy(name):
    _assert_name_planar('cupy', name)
    _assert_name_geodesic('cupy', name)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("name", [None, 'slope'])
def test_name_consistent_dask_cupy(name):
    _assert_name_planar('dask+cupy', name)
    _assert_name_geodesic('dask+cupy', name)


# Degenerate shapes: a dimension too small for the 3x3 kernel to have an
# interior. The planar CPU kernel loops over range(1, rows-1) (empty when
# rows<=2), so the output keeps the input shape and comes back all-NaN. Pin
# this behaviour so a future kernel-bound change can't silently start raising
# or returning the wrong shape on these inputs.
DEGENERATE_SHAPES = [(1, 1), (1, 5), (5, 1)]


def _degenerate_data(shape):
    return np.arange(np.prod(shape), dtype=np.float64).reshape(shape)


@pytest.mark.parametrize("shape", DEGENERATE_SHAPES)
def test_degenerate_shape_numpy(shape):
    agg = create_test_raster(_degenerate_data(shape), backend='numpy',
                             attrs={'res': (1, 1)})
    result = slope(agg)
    general_output_checks(agg, result, verify_attrs=False)
    assert result.shape == shape
    assert np.all(np.isnan(result.data))


@dask_array_available
@pytest.mark.parametrize("shape", DEGENERATE_SHAPES)
def test_degenerate_shape_dask_numpy(shape):
    data = _degenerate_data(shape)
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=shape)
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, slope, nan_edges=False,
                                   verify_attrs=False)


@cuda_and_cupy_available
@pytest.mark.parametrize("shape", DEGENERATE_SHAPES)
def test_degenerate_shape_cupy(shape):
    data = _degenerate_data(shape)
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    cupy_agg = create_test_raster(data, backend='cupy', attrs={'res': (1, 1)})
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, slope, nan_edges=False,
                             verify_attrs=False)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("shape", DEGENERATE_SHAPES)
def test_degenerate_shape_dask_cupy(shape):
    data = _degenerate_data(shape)
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy',
                                       attrs={'res': (1, 1)}, chunks=shape)
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, slope, nan_edges=False,
                                  verify_attrs=False)


@pytest.mark.parametrize("shape", DEGENERATE_SHAPES)
def test_degenerate_shape_geodesic(shape):
    H, W = shape
    lat = np.linspace(40.0, 41.0, H)
    lon = np.linspace(10.0, 11.0, W)
    raster = xr.DataArray(
        _degenerate_data(shape),
        dims=['lat', 'lon'],
        coords={'lat': lat, 'lon': lon},
    )
    result = slope(raster, method='geodesic')
    general_output_checks(raster, result, verify_attrs=False)
    assert result.shape == shape
    assert np.all(np.isnan(result.data))


def _degree_coord_meter_elevation_raster():
    # degree-like coords (lon/lat) with meter-scale elevation values
    data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)
    y = np.linspace(5.0, 5.0025, 10)
    x = np.linspace(-74.93, -74.9275, 10)
    return xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={'y': y, 'x': x},
        attrs={'units': 'm'},
    )


def _projected_meter_raster():
    # projected coords in meters (UTM-ish) with meter-scale elevation values
    data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)
    y = np.arange(10) * 30.0
    x = 500_000.0 + np.arange(10) * 30.0
    return xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={'y': y, 'x': x},
        attrs={'units': 'm'},
    )


def test_planar_warns_on_degree_coords_meter_elevation():
    raster = _degree_coord_meter_elevation_raster()
    with pytest.warns(UserWarning, match="appears to have coordinates in degrees"):
        slope(raster)  # method='planar' is the default


def test_planar_no_warn_on_projected_meter_coords():
    raster = _projected_meter_raster()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        slope(raster)  # must not raise: no unit-mismatch warning expected


# A NoData (NaN) center cell must stay NaN in the slope output, even when all 8
# of its neighbours are valid. The planar kernels read the center cell, so a
# hole in the DEM can't masquerade as valid flat terrain. Regression for #2761.
def _center_nan_data():
    data = np.zeros((5, 5), dtype=np.float64)
    data[2, 2] = np.nan
    return data


def test_center_nan_propagates_numpy():
    agg = create_test_raster(_center_nan_data(), backend='numpy', attrs={'res': (1, 1)})
    result = slope(agg)
    general_output_checks(agg, result, verify_attrs=False)
    assert np.isnan(result.data[2, 2])


@dask_array_available
def test_center_nan_propagates_dask_numpy():
    data = _center_nan_data()
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=(3, 3))
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, slope, nan_edges=False,
                                   verify_attrs=False)
    assert np.isnan(slope(dask_agg).data.compute()[2, 2])


@cuda_and_cupy_available
def test_center_nan_propagates_cupy():
    data = _center_nan_data()
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    cupy_agg = create_test_raster(data, backend='cupy', attrs={'res': (1, 1)})
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, slope, nan_edges=False,
                             verify_attrs=False)
    assert np.isnan(slope(cupy_agg).data.get()[2, 2])


@dask_array_available
@cuda_and_cupy_available
def test_center_nan_propagates_dask_cupy():
    data = _center_nan_data()
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy',
                                       attrs={'res': (1, 1)}, chunks=(3, 3))
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, slope, nan_edges=False,
                                  verify_attrs=False)


# ---------------------------------------------------------------------------
# Planar cell-size validation (issue #2897)
# ---------------------------------------------------------------------------

INVALID_RES = [
    (0, 0),         # zero on both axes
    (0, 1),         # zero on x
    (1, 0),         # zero on y
    (-1, 1),        # negative on x
    (1, -1),        # negative on y
    (np.inf, 1),    # inf on x
    (1, np.inf),    # inf on y
    (np.nan, 1),    # nan on x
    (1, np.nan),    # nan on y
]


@pytest.mark.parametrize("res", INVALID_RES)
def test_planar_invalid_cellsize_numpy(res):
    """slope(method='planar') must reject zero, negative, or non-finite cell
    sizes with a clear ValueError instead of a raw ZeroDivisionError, silent
    NaN interior, or plausible-but-wrong values.
    """
    data = np.zeros((5, 5), dtype=np.float32)
    agg = create_test_raster(data, backend='numpy', attrs={'res': res})
    with pytest.raises(ValueError, match="positive, finite cell size"):
        slope(agg)


@dask_array_available
@pytest.mark.parametrize("res", INVALID_RES)
def test_planar_invalid_cellsize_dask_numpy(res):
    data = np.zeros((5, 5), dtype=np.float32)
    agg = create_test_raster(data, backend='dask+numpy',
                             attrs={'res': res}, chunks=(3, 3))
    with pytest.raises(ValueError, match="positive, finite cell size"):
        slope(agg)


@cuda_and_cupy_available
@pytest.mark.parametrize("res", INVALID_RES)
def test_planar_invalid_cellsize_cupy(res):
    data = np.zeros((5, 5), dtype=np.float32)
    agg = create_test_raster(data, backend='cupy', attrs={'res': res})
    with pytest.raises(ValueError, match="positive, finite cell size"):
        slope(agg)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("res", INVALID_RES)
def test_planar_invalid_cellsize_dask_cupy(res):
    data = np.zeros((5, 5), dtype=np.float32)
    agg = create_test_raster(data, backend='dask+cupy',
                             attrs={'res': res}, chunks=(3, 3))
    with pytest.raises(ValueError, match="positive, finite cell size"):
        slope(agg)


def test_planar_valid_cellsize_still_works():
    """A normal positive cell size must not trip the new guard."""
    data = np.zeros((5, 5), dtype=np.float32)
    agg = create_test_raster(data, backend='numpy', attrs={'res': (10, 10)})
    result = slope(agg)
    # slope() sets attrs['units'] = 'degrees' (#2894), so the output attrs
    # legitimately differ from the input; the units contract is covered
    # separately by the units tests below.
    general_output_checks(agg, result, verify_attrs=False)


def test_geodesic_unaffected_by_invalid_res_attr():
    """The geodesic path takes its scale from lat/lon coords, so a bogus 'res'
    attribute must not trigger the planar cell-size guard.
    """
    lat = np.linspace(40.0, 40.4, 5)
    lon = np.linspace(-105.0, -104.6, 5)
    data = np.zeros((5, 5), dtype=np.float64)
    agg = xr.DataArray(
        data,
        coords={'y': lat, 'x': lon},
        dims=['y', 'x'],
        attrs={'res': (0, 0)},
    )
    # Must not raise the planar guard; geodesic uses lat/lon coords.
    result = slope(agg, method='geodesic')
    general_output_checks(agg, result, verify_attrs=False)


# slope is reported in degrees, so the output must advertise units='degrees'
# rather than leaking the input elevation units (e.g. 'm'). Regression for #2894.
def test_units_overridden_to_degrees_numpy():
    agg = create_test_raster(_degenerate_data((5, 5)), backend='numpy',
                             attrs={'res': (1, 1), 'units': 'm', 'crs': 'EPSG:5070'})
    result = slope(agg)
    assert result.attrs['units'] == 'degrees'
    # other attrs are preserved untouched
    assert result.attrs['res'] == (1, 1)
    assert result.attrs['crs'] == 'EPSG:5070'
    # input is not mutated
    assert agg.attrs['units'] == 'm'


@dask_array_available
def test_units_overridden_to_degrees_dask_numpy():
    agg = create_test_raster(_degenerate_data((5, 5)), backend='dask+numpy',
                             attrs={'res': (1, 1), 'units': 'm', 'crs': 'EPSG:5070'},
                             chunks=(3, 3))
    result = slope(agg)
    assert result.attrs['units'] == 'degrees'
    # non-units attrs still propagate through the dask path
    assert result.attrs['res'] == (1, 1)
    assert result.attrs['crs'] == 'EPSG:5070'
    assert agg.attrs['units'] == 'm'


@cuda_and_cupy_available
def test_units_overridden_to_degrees_cupy():
    agg = create_test_raster(_degenerate_data((5, 5)), backend='cupy',
                             attrs={'res': (1, 1), 'units': 'm'})
    result = slope(agg)
    assert result.attrs['units'] == 'degrees'
    assert agg.attrs['units'] == 'm'


@dask_array_available
@cuda_and_cupy_available
def test_units_overridden_to_degrees_dask_cupy():
    agg = create_test_raster(_degenerate_data((5, 5)), backend='dask+cupy',
                             attrs={'res': (1, 1), 'units': 'm'}, chunks=(3, 3))
    result = slope(agg)
    assert result.attrs['units'] == 'degrees'
    assert agg.attrs['units'] == 'm'


def test_units_set_to_degrees_when_input_has_no_units():
    agg = create_test_raster(_degenerate_data((5, 5)), backend='numpy',
                             attrs={'res': (1, 1)})
    result = slope(agg)
    assert result.attrs['units'] == 'degrees'


def test_units_overridden_to_degrees_geodesic():
    H, W = 5, 5
    raster = xr.DataArray(
        _degenerate_data((H, W)),
        dims=['lat', 'lon'],
        coords={'lat': np.linspace(40.0, 41.0, H), 'lon': np.linspace(10.0, 11.0, W)},
        attrs={'units': 'm'},
    )
    result = slope(raster, method='geodesic')
    assert result.attrs['units'] == 'degrees'
    assert raster.attrs['units'] == 'm'


# The planar CPU kernel has no per-cell NaN early-out (#3739): neighbour NaN
# flows through the Horn stencil and the centre is masked with a select. Pin
# the resulting footprint on speckled nodata so a future edit can't widen or
# shrink it: NaN at every centre-NaN cell and its 8-neighbours, finite
# everywhere else in the interior. The old kernel produced the same footprint,
# so these pass on either version; they pin behaviour rather than reproduce a
# regression.
def _speckled_nan_data():
    rng = np.random.default_rng(3739)
    data = rng.normal(0, 2, size=(40, 60))
    data[rng.random(data.shape) < 0.1] = np.nan
    return data


def _expected_nan_footprint(data):
    nan_mask = np.isnan(data)
    footprint = np.zeros_like(nan_mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            footprint[1:-1, 1:-1] |= nan_mask[1 + dy:data.shape[0] - 1 + dy,
                                              1 + dx:data.shape[1] - 1 + dx]
    footprint[0, :] = footprint[-1, :] = footprint[:, 0] = footprint[:, -1] = True
    return footprint


def test_speckled_nan_footprint_numpy():
    data = _speckled_nan_data()
    agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    result = slope(agg)
    general_output_checks(agg, result, verify_attrs=False)
    np.testing.assert_array_equal(np.isnan(result.data), _expected_nan_footprint(data))
    assert np.all(np.isfinite(result.data[~np.isnan(result.data)]))


@dask_array_available
def test_speckled_nan_footprint_dask_numpy():
    data = _speckled_nan_data()
    numpy_agg = create_test_raster(data, backend='numpy', attrs={'res': (1, 1)})
    dask_agg = create_test_raster(data, backend='dask+numpy',
                                  attrs={'res': (1, 1)}, chunks=(16, 24))
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, slope, nan_edges=False,
                                   verify_attrs=False)
    np.testing.assert_array_equal(np.isnan(slope(dask_agg).data.compute()),
                                  _expected_nan_footprint(data))
