import warnings

import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from xrspatial import aspect, eastness, northness
from xrspatial.utils import has_dask_array
from xrspatial.tests.general_checks import (assert_boundary_mode_correctness,
                                            assert_nan_edges_effect,
                                            assert_numpy_equals_cupy,
                                            assert_numpy_equals_dask_cupy,
                                            assert_numpy_equals_dask_numpy,
                                            dask_array_available,
                                            create_test_raster,
                                            cuda_and_cupy_available, general_output_checks)


def input_data(data, backend='numpy'):
    raster = create_test_raster(data, backend)
    return raster


@pytest.fixture
def qgis_aspect():
    result = np.array([
        [    np.nan,     np.nan,     np.nan,     np.nan,     np.nan,    np.nan],
        [    np.nan,     np.nan,     np.nan,     np.nan,     np.nan,    np.nan],
        [233.19478 , 278.358   ,  45.18813 , 306.6476  , 358.34296 , 106.45898 ],
        [267.7002  , 274.42487 ,  11.035832, 357.9641  , 129.98279 , 50.069843],
        [263.18484 , 238.47426 , 196.37103 , 149.25227 , 187.85748 , 263.684   ],
        [266.63937 , 271.05124 , 312.09726 , 348.89136 , 351.618   , 315.59424 ],
        [279.90872 , 314.11356 , 345.76315 , 327.5568  , 339.5455  , 312.9249  ],
        [271.93985 , 268.81046 ,  24.793104, 185.978   , 299.82904 ,159.0188  ]], dtype=np.float32)
    return result


def test_numpy_equals_qgis(elevation_raster, qgis_aspect):
    numpy_agg = input_data(elevation_raster, backend='numpy')
    xrspatial_aspect = aspect(numpy_agg, name='numpy_aspect')

    general_output_checks(numpy_agg, xrspatial_aspect, verify_dtype=True)
    assert xrspatial_aspect.name == 'numpy_aspect'

    xrspatial_vals = xrspatial_aspect.data[1:-1, 1:-1]
    qgis_vals = qgis_aspect[1:-1, 1:-1]
    # aspect is nan if nan input
    # aspect is invalid (-1) if slope equals 0
    # otherwise aspect are from 0 to 360
    np.testing.assert_allclose(xrspatial_vals, qgis_vals, rtol=1e-05, equal_nan=True)
    # nan edge effect
    assert_nan_edges_effect(xrspatial_aspect)


@dask_array_available
def test_numpy_equals_dask_qgis_data(elevation_raster):
    # compare using the data run through QGIS
    numpy_agg = input_data(elevation_raster, 'numpy')
    dask_agg = input_data(elevation_raster, 'dask+numpy')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, aspect)


@dask_array_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_agg = create_test_raster(random_data, backend='dask')
    assert_numpy_equals_dask_numpy(numpy_agg, dask_agg, aspect)


@cuda_and_cupy_available
def test_numpy_equals_cupy_qgis_data(elevation_raster):
    # compare using the data run through QGIS
    numpy_agg = input_data(elevation_raster)
    cupy_agg = input_data(elevation_raster, 'cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, aspect)


@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_cupy_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    cupy_agg = create_test_raster(random_data, backend='cupy')
    assert_numpy_equals_cupy(numpy_agg, cupy_agg, aspect, atol=1e-6, rtol=1e-6)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("size", [(2, 4), (10, 15)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64])
def test_numpy_equals_dask_cupy_random_data(random_data):
    numpy_agg = create_test_raster(random_data, backend='numpy')
    dask_cupy_agg = create_test_raster(random_data, backend='dask+cupy')
    assert_numpy_equals_dask_cupy(numpy_agg, dask_cupy_agg, aspect, atol=1e-6, rtol=1e-6)


@dask_array_available
def test_boundary_modes(elevation_raster):
    numpy_agg = input_data(elevation_raster, 'numpy')
    dask_agg = input_data(elevation_raster, 'dask+numpy')
    assert_boundary_mode_correctness(numpy_agg, dask_agg, aspect)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
@pytest.mark.parametrize("size,chunks", [
    ((6, 8), (3, 4)),
    ((7, 9), (3, 3)),
    ((10, 15), (5, 5)),
    ((10, 15), (10, 15)),
    ((5, 5), (2, 2)),
])
def test_boundary_numpy_equals_dask(boundary, size, chunks):
    rng = np.random.default_rng(42)
    data = rng.random(size).astype(np.float64) * 500
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    np_result = aspect(numpy_agg, boundary=boundary)
    da_result = aspect(dask_agg, boundary=boundary)
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


@dask_array_available
@pytest.mark.parametrize("boundary", ['nearest', 'reflect', 'wrap'])
def test_boundary_no_nan_edges(boundary, elevation_raster_no_nans):
    """Non-nan modes produce no NaN output when source has no NaN."""
    numpy_agg = create_test_raster(elevation_raster_no_nans, backend='numpy')
    dask_agg = create_test_raster(elevation_raster_no_nans, backend='dask+numpy',
                                  chunks=(4, 3))
    np_result = aspect(numpy_agg, boundary=boundary)
    da_result = aspect(dask_agg, boundary=boundary)
    # aspect returns -1 for flat areas, but should not return NaN
    assert not np.any(np.isnan(np_result.data))
    assert not np.any(np.isnan(da_result.data.compute()))
    np.testing.assert_allclose(
        np_result.data, da_result.data.compute(), equal_nan=True, rtol=1e-5)


def test_boundary_invalid():
    data = np.ones((4, 5), dtype=np.float32)
    agg = create_test_raster(data)
    with pytest.raises(ValueError, match="boundary must be one of"):
        aspect(agg, boundary='invalid')


@dask_array_available
@pytest.mark.parametrize("boundary", ['nan', 'nearest', 'reflect', 'wrap'])
def test_dask_numpy_advertised_dtype_matches_computed(boundary):
    # planar dask map_overlap must advertise float32, matching the realized
    # data and the numpy/cupy backends (issue #2682).
    data = np.random.default_rng(0).random((8, 10)).astype(np.float64) * 100
    numpy_agg = create_test_raster(data, backend='numpy')
    dask_agg = create_test_raster(data, backend='dask+numpy')
    np_result = aspect(numpy_agg, boundary=boundary)
    da_result = aspect(dask_agg, boundary=boundary)
    assert np_result.dtype == np.float32
    assert da_result.dtype == np.float32
    assert da_result.data.compute().dtype == np.float32


@dask_array_available
@cuda_and_cupy_available
def test_dask_cupy_advertised_dtype_matches_computed():
    import cupy
    data = np.random.default_rng(0).random((8, 10)).astype(np.float64) * 100
    dask_cupy_agg = create_test_raster(data, backend='dask+cupy')
    da_result = aspect(dask_cupy_agg)
    assert da_result.dtype == cupy.float32
    assert da_result.data.compute().dtype == cupy.float32


# ---- Degenerate raster shapes (1x1, Nx1, 1xN) ----
#
# A 3x3 kernel has no interior cell when a dimension is smaller than 3,
# so the correct planar result is all-NaN with the input shape preserved.
# These guard against a regression that would crash or reshape on the
# kernel-boundary path. See issue #2742.

_DEGENERATE_SHAPES = [(1, 1), (1, 5), (5, 1), (3, 1), (1, 3)]


def _to_numpy(result_agg):
    data = result_agg.data
    if has_dask_array() and isinstance(data, da.Array):
        data = data.compute()
    if hasattr(data, 'get'):  # cupy
        data = data.get()
    return data


@pytest.mark.parametrize("func", [aspect, northness, eastness])
@pytest.mark.parametrize("shape", _DEGENERATE_SHAPES)
def test_degenerate_shape_numpy(func, shape):
    data = np.ones(shape, dtype=np.float32)
    if shape[0] > 1:
        data[0, :] = 2.0
    agg = create_test_raster(data, backend='numpy')
    result = func(agg)
    general_output_checks(agg, result)
    assert result.shape == shape
    assert np.all(np.isnan(result.data))


@dask_array_available
@pytest.mark.parametrize("func", [aspect, northness, eastness])
@pytest.mark.parametrize("shape", _DEGENERATE_SHAPES)
def test_degenerate_shape_dask_numpy(func, shape):
    data = np.ones(shape, dtype=np.float32)
    if shape[0] > 1:
        data[0, :] = 2.0
    chunks = (min(2, shape[0]), min(2, shape[1]))
    agg = create_test_raster(data, backend='dask+numpy', chunks=chunks)
    result = func(agg)
    assert result.shape == shape
    assert np.all(np.isnan(_to_numpy(result)))


@cuda_and_cupy_available
@pytest.mark.parametrize("func", [aspect, northness, eastness])
@pytest.mark.parametrize("shape", _DEGENERATE_SHAPES)
def test_degenerate_shape_cupy(func, shape):
    data = np.ones(shape, dtype=np.float32)
    if shape[0] > 1:
        data[0, :] = 2.0
    agg = create_test_raster(data, backend='cupy')
    result = func(agg)
    assert result.shape == shape
    assert np.all(np.isnan(_to_numpy(result)))


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("func", [aspect, northness, eastness])
@pytest.mark.parametrize("shape", _DEGENERATE_SHAPES)
def test_degenerate_shape_dask_cupy(func, shape):
    data = np.ones(shape, dtype=np.float32)
    if shape[0] > 1:
        data[0, :] = 2.0
    chunks = (min(2, shape[0]), min(2, shape[1]))
    agg = create_test_raster(data, backend='dask+cupy', chunks=chunks)
    result = func(agg)
    assert result.shape == shape
    assert np.all(np.isnan(_to_numpy(result)))


# ---- Rectangular-cell correctness (issue #2760) ----
#
# Planar aspect must honour cellsize_x / cellsize_y. For a raster whose
# elevation increases by one unit per map-unit in both x and y, the
# East and North gradients are equal in map units, so the downslope
# points southwest and aspect is 315 degrees regardless of cell aspect
# ratio. With the divide-by-8-only kernels (no cell-size correction)
# this case returned 275.7106. See issue #2760.

def _rectangular_cell_raster(backend='numpy', xres=10.0, yres=1.0):
    ny, nx = 6, 6
    yc = np.arange(ny) * yres
    xc = np.arange(nx) * xres
    X, Y = np.meshgrid(xc, yc)
    data = (X + Y).astype(np.float64)
    return create_test_raster(
        data, backend=backend,
        attrs={'res': (xres, yres), 'crs': 'EPSG: 5070'})


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_rectangular_cell_aspect(backend):
    if backend.startswith('dask') and not dask_array_available:
        pytest.skip("dask not available")
    agg = _rectangular_cell_raster(backend=backend)
    result = aspect(agg)
    interior = _to_numpy(result)[1:-1, 1:-1]
    np.testing.assert_allclose(interior, 315.0, rtol=1e-4)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy'])
def test_rectangular_cell_northness_eastness(backend):
    if backend.startswith('dask') and not dask_array_available:
        pytest.skip("dask not available")
    agg = _rectangular_cell_raster(backend=backend)
    north = _to_numpy(northness(agg))[1:-1, 1:-1]
    east = _to_numpy(eastness(agg))[1:-1, 1:-1]
    # aspect == 315: cos(315) = +sqrt(2)/2, sin(315) = -sqrt(2)/2
    np.testing.assert_allclose(north, np.cos(np.deg2rad(315.0)), atol=1e-5)
    np.testing.assert_allclose(east, np.sin(np.deg2rad(315.0)), atol=1e-5)


@cuda_and_cupy_available
def test_rectangular_cell_aspect_cupy():
    agg = _rectangular_cell_raster(backend='cupy')
    interior = _to_numpy(aspect(agg))[1:-1, 1:-1]
    np.testing.assert_allclose(interior, 315.0, rtol=1e-4)


@dask_array_available
@cuda_and_cupy_available
def test_rectangular_cell_aspect_dask_cupy():
    agg = _rectangular_cell_raster(backend='dask+cupy')
    interior = _to_numpy(aspect(agg))[1:-1, 1:-1]
    np.testing.assert_allclose(interior, 315.0, rtol=1e-4)


# ---- Independent analytic oracle for planar aspect ----
#
# The other planar tests only prove the four backends agree with each
# other. They all share the same _cpu/_gpu kernels, so a defect common to
# every backend (like the missing-cellsize bug in #2760) passes anyway.
# The QGIS fixture is a real external reference but uses square cells, so
# it never checks rectangular cells or that cell size is read from the
# coordinates.
#
# These tests build a surface with a known constant gradient and compare
# aspect() against the closed-form Horn value computed from the real cell
# sizes, not against another backend. The raster carries no 'res' attr, so
# resolution must come from the x/y coordinate spacing.

_RADIAN = 180 / np.pi


def _aspect_oracle(gx, gy):
    """Closed-form compass aspect for a plane z = gx*X + gy*Y (X east, Y
    south-by-row). The cell-size-correct Horn gradients of such a plane are
    exactly gx and gy, independent of cell size, so the oracle only needs
    the gradients.

    The arctan2/compass conversion below is duplicated from aspect._cpu on
    purpose: an oracle that called aspect() would not be independent. The
    square-cell tests re-check this against the real function every run, so
    drift is caught.
    """
    if gx == 0 and gy == 0:
        return -1.0
    a = np.arctan2(gy, -gx) * _RADIAN
    if a < 0:
        return 90.0 - a
    elif a > 90.0:
        return 360.0 - a + 90.0
    else:
        return 90.0 - a


def _gradient_plane(gx, gy, cellsize_x, cellsize_y, n=7):
    """A DataArray of z = gx*X + gy*Y on a grid with the given cell sizes.

    Coordinates encode the resolution; no 'res' attr is set so cell size
    must be recovered from the coordinate spacing.
    """
    xs = np.arange(n) * cellsize_x
    ys = np.arange(n) * cellsize_y
    z = gx * xs[None, :] + gy * ys[:, None]
    return xr.DataArray(
        z.astype(np.float64), dims=['y', 'x'], coords={'y': ys, 'x': xs},
        name='gradient_plane')


def _to_backend(agg, backend, chunks=(3, 3)):
    data = agg.data
    if 'cupy' in backend:
        import cupy
        data = cupy.asarray(data)
    if 'dask' in backend:
        data = da.from_array(data, chunks=chunks)
    return xr.DataArray(data, dims=agg.dims, coords=agg.coords,
                        attrs=agg.attrs, name=agg.name)


_ORACLE_BACKENDS = ['numpy']
if has_dask_array():
    _ORACLE_BACKENDS.append('dask+numpy')


def _aspect_interior(agg):
    """Compute aspect and return the realized numpy interior (drop the
    NaN edge ring)."""
    result = aspect(agg, method='planar')
    return _to_numpy(result)[1:-1, 1:-1]


# Square cells: aspect is correct today, so this oracle must pass
# unconditionally on every backend.
@pytest.mark.parametrize("backend", _ORACLE_BACKENDS)
@pytest.mark.parametrize("gx,gy", [(1.0, 1.0), (2.0, -3.0), (-1.0, 0.0), (0.0, 0.0)])
@pytest.mark.parametrize("cellsize", [1.0, 0.5, 10.0])
def test_planar_aspect_oracle_square(backend, gx, gy, cellsize):
    agg = _to_backend(_gradient_plane(gx, gy, cellsize, cellsize), backend)
    interior = _aspect_interior(agg)
    expected = _aspect_oracle(gx, gy)
    np.testing.assert_allclose(interior, expected, rtol=1e-4, atol=1e-3)


# Rectangular cells (xres != yres). aspect() now reads cell size from the
# coordinate spacing (#2760), so the East/North gradient ratio is correct.
@pytest.mark.parametrize("backend", _ORACLE_BACKENDS)
@pytest.mark.parametrize("gx,gy", [(1.0, 1.0), (2.0, -3.0)])
@pytest.mark.parametrize("cellsize_x,cellsize_y", [(10.0, 1.0), (1.0, 4.0)])
def test_planar_aspect_oracle_rectangular(
        backend, gx, gy, cellsize_x, cellsize_y):
    agg = _to_backend(_gradient_plane(gx, gy, cellsize_x, cellsize_y), backend)
    interior = _aspect_interior(agg)
    expected = _aspect_oracle(gx, gy)
    np.testing.assert_allclose(interior, expected, rtol=1e-4, atol=1e-3)


# Cupy and dask+cupy oracle (separate functions so the cuda/cupy skip
# marker applies). Same assertion target as the numpy/dask versions.
@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ['cupy', 'dask+cupy'])
@pytest.mark.parametrize("gx,gy", [(1.0, 1.0), (2.0, -3.0), (-1.0, 0.0)])
@pytest.mark.parametrize("cellsize", [1.0, 0.5, 10.0])
def test_planar_aspect_oracle_square_cupy(backend, gx, gy, cellsize):
    if backend == 'dask+cupy' and not has_dask_array():
        pytest.skip("Requires dask.Array")
    agg = _to_backend(_gradient_plane(gx, gy, cellsize, cellsize), backend)
    interior = _aspect_interior(agg)
    expected = _aspect_oracle(gx, gy)
    np.testing.assert_allclose(interior, expected, rtol=1e-4, atol=1e-3)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ['cupy', 'dask+cupy'])
@pytest.mark.parametrize("gx,gy", [(1.0, 1.0), (2.0, -3.0)])
@pytest.mark.parametrize("cellsize_x,cellsize_y", [(10.0, 1.0), (1.0, 4.0)])
def test_planar_aspect_oracle_rectangular_cupy(
        backend, gx, gy, cellsize_x, cellsize_y):
    if backend == 'dask+cupy' and not has_dask_array():
        pytest.skip("Requires dask.Array")
    agg = _to_backend(_gradient_plane(gx, gy, cellsize_x, cellsize_y), backend)
    interior = _aspect_interior(agg)
    expected = _aspect_oracle(gx, gy)
    np.testing.assert_allclose(interior, expected, rtol=1e-4, atol=1e-3)


# ---- Near-360 aspect band: cupy must match numpy (issue #2827) ----
#
# A plane with a tiny east gradient and a strong north gradient produces an
# aspect just under 360 (e.g. 359.99994). The numpy/numba kernel emits that
# value; the cupy kernel used to snap anything above 359.999 to 0, so the two
# backends disagreed by ~360 on the same input. The random-data cross-backend
# tests never land in this band, so it went unnoticed. See issue #2827.

def _near_360_plane(backend='numpy'):
    # gx tiny positive, gy positive -> downslope faces just west of north.
    return _to_backend(_gradient_plane(1e-6, 1.0, 1.0, 1.0, n=5), backend)


def test_near_360_aspect_numpy_lands_in_band():
    """numpy emits a value in (359.999, 360); it never snaps to 0."""
    interior = _aspect_interior(_near_360_plane('numpy'))
    assert np.all(interior > 359.999)
    assert np.all(interior < 360.0)


@pytest.mark.parametrize("backend", _ORACLE_BACKENDS)
def test_near_360_aspect_oracle(backend):
    interior = _aspect_interior(_near_360_plane(backend))
    expected = _aspect_oracle(1e-6, 1.0)
    np.testing.assert_allclose(interior, expected, rtol=1e-6, atol=1e-3)


@cuda_and_cupy_available
@pytest.mark.parametrize("backend", ['cupy', 'dask+cupy'])
def test_near_360_aspect_cupy_matches_numpy(backend):
    if backend == 'dask+cupy' and not has_dask_array():
        pytest.skip("Requires dask.Array")
    np_interior = _aspect_interior(_near_360_plane('numpy'))
    gpu_interior = _aspect_interior(_near_360_plane(backend))
    np.testing.assert_allclose(
        gpu_interior, np_interior, rtol=1e-6, atol=1e-3)


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
        aspect(raster)  # method='planar' is the default


def test_planar_warning_names_aspect_not_slope():
    # The shared warning helper defaults to naming `slope`; aspect must pass
    # its own name so the advice is not misleading. Regression for #2782.
    raster = _degree_coord_meter_elevation_raster()
    with pytest.warns(UserWarning, match="before calling `aspect`"):
        aspect(raster)


def test_planar_no_warn_on_projected_meter_coords():
    raster = _projected_meter_raster()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        aspect(raster)  # must not raise: no unit-mismatch warning expected


# The output .name must agree across backends. xr.DataArray keeps a dask
# array's internal graph token (e.g. '_trim-<hash>' planar, 'getitem-<hash>'
# geodesic) as .name when name=None, so the dask backends used to disagree
# with numpy/cupy. Regression for the aspect .name leak (same class as zonal
# #2611, focal #2733, slope #2838).
def _assert_name_planar(backend, name):
    data = np.random.default_rng(0).random((8, 10)).astype(np.float64) * 100
    agg = create_test_raster(data, backend=backend, attrs={'res': (1, 1)},
                             chunks=(3, 4))
    assert aspect(agg, name=name).name == name


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
    result = aspect(_geodesic_raster(backend), method='geodesic', name=name)
    assert result.name == name


# northness() and eastness() wrap aspect() and build their own DataArray, so
# they leak the same dask graph token (e.g. 'where-<hash>') when name=None.
def _assert_name_derived(backend, name):
    data = np.random.default_rng(2).random((8, 10)).astype(np.float64) * 100
    agg = create_test_raster(data, backend=backend, attrs={'res': (1, 1)},
                             chunks=(3, 4))
    assert northness(agg, name=name).name == name
    assert eastness(agg, name=name).name == name


@pytest.mark.parametrize("name", [None, 'aspect'])
def test_name_consistent_numpy(name):
    _assert_name_planar('numpy', name)
    _assert_name_geodesic('numpy', name)
    _assert_name_derived('numpy', name)


@dask_array_available
@pytest.mark.parametrize("name", [None, 'aspect'])
def test_name_consistent_dask_numpy(name):
    _assert_name_planar('dask+numpy', name)
    _assert_name_geodesic('dask+numpy', name)
    _assert_name_derived('dask+numpy', name)


@cuda_and_cupy_available
@pytest.mark.parametrize("name", [None, 'aspect'])
def test_name_consistent_cupy(name):
    _assert_name_planar('cupy', name)
    _assert_name_geodesic('cupy', name)
    _assert_name_derived('cupy', name)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.parametrize("name", [None, 'aspect'])
def test_name_consistent_dask_cupy(name):
    _assert_name_planar('dask+cupy', name)
    _assert_name_geodesic('dask+cupy', name)
    _assert_name_derived('dask+cupy', name)
