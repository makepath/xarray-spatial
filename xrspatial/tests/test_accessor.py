"""Tests for the .xrs xarray accessors."""

import inspect

import numpy as np
import pytest
import xarray as xr

import xrspatial  # noqa: F401 — triggers accessor registration
from xrspatial.accessor import (
    XrsSpatialDataArrayAccessor,
    XrsSpatialDatasetAccessor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def elevation():
    """Small synthetic elevation raster."""
    np.random.seed(42)
    data = np.random.rand(10, 10).astype(np.float64) * 1000
    return xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'x': np.arange(10, dtype=float), 'y': np.arange(10, dtype=float)},
        name='elevation',
    )


@pytest.fixture
def nir():
    """Synthetic NIR band."""
    np.random.seed(1)
    return xr.DataArray(
        np.random.rand(10, 10).astype(np.float64),
        dims=['y', 'x'],
        name='nir',
    )


@pytest.fixture
def red():
    """Synthetic red band."""
    np.random.seed(2)
    return xr.DataArray(
        np.random.rand(10, 10).astype(np.float64),
        dims=['y', 'x'],
        name='red',
    )


@pytest.fixture
def blue():
    """Synthetic blue band."""
    np.random.seed(3)
    return xr.DataArray(
        np.random.rand(10, 10).astype(np.float64),
        dims=['y', 'x'],
        name='blue',
    )


# ---------------------------------------------------------------------------
# 1. Accessor registration
# ---------------------------------------------------------------------------

def test_dataarray_accessor_registered():
    da = xr.DataArray(np.zeros((3, 3)), dims=['y', 'x'])
    assert hasattr(da, 'xrs')


def test_dataset_accessor_registered():
    ds = xr.Dataset({'a': xr.DataArray(np.zeros((3, 3)), dims=['y', 'x'])})
    assert hasattr(ds, 'xrs')


# ---------------------------------------------------------------------------
# 2. Tab-completion — expected methods present
# ---------------------------------------------------------------------------

def test_dataarray_accessor_has_expected_methods(elevation):
    names = dir(elevation.xrs)
    expected = [
        'slope', 'aspect', 'hillshade', 'curvature', 'viewshed',
        'natural_breaks', 'equal_interval', 'quantile', 'reclassify',
        'binary', 'percentiles',
        'focal_mean',
        'morph_erode', 'morph_dilate', 'morph_opening', 'morph_closing',
        'morph_gradient', 'morph_white_tophat', 'morph_black_tophat',
        'proximity', 'allocation', 'direction', 'cost_distance',
        'a_star_search',
        'zonal_stats', 'zonal_apply', 'zonal_crosstab', 'crop', 'trim',
        'regions',
        'generate_terrain', 'perlin',
        'ndvi', 'evi', 'arvi', 'savi', 'nbr', 'sipi',
        'rasterize',
        'rechunk_no_shuffle',
        'fused_overlap',
        'multi_overlap',
    ]
    for name in expected:
        assert name in names, f"Missing method: {name}"


def test_dataset_accessor_has_expected_methods():
    ds = xr.Dataset({'a': xr.DataArray(np.zeros((3, 3)), dims=['y', 'x'])})
    names = dir(ds.xrs)
    expected = [
        'slope', 'aspect', 'hillshade', 'curvature',
        'natural_breaks', 'equal_interval', 'quantile', 'reclassify',
        'binary', 'percentiles',
        'focal_mean',
        'morph_erode', 'morph_dilate', 'morph_opening', 'morph_closing',
        'morph_gradient', 'morph_white_tophat', 'morph_black_tophat',
        'proximity', 'allocation', 'direction', 'cost_distance',
        'ndvi', 'evi', 'arvi', 'savi', 'nbr', 'sipi',
        'rasterize',
    ]
    for name in expected:
        assert name in names, f"Missing method: {name}"


# ---------------------------------------------------------------------------
# 3. DataArray single-input — accessor matches direct call
# ---------------------------------------------------------------------------

def test_da_slope(elevation):
    from xrspatial.slope import slope
    expected = slope(elevation)
    result = elevation.xrs.slope()
    xr.testing.assert_identical(result, expected)


def test_da_aspect(elevation):
    from xrspatial.aspect import aspect
    expected = aspect(elevation)
    result = elevation.xrs.aspect()
    xr.testing.assert_identical(result, expected)


def test_da_hillshade(elevation):
    from xrspatial.hillshade import hillshade
    expected = hillshade(elevation, azimuth=315, angle_altitude=45)
    result = elevation.xrs.hillshade(azimuth=315, angle_altitude=45)
    xr.testing.assert_identical(result, expected)


def test_da_curvature(elevation):
    from xrspatial.curvature import curvature
    expected = curvature(elevation)
    result = elevation.xrs.curvature()
    xr.testing.assert_identical(result, expected)


def test_da_focal_mean(elevation):
    from xrspatial.focal import mean
    expected = mean(elevation, passes=1)
    result = elevation.xrs.focal_mean(passes=1)
    xr.testing.assert_identical(result, expected)


def test_da_equal_interval(elevation):
    from xrspatial.classify import equal_interval
    expected = equal_interval(elevation, k=5)
    result = elevation.xrs.equal_interval(k=5)
    xr.testing.assert_identical(result, expected)


def test_da_quantile(elevation):
    from xrspatial.classify import quantile
    expected = quantile(elevation, k=4)
    result = elevation.xrs.quantile(k=4)
    xr.testing.assert_identical(result, expected)


# ---------------------------------------------------------------------------
# 4. DataArray multi-input — multispectral
# ---------------------------------------------------------------------------

def test_da_ndvi(nir, red):
    from xrspatial.multispectral import ndvi
    expected = ndvi(nir, red)
    result = nir.xrs.ndvi(red)
    xr.testing.assert_identical(result, expected)


def test_da_evi(nir, red, blue):
    from xrspatial.multispectral import evi
    expected = evi(nir, red, blue)
    result = nir.xrs.evi(red, blue)
    xr.testing.assert_identical(result, expected)


def test_da_savi(nir, red):
    from xrspatial.multispectral import savi
    expected = savi(nir, red, soil_factor=0.5)
    result = nir.xrs.savi(red, soil_factor=0.5)
    xr.testing.assert_identical(result, expected)


def test_da_nbr(nir):
    np.random.seed(7)
    swir2 = xr.DataArray(
        np.random.rand(10, 10).astype(np.float64), dims=['y', 'x'],
    )
    from xrspatial.multispectral import nbr
    expected = nbr(nir, swir2)
    result = nir.xrs.nbr(swir2)
    xr.testing.assert_identical(result, expected)


def test_da_sipi(nir, red, blue):
    from xrspatial.multispectral import sipi
    expected = sipi(nir, red, blue)
    result = nir.xrs.sipi(red, blue)
    xr.testing.assert_identical(result, expected)


def test_da_arvi(nir, red, blue):
    from xrspatial.multispectral import arvi
    expected = arvi(nir, red, blue)
    result = nir.xrs.arvi(red, blue)
    xr.testing.assert_identical(result, expected)


# ---------------------------------------------------------------------------
# 4b. DataArray morphology — accessor matches direct call
# ---------------------------------------------------------------------------

_MORPH_KERNEL = np.ones((3, 3), dtype=np.uint8)


def test_da_morph_erode(elevation):
    from xrspatial.morphology import morph_erode
    expected = morph_erode(elevation, kernel=_MORPH_KERNEL)
    result = elevation.xrs.morph_erode(kernel=_MORPH_KERNEL)
    xr.testing.assert_identical(result, expected)


def test_da_morph_gradient(elevation):
    from xrspatial.morphology import morph_gradient
    expected = morph_gradient(elevation, kernel=_MORPH_KERNEL)
    result = elevation.xrs.morph_gradient(kernel=_MORPH_KERNEL)
    xr.testing.assert_identical(result, expected)


def test_da_morph_white_tophat(elevation):
    from xrspatial.morphology import morph_white_tophat
    expected = morph_white_tophat(elevation, kernel=_MORPH_KERNEL)
    result = elevation.xrs.morph_white_tophat(kernel=_MORPH_KERNEL)
    xr.testing.assert_identical(result, expected)


def test_da_morph_black_tophat(elevation):
    from xrspatial.morphology import morph_black_tophat
    expected = morph_black_tophat(elevation, kernel=_MORPH_KERNEL)
    result = elevation.xrs.morph_black_tophat(kernel=_MORPH_KERNEL)
    xr.testing.assert_identical(result, expected)


def test_ds_morph_gradient(elevation):
    from xrspatial.morphology import morph_gradient
    ds = xr.Dataset({'elev': elevation})
    expected = morph_gradient(ds, kernel=_MORPH_KERNEL)
    result = ds.xrs.morph_gradient(kernel=_MORPH_KERNEL)
    xr.testing.assert_identical(result, expected)


# ---------------------------------------------------------------------------
# 5. Dataset single-input — accessor matches direct call
# ---------------------------------------------------------------------------

def test_ds_slope(elevation):
    from xrspatial.slope import slope
    ds = xr.Dataset({'elev': elevation})
    expected = slope(ds)
    result = ds.xrs.slope()
    xr.testing.assert_identical(result, expected)


def test_ds_hillshade(elevation):
    from xrspatial.hillshade import hillshade
    ds = xr.Dataset({'elev': elevation})
    expected = hillshade(ds, azimuth=315)
    result = ds.xrs.hillshade(azimuth=315)
    xr.testing.assert_identical(result, expected)


# ---------------------------------------------------------------------------
# 6. Dataset multi-input — multispectral band mapping
# ---------------------------------------------------------------------------

def test_ds_ndvi(nir, red):
    from xrspatial.multispectral import ndvi
    ds = xr.Dataset({'b8': nir, 'b4': red})
    expected = ndvi(ds, nir='b8', red='b4')
    result = ds.xrs.ndvi(nir='b8', red='b4')
    xr.testing.assert_identical(result, expected)


def test_ds_evi(nir, red, blue):
    from xrspatial.multispectral import evi
    ds = xr.Dataset({'b8': nir, 'b4': red, 'b2': blue})
    expected = evi(ds, nir='b8', red='b4', blue='b2')
    result = ds.xrs.evi(nir='b8', red='b4', blue='b2')
    xr.testing.assert_identical(result, expected)


# ---------------------------------------------------------------------------
# 7. Rasterize accessor
# ---------------------------------------------------------------------------

def _make_template(width=20, height=20, bounds=(0, 0, 100, 100)):
    xmin, ymin, xmax, ymax = bounds
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height
    x = np.linspace(xmin + px / 2, xmax - px / 2, width)
    y = np.linspace(ymax - py / 2, ymin + py / 2, height)
    return xr.DataArray(
        np.zeros((height, width), dtype=np.float64),
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
    )


def test_da_rasterize():
    pytest.importorskip('shapely')
    from shapely.geometry import box
    from xrspatial.rasterize import rasterize

    template = _make_template()
    gdf = pytest.importorskip('geopandas').GeoDataFrame(
        {'val': [1.0]}, geometry=[box(10, 10, 50, 50)])

    expected = rasterize(gdf, like=template, column='val')
    result = template.xrs.rasterize(gdf, column='val')
    xr.testing.assert_identical(result, expected)


def test_da_rasterize_uses_coords():
    """Accessor output matches the template grid dimensions and coords."""
    pytest.importorskip('shapely')
    from shapely.geometry import box

    template = _make_template(width=30, height=15, bounds=(0, 0, 60, 30))
    pairs = [(box(5, 5, 25, 25), 7.0)]

    result = template.xrs.rasterize(pairs)
    assert result.sizes['x'] == 30
    assert result.sizes['y'] == 15
    np.testing.assert_allclose(result.coords['x'].values,
                               template.coords['x'].values)
    np.testing.assert_allclose(result.coords['y'].values,
                               template.coords['y'].values)


def test_ds_rasterize():
    pytest.importorskip('shapely')
    from shapely.geometry import box
    from xrspatial.rasterize import rasterize

    template = _make_template()
    ds = xr.Dataset({'elev': template})
    pairs = [(box(10, 10, 50, 50), 1.0)]

    expected = rasterize(pairs, like=template)
    result = ds.xrs.rasterize(pairs)
    xr.testing.assert_identical(result, expected)


def test_ds_rasterize_no_valid_var():
    ds = xr.Dataset({'a': xr.DataArray(np.zeros(5), dims=['z'])})
    with pytest.raises(ValueError, match="no 2D variable"):
        ds.xrs.rasterize([])


# ---------------------------------------------------------------------------
# Optional matplotlib (the `plot` extra) — issue #2494
# ---------------------------------------------------------------------------

def test_compute_imports_without_matplotlib():
    """Core compute modules import in a fresh interpreter with no matplotlib.

    Runs in a subprocess so the import happens against a clean module cache
    with matplotlib blocked, rather than reloading already-imported modules
    in the test session.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        sys.modules['matplotlib'] = None
        sys.modules['matplotlib.pyplot'] = None

        import xrspatial  # noqa: F401
        import xrspatial.focal  # noqa: F401
        import xrspatial.zonal  # noqa: F401
        import xrspatial.dasymetric  # noqa: F401

        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pass
        else:
            raise SystemExit('matplotlib was unexpectedly importable')
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_require_matplotlib_message(monkeypatch):
    """The helper points users at the `plot` extra when matplotlib is gone."""
    import sys

    from xrspatial.accessor import _require_matplotlib

    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    with pytest.raises(ImportError, match=r"xarray-spatial\[plot\]"):
        _require_matplotlib()


def test_da_plot_without_matplotlib_raises(monkeypatch, elevation):
    """``.xrs.plot()`` raises the friendly error when matplotlib is absent."""
    import sys

    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', None)
    with pytest.raises(ImportError, match=r"xarray-spatial\[plot\]"):
        elevation.xrs.plot()


def test_ds_plot_without_matplotlib_raises(monkeypatch, elevation):
    """Dataset ``.xrs.plot()`` raises the friendly error too."""
    import sys

    ds = xr.Dataset({'elev': elevation})
    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', None)
    with pytest.raises(ImportError, match=r"xarray-spatial\[plot\]"):
        ds.xrs.plot()


# ---------------------------------------------------------------------------
# 9. help() — standalone docstrings surfaced on accessor methods (#2981)
# ---------------------------------------------------------------------------

# The hydrology unified wrappers carry only this generic dispatcher stub; the
# accessor must surface the documented *_d8 variant docs instead.
_GENERIC_HYDRO_DOC = 'Map routing algorithm names'


@pytest.mark.parametrize('method_name, source', [
    ('slope', 'slope'),
    ('aspect', 'aspect'),
    ('hillshade', 'hillshade'),
    ('curvature', 'curvature'),
    ('focal_mean', 'mean'),            # method name differs from function
    ('ndvi', 'ndvi'),
    ('fill', 'fill_d8'),              # hydro wrapper -> documented d8 variant
    ('watershed', 'watershed_d8'),
])
def test_accessor_docstring_matches_source(method_name, source):
    func = getattr(xrspatial, source)
    method = getattr(XrsSpatialDataArrayAccessor, method_name)
    assert inspect.getdoc(method), f'{method_name} has no docstring'
    assert inspect.getdoc(method) == inspect.getdoc(func)


def test_help_text_surfaces_on_instance(elevation):
    """help(da.xrs.slope) sees the same docstring as help(slope)."""
    from xrspatial import slope
    assert inspect.getdoc(elevation.xrs.slope) == inspect.getdoc(slope)


def test_handwritten_method_docs_preserved():
    """Methods with their own docstrings are not overwritten."""
    assert XrsSpatialDataArrayAccessor.plot.__doc__.lstrip().startswith(
        'Plot the DataArray'
    )
    assert XrsSpatialDataArrayAccessor.to_geotiff.__doc__.lstrip().startswith(
        'Write this DataArray'
    )


@pytest.mark.parametrize(
    'cls', [XrsSpatialDataArrayAccessor, XrsSpatialDatasetAccessor]
)
def test_every_public_method_documented(cls):
    """Drift guard: every public accessor method has a useful docstring."""
    undocumented = []
    for name, member in vars(cls).items():
        if name.startswith('_') or not callable(member):
            continue
        doc = (member.__doc__ or '').strip()
        if not doc or doc.startswith(_GENERIC_HYDRO_DOC):
            undocumented.append(name)
    assert not undocumented, f'methods lacking a useful docstring: {undocumented}'


# ---------------------------------------------------------------------------
# 10. Hydrology accessor methods import and run (regression for #2981)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method_name', [
    'fill', 'flow_direction', 'flow_accumulation', 'sink', 'basin', 'flow_length',
])
def test_hydro_accessor_matches_direct(method_name, elevation):
    """Single-input hydro accessor methods match the standalone function."""
    func = getattr(xrspatial, method_name)
    expected = func(elevation)
    result = getattr(elevation.xrs, method_name)()
    xr.testing.assert_identical(result, expected)


_ALL_HYDRO_METHODS = [
    'flow_direction', 'flow_direction_dinf', 'flow_direction_mfd',
    'flow_accumulation', 'flow_accumulation_mfd', 'watershed', 'basin', 'basins',
    'sink', 'fill', 'stream_order', 'stream_link', 'snap_pour_point',
    'flow_path', 'flow_length', 'twi', 'hand',
]


@pytest.mark.parametrize('method_name', _ALL_HYDRO_METHODS)
def test_hydro_accessor_delegation_resolves(method_name, elevation):
    """The lazy import no longer points at a removed per-algorithm module.

    Methods needing extra positional args raise TypeError (or another
    runtime error) once the import succeeds; only a ModuleNotFoundError
    means the delegation is still broken.
    """
    try:
        getattr(elevation.xrs, method_name)()
    except ModuleNotFoundError as exc:  # pragma: no cover - the bug we fixed
        pytest.fail(f'{method_name} delegation still broken: {exc}')
    except Exception:
        pass  # any non-import error proves the import path resolved
