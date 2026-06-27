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
        'a_star_search', 'multi_stop_search',
        'zonal_stats', 'zonal_apply', 'zonal_crosstab', 'crop', 'trim',
        'regions',
        'generate_terrain', 'perlin',
        'ndvi', 'evi', 'arvi', 'savi', 'nbr', 'sipi',
        'rasterize',
        'coregister',
        'rechunk_no_shuffle',
        'fused_overlap',
        'multi_overlap',
        'validate',
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
        'multi_stop_search',
        'ndvi', 'evi', 'arvi', 'savi', 'nbr', 'sipi',
        'rasterize',
        'validate',
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
# 4c. DataArray pathfinding — accessor matches direct call
# ---------------------------------------------------------------------------

def test_da_a_star_search(elevation):
    from xrspatial.pathfinding import a_star_search
    start, goal = (0, 0), (9, 9)
    expected = a_star_search(elevation, start, goal)
    result = elevation.xrs.a_star_search(start, goal)
    xr.testing.assert_identical(result, expected)


def test_da_multi_stop_search(elevation):
    from xrspatial.pathfinding import multi_stop_search
    waypoints = [(0, 0), (5, 5), (9, 9)]
    expected = multi_stop_search(elevation, waypoints)
    result = elevation.xrs.multi_stop_search(waypoints)
    xr.testing.assert_identical(result, expected)


def test_da_multi_stop_search_kwargs(elevation):
    from xrspatial.pathfinding import multi_stop_search
    waypoints = [(0, 0), (9, 0), (9, 9)]
    expected = multi_stop_search(elevation, waypoints, optimize_order=True)
    result = elevation.xrs.multi_stop_search(waypoints, optimize_order=True)
    xr.testing.assert_identical(result, expected)


def test_ds_multi_stop_search(elevation):
    from xrspatial.pathfinding import multi_stop_search
    ds = xr.Dataset({'a': elevation, 'b': elevation + 100})
    waypoints = [(0, 0), (5, 5), (9, 9)]
    expected = multi_stop_search(ds, waypoints)
    result = ds.xrs.multi_stop_search(waypoints)
    xr.testing.assert_identical(result, expected)
    # supports_dataset routes each variable through its own surface
    assert set(result.data_vars) == {'a', 'b'}


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
    # hydro *_d8 doc sources are internal to xrspatial.hydro, not top level
    func = getattr(xrspatial, source, None) or getattr(xrspatial.hydro, source)
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


# ---------------------------------------------------------------------------
# 11. Rich accessor repr — discoverable tool listing by category (#3476)
# ---------------------------------------------------------------------------

from xrspatial.accessor import (  # noqa: E402
    _accessor_categories,
    _accessor_repr_html,
    _accessor_repr_text,
    _public_accessor_methods,
)

_ACCESSOR_CLASSES = [XrsSpatialDataArrayAccessor, XrsSpatialDatasetAccessor]


@pytest.mark.parametrize('cls', _ACCESSOR_CLASSES)
def test_categories_cover_every_public_method(cls):
    """Every public method falls under exactly one category banner.

    Guards against a method added without a ``# ---- Category ----`` banner
    above it, which would silently drop it from the repr.
    """
    categorized = [m for _, methods in _accessor_categories(cls) for m in methods]
    assert sorted(categorized) == sorted(_public_accessor_methods(cls))
    assert len(categorized) == len(set(categorized)), 'method listed twice'


@pytest.mark.parametrize('cls', _ACCESSOR_CLASSES)
def test_repr_text_lists_categories_and_methods(cls):
    text = _accessor_repr_text(cls)
    assert cls.__name__ in text
    assert '.xrs.<name>(...)' in text
    # A representative category banner and a method beneath it.
    assert 'Surface:' in text
    assert 'slope' in text


def test_da_repr_distinct_from_ds_repr():
    da_text = _accessor_repr_text(XrsSpatialDataArrayAccessor)
    ds_text = _accessor_repr_text(XrsSpatialDatasetAccessor)
    # Only the DataArray accessor exposes viewshed / interpolation tools.
    assert 'viewshed' in da_text
    assert 'viewshed' not in ds_text


def test_repr_on_instance(elevation):
    """repr(da.xrs) renders the categorized listing, not the object id."""
    text = repr(elevation.xrs)
    assert 'object at 0x' not in text
    assert 'ndvi' in text


@pytest.mark.parametrize('cls', _ACCESSOR_CLASSES)
def test_repr_html_renders_methods(cls):
    out = _accessor_repr_html(cls)
    assert '<table>' in out
    assert '<code>slope</code>' in out
    assert 'xarray-spatial' in out


def test_repr_html_escapes_method_names(monkeypatch):
    """HTML-special characters in a method name are escaped, not injected."""
    monkeypatch.setattr(
        'xrspatial.accessor._accessor_categories',
        lambda cls: (('Danger', ('<script>',)),),
    )
    out = _accessor_repr_html(XrsSpatialDataArrayAccessor)
    assert '<script>' not in out
    assert '&lt;script&gt;' in out


def test_categories_fallback_when_source_unavailable():
    """A class with no retrievable source yields a flat fallback listing."""
    Dummy = type('DummyAccessor', (), {'foo': lambda self: None})
    assert _accessor_categories(Dummy) == ()
    # Both reprs fall back to the same single "Available tools" group.
    text = _accessor_repr_text(Dummy)
    assert 'Available tools:' in text
    assert 'foo' in text
    html_out = _accessor_repr_html(Dummy)
    assert 'Available tools' in html_out
    assert '<code>foo</code>' in html_out


# ---------------------------------------------------------------------------
# 12. Per-tool repr — da.xrs.slope shows slope's own info, not the catalog (#3478)
# ---------------------------------------------------------------------------

from xrspatial.accessor import _AccessorTool  # noqa: E402


def test_tool_repr_is_scoped_to_the_tool(elevation):
    """repr(da.xrs.slope) shows slope's signature + docstring, not the catalog."""
    from xrspatial import slope

    text = repr(elevation.xrs.slope)
    assert text.startswith('.xrs.slope(')
    # The catalog repr's header and other categories must not leak in.
    assert 'call as: .xrs.<name>(...)' not in text
    assert 'Surface:' not in text
    assert 'Hydrology:' not in text
    # The slope docstring is surfaced instead.
    assert inspect.getdoc(slope) in text


def test_tool_repr_distinct_per_tool(elevation):
    """Different tools render different reprs (no shared catalog block)."""
    assert repr(elevation.xrs.slope) != repr(elevation.xrs.aspect)
    assert repr(elevation.xrs.aspect).startswith('.xrs.aspect(')


def test_tool_access_returns_callable_proxy(elevation):
    """The wrapped tool is an _AccessorTool that still forwards calls."""
    from xrspatial.slope import slope

    tool = elevation.xrs.slope
    assert isinstance(tool, _AccessorTool)
    xr.testing.assert_identical(tool(), slope(elevation))


def test_tool_proxy_preserves_help_metadata(elevation):
    """help()/inspect see the delegated docstring and the method name."""
    from xrspatial import slope

    tool = elevation.xrs.slope
    assert inspect.getdoc(tool) == inspect.getdoc(slope)
    assert tool.__name__ == 'slope'


def test_tool_repr_html_scoped_to_tool(elevation):
    """The notebook repr of a single tool is the tool's own block, not a table."""
    out = elevation.xrs.slope._repr_html_()
    assert '<pre' in out
    assert '.xrs.slope(' in out
    # No catalog table / other categories.
    assert '<table>' not in out
    assert 'Hydrology' not in out


def test_tool_repr_on_dataset_accessor(elevation):
    """The Dataset accessor scopes its per-tool repr the same way."""
    ds = xr.Dataset({'elev': elevation})
    text = repr(ds.xrs.slope)
    assert text.startswith('.xrs.slope(')
    assert 'Surface:' not in text


def test_catalog_repr_still_works_with_proxy(elevation):
    """The accessor-level catalog repr is unchanged by the per-tool wrapping."""
    text = repr(elevation.xrs)
    assert 'Surface:' in text
    assert 'slope' in text
    # _repr_html_ on the accessor itself still renders the table.
    assert '<table>' in elevation.xrs._repr_html_()


# ---------------------------------------------------------------------------
# 12. coregister — put your own data on a template grid (#3561)
# ---------------------------------------------------------------------------

def _conus_grid(resolution=20000):
    """Small CONUS template (EPSG:5070) to coregister onto."""
    from xrspatial import from_template
    return from_template('conus', resolution=resolution)


def _latlon_constant(value=7.0, shape=(50, 70)):
    """A constant-valued EPSG:4326 raster covering the lower 48."""
    ys = np.linspace(50, 24, shape[0])
    xs = np.linspace(-125, -66, shape[1])
    return xr.DataArray(
        np.full(shape, value, dtype='float32'),
        dims=['y', 'x'], coords={'y': ys, 'x': xs}, attrs={'crs': 4326},
    )


def test_coregister_raster_lands_on_template_grid():
    pytest.importorskip('pyproj')
    grid = _conus_grid()
    out = grid.xrs.coregister(_latlon_constant(value=7.0))
    # exact grid match: same shape, same coordinates, template CRS carried over
    assert out.shape == grid.shape
    assert np.array_equal(out['x'].values, grid['x'].values)
    assert np.array_equal(out['y'].values, grid['y'].values)
    assert out.attrs['crs'] == grid.attrs['crs'] == 5070
    # a constant field stays constant through the bilinear resample
    interior = out.values[np.isfinite(out.values)]
    assert interior.size > 0
    np.testing.assert_allclose(interior, 7.0, atol=1e-4)


def test_coregister_vector_matches_rasterize_coregister():
    pytest.importorskip('pyproj')
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import box
    grid = _conus_grid()
    gdf = gpd.GeoDataFrame({'v': [1]}, geometry=[box(-110, 35, -95, 45)], crs=4326)
    out = grid.xrs.coregister(gdf)
    direct = grid.xrs.rasterize(gdf, coregister=True)
    assert np.array_equal(np.nan_to_num(out.values), np.nan_to_num(direct.values))
    assert out.shape == grid.shape


def test_coregister_rejects_unsupported_type():
    grid = _conus_grid()
    with pytest.raises(TypeError, match='coregister expects'):
        grid.xrs.coregister(123)


def test_coregister_raster_requires_target_crs():
    pytest.importorskip('pyproj')
    # a caller grid with no detectable CRS cannot anchor the reprojection
    grid = xr.DataArray(
        np.zeros((10, 10), dtype='float32'), dims=['y', 'x'],
        coords={'y': np.arange(10, dtype=float), 'x': np.arange(10, dtype=float)},
    )
    with pytest.raises(ValueError, match='needs a CRS'):
        grid.xrs.coregister(_latlon_constant(shape=(10, 10)))


def test_coregister_raster_nearest_resampling_runs():
    pytest.importorskip('pyproj')
    grid = _conus_grid()
    out = grid.xrs.coregister(_latlon_constant(value=3.0), resampling='nearest')
    assert out.shape == grid.shape
    interior = out.values[np.isfinite(out.values)]
    # nearest keeps the exact source value (no interpolation blur)
    assert set(np.unique(interior)).issubset({3.0})


def test_coregister_raster_handles_ascending_y_target():
    pytest.importorskip('pyproj')
    # A caller grid stored with ascending y must still line up by geography:
    # the north (large-y) row should hold the northern source value, not get
    # silently flipped by the coordinate snap.
    ys = np.linspace(2.7e5, 3.2e6, 40)   # ascending y (EPSG:5070 metres)
    xs = np.linspace(-2.3e6, 2.3e6, 60)
    grid = xr.DataArray(
        np.zeros((40, 60), dtype='float32'), dims=['y', 'x'],
        coords={'y': ys, 'x': xs}, attrs={'crs': 5070, 'res': (76667.0, 76250.0)},
    )
    # source value == latitude, so larger value is further north
    sys = np.linspace(50, 24, 50)
    sxs = np.linspace(-125, -66, 70)
    src = xr.DataArray(
        np.repeat(sys[:, None], 70, axis=1).astype('float32'),
        dims=['y', 'x'], coords={'y': sys, 'x': sxs}, attrs={'crs': 4326},
    )
    out = grid.xrs.coregister(src)
    assert np.array_equal(out['y'].values, ys)
    col = out.values[:, out.shape[1] // 2]
    fin = col[np.isfinite(col)]
    # ascending y: last row is the northernmost, so it holds the larger latitude
    assert fin[-1] > fin[0]


def test_coregister_raster_preserves_dask_backend():
    pytest.importorskip('pyproj')
    da = pytest.importorskip('dask.array')
    grid = _conus_grid()
    src = _latlon_constant(value=2.0)
    src.data = da.from_array(src.data, chunks=(25, 35))
    out = grid.xrs.coregister(src)
    assert isinstance(out.data, da.Array)
    assert out.shape == grid.shape
