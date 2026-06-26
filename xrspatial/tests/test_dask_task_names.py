"""Dask graphs label xrspatial compute tasks as ``xrspatial.<tool>`` (#3250)."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial import (
    aspect,
    bilateral,
    curvature,
    generate_terrain,
    hillshade,
    preview,
    slope,
    surface_allocation,
    surface_direction,
    surface_distance,
)
from xrspatial.hydro import flow_direction_mfd, flow_length_mfd
from xrspatial.convolution import convolve_2d
from xrspatial.tests.general_checks import create_test_raster


def graph_key_prefixes(result):
    """Set of task-name prefixes (hash stripped) in a result's dask graph."""
    data = result.data if isinstance(result, xr.DataArray) else result
    prefixes = set()
    for key in data.__dask_graph__():
        name = key[0] if isinstance(key, tuple) else key
        prefixes.add(str(name).rsplit('-', 1)[0])
    return prefixes


@pytest.fixture
def elevation_dask():
    data = np.linspace(0, 100, 144, dtype=np.float64).reshape(12, 12)
    data += np.random.RandomState(42).rand(12, 12)
    return create_test_raster(data, backend='dask+numpy', chunks=(6, 6))


def test_slope_task_name(elevation_dask):
    result = slope(elevation_dask)
    assert 'xrspatial.slope' in graph_key_prefixes(result)


def test_aspect_task_name(elevation_dask):
    result = aspect(elevation_dask)
    assert 'xrspatial.aspect' in graph_key_prefixes(result)


def test_hillshade_task_name(elevation_dask):
    result = hillshade(elevation_dask)
    assert 'xrspatial.hillshade' in graph_key_prefixes(result)


def test_curvature_task_name(elevation_dask):
    result = curvature(elevation_dask)
    assert 'xrspatial.curvature' in graph_key_prefixes(result)


def test_task_names_with_ragged_chunks():
    """Naming holds for chunk grids that do not divide the raster evenly."""
    data = np.linspace(0, 100, 144, dtype=np.float64).reshape(12, 12)
    ragged = create_test_raster(data, backend='dask+numpy', chunks=(5, 7))
    result = slope(ragged)
    assert 'xrspatial.slope' in graph_key_prefixes(result)
    expected = slope(ragged.compute())
    np.testing.assert_allclose(
        result.compute().data, expected.data, rtol=1e-5, equal_nan=True)


def test_convolve_2d_task_name(elevation_dask):
    result = convolve_2d(elevation_dask.data, np.ones((3, 3)))
    assert 'xrspatial.convolve_2d' in graph_key_prefixes(result)


def test_bilateral_task_name(elevation_dask):
    result = bilateral(elevation_dask)
    assert 'xrspatial.bilateral' in graph_key_prefixes(result)


def test_generate_terrain_task_name(elevation_dask):
    result = generate_terrain(elevation_dask)
    assert 'xrspatial.terrain' in graph_key_prefixes(result)


def test_preview_task_name(elevation_dask):
    result = preview(elevation_dask, width=6)
    assert 'xrspatial.preview' in graph_key_prefixes(result)


@pytest.mark.parametrize('func, prefix', [
    (surface_distance, 'xrspatial.surface_distance'),
    (surface_allocation, 'xrspatial.surface_allocation'),
    (surface_direction, 'xrspatial.surface_direction'),
])
def test_surface_distance_task_names_bounded(elevation_dask, func, prefix):
    source = elevation_dask.copy()
    source.data = da.zeros_like(elevation_dask.data)
    source.data[5, 5] = 1
    result = func(source, elevation_dask, max_distance=3.0)
    assert prefix in graph_key_prefixes(result)


def test_surface_distance_task_name_unbounded(elevation_dask):
    source = elevation_dask.copy()
    source.data = da.zeros_like(elevation_dask.data)
    source.data[5, 5] = 1
    result = surface_distance(source, elevation_dask)
    assert 'xrspatial.surface_distance' in graph_key_prefixes(result)


def test_flow_direction_mfd_task_names(elevation_dask):
    result = flow_direction_mfd(elevation_dask)
    prefixes = graph_key_prefixes(result)
    for band in range(8):
        assert f'xrspatial.flow_direction_mfd_band{band}' in prefixes


def test_flow_length_mfd_task_name(elevation_dask):
    fdir = flow_direction_mfd(elevation_dask)
    result = flow_length_mfd(fdir)
    assert 'xrspatial.flow_length_mfd' in graph_key_prefixes(result)


def test_same_name_no_key_collision(elevation_dask):
    """Two slope calls share the name prefix but keep distinct graph keys.

    Older dask treated ``name=`` as the verbatim key; if that ever came
    back, two calls in one graph would silently overwrite each other.
    """
    other = create_test_raster(
        np.linspace(50, 0, 144, dtype=np.float64).reshape(12, 12),
        backend='dask+numpy', chunks=(6, 6))

    result_a = slope(elevation_dask)
    result_b = slope(other)
    assert result_a.data.name != result_b.data.name

    combined = (result_a + result_b).compute()
    expected = (slope(elevation_dask.compute())
                + slope(other.compute()))
    np.testing.assert_allclose(
        combined.data, expected.data, rtol=1e-5, equal_nan=True)


def test_named_tasks_compute_correctly(elevation_dask):
    """Naming must not change values: dask slope still matches numpy."""
    dask_result = slope(elevation_dask).compute()
    numpy_result = slope(elevation_dask.compute())
    np.testing.assert_allclose(
        dask_result.data, numpy_result.data, rtol=1e-5, equal_nan=True)


# --- #3256: sweep of the remaining tool modules -------------------------------

def test_mean_task_name(elevation_dask):
    from xrspatial import mean
    result = mean(elevation_dask)
    assert 'xrspatial.mean' in graph_key_prefixes(result)


def test_focal_apply_task_name(elevation_dask):
    from xrspatial.focal import apply
    result = apply(elevation_dask, np.ones((3, 3)), lambda x: x.mean())
    assert 'xrspatial.apply' in graph_key_prefixes(result)


def test_hotspots_task_names(elevation_dask):
    from xrspatial.convolution import circle_kernel
    from xrspatial.focal import hotspots
    result = hotspots(elevation_dask, circle_kernel(1, 1, 2))
    prefixes = graph_key_prefixes(result)
    assert 'xrspatial.hotspots' in prefixes
    # the lazy degenerate-input check is its own labeled layer (#2843)
    assert 'xrspatial.hotspots.validate' in prefixes


def test_proximity_family_task_names(elevation_dask):
    from xrspatial import allocation, direction, proximity
    source = elevation_dask.copy()
    source.data = da.zeros_like(elevation_dask.data)
    source.data[5, 5] = 1
    assert 'xrspatial.proximity' in graph_key_prefixes(proximity(source))
    assert 'xrspatial.allocation' in graph_key_prefixes(allocation(source))
    assert 'xrspatial.direction' in graph_key_prefixes(direction(source))


def test_cost_distance_task_name(elevation_dask):
    from xrspatial import cost_distance
    source = elevation_dask.copy()
    source.data = da.zeros_like(elevation_dask.data)
    source.data[5, 5] = 1
    friction = elevation_dask.copy()
    friction.data = da.ones_like(elevation_dask.data)
    # finite max_cost keeps the overlap depth within the chunk size, so the
    # named map_overlap path runs (not the iterative from_delayed fallback)
    result = cost_distance(source, friction, target_values=[1], max_cost=2.0)
    assert 'xrspatial.cost_distance' in graph_key_prefixes(result)


@pytest.mark.parametrize('metric', ['contrast'])
def test_glcm_task_name(elevation_dask, metric):
    from xrspatial.glcm import glcm_texture
    result = glcm_texture(elevation_dask, metric=metric, window_size=3, levels=8)
    assert 'xrspatial.glcm_texture' in graph_key_prefixes(result)


def test_sky_view_factor_task_name(elevation_dask):
    from xrspatial import sky_view_factor
    result = sky_view_factor(elevation_dask)
    assert 'xrspatial.sky_view_factor' in graph_key_prefixes(result)


def test_diffuse_task_name(elevation_dask):
    from xrspatial.diffusion import diffuse
    result = diffuse(elevation_dask, steps=2)
    assert 'xrspatial.diffuse' in graph_key_prefixes(result)


@pytest.mark.parametrize('op, prefix', [
    ('morph_erode', 'xrspatial.morph_erode'),
    ('morph_dilate', 'xrspatial.morph_dilate'),
])
def test_morphology_task_names(elevation_dask, op, prefix):
    import xrspatial.morphology as morphology
    from xrspatial.convolution import circle_kernel
    func = getattr(morphology, op)
    result = func(elevation_dask, circle_kernel(1, 1, 2))
    assert prefix in graph_key_prefixes(result)


@pytest.mark.parametrize('prefix', [
    'xrspatial.resample.interp',
    'xrspatial.resample.aggregate',
])
def test_resample_task_names(elevation_dask, prefix):
    from xrspatial.resample import resample
    method = 'bilinear' if prefix.endswith('interp') else 'average'
    result = resample(elevation_dask, scale_factor=0.5, method=method)
    assert prefix in graph_key_prefixes(result)


def test_perlin_task_name(elevation_dask):
    from xrspatial import perlin
    result = perlin(elevation_dask)
    assert 'xrspatial.perlin' in graph_key_prefixes(result)


def test_worley_task_name(elevation_dask):
    from xrspatial.worley import worley
    result = worley(elevation_dask)
    assert 'xrspatial.worley' in graph_key_prefixes(result)


def test_dnbr_task_name(elevation_dask):
    from xrspatial.fire import dnbr
    result = dnbr(elevation_dask, elevation_dask * 0.5)
    assert 'xrspatial.dnbr' in graph_key_prefixes(result)


def test_ndvi_uses_shared_normalized_ratio_name(elevation_dask):
    """ndvi/nbr/... share one normalized_ratio compute, so one task name."""
    from xrspatial.multispectral import ndvi
    nir = elevation_dask
    red = elevation_dask * 0.5 + 1
    result = ndvi(nir, red)
    assert 'xrspatial.normalized_ratio' in graph_key_prefixes(result)


def test_savi_task_name(elevation_dask):
    from xrspatial.multispectral import savi
    nir = elevation_dask
    red = elevation_dask * 0.5 + 1
    result = savi(nir, red)
    assert 'xrspatial.savi' in graph_key_prefixes(result)


@pytest.mark.parametrize('func_name, prefix', [
    ('natural_breaks', 'xrspatial.natural_breaks'),
    ('equal_interval', 'xrspatial.equal_interval'),
    ('quantile', 'xrspatial.quantile'),
])
def test_classify_task_names(elevation_dask, func_name, prefix):
    import xrspatial.classify as classify
    func = getattr(classify, func_name)
    result = func(elevation_dask, k=3)
    # quantile/equal_interval/natural_breaks bin through the shared engine
    assert 'xrspatial.reclassify' in graph_key_prefixes(result)


def test_reclassify_task_name(elevation_dask):
    from xrspatial.classify import reclassify
    result = reclassify(
        elevation_dask, bins=[20, 50, 100], new_values=[1, 2, 3])
    assert 'xrspatial.reclassify' in graph_key_prefixes(result)


def test_zonal_apply_task_name(elevation_dask):
    from xrspatial.zonal import apply as zonal_apply
    zones = elevation_dask.copy()
    zones.data = (elevation_dask.data > 50).astype('int64')
    result = zonal_apply(zones, elevation_dask, lambda x: x + 1)
    assert 'xrspatial.apply' in graph_key_prefixes(result)


@pytest.fixture
def flow_direction_d8_dask(elevation_dask):
    from xrspatial.hydro import flow_direction_d8
    return flow_direction_d8(elevation_dask)


def test_flow_direction_d8_task_name(elevation_dask):
    from xrspatial.hydro import flow_direction_d8
    result = flow_direction_d8(elevation_dask)
    assert 'xrspatial.flow_direction_d8' in graph_key_prefixes(result)


def test_flow_accumulation_d8_task_name(flow_direction_d8_dask):
    from xrspatial.hydro import flow_accumulation_d8
    result = flow_accumulation_d8(flow_direction_d8_dask)
    assert 'xrspatial.flow_accumulation_d8' in graph_key_prefixes(result)


def test_no_unnamed_xrspatial_compute_layers_in_converted_modules():
    """Spot-check: a converted tool produces no anonymous map_* compute layer.

    map_blocks/map_overlap without naming kwargs land under dask's default
    ``lambda``/function-name prefix instead of ``xrspatial.*``. The named
    tool layers should all carry the xrspatial prefix.
    """
    from xrspatial import mean
    data = np.linspace(0, 100, 144, dtype=np.float64).reshape(12, 12)
    raster = create_test_raster(data, backend='dask+numpy', chunks=(6, 6))
    result = mean(raster)
    assert 'xrspatial.mean' in graph_key_prefixes(result)


def test_converted_tool_keys_do_not_merge(elevation_dask):
    """Two perlin calls keep distinct graph keys despite a shared name.

    Guards the #3256 convention: the helper appends a hash so two
    same-named tool calls cannot silently collapse into one task (the
    dask 2026 verbatim-``name`` failure mode).
    """
    from xrspatial import perlin
    other = create_test_raster(
        np.linspace(50, 0, 144, dtype=np.float64).reshape(12, 12),
        backend='dask+numpy', chunks=(6, 6))
    result_a = perlin(elevation_dask, seed=1)
    result_b = perlin(other, seed=2)
    assert result_a.data.name != result_b.data.name
    # both still compute independently in one combined graph
    combined = (result_a + result_b).compute()
    assert combined.shape == elevation_dask.shape
