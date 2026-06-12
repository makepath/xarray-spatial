"""Dask graphs label xrspatial compute tasks as ``xrspatial.<tool>`` (#3250)."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial import (
    aspect,
    bilateral,
    curvature,
    flow_direction_mfd,
    flow_length_mfd,
    generate_terrain,
    hillshade,
    preview,
    slope,
    surface_allocation,
    surface_direction,
    surface_distance,
)
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
