"""Comprehensive input-validation tests for every public function.

Ensures that every public API entry point rejects bad input with a clear
TypeError or ValueError *before* hitting numba JIT or backend dispatch.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.aspect import aspect
from xrspatial.bump import bump
from xrspatial.classify import (
    binary,
    box_plot,
    equal_interval,
    head_tail_breaks,
    maximum_breaks,
    natural_breaks,
    percentiles,
    quantile,
    reclassify,
    std_mean,
)
from xrspatial.cost_distance import cost_distance
from xrspatial.curvature import curvature
from xrspatial.focal import apply as focal_apply, focal_stats, hotspots, mean
from xrspatial.hillshade import hillshade
from xrspatial.multispectral import (
    arvi,
    ebbi,
    evi,
    gci,
    nbr,
    nbr2,
    ndmi,
    ndvi,
    savi,
    sipi,
    true_color,
)
from xrspatial.perlin import perlin
from xrspatial.proximity import allocation, direction, proximity
from xrspatial.slope import slope
from xrspatial.terrain import generate_terrain
from xrspatial.viewshed import viewshed
from xrspatial.zonal import (
    apply as zonal_apply,
    crop,
    crosstab,
    regions,
    stats as zonal_stats,
    trim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_raster_2d = xr.DataArray(
    np.zeros((5, 5), dtype=np.float64), dims=['y', 'x']
)
_kernel_3x3 = np.ones((3, 3), dtype=np.float64)
_friction_2d = xr.DataArray(
    np.ones((5, 5), dtype=np.float64), dims=['y', 'x']
)
_zones_int = xr.DataArray(
    np.ones((5, 5), dtype=np.int32), dims=['y', 'x']
)


# ---------------------------------------------------------------------------
# Type checks — every public function rejects non-DataArray input
# ---------------------------------------------------------------------------
class TestTypeChecks:
    """Every public function rejects non-DataArray input with TypeError."""

    # Surface analysis
    @pytest.mark.parametrize('func,args', [
        (slope, ()),
        (aspect, ()),
        (hillshade, ()),
        (curvature, ()),
    ])
    def test_surface_rejects_ndarray(self, func, args):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            func(np.zeros((5, 5)), *args)

    # Focal
    def test_mean_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            mean(np.zeros((5, 5)))

    def test_focal_apply_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            focal_apply(np.zeros((5, 5)), _kernel_3x3)

    def test_focal_stats_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            focal_stats(np.zeros((5, 5)), _kernel_3x3)

    def test_hotspots_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            hotspots(np.zeros((5, 5)), _kernel_3x3)

    # Proximity
    @pytest.mark.parametrize('func', [proximity, allocation, direction])
    def test_proximity_rejects_ndarray(self, func):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            func(np.zeros((5, 5)))

    # Cost distance
    def test_cost_distance_rejects_ndarray_raster(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            cost_distance(np.zeros((5, 5)), _friction_2d)

    def test_cost_distance_rejects_ndarray_friction(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            cost_distance(_raster_2d, np.ones((5, 5)))

    # Zonal
    def test_zonal_stats_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            zonal_stats(np.zeros((5, 5)), _raster_2d)

    def test_crosstab_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            crosstab(np.zeros((5, 5)), _raster_2d)

    def test_zonal_apply_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            zonal_apply(np.zeros((5, 5)), _raster_2d, lambda x: x)

    def test_regions_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            regions(np.zeros((5, 5)))

    def test_trim_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            trim(np.zeros((5, 5)))

    def test_crop_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            crop(np.zeros((5, 5)), _raster_2d, zones_ids=[1])

    # Classify — all 10 functions
    @pytest.mark.parametrize('func,args', [
        (binary, ([0, 1],)),
        (reclassify, ([0, 1], [10, 20],)),
        (quantile, (3,)),
        (natural_breaks, ()),
        (equal_interval, ()),
        (std_mean, ()),
        (head_tail_breaks, ()),
        (percentiles, ()),
        (maximum_breaks, ()),
        (box_plot, ()),
    ])
    def test_classify_rejects_ndarray(self, func, args):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            func(np.zeros((5, 5)), *args)

    # Multispectral — representative sample
    def test_ndvi_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            ndvi(np.zeros((5, 5)), _raster_2d)

    def test_arvi_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            arvi(np.zeros((5, 5)), _raster_2d, _raster_2d)

    def test_evi_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            evi(np.zeros((5, 5)), _raster_2d, _raster_2d)

    def test_gci_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            gci(np.zeros((5, 5)), _raster_2d)

    def test_savi_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            savi(np.zeros((5, 5)), _raster_2d)

    def test_ebbi_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            ebbi(np.zeros((5, 5)), _raster_2d, _raster_2d)

    def test_true_color_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            true_color(np.zeros((5, 5)), _raster_2d, _raster_2d)

    # Other modules
    def test_viewshed_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            viewshed(np.zeros((5, 5)), 0, 0)

    def test_perlin_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            perlin(np.zeros((5, 5)))

    def test_generate_terrain_rejects_ndarray(self):
        with pytest.raises(TypeError, match="xarray.DataArray"):
            generate_terrain(np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# Ndim checks — 2D-only functions reject wrong dimensionality
# ---------------------------------------------------------------------------
class TestNdimChecks:
    """Functions expecting 2D reject wrong dimensionality."""

    _agg_3d = xr.DataArray(
        np.zeros((3, 5, 5), dtype=np.float64), dims=['z', 'y', 'x']
    )

    @pytest.mark.parametrize('func,args', [
        (slope, ()),
        (aspect, ()),
        (hillshade, ()),
        (curvature, ()),
    ])
    def test_surface_rejects_3d(self, func, args):
        with pytest.raises(ValueError, match="2D"):
            func(self._agg_3d, *args)

    def test_mean_rejects_3d(self):
        with pytest.raises(ValueError, match="2D"):
            mean(self._agg_3d)

    def test_focal_apply_rejects_3d(self):
        with pytest.raises(ValueError, match="2D"):
            focal_apply(self._agg_3d, _kernel_3x3)

    @pytest.mark.parametrize('func', [proximity, allocation, direction])
    def test_proximity_rejects_3d(self, func):
        with pytest.raises(ValueError, match="2D"):
            func(self._agg_3d)

    def test_viewshed_rejects_3d(self):
        with pytest.raises(ValueError, match="2D"):
            viewshed(self._agg_3d, 0, 0)

    def test_regions_rejects_3d(self):
        with pytest.raises(ValueError, match="2D"):
            regions(self._agg_3d)

    def test_trim_rejects_3d(self):
        with pytest.raises(ValueError, match="2D"):
            trim(self._agg_3d)

    def test_perlin_rejects_3d(self):
        with pytest.raises(ValueError, match="2D"):
            perlin(self._agg_3d)


# ---------------------------------------------------------------------------
# Dtype checks — numeric functions reject string dtype
# ---------------------------------------------------------------------------
class TestDtypeChecks:
    """Functions requiring numeric input reject string dtype."""

    _str_agg = xr.DataArray(
        np.array([['a', 'b'], ['c', 'd']]), dims=['y', 'x']
    )

    def test_slope_rejects_string(self):
        with pytest.raises(ValueError, match="numeric dtype"):
            slope(self._str_agg)

    def test_aspect_rejects_string(self):
        with pytest.raises(ValueError, match="numeric dtype"):
            aspect(self._str_agg)

    def test_hillshade_rejects_string(self):
        with pytest.raises(ValueError, match="numeric dtype"):
            hillshade(self._str_agg)

    def test_curvature_rejects_string(self):
        with pytest.raises(ValueError, match="numeric dtype"):
            curvature(self._str_agg)

    def test_mean_rejects_string(self):
        with pytest.raises(ValueError, match="numeric dtype"):
            mean(self._str_agg)

    def test_proximity_rejects_string(self):
        with pytest.raises(ValueError, match="numeric dtype"):
            proximity(self._str_agg)

    def test_ndvi_rejects_string(self):
        with pytest.raises(ValueError, match="numeric dtype"):
            ndvi(self._str_agg, _raster_2d)

    def test_classify_rejects_string(self):
        str_1d = xr.DataArray(np.array(['a', 'b', 'c']), dims=['x'])
        with pytest.raises(ValueError, match="numeric dtype"):
            quantile(str_1d, k=3)


# ---------------------------------------------------------------------------
# Scalar parameter validation
# ---------------------------------------------------------------------------
class TestScalarChecks:
    """Scalar parameter validation."""

    def test_hillshade_azimuth_range(self):
        with pytest.raises(ValueError, match="azimuth"):
            hillshade(_raster_2d, azimuth=400)

    def test_hillshade_angle_altitude_range(self):
        with pytest.raises(ValueError, match="angle_altitude"):
            hillshade(_raster_2d, angle_altitude=100)

    def test_mean_passes_min(self):
        with pytest.raises(ValueError, match="passes"):
            mean(_raster_2d, passes=0)

    def test_mean_passes_type(self):
        with pytest.raises(TypeError, match="passes"):
            mean(_raster_2d, passes=1.5)

    def test_quantile_k_min(self):
        with pytest.raises(ValueError, match="k"):
            quantile(_raster_2d, k=0)

    def test_natural_breaks_k_min(self):
        with pytest.raises(ValueError, match="k"):
            natural_breaks(_raster_2d, k=0)

    def test_equal_interval_k_min(self):
        with pytest.raises(ValueError, match="k"):
            equal_interval(_raster_2d, k=0)


    def test_maximum_breaks_k_min(self):
        with pytest.raises(ValueError, match="k"):
            maximum_breaks(_raster_2d, k=0)

    def test_bump_width_min(self):
        with pytest.raises(ValueError, match="width"):
            bump(width=0, height=10)

    def test_bump_height_min(self):
        with pytest.raises(ValueError, match="height"):
            bump(width=10, height=0)

    def test_bump_width_type(self):
        with pytest.raises(TypeError, match="width"):
            bump(width=10.5, height=10)

    def test_bump_height_type(self):
        with pytest.raises(TypeError, match="height"):
            bump(width=10, height=10.5)


# ---------------------------------------------------------------------------
# Integer-only checks (zonal.apply zones)
# ---------------------------------------------------------------------------
class TestIntegerOnlyChecks:
    """Functions that require integer zones reject float dtype."""

    def test_zonal_apply_rejects_float_zones(self):
        zones_float = xr.DataArray(
            np.array([[1.0, 2.0]], dtype=np.float64), dims=['y', 'x']
        )
        values = xr.DataArray(
            np.array([[1.0, 2.0]]), dims=['y', 'x']
        )
        with pytest.raises(ValueError, match="integer dtype"):
            zonal_apply(zones_float, values, lambda x: x)
