"""xarray accessors for xarray-spatial.

Registers ``.xrs`` accessors on :class:`xarray.DataArray` and
:class:`xarray.Dataset` so that spatial operations can be called
directly via tab-completion::

    import xrspatial  # registers accessors

    raster.xrs.slope()
    raster.xrs.hillshade(azimuth=315)
    nir.xrs.ndvi(red)
"""

import xarray as xr


@xr.register_dataarray_accessor("xrs")
class XrsSpatialDataArrayAccessor:
    """DataArray accessor exposing xarray-spatial operations."""

    def __init__(self, obj):
        self._obj = obj

    # ---- Surface ----

    def slope(self, **kwargs):
        from .slope import slope
        return slope(self._obj, **kwargs)

    def aspect(self, **kwargs):
        from .aspect import aspect
        return aspect(self._obj, **kwargs)

    def hillshade(self, **kwargs):
        from .hillshade import hillshade
        return hillshade(self._obj, **kwargs)

    def curvature(self, **kwargs):
        from .curvature import curvature
        return curvature(self._obj, **kwargs)

    # ---- Terrain Metrics ----

    def tri(self, **kwargs):
        from .terrain_metrics import tri
        return tri(self._obj, **kwargs)

    def tpi(self, **kwargs):
        from .terrain_metrics import tpi
        return tpi(self._obj, **kwargs)

    def roughness(self, **kwargs):
        from .terrain_metrics import roughness
        return roughness(self._obj, **kwargs)

    # ---- Hydrology ----

    def flow_direction(self, **kwargs):
        from .flow_direction import flow_direction
        return flow_direction(self._obj, **kwargs)

    def flow_direction_dinf(self, **kwargs):
        from .flow_direction_dinf import flow_direction_dinf
        return flow_direction_dinf(self._obj, **kwargs)

    def flow_accumulation(self, **kwargs):
        from .flow_accumulation import flow_accumulation
        return flow_accumulation(self._obj, **kwargs)

    def watershed(self, pour_points, **kwargs):
        from .watershed import watershed
        return watershed(self._obj, pour_points, **kwargs)

    def basin(self, **kwargs):
        from .basin import basin
        return basin(self._obj, **kwargs)

    def basins(self, **kwargs):
        from .watershed import basins
        return basins(self._obj, **kwargs)

    def sink(self, **kwargs):
        from .sink import sink
        return sink(self._obj, **kwargs)

    def fill(self, **kwargs):
        from .fill import fill
        return fill(self._obj, **kwargs)

    def stream_order(self, flow_accum, **kwargs):
        from .stream_order import stream_order
        return stream_order(self._obj, flow_accum, **kwargs)

    def stream_link(self, flow_accum, **kwargs):
        from .stream_link import stream_link
        return stream_link(self._obj, flow_accum, **kwargs)

    def snap_pour_point(self, pour_points, **kwargs):
        from .snap_pour_point import snap_pour_point
        return snap_pour_point(self._obj, pour_points, **kwargs)

    def flow_path(self, start_points, **kwargs):
        from .flow_path import flow_path
        return flow_path(self._obj, start_points, **kwargs)

    def flow_length(self, **kwargs):
        from .flow_length import flow_length
        return flow_length(self._obj, **kwargs)

    def twi(self, slope_agg, **kwargs):
        from .twi import twi
        return twi(self._obj, slope_agg, **kwargs)

    def hand(self, flow_accum, elevation, **kwargs):
        from .hand import hand
        return hand(self._obj, flow_accum, elevation, **kwargs)

    # ---- Flood ----

    def flood_depth(self, water_level, **kwargs):
        from .flood import flood_depth
        return flood_depth(self._obj, water_level, **kwargs)

    def inundation(self, water_level, **kwargs):
        from .flood import inundation
        return inundation(self._obj, water_level, **kwargs)

    def curve_number_runoff(self, curve_number, **kwargs):
        from .flood import curve_number_runoff
        return curve_number_runoff(self._obj, curve_number, **kwargs)

    def travel_time(self, slope_agg, mannings_n, **kwargs):
        from .flood import travel_time
        return travel_time(self._obj, slope_agg, mannings_n, **kwargs)

    def viewshed(self, x, y, **kwargs):
        from .viewshed import viewshed
        return viewshed(self._obj, x, y, **kwargs)

    def min_observable_height(self, x, y, **kwargs):
        from .experimental.min_observable_height import min_observable_height
        return min_observable_height(self._obj, x, y, **kwargs)

    # ---- Classification ----

    def natural_breaks(self, **kwargs):
        from .classify import natural_breaks
        return natural_breaks(self._obj, **kwargs)

    def equal_interval(self, **kwargs):
        from .classify import equal_interval
        return equal_interval(self._obj, **kwargs)

    def quantile(self, **kwargs):
        from .classify import quantile
        return quantile(self._obj, **kwargs)

    def reclassify(self, bins, new_values, **kwargs):
        from .classify import reclassify
        return reclassify(self._obj, bins, new_values, **kwargs)

    def binary(self, values, **kwargs):
        from .classify import binary
        return binary(self._obj, values, **kwargs)

    def percentiles(self, **kwargs):
        from .classify import percentiles
        return percentiles(self._obj, **kwargs)

    # ---- Focal ----

    def focal_mean(self, **kwargs):
        from .focal import mean
        return mean(self._obj, **kwargs)

    # ---- Proximity / Distance ----

    def proximity(self, **kwargs):
        from .proximity import proximity
        return proximity(self._obj, **kwargs)

    def allocation(self, **kwargs):
        from .proximity import allocation
        return allocation(self._obj, **kwargs)

    def direction(self, **kwargs):
        from .proximity import direction
        return direction(self._obj, **kwargs)

    def cost_distance(self, friction, **kwargs):
        from .cost_distance import cost_distance
        return cost_distance(self._obj, friction, **kwargs)

    def surface_distance(self, elevation, **kwargs):
        from .surface_distance import surface_distance
        return surface_distance(self._obj, elevation, **kwargs)

    def surface_allocation(self, elevation, **kwargs):
        from .surface_distance import surface_allocation
        return surface_allocation(self._obj, elevation, **kwargs)

    def surface_direction(self, elevation, **kwargs):
        from .surface_distance import surface_direction
        return surface_direction(self._obj, elevation, **kwargs)

    # ---- Pathfinding ----

    def a_star_search(self, start, goal, **kwargs):
        from .pathfinding import a_star_search
        return a_star_search(self._obj, start, goal, **kwargs)

    # ---- Zonal ----

    def zonal_stats(self, zones, **kwargs):
        from .zonal import stats
        return stats(zones, self._obj, **kwargs)

    def zonal_apply(self, zones, func, **kwargs):
        from .zonal import apply
        return apply(zones, self._obj, func, **kwargs)

    def zonal_crosstab(self, zones, **kwargs):
        from .zonal import crosstab
        return crosstab(zones, self._obj, **kwargs)

    def crop(self, zones, zones_ids, **kwargs):
        from .zonal import crop
        return crop(zones, self._obj, zones_ids, **kwargs)

    def trim(self, **kwargs):
        from .zonal import trim
        return trim(self._obj, **kwargs)

    def regions(self, **kwargs):
        from .zonal import regions
        return regions(self._obj, **kwargs)

    # ---- Terrain generation ----

    def generate_terrain(self, **kwargs):
        from .terrain import generate_terrain
        return generate_terrain(self._obj, **kwargs)

    def perlin(self, **kwargs):
        from .perlin import perlin
        return perlin(self._obj, **kwargs)

    # ---- Mahalanobis ----

    def mahalanobis(self, other_bands, **kwargs):
        from .mahalanobis import mahalanobis
        return mahalanobis([self._obj] + list(other_bands), **kwargs)

    # ---- Interpolation ----

    def idw(self, x, y, z, **kwargs):
        from .interpolate import idw
        return idw(x, y, z, self._obj, **kwargs)

    def kriging(self, x, y, z, **kwargs):
        from .interpolate import kriging
        return kriging(x, y, z, self._obj, **kwargs)

    def spline(self, x, y, z, **kwargs):
        from .interpolate import spline
        return spline(x, y, z, self._obj, **kwargs)

    # ---- Raster to vector ----

    def polygonize(self, **kwargs):
        from .polygonize import polygonize
        return polygonize(self._obj, **kwargs)

    # ---- Fire ----

    def dnbr(self, post_nbr_agg, **kwargs):
        from .fire import dnbr
        return dnbr(self._obj, post_nbr_agg, **kwargs)

    def rdnbr(self, pre_nbr_agg, **kwargs):
        from .fire import rdnbr
        return rdnbr(self._obj, pre_nbr_agg, **kwargs)

    def burn_severity_class(self, **kwargs):
        from .fire import burn_severity_class
        return burn_severity_class(self._obj, **kwargs)

    def fireline_intensity(self, spread_rate_agg, **kwargs):
        from .fire import fireline_intensity
        return fireline_intensity(self._obj, spread_rate_agg, **kwargs)

    def flame_length(self, **kwargs):
        from .fire import flame_length
        return flame_length(self._obj, **kwargs)

    def rate_of_spread(self, wind_speed_agg, fuel_moisture_agg, **kwargs):
        from .fire import rate_of_spread
        return rate_of_spread(self._obj, wind_speed_agg, fuel_moisture_agg, **kwargs)

    def kbdi(self, max_temp_agg, precip_agg, annual_precip, **kwargs):
        from .fire import kbdi
        return kbdi(self._obj, max_temp_agg, precip_agg, annual_precip, **kwargs)

    # ---- Multispectral (self = NIR band) ----

    def ndvi(self, red_agg, **kwargs):
        from .multispectral import ndvi
        return ndvi(self._obj, red_agg, **kwargs)

    def evi(self, red_agg, blue_agg, **kwargs):
        from .multispectral import evi
        return evi(self._obj, red_agg, blue_agg, **kwargs)

    def arvi(self, red_agg, blue_agg, **kwargs):
        from .multispectral import arvi
        return arvi(self._obj, red_agg, blue_agg, **kwargs)

    def savi(self, red_agg, **kwargs):
        from .multispectral import savi
        return savi(self._obj, red_agg, **kwargs)

    def nbr(self, swir2_agg, **kwargs):
        from .multispectral import nbr
        return nbr(self._obj, swir2_agg, **kwargs)

    def sipi(self, red_agg, blue_agg, **kwargs):
        from .multispectral import sipi
        return sipi(self._obj, red_agg, blue_agg, **kwargs)


@xr.register_dataset_accessor("xrs")
class XrsSpatialDatasetAccessor:
    """Dataset accessor exposing xarray-spatial operations.

    Single-input functions apply the operation to each data variable
    (via the existing ``@supports_dataset`` decorator).  Multi-input
    (multispectral) functions accept string kwargs that map band
    aliases to Dataset variable names.
    """

    def __init__(self, obj):
        self._obj = obj

    # ---- Surface ----

    def slope(self, **kwargs):
        from .slope import slope
        return slope(self._obj, **kwargs)

    def aspect(self, **kwargs):
        from .aspect import aspect
        return aspect(self._obj, **kwargs)

    def hillshade(self, **kwargs):
        from .hillshade import hillshade
        return hillshade(self._obj, **kwargs)

    def curvature(self, **kwargs):
        from .curvature import curvature
        return curvature(self._obj, **kwargs)

    # ---- Terrain Metrics ----

    def tri(self, **kwargs):
        from .terrain_metrics import tri
        return tri(self._obj, **kwargs)

    def tpi(self, **kwargs):
        from .terrain_metrics import tpi
        return tpi(self._obj, **kwargs)

    def roughness(self, **kwargs):
        from .terrain_metrics import roughness
        return roughness(self._obj, **kwargs)

    # ---- Hydrology ----

    def flow_direction(self, **kwargs):
        from .flow_direction import flow_direction
        return flow_direction(self._obj, **kwargs)

    def flow_direction_dinf(self, **kwargs):
        from .flow_direction_dinf import flow_direction_dinf
        return flow_direction_dinf(self._obj, **kwargs)

    def flow_accumulation(self, **kwargs):
        from .flow_accumulation import flow_accumulation
        return flow_accumulation(self._obj, **kwargs)

    def watershed(self, pour_points, **kwargs):
        from .watershed import watershed
        return watershed(self._obj, pour_points, **kwargs)

    def basin(self, **kwargs):
        from .basin import basin
        return basin(self._obj, **kwargs)

    def basins(self, **kwargs):
        from .watershed import basins
        return basins(self._obj, **kwargs)

    def sink(self, **kwargs):
        from .sink import sink
        return sink(self._obj, **kwargs)

    def fill(self, **kwargs):
        from .fill import fill
        return fill(self._obj, **kwargs)

    def stream_order(self, flow_accum, **kwargs):
        from .stream_order import stream_order
        return stream_order(self._obj, flow_accum, **kwargs)

    def stream_link(self, flow_accum, **kwargs):
        from .stream_link import stream_link
        return stream_link(self._obj, flow_accum, **kwargs)

    def snap_pour_point(self, pour_points, **kwargs):
        from .snap_pour_point import snap_pour_point
        return snap_pour_point(self._obj, pour_points, **kwargs)

    def flow_path(self, start_points, **kwargs):
        from .flow_path import flow_path
        return flow_path(self._obj, start_points, **kwargs)

    def flow_length(self, **kwargs):
        from .flow_length import flow_length
        return flow_length(self._obj, **kwargs)

    def twi(self, slope_agg, **kwargs):
        from .twi import twi
        return twi(self._obj, slope_agg, **kwargs)

    def hand(self, flow_accum, elevation, **kwargs):
        from .hand import hand
        return hand(self._obj, flow_accum, elevation, **kwargs)

    # ---- Flood ----

    def flood_depth(self, water_level, **kwargs):
        from .flood import flood_depth
        return flood_depth(self._obj, water_level, **kwargs)

    def inundation(self, water_level, **kwargs):
        from .flood import inundation
        return inundation(self._obj, water_level, **kwargs)

    def curve_number_runoff(self, curve_number, **kwargs):
        from .flood import curve_number_runoff
        return curve_number_runoff(self._obj, curve_number, **kwargs)

    def travel_time(self, slope_agg, mannings_n, **kwargs):
        from .flood import travel_time
        return travel_time(self._obj, slope_agg, mannings_n, **kwargs)

    # ---- Classification ----

    def natural_breaks(self, **kwargs):
        from .classify import natural_breaks
        return natural_breaks(self._obj, **kwargs)

    def equal_interval(self, **kwargs):
        from .classify import equal_interval
        return equal_interval(self._obj, **kwargs)

    def quantile(self, **kwargs):
        from .classify import quantile
        return quantile(self._obj, **kwargs)

    def reclassify(self, bins, new_values, **kwargs):
        from .classify import reclassify
        return reclassify(self._obj, bins, new_values, **kwargs)

    def binary(self, values, **kwargs):
        from .classify import binary
        return binary(self._obj, values, **kwargs)

    def percentiles(self, **kwargs):
        from .classify import percentiles
        return percentiles(self._obj, **kwargs)

    # ---- Focal ----

    def focal_mean(self, **kwargs):
        from .focal import mean
        return mean(self._obj, **kwargs)

    # ---- Proximity ----

    def proximity(self, **kwargs):
        from .proximity import proximity
        return proximity(self._obj, **kwargs)

    def allocation(self, **kwargs):
        from .proximity import allocation
        return allocation(self._obj, **kwargs)

    def direction(self, **kwargs):
        from .proximity import direction
        return direction(self._obj, **kwargs)

    def cost_distance(self, friction, **kwargs):
        from .cost_distance import cost_distance
        return cost_distance(self._obj, friction, **kwargs)

    def surface_distance(self, elevation, **kwargs):
        from .surface_distance import surface_distance
        return surface_distance(self._obj, elevation, **kwargs)

    def surface_allocation(self, elevation, **kwargs):
        from .surface_distance import surface_allocation
        return surface_allocation(self._obj, elevation, **kwargs)

    def surface_direction(self, elevation, **kwargs):
        from .surface_distance import surface_direction
        return surface_direction(self._obj, elevation, **kwargs)

    # ---- Fire ----

    def burn_severity_class(self, **kwargs):
        from .fire import burn_severity_class
        return burn_severity_class(self._obj, **kwargs)

    def flame_length(self, **kwargs):
        from .fire import flame_length
        return flame_length(self._obj, **kwargs)

    # ---- Multispectral (band mapping via kwargs) ----

    def ndvi(self, nir, red, **kwargs):
        from .multispectral import ndvi
        return ndvi(self._obj, nir=nir, red=red, **kwargs)

    def evi(self, nir, red, blue, **kwargs):
        from .multispectral import evi
        return evi(self._obj, nir=nir, red=red, blue=blue, **kwargs)

    def arvi(self, nir, red, blue, **kwargs):
        from .multispectral import arvi
        return arvi(self._obj, nir=nir, red=red, blue=blue, **kwargs)

    def savi(self, nir, red, **kwargs):
        from .multispectral import savi
        return savi(self._obj, nir=nir, red=red, **kwargs)

    def nbr(self, nir, swir2, **kwargs):
        from .multispectral import nbr
        return nbr(self._obj, nir=nir, swir2=swir2, **kwargs)

    def sipi(self, nir, red, blue, **kwargs):
        from .multispectral import sipi
        return sipi(self._obj, nir=nir, red=red, blue=blue, **kwargs)
