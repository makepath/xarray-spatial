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

    def viewshed(self, x, y, **kwargs):
        from .viewshed import viewshed
        return viewshed(self._obj, x, y, **kwargs)

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
