import numpy as np
import xarray as xr

from xrspatial import polygonize
from xrspatial.polygonize import (
    _calculate_regions, _scan,
)

try:
    import cupy
    from xrspatial.polygonize import _calculate_regions_cupy
    _has_cupy = True
except ImportError:
    _has_cupy = False


class Polygonize:
    params = (
        [100, 300, 1000],
        ["numpy", "geopandas", "spatialpandas", "rasterio",
            "rasterio-geopandas"],
    )
    param_names = ("nx", "ret")

    def setup(self, nx, ret):
        # Raster and mask with many small regions.
        ny = nx // 2
        rng = np.random.default_rng(9461713)
        raster = rng.integers(low=0, high=4, size=(ny, nx), dtype=np.int32)
        mask = rng.uniform(0, 1, size=(ny, nx)) < 0.9
        self.raster = xr.DataArray(raster)
        self.mask = xr.DataArray(mask)

    def time_polygonize(self, nx, ret):
        if ret.startswith("rasterio"):
            import rasterio.features
            if ret == "rasterio":
                # Cast to list to ensure generator is run.
                list(rasterio.features.shapes(
                    self.raster.data, self.mask.data))
            else:
                import geopandas as gpd
                from shapely.geometry import shape
                values = []
                shapes = []
                for shape_dict, value in rasterio.features.shapes(
                        self.raster.data, self.mask.data):
                    shapes.append(shape(shape_dict))
                    values.append(value)
                gpd.GeoDataFrame({"DN": values, "geometry": shapes})
        else:
            polygonize(self.raster, mask=self.mask, return_type=ret)


class PolygonizeCuPy:
    params = ([100, 300, 1000],)
    param_names = ("nx",)

    def setup(self, nx):
        if not _has_cupy:
            raise NotImplementedError("CuPy not available")
        ny = nx // 2
        rng = np.random.default_rng(9461713)
        raster = rng.integers(low=0, high=4, size=(ny, nx), dtype=np.int32)
        mask = rng.uniform(0, 1, size=(ny, nx)) < 0.9
        self.raster = xr.DataArray(cupy.asarray(raster))
        self.mask = xr.DataArray(cupy.asarray(mask))

    def time_polygonize(self, nx):
        polygonize(self.raster, mask=self.mask)


class PolygonizeCCL:
    """Benchmark connected-component labeling phase."""
    params = ([100, 300, 1000], ["numpy", "cupy"])
    param_names = ("nx", "backend")

    def setup(self, nx, backend):
        if backend == "cupy" and not _has_cupy:
            raise NotImplementedError("CuPy not available")
        ny = nx // 2
        rng = np.random.default_rng(9461713)
        raster = rng.integers(low=0, high=4, size=(ny, nx), dtype=np.int32)
        mask = (rng.uniform(0, 1, size=(ny, nx)) < 0.9)
        if backend == "cupy":
            self.data = cupy.asarray(raster)
            self.mask = cupy.asarray(mask)
        else:
            self.data = raster
            self.mask = mask
        self.ny, self.nx = ny, nx
        self.backend = backend

    def time_calculate_regions(self, nx, backend):
        if backend == "cupy":
            _calculate_regions_cupy(self.data, self.mask, False)
            cupy.cuda.Device().synchronize()
        else:
            _calculate_regions(
                self.data.ravel(), self.mask.ravel(), False, self.nx, self.ny)


class PolygonizeTracing:
    """Benchmark boundary tracing phase (always CPU)."""
    params = ([100, 300, 1000],)
    param_names = ("nx",)

    def setup(self, nx):
        ny = nx // 2
        rng = np.random.default_rng(9461713)
        raster = rng.integers(low=0, high=4, size=(ny, nx), dtype=np.int32)
        mask = (rng.uniform(0, 1, size=(ny, nx)) < 0.9)
        values_flat = raster.ravel()
        mask_flat = mask.ravel()
        self.regions = _calculate_regions(
            values_flat, mask_flat, False, nx, ny)
        self.values = values_flat
        self.mask = mask_flat
        self.nx = nx
        self.ny = ny

    def time_scan(self, nx):
        _scan(self.regions, self.values, self.mask, False, None,
              self.nx, self.ny)
