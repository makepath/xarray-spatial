import numpy as np
import xarray as xr

from xrspatial.experimental import polygonize


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
