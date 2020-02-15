![title](https://via.placeholder.com/150x150.png) xarray-spatial: Raster-based Spatial Analysis in Python
-------

Rasters are regularly gridded datasets like GeoTIFFs, JPGs, and PNGs.

In the GIS world, folks use rasters for representing continuous phenomena (e.g. elevation, rainfall, distance).   


`xarray-spatial` implements common raster analysis functions using numba and provides an easy-to-install, easy-to-extend codebase for raster analysis.

`xarray-spatial` does not depend on GDAL / GEOS, and because of this has limited breadth.  The plan is to implement core raster analysis functions in terminology common to GIS developers / analysts.

`xarray-spatial` is a generalization which fell out of the datashader projects.


### Installation
```bash
# via conda
conda install -c makepath xarray-spatial

# via pip
pip install xarray-spatial
```

### Usage
```python
from xrspatial import hillshade

hillshade_xarray_dataarray = hillshade(my_xarray_dataarray)
```

### Dependencies
![title](https://via.placeholder.com/300x400.png)
<Add dependency graph image here>

### Notes on GDAL

Within the Python ecosystem, many geospatial libraries interface with GDAL (C++) for raster input / output and analysis (e.g. rasterio, raster-stats). People wrap GDAL because its robust, performant and has decades of great work behind it. Off-loading expensive computations to the C/C++ level has been a key performance strategy for Python libraries (obviously...Python itself is implemented in C).

Wrapping GDAL has a few drawbacks for Python developers and data scientists:
- GDAL can be a pain to build / install.
- GDAL is hard for Python developers/analysts to extend.

With the introduction of projects like numba, Python gained new ways to improve performance without writing C/C++ extensions.
