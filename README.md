:earth_africa: xarray-spatial: Raster-Based Spatial Analysis in Python
-------

[![Build Status](https://travis-ci.org/makepath/xarray-spatial.svg?branch=master)](https://travis-ci.org/makepath/xarray-spatial)
[![Build status](https://ci.appveyor.com/api/projects/status/4aco2mfbk14vds77?svg=true)](https://ci.appveyor.com/project/brendancol/xarray-spatial)
[![PyPI version](https://badge.fury.io/py/xarray-spatial.svg)](https://badge.fury.io/py/xarray-spatial)
[![Downloads](https://img.shields.io/pypi/dm/xarray-spatial.svg)]()
[![License](https://img.shields.io/pypi/l/xarray-spatial.svg)]()


-------
![title](composite_map.gif)
-------
:round_pushpin: Fast, Accurate Python library for Raster Operations

:zap: Extensible with [Numba](http://numba.pydata.org/)

:fast_forward: Scalable with [Dask](http://dask.pydata.org)

:confetti_ball: Free of GDAL / GEOS Dependencies

:earth_africa: General-Purpose Spatial Processing, Geared Towards GIS Professionals

-------

Xarray-Spatial implements common raster analysis functions using Numba and provides an easy-to-install, easy-to-extend codebase for raster analysis.

#### Installation
```bash
# via pip
pip install xarray-spatial

# via conda
conda install -c conda-forge xarray-spatial
```

| | | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|<a href="/examples/"><img width="1604" src="img/0-0.png"></a>                           | <a href="/examples/user_guide/2_Proximity.ipynb"><img width="1604" src="img/0-1.png"></a>     |<a href="/examples/user_guide/2_Proximity.ipynb"><img width="1604" src="img/0-2.png"></a>     |<a href="/examples/user_guide/2_Proximity.ipynb"><img width="1604" src="img/0-3.png"></a>     |<a href="/examples/pharmacy-deserts.ipynb"><img width="1604" src="img/0-4.png"></a>|
|<a href="/examples/user_guide/1_Surface.ipynb"><img width="1604" src="img/1-0.png"></a> | <a href="/examples/user_guide/1_Surface.ipynb"><img width="1604" src="img/1-1.png"></a>       |<a href="/examples/user_guide/1_Surface.ipynb"><img width="1604" src="img/1-2.png"></a>       |<a href="/examples/user_guide/1_Surface.ipynb"><img width="1604" src="img/1-3.png"></a>       |<a href="/examples/pharmacy-deserts.ipynb"><img width="1604" src="img/1-4.png"></a>|
|<a href="/examples/user_guide/1_Surface.ipynb"><img width="1604" src="img/2-0.png"></a> | <a href="/examples/user_guide/8_Remote_Sensing.ipynb"><img width="1604" src="img/2-1.png"></a>|<a href="/examples/user_guide/8_Remote_Sensing.ipynb"><img width="1604" src="img/2-2.png"></a>|<a href="/examples/user_guide/5_Classification.ipynb"><img width="1604" src="img/2-3.png"></a>|<a href="/examples/pharmacy-deserts.ipynb"><img width="1604" src="img/2-4.png"></a>|
|<a href="/examples/"><img width="1604" src="img/3-0.png"></a>                           | <a href="/examples/"><img width="1604" src="img/3-1.png"></a>                                 |<a href="/examples/user_guide/5_Classification.ipynb"><img width="1604" src="img/3-2.png"></a>|<a href="/examples/pharmacy-deserts.ipynb"><img width="1604" src="img/3-3.png"></a>|<a href="/examples/"><img width="1604" src="img/3-4.png"></a>|
|<a href="/examples/Path-finding_City-of-Austin-Road-Network.ipynb"><img width="1604" src="img/4-0.png"></a> | | | | |


`xarray-spatial` grew out of the [Datashader project](https://datashader.org/), which provides fast rasterization of vector data (points, lines, polygons, meshes, and rasters) for use with xarray-spatial.

`xarray-spatial` does not depend on GDAL / GEOS, which makes it fully extensible in Python but does limit the breadth of operations that can be covered.  xarray-spatial is meant to include the core raster-analysis functions needed for GIS developers / analysts, implemented independently of the non-Python geo stack.


Our documentation is still under constructions, but [docs can be found here](https://makepath.github.io/xarray-spatial/).


#### Raster-huh?

Rasters are regularly gridded datasets like GeoTIFFs, JPGs, and PNGs.

In the GIS world, rasters are used for representing continuous phenomena (e.g. elevation, rainfall, distance), either directly as numerical values, or as RGB images created for humans to view. Rasters typically have two spatial dimensions, but may have any number of other dimensions (time, type of measurement, etc.)

#### Supported Spatial Functions

| Name / Module | xr.DataArary Support | xr.Dataset Support | GPU Support (CUDA)| Dask Support|
|:----------:|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Aspect / aspect.py](xrspatial/aspect.py) | YES | NO | YES ||
| [Bump Mapping / bump.py](xrspatial/bump.py) | YES | NO | NO ||
| [Equal Interval / classify.py](xrspatial/classify.py) | YES | NO | NO ||
| [Natural Breaks / classify.py](xrspatial/classify.py) | YES | NO | NO ||
| [Reclassify / classify.py](xrspatial/classify.py) | YES | NO | NO ||
| [Quantile / classify.py](xrspatial/classify.py) | YES | NO | NO ||
| [Curvature / curvature.py](xrspatial/curvature.py) | YES | NO | NO ||
| [Apply / focal.py](xrspatial/focal.py) | YES | NO | NO ||
| [Hotspots / focal.py](xrspatial/focal.py) | YES | NO | NO ||
| [Mean / focal.py](xrspatial/focal.py) | YES | NO  | NO ||
| [Focal Statistics / focal.py](xrspatial/focal.py) | YES | NO  | YES ||
| [Hillshade / hillshade.py](xrspatial/hillshade.py) | YES | NO  | NO ||
| [Atmospherically Resistant Vegetation Index (ARVI) / multispectral.py](xrspatial/multispectral.py) | YES | NO  | YES ||
| [Enhanced Built-Up and Bareness Index (EBBI) / multispectral.py](xrspatial/multispectral.py) | YES | NO  | YES ||
| [Enhanced Vegetation Index (EVI) / multispectral.py](xrspatial/multispectral.py) | YES | NO  | YES ||
| [Green Chlorophyll Index (GCI) / multispectral.py](xrspatial/multispectral.py) | YES | NO  | YES ||
| [Normalized Burn Ratio (NBR) / multispectral.py](xrspatial/multispectral.py) | YES | NO  | YES ||
| [Normalized Burn Ratio 2 (NBR2) / multispectral.py](xrspatial/multispectral.py) | YES | NO  | YES ||
| [Normalized Difference Moisture Index (NDMI) / multispectral.py](xrspatial/multispectral.py) | YES | NO  | YES ||
| [Normalized Difference Vegetation Index (NDVI) / multispectral.py](xrspatial/multispectral.py) | YES | NO  | YES ||
| [Soil Adjusted Vegetation Index (SAVI) / multispectral.py](xrspatial/multispectral.py) | YES | NO  | YES ||
| [Structure Insensitive Pigment Index (SIPI) / multispectral.py](xrspatial/multispectral.py) | YES | NO  | YES ||
| [Pathfinding / pathfinding.py](xrspatial/pathfinding.py) | YES | NO  | NO ||
| [Perlin Noise / perlin.py](xrspatial/perlin.py) | YES | NO  | NO ||
| [Allocation / proximity.py](xrspatial/proximity.py) | YES | NO  | NO ||
| [Direction / proximity.py](xrspatial/proximity.py) | YES | NO  | NO ||
| [Proximity / proximity.py](xrspatial/proximity.py) | YES | NO  | NO ||
| [Slope / slope.py](xrspatial/slope.py) | YES  | NO  | YES ||
| [Terrain Generation / terrain.py](xrspatial/terrain.py) | YES | NO  | NO ||
| [Viewshed / viewshed.py](xrspatial/viewshed.py) | YES | NO  | NO ||
| [Apply / zonal.py](xrspatial/zonal.py) | YES | NO  | NO ||
| [Crop / zonal.py](xrspatial/zonal.py) | YES | NO  | NO ||
| [Regions / zonal.py](xrspatial/zonal.py) | YES | NO  | NO ||
| [Trim / zonal.py](xrspatial/zonal.py) | YES | NO  | NO ||
| [Zonal Statistics / zonal.py](xrspatial/zonal.py) | YES | NO  | NO ||
| [Zonal Cross Tabulate / zonal.py](xrspatial/zonal.py) | YES | NO  | NO ||


#### Usage

##### Basic Pattern
```python
import xarray as xr
from xrspatial import hillshade

my_dataarray = xr.DataArray(...)
hillshaded_dataarray = hillshade(my_dataarray)
```

Check out the user guide [here](/examples/user_guide/).

------
Check out [Xarray-Spatial on YouTube](https://www.youtube.com/watch?v=z4xrkglmg80)
------


![title](composite_map.png)
![title](makepath-supply-chain-international-shipping.png)

#### Dependencies

`xarray-spatial` currently depends on Datashader, but will soon be updated to depend only on `xarray` and `numba`, while still being able to make use of Datashader output when available. 

![title](dependencies.svg)

#### Notes on GDAL

Within the Python ecosystem, many geospatial libraries interface with the GDAL C++ library for raster and vector input, output, and analysis (e.g. rasterio, rasterstats, geopandas). GDAL is robust, performant, and has decades of great work behind it. For years, off-loading expensive computations to the C/C++ level in this way has been a key performance strategy for Python libraries (obviously...Python itself is implemented in C!).

However, wrapping GDAL has a few drawbacks for Python developers and data scientists:
- GDAL can be a pain to build / install.
- GDAL is hard for Python developers/analysts to extend, because it requires understanding multiple languages.
- GDAL's data structures are defined at the C/C++ level, which constrains how they can be accessed from Python.

With the introduction of projects like Numba, Python gained new ways to provide high-performance code directly in Python, without depending on or being constrained by separate C/C++ extensions. `xarray-spatial` implements algorithms using Numba and Dask, making all of its source code available as pure Python without any "black box" barriers that obscure what is going on and prevent full optimization. Projects can make use of the functionality provided by `xarray-spatial` where available, while still using GDAL where required for other tasks.

#### Contributors

- @brendancol
- @thuydotm
- @jbednar
- @pablomakepath
- @kristinepetrosyan
- @sjsrey
- @giancastro
- @ocefpaf
- @rsignell-usgs
- @marcozimmermannpm
- @jthetzel
- @chase-dwelle
