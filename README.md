<img src="img/Xarray-Spatial-logo.svg"/>

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <div>
    <img src="https://anaconda.org/conda-forge/xarray-spatial/badges/latest_release_date.svg" alt="Last Conda Release"/>
        <img src="https://badge.fury.io/py/xarray-spatial.svg" alt="pypi version" />
        <img src="https://anaconda.org/conda-forge/xarray-spatial/badges/version.svg" alt="conda-forge version" />
    </div>
    </a>
  </td>

  <td>Downloads</td>
  <td>
    <div>
    <img src="https://img.shields.io/pypi/dm/xarray-spatial?label=PyPI"
         alt="PyPI downloads per month" />
    </div>
  </td>
</tr>

<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/xarray-contrib/xarray-spatial/blob/master/LICENSE.txt">
    <img src="https://img.shields.io/pypi/l/xarray-spatial.svg"
         alt="MIT" />
    </a>
  </td>

  <td>People</td>
  <td>
    <img src="https://img.shields.io/github/contributors/xarray-contrib/xarray-spatial"
         alt="GitHub contributors" />
  </td>
</tr>

<tr>
  <td>Build Status</td>
  <td>
    <div>
    <a href="https://github.com/xarray-contrib/xarray-spatial/actions/workflows/test.yml">
    <img src="https://github.com/xarray-contrib/xarray-spatial/actions/workflows/test.yml/badge.svg"
         alt="Current github actions build status" />
    </a>
    </div>
    <div>
    <a href="https://github.com/xarray-contrib/xarray-spatial/actions/workflows/pypi-publish.yml">
    <img src="https://github.com/xarray-contrib/xarray-spatial/actions/workflows/pypi-publish.yml/badge.svg"
         alt="Current github actions build status" />
    </a>
    </div>
    <div>
      <a href='https://xarray-spatial.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/xarray-spatial/badge/?version=latest' alt='Documentation Status' />
      </a>
    </div>
  </td>
  <td>Coverage</td>
  <td>
    <div>
      <a href="https:https://codecov.io/gh/xarray-contrib/xarray-spatial">
      <img alt="Language grade: Python" src="https://codecov.io/gh/xarray-contrib/xarray-spatial/branch/master/graph/badge.svg"/>
      </a>
    </div>
  </td>

</tr>
</table>

-------
![title](img/composite_map.gif)
-------
:round_pushpin: Fast, Accurate Python library for Raster Operations

:zap: Extensible with [Numba](http://numba.pydata.org/)

:fast_forward: Scalable with [Dask](http://dask.pydata.org)

:confetti_ball: Free of GDAL / GEOS Dependencies

:earth_africa: General-Purpose Spatial Processing, Geared Towards GIS Professionals

-------

Xarray-Spatial implements common raster analysis functions using Numba and provides an easy-to-install, easy-to-extend codebase for raster analysis.

### Installation
```bash
# via pip
pip install xarray-spatial

# via conda
conda install -c conda-forge xarray-spatial
```

### Downloading our starter examples and data
Once you have xarray-spatial installed in your environment, you can use one of the following in your terminal (with the environment active) to download our examples and/or sample data into your local directory.

```xrspatial examples``` : Download the examples notebooks and the data used.

```xrspatial copy-examples``` : Download the examples notebooks but not the data. Note: you won't be able to run many of the examples.

```xrspatial fetch-data``` : Download just the data and not the notebooks.

In all the above, the command will download and store the files into your current directory inside a folder named 'xrspatial-examples'.


| | | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|<a href="/examples/"><img width="1604" src="img/0-0.png"></a>                           | <a href="/examples/user_guide/2_Proximity.ipynb"><img width="1604" src="img/0-1.png"></a>     |<a href="/examples/user_guide/2_Proximity.ipynb"><img width="1604" src="img/0-2.png"></a>     |<a href="/examples/user_guide/2_Proximity.ipynb"><img width="1604" src="img/0-3.png"></a>     |<a href="/examples/pharmacy-deserts.ipynb"><img width="1604" src="img/0-4.png"></a>|
|<a href="/examples/user_guide/1_Surface.ipynb"><img width="1604" src="img/1-0.png"></a> | <a href="/examples/user_guide/1_Surface.ipynb"><img width="1604" src="img/1-1.png"></a>       |<a href="/examples/user_guide/1_Surface.ipynb"><img width="1604" src="img/1-2.png"></a>       |<a href="/examples/user_guide/1_Surface.ipynb"><img width="1604" src="img/1-3.png"></a>       |<a href="/examples/pharmacy-deserts.ipynb"><img width="1604" src="img/1-4.png"></a>|
|<a href="/examples/user_guide/1_Surface.ipynb"><img width="1604" src="img/2-0.png"></a> | <a href="/examples/user_guide/8_Remote_Sensing.ipynb"><img width="1604" src="img/2-1.png"></a>|<a href="/examples/user_guide/8_Remote_Sensing.ipynb"><img width="1604" src="img/2-2.png"></a>|<a href="/examples/user_guide/5_Classification.ipynb"><img width="1604" src="img/2-3.png"></a>|<a href="/examples/pharmacy-deserts.ipynb"><img width="1604" src="img/2-4.png"></a>|
|<a href="/examples/"><img width="1604" src="img/3-0.png"></a>                           | <a href="/examples/"><img width="1604" src="img/3-1.png"></a>                                 |<a href="/examples/user_guide/5_Classification.ipynb"><img width="1604" src="img/3-2.png"></a>|<a href="/examples/pharmacy-deserts.ipynb"><img width="1604" src="img/3-3.png"></a>|<a href="/examples/"><img width="1604" src="img/3-4.png"></a>|
|<a href="/examples/Pathfinding_Austin_Road_Network.ipynb"><img width="1604" src="img/4-0.png"></a> |<a href="/examples/user_guide/1_Surface.ipynb#Hillshade"><img width="1604" src="img/4-1.png"></a> | <a href="/examples/user_guide/1_Surface.ipynb#Hillshade"><img width="1604" src="img/4-2.png"></a>| <a href="/examples/user_guide/1_Surface.ipynb#Slope"><img width="1604" src="img/4-3.png"></a>| <a href="/examples/pharmacy-deserts.ipynb#Create-a-%22Distance-to-Nearest-Pharmacy%22-Layer-&-Classify-into-5-Groups"><img width="1604" src="img/4-4.png"></a>|


`xarray-spatial` grew out of the [Datashader project](https://datashader.org/), which provides fast rasterization of vector data (points, lines, polygons, meshes, and rasters) for use with xarray-spatial.

`xarray-spatial` does not depend on GDAL / GEOS, which makes it fully extensible in Python but does limit the breadth of operations that can be covered.  xarray-spatial is meant to include the core raster-analysis functions needed for GIS developers / analysts, implemented independently of the non-Python geo stack.

Our documentation is still under construction, but [docs can be found here](https://xarray-spatial.readthedocs.io/en/latest/).

#### Raster-huh?

Rasters are regularly gridded datasets like GeoTIFFs, JPGs, and PNGs.

In the GIS world, rasters are used for representing continuous phenomena (e.g. elevation, rainfall, distance), either directly as numerical values, or as RGB images created for humans to view. Rasters typically have two spatial dimensions, but may have any number of other dimensions (time, type of measurement, etc.)

#### Supported Spatial Functions with Supported Inputs

-------

### **Classification**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Box Plot](xrspatial/classify.py) | Classifies values into bins based on box plot quartile boundaries | ✅️ |✅ | ✅ |✅ |
| [Equal Interval](xrspatial/classify.py) | Divides the value range into equal-width bins | ✅️ |✅ | ✅ |✅ |
| [Head/Tail Breaks](xrspatial/classify.py) | Classifies heavy-tailed distributions using recursive mean splitting | ✅️ |✅ | ✅ |✅ |
| [Maximum Breaks](xrspatial/classify.py) | Finds natural groupings by maximizing differences between sorted values | ✅️ |✅ | ✅ |✅ |
| [Natural Breaks](xrspatial/classify.py) | Optimizes class boundaries to minimize within-class variance (Jenks) | ✅️ |✅ | ✅ |✅ |
| [Percentiles](xrspatial/classify.py) | Assigns classes based on user-defined percentile breakpoints | ✅️ |✅ | ✅ |✅ |
| [Quantile](xrspatial/classify.py) | Distributes values into classes with equal observation counts | ✅️ |✅ | ✅ |✅ |
| [Reclassify](xrspatial/classify.py) | Remaps pixel values to new classes using a user-defined lookup | ✅️ |✅ | ✅ |✅ |
| [Std Mean](xrspatial/classify.py) | Classifies values by standard deviation intervals from the mean | ✅️ |✅ | ✅ |✅ |

-------

### **Focal**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Apply](xrspatial/focal.py) | Applies a custom function over a sliding neighborhood window | ✅️ | ✅️ |  |  |
| [Hotspots](xrspatial/focal.py) | Identifies statistically significant spatial clusters using Getis-Ord Gi* | ✅️ | ✅️ | ✅️ |  |
| [Mean](xrspatial/focal.py) | Computes the mean value within a sliding neighborhood window | ✅️ | ✅️ | ✅️ | |
| [Focal Statistics](xrspatial/focal.py) | Computes summary statistics over a sliding neighborhood window | ✅️ | ✅️ | ✅️ | |

-------

### **Multispectral**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Atmospherically Resistant Vegetation Index (ARVI)](xrspatial/multispectral.py) | Vegetation index resistant to atmospheric effects using blue band correction | ✅️ |✅️ | ✅️ |✅️ |
| [Enhanced Built-Up and Bareness Index (EBBI)](xrspatial/multispectral.py) | Highlights built-up areas and barren land from thermal and SWIR bands | ✅️ |✅️  | ✅️ |✅️ |
| [Enhanced Vegetation Index (EVI)](xrspatial/multispectral.py) | Enhanced vegetation index reducing soil and atmospheric noise | ✅️ |✅️ | ✅️ |✅️ |
| [Green Chlorophyll Index (GCI)](xrspatial/multispectral.py) | Estimates leaf chlorophyll content from green and NIR reflectance | ✅️ |✅️ | ✅️ |✅️ |
| [Normalized Burn Ratio (NBR)](xrspatial/multispectral.py) | Measures burn severity using NIR and SWIR band difference | ✅️ |✅️ | ✅️ |✅️ |
| [Normalized Burn Ratio 2 (NBR2)](xrspatial/multispectral.py) | Refines burn severity mapping using two SWIR bands | ✅️ |✅️ | ✅️ |✅️ |
| [Normalized Difference Moisture Index (NDMI)](xrspatial/multispectral.py) | Detects vegetation moisture stress from NIR and SWIR reflectance | ✅️ |✅️ | ✅️ |✅️ |
| [Normalized Difference Vegetation Index (NDVI)](xrspatial/multispectral.py) | Quantifies vegetation density from red and NIR band difference | ✅️ |✅️ | ✅️ |✅️ |
| [Soil Adjusted Vegetation Index (SAVI)](xrspatial/multispectral.py) | Vegetation index with soil brightness correction factor | ✅️ |✅️ | ✅️ |✅️ |
| [Structure Insensitive Pigment Index (SIPI)](xrspatial/multispectral.py) | Estimates carotenoid-to-chlorophyll ratio for plant stress detection | ✅️ |✅️ | ✅️ |✅️ |
| [True Color](xrspatial/multispectral.py) | Composites red, green, and blue bands into a natural color image | ✅️ | ️ | ✅️ | ️ |

-------


### **Pathfinding**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [A* Pathfinding](xrspatial/pathfinding.py) | Finds the least-cost path between two cells on a cost surface | ✅️ |  | | |

----------

### **Proximity**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Allocation](xrspatial/proximity.py) | Assigns each cell to the identity of the nearest source feature | ✅️ | ✅ | | |
| [Cost Distance](xrspatial/cost_distance.py) | Computes minimum accumulated cost to the nearest source through a friction surface | ✅️ | ✅ | | |
| [Direction](xrspatial/proximity.py) | Computes the direction from each cell to the nearest source feature | ✅️ | ✅ | | |
| [Proximity](xrspatial/proximity.py) | Computes the distance from each cell to the nearest source feature | ✅️ | ✅ | | |

--------

### **Raster to vector**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:-----|:------------|:------------------:|:-----------------:|:---------------------:|:---------------------:|
| [Polygonize](xrspatial/experimental/polygonize.py) | Converts contiguous regions of equal value into vector polygons | ✅️ | | | |

--------

### **Surface**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Aspect](xrspatial/aspect.py) | Computes downslope direction of each cell in degrees | ✅️ | ✅️ | ✅️ | ✅️ |
| [Curvature](xrspatial/curvature.py) | Measures rate of slope change (concavity/convexity) at each cell | ✅️ |✅️ |✅️ | ✅️  |
| [Hillshade](xrspatial/hillshade.py) | Simulates terrain illumination from a given sun angle and azimuth | ✅️ | ✅️  | | |
| [Slope](xrspatial/slope.py) | Computes terrain gradient steepness at each cell in degrees | ✅️  | ✅️  | ✅️ | ✅️  |
| [Terrain Generation](xrspatial/terrain.py) | Generates synthetic terrain elevation using fractal noise | ✅️ | ✅️ | ✅️ | |
| [Viewshed](xrspatial/viewshed.py) | Determines visible cells from a given observer point on terrain | ✅️ |  | | |
| [Perlin Noise](xrspatial/perlin.py) | Generates smooth continuous random noise for procedural textures | ✅️ | ✅️ | ✅️ | |
| [Bump Mapping](xrspatial/bump.py) | Adds randomized bump features to simulate natural terrain variation | ✅️ | | | |

-----------

### **Zonal**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Apply](xrspatial/zonal.py) | Applies a custom function to each zone in a classified raster | ✅️ | ✅️ | | |
| [Crop](xrspatial/zonal.py) | Extracts the bounding rectangle of a specific zone | ✅️ | | | |
| [Regions](xrspatial/zonal.py) | Identifies connected regions of non-zero cells | |  | | |
| [Trim](xrspatial/zonal.py) | Removes nodata border rows and columns from a raster | ✅️ |  | | |
| [Zonal Statistics](xrspatial/zonal.py) | Computes summary statistics for a value raster within each zone | ✅️ | ✅️| | |
| [Zonal Cross Tabulate](xrspatial/zonal.py) | Cross-tabulates agreement between two categorical rasters | ✅️ | ✅️| | |

-----------

### **Local**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Cell Stats](xrspatial/local.py) | Computes summary statistics across multiple rasters per cell | ✅️ |  | | |
| [Combine](xrspatial/local.py) | Assigns unique IDs to each distinct combination across rasters | ✅️ | | | |
| [Lesser Frequency](xrspatial/local.py) | Counts how many rasters have values less than a reference | ✅️ |  | | |
| [Equal Frequency](xrspatial/local.py) | Counts how many rasters have values equal to a reference | ✅️ |  | | |
| [Greater Frequency](xrspatial/local.py) | Counts how many rasters have values greater than a reference | ✅️ |  | | |
| [Lowest Position](xrspatial/local.py) | Identifies which raster has the minimum value at each cell | ✅️ | | | |
| [Highest Position](xrspatial/local.py) | Identifies which raster has the maximum value at each cell | ✅️ | | | |
| [Popularity](xrspatial/local.py) | Returns the value from the most common position across rasters | ✅️ | | | |
| [Rank](xrspatial/local.py) | Ranks cell values across multiple rasters per cell | ✅️ | | | |

#### Usage

##### Basic Pattern
```python
import xarray as xr
from xrspatial import hillshade

my_dataarray = xr.DataArray(...)
hillshaded_dataarray = hillshade(my_dataarray)
```

##### Dataset Support

Most functions also accept an `xr.Dataset`. Single-input functions (surface, classification, focal, proximity) apply the operation to each data variable and return a Dataset. Multi-input functions (multispectral indices) accept a Dataset with band-name keyword arguments.

```python
# Single-input: returns a Dataset with slope computed for each variable
slope_ds = slope(my_dataset)

# Multi-input: map Dataset variables to band parameters
ndvi_result = ndvi(my_dataset, nir='band_5', red='band_4')

# Zonal stats: columns prefixed by variable name
stats_df = zonal.stats(zones, my_dataset)  # → elevation_mean, temperature_mean, ...
```

Check out the user guide [here](/examples/user_guide/).

------

![title](img/composite_map.png)
![title](img/makepath-supply-chain-international-shipping.png)

#### Dependencies

`xarray-spatial` currently depends on Datashader, but will soon be updated to depend only on `xarray` and `numba`, while still being able to make use of Datashader output when available.

![title](img/dependencies.svg)

#### Notes on GDAL

Within the Python ecosystem, many geospatial libraries interface with the GDAL C++ library for raster and vector input, output, and analysis (e.g. rasterio, rasterstats, geopandas). GDAL is robust, performant, and has decades of great work behind it. For years, off-loading expensive computations to the C/C++ level in this way has been a key performance strategy for Python libraries (obviously...Python itself is implemented in C!).

However, wrapping GDAL has a few drawbacks for Python developers and data scientists:
- GDAL can be a pain to build / install.
- GDAL is hard for Python developers/analysts to extend, because it requires understanding multiple languages.
- GDAL's data structures are defined at the C/C++ level, which constrains how they can be accessed from Python.

With the introduction of projects like Numba, Python gained new ways to provide high-performance code directly in Python, without depending on or being constrained by separate C/C++ extensions. `xarray-spatial` implements algorithms using Numba and Dask, making all of its source code available as pure Python without any "black box" barriers that obscure what is going on and prevent full optimization. Projects can make use of the functionality provided by `xarray-spatial` where available, while still using GDAL where required for other tasks.

#### Citation
Cite this code:

`xarray-contrib/xarray-spatial, https://github.com/xarray-contrib/xarray-spatial, ©2020-2026.`
