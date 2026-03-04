<img src="img/Xarray-Spatial-logo.svg"/>

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <div>
        <img src="https://badge.fury.io/py/xarray-spatial.svg" alt="pypi version" />
        <img src="https://img.shields.io/conda/vn/conda-forge/xarray-spatial.svg" alt="conda-forge version" />
    </div>
  </td>

  <td>Downloads</td>
  <td>
    <div>
    <img src="https://img.shields.io/pypi/dm/xarray-spatial?label=PyPI"
         alt="PyPI downloads per month" />
    <img src="https://img.shields.io/conda/dn/conda-forge/xarray-spatial?label=conda-forge"
         alt="conda-forge downloads" />
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

✅ = native backend &nbsp;&nbsp; 🔄 = accepted (CPU fallback)

-------

### **Classification**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Box Plot](xrspatial/classify.py) | Classifies values into bins based on box plot quartile boundaries | ✅️ |✅ | ✅ | 🔄 |
| [Equal Interval](xrspatial/classify.py) | Divides the value range into equal-width bins | ✅️ |✅ | ✅ |✅ |
| [Head/Tail Breaks](xrspatial/classify.py) | Classifies heavy-tailed distributions using recursive mean splitting | ✅️ |✅ | 🔄 | 🔄 |
| [Maximum Breaks](xrspatial/classify.py) | Finds natural groupings by maximizing differences between sorted values | ✅️ |✅ | 🔄 | 🔄 |
| [Natural Breaks](xrspatial/classify.py) | Optimizes class boundaries to minimize within-class variance (Jenks) | ✅️ |✅ | 🔄 | 🔄 |
| [Percentiles](xrspatial/classify.py) | Assigns classes based on user-defined percentile breakpoints | ✅️ |✅ | ✅ | 🔄 |
| [Quantile](xrspatial/classify.py) | Distributes values into classes with equal observation counts | ✅️ |✅ | ✅ | 🔄 |
| [Reclassify](xrspatial/classify.py) | Remaps pixel values to new classes using a user-defined lookup | ✅️ |✅ | ✅ |✅ |
| [Std Mean](xrspatial/classify.py) | Classifies values by standard deviation intervals from the mean | ✅️ |✅ | ✅ |✅ |

-------

### **Diffusion**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Diffuse](xrspatial/diffusion.py) | Runs explicit forward-Euler diffusion on a 2D scalar field | ✅️ | ✅️ | ✅️ | ✅️ |

-------

### **Focal**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Apply](xrspatial/focal.py) | Applies a custom function over a sliding neighborhood window | ✅️ | ✅️ | ✅️ | ✅️ |
| [Hotspots](xrspatial/focal.py) | Identifies statistically significant spatial clusters using Getis-Ord Gi* | ✅️ | ✅️ | ✅️ | ✅️ |
| [Emerging Hotspots](xrspatial/emerging_hotspots.py) | Classifies time-series hot/cold spot trends using Gi* and Mann-Kendall | ✅️ | ✅️ | ✅️ | ✅️ |
| [Mean](xrspatial/focal.py) | Computes the mean value within a sliding neighborhood window | ✅️ | ✅️ | ✅️ | ✅️ |
| [Focal Statistics](xrspatial/focal.py) | Computes summary statistics over a sliding neighborhood window | ✅️ | ✅️ | ✅️ | ✅️ |

-------

### **Fire**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [dNBR](xrspatial/fire.py) | Differenced Normalized Burn Ratio (pre minus post NBR) | ✅️ | ✅️ | ✅️ | ✅️ |
| [RdNBR](xrspatial/fire.py) | Relative dNBR normalized by pre-fire vegetation density | ✅️ | ✅️ | ✅️ | ✅️ |
| [Burn Severity Class](xrspatial/fire.py) | USGS 7-class burn severity from dNBR thresholds | ✅️ | ✅️ | ✅️ | ✅️ |
| [Fireline Intensity](xrspatial/fire.py) | Byram's fireline intensity from fuel load and spread rate (kW/m) | ✅️ | ✅️ | ✅️ | ✅️ |
| [Flame Length](xrspatial/fire.py) | Flame length derived from fireline intensity (m) | ✅️ | ✅️ | ✅️ | ✅️ |
| [Rate of Spread](xrspatial/fire.py) | Simplified Rothermel spread rate with Anderson 13 fuel models (m/min) | ✅️ | ✅️ | ✅️ | ✅️ |
| [KBDI](xrspatial/fire.py) | Keetch-Byram Drought Index single time-step update (0-800 mm) | ✅️ | ✅️ | ✅️ | ✅️ |

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
| [True Color](xrspatial/multispectral.py) | Composites red, green, and blue bands into a natural color image | ✅️ | ✅️ | ✅️ | ✅️ |

-------


### **Multivariate**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Mahalanobis Distance](xrspatial/mahalanobis.py) | Measures statistical distance from a multi-band reference distribution, accounting for band correlations | ✅️ |✅️ | ✅️ |✅️ |

-------

### **Pathfinding**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [A* Pathfinding](xrspatial/pathfinding.py) | Finds the least-cost path between two cells on a cost surface | ✅️ | ✅ | 🔄 | 🔄 |
| [Multi-Stop Search](xrspatial/pathfinding.py) | Routes through N waypoints in sequence, with optional TSP reordering | ✅️ | ✅ | 🔄 | 🔄 |

----------

### **Proximity**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Allocation](xrspatial/proximity.py) | Assigns each cell to the identity of the nearest source feature | ✅️ | ✅ | ✅️ | ✅️ |
| [Balanced Allocation](xrspatial/balanced_allocation.py) | Partitions a cost surface into territories of roughly equal cost-weighted area | ✅️ | ✅ | ✅️ | ✅️ |
| [Cost Distance](xrspatial/cost_distance.py) | Computes minimum accumulated cost to the nearest source through a friction surface | ✅️ | ✅ | ✅️ | ✅️ |
| [Direction](xrspatial/proximity.py) | Computes the direction from each cell to the nearest source feature | ✅️ | ✅ | ✅️ | ✅️ |
| [Proximity](xrspatial/proximity.py) | Computes the distance from each cell to the nearest source feature | ✅️ | ✅ | ✅️ | ✅️ |
| [Surface Distance](xrspatial/surface_distance.py) | Computes distance along the 3D terrain surface to the nearest source | ✅️ | ✅ | ✅️ | ✅️ |
| [Surface Allocation](xrspatial/surface_distance.py) | Assigns each cell to the nearest source by terrain surface distance | ✅️ | ✅ | ✅️ | ✅️ |
| [Surface Direction](xrspatial/surface_distance.py) | Computes compass direction to the nearest source by terrain surface distance | ✅️ | ✅ | ✅️ | ✅️ |

--------

### **Raster to vector**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:-----|:------------|:------------------:|:-----------------:|:---------------------:|:---------------------:|
| [Polygonize](xrspatial/polygonize.py) | Converts contiguous regions of equal value into vector polygons | ✅️ | ✅️ | ✅️ | 🔄 |

--------

### **Surface**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Aspect](xrspatial/aspect.py) | Computes downslope direction of each cell in degrees | ✅️ | ✅️ | ✅️ | ✅️ |
| [Curvature](xrspatial/curvature.py) | Measures rate of slope change (concavity/convexity) at each cell | ✅️ |✅️ |✅️ | ✅️  |
| [Hillshade](xrspatial/hillshade.py) | Simulates terrain illumination from a given sun angle and azimuth | ✅️ | ✅️ | ✅️ | ✅️ |
| [Roughness](xrspatial/terrain_metrics.py) | Computes local relief as max minus min elevation in a 3×3 window | ✅️ | ✅️ | ✅️ | ✅️ |
| [Slope](xrspatial/slope.py) | Computes terrain gradient steepness at each cell in degrees | ✅️  | ✅️  | ✅️ | ✅️  |
| [Terrain Generation](xrspatial/terrain.py) | Generates synthetic terrain from fBm or ridged fractal noise with optional domain warping, Worley blending, and hydraulic erosion | ✅️ | ✅️ | ✅️ | ✅️ |
| [TPI](xrspatial/terrain_metrics.py) | Computes Topographic Position Index (center minus mean of neighbors) | ✅️ | ✅️ | ✅️ | ✅️ |
| [TRI](xrspatial/terrain_metrics.py) | Computes Terrain Ruggedness Index (local elevation variation) | ✅️ | ✅️ | ✅️ | ✅️ |
| [Viewshed](xrspatial/viewshed.py) | Determines visible cells from a given observer point on terrain | ✅️ | ✅️ | ✅️ | ✅️ |
| [Min Observable Height](xrspatial/experimental/min_observable_height.py) | Finds the minimum observer height needed to see each cell *(experimental)* | ✅️ | | | |
| [Perlin Noise](xrspatial/perlin.py) | Generates smooth continuous random noise for procedural textures | ✅️ | ✅️ | ✅️ | ✅️ |
| [Worley Noise](xrspatial/worley.py) | Generates cellular (Voronoi) noise returning distance to the nearest feature point | ✅️ | ✅️ | ✅️ | ✅️ |
| [Hydraulic Erosion](xrspatial/erosion.py) | Simulates particle-based water erosion to carve valleys and deposit sediment | ✅️ | 🔄 | 🔄 | 🔄 |
| [Bump Mapping](xrspatial/bump.py) | Adds randomized bump features to simulate natural terrain variation | ✅️ | ✅️ | ✅️ | ✅️ |

-----------

### **Hydrology**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Flow Direction (D8)](xrspatial/flow_direction.py) | Computes D8 flow direction from each cell toward the steepest downhill neighbor | ✅️ | ✅️ | ✅️ | ✅️ |
| [Flow Direction (Dinf)](xrspatial/flow_direction_dinf.py) | Computes D-infinity flow direction as a continuous angle toward the steepest downslope facet | ✅️ | ✅️ | ✅️ | ✅️ |
| [Flow Direction (MFD)](xrspatial/flow_direction_mfd.py) | Partitions flow to all downslope neighbors with an adaptive exponent (Qin et al. 2007) | ✅️ | ✅️ | ✅️ | ✅️ |
| [Flow Accumulation (D8)](xrspatial/flow_accumulation.py) | Counts upstream cells draining through each cell in a D8 flow direction grid | ✅️ | ✅️ | ✅️ | 🔄 |
| [Watershed](xrspatial/watershed.py) | Labels each cell with the pour point it drains to via D8 flow direction | ✅️ | ✅️ | ✅️ | 🔄 |
| [Basins](xrspatial/watershed.py) | Delineates drainage basins by labeling each cell with its outlet ID | ✅️ | ✅️ | ✅️ | 🔄 |
| [Stream Order](xrspatial/stream_order.py) | Assigns Strahler or Shreve stream order to cells in a drainage network | ✅️ | ✅️ | ✅️ | 🔄 |
| [Stream Link](xrspatial/stream_link.py) | Assigns unique IDs to each stream segment between junctions | ✅️ | ✅️ | ✅️ | 🔄 |
| [Snap Pour Point](xrspatial/snap_pour_point.py) | Snaps pour points to the highest-accumulation cell within a search radius | ✅️ | ✅️ | 🔄 | 🔄 |
| [Flow Path](xrspatial/flow_path.py) | Traces downstream flow paths from start points through a D8 direction grid | ✅️ | ✅️ | 🔄 | 🔄 |

-----------

### **Flood**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Flood Depth](xrspatial/flood.py) | Computes water depth above terrain from a HAND raster and water level | ✅️ | ✅️ | ✅️ | ✅️ |
| [Inundation](xrspatial/flood.py) | Produces a binary flood/no-flood mask from a HAND raster and water level | ✅️ | ✅️ | ✅️ | ✅️ |
| [Curve Number Runoff](xrspatial/flood.py) | Estimates runoff depth from rainfall using the SCS/NRCS curve number method | ✅️ | ✅️ | ✅️ | ✅️ |
| [Travel Time](xrspatial/flood.py) | Estimates overland flow travel time via simplified Manning's equation | ✅️ | ✅️ | ✅️ | ✅️ |

-----------

### **Interpolation**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [IDW](xrspatial/interpolate/_idw.py) | Inverse Distance Weighting from scattered points to a raster grid | ✅️ | ✅️ | ✅️ | ✅️ |
| [Kriging](xrspatial/interpolate/_kriging.py) | Ordinary Kriging with automatic variogram fitting (spherical, exponential, gaussian) | ✅️ | ✅️ | | |
| [Spline](xrspatial/interpolate/_spline.py) | Thin Plate Spline interpolation with optional smoothing | ✅️ | ✅️ | ✅️ | ✅️ |

-----------

### **Zonal**

| Name | Description | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Apply](xrspatial/zonal.py) | Applies a custom function to each zone in a classified raster | ✅️ | ✅️ | ✅️ | ✅️ |
| [Crop](xrspatial/zonal.py) | Extracts the bounding rectangle of a specific zone | ✅️ | ✅️ | ✅️ | ✅️ |
| [Regions](xrspatial/zonal.py) | Identifies connected regions of non-zero cells | ✅️ | ✅️ | ✅️ | ✅️ |
| [Trim](xrspatial/zonal.py) | Removes nodata border rows and columns from a raster | ✅️ | ✅️ | ✅️ | ✅️ |
| [Zonal Statistics](xrspatial/zonal.py) | Computes summary statistics for a value raster within each zone | ✅️ | ✅️| ✅️ | 🔄 |
| [Zonal Cross Tabulate](xrspatial/zonal.py) | Cross-tabulates agreement between two categorical rasters | ✅️ | ✅️| 🔄 | 🔄 |

#### Usage

##### Quick Start

Importing `xrspatial` registers an `.xrs` accessor on DataArrays and Datasets, giving you tab-completable access to every spatial operation:

```python
import numpy as np
import xarray as xr
import xrspatial

# Create or load a raster
elevation = xr.DataArray(np.random.rand(100, 100) * 1000, dims=['y', 'x'])

# Surface analysis — call operations directly on the DataArray
slope = elevation.xrs.slope()
hillshaded = elevation.xrs.hillshade(azimuth=315, angle_altitude=45)
aspect = elevation.xrs.aspect()

# Classification
classes = elevation.xrs.equal_interval(k=5)
breaks = elevation.xrs.natural_breaks(k=10)

# Proximity
distance = elevation.xrs.proximity(target_values=[1])

# Multispectral — call on the NIR band, pass other bands as arguments
nir = xr.DataArray(np.random.rand(100, 100), dims=['y', 'x'])
red = xr.DataArray(np.random.rand(100, 100), dims=['y', 'x'])
blue = xr.DataArray(np.random.rand(100, 100), dims=['y', 'x'])

vegetation = nir.xrs.ndvi(red)
enhanced_vi = nir.xrs.evi(red, blue)
```

##### Dataset Support

The `.xrs` accessor works on Datasets too. Single-input functions apply the operation to each data variable. Multi-input functions (multispectral indices) accept string kwargs that map band aliases to variable names:

```python
ds = xr.Dataset({'band_4': red, 'band_5': nir})

# Single-input: slope computed for each variable
slope_ds = ds.xrs.slope()

# Multi-input: map variable names to band parameters
ndvi_result = ds.xrs.ndvi(nir='band_5', red='band_4')
```

##### Function Import Style

All operations are also available as standalone functions if you prefer explicit imports:

```python
from xrspatial import hillshade, slope, ndvi

hillshaded = hillshade(elevation)
slope_result = slope(elevation)
vegetation = ndvi(nir, red)
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
