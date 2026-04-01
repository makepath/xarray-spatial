..  _user_guide.data_types:

************************
Data Type Handling
************************

This guide explains how xarray-spatial handles floating-point data types (float32 vs float64) and the implications for memory usage, precision, and performance.

Overview
========

xarray-spatial standardizes on **float32** (32-bit floating-point) for output data in most analytical functions. This design decision provides a balance between computational precision, memory efficiency, and performance that is well-suited for raster analysis tasks.

.. warning::

   All multispectral indices, terrain analysis functions, and most other
   operations in xarray-spatial output **float32** data, regardless of input
   data type.  If you pass float64 data, the output is float32.  Cast the
   result with ``.astype('float64')`` if you need higher precision.


Why Float32?
============

The decision to use float32 as the standard output type is based on several considerations:

Memory Efficiency
-----------------

Float32 uses half the memory of float64:

For large rasters or when working with multiple bands, the memory savings can be substantial.

Adequate Precision
------------------

Float32 provides approximately 7 significant decimal digits of precision, which is more than sufficient for most geospatial analysis tasks:

- **Vegetation indices** (NDVI, EVI, SAVI, etc.): Values typically range from -1.0 to +1.0
- **Terrain metrics** (slope, aspect, curvature): Values are typically expressed in degrees or simple ratios
- **Spectral indices**: Results are normalized ratios with limited dynamic range

Industry Standard
-----------------

Float32 is the de facto standard in the remote sensing and GIS industry:

- **GDAL** defaults to float32 for many raster operations
- **QGIS** raster calculators commonly output float32
- **Satellite imagery** (Landsat, Sentinel) is distributed in integer formats that easily fit within float32 precision


Input Data Type Handling
========================

xarray-spatial accepts a wide variety of input data types and automatically converts them to float32 for calculations:

.. code-block:: python

   import numpy as np
   import xarray as xr
   from xrspatial.multispectral import ndvi

   # Integer input (common for raw satellite imagery)
   nir_uint16 = xr.DataArray(np.array([[1000, 1500], [2000, 2500]], dtype=np.uint16))
   red_uint16 = xr.DataArray(np.array([[500, 700], [800, 900]], dtype=np.uint16))

   # Output will be float32
   result = ndvi(nir_agg=nir_uint16, red_agg=red_uint16)
   print(result.dtype)  # float32

   # Float64 input
   nir_float64 = xr.DataArray(np.array([[0.1, 0.15], [0.2, 0.25]], dtype=np.float64))
   red_float64 = xr.DataArray(np.array([[0.05, 0.07], [0.08, 0.09]], dtype=np.float64))

   # Output will still be float32
   result = ndvi(nir_agg=nir_float64, red_agg=red_float64)
   print(result.dtype)  # float32


Multispectral Functions
=======================

All multispectral indices convert input data to float32 before performing calculations and return float32 results


Example: NDVI with Different Input Types
----------------------------------------

.. code-block:: python

   import numpy as np
   import xarray as xr
   from xrspatial.multispectral import ndvi

   # Create sample data with different dtypes
   shape = (100, 100)

   # Simulate Landsat-style uint16 data (common for satellite imagery)
   nir_data = np.random.randint(5000, 15000, shape, dtype=np.uint16)
   red_data = np.random.randint(2000, 8000, shape, dtype=np.uint16)

   nir = xr.DataArray(nir_data, dims=['y', 'x'])
   red = xr.DataArray(red_data, dims=['y', 'x'])

   # Calculate NDVI - automatically converts to float32 internally
   vegetation_index = ndvi(nir_agg=nir, red_agg=red)

   print(f"Input dtype: {nir.dtype}")           # uint16
   print(f"Output dtype: {vegetation_index.dtype}")  # float32
   print(f"Output range: [{vegetation_index.min().values:.3f}, {vegetation_index.max().values:.3f}]")


Terrain Analysis Functions
==========================

Surface and terrain analysis functions follow the same float32 convention:

.. code-block:: python

   import numpy as np
   import xarray as xr
   from xrspatial import slope, aspect

   # Integer elevation data (e.g., from a DEM in meters)
   elevation = xr.DataArray(
       np.random.randint(0, 3000, (100, 100), dtype=np.int16),
       dims=['y', 'x']
   )

   # Both outputs will be float32
   slope_result = slope(elevation)
   aspect_result = aspect(elevation)

   print(f"Slope dtype: {slope_result.dtype}")    # float32
   print(f"Aspect dtype: {aspect_result.dtype}")  # float32


Focal Operations
================

Focal statistics and convolution operations also use float32:

.. code-block:: python

   from xrspatial.focal import mean, focal_stats
   from xrspatial.convolution import convolve_2d, circle_kernel

   # Integer input data
   data = xr.DataArray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))

   # Focal operations convert to and output float32
   kernel = circle_kernel(1, 1, 3)
   smoothed = convolve_2d(data, kernel)
   print(f"Convolution output dtype: {smoothed.dtype}")  # float32


Backend Consistency
===================

xarray-spatial ensures consistent float32 output across all computational backends:

NumPy Backend
-------------

.. code-block:: python

   import numpy as np
   import xarray as xr
   from xrspatial.multispectral import ndvi

   nir = xr.DataArray(np.random.rand(100, 100).astype(np.float64))
   red = xr.DataArray(np.random.rand(100, 100).astype(np.float64))

   result = ndvi(nir, red)
   print(result.dtype)  # float32

Dask Backend
------------

.. code-block:: python

   import dask.array as da
   import xarray as xr
   from xrspatial.multispectral import ndvi

   # Dask arrays for out-of-core computation
   nir = xr.DataArray(da.random.random((1000, 1000), chunks=(250, 250)))
   red = xr.DataArray(da.random.random((1000, 1000), chunks=(250, 250)))

   result = ndvi(nir, red)
   print(result.dtype)  # float32

CuPy Backend (GPU)
------------------

.. code-block:: python

   import cupy as cp
   import xarray as xr
   from xrspatial.multispectral import ndvi

   # CuPy arrays for GPU computation
   nir = xr.DataArray(cp.random.rand(100, 100).astype(cp.float64))
   red = xr.DataArray(cp.random.rand(100, 100).astype(cp.float64))

   result = ndvi(nir, red)
   print(result.dtype)  # float32


Best Practices
==============

1. **Don't upcast unnecessarily**: If your input data is uint8 or uint16, there's no need to convert to float64 before passing to xarray-spatial functions.

2. **Trust the output type**: The float32 output is intentional and provides adequate precision for geospatial analysis.

3. **Consider memory when scaling**: When working with large rasters or many bands, the 50% memory savings of float32 vs float64 can be significant.

4. **Chain operations efficiently**: xarray-spatial functions can be chained together without precision loss, as intermediate results maintain float32 precision.

.. code-block:: python

   from xrspatial.multispectral import ndvi, savi

   # Efficient chaining - all operations use float32 internally
   ndvi_result = ndvi(nir, red)
   savi_result = savi(nir, red)

   # Combine results (still float32)
   combined = (ndvi_result + savi_result) / 2


Dataset Input Support
=====================

Most functions accept an ``xr.Dataset`` in addition to ``xr.DataArray``.
When a Dataset is passed, the operation is applied to each data variable
independently and the result is returned as a new Dataset.

Single-input functions (surface, classification, focal, proximity):

.. code-block:: python

   from xrspatial import slope

   # Apply slope to every variable in the Dataset
   slope_ds = slope(my_dataset)
   # Returns an xr.Dataset with the same variable names

Multi-input functions (multispectral indices) accept a Dataset with keyword
arguments that map band aliases to variable names:

.. code-block:: python

   from xrspatial.multispectral import ndvi

   # Map Dataset variables to band parameters
   ndvi_result = ndvi(my_dataset, nir='band_5', red='band_4')

``zonal.stats`` also accepts a Dataset for the ``values`` parameter, returning
a merged DataFrame with columns prefixed by variable name:

.. code-block:: python

   from xrspatial.zonal import stats

   df = stats(zones, my_dataset)
   # Columns: zone, elevation_mean, elevation_max, ..., temperature_mean, ...


Summary
=======

- **Input**: xarray-spatial accepts any numeric data type (int or float), as either ``xr.DataArray`` or ``xr.Dataset``
- **Processing**: All calculations are performed in float32 precision
- **Output**: Results are returned as float32 DataArrays (or a Dataset of float32 DataArrays when a Dataset is passed)
- **Consistency**: This behavior is consistent across NumPy, Dask, and CuPy backends
- **Rationale**: Float32 provides adequate precision for geospatial analysis while using half the memory of float64
