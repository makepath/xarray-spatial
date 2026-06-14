..  _getting_started.why_xarray_spatial:

*******************
Why xarray-spatial?
*******************

What is a raster?
=================

Rasters are regularly gridded datasets like GeoTIFFs, JPGs, and PNGs.

In the GIS world, rasters are used for representing continuous phenomena
(e.g. elevation, rainfall, distance), either directly as numerical values,
or as RGB images created for humans to view. Rasters typically have two
spatial dimensions, but may have any number of other dimensions
(time, type of measurement, etc.)

What xarray-spatial is
======================

``xarray-spatial`` is a raster analysis library built on xarray. It has
150+ functions for surface analysis, hydrology, fire behavior, flood
modeling, multispectral indices, proximity, classification, pathfinding,
and interpolation. Every function takes an ``xr.DataArray`` and returns
an ``xr.DataArray``, so operations chain without conversions.

Functions dispatch automatically across four backends based on the type
of the input array: NumPy, Dask, CuPy, and Dask+CuPy. The same
``slope(terrain)`` call works on an in-memory array, an out-of-core
chunked array, or a GPU array. Not every function reaches the same
maturity on every backend; the feature matrix in the `README
<https://github.com/xarray-contrib/xarray-spatial#readme>`_ and the
:doc:`/user_guide/stability_policy` page track the per-function,
per-backend tiers.

No GDAL, no GEOS
================

Within the Python ecosystem, many geospatial libraries interface with the
GDAL C++ library for raster and vector input, output, and analysis (e.g.
rasterio, rasterstats, geopandas). GDAL is robust, performant, and has
decades of great work behind it. For years, off-loading expensive
computations to the C/C++ level in this way has been a key performance
strategy for Python libraries (obviously...Python itself is implemented
in C!).

However, wrapping GDAL has a few drawbacks for Python developers and data
scientists:

* GDAL can be a pain to build / install.
* GDAL is hard for Python developers/analysts to extend, because it
  requires understanding multiple languages.
* GDAL's data structures are defined at the C/C++ level, which constrains
  how they can be accessed from Python.

``xarray-spatial`` does not depend on GDAL or GEOS at all. The compute
functions are implemented with Numba and Dask, and raster I/O,
reprojection, compression codecs, and coordinate handling are pure
Python and Numba (see :doc:`/reference/geotiff`). All of the source is
readable Python without any "black box" barriers that obscure what is
going on and prevent full optimization. Projects can use
``xarray-spatial`` where it fits while still using GDAL-based tools for
other tasks.

How it fits the ecosystem
=========================

Any 2D ``xr.DataArray`` works as input, including ones produced by other
libraries such as rioxarray; nothing about the inputs is specific to
``xarray-spatial``'s own reader. The project grew out of `Datashader
<https://datashader.org/>`_, which provides fast rasterization of vector
data (points, lines, polygons, meshes) for use with ``xarray-spatial``.
