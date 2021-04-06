..  _index:

*******************************************************
Xarray-spatial: Raster-Based Spatial Analysis in Python
*******************************************************

**Xarray-Spatial implements common raster analysis functions using Numba and provides an easy-to-install, easy-to-extend codebase for raster analysis.**

xarray-spatial grew out of the `Datashader project <https://datashader.org/>`_, which provides fast rasterization of vector data (points, lines, polygons, meshes, and rasters) for use with xarray-spatial.

xarray-spatial does not depend on GDAL / GEOS, which makes it fully extensible in Python but does limit the breadth of operations that can be covered.  xarray-spatial is meant to include the core raster-a

-------

.. raw:: html
   :file: _templates/description_band.html

-------

.. raw:: html
   :file: _templates/examples.html

Installation
============

.. code-block:: bash

   # via pip
   pip install xarray-spatial

   # via conda
   conda install -c conda-forge xarray-spatial


Raster-huh?
===========

Rasters are regularly gridded datasets like GeoTIFFs, JPGs, and PNGs.

In the GIS world, rasters are used for representing continuous phenomena (e.g. elevation, rainfall, distance), either directly as numerical values, or as RGB images created for humans to view. Rasters typically have two spatial dimensions, but may have any number of other dimensions (time, type of measurement, etc.)


Supported Spatial Functions with Supported Inputs
=================================================
.. toctree::
   :hidden:

   self


.. toctree::
   :maxdepth: 1

   classification
   focal
   multispectral
   pathfinding
   proximity
   surface
   zonal

Usage
=====

Basic Pattern
-------------

.. code-block:: python

   import xarray as xr
   from xrspatial import hillshade

   my_dataarray = xr.DataArray(...)
   hillshaded_dataarray = hillshade(my_dataarray)

Check out the user guide `here <https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide>`_.


Dependencies
------------

``xarray-spatial`` currently depends on Datashader, but will soon be updated to depend only on ``xarray`` and ``numba``\ , while still being able to make use of Datashader output when available. 


.. image:: ./_static/img/dependencies.svg
   :target: ./_static/img/dependencies.svg
   :alt: title


Notes on GDAL
-------------

Within the Python ecosystem, many geospatial libraries interface with the GDAL C++ library for raster and vector input, output, and analysis (e.g. rasterio, rasterstats, geopandas). GDAL is robust, performant, and has decades of great work behind it. For years, off-loading expensive computations to the C/C++ level in this way has been a key performance strategy for Python libraries (obviously...Python itself is implemented in C!).

However, wrapping GDAL has a few drawbacks for Python developers and data scientists:


* GDAL can be a pain to build / install.
* GDAL is hard for Python developers/analysts to extend, because it requires understanding multiple languages.
* GDAL's data structures are defined at the C/C++ level, which constrains how they can be accessed from Python.

With the introduction of projects like Numba, Python gained new ways to provide high-performance code directly in Python, without depending on or being constrained by separate C/C++ extensions. ``xarray-spatial`` implements algorithms using Numba and Dask, making all of its source code available as pure Python without any "black box" barriers that obscure what is going on and prevent full optimization. Projects can make use of the functionality provided by ``xarray-spatial`` where available, while still using GDAL where required for other tasks.
