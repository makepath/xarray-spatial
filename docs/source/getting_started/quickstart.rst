..  _getting_started.quickstart:

**********
Quickstart
**********

Everything on this page runs with just ``pip install xarray-spatial``.
No data download, no GDAL, no configuration.

Your first analysis
===================

``generate_terrain`` builds synthetic elevation data, so you can try the
library before touching real rasters:

.. code-block:: python

   import numpy as np
   import xarray as xr
   from xrspatial import generate_terrain, hillshade, slope

   template = xr.DataArray(np.zeros((400, 600)), dims=['y', 'x'])
   terrain = generate_terrain(template, x_range=(0, 6000), y_range=(0, 4000))

   shaded = hillshade(terrain)
   incline = slope(terrain)

Each function takes an ``xr.DataArray`` and returns a new ``xr.DataArray``
with the same shape, dims, and coords. If you have matplotlib installed
(the ``plot`` extra), look at the result with:

.. code-block:: python

   shaded.plot.imshow(cmap='gray')

The ``.xrs`` accessor
=====================

The same operations are available as methods on any DataArray through
the ``xrs`` accessor, which is handy for tab completion and method
chaining. These two lines are equivalent:

.. code-block:: python

   incline = slope(terrain)
   incline = terrain.xrs.slope()

To see what's available without tab completion or an editor, print the
accessor. Its repr lists the operations grouped by category, and renders
as a table in a Jupyter notebook:

.. code-block:: python

   terrain.xrs        # prints slope, hillshade, ndvi, watershed, ... by category

Reading and writing GeoTIFFs
============================

``xrspatial.geotiff`` reads and writes GeoTIFF / COG natively, GDAL not
required. A round trip:

.. code-block:: python

   from xrspatial.geotiff import open_geotiff, to_geotiff

   to_geotiff(terrain, 'terrain.tif', crs=3857, cog=True)

   dem = open_geotiff('terrain.tif')   # 2D DataArray for a single band
   shaded = hillshade(dem)

``open_geotiff`` also accepts ``http(s)://`` URLs and, with fsspec
installed, ``s3://`` / ``gs://`` / ``az://`` ones. See
:doc:`/reference/geotiff` for the full I/O surface and
:doc:`/user_guide/geotiff_safe_io` for its validation behavior.

Datasets
========

Most functions also accept an ``xr.Dataset``. Single-input functions apply
the operation to each data variable and return a Dataset. Multi-input
functions (multispectral indices) accept a Dataset with band-name keyword
arguments.

.. code-block:: python

   from xrspatial import slope
   from xrspatial.multispectral import ndvi

   # Single-input: returns a Dataset with slope for each variable
   slope_ds = slope(my_dataset)

   # Multi-input: map Dataset variables to band parameters
   ndvi_result = ndvi(my_dataset, nir='band_5', red='band_4')

Scaling up
==========

For rasters that don't fit in memory, chunk the input and the same
function call returns a lazy dask-backed result (needs the ``dask``
extra):

.. code-block:: python

   chunked = terrain.chunk({'y': 200, 'x': 200})
   lazy = slope(chunked)        # nothing computed yet
   result = lazy.compute()

See :doc:`/reference/dask_laziness` for which functions stay lazy and
how chunk boundaries are handled. For GPU execution, pass a
cupy-backed DataArray to the same functions; the :doc:`installation`
page covers GPU setup.

Before you rely on results
==========================

.. warning::

   Read :doc:`/user_guide/caveats` before production use. The big ones:
   geodesic terrain functions assume WGS84 coordinates, outputs are
   float32, and NaN is the nodata sentinel. The
   :doc:`/user_guide/stability_policy` page explains the per-function,
   per-backend feature tiers.

Check out the :doc:`user guide </user_guide/index>` for worked examples
of every module.
