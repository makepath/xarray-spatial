..  _getting_started.installation:

************
Installation
************

``xarray-spatial`` requires Python 3.12 or newer.

.. code-block:: bash

   # via pip
   pip install xarray-spatial

   # with plotting helpers (matplotlib)
   pip install xarray-spatial[plot]

   # with vector rasterization (shapely): rasterize, polygonize
   pip install xarray-spatial[vector]

   # via conda
   conda install -c conda-forge xarray-spatial

matplotlib and shapely are optional dependencies. The compute functions work
without either; install the ``plot`` extra for the ``.xrs.plot`` accessor
helpers, and the ``vector`` extra for the ``rasterize`` and ``polygonize``
vector-to-raster paths.