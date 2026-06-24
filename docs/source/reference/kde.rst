..  _reference.kde:

***
KDE
***

.. note::

   Kernel density estimation converts point or line data into
   continuous density surfaces on a raster grid.

   ``kde`` accepts either raw ``x``, ``y`` arrays or a GeoDataFrame of
   Point geometries (``column`` selects per-point weights).  It is also
   available on the ``.xrs`` accessor, where the caller raster supplies the
   output grid and CRS::

       grid.xrs.kde(points_gdf, coregister=True)

   ``coregister=True`` reprojects the points from their CRS into the
   caller's CRS first.

KDE
===
.. autosummary::
   :toctree: _autosummary

   xrspatial.kde.kde

Line Density
============
.. autosummary::
   :toctree: _autosummary

   xrspatial.kde.line_density
