..  _reference.interpolation:

*************
Interpolation
*************

.. note::

   Each function accepts either raw ``x``, ``y``, ``z`` arrays or a
   GeoDataFrame of Point geometries (``column`` selects the value to
   interpolate, defaulting to the first numeric column).  The same
   operations are available on the ``.xrs`` accessor, where the caller
   raster supplies the output grid, chunks, and CRS::

       dem.xrs.kriging(points_gdf, column="elevation")
       dem.xrs.spline(points_gdf, coregister=True)

   ``coregister=True`` reprojects a GeoDataFrame's points from their CRS
   into the caller's CRS before interpolation.

Inverse Distance Weighting (IDW)
=================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.interpolate.idw

Kriging
=======
.. autosummary::
    :toctree: _autosummary

    xrspatial.interpolate.kriging

Spline
======
.. autosummary::
    :toctree: _autosummary

    xrspatial.interpolate.spline
