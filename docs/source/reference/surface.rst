..  _surface:

*******
Surface
*******

.. danger::

   ``slope()``, ``aspect()``, ``curvature()``, and ``hillshade()`` with
   ``method='geodesic'`` assume the **WGS84 ellipsoid** and require
   coordinates in **degrees** (geographic CRS).  Passing projected
   coordinates (metres) to the geodesic method produces wrong results.
   Use ``method='planar'`` (the default) for projected data.

.. note::

   All surface functions output **float32** regardless of input dtype.
   Edge cells within the 3x3 kernel radius are NaN by default
   (``boundary='nan'``).

Aspect
======
.. autosummary::
    :toctree: _autosummary

    xrspatial.aspect.aspect

Northness
=========
.. autosummary::
    :toctree: _autosummary

    xrspatial.aspect.northness

Eastness
========
.. autosummary::
    :toctree: _autosummary

    xrspatial.aspect.eastness

Curvature
=========
.. autosummary::
    :toctree: _autosummary

    xrspatial.curvature.curvature

Hillshade
=========
.. autosummary::
    :toctree: _autosummary

    xrspatial.hillshade.hillshade

Slope
=====
.. autosummary::
    :toctree: _autosummary

    xrspatial.slope.slope

Terrain Generation
==================
.. autosummary::
    :toctree: _autosummary

    xrspatial.terrain.generate_terrain

Sky-View Factor
===============
.. autosummary::
    :toctree: _autosummary

    xrspatial.sky_view_factor.sky_view_factor

Viewshed
========
.. autosummary::
    :toctree: _autosummary

    xrspatial.viewshed.viewshed

Cumulative Viewshed
===================
.. autosummary::
    :toctree: _autosummary

    xrspatial.visibility.cumulative_viewshed

Visibility Frequency
====================
.. autosummary::
    :toctree: _autosummary

    xrspatial.visibility.visibility_frequency

Line of Sight
=============
.. autosummary::
    :toctree: _autosummary

    xrspatial.visibility.line_of_sight

Perlin Noise
============
.. autosummary::
    :toctree: _autosummary

    xrspatial.perlin.perlin

Bump Mapping
============
.. autosummary::
    :toctree: _autosummary

    xrspatial.bump.bump

Erosion
=======
.. autosummary::
    :toctree: _autosummary

    xrspatial.erosion.erode
