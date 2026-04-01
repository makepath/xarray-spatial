..  _pathfinding:

***********
Pathfinding
***********

.. caution::

   A* allocates about 65 bytes per pixel and will raise ``MemoryError``
   if the required memory exceeds 80 % of available RAM.  Use Dask or
   set ``search_radius`` to limit the search area for large rasters.

.. warning::

   NaN and non-positive friction values are treated as impassable
   barriers.  Cells must have finite positive friction to be traversable.

A* Pathfinding
==============
.. autosummary::
    :toctree: _autosummary

    xrspatial.pathfinding.a_star_search
