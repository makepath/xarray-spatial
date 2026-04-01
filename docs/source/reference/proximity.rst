..  _proximity:

*********
Proximity
*********

.. warning::

   ``proximity()`` returns distances in **pixel units** by default
   (``distance_metric='EUCLIDEAN'``).  Multiply by cell size to get
   real-world units, or use ``'GREAT_CIRCLE'`` for lat/lon data
   (returns kilometres).

.. caution::

   With Dask, ``proximity()`` expands each chunk by ``max_distance`` cells.
   If ``max_distance`` is infinite (the default), the whole array is loaded
   into a single chunk.  Set a finite ``max_distance`` to keep memory
   bounded.

Allocation
==========
.. autosummary::
    :toctree: _autosummary

    xrspatial.proximity.allocation

Direction
==========
.. autosummary::
    :toctree: _autosummary

    xrspatial.proximity.direction

Proximity
==========
.. autosummary::
    :toctree: _autosummary

    xrspatial.proximity.euclidean_distance
    xrspatial.proximity.great_circle_distance
    xrspatial.proximity.manhattan_distance
    xrspatial.proximity.proximity

Cost Distance
==============
.. autosummary::
    :toctree: _autosummary

    xrspatial.cost_distance.cost_distance

Least-Cost Corridor
====================
.. autosummary::
    :toctree: _autosummary

    xrspatial.corridor.least_cost_corridor

Balanced Allocation
====================
.. autosummary::
    :toctree: _autosummary

    xrspatial.balanced_allocation.balanced_allocation

Surface Distance
================
.. autosummary::
    :toctree: _autosummary

    xrspatial.surface_distance.surface_distance

Surface Allocation
==================
.. autosummary::
    :toctree: _autosummary

    xrspatial.surface_distance.surface_allocation

Surface Direction
=================
.. autosummary::
    :toctree: _autosummary

    xrspatial.surface_distance.surface_direction
