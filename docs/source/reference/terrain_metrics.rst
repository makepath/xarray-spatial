..  _reference.terrain_metrics:

***************
Terrain Metrics
***************

.. note::

   Terrain metrics use a 3x3 kernel and output **float32**.  Edge cells
   are NaN by default.  These functions use the planar (Horn) algorithm
   and assume the input is on a regular grid with uniform cell spacing.

Roughness
=========
.. autosummary::
    :toctree: _autosummary

    xrspatial.terrain_metrics.roughness

Topographic Position Index (TPI)
================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.terrain_metrics.tpi

Terrain Ruggedness Index (TRI)
==============================
.. autosummary::
    :toctree: _autosummary

    xrspatial.terrain_metrics.tri

Landform Classification
=======================
.. autosummary::
    :toctree: _autosummary

    xrspatial.terrain_metrics.landforms
