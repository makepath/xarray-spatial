..  _zonal:

*****
Zonal
*****

.. note::

   NaN values are excluded from all zonal aggregations.  A zone where
   every cell is NaN returns NaN (not zero) for sum, mean, etc.

Apply
=====
.. autosummary::
    :toctree: _autosummary

    xrspatial.zonal.apply

Crop
====
.. autosummary::
    :toctree: _autosummary

    xrspatial.zonal.crop

Regions
=======
.. autosummary::
    :toctree: _autosummary

    xrspatial.zonal.regions

Sieve
=====
.. autosummary::
    :toctree: _autosummary

    xrspatial.sieve.sieve

Trim
====
.. autosummary::
    :toctree: _autosummary

    xrspatial.zonal.trim

Zonal Statistics
================
.. autosummary::
    :toctree: _autosummary

    xrspatial.zonal.get_full_extent
    xrspatial.zonal.stats
    xrspatial.zonal.suggest_zonal_canvas

Zonal Cross Tabulate
====================
.. autosummary::
    :toctree: _autosummary

    xrspatial.zonal.crosstab
