.. _reference.dask_laziness:

*********************
Dask backend behavior
*********************

When you pass a dask-backed ``DataArray`` to an xarray-spatial function, the
result *should* also be dask-backed so your pipeline stays lazy until you call
``.compute()``.  Most functions do this, but some algorithms need random access
to the full array and have to materialize intermediate results.

This page lists every public function and its laziness level so you can plan
dask pipelines without reading source code.

Laziness levels
===============

**Fully lazy** -- the function returns a dask array without triggering any
computation.  Safe for arbitrarily large out-of-core datasets.

**Partially lazy** -- the function computes small bounded statistics (scalars,
quartiles, a ~20K sample) during setup, then returns a dask array for the main
result.  The statistics are cheap; the heavy work stays lazy.

**Fully materialized** -- the algorithm needs the entire array in memory
(connected-component labeling, A* search, viewshed sweepline, etc.).  The
result may be re-wrapped as dask, but the function calls ``.compute()``
internally.  Watch your memory on large inputs.


Terrain metrics
===============

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Laziness
     - Notes
   * - ``slope``
     - Fully lazy
     - ``map_overlap``, planar and geodesic
   * - ``aspect``
     - Fully lazy
     - ``map_overlap``, planar and geodesic
   * - ``curvature``
     - Fully lazy
     - ``map_overlap``
   * - ``hillshade``
     - Fully lazy
     - ``map_overlap``
   * - ``northness``
     - Fully lazy
     - Uses ``da.cos`` / ``da.deg2rad`` on aspect output
   * - ``eastness``
     - Fully lazy
     - Uses ``da.sin`` / ``da.deg2rad`` on aspect output


Focal operations
================

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Laziness
     - Notes
   * - ``mean``
     - Fully lazy
     - Iterative ``map_overlap``
   * - ``apply``
     - Fully lazy
     - ``map_overlap`` with user kernel
   * - ``focal_stats``
     - Fully lazy
     - Multiple stats via ``map_overlap``, 3D output
   * - ``hotspots``
     - Partially lazy
     - Computes global mean and std, result is dask


Classification
==============

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Laziness
     - Notes
   * - ``binary``
     - Fully lazy
     - ``map_blocks``
   * - ``reclassify``
     - Fully lazy
     - ``map_blocks``
   * - ``quantile``
     - Partially lazy
     - Computes percentiles from ~20K sample
   * - ``natural_breaks``
     - Partially lazy
     - Computes Jenks breaks from ~20K sample + scalar max
   * - ``equal_interval``
     - Partially lazy
     - Computes scalar min/max
   * - ``std_mean``
     - Partially lazy
     - Computes scalar mean/std/max
   * - ``head_tail_breaks``
     - Partially lazy
     - Computes O(log N) scalar means
   * - ``percentiles``
     - Partially lazy
     - Computes percentiles from ~20K sample
   * - ``maximum_breaks``
     - Partially lazy
     - Computes breaks from ~20K sample
   * - ``box_plot``
     - Partially lazy
     - Computes scalar quartiles and max


Normalization
=============

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Laziness
     - Notes
   * - ``rescale``
     - Fully lazy
     - ``da.nanmin`` / ``da.nanmax`` (lazy reductions)
   * - ``standardize``
     - Fully lazy
     - ``da.nanmean`` / ``da.nanstd`` (lazy reductions)


Visibility
==========

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Laziness
     - Notes
   * - ``viewshed``
     - Fully materialized
     - Sweepline algorithm needs random access
   * - ``line_of_sight``
     - Fully materialized
     - Extracts 1D transect via ``.compute()``
   * - ``cumulative_viewshed``
     - Fully materialized
     - Runs multiple viewshed calls
   * - ``visibility_frequency``
     - Fully materialized
     - Wraps ``cumulative_viewshed``


Morphology
==========

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Laziness
     - Notes
   * - ``sieve``
     - Fully materialized
     - Connected-component labeling needs the full array; result re-wrapped as dask


Proximity
=========

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Laziness
     - Notes
   * - ``proximity``
     - Fully materialized
     - Distance computation needs full array
   * - ``allocation``
     - Fully materialized
     - Nearest-source allocation
   * - ``direction``
     - Fully materialized
     - Direction to nearest source


Zonal
=====

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Laziness
     - Notes
   * - ``zonal_stats`` / ``stats``
     - Partially lazy
     - Groupby aggregation via dask dataframe
   * - ``zonal_crosstab`` / ``crosstab``
     - Partially lazy
     - Groupby cross-tabulation
   * - ``zonal_apply`` / ``apply``
     - Fully lazy
     - ``map_blocks`` per zone
   * - ``regions``
     - Fully materialized
     - Connected-component labeling
   * - ``trim``
     - Fully lazy
     - Lazy slicing
   * - ``crop``
     - Fully lazy
     - Lazy slicing


Pathfinding
===========

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Laziness
     - Notes
   * - ``a_star_search``
     - Fully materialized
     - A* needs random access and visited-set tracking
   * - ``multi_stop_search``
     - Fully materialized
     - Iterative A*
