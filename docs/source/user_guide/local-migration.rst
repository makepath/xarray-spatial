.. _local-migration:

***************************************************
Migrating from ``xrspatial.local`` to native xarray
***************************************************

The ``xrspatial.local`` module was removed in v0.6.  The functions it provided
(``cell_stats``, ``combine``, ``lesser_frequency``, ``equal_frequency``,
``greater_frequency``, ``lowest_position``, ``highest_position``,
``popularity``, ``rank``) were thin wrappers around operations that xarray and
NumPy already support natively.

The xarray equivalents below are vectorized, support Dask for lazy/parallel
evaluation, and work with CuPy-backed arrays for GPU acceleration — none of
which the old ``xrspatial.local`` functions supported.

Setup used in all examples
==========================

.. code-block:: python

    import numpy as np
    import xarray as xr

    arr1 = xr.DataArray([[np.nan, 4, 2, 0],
                         [2, 3, np.nan, 1],
                         [5, 1, 2, 0],
                         [1, 3, 2, np.nan]], name="arr1")

    arr2 = xr.DataArray([[3, 1, 1, 2],
                         [4, 1, 2, 5],
                         [0, 0, 0, 0],
                         [np.nan, 1, 1, 1]], name="arr2")

    arr3 = xr.DataArray([[3, 3, 2, 0],
                         [4, 1, 3, 1],
                         [6, 1, 2, 2],
                         [0, 0, 1, 1]], name="arr3")

    ds = xr.merge([arr1, arr2, arr3])

    # Stack all variables into a single DataArray with a "var" dimension.
    # This is the key building block for all the replacements below.
    stacked = ds.to_array(dim="var")


Cell Statistics
===============

``cell_stats(ds, func='sum')`` computed per-cell statistics across variables.

.. code-block:: python

    # sum (default)
    stacked.sum(dim="var")

    # max / mean / median / min / std
    stacked.max(dim="var")
    stacked.mean(dim="var")
    stacked.median(dim="var")
    stacked.min(dim="var")
    stacked.std(dim="var")


Combine
=======

``combine(ds)`` assigned a unique integer ID to each distinct combination of
values across variables.

.. code-block:: python

    import numpy as np
    import xarray as xr

    # Build a structured view of each cell's value-tuple, then use np.unique
    vals = np.stack([ds[v].values for v in ds.data_vars], axis=-1)   # (H, W, N)
    shape = vals.shape[:2]
    flat = vals.reshape(-1, vals.shape[-1])

    # Mask rows containing any NaN
    has_nan = np.isnan(flat).any(axis=1)
    _, inverse = np.unique(flat[~has_nan], axis=0, return_inverse=True)

    result = np.full(flat.shape[0], np.nan)
    result[~has_nan] = inverse + 1   # 1-based IDs, matching old behaviour

    combined = xr.DataArray(result.reshape(shape))


Lesser / Equal / Greater Frequency
===================================

``lesser_frequency(ds, ref_var)``, ``equal_frequency(ds, ref_var)``, and
``greater_frequency(ds, ref_var)`` counted how many variables had values less
than, equal to, or greater than a reference variable at each cell.

.. code-block:: python

    ref = ds["arr1"]

    # lesser_frequency — count of variables whose value < ref
    (stacked < ref).sum(dim="var")

    # equal_frequency
    (stacked == ref).sum(dim="var")

    # greater_frequency
    (stacked > ref).sum(dim="var")

.. note::

    If ``ref_var`` was one of the ``data_vars``, the old function excluded it
    from the comparison set.  To replicate that, drop it from the stack first::

        others = ds.drop_vars("arr1").to_array(dim="var")
        (others < ds["arr1"]).sum(dim="var")


Lowest / Highest Position
=========================

``lowest_position(ds)`` and ``highest_position(ds)`` returned the 1-based
index of the variable with the minimum or maximum value at each cell.

.. code-block:: python

    # lowest_position (1-based)
    stacked.argmin(dim="var") + 1

    # highest_position (1-based)
    stacked.argmax(dim="var") + 1


Popularity
==========

``popularity(ds, ref_var)`` returned the *n*-th most common unique value across
the other variables, where *n* came from the reference variable.

There is no single-expression xarray equivalent — use a small NumPy helper:

.. code-block:: python

    import numpy as np
    import xarray as xr

    def popularity(ds, ref_var, data_vars=None):
        if data_vars is None:
            data_vars = [v for v in ds.data_vars if v != ref_var]
        vals = np.stack([ds[v].values for v in data_vars], axis=-1)
        ref = ds[ref_var].values
        out = np.full(ref.shape, np.nan)
        for idx in np.ndindex(ref.shape):
            cell = vals[idx]
            if np.isnan(cell).any():
                continue
            unique_sorted = np.unique(cell)
            n = int(ref[idx]) - 1
            if 0 <= n < len(unique_sorted) and len(unique_sorted) < len(cell):
                out[idx] = unique_sorted[n]
        return xr.DataArray(out)


Rank
====

``rank(ds, ref_var)`` returned the value at the *n*-th sorted position across
the other variables, where *n* came from the reference variable.

.. code-block:: python

    import numpy as np
    import xarray as xr

    def rank(ds, ref_var, data_vars=None):
        if data_vars is None:
            data_vars = [v for v in ds.data_vars if v != ref_var]
        vals = np.stack([ds[v].values for v in data_vars], axis=-1)
        ref = ds[ref_var].values
        out = np.full(ref.shape, np.nan)
        for idx in np.ndindex(ref.shape):
            cell = vals[idx]
            if np.isnan(cell).any():
                continue
            n = int(ref[idx]) - 1
            if 0 <= n < len(cell):
                out[idx] = np.sort(cell)[n]
        return xr.DataArray(out)
