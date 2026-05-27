..  _getting_started.installation:

************
Installation
************

.. code-block:: bash

   # via pip
   pip install xarray-spatial

   # with plotting helpers (matplotlib)
   pip install xarray-spatial[plot]

   # via conda
   conda install -c conda-forge xarray-spatial

matplotlib is an optional dependency. The compute functions work without it;
install the ``plot`` extra to use the ``.xrs.plot`` accessor helpers.