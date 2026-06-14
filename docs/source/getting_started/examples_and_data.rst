..  _getting_started.examples_and_data:

******************
Examples and data
******************

The quickstart runs on synthetic terrain and needs no downloads. The
example notebooks are different: they work on real rasters and vector
files that ship separately. The ``xrspatial`` command-line tool fetches
both.

The command-line tool
=====================

Installing the package puts an ``xrspatial`` command on your ``PATH``.
Run it inside the same environment where you installed the library. Three
subcommands download the starter material:

.. code-block:: bash

   xrspatial examples         # notebooks and the data they use
   xrspatial copy-examples    # notebooks only
   xrspatial fetch-data       # data only

Each one writes into a folder named ``xrspatial-examples`` in your
current directory. A fourth subcommand, ``xrspatial clean-data``,
removes downloaded data files.

The command list and per-command options are available with ``--help``:

.. code-block:: bash

   xrspatial --help
   xrspatial examples --help

If ``xrspatial`` reports a missing command, install the CLI helper with
``pip install pyct``.

Getting started with the examples
=================================

Download everything and open the notebooks:

.. code-block:: bash

   xrspatial examples
   cd xrspatial-examples
   jupyter notebook

Notebooks live under ``user_guide/`` and cover surface analysis,
proximity, classification, remote sensing, and the rest of the modules.
Downloaded datasets land in ``xrspatial-examples/data/``. The same
notebooks are rendered in the :doc:`user guide </user_guide/index>` if
you would rather read than run.

Re-running a download overwrites nothing by default. Pass ``--force`` to
replace files that already exist, or ``--path`` to write somewhere other
than the current directory:

.. code-block:: bash

   xrspatial examples --path ~/xrspatial-demo --force

Using your own data
===================

You do not need the example data to use the library. Any 2D
``xr.DataArray`` works as input, whether you build it yourself, read it
with :doc:`open_geotiff </reference/geotiff>`, or get it from another
library. The :doc:`quickstart` shows the no-download path end to end, and
:doc:`/reference/geotiff` covers reading local files and ``http(s)`` /
``s3`` / ``gs`` / ``az`` URLs.
