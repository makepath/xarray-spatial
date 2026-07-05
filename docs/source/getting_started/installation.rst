..  _getting_started.installation:

************
Installation
************

Requirements
============

``xarray-spatial`` requires Python 3.12 or newer and runs on Linux, macOS,
and Windows. The required core is small: ``numpy``, ``numba``, ``scipy``,
``xarray``, ``urllib3``, and ``zstandard``. There is no GDAL or GEOS
anywhere in the stack, so nothing needs to be compiled and there are no
system libraries to hunt down.

Setting up an environment
=========================

Install into a fresh environment rather than your system Python. Any of
the following works.

With ``venv`` (standard library):

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install xarray-spatial

With conda or mamba:

.. code-block:: bash

   conda create -n xrspatial python=3.12
   conda activate xrspatial
   conda install -c conda-forge xarray-spatial

With `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv venv
   source .venv/bin/activate
   uv pip install xarray-spatial

Optional dependencies
=====================

The base install covers the raster compute functions plus GeoTIFF / COG
read and write. The extras below add features on top. Combine them as
needed:

.. code-block:: bash

   pip install 'xarray-spatial[plot,vector,geotiff,reproject,dask]'

.. list-table::
   :header-rows: 1
   :widths: 14 26 60

   * - Extra
     - Installs
     - Enables
   * - ``plot``
     - matplotlib
     - The ``.xrs.plot`` accessor helpers.
   * - ``vector``
     - shapely
     - The vector-to-raster paths, ``rasterize`` and ``polygonize``.
   * - ``geotiff``
     - deflate, pyproj
     - Faster DEFLATE compression (libdeflate) and full CRS support in
       the GeoTIFF writer. Without pyproj the writer only recognizes a
       small allowlist of EPSG codes.
   * - ``reproject``
     - pyproj
     - WKT / PROJ CRS resolution for reprojection.
   * - ``dask``
     - dask[array], dask-geopandas
     - Chunked, lazy, out-of-core processing on a single machine or a
       cluster. See :doc:`/reference/dask_laziness`.
   * - ``gpu``
     - cupy, cuspatial
     - The CuPy GPU backend. Needs an NVIDIA GPU and a CUDA toolkit
       matching your cupy build.
   * - ``optional``
     - awkward, geopandas, spatialpandas, rtxpy
     - Additional ``polygonize`` return types and the ray-traced
       ``gpu_rtx`` functions (rtxpy also needs cupy).
   * - ``examples``
     - matplotlib, geopandas, shapely
     - Used by the example notebooks for rendering and vector
       rasterization. datashader is no longer required.
   * - ``doc``, ``tests``
     - sphinx, pytest, ...
     - Building this documentation and running the test suite.

GPU notes
=========

``pip install 'xarray-spatial[gpu]'`` pulls in cupy and cuspatial. You
also need an NVIDIA driver and a CUDA toolkit compatible with your cupy
build; see the `CuPy install guide
<https://docs.cupy.dev/en/stable/install.html>`_ if you are unsure which
cupy package to pick.

Two GPU features have runtime dependencies that are *not* part of the
``gpu`` extra because they ship as system libraries:

* ``libnvcomp`` -- GPU batch decompression (DEFLATE, ZSTD) for the
  GeoTIFF GPU read path
* ``kvikio`` -- GPUDirect Storage, reading straight from SSD into GPU
  memory

Install both via conda from the ``rapidsai`` / ``nvidia`` channels. The
rest of the GPU path works without them.

Cloud storage
=============

``open_geotiff`` reads ``s3://``, ``gs://``, and ``az://`` URLs through
``fsspec``. Install ``fsspec`` plus the filesystem package you need:
``s3fs`` for S3, ``gcsfs`` for Google Cloud Storage, or ``adlfs`` for
Azure. Plain ``http(s)://`` URLs work with no extra packages. Remote
reads are an advanced-tier feature (see the feature matrix in the README
and :doc:`/user_guide/stability_policy`); :doc:`/reference/geotiff` has
the details.

Verifying the install
=====================

.. code-block:: bash

   python -c "import xrspatial; print(xrspatial.__version__)"

A quick functional check that exercises the compute path, with no data
download:

.. code-block:: python

   import numpy as np
   import xarray as xr
   from xrspatial import generate_terrain, hillshade

   terrain = generate_terrain(xr.DataArray(np.zeros((300, 400)), dims=['y', 'x']))
   print(hillshade(terrain).shape)
