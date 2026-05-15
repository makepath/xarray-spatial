"""Per-backend implementations for the geotiff read entry points.

Step 6 of issue #1813 creates this subpackage and adds
``_gpu_helpers.py``. Later steps add ``gpu.py``, ``dask.py``, and
``vrt.py`` for the backend entry-point bodies. The package
``__init__`` stays empty so nothing leaks into ``xrspatial.geotiff``
through implicit re-exports.
"""
