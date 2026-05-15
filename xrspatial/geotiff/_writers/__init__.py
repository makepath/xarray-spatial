"""Writer entry points for the geotiff module.

Step 10 of issue #1813. Holds ``to_geotiff`` (and its eager-path
helpers), ``write_geotiff_gpu``, and ``write_vrt`` in sibling modules.
The package ``__init__`` stays empty so nothing leaks into
``xrspatial.geotiff`` through implicit re-exports.
"""
