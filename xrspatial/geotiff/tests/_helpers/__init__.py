"""Shared helpers for the GeoTIFF test suite.

Centralises the TIFF-builder factory, byte-surgery helpers, and pytest
marker definitions that used to live in ``conftest.py`` and at the top
level of ``xrspatial/geotiff/tests/``. The ``conftest.py`` re-exports
``make_minimal_tiff`` and the marker names so existing
``from .conftest import ...`` test imports keep working.
"""
