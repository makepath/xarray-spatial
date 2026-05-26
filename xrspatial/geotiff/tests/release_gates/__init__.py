"""Release-gate test registry.

Every ``@pytest.mark.release_gate`` test for the GeoTIFF surface lives
under this package. The release process's audit point is one module:
``test_stable_features.py`` -- run ``pytest xrspatial/geotiff/tests/
release_gates/ -m release_gate`` to execute the union of every promise
the release notes are allowed to make.
"""
