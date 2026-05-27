"""Tests for shapely being an optional dependency (the `vector` extra).

shapely backs the vector-to-raster paths (rasterize, polygonize) but is
imported lazily, so `import xrspatial` works without it. See issue #2496.

These tests run whether or not shapely is installed: they either spawn a
fresh interpreter with shapely blocked, or block it in ``sys.modules`` and
reset the cached module in ``xrspatial.rasterize``.
"""
import sys

import numpy as np
import pytest
import xarray as xr


def test_import_xrspatial_without_shapely():
    """`import xrspatial` and the compute modules work with no shapely.

    Runs in a subprocess so the import happens against a clean module cache
    with shapely blocked.
    """
    import subprocess
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        sys.modules['shapely'] = None

        import xrspatial  # noqa: F401
        import xrspatial.focal  # noqa: F401
        import xrspatial.rasterize  # noqa: F401
        import xrspatial.polygonize  # noqa: F401

        # shapely must not have been imported as a side effect.
        if 'shapely' in sys.modules and sys.modules['shapely'] is not None:
            raise SystemExit('shapely was imported on import xrspatial')

        try:
            import shapely  # noqa: F401
        except ImportError:
            pass
        else:
            raise SystemExit('shapely was unexpectedly importable')
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_require_shapely_message(monkeypatch):
    """The helper points users at the `vector` extra when shapely is gone."""
    import importlib
    rasterize_mod = importlib.import_module('xrspatial.rasterize')

    monkeypatch.setattr(rasterize_mod, '_shapely', None)
    monkeypatch.setitem(sys.modules, 'shapely', None)
    with pytest.raises(ImportError, match=r"xarray-spatial\[vector\]"):
        rasterize_mod._require_shapely()


def test_rasterize_without_shapely_raises(monkeypatch):
    """`rasterize()` raises the friendly error up front when shapely is absent."""
    import importlib
    rasterize_mod = importlib.import_module('xrspatial.rasterize')

    monkeypatch.setattr(rasterize_mod, '_shapely', None)
    monkeypatch.setitem(sys.modules, 'shapely', None)

    template = xr.DataArray(
        np.zeros((4, 4)),
        dims=['y', 'x'],
        coords={'x': np.arange(4.0), 'y': np.arange(4.0)},
    )
    with pytest.raises(ImportError, match=r"xarray-spatial\[vector\]"):
        rasterize_mod.rasterize([], like=template)
