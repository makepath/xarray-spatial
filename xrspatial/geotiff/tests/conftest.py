"""Shared fixtures for geotiff tests.

The TIFF builder, markers, and capability probes now live under
``_helpers/``. This module re-exports them so legacy imports such as
``from .conftest import make_minimal_tiff`` or
``from xrspatial.geotiff.tests.conftest import requires_gpu`` keep
working. PR 11 of the consolidation epic (#2390) drops the
``pytest_collection_modifyitems`` socketserver hack below; until then
it stays in place to keep the existing loopback-skip behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from ._helpers.markers import (
    _HAS_LOOPBACK,
    gpu_available,
    loopback_available,
    requires_gpu,
    requires_integration,
    requires_loopback,
)
from ._helpers.tiff_builders import make_minimal_tiff

__all__ = [
    "gpu_available",
    "loopback_available",
    "make_minimal_tiff",
    "requires_gpu",
    "requires_integration",
    "requires_loopback",
]


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that stand up a loopback HTTP server when the
    sandbox denies socket bind.

    A test needs loopback iff its function body or any fixture in its
    closure references ``socketserver.TCPServer(`` or invokes the
    file-local ``_serve(`` helper. This is finer-grained than skipping
    every test in a module that imports ``socketserver``: mixed files
    (e.g. ``test_miniswhite_backend_parity_1797.py`` has both HTTP and
    a local-file GPU test) keep their non-HTTP coverage in restricted
    sandboxes.
    """
    if _HAS_LOOPBACK:
        return

    import inspect

    def _source_of(obj) -> str:
        try:
            return inspect.getsource(obj)
        except (OSError, TypeError):
            return ''

    def _references_loopback(src: str) -> bool:
        return 'socketserver.TCPServer(' in src or '_serve(' in src

    skip_marker = pytest.mark.skip(
        reason="loopback bind unavailable in this environment"
    )
    for item in items:
        needs_skip = False

        func = getattr(item, 'function', None)
        if func is not None and _references_loopback(_source_of(func)):
            needs_skip = True

        if not needs_skip:
            fixtureinfo = getattr(item, '_fixtureinfo', None)
            if fixtureinfo is not None:
                name2defs = getattr(fixtureinfo, 'name2fixturedefs', {})
                for fname in getattr(fixtureinfo, 'names_closure', ()):
                    for fdef in name2defs.get(fname, ()):
                        ffunc = getattr(fdef, 'func', None)
                        if ffunc is not None and _references_loopback(
                            _source_of(ffunc)
                        ):
                            needs_skip = True
                            break
                    if needs_skip:
                        break

        if needs_skip:
            item.add_marker(skip_marker)


@pytest.fixture
def simple_float32_tiff():
    """4x4 float32 stripped TIFF with sequential values."""
    return make_minimal_tiff(4, 4, np.dtype('float32'))


@pytest.fixture
def simple_uint16_tiff():
    """4x4 uint16 stripped TIFF."""
    return make_minimal_tiff(4, 4, np.dtype('uint16'))


@pytest.fixture
def geo_tiff_data():
    """4x4 float32 TIFF with geo transform and EPSG 4326."""
    return make_minimal_tiff(
        4, 4, np.dtype('float32'),
        geo_transform=(-120.0, 45.0, 0.001, -0.001),
        epsg=4326,
    )


@pytest.fixture
def tiled_tiff_data():
    """8x8 float32 tiled TIFF with 4x4 tiles."""
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    return make_minimal_tiff(
        8, 8, np.dtype('float32'),
        pixel_data=data,
        tiled=True,
        tile_size=4,
    )
