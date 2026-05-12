"""Regression test for #1708: ``read_to_array`` no longer leaks into
``xrspatial.geotiff``'s public namespace.

Before the fix, ``xrspatial/geotiff/__init__.py`` did

    from ._reader import UnsafeURLError, read_to_array

so ``from xrspatial.geotiff import read_to_array`` worked even though
``read_to_array`` was not in ``__all__`` and not documented as public.
The api-consistency sweep on 2026-05-12 flagged this as an orphan API
(Cat 5): the name is reachable from the public namespace and tests
relied on it, but the maintainer had not committed to it as a
supported entry point. A future cleanup that removed the top-level
import would have broken users with no deprecation path.

The fix binds the helper under a leading-underscore name in
``__init__.py`` (``_read_to_array``) and updates the single internal
test that depended on the top-level import to go through
``xrspatial.geotiff._reader`` directly.

This module pins the public-namespace contract so a future maintainer
who re-adds a bare ``from ._reader import read_to_array`` to
``__init__.py`` fails CI before merging.
"""
from __future__ import annotations

import pytest

import xrspatial.geotiff as g


def test_read_to_array_not_in_module_namespace():
    """``read_to_array`` is internal and must not be a public attribute
    of ``xrspatial.geotiff``. Tests and internal callers go through
    ``xrspatial.geotiff._reader`` directly."""
    assert not hasattr(g, 'read_to_array'), (
        "read_to_array leaked into xrspatial.geotiff's public namespace. "
        "It should be bound as _read_to_array (underscore-prefix) so it "
        "does not appear in `from xrspatial.geotiff import *` or tab "
        "completion. See issue #1708."
    )


def test_read_to_array_not_in_all():
    """``__all__`` does not advertise ``read_to_array``. (Pre-existing
    state, pinned so a future change can't quietly promote it without
    also adding it to the module-level Public API docstring.)"""
    assert 'read_to_array' not in g.__all__


def test_read_to_array_still_importable_from_reader_submodule():
    """The function itself is still reachable via the canonical private
    path so tests and internal code keep working."""
    from xrspatial.geotiff._reader import read_to_array

    assert callable(read_to_array)


def test_import_from_top_level_now_fails():
    """Direct top-level import is the regression we are guarding
    against. Before #1708 this succeeded; afterwards it must raise
    ImportError."""
    with pytest.raises(ImportError):
        from xrspatial.geotiff import read_to_array  # noqa: F401
