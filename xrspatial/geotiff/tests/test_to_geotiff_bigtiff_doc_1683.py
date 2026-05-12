"""Regression test for #1683: ``to_geotiff`` Parameters docstring omits
``bigtiff``.

The api-consistency sweep on 2026-05-12 flagged that ``to_geotiff``
accepts a ``bigtiff`` kwarg (signature at ``xrspatial/geotiff/__init__.py``)
but the Parameters block of the docstring jumps from
``overview_resampling`` directly to ``gpu``. ``write_geotiff_gpu``
documents the same kwarg correctly, so users learning the API from
``to_geotiff(...)`` could not tell the option existed.

This module pins the docstring entry against future drift.
"""
from __future__ import annotations

import inspect
import re

from xrspatial.geotiff import to_geotiff, write_geotiff_gpu


def _documented_params(fn) -> list[str]:
    """Return the parameter names listed under the docstring's
    ``Parameters`` section, in document order.
    """
    doc = inspect.getdoc(fn) or ""
    documented: list[str] = []
    in_params = False
    for line in doc.splitlines():
        if re.match(r"^Parameters\s*$", line.strip()):
            in_params = True
            continue
        if in_params and re.match(r"^[A-Z][a-z]+\s*$", line.strip()):
            # Hit the next docstring section heading (Returns, Notes, ...).
            in_params = False
        if in_params:
            m = re.match(r"^(\S+(?:,\s*\S+)*)\s*:\s*", line)
            if m:
                for name in m.group(1).split(","):
                    documented.append(name.strip())
    return documented


def test_to_geotiff_bigtiff_documented():
    """``bigtiff`` is in the signature and must be in the docstring too."""
    params = list(inspect.signature(to_geotiff).parameters)
    assert "bigtiff" in params, (
        "to_geotiff signature lost the bigtiff kwarg")
    documented = _documented_params(to_geotiff)
    assert "bigtiff" in documented, (
        f"to_geotiff docstring is missing the bigtiff parameter "
        f"description (documented params: {documented})"
    )


def test_to_geotiff_parameters_match_signature():
    """Every public kwarg of ``to_geotiff`` is documented."""
    params = [p for p in inspect.signature(to_geotiff).parameters]
    documented = _documented_params(to_geotiff)
    missing = [p for p in params if p not in documented]
    assert not missing, (
        f"to_geotiff docstring is missing parameter descriptions for "
        f"{missing}; documented params were {documented}"
    )


def test_write_geotiff_gpu_parameters_match_signature():
    """Sibling writer keeps its full parameter set documented too."""
    params = [p for p in inspect.signature(write_geotiff_gpu).parameters]
    documented = _documented_params(write_geotiff_gpu)
    missing = [p for p in params if p not in documented]
    assert not missing, (
        f"write_geotiff_gpu docstring is missing parameter "
        f"descriptions for {missing}; documented params were {documented}"
    )
