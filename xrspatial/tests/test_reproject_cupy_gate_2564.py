"""Regression test for #2564.

The reproject test modules previously used an import-only ``HAS_CUPY``
gate (``try: import cupy; HAS_CUPY = True``). On hosts where ``cupy``
imports but the CUDA driver is missing or too old, every GPU-only test
erred at allocation time with ``cudaErrorInsufficientDriver`` instead of
skipping. The fix is to bind ``HAS_CUPY`` to ``has_cuda_and_cupy()``,
which probes both ``cupy`` and a usable CUDA device.

This test pins that the gate tracks the runtime probe by monkey-patching
``xrspatial.utils.has_cuda_and_cupy`` to return ``False`` and reimporting
the reproject test modules; the gate must follow.
"""
from __future__ import annotations

import importlib
import sys

import pytest

from xrspatial import utils as xu


@pytest.mark.parametrize(
    "module_name",
    [
        "xrspatial.tests.test_reproject",
        "xrspatial.tests.test_reproject_coverage_2026_05_27",
    ],
)
def test_has_cupy_tracks_runtime_probe(module_name, monkeypatch):
    """``HAS_CUPY`` must reflect ``has_cuda_and_cupy()``, not just import."""
    monkeypatch.setattr(xu, "has_cuda_and_cupy", lambda: False)

    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    assert module.HAS_CUPY is False, (
        f"{module_name}.HAS_CUPY should track the runtime CUDA probe, "
        "not just whether ``import cupy`` succeeds."
    )
