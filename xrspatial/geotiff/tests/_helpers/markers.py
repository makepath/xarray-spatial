"""Pytest markers and capability probes for the GeoTIFF test suite.

Relocated from ``conftest.py``. The marker names are unchanged
(``requires_gpu``, ``requires_loopback``, ``requires_integration``) so
test files that already import them via ``from .conftest import ...``
keep working through the conftest re-export.
"""
from __future__ import annotations

import importlib.util
import os
import socket

import pytest


def gpu_available() -> bool:
    """True iff cupy imports AND a CUDA device is actually usable.

    Some sandboxes ship cupy without a working CUDA runtime. A bare
    ``import cupy`` succeeds there but every device call fails, so test
    files that gate on the import alone show false failures.
    """
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


def loopback_available() -> bool:
    """True iff a loopback TCP bind succeeds on this host.

    Some sandboxed environments deny ``bind(('127.0.0.1', 0))``. HTTP
    tests that stand up a loopback server should skip rather than error
    in that case.
    """
    try:
        s = socket.socket()
        try:
            s.bind(('127.0.0.1', 0))
        finally:
            s.close()
    except OSError:
        return False
    return True


_HAS_GPU = gpu_available()
_HAS_LOOPBACK = loopback_available()

requires_gpu = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required"
)
requires_loopback = pytest.mark.skipif(
    not _HAS_LOOPBACK, reason="loopback bind unavailable in this environment"
)

_RUN_INTEGRATION = os.environ.get("XRSPATIAL_RUN_INTEGRATION", "") not in (
    "", "0", "false", "False"
)
requires_integration = pytest.mark.skipif(
    not _RUN_INTEGRATION,
    reason="integration test; set XRSPATIAL_RUN_INTEGRATION=1 to run locally",
)
