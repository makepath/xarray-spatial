"""Tests for ``has_rtx()`` dispatch gating (Issue #2849).

``has_rtx()`` gates the GPU dispatch path in ``viewshed()`` and
``hillshade()``.  It must return ``False`` whenever CUDA/CuPy is unusable,
even if ``rtxpy`` imported.  A regression here once let ``has_rtx()`` report
``True`` on machines without a working CUDA driver because the call
parentheses on ``has_cuda_and_cupy`` were missing, so the bare function
reference was always truthy.

These are pure logic tests: they patch the two inputs to ``has_rtx`` and do
not need a GPU.
"""

from unittest.mock import patch

import xrspatial.gpu_rtx as gpu_rtx
from xrspatial.gpu_rtx import has_rtx


def test_has_rtx_false_when_no_cuda_even_if_rtx_present():
    """If CUDA/CuPy is unusable, has_rtx() is False even when RTX imported."""
    with patch.object(gpu_rtx, "has_cuda_and_cupy", return_value=False), \
            patch.object(gpu_rtx, "RTX", object()):
        assert has_rtx() is False


def test_has_rtx_false_when_rtx_missing_even_if_cuda_present():
    """If rtxpy did not import, has_rtx() is False even when CUDA is usable."""
    with patch.object(gpu_rtx, "has_cuda_and_cupy", return_value=True), \
            patch.object(gpu_rtx, "RTX", None):
        assert has_rtx() is False


def test_has_rtx_true_when_both_available():
    """Both CUDA/CuPy usable and rtxpy imported -> has_rtx() is True."""
    with patch.object(gpu_rtx, "has_cuda_and_cupy", return_value=True), \
            patch.object(gpu_rtx, "RTX", object()):
        assert has_rtx() is True


def test_has_rtx_calls_has_cuda_and_cupy():
    """has_rtx() must call has_cuda_and_cupy, not test the bare reference.

    Without the call parentheses the function object is always truthy, so
    this asserts the mock was actually invoked.
    """
    with patch.object(gpu_rtx, "has_cuda_and_cupy", return_value=False) as m, \
            patch.object(gpu_rtx, "RTX", object()):
        has_rtx()
    m.assert_called_once_with()
