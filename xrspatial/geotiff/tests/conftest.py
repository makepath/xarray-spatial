"""Shared fixtures for geotiff tests.

The TIFF builder, markers, and capability probes live under
``_helpers/``. This module re-exports them so legacy imports such as
``from .conftest import make_minimal_tiff`` or
``from xrspatial.geotiff.tests.conftest import requires_gpu`` keep
working without touching every test file. New tests should import the
symbols directly from ``_helpers`` instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from ._helpers.markers import (gpu_available, loopback_available, requires_gpu,
                               requires_integration, requires_loopback)
from ._helpers.tiff_builders import make_minimal_tiff

__all__ = [
    "gpu_available",
    "loopback_available",
    "make_minimal_tiff",
    "requires_gpu",
    "requires_integration",
    "requires_loopback",
]


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
