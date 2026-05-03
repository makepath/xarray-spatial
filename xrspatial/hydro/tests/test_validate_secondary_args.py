"""Tests for issue #1425: hydro public APIs validate secondary DataArray args.

Each function below previously validated only its primary raster.  Passing
a non-DataArray, a 1-D DataArray, or `None` for the secondary arg raised
a confusing AttributeError / IndexError from inside the implementation.
The fix adds `_validate_raster` on the secondary arg in the public API.
"""

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro import (
    flow_direction_d8,
    flow_direction_dinf,
    flow_direction_mfd,
    flow_path_d8,
    flow_path_dinf,
    flow_path_mfd,
    snap_pour_point_d8,
    stream_link_d8,
    stream_link_dinf,
    stream_link_mfd,
    stream_order_d8,
    stream_order_dinf,
    stream_order_mfd,
    watershed_d8,
    watershed_dinf,
    watershed_mfd,
)
from xrspatial.tests.general_checks import create_test_raster


def _elev():
    """Small bowl elevation raster used as the primary input."""
    return create_test_raster(np.array([
        [9, 9, 9, 9, 9],
        [9, 8, 7, 6, 9],
        [9, 7, 5, 4, 9],
        [9, 6, 4, 3, 9],
        [9, 9, 9, 9, 9],
    ], dtype=np.float64))


# ---------------------------------------------------------------------------
# watershed_*
# ---------------------------------------------------------------------------

class TestWatershedPourPoints:
    def test_watershed_d8_rejects_non_dataarray_pour_points(self):
        fd = flow_direction_d8(_elev())
        with pytest.raises(TypeError):
            watershed_d8(fd, pour_points=np.zeros((5, 5)))

    def test_watershed_dinf_rejects_non_dataarray_pour_points(self):
        fd = flow_direction_dinf(_elev())
        with pytest.raises(TypeError):
            watershed_dinf(fd, pour_points=np.zeros((5, 5)))

    def test_watershed_mfd_rejects_non_dataarray_pour_points(self):
        fd = flow_direction_mfd(_elev())
        with pytest.raises(TypeError):
            watershed_mfd(fd, pour_points=np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# snap_pour_point_d8
# ---------------------------------------------------------------------------

class TestSnapPourPoint:
    def test_rejects_non_dataarray_pour_points(self):
        fa = create_test_raster(np.ones((5, 5), dtype=np.float64))
        with pytest.raises(TypeError):
            snap_pour_point_d8(fa, pour_points=np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# flow_path_*
# ---------------------------------------------------------------------------

class TestFlowPathStartPoints:
    def test_flow_path_d8_rejects_non_dataarray_start_points(self):
        fd = flow_direction_d8(_elev())
        with pytest.raises(TypeError):
            flow_path_d8(fd, start_points=np.zeros((5, 5)))

    def test_flow_path_dinf_rejects_non_dataarray_start_points(self):
        fd = flow_direction_dinf(_elev())
        with pytest.raises(TypeError):
            flow_path_dinf(fd, start_points=np.zeros((5, 5)))

    def test_flow_path_mfd_rejects_non_dataarray_start_points(self):
        fd = flow_direction_mfd(_elev())
        with pytest.raises(TypeError):
            flow_path_mfd(fd, start_points=np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# stream_link_*
# ---------------------------------------------------------------------------

class TestStreamLinkFlowAccum:
    def test_stream_link_d8_rejects_non_dataarray_flow_accum(self):
        fd = flow_direction_d8(_elev())
        with pytest.raises(TypeError):
            stream_link_d8(fd, flow_accum=np.zeros((5, 5)))

    def test_stream_link_dinf_rejects_non_dataarray_flow_accum(self):
        fd = flow_direction_dinf(_elev())
        with pytest.raises(TypeError):
            stream_link_dinf(fd, flow_accum=np.zeros((5, 5)))

    def test_stream_link_mfd_rejects_non_dataarray_flow_accum(self):
        fd = flow_direction_mfd(_elev())
        with pytest.raises(TypeError):
            stream_link_mfd(fd, flow_accum=np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# stream_order_*
# ---------------------------------------------------------------------------

class TestStreamOrderFlowAccum:
    def test_stream_order_d8_rejects_non_dataarray_flow_accum(self):
        fd = flow_direction_d8(_elev())
        with pytest.raises(TypeError):
            stream_order_d8(fd, flow_accum=np.zeros((5, 5)))

    def test_stream_order_dinf_rejects_non_dataarray_flow_accum(self):
        fd = flow_direction_dinf(_elev())
        with pytest.raises(TypeError):
            stream_order_dinf(fd, flow_accum=np.zeros((5, 5)))

    def test_stream_order_mfd_rejects_non_dataarray_flow_accum(self):
        fd = flow_direction_mfd(_elev())
        with pytest.raises(TypeError):
            stream_order_mfd(fd, flow_accum=np.zeros((5, 5)))
