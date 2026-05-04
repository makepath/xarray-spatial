"""Tests for issue #1429: hydro cellsize validation.

Functions that divide by `get_dataarray_resolution()` previously
silently produced inf/-inf/zero output for degenerate input coords
(1-D coordinates resolving to zero spacing, NaN coords, etc.).
"""

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro import (
    flow_direction_d8,
    flow_direction_dinf,
    flow_direction_mfd,
    flow_length_d8,
    flow_length_dinf,
    flow_length_mfd,
    twi_d8,
)


def _zero_cellsize_raster(shape=(5, 5)):
    """Build a DataArray whose coords resolve to zero cellsize.

    A length-1 coordinate axis means there is no diff to compute.
    """
    data = np.ones(shape, dtype=np.float64)
    return xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={
            'y': np.zeros(shape[0], dtype=np.float64),
            'x': np.zeros(shape[1], dtype=np.float64),
        },
        attrs={'res': (0.0, 0.0)},
    )


# ---------------------------------------------------------------------------
# flow_direction_*
# ---------------------------------------------------------------------------

class TestFlowDirectionCellsize:
    def test_d8_rejects_zero_cellsize(self):
        with pytest.raises(ValueError, match="cellsize"):
            flow_direction_d8(_zero_cellsize_raster())

    def test_dinf_rejects_zero_cellsize(self):
        with pytest.raises(ValueError, match="cellsize"):
            flow_direction_dinf(_zero_cellsize_raster())

    def test_mfd_rejects_zero_cellsize(self):
        with pytest.raises(ValueError, match="cellsize"):
            flow_direction_mfd(_zero_cellsize_raster())


# ---------------------------------------------------------------------------
# flow_length_*
# ---------------------------------------------------------------------------

class TestFlowLengthCellsize:
    def test_d8_rejects_zero_cellsize(self):
        fd = xr.DataArray(
            np.full((5, 5), 1.0, dtype=np.float64),
            dims=('y', 'x'),
            coords={
                'y': np.zeros(5, dtype=np.float64),
                'x': np.zeros(5, dtype=np.float64),
            },
            attrs={'res': (0.0, 0.0)},
        )
        with pytest.raises(ValueError, match="cellsize"):
            flow_length_d8(fd)

    def test_dinf_rejects_zero_cellsize(self):
        fd = xr.DataArray(
            np.full((5, 5), 0.0, dtype=np.float64),
            dims=('y', 'x'),
            coords={
                'y': np.zeros(5, dtype=np.float64),
                'x': np.zeros(5, dtype=np.float64),
            },
            attrs={'res': (0.0, 0.0)},
        )
        with pytest.raises(ValueError, match="cellsize"):
            flow_length_dinf(fd)

    def test_mfd_rejects_zero_cellsize(self):
        fractions = xr.DataArray(
            np.full((8, 5, 5), 0.125, dtype=np.float64),
            dims=('neighbor', 'y', 'x'),
            coords={
                'neighbor': ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE'],
                'y': np.zeros(5, dtype=np.float64),
                'x': np.zeros(5, dtype=np.float64),
            },
            attrs={'res': (0.0, 0.0)},
        )
        with pytest.raises(ValueError, match="cellsize"):
            flow_length_mfd(fractions)


# ---------------------------------------------------------------------------
# twi_d8
# ---------------------------------------------------------------------------

class TestTwiCellsize:
    def test_rejects_zero_cellsize(self):
        fa = _zero_cellsize_raster()
        slope = _zero_cellsize_raster()
        with pytest.raises(ValueError, match="cellsize"):
            twi_d8(fa, slope)
