"""Tests for issue #3292: the cupy integer polygonize backend must raise
RuntimeError when the cumulative region count would exceed the uint32
ceiling, matching the CPU ``_calculate_regions`` guard, instead of letting
the uint32 addition wrap region IDs (first to the masked-out sentinel 0).
"""

import numpy as np
import pytest
import xarray as xr

from xrspatial.polygonize import polygonize

from .general_checks import cuda_and_cupy_available


@cuda_and_cupy_available
def test_polygonize_cupy_too_many_regions_raises_3292(monkeypatch):
    import cupy
    import cupyx.scipy.ndimage

    real_label = cupyx.scipy.ndimage.label

    # Inflate the per-value feature count so the cumulative region count
    # crosses the uint32 ceiling on the second unique value, without
    # allocating a billions-of-pixels raster.  2**31 fits in cp_label's
    # int32 label dtype, and two values sum to 2**32 > uint32 max.
    def fake_label(mask, structure=None):
        labeled, _ = real_label(mask, structure=structure)
        return labeled, 2 ** 31

    monkeypatch.setattr(cupyx.scipy.ndimage, "label", fake_label)

    data = np.array([[1, 2], [1, 2]], dtype=np.int32)
    raster = xr.DataArray(cupy.asarray(data))

    with pytest.raises(RuntimeError, match="too many polygons"):
        polygonize(raster)


@cuda_and_cupy_available
def test_polygonize_cupy_guard_does_not_trip_normally_3292():
    import cupy

    data = np.array([[1, 2, 2], [1, 1, 3], [4, 4, 3]], dtype=np.int32)
    column, polygon_points = polygonize(xr.DataArray(cupy.asarray(data)))
    assert sorted(column) == [1, 2, 3, 4]
    assert len(polygon_points) == 4
