"""Regression tests for issue #2075.

``to_geotiff`` used to accept arrays with a zero-height or zero-width
spatial dim and write a TIFF whose IFD claimed shape ``(0, N)`` or
``(N, 0)``. The reader then rejected the file with the generic
"Invalid image dimensions" message that never named the writer as the
source.

The fix raises ``ValueError`` at the write entry point. The failure
happens before any bytes hit disk, and the message names the offending
dimension so callers know which axis went empty (a clip / window
operation is the common cause).
"""
from __future__ import annotations

import dask.array as dsk
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff


_EMPTY_SHAPES = [
    pytest.param((0, 5), id="zero-height"),
    pytest.param((5, 0), id="zero-width"),
    pytest.param((0, 0), id="both-zero"),
]


@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
def test_to_geotiff_rejects_empty_numpy(tmp_path, shape):
    h, w = shape
    da = xr.DataArray(
        np.zeros(shape, dtype=np.float32),
        dims=("y", "x"),
    )
    out = tmp_path / f"tmp_2075_empty_{h}x{w}.tif"
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, str(out))
    msg = str(excinfo.value)
    # The message must name the writer (so callers see the source) and
    # mention which axis is zero. Accept either height/width by name or
    # the literal shape so we don't pin the exact wording.
    assert "to_geotiff" in msg or "empty" in msg.lower() or "0" in msg
    if h == 0:
        assert "height" in msg.lower() or f"({h}, {w})" in msg
    if w == 0:
        assert "width" in msg.lower() or f"({h}, {w})" in msg
    # Nothing should have been written.
    assert not out.exists()


def test_to_geotiff_rejects_empty_dask(tmp_path):
    # One dask variant is enough to exercise the streaming entry point.
    shape = (0, 5)
    da = xr.DataArray(
        dsk.zeros(shape, dtype=np.float32, chunks=shape if 0 not in shape
                  else (1, 1)),
        dims=("y", "x"),
    )
    out = tmp_path / "tmp_2075_empty_dask_0x5.tif"
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, str(out))
    msg = str(excinfo.value).lower()
    assert "height" in msg or "empty" in msg or "(0, 5)" in msg
    assert not out.exists()
