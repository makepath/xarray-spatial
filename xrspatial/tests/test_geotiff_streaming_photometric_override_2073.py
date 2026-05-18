"""Regression tests for issue #2073.

The eager writer rejects an ``extra_tags`` entry that overrides
``TAG_PHOTOMETRIC`` across the MinIsWhite boundary for a single-band
raster (``xrspatial/geotiff/_writer.py:1600-1617``) because the reader
unconditionally inverts MinIsWhite single-band data and the writer must
pre-invert pixels to keep the round-trip honest.

The streaming dask path checked the ``photometric`` kwarg but accepted
the ``extra_tags`` override without the same guard, so dask writers
silently produced inverted on-disk values. These tests pin the guard on
the streaming path and confirm the non-MinIsWhite override case still
round-trips.
"""
from __future__ import annotations

import os

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


TAG_PHOTOMETRIC = 262
TYPE_SHORT = 3


def test_streaming_extra_tags_miniswhite_override_rejected_2073(tmp_path):
    """Dask write with extra_tags forcing photometric=0 must raise."""
    arr = xr.DataArray(
        da.from_array(
            np.array([[10, 20], [30, 40]], dtype=np.uint8),
            chunks=(1, 2),
        ),
    )
    arr.attrs['extra_tags'] = [(TAG_PHOTOMETRIC, TYPE_SHORT, 1, 0)]

    out = tmp_path / 'tmp_2073_streaming_miniswhite.tif'
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(arr, str(out))

    msg = str(excinfo.value)
    assert 'extra_tags' in msg
    assert 'photometric' in msg.lower() or 'MinIsWhite' in msg


def test_streaming_extra_tags_minisblack_override_roundtrips_2073(tmp_path):
    """The valid (non-MinIsWhite-crossing) override should still work."""
    src = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    arr = xr.DataArray(
        da.from_array(src, chunks=(1, 2)),
        dims=('y', 'x'),
        coords={'y': [1.0, 0.0], 'x': [0.0, 1.0]},
    )
    # photometric=1 (MinIsBlack) matches what the writer picks for a
    # single-band raster anyway: no pre-inversion needed, so the guard
    # must not fire.
    arr.attrs['extra_tags'] = [(TAG_PHOTOMETRIC, TYPE_SHORT, 1, 1)]

    out = tmp_path / 'tmp_2073_streaming_minisblack.tif'
    to_geotiff(arr, str(out))
    assert os.path.exists(out)

    back = open_geotiff(str(out))
    np.testing.assert_array_equal(np.asarray(back.values), src)
