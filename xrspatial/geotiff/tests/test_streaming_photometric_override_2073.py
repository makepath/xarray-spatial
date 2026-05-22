"""Regression tests for issue #2073.

The eager writer rejects an ``extra_tags`` entry that overrides
``TAG_PHOTOMETRIC`` across the MinIsWhite boundary for a single-band
raster because the reader unconditionally inverts MinIsWhite single-band
data and the writer must pre-invert pixels to keep the round-trip honest.
The streaming dask path now shares the same guard via
``_reject_disagreeing_photometric_override`` in ``_writer.py``.

Three pins:

* the MinIsWhite-crossing single-band override is rejected;
* the non-MinIsWhite-crossing override still round-trips;
* multi-band rasters do not trigger the guard (the writer never
  pre-inverts there).
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


def test_streaming_extra_tags_miniswhite_override_multiband_not_rejected_2073(
    tmp_path,
):
    """The guard fires only on single-band rasters.

    Multi-band rasters do not pre-invert MinIsWhite, so a
    ``TAG_PHOTOMETRIC`` override that crosses the MinIsWhite boundary
    is not the kind of corruption the guard exists to prevent. Pins
    the ``samples == 1`` gate inside
    ``_reject_disagreeing_photometric_override``: a regression that
    dropped or flipped the gate would surface as a spurious
    ``ValueError`` here.

    Whether a 3-band raster tagged MinIsWhite is semantically useful
    is a separate concern; this test only locks in the guard's scope.
    """
    src = np.zeros((2, 2, 3), dtype=np.uint8)
    src[..., 0] = 10
    src[..., 1] = 20
    src[..., 2] = 30
    arr = xr.DataArray(
        da.from_array(src, chunks=(2, 2, 3)),
        dims=('y', 'x', 'band'),
        coords={'y': [1.0, 0.0], 'x': [0.0, 1.0]},
    )
    arr.attrs['extra_tags'] = [(TAG_PHOTOMETRIC, TYPE_SHORT, 1, 0)]

    out = tmp_path / 'tmp_2073_streaming_miniswhite_multiband.tif'
    # Must not raise: the writer does not pre-invert multi-band data,
    # so the override is not in the "corruption that the guard exists
    # to prevent" set. If it raises for an unrelated reason
    # (e.g. RGB-requires-3-bands check elsewhere), let the test
    # surface that as a real failure rather than swallowing it.
    to_geotiff(arr, str(out))
    assert os.path.exists(out)
