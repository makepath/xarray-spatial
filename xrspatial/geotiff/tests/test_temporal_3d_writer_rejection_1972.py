"""Refuse ``(y, x, <time>)`` 3D writer inputs (#1972).

The existing ``_validate_3d_writer_dims`` (issue #1812) rejected the
``(time, y, x)`` case, but the symmetric ``(y, x, time)`` slipped
through the ``(y, x, *)`` band-position fallback and was silently
written as a 3-band TIFF. This locks the temporal-trailing case down
across the eager, GPU (import-time-check only), and `.vrt` writer
entry points.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._validation import _validate_3d_writer_dims


@pytest.mark.parametrize(
    "temporal",
    ['time', 't', 'date', 'datetime', 'times', 'dates'],
)
def test_validate_3d_rejects_yx_temporal(temporal):
    with pytest.raises(ValueError, match="temporal trailing dim"):
        _validate_3d_writer_dims(('y', 'x', temporal))


@pytest.mark.parametrize(
    "yx",
    [('y', 'x'), ('lat', 'lon'), ('latitude', 'longitude'), ('row', 'col')],
)
def test_validate_3d_rejects_yx_aliases_with_temporal(yx):
    with pytest.raises(ValueError, match="temporal trailing dim"):
        _validate_3d_writer_dims((yx[0], yx[1], 'time'))


def test_validate_3d_still_accepts_yx_band():
    _validate_3d_writer_dims(('y', 'x', 'band'))
    _validate_3d_writer_dims(('band', 'y', 'x'))


def test_validate_3d_still_accepts_unknown_trailing_dim():
    # The (y, x, *) fallback for raw numpy callers stays in place for
    # genuinely unknown dim names; only known temporal names trip.
    _validate_3d_writer_dims(('y', 'x', 'channel'))
    _validate_3d_writer_dims(('y', 'x', 'foo'))


def test_validate_3d_still_rejects_time_y_x():
    # Leading temporal dim was already rejected (asymmetry that #1972
    # closes); keep the regression test for it.
    with pytest.raises(ValueError, match="ambiguous dims"):
        _validate_3d_writer_dims(('time', 'y', 'x'))


def test_to_geotiff_rejects_yxtime_stack():
    da = xr.DataArray(
        np.zeros((4, 4, 3), dtype=np.float32),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0),
                'time': np.arange(3)},
        dims=('y', 'x', 'time'),
    )
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="temporal trailing dim"):
        to_geotiff(da, buf)


def test_error_message_suggests_isel_and_band_rename():
    da = xr.DataArray(
        np.zeros((4, 4, 3), dtype=np.float32),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0),
                'time': np.arange(3)},
        dims=('y', 'x', 'time'),
    )
    buf = io.BytesIO()
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, buf)
    msg = str(excinfo.value)
    assert "isel(time=0)" in msg
    assert "band" in msg.lower()
