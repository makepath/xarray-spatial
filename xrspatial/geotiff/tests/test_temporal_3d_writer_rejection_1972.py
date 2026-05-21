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
    "temporal",
    ['TIME', 'Time', 'TiMe', 'DATE', 'Datetime', 'DATES', 'T'],
)
def test_validate_3d_rejects_yx_temporal_case_insensitive(temporal):
    # CF allows ``'TIME'`` / ``'Time'``; the lowercase _TIME_DIM_NAMES
    # tuple must still match via case-insensitive comparison so the
    # mixed-case stack does not slip through the (y, x, *) fallback and
    # silently write a 3-band TIFF (#1972 self-review).
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


def test_validate_3d_still_accepts_recognized_band_alias_trailing_dim():
    # Recognized band aliases at the trailing position remain accepted.
    # The loose ``(y, x, *)`` fallback for arbitrary unknown trailing
    # names (``'foo'``, ``'z'``, ``'scenario'``) was removed in #2240
    # because it silently wrote those values as TIFF bands. The
    # regression coverage for the rejection lives in
    # ``test_validate_3d_non_band_trailing_dim_2240.py``.
    _validate_3d_writer_dims(('y', 'x', 'channel'))
    _validate_3d_writer_dims(('y', 'x', 'bands'))


def test_validate_3d_still_rejects_time_y_x():
    # Leading temporal dim was already rejected; the symmetrised path
    # now emits the dedicated temporal message (#1972 self-review nit 2)
    # instead of the generic "ambiguous dims" wording.
    with pytest.raises(ValueError, match="temporal leading dim"):
        _validate_3d_writer_dims(('time', 'y', 'x'))


@pytest.mark.parametrize(
    "temporal",
    ['time', 'TIME', 'Time', 't', 'T', 'date', 'datetime', 'dates'],
)
def test_validate_3d_rejects_temporal_y_x_case_insensitive(temporal):
    # Mirror the trailing-dim case-insensitive coverage for the leading
    # temporal axis (#1972 self-review nit 2).
    with pytest.raises(ValueError, match="temporal leading dim"):
        _validate_3d_writer_dims((temporal, 'y', 'x'))


def test_validate_3d_rejects_temporal_yx_alias_leading():
    # Leading-dim friendly message should fire for y/x aliases too.
    with pytest.raises(ValueError, match="temporal leading dim"):
        _validate_3d_writer_dims(('time', 'lat', 'lon'))


def test_validate_3d_still_rejects_other_ambiguous_leading():
    # The symmetric temporal message must not swallow the generic
    # ambiguous-dims path for non-temporal, non-band leading names.
    with pytest.raises(ValueError, match="ambiguous dims"):
        _validate_3d_writer_dims(('foo', 'y', 'x'))


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
