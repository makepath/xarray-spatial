"""Refuse ``(y, x, <non-band>)`` 3D writer inputs (#2240).

``_validate_3d_writer_dims`` (introduced in #1812 and extended for the
temporal case in #1972) used to accept any ``(y_alias, x_alias, *)``
DataArray dim tuple whose trailing dim was not a recognized temporal
name. That meant DataArrays with dims like ``('y', 'x', 'z')``,
``('y', 'x', 'level')``, or ``('lat', 'lon', 'scenario')`` slipped
through and were silently written as multiband TIFFs with the trailing
axis stuffed into the band slot. #2240 closes that escape hatch.

The intent of the original fallback was raw-ndarray callers building
band-last arrays without dim metadata. Those callers never reach this
validator (it is gated on ``isinstance(data, xr.DataArray)`` in every
writer entry point), so the fallback's only effect was on DataArray
inputs -- and there it was silent data corruption.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._validation import _validate_3d_writer_dims

# --- Validator-level coverage ------------------------------------------------


@pytest.mark.parametrize(
    "trailing",
    ['z', 'level', 'scenario', 'depth', 'member', 'realization',
     'foo', 'bar', 'baz'],
)
def test_validate_3d_rejects_yx_non_band_trailing(trailing):
    """``(y, x, <non-band, non-temporal>)`` now raises with a clear message."""
    with pytest.raises(ValueError, match="non-band trailing dim"):
        _validate_3d_writer_dims(('y', 'x', trailing))


@pytest.mark.parametrize(
    "yx",
    [('y', 'x'), ('lat', 'lon'), ('latitude', 'longitude'), ('row', 'col')],
)
@pytest.mark.parametrize(
    "trailing",
    ['z', 'level', 'scenario'],
)
def test_validate_3d_rejects_yx_aliases_with_non_band_trailing(yx, trailing):
    """Non-band trailing dim is rejected for every recognized y/x alias."""
    with pytest.raises(ValueError, match="non-band trailing dim"):
        _validate_3d_writer_dims((yx[0], yx[1], trailing))


def test_validate_3d_still_accepts_band_alias_trailing():
    """Recognized band aliases at the trailing position still succeed."""
    _validate_3d_writer_dims(('y', 'x', 'band'))
    _validate_3d_writer_dims(('y', 'x', 'bands'))
    _validate_3d_writer_dims(('y', 'x', 'channel'))


def test_validate_3d_still_accepts_band_alias_leading():
    """``(band, y, x)`` and its aliases still succeed."""
    _validate_3d_writer_dims(('band', 'y', 'x'))
    _validate_3d_writer_dims(('bands', 'y', 'x'))
    _validate_3d_writer_dims(('channel', 'y', 'x'))


def test_validate_3d_still_routes_temporal_to_temporal_message():
    """Temporal trailing dims still take the dedicated temporal error path.

    The #1972 message gives more specific remediation (``isel`` /
    ``mean`` along the time axis) than the #2240 generic non-band
    message, so the temporal-name branch must fire first.
    """
    with pytest.raises(ValueError, match="temporal trailing dim"):
        _validate_3d_writer_dims(('y', 'x', 'time'))
    with pytest.raises(ValueError, match="temporal trailing dim"):
        _validate_3d_writer_dims(('lat', 'lon', 'date'))


def test_validate_3d_still_rejects_other_ambiguous_leading():
    """Generic ambiguous-dim message still fires for non-y/x leading dims."""
    with pytest.raises(ValueError, match="ambiguous dims"):
        _validate_3d_writer_dims(('foo', 'y', 'x'))
    with pytest.raises(ValueError, match="ambiguous dims"):
        _validate_3d_writer_dims(('scenario', 'y', 'x'))


def test_validate_3d_2d_dims_unchanged():
    """2D dim tuples are still pass-through (validator only runs on 3D)."""
    _validate_3d_writer_dims(('y', 'x'))
    _validate_3d_writer_dims(('lat', 'lon'))


# --- End-to-end writer coverage ----------------------------------------------

def test_to_geotiff_rejects_yxz_dataarray():
    """End-to-end: ``(y, x, z)`` DataArray writes are rejected."""
    da = xr.DataArray(
        np.zeros((4, 4, 3), dtype=np.float32),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0),
                'z': np.arange(3)},
        dims=('y', 'x', 'z'),
    )
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="non-band trailing dim"):
        to_geotiff(da, buf)


def test_to_geotiff_rejects_lat_lon_scenario_dataarray():
    """End-to-end: ``(lat, lon, scenario)`` is rejected on the writer entry."""
    da = xr.DataArray(
        np.zeros((4, 4, 3), dtype=np.float32),
        coords={'lat': np.arange(4.0), 'lon': np.arange(4.0),
                'scenario': np.arange(3)},
        dims=('lat', 'lon', 'scenario'),
    )
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="non-band trailing dim"):
        to_geotiff(da, buf)


def test_error_message_is_actionable():
    """The error names the offending dim and points at fixes."""
    da = xr.DataArray(
        np.zeros((4, 4, 3), dtype=np.float32),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0),
                'scenario': np.arange(3)},
        dims=('y', 'x', 'scenario'),
    )
    buf = io.BytesIO()
    with pytest.raises(ValueError) as excinfo:
        to_geotiff(da, buf)
    msg = str(excinfo.value)
    # Names the offending dim
    assert "'scenario'" in msg
    # Mentions accepted band aliases
    assert "band" in msg
    # Points at concrete remediations
    assert "isel(scenario=0)" in msg or "isel" in msg
    assert "raw ndarray" in msg.lower() or "ndarray" in msg.lower()
    # References the new issue
    assert "#2240" in msg


def test_to_geotiff_still_accepts_yx_band_dataarray(tmp_path):
    """``(y, x, band)`` DataArrays still round-trip cleanly."""
    arr = np.empty((4, 5, 3), dtype=np.uint8)
    for k in range(3):
        arr[:, :, k] = k + 1
    da = xr.DataArray(arr, dims=('y', 'x', 'band'),
                      attrs={'crs': 'EPSG:4326'})
    out = tmp_path / 'tmp_2240_yx_band.tif'
    to_geotiff(da, str(out), crs=4326)
    rt = open_geotiff(str(out))
    assert rt.shape == (4, 5, 3)
    for k in range(3):
        assert int(rt.values[:, :, k].sum()) == (k + 1) * 20


def test_to_geotiff_still_accepts_band_yx_dataarray(tmp_path):
    """``(band, y, x)`` DataArrays still round-trip cleanly."""
    arr = np.empty((3, 4, 5), dtype=np.uint8)
    for k in range(3):
        arr[k] = k + 1
    da = xr.DataArray(arr, dims=('band', 'y', 'x'),
                      attrs={'crs': 'EPSG:4326'})
    out = tmp_path / 'tmp_2240_band_yx.tif'
    to_geotiff(da, str(out), crs=4326)
    rt = open_geotiff(str(out))
    assert rt.shape == (4, 5, 3)
    for k in range(3):
        assert int(rt.values[:, :, k].sum()) == (k + 1) * 20


def test_raw_ndarray_band_last_still_writes(tmp_path):
    """Raw ndarray inputs with band-last layout are unaffected by #2240.

    The validator is only invoked from the ``isinstance(data, xr.DataArray)``
    branch of every writer entry point, so a bare numpy array never goes
    through the dim check. This regression guards the inspection-only
    claim in the docstring that raw-ndarray band-last writes still work
    after the tightening.
    """
    arr = np.empty((4, 5, 3), dtype=np.uint8)
    for k in range(3):
        arr[:, :, k] = k + 1
    out = tmp_path / 'tmp_2240_raw_ndarray_band_last.tif'
    to_geotiff(arr, str(out), crs=4326)
    rt = open_geotiff(str(out))
    assert rt.shape == (4, 5, 3)
    for k in range(3):
        assert int(rt.values[:, :, k].sum()) == (k + 1) * 20


def test_raw_ndarray_unusual_third_axis_still_writes(tmp_path):
    """Raw ndarray with no dim metadata is band-last by definition.

    Even if a caller's mental model is ``(y, x, scenario)``, passing a
    bare ndarray bypasses the DataArray dim contract entirely. The
    writer treats the trailing axis as bands -- which is exactly what
    the band-last raw-ndarray API has always done. The #2240
    tightening only constrains DataArray inputs.
    """
    arr = np.empty((4, 5, 3), dtype=np.float32)
    for k in range(3):
        arr[:, :, k] = float(k + 1)
    out = tmp_path / 'tmp_2240_raw_ndarray_band_last_floats.tif'
    to_geotiff(arr, str(out), crs=4326)
    rt = open_geotiff(str(out))
    assert rt.shape == (4, 5, 3)
    for k in range(3):
        assert float(rt.values[:, :, k].sum()) == float(k + 1) * 20
