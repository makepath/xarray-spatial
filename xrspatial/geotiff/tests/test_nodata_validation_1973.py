"""Refuse non-numeric ``nodata=`` / ``attrs['_FillValue']`` (#1973).

The writer compares the resolved nodata against pixel values via
``np.isnan`` and casts it to the array dtype. A non-numeric value
used to fall through ``_resolve_nodata_attr`` (returned verbatim) or
the ``nodata=`` kwarg path and then crash inside NumPy with
``ufunc 'isnan' not supported``. Both the entry point and the attr
resolution path now refuse non-numeric values up front with a clear
error.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._attrs import _resolve_nodata_attr
from xrspatial.geotiff._validation import _validate_nodata_arg


def _nan_square():
    return xr.DataArray(
        np.full((4, 4), np.nan, dtype=np.float32),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )


@pytest.mark.parametrize("bad", ['missing', object(), [1, 2]])
def test_validate_nodata_arg_rejects_non_numeric(bad):
    with pytest.raises(ValueError, match="nodata must be numeric"):
        _validate_nodata_arg(bad)


@pytest.mark.parametrize("ok", [None, 0, -9999, 1.5, np.float32(-1), np.int64(0)])
def test_validate_nodata_arg_accepts_numeric_and_none(ok):
    _validate_nodata_arg(ok)


def test_resolve_nodata_attr_rejects_non_numeric_fillvalue():
    with pytest.raises(ValueError, match="_FillValue"):
        _resolve_nodata_attr({'_FillValue': 'missing'})


def test_resolve_nodata_attr_rejects_non_numeric_nodata_attr():
    with pytest.raises(ValueError, match=r"attrs\['nodata'\]"):
        _resolve_nodata_attr({'nodata': 'missing'})


def test_resolve_nodata_attr_skips_non_numeric_in_nodatavals():
    # nodatavals (rioxarray's per-band tuple) keeps its skip-on-non-numeric
    # behaviour: those values often come from arbitrary upstream pipelines
    # and a single bad entry should not block writing.
    assert _resolve_nodata_attr({'nodatavals': ('NaN ', -9999.0)}) == -9999.0


def test_resolve_nodata_attr_still_accepts_numeric_fillvalue():
    assert _resolve_nodata_attr({'_FillValue': -9999}) == -9999


def test_resolve_nodata_attr_returns_none_for_nan_fillvalue():
    assert _resolve_nodata_attr({'_FillValue': float('nan')}) is None


def test_to_geotiff_rejects_non_numeric_nodata_kwarg():
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="nodata must be numeric"):
        to_geotiff(_nan_square(), buf, nodata='missing')


def test_to_geotiff_rejects_non_numeric_fillvalue_attr():
    da = _nan_square()
    da.attrs['_FillValue'] = 'missing'
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="_FillValue"):
        to_geotiff(da, buf)


def test_to_geotiff_vrt_path_rejects_non_numeric_nodata(tmp_path):
    vrt_path = str(tmp_path / "tmp_1973_vrt.vrt")
    with pytest.raises(ValueError, match="nodata must be numeric"):
        to_geotiff(_nan_square(), vrt_path, nodata='missing')


def test_to_geotiff_accepts_numeric_nodata_kwarg():
    buf = io.BytesIO()
    to_geotiff(_nan_square(), buf, nodata=-9999)
    assert buf.getbuffer().nbytes > 0
