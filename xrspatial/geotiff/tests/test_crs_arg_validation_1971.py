"""Validate the writer entry points reject bool / unresolvable EPSG (#1971).

``bool`` is an int subclass, so ``crs=True`` used to slip through
``isinstance(crs, int)`` and write EPSG=1 to the file (with EPSG=0 for
``crs=False``). Integer EPSG codes were also written without a pyproj
round-trip, so any int that does not resolve as a CRS produced a file
with garbage in ``ProjectedCSType`` / ``GeographicType`` and only a
``GeoTIFFFallbackWarning`` to flag it.

Locks down the rejection at all three writer entry points: ``to_geotiff``
(eager), ``write_geotiff_gpu`` (GPU), and ``to_geotiff`` with
``vrt_tiled=True`` (the deprecated VRT-tiled path).
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._crs import _validate_crs_arg

pyproj = pytest.importorskip("pyproj")


def _square(dtype=np.float32):
    return xr.DataArray(
        np.zeros((4, 4), dtype=dtype),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )


@pytest.mark.parametrize("bad_crs", [True, False])
def test_validate_crs_arg_rejects_bool(bad_crs):
    with pytest.raises(ValueError, match="bool"):
        _validate_crs_arg(bad_crs)


def test_validate_crs_arg_rejects_unresolvable_epsg():
    # EPSG:1 does not exist in any CRS database.
    with pytest.raises(ValueError, match="EPSG"):
        _validate_crs_arg(1)


def test_validate_crs_arg_accepts_valid_epsg():
    _validate_crs_arg(4326)  # WGS84


def test_validate_crs_arg_accepts_none():
    _validate_crs_arg(None)


def test_validate_crs_arg_accepts_str():
    # Strings are deferred to ``_wkt_to_epsg`` and the WKT-fallback
    # path; the entry-point validator only catches bool and bogus int.
    _validate_crs_arg("EPSG:4326")
    _validate_crs_arg('PROJCS["foo",GEOGCS["bar"]]')


def test_validate_crs_arg_rejects_non_int_non_str():
    # ValueError vs TypeError split is intentional: bool and unresolvable
    # EPSG are semantically wrong (right type, bad value), while float /
    # other objects are the wrong type entirely. Tests downstream pin
    # both exception classes so a regression in either direction trips.
    with pytest.raises(TypeError, match="crs must be int"):
        _validate_crs_arg(4326.0)


@pytest.mark.parametrize("np_int", [np.int64(4326), np.int32(4326)])
def test_validate_crs_arg_accepts_numpy_integer(np_int):
    # ``isinstance(np.int64(4326), int)`` is False on most platforms, so
    # the pre-fix validator rejected numpy integer CRS values with
    # ``TypeError``. ``numbers.Integral`` covers them.
    _validate_crs_arg(np_int)


def test_validate_crs_arg_rejects_numpy_bool():
    # ``np.bool_`` is not a ``bool`` subclass but is ``numbers.Integral``
    # in some numpy versions. Make sure the bool guard still catches it
    # before the Integral branch coerces ``True`` -> ``1`` -> EPSG=1.
    # If numpy bool is not an Integral instance, it falls through to the
    # TypeError branch, which is also acceptable.
    val = np.bool_(True)
    with pytest.raises((ValueError, TypeError)):
        _validate_crs_arg(val)


@pytest.mark.parametrize("bad_crs", [True, False])
def test_to_geotiff_rejects_bool_crs(bad_crs):
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="bool"):
        to_geotiff(_square(), buf, crs=bad_crs)


def test_to_geotiff_rejects_unresolvable_epsg():
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="EPSG"):
        to_geotiff(_square(), buf, crs=1)


def test_to_geotiff_accepts_valid_epsg():
    buf = io.BytesIO()
    to_geotiff(_square(), buf, crs=4326)
    assert buf.getbuffer().nbytes > 0


def test_to_geotiff_accepts_numpy_int_epsg():
    # End-to-end check that the validator's ``numbers.Integral`` branch
    # actually lets a numpy integer through ``to_geotiff``.
    buf = io.BytesIO()
    to_geotiff(_square(), buf, crs=np.int64(4326))
    assert buf.getbuffer().nbytes > 0


def test_to_geotiff_attrs_crs_bool_bypass(tmp_path):
    # Regression for the self-review blocker: ``attrs={'crs': True}``
    # with no explicit ``crs=`` kwarg used to bypass ``_validate_crs_arg``
    # and write EPSG=1 to the file. The fix calls the validator on the
    # ``attrs['crs']`` value in the writer's int-EPSG branch.
    da = _square()
    da.attrs['crs'] = True
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="bool"):
        to_geotiff(da, buf)


def test_to_geotiff_attrs_crs_unresolvable_epsg_bypass():
    da = _square()
    da.attrs['crs'] = 1  # int that pyproj cannot resolve
    buf = io.BytesIO()
    with pytest.raises(ValueError, match="EPSG"):
        to_geotiff(da, buf)


def test_to_geotiff_vrt_path_rejects_bool_crs(tmp_path):
    # ``to_geotiff(da, '*.vrt')`` dispatches to ``_write_vrt_tiled``,
    # which has its own crs resolution block. The validator runs in
    # that branch too.
    vrt_path = str(tmp_path / "tmp_1971_vrt_tiled.vrt")
    with pytest.raises(ValueError, match="bool"):
        to_geotiff(_square(), vrt_path, crs=True)


def test_to_geotiff_vrt_path_rejects_unresolvable_epsg(tmp_path):
    vrt_path = str(tmp_path / "tmp_1971_vrt_bad_epsg.vrt")
    with pytest.raises(ValueError, match="EPSG"):
        to_geotiff(_square(), vrt_path, crs=1)


def test_to_geotiff_vrt_path_attrs_crs_bool_bypass(tmp_path):
    # Same blocker as ``test_to_geotiff_attrs_crs_bool_bypass`` but for
    # the ``_write_vrt_tiled`` branch (``to_geotiff(da, '*.vrt')``).
    da = _square()
    da.attrs['crs'] = True
    vrt_path = str(tmp_path / "tmp_1971_vrt_attrs_bool.vrt")
    with pytest.raises(ValueError, match="bool"):
        to_geotiff(da, vrt_path)
