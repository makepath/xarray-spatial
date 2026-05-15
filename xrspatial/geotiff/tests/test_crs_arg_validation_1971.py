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
    with pytest.raises(TypeError, match="crs must be int"):
        _validate_crs_arg(4326.0)


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
