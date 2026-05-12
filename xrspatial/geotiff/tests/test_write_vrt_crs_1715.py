"""Regression test for #1715: write_vrt accepts ``crs`` for parity with
``to_geotiff`` / ``write_geotiff_gpu``.

The api-consistency sweep on 2026-05-12 flagged that ``write_vrt`` was
the only writer in ``xrspatial.geotiff`` using ``crs_wkt`` instead of
``crs``, breaking the "forward the same kwargs to whichever writer
matches the output extension" pattern. The fix adds ``crs`` as the
canonical kwarg and keeps ``crs_wkt`` as a deprecated alias.

This module pins:

* ``crs`` accepts ``int`` (EPSG) and ``str`` (WKT) and ``None``,
  matching ``to_geotiff``/``write_geotiff_gpu``.
* The ``crs_wkt`` alias still works but emits ``DeprecationWarning``.
* Passing both ``crs`` and ``crs_wkt`` raises ``TypeError``.
* The deprecation shim does NOT warn when neither kwarg is supplied
  (the no-crs path picks from the first source, unchanged from
  pre-#1715 behaviour).
* Read-back round trip: ``read_vrt(written).attrs['crs'] == 4326``
  when the writer was given ``crs=4326``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    read_vrt,
    to_geotiff,
    write_vrt,
)


def _build_source_tif(tmp_path, name='src.tif'):
    """Create a small GeoTIFF used as the VRT's source file."""
    arr = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )
    p = str(tmp_path / name)
    to_geotiff(da, p)
    return p


# --- Signature pins ---


def test_write_vrt_accepts_crs_kwarg():
    """``crs`` is in the signature and defaults to ``None``."""
    import inspect

    sig = inspect.signature(write_vrt)
    assert 'crs' in sig.parameters
    assert sig.parameters['crs'].default is None


def test_write_vrt_crs_annotation_matches_writer_trio():
    """``crs`` is annotated ``int | str | None``, identical to
    ``to_geotiff(..., crs=...)`` and ``write_geotiff_gpu(..., crs=...)``.
    """
    import inspect

    sig = inspect.signature(write_vrt)
    ann = str(sig.parameters['crs'].annotation)
    assert ann == 'int | str | None'


# --- Runtime: ``crs=<EPSG int>`` writes an EPSG-resolved WKT ---


def test_write_vrt_crs_epsg_int_writes_wkt_to_xml(tmp_path):
    """``crs=4326`` resolves to a WKT string in the VRT's <SRS> element.

    The current implementation forwards the WKT to ``_vrt.write_vrt``,
    which interpolates it into the <SRS> XML node. Reading the file
    back with ``read_vrt`` must therefore produce
    ``attrs['crs'] == 4326`` (because ``_wkt_to_epsg`` round-trips
    EPSG:4326's WKT cleanly).
    """
    src = _build_source_tif(tmp_path, 'epsg_int.tif')
    vrt_path = str(tmp_path / 'epsg_int.vrt')

    out = write_vrt(vrt_path, [src], crs=4326)
    assert out == vrt_path
    assert os.path.exists(vrt_path)

    da = read_vrt(vrt_path)
    assert da.attrs.get('crs') == 4326


def test_write_vrt_crs_wkt_string(tmp_path):
    """``crs=<WKT string>`` passes the WKT through verbatim."""
    src = _build_source_tif(tmp_path, 'wkt.tif')
    vrt_path = str(tmp_path / 'wkt.vrt')

    # Build a WKT for EPSG:4326 directly via pyproj
    from pyproj import CRS

    wkt = CRS.from_epsg(4326).to_wkt()

    out = write_vrt(vrt_path, [src], crs=wkt)
    assert out == vrt_path
    da = read_vrt(vrt_path)
    # WKT round-trips back to EPSG:4326 via _wkt_to_epsg
    assert da.attrs.get('crs') == 4326


def test_write_vrt_crs_none_falls_through(tmp_path):
    """``crs=None`` (the default) picks the CRS from the first source."""
    src = _build_source_tif(tmp_path, 'none.tif')
    vrt_path = str(tmp_path / 'none.vrt')

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        out = write_vrt(vrt_path, [src], crs=None)
    assert out == vrt_path
    da = read_vrt(vrt_path)
    # The source TIFF was written with EPSG:4326; VRT inherits it.
    assert da.attrs.get('crs') == 4326


def test_write_vrt_no_crs_kwarg_no_warning(tmp_path):
    """Omitting ``crs`` entirely (the most common call shape) does not
    emit any warning. The deprecation shim only fires when ``crs_wkt``
    is supplied explicitly."""
    src = _build_source_tif(tmp_path, 'no_kwarg.tif')
    vrt_path = str(tmp_path / 'no_kwarg.vrt')

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        write_vrt(vrt_path, [src])  # neither kwarg supplied
    assert os.path.exists(vrt_path)


# --- Deprecation shim: ``crs_wkt=`` still works but warns ---


def test_write_vrt_crs_wkt_deprecated_warns(tmp_path):
    """Passing ``crs_wkt=<wkt>`` emits ``DeprecationWarning`` but still
    produces a working VRT."""
    src = _build_source_tif(tmp_path, 'depr.tif')
    vrt_path = str(tmp_path / 'depr.vrt')

    from pyproj import CRS

    wkt = CRS.from_epsg(4326).to_wkt()

    with pytest.warns(DeprecationWarning, match='crs_wkt'):
        out = write_vrt(vrt_path, [src], crs_wkt=wkt)
    assert out == vrt_path
    da = read_vrt(vrt_path)
    assert da.attrs.get('crs') == 4326


def test_write_vrt_crs_wkt_none_still_warns(tmp_path):
    """``crs_wkt=None`` (explicit) was a documented shape in the old
    signature -- it now warns because the caller is using the
    deprecated kwarg name, even if the value is None."""
    src = _build_source_tif(tmp_path, 'depr_none.tif')
    vrt_path = str(tmp_path / 'depr_none.vrt')

    with pytest.warns(DeprecationWarning, match='crs_wkt'):
        write_vrt(vrt_path, [src], crs_wkt=None)
    assert os.path.exists(vrt_path)


def test_write_vrt_both_crs_and_crs_wkt_rejected(tmp_path):
    """Passing both raises ``TypeError`` rather than silently picking
    one. The error message names both kwargs so the caller can fix
    their call quickly."""
    src = _build_source_tif(tmp_path, 'both.tif')
    vrt_path = str(tmp_path / 'both.vrt')

    from pyproj import CRS

    wkt = CRS.from_epsg(4326).to_wkt()

    with pytest.raises(TypeError, match='crs.*crs_wkt'):
        write_vrt(vrt_path, [src], crs=4326, crs_wkt=wkt)


# --- Cross-writer parity: same kwarg name on all three writers ---


def test_writer_trio_all_accept_crs_kwarg():
    """``crs`` is the canonical kwarg on every public writer in the trio.
    A caller forwarding ``crs=<value>`` to whichever writer matches the
    output extension never has to special-case the kwarg name (issue
    #1715)."""
    import inspect

    from xrspatial.geotiff import to_geotiff, write_geotiff_gpu, write_vrt

    for fn in (to_geotiff, write_geotiff_gpu, write_vrt):
        sig = inspect.signature(fn)
        assert 'crs' in sig.parameters, f"{fn.__name__} missing crs kwarg"
        assert (
            str(sig.parameters['crs'].annotation) == 'int | str | None'
        ), f"{fn.__name__}.crs annotation drift"


# --- Negative tests: bad input shapes ---


def test_write_vrt_crs_invalid_type_rejected(tmp_path):
    """``crs=<list>`` (or any non-int/str/None) raises ``TypeError`` from
    the public wrapper rather than from deep inside the writer."""
    src = _build_source_tif(tmp_path, 'bad_type.tif')
    vrt_path = str(tmp_path / 'bad_type.vrt')

    with pytest.raises(TypeError, match='crs must be'):
        write_vrt(vrt_path, [src], crs=[4326])


def test_write_vrt_crs_unparseable_string_rejected(tmp_path):
    """``crs='not a CRS'`` raises ``ValueError`` from the public
    wrapper (the WKT keyword heuristic recognises PROJCS/GEOGCS only;
    everything else is sent through pyproj which will reject it)."""
    src = _build_source_tif(tmp_path, 'bad_str.tif')
    vrt_path = str(tmp_path / 'bad_str.vrt')

    with pytest.raises(ValueError, match='Could not parse crs'):
        write_vrt(vrt_path, [src], crs='not-a-real-crs-string')
