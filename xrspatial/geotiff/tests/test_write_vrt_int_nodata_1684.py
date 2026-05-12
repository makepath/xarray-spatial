"""Regression test for #1684: ``write_vrt`` nodata annotation rejected ints.

The api-consistency sweep on 2026-05-12 flagged that
``xrspatial.geotiff.write_vrt`` annotated ``nodata`` as ``float | None``
even though the sibling writers ``to_geotiff`` and ``write_geotiff_gpu``
accept ``float``, ``int``, or ``None``.  Integer sentinels (``65535`` for
uint16, ``-9999`` for int32) flow through the rest of the I/O surface
unchanged, so the float-only hint forced callers either to cast (losing
the exact sentinel) or to ignore the static-type complaint.

This module pins the widened annotation and confirms an integer nodata
round-trips through ``write_vrt`` -> ``read_vrt`` losslessly.
"""
from __future__ import annotations

import inspect
import typing

import numpy as np
import xarray as xr

from xrspatial.geotiff import to_geotiff, write_vrt
from xrspatial.geotiff import _vrt as _vrt_module


def _nodata_annotation(fn):
    sig = inspect.signature(fn)
    return sig.parameters["nodata"].annotation


def test_write_vrt_public_nodata_accepts_int_annotation():
    """The public wrapper widens the annotation to include int."""
    ann = _nodata_annotation(write_vrt)
    # Allow either typing.Union[float, int, None] or float | int | None.
    if isinstance(ann, str):
        # Forward-referenced string annotation (rare here; defensive).
        assert "int" in ann, ann
        return
    if hasattr(typing, "get_args"):
        args = set(typing.get_args(ann))
        if args:
            assert int in args, args
            return
    # Fallback: stringify the annotation.
    assert "int" in str(ann), str(ann)


def test_write_vrt_internal_nodata_accepts_int_annotation():
    """The internal helper in `_vrt.py` mirrors the public surface."""
    ann = _nodata_annotation(_vrt_module.write_vrt)
    if isinstance(ann, str):
        assert "int" in ann, ann
        return
    if hasattr(typing, "get_args"):
        args = set(typing.get_args(ann))
        if args:
            assert int in args, args
            return
    assert "int" in str(ann), str(ann)


def test_write_vrt_int_nodata_round_trips(tmp_path):
    """An int nodata renders to ``<NoDataValue>`` and parses back the same."""
    # Build a tiny uint16 tile so the sentinel makes sense.
    arr = np.array([[100, 200, 65535],
                    [300, 400, 500]], dtype=np.uint16)
    da = xr.DataArray(
        arr,
        dims=["y", "x"],
        coords={
            "y": np.array([0.5, 1.5]),
            "x": np.array([0.5, 1.5, 2.5]),
        },
        attrs={"crs": 4326},
    )
    tif_path = tmp_path / "source.tif"
    to_geotiff(da, str(tif_path))

    vrt_path = tmp_path / "mosaic.vrt"
    # Passing an int sentinel must not raise; the surface should match
    # to_geotiff's "float, int, or None" contract.
    write_vrt(str(vrt_path), [str(tif_path)], nodata=65535)

    # Confirm the int round-trips through the parser back into a VRT band.
    parsed = _vrt_module.parse_vrt(
        vrt_path.read_text(), vrt_dir=str(tmp_path))
    band_nodata = parsed.bands[0].nodata
    assert band_nodata == 65535, band_nodata
