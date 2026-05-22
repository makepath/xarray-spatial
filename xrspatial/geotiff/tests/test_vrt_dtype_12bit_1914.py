"""Regression tests for issue #1914.

``write_vrt`` used to build its GDAL ``dataType`` attribute from a local
if/elif/else ladder keyed on ``sample_format`` and ``bps`` rather than
going through the central ``tiff_dtype_to_numpy`` resolver. For
``bps=12, sample_format=1`` that ladder fell through to ``'Byte'``,
even though the reader promotes the same sample to ``uint16``. A VRT
over a valid 12-bit unsigned source would then advertise a narrower
type and could be truncated by downstream GDAL/VRT consumers.

These tests pin the VRT dtype name for every TIFF (bps, sample_format)
the resolver supports, including the 12-bit regression case. The
helper ``_vrt_dtype_name_for`` is called directly because the on-disk
writers in ``to_geotiff`` only emit standard 8/16/32/64-bit samples,
so a real 12-bit TIFF can't be round-tripped through ``write_vrt`` in
the test environment.

A second block exercises ``write_vrt`` end-to-end with normal 16-bit
unsigned tiles and asserts the generated XML carries ``UInt16`` (not
``Byte``) so the actual writer path is covered too.
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._vrt import _NP_TO_VRT_DTYPE, _vrt_dtype_name_for, write_vrt

# ---------------------------------------------------------------------------
# Direct helper tests: every supported (bps, sample_format) pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bps,sf,expected",
    [
        # The regression case. ``_dtypes.tiff_dtype_to_numpy`` promotes
        # ``bps=12, sf=1`` to ``uint16``; the VRT dtype must follow.
        (12, 1, "UInt16"),
        (12, 4, "UInt16"),  # SAMPLE_FORMAT_UNDEFINED -> uint
        # Sub-byte unsigned bit depths the reader promotes to uint8.
        (1, 1, "Byte"),
        (2, 1, "Byte"),
        (4, 1, "Byte"),
        # Standard unsigned bit depths.
        (8, 1, "Byte"),
        (16, 1, "UInt16"),
        (32, 1, "UInt32"),
        (64, 1, "UInt64"),
        # Signed integers.
        (8, 2, "Int8"),
        (16, 2, "Int16"),
        (32, 2, "Int32"),
        (64, 2, "Int64"),
        # Floats. The previous ladder defaulted ``sf=3`` to ``Float32``
        # for any non-64 bps, which silently downcast ``bps=16`` halfs.
        # The resolver rejects bps=16 floats outright (no IEEE half
        # support), so we only pin the supported widths here.
        (32, 3, "Float32"),
        (64, 3, "Float64"),
    ],
)
def test_vrt_dtype_name_for_supported(bps, sf, expected):
    assert _vrt_dtype_name_for(bps, sf) == expected


def test_vrt_dtype_name_for_sample_format_sequence_resolves():
    # ``ifd.sample_format`` is sometimes a tuple of per-band values; the
    # helper must funnel it through ``resolve_sample_format`` rather
    # than treating the raw tuple as an int.
    assert _vrt_dtype_name_for(8, [1, 1]) == "Byte"
    assert _vrt_dtype_name_for(16, (2, 2)) == "Int16"


def test_vrt_dtype_name_for_unsupported_raises():
    # bps=24 / sf=2 isn't in ``tiff_dtype_to_numpy``; should surface as
    # a ValueError from the resolver rather than silently mapping to
    # 'Byte' or 'Int32' the way the old ladder did.
    with pytest.raises(ValueError):
        _vrt_dtype_name_for(24, 2)


def test_np_to_vrt_dtype_table_covers_all_resolver_outputs():
    # Defensive: every numpy dtype that ``tiff_dtype_to_numpy`` can
    # produce must have an entry in ``_NP_TO_VRT_DTYPE`` or else
    # ``_vrt_dtype_name_for`` will raise the catch-all ValueError in
    # normal use. This catches future additions to the resolver that
    # forget to wire up a GDAL name.
    from xrspatial.geotiff._dtypes import tiff_dtype_to_numpy

    pairs = [
        (8, 1), (8, 2),
        (16, 1), (16, 2),
        (32, 1), (32, 2), (32, 3),
        (64, 1), (64, 2), (64, 3),
        (1, 1), (2, 1), (4, 1), (12, 1),
    ]
    for bps, sf in pairs:
        np_dtype = tiff_dtype_to_numpy(bps, sf)
        assert np_dtype.type in _NP_TO_VRT_DTYPE, (
            f"resolver yields {np_dtype} for bps={bps}, sf={sf} but "
            f"_NP_TO_VRT_DTYPE has no entry for it"
        )


# ---------------------------------------------------------------------------
# End-to-end write_vrt sanity: uint16 source produces UInt16 in the VRT
# ---------------------------------------------------------------------------


def _unique_dir(tmp_path, label: str) -> str:
    d = tmp_path / f"vrt_1914_{label}_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    return str(d)


def _write_uint16_tif(path: str, *, h: int = 4, w: int = 4,
                      origin_x: float = 0.0) -> None:
    arr = np.arange(h * w, dtype=np.uint16).reshape(h, w)
    y = 100.0 + (np.arange(h) + 0.5) * -1.0
    x = origin_x + (np.arange(w) + 0.5) * 1.0
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326},
    )
    to_geotiff(da, path, compression='none')


def test_uint16_source_writes_uint16_vrt_datatype(tmp_path):
    # Pre-fix this would have produced ``dataType="UInt16"`` too, since
    # bps=16/sf=1 happened to be in the old ladder. The point of this
    # test is to assert the XML still says UInt16 after refactoring the
    # dtype path through the central resolver -- i.e. we didn't
    # accidentally regress the easy case while fixing the 12-bit one.
    d = _unique_dir(tmp_path, "u16")
    a = os.path.join(d, "a.tif")
    b = os.path.join(d, "b.tif")
    _write_uint16_tif(a)
    _write_uint16_tif(b, origin_x=4.0)
    vrt = os.path.join(d, "out.vrt")
    write_vrt(vrt, [a, b])
    with open(vrt) as f:
        xml = f.read()
    assert 'dataType="UInt16"' in xml
    assert 'dataType="Byte"' not in xml


def test_int16_source_writes_int16_vrt_datatype(tmp_path):
    # The old ladder mapped sf=2, bps=16 to 'Int16' correctly; pin that
    # behaviour so the new resolver path doesn't drift.
    d = _unique_dir(tmp_path, "i16")
    a = os.path.join(d, "a.tif")
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    y = 100.0 + (np.arange(4) + 0.5) * -1.0
    x = (np.arange(4) + 0.5) * 1.0
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326},
    )
    to_geotiff(da, a, compression='none')
    vrt = os.path.join(d, "out.vrt")
    write_vrt(vrt, [a])
    with open(vrt) as f:
        xml = f.read()
    assert 'dataType="Int16"' in xml
