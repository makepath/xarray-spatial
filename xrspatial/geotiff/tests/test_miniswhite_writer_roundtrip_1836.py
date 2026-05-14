"""Regression tests for ``photometric='miniswhite'`` round-tripping.

The reader unconditionally inverts single-band MinIsWhite data at
``_reader._apply_photometric_miniswhite``. Before issue #1836 the writer
set the TIFF Photometric tag to 0 without inverting pixel values, so
``to_geotiff(..., photometric='miniswhite')`` followed by ``open_geotiff``
returned ``iinfo(dtype).max - original`` for unsigned-integer pixels and
``-original`` for floating-point pixels. Signed integers passed through
both sides unchanged, matching the reader.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


def _da(arr: np.ndarray) -> xr.DataArray:
    h, w = arr.shape
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(h, dtype=np.float64),
                'x': np.arange(w, dtype=np.float64)},
        attrs={'res': (1.0, 1.0)},
    )


def test_uint8_miniswhite_round_trip(tmp_path):
    arr = np.array([[0, 1, 127, 254, 255]], dtype=np.uint8)
    path = tmp_path / 'u8_msw_1836.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    r = open_geotiff(str(path))
    np.testing.assert_array_equal(np.asarray(r.values), arr)


def test_uint16_miniswhite_round_trip(tmp_path):
    info = np.iinfo(np.uint16)
    arr = np.array([[0, 1, info.max // 2, info.max - 1, info.max]],
                   dtype=np.uint16)
    path = tmp_path / 'u16_msw_1836.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    r = open_geotiff(str(path))
    np.testing.assert_array_equal(np.asarray(r.values), arr)


def test_float32_miniswhite_round_trip(tmp_path):
    arr = np.array([[-3.5, 0.0, 0.25, 7.5]], dtype=np.float32)
    path = tmp_path / 'f32_msw_1836.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    r = open_geotiff(str(path))
    np.testing.assert_allclose(np.asarray(r.values), arr)


def test_miniswhite_with_nodata_round_trip(tmp_path):
    arr = np.array([[10.0, np.nan, 20.0, 30.0]], dtype=np.float32)
    path = tmp_path / 'f32_msw_nd_1836.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite', nodata=-9999.0)
    r = open_geotiff(str(path))
    out = np.asarray(r.values)
    assert np.isnan(out[0, 1]), (
        "nodata position must round-trip back to NaN after MinIsWhite "
        f"inversion; got {out!r}"
    )
    np.testing.assert_allclose(out[0, [0, 2, 3]], arr[0, [0, 2, 3]])


def test_miniswhite_int_passthrough(tmp_path):
    """The reader does not invert signed integer MinIsWhite data, so the
    writer must also pass it through unchanged. Otherwise the round-trip
    would silently corrupt signed data."""
    arr = np.array([[-5, -1, 0, 1, 5]], dtype=np.int16)
    path = tmp_path / 'i16_msw_1836.tif'
    to_geotiff(_da(arr), str(path), photometric='miniswhite')
    r = open_geotiff(str(path))
    np.testing.assert_array_equal(np.asarray(r.values), arr)


def test_uint16_miniswhite_with_in_range_nodata_round_trip(tmp_path):
    """Cover the unsigned-nodata inversion branch: an in-range integer
    sentinel must round-trip back to NaN despite the writer pre-inverting
    both pixels and the sentinel value.

    The on-disk sentinel byte is ``iinfo(uint16).max - 9999 = 55536``,
    which collides with a real ``55536`` pixel post-inversion only when
    the reader's mask compares against the right sentinel. The mid-range
    pixels ``200`` and ``65000`` exercise both ends of the dtype range.
    """
    arr = np.array([[10, 9999, 200, 65000]], dtype=np.uint16)
    path = tmp_path / 'u16_msw_nd_1836.tif'
    da_in = xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(1, dtype=np.float64),
                'x': np.arange(4, dtype=np.float64)},
        attrs={'res': (1.0, 1.0), 'nodata': 9999},
    )
    to_geotiff(da_in, str(path), photometric='miniswhite', nodata=9999)
    r = open_geotiff(str(path))
    out = np.asarray(r.values)
    # The reader casts uint16 to float to express NaN at the masked
    # position. The other three pixels must round-trip exactly.
    assert np.isnan(out[0, 1]), (
        f"in-range uint nodata must round-trip to NaN through the "
        f"writer-side sentinel inversion; got {out!r}"
    )
    np.testing.assert_array_equal(
        out[0, [0, 2, 3]].astype(np.uint16), arr[0, [0, 2, 3]])


def test_miniswhite_rejected_with_cog_no_overviews(tmp_path):
    arr = np.zeros((512, 512), dtype=np.uint8)
    path = tmp_path / 'cog_msw_1836.tif'
    with pytest.raises(NotImplementedError, match='miniswhite'):
        to_geotiff(_da(arr), str(path), photometric='miniswhite', cog=True)


def test_miniswhite_rejected_with_explicit_overviews_no_cog(tmp_path):
    """Isolate the ``overview_levels`` branch: ``cog=False`` so the cog
    gate alone would not trigger, only the overview branch does.
    """
    from xrspatial.geotiff._writer import write as _eager_write
    arr = np.zeros((256, 256), dtype=np.uint8)
    path = str(tmp_path / 'ov_msw_1836.tif')
    with pytest.raises(NotImplementedError, match='miniswhite'):
        _eager_write(arr, path, photometric='miniswhite',
                     cog=False, overview_levels=[2])


def test_miniswhite_rejected_on_dask_path(tmp_path):
    """``to_geotiff`` on a dask-backed DataArray must refuse miniswhite
    because the streaming writer does not pre-invert pixels per tile.
    """
    dask_array = pytest.importorskip("dask.array")
    arr = dask_array.zeros((64, 64), dtype=np.uint8, chunks=32)
    da_in = xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': np.arange(64, dtype=np.float64),
                'x': np.arange(64, dtype=np.float64)},
        attrs={'res': (1.0, 1.0)},
    )
    path = tmp_path / 'dask_msw_1836.tif'
    with pytest.raises(NotImplementedError, match='miniswhite'):
        to_geotiff(da_in, str(path), photometric='miniswhite')


def test_miniswhite_rejected_with_extra_tags_photometric_override(tmp_path):
    """An ``extra_tags`` TAG_PHOTOMETRIC override that disagrees with the
    kwarg-resolved photometric for the MinIsWhite single-band case must
    be refused -- otherwise the writer transforms pixels for one tag
    while the IFD advertises a different one.
    """
    from xrspatial.geotiff._writer import write as _eager_write
    from xrspatial.geotiff._dtypes import SHORT
    from xrspatial.geotiff._header import TAG_PHOTOMETRIC
    arr = np.array([[10, 20]], dtype=np.uint8)
    path = str(tmp_path / 'extras_msw_1836.tif')
    # photometric kwarg defaults to MinIsBlack (1), extra_tags forces
    # MinIsWhite (0) on disk. Without the guard this would write pixels
    # in MinIsBlack domain but tag them MinIsWhite, so the reader would
    # invert and corrupt them.
    with pytest.raises(ValueError, match='TAG_PHOTOMETRIC'):
        _eager_write(
            arr, path,
            photometric='auto',
            extra_tags=[(TAG_PHOTOMETRIC, SHORT, 1, 0)],
        )
