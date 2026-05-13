"""Regression tests for issue #1769: photometric kwarg and extra_tags override.

Before this fix, the writer silently labelled any 3+ band array as RGB,
with the 4th band tagged as unassociated alpha. Scientific multispectral
rasters (e.g. R, G, B, NIR) were written with Photometric=2 (RGB) and
ExtraSamples=[2] (alpha), so downstream readers composited the NIR band
as transparency.

A second problem made the bug hard to work around: a user passing
``extra_tags=[(TAG_EXTRA_SAMPLES, ...)]`` to ``to_geotiff`` could not
override the writer's auto tag, because the dedup loop dropped any
user-supplied tag whose id was already emitted.

The fix:

* Adds a ``photometric`` kwarg to ``to_geotiff`` / ``write_geotiff_gpu``
  with the default ``'auto'`` mapping to MinIsBlack for any band count.
  RGB is opt-in via ``photometric='rgb'`` or ``photometric='rgba'``.
* Lets a user-supplied ``extra_tags`` entry of ``TAG_PHOTOMETRIC`` or
  ``TAG_EXTRA_SAMPLES`` win outright over the writer's chosen value.

These tests pin the new defaults and the override behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._dtypes import SHORT
from xrspatial.geotiff._header import (
    TAG_EXTRA_SAMPLES,
    TAG_PHOTOMETRIC,
    parse_header,
    parse_ifd,
)


def _read_primary_ifd(path: str):
    """Parse the primary IFD of ``path`` and return it."""
    with open(path, 'rb') as f:
        raw = f.read()
    hdr = parse_header(raw[:16])
    return parse_ifd(raw, hdr.first_ifd_offset, hdr)


def _to_da(arr: np.ndarray) -> xr.DataArray:
    if arr.ndim == 3:
        return xr.DataArray(arr, dims=('y', 'x', 'band'))
    return xr.DataArray(arr, dims=('y', 'x'))


def test_four_band_default_is_minisblack_with_unspecified_extras(tmp_path):
    """Default photometric='auto' on a 4-band raster must write
    MinIsBlack + 3 ExtraSamples=unspecified, not RGB+alpha."""
    arr = np.zeros((32, 32, 4), dtype=np.uint16)
    path = str(tmp_path / 'four_band_default_1769.tif')
    to_geotiff(_to_da(arr), path)

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 1  # MinIsBlack
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (0, 0, 0)


def test_four_band_photometric_rgba_writes_rgb_plus_alpha(tmp_path):
    """photometric='rgba' is the opt-in for the old RGB+alpha behaviour."""
    arr = np.zeros((32, 32, 4), dtype=np.uint16)
    path = str(tmp_path / 'four_band_rgba_1769.tif')
    to_geotiff(_to_da(arr), path, photometric='rgba')

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 2  # RGB
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (2,)  # unassociated alpha


def test_four_band_photometric_rgb_writes_unspecified_extras(tmp_path):
    """photometric='rgb' on a 4-band emits Photometric=RGB with the
    leftover band tagged as unspecified (not alpha)."""
    arr = np.zeros((32, 32, 4), dtype=np.uint16)
    path = str(tmp_path / 'four_band_rgb_1769.tif')
    to_geotiff(_to_da(arr), path, photometric='rgb')

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 2
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (0,)


def test_three_band_default_is_minisblack_regression_1769(tmp_path):
    """Default on a 3-band raster must no longer claim RGB.

    The previous default treated samples_per_pixel >= 3 as RGB; the new
    'auto' default writes MinIsBlack regardless of band count so that
    multispectral 3-band rasters (e.g. R, NIR, SWIR) are not silently
    tagged as colour."""
    arr = np.zeros((32, 32, 3), dtype=np.uint16)
    path = str(tmp_path / 'three_band_default_1769.tif')
    to_geotiff(_to_da(arr), path)

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 1
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (0, 0)


def test_single_band_default_unchanged_1769(tmp_path):
    """1-band rasters stay MinIsBlack with no ExtraSamples tag."""
    arr = np.zeros((16, 16), dtype=np.uint8)
    path = str(tmp_path / 'one_band_default_1769.tif')
    to_geotiff(_to_da(arr), path)

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 1
    # No ExtraSamples tag at all for single-band.
    assert ifd.get_values(TAG_EXTRA_SAMPLES) is None


def test_user_extra_tags_override_extra_samples_1769(tmp_path):
    """A user-supplied (TAG_EXTRA_SAMPLES, ...) entry wins over the
    writer's auto value, even when photometric='rgb' would otherwise
    emit ExtraSamples=[0] for the 4th band."""
    arr = np.zeros((32, 32, 4), dtype=np.uint16)
    da = xr.DataArray(
        arr, dims=('y', 'x', 'band'),
        attrs={'extra_tags': [
            (TAG_EXTRA_SAMPLES, SHORT, 3, [0, 0, 0]),
        ]},
    )
    path = str(tmp_path / 'override_extras_1769.tif')
    to_geotiff(da, path, photometric='rgb')

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 2  # RGB from kwarg
    # User override gives 3 unspecified entries, not the auto [0] for
    # the single 4th band.
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (0, 0, 0)


def test_user_extra_tags_override_photometric_1769(tmp_path):
    """A user-supplied (TAG_PHOTOMETRIC, ...) entry wins over the
    photometric kwarg."""
    arr = np.zeros((32, 32, 4), dtype=np.uint16)
    da = xr.DataArray(
        arr, dims=('y', 'x', 'band'),
        attrs={'extra_tags': [
            (TAG_PHOTOMETRIC, SHORT, 1, 0),  # MinIsWhite
        ]},
    )
    path = str(tmp_path / 'override_photometric_1769.tif')
    # photometric='rgb' would otherwise emit Photometric=2.
    to_geotiff(da, path, photometric='rgb')

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 0  # MinIsWhite from override


def test_explicit_integer_photometric_1769(tmp_path):
    """An int passed as ``photometric`` is written verbatim."""
    arr = np.zeros((32, 32), dtype=np.uint8)
    path = str(tmp_path / 'photometric_int_1769.tif')
    # 0 = MinIsWhite
    to_geotiff(_to_da(arr), path, photometric=0)
    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 0


def test_invalid_photometric_name_raises_1769(tmp_path):
    """An unknown photometric name surfaces a clear ValueError."""
    arr = np.zeros((16, 16), dtype=np.uint8)
    path = str(tmp_path / 'invalid_photo_1769.tif')
    with pytest.raises(ValueError, match='not a valid name'):
        to_geotiff(_to_da(arr), path, photometric='not-a-thing')


def test_rgba_requires_four_bands_1769(tmp_path):
    """photometric='rgba' on a 3-band raster surfaces a clear error."""
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    path = str(tmp_path / 'rgba_three_band_1769.tif')
    with pytest.raises(ValueError, match='at least 4 bands'):
        to_geotiff(_to_da(arr), path, photometric='rgba')


def test_rgb_requires_three_bands_1769(tmp_path):
    """photometric='rgb' on a 2-band raster surfaces a clear error."""
    arr = np.zeros((16, 16, 2), dtype=np.uint8)
    path = str(tmp_path / 'rgb_two_band_1769.tif')
    with pytest.raises(ValueError, match='at least 3 bands'):
        to_geotiff(_to_da(arr), path, photometric='rgb')


def test_explicit_int_rgb_requires_three_bands_1769(tmp_path):
    """photometric=2 (RGB by int) on a 1-band raster also raises."""
    arr = np.zeros((16, 16), dtype=np.uint8)
    path = str(tmp_path / 'rgb_int_one_band_1769.tif')
    with pytest.raises(ValueError, match='at least 3 bands'):
        to_geotiff(_to_da(arr), path, photometric=2)


def test_dask_streaming_default_is_minisblack_1769(tmp_path):
    """The dask streaming write path honours the new default too."""
    dask = pytest.importorskip('dask.array')
    arr = dask.zeros((64, 64, 4), dtype=np.uint16, chunks=(32, 32, 4))
    da = xr.DataArray(arr, dims=('y', 'x', 'band'))
    path = str(tmp_path / 'four_band_dask_1769.tif')
    to_geotiff(da, path)

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 1
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (0, 0, 0)


def test_cog_overviews_carry_same_photometric_1769(tmp_path):
    """COG overviews must share the primary IFD's Photometric so the
    pyramid stays internally consistent."""
    # Use a non-default photometric so we can tell the value propagated
    # rather than matching by chance.
    arr = np.zeros((512, 512, 4), dtype=np.uint8)
    path = str(tmp_path / 'cog_overviews_1769.tif')
    to_geotiff(
        _to_da(arr), path, cog=True, tile_size=128,
        overview_levels=[2, 4], photometric='rgba',
    )

    with open(path, 'rb') as f:
        raw = f.read()
    hdr = parse_header(raw[:16])
    offset = hdr.first_ifd_offset
    seen = []
    while offset:
        ifd = parse_ifd(raw, offset, hdr)
        seen.append(ifd.get_value(TAG_PHOTOMETRIC))
        offset = ifd.next_ifd_offset
    # Primary + two overviews -- all three must be Photometric=RGB.
    assert seen == [2, 2, 2]
