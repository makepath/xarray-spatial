"""Regression tests for issue #1716.

``to_geotiff`` previously stored ``abs(pixel_width)`` / ``abs(pixel_height)``
in ModelPixelScaleTag and the reader hard-coded a north-up reconstruction.
DataArrays with descending x or ascending y silently round-tripped with the
wrong georeference.  The writer now emits ModelTransformationTag (34264)
for non-standard orientations so the sign survives the round trip.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import (TAG_MODEL_PIXEL_SCALE, TAG_MODEL_TIEPOINT,
                                        TAG_MODEL_TRANSFORMATION)
from xrspatial.geotiff._header import parse_all_ifds, parse_header


def _ifd_tag_ids(path: str) -> set[int]:
    with open(path, 'rb') as fh:
        data = fh.read()
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    return set(ifds[0].entries.keys())


def _make_da(x_coords: np.ndarray, y_coords: np.ndarray) -> xr.DataArray:
    arr = np.arange(len(y_coords) * len(x_coords), dtype=np.float32)
    arr = arr.reshape(len(y_coords), len(x_coords))
    return xr.DataArray(
        arr,
        dims=('y', 'x'),
        coords={'y': y_coords, 'x': x_coords},
    )


def test_descending_x_roundtrip(tmp_path):
    """Descending x coords survive the round trip."""
    # x decreases left-to-right (unusual but valid)
    x = np.array([200.0, 190.0, 180.0, 170.0, 160.0], dtype=np.float64)
    y = np.array([50.0, 40.0, 30.0, 20.0], dtype=np.float64)  # north-up
    da = _make_da(x, y)

    out = tmp_path / 'tmp_1716_desc_x.tif'
    to_geotiff(da, str(out), crs=4326)

    loaded = open_geotiff(str(out))
    np.testing.assert_allclose(loaded.coords['x'].values, x)
    np.testing.assert_allclose(loaded.coords['y'].values, y)
    np.testing.assert_array_equal(loaded.values, da.values)


def test_ascending_y_roundtrip(tmp_path):
    """Ascending y coords survive the round trip."""
    x = np.array([160.0, 170.0, 180.0, 190.0, 200.0], dtype=np.float64)
    # y increases top-to-bottom (south-up)
    y = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    da = _make_da(x, y)

    out = tmp_path / 'tmp_1716_asc_y.tif'
    to_geotiff(da, str(out), crs=4326)

    loaded = open_geotiff(str(out))
    np.testing.assert_allclose(loaded.coords['x'].values, x)
    np.testing.assert_allclose(loaded.coords['y'].values, y)
    np.testing.assert_array_equal(loaded.values, da.values)


def test_descending_x_and_ascending_y_roundtrip(tmp_path):
    """Both axes flipped relative to north-up."""
    x = np.array([200.0, 190.0, 180.0, 170.0, 160.0], dtype=np.float64)
    y = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    da = _make_da(x, y)

    out = tmp_path / 'tmp_1716_desc_x_asc_y.tif'
    to_geotiff(da, str(out), crs=4326)

    loaded = open_geotiff(str(out))
    np.testing.assert_allclose(loaded.coords['x'].values, x)
    np.testing.assert_allclose(loaded.coords['y'].values, y)
    np.testing.assert_array_equal(loaded.values, da.values)


def test_north_up_still_uses_pixel_scale_and_tiepoint(tmp_path):
    """Standard north-up orientation keeps ModelPixelScale + ModelTiepoint."""
    x = np.array([160.0, 170.0, 180.0, 190.0, 200.0], dtype=np.float64)
    y = np.array([50.0, 40.0, 30.0, 20.0], dtype=np.float64)
    da = _make_da(x, y)

    out = tmp_path / 'tmp_1716_north_up.tif'
    to_geotiff(da, str(out), crs=4326)

    tag_ids = _ifd_tag_ids(str(out))
    assert TAG_MODEL_PIXEL_SCALE in tag_ids
    assert TAG_MODEL_TIEPOINT in tag_ids
    assert TAG_MODEL_TRANSFORMATION not in tag_ids


def test_descending_x_uses_transformation_tag(tmp_path):
    """Non-standard orientation emits ModelTransformationTag and skips
    the scale/tiepoint pair."""
    x = np.array([200.0, 190.0, 180.0, 170.0, 160.0], dtype=np.float64)
    y = np.array([50.0, 40.0, 30.0, 20.0], dtype=np.float64)
    da = _make_da(x, y)

    out = tmp_path / 'tmp_1716_desc_x_tags.tif'
    to_geotiff(da, str(out), crs=4326)

    tag_ids = _ifd_tag_ids(str(out))
    assert TAG_MODEL_TRANSFORMATION in tag_ids
    assert TAG_MODEL_PIXEL_SCALE not in tag_ids
    assert TAG_MODEL_TIEPOINT not in tag_ids


def test_ascending_y_uses_transformation_tag(tmp_path):
    x = np.array([160.0, 170.0, 180.0, 190.0, 200.0], dtype=np.float64)
    y = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    da = _make_da(x, y)

    out = tmp_path / 'tmp_1716_asc_y_tags.tif'
    to_geotiff(da, str(out), crs=4326)

    tag_ids = _ifd_tag_ids(str(out))
    assert TAG_MODEL_TRANSFORMATION in tag_ids
    assert TAG_MODEL_PIXEL_SCALE not in tag_ids
    assert TAG_MODEL_TIEPOINT not in tag_ids
