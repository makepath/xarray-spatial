"""Tests for the ``bbox=`` parameter on ``open_geotiff`` (#2555).

``bbox=(x_min, y_min, x_max, y_max)`` is resolved to a pixel
``window=`` via a header-only metadata read and forwarded to the
existing backend dispatch. The tests cover the round-trip against
``window=``, mutual exclusion, malformed inputs, clamping to file
extent, georef requirement, and rotated-affine rejection.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff

from .._helpers.markers import requires_gpu


@pytest.fixture
def georef_raster_2555(tmp_path):
    """Build a 100x100 EPSG:4326 raster spanning -122.5..-122.0 lon,
    37.3..37.8 lat (north-up). Returns the file path."""
    data = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    xs = np.linspace(-122.5, -122.0, 100)
    ys = np.linspace(37.8, 37.3, 100)  # decreasing -> north-up
    arr = xr.DataArray(
        data,
        dims=('y', 'x'),
        coords={'y': ys, 'x': xs},
        attrs={'crs': 'EPSG:4326'},
    )
    path = os.path.join(tmp_path, 'src_2555.tif')
    to_geotiff(arr, path)
    return path


class TestBboxResolution:
    def test_full_bbox_matches_full_read(self, georef_raster_2555):
        """A bbox covering the file's extent matches a no-window read."""
        full = open_geotiff(georef_raster_2555)
        # Pad the bbox slightly outside the extent; _extent_to_window
        # clamps to file bounds so the result must still be the full
        # raster.
        bb = (-122.6, 37.2, -121.9, 37.9)
        sub = open_geotiff(georef_raster_2555, bbox=bb)
        assert sub.shape == full.shape
        np.testing.assert_array_equal(sub.values, full.values)

    def test_bbox_matches_equivalent_pixel_window(self, georef_raster_2555):
        """bbox= and the pixel window it resolves to must produce the
        same array. Resolved via the internal helper so the test
        doesn't have to redo the affine math. Asserts both shape and
        pixel values so a row/col swap in the resolver would surface."""
        from xrspatial.geotiff import _bbox_to_window
        bb = (-122.4, 37.4, -122.1, 37.7)
        pixel_window = _bbox_to_window(georef_raster_2555, bb)
        a = open_geotiff(georef_raster_2555, bbox=bb)
        b = open_geotiff(georef_raster_2555, window=pixel_window)
        assert a.shape == b.shape
        np.testing.assert_array_equal(a.values, b.values)
        # Pixel-value parity against the same slice of the full read.
        full = open_geotiff(georef_raster_2555)
        r0, c0, r1, c1 = pixel_window
        np.testing.assert_array_equal(a.values, full.values[r0:r1, c0:c1])

    def test_bbox_clamps_to_file_extent(self, georef_raster_2555):
        """An out-of-extent bbox clamps rather than erroring, as long
        as it overlaps the file at all."""
        full = open_geotiff(georef_raster_2555)
        # bbox extends west / south past the file; only the eastern /
        # northern portion intersects.
        bb = (-123.0, 36.5, -122.25, 37.65)
        sub = open_geotiff(georef_raster_2555, bbox=bb)
        # Result must be a strict sub-region (smaller than full) but
        # non-empty.
        assert 0 < sub.shape[0] < full.shape[0]
        assert 0 < sub.shape[1] < full.shape[1]


class TestBboxValidation:
    def test_bbox_and_window_are_mutually_exclusive(self, georef_raster_2555):
        with pytest.raises(ValueError, match="mutually exclusive"):
            open_geotiff(
                georef_raster_2555,
                window=(0, 0, 10, 10),
                bbox=(-122.4, 37.4, -122.1, 37.7),
            )

    def test_bbox_wrong_arity_rejected(self, georef_raster_2555):
        with pytest.raises(ValueError, match="4-tuple"):
            open_geotiff(georef_raster_2555, bbox=(-122.4, 37.4, -122.1))

    def test_bbox_not_a_sequence_rejected(self, georef_raster_2555):
        with pytest.raises(ValueError, match="4-tuple"):
            open_geotiff(georef_raster_2555, bbox="not-a-bbox")

    def test_bbox_inverted_x_rejected(self, georef_raster_2555):
        with pytest.raises(ValueError, match="non-positive size"):
            open_geotiff(
                georef_raster_2555,
                bbox=(-122.1, 37.4, -122.4, 37.7),  # x_min > x_max
            )

    def test_bbox_inverted_y_rejected(self, georef_raster_2555):
        with pytest.raises(ValueError, match="non-positive size"):
            open_geotiff(
                georef_raster_2555,
                bbox=(-122.4, 37.7, -122.1, 37.4),  # y_min > y_max
            )

    def test_bbox_outside_extent_rejected(self, georef_raster_2555):
        """A bbox that does not overlap the file's extent at all
        raises rather than returning an empty array."""
        with pytest.raises(ValueError, match="does not overlap"):
            open_geotiff(
                georef_raster_2555,
                bbox=(10.0, 10.0, 20.0, 20.0),  # eastern hemisphere
            )

    def test_bbox_requires_georef(self, tmp_path):
        """A file with no GeoTIFF tags is rejected with a clear error
        rather than silently producing nonsense pixel windows."""
        from xrspatial.geotiff.tests.conftest import make_minimal_tiff
        data = make_minimal_tiff(10, 10, np.dtype('float32'))
        path = str(tmp_path / 'plain_2555.tif')
        with open(path, 'wb') as f:
            f.write(data)
        with pytest.raises(ValueError, match="no georeferencing"):
            open_geotiff(path, bbox=(-1.0, -1.0, 1.0, 1.0))

    def test_bbox_rejects_nan(self, georef_raster_2555):
        """NaN coordinates pass the ordering check (NaN >= NaN is
        False), so reject them upfront with a clear error rather than
        surfacing the downstream integer-cast failure."""
        import math
        for bad in (
            (math.nan, 37.4, -122.1, 37.7),
            (-122.4, math.nan, -122.1, 37.7),
            (-122.4, 37.4, math.nan, 37.7),
            (-122.4, 37.4, -122.1, math.nan),
        ):
            with pytest.raises(ValueError, match="finite coordinates"):
                open_geotiff(georef_raster_2555, bbox=bad)

    def test_bbox_rejects_inf(self, georef_raster_2555):
        """Infinite coordinates are equally meaningless."""
        with pytest.raises(ValueError, match="finite coordinates"):
            open_geotiff(
                georef_raster_2555,
                bbox=(float('-inf'), 37.4, -122.1, 37.7),
            )

    def test_bbox_rejects_rotated_affine(self, tmp_path):
        """A file with a rotated affine is rejected with guidance to
        pass allow_rotated=True (which clears the rotation) and re-call."""
        from xrspatial.geotiff.tests.read.test_crs import _write_rotated_tiff
        arr = np.arange(20, dtype='<u2').reshape(4, 5)
        path = str(tmp_path / 'rotated_2555.tif')
        _write_rotated_tiff(path, arr)
        # ``allow_rotated=True`` is required to even open the file;
        # without it the file is rejected upstream. With it, the
        # transform's rotated_affine is set and bbox= still refuses.
        with pytest.raises(ValueError, match="rotated affine"):
            open_geotiff(
                path,
                bbox=(0.0, 0.0, 10.0, 10.0),
                allow_rotated=True,
            )


class TestBboxOverviewLevel:
    def test_bbox_with_overview_level_uses_overview_transform(self, tmp_path):
        """``overview_level=`` selects a downsampled IFD whose transform
        has different pixel sizes. ``bbox=`` must resolve against the
        overview's transform, not the base IFD's, so the resulting pixel
        window matches the overview's dimensions."""
        # Build a 256x256 COG with 2x and 4x reductions. Coarser pixels
        # at deeper overview levels give the bbox a smaller pixel
        # window for the same geographic extent.
        base = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        xs = np.linspace(-122.5, -122.0, 256)
        ys = np.linspace(37.8, 37.3, 256)
        arr = xr.DataArray(
            base, dims=('y', 'x'),
            coords={'y': ys, 'x': xs},
            attrs={'crs': 'EPSG:4326'},
        )
        path = str(tmp_path / 'overview_2555.tif')
        to_geotiff(arr, path, cog=True, overview_levels=[2, 4])

        bb = (-122.4, 37.5, -122.1, 37.7)
        # The overview-level read should succeed and return a sub-region
        # bounded by the overview's smaller pixel grid, not the base
        # IFD's 256x256 grid.
        sub_base = open_geotiff(path, bbox=bb, overview_level=0)
        sub_ov = open_geotiff(path, bbox=bb, overview_level=1)
        sub_ov2 = open_geotiff(path, bbox=bb, overview_level=2)
        # Each successive overview level halves the resolution, so the
        # bbox slice should shrink monotonically with level.
        assert sub_base.shape[0] >= sub_ov.shape[0] >= sub_ov2.shape[0]
        assert sub_base.shape[1] >= sub_ov.shape[1] >= sub_ov2.shape[1]
        # All slices must be non-empty.
        assert sub_ov2.shape[0] > 0 and sub_ov2.shape[1] > 0


class TestBboxBackendParity:
    """``bbox=`` resolves to a pixel window at the dispatcher and is
    forwarded to every backend. The eager-numpy path is covered above;
    these cells assert the GPU, dask+numpy, and dask+gpu paths consume
    the bbox-derived window and return pixels identical to the eager
    read. Without them a regression in the bbox->window->backend
    forwarding on any non-eager path ships undetected.
    """

    _BB = (-122.4, 37.4, -122.1, 37.7)

    @staticmethod
    def _materialise(da):
        raw = da.data
        if hasattr(raw, "compute"):
            raw = raw.compute()
        if hasattr(raw, "get"):
            raw = raw.get()
        return np.asarray(raw)

    @requires_gpu
    def test_bbox_gpu_matches_eager(self, georef_raster_2555):
        eager = open_geotiff(georef_raster_2555, bbox=self._BB)
        gpu = open_geotiff(georef_raster_2555, bbox=self._BB, gpu=True)
        assert gpu.shape == eager.shape
        np.testing.assert_array_equal(
            self._materialise(gpu), eager.values)

    def test_bbox_dask_matches_eager(self, georef_raster_2555):
        eager = open_geotiff(georef_raster_2555, bbox=self._BB)
        dask = open_geotiff(georef_raster_2555, bbox=self._BB, chunks=16)
        assert dask.shape == eager.shape
        np.testing.assert_array_equal(
            self._materialise(dask), eager.values)

    @requires_gpu
    def test_bbox_dask_gpu_matches_eager(self, georef_raster_2555):
        eager = open_geotiff(georef_raster_2555, bbox=self._BB)
        dask_gpu = open_geotiff(
            georef_raster_2555, bbox=self._BB, gpu=True, chunks=16)
        assert dask_gpu.shape == eager.shape
        np.testing.assert_array_equal(
            self._materialise(dask_gpu), eager.values)


class TestSafetyHintMentionsBbox:
    def test_hint_includes_bbox_line(self):
        """The PixelSafetyLimitError recovery hint must surface the
        new bbox= option alongside window= and chunks=."""
        from xrspatial.geotiff._layout import (MAX_PIXELS_DEFAULT, PixelSafetyLimitError,
                                               _check_dimensions)
        with pytest.raises(PixelSafetyLimitError) as exc:
            _check_dimensions(50_000, 50_000, 1, MAX_PIXELS_DEFAULT)
        msg = str(exc.value)
        assert "bbox=(x_min, y_min, x_max, y_max)" in msg
        assert "in the file's CRS" in msg
        # Previous options must still be present.
        assert "window=(r0, c0, r1, c1)" in msg
        assert "chunks=" in msg
