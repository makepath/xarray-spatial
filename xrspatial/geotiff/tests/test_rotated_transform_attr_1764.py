"""Regression tests for #1764.

``_transform_from_attr`` previously unpacked the rasterio ``Affine``
6-tuple as ``a, _b, c, _d, e, f`` and silently discarded the rotation
(``b``) and shear (``d``) coefficients. A user passing an
``attrs['transform']`` from a rotated raster would get an axis-aligned
GeoTIFF written at the wrong location with no warning.

The helper now rejects any transform whose ``b`` or ``d`` exceeds a
small float tolerance (1e-12). Axis-aligned transforms still pass
through unchanged so existing round-trip behaviour is preserved.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _transform_from_attr, open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import GeoTransform


def _make_da(transform_tuple):
    arr = np.zeros((4, 5), dtype=np.float32)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": np.arange(4, dtype=np.float64),
            "x": np.arange(5, dtype=np.float64),
        },
        attrs={"transform": transform_tuple, "crs": 4326},
    )


class TestTransformFromAttrRejection:
    """Direct tests on the helper."""

    def test_rotation_b_nonzero_raises(self):
        # b=0.5 -> meaningful rotation
        t = (1.0, 0.5, 100.0, 0.0, -1.0, 200.0)
        with pytest.raises(ValueError, match="rotation/shear"):
            _transform_from_attr(t)

    def test_shear_d_nonzero_raises(self):
        # d=0.25 -> meaningful shear
        t = (1.0, 0.0, 100.0, 0.25, -1.0, 200.0)
        with pytest.raises(ValueError, match="rotation/shear"):
            _transform_from_attr(t)

    def test_both_b_and_d_nonzero_raises(self):
        t = (1.0, 0.7, 100.0, -0.3, -1.0, 200.0)
        with pytest.raises(ValueError) as exc_info:
            _transform_from_attr(t)
        msg = str(exc_info.value)
        # Error message names the offending values
        assert "0.7" in msg
        assert "-0.3" in msg

    def test_axis_aligned_passes(self):
        t = (30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)
        gt = _transform_from_attr(t)
        assert isinstance(gt, GeoTransform)
        assert gt.pixel_width == 30.0
        assert gt.pixel_height == -30.0
        assert gt.origin_x == 500000.0
        assert gt.origin_y == 4000000.0

    def test_float_noise_at_zero_accepted(self):
        # 1e-15 is well below the 1e-12 tolerance: should pass
        t = (1.0, 1e-15, 100.0, -1e-15, -1.0, 200.0)
        gt = _transform_from_attr(t)
        assert isinstance(gt, GeoTransform)
        assert gt.origin_x == 100.0

    def test_tolerance_boundary_rejected(self):
        # 1e-10 is well above the 1e-12 tolerance: should be rejected
        t = (1.0, 1e-10, 100.0, 0.0, -1.0, 200.0)
        with pytest.raises(ValueError, match="rotation/shear"):
            _transform_from_attr(t)

    def test_geotransform_instance_passes_through(self):
        gt_in = GeoTransform(
            origin_x=1.0, origin_y=2.0, pixel_width=0.1, pixel_height=-0.1,
        )
        gt_out = _transform_from_attr(gt_in)
        assert gt_out is gt_in

    def test_none_returns_none(self):
        assert _transform_from_attr(None) is None

    def test_wrong_length_returns_none(self):
        # Preserve existing behaviour: invalid shape is silently ignored
        assert _transform_from_attr((1.0, 0.0, 100.0)) is None


class TestToGeotiffRejectsRotated:
    """End-to-end: to_geotiff surfaces the ValueError."""

    def test_to_geotiff_rotated_raises(self, tmp_path):
        da = _make_da((1.0, 0.5, 100.0, 0.0, -1.0, 200.0))
        out = str(tmp_path / "rotated_1764.tif")
        with pytest.raises(ValueError, match="rotation/shear"):
            to_geotiff(da, out, compression="none")

    def test_to_geotiff_skewed_raises(self, tmp_path):
        da = _make_da((1.0, 0.0, 100.0, 0.4, -1.0, 200.0))
        out = str(tmp_path / "skewed_1764.tif")
        with pytest.raises(ValueError, match="rotation/shear"):
            to_geotiff(da, out, compression="none")

    def test_to_geotiff_axis_aligned_still_writes(self, tmp_path):
        da = _make_da((30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0))
        out = str(tmp_path / "axis_aligned_1764.tif")
        to_geotiff(da, out, compression="none")
        # Round-trip: read back and confirm the transform survived
        result = open_geotiff(out)
        a, b, c, d, e, f = result.attrs["transform"]
        assert b == 0.0 and d == 0.0
        assert a == pytest.approx(30.0)
        assert e == pytest.approx(-30.0)
        assert c == pytest.approx(500000.0)
        assert f == pytest.approx(4000000.0)
