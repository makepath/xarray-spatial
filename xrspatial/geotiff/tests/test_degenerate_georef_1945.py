"""Georeferenced 1xN / Nx1 / 1x1 writes must preserve their transform.

Issue #1945: ``coords_to_transform`` returned ``None`` whenever either
spatial coord array had length < 2, and ``to_geotiff`` /
``write_geotiff_gpu`` then fell through and produced a non-georeferenced
TIFF. A DataArray with real x/y coords round-tripped back with integer
pixel coords and no ``transform`` attribute, so the spatial metadata
silently disappeared.

These tests pin the round-trip invariant for the three degenerate
georeferenced shapes (1xN, Nx1, 1x1) across the eager numpy, dask+numpy,
GPU (cupy), and dask+cupy writers, plus the VRT writer.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (
    open_geotiff,
    to_geotiff,
    write_geotiff_gpu,
)


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")


# ---------------------------------------------------------------------------
# Reference georeferenced DataArrays.
#
# A 1xN strip: one row at y=45.0, 8 columns spanning -120.0..-113.0 (1° step).
# An Nx1 profile: one column at x=-120.0, 8 rows spanning 45.0..38.0 (-1° step).
# A 1x1 pixel: must carry its transform on ``attrs['transform']`` because the
# coord arrays alone cannot pin a pixel size in this case.
# ---------------------------------------------------------------------------

PIXEL = 1.0
X0 = -120.0
Y0 = 45.0


def _strip_1xN(raster_type: str = "area") -> xr.DataArray:
    # The borrow-from-other-axis fallback (#1945) is now opt-in (#2214).
    # These tests document the borrow path, so they set the opt-in flag.
    # Round-trip semantics are unchanged once the flag is set.
    attrs = {"crs": 4326, "assume_square_pixels_for_degenerate_axis": True}
    if raster_type == "point":
        attrs["raster_type"] = "point"
    return xr.DataArray(
        np.arange(8, dtype="float32").reshape(1, 8),
        dims=("y", "x"),
        coords={
            "x": X0 + np.arange(8, dtype="float64") * PIXEL,
            "y": np.array([Y0], dtype="float64"),
        },
        attrs=attrs,
    )


def _strip_Nx1(raster_type: str = "area") -> xr.DataArray:
    attrs = {"crs": 4326, "assume_square_pixels_for_degenerate_axis": True}
    if raster_type == "point":
        attrs["raster_type"] = "point"
    return xr.DataArray(
        np.arange(8, dtype="float32").reshape(8, 1),
        dims=("y", "x"),
        coords={
            "x": np.array([X0], dtype="float64"),
            "y": Y0 - np.arange(8, dtype="float64") * PIXEL,
        },
        attrs=attrs,
    )


def _pixel_1x1_with_transform() -> xr.DataArray:
    # rasterio-style 6-tuple: (a, b, c, d, e, f) = (px, 0, ox, 0, py, oy)
    transform = (PIXEL, 0.0, X0 - PIXEL * 0.5, 0.0, -PIXEL, Y0 + PIXEL * 0.5)
    return xr.DataArray(
        np.array([[42.0]], dtype="float32"),
        dims=("y", "x"),
        coords={"x": np.array([X0]), "y": np.array([Y0])},
        attrs={"crs": 4326, "transform": transform},
    )


def _pixel_1x1_no_transform() -> xr.DataArray:
    return xr.DataArray(
        np.array([[42.0]], dtype="float32"),
        dims=("y", "x"),
        coords={"x": np.array([X0]), "y": np.array([Y0])},
        attrs={"crs": 4326},
    )


def _assert_round_trip(path: str, expected: xr.DataArray) -> None:
    """Read back ``path`` and check coords/transform survived the write."""
    r = open_geotiff(path)

    # Coordinates must be float (georeferenced), not integer pixel indices.
    assert np.issubdtype(r.coords["x"].dtype, np.floating), (
        f"x coord came back as {r.coords['x'].dtype}; "
        "writer stripped georeferencing"
    )
    assert np.issubdtype(r.coords["y"].dtype, np.floating), (
        f"y coord came back as {r.coords['y'].dtype}; "
        "writer stripped georeferencing"
    )

    np.testing.assert_allclose(
        r.coords["x"].values, expected.coords["x"].values, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        r.coords["y"].values, expected.coords["y"].values, rtol=0, atol=1e-9
    )

    assert r.attrs.get("transform") is not None, "transform missing from attrs"

    np.testing.assert_array_equal(np.asarray(r.values), np.asarray(expected.values))


# ===========================================================================
# Eager numpy writer (to_geotiff)
# ===========================================================================

class TestEagerWriterDegenerateGeoref:

    def test_1xN_round_trip_preserves_transform(self, tmp_path):
        da = _strip_1xN()
        p = str(tmp_path / "georef_1xN_eager_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip(p, da)

    def test_Nx1_round_trip_preserves_transform(self, tmp_path):
        da = _strip_Nx1()
        p = str(tmp_path / "georef_Nx1_eager_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip(p, da)

    def test_1x1_with_transform_attr_round_trips(self, tmp_path):
        da = _pixel_1x1_with_transform()
        p = str(tmp_path / "georef_1x1_eager_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip(p, da)

    def test_1x1_without_transform_raises(self, tmp_path):
        """1x1 cannot recover pixel size from coords alone; raise rather
        than silently strip georeferencing."""
        da = _pixel_1x1_no_transform()
        p = str(tmp_path / "georef_1x1_no_tx_eager_1945.tif")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            to_geotiff(da, p)


# ===========================================================================
# Dask + numpy writer
# ===========================================================================

class TestDaskNumpyWriterDegenerateGeoref:

    def test_1xN_dask_round_trip(self, tmp_path):
        da = _strip_1xN().chunk({"x": 4, "y": 1})
        p = str(tmp_path / "georef_1xN_dask_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip(p, _strip_1xN())

    def test_Nx1_dask_round_trip(self, tmp_path):
        da = _strip_Nx1().chunk({"x": 1, "y": 4})
        p = str(tmp_path / "georef_Nx1_dask_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip(p, _strip_Nx1())


# ===========================================================================
# GPU writer (write_geotiff_gpu)
# ===========================================================================

@_gpu_only
class TestGpuWriterDegenerateGeoref:

    def test_1xN_round_trip_preserves_transform(self, tmp_path):
        import cupy
        da_cpu = _strip_1xN()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_1xN_gpu_1945.tif")
        write_geotiff_gpu(da_gpu, p)
        _assert_round_trip(p, da_cpu)

    def test_Nx1_round_trip_preserves_transform(self, tmp_path):
        import cupy
        da_cpu = _strip_Nx1()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_Nx1_gpu_1945.tif")
        write_geotiff_gpu(da_gpu, p)
        _assert_round_trip(p, da_cpu)

    def test_1x1_with_transform_attr_round_trips(self, tmp_path):
        import cupy
        da_cpu = _pixel_1x1_with_transform()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_1x1_gpu_1945.tif")
        write_geotiff_gpu(da_gpu, p)
        _assert_round_trip(p, da_cpu)

    def test_1x1_without_transform_raises(self, tmp_path):
        import cupy
        da_cpu = _pixel_1x1_no_transform()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_1x1_no_tx_gpu_1945.tif")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            write_geotiff_gpu(da_gpu, p)


# ===========================================================================
# Dask + cupy writer (to_geotiff with chunked cupy array)
# ===========================================================================

@_gpu_only
class TestDaskCupyWriterDegenerateGeoref:

    def test_1xN_dask_cupy_round_trip(self, tmp_path):
        import cupy
        da_cpu = _strip_1xN()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu = da_gpu.chunk({"x": 4, "y": 1})
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_1xN_dask_cupy_1945.tif")
        to_geotiff(da_gpu, p)
        _assert_round_trip(p, da_cpu)

    def test_Nx1_dask_cupy_round_trip(self, tmp_path):
        import cupy
        da_cpu = _strip_Nx1()
        da_gpu = da_cpu.copy(data=cupy.asarray(da_cpu.values))
        da_gpu = da_gpu.chunk({"x": 1, "y": 4})
        da_gpu.attrs = dict(da_cpu.attrs)
        p = str(tmp_path / "georef_Nx1_dask_cupy_1945.tif")
        to_geotiff(da_gpu, p)
        _assert_round_trip(p, da_cpu)


# ===========================================================================
# VRT writer
# ===========================================================================

class TestVrtTiledWriterDegenerateGeoref:
    """``to_geotiff(da, '*.vrt')`` dispatches to ``_write_vrt_tiled``."""

    def test_1xN_vrt_round_trip(self, tmp_path):
        da = _strip_1xN()
        p = str(tmp_path / "georef_1xN_1945.vrt")
        to_geotiff(da, p)
        _assert_round_trip(p, da)

    def test_Nx1_vrt_round_trip(self, tmp_path):
        da = _strip_Nx1()
        p = str(tmp_path / "georef_Nx1_1945.vrt")
        to_geotiff(da, p)
        _assert_round_trip(p, da)

    def test_1x1_vrt_without_transform_raises(self, tmp_path):
        da = _pixel_1x1_no_transform()
        p = str(tmp_path / "georef_1x1_no_tx_1945.vrt")
        with pytest.raises(ValueError, match="(?i)pixel size|transform"):
            to_geotiff(da, p)


# ===========================================================================
# raster_type='point' round trip
#
# Pins the no-half-pixel-shift path on 1xN / Nx1 degenerate writes. With
# PixelIsPoint the tiepoint already sits at the pixel center, so the
# writer must not add a 0.5*pixel offset when synthesising the transform
# from coords.
# ===========================================================================

class TestEagerWriterPointRasterDegenerate:

    @pytest.mark.parametrize(
        "factory, name",
        [(_strip_1xN, "1xN"), (_strip_Nx1, "Nx1")],
    )
    def test_point_raster_round_trip(self, tmp_path, factory, name):
        da = factory(raster_type="point")
        p = str(tmp_path / f"georef_{name}_point_eager_1945.tif")
        to_geotiff(da, p)
        _assert_round_trip(p, da)


# ===========================================================================
# Sign pinning on borrowed axis (#1953 review)
#
# When one axis is degenerate, ``coords_to_transform`` borrows the pixel
# size magnitude from the non-degenerate axis and applies a convention
# for the borrowed axis: pixel_width defaults positive (x increases
# left-to-right), pixel_height defaults negative (y decreases top-to-
# bottom). This strips the sign of the *source* axis if it doesn't match
# convention. These tests pin that behaviour so a future refactor does
# not silently change it.
# ===========================================================================

class TestCoordsToTransformBorrowSignPinning:
    """Pin the sign convention of the borrowed (degenerate) axis."""

    def test_y_ascending_Nx1_borrows_positive_pixel_width(self):
        from xrspatial.geotiff._coords import coords_to_transform

        da = xr.DataArray(
            np.arange(8, dtype="float32").reshape(8, 1),
            dims=("y", "x"),
            coords={
                "x": np.array([X0], dtype="float64"),
                # y ascending: 38, 39, 40, ... so pixel_height = +1.0
                "y": 38.0 + np.arange(8, dtype="float64") * PIXEL,
            },
            # Opt in to the borrow-from-other-axis path (#2214).
            attrs={"assume_square_pixels_for_degenerate_axis": True},
        )
        t = coords_to_transform(da)
        # Non-degenerate axis (y) keeps its sign.
        assert t.pixel_height == pytest.approx(1.0)
        # Borrowed pixel_width takes ``abs(pixel_height)``: always positive,
        # regardless of y's sign. This pins the current convention.
        assert t.pixel_width == pytest.approx(1.0)

    def test_x_descending_1xN_borrows_negative_pixel_height(self):
        from xrspatial.geotiff._coords import coords_to_transform

        da = xr.DataArray(
            np.arange(8, dtype="float32").reshape(1, 8),
            dims=("y", "x"),
            coords={
                # x descending: pixel_width = -1.0
                "x": -113.0 - np.arange(8, dtype="float64") * PIXEL,
                "y": np.array([Y0], dtype="float64"),
            },
            attrs={"assume_square_pixels_for_degenerate_axis": True},
        )
        t = coords_to_transform(da)
        # Non-degenerate axis (x) keeps its sign.
        assert t.pixel_width == pytest.approx(-1.0)
        # Borrowed pixel_height takes ``-abs(pixel_width)``: always negative,
        # regardless of x's sign. This pins the current convention.
        assert t.pixel_height == pytest.approx(-1.0)
