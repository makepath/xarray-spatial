"""Regression tests for issue #1642.

PR #1641 (issue #1640) inherits level-0 georef on overview reads, but
keeps the level-0 ``origin_x`` / ``origin_y`` unchanged. That is
correct for the default ``PixelIsArea`` raster_type, where the origin
is the upper-left corner of pixel (0, 0). It is *wrong* for
``PixelIsPoint`` (GeoKey 1025 = 2), where the origin is the **center**
of pixel (0, 0): an overview pixel that spans the first ``scale_x``
columns of level 0 has its center at the centroid of those level-0
pixels, i.e. ``origin + (scale - 1) * 0.5 * pixel_size_lvl0`` along
each axis.

These tests pin the corrected behaviour across all four backends and
sanity-check the ``PixelIsArea`` path so the fix doesn't regress
issue #1640 itself.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._geotags import (
    GeoTransform,
    RASTER_PIXEL_IS_AREA,
    RASTER_PIXEL_IS_POINT,
    GeoInfo,
    extract_geo_info_with_overview_inheritance,
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


def _make_pp_cog(path: str, size: int = 1024, pixel: float = 10.0) -> xr.DataArray:
    """Write a PixelIsPoint COG with three overview levels.

    Pixel (0, 0) is centred at world (0, 0); each pixel is ``pixel``
    units wide. EPSG:32610 keeps the writer happy without inventing
    geographic semantics. Returns the source DataArray.
    """
    arr = np.arange(size * size, dtype=np.float32).reshape(size, size)
    x = np.arange(size, dtype=np.float64) * pixel
    y = -(np.arange(size, dtype=np.float64) * pixel)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 'EPSG:32610',
                             'raster_type': 'point'})
    to_geotiff(da, path, cog=True, overview_levels=[2, 4, 8])
    return da


def _make_pa_cog(path: str, size: int = 1024, pixel: float = 10.0) -> xr.DataArray:
    """PixelIsArea companion of :func:`_make_pp_cog` for regression checks."""
    arr = np.arange(size * size, dtype=np.float32).reshape(size, size)
    x = np.arange(size, dtype=np.float64) * pixel + 0.5 * pixel
    y = -(np.arange(size, dtype=np.float64) * pixel + 0.5 * pixel)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 'EPSG:32610'})
    to_geotiff(da, path, cog=True, overview_levels=[2, 4, 8])
    return da


# ---------------------------------------------------------------------------
# End-to-end cross-backend tests on a real COG.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_kwargs", [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 256}, id="dask+numpy"),
    pytest.param({"gpu": True}, id="cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
    pytest.param({"gpu": True, "chunks": 256}, id="dask+cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
])
def test_point_overview_first_pixel_center_at_block_centroid(tmp_path,
                                                             backend_kwargs):
    """For PixelIsPoint COGs, overview pixel-0 center is the level-0 centroid."""
    path = str(tmp_path / "pp_1642_centroid.tif")
    _make_pp_cog(path)

    # Level-0 pixel (0, 0) center is at (0, 0). Level-1 pixel (0, 0)
    # covers level-0 pixels with centers at x=0 and x=10, y=0 and y=-10;
    # the centroid is (5, -5). Level-2 covers level-0 0..3 with centers
    # 0, 10, 20, 30 -> centroid 15; same for y -> -15. Level-3 covers
    # 0..7 -> centroid 35; y -> -35.
    expected = {
        1: (5.0, -5.0),
        2: (15.0, -15.0),
        3: (35.0, -35.0),
    }
    for lvl, (exp_x, exp_y) in expected.items():
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        x0 = float(np.asarray(da.coords['x'])[0])
        y0 = float(np.asarray(da.coords['y'])[0])
        assert abs(x0 - exp_x) < 1e-9, (
            f"backend={backend_kwargs}, lvl={lvl}: first pixel x={x0}, "
            f"expected {exp_x}")
        assert abs(y0 - exp_y) < 1e-9, (
            f"backend={backend_kwargs}, lvl={lvl}: first pixel y={y0}, "
            f"expected {exp_y}")
        # raster_type is inherited from level 0.
        assert da.attrs.get('raster_type') == 'point'


@pytest.mark.parametrize("backend_kwargs", [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 256}, id="dask+numpy"),
    pytest.param({"gpu": True}, id="cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
    pytest.param({"gpu": True, "chunks": 256}, id="dask+cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
])
def test_point_overview_transform_origin_shifted(tmp_path, backend_kwargs):
    """``transform`` attr carries the shifted origin for PixelIsPoint overviews."""
    path = str(tmp_path / "pp_1642_transform.tif")
    _make_pp_cog(path)

    # transform tuple: (pixel_width, 0, origin_x, 0, pixel_height, origin_y).
    expected = {
        0: (10.0, 0.0, -10.0, 0.0),    # level 0: no shift
        1: (20.0, 5.0, -20.0, -5.0),
        2: (40.0, 15.0, -40.0, -15.0),
        3: (80.0, 35.0, -80.0, -35.0),
    }
    for lvl, (exp_pw, exp_ox, exp_ph, exp_oy) in expected.items():
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        t = da.attrs['transform']
        assert abs(t[0] - exp_pw) < 1e-9, (
            f"lvl={lvl}: pixel_width={t[0]}, expected {exp_pw}")
        assert abs(t[2] - exp_ox) < 1e-9, (
            f"lvl={lvl}: origin_x={t[2]}, expected {exp_ox}")
        assert abs(t[4] - exp_ph) < 1e-9, (
            f"lvl={lvl}: pixel_height={t[4]}, expected {exp_ph}")
        assert abs(t[5] - exp_oy) < 1e-9, (
            f"lvl={lvl}: origin_y={t[5]}, expected {exp_oy}")


def test_point_overview_coords_are_uniform(tmp_path):
    """The shifted origin still yields a regular grid -- step == pixel size."""
    path = str(tmp_path / "pp_1642_uniform.tif")
    _make_pp_cog(path)

    for lvl in (0, 1, 2, 3):
        da = open_geotiff(path, overview_level=lvl)
        x = np.asarray(da.coords['x'])
        y = np.asarray(da.coords['y'])
        if x.size > 1:
            dx = np.diff(x)
            assert np.allclose(dx, dx[0], atol=1e-9), (
                f"lvl={lvl}: x is non-uniform: {dx[:5]}")
            assert abs(float(dx[0]) - da.attrs['transform'][0]) < 1e-9
        if y.size > 1:
            dy = np.diff(y)
            assert np.allclose(dy, dy[0], atol=1e-9)
            assert abs(float(dy[0]) - da.attrs['transform'][4]) < 1e-9


# ---------------------------------------------------------------------------
# Direct helper test: synthetic GeoInfo round-trips.
# ---------------------------------------------------------------------------

class _StubIFD:
    """Minimal IFD-like stub for unit-testing the helper directly."""
    def __init__(self, subfile_type: int, width: int, height: int):
        self.subfile_type = subfile_type
        self.width = width
        self.height = height


def test_helper_pixel_is_point_origin_shift_unit(monkeypatch):
    """Unit-level: the helper applies the correct origin shift for
    PixelIsPoint inheritance, given a stubbed ``extract_geo_info``.

    This exercises the math without going through the writer/reader so a
    regression in the formula is caught even if the COG pipeline changes.
    """
    from xrspatial.geotiff import _geotags as _gt

    base_ifd = _StubIFD(subfile_type=0, width=1024, height=1024)
    ov_ifd = _StubIFD(subfile_type=1, width=128, height=128)

    base_info = GeoInfo(
        transform=GeoTransform(origin_x=0.0, origin_y=0.0,
                               pixel_width=10.0, pixel_height=-10.0),
        has_georef=True,
        raster_type=RASTER_PIXEL_IS_POINT,
        crs_epsg=32610,
    )
    ov_info = GeoInfo(has_georef=False,
                      raster_type=RASTER_PIXEL_IS_AREA)

    calls = {'count': 0}

    def fake_extract(ifd, data, byte_order, *, allow_rotated=False):
        calls['count'] += 1
        if ifd is base_ifd:
            return base_info
        return ov_info

    monkeypatch.setattr(_gt, 'extract_geo_info', fake_extract)

    out = extract_geo_info_with_overview_inheritance(
        ov_ifd, [base_ifd, ov_ifd], b'', '<')

    # scale = 1024 / 128 = 8. Shift = (8 - 1) * 0.5 * 10 = 35.
    assert abs(out.transform.pixel_width - 80.0) < 1e-9
    assert abs(out.transform.pixel_height - (-80.0)) < 1e-9
    assert abs(out.transform.origin_x - 35.0) < 1e-9
    assert abs(out.transform.origin_y - (-35.0)) < 1e-9
    assert out.raster_type == RASTER_PIXEL_IS_POINT
    assert out.has_georef


def test_helper_pixel_is_area_no_origin_shift_unit(monkeypatch):
    """PixelIsArea path is unchanged: origin stays at the level-0 corner."""
    from xrspatial.geotiff import _geotags as _gt

    base_ifd = _StubIFD(subfile_type=0, width=1024, height=1024)
    ov_ifd = _StubIFD(subfile_type=1, width=128, height=128)

    base_info = GeoInfo(
        transform=GeoTransform(origin_x=100.0, origin_y=200.0,
                               pixel_width=0.5, pixel_height=-0.5),
        has_georef=True,
        raster_type=RASTER_PIXEL_IS_AREA,
        crs_epsg=4326,
    )
    ov_info = GeoInfo(has_georef=False, raster_type=RASTER_PIXEL_IS_AREA)

    def fake_extract(ifd, data, byte_order, *, allow_rotated=False):
        return base_info if ifd is base_ifd else ov_info

    monkeypatch.setattr(_gt, 'extract_geo_info', fake_extract)

    out = extract_geo_info_with_overview_inheritance(
        ov_ifd, [base_ifd, ov_ifd], b'', '<')

    assert abs(out.transform.origin_x - 100.0) < 1e-9
    assert abs(out.transform.origin_y - 200.0) < 1e-9
    assert abs(out.transform.pixel_width - 4.0) < 1e-9
    assert abs(out.transform.pixel_height - (-4.0)) < 1e-9
    assert out.raster_type == RASTER_PIXEL_IS_AREA


def test_helper_point_overview_with_own_geokeys_not_shifted(monkeypatch):
    """If the overview IFD carries its own georef, the shift never fires."""
    from xrspatial.geotiff import _geotags as _gt

    base_ifd = _StubIFD(subfile_type=0, width=1024, height=1024)
    ov_ifd = _StubIFD(subfile_type=1, width=128, height=128)

    base_info = GeoInfo(
        transform=GeoTransform(origin_x=0.0, origin_y=0.0,
                               pixel_width=10.0, pixel_height=-10.0),
        has_georef=True, raster_type=RASTER_PIXEL_IS_POINT, crs_epsg=32610,
    )
    own_ov_info = GeoInfo(
        transform=GeoTransform(origin_x=999.0, origin_y=-999.0,
                               pixel_width=77.0, pixel_height=-77.0),
        has_georef=True, raster_type=RASTER_PIXEL_IS_POINT, crs_epsg=32610,
    )

    def fake_extract(ifd, data, byte_order, *, allow_rotated=False):
        return base_info if ifd is base_ifd else own_ov_info

    monkeypatch.setattr(_gt, 'extract_geo_info', fake_extract)

    out = extract_geo_info_with_overview_inheritance(
        ov_ifd, [base_ifd, ov_ifd], b'', '<')

    # Untouched: helper returns the overview's own info verbatim.
    assert abs(out.transform.origin_x - 999.0) < 1e-9
    assert abs(out.transform.origin_y - (-999.0)) < 1e-9
    assert abs(out.transform.pixel_width - 77.0) < 1e-9


# ---------------------------------------------------------------------------
# PixelIsArea regression: ensure the #1640 fix still passes.
# ---------------------------------------------------------------------------

def test_area_overview_origin_unchanged_regression(tmp_path):
    """PixelIsArea overview origin must still equal level-0 origin (#1640)."""
    path = str(tmp_path / "pa_1642_regression.tif")
    _make_pa_cog(path)
    base = open_geotiff(path, overview_level=0)
    base_t = base.attrs['transform']
    for lvl in (1, 2, 3):
        da = open_geotiff(path, overview_level=lvl)
        t = da.attrs['transform']
        assert abs(t[2] - base_t[2]) < 1e-9, (
            f"PixelIsArea lvl={lvl}: origin_x drifted from {base_t[2]} "
            f"to {t[2]}")
        assert abs(t[5] - base_t[5]) < 1e-9
