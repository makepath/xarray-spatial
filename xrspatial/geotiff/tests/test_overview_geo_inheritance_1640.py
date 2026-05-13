"""Regression tests for issue #1640.

Overview IFDs in COGs typically carry no GeoKeys, ModelPixelScale, or
ModelTiepoint -- the writer puts those tags only on the level-0 IFD.
Before the fix, ``open_geotiff(path, overview_level=N)`` for ``N >= 1``
returned a DataArray whose ``transform`` attr was the default unit
transform and whose ``crs`` attr was absent. Downstream slope/aspect
ops then used pixel sizes from the (integer) coord arrays and produced
silently-wrong answers.

The fix inherits the georef from the level-0 IFD and rescales the pixel
size by the overview's reduction factor.

These tests cover all four backends (numpy, dask+numpy, cupy,
dask+cupy) and assert that the transform / crs / coords match the
level-0 read up to the expected scale factor.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


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


def _make_cog_with_overviews(path: str) -> xr.DataArray:
    """Write a 1024x1024 COG with three overview levels and known geo.

    Origin (100, 200), 0.5 unit/pixel, EPSG:4326. Returns the source
    DataArray so the caller can compare extents.
    """
    arr = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    y = np.arange(1024, dtype=np.float64) * (-0.5) + 200.0
    x = np.arange(1024, dtype=np.float64) *   0.5 + 100.0
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326})
    to_geotiff(da, path, cog=True, overview_levels=[2, 4, 8])
    return da


def _materialise(da: xr.DataArray) -> np.ndarray:
    """Return a numpy view of the data regardless of backend."""
    raw = da.data
    if hasattr(raw, 'compute'):
        raw = raw.compute()
    if hasattr(raw, 'get'):
        raw = raw.get()
    return np.asarray(raw)


@pytest.mark.parametrize("backend_kwargs", [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 128}, id="dask+numpy"),
    pytest.param({"gpu": True}, id="cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
    pytest.param({"gpu": True, "chunks": 128}, id="dask+cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
])
def test_overview_inherits_crs_across_backends(tmp_path, backend_kwargs):
    """Every backend keeps ``crs`` on overview reads."""
    path = str(tmp_path / "overview_inherit_1640_crs.tif")
    _make_cog_with_overviews(path)

    for lvl in (0, 1, 2, 3):
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        assert da.attrs.get('crs') == 4326, (
            f"backend={backend_kwargs}, overview_level={lvl}: expected "
            f"crs=4326, got {da.attrs.get('crs')!r}; full attrs="
            f"{sorted(da.attrs.keys())}"
        )


@pytest.mark.parametrize("backend_kwargs", [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 128}, id="dask+numpy"),
    pytest.param({"gpu": True}, id="cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
    pytest.param({"gpu": True, "chunks": 128}, id="dask+cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
])
def test_overview_transform_scales_by_reduction_factor(tmp_path,
                                                       backend_kwargs):
    """Pixel size doubles per overview level; origin is preserved."""
    path = str(tmp_path / "overview_inherit_1640_transform.tif")
    _make_cog_with_overviews(path)

    base = open_geotiff(path, overview_level=0, **backend_kwargs)
    base_t = base.attrs['transform']
    base_w = base.shape[-1]

    for lvl, expected_scale in ((1, 2.0), (2, 4.0), (3, 8.0)):
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        t = da.attrs.get('transform')
        assert t is not None, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"transform attr is missing")

        # The pixel-size scale is base_w / overview_w; for a power-of-two
        # COG that lands exactly on the integer reduction factor. Allow a
        # tiny tolerance for floating-point round-off only.
        ov_w = da.shape[-1]
        observed_scale = base_w / ov_w
        assert abs(observed_scale - expected_scale) < 1e-9

        # Pixel width / height should scale by the same factor.
        assert abs(t[0] - base_t[0] * expected_scale) < 1e-9, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"pixel width {t[0]} != base_pw*{expected_scale} "
            f"({base_t[0] * expected_scale})"
        )
        assert abs(t[4] - base_t[4] * expected_scale) < 1e-9, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"pixel height {t[4]} != base_ph*{expected_scale} "
            f"({base_t[4] * expected_scale})"
        )
        # Origin should not move between levels.
        assert abs(t[2] - base_t[2]) < 1e-9, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"origin_x drifted: base={base_t[2]}, ov={t[2]}")
        assert abs(t[5] - base_t[5]) < 1e-9, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"origin_y drifted: base={base_t[5]}, ov={t[5]}")


@pytest.mark.parametrize("backend_kwargs", [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 128}, id="dask+numpy"),
    pytest.param({"gpu": True}, id="cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
    pytest.param({"gpu": True, "chunks": 128}, id="dask+cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
])
def test_overview_coords_cover_same_extent(tmp_path, backend_kwargs):
    """Pixel-center coords on an overview span the same extent as level 0."""
    path = str(tmp_path / "overview_inherit_1640_coords.tif")
    _make_cog_with_overviews(path)

    base = open_geotiff(path, overview_level=0, **backend_kwargs)
    by = np.asarray(base.coords['y'])
    bx = np.asarray(base.coords['x'])

    # Total geographic extent on level 0 (PixelIsArea: edge = center +/-
    # half_pixel). Recover from coord arrays + pixel size.
    base_pw = base.attrs['transform'][0]
    base_ph = base.attrs['transform'][4]
    base_x_min = float(bx[0]) - 0.5 * base_pw
    base_x_max = float(bx[-1]) + 0.5 * base_pw
    base_y_top = float(by[0]) - 0.5 * base_ph  # base_ph is negative
    base_y_bot = float(by[-1]) + 0.5 * base_ph

    for lvl in (1, 2, 3):
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        y = np.asarray(da.coords['y'])
        x = np.asarray(da.coords['x'])
        pw = da.attrs['transform'][0]
        ph = da.attrs['transform'][4]
        x_min = float(x[0]) - 0.5 * pw
        x_max = float(x[-1]) + 0.5 * pw
        y_top = float(y[0]) - 0.5 * ph
        y_bot = float(y[-1]) + 0.5 * ph

        # Extents should agree to within one full-resolution pixel
        # (rounding of width_full / width_overview can leave a half-pixel
        # of slack at the edges for non-aligned dimensions; we use a
        # power-of-two case here so it's tighter than that).
        assert abs(x_min - base_x_min) < 1e-9, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"x_min drifted: base={base_x_min}, ov={x_min}")
        assert abs(x_max - base_x_max) < 1e-9, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"x_max drifted: base={base_x_max}, ov={x_max}")
        assert abs(y_top - base_y_top) < 1e-9, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"y_top drifted: base={base_y_top}, ov={y_top}")
        assert abs(y_bot - base_y_bot) < 1e-9, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"y_bot drifted: base={base_y_bot}, ov={y_bot}")


def test_overview_with_own_geokeys_is_not_overwritten(tmp_path):
    """If an overview IFD carries its own valid georef, keep it.

    Some writers (rasterio with ``COPY_SRC_OVERVIEWS=YES``) replicate
    geokeys on every overview. The inheritance helper must not stomp
    those; it should fall back to the parent only when the overview
    itself has no georef.

    We build a synthetic two-IFD file where the overview IFD carries its
    own ModelPixelScale + ModelTiepoint that intentionally differ from
    a naive parent-rescale. The reader must return the overview's own
    values, not the inherited ones.
    """
    tifffile = pytest.importorskip("tifffile")

    path = str(tmp_path / "overview_own_geokeys_1640.tif")

    base = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    ov = base[::2, ::2]
    # GeoKeyDirectory: header (4) + GTModelType=2 (geographic) +
    # GeographicType=4326. Same on both IFDs.
    geokeys = (1, 1, 0, 2,  1024, 0, 1, 2,  2048, 0, 1, 4326)

    with tifffile.TiffWriter(path) as tw:
        tw.write(base, tile=(32, 32),
                 extratags=[
                     (33550, 12, 3, (0.5, 0.5, 0.0)),   # ModelPixelScale
                     (33922, 12, 6,
                      (0.0, 0.0, 0.0, 100.0, 200.0, 0.0)),  # Tiepoint
                     (34735, 3, 12, geokeys),
                 ])
        tw.write(ov, tile=(16, 16), subfiletype=1,
                 extratags=[
                     # Overview carries its OWN scale (not just 2*0.5) and
                     # tiepoint deliberately shifted by 10 units to make
                     # the inheritance vs own-geo distinction observable.
                     (33550, 12, 3, (1.25, 1.25, 0.0)),
                     (33922, 12, 6,
                      (0.0, 0.0, 0.0, 110.0, 210.0, 0.0)),
                     (34735, 3, 12, geokeys),
                 ])

    da_ov = open_geotiff(path, overview_level=1)
    t = da_ov.attrs['transform']
    # The overview's own values must survive: pixel_width=1.25 (not
    # 2*0.5=1.0 from rescaling), origin_x=110 (not 100 from the parent).
    assert abs(t[0] - 1.25) < 1e-9, (
        f"overview's own pixel_width clobbered: transform={t}")
    assert abs(t[2] - 110.0) < 1e-9, (
        f"overview's own origin_x clobbered: transform={t}")


def test_overview_without_full_res_sibling_falls_back_gracefully(tmp_path):
    """No full-res IFD => return the overview's own (empty) geo info.

    Pathological but well-formed: a TIFF whose only IFD is marked
    reduced-resolution. The helper should not raise; it returns the
    overview's own ``GeoInfo`` (with ``has_georef=False``) so callers
    fall back to integer pixel coords, matching the pre-fix behaviour
    for files that genuinely have no georef anywhere.
    """
    tifffile = pytest.importorskip("tifffile")

    path = str(tmp_path / "overview_no_parent_1640.tif")
    arr = np.zeros((16, 16), dtype=np.float32)
    tifffile.imwrite(path, arr, tile=(16, 16), subfiletype=1)

    da = open_geotiff(path, overview_level=0)
    # No georef anywhere -> default coords, no crs/transform/etc.
    assert da.attrs.get('crs') is None
    # ``transform`` may or may not be emitted (the default unit tuple is
    # still considered "no georef" by the helper); the key contract is
    # that the read succeeds without raising and produces a 2-D array.
    assert da.shape == (16, 16)


def test_overview_level_0_path_unchanged(tmp_path):
    """For overview_level=0, the helper must be a no-op.

    Pin the contract that level-0 reads still get exactly the geo info
    they did before #1640.
    """
    path = str(tmp_path / "overview_lvl0_passthrough_1640.tif")
    src = _make_cog_with_overviews(path)

    da = open_geotiff(path, overview_level=0)
    t = da.attrs['transform']

    # Origin matches the source DataArray's first pixel edge.
    src_y = np.asarray(src.coords['y'])
    src_x = np.asarray(src.coords['x'])
    expected_origin_x = float(src_x[0]) - 0.5 * 0.5  # edge = center - half
    expected_origin_y = float(src_y[0]) - 0.5 * (-0.5)
    assert abs(t[2] - expected_origin_x) < 1e-9
    assert abs(t[5] - expected_origin_y) < 1e-9
    assert abs(t[0] - 0.5) < 1e-9
    assert abs(t[4] - (-0.5)) < 1e-9
