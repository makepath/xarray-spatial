"""Read-side overview tests: IFD selection, georef inheritance, and
``overview_level`` type validation.

Consolidates three top-level files for epic #2424 (cluster 8):

* ``test_overview_filter.py`` (#1504) -- ``select_overview_ifd`` skips
  mask / page IFDs and ``open_geotiff(overview_level=...)`` lands on the
  real pyramid, raising a clear ``ValueError`` out of range.
* ``test_overview_geo_inheritance_1640.py`` (#1640) -- overview reads
  inherit the level-0 georef and rescale the pixel size by the reduction
  factor, across all four backends.
* ``test_overview_level_type_validation_2074.py`` (#2074) and
  ``test_overview_level_validation_backends_2160.py`` (#2160) --
  ``overview_level`` type checks fire up front (and before unrelated
  source / chunk / GPU-policy errors) on ``open_geotiff``,
  ``read_geotiff_dask``, and ``read_geotiff_gpu``.

Tests-only restructure.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrspatial.geotiff import open_geotiff


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

_BACKENDS = [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 128}, id="dask+numpy"),
    pytest.param({"gpu": True}, id="cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
    pytest.param({"gpu": True, "chunks": 128}, id="dask+cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
]


def _materialise(da) -> np.ndarray:
    """Return a numpy view of the data regardless of backend."""
    raw = da.data
    if hasattr(raw, 'compute'):
        raw = raw.compute()
    if hasattr(raw, 'get'):
        raw = raw.get()
    return np.asarray(raw)


# =========================================================================
# Section: overview_level skips mask / page IFDs (issue #1504)
# =========================================================================
#
# GDAL COG variants can interleave NewSubfileType=4 (transparency mask)
# IFDs with the overview pyramid. Indexing the raw IFD list by
# overview_level then returns a 1-bit mask instead of a reduced-resolution
# overview. The reader filters out mask IFDs before resolving
# overview_level, and raises a clear ValueError when the requested level
# is out of range.


def _write_tiff_with_mask(path, full_res, mask, overview):
    """Write a 3-IFD TIFF: full-res, mask (subfiletype=4), overview.

    All IFDs are tiled so that the xrspatial reader exercises its
    tiled-COG path. Tiles are 16x16 to keep the test files small.
    """
    import tifffile

    with tifffile.TiffWriter(str(path)) as tw:
        # IFD 0: full resolution (subfiletype=0 implicit).
        tw.write(full_res, tile=(16, 16), photometric='minisblack')
        # IFD 1: transparency mask (subfiletype=4).
        tw.write(
            mask,
            tile=(16, 16),
            photometric='minisblack',
            subfiletype=4,
        )
        # IFD 2: reduced-resolution overview (subfiletype=1).
        tw.write(
            overview,
            tile=(16, 16),
            photometric='minisblack',
            subfiletype=1,
        )


def _write_normal_cog(path, full_res, overviews):
    """Write a typical COG: full-res then a chain of overviews (subfiletype=1)."""
    import tifffile

    with tifffile.TiffWriter(str(path)) as tw:
        tw.write(full_res, tile=(16, 16), photometric='minisblack')
        for ov in overviews:
            tw.write(
                ov,
                tile=(16, 16),
                photometric='minisblack',
                subfiletype=1,
            )


# ---------------------------------------------------------------------------
# select_overview_ifd unit tests (operate on parsed IFD lists directly)
# ---------------------------------------------------------------------------

class TestSelectOverviewIFD:
    def _ifds_for(self, path):
        from xrspatial.geotiff._header import parse_all_ifds, parse_header

        with open(path, 'rb') as f:
            data = f.read()
        return parse_all_ifds(data, parse_header(data))

    def test_skips_mask_ifd(self, tmp_path):
        from xrspatial.geotiff._header import select_overview_ifd

        path = tmp_path / 'with_mask.tif'
        full = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))
        mask = np.zeros((64, 64), dtype=bool)
        ov = (np.arange(32 * 32, dtype=np.uint16).reshape(32, 32))
        _write_tiff_with_mask(path, full, mask, ov)

        ifds = self._ifds_for(path)
        assert len(ifds) == 3
        # Sanity: middle IFD really is the mask.
        assert ifds[1].subfile_type & 4 == 4
        assert ifds[1].is_mask

        # Level 0 is full-res (NOT the mask).
        sel0 = select_overview_ifd(ifds, 0)
        assert sel0.width == 64 and sel0.height == 64
        assert not sel0.is_mask

        # Level 1 is the overview, jumping over the mask IFD.
        sel1 = select_overview_ifd(ifds, 1)
        assert sel1.width == 32 and sel1.height == 32
        assert not sel1.is_mask

    def test_none_returns_full_res(self, tmp_path):
        from xrspatial.geotiff._header import select_overview_ifd

        path = tmp_path / 'plain.tif'
        full = np.zeros((32, 32), dtype=np.uint8)
        _write_normal_cog(path, full, [])
        ifds = self._ifds_for(path)
        assert select_overview_ifd(ifds, None).width == 32

    def test_out_of_range_raises(self, tmp_path):
        from xrspatial.geotiff._header import select_overview_ifd

        path = tmp_path / 'with_mask.tif'
        full = np.zeros((64, 64), dtype=np.uint16)
        mask = np.zeros((64, 64), dtype=bool)
        ov = np.zeros((32, 32), dtype=np.uint16)
        _write_tiff_with_mask(path, full, mask, ov)
        ifds = self._ifds_for(path)

        with pytest.raises(ValueError) as excinfo:
            select_overview_ifd(ifds, 99)
        msg = str(excinfo.value)
        assert 'overview_level=99' in msg
        # Useful diagnostic: tells the user how many real IFDs there are.
        assert '2 pyramid IFDs' in msg
        assert 'non-pyramid' in msg.lower() or 'mask' in msg.lower()

    def test_negative_raises(self, tmp_path):
        from xrspatial.geotiff._header import select_overview_ifd

        path = tmp_path / 'plain.tif'
        full = np.zeros((32, 32), dtype=np.uint8)
        _write_normal_cog(path, full, [])
        ifds = self._ifds_for(path)
        with pytest.raises(ValueError, match='must be >= 0'):
            select_overview_ifd(ifds, -1)

    def test_skips_page_ifd(self, tmp_path):
        """NewSubfileType bit 1 (multi-page document page) is also filtered.

        Even though geotiff usage rarely sets bit 1, the spec lets it
        coexist with overviews. ``overview_level`` should index the
        pyramid only and ignore page IFDs the same way it ignores masks.
        """
        import tifffile

        from xrspatial.geotiff._header import select_overview_ifd

        path = tmp_path / 'with_page.tif'
        full = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        page = np.zeros((64, 64), dtype=np.uint16)
        ov = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)

        with tifffile.TiffWriter(str(path)) as tw:
            tw.write(full, tile=(16, 16), photometric='minisblack')
            # subfiletype=2 -> bit 1 set, page-of-multi-page-doc.
            tw.write(page, tile=(16, 16), photometric='minisblack',
                     subfiletype=2)
            tw.write(ov, tile=(16, 16), photometric='minisblack',
                     subfiletype=1)

        ifds = self._ifds_for(path)
        assert len(ifds) == 3
        assert ifds[1].subfile_type == 2  # page

        sel0 = select_overview_ifd(ifds, 0)
        assert sel0.width == 64 and sel0.height == 64
        sel1 = select_overview_ifd(ifds, 1)
        # Must skip the page IFD and land on the 32x32 overview.
        assert sel1.width == 32 and sel1.height == 32

    def test_skips_overview_of_mask(self, tmp_path):
        """An overview-of-mask IFD (subfile_type=5: bits 0+2) is excluded.

        The presence of the mask bit dominates -- this is a mask, even if
        it happens to be a reduced-resolution one.
        """
        import tifffile

        from xrspatial.geotiff._header import select_overview_ifd

        path = tmp_path / 'with_overview_mask.tif'
        full = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        ov = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
        ov_mask = np.zeros((32, 32), dtype=bool)

        with tifffile.TiffWriter(str(path)) as tw:
            tw.write(full, tile=(16, 16), photometric='minisblack')
            tw.write(ov, tile=(16, 16), photometric='minisblack',
                     subfiletype=1)
            # subfiletype=5 -> bits 0+2: reduced-resolution mask.
            tw.write(ov_mask, tile=(16, 16), photometric='minisblack',
                     subfiletype=5)

        ifds = self._ifds_for(path)
        assert ifds[2].subfile_type == 5
        assert ifds[2].is_mask  # mask-bit dominates

        sel1 = select_overview_ifd(ifds, 1)
        assert sel1.width == 32  # the real overview, not the masked one
        assert not sel1.is_mask

        # No level 2 -- only 2 pyramid IFDs.
        with pytest.raises(ValueError, match='2 pyramid IFDs'):
            select_overview_ifd(ifds, 2)

    def test_normal_cog_works(self, tmp_path):
        from xrspatial.geotiff._header import select_overview_ifd

        path = tmp_path / 'normal_cog.tif'
        full = np.full((128, 128), 42, dtype=np.uint16)
        ovs = [
            np.full((64, 64), 43, dtype=np.uint16),
            np.full((32, 32), 44, dtype=np.uint16),
            np.full((16, 16), 45, dtype=np.uint16),
        ]
        _write_normal_cog(path, full, ovs)
        ifds = self._ifds_for(path)
        assert len(ifds) == 4

        for level, expected_w in [(0, 128), (1, 64), (2, 32), (3, 16)]:
            sel = select_overview_ifd(ifds, level)
            assert sel.width == expected_w


# ---------------------------------------------------------------------------
# End-to-end: open_geotiff(overview_level=...) on a file with a mask IFD
# ---------------------------------------------------------------------------

class TestOpenGeotiffSkipsMask:
    def test_overview_level_1_returns_overview_not_mask(self, tmp_path):
        path = tmp_path / 'gdal_style_cog.tif'
        # Distinct fill values per IFD so the test cannot be fooled by shape.
        full = np.full((64, 64), 100, dtype=np.uint16)
        mask = np.zeros((64, 64), dtype=bool)
        overview = np.full((32, 32), 200, dtype=np.uint16)
        _write_tiff_with_mask(path, full, mask, overview)

        # Sanity: full-res still works.
        da_full = open_geotiff(str(path), overview_level=0)
        assert da_full.shape == (64, 64)
        assert int(da_full.values[0, 0]) == 100
        assert da_full.dtype == np.uint16

        # The bug: overview_level=1 used to land on the mask IFD.
        da_ov = open_geotiff(str(path), overview_level=1)
        assert da_ov.shape == (32, 32), (
            'overview_level=1 returned wrong shape; likely picked the mask IFD')
        assert int(da_ov.values[0, 0]) == 200
        assert da_ov.dtype == np.uint16

    def test_out_of_range_raises_value_error(self, tmp_path):
        path = tmp_path / 'gdal_style_cog.tif'
        full = np.zeros((64, 64), dtype=np.uint16)
        mask = np.zeros((64, 64), dtype=bool)
        overview = np.zeros((32, 32), dtype=np.uint16)
        _write_tiff_with_mask(path, full, mask, overview)

        with pytest.raises(ValueError) as excinfo:
            open_geotiff(str(path), overview_level=99)
        msg = str(excinfo.value)
        assert 'overview_level=99' in msg
        assert '2 pyramid IFDs' in msg

    def test_normal_cog_unchanged(self, tmp_path):
        path = tmp_path / 'normal_cog.tif'
        full = np.full((128, 128), 1, dtype=np.uint16)
        ovs = [
            np.full((64, 64), 2, dtype=np.uint16),
            np.full((32, 32), 3, dtype=np.uint16),
            np.full((16, 16), 4, dtype=np.uint16),
        ]
        _write_normal_cog(path, full, ovs)

        for level, expected_shape, expected_val in [
            (0, (128, 128), 1),
            (1, (64, 64), 2),
            (2, (32, 32), 3),
            (3, (16, 16), 4),
        ]:
            da = open_geotiff(str(path), overview_level=level)
            assert da.shape == expected_shape
            assert int(da.values[0, 0]) == expected_val


# =========================================================================
# Section: overview reads inherit level-0 georef (issue #1640)
# =========================================================================
#
# Overview IFDs in COGs typically carry no GeoKeys, ModelPixelScale, or
# ModelTiepoint -- the writer puts those tags only on the level-0 IFD.
# Before the fix, ``open_geotiff(path, overview_level=N)`` for ``N >= 1``
# returned a DataArray whose ``transform`` attr was the default unit
# transform and whose ``crs`` attr was absent. The fix inherits the
# georef from the level-0 IFD and rescales the pixel size by the
# overview's reduction factor.


def _make_cog_with_overviews(path: str):
    """Write a 1024x1024 COG with three overview levels and known geo.

    Origin (100, 200), 0.5 unit/pixel, EPSG:4326. Returns the source
    DataArray so the caller can compare extents.
    """
    import xarray as xr

    from xrspatial.geotiff import to_geotiff

    arr = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    y = np.arange(1024, dtype=np.float64) * (-0.5) + 200.0
    x = np.arange(1024, dtype=np.float64) * 0.5 + 100.0
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326})
    to_geotiff(da, path, cog=True, overview_levels=[2, 4, 8])
    return da


@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
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


@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
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


@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
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


# =========================================================================
# Section: overview_level type validation on open_geotiff (issue #2074)
# =========================================================================
#
# The selector in ``_header.select_overview_ifd`` compares
# ``overview_level`` numerically and indexes a list with it. Without an
# upfront type check, ``True`` is silently coerced to ``1`` (because
# ``bool`` is a subclass of ``int``), so a caller passing a bool by
# mistake gets back the first overview level instead of an error. Non-int
# types like ``str`` and ``float`` leak raw ``TypeError`` messages from
# the internal comparison or indexing.


def _write_cog_one_overview_2074(path: str) -> np.ndarray:
    """Write a 64x64 single-band TIFF with one half-resolution overview."""
    import tifffile

    rng = np.random.RandomState(0x2074)
    arr = rng.randint(0, 256, size=(64, 64), dtype=np.uint8)
    half = arr[::2, ::2]
    with tifffile.TiffWriter(path) as tw:
        tw.write(arr, tile=(32, 32), photometric="minisblack")
        tw.write(half, tile=(32, 32), photometric="minisblack",
                 subfiletype=1)
    return arr


@pytest.fixture
def cog_with_overview_2074(tmp_path):
    pytest.importorskip("tifffile")
    path = str(tmp_path / "cog_overview_level_type_2074.tif")
    arr = _write_cog_one_overview_2074(path)
    return path, arr


@pytest.mark.parametrize("value", [True, False])
def test_overview_level_bool_raises_typeerror(cog_with_overview_2074, value):
    path, _ = cog_with_overview_2074
    with pytest.raises(TypeError, match="bool"):
        open_geotiff(path, overview_level=value)


def test_overview_level_str_raises_typeerror(cog_with_overview_2074):
    path, _ = cog_with_overview_2074
    with pytest.raises(TypeError, match="str"):
        open_geotiff(path, overview_level="0")


def test_overview_level_float_raises_typeerror(cog_with_overview_2074):
    path, _ = cog_with_overview_2074
    with pytest.raises(TypeError, match="float"):
        open_geotiff(path, overview_level=1.0)


def test_overview_level_zero_succeeds(cog_with_overview_2074):
    path, arr = cog_with_overview_2074
    result = open_geotiff(path, overview_level=0)
    assert result.shape == arr.shape


def test_overview_level_one_succeeds(cog_with_overview_2074):
    path, arr = cog_with_overview_2074
    result = open_geotiff(path, overview_level=1)
    # Half-resolution overview of a 64x64 source.
    assert result.shape == (arr.shape[0] // 2, arr.shape[1] // 2)


def test_overview_level_none_succeeds(cog_with_overview_2074):
    path, arr = cog_with_overview_2074
    result = open_geotiff(path, overview_level=None)
    assert result.shape == arr.shape


@pytest.mark.parametrize("value", [np.int64(0), np.int32(0)])
def test_overview_level_numpy_int_zero_succeeds(cog_with_overview_2074, value):
    """``np.int64`` / ``np.int32`` should be accepted like Python ints."""
    path, arr = cog_with_overview_2074
    result = open_geotiff(path, overview_level=value)
    assert result.shape == arr.shape


@pytest.mark.parametrize("value", [np.int64(1), np.int32(1)])
def test_overview_level_numpy_int_one_succeeds(cog_with_overview_2074, value):
    """``np.int64`` / ``np.int32`` reach the overview level just like int."""
    path, arr = cog_with_overview_2074
    result = open_geotiff(path, overview_level=value)
    assert result.shape == (arr.shape[0] // 2, arr.shape[1] // 2)


def test_overview_level_typeerror_names_value(cog_with_overview_2074):
    """Error message should name the offending value, not just the type."""
    path, _ = cog_with_overview_2074
    with pytest.raises(TypeError) as exc_info:
        open_geotiff(path, overview_level="not-an-int")
    msg = str(exc_info.value)
    assert "str" in msg
    assert "not-an-int" in msg


# =========================================================================
# Section: overview_level type validation on direct backends (issue #2160)
# =========================================================================
#
# Issue #2074 added the up-front guard to ``open_geotiff``. The direct
# backends (``read_geotiff_dask``, ``read_geotiff_gpu``) reach the same
# selector but only after source coercion, chunk validation, and (on the
# GPU path) ``on_gpu_failure`` resolution. This section mirrors the #2074
# tests against the two direct backends and asserts ordering: the
# ``overview_level`` type check fires before ``_coerce_path``,
# ``_validate_chunks_arg``, or the ``on_gpu_failure`` alias handling.


def _write_cog_one_overview_2160(path: str) -> np.ndarray:
    """Write a 64x64 single-band TIFF with one half-resolution overview."""
    import tifffile

    rng = np.random.RandomState(0x2160)
    arr = rng.randint(0, 256, size=(64, 64), dtype=np.uint8)
    half = arr[::2, ::2]
    with tifffile.TiffWriter(path) as tw:
        tw.write(arr, tile=(32, 32), photometric="minisblack")
        tw.write(half, tile=(32, 32), photometric="minisblack",
                 subfiletype=1)
    return arr


@pytest.fixture
def cog_with_overview_2160(tmp_path):
    pytest.importorskip("tifffile")
    path = str(tmp_path / "cog_overview_backend_2160.tif")
    arr = _write_cog_one_overview_2160(path)
    return path, arr


# ---------------------------------------------------------------------------
# read_geotiff_dask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_dask_overview_level_bool_raises_typeerror(cog_with_overview_2160, value):
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_dask(path, overview_level=value)


def test_dask_overview_level_str_raises_typeerror(cog_with_overview_2160):
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError, match="str"):
        read_geotiff_dask(path, overview_level="0")


def test_dask_overview_level_float_raises_typeerror(cog_with_overview_2160):
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError, match="float"):
        read_geotiff_dask(path, overview_level=1.0)


def test_dask_overview_level_zero_succeeds(cog_with_overview_2160):
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = cog_with_overview_2160
    result = read_geotiff_dask(path, overview_level=0)
    assert result.shape == arr.shape


def test_dask_overview_level_one_succeeds(cog_with_overview_2160):
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = cog_with_overview_2160
    result = read_geotiff_dask(path, overview_level=1)
    assert result.shape == (arr.shape[0] // 2, arr.shape[1] // 2)


def test_dask_overview_level_none_succeeds(cog_with_overview_2160):
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = cog_with_overview_2160
    result = read_geotiff_dask(path, overview_level=None)
    assert result.shape == arr.shape


@pytest.mark.parametrize("value", [np.int64(0), np.int32(0)])
def test_dask_overview_level_numpy_int_zero_succeeds(cog_with_overview_2160, value):
    """``np.int64`` / ``np.int32`` should be accepted like Python ints."""
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = cog_with_overview_2160
    result = read_geotiff_dask(path, overview_level=value)
    assert result.shape == arr.shape


@pytest.mark.parametrize("value", [np.int64(1), np.int32(1)])
def test_dask_overview_level_numpy_int_one_succeeds(cog_with_overview_2160, value):
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = cog_with_overview_2160
    result = read_geotiff_dask(path, overview_level=value)
    assert result.shape == (arr.shape[0] // 2, arr.shape[1] // 2)


def test_dask_overview_level_typeerror_names_value(cog_with_overview_2160):
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError) as exc_info:
        read_geotiff_dask(path, overview_level="not-an-int")
    msg = str(exc_info.value)
    assert "str" in msg
    assert "not-an-int" in msg


# Ordering checks: the overview_level type error must fire before the
# unrelated source / chunk errors, matching open_geotiff's behaviour.


def test_dask_overview_level_check_runs_before_source_coercion():
    """Bad source + bad overview_level should report overview_level first."""
    from xrspatial.geotiff import read_geotiff_dask

    with pytest.raises(TypeError, match="bool"):
        read_geotiff_dask("/nonexistent/path-2160.tif", overview_level=True)


def test_dask_overview_level_check_runs_before_chunks_validation(
        cog_with_overview_2160):
    """Bad chunks + bad overview_level should report overview_level first."""
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_dask(path, chunks=0, overview_level=True)


# ---------------------------------------------------------------------------
# read_geotiff_gpu
# ---------------------------------------------------------------------------
#
# These tests must not import cupy. The validator runs at the top of
# ``read_geotiff_gpu`` before the cupy import, so the bad-input cases
# raise ``TypeError`` on a CPU-only machine. The "succeeds" cases that
# actually need a GPU stay gated on cupy via ``importorskip``.


@pytest.mark.parametrize("value", [True, False])
def test_gpu_overview_level_bool_raises_typeerror_no_cupy(
        cog_with_overview_2160, value):
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_gpu(path, overview_level=value)


def test_gpu_overview_level_str_raises_typeerror_no_cupy(cog_with_overview_2160):
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError, match="str"):
        read_geotiff_gpu(path, overview_level="0")


def test_gpu_overview_level_float_raises_typeerror_no_cupy(cog_with_overview_2160):
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError, match="float"):
        read_geotiff_gpu(path, overview_level=1.0)


def test_gpu_overview_level_typeerror_names_value_no_cupy(cog_with_overview_2160):
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError) as exc_info:
        read_geotiff_gpu(path, overview_level="not-an-int")
    msg = str(exc_info.value)
    assert "str" in msg
    assert "not-an-int" in msg


def test_gpu_overview_level_check_runs_before_source_coercion():
    """Bad source + bad overview_level should report overview_level first."""
    from xrspatial.geotiff import read_geotiff_gpu

    with pytest.raises(TypeError, match="bool"):
        read_geotiff_gpu("/nonexistent/path-2160.tif", overview_level=True)


def test_gpu_overview_level_check_runs_before_chunks_validation(
        cog_with_overview_2160):
    """Bad chunks + bad overview_level should report overview_level first."""
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_gpu(path, chunks=0, overview_level=True)


def test_gpu_overview_level_check_runs_before_on_gpu_failure_validation(
        cog_with_overview_2160):
    """Bad on_gpu_failure + bad overview_level reports overview_level first.

    The ``on_gpu_failure`` alias handling and the ``not in ('auto',
    'strict')`` check sit before the cupy import. Without the up-front
    overview validator, a caller who passes both a bad
    ``on_gpu_failure`` and a bad ``overview_level`` would get the
    ValueError from the policy check, masking the real type bug.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_gpu(path, on_gpu_failure="bogus", overview_level=True)


def test_gpu_overview_level_check_runs_before_chunked_dispatch(
        cog_with_overview_2160):
    """``chunks=`` routes through ``_read_geotiff_gpu_chunked``; the
    validator at the top of ``read_geotiff_gpu`` must fire before that
    branch, otherwise a bad ``overview_level`` would only surface via
    the inner ``read_geotiff_dask`` call (which now also validates,
    but the contract is that the outer entry point reports it first).
    """
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview_2160
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_gpu(path, chunks=32, overview_level=True)
