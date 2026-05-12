"""Regression tests for issue #1739.

Overview IFDs in COGs typically carry no GDAL_NODATA tag (and no
GDAL_METADATA, XResolution, YResolution, ResolutionUnit, ColorMap,
ImageDescription, or ExtraSamples either) -- the writer puts those
tags only on the level-0 IFD. Before the fix, ``open_geotiff(path,
overview_level=N)`` for ``N >= 1`` returned a DataArray whose
``attrs['nodata']`` was None even though the level-0 IFD declared
a sentinel, and the overview's on-disk pixels still carried that
sentinel (the writer rewrites NaN to sentinel before reducing).
The result was silent numerical corruption: sentinel pixels survived
into stats / threshold / plot pipelines.

The fix wires the per-IFD pass-through tags into
``extract_geo_info_with_overview_inheritance`` so an overview without
its own copy inherits from level 0, mirroring the original georef
inheritance from #1640.

These tests cover all four backends (numpy, dask+numpy, cupy,
dask+cupy) and assert that the nodata sentinel + GDAL metadata + the
resolution tags survive the level 0 -> overview transition, and that
sentinel pixels come back as NaN on every backend.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._writer import write
from xrspatial.geotiff._geotags import GeoTransform


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
    pytest.param({"chunks": 16}, id="dask+numpy"),
    pytest.param({"gpu": True}, id="cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
    pytest.param({"gpu": True, "chunks": 16}, id="dask+cupy",
                 marks=pytest.mark.skipif(
                     not _HAS_GPU, reason="cupy + CUDA required")),
]


def _materialise(da: xr.DataArray) -> np.ndarray:
    """Return a numpy view of the data regardless of backend."""
    raw = da.data
    if hasattr(raw, 'compute'):
        raw = raw.compute()
    if hasattr(raw, 'get'):
        raw = raw.get()
    return np.asarray(raw)


def _make_cog_with_nodata(path: str, sentinel: float = -9999.0) -> None:
    """Write a 64x64 COG with two overview levels + a sentinel column.

    The first 16x16 block is the sentinel; the rest is 100.0. Using
    nearest-neighbour overview resampling preserves exactly 64 sentinel
    pixels in level 1 and 16 in level 2, so the test can assert counts
    without floating-point fudge factors.
    """
    arr = np.full((64, 64), 100.0, dtype=np.float32)
    arr[0:16, 0:16] = sentinel
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(64, dtype=np.float64),
                'x': np.arange(64, dtype=np.float64)},
        attrs={'crs': 4326},
    )
    to_geotiff(da, path, cog=True, tile_size=16,
               overview_levels=[2, 4], nodata=sentinel,
               overview_resampling='nearest')


@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_overview_inherits_nodata_attr(tmp_path, backend_kwargs):
    """attrs['nodata'] is set on every overview level, not just level 0."""
    path = str(tmp_path / "overview_nodata_inherit_1739.tif")
    _make_cog_with_nodata(path)

    for lvl in (0, 1, 2):
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        assert da.attrs.get('nodata') == -9999.0, (
            f"backend={backend_kwargs}, overview_level={lvl}: expected "
            f"nodata=-9999.0, got {da.attrs.get('nodata')!r}"
        )


@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_overview_sentinel_pixels_masked_to_nan(tmp_path, backend_kwargs):
    """Pixels at the sentinel value come back as NaN on every overview level.

    Without the inheritance fix, an overview read returned a DataArray
    with literal -9999.0 pixels and no nodata attr, silently poisoning
    downstream stats. With the fix, the reader inherits the sentinel
    and applies the same NaN-mask substitution it applies at level 0.
    """
    path = str(tmp_path / "overview_nodata_mask_1739.tif")
    _make_cog_with_nodata(path)

    expected_nan_counts = {0: 256, 1: 64, 2: 16}
    for lvl, expected in expected_nan_counts.items():
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        vals = _materialise(da)
        actual_nan = int(np.isnan(vals).sum())
        sentinel_remaining = int((vals == -9999.0).sum())

        assert sentinel_remaining == 0, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"{sentinel_remaining} sentinel pixels survived as raw "
            f"values; expected all masked to NaN"
        )
        assert actual_nan == expected, (
            f"backend={backend_kwargs}, overview_level={lvl}: expected "
            f"{expected} NaN pixels, got {actual_nan}"
        )


@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_overview_nanmean_matches_pre_sentinel_value(tmp_path, backend_kwargs):
    """nanmean on every overview level equals the non-sentinel value.

    With the fix, sentinel pixels are NaN-masked, so np.nanmean returns
    100.0 (the value of every other pixel). Without it, the sentinel
    survives as -9999.0 and the mean is far below 100.0.
    """
    path = str(tmp_path / "overview_nodata_mean_1739.tif")
    _make_cog_with_nodata(path)

    for lvl in (0, 1, 2):
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        vals = _materialise(da)
        assert np.nanmean(vals) == pytest.approx(100.0), (
            f"backend={backend_kwargs}, overview_level={lvl}: nanmean="
            f"{np.nanmean(vals)} (expected 100.0); sentinel pixels "
            "are leaking into the average."
        )


@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_overview_inherits_gdal_metadata(tmp_path, backend_kwargs):
    """attrs['gdal_metadata'] and ['gdal_metadata_xml'] come from level 0.

    The COG writer emits the GDAL_METADATA tag only on the level-0 IFD.
    Without the inheritance fix, overview reads dropped the dict, so a
    user-supplied scaling factor or band-stats payload only appeared at
    level 0 -- silently inconsistent across overview reads.
    """
    arr = np.full((64, 64), 100.0, dtype=np.float32)
    gt = GeoTransform(origin_x=0.0, origin_y=0.0,
                      pixel_width=1.0, pixel_height=-1.0)
    path = str(tmp_path / "overview_gdal_md_inherit_1739.tif")
    write(arr, path, geo_transform=gt, crs_epsg=4326,
          cog=True, tile_size=16, overview_levels=[2, 4],
          gdal_metadata_xml=(
              '<GDALMetadata>\n'
              '  <Item name="MY_SCALING">1.5</Item>\n'
              '</GDALMetadata>\n'),
          compression='none')

    for lvl in (0, 1, 2):
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        assert da.attrs.get('gdal_metadata') is not None, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"gdal_metadata attr is missing")
        assert da.attrs['gdal_metadata'].get('MY_SCALING') == '1.5', (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"expected gdal_metadata['MY_SCALING']='1.5', got "
            f"{da.attrs['gdal_metadata']}"
        )
        assert da.attrs.get('gdal_metadata_xml') is not None, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"gdal_metadata_xml attr is missing")


@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_overview_inherits_resolution_tags(tmp_path, backend_kwargs):
    """XResolution / YResolution / ResolutionUnit propagate to overviews.

    Same pattern as gdal_metadata: the writer puts these tags only on
    the level-0 IFD, so an overview read used to come back with the
    resolution tags missing. The inheritance fix surfaces them on every
    level.
    """
    arr = np.full((64, 64), 100.0, dtype=np.float32)
    gt = GeoTransform(origin_x=0.0, origin_y=0.0,
                      pixel_width=1.0, pixel_height=-1.0)
    path = str(tmp_path / "overview_res_inherit_1739.tif")
    write(arr, path, geo_transform=gt, crs_epsg=4326,
          cog=True, tile_size=16, overview_levels=[2, 4],
          x_resolution=300.0, y_resolution=300.0, resolution_unit=2,
          compression='none')

    for lvl in (0, 1, 2):
        da = open_geotiff(path, overview_level=lvl, **backend_kwargs)
        assert da.attrs.get('x_resolution') == 300.0, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"x_resolution={da.attrs.get('x_resolution')!r}")
        assert da.attrs.get('y_resolution') == 300.0, (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"y_resolution={da.attrs.get('y_resolution')!r}")
        assert da.attrs.get('resolution_unit') == 'inch', (
            f"backend={backend_kwargs}, overview_level={lvl}: "
            f"resolution_unit={da.attrs.get('resolution_unit')!r}")


def test_attrs_keysets_consistent_across_overview_levels(tmp_path):
    """The set of attrs keys is identical at every overview level.

    Strong contract that catches future regressions where one of the
    inherited fields gets dropped without removing the entry from the
    main code path (e.g. a refactor that splits the inheritance block
    into helpers and forgets one field).
    """
    arr = np.full((64, 64), 100.0, dtype=np.float32)
    arr[0:16, 0:16] = -9999.0
    gt = GeoTransform(origin_x=0.0, origin_y=0.0,
                      pixel_width=1.0, pixel_height=-1.0)
    path = str(tmp_path / "overview_attr_keysets_1739.tif")
    write(arr, path, geo_transform=gt, crs_epsg=4326, nodata=-9999.0,
          cog=True, tile_size=16, overview_levels=[2, 4],
          x_resolution=300, y_resolution=300, resolution_unit=2,
          gdal_metadata_xml=(
              '<GDALMetadata>\n  <Item name="K">V</Item>\n'
              '</GDALMetadata>\n'))

    level_keysets = {
        lvl: set(open_geotiff(path, overview_level=lvl).attrs.keys())
        for lvl in (0, 1, 2)
    }
    base = level_keysets[0]
    for lvl in (1, 2):
        diff = base ^ level_keysets[lvl]
        assert not diff, (
            f"overview_level={lvl} attrs keyset differs from level 0:\n"
            f"  level 0: {sorted(base)}\n"
            f"  level {lvl}: {sorted(level_keysets[lvl])}\n"
            f"  symmetric diff: {sorted(diff)}"
        )


def test_overview_with_own_nodata_keeps_own_value(tmp_path):
    """Overview IFDs that re-declare GDAL_NODATA keep their own value.

    The inheritance is per-field and only fills in missing entries, so
    an overview that does carry its own tag (rare but valid -- some
    writers do this) is not stomped on by the parent's value.

    This is exercised by reading the same COG with overview level 0
    twice: once directly and once after manipulating the level-0
    nodata. Since the writer always emits the same nodata on level 0,
    hand-building a TIFF here would be overkill; the simpler contract
    test in test_attrs_keysets_consistent_across_* already pins the
    typical writer path. This test pins the "overview already has its
    own value" branch by simulating it directly against
    ``extract_geo_info_with_overview_inheritance``.
    """
    from xrspatial.geotiff._geotags import (
        extract_geo_info_with_overview_inheritance,
        GeoInfo, GeoTransform as _GT,
    )
    # Drive the helper with fake IFD objects + a fake base IFD to
    # avoid having to hand-pack a TIFF with re-declared overview
    # nodata. The helper only inspects ``ifd.subfile_type`` /
    # ``.width`` / ``.height`` and forwards the rest to
    # ``extract_geo_info``; we monkeypatch that to return controlled
    # GeoInfo instances.
    import xrspatial.geotiff._geotags as _gt_mod

    class _StubIFD:
        def __init__(self, subfile_type, width, height):
            self.subfile_type = subfile_type
            self.width = width
            self.height = height

    base_ifd = _StubIFD(0, 64, 64)
    ov_ifd = _StubIFD(1, 32, 32)  # overview (NewSubfileType bit 0)

    # Overview already has its own nodata (-5555); base has -9999.
    # Test: inheritance leaves the overview's -5555 untouched.
    def fake_extract(ifd, data, byte_order):
        if ifd is ov_ifd:
            gi = GeoInfo()
            gi.nodata = -5555.0
            gi.has_georef = False
            return gi
        if ifd is base_ifd:
            gi = GeoInfo()
            gi.nodata = -9999.0
            gi.transform = _GT(0.0, 0.0, 1.0, -1.0)
            gi.has_georef = True
            gi.crs_epsg = 4326
            return gi
        return GeoInfo()

    orig = _gt_mod.extract_geo_info
    _gt_mod.extract_geo_info = fake_extract
    try:
        out = extract_geo_info_with_overview_inheritance(
            ov_ifd, [base_ifd, ov_ifd], b'', '<')
    finally:
        _gt_mod.extract_geo_info = orig

    assert out.nodata == -5555.0, (
        f"overview's own nodata=-5555 was overwritten by base's -9999; "
        f"got {out.nodata}"
    )
